//! HighsCliqueTable (highs/mip/HighsCliqueTable.cpp): the conflict graph of
//! the binary columns as a set of cliques, with substitutions, fixings,
//! clique extraction from rows, cuts and the objective, clique merging and
//! subsumption, clique partitions and the Bron-Kerbosch clique separation.
//!
//! # The boundary
//!
//! Rust owns the table (`HighsCliqueTable` in C++ holds a handle, see
//! clique_ffi.rs). The domain it fixes columns in, and the parts of the MIP
//! solver it touches (node queue, implications, cut pool), are C++, reached
//! through the callbacks of [`CDom`] and [`CMip`]; column bounds are read
//! through raw pointers each time.
//!
//! A bound change through [`CDom`] fixing a binary column calls back into
//! [`CliqueTable::add_implications`] for the same table (re-entrant).
//! That method only reads the table, so the methods that call into the
//! domain never hold a borrow of the table across the call: they work on a
//! [`Ctx`], which holds the raw pointer and lends the table out between
//! calls (`Ctx::t`), so the borrow checker keeps a `&mut` from living
//! across a callback. add_implications itself reads through a shared
//! borrow, which may be held across its own (re-entrant, also shared)
//! callbacks.
//!
//! The only floating-point contraction clang makes here is
//! `rhs -= val * bound` when extracting cliques from rows.

use crate::mip::cuts::sort::{partition, pdqsort, pdqsort_branchless};
use crate::util::cdouble::CDouble;
use crate::util::fma::ClangFma;
use crate::util::hash::Pod;
use crate::util::hash_table::HighsHashTable;
use crate::util::hash_tree::HighsHashTree;
use crate::util::random::HighsRandom;
use std::collections::BTreeSet;
use std::ffi::c_void;

pub const IINF: i32 = i32::MAX;
const INF: f64 = f64::INFINITY;

// HighsBoundType
pub const LOWER: i32 = 0;
pub const UPPER: i32 = 1;
// HighsDomain::Reason types
pub const REASON_UNKNOWN: i32 = -2;
pub const REASON_CLIQUE_TABLE: i32 = -5;

/// HighsCliqueTable::CliqueVar: a column (31 bits) and a value (top bit),
/// the clang bit-field layout
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub struct CliqueVar(pub u32);

impl CliqueVar {
    #[inline(always)]
    pub fn new(col: i32, val: i32) -> Self {
        CliqueVar((col as u32 & 0x7fff_ffff) | ((val as u32 & 1) << 31))
    }
    #[inline(always)]
    pub fn col(self) -> i32 {
        (self.0 & 0x7fff_ffff) as i32
    }
    #[inline(always)]
    pub fn val(self) -> i32 {
        (self.0 >> 31) as i32
    }
    #[inline(always)]
    pub fn index(self) -> usize {
        2 * self.col() as usize + self.val() as usize
    }
    #[inline(always)]
    pub fn complement(self) -> Self {
        CliqueVar(self.0 ^ 0x8000_0000)
    }
    #[inline(always)]
    pub fn weight(self, sol: &[f64]) -> f64 {
        if self.val() != 0 {
            sol[self.col() as usize]
        } else {
            1.0 - sol[self.col() as usize]
        }
    }
}

impl Pod for CliqueVar {
    const SIZE: usize = 4;
    fn write(&self, out: &mut [u8]) {
        self.0.write(out)
    }
}

/// HighsCliqueTable::Clique
#[derive(Clone, Copy, Default, Debug)]
pub struct Clique {
    pub start: i32,
    pub end: i32,
    pub origin: i32,
    pub num_zero_fixed: i32,
    pub equality: bool,
}

impl Clique {
    fn num_active(&self) -> i32 {
        self.end - self.start - self.num_zero_fixed
    }
    fn len(&self) -> i32 {
        self.end - self.start
    }
}

/// HighsCliqueTable::Substitution
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct Substitution {
    pub substcol: i32,
    pub replace: CliqueVar,
}

/// std::pair<HighsInt, CliqueVar> of the clique extensions
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct CliqueExtension {
    pub row: i32,
    pub var: CliqueVar,
}

/// HighsDomainChange
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DomChg {
    pub boundval: f64,
    pub column: i32,
    pub boundtype: i32,
}

/// A HighsDomain: its column bounds (written by C++ during the callbacks,
/// so read through the pointers each time) and the operations on it
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CDom {
    pub ctx: *mut c_void,
    pub col_lower: *const f64,
    pub col_upper: *const f64,
    /// mipsolver->model_->integrality_ (HighsVarType; 0 = continuous)
    pub integrality: *const u8,
    pub num_col: i32,
    /// numModelNonzeros()
    pub num_nonzero: i32,
    /// feastol()
    pub feastol: f64,
    pub infeasible: unsafe extern "C" fn(*mut c_void) -> bool,
    /// changeBound(boundtype, col, val, Reason{type, index})
    pub change_bound: unsafe extern "C" fn(*mut c_void, i32, i32, f64, i32, i32),
    /// fixCol(col, val)
    pub fix_col: unsafe extern "C" fn(*mut c_void, i32, f64),
    pub propagate: unsafe extern "C" fn(*mut c_void),
    /// the domain change stack (pointer, length)
    pub domchg_stack: unsafe extern "C" fn(*mut c_void, *mut i32) -> *const DomChg,
}

impl CDom {
    #[inline(always)]
    pub fn lower(&self, col: i32) -> f64 {
        debug_assert!(col >= 0 && col < self.num_col);
        // SAFETY: CDom's contract: num_col readable doubles, which C++ may
        // write between reads (no reference is formed)
        unsafe { self.col_lower.add(col as usize).read() }
    }
    #[inline(always)]
    pub fn upper(&self, col: i32) -> f64 {
        debug_assert!(col >= 0 && col < self.num_col);
        // SAFETY: as lower
        unsafe { self.col_upper.add(col as usize).read() }
    }
    #[inline(always)]
    pub fn is_fixed(&self, col: i32) -> bool {
        self.lower(col) == self.upper(col)
    }
    #[inline(always)]
    pub fn is_integral(&self, col: i32) -> bool {
        debug_assert!(col >= 0 && col < self.num_col);
        // SAFETY: as lower
        unsafe { self.integrality.add(col as usize).read() != 0 }
    }
    #[inline(always)]
    pub fn is_binary(&self, col: i32) -> bool {
        self.is_integral(col) && self.lower(col) == 0.0 && self.upper(col) == 1.0
    }
    #[inline(always)]
    pub fn infeasible(&self) -> bool {
        // SAFETY: a callback of the domain's C++ side
        unsafe { (self.infeasible)(self.ctx) }
    }
    #[inline(always)]
    pub fn change_bound(&self, boundtype: i32, col: i32, val: f64, reason_type: i32, reason_index: i32) {
        // SAFETY: as infeasible
        unsafe { (self.change_bound)(self.ctx, boundtype, col, val, reason_type, reason_index) }
    }
    pub fn fix_col(&self, col: i32, val: f64) {
        // SAFETY: as infeasible
        unsafe { (self.fix_col)(self.ctx, col, val) }
    }
    pub fn propagate(&self) {
        // SAFETY: as infeasible
        unsafe { (self.propagate)(self.ctx) }
    }
    pub fn domchg_len(&self) -> usize {
        let mut n = 0;
        // SAFETY: as infeasible
        unsafe { (self.domchg_stack)(self.ctx, &mut n) };
        n as usize
    }
    pub fn domchg(&self, k: usize) -> DomChg {
        let mut n = 0;
        // SAFETY: the callback returns the stack's data and length
        unsafe {
            let p = (self.domchg_stack)(self.ctx, &mut n);
            assert!(k < n as usize);
            p.add(k).read()
        }
    }
}

/// The MIP solver's parts the clique table uses
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CMip {
    pub ctx: *mut c_void,
    pub feastol: f64,
    pub epsilon: f64,
    /// mipdata_->numCliqueEntriesAfterPresolve
    pub num_clique_entries_after_presolve: i32,
    /// mipsolver.numNonzero()
    pub num_nonzero: i32,
    /// the node queue pruning of addClique for a new edge (v1, v2)
    pub prune_edge: unsafe extern "C" fn(*mut c_void, CliqueVar, CliqueVar),
    /// implications.tooManyVarBounds()
    pub too_many_var_bounds: unsafe extern "C" fn(*mut c_void) -> bool,
    /// implications.addVUB/addVLB(col, bincol, coef, constant)
    pub add_vub: unsafe extern "C" fn(*mut c_void, i32, i32, f64, f64),
    pub add_vlb: unsafe extern "C" fn(*mut c_void, i32, i32, f64, f64),
}

impl CMip {
    fn prune_edge(&self, v1: CliqueVar, v2: CliqueVar) {
        // SAFETY: a callback of the C++ side
        unsafe { (self.prune_edge)(self.ctx, v1, v2) }
    }
    fn too_many_var_bounds(&self) -> bool {
        // SAFETY: as prune_edge
        unsafe { (self.too_many_var_bounds)(self.ctx) }
    }
    fn add_vub(&self, col: i32, bincol: i32, coef: f64, constant: f64) {
        // SAFETY: as prune_edge
        unsafe { (self.add_vub)(self.ctx, col, bincol, coef, constant) }
    }
    fn add_vlb(&self, col: i32, bincol: i32, coef: f64, constant: f64) {
        // SAFETY: as prune_edge
        unsafe { (self.add_vlb)(self.ctx, col, bincol, coef, constant) }
    }
}

fn sorted_edge(v1: CliqueVar, v2: CliqueVar) -> (CliqueVar, CliqueVar) {
    if v1.col() > v2.col() {
        (v2, v1)
    } else {
        (v1, v2)
    }
}

/// std::remove_if: keeps the elements failing `remove` in order, returns
/// their number
fn remove_if<T: Copy>(v: &mut [T], mut remove: impl FnMut(&T) -> bool) -> usize {
    let mut k = 0;
    for i in 0..v.len() {
        if !remove(&v[i]) {
            v[k] = v[i];
            k += 1;
        }
    }
    k
}

/// std::make_pair(a1, i1) > std::make_pair(a2, i2)
#[inline]
fn pair_gt(a1: f64, i1: i64, a2: f64, i2: i64) -> bool {
    a2 < a1 || (!(a1 < a2) && i2 < i1)
}

pub struct CliqueTable {
    pub entries: Vec<CliqueVar>,
    inv: Vec<HighsHashTree<i32, i32>>,
    inv2: Vec<HighsHashTree<i32>>,
    size_two: HighsHashTable<(CliqueVar, CliqueVar), i32>,
    freespaces: BTreeSet<(i32, i32)>,
    freeslots: Vec<i32>,
    pub cliques: Vec<Clique>,
    numcliquesvar: Vec<i32>,
    infeasvertexstack: Vec<CliqueVar>,
    pub colsubstituted: Vec<i32>,
    pub substitutions: Vec<Substitution>,
    pub deletedrows: Vec<i32>,
    pub cliqueextensions: Vec<CliqueExtension>,
    iscandidate: Vec<u8>,
    col_deleted: Vec<u8>,
    cliquehits: Vec<u32>,
    cliquehitinds: Vec<i32>,
    pub randgen: HighsRandom,
    pub nfixings: i32,
    pub num_entries: i32,
    pub max_entries: i32,
    pub min_entries_for_parallelism: i32,
    pub in_presolve: bool,
    pub allow_parallel: bool,
    pub num_neighbourhood_queries: i64,
}

/// BronKerboschData
struct BkData<'a> {
    sol: &'a [f64],
    p: Vec<CliqueVar>,
    r: Vec<CliqueVar>,
    z: Vec<CliqueVar>,
    cliques: Vec<Vec<CliqueVar>>,
    nbr_inds: Vec<i32>,
    w_r: f64,
    min_w: f64,
    feastol: f64,
    ncalls: i32,
    maxcalls: i32,
    maxcliques: i32,
    max_nq: i64,
    nq: i64,
}

impl<'a> BkData<'a> {
    fn new(sol: &'a [f64]) -> Self {
        BkData {
            sol,
            p: Vec::new(),
            r: Vec::new(),
            z: Vec::new(),
            cliques: Vec::new(),
            nbr_inds: Vec::new(),
            w_r: 0.0,
            min_w: 1.05,
            feastol: 1e-6,
            ncalls: 0,
            maxcalls: 10000,
            maxcliques: 100,
            max_nq: i64::MAX,
            nq: 0,
        }
    }
    fn stop(&self) -> bool {
        self.maxcalls == self.ncalls || self.cliques.len() as i32 == self.maxcliques || self.nq > self.max_nq
    }
}

impl CliqueTable {
    pub fn new(ncols: i32) -> Self {
        let n = ncols as usize;
        CliqueTable {
            entries: Vec::new(),
            inv: (0..2 * n).map(|_| HighsHashTree::new()).collect(),
            inv2: (0..2 * n).map(|_| HighsHashTree::new()).collect(),
            size_two: HighsHashTable::new(),
            freespaces: BTreeSet::new(),
            freeslots: Vec::new(),
            cliques: Vec::new(),
            numcliquesvar: vec![0; 2 * n],
            infeasvertexstack: Vec::new(),
            colsubstituted: vec![0; n],
            substitutions: Vec::new(),
            deletedrows: Vec::new(),
            cliqueextensions: Vec::new(),
            iscandidate: Vec::new(),
            col_deleted: vec![0; n],
            cliquehits: Vec::new(),
            cliquehitinds: Vec::new(),
            randgen: HighsRandom::new(0),
            nfixings: 0,
            num_entries: 0,
            max_entries: IINF,
            min_entries_for_parallelism: IINF,
            in_presolve: false,
            allow_parallel: true,
            num_neighbourhood_queries: 0,
        }
    }

    pub fn num_cliques_total(&self) -> i32 {
        (self.cliques.len() - self.freeslots.len()) as i32
    }

    #[inline]
    pub fn num_cliques(&self, v: CliqueVar) -> i32 {
        self.numcliquesvar[v.index()]
    }

    pub fn is_full(&self) -> bool {
        self.num_entries >= self.max_entries
    }

    pub fn set_max_entries(&mut self, num_nz: i32) {
        self.max_entries = 2000000 + 10 * num_nz;
    }

    fn unlink(&mut self, pos: i32, cliqueid: i32) {
        let v = self.entries[pos as usize];
        self.numcliquesvar[v.index()] -= 1;
        if self.cliques[cliqueid as usize].len() == 2 {
            self.inv2[v.index()].erase(&cliqueid);
        } else {
            self.inv[v.index()].erase(&cliqueid);
        }
    }

    fn link(&mut self, pos: i32, cliqueid: i32) {
        let v = self.entries[pos as usize];
        debug_assert!(self.col_deleted[v.col() as usize] == 0);
        self.numcliquesvar[v.index()] += 1;
        if self.cliques[cliqueid as usize].len() == 2 {
            self.inv2[v.index()].insert(cliqueid, ());
        } else {
            self.inv[v.index()].insert(cliqueid, pos);
        }
    }

    pub fn find_common_clique_id_q(&self, num_queries: &mut i64, v1: CliqueVar, v2: CliqueVar) -> i32 {
        *num_queries += 1;
        if !self.inv2[v1.index()].is_empty() && !self.inv2[v2.index()].is_empty() {
            if let Some(&id) = self.size_two.find(&sorted_edge(v1, v2)) {
                return id;
            }
        }
        match self.inv[v1.index()].find_common(&self.inv[v2.index()]) {
            Some((&k, _)) => k,
            None => -1,
        }
    }

    pub fn find_common_clique_id(&mut self, v1: CliqueVar, v2: CliqueVar) -> i32 {
        let mut nq = self.num_neighbourhood_queries;
        let r = self.find_common_clique_id_q(&mut nq, v1, v2);
        self.num_neighbourhood_queries = nq;
        r
    }

    pub fn have_common_clique(&mut self, v1: CliqueVar, v2: CliqueVar) -> bool {
        if v1.col() == v2.col() {
            return false;
        }
        self.find_common_clique_id(v1, v2) != -1
    }

    pub fn have_common_clique_q(&self, num_queries: &mut i64, v1: CliqueVar, v2: CliqueVar) -> bool {
        if v1.col() == v2.col() {
            return false;
        }
        self.find_common_clique_id_q(num_queries, v1, v2) != -1
    }

    pub fn resolve_substitution(&self, v: &mut CliqueVar) {
        while self.colsubstituted[v.col() as usize] != 0 {
            let subst = self.substitutions[self.colsubstituted[v.col() as usize] as usize - 1];
            *v = if v.val() == 1 { subst.replace } else { subst.replace.complement() };
        }
    }

    pub fn resolve_substitution_val(&self, col: &mut i32, val: &mut f64, offset: &mut f64) {
        while self.colsubstituted[*col as usize] != 0 {
            let subst = self.substitutions[self.colsubstituted[*col as usize] as usize - 1];
            if subst.replace.val() == 0 {
                *offset += *val;
                *val = -*val;
            }
            *col = subst.replace.col();
        }
    }

    pub fn get_substitution(&self, col: i32) -> Option<&Substitution> {
        match self.colsubstituted[col as usize] {
            0 => None,
            k => Some(&self.substitutions[k as usize - 1]),
        }
    }

    /// cliqueSubsumption: returns (redundant, dominatingOrigin)
    fn clique_subsumption(&mut self, clique: &[CliqueVar], mut remove: impl FnMut(&mut Self, i32)) -> (bool, i32) {
        self.collect_cliques(clique);
        let mut redundant = false;
        let mut dominating_origin = IINF;
        let inds = std::mem::take(&mut self.cliquehitinds);
        for &cliqueid in &inds {
            let id = cliqueid as usize;
            let hits = self.cliquehits[id] as i32;
            self.cliquehits[id] = 0;
            if hits == clique.len() as i32 {
                redundant = true;
                let origin = self.cliques[id].origin;
                if origin != IINF && origin != -1 {
                    dominating_origin = origin;
                }
            } else if self.cliques[id].num_active() == hits {
                if self.cliques[id].equality {
                    let size_two = self.cliques[id].len() == 2;
                    for &v in clique {
                        let has = if size_two {
                            self.inv2[v.index()].contains(&cliqueid)
                        } else {
                            self.inv[v.index()].contains(&cliqueid)
                        };
                        if !has {
                            self.infeasvertexstack.push(v);
                        }
                    }
                } else {
                    remove(self, cliqueid);
                }
            }
        }
        self.cliquehitinds = inds;
        self.cliquehitinds.clear();
        (redundant, dominating_origin)
    }

    fn collect_cliques(&mut self, clique: &[CliqueVar]) {
        if self.cliquehits.len() < self.cliques.len() {
            self.cliquehits.resize(self.cliques.len(), 0);
        }
        let hits = &mut self.cliquehits;
        let inds = &mut self.cliquehitinds;
        for &v in clique {
            self.inv[v.index()].for_each(|&id, _| {
                if hits[id as usize] == 0 {
                    inds.push(id);
                }
                hits[id as usize] += 1;
            });
            self.inv2[v.index()].for_each(|&id, _| {
                if hits[id as usize] == 0 {
                    inds.push(id);
                }
                hits[id as usize] += 1;
            });
        }
    }

    fn bron_kerbosch_recurse(&self, d: &mut BkData, mut plen: usize, x: &[CliqueVar]) {
        let mut w = d.w_r;
        for i in 0..plen {
            w += d.p[i].weight(d.sol);
        }
        if w < d.min_w - d.feastol {
            return;
        }
        if plen == 0 && x.is_empty() {
            let clique = d.r.clone();
            if d.min_w < w - d.feastol {
                d.maxcliques -= d.cliques.len() as i32;
                d.cliques.clear();
                d.min_w = w;
            }
            d.cliques.push(clique);
            return;
        }
        d.ncalls += 1;
        if d.stop() {
            return;
        }

        let mut pivweight = -1.0;
        let mut pivot = CliqueVar::new(0, 0);
        for &v in x {
            if v.weight(d.sol) > pivweight {
                pivweight = v.weight(d.sol);
                pivot = v;
                if pivweight >= 1.0 - d.feastol {
                    break;
                }
            }
        }
        if pivweight < 1.0 - d.feastol {
            for i in 0..plen {
                if d.p[i].weight(d.sol) > pivweight {
                    pivweight = d.p[i].weight(d.sol);
                    pivot = d.p[i];
                    if pivweight >= 1.0 - d.feastol {
                        break;
                    }
                }
            }
        }

        let mut p_minus_nu = Vec::with_capacity(plen);
        self.query_neighbourhood(&mut d.nbr_inds, &mut d.nq, pivot, &d.p[..plen]);
        d.nbr_inds.push(plen as i32);
        let mut k = 0usize;
        for &i in &d.nbr_inds {
            while k < i as usize {
                p_minus_nu.push(d.p[k]);
                k += 1;
            }
            k += 1;
        }
        let sol = d.sol;
        pdqsort(&mut p_minus_nu, |a: &CliqueVar, b: &CliqueVar| {
            pair_gt(a.weight(sol), a.index() as i64, b.weight(sol), b.index() as i64)
        });

        let mut local_x = x.to_vec();
        for v in p_minus_nu {
            let new_plen = self.partition_neighbourhood(&mut d.nbr_inds, &mut d.nq, v, &mut d.p[..plen]);
            let new_xlen = self.partition_neighbourhood(&mut d.nbr_inds, &mut d.nq, v, &mut local_x);

            d.r.push(v);
            let wv = v.weight(d.sol);
            d.w_r += wv;
            self.bron_kerbosch_recurse(d, new_plen, &local_x[..new_xlen]);
            if d.stop() {
                return;
            }
            d.r.pop();
            d.w_r -= wv;

            w -= wv;
            if w < d.min_w {
                return;
            }
            let vpos = (new_plen..plen).find(|&i| d.p[i] == v).expect("v in P");
            plen -= 1;
            d.p.swap(vpos, plen);
            local_x.push(v);
        }
    }

    pub fn do_add_clique(&mut self, cliquevars: &[CliqueVar], equality: bool, origin: i32) {
        let numcliquevars = cliquevars.len() as i32;
        let cliqueid = match self.freeslots.pop() {
            None => {
                self.cliques.push(Clique::default());
                self.cliques.len() as i32 - 1
            }
            Some(id) => id,
        };
        let id = cliqueid as usize;
        self.cliques[id].equality = equality;
        self.cliques[id].origin = origin;

        let max_end;
        match self.freespaces.range((numcliquevars, -1)..).next().copied() {
            None => {
                self.cliques[id].start = self.entries.len() as i32;
                self.cliques[id].end = self.cliques[id].start + numcliquevars;
                max_end = self.cliques[id].end;
                self.entries.resize(max_end as usize, CliqueVar::default());
            }
            Some(fs) => {
                self.freespaces.remove(&fs);
                self.cliques[id].start = fs.1;
                self.cliques[id].end = fs.1 + numcliquevars;
                max_end = fs.1 + fs.0;
            }
        }
        self.cliques[id].num_zero_fixed = 0;

        let start = self.cliques[id].start;
        let mut fixtozero = false;
        let mut k = start;
        for &v0 in cliquevars {
            let mut v = v0;
            self.resolve_substitution(&mut v);
            if fixtozero {
                self.infeasvertexstack.push(v);
                continue;
            }
            let clq_has_v_compl = if numcliquevars == 2 {
                self.inv2[v.complement().index()].contains(&cliqueid)
            } else {
                self.inv[v.complement().index()].contains(&cliqueid)
            };
            if clq_has_v_compl {
                fixtozero = true;
                for j in start..k {
                    if self.entries[j as usize].col() != v.col() {
                        self.infeasvertexstack.push(self.entries[j as usize]);
                    }
                    self.unlink(j, cliqueid);
                }
                k = start;
                continue;
            }
            let inserted = if numcliquevars == 2 {
                self.inv2[v.index()].insert(cliqueid, ())
            } else {
                self.inv[v.index()].insert(cliqueid, k)
            };
            if !inserted {
                self.infeasvertexstack.push(v);
                continue;
            }
            self.entries[k as usize] = v;
            self.numcliquesvar[v.index()] += 1;
            k += 1;
        }

        if max_end > k {
            if self.entries.len() as i32 == max_end {
                self.entries.truncate(k as usize);
            } else {
                self.freespaces.insert((max_end - k, k));
            }
            if self.cliques[id].end > k {
                match k - start {
                    0 => {
                        self.cliques[id].start = -1;
                        self.cliques[id].end = -1;
                        self.freeslots.push(cliqueid);
                        return;
                    }
                    1 => {
                        self.unlink(start, cliqueid);
                        self.cliques[id].start = -1;
                        self.cliques[id].end = -1;
                        self.freeslots.push(cliqueid);
                        return;
                    }
                    2 => {
                        self.unlink(start, cliqueid);
                        self.unlink(start + 1, cliqueid);
                        self.cliques[id].end = k;
                        self.link(start, cliqueid);
                        self.link(start + 1, cliqueid);
                    }
                    _ => self.cliques[id].end = k,
                }
            }
        }

        let c = self.cliques[id];
        self.num_entries += c.len();
        if c.len() == 2 {
            self.size_two.insert(
                sorted_edge(self.entries[c.start as usize], self.entries[c.start as usize + 1]),
                cliqueid,
            );
        }
    }

    /// queryNeighbourhood, serially (the C++ may split the loop over
    /// threads; it sorts the indices, so the result is the same)
    // ponytail: serial; parallelize over q if large clique tables with
    // threads > 1 show in profiles
    pub fn query_neighbourhood(&self, inds: &mut Vec<i32>, num_queries: &mut i64, v: CliqueVar, q: &[CliqueVar]) {
        inds.clear();
        if self.num_cliques(v) == 0 {
            return;
        }
        for (i, &w) in q.iter().enumerate() {
            if self.have_common_clique_q(num_queries, v, w) {
                inds.push(i as i32);
            }
        }
    }

    pub fn partition_neighbourhood(
        &self,
        inds: &mut Vec<i32>,
        num_queries: &mut i64,
        v: CliqueVar,
        q: &mut [CliqueVar],
    ) -> usize {
        self.query_neighbourhood(inds, num_queries, v, q);
        for i in 0..inds.len() {
            q.swap(i, inds[i] as usize);
        }
        inds.len()
    }

    pub fn shrink_to_neighbourhood(
        &self,
        inds: &mut Vec<i32>,
        num_queries: &mut i64,
        v: CliqueVar,
        q: &mut [CliqueVar],
    ) -> usize {
        self.query_neighbourhood(inds, num_queries, v, q);
        for i in 0..inds.len() {
            q[i] = q[inds[i] as usize];
        }
        inds.len()
    }

    /// shrinkToNeighbourhood counting in the table's own query counter
    fn shrink_own(&mut self, inds: &mut Vec<i32>, v: CliqueVar, q: &mut [CliqueVar]) -> usize {
        let mut nq = self.num_neighbourhood_queries;
        let r = self.shrink_to_neighbourhood(inds, &mut nq, v, q);
        self.num_neighbourhood_queries = nq;
        r
    }

    /// partitionNeighbourhood counting in the table's own query counter
    fn partition_own(&mut self, inds: &mut Vec<i32>, v: CliqueVar, q: &mut [CliqueVar]) -> usize {
        let mut nq = self.num_neighbourhood_queries;
        let r = self.partition_neighbourhood(inds, &mut nq, v, q);
        self.num_neighbourhood_queries = nq;
        r
    }

    pub fn remove_clique(&mut self, cliqueid: i32) {
        let id = cliqueid as usize;
        let origin = self.cliques[id].origin;
        if origin != IINF && origin != -1 {
            self.deletedrows.push(origin);
        }
        let start = self.cliques[id].start;
        assert!(start != -1);
        let end = self.cliques[id].end;
        let len = end - start;
        if len == 2 {
            self.size_two
                .erase(&sorted_edge(self.entries[start as usize], self.entries[start as usize + 1]));
        }
        for i in start..end {
            self.unlink(i, cliqueid);
        }
        self.freeslots.push(cliqueid);
        self.freespaces.insert((len, start));
        self.cliques[id].start = -1;
        self.cliques[id].end = -1;
        self.num_entries -= len;
    }

    /// cliquePartition(clqVars, partitionStart)
    pub fn clique_partition(&mut self, clq_vars: &mut [CliqueVar], partition_start: &mut Vec<i32>) {
        self.randgen.shuffle(clq_vars);
        let mut inds = Vec::with_capacity(clq_vars.len());
        let n = clq_vars.len();
        partition_start.clear();
        partition_start.reserve(n);
        let mut extension_end = n;
        partition_start.push(0);
        for i in 0..n {
            if i == extension_end {
                partition_start.push(i as i32);
                extension_end = n;
            }
            let v = clq_vars[i];
            let extension_start = i + 1;
            extension_end = self.partition_own(&mut inds, v, &mut clq_vars[extension_start..extension_end])
                + extension_start;
        }
        partition_start.push(n as i32);
    }

    /// cliquePartition(objective, clqVars, partitionStart)
    pub fn clique_partition_obj(&mut self, objective: &[f64], clq_vars: &mut [CliqueVar], partition_start: &mut Vec<i32>) {
        self.randgen.shuffle(clq_vars);
        let comp = |v1: &CliqueVar, v2: &CliqueVar| {
            (2 * v1.val() - 1) as f64 * objective[v1.col() as usize] > (2 * v2.val() - 1) as f64 * objective[v2.col() as usize]
        };
        pdqsort_branchless(clq_vars, comp);
        let mut inds = Vec::with_capacity(clq_vars.len());
        let n = clq_vars.len();
        partition_start.clear();
        partition_start.reserve(n);
        let mut extension_end = n;
        partition_start.push(0);
        let mut last_swapped = 0usize;
        for i in 0..n {
            if i == extension_end {
                partition_start.push(i as i32);
                extension_end = n;
                if last_swapped >= i {
                    pdqsort_branchless(&mut clq_vars[i..last_swapped + 1], comp);
                }
                last_swapped = 0;
            }
            let v = clq_vars[i];
            let extension_start = i + 1;
            extension_end = self.partition_own(&mut inds, v, &mut clq_vars[extension_start..extension_end])
                + extension_start;
            if let Some(&last) = inds.last() {
                last_swapped = last_swapped.max(last as usize + extension_start);
            }
        }
        partition_start.push(n as i32);
    }

    pub fn get_num_implications(&self, col: i32) -> i32 {
        let i0 = CliqueVar::new(col, 0).index();
        let i1 = CliqueVar::new(col, 1).index();
        let mut numimplics = self.numcliquesvar[i0] + self.numcliquesvar[i1];
        let mut count = |&id: &i32, _: &i32| {
            let c = &self.cliques[id as usize];
            let mut nimplics = c.len() - 1;
            nimplics *= 1 + c.equality as i32;
            numimplics += nimplics - 1;
        };
        self.inv[i0].for_each(&mut count);
        self.inv[i1].for_each(&mut count);
        numimplics
    }

    pub fn get_num_implications_val(&self, col: i32, val: bool) -> i32 {
        let iv = CliqueVar::new(col, val as i32).index();
        let mut numimplics = self.numcliquesvar[iv];
        self.inv[iv].for_each(|&id, _| {
            let c = &self.cliques[id as usize];
            let mut nimplics = c.len() - 1;
            nimplics *= 1 + c.equality as i32;
            numimplics += nimplics - 1;
        });
        numimplics
    }

    /// computeMaximalCliques
    pub fn compute_maximal_cliques(&self, vars: &[CliqueVar], feastol: f64) -> Vec<Vec<CliqueVar>> {
        if vars.is_empty() {
            return Vec::new();
        }
        let maxcolindex = vars.iter().map(|v| v.col() as usize).max().unwrap_or(0);
        let mut sol = vec![0.0; maxcolindex + 1];
        for v in vars {
            sol[v.col() as usize] = v.val() as f64;
        }
        let mut d = BkData::new(&sol);
        d.feastol = feastol;
        for &v in vars {
            if self.colsubstituted[v.col() as usize] != 0 || self.col_deleted[v.col() as usize] != 0 {
                continue;
            }
            if self.num_cliques(v) != 0 {
                d.p.push(v);
            }
        }
        let plen = d.p.len();
        self.bron_kerbosch_recurse(&mut d, plen, &[]);
        d.cliques
    }

    /// addImplications: the fixings implied by fixing col to val, applied
    /// to `dom` (only reads the table; may be re-entered from `dom`)
    pub fn add_implications(&self, dom: &CDom, col: i32, val: i32) {
        let mut v = CliqueVar::new(col, val);
        let reason = 2 * col + val;
        while self.colsubstituted[v.col() as usize] != 0 {
            let subst = self.substitutions[self.colsubstituted[v.col() as usize] as usize - 1];
            v = if v.val() == 1 { subst.replace } else { subst.replace.complement() };
            if v.val() == 1 {
                if dom.lower(v.col()) == 1.0 {
                    continue;
                }
                dom.change_bound(LOWER, v.col(), 1.0, REASON_CLIQUE_TABLE, reason);
                if dom.infeasible() {
                    return;
                }
            } else {
                if dom.upper(v.col()) == 0.0 {
                    continue;
                }
                dom.change_bound(UPPER, v.col(), 0.0, REASON_CLIQUE_TABLE, reason);
                if dom.infeasible() {
                    return;
                }
            }
        }

        let do_fixings = |&cliqueid: &i32| -> Option<()> {
            let c = self.cliques[cliqueid as usize];
            for i in c.start..c.end {
                let e = self.entries[i as usize];
                if e.col() == v.col() {
                    continue;
                }
                if e.val() == 1 {
                    if dom.upper(e.col()) == 0.0 {
                        continue;
                    }
                    dom.change_bound(UPPER, e.col(), 0.0, REASON_CLIQUE_TABLE, reason);
                    if dom.infeasible() {
                        return Some(());
                    }
                } else {
                    if dom.lower(e.col()) == 1.0 {
                        continue;
                    }
                    dom.change_bound(LOWER, e.col(), 1.0, REASON_CLIQUE_TABLE, reason);
                    if dom.infeasible() {
                        return Some(());
                    }
                }
            }
            None
        };
        if self.inv[v.index()].find_map(|id, _| do_fixings(id)).is_some() {
            return;
        }
        self.inv2[v.index()].find_map(|id, _| do_fixings(id));
    }

    /// rebuild: `keep[col]` for the reduced columns says whether the column
    /// is binary and linearly transformable
    pub fn rebuild(&mut self, ncols: i32, orig2reducedcol: &[i32], keep: &[u8]) {
        let mut new = CliqueTable::new(ncols);
        new.in_presolve = self.in_presolve;
        new.min_entries_for_parallelism = self.min_entries_for_parallelism;
        for i in 0..self.cliques.len() {
            let c = self.cliques[i];
            if c.start == -1 {
                continue;
            }
            let oldnumvars = c.len();
            for k in c.start..c.end {
                let col = orig2reducedcol[self.entries[k as usize].col() as usize];
                let e = &mut self.entries[k as usize];
                if col == -1 || keep[col as usize] == 0 {
                    *e = CliqueVar::new(IINF, e.val());
                } else {
                    *e = CliqueVar::new(col, e.val());
                }
            }
            let numvars = remove_if(&mut self.entries[c.start as usize..c.end as usize], |v| v.col() == IINF) as i32;
            if numvars <= 1 {
                continue;
            }
            let origin = if c.origin != IINF { -1 } else { IINF };
            new.do_add_clique(
                &self.entries[c.start as usize..(c.start + numvars) as usize],
                if numvars != oldnumvars { false } else { c.equality },
                origin,
            );
        }
        new.allow_parallel = self.allow_parallel;
        *self = new;
    }

    /// buildFrom: the cliques of `init` on the columns that are binary in
    /// the original model (bounds `orig_lower`, `orig_upper`)
    pub fn build_from(&mut self, orig_lower: &[f64], orig_upper: &[f64], init: &CliqueTable) {
        assert_eq!(init.colsubstituted.len(), self.colsubstituted.len());
        let ncols = init.colsubstituted.len() as i32;
        let mut new = CliqueTable::new(ncols);
        new.in_presolve = self.in_presolve;
        new.min_entries_for_parallelism = self.min_entries_for_parallelism;
        let mut buf: Vec<CliqueVar> = Vec::with_capacity(2 * orig_lower.len());
        for c in &init.cliques {
            if c.start == -1 || c.num_active() <= 1 {
                continue;
            }
            buf.clear();
            buf.extend(
                init.entries[c.start as usize..c.end as usize]
                    .iter()
                    .filter(|v| orig_lower[v.col() as usize] == 0.0 && orig_upper[v.col() as usize] == 1.0),
            );
            if buf.len() <= 1 {
                continue;
            }
            let origin = if c.origin != IINF { -1 } else { IINF };
            new.do_add_clique(&buf, false, origin);
        }
        new.colsubstituted = init.colsubstituted.clone();
        new.substitutions = init.substitutions.clone();
        new.allow_parallel = false;
        *self = new;
    }
}

/// The table behind a raw pointer, for the methods that call into the
/// domain (which may re-enter add_implications): `t()` lends the table out
/// only between such calls
pub struct Ctx {
    t: *mut CliqueTable,
    pub dom: CDom,
}

impl Ctx {
    /// # Safety
    /// `t` a live table, `dom` valid as described for [`CDom`]; while the
    /// Ctx is used, the table is only reached through it or by re-entrant
    /// add_implications
    pub unsafe fn new(t: *mut CliqueTable, dom: CDom) -> Ctx {
        Ctx { t, dom }
    }

    #[inline(always)]
    pub fn t(&mut self) -> &mut CliqueTable {
        // SAFETY: Ctx::new's contract; the borrow of self keeps this the
        // only reference until the next callback
        unsafe { &mut *self.t }
    }

    fn fix_col(&mut self, v: CliqueVar, do_process: bool) -> bool {
        let wasfixed = self.dom.is_fixed(v.col());
        self.dom.fix_col(v.col(), (1 - v.val()) as f64);
        if self.dom.infeasible() {
            return false;
        }
        if !wasfixed {
            let t = self.t();
            t.nfixings += 1;
            t.infeasvertexstack.push(v);
            if do_process {
                self.process_infeasible_vertices();
            }
        }
        true
    }

    fn process_new_edge(&mut self, mut v1: CliqueVar, mut v2: CliqueVar) -> bool {
        if v1.col() == v2.col() {
            if v1.val() == v2.val() {
                self.fix_col(v1, true);
                return false;
            }
            return true;
        }
        if self.t().have_common_clique(v1.complement(), v2) {
            self.fix_col(v2, true);
            false
        } else if self.t().have_common_clique(v2.complement(), v1) {
            self.fix_col(v1, true);
            false
        } else {
            if !self.found_cover(v1.complement(), v2.complement()) {
                return false;
            }
            if self.dom.infeasible() {
                return true;
            }
            self.found_cover(v1, v2);
            if self.dom.is_fixed(v1.col()) || self.dom.is_fixed(v2.col()) || self.dom.infeasible() {
                return true;
            }

            let substitution = if v2.col() < v1.col() {
                if v1.val() == 1 {
                    v2 = v2.complement();
                }
                Substitution { substcol: v1.col(), replace: v2 }
            } else {
                if v2.val() == 1 {
                    v1 = v1.complement();
                }
                Substitution { substcol: v2.col(), replace: v1 }
            };
            let t = self.t();
            t.substitutions.push(substitution);
            t.colsubstituted[substitution.substcol as usize] = t.substitutions.len() as i32;

            let replace = |t: &mut CliqueTable, substituted: CliqueVar, replacement: CliqueVar| {
                let (si, ri) = (substituted.index(), replacement.index());
                t.numcliquesvar[ri] += t.numcliquesvar[si];
                t.numcliquesvar[si] = 0;
                let subst_list = std::mem::take(&mut t.inv[si]);
                subst_list.for_each(|&id, &loc| {
                    t.inv[ri].insert(id, loc);
                    t.entries[loc as usize] = replacement;
                });
                let subst_list2 = std::mem::take(&mut t.inv2[si]);
                subst_list2.for_each(|&id, _| {
                    let mut pos = t.cliques[id as usize].start as usize;
                    let mut other = pos + 1;
                    if t.entries[other] == substituted {
                        std::mem::swap(&mut pos, &mut other);
                    }
                    t.inv2[ri].insert(id, ());
                    t.entries[pos] = replacement;
                    t.size_two.erase(&sorted_edge(substituted, t.entries[other]));
                    t.size_two.insert(sorted_edge(replacement, t.entries[other]), id);
                });
            };
            replace(t, CliqueVar::new(substitution.substcol, 1), substitution.replace);
            replace(t, CliqueVar::new(substitution.substcol, 0), substitution.replace.complement());
            true
        }
    }

    /// fixAllVarsInClique of addClique
    fn fix_all_vars_in_clique(&mut self, mip: &CMip, vars: &[CliqueVar], has_new_edge: &mut bool) -> bool {
        let n = vars.len();
        for i in 0..n {
            if !self.dom.is_fixed(vars[i].col()) || vars[i].val() as f64 != self.dom.lower(vars[i].col()) {
                continue;
            }
            for k in 0..n {
                if k == i {
                    continue;
                }
                if !self.fix_col(vars[k], false) {
                    return false;
                }
            }
            self.process_infeasible_vertices();
            return true;
        }
        if n > 100 {
            return false;
        }
        for i in 0..n.saturating_sub(1) {
            if self.dom.is_fixed(vars[i].col()) {
                continue;
            }
            if self.t().num_cliques(vars[i]) == 0 && self.t().num_cliques(vars[i].complement()) == 0 {
                *has_new_edge = true;
                continue;
            }
            for j in i + 1..n {
                if self.dom.is_fixed(vars[j].col()) {
                    continue;
                }
                if self.t().have_common_clique(vars[i], vars[j]) {
                    continue;
                }
                *has_new_edge = true;
                let iscover = self.process_new_edge(vars[i], vars[j]);
                if self.dom.infeasible() {
                    return false;
                }
                mip.prune_edge(vars[i], vars[j]);
                if iscover {
                    for k in 0..n {
                        if k == i || k == j {
                            continue;
                        }
                        if !self.fix_col(vars[k], false) {
                            return false;
                        }
                    }
                    self.process_infeasible_vertices();
                    return true;
                }
            }
        }
        false
    }

    /// addClique on the global domain; changes `vars` in place as the C++
    pub fn add_clique(&mut self, mip: &CMip, vars: &mut [CliqueVar], equality: bool, origin: i32) {
        let check_clique = |ctx: &mut Ctx, vars: &mut [CliqueVar], hne: &mut bool, complement: bool| {
            if complement {
                vars.iter_mut().for_each(|v| *v = v.complement());
            }
            let done = ctx.fix_all_vars_in_clique(mip, vars, hne);
            if complement {
                vars.iter_mut().for_each(|v| *v = v.complement());
            }
            done
        };
        for v in vars.iter_mut() {
            self.t().resolve_substitution(v);
        }
        let mut has_new_edge = false;
        if check_clique(self, vars, &mut has_new_edge, false) {
            return;
        }
        if self.dom.infeasible() {
            return;
        }
        if vars.len() == 2 && equality {
            if check_clique(self, vars, &mut has_new_edge, true) {
                return;
            }
            if self.dom.infeasible() {
                return;
            }
        }
        if !has_new_edge && origin == IINF {
            return;
        }
        let dom = self.dom;
        let n = remove_if(vars, |v| dom.is_fixed(v.col()));
        if n < 2 {
            return;
        }
        self.t().do_add_clique(&vars[..n], equality, origin);
        self.process_infeasible_vertices();
    }

    pub fn found_cover(&mut self, v1: CliqueVar, v2: CliqueVar) -> bool {
        let mut commonclique = self.t().find_common_clique_id(v1, v2);
        if commonclique == -1 {
            return false;
        }
        while commonclique != -1 {
            let c = self.t().cliques[commonclique as usize];
            for i in c.start..c.end {
                let e = self.t().entries[i as usize];
                if e == v1 || e == v2 {
                    continue;
                }
                if !self.fix_col(e, false) {
                    return true;
                }
            }
            let t = self.t();
            t.remove_clique(commonclique);
            commonclique = t.find_common_clique_id(v1, v2);
        }
        self.process_infeasible_vertices();
        true
    }

    /// processInfeasibleVertices
    pub fn process_infeasible_vertices(&mut self) {
        let dom = self.dom;
        while !self.t().infeasvertexstack.is_empty() && !self.dom.infeasible() {
            let t = self.t();
            let mut v = t.infeasvertexstack.pop().unwrap().complement();
            t.resolve_substitution(&mut v);
            let wasfixed = self.dom.is_fixed(v.col());
            self.dom.fix_col(v.col(), v.val() as f64);
            if self.dom.infeasible() {
                return;
            }
            let t = self.t();
            if !wasfixed {
                t.nfixings += 1;
            }
            if t.col_deleted[v.col() as usize] != 0 {
                continue;
            }
            t.col_deleted[v.col() as usize] = 1;

            let lists = std::mem::take(&mut t.inv[v.index()]);
            let lists2 = std::mem::take(&mut t.inv2[v.index()]);
            let fix_others = |ctx: &mut Ctx, cliqueid: i32| -> Option<()> {
                let c = ctx.t().cliques[cliqueid as usize];
                for i in c.start..c.end {
                    let e = ctx.t().entries[i as usize];
                    if e.col() == v.col() {
                        continue;
                    }
                    if !ctx.fix_col(e, false) {
                        return Some(());
                    }
                }
                ctx.t().remove_clique(cliqueid);
                None
            };
            if lists.find_map(|&id, _| fix_others(self, id)).is_some() {
                return;
            }
            if lists2.find_map(|&id, _| fix_others(self, id)).is_some() {
                return;
            }

            let t = self.t();
            let lists = std::mem::take(&mut t.inv[v.complement().index()]);
            let lists2 = std::mem::take(&mut t.inv2[v.complement().index()]);

            if t.in_presolve {
                lists.for_each(|&id, _| {
                    t.cliques[id as usize].num_zero_fixed += 1;
                    if t.cliques[id as usize].num_active() <= 1 {
                        t.remove_clique(id);
                    }
                });
                continue;
            }

            lists2.for_each(|&id, _| t.remove_clique(id));

            debug_assert!(t.cliquehitinds.is_empty());
            let mut clq: Vec<CliqueVar> = Vec::new();
            lists.for_each(|&id, _| {
                let c = &mut t.cliques[id as usize];
                c.num_zero_fixed += 1;
                if c.num_active() <= 1 {
                    t.remove_clique(id);
                } else if c.num_zero_fixed >= 10.max(c.len() >> 1) {
                    let (start, end) = (c.start as usize, c.end as usize);
                    clq.clear();
                    clq.extend_from_slice(&t.entries[start..end]);
                    t.remove_clique(id);
                    clq.retain(|x| !(dom.is_fixed(x.col()) && dom.lower(x.col()) == (1 - x.val()) as f64));
                    if clq.len() > 1 {
                        t.do_add_clique(&clq, false, IINF);
                    }
                }
            });
        }
        self.propagate_and_cleanup();
    }

    fn propagate_and_cleanup(&mut self) {
        let mut start = self.dom.domchg_len();
        self.dom.propagate();
        let mut end = self.dom.domchg_len();
        while !self.dom.infeasible() && start != end {
            for k in start..end {
                let col = self.dom.domchg(k).column;
                if !self.dom.is_fixed(col) {
                    continue;
                }
                let lb = self.dom.lower(col);
                if lb != 1.0 && lb != 0.0 {
                    continue;
                }
                let fixval = lb as i32;
                if self.t().num_cliques(CliqueVar::new(col, 1 - fixval)) != 0 {
                    self.vertex_infeasible(col, 1 - fixval);
                    if self.dom.infeasible() {
                        return;
                    }
                }
            }
            start = self.dom.domchg_len();
            self.dom.propagate();
            end = self.dom.domchg_len();
        }
    }

    pub fn vertex_infeasible(&mut self, col: i32, val: i32) {
        let wasfixed = self.dom.is_fixed(col);
        self.dom.fix_col(col, (1 - val) as f64);
        if self.dom.infeasible() {
            return;
        }
        let t = self.t();
        if !wasfixed {
            t.nfixings += 1;
        }
        t.infeasvertexstack.push(CliqueVar::new(col, val));
        self.process_infeasible_vertices();
    }

    pub fn cleanup_fixed(&mut self) {
        let numcol = self.dom.num_col;
        let oldnfixings = self.t().nfixings;
        for i in 0..numcol {
            if self.t().col_deleted[i as usize] != 0 || !self.dom.is_fixed(i) {
                continue;
            }
            let lb = self.dom.lower(i);
            if lb != 1.0 && lb != 0.0 {
                continue;
            }
            let fixval = lb as i32;
            self.vertex_infeasible(i, 1 - fixval);
            if self.dom.infeasible() {
                return;
            }
        }
        if self.t().nfixings != oldnfixings {
            self.propagate_and_cleanup();
        }
    }

    /// runCliqueSubsumption(globaldom, clique)
    fn run_clique_subsumption(&mut self, clique: &mut Vec<CliqueVar>) -> i32 {
        let t = self.t();
        clique.retain(|v| t.col_deleted[v.col() as usize] == 0);
        if clique.len() <= 2 {
            return 0;
        }
        let mut nremoved = 0;
        let (redundant, _) = t.clique_subsumption(clique, |t, id| {
            nremoved += 1;
            t.cliques[id as usize].origin = IINF;
            t.remove_clique(id);
        });
        if redundant {
            clique.clear();
        }
        if !t.infeasvertexstack.is_empty() {
            let dom = self.dom;
            clique.retain(|v| !dom.is_fixed(v.col()));
        }
        nremoved
    }

    /// runCliqueMerging(globaldomain, clique, equation)
    pub fn run_clique_merging_clique(&mut self, clique: &mut Vec<CliqueVar>, equation: bool) {
        let dom = self.dom;
        let t = self.t();
        let mut extensionstart = CliqueVar::default();
        let mut numcliques = IINF;
        t.iscandidate.resize(t.inv.len(), 0);
        let mut inds: Vec<i32> = Vec::with_capacity(t.inv.len());

        let initial = clique.len();
        for i in 0..initial {
            if dom.is_fixed(clique[i].col()) {
                continue;
            }
            let n = t.num_cliques(clique[i]);
            if n < numcliques {
                numcliques = n;
                extensionstart = clique[i];
            }
        }
        if numcliques == IINF {
            clique.clear();
            return;
        }
        for i in 0..initial {
            t.iscandidate[clique[i].index()] = 1;
        }
        {
            let (iscandidate, cliques, entries) = (&mut t.iscandidate, &t.cliques, &t.entries);
            let mut add_cands = |id: i32| {
                let c = cliques[id as usize];
                for i in c.start..c.end {
                    let e = entries[i as usize];
                    if iscandidate[e.index()] != 0 || dom.is_fixed(e.col()) {
                        continue;
                    }
                    iscandidate[e.index()] = 1;
                    clique.push(e);
                }
            };
            t.inv[extensionstart.index()].for_each(|&id, _| add_cands(id));
            t.inv2[extensionstart.index()].for_each(|&id, _| add_cands(id));
        }
        for v in clique.iter() {
            t.iscandidate[v.index()] = 0;
        }

        let mut i = 0;
        while i != initial && initial < clique.len() {
            if clique[i] != extensionstart {
                let v = clique[i];
                let new_size = initial + t.shrink_own(&mut inds, v, &mut clique[initial..]);
                clique.truncate(new_size);
            }
            i += 1;
        }

        if initial < clique.len() {
            t.randgen.shuffle(&mut clique[initial..]);
            let mut i = initial;
            while i < clique.len() {
                let extvar = clique[i];
                i += 1;
                let new_size = i + t.shrink_own(&mut inds, extvar, &mut clique[i..]);
                clique.truncate(new_size);
            }
        }

        if equation {
            for i in initial..clique.len() {
                let v = clique[i];
                self.vertex_infeasible(v.col(), v.val());
            }
        } else {
            self.run_clique_subsumption(clique);
            if !clique.is_empty() {
                clique.retain(|v| !(dom.is_fixed(v.col()) && dom.lower(v.col()) as i32 == 1 - v.val()));
            }
        }
        self.process_infeasible_vertices();
    }

    /// runCliqueMerging(globaldomain)
    pub fn run_clique_merging(&mut self) {
        let dom = self.dom;
        let mut extensionvars: Vec<CliqueVar> = Vec::new();
        let t = self.t();
        t.iscandidate.resize(t.inv.len(), 0);
        let mut inds: Vec<i32> = Vec::with_capacity(t.inv.len());

        let numcliqueslots = t.cliques.len();
        let max_new_entries = t.num_entries + dom.num_nonzero;
        let mut have_non_model = false;
        let mut clqvars: Vec<CliqueVar> = Vec::new();
        for k in 0..numcliqueslots {
            let t = self.t();
            let c = t.cliques[k];
            if c.start == -1 {
                continue;
            }
            if !c.equality && c.origin == IINF {
                continue;
            }
            if c.origin == -1 {
                have_non_model = true;
                continue;
            }
            let numclqvars = c.len() as usize;
            if numclqvars == 0 {
                continue;
            }
            clqvars.clear();
            clqvars.extend_from_slice(&t.entries[c.start as usize..c.end as usize]);

            let mut extensionstart = clqvars[0];
            let mut numcliques = t.num_cliques(clqvars[0]);
            for &v in &clqvars[1..] {
                let n = t.num_cliques(v);
                if n < numcliques {
                    numcliques = n;
                    extensionstart = v;
                }
            }
            for v in &clqvars {
                t.iscandidate[v.index()] = 1;
            }
            {
                let (iscandidate, cliques, entries) = (&mut t.iscandidate, &t.cliques, &t.entries);
                let mut add_cands = |id: i32| {
                    let c = cliques[id as usize];
                    for i in c.start..c.end {
                        let e = entries[i as usize];
                        if iscandidate[e.index()] != 0 || dom.is_fixed(e.col()) {
                            continue;
                        }
                        iscandidate[e.index()] = 1;
                        extensionvars.push(e);
                    }
                };
                t.inv[extensionstart.index()].for_each(|&id, _| add_cands(id));
                t.inv2[extensionstart.index()].for_each(|&id, _| add_cands(id));
            }
            for v in &clqvars {
                t.iscandidate[v.index()] = 0;
            }
            for v in &extensionvars {
                t.iscandidate[v.index()] = 0;
            }

            let mut i = 0;
            while i != numclqvars && !extensionvars.is_empty() {
                if clqvars[i] != extensionstart {
                    let new_size = t.shrink_own(&mut inds, clqvars[i], &mut extensionvars);
                    extensionvars.truncate(new_size);
                }
                i += 1;
            }
            if !extensionvars.is_empty() {
                t.randgen.shuffle(&mut extensionvars);
                let mut i = 0;
                while i < extensionvars.len() {
                    let extvar = extensionvars[i];
                    i += 1;
                    let new_size = i + t.shrink_own(&mut inds, extvar, &mut extensionvars[i..]);
                    extensionvars.truncate(new_size);
                }
            }

            if c.equality {
                for j in 0..extensionvars.len() {
                    let v = extensionvars[j];
                    self.vertex_infeasible(v.col(), v.val());
                }
            } else {
                let originrow = c.origin;
                t.cliques[k].origin = IINF;
                let num_extensions = extensionvars.len();
                extensionvars.extend_from_slice(&t.entries[c.start as usize..c.end as usize]);
                let kept = remove_if(&mut extensionvars[num_extensions..], |v| t.col_deleted[v.col() as usize] != 0);
                extensionvars.truncate(num_extensions + kept);
                t.remove_clique(k as i32);

                let (redundant, dominating_origin) =
                    t.clique_subsumption(&extensionvars, |t, id| t.remove_clique(id));
                if !redundant {
                    for &v in &extensionvars[..num_extensions] {
                        t.cliqueextensions.push(CliqueExtension { row: originrow, var: v });
                    }
                    extensionvars.retain(|v| !(dom.is_fixed(v.col()) && dom.lower(v.col()) as i32 == 1 - v.val()));
                    if extensionvars.len() > 1 {
                        t.do_add_clique(&extensionvars, false, originrow);
                    }
                } else if dominating_origin != IINF {
                    t.deletedrows.push(originrow);
                } else {
                    for &v in &extensionvars[..num_extensions] {
                        t.cliqueextensions.push(CliqueExtension { row: originrow, var: v });
                    }
                }
            }
            extensionvars.clear();
            self.process_infeasible_vertices();
            if self.t().num_entries >= max_new_entries {
                break;
            }
        }

        if have_non_model {
            for k in 0..numcliqueslots {
                let t = self.t();
                let c = t.cliques[k];
                if c.start == -1 || c.origin != -1 {
                    continue;
                }
                extensionvars.clear();
                extensionvars.extend_from_slice(&t.entries[c.start as usize..c.end as usize]);
                t.remove_clique(k as i32);
                self.run_clique_merging_clique(&mut extensionvars, false);
                if extensionvars.len() > 1 {
                    self.t().do_add_clique(&extensionvars, false, IINF);
                }
            }
        }
    }

    /// extractCliques(mipsolver, inds, vals, complementation, rhs, ...)
    /// for one row side
    fn extract_cliques_row(
        &mut self,
        mip: &CMip,
        inds: &[i32],
        vals: &[f64],
        complementation: &[i8],
        rhs: f64,
        perm: &mut Vec<i32>,
        clique: &mut Vec<CliqueVar>,
        feastol: f64,
    ) {
        let dom = self.dom;
        perm.clear();
        perm.extend(0..inds.len() as i32);
        let nbin = partition(perm, |&pos| dom.is_binary(inds[pos as usize]));
        let ntotal = perm.len();

        if nbin < ntotal {
            for i in 0..nbin {
                let bincol = inds[perm[i] as usize];
                let impliedub = CDouble::from(rhs) - vals[perm[i] as usize];
                if mip.too_many_var_bounds() {
                    break;
                }
                for j in nbin..ntotal {
                    let col = inds[perm[j] as usize];
                    if dom.is_fixed(col) {
                        continue;
                    }
                    let colub = CDouble::from(dom.upper(col)) - dom.lower(col);
                    let mut implcolub = impliedub / vals[perm[j] as usize];
                    if dom.is_integral(col) {
                        implcolub = CDouble::from((implcolub.to_f64() + mip.feastol).floor());
                    }
                    if implcolub < colub - feastol {
                        let (coef, mut constant);
                        if complementation[perm[i] as usize] == -1 {
                            coef = colub - implcolub;
                            constant = implcolub;
                        } else {
                            coef = implcolub - colub;
                            constant = colub;
                        }
                        if complementation[perm[j] as usize] == -1 {
                            constant -= dom.upper(col);
                            mip.add_vlb(col, bincol, -coef.to_f64(), -constant.to_f64());
                        } else {
                            constant += dom.lower(col);
                            mip.add_vub(col, bincol, coef.to_f64(), constant.to_f64());
                        }
                    }
                }
            }
        }

        if nbin <= 1 {
            return;
        }
        pdqsort(&mut perm[..nbin], |&p1, &p2| pair_gt(vals[p1 as usize], p1 as i64, vals[p2 as usize], p2 as i64));
        let v = |k: usize| vals[perm[k] as usize];
        if v(0) + v(1) <= rhs + feastol {
            return;
        }
        let cv = |pos: usize| {
            CliqueVar::new(inds[pos], if complementation[pos] == -1 { 0 } else { 1 })
        };
        if (v(0) - v(nbin - 1)).abs() <= feastol && rhs < 2.0 * v(nbin - 1) - feastol {
            clique.clear();
            for j in 0..nbin {
                clique.push(cv(perm[j] as usize));
            }
            self.add_clique(mip, &mut clique[..nbin], false, IINF);
            return;
        }
        for k in (1..nbin).rev() {
            let mincliqueval = rhs - v(k) + feastol;
            let cliqueend = perm[..k].partition_point(|&p| vals[p as usize] > mincliqueval);
            if cliqueend == 0 {
                continue;
            }
            clique.clear();
            for j in 0..cliqueend {
                clique.push(cv(perm[j] as usize));
            }
            clique.push(cv(perm[k] as usize));
            if clique.len() >= 2 {
                let n = clique.len();
                self.add_clique(mip, &mut clique[..n], false, IINF);
                if self.dom.infeasible() {
                    return;
                }
            }
            if cliqueend == k {
                return;
            }
        }
    }

    /// extractCliques(mipsolver, transformRows) on the rows of `rows`
    pub fn extract_cliques(&mut self, mip: &CMip, rows: &CRows, transform_rows: bool) {
        // SAFETY: CRows's contract
        let (ar_start, ar_index, ar_value, row_lower, row_upper) = unsafe { rows.slices() };
        let dom = self.dom;
        let mut inds: Vec<i32> = Vec::new();
        let mut vals: Vec<f64> = Vec::new();
        let mut perm: Vec<i32> = Vec::new();
        let mut complementation: Vec<i8> = Vec::new();
        let mut clique: Vec<CliqueVar> = Vec::new();
        let mut entries: HighsHashTable<i32, f64> = HighsHashTable::new();

        for i in 0..rows.num_row as usize {
            let (start, end) = (ar_start[i] as usize, ar_start[i + 1] as usize);
            if row_upper[i] == 1.0 {
                let mut issetppc = true;
                clique.clear();
                for j in start..end {
                    let col = ar_index[j];
                    if dom.upper(col) == 0.0 && dom.lower(col) == 0.0 {
                        continue;
                    }
                    issetppc = dom.is_binary(col) && ar_value[j] == 1.0;
                    if !issetppc {
                        break;
                    }
                    clique.push(CliqueVar::new(col, 1));
                }
                if issetppc {
                    let n = clique.len();
                    self.add_clique(mip, &mut clique[..n], row_lower[i] == 1.0, i as i32);
                    if self.dom.infeasible() {
                        return;
                    }
                    continue;
                }
            }
            if !transform_rows || self.t().is_full() {
                continue;
            }
            let mut offset = 0.0;
            for j in start..end {
                let mut col = ar_index[j];
                let mut val = ar_value[j];
                self.t().resolve_substitution_val(&mut col, &mut val, &mut offset);
                *entries.get_or_insert_default(col) += val;
            }

            for (row_rhs, direction) in [(row_upper[i], 1i32), (row_lower[i], -1i32)] {
                if direction as f64 * row_rhs == INF {
                    continue;
                }
                let mut rhs = direction as f64 * (row_rhs - offset);
                inds.clear();
                vals.clear();
                complementation.clear();
                let mut freevar = false;
                let mut nbin = 0;
                for (&col, &value) in entries.iter() {
                    let val = direction as f64 * value;
                    if val.abs() < mip.epsilon {
                        continue;
                    }
                    if dom.is_binary(col) {
                        nbin += 1;
                    }
                    if val < 0.0 {
                        freevar = dom.upper(col) == INF;
                        if freevar {
                            break;
                        }
                        vals.push(-val);
                        inds.push(col);
                        complementation.push(-1);
                        rhs = (-val).mul_add_c(dom.upper(col), rhs);
                    } else {
                        freevar = dom.lower(col) == -INF;
                        if freevar {
                            break;
                        }
                        vals.push(val);
                        inds.push(col);
                        complementation.push(1);
                        rhs = (-val).mul_add_c(dom.lower(col), rhs);
                    }
                }
                if !freevar && nbin != 0 {
                    // (the C++ returns from its checkRow lambda here when
                    // infeasible: no effect)
                    self.extract_cliques_row(mip, &inds, &vals, &complementation, rhs, &mut perm, &mut clique, mip.feastol);
                }
            }
            entries.clear();
        }
    }

    /// extractCliquesFromCut
    pub fn extract_cliques_from_cut(&mut self, mip: &CMip, inds: &[i32], vals: &[f64], rhs: f64) {
        if self.t().is_full() {
            return;
        }
        let dom = self.dom;
        let feastol = mip.feastol;
        let len = inds.len();

        let mut minact = CDouble::from(0.0);
        let mut nbin = 0;
        for i in 0..len {
            if dom.is_binary(inds[i]) {
                nbin += 1;
            }
            if vals[i] > 0.0 {
                if dom.lower(inds[i]) == -INF {
                    return;
                }
                minact += vals[i] * dom.lower(inds[i]);
            } else {
                if dom.upper(inds[i]) == INF {
                    return;
                }
                minact += vals[i] * dom.upper(inds[i]);
            }
        }
        if rhs - minact < 0.0 {
            minact = CDouble::from(rhs);
        }

        for i in 0..len {
            if !dom.is_integral(inds[i]) {
                continue;
            }
            let mut bound_val = ((rhs - minact) / vals[i]).to_f64();
            if vals[i] > 0.0 {
                bound_val = (bound_val + dom.lower(inds[i]) + dom.feastol).floor();
                dom.change_bound(UPPER, inds[i], bound_val, REASON_UNKNOWN, 0);
            } else {
                bound_val = (bound_val + dom.upper(inds[i]) - dom.feastol).ceil();
                dom.change_bound(LOWER, inds[i], bound_val, REASON_UNKNOWN, 0);
            }
            if dom.infeasible() {
                return;
            }
        }
        if nbin <= 1 {
            return;
        }

        let mut perm: Vec<i32> = (0..len as i32).collect();
        let nbin = partition(&mut perm, |&pos| dom.is_binary(inds[pos as usize]));

        if nbin < len {
            for i in 0..nbin {
                let bincol = inds[perm[i] as usize];
                let implied_activity = rhs - minact - vals[perm[i] as usize].abs();
                for j in nbin..len {
                    let pj = perm[j] as usize;
                    let col = inds[pj];
                    if dom.is_fixed(col) {
                        continue;
                    }
                    if vals[pj] > 0.0 {
                        let mut implcolub = (implied_activity + vals[pj] * dom.lower(col)).to_f64() / vals[pj];
                        if dom.is_integral(col) {
                            implcolub = (implcolub + mip.feastol).floor();
                        }
                        if implcolub < dom.upper(col) - feastol {
                            let (coef, constant);
                            if vals[perm[i] as usize] < 0.0 {
                                coef = dom.upper(col) - implcolub;
                                constant = implcolub;
                            } else {
                                if dom.upper(col) == INF {
                                    continue;
                                }
                                coef = implcolub - dom.upper(col);
                                constant = dom.upper(col);
                            }
                            mip.add_vub(col, bincol, coef, constant);
                        }
                    } else {
                        let mut implcollb = (implied_activity + vals[pj] * dom.upper(col)).to_f64() / vals[pj];
                        if dom.is_integral(col) {
                            implcollb = (implcollb - mip.feastol).ceil();
                        }
                        if implcollb > dom.lower(col) + feastol {
                            let (coef, constant);
                            if vals[perm[i] as usize] < 0.0 {
                                coef = dom.lower(col) - implcollb;
                                constant = implcollb;
                            } else {
                                if dom.lower(col) == -INF {
                                    continue;
                                }
                                coef = implcollb - dom.lower(col);
                                constant = dom.lower(col);
                            }
                            mip.add_vlb(col, bincol, coef, constant);
                        }
                    }
                }
            }
        }

        if nbin <= 1 {
            return;
        }
        let mut clique: Vec<CliqueVar> = Vec::with_capacity(nbin);
        pdqsort(&mut perm[..nbin], |&p1, &p2| {
            pair_gt(vals[p1 as usize].abs(), p1 as i64, vals[p2 as usize].abs(), p2 as i64)
        });
        let a = |k: usize| vals[perm[k] as usize].abs();
        if a(0) + a(1) <= (rhs - minact + feastol).to_f64() {
            return;
        }
        let max_new_entries = (mip.num_clique_entries_after_presolve + 100000 + 4 * mip.num_nonzero)
            .min(self.t().num_entries + 10 * nbin as i32);
        let cv = |pos: usize| CliqueVar::new(inds[pos], if vals[pos] < 0.0 { 0 } else { 1 });

        let mut k = nbin - 1;
        while k != 0 && self.t().num_entries < max_new_entries {
            let mincliqueval = (rhs - minact - a(k) + feastol).to_f64();
            let cliqueend = perm[..k].partition_point(|&p| vals[p as usize].abs() > mincliqueval);
            if cliqueend != 0 {
                clique.clear();
                for j in 0..cliqueend {
                    clique.push(cv(perm[j] as usize));
                }
                clique.push(cv(perm[k] as usize));
                if clique.len() >= 2 {
                    let n = clique.len();
                    self.add_clique(mip, &mut clique[..n], false, IINF);
                    if self.dom.infeasible() || self.t().num_entries >= max_new_entries {
                        return;
                    }
                }
                if cliqueend == k {
                    return;
                }
            }
            k -= 1;
        }
    }

    /// extractObjCliques after the C++ part: the cutoff constraint
    /// (vals, inds, rhs), its minimal activity and the number of binaries
    /// in the objective
    pub fn extract_obj_cliques(&mut self, mip: &CMip, nbin: usize, vals: &[f64], inds: &[i32], rhs: f64, minact: CDouble) {
        let dom = self.dom;
        let mut perm: Vec<i32> = (0..nbin as i32).collect();
        let nbin = partition(&mut perm, |&pos| vals[pos as usize] != 0.0 && !dom.is_fixed(inds[pos as usize]));
        if nbin <= 1 {
            return;
        }
        let mut clique: Vec<CliqueVar> = Vec::with_capacity(nbin);
        pdqsort(&mut perm[..nbin], |&p1, &p2| {
            pair_gt(vals[p1 as usize].abs(), p1 as i64, vals[p2 as usize].abs(), p2 as i64)
        });
        let feastol = mip.feastol;
        let a = |k: usize| vals[perm[k] as usize].abs();
        if a(0) + a(1) <= (rhs - minact + feastol).to_f64() {
            return;
        }
        let cv = |pos: usize| CliqueVar::new(inds[pos], if vals[pos] < 0.0 { 0 } else { 1 });
        for k in (1..nbin).rev() {
            let mincliqueval = (rhs - minact - a(k) + feastol).to_f64();
            let cliqueend = perm[..k].partition_point(|&p| vals[p as usize].abs() > mincliqueval);
            if cliqueend == 0 {
                continue;
            }
            clique.clear();
            for j in 0..cliqueend {
                clique.push(cv(perm[j] as usize));
            }
            clique.push(cv(perm[k] as usize));
            if clique.len() >= 2 {
                let n = clique.len();
                self.add_clique(mip, &mut clique[..n], false, IINF);
                if self.dom.infeasible() {
                    return;
                }
            }
            if cliqueend == k {
                return;
            }
        }
    }
}

/// The rows of extractCliques: mipdata's row-wise matrix (num_row + 1
/// starts) and the model's row bounds, up to the first row that is not an
/// original row
#[repr(C)]
pub struct CRows {
    pub num_row: i32,
    pub ar_start: *const i32,
    pub ar_index: *const i32,
    pub ar_value: *const f64,
    pub num_nz: i32,
    pub row_lower: *const f64,
    pub row_upper: *const f64,
}

impl CRows {
    /// # Safety
    /// the pointers valid for their lengths
    unsafe fn slices(&self) -> (&[i32], &[i32], &[f64], &[f64], &[f64]) {
        use crate::ffi::sl;
        (
            sl(self.ar_start, self.num_row + 1),
            sl(self.ar_index, self.num_nz),
            sl(self.ar_value, self.num_nz),
            sl(self.row_lower, self.num_row),
            sl(self.row_upper, self.num_row),
        )
    }
}

/// separateCliques: the C++ cut pool
#[repr(C)]
pub struct CSepaCliques {
    pub sol: *const f64,
    pub num_col: i32,
    pub integral_cols: *const i32,
    pub num_integral_cols: i32,
    pub feastol: f64,
    /// 1000000 + 100 * numNonzero + total_lp_iterations * 1000
    pub max_neighbourhood_queries: i64,
    pub ctx: *mut c_void,
    /// cutpool.addCut(mipsolver, inds, vals, len, rhs, true, false, false)
    pub add_cut: unsafe extern "C" fn(*mut c_void, *mut i32, *mut f64, i32, f64),
}

impl Ctx {
    /// separateCliques. `randgen` and `local_nq` may point into the table
    /// (its own generator and query counter)
    ///
    /// # Safety
    /// `s` valid as described; `randgen`, `local_nq` valid. When `randgen`
    /// is not the table's own, other threads may read the table
    /// concurrently, so the table is then only read
    pub unsafe fn separate_cliques(&mut self, s: &CSepaCliques, randgen: *mut HighsRandom, local_nq: *mut i64) {
        use crate::ffi::sl;
        let own_randgen = std::ptr::eq(randgen, std::ptr::addr_of!((*self.t).randgen));
        let sol = sl(s.sol, s.num_col);
        let mut d = BkData::new(sol);
        d.feastol = s.feastol;
        d.max_nq = s.max_neighbourhood_queries;
        let feastol = s.feastol;
        let mut rg = (*randgen).clone();
        let mut runcliquesubsumption = false;
        {
            let t: &CliqueTable = &*self.t;
            if t.num_neighbourhood_queries > d.max_nq {
                return;
            }
            d.max_nq -= t.num_neighbourhood_queries;
            for &i in sl(s.integral_cols, s.num_integral_cols) {
                if t.colsubstituted[i as usize] != 0 || t.col_deleted[i as usize] != 0 {
                    continue;
                }
                for val in 0..2 {
                    let v = CliqueVar::new(i, val);
                    if t.num_cliques(v) != 0 {
                        if v.weight(sol) > feastol {
                            d.p.push(v);
                        } else {
                            d.z.push(v);
                        }
                    }
                }
            }
            let plen = d.p.len();
            t.bron_kerbosch_recurse(&mut d, plen, &[]);

            let mut inds: Vec<i32> = Vec::new();
            let mut vals: Vec<f64> = Vec::new();
            for ci in 0..d.cliques.len() {
                let mut extensionend = d.z.len();
                for j in 0..d.cliques[ci].len() {
                    let v = d.cliques[ci][j];
                    extensionend = t.partition_neighbourhood(&mut d.nbr_inds, &mut d.nq, v, &mut d.z[..extensionend]);
                    if extensionend == 0 {
                        break;
                    }
                }
                if extensionend != 0 {
                    rg.shuffle(&mut d.z[..extensionend]);
                    let mut i = 0;
                    while i < extensionend {
                        let k = i + 1;
                        let zi = d.z[i];
                        extensionend = k + t.partition_neighbourhood(&mut d.nbr_inds, &mut d.nq, zi, &mut d.z[k..extensionend]);
                        i += 1;
                    }
                    let ext = d.z[..extensionend].to_vec();
                    d.cliques[ci].extend_from_slice(&ext);
                }

                let mut rhs = 1.0;
                runcliquesubsumption = t.cliques.len() > 2;
                inds.clear();
                vals.clear();
                for v in &d.cliques[ci] {
                    inds.push(v.col());
                    if v.val() == 0 {
                        vals.push(-1.0);
                        rhs -= 1.0;
                    } else {
                        vals.push(1.0);
                    }
                }
                rhs = (rhs + 0.5_f64).floor();
                (s.add_cut)(s.ctx, inds.as_mut_ptr(), vals.as_mut_ptr(), inds.len() as i32, rhs);
            }
        }
        *randgen = rg;
        *local_nq += d.nq;

        if runcliquesubsumption && own_randgen {
            for mut clique in std::mem::take(&mut d.cliques) {
                let nremoved = self.run_clique_subsumption(&mut clique);
                if clique.is_empty() {
                    continue;
                }
                if nremoved != 0 {
                    self.t().do_add_clique(&clique, false, -1);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_find_remove() {
        let v = CliqueVar::new;
        let mut t = CliqueTable::new(6);
        t.do_add_clique(&[v(0, 1), v(1, 1), v(2, 0)], false, IINF);
        t.do_add_clique(&[v(3, 1), v(4, 1)], true, 7);
        assert_eq!(t.num_cliques_total(), 2);
        assert_eq!(t.num_entries, 5);
        assert!(t.have_common_clique(v(0, 1), v(2, 0)));
        assert!(!t.have_common_clique(v(0, 1), v(2, 1)));
        assert!(t.have_common_clique(v(4, 1), v(3, 1)));
        assert_eq!(t.get_num_implications_val(0, true), 2);
        // a duplicate entry is reported infeasible, not stored
        t.do_add_clique(&[v(5, 1), v(5, 1), v(0, 0)], false, IINF);
        assert_eq!(t.infeasvertexstack, vec![v(5, 1)]);
        let id = t.find_common_clique_id(v(3, 1), v(4, 1));
        t.remove_clique(id);
        assert_eq!(t.deletedrows, vec![7]);
        assert!(!t.have_common_clique(v(3, 1), v(4, 1)));
        // the freed slot is reused
        t.do_add_clique(&[v(3, 0), v(4, 0)], false, IINF);
        assert_eq!(t.find_common_clique_id(v(3, 0), v(4, 0)), id);
        assert_eq!(t.num_cliques_total(), 3);
    }
}
