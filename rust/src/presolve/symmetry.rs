//! HighsSymmetry (highs/presolve/HighsSymmetry.cpp): symmetry detection on
//! the colored column/row graph of a MIP (partition refinement and a
//! search tree with bliss-style certificates), full orbitope detection,
//! orbitopal fixing and orbital fixing on stabilizer orbits.
//!
//! Same steps as the C++, so the same generators, orbitopes and fixings
//! (in the same order) come out. The C++ containers whose iteration order
//! would matter are not iterated: the hash tables are only searched, the
//! refinement queue holds distinct cells (any min-heap pops them in the
//! same order) and std::partition is the same two-ended algorithm in
//! libc++ and libstdc++. Sorts with ties use the pdqsort port.
//!
//! The domain is reached through `SymDom` (a CDom plus what the orbital
//! fixing reads), the clique table directly (it is Rust).

use crate::mip::clique::{CDom, CliqueTable, CliqueVar};
use crate::mip::cuts::sort::{partition, pdqsort, pdqsort_branchless};
use crate::util::disjoint_sets::HighsDisjointSets;
use crate::util::hash::{pair_hash, sparse_combine32};
use crate::util::hash_table::HighsHashTable;
use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::ffi::c_void;

const K_LOWER: i32 = 0;
const K_UPPER: i32 = 1;
/// HighsDomain::Reason::unspecified()
const REASON_UNKNOWN: i32 = -2;

/// A HighsDomain for the orbital fixing: the CDom, the model bounds of
/// isGlobalBinary, the branching positions (fixed during a call) and
/// markInfeasible.
#[repr(C)]
pub struct SymDom {
    pub dom: CDom,
    pub model_lower: *const f64,
    pub model_upper: *const f64,
    pub branch_pos: *const i32,
    pub num_branch_pos: i32,
    pub mark_infeasible: unsafe extern "C" fn(*mut c_void),
}

impl SymDom {
    fn branch_pos(&self) -> &[i32] {
        // SAFETY: SymDom's contract: num_branch_pos readable ints, unchanged
        // during the call
        unsafe { crate::ffi::sl(self.branch_pos, self.num_branch_pos) }
    }
    fn is_global_binary(&self, col: i32) -> bool {
        // SAFETY: the model's num_col bounds
        self.dom.is_integral(col)
            && unsafe { *self.model_lower.add(col as usize) } == 0.0
            && unsafe { *self.model_upper.add(col as usize) } == 1.0
    }
    fn mark_infeasible(&self) {
        // SAFETY: a callback of the domain's C++ side
        unsafe { (self.mark_infeasible)(self.dom.ctx) }
    }
    fn change_bound(&self, boundtype: i32, col: i32, val: f64) {
        self.dom.change_bound(boundtype, col, val, REASON_UNKNOWN, 0);
    }
}

/// std::map<double, u32> of HighsMatrixColoring (no NaN)
#[derive(Clone, Copy, PartialEq, PartialOrd)]
struct Key(f64);
impl Eq for Key {}
impl Ord for Key {
    fn cmp(&self, o: &Self) -> Ordering {
        self.partial_cmp(o).unwrap()
    }
}

/// HighsMatrixColoring: one color per distinct value (within tolerance)
struct Coloring {
    map: BTreeMap<Key, u32>,
    tolerance: f64,
}

impl Coloring {
    fn new(tolerance: f64) -> Self {
        let map = [(0.0, 1), (1.0, 2), (f64::NEG_INFINITY, 3), (f64::INFINITY, 4)]
            .into_iter()
            .map(|(k, v)| (Key(k), v))
            .collect();
        Coloring { map, tolerance }
    }
    fn color(&mut self, value: f64) -> u32 {
        if let Some((k, &c)) = self.map.range(Key(value - self.tolerance)..).next() {
            if k.0 <= value + self.tolerance {
                return c;
            }
        }
        let c = self.map.len() as u32 + 1;
        self.map.insert(Key(value), c);
        c
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(i8)]
pub enum RowPacking {
    Undetermined = -1,
    NotPacking = 0,
    Packing = 1,
    PackingNegated = 2,
}

/// HighsOrbitopeMatrix
#[derive(Clone)]
pub struct OrbitopeMatrix {
    pub row_length: i32,
    pub num_rows: i32,
    pub num_set_packing_rows: i32,
    pub column_to_row: HighsHashTable<i32, i32>,
    pub row_is_set_packing: Vec<RowPacking>,
    /// column-major, num_rows x row_length
    pub matrix: Vec<i32>,
}

impl OrbitopeMatrix {
    #[inline]
    fn entry(&self, i: i32, j: i32) -> i32 {
        self.matrix[i as usize + j as usize * self.num_rows as usize]
    }

    pub fn determine_orbitope_type(&mut self, cliquetable: &mut CliqueTable) {
        for j in 0..self.row_length {
            for i in 0..self.num_rows {
                self.column_to_row.insert(self.entry(i, j), i);
            }
        }
        self.row_is_set_packing = vec![RowPacking::Undetermined; self.num_rows as usize];
        self.num_set_packing_rows = 0;
        self.find_packing_rows(cliquetable, 1, RowPacking::Packing);
        // rows without set packing structure: try with all columns negated
        for r in self.row_is_set_packing.iter_mut() {
            if *r == RowPacking::NotPacking {
                *r = RowPacking::Undetermined;
            }
        }
        self.find_packing_rows(cliquetable, 0, RowPacking::PackingNegated);
    }

    /// The two passes of determineOrbitopeType: cliques of value `val`
    fn find_packing_rows(&mut self, cliquetable: &mut CliqueTable, val: i32, mark: RowPacking) {
        'outer: for j in 1..self.row_length {
            for j0 in 0..j {
                for i in 0..self.num_rows {
                    if self.row_is_set_packing[i as usize] != RowPacking::Undetermined {
                        continue;
                    }
                    let v0 = CliqueVar::new(self.entry(i, j0), val);
                    let v1 = CliqueVar::new(self.entry(i, j), val);
                    // findCommonClique
                    let clq = if v0 == v1 { -1 } else { cliquetable.find_common_clique_id(v0, v1) };
                    if clq == -1 {
                        self.row_is_set_packing[i as usize] = RowPacking::NotPacking;
                        continue;
                    }
                    let c = cliquetable.cliques[clq as usize];
                    let mut overlap = 0;
                    for v in &cliquetable.entries[c.start as usize..c.end as usize] {
                        if v.val() != val {
                            continue;
                        }
                        if self.column_to_row.find(&v.col()) == Some(&i) {
                            overlap += 1;
                        }
                    }
                    if overlap == self.row_length {
                        self.row_is_set_packing[i as usize] = mark;
                        self.num_set_packing_rows += 1;
                        if self.num_set_packing_rows == self.num_rows {
                            break 'outer;
                        }
                    }
                }
            }
        }
    }

    pub fn get_branching_column(&self, col_lower: &[f64], col_upper: &[f64], col: i32) -> i32 {
        if let Some(&i) = self.column_to_row.find(&col) {
            if self.row_is_set_packing[i as usize] > RowPacking::NotPacking {
                for j in 0..self.row_length {
                    let branch_col = self.entry(i, j);
                    if branch_col == col {
                        break;
                    }
                    if col_lower[branch_col as usize] != col_upper[branch_col as usize] {
                        return branch_col;
                    }
                }
            }
        }
        col
    }

    pub fn orbital_fixing(&self, domain: &SymDom) -> i32 {
        let mut rows = Vec::with_capacity(self.num_rows as usize);
        let mut row_used = vec![false; self.num_rows as usize];
        let mut is_packing = true;
        for &pos in domain.branch_pos() {
            let col = domain.dom.domchg(pos as usize).column;
            if let Some(&i) = self.column_to_row.find(&col) {
                if !row_used[i as usize] {
                    row_used[i as usize] = true;
                    is_packing = is_packing && self.row_is_set_packing[i as usize] > RowPacking::NotPacking;
                    rows.push(i);
                }
            }
        }
        if rows.is_empty() {
            return 0;
        }
        if is_packing {
            self.orbital_fixing_packing(&rows, domain)
        } else {
            self.orbital_fixing_full(&rows, domain)
        }
    }

    fn orbital_fixing_packing(&self, rows: &[i32], domain: &SymDom) -> i32 {
        let d = &domain.dom;
        let n = rows.len();
        let negated = |r: i32| self.row_is_set_packing[r as usize] == RowPacking::PackingNegated;
        // entry not fixed to zero (to one for a negated row)
        let not_zero_fixed = |r: i32, col: i32| {
            if negated(r) {
                d.lower(col) < 0.5
            } else {
                d.upper(col) > 0.5
            }
        };
        let mut first_one_in_row = vec![-1; n];
        for j in 0..self.row_length {
            for i in 0..n {
                if first_one_in_row[i] != -1 {
                    continue;
                }
                let r = rows[i];
                let colrj = self.entry(r, j);
                if self.row_is_set_packing[r as usize] == RowPacking::Packing {
                    if d.lower(colrj) > 0.5 {
                        first_one_in_row[i] = j;
                    }
                } else {
                    debug_assert!(negated(r));
                    if d.upper(colrj) < 0.5 {
                        first_one_in_row[i] = j;
                    }
                }
            }
        }

        // fixes the entries k of column j to zero (one when negated);
        // false when infeasible
        let mut num_fixed = 0;
        let fix_zero = |k: i32, j: i32, num_fixed: &mut i32| {
            let col_kj = self.entry(k, j);
            if negated(k) {
                if d.lower(col_kj) > 0.5 {
                    return true;
                }
                domain.change_bound(K_LOWER, col_kj, 1.0);
            } else {
                if d.upper(col_kj) < 0.5 {
                    return true;
                }
                domain.change_bound(K_UPPER, col_kj, 0.0);
            }
            *num_fixed += 1;
            !d.infeasible()
        };

        let mut j = 0;
        for i in 0..n {
            if first_one_in_row[i] > j {
                domain.mark_infeasible();
                return num_fixed;
            }
            let col_ij = self.entry(rows[i], j);
            let negate_i = negated(rows[i]);
            if not_zero_fixed(rows[i], col_ij) {
                let mut j0 = j;
                for k in i + 1..n {
                    if first_one_in_row[k] > j0 {
                        if negate_i {
                            domain.change_bound(K_UPPER, col_ij, 0.0);
                        } else {
                            domain.change_bound(K_LOWER, col_ij, 1.0);
                        }
                        num_fixed += 1;
                        if d.infeasible() {
                            return num_fixed;
                        }
                        break;
                    }
                    if not_zero_fixed(rows[k], self.entry(rows[k], j0)) {
                        j0 += 1;
                        if j0 == self.row_length {
                            break;
                        }
                    }
                }
                j += 1;
                if j == self.row_length {
                    break;
                }
                for k in 0..=i {
                    debug_assert!(first_one_in_row[k] < j);
                    if !fix_zero(rows[k], j, &mut num_fixed) {
                        return num_fixed;
                    }
                }
            }
        }

        // columns that can be fixed to zero completely
        loop {
            j += 1;
            if j >= self.row_length {
                break;
            }
            for k in 0..n {
                debug_assert!(first_one_in_row[k] < j);
                if !fix_zero(rows[k], j, &mut num_fixed) {
                    return num_fixed;
                }
            }
        }

        if !d.infeasible() && num_fixed != 0 {
            d.propagate();
        }
        num_fixed
    }

    fn orbital_fixing_full(&self, rows: &[i32], domain: &SymDom) -> i32 {
        let d = &domain.dom;
        let n = rows.len();
        let rl = self.row_length as usize;
        let nr = self.num_rows as usize;
        let mut mmin = vec![-1i8; n * rl];
        for j in 0..rl {
            for i in 0..n {
                let colij = self.matrix[rows[i] as usize + j * nr];
                if d.lower(colij) == 1.0 {
                    mmin[i + j * n] = 1;
                } else if d.upper(colij) == 0.0 {
                    mmin[i + j * n] = 0;
                }
            }
        }
        let mut mmax = mmin.clone();
        for k in 0..n {
            if mmin[n * (rl - 1) + k] == -1 {
                mmin[n * (rl - 1) + k] = 0;
            }
            if mmax[k] == -1 {
                mmax[k] = 1;
            }
        }

        let i_fixed = |c0: &[i8], c1: &[i8]| (0..n).find(|&i| c0[i] != -1 && c1[i] != -1 && c0[i] != c1[i]).unwrap_or(n);
        let i_discr = |c0: &[i8], c1: &[i8], i_f: usize| -> i64 {
            let mut i = i_f as i64;
            while i >= 0 {
                if c0[i as usize] != 0 && c1[i as usize] != 1 {
                    return i;
                }
                i -= 1;
            }
            -1
        };
        // free entries take the value of the neighbouring column
        let follow = |c: &mut [i8], other: &[i8], k: usize| {
            let is_free = (c[k] == -1) as i8;
            c[k] += (is_free & other[k]) + is_free;
        };

        for j in (0..rl.saturating_sub(1)).rev() {
            let (left, right) = mmin.split_at_mut((j + 1) * n);
            let c0 = &mut left[j * n..];
            let c1 = &right[..];
            let i_f = i_fixed(c0, c1);
            if i_f == n {
                for k in 0..n {
                    follow(c0, c1, k);
                }
            } else {
                let i_d = i_discr(c0, c1, i_f);
                if i_d == -1 {
                    domain.mark_infeasible();
                    return 0;
                }
                let i_d = i_d as usize;
                for k in 0..i_d {
                    follow(c0, c1, k);
                }
                c0[i_d] = 1;
                for k in i_d + 1..n {
                    c0[k] += (c0[k] == -1) as i8;
                }
            }
        }

        for j in 1..rl {
            let (left, right) = mmax.split_at_mut(j * n);
            let c1 = &left[(j - 1) * n..];
            let c0 = &mut right[..n];
            let i_f = i_fixed(c1, c0);
            if i_f == n {
                for k in 0..n {
                    follow(c0, c1, k);
                }
            } else {
                let i_d = i_discr(c1, c0, i_f);
                if i_d == -1 {
                    domain.mark_infeasible();
                    return 0;
                }
                let i_d = i_d as usize;
                for k in 0..i_d {
                    follow(c0, c1, k);
                }
                c0[i_d] = 0;
                for k in i_d + 1..n {
                    c0[k] += 2 * (c0[k] == -1) as i8;
                }
            }
        }

        let mut num_fixed = 0;
        'cols: for j in 0..rl {
            for i in 0..n {
                let (lo, hi) = (mmin[i + j * n], mmax[i + j * n]);
                if lo != hi {
                    debug_assert!(lo < hi);
                    break;
                }
                let colrj = self.matrix[rows[i] as usize + j * nr];
                if d.is_fixed(colrj) {
                    continue;
                }
                num_fixed += 1;
                if lo == 1 {
                    domain.change_bound(K_LOWER, colrj, 1.0);
                } else {
                    domain.change_bound(K_UPPER, colrj, 0.0);
                }
                if d.infeasible() {
                    break 'cols;
                }
            }
        }
        if !d.infeasible() {
            d.propagate();
        }
        num_fixed
    }
}

/// StabilizerOrbits: the orbits of the stabilizer of the branchings
#[derive(Clone, Default)]
pub struct StabilizerOrbits {
    pub orbit_cols: Vec<i32>,
    pub orbit_starts: Vec<i32>,
    pub stabilized_cols: Vec<i32>,
}

/// HighsSymmetries: the generators (restricted to the columns they move)
/// and the full orbitopes
#[derive(Clone, Default)]
pub struct Symmetries {
    pub permutation_columns: Vec<i32>,
    pub permutations: Vec<i32>,
    pub column_position: Vec<i32>,
    pub orbitopes: Vec<OrbitopeMatrix>,
    pub column_to_orbitope: HighsHashTable<i32, i32>,
    pub num_perms: i32,
    pub num_generators: i32,
}

/// The orbit union-find of computeStabilizerOrbits
struct OrbitSets<'a> {
    column_position: &'a [i32],
    partition: Vec<i32>,
    size: Vec<i32>,
    stack: Vec<i32>,
}

impl OrbitSets<'_> {
    fn get_orbit(&mut self, col: i32) -> i32 {
        let mut i = self.column_position[col as usize];
        if i == -1 {
            return -1;
        }
        let p = &mut self.partition;
        let mut orbit = p[i as usize];
        if orbit != p[orbit as usize] {
            loop {
                self.stack.push(i);
                i = orbit;
                orbit = p[orbit as usize];
                if orbit == p[orbit as usize] {
                    break;
                }
            }
            while let Some(k) = self.stack.pop() {
                p[k as usize] = orbit;
            }
        }
        orbit
    }
    fn merge(&mut self, v1: i32, v2: i32) {
        if v1 == v2 {
            return;
        }
        let o1 = self.get_orbit(v1);
        let o2 = self.get_orbit(v2);
        if o1 == o2 {
            return;
        }
        let (o1, o2) = (o1 as usize, o2 as usize);
        if self.size[o2] < self.size[o1] {
            self.partition[o2] = o1 as i32;
            self.size[o1] += self.size[o2];
        } else {
            self.partition[o1] = o2 as i32;
            self.size[o2] += self.size[o1];
        }
    }
}

impl Symmetries {
    pub fn clear(&mut self) {
        *self = Symmetries::default();
    }

    pub fn get_branching_column(&self, col_lower: &[f64], col_upper: &[f64], col: i32) -> i32 {
        if self.column_to_orbitope.is_empty() {
            return col;
        }
        match self.column_to_orbitope.find(&col) {
            Some(&o) if self.orbitopes[o as usize].num_set_packing_rows != 0 => {
                self.orbitopes[o as usize].get_branching_column(col_lower, col_upper, col)
            }
            _ => col,
        }
    }

    pub fn propagate_orbitopes(&self, domain: &SymDom) -> i32 {
        if self.column_to_orbitope.is_empty() || domain.num_branch_pos == 0 {
            return 0;
        }
        let mut which = BTreeSet::new();
        for &pos in domain.branch_pos() {
            if let Some(&o) = self.column_to_orbitope.find(&domain.dom.domchg(pos as usize).column) {
                which.insert(o);
            }
        }
        let mut num_fixed = 0;
        for o in which {
            num_fixed += self.orbitopes[o as usize].orbital_fixing(domain);
            if domain.dom.infeasible() {
                break;
            }
        }
        num_fixed
    }

    pub fn is_stabilized(&self, orbits: &StabilizerOrbits, col: i32) -> bool {
        self.column_position[col as usize] == -1 || orbits.stabilized_cols.binary_search(&col).is_ok()
    }

    pub fn compute_stabilizer_orbits(&self, localdom: &SymDom) -> StabilizerOrbits {
        let d = &localdom.dom;
        let mut so = StabilizerOrbits::default();
        so.stabilized_cols.reserve(self.permutation_columns.len());
        for &i in localdom.branch_pos() {
            let chg = d.domchg(i as usize);
            let col = chg.column;
            if self.column_position[col as usize] == -1 {
                continue;
            }
            debug_assert!(d.is_integral(col));
            // stabilize on columns branched upwards and on general integers
            if chg.boundtype == K_LOWER || !localdom.is_global_binary(col) {
                so.stabilized_cols.push(self.column_position[col as usize]);
            }
        }

        let perm_length = self.permutation_columns.len();
        let mut ws = OrbitSets {
            column_position: &self.column_position,
            partition: (0..perm_length as i32).collect(),
            size: vec![1; perm_length],
            stack: Vec::new(),
        };
        for p in 0..self.num_perms as usize {
            let perm = &self.permutations[p * perm_length..(p + 1) * perm_length];
            if so.stabilized_cols.iter().any(|&i| self.permutation_columns[i as usize] != perm[i as usize]) {
                continue;
            }
            for j in 0..perm_length {
                ws.merge(self.permutation_columns[j], perm[j]);
            }
        }

        so.stabilized_cols.clear();
        so.orbit_cols.reserve(perm_length);
        for &col in &self.permutation_columns {
            if !d.is_integral(col) {
                continue;
            }
            let orbit = ws.get_orbit(col);
            if ws.size[orbit as usize] == 1 {
                so.stabilized_cols.push(col);
            } else if localdom.is_global_binary(col) {
                so.orbit_cols.push(col);
            }
        }
        pdqsort_branchless(&mut so.stabilized_cols, |a, b| a < b);
        if !so.orbit_cols.is_empty() {
            pdqsort(&mut so.orbit_cols, |&a, &b| ws.get_orbit(a) < ws.get_orbit(b));
            let n = so.orbit_cols.len();
            so.orbit_starts.reserve(n + 1);
            so.orbit_starts.push(0);
            for i in 1..n {
                if ws.get_orbit(so.orbit_cols[i]) != ws.get_orbit(so.orbit_cols[i - 1]) {
                    so.orbit_starts.push(i as i32);
                }
            }
            so.orbit_starts.push(n as i32);
        }
        so
    }

    /// StabilizerOrbits::orbitalFixing
    pub fn orbital_fixing(&self, orbit_cols: &[i32], orbit_starts: &[i32], domain: &SymDom) -> i32 {
        let d = &domain.dom;
        let mut num_fixed = self.propagate_orbitopes(domain);
        if d.infeasible() || orbit_cols.is_empty() {
            return num_fixed;
        }
        let num_orbits = orbit_starts.len() as i32 - 1;
        let mut i = 0;
        while i < num_orbits {
            let orbit = &orbit_cols[orbit_starts[i as usize] as usize..orbit_starts[i as usize + 1] as usize];
            if let Some(&fixcol) = orbit.iter().find(|&&c| d.is_fixed(c)) {
                let old_num_fixed = num_fixed;
                let old_size = d.domchg_len();
                if d.lower(fixcol) == 1.0 {
                    for &c in orbit {
                        if d.lower(c) == 1.0 {
                            continue;
                        }
                        num_fixed += 1;
                        domain.change_bound(K_LOWER, c, 1.0);
                        if d.infeasible() {
                            return num_fixed;
                        }
                    }
                } else {
                    for &c in orbit {
                        if d.upper(c) == 0.0 {
                            continue;
                        }
                        num_fixed += 1;
                        domain.change_bound(K_UPPER, c, 0.0);
                        if d.infeasible() {
                            return num_fixed;
                        }
                    }
                }
                let new_fixed = num_fixed - old_num_fixed;
                if new_fixed != 0 {
                    d.propagate();
                    if d.infeasible() {
                        return num_fixed;
                    }
                    if (d.domchg_len() - old_size) as i32 > new_fixed {
                        i = -1;
                    }
                }
            }
            i += 1;
        }
        num_fixed
    }

    pub fn determine_orbitope_types(&mut self, cliquetable: &mut CliqueTable) {
        for o in &mut self.orbitopes {
            o.determine_orbitope_type(cliquetable);
        }
    }
}

#[derive(Clone, Copy)]
struct Node {
    stack_start: i32,
    certificate_end: i32,
    target_cell: i32,
    last_distinguished: i32,
}

/// The column-wise model the graph is built from
pub struct ModelView<'a> {
    pub num_col: i32,
    pub num_row: i32,
    pub a_start: &'a [i32],
    pub a_index: &'a [i32],
    pub a_value: &'a [f64],
    pub col_cost: &'a [f64],
    pub col_lower: &'a [f64],
    pub col_upper: &'a [f64],
    pub integrality: &'a [u8],
    pub row_lower: &'a [f64],
    pub row_upper: &'a [f64],
}

struct ComponentData {
    components: HighsDisjointSets,
    component_starts: Vec<i32>,
    component_sets: Vec<i32>,
    perm_component_starts: Vec<i32>,
    perm_components: Vec<i32>,
    first_unfixed: Vec<i32>,
    num_unfixed: Vec<i32>,
}

/// HighsSymmetryDetection
#[derive(Default)]
pub struct SymmetryDetection {
    col_lower: Vec<f64>,
    col_upper: Vec<f64>,
    integrality: Vec<u8>,

    gstart: Vec<i32>,
    gend: Vec<i32>,
    gedge: Vec<(i32, u32)>,

    current_partition: Vec<i32>,
    current_partition_links: Vec<i32>,
    vertex_to_cell: Vec<i32>,
    vertex_position: Vec<i32>,
    vertex_ground_set: Vec<i32>,
    orbit_partition: Vec<i32>,
    orbit_size: Vec<i32>,

    cell_creation_stack: Vec<i32>,
    cell_in_refinement_queue: Vec<bool>,
    refinement_queue: BinaryHeap<Reverse<i32>>,
    /// positions in current_partition
    distinguish_cands: Vec<usize>,
    automorphisms: Vec<i32>,

    link_compression_stack: Vec<i32>,

    curr_node_certificate: Vec<u32>,
    first_leave_certificate: Vec<u32>,
    best_leave_certificate: Vec<u32>,
    first_leave_partition: Vec<i32>,
    best_leave_partition: Vec<i32>,

    vertex_hash: HighsHashTable<i32, u32>,
    first_leave_graph: HighsHashTable<(i32, i32, u32)>,
    best_leave_graph: HighsHashTable<(i32, i32, u32)>,

    first_leave_prefix_len: i32,
    best_leave_prefix_len: i32,
    first_path_depth: i32,
    best_path_depth: i32,

    num_automorphisms: i32,
    num_col: i32,
    num_row: i32,
    num_vertices: i32,
    num_active_cols: i32,

    node_stack: Vec<Node>,
}

/// The (u32-hashed) MatrixColumn and MatrixRow keys
type MatrixColumn = [u32; 5];
type MatrixRow = [u32; 3];

impl SymmetryDetection {
    #[inline]
    fn cell_size(&self, cell: i32) -> i32 {
        self.current_partition_links[cell as usize] - cell
    }

    fn remove_fix_points(&mut self) {
        let nv = self.num_vertices as usize;
        self.gend.resize(nv, 0);
        {
            let links = &self.current_partition_links;
            let v2c = &self.vertex_to_cell;
            for i in 0..nv {
                let (s, e) = (self.gstart[i] as usize, self.gstart[i + 1] as usize);
                let k = partition(&mut self.gedge[s..e], |&(v, _)| {
                    let c = v2c[v as usize];
                    links[c as usize] - c > 1
                });
                self.gend[i] = (s + k) as i32;
            }
        }

        let mut unit_cell_index = self.num_vertices;
        let mut out = 0;
        for k in 0..self.current_partition.len() {
            let vertex = self.current_partition[k];
            let cell = self.vertex_to_cell[vertex as usize];
            if self.cell_size(cell) == 1 {
                unit_cell_index -= 1;
                self.vertex_to_cell[vertex as usize] = unit_cell_index;
            } else {
                self.current_partition[out] = vertex;
                out += 1;
            }
        }
        self.current_partition.truncate(out);

        for i in 0..nv {
            for j in self.gend[i] as usize..self.gstart[i + 1] as usize {
                self.gedge[j].0 = self.vertex_to_cell[self.gedge[j].0 as usize];
            }
        }

        if (self.current_partition.len() as i32) < self.num_vertices {
            self.num_vertices = self.current_partition.len() as i32;
            if self.num_vertices == 0 {
                self.num_active_cols = 0;
                return;
            }
            let nv = self.num_vertices as usize;
            self.current_partition_links.resize(nv, 0);
            self.cell_in_refinement_queue = vec![false; nv];
            self.refinement_queue.clear();
            let mut cell_start = 0;
            let mut cell_number = 0;
            for i in 0..nv as i32 {
                let vertex = self.current_partition[i as usize];
                if cell_number != self.vertex_to_cell[vertex as usize] {
                    cell_number = self.vertex_to_cell[vertex as usize];
                    self.current_partition_links[cell_start as usize] = i;
                    cell_start = i;
                }
                self.update_cell_membership(i, cell_start, false);
            }
            self.current_partition_links[cell_start as usize] = self.num_vertices;
            let num_col = self.num_col;
            self.num_active_cols = self.current_partition.partition_point(|&v| v < num_col) as i32;
        } else {
            self.num_active_cols = self.num_col;
        }
    }

    fn initialize_ground_set(&mut self) {
        let nv = self.num_vertices as usize;
        self.vertex_ground_set = self.current_partition.clone();
        pdqsort_branchless(&mut self.vertex_ground_set, |a, b| a < b);
        self.vertex_position = vec![-1; self.vertex_to_cell.len()];
        for i in 0..nv {
            self.vertex_position[self.vertex_ground_set[i] as usize] = i as i32;
        }
        self.orbit_partition = (0..nv as i32).collect();
        self.orbit_size = vec![1; nv];
        self.automorphisms = vec![0; nv * 64];
        self.num_automorphisms = 0;
        self.curr_node_certificate.reserve(nv);
    }

    fn merge_orbits(&mut self, v1: i32, v2: i32) -> bool {
        if v1 == v2 {
            return false;
        }
        let o1 = self.get_orbit(v1);
        let o2 = self.get_orbit(v2);
        if o1 == o2 {
            return false;
        }
        let (keep, child) = if o1 < o2 { (o1, o2) } else { (o2, o1) };
        self.orbit_partition[child as usize] = keep;
        self.orbit_size[keep as usize] += self.orbit_size[child as usize];
        true
    }

    fn get_orbit(&mut self, vertex: i32) -> i32 {
        let mut i = self.vertex_position[vertex as usize];
        let p = &mut self.orbit_partition;
        let mut orbit = p[i as usize];
        if orbit != p[orbit as usize] {
            loop {
                self.link_compression_stack.push(i);
                i = orbit;
                orbit = p[orbit as usize];
                if orbit == p[orbit as usize] {
                    break;
                }
            }
            while let Some(k) = self.link_compression_stack.pop() {
                p[k as usize] = orbit;
            }
        }
        orbit
    }

    fn initialize_hash_values(&mut self) {
        for i in 0..self.num_vertices as usize {
            let cell = self.vertex_to_cell[i];
            for j in self.gstart[i] as usize..self.gend[i] as usize {
                let (v, c) = self.gedge[j];
                sparse_combine32(self.vertex_hash.get_or_insert_default(v), cell, c as u64);
            }
            self.mark_cell_for_refinement(cell);
        }
    }

    fn update_cell_membership(&mut self, i: i32, cell: i32, mark_for_refinement: bool) -> bool {
        let vertex = self.current_partition[i as usize];
        if self.vertex_to_cell[vertex as usize] == cell {
            return false;
        }
        self.vertex_to_cell[vertex as usize] = cell;
        if i != cell {
            self.current_partition_links[i as usize] = cell;
        }
        if mark_for_refinement {
            for j in self.gstart[vertex as usize] as usize..self.gend[vertex as usize] as usize {
                let (v, c) = self.gedge[j];
                let dest_cell = self.vertex_to_cell[v as usize];
                if self.cell_size(dest_cell) == 1 {
                    continue;
                }
                sparse_combine32(self.vertex_hash.get_or_insert_default(v), cell, c as u64);
                self.mark_cell_for_refinement(dest_cell);
            }
        }
        true
    }

    fn split_cell(&mut self, cell: i32, split_point: i32) -> bool {
        let h_split = self.get_vertex_hash(self.current_partition[split_point as usize]);
        let h_cell = self.get_vertex_hash(self.current_partition[cell as usize]);
        let certificate_val = (pair_hash::<0>(h_split, h_cell)
            .wrapping_add(pair_hash::<1>(
                cell as u32,
                (self.current_partition_links[cell as usize] - split_point) as u32,
            ))
            .wrapping_add(pair_hash::<2>(split_point as u32, (split_point - cell) as u32))
            >> 32) as u32;

        // prefix pruning as in bliss
        if !self.first_leave_certificate.is_empty() {
            let n = self.curr_node_certificate.len() as i32;
            self.first_leave_prefix_len += ((self.first_leave_prefix_len == n)
                && certificate_val == self.first_leave_certificate[n as usize])
                as i32;
            self.best_leave_prefix_len +=
                ((self.best_leave_prefix_len == n) && certificate_val == self.best_leave_certificate[n as usize])
                    as i32;
            // not a prefix of the first leave's certificate and
            // lexicographically after the smallest leave certificate: prune
            if self.first_leave_prefix_len <= n && self.best_leave_prefix_len <= n {
                let bp = self.best_leave_prefix_len as usize;
                let diff_val = if bp as i32 == n { certificate_val } else { self.curr_node_certificate[bp] };
                if diff_val > self.best_leave_certificate[bp] {
                    return false;
                }
            }
        }

        self.current_partition_links[split_point as usize] = self.current_partition_links[cell as usize];
        self.current_partition_links[cell as usize] = split_point;
        self.cell_creation_stack.push(split_point);
        self.curr_node_certificate.push(certificate_val);
        true
    }

    fn mark_cell_for_refinement(&mut self, cell: i32) {
        if self.cell_size(cell) == 1 || self.cell_in_refinement_queue[cell as usize] {
            return;
        }
        self.cell_in_refinement_queue[cell as usize] = true;
        self.refinement_queue.push(Reverse(cell));
    }

    fn get_vertex_hash(&self, v: i32) -> u32 {
        self.vertex_hash.find(&v).copied().unwrap_or(0)
    }

    fn clear_refinement_queue(&mut self) {
        for Reverse(c) in self.refinement_queue.drain() {
            self.cell_in_refinement_queue[c as usize] = false;
        }
        self.vertex_hash.clear();
    }

    fn partition_refinement(&mut self) -> bool {
        while let Some(Reverse(mut cell_start)) = self.refinement_queue.pop() {
            let first_cell_start = cell_start;
            self.cell_in_refinement_queue[cell_start as usize] = false;
            if self.cell_size(cell_start) == 1 {
                continue;
            }
            let cell_end = self.current_partition_links[cell_start as usize];
            debug_assert!(cell_end >= cell_start);

            // vertices with updated hash values go to the end of the cell
            let vh = &self.vertex_hash;
            let refine_start = cell_start
                + partition(
                    &mut self.current_partition[cell_start as usize..cell_end as usize],
                    |v| vh.find(v).is_none(),
                ) as i32;
            if refine_start == cell_end {
                continue;
            }
            pdqsort(
                &mut self.current_partition[refine_start as usize..cell_end as usize],
                |a, b| vh.find(a).unwrap() < vh.find(b).unwrap(),
            );

            if refine_start != cell_start {
                if !self.split_cell(cell_start, refine_start) {
                    self.clear_refinement_queue();
                    return false;
                }
                cell_start = refine_start;
                self.update_cell_membership(cell_start, cell_start, true);
            }

            let mut prune = false;
            let mut last_hash = self.get_vertex_hash(self.current_partition[cell_start as usize]);
            let mut i = cell_start + 1;
            while i < cell_end {
                let hash = self.get_vertex_hash(self.current_partition[i as usize]);
                if hash != last_hash {
                    if !self.split_cell(cell_start, i) {
                        prune = true;
                        break;
                    }
                    cell_start = i;
                    last_hash = hash;
                }
                self.update_cell_membership(i, cell_start, true);
                i += 1;
            }

            if prune {
                self.clear_refinement_queue();
                self.current_partition_links[first_cell_start as usize] = cell_end;
                // undo the incomplete changes to the cells
                i -= 1;
                while i >= refine_start {
                    self.update_cell_membership(i, first_cell_start, false);
                    i -= 1;
                }
                return false;
            }
            debug_assert_eq!(self.current_partition_links[cell_start as usize], cell_end);
        }
        self.vertex_hash.clear();
        true
    }

    fn select_target_cell(&self) -> i32 {
        let mut i = 0;
        if self.node_stack.len() > 1 {
            i = self.node_stack[self.node_stack.len() - 2].target_cell;
        }
        while i < self.num_vertices {
            if self.cell_size(i) > 1 {
                return i;
            }
            i += 1;
        }
        -1
    }

    fn check_stored_automorphism(&self, vertex: i32) -> bool {
        let num_check = self.num_automorphisms.min(64) as usize;
        let nv = self.num_vertices as usize;
        for i in 0..num_check {
            let aut = &self.automorphisms[i * nv..(i + 1) * nv];
            let mut useful = true;
            let mut j = self.node_stack.len() as i32 - 2;
            while j >= self.first_path_depth {
                let fix_pos = self.vertex_position[self.node_stack[j as usize].last_distinguished as usize] as usize;
                if aut[fix_pos] != self.vertex_ground_set[fix_pos] {
                    useful = false;
                    break;
                }
                j -= 1;
            }
            if !useful {
                continue;
            }
            if aut[self.vertex_position[vertex as usize] as usize] < vertex {
                return false;
            }
        }
        true
    }

    fn determine_next_to_distinguish(&mut self) -> bool {
        let node = *self.node_stack.last().unwrap();
        self.distinguish_cands.clear();
        let cs = node.target_cell as usize;
        let ce = self.current_partition_links[cs] as usize;
        // the first position holding the minimal candidate vertex
        let mut best: Option<usize> = None;
        if node.last_distinguished == -1 {
            for i in cs..ce {
                if best.map_or(true, |b| self.current_partition[i] < self.current_partition[b]) {
                    best = Some(i);
                }
            }
        } else {
            let check_aut = self.node_stack.len() as i32 > self.first_path_depth;
            for i in cs..ce {
                let v = self.current_partition[i];
                if v <= node.last_distinguished {
                    continue;
                }
                let cand = if check_aut {
                    self.check_stored_automorphism(v)
                } else {
                    let o = self.get_orbit(v);
                    self.vertex_ground_set[o as usize] == v
                };
                if cand && best.map_or(true, |b| v < self.current_partition[b]) {
                    best = Some(i);
                }
            }
            if best.is_none() {
                return false;
            }
        }
        self.distinguish_cands.push(best.unwrap());
        true
    }

    fn distinguish_vertex(&mut self, target_cell: i32) -> bool {
        debug_assert_eq!(self.distinguish_cands.len(), 1);
        let new_cell = self.current_partition_links[target_cell as usize] - 1;
        self.current_partition.swap(self.distinguish_cands[0], new_cell as usize);
        self.node_stack.last_mut().unwrap().last_distinguished = self.current_partition[new_cell as usize];
        if !self.split_cell(target_cell, new_cell) {
            return false;
        }
        self.update_cell_membership(new_cell, new_cell, true);
        true
    }

    fn backtrack(&mut self, new_end: i32, end: i32) {
        // backtracking always starts at a leaf (a discrete partition), so
        // every new cell is on the cell creation stack
        let mut pos = end - 1;
        while pos >= new_end {
            let cell = self.cell_creation_stack[pos as usize];
            let new_start = self.get_cell_start(cell - 1);
            let curr_end = self.current_partition_links[cell as usize];
            self.current_partition_links[cell as usize] = new_start;
            self.current_partition_links[new_start as usize] = curr_end;
            pos -= 1;
        }
    }

    fn cleanup_backtrack(&mut self, stack_pos: i32) {
        let mut sp = self.cell_creation_stack.len() as i32 - 1;
        while sp >= stack_pos {
            let cell = self.cell_creation_stack[sp as usize];
            let cell_start = self.get_cell_start(cell);
            let cell_end = self.current_partition_links[cell_start as usize];
            let mut v = cell;
            while v < cell_end && self.vertex_to_cell[self.current_partition[v as usize] as usize] == cell {
                self.update_cell_membership(v, cell_start, false);
                v += 1;
            }
            sp -= 1;
        }
        self.cell_creation_stack.truncate(stack_pos as usize);
    }

    fn get_cell_start(&mut self, mut pos: i32) -> i32 {
        let links = &mut self.current_partition_links;
        let mut start = links[pos as usize];
        if start > pos {
            return pos;
        }
        if links[start as usize] < start {
            loop {
                self.link_compression_stack.push(pos);
                pos = start;
                start = links[start as usize];
                if links[start as usize] >= start {
                    break;
                }
            }
            while let Some(p) = self.link_compression_stack.pop() {
                links[p as usize] = start;
            }
        }
        start
    }

    fn create_node(&mut self) {
        self.node_stack.push(Node {
            stack_start: self.cell_creation_stack.len() as i32,
            certificate_end: self.curr_node_certificate.len() as i32,
            target_cell: -1,
            last_distinguished: -1,
        });
    }

    pub fn load_model_as_graph(&mut self, model: &ModelView, epsilon: f64) {
        let num_col = model.num_col;
        let num_row = model.num_row;
        let (nc, nr) = (num_col as usize, num_row as usize);
        self.num_col = num_col;
        self.num_row = num_row;
        self.num_vertices = num_col + num_row;
        let nv = nc + nr;
        self.col_lower = model.col_lower[..nc].to_vec();
        self.col_upper = model.col_upper[..nc].to_vec();
        self.integrality = model.integrality[..nc].to_vec();

        self.cell_in_refinement_queue = vec![false; nv];
        self.vertex_to_cell = vec![0; nv];
        self.curr_node_certificate.reserve(nv);

        let mut column_set: HighsHashTable<MatrixColumn, i32> = HighsHashTable::new();
        let mut row_set: HighsHashTable<MatrixRow, i32> = HighsHashTable::new();
        let mut coloring = Coloring::new(epsilon);
        let num_nz = model.a_index.len();
        self.gedge = model.a_index.iter().map(|&r| (r + num_col, 0u32)).collect();
        self.gedge.resize(2 * num_nz, (0, 0));
        self.gstart = vec![0; nv + 1];
        self.gstart[..=nc].copy_from_slice(&model.a_start[..=nc]);

        // column colors and row sizes
        let mut row_sizes = vec![0i32; nr];
        for i in 0..nc {
            for j in self.gstart[i] as usize..self.gstart[i + 1] as usize {
                self.gedge[j].1 = coloring.color(model.a_value[j]);
                row_sizes[model.a_index[j] as usize] += 1;
            }
        }
        let mut offset = num_nz as i32;
        for i in 0..nr {
            self.gstart[nc + i] = offset;
            offset += row_sizes[i];
        }
        self.gstart[nc + nr] = offset;
        self.gend = self.gstart[1..].to_vec();

        // the row-wise copy
        for i in 0..nc {
            for j in self.gstart[i] as usize..self.gstart[i + 1] as usize {
                let row = model.a_index[j] as usize;
                let ar_pos = (self.gstart[nc + row + 1] - row_sizes[row]) as usize;
                row_sizes[row] -= 1;
                self.gedge[ar_pos] = (i as i32, self.gedge[j].1);
            }
        }

        // initial cells: columns by cost, bounds, integrality and length
        let index_offset = num_col + 1;
        for i in 0..nc {
            let key: MatrixColumn = [
                coloring.color(model.col_cost[i]),
                coloring.color(model.col_lower[i]),
                coloring.color(model.col_upper[i]),
                model.integrality[i] as u32,
                (self.gstart[i + 1] - self.gstart[i]) as u32,
            ];
            let mut cell = *column_set.get_or_insert_default(key);
            if cell == 0 {
                cell = column_set.len() as i32;
                if model.col_lower[i] != 0.0 || model.col_upper[i] != 1.0 || model.integrality[i] == 0 {
                    cell += index_offset;
                }
                *column_set.find_mut(&key).unwrap() = cell;
            }
            self.vertex_to_cell[i] = cell;
        }
        let index_offset = 2 * num_col + 1;
        for i in 0..nr {
            let key: MatrixRow = [
                coloring.color(model.row_lower[i]),
                coloring.color(model.row_upper[i]),
                (self.gstart[nc + i + 1] - self.gstart[nc + i]) as u32,
            ];
            let mut cell = *row_set.get_or_insert_default(key);
            if cell == 0 {
                cell = row_set.len() as i32;
                *row_set.find_mut(&key).unwrap() = cell;
            }
            self.vertex_to_cell[nc + i] = index_offset + cell;
        }

        // the initial partition, sorted by those cell numbers
        self.current_partition = (0..nv as i32).collect();
        {
            let v2c = &self.vertex_to_cell;
            pdqsort(&mut self.current_partition, |&a, &b| v2c[a as usize] < v2c[b as usize]);
        }
        self.current_partition_links = vec![0; nv];
        let mut cell_start = 0;
        let mut cell_number = 0;
        for i in 0..nv as i32 {
            let vertex = self.current_partition[i as usize] as usize;
            if cell_number != self.vertex_to_cell[vertex] {
                cell_number = self.vertex_to_cell[vertex];
                self.current_partition_links[cell_start as usize] = i;
                cell_start = i;
            }
            self.vertex_to_cell[vertex] = cell_start;
            self.current_partition_links[i as usize] = cell_start;
        }
        self.current_partition_links[cell_start as usize] = nv as i32;
    }

    fn dump_current_graph(&self) -> HighsHashTable<(i32, i32, u32)> {
        let mut g = HighsHashTable::new();
        for i in 0..self.num_col as usize {
            let col_cell = self.vertex_to_cell[i];
            for j in self.gstart[i] as usize..self.gend[i] as usize {
                let (v, c) = self.gedge[j];
                g.insert((self.vertex_to_cell[v as usize], col_cell, c), ());
            }
            for j in self.gend[i] as usize..self.gstart[i + 1] as usize {
                let (v, c) = self.gedge[j];
                g.insert((v, col_cell, c), ());
            }
        }
        g
    }

    fn switch_to_next_node(&mut self, backtrack_depth: i32) {
        let mut stack_end = self.cell_creation_stack.len() as i32;
        self.node_stack.truncate(backtrack_depth as usize);
        while let Some(&node) = self.node_stack.last() {
            self.backtrack(node.stack_start, stack_end);
            stack_end = node.stack_start;
            let depth = self.node_stack.len() as i32;
            self.first_path_depth = self.first_path_depth.min(depth);
            self.best_path_depth = self.best_path_depth.min(depth);
            self.first_leave_prefix_len = self.first_leave_prefix_len.min(node.certificate_end);
            self.best_leave_prefix_len = self.best_leave_prefix_len.min(node.certificate_end);
            self.curr_node_certificate.truncate(node.certificate_end as usize);
            if !self.determine_next_to_distinguish() {
                self.node_stack.pop();
                continue;
            }
            // with the final stack end, so that the hashes are up to date
            // and the link arrays hold no chains anymore
            self.cleanup_backtrack(stack_end);
            if !self.distinguish_vertex(node.target_cell) {
                // its certificate is lexicographically above the best leave's
                self.node_stack.pop();
                continue;
            }
            if !self.partition_refinement() {
                stack_end = self.cell_creation_stack.len() as i32;
                continue;
            }
            self.create_node();
            break;
        }
    }

    /// compareCurrentGraph: Err(cell) names the column cell whose
    /// neighbourhood differs
    fn compare_current_graph(&self, other: &HighsHashTable<(i32, i32, u32)>) -> Result<(), i32> {
        for i in 0..self.num_col as usize {
            let col_cell = self.vertex_to_cell[i];
            for j in self.gstart[i] as usize..self.gend[i] as usize {
                let (v, c) = self.gedge[j];
                if other.find(&(self.vertex_to_cell[v as usize], col_cell, c)).is_none() {
                    // a hash collision, rarely: backtrack to where this cell
                    // was targeted
                    return Err(col_cell);
                }
            }
            for j in self.gend[i] as usize..self.gstart[i + 1] as usize {
                let (v, c) = self.gedge[j];
                if other.find(&(v, col_cell, c)).is_none() {
                    return Err(col_cell);
                }
            }
        }
        Ok(())
    }

    fn is_binary_col(&self, col: usize) -> bool {
        self.col_lower[col] == 0.0 && self.col_upper[col] == 1.0 && self.integrality[col] != 0
    }

    fn is_from_binary_column(&self, pos: i32) -> bool {
        pos < self.num_active_cols && self.is_binary_col(self.current_partition[pos as usize] as usize)
    }

    fn compute_component_data(&mut self, symmetries: &Symmetries) -> ComponentData {
        let nac = self.num_active_cols as usize;
        let np = symmetries.num_perms as usize;
        let mut cd = ComponentData {
            components: HighsDisjointSets::new(nac),
            component_starts: Vec::new(),
            component_sets: Vec::new(),
            perm_component_starts: Vec::new(),
            perm_components: Vec::new(),
            first_unfixed: vec![-1; np],
            num_unfixed: vec![0; np],
        };
        for i in 0..np {
            let perm = &symmetries.permutations[i * nac..(i + 1) * nac];
            for j in 0..nac {
                if perm[j] != self.vertex_ground_set[j] {
                    let pos = self.vertex_position[perm[j] as usize];
                    cd.num_unfixed[i] += 1;
                    if cd.first_unfixed[i] != -1 {
                        cd.components.merge(cd.first_unfixed[i], pos);
                    } else {
                        cd.first_unfixed[i] = pos;
                    }
                }
            }
        }

        cd.component_sets = self.vertex_ground_set[..nac].to_vec();
        {
            let comps = &mut cd.components;
            let vp = &self.vertex_position;
            pdqsort(&mut cd.component_sets, |&u, &v| {
                let uc = comps.get_set(vp[u as usize]);
                let vc = comps.get_set(vp[v as usize]);
                (comps.get_set_size(uc) == 1, uc) < (comps.get_set_size(vc) == 1, vc)
            });
        }

        let mut current_component = -1;
        for i in 0..nac {
            let comp = cd.components.get_set(self.vertex_position[cd.component_sets[i] as usize]);
            if cd.components.get_set_size(comp) == 1 {
                break;
            }
            if comp != current_component {
                current_component = comp;
                cd.component_starts.push(i as i32);
            }
        }

        cd.perm_components = (0..np as i32).filter(|&i| cd.first_unfixed[i as usize] != -1).collect();
        {
            let comps = &mut cd.components;
            let (fu, nu) = (&cd.first_unfixed, &cd.num_unfixed);
            pdqsort(&mut cd.perm_components, |&i, &j| {
                let si = comps.get_set(fu[i as usize]);
                let sj = comps.get_set(fu[j as usize]);
                (si, nu[i as usize]) < (sj, nu[j as usize])
            });
        }

        let mut current_component = -1;
        let num_used_perms = cd.perm_components.len();
        for i in 0..num_used_perms {
            let p = cd.perm_components[i] as usize;
            let comp = cd.components.get_set(cd.first_unfixed[p]);
            if comp != current_component {
                current_component = comp;
                cd.perm_component_starts.push(i as i32);
            }
        }
        debug_assert_eq!(cd.perm_component_starts.len(), cd.component_starts.len());
        cd.perm_component_starts.push(num_used_perms as i32);
        cd.component_starts.push(nac as i32);
        cd
    }

    fn is_full_orbitope(&self, cd: &ComponentData, component: usize, symmetries: &mut Symmetries) -> bool {
        let nac = self.num_active_cols as usize;
        let (c0, c1) = (cd.component_starts[component] as usize, cd.component_starts[component + 1] as usize);
        let component_size = (c1 - c0) as i32;
        if component_size == 1 {
            return false;
        }
        // only binary columns
        if cd.component_sets[c0..c1].iter().any(|&col| !self.is_binary_col(col as usize)) {
            return false;
        }
        let (p0s, p1s) = (
            cd.perm_component_starts[component] as usize,
            cd.perm_component_starts[component + 1] as usize,
        );
        let p0 = cd.perm_components[p0s] as usize;
        // all unfixed columns of a permutation are in two-cycles, one per row
        if cd.num_unfixed[p0] & 1 != 0 {
            return false;
        }
        if cd.perm_components[p0s + 1..p1s].iter().any(|&p| cd.num_unfixed[p as usize] != cd.num_unfixed[p0]) {
            return false;
        }
        let num_rows = cd.num_unfixed[p0] >> 1;
        let orbit_size = component_size / num_rows;
        if orbit_size * num_rows != component_size {
            return false;
        }
        // n - 1 permutations for orbits of size n
        if (p1s - p0s) as i32 != orbit_size - 1 {
            return false;
        }

        // the first two columns from the first permutation
        let nr = num_rows as usize;
        let mut matrix = vec![-1i32; component_size as usize];
        let mut perm = &symmetries.permutations[p0 * nac..(p0 + 1) * nac];
        let mut col_set: HighsHashTable<i32> = HighsHashTable::new();
        let mut m = 0;
        for j in 0..nac {
            let j_image_pos = self.vertex_position[perm[j] as usize];
            if j_image_pos <= j as i32 {
                continue;
            }
            if m == nr {
                return false;
            }
            if perm[j_image_pos as usize] != self.vertex_ground_set[j] {
                return false;
            }
            matrix[m] = self.vertex_ground_set[j];
            matrix[nr + m] = perm[j];
            // each column only once
            if !col_set.insert(self.vertex_ground_set[j], ()) {
                return false;
            }
            if !col_set.insert(perm[j], ()) {
                return false;
            }
            m += 1;
        }

        let row_length = orbit_size as usize;
        let mut num_cols_added = 2;
        let mut tried_left_extension = false;
        while num_cols_added < row_length {
            if col_set.len() != num_cols_added * nr {
                return false;
            }
            let mut prev_col = num_cols_added - 1;
            let mut found_cand = false;
            loop {
                let move_pos = self.vertex_position[matrix[prev_col * nr] as usize] as usize;
                for k in p0s + 1..p1s {
                    let p = cd.perm_components[k] as usize;
                    perm = &symmetries.permutations[p * nac..(p + 1) * nac];
                    if perm[move_pos] != self.vertex_ground_set[move_pos] && col_set.find(&perm[move_pos]).is_none() {
                        found_cand = true;
                        break;
                    }
                }
                if !found_cand && !tried_left_extension {
                    // extend column zero instead
                    prev_col = 0;
                    tried_left_extension = true;
                    continue;
                }
                break;
            }
            if !found_cand {
                return false;
            }
            for j in 0..nr {
                let prev = matrix[prev_col * nr + j];
                let next_vertex = perm[self.vertex_position[prev as usize] as usize];
                matrix[num_cols_added * nr + j] = next_vertex;
                // a two-cycle
                if perm[self.vertex_position[next_vertex as usize] as usize] != prev {
                    return false;
                }
                if !col_set.insert(next_vertex, ()) {
                    return false;
                }
            }
            num_cols_added += 1;
        }
        if col_set.len() != component_size as usize {
            return false;
        }

        let index = symmetries.orbitopes.len() as i32;
        for &col in &matrix {
            symmetries.column_to_orbitope.insert(col, index);
        }
        symmetries.orbitopes.push(OrbitopeMatrix {
            row_length: orbit_size,
            num_rows,
            num_set_packing_rows: 0,
            column_to_row: HighsHashTable::new(),
            row_is_set_packing: Vec::new(),
            matrix,
        });
        true
    }

    pub fn initialize_detection(&mut self) -> bool {
        self.initialize_hash_values();
        self.partition_refinement();
        self.remove_fix_points();
        self.num_active_cols != 0
    }

    /// Stores the automorphism mapping the current leave to the leave with
    /// `partition`; returns whether the search should stop (enough perms)
    fn store_automorphism(&mut self, symmetries: &mut Symmetries, best: bool, max_perms: i32) -> bool {
        let nv = self.num_vertices as usize;
        let k = (self.num_automorphisms & 63) as usize;
        self.num_automorphisms += 1;
        for i in 0..nv {
            let leave_col =
                if best { self.best_leave_partition[i] } else { self.first_leave_partition[i] };
            let p = self.vertex_position[self.current_partition[i] as usize] as usize;
            self.automorphisms[k * nv + p] = leave_col;
        }
        let mut report = false;
        for i in 0..nv {
            let (a, g) = (self.automorphisms[k * nv + i], self.vertex_ground_set[i]);
            if self.merge_orbits(a, g) && i < self.num_active_cols as usize {
                debug_assert!(a < self.num_col);
                report = true;
            }
        }
        if report {
            let nac = self.num_active_cols as usize;
            symmetries.permutations.extend_from_slice(&self.automorphisms[k * nv..k * nv + nac]);
            symmetries.num_perms += 1;
            if symmetries.num_perms == max_perms {
                return true;
            }
        }
        false
    }

    /// run; `interrupted` is polled at each leave (HighsSplitDeque::
    /// checkInterrupt); returns false when it stopped the search
    pub fn run(&mut self, symmetries: &mut Symmetries, mut interrupted: impl FnMut() -> bool) -> bool {
        debug_assert!(self.num_active_cols != 0);
        self.initialize_ground_set();
        self.curr_node_certificate.clear();
        self.cell_creation_stack.clear();
        self.create_node();
        let max_perms = 64_000_000 / self.num_active_cols;
        while !self.node_stack.is_empty() {
            let target_cell = self.select_target_cell();
            if target_cell == -1 {
                if self.first_leave_partition.is_empty() {
                    self.first_leave_partition = self.current_partition.clone();
                    self.first_leave_certificate = self.curr_node_certificate.clone();
                    self.best_leave_certificate = self.curr_node_certificate.clone();
                    self.first_leave_graph = self.dump_current_graph();
                    self.first_path_depth = self.node_stack.len() as i32;
                    self.best_path_depth = self.node_stack.len() as i32;
                    self.first_leave_prefix_len = self.curr_node_certificate.len() as i32;
                    self.best_leave_prefix_len = self.curr_node_certificate.len() as i32;

                    let mut backtrack_depth = self.first_path_depth - 1;
                    while backtrack_depth > 0
                        && !self.is_from_binary_column(self.node_stack[backtrack_depth as usize - 1].target_cell)
                    {
                        backtrack_depth -= 1;
                    }
                    self.switch_to_next_node(backtrack_depth);
                } else {
                    let mut wrong_cell = -1;
                    let mut backtrack_depth = self.node_stack.len() as i32 - 1;
                    let n = self.curr_node_certificate.len() as i32;
                    debug_assert_eq!(n as usize, self.first_leave_certificate.len());
                    if self.first_leave_prefix_len == n || self.best_leave_prefix_len == n {
                        let first_match = self.first_leave_prefix_len == n
                            && match self.compare_current_graph(&self.first_leave_graph) {
                                Ok(()) => true,
                                Err(c) => {
                                    wrong_cell = c;
                                    false
                                }
                            };
                        let best_match = !first_match
                            && !self.best_leave_partition.is_empty()
                            && self.best_leave_prefix_len == n
                            && match self.compare_current_graph(&self.best_leave_graph) {
                                Ok(()) => true,
                                Err(c) => {
                                    wrong_cell = c;
                                    false
                                }
                            };
                        if first_match {
                            if self.store_automorphism(symmetries, false, max_perms) {
                                break;
                            }
                            backtrack_depth = backtrack_depth.min(self.first_path_depth);
                        } else if best_match {
                            if self.store_automorphism(symmetries, true, max_perms) {
                                break;
                            }
                            backtrack_depth = backtrack_depth.min(self.best_path_depth);
                        } else if self.best_leave_prefix_len < n
                            && self.curr_node_certificate[self.best_leave_prefix_len as usize]
                                > self.best_leave_certificate[self.best_leave_prefix_len as usize]
                        {
                            // lexicographically above the smallest certificate
                            // seen: maybe backtrack higher
                            let mut p = self.first_path_depth - 1;
                            while self.node_stack[p as usize].certificate_end <= self.best_leave_prefix_len {
                                p += 1;
                            }
                            backtrack_depth = backtrack_depth.min(p);
                        } else {
                            // a hash collision found by the graph comparison:
                            // backtrack to where the mismatching cell was
                            // targeted last
                            let mut p = backtrack_depth;
                            while p >= 0 {
                                if self.node_stack[p as usize].target_cell == wrong_cell {
                                    backtrack_depth = p;
                                    break;
                                }
                                p -= 1;
                            }
                        }
                    } else {
                        // a lexicographically smaller certificate than the
                        // best leave's
                        self.best_leave_certificate = self.curr_node_certificate.clone();
                        self.best_leave_graph = self.dump_current_graph();
                        self.best_leave_partition = self.current_partition.clone();
                        self.best_path_depth = self.node_stack.len() as i32;
                        self.best_leave_prefix_len = n;
                    }
                    self.switch_to_next_node(backtrack_depth);
                }
                if interrupted() {
                    return false;
                }
            } else {
                self.node_stack.last_mut().unwrap().target_cell = target_cell;
                let success = self.determine_next_to_distinguish();
                debug_assert!(success);
                if !self.distinguish_vertex(target_cell) {
                    self.switch_to_next_node(self.node_stack.len() as i32 - 1);
                    continue;
                }
                if !self.partition_refinement() {
                    self.switch_to_next_node(self.node_stack.len() as i32);
                    continue;
                }
                self.create_node();
            }
        }

        symmetries.num_generators = symmetries.num_perms;
        if symmetries.num_perms > 0 {
            self.post_process(symmetries);
        }
        true
    }

    /// orbitopes, and the generators restricted to the columns that are in
    /// non-trivial orbits and not in an orbitope
    fn post_process(&mut self, symmetries: &mut Symmetries) {
        self.vertex_position.truncate(self.num_col as usize);
        let cd = self.compute_component_data(symmetries);
        let num_components = cd.component_starts.len() - 1;
        for i in 0..num_components {
            if cd.component_starts[i + 1] - cd.component_starts[i] == 1 {
                continue;
            }
            self.is_full_orbitope(&cd, i, symmetries);
        }

        let nac = self.num_active_cols as usize;
        let mut deleted_perms = vec![false; symmetries.num_perms as usize];
        for (p, d) in deleted_perms.iter_mut().enumerate() {
            let perm = &symmetries.permutations[p * nac..(p + 1) * nac];
            *d = (0..nac).any(|i| {
                perm[i] != self.vertex_ground_set[i] && symmetries.column_to_orbitope.find(&perm[i]).is_some()
            });
        }

        let mut num_fixed = 0;
        for i in 0..nac {
            let v = self.vertex_ground_set[i];
            let o = self.get_orbit(v);
            if self.orbit_size[o as usize] == 1 || symmetries.column_to_orbitope.find(&v).is_some() {
                self.vertex_position[v as usize] = -1;
                self.vertex_ground_set[i] = -1;
                num_fixed += 1;
            }
        }

        if num_fixed != 0 {
            // compress the generators and the ground set
            let mut out = 0;
            for (p, &deleted) in deleted_perms.iter().enumerate() {
                if deleted {
                    symmetries.num_perms -= 1;
                    continue;
                }
                for i in 0..nac {
                    if self.vertex_ground_set[i] == -1 {
                        continue;
                    }
                    symmetries.permutations[out] = symmetries.permutations[p * nac + i];
                    out += 1;
                }
            }
            let mut out_pos = 0;
            for i in 0..nac {
                let v = self.vertex_ground_set[i];
                if v == -1 {
                    continue;
                }
                self.vertex_ground_set[out_pos] = v;
                self.vertex_position[v as usize] = out_pos as i32;
                out_pos += 1;
            }
            self.num_active_cols -= num_fixed;
            debug_assert_eq!(out, (symmetries.num_perms * self.num_active_cols) as usize);
        }

        let nac = self.num_active_cols as usize;
        self.vertex_ground_set.truncate(nac);
        symmetries.permutation_columns = std::mem::take(&mut self.vertex_ground_set);
        symmetries.column_position = std::mem::take(&mut self.vertex_position);
        symmetries.permutations.truncate(symmetries.num_perms as usize * nac);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two interchangeable bins: x0 + x1 <= 1, x2 + x3 <= 1 and the
    /// assignments x0 + x2 = 1, x1 + x3 = 1 have the symmetry swapping the
    /// bins (x0 <-> x1, x2 <-> x3) and the items (x0 <-> x2, x1 <-> x3).
    #[test]
    fn two_bins_two_items() {
        let start = [0, 2, 4, 6, 8];
        let index = [0, 2, 0, 3, 1, 2, 1, 3];
        let value = [1.0; 8];
        let cost = [1.0; 4];
        let (lo, up) = ([0.0; 4], [1.0; 4]);
        let integ = [1u8; 4];
        let (rlo, rup) = ([f64::NEG_INFINITY, f64::NEG_INFINITY, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        let model = ModelView {
            num_col: 4,
            num_row: 4,
            a_start: &start,
            a_index: &index,
            a_value: &value,
            col_cost: &cost,
            col_lower: &lo,
            col_upper: &up,
            integrality: &integ,
            row_lower: &rlo,
            row_upper: &rup,
        };
        let mut det = SymmetryDetection::default();
        det.load_model_as_graph(&model, 1e-9);
        assert!(det.initialize_detection());
        let mut sym = Symmetries::default();
        assert!(det.run(&mut sym, || false));
        assert_eq!(sym.num_generators, 2);
        // every column moves: all four are in one orbit
        assert_eq!(sym.permutation_columns.len() + sym.column_to_orbitope.len(), 4);
    }
}
