//! The C++ parts of the MIP solver's tables: HighsObjectiveFunction, the
//! HighsPseudocost handle with HighsPseudocostInitialization, the
//! HighsImplications handle with the implications' calls back
//! (`ImplicsHost`), the clique table's calls into the solver (`CliqueMip`)
//! and its operations as HighsCliqueTable made them, reduced cost fixing's
//! C++ steps and the symmetries' stabilizer orbits.

use super::dom::DomS;
use super::lp::LpS;
use super::pools::{ConflictPoolS, CutPoolS};
use super::solver::MipSolver;
use super::{prof, Prof};
use crate::lp_data::lp_presolve::PostsolveStack;
use crate::mip::clique::{CDom, CMip, CRows, CSepaCliques, CliqueTable, CliqueVar};
use crate::mip::clique_ffi as cq;
use crate::mip::cuts::sort::{partition, pdqsort};
use crate::mip::domain::DomChg;
use crate::mip::implications::{CImp, Implications};
use crate::mip::implications_ffi as imf;
use crate::mip::pseudocost::{ffi as psf, CPscostInit, Pseudocost};
use crate::mip::redcost::{ffi as rcf, CRedcost, RedcostFixing};
use crate::presolve::hpresolve::ffi::LiftOpp;
use crate::presolve::symmetry::{StabilizerOrbits, Symmetries};
use crate::util::cdouble::CDouble;
use crate::util::hash::HighsHash;
use crate::util::random::HighsRandom;
use std::ffi::c_void;

// ---- HighsObjectiveFunction ----

pub struct ObjFunc {
    obj_int_scale: f64,
    num_integral: i32,
    num_binary: i32,
    objective_nonzeros: Vec<i32>,
    objective_vals: Vec<f64>,
    clique_partition_start: Vec<i32>,
    col_to_partition: Vec<i32>,
}

impl ObjFunc {
    /// HighsObjectiveFunction(mipsolver)
    pub fn new(ms: &MipSolver) -> ObjFunc {
        let model = ms.model();
        let n = model.num_col as usize;
        let mut nz: Vec<i32> = Vec::with_capacity(n);
        for i in 0..n {
            if model.col_cost[i] != 0.0 {
                nz.push(i as i32);
            }
        }
        let mut f = ObjFunc {
            obj_int_scale: 0.0,
            num_integral: 0,
            num_binary: 0,
            objective_nonzeros: nz,
            objective_vals: Vec::new(),
            clique_partition_start: vec![0],
            col_to_partition: vec![-1; n],
        };
        if f.objective_nonzeros.is_empty() {
            f.obj_int_scale = 1.0;
            return f;
        }
        let integ = &model.integrality;
        f.num_integral = partition(&mut f.objective_nonzeros, |&i| integ[i as usize] != 0) as i32;
        f.num_binary = if f.num_integral == 0 {
            0
        } else {
            partition(&mut f.objective_nonzeros[..f.num_integral as usize], |&i| {
                model.col_lower[i as usize] == 0.0 && model.col_upper[i as usize] == 1.0
            }) as i32
        };
        f.objective_vals = f.objective_nonzeros.iter().map(|&i| model.col_cost[i as usize]).collect();
        f
    }

    /// setupCliquePartition(globaldom, cliqueTable)
    pub fn setup_clique_partition(&mut self, ms: &MipSolver, table: &mut CliqueTable) {
        if self.num_binary <= 1 {
            return;
        }
        let cost = &ms.model().col_cost;
        let nb = self.num_binary as usize;
        let mut clqvars: Vec<CliqueVar> =
            self.objective_nonzeros[..nb].iter().map(|&col| CliqueVar::new(col, (cost[col as usize] < 0.0) as i32)).collect();
        let mut starts = Vec::new();
        table.clique_partition_obj(cost, &mut clqvars, &mut starts);
        self.clique_partition_start = starts;
        let num_partitions = self.clique_partition_start.len() - 1;
        if num_partitions == nb {
            self.clique_partition_start.truncate(1);
        } else {
            let mut p = 0;
            let mut k = 0;
            for i in 0..num_partitions {
                let (s, e) = (self.clique_partition_start[i] as usize, self.clique_partition_start[i + 1] as usize);
                if e - s == 1 {
                    continue;
                }
                self.clique_partition_start[p] = k;
                for v in &clqvars[s..e] {
                    self.col_to_partition[v.col() as usize] = k;
                    k += 1;
                }
                p += 1;
            }
            self.clique_partition_start[p] = k;
            self.clique_partition_start.truncate(p + 1);
            let ctp = &self.col_to_partition;
            pdqsort(&mut self.objective_nonzeros[..nb], |&i, &j| {
                let (pi, pj) = (ctp[i as usize] as u32, ctp[j as usize] as u32);
                (pi, i.highs_hash()) < (pj, j.highs_hash())
            });
            for i in 0..nb {
                self.objective_vals[i] = cost[self.objective_nonzeros[i] as usize];
            }
        }
    }

    /// checkIntegrality(epsilon)
    pub fn check_integrality(&mut self, epsilon: f64) {
        if self.num_integral as usize == self.objective_nonzeros.len() {
            if self.num_integral != 0 {
                self.obj_int_scale = crate::mip::cuts::integers::integral_scale(&self.objective_vals, epsilon, epsilon);
                if self.obj_int_scale * 1e-14 > epsilon {
                    self.obj_int_scale = 0.0;
                }
            } else {
                self.obj_int_scale = 1.0;
            }
        }
    }

    pub fn objective_nonzeros(&self) -> &[i32] {
        &self.objective_nonzeros
    }
    pub fn objective_vals(&self) -> &[f64] {
        &self.objective_vals
    }
    pub fn num_binaries(&self) -> i32 {
        self.num_binary
    }
    pub fn clique_partition_start(&self) -> &[i32] {
        &self.clique_partition_start
    }
    pub fn num_clique_partitions(&self) -> i32 {
        self.clique_partition_start.len() as i32 - 1
    }
    pub fn col_to_partition(&self) -> &[i32] {
        &self.col_to_partition
    }
    pub fn integral_scale(&self) -> f64 {
        self.obj_int_scale
    }
    pub fn is_integral(&self) -> bool {
        self.obj_int_scale != 0.0
    }
}

// ---- HighsPseudocost ----

/// HighsPseudocost: a handle of the Rust pseudocosts (null when default
/// constructed)
pub struct PscostS {
    pub rs: *mut Pseudocost,
}

// SAFETY: as the solver's
unsafe impl Send for PscostS {}

impl PscostS {
    pub fn null() -> PscostS {
        PscostS { rs: std::ptr::null_mut() }
    }
    /// HighsPseudocost(mipsolver), initialised from mipsolver.pscostinit
    pub fn new(ms: &MipSolver) -> PscostS {
        let ncol = ms.num_col();
        let rs = psf::highs_rs_pscost_new(ncol, ms.opts.mip_pscost_minreliable);
        if !ms.pscostinit.is_null() {
            let st = &ms.d().postsolve_stack;
            let orig: Vec<i32> = (0..ncol as usize).map(|i| st.orig_col_index[i]).collect();
            // SAFETY: the initialization outlives the solver's setup
            let init = unsafe { &*ms.pscostinit };
            let a = init.arrays();
            // SAFETY: the arrays and the index map of the columns
            unsafe { psf::highs_rs_pscost_init(rs, &a, orig.as_ptr()) };
        }
        PscostS { rs }
    }
    /// The copy constructor
    pub fn copy(other: &PscostS) -> PscostS {
        // SAFETY: a live handle or null
        PscostS { rs: unsafe { psf::highs_rs_pscost_clone(other.rs) } }
    }
    /// operator= of a new value (move: the old freed)
    pub fn set(&mut self, other: PscostS) {
        let mut other = other;
        std::mem::swap(&mut self.rs, &mut other.rs);
    }
    pub fn p<'a>(&self) -> &'a mut Pseudocost {
        // SAFETY: the live pseudocosts, used as the C++ handle's
        unsafe { &mut *self.rs }
    }
    pub fn add_inference_observation(&self, col: i32, n: i32, up: bool) {
        // SAFETY: a live handle
        unsafe { psf::highs_rs_pscost_set(self.rs, 6, col, n, if up { 1.0 } else { 0.0 }) }
    }
    pub fn remove_changed(&self) {
        // SAFETY: a live handle
        unsafe { psf::highs_rs_pscost_set(self.rs, 7, 0, 0, 0.0) }
    }
    pub fn set_min_reliable(&self, r: i32) {
        // SAFETY: a live handle
        unsafe { psf::highs_rs_pscost_set(self.rs, 0, 0, r, 0.0) }
    }
    /// flushPseudoCost(other) / syncPseudoCost(other)
    pub fn flush(&self, other: &PscostS, sync: bool) {
        // SAFETY: distinct live handles
        unsafe { psf::highs_rs_pscost_flush(self.rs, other.rs, sync) }
    }
}

impl Drop for PscostS {
    fn drop(&mut self) {
        // SAFETY: owned (or null)
        unsafe { psf::highs_rs_pscost_free(self.rs) };
    }
}

/// HighsPseudocostInitialization
pub struct PscostInit {
    pub pseudocostup: Vec<f64>,
    pub pseudocostdown: Vec<f64>,
    pub nsamplesup: Vec<i32>,
    pub nsamplesdown: Vec<i32>,
    pub inferencesup: Vec<f64>,
    pub inferencesdown: Vec<f64>,
    pub ninferencesup: Vec<i32>,
    pub ninferencesdown: Vec<i32>,
    pub conflictscoreup: Vec<f64>,
    pub conflictscoredown: Vec<f64>,
    pub cost_total: f64,
    pub inferences_total: f64,
    pub conflict_avg_score: f64,
    pub nsamplestotal: i64,
    pub ninferencestotal: i64,
}

impl PscostInit {
    fn sized(n: usize) -> PscostInit {
        PscostInit {
            pseudocostup: vec![0.0; n],
            pseudocostdown: vec![0.0; n],
            nsamplesup: vec![0; n],
            nsamplesdown: vec![0; n],
            inferencesup: vec![0.0; n],
            inferencesdown: vec![0.0; n],
            ninferencesup: vec![0; n],
            ninferencesdown: vec![0; n],
            conflictscoreup: vec![0.0; n],
            conflictscoredown: vec![0.0; n],
            cost_total: 0.0,
            inferences_total: 0.0,
            conflict_avg_score: 0.0,
            nsamplestotal: 0,
            ninferencestotal: 0,
        }
    }
    fn arrays(&self) -> CPscostInit {
        fn m<T>(v: &[T]) -> *mut T {
            v.as_ptr() as *mut T
        }
        CPscostInit {
            pseudocostup: m(&self.pseudocostup),
            pseudocostdown: m(&self.pseudocostdown),
            nsamplesup: m(&self.nsamplesup),
            nsamplesdown: m(&self.nsamplesdown),
            inferencesup: m(&self.inferencesup),
            inferencesdown: m(&self.inferencesdown),
            ninferencesup: m(&self.ninferencesup),
            ninferencesdown: m(&self.ninferencesdown),
            conflictscoreup: m(&self.conflictscoreup),
            conflictscoredown: m(&self.conflictscoredown),
            n: self.pseudocostup.len() as i32,
            cost_total: self.cost_total,
            inferences_total: self.inferences_total,
            conflict_avg_score: self.conflict_avg_score,
            nsamplestotal: self.nsamplestotal,
            ninferencestotal: self.ninferencestotal,
        }
    }
    fn export(&mut self, ps: &PscostS, max_count: i32, orig: *const i32) {
        let mut a = self.arrays();
        // SAFETY: the arrays (written in place) and the index map
        unsafe { psf::highs_rs_pscost_export(ps.rs, max_count, orig, &mut a) };
        self.cost_total = a.cost_total;
        self.inferences_total = a.inferences_total;
        self.conflict_avg_score = a.conflict_avg_score;
        self.nsamplestotal = a.nsamplestotal;
        self.ninferencestotal = a.ninferencestotal;
    }
    /// HighsPseudocostInitialization(pscost, maxCount)
    pub fn new(ps: &PscostS, max_count: i32) -> PscostInit {
        // SAFETY: a live handle
        let n = unsafe { psf::highs_rs_pscost_geti(ps.rs, 6, 0) };
        let mut i = PscostInit::sized(n as usize);
        i.export(ps, max_count, std::ptr::null());
        i
    }
    /// HighsPseudocostInitialization(pscost, maxCount, postsolveStack)
    pub fn new_presolved(ps: &PscostS, max_count: i32, stack: &PostsolveStack) -> PscostInit {
        let mut i = PscostInit::sized(stack.orig_num_col as usize);
        // SAFETY: a live handle
        let ncols = unsafe { psf::highs_rs_pscost_geti(ps.rs, 6, 0) } as usize;
        let orig: Vec<i32> = (0..ncols).map(|k| stack.orig_col_index[k]).collect();
        i.export(ps, max_count, orig.as_ptr());
        i
    }
}

// ---- HighsImplications ----

/// HighsImplications: the Rust implications and the lifting collector of
/// probing (storeLiftingOpportunity)
pub struct ImplicsS {
    rs: Box<Implications>,
    pub mipsolver: *mut MipSolver,
    pub store_lifting: *mut Vec<LiftOpp>,
}

// SAFETY: as the solver's
unsafe impl Send for ImplicsS {}

/// The ImpCtx of the callbacks
struct ImpCtx {
    imp: *mut ImplicsS,
    cutpool: *const CutPoolS,
}

impl ImplicsS {
    pub fn new(ms: &MipSolver) -> ImplicsS {
        ImplicsS {
            rs: Box::new(Implications::new(ms.num_col(), ms.num_nonzero())),
            mipsolver: ms as *const MipSolver as *mut MipSolver,
            store_lifting: std::ptr::null_mut(),
        }
    }
    pub fn rs(&self) -> *mut Implications {
        &*self.rs as *const Implications as *mut Implications
    }
    pub fn imp<'a>(&self) -> &'a mut Implications {
        // SAFETY: the solver's implications, used as the C++ handle's
        unsafe { &mut *self.rs() }
    }
    fn ms<'a>(&self) -> &'a MipSolver {
        // SAFETY: the solver outlives its implications
        unsafe { &*self.mipsolver }
    }
    fn host(&self, ctx: &ImpCtx) -> CImp {
        let ms = self.ms();
        let sc = &ms.d().sc;
        CImp {
            ctx: ctx as *const ImpCtx as *mut c_void,
            feastol: sc.feastol,
            epsilon: sc.epsilon,
            num_nonzero: ms.num_nonzero(),
            num_nodes_down: imp_nodes_down,
            num_nodes_up: imp_nodes_up,
            lifting_begin: imp_lifting_begin,
            lifting_store: imp_lifting_store,
            domchg_reason: imp_domchg_reason,
            changed_cols_len: imp_changed_cols_len,
            backtrack: imp_backtrack,
            vertex_infeasible: imp_vertex_infeasible,
            add_inference_observation: imp_inference,
            clique_num_entries: imp_clique_num_entries,
            add_clique2: imp_add_clique2,
            clique_substituted: imp_clique_substituted,
            parallel_lock_active: imp_parallel_lock,
            clique_is_full: imp_clique_is_full,
            clique_num_queries: imp_clique_num_queries,
            run_clique_merging: imp_run_clique_merging,
            num_clique_entries_after_first_presolve: imp_entries_after_first_presolve,
            probing_clock: imp_probing_clock,
            add_cut: imp_add_cut,
        }
    }
    fn ctx(&self, cutpool: *const CutPoolS) -> ImpCtx {
        ImpCtx { imp: self as *const ImplicsS as *mut ImplicsS, cutpool }
    }

    pub fn too_many_var_bounds(&self) -> bool {
        // SAFETY: a live table
        unsafe { imf::highs_rs_implics_get(self.rs(), 1) != 0 }
    }
    /// addVUB / addVLB(col, vbcol, coef, constant) with the global
    /// domain's bound
    pub fn add_vb(&self, vlb: bool, col: i32, vbcol: i32, coef: f64, constant: f64) {
        let ms = self.ms();
        let gd = ms.d().get_domain();
        let bound = if vlb { gd.col_lower()[col as usize] } else { gd.col_upper()[col as usize] };
        // SAFETY: a live table
        unsafe {
            imf::highs_rs_implics_add_vb(
                self.rs(),
                vlb,
                col,
                vbcol,
                coef,
                constant,
                bound,
                ms.is_col_integral(col),
                ms.d().sc.feastol,
            )
        };
    }
    /// getBestVub / getBestVlb on the LP solution: (column, coef,
    /// constant)
    pub fn get_best_vb(
        &self,
        vlb: bool,
        col: i32,
        col_value: &[f64],
        col_dual: &[f64],
        bound: &mut f64,
        globaldom: &mut DomS,
    ) -> (i32, f64, f64) {
        let ctx = self.ctx(std::ptr::null());
        let h = self.host(&ctx);
        let d = globaldom.cdom();
        let mut vb = crate::mip::cuts::round::VarBound { coef: 0.0, constant: 0.0 };
        // SAFETY: the views and arrays of num_col entries
        let c = unsafe {
            imf::highs_rs_implics_best_vb(
                self.rs(),
                vlb,
                &h,
                &d,
                col,
                col_value.as_ptr(),
                col_dual.as_ptr(),
                col_value.len() as i32,
                bound,
                &mut vb,
            )
        };
        (c, vb.coef, vb.constant)
    }
    /// runProbing(col, numReductions)
    pub fn run_probing(&self, col: i32, num_reductions: &mut i32) -> bool {
        let ctx = self.ctx(std::ptr::null());
        let h = self.host(&ctx);
        let g = self.ms().d().get_domain().cdom();
        // SAFETY: the views
        unsafe { imf::highs_rs_implics_run_probing(self.rs(), &g, &h, col, num_reductions) }
    }
    /// rebuild(ncols, orig2reducedcol, orig2reducedrow)
    pub fn rebuild(&self, ncols: i32, orig2reducedcol: &[i32]) {
        let ms = self.ms();
        let st = &ms.d().postsolve_stack;
        let transformable: Vec<u8> = (0..ncols).map(|c| col_linearly_transformable(st, c) as u8).collect();
        let g = ms.d().get_domain().cdom();
        // SAFETY: the views and arrays
        unsafe {
            imf::highs_rs_implics_rebuild(
                self.rs(),
                &g,
                ncols,
                orig2reducedcol.as_ptr(),
                orig2reducedcol.len() as i32,
                transformable.as_ptr(),
            )
        };
    }
    /// buildFrom(init)
    pub fn build_from(&self, init: &ImplicsS) {
        let g = self.ms().d().get_domain().cdom();
        // SAFETY: live tables
        unsafe { imf::highs_rs_implics_build_from(self.rs(), &g, init.rs()) };
    }
    /// separateImpliedBounds(lp, sol, cutpool, feastol, globaldom,
    /// thread_safe)
    pub fn separate_implied_bounds(
        &self,
        lp: &LpS,
        sol: &[f64],
        cutpool: &CutPoolS,
        feastol: f64,
        globaldom: &mut DomS,
        thread_safe: bool,
    ) {
        let ctx = self.ctx(cutpool);
        let h = self.host(&ctx);
        let g = self.ms().d().get_domain().cdom();
        let d = globaldom.cdom();
        let frac = lp.frac();
        // SAFETY: the views and arrays
        unsafe {
            imf::highs_rs_implics_separate(
                self.rs(),
                &g,
                &h,
                &d,
                frac.as_ptr() as *const imf::FracInt,
                frac.len() as i32,
                sol.as_ptr(),
                sol.len() as i32,
                feastol,
                thread_safe,
            )
        };
    }
    /// cleanupVarbounds(col)
    pub fn cleanup_varbounds(&self, col: i32) {
        let ctx = self.ctx(std::ptr::null());
        let h = self.host(&ctx);
        let g = self.ms().d().get_domain().cdom();
        // SAFETY: the views
        unsafe { imf::highs_rs_implics_cleanup_varbounds(self.rs(), &g, &h, col) };
    }
}

fn imp_ctx<'a>(p: *mut c_void) -> (&'a ImplicsS, &'a MipSolver, *const CutPoolS) {
    // SAFETY: the ImpCtx of the call
    unsafe {
        let c = &*(p as *const ImpCtx);
        let i = &*c.imp;
        (i, &*i.mipsolver, c.cutpool)
    }
}

unsafe extern "C" fn imp_nodes_down(p: *mut c_void, col: i32) -> i64 {
    imp_ctx(p).1.d().nodequeue.num_nodes_down(col)
}
unsafe extern "C" fn imp_nodes_up(p: *mut c_void, col: i32) -> i64 {
    imp_ctx(p).1.d().nodequeue.num_nodes_up(col)
}
unsafe extern "C" fn imp_lifting_begin(p: *mut c_void) {
    let (i, ms, _) = imp_ctx(p);
    let gd = ms.d().get_domain();
    debug_assert!(gd.redundant_rows.is_empty());
    if !i.store_lifting.is_null() {
        gd.record_redundant_rows = true;
    }
}
unsafe extern "C" fn imp_lifting_store(p: *mut c_void, col: i32, val: bool) {
    let (i, ms, _) = imp_ctx(p);
    let gd = ms.d().get_domain();
    if !i.store_lifting.is_null() {
        let store = &mut *i.store_lifting;
        let rows: Vec<i32> = gd.redundant_rows.iter().map(|(&r, _)| r).collect();
        for row in rows {
            let coef = if val { -1.0 } else { 1.0 } * gd.get_redundant_row_value(row);
            store.push(LiftOpp { row, col, val: val as i32, coef });
        }
        gd.redundant_rows.clear();
        gd.record_redundant_rows = false;
    }
}
unsafe extern "C" fn imp_domchg_reason(p: *mut c_void, k: i32, index: *mut i32) -> i32 {
    let r = imp_ctx(p).1.d().get_domain().reasons()[k as usize];
    *index = r.index;
    r.kind
}
unsafe extern "C" fn imp_changed_cols_len(p: *mut c_void) -> usize {
    imp_ctx(p).1.d().get_domain().changed_cols().len()
}
unsafe extern "C" fn imp_backtrack(p: *mut c_void, changedend: usize) {
    let gd = imp_ctx(p).1.d().get_domain();
    gd.backtrack();
    gd.clear_changed_cols_from(changedend);
}
unsafe extern "C" fn imp_vertex_infeasible(p: *mut c_void, col: i32, val: i32) {
    let ms = imp_ctx(p).1;
    vertex_infeasible(ms, ms.d().get_domain(), col, val);
}
unsafe extern "C" fn imp_inference(p: *mut c_void, col: i32, n: i32, val: bool) {
    imp_ctx(p).1.d().get_pseudo_cost().add_inference_observation(col, n, val);
}
unsafe extern "C" fn imp_clique_num_entries(p: *mut c_void) -> i32 {
    imp_ctx(p).1.d().cliquetable.num_entries
}
unsafe extern "C" fn imp_add_clique2(p: *mut c_void, clique: *mut CliqueVar) {
    let ms = imp_ctx(p).1;
    add_clique(ms, std::slice::from_raw_parts_mut(clique, 2), false, i32::MAX);
}
unsafe extern "C" fn imp_clique_substituted(p: *mut c_void, col: i32) -> bool {
    imp_ctx(p).1.d().cliquetable.get_substitution(col).is_some()
}
unsafe extern "C" fn imp_parallel_lock(p: *mut c_void) -> bool {
    imp_ctx(p).1.d().parallel_lock_active()
}
unsafe extern "C" fn imp_clique_is_full(p: *mut c_void) -> bool {
    imp_ctx(p).1.d().cliquetable.is_full()
}
unsafe extern "C" fn imp_clique_num_queries(p: *mut c_void) -> *mut i64 {
    &mut imp_ctx(p).1.d().cliquetable.num_neighbourhood_queries
}
unsafe extern "C" fn imp_run_clique_merging(p: *mut c_void) {
    let ms = imp_ctx(p).1;
    run_clique_merging(ms, ms.d().get_domain());
}
unsafe extern "C" fn imp_entries_after_first_presolve(p: *mut c_void) -> i32 {
    imp_ctx(p).1.d().sc.num_clique_entries_after_first_presolve
}
unsafe extern "C" fn imp_probing_clock(p: *mut c_void, start: bool) {
    let ms = imp_ctx(p).1;
    if start {
        ms.prof.start(prof::PROBING_IMPLICATIONS);
    } else {
        ms.prof.stop(prof::PROBING_IMPLICATIONS);
    }
}
unsafe extern "C" fn imp_add_cut(p: *mut c_void, inds: *mut i32, vals: *mut f64, len: i32, rhs: f64, integral: bool, propagate: bool) {
    let (_, ms, cutpool) = imp_ctx(p);
    let (i, v) = (std::slice::from_raw_parts_mut(inds, len as usize), std::slice::from_raw_parts_mut(vals, len as usize));
    (*cutpool).add_cut(ms, i, v, rhs, integral, propagate, false, false);
}

// ---- the clique table ----

unsafe extern "C" fn clq_prune_edge(p: *mut c_void, v1: CliqueVar, v2: CliqueVar) {
    let ms = &*(p as *const MipSolver);
    let d = ms.d();
    if d.nodequeue.num_nodes() == 0 {
        return;
    }
    let mut tw = d.sc.pruned_treeweight;
    crate::mip::nodequeue::ffi::highs_rs_nodequeue_prune_edge(&mut *d.nodequeue, v1.col(), v1.val(), v2.col(), v2.val(), &mut tw);
    d.sc.pruned_treeweight = tw;
}
unsafe extern "C" fn clq_too_many_var_bounds(p: *mut c_void) -> bool {
    (*(p as *const MipSolver)).d().implications.too_many_var_bounds()
}
unsafe extern "C" fn clq_add_vub(p: *mut c_void, col: i32, bincol: i32, coef: f64, constant: f64) {
    (*(p as *const MipSolver)).d().implications.add_vb(false, col, bincol, coef, constant);
}
unsafe extern "C" fn clq_add_vlb(p: *mut c_void, col: i32, bincol: i32, coef: f64, constant: f64) {
    (*(p as *const MipSolver)).d().implications.add_vb(true, col, bincol, coef, constant);
}

/// highs_rs::cliqueMip
pub fn cmip(ms: &MipSolver) -> CMip {
    let d = ms.d();
    CMip {
        ctx: ms as *const MipSolver as *mut c_void,
        feastol: d.sc.feastol,
        epsilon: d.sc.epsilon,
        num_clique_entries_after_presolve: d.sc.num_clique_entries_after_presolve,
        num_nonzero: ms.num_nonzero(),
        prune_edge: clq_prune_edge,
        too_many_var_bounds: clq_too_many_var_bounds,
        add_vub: clq_add_vub,
        add_vlb: clq_add_vlb,
    }
}

fn table(ms: &MipSolver) -> *mut CliqueTable {
    &mut *ms.d().cliquetable
}

/// cliquetable.addClique(mipsolver, vars, equality, origin)
pub fn add_clique(ms: &MipSolver, vars: &mut [CliqueVar], equality: bool, origin: i32) {
    let dom = ms.d().get_domain().cdom();
    let m = cmip(ms);
    // SAFETY: the solver's table and views
    unsafe { cq::highs_rs_clique_add_clique(table(ms), &dom, &m, vars.as_mut_ptr(), vars.len() as i32, equality, origin) };
}

/// cliquetable.extractCliques(mipsolver, transformRows)
pub fn extract_cliques(ms: &MipSolver, transform_rows: bool) {
    let d = ms.d();
    // the rows up to the first that is not an original row
    let mut num_row = 0;
    let orig_rows = ms.orig().num_row;
    while num_row != ms.num_row() && d.postsolve_stack.orig_row_index[num_row as usize] < orig_rows {
        num_row += 1;
    }
    let r = CRows {
        num_row,
        ar_start: d.vecs.ar_start.as_slice().as_ptr(),
        ar_index: d.vecs.ar_index.as_slice().as_ptr(),
        ar_value: d.vecs.ar_value.as_slice().as_ptr(),
        num_nz: d.vecs.ar_index.len() as i32,
        row_lower: ms.model().row_lower.as_ptr(),
        row_upper: ms.model().row_upper.as_ptr(),
    };
    let dom = d.get_domain().cdom();
    let m = cmip(ms);
    // SAFETY: the solver's table and views
    unsafe { cq::highs_rs_clique_extract_cliques(table(ms), &dom, &m, &r, transform_rows) };
}

/// cliquetable.extractCliquesFromCut
pub fn extract_cliques_from_cut(ms: &MipSolver, inds: &[i32], vals: &[f64], rhs: f64) {
    let dom = ms.d().get_domain().cdom();
    let m = cmip(ms);
    // SAFETY: the solver's table and views
    unsafe {
        cq::highs_rs_clique_extract_from_cut(table(ms), &dom, &m, inds.as_ptr(), vals.as_ptr(), inds.len() as i32, rhs)
    };
}

/// cliquetable.extractObjCliques(mipsolver)
pub fn extract_obj_cliques(ms: &MipSolver) {
    let d = ms.d();
    let nbin = d.objective_function.num_binaries();
    if nbin <= 1 {
        return;
    }
    let gd = d.get_domain();
    if gd.get_objective_lower_bound() == -f64::INFINITY {
        return;
    }
    let (vals, inds, len, rhs) = gd.get_cutoff_constraint();
    // SAFETY: the cutoff constraint's arrays, valid until the domain changes
    let (iv, vv) = unsafe { (crate::ffi::sl(inds, len), crate::ffi::sl(vals, len)) };
    let (_, minact) = gd.compute_activity(iv, vv, false);
    let dom = gd.cdom();
    let m = cmip(ms);
    // SAFETY: the solver's table and views
    unsafe {
        cq::highs_rs_clique_extract_obj(table(ms), &dom, &m, nbin, vals, inds, len, rhs, minact.hi, minact.lo)
    };
}

/// cliquetable.vertexInfeasible(globaldom, col, val)
pub fn vertex_infeasible(ms: &MipSolver, globaldom: &mut DomS, col: i32, val: i32) {
    let dom = globaldom.cdom();
    // SAFETY: the solver's table and views
    unsafe { cq::highs_rs_clique_vertex_infeasible(table(ms), &dom, col, val) };
}

unsafe extern "C" fn clq_add_cut(p: *mut c_void, inds: *mut i32, vals: *mut f64, len: i32, rhs: f64) {
    let (ms, cutpool) = *(p as *const (*const MipSolver, *const CutPoolS));
    let (i, v) = (std::slice::from_raw_parts_mut(inds, len as usize), std::slice::from_raw_parts_mut(vals, len as usize));
    (*cutpool).add_cut(&*ms, i, v, rhs, true, false, false, false);
}

/// cliquetable.separateCliques(mipsolver, sol, cutpool, feastol, randgen,
/// localNumNeighbourhoodQueries)
pub fn separate_cliques(
    ms: &MipSolver,
    sol: &[f64],
    cutpool: &CutPoolS,
    feastol: f64,
    randgen: *mut HighsRandom,
    local_num_queries: *mut i64,
) {
    let d = ms.d();
    let ctx: (*const MipSolver, *const CutPoolS) = (ms, cutpool);
    let s = CSepaCliques {
        sol: sol.as_ptr(),
        num_col: sol.len() as i32,
        integral_cols: d.vecs.integral_cols.as_slice().as_ptr(),
        num_integral_cols: d.vecs.integral_cols.len() as i32,
        feastol,
        max_neighbourhood_queries: 1000000 + 100 * ms.num_nonzero() as i64 + d.sc.total_lp_iterations * 1000,
        ctx: &ctx as *const _ as *mut c_void,
        add_cut: clq_add_cut,
    };
    let dom = d.get_domain().cdom();
    // SAFETY: the solver's table and views
    unsafe { cq::highs_rs_clique_separate(table(ms), &dom, &s, randgen, local_num_queries) };
}

/// cliquetable.cleanupFixed(globaldom)
pub fn cleanup_fixed(ms: &MipSolver, globaldom: &mut DomS) {
    let dom = globaldom.cdom();
    // SAFETY: the solver's table and views
    unsafe { cq::highs_rs_clique_cleanup_fixed(table(ms), &dom) };
}

/// cliquetable.runCliqueMerging(globaldom)
pub fn run_clique_merging(ms: &MipSolver, globaldom: &mut DomS) {
    let dom = globaldom.cdom();
    // SAFETY: the solver's table and views
    unsafe { cq::highs_rs_clique_run_merging(table(ms), &dom) };
}

/// cliquetable.rebuild(ncols, postSolveStack, globaldom, orig2reducedcol)
/// isColLinearlyTransformable: of the reduced column
fn col_linearly_transformable(st: &crate::lp_data::lp_presolve::PostsolveStack, col: i32) -> bool {
    st.linearly_transformable[st.orig_col_index[col as usize] as usize] != 0
}

pub fn rebuild_cliques(ms: &MipSolver, ncols: i32, orig2reducedcol: &[i32]) {
    let d = ms.d();
    let gd = d.get_domain();
    let st = &d.postsolve_stack;
    let keep: Vec<u8> = (0..ncols).map(|c| (gd.is_binary(c) && col_linearly_transformable(st, c)) as u8).collect();
    // SAFETY: the solver's table, the arrays
    unsafe {
        cq::highs_rs_clique_rebuild(table(ms), ncols, orig2reducedcol.as_ptr(), orig2reducedcol.len() as i32, keep.as_ptr())
    };
}

// ---- reduced cost fixing ----

/// redcostfixing.propagateRootRedcost(mipsolver)
pub fn propagate_root_redcost(ms: &MipSolver) {
    let d = ms.d();
    let dom = d.get_domain().cdom();
    let cols = d.vecs.integral_cols.as_slice();
    let r: *mut RedcostFixing = &mut *d.redcostfixing;
    // SAFETY: the solver's object and views
    unsafe {
        rcf::highs_rs_redcost_propagate_root(r, &dom, cols.as_ptr(), cols.len() as i32, d.sc.lower_bound, d.sc.upper_limit)
    };
}

/// redcostfixing.addRootRedcost(mipsolver, lpredcost, lpobjective)
pub fn add_root_redcost(ms: &MipSolver, redcost: *const f64, lp_objective: f64) {
    let d = ms.d();
    // the provided domains are only used for the dual proof
    let gd = d.get_domain() as *mut DomS;
    // SAFETY: the solver's global domain, passed as both domains
    unsafe {
        d.get_lp().compute_basic_degenerate_duals(
            d.sc.feastol,
            &mut *gd,
            &mut *gd,
            d.get_conflict_pool(),
            d.get_pseudo_cost(),
            false,
        )
    };
    let dom = d.get_domain();
    let cols = d.vecs.integral_cols.as_slice();
    let r: *mut RedcostFixing = &mut *d.redcostfixing;
    // SAFETY: the solver's object, the LP's reduced costs of numCol
    unsafe {
        rcf::highs_rs_redcost_add_root(
            r,
            ms.num_col(),
            cols.as_ptr(),
            cols.len() as i32,
            dom.col_lower().as_ptr(),
            dom.col_upper().as_ptr(),
            redcost,
            lp_objective,
            d.sc.feastol,
            d.sc.lower_bound,
        )
    };
}

/// The RedcostCtx of propagateRedCost
struct RedcostCtx {
    localdom: *mut DomS,
    globaldom: *mut DomS,
    lp: *const LpS,
    conflictpool: *mut ConflictPoolS,
    pseudocost: *const PscostS,
    upper_limit: f64,
    inds: Vec<i32>,
    vals: Vec<f64>,
}

unsafe extern "C" fn redcost_dual_proof(
    p: *mut c_void,
    inds: *mut *const i32,
    vals: *mut *const f64,
    len: *mut i32,
    rhs: *mut f64,
) -> bool {
    let c = &mut *(p as *mut RedcostCtx);
    let Some((i, v, r)) = (*c.lp).compute_dual_proof(&*c.globaldom, c.upper_limit, false) else { return false };
    c.inds = i;
    c.vals = v;
    *rhs = r;
    *inds = c.inds.as_ptr();
    *vals = c.vals.as_ptr();
    *len = c.inds.len() as i32;
    true
}

unsafe extern "C" fn redcost_reconvergence(p: *mut c_void, domchg: DomChg, inds: *const i32, vals: *const f64, len: i32, rhs: f64) {
    let c = &mut *(p as *mut RedcostCtx);
    (*c.localdom).conflict_analyze_reconvergence(
        domchg,
        crate::ffi::sl(inds, len),
        crate::ffi::sl(vals, len),
        rhs,
        &mut *c.conflictpool,
        &mut *c.globaldom,
        &*c.pseudocost,
    );
}

/// HighsRedcostFixing::propagateRedCost
#[allow(clippy::too_many_arguments)]
pub fn propagate_red_cost(
    ms: &MipSolver,
    localdom: &mut DomS,
    globaldom: &mut DomS,
    lp: &LpS,
    conflictpool: &mut ConflictPoolS,
    pseudocost: &PscostS,
    upper_limit: f64,
) {
    let d = ms.d();
    let mut ctx = RedcostCtx {
        localdom,
        globaldom,
        lp,
        conflictpool,
        pseudocost,
        upper_limit,
        inds: Vec::new(),
        vals: Vec::new(),
    };
    let local = localdom.cdom();
    let global = globaldom.cdom();
    let cols = d.vecs.integral_cols.as_slice();
    let r = CRedcost {
        local: &local,
        global: &global,
        integral_cols: cols.as_ptr(),
        num_integral: cols.len() as i32,
        redcost: lp.col_dual().as_ptr(),
        lp_objective: lp.objective(),
        upper_limit,
        feastol: d.sc.feastol,
        epsilon: d.sc.epsilon,
        pool: conflictpool.rs(),
        ctx: &mut ctx as *mut RedcostCtx as *mut c_void,
        dual_proof: redcost_dual_proof,
        reconvergence: redcost_reconvergence,
    };
    // SAFETY: the views and the context of the call
    unsafe { rcf::highs_rs_redcost_propagate(&r) };
}

// ---- the symmetries ----

/// StabilizerOrbits with its symmetries
pub struct StabS {
    pub orbits: StabilizerOrbits,
    pub symmetries: *const Symmetries,
}

// SAFETY: read-only after construction, shared by the search's nodes
unsafe impl Send for StabS {}
unsafe impl Sync for StabS {}

impl StabS {
    /// symmetries.computeStabilizerOrbits(localdom)
    pub fn compute(symmetries: &Symmetries, localdom: &mut DomS) -> StabS {
        let d = localdom.sym_dom();
        StabS { orbits: symmetries.compute_stabilizer_orbits(&d), symmetries }
    }
    /// orbitalFixing(domain)
    pub fn orbital_fixing(&self, domain: &mut DomS) -> i32 {
        let d = domain.sym_dom();
        // SAFETY: the symmetries outlive the orbits computed from them
        let s = unsafe { &*self.symmetries };
        s.orbital_fixing(&self.orbits.orbit_cols, &self.orbits.orbit_starts, &d)
    }
    /// isStabilized(col)
    pub fn is_stabilized(&self, col: i32) -> bool {
        // SAFETY: as orbital_fixing
        let s = unsafe { &*self.symmetries };
        s.column_position.get(col as usize).copied().unwrap_or(-1) == -1
            || self.orbits.stabilized_cols.binary_search(&col).is_ok()
    }
}

/// symmetries.propagateOrbitopes(domain)
pub fn propagate_orbitopes(symmetries: &Symmetries, domain: &mut DomS) -> i32 {
    let d = domain.sym_dom();
    symmetries.propagate_orbitopes(&d)
}

/// The profiling of a heuristic's clock pair, none if the solver is not
/// timing
pub fn _prof(_: &Prof) {}

#[allow(dead_code)]
fn _cdouble(_: CDouble) {}
#[allow(dead_code)]
fn _cdom(_: CDom) {}
