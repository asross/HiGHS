//! HighsDomain's shell (highs/mip/HighsDomain.h, HighsDomain.cpp,
//! HighsDomainRust.h): the vectors (domain.rs `DomainVecs`), the objective
//! propagation's state, the pools' propagation shells registered with their
//! pools, the infeasibility flags, the redundant rows of probing's lifting,
//! and the view (`CDomain`) that the ported domain code works on, cached
//! and refilled where a vector it points into may move. The calls the
//! domain code makes back (the clique table's and implications' fixings,
//! the pools' ages, a redundant row) are here, and so are the views of the
//! domain that the clique table, implications, symmetries and conflict
//! analysis take.

use super::pools::{ConflictPoolS, CutPoolS};
use super::solver::MipSolver;
use super::tables::{ObjFunc, PscostS};
use super::P;
use crate::mip::clique::CDom;
use crate::mip::conflict::CConflict;
use crate::mip::domain::poolprop_ffi as pp;
use crate::mip::domain::{
    self as dm, CBounds, CConfProp, CCutProp, CDomain, CSlice, ConfPropState, CutPropState, Ctx, DomChg, DomainVecs,
    PrevBound, Ptr, Reason, StdVec, LOWER, UPPER,
};
use crate::mip::objprop::{CObjProp, ObjPropState};
use crate::presolve::symmetry::SymDom;
use crate::util::cdouble::CDouble;
use crate::util::hash_table::HighsHashTable;
use std::ffi::c_void;

/// HighsDomain::CutpoolPropagation: its state (CutPropState) and pool
pub struct CutPropS {
    pub cutpoolindex: i32,
    pub domain: *mut DomS,
    pub cutpool: *mut CutPoolS,
    pub rs: Box<CutPropState>,
}

/// HighsDomain::ConflictPoolPropagation
pub struct ConfPropS {
    pub conflictpoolindex: i32,
    pub domain: *mut DomS,
    pub conflictpool: *mut ConflictPoolS,
    pub rs: Box<ConfPropState>,
}

/// HighsDomain::ObjectivePropagation (active when it has a domain)
pub struct ObjPropS {
    pub active: bool,
    pub objfunc: *const ObjFunc,
    pub cost: *const f64,
    pub rs: Option<Box<ObjPropState>>,
}

impl ObjPropS {
    fn none() -> Self {
        ObjPropS { active: false, objfunc: std::ptr::null(), cost: std::ptr::null(), rs: None }
    }
    fn clone_of(o: &ObjPropS) -> Self {
        ObjPropS { active: o.active, objfunc: o.objfunc, cost: o.cost, rs: o.rs.clone() }
    }
}

/// The cached view (DomainCache)
struct Cache {
    d: CDomain,
    cuts: Vec<CCutProp>,
    conflicts: Vec<CConfProp>,
    valid: bool,
}

/// HighsDomain
pub struct DomS {
    pub v: Box<DomainVecs>,
    pub objprop: ObjPropS,
    pub mipsolver: *mut MipSolver,
    pub cutprops: Vec<Box<CutPropS>>,
    pub confprops: Vec<Box<ConfPropS>>,
    pub infeasible: bool,
    pub infeasible_reason: Reason,
    pub infeasible_pos: i32,
    cache: Option<Box<Cache>>,
    pub redundant_rows: HighsHashTable<i32>,
    pub record_redundant_rows: bool,
}

// SAFETY: as the solver's
unsafe impl Send for DomS {}

fn vecs_new(ncol: i32, lower: &[f64], upper: &[f64]) -> Box<DomainVecs> {
    // SAFETY: the model's bounds of ncol columns
    unsafe { Box::from_raw(dm::highs_rs_domain_vecs_new(ncol, lower.as_ptr(), upper.as_ptr())) }
}

impl DomS {
    pub fn ms<'a>(&self) -> &'a MipSolver {
        // SAFETY: the solver outlives its domains
        unsafe { &*self.mipsolver }
    }

    /// HighsDomain(mipsolver)
    pub fn new(ms: &MipSolver) -> Box<DomS> {
        let m = ms.model();
        Box::new(DomS {
            v: vecs_new(m.num_col, &m.col_lower, &m.col_upper),
            objprop: ObjPropS::none(),
            mipsolver: ms as *const MipSolver as *mut MipSolver,
            cutprops: Vec::new(),
            confprops: Vec::new(),
            infeasible: false,
            infeasible_reason: Reason::UNSPECIFIED,
            infeasible_pos: 0,
            cache: None,
            redundant_rows: HighsHashTable::new(),
            record_redundant_rows: false,
        })
    }

    /// HighsDomain(other): the pools' propagations copied (and registered,
    /// unless a copy of the global pool's under the parallel lock)
    pub fn copy(other: &DomS) -> Box<DomS> {
        let mut d = Box::new(DomS {
            v: Box::new((*other.v).clone()),
            objprop: ObjPropS::clone_of(&other.objprop),
            mipsolver: other.mipsolver,
            cutprops: Vec::new(),
            confprops: Vec::new(),
            infeasible: other.infeasible,
            infeasible_reason: other.infeasible_reason,
            infeasible_pos: other.infeasible_pos,
            cache: None,
            redundant_rows: HighsHashTable::new(),
            record_redundant_rows: false,
        });
        let dp: *mut DomS = &mut *d;
        for cp in &other.cutprops {
            d.cutprops.push(CutPropS::copy(cp, dp));
        }
        for cp in &other.confprops {
            d.confprops.push(ConfPropS::copy(cp, dp));
        }
        d
    }

    /// operator=: the vectors assigned, the pools' propagations assigned
    /// as std::deque does (element-wise, then copies or destruction)
    pub fn assign(&mut self, other: &DomS) {
        if std::ptr::eq(self, other) {
            return;
        }
        // SAFETY: both live
        unsafe { dm::highs_rs_domain_vecs_assign(&mut *self.v, &*other.v) };
        self.objprop = ObjPropS::clone_of(&other.objprop);
        self.mipsolver = other.mipsolver;
        let dp: *mut DomS = self;
        let n = other.cutprops.len();
        for i in 0..n.min(self.cutprops.len()) {
            self.cutprops[i].assign(&other.cutprops[i]);
        }
        for i in self.cutprops.len()..n {
            self.cutprops.push(CutPropS::copy(&other.cutprops[i], dp));
        }
        self.cutprops.truncate(n);
        let n = other.confprops.len();
        for i in 0..n.min(self.confprops.len()) {
            self.confprops[i].assign(&other.confprops[i]);
        }
        for i in self.confprops.len()..n {
            self.confprops.push(ConfPropS::copy(&other.confprops[i], dp));
        }
        self.confprops.truncate(n);
        self.infeasible = other.infeasible;
        self.infeasible_reason = other.infeasible_reason;
        self.invalidate();
        for cp in &mut self.cutprops {
            cp.domain = dp;
        }
        for cp in &mut self.confprops {
            cp.domain = dp;
        }
    }

    pub fn invalidate(&mut self) {
        if let Some(c) = &mut self.cache {
            c.valid = false;
        }
    }

    /// addCutpool
    pub fn add_cutpool(&mut self, cutpool: &mut CutPoolS) {
        self.invalidate();
        let index = self.cutprops.len() as i32;
        let dp: *mut DomS = self;
        self.cutprops.push(CutPropS::new(index, dp, cutpool));
    }

    /// addConflictPool
    pub fn add_conflict_pool(&mut self, pool: &mut ConflictPoolS) {
        self.invalidate();
        let index = self.confprops.len() as i32;
        let dp: *mut DomS = self;
        let ncol = self.ms().num_col();
        self.confprops.push(ConfPropS::new(index, dp, pool, ncol));
    }

    /// clearPoolPropagation
    pub fn clear_pool_propagation(&mut self) {
        self.invalidate();
        self.cutprops.clear();
        self.confprops.clear();
    }

    /// setupObjectivePropagation
    pub fn setup_objective_propagation(&mut self) {
        self.invalidate();
        let ms = self.ms();
        let f: *const ObjFunc = &ms.d().objective_function;
        let cost = ms.model().col_cost.as_ptr();
        let b = self.bounds();
        // SAFETY: the objective function's vectors and the bounds, live
        let f_ = unsafe { &*f };
        let starts = f_.clique_partition_start();
        let rs = unsafe {
            Box::from_raw(crate::mip::objprop::highs_rs_objprop_new(
                &b,
                cost,
                ms.num_col(),
                f_.objective_nonzeros().as_ptr(),
                f_.objective_nonzeros().len() as i32,
                starts.as_ptr(),
                f_.num_clique_partitions() + 1,
                f_.objective_vals().as_ptr(),
                f_.objective_vals().len() as i32,
            ))
        };
        self.objprop = ObjPropS { active: true, objfunc: f, cost, rs: Some(rs) };
    }

    // ---- the view ----

    /// DomainAccess::bounds: the data of the const methods
    pub fn bounds(&self) -> CBounds {
        let ms = self.ms();
        let sc = &ms.d().sc;
        CBounds {
            feastol: sc.feastol,
            epsilon: sc.epsilon,
            col_lower: CSlice::of(self.v.col_lower.as_slice()),
            col_upper: CSlice::of(self.v.col_upper.as_slice()),
            integrality: CSlice::of(&ms.model().integrality),
            col_lower_pos: CSlice::of(self.v.col_lower_pos.as_slice()),
            col_upper_pos: CSlice::of(self.v.col_upper_pos.as_slice()),
            prevboundval: CSlice::of(self.v.prevboundval.as_slice()),
            infeasible: self.infeasible,
            infeasible_pos: self.infeasible_pos,
        }
    }

    fn fill_cut_prop(cp: &mut CutPropS) -> CCutProp {
        let mut m = std::mem::MaybeUninit::<crate::mip::cutpool::CMatrixView>::uninit();
        // SAFETY: the live pool; the view is written whole
        let m = unsafe {
            crate::mip::cutpool::ffi::highs_rs_cutpool_view((*cp.cutpool).rs(), m.as_mut_ptr());
            m.assume_init()
        };
        fn cs<T>(p: *const T, n: i32) -> CSlice<T> {
            // SAFETY: the pool's arrays of these lengths
            CSlice::of(unsafe { crate::ffi::sl(p, n) })
        }
        let s = &mut *cp.rs;
        CCutProp {
            cutpoolindex: cp.cutpoolindex,
            cutpool: cp.cutpool as *const c_void,
            activitycuts: CSlice::of(s.activitycuts.as_slice()),
            activitycutsinf: CSlice::of(s.activitycutsinf.as_slice()),
            propagatecutflags: CSlice::of(s.propagatecutflags.as_slice()),
            capacity_threshold: CSlice::of(s.capacity_threshold.as_slice()),
            propagatecutinds: Ptr(&mut s.propagatecutinds),
            ar_range: cs(m.ar_range, m.num_rows),
            ar_index: cs(m.ar_index, m.num_nz),
            ar_value: cs(m.ar_value, m.num_nz),
            ar_rowindex: cs(m.ar_rowindex, m.num_nz),
            next_pos: cs(m.next_pos, m.num_nz),
            next_neg: cs(m.next_neg, m.num_nz),
            head_pos: cs(m.head_pos, m.num_cols),
            head_neg: cs(m.head_neg, m.num_cols),
            rhs: cs(m.rhs, m.num_rhs),
        }
    }

    fn fill_conf_prop(cp: &mut ConfPropS) -> CConfProp {
        let s = &mut *cp.rs;
        CConfProp {
            col_lower_watched: CSlice::of(s.col_lower_watched.as_slice()),
            col_upper_watched: CSlice::of(s.col_upper_watched.as_slice()),
            watched: CSlice::of(s.watched.as_slice()),
            conflict_flag: CSlice::of(s.conflict_flag.as_slice()),
            propagate_conflict_inds: Ptr(&mut s.propagate_conflict_inds),
            // SAFETY: the live pool
            pool: unsafe { (*cp.conflictpool).rs() },
        }
    }

    fn fill(&mut self) {
        let ms = self.ms();
        let d = ms.d();
        let model = ms.model();
        let cuts: Vec<CCutProp> = self.cutprops.iter_mut().map(|cp| Self::fill_cut_prop(cp)).collect();
        let conflicts: Vec<CConfProp> = self.confprops.iter_mut().map(|cp| Self::fill_conf_prop(cp)).collect();
        let objprop = match (&mut self.objprop.rs, self.objprop.active) {
            (Some(st), true) => {
                // SAFETY: the objective function lives with the solver
                let f = unsafe { &*self.objprop.objfunc };
                CObjProp {
                    active: true,
                    cost: CSlice::of(&model.col_cost),
                    obj_nonzeros: CSlice::of(f.objective_nonzeros()),
                    partition_starts: CSlice::of(f.clique_partition_start()),
                    col_to_partition: CSlice::of(f.col_to_partition()),
                    num_binaries: f.num_binaries(),
                    contributions: CSlice::of(st.contributions.as_slice()),
                    partition_sets: CSlice::of(st.partition_sets.as_slice()),
                    objective_lower: &mut st.objective_lower,
                    num_inf_obj_lower: &mut st.num_inf_obj_lower,
                    capacity_threshold: &mut st.capacity_threshold,
                    is_propagated: &mut st.is_propagated,
                    obj_vals: CSlice::of(f.objective_vals()),
                    clique_data: CSlice::of(st.clique_data.as_slice()),
                    cons_buffer: CSlice::of(st.cons_buffer.as_slice()),
                }
            }
            _ => CObjProp::inactive(),
        };
        let v = &mut *self.v;
        let cd = CDomain {
            feastol: &d.sc.feastol,
            epsilon: &d.sc.epsilon,
            upper_limit: &d.sc.upper_limit,
            a_start: CSlice::of(&model.a.start),
            a_index: CSlice::of(&model.a.index),
            a_value: CSlice::of(&model.a.value),
            ar_start: CSlice::of(d.vecs.ar_start.as_slice()),
            ar_index: CSlice::of(d.vecs.ar_index.as_slice()),
            ar_value: CSlice::of(d.vecs.ar_value.as_slice()),
            row_lower: CSlice::of(&model.row_lower),
            row_upper: CSlice::of(&model.row_upper),
            integrality: CSlice::of(&model.integrality),
            col_lower: CSlice::of(v.col_lower.as_slice()),
            col_upper: CSlice::of(v.col_upper.as_slice()),
            activitymin: CSlice::of(v.activitymin.as_slice()),
            activitymax: CSlice::of(v.activitymax.as_slice()),
            activitymininf: CSlice::of(v.activitymininf.as_slice()),
            activitymaxinf: CSlice::of(v.activitymaxinf.as_slice()),
            capacity_threshold: CSlice::of(v.capacity_threshold.as_slice()),
            propagateflags: CSlice::of(v.propagateflags.as_slice()),
            col_lower_pos: CSlice::of(v.col_lower_pos.as_slice()),
            col_upper_pos: CSlice::of(v.col_upper_pos.as_slice()),
            changedcolsflags: CSlice::of(v.changedcolsflags.as_slice()),
            propagateinds: Ptr(&mut v.propagateinds),
            changedcols: Ptr(&mut v.changedcols),
            branchpos: Ptr(&mut v.branch_pos),
            domchgstack: Ptr(&mut v.domchgstack),
            domchgreason: Ptr(&mut v.domchgreason),
            prevboundval: Ptr(&mut v.prevboundval),
            scratch_inds: Ptr(&mut v.scratch_inds),
            scratch_bounds: Ptr(&mut v.scratch_bounds),
            scratch_counts: Ptr(&mut v.scratch_counts),
            infeasible: Ptr(&mut self.infeasible),
            infeasible_reason: Ptr(&mut self.infeasible_reason),
            infeasible_pos: Ptr(&mut self.infeasible_pos),
            record_redundant_rows: Ptr(&mut self.record_redundant_rows),
            cutpools: CSlice::of(&[]),
            conflictpools: CSlice::of(&[]),
            objprop,
            dom: self as *mut DomS as *mut c_void,
            implications: cb_implications,
            redundant_row: cb_redundant_row,
            cut_reset_age: cb_cut_reset_age,
            conflict_reset_age: cb_conflict_reset_age,
            reserve_i32: dm::highs_rs_reserve_i32,
            reserve_domchg: dm::highs_rs_reserve_domchg,
            reserve_reason: dm::highs_rs_reserve_reason,
            reserve_prev: dm::highs_rs_reserve_prev,
            reserve_pair: dm::highs_rs_reserve_pair,
        };
        let c = match &mut self.cache {
            Some(c) => {
                c.d = cd;
                c.cuts = cuts;
                c.conflicts = conflicts;
                c
            }
            None => self.cache.insert(Box::new(Cache { d: cd, cuts, conflicts, valid: false })),
        };
        c.d.cutpools = CSlice::of(&c.cuts);
        c.d.conflictpools = CSlice::of(&c.conflicts);
        c.valid = true;
    }

    /// DomainAccess::view: the cached view, refilled if it was invalidated
    pub fn view(&mut self) -> *const CDomain {
        if !self.cache.as_ref().is_some_and(|c| c.valid) {
            self.fill();
        }
        &self.cache.as_ref().unwrap().d
    }

    /// cutPoolChanged: a pool's part of a valid view refreshed
    fn cut_pool_changed(&mut self, pool: usize) {
        if self.cache.as_ref().is_some_and(|c| c.valid) {
            let cp = Self::fill_cut_prop(&mut self.cutprops[pool]);
            self.cache.as_mut().unwrap().cuts[pool] = cp;
        }
    }

    fn conflict_pool_changed(&mut self, pool: usize) {
        if self.cache.as_ref().is_some_and(|c| c.valid) {
            let cp = Self::fill_conf_prop(&mut self.confprops[pool]);
            self.cache.as_mut().unwrap().conflicts[pool] = cp;
        }
    }

    // ---- the operations ----

    /// computeRowActivities
    pub fn compute_row_activities(&mut self) {
        self.invalidate();
        let nrow = self.ms().num_row();
        // SAFETY: the domain's vectors
        unsafe { dm::highs_rs_domain_vecs_size_rows(&mut *self.v, nrow) };
        let v = self.view();
        // SAFETY: the domain's view
        unsafe { dm::ffi::highs_rs_domain_compute_row_activities(v) };
    }

    /// changeBound(boundchg, reason)
    pub fn change_bound(&mut self, chg: DomChg, reason: Reason) {
        let v = self.view();
        // SAFETY: the domain's view (re-entered by the implications)
        unsafe { Ctx::new(v).change_bound(chg, reason) };
    }

    /// fixCol(col, val, reason)
    pub fn fix_col(&mut self, col: i32, val: f64, reason: Reason) {
        let c = col as usize;
        if self.v.col_lower[c] < val {
            self.change_bound(DomChg { boundval: val, column: col, boundtype: LOWER }, reason);
            if !self.infeasible {
                self.propagate();
            }
        }
        if !self.infeasible && self.v.col_upper[c] > val {
            self.change_bound(DomChg { boundval: val, column: col, boundtype: UPPER }, reason);
        }
    }

    /// setDomainChangeStack(stack[, branchingPositions])
    pub fn set_domain_change_stack(&mut self, stack: &[DomChg], branching: Option<&[i32]>) {
        let v = self.view();
        // SAFETY: the domain's view
        unsafe { Ctx::new(v).set_domain_change_stack(stack, branching) };
    }

    pub fn backtrack_to_global(&mut self) {
        let v = self.view();
        // SAFETY: the domain's view
        unsafe { dm::ffi::highs_rs_domain_backtrack(v, true) };
    }

    pub fn backtrack(&mut self) -> DomChg {
        let v = self.view();
        // SAFETY: the domain's view
        unsafe { dm::ffi::highs_rs_domain_backtrack(v, false) }
    }

    pub fn propagate(&mut self) -> bool {
        let v = self.view();
        // SAFETY: the domain's view
        unsafe { Ctx::new(v).propagate() }
    }

    /// getColLowerPos / getColUpperPos: (bound, pos)
    pub fn get_col_pos(&self, col: i32, stackpos: i32, upper: bool) -> (f64, i32) {
        let c = col as usize;
        let prev = self.v.prevboundval.as_slice();
        let (mut b, mut pos) =
            if upper { (self.v.col_upper[c], self.v.col_upper_pos[c]) } else { (self.v.col_lower[c], self.v.col_lower_pos[c]) };
        while pos > stackpos || (pos != -1 && prev[pos as usize].val == b) {
            b = prev[pos as usize].val;
            pos = prev[pos as usize].pos;
        }
        (b, pos)
    }

    fn conflict(&mut self, global: &mut DomS, pool: &mut ConflictPoolS, pseudocost: &PscostS) -> CConflict {
        let ms = self.ms();
        let d = ms.d();
        CConflict {
            local: self.view(),
            global: global.view(),
            pool: pool as *mut ConflictPoolS as *mut c_void,
            pseudocost: pseudocost.rs,
            nodequeue: &*d.nodequeue,
            num_integral: d.vecs.integral_cols.len() as i32,
            add_cut: cb_conflict_add_cut,
        }
    }

    /// conflictAnalysis(conflictPool, globaldom, pseudocost)
    pub fn conflict_analysis(&mut self, pool: &mut ConflictPoolS, global: &mut DomS, pseudocost: &PscostS) {
        if std::ptr::eq(self, global) || global.infeasible || !self.infeasible {
            return;
        }
        global.propagate();
        if global.infeasible {
            return;
        }
        let c = self.conflict(global, pool, pseudocost);
        // SAFETY: the views of the two live domains
        unsafe { crate::mip::conflict::ffi::highs_rs_conflict_analysis(&c) };
    }

    /// conflictAnalysis(proof, conflictPool, globaldom, pseudocost)
    pub fn conflict_analysis_proof(
        &mut self,
        inds: &[i32],
        vals: &[f64],
        rhs: f64,
        pool: &mut ConflictPoolS,
        global: &mut DomS,
        pseudocost: &PscostS,
    ) {
        if std::ptr::eq(self, global) || global.infeasible {
            return;
        }
        global.propagate();
        if global.infeasible {
            return;
        }
        let c = self.conflict(global, pool, pseudocost);
        // SAFETY: as conflict_analysis
        unsafe {
            crate::mip::conflict::ffi::highs_rs_conflict_analysis_proof(
                &c,
                inds.as_ptr(),
                vals.as_ptr(),
                inds.len() as i32,
                rhs,
            )
        };
    }

    /// conflictAnalyzeReconvergence
    #[allow(clippy::too_many_arguments)]
    pub fn conflict_analyze_reconvergence(
        &mut self,
        domchg: DomChg,
        inds: &[i32],
        vals: &[f64],
        rhs: f64,
        pool: &mut ConflictPoolS,
        global: &mut DomS,
        pseudocost: &PscostS,
    ) {
        if std::ptr::eq(self, global) || global.infeasible {
            return;
        }
        global.propagate();
        if global.infeasible {
            return;
        }
        let c = self.conflict(global, pool, pseudocost);
        // SAFETY: as conflict_analysis
        unsafe {
            crate::mip::conflict::ffi::highs_rs_conflict_reconvergence(
                &c,
                domchg,
                inds.as_ptr(),
                vals.as_ptr(),
                inds.len() as i32,
                rhs,
            )
        };
    }

    /// tightenCoefficients
    pub fn tighten_coefficients(&self, inds: &[i32], vals: &mut [f64], rhs: &mut f64) {
        let b = self.bounds();
        // SAFETY: the domain's bounds
        unsafe {
            dm::ffi::highs_rs_domain_tighten_coefficients(&b, inds.as_ptr(), vals.as_mut_ptr(), inds.len() as i32, rhs)
        };
    }

    /// computeMinActivity / computeMaxActivity of a row (ninf, activity)
    pub fn compute_activity(&self, index: &[i32], value: &[f64], max: bool) -> (i32, CDouble) {
        let b = self.bounds();
        let (mut n, mut a) = (0, CDouble::from(0.0));
        // SAFETY: the domain's bounds
        unsafe {
            dm::ffi::highs_rs_domain_compute_activity(
                &b,
                index.as_ptr(),
                value.as_ptr(),
                index.len() as i32,
                max,
                &mut n,
                &mut a,
            )
        };
        (n, a)
    }

    /// getMinCutActivity
    pub fn get_min_cut_activity(&self, cutpool: *const CutPoolS, cut: i32) -> f64 {
        for cp in &self.cutprops {
            if std::ptr::eq(cp.cutpool, cutpool) {
                let c = cut as usize;
                let s = &cp.rs;
                return if c < s.propagatecutflags.len() && (s.propagatecutflags[c] & 2) == 0 && s.activitycutsinf[c] == 0
                {
                    s.activitycuts[c].to_f64()
                } else {
                    -f64::INFINITY
                };
            }
        }
        -f64::INFINITY
    }

    pub fn get_min_activity(&self, row: i32) -> f64 {
        let r = row as usize;
        if self.v.activitymininf[r] == 0 {
            self.v.activitymin[r].to_f64()
        } else {
            -f64::INFINITY
        }
    }
    pub fn get_max_activity(&self, row: i32) -> f64 {
        let r = row as usize;
        if self.v.activitymaxinf[r] == 0 {
            self.v.activitymax[r].to_f64()
        } else {
            f64::INFINITY
        }
    }

    /// flip(domchg)
    pub fn flip(&self, domchg: &DomChg) -> DomChg {
        let ms = self.ms();
        let feastol = ms.d().sc.feastol;
        if domchg.boundtype == LOWER {
            let mut f = DomChg { boundval: domchg.boundval - feastol, column: domchg.column, boundtype: UPPER };
            if ms.is_col_integral(domchg.column) {
                f.boundval = f.boundval.floor();
            }
            f
        } else {
            let mut f = DomChg { boundval: domchg.boundval + feastol, column: domchg.column, boundtype: LOWER };
            if ms.is_col_integral(domchg.column) {
                f.boundval = f.boundval.ceil();
            }
            f
        }
    }

    pub fn feastol(&self) -> f64 {
        self.ms().d().sc.feastol
    }

    pub fn mark_infeasible(&mut self, reason: Reason) {
        self.infeasible = true;
        self.infeasible_pos = self.v.domchgstack.len() as i32;
        self.infeasible_reason = reason;
    }

    pub fn clear_changed_cols(&mut self) {
        let v = &mut *self.v;
        for &i in v.changedcols.as_slice() {
            v.changedcolsflags.as_mut_slice()[i as usize] = 0;
        }
        v.changedcols.clear();
    }

    /// clearChangedCols(start)
    pub fn clear_changed_cols_from(&mut self, start: usize) {
        let v = &mut *self.v;
        let n = v.changedcols.len();
        for i in start..n {
            let c = v.changedcols[i] as usize;
            v.changedcolsflags.as_mut_slice()[c] = 0;
        }
        v.changedcols.truncate(start);
    }

    /// removeContinuousChangedCols
    pub fn remove_continuous_changed_cols(&mut self) {
        let ms = self.ms();
        let v = &mut *self.v;
        let cols = v.changedcols.as_mut_slice();
        let flags = v.changedcolsflags.as_mut_slice();
        for &i in cols.iter() {
            flags[i as usize] = ms.is_col_integral(i) as u8;
        }
        let mut k = 0;
        for j in 0..cols.len() {
            if flags[cols[j] as usize] != 0 {
                cols[k] = cols[j];
                k += 1;
            }
        }
        v.changedcols.truncate(k);
    }

    pub fn changed_cols(&self) -> &[i32] {
        self.v.changedcols.as_slice()
    }
    pub fn stack(&self) -> &[DomChg] {
        self.v.domchgstack.as_slice()
    }
    pub fn reasons(&self) -> &[Reason] {
        self.v.domchgreason.as_slice()
    }
    pub fn prev_bounds(&self) -> &[PrevBound] {
        self.v.prevboundval.as_slice()
    }
    pub fn branching_positions(&self) -> &[i32] {
        self.v.branch_pos.as_slice()
    }
    pub fn col_lower(&self) -> &[f64] {
        self.v.col_lower.as_slice()
    }
    pub fn col_upper(&self) -> &[f64] {
        self.v.col_upper.as_slice()
    }

    /// getObjectiveLowerBound
    pub fn get_objective_lower_bound(&self) -> f64 {
        match (&self.objprop.rs, self.objprop.active) {
            (Some(s), true) => {
                if s.num_inf_obj_lower == 0 {
                    s.objective_lower.to_f64()
                } else {
                    -f64::INFINITY
                }
            }
            _ => -f64::INFINITY,
        }
    }

    /// getCutoffConstraint: (vals, inds, rhs), valid until the domain
    /// changes
    pub fn get_cutoff_constraint(&mut self) -> (*const f64, *const i32, i32, f64) {
        let stacksize = self.v.domchgstack.len() as i32;
        let v = self.view();
        let (mut vals, mut inds, mut len, mut rhs) = (std::ptr::null(), std::ptr::null(), 0, 0.0);
        // SAFETY: the domain's view with an active objective propagation
        unsafe {
            crate::mip::objprop::highs_rs_domain_obj_propagation_constraint(
                v, stacksize, -1, &mut vals, &mut inds, &mut len, &mut rhs,
            )
        };
        (vals, inds, len, rhs)
    }

    /// getReducedDomainChangeStack: (stack, branching positions)
    pub fn get_reduced_domain_change_stack(&self) -> (Vec<DomChg>, Vec<i32>) {
        let v = &*self.v;
        let stack = v.domchgstack.as_slice();
        let (lpos, upos) = (v.col_lower_pos.as_slice(), v.col_upper_pos.as_slice());
        let reasons = v.domchgreason.as_slice();
        let prev = v.prevboundval.as_slice();
        let mut reduced = Vec::with_capacity(stack.len());
        let mut branching = Vec::with_capacity(v.branch_pos.len());
        for i in 0..stack.len() {
            let c = stack[i].column as usize;
            if (stack[i].boundtype == LOWER && lpos[c] != i as i32) || (stack[i].boundtype == UPPER && upos[c] != i as i32) {
                continue;
            }
            if reasons[i].kind == dm::REASON_BRANCHING {
                branching.push(reduced.len() as i32);
            } else {
                let mut k = i;
                while prev[k].pos != -1 {
                    k = prev[k].pos as usize;
                    if reasons[k].kind == dm::REASON_BRANCHING {
                        branching.push(reduced.len() as i32);
                        break;
                    }
                }
            }
            reduced.push(stack[i]);
        }
        reduced.shrink_to_fit();
        (reduced, branching)
    }

    pub fn is_binary(&self, col: i32) -> bool {
        let c = col as usize;
        self.ms().is_col_integral(col) && self.v.col_lower[c] == 0.0 && self.v.col_upper[c] == 1.0
    }
    pub fn is_global_binary(&self, col: i32) -> bool {
        let ms = self.ms();
        let c = col as usize;
        ms.is_col_integral(col) && ms.model().col_lower[c] == 0.0 && ms.model().col_upper[c] == 1.0
    }
    pub fn is_fixed(&self, col: i32) -> bool {
        self.v.col_lower[col as usize] == self.v.col_upper[col as usize]
    }

    /// getRedundantRowValue
    pub fn get_redundant_row_value(&self, row: i32) -> f64 {
        let m = self.ms().model();
        let r = row as usize;
        if m.row_lower[r] != -f64::INFINITY {
            (self.v.activitymin[r] - m.row_lower[r]).to_f64()
        } else {
            (self.v.activitymax[r] - m.row_upper[r]).to_f64()
        }
    }

    // ---- the views other parts take ----

    /// highs_rs::cliqueDom
    pub fn cdom(&mut self) -> CDom {
        let ms = self.ms();
        CDom {
            ctx: self as *mut DomS as *mut c_void,
            col_lower: self.v.col_lower.as_slice().as_ptr(),
            col_upper: self.v.col_upper.as_slice().as_ptr(),
            integrality: ms.model().integrality.as_ptr(),
            num_col: self.v.col_lower.len() as i32,
            num_nonzero: ms.num_nonzero(),
            feastol: ms.d().sc.feastol,
            infeasible: cdom_infeasible,
            change_bound: cdom_change_bound,
            fix_col: cdom_fix_col,
            propagate: cdom_propagate,
            domchg_stack: cdom_stack,
        }
    }

    /// SymmetryAccess::symDom
    pub fn sym_dom(&mut self) -> SymDom {
        let ms = self.ms();
        let (lo, up) = (ms.model().col_lower.as_ptr(), ms.model().col_upper.as_ptr());
        SymDom {
            dom: self.cdom(),
            model_lower: lo,
            model_upper: up,
            branch_pos: self.v.branch_pos.as_slice().as_ptr(),
            num_branch_pos: self.v.branch_pos.len() as i32,
            mark_infeasible: sym_mark_infeasible,
        }
    }
}

// ---- the domain code's calls back ----

fn dom<'a>(p: *mut c_void) -> &'a mut DomS {
    // SAFETY: the domain the callback was given
    unsafe { &mut *(p as *mut DomS) }
}

/// the fixings implied by fixing binary col to val
unsafe extern "C" fn cb_implications(p: *mut c_void, col: i32, val: i32) {
    let d = dom(p);
    let md = (*d.mipsolver).d();
    let cd = d.cdom();
    crate::mip::clique_ffi::highs_rs_clique_add_implications(&*md.cliquetable, &cd, col, val);
    if !d.infeasible {
        let cd = d.cdom();
        crate::mip::implications_ffi::highs_rs_implics_apply(md.implications.rs(), &cd, col, val);
    }
}

unsafe extern "C" fn cb_redundant_row(p: *mut c_void, row: i32) {
    dom(p).redundant_rows.insert(row, ());
}

unsafe extern "C" fn cb_cut_reset_age(p: *mut c_void, pool: i32, cut: i32) {
    let d = dom(p);
    let lock = (*d.mipsolver).d().parallel_lock_active();
    (*d.cutprops[pool as usize].cutpool).reset_age(cut, lock);
}

unsafe extern "C" fn cb_conflict_reset_age(p: *mut c_void, pool: i32, conflict: i32) {
    let d = dom(p);
    (*d.confprops[pool as usize].conflictpool).reset_age(conflict);
}

/// Conflict::addCut: the frontier's entries as a conflict (or a
/// reconvergence cut), then the views refilled where they moved
unsafe extern "C" fn cb_conflict_add_cut(c: *const CConflict, entries: *const DomChg, len: i32, domchg: *const DomChg) {
    let c = &*c;
    let local = dom((*c.local).dom);
    let global = dom((*c.global).dom);
    let pool = &mut *(c.pool as *mut ConflictPoolS);
    let e = crate::ffi::sl(entries, len);
    if domchg.is_null() {
        pool.add_conflict_cut(local, e);
    } else {
        pool.add_reconvergence_cut(local, e, &*domchg);
    }
    local.view();
    global.view();
}

unsafe extern "C" fn cdom_infeasible(p: *mut c_void) -> bool {
    dom(p).infeasible
}
unsafe extern "C" fn cdom_change_bound(p: *mut c_void, t: i32, col: i32, val: f64, rt: i32, ri: i32) {
    dom(p).change_bound(DomChg { boundval: val, column: col, boundtype: t }, Reason { kind: rt, index: ri });
}
unsafe extern "C" fn cdom_fix_col(p: *mut c_void, col: i32, val: f64) {
    dom(p).fix_col(col, val, Reason::UNSPECIFIED);
}
unsafe extern "C" fn cdom_propagate(p: *mut c_void) {
    dom(p).propagate();
}
unsafe extern "C" fn cdom_stack(p: *mut c_void, len: *mut i32) -> *const crate::mip::clique::DomChg {
    let s = dom(p).stack();
    *len = s.len() as i32;
    s.as_ptr() as *const crate::mip::clique::DomChg
}
unsafe extern "C" fn sym_mark_infeasible(p: *mut c_void) {
    dom(p).mark_infeasible(Reason::UNSPECIFIED);
}

// ---- the pools' propagation shells ----

impl CutPropS {
    /// CutpoolPropagation(index, domain, cutpool): registered with the pool
    fn new(index: i32, domain: *mut DomS, cutpool: &mut CutPoolS) -> Box<CutPropS> {
        // SAFETY: a new state
        let rs = unsafe { Box::from_raw(pp::highs_rs_cutprop_new()) };
        let mut cp = Box::new(CutPropS { cutpoolindex: index, domain, cutpool, rs });
        cutpool.add_propagation_domain(&mut *cp);
        cp
    }

    /// The copy constructor
    fn copy(other: &CutPropS, domain: *mut DomS) -> Box<CutPropS> {
        let mut cp = Box::new(CutPropS {
            cutpoolindex: other.cutpoolindex,
            domain: other.domain,
            cutpool: other.cutpool,
            rs: Box::new((*other.rs).clone()),
        });
        // SAFETY: the domain the copy belongs to (other's while copying, as
        // the C++, which sets it afterwards) and the live pool
        unsafe {
            let ms = &*(*other.domain).mipsolver;
            let d = ms.d();
            if !d.parallel_lock_active() || !std::ptr::eq(cp.cutpool, d.get_cut_pool()) {
                (*cp.cutpool).add_propagation_domain(&mut *cp);
            }
        }
        cp.domain = domain;
        cp
    }

    /// operator=
    fn assign(&mut self, other: &CutPropS) {
        if std::ptr::eq(self, other) {
            return;
        }
        // SAFETY: the live pools
        unsafe {
            if !self.cutpool.is_null() {
                (*self.cutpool).remove_propagation_domain(self);
            }
            self.cutpoolindex = other.cutpoolindex;
            self.domain = other.domain;
            self.cutpool = other.cutpool;
            pp::highs_rs_cutprop_assign(&mut *self.rs, &*other.rs);
            if !self.cutpool.is_null() {
                (*self.cutpool).add_propagation_domain(self);
            }
        }
    }

    /// cutAdded
    pub fn cut_added(&mut self, cut: i32, propagate: bool) {
        // SAFETY: the live domain and pool
        let d = unsafe { &mut *self.domain };
        let global = std::ptr::eq(d, d.ms().d().get_domain());
        if propagate || global {
            let b = d.bounds();
            // SAFETY: as above
            unsafe {
                pp::highs_rs_cutprop_cut_added(&mut *self.rs, (*self.cutpool).rs(), cut, &b, propagate, global)
            };
        }
        d.cut_pool_changed(self.cutpoolindex as usize);
    }

    /// cutDeleted
    pub fn cut_deleted(&mut self, cut: i32, deleted_only_for_propagation: bool) {
        // SAFETY: the live domain
        let d = unsafe { &*self.domain };
        let keep = deleted_only_for_propagation && std::ptr::eq(d, d.ms().d().get_domain());
        // SAFETY: the state
        unsafe { pp::highs_rs_cutprop_cut_deleted(&mut *self.rs, cut, keep) };
    }
}

impl Drop for CutPropS {
    fn drop(&mut self) {
        // SAFETY: the pool outlives the domains propagating it
        unsafe { (*self.cutpool).remove_propagation_domain(self) };
    }
}

impl ConfPropS {
    fn new(index: i32, domain: *mut DomS, pool: &mut ConflictPoolS, ncol: i32) -> Box<ConfPropS> {
        // SAFETY: a new state
        let rs = unsafe { Box::from_raw(pp::highs_rs_confprop_new(ncol)) };
        let mut cp = Box::new(ConfPropS { conflictpoolindex: index, domain, conflictpool: pool, rs });
        pool.add_propagation_domain(&mut *cp);
        cp
    }

    fn copy(other: &ConfPropS, domain: *mut DomS) -> Box<ConfPropS> {
        let mut cp = Box::new(ConfPropS {
            conflictpoolindex: other.conflictpoolindex,
            domain: other.domain,
            conflictpool: other.conflictpool,
            rs: Box::new((*other.rs).clone()),
        });
        // SAFETY: as CutPropS::copy
        unsafe {
            let ms = &*(*other.domain).mipsolver;
            let d = ms.d();
            if !d.parallel_lock_active() || !std::ptr::eq(cp.conflictpool, d.get_conflict_pool()) {
                (*cp.conflictpool).add_propagation_domain(&mut *cp);
            }
        }
        cp.domain = domain;
        cp
    }

    fn assign(&mut self, other: &ConfPropS) {
        if std::ptr::eq(self, other) {
            return;
        }
        // SAFETY: the live pools
        unsafe {
            if !self.conflictpool.is_null() {
                (*self.conflictpool).remove_propagation_domain(self);
            }
            self.conflictpoolindex = other.conflictpoolindex;
            self.domain = other.domain;
            self.conflictpool = other.conflictpool;
            pp::highs_rs_confprop_assign(&mut *self.rs, &*other.rs);
            if !self.conflictpool.is_null() {
                (*self.conflictpool).add_propagation_domain(self);
            }
        }
    }

    pub fn conflict_deleted(&mut self, conflict: i32) {
        // SAFETY: the state
        unsafe { pp::highs_rs_confprop_conflict_deleted(&mut *self.rs, conflict) };
    }

    pub fn conflict_added(&mut self, conflict: i32) {
        // SAFETY: the live domain and pool
        let d = unsafe { &mut *self.domain };
        let b = d.bounds();
        // SAFETY: as above
        unsafe { pp::highs_rs_confprop_conflict_added(&mut *self.rs, (*self.conflictpool).rs(), conflict, &b) };
        d.conflict_pool_changed(self.conflictpoolindex as usize);
    }
}

impl Drop for ConfPropS {
    fn drop(&mut self) {
        // SAFETY: the pool outlives the domains propagating it
        unsafe { (*self.conflictpool).remove_propagation_domain(self) };
    }
}

/// A domain's address as the handle P of glue.rs
pub fn dp(d: &DomS) -> P {
    d as *const DomS as P
}

#[allow(dead_code)]
fn _stdvec_used(_: &StdVec<i32>) {}
