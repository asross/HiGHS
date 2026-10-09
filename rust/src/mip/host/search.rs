//! HighsSearch's shell (highs/mip/HighsSearch.cpp): the Rust search
//! (search.rs) with the local domain, the LP relaxation it uses (and the
//! fresh one of branch's fallback, the strong branching playground), the
//! conflict scratch, and the search's calls back (`CSearchFns`) on them.

use super::dom::DomS;
use super::lp::LpS;
use super::solver::MipSolver;
use super::tables::{self, PscostS, StabS};
use super::worker::WorkerS;
use super::INF;
use crate::lp_data::lp_run::Basis;
use crate::lp_data::opts::OptValue;
use crate::lp_data::LogType;
use crate::mip::domain::{DomChg, Reason};
use crate::mip::glue::source;
use crate::mip::nodequeue::NodeQueue;
use crate::mip::search::{ffi as sf, CModel, CSearchFns, Search, Sources, Stats};
use std::ffi::c_void;
use std::sync::Arc;

/// A std::shared_ptr of the search's nodes: a node basis or stabilizer
/// orbits
pub enum SharedBox {
    Basis(Arc<Basis>),
    Orbits(Arc<StabS>),
}

fn box_basis(b: Option<Arc<Basis>>) -> *mut c_void {
    b.map_or(std::ptr::null_mut(), |b| Box::into_raw(Box::new(SharedBox::Basis(b))) as *mut c_void)
}
fn box_orbits(o: Option<Arc<StabS>>) -> *mut c_void {
    o.map_or(std::ptr::null_mut(), |o| Box::into_raw(Box::new(SharedBox::Orbits(o))) as *mut c_void)
}
fn unbox<'a>(b: *mut c_void) -> &'a SharedBox {
    // SAFETY: a box of box_basis / box_orbits
    unsafe { &*(b as *const SharedBox) }
}
fn take_basis(b: *mut c_void) -> Option<Arc<Basis>> {
    if b.is_null() {
        return None;
    }
    // SAFETY: a box of box_basis, taken over
    match *unsafe { Box::from_raw(b as *mut SharedBox) } {
        SharedBox::Basis(b) => Some(b),
        SharedBox::Orbits(_) => None,
    }
}

/// HighsLpRelaxation::Playground
struct Playground {
    lp: *mut LpS,
    iterate_stored: bool,
}

/// HighsSearch
pub struct SearchS {
    rs: *mut Search,
    pub worker: *mut WorkerS,
    pub mipsolver: *mut MipSolver,
    pub lp: *mut LpS,
    pub localdom: Box<DomS>,
    pub pseudocost: *mut PscostS,
    fallback_lp: Option<Box<LpS>>,
    fallback_swapped: *mut LpS,
}

// SAFETY: a search is used by its worker's task
unsafe impl Send for SearchS {}

impl Drop for SearchS {
    fn drop(&mut self) {
        // SAFETY: the owned search
        unsafe { sf::highs_rs_search_free(self.rs) };
    }
}

fn model_of(ms: &MipSolver) -> CModel {
    let model = ms.model();
    let d = ms.d();
    CModel {
        num_col: ms.num_col(),
        col_cost: model.col_cost.as_ptr(),
        integrality: model.integrality.as_ptr(),
        root_lp_sol: d.vecs.rootlpsol.as_slice().as_ptr(),
        num_root_lp_sol: d.vecs.rootlpsol.len() as i32,
        integral_cols: d.vecs.integral_cols.as_slice().as_ptr(),
        num_integral_cols: d.vecs.integral_cols.len() as i32,
        sources: Sources { heuristic: source::HEURISTIC, branching: source::BRANCHING, evaluate_node: source::EVALUATE_NODE },
    }
}

impl SearchS {
    /// HighsSearch(mipworker, pseudocost)
    pub fn new(worker: *mut WorkerS, pseudocost: *mut PscostS) -> Box<SearchS> {
        // SAFETY: the live worker
        let w = unsafe { &*worker };
        let ms = w.ms();
        let mut s = Box::new(SearchS {
            rs: std::ptr::null_mut(),
            worker,
            mipsolver: w.mipsolver,
            lp: std::ptr::null_mut(),
            localdom: DomS::copy(w.get_global_domain()),
            pseudocost,
            fallback_lp: None,
            fallback_swapped: std::ptr::null_mut(),
        });
        let m = model_of(ms);
        let ctx = &mut *s as *mut SearchS as *mut c_void;
        // SAFETY: the static table, this shell as the context
        s.rs = unsafe { sf::highs_rs_search_new(&SEARCH_FNS, ctx, &m, ms.submip) };
        debug_assert!(!s.localdom.infeasible);
        s.localdom.set_domain_change_stack(&[], None);
        s
    }

    pub fn ms<'a>(&self) -> &'a MipSolver {
        // SAFETY: the solver outlives its searches
        unsafe { &*self.mipsolver }
    }
    pub fn w<'a>(&self) -> &'a WorkerS {
        // SAFETY: the worker owning the search
        unsafe { &*self.worker }
    }
    pub fn rs(&self) -> *mut Search {
        self.rs
    }
    pub fn stats<'a>(&self) -> &'a mut Stats {
        // SAFETY: the search's statistics, used in place as the C++
        unsafe { &mut *sf::highs_rs_search_stats(self.rs) }
    }
    #[allow(clippy::mut_from_ref)]
    pub fn lp<'a>(&self) -> &'a mut LpS {
        // SAFETY: the LP the search uses, live
        unsafe { &mut *self.lp }
    }
    pub fn set_lp(&mut self, lp: *mut LpS) {
        self.lp = lp;
    }
    fn pscost<'a>(&self) -> &'a PscostS {
        // SAFETY: the search's pseudocosts, live
        unsafe { &*self.pseudocost }
    }
    fn nq(&self) -> *const NodeQueue {
        &*self.ms().d().nodequeue
    }

    // ---- HighsSearch::op / run ----

    fn op(&self, which: i32, i: i32, x: f64, y: f64, n: i64, q: *mut NodeQueue) {
        // SAFETY: the live search, pseudocosts and queues
        unsafe { sf::highs_rs_search_op(self.rs, self.pscost().rs, self.nq(), which, i, x, y, n, q as *mut c_void) }
    }
    fn run(&self, which: i32, i: i32, n: i64, q: *mut NodeQueue) -> i32 {
        // SAFETY: as op
        unsafe { sf::highs_rs_search_run(self.rs, self.pscost().rs, self.nq(), which, i, n, q as *mut c_void) }
    }
    pub fn cutoff_node(&self) {
        self.op(1, 0, 0.0, 0.0, 0, std::ptr::null_mut());
    }
    pub fn current_node_to_queue(&self, q: *mut NodeQueue) {
        self.op(5, 0, 0.0, 0.0, 0, q);
    }
    pub fn open_nodes_to_queue(&self, q: *mut NodeQueue) {
        self.op(6, 0, 0.0, 0.0, 0, q);
    }
    pub fn evaluate_node(&self) -> i32 {
        self.run(0, 0, 0, std::ptr::null_mut())
    }
    pub fn backtrack(&self, recover_basis: bool) -> bool {
        self.run(2, recover_basis as i32, 0, std::ptr::null_mut()) != 0
    }
    pub fn backtrack_plunge(&self, q: *mut NodeQueue) -> bool {
        self.run(3, 0, 0, q) != 0
    }
    pub fn dive(&self, node_lim: i64) -> i32 {
        self.run(5, 0, node_lim, std::ptr::null_mut())
    }
    pub fn has_node(&self) -> bool {
        self.run(6, 0, 0, std::ptr::null_mut()) != 0
    }
    pub fn current_node_pruned(&self) -> bool {
        self.run(7, 0, 0, std::ptr::null_mut()) != 0
    }
    pub fn get_current_estimate(&self) -> f64 {
        // SAFETY: the live search
        unsafe { sf::highs_rs_search_value(self.rs, 0) }
    }
    /// installNode(node)
    pub fn install_node(&self, domchgs: &[DomChg], branchings: &[i32], lower_bound: f64, estimate: f64, depth: i32) {
        // SAFETY: as op
        unsafe {
            sf::highs_rs_search_install(
                self.rs,
                self.pscost().rs,
                self.nq(),
                domchgs.as_ptr(),
                domchgs.len() as i32,
                branchings.as_ptr(),
                branchings.len() as i32,
                lower_bound,
                estimate,
                depth,
            )
        }
    }

    // ---- HighsSearch.cpp ----

    pub fn get_upper_limit(&self) -> f64 {
        let d = self.ms().d();
        if !d.parallel_lock_active() {
            d.sc.upper_limit
        } else {
            self.w().st().upper_limit
        }
    }
    pub fn get_optimality_limit(&self) -> f64 {
        let d = self.ms().d();
        if !d.parallel_lock_active() {
            d.sc.optimality_limit
        } else {
            self.w().st().optimality_limit
        }
    }

    /// addBoundExceedingConflict
    fn add_bound_exceeding_conflict(&mut self) {
        let ul = self.get_upper_limit();
        if ul == INF {
            return;
        }
        let w = self.w();
        let gd = w.get_global_domain();
        let Some((inds, vals, rhs)) = self.lp().compute_dual_proof(gd, ul, true) else { return };
        if gd.infeasible {
            return;
        }
        self.localdom.conflict_analysis_proof(&inds, &vals, rhs, w.get_conflict_pool(), gd, self.pscost());
        let (mut inds, mut vals, mut rhs) = (inds, vals, rhs);
        super::sepa::generate_conflict(self.lp(), w.get_cut_pool(), &self.localdom, gd, &mut inds, &mut vals, &mut rhs);
    }

    /// addInfeasibleConflict
    fn add_infeasible_conflict(&mut self) {
        // kObjectiveBound
        if self.lp().lp_model_status() == 11 {
            self.lp().perform_aging(false);
        }
        let w = self.w();
        let gd = w.get_global_domain();
        let Some((mut inds, mut vals, mut rhs)) = self.lp().compute_dual_inf_proof() else { return };
        if gd.infeasible {
            return;
        }
        self.localdom.conflict_analysis_proof(&inds, &vals, rhs, w.get_conflict_pool(), gd, self.pscost());
        super::sepa::generate_conflict(self.lp(), w.get_cut_pool(), &self.localdom, gd, &mut inds, &mut vals, &mut rhs);
    }

    /// flushStatistics
    pub fn flush_statistics(&self) {
        let sc = &mut self.ms().d().sc;
        let st = self.stats();
        sc.num_nodes += st.nnodes;
        st.nnodes = 0;
        sc.num_leaves += st.nleaves;
        st.nleaves = 0;
        sc.pruned_treeweight += st.treeweight;
        st.treeweight = 0.0.into();
        sc.total_lp_iterations += st.lpiterations;
        st.lpiterations = 0;
        sc.heuristic_lp_iterations += st.heurlpiterations;
        st.heurlpiterations = 0;
        sc.sb_lp_iterations += st.sblpiterations;
        st.sblpiterations = 0;
    }

    /// resetLocalDomain
    pub fn reset_local_domain(&mut self) {
        let gd = self.w().get_global_domain();
        self.lp().reset_to_global_domain(gd);
        self.localdom.assign(gd);
    }

    /// checkLimits(nodeOffset)
    pub fn check_limits(&self, node_offset: i64) -> bool {
        let ms = self.ms();
        if ms.d().parallel_lock_active() {
            return self.check_local_limits();
        }
        ms.mip_data().check_limits_rs(node_offset)
    }

    /// checkLocalLimits
    pub fn check_local_limits(&self) -> bool {
        let ms = self.ms();
        let d = ms.d();
        if d.terminator_active() && d.terminator_terminated() {
            return true;
        }
        let o = &ms.opts;
        let w = self.w().st();
        if !ms.submip && w.upper_bound < INF && o.objective_target > -INF {
            let internal_target = ms.orig().sense as f64 * o.objective_target - ms.model().offset;
            if w.upper_bound < internal_target {
                return true;
            }
        }
        let st = self.stats();
        if o.mip_max_nodes != i32::MAX && d.sc.num_nodes + st.nnodes >= o.mip_max_nodes as i64 {
            return true;
        }
        if o.mip_max_leaves != i32::MAX && d.sc.num_leaves + st.nleaves >= o.mip_max_leaves as i64 {
            return true;
        }
        if o.time_limit < INF && ms.timer.read(0) >= o.time_limit {
            return true;
        }
        false
    }

    /// addIncumbent(sol, solobj, solution_source)
    fn add_incumbent(&self, sol: &[f64], obj: f64, source: i32) {
        let ms = self.ms();
        if ms.d().parallel_lock_active() {
            self.w().add_incumbent(sol, obj, source);
        } else {
            super::fns::add_incumbent(ms, sol, obj, source, true, false);
        }
    }

    /// The search's handle (the context of its callbacks)
    pub fn p(&self) -> *mut c_void {
        self as *const SearchS as *mut c_void
    }
}

// ---- the search's calls back (SearchAccess) ----

fn s<'a>(p: *mut c_void) -> &'a mut SearchS {
    // SAFETY: the SearchS the search was made with
    unsafe { &mut *(p as *mut SearchS) }
}

unsafe extern "C" fn shared_clone(b: *mut c_void) -> *mut c_void {
    let c = match unbox(b) {
        SharedBox::Basis(x) => SharedBox::Basis(x.clone()),
        SharedBox::Orbits(x) => SharedBox::Orbits(x.clone()),
    };
    Box::into_raw(Box::new(c)) as *mut c_void
}
unsafe extern "C" fn shared_free(b: *mut c_void) {
    drop(Box::from_raw(b as *mut SharedBox));
}
unsafe extern "C" fn change_bound(p: *mut c_void, d: DomChg) {
    s(p).localdom.change_bound(d, Reason { kind: crate::mip::domain::REASON_BRANCHING, index: 0 });
}
unsafe extern "C" fn propagate(p: *mut c_void) {
    s(p).localdom.propagate();
}
unsafe extern "C" fn infeasible(p: *mut c_void) -> bool {
    s(p).localdom.infeasible
}
unsafe extern "C" fn backtrack(p: *mut c_void) -> DomChg {
    s(p).localdom.backtrack()
}
unsafe extern "C" fn backtrack_to_global(p: *mut c_void) {
    s(p).localdom.backtrack_to_global();
}
unsafe extern "C" fn stack(p: *mut c_void, n: *mut i32) -> *const DomChg {
    let st = s(p).localdom.stack();
    *n = st.len() as i32;
    st.as_ptr()
}
unsafe extern "C" fn num_changed_cols(p: *mut c_void) -> i32 {
    s(p).localdom.changed_cols().len() as i32
}
unsafe extern "C" fn clear_changed_cols(p: *mut c_void, start: i32) {
    if start < 0 {
        s(p).localdom.clear_changed_cols();
    } else {
        s(p).localdom.clear_changed_cols_from(start as usize);
    }
}
unsafe extern "C" fn conflict_analysis(p: *mut c_void) {
    let x = s(p);
    let w = &*x.worker;
    x.localdom.conflict_analysis(w.get_conflict_pool(), w.get_global_domain(), &*x.pseudocost);
}
unsafe extern "C" fn bounds(p: *mut c_void, lo: *mut *const f64, up: *mut *const f64) {
    *lo = s(p).localdom.col_lower().as_ptr();
    *up = s(p).localdom.col_upper().as_ptr();
}
unsafe extern "C" fn objective_lower_bound(p: *mut c_void) -> f64 {
    s(p).localdom.get_objective_lower_bound()
}
unsafe extern "C" fn col_pos(p: *mut c_void, col: i32, upper: bool) -> i32 {
    let d = &s(p).localdom;
    d.get_col_pos(col, d.stack().len() as i32, upper).1
}
unsafe extern "C" fn is_binary(p: *mut c_void, col: i32, global: bool) -> bool {
    let x = s(p);
    if global {
        (*x.worker).get_global_domain().is_binary(col)
    } else {
        x.localdom.is_global_binary(col)
    }
}
unsafe extern "C" fn set_stack(p: *mut c_void, d: *const DomChg, n: i32, b: *const i32, nb: i32) {
    let st = crate::ffi::sl(d, n).to_vec();
    let br = crate::ffi::sl(b, nb).to_vec();
    s(p).localdom.set_domain_change_stack(&st, Some(&br));
}
unsafe extern "C" fn branching_positions(p: *mut c_void, n: *mut i32) -> *const i32 {
    let b = s(p).localdom.branching_positions();
    *n = b.len() as i32;
    b.as_ptr()
}
unsafe extern "C" fn node_to_queue(p: *mut c_void, q: *mut c_void, lb: f64, estimate: f64, depth: i32) -> f64 {
    let (stack, branch) = s(p).localdom.get_reduced_domain_change_stack();
    (*(q as *mut NodeQueue)).emplace_node(&stack, &branch, lb, estimate, depth)
}
unsafe extern "C" fn lp_flush_domain(p: *mut c_void) {
    let x = s(p);
    let d: *mut DomS = &mut *x.localdom;
    x.lp().flush_domain(&mut *d, false);
}
unsafe extern "C" fn lp_set_objective_limit(p: *mut c_void, x: f64) {
    s(p).lp().set_objective_limit(x);
}
unsafe extern "C" fn lp_resolve(p: *mut c_void) -> i32 {
    let x = s(p);
    let d: *mut DomS = &mut *x.localdom;
    x.lp().resolve_lp(Some(&mut *d))
}
unsafe extern "C" fn lp_rust(p: *mut c_void) -> *mut crate::mip::lp_relaxation::LpRelax {
    s(p).lp().rs_ptr()
}
unsafe extern "C" fn lp_store_basis(p: *mut c_void, get: bool) -> *mut c_void {
    if !get {
        s(p).lp().store_basis();
        return std::ptr::null_mut();
    }
    box_basis(s(p).lp().get_stored_basis())
}
unsafe extern "C" fn lp_set_stored_basis(p: *mut c_void, b: *mut c_void) {
    s(p).lp().set_stored_basis(take_basis(b));
}
unsafe extern "C" fn lp_recover_basis(p: *mut c_void) {
    s(p).lp().recover_basis();
}
unsafe extern "C" fn basis_rows(b: *mut c_void) -> i32 {
    match unbox(b) {
        SharedBox::Basis(x) => x.b.row_status.len() as i32,
        SharedBox::Orbits(_) => 0,
    }
}
unsafe extern "C" fn lp_perform_aging(p: *mut c_void) {
    s(p).lp().perform_aging(false);
}
unsafe extern "C" fn lp_degenerate_duals(p: *mut c_void, threshold: f64) {
    let x = s(p);
    let w = &*x.worker;
    let d: *mut DomS = &mut *x.localdom;
    x.lp().compute_basic_degenerate_duals(
        threshold,
        &mut *d,
        w.get_global_domain(),
        w.get_conflict_pool(),
        w.get_pseudocost(),
        true,
    );
}
unsafe extern "C" fn lp_degeneracy(p: *mut c_void) -> f64 {
    let x = s(p);
    x.lp().compute_lp_degeneracy(&x.localdom)
}
unsafe extern "C" fn playground_new(p: *mut c_void) -> *mut c_void {
    Box::into_raw(Box::new(Playground { lp: s(p).lp, iterate_stored: false })) as *mut c_void
}
unsafe extern "C" fn playground_solve(p: *mut c_void, pg: *mut c_void) -> i32 {
    let x = s(p);
    let g = &mut *(pg as *mut Playground);
    let lp = &mut *g.lp;
    let d: *mut DomS = &mut *x.localdom;
    if g.iterate_stored {
        lp.flush_domain(&mut *d, false);
        lp.lph().get_iterate();
    } else {
        lp.lph().put_iterate();
        lp.flush_domain(&mut *d, false);
        g.iterate_stored = true;
    }
    lp.run(false)
}
unsafe extern "C" fn playground_free(pg: *mut c_void) {
    let g = Box::from_raw(pg as *mut Playground);
    if g.iterate_stored {
        let lp = &mut *g.lp;
        lp.lph().get_iterate();
        lp.run(true);
    }
}
unsafe extern "C" fn lp_fallback(p: *mut c_void, step: i32) {
    let x = s(p);
    let ms = &*x.mipsolver;
    match step {
        0 => {
            x.lp().set_iteration_limit(i32::MAX);
            // a fresh LP only with model rows: all integer columns are
            // fixed, the cuts are not required and the LP could not be
            // solved, so it is made as easy as possible
            let mut fresh = LpS::new(ms);
            fresh.set_profiling(ms.prof.p);
            fresh.load_model();
            let n = ms.num_col();
            fresh.lph().change_col_bounds_interval(0, n - 1, x.localdom.col_lower(), x.localdom.col_upper());
            x.fallback_swapped = x.lp;
            x.fallback_lp = Some(fresh);
            x.lp = &mut **x.fallback_lp.as_mut().unwrap();
            x.lp().set_option("presolve", OptValue::Str(b"on"));
        }
        1 => {
            x.lp().lph().clear_solver();
            // kSimplexStrategyPrimal
            x.lp().set_option("simplex_strategy", OptValue::Int(4));
        }
        // kSimplexStrategyDual
        2 => x.lp().set_option("simplex_strategy", OptValue::Int(1)),
        3 => {
            x.lp().lph().clear_solver();
            x.lp().set_option("solver", OptValue::Str(b"ipm"));
        }
        4 => crate::log_user!(
            ms.log,
            LogType::Warning,
            "Failed to solve node with all integer columns fixed. Declaring node infeasible.\n"
        ),
        _ => {
            x.lp = x.fallback_swapped;
            x.fallback_lp = None;
        }
    }
}
unsafe extern "C" fn num_perms(p: *mut c_void) -> i32 {
    s(p).ms().d().symmetries.num_perms
}
unsafe extern "C" fn global_orbits(p: *mut c_void) -> *mut c_void {
    box_orbits(s(p).ms().d().global_orbits.clone())
}
unsafe extern "C" fn column_position(p: *mut c_void, col: i32) -> i32 {
    s(p).ms().d().symmetries.column_position.get(col as usize).copied().unwrap_or(-1)
}
unsafe extern "C" fn compute_stabilizer_orbits(p: *mut c_void) -> *mut c_void {
    let x = s(p);
    let sym: *const crate::presolve::symmetry::Symmetries = &x.ms().d().symmetries;
    let o = StabS::compute(&*sym, &mut x.localdom);
    box_orbits(Some(Arc::new(o)))
}
unsafe extern "C" fn orbits_query(b: *mut c_void, col: i32) -> bool {
    match unbox(b) {
        SharedBox::Orbits(o) => {
            if col < 0 {
                o.orbits.orbit_cols.is_empty()
            } else {
                o.is_stabilized(col)
            }
        }
        SharedBox::Basis(_) => false,
    }
}
unsafe extern "C" fn orbital_fixing(p: *mut c_void, b: *mut c_void) {
    if let SharedBox::Orbits(o) = unbox(b) {
        o.orbital_fixing(&mut s(p).localdom);
    }
}
unsafe extern "C" fn propagate_orbitopes(p: *mut c_void) {
    let x = s(p);
    let sym: *const crate::presolve::symmetry::Symmetries = &x.ms().d().symmetries;
    tables::propagate_orbitopes(&*sym, &mut x.localdom);
}
unsafe extern "C" fn mip_value(p: *mut c_void, which: i32) -> f64 {
    let x = s(p);
    let d = x.ms().d();
    match which {
        0 => d.sc.feastol,
        1 => d.sc.epsilon,
        2 => x.get_upper_limit(),
        _ => x.get_optimality_limit(),
    }
}
unsafe extern "C" fn mip_stat(p: *mut c_void, which: i32) -> i64 {
    let sc = &s(p).ms().d().sc;
    match which {
        0 => sc.heuristic_lp_iterations,
        1 => sc.total_lp_iterations,
        2 => sc.sb_lp_iterations,
        _ => (sc.lns_tree_next >= 0) as i64,
    }
}
unsafe extern "C" fn check_limits(p: *mut c_void, offset: i64) -> bool {
    s(p).check_limits(offset)
}
unsafe extern "C" fn add_incumbent(p: *mut c_void, sol: *const f64, n: i32, obj: f64, source: i32) {
    let v = crate::ffi::sl(sol, n).to_vec();
    s(p).add_incumbent(&v, obj, source);
}
unsafe extern "C" fn add_bound_exceeding_conflict(p: *mut c_void) {
    s(p).add_bound_exceeding_conflict();
}
unsafe extern "C" fn add_infeasible_conflict(p: *mut c_void) {
    s(p).add_infeasible_conflict();
}
unsafe extern "C" fn propagate_redcost(p: *mut c_void) {
    let x = s(p);
    let w = &*x.worker;
    let ul = x.get_upper_limit();
    let d: *mut DomS = &mut *x.localdom;
    tables::propagate_red_cost(x.ms(), &mut *d, w.get_global_domain(), x.lp(), w.get_conflict_pool(), w.get_pseudocost(), ul);
}
unsafe extern "C" fn log_frac_error(p: *mut c_void, upper: bool, fracval: f64, bound: f64, colbound: f64, feastol: f64, residual: f64) {
    let log = &s(p).ms().log;
    if !upper {
        crate::log_user!(
            log,
            LogType::Error,
            "HighsSearch::selectBranchingCandidate Error fracval = %g <= %g = %g + %g = localdom.col_lower_[col] + getFeasTol(): Residual %g\n",
            fracval,
            bound,
            colbound,
            feastol,
            residual
        );
    } else {
        crate::log_user!(
            log,
            LogType::Error,
            "HighsSearch::selectBranchingCandidate Error fracval = %g >= %g = %g - %g = localdom.col_upper_[col] - getFeasTol(): Residual %g\n",
            fracval,
            bound,
            colbound,
            feastol,
            residual
        );
    }
}
unsafe extern "C" fn model(p: *mut c_void, m: *mut CModel) {
    m.write(model_of(s(p).ms()));
}

static SEARCH_FNS: CSearchFns = CSearchFns {
    shared_clone,
    shared_free,
    change_bound,
    propagate,
    infeasible,
    backtrack,
    backtrack_to_global,
    stack,
    num_changed_cols,
    clear_changed_cols,
    conflict_analysis,
    bounds,
    objective_lower_bound,
    col_pos,
    is_binary,
    set_stack,
    branching_positions,
    node_to_queue,
    lp_flush_domain,
    lp_set_objective_limit,
    lp_resolve,
    lp_rust,
    lp_store_basis,
    lp_set_stored_basis,
    lp_recover_basis,
    basis_rows,
    lp_perform_aging,
    lp_degenerate_duals,
    lp_degeneracy,
    playground_new,
    playground_solve,
    playground_free,
    lp_fallback,
    num_perms,
    global_orbits,
    column_position,
    compute_stabilizer_orbits,
    orbits_query,
    orbital_fixing,
    propagate_orbitopes,
    mip_value,
    mip_stat,
    check_limits,
    add_incumbent,
    add_bound_exceeding_conflict,
    add_infeasible_conflict,
    propagate_redcost,
    log_frac_error,
    model,
};
