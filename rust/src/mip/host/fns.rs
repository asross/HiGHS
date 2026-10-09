//! The operations of glue.rs `CMipFns` on the Rust objects: the domain,
//! LP relaxation, search and worker handles, the sub-MIP, the concurrent
//! helper, the repair LP, the scratch solutions, and the scalar operations
//! of mip_data.rs, root.rs, driver.rs, setup.rs and workers.rs
//! (`CMipFns::op` codes), each the step the C++ made on its objects
//! (HighsPrimalHeuristics.cpp's mipglue, HighsMipSolver.cpp's
//! mipDriverOp, HighsMipSolverData.cpp's mipSetupOp).

use super::dom::DomS;
use super::lp::LpS;
use super::pools::{ConflictPoolS, CutPoolS, CutSet};
use super::sepa::SepaS;
use super::solver::{MipSolver, SolverData, MODEL_STATUS_NOTSET};
use super::tables::{self, PscostInit, PscostS, StabS};
use super::worker::WorkerS;
use super::{prof, max2, Own, Prof, SendPtr, IINF, INF, P};
use crate::lp_data::lp::Lp;
use crate::lp_data::lp_handle::LpHandle;
use crate::lp_data::lp_run::{Basis, Solution};
use crate::lp_data::opts::{OptValue, Opts};
use crate::lp_data::{Log, LogType};
use crate::mip::concurrent::Pool;
use crate::mip::domain::{DomChg, Reason};
use crate::mip::driver::CleanupResult;
use crate::mip::glue::{
    self, CMipFns, MipData, MipScalars, Pdi, ScratchView, SearchParts, SubMipResult, SubMipSpec, WorkerData,
};
use crate::mip::lp_relaxation::LpRelax;
use crate::mip::setup::CallbackOut;
use crate::parallel::TaskGroup;
use crate::presolve::symmetry::{ModelView, SymmetryDetection, Symmetries};
use crate::util::cdouble::CDouble;
use std::ffi::c_void;
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// HighsMipScalars' initial values
pub fn scalars_default() -> MipScalars {
    MipScalars {
        feastol: 0.0,
        epsilon: 0.0,
        heuristic_effort: 0.0,
        dispfreq: 0,
        firstlpsolobj: -INF,
        rootlpsolobj: -INF,
        numintegercols: 0,
        max_tree_size_log2: 0,
        pruned_treeweight: CDouble::from(0.0),
        avgrootlpiters: 0.0,
        disptime: 0.0,
        last_disptime: 0.0,
        firstrootlpiters: 0,
        num_nodes: 0,
        num_leaves: 0,
        num_leaves_before_run: 0,
        num_nodes_before_run: 0,
        total_repair_lp: 0,
        total_repair_lp_feasible: 0,
        total_repair_lp_iterations: 0,
        total_lp_iterations: 0,
        heuristic_lp_iterations: 0,
        sepa_lp_iterations: 0,
        sb_lp_iterations: 0,
        total_lp_iterations_before_run: 0,
        heuristic_lp_iterations_before_run: 0,
        sepa_lp_iterations_before_run: 0,
        sb_lp_iterations_before_run: 0,
        num_disp_lines: 0,
        num_improving_sols: 0,
        lower_bound: -INF,
        upper_bound: INF,
        upper_limit: INF,
        optimality_limit: INF,
        num_restarts: 0,
        num_restarts_root: 0,
        num_clique_entries_after_presolve: 0,
        num_clique_entries_after_first_presolve: 0,
        lns_tree_next: -1,
        lns_tree_wait: 0,
        lns_quick_lp_iterations: 0,
        concurrent_lns_seen: 0,
        pdi: Pdi::default(),
        cliques_extracted: false,
        row_matrix_set: false,
        analytic_center_computed: false,
        detect_symmetries: false,
        lns_quick_improved: false,
        crossover_start_logged: false,
        root_cuts_imported: false,
        concurrent_lns: std::ptr::null_mut(),
    }
}

/// The symmetry detection's data (SymmetryDetectionData)
pub struct SymData {
    det: SymmetryDetection,
    symmetries: Symmetries,
    detection_time: f64,
}

/// The locals of evaluateRootNode (dropped in the C++'s reverse order of
/// declaration: the separator, the task group, the symmetry data)
pub struct RootCtx {
    sepa: Option<Box<SepaS>>,
    tg: TaskGroup,
    sym_data: Option<Box<SymData>>,
    sepa_status: i32,
}

/// The locals of performRestart
pub struct RestartCtx {
    pub root_basis: Basis,
    pub pscostinit: PscostInit,
}

/// The task group of HighsMipSolver::run
pub struct RunCtx {
    tg: TaskGroup,
}

fn ms_of<'a>(m: P) -> &'a mut MipSolver {
    // SAFETY: the MipSolver the MipData was made for
    unsafe { &mut *(m as *mut MipSolver) }
}
fn dom<'a>(p: P) -> &'a mut DomS {
    // SAFETY: a domain handle
    unsafe { &mut *(p as *mut DomS) }
}
fn lpr<'a>(p: P) -> &'a mut LpS {
    // SAFETY: an LP relaxation handle
    unsafe { &mut *(p as *mut LpS) }
}
fn wk<'a>(p: P) -> &'a mut WorkerS {
    // SAFETY: a worker handle
    unsafe { &mut *(p as *mut WorkerS) }
}

// ---- the solver's solutions (HighsMipSolverData's methods as the C++
// callers made them) ----

/// addIncumbent
pub fn add_incumbent(ms: &MipSolver, sol: &[f64], obj: f64, source: i32, print: bool, is_user: bool) -> bool {
    ms.mip_data().add_incumbent_rs(sol, obj, source, print, is_user)
}
/// trySolution
pub fn try_solution(ms: &MipSolver, sol: &[f64], source: i32) -> bool {
    ms.mip_data().try_solution_rs(sol, source)
}
/// checkSolution
pub fn check_solution(ms: &MipSolver, sol: &[f64]) -> bool {
    ms.mip_data().check_solution(sol)
}

// ---- CMipFns: the domains ----

unsafe extern "C" fn dom_copy(d: P) -> P {
    Box::into_raw(DomS::copy(dom(d))) as P
}
unsafe extern "C" fn dom_free(d: P) {
    drop(Box::from_raw(d as *mut DomS));
}
unsafe extern "C" fn dom_assign(d: P, other: P) {
    let o: *const DomS = dom(other);
    dom(d).assign(&*o);
}
unsafe extern "C" fn dom_bounds(d: P, lo: *mut *const f64, up: *mut *const f64) {
    *lo = dom(d).col_lower().as_ptr();
    *up = dom(d).col_upper().as_ptr();
}
unsafe extern "C" fn dom_change_bound(d: P, chg: DomChg, reason: Reason) {
    dom(d).change_bound(chg, reason);
}
unsafe extern "C" fn dom_fix_col(d: P, col: i32, val: f64, reason: Reason) {
    dom(d).fix_col(col, val, reason);
}
unsafe extern "C" fn dom_propagate(d: P) -> bool {
    dom(d).propagate()
}
unsafe extern "C" fn dom_infeasible(d: P) -> bool {
    dom(d).infeasible
}
unsafe extern "C" fn dom_backtrack(d: P) -> DomChg {
    dom(d).backtrack()
}
unsafe extern "C" fn dom_conflict_analysis(d: P, w: P) {
    let w = wk(w);
    dom(d).conflict_analysis(w.get_conflict_pool(), w.get_global_domain(), w.get_pseudocost());
}
unsafe extern "C" fn dom_stack(d: P, n: *mut i32) -> *const DomChg {
    let s = dom(d).stack();
    *n = s.len() as i32;
    s.as_ptr()
}
unsafe extern "C" fn dom_branch_depth(d: P) -> i32 {
    dom(d).branching_positions().len() as i32
}
unsafe extern "C" fn dom_clear_changed_cols(d: P) {
    dom(d).clear_changed_cols();
}
unsafe extern "C" fn dom_clear_pool_propagation(d: P) {
    dom(d).clear_pool_propagation();
}
unsafe extern "C" fn dom_num_changed_cols(d: P) -> i32 {
    dom(d).changed_cols().len() as i32
}

// ---- the LP relaxations ----

unsafe extern "C" fn lp_copy(lp: P, w: P) -> P {
    let mut p = LpS::copy(lpr(lp));
    p.set_mip_worker(w as *mut WorkerS);
    p.set_profiling(wk(w).ms().prof.p);
    Box::into_raw(p) as P
}
unsafe extern "C" fn lp_new(m: P, w: P) -> P {
    let ms = ms_of(m);
    let mut p = LpS::new(ms);
    p.set_mip_worker(w as *mut WorkerS);
    p.set_profiling(ms.prof.p);
    p.load_model();
    Box::into_raw(p) as P
}
unsafe extern "C" fn lp_free(lp: P) {
    drop(Box::from_raw(lp as *mut LpS));
}
unsafe extern "C" fn lp_rust(lp: P) -> *mut LpRelax {
    lpr(lp).rs_ptr()
}
unsafe extern "C" fn lp_set_root_basis(lp: P, origin: *const u8) {
    let l = lpr(lp);
    let b = l.ms().d().firstrootbasis.clone();
    let o = std::ffi::CStr::from_ptr(origin as *const std::ffi::c_char).to_string_lossy();
    l.set_lp_basis(b, &o);
}
unsafe extern "C" fn lp_resolve(lp: P, d: P) -> i32 {
    lpr(lp).resolve_lp(if d.is_null() { None } else { Some(dom(d)) })
}
unsafe extern "C" fn lp_set_objective_limit(lp: P, lim: f64) {
    lpr(lp).set_objective_limit(lim);
}
unsafe extern "C" fn lp_flush_domain(lp: P, d: P) {
    lpr(lp).flush_domain(dom(d), false);
}
unsafe extern "C" fn lp_remove_obsolete_rows(lp: P, notify: bool) {
    lpr(lp).remove_obsolete_rows(notify);
}
unsafe extern "C" fn lp_infeasible_conflict(lp: P, w: P, localdom: P) {
    let w = wk(w);
    let l = lpr(lp);
    if let Some((mut inds, mut vals, mut rhs)) = l.compute_dual_inf_proof() {
        super::sepa::generate_conflict(l, w.get_cut_pool(), dom(localdom), w.get_global_domain(), &mut inds, &mut vals, &mut rhs);
    }
}

// ---- the searches of the heuristics ----

/// a heuristic's search, with its own copy of the worker's pseudocosts
struct HeurSearch {
    pscost: Box<PscostS>,
    search: Option<Box<super::search::SearchS>>,
}

unsafe extern "C" fn search_new(w: P, parts: *mut SearchParts) {
    let worker = wk(w);
    let mut h = Box::new(HeurSearch { pscost: Box::new(PscostS::copy(worker.get_pseudocost())), search: None });
    let ps: *mut PscostS = &mut *h.pscost;
    h.search = Some(super::search::SearchS::new(worker, ps));
    let s = h.search.as_mut().unwrap();
    let p = &mut *parts;
    p.rs = s.rs();
    p.ps = h.pscost.rs;
    p.nq = &*worker.ms().d().nodequeue;
    p.localdom = &mut *s.localdom as *mut DomS as P;
    p.cpp = Box::into_raw(h) as P;
}
unsafe extern "C" fn search_free(h: P) {
    let mut s = Box::from_raw(h as *mut HeurSearch);
    // the search before its pseudocosts, as in the C++ heuristics
    s.search = None;
    drop(s);
}
unsafe extern "C" fn search_set_lp(h: P, lp: P) {
    (*(h as *mut HeurSearch)).search.as_mut().unwrap().set_lp(lp as *mut LpS);
}

// ---- the solver ----

unsafe extern "C" fn check_limits(m: P) -> bool {
    ms_of(m).mip_data().check_limits_rs(0)
}
unsafe extern "C" fn update_lower_bound(m: P, lb: f64) {
    ms_of(m).mip_data().update_lower_bound_ex(lb, true, true);
}
unsafe extern "C" fn parallel_lock_active(m: P) -> bool {
    ms_of(m).d().parallel_lock_active()
}
unsafe extern "C" fn num_workers(m: P) -> i32 {
    ms_of(m).d().workers.len() as i32
}
unsafe extern "C" fn worker_view(w: P, d: *mut WorkerData) {
    let worker = wk(w);
    let st = worker.st();
    let d = &mut *d;
    d.upper_limit = &mut st.upper_limit;
    d.heur = &mut st.heur;
    d.randgen = &mut st.randgen;
    d.globaldom = worker.globaldom as P;
    d.lp = worker.lp as P;
    d.upper_bound = &mut st.upper_bound;
    d.optimality_limit = &mut st.optimality_limit;
    d.state = st;
}

/// HighsPrimalHeuristics::solveSubMip's run of the sub-MIP of the spec
unsafe extern "C" fn sub_mip(m: P, w: P, spec: *const SubMipSpec, r: *mut SubMipResult, sol: *mut f64) {
    let ms = ms_of(m);
    let worker = wk(w);
    let spec = &*spec;
    let r = &mut *r;
    let d = ms.d();
    // the LP relaxation's model and basis (copies), or the MIP's
    let (lp_model, basis) = if spec.lp.is_null() {
        (ms.model().clone(), d.firstrootbasis.clone())
    } else {
        let l = lpr(spec.lp);
        (l.get_lp_copy(), l.get_lp_basis())
    };
    let mut o: Opts = ms.opts.clone();
    let n = lp_model.num_col as usize;
    let mut submip = lp_model;
    submip.col_lower = crate::ffi::sl(spec.col_lower, n as i32).to_vec();
    submip.col_upper = crate::ffi::sl(spec.col_upper, n as i32).to_vec();
    submip.integrality = ms.model().integrality.clone();
    submip.offset = 0.0;
    o.mip_max_leaves = spec.mip_max_leaves;
    o.output_flag = spec.output_flag;
    o.mip_max_nodes = spec.mip_max_nodes;
    o.mip_max_stall_nodes = spec.mip_max_stall_nodes;
    o.mip_pscost_minreliable = spec.mip_pscost_minreliable;
    o.time_limit = spec.time_limit;
    o.objective_bound = spec.objective_bound;
    if !spec.mip_abs_gap.is_nan() {
        o.mip_rel_gap = spec.mip_rel_gap;
        o.mip_abs_gap = spec.mip_abs_gap;
    }
    o.set("presolve", OptValue::Str(if spec.presolve { b"on" } else { b"off" }));
    o.mip_detect_symmetry = spec.mip_detect_symmetry;
    o.mip_heuristic_effort = spec.mip_heuristic_effort;
    if spec.heur_flags >= 0 {
        o.mip_heuristic_run_rins = spec.heur_flags & 1 != 0;
        o.mip_heuristic_run_rens = (spec.heur_flags >> 1) & 1 != 0;
        o.mip_heuristic_run_root_reduced_cost = (spec.heur_flags >> 2) & 1 != 0;
    }
    let start: Option<(Vec<f64>, Vec<f64>)> = if spec.start_cols.is_null() {
        None
    } else {
        Some((
            crate::ffi::sl(spec.start_cols, ms.num_col()).to_vec(),
            crate::ffi::sl(spec.start_rows, spec.num_start_rows).to_vec(),
        ))
    };
    let prof = ms.prof;
    let lock = d.parallel_lock_active();
    if !ms.submip && !lock {
        prof.start(prof::SUB_MIP_SOLVE);
    }
    prof.solve_call(4, ms.submip);
    let log = if o.output_flag { ms.log } else { Log::none() };
    let name = Vec::new();
    let mut sub = MipSolver::new(
        ms.host,
        prof,
        o,
        log,
        submip,
        name,
        start.as_ref().map(|(c, r)| (c.as_slice(), r.as_slice())),
        true,
        ms.submip_level + 1,
    );
    sub.initialise_terminator_from(ms);
    sub.lns_target_reached = spec.lns_target;
    sub.rootbasis = &basis;
    let pscostinit = PscostInit::new(worker.get_pseudocost(), 1);
    sub.pscostinit = &pscostinit;
    sub.clqtableinit = &*d.cliquetable;
    sub.implicinit = &*d.implications;
    let was_running_solve = prof.running(prof::SOLVE_TIME);
    if was_running_solve {
        prof.stop(prof::SOLVE_TIME);
    }
    if prof.sub_solver() {
        crate::io::log::c_stdout_flush(
            format!(
                "\nHighsPrimalHeuristics::solveSubMip Before run() for {}MIP at depth {:2} on thread {:2}\n",
                if ms.submip { "sub-" } else { "    " },
                ms.submip_level,
                prof.my_thread()
            )
            .as_bytes(),
        );
    }
    if !ms.submip {
        prof.start(prof::SUB_SOLVER_SUB_MIP);
    }
    prof.set_submip(true);
    run(&mut sub);
    if prof.sub_solver() {
        crate::io::log::c_stdout_flush(
            format!(
                "HighsPrimalHeuristics::solveSubMip After  run() for {}MIP at depth {:2} on thread {:2}\n\n",
                if ms.submip { "sub-" } else { "    " },
                ms.submip_level,
                prof.my_thread()
            )
            .as_bytes(),
        );
    }
    prof.set_submip(ms.submip);
    if !ms.submip {
        prof.stop(prof::SUB_SOLVER_SUB_MIP);
    }
    if !ms.submip && !lock {
        prof.stop(prof::SUB_MIP_SOLVE);
    }
    if was_running_solve {
        prof.restart(prof::SOLVE_TIME);
    }
    let sd = sub.d();
    r.termination_status = sub.termination_status;
    r.model_status = sub.modelstatus;
    r.node_count = sub.node_count;
    r.max_submip_level = sub.max_submip_level;
    r.total_lp_iterations = sd.sc.total_lp_iterations;
    r.total_repair_lp = sd.sc.total_repair_lp;
    r.total_repair_lp_feasible = sd.sc.total_repair_lp_feasible;
    r.total_repair_lp_iterations = sd.sc.total_repair_lp_iterations;
    r.has_solution = !sub.solution.is_empty();
    if r.has_solution {
        std::ptr::copy_nonoverlapping(sub.solution.as_ptr(), sol, sub.solution.len());
    }
}

/// The primal postsolve of a solution into the scratch solution, with its
/// row values on the original model
fn postsolve_into(ms: &MipSolver, scratch: &mut Solution, sol: &[f64]) {
    *scratch = Solution::default();
    scratch.col_value = sol.to_vec();
    scratch.value_valid = true;
    super::presolve::undo_primal(ms, &ms.d().postsolve_stack, scratch);
    let orig = ms.orig();
    scratch.row_value = vec![0.0; orig.num_row as usize];
    crate::lp_data::edit::calculate_row_values_quad(
        &orig.a.start,
        &orig.a.index,
        &orig.a.value,
        &scratch.col_value,
        &mut scratch.row_value,
    );
}

fn view_of(s: &Solution, v: &mut ScratchView) {
    v.col = s.col_value.as_ptr();
    v.ncol = s.col_value.len() as i32;
    v.row = s.row_value.as_ptr();
    v.nrow = s.row_value.len() as i32;
}

unsafe extern "C" fn scratch_solution(m: P, sol: *const f64, n: i32, v: *mut ScratchView) {
    let ms = ms_of(m);
    let d = ms.d();
    if n >= 0 {
        let x = crate::ffi::sl(sol, n).to_vec();
        postsolve_into(ms, &mut d.scratch, &x);
    }
    view_of(&d.scratch, &mut *v);
}

unsafe extern "C" fn refill(m: P, out: *mut MipData) {
    out.write(ms_of(m).mip_data());
}

unsafe extern "C" fn master_worker(m: P) -> P {
    ms_of(m).d().workers[0].ptr() as P
}

/// runTask(processNode) over the indices (a parallel lock, tasks if more
/// than one)
unsafe extern "C" fn run_process_nodes(m: P, idx: *const i32, n: i32, ctx: *const c_void) {
    let ms = ms_of(m);
    let indices = crate::ffi::sl(idx, n).to_vec();
    if indices.is_empty() {
        return;
    }
    ms.set_parallel_lock(true);
    let spawn_tasks = indices.len() > 1 && !ms.opts.mip_search_simulate_concurrency;
    let tg: *const TaskGroup = &ms.d().run.as_ref().expect("run context").tg;
    let ctx = SendPtr(ctx as *mut c_void);
    for &i in &indices {
        if spawn_tasks {
            (*tg).spawn(move || {
                let c = ctx;
                process_node(c.0, i);
            });
        } else {
            process_node(ctx.0, i);
        }
    }
    if spawn_tasks {
        (*tg).task_wait();
    }
    ms.set_parallel_lock(false);
}

fn process_node(ctx: *mut c_void, i: i32) {
    // SAFETY: the ProcessCtx of the call, alive until the tasks are synced
    unsafe { crate::mip::driver::highs_rs_mip_process_node(&MIP_FNS, ctx, i) }
}

unsafe extern "C" fn set_cleanup_result(m: P, r: *const c_void) {
    let ms = ms_of(m);
    let r = &*(r as *const CleanupResult);
    ms.dual_bound = r.dual_bound;
    ms.primal_bound = r.primal_bound;
    ms.gap = r.gap;
    ms.node_count = r.node_count;
    ms.total_lp_iterations = r.total_lp_iterations;
    ms.primal_dual_integral = r.primal_dual_integral;
}

unsafe extern "C" fn model_name(m: P, n: *mut i32) -> *const u8 {
    let name = &ms_of(m).model_name;
    *n = name.len() as i32;
    name.as_ptr()
}

unsafe extern "C" fn max_submip_level(m: P) -> i32 {
    ms_of(m).max_submip_level
}

/// The concurrent LNS helper's options, model and root basis
struct HelperData {
    opts: Opts,
    model: Lp,
    basis: Basis,
    host: P,
}

unsafe extern "C" fn helper_new(m: P, time_left: f64) -> P {
    let ms = ms_of(m);
    let mut o = ms.opts.clone();
    o.set("presolve", OptValue::Str(b"off"));
    o.output_flag = false;
    o.mip_improving_solution_save = false;
    o.mip_detect_symmetry = false;
    o.mip_heuristic_run_rens = false;
    o.mip_heuristic_run_rins = false;
    o.mip_heuristic_run_root_reduced_cost = false;
    o.mip_heuristic_run_feasibility_jump = false;
    o.mip_concurrent_helper = false;
    o.random_seed = ms.opts.random_seed + 1;
    o.time_limit = time_left;
    let data = HelperData { opts: o, model: ms.model().clone(), basis: ms.d().firstrootbasis.clone(), host: ms.host };
    Box::into_raw(Box::new(data)) as P
}

/// In the helper's thread: its own single-thread task scheduler and
/// solver
unsafe extern "C" fn helper_run(d: P, pool: *const c_void) {
    let data = Box::from_raw(d as *mut HelperData);
    crate::parallel::initialize_thread(1);
    let HelperData { opts, model, basis, host } = *data;
    let mut helper = MipSolver::new(host, Prof::none(), opts, Log::none(), model, Vec::new(), None, true, 1);
    helper.concurrent_lns = pool as *const Pool;
    helper.rootbasis = &basis;
    run(&mut helper);
}

unsafe extern "C" fn add_root_cut(m: P, inds: *const i32, vals: *const f64, len: i32, rhs: f64, integral: bool) {
    let ms = ms_of(m);
    let mut i = crate::ffi::sl(inds, len).to_vec();
    let mut v = crate::ffi::sl(vals, len).to_vec();
    ms.d().get_cut_pool().add_cut(ms, &mut i, &mut v, rhs, integral, true, false, false);
}

unsafe extern "C" fn vec_ptr(m: P, which: i32, n: *mut i32) -> *const c_void {
    let ms = ms_of(m);
    let d = ms.d();
    fn vd<T>(v: &[T], n: *mut i32) -> *const c_void {
        // SAFETY: the caller's output
        unsafe { *n = v.len() as i32 };
        v.as_ptr() as *const c_void
    }
    match which {
        0 => vd(&d.firstrootbasis.b.col_status, n),
        1 => vd(&d.firstrootbasis.b.row_status, n),
        2 => {
            if ms.rootbasis.is_null() {
                std::ptr::null()
            } else {
                vd(&(*ms.rootbasis).b.col_status, n)
            }
        }
        3 => {
            if ms.rootbasis.is_null() {
                std::ptr::null()
            } else {
                vd(&(*ms.rootbasis).b.row_status, n)
            }
        }
        4 => vd(&d.postsolve_stack.orig_col_index, n),
        5 => vd(&d.postsolve_stack.orig_row_index, n),
        6 => {
            let mut len = 0;
            let p = match super::highs_fns() {
                Some(f) if !ms.host.is_null() => (f.user_solution)(ms.host, &mut len),
                _ => std::ptr::null(),
            };
            *n = len;
            if p.is_null() {
                std::ptr::NonNull::<f64>::dangling().as_ptr() as *const c_void
            } else {
                p as *const c_void
            }
        }
        _ => vd(&d.scratch.col_value, n),
    }
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn set_basis(m: P, which: i32, col: *const u8, ncol: i32, row: *const u8, nrow: i32, valid: bool, alien: bool, useful: bool) {
    let ms = ms_of(m);
    let d = ms.d();
    let b: &mut Basis = if which == 0 { &mut d.firstrootbasis } else { &mut d.restart.as_mut().unwrap().root_basis };
    b.b.col_status = crate::ffi::sl(col, ncol).to_vec();
    b.b.row_status = crate::ffi::sl(row, nrow).to_vec();
    b.b.valid = valid;
    b.b.alien = alien;
    b.b.useful = useful;
    if which == 1 {
        ms.rootbasis = b;
    }
}

unsafe extern "C" fn callback(m: P, t: i32, out: *const CallbackOut, msg: *const u8, len: i32) -> bool {
    let ms = ms_of(m);
    let Some(f) = super::highs_fns() else { return false };
    if ms.host.is_null() {
        return false;
    }
    if out.is_null() {
        return (f.callback)(ms.host, 0, t, out, std::ptr::null(), 0, msg, len);
    }
    let (sp, sn) = match (*out).solution {
        1 => (ms.solution.as_ptr(), ms.solution.len() as i32),
        2 => (ms.d().scratch.col_value.as_ptr(), ms.d().scratch.col_value.len() as i32),
        _ => (std::ptr::null(), 0),
    };
    (f.callback)(ms.host, 0, t, out, sp, sn, msg, len)
}

unsafe extern "C" fn worker(m: P, k: i32) -> P {
    ms_of(m).d().workers[k as usize].ptr() as P
}

unsafe extern "C" fn worker_scratch(m: P, w: P, x: *const f64, n: i32, v: *mut ScratchView) {
    let ms = ms_of(m);
    let worker = wk(w);
    let sol = crate::ffi::sl(x, n).to_vec();
    postsolve_into(ms, &mut worker.scratch, &sol);
    view_of(&worker.scratch, &mut *v);
}

/// transformNewIntegerFeasibleSolution's repair LP: the original model
/// with the given bounds, no integers
unsafe extern "C" fn repair_lp(m: P, lower: *const f64, upper: *const f64, time_limit: f64, feastol: f64, presolve: bool, iterations: *mut i64) -> bool {
    let ms = ms_of(m);
    let mut fixed = ms.orig().clone();
    let n = fixed.num_col;
    fixed.integrality.clear();
    fixed.col_lower = crate::ffi::sl(lower, n).to_vec();
    fixed.col_upper = crate::ffi::sl(upper, n).to_vec();
    fixed.is_moved = false;
    let mut h = LpHandle::new();
    h.profiling = ms.prof.p;
    h.set_option("output_flag", OptValue::Bool(false));
    h.set_option("time_limit", OptValue::Double(time_limit));
    h.set_option("primal_feasibility_tolerance", OptValue::Double(feastol));
    h.set_option("presolve", OptValue::Str(if presolve { b"choose" } else { b"off" }));
    h.pass_model(fixed);
    h.set_option("solver", OptValue::Str(b"simplex"));
    h.optimize_lp();
    let info = h.info();
    *iterations = info.simplex_iteration_count as i64;
    // kSolutionStatusFeasible
    if info.primal_solution_status != 2 {
        return false;
    }
    let s = h.solution();
    let d = ms.d();
    d.scratch = Solution {
        value_valid: s.value_valid,
        dual_valid: s.dual_valid,
        col_value: s.col_value.clone(),
        col_dual: s.col_dual.clone(),
        row_value: s.row_value.clone(),
        row_dual: s.row_dual.clone(),
    };
    true
}

// ---- the scalar operations ----

unsafe extern "C" fn op(m: P, which: i32, w: P, i: i64, x: f64) -> f64 {
    let ms = ms_of(m);
    let d = ms.d();
    match which {
        0 => ms.timer.read(0),
        1 => (d.terminator_active() && d.terminator_terminated()) as i32 as f64,
        4 => d.get_cut_pool().num_cuts() as f64,
        5 => d.get_conflict_pool().num_conflicts() as f64,
        6 => d.get_lp().num_rows() as f64,
        7 => d.objective_function.integral_scale(),
        8 => d.cliquetable.substitutions.len() as f64,
        14 => {
            for wk in &mut d.workers {
                let s = wk.st();
                if i == 0 {
                    s.upper_bound = d.sc.upper_bound;
                } else {
                    s.upper_limit = d.sc.upper_limit;
                    s.optimality_limit = d.sc.optimality_limit;
                }
            }
            0.0
        }
        // debugSolution.newIncumbentFound (no debug solution)
        15 => 0.0,
        16 => {
            tables::propagate_root_redcost(ms);
            0.0
        }
        17 => {
            tables::extract_obj_cliques(ms);
            0.0
        }
        18 => {
            if let Some(o) = d.global_orbits.clone() {
                o.orbital_fixing(d.get_domain());
            }
            0.0
        }
        20 => {
            ms.solution = std::mem::take(&mut d.scratch.col_value);
            ms.solution_objective = x;
            0.0
        }
        21 => ms.num_row() as f64,
        22 => (d.get_lp().lp_model_status() == MODEL_STATUS_NOTSET) as i32 as f64,
        23 => {
            let lp = d.get_lp();
            tables::add_root_redcost(ms, lp.col_dual().as_ptr(), lp.objective());
            0.0
        }
        24 => {
            let sol = d.get_lp().col_value().to_vec();
            heur(ms, w, 8, Some(&sol));
            0.0
        }
        25 => ms.solution.is_empty() as i32 as f64,
        100..=150 => root_op(ms, which, w, i, x),
        200..=299 => driver_op(ms, which, w, i, x),
        _ => setup_op(ms, which, w, i, x),
    }
}

/// highs_rs_heur_run of heuristic `which` on a point or none
pub fn heur(ms: &MipSolver, w: P, which: i32, x: Option<&[f64]>) {
    let m = ms.mip_data();
    let (p, n) = x.map_or((std::ptr::null(), 0), |x| (x.as_ptr(), x.len() as i32));
    // SAFETY: the solver's heuristics and data
    unsafe { crate::mip::primal::ffi::highs_rs_heur_run(&mut *ms.d().heuristics, &MIP_FNS, &m, w, which, p, n) };
}

/// heuristics.graphLNS(worker, x, deep, maxLpIters)
fn graph_lns(ms: &MipSolver, w: P, x: &[f64], deep: bool, max_lp_iters: i64) {
    let m = ms.mip_data();
    // SAFETY: as heur
    unsafe {
        crate::mip::graph_lns::highs_rs_heur_graph_lns(
            &mut *ms.d().heuristics,
            &MIP_FNS,
            &m,
            w,
            x.as_ptr(),
            x.len() as i32,
            deep,
            max_lp_iters,
        )
    };
}

/// heuristics.flushStatistics(mipsolver, worker)
pub fn flush_heur_statistics(ms: &mut MipSolver, w: &WorkerS) {
    let h = &w.st().heur;
    let d = ms.d();
    d.sc.total_repair_lp += h.total_repair_lp;
    d.sc.total_repair_lp_feasible += h.total_repair_lp_feasible;
    d.sc.total_repair_lp_iterations += h.total_repair_lp_iterations;
    d.sc.heuristic_lp_iterations += h.lp_iterations;
    d.sc.total_lp_iterations += h.lp_iterations;
    ms.max_submip_level = ms.max_submip_level.max(h.max_submip_level);
    if h.termination_status != MODEL_STATUS_NOTSET && ms.termination_status == MODEL_STATUS_NOTSET {
        ms.termination_status = h.termination_status;
    }
    d.heuristics.add_observations(h.success_observations, h.num_success_observations, h.infeas_observations, h.num_infeas_observations);
    w.reset_heur_stats();
}

/// the root node operations of root.rs (mod op)
unsafe fn root_op(ms: &mut MipSolver, which: i32, w: P, i: i64, x: f64) -> f64 {
    let d = ms.d();
    let p = ms.prof;
    match which {
        100 => p.start(i as i32),
        101 => p.stop(i as i32),
        102 => return p.running(i as i32) as i32 as f64,
        103 => return p.mip() as i32 as f64,
        104 => return p.is_submip() as i32 as f64,
        105 => {
            d.root = Some(Box::new(RootCtx { sepa: None, tg: TaskGroup::new(), sym_data: None, sepa_status: 0 }));
        }
        106 => d.root = None,
        107 => d.root.as_ref().unwrap().tg.cancel(),
        108 => d.root.as_ref().unwrap().tg.task_wait(),
        109 => start_symmetry_detection(ms),
        110 => start_analytic_center(ms),
        113 => d.get_lp().set_iteration_limit(if i < 0 { IINF } else { i as i32 }),
        114 => d.get_lp().load_model(),
        115 => d.get_domain().clear_changed_cols(),
        116 => d.get_lp().set_objective_limit(x),
        117 => return d.get_domain().get_objective_lower_bound(),
        119 => {
            let lp = d.get_lp();
            if d.firstrootbasis.b.valid {
                lp.set_lp_basis(d.firstrootbasis.clone(), "HighsMipSolverData::evaluateRootNode");
            } else if ms.opts.mip_root_presolve_only {
                lp.set_option("presolve", OptValue::Str(b"off"));
            } else {
                lp.set_option("presolve", OptValue::Str(b"on"));
            }
            if ms.opts.highs_debug_level != 0 {
                lp.set_option("output_flag", OptValue::Bool(ms.opts.output_flag));
            }
        }
        120 => d.get_lp().set_race_ipx(i != 0),
        121 => return use_concurrent_helper(ms) as i32 as f64,
        122 => return d.firstrootbasis.b.valid as i32 as f64,
        123 => {
            let lp = d.get_lp();
            lp.set_option("output_flag", OptValue::Bool(false));
            lp.set_option("presolve", OptValue::Str(b"off"));
            lp.set_option("parallel", OptValue::Str(b"off"));
        }
        124 => {
            let lp = d.get_lp();
            let v = lp.col_value().to_vec();
            d.vecs.set_f64(crate::mip::mip_data::vec::FIRSTLPSOL, &v);
            d.sc.firstlpsolobj = lp.objective();
            d.sc.rootlpsolobj = d.sc.firstlpsolobj;
        }
        125 => {
            let lp = d.get_lp();
            if lp.lp_basis_valid() && lp.num_rows() == ms.num_row() {
                d.firstrootbasis = lp.get_lp_basis();
            } else {
                // the root basis is later expected to be consistent for the
                // model without cuts, so the slack basis if the current one
                // includes cuts
                d.firstrootbasis.b.col_status = vec![2; ms.num_col() as usize];
                d.firstrootbasis.b.row_status = vec![1; ms.num_row() as usize];
                d.firstrootbasis.b.valid = true;
                d.firstrootbasis.b.useful = true;
            }
        }
        126 => {
            let mut cutset = CutSet::default();
            d.get_cut_pool().separate_lp_cuts_after_restart(&mut cutset);
            d.get_lp().add_cuts(&mut cutset);
        }
        127 => d.get_lp().remove_obsolete_rows(true),
        128 => {
            let worker = wk(w);
            let fl = d.vecs.firstlpsol.as_slice().to_vec();
            let rl = d.vecs.rootlpsol.as_slice().to_vec();
            match i {
                0 => heur(ms, w, 8, Some(&fl)),
                1 => heur(ms, w, 6, Some(&fl)),
                2 => heur(ms, w, 7, Some(&fl)),
                3 => graph_lns(ms, w, &fl, false, -1),
                4 => graph_lns(ms, w, &rl, true, -1),
                5 => flush_heur_statistics(ms, worker),
                6 => heur(ms, w, 5, None),
                7 => heur(ms, w, 3, None),
                8 => heur(ms, w, 1, Some(&rl)),
                9 => heur(ms, w, 4, None),
                10 => heur(ms, w, 7, Some(&rl)),
                11 => {
                    let s = d.get_lp().col_value().to_vec();
                    heur(ms, w, 6, Some(&s))
                }
                _ => {
                    let s = d.get_lp().col_value().to_vec();
                    heur(ms, w, 7, Some(&s))
                }
            }
        }
        133 => {
            let mut cutset = CutSet::default();
            let sol = d.get_lp().col_value().to_vec();
            d.get_cut_pool().separate(&sol, d.get_domain(), &mut cutset, d.sc.feastol, &d.cutpools, false);
            if cutset.is_empty() {
                return 0.0;
            }
            d.get_lp().add_cuts(&mut cutset);
            return 1.0;
        }
        135 => {
            // the LP's cut rows for the main solver
            let lp = d.get_lp();
            let (mut start, mut index, mut value, mut rhs, mut integral) = (vec![0i32], Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for row in ms.num_row()..lp.num_rows() {
                let (ri, rv) = lp.get_row(row);
                index.extend_from_slice(ri);
                value.extend_from_slice(rv);
                start.push(index.len() as i32);
                rhs.push(lp.row_upper(row));
                integral.push(lp.is_row_integral(row) as u8);
            }
            crate::mip::concurrent::highs_rs_concurrent_lns_set_root_cuts(
                ms.concurrent_lns,
                start.as_ptr(),
                rhs.len() as i32,
                index.as_ptr(),
                value.as_ptr(),
                index.len() as i32,
                rhs.as_ptr(),
                integral.as_ptr(),
            );
        }
        137 => {
            let est = d.get_lp().compute_best_estimate(wk(w).get_pseudocost());
            d.nodequeue.emplace_node(&[], &[], d.sc.lower_bound, est, 1);
        }
        138 => {
            let mut sepa = SepaS::new(w as *mut WorkerS);
            sepa.lp = d.get_lp();
            d.root.as_mut().unwrap().sepa = Some(sepa);
        }
        139 => {
            let gd: *mut DomS = d.get_domain();
            let ctx = d.root.as_mut().unwrap();
            ctx.sepa_status = i as i32;
            let mut st = ctx.sepa_status;
            let n = ctx.sepa.as_mut().unwrap().separation_round(&mut *gd, &mut st);
            d.root.as_mut().unwrap().sepa_status = st;
            return n as f64;
        }
        140 => return d.root.as_ref().unwrap().sepa_status as f64,
        141 => d.root.as_mut().unwrap().sepa = None,
        142 => return ms.terminate() as i32 as f64,
        143 => return d.get_lp().avg_solve_iters(),
        146 => return d.get_lp().lp_basis_valid() as i32 as f64,
        147 => {
            let v = d.get_lp().col_value().to_vec();
            d.vecs.set_f64(crate::mip::mip_data::vec::ROOTLPSOL, &v);
        }
        148 => return d.get_domain().changed_cols().len() as f64,
        149 => return wk(w).st().heur.lp_iterations as f64,
        150 => d.skip_analytic_center.store(i != 0, Ordering::SeqCst),
        _ => unreachable!("root op {which}"),
    }
    0.0
}

/// useConcurrentHelper
pub fn use_concurrent_helper(ms: &MipSolver) -> bool {
    let o = &ms.opts;
    !ms.submip
        && o.mip_concurrent_helper
        && o.threads != 1
        && std::thread::available_parallelism().map_or(1, |n| n.get()) >= 2
        && o.mip_heuristic_run_graph_lns
        && o.mip_rel_gap >= 1e-3
}

/// startSymmetryDetection(taskGroup, symData)
fn start_symmetry_detection(ms: &mut MipSolver) {
    let d = ms.d();
    let model = &d.presolved_model;
    let mut sym = Box::new(SymData { det: SymmetryDetection::default(), symmetries: Symmetries::default(), detection_time: 0.0 });
    let view = ModelView {
        num_col: model.num_col,
        num_row: model.num_row,
        a_start: &model.a.start,
        a_index: &model.a.index,
        a_value: &model.a.value,
        col_cost: &model.col_cost,
        col_lower: &model.col_lower,
        col_upper: &model.col_upper,
        integrality: &model.integrality,
        row_lower: &model.row_lower,
        row_upper: &model.row_upper,
    };
    sym.det.load_model_as_graph(&view, ms.opts.small_matrix_value);
    d.sc.detect_symmetries = sym.det.initialize_detection();
    if d.sc.detect_symmetries {
        let sp = SendPtr(&mut *sym as *mut SymData);
        let ctx = d.root.as_mut().unwrap();
        ctx.sym_data = Some(sym);
        ctx.tg.spawn(move || {
            let s = sp;
            // SAFETY: the data lives in the root context until the task
            // group is synced
            let sd = unsafe { &mut *s.0 };
            let start = wall_time();
            let completed = sd.det.run(&mut sd.symmetries, || {
                let q = crate::parallel::this_worker_deque();
                // SAFETY: this thread's deque
                !q.is_null() && unsafe { (*q).check_interrupt() }
            });
            if completed {
                sd.detection_time = wall_time() - start;
            }
        });
    }
}

fn wall_time() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

/// startAnalyticCenterComputation: the IPM solve of the analytic centre
fn start_analytic_center(ms: &mut MipSolver) {
    let d = ms.d();
    let dp = SendPtr(d as *mut SolverData);
    d.root.as_ref().unwrap().tg.spawn(move || {
        let dp = dp;
        // SAFETY: the solver's data, alive until the root's task group is
        // synced
        unsafe { analytic_center(&mut *dp.0) }
    });
}

unsafe fn analytic_center(d: &mut SolverData) {
    // first check if the computation should be cancelled, e.g. due to an
    // early return in the root node evaluation
    if d.skip_analytic_center.load(Ordering::SeqCst) {
        return;
    }
    let ms = d.ms();
    let mut ipm = LpHandle::new();
    ipm.profiling = ms.prof.p;
    ipm.set_option("output_flag", OptValue::Bool(false));
    // no presolve: postsolve may put integer variables on bounds
    ipm.set_option("presolve", OptValue::Str(b"off"));
    ipm.set_option("solver", OptValue::Str(b"ipx"));
    ipm.set_option("ipm_iteration_limit", OptValue::Int(200));
    ipm.set_option("time_limit", OptValue::Double(max2(0.0, ms.opts.time_limit - ms.timer.read(0))));
    ipm.set_option("run_crossover", OptValue::Str(b"off"));
    ipm.set_option("run_centring", OptValue::Bool(true));
    let mut lpmodel = ms.model().clone();
    lpmodel.col_cost = vec![0.0; lpmodel.num_col as usize];
    lpmodel.integrality.clear();
    lpmodel.is_moved = false;
    ipm.pass_model(lpmodel);
    ms.prof.start(prof::SUB_SOLVER_IPX_AC);
    ipm.optimize_lp();
    // a cancelled task: the C++ threw out of the task
    if ipm.task_interrupted {
        return;
    }
    ms.prof.stop(prof::SUB_SOLVER_IPX_AC);
    let s = ipm.solution();
    if s.col_value.len() != ms.num_col() as usize {
        return;
    }
    d.analytic_center_status.store(ipm.model_status(), Ordering::SeqCst);
    let v = s.col_value.clone();
    d.vecs.set_f64(crate::mip::mip_data::vec::ANALYTIC_CENTER, &v);
}

/// the driver's operations of driver.rs (mod op)
unsafe fn driver_op(ms: &mut MipSolver, which: i32, _w: P, i: i64, x: f64) -> f64 {
    let d = ms.d();
    let prof = ms.prof;
    let lock = d.parallel_lock_active();
    match which {
        202 => crate::log_user!(
            ms.log,
            LogType::Info,
            "Presolve: %s\n",
            crate::mip::driver::model_status_to_string(ms.modelstatus)
        ),
        204 => {
            let w = WorkerS::new(ms, d.get_lp(), d.get_domain(), d.get_cut_pool(), d.get_conflict_pool(), d.get_pseudo_cost());
            d.workers.push(Own::new(w));
        }
        205 => return feasibility_jump(ms) as f64,
        206 => d.get_cut_pool().perform_aging(),
        207 => return d.workers.len() as f64,
        208 => return ms.get_max_num_workers() as f64,
        209 => d.run = Some(Box::new(RunCtx { tg: TaskGroup::new() })),
        210 => d.run = None,
        211 => {
            if d.workers.len() <= 1 {
                return 0.0;
            }
            // (each member's own order of destruction, from the back)
            d.domains.truncate(1);
            d.lps.truncate(1);
            d.pseudocosts.truncate(1);
            d.workers.truncate(1);
            d.cutpools.truncate(1);
            d.conflictpools.truncate(1);
        }
        212 => create_new_workers(ms, i as i32),
        213 => {
            let age = ms.opts.mip_pool_age_limit;
            let soft = ms.opts.mip_pool_soft_limit;
            debug_assert!(d.cutpools.len() == 1 && d.conflictpools.len() == 1);
            d.cutpools.push(Own::new(CutPoolS::new(ms.num_col(), age, soft, 1)));
            let worker = d.worker(0);
            worker.cutpool = d.cutpools.last().unwrap().ptr();
            d.conflictpools.push(Own::new(ConflictPoolS::new(5 * age, soft)));
            worker.conflictpool = d.conflictpools.last().unwrap().ptr();
            d.domains.push(Own::new(DomS::copy(d.get_domain())));
            worker.globaldom = d.domains.last().unwrap().ptr();
            let gd = worker.get_global_domain();
            gd.add_cutpool(worker.get_cut_pool());
            gd.add_conflict_pool(worker.get_conflict_pool());
            d.pseudocosts.push(Own::new(Box::new(PscostS::new(ms))));
            worker.pseudocost = d.pseudocosts.last().unwrap().ptr();
            let wp: *mut WorkerS = worker;
            worker.get_lp().set_mip_worker(wp);
            worker.reset_search();
            worker.reset_sepa();
            worker.nodequeue.clear();
            worker.nodequeue.set_num_col(ms.num_col());
        }
        215 => {
            if !d.has_multiple_workers() || lock {
                return 0.0;
            }
            for k in 0..i as usize {
                let w = d.worker(k);
                w.get_conflict_pool().sync_conflict_pool(d.get_conflict_pool());
                w.get_cut_pool().sync_cut_pool(ms, d.get_cut_pool());
            }
            d.get_cut_pool().perform_aging();
            d.get_conflict_pool().perform_aging(false);
        }
        218 => {
            let cols = d.get_domain().changed_cols().to_vec();
            for col in cols {
                d.implications.cleanup_varbounds(col);
            }
            d.get_domain().set_domain_change_stack(&[], None);
            if i == 0 {
                d.worker(0).search().reset_local_domain();
            }
        }
        219 => d.get_domain().clear_changed_cols(),
        220 => {
            if !d.has_multiple_workers() {
                return 0.0;
            }
            for k in 0..d.workers.len() {
                d.get_pseudo_cost().flush(d.worker(k).get_pseudocost(), false);
            }
        }
        221 => {
            if !d.has_multiple_workers() {
                return 0.0;
            }
            let n = i as usize;
            let tg: *const TaskGroup = &d.run.as_ref().unwrap().tg;
            ms.set_parallel_lock(false);
            let dp = SendPtr(d as *mut SolverData);
            let spawn = n > 1 && !ms.opts.mip_search_simulate_concurrency;
            for k in 0..n {
                let f = move || {
                    let dp = dp;
                    // SAFETY: each task syncs its own worker's pseudocosts
                    let d = unsafe { &*dp.0 };
                    d.get_pseudo_cost().flush(d.worker(k).get_pseudocost(), true);
                };
                if spawn {
                    (*tg).spawn(f);
                } else {
                    f();
                }
            }
            if spawn {
                (*tg).task_wait();
            }
            ms.set_parallel_lock(false);
        }
        222 => {
            let w = d.worker(0);
            w.reset_search();
            w.reset_sepa();
            w.nodequeue.clear();
            w.nodequeue.set_num_col(ms.num_col());
            let s = w.st();
            s.upper_bound = d.sc.upper_bound;
            s.upper_limit = d.sc.upper_limit;
            s.optimality_limit = d.sc.optimality_limit;
        }
        // debugSolution.registerDomain (no debug solution)
        223 => {}
        224 => return d.worker(i as usize).search().stats().nnodes as f64,
        225 => return d.worker(i as usize).search().stats().nleaves as f64,
        226 => {
            let tw = d.worker(i as usize).search().stats().treeweight;
            return if x == 0.0 { tw.hi } else { tw.lo };
        }
        227 => return d.worker(i as usize).search().has_node() as i32 as f64,
        229 => d.worker(i as usize).st().heuristics_allowed = x != 0.0,
        230 => {
            let w = d.worker(i as usize);
            let mut p = std::mem::MaybeUninit::<crate::mip::nodequeue::CPopped>::uninit();
            if x == 0.0 {
                crate::mip::nodequeue::ffi::highs_rs_nodequeue_pop(&mut *d.nodequeue, true, p.as_mut_ptr());
                let p = p.assume_init();
                install(w, &p);
                return 1.0;
            }
            let bb_size = d.nodequeue.best_bound_domchg_stack_size();
            let bb_lb = d.nodequeue.best_lower_bound();
            crate::mip::nodequeue::ffi::highs_rs_nodequeue_pop(&mut *d.nodequeue, false, p.as_mut_ptr());
            let p = p.assume_init();
            let is_best = p.lower_bound == bb_lb && p.num_domchgs == bb_size;
            install(w, &p);
            return is_best as i32 as f64;
        }
        231 => return d.worker(i as usize).search().get_current_estimate(),
        232 => {
            if !lock {
                prof.start(prof::EVALUATE_NODE1);
            }
            let w = d.worker(i as usize);
            let r = if w.search().evaluate_node() == crate::mip::search::SUB_OPTIMAL {
                let q: *mut crate::mip::nodequeue::NodeQueue = if lock { &mut *w.nodequeue } else { &mut *d.nodequeue };
                w.search().current_node_to_queue(q);
                1.0
            } else {
                0.0
            };
            if !lock {
                prof.stop(prof::EVALUATE_NODE1);
            }
            return r;
        }
        233 => {
            if !lock {
                prof.start(prof::NODE_PRUNED_LOOP);
            }
            let w = d.worker(i as usize);
            let mut pruned = false;
            let s = w.search();
            if s.current_node_pruned() {
                s.backtrack(true);
                w.get_global_domain().propagate();
                pruned = true;
                s.stats().nnodes += 1;
                s.stats().nleaves += 1;
            }
            if !lock {
                prof.stop(prof::NODE_PRUNED_LOOP);
            }
            return (w.get_global_domain().infeasible || pruned) as i32 as f64;
        }
        234 => return d.worker(i as usize).search().check_local_limits() as i32 as f64,
        235 => {
            let w = d.worker(i as usize);
            if ms.opts.mip_allow_cut_separation_at_nodes {
                if !lock {
                    prof.start(prof::NODE_SEARCH_SEPARATION);
                }
                let ld: *mut DomS = &mut *w.search().localdom;
                w.sepa().separate(&mut *ld);
                if !lock {
                    prof.stop(prof::NODE_SEARCH_SEPARATION);
                }
            } else {
                w.get_cut_pool().perform_aging();
            }
            if w.get_global_domain().infeasible {
                w.search().cutoff_node();
                let q: *mut crate::mip::nodequeue::NodeQueue = if lock { &mut *w.nodequeue } else { &mut *d.nodequeue };
                w.search().open_nodes_to_queue(q);
                return 1.0;
            }
            let lp = w.get_lp();
            let st = lp.status();
            if st != crate::mip::lp_relaxation::ERROR && st != crate::mip::lp_relaxation::NOT_SET {
                lp.store_basis();
            }
            let basis = lp.get_stored_basis();
            let consistent = basis.as_ref().is_some_and(|b| lp.is_basis_consistent(b));
            if !consistent {
                let mut b = d.firstrootbasis.clone();
                b.b.row_status.resize(lp.num_rows() as usize, 1);
                lp.set_stored_basis(Some(Arc::new(b)));
            }
        }
        236 => d.worker(i as usize).get_conflict_pool().perform_aging(false),
        237 => return d.get_lp().avg_solve_iters(),
        238 => d.worker(i as usize).get_lp().set_iteration_limit(x as i32),
        239 => return d.worker(i as usize).st().heuristics_allowed as i32 as f64,
        240 => {
            let w = d.worker(i as usize);
            if !lock {
                prof.start(prof::DIVE_EVALUATE_NODE);
            }
            let r = w.search().evaluate_node();
            if !lock {
                prof.stop(prof::DIVE_EVALUATE_NODE);
            }
            if r == crate::mip::search::SUB_OPTIMAL {
                return 1.0;
            }
            if w.search().current_node_pruned() {
                w.search().stats().nleaves += 1;
                return 2.0;
            }
            if !lock {
                prof.start(prof::DIVE_PRIMAL_HEURISTICS);
            }
        }
        241 => {
            let w = d.worker(i as usize);
            let wp = w as *mut WorkerS as P;
            let sol = w.get_lp().col_value().to_vec();
            if x == 0.0 {
                if !lock {
                    prof.start(prof::DIVE_RANDOMIZED_ROUNDING);
                }
                heur(ms, wp, 6, Some(&sol));
                if !lock {
                    prof.stop(prof::DIVE_RANDOMIZED_ROUNDING);
                }
            } else if x == 1.0 {
                if !lock {
                    prof.start(prof::DIVE_RENS);
                }
                heur(ms, wp, 1, Some(&sol));
                if !lock {
                    prof.stop(prof::DIVE_RENS);
                }
            } else if x == 2.0 {
                if !lock {
                    prof.start(prof::DIVE_RINS);
                }
                heur(ms, wp, 2, Some(&sol));
                if !lock {
                    prof.stop(prof::DIVE_RINS);
                }
            } else if !lock {
                prof.stop(prof::DIVE_PRIMAL_HEURISTICS);
            }
        }
        242 => return d.worker(i as usize).get_global_domain().infeasible as i32 as f64,
        243 => {
            let w = d.worker(i as usize);
            let node_lim = x as i32;
            let s = w.search();
            let dive_node_lim = if node_lim == IINF { i64::MAX } else { s.stats().nnodes + node_lim as i64 };
            if !s.current_node_pruned() {
                if !lock {
                    prof.start(prof::THE_DIVE);
                }
                let r = s.dive(dive_node_lim);
                if !lock {
                    prof.stop(prof::THE_DIVE);
                }
                if r == crate::mip::search::SUB_OPTIMAL {
                    return 1.0;
                }
                s.stats().nleaves += 1;
            }
            return (node_lim != IINF && s.stats().nnodes >= dive_node_lim) as i32 as f64;
        }
        244 => {
            let s = d.worker(i as usize).search();
            return s.check_limits(s.stats().nnodes) as i32 as f64;
        }
        245 => {
            let w = d.worker(i as usize);
            if !lock {
                prof.start(prof::BACKTRACK_PLUNGE);
            }
            let q: *mut crate::mip::nodequeue::NodeQueue = if lock { &mut *w.nodequeue } else { &mut *d.nodequeue };
            let bp = w.search().backtrack_plunge(q);
            if !lock {
                prof.stop(prof::BACKTRACK_PLUNGE);
            }
            if !bp {
                return 1.0;
            }
            if w.get_conflict_pool().num_conflicts() > ms.opts.mip_pool_soft_limit {
                w.get_conflict_pool().perform_aging(false);
            }
        }
        246 => d.worker(i as usize).search().flush_statistics(),
        248 => {
            let w = d.worker(i as usize);
            let infeasible = w.get_global_domain().infeasible;
            prof.start(prof::OPEN_NODES_TO_QUEUE0);
            w.search().open_nodes_to_queue(&mut *d.nodequeue);
            while w.nodequeue.num_nodes() > 0 {
                let mut p = std::mem::MaybeUninit::<crate::mip::nodequeue::CPopped>::uninit();
                crate::mip::nodequeue::ffi::highs_rs_nodequeue_pop(&mut *w.nodequeue, false, p.as_mut_ptr());
                let p = p.assume_init();
                let st = crate::ffi::sl(p.domchgstack, p.num_domchgs).to_vec();
                let br = crate::ffi::sl(p.branchings, p.num_branchings).to_vec();
                d.nodequeue.emplace_node(&st, &br, p.lower_bound, p.estimate, p.depth);
            }
            prof.stop(prof::OPEN_NODES_TO_QUEUE0);
            w.search().flush_statistics();
            // syncSepaStats
            d.cliquetable.num_neighbourhood_queries += w.st().num_neighbourhood_queries;
            d.sc.sepa_lp_iterations += w.st().sepa_lp_iterations;
            d.sc.total_lp_iterations += w.st().sepa_lp_iterations;
            w.reset_sepa_stats();
            flush_heur_statistics(ms, w);
            return infeasible as i32 as f64;
        }
        249 => return prune_infeasible_nodes(ms),
        250 => d.get_pseudo_cost().remove_changed(),
        251 => {
            let rl = d.vecs.rootlpsol.as_slice().to_vec();
            let wp = d.worker(0) as *mut WorkerS as P;
            graph_lns(ms, wp, &rl, true, x as i64);
            flush_heur_statistics(ms, d.worker(0));
        }
        255 => {
            if i == 0 {
                return d.terminator_active() as i32 as f64;
            }
            if i == 1 {
                return d.terminator_terminated() as i32 as f64;
            }
            d.terminator_terminate();
        }
        257 => return d.worker(i as usize).st().upper_limit,
        258 => return ms.termination_status as f64,
        259 => ms.timer.stop(0),
        260 => {
            let log = &ms.log;
            let call_record = |clock: i32| {
                let mip_time = prof.read(clock, prof::MIP_RECORD);
                let submip_time = prof.read(clock, prof::SUB_MIP_RECORD);
                let mip_calls = prof.num_call(clock, prof::MIP_RECORD);
                let submip_calls = prof.num_call(clock, prof::SUB_MIP_RECORD);
                let total_time = mip_time + submip_time;
                let name = match clock {
                    prof::PRESOLVE_TIME => "Presolve",
                    prof::SOLVE_TIME => "Solve",
                    _ => "Postsolve",
                };
                crate::log_user!(log, LogType::Info, "                    %.2f (%s)\n", total_time, name);
                if mip_calls > 1 || submip_calls > 0 {
                    crate::log_user!(
                        log,
                        LogType::Info,
                        "                        MIP    time [calls] = %.2f [%d]\n",
                        mip_time,
                        mip_calls
                    );
                    if submip_calls > 0 {
                        crate::log_user!(
                            log,
                            LogType::Info,
                            "                        subMIP time [calls] = %.2f [%d]\n",
                            submip_time,
                            submip_calls
                        );
                    }
                }
            };
            let total = ms.timer.read(0);
            crate::log_user!(log, LogType::Info, "  Timing            %.2f\n", total);
            call_record(prof::PRESOLVE_TIME);
            call_record(prof::SOLVE_TIME);
            call_record(prof::POSTSOLVE_TIME);
        }
        261 => {
            if ms.improving_file_open {
                if let Some(f) = super::highs_fns() {
                    (f.improving_file)(ms.host, 2, std::ptr::null(), 0);
                }
                ms.improving_file_open = false;
            }
        }
        _ => unreachable!("driver op {which}"),
    }
    0.0
}

fn install(w: &WorkerS, p: &crate::mip::nodequeue::CPopped) {
    // SAFETY: the popped node's arrays, valid until the next pop
    let (st, br) = unsafe { (crate::ffi::sl(p.domchgstack, p.num_domchgs).to_vec(), crate::ffi::sl(p.branchings, p.num_branchings).to_vec()) };
    w.search().install_node(&st, &br, p.lower_bound, p.estimate, p.depth);
}

/// createNewWorkers(numNewWorkers)
fn create_new_workers(ms: &mut MipSolver, num_new_workers: i32) {
    if num_new_workers <= 0 {
        return;
    }
    let d = ms.d();
    let age = ms.opts.mip_pool_age_limit;
    let soft = ms.opts.mip_pool_soft_limit;
    // remove all cuts from non-global pool for copied LP
    let lp = LpS::copy(d.get_lp());
    lp.set_profiling(ms.prof.p);
    lp.remove_worker_specific_rows();
    d.lps.push(Own::new(lp));
    for k in 0..num_new_workers {
        if k != 0 {
            let lp = LpS::copy(d.lps.last().unwrap());
            lp.set_profiling(ms.prof.p);
            d.lps.push(Own::new(lp));
        }
        d.domains.push(Own::new(DomS::copy(d.get_domain())));
        d.cutpools.push(Own::new(CutPoolS::new(ms.num_col(), age, soft, d.cutpools.len() as i32)));
        d.conflictpools.push(Own::new(ConflictPoolS::new(5 * age, soft)));
        let dom: *mut DomS = d.domains.last().unwrap().ptr();
        let cp: *mut CutPoolS = d.cutpools.last().unwrap().ptr();
        let cfp: *mut ConflictPoolS = d.conflictpools.last().unwrap().ptr();
        // SAFETY: the new worker's objects
        unsafe {
            (*dom).add_cutpool(&mut *cp);
            (*dom).add_conflict_pool(&mut *cfp);
        }
        d.pseudocosts.push(Own::new(Box::new(PscostS::new(ms))));
        let ps: *mut PscostS = d.pseudocosts.last().unwrap().ptr();
        let lpp: *mut LpS = d.lps.last().unwrap().ptr();
        let w = WorkerS::new(ms, lpp, dom, cp, cfp, ps);
        d.workers.push(Own::new(w));
        let wp: *mut WorkerS = d.workers.last().unwrap().ptr();
        // SAFETY: the new worker and its LP
        unsafe {
            (*lpp).set_mip_worker(wp);
            (*lpp).notify_cut_pools_lp_copied(1);
            (*wp).st().randgen = crate::util::random::HighsRandom::new((ms.opts.random_seed + d.workers.len() as i32 - 1) as u32);
            (*wp).nodequeue.set_num_col(ms.num_col());
        }
    }
}

/// pruneInfeasibleNodes(globaldom, feastol): the pruned tree weight
fn prune_infeasible_nodes(ms: &MipSolver) -> f64 {
    let d = ms.d();
    let gd = d.get_domain();
    let feastol = d.sc.feastol;
    let mut treeweight = CDouble::from(0.0);
    let num_col = gd.col_lower().len();
    loop {
        if gd.infeasible {
            break;
        }
        let numchgs = gd.stack().len();
        // SAFETY: the queue and the domain's bounds
        unsafe {
            crate::mip::nodequeue::ffi::highs_rs_nodequeue_check_global_bounds(
                &mut *d.nodequeue,
                gd.col_lower().as_ptr(),
                gd.col_upper().as_ptr(),
                feastol,
                &mut treeweight,
            )
        };
        if d.nodequeue.num_nodes() == 0 {
            break;
        }
        for i in 0..num_col {
            if let Some(globallb) = d.nodequeue.common_bound(i as i32, true) {
                if globallb > gd.col_lower()[i] {
                    gd.change_bound(
                        DomChg { boundval: globallb, column: i as i32, boundtype: crate::mip::domain::LOWER },
                        Reason::UNSPECIFIED,
                    );
                    if gd.infeasible {
                        break;
                    }
                }
            }
            if let Some(globalub) = d.nodequeue.common_bound(i as i32, false) {
                if globalub < gd.col_upper()[i] {
                    gd.change_bound(
                        DomChg { boundval: globalub, column: i as i32, boundtype: crate::mip::domain::UPPER },
                        Reason::UNSPECIFIED,
                    );
                    if gd.infeasible {
                        break;
                    }
                }
            }
        }
        gd.propagate();
        if numchgs == gd.stack().len() {
            break;
        }
    }
    treeweight.to_f64()
}

/// feasibilityJump (the model status)
fn feasibility_jump(ms: &MipSolver) -> i32 {
    let d = ms.d();
    let model = ms.model();
    let feastol = d.sc.feastol;
    let n = model.num_col as usize;
    let sense = model.sense as f64;
    let incumbent = d.vecs.incumbent.as_slice();
    let use_incumbent = !incumbent.is_empty();
    let mut col_value = vec![0.0; n];
    let (mut lo, mut up, mut cost) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let mut integer = vec![0u8; n];
    for col in 0..n {
        let mut lower = model.col_lower[col];
        let mut upper = model.col_upper[col];
        let is_int = model.integrality[col] != 0;
        if is_int {
            lower = (lower - feastol).ceil();
            upper = (upper + feastol).floor();
        }
        let legal = lower <= upper && lower < INF && upper > -INF && !lower.is_nan() && !upper.is_nan();
        if !legal {
            crate::log_user!(
                ms.log,
                LogType::Info,
                "HighsMipSolverData::feasibilityJump() has detected infeasible/illegal bounds [%g, %g] for column %d: MIP is infeasible\n",
                lower,
                upper,
                col as i32
            );
            return 8;
        }
        lo[col] = lower;
        up[col] = upper;
        cost[col] = sense * model.col_cost[col];
        integer[col] = is_int as u8;
        col_value[col] = if use_incumbent && incumbent[col].is_finite() {
            max2(lower, super::min2(upper, incumbent[col]))
        } else if lower.is_finite() {
            lower
        } else if upper.is_finite() {
            upper
        } else {
            0.0
        };
    }
    // the row-wise matrix (createRowwise)
    let m = model.num_row as usize;
    let mut ar_start = vec![0i32; m + 1];
    for &r in &model.a.index[..model.a.start[n] as usize] {
        ar_start[r as usize + 1] += 1;
    }
    for r in 0..m {
        ar_start[r + 1] += ar_start[r];
    }
    let nnz = ar_start[m] as usize;
    let mut put: Vec<i32> = ar_start[..m].to_vec();
    let (mut ar_index, mut ar_value) = (vec![0i32; nnz], vec![0.0; nnz]);
    for c in 0..n {
        for k in model.a.start[c] as usize..model.a.start[c + 1] as usize {
            let r = model.a.index[k] as usize;
            let p = put[r] as usize;
            put[r] += 1;
            ar_index[p] = c as i32;
            ar_value[p] = model.a.value[k];
        }
    }
    let inp = crate::mip::feasjump::Input {
        col_lower: &lo,
        col_upper: &up,
        col_cost: &cost,
        col_integer: &integer,
        ar_start: &ar_start,
        ar_index: &ar_index,
        ar_value: &ar_value,
        row_lower: &model.row_lower,
        row_upper: &model.row_upper,
        seed: ms.opts.random_seed as u32,
        equality_tolerance: d.sc.epsilon,
        violation_tolerance: feastol,
        max_total_effort: nnz << 10,
        max_effort_since_improvement: nnz << 8,
    };
    let logging_on = ms.opts.output_flag && ms.opts.log_dev_level != 0;
    let log = ms.log;
    let mut cb = |t: i32, msg: &str| {
        let t = match t {
            2 => LogType::Detailed,
            3 => LogType::Verbose,
            4 => LogType::Warning,
            5 => LogType::Error,
            _ => LogType::Info,
        };
        log.dev(t, msg);
    };
    if crate::mip::feasjump::solve(&inp, &mut col_value, logging_on, &mut cb) {
        try_solution(ms, &col_value, crate::mip::glue::source::FEASIBILITY_JUMP);
    }
    MODEL_STATUS_NOTSET
}

/// the setup's operations of setup.rs (mod op) and workers.rs
unsafe fn setup_op(ms: &mut MipSolver, which: i32, _w: P, i: i64, _x: f64) -> f64 {
    let d = ms.d();
    match which {
        300 => {
            d.postsolve_stack.initialize_index_maps(ms.num_row(), ms.num_col());
            ms.orig_model = ms.model;
        }
        301 => {
            if !ms.clqtableinit.is_null() {
                let orig = ms.orig();
                crate::mip::clique_ffi::highs_rs_clique_build_from(
                    &mut *d.cliquetable,
                    orig.col_lower.as_ptr(),
                    orig.col_upper.as_ptr(),
                    orig.num_col,
                    ms.clqtableinit,
                );
            }
            d.cliquetable.min_entries_for_parallelism =
                if crate::parallel::num_threads() > 1 { ms.opts.mip_min_cliquetable_entries_for_parallelism } else { IINF };
            if !ms.implicinit.is_null() {
                d.implications.build_from(&*ms.implicinit);
            }
        }
        302 => {
            if i == 0 {
                ms.timer.start(2);
            } else {
                ms.timer.stop(2);
            }
        }
        303 => super::presolve::run_mip_presolve(ms, i as i32),
        304 => {
            let orig = ms.orig();
            let model = ms.model();
            let nz = |m: &Lp| m.a.start.get(m.num_col as usize).copied().unwrap_or(0);
            let from = [orig.num_col, orig.num_row, nz(orig)];
            let reduced = d.presolve_status == crate::lp_data::run::PS_REDUCED || d.presolve_status == crate::lp_data::run::PS_TIMEOUT;
            let to = if reduced { [model.num_col, model.num_row, nz(model)] } else { [0, 0, 0] };
            crate::lp_data::run::report_presolve_reductions(&ms.log, ms.opts.output_flag, d.presolve_status, from, to);
        }
        305 => d.get_lp().set_solved_first_lp(false),
        306 => {
            let x = super::presolve::reduced_primal(&d.postsolve_stack, &ms.solution);
            d.vecs.set_f64(crate::mip::mip_data::vec::INCUMBENT, &x);
        }
        307 => {
            *d.redcostfixing = crate::mip::redcost::RedcostFixing::new();
            let p = PscostS::new(ms);
            d.get_pseudo_cost().set(p);
        }
        308 => {
            let ct: *mut crate::mip::clique::CliqueTable = &mut *d.cliquetable;
            d.objective_function.setup_clique_partition(ms, &mut *ct);
            let gd = d.get_domain();
            gd.setup_objective_propagation();
            gd.compute_row_activities();
            gd.propagate();
        }
        309 => {
            let cols = d.get_domain().changed_cols().to_vec();
            for col in cols {
                d.implications.cleanup_varbounds(col);
            }
            d.get_domain().clear_changed_cols();
        }
        310 => d.get_lp().set_option("presolve", OptValue::Str(b"off")),
        311 => d.objective_function.check_integrality(d.sc.epsilon),
        312 => heur(ms, std::ptr::null_mut(), 0, None),
        313 => {
            d.analytic_center_status.store(MODEL_STATUS_NOTSET, Ordering::SeqCst);
            d.vecs.set_f64(crate::mip::mip_data::vec::ANALYTIC_CENTER, &[]);
            d.symmetries.clear();
        }
        314 => return d.cliquetable.num_entries as f64,
        315 => return crate::parallel::num_threads() as f64,
        316 => return std::thread::available_parallelism().map_or(0, |n| n.get()) as f64,
        317 => {
            let init = PscostInit::new_presolved(d.get_pseudo_cost(), ms.opts.mip_pscost_minreliable, &d.postsolve_stack);
            d.restart = Some(Box::new(RestartCtx { root_basis: Basis::default(), pscostinit: init }));
            ms.pscostinit = &d.restart.as_ref().unwrap().pscostinit;
        }
        318 => {
            if let Some(r) = &d.restart {
                if std::ptr::eq(ms.rootbasis, &r.root_basis) {
                    ms.rootbasis = std::ptr::null();
                }
            }
            ms.pscostinit = std::ptr::null();
            d.restart = None;
        }
        320 => {
            let num_cuts = i as i32;
            if num_cuts > 0 {
                super::presolve::append_cuts_to_model(&mut d.postsolve_stack, num_cuts);
            }
            let integrality = std::mem::take(&mut d.presolved_model.integrality);
            let offset = d.presolved_model.offset;
            let mut m = d.get_lp().get_lp_copy();
            m.model_name = d.presolved_model.model_name.clone();
            d.presolved_model = m;
            d.presolved_model.offset = offset;
            d.presolved_model.integrality = integrality;
        }
        321 => d.global_orbits = None,
        322 => {
            return match i {
                1 => d.postsolve_stack.orig_num_col as f64,
                2 => d.postsolve_stack.orig_num_row as f64,
                _ => d.postsolve_stack.reductions.len() as f64,
            }
        }
        323 => super::presolve::remove_cuts_from_model(&mut d.postsolve_stack, i as i32),
        324 => {
            if !d.workers.is_empty() {
                let w0 = d.worker(0);
                w0.cutpool = d.get_cut_pool();
                w0.conflictpool = d.get_conflict_pool();
                w0.globaldom = d.get_domain();
                w0.pseudocost = d.get_pseudo_cost();
                let s = w0.st();
                s.upper_bound = d.sc.upper_bound;
                s.upper_limit = d.sc.upper_limit;
                s.optimality_limit = d.sc.optimality_limit;
            }
        }
        325 => d.root.as_ref().unwrap().tg.sync(),
        326 => return d.analytic_center_status.load(Ordering::SeqCst) as f64,
        327 => {
            let ctx = d.root.as_mut().unwrap();
            ctx.tg.sync();
            let sd = ctx.sym_data.as_mut().unwrap();
            d.symmetries = std::mem::take(&mut sd.symmetries);
            return sd.detection_time;
        }
        328 => {
            let s = &d.symmetries;
            return match i {
                0 => s.num_generators as f64,
                1 => s.num_perms as f64,
                2 => s.orbitopes.len() as f64,
                _ => s.column_to_orbitope.len() as f64,
            };
        }
        329 => {
            d.root.as_mut().unwrap().sym_data = None;
            d.symmetries.determine_orbitope_types(&mut d.cliquetable);
            if d.symmetries.num_perms != 0 {
                let sym: *const Symmetries = &d.symmetries;
                d.global_orbits = Some(Arc::new(StabS::compute(&*sym, d.get_domain())));
            }
        }
        330 => ms.saved_objective_and_solution.push((ms.solution_objective, ms.solution.clone())),
        331 => {
            if ms.improving_file_open {
                if let Some(f) = super::highs_fns() {
                    (f.improving_file)(ms.host, 1, ms.solution.as_ptr(), ms.solution.len() as i32);
                }
            }
        }
        332 => {
            let mut len = 0;
            let us = match super::highs_fns() {
                Some(f) if !ms.host.is_null() => (f.user_solution)(ms.host, &mut len),
                _ => std::ptr::null(),
            };
            let user = crate::ffi::sl(us, len).to_vec();
            d.scratch.col_value = super::presolve::reduced_primal(&d.postsolve_stack, &user);
        }
        333 => {
            let Some(f) = super::highs_fns() else { return 0.0 };
            if ms.host.is_null() {
                return 0.0;
            }
            return (f.callback)(ms.host, 1 + i as i32, 0, std::ptr::null(), std::ptr::null(), 0, std::ptr::null(), 0) as i32
                as f64;
        }
        334 => return std::ptr::eq(ms.model, &d.presolved_model) as i32 as f64,
        335 => {
            let p = PscostS::new(ms);
            d.get_pseudo_cost().set(p);
        }
        336 => {
            let fresh = DomS::new(ms);
            d.get_domain().assign(&fresh);
            drop(fresh);
            d.get_domain().compute_row_activities();
        }
        337 => {
            let (num_col, num_cut, lower, upper, start, index, value) = d.get_lp().get_cut_pool();
            if let Some(f) = super::highs_fns() {
                if !ms.host.is_null() {
                    (f.cut_pool_output)(
                        ms.host,
                        num_col,
                        num_cut,
                        lower.as_ptr(),
                        upper.as_ptr(),
                        start.as_ptr(),
                        index.as_ptr(),
                        value.as_ptr(),
                        index.len() as i32,
                    );
                }
            }
        }
        // workers.rs
        401 => tables::cleanup_fixed(ms, d.get_domain()),
        402 => {
            // the end of resetGlobalDomain's doResetWorkerDomain: resetting
            // the local domain cannot be done in parallel
            let w = d.worker(i as usize);
            w.get_global_domain().set_domain_change_stack(&[], None);
            w.search().reset_local_domain();
            w.get_global_domain().clear_changed_cols();
        }
        403 => ms.set_parallel_lock(i != 0),
        _ => unreachable!("setup op {which}"),
    }
    0.0
}

/// The functions of glue.rs, on the Rust objects
pub static MIP_FNS: CMipFns = CMipFns {
    dom_copy,
    dom_free,
    dom_assign,
    dom_bounds,
    dom_change_bound,
    dom_fix_col,
    dom_propagate,
    dom_infeasible,
    dom_backtrack,
    dom_conflict_analysis,
    dom_stack,
    dom_branch_depth,
    dom_clear_changed_cols,
    dom_clear_pool_propagation,
    dom_num_changed_cols,
    lp_copy,
    lp_new,
    lp_free,
    lp_rust,
    lp_set_root_basis,
    lp_resolve,
    lp_set_objective_limit,
    lp_flush_domain,
    lp_remove_obsolete_rows,
    lp_infeasible_conflict,
    search_new,
    search_free,
    search_set_lp,
    check_limits,
    update_lower_bound,
    parallel_lock_active,
    num_workers,
    worker_view,
    sub_mip,
    op,
    scratch_solution,
    refill,
    master_worker,
    run_process_nodes,
    set_cleanup_result,
    model_name,
    max_submip_level,
    helper_new,
    helper_run,
    add_root_cut,
    vec_ptr,
    set_basis,
    callback,
    worker,
    worker_scratch,
    repair_lp,
};

/// HighsMipSolver::run
pub fn run(ms: &mut MipSolver) {
    ms.modelstatus = MODEL_STATUS_NOTSET;
    ms.max_submip_level = ms.max_submip_level.max(ms.submip_level);
    ms.timer.start(0);
    ms.improving_file_open = false;
    if !ms.submip && !ms.opts.mip_improving_solution_file.is_empty() {
        if let Some(f) = super::highs_fns() {
            if !ms.host.is_null() {
                // SAFETY: the solve's host
                unsafe { (f.improving_file)(ms.host, 0, std::ptr::null(), 0) };
                ms.improving_file_open = true;
            }
        }
    }
    SolverData::create(ms);
    for lp in &ms.d().lps {
        lp.set_profiling(ms.prof.p);
    }
    glue::set_fns(&MIP_FNS);
    let mut m = ms.mip_data();
    // SAFETY: the solver's view, refilled after the model changes
    unsafe { crate::mip::driver::run(&mut m) };
}
