//! The MIP solver's entry points for the C++ Highs object
//! (HighsRunRust.cpp): a solve (Highs::callSolveMip's HighsMipSolver run)
//! and a presolve (Highs::runPresolve's runMipPresolve) of a model with the
//! Highs object's options, log, profiling and callback host, and the
//! result handed back.

use super::solver::MipSolver;
use super::{Prof, P};
use crate::lp_data::ffi::CLp;
use crate::lp_data::lp::Lp;
use crate::lp_data::opts::Opts;
use crate::lp_data::Log;

/// What the C++ passes
#[repr(C)]
pub struct MipIn {
    /// the MipHost of the solve (HighsFns' context)
    pub host: P,
    /// the Highs object's HighsProfiling
    pub profiling: P,
    /// rsLog(options.log_options)
    pub log: Log,
    /// an LP solver whose options were synced from the HighsOptions
    pub opts: *const crate::lp_data::lp_handle::LpHandle,
    /// the model (column-wise) and its name
    pub lp: *const CLp,
    pub model_name: *const u8,
    pub model_name_len: usize,
    /// the start solution, if value_valid
    pub col_value: *const f64,
    pub num_col_value: usize,
    pub row_value: *const f64,
    pub num_row_value: usize,
    pub value_valid: bool,
    /// runMipPresolve's reduction limit (presolve only)
    pub presolve_reduction_limit: i32,
}

/// The result: HighsMipSolver's fields Highs reads, the presolve's model
/// and stack (presolve only)
#[repr(C)]
pub struct MipOut {
    pub model_status: i32,
    pub solution_objective: f64,
    pub node_count: i64,
    pub total_lp_iterations: i64,
    pub dual_bound: f64,
    pub primal_bound: f64,
    pub gap: f64,
    pub primal_dual_integral: f64,
    pub row_violation: f64,
    pub bound_violation: f64,
    pub integrality_violation: f64,
    pub solution: *const f64,
    pub num_solution: usize,
    pub num_saved: usize,
    // presolve only: the status, the presolved model and the stack
    pub presolve_status: i32,
    pub presolved: CLp,
    pub presolved_name: *const u8,
    pub presolved_name_len: usize,
    pub data: *const u8,
    pub data_len: usize,
    pub reductions: *const c_void_reduction,
    pub num_reductions: usize,
    pub orig_col_index: *const i32,
    pub num_col: usize,
    pub orig_row_index: *const i32,
    pub num_row: usize,
    pub linearly_transformable: *const u8,
    pub num_lt: usize,
    pub orig_num_col: i32,
    pub orig_num_row: i32,
    /// the solver, owning what the pointers point into
    solver: *mut MipSolver,
}

#[allow(non_camel_case_types)]
pub type c_void_reduction = crate::presolve::postsolve::Reduction;

unsafe fn solver_of(i: &MipIn) -> Box<MipSolver> {
    let opts: Opts = (*i.opts).opts.clone();
    let mut lp = Lp::default();
    let name = crate::ffi::sl(i.model_name, i.model_name_len as i32).to_vec();
    lp.import(&*i.lp, &name);
    let start = if i.value_valid {
        Some((
            crate::ffi::sl(i.col_value, i.num_col_value as i32),
            crate::ffi::sl(i.row_value, i.num_row_value as i32),
        ))
    } else {
        None
    };
    // a MIP solver's log is the Highs object's (its output_flag decides)
    MipSolver::new(i.host, Prof { p: i.profiling }, opts, i.log, lp, name, start, false, 0)
}

fn out_of(ms: Box<MipSolver>) -> *mut MipOut {
    let ms = Box::into_raw(ms);
    // SAFETY: the solver, owned by the result
    let s = unsafe { &mut *ms };
    let mut o = Box::new(MipOut {
        model_status: s.modelstatus,
        solution_objective: s.solution_objective,
        node_count: s.node_count,
        total_lp_iterations: s.total_lp_iterations,
        dual_bound: s.dual_bound,
        primal_bound: s.primal_bound,
        gap: s.gap,
        primal_dual_integral: s.primal_dual_integral,
        row_violation: s.row_violation,
        bound_violation: s.bound_violation,
        integrality_violation: s.integrality_violation,
        solution: s.solution.as_ptr(),
        num_solution: s.solution.len(),
        num_saved: s.saved_objective_and_solution.len(),
        presolve_status: 0,
        // SAFETY: plain data, filled below if presolved
        presolved: unsafe { std::mem::zeroed() },
        presolved_name: std::ptr::null(),
        presolved_name_len: 0,
        data: std::ptr::null(),
        data_len: 0,
        reductions: std::ptr::null(),
        num_reductions: 0,
        orig_col_index: std::ptr::null(),
        num_col: 0,
        orig_row_index: std::ptr::null(),
        num_row: 0,
        linearly_transformable: std::ptr::null(),
        num_lt: 0,
        orig_num_col: 0,
        orig_num_row: 0,
        solver: ms,
    });
    if s.mipdata.is_some() {
        let d = s.d();
        o.presolve_status = d.presolve_status;
        o.presolved = d.presolved_model.view();
        o.presolved_name = d.presolved_model.model_name.as_ptr();
        o.presolved_name_len = d.presolved_model.model_name.len();
        let st = &d.postsolve_stack;
        o.data = st.data.as_ptr();
        o.data_len = st.data.len();
        o.reductions = st.reductions.as_ptr();
        o.num_reductions = st.reductions.len();
        o.orig_col_index = st.orig_col_index.as_ptr();
        o.num_col = st.orig_col_index.len();
        o.orig_row_index = st.orig_row_index.as_ptr();
        o.num_row = st.orig_row_index.len();
        o.linearly_transformable = st.linearly_transformable.as_ptr();
        o.num_lt = st.linearly_transformable.len();
        o.orig_num_col = st.orig_num_col;
        o.orig_num_row = st.orig_num_row;
    }
    Box::into_raw(o)
}

/// Highs::callSolveMip's solve: the MIP solver run on the model
///
/// # Safety
/// The input's pointers valid for the call; the host's and profiling's
/// C++ objects alive until the result is freed
#[no_mangle]
pub unsafe extern "C" fn highs_rs_mip_solve(i: *const MipIn) -> *mut MipOut {
    let i = &*i;
    let mut ms = solver_of(i);
    super::fns::run(&mut ms);
    out_of(ms)
}

/// Highs::runPresolve's MIP presolve (runMipPresolve: init and presolve)
///
/// # Safety
/// as highs_rs_mip_solve
#[no_mangle]
pub unsafe extern "C" fn highs_rs_mip_presolve(i: *const MipIn) -> *mut MipOut {
    let i = &*i;
    let mut ms = solver_of(i);
    ms.timer.start(0);
    super::solver::SolverData::create(&mut ms);
    crate::mip::glue::set_fns(&super::fns::MIP_FNS);
    let m = ms.mip_data();
    m.init_rs();
    m.run_mip_presolve(i.presolve_reduction_limit);
    out_of(ms)
}

/// The saved improving solution k: its objective, values and length
///
/// # Safety
/// `o` a live result, k < num_saved
#[no_mangle]
pub unsafe extern "C" fn highs_rs_mip_saved(o: *const MipOut, k: usize, objective: *mut f64, n: *mut usize) -> *const f64 {
    let s = &*(*o).solver;
    let (obj, v) = &s.saved_objective_and_solution[k];
    *objective = *obj;
    *n = v.len();
    v.as_ptr()
}

/// Frees a result and its solver
///
/// # Safety
/// `o` from highs_rs_mip_solve / presolve
#[no_mangle]
pub unsafe extern "C" fn highs_rs_mip_out_free(o: *mut MipOut) {
    let o = Box::from_raw(o);
    drop(Box::from_raw(o.solver));
}
