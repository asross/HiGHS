//! `extern "C"` interface of the Rust IPX, called by the C++ wrapper
//! ipx::LpSolver (highs/ipm/ipx/lp_solver_rs.cc). Pointers are NULL or
//! valid for the lengths that the IPX API documents (derived here from the
//! model dimensions), as for the C++ LpSolver.

use super::control::Hooks;
use super::lp_solver::{status_string, LpSolver};
use super::model::UserLp;
use super::{Info, Int, Parameters};
use crate::ffi::{sl, sl_mut};
use std::ffi::{c_char, c_void};

/// The solver behind an opaque handle
///
/// # Safety
/// p must come from ipx_rs_new and not be freed
unsafe fn solver<'a>(p: *mut c_void) -> &'a mut LpSolver {
    &mut *(p as *mut LpSolver)
}

/// Optional input array of n elements
unsafe fn opt<'a, T>(p: *const T, n: usize) -> Option<&'a [T]> {
    (!p.is_null()).then(|| sl(p, n as Int))
}

/// Optional output array of n elements
unsafe fn opt_mut<'a, T>(p: *mut T, n: usize) -> Option<&'a mut [T]> {
    (!p.is_null()).then(|| sl_mut(p, n as Int))
}

#[no_mangle]
pub extern "C" fn ipx_rs_new() -> *mut c_void {
    Box::into_raw(Box::new(LpSolver::new())) as *mut c_void
}

/// # Safety
/// p from ipx_rs_new (or NULL); not used afterwards
#[no_mangle]
pub unsafe extern "C" fn ipx_rs_free(p: *mut c_void) {
    if !p.is_null() {
        drop(Box::from_raw(p as *mut LpSolver));
    }
}

/// # Safety
/// p from ipx_rs_new; hooks valid for the lifetime of p
#[no_mangle]
pub unsafe extern "C" fn ipx_rs_set_hooks(p: *mut c_void, hooks: *const Hooks) {
    solver(p).set_hooks(*hooks);
}

/// # Safety
/// p from ipx_rs_new; params points to a struct ipx_parameters
#[no_mangle]
pub unsafe extern "C" fn ipx_rs_set_parameters(p: *mut c_void, params: *const Parameters) {
    solver(p).set_parameters(*params);
}

/// # Safety
/// p from ipx_rs_new; params points to a struct ipx_parameters
#[no_mangle]
pub unsafe extern "C" fn ipx_rs_get_parameters(p: *mut c_void, params: *mut Parameters) {
    *params = solver(p).get_parameters();
}

/// # Safety
/// p from ipx_rs_new
#[no_mangle]
pub unsafe extern "C" fn ipx_rs_set_timer_offset(p: *mut c_void, offset: f64) {
    solver(p).set_timer_offset(offset);
}

/// LpSolver::LoadModel
///
/// # Safety
/// p from ipx_rs_new; the arrays NULL or of the documented sizes (obj, lb,
/// ub: num_var; rhs, constr_type: num_constr; Ap: num_var+1; Ai, Ax:
/// Ap[num_var])
#[no_mangle]
pub unsafe extern "C" fn ipx_rs_load_model(
    p: *mut c_void,
    num_var: Int,
    offset: f64,
    obj: *const f64,
    lb: *const f64,
    ub: *const f64,
    num_constr: Int,
    ap: *const Int,
    ai: *const Int,
    ax: *const f64,
    rhs: *const f64,
    constr_type: *const c_char,
) -> Int {
    let any_null = obj.is_null()
        || lb.is_null()
        || ub.is_null()
        || ap.is_null()
        || ai.is_null()
        || ax.is_null()
        || rhs.is_null()
        || constr_type.is_null();
    let valid_dims = num_var > 0 && num_constr >= 0;
    let lp = if any_null || !valid_dims {
        // Model::Load reports the error before reading any array
        let e: Option<&[f64]> = (!any_null).then_some(&[]);
        UserLp {
            num_constr,
            num_var,
            ap: (!any_null).then_some(&[]),
            ai: (!any_null).then_some(&[]),
            ax: e,
            rhs: e,
            constr_type: (!any_null).then_some(&[]),
            offset,
            obj: e,
            lbuser: e,
            ubuser: e,
        }
    } else {
        let nv = num_var as usize;
        let nc = num_constr as usize;
        let ap = sl(ap, num_var + 1);
        let nz = std::cmp::max(ap[nv], 0);
        UserLp {
            num_constr,
            num_var,
            ap: Some(ap),
            ai: Some(sl(ai, nz)),
            ax: Some(sl(ax, nz)),
            rhs: opt(rhs, nc),
            constr_type: opt(constr_type as *const u8, nc),
            offset,
            obj: opt(obj, nv),
            lbuser: opt(lb, nv),
            ubuser: opt(ub, nv),
        }
    };
    solver(p).load_model(&lp)
}

/// LpSolver::LoadIPMStartingPoint
///
/// # Safety
/// p from ipx_rs_new; arrays NULL or of user dimensions (slack, y:
/// num_constr; the others num_var)
#[no_mangle]
pub unsafe extern "C" fn ipx_rs_load_ipm_starting_point(
    p: *mut c_void,
    x: *const f64,
    xl: *const f64,
    xu: *const f64,
    slack: *const f64,
    y: *const f64,
    zl: *const f64,
    zu: *const f64,
) -> Int {
    let s = solver(p);
    let info = s.get_info();
    let (nv, nc) = (info.num_var as usize, info.num_constr as usize);
    s.load_ipm_starting_point([
        opt(x, nv),
        opt(xl, nv),
        opt(xu, nv),
        opt(slack, nc),
        opt(y, nc),
        opt(zl, nv),
        opt(zu, nv),
    ])
}

/// # Safety
/// p from ipx_rs_new
#[no_mangle]
pub unsafe extern "C" fn ipx_rs_solve(p: *mut c_void) -> Int {
    solver(p).solve()
}

/// # Safety
/// p from ipx_rs_new; info points to a struct ipx_info
#[no_mangle]
pub unsafe extern "C" fn ipx_rs_get_info(p: *mut c_void, info: *mut Info) {
    *info = solver(p).get_info();
}

/// Nonzero if the HiGHS task running the solver was cancelled
///
/// # Safety
/// p from ipx_rs_new
#[no_mangle]
pub unsafe extern "C" fn ipx_rs_cancelled(p: *mut c_void) -> Int {
    solver(p).cancelled() as Int
}

/// LpSolver::GetInteriorSolution
///
/// # Safety
/// p from ipx_rs_new; arrays NULL or of user dimensions
#[no_mangle]
pub unsafe extern "C" fn ipx_rs_get_interior_solution(
    p: *mut c_void,
    x: *mut f64,
    xl: *mut f64,
    xu: *mut f64,
    slack: *mut f64,
    y: *mut f64,
    zl: *mut f64,
    zu: *mut f64,
) -> Int {
    let s = solver(p);
    let info = s.get_info();
    let (nv, nc) = (info.num_var as usize, info.num_constr as usize);
    s.get_interior_solution([
        opt_mut(x, nv),
        opt_mut(xl, nv),
        opt_mut(xu, nv),
        opt_mut(slack, nc),
        opt_mut(y, nc),
        opt_mut(zl, nv),
        opt_mut(zu, nv),
    ])
}

/// LpSolver::GetBasicSolution
///
/// # Safety
/// p from ipx_rs_new; arrays NULL or of user dimensions
#[no_mangle]
pub unsafe extern "C" fn ipx_rs_get_basic_solution(
    p: *mut c_void,
    x: *mut f64,
    slack: *mut f64,
    y: *mut f64,
    z: *mut f64,
    cbasis: *mut Int,
    vbasis: *mut Int,
) -> Int {
    let s = solver(p);
    let info = s.get_info();
    let (nv, nc) = (info.num_var as usize, info.num_constr as usize);
    s.get_basic_solution(
        opt_mut(x, nv),
        opt_mut(slack, nc),
        opt_mut(y, nc),
        opt_mut(z, nv),
        opt_mut(cbasis, nc),
        opt_mut(vbasis, nv),
    )
}

/// # Safety
/// p from ipx_rs_new
#[no_mangle]
pub unsafe extern "C" fn ipx_rs_clear_model(p: *mut c_void) {
    solver(p).clear_model();
}

/// # Safety
/// p from ipx_rs_new
#[no_mangle]
pub unsafe extern "C" fn ipx_rs_clear_ipm_starting_point(p: *mut c_void) {
    solver(p).clear_ipm_starting_point();
}

/// LpSolver::CrossoverFromStartingPoint
///
/// # Safety
/// p from ipx_rs_new; arrays NULL or of user dimensions
#[no_mangle]
pub unsafe extern "C" fn ipx_rs_crossover_from_starting_point(
    p: *mut c_void,
    x: *const f64,
    slack: *const f64,
    y: *const f64,
    z: *const f64,
) -> Int {
    let s = solver(p);
    let info = s.get_info();
    let (nv, nc) = (info.num_var as usize, info.num_constr as usize);
    s.crossover_from_starting_point(opt(x, nv), opt(slack, nc), opt(y, nc), opt(z, nv))
}

/// LpSolver::GetIterate (solver dimensions)
///
/// # Safety
/// p from ipx_rs_new; arrays NULL or of solver dimensions
#[no_mangle]
pub unsafe extern "C" fn ipx_rs_get_iterate(
    p: *mut c_void,
    x: *mut f64,
    y: *mut f64,
    zl: *mut f64,
    zu: *mut f64,
    xl: *mut f64,
    xu: *mut f64,
) -> Int {
    let s = solver(p);
    let info = s.get_info();
    let (nm, m) = (info.num_cols_solver as usize, info.num_rows_solver as usize);
    s.get_iterate([
        opt_mut(x, nm),
        opt_mut(y, m),
        opt_mut(zl, nm),
        opt_mut(zu, nm),
        opt_mut(xl, nm),
        opt_mut(xu, nm),
    ])
}

/// LpSolver::GetBasis
///
/// # Safety
/// p from ipx_rs_new; arrays NULL or of user dimensions
#[no_mangle]
pub unsafe extern "C" fn ipx_rs_get_basis(p: *mut c_void, cbasis: *mut Int, vbasis: *mut Int) -> Int {
    let s = solver(p);
    let info = s.get_info();
    let (nv, nc) = (info.num_var as usize, info.num_constr as usize);
    s.get_basis(opt_mut(cbasis, nc), opt_mut(vbasis, nv))
}

/// LpSolver::GetKKTMatrix
///
/// # Safety
/// p from ipx_rs_new; arrays NULL or of solver dimensions
#[no_mangle]
pub unsafe extern "C" fn ipx_rs_get_kkt_matrix(
    p: *mut c_void,
    aip: *mut Int,
    aii: *mut Int,
    aix: *mut f64,
    g: *mut f64,
) -> Int {
    let s = solver(p);
    let info = s.get_info();
    let nm = info.num_cols_solver as usize;
    let nz = info.num_entries_solver as usize;
    s.get_kkt_matrix(opt_mut(aip, nm + 1), opt_mut(aii, nz), opt_mut(aix, nz), opt_mut(g, nm))
}

/// LpSolver::SymbolicInvert
///
/// # Safety
/// p from ipx_rs_new; arrays NULL or of num_rows_solver elements
#[no_mangle]
pub unsafe extern "C" fn ipx_rs_symbolic_invert(p: *mut c_void, rowcounts: *mut Int, colcounts: *mut Int) -> Int {
    let s = solver(p);
    let m = s.get_info().num_rows_solver as usize;
    s.symbolic_invert(opt_mut(rowcounts, m), opt_mut(colcounts, m))
}

/// StatusString as a static NUL-terminated string
#[no_mangle]
pub extern "C" fn ipx_rs_status_string(status: Int) -> *const c_char {
    // the &'static str table plus a NUL per entry
    macro_rules! table {
        ($($s:literal),*) => {
            match status_string(status) { $($s => concat!($s, "\0"),)* _ => "unknown\0" }
        };
    }
    table!(
        "not run", "solved", "stopped", "no model", "out of memory", "internal error",
        "optimal", "imprecise", "primal infeas", "dual infeas", "time limit", "iter limit",
        "no progress", "failed", "debug"
    )
    .as_ptr() as *const c_char
}
