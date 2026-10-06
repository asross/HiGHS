//! `extern "C"` shims for HighsImplications (highs/mip/HighsImplications.cpp
//! under HIGHS_RUST).

use super::clique::CDom;
use super::implications::{CImp, ICtx, Implications};
use crate::ffi::sl;
use crate::mip::cuts::round::VarBound;
use std::ffi::c_void;

#[no_mangle]
pub extern "C" fn highs_rs_implics_new(numcol: i32, num_nonzero: i32) -> *mut Implications {
    Box::into_raw(Box::new(Implications::new(numcol, num_nonzero)))
}

/// # Safety
/// `t` from highs_rs_implics_new, or null
#[no_mangle]
pub unsafe extern "C" fn highs_rs_implics_free(t: *mut Implications) {
    if !t.is_null() {
        drop(Box::from_raw(t));
    }
}

/// The substitutions (which = 0), length in *len
///
/// # Safety
/// `t` live
#[no_mangle]
pub unsafe extern "C" fn highs_rs_implics_vec(t: *mut Implications, _which: i32, len: *mut usize) -> *mut c_void {
    let t = &mut *t;
    *len = t.substitutions.len();
    t.substitutions.as_mut_ptr() as *mut c_void
}

/// # Safety
/// `t` live
#[no_mangle]
pub unsafe extern "C" fn highs_rs_implics_vec_clear(t: *mut Implications, _which: i32) {
    (*t).substitutions.clear();
}

/// 0: getNumImplications, 1: tooManyVarBounds
///
/// # Safety
/// `t` live
#[no_mangle]
pub unsafe extern "C" fn highs_rs_implics_get(t: *const Implications, which: i32) -> i64 {
    match which {
        0 => (*t).num_implications,
        _ => (*t).too_many_var_bounds() as i64,
    }
}

/// addVUB (vlb = false) or addVLB with the given bound and integrality
///
/// # Safety
/// `t` live
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn highs_rs_implics_add_vb(
    t: *mut Implications,
    vlb: bool,
    col: i32,
    vbcol: i32,
    coef: f64,
    constant: f64,
    bound: f64,
    isint: bool,
    feastol: f64,
) {
    if vlb {
        (*t).add_vlb(col, vbcol, coef, constant, bound, isint, feastol);
    } else {
        (*t).add_vub(col, vbcol, coef, constant, bound, isint, feastol);
    }
}

/// # Safety
/// `t` live
#[no_mangle]
pub unsafe extern "C" fn highs_rs_implics_column_transformed(t: *mut Implications, col: i32, scale: f64, constant: f64) {
    (*t).column_transformed(col, scale, constant);
}

/// getBestVub (vlb = false) or getBestVlb: the column (-1 if none), the
/// bound in *vb, *bound updated
///
/// # Safety
/// live, valid pointers (col_value, col_dual of num_col)
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn highs_rs_implics_best_vb(
    t: *const Implications,
    vlb: bool,
    imp: *const CImp,
    dom: *const CDom,
    col: i32,
    col_value: *const f64,
    col_dual: *const f64,
    num_col: i32,
    bound: *mut f64,
    vb: *mut VarBound,
) -> i32 {
    let (cv, cd) = (sl(col_value, num_col), sl(col_dual, num_col));
    let (c, b) = if vlb {
        (*t).get_best_vlb(&*imp, &*dom, col, cv, cd, &mut *bound)
    } else {
        (*t).get_best_vub(&*imp, &*dom, col, cv, cd, &mut *bound)
    };
    *vb = b;
    c
}

/// cleanupVub (vlb = false) or cleanupVlb on `*vb`
///
/// # Safety
/// valid pointers
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn highs_rs_implics_cleanup_vb(
    vlb: bool,
    g: *const CDom,
    feastol: f64,
    epsilon: f64,
    col: i32,
    vbcol: i32,
    vb: *mut VarBound,
    bound: f64,
    allow_bound_changes: bool,
    redundant: *mut bool,
    infeasible: *mut bool,
) {
    let (r, i) = if vlb {
        Implications::cleanup_vlb(&*g, feastol, epsilon, col, vbcol, &mut *vb, bound, allow_bound_changes)
    } else {
        Implications::cleanup_vub(&*g, feastol, epsilon, col, vbcol, &mut *vb, bound, allow_bound_changes)
    };
    *redundant = r;
    *infeasible = i;
}

/// # Safety
/// live, valid pointers
#[no_mangle]
pub unsafe extern "C" fn highs_rs_implics_run_probing(
    t: *mut Implications,
    g: *const CDom,
    imp: *const CImp,
    col: i32,
    num_reductions: *mut i32,
) -> bool {
    ICtx::new(t, *g, *imp).run_probing(col, &mut *num_reductions)
}

/// # Safety
/// live, valid pointers
#[no_mangle]
pub unsafe extern "C" fn highs_rs_implics_cleanup_varbounds(t: *mut Implications, g: *const CDom, imp: *const CImp, col: i32) {
    ICtx::new(t, *g, *imp).cleanup_varbounds(col);
}

/// std::pair<HighsInt, double>
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FracInt {
    col: i32,
    val: f64,
}

/// separateImpliedBounds
///
/// # Safety
/// live, valid pointers
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn highs_rs_implics_separate(
    t: *mut Implications,
    g: *const CDom,
    imp: *const CImp,
    dom: *const CDom,
    fracints: *const FracInt,
    num_frac: i32,
    sol: *const f64,
    num_sol: i32,
    feastol: f64,
    thread_safe: bool,
) {
    let fr: Vec<(i32, f64)> = sl(fracints, num_frac).iter().map(|f| (f.col, f.val)).collect();
    ICtx::new(t, *g, *imp).separate_implied_bounds(&*dom, &fr, sl(sol, num_sol), feastol, thread_safe);
}

/// applyImplications (re-entrant, reads only the implications)
///
/// # Safety
/// live, valid pointers
#[no_mangle]
pub unsafe extern "C" fn highs_rs_implics_apply(t: *const Implications, dom: *const CDom, col: i32, val: i32) {
    Implications::apply_implications(t, &*dom, col, val);
}

/// # Safety
/// live, valid pointers
#[no_mangle]
pub unsafe extern "C" fn highs_rs_implics_rebuild(
    t: *mut Implications,
    g: *const CDom,
    ncols: i32,
    orig2reducedcol: *const i32,
    norig: i32,
    transformable: *const u8,
) {
    (*t).rebuild(&*g, ncols, sl(orig2reducedcol, norig), sl(transformable, ncols));
}

/// # Safety
/// live, valid pointers
#[no_mangle]
pub unsafe extern "C" fn highs_rs_implics_build_from(t: *mut Implications, g: *const CDom, init: *const Implications) {
    (*t).build_from(&*g, &*init);
}
