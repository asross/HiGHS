//! `extern "C"` shims of HighsSymmetry (highs/presolve/HighsSymmetry.cpp
//! under HIGHS_RUST)

use super::symmetry::{ModelView, StabilizerOrbits, SymDom, SymmetryDetection, Symmetries};
use crate::ffi::sl;
use crate::mip::clique::CliqueTable;
use std::ffi::c_void;

#[no_mangle]
pub extern "C" fn highs_rs_symdet_new() -> *mut SymmetryDetection {
    Box::into_raw(Box::default())
}

/// # Safety
/// `d` from highs_rs_symdet_new, or null
#[no_mangle]
pub unsafe extern "C" fn highs_rs_symdet_free(d: *mut SymmetryDetection) {
    if !d.is_null() {
        drop(Box::from_raw(d));
    }
}

/// loadModelAsGraph of a column-wise model (`nnz` = index_.size())
///
/// # Safety
/// live detection, arrays of the given lengths
#[no_mangle]
pub unsafe extern "C" fn highs_rs_symdet_load(
    d: *mut SymmetryDetection,
    num_col: i32,
    num_row: i32,
    a_start: *const i32,
    a_index: *const i32,
    a_value: *const f64,
    nnz: i32,
    col_cost: *const f64,
    col_lower: *const f64,
    col_upper: *const f64,
    integrality: *const u8,
    row_lower: *const f64,
    row_upper: *const f64,
    epsilon: f64,
) {
    let model = ModelView {
        num_col,
        num_row,
        a_start: sl(a_start, num_col + 1),
        a_index: sl(a_index, nnz),
        a_value: sl(a_value, nnz),
        col_cost: sl(col_cost, num_col),
        col_lower: sl(col_lower, num_col),
        col_upper: sl(col_upper, num_col),
        integrality: sl(integrality, num_col),
        row_lower: sl(row_lower, num_row),
        row_upper: sl(row_upper, num_row),
    };
    (*d).load_model_as_graph(&model, epsilon);
}

/// # Safety
/// live detection
#[no_mangle]
pub unsafe extern "C" fn highs_rs_symdet_init(d: *mut SymmetryDetection) -> bool {
    (*d).initialize_detection()
}

/// run; false when `interrupted(ctx)` stopped it
///
/// # Safety
/// live detection and symmetries
#[no_mangle]
pub unsafe extern "C" fn highs_rs_symdet_run(
    d: *mut SymmetryDetection,
    s: *mut Symmetries,
    ctx: *mut c_void,
    interrupted: unsafe extern "C" fn(*mut c_void) -> bool,
) -> bool {
    (*d).run(&mut *s, || interrupted(ctx))
}

#[no_mangle]
pub extern "C" fn highs_rs_sym_new() -> *mut Symmetries {
    Box::into_raw(Box::default())
}

/// # Safety
/// `s` from highs_rs_sym_new, or null
#[no_mangle]
pub unsafe extern "C" fn highs_rs_sym_free(s: *mut Symmetries) {
    if !s.is_null() {
        drop(Box::from_raw(s));
    }
}

/// # Safety
/// live symmetries
#[no_mangle]
pub unsafe extern "C" fn highs_rs_sym_clear(s: *mut Symmetries) {
    (*s).clear();
}

/// `which`: 0 numPerms, 1 numGenerators, 2 orbitopes.size(),
/// 3 columnToOrbitope.size()
///
/// # Safety
/// live symmetries
#[no_mangle]
pub unsafe extern "C" fn highs_rs_sym_get(s: *const Symmetries, which: i32) -> i32 {
    let s = &*s;
    match which {
        0 => s.num_perms,
        1 => s.num_generators,
        2 => s.orbitopes.len() as i32,
        _ => s.column_to_orbitope.len() as i32,
    }
}

/// # Safety
/// live symmetries, `n` bounds each
#[no_mangle]
pub unsafe extern "C" fn highs_rs_sym_branching_column(
    s: *const Symmetries,
    col_lower: *const f64,
    col_upper: *const f64,
    n: i32,
    col: i32,
) -> i32 {
    (*s).get_branching_column(sl(col_lower, n), sl(col_upper, n), col)
}

/// # Safety
/// live symmetries and domain
#[no_mangle]
pub unsafe extern "C" fn highs_rs_sym_propagate_orbitopes(s: *const Symmetries, dom: *const SymDom) -> i32 {
    (*s).propagate_orbitopes(&*dom)
}

/// # Safety
/// live symmetries and clique table
#[no_mangle]
pub unsafe extern "C" fn highs_rs_sym_determine_orbitope_types(s: *mut Symmetries, t: *mut CliqueTable) {
    (*s).determine_orbitope_types(&mut *t);
}

/// computeStabilizerOrbits; free the result with highs_rs_stab_free
///
/// # Safety
/// live symmetries and domain
#[no_mangle]
pub unsafe extern "C" fn highs_rs_sym_stabilizer_orbits(
    s: *const Symmetries,
    dom: *const SymDom,
) -> *mut StabilizerOrbits {
    Box::into_raw(Box::new((*s).compute_stabilizer_orbits(&*dom)))
}

/// `which`: 0 orbitCols, 1 orbitStarts, 2 stabilizedCols
///
/// # Safety
/// live result
#[no_mangle]
pub unsafe extern "C" fn highs_rs_stab_vec(o: *const StabilizerOrbits, which: i32, len: *mut i32) -> *const i32 {
    let o = &*o;
    let v = match which {
        0 => &o.orbit_cols,
        1 => &o.orbit_starts,
        _ => &o.stabilized_cols,
    };
    *len = v.len() as i32;
    v.as_ptr()
}

/// # Safety
/// from highs_rs_sym_stabilizer_orbits
#[no_mangle]
pub unsafe extern "C" fn highs_rs_stab_free(o: *mut StabilizerOrbits) {
    drop(Box::from_raw(o));
}

/// StabilizerOrbits::orbitalFixing
///
/// # Safety
/// live symmetries and domain, arrays of the given lengths
#[no_mangle]
pub unsafe extern "C" fn highs_rs_sym_orbital_fixing(
    s: *const Symmetries,
    dom: *const SymDom,
    orbit_cols: *const i32,
    num_orbit_cols: i32,
    orbit_starts: *const i32,
    num_orbit_starts: i32,
) -> i32 {
    (*s).orbital_fixing(sl(orbit_cols, num_orbit_cols), sl(orbit_starts, num_orbit_starts), &*dom)
}

/// columnPosition[col] (-1 for columns no generator moves)
///
/// # Safety
/// live symmetries
#[no_mangle]
pub unsafe extern "C" fn highs_rs_sym_column_position(s: *const Symmetries, col: i32) -> i32 {
    let s = &*s;
    s.column_position.get(col as usize).copied().unwrap_or(-1)
}
