//! formSimplexLpBasisAndFactor and accommodateAlienBasis
//! (HighsSolution.cpp): the decisions, and an alien basis made a basis by
//! factorizing its basic columns (the Rust HFactor) and completing it with
//! the logicals of rows without a pivot. Each step on a C++ object (the LP,
//! scaling, the HEkk instance) is one `op` of HighsSolutionRust.cpp.

use super::ffi::CLp;
use super::{Log, LogType, Status, INF};
use crate::factor::{AMatrix, HFactor};
use crate::log_dev;
use std::ffi::c_void;

const BASIC: u8 = 1;
const NONBASIC: u8 = 4;
const UPDATE_METHOD_FT: i32 = 1;
const DEFAULT_PIVOT_THRESHOLD: f64 = 0.1;
const DEFAULT_PIVOT_TOLERANCE: f64 = 1e-10;

/// accommodateAlienBasis on an LP's matrix (column-wise) and basis
/// statuses. `factor_log` is the HFactor's log (no callbacks).
pub fn accommodate_alien_basis(factor_log: &Log, lp: &CLp, col_status: &mut [u8], row_status: &mut [u8]) {
    let (num_col, num_row) = (lp.num_col as usize, lp.num_row as usize);
    let mut basic_index: Vec<i32> = Vec::new();
    for (col, &s) in col_status[..num_col].iter().enumerate() {
        if s == BASIC {
            basic_index.push(col as i32);
        }
    }
    for (row, &s) in row_status[..num_row].iter().enumerate() {
        if s == BASIC {
            basic_index.push((num_col + row) as i32);
        }
    }
    let num_basic = basic_index.len();
    // SAFETY: the LP's matrix, read only
    let (start, index, value) = unsafe { (lp.a.start.get(), lp.a.index.get(), lp.a.value.get()) };
    let a = AMatrix { num_col: lp.a.num_col, start, index, value };
    let mut factor = HFactor::default();
    factor.setup(lp.a.num_col, lp.a.num_row, num_basic as i32, start, UPDATE_METHOD_FT);
    let rank_deficiency = factor.build_with_refactor_info(
        DEFAULT_PIVOT_THRESHOLD,
        DEFAULT_PIVOT_TOLERANCE,
        INF,
        &a,
        &mut basic_index,
    );
    if rank_deficiency != 0 && num_basic == lp.a.num_row as usize {
        log_dev!(factor_log, LogType::Warning, "Rank deficiency of %d identified in basis matrix\n", rank_deficiency);
    }
    for s in col_status[..num_col].iter_mut().chain(row_status[..num_row].iter_mut()) {
        if *s == BASIC {
            *s = NONBASIC;
        }
    }
    let use_basic = num_row.min(num_basic);
    for &var in &basic_index[..use_basic] {
        let var = var as usize;
        if var < num_col {
            col_status[var] = BASIC;
        } else {
            row_status[var - num_col] = BASIC;
        }
    }
    let rd = rank_deficiency as usize;
    for k in 0..num_row - use_basic {
        row_status[factor.row_with_no_pivot[rd + k] as usize] = BASIC;
    }
}

// The ops of formSimplexLpBasisAndFactor (HighsSolutionRust.cpp)
const OP_ENSURE_COLWISE: i32 = 0;
/// considerScaling(options, lp)
const OP_CONSIDER_SCALING: i32 = 1;
/// The LP view into `out` (a CLp)
const OP_LP_VIEW: i32 = 2;
const OP_UNAPPLY_SCALE: i32 = 3;
/// ekk_instance.moveLp(solver_object)
const OP_MOVE_LP: i32 = 4;
/// ekk_instance.status_.has_basis
const OP_EKK_HAS_BASIS: i32 = 5;
/// ekk_instance.setBasis(basis): the HighsStatus
const OP_EKK_SET_BASIS: i32 = 6;
/// ekk_instance.initialiseSimplexLpBasisAndFactor(arg): the HighsStatus
const OP_EKK_INITIALISE: i32 = 7;
/// lp.moveBackLpAndUnapplyScaling(ekk_lp) if lp.is_moved_
const OP_MOVE_BACK: i32 = 8;

/// What formSimplexLpBasisAndFactor works on
#[repr(C)]
pub struct CFormHost {
    pub ctx: *mut c_void,
    pub op: unsafe extern "C" fn(*mut c_void, i32, i32, *mut c_void) -> i32,
    pub log: Log,
    pub factor_log: Log,
    pub is_scaled: *const bool,
    pub basis_valid: bool,
    pub basis_useful: bool,
    pub basis_alien: *mut bool,
    pub col_status: super::ffi::RsMut<u8>,
    pub row_status: super::ffi::RsMut<u8>,
}

impl CFormHost {
    fn op(&self, code: i32, arg: i32, out: *mut c_void) -> i32 {
        // SAFETY: the host's op with its context
        unsafe { (self.op)(self.ctx, code, arg, out) }
    }
}

/// formSimplexLpBasisAndFactor: returns the HighsStatus
///
/// # Safety
/// The host's pointers and views valid, the op with its context
pub unsafe fn form_simplex_lp_basis_and_factor(h: &CFormHost, only_from_known_basis: bool) -> Status {
    let none = std::ptr::null_mut();
    h.op(OP_ENSURE_COLWISE, 0, none);
    let passed_scaled = *h.is_scaled;
    if !passed_scaled {
        h.op(OP_CONSIDER_SCALING, 0, none);
    }
    let check_basis = *h.basis_alien || (!h.basis_valid && h.basis_useful);
    if check_basis {
        *h.basis_alien = true;
        let mut lp = std::mem::zeroed::<CLp>();
        h.op(OP_LP_VIEW, 0, &mut lp as *mut CLp as *mut c_void);
        accommodate_alien_basis(&h.factor_log, &lp, h.col_status.get_mut(), h.row_status.get_mut());
        *h.basis_alien = false;
        if !passed_scaled {
            h.op(OP_UNAPPLY_SCALE, 0, none);
        }
        return Status::Ok;
    }
    h.op(OP_MOVE_LP, 0, none);
    let mut return_status = Status::Ok;
    if h.op(OP_EKK_HAS_BASIS, 0, none) == 0 {
        let call_status = status(h.op(OP_EKK_SET_BASIS, 0, none));
        return_status = h.log.interpret(call_status, return_status, "setBasis");
        if return_status == Status::Error {
            h.op(OP_MOVE_BACK, 0, none);
            return return_status;
        }
    }
    let call_status = status(h.op(OP_EKK_INITIALISE, only_from_known_basis as i32, none));
    h.op(OP_MOVE_BACK, 0, none);
    if call_status != Status::Ok {
        Status::Error
    } else {
        Status::Ok
    }
}

fn status(s: i32) -> Status {
    match s {
        0 => Status::Ok,
        1 => Status::Warning,
        _ => Status::Error,
    }
}

/// # Safety
/// As form_simplex_lp_basis_and_factor
#[no_mangle]
pub unsafe extern "C" fn highs_rs_form_simplex_lp_basis_and_factor(h: *const CFormHost, only_from_known_basis: bool) -> i32 {
    form_simplex_lp_basis_and_factor(&*h, only_from_known_basis) as i32
}

/// # Safety
/// The LP view and statuses valid (at least num_col / num_row long)
#[no_mangle]
pub unsafe extern "C" fn highs_rs_accommodate_alien_basis(
    factor_log: *const Log,
    lp: *const CLp,
    col_status: super::ffi::RsMut<u8>,
    row_status: super::ffi::RsMut<u8>,
) {
    accommodate_alien_basis(&*factor_log, &*lp, col_status.get_mut(), row_status.get_mut());
}
