//! The basis functions of HighsSolution.cpp (refineBasis,
//! isBasisConsistent) and the conversions of an IPX solution
//! (ipxSolutionToHighsSolution, ipxBasicSolutionToHighsBasicSolution).
//! C++ (HighsSolutionRust.cpp) sizes the solution and basis vectors.
//!
//! clang fuses the row activities `a += x * v` of free rows.

use super::ffi::{CLp, RsMut};
use super::{Log, LogType, Status, INF};
use crate::util::fma::ClangFma;
use crate::{log_dev, log_user};

// HighsBasisStatus
const LOWER: u8 = 0;
const BASIC: u8 = 1;
const UPPER: u8 = 2;
const ZERO: u8 = 3;
const NONBASIC: u8 = 4;

/// The refined status of a kNonbasic variable (refineBasis)
fn refined(lower: f64, upper: f64, value: Option<f64>) -> u8 {
    if lower == upper {
        LOWER
    } else if -lower < INF {
        if upper < INF {
            match value {
                Some(v) => {
                    if v < 0.5 * (lower + upper) {
                        LOWER
                    } else {
                        UPPER
                    }
                }
                None => {
                    if lower.abs() < upper.abs() {
                        LOWER
                    } else {
                        UPPER
                    }
                }
            }
        } else {
            LOWER
        }
    } else if upper < INF {
        UPPER
    } else {
        ZERO
    }
}

/// refineBasis: kNonbasic statuses set from the bounds and any primal
/// values (empty if there is no solution)
pub fn refine_basis(lower: &[f64], upper: &[f64], value: &[f64], status: &mut [u8]) {
    for i in 0..status.len() {
        if status[i] != NONBASIC {
            continue;
        }
        let v = if value.is_empty() { None } else { Some(value[i]) };
        status[i] = refined(lower[i], upper[i], v);
    }
}

/// isBasisConsistent, given that the basis has the right size
pub fn basis_consistent(col_status: &[u8], row_status: &[u8]) -> bool {
    let n = col_status.iter().chain(row_status).filter(|&&s| s == BASIC).count();
    n == row_status.len()
}

/// The solution and basis written by the IPX conversions (sized by C++)
pub struct Out<'a> {
    pub col_value: &'a mut [f64],
    pub col_dual: &'a mut [f64],
    pub row_value: &'a mut [f64],
    pub row_dual: &'a mut [f64],
    pub col_status: &'a mut [u8],
    pub row_status: &'a mut [u8],
}

/// The row activities of free rows, which IPX drops, if there are any
struct Activities(Vec<f64>);

impl Activities {
    fn new(lp: &CLp, ipx_num_row: i32) -> Activities {
        Activities(if ipx_num_row < lp.num_row { vec![0.0; lp.num_row as usize] } else { Vec::new() })
    }
    fn add_col(&mut self, start: &[i32], index: &[i32], value: &[f64], col: usize, x: f64) {
        if self.0.is_empty() {
            return;
        }
        for el in start[col] as usize..start[col + 1] as usize {
            let r = index[el] as usize;
            self.0[r] = x.mul_add_c(value[el], self.0[r]);
        }
    }
    fn get(&self, row: usize) -> f64 {
        // The C++ reads past an empty vector if a free row was kept
        self.0.get(row).copied().unwrap_or(0.0)
    }
}

fn is_free(lower: f64, upper: f64) -> bool {
    lower <= -INF && upper >= INF
}

fn is_boxed(lower: f64, upper: f64) -> bool {
    (lower > -INF && upper < INF) && lower < upper
}

/// ipxSolutionToHighsSolution
#[allow(clippy::too_many_arguments)]
pub fn ipx_solution_to_highs_solution(
    lp: &CLp,
    rhs: &[f64],
    ipx_num_row: i32,
    ipx_x: &[f64],
    ipx_slack_vars: &[f64],
    ipx_y: &[f64],
    ipx_zl: &[f64],
    ipx_zu: &[f64],
    out: &mut Out,
) {
    // SAFETY: the LP's arrays, read only
    let (start, index, value, row_lower, row_upper) =
        unsafe { (lp.a.start.get(), lp.a.index.get(), lp.a.value.get(), lp.row_lower.get(), lp.row_upper.get()) };
    let num_col = lp.num_col as usize;
    let mut act = Activities::new(lp, ipx_num_row);
    for col in 0..num_col {
        let x = ipx_x[col];
        act.add_col(start, index, value, col, x);
        out.col_value[col] = x;
        out.col_dual[col] = ipx_zl[col] - ipx_zu[col];
    }
    let mut ipx_row = 0usize;
    let mut ipx_slack = num_col;
    for row in 0..lp.num_row as usize {
        let (lower, upper) = (row_lower[row], row_upper[row]);
        if is_free(lower, upper) {
            out.row_value[row] = act.get(row);
            out.row_dual[row] = 0.0;
            continue;
        }
        let (v, d) = if is_boxed(lower, upper) {
            let vd = (ipx_x[ipx_slack], ipx_zl[ipx_slack] - ipx_zu[ipx_slack]);
            ipx_slack += 1;
            vd
        } else {
            (rhs[ipx_row] - ipx_slack_vars[ipx_row], ipx_y[ipx_row])
        };
        out.row_value[row] = v;
        out.row_dual[row] = d;
        ipx_row += 1;
    }
    if lp.sense == -1 {
        for d in out.col_dual.iter_mut().chain(out.row_dual.iter_mut()) {
            *d *= -1.0;
        }
    }
}

/// IpxSolution
pub struct IpxSolution<'a> {
    pub num_col: i32,
    pub num_row: i32,
    pub col_value: &'a [f64],
    pub row_value: &'a [f64],
    pub col_dual: &'a [f64],
    pub row_dual: &'a [f64],
    pub col_status: &'a [i32],
    pub row_status: &'a [i32],
}

const IPX_BASIC: i32 = 0;
const IPX_AT_LB: i32 = -1;
const IPX_AT_UB: i32 = -2;
const IPX_SUPERBASIC: i32 = -3;

/// ipxBasicSolutionToHighsBasicSolution
pub fn ipx_basic_solution_to_highs_basic_solution(
    log: &Log,
    lp: &CLp,
    rhs: &[f64],
    constraint_type: &[u8],
    ipx: &IpxSolution,
    out: &mut Out,
) -> Status {
    // SAFETY: the LP's arrays, read only
    let (start, index, value, col_lower, col_upper, row_lower, row_upper) = unsafe {
        (
            lp.a.start.get(),
            lp.a.index.get(),
            lp.a.value.get(),
            lp.col_lower.get(),
            lp.col_upper.get(),
            lp.row_lower.get(),
            lp.row_upper.get(),
        )
    };
    let num_col = lp.num_col as usize;
    let mut act = Activities::new(lp, ipx.num_row);
    for col in 0..num_col {
        let st = ipx.col_status[col];
        let status = match st {
            IPX_BASIC => BASIC,
            IPX_AT_LB => LOWER,
            IPX_AT_UB => UPPER,
            IPX_SUPERBASIC => ZERO,
            _ => {
                log_dev!(
                    log,
                    LogType::Error,
                    "\nError in IPX conversion: Unrecognised value ipx_col_status[%2d] = %d\n",
                    col,
                    st
                );
                log_dev!(log, LogType::Error, "Bounds [%11.4g, %11.4g]\n", col_lower[col], col_upper[col]);
                log_dev!(
                    log,
                    LogType::Error,
                    "Col %2d ipx_col_status[%2d] = %2d; x[%2d] = %11.4g; z[%2d] = %11.4g\n",
                    col,
                    col,
                    st,
                    col,
                    ipx.col_value[col],
                    col,
                    ipx.col_dual[col]
                );
                log_user!(log, LogType::Error, "Unrecognised ipx_col_status value from IPX\n");
                return Status::Error;
            }
        };
        out.col_status[col] = status;
        out.col_value[col] = ipx.col_value[col];
        out.col_dual[col] = if status == BASIC { 0.0 } else { ipx.col_dual[col] };
        act.add_col(start, index, value, col, out.col_value[col]);
    }
    let mut ipx_row = 0usize;
    let mut ipx_slack = num_col;
    let mut num_boxed_rows = 0;
    let mut num_boxed_rows_basic = 0;
    let mut num_boxed_row_slacks_basic = 0;
    for row in 0..lp.num_row as usize {
        let (lower, upper) = (row_lower[row], row_upper[row]);
        let this_ipx_row = ipx_row;
        let mut unrecognised = false;
        if is_free(lower, upper) {
            out.row_status[row] = BASIC;
            out.row_value[row] = act.get(row);
            out.row_dual[row] = 0.0;
        } else {
            if is_boxed(lower, upper) {
                num_boxed_rows += 1;
                let slack_value = ipx.col_value[ipx_slack];
                let slack_dual = ipx.col_dual[ipx_slack];
                let slack_status = ipx.col_status[ipx_slack];
                let set = |out: &mut Out, s: u8, d: f64| {
                    out.row_status[row] = s;
                    out.row_value[row] = slack_value;
                    out.row_dual[row] = d;
                };
                if ipx.row_status[ipx_row] == IPX_BASIC {
                    num_boxed_rows_basic += 1;
                    set(out, BASIC, 0.0);
                } else if slack_status == IPX_BASIC {
                    num_boxed_row_slacks_basic += 1;
                    set(out, BASIC, 0.0);
                } else if slack_status == IPX_AT_LB {
                    set(out, LOWER, slack_dual);
                } else if slack_status == IPX_AT_UB {
                    set(out, UPPER, slack_dual);
                } else {
                    unrecognised = true;
                    log_dev!(
                        log,
                        LogType::Error,
                        "Error in IPX conversion: Row %2d (IPX row %2d) has unrecognised value ipx_col_status[%2d] = %d\n",
                        row,
                        ipx_row,
                        ipx_slack,
                        slack_status
                    );
                }
                ipx_slack += 1;
            } else if ipx.row_status[ipx_row] == IPX_BASIC {
                out.row_status[row] = BASIC;
                out.row_value[row] = rhs[ipx_row] - ipx.row_value[ipx_row];
                out.row_dual[row] = 0.0;
            } else {
                let v = rhs[ipx_row] - ipx.row_value[ipx_row];
                let d = ipx.row_dual[ipx_row];
                let status = match constraint_type[ipx_row] {
                    b'>' => Some(LOWER),
                    b'<' => Some(UPPER),
                    b'=' => Some(if d >= 0.0 { LOWER } else { UPPER }),
                    _ => None,
                };
                match status {
                    Some(s) => {
                        out.row_status[row] = s;
                        out.row_value[row] = v;
                        out.row_dual[row] = d;
                    }
                    None => {
                        unrecognised = true;
                        log_dev!(
                            log,
                            LogType::Error,
                            "Error in IPX conversion: Row %2d: cannot handle constraint_type[%2d] = %d\n",
                            row,
                            ipx_row,
                            constraint_type[ipx_row] as i8 as i32
                        );
                    }
                }
            }
            ipx_row += 1;
        }
        if unrecognised {
            log_dev!(log, LogType::Error, "Bounds [%11.4g, %11.4g]\n", lower, upper);
            log_dev!(
                log,
                LogType::Error,
                "Row %2d ipx_row_status[%2d] = %2d; s[%2d] = %11.4g; y[%2d] = %11.4g\n",
                row,
                this_ipx_row,
                ipx.row_status[this_ipx_row],
                this_ipx_row,
                ipx.row_value[this_ipx_row],
                this_ipx_row,
                ipx.row_dual[this_ipx_row]
            );
            log_user!(log, LogType::Error, "Unrecognised ipx_row_status value from IPX\n");
            return Status::Error;
        }
    }
    if lp.sense == -1 {
        for d in out.col_dual.iter_mut().chain(out.row_dual.iter_mut()) {
            *d *= -1.0;
        }
    }
    if num_boxed_rows != 0 {
        log_dev!(
            log,
            LogType::Info,
            "Of %d boxed rows: %d are basic and %d have basic slacks\n",
            num_boxed_rows,
            num_boxed_rows_basic,
            num_boxed_row_slacks_basic
        );
    }
    Status::Ok
}

// The C++ entry points (highs/lp_data/HighsSolutionRust.cpp)

/// The solution and basis vectors of the conversions
#[repr(C)]
pub struct COut {
    pub col_value: RsMut<f64>,
    pub col_dual: RsMut<f64>,
    pub row_value: RsMut<f64>,
    pub row_dual: RsMut<f64>,
    pub col_status: RsMut<u8>,
    pub row_status: RsMut<u8>,
}

impl COut {
    pub(crate) unsafe fn view(&self) -> Out<'_> {
        Out {
            col_value: self.col_value.get_mut(),
            col_dual: self.col_dual.get_mut(),
            row_value: self.row_value.get_mut(),
            row_dual: self.row_dual.get_mut(),
            col_status: self.col_status.get_mut(),
            row_status: self.row_status.get_mut(),
        }
    }
}

/// # Safety
/// The arrays valid; `value` empty for no solution
#[no_mangle]
pub unsafe extern "C" fn highs_rs_refine_basis(lower: RsMut<f64>, upper: RsMut<f64>, value: RsMut<f64>, status: RsMut<u8>) {
    refine_basis(lower.get(), upper.get(), value.get(), status.get_mut());
}

/// # Safety
/// The arrays valid
#[no_mangle]
pub unsafe extern "C" fn highs_rs_basis_consistent(col_status: RsMut<u8>, row_status: RsMut<u8>) -> bool {
    basis_consistent(col_status.get(), row_status.get())
}

/// # Safety
/// The arrays valid and sized by C++
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn highs_rs_ipx_solution_to_highs_solution(
    lp: *const CLp,
    rhs: RsMut<f64>,
    ipx_num_row: i32,
    ipx_x: RsMut<f64>,
    ipx_slack_vars: RsMut<f64>,
    ipx_y: RsMut<f64>,
    ipx_zl: RsMut<f64>,
    ipx_zu: RsMut<f64>,
    out: *const COut,
) {
    ipx_solution_to_highs_solution(
        &*lp,
        rhs.get(),
        ipx_num_row,
        ipx_x.get(),
        ipx_slack_vars.get(),
        ipx_y.get(),
        ipx_zl.get(),
        ipx_zu.get(),
        &mut (*out).view(),
    );
}

/// IpxSolution's vectors
#[repr(C)]
pub struct CIpxSolution {
    pub num_col: i32,
    pub num_row: i32,
    pub col_value: RsMut<f64>,
    pub row_value: RsMut<f64>,
    pub col_dual: RsMut<f64>,
    pub row_dual: RsMut<f64>,
    pub col_status: RsMut<i32>,
    pub row_status: RsMut<i32>,
}

/// # Safety
/// The arrays valid and sized by C++
#[no_mangle]
pub unsafe extern "C" fn highs_rs_ipx_basic_solution_to_highs_basic_solution(
    log: *const Log,
    lp: *const CLp,
    rhs: RsMut<f64>,
    constraint_type: RsMut<u8>,
    ipx: *const CIpxSolution,
    out: *const COut,
) -> i32 {
    let i = &*ipx;
    let ipx = IpxSolution {
        num_col: i.num_col,
        num_row: i.num_row,
        col_value: i.col_value.get(),
        row_value: i.row_value.get(),
        col_dual: i.col_dual.get(),
        row_dual: i.row_dual.get(),
        col_status: i.col_status.get(),
        row_status: i.row_status.get(),
    };
    ipx_basic_solution_to_highs_basic_solution(&*log, &*lp, rhs.get(), constraint_type.get(), &ipx, &mut (*out).view())
        as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refine() {
        let lower = [0.0, -INF, -INF, 1.0, -2.0];
        let upper = [0.0, INF, 3.0, INF, 4.0];
        let mut status = [NONBASIC, NONBASIC, NONBASIC, BASIC, NONBASIC];
        refine_basis(&lower, &upper, &[], &mut status);
        assert_eq!(status, [LOWER, ZERO, UPPER, BASIC, LOWER]);
        let mut status = [NONBASIC; 5];
        refine_basis(&lower, &upper, &[0.0, 0.0, 0.0, 0.0, 1.5], &mut status);
        assert_eq!(status[4], UPPER);
        assert!(!basis_consistent(&[BASIC, BASIC], &[UPPER, BASIC]));
        assert!(basis_consistent(&[BASIC, LOWER], &[UPPER]));
    }
}
