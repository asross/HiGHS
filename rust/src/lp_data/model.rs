//! Model-level data logic of HighsInterface.cpp: handleInfCost and
//! restoreInfCost (fixing variables with infinite costs and undoing it),
//! basisForSolution's statuses, and reportModelStats.

use super::ffi::RsMut;
use super::var_type::{INTEGER, SEMI_CONTINUOUS, SEMI_INTEGER};
use super::{Log, LogType, Status, INF};
use crate::util::fma::ClangFma;
use crate::{log_dev, log_user};

// HighsBasisStatus
const LOWER: u8 = 0;
const BASIC: u8 = 1;
const UPPER: u8 = 2;

/// The records of handleInfCost to append to HighsLpMods (buffers of the
/// model's size)
#[repr(C)]
pub struct CInfCostMods {
    pub index: RsMut<i32>,
    pub cost: RsMut<f64>,
    pub lower: RsMut<f64>,
    pub upper: RsMut<f64>,
    pub num: i32,
}

/// Highs::handleInfCost for an LP with infinite costs: the variables are
/// fixed at the bound that the cost pushes them to, or an error is
/// logged (before any change)
///
/// # Safety
/// The buffers of `m` hold num_col entries
#[allow(clippy::too_many_arguments)]
pub unsafe fn handle_inf_cost(
    log: &Log,
    inf_cost: f64,
    minimize: bool,
    is_mip: bool,
    integrality: &[u8],
    cost: &mut [f64],
    col_lower: &mut [f64],
    col_upper: &mut [f64],
    m: &mut CInfCostMods,
) -> Status {
    let (mi, mc, ml, mu) = (m.index.get_mut(), m.cost.get_mut(), m.lower.get_mut(), m.upper.get_mut());
    let mut n = 0;
    for k in 0..2 {
        for j in 0..cost.len() {
            let c = cost[j];
            if c > -inf_cost && c < inf_cost {
                continue;
            }
            let mut lower = col_lower[j];
            let mut upper = col_upper[j];
            if is_mip && integrality[j] == INTEGER {
                lower = lower.ceil();
                upper = upper.floor();
            }
            let error = |what: &str, bound: &str, value: f64| {
                log_user!(
                    log,
                    LogType::Error,
                    "Cannot %s with a cost on variable %d of %g and %s bound of %g\n",
                    what,
                    j,
                    c,
                    bound,
                    value
                );
                Status::Error
            };
            // Fix at upper (to_upper) or lower bound
            let to_upper = (c <= -inf_cost) == minimize;
            if to_upper {
                if upper < INF {
                    if k == 1 {
                        col_lower[j] = upper;
                    }
                } else {
                    return error(if minimize { "minimize" } else { "maximize" }, "upper", upper);
                }
            } else if lower > -INF {
                if k == 1 {
                    col_upper[j] = lower;
                }
            } else {
                return error(if minimize { "minimize" } else { "maximize" }, "lower", lower);
            }
            if k == 1 {
                mi[n] = j as i32;
                mc[n] = c;
                ml[n] = lower;
                mu[n] = upper;
                n += 1;
                cost[j] = 0.0;
            }
        }
    }
    m.num = n as i32;
    Status::Ok
}

/// Highs::restoreInfCost before the model status check: the costs and
/// bounds are restored, fixed nonbasic statuses (if `col_status` is not
/// empty) set by the bound, and the objective corrected
#[allow(clippy::too_many_arguments)]
pub fn restore_inf_cost(
    index: &[i32],
    saved_cost: &[f64],
    saved_lower: &[f64],
    saved_upper: &[f64],
    col_value: &[f64],
    col_status: &mut [u8],
    cost: &mut [f64],
    col_lower: &mut [f64],
    col_upper: &mut [f64],
    objective: &mut f64,
) {
    for (x, &j) in index.iter().enumerate() {
        let j = j as usize;
        let (c, lower, upper) = (saved_cost[x], saved_lower[x], saved_upper[x]);
        let value = if col_value.is_empty() { 0.0 } else { col_value[j] };
        if !col_status.is_empty() {
            col_status[j] = if col_lower[j] == lower { LOWER } else { UPPER };
        }
        if value != 0.0 {
            *objective = value.mul_add_c(c, *objective);
        }
        cost[j] = c;
        col_lower[j] = lower;
        col_upper[j] = upper;
    }
}

/// The statuses of Highs::basisForSolution: at a bound (within the
/// tolerance) or basic; returns the number of basic variables
#[allow(clippy::too_many_arguments)]
pub fn basis_for_solution(
    log: &Log,
    tol: f64,
    col_lower: &[f64],
    col_upper: &[f64],
    col_value: &[f64],
    row_lower: &[f64],
    row_upper: &[f64],
    row_value: &[f64],
    col_status: &mut [u8],
    row_status: &mut [u8],
) -> i32 {
    let status = |lower: f64, upper: f64, value: f64, n: &mut i32| {
        if (lower - value).abs() <= tol {
            LOWER
        } else if (upper - value).abs() <= tol {
            UPPER
        } else {
            *n += 1;
            BASIC
        }
    };
    let mut num_basic = 0;
    for j in 0..col_status.len() {
        col_status[j] = status(col_lower[j], col_upper[j], col_value[j], &mut num_basic);
    }
    let num_basic_col = num_basic;
    for i in 0..row_status.len() {
        row_status[i] = status(row_lower[i], row_upper[i], row_value[i], &mut num_basic);
    }
    let num_basic_row = num_basic - num_basic_col;
    log_dev!(
        log,
        LogType::Info,
        "LP has %d rows and solution yields %d possible basic variables (%d / %d; %d / %d)\n",
        row_status.len(),
        num_basic,
        num_basic_col,
        col_status.len(),
        num_basic_row,
        row_status.len()
    );
    num_basic
}

/// Highs::reportModelStats (when output_flag is set)
#[allow(clippy::too_many_arguments)]
pub fn report_model_stats(
    log: &Log,
    dev: bool,
    name: &str,
    num_col: i32,
    num_row: i32,
    a_num_nz: i32,
    hessian_dim: i32,
    q_num_nz: i32,
    integrality: &[u8],
    col_lower: &[f64],
    col_upper: &[f64],
) {
    let (mut num_integer, mut num_binary, mut num_semi_continuous, mut num_semi_integer) = (0, 0, 0, 0);
    for (j, &t) in integrality.iter().enumerate() {
        match t {
            INTEGER => {
                num_integer += 1;
                if col_lower[j] == 0.0 && col_upper[j] == 1.0 {
                    num_binary += 1;
                }
            }
            SEMI_CONTINUOUS => num_semi_continuous += 1,
            SEMI_INTEGER => num_semi_integer += 1,
            _ => {}
        }
    }
    let non_continuous = num_integer + num_semi_continuous + num_semi_integer != 0;
    let problem_type = match (hessian_dim != 0, non_continuous) {
        (true, true) => "MIQP",
        (true, false) => "QP",
        (false, true) => "MIP",
        (false, false) => "LP",
    };
    let s = |n: i32| if n == 1 { "" } else { "s" };
    if dev {
        log_dev!(log, LogType::Info, "%4s      : %s\n", problem_type, name);
        log_dev!(log, LogType::Info, "Row%s      : %d\n", s(num_row), num_row);
        log_dev!(log, LogType::Info, "Col%s      : %d\n", s(num_col), num_col);
        if q_num_nz != 0 {
            log_dev!(log, LogType::Info, "Matrix Nz : %d\n", a_num_nz);
            log_dev!(log, LogType::Info, "Hessian Nz: %d\n", q_num_nz);
        } else {
            log_dev!(log, LogType::Info, "Nonzero%s  : %d\n", s(a_num_nz), a_num_nz);
        }
        if num_integer != 0 {
            log_dev!(log, LogType::Info, "Integer   : %d (%d binary)\n", num_integer, num_binary);
        }
        if num_semi_continuous != 0 {
            log_dev!(log, LogType::Info, "SemiConts : %d\n", num_semi_continuous);
        }
        if num_semi_integer != 0 {
            log_dev!(log, LogType::Info, "SemiInt   : %d\n", num_semi_integer);
        }
    } else {
        let mut line = String::from(problem_type);
        if !name.is_empty() {
            line += " ";
            line += name;
        }
        line += &format!(" has {} row{}; {} col{}", num_row, s(num_row), num_col, s(num_col));
        if q_num_nz != 0 {
            line += &format!("; {} matrix nonzero{}", a_num_nz, s(a_num_nz));
            line += &format!("; {} Hessian nonzero{}", q_num_nz, s(q_num_nz));
        } else {
            line += &format!("; {} nonzero{}", a_num_nz, s(a_num_nz));
        }
        if num_integer != 0 {
            // As in the C++, the plural follows the number of nonzeros
            line += &format!("; {} integer variable{} ({} binary)", num_integer, s(a_num_nz), num_binary);
        }
        if num_semi_continuous != 0 {
            line += &format!("; {} semi-continuous variables", num_semi_continuous);
        }
        if num_semi_integer != 0 {
            line += &format!("; {} semi-integer variables", num_semi_integer);
        }
        log_user!(log, LogType::Info, "%s\n", &*line);
    }
}

// The C++ entry points (HighsRunRust.cpp)

/// # Safety
/// The arrays and the name valid
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn highs_rs_report_model_stats(
    log: *const Log,
    dev: bool,
    name: *const u8,
    name_len: usize,
    num_col: i32,
    num_row: i32,
    a_num_nz: i32,
    hessian_dim: i32,
    q_num_nz: i32,
    integrality: RsMut<u8>,
    col_lower: RsMut<f64>,
    col_upper: RsMut<f64>,
) {
    let name = String::from_utf8_lossy(super::options::RsStr { ptr: name, len: name_len }.get());
    report_model_stats(
        &*log,
        dev,
        &name,
        num_col,
        num_row,
        a_num_nz,
        hessian_dim,
        q_num_nz,
        integrality.get(),
        col_lower.get(),
        col_upper.get(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inf_cost() {
        let mut cost = [INF, 1.0, -INF];
        let mut lower = [0.5, 0.0, -INF];
        let mut upper = [2.5, 1.0, 3.0];
        let integrality = [INTEGER, 0, 0];
        let (mut a, mut b, mut c, mut d) = ([0i32; 3], [0f64; 3], [0f64; 3], [0f64; 3]);
        let mut m = CInfCostMods {
            index: RsMut { ptr: a.as_mut_ptr(), len: 3 },
            cost: RsMut { ptr: b.as_mut_ptr(), len: 3 },
            lower: RsMut { ptr: c.as_mut_ptr(), len: 3 },
            upper: RsMut { ptr: d.as_mut_ptr(), len: 3 },
            num: 0,
        };
        let s = unsafe {
            handle_inf_cost(&Log::none(), 1e20, true, true, &integrality, &mut cost, &mut lower, &mut upper, &mut m)
        };
        assert_eq!(s, Status::Ok);
        assert_eq!(m.num, 2);
        // Column 0 fixed at its rounded lower bound, column 2 at its upper
        assert_eq!((lower, upper), ([0.5, 0.0, 3.0], [1.0, 1.0, 3.0]));
        assert_eq!(cost, [0.0, 1.0, 0.0]);
        let mut obj = 1.0;
        let mut status = [LOWER, BASIC, UPPER];
        restore_inf_cost(&a[..2], &b[..2], &c[..2], &d[..2], &[], &mut status, &mut cost, &mut lower, &mut upper, &mut obj);
        assert_eq!(cost, [INF, 1.0, -INF]);
        // As in the C++, the saved bounds are the rounded ones
        assert_eq!(status, [UPPER, BASIC, UPPER]);
        assert_eq!((lower, upper), ([1.0, 0.0, -INF], [2.0, 1.0, 3.0]));
    }
}
