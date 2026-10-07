//! User objective and bound scaling (HighsLpUtils.cpp: userScaleLp and its
//! parts, userScaleStatus and HighsUserScaleData's messages;
//! HighsInterface.cpp: the scaling of a solution in userScaleSolution).
//! The Hessian's scaling stays C++ (model/HighsHessianUtils.cpp).

use super::ffi::{CLp, RsMut};
use super::{var_type, Log, LogType, Status, INF};
use crate::log_user;

/// HighsUserScaleData (a standard-layout struct, read and written in place)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct UserScaleData {
    pub user_objective_scale: i32,
    pub user_bound_scale: i32,
    pub infinite_cost: f64,
    pub infinite_bound: f64,
    pub small_matrix_value: f64,
    pub large_matrix_value: f64,
    pub num_infinite_costs: i32,
    pub num_infinite_hessian_values: i32,
    pub num_infinite_col_bounds: i32,
    pub num_infinite_row_bounds: i32,
    pub num_small_matrix_values: i32,
    pub num_large_matrix_values: i32,
    pub suggested_user_objective_scale: i32,
    pub suggested_user_bound_scale: i32,
    pub applied: bool,
}

/// std::pow(2, n)
fn pow2(n: i32) -> f64 {
    2f64.powf(n as f64)
}

fn continuous(integrality: &[u8], j: usize) -> bool {
    integrality.is_empty() || integrality[j] == var_type::CONTINUOUS
}

/// userScaleCosts
pub fn user_scale_costs(integrality: &[u8], cost: &mut [f64], d: &mut UserScaleData, apply: bool) {
    d.num_infinite_costs = 0;
    if d.user_bound_scale == 0 && d.user_objective_scale == 0 {
        return;
    }
    let bound_scale_value = pow2(d.user_bound_scale);
    let objective_scale_value = pow2(d.user_objective_scale);
    for j in 0..cost.len() {
        let mut value = cost[j];
        if !continuous(integrality, j) {
            value *= bound_scale_value;
        }
        value *= objective_scale_value;
        if value.abs() > d.infinite_cost {
            d.num_infinite_costs += 1;
        }
        if apply {
            cost[j] = value;
        }
    }
}

/// The bound scaling of userScaleColBounds (`integrality` given) and
/// userScaleRowBounds: returns the number of infinite bounds
fn scale_bounds(integrality: Option<&[u8]>, lower: &mut [f64], upper: &mut [f64], d: &UserScaleData, apply: bool) -> i32 {
    let mut num_infinite = 0;
    let bound_scale_value = pow2(d.user_bound_scale);
    for i in 0..lower.len() {
        if let Some(integrality) = integrality {
            if !continuous(integrality, i) {
                continue;
            }
        }
        let (finite_lower, finite_upper) = (lower[i] > -INF, upper[i] < INF);
        for (b, finite) in [(&mut lower[i], finite_lower), (&mut upper[i], finite_upper)] {
            if finite {
                let value = *b * bound_scale_value;
                if value.abs() > d.infinite_bound {
                    num_infinite += 1;
                }
                if apply {
                    *b = value;
                }
            }
        }
    }
    num_infinite
}

/// userScaleColBounds
pub fn user_scale_col_bounds(integrality: &[u8], lower: &mut [f64], upper: &mut [f64], d: &mut UserScaleData, apply: bool) {
    d.num_infinite_col_bounds = 0;
    if d.user_bound_scale == 0 || lower.is_empty() {
        return;
    }
    d.num_infinite_col_bounds = scale_bounds(Some(integrality), lower, upper, d, apply);
}

/// userScaleRowBounds
pub fn user_scale_row_bounds(lower: &mut [f64], upper: &mut [f64], d: &mut UserScaleData, apply: bool) {
    d.num_infinite_row_bounds = 0;
    if d.user_bound_scale == 0 || lower.is_empty() {
        return;
    }
    d.num_infinite_row_bounds = scale_bounds(None, lower, upper, d, apply);
}

/// userScaleMatrix: the columns of non-continuous variables
pub fn user_scale_matrix(integrality: &[u8], lp: &CLp, d: &mut UserScaleData, apply: bool) {
    d.num_small_matrix_values = 0;
    d.num_large_matrix_values = 0;
    if d.user_bound_scale == 0 || integrality.is_empty() || lp.a.num_col <= 0 || lp.a.num_row <= 0 {
        return;
    }
    let bound_scale_value = pow2(d.user_bound_scale);
    // SAFETY: the LP's matrix, which nothing else touches during the call
    let (start, index, value) = unsafe { (lp.a.start.get(), lp.a.index.get(), lp.a.value.get_mut()) };
    let mut scale = |el: usize, d: &mut UserScaleData| {
        let v = value[el] * bound_scale_value;
        let abs_value = v.abs();
        if abs_value <= d.small_matrix_value {
            d.num_small_matrix_values += 1;
        } else if abs_value >= d.large_matrix_value {
            d.num_large_matrix_values += 1;
        }
        if apply {
            value[el] = v;
        }
    };
    if lp.a.format == super::matrix_format::COLWISE {
        for j in 0..lp.a.num_col as usize {
            if integrality[j] == var_type::CONTINUOUS {
                continue;
            }
            for el in start[j] as usize..start[j + 1] as usize {
                scale(el, d);
            }
        }
    } else {
        for i in 0..lp.a.num_row as usize {
            for el in start[i] as usize..start[i + 1] as usize {
                if integrality[index[el] as usize] == var_type::CONTINUOUS {
                    continue;
                }
                scale(el, d);
            }
        }
    }
}

/// userScaleLp(lp, data, apply)
pub fn user_scale_lp(lp: &CLp, d: &mut UserScaleData, apply: bool) {
    // SAFETY: the LP's arrays, which nothing else touches during the call
    let (integrality, cost, col_lower, col_upper, row_lower, row_upper) = unsafe {
        (
            lp.integrality.get(),
            lp.col_cost.get_mut(),
            lp.col_lower.get_mut(),
            lp.col_upper.get_mut(),
            lp.row_lower.get_mut(),
            lp.row_upper.get_mut(),
        )
    };
    user_scale_costs(integrality, cost, d, apply);
    user_scale_col_bounds(integrality, col_lower, col_upper, d, apply);
    user_scale_matrix(integrality, lp, d, apply);
    user_scale_row_bounds(row_lower, row_upper, d, apply);
}

fn plural(n: i32) -> &'static str {
    if n > 1 {
        "s"
    } else {
        ""
    }
}

/// HighsUserScaleData::scaleError
pub fn scale_error(d: &UserScaleData) -> Option<String> {
    let (c, h, cb, rb, lm) = (
        d.num_infinite_costs,
        d.num_infinite_hessian_values,
        d.num_infinite_col_bounds,
        d.num_infinite_row_bounds,
        d.num_large_matrix_values,
    );
    if c + h + cb + rb + lm == 0 {
        return None;
    }
    let mut s = String::from("User scaling of");
    if d.user_objective_scale != 0 {
        s += &format!(" 2**({}) for costs", d.user_objective_scale);
    }
    if d.user_bound_scale != 0 {
        if d.user_objective_scale != 0 {
            s += " and";
        }
        s += &format!(" 2**({}) for bounds", d.user_bound_scale);
    }
    s += " yields";
    if c != 0 {
        s += &format!(" {} infinite cost{}", c, plural(c));
    }
    if h != 0 {
        if c != 0 {
            s += if cb != 0 || rb != 0 { "," } else { " and" };
        }
        s += &format!(" {} infinite Hessian values{}", h, plural(h));
    }
    if cb != 0 {
        if c != 0 || h != 0 {
            s += if rb != 0 { "," } else { " and" };
        }
        s += &format!(" {} infinite column bound{}", cb, plural(cb));
    }
    if rb != 0 {
        if c != 0 || h != 0 || cb != 0 {
            s += " and";
        }
        s += &format!(" {} infinite row bound{}", rb, plural(rb));
    }
    if lm != 0 {
        if c + h + cb + rb > 0 {
            s += ", and";
        }
        s += &format!(" {} large matrix value{}", lm, plural(lm));
    }
    s += "\n";
    Some(s)
}

/// HighsUserScaleData::scaleWarning
pub fn scale_warning(d: &UserScaleData) -> Option<String> {
    let n = d.num_small_matrix_values;
    if n == 0 {
        return None;
    }
    Some(format!(
        "User scaling of 2**({}) for bounds yields {} small matrix value{}\n",
        d.user_bound_scale,
        n,
        plural(n)
    ))
}

/// userScaleStatus
pub fn user_scale_status(log: &Log, d: &UserScaleData) -> Status {
    let mut return_status = Status::Ok;
    if let Some(m) = scale_warning(d) {
        log_user!(log, LogType::Warning, "%s\n", &*m);
        return_status = Status::Warning;
    }
    if let Some(m) = scale_error(d) {
        log_user!(log, LogType::Error, "%s\n", &*m);
        return_status = Status::Error;
    }
    return_status
}

/// The scaling of the solution in Highs::userScaleSolution (primal and
/// dual parts if their status is not None); with `update_kkt` returns the
/// scaled objective value
#[allow(clippy::too_many_arguments)]
pub fn user_scale_solution(
    d: &UserScaleData,
    integrality: &[u8],
    primal: bool,
    dual: bool,
    col_value: &mut [f64],
    row_value: &mut [f64],
    col_dual: &mut [f64],
    row_dual: &mut [f64],
    objective: f64,
    offset: f64,
) -> f64 {
    let objective_scale_value = pow2(d.user_objective_scale);
    let bound_scale_value = pow2(d.user_bound_scale);
    if primal && d.user_bound_scale != 0 {
        for (j, x) in col_value.iter_mut().enumerate() {
            if continuous(integrality, j) {
                *x *= bound_scale_value;
            }
        }
        for x in row_value.iter_mut() {
            *x *= bound_scale_value;
        }
    }
    if dual && d.user_objective_scale != 0 {
        for y in col_dual.iter_mut().chain(row_dual.iter_mut()) {
            *y *= objective_scale_value;
        }
    }
    let mut objective_function_value = objective - offset;
    objective_function_value *= bound_scale_value * objective_scale_value;
    objective_function_value += offset;
    objective_function_value
}

// The C++ entry points (HighsLpUtilsRust.cpp)

/// # Safety
/// `lp` and `d` valid
#[no_mangle]
pub unsafe extern "C" fn highs_rs_user_scale_lp(lp: *const CLp, d: *mut UserScaleData, apply: bool) {
    user_scale_lp(&*lp, &mut *d, apply);
}

/// # Safety
/// `log` and `d` valid
#[no_mangle]
pub unsafe extern "C" fn highs_rs_user_scale_status(log: *const Log, d: *const UserScaleData) -> i32 {
    user_scale_status(&*log, &*d) as i32
}

/// # Safety
/// `d` valid; which 0 the error, 1 the warning message, written through
/// `set` to `ctx` if there is one
#[no_mangle]
pub unsafe extern "C" fn highs_rs_user_scale_message(
    d: *const UserScaleData,
    which: i32,
    ctx: *mut std::ffi::c_void,
    set: unsafe extern "C" fn(*mut std::ffi::c_void, *const u8, usize),
) -> bool {
    let m = if which == 0 { scale_error(&*d) } else { scale_warning(&*d) };
    match m {
        Some(m) => {
            set(ctx, m.as_ptr(), m.len());
            true
        }
        None => false,
    }
}

/// # Safety
/// The arrays valid
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn highs_rs_user_scale_solution(
    d: *const UserScaleData,
    integrality: RsMut<u8>,
    primal: bool,
    dual: bool,
    col_value: RsMut<f64>,
    row_value: RsMut<f64>,
    col_dual: RsMut<f64>,
    row_dual: RsMut<f64>,
    objective: f64,
    offset: f64,
) -> f64 {
    user_scale_solution(
        &*d,
        integrality.get(),
        primal,
        dual,
        col_value.get_mut(),
        row_value.get_mut(),
        col_dual.get_mut(),
        row_dual.get_mut(),
        objective,
        offset,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> UserScaleData {
        UserScaleData {
            user_objective_scale: 2,
            user_bound_scale: -1,
            infinite_cost: 1e20,
            infinite_bound: 1e20,
            small_matrix_value: 1e-9,
            large_matrix_value: 1e15,
            num_infinite_costs: 0,
            num_infinite_hessian_values: 0,
            num_infinite_col_bounds: 0,
            num_infinite_row_bounds: 0,
            num_small_matrix_values: 0,
            num_large_matrix_values: 0,
            suggested_user_objective_scale: 0,
            suggested_user_bound_scale: 0,
            applied: false,
        }
    }

    #[test]
    fn messages_and_scaling() {
        let mut d = data();
        let mut cost = [1.0, 3e19];
        user_scale_costs(&[0, 1], &mut cost, &mut d, true);
        assert_eq!(cost, [4.0, 6e19]);
        assert_eq!(d.num_infinite_costs, 0);
        let mut lower = [-INF, 2.0];
        let mut upper = [4.0, INF];
        user_scale_col_bounds(&[], &mut lower, &mut upper, &mut d, true);
        assert_eq!((lower, upper), ([-INF, 1.0], [2.0, INF]));
        d.num_infinite_costs = 1;
        d.num_infinite_row_bounds = 2;
        d.num_large_matrix_values = 1;
        assert_eq!(
            scale_error(&d).unwrap(),
            "User scaling of 2**(2) for costs and 2**(-1) for bounds yields 1 infinite cost and 2 infinite row bounds, and 1 large matrix value\n"
        );
        d.num_small_matrix_values = 3;
        assert_eq!(scale_warning(&d).unwrap(), "User scaling of 2**(-1) for bounds yields 3 small matrix values\n");
    }
}
