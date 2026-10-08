//! LP reporting and the assessment of a primal solution (HighsLpUtils.cpp:
//! reportLp and its parts, reportMatrix, assessColPrimalSolution,
//! assessLpPrimalSolution, getNumInt, isLessInfeasibleDSECandidate, the
//! user data NULL checks, applyScalingToLpCol/Row, unscaleSolution,
//! highsVarTypeToString). Names come as `RsName` lists.

use super::edit::calculate_row_values_quad;
use super::ffi::{CLp, RsMut, RsName};
use super::lp_utils::cmax;
use super::var_type::{CONTINUOUS, IMPLICIT_INTEGER, INTEGER, SEMI_CONTINUOUS, SEMI_INTEGER};
use super::{matrix_format, Log, LogType, Status, INF};
use crate::{log_dev, log_user};

/// highs_isInfinity
fn is_inf(v: f64) -> bool {
    v >= INF
}

/// getNumInt
pub fn get_num_int(integrality: &[u8], num_col: usize) -> i32 {
    if integrality.is_empty() {
        return 0;
    }
    integrality[..num_col].iter().filter(|&&t| t == INTEGER).count() as i32
}

fn plural(n: i32) -> &'static str {
    if n == 1 {
        ""
    } else {
        "s"
    }
}

/// reportLpDimensions
fn report_lp_dimensions(log: &Log, lp: &CLp) {
    // SAFETY: the views are the C++ LP's
    let start = unsafe { lp.a.start.get() };
    let num_nz = if lp.num_col == 0 { 0 } else { start[lp.num_col as usize] };
    log_user!(
        log,
        LogType::Info,
        "LP has %d row%s, %d column%s",
        lp.num_row,
        plural(lp.num_row),
        lp.num_col,
        plural(lp.num_col)
    );
    // SAFETY: as above
    let num_int = get_num_int(unsafe { lp.integrality.get() }, lp.num_col as usize);
    if num_int != 0 {
        log_user!(
            log,
            LogType::Info,
            ", %d nonzero%s and %d integer column%s\n",
            num_nz,
            plural(num_nz),
            num_int,
            plural(num_int)
        );
    } else {
        log_user!(log, LogType::Info, " and %d nonzero%s\n", num_nz, plural(num_nz));
    }
}

/// reportLpObjSense
fn report_lp_obj_sense(log: &Log, lp: &CLp) {
    match lp.sense {
        1 => log_user!(log, LogType::Info, "Objective sense is minimize\n"),
        -1 => log_user!(log, LogType::Info, "Objective sense is maximize\n"),
        s => log_user!(log, LogType::Info, "Objective sense is ill-defined as %d\n", s),
    }
}

/// getBoundType
fn bound_type(lower: f64, upper: f64) -> &'static str {
    if is_inf(-lower) {
        if is_inf(upper) {
            "FR"
        } else {
            "UB"
        }
    } else if is_inf(upper) {
        "LB"
    } else if lower < upper {
        "BX"
    } else {
        "FX"
    }
}

/// reportLpColVectors
fn report_lp_col_vectors(log: &Log, lp: &CLp, col_names: &[RsName]) {
    if lp.num_col <= 0 {
        return;
    }
    // SAFETY: the views are the C++ LP's
    let (lower, upper, cost, start, integrality) =
        unsafe { (lp.col_lower.get(), lp.col_upper.get(), lp.col_cost.get(), lp.a.start.get(), lp.integrality.get()) };
    let have_integer_columns = get_num_int(integrality, lp.num_col as usize) != 0;
    let have_col_names = col_names.len() == lp.num_col as usize;
    log_user!(log, LogType::Info, "  Column        Lower        Upper         Cost       Type        Count");
    if have_integer_columns {
        log_user!(log, LogType::Info, "  Discrete");
    }
    if have_col_names {
        log_user!(log, LogType::Info, "  Name");
    }
    log_user!(log, LogType::Info, "\n");
    for j in 0..lp.num_col as usize {
        let count = start[j + 1] - start[j];
        log_user!(
            log,
            LogType::Info,
            "%8d %12g %12g %12g         %2s %12d",
            j,
            lower[j],
            upper[j],
            cost[j],
            bound_type(lower[j], upper[j]),
            count
        );
        if have_integer_columns {
            let integer_column = if integrality[j] == INTEGER {
                if lower[j] == 0.0 && upper[j] == 1.0 {
                    "Binary"
                } else {
                    "Integer"
                }
            } else {
                ""
            };
            log_user!(log, LogType::Info, "  %-8s", integer_column);
        }
        if have_col_names {
            log_user!(log, LogType::Info, "  %-s", col_names[j].text().as_ref());
        }
        log_user!(log, LogType::Info, "\n");
    }
}

/// reportLpRowVectors
fn report_lp_row_vectors(log: &Log, lp: &CLp, row_names: &[RsName]) {
    if lp.num_row <= 0 {
        return;
    }
    // SAFETY: the views are the C++ LP's
    let (lower, upper, start, index) = unsafe { (lp.row_lower.get(), lp.row_upper.get(), lp.a.start.get(), lp.a.index.get()) };
    let have_row_names = !row_names.is_empty();
    let mut count = vec![0i32; lp.num_row as usize];
    if lp.num_col > 0 {
        for &i in &index[..start[lp.num_col as usize] as usize] {
            count[i as usize] += 1;
        }
    }
    log_user!(log, LogType::Info, "     Row        Lower        Upper       Type        Count");
    if have_row_names {
        log_user!(log, LogType::Info, "  Name");
    }
    log_user!(log, LogType::Info, "\n");
    for i in 0..lp.num_row as usize {
        log_user!(
            log,
            LogType::Info,
            "%8d %12g %12g         %2s %12d",
            i,
            lower[i],
            upper[i],
            bound_type(lower[i], upper[i]),
            count[i]
        );
        if have_row_names {
            let name = row_names.get(i).map(|n| n.text()).unwrap_or_default();
            log_user!(log, LogType::Info, "  %-s", name.as_ref());
        }
        log_user!(log, LogType::Info, "\n");
    }
}

/// reportMatrix
pub fn report_matrix(log: &Log, message: &str, num_col: i32, num_nz: i32, start: &[i32], index: &[i32], value: &[f64]) {
    if num_col <= 0 {
        return;
    }
    log_user!(log, LogType::Info, "%-7s Index              Value\n", message);
    for col in 0..num_col {
        log_user!(log, LogType::Info, "    %8d Start   %10d\n", col, start[col as usize]);
        let to_el = if col < num_col - 1 { start[col as usize + 1] } else { num_nz };
        for el in start[col as usize]..to_el {
            log_user!(log, LogType::Info, "          %8d %12g\n", index[el as usize], value[el as usize]);
        }
    }
    log_user!(log, LogType::Info, "             Start   %10d\n", num_nz);
}

/// reportLp at a HighsLogType level: brief, then the vectors (detailed)
/// and the matrix (verbose)
pub fn report_lp(log: &Log, lp: &CLp, col_names: &[RsName], row_names: &[RsName], level: i32) {
    report_lp_dimensions(log, lp);
    report_lp_obj_sense(log, lp);
    if level >= LogType::Detailed as i32 {
        report_lp_col_vectors(log, lp, col_names);
        report_lp_row_vectors(log, lp, row_names);
        if level >= LogType::Verbose as i32 && lp.num_col > 0 {
            // SAFETY: the views are the C++ LP's
            let (start, index, value) = unsafe { (lp.a.start.get(), lp.a.index.get(), lp.a.value.get()) };
            let num_nz = start[lp.num_col as usize];
            report_matrix(log, "Column", lp.num_col, num_nz, start, index, value);
        }
    }
}

/// fractionality
fn fractionality(v: f64) -> f64 {
    (v - v.round()).abs()
}

/// assessColPrimalSolution: (column infeasibility, integer infeasibility)
pub fn assess_col_primal_solution(pft: f64, mft: f64, primal: f64, lower: f64, upper: f64, t: u8) -> (f64, f64) {
    let mut col_infeasibility = 0.0;
    if primal < lower - pft {
        col_infeasibility = lower - primal;
    } else if primal > upper + pft {
        col_infeasibility = primal - upper;
    }
    let mut integer_infeasibility = 0.0;
    if t == INTEGER || t == SEMI_INTEGER {
        integer_infeasibility = fractionality(primal);
    }
    if col_infeasibility > 0.0 && (t == SEMI_CONTINUOUS || t == SEMI_INTEGER) {
        if primal.abs() <= mft {
            col_infeasibility = 0.0;
        }
        if col_infeasibility != 0.0 && primal < upper {
            integer_infeasibility = cmax(col_infeasibility, integer_infeasibility);
        }
    }
    (col_infeasibility, integer_infeasibility)
}

/// What assessLpPrimalSolution finds
#[repr(C)]
#[derive(Default)]
pub struct PrimalAssessment {
    pub valid: bool,
    pub integral: bool,
    pub feasible: bool,
}

/// assessLpPrimalSolution
#[allow(clippy::too_many_arguments)]
pub fn assess_lp_primal_solution(
    log: &Log,
    message: &str,
    pft: f64,
    mft: f64,
    lp: &CLp,
    col_names: &[RsName],
    row_names: &[RsName],
    value_valid: bool,
    col_value: &[f64],
    row_value_sol: &[f64],
    out: &mut PrimalAssessment,
) -> Status {
    *out = PrimalAssessment::default();
    // SAFETY: the views are the C++ LP's
    let (integrality, col_lower, col_upper, row_lower, row_upper) = unsafe {
        (lp.integrality.get(), lp.col_lower.get(), lp.col_upper.get(), lp.row_lower.get(), lp.row_upper.get())
    };
    let num_col = lp.num_col as usize;
    let num_row = lp.num_row as usize;
    let is_mip = integrality.iter().take(num_col).any(|&t| t != CONTINUOUS);
    let row_residual_tolerance = pft;
    let feasibility_tolerance = if is_mip { mft } else { pft };
    log_user!(
        log,
        LogType::Info,
        "%sAssessing feasibility of %s tolerance of %11.4g\n",
        message,
        if is_mip { "MIP using primal feasibility and integrality" } else { "LP using primal feasibility" },
        feasibility_tolerance
    );
    if !value_valid {
        return Status::Error;
    }
    let have_integrality = !integrality.is_empty();
    let have_col_names = col_names.len() == num_col;
    let name_of = |names: &[RsName], have: bool, k: usize| -> String {
        if have {
            format!(" ({})", names[k].text())
        } else {
            String::new()
        }
    };
    let (mut num_col_inf, mut max_col_inf, mut sum_col_inf) = (0, 0.0, 0.0);
    let (mut num_int_inf, mut max_int_inf, mut sum_int_inf) = (0, 0.0, 0.0);
    for j in 0..num_col {
        let primal = col_value[j];
        let (lower, upper) = (col_lower[j], col_upper[j]);
        let t = if have_integrality { integrality[j] } else { CONTINUOUS };
        let (col_inf, int_inf) = assess_col_primal_solution(pft, mft, primal, lower, upper, t);
        if col_inf > 0.0 {
            if col_inf > feasibility_tolerance {
                if col_inf > 2.0 * max_col_inf {
                    log_user!(
                        log,
                        LogType::Warning,
                        "Col %6d%s has         infeasibility of %11.4g from [lower, value, upper] = [%15.8g; %15.8g; %15.8g]\n",
                        j,
                        &name_of(col_names, have_col_names, j),
                        col_inf,
                        lower,
                        primal,
                        upper
                    );
                }
                num_col_inf += 1;
            }
            max_col_inf = cmax(col_inf, max_col_inf);
            sum_col_inf += col_inf;
        }
        if int_inf > 0.0 {
            if int_inf > mft {
                if int_inf > 2.0 * max_int_inf {
                    log_user!(
                        log,
                        LogType::Warning,
                        "Col %6d%s has integer infeasibility of %11.4g\n",
                        j,
                        &name_of(col_names, have_col_names, j),
                        int_inf
                    );
                }
                num_int_inf += 1;
            }
            max_int_inf = cmax(int_inf, max_int_inf);
            sum_int_inf += int_inf;
        }
    }
    // calculateRowValuesQuad
    if col_value.len() != num_col || lp.a.format != matrix_format::COLWISE {
        return Status::Error;
    }
    let mut row_value = vec![0.0; num_row];
    // SAFETY: the views are the C++ LP's
    let (start, index, value) = unsafe { (lp.a.start.get(), lp.a.index.get(), lp.a.value.get()) };
    calculate_row_values_quad(start, index, value, col_value, &mut row_value);
    let have_row_names = row_names.len() >= num_row;
    let (mut num_row_inf, mut max_row_inf, mut sum_row_inf) = (0, 0.0, 0.0);
    let (mut num_row_res, mut max_row_res, mut sum_row_res) = (0, 0.0, 0.0);
    for i in 0..num_row {
        let primal = row_value_sol[i];
        let (lower, upper) = (row_lower[i], row_upper[i]);
        let mut row_inf = 0.0;
        if primal < lower - feasibility_tolerance {
            row_inf = lower - primal;
        } else if primal > upper + feasibility_tolerance {
            row_inf = primal - upper;
        }
        if row_inf > 0.0 {
            if row_inf > feasibility_tolerance {
                if row_inf > 2.0 * max_row_inf {
                    log_user!(
                        log,
                        LogType::Warning,
                        "Row %6d%s has         infeasibility of %11.4g from [lower, value, upper] = [%15.8g; %15.8g; %15.8g]\n",
                        i,
                        &name_of(row_names, have_row_names, i),
                        row_inf,
                        lower,
                        primal,
                        upper
                    );
                }
                num_row_inf += 1;
            }
            max_row_inf = cmax(row_inf, max_row_inf);
            sum_row_inf += row_inf;
        }
        let row_residual = (primal - row_value[i]).abs();
        if row_residual > row_residual_tolerance {
            if row_residual > 2.0 * max_row_res {
                log_user!(
                    log,
                    LogType::Warning,
                    "Row %6d%s has         residual      of %11.4g\n",
                    i,
                    &name_of(row_names, have_row_names, i),
                    row_residual
                );
            }
            num_row_res += 1;
        }
        max_row_res = cmax(row_residual, max_row_res);
        sum_row_res += row_residual;
    }
    log_user!(log, LogType::Info, "Solution has               num          max          sum\n");
    log_user!(log, LogType::Info, "Col     infeasibilities %6d  %11.4g  %11.4g\n", num_col_inf, max_col_inf, sum_col_inf);
    if is_mip {
        log_user!(log, LogType::Info, "Integer infeasibilities %6d  %11.4g  %11.4g\n", num_int_inf, max_int_inf, sum_int_inf);
    }
    log_user!(log, LogType::Info, "Row     infeasibilities %6d  %11.4g  %11.4g\n", num_row_inf, max_row_inf, sum_row_inf);
    log_user!(log, LogType::Info, "Row     residuals       %6d  %11.4g  %11.4g\n", num_row_res, max_row_res, sum_row_res);
    out.valid = num_row_res == 0;
    out.integral = out.valid && num_int_inf == 0;
    out.feasible = out.valid && num_col_inf == 0 && num_int_inf == 0 && num_row_inf == 0;
    if !(out.integral && out.feasible) {
        return Status::Warning;
    }
    Status::Ok
}

/// isLessInfeasibleDSECandidate
pub fn is_less_infeasible_dse_candidate(log: &Log, model_name: &str, num_col: i32, start: &[i32], value: &[f64]) -> bool {
    let mut max_col_num_en = -1;
    const MAX_ALLOWED_COL_NUM_EN: i32 = 24;
    const MAX_ASSESS_COL_NUM_EN: i32 = 24;
    const MAX_AVERAGE_COL_NUM_EN: i32 = 6;
    for col in 0..num_col as usize {
        let col_num_en = start[col + 1] - start[col];
        max_col_num_en = max_col_num_en.max(col_num_en);
        if col_num_en > MAX_ASSESS_COL_NUM_EN {
            return false;
        }
        for &v in &value[start[col] as usize..start[col + 1] as usize] {
            if v.abs() != 1.0 {
                return false;
            }
        }
    }
    let average_col_num_en = start[num_col as usize] as f64 / num_col as f64;
    let candidate = average_col_num_en <= MAX_AVERAGE_COL_NUM_EN as f64;
    log_dev!(
        log,
        LogType::Info,
        "LP %s has all |entries|=1; max column count = %d (limit %d); average column count = %0.2g (limit %d): LP %s a candidate for LiDSE\n",
        model_name,
        max_col_num_en,
        MAX_ALLOWED_COL_NUM_EN,
        average_col_num_en,
        MAX_AVERAGE_COL_NUM_EN,
        if candidate { "is" } else { "is not" }
    );
    candidate
}

/// The user data NULL checks (isColDataNull, isRowDataNull,
/// isMatrixDataNull): `null` flags which of the named arrays are NULL
pub fn user_data_null(log: &Log, names: &[&str], null: &[bool]) -> bool {
    let mut any = false;
    for (name, &n) in names.iter().zip(null) {
        if n {
            log_user!(log, LogType::Error, "User-supplied %s are NULL\n", *name);
            any = true;
        }
    }
    any
}

/// highsVarTypeToString of an integer
pub fn var_type_string(t: i32) -> &'static str {
    match t {
        x if x == CONTINUOUS as i32 => "continuous",
        x if x == INTEGER as i32 => "integer",
        x if x == SEMI_CONTINUOUS as i32 => "semi continuous",
        x if x == SEMI_INTEGER as i32 => "semi integer",
        x if x == IMPLICIT_INTEGER as i32 => "implicit integer",
        _ => "unknown",
    }
}

/// unscaleSolution
pub fn unscale_solution(
    col: &[f64],
    row: &[f64],
    cost: f64,
    col_value: &mut [f64],
    col_dual: &mut [f64],
    row_value: &mut [f64],
    row_dual: &mut [f64],
) {
    for j in 0..col.len() {
        col_value[j] *= col[j];
        col_dual[j] /= col[j] / cost;
    }
    for i in 0..row.len() {
        row_value[i] /= row[i];
        row_dual[i] *= row[i] * cost;
    }
}

/// applyScalingToLpCol (is_col) or applyScalingToLpRow
pub fn apply_scaling_to_lp(lp: &mut CLp, is_col: bool, ix: i32, scale: f64) -> Status {
    let dim = if is_col { lp.num_col } else { lp.num_row };
    if ix < 0 || ix >= dim || scale == 0.0 {
        return Status::Error;
    }
    let ix = ix as usize;
    // SAFETY: the views are the C++ LP's, unaliased during the call
    unsafe {
        // HighsSparseMatrix::scaleCol / scaleRow
        let (start, index, value) = (lp.a.start.get(), lp.a.index.get(), lp.a.value.get_mut());
        let colwise = lp.a.format == matrix_format::COLWISE;
        if colwise == is_col {
            for v in &mut value[start[ix] as usize..start[ix + 1] as usize] {
                *v *= scale;
            }
        } else {
            let num_vec = if colwise { lp.a.num_col } else { lp.a.num_row } as usize;
            for k in 0..num_vec {
                for el in start[k] as usize..start[k + 1] as usize {
                    if index[el] as usize == ix {
                        value[el] *= scale;
                    }
                }
            }
        }
        if is_col {
            lp.col_cost.get_mut()[ix] *= scale;
            let (lower, upper) = (lp.col_lower.get_mut(), lp.col_upper.get_mut());
            if scale > 0.0 {
                lower[ix] /= scale;
                upper[ix] /= scale;
            } else {
                let new_upper = lower[ix] / scale;
                lower[ix] = upper[ix] / scale;
                upper[ix] = new_upper;
            }
        } else {
            let (lower, upper) = (lp.row_lower.get_mut(), lp.row_upper.get_mut());
            if scale > 0.0 {
                lower[ix] *= scale;
                upper[ix] *= scale;
            } else {
                let new_upper = lower[ix] * scale;
                lower[ix] = upper[ix] * scale;
                upper[ix] = new_upper;
            }
        }
    }
    Status::Ok
}

// ------------------------------------------------------------ entry points

/// reportLp
///
/// # Safety
/// The views are the C++ objects'
#[no_mangle]
pub unsafe extern "C" fn highs_rs_report_lp(
    log: *const Log,
    lp: *const CLp,
    col_names: RsMut<RsName>,
    row_names: RsMut<RsName>,
    level: i32,
) {
    report_lp(&*log, &*lp, col_names.get(), row_names.get(), level)
}

/// reportMatrix
///
/// # Safety
/// start holds num_col entries (num_col + 1 if num_nz), index and value
/// num_nz (or are NULL)
#[no_mangle]
pub unsafe extern "C" fn highs_rs_report_matrix(
    log: *const Log,
    message: *const u8,
    message_len: usize,
    num_col: i32,
    num_nz: i32,
    start: *const i32,
    index: *const i32,
    value: *const f64,
) {
    if num_col <= 0 {
        return;
    }
    let message = std::str::from_utf8_unchecked(std::slice::from_raw_parts(message, message_len));
    let nnz = num_nz.max(0) as usize;
    let index = if index.is_null() || nnz == 0 { &[][..] } else { std::slice::from_raw_parts(index, nnz) };
    let value = if value.is_null() || nnz == 0 { &[][..] } else { std::slice::from_raw_parts(value, nnz) };
    report_matrix(&*log, message, num_col, num_nz, std::slice::from_raw_parts(start, num_col as usize), index, value)
}

/// assessLpPrimalSolution
///
/// # Safety
/// The views are the C++ objects'; `message` holds `message_len` bytes
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn highs_rs_assess_lp_primal_solution(
    log: *const Log,
    message: *const u8,
    message_len: usize,
    pft: f64,
    mft: f64,
    lp: *const CLp,
    col_names: RsMut<RsName>,
    row_names: RsMut<RsName>,
    value_valid: bool,
    col_value: RsMut<f64>,
    row_value: RsMut<f64>,
    out: *mut PrimalAssessment,
) -> i32 {
    let message =
        if message_len == 0 { "" } else { std::str::from_utf8_unchecked(std::slice::from_raw_parts(message, message_len)) };
    assess_lp_primal_solution(
        &*log,
        message,
        pft,
        mft,
        &*lp,
        col_names.get(),
        row_names.get(),
        value_valid,
        col_value.get(),
        row_value.get(),
        &mut *out,
    ) as i32
}

/// assessColPrimalSolution
///
/// # Safety
/// The outputs are valid
#[no_mangle]
pub unsafe extern "C" fn highs_rs_assess_col_primal_solution(
    pft: f64,
    mft: f64,
    primal: f64,
    lower: f64,
    upper: f64,
    t: u8,
    col_infeasibility: *mut f64,
    integer_infeasibility: *mut f64,
) {
    let (c, i) = assess_col_primal_solution(pft, mft, primal, lower, upper, t);
    *col_infeasibility = c;
    *integer_infeasibility = i;
}

/// isLessInfeasibleDSECandidate
///
/// # Safety
/// The views are the C++ objects'
#[no_mangle]
pub unsafe extern "C" fn highs_rs_is_less_infeasible_dse_candidate(
    log: *const Log,
    model_name: *const u8,
    model_name_len: usize,
    num_col: i32,
    start: RsMut<i32>,
    value: RsMut<f64>,
) -> bool {
    let name = if model_name_len == 0 { &[][..] } else { std::slice::from_raw_parts(model_name, model_name_len) };
    is_less_infeasible_dse_candidate(&*log, &String::from_utf8_lossy(name), num_col, start.get(), value.get())
}

/// isColDataNull (0), isRowDataNull (1), isMatrixDataNull (2): `null`
/// flags the arrays that are NULL, in the C++ argument order
///
/// # Safety
/// `null` holds 3 (0, 2) or 2 (1) flags
#[no_mangle]
pub unsafe extern "C" fn highs_rs_user_data_null(log: *const Log, which: i32, null: *const bool) -> bool {
    let names: &[&str] = match which {
        0 => &["column costs", "column lower bounds", "column upper bounds"],
        1 => &["row lower bounds", "row upper bounds"],
        _ => &["matrix starts", "matrix indices", "matrix values"],
    };
    user_data_null(&*log, names, std::slice::from_raw_parts(null, names.len()))
}

/// highsVarTypeToString
#[no_mangle]
pub extern "C" fn highs_rs_var_type_string(t: i32) -> RsName {
    let s = var_type_string(t);
    RsName { ptr: s.as_ptr(), len: s.len() }
}

/// unscaleSolution
///
/// # Safety
/// The views are the C++ objects', the solution of the scale's size
#[no_mangle]
pub unsafe extern "C" fn highs_rs_unscale_solution(
    col: RsMut<f64>,
    row: RsMut<f64>,
    cost: f64,
    col_value: RsMut<f64>,
    col_dual: RsMut<f64>,
    row_value: RsMut<f64>,
    row_dual: RsMut<f64>,
) {
    unscale_solution(
        col.get(),
        row.get(),
        cost,
        col_value.get_mut(),
        col_dual.get_mut(),
        row_value.get_mut(),
        row_dual.get_mut(),
    )
}

/// applyScalingToLpCol (is_col) / applyScalingToLpRow
///
/// # Safety
/// The view is the C++ LP's
#[no_mangle]
pub unsafe extern "C" fn highs_rs_apply_scaling_to_lp(lp: *mut CLp, is_col: bool, ix: i32, scale: f64) -> i32 {
    apply_scaling_to_lp(&mut *lp, is_col, ix, scale) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn col_primal_solution() {
        // Semi-continuous at zero: feasible
        assert_eq!(assess_col_primal_solution(1e-7, 1e-6, 0.0, 2.0, 5.0, SEMI_CONTINUOUS), (0.0, 0.0));
        // Semi-integer below its lower bound
        assert_eq!(assess_col_primal_solution(1e-7, 1e-6, 1.0, 2.0, 5.0, SEMI_INTEGER), (1.0, 1.0));
        assert_eq!(assess_col_primal_solution(1e-7, 1e-6, 2.5, 0.0, 5.0, INTEGER), (0.0, 0.5));
        assert_eq!(var_type_string(3), "semi integer");
    }
}
