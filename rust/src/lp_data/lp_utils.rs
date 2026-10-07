//! HighsLpUtils.cpp and HighsMatrixUtils.cpp: LP validation (assessLp,
//! assessCosts, assessBounds, assessMatrix), simplex scaling (scaleLp
//! with equilibration or max-value scaling, applying and unapplying the
//! factors) and cleanBounds. Same arithmetic in the same order as the
//! C++ (none of it is contracted), and the C++'s std::min/std::max,
//! which differ from f64::min/max on NaN.

use super::ffi::{CLp, CLpOptions};
use super::{matrix_format, var_type, Log, LogType, Status, INF};
use crate::{log_dev, log_user};

/// std::min(a, b)
#[inline]
pub fn cmin(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// std::max(a, b)
#[inline]
pub fn cmax(a: f64, b: f64) -> f64 {
    if a < b {
        b
    } else {
        a
    }
}

/// HighsIndexCollection
pub struct IndexCollection<'a> {
    pub dimension: i32,
    pub is_interval: bool,
    pub from: i32,
    pub to: i32,
    pub is_set: bool,
    pub set_num_entries: i32,
    pub set: &'a [i32],
    pub is_mask: bool,
    pub mask: &'a [i32],
}

impl IndexCollection<'_> {
    pub fn interval(dimension: i32, from: i32, to: i32) -> IndexCollection<'static> {
        IndexCollection {
            dimension,
            is_interval: true,
            from,
            to,
            is_set: false,
            set_num_entries: -1,
            set: &[],
            is_mask: false,
            mask: &[],
        }
    }
    /// limits()
    pub fn limits(&self) -> (i32, i32) {
        if self.is_interval {
            (self.from, self.to)
        } else if self.is_set {
            (0, self.set_num_entries - 1)
        } else {
            (0, self.dimension - 1)
        }
    }
}

/// A HighsSparseMatrix
pub struct MatrixMut<'a> {
    pub format: i32,
    pub num_col: i32,
    pub num_row: i32,
    pub start: &'a mut [i32],
    pub p_end: &'a mut [i32],
    pub index: &'a mut [i32],
    pub value: &'a mut [f64],
}

impl MatrixMut<'_> {
    pub fn is_colwise(&self) -> bool {
        self.format == matrix_format::COLWISE
    }
    pub fn num_nz(&self) -> i32 {
        let n = if self.is_colwise() { self.num_col } else { self.num_row };
        self.start[n as usize]
    }
}

/// assessLp (the matrix's index and value are shrunk by C++ to its
/// number of nonzeros)
pub fn assess_lp(lp: &mut CLp, o: &CLpOptions) -> Status {
    let log = &o.log;
    let call = if lp_dimensions_ok(log, "assessLp", lp) { Status::Ok } else { Status::Error };
    let mut ret = log.interpret(call, Status::Ok, "assessLpDimensions");
    if ret == Status::Error {
        return ret;
    }
    // SAFETY: the view's arrays are valid for their lengths (see ffi.rs)
    let (col_cost, col_lower, col_upper, row_lower, row_upper, integrality) = unsafe {
        (
            lp.col_cost.get_mut(),
            lp.col_lower.get_mut(),
            lp.col_upper.get_mut(),
            lp.row_lower.get_mut(),
            lp.row_upper.get_mut(),
            lp.integrality.get(),
        )
    };
    if lp.num_col != 0 {
        let ic = IndexCollection::interval(lp.num_col, 0, lp.num_col - 1);
        let call = assess_costs(log, &ic, col_cost, &mut lp.has_infinite_cost, o.infinite_cost);
        ret = log.interpret(call, ret, "assessCosts");
        if ret == Status::Error {
            return ret;
        }
        let is_mip = integrality[..lp.num_col.min(integrality.len() as i32) as usize]
            .iter()
            .any(|&t| t != var_type::CONTINUOUS);
        let call = assess_bounds(
            log,
            "Col",
            0,
            &ic,
            col_lower,
            col_upper,
            o.infinite_bound,
            if is_mip { Some(integrality) } else { None },
        );
        ret = log.interpret(call, ret, "assessBounds");
        if ret == Status::Error {
            return ret;
        }
    }
    if lp.num_row != 0 {
        let ic = IndexCollection::interval(lp.num_row, 0, lp.num_row - 1);
        let call = assess_bounds(log, "Row", 0, &ic, row_lower, row_upper, o.infinite_bound, None);
        ret = log.interpret(call, ret, "assessBounds");
        if ret == Status::Error {
            return ret;
        }
    }
    if lp.num_col == 0 {
        return Status::Ok;
    }
    // SAFETY: as above
    let a = unsafe { lp.a.view() };
    let (vec_dim, num_vec) = if a.is_colwise() { (a.num_row, a.num_col) } else { (a.num_col, a.num_row) };
    let partitioned = a.format == matrix_format::ROWWISE_PARTITIONED;
    let call = assess_matrix(
        log,
        "LP",
        vec_dim,
        num_vec,
        partitioned,
        a.start,
        a.p_end,
        a.index,
        a.value,
        o.small_matrix_value,
        o.large_matrix_value,
        false,
    );
    ret = log.interpret(call, ret, "assessMatrix");
    if ret == Status::Error {
        return ret;
    }
    if ret != Status::Ok {
        log_dev!(log, LogType::Info, "assessLp returns HighsStatus = %s\n", ret.as_str());
    }
    ret
}

/// lpDimensionsOk
pub fn lp_dimensions_ok(log: &Log, message: &str, lp: &CLp) -> bool {
    let mut ok = true;
    let num_col = lp.num_col;
    let num_row = lp.num_row;
    if num_col < 0 {
        log_user!(log, LogType::Error, "LP dimension validation (%s) fails on num_col = %d >= 0\n", message, num_col);
    }
    ok = num_col >= 0 && ok;
    if num_row < 0 {
        log_user!(log, LogType::Error, "LP dimension validation (%s) fails on num_row = %d >= 0\n", message, num_row);
    }
    ok = num_row >= 0 && ok;
    if !ok {
        return ok;
    }
    let size_check = |ok: &mut bool, size: usize, num: i32, what: &str, dim: &str| {
        let size = size as i32;
        let legal = size >= num;
        if !legal {
            log_user!(
                log,
                LogType::Error,
                &format!("LP dimension validation (%s) fails on {what}.size() = %d < %d = {dim}\n"),
                message,
                size,
                num
            );
        }
        *ok = legal && *ok;
    };
    size_check(&mut ok, lp.col_cost.len, num_col, "col_cost", "num_col");
    size_check(&mut ok, lp.col_lower.len, num_col, "col_lower", "num_col");
    size_check(&mut ok, lp.col_upper.len, num_col, "col_upper", "num_col");

    let legal_format = lp.a.format == matrix_format::COLWISE || lp.a.format == matrix_format::ROWWISE;
    if !legal_format {
        log_user!(log, LogType::Error, "LP dimension validation (%s) fails on a_matrix_.format\n", message);
    }
    ok = legal_format && ok;
    let num_vec = if lp.a.format == matrix_format::COLWISE { num_col } else { num_row };
    // SAFETY: the view's arrays are valid for their lengths
    let legal_matrix_dimensions = assess_matrix_dimensions(
        log,
        num_vec,
        false,
        unsafe { lp.a.start.get() },
        &[],
        lp.a.index.len,
        lp.a.value.len,
    ) == Status::Ok;
    if !legal_matrix_dimensions {
        log_user!(log, LogType::Error, "LP dimension validation (%s) fails on a_matrix dimensions\n", message);
    }
    ok = legal_matrix_dimensions && ok;

    size_check(&mut ok, lp.row_lower.len, num_row, "row_lower", "num_row");
    size_check(&mut ok, lp.row_upper.len, num_row, "row_upper", "num_row");

    let legal_a_matrix_num_col = lp.a.num_col == num_col;
    let legal_a_matrix_num_row = lp.a.num_row == num_row;
    if !legal_a_matrix_num_col {
        log_user!(
            log,
            LogType::Error,
            "LP dimension validation (%s) fails on a_matrix.num_col_ = %d != %d = num_col\n",
            message,
            lp.a.num_col,
            num_col
        );
    }
    ok = legal_a_matrix_num_col && ok;
    if !legal_a_matrix_num_row {
        log_user!(
            log,
            LogType::Error,
            "LP dimension validation (%s) fails on a_matrix.num_row_ = %d != %d = num_row\n",
            message,
            lp.a.num_row,
            num_row
        );
    }
    ok = legal_a_matrix_num_row && ok;

    let legal_scale_strategy = lp.scale_strategy >= 0;
    if !legal_scale_strategy {
        log_user!(log, LogType::Error, "LP dimension validation (%s) fails on scale_.scale_strategy\n", message);
    }
    ok = legal_scale_strategy && ok;
    let scale_row_size = lp.scale_row.len as i32;
    let scale_col_size = lp.scale_col.len as i32;
    let has = lp.scale_has_scaling;
    let (legal_scale_num_col, legal_scale_num_row, legal_scale_row_size, legal_scale_col_size) = if has {
        (
            lp.scale_num_col == num_col,
            lp.scale_num_row == num_row,
            scale_row_size >= num_row,
            scale_col_size >= num_col,
        )
    } else {
        (lp.scale_num_col == 0, lp.scale_num_row == 0, scale_row_size == 0, scale_col_size == 0)
    };
    if !legal_scale_num_col {
        log_user!(
            log,
            LogType::Error,
            "LP dimension validation (%s) fails on scale_.num_col = %d != %d\n",
            message,
            lp.scale_num_col,
            if has { num_col } else { 0 }
        );
    }
    ok = legal_scale_num_col && ok;
    if !legal_scale_num_row {
        log_user!(
            log,
            LogType::Error,
            "LP dimension validation (%s) fails on scale_.num_row = %d != %d\n",
            message,
            lp.scale_num_row,
            if has { num_row } else { 0 }
        );
    }
    ok = legal_scale_num_row && ok;
    if !legal_scale_col_size {
        log_user!(
            log,
            LogType::Error,
            "LP dimension validation (%s) fails on scale_.col.size() = %d %s %d\n",
            message,
            scale_col_size,
            if has { ">=" } else { "==" },
            if has { num_col } else { 0 }
        );
    }
    ok = legal_scale_col_size && ok;
    if !legal_scale_row_size {
        log_user!(
            log,
            LogType::Error,
            "LP dimension validation (%s) fails on scale_.row.size() = %d %s %d\n",
            message,
            scale_row_size,
            if has { ">=" } else { "==" },
            if has { num_row } else { 0 }
        );
    }
    ok = legal_scale_row_size && ok;
    if !ok {
        log_user!(log, LogType::Error, "LP dimension validation (%s) fails\n", message);
    }
    ok
}

/// The k-th entry of an index collection: (local index, user index), or
/// None if masked out. `usr` carries the running interval counter.
#[inline]
fn ic_entry(ic: &IndexCollection, k: i32, usr: &mut i32) -> Option<i32> {
    let local = if ic.is_interval || ic.is_mask { k } else { ic.set[k as usize] };
    if ic.is_interval {
        *usr += 1;
    } else {
        *usr = k;
    }
    if ic.is_mask && ic.mask[local as usize] == 0 {
        return None;
    }
    Some(local)
}

/// assessCosts
pub fn assess_costs(
    log: &Log,
    ic: &IndexCollection,
    cost: &mut [f64],
    has_infinite_cost: &mut bool,
    infinite_cost: f64,
) -> Status {
    let (from_k, to_k) = ic.limits();
    if from_k > to_k {
        return Status::Ok;
    }
    let mut usr = -1;
    let mut num_infinite_cost = 0;
    for k in from_k..=to_k {
        if ic_entry(ic, k, &mut usr).is_none() {
            continue;
        }
        let c = &mut cost[usr as usize];
        if *c >= infinite_cost {
            num_infinite_cost += 1;
            *c = INF;
        } else if *c <= -infinite_cost {
            num_infinite_cost += 1;
            *c = -INF;
        }
    }
    if num_infinite_cost > 0 {
        *has_infinite_cost = true;
        log_user!(
            log,
            LogType::Info,
            "%d |cost| values greater than or equal to %12g are treated as Infinity\n",
            num_infinite_cost,
            infinite_cost
        );
    }
    Status::Ok
}

/// assessBounds
#[allow(clippy::too_many_arguments)]
pub fn assess_bounds(
    log: &Log,
    kind: &str,
    ml_ix_os: i32,
    ic: &IndexCollection,
    lower: &mut [f64],
    upper: &mut [f64],
    infinite_bound: f64,
    integrality: Option<&[u8]>,
) -> Status {
    let (from_k, to_k) = ic.limits();
    if from_k > to_k {
        return Status::Ok;
    }
    let mut error_found = false;
    let mut warning_found = false;
    let mut num_infinite_lower_bound = 0;
    let mut num_infinite_upper_bound = 0;
    let mut usr = -1;
    for k in from_k..=to_k {
        let Some(local) = ic_entry(ic, k, &mut usr) else { continue };
        let u = usr as usize;
        let ml_ix = ml_ix_os + local;
        // highs_isInfinity(-lower): -lower >= kHighsInf
        if -lower[u] < INF && lower[u] <= -infinite_bound {
            lower[u] = -INF;
            num_infinite_lower_bound += 1;
        }
        if upper[u] < INF && upper[u] >= infinite_bound {
            upper[u] = INF;
            num_infinite_upper_bound += 1;
        }
        let mut legal_lower_upper = lower[u] <= upper[u];
        if let Some(t) = integrality {
            if t[u] == var_type::SEMI_CONTINUOUS || t[u] == var_type::SEMI_INTEGER {
                legal_lower_upper = true;
            }
        }
        if !legal_lower_upper {
            log_user!(
                log,
                LogType::Warning,
                "%3s  %12d has inconsistent bounds [%12g, %12g]\n",
                kind,
                ml_ix,
                lower[u],
                upper[u]
            );
            warning_found = true;
        }
        if !(lower[u] < infinite_bound) {
            log_user!(
                log,
                LogType::Error,
                "%3s  %12d has lower bound of %12g >= %12g\n",
                kind,
                ml_ix,
                lower[u],
                infinite_bound
            );
            error_found = true;
        }
        if !(upper[u] > -infinite_bound) {
            log_user!(
                log,
                LogType::Error,
                "%3s  %12d has upper bound of %12g <= %12g\n",
                kind,
                ml_ix,
                upper[u],
                -infinite_bound
            );
            error_found = true;
        }
    }
    if num_infinite_lower_bound != 0 {
        log_user!(
            log,
            LogType::Info,
            "%3ss:%12d lower bounds    less than or equal to %12g are treated as -Infinity\n",
            kind,
            num_infinite_lower_bound,
            -infinite_bound
        );
    }
    if num_infinite_upper_bound != 0 {
        log_user!(
            log,
            LogType::Info,
            "%3ss:%12d upper bounds greater than or equal to %12g are treated as +Infinity\n",
            kind,
            num_infinite_upper_bound,
            infinite_bound
        );
    }
    if error_found {
        Status::Error
    } else if warning_found {
        Status::Warning
    } else {
        Status::Ok
    }
}

/// assessMatrixDimensions (index and value given by their sizes)
pub fn assess_matrix_dimensions(
    log: &Log,
    num_vec: i32,
    partitioned: bool,
    start: &[i32],
    p_end: &[i32],
    index_size: usize,
    value_size: usize,
) -> Status {
    let mut ok = true;
    let legal_num_vec = num_vec >= 0;
    if !legal_num_vec {
        log_user!(log, LogType::Error, "Matrix dimension validation fails on number of vectors = %d < 0\n", num_vec);
    }
    ok = legal_num_vec && ok;
    let legal_start_size = start.len() as i32 >= num_vec + 1;
    if !legal_start_size {
        log_user!(
            log,
            LogType::Error,
            "Matrix dimension validation fails on start size = %d < %d = num vectors + 1\n",
            start.len() as i32,
            num_vec + 1
        );
    }
    ok = legal_start_size && ok;
    if partitioned {
        if (p_end.len() as i32) < num_vec + 1 {
            log_user!(
                log,
                LogType::Error,
                "Matrix dimension validation fails on p_end size = %d < %d = num vectors + 1\n",
                p_end.len() as i32,
                num_vec + 1
            );
        }
        ok = p_end.len() as i32 >= num_vec + 1 && ok;
    }
    let num_nz = if legal_start_size { start[num_vec as usize] } else { 0 };
    if num_nz >= 0 {
        let legal_index_size = index_size as i32 >= num_nz;
        if !legal_index_size {
            log_user!(
                log,
                LogType::Error,
                "Matrix dimension validation fails on index size = %d < %d = number of nonzeros\n",
                index_size as i32,
                num_nz
            );
        }
        ok = legal_index_size && ok;
        let legal_value_size = value_size as i32 >= num_nz;
        if !legal_value_size {
            log_user!(
                log,
                LogType::Error,
                "Matrix dimension validation fails on value size = %d < %d = number of nonzeros\n",
                value_size as i32,
                num_nz
            );
        }
        ok = legal_value_size && ok;
    } else {
        log_user!(log, LogType::Error, "Matrix dimension validation fails on number of nonzeros = %d < 0\n", num_nz);
        ok = false;
    }
    if ok {
        Status::Ok
    } else {
        Status::Error
    }
}

/// assessMatrix: checks the starts and indices, sums or rejects duplicate
/// indices and removes small values, compacting the matrix in place
#[allow(clippy::too_many_arguments)]
pub fn assess_matrix(
    log: &Log,
    name: &str,
    vec_dim: i32,
    num_vec: i32,
    partitioned: bool,
    start: &mut [i32],
    p_end: &[i32],
    index: &mut [i32],
    value: &mut [f64],
    small_matrix_value: f64,
    large_matrix_value: f64,
    sum_duplicates: bool,
) -> Status {
    if assess_matrix_dimensions(log, num_vec, partitioned, start, p_end, index.len(), value.len()) == Status::Error {
        return Status::Error;
    }
    let mut error_found = false;
    let mut warning_found = false;
    let num_nz = start[num_vec as usize];
    if start[0] != 0 {
        log_user!(log, LogType::Error, "%s matrix start vector begins with %d rather than 0\n", name, start[0]);
        return Status::Error;
    }
    let mut previous_start = start[0];
    let mut this_start = start[0];
    let mut this_p_end = 0;
    if partitioned {
        this_p_end = p_end[0];
    }
    for ix in 0..num_vec {
        this_start = start[ix as usize];
        if this_start < previous_start {
            log_user!(
                log,
                LogType::Error,
                "%s matrix packed vector %d has illegal start of %d < %d = previous start\n",
                name,
                ix,
                this_start,
                previous_start
            );
            return Status::Error;
        }
        if partitioned {
            this_p_end = p_end[ix as usize];
            if this_p_end < this_start {
                log_user!(
                    log,
                    LogType::Error,
                    "%s matrix packed vector %d has illegal partition end of %d < %d =  start\n",
                    name,
                    ix,
                    this_p_end,
                    this_start
                );
                return Status::Error;
            }
        }
        previous_start = this_start;
    }
    if this_start > num_nz {
        log_user!(
            log,
            LogType::Error,
            "%s matrix packed vector %d has illegal start of %d > %d = number of nonzeros\n",
            name,
            num_vec,
            this_start,
            num_nz
        );
        return Status::Error;
    }
    if partitioned && this_p_end > num_nz {
        log_user!(
            log,
            LogType::Error,
            "%s matrix packed vector %d has illegal partition end of %d > %d = number of nonzeros\n",
            name,
            num_vec,
            this_p_end,
            num_nz
        );
        return Status::Error;
    }
    let mut num_new_nz: i32 = 0;
    let mut num_small_value = 0;
    let mut max_small_value = 0.0;
    let mut min_small_value = INF;
    let mut num_large_value = 0;
    let mut max_large_value = 0.0;
    let mut min_large_value = INF;
    let mut num_duplicate = 0;
    // Where each index last occurred: (vector + 1, new position); the
    // C++ uses a hash set (or map) cleared per vector
    let mut seen: Vec<(i32, i32)> = vec![(0, 0); vec_dim.max(0) as usize];
    for ix in 0..num_vec {
        let from_el = start[ix as usize];
        let to_el = start[ix as usize + 1];
        start[ix as usize] = num_new_nz;
        for el in from_el..to_el {
            let component = index[el as usize];
            if component < 0 {
                log_user!(
                    log,
                    LogType::Error,
                    "%s matrix packed vector %d, entry %d, is illegal index %d\n",
                    name,
                    ix,
                    el,
                    component
                );
                return Status::Error;
            }
            if component >= vec_dim {
                log_user!(
                    log,
                    LogType::Error,
                    "%s matrix packed vector %d, entry %d, is illegal index %12d >= %d = vector dimension\n",
                    name,
                    ix,
                    el,
                    component,
                    vec_dim
                );
                return Status::Error;
            }
            let s = seen[component as usize];
            if s.0 == ix + 1 {
                if sum_duplicates {
                    num_duplicate += 1;
                    value[s.1 as usize] += value[el as usize];
                    continue;
                }
                log_user!(
                    log,
                    LogType::Error,
                    "%s matrix packed vector %d, entry %d, is duplicate index %d\n",
                    name,
                    ix,
                    el,
                    component
                );
                return Status::Error;
            }
            index[num_new_nz as usize] = index[el as usize];
            value[num_new_nz as usize] = value[el as usize];
            seen[component as usize] = (ix + 1, num_new_nz);
            num_new_nz += 1;
        }
        let from_el = start[ix as usize];
        let to_el = num_new_nz;
        num_new_nz = start[ix as usize];
        for el in from_el..to_el {
            let abs_value = value[el as usize].abs();
            if abs_value >= large_matrix_value {
                if max_large_value < abs_value {
                    max_large_value = abs_value;
                }
                if min_large_value > abs_value {
                    min_large_value = abs_value;
                }
                num_large_value += 1;
            }
            let ok_value = abs_value > small_matrix_value;
            if !ok_value {
                if max_small_value < abs_value {
                    max_small_value = abs_value;
                }
                if min_small_value > abs_value {
                    min_small_value = abs_value;
                }
                num_small_value += 1;
            } else {
                index[num_new_nz as usize] = index[el as usize];
                value[num_new_nz as usize] = value[el as usize];
                num_new_nz += 1;
            }
        }
    }
    if num_duplicate != 0 {
        log_user!(
            log,
            LogType::Info,
            "%s matrix packed vector contains %d duplicate entr%s: summed\n",
            name,
            num_duplicate,
            if num_duplicate == 1 { "y" } else { "ies" }
        );
    }
    if num_large_value != 0 {
        log_user!(
            log,
            LogType::Error,
            "%s matrix packed vector contains %d |value| in [%g, %g] greater than %g\n",
            name,
            num_large_value,
            min_large_value,
            max_large_value,
            large_matrix_value
        );
        error_found = true;
    }
    if num_small_value != 0 {
        if partitioned {
            log_user!(
                log,
                LogType::Error,
                "%s matrix packed partitioned vector contains %d |value| in [%g, %g] less than or equal to %g: ignored\n",
                name,
                num_small_value,
                min_small_value,
                max_small_value,
                small_matrix_value
            );
            error_found = true;
        }
        if max_small_value > 0.0 {
            log_user!(
                log,
                LogType::Warning,
                "%s matrix packed vector contains %d |value| in [%g, %g] less than or equal to %g: ignored\n",
                name,
                num_small_value,
                min_small_value,
                max_small_value,
                small_matrix_value
            );
            warning_found = true;
        }
    }
    start[num_vec as usize] = num_new_nz;
    if error_found {
        Status::Error
    } else if warning_found {
        Status::Warning
    } else {
        Status::Ok
    }
}

// Simplex scale strategies
const SCALE_CHOOSE: i32 = 1;
const SCALE_EQUILIBRATION: i32 = 2;
const SCALE_FORCED_EQUILIBRATION: i32 = 3;

/// The arrays of an LP view
struct LpArrays<'a> {
    col_cost: &'a mut [f64],
    col_lower: &'a mut [f64],
    col_upper: &'a mut [f64],
    row_lower: &'a mut [f64],
    row_upper: &'a mut [f64],
    start: &'a [i32],
    index: &'a [i32],
    value: &'a mut [f64],
    scale_col: &'a mut [f64],
    scale_row: &'a mut [f64],
}

fn arrays<'a>(lp: &CLp) -> LpArrays<'a> {
    // SAFETY: the view's arrays are valid for their lengths and distinct
    unsafe {
        LpArrays {
            col_cost: lp.col_cost.get_mut(),
            col_lower: lp.col_lower.get_mut(),
            col_upper: lp.col_upper.get_mut(),
            row_lower: lp.row_lower.get_mut(),
            row_upper: lp.row_upper.get_mut(),
            start: lp.a.start.get(),
            index: lp.a.index.get(),
            value: lp.a.value.get_mut(),
            scale_col: lp.scale_col.get_mut(),
            scale_row: lp.scale_row.get_mut(),
        }
    }
}

/// scaleLp, after C++ has cleared the scaling and sized scale.col/row:
/// returns whether the LP is scaled; C++ clears the scaling if not.
/// Sets the scale strategy (and scalars if scaled).
pub fn scale_lp(lp: &mut CLp, o: &CLpOptions, force_scaling: bool) -> bool {
    let num_col = lp.num_col as usize;
    let num_row = lp.num_row as usize;
    let use_scale_strategy =
        if o.simplex_scale_strategy == SCALE_CHOOSE { SCALE_FORCED_EQUILIBRATION } else { o.simplex_scale_strategy };
    let a = arrays(lp);
    let no_min = 0.2;
    let no_max = 5.0;
    let mut original_matrix_min_value = INF;
    let mut original_matrix_max_value = 0.0;
    let num_nz = a.start[if lp.a.format == matrix_format::COLWISE { num_col } else { num_row }] as usize;
    for v in &a.value[..num_nz] {
        let v = v.abs();
        original_matrix_min_value = cmin(original_matrix_min_value, v);
        original_matrix_max_value = cmax(original_matrix_max_value, v);
    }
    let no_scaling = !force_scaling && original_matrix_min_value >= no_min && original_matrix_max_value <= no_max;
    let mut scaled = false;
    if no_scaling {
        if o.highs_analysis_level != 0 {
            log_dev!(
                o.log,
                LogType::Info,
                "Scaling: Matrix has [min, max] values of [%g, %g] within [%g, %g] so no scaling performed\n",
                original_matrix_min_value,
                original_matrix_max_value,
                no_min,
                no_max
            );
        }
    } else {
        a.scale_col.fill(1.0);
        a.scale_row.fill(1.0);
        let equilibration =
            use_scale_strategy == SCALE_EQUILIBRATION || use_scale_strategy == SCALE_FORCED_EQUILIBRATION;
        let mut a = a;
        scaled = if equilibration {
            equilibration_scale_matrix(o, &mut a, num_col, num_row, use_scale_strategy)
        } else {
            max_value_scale_matrix(o, &mut a, num_col, num_row)
        };
        if scaled {
            for i in 0..num_col {
                a.col_lower[i] /= a.scale_col[i];
                a.col_upper[i] /= a.scale_col[i];
                a.col_cost[i] *= a.scale_col[i];
            }
            for i in 0..num_row {
                a.row_lower[i] *= a.scale_row[i];
                a.row_upper[i] *= a.scale_row[i];
            }
            lp.scale_has_scaling = true;
            lp.scale_num_col = lp.num_col;
            lp.scale_num_row = lp.num_row;
            lp.scale_cost = 1.0;
            lp.is_scaled = true;
        }
    }
    lp.scale_strategy = use_scale_strategy;
    scaled
}

fn equilibration_scale_matrix(o: &CLpOptions, a: &mut LpArrays, num_col: usize, num_row: usize, strategy: i32) -> bool {
    let (start, index) = (a.start, a.index);
    let col_cost = &*a.col_cost;
    let (col_scale, row_scale) = (&mut *a.scale_col, &mut *a.scale_row);
    let value = &mut *a.value;
    let mut original_matrix_min_value = INF;
    let mut original_matrix_max_value = 0.0;
    for v in &value[..start[num_col] as usize] {
        let v = v.abs();
        original_matrix_min_value = cmin(original_matrix_min_value, v);
        original_matrix_max_value = cmax(original_matrix_max_value, v);
    }
    let mut min_nonzero_cost = INF;
    for &c in &col_cost[..num_col] {
        if c != 0.0 {
            min_nonzero_cost = cmin(c.abs(), min_nonzero_cost);
        }
    }
    let include_cost_in_scaling = min_nonzero_cost < 0.1;
    let finite_infinity = 1e200;
    let max_allow_scale = 2f64.powf(o.allowed_matrix_scale_factor as f64);
    let min_allow_scale = 1.0 / max_allow_scale;
    let mut row_min_value = vec![finite_infinity; num_row];
    let mut row_max_value = vec![1.0 / finite_infinity; num_row];
    for _ in 0..6 {
        for c in 0..num_col {
            let mut col_min_value = finite_infinity;
            let mut col_max_value = 1.0 / finite_infinity;
            let abs_col_cost = col_cost[c].abs();
            if include_cost_in_scaling && abs_col_cost != 0.0 {
                col_min_value = cmin(col_min_value, abs_col_cost);
                col_max_value = cmax(col_max_value, abs_col_cost);
            }
            let (s, e) = (start[c] as usize, start[c + 1] as usize);
            for k in s..e {
                let v = value[k].abs() * row_scale[index[k] as usize];
                col_min_value = cmin(col_min_value, v);
                col_max_value = cmax(col_max_value, v);
            }
            let col_equilibration = 1.0 / (col_min_value * col_max_value).sqrt();
            col_scale[c] = cmin(cmax(min_allow_scale, col_equilibration), max_allow_scale);
            for k in s..e {
                let r = index[k] as usize;
                let v = value[k].abs() * col_scale[c];
                row_min_value[r] = cmin(row_min_value[r], v);
                row_max_value[r] = cmax(row_max_value[r], v);
            }
        }
        for r in 0..num_row {
            let row_equilibration = 1.0 / (row_min_value[r] * row_max_value[r]).sqrt();
            row_scale[r] = cmin(cmax(min_allow_scale, row_equilibration), max_allow_scale);
        }
        row_min_value.fill(finite_infinity);
        row_max_value.fill(1.0 / finite_infinity);
    }
    let mut min_col_scale = finite_infinity;
    let mut max_col_scale = 1.0 / finite_infinity;
    let mut min_row_scale = finite_infinity;
    let mut max_row_scale = 1.0 / finite_infinity;
    let log2 = 2f64.ln();
    for s in col_scale[..num_col].iter_mut() {
        *s = 2f64.powf((s.ln() / log2 + 0.5).floor());
        min_col_scale = cmin(*s, min_col_scale);
        max_col_scale = cmax(*s, max_col_scale);
    }
    for s in row_scale[..num_row].iter_mut() {
        *s = 2f64.powf((s.ln() / log2 + 0.5).floor());
        min_row_scale = cmin(*s, min_row_scale);
        max_row_scale = cmax(*s, max_row_scale);
    }
    let mut matrix_min_value = finite_infinity;
    let mut matrix_max_value = 0.0;
    let mut min_original_col_equilibration = finite_infinity;
    let mut sum_original_log_col_equilibration = 0.0;
    let mut max_original_col_equilibration = 0.0;
    let mut min_original_row_equilibration = finite_infinity;
    let mut sum_original_log_row_equilibration = 0.0;
    let mut max_original_row_equilibration = 0.0;
    let mut min_col_equilibration = finite_infinity;
    let mut sum_log_col_equilibration = 0.0;
    let mut max_col_equilibration = 0.0;
    let mut min_row_equilibration = finite_infinity;
    let mut sum_log_row_equilibration = 0.0;
    let mut max_row_equilibration = 0.0;
    let mut original_row_min_value = vec![finite_infinity; num_row];
    let mut original_row_max_value = vec![1.0 / finite_infinity; num_row];
    row_min_value.fill(finite_infinity);
    row_max_value.fill(1.0 / finite_infinity);
    for c in 0..num_col {
        let mut original_col_min_value = finite_infinity;
        let mut original_col_max_value = 1.0 / finite_infinity;
        let mut col_min_value = finite_infinity;
        let mut col_max_value = 1.0 / finite_infinity;
        for k in start[c] as usize..start[c + 1] as usize {
            let r = index[k] as usize;
            let original_value = value[k].abs();
            original_col_min_value = cmin(original_value, original_col_min_value);
            original_col_max_value = cmax(original_value, original_col_max_value);
            original_row_min_value[r] = cmin(original_row_min_value[r], original_value);
            original_row_max_value[r] = cmax(original_row_max_value[r], original_value);
            value[k] *= col_scale[c] * row_scale[r];
            let v = value[k].abs();
            col_min_value = cmin(v, col_min_value);
            col_max_value = cmax(v, col_max_value);
            row_min_value[r] = cmin(row_min_value[r], v);
            row_max_value[r] = cmax(row_max_value[r], v);
        }
        matrix_min_value = cmin(matrix_min_value, col_min_value);
        matrix_max_value = cmax(matrix_max_value, col_max_value);
        let original_col_equilibration = 1.0 / (original_col_min_value * original_col_max_value).sqrt();
        min_original_col_equilibration = cmin(original_col_equilibration, min_original_col_equilibration);
        sum_original_log_col_equilibration += original_col_equilibration.ln();
        max_original_col_equilibration = cmax(original_col_equilibration, max_original_col_equilibration);
        let col_equilibration = 1.0 / (col_min_value * col_max_value).sqrt();
        min_col_equilibration = cmin(col_equilibration, min_col_equilibration);
        sum_log_col_equilibration += col_equilibration.ln();
        max_col_equilibration = cmax(col_equilibration, max_col_equilibration);
    }
    for r in 0..num_row {
        let original_row_equilibration = 1.0 / (original_row_min_value[r] * original_row_max_value[r]).sqrt();
        min_original_row_equilibration = cmin(original_row_equilibration, min_original_row_equilibration);
        sum_original_log_row_equilibration += original_row_equilibration.ln();
        max_original_row_equilibration = cmax(original_row_equilibration, max_original_row_equilibration);
        let row_equilibration = 1.0 / (row_min_value[r] * row_max_value[r]).sqrt();
        min_row_equilibration = cmin(row_equilibration, min_row_equilibration);
        sum_log_row_equilibration += row_equilibration.ln();
        max_row_equilibration = cmax(row_equilibration, max_row_equilibration);
    }
    let geomean_original_col_equilibration = (sum_original_log_col_equilibration / num_col as f64).exp();
    let geomean_original_row_equilibration = (sum_original_log_row_equilibration / num_row as f64).exp();
    let geomean_col_equilibration = (sum_log_col_equilibration / num_col as f64).exp();
    let geomean_row_equilibration = (sum_log_row_equilibration / num_row as f64).exp();
    let log = &o.log;
    let dev = o.log_dev_level != 0;
    if dev {
        log_dev!(
            log,
            LogType::Info,
            "Scaling: Original equilibration: min/mean/max %11.4g/%11.4g/%11.4g (cols); min/mean/max %11.4g/%11.4g/%11.4g (rows)\n",
            min_original_col_equilibration,
            geomean_original_col_equilibration,
            max_original_col_equilibration,
            min_original_row_equilibration,
            geomean_original_row_equilibration,
            max_original_row_equilibration
        );
        log_dev!(
            log,
            LogType::Info,
            "Scaling: Final    equilibration: min/mean/max %11.4g/%11.4g/%11.4g (cols); min/mean/max %11.4g/%11.4g/%11.4g (rows)\n",
            min_col_equilibration,
            geomean_col_equilibration,
            max_col_equilibration,
            min_row_equilibration,
            geomean_row_equilibration,
            max_row_equilibration
        );
    }
    let geomean_original_col = cmax(geomean_original_col_equilibration, 1.0 / geomean_original_col_equilibration);
    let geomean_original_row = cmax(geomean_original_row_equilibration, 1.0 / geomean_original_row_equilibration);
    let geomean_col = cmax(geomean_col_equilibration, 1.0 / geomean_col_equilibration);
    let geomean_row = cmax(geomean_row_equilibration, 1.0 / geomean_row_equilibration);
    let mean_equilibration_improvement =
        ((geomean_original_col * geomean_original_row) / (geomean_col * geomean_row)).sqrt();
    let original_col_ratio = max_original_col_equilibration / min_original_col_equilibration;
    let original_row_ratio = max_original_row_equilibration / min_original_row_equilibration;
    let col_ratio = max_col_equilibration / min_col_equilibration;
    let row_ratio = max_row_equilibration / min_row_equilibration;
    let extreme_equilibration_improvement = (original_col_ratio + original_row_ratio) / (col_ratio + row_ratio);
    let matrix_value_ratio = matrix_max_value / matrix_min_value;
    let original_matrix_value_ratio = original_matrix_max_value / original_matrix_min_value;
    let matrix_value_ratio_improvement = original_matrix_value_ratio / matrix_value_ratio;
    if dev {
        log_dev!(
            log,
            LogType::Info,
            "Scaling: Extreme equilibration improvement =      ( %11.4g + %11.4g) / ( %11.4g + %11.4g)  =      %11.4g / %11.4g  = %11.4g\n",
            original_col_ratio,
            original_row_ratio,
            col_ratio,
            row_ratio,
            original_col_ratio + original_row_ratio,
            col_ratio + row_ratio,
            extreme_equilibration_improvement
        );
        log_dev!(
            log,
            LogType::Info,
            "Scaling: Mean    equilibration improvement = sqrt(( %11.4g * %11.4g) / ( %11.4g * %11.4g)) = sqrt(%11.4g / %11.4g) = %11.4g\n",
            geomean_original_col,
            geomean_original_row,
            geomean_col,
            geomean_row,
            geomean_original_col * geomean_original_row,
            geomean_col * geomean_row,
            mean_equilibration_improvement
        );
        log_dev!(
            log,
            LogType::Info,
            "Scaling: Yields [min, max, ratio] matrix values of [%0.4g, %0.4g, %0.4g]; Originally [%0.4g, %0.4g, %0.4g]: Improvement of %0.4g\n",
            matrix_min_value,
            matrix_max_value,
            matrix_value_ratio,
            original_matrix_min_value,
            original_matrix_max_value,
            original_matrix_value_ratio,
            matrix_value_ratio_improvement
        );
        log_dev!(
            log,
            LogType::Info,
            "Scaling: Improves    mean equilibration by a factor %0.4g\n",
            mean_equilibration_improvement
        );
        log_dev!(
            log,
            LogType::Info,
            "Scaling: Improves extreme equilibration by a factor %0.4g\n",
            extreme_equilibration_improvement
        );
        log_dev!(
            log,
            LogType::Info,
            "Scaling: Improves max/min matrix values by a factor %0.4g\n",
            matrix_value_ratio_improvement
        );
    }
    let possibly_abandon_scaling = strategy != SCALE_FORCED_EQUILIBRATION;
    let improvement_factor =
        extreme_equilibration_improvement * mean_equilibration_improvement * matrix_value_ratio_improvement;
    let improvement_factor_required = 1.0;
    let poor_improvement = improvement_factor < improvement_factor_required;
    if possibly_abandon_scaling && poor_improvement {
        for c in 0..num_col {
            for k in start[c] as usize..start[c + 1] as usize {
                value[k] /= col_scale[c] * row_scale[index[k] as usize];
            }
        }
        if dev {
            log_dev!(
                log,
                LogType::Info,
                "Scaling: Improvement factor %0.4g < %0.4g required, so no scaling applied\n",
                improvement_factor,
                improvement_factor_required
            );
        }
        return false;
    }
    if dev {
        log_dev!(
            log,
            LogType::Info,
            "Scaling: Factors are in [%0.4g, %0.4g] for columns and in [%0.4g, %0.4g] for rows\n",
            min_col_scale,
            max_col_scale,
            min_row_scale,
            max_row_scale
        );
        log_dev!(
            log,
            LogType::Info,
            "Scaling: Improvement factor is %0.4g >= %0.4g so scale LP\n",
            improvement_factor,
            improvement_factor_required
        );
        if extreme_equilibration_improvement < 1.0 {
            log_dev!(
                log,
                LogType::Warning,
                "Scaling: Applying scaling with extreme improvement of %0.4g\n",
                extreme_equilibration_improvement
            );
        }
        if mean_equilibration_improvement < 1.0 {
            log_dev!(
                log,
                LogType::Warning,
                "Scaling: Applying scaling with mean improvement of %0.4g\n",
                mean_equilibration_improvement
            );
        }
        if matrix_value_ratio_improvement < 1.0 {
            log_dev!(
                log,
                LogType::Warning,
                "Scaling: Applying scaling with matrix value ratio improvement of %0.4g\n",
                matrix_value_ratio_improvement
            );
        }
        if improvement_factor < 10.0 * improvement_factor_required {
            log_dev!(
                log,
                LogType::Warning,
                "Scaling: Applying scaling with improvement factor %0.4g < 10*(%0.4g) improvement\n",
                improvement_factor,
                improvement_factor_required
            );
        }
    }
    true
}

fn max_value_scale_matrix(o: &CLpOptions, a: &mut LpArrays, num_col: usize, num_row: usize) -> bool {
    let (start, index) = (a.start, a.index);
    let (col_scale, row_scale) = (&mut *a.scale_col, &mut *a.scale_row);
    let value = &mut *a.value;
    let log2 = 2f64.ln();
    let max_allow_scale = 2f64.powf(o.allowed_matrix_scale_factor as f64);
    let min_allow_scale = 1.0 / max_allow_scale;
    let mut min_row_scale = INF;
    let mut max_row_scale = 0.0;
    let mut original_matrix_min_value = INF;
    let mut original_matrix_max_value = 0.0;
    let mut row_max_value = vec![0.0; num_row];
    for c in 0..num_col {
        for k in start[c] as usize..start[c + 1] as usize {
            let r = index[k] as usize;
            let v = value[k].abs();
            row_max_value[r] = cmax(row_max_value[r], v);
            original_matrix_min_value = cmin(original_matrix_min_value, v);
            original_matrix_max_value = cmax(original_matrix_max_value, v);
        }
    }
    for r in 0..num_row {
        if row_max_value[r] != 0.0 {
            let mut s = 1.0 / row_max_value[r];
            s = 2f64.powf((s.ln() / log2 + 0.5).floor());
            s = cmin(cmax(min_allow_scale, s), max_allow_scale);
            min_row_scale = cmin(s, min_row_scale);
            max_row_scale = cmax(s, max_row_scale);
            row_scale[r] = s;
        }
    }
    let mut min_col_scale = INF;
    let mut max_col_scale = 0.0;
    let mut matrix_min_value = INF;
    let mut matrix_max_value = 0.0;
    for c in 0..num_col {
        let mut col_max_value = 0.0;
        let (s0, e0) = (start[c] as usize, start[c + 1] as usize);
        for k in s0..e0 {
            value[k] *= row_scale[index[k] as usize];
            col_max_value = cmax(col_max_value, value[k].abs());
        }
        if col_max_value != 0.0 {
            let mut s = 1.0 / col_max_value;
            s = 2f64.powf((s.ln() / log2 + 0.5).floor());
            s = cmin(cmax(min_allow_scale, s), max_allow_scale);
            min_col_scale = cmin(s, min_col_scale);
            max_col_scale = cmax(s, max_col_scale);
            col_scale[c] = s;
            for k in s0..e0 {
                value[k] *= col_scale[c];
                let v = value[k].abs();
                matrix_min_value = cmin(matrix_min_value, v);
                matrix_max_value = cmax(matrix_max_value, v);
            }
        }
    }
    let matrix_value_ratio = matrix_max_value / matrix_min_value;
    let original_matrix_value_ratio = original_matrix_max_value / original_matrix_min_value;
    let matrix_value_ratio_improvement = original_matrix_value_ratio / matrix_value_ratio;
    let improvement_factor = matrix_value_ratio_improvement;
    let improvement_factor_required = 1.0;
    let log = &o.log;
    let dev = o.log_dev_level != 0;
    if improvement_factor <= improvement_factor_required {
        for c in 0..num_col {
            for k in start[c] as usize..start[c + 1] as usize {
                value[k] /= col_scale[c] * row_scale[index[k] as usize];
            }
        }
        if dev {
            log_dev!(
                log,
                LogType::Info,
                "Scaling: Improvement factor %0.4g < %0.4g required, so no scaling applied\n",
                improvement_factor,
                improvement_factor_required
            );
        }
        return false;
    }
    if dev {
        log_dev!(
            log,
            LogType::Info,
            "Scaling: Factors are in [%0.4g, %0.4g] for columns and in [%0.4g, %0.4g] for rows\n",
            min_col_scale,
            max_col_scale,
            min_row_scale,
            max_row_scale
        );
        log_dev!(
            log,
            LogType::Info,
            "Scaling: Yields [min, max, ratio] matrix values of [%0.4g, %0.4g, %0.4g]; Originally [%0.4g, %0.4g, %0.4g]: Improvement of %0.4g\n",
            matrix_min_value,
            matrix_max_value,
            matrix_value_ratio,
            original_matrix_min_value,
            original_matrix_max_value,
            original_matrix_value_ratio,
            matrix_value_ratio_improvement
        );
    }
    true
}

/// HighsSparseMatrix::applyScale (apply) or unapplyScale
fn matrix_scale(lp: &CLp, apply: bool) {
    let a = arrays(lp);
    let colwise = lp.a.format == matrix_format::COLWISE;
    let num_vec = if colwise { lp.a.num_col } else { lp.a.num_row } as usize;
    for v in 0..num_vec {
        for k in a.start[v] as usize..a.start[v + 1] as usize {
            let o = a.index[k] as usize;
            let (c, r) = if colwise { (v, o) } else { (o, v) };
            if apply {
                a.value[k] *= a.scale_col[c] * a.scale_row[r];
            } else {
                a.value[k] /= a.scale_col[c] * a.scale_row[r];
            }
        }
    }
}

/// HighsLp::applyScale
pub fn apply_scale(lp: &mut CLp) {
    if lp.is_scaled {
        return;
    }
    lp.is_scaled = false;
    if lp.scale_has_scaling {
        let a = arrays(lp);
        for i in 0..lp.num_col as usize {
            a.col_lower[i] /= a.scale_col[i];
            a.col_upper[i] /= a.scale_col[i];
            a.col_cost[i] *= a.scale_col[i];
        }
        for i in 0..lp.num_row as usize {
            a.row_lower[i] *= a.scale_row[i];
            a.row_upper[i] *= a.scale_row[i];
        }
        matrix_scale(lp, true);
        lp.is_scaled = true;
    }
}

/// HighsLp::unapplyScale
pub fn unapply_scale(lp: &mut CLp) {
    if !lp.is_scaled {
        return;
    }
    let a = arrays(lp);
    for i in 0..lp.num_col as usize {
        a.col_lower[i] *= a.scale_col[i];
        a.col_upper[i] *= a.scale_col[i];
        a.col_cost[i] /= a.scale_col[i];
    }
    for i in 0..lp.num_row as usize {
        a.row_lower[i] /= a.scale_row[i];
        a.row_upper[i] /= a.scale_row[i];
    }
    matrix_scale(lp, false);
    lp.is_scaled = false;
}

/// cleanBounds
pub fn clean_bounds(lp: &mut CLp, o: &CLpOptions) -> Status {
    let a = arrays(lp);
    let log = &o.log;
    let tol = o.primal_feasibility_tolerance;
    let mut max_residual = 0.0;
    let mut num_change = 0;
    for (kind, lower, upper, n) in [
        ("Column", a.col_lower, a.col_upper, lp.num_col),
        ("Row", a.row_lower, a.row_upper, lp.num_row),
    ] {
        for i in 0..n as usize {
            let residual = lower[i] - upper[i];
            if residual > tol {
                let fmt = if kind == "Column" {
                    "Column %d has inconsistent bounds [%g, %g] (residual = %g) after presolve\n"
                } else {
                    "Row %d has inconsistent bounds [%g, %g] (residual = %g) after presolve\n"
                };
                log_user!(log, LogType::Error, fmt, i as i32, lower[i], upper[i], residual);
                return Status::Error;
            } else if residual > 0.0 {
                num_change += 1;
                max_residual = cmax(residual, max_residual);
                let mid = 0.5 * (lower[i] + upper[i]);
                lower[i] = mid;
                upper[i] = mid;
            }
        }
    }
    if num_change != 0 {
        log_user!(
            log,
            LogType::Warning,
            "Resolved %d inconsistent bounds (maximum residual = %9.4g) after presolve\n",
            num_change,
            max_residual
        );
        return Status::Warning;
    }
    Status::Ok
}
