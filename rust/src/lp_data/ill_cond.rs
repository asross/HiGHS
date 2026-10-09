//! Highs::computeIllConditioning with formIllConditioningLp0/1
//! (HighsInterface.cpp): the LP whose solution gives the multipliers of
//! a near-singular combination of the basis matrix's rows (constraint
//! view) or columns (column view), and the report of those multipliers.
//!
//! C++ (lp_data/HighsIllCondRust.cpp) clears the result, makes the
//! incumbent matrix column-wise and passes views, and stores the records
//! through `op`; the analysis LP is solved by a silent LP solver (an
//! LpHandle, as the C++ ran a silent Highs).

use super::ffi::{CLp, RsMut};
use super::{Log, LogType, Status, INF};
use crate::log_user;
use crate::util::printf::sprintf;
use std::ffi::{c_char, c_void, CStr};

const BASIC: u8 = 1;
// HighsModelStatus
const MS_OPTIMAL: i32 = 7;
const MS_INFEASIBLE: i32 = 8;

/// What computeIllConditioning works on (HighsIllCondRust.cpp)
#[repr(C)]
pub struct CIllHost {
    pub ctx: *mut c_void,
    /// Add the record (index, x) (code 1)
    pub op: unsafe extern "C" fn(*mut c_void, i32, *mut c_void, i32, f64),
    pub log: Log,
    pub lp: CLp,
    pub col_status: RsMut<u8>,
    pub row_status: RsMut<u8>,
    pub col_names: RsMut<*const c_char>,
    pub row_names: RsMut<*const c_char>,
    pub constraint: bool,
    pub method: i32,
    pub bound: f64,
}

/// The analysis LP being formed (column-wise)
#[derive(Default)]
struct Lp {
    num_col: usize,
    num_row: usize,
    cost: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    row_lower: Vec<f64>,
    row_upper: Vec<f64>,
    start: Vec<i32>,
    index: Vec<i32>,
    value: Vec<f64>,
}

impl Lp {
    fn col(&mut self, cost: f64, lower: f64, upper: f64) {
        self.cost.push(cost);
        self.lower.push(lower);
        self.upper.push(upper);
    }
    fn el(&mut self, i: usize, v: f64) {
        self.index.push(i as i32);
        self.value.push(v);
    }
    fn end_col(&mut self) {
        self.start.push(self.index.len() as i32);
    }
    /// ensureRowwise of the num_row x num_col column-wise matrix, then
    /// read as column-wise (the transpose)
    fn transpose(&mut self, num_col: usize, num_row: usize) {
        let num_nz = self.start[num_col] as usize;
        if num_nz == 0 {
            self.start = vec![0; num_row + 1];
            self.index.clear();
            self.value.clear();
            return;
        }
        let (a_start, a_index, a_value) = (self.start.clone(), self.index.clone(), self.value.clone());
        self.start.resize(num_row + 1, 0);
        self.index.resize(num_nz, 0);
        self.value.resize(num_nz, 0.0);
        let mut length = vec![0i32; num_row];
        for el in a_start[0] as usize..num_nz {
            length[a_index[el] as usize] += 1;
        }
        self.start[0] = 0;
        for r in 0..num_row {
            self.start[r + 1] = self.start[r] + length[r];
        }
        for c in 0..num_col {
            for el in a_start[c] as usize..a_start[c + 1] as usize {
                let r = a_index[el] as usize;
                let to = self.start[r] as usize;
                self.index[to] = c as i32;
                self.value[to] = a_value[el];
                self.start[r] += 1;
            }
        }
        self.start[0] = 0;
        for r in 0..num_row {
            self.start[r + 1] = self.start[r] + length[r];
        }
    }
}

struct Incumbent<'a> {
    num_col: usize,
    num_row: usize,
    start: &'a [i32],
    index: &'a [i32],
    value: &'a [f64],
    col_status: &'a [u8],
    row_status: &'a [u8],
}

/// The basic columns, each with its matrix column (`extra` appends
/// entries), then the basic slacks; the basic variables in order
fn basic_columns(inc: &Incumbent, lp: &mut Lp, extra: impl Fn(&mut Lp, usize)) -> Vec<usize> {
    let mut basic_var = Vec::new();
    let mut k = 0;
    for col in 0..inc.num_col {
        if inc.col_status[col] != BASIC {
            continue;
        }
        basic_var.push(col);
        lp.col(0.0, -INF, INF);
        for el in inc.start[col] as usize..inc.start[col + 1] as usize {
            lp.el(inc.index[el] as usize, inc.value[el]);
        }
        extra(lp, k);
        lp.end_col();
        k += 1;
    }
    for row in 0..inc.num_row {
        if inc.row_status[row] != BASIC {
            continue;
        }
        basic_var.push(inc.num_col + row);
        lp.col(0.0, -INF, INF);
        lp.el(row, -1.0);
        extra(lp, k);
        lp.end_col();
        k += 1;
    }
    basic_var
}

/// formIllConditioningLp0
fn form_lp0(inc: &Incumbent, constraint: bool) -> (Lp, Vec<usize>) {
    let m = inc.num_row;
    let mut lp = Lp { start: vec![0], ..Default::default() };
    lp.num_row = m + 1;
    lp.row_lower = vec![0.0; m];
    lp.row_upper = vec![0.0; m];
    lp.row_lower.push(1.0);
    lp.row_upper.push(1.0);
    let e_row = m;
    let basic_var = basic_columns(inc, &mut lp, |lp, _| {
        if !constraint {
            lp.el(e_row, 1.0);
        }
    });
    if constraint {
        for row in 0..m {
            lp.el(row, 1.0);
        }
        lp.end_col();
        lp.transpose(m + 1, m);
    }
    for row in 0..m {
        lp.col(1.0, 0.0, INF);
        lp.el(row, 1.0);
        lp.end_col();
        lp.col(1.0, 0.0, INF);
        lp.el(row, -1.0);
        lp.end_col();
    }
    lp.num_col = 3 * m;
    (lp, basic_var)
}

/// formIllConditioningLp1 (Klotz14's formulation)
fn form_lp1(inc: &Incumbent, constraint: bool, bound: f64) -> (Lp, Vec<usize>) {
    let m = inc.num_row;
    let (c4, c1, c7, c6, c5) = (0, m, 2 * m, 3 * m, 3 * m + 1);
    let mut lp = Lp { start: vec![0], ..Default::default() };
    lp.row_lower = vec![0.0; c6];
    lp.row_upper = vec![0.0; c6];
    let basic_var = basic_columns(inc, &mut lp, |lp, k| {
        if !constraint {
            lp.el(c1 + k, 1.0);
            lp.el(c6, 1.0);
        }
    });
    if constraint {
        for row in 0..m {
            lp.el(row, 1.0);
            lp.end_col();
        }
        for _ in 0..m {
            lp.end_col();
        }
        for row in 0..m {
            lp.el(row, 1.0);
        }
        lp.end_col();
        lp.transpose(c6 + 1, m);
    }
    lp.num_row = 3 * m + 2;
    for row in 0..m {
        for w in [-1.0, 1.0] {
            lp.col(0.0, 0.0, INF);
            lp.el(c1 + row, w);
            lp.el(c7 + row, 1.0);
            lp.end_col();
        }
    }
    for row in 0..m {
        for w in [-1.0, 1.0] {
            lp.col(0.0, 0.0, INF);
            lp.el(c4 + row, w);
            lp.el(c5, 1.0);
            lp.end_col();
        }
    }
    lp.row_lower.push(1.0);
    lp.row_upper.push(1.0);
    lp.row_lower.push(-INF);
    lp.row_upper.push(bound);
    for row in 0..m {
        for w in [-1.0, 1.0] {
            lp.col(1.0, 0.0, INF);
            lp.el(c7 + row, w);
            lp.end_col();
        }
    }
    lp.num_col = 7 * m;
    (lp, basic_var)
}

/// Highs::run of a silent Highs whose model's members are the analysis
/// LP (set in place: no passModel): the run's status, the model status,
/// the objective, the first `m` column values and the last row value
fn solve_analysis_lp(a: Lp, m: usize) -> (Status, i32, f64, Vec<f64>, f64) {
    use super::lp_handle::LpHandle;
    use super::opts::OptValue;
    let mut h = LpHandle::new();
    h.set_option("output_flag", OptValue::Bool(false));
    let lp = &mut h.model.g;
    lp.num_col = a.num_col as i32;
    lp.num_row = a.num_row as i32;
    lp.col_cost = a.cost;
    lp.col_lower = a.lower;
    lp.col_upper = a.upper;
    lp.row_lower = a.row_lower;
    lp.row_upper = a.row_upper;
    lp.a.num_col = a.num_col as i32;
    lp.a.num_row = a.num_row as i32;
    lp.a.start = a.start;
    lp.a.index = a.index;
    lp.a.value = a.value;
    let run_status = h.run_lp();
    let s = h.solution();
    let mut col_value = vec![0.0; m];
    for (x, &v) in col_value.iter_mut().zip(&s.col_value) {
        *x = v;
    }
    let num_row = h.model.num_row as usize;
    let last_row_value = if s.row_value.len() == num_row && num_row > 0 { s.row_value[num_row - 1] } else { 0.0 };
    (run_status, h.model_status(), h.info().objective_function_value, col_value, last_row_value)
}

/// `ss << x` of a double (precision 6, like %g)
fn g(x: f64) -> String {
    sprintf("%g", &[x.into()])
}

/// The names of an index set, if there is one for each index
unsafe fn names(v: &RsMut<*const c_char>, n: usize) -> Option<Vec<String>> {
    let v = v.get();
    (v.len() == n).then(|| v.iter().map(|&p| CStr::from_ptr(p).to_string_lossy().into_owned()).collect())
}

/// computeIllConditioning
///
/// # Safety
/// The host's views valid, the op with its context
pub unsafe fn compute_ill_conditioning(h: &CIllHost) -> Status {
    let log = &h.log;
    let lp = &h.lp;
    let (n, m) = (lp.num_col as usize, lp.num_row as usize);
    let inc = Incumbent {
        num_col: n,
        num_row: m,
        start: lp.a.start.get(),
        index: lp.a.index.get(),
        value: lp.a.value.get(),
        col_status: h.col_status.get(),
        row_status: h.row_status.get(),
    };
    let (a, basic_var) = if h.method == 0 { form_lp0(&inc, h.constraint) } else { form_lp1(&inc, h.constraint, h.bound) };
    let (run_status, ms, objective, col_value, last_row_value) = solve_analysis_lp(a, m);
    let ty = if h.constraint { "Constraint" } else { "Column" };
    let failed = run_status != Status::Ok
        || (h.method == 0 && ms != MS_OPTIMAL)
        || (h.method == 1 && ms != MS_OPTIMAL && ms != MS_INFEASIBLE);
    if failed {
        log_user!(log, LogType::Info, "\n%s view ill-conditioning analysis has failed\n", ty);
        return Status::Error;
    }
    if h.method == 1 && ms == MS_INFEASIBLE {
        log_user!(
            log,
            LogType::Info,
            "\n%s view ill-conditioning bound of %g is insufficient for analysis: try %g\n",
            ty,
            h.bound,
            1e1 * h.bound
        );
        return Status::Ok;
    }
    let mut norm = 0.0;
    for v in &col_value {
        norm += v.abs();
    }
    let measure = (if h.method == 0 { objective } else { last_row_value }) / norm;
    log_user!(
        log,
        LogType::Info,
        "\n%s view ill-conditioning analysis: 1-norm distance of basis matrix from singularity is estimated to be %g\n",
        ty,
        measure
    );
    let mut abs_list: Vec<(f64, usize)> = Vec::new();
    for (row, v) in col_value.iter().enumerate() {
        let abs = v.abs() / norm;
        if abs <= 1e-6 {
            continue;
        }
        abs_list.push((abs, row));
    }
    abs_list.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap().then(x.1.cmp(&y.1)));
    let records: Vec<(usize, f64)> = abs_list.iter().rev().map(|&(_, row)| (row, col_value[row] / norm)).collect();
    for &(index, multiplier) in &records {
        (h.op)(h.ctx, 1, std::ptr::null_mut(), index as i32, multiplier);
    }
    let row_names = names(&h.row_names, m);
    let col_names = names(&h.col_names, n);
    let row_name = |r: usize| row_names.as_ref().map_or_else(|| format!("R{r}"), |v| v[r].clone());
    let col_name = |c: usize| col_names.as_ref().map_or_else(|| format!("C{c}"), |v| v[c].clone());
    let tol = 1e-8;
    let coefficient = |ss: &mut String, x: f64, first: bool| {
        if x.abs() < tol {
            ss.push_str("+ 0");
        } else if (x - 1.0).abs() < tol {
            ss.push_str(if first { "" } else { "+ " });
        } else if (x + 1.0).abs() < tol {
            ss.push_str(if first { "-" } else { "- " });
        } else if x < 0.0 {
            ss.push_str(if first { "-" } else { "- " });
            ss.push_str(&g(-x));
            ss.push(' ');
        } else {
            ss.push_str(if first { "" } else { "+ " });
            ss.push_str(&g(x));
            ss.push(' ');
        }
    };
    let emit = |ss: &str| log_user!(log, LogType::Info, "%s\n", ss);
    let (row_lower, row_upper) = (lp.row_lower.get(), lp.row_upper.get());
    if h.constraint {
        for &(row, multiplier) in &records {
            let mut ss = String::new();
            let mut newline = false;
            // getRow of the column-wise matrix: the first entry in each column
            let mut entries = Vec::new();
            for col in 0..n {
                if let Some(el) = (inc.start[col] as usize..inc.start[col + 1] as usize).find(|&el| inc.index[el] as usize == row) {
                    entries.push((col, inc.value[el]));
                }
            }
            ss += &format!("(Mu={}){}: ", g(multiplier), row_name(row));
            let (lower, upper) = (row_lower[row], row_upper[row]);
            if lower > -INF && lower != upper {
                ss += &format!("{} <= ", g(lower));
            }
            let num_nz = entries.len();
            for (k, &(col, v)) in entries.iter().enumerate() {
                if newline {
                    ss.push_str("  ");
                    newline = false;
                }
                coefficient(&mut ss, v, k == 0);
                ss += &col_name(col);
                ss.push(' ');
                if ss.len() > 72 && k + 1 < num_nz {
                    emit(&ss);
                    ss.clear();
                    newline = true;
                }
            }
            if upper < INF {
                ss += &format!("{} {}", if lower == upper { "=" } else { "<=" }, g(upper));
            }
            if !ss.is_empty() {
                emit(&ss);
            }
        }
    } else {
        for &(index, multiplier) in &records {
            let mut ss = String::new();
            let mut newline = false;
            let var = basic_var[index];
            if var < n {
                ss += &format!("(Mu={}){}: ", g(multiplier), col_name(var));
                let (first, last) = (inc.start[var] as usize, inc.start[var + 1] as usize);
                for el in first..last {
                    if newline {
                        ss.push_str("  ");
                        newline = false;
                    } else if el > first {
                        ss.push_str(" | ");
                    }
                    coefficient(&mut ss, inc.value[el], true);
                    ss += &row_name(inc.index[el] as usize);
                    if ss.len() > 72 && el + 1 < last {
                        ss.push_str(" | ");
                        emit(&ss);
                        ss.clear();
                        newline = true;
                    }
                }
            } else {
                let row = var - n;
                let name = row_names.as_ref().map_or_else(|| format!("Slack_R{row}"), |v| format!("Slack_{}", v[row]));
                ss += &format!("(Mu={}){}: ", g(multiplier), name);
            }
            if !ss.is_empty() {
                emit(&ss);
            }
        }
    }
    Status::Ok
}

/// # Safety
/// As compute_ill_conditioning
#[no_mangle]
pub unsafe extern "C" fn highs_rs_compute_ill_conditioning(h: *const CIllHost) -> i32 {
    compute_ill_conditioning(&*h) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_matches_ensure_rowwise() {
        // 2 x 3 column-wise [[1, 0, 3], [2, 4, 0]]
        let mut lp = Lp { start: vec![0, 2, 3, 4], index: vec![0, 1, 1, 0], value: vec![1.0, 2.0, 4.0, 3.0], ..Default::default() };
        lp.transpose(3, 2);
        assert_eq!(lp.start, vec![0, 2, 4]);
        assert_eq!(lp.index, vec![0, 2, 0, 1]);
        assert_eq!(lp.value, vec![1.0, 3.0, 2.0, 4.0]);
    }
}
