//! The PDLP solver cuPDLP-C (highs/pdlp/cupdlp, CPU build) and the glue of
//! CupdlpWrapper.cpp that builds its LP from a HiGHS LP and returns the
//! solution. C++ keeps reading the HiGHS options and mapping the
//! termination code to a model status (highs/pdlp/CupdlpWrapperRs.cpp).
//!
//! The port does the floating-point operations of libhighs in the same
//! order, so the iterations, objectives and residuals are bit-identical
//! (see linalg.rs for the reductions and solver.rs for the contractions).
//! Not ported, being unreachable from HiGHS: the GPU paths, L2 scaling,
//! the Malitsky-Pock step, the "CPU" restart (a no-op in C), the JSON and
//! solution writers and the debug logs.

pub mod ffi;
mod linalg;
mod scaling;
mod solver;

#[cfg(test)]
mod tests;

use linalg::{nrminf, Sparse};
use scaling::{ScaledLp, Scaling};
pub(crate) use solver::TermCode;
use solver::{Pdhg, Problem};
use std::ffi::{c_char, CString};

/// ConstraintType of HConst.h
const EQ: i32 = 0;
const LEQ: i32 = 1;
const GEQ: i32 = 2;
const BOUND: i32 = 3;

/// The settings that CupdlpWrapper.cpp derives from the HiGHS options
/// (getUserParamsFromOptions); mirrored by PdlpRsParams in C++
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub primal_tol: f64,
    pub dual_tol: f64,
    pub gap_tol: f64,
    pub time_lim: f64,
    pub iter_lim: i32,
    pub log_level: i32,
    /// 1: scale the LP
    pub scaling: i32,
    /// 0: constant step from the power method; 2: adaptive step
    pub line_search: i32,
    /// 1: restarts, 0: none
    pub restart: i32,
}

/// cupdlp_printf: through a C printf so that the output interleaves with
/// that of HiGHS; to stdout without one (tests)
pub(crate) struct Log {
    pub(crate) level: i32,
    print: Option<extern "C" fn(*const c_char)>,
}

impl Log {
    pub(crate) fn out(&self, s: &str) {
        match self.print {
            Some(print) => {
                let s = CString::new(s).expect("no NUL in log text");
                print(s.as_ptr());
            }
            None => print!("{s}"),
        }
    }

    /// Output at log level 1 and above
    pub(crate) fn info(&self, s: &str) {
        if self.level > 0 {
            self.out(s);
        }
    }
}

/// printf's "%.{prec}e" (macOS prints a NaN as "nan", without sign)
fn e(v: f64, prec: usize) -> String {
    if v.is_nan() {
        "nan".into()
    } else {
        crate::ipx::fmt::sci_raw(v, prec)
    }
}

/// The '+' flag of printf for a formatted number
fn plus(s: String) -> String {
    if s.starts_with('-') || s == "nan" {
        s
    } else {
        format!("+{s}")
    }
}

/// printf's "%.{prec}g"
fn g(v: f64, prec: usize) -> String {
    if !v.is_finite() {
        return e(v, 0);
    }
    let p = prec.max(1);
    let strip = |s: String| {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s
        }
    };
    let s = format!("{v:.*e}", p - 1);
    let (mantissa, exp) = s.split_once('e').unwrap();
    let x: i32 = exp.parse().unwrap();
    if x < -4 || x >= p as i32 {
        let sign = if x < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", strip(mantissa.to_string()), x.abs())
    } else {
        strip(format!("{:.*}", (p as i32 - 1 - x) as usize, v))
    }
}

/// A HiGHS LP, column-wise
pub struct Lp<'a> {
    pub start: &'a [i32],
    pub index: &'a [i32],
    pub value: &'a [f64],
    pub col_cost: &'a [f64],
    pub col_lower: &'a [f64],
    pub col_upper: &'a [f64],
    pub row_lower: &'a [f64],
    pub row_upper: &'a [f64],
    pub offset: f64,
    /// 1 to minimize, -1 to maximize
    pub sense: f64,
}

/// The HiGHS solution: the hot start when both its values and duals are
/// valid, and the result
pub struct Solution<'a> {
    pub col_value: &'a mut [f64],
    pub col_dual: &'a mut [f64],
    pub row_value: &'a mut [f64],
    pub row_dual: &'a mut [f64],
    pub value_valid: bool,
    pub dual_valid: bool,
}

/// How the rows of the HiGHS LP map to cuPDLP-C rows and slack columns
struct Formulated {
    ncols_origin: usize,
    constraint_type: Vec<i32>,
    new_idx: Vec<usize>,
}

/// formulateLP_highs: the LP as min c'x, A_eq x = b_eq, A_ineq x >= b_ineq,
/// l <= x <= u, with a slack column -z in each boxed (or free) row and the
/// equations first; the scaled-dependent data of Problem is filled later
fn formulate(lp: &Lp, log: &Log) -> (Problem, Formulated) {
    let n0 = lp.col_cost.len();
    let m = lp.row_lower.len();
    let (lhs, rhs_clp) = (lp.row_lower, lp.row_upper);
    let mut ncols = n0;
    let mut neqs = 0;
    let mut types = vec![0; m];
    for i in 0..m {
        let has_lower = lhs[i] > -1e20;
        let has_upper = rhs_clp[i] < 1e20;
        types[i] = if has_lower && has_upper && lhs[i] == rhs_clp[i] {
            neqs += 1;
            EQ
        } else if has_lower && !has_upper {
            GEQ
        } else if !has_lower && has_upper {
            LEQ
        } else {
            if !(has_lower && has_upper) {
                log.out(&format!(
                    "Warning: constraint {i} has no lower and upper bound\n"
                ));
            }
            ncols += 1;
            neqs += 1;
            BOUND
        };
    }

    let mut cost = vec![0.0; ncols];
    let mut lower = vec![0.0; ncols];
    let mut upper = vec![0.0; ncols];
    for i in 0..n0 {
        cost[i] = lp.col_cost[i] * lp.sense;
        lower[i] = lp.col_lower[i];
        upper[i] = lp.col_upper[i];
    }
    let mut j = n0;
    for i in 0..m {
        if types[i] == BOUND {
            lower[j] = lhs[i];
            upper[j] = rhs_clp[i];
            j += 1;
        }
    }
    for i in 0..ncols {
        if lower[i] < -1e20 {
            lower[i] = f64::NEG_INFINITY;
        }
        if upper[i] > 1e20 {
            upper[i] = f64::INFINITY;
        }
    }

    // Rows: equations and boxed rows first, then the inequalities as >=
    let mut rhs = vec![0.0; m];
    let mut new_idx = vec![0; m];
    let mut j = 0;
    for i in 0..m {
        if types[i] == EQ || types[i] == BOUND {
            rhs[j] = if types[i] == EQ { lhs[i] } else { 0.0 };
            new_idx[i] = j;
            j += 1;
        }
    }
    for i in 0..m {
        if types[i] == LEQ || types[i] == GEQ {
            rhs[j] = if types[i] == LEQ { -rhs_clp[i] } else { lhs[i] };
            new_idx[i] = j;
            j += 1;
        }
    }

    // The matrix, each column in the same row order
    let nnz = lp.start[n0] as usize + (ncols - n0);
    let mut csc = Sparse {
        start: Vec::with_capacity(ncols + 1),
        index: Vec::with_capacity(nnz),
        value: Vec::with_capacity(nnz),
    };
    csc.start.push(0);
    for c in 0..n0 {
        let range = lp.start[c] as usize..lp.start[c + 1] as usize;
        for first in [true, false] {
            for p in range.clone() {
                let i = lp.index[p] as usize;
                let ty = types[i];
                if (ty == EQ || ty == BOUND) == first {
                    csc.index.push(new_idx[i] as u32);
                    csc.value
                        .push(if ty == LEQ { -lp.value[p] } else { lp.value[p] });
                }
            }
        }
        csc.start.push(csc.index.len());
    }
    for i in 0..m {
        if types[i] == BOUND {
            csc.index.push(new_idx[i] as u32);
            csc.value.push(-1.0);
            csc.start.push(csc.index.len());
        }
    }

    let problem = Problem {
        nrows: m,
        ncols,
        neqs,
        csc,
        csr: Sparse::default(),
        mat_inf_norm: 0.0,
        cost,
        rhs,
        lower,
        upper,
        has_lower: Vec::new(),
        has_upper: Vec::new(),
        offset: lp.offset,
        sense: lp.sense,
    };
    let f = Formulated {
        ncols_origin: n0,
        constraint_type: types,
        new_idx,
    };
    (problem, f)
}

const HUGE_CUPDHG: &str = r"
  ____ _   _ ____  ____  _     ____
 / ___| | | |  _ \|  _ \| |   |  _ \
| |   | | | | |_) | | | | |   | |_) |
| |___| |_| |  __/| |_| | |___|  __/
 \____|\___/|_|   |____/|_____|_|

";

/// solveLpCupdlp up to the model status: (termination code, iterations)
pub(crate) fn solve(lp: &Lp, params: &Params, log: &Log, sol: &mut Solution) -> (TermCode, i32) {
    log.info("Solving with cuPDLP-C\n");
    let (mut p, f) = formulate(lp, log);
    let mut sc = Scaling::new(log, &p.cost, &p.rhs);
    let scaling_begin = solver::time_stamp();
    let mut slp = ScaledLp {
        csc: &mut p.csc,
        cost: &mut p.cost,
        lower: &mut p.lower,
        upper: &mut p.upper,
        rhs: &mut p.rhs,
    };
    sc.scale(log, params.scaling != 0, &mut slp);
    let scaling_time = solver::time_stamp() - scaling_begin;

    // problem_alloc
    p.csr = p.csc.transpose(p.nrows);
    p.mat_inf_norm = nrminf(&p.csc.value);
    p.has_lower = p
        .lower
        .iter()
        .map(|&l| if l > f64::NEG_INFINITY { 1.0 } else { 0.0 })
        .collect();
    p.has_upper = p
        .upper
        .iter()
        .map(|&u| if u < f64::INFINITY { 1.0 } else { 0.0 })
        .collect();

    // LP_SolvePDHG
    let mut w = Pdhg::new(p, sc, log, params);
    w.t.scaling_time = scaling_time;
    if log.level > 1 {
        log.out(HUGE_CUPDHG);
    }
    if sol.value_valid && sol.dual_valid {
        w.presolve(&f, sol);
    }
    let has_variables = sol.value_valid || sol.dual_valid;
    if has_variables {
        log.info("Hot starting with given column primal values and row dual values\n");
    }
    w.solve(has_variables);
    w.postsolve(&f, sol);
    sol.value_valid = true;
    sol.dual_valid = true;
    w.result()
}
