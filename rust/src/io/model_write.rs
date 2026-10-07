//! Model file writers: MPS (writeMps in HMPSIO.cpp) and LP
//! (FilereaderLp::writeModelToFile). C++ opens the file and checks the
//! names; these write every byte that C++ fprintf's, and log the same
//! messages.

use super::write::{g, g_sign, pad, COut, CWriteModel, Model, Out, LOG_DEV, LOG_INFO, LOG_WARNING};
use crate::lp_data::var_type::{INTEGER, SEMI_CONTINUOUS, SEMI_INTEGER};
use crate::sprintf;

const INF: f64 = f64::INFINITY;

fn bool_str(b: bool) -> &'static str {
    if b {
        "true"
    } else {
        "false"
    }
}

/// writeMps from the computation of the row types on
pub fn write_mps(out: &mut Out, m: &Model) {
    let (num_row, num_col) = (m.num_row, m.num_col);
    let obj = m.objective_name;
    #[derive(Clone, Copy, PartialEq)]
    enum Ty {
        N,
        E,
        L,
        G,
    }
    let mut r_ty = vec![Ty::N; num_row];
    let mut rhs = vec![0.0; num_row];
    let mut ranges = vec![0.0; num_row];
    for r in 0..num_row {
        let (lo, up) = (m.row_lower[r], m.row_upper[r]);
        if lo == up {
            r_ty[r] = Ty::E;
            rhs[r] = lo;
        } else if !(up >= INF) {
            r_ty[r] = Ty::L;
            rhs[r] = up;
            if !(-lo >= INF) {
                ranges[r] = up - lo;
            }
        } else if !(-lo >= INF) {
            r_ty[r] = Ty::G;
            rhs[r] = lo;
        }
    }
    let have_rhs = rhs.iter().any(|&v| v != 0.0) || m.offset != 0.0;
    let have_ranges = ranges.iter().any(|&v| v != 0.0);
    let is_discrete = |t: u8| t == INTEGER || t == SEMI_CONTINUOUS || t == SEMI_INTEGER;
    let have_int =
        !m.integrality.is_empty() && m.integrality[..num_col].iter().any(|&t| is_discrete(t));
    let discrete = |c: usize| have_int && is_discrete(m.integrality[c]);
    let mut have_bounds = false;
    for c in 0..num_col {
        if m.col_lower[c] != 0.0 || !(m.col_upper[c] >= INF) || discrete(c) {
            have_bounds = true;
            break;
        }
    }
    out.msg(
        LOG_DEV,
        &sprintf!(
            "Model: RHS =     %s\n       RANGES =  %s\n       BOUNDS =  %s\n",
            bool_str(have_rhs),
            bool_str(have_ranges),
            bool_str(have_bounds)
        ),
    );

    // "    %-8s  %-8s  %.15g\n"
    let entry = |out: &mut Out, a: &[u8], b: &[u8], v: f64| {
        out.s(b"    ");
        pad(&mut out.buf, a, 8, true);
        out.s(b"  ");
        pad(&mut out.buf, b, 8, true);
        out.s(b"  ");
        g(&mut out.buf, v, 15);
        out.s(b"\n").line();
    };
    // "<head>%-8s  %.15g\n" or, without a value, "<head>%-8s\n"
    let line = |out: &mut Out, head: &[u8], name: &[u8], v: Option<f64>| {
        out.s(head);
        pad(&mut out.buf, name, 8, true);
        if let Some(v) = v {
            out.s(b"  ");
            g(&mut out.buf, v, 15);
        }
        out.s(b"\n").line();
    };

    out.s(b"NAME        ").s(m.model_name).s(b"\n");
    if m.sense == -1 {
        out.s(b"OBJSENSE\n  MAX\n");
    }
    out.s(b"ROWS\n");
    line(out, b" N  ", obj, None);
    for r in 0..num_row {
        let head: &[u8] = match r_ty[r] {
            Ty::E => b" E  ",
            Ty::G => b" G  ",
            Ty::L => b" L  ",
            Ty::N => b" N  ",
        };
        line(out, head, m.row_names[r], None);
    }
    let mut num_no_cost_zero_columns = 0;
    let mut num_no_cost_zero_columns_in_bounds_section = 0;
    let mut integer_fg = false;
    let mut n_integer_mk = 0;
    let marker = |out: &mut Out, n: &mut i32, kind: &str| {
        out.s(sprintf!("    MARK%04d  'MARKER'                 '%s'\n", *n, kind).as_bytes());
        *n += 1;
    };
    out.s(b"COLUMNS\n");
    for c in 0..num_col {
        let (start, end) = (m.a_start[c] as usize, m.a_start[c + 1] as usize);
        let name = m.col_names[c];
        if m.col_cost[c] == 0.0 && start == end {
            // Give the column a presence by writing out a zero cost
            num_no_cost_zero_columns += 1;
            entry(out, name, obj, 0.0);
            continue;
        }
        if have_int {
            if m.integrality[c] == INTEGER && !integer_fg {
                marker(out, &mut n_integer_mk, "INTORG");
                integer_fg = true;
            } else if m.integrality[c] != INTEGER && integer_fg {
                marker(out, &mut n_integer_mk, "INTEND");
                integer_fg = false;
            }
        }
        if m.col_cost[c] != 0.0 {
            entry(out, name, obj, m.col_cost[c]);
        }
        for el in start..end {
            entry(
                out,
                name,
                m.row_names[m.a_index[el] as usize],
                m.a_value[el],
            );
        }
    }
    if integer_fg {
        marker(out, &mut n_integer_mk, "INTEND");
    }
    // The RHS section is always written
    out.s(b"RHS\n");
    if m.offset != 0.0 {
        line(out, b"    RHS_V     ", obj, Some(-m.offset));
    }
    for r in 0..num_row {
        if rhs[r] != 0.0 {
            line(out, b"    RHS_V     ", m.row_names[r], Some(rhs[r]));
        }
    }
    if have_ranges {
        out.s(b"RANGES\n");
        for r in 0..num_row {
            if ranges[r] != 0.0 {
                line(out, b"    RANGE     ", m.row_names[r], Some(ranges[r]));
            }
        }
    }
    if have_bounds {
        out.s(b"BOUNDS\n");
        for c in 0..num_col {
            let (lb, ub) = (m.col_lower[c], m.col_upper[c]);
            let name = m.col_names[c];
            let name_str = String::from_utf8_lossy(name);
            if m.col_cost[c] == 0.0
                && m.a_start[c] == m.a_start[c + 1]
                && (!(ub >= INF) || lb != 0.0)
            {
                num_no_cost_zero_columns_in_bounds_section += 1;
            }
            if lb == ub {
                line(out, b" FX BOUND     ", name, Some(lb));
            } else if -lb >= INF && ub >= INF {
                line(out, b" FR BOUND     ", name, None);
            } else if discrete(c) {
                let t = m.integrality[c];
                if t == INTEGER || t == SEMI_INTEGER {
                    // Warn about non-integer bounds (static_cast<HighsInt>)
                    if lb > -INF && lb - (lb as i32) as f64 != 0.0 {
                        out.msg(
                            LOG_WARNING,
                            &sprintf!(
                                "Lower bound for integer or semi-integer column \"%s\" is %g: not integer\n",
                                &*name_str,
                                lb
                            ),
                        );
                    }
                    if ub < INF && ub - (ub as i32) as f64 != 0.0 {
                        out.msg(
                            LOG_WARNING,
                            &sprintf!(
                                "Upper bound for integer or semi-integer column \"%s\" is %g: not integer\n",
                                &*name_str,
                                ub
                            ),
                        );
                    }
                }
                if t == INTEGER {
                    if lb == 0.0 && ub == 1.0 {
                        line(out, b" BV BOUND     ", name, None);
                    } else {
                        if !(-lb >= INF) {
                            // No need to state a zero lower bound unless
                            // the upper bound is infinite
                            if lb != 0.0 || ub >= INF {
                                line(out, b" LI BOUND     ", name, Some(lb));
                            }
                        } else {
                            line(out, b" MI BOUND     ", name, None);
                        }
                        if !(ub >= INF) {
                            line(out, b" UI BOUND     ", name, Some(ub));
                        }
                    }
                } else {
                    // Semi-variables: infinite bounds are written as 1e30
                    let infinite_bound = 1e30;
                    let inf_lb = lb <= -INF;
                    let use_lb = if inf_lb { -infinite_bound } else { lb };
                    let inf_ub = ub >= INF;
                    let use_ub = if inf_ub { infinite_bound } else { ub };
                    if inf_lb {
                        out.msg(
                            LOG_WARNING,
                            &sprintf!(
                                "Lower bound for semi-variable \"%s\" is %g but writing %g\n",
                                &*name_str,
                                lb,
                                use_lb
                            ),
                        );
                    }
                    if inf_ub {
                        out.msg(
                            LOG_WARNING,
                            &sprintf!(
                                "Upper bound for semi-variable \"%s\" is %g but writing %g\n",
                                &*name_str,
                                ub,
                                use_ub
                            ),
                        );
                    }
                    line(out, b" LO BOUND     ", name, Some(use_lb));
                    let head: &[u8] = if t == SEMI_INTEGER {
                        b" SI BOUND     "
                    } else {
                        b" SC BOUND     "
                    };
                    line(out, head, name, Some(use_ub));
                }
            } else {
                if !(-lb >= INF) {
                    if lb != 0.0 {
                        line(out, b" LO BOUND     ", name, Some(lb));
                    }
                } else {
                    line(out, b" MI BOUND     ", name, None);
                }
                if !(ub >= INF) {
                    line(out, b" UP BOUND     ", name, Some(ub));
                }
            }
        }
    }
    if m.q_dim > 0 {
        // The lower triangle, column-wise
        out.s(b"QUADOBJ\n");
        for col in 0..m.q_dim {
            for el in m.q_start[col] as usize..m.q_start[col + 1] as usize {
                let row = m.q_index[el] as usize;
                if m.q_value[el] != 0.0 {
                    entry(out, m.col_names[col], m.col_names[row], m.q_value[el]);
                }
            }
        }
    }
    out.s(b"ENDATA\n");
    out.flush();
    if num_no_cost_zero_columns > 0 {
        out.msg(
            LOG_INFO,
            &sprintf!(
                "Model has %d zero columns with no costs: %d have finite upper bounds or nonzero lower bounds and are %swritten in MPS file\n",
                num_no_cost_zero_columns,
                num_no_cost_zero_columns_in_bounds_section,
                ""
            ),
        );
    }
}

const LP_MAX_LINE_LENGTH: usize = 560;
const LP_COMMENT_FILESTART: &[u8] = b"File written by HiGHS .lp file handler";

/// FilereaderLp's writeToFile*: tokens, with a line break before one
/// that would take the line to LP_MAX_LINE_LENGTH
struct LpOut<'o, 'a> {
    out: &'o mut Out<'a>,
    line_length: usize,
    token: Vec<u8>,
}

impl LpOut<'_, '_> {
    /// writeToFile of the text in `self.token`, which vsnprintf
    /// truncates to LP_MAX_LINE_LENGTH bytes
    fn put(&mut self) {
        let len = self.token.len();
        let text = &self.token[..len.min(LP_MAX_LINE_LENGTH)];
        if self.line_length + len >= LP_MAX_LINE_LENGTH {
            self.out.s(b"\n");
            self.line_length = len;
        } else {
            self.line_length += len;
        }
        self.out.s(text);
        self.token.clear();
    }
    fn text(&mut self, s: &[u8]) {
        self.token.extend_from_slice(s);
        self.put();
    }
    /// writeToFileVar: " %s"
    fn var(&mut self, name: &[u8]) {
        self.token.push(b' ');
        self.text(name);
    }
    /// writeToFileValue: " %.15g" or " %+.15g"
    fn value(&mut self, v: f64, force_plus: bool) {
        self.token.push(b' ');
        g_sign(&mut self.token, v, 15, force_plus);
        self.put();
    }
    /// writeToFileLineEnd
    fn end(&mut self) {
        self.out.s(b"\n").line();
        self.line_length = 0;
    }
}

/// FilereaderLp::writeModelToFile after the file is opened; the matrix
/// of `m` is row-wise
pub fn write_lp(out: &mut Out, m: &Model) {
    let mut w = LpOut {
        out,
        line_length: 0,
        token: Vec::with_capacity(64),
    };
    let matrix_row = |w: &mut LpOut, r: usize| {
        for el in m.a_start[r] as usize..m.a_start[r + 1] as usize {
            w.value(m.a_value[el], true);
            w.var(m.col_names[m.a_index[el] as usize]);
        }
    };
    w.token.extend_from_slice(b"\\ ");
    w.text(LP_COMMENT_FILESTART);
    w.end();
    w.text(if m.sense == 1 { b"min" } else { b"max" });
    w.end();
    w.text(b" obj:");
    for c in 0..m.num_col {
        if m.col_cost[c] != 0.0 {
            w.value(m.col_cost[c], true);
            w.var(m.col_names[c]);
        }
    }
    w.text(b" ");
    if m.q_dim != 0 {
        w.text(b"+ [");
        for c in 0..m.num_col {
            for el in m.q_start[c] as usize..m.q_start[c + 1] as usize {
                let r = m.q_index[el] as usize;
                if c <= r {
                    let mut coef = m.q_value[el];
                    if c != r {
                        coef *= 2.0;
                    }
                    if coef != 0.0 {
                        w.value(coef, true);
                        w.var(m.col_names[c]);
                        w.text(b" *");
                        w.var(m.col_names[r]);
                    }
                }
            }
        }
        w.text(b"  ]/2 ");
    }
    if m.offset != 0.0 {
        w.value(m.offset, true);
    }
    w.end();
    w.text(b"st");
    w.end();
    for r in 0..m.num_row {
        let (lo, up) = (m.row_lower[r], m.row_upper[r]);
        let name = m.row_names[r];
        if lo == up {
            w.var(name);
            w.text(b":");
            matrix_row(&mut w, r);
            w.text(b" =");
            w.value(lo, true);
            w.end();
        } else {
            let boxed = lo > -INF && up < INF;
            if lo > -INF {
                w.var(name);
                w.text(if boxed { b"lo:" } else { b":" });
                matrix_row(&mut w, r);
                w.text(b" >=");
                w.value(lo, true);
                w.end();
            }
            if up < INF {
                w.var(name);
                w.text(if boxed { b"up:" } else { b":" });
                matrix_row(&mut w, r);
                w.text(b" <=");
                w.value(up, true);
                w.end();
            }
        }
    }
    w.text(b"bounds");
    w.end();
    for c in 0..m.num_col {
        let (lo, up) = (m.col_lower[c], m.col_upper[c]);
        if lo == 0.0 && up == INF {
            continue;
        }
        if lo <= -INF && up >= INF {
            w.var(m.col_names[c]);
            w.text(b" free");
        } else if lo == up {
            w.var(m.col_names[c]);
            w.text(b" =");
            w.value(up, false);
        } else {
            if lo != 0.0 {
                w.value(lo, false);
                w.text(b" <=");
            }
            w.var(m.col_names[c]);
            if up < INF {
                w.text(b" <=");
                w.value(up, false);
            }
        }
        w.end();
    }
    if !m.integrality.is_empty() {
        let t = m.integrality;
        w.text(b"bin");
        w.end();
        for c in 0..m.num_col {
            if t[c] == INTEGER && m.col_lower[c] == 0.0 && m.col_upper[c] == 1.0 {
                w.var(m.col_names[c]);
                w.end();
            }
        }
        w.text(b"gen");
        w.end();
        for c in 0..m.num_col {
            if t[c] == INTEGER && (m.col_lower[c] != 0.0 || m.col_upper[c] != 1.0) {
                w.var(m.col_names[c]);
                w.end();
            }
        }
        w.text(b"semi");
        w.end();
        for c in 0..m.num_col {
            if t[c] == SEMI_CONTINUOUS || t[c] == SEMI_INTEGER {
                w.var(m.col_names[c]);
                w.end();
            }
        }
    }
    w.text(b"end");
    w.end();
}

/// writeMps after the file is opened and the names checked
///
/// # Safety
/// The pointers must be valid, as C++ passes them
#[no_mangle]
pub unsafe extern "C" fn highs_rs_write_mps(out: *const COut, model: *const CWriteModel) {
    let mut o = Out::new(Some(&*out));
    write_mps(&mut o, &(*model).view());
}

/// FilereaderLp::writeModelToFile after the file is opened
///
/// # Safety
/// The pointers must be valid, as C++ passes them
#[no_mangle]
pub unsafe extern "C" fn highs_rs_write_lp(out: *const COut, model: *const CWriteModel) {
    let mut o = Out::new(Some(&*out));
    write_lp(&mut o, &(*model).view());
}
