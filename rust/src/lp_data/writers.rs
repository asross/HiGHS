//! Solution and basis file writers (HighsModelUtils.cpp: writeSolutionFile
//! in every style, writePrimalSolution, writeObjectiveValue;
//! HighsLpUtils.cpp: writeBasisFile). Byte-identical to the C++: the
//! text C++ prints with one highsFprintfString is one chunk of `Out`
//! (stdout logs them one by one), and a piece C++ formats with
//! highsFormatToString is cut to its 1023-byte buffer (`cap`).

use super::solution::{CBasis, CSolution, Info, PrimalDualErrors};
use crate::io::write::{
    double_to_string, g, g_w, int, pad, COut, CWriteModel, Model, Out, RsStr, LOG_WARNING,
};
use crate::lp_data::ffi::RsMut;
use crate::lp_data::var_type::CONTINUOUS;
use crate::sprintf;

const INF: f64 = f64::INFINITY;
/// kHighsSolutionValueToStringTolerance, kGlpsolSolutionValueToStringTolerance
const VALUE_TOL: f64 = 1e-13;
const GLPSOL_VALUE_TOL: f64 = 1e-12;

const SOLUTION_STATUS_NONE: i32 = 0;
const SOLUTION_STATUS_INFEASIBLE: i32 = 1;
const SOLUTION_STATUS_FEASIBLE: i32 = 2;

const BASIS_LOWER: u8 = 0;
const BASIS_BASIC: u8 = 1;
const BASIS_UPPER: u8 = 2;
const BASIS_ZERO: u8 = 3;
const BASIS_NONBASIC: u8 = 4;

/// Solution styles (kSolutionStyle*), and the parts of the raw style
/// that writeModelSolution writes
pub const STYLE_OLD_RAW: i32 = -1;
pub const STYLE_RAW: i32 = 0;
pub const STYLE_PRETTY: i32 = 1;
pub const STYLE_GLPSOL_RAW: i32 = 2;
pub const STYLE_GLPSOL_PRETTY: i32 = 3;
pub const STYLE_SPARSE: i32 = 4;
pub const PART_MODEL_SOLUTION: i32 = 5;
pub const PART_MODEL_SOLUTION_SPARSE: i32 = 6;

/// kGlpsolCostRowLocation*
const COST_ROW_LAST: i32 = -2;
const COST_ROW_NONE: i32 = -1;
const COST_ROW_NONE_IF_EMPTY: i32 = 0;

/// Cut what was written since `start` to highsFormatToString's buffer
fn cap(buf: &mut Vec<u8>, start: usize) {
    if buf.len() - start > 1023 {
        buf.truncate(start + 1023);
    }
}

pub struct Solution<'a> {
    pub value_valid: bool,
    pub dual_valid: bool,
    pub col_value: &'a [f64],
    pub col_dual: &'a [f64],
    pub row_value: &'a [f64],
    pub row_dual: &'a [f64],
}

pub struct Basis<'a> {
    pub valid: bool,
    pub col_status: &'a [u8],
    pub row_status: &'a [u8],
}

/// statusToString
fn status_str(status: u8, lower: f64, upper: f64) -> &'static [u8] {
    match status {
        BASIS_LOWER | BASIS_UPPER if lower == upper => b"FX",
        BASIS_LOWER => b"LB",
        BASIS_BASIC => b"BS",
        BASIS_UPPER => b"UB",
        BASIS_ZERO => b"FR",
        BASIS_NONBASIC => b"NB",
        _ => b"",
    }
}

/// typeToString
fn type_str(t: u8) -> &'static [u8] {
    match t {
        0 => b"Continuous",
        1 => b"Integer   ",
        2 => b"Semi-conts",
        3 => b"Semi-int  ",
        4 => b"ImpliedInt",
        _ => b"",
    }
}

/// writeModelBoundSolution
#[allow(clippy::too_many_arguments)]
pub fn write_model_bound_solution(
    out: &mut Out,
    columns: bool,
    lower: &[f64],
    upper: &[f64],
    names: &[&[u8]],
    primal: Option<&[f64]>,
    dual: Option<&[f64]>,
    status: Option<&[u8]>,
    integrality: Option<&[u8]>,
) {
    out.s(if columns { b"Columns\n" } else { b"Rows\n" })
        .chunk();
    out.s(b"    Index Status        Lower        Upper       Primal         Dual");
    if integrality.is_some() {
        out.s(b"  Type      ");
    }
    out.s(b"  Name\n").chunk();
    for ix in 0..names.len() {
        let b = &mut out.buf;
        let st = status.map_or(&b""[..], |s| status_str(s[ix], lower[ix], upper[ix]));
        pad(b, ix.to_string().as_bytes(), 9, false);
        b.extend_from_slice(b"   ");
        pad(b, st, 4, false);
        b.push(b' ');
        g_w(b, lower[ix], 6, 12);
        b.push(b' ');
        g_w(b, upper[ix], 6, 12);
        for v in [primal, dual] {
            match v {
                Some(v) => {
                    b.push(b' ');
                    g_w(b, v[ix], 6, 12);
                }
                None => b.extend_from_slice(b"             "),
            }
        }
        if let Some(t) = integrality {
            b.extend_from_slice(b"  ");
            b.extend_from_slice(type_str(t[ix]));
        }
        let start = b.len();
        b.extend_from_slice(b"  ");
        b.extend_from_slice(names[ix]);
        b.push(b'\n');
        cap(b, start);
        out.chunk();
    }
}

/// writeObjectiveValue
pub fn write_objective_value(out: &mut Out, v: f64) {
    out.s(b"Objective ");
    double_to_string(&mut out.buf, v, VALUE_TOL);
    out.s(b"\n").chunk();
}

/// writePrimalSolution; without names, every name is "NoName"
pub fn write_primal_solution(
    out: &mut Out,
    names: &[&[u8]],
    primal: &[f64],
    num_col: usize,
    sparse: bool,
) {
    let num_col_field = if sparse {
        -(primal[..num_col].iter().filter(|&&v| v != 0.0).count() as i64)
    } else {
        num_col as i64
    };
    out.s(b"# Columns ").d(num_col_field).s(b"\n").chunk();
    for (ix, &v) in primal[..num_col].iter().enumerate() {
        if sparse && v == 0.0 {
            continue;
        }
        let b = &mut out.buf;
        let start = b.len();
        b.extend_from_slice(if names.is_empty() {
            b"NoName"
        } else {
            names[ix]
        });
        b.push(b' ');
        double_to_string(b, v, VALUE_TOL);
        cap(b, start);
        if sparse {
            b.push(b' ');
            int(b, ix as i64);
        }
        b.push(b'\n');
        out.chunk();
    }
}

/// "%-s %s\n" of each name and value
fn write_values(out: &mut Out, names: &[&[u8]], values: &[f64]) {
    for (name, &v) in names.iter().zip(values) {
        let b = &mut out.buf;
        let start = b.len();
        b.extend_from_slice(name);
        b.push(b' ');
        double_to_string(b, v, VALUE_TOL);
        b.push(b'\n');
        cap(b, start);
        out.chunk();
    }
}

fn feasibility(status: i32) -> &'static [u8] {
    if status == SOLUTION_STATUS_FEASIBLE {
        b"Feasible\n"
    } else {
        b"Infeasible\n"
    }
}

/// writeModelSolution; `objective` is the model's objective value at the
/// primal solution, which C++ computes (HighsCDouble sums)
pub fn write_model_solution(
    out: &mut Out,
    m: &Model,
    sol: &Solution,
    info: &Info,
    objective: f64,
    sparse: bool,
) {
    let (num_col, num_row) = (m.num_col, m.num_row);
    out.s(b"\n# Primal solution values\n").chunk();
    if !sol.value_valid || info.primal_solution_status == SOLUTION_STATUS_NONE {
        out.s(b"None\n").chunk();
    } else {
        out.s(feasibility(info.primal_solution_status)).chunk();
        write_objective_value(out, objective);
        write_primal_solution(out, &m.col_names, sol.col_value, num_col, sparse);
        if sparse {
            return;
        }
        out.s(b"# Rows ").d(num_row as i64).s(b"\n").chunk();
        write_values(out, &m.row_names, &sol.row_value[..num_row]);
    }
    out.s(b"\n# Dual solution values\n").chunk();
    if !sol.dual_valid || info.dual_solution_status == SOLUTION_STATUS_NONE {
        out.s(b"None\n").chunk();
    } else {
        out.s(feasibility(info.dual_solution_status)).chunk();
        out.s(b"# Columns ").d(num_col as i64).s(b"\n").chunk();
        write_values(out, &m.col_names, &sol.col_dual[..num_col]);
        out.s(b"# Rows ").d(num_row as i64).s(b"\n").chunk();
        write_values(out, &m.row_names, &sol.row_dual[..num_row]);
    }
}

/// writeOldRawSolution
pub fn write_old_raw_solution(out: &mut Out, m: &Model, sol: &Solution, basis: &Basis) {
    let (have_value, have_dual, have_basis) = (sol.value_valid, sol.dual_valid, basis.valid);
    if !have_value && !have_dual && !have_basis {
        return;
    }
    out.s(sprintf!(
        "%d %d : Number of columns and rows for primal or dual solution or basis\n",
        m.num_col,
        m.num_row
    )
    .as_bytes())
        .chunk();
    let tf = |b: bool| if b { b"T" } else { b"F" };
    out.s(tf(have_value)).s(b" Primal solution\n").chunk();
    out.s(tf(have_dual)).s(b" Dual solution\n").chunk();
    out.s(tf(have_basis)).s(b" Basis\n").chunk();
    for (head, n, value, dual, status) in [
        (
            &b"Columns\n"[..],
            m.num_col,
            sol.col_value,
            sol.col_dual,
            basis.col_status,
        ),
        (
            &b"Rows\n"[..],
            m.num_row,
            sol.row_value,
            sol.row_dual,
            basis.row_status,
        ),
    ] {
        out.s(head).chunk();
        for i in 0..n {
            let b = &mut out.buf;
            if have_value {
                g(b, value[i], 15);
                b.push(b' ');
            }
            if have_dual {
                g(b, dual[i], 15);
                b.push(b' ');
            }
            if have_basis {
                int(b, status[i] as i64);
            }
            b.push(b'\n');
            out.chunk();
        }
    }
}

/// The data of writeGlpsolSolution besides the model and solution
pub struct Glpsol<'a> {
    pub num_nz: usize,
    pub cost_row_option: i32,
    pub model_status: i32,
    pub info: &'a Info,
    pub raw: bool,
}

/// HighsModelStatus values writeGlpsolSolution distinguishes
const MODEL_STATUS_OPTIMAL: i32 = 7;
const MODEL_STATUS_INFEASIBLE: i32 = 8;
const MODEL_STATUS_UNBOUNDED: i32 = 10;

/// writeGlpsolCostRow
fn write_glpsol_cost_row(
    out: &mut Out,
    raw: bool,
    is_mip: bool,
    row_id: i64,
    objective_name: &[u8],
    objective: f64,
) {
    let b = &mut out.buf;
    if raw {
        b.extend_from_slice(b"i ");
        int(b, row_id);
        b.push(b' ');
        if !is_mip {
            b.extend_from_slice(b"b ");
        }
        double_to_string(b, objective, GLPSOL_VALUE_TOL);
        if !is_mip {
            b.extend_from_slice(b" 0");
        }
        b.push(b'\n');
    } else {
        pad(b, row_id.to_string().as_bytes(), 6, false);
        b.push(b' ');
        name_field(b, objective_name, None);
        b.extend_from_slice(if is_mip { b"   " } else { b"B  " });
        g_w(b, objective, 6, 13);
        // " %13s %13s \n" of empty strings
        b.extend_from_slice(&[b' '; 29]);
        b.push(b'\n');
    }
    out.chunk();
}

/// "%-12s " of a name of up to 12 characters, else "%s\n" and then 20
/// spaces, which start a new chunk when `new_chunk` is given
fn name_field(b: &mut Vec<u8>, name: &[u8], new_chunk: Option<&mut bool>) {
    if name.len() <= 12 {
        pad(b, name, 12, true);
        b.push(b' ');
    } else {
        let start = b.len();
        b.extend_from_slice(name);
        b.push(b'\n');
        match new_chunk {
            Some(c) => *c = true,
            None => b.extend_from_slice(&[b' '; 20]),
        }
        cap(b, start);
    }
}

/// writeGlpsolSolution up to the KKT report (`write_glpsol_kkt`, after
/// C++'s getKktFailures); returns whether that report follows
pub fn write_glpsol_solution(
    out: &mut Out,
    m: &Model,
    sol: &Solution,
    basis: &Basis,
    o: &Glpsol,
) -> bool {
    const PRINT_AS_ZERO: f64 = 1e-9;
    let info = o.info;
    let raw = o.raw;
    let (have_value, have_dual, have_basis) = (sol.value_valid, sol.dual_valid, basis.valid);
    let (num_row, num_col) = (m.num_row, m.num_col);
    let mut num_nz = o.num_nz + m.col_cost[..num_col].iter().filter(|&&c| c != 0.0).count();
    let empty_cost_row = num_nz == o.num_nz;
    let has_objective = !empty_cost_row || m.q_dim != 0;
    let mut cost_row_location: usize = 0;
    let artificial_cost_row = m.objective_name == b"R0000000";
    if artificial_cost_row {
        out.msg(
            LOG_WARNING,
            &sprintf!(
                "The cost row name of \"%s\" is assumed to be artificial and will not be reported in the Glpsol solution file\n",
                &*String::from_utf8_lossy(m.objective_name)
            ),
        );
    }
    let opt = o.cost_row_option;
    if opt <= COST_ROW_LAST || opt as i64 > num_row as i64 {
        cost_row_location = num_row + 1;
    } else if opt == COST_ROW_NONE {
    } else if opt == COST_ROW_NONE_IF_EMPTY {
        if !(empty_cost_row && artificial_cost_row) {
            if m.cost_row_location >= 0 {
                cost_row_location = m.cost_row_location as usize + 1;
            } else {
                cost_row_location = num_row + 1;
                out.msg(
                    LOG_WARNING,
                    "The cost row for the Glpsol solution file is reported last since there is no indication of where it should be\n",
                );
            }
        }
    } else {
        cost_row_location = opt as usize;
    }
    let glpsol_num_row = num_row + (cost_row_location > 0) as usize;
    if cost_row_location == 0 {
        num_nz = o.num_nz;
    }
    let mut num_integer = 0;
    let mut num_binary = 0;
    let mut is_mip = false;
    if m.integrality.len() == num_col {
        for c in 0..num_col {
            if m.integrality[c] != CONTINUOUS {
                is_mip = true;
                num_integer += 1;
                if m.col_lower[c] == 0.0 && m.col_upper[c] == 1.0 {
                    num_binary += 1;
                }
            }
        }
    }
    let prefix: &[u8] = if raw { b"c " } else { b"" };
    {
        let b = &mut out.buf;
        let start = b.len();
        b.extend_from_slice(prefix);
        pad(b, b"Problem:", 12, true);
        b.extend_from_slice(m.model_name);
        b.push(b'\n');
        cap(b, start);
    }
    out.chunk();
    out.s(prefix);
    pad(&mut out.buf, b"Rows:", 12, true);
    out.d(glpsol_num_row as i64).s(b"\n").chunk();
    out.s(prefix);
    pad(&mut out.buf, b"Columns:", 12, true);
    out.d(num_col as i64);
    if !raw && is_mip {
        out.s(sprintf!(" (%d integer, %d binary)", num_integer, num_binary).as_bytes());
    }
    out.s(b"\n").chunk();
    out.s(prefix);
    pad(&mut out.buf, b"Non-zeros:", 12, true);
    out.d(num_nz as i64).s(b"\n").chunk();
    let primal_feasible = info.primal_solution_status == SOLUTION_STATUS_FEASIBLE;
    let (model_status_text, solution_status_char): (&[u8], &[u8]) = match o.model_status {
        MODEL_STATUS_OPTIMAL if is_mip => (b"INTEGER OPTIMAL", b"o"),
        MODEL_STATUS_OPTIMAL => (b"OPTIMAL", b"?"),
        MODEL_STATUS_INFEASIBLE if is_mip => (b"INTEGER EMPTY", b"n"),
        MODEL_STATUS_INFEASIBLE => (b"INFEASIBLE (FINAL)", b"?"),
        MODEL_STATUS_UNBOUNDED => (b"UNBOUNDED", if is_mip { b"u" } else { b"?" }),
        _ if primal_feasible && is_mip => (b"INTEGER NON-OPTIMAL", b"f"),
        _ if primal_feasible => (b"FEASIBLE", b"?"),
        _ => (b"UNDEFINED", if is_mip { b"u" } else { b"?" }),
    };
    out.s(prefix);
    pad(&mut out.buf, b"Status:", 12, true);
    out.s(model_status_text).s(b"\n").chunk();
    if !info.valid {
        return false;
    }
    let objective_value = if has_objective {
        info.objective_function_value
    } else {
        0.0
    };
    {
        let b = &mut out.buf;
        let start = b.len();
        b.extend_from_slice(prefix);
        pad(b, b"Objective:", 12, true);
        if has_objective && !m.objective_name.is_empty() {
            b.extend_from_slice(m.objective_name);
            b.extend_from_slice(b" = ");
        }
        g(b, objective_value, 10);
        b.extend_from_slice(if m.sense == 1 {
            b" (MINimum)\n"
        } else {
            b" (MAXimum)\n"
        });
        cap(b, start);
    }
    out.chunk();
    // No space after "c" on the blank line
    out.s(if raw { b"c\n" } else { b"\n" }).chunk();
    if raw {
        out.s(if is_mip { b"s mip " } else { b"s bas " });
        out.d(glpsol_num_row as i64)
            .s(b" ")
            .d(num_col as i64)
            .s(b" ");
        if is_mip {
            out.s(solution_status_char);
        } else {
            let c = |s: i32| match s {
                SOLUTION_STATUS_NONE => b"u",
                SOLUTION_STATUS_INFEASIBLE => b"i",
                SOLUTION_STATUS_FEASIBLE => b"f",
                _ => b"?",
            };
            out.s(c(info.primal_solution_status))
                .s(b" ")
                .s(c(info.dual_solution_status));
        }
        out.s(b" ");
        double_to_string(&mut out.buf, objective_value, VALUE_TOL);
        out.s(b"\n").chunk();
    }
    if !have_value {
        return false;
    }
    let table_header = |out: &mut Out, head: &[u8]| {
        out.s(head).s(if have_basis { b"St" } else { b"  " });
        out.s(b"   Activity     Lower bound   Upper bound");
        if have_dual {
            out.s(b"    Marginal");
        }
        out.s(b"\n").chunk();
        out.s(b"------ ------------ ")
            .s(if have_basis { b"--" } else { b"  " });
        out.s(b" ------------- ------------- -------------");
        if have_dual {
            out.s(b" -------------");
        }
        out.s(b"\n").chunk();
    };
    // The rest of a row or column line, after its name
    let line_rest = |out: &mut Out,
                     status: Option<u8>,
                     integer: bool,
                     lower: f64,
                     upper: f64,
                     value: f64,
                     dual: f64| {
        let (status_text, status_char): (&[u8], &[u8]) = match status {
            Some(BASIS_BASIC) => (b"B ", b"b"),
            Some(BASIS_LOWER) | Some(BASIS_UPPER) if lower == upper => (b"NS", b"s"),
            Some(BASIS_LOWER) => (b"NL", b"l"),
            Some(BASIS_UPPER) => (b"NU", b"u"),
            Some(BASIS_ZERO) => (b"NF", b"f"),
            Some(_) => (b"??", b"?"),
            None if integer => (b"* ", b""),
            None => (b"  ", b""),
        };
        let b = &mut out.buf;
        if raw {
            b.extend_from_slice(status_char);
            b.push(b' ');
            double_to_string(b, value, VALUE_TOL);
            b.push(b' ');
        } else {
            b.extend_from_slice(status_text);
            b.push(b' ');
            g_w(
                b,
                if value.abs() <= PRINT_AS_ZERO {
                    0.0
                } else {
                    value
                },
                6,
                13,
            );
            b.push(b' ');
            if lower > -INF {
                g_w(b, lower, 6, 13);
            } else {
                b.extend_from_slice(&[b' '; 13]);
            }
            b.push(b' ');
            if lower != upper && upper < INF {
                g_w(b, upper, 6, 13);
            } else {
                pad(b, if lower == upper { b"=" } else { b"" }, 13, false);
            }
            b.push(b' ');
        }
        if have_dual {
            if raw {
                double_to_string(b, dual, VALUE_TOL);
            } else if have_basis && status != Some(BASIS_BASIC) {
                // Only duals of variables known to be nonbasic are shown
                if dual.abs() <= PRINT_AS_ZERO {
                    b.extend_from_slice(b"        < eps");
                } else {
                    g_w(b, dual, 6, 13);
                    b.push(b' ');
                }
            }
        }
        b.push(b'\n');
        out.chunk();
    };
    // The start of a line: its number, and the name if pretty. Returns
    // false for a MIP's raw line, which is complete
    let line_start =
        |out: &mut Out, id: usize, name: &[u8], value: f64, raw_prefix: &[u8]| -> bool {
            if raw {
                out.s(raw_prefix).d(id as i64).s(b" ");
                if is_mip {
                    double_to_string(&mut out.buf, value, VALUE_TOL);
                    out.s(b"\n").chunk();
                    return false;
                }
            } else {
                pad(&mut out.buf, id.to_string().as_bytes(), 6, false);
                out.s(b" ");
                let mut new_chunk = false;
                name_field(&mut out.buf, name, Some(&mut new_chunk));
                if new_chunk {
                    out.chunk();
                    out.s(&[b' '; 20]);
                }
            }
            true
        };
    if !raw {
        table_header(out, b"   No.   Row name   ");
    }
    let objective_name = m.objective_name;
    let mut row_id = 0;
    for r in 0..num_row {
        row_id += 1;
        if row_id == cost_row_location {
            write_glpsol_cost_row(
                out,
                raw,
                is_mip,
                row_id as i64,
                objective_name,
                info.objective_function_value,
            );
            row_id += 1;
        }
        let value = sol.row_value[r];
        if !line_start(out, row_id, m.row_names[r], value, b"i ") {
            continue;
        }
        let dual = if have_dual { sol.row_dual[r] } else { 0.0 };
        let status = if have_basis {
            Some(basis.row_status[r])
        } else {
            None
        };
        line_rest(
            out,
            status,
            false,
            m.row_lower[r],
            m.row_upper[r],
            value,
            dual,
        );
    }
    if cost_row_location == num_row + 1 {
        row_id += 1;
        write_glpsol_cost_row(
            out,
            raw,
            is_mip,
            row_id as i64,
            objective_name,
            info.objective_function_value,
        );
    }
    if !raw {
        out.s(b"\n").chunk();
        table_header(out, b"   No. Column name  ");
    }
    for c in 0..num_col {
        let value = sol.col_value[c];
        if !line_start(out, c + 1, m.col_names[c], value, b"j ") {
            continue;
        }
        let dual = if have_dual { sol.col_dual[c] } else { 0.0 };
        let status = if have_basis {
            Some(basis.col_status[c])
        } else {
            None
        };
        let integer = is_mip && m.integrality[c] != CONTINUOUS;
        line_rest(
            out,
            status,
            integer,
            m.col_lower[c],
            m.col_upper[c],
            value,
            dual,
        );
    }
    if raw {
        out.s(b"e o f\n").chunk();
        return false;
    }
    true
}

/// The KKT report that ends a pretty Glpsol solution file
pub fn write_glpsol_kkt(
    out: &mut Out,
    e: &PrimalDualErrors,
    num_col: i32,
    is_mip: bool,
    have_dual: bool,
) {
    const HIGH: f64 = 1e-9;
    const MEDIUM: f64 = 1e-6;
    const LOW: f64 = 1e-3;
    let quality = |out: &mut Out, rel: f64, bad: &str| {
        let q = if rel <= HIGH {
            "High quality"
        } else if rel <= MEDIUM {
            "Medium quality"
        } else if rel <= LOW {
            "Low quality"
        } else {
            bad
        };
        out.s(sprintf!("%8s%s\n", "", q).as_bytes()).s(b"\n");
    };
    out.s(b"\n").chunk();
    out.s(if is_mip {
        &b"Integer feasibility conditions:\n"[..]
    } else {
        b"Karush-Kuhn-Tucker optimality conditions:\n"
    })
    .chunk();
    out.s(b"\n").chunk();
    // (value, index from 1 or 0 for a zero value) of an error
    let idx = |v: f64, i: i32| if v == 0.0 { 0 } else { i + 1 };
    let col_or_row = |i: i32| {
        if i > 0 && i <= num_col {
            "column"
        } else {
            "row"
        }
    };
    let col_row_index = |i: i32| if i <= num_col { i } else { i - num_col };

    let r = &e.glpsol_max_primal_residual;
    let (av, ai, rv, ri) = (
        r.absolute_value,
        idx(r.absolute_value, r.absolute_index),
        r.relative_value,
        idx(r.relative_value, r.relative_index),
    );
    out.s(sprintf!("KKT.PE: max.abs.err = %.2e on row %d\n", av, ai).as_bytes())
        .chunk();
    out.s(sprintf!(
        "        max.rel.err = %.2e on row %d\n",
        rv,
        if ai == 0 { 0 } else { ri }
    )
    .as_bytes())
        .chunk();
    quality(out, rv, "PRIMAL SOLUTION IS WRONG");
    out.chunk();

    // Primal and dual infeasibility, on a column or row
    let infeasibility = |out: &mut Out, r: &super::solution::HError, tag: &str, bad: &str| {
        let (av, ai, rv, ri) = (
            r.absolute_value,
            idx(r.absolute_value, r.absolute_index),
            r.relative_value,
            idx(r.relative_value, r.relative_index),
        );
        out.s(sprintf!(
            "KKT.%s: max.abs.err = %.2e on %s %d\n",
            tag,
            av,
            col_or_row(ai),
            col_row_index(ai)
        )
        .as_bytes())
            .chunk();
        out.s(sprintf!(
            "        max.rel.err = %.2e on %s %d\n",
            rv,
            col_or_row(ri),
            col_row_index(ri)
        )
        .as_bytes())
            .chunk();
        quality(out, rv, bad);
        out.chunk();
    };
    infeasibility(
        out,
        &e.glpsol_max_primal_infeasibility,
        "PB",
        "PRIMAL SOLUTION IS INFEASIBLE",
    );
    if have_dual {
        let r = &e.glpsol_max_dual_residual;
        let (av, ai, rv, ri) = (
            r.absolute_value,
            idx(r.absolute_value, r.absolute_index),
            r.relative_value,
            idx(r.relative_value, r.relative_index),
        );
        out.s(sprintf!("KKT.DE: max.abs.err = %.2e on column %d\n", av, ai).as_bytes());
        out.s(sprintf!("        max.rel.err = %.2e on column %d\n", rv, ri).as_bytes());
        quality(out, rv, "DUAL SOLUTION IS WRONG");
        out.chunk();
        infeasibility(
            out,
            &e.glpsol_max_dual_infeasibility,
            "DB",
            "DUAL SOLUTION IS INFEASIBLE",
        );
    }
    out.s(b"End of output\n").chunk();
}

/// writeBasisFile
pub fn write_basis_file(out: &mut Out, m: &Model, basis: &Basis) {
    out.s(b"HiGHS_basis_file v2\n").chunk();
    if !basis.valid {
        out.s(b"None\n").chunk();
        return;
    }
    out.s(b"Valid\n").chunk();
    for (head, names, status) in [
        (&b"# Columns "[..], &m.col_names, basis.col_status),
        (&b"# Rows "[..], &m.row_names, basis.row_status),
    ] {
        out.s(head).d(names.len() as i64).s(b"\n").chunk();
        for (name, &s) in names.iter().zip(status) {
            let b = &mut out.buf;
            let start = b.len();
            b.extend_from_slice(name);
            b.push(b' ');
            int(b, s as i64);
            b.push(b'\n');
            cap(b, start);
            out.chunk();
        }
    }
}

/// writeSolutionFile (style from STYLE_OLD_RAW to STYLE_SPARSE) or one of
/// its parts (PART_*); returns whether the Glpsol KKT report follows
pub fn write_solution_file(
    out: &mut Out,
    m: &Model,
    sol: &Solution,
    basis: &Basis,
    s: &SolutionFile,
) -> bool {
    match s.style {
        STYLE_OLD_RAW => write_old_raw_solution(out, m, sol, basis),
        STYLE_PRETTY => {
            let integrality = if m.integrality.is_empty() {
                None
            } else {
                Some(m.integrality)
            };
            let primal = sol.value_valid.then_some((sol.col_value, sol.row_value));
            let dual = sol.dual_valid.then_some((sol.col_dual, sol.row_dual));
            let status = basis.valid.then_some((basis.col_status, basis.row_status));
            write_model_bound_solution(
                out,
                true,
                m.col_lower,
                m.col_upper,
                &m.col_names,
                primal.map(|p| p.0),
                dual.map(|d| d.0),
                status.map(|s| s.0),
                integrality,
            );
            write_model_bound_solution(
                out,
                false,
                m.row_lower,
                m.row_upper,
                &m.row_names,
                primal.map(|p| p.1),
                dual.map(|d| d.1),
                status.map(|s| s.1),
                None,
            );
            out.s(b"\n").chunk();
            out.s(b"Model status: ")
                .s(s.model_status_string)
                .s(b"\n")
                .chunk();
            out.s(b"\n").chunk();
            out.s(b"Objective value: ");
            double_to_string(&mut out.buf, s.info.objective_function_value, VALUE_TOL);
            out.s(b"\n").chunk();
        }
        STYLE_GLPSOL_RAW | STYLE_GLPSOL_PRETTY => {
            let o = Glpsol {
                num_nz: s.num_nz,
                cost_row_option: s.glpsol_cost_row_location,
                model_status: s.model_status,
                info: s.info,
                raw: s.style == STYLE_GLPSOL_RAW,
            };
            return write_glpsol_solution(out, m, sol, basis, &o);
        }
        PART_MODEL_SOLUTION | PART_MODEL_SOLUTION_SPARSE => write_model_solution(
            out,
            m,
            sol,
            s.info,
            s.objective,
            s.style == PART_MODEL_SOLUTION_SPARSE,
        ),
        _ => {
            out.s(b"Model status\n").chunk();
            out.s(s.model_status_string).s(b"\n").chunk();
            write_model_solution(out, m, sol, s.info, s.objective, s.style == STYLE_SPARSE);
        }
    }
    false
}

/// What writeSolutionFile reads besides the model, solution and basis
pub struct SolutionFile<'a> {
    pub style: i32,
    pub info: &'a Info,
    pub model_status: i32,
    pub model_status_string: &'a [u8],
    /// The model's objective value at the primal solution
    pub objective: f64,
    /// The number of nonzeros of the constraint matrix
    pub num_nz: usize,
    pub glpsol_cost_row_location: i32,
}

/// What C++ passes for writeSolutionFile (HighsWritersRust.cpp)
#[repr(C)]
pub struct CSolutionFile {
    pub style: i32,
    pub info: *const Info,
    pub model_status: i32,
    pub model_status_string: RsStr,
    pub objective: f64,
    pub num_nz: i32,
    pub glpsol_cost_row_location: i32,
}

unsafe fn solution<'a>(s: &CSolution) -> Solution<'a> {
    Solution {
        value_valid: s.value_valid,
        dual_valid: s.dual_valid,
        col_value: s.col_value.get(),
        col_dual: s.col_dual.get(),
        row_value: s.row_value.get(),
        row_dual: s.row_dual.get(),
    }
}

unsafe fn basis<'a>(b: &CBasis) -> Basis<'a> {
    Basis {
        valid: b.valid,
        col_status: b.col_status.get(),
        row_status: b.row_status.get(),
    }
}

unsafe fn names<'a>(n: &RsMut<RsStr>) -> Vec<&'a [u8]> {
    n.get().iter().map(|s| s.get()).collect()
}

/// writeSolutionFile; returns whether C++ is to call getKktFailures and
/// then `highs_rs_write_glpsol_kkt`
///
/// # Safety
/// The pointers must be valid, as C++ passes them
#[no_mangle]
pub unsafe extern "C" fn highs_rs_write_solution_file(
    out: *const COut,
    model: *const CWriteModel,
    sol: *const CSolution,
    b: *const CBasis,
    s: *const CSolutionFile,
) -> bool {
    let s = &*s;
    let file = SolutionFile {
        style: s.style,
        info: &*s.info,
        model_status: s.model_status,
        model_status_string: s.model_status_string.get(),
        objective: s.objective,
        num_nz: s.num_nz as usize,
        glpsol_cost_row_location: s.glpsol_cost_row_location,
    };
    let mut o = Out::new(Some(&*out));
    write_solution_file(
        &mut o,
        &(*model).view(),
        &solution(&*sol),
        &basis(&*b),
        &file,
    )
}

/// The KKT report of writeGlpsolSolution
///
/// # Safety
/// The pointers must be valid, as C++ passes them
#[no_mangle]
pub unsafe extern "C" fn highs_rs_write_glpsol_kkt(
    out: *const COut,
    e: *const PrimalDualErrors,
    num_col: i32,
    is_mip: bool,
    have_dual: bool,
) {
    let mut o = Out::new(Some(&*out));
    write_glpsol_kkt(&mut o, &*e, num_col, is_mip, have_dual);
}

/// writeModelBoundSolution; absent arrays are empty
///
/// # Safety
/// The pointers must be valid, as C++ passes them
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn highs_rs_write_model_bound_solution(
    out: *const COut,
    columns: bool,
    lower: RsMut<f64>,
    upper: RsMut<f64>,
    names_: RsMut<RsStr>,
    primal: *const RsMut<f64>,
    dual: *const RsMut<f64>,
    status: *const RsMut<u8>,
    integrality: *const RsMut<u8>,
) {
    let mut o = Out::new(Some(&*out));
    write_model_bound_solution(
        &mut o,
        columns,
        lower.get(),
        upper.get(),
        &names(&names_),
        primal.as_ref().map(|p| p.get()),
        dual.as_ref().map(|p| p.get()),
        status.as_ref().map(|p| p.get()),
        integrality.as_ref().map(|p| p.get()),
    );
}

/// writePrimalSolution
///
/// # Safety
/// The pointers must be valid, as C++ passes them
#[no_mangle]
pub unsafe extern "C" fn highs_rs_write_primal_solution(
    out: *const COut,
    num_col: i32,
    col_names: RsMut<RsStr>,
    primal: RsMut<f64>,
    sparse: bool,
) {
    let mut o = Out::new(Some(&*out));
    write_primal_solution(
        &mut o,
        &names(&col_names),
        primal.get(),
        num_col as usize,
        sparse,
    );
}

/// writeObjectiveValue
///
/// # Safety
/// `out` must be valid
#[no_mangle]
pub unsafe extern "C" fn highs_rs_write_objective_value(out: *const COut, v: f64) {
    let mut o = Out::new(Some(&*out));
    write_objective_value(&mut o, v);
}

/// writeBasisFile
///
/// # Safety
/// The pointers must be valid, as C++ passes them
#[no_mangle]
pub unsafe extern "C" fn highs_rs_write_basis_file(
    out: *const COut,
    model: *const CWriteModel,
    b: *const CBasis,
) {
    let mut o = Out::new(Some(&*out));
    write_basis_file(&mut o, &(*model).view(), &basis(&*b));
}

/// kRangingValueToStringTolerance
const RANGING_TOL: f64 = 1e-13;

/// HighsRanging's records for writeRangingFile: (value, objective) of
/// col_cost_up, col_cost_dn, col_bound_up, col_bound_dn, row_bound_up and
/// row_bound_dn
pub struct Ranging<'a> {
    pub valid: bool,
    pub rec: [(&'a [f64], &'a [f64]); 6],
}

/// `%-10.4g `
fn g10(b: &mut Vec<u8>, v: f64) {
    let mut t = Vec::new();
    g(&mut t, v, 4);
    pad(b, &t, 10, true);
    b.push(b' ');
}

/// `%6d   %4s  `
fn ranging_head(b: &mut Vec<u8>, i: usize, status: &[u8]) {
    let mut t = Vec::new();
    int(&mut t, i as i64);
    pad(b, &t, 6, false);
    b.extend_from_slice(b"   ");
    pad(b, status, 4, false);
    b.extend_from_slice(b"  ");
}

/// `%-s %s %s %s %s\n` of highsDoubleToString's
fn ranging_raw(b: &mut Vec<u8>, name: &[u8], v: [f64; 4]) {
    b.extend_from_slice(name);
    for x in v {
        b.push(b' ');
        double_to_string(b, x, RANGING_TOL);
    }
    b.push(b'\n');
}

/// writeRangingFile (fprintf's to the file, not highsFprintfString)
pub fn write_ranging_file(
    out: &mut Out,
    m: &Model,
    objective: f64,
    basis: &Basis,
    sol: &Solution,
    r: &Ranging,
    pretty: bool,
) {
    if !r.valid {
        out.s(b"None\n").line();
        return;
    }
    out.s(b"Valid\n").line();
    out.s(b"Objective ");
    double_to_string(&mut out.buf, objective, RANGING_TOL);
    out.s(b"\n").line();
    let [cost_up, cost_dn, col_up, col_dn, row_up, row_dn] = r.rec;
    if pretty {
        out.s(b"\n                                            Cost ranging\nColumn Status  DownObj    Down                  Value                 Up         UpObj      Name\n");
    } else {
        out.s(b"\n# Cost ranging\n");
    }
    out.line();
    for i in 0..m.num_col {
        let b = &mut out.buf;
        if pretty {
            ranging_head(b, i, status_str(basis.col_status[i], m.col_lower[i], m.col_upper[i]));
            g10(b, cost_dn.1[i]);
            g10(b, cost_dn.0[i]);
            b.extend_from_slice(b"           ");
            g10(b, m.col_cost[i]);
            b.extend_from_slice(b"           ");
            g10(b, cost_up.0[i]);
            g10(b, cost_up.1[i]);
            b.extend_from_slice(m.col_names[i]);
            b.push(b'\n');
        } else {
            ranging_raw(b, m.col_names[i], [cost_dn.1[i], cost_dn.0[i], cost_up.0[i], cost_up.1[i]]);
        }
        out.line();
    }
    if pretty {
        out.s(b"\n                                            Bound ranging\nColumn Status  DownObj    Down       Lower      Value      Upper      Up         UpObj      Name\n");
    } else {
        out.s(b"\n# Bound ranging\n# Columns\n");
    }
    out.line();
    let parts = [
        (m.num_col, &m.col_names, m.col_lower, m.col_upper, sol.col_value, basis.col_status, col_dn, col_up),
        (m.num_row, &m.row_names, m.row_lower, m.row_upper, sol.row_value, basis.row_status, row_dn, row_up),
    ];
    for (k, (n, names, lower, upper, value, status, dn, up)) in parts.into_iter().enumerate() {
        if k == 1 {
            if pretty {
                out.s(b"                                            Bound ranging\n   Row Status  DownObj    Down       Lower      Value      Upper      Up         UpObj      Name\n");
            } else {
                out.s(b"# Rows\n");
            }
            out.line();
        }
        for i in 0..n {
            let b = &mut out.buf;
            if pretty {
                ranging_head(b, i, status_str(status[i], lower[i], upper[i]));
                for x in [dn.1[i], dn.0[i], lower[i], value[i], upper[i], up.0[i], up.1[i]] {
                    g10(b, x);
                }
                b.extend_from_slice(names[i]);
                b.push(b'\n');
            } else {
                ranging_raw(b, names[i], [dn.1[i], dn.0[i], up.0[i], up.1[i]]);
            }
            out.line();
        }
    }
}

/// What C++ passes for writeRangingFile: the records' value and
/// objective arrays in `Ranging`'s order
#[repr(C)]
pub struct CRangingFile {
    pub valid: bool,
    pub pretty: bool,
    pub objective: f64,
    pub rec: [RsMut<f64>; 12],
}

/// writeRangingFile
///
/// # Safety
/// The pointers must be valid, as C++ passes them
#[no_mangle]
pub unsafe extern "C" fn highs_rs_write_ranging_file(
    out: *const COut,
    model: *const CWriteModel,
    sol: *const CSolution,
    b: *const CBasis,
    r: *const CRangingFile,
) {
    let r = &*r;
    let rec = |k: usize| (r.rec[2 * k].get(), r.rec[2 * k + 1].get());
    let ranging = Ranging {
        valid: r.valid,
        rec: [rec(0), rec(1), rec(2), rec(3), rec(4), rec(5)],
    };
    let mut o = Out::new(Some(&*out));
    write_ranging_file(&mut o, &(*model).view(), r.objective, &basis(&*b), &solution(&*sol), &ranging, r.pretty);
}
