//! HighsPathSeparator::separateLpSolution: aggregations of rows along paths
//! of continuous columns, and path mixing cuts.

use super::cut_generation::CutGeneration;
use super::round::{MinCpp, SepaRound};
use crate::util::cdouble::CDouble;
use crate::util::random::HighsRandom;

const K_UNUSABLE: i8 = -2;
const K_GEQ: i8 = -1;
const K_EQ: i8 = 0;
const K_LEQ: i8 = 1;
const MAX_PATH_LEN: usize = 6;
const K_HIGHS_TINY: f64 = 1e-14;

/// The separator's state across rounds: its random generator and work
/// space
pub struct PathSeparator {
    pub randgen: HighsRandom,
    cutgen: CutGeneration,
    rowtype: Vec<i8>,
    num_continuous: Vec<i32>,
    col_substitutions: Vec<(i32, f64)>,
    in_arc_rows: Vec<(i32, f64)>,
    col_in_arcs: Vec<(i32, i32)>,
    out_arc_rows: Vec<(i32, f64)>,
    col_out_arcs: Vec<(i32, i32)>,
    base_row_inds: Vec<i32>,
    base_row_vals: Vec<f64>,
    aggregated_path: Vec<(Vec<i32>, Vec<f64>)>,
    index_pos: Vec<i32>,
    inds: Vec<i32>,
    solval: Vec<f64>,
    upper: Vec<f64>,
    is_integral: Vec<bool>,
    tmp_upper: Vec<f64>,
    tmp_solval: Vec<f64>,
}

/// Rows and their coefficients of the arcs of a column
#[derive(Clone, Copy)]
struct Arcs<'a> {
    col_arcs: &'a [(i32, i32)],
    arc_rows: &'a [(i32, f64)],
}

impl PathSeparator {
    pub fn new(seed: u32) -> Self {
        PathSeparator {
            randgen: HighsRandom::new(seed),
            cutgen: CutGeneration::default(),
            rowtype: Vec::new(),
            num_continuous: Vec::new(),
            col_substitutions: Vec::new(),
            in_arc_rows: Vec::new(),
            col_in_arcs: Vec::new(),
            out_arc_rows: Vec::new(),
            col_out_arcs: Vec::new(),
            base_row_inds: Vec::new(),
            base_row_vals: Vec::new(),
            aggregated_path: Vec::new(),
            index_pos: Vec::new(),
            inds: Vec::new(),
            solval: Vec::new(),
            upper: Vec::new(),
            is_integral: Vec::new(),
            tmp_upper: Vec::new(),
            tmp_solval: Vec::new(),
        }
    }

    pub fn separate(&mut self, lp: &mut SepaRound, cutgen_seed: u32) {
        self.cutgen.reset(cutgen_seed, lp.feastol, lp.epsilon);
        let feastol = lp.feastol;
        let epsilon = lp.epsilon;
        let num_row = lp.num_row;
        let num_col = lp.num_col;
        let row_lower = lp.row_lower;
        let row_upper = lp.row_upper;
        let row_value = lp.row_value;
        let (row_lower, row_upper, row_value) = (row_lower.get(), row_upper.get(), row_value.get());
        let row_dual = lp.row_dual;
        let row_dual = row_dual.get();
        let (a_start, a_index, a_value) = (lp.a_start, lp.a_index, lp.a_value);
        let (a_start, a_index, a_value) = (a_start.get(), a_index.get(), a_value.get());
        let continuous_cols = lp.continuous_cols;
        let continuous_cols = continuous_cols.get();

        let rowtype = &mut self.rowtype;
        rowtype.clear();
        rowtype.resize(num_row, 0);
        for i in 0..num_row {
            if row_lower[i] == row_upper[i] {
                rowtype[i] = K_EQ;
                continue;
            }
            let mut lowerslack = f64::INFINITY;
            let mut upperslack = f64::INFINITY;
            if row_lower[i] != f64::NEG_INFINITY {
                lowerslack = row_value[i] - row_lower[i];
            }
            if row_upper[i] != f64::INFINITY {
                upperslack = row_upper[i] - row_value[i];
            }
            rowtype[i] = if lowerslack > feastol && upperslack > feastol {
                K_UNUSABLE
            } else if lowerslack < upperslack {
                K_GEQ
            } else {
                K_LEQ
            };
        }

        let num_continuous = &mut self.num_continuous;
        num_continuous.clear();
        num_continuous.resize(num_row, 0);
        let mut max_aggr_row_size = 0usize;
        for &col in continuous_cols {
            let c = col as usize;
            if lp.cols[c].bound_dist() == 0.0 {
                continue;
            }
            let (s, e) = (a_start[c] as usize, a_start[c + 1] as usize);
            max_aggr_row_size += e - s;
            for &r in &a_index[s..e] {
                num_continuous[r as usize] += 1;
            }
        }

        let col_substitutions = &mut self.col_substitutions;
        col_substitutions.clear();
        col_substitutions.resize(num_col, (-1, 0.0));
        for i in 0..num_row {
            if rowtype[i] != K_EQ || num_continuous[i] != 1 {
                continue;
            }
            let (rinds, rvals) = lp.row(i);
            let mut col = -1i32;
            let mut val = 0.0;
            for (&c, &v) in rinds.iter().zip(rvals) {
                let d = &lp.cols[c as usize];
                if d.integral || d.bound_dist() == 0.0 {
                    continue;
                }
                col = c;
                val = v;
                break;
            }
            let col = col as usize;
            if col_substitutions[col].0 != -1 {
                continue;
            }
            col_substitutions[col] = (i as i32, val);
            rowtype[i] = K_UNUSABLE;
        }

        let in_arc_rows = &mut self.in_arc_rows;
        let out_arc_rows = &mut self.out_arc_rows;
        in_arc_rows.clear();
        in_arc_rows.reserve(max_aggr_row_size);
        out_arc_rows.clear();
        out_arc_rows.reserve(max_aggr_row_size);
        let col_in_arcs = &mut self.col_in_arcs;
        let col_out_arcs = &mut self.col_out_arcs;
        col_in_arcs.clear();
        col_in_arcs.resize(num_col, (0, 0));
        col_out_arcs.clear();
        col_out_arcs.resize(num_col, (0, 0));
        for &col in continuous_cols {
            let c = col as usize;
            if lp.cols[c].bound_dist() == 0.0 {
                continue;
            }
            if col_substitutions[c].0 != -1 {
                continue;
            }
            col_in_arcs[c].0 = in_arc_rows.len() as i32;
            col_out_arcs[c].0 = out_arc_rows.len() as i32;
            for k in a_start[c] as usize..a_start[c + 1] as usize {
                let r = a_index[k];
                let v = a_value[k];
                match rowtype[r as usize] {
                    K_UNUSABLE => continue,
                    K_LEQ => {
                        if v < 0.0 {
                            in_arc_rows.push((r, v));
                        } else {
                            out_arc_rows.push((r, v));
                        }
                    }
                    K_GEQ => {
                        if v > 0.0 {
                            in_arc_rows.push((r, v));
                        } else {
                            out_arc_rows.push((r, v));
                        }
                    }
                    _ => {
                        in_arc_rows.push((r, v));
                        out_arc_rows.push((r, v));
                    }
                }
            }
            col_in_arcs[c].1 = in_arc_rows.len() as i32;
            col_out_arcs[c].1 = out_arc_rows.len() as i32;
        }

        let in_arcs = Arcs { col_arcs: col_in_arcs, arc_rows: in_arc_rows };
        let out_arcs = Arcs { col_arcs: col_out_arcs, arc_rows: out_arc_rows };
        let cutgen = &mut self.cutgen;
        let randgen = &mut self.randgen;
        let base_row_inds = &mut self.base_row_inds;
        let base_row_vals = &mut self.base_row_vals;
        let aggregated_path = &mut self.aggregated_path;
        let mut current_path = [0i32; MAX_PATH_LEN];
        let index_pos = &mut self.index_pos;
        index_pos.resize(num_col + num_row, 0);
        let inds = &mut self.inds;
        let solval = &mut self.solval;
        let upper = &mut self.upper;
        let is_integral = &mut self.is_integral;
        let tmp_upper = &mut self.tmp_upper;
        let tmp_solval = &mut self.tmp_solval;
        let max_weight = 1.0 / feastol;
        let min_weight = feastol;
        let check_weight = |w: f64| {
            let w = w.abs();
            w <= max_weight && w >= min_weight
        };

        for i in 0..num_row {
            let scales: [f64; 2] = match rowtype[i] {
                K_UNUSABLE => continue,
                K_EQ => {
                    if row_dual[i] <= epsilon {
                        [1.0, -1.0]
                    } else {
                        [-1.0, 1.0]
                    }
                }
                K_LEQ => [1.0, -1.0],
                _ => [-1.0, 1.0],
            };

            for &scale in &scales {
                debug_assert!(lp.aggr_is_empty());
                lp.aggr_add_row(i, scale);
                current_path[0] = i as i32;
                let mut curr_path_len = 1usize;
                let mut try_negated_scale = false;
                let mut num_paths = 0usize;
                let mut path_fractional = false;
                let mut success;

                while curr_path_len != MAX_PATH_LEN {
                    let in_path = |row: i32, len: usize| current_path[..len].contains(&row);
                    lp.aggr_get(base_row_inds, base_row_vals, false);
                    let mut added_substitution_rows = false;
                    let mut best_out_arc_col = -1i32;
                    let mut out_arc_col_val = 0.0;
                    let mut out_arc_col_bound_dist = 0.0;
                    let mut best_in_arc_col = -1i32;
                    let mut in_arc_col_val = 0.0;
                    let mut in_arc_col_bound_dist = 0.0;
                    let mut fractional = false;

                    // skipCol of the C++ (updates try_negated_scale)
                    let skip_col = |col: usize, arcs: Arcs, other: Arcs, tns: &mut bool| -> bool {
                        if curr_path_len == 1 && !*tns {
                            let (s, e) = arcs.col_arcs[col];
                            if (e - s) as usize <= curr_path_len {
                                for k in s..e {
                                    if arcs.arc_rows[k as usize].0 != i as i32 {
                                        *tns = true;
                                        break;
                                    }
                                }
                            } else {
                                *tns = true;
                            }
                        }
                        let (s, e) = other.col_arcs[col];
                        if s == e {
                            return true;
                        }
                        if (e - s) as usize <= curr_path_len {
                            for k in s..e {
                                if !in_path(other.arc_rows[k as usize].0, curr_path_len) {
                                    return false;
                                }
                            }
                            return true;
                        }
                        false
                    };

                    for j in 0..base_row_inds.len() {
                        let col = base_row_inds[j] as usize;
                        let d = &lp.cols[col];
                        fractional = fractional || d.fractional;
                        let bd = d.bound_dist();
                        if col >= num_col || bd == 0.0 || d.integral {
                            continue;
                        }
                        let subst = col_substitutions[col];
                        if subst.0 != -1 {
                            added_substitution_rows = true;
                            lp.aggr_add_row(subst.0 as usize, -base_row_vals[j] / subst.1);
                            continue;
                        }
                        if added_substitution_rows {
                            continue;
                        }
                        if base_row_vals[j] < 0.0 {
                            if skip_col(col, out_arcs, in_arcs, &mut try_negated_scale) {
                                continue;
                            }
                            if best_out_arc_col == -1 || bd > out_arc_col_bound_dist {
                                best_out_arc_col = col as i32;
                                out_arc_col_val = base_row_vals[j];
                                out_arc_col_bound_dist = bd;
                            }
                        } else {
                            if skip_col(col, in_arcs, out_arcs, &mut try_negated_scale) {
                                continue;
                            }
                            if best_in_arc_col == -1 || bd > in_arc_col_bound_dist {
                                best_in_arc_col = col as i32;
                                in_arc_col_val = base_row_vals[j];
                                in_arc_col_bound_dist = bd;
                            }
                        }
                    }

                    if added_substitution_rows {
                        continue;
                    }
                    path_fractional = path_fractional || fractional;

                    let mut rhs = 0.0;
                    success = fractional && cutgen.generate_cut(lp, base_row_inds, base_row_vals, &mut rhs, false);

                    lp.aggr_get(base_row_inds, base_row_vals, true);
                    if num_paths != 0 || best_out_arc_col != -1 || best_in_arc_col != -1 {
                        if aggregated_path.len() == num_paths {
                            aggregated_path.push((Vec::new(), Vec::new()));
                        }
                        let p = &mut aggregated_path[num_paths];
                        p.0.clone_from(base_row_inds);
                        p.1.clone_from(base_row_vals);
                        num_paths += 1;
                    }

                    rhs = 0.0;
                    success |= fractional && cutgen.generate_cut(lp, base_row_inds, base_row_vals, &mut rhs, false);

                    if success || (best_out_arc_col == -1 && best_in_arc_col == -1) {
                        break;
                    }

                    // findRow of the C++
                    let mut find_row = |best_arc_col: i32, val: f64, arcs: Arcs| -> Option<(i32, f64)> {
                        let (s, e) = arcs.col_arcs[best_arc_col as usize];
                        let arc_row = randgen.integer_between(s, e);
                        let try_row = |k: i32| {
                            let (r, a) = arcs.arc_rows[k as usize];
                            let w = -val / a;
                            if !in_path(r, curr_path_len) && check_weight(w) {
                                Some((r, w))
                            } else {
                                None
                            }
                        };
                        if let Some(rw) = try_row(arc_row) {
                            return Some(rw);
                        }
                        for next_row in arc_row + 1..e {
                            if let Some(rw) = try_row(next_row) {
                                return Some(rw);
                            }
                        }
                        for next_row in s..arc_row {
                            if let Some(rw) = try_row(next_row) {
                                return Some(rw);
                            }
                        }
                        None
                    };

                    let found = if best_in_arc_col == -1
                        || (best_out_arc_col != -1 && out_arc_col_bound_dist >= in_arc_col_bound_dist - feastol)
                    {
                        match find_row(best_out_arc_col, out_arc_col_val, in_arcs) {
                            Some(rw) => Some(rw),
                            None => {
                                if best_in_arc_col == -1 {
                                    None
                                } else {
                                    find_row(best_in_arc_col, in_arc_col_val, out_arcs)
                                }
                            }
                        }
                    } else {
                        find_row(best_in_arc_col, in_arc_col_val, out_arcs)
                    };
                    let Some((row, weight)) = found else { break };
                    current_path[curr_path_len] = row;
                    lp.aggr_add_row(row as usize, weight);
                    curr_path_len += 1;
                }

                // path mixing cut
                let mut path_len = num_paths;
                if path_len > 1 && path_fractional {
                    for &c in inds.iter() {
                        index_pos[c as usize] = 0;
                    }
                    inds.clear();
                    solval.clear();
                    upper.clear();
                    is_integral.clear();
                    let mut rhs = [0.0f64; MAX_PATH_LEN];
                    let mut delta = 1.0f64;
                    for k in 0..path_len {
                        let mut integral_positive = false;
                        let (pinds, pvals) = &mut aggregated_path[k];
                        if !lp.transform(pvals, tmp_upper, tmp_solval, pinds, &mut rhs[k], &mut integral_positive, false)
                        {
                            path_len = k;
                            break;
                        }
                        if k == 0 {
                            if rhs[k] > K_HIGHS_TINY {
                                path_len = k;
                                break;
                            }
                            if rhs[k] >= -feastol {
                                rhs[k] = 0.0;
                            }
                        } else if rhs[k] >= rhs[k - 1] - feastol {
                            path_len = k;
                            break;
                        }
                        delta = rhs[k].abs().max_cpp(delta);
                        for j in 0..pinds.len() {
                            let index = pinds[j];
                            let pos = &mut index_pos[index as usize];
                            if *pos == 0 {
                                inds.push(index);
                                solval.push(tmp_solval[j]);
                                upper.push(tmp_upper[j]);
                                let integral = lp.cols[index as usize].integral;
                                is_integral.push(integral);
                                if integral {
                                    delta = pvals[j].abs().max_cpp(delta);
                                }
                                *pos = inds.len() as i32;
                            } else if is_integral[*pos as usize - 1] {
                                delta = pvals[j].abs().max_cpp(delta);
                            }
                        }
                    }

                    if path_len > 1 {
                        delta = (delta + 1.0).log2().ceil().exp2();
                        let mut num_inds = inds.len();
                        let mut cut_rhs = CDouble::from(0.0);
                        let mut cut_vals = vec![0.0f64; num_inds];
                        let mut max_frac = vec![0.0f64; num_inds];
                        let mut down_sum = vec![CDouble::from(0.0); num_inds];
                        let mut f_sum = vec![CDouble::from(0.0); num_inds];
                        let mut f_last = 0.0;
                        let scale = -1.0 / delta;
                        for k in 0..path_len {
                            let f = rhs[k] * scale;
                            let f_diff = CDouble::from(f) - f_last;
                            cut_rhs += f_diff;
                            let (pinds, pvals) = &aggregated_path[k];
                            for j in 0..pinds.len() {
                                let ii = (index_pos[pinds[j] as usize] - 1) as usize;
                                let gj = pvals[j] * scale;
                                if !is_integral[ii] {
                                    cut_vals[ii] = cut_vals[ii].max_cpp(gj);
                                } else {
                                    let gjdown = gj.floor();
                                    let hj = gj - gjdown;
                                    max_frac[ii] = max_frac[ii].max_cpp(hj);
                                    down_sum[ii] += f_diff * gjdown;
                                    f_sum[ii] += f_diff;
                                    cut_vals[ii] = if f_sum[ii] < max_frac[ii] {
                                        (down_sum[ii] + f_sum[ii]).to_f64()
                                    } else {
                                        (down_sum[ii] + max_frac[ii]).to_f64()
                                    };
                                }
                            }
                            if k > 0 {
                                // viol -= solval[j] * cutVals[j]: the LTO build
                                // vectorizes the loop in blocks of 8 with
                                // rounded products, the remainder is fused
                                let mut viol = cut_rhs.to_f64();
                                let blocked = if num_inds >= 8 { num_inds & !7 } else { 0 };
                                for j in 0..blocked {
                                    viol += -solval[j] * cut_vals[j];
                                }
                                for j in blocked..num_inds {
                                    viol = (-solval[j]).mul_add(cut_vals[j], viol);
                                }
                                viol *= delta;
                                if viol > 10.0 * feastol {
                                    let scale = -delta;
                                    let mut rhs = cut_rhs.to_f64() * scale;
                                    for v in cut_vals.iter_mut() {
                                        *v *= scale;
                                    }
                                    let mut j = num_inds;
                                    while j > 0 {
                                        j -= 1;
                                        if cut_vals[j].abs() <= epsilon {
                                            num_inds -= 1;
                                            cut_vals.swap(j, num_inds);
                                            inds.swap(j, num_inds);
                                        }
                                    }
                                    cut_vals.truncate(num_inds);
                                    // the cut is built in a copy so that inds
                                    // keeps the positions to clear
                                    let mut cut_inds = inds[..num_inds].to_vec();
                                    if lp.untransform(&mut cut_vals, &mut cut_inds, &mut rhs, false) {
                                        cutgen.finalize_and_add_cut(lp, &mut cut_inds, &mut cut_vals, &mut rhs);
                                    }
                                    break;
                                }
                            }
                            f_last = f;
                        }
                    }
                }

                lp.aggr_clear();
                if !try_negated_scale {
                    break;
                }
            }
        }
        for &c in inds.iter() {
            index_pos[c as usize] = 0;
        }
        inds.clear();
    }
}
