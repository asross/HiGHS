//! The rounding heuristics of HighsPrimalHeuristics that work on a point
//! alone: ziRound and shifting (HighsPrimalHeuristics.cpp). What they
//! find is still tried by the C++ (trySolution, tryRoundedPoint).
//!
//! Same arithmetic as the C++, with less of it: ziRound recomputes all row
//! activities (in double-double) before each fractional column, and
//! shifting all row infeasibilities after each shift; here they are
//! recomputed only where the point changed since, which gives the same
//! values. Shifting's fractional set has a membership array next to its
//! ordered list.

use crate::util::cdouble::CDouble;
use crate::util::random::HighsRandom;

/// The (presolved) model: column-wise matrix, bounds, costs
pub struct Lp<'a> {
    pub a_start: &'a [i32],
    pub a_index: &'a [i32],
    pub a_value: &'a [f64],
    pub col_lower: &'a [f64],
    pub col_upper: &'a [f64],
    pub col_cost: &'a [f64],
    pub row_lower: &'a [f64],
    pub row_upper: &'a [f64],
    pub minimize: bool,
}

/// std::min(a, b)
#[inline]
fn cmin(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// std::max(a, b)
#[inline]
fn cmax(a: f64, b: f64) -> f64 {
    if a < b {
        b
    } else {
        a
    }
}

/// calculateRowValuesQuad: row activities summed column by column
fn row_values_quad(lp: &Lp, x: &[f64], out: &mut [f64], quad: &mut Vec<CDouble>) {
    quad.clear();
    quad.resize(out.len(), CDouble::from(0.0));
    for col in 0..x.len() {
        let xc = CDouble::from(x[col]);
        for p in lp.a_start[col] as usize..lp.a_start[col + 1] as usize {
            quad[lp.a_index[p] as usize] += xc * lp.a_value[p];
        }
    }
    for (o, q) in out.iter_mut().zip(quad.iter()) {
        *o = q.to_f64();
    }
}

/// HighsPrimalHeuristics::ziRound on `x` (in place). Returns false where
/// the C++ returns before trying the point (it is already integral).
pub fn zi_round(lp: &Lp, intcols: &[i32], feastol: f64, x: &mut [f64]) -> bool {
    let zi = |v: f64| cmin((v - feastol).ceil() - v, v - (v + feastol).floor());
    let mut zi_total = CDouble::from(0.0);
    for &i in intcols {
        zi_total += zi(x[i as usize]);
    }
    if zi_total <= feastol {
        return false;
    }
    let nrow = lp.row_lower.len();
    let mut act = vec![0.0; nrow];
    let mut quad = Vec::new();
    // the activities are those of x unless x changed since
    let mut stale = true;
    let mut loop_count = 0;
    let mut improvement = CDouble::from(f64::INFINITY);
    while zi_total > feastol && improvement > feastol && loop_count <= 5 {
        let previous = zi_total;
        loop_count += 1;
        for &jj in intcols {
            let j = jj as usize;
            let rs = x[j];
            if (rs - rs.round()).abs() <= feastol {
                continue;
            }
            if stale {
                row_values_quad(lp, x, &mut act, &mut quad);
                stale = false;
            }
            let mut min_up = f64::INFINITY;
            let mut min_lo = f64::INFINITY;
            for p in lp.a_start[j] as usize..lp.a_start[j + 1] as usize {
                let i = lp.a_index[p] as usize;
                let aij = lp.a_value[p];
                let slack_upper = lp.row_upper[i] - act[i];
                let slack_lower = act[i] - lp.row_lower[i];
                min_up = cmin(min_up, (if aij > 0.0 { slack_upper } else { -slack_lower }) / aij);
                min_lo = cmin(min_lo, (if aij > 0.0 { slack_lower } else { -slack_upper }) / aij);
            }
            let ub = cmin(lp.col_upper[j] - rs, min_up);
            let lb = cmin(rs - lp.col_lower[j], min_lo);
            let mut update = |change: f64| {
                let old = x[j];
                x[j] += change;
                zi_total = zi_total - zi(old) + zi(x[j]);
                stale = true;
            };
            let (zu, zl, z0) = (zi(rs + ub), zi(rs - lb), zi(rs));
            if (zu - zl).abs() <= feastol && zu < z0 {
                let c = lp.col_cost[j];
                let ub_smaller = c * (rs + ub) <= c * (rs - lb);
                if lp.minimize == ub_smaller {
                    update(ub);
                } else {
                    update(-lb);
                }
            } else if zu < zl && zu < z0 {
                update(ub);
            } else if zu > zl && zl < z0 {
                update(-lb);
            }
        }
        improvement = previous - zi_total;
    }
    true
}

/// The model's row-wise matrix and what shifting reads of the MIP
pub struct MipRows<'a> {
    pub ar_start: &'a [i32],
    pub ar_index: &'a [i32],
    pub ar_value: &'a [f64],
    pub integrality: &'a [u8],
    pub uplocks: &'a [i32],
    pub downlocks: &'a [i32],
    /// the original model maximizes
    pub maximize: bool,
    pub num_integer_cols: usize,
}

const K_INTEGER: u8 = 1;

/// The row activities (double-double, row-wise as getInfeasibleRows) of
/// the rows marked stale
fn refresh_rows(mr: &MipRows, x: &[f64], act: &mut [f64], stale: &mut Vec<u32>, is_stale: &mut [bool]) {
    for &r in stale.iter() {
        let r = r as usize;
        let mut q = CDouble::from(0.0);
        for p in mr.ar_start[r] as usize..mr.ar_start[r + 1] as usize {
            q += CDouble::from(x[mr.ar_index[p] as usize]) * mr.ar_value[p];
        }
        act[r] = q.to_f64();
        is_stale[r] = false;
    }
    stale.clear();
}

/// HighsMipSolverData::getInfeasibleRows from the activities
fn infeasible_rows(lp: &Lp, act: &[f64], feastol: f64, out: &mut Vec<(usize, i32, f64)>) {
    out.clear();
    for (i, &a) in act.iter().enumerate() {
        if a > lp.row_upper[i] + feastol {
            out.push((i, 1, (a - lp.row_upper[i]).abs()));
        }
        if a < lp.row_lower[i] - feastol {
            out.push((i, -1, (lp.row_lower[i] - a).abs()));
        }
    }
}

/// HighsPrimalHeuristics::shifting from `x` (in place), with the LP's
/// fractional integers `frac` (left with those still fractional). Returns
/// whether rows are still infeasible.
pub fn shifting(
    lp: &Lp,
    mr: &MipRows,
    feastol: f64,
    frac: &mut Vec<(i32, f64)>,
    rng: &mut HighsRandom,
    x: &mut [f64],
) -> bool {
    let ncol = x.len();
    let nrow = lp.row_lower.len();
    let mut in_frac = vec![false; ncol];
    for &(c, _) in frac.iter() {
        in_frac[c as usize] = true;
    }
    let mut act = vec![0.0; nrow];
    let mut is_stale = vec![true; nrow];
    let mut stale: Vec<u32> = (0..nrow as u32).collect();
    refresh_rows(mr, x, &mut act, &mut stale, &mut is_stale);
    let mut infeasible = Vec::new();
    infeasible_rows(lp, &act, feastol, &mut infeasible);
    let mut previous_size = infeasible.len();
    let mut has_infeasible = !infeasible.is_empty();
    let mut no_reduction = 0;
    let mut shift_iterations: std::collections::HashMap<usize, Vec<i32>> = Default::default();
    let mut t: i32 = 0;
    let erase = |frac: &mut Vec<(i32, f64)>, in_frac: &mut [bool], col: usize| {
        if let Some(p) = frac.iter().position(|f| f.0 as usize == col) {
            frac.remove(p);
            in_frac[col] = false;
        }
    };
    while (!frac.is_empty() || has_infeasible) && no_reduction <= 5 && t as i64 <= mr.num_integer_cols as i64 {
        t += 1;
        let mut reduced = false;
        no_reduction += 1;
        if has_infeasible {
            // the first infeasible row with a fractional integer, else a
            // random one
            let mut found = false;
            let mut r_index = 0;
            while !found && r_index != infeasible.len() {
                let r = infeasible[r_index].0;
                found = (mr.ar_start[r] as usize..mr.ar_start[r + 1] as usize)
                    .any(|p| in_frac[mr.ar_index[p] as usize]);
                r_index += 1;
            }
            let row_index = if found {
                r_index - 1
            } else {
                rng.integer_below(infeasible.len() as i32) as usize
            };
            let (row, row_sense, infeasibility) = infeasible[row_index];
            let mut score_min = f64::INFINITY;
            let mut j_min = usize::MAX;
            let mut x_j_min = f64::INFINITY;
            let mut aij_min = 0.0;
            let mut move_up = false;
            for p in mr.ar_start[row] as usize..mr.ar_start[row + 1] as usize {
                let j = mr.ar_index[p] as usize;
                if lp.col_lower[j] == lp.col_upper[j] {
                    continue;
                }
                let coef = mr.ar_value[p];
                let cost = lp.col_cost[j];
                let is_integer = mr.integrality[j] == K_INTEGER;
                let mut repair = |direction: i32, num_locks: i32, at_bound: bool| {
                    let dc = direction as f64 * coef;
                    if (row_sense < 0 || dc > 0.0) && (row_sense > 0 || dc < 0.0) {
                        return;
                    }
                    if at_bound {
                        return;
                    }
                    let score;
                    if in_frac[j] {
                        score = -1.0 + 1.0 / (num_locks + 1) as f64;
                    } else {
                        let mut s;
                        match shift_iterations.get(&j) {
                            None => s = direction as f64 * if mr.maximize { -cost } else { cost },
                            Some(shifts) => {
                                s = 0.0;
                                for &shift in shifts {
                                    let ds = direction as f64 * shift as f64;
                                    if ds > 0.0 {
                                        s += 1.1f64.powf(ds - t as f64);
                                    }
                                }
                            }
                        }
                        if is_integer {
                            s += 1.0;
                        }
                        score = s;
                    }
                    if score < score_min {
                        score_min = score;
                        j_min = j;
                        aij_min = coef;
                        x_j_min = x[j];
                        move_up = direction > 0;
                    }
                };
                repair(1, mr.downlocks[j], (lp.col_upper[j] - x[j]).abs() <= feastol);
                repair(-1, mr.uplocks[j], (x[j] - lp.col_lower[j]).abs() <= feastol);
            }
            if j_min != usize::MAX {
                if in_frac[j_min] {
                    erase(frac, &mut in_frac, j_min);
                    reduced = true;
                }
                let integer = mr.integrality[j_min] == K_INTEGER;
                if move_up {
                    x[j_min] = if reduced {
                        (x_j_min - feastol).ceil()
                    } else if integer {
                        x_j_min + 1.0
                    } else {
                        cmin(x_j_min + infeasibility / aij_min.abs(), lp.col_upper[j_min] + feastol)
                    };
                    if !reduced {
                        shift_iterations.entry(j_min).or_default().push(t);
                    }
                } else {
                    x[j_min] = if reduced {
                        (x_j_min + feastol).floor()
                    } else if integer {
                        x_j_min - 1.0
                    } else {
                        cmax(x_j_min - infeasibility / aij_min.abs(), lp.col_lower[j_min] - feastol)
                    };
                    if !reduced {
                        shift_iterations.entry(j_min).or_default().push(-t);
                    }
                }
                mark_column(lp, j_min, &mut stale, &mut is_stale);
            }
        } else {
            let mut xi_max = -1.0;
            let mut delta_c_min = f64::INFINITY;
            let mut pind = usize::MAX;
            let mut j_min = usize::MAX;
            let mut x_j_min = f64::INFINITY;
            let mut sigma = 0;
            for (i, &(c, v)) in frac.iter().enumerate() {
                let col = c as usize;
                let mut is_better = |xi: f64, rounded: f64, direction: i32| {
                    let c_min = lp.col_cost[col] * (rounded - v);
                    if xi > xi_max || (xi == xi_max && c_min < delta_c_min) {
                        xi_max = xi;
                        delta_c_min = c_min;
                        pind = i;
                        j_min = col;
                        x_j_min = rounded;
                        sigma = direction;
                    }
                };
                is_better(mr.uplocks[col] as f64, (v + feastol).floor(), -1);
                is_better(mr.downlocks[col] as f64, (v - feastol).ceil(), 1);
            }
            if sigma != 0 {
                x[j_min] = x_j_min;
                mark_column(lp, j_min, &mut stale, &mut is_stale);
            }
            if pind != usize::MAX {
                let (c, _) = frac.remove(pind);
                in_frac[c as usize] = false;
                reduced = true;
            }
        }
        refresh_rows(mr, x, &mut act, &mut stale, &mut is_stale);
        infeasible_rows(lp, &act, feastol, &mut infeasible);
        has_infeasible = !infeasible.is_empty();
        if infeasible.len() < previous_size || reduced {
            no_reduction = 0;
        }
        previous_size = infeasible.len();
    }
    has_infeasible
}

/// Marks the rows of column `j` for recomputation
fn mark_column(lp: &Lp, j: usize, stale: &mut Vec<u32>, is_stale: &mut [bool]) {
    for p in lp.a_start[j] as usize..lp.a_start[j + 1] as usize {
        let r = lp.a_index[p] as usize;
        if !is_stale[r] {
            is_stale[r] = true;
            stale.push(r as u32);
        }
    }
}

