//! HighsModkSeparator::separateLpSolution: maximally violated mod-k cuts
//! from the congruence system of the tight integral rows.

use super::cut_generation::CutGeneration;
use super::gfk::{GfkSolve, SolutionEntry};
use super::integers::{integral_scale, nearest_integer};
use super::round::SepaRound;
use super::sort::pdqsort;
use std::collections::HashSet;

pub fn separate(lp: &mut SepaRound, cutgen: &mut CutGeneration) {
    let num_row = lp.num_row;
    let num_col = lp.num_col;
    let feastol = lp.feastol;
    let epsilon = lp.epsilon;
    let (a_start, a_index) = (lp.a_start, lp.a_index);
    let (a_start, a_index) = (a_start.get(), a_index.get());
    let (row_lower, row_upper, row_value) = (lp.row_lower, lp.row_upper, lp.row_value);
    let (row_lower, row_upper, row_value) = (row_lower.get(), row_upper.get(), row_value.get());

    let mut skip_row = vec![false; num_row];
    let continuous_cols = lp.continuous_cols;
    for &col in continuous_cols.get() {
        let c = col as usize;
        if lp.cols[c].bound_dist() == 0.0 {
            continue;
        }
        for &r in &a_index[a_start[c] as usize..a_start[c + 1] as usize] {
            skip_row[r as usize] = true;
        }
    }

    let mut integral_scales: Vec<(i32, f64)> = Vec::new();
    let mut int_system_value: Vec<i64> = Vec::with_capacity(a_index.len() + num_row);
    let mut int_system_index: Vec<i32> = Vec::with_capacity(a_index.len() + num_row);
    let mut int_system_start: Vec<i32> = Vec::with_capacity(num_row + 1);
    int_system_start.push(0);
    let mut inds: Vec<i32> = Vec::with_capacity(num_col);
    let mut vals: Vec<f64> = Vec::with_capacity(num_col);
    let mut scale_vals: Vec<f64> = Vec::with_capacity(num_col);
    let mut upper: Vec<f64> = Vec::new();
    let mut solval: Vec<f64> = Vec::new();
    let mut num_nonzero_rhs = 0;
    // 1000 + 0.1 * num_col_, fused by clang
    let max_int_row_len = (num_col as i32 as f64).mul_add(0.1, 1000.0) as i32 as usize;

    for row in 0..num_row {
        if skip_row[row] {
            continue;
        }
        let leq_row = if row_upper[row] - row_value[row] <= feastol {
            true
        } else if row_value[row] - row_lower[row] <= feastol {
            false
        } else {
            continue;
        };
        let mut rhs;
        {
            let (rinds, rvals) = lp.row(row);
            inds.clear();
            inds.extend_from_slice(rinds);
            vals.clear();
            if leq_row {
                rhs = row_upper[row];
                vals.extend_from_slice(rvals);
            } else {
                rhs = -row_lower[row];
                vals.extend(rvals.iter().map(|&x| -x));
            }
        }
        let mut integral_positive = false;
        if !lp.transform(&mut vals, &mut upper, &mut solval, &mut inds, &mut rhs, &mut integral_positive, true) {
            continue;
        }
        let rowlen = inds.len();
        if rowlen > max_int_row_len {
            let mut int_row_len = 0;
            for i in 0..rowlen {
                if solval[i] <= feastol {
                    continue;
                }
                if !lp.cols[inds[i] as usize].integral {
                    continue;
                }
                int_row_len += 1;
            }
            if int_row_len > max_int_row_len || (int_row_len == 0 && rhs.abs() <= epsilon) {
                continue;
            }
        }
        let intscale;
        let intrhs;
        if !lp.cols[num_col + row].integral {
            scale_vals.clear();
            for i in 0..rowlen {
                if !lp.cols[inds[i] as usize].integral {
                    continue;
                }
                if solval[i] > feastol {
                    scale_vals.push(vals[i]);
                }
            }
            if rhs.abs() > epsilon {
                scale_vals.push(-rhs);
            }
            if scale_vals.is_empty() {
                continue;
            }
            intscale = integral_scale(&scale_vals, feastol, epsilon);
            if intscale == 0.0 || intscale > 1e6 {
                continue;
            }
            intrhs = nearest_integer(intscale * rhs);
            for i in 0..rowlen {
                if !lp.cols[inds[i] as usize].integral {
                    continue;
                }
                if solval[i] > feastol {
                    int_system_index.push(inds[i]);
                    int_system_value.push(nearest_integer(intscale * vals[i]));
                }
            }
        } else {
            intscale = 1.0;
            intrhs = nearest_integer(rhs);
            for i in 0..rowlen {
                if solval[i] > feastol {
                    int_system_index.push(inds[i]);
                    int_system_value.push(nearest_integer(vals[i]));
                }
            }
        }
        num_nonzero_rhs += (intrhs != 0) as i32;
        int_system_index.push(num_col as i32);
        int_system_value.push(intrhs);
        int_system_start.push(int_system_value.len() as i32);
        integral_scales.push((row as i32, intscale));
    }

    if integral_scales.is_empty() || num_nonzero_rhs == 0 {
        return;
    }

    let mut used_weights: HashSet<Vec<SolutionEntry>> = HashSet::new();
    let mut gfk = GfkSolve::default();
    for k in [2u32, 3, 5, 7] {
        used_weights.clear();
        let num_cuts = lp.num_cuts();
        gfk.from_csc(k, &int_system_value, &int_system_index, &int_system_start, num_col + 1);
        gfk.set_rhs(k, num_col, 1);
        gfk.solve(k, |weights| {
            if weights.is_empty() {
                return;
            }
            pdqsort(weights, |a, b| a.index < b.index);
            if !used_weights.insert(weights.clone()) {
                return;
            }
            for w in weights.iter() {
                let weight = integral_scales[w.index as usize].1
                    * ((w.weight.wrapping_mul(k - 1) % k) as f64 / k as f64);
                lp.aggr_add_row(integral_scales[w.index as usize].0 as usize, weight);
            }
            lp.aggr_get(&mut inds, &mut vals, false);
            let mut rhs = 0.0;
            cutgen.generate_cut(lp, &mut inds, &mut vals, &mut rhs, true);
            if k != 2 {
                lp.aggr_clear();
                for w in weights.iter() {
                    let weight = integral_scales[w.index as usize].1 * (w.weight as f64 / k as f64);
                    lp.aggr_add_row(integral_scales[w.index as usize].0 as usize, weight);
                }
            }
            lp.aggr_get(&mut inds, &mut vals, true);
            rhs = 0.0;
            cutgen.generate_cut(lp, &mut inds, &mut vals, &mut rhs, true);
            lp.aggr_clear();
        });
        if lp.num_cuts() != num_cuts {
            return;
        }
    }
}
