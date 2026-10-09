//! HighsTableauSeparator::separateLpSolution: cuts from rows of the simplex
//! tableau of fractional basic integer variables.

use crate::ffi::CHVec;
use crate::hvector::OwnedHVec;
use crate::util::fma::ClangFma;
use super::cut_generation::CutGeneration;
use super::round::{MinCpp, SepaRound};
use super::sort::{pdqsort, pdqsort_branchless};
use crate::util::hash::pair_hash;

#[derive(Clone, Copy)]
struct FractionalInteger {
    fractionality: f64,
    row_ep_norm2: f64,
    score: f64,
    basis_index: i32,
    /// row_ep: a range of TableauSeparator::row_ep
    ep_start: usize,
    ep_len: usize,
}

/// HighsHashHelpers::hash(int64_t)
#[inline]
fn hash_i64(v: i64) -> u64 {
    let (a, b) = (v as u64 as u32, (v as u64 >> 32) as u32);
    pair_hash::<1>(a, b) ^ (pair_hash::<0>(a, b) >> 32)
}

/// fractionality of HighsUtils.h
#[inline]
fn fractionality(x: f64) -> f64 {
    (x - x.round()).abs()
}

/// std::make_pair(a1, a2) > std::make_pair(b1, b2)
#[inline]
fn pair_greater(a: (f64, u64), b: (f64, u64)) -> bool {
    b.0 < a.0 || (!(a.0 < b.0) && b.1 < a.1)
}

#[derive(Default)]
pub struct TableauSeparator {
    pub num_tries: i64,
    fracvars: Vec<FractionalInteger>,
    row_ep: Vec<(i32, f64)>,
    base_row_inds: Vec<i32>,
    base_row_vals: Vec<f64>,
}

impl TableauSeparator {
    /// `num_calls`: HighsSeparator::getNumCalls(); returns early as the C++
    /// (the caller checks hasInvert)
    /// `basisinds`: Highs::getBasicVariablesArray(); `lp_iterations`:
    /// total_lp_iterations - heuristic_lp_iterations
    pub fn separate(
        &mut self,
        lp: &mut SepaRound,
        cutgen: &mut CutGeneration,
        num_calls: i32,
        basisinds: &[i32],
        lp_iterations: i64,
    ) {
        // SAFETY: C++ queries
        let num_available = unsafe { (lp.host.num_available_cuts)(lp.host.ctx) };
        if num_available > lp.mip_pool_soft_limit {
            return;
        }
        let num_row = lp.num_row;
        let num_col = lp.num_col;
        let feastol = lp.feastol;
        let row_value = lp.row_value;
        let col_value = lp.col_value;
        let (row_value, col_value) = (row_value.get(), col_value.get());

        let fracvars = &mut self.fracvars;
        fracvars.clear();
        self.row_ep.clear();
        for i in 0..num_row {
            let b = basisinds[i] as usize;
            let my_fractionality = if b >= num_col {
                let row = b - num_col;
                if !lp.cols[b].integral {
                    continue;
                }
                fractionality(row_value[row])
            } else {
                if !lp.cols[b].integral {
                    continue;
                }
                fractionality(col_value[b])
            };
            if my_fractionality < 1000.0 * feastol {
                continue;
            }
            fracvars.push(FractionalInteger {
                fractionality: my_fractionality,
                row_ep_norm2: 0.0,
                score: -1.0,
                basis_index: i as i32,
                ep_start: 0,
                ep_len: 0,
            });
        }
        if fracvars.is_empty() {
            return;
        }
        let mut max_tries: i64 = 5000 + num_calls as i64 * 50 + lp_iterations / 10;
        if self.num_tries >= max_tries {
            return;
        }
        max_tries -= self.num_tries;
        let num_integral_cols = lp.integral_cols.get().len() as i32;
        max_tries = max_tries.min(200 + (0.1 * (num_row as i32).min(num_integral_cols) as f64) as i64);

        let num_tries = self.num_tries;
        if fracvars.len() as i64 > max_tries {
            // SAFETY: the LP solver's DSE weights (num_row entries), if any
            let edge_wt = unsafe { (*lp.lph).dual_edge_weights() };
            if !edge_wt.is_null() {
                let edge_wt = unsafe { std::slice::from_raw_parts(edge_wt, num_row) };
                pdqsort(fracvars, |f1, f2| {
                    let score1 = f1.fractionality * (1.0 - f1.fractionality) / edge_wt[f1.basis_index as usize];
                    let score2 = f2.fractionality * (1.0 - f2.fractionality) / edge_wt[f2.basis_index as usize];
                    pair_greater(
                        (score1, hash_i64(num_tries + f1.basis_index as i64)),
                        (score2, hash_i64(num_tries + f2.basis_index as i64)),
                    )
                });
            } else {
                pdqsort(fracvars, |f1, f2| {
                    pair_greater(
                        (f1.fractionality, hash_i64(num_tries + f1.basis_index as i64)),
                        (f2.fractionality, hash_i64(num_tries + f2.basis_index as i64)),
                    )
                });
            }
            fracvars.truncate(max_tries as usize);
        }
        self.num_tries += fracvars.len() as i64;

        // the row vector of the round, sized for the LP's rows
        if lp.row_ep.size != num_row as i32 {
            lp.row_ep = OwnedHVec::new(num_row as i32);
        }
        for fv in fracvars.iter_mut() {
            // getBasisInverseRowSparse(basis_index): row_ep's count, index
            // and array (num_row entries), valid until its next use
            let mut v = CHVec::of(&mut lp.row_ep);
            // SAFETY: the view of row_ep, of num_row entries
            unsafe { (*lp.lph).basis_inverse_row_sparse(fv.basis_index, &mut v) };
            v.store_into(&mut lp.row_ep);
            let (count, index, array) = (lp.row_ep.count, lp.row_ep.index.as_ptr(), lp.row_ep.array.as_ptr());
            if count == 1 {
                continue;
            }
            let index = unsafe { crate::ffi::sl(index, count) };
            let array = unsafe { std::slice::from_raw_parts(array, num_row) };
            fv.row_ep_norm2 = 0.0;
            let mut min_weight = f64::INFINITY;
            let mut max_weight = 0.0f64;
            fv.ep_start = self.row_ep.len();
            for &row in index {
                let weight = array[row as usize];
                let max_abs_row_val = lp.row_max_abs[row as usize];
                let scaled_weight = max_abs_row_val * weight.abs();
                if scaled_weight <= feastol {
                    continue;
                }
                min_weight = min_weight.min_cpp(scaled_weight);
                max_weight = max_weight.max_cpp(scaled_weight);
                fv.row_ep_norm2 = scaled_weight.mul_add_c(scaled_weight, fv.row_ep_norm2);
                self.row_ep.push((row, weight));
            }
            fv.ep_len = self.row_ep.len() - fv.ep_start;
            if fv.ep_len <= 1 {
                continue;
            }
            if max_weight / min_weight <= 1e4 {
                fv.score = fv.fractionality * (1.0 - fv.fractionality) / fv.row_ep_norm2;
            }
        }

        fracvars.retain(|f| !(f.score <= feastol));
        if fracvars.is_empty() {
            return;
        }
        pdqsort_branchless(fracvars, |a, b| a.score > b.score);
        let mut best_score = -1.0;
        let num_cuts = lp.num_cuts();
        const BEST_SCORE_FAC: [f64; 2] = [0.0025, 0.01];

        for fv in fracvars.iter() {
            let found = lp.num_cuts() - num_cuts;
            if found >= 1000 {
                break;
            }
            if fv.score < BEST_SCORE_FAC[(found >= 50) as usize] * best_score {
                break;
            }
            let row_ep = &self.row_ep[fv.ep_start..fv.ep_start + fv.ep_len];
            for &(row, weight) in row_ep {
                lp.aggr_add_row(row as usize, weight);
            }
            lp.aggr_get(&mut self.base_row_inds, &mut self.base_row_vals, false);
            if 10usize.wrapping_mul(self.base_row_inds.len().wrapping_sub(fv.ep_len)) > 10000 + num_col {
                lp.aggr_clear();
                continue;
            }
            let len = self.base_row_inds.len();
            if len > fv.ep_len {
                let mut max_abs_val = 0.0f64;
                let mut min_abs_val = f64::INFINITY;
                for (&c, &v) in self.base_row_inds.iter().zip(&self.base_row_vals) {
                    if (c as usize) < num_col {
                        max_abs_val = v.abs().max_cpp(max_abs_val);
                        min_abs_val = v.abs().min_cpp(min_abs_val);
                    }
                }
                if max_abs_val / min_abs_val > 1e6 {
                    lp.aggr_clear();
                    continue;
                }
            }
            let mut rhs = 0.0;
            cutgen.generate_cut(lp, &mut self.base_row_inds, &mut self.base_row_vals, &mut rhs, false);
            if lp.dom_infeasible() {
                break;
            }
            lp.aggr_get(&mut self.base_row_inds, &mut self.base_row_vals, true);
            rhs = 0.0;
            cutgen.generate_cut(lp, &mut self.base_row_inds, &mut self.base_row_vals, &mut rhs, false);
            if lp.dom_infeasible() {
                break;
            }
            lp.aggr_clear();
            if best_score == -1.0 && lp.num_cuts() != num_cuts {
                best_score = fv.score;
            }
        }
    }
}
