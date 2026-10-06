//! detectParallelRowsAndCols: rows and columns hashed by their scaled
//! coefficients, and the (nearly) parallel ones merged, fixed or added.
//!
//! The C++ buckets are a std::unordered_multimap that is only probed by
//! key, so only the order within a key's group matters: a group keeps its
//! elements together through rehashes, and emplace_hint(last, ...) inserts
//! before `last`, the last element of the group that was visited (libc++),
//! or after it (libstdc++: _M_insert_multi_node with an equivalent hint).

use super::*;
use crate::util::hash::{double_hash_code, sparse_combine};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// The keys are hash values already: use them as they are
#[derive(Default)]
struct IdHasher(u64);
impl Hasher for IdHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, _: &[u8]) {
        unreachable!()
    }
    fn write_u64(&mut self, v: u64) {
        self.0 = v;
    }
}

const MERGE_PARALLEL_COLS: i32 = 0;
const DOMINANCE_COL_TO_UPPER: i32 = 1;
const DOMINANCE_COL_TO_LOWER: i32 = 2;
const DOMINANCE_DUPLICATE_COL_TO_UPPER: i32 = 3;
const DOMINANCE_DUPLICATE_COL_TO_LOWER: i32 = 4;

/// The buckets of one hash value, and the visited ones
#[derive(Default)]
struct Buckets {
    groups: HashMap<u64, Vec<i32>, BuildHasherDefault<IdHasher>>,
}

impl Buckets {
    /// emplace_hint(last, h, v): last is the index in the group of the last
    /// visited element, None for end()
    fn emplace_hint(&mut self, h: u64, last: Option<usize>, v: i32) {
        let g = self.groups.entry(h).or_default();
        match last {
            Some(k) => g.insert(if cfg!(feature = "libstdcxx") { k + 1 } else { k }, v),
            None => g.push(v),
        }
    }
    fn erase(&mut self, h: u64, k: usize) {
        if let Some(g) = self.groups.get_mut(&h) {
            g.remove(k);
        }
    }
}

impl Presolve<'_> {
    pub(crate) fn detect_parallel_rows_and_cols(&mut self) -> R {
        let logging_on = self.analysis.logging_on;
        if logging_on {
            self.start_rule_log(RULE_PARALLEL_ROWS_AND_COLS);
        }
        let small = self.opt.small_matrix_value;
        let mut row_max: Vec<(f64, i32)> = vec![(0.0, 0); self.rowsize.len()];
        let mut col_max: Vec<(f64, i32)> = vec![(0.0, 0); self.colsize.len()];
        let mut num_row_singletons: Vec<i32> = vec![0; self.rowsize.len()];
        let nnz = self.a_value.len();
        let mut row_hashes: Vec<u64> = self.rowsize.iter().map(|&s| s as i64 as u64).collect();
        let mut col_hashes: Vec<u64> = self.colsize.iter().map(|&s| s as i64 as u64).collect();

        for i in 0..nnz {
            if self.a_value[i] == 0.0 {
                continue;
            }
            let ac = self.a_col[i] as usize;
            let ar = self.a_row[i] as usize;
            if self.colsize[ac] == 1 {
                col_max[ac].0 = self.a_value[i];
                row_hashes[ar] = row_hashes[ar].wrapping_sub(1);
                num_row_singletons[ar] += 1;
                continue;
            }
            let abs_val = self.a_value[i].abs();
            let abs_row_max = row_max[ar].0.abs();
            if abs_val >= abs_row_max - small && (abs_val > abs_row_max + small || (ac as i32) < row_max[ar].1) {
                row_max[ar] = (self.a_value[i], ac as i32);
            }
            let abs_col_max = col_max[ac].0.abs();
            if abs_val >= abs_col_max - small && (abs_val > abs_col_max + small || (ar as i32) < col_max[ac].1) {
                col_max[ac] = (self.a_value[i], ar as i32);
            }
        }

        for i in 0..nnz {
            if self.a_value[i] == 0.0 {
                continue;
            }
            let ac = self.a_col[i] as usize;
            let ar = self.a_row[i] as usize;
            if self.colsize[ac] == 1 {
                col_hashes[ac] = self.a_row[i] as i64 as u64;
            } else {
                sparse_combine(
                    &mut row_hashes[ar],
                    self.a_col[i],
                    double_hash_code(self.a_value[i] / row_max[ar].0) as u64,
                );
                sparse_combine(
                    &mut col_hashes[ac],
                    self.a_row[i],
                    double_hash_code(self.a_value[i] / col_max[ac].0) as u64,
                );
            }
        }

        let mut buckets = Buckets::default();
        for i in 0..self.num_col {
            let iu = i as usize;
            if self.col_deleted[iu] != 0 {
                continue;
            }
            if self.colsize[iu] == 0 {
                self.col_presolve(i)?;
                continue;
            }
            let h = col_hashes[iu];
            let group: &[i32] = buckets.groups.get(&h).map_or(&[], |g| g.as_slice());
            let mut last: Option<usize> = None;
            let mut del_col = -1;

            for (k, &cand) in group.iter().enumerate() {
                last = Some(k);
                let cu = cand as usize;
                if self.colsize[iu] != self.colsize[cu] {
                    continue;
                }
                let col;
                let duplicate_col;
                let col_scale;
                let mut check_col_impl_bounds = true;
                let mut check_duplicate_col_impl_bounds = true;

                if self.integrality[iu] == INTEGER && self.integrality[cu] == INTEGER {
                    if col_max[iu].0.abs() < col_max[cu].0.abs() {
                        col = i;
                        duplicate_col = cand;
                    } else {
                        col = cand;
                        duplicate_col = i;
                    }
                    let scale_cand = col_max[duplicate_col as usize].0 / col_max[col as usize].0;
                    let r = scale_cand.round();
                    if (scale_cand - r).abs() > small {
                        continue;
                    }
                    col_scale = r;
                    if col_scale != 1.0 {
                        check_duplicate_col_impl_bounds = false;
                    }
                } else if self.integrality[iu] == INTEGER {
                    col = i;
                    duplicate_col = cand;
                    col_scale = col_max[duplicate_col as usize].0 / col_max[col as usize].0;
                    check_col_impl_bounds = false;
                } else {
                    col = cand;
                    duplicate_col = i;
                    col_scale = col_max[duplicate_col as usize].0 / col_max[col as usize].0;
                    check_col_impl_bounds = self.integrality[cu] != INTEGER;
                }
                let c = col as usize;
                let dc = duplicate_col as usize;
                let lp = self.mip.is_none();

                let col_upper_inf = |s: &Self| -> bool {
                    if !check_col_impl_bounds {
                        return false;
                    }
                    if lp {
                        if col_scale > 0.0 {
                            s.is_upper_strictly_implied(col, None)
                        } else {
                            s.is_lower_strictly_implied(col, None)
                        }
                    } else if col_scale > 0.0 {
                        s.is_upper_implied(col)
                    } else {
                        s.is_lower_implied(col)
                    }
                };
                let col_lower_inf = |s: &Self| -> bool {
                    if !check_col_impl_bounds {
                        return false;
                    }
                    if lp {
                        if col_scale > 0.0 {
                            s.is_lower_strictly_implied(col, None)
                        } else {
                            s.is_upper_strictly_implied(col, None)
                        }
                    } else if col_scale > 0.0 {
                        s.is_lower_implied(col)
                    } else {
                        s.is_upper_implied(col)
                    }
                };
                let dup_upper_inf = |s: &Self| -> bool {
                    if !check_duplicate_col_impl_bounds {
                        return false;
                    }
                    if lp {
                        s.is_upper_strictly_implied(duplicate_col, None)
                    } else {
                        s.is_upper_implied(duplicate_col)
                    }
                };
                let dup_lower_inf = |s: &Self| -> bool {
                    if !check_duplicate_col_impl_bounds {
                        return false;
                    }
                    if lp {
                        s.is_lower_strictly_implied(duplicate_col, None)
                    } else {
                        s.is_lower_implied(duplicate_col)
                    }
                };

                let obj_diff = (self.col_cost[c] * CDouble::from(col_scale) - self.col_cost[dc]).to_f64();
                let dtol = self.opt.dual_feasibility_tolerance;
                let mut reduction_case = MERGE_PARALLEL_COLS;
                if obj_diff < -dtol {
                    if col_upper_inf(self) && self.col_lower[dc] != -INF {
                        reduction_case = DOMINANCE_DUPLICATE_COL_TO_LOWER;
                    } else if dup_lower_inf(self)
                        && (col_scale < 0.0 || self.col_upper[c] != INF)
                        && (col_scale > 0.0 || self.col_lower[c] != -INF)
                    {
                        reduction_case = if col_scale > 0.0 { DOMINANCE_COL_TO_UPPER } else { DOMINANCE_COL_TO_LOWER };
                    } else {
                        continue;
                    }
                } else if obj_diff > dtol {
                    if col_lower_inf(self) && self.col_upper[dc] != INF {
                        reduction_case = DOMINANCE_DUPLICATE_COL_TO_UPPER;
                    } else if dup_upper_inf(self)
                        && (col_scale < 0.0 || self.col_lower[c] != -INF)
                        && (col_scale > 0.0 || self.col_upper[c] != INF)
                    {
                        reduction_case = if col_scale > 0.0 { DOMINANCE_COL_TO_LOWER } else { DOMINANCE_COL_TO_UPPER };
                    } else {
                        continue;
                    }
                } else if col_upper_inf(self) && self.col_lower[dc] != -INF {
                    reduction_case = DOMINANCE_DUPLICATE_COL_TO_LOWER;
                } else if col_lower_inf(self) && self.col_upper[dc] != INF {
                    reduction_case = DOMINANCE_DUPLICATE_COL_TO_UPPER;
                } else if dup_upper_inf(self)
                    && (col_scale < 0.0 || self.col_lower[c] != -INF)
                    && (col_scale > 0.0 || self.col_upper[c] != INF)
                {
                    reduction_case = if col_scale > 0.0 { DOMINANCE_COL_TO_LOWER } else { DOMINANCE_COL_TO_UPPER };
                } else if dup_lower_inf(self)
                    && (col_scale < 0.0 || self.col_upper[c] != INF)
                    && (col_scale > 0.0 || self.col_lower[c] != -INF)
                {
                    reduction_case = if col_scale > 0.0 { DOMINANCE_COL_TO_UPPER } else { DOMINANCE_COL_TO_LOWER };
                }

                if reduction_case == MERGE_PARALLEL_COLS {
                    let x_int = self.integrality[c] == INTEGER;
                    if x_int {
                        let illegal_scale = if self.integrality[dc] != INTEGER {
                            (col_scale * (self.col_upper[dc] - self.col_lower[dc])).abs() < 1.0 - self.primal_feastol
                        } else {
                            let scale_limit = self.col_upper[c] - self.col_lower[c] + 1.0 + self.primal_feastol;
                            col_scale.abs() > scale_limit
                        };
                        if illegal_scale {
                            continue;
                        }
                    }
                }

                let mut parallel = true;
                let mut p = self.colhead[c];
                while p != -1 {
                    let pu = p as usize;
                    let row = self.a_row[pu];
                    let v = self.a_value[pu];
                    p = self.a_next[pu];
                    let dup_pos = self.find_nonzero(row, duplicate_col);
                    parallel = dup_pos != -1;
                    if !parallel {
                        break;
                    }
                    parallel = (self.a_value[dup_pos as usize] - CDouble::from(col_scale) * v).to_f64().abs() <= small;
                    if !parallel {
                        break;
                    }
                }
                if !parallel {
                    continue;
                }

                let dec_singleton = |s: &Self, m: &mut Vec<i32>, x: i32| {
                    if s.colsize[x as usize] == 1 {
                        let row = s.a_row[s.colhead[x as usize] as usize];
                        m[row as usize] -= 1;
                    }
                };
                match reduction_case {
                    DOMINANCE_DUPLICATE_COL_TO_LOWER => {
                        del_col = duplicate_col;
                        dec_singleton(self, &mut num_row_singletons, duplicate_col);
                        self.fix_col_to_lower(duplicate_col)?;
                    }
                    DOMINANCE_DUPLICATE_COL_TO_UPPER => {
                        del_col = duplicate_col;
                        dec_singleton(self, &mut num_row_singletons, duplicate_col);
                        self.fix_col_to_upper(duplicate_col)?;
                    }
                    DOMINANCE_COL_TO_LOWER => {
                        del_col = col;
                        dec_singleton(self, &mut num_row_singletons, col);
                        self.fix_col_to_lower(col)?;
                    }
                    DOMINANCE_COL_TO_UPPER => {
                        del_col = col;
                        dec_singleton(self, &mut num_row_singletons, col);
                        self.fix_col_to_upper(col)?;
                    }
                    _ => {
                        // kMergeParallelCols: the C++ merges even if the
                        // postsolve stack refused the record
                        let _ok_merge = self.ps.duplicate_column(
                            col_scale,
                            self.col_lower[c],
                            self.col_upper[c],
                            self.col_lower[dc],
                            self.col_upper[dc],
                            col,
                            duplicate_col,
                            self.integrality[c] == INTEGER,
                            self.integrality[dc] == INTEGER,
                            self.opt.mip_feasibility_tolerance,
                        );
                        let rowsize_int_reduction =
                            self.integrality[dc] != INTEGER && self.integrality[c] == INTEGER;
                        if rowsize_int_reduction {
                            self.integrality[c] = CONTINUOUS;
                        }
                        self.mark_changed_col(col);
                        dec_singleton(self, &mut num_row_singletons, duplicate_col);
                        let (merge_lower, merge_upper) = if col_scale > 0.0 {
                            (
                                col_scale.mul_add_c(self.col_lower[dc], self.col_lower[c]),
                                col_scale.mul_add_c(self.col_upper[dc], self.col_upper[c]),
                            )
                        } else {
                            (
                                col_scale.mul_add_c(self.col_upper[dc], self.col_lower[c]),
                                col_scale.mul_add_c(self.col_lower[dc], self.col_upper[c]),
                            )
                        };
                        self.change_col_bounds(col, merge_lower, merge_upper)?;
                        self.mark_col_deleted(duplicate_col);
                        let mut coliter = self.colhead[dc];
                        while coliter != -1 {
                            let cp = coliter as usize;
                            let colrow = self.a_row[cp];
                            if rowsize_int_reduction {
                                self.rowsize_integer[colrow as usize] -= 1;
                            }
                            coliter = self.a_next[cp];
                            self.unlink(cp as i32);
                            self.reinsert_equation(colrow);
                        }
                        self.col_cost[dc] = 0.0;
                        del_col = duplicate_col;
                        self.reset_col_implied_bounds(col, -1);
                        if rowsize_int_reduction && self.integrality[dc] == IMPLICIT_INTEGER {
                            let implied_integer = self.is_implied_integer(col)?;
                            if implied_integer {
                                let _ = self.convert_implied_integer(col, -1, true);
                            }
                        }
                    }
                }
                break;
            }

            if del_col != -1 {
                if del_col != i {
                    if let Some(k) = last {
                        buckets.erase(h, k);
                    }
                }
                self.check_limits()?;
                self.remove_row_singletons()?;
            } else {
                buckets.emplace_hint(h, last, i);
            }
        }

        let mut buckets = Buckets::default();
        let lp_basis = self.mip.is_none() && self.opt.lp_presolve_requires_basis_postsolve;
        for i in 0..self.num_row {
            let iu = i as usize;
            if self.row_deleted[iu] != 0 {
                continue;
            }
            if self.rowsize[iu] <= 1 || (self.rowsize[iu] == 2 && self.is_equation(i)) {
                self.row_presolve(i)?;
                continue;
            }
            let h = row_hashes[iu];
            let group: &[i32] = buckets.groups.get(&h).map_or(&[], |g| g.as_slice());
            let mut last: Option<usize> = None;
            let get_num_singletons = |m: &Vec<i32>, row: i32| m[row as usize];
            let num_singleton = get_num_singletons(&num_row_singletons, i);
            if lp_basis && num_singleton != 0 {
                continue;
            }
            let mut del_row = -1;
            if !group.is_empty() {
                self.store_row(i);
            }
            for (k, &cand) in group.iter().enumerate() {
                last = Some(k);
                let cu = cand as usize;
                let num_singleton_candidate = get_num_singletons(&num_row_singletons, cand);
                if lp_basis && num_singleton_candidate != 0 {
                    continue;
                }
                if self.rowsize[iu] - num_singleton != self.rowsize[cu] - num_singleton_candidate {
                    continue;
                }
                if num_singleton_candidate > 1 || num_singleton > 1 {
                    if (num_singleton != 0 || !self.is_equation(i))
                        && (num_singleton_candidate != 0 || !self.is_equation(cand))
                    {
                        continue;
                    }
                } else if num_singleton_candidate != num_singleton && !self.is_equation(i) && !self.is_equation(cand) {
                    continue;
                }

                let row_scale = row_max[cu].0 / row_max[iu].0;
                let mut parallel = true;
                for kk in 0..self.rp_len {
                    let p = self.rowpositions[kk] as usize;
                    let ci = self.a_col[p];
                    let v = self.a_value[p];
                    if self.colsize[ci as usize] == 1 {
                        continue;
                    }
                    let nz_pos = self.find_nonzero(cand, ci);
                    parallel = nz_pos != -1;
                    if !parallel {
                        break;
                    }
                    parallel = (self.a_value[nz_pos as usize] - CDouble::from(row_scale) * v).to_f64().abs() <= small;
                    if !parallel {
                        break;
                    }
                }
                if !parallel {
                    continue;
                }

                if num_singleton == 0 && num_singleton_candidate == 0 {
                    let feastol = self.primal_feastol;
                    let mut row_lower_tightened = false;
                    let mut row_upper_tightened = false;
                    let (mut new_upper, mut new_lower) = if row_scale > 0.0 {
                        (self.row_upper[iu] * row_scale, self.row_lower[iu] * row_scale)
                    } else {
                        (self.row_lower[iu] * row_scale, self.row_upper[iu] * row_scale)
                    };
                    if new_upper < self.row_upper[cu] {
                        if new_upper < self.row_lower[cu] - feastol {
                            return Err(Stop::PrimalInfeasible);
                        }
                        if new_upper <= self.row_lower[cu] + feastol {
                            new_upper = self.row_lower[cu];
                        }
                        if new_upper < self.row_upper[cu] {
                            row_upper_tightened = true;
                            if row_scale > 0.0 {
                                let tmp = self.row_dual_lower[iu] / row_scale;
                                self.row_dual_lower[iu] = self.row_dual_lower[cu] * row_scale;
                                self.row_dual_lower[cu] = tmp;
                            } else {
                                let tmp = self.row_dual_upper[iu] / row_scale;
                                self.row_dual_upper[iu] = self.row_dual_lower[cu] * row_scale;
                                self.row_dual_lower[cu] = tmp;
                            }
                            self.row_upper[cu] = new_upper;
                        }
                    }
                    if new_lower > self.row_lower[cu] {
                        if new_lower > self.row_upper[cu] + feastol {
                            return Err(Stop::PrimalInfeasible);
                        }
                        if new_lower >= self.row_upper[cu] - feastol {
                            new_lower = self.row_upper[cu];
                        }
                        if new_lower > self.row_lower[cu] {
                            row_lower_tightened = true;
                            if row_scale > 0.0 {
                                let tmp = self.row_dual_upper[iu] / row_scale;
                                self.row_dual_upper[iu] = self.row_dual_upper[cu] * row_scale;
                                self.row_dual_upper[cu] = tmp;
                            } else {
                                let tmp = self.row_dual_lower[iu] / row_scale;
                                self.row_dual_lower[iu] = self.row_dual_upper[cu] * row_scale;
                                self.row_dual_upper[cu] = tmp;
                            }
                            self.row_lower[cu] = new_lower;
                        }
                    }
                    self.reset_row_dual_implied_bounds(cand, -1);
                    self.ps.duplicate_row(cand, row_upper_tightened, row_lower_tightened, i, row_scale);
                    del_row = i;
                    self.mark_row_deleted(i);
                    let n = self.rp_len;
                    for kk in 0..n {
                        let p = self.rowpositions[kk];
                        self.unlink(p);
                    }
                    break;
                } else if self.is_equation(i) {
                    let rowvector: Vec<(i32, f64)> = stored_row!(self).collect();
                    self.equality_row_addition(i, cand, -row_scale, &rowvector)?;
                    del_row = cand;
                } else if self.is_equation(cand) {
                    let rowvector: Vec<(i32, f64)> = row_iter!(self, cand).collect();
                    let scale = -row_max[iu].0 / row_max[cu].0;
                    self.equality_row_addition(cand, i, scale, &rowvector)?;
                    del_row = i;
                }
            }

            if del_row != -1 {
                if del_row != i {
                    if let Some(k) = last {
                        buckets.erase(h, k);
                    }
                }
                self.check_limits()?;
            } else {
                buckets.emplace_hint(h, last, i);
            }
        }

        self.analysis.logging_on = logging_on;
        if logging_on {
            self.stop_rule_log(RULE_PARALLEL_ROWS_AND_COLS);
        }
        Ok(())
    }
}
