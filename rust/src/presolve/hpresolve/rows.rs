//! Row reductions: rowPresolve, singletonRow, doubletonEq, the MIP row
//! tightenings (simple probing on equations, integral scaling, coefficient
//! strengthening, Chvatal-Gomory strengthening) and forcing rows.

use super::*;
use crate::mip::cuts::integers::{gcd, integral_scale};
use crate::presolve::postsolve::{EQ, GEQ, LEQ};

/// HighsIntegers::mod for int64
#[inline]
fn imod(a: i64, m: i64) -> i64 {
    let r = a % m;
    r + (r < 0) as i64 * m
}

/// HighsIntegers::mod for double
#[inline]
fn dmod(a: f64, m: f64) -> f64 {
    (a % m).trunc() + (a < 0.0) as i32 as f64 * m
}

/// HighsIntegers::modularInverse
fn modular_inverse(mut a: i64, mut m: i64) -> i64 {
    let mut y = 0i64;
    let mut x = 1i64;
    if m == 1 {
        return 0;
    }
    a = imod(a, m);
    while a > 1 {
        let q = a / m;
        let mut r = a - q * m;
        a = m;
        m = r;
        r = x - q * y;
        x = y;
        y = r;
    }
    x
}

/// computeDynamism over values
pub(crate) fn compute_dynamism(vals: impl Iterator<Item = f64>) -> CDouble {
    let mut min_abs = INF;
    let mut max_abs = -INF;
    for v in vals {
        let a = v.abs();
        min_abs = std_min(min_abs, a);
        max_abs = std_max(max_abs, a);
    }
    CDouble::from(max_abs) / CDouble::from(min_abs)
}

/// std::set<double>::emplace on a sorted vector
fn set_insert(v: &mut Vec<f64>, x: f64) {
    let mut i = 0;
    while i < v.len() && v[i] < x {
        i += 1;
    }
    if i < v.len() && !(x < v[i]) {
        return;
    }
    v.insert(i, x);
}

impl Presolve<'_> {
    pub(crate) fn singleton_row(&mut self, row: i32) -> R {
        let logging_on = self.analysis.logging_on;
        if logging_on {
            self.start_rule_log(RULE_SINGLETON_ROW);
        }
        let r = row as usize;
        let nz_pos = self.rowroot[r];
        let p = nz_pos as usize;
        let col = self.a_col[p];
        let c = col as usize;
        let val = self.a_value[p];

        self.mark_row_deleted(row);
        self.unlink(nz_pos);

        let feastol = self.primal_feastol;
        if val > 0.0 {
            if self.col_upper[c] * val <= self.row_upper[r] + feastol
                && self.col_lower[c] * val >= self.row_lower[r] - feastol
            {
                self.ps.redundant_row(row);
                self.analysis.logging_on = logging_on;
                if logging_on {
                    self.stop_rule_log(RULE_SINGLETON_ROW);
                }
                return self.check_limits();
            }
        } else if self.col_lower[c] * val <= self.row_upper[r] + feastol
            && self.col_upper[c] * val >= self.row_lower[r] - feastol
        {
            self.ps.redundant_row(row);
            self.analysis.logging_on = logging_on;
            if logging_on {
                self.stop_rule_log(RULE_SINGLETON_ROW);
            }
            return self.check_limits();
        }

        let mut new_col_upper = INF;
        let mut new_col_lower = -INF;
        if val > 0.0 {
            if self.row_upper[r] != INF {
                new_col_upper = self.row_upper[r] / val;
            }
            if self.row_lower[r] != -INF {
                new_col_lower = self.row_lower[r] / val;
            }
        } else {
            if self.row_upper[r] != INF {
                new_col_lower = self.row_upper[r] / val;
            }
            if self.row_lower[r] != -INF {
                new_col_upper = self.row_lower[r] / val;
            }
        }

        let bound_tol = feastol / std_max(1.0, val.abs());
        let is_integral = self.integrality[c] != CONTINUOUS;
        let mut lower_tightened = new_col_lower > self.col_lower[c] + bound_tol;
        let mut upper_tightened = new_col_upper < self.col_upper[c] - bound_tol;

        let mut lb;
        let mut ub;
        if lower_tightened {
            if is_integral {
                new_col_lower = (new_col_lower - bound_tol).ceil();
            }
            lb = new_col_lower;
        } else {
            lb = self.col_lower[c];
        }
        if upper_tightened {
            if is_integral {
                new_col_upper = (new_col_upper + bound_tol).floor();
            }
            ub = new_col_upper;
        } else {
            ub = self.col_upper[c];
        }

        if ub <= lb + feastol {
            if ub < lb - feastol {
                return Err(Stop::PrimalInfeasible);
            }
            if ub < lb || (ub > lb && (ub - lb) * std_max(val.abs(), self.get_max_abs_col_val(col)) <= feastol) {
                if lower_tightened && upper_tightened {
                    ub = 0.5 * (ub + lb);
                    lb = ub;
                    lower_tightened = lb > self.col_lower[c];
                    upper_tightened = ub < self.col_upper[c];
                } else if lower_tightened {
                    lb = ub;
                    lower_tightened = lb > self.col_lower[c];
                } else {
                    ub = lb;
                    upper_tightened = ub < self.col_upper[c];
                }
            }
        }

        self.ps.singleton_row(row, col, val, lower_tightened, upper_tightened);

        if lower_tightened {
            self.change_col_lower(col, lb)?;
        }
        if ub == lb {
            self.ps.removed_fixed_col(col, lb, self.col_cost[c], col_iter!(self, col));
            self.remove_fixed_col_logged(col);
        } else if upper_tightened {
            self.change_col_upper(col, ub)?;
        }

        if self.col_deleted[c] == 0 && self.colsize[c] == 0 {
            let result = self.empty_col(col);
            self.analysis.logging_on = logging_on;
            if logging_on {
                self.stop_rule_log(RULE_SINGLETON_ROW);
            }
            return result;
        }
        self.analysis.logging_on = logging_on;
        if logging_on {
            self.stop_rule_log(RULE_SINGLETON_ROW);
        }
        self.check_limits()
    }

    pub(crate) fn doubleton_eq(&mut self, row: i32, row_type: i32) -> R {
        let logging_on = self.analysis.logging_on;
        if logging_on {
            self.start_rule_log(RULE_DOUBLETON_EQUATION);
        }
        let r = row as usize;
        let nz_pos1 = self.rowroot[r] as usize;
        let nz_pos2 = if self.ar_right[nz_pos1] != -1 { self.ar_right[nz_pos1] } else { self.ar_left[nz_pos1] } as usize;

        let small = self.opt.small_matrix_value;
        let col_at_pos1_better = {
            let c1 = self.a_col[nz_pos1] as usize;
            let c2 = self.a_col[nz_pos2] as usize;
            if self.integrality[c1] == INTEGER {
                if self.integrality[c2] == INTEGER {
                    if self.a_value[nz_pos1].abs() < self.a_value[nz_pos2].abs() - small {
                        true
                    } else if self.a_value[nz_pos2].abs() < self.a_value[nz_pos1].abs() - small {
                        false
                    } else {
                        self.colsize[c1] < self.colsize[c2]
                    }
                } else {
                    false
                }
            } else if self.integrality[c2] == INTEGER {
                true
            } else {
                let col1_size = self.colsize[c1];
                if col1_size == 1 {
                    true
                } else {
                    let col2_size = self.colsize[c2];
                    if col2_size == 1 {
                        false
                    } else {
                        let abs1 = self.a_value[nz_pos1].abs();
                        let abs2 = self.a_value[nz_pos2].abs();
                        if col1_size != col2_size && std_max(abs1, abs2) <= 2.0 * std_min(abs1, abs2) {
                            col1_size < col2_size
                        } else {
                            abs1 > abs2
                        }
                    }
                }
            }
        };

        let (substcol, staycol, substcoef, mut staycoef) = if col_at_pos1_better {
            (self.a_col[nz_pos1], self.a_col[nz_pos2], self.a_value[nz_pos1], self.a_value[nz_pos2])
        } else {
            (self.a_col[nz_pos2], self.a_col[nz_pos1], self.a_value[nz_pos2], self.a_value[nz_pos1])
        };
        let sc = substcol as usize;
        let st = staycol as usize;

        let mut rhs = self.row_upper[r];
        if self.integrality[sc] == INTEGER && self.integrality[st] == INTEGER {
            let round_coef = (staycoef / substcoef).round() * substcoef;
            if (round_coef - staycoef).abs() > small {
                return Ok(());
            }
            staycoef = round_coef;
            let round_rhs = (rhs / substcoef).round() * substcoef;
            if (rhs - round_rhs).abs() > self.primal_feastol {
                return Err(Stop::PrimalInfeasible);
            }
            rhs = round_rhs;
        }

        let old_stay_lower = self.col_lower[st];
        let old_stay_upper = self.col_upper[st];
        let subst_lower = self.col_lower[sc];
        let subst_upper = self.col_upper[sc];

        let implied = |bound: f64| ((CDouble::from(rhs) - substcoef * bound) / staycoef).to_f64();
        let (stay_impl_lower, stay_impl_upper) = if substcoef.is_sign_negative() != staycoef.is_sign_negative() {
            (
                if subst_lower == -INF { -INF } else { implied(subst_lower) },
                if subst_upper == INF { INF } else { implied(subst_upper) },
            )
        } else {
            (
                if subst_upper == INF { -INF } else { implied(subst_upper) },
                if subst_lower == -INF { INF } else { implied(subst_lower) },
            )
        };

        let lower_tightened = stay_impl_lower > old_stay_lower + self.primal_feastol;
        if lower_tightened {
            self.change_col_lower(staycol, stay_impl_lower)?;
        }
        let upper_tightened = stay_impl_upper < old_stay_upper - self.primal_feastol;
        if upper_tightened {
            self.change_col_upper(staycol, stay_impl_upper)?;
        }

        self.ps.doubleton_equation(
            row,
            substcol,
            staycol,
            substcoef,
            staycoef,
            rhs,
            subst_lower,
            subst_upper,
            self.col_cost[sc],
            lower_tightened,
            upper_tightened,
            row_type,
            col_iter!(self, substcol),
        );

        self.mark_col_deleted(substcol);
        self.remove_row(row);
        self.substitute_cols(substcol, staycol, rhs / substcoef, -staycoef / substcoef);

        self.analysis.logging_on = logging_on;
        if logging_on {
            self.stop_rule_log(RULE_DOUBLETON_EQUATION);
        }
        self.remove_row_singletons()?;
        self.check_limits()
    }

    pub(crate) fn row_presolve(&mut self, row: i32) -> R {
        let r = row as usize;
        let logging_on = self.analysis.logging_on;
        let feastol = self.primal_feastol;
        let small = self.opt.small_matrix_value;

        // checkRowInfeasible
        if self.implied_row_bounds.sum_lower(row) > self.row_upper[r] + feastol
            || self.implied_row_bounds.sum_upper(row) < self.row_lower[r] - feastol
        {
            return Err(Stop::PrimalInfeasible);
        }

        if self.rowsize[r] == 1 {
            return self.singleton_row(row);
        }

        self.check_row_redundant(row, logging_on)?;
        if self.row_deleted[r] != 0 {
            return Ok(());
        }

        if self.rowsize_integer[r] != 0 || self.rowsize_impl_int[r] != 0 {
            let mut w = PreOrder::new(self.rowroot[r]);
            while let Some(p) = w.next(&self.ar_left, &self.ar_right) {
                let col = self.a_col[p];
                let c = col as usize;
                let val = self.a_value[p];
                if self.integrality[c] == CONTINUOUS || self.col_upper[c] != self.col_lower[c] + 1.0 {
                    continue;
                }
                let compute_offset =
                    |s: &Self| val.abs() * (CDouble::from(s.col_upper[c]) - CDouble::from(s.col_lower[c]));
                // degree1Tests(col, val, 1, sumUpperOrig - offset, rowLower)
                let off = compute_offset(self);
                let rab = self.implied_row_bounds.sum_upper_orig_off(row, -off);
                self.degree1_test(col, val, 1, rab, self.row_lower[r])?;
                let off = compute_offset(self);
                let rab = self.implied_row_bounds.sum_lower_orig_off(row, off);
                self.degree1_test(col, val, -1, rab, self.row_upper[r])?;
            }
        }

        self.check_row_redundant(row, logging_on)?;
        if self.row_deleted[r] != 0 {
            return Ok(());
        }

        let orig_row_upper = self.row_upper[r];
        let orig_row_lower = self.row_lower[r];

        if !self.is_equation(row) {
            if self.is_implied_equation_at_lower(row) {
                self.row_upper[r] = self.row_lower[r];
                self.change_row_dual_lower(row, -INF);
                if self.mip.is_none() {
                    let src = self.row_dual_lower_source[r];
                    self.check_redundant_bounds(src, row)?;
                }
            } else if self.is_implied_equation_at_upper(row) {
                self.row_lower[r] = self.row_upper[r];
                self.change_row_dual_upper(row, INF);
                if self.mip.is_none() {
                    let src = self.row_dual_upper_source[r];
                    self.check_redundant_bounds(src, row)?;
                }
            }
        }

        let row_upper = self.row_upper[r];
        let row_lower = self.row_lower[r];

        if self.rowsize[r] == 2 && row_lower == row_upper && self.analysis.allow_rule[RULE_DOUBLETON_EQUATION] {
            let row_type = if orig_row_lower == orig_row_upper {
                EQ
            } else if orig_row_upper != INF {
                LEQ
            } else {
                GEQ
            };
            return self.doubleton_eq(row, row_type);
        }

        if self.rowsize_integer[r] != 0 || self.rowsize_impl_int[r] != 0 {
            if row_lower == row_upper {
                let implied_row_lower = self.implied_row_bounds.sum_lower(row);
                let implied_row_upper = self.implied_row_bounds.sum_upper(row);
                if implied_row_lower != -INF
                    && implied_row_upper != INF
                    && (implied_row_lower + implied_row_upper - 2.0 * row_upper).abs() <= small
                {
                    let mut bin_col = -1;
                    let mut bin_coef = (implied_row_upper - row_upper).abs();
                    self.store_row(row);
                    for k in 0..self.rp_len {
                        let p = self.rowpositions[k] as usize;
                        let ci = self.a_col[p];
                        let cu = ci as usize;
                        let v = self.a_value[p];
                        if (v.abs() - bin_coef).abs() <= small
                            && self.integrality[cu] == INTEGER
                            && (self.col_upper[cu] - self.col_lower[cu] - 1.0).abs() <= feastol
                        {
                            bin_col = ci;
                            bin_coef = v;
                            break;
                        }
                    }

                    if bin_col != -1 {
                        let n = self.rp_len;
                        for k in 0..n {
                            let rowiter = self.rowpositions[k] as usize;
                            let col = self.a_col[rowiter];
                            if col == bin_col {
                                continue;
                            }
                            let c = col as usize;
                            let b = rb!(self);
                            let col_lower = b.impl_var_lower(row, c);
                            let col_upper = b.impl_var_upper(row, c);
                            if self.col_lower[c] == self.col_upper[c] {
                                self.ps.removed_fixed_col(col, self.col_lower[c], 0.0, std::iter::empty());
                                self.remove_fixed_col_logged(col);
                                continue;
                            }
                            let direction: i32 = if bin_coef.is_sign_negative() == self.a_value[rowiter].is_sign_negative() {
                                1
                            } else {
                                -1
                            };
                            // remDoubletonEq
                            let bound = if direction >= 0 { col_upper } else { col_lower };
                            let scale = direction as f64 * (col_lower - col_upper);
                            let offset = (-self.col_lower[bin_col as usize]).mul_add_c(scale, bound);
                            self.ps.doubleton_equation(
                                -1,
                                col,
                                bin_col,
                                1.0,
                                -scale,
                                offset,
                                col_lower,
                                col_upper,
                                0.0,
                                false,
                                false,
                                EQ,
                                std::iter::empty(),
                            );
                            self.substitute_cols(col, bin_col, offset, scale);
                        }
                        self.remove_row(row);
                        self.check_limits()?;
                        return self.remove_row_singletons();
                    }
                }

                if self.rowsize_integer[r] + self.rowsize_impl_int[r] >= self.rowsize[r] - 1 {
                    let mut continuous_col = -1;
                    let mut continuous_coef = 0.0;
                    let mut row_coefs_int: Vec<f64> = Vec::with_capacity(self.rowsize[r] as usize);
                    self.store_row(row);
                    for k in 0..self.rp_len {
                        let p = self.rowpositions[k] as usize;
                        if self.integrality[self.a_col[p] as usize] == CONTINUOUS {
                            continuous_coef = self.a_value[p];
                            continuous_col = self.a_col[p];
                            continue;
                        }
                        row_coefs_int.push(self.a_value[p]);
                    }

                    if continuous_coef != 0.0 {
                        row_coefs_int.push(row_upper);
                        let int_scale = integral_scale(&row_coefs_int, small, small);
                        if int_scale != 0.0 && int_scale <= 1e3 {
                            let scale = 1.0 / (continuous_coef * int_scale).abs();
                            if scale != 1.0 {
                                self.transform_column(continuous_col, scale, 0.0)?;
                                self.convert_implied_integer(continuous_col, -1, true)?;
                                if int_scale != 1.0 {
                                    self.scale_stored_row(row, int_scale, true);
                                }
                            }
                        }
                    } else {
                        let mut int_scale = integral_scale(&row_coefs_int, small, small);
                        if int_scale != 0.0 && int_scale <= 1e3 {
                            let mut rhs = row_upper * int_scale;
                            if fractionality(rhs) > feastol {
                                return Err(Stop::PrimalInfeasible);
                            }
                            rhs = rhs.round();

                            let mut x1_cand: i32 = -1;
                            let mut d: i64 = 0;
                            for i in 0..self.rp_len {
                                let v = self.a_value[self.rowpositions[i] as usize];
                                let newgcd = if d == 0 {
                                    (int_scale * v).round().abs() as i64
                                } else {
                                    gcd((int_scale * v).round().abs() as i64, d)
                                };
                                if newgcd == 1 {
                                    if x1_cand != -1 {
                                        x1_cand = -1;
                                        break;
                                    }
                                    x1_cand = i as i32;
                                } else {
                                    d = newgcd;
                                }
                            }

                            if x1_cand != -1 {
                                let x1_pos = self.rowpositions[x1_cand as usize];
                                let x1 = self.a_col[x1_pos as usize];
                                let x1u = x1 as usize;
                                let rhs2 = rhs / d as f64;
                                if fractionality(rhs2) <= self.mip.map_or(0.0, |m| m.epsilon) {
                                    self.transform_column(x1, d as f64, 0.0)?;
                                } else {
                                    let mut a1 = (int_scale * self.a_value[x1_pos as usize]).round() as i64;
                                    a1 = imod(a1, d);
                                    let a1_inverse = modular_inverse(a1, d);
                                    let b = dmod(a1_inverse as f64 * rhs, d as f64);
                                    let z_lower = ((self.col_lower[x1u] - b) / d as f64 - feastol).ceil();
                                    let z_upper = ((self.col_upper[x1u] - b) / d as f64 + feastol).floor();
                                    if z_lower == z_upper {
                                        let fix_val = z_lower.mul_add_c(d as f64, b);
                                        self.change_col_bounds(x1, fix_val, fix_val)?;
                                        self.ps.removed_fixed_col(x1, fix_val, self.col_cost[x1u], col_iter!(self, x1));
                                        self.remove_fixed_col_logged(x1);
                                        // rowpositions.erase(begin + x1Cand)
                                        let k = x1_cand as usize;
                                        self.rowpositions.copy_within(k + 1..self.rp_len, k);
                                        self.rp_len -= 1;
                                    } else {
                                        self.transform_column(x1, d as f64, b)?;
                                    }
                                }
                                int_scale /= d as f64;
                            }

                            if int_scale != 1.0 {
                                self.scale_stored_row(row, int_scale, true);
                            }
                        }
                    }
                }
            } else {
                self.store_row(row);
                if self.rowsize[r] == self.rowsize_integer[r] + self.rowsize_impl_int[r] {
                    self.integral_row_tightening(row)?;
                }

                self.check_row_redundant(row, logging_on)?;
                if self.row_deleted[r] != 0 {
                    return Ok(());
                }

                if self.row_lower[r] == -INF {
                    let implied_row_upper = self.implied_row_bounds.sum_upper(row);
                    if implied_row_upper != INF {
                        let mut rhs = CDouble::from(self.row_upper[r]);
                        let max_abs = CDouble::from(implied_row_upper) - self.row_upper[r];
                        self.strengthen_coefs(row, &mut rhs, 1, max_abs);
                        self.row_upper[r] = rhs.to_f64();
                    }
                }

                if self.row_upper[r] == INF {
                    let implied_row_lower = self.implied_row_bounds.sum_lower(row);
                    if implied_row_lower != -INF {
                        let mut rhs = CDouble::from(self.row_lower[r]);
                        let max_abs = self.row_lower[r] - CDouble::from(implied_row_lower);
                        self.strengthen_coefs(row, &mut rhs, -1, max_abs);
                        self.row_lower[r] = rhs.to_f64();
                    }
                }
            }
        }

        if self.analysis.allow_rule[RULE_FORCING_ROW] {
            self.store_row(row);
            let dynamism = compute_dynamism(stored_row!(self).map(|(_, v)| v));
            let side = self.row_lower[r];
            let bound = self.implied_row_bounds.sum_upper_orig(row);
            self.check_forcing_row(row, 1, side, bound, dynamism, GEQ, logging_on)?;
            if self.row_deleted[r] != 0 {
                return Ok(());
            }
            let side = self.row_upper[r];
            let bound = self.implied_row_bounds.sum_lower_orig(row);
            self.check_forcing_row(row, -1, side, bound, dynamism, LEQ, logging_on)?;
            if self.row_deleted[r] != 0 {
                return Ok(());
            }
        }

        self.update_col_implied_bounds(row)?;
        self.extract_var_bounds(row);
        self.check_limits()
    }

    fn check_row_redundant(&mut self, row: i32, logging_on: bool) -> R {
        if self.is_redundant(row) {
            let rule = if self.rowsize[row as usize] != 0 { RULE_REDUNDANT_ROW } else { RULE_EMPTY_ROW };
            if logging_on {
                self.start_rule_log(rule);
            }
            self.ps.redundant_row(row);
            self.remove_row(row);
            self.analysis.logging_on = logging_on;
            if logging_on {
                self.stop_rule_log(rule);
            }
            return self.check_limits();
        }
        Ok(())
    }

    fn degree1_test(&mut self, col: i32, val: f64, direction: i32, row_activity_bound: f64, row_bound: f64) -> R {
        let d = direction as f64;
        if d * row_activity_bound >= d.mul_add_c(row_bound, -self.primal_feastol) {
            return Ok(());
        }
        let c = col as usize;
        if d * val > 0.0 {
            self.change_col_lower(col, self.col_lower[c] + 1.0)?;
        } else {
            self.change_col_upper(col, self.col_upper[c] - 1.0)?;
        }
        Ok(())
    }

    fn check_redundant_bounds(&mut self, col: i32, row: i32) -> R {
        let _ = row;
        let c = col as usize;
        if self.colsize[c] != 1 {
            return Ok(());
        }
        if self.col_cost[c] > 0.0 {
            if self.col_lower[c] > self.impl_col_lower[c] - self.primal_feastol {
                self.change_col_lower(col, -INF)?;
            }
        } else if self.col_upper[c] < self.impl_col_upper[c] + self.primal_feastol {
            self.change_col_upper(col, INF)?;
        }
        Ok(())
    }

    /// The inequality branch of rowPresolve for a row of integral columns:
    /// scaling to integral coefficients, or Chvatal-Gomory strengthening.
    /// The row is stored.
    fn integral_row_tightening(&mut self, row: i32) -> R {
        let r = row as usize;
        let feastol = self.primal_feastol;
        let small = self.opt.small_matrix_value;
        let mut row_coefs: Vec<f64> = Vec::with_capacity(self.rowsize[r] as usize);
        let mut row_index: Vec<i32> = Vec::with_capacity(self.rowsize[r] as usize);
        for (ci, v) in stored_row!(self) {
            row_coefs.push(v);
            row_index.push(ci);
        }

        let int_scale = integral_scale(
            &row_coefs,
            if self.row_lower[r] == -INF { feastol } else { small },
            if self.row_upper[r] == INF { feastol } else { small },
        );

        // roundRhs
        let round_rhs = |rhs: CDouble, min_tightening: f64, direction: i32| -> (CDouble, CDouble, bool) {
            let d = direction as f64;
            let rounded = d * (d * rhs + feastol).floor();
            let fraction = d * (rhs - rounded);
            let tightened = fraction >= min_tightening - small;
            (rounded, fraction, tightened)
        };

        if int_scale != 0.0 {
            // checkScaleRow
            let mut lhs = CDouble::from(self.row_lower[r]);
            let mut rhs = CDouble::from(self.row_upper[r]);
            let lhs_finite = lhs != -INF;
            let rhs_finite = rhs != INF;
            if lhs_finite {
                lhs = lhs * int_scale;
            }
            if rhs_finite {
                rhs = rhs * int_scale;
            }
            let mut rounded_lhs = CDouble::from(-INF);
            let mut rounded_rhs = CDouble::from(INF);
            let mut fraction_lhs = CDouble::from(0.0);
            let mut fraction_rhs = CDouble::from(0.0);
            let mut min_rhs_tightening = 0.0f64;
            let mut min_lhs_tightening = 0.0f64;
            let mut max_val = 0.0f64;
            let mut lhs_tightened = false;
            let mut rhs_tightened = false;
            let mut ok = true;
            for i in 0..row_coefs.len() {
                let scale_coef = CDouble::from(row_coefs[i]) * int_scale;
                let int_coef = (scale_coef + 0.5).floor();
                let coef_delta = int_coef - scale_coef;
                row_coefs[i] = int_coef.to_f64();
                max_val = std_max(row_coefs[i].abs(), max_val);
                let ub = self.col_upper[row_index[i] as usize];
                if coef_delta < -small {
                    if lhs_finite {
                        if ub == INF {
                            ok = false;
                            break;
                        }
                        lhs += ub * coef_delta;
                    }
                    min_rhs_tightening = std_max(-coef_delta.to_f64(), min_rhs_tightening);
                } else if coef_delta > small {
                    if rhs_finite {
                        if ub == INF {
                            ok = false;
                            break;
                        }
                        rhs += ub * coef_delta;
                    }
                    min_lhs_tightening = std_max(coef_delta.to_f64(), min_lhs_tightening);
                }
            }
            if ok {
                if lhs_finite {
                    (rounded_lhs, fraction_lhs, lhs_tightened) = round_rhs(lhs, min_lhs_tightening, -1);
                }
                if rhs_finite {
                    (rounded_rhs, fraction_rhs, rhs_tightened) = round_rhs(rhs, min_rhs_tightening, 1);
                }
                let is_infeasible = lhs_finite && rhs_finite && rounded_rhs < rounded_lhs - 0.5;
                if is_infeasible {
                    return Err(Stop::PrimalInfeasible);
                }
                let ranged_or_equation_row = lhs_tightened && rhs_tightened;
                if ranged_or_equation_row
                    || (lhs_tightened && self.row_upper[r] == INF)
                    || (rhs_tightened && self.row_lower[r] == -INF)
                {
                    // scaleRowIntVals
                    let scaled_int = if max_val > 1000.0 && int_scale > 100.0 {
                        false
                    } else {
                        self.scale_row_rounded(row, rounded_lhs, rounded_rhs, 1.0, false, &row_coefs, &row_index);
                        true
                    };
                    if !scaled_int {
                        if ranged_or_equation_row {
                            rounded_lhs /= int_scale;
                            rounded_rhs /= int_scale;
                            if rounded_rhs < self.row_upper[r] - feastol {
                                self.row_upper[r] = rounded_rhs.to_f64();
                            }
                            if rounded_lhs > self.row_lower[r] + feastol {
                                self.row_lower[r] = rounded_lhs.to_f64();
                            }
                        } else if (rhs_tightened && fraction_rhs < min_rhs_tightening - feastol)
                            || (lhs_tightened && fraction_lhs < min_lhs_tightening - feastol)
                        {
                            self.scale_row_rounded(row, rounded_lhs, rounded_rhs, int_scale, true, &row_coefs, &row_index);
                        }
                    }
                }
            }
        } else if !self.is_ranged(row) {
            self.chvatal_gomory(row, &row_coefs, &row_index);
        }
        Ok(())
    }

    /// the scaleRow lambda: replaces the row by rowCoefs / scalar with the
    /// rounded sides
    #[allow(clippy::too_many_arguments)]
    fn scale_row_rounded(
        &mut self,
        row: i32,
        rounded_lhs: CDouble,
        rounded_rhs: CDouble,
        scalar: f64,
        check_delta: bool,
        row_coefs: &[f64],
        row_index: &[i32],
    ) {
        let r = row as usize;
        if rounded_lhs != -INF {
            self.row_lower[r] = (rounded_lhs / scalar).to_f64();
        }
        if rounded_rhs != INF {
            self.row_upper[r] = (rounded_rhs / scalar).to_f64();
        }
        for i in 0..row_coefs.len() {
            let delta = (CDouble::from(row_coefs[i]) / scalar - self.a_value[self.rowpositions[i] as usize]).to_f64();
            if !check_delta || delta.abs() > self.opt.small_matrix_value {
                self.add_to_matrix(row, row_index[i], delta);
            }
        }
    }

    /// Chvatal-Gomory strengthening of an inequality of integral columns
    fn chvatal_gomory(&mut self, row: i32, row_coefs: &[f64], row_index: &[i32]) {
        let r = row as usize;
        let feastol = self.primal_feastol;
        let small = self.opt.small_matrix_value;
        const MAX_DYNAMISM: f64 = 1e5;
        let mut rounded_row_coefs = vec![0.0f64; self.rowsize[r] as usize];
        let mut scalars: Vec<f64> = vec![1.0];

        let direction: i32 = if self.row_upper[r] != INF { -1 } else { 1 };
        let mut rhs: CDouble =
            if direction < 0 { CDouble::from(-self.row_upper[r]) } else { CDouble::from(self.row_lower[r]) };
        let mut rounded_rhs = CDouble::from(0.0);

        // complementOrShift
        let mut min_abs_coef = INF;
        let mut max_abs_coef = 0.0f64;
        for i in 0..row_coefs.len() {
            let c = row_index[i] as usize;
            let val = direction as f64 * row_coefs[i];
            let absval = val.abs();
            min_abs_coef = std_min(min_abs_coef, absval);
            max_abs_coef = std_max(max_abs_coef, absval);
            if val < 0.0 && self.col_upper[c] != INF {
                rhs -= val * CDouble::from(self.col_upper[c]);
            } else if val > 0.0 && self.col_lower[c] != -INF {
                rhs -= val * CDouble::from(self.col_lower[c]);
            } else {
                return;
            }
        }
        let dynamism = max_abs_coef / min_abs_coef;

        if dynamism <= MAX_DYNAMISM {
            for t in 1..=5 {
                set_insert(&mut scalars, t as f64 / max_abs_coef);
                set_insert(&mut scalars, t as f64 / min_abs_coef);
                set_insert(&mut scalars, (2 * t - 1) as f64 / (2.0 * min_abs_coef));
            }
        }

        // rowCanBeTightened
        let mut found = false;
        for &s in &scalars {
            // roundRow
            let mut accept = false;
            let scalar = CDouble::from(s);
            rounded_rhs = (rhs * scalar - feastol).ceil();
            if rounded_rhs <= feastol {
                continue;
            }
            let rhs_ratio = rhs / rounded_rhs;
            let mut weaker = false;
            for i in 0..row_coefs.len() {
                let abs_coef = row_coefs[i].abs();
                rounded_row_coefs[i] = (abs_coef * scalar - TINY).ceil().to_f64();
                let threshold = (rounded_row_coefs[i] * rhs_ratio).to_f64();
                if abs_coef < threshold - small {
                    weaker = true;
                    break;
                }
                accept = accept || (abs_coef > threshold + small);
            }
            if !weaker && accept {
                found = true;
                break;
            }
        }
        if !found {
            return;
        }

        // undoComplementOrShift
        for i in 0..row_coefs.len() {
            let c = row_index[i] as usize;
            let val = direction as f64 * row_coefs[i];
            if val < 0.0 && self.col_upper[c] != INF {
                rounded_row_coefs[i] = -rounded_row_coefs[i];
                rounded_rhs += rounded_row_coefs[i] * CDouble::from(self.col_upper[c]);
            } else if val > 0.0 && self.col_lower[c] != -INF {
                rounded_rhs += rounded_row_coefs[i] * CDouble::from(self.col_lower[c]);
            }
            rounded_row_coefs[i] *= direction as f64;
        }
        rounded_rhs *= direction as f64;

        // updateRow
        if direction < 0 {
            self.row_upper[r] = rounded_rhs.to_f64();
        } else {
            self.row_lower[r] = rounded_rhs.to_f64();
        }
        for i in 0..row_coefs.len() {
            let delta = (CDouble::from(rounded_row_coefs[i]) - row_coefs[i]).to_f64();
            if delta.abs() > small {
                self.add_to_matrix(row, row_index[i], delta);
            }
        }
    }

    /// the strengthenCoefs lambda (the row is stored)
    fn strengthen_coefs(&mut self, row: i32, rhs: &mut CDouble, direction: i32, max_abs_coef_value: CDouble) {
        let feastol = self.primal_feastol;
        let d = direction as f64;
        let n = self.rp_len;
        for k in 0..n {
            let rowiter = self.rowpositions[k] as usize;
            let col = self.a_col[rowiter];
            let c = col as usize;
            let val = d * self.a_value[rowiter];
            let b = rb!(self);
            let col_lower = b.impl_var_lower(row, c);
            let col_upper = b.impl_var_upper(row, c);
            if self.integrality[c] == CONTINUOUS {
                continue;
            }
            if val > (max_abs_coef_value + feastol).to_f64() {
                let delta = d * (max_abs_coef_value - val);
                self.add_to_matrix(row, col, delta.to_f64());
                *rhs += delta * col_upper;
            } else if val < (-max_abs_coef_value - feastol).to_f64() {
                let delta = (-direction) as f64 * (max_abs_coef_value + val);
                self.add_to_matrix(row, col, delta.to_f64());
                *rhs += delta * col_lower;
            }
        }
    }

    /// the checkForcingRow lambda (the row is stored)
    #[allow(clippy::too_many_arguments)]
    fn check_forcing_row(
        &mut self,
        row: i32,
        direction: i32,
        row_side: f64,
        implied_row_bound: f64,
        dynamism: CDouble,
        row_type: i32,
        logging_on: bool,
    ) -> R {
        let d = direction as f64;
        if d * row_side == -INF
            || d * implied_row_bound == INF
            || (CDouble::from(row_side) - CDouble::from(implied_row_bound)).abs()
                > CDouble::from(self.primal_feastol) / dynamism
        {
            return Ok(());
        }
        let n = self.rp_len;
        let mut nfixings = 0;
        for k in 0..n {
            let p = self.rowpositions[k] as usize;
            let c = self.a_col[p] as usize;
            if d * self.a_value[p] > 0.0 {
                if self.col_upper[c] <= self.impl_col_upper[c] {
                    nfixings += 1;
                }
            } else if self.col_lower[c] >= self.impl_col_lower[c] {
                nfixings += 1;
            }
        }
        if nfixings != self.rowsize[row as usize] {
            return Ok(());
        }
        if logging_on {
            self.start_rule_log(RULE_FORCING_ROW);
        }
        self.ps.forcing_row(row, stored_row!(self), row_side, row_type);
        self.mark_row_deleted(row);
        for k in 0..n {
            let p = self.rowpositions[k] as usize;
            let col = self.a_col[p];
            if d * self.a_value[p] > 0.0 {
                self.fix_col_to_upper(col)?;
            } else {
                self.fix_col_to_lower(col)?;
            }
        }
        self.ps.redundant_row(row);
        self.remove_row_singletons()?;
        self.analysis.logging_on = logging_on;
        if logging_on {
            self.stop_rule_log(RULE_FORCING_ROW);
        }
        self.check_limits()
    }

    pub(crate) fn remove_row_singletons(&mut self) -> R {
        let mut i = 0;
        while i < self.singleton_rows.len() {
            let row = self.singleton_rows[i];
            let r = row as usize;
            i += 1;
            if self.row_deleted[r] != 0 || self.rowsize[r] > 1 {
                continue;
            }
            self.row_presolve(row)?;
        }
        self.singleton_rows.clear();
        Ok(())
    }

    pub(crate) fn presolve_changed_rows(&mut self) -> R {
        let mut changed_rows: Vec<i32> = Vec::with_capacity((self.num_row - self.num_deleted_rows).max(0) as usize);
        std::mem::swap(&mut changed_rows, &mut self.changed_row_indices);
        for &row in &changed_rows {
            if self.row_deleted[row as usize] != 0 {
                continue;
            }
            self.row_presolve(row)?;
            self.changed_row_flag[row as usize] = self.row_deleted[row as usize];
        }
        Ok(())
    }

    pub(crate) fn remove_doubleton_equations(&mut self) -> R {
        let mut eq = self.equations.iter().next().copied();
        while let Some(key) = eq {
            let eqrow = key.1;
            if self.rowsize[eqrow as usize] > 2 {
                return Ok(());
            }
            self.row_presolve(eqrow)?;
            if self.row_deleted[eqrow as usize] != 0 {
                eq = self.equations.iter().next().copied();
            } else {
                use std::ops::Bound::{Excluded, Unbounded};
                eq = self.equations.range((Excluded(key), Unbounded)).next().copied();
            }
        }
        Ok(())
    }
}
