//! Column reductions: colPresolve, empty and singleton columns, (weakly)
//! dominated and forcing columns, dual fixing, singleton column stuffing,
//! and the column bounds implied by the rows.

use super::*;
use crate::mip::cuts::sort::pdqsort;

impl Presolve<'_> {
    pub(crate) fn empty_col(&mut self, col: i32) -> R {
        let logging_on = self.analysis.logging_on;
        if logging_on {
            self.start_rule_log(RULE_EMPTY_COL);
        }
        let c = col as usize;
        if (self.col_cost[c] > 0.0 && self.col_lower[c] == -INF) || (self.col_cost[c] < 0.0 && self.col_upper[c] == INF) {
            if self.col_cost[c].abs() <= self.opt.dual_feasibility_tolerance {
                self.col_cost[c] = 0.0;
            } else {
                return Err(Stop::DualInfeasible);
            }
        }
        if self.col_cost[c] > 0.0 {
            self.fix_col_to_lower(col)?;
        } else if self.col_cost[c] < 0.0 || self.col_upper[c].abs() < self.col_lower[c].abs() {
            self.fix_col_to_upper(col)?;
        } else if self.col_lower[c] != -INF {
            self.fix_col_to_lower(col)?;
        } else {
            self.fix_col_to_zero(col);
        }
        self.analysis.logging_on = logging_on;
        if logging_on {
            self.stop_rule_log(RULE_EMPTY_COL);
        }
        self.check_limits()
    }

    pub(crate) fn col_presolve(&mut self, col: i32) -> R {
        let c = col as usize;
        let mut is_fixed = false;
        self.check_col_bounds(col, Some(&mut is_fixed))?;
        if is_fixed {
            self.ps.removed_fixed_col(col, self.col_lower[c], self.col_cost[c], col_iter!(self, col));
            self.remove_fixed_col_logged(col);
            return self.check_limits();
        }
        match self.colsize[c] {
            0 => return self.empty_col(col),
            1 => return self.singleton_col(col),
            _ => {}
        }

        self.detect_dominated_col(col, true)?;
        if self.col_deleted[c] != 0 {
            return Ok(());
        }

        if self.mip.is_some() {
            let lower_implied = self.is_lower_implied(col);
            let ninf = self.implied_dual_row_bounds.num_inf_sum_upper_orig(col);
            self.modify_implied_row_dual_bound(col, self.col_lower_source[c], 1, lower_implied, ninf);
            let upper_implied = self.is_upper_implied(col);
            let ninf = self.implied_dual_row_bounds.num_inf_sum_lower_orig(col);
            self.modify_implied_row_dual_bound(col, self.col_upper_source[c], -1, upper_implied, ninf);

            self.convert_implied_integer(col, -1, false)?;

            if self.integrality[c] != CONTINUOUS
                && self.col_lower[c] != 0.0
                && (self.col_lower[c] != -INF || self.col_upper[c] != INF)
                && self.col_upper[c] - self.col_lower[c] > 0.5
            {
                if self.col_upper[c].abs() > self.col_lower[c].abs() {
                    if self.col_lower[c].abs() < 1000.5 {
                        self.transform_column(col, 1.0, self.col_lower[c])?;
                    }
                } else if self.col_upper[c].abs() < 1000.5 {
                    self.transform_column(col, -1.0, self.col_upper[c])?;
                }
            }
        }

        self.dual_fixing(col)?;
        if self.col_deleted[c] != 0 {
            return Ok(());
        }
        self.singleton_col_stuffing(col)?;
        if self.col_deleted[c] != 0 {
            return Ok(());
        }
        if self.integrality[c] != INTEGER {
            self.update_row_dual_implied_bounds(col);
        }
        Ok(())
    }

    fn modify_implied_row_dual_bound(&mut self, col: i32, row: i32, direction: i32, is_bound_implied: bool, num_inf: i32) {
        let c = col as usize;
        if is_bound_implied && row != -1 && num_inf == 1 && direction as f64 * self.col_cost[c] >= 0.0 && !self.is_ranged(row) {
            let nz_pos = self.find_nonzero(row, col);
            let r = row as usize;
            if self.integrality[c] != INTEGER
                || (self.rowsize_integer[r] == self.rowsize[r]
                    && self.row_coefficients_integral(row, 1.0 / self.a_value[nz_pos as usize]))
            {
                if direction as f64 * self.a_value[nz_pos as usize] > 0.0 {
                    self.change_impl_row_dual_lower(row, 0.0, col);
                } else {
                    self.change_impl_row_dual_upper(row, 0.0, col);
                }
            }
        }
    }

    pub(crate) fn singleton_col(&mut self, col: i32) -> R {
        let c = col as usize;
        let nz_pos = self.colhead[c] as usize;
        let row = self.a_row[nz_pos];
        let col_coef = self.a_value[nz_pos];

        if self.rowsize[row as usize] == 1 {
            self.singleton_row(row)?;
            if self.col_deleted[c] == 0 {
                return self.empty_col(col);
            }
            return Ok(());
        }

        self.detect_dominated_col(col, false)?;
        if self.col_deleted[c] != 0 {
            return Ok(());
        }

        if self.mip.is_some() {
            self.convert_implied_integer(col, row, false)?;
        }

        self.dual_fixing(col)?;
        if self.col_deleted[c] != 0 {
            return Ok(());
        }

        self.singleton_col_stuffing(col)?;
        if self.col_deleted[c] != 0 {
            return Ok(());
        }

        self.update_col_implied_bounds_nz(row, col, col_coef)?;

        if self.integrality[c] != INTEGER {
            self.update_row_dual_implied_bounds_nz(row, col, col_coef);
        }

        if self.is_dual_implied_free(row)
            && self.is_implied_free(col)
            && self.analysis.allow_rule[RULE_FREE_COL_SUBSTITUTION]
        {
            if self.integrality[c] == INTEGER {
                let implied_integral = self.is_implied_integral(col)?;
                if !implied_integral {
                    return Ok(());
                }
            }
            let logging_on = self.analysis.logging_on;
            if logging_on {
                self.start_rule_log(RULE_FREE_COL_SUBSTITUTION);
            }
            self.store_row(row);
            self.substitute_free_col(row, col, false);
            self.analysis.logging_on = logging_on;
            if logging_on {
                self.stop_rule_log(RULE_FREE_COL_SUBSTITUTION);
            }
            return self.check_limits();
        }
        Ok(())
    }

    /// substituteFreeCol (the row is stored)
    pub(crate) fn substitute_free_col(&mut self, row: i32, col: i32, relax_row_dual_bounds: bool) {
        let (rhs, row_type) = self.dual_implied_free_get_rhs_and_row_type(row, relax_row_dual_bounds);
        self.ps.free_col_substitution(
            row,
            col,
            rhs,
            self.col_cost[col as usize],
            row_type,
            stored_row!(self),
            col_iter!(self, col),
        );
        self.substitute_row_col(row, col, rhs);
    }

    pub(crate) fn detect_dominated_col(&mut self, col: i32, handle_singleton_rows: bool) -> R {
        let c = col as usize;
        let cost_off = CDouble::from(-self.col_cost[c]);
        let col_dual_upper = -self.implied_dual_row_bounds.sum_lower_off(col, cost_off);
        let col_dual_lower = -self.implied_dual_row_bounds.sum_upper_off(col, cost_off);
        let dynamism = super::rows::compute_dynamism(col_iter!(self, col).map(|(_, v)| v));
        let logging_on = self.analysis.logging_on;

        // dominatedCol
        let dominated_col = |s: &mut Self, dual_bound: f64, bound: f64, direction: i32| -> R {
            let d = direction as f64;
            if d * dual_bound <= s.opt.dual_feasibility_tolerance {
                return Ok(());
            }
            if d * bound == -INF {
                return Err(Stop::DualInfeasible);
            }
            if logging_on {
                s.start_rule_log(RULE_DOMINATED_COL);
            }
            if direction > 0 {
                s.fix_col_to_lower(col)?;
            } else {
                s.fix_col_to_upper(col)?;
            }
            s.analysis.logging_on = logging_on;
            if logging_on {
                s.stop_rule_log(RULE_DOMINATED_COL);
            }
            if handle_singleton_rows {
                s.remove_row_singletons()?;
            }
            s.check_limits()
        };

        let weakly_dominated_col = |s: &mut Self, dual_bound: f64, bound: f64, other_bound: f64, direction: i32| -> R {
            let d = direction as f64;
            if d * dual_bound < -s.opt.dual_feasibility_tolerance {
                return Ok(());
            }
            if d * bound != -INF {
                if logging_on {
                    s.start_rule_log(RULE_DOMINATED_COL);
                }
                if direction > 0 {
                    s.fix_col_to_lower(col)?;
                } else {
                    s.fix_col_to_upper(col)?;
                }
                s.analysis.logging_on = logging_on;
                if logging_on {
                    s.stop_rule_log(RULE_DOMINATED_COL);
                }
                if handle_singleton_rows {
                    s.remove_row_singletons()?;
                }
                return s.check_limits();
            } else if s.analysis.allow_rule[RULE_FORCING_COL] {
                let cost_off = CDouble::from(-s.col_cost[c]);
                let bound_on_col_dual = if direction > 0 {
                    -s.implied_dual_row_bounds.sum_upper_orig_off(col, cost_off)
                } else {
                    -s.implied_dual_row_bounds.sum_lower_orig_off(col, cost_off)
                };
                if CDouble::from(bound_on_col_dual.abs())
                    <= CDouble::from(s.opt.dual_feasibility_tolerance) / dynamism
                {
                    if logging_on {
                        s.start_rule_log(RULE_FORCING_COL);
                    }
                    s.ps.forcing_column(
                        col,
                        col_iter!(s, col),
                        s.col_cost[c],
                        other_bound,
                        direction < 0,
                        s.integrality[c] == INTEGER,
                    );
                    s.mark_col_deleted(col);
                    let mut coliter = s.colhead[c];
                    while coliter != -1 {
                        let p = coliter as usize;
                        let row = s.a_row[p];
                        let rhs =
                            if d * s.a_value[p] > 0.0 { s.row_upper[row as usize] } else { s.row_lower[row as usize] };
                        coliter = s.a_next[p];
                        s.ps.forcing_column_removed_row(col, row, rhs, row_iter!(s, row));
                        s.remove_row(row);
                    }
                    s.analysis.logging_on = logging_on;
                    if logging_on {
                        s.stop_rule_log(RULE_FORCING_COL);
                    }
                    return s.check_limits();
                }
            }
            Ok(())
        };

        let lower = self.col_lower[c];
        dominated_col(self, col_dual_lower, lower, 1)?;
        if self.col_deleted[c] != 0 {
            return Ok(());
        }
        let upper = self.col_upper[c];
        dominated_col(self, col_dual_upper, upper, -1)?;
        if self.col_deleted[c] != 0 {
            return Ok(());
        }
        let (lower, upper) = (self.col_lower[c], self.col_upper[c]);
        weakly_dominated_col(self, col_dual_lower, lower, upper, 1)?;
        if self.col_deleted[c] != 0 {
            return Ok(());
        }
        let (lower, upper) = (self.col_lower[c], self.col_upper[c]);
        weakly_dominated_col(self, col_dual_upper, upper, lower, -1)?;
        Ok(())
    }

    /// computeLocks with the lock counting callback of dualFixing:
    /// (numDownLocks, numUpLocks, downLockRow, upLockRow)
    fn compute_locks(&self, col: i32) -> (i32, i32, i32, i32) {
        let mut num_down = 0;
        let mut num_up = 0;
        let mut down_row = -1;
        let mut up_row = -1;
        let mut cb = |row: i32, has_down: bool, has_up: bool| -> bool {
            if has_up {
                num_up += 1;
                up_row = row;
            }
            if has_down {
                num_down += 1;
                down_row = row;
            }
            num_down > 1 && num_up > 1
        };
        let c = col as usize;
        let mut stop = false;
        if self.col_cost[c] < 0.0 {
            stop = cb(-1, true, false);
        } else if self.col_cost[c] > 0.0 {
            stop = cb(-1, false, true);
        }
        if !stop {
            for (row, val) in col_iter!(self, col) {
                let has_down = self.yields_implied_lower_bound(row, val);
                let has_up = self.yields_implied_upper_bound(row, val);
                if (has_down || has_up) && cb(row, has_down, has_up) {
                    break;
                }
            }
        }
        (num_down, num_up, down_row, up_row)
    }

    pub(crate) fn dual_fixing(&mut self, col: i32) -> R {
        let c = col as usize;
        if self.col_lower[c] == self.col_upper[c] {
            return Ok(());
        }
        let (num_down, num_up, down_row, up_row) = self.compute_locks(col);
        let feastol = self.primal_feastol;

        if num_down == 0 || num_up == 0 {
            if num_down == 0 {
                self.fix_col_to_lower(col)?;
            } else {
                self.fix_col_to_upper(col)?;
            }
        } else {
            let has_single_down = num_down == 1 && down_row != -1;
            let has_single_up = num_up == 1 && up_row != -1;
            if has_single_down || has_single_up {
                let equation_row = if has_single_down && self.is_equation(down_row) {
                    down_row
                } else if has_single_up && self.is_equation(up_row) {
                    up_row
                } else {
                    -1
                };
                if equation_row != -1 && self.single_equation_checked[equation_row as usize] == 0 {
                    self.handle_single_equation(equation_row)?;
                    self.single_equation_checked[equation_row as usize] = 1;
                    if self.col_deleted[c] != 0 {
                        return Ok(());
                    }
                } else if self.mip.is_some() && self.col_lower[c] != -INF && self.col_upper[c] != INF {
                    if has_single_down {
                        let (cu, cl) = (self.col_upper[c], self.col_lower[c]);
                        self.dual_substitute_col(col, down_row, 1, cu, cl)?;
                        if self.col_deleted[c] != 0 {
                            return Ok(());
                        }
                    }
                    if has_single_up {
                        let (cl, cu) = (self.col_lower[c], self.col_upper[c]);
                        self.dual_substitute_col(col, up_row, -1, cl, cu)?;
                        if self.col_deleted[c] != 0 {
                            return Ok(());
                        }
                    }
                }
            }
            // hasTighterBound
            let has_tighter_bound = |s: &mut Self, direction: i32, current_bound: f64| -> Option<f64> {
                let d = direction as f64;
                if d * s.col_cost[c] < 0.0 {
                    return None;
                }
                let huge_bound = feastol / TINY;
                let mut new_bound = if direction > 0 {
                    s.compute_worst_case_lower_bound(col, -1, INF, 0)
                } else {
                    -s.compute_worst_case_upper_bound(col, -1, INF, 0)
                };
                if new_bound != -INF && (new_bound >= d.mul_add_c(current_bound, -feastol) || new_bound.abs() > huge_bound)
                {
                    return None;
                }
                if s.integrality[c] != CONTINUOUS {
                    new_bound = (new_bound - feastol).ceil();
                }
                new_bound *= d;
                Some(new_bound)
            };
            let cu = self.col_upper[c];
            if let Some(nb) = has_tighter_bound(self, 1, cu) {
                let new_bound = std_max(nb, self.col_lower[c]);
                if new_bound < self.col_upper[c] - feastol {
                    if new_bound == self.col_lower[c] {
                        self.fix_col_to_lower(col)?;
                    } else if self.integrality[c] != CONTINUOUS {
                        self.change_col_upper(col, new_bound)?;
                    }
                }
            } else {
                let cl = self.col_lower[c];
                if let Some(nb) = has_tighter_bound(self, -1, cl) {
                    let new_bound = std_min(nb, self.col_upper[c]);
                    if new_bound > self.col_lower[c] + feastol {
                        if new_bound == self.col_upper[c] {
                            self.fix_col_to_upper(col)?;
                        } else if self.integrality[c] != CONTINUOUS {
                            self.change_col_lower(col, new_bound)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// the substituteCol lambda of dualFixing
    fn dual_substitute_col(&mut self, col: i32, row: i32, direction: i32, col_bound: f64, other_col_bound: f64) -> R {
        let r = row as usize;
        let c = col as usize;
        let feastol = self.primal_feastol;
        let lhs_finite = self.row_lower[r] != -INF;
        let rhs_finite = self.row_upper[r] != INF;
        self.store_row(row);
        let n = self.rp_len;
        for k in 0..n {
            let p = self.rowpositions[k] as usize;
            let nzcol = self.a_col[p];
            let nzval = self.a_value[p];
            if nzcol == col {
                continue;
            }
            let nc = nzcol as usize;
            if self.integrality[nc] != INTEGER || self.col_lower[nc] != 0.0 || self.col_upper[nc] != 1.0 {
                continue;
            }
            if (rhs_finite
                && self.implied_row_bounds.residual_sum_upper_orig(row, nzcol, nzval, -1, -INF, -INF, &rb!(self))
                    > self.row_upper[r] + feastol)
                || (lhs_finite
                    && self.implied_row_bounds.residual_sum_lower_orig(row, nzcol, nzval, -1, -INF, -INF, &rb!(self))
                        < self.row_lower[r] - feastol)
            {
                continue;
            }
            let best_bound = if direction > 0 {
                self.compute_implied_lower_bound(col, nzcol, self.col_upper[nc], 0)
            } else {
                -self.compute_implied_upper_bound(col, nzcol, self.col_upper[nc], 0)
            };
            if best_bound >= (direction as f64).mul_add_c(col_bound, -feastol) {
                let offset = other_col_bound;
                let scale = col_bound - other_col_bound;
                self.ps.doubleton_equation(
                    -1,
                    col,
                    nzcol,
                    1.0,
                    -scale,
                    offset,
                    self.col_lower[c],
                    self.col_upper[c],
                    0.0,
                    false,
                    false,
                    crate::presolve::postsolve::EQ,
                    std::iter::empty(),
                );
                self.mark_col_deleted(col);
                self.substitute_cols(col, nzcol, offset, scale);
                self.check_limits()?;
                break;
            }
        }
        Ok(())
    }

    /// the checkColumn lambda of dualFixing
    fn check_column(&self, col: i32, col_direction: i32, row: i32) -> bool {
        if col_direction as f64 * self.col_cost[col as usize] < 0.0 {
            return false;
        }
        for (r2, v) in col_iter!(self, col) {
            if r2 == row {
                continue;
            }
            if self.is_redundant(r2) {
                continue;
            }
            if self.is_ranged(r2) {
                return false;
            }
            let ru = r2 as usize;
            let row_direction = if self.row_lower[ru] == -INF && self.row_upper[ru] != INF { 1 } else { -1 };
            if (col_direction * row_direction) as f64 * v < 0.0 {
                return false;
            }
        }
        true
    }

    /// the handleSingleEquation lambda of dualFixing
    fn handle_single_equation(&mut self, row: i32) -> R {
        struct EqNz {
            col: i32,
            val: f64,
            mark: i32,
        }
        let r = row as usize;
        let mut eq: Vec<EqNz> = Vec::with_capacity(self.rowsize[r] as usize);
        let mut num_s_plus = 0;
        let mut num_s_minus = 0;
        for (ci, v) in row_iter!(self, row) {
            let col_direction: i32 = if v.is_sign_negative() { -1 } else { 1 };
            let mut mark = 0;
            if self.check_column(ci, col_direction, row) {
                num_s_plus += 1;
                mark = 1;
            } else if self.check_column(ci, -col_direction, row) {
                num_s_minus += 1;
                mark = -1;
            }
            eq.push(EqNz { col: ci, val: v, mark });
        }
        if num_s_plus == 0 && num_s_minus == 0 {
            return Ok(());
        }
        let mut t_plus = CDouble::from(0.0);
        let mut t_minus = CDouble::from(0.0);
        let mut sc_plus = CDouble::from(0.0);
        let mut sc_minus = CDouble::from(0.0);
        let mut t_plus_finite = true;
        let mut t_minus_finite = true;
        let mut sc_plus_finite = true;
        let mut sc_minus_finite = true;
        let compute_activity = |s: &Self, col: i32, val: f64, activity: &mut CDouble, finite: &mut bool, direction: i32| {
            let c = col as usize;
            let bound = if direction as f64 * val > 0.0 { s.col_upper[c] } else { s.col_lower[c] };
            *finite = *finite && bound.abs() != INF;
            if *finite {
                *activity += CDouble::from(val) * bound;
            }
        };
        for nz in &eq {
            let is_cont = self.integrality[nz.col as usize] == CONTINUOUS;
            if nz.mark <= 0 || !is_cont {
                compute_activity(self, nz.col, nz.val, &mut t_plus, &mut t_plus_finite, 1);
            } else {
                compute_activity(self, nz.col, nz.val, &mut sc_plus, &mut sc_plus_finite, -1);
            }
            if nz.mark >= 0 || !is_cont {
                compute_activity(self, nz.col, nz.val, &mut t_minus, &mut t_minus_finite, -1);
            } else {
                compute_activity(self, nz.col, nz.val, &mut sc_minus, &mut sc_minus_finite, 1);
            }
            if (num_s_minus == 0 || !t_plus_finite || !sc_plus_finite)
                && (num_s_plus == 0 || !t_minus_finite || !sc_minus_finite)
            {
                break;
            }
        }
        let fix_cols = |s: &mut Self, direction: i32| -> R {
            for nz in &eq {
                if direction * nz.mark >= 0 {
                    continue;
                }
                if direction as f64 * nz.val > 0.0 {
                    s.fix_col_to_upper(nz.col)?;
                } else {
                    s.fix_col_to_lower(nz.col)?;
                }
            }
            Ok(())
        };
        let feastol = self.primal_feastol;
        if num_s_minus > 0
            && t_plus_finite
            && sc_plus_finite
            && t_plus + sc_plus <= self.row_lower[r] + feastol
        {
            fix_cols(self, 1)?;
        } else if num_s_plus > 0
            && t_minus_finite
            && sc_minus_finite
            && t_minus + sc_minus >= self.row_lower[r] - feastol
        {
            fix_cols(self, -1)?;
        }
        Ok(())
    }

    pub(crate) fn singleton_col_stuffing(&mut self, col: i32) -> R {
        #[derive(Clone, Copy)]
        struct Candidate {
            col: i32,
            val: f64,
            multiplier: i32,
        }
        let c = col as usize;
        let is_singleton =
            |s: &Self, j: i32| s.col_deleted[j as usize] == 0 && s.colsize[j as usize] == 1 && s.col_lower[j as usize] != s.col_upper[j as usize];
        if !is_singleton(self, col) {
            return Ok(());
        }
        let row = self.a_row[self.colhead[c] as usize];
        let r = row as usize;
        if self.rowsize[r] <= 1 || self.is_ranged(row) {
            return Ok(());
        }
        let mut num_fixed_cols = 0;

        struct Acts {
            sum_lower: CDouble,
            sum_upper: CDouble,
            sum_lower_finite: bool,
            sum_upper_finite: bool,
        }

        // computeCandidates
        let compute_candidates = |s: &Self,
                                  direction: i32,
                                  candidates: &mut Vec<Candidate>,
                                  a: &mut Acts,
                                  num_integer_candidates: &mut usize,
                                  min_weight: &mut f64,
                                  max_weight: &mut f64,
                                  allow_integer_candidates: bool|
         -> bool {
            candidates.clear();
            candidates.reserve(s.rowsize[r] as usize);
            a.sum_lower = CDouble::from(0.0);
            a.sum_upper = CDouble::from(0.0);
            a.sum_lower_finite = true;
            a.sum_upper_finite = true;
            *num_integer_candidates = 0;
            *min_weight = INF;
            *max_weight = -INF;
            let mut add_candidate = |cands: &mut Vec<Candidate>, j: i32, val: f64, dir: i32| {
                if s.integrality[j as usize] == INTEGER {
                    *num_integer_candidates += 1;
                }
                *min_weight = std_min(*min_weight, dir as f64 * val);
                *max_weight = std_max(*max_weight, dir as f64 * val);
                cands.push(Candidate { col: j, val, multiplier: dir });
            };
            for (j, v) in row_iter!(s, row) {
                let ju = j as usize;
                let aj = direction as f64 * v;
                let cj = s.col_cost[ju];
                let mut sum_lower_bound = s.col_lower[ju];
                let mut sum_upper_bound = s.col_upper[ju];
                let is_candidate = allow_integer_candidates || s.integrality[ju] != INTEGER;
                if is_singleton(s, j) {
                    if aj > 0.0 {
                        if cj >= 0.0 {
                            sum_upper_bound = sum_lower_bound;
                        } else if is_candidate {
                            sum_upper_bound = sum_lower_bound;
                            add_candidate(candidates, j, aj, 1);
                        }
                    } else if cj <= 0.0 {
                        sum_lower_bound = sum_upper_bound;
                    } else if is_candidate {
                        sum_lower_bound = sum_upper_bound;
                        add_candidate(candidates, j, aj, -1);
                    }
                }
                if aj < 0.0 {
                    std::mem::swap(&mut sum_lower_bound, &mut sum_upper_bound);
                }
                // updateActivityBounds
                a.sum_lower_finite = a.sum_lower_finite && sum_lower_bound.abs() != INF;
                a.sum_upper_finite = a.sum_upper_finite && sum_upper_bound.abs() != INF;
                if a.sum_lower_finite {
                    a.sum_lower += aj * CDouble::from(sum_lower_bound);
                }
                if a.sum_upper_finite {
                    a.sum_upper += aj * CDouble::from(sum_upper_bound);
                }
                if !a.sum_lower_finite && !a.sum_upper_finite {
                    return false;
                }
            }
            true
        };

        let check_candidates = |s: &Self, direction: i32, candidates: &mut Vec<Candidate>, a: &mut Acts| -> bool {
            let mut num_integer_candidates = 0usize;
            let mut min_weight = 0.0;
            let mut max_weight = 0.0;
            if compute_candidates(
                s,
                direction,
                candidates,
                a,
                &mut num_integer_candidates,
                &mut min_weight,
                &mut max_weight,
                true,
            ) {
                if num_integer_candidates == 0 {
                    return true;
                }
                if num_integer_candidates == candidates.len() {
                    return min_weight == max_weight;
                }
            }
            num_integer_candidates > 0
                && compute_candidates(
                    s,
                    direction,
                    candidates,
                    a,
                    &mut num_integer_candidates,
                    &mut min_weight,
                    &mut max_weight,
                    false,
                )
        };

        let mut check_row = |s: &mut Self, rhs: f64, direction: i32| -> R {
            let d = direction as f64;
            if d * rhs == INF {
                return Ok(());
            }
            let mut candidates: Vec<Candidate> = Vec::new();
            let mut a = Acts {
                sum_lower: CDouble::from(0.0),
                sum_upper: CDouble::from(0.0),
                sum_lower_finite: false,
                sum_upper_finite: false,
            };
            if !check_candidates(s, direction, &mut candidates, &mut a) {
                return Ok(());
            }
            {
                let cost = &s.col_cost;
                pdqsort(&mut candidates, |c1, c2| {
                    cost[c1.col as usize] / c1.val < cost[c2.col as usize] / c2.val
                });
            }
            let feastol = s.primal_feastol;
            for t in &candidates {
                let tc = t.col as usize;
                if s.col_lower[tc] == -INF || s.col_upper[tc] == INF {
                    break;
                }
                let delta =
                    (t.multiplier as f64 * t.val) * (CDouble::from(s.col_upper[tc]) - CDouble::from(s.col_lower[tc]));
                if a.sum_upper_finite && delta <= d * rhs - a.sum_upper + feastol {
                    num_fixed_cols += 1;
                    if t.multiplier < 0 {
                        s.fix_col_to_lower(t.col)?;
                    } else {
                        s.fix_col_to_upper(t.col)?;
                    }
                } else if a.sum_lower_finite && delta <= a.sum_lower - d * rhs + feastol {
                    num_fixed_cols += 1;
                    if -t.multiplier < 0 {
                        s.fix_col_to_lower(t.col)?;
                    } else {
                        s.fix_col_to_upper(t.col)?;
                    }
                }
                if a.sum_lower_finite {
                    a.sum_lower += delta;
                }
                if a.sum_upper_finite {
                    a.sum_upper += delta;
                }
            }
            Ok(())
        };

        let ru = self.row_upper[r];
        check_row(self, ru, 1)?;
        let rl = self.row_lower[r];
        check_row(self, rl, -1)?;

        if num_fixed_cols > 0 {
            let msg = crate::util::printf::sprintf(
                "Singleton column stuffing fixed %d columns\n",
                &[num_fixed_cols.into()],
            );
            self.log_dev(LOG_DETAILED, &msg);
        }
        Ok(())
    }

    // ---------------------------------------------------- column bounds

    pub(crate) fn compute_implied_lower_bound(&mut self, col: i32, bound_col: i32, bound_col_value: f64, pattern: i32) -> f64 {
        self.compute_col_bounds(col, bound_col, bound_col_value, pattern, [true, false, false, false])[0]
    }

    pub(crate) fn compute_implied_upper_bound(&mut self, col: i32, bound_col: i32, bound_col_value: f64, pattern: i32) -> f64 {
        self.compute_col_bounds(col, bound_col, bound_col_value, pattern, [false, true, false, false])[1]
    }

    pub(crate) fn compute_worst_case_lower_bound(
        &mut self,
        col: i32,
        bound_col: i32,
        bound_col_value: f64,
        pattern: i32,
    ) -> f64 {
        self.compute_col_bounds(col, bound_col, bound_col_value, pattern, [false, false, true, false])[2]
    }

    pub(crate) fn compute_worst_case_upper_bound(
        &mut self,
        col: i32,
        bound_col: i32,
        bound_col_value: f64,
        pattern: i32,
    ) -> f64 {
        self.compute_col_bounds(col, bound_col, bound_col_value, pattern, [false, false, false, true])[3]
    }

    /// computeColBounds: [lower, upper, worst case lower, worst case upper]
    /// for the requested ones
    pub(crate) fn compute_col_bounds(
        &mut self,
        col: i32,
        bound_col: i32,
        bound_col_value: f64,
        pattern: i32,
        want: [bool; 4],
    ) -> [f64; 4] {
        let mut out = [0.0f64; 4];
        if !want.iter().any(|&w| w) {
            return out;
        }
        let lower_requested = want[0] || want[2];
        let upper_requested = want[1] || want[3];
        let skip_non_zero = |s: &Self, row: i32, val: f64| -> bool {
            let has_lower = s.yields_implied_lower_bound(row, val);
            let has_upper = s.yields_implied_upper_bound(row, val);
            (!upper_requested && lower_requested && !has_lower)
                || (!lower_requested && upper_requested && !has_upper)
                || (upper_requested && lower_requested && !has_lower && !has_upper)
        };

        // (row, jval, kval)
        let mut nzs: Vec<(i32, f64, f64)> = Vec::with_capacity(self.colsize[col as usize] as usize);
        let store_triplet = |nzs: &mut Vec<(i32, f64, f64)>, row: i32, jval: f64, kval: f64| {
            if pattern != 0 && jval.is_sign_negative() != (pattern as f64 * kval).is_sign_negative() {
                return;
            }
            nzs.push((row, jval, kval));
        };

        if bound_col != -1 {
            if self.colsize[col as usize] < self.colsize[bound_col as usize] {
                let mut p = self.colhead[col as usize];
                while p != -1 {
                    let pu = p as usize;
                    let row = self.a_row[pu];
                    let v = self.a_value[pu];
                    p = self.a_next[pu];
                    if skip_non_zero(self, row, v) {
                        continue;
                    }
                    let nz_pos = self.find_nonzero(row, bound_col);
                    if nz_pos == -1 {
                        continue;
                    }
                    store_triplet(&mut nzs, row, v, self.a_value[nz_pos as usize]);
                }
            } else {
                let mut p = self.colhead[bound_col as usize];
                while p != -1 {
                    let pu = p as usize;
                    let row = self.a_row[pu];
                    let v = self.a_value[pu];
                    p = self.a_next[pu];
                    let nz_pos = self.find_nonzero(row, col);
                    if nz_pos == -1 {
                        continue;
                    }
                    let jv = self.a_value[nz_pos as usize];
                    if skip_non_zero(self, row, jv) {
                        continue;
                    }
                    store_triplet(&mut nzs, row, jv, v);
                }
            }
        } else {
            for (row, v) in col_iter!(self, col) {
                if skip_non_zero(self, row, v) {
                    continue;
                }
                nzs.push((row, v, -INF));
            }
        }

        out = [-INF, INF, -INF, INF];

        let compute_bound = |s: &Self, t: &(i32, f64, f64), rhs: f64, direction: i32, worst: bool| -> f64 {
            let residual: f64;
            let b = rb!(s);
            if (direction > 0 && !worst) || (direction < 0 && worst) {
                residual = if worst {
                    s.implied_row_bounds.residual_sum_lower(t.0, col, t.1, bound_col, t.2, bound_col_value, &b)
                } else {
                    s.implied_row_bounds.residual_sum_lower_orig(t.0, col, t.1, bound_col, t.2, bound_col_value, &b)
                };
                if residual == -INF {
                    return INF.copysign(t.1);
                }
            } else {
                residual = if worst {
                    s.implied_row_bounds.residual_sum_upper(t.0, col, t.1, bound_col, t.2, bound_col_value, &b)
                } else {
                    s.implied_row_bounds.residual_sum_upper_orig(t.0, col, t.1, bound_col, t.2, bound_col_value, &b)
                };
                if residual == INF {
                    return -INF.copysign(t.1);
                }
            }
            ((CDouble::from(rhs) - CDouble::from(residual)) / t.1).to_f64()
        };

        let update_bounds = |s: &Self, out: &mut [f64; 4], t: &(i32, f64, f64), rhs: f64, direction: i32| {
            if direction as f64 * rhs == INF {
                return;
            }
            if direction as f64 * t.1 < 0.0 {
                if want[0] {
                    out[0] = std_max(out[0], compute_bound(s, t, rhs, direction, false));
                }
                if want[2] && out[2] != INF {
                    out[2] = std_max(out[2], compute_bound(s, t, rhs, direction, true));
                }
            } else {
                if want[1] {
                    out[1] = std_min(out[1], compute_bound(s, t, rhs, direction, false));
                }
                if want[3] && out[3] != -INF {
                    out[3] = std_min(out[3], compute_bound(s, t, rhs, direction, true));
                }
            }
        };

        for t in &nzs {
            let ru = self.row_upper[t.0 as usize];
            update_bounds(self, &mut out, t, ru, 1);
            let rl = self.row_lower[t.0 as usize];
            update_bounds(self, &mut out, t, rl, -1);
        }
        out
    }

    pub(crate) fn presolve_col_singletons(&mut self) -> R {
        let mut i = 0;
        while i < self.singleton_columns.len() {
            let col = self.singleton_columns[i];
            i += 1;
            if self.col_deleted[col as usize] != 0 {
                continue;
            }
            self.col_presolve(col)?;
        }
        let (cd, cs) = (&self.col_deleted, &self.colsize);
        self.singleton_columns.retain(|&c| !(cd[c as usize] != 0 || cs[c as usize] > 1));
        Ok(())
    }

    pub(crate) fn presolve_changed_cols(&mut self) -> R {
        let mut changed_cols: Vec<i32> = Vec::with_capacity((self.num_col - self.num_deleted_cols).max(0) as usize);
        std::mem::swap(&mut changed_cols, &mut self.changed_col_indices);
        for &col in &changed_cols {
            if self.col_deleted[col as usize] != 0 {
                continue;
            }
            self.col_presolve(col)?;
            self.changed_col_flag[col as usize] = self.col_deleted[col as usize];
        }
        Ok(())
    }
}
