//! Matrix modifications shared by the reductions (fixing and substituting
//! columns, removing rows, scaling), and the aggregator, sparsification,
//! slack removal, inequality strengthening and dependent equations.

use super::*;
use crate::mip::cuts::sort::pdqsort;
use crate::util::hash::HighsHash;

impl Presolve<'_> {
    pub(crate) fn transform_column(&mut self, col: i32, scale: f64, constant: f64) -> R {
        let c = col as usize;
        if self.mip.is_some() {
            self.implications().column_transformed(col, scale, constant);
        }
        self.ps.linear_transform(col, scale, constant);

        if constant != 0.0 {
            let old_lower = self.col_lower[c];
            let old_upper = self.col_upper[c];
            self.col_upper[c] -= constant;
            self.col_lower[c] -= constant;
            let mut p = self.colhead[c];
            while p != -1 {
                let pu = p as usize;
                let row = self.a_row[pu];
                let val = self.a_value[pu];
                self.implied_row_bounds.updated_var_lower(row, col, val, old_lower, &rb!(self));
                self.implied_row_bounds.updated_var_upper(row, col, val, old_upper, &rb!(self));
                p = self.a_next[pu];
            }
            let old_impl_lower = self.impl_col_lower[c];
            let old_impl_upper = self.impl_col_upper[c];
            self.impl_col_lower[c] -= constant;
            self.impl_col_upper[c] -= constant;
            let mut p = self.colhead[c];
            while p != -1 {
                let pu = p as usize;
                let row = self.a_row[pu];
                let val = self.a_value[pu];
                let ls = self.col_lower_source[c];
                let us = self.col_upper_source[c];
                self.implied_row_bounds.updated_impl_var_lower(row, col, val, old_impl_lower, ls, &rb!(self));
                self.implied_row_bounds.updated_impl_var_upper(row, col, val, old_impl_upper, us, &rb!(self));
                p = self.a_next[pu];
            }
        }

        self.implied_dual_row_bounds.sum_scaled(col, scale);

        let bound_scale = 1.0 / scale;
        self.col_lower[c] *= bound_scale;
        self.col_upper[c] *= bound_scale;
        self.impl_col_lower[c] *= bound_scale;
        self.impl_col_upper[c] *= bound_scale;
        if scale < 0.0 {
            let (l, u) = (self.col_lower[c], self.col_upper[c]);
            self.col_lower[c] = u;
            self.col_upper[c] = l;
            let (l, u) = (self.impl_col_lower[c], self.impl_col_upper[c]);
            self.impl_col_lower[c] = u;
            self.impl_col_upper[c] = l;
            let (l, u) = (self.col_lower_source[c], self.col_upper_source[c]);
            self.col_lower_source[c] = u;
            self.col_upper_source[c] = l;
        }

        self.offset = self.col_cost[c].mul_add_c(constant, self.offset);
        self.col_cost[c] *= scale;

        let mut p = self.colhead[c];
        while p != -1 {
            let pu = p as usize;
            let val = self.a_value[pu];
            self.a_value[pu] *= scale;
            let row = self.a_row[pu] as usize;
            let row_constant = val * constant;
            if self.row_lower[row] != -INF {
                self.row_lower[row] -= row_constant;
            }
            if self.row_upper[row] != INF {
                self.row_upper[row] -= row_constant;
            }
            p = self.a_next[pu];
        }

        if self.integrality[c] != CONTINUOUS {
            self.change_col_bounds(col, self.col_lower[c], self.col_upper[c])?;
        }
        self.mark_changed_col(col);
        Ok(())
    }

    pub(crate) fn scale_stored_row(&mut self, row: i32, scale: f64, integral: bool) {
        let r = row as usize;
        self.row_upper[r] *= scale;
        self.row_lower[r] *= scale;
        self.impl_row_dual_lower[r] /= scale;
        self.impl_row_dual_upper[r] /= scale;
        if integral {
            if self.row_upper[r] != INF {
                self.row_upper[r] = self.row_upper[r].round();
            }
            if self.row_lower[r] != -INF {
                self.row_lower[r] = self.row_lower[r].round();
            }
        }
        let n = self.rp_len;
        for j in 0..n {
            let p = self.rowpositions[j];
            self.a_value[p as usize] *= scale;
            if self.a_value[p as usize].abs() <= self.opt.small_matrix_value {
                self.unlink(p);
            }
        }
        self.implied_row_bounds.sum_scaled(row, scale);
        if scale < 0.0 {
            let (l, u) = (self.row_dual_lower[r], self.row_dual_upper[r]);
            self.row_dual_lower[r] = u;
            self.row_dual_upper[r] = l;
            let (l, u) = (self.impl_row_dual_lower[r], self.impl_row_dual_upper[r]);
            self.impl_row_dual_lower[r] = u;
            self.impl_row_dual_upper[r] = l;
            let (l, u) = (self.row_dual_lower_source[r], self.row_dual_upper_source[r]);
            self.row_dual_lower_source[r] = u;
            self.row_dual_upper_source[r] = l;
            let (l, u) = (self.row_lower[r], self.row_upper[r]);
            self.row_lower[r] = u;
            self.row_upper[r] = l;
        }
    }

    /// substitute(row, col, rhs): substitutes the implied free column of the
    /// (stored) row
    pub(crate) fn substitute_row_col(&mut self, row: i32, col: i32, rhs: f64) {
        let pos = self.find_nonzero(row, col);
        debug_assert!(pos != -1);
        let substrowscale = -1.0 / self.a_value[pos as usize];
        self.mark_row_deleted(row);
        self.mark_col_deleted(col);

        let c = col as usize;
        let mut coliter = self.colhead[c];
        while coliter != -1 {
            let cp = coliter as usize;
            let colrow = self.a_row[cp];
            let colval = self.a_value[cp];
            let colpos = coliter;
            coliter = self.a_next[cp];
            if row == colrow {
                continue;
            }
            self.unlink(colpos);
            let scale = colval * substrowscale;
            let cr = colrow as usize;
            if self.row_lower[cr] != -INF {
                self.row_lower[cr] = scale.mul_add_c(rhs, self.row_lower[cr]);
            }
            if self.row_upper[cr] != INF {
                self.row_upper[cr] = scale.mul_add_c(rhs, self.row_upper[cr]);
            }
            let n = self.rp_len;
            for k in 0..n {
                let rowiter = self.rowpositions[k] as usize;
                if self.a_col[rowiter] != col {
                    let ac = self.a_col[rowiter];
                    let v = scale * self.a_value[rowiter];
                    self.add_to_matrix(colrow, ac, v);
                }
            }
            self.reinsert_equation(colrow);
        }

        if self.col_cost[c] != 0.0 {
            let objscale = CDouble::from(self.col_cost[c] * substrowscale);
            self.offset = (self.offset - objscale * rhs).to_f64();
            let n = self.rp_len;
            for k in 0..n {
                let rowiter = self.rowpositions[k] as usize;
                let ac = self.a_col[rowiter] as usize;
                self.col_cost[ac] = (self.col_cost[ac] + objscale * self.a_value[rowiter]).to_f64();
                if self.col_cost[ac].abs() <= self.opt.small_matrix_value {
                    self.col_cost[ac] = 0.0;
                }
            }
            self.col_cost[c] = 0.0;
        }

        let n = self.rp_len;
        for k in 0..n {
            let rowiter = self.rowpositions[k];
            self.unlink(rowiter);
        }
    }

    /// substitute(substcol, staycol, offset, scale): substcol = offset +
    /// scale * staycol
    pub(crate) fn substitute_cols(&mut self, substcol: i32, staycol: i32, offset: f64, scale: f64) {
        let sc = substcol as usize;
        let mut coliter = self.colhead[sc];
        while coliter != -1 {
            let cp = coliter as usize;
            let colrow = self.a_row[cp];
            let colval = self.a_value[cp];
            let colpos = coliter;
            coliter = self.a_next[cp];
            self.unlink(colpos);
            let cr = colrow as usize;
            if self.row_lower[cr] != -INF {
                self.row_lower[cr] = (-colval).mul_add_c(offset, self.row_lower[cr]);
            }
            if self.row_upper[cr] != INF {
                self.row_upper[cr] = (-colval).mul_add_c(offset, self.row_upper[cr]);
            }
            self.add_to_matrix(colrow, staycol, scale * colval);
            self.reinsert_equation(colrow);
        }
        if self.col_cost[sc] != 0.0 {
            self.offset = self.col_cost[sc].mul_add_c(offset, self.offset);
            let st = staycol as usize;
            self.col_cost[st] = scale.mul_add_c(self.col_cost[sc], self.col_cost[st]);
            if self.col_cost[st].abs() <= self.opt.small_matrix_value {
                self.col_cost[st] = 0.0;
            }
            self.col_cost[sc] = 0.0;
        }
    }

    pub(crate) fn fix_col_to_lower(&mut self, col: i32) -> R {
        let c = col as usize;
        let fixval = self.col_lower[c];
        if fixval == -INF {
            return Err(Stop::DualInfeasible);
        }
        let logging_on = self.analysis.logging_on;
        if logging_on {
            self.start_rule_log(RULE_FIXED_COL);
        }
        self.ps.fixed_col_at_lower(col, fixval, self.col_cost[c], col_iter!(self, col));
        self.remove_fixed_col(col, fixval);
        self.analysis.logging_on = logging_on;
        if logging_on {
            self.stop_rule_log(RULE_FIXED_COL);
        }
        Ok(())
    }

    pub(crate) fn fix_col_to_upper(&mut self, col: i32) -> R {
        let c = col as usize;
        let fixval = self.col_upper[c];
        if fixval == INF {
            return Err(Stop::DualInfeasible);
        }
        let logging_on = self.analysis.logging_on;
        if logging_on {
            self.start_rule_log(RULE_FIXED_COL);
        }
        self.ps.fixed_col_at_upper(col, fixval, self.col_cost[c], col_iter!(self, col));
        self.remove_fixed_col(col, fixval);
        self.analysis.logging_on = logging_on;
        if logging_on {
            self.stop_rule_log(RULE_FIXED_COL);
        }
        Ok(())
    }

    pub(crate) fn fix_col_to_zero(&mut self, col: i32) {
        let logging_on = self.analysis.logging_on;
        if logging_on {
            self.start_rule_log(RULE_FIXED_COL);
        }
        self.ps.fixed_col_at_zero(col, self.col_cost[col as usize], col_iter!(self, col));
        self.remove_fixed_col(col, 0.0);
        self.analysis.logging_on = logging_on;
        if logging_on {
            self.stop_rule_log(RULE_FIXED_COL);
        }
    }

    pub(crate) fn remove_row(&mut self, row: i32) {
        self.mark_row_deleted(row);
        self.store_row(row);
        let n = self.rp_len;
        for k in 0..n {
            let p = self.rowpositions[k];
            self.unlink(p);
        }
    }

    /// removeFixedCol(col): fixed at its lower bound, logged
    pub(crate) fn remove_fixed_col_logged(&mut self, col: i32) {
        let logging_on = self.analysis.logging_on;
        if logging_on {
            self.start_rule_log(RULE_FIXED_COL);
        }
        let v = self.col_lower[col as usize];
        self.remove_fixed_col(col, v);
        self.analysis.logging_on = logging_on;
        if logging_on {
            self.stop_rule_log(RULE_FIXED_COL);
        }
    }

    pub(crate) fn remove_fixed_col(&mut self, col: i32, fixval: f64) {
        self.mark_col_deleted(col);
        let c = col as usize;
        let mut coliter = self.colhead[c];
        while coliter != -1 {
            let cp = coliter as usize;
            let colrow = self.a_row[cp];
            let colval = self.a_value[cp];
            let colpos = coliter;
            coliter = self.a_next[cp];
            let cr = colrow as usize;
            if self.row_lower[cr] != -INF {
                self.row_lower[cr] = (-colval).mul_add_c(fixval, self.row_lower[cr]);
            }
            if self.row_upper[cr] != INF {
                self.row_upper[cr] = (-colval).mul_add_c(fixval, self.row_upper[cr]);
            }
            self.unlink(colpos);
            self.reinsert_equation(colrow);
        }
        self.offset = self.col_cost[c].mul_add_c(fixval, self.offset);
        self.col_cost[c] = 0.0;
    }

    pub(crate) fn count_fillin(&mut self, row: i32) -> i32 {
        let mut fillin = 0;
        let n = self.rp_len;
        for k in 0..n {
            let rowiter = self.rowpositions[k] as usize;
            let ac = self.a_col[rowiter];
            if self.find_nonzero(row, ac) == -1 {
                fillin += 1;
            }
        }
        fillin
    }

    pub(crate) fn aggregator(&mut self) -> R {
        let logging_on = self.analysis.logging_on;
        if logging_on {
            self.start_rule_log(RULE_AGGREGATOR);
        }
        let mut subs = std::mem::take(&mut self.substitution_opportunities);
        subs.retain(|&(row, col)| {
            !(self.row_deleted[row as usize] != 0
                || self.col_deleted[col as usize] != 0
                || !self.is_implied_free(col)
                || !self.is_dual_implied_free(row))
        });
        {
            let rowsize = &self.rowsize;
            let colsize = &self.colsize;
            pdqsort(&mut subs, |nz1, nz2| {
                let min_len1 = rowsize[nz1.0 as usize].min(colsize[nz1.1 as usize]);
                let min_len2 = rowsize[nz2.0 as usize].min(colsize[nz2.1 as usize]);
                if min_len1 == 2 && min_len2 != 2 {
                    return true;
                }
                if min_len2 == 2 && min_len1 != 2 {
                    return false;
                }
                let size_prod1 = rowsize[nz1.0 as usize] as i64 * colsize[nz1.1 as usize] as i64;
                let size_prod2 = rowsize[nz2.0 as usize] as i64 * colsize[nz2.1 as usize] as i64;
                if size_prod1 < size_prod2 {
                    return true;
                }
                if size_prod2 < size_prod1 {
                    return false;
                }
                if min_len1 < min_len2 {
                    return true;
                }
                if min_len2 < min_len1 {
                    return false;
                }
                let h1 = (nz1.0 as u32, nz1.1 as u32).highs_hash();
                let h2 = (nz2.0 as u32, nz2.1 as u32).highs_hash();
                (h1, nz1.0, nz1.1) < (h2, nz2.0, nz2.1)
            });
        }
        self.substitution_opportunities = subs;

        let mut nfail = 0;
        let mut i = 0;
        let result = loop {
            if i >= self.substitution_opportunities.len() {
                break Ok(());
            }
            let (row, col) = self.substitution_opportunities[i];
            let idx = i;
            i += 1;
            let r = row as usize;
            let c = col as usize;
            if self.row_deleted[r] != 0 || self.col_deleted[c] != 0 || !self.is_implied_free(col) || !self.is_dual_implied_free(row)
            {
                self.substitution_opportunities[idx].0 = -1;
                continue;
            }
            let nz_pos = self.find_nonzero(row, col);
            if nz_pos == -1 {
                self.substitution_opportunities[idx].0 = -1;
                continue;
            }
            if self.integrality[c] == INTEGER {
                match self.is_implied_integral(col) {
                    Err(e) => break Err(e),
                    Ok(false) => continue,
                    Ok(true) => {}
                }
            }
            if self.rowsize[r] == 2 || self.colsize[c] == 2 {
                self.store_row(row);
                self.substitute_free_col(row, col, true);
                self.substitution_opportunities[idx].0 = -1;
                if let Err(e) = self.remove_row_singletons() {
                    break Err(e);
                }
                if let Err(e) = self.check_limits() {
                    break Err(e);
                }
                continue;
            }
            let thr = self.opt.presolve_pivot_threshold;
            let mut max_val =
                if self.rowsize[r] < self.colsize[c] { self.get_max_abs_row_val(row) } else { self.get_max_abs_col_val(col) };
            if self.a_value[nz_pos as usize].abs() < max_val * thr {
                max_val = if self.rowsize[r] < self.colsize[c] {
                    self.get_max_abs_col_val(col)
                } else {
                    self.get_max_abs_row_val(row)
                };
                if self.a_value[nz_pos as usize].abs() < max_val * thr {
                    self.substitution_opportunities[idx].0 = -1;
                    continue;
                }
            }

            self.store_row(row);
            let maxfillin = self.opt.presolve_substitution_maxfillin;
            let mut fillin = -(self.rowsize[r] + self.colsize[c] - 1);
            let mut p = self.colhead[c];
            while p != -1 {
                let pu = p as usize;
                let nzrow = self.a_row[pu];
                p = self.a_next[pu];
                if nzrow == row {
                    continue;
                }
                fillin += self.count_fillin(nzrow);
                if fillin > maxfillin {
                    break;
                }
            }
            if fillin > maxfillin {
                nfail += 1;
                if nfail == 3 {
                    break Ok(());
                }
                continue;
            }
            nfail = 0;
            self.substitute_free_col(row, col, true);
            self.substitution_opportunities[idx].0 = -1;
            if let Err(e) = self.remove_row_singletons() {
                break Err(e);
            }
            if let Err(e) = self.check_limits() {
                break Err(e);
            }
        };
        result?;

        self.substitution_opportunities.retain(|p| p.0 != -1);
        self.analysis.logging_on = logging_on;
        if logging_on {
            self.stop_rule_log(RULE_AGGREGATOR);
        }
        Ok(())
    }

    pub(crate) fn equality_row_addition(
        &mut self,
        stayrow: i32,
        removerow: i32,
        scale: f64,
        rowvector: &[(i32, f64)],
    ) -> R {
        let stay_rowpositions = self.get_row_positions(stayrow);
        self.ps.equality_row_addition(removerow, stayrow, scale, rowvector.iter().copied());
        for &rowiter in &stay_rowpositions {
            let ac = self.a_col[rowiter as usize];
            let pos = self.find_nonzero(removerow, ac);
            if pos != -1 {
                self.unlink(pos);
            } else {
                let v = scale * self.a_value[rowiter as usize];
                self.add_to_matrix(removerow, ac, v);
            }
        }
        let rr = removerow as usize;
        let sr = stayrow as usize;
        if self.row_upper[rr] != INF {
            self.row_upper[rr] = (self.row_upper[rr] + CDouble::from(scale) * self.row_upper[sr]).to_f64();
        }
        if self.row_lower[rr] != -INF {
            self.row_lower[rr] = (self.row_lower[rr] + CDouble::from(scale) * self.row_upper[sr]).to_f64();
        }
        self.row_presolve(removerow)
    }

    pub(crate) fn remove_slacks(&mut self) -> R {
        for i_col in 0..self.num_col {
            let c = i_col as usize;
            if self.col_deleted[c] != 0 {
                continue;
            }
            if self.colsize[c] != 1 {
                continue;
            }
            if self.integrality[c] == INTEGER {
                continue;
            }
            let coliter = self.colhead[c];
            let i_row = self.a_row[coliter as usize];
            if !self.is_equation(i_row) {
                continue;
            }
            let r = i_row as usize;
            let lower = self.col_lower[c];
            let upper = self.col_upper[c];
            let cost = self.col_cost[c];
            let rhs = self.row_lower[r];
            let coeff = self.a_value[coliter as usize];
            self.row_lower[r] = if coeff > 0.0 { (-coeff).mul_add_c(upper, rhs) } else { (-coeff).mul_add_c(lower, rhs) };
            self.row_upper[r] = if coeff > 0.0 { (-coeff).mul_add_c(lower, rhs) } else { (-coeff).mul_add_c(upper, rhs) };
            if cost != 0.0 {
                let multiplier = cost / coeff;
                let mut w = PreOrder::new(self.rowroot[r]);
                while let Some(p) = w.next(&self.ar_left, &self.ar_right) {
                    let lc = self.a_col[p] as usize;
                    let lv = self.a_value[p];
                    self.col_cost[lc] = (-multiplier).mul_add_c(lv, self.col_cost[lc]);
                }
                self.offset = multiplier.mul_add_c(rhs, self.offset);
            }
            self.ps.slack_col_substitution(i_row, i_col, rhs, row_iter!(self, i_row));
            self.mark_col_deleted(i_col);
            self.unlink(coliter);
        }
        Ok(())
    }

    pub(crate) fn sparsify(&mut self) -> R {
        let logging_on = self.analysis.logging_on;
        if logging_on {
            self.start_rule_log(RULE_SPARSIFY);
        }
        self.remove_row_singletons()?;
        self.remove_doubleton_equations()?;
        let tmp_equations: Vec<i32> = self.equations.iter().map(|e| e.1).collect();
        let min_nonzero_val = self.primal_feastol.sqrt();
        let small = self.opt.small_matrix_value;
        let mut sparsify_rows: Vec<(i32, f64)> = Vec::new();
        // std::map<double, HighsInt> possibleScales: a sorted vector
        let mut possible_scales: Vec<(f64, i32)> = Vec::new();

        for &eqrow in &tmp_equations {
            if self.row_deleted[eqrow as usize] != 0 {
                continue;
            }
            self.store_row(eqrow);

            let mut sparsest_col = -1;
            let mut second_sparsest_col = -1;
            let mut sparsest_col_len = IINF;
            let mut second_sparsest_col_len = IINF;
            for k in 0..self.rp_len {
                let col = self.a_col[self.rowpositions[k] as usize];
                let cs = self.colsize[col as usize];
                if cs < sparsest_col_len {
                    second_sparsest_col = sparsest_col;
                    second_sparsest_col_len = sparsest_col_len;
                    sparsest_col = col;
                    sparsest_col_len = cs;
                } else if cs < second_sparsest_col_len {
                    second_sparsest_col = col;
                    second_sparsest_col_len = cs;
                }
            }

            sparsify_rows.clear();
            let eqr = eqrow as usize;

            let mut p = self.colhead[sparsest_col as usize];
            while p != -1 {
                let pu = p as usize;
                let cand_row = self.a_row[pu];
                let colnz_val = self.a_value[pu];
                p = self.a_next[pu];
                if cand_row == eqrow {
                    continue;
                }
                possible_scales.clear();
                let mut misses = 0;
                let mut max_misses = 1;
                if self.rowsize_integer[eqr] == 0 && self.rowsize_integer[cand_row as usize] != 0 {
                    max_misses -= 1;
                }
                for k in 0..self.rp_len {
                    let sp = self.rowpositions[k] as usize;
                    let nzcol = self.a_col[sp];
                    let nzval = self.a_value[sp];
                    let cand_row_val;
                    if nzcol == sparsest_col {
                        cand_row_val = colnz_val;
                    } else {
                        let nz_pos = self.find_nonzero(cand_row, nzcol);
                        if nz_pos == -1 {
                            let nc = nzcol as usize;
                            if self.integrality[nc] == INTEGER && self.col_upper[nc] - self.col_lower[nc] > 1.5 {
                                misses = 2;
                                break;
                            }
                            misses += 1;
                            if misses > max_misses {
                                break;
                            }
                            continue;
                        }
                        cand_row_val = self.a_value[nz_pos as usize];
                    }
                    let scale = -cand_row_val / nzval;
                    if scale.abs() > 1e3 {
                        continue;
                    }
                    let scale_tolerance = min_nonzero_val / nzval.abs();
                    let lb = lower_bound(&possible_scales, scale - scale_tolerance);
                    if lb < possible_scales.len() && (possible_scales[lb].0 - scale).abs() <= scale_tolerance {
                        if possible_scales[lb].1 == -1 {
                            continue;
                        }
                        if possible_scales[lb].0.mul_add_c(nzval, cand_row_val).abs() <= small {
                            possible_scales[lb].1 += 1;
                        } else {
                            possible_scales[lb].1 = -1;
                        }
                    } else {
                        map_emplace(&mut possible_scales, scale, 1);
                    }
                }
                if misses > max_misses || possible_scales.is_empty() {
                    continue;
                }
                let mut num_cancel = 0;
                let mut scale = 0.0f64;
                for &(s, cnt) in &possible_scales {
                    if cnt <= misses {
                        continue;
                    }
                    if cnt > num_cancel || (cnt == num_cancel && s.abs() < scale.abs()) {
                        scale = s;
                        num_cancel = cnt;
                    }
                }
                if num_cancel > misses {
                    sparsify_rows.push((cand_row, scale));
                }
            }

            let sc = sparsest_col as usize;
            if self.integrality[sc] != INTEGER || (self.col_upper[sc] - self.col_lower[sc]) < 1.5 {
                let mut p = self.colhead[second_sparsest_col as usize];
                while p != -1 {
                    let pu = p as usize;
                    let cand_row = self.a_row[pu];
                    let colnz_val = self.a_value[pu];
                    p = self.a_next[pu];
                    if cand_row == eqrow {
                        continue;
                    }
                    if self.rowsize_integer[eqr] == 0 && self.rowsize_integer[cand_row as usize] != 0 {
                        continue;
                    }
                    let sparsest_col_pos = self.find_nonzero(cand_row, sparsest_col);
                    if sparsest_col_pos != -1 {
                        continue;
                    }
                    possible_scales.clear();
                    let mut skip = false;
                    for k in 0..self.rp_len {
                        let sp = self.rowpositions[k] as usize;
                        let nzcol = self.a_col[sp];
                        let nzval = self.a_value[sp];
                        let cand_row_val;
                        if nzcol == second_sparsest_col {
                            cand_row_val = colnz_val;
                        } else {
                            let nz_pos = self.find_nonzero(cand_row, nzcol);
                            skip = nz_pos == -1;
                            if skip {
                                break;
                            }
                            cand_row_val = self.a_value[nz_pos as usize];
                        }
                        let scale = -cand_row_val / nzval;
                        if scale.abs() > 1e3 {
                            continue;
                        }
                        let scale_tolerance = min_nonzero_val / nzval.abs();
                        let lb = lower_bound(&possible_scales, scale - scale_tolerance);
                        if lb < possible_scales.len() && (possible_scales[lb].0 - scale).abs() <= scale_tolerance {
                            if possible_scales[lb].1 == -1 {
                                continue;
                            }
                            if possible_scales[lb].0.mul_add_c(nzval, cand_row_val).abs() <= small {
                                possible_scales[lb].1 += 1;
                            } else {
                                possible_scales[lb].1 = -1;
                                continue;
                            }
                        } else {
                            map_emplace(&mut possible_scales, scale, 1);
                        }
                    }
                    if skip || possible_scales.is_empty() {
                        continue;
                    }
                    let mut num_cancel = 0;
                    let mut scale = 0.0f64;
                    for &(s, cnt) in &possible_scales {
                        if cnt <= 1 {
                            continue;
                        }
                        if cnt > num_cancel || (cnt == num_cancel && s.abs() < scale.abs()) {
                            scale = s;
                            num_cancel = cnt;
                        }
                    }
                    if num_cancel > 1 {
                        sparsify_rows.push((cand_row, scale));
                    }
                }
            }

            if sparsify_rows.is_empty() {
                continue;
            }

            self.ps.equality_row_additions(eqrow, stored_row!(self), &sparsify_rows);
            let rhs = self.row_lower[eqr];
            for &(row, scale) in &sparsify_rows {
                let r = row as usize;
                if self.row_lower[r] != -INF {
                    self.row_lower[r] = scale.mul_add_c(rhs, self.row_lower[r]);
                }
                if self.row_upper[r] != INF {
                    self.row_upper[r] = scale.mul_add_c(rhs, self.row_upper[r]);
                }
                for k in 0..self.rp_len {
                    let pos = self.rowpositions[k] as usize;
                    let ac = self.a_col[pos];
                    let v = scale * self.a_value[pos];
                    self.add_to_matrix(row, ac, v);
                }
                self.reinsert_equation(row);
            }

            self.check_limits()?;
            self.remove_row_singletons()?;
            self.remove_doubleton_equations()?;
        }

        self.analysis.logging_on = logging_on;
        if logging_on {
            self.stop_rule_log(RULE_SPARSIFY);
        }
        Ok(())
    }

    pub(crate) fn strengthen_inequalities(&mut self, num_strengthened: &mut i32) -> R {
        let mut complementation: Vec<i8> = Vec::new();
        let mut reducedcost: Vec<f64> = Vec::new();
        let mut upper: Vec<f64> = Vec::new();
        let mut indices: Vec<i32> = Vec::new();
        let mut positions: Vec<i32> = Vec::new();
        let mut stack: Vec<i32> = Vec::new();
        let mut coefs: Vec<f64> = Vec::new();
        let mut cover: Vec<i32> = Vec::new();
        let feastol = self.primal_feastol;

        *num_strengthened = 0;
        const CHECK_TIME_FREQUENCY: i32 = 100;

        for row in 0..self.num_row {
            let r = row as usize;
            if self.rowsize[r] <= 1 {
                continue;
            }
            if self.is_ranged(row) {
                continue;
            }
            let rowsize_limit = 1000.max((self.num_col - self.num_deleted_cols) / 20);
            if self.rowsize[r] > rowsize_limit {
                continue;
            }

            let mut maxviolation: CDouble;
            let mut continuouscontribution = CDouble::from(0.0);
            let scale: f64;
            if self.row_lower[r] != -INF {
                maxviolation = CDouble::from(self.row_lower[r]);
                scale = -1.0;
            } else {
                maxviolation = CDouble::from(-self.row_upper[r]);
                scale = 1.0;
            }

            complementation.clear();
            reducedcost.clear();
            upper.clear();
            indices.clear();
            positions.clear();
            stack.push(self.rowroot[r]);

            let mut skiprow = false;
            while let Some(pos) = stack.pop() {
                let pu = pos as usize;
                if self.ar_right[pu] != -1 {
                    stack.push(self.ar_right[pu]);
                }
                if self.ar_left[pu] != -1 {
                    stack.push(self.ar_left[pu]);
                }
                let col = self.a_col[pu];
                let c = col as usize;
                skiprow = self.col_lower[c] == -INF || self.col_upper[c] == INF;
                if skiprow {
                    break;
                }
                let comp: i8;
                let mut weight = self.a_value[pu] * scale;
                let ub = self.col_upper[c] - self.col_lower[c];
                if weight > 0.0 {
                    comp = 1;
                    maxviolation += self.col_upper[c] * weight;
                } else {
                    comp = -1;
                    maxviolation += self.col_lower[c] * weight;
                    weight = -weight;
                }
                if ub <= feastol || weight <= feastol {
                    continue;
                }
                if self.integrality[c] == CONTINUOUS {
                    continuouscontribution += weight * ub;
                    continue;
                }
                indices.push(reducedcost.len() as i32);
                positions.push(pos);
                reducedcost.push(weight);
                complementation.push(comp);
                upper.push(ub);
            }

            if (row & CHECK_TIME_FREQUENCY) == 0 || 10 * self.rowsize[r] > rowsize_limit {
                self.check_time_limit()?;
            }

            if skiprow {
                stack.clear();
                continue;
            }

            if maxviolation <= feastol {
                self.row_presolve(row)?;
                continue;
            }

            let small_val = std_max(100.0 * feastol, feastol * maxviolation.to_f64());
            loop {
                if maxviolation - continuouscontribution <= small_val || indices.is_empty() {
                    break;
                }
                pdqsort(&mut indices, |&i1, &i2| {
                    let (a, b) = (reducedcost[i1 as usize], reducedcost[i2 as usize]);
                    // std::make_pair(rc[i1], i1) > std::make_pair(rc[i2], i2)
                    b < a || (!(a < b) && i2 < i1)
                });
                let mut lambda = maxviolation - continuouscontribution;
                cover.clear();
                cover.reserve(indices.len());
                for i in (1..=indices.len()).rev() {
                    let index = indices[i - 1] as usize;
                    let delta = upper[index] * reducedcost[index];
                    if upper[index] <= 1000.0 && reducedcost[index] > small_val && lambda - delta <= small_val {
                        cover.push(index as i32);
                    } else {
                        lambda -= delta;
                    }
                }
                if cover.is_empty() || lambda <= small_val {
                    break;
                }
                // std::min_element with the comparator
                let less = |i1: i32, i2: i32| -> bool {
                    let (a, b) = (reducedcost[i1 as usize], reducedcost[i2 as usize]);
                    if a <= 1e-3 || b <= 1e-3 {
                        return a > b;
                    }
                    a < b
                };
                let mut alpos = cover[0];
                for &ci in &cover[1..] {
                    if less(ci, alpos) {
                        alpos = ci;
                    }
                }
                let al = reducedcost[alpos as usize];
                coefs.resize(cover.len(), 0.0);
                let coverrhs = std_max((lambda / al - feastol).to_f64().ceil(), 1.0);
                let mut slackupper = CDouble::from(-coverrhs);
                let mut step = INF;
                for i in 0..cover.len() {
                    let ci = cover[i] as usize;
                    coefs[i] = (std_min(reducedcost[ci], lambda.to_f64()) / al - self.opt.small_matrix_value).ceil();
                    slackupper += upper[ci] * coefs[i];
                    step = std_min(step, reducedcost[ci] / coefs[i]);
                }
                step = std_min(step, (maxviolation / coverrhs).to_f64());
                maxviolation -= step * coverrhs;

                let slackind = reducedcost.len() as i32;
                reducedcost.push(step);
                upper.push(slackupper.to_f64());
                for i in 0..cover.len() {
                    let ci = cover[i] as usize;
                    reducedcost[ci] = (-step).mul_add_c(coefs[i], reducedcost[ci]);
                }
                indices.retain(|&i| !(reducedcost[i as usize] <= feastol));
                indices.push(slackind);
            }

            let threshold = (maxviolation + feastol).to_f64();
            indices.retain(|&i| !(i as usize >= positions.len() || reducedcost[i as usize].abs() <= threshold));
            if indices.is_empty() {
                continue;
            }

            let update_non_zeros = |s: &mut Self, rhs: &mut CDouble, direction: i32| {
                for &i in &indices {
                    let iu = i as usize;
                    let coefdelta = direction as f64 * (reducedcost[iu] - maxviolation).to_f64();
                    let col = s.a_col[positions[iu] as usize];
                    let c = col as usize;
                    if complementation[iu] == -1 {
                        *rhs += coefdelta * s.col_lower[c];
                        s.add_to_matrix(row, col, coefdelta);
                    } else {
                        *rhs -= coefdelta * s.col_upper[c];
                        s.add_to_matrix(row, col, -coefdelta);
                    }
                }
            };
            if scale < 0.0 {
                let mut lhs = CDouble::from(self.row_lower[r]);
                update_non_zeros(self, &mut lhs, -1);
                self.row_lower[r] = lhs.to_f64();
            } else {
                let mut rhs = CDouble::from(self.row_upper[r]);
                update_non_zeros(self, &mut rhs, 1);
                self.row_upper[r] = rhs.to_f64();
            }
            *num_strengthened += indices.len() as i32;
        }
        Ok(())
    }

    pub(crate) fn remove_dependent_equations(&mut self) -> R {
        let logging_on = self.analysis.logging_on;
        if self.equations.is_empty() {
            return Ok(());
        }
        if logging_on {
            self.start_rule_log(RULE_DEPENDENT_EQUATIONS);
        }
        let num_col = self.equations.len();
        let num_row = self.num_col + 1;
        let max_capacity = self.num_nonzeros() as usize + num_col;
        let mut start: Vec<i32> = Vec::with_capacity(num_col + 1);
        let mut value: Vec<f64> = Vec::with_capacity(max_capacity);
        let mut index: Vec<i32> = Vec::with_capacity(max_capacity);
        let mut eq_set: Vec<i32> = Vec::with_capacity(num_col);
        start.push(0);
        for &(_, eq) in &self.equations {
            eq_set.push(eq);
            for (ci, v) in row_iter!(self, eq) {
                value.push(v);
                index.push(ci);
            }
            if self.row_lower[eq as usize] != 0.0 {
                value.push(self.row_lower[eq as usize]);
                index.push(self.num_col);
            }
            start.push(value.len() as i32);
        }

        let time_limit = std_max(1.0, std_min(0.01 * self.opt.time_limit, 1000.0));
        let silent = self.silent_log();
        if !silent {
            let msg = crate::util::printf::sprintf(
                "Dependent equations search running on %d equations with time limit of %.2fs\n",
                &[(num_col as i32).into(), time_limit.into()],
            );
            self.log_user(LOG_INFO, &msg);
        }
        let (build_return, time_taken, var_with_no_pivot) =
            self.host.dependent_equations(num_col, num_row, &start, &index, &value, time_limit);
        if build_return == -1 {
            // kBuildKernelReturnTimeout
            if !silent {
                let msg = crate::util::printf::sprintf(
                    "Dependent equations search terminated after %.3gs due to expected time exceeding limit\n",
                    &[time_taken.into()],
                );
                self.log_user(LOG_INFO, &msg);
            }
            self.analysis.logging_on = logging_on;
            if logging_on {
                self.stop_rule_log(RULE_DEPENDENT_FREE_COLS);
            }
            return Ok(());
        } else {
            let pct_off_timeout = 1e2 * (time_taken - time_limit).abs() / time_limit;
            if !silent && pct_off_timeout < 1.0 {
                let msg = crate::util::printf::sprintf(
                    "Dependent equations search finished within %.2f%% of limit of %.2fs: risk of non-deterministic behaviour if solve is repeated\n",
                    &[pct_off_timeout.into(), time_limit.into()],
                );
                self.log_user(LOG_WARNING, &msg);
            }
        }
        let rank_deficiency = build_return;
        let mut num_removed_row = 0;
        let mut num_removed_nz = 0;
        let mut num_fictitious_rows_skipped = 0;
        for k in 0..rank_deficiency as usize {
            if var_with_no_pivot[k] >= 0 {
                let redundant_row = eq_set[var_with_no_pivot[k] as usize];
                num_removed_row += 1;
                num_removed_nz += self.rowsize[redundant_row as usize];
                self.ps.redundant_row(redundant_row);
                self.remove_row(redundant_row);
            } else {
                num_fictitious_rows_skipped += 1;
            }
        }
        if !silent {
            let msg = crate::util::printf::sprintf(
                "Dependent equations search removed %d rows and %d nonzeros in %.2fs (limit = %.2fs)\n",
                &[num_removed_row.into(), num_removed_nz.into(), time_taken.into(), time_limit.into()],
            );
            self.log_user(LOG_INFO, &msg);
        }
        if num_fictitious_rows_skipped != 0 {
            let msg = crate::util::printf::sprintf(", avoiding %d fictitious rows", &[num_fictitious_rows_skipped.into()]);
            self.log_dev(LOG_INFO, &msg);
        }
        self.log_dev(LOG_INFO, "\n");
        self.analysis.logging_on = logging_on;
        if logging_on {
            self.stop_rule_log(RULE_DEPENDENT_EQUATIONS);
        }
        Ok(())
    }
}

/// std::map::lower_bound on a sorted (key, value) vector
pub(crate) fn lower_bound(v: &[(f64, i32)], key: f64) -> usize {
    v.partition_point(|e| e.0 < key)
}

/// std::map::emplace on a sorted (key, value) vector (no-op if present)
pub(crate) fn map_emplace(v: &mut Vec<(f64, i32)>, key: f64, val: i32) {
    let i = lower_bound(v, key);
    if i < v.len() && !(key < v[i].0) {
        return;
    }
    v.insert(i, (key, val));
}
