//! The MIP reductions: dominated columns, probing (prepareProbing, the
//! probing loop of probing.rs, finaliseProbing, lifting for probing),
//! solution enumeration (enumeration.rs), the substitutions of the conflict graph, variable bounds,
//! implied integers and the scaling of MIP rows and columns.

use super::*;
use crate::mip::clique::CliqueVar;
use crate::mip::cuts::sort::pdqsort;
use crate::presolve::postsolve::EQ;
use crate::util::hash::HighsHash;
use std::collections::BTreeMap;

/// Counters of dominatedColumns
struct DomCtx {
    signatures: Vec<(u32, u32)>,
    num_dom_checks: usize,
    num_dom_checks_pred_bnd_analysis: usize,
    num_fixed_cols: i32,
    num_fixed_cols_pred_bnd_analysis: i32,
    num_modified_bnds_pred_bnd_analysis: i32,
}

impl Presolve<'_> {
    fn is_binary(&self, i: i32) -> bool {
        let c = i as usize;
        self.integrality[c] == INTEGER && self.col_lower[c] == 0.0 && self.col_upper[c] == 1.0
    }

    fn check_domination_non_zero(&self, row: i32, mut aj: f64, mut ak: f64) -> bool {
        let small = self.opt.small_matrix_value;
        if self.is_ranged(row) {
            return (aj - ak).abs() <= small;
        }
        if self.row_upper[row as usize] == INF {
            aj = -aj;
            ak = -ak;
        }
        !(aj > ak + small)
    }

    fn check_domination(&mut self, ctx: &mut DomCtx, scalj: i32, j: i32, scalk: i32, k: i32) -> bool {
        ctx.num_dom_checks += 1;
        let (ju, ku) = (j as usize, k as usize);
        if self.integrality[ju] == INTEGER && self.integrality[ku] == CONTINUOUS {
            return false;
        }
        let (mut sj_minus, mut sj_plus) = ctx.signatures[ju];
        if scalj == -1 {
            std::mem::swap(&mut sj_plus, &mut sj_minus);
        }
        let (mut sk_minus, mut sk_plus) = ctx.signatures[ku];
        if scalk == -1 {
            std::mem::swap(&mut sk_plus, &mut sk_minus);
        }
        if (!sj_minus & sk_minus) != 0 {
            return false;
        }
        if (sj_plus & !sk_plus) != 0 {
            return false;
        }
        if scalj as f64 * self.col_cost[ju]
            > (scalk as f64).mul_add_c(self.col_cost[ku], self.opt.small_matrix_value)
        {
            return false;
        }
        let mut p = self.colhead[ju];
        while p != -1 {
            let pu = p as usize;
            let row = self.a_row[pu];
            let v = self.a_value[pu];
            p = self.a_next[pu];
            let ak_pos = self.find_nonzero(row, k);
            let ak = if ak_pos == -1 { 0.0 } else { self.a_value[ak_pos as usize] };
            if !self.check_domination_non_zero(row, scalj as f64 * v, scalk as f64 * ak) {
                return false;
            }
        }
        let mut p = self.colhead[ku];
        while p != -1 {
            let pu = p as usize;
            let row = self.a_row[pu];
            let v = self.a_value[pu];
            p = self.a_next[pu];
            let aj_pos = self.find_nonzero(row, j);
            if aj_pos != -1 {
                continue;
            }
            if !self.check_domination_non_zero(row, 0.0, scalk as f64 * v) {
                return false;
            }
        }
        true
    }

    fn dom_fix_col(&mut self, ctx: &mut DomCtx, col: i32, direction: i32) -> R {
        ctx.num_fixed_cols += 1;
        if direction > 0 {
            self.fix_col_to_upper(col)?;
        } else {
            self.fix_col_to_lower(col)?;
        }
        self.remove_row_singletons()?;
        self.remove_doubleton_equations()?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn dom_tighten_bounds(
        &mut self,
        ctx: &mut DomCtx,
        col: i32,
        col_bound: f64,
        col_is_at_upper: bool,
        other_col: i32,
        other_col_bound: f64,
        pattern: i32,
    ) -> R {
        let c = col as usize;
        let feastol = self.primal_feastol;
        if self.col_lower[c] == self.col_upper[c] {
            return Ok(());
        }
        let mut lower_bound;
        let mut upper_bound;
        if col_is_at_upper {
            upper_bound = self.compute_implied_upper_bound(col, other_col, other_col_bound, pattern);
            lower_bound = std_min(col_bound, self.compute_implied_lower_bound(col, other_col, other_col_bound, pattern));
            if self.col_cost[c] <= 0.0 {
                let mut worst_case_upper = self.compute_worst_case_upper_bound(col, other_col, other_col_bound, pattern);
                if self.integrality[c] != CONTINUOUS {
                    worst_case_upper = (worst_case_upper + feastol).floor();
                }
                lower_bound = std_max(lower_bound, std_min(col_bound, worst_case_upper));
            }
        } else {
            lower_bound = self.compute_implied_lower_bound(col, other_col, other_col_bound, pattern);
            upper_bound = std_max(col_bound, self.compute_implied_upper_bound(col, other_col, other_col_bound, pattern));
            if self.col_cost[c] >= 0.0 {
                let mut worst_case_lower = self.compute_worst_case_lower_bound(col, other_col, other_col_bound, pattern);
                if self.integrality[c] != CONTINUOUS {
                    worst_case_lower = (worst_case_lower - feastol).ceil();
                }
                upper_bound = std_min(upper_bound, std_max(col_bound, worst_case_lower));
            }
        }
        if lower_bound > self.col_lower[c] + feastol {
            if self.integrality[c] != CONTINUOUS {
                lower_bound = (lower_bound - feastol).ceil();
            }
            if lower_bound == self.col_upper[c] {
                ctx.num_fixed_cols_pred_bnd_analysis += 1;
                self.dom_fix_col(ctx, col, 1)?;
            } else if self.integrality[c] != CONTINUOUS {
                ctx.num_modified_bnds_pred_bnd_analysis += 1;
                self.change_col_lower(col, lower_bound)?;
            }
        }
        if upper_bound < self.col_upper[c] - feastol {
            if self.integrality[c] != CONTINUOUS {
                upper_bound = (upper_bound + feastol).floor();
            }
            if upper_bound == self.col_lower[c] {
                ctx.num_fixed_cols_pred_bnd_analysis += 1;
                self.dom_fix_col(ctx, col, -1)?;
            } else if self.integrality[c] != CONTINUOUS {
                ctx.num_modified_bnds_pred_bnd_analysis += 1;
                self.change_col_upper(col, upper_bound)?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn dom_check_cols(
        &mut self,
        ctx: &mut DomCtx,
        col: i32,
        k: i32,
        direction: i32,
        multiplier: i32,
        bound_implied: bool,
        has_cliques: bool,
        same_var_type: bool,
    ) -> R {
        let c = col as usize;
        let ku = k as usize;
        let direction_k = multiplier * direction;
        let dominating_bound = if direction > 0 { self.col_upper[c] } else { self.col_lower[c] };
        let dominated_bound = if direction_k > 0 { self.col_lower[ku] } else { self.col_upper[ku] };
        let is_dominating_bound_finite = direction as f64 * dominating_bound != INF;
        let is_dominated_bound_finite = direction_k as f64 * dominated_bound != -INF;
        let try_to_fix = is_dominated_bound_finite && (bound_implied || has_cliques);
        let try_to_strengthen_bounds = (is_dominating_bound_finite || is_dominated_bound_finite) && same_var_type;
        if try_to_fix || try_to_strengthen_bounds {
            if !try_to_fix {
                ctx.num_dom_checks_pred_bnd_analysis += 1;
            }
            if self.check_domination(ctx, direction, col, direction_k, k) {
                let current_bound_implied =
                    if direction > 0 { self.is_upper_implied(col) } else { self.is_lower_implied(col) };
                if try_to_fix
                    && (current_bound_implied
                        || self.cliquetable().have_common_clique(
                            CliqueVar::new(col, (direction > 0) as i32),
                            CliqueVar::new(k, (direction_k > 0) as i32),
                        ))
                {
                    self.dom_fix_col(ctx, k, -direction_k)?;
                } else if try_to_strengthen_bounds {
                    if is_dominated_bound_finite {
                        self.dom_tighten_bounds(
                            ctx,
                            col,
                            dominating_bound,
                            direction > 0,
                            k,
                            dominated_bound,
                            direction * direction_k,
                        )?;
                    }
                    if self.col_deleted[c] == 0 && is_dominating_bound_finite {
                        self.dom_tighten_bounds(
                            ctx,
                            k,
                            dominated_bound,
                            direction_k < 0,
                            col,
                            dominating_bound,
                            direction * direction_k,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn dom_check_row(
        &mut self,
        ctx: &mut DomCtx,
        row: i32,
        col: i32,
        direction: i32,
        best_val: f64,
        bound_implied: bool,
        has_cliques: bool,
    ) -> R {
        self.store_row(row);
        let only_pred_bnd_analysis = !bound_implied && !has_cliques;
        let n = self.rp_len;
        let c = col as usize;
        let d = direction as f64;
        for kk in 0..n {
            let p = self.rowpositions[kk] as usize;
            let k = self.a_col[p];
            if k == col || self.col_deleted[k as usize] != 0 {
                continue;
            }
            let ak = self.a_value[p];
            let ku = k as usize;
            let same_var_type = (self.integrality[c] != CONTINUOUS && self.integrality[ku] != CONTINUOUS)
                || (self.integrality[c] != INTEGER && self.integrality[ku] != INTEGER);
            if only_pred_bnd_analysis && !same_var_type {
                continue;
            }
            if self.check_domination_non_zero(row, d * best_val, d * ak) {
                self.dom_check_cols(ctx, col, k, direction, 1, bound_implied, has_cliques, same_var_type)?;
                if self.col_deleted[c] != 0 {
                    break;
                }
            }
            if self.col_deleted[ku] == 0 && self.check_domination_non_zero(row, d * best_val, -d * ak) {
                self.dom_check_cols(ctx, col, k, direction, -1, bound_implied, has_cliques, same_var_type)?;
                if self.col_deleted[c] != 0 {
                    break;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn dominated_columns(&mut self) -> R {
        let mut ctx = DomCtx {
            signatures: vec![(0, 0); self.num_col as usize],
            num_dom_checks: 0,
            num_dom_checks_pred_bnd_analysis: 0,
            num_fixed_cols: 0,
            num_fixed_cols_pred_bnd_analysis: 0,
            num_modified_bnds_pred_bnd_analysis: 0,
        };
        for i in 0..self.a_value.len() {
            if self.a_value[i] == 0.0 {
                continue;
            }
            let row = self.a_row[i];
            let col = self.a_col[i] as usize;
            let lf = (self.row_lower[row as usize] != -INF) as u32;
            let uf = (self.row_upper[row as usize] != INF) as u32;
            let pos = (row.highs_hash() >> 59) as u32;
            let (a, b) = if self.a_value[i] > 0.0 { (lf, uf) } else { (uf, lf) };
            ctx.signatures[col].0 |= a << pos;
            ctx.signatures[col].1 |= b << pos;
        }

        let mut num_cols = 0i32;
        const MAX_AVG_CHECKS: usize = 10000;
        const MIN_AVG_REDS: f64 = 1e-2;
        let mut allow_pred_bnd_analysis = true;

        for j in 0..self.num_col {
            let ju = j as usize;
            if self.col_deleted[ju] != 0 {
                continue;
            }
            if (j & 127) == 0 {
                self.check_time_limit()?;
            }
            num_cols += 1;

            let mut best_row_plus = -1;
            let mut best_row_plus_len = IINF;
            let mut aj_best_row_plus = 0.0;
            let mut best_row_minus = -1;
            let mut best_row_minus_len = IINF;
            let mut aj_best_row_minus = 0.0;
            for (row, v) in col_iter!(self, j) {
                let r = row as usize;
                let scale = if self.row_upper[r] != INF { 1.0 } else { -1.0 };
                let val = scale * v;
                if val > 0.0 && self.rowsize[r] < best_row_plus_len {
                    best_row_plus = row;
                    best_row_plus_len = self.rowsize[r];
                    aj_best_row_plus = v;
                }
                if val < 0.0 && self.rowsize[r] < best_row_minus_len {
                    best_row_minus = row;
                    best_row_minus_len = self.rowsize[r];
                    aj_best_row_minus = v;
                }
            }

            let lower_implied = self.is_lower_implied(j);
            let upper_implied = self.is_upper_implied(j);
            let has_neg_cliques = self.is_binary(j) && self.cliquetable().num_cliques(CliqueVar::new(j, 0)) > 0;
            let has_pos_cliques = self.is_binary(j) && self.cliquetable().num_cliques(CliqueVar::new(j, 1)) > 0;

            if best_row_minus != -1 && (allow_pred_bnd_analysis || lower_implied || has_neg_cliques) {
                self.dom_check_row(&mut ctx, best_row_minus, j, -1, aj_best_row_minus, lower_implied, has_neg_cliques)?;
            }
            if self.col_deleted[ju] == 0
                && best_row_plus != -1
                && (allow_pred_bnd_analysis || upper_implied || has_pos_cliques)
            {
                self.dom_check_row(&mut ctx, best_row_plus, j, 1, aj_best_row_plus, upper_implied, has_pos_cliques)?;
            }

            let avg_checks = ctx.num_dom_checks_pred_bnd_analysis / num_cols as usize;
            let avg_reds = (ctx.num_fixed_cols_pred_bnd_analysis + ctx.num_modified_bnds_pred_bnd_analysis) as f64
                / num_cols as f64;
            allow_pred_bnd_analysis = allow_pred_bnd_analysis
                && (ctx.num_dom_checks_pred_bnd_analysis <= 30 * MAX_AVG_CHECKS
                    || (avg_checks <= MAX_AVG_CHECKS && avg_reds >= MIN_AVG_REDS));
        }
        let _ = ctx.num_dom_checks;

        if ctx.num_fixed_cols > 0 || ctx.num_modified_bnds_pred_bnd_analysis > 0 {
            let msg = crate::util::printf::sprintf(
                "Fixed %d dominated columns and strengthened %d bounds\n",
                &[ctx.num_fixed_cols.into(), ctx.num_modified_bnds_pred_bnd_analysis.into()],
            );
            self.log_dev(LOG_INFO, &msg);
        }
        Ok(())
    }

    // ------------------------------------------------------------- probing

    /// prepareProbing: returns firstCall
    pub(crate) fn prepare_probing(&mut self) -> Result<bool, Stop> {
        self.shrink_problem();
        self.to_csc_and_back();
        let nnz = self.num_nonzeros();
        self.cliquetable().set_max_entries(nnz);

        let huge_bound = self.primal_feastol / TINY;
        for i in 0..self.num_col {
            let iu = i as usize;
            if self.impl_col_lower[iu].abs() <= huge_bound && self.impl_col_lower[iu] > self.col_lower[iu] {
                self.change_col_lower(i, self.impl_col_lower[iu])?;
            }
            if self.impl_col_upper[iu].abs() <= huge_bound && self.impl_col_upper[iu] < self.col_upper[iu] {
                self.change_col_upper(i, self.impl_col_upper[iu])?;
            }
        }
        self.sync_model();
        let (infeasible, first_call) = self.host.probing_prepare(nnz, self.offset);
        if infeasible {
            return Err(Stop::PrimalInfeasible);
        }
        Ok(first_call)
    }

    /// finaliseProbing: (numVarsFixed, numBndsTightened, numVarsSubstituted,
    /// liftedNonZeros) added to the counters
    pub(crate) fn finalise_probing(&mut self, first_call: bool, counts: &mut [i32; 4]) -> R {
        let (deleted_rows, extensions) = self.host.finalise_begin(first_call);
        for &delrow in &deleted_rows {
            if self.row_deleted[delrow as usize] == 0 {
                self.remove_row(delrow);
            }
        }
        counts[3] += extensions.len() as i32;
        for &(row, col, val) in &extensions {
            let r = row as usize;
            if self.row_deleted[r] != 0 {
                counts[3] -= 1;
                continue;
            }
            let mut v = 1.0;
            if val == 0 {
                self.row_lower[r] -= 1.0;
                self.row_upper[r] -= 1.0;
                v = -1.0;
            }
            self.add_to_matrix(row, col, v);
        }

        let (dom_lower, dom_upper) = self.host.domain_bounds();
        for i in 0..self.num_col {
            let iu = i as usize;
            if self.col_deleted[iu] != 0 {
                continue;
            }
            let new_lower_bnd = self.col_lower[iu] < dom_lower[iu];
            let new_upper_bnd = self.col_upper[iu] > dom_upper[iu];
            if new_lower_bnd {
                self.change_col_lower(i, dom_lower[iu])?;
            }
            if new_upper_bnd {
                self.change_col_upper(i, dom_upper[iu])?;
            }
            if dom_lower[iu] == dom_upper[iu] {
                counts[0] += 1;
                self.ps.removed_fixed_col(i, self.col_lower[iu], 0.0, std::iter::empty());
                self.remove_fixed_col_logged(i);
            } else {
                if new_lower_bnd {
                    counts[1] += 1;
                }
                if new_upper_bnd {
                    counts[1] += 1;
                }
            }
            self.check_limits()?;
        }

        let mut num_del_col = 0;
        let r = self.apply_conflict_graph_substitutions(&mut num_del_col);
        counts[2] += num_del_col;
        r?;
        self.check_limits()
    }

    pub(crate) fn apply_conflict_graph_substitutions(&mut self, num_del_col: &mut i32) -> R {
        let subs = self.implications().substitutions.clone();
        for s in &subs {
            if self.col_deleted[s.substcol as usize] != 0 || self.col_deleted[s.staycol as usize] != 0 {
                continue;
            }
            *num_del_col += 1;
            let sc = s.substcol as usize;
            self.ps.doubleton_equation(
                -1,
                s.substcol,
                s.staycol,
                1.0,
                -s.scale,
                s.offset,
                self.col_lower[sc],
                self.col_upper[sc],
                0.0,
                false,
                false,
                EQ,
                std::iter::empty(),
            );
            self.mark_col_deleted(s.substcol);
            self.substitute_cols(s.substcol, s.staycol, s.offset, s.scale);
            self.check_limits()?;
        }
        self.implications().substitutions.clear();

        let subs = self.cliquetable().substitutions.clone();
        for s in &subs {
            let (substcol, rcol, rval) = (s.substcol, s.replace.col(), s.replace.val());
            if self.col_deleted[substcol as usize] != 0 || self.col_deleted[rcol as usize] != 0 {
                continue;
            }
            let (scale, offset) = if rval == 0 { (-1.0, 1.0) } else { (1.0, 0.0) };
            *num_del_col += 1;
            let sc = substcol as usize;
            self.ps.doubleton_equation(
                -1,
                substcol,
                rcol,
                1.0,
                -scale,
                offset,
                self.col_lower[sc],
                self.col_upper[sc],
                0.0,
                false,
                false,
                EQ,
                std::iter::empty(),
            );
            self.mark_col_deleted(substcol);
            self.substitute_cols(substcol, rcol, offset, scale);
            self.check_limits()?;
        }
        self.cliquetable().substitutions.clear();
        Ok(())
    }

    pub(crate) fn run_probing(&mut self) -> R {
        self.host.profiling(true, 0);
        self.probing_early_abort = false;
        let old_num_probed = self.num_probed;

        let first_call = match self.prepare_probing() {
            Ok(f) => f,
            Err(e) => {
                self.host.profiling(false, 0);
                return Err(e);
            }
        };

        if !self.probing_loop()? {
            // no binaries
            self.host.profiling(false, 0);
            return self.check_limits();
        }

        let mut counts = [0i32; 4];
        self.finalise_probing(first_call, &mut counts)?;
        self.probing_num_del_col += counts[2];

        let msg = crate::util::printf::sprintf(
            "%d probing evaluations: %d deleted rows, %d deleted columns, %d lifted nonzeros\n",
            &[
                (self.num_probed - old_num_probed).into(),
                self.num_deleted_rows.into(),
                self.num_deleted_cols.into(),
                counts[3].into(),
            ],
        );
        self.log_dev(LOG_INFO, &msg);

        if self.opt.mip_lifting_for_probing != -1 {
            if self.num_deleted_rows == 0 && self.num_deleted_cols == 0 && counts[3] == 0 {
                self.lifting_for_probing()?;
            }
            self.lifting.clear();
        }

        self.host.profiling(false, 0);
        self.check_limits()
    }

    fn lifting_for_probing(&mut self) -> R {
        // (row, clique of (col, val, coef), score, nfill)
        let mut liftingtable: Vec<(i32, Vec<((i32, i32), f64)>, f64, i32)> = Vec::with_capacity(self.lifting.len());
        let mut bestscoretotal = -INF;
        let fillallowed = self.opt.mip_lifting_for_probing > 0;
        let (dom_lower, dom_upper) = self.host.domain_bounds();
        let mut coefficients: BTreeMap<(i32, i32), (f64, i32)> = BTreeMap::new();
        let mut numrowsremoved = 0usize;
        let order: Vec<i32> = self.lifting.order.iter().rev().copied().collect();

        for &row in &order {
            let r = row as usize;
            if self.row_deleted[r] != 0 {
                continue;
            }
            let dense = self.rowsize[r] > 1000.max((self.num_col - self.num_deleted_cols) / 20);
            let tree = match self.lifting.trees[r].take() {
                Some(t) => t,
                None => continue,
            };
            let mut isredundant = false;
            let mut entries: Vec<((i32, i32), f64)> = Vec::new();
            tree.for_each(|k, &coef| entries.push((*k, coef)));
            for &((col, val), coef) in &entries {
                let pos = self.find_nonzero(row, col);
                isredundant = isredundant || tree.contains(&(col, 1 - val));
                let cu = col as usize;
                if !dense
                    && (fillallowed || pos != -1)
                    && self.col_deleted[cu] == 0
                    && dom_lower[cu] != dom_upper[cu]
                {
                    coefficients.insert((col, val), (coef, pos));
                }
            }
            self.lifting.trees[r] = Some(tree);

            if isredundant {
                numrowsremoved += 1;
                self.ps.redundant_row(row);
                self.remove_row(row);
                coefficients.clear();
                self.check_limits()?;
                continue;
            }
            if coefficients.is_empty() {
                continue;
            }

            let coeff_diff = |s: &Self, newvalue: f64, nzpos: i32| -> f64 {
                (newvalue - if nzpos == -1 { 0.0 } else { s.a_value[nzpos as usize] }).abs()
            };
            let mut bestclique: Vec<((i32, i32), f64)> = Vec::new();
            let mut bestscore = -INF;
            let mut bestnfill = 0;
            let mut candidates: Vec<(i32, i32)> = Vec::with_capacity(coefficients.len());
            for (&k, &(coef, pos)) in &coefficients {
                candidates.push(k);
                let score = coeff_diff(self, coef, pos);
                if score > bestscore {
                    bestscore = score;
                    bestnfill = if pos == -1 { 1 } else { 0 };
                    bestclique = vec![(k, coef)];
                }
            }
            if candidates.len() > 1 {
                let vars: Vec<CliqueVar> = candidates.iter().map(|&(c, v)| CliqueVar::new(c, v)).collect();
                let cliques: Vec<Vec<(i32, i32)>> = self
                    .cliquetable()
                    .compute_maximal_cliques(&vars, self.primal_feastol)
                    .iter()
                    .map(|c| c.iter().map(|v| (v.col(), v.val())).collect())
                    .collect();
                for clique in &cliques {
                    let mut score = CDouble::from(0.0);
                    let mut nfill = 0;
                    for cv in clique {
                        let (coef, pos) = *coefficients.entry(*cv).or_insert((0.0, 0));
                        score += coeff_diff(self, coef, pos);
                        if pos == -1 {
                            nfill += 1;
                        }
                    }
                    if score > bestscore {
                        bestscore = score.to_f64();
                        bestnfill = nfill;
                        bestclique.clear();
                        for cv in clique {
                            let coef = coefficients.entry(*cv).or_insert((0.0, 0)).0;
                            bestclique.push((*cv, coef));
                        }
                    }
                }
            }
            bestscoretotal = std_max(bestscoretotal, bestscore);
            liftingtable.push((row, bestclique, bestscore, bestnfill));
            coefficients.clear();
        }

        let overall = |score: f64, numelms: i32, numfillin: i32| -> f64 {
            let weight = 0.5;
            weight.mul_add_c(score / bestscoretotal, (1.0 - weight) * (numelms - numfillin) as f64 / numelms as f64)
        };
        let mut idx: Vec<usize> = (0..liftingtable.len()).collect();
        pdqsort(&mut idx, |&a, &b| {
            let o1 = &liftingtable[a];
            let o2 = &liftingtable[b];
            let s1 = overall(o1.2, o1.1.len() as i32, o1.3);
            let s2 = overall(o2.2, o2.1.len() as i32, o2.3);
            if s1 == s2 {
                o1.0 < o2.0
            } else {
                s1 > s2
            }
        });

        let mut nfill = 0usize;
        let mut nmod = 0usize;
        let mut numrowsmodified = 0usize;
        let maxnfill = (10 * liftingtable.len()).max(self.num_nonzeros() as usize / 100);
        for &li in &idx {
            let (row, ref bestclique, _, newfill) = liftingtable[li];
            let newfill = newfill as usize;
            if nfill + newfill > maxnfill {
                break;
            }
            nfill += newfill;
            nmod += bestclique.len() - newfill;
            let mut update = CDouble::from(0.0);
            for &((col, val), coeff) in bestclique {
                self.add_to_matrix(row, col, coeff);
                if val == 0 {
                    update += coeff;
                }
            }
            numrowsmodified += 1;
            let r = row as usize;
            if self.row_lower[r] != -INF {
                self.row_lower[r] += update.to_f64();
            }
            if self.row_upper[r] != INF {
                self.row_upper[r] += update.to_f64();
            }
        }

        let msg = crate::util::printf::sprintf(
            "Lifting for probing removed %d and modified %d row(s), added %d new and modified %d existing nonzero(s)\n",
            &[
                (numrowsremoved as i32).into(),
                (numrowsmodified as i32).into(),
                (nfill as i32).into(),
                (nmod as i32).into(),
            ],
        );
        self.log_dev(LOG_INFO, &msg);
        Ok(())
    }

    pub(crate) fn enumerate_solutions(&mut self) -> R {
        self.host.profiling(true, 1);
        let first_call = match self.prepare_probing() {
            Ok(f) => f,
            Err(e) => {
                self.host.profiling(false, 1);
                return Err(e);
            }
        };
        self.enumeration_loop()?;
        let mut counts = [0i32; 4];
        self.finalise_probing(first_call, &mut counts)?;
        if counts[0] > 0 || counts[1] > 0 || counts[2] > 0 {
            let msg = crate::util::printf::sprintf(
                "Enumeration presolve fixed %d columns, tightened %d bounds and performed %d substitutions\n",
                &[counts[0].into(), counts[1].into(), counts[2].into()],
            );
            self.log_dev(LOG_INFO, &msg);
        }
        self.host.profiling(false, 1);
        self.check_limits()
    }

    pub(crate) fn extract_var_bounds(&mut self, row: i32) {
        let r = row as usize;
        if self.mip.is_none() || self.rowsize[r] <= 1 || self.rowsize_integer[r] == 0 {
            return;
        }
        let num_inf_sum_lower = self.implied_row_bounds.num_inf_sum_lower(row);
        let num_inf_sum_upper = self.implied_row_bounds.num_inf_sum_upper(row);
        let mut use_lhs = self.row_lower[r] != -INF && num_inf_sum_upper <= 1;
        let mut use_rhs = self.row_upper[r] != INF && num_inf_sum_lower <= 1;
        if !use_lhs && !use_rhs {
            return;
        }
        let feastol = self.mip.expect("MIP presolve").feastol;
        let mut bin_col = -1;
        let mut bin_coef = 0.0;
        for (ci, v) in row_iter!(self, row) {
            let c = ci as usize;
            if self.col_lower[c] == self.col_upper[c] {
                continue;
            }
            if self.integrality[c] == INTEGER && self.col_lower[c] == 0.0 && self.col_upper[c] == 1.0 {
                if bin_col != -1 {
                    return;
                }
                bin_col = ci;
                bin_coef = v;
            }
        }
        if bin_col == -1 {
            return;
        }
        let mut w = PreOrder::new(self.rowroot[r]);
        while let Some(p) = w.next(&self.ar_left, &self.ar_right) {
            let ci = self.a_col[p];
            let v = self.a_value[p];
            let c = ci as usize;
            if self.col_lower[c] == self.col_upper[c] {
                continue;
            }
            if ci == bin_col {
                continue;
            }
            let mut vlb_constant = -INF;
            if use_lhs {
                let residual = self.implied_row_bounds.residual_sum_upper(row, ci, v, bin_col, bin_coef, 0.0, &rb!(self));
                if residual != INF {
                    vlb_constant = ((CDouble::from(self.row_lower[r]) - residual) / v.abs()).to_f64();
                    use_lhs = num_inf_sum_upper == 0;
                }
            }
            let mut vub_constant = INF;
            if use_rhs {
                let residual = self.implied_row_bounds.residual_sum_lower(row, ci, v, bin_col, bin_coef, 0.0, &rb!(self));
                if residual != -INF {
                    vub_constant = ((CDouble::from(self.row_upper[r]) - residual) / v.abs()).to_f64();
                    use_rhs = num_inf_sum_lower == 0;
                }
            }
            if v < 0.0 {
                vlb_constant *= -1.0;
                vub_constant *= -1.0;
                std::mem::swap(&mut vlb_constant, &mut vub_constant);
            }
            let vb_coef = -bin_coef / v;
            let is_int = self.integrality[c] != CONTINUOUS;
            if vlb_constant != -INF {
                self.implications().add_vlb(ci, bin_col, vb_coef, vlb_constant, self.col_lower[c], is_int, feastol);
            }
            if vub_constant != INF {
                self.implications().add_vub(ci, bin_col, vb_coef, vub_constant, self.col_upper[c], is_int, feastol);
            }
            if !use_lhs && !use_rhs {
                break;
            }
        }
    }

    pub(crate) fn detect_implied_integers(&mut self) -> R {
        for col in 0..self.num_col {
            self.convert_implied_integer(col, -1, false)?;
        }
        Ok(())
    }

    pub(crate) fn scale_mip(&mut self) -> R {
        for i in 0..self.num_row {
            let iu = i as usize;
            if self.row_deleted[iu] != 0
                || self.rowsize[iu] < 1
                || self.rowsize_integer[iu] + self.rowsize_impl_int[iu] == self.rowsize[iu]
            {
                continue;
            }
            self.store_row(i);
            let mut max_abs_val = 0.0f64;
            for j in 0..self.rp_len {
                let nz_pos = self.rowpositions[j] as usize;
                if self.integrality[self.a_col[nz_pos] as usize] != CONTINUOUS {
                    continue;
                }
                max_abs_val = std_max(self.a_value[nz_pos].abs(), max_abs_val);
            }
            let mut scale = (-max_abs_val.log2()).round().exp2();
            if scale == 1.0 {
                continue;
            }
            if self.row_upper[iu] == INF {
                scale = -scale;
            }
            self.scale_stored_row(i, scale, false);
        }
        for i in 0..self.num_col {
            let iu = i as usize;
            if self.col_deleted[iu] != 0 || self.colsize[iu] < 1 || self.integrality[iu] != CONTINUOUS {
                continue;
            }
            let mut max_abs_val = 0.0f64;
            for (_, v) in col_iter!(self, i) {
                max_abs_val = std_max(v.abs(), max_abs_val);
            }
            let scale = (-max_abs_val.log2()).round().exp2();
            if scale == 1.0 {
                continue;
            }
            self.transform_column(i, scale, 0.0)?;
        }
        Ok(())
    }
}
