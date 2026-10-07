//! Setup (okSetInput / okFromCSC), the presolve loop, run, shrinkProblem,
//! the transfers of the model to the C++ HighsLp, and the presolve rule
//! log.

use super::*;
use crate::util::printf::sprintf;

const RULE_NAMES: [&str; RULE_COUNT] = [
    "Empty row",
    "Singleton row",
    "Redundant row",
    "Empty column",
    "Fixed column",
    "Dominated col",
    "Forcing row",
    "Forcing col",
    "Free col substitution",
    "Doubleton equation",
    "Dependent equations",
    "Dependent free columns",
    "Aggregator",
    "Parallel rows and columns",
    "Sparsify",
    "Probing",
    "Enumeration",
    "Dual fixing",
    "Col stuffing",
    "Initial sweep",
];

/// The model handed over by the C++ (HighsLp after okSetInput)
pub struct Input<'a> {
    pub num_col: i32,
    pub num_row: i32,
    pub col_cost: &'a [f64],
    pub col_lower: &'a [f64],
    pub col_upper: &'a [f64],
    pub row_lower: &'a [f64],
    pub row_upper: &'a [f64],
    pub integrality: &'a [u8],
    pub offset: f64,
    pub maximize: bool,
    pub a_start: &'a [i32],
    pub a_index: &'a [i32],
    pub a_value: &'a [f64],
    pub orig_col_index: &'a [i32],
    pub orig_row_index: &'a [i32],
    pub stack_data_size: usize,
    pub num_reductions: usize,
    pub presolve_reduction_limit: i32,
    pub model_name: String,
}

impl<'h> Presolve<'h> {
    pub fn new(host: &'h Host, opt: Options, mip: Option<MipInfo>, inp: &Input) -> Self {
        let nc = inp.num_col as usize;
        let nr = inp.num_row as usize;
        let mut p = Presolve {
            host,
            opt,
            mip,
            primal_feastol: if mip.is_none() { opt.primal_feasibility_tolerance } else { opt.mip_feasibility_tolerance },
            num_col: inp.num_col,
            num_row: inp.num_row,
            col_cost: inp.col_cost.to_vec(),
            col_lower: inp.col_lower.to_vec(),
            col_upper: inp.col_upper.to_vec(),
            row_lower: inp.row_lower.to_vec(),
            row_upper: inp.row_upper.to_vec(),
            integrality: inp.integrality.to_vec(),
            offset: inp.offset,
            maximize: inp.maximize,
            model_name: inp.model_name.clone(),
            a_value: Vec::new(),
            a_row: Vec::new(),
            a_col: Vec::new(),
            colhead: Vec::new(),
            a_next: Vec::new(),
            a_prev: Vec::new(),
            rowroot: Vec::new(),
            ar_left: Vec::new(),
            ar_right: Vec::new(),
            rowsize: Vec::new(),
            rowsize_integer: Vec::new(),
            rowsize_impl_int: Vec::new(),
            colsize: Vec::new(),
            rowpositions: Vec::new(),
            rp_len: 0,
            freeslots: Vec::new(),
            impl_col_lower: vec![-INF; nc],
            impl_col_upper: vec![INF; nc],
            col_lower_source: vec![-1; nc],
            col_upper_source: vec![-1; nc],
            row_dual_lower: vec![-INF; nr],
            row_dual_upper: vec![INF; nr],
            impl_row_dual_lower: vec![-INF; nr],
            impl_row_dual_upper: vec![INF; nr],
            row_dual_lower_source: vec![-1; nr],
            row_dual_upper_source: vec![-1; nr],
            col_impl_source_by_row: vec![Default::default(); nr],
            impl_row_dual_source_by_col: vec![Default::default(); nc],
            implied_row_bounds: Default::default(),
            implied_dual_row_bounds: Default::default(),
            changed_row_indices: Vec::with_capacity(nr),
            changed_row_flag: vec![1; nr],
            changed_col_indices: Vec::with_capacity(nc),
            changed_col_flag: vec![1; nc],
            substitution_opportunities: Vec::new(),
            lifting: Lifting::default(),
            equations: Default::default(),
            eqsize: Vec::new(),
            shrink_problem_enabled: true,
            reduction_limit: usize::MAX,
            singleton_rows: Vec::new(),
            singleton_columns: Vec::new(),
            row_deleted: vec![0; nr],
            col_deleted: vec![0; nc],
            single_equation_checked: vec![0; nr],
            num_probes: Vec::new(),
            probing_contingent: 0,
            probing_num_del_col: 0,
            num_probed: 0,
            num_deleted_rows: 0,
            num_deleted_cols: 0,
            old_num_col: 0,
            old_num_row: 0,
            probing_early_abort: false,
            presolve_status: PS_NOT_SET,
            analysis: Analysis::default(),
            ps: Recorder {
                orig_col_index: inp.orig_col_index.to_vec(),
                orig_row_index: inp.orig_row_index.to_vec(),
                base: inp.stack_data_size,
                num_reductions: inp.num_reductions,
                ..Default::default()
            },
        };
        if mip.is_some() {
            p.probing_contingent = 1000;
            p.probing_num_del_col = 0;
            p.num_probed = 0;
            p.num_probes = vec![0; nc];
        }
        for i in 0..nr {
            if p.row_lower[i] == -INF {
                p.row_dual_upper[i] = 0.0;
            }
            if p.row_upper[i] == INF {
                p.row_dual_lower[i] = 0.0;
            }
        }
        p.from_csc(inp.a_start, inp.a_index, inp.a_value);

        for row in 0..p.num_row {
            if !p.is_dual_implied_free(row) {
                continue;
            }
            for (ci, _) in row_iter!(p, row) {
                if p.is_implied_free(ci) {
                    p.substitution_opportunities.push((row, ci));
                }
            }
        }
        p.reduction_limit =
            if inp.presolve_reduction_limit < 0 { usize::MAX } else { inp.presolve_reduction_limit as usize };
        if !p.opt.presolve_off && p.reduction_limit < usize::MAX {
            let msg = sprintf("HPresolve::okSetInput reductionLimit = %d\n", &[(p.reduction_limit as i32).into()]);
            p.log_dev(LOG_INFO, &msg);
        }
        p
    }

    /// okFromCSC
    pub(crate) fn from_csc(&mut self, start: &[i32], index: &[i32], value: &[f64]) {
        let nc = self.num_col as usize;
        let nr = self.num_row as usize;
        self.a_value.clear();
        self.a_col.clear();
        self.a_row.clear();
        self.freeslots.clear();
        self.colhead = vec![-1; nc];
        self.rowroot = vec![-1; nr];
        self.colsize = vec![0; nc];
        self.rowsize = vec![0; nr];
        self.rowsize_integer = vec![0; nr];
        self.rowsize_impl_int = vec![0; nr];
        self.implied_row_bounds = Default::default();
        self.implied_dual_row_bounds = Default::default();
        self.implied_row_bounds.set_num_sums(nr);
        self.implied_dual_row_bounds.set_num_sums(nc);

        let nnz = value.len();
        self.a_value.extend_from_slice(value);
        self.a_col.reserve(nnz);
        self.a_row.reserve(nnz);
        for i in 0..nc {
            let (s, e) = (start[i] as usize, start[i + 1] as usize);
            self.a_col.extend(std::iter::repeat(i as i32).take(e - s));
            self.a_row.extend_from_slice(&index[s..e]);
        }
        self.a_next = vec![0; nnz];
        self.a_prev = vec![0; nnz];
        self.ar_left = vec![0; nnz];
        self.ar_right = vec![0; nnz];
        for pos in 0..nnz {
            self.link(pos as i32);
        }
        if self.equations.is_empty() {
            self.eqsize = vec![-1; nr];
            for i in 0..self.num_row {
                if self.is_equation(i) {
                    self.eqsize[i as usize] = self.rowsize[i as usize];
                    self.equations.insert((self.rowsize[i as usize], i));
                }
            }
        }
    }

    /// toCSC into the C++ model's matrix: the CSC arrays
    pub(crate) fn to_csc(&mut self) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
        let numcol = self.colsize.len();
        let mut start = vec![0i32; numcol + 1];
        let mut nnz = 0i32;
        for i in 0..numcol {
            start[i] = nnz;
            nnz += self.colsize[i];
        }
        start[numcol] = nnz;
        let mut aval = vec![0.0f64; nnz as usize];
        let mut aindex = vec![0i32; nnz as usize];
        for i in 0..self.a_value.len() {
            if self.a_value[i] == 0.0 {
                continue;
            }
            let c = self.a_col[i] as usize;
            let pos = (start[c + 1] - self.colsize[c]) as usize;
            self.colsize[c] -= 1;
            aval[pos] = self.a_value[i];
            aindex[pos] = self.a_row[i];
        }
        self.host.set_matrix(&start, &aindex, &aval);
        (start, aindex, aval)
    }

    /// toCSC followed by okFromCSC
    pub(crate) fn to_csc_and_back(&mut self) {
        let (s, i, v) = self.to_csc();
        self.from_csc(&s, &i, &v);
    }

    /// Writes the model to the C++ HighsLp (costs, bounds, integrality,
    /// offset, sense and dimensions) and flushes the postsolve records
    pub(crate) fn sync_model(&mut self) {
        self.flush();
        self.host.sync_model(self);
    }

    /// Appends the buffered reductions to the C++ postsolve stack
    pub(crate) fn flush(&mut self) {
        if self.ps.reductions.is_empty() && self.ps.not_transformable.is_empty() {
            return;
        }
        self.host.flush(&self.ps.buf, &self.ps.reductions, &self.ps.not_transformable);
        self.ps.base += self.ps.buf.len();
        self.ps.buf.clear();
        self.ps.reductions.clear();
        self.ps.not_transformable.clear();
    }

    pub(crate) fn shrink_problem(&mut self) {
        let old_num_col = self.num_col as usize;
        self.num_col = 0;
        let mut new_col_index = vec![0i32; old_num_col];
        for i in 0..old_num_col {
            if self.col_deleted[i] != 0 {
                new_col_index[i] = -1;
            } else {
                new_col_index[i] = self.num_col;
                self.num_col += 1;
                let n = new_col_index[i] as usize;
                if n < i {
                    self.col_cost[n] = self.col_cost[i];
                    self.col_lower[n] = self.col_lower[i];
                    self.col_upper[n] = self.col_upper[i];
                    self.integrality[n] = self.integrality[i];
                    self.impl_col_lower[n] = self.impl_col_lower[i];
                    self.impl_col_upper[n] = self.impl_col_upper[i];
                    self.col_lower_source[n] = self.col_lower_source[i];
                    self.col_upper_source[n] = self.col_upper_source[i];
                    self.impl_row_dual_source_by_col[n] = self.impl_row_dual_source_by_col[i].clone();
                    self.colhead[n] = self.colhead[i];
                    self.colsize[n] = self.colsize[i];
                    self.changed_col_flag[n] = self.changed_col_flag[i];
                }
            }
        }
        let nc = self.num_col as usize;
        self.col_deleted = vec![0; nc];
        self.col_cost.truncate(nc);
        self.col_lower.truncate(nc);
        self.col_upper.truncate(nc);
        self.integrality.truncate(nc);
        self.impl_col_lower.truncate(nc);
        self.impl_col_upper.truncate(nc);
        self.col_lower_source.truncate(nc);
        self.col_upper_source.truncate(nc);
        self.impl_row_dual_source_by_col.truncate(nc);
        self.colhead.truncate(nc);
        self.colsize.truncate(nc);
        self.changed_col_flag.truncate(nc);
        self.num_deleted_cols = 0;

        let old_num_row = self.num_row as usize;
        self.num_row = 0;
        let mut new_row_index = vec![0i32; old_num_row];
        for i in 0..old_num_row {
            if self.row_deleted[i] != 0 {
                new_row_index[i] = -1;
            } else {
                new_row_index[i] = self.num_row;
                self.num_row += 1;
                let n = new_row_index[i] as usize;
                if n < i {
                    self.row_lower[n] = self.row_lower[i];
                    self.row_upper[n] = self.row_upper[i];
                    self.row_dual_lower[n] = self.row_dual_lower[i];
                    self.row_dual_upper[n] = self.row_dual_upper[i];
                    self.impl_row_dual_lower[n] = self.impl_row_dual_lower[i];
                    self.impl_row_dual_upper[n] = self.impl_row_dual_upper[i];
                    self.row_dual_lower_source[n] = self.row_dual_lower_source[i];
                    self.row_dual_upper_source[n] = self.row_dual_upper_source[i];
                    self.col_impl_source_by_row[n] = self.col_impl_source_by_row[i].clone();
                    self.rowroot[n] = self.rowroot[i];
                    self.rowsize[n] = self.rowsize[i];
                    self.rowsize_integer[n] = self.rowsize_integer[i];
                    self.rowsize_impl_int[n] = self.rowsize_impl_int[i];
                    self.changed_row_flag[n] = self.changed_row_flag[i];
                    self.single_equation_checked[n] = self.single_equation_checked[i];
                }
            }
        }
        for i in 0..nc {
            if self.col_lower_source[i] != -1 {
                self.col_lower_source[i] = new_row_index[self.col_lower_source[i] as usize];
            }
            if self.col_upper_source[i] != -1 {
                self.col_upper_source[i] = new_row_index[self.col_upper_source[i] as usize];
            }
        }
        let nr = self.num_row as usize;
        for i in 0..nr {
            if self.row_dual_lower_source[i] != -1 {
                self.row_dual_lower_source[i] = new_col_index[self.row_dual_lower_source[i] as usize];
            }
            if self.row_dual_upper_source[i] != -1 {
                self.row_dual_upper_source[i] = new_col_index[self.row_dual_upper_source[i] as usize];
            }
        }
        for i in 0..nc {
            let set = std::mem::take(&mut self.impl_row_dual_source_by_col[i]);
            self.impl_row_dual_source_by_col[i] =
                set.iter().filter(|&&r| new_row_index[r as usize] != -1).map(|&r| new_row_index[r as usize]).collect();
        }
        for i in 0..nr {
            let set = std::mem::take(&mut self.col_impl_source_by_row[i]);
            self.col_impl_source_by_row[i] =
                set.iter().filter(|&&c| new_col_index[c as usize] != -1).map(|&c| new_col_index[c as usize]).collect();
        }
        self.row_deleted = vec![0; nr];
        self.row_lower.truncate(nr);
        self.row_upper.truncate(nr);
        self.row_dual_lower.truncate(nr);
        self.row_dual_upper.truncate(nr);
        self.impl_row_dual_lower.truncate(nr);
        self.impl_row_dual_upper.truncate(nr);
        self.row_dual_lower_source.truncate(nr);
        self.row_dual_upper_source.truncate(nr);
        self.col_impl_source_by_row.truncate(nr);
        self.rowroot.truncate(nr);
        self.rowsize.truncate(nr);
        self.rowsize_integer.truncate(nr);
        self.rowsize_impl_int.truncate(nr);
        self.changed_row_flag.truncate(nr);
        self.single_equation_checked.truncate(nr);
        self.num_deleted_rows = 0;

        // postsolve stack: the records so far, then the index maps
        self.flush();
        let (r, c) = crate::presolve::postsolve::compress_index_maps(
            &mut self.ps.orig_row_index,
            &mut self.ps.orig_col_index,
            &new_row_index,
            &new_col_index,
        );
        self.ps.orig_row_index.truncate(r);
        self.ps.orig_col_index.truncate(c);

        self.implied_row_bounds.shrink(&new_row_index, nr);
        self.implied_dual_row_bounds.shrink(&new_col_index, nc);

        for i in 0..self.a_value.len() {
            if self.a_value[i] == 0.0 {
                continue;
            }
            self.a_col[i] = new_col_index[self.a_col[i] as usize];
            self.a_row[i] = new_row_index[self.a_row[i] as usize];
        }

        for x in self.singleton_columns.iter_mut() {
            *x = new_col_index[*x as usize];
        }
        self.singleton_columns.retain(|&x| x != -1);
        for x in self.changed_col_indices.iter_mut() {
            *x = new_col_index[*x as usize];
        }
        self.changed_col_indices.retain(|&x| x != -1);
        for x in self.singleton_rows.iter_mut() {
            *x = new_row_index[*x as usize];
        }
        self.singleton_rows.retain(|&x| x != -1);
        for x in self.changed_row_indices.iter_mut() {
            *x = new_row_index[*x as usize];
        }
        self.changed_row_indices.retain(|&x| x != -1);

        for p in self.substitution_opportunities.iter_mut() {
            if p.0 == -1 {
                continue;
            }
            p.0 = new_row_index[p.0 as usize];
            p.1 = new_col_index[p.1 as usize];
        }
        self.substitution_opportunities.retain(|p| p.0 != -1 && p.1 != -1);

        self.equations.clear();
        self.eqsize = vec![-1; nr];
        for i in 0..self.num_row {
            if self.is_equation(i) {
                self.eqsize[i as usize] = self.rowsize[i as usize];
                self.equations.insert((self.rowsize[i as usize], i));
            }
        }

        if self.mip.is_some() {
            for i in 0..old_num_col {
                if new_col_index[i] != -1 {
                    self.num_probes[new_col_index[i] as usize] = self.num_probes[i];
                }
            }
            self.num_probes.truncate(nc);
        }

        // C++: the names, the C++ postsolve index maps, the MIP solver's
        // data structures and the matrix dimensions
        self.host.sync_model(self);
        self.host.shrink(&new_col_index, &new_row_index);

        // analysis_.resetNumDeleted()
        self.analysis.num_deleted_rows0 = 0;
        self.analysis.num_deleted_cols0 = 0;
    }

    pub(crate) fn initial_row_and_col_presolve(&mut self) -> R {
        for row in 0..self.num_row {
            if self.row_deleted[row as usize] != 0 {
                continue;
            }
            self.row_presolve(row)?;
            self.changed_row_flag[row as usize] = 0;
        }
        for col in 0..self.num_col {
            let c = col as usize;
            if self.col_deleted[c] != 0 {
                continue;
            }
            if self.integrality[c] != CONTINUOUS {
                self.change_col_bounds(col, self.col_lower[c], self.col_upper[c])?;
            }
            self.col_presolve(col)?;
            self.changed_col_flag[c] = 0;
        }
        self.check_limits()
    }

    pub(crate) fn fast_presolve_loop(&mut self) -> R {
        loop {
            self.store_current_problem_size();
            self.remove_row_singletons()?;
            self.presolve_changed_rows()?;
            self.remove_doubleton_equations()?;
            self.presolve_col_singletons()?;
            self.presolve_changed_cols()?;
            if !(self.problem_size_reduction() > 0.01) {
                break;
            }
        }
        Ok(())
    }

    fn report(&self, silent: bool) {
        if silent {
            return;
        }
        let num_col = self.num_col - self.num_deleted_cols;
        let num_row = self.num_row - self.num_deleted_rows;
        let num_nonz = self.num_nonzeros();
        let time_str = if self.opt.output_flag && !self.opt.timeless_log {
            self.host.time_string(self.host.timer_read())
        } else if self.opt.timeless_log {
            String::new()
        } else {
            self.host.time_string(0.0)
        };
        let msg = sprintf("%d rows, %d cols, %d nonzeros %s\n", &[num_row.into(), num_col.into(), num_nonz.into(), (&time_str).into()]);
        self.log_user(LOG_INFO, &msg);
    }

    fn presolve_return(&mut self) -> R {
        if self.mip.is_some() {
            self.scale_mip()?;
        }
        self.analyse_presolve_rule_log(true);
        Ok(())
    }

    pub(crate) fn presolve(&mut self) -> R {
        if self.maximize {
            for c in self.col_cost.iter_mut() {
                *c = -*c;
            }
            self.offset = -self.offset;
            self.maximize = false;
        }
        let silent = self.silent_log();
        if !self.opt.presolve_off && !silent {
            self.log_user(LOG_INFO, "Presolving model\n");
        }
        self.analysis_setup(silent);

        if !self.opt.presolve_off {
            if self.mip.is_some() {
                self.cliquetable().in_presolve = true;
            }
            if self.opt.presolve_rule_test != 0 {
                self.presolve_rule_test()?;
                return self.presolve_return();
            }
            self.initial_row_and_col_presolve()?;

            let mut num_parallel_row_col_calls = 0;
            let mut try_sparsify = self.mip.is_some() || !self.opt.lp_presolve_requires_basis_postsolve;
            let mut try_probing = self.mip.is_some();
            let mut num_cliques_before_probing = -1;
            let mut domcol_after_probing_called = false;
            let mut dependent_equations_called = self.mip.is_some();
            let mut last_print_size = IINF;

            loop {
                let curr_size = self.num_col - self.num_deleted_cols + self.num_row - self.num_deleted_rows;
                if (curr_size as f64) < 0.85 * last_print_size as f64 {
                    last_print_size = curr_size;
                    self.report(silent);
                }

                self.fast_presolve_loop()?;
                self.store_current_problem_size();

                if self.mip.is_some() {
                    let mut num_del_col = 0;
                    self.apply_conflict_graph_substitutions(&mut num_del_col)?;
                }

                if self.analysis.allow_rule[RULE_AGGREGATOR] {
                    self.aggregator()?;
                }

                if self.problem_size_reduction() > 0.05 {
                    continue;
                }

                if try_sparsify && self.analysis.allow_rule[RULE_SPARSIFY] {
                    let num_nz = self.num_nonzeros();
                    self.sparsify()?;
                    let nz_reduction = 100.0 * (1.0 - (self.num_nonzeros() as f64 / num_nz as f64));
                    if nz_reduction > 0.0 {
                        let msg = sprintf("Sparsify removed %.1f%% of nonzeros\n", &[nz_reduction.into()]);
                        self.log_dev(LOG_INFO, &msg);
                        self.fast_presolve_loop()?;
                    }
                    try_sparsify = false;
                }

                if self.analysis.allow_rule[RULE_PARALLEL_ROWS_AND_COLS] && num_parallel_row_col_calls < 5 {
                    if self.shrink_problem_enabled
                        && (self.num_deleted_cols >= self.num_col / 2 || self.num_deleted_rows >= self.num_row / 2)
                    {
                        self.shrink_problem();
                        self.to_csc_and_back();
                    }
                    self.store_current_problem_size();
                    self.detect_parallel_rows_and_cols()?;
                    num_parallel_row_col_calls += 1;
                    if self.problem_size_reduction() > 0.05 {
                        continue;
                    }
                }

                self.fast_presolve_loop()?;

                if self.mip.is_some() {
                    let mut num_strengthened = -1;
                    self.strengthen_inequalities(&mut num_strengthened)?;
                    if num_strengthened > 0 {
                        let msg = sprintf("Strengthened %d coefficients\n", &[num_strengthened.into()]);
                        self.log_dev(LOG_INFO, &msg);
                    }
                }

                self.fast_presolve_loop()?;

                if self.mip.is_some() && num_cliques_before_probing == -1 {
                    num_cliques_before_probing = self.cliquetable().num_cliques_total();
                    self.store_current_problem_size();
                    self.dominated_columns()?;
                    if self.problem_size_reduction() > 0.0 {
                        self.fast_presolve_loop()?;
                    }
                    if self.problem_size_reduction() > 0.05 {
                        continue;
                    }
                }

                if self.mip.is_some() && self.analysis.allow_rule[RULE_ENUMERATION] {
                    self.store_current_problem_size();
                    self.enumerate_solutions()?;
                    if self.problem_size_reduction() > 0.05 {
                        continue;
                    }
                }

                if try_probing && self.analysis.allow_rule[RULE_PROBING] {
                    self.detect_implied_integers()?;
                    self.store_current_problem_size();
                    self.run_probing()?;
                    try_probing = self.probing_contingent > self.num_probed as i64
                        && (self.problem_size_reduction() > 1.0 || self.probing_early_abort);
                    try_sparsify = true;
                    if self.problem_size_reduction() > 0.05 || try_probing {
                        continue;
                    }
                    self.fast_presolve_loop()?;
                }

                if !dependent_equations_called {
                    if self.shrink_problem_enabled
                        && (self.num_deleted_cols >= self.num_col / 2 || self.num_deleted_rows >= self.num_row / 2)
                    {
                        self.shrink_problem();
                        self.to_csc_and_back();
                    }
                    self.store_current_problem_size();
                    if self.analysis.allow_rule[RULE_DEPENDENT_EQUATIONS] {
                        self.remove_dependent_equations()?;
                        dependent_equations_called = true;
                    }
                    if self.problem_size_reduction() > 0.05 {
                        continue;
                    }
                }

                if self.mip.is_some()
                    && self.cliquetable().num_cliques_total() > num_cliques_before_probing
                    && !domcol_after_probing_called
                {
                    domcol_after_probing_called = true;
                    self.store_current_problem_size();
                    self.dominated_columns()?;
                    if self.problem_size_reduction() > 0.0 {
                        self.fast_presolve_loop()?;
                    }
                    if self.problem_size_reduction() > 0.05 {
                        continue;
                    }
                }
                break;
            }

            if self.opt.presolve_remove_slacks {
                self.remove_slacks()?;
            }
            self.report(silent);
        } else {
            self.log_user(LOG_INFO, "\nPresolve is switched off\n");
        }
        self.presolve_return()
    }

    fn presolve_rule_test(&mut self) -> R {
        if self.opt.presolve_rule_test as usize == RULE_COL_STUFFING {
            self.log_user(LOG_INFO, "HPresolve::presolveRuleTestColStuffing\n");
            for col in 0..self.num_col {
                if self.col_deleted[col as usize] != 0 {
                    continue;
                }
                self.singleton_col_stuffing(col)?;
            }
            let msg = sprintf(
                "HPresolve::presolveRuleTestColStuffing: Stuffing removed %d rows and %d columns\n",
                &[self.num_deleted_rows.into(), self.num_deleted_cols.into()],
            );
            self.log_user(LOG_INFO, &msg);
            return self.row_presolve(0);
        }
        Ok(())
    }

    /// run: returns the HighsModelStatus
    pub fn run(&mut self) -> i32 {
        self.presolve_status = PS_NOT_SET;
        self.shrink_problem_enabled = true;
        let report_reductions = |s: &Self| {
            if !s.opt.presolve_off && s.reduction_limit < usize::MAX {
                let msg = sprintf(
                    "Presolve performed %ld of %ld permitted reductions\n",
                    &[(s.ps.num_reductions as i64).into(), (s.reduction_limit as i64).into()],
                );
                s.log_user(LOG_INFO, &msg);
            }
        };
        match self.presolve() {
            Ok(()) | Err(Stop::Stopped) => {}
            Err(Stop::PrimalInfeasible) => {
                self.presolve_status = PS_INFEASIBLE;
                report_reductions(self);
                self.sync_model();
                return MS_INFEASIBLE;
            }
            Err(Stop::DualInfeasible) => {
                self.presolve_status = PS_UNBOUNDED_OR_INFEASIBLE;
                report_reductions(self);
                self.sync_model();
                return MS_UNBOUNDED_OR_INFEASIBLE;
            }
        }
        report_reductions(self);

        self.shrink_problem();

        if let Some(m) = self.mip {
            let nnz = self.num_nonzeros();
            self.host.mip_finish_presolve(nnz);
            if m.num_restarts != 0 {
                let mut cutinds: Vec<i32> = Vec::with_capacity(self.num_col as usize);
                let mut cutvals: Vec<f64> = Vec::with_capacity(self.num_col as usize);
                let mut numcuts = 0;
                let mut i = self.num_row - 1;
                while i >= 0 {
                    if self.ps.orig_row_index[i as usize] < m.orig_num_row {
                        break;
                    }
                    numcuts += 1;
                    self.store_row(i);
                    cutinds.clear();
                    cutvals.clear();
                    for k in 0..self.rp_len {
                        let j = self.rowpositions[k] as usize;
                        cutinds.push(self.a_col[j]);
                        cutvals.push(self.a_value[j]);
                    }
                    let iu = i as usize;
                    let integral =
                        self.rowsize_integer[iu] + self.rowsize_impl_int[iu] == self.rowsize[iu]
                            && self.row_coefficients_integral(i, 1.0);
                    self.host.add_cut(&cutinds, &cutvals, self.row_upper[iu], integral);
                    self.mark_row_deleted(i);
                    for k in 0..self.rp_len {
                        let j = self.rowpositions[k];
                        self.unlink(j);
                    }
                    i -= 1;
                }
                self.num_row -= numcuts;
                self.row_lower.truncate(self.num_row as usize);
                self.row_upper.truncate(self.num_row as usize);
            }
        }

        self.to_csc();

        let status;
        if self.num_col == 0 {
            if self.mip.is_some() {
                if self.offset > self.host.upper_limit() {
                    self.presolve_status = PS_INFEASIBLE;
                    self.sync_model();
                    return MS_INFEASIBLE;
                }
                self.host.set_lower_bound_zero();
            } else {
                let num_row_ok = self.num_row == 0 || self.ps.num_reductions >= self.reduction_limit;
                if !num_row_ok {
                    self.presolve_status = PS_NOT_PRESOLVED;
                    self.sync_model();
                    return MS_NOTSET;
                }
            }
            self.presolve_status = PS_REDUCED_TO_EMPTY;
            status = if self.zero_row_activity_feasible() { MS_OPTIMAL } else { MS_INFEASIBLE };
            self.sync_model();
            return status;
        } else if self.ps.num_reductions > 0 {
            self.presolve_status = PS_REDUCED;
        } else {
            self.presolve_status = PS_NOT_REDUCED;
        }

        if self.mip.is_none() && self.opt.use_implied_bounds_from_presolve {
            self.set_relaxed_implied_bounds();
        }
        self.sync_model();
        MS_NOTSET
    }

    fn zero_row_activity_feasible(&self) -> bool {
        for i in 0..self.num_row as usize {
            if self.row_lower[i] > self.primal_feastol || self.row_upper[i] < -self.primal_feastol {
                return false;
            }
        }
        true
    }

    fn set_relaxed_implied_bounds(&mut self) {
        let huge_bound = self.primal_feastol / TINY;
        for i in 0..self.num_col {
            let iu = i as usize;
            if self.col_lower[iu] >= self.impl_col_lower[iu] && self.col_upper[iu] <= self.impl_col_upper[iu] {
                continue;
            }
            if self.impl_col_lower[iu].abs() <= huge_bound {
                let nz_pos = self.find_nonzero(self.col_lower_source[iu], i);
                let bound_relax = std_max(1000.0, self.impl_col_lower[iu].abs()) * self.primal_feastol
                    / std_min(1.0, self.a_value[nz_pos as usize].abs());
                let new_lb = self.impl_col_lower[iu] - bound_relax;
                if new_lb > self.col_lower[iu] + bound_relax {
                    self.col_lower[iu] = new_lb;
                }
            }
            if self.impl_col_upper[iu].abs() <= huge_bound {
                let nz_pos = self.find_nonzero(self.col_upper_source[iu], i);
                let bound_relax = std_max(1000.0, self.impl_col_upper[iu].abs()) * self.primal_feastol
                    / std_min(1.0, self.a_value[nz_pos as usize].abs());
                let new_ub = self.impl_col_upper[iu] + bound_relax;
                if new_ub < self.col_upper[iu] - bound_relax {
                    self.col_upper[iu] = new_ub;
                }
            }
        }
    }

    // ------------------------------------------------- HPresolveAnalysis

    fn analysis_setup(&mut self, silent: bool) {
        let mut allow = [1u8; RULE_COUNT];
        let allow_logging = self.host.analysis_setup(silent, &mut allow);
        for (a, &v) in self.analysis.allow_rule.iter_mut().zip(allow.iter()) {
            *a = v != 0;
        }
        self.analysis.allow_logging = allow_logging;
        self.analysis.logging_on = allow_logging;
        self.analysis.log_rule_type = RULE_ILLEGAL;
        self.analysis.num_deleted_rows0 = 0;
        self.analysis.num_deleted_cols0 = 0;
        self.analysis.log = [(0, 0, 0); RULE_COUNT];
        self.analysis.original_num_col = self.num_col;
        self.analysis.original_num_row = self.num_row;
    }

    /// analysePresolveRuleLog(report)
    fn analyse_presolve_rule_log(&mut self, report: bool) -> bool {
        let a = &self.analysis;
        if !a.allow_logging {
            return true;
        }
        let mut sum_removed_row = 0;
        let mut sum_removed_col = 0;
        for r in 0..RULE_COUNT {
            sum_removed_row += a.log[r].2;
            sum_removed_col += a.log[r].1;
        }
        if report && sum_removed_row + sum_removed_col != 0 {
            let rule = "-------------------------------------------------------";
            self.log_user(LOG_INFO, &sprintf("%s\n", &[rule.into()]));
            self.log_user(
                LOG_INFO,
                &sprintf("%-25s      Rows      Cols     Calls\n", &["Presolve rule removed".into()]),
            );
            self.log_user(LOG_INFO, &sprintf("%s\n", &[rule.into()]));
            for r in 0..RULE_COUNT {
                let (call, col_removed, row_removed) = a.log[r];
                if call != 0 || row_removed != 0 || col_removed != 0 {
                    self.log_user(
                        LOG_INFO,
                        &sprintf(
                            "%-25s %9d %9d %9d\n",
                            &[RULE_NAMES[r].into(), row_removed.into(), col_removed.into(), call.into()],
                        ),
                    );
                }
            }
            self.log_user(LOG_INFO, &sprintf("%s\n", &[rule.into()]));
            self.log_user(
                LOG_INFO,
                &sprintf("%-25s %9d %9d\n", &["Total reductions".into(), sum_removed_row.into(), sum_removed_col.into()]),
            );
            self.log_user(LOG_INFO, &sprintf("%s\n", &[rule.into()]));
            self.log_user(
                LOG_INFO,
                &sprintf(
                    "%-25s %9d %9d\n",
                    &["Original  model".into(), a.original_num_row.into(), a.original_num_col.into()],
                ),
            );
            self.log_user(
                LOG_INFO,
                &sprintf(
                    "%-25s %9d %9d\n",
                    &[
                        "Presolved model".into(),
                        (a.original_num_row - sum_removed_row).into(),
                        (a.original_num_col - sum_removed_col).into(),
                    ],
                ),
            );
            self.log_user(LOG_INFO, &sprintf("%s\n", &[rule.into()]));
        }
        if a.original_num_row == self.num_row && a.original_num_col == self.num_col {
            if sum_removed_row != self.num_deleted_rows {
                let msg = sprintf(
                    "%d = sum_removed_row != numDeletedRows = %d\n",
                    &[sum_removed_row.into(), self.num_deleted_rows.into()],
                );
                self.log_dev(LOG_ERROR, &msg);
                return false;
            }
            if sum_removed_col != self.num_deleted_cols {
                let msg = sprintf(
                    "%d = sum_removed_col != numDeletedCols = %d\n",
                    &[sum_removed_col.into(), self.num_deleted_cols.into()],
                );
                self.log_dev(LOG_ERROR, &msg);
                return false;
            }
        }
        true
    }
}
