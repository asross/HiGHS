//! The probing loop of HPresolve::runProbing: the binary columns ordered by
//! their implications on other binaries, each probed by
//! implications.runProbing (mip/implications.rs, reached through its C++
//! glue) while the work limits allow.
//!
//! The clique table and implications are the Rust ones (borrowed only
//! between calls into C++), the global domain is read through its Rust view
//! (mip/domain.rs). The limits are the C++ ones: the deterministic counters
//! (probing contingent, the clique table's neighbourhood queries, the
//! numbers of cliques, implications and deletions) and the time limit.

use super::*;
use crate::mip::clique::CliqueTable;
use crate::mip::domain::Ctx as DomCtx;
use crate::util::random::HighsRandom;

impl Presolve<'_> {
    /// computeProbingScore: (implications up * down capped, by the number
    /// of probes; implications up + down capped)
    pub(crate) fn probing_score(&self, t: &CliqueTable, col: i32) -> (i64, i32) {
        let up = t.get_num_implications_val(col, true);
        let down = t.get_num_implications_val(col, false);
        (
            5000i64.min(up as i64 * down as i64) / (1 + self.num_probes[col as usize] as i64),
            100.min(up + down),
        )
    }

    /// highsTimeToString
    fn time_to_string(&self, t: f64) -> String {
        let mut s = self.host.time_string(t);
        s.pop();
        s
    }

    /// The loop of runProbing, between prepareProbing and finaliseProbing:
    /// false if there are no binaries to probe. Infeasibility (profiling
    /// stopped) and the time limit are errors
    pub(crate) fn probing_loop(&mut self) -> Result<bool, Stop> {
        let env = self.host.mip_env();
        // SAFETY: the view of the global domain, valid during the loop (no
        // presolve reduction is made)
        let mut d = unsafe { DomCtx::new(env.domain) };
        let submip = self.mip.expect("MIP presolve").submip;
        let old_num_probed = self.num_probed;

        let mut binaries: Vec<(i64, i32, i32, i32)> = Vec::new();
        if !self.cliquetable().is_full() {
            binaries.reserve(self.num_col as usize);
            let mut random = HighsRandom::new(self.opt.random_seed as u32);
            let dom = d.dom();
            for i in 0..self.num_col {
                if dom.is_binary(i as usize) {
                    let score = self.probing_score(self.cliquetable(), i);
                    binaries.push((-score.0, -score.1, random.integer(), i));
                }
            }
        }
        if binaries.is_empty() {
            return Ok(false);
        }
        // the keys are distinct (by column): any sort gives pdqsort's order
        binaries.sort_unstable();

        let mut num_changed_cols = 0usize;
        let count_fixed = |d: &mut DomCtx, num_changed_cols: &mut usize, num_del_col: &mut i32| {
            let dom = d.dom();
            let changed = dom.changed_cols();
            while *num_changed_cols != changed.len() {
                if dom.is_fixed(changed[*num_changed_cols] as usize) {
                    *num_del_col += 1;
                }
                *num_changed_cols += 1;
            }
        };
        count_fixed(&mut d, &mut num_changed_cols, &mut self.probing_num_del_col);

        let num_cliques_start = self.cliquetable().num_cliques_total();
        let num_implics_start = self.implications().num_implications as i32;
        let num_del_start = self.probing_num_del_col;
        let calc_num_del = |p: &Self| {
            p.probing_num_del_col - num_del_start
                + (p.implications().substitutions.len() + p.cliquetable().substitutions.len()) as i32
        };
        let mut num_del = calc_num_del(self);
        let num_nonzeros = self.num_nonzeros();
        let mut splay_contingent =
            self.cliquetable().num_neighbourhood_queries + (if submip { 0 } else { 100000 }).max(10 * num_nonzeros) as i64;
        let mut num_fail = 0i32;

        // lifting opportunities only if at least 2 percent of the columns
        // are continuous, up to 10 per row
        let model_has_percentage_cont_vars = |p: &Self, percentage: usize| {
            let (mut num_cols, mut num_cont_cols) = (0usize, 0usize);
            for col in 0..p.colsize.len() {
                if p.col_deleted[col] != 0 {
                    continue;
                }
                num_cols += 1;
                if p.integrality[col] == CONTINUOUS {
                    num_cont_cols += 1;
                }
            }
            100 * num_cont_cols >= percentage * num_cols
        };
        let max_num_lift_opps = 100000usize.max(10 * self.num_row as usize);
        let collect_lifting = self.opt.mip_lifting_for_probing != -1 && model_has_percentage_cont_vars(self, 2);
        let mut lifting = collect_lifting;
        if lifting {
            self.host.set_lifting(true);
        }

        let silent = self.silent_log();
        let mut i_bin: i32 = -1;
        let mut i_bin_probed: i32 = -1;
        let num_binary = binaries.len() as i32;
        let mut tt = self.host.timer_read();
        let tt0 = tt;
        let mut log_tt = tt0;
        let mut log_i_bin_probed = i_bin_probed;

        let mut result = Ok(true);
        for &(_, _, _, i) in &binaries {
            i_bin += 1;
            if self.cliquetable().get_substitution(i).is_some() || !d.dom().is_binary(i as usize) {
                continue;
            }
            i_bin_probed += 1;

            tt = self.host.timer_read();
            if tt > self.opt.time_limit {
                let msg = crate::util::printf::sprintf(
                    "Time limit reached in probing: consider not using probing by setting option presolve_rule_off to 2^%-d = %d\n",
                    &[(RULE_PROBING as i32).into(), (1i32 << RULE_PROBING).into()],
                );
                self.log_user(LOG_INFO, &msg);
                result = Err(Stop::Stopped);
                break;
            }

            // log the progress every 5 seconds
            if !silent && !self.opt.timeless_log && tt > log_tt + 5.0 && i_bin_probed > log_i_bin_probed {
                let rate0 = (tt - tt0) / i_bin_probed as f64;
                let rate1 = (tt - log_tt) / (i_bin_probed - log_i_bin_probed) as f64;
                let rate = std_max(rate0, rate1);
                let rate_str = format!(" (rate {}/ms", self.time_to_string(1e3 * rate));
                let finish = tt + rate * (num_binary - i_bin_probed) as f64;
                let finish_str = format!(" => expected probing finish time {})", self.host.time_string(finish));
                let msg = crate::util::printf::sprintf(
                    "   Considered %d / %d binaries; %d probed %s%s %s\n",
                    &[
                        i_bin.into(),
                        num_binary.into(),
                        i_bin_probed.into(),
                        rate_str.as_str().into(),
                        finish_str.as_str().into(),
                        self.host.time_string(tt).as_str().into(),
                    ],
                );
                self.log_user(LOG_INFO, &msg);
                log_tt = tt;
                log_i_bin_probed = i_bin_probed;
            }

            let tighten_limits = (self.num_probed - old_num_probed) >= 2500;
            let size_limit = (self.num_row + self.num_col) / 20;
            self.probing_early_abort =
                if !tighten_limits { num_del > 1000.max(size_limit) } else { num_del > 1000.min(size_limit) };
            if self.probing_early_abort {
                break;
            }

            // too many new implications: stop rather than spend ages
            let t = self.cliquetable();
            let max_new = 1000000.max(2 * num_nonzeros);
            if t.is_full()
                || t.num_cliques_total() - num_cliques_start > max_new
                || self.implications().num_implications as i32 - num_implics_start > max_new
            {
                break;
            }
            if t.num_neighbourhood_queries > splay_contingent {
                break;
            }
            if self.probing_contingent - (self.num_probed as i64) < 0 {
                break;
            }

            let mut num_bound_chgs = 0;
            let mut num_new_cliques = -self.cliquetable().num_cliques_total();
            if !self.host.probe(i, &mut num_bound_chgs) {
                continue;
            }
            self.probing_contingent += num_bound_chgs as i64;
            num_new_cliques += self.cliquetable().num_cliques_total();
            num_new_cliques = num_new_cliques.max(0);
            count_fixed(&mut d, &mut num_changed_cols, &mut self.probing_num_del_col);
            let new_num_del = calc_num_del(self);

            if new_num_del > num_del {
                self.probing_contingent += num_del as i64;
                if !submip {
                    splay_contingent += (100 * (new_num_del + num_del_start)) as i64;
                    splay_contingent += (1000 * num_new_cliques) as i64;
                }
                num_del = new_num_del;
                num_fail = 0;
            } else if submip || num_new_cliques == 0 {
                splay_contingent -= ((if tighten_limits { 250 } else { 100 }) * num_fail) as i64;
                num_fail += 1;
            } else {
                splay_contingent += (1000 * num_new_cliques) as i64;
                num_fail = 0;
            }

            self.num_probed += 1;
            self.num_probes[i as usize] = self.num_probes[i as usize].wrapping_add(1);

            if lifting && self.host.num_lifting_opps() >= max_num_lift_opps {
                self.host.set_lifting(false);
                lifting = false;
            }

            if d.infeasible() {
                self.host.profiling(false, 0);
                result = Err(Stop::PrimalInfeasible);
                break;
            }
        }
        if lifting {
            self.host.set_lifting(false);
        }
        result?;
        if collect_lifting {
            for (row, key, coef) in self.host.lifting_opps() {
                let (_, inserted) = self.lifting.entry(row).insert_or_get(key, coef);
                debug_assert!(inserted);
            }
        }
        Ok(true)
    }
}
