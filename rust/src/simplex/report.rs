//! The simplex logs of HighsSimplexAnalysis (highs/simplex/
//! HighsSimplexAnalysis.cpp): the iteration report (dev log, verbose), the
//! INVERT report (dev log) and the user INVERT report (the user log's
//! iteration lines), with the data the dual and primal simplex record for
//! them (HEkkDual/HEkkPrimal::iterationAnalysisData).
//!
//! The data and the header counters live in HighsSimplexAnalysis
//! (`highs_rs::SimplexReport` in HighsSimplexAnalysis.h), so they persist
//! over solves as the C++ fields did: a solve that records nothing reports
//! the previous solve's values. HighsSimplexAnalysis::setup resets what it
//! reset before.

use super::dual::AnalysisData;
use super::hekk::{CHekk, LOG_INFO, LOG_VERBOSE};
use crate::sprintf;

/// kSimplexStrategyPrimal
const STRATEGY_PRIMAL: i32 = 4;

/// The fields of HighsSimplexAnalysis that the reports read; mirrored by
/// highs_rs::SimplexReport in highs/simplex/HighsSimplexAnalysis.h
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SimplexReport {
    pub simplex_strategy: i32,
    pub solve_phase: i32,
    pub simplex_iteration_count: i32,
    pub pivotal_row_index: i32,
    pub entering_variable: i32,
    pub rebuild_reason: i32,
    pub num_primal_infeasibility: i32,
    pub num_dual_infeasibility: i32,
    pub num_iteration_report_since_last_header: i32,
    pub num_invert_report_since_last_header: i32,
    pub objective_value: f64,
    pub sum_primal_infeasibility: f64,
    pub sum_dual_infeasibility: f64,
    pub highs_run_time: f64,
    pub last_user_log_time: f64,
    pub delta_user_log_time: f64,
    /// The densities that HEkk::returnFromEkkSolve gives the simplex stats
    pub col_aq_density: f64,
    pub row_ep_density: f64,
    pub row_ap_density: f64,
    pub row_dse_density: f64,
    pub timeless_log: bool,
}

/// HEkk::rebuildReason
pub fn rebuild_reason_string(reason: i32) -> &'static str {
    match reason {
        -1 => "Perturbation cleanup",
        0 => "No reason",
        1 => "Update limit reached",
        2 => "Synthetic clock",
        3 => "Possibly optimal",
        4 => "Possibly phase 1 feasible",
        5 => "Possibly primal unbounded",
        6 => "Possibly dual unbounded",
        7 => "Possibly singular basis",
        8 => "Primal infeasible in primal simplex",
        9 => "Choose column failure",
        _ => "Unidentified",
    }
}

impl SimplexReport {
    fn dual_algorithm(&self) -> bool {
        matches!(self.simplex_strategy, 1..=3)
    }

    fn algorithm_phase(&self, header: bool, s: &mut String) {
        if header {
            s.push_str("     ");
        } else {
            let name = if self.dual_algorithm() { "Du" } else { "Pr" };
            s.push_str(&sprintf!("%2sPh%1d", name, self.solve_phase));
        }
    }

    fn iteration_objective(&self, header: bool, s: &mut String) {
        if header {
            s.push_str("  Iteration        Objective    ");
        } else {
            s.push_str(&sprintf!(" %10d %20.10e", self.simplex_iteration_count, self.objective_value));
        }
    }

    fn infeasibility(&self, header: bool, s: &mut String) {
        if header {
            s.push_str(" Infeasibilities num(sum)");
            return;
        }
        // Primal infeasibility information may not be known if dual ray
        // has proved primal infeasibility
        if self.num_primal_infeasibility <= -1 || self.sum_primal_infeasibility >= f64::INFINITY {
            return;
        }
        let phase = if self.solve_phase == 1 { " Ph1: %d(%g)" } else { " Pr: %d(%g)" };
        s.push_str(&sprintf!(phase, self.num_primal_infeasibility, self.sum_primal_infeasibility));
        if self.sum_dual_infeasibility > 0.0 {
            s.push_str(&sprintf!("; Du: %d(%g)", self.num_dual_infeasibility, self.sum_dual_infeasibility));
        }
    }
}

impl CHekk {
    fn rep(&self) -> SimplexReport {
        self.report.get()
    }

    /// HighsSimplexAnalysis::iterationReport(header)
    fn iteration_report_line(&self, header: bool) {
        let r = self.rep();
        if !header && (if r.dual_algorithm() { r.pivotal_row_index < 0 } else { r.entering_variable < 0 }) {
            return;
        }
        let mut s = String::new();
        r.algorithm_phase(header, &mut s);
        r.iteration_objective(header, &mut s);
        self.dev(LOG_VERBOSE, || s + "\n");
        if !header {
            let mut r = self.rep();
            r.num_iteration_report_since_last_header += 1;
            self.report.set(r);
        }
    }

    /// HighsSimplexAnalysis::iterationReport
    fn iteration_report(&self) {
        if !self.iteration_report {
            return;
        }
        let mut r = self.rep();
        let since = r.num_iteration_report_since_last_header;
        if since < 0 || since > 49 {
            self.iteration_report_line(true);
            r = self.rep();
            r.num_iteration_report_since_last_header = 0;
            self.report.set(r);
        }
        self.iteration_report_line(false);
    }

    /// HighsSimplexAnalysis::invertReport(header)
    fn invert_report_line(&self, header: bool) {
        let r = self.rep();
        let mut s = String::new();
        r.algorithm_phase(header, &mut s);
        r.iteration_objective(header, &mut s);
        r.infeasibility(header, &mut s);
        if !header {
            s.push(' ');
            s.push_str(rebuild_reason_string(r.rebuild_reason));
        }
        self.dev(LOG_INFO, || s + "\n");
        if !header {
            let mut r = self.rep();
            r.num_invert_report_since_last_header += 1;
            self.report.set(r);
        }
    }

    /// HighsSimplexAnalysis::invertReport
    fn invert_report(&self) {
        if self.log_dev_level != 0 {
            let r = self.rep();
            let since = r.num_invert_report_since_last_header;
            if since < 0 || since > 49 || r.num_iteration_report_since_last_header >= 0 {
                self.invert_report_line(true);
                let mut r = self.rep();
                r.num_invert_report_since_last_header = 0;
                self.report.set(r);
            }
            self.invert_report_line(false);
            // Force an iteration report header if this is an INVERT report
            // without a rebuild_reason
            let mut r = self.rep();
            if r.rebuild_reason == 0 {
                r.num_iteration_report_since_last_header = -1;
                self.report.set(r);
            }
        } else {
            self.user_invert_report(false);
        }
    }

    /// HighsSimplexAnalysis::userInvertReport(header, force)
    fn user_invert_report_line(&self, header: bool, force: bool) {
        let mut r = self.rep();
        r.highs_run_time = if r.timeless_log { r.highs_run_time + 1.0 } else { (self.host.timer_read)(self.host.ctx) };
        self.report.set(r);
        if !force && r.highs_run_time < r.last_user_log_time + r.delta_user_log_time {
            return;
        }
        let mut s = String::new();
        r.iteration_objective(header, &mut s);
        r.infeasibility(header, &mut s);
        if !r.timeless_log && !header {
            // reportRunTime of a build with NDEBUG
            s.push_str(&sprintf!(" %.1fs", r.highs_run_time));
        }
        s.push('\n');
        self.user(LOG_INFO, &s);
        if !header {
            r.last_user_log_time = r.highs_run_time;
        }
        if r.highs_run_time > 200.0 * r.delta_user_log_time {
            r.delta_user_log_time *= 10.0;
        }
        self.report.set(r);
    }

    /// HighsSimplexAnalysis::userInvertReport(force)
    pub fn user_invert_report(&self, force: bool) {
        if self.rep().last_user_log_time < 0.0 {
            self.user_invert_report_line(true, force);
        }
        self.user_invert_report_line(false, force);
    }

    /// The dual simplex's analysis data (kind 0), with its iteration
    /// report (1) or rebuild report (2, with the reason): the part of
    /// HEkkDual::iterationAnalysisData the reports read
    pub fn dual_report(&self, kind: i32, s: &AnalysisData, reason: i32, sense: i32) {
        let d = &s.state;
        let mut r = self.rep();
        r.simplex_strategy = self.simplex_strategy.get();
        r.solve_phase = d.solve_phase;
        r.simplex_iteration_count = s.iteration_count;
        r.pivotal_row_index = d.row_out;
        r.entering_variable = d.variable_in;
        r.rebuild_reason = d.rebuild_reason;
        r.objective_value = s.updated_dual_objective_value;
        if d.solve_phase == 2 {
            r.objective_value *= sense as f64;
        }
        r.num_primal_infeasibility = s.num_primal_infeasibilities;
        r.sum_primal_infeasibility = s.sum_primal_infeasibilities;
        r.num_dual_infeasibility = s.num_dual_infeasibilities;
        r.sum_dual_infeasibility = s.sum_dual_infeasibilities;
        r.col_aq_density = s.col_aq_density;
        r.row_ep_density = s.row_ep_density;
        r.row_ap_density = s.row_ap_density;
        r.row_dse_density = s.row_dse_density;
        self.report.set(r);
        if kind == 1 {
            self.iteration_report();
        } else if kind == 2 {
            r.rebuild_reason = reason;
            self.report.set(r);
            if self.output_flag {
                self.invert_report();
            }
        }
    }

    /// The primal simplex's iteration report (kind 0), rebuild report (1,
    /// with the reason) or analysis data (2): the part of
    /// HEkkPrimal::iterationAnalysisData the reports read
    #[allow(clippy::too_many_arguments)]
    pub fn primal_report(
        &self, kind: i32, solve_phase: i32, row_out: i32, variable_in: i32, rebuild_reason: i32,
        reason_for_rebuild: i32, objective: f64, infeasibility: (i32, f64, i32, f64), densities: [f64; 4],
    ) {
        let mut r = self.rep();
        r.simplex_strategy = STRATEGY_PRIMAL;
        r.solve_phase = solve_phase;
        r.simplex_iteration_count = self.iteration_count.get();
        r.pivotal_row_index = row_out;
        r.entering_variable = variable_in;
        r.rebuild_reason = rebuild_reason;
        r.objective_value = objective;
        (r.num_primal_infeasibility, r.sum_primal_infeasibility, r.num_dual_infeasibility, r.sum_dual_infeasibility) =
            infeasibility;
        [r.col_aq_density, r.row_ep_density, r.row_ap_density, r.row_dse_density] = densities;
        self.report.set(r);
        if kind == 0 {
            self.iteration_report();
        } else if kind == 1 {
            r.rebuild_reason = reason_for_rebuild;
            self.report.set(r);
            if self.output_flag {
                self.invert_report();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn layout_matches_cpp() {
        // static_assert in highs/simplex/HEkkRustSolve.cpp
        assert_eq!(std::mem::size_of::<super::SimplexReport>(), 128);
    }
}
