//! HighsMipSolverData::evaluateRootNode and rootSeparationRound
//! (highs/mip/HighsMipSolverData.cpp): the first root LP, the root
//! heuristics (with graph LNS and the concurrent helper), the restarts at
//! the root, the cut loop and its stall test, and the root node put on the
//! node queue. The task group of the symmetry detection and the analytic
//! centre, the separator, the profiling clocks and the C++ objects are
//! reached through CMipFns::op (codes in `op` below).

use super::glue::{self, lp_status, Lp, MipData, Worker, P};
use super::mip_data::{op as mop, status};
use crate::lp_data::LogType;
use crate::util::cdouble::CDouble;
use crate::util::fma::ClangFma;

const INF: f64 = f64::INFINITY;
const IINF: i32 = i32::MAX;

/// The operations of the root node (CMipFns::op codes from 100)
pub mod op {
    /// profiling->start(clock i) / stop / running / mip_ / isSubMip()
    pub const PROF_START: i32 = 100;
    pub const PROF_STOP: i32 = 101;
    pub const PROF_RUNNING: i32 = 102;
    pub const PROF_MIP: i32 = 103;
    pub const PROF_IS_SUBMIP: i32 = 104;
    /// the task group and symmetry data of evaluateRootNode: create,
    /// free, cancel, taskWait
    pub const ROOT_CTX_NEW: i32 = 105;
    pub const ROOT_CTX_FREE: i32 = 106;
    pub const TG_CANCEL: i32 = 107;
    pub const TG_TASK_WAIT: i32 = 108;
    pub const START_SYMMETRY_DETECTION: i32 = 109;
    pub const START_ANALYTIC_CENTER: i32 = 110;
    /// getLp().setIterationLimit(i) (i < 0: no limit)
    pub const LP_SET_ITERATION_LIMIT: i32 = 113;
    pub const LP_LOAD_MODEL: i32 = 114;
    pub const DOM_CLEAR_CHANGED_COLS: i32 = 115;
    pub const LP_SET_OBJECTIVE_LIMIT: i32 = 116;
    pub const DOM_OBJECTIVE_LOWER_BOUND: i32 = 117;
    /// the root LP's basis or presolve setting before the first solve
    pub const LP_FIRST_SOLVE_SETUP: i32 = 119;
    pub const LP_SET_RACE_IPX: i32 = 120;
    pub const USE_CONCURRENT_HELPER: i32 = 121;
    pub const FIRSTROOTBASIS_VALID: i32 = 122;
    /// the LP solver options after the first root LP
    pub const LP_AFTER_FIRST_SOLVE: i32 = 123;
    /// firstlpsol, firstlpsolobj and rootlpsolobj from the LP
    pub const SAVE_FIRST_LP_SOL: i32 = 124;
    /// firstrootbasis from the LP (or the slack basis)
    pub const SET_FIRST_ROOT_BASIS: i32 = 125;
    /// separateLpCutsAfterRestart and addCuts
    pub const RESTART_CUTS: i32 = 126;
    /// getLp().removeObsoleteRows()
    pub const LP_REMOVE_OBSOLETE_ROWS: i32 = 127;
    /// a heuristic on the root (i: see heur below)
    pub const HEUR: i32 = 128;
    pub const START_CONCURRENT_LNS: i32 = 129;
    pub const SYNC_CONCURRENT_LNS: i32 = 130;
    pub const CROSSOVER_WITH_MAIN: i32 = 131;
    /// if the helper and the main solver search independently: sync, and
    /// the main solver's quick search is done
    pub const MAIN_QUICK_DONE: i32 = 132;
    pub const IMPORT_ROOT_CUTS: i32 = 134;
    pub const PUBLISH_ROOT_CUTS: i32 = 135;
    /// the root node on the node queue
    pub const NODEQUEUE_ROOT: i32 = 137;
    /// the root separator: create, separationRound (status i, returns
    /// ncuts; the status is read back with SEPA_STATUS), free
    pub const SEPA_NEW: i32 = 138;
    pub const SEPA_ROUND: i32 = 139;
    pub const SEPA_STATUS: i32 = 140;
    pub const SEPA_FREE: i32 = 141;
    /// mipsolver.terminate()
    pub const TERMINATE: i32 = 142;
    /// getLp().getAvgSolveIters()
    pub const LP_AVG_SOLVE_ITERS: i32 = 143;
    /// whether the helper and the main solver search independently
    pub const CONCURRENT_INDEPENDENT: i32 = 144;
    /// the LP solver basis is valid and the LP has only model rows
    pub const LP_BASIS_VALID: i32 = 145;
    /// getLp().getLpSolver().getBasis().valid
    pub const LP_SOLVER_BASIS_VALID: i32 = 146;
    /// rootlpsol = the LP solver's col_value
    pub const SAVE_ROOT_LP_SOL: i32 = 147;
    /// the number of the global domain's changed columns
    pub const DOM_NUM_CHANGED_COLS: i32 = 148;
    /// worker.getHeurLpIterations()
    pub const WORKER_HEUR_LP_ITERATIONS: i32 = 149;
    /// skipAnalyticCenter = i
    pub const SET_SKIP_ANALYTIC_CENTER: i32 = 150;
}

/// The heuristics of op::HEUR
pub mod heur {
    pub const ZI_ROUND_FIRST: i64 = 0;
    pub const RANDOMIZED_ROUNDING_FIRST: i64 = 1;
    pub const SHIFTING_FIRST: i64 = 2;
    pub const GRAPH_LNS_QUICK_FIRST: i64 = 3;
    pub const GRAPH_LNS_DEEP_ROOT: i64 = 4;
    pub const FLUSH: i64 = 5;
    pub const CENTRAL_ROUNDING: i64 = 6;
    pub const ROOT_REDUCED_COST: i64 = 7;
    pub const RENS_ROOT: i64 = 8;
    pub const FEASIBILITY_PUMP: i64 = 9;
    pub const SHIFTING_ROOT: i64 = 10;
    pub const RANDOMIZED_ROUNDING_LP: i64 = 11;
    pub const SHIFTING_LP: i64 = 12;
}

/// The profiling clocks (indices of HighsPrimalHeuristics.cpp's
/// rsMipClocks)
pub mod clk {
    pub const EVALUATE_ROOT_NODE0: i64 = 0;
    pub const EVALUATE_ROOT_NODE1: i64 = 1;
    pub const EVALUATE_ROOT_NODE2: i64 = 2;
    pub const START_SYMMETRY_DETECTION: i64 = 3;
    pub const START_ANALYTIC_CENTRE: i64 = 4;
    pub const EVALUATE_ROOT_LP: i64 = 5;
    pub const SEPARATE_LP_CUTS: i64 = 6;
    pub const RANDOMIZED_ROUNDING: i64 = 7;
    pub const PERFORM_RESTART: i64 = 8;
    pub const ROOT_SEPARATION: i64 = 9;
    pub const FINISH_ANALYTIC_CENTRE: i64 = 10;
    pub const ROOT_CENTRAL_ROUNDING: i64 = 11;
    pub const ROOT_SEPARATION_ROUND0: i64 = 12;
    pub const ROOT_HEURISTICS_REDUCED_COST: i64 = 13;
    pub const ROOT_SEPARATION_ROUND1: i64 = 14;
    pub const ROOT_HEURISTICS_RENS: i64 = 15;
    pub const ROOT_SEPARATION_ROUND2: i64 = 16;
    pub const ROOT_FEASIBILITY_PUMP: i64 = 17;
    pub const ROOT_SEPARATION_ROUND3: i64 = 18;
    pub const ROOT_SEPARATION_ROUND: i64 = 19;
    pub const ROOT_SEPARATION_FINISH_ANALYTIC_CENTRE: i64 = 20;
    pub const ROOT_SEPARATION_CENTRAL_ROUNDING: i64 = 21;
    pub const ROOT_SEPARATION_EVALUATE_ROOT_LP: i64 = 22;
}

/// External solution query origins (ExternalMipSolutionQueryOrigin)
mod origin {
    pub const EVALUATE_ROOT_NODE0: i64 = 2;
    pub const EVALUATE_ROOT_NODE1: i64 = 3;
    pub const EVALUATE_ROOT_NODE2: i64 = 4;
    pub const EVALUATE_ROOT_NODE3: i64 = 5;
    pub const EVALUATE_ROOT_NODE4: i64 = 6;
}

fn scaled_optimal(st: i32) -> bool {
    matches!(
        st,
        lp_status::OPTIMAL
            | lp_status::UNSCALED_DUAL_FEASIBLE
            | lp_status::UNSCALED_PRIMAL_FEASIBLE
            | lp_status::UNSCALED_INFEASIBLE
    )
}

#[inline(always)]
fn cmax(a: f64, b: f64) -> f64 {
    if a < b {
        b
    } else {
        a
    }
}

impl MipData {
    pub(super) fn o(&self, code: i32) -> f64 {
        self.op(code, None, 0, 0.0)
    }
    pub(super) fn oi(&self, code: i32, i: i64) -> f64 {
        self.op(code, None, i, 0.0)
    }
    fn ob(&self, code: i32) -> bool {
        self.op(code, None, 0, 0.0) != 0.0
    }
    fn start(&self, clock: i64) {
        self.oi(op::PROF_START, clock);
    }
    fn stop(&self, clock: i64) {
        self.oi(op::PROF_STOP, clock);
    }
    fn heur(&self, w: &Worker, h: i64) {
        self.op(op::HEUR, Some(w), h, 0.0);
    }
    fn eval_root_lp_timed(&self, w: &Worker) -> i32 {
        self.start(clk::EVALUATE_ROOT_LP);
        let st = self.evaluate_root_lp(w);
        self.stop(clk::EVALUATE_ROOT_LP);
        st
    }
    fn set_iteration_limit(&self, limit: i32) {
        self.oi(op::LP_SET_ITERATION_LIMIT, limit as i64);
    }
    fn mip_timing(&self, what: &str) {
        crate::log_user!(self.log, LogType::Info, "MIP-Timing: %11.2g - %s\n", self.timer_read(), what);
    }

    /// clockOff
    fn clock_off(&self) {
        if !self.ob(op::PROF_MIP) || self.ob(op::PROF_IS_SUBMIP) {
            return;
        }
        let r0 = self.oi(op::PROF_RUNNING, clk::EVALUATE_ROOT_NODE0) != 0.0;
        let r1 = self.oi(op::PROF_RUNNING, clk::EVALUATE_ROOT_NODE1) != 0.0;
        let r2 = self.oi(op::PROF_RUNNING, clk::EVALUATE_ROOT_NODE2) != 0.0;
        if !(r0 || r1 || r2) {
            print!(
                "{}",
                crate::sprintf!(
                    "HighsMipSolverData::clockOff Clocks running are (%d; %d; %d)\n",
                    r0 as i32,
                    r1 as i32,
                    r2 as i32
                )
            );
        }
        if r0 {
            self.stop(clk::EVALUATE_ROOT_NODE0);
        }
        if r1 {
            self.stop(clk::EVALUATE_ROOT_NODE1);
        }
        if r2 {
            self.stop(clk::EVALUATE_ROOT_NODE2);
        }
    }

    /// rootSeparationRound; returns true if the LP is infeasible
    fn root_separation_round(&self, w: &Worker, ncuts: &mut i32, status: &mut i32) -> bool {
        let lp = Lp::borrowed(self.lp);
        let sc = self.sc();
        let mut tmp_lp_iters = -lp.num_lp_iterations();
        *ncuts = self.op(op::SEPA_ROUND, Some(w), *status as i64, 0.0) as i32;
        *status = self.o(op::SEPA_STATUS) as i32;
        tmp_lp_iters += lp.num_lp_iterations();
        sc.avgrootlpiters = lp.avg_solve_iters();
        sc.total_lp_iterations += tmp_lp_iters;
        sc.sepa_lp_iterations += tmp_lp_iters;
        *status = self.evaluate_root_lp(w);
        if *status == lp_status::INFEASIBLE {
            return true;
        }
        if self.submip || self.incumbent().is_empty() {
            self.heur(w, heur::RANDOMIZED_ROUNDING_LP);
            if self.opts.run_shifting {
                self.heur(w, heur::SHIFTING_LP);
            }
            self.heur(w, heur::FLUSH);
            *status = self.evaluate_root_lp(w);
            if *status == lp_status::INFEASIBLE {
                return true;
            }
        }
        false
    }

    /// One pass of evaluateRootNode up to a restart (true: restart, with
    /// `max_sepa_rounds` updated; the solver's data is then refetched)
    fn evaluate_root_node_pass(&self, w: &Worker, max_sepa_rounds: &mut i32) -> bool {
        let o = &self.opts;
        let lp = Lp::borrowed(self.lp);
        // not in a concurrent LNS helper, nor in a main solver that has one
        let compute_analytic_centre = !self.concurrent_helper && !self.ob(op::USE_CONCURRENT_HELPER);
        let profiling_mip = self.ob(op::PROF_MIP);
        #[allow(clippy::never_loop)]
        loop {
            self.start(clk::EVALUATE_ROOT_NODE0);
            if self.sc().detect_symmetries {
                self.start(clk::START_SYMMETRY_DETECTION);
                self.o(op::START_SYMMETRY_DETECTION);
                self.stop(clk::START_SYMMETRY_DETECTION);
            }
            if compute_analytic_centre && !self.sc().analytic_center_computed {
                if profiling_mip {
                    self.mip_timing("starting analytic centre calculation");
                }
                self.start(clk::START_ANALYTIC_CENTRE);
                self.o(op::START_ANALYTIC_CENTER);
                self.stop(clk::START_ANALYTIC_CENTRE);
            }
            self.set_iteration_limit(-1);
            self.o(op::LP_LOAD_MODEL);
            self.o(op::DOM_CLEAR_CHANGED_COLS);
            self.op(op::LP_SET_OBJECTIVE_LIMIT, None, 0, self.sc().upper_limit);
            let lb = self.sc().lower_bound;
            self.update_lower_bound_ex(cmax(lb, self.o(op::DOM_OBJECTIVE_LOWER_BOUND)), true, true);
            self.print_display_line(-1);
            if !self.submip {
                self.query_external_solution(self.sol_objective(), origin::EVALUATE_ROOT_NODE0 as i32);
            }
            self.o(op::LP_FIRST_SOLVE_SETUP);
            // with a core to spare, IPX races the dual simplex on a large
            // first LP
            let nnz = self.a_start()[self.num_col as usize];
            let race = !self.ob(op::FIRSTROOTBASIS_VALID) && self.ob(op::USE_CONCURRENT_HELPER) && nnz >= 10000;
            self.oi(op::LP_SET_RACE_IPX, race as i64);
            let mut status = self.eval_root_lp_timed(w);
            self.oi(op::LP_SET_RACE_IPX, 0);
            if self.sc().num_restarts == 0 {
                self.sc().firstrootlpiters = self.sc().total_lp_iterations;
            }
            self.o(op::LP_AFTER_FIRST_SOLVE);
            if status == lp_status::INFEASIBLE || status == lp_status::UNBOUNDED {
                return self.clock_off_done();
            }
            self.o(op::SAVE_FIRST_LP_SOL);
            self.o(op::SET_FIRST_ROOT_BASIS);
            if self.o(mop::NUM_CUTS) != 0.0 {
                self.start(clk::SEPARATE_LP_CUTS);
                self.o(op::RESTART_CUTS);
                self.stop(clk::SEPARATE_LP_CUTS);
                status = self.eval_root_lp_timed(w);
                self.o(op::LP_REMOVE_OBSOLETE_ROWS);
                if status == lp_status::INFEASIBLE {
                    return self.clock_off_done();
                }
            }
            self.set_iteration_limit(10000.max((10.0 * self.sc().avgrootlpiters) as i32));
            // make sure first line after solving root LP is printed
            self.sc().last_disptime = -INF;
            self.sc().disptime = 0.0;
            if o.run_zi_round {
                self.heur(w, heur::ZI_ROUND_FIRST);
            }
            self.start(clk::RANDOMIZED_ROUNDING);
            self.heur(w, heur::RANDOMIZED_ROUNDING_FIRST);
            self.stop(clk::RANDOMIZED_ROUNDING);
            if o.run_shifting {
                self.heur(w, heur::SHIFTING_FIRST);
            }
            // graph LNS is for a loose target gap
            let run_graph_lns = o.run_graph_lns && o.mip_rel_gap >= 1e-3;
            if run_graph_lns {
                self.o(op::START_CONCURRENT_LNS);
                if self.sc().num_restarts == 0 && (!self.concurrent_helper || self.ob(op::CONCURRENT_INDEPENDENT)) {
                    let before = self.sc().upper_bound;
                    let quick_iters = -(self.op(op::WORKER_HEUR_LP_ITERATIONS, Some(w), 0, 0.0) as i64);
                    self.heur(w, heur::GRAPH_LNS_QUICK_FIRST);
                    let sc = self.sc();
                    sc.lns_quick_lp_iterations +=
                        quick_iters + self.op(op::WORKER_HEUR_LP_ITERATIONS, Some(w), 0, 0.0) as i64;
                    sc.lns_quick_improved = sc.upper_bound < before;
                    let skip = sc.lns_quick_improved
                        && sc.upper_bound - sc.lower_bound <= 3.0 * (sc.upper_bound - sc.optimality_limit);
                    self.oi(op::SET_SKIP_ANALYTIC_CENTER, skip as i64);
                }
                self.o(op::MAIN_QUICK_DONE);
                self.op(op::CROSSOVER_WITH_MAIN, Some(w), 0, 0.0);
            }
            self.heur(w, heur::FLUSH);
            status = self.eval_root_lp_timed(w);
            if status == lp_status::INFEASIBLE {
                return self.clock_off_done();
            }
            self.sc().rootlpsolobj = self.sc().firstlpsolobj;
            self.remove_fixed_indices();
            if o.mip_allow_restart && !o.presolve_off {
                let fixing_rate = self.percentage_inactive_integers();
                if fixing_rate >= 10.0 {
                    self.o(op::TG_CANCEL);
                    crate::log_user!(
                        self.log,
                        LogType::Info,
                        "\n%.1f%% inactive integer columns, restarting\n",
                        fixing_rate
                    );
                    self.o(op::TG_TASK_WAIT);
                    self.start(clk::PERFORM_RESTART);
                    super::setup::perform_restart(self);
                    self.stop(clk::PERFORM_RESTART);
                    self.sc().num_restarts_root += 1;
                    if self.modelstatus() == status::NOTSET {
                        self.clock_off();
                        return true;
                    }
                    self.clock_off();
                    return false;
                }
            }

            // begin separation
            if profiling_mip {
                self.mip_timing("starting  separation");
            }
            self.start(clk::ROOT_SEPARATION);
            let n = self.num_col as usize;
            let mut avgdirection = vec![0.0; n];
            let mut curdirection = vec![0.0; n];
            let mut stall = 0;
            let mut smoothprogress = 0.0;
            let mut nseparounds: i32 = 0;
            self.op(op::SEPA_NEW, Some(w), 0, 0.0);
            while scaled_optimal(status) && !lp.frac().is_empty() && stall < 3 {
                self.print_display_line(-1);
                if self.check_limits_rs(0) {
                    self.stop(clk::ROOT_SEPARATION);
                    return self.clock_off_done();
                }
                if nseparounds == *max_sepa_rounds {
                    break;
                }
                self.remove_fixed_indices();
                if !self.submip && !o.presolve_off {
                    let fixing_rate = self.percentage_inactive_integers();
                    if fixing_rate >= 10.0 {
                        stall = -1;
                        break;
                    }
                }
                nseparounds += 1;
                self.o(op::SYNC_CONCURRENT_LNS);
                self.op(op::CROSSOVER_WITH_MAIN, Some(w), 0, 0.0);
                if self.op(op::IMPORT_ROOT_CUTS, Some(w), 0, 0.0) != 0.0 {
                    self.stop(clk::ROOT_SEPARATION);
                    return self.clock_off_done();
                }
                status = lp.status();
                let mut ncuts = 0;
                self.start(clk::ROOT_SEPARATION_ROUND);
                let infeasible = self.root_separation_round(w, &mut ncuts, &mut status);
                self.stop(clk::ROOT_SEPARATION_ROUND);
                if infeasible {
                    self.stop(clk::ROOT_SEPARATION);
                    return self.clock_off_done();
                }
                if nseparounds >= 5 && !self.submip && !self.sc().analytic_center_computed && compute_analytic_centre
                {
                    if self.check_limits_rs(0) {
                        self.stop(clk::ROOT_SEPARATION);
                        return self.clock_off_done();
                    }
                    self.start(clk::ROOT_SEPARATION_FINISH_ANALYTIC_CENTRE);
                    self.finish_analytic_center();
                    self.stop(clk::ROOT_SEPARATION_FINISH_ANALYTIC_CENTRE);
                    self.start(clk::ROOT_SEPARATION_CENTRAL_ROUNDING);
                    self.heur(w, heur::CENTRAL_ROUNDING);
                    self.stop(clk::ROOT_SEPARATION_CENTRAL_ROUNDING);
                    self.heur(w, heur::FLUSH);
                    if self.check_limits_rs(0) {
                        self.stop(clk::ROOT_SEPARATION);
                        return self.clock_off_done();
                    }
                    self.start(clk::ROOT_SEPARATION_EVALUATE_ROOT_LP);
                    status = self.evaluate_root_lp(w);
                    self.stop(clk::ROOT_SEPARATION_EVALUATE_ROOT_LP);
                    if status == lp_status::INFEASIBLE {
                        self.stop(clk::ROOT_SEPARATION);
                        return self.clock_off_done();
                    }
                }
                let mut sqrnorm = CDouble::from(0.0);
                {
                    let solvals = lp.col_value();
                    let firstlpsol = self.firstlpsol();
                    for i in 0..n {
                        curdirection[i] = firstlpsol[i] - solvals[i];
                        sqrnorm += curdirection[i] * curdirection[i];
                    }
                }
                let scale = (1.0 / sqrnorm.sqrt()).to_f64();
                sqrnorm = CDouble::from(0.0);
                let mut dotproduct = CDouble::from(0.0);
                for i in 0..n {
                    avgdirection[i] = scale.mul_add_c(curdirection[i], -avgdirection[i]) / nseparounds as f64;
                    sqrnorm += avgdirection[i] * avgdirection[i];
                    dotproduct += avgdirection[i] * curdirection[i];
                }
                let progress = (dotproduct / sqrnorm.sqrt()).to_f64();
                if nseparounds == 1 {
                    smoothprogress = progress;
                } else {
                    let alpha = 1.0 / 3.0;
                    let nextprogress = (1.0 - alpha).mul_add_c(smoothprogress, alpha * progress);
                    let sc = self.sc();
                    if nextprogress < smoothprogress * 1.01
                        && (lp.objective() - sc.firstlpsolobj) <= (sc.rootlpsolobj - sc.firstlpsolobj) * 1.001
                    {
                        stall += 1;
                    } else {
                        stall = 0;
                    }
                    smoothprogress = nextprogress;
                }
                self.sc().rootlpsolobj = lp.objective();
                self.set_iteration_limit(10000.max((10.0 * self.sc().avgrootlpiters) as i32));
                if ncuts == 0 {
                    break;
                }
                if !self.submip {
                    self.query_external_solution(self.sol_objective(), origin::EVALUATE_ROOT_NODE1 as i32);
                }
            }
            self.stop(clk::ROOT_SEPARATION);
            if profiling_mip {
                self.mip_timing("completed separation");
            }

            self.set_iteration_limit(-1);
            status = self.eval_root_lp_timed(w);
            if status == lp_status::INFEASIBLE {
                return self.clock_off_done();
            }
            self.o(op::SAVE_ROOT_LP_SOL);
            self.sc().rootlpsolobj = lp.objective();
            self.set_iteration_limit(10000.max((10.0 * self.sc().avgrootlpiters) as i32));
            if o.run_zi_round {
                self.heur(w, heur::ZI_ROUND_FIRST);
                self.heur(w, heur::FLUSH);
            }
            if o.run_shifting {
                self.heur(w, heur::SHIFTING_ROOT);
                self.heur(w, heur::FLUSH);
            }
            if !self.sc().analytic_center_computed && compute_analytic_centre {
                if self.check_limits_rs(0) {
                    return self.clock_off_done();
                }
                self.start(clk::FINISH_ANALYTIC_CENTRE);
                self.finish_analytic_center();
                self.stop(clk::FINISH_ANALYTIC_CENTRE);
                self.start(clk::ROOT_CENTRAL_ROUNDING);
                self.heur(w, heur::CENTRAL_ROUNDING);
                self.stop(clk::ROOT_CENTRAL_ROUNDING);
                self.heur(w, heur::FLUSH);
                // if there are new global bound changes we re-evaluate the LP
                // and do one more separation round
                if self.check_limits_rs(0) {
                    return self.clock_off_done();
                }
                let separate = self.o(op::DOM_NUM_CHANGED_COLS) != 0.0;
                status = self.eval_root_lp_timed(w);
                if status == lp_status::INFEASIBLE {
                    return self.clock_off_done();
                }
                if separate && scaled_optimal(status) {
                    let mut ncuts = 0;
                    self.start(clk::ROOT_SEPARATION_ROUND0);
                    let r = self.root_separation_round(w, &mut ncuts, &mut status);
                    self.stop(clk::ROOT_SEPARATION_ROUND0);
                    if r {
                        return self.clock_off_done();
                    }
                    nseparounds += 1;
                    self.print_display_line(-1);
                }
            }
            self.print_display_line(-1);
            if !self.submip {
                self.query_external_solution(self.sol_objective(), origin::EVALUATE_ROOT_NODE2 as i32);
            }
            // possible cut extraction callback
            self.callback_get_cut_pool();
            if self.check_limits_rs(0) {
                return self.clock_off_done();
            }
            // a deeper graph-LNS search on the LP with the root cuts (see
            // HighsMipSolverData.cpp); a helper's root cuts are done
            if self.concurrent_helper {
                self.o(op::PUBLISH_ROOT_CUTS);
            }
            {
                let sc = self.sc();
                if run_graph_lns
                    && !self.rootlpsol().is_empty()
                    && (self.concurrent_helper
                        || (sc.lns_quick_improved
                            && sc.upper_bound - sc.lower_bound <= 3.0 * (sc.upper_bound - sc.optimality_limit)))
                {
                    let lns_iters = -sc.total_lp_iterations;
                    let lns_upper_bound = sc.upper_bound;
                    self.heur(w, heur::GRAPH_LNS_DEEP_ROOT);
                    self.heur(w, heur::FLUSH);
                    // if it pays, continue it during the tree search
                    let sc = self.sc();
                    if sc.upper_bound < lns_upper_bound && !self.submip {
                        sc.lns_tree_wait = 1000i64.max(lns_iters + sc.total_lp_iterations);
                        sc.lns_tree_next = sc.total_lp_iterations + sc.lns_tree_wait;
                    }
                    // a concurrent LNS helper keeps searching from the best
                    // solution either solver has found
                    if self.concurrent_helper {
                        let mut round = 0;
                        while round < 50 && !self.check_limits_rs(0) {
                            self.o(op::SYNC_CONCURRENT_LNS);
                            self.op(op::CROSSOVER_WITH_MAIN, Some(w), 0, 0.0);
                            self.heur(w, heur::GRAPH_LNS_DEEP_ROOT);
                            self.heur(w, heur::FLUSH);
                            round += 1;
                        }
                        return self.clock_off_done();
                    }
                    if self.check_limits_rs(0) {
                        return self.clock_off_done();
                    }
                }
            }

            self.stop(clk::EVALUATE_ROOT_NODE0);
            self.start(clk::EVALUATE_ROOT_NODE1);
            // the root heuristics below are pointless once the target gap is
            // reached
            let root_gap_closed = |m: &MipData| m.sc().lower_bound > m.sc().optimality_limit;
            #[allow(clippy::never_loop)]
            loop {
                if self.rootlpsol().is_empty() {
                    break;
                }
                if self.sc().upper_limit != INF && !self.more_heuristics_allowed() {
                    break;
                }
                if root_gap_closed(self) {
                    break;
                }
                if o.run_root_reduced_cost {
                    self.start(clk::ROOT_HEURISTICS_REDUCED_COST);
                    self.heur(w, heur::ROOT_REDUCED_COST);
                    self.stop(clk::ROOT_HEURISTICS_REDUCED_COST);
                    self.heur(w, heur::FLUSH);
                }
                if self.check_limits_rs(0) {
                    return self.clock_off_done();
                }
                let separate = self.o(op::DOM_NUM_CHANGED_COLS) != 0.0;
                status = self.eval_root_lp_timed(w);
                if status == lp_status::INFEASIBLE {
                    return self.clock_off_done();
                }
                if separate && scaled_optimal(status) {
                    let mut ncuts = 0;
                    self.start(clk::ROOT_SEPARATION_ROUND1);
                    let r = self.root_separation_round(w, &mut ncuts, &mut status);
                    self.stop(clk::ROOT_SEPARATION_ROUND1);
                    if r {
                        return self.clock_off_done();
                    }
                    nseparounds += 1;
                    self.print_display_line(-1);
                }
                if self.sc().upper_limit != INF && !self.more_heuristics_allowed() {
                    break;
                }
                if root_gap_closed(self) {
                    break;
                }
                if self.check_limits_rs(0) {
                    return self.clock_off_done();
                }
                if o.run_rens {
                    self.start(clk::ROOT_HEURISTICS_RENS);
                    self.heur(w, heur::RENS_ROOT);
                    self.stop(clk::ROOT_HEURISTICS_RENS);
                    self.heur(w, heur::FLUSH);
                }
                if self.check_limits_rs(0) {
                    return self.clock_off_done();
                }
                let separate = self.o(op::DOM_NUM_CHANGED_COLS) != 0.0;
                status = self.eval_root_lp_timed(w);
                if status == lp_status::INFEASIBLE {
                    return self.clock_off_done();
                }
                if separate && scaled_optimal(status) {
                    let mut ncuts = 0;
                    self.start(clk::ROOT_SEPARATION_ROUND2);
                    let r = self.root_separation_round(w, &mut ncuts, &mut status);
                    self.stop(clk::ROOT_SEPARATION_ROUND2);
                    if r {
                        return self.clock_off_done();
                    }
                    nseparounds += 1;
                    self.print_display_line(-1);
                    if !self.submip {
                        self.query_external_solution(self.sol_objective(), origin::EVALUATE_ROOT_NODE3 as i32);
                    }
                }
                if self.sc().upper_limit != INF || self.submip {
                    break;
                }
                if self.check_limits_rs(0) {
                    return self.clock_off_done();
                }
                self.start(clk::ROOT_FEASIBILITY_PUMP);
                self.heur(w, heur::FEASIBILITY_PUMP);
                self.stop(clk::ROOT_FEASIBILITY_PUMP);
                self.heur(w, heur::FLUSH);
                if self.check_limits_rs(0) {
                    return self.clock_off_done();
                }
                status = self.eval_root_lp_timed(w);
                if status == lp_status::INFEASIBLE {
                    return self.clock_off_done();
                }
                break;
            }

            self.stop(clk::EVALUATE_ROOT_NODE1);
            self.start(clk::EVALUATE_ROOT_NODE2);
            {
                let sc = self.sc();
                if sc.lower_bound > sc.upper_limit {
                    self.set_modelstatus(status::OPTIMAL);
                    sc.pruned_treeweight = CDouble::from(1.0);
                    sc.num_nodes += 1;
                    sc.num_leaves += 1;
                    return self.clock_off_done();
                }
            }
            let separate = self.o(op::DOM_NUM_CHANGED_COLS) != 0.0;
            status = self.eval_root_lp_timed(w);
            if status == lp_status::INFEASIBLE {
                return self.clock_off_done();
            }
            if separate && scaled_optimal(status) {
                let mut ncuts = 0;
                self.start(clk::ROOT_SEPARATION_ROUND3);
                let r = self.root_separation_round(w, &mut ncuts, &mut status);
                self.stop(clk::ROOT_SEPARATION_ROUND3);
                if r {
                    return self.clock_off_done();
                }
                nseparounds += 1;
                self.print_display_line(-1);
            }
            if !self.submip {
                self.query_external_solution(self.sol_objective(), origin::EVALUATE_ROOT_NODE4 as i32);
            }
            self.remove_fixed_indices();
            if self.ob(op::LP_SOLVER_BASIS_VALID) {
                self.o(op::LP_REMOVE_OBSOLETE_ROWS);
            }
            self.sc().rootlpsolobj = lp.objective();
            self.print_display_line(-1);

            if self.sc().lower_bound <= self.sc().upper_limit {
                if !self.submip && o.mip_allow_restart && !o.presolve_off {
                    if !self.sc().analytic_center_computed && compute_analytic_centre {
                        self.start(clk::FINISH_ANALYTIC_CENTRE);
                        self.finish_analytic_center();
                        self.stop(clk::FINISH_ANALYTIC_CENTRE);
                    }
                    let fixing_rate = self.percentage_inactive_integers();
                    // (2.5 + 7.5 * submip, and submip is false here)
                    if fixing_rate >= 2.5 || (fixing_rate > 0.0 && self.sc().num_restarts == 0) {
                        self.o(op::TG_CANCEL);
                        crate::log_user!(
                            self.log,
                            LogType::Info,
                            "\n%.1f%% inactive integer columns, restarting\n",
                            fixing_rate
                        );
                        if stall != -1 {
                            *max_sepa_rounds = (*max_sepa_rounds).min(nseparounds);
                        }
                        self.o(op::TG_TASK_WAIT);
                        self.start(clk::PERFORM_RESTART);
                        super::setup::perform_restart(self);
                        self.stop(clk::PERFORM_RESTART);
                        if self.ob(op::TERMINATE) {
                            return false;
                        }
                        self.sc().num_restarts_root += 1;
                        if self.modelstatus() == status::NOTSET {
                            self.clock_off();
                            // (the separator goes out of scope)
                            self.o(op::SEPA_FREE);
                            return true;
                        }
                        self.clock_off();
                        return false;
                    }
                }
                if self.sc().detect_symmetries {
                    self.finish_symmetry_detection();
                    status = self.eval_root_lp_timed(w);
                    if status == lp_status::INFEASIBLE {
                        return self.clock_off_done();
                    }
                }
                // add the root node to the node queue to initialize the
                // search
                self.op(op::NODEQUEUE_ROOT, Some(w), 0, 0.0);
            }
            return self.clock_off_done();
        }
    }

    fn clock_off_done(&self) -> bool {
        self.clock_off();
        false
    }
}

/// evaluateRootNode, refetching the solver's data after each restart
///
/// # Safety
/// `m` filled for this call by the C++ (and refilled through
/// CMipFns::refill), `w` the worker
pub unsafe fn evaluate_root_node(m: *mut MipData, w: P) {
    let md = &*m;
    md.op(op::ROOT_CTX_NEW, None, 0, 0.0);
    let mut max_sepa_rounds = if md.submip { 5 } else { IINF };
    if md.sc().num_restarts == 0 {
        max_sepa_rounds = ((2.0 * (md.sc().max_tree_size_log2 as f64).sqrt()) as i32).min(max_sepa_rounds);
    }
    loop {
        let md = &*m;
        let worker = Worker::new(w);
        if !md.evaluate_root_node_pass(&worker, &mut max_sepa_rounds) {
            break;
        }
        glue::refill(m);
    }
    (*m).op(op::ROOT_CTX_FREE, None, 0, 0.0);
}
