//! HighsMipSolver::run and cleanupSolve (highs/mip/HighsMipSolver.cpp): the
//! presolve and setup calls, the pre-root heuristics, the root node, the
//! branch-and-bound loop (node selection, plunging, the dives with their
//! heuristics, the restart votes, the workers and their synchronization,
//! graph-LNS rounds in the tree) and the final report. The workers, their
//! searches, separators, domains and pools are C++ objects reached through
//! CMipFns::op (codes from 200 below); the tasks of the parallel search
//! are spawned by the C++ (HighsMipSolver::runTask) and run
//! [`process_node`] here.

use super::glue::{self, MipData};
use super::mip_data::{op as mop, src, status};
use super::root::op as rop;
use crate::lp_data::LogType;
use crate::util::cdouble::CDouble;
use crate::util::fma::ClangFma;

const INF: f64 = f64::INFINITY;
const IINF: i32 = i32::MAX;

/// The operations of the driver (CMipFns::op codes from 200; `i` is mostly
/// a worker index)
pub mod op {
    /// mipdata_->init()
    pub const INIT: i32 = 200;
    /// mipdata_->runMipPresolve(presolve_reduction_limit)
    pub const RUN_MIP_PRESOLVE: i32 = 201;
    /// log "Presolve: <model status>"
    pub const LOG_PRESOLVE_STATUS: i32 = 202;
    /// mipdata_->runSetup()
    pub const RUN_SETUP: i32 = 203;
    /// the master worker
    pub const MASTER_WORKER_NEW: i32 = 204;
    /// mipdata_->feasibilityJump() (the model status)
    pub const FEASIBILITY_JUMP: i32 = 205;
    /// getCutPool().performAging()
    pub const CUTPOOL_PERFORM_AGING: i32 = 206;
    /// the number of workers
    pub const NUM_WORKERS: i32 = 207;
    /// getMaxNumWorkers()
    pub const MAX_NUM_WORKERS: i32 = 208;
    /// the run's task group: create, free
    pub const RUN_CTX_NEW: i32 = 209;
    pub const RUN_CTX_FREE: i32 = 210;
    /// destroyOldWorkers
    pub const DESTROY_OLD_WORKERS: i32 = 211;
    /// createNewWorkers(i)
    pub const CREATE_NEW_WORKERS: i32 = 212;
    /// constructAdditionalWorkerData(master worker)
    pub const CONSTRUCT_ADDITIONAL_WORKER_DATA: i32 = 213;
    /// syncSolutions
    pub const SYNC_SOLUTIONS: i32 = 214;
    /// syncPools and syncGlobalDomain of the search indices
    pub const SYNC_POOLS: i32 = 215;
    pub const SYNC_GLOBAL_DOMAIN: i32 = 216;
    /// the start of resetGlobalDomain (cleanupFixed and the workers'
    /// domains if i != 0), its middle (cleanupVarbounds of the changed
    /// columns, the empty domain change stack, the local domain if there is
    /// one worker) and its end (clearChangedCols)
    pub const RESET_GLOBAL_DOMAIN_START: i32 = 217;
    pub const RESET_GLOBAL_DOMAIN_MIDDLE: i32 = 218;
    pub const RESET_GLOBAL_DOMAIN_END: i32 = 219;
    /// syncGlobalPseudoCost, resetWorkerPseudoCosts
    pub const SYNC_GLOBAL_PSEUDOCOST: i32 = 220;
    pub const RESET_WORKER_PSEUDOCOSTS: i32 = 221;
    /// the master worker's search, separator and queue for a new run
    pub const RESET_MASTER_WORKER: i32 = 222;
    /// debugSolution.registerDomain of the master worker's local domain
    pub const REGISTER_DEBUG_DOMAIN: i32 = 223;
    /// worker i's search: local nodes (i), leaves, tree weight (hi; lo with
    /// x = 1), hasNode
    pub const SEARCH_NNODES: i32 = 224;
    pub const SEARCH_NLEAVES: i32 = 225;
    pub const SEARCH_TREEWEIGHT: i32 = 226;
    pub const SEARCH_HAS_NODE: i32 = 227;
    /// mipdata_->performRestart()
    pub const PERFORM_RESTART: i32 = 228;
    /// worker i allows heuristics (x != 0)
    pub const SET_ALLOW_HEURISTICS: i32 = 229;
    /// install the best bound node (x = 0) or the best node (x = 1) on
    /// worker i; returns 1 if the node popped is the best bound node
    pub const INSTALL_NODE: i32 = 230;
    /// worker i's search: getCurrentEstimate
    pub const SEARCH_CURRENT_ESTIMATE: i32 = 231;
    /// evaluateNode on worker i, the node to the queue if suboptimal
    /// (returns 1 then)
    pub const EVALUATE_NODE: i32 = 232;
    /// pruneNode on worker i
    pub const PRUNE_NODE: i32 = 233;
    /// worker i's search: checkLocalLimits
    pub const SEARCH_CHECK_LOCAL_LIMITS: i32 = 234;
    /// separateAndStoreBasis on worker i
    pub const SEPARATE_AND_STORE_BASIS: i32 = 235;
    /// worker i's conflict pool performAging
    pub const CONFLICT_PERFORM_AGING: i32 = 236;
    /// getLp().getAvgSolveIters() of the master LP
    pub const LP_AVG_SOLVE_ITERS: i32 = 237;
    /// worker i's LP: setIterationLimit(x)
    pub const WORKER_SET_ITERATION_LIMIT: i32 = 238;
    /// worker i: getAllowHeuristics
    pub const ALLOW_HEURISTICS: i32 = 239;
    /// runHeuristics' evaluateNode on worker i: 1 suboptimal, 2 pruned (a
    /// leaf counted), 0 otherwise
    pub const DIVE_EVALUATE_NODE: i32 = 240;
    /// the dive heuristics of worker i: randomizedRounding (x 0), RENS (1),
    /// RINS (2) on the worker LP's solution
    pub const DIVE_HEURISTIC: i32 = 241;
    /// worker i's global domain is infeasible
    pub const WORKER_DOMAIN_INFEASIBLE: i32 = 242;
    /// dive(i, nodeLim): returns 1 if the processing of the node stops
    pub const DIVE: i32 = 243;
    /// worker i's search: checkLimits(getLocalNodes())
    pub const SEARCH_CHECK_LIMITS: i32 = 244;
    /// backtrackPlunge(i): returns 1 if the processing stops
    pub const BACKTRACK_PLUNGE: i32 = 245;
    /// worker i's search: flushStatistics
    pub const SEARCH_FLUSH_STATISTICS: i32 = 246;
    /// runTask(processNode) over the search indices
    pub const RUN_PROCESS_NODES: i32 = 247;
    /// the open nodes of worker i to the global queue, its queue too,
    /// flushStatistics of its search, syncSepaStats and the heuristics'
    /// flushStatistics; returns 1 if its global domain is infeasible
    pub const FLUSH_WORKER: i32 = 248;
    /// pruneInfeasibleNodes on the global domain (the pruned weight)
    pub const PRUNE_INFEASIBLE_NODES: i32 = 249;
    /// getPseudoCost().removeChanged()
    pub const PSEUDOCOST_REMOVE_CHANGED: i32 = 250;
    /// the graph-LNS round of the tree search (deep, at most x LP
    /// iterations) and the heuristics' statistics
    pub const TREE_GRAPH_LNS: i32 = 251;
    /// the solve's end in C++ (before cleanupSolve): flags of the solve
    pub const CLEANUP_START: i32 = 252;
    /// the concurrent helper: its lower bound (into the solve), stop
    pub const CONCURRENT_HELPER_BOUND: i32 = 253;
    pub const STOP_CONCURRENT_LNS: i32 = 254;
    /// terminatorActive / terminatorTerminated / terminatorTerminate
    pub const TERMINATOR: i32 = 255;
    /// the end of cleanupSolve in C++: the solver's result fields from the
    /// values given in the CleanupResult, the report, the timer
    pub const CLEANUP_END: i32 = 256;
    /// worker i's upper_limit
    pub const WORKER_UPPER_LIMIT: i32 = 257;
    /// mipsolver.terminationStatus()
    pub const TERMINATION_STATUS: i32 = 258;
    /// timer_.stop()
    pub const TIMER_STOP: i32 = 259;
    /// the timing lines of solvingReport
    pub const REPORT_TIMING: i32 = 260;
    /// fclose of the improving solution file
    pub const CLOSE_IMPROVING_FILE: i32 = 261;
}

/// The clocks of the driver (indices into rsMipClocks)
mod clk {
    pub const PRESOLVE_TIME: i64 = 23;
    pub const INIT: i64 = 24;
    pub const RUN_PRESOLVE: i64 = 25;
    pub const SOLVE_TIME: i64 = 26;
    pub const RUN_SETUP: i64 = 27;
    pub const TRIVIAL_HEURISTICS: i64 = 28;
    pub const FEASIBILITY_JUMP: i64 = 29;
    pub const EVALUATE_ROOT_NODE: i64 = 30;
    pub const PERFORM_AGING0: i64 = 31;
    pub const SEARCH: i64 = 32;
    pub const UPDATE_LOCAL_DOMAIN: i64 = 33;
    pub const DOMAIN_PROPAGATE: i64 = 45;
    pub const PRUNE_INFEASIBLE_NODES: i64 = 46;
    pub const POSTSOLVE_TIME: i64 = 47;
}

/// RestartVote
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Vote {
    NoCheck,
    NoHugeTree,
    HugeTree,
    WouldRestart,
}

/// What the tasks of processNodes read (one per call of processNodes)
#[repr(C)]
pub struct ProcessCtx {
    pub md: *const MipData,
    pub skip_separation: bool,
    pub node_lim: i32,
    pub plunge_limit: i32,
    pub avgiter: f64,
    /// the restart votes, by worker index
    pub restarts: *mut Vote,
    /// the restart check's state (see Run)
    pub run: *const Run,
}

/// The state of the restart check (checkRestart's captures)
pub struct Run {
    num_huge_tree_estim: i32,
    num_nodes_last_check: i64,
    next_check: i64,
    treeweight_last_check: f64,
    upper_lim_last_check: f64,
    lower_bound_last_check: f64,
}

impl MipData {
    fn d(&self, code: i32, i: i64, x: f64) -> f64 {
        self.op(code, None, i, x)
    }
    fn dstart(&self, clock: i64) {
        self.op(rop::PROF_START, None, clock, 0.0);
    }
    fn dstop(&self, clock: i64) {
        self.op(rop::PROF_STOP, None, clock, 0.0);
    }
    fn profiling_mip_log(&self, what: &str) {
        if !self.submip && self.op(rop::PROF_MIP, None, 0, 0.0) != 0.0 {
            crate::log_user!(self.log, LogType::Info, "MIP-Timing: %11.2g - %s\n", self.timer_read(), what);
        }
    }
    fn parallel(&self) -> bool {
        self.parallel_lock_active()
    }
    fn worker_nnodes(&self, i: i64) -> i64 {
        self.d(op::SEARCH_NNODES, i, 0.0) as i64
    }

    /// checkRestart
    fn check_restart(&self, run: &Run, i: i64, mut num_worker_votes: i32) -> Vote {
        let sc = self.sc();
        let n_nodes = self.worker_nnodes(i) + sc.num_nodes;
        if !self.submip && n_nodes >= run.next_check && self.opts.mip_allow_restart {
            let n_tree_restarts = sc.num_restarts - sc.num_restarts_root;
            let tw = CDouble::new(self.d(op::SEARCH_TREEWEIGHT, i, 0.0), self.d(op::SEARCH_TREEWEIGHT, i, 1.0));
            let tree_weight = tw + sc.pruned_treeweight;
            let curr_node_estim = (run.num_nodes_last_check - sc.num_nodes_before_run) as f64
                + (n_nodes - run.num_nodes_last_check) as f64 * (1.0 - tree_weight).to_f64()
                    / max2((tree_weight - run.treeweight_last_check).to_f64(), sc.epsilon);
            let mut active_integer_ratio = 1.0 - self.percentage_inactive_integers() / 100.0;
            active_integer_ratio *= active_integer_ratio;
            let mut gap_reduction = 1.0;
            let upper_limit = self.d(op::WORKER_UPPER_LIMIT, i, 0.0);
            if upper_limit != INF {
                let old_gap = run.upper_lim_last_check - run.lower_bound_last_check;
                let new_gap = upper_limit - sc.lower_bound;
                gap_reduction = old_gap / new_gap;
            }
            let huge_tree = if gap_reduction < 1.0 + (0.05 / active_integer_ratio)
                && curr_node_estim >= active_integer_ratio * 20.0 * (n_nodes - sc.num_nodes_before_run) as f64
            {
                true
            } else {
                num_worker_votes = -IINF;
                false
            };
            let min_huge_tree_offset =
                (sc.num_leaves + self.d(op::SEARCH_NLEAVES, i, 0.0) as i64 - sc.num_leaves_before_run) / 1000;
            let x = active_integer_ratio * (10 + min_huge_tree_offset) as f64 * 1.5f64.powf(n_tree_restarts as f64);
            let min_huge_tree_estim = (x + 0.5f64.copysign(x)) as i64;
            let do_restart = (run.num_huge_tree_estim as i64 + num_worker_votes as i64) >= min_huge_tree_estim;
            if do_restart {
                return Vote::WouldRestart;
            }
            if huge_tree {
                return Vote::HugeTree;
            }
            return Vote::NoHugeTree;
        }
        Vote::NoCheck
    }

    /// processNode (the task of processNodes for worker i)
    pub fn process_node(&self, ctx: &ProcessCtx, i: i64) {
        let parallel = self.parallel();
        let mut nodes_explored: i64 = 0;
        if !ctx.skip_separation {
            self.d(op::EVALUATE_NODE, i, 0.0);
            if self.d(op::PRUNE_NODE, i, 0.0) != 0.0 {
                return;
            }
            if !parallel {
                if self.check_limits_rs(0) {
                    return;
                }
                self.print_display_line(src::NONE);
            } else if self.d(op::SEARCH_CHECK_LOCAL_LIMITS, i, 0.0) != 0.0 {
                return;
            }
            if self.d(op::SEPARATE_AND_STORE_BASIS, i, 0.0) != 0.0 {
                return;
            }
        }
        self.d(op::CONFLICT_PERFORM_AGING, i, 0.0);
        let sc = self.sc();
        let mut iterlimit = (10.0 * max2(ctx.avgiter, sc.avgrootlpiters)) as i32;
        iterlimit = 10000.max(iterlimit).max(((3 * sc.firstrootlpiters) / 2) as i32);
        self.d(op::WORKER_SET_ITERATION_LIMIT, i, iterlimit as f64);
        let mut consider_heuristics = true;
        loop {
            if consider_heuristics
                && (ctx.skip_separation || ctx.node_lim == IINF)
                && self.d(op::ALLOW_HEURISTICS, i, 0.0) != 0.0
                && self.more_heuristics_allowed()
                && self.run_heuristics(i)
            {
                break;
            }
            consider_heuristics = false;
            if self.d(op::WORKER_DOMAIN_INFEASIBLE, i, 0.0) != 0.0 {
                break;
            }
            if self.d(op::DIVE, i, ctx.node_lim as f64) != 0.0 {
                break;
            }
            if self.d(op::SEARCH_CHECK_LIMITS, i, 0.0) != 0.0 {
                break;
            }
            if self.worker_nnodes(i) + nodes_explored >= ctx.plunge_limit as i64 {
                break;
            }
            if !parallel {
                nodes_explored += self.worker_nnodes(i);
            }
            if self.d(op::BACKTRACK_PLUNGE, i, 0.0) != 0.0 {
                break;
            }
            if !parallel {
                self.d(op::SEARCH_FLUSH_STATISTICS, i, 0.0);
                self.print_display_line(src::NONE);
            }
        }
        if ctx.node_lim == IINF {
            // SAFETY: each task writes its own worker's vote
            unsafe {
                let run = &*ctx.run;
                *ctx.restarts.add(i as usize) = self.check_restart(run, i, 1);
            }
        }
    }

    /// runHeuristics: true if the processing of the node stops
    fn run_heuristics(&self, i: i64) -> bool {
        match self.d(op::DIVE_EVALUATE_NODE, i, 0.0) as i32 {
            1 => return true,
            2 => return false,
            _ => {}
        }
        // (the clocks and the heuristics are in the C++ op)
        if self.incumbent().is_empty() {
            self.d(op::DIVE_HEURISTIC, i, 0.0);
        }
        if self.incumbent().is_empty() {
            if self.opts.run_rens {
                self.d(op::DIVE_HEURISTIC, i, 1.0);
            }
        } else if self.opts.run_rins {
            self.d(op::DIVE_HEURISTIC, i, 2.0);
        }
        self.d(op::DIVE_HEURISTIC, i, -1.0);
        self.d(op::WORKER_DOMAIN_INFEASIBLE, i, 0.0) != 0.0
    }

    /// resetGlobalDomain
    fn reset_global_domain(&self, force: bool, reset_workers: bool, num_workers: i32) {
        let gd = self.domain();
        if gd.num_changed_cols() != 0 || force {
            self.dstart(clk::UPDATE_LOCAL_DOMAIN);
            crate::log_dev!(self.log, LogType::Info, "added %d global bound changes\n", gd.num_changed_cols());
            let multiple = self.d(op::NUM_WORKERS, 0, 0.0) > 1.0;
            self.d(op::RESET_GLOBAL_DOMAIN_START, (multiple && reset_workers) as i64, num_workers as f64);
            self.d(op::RESET_GLOBAL_DOMAIN_MIDDLE, multiple as i64, 0.0);
            self.d(op::RESET_GLOBAL_DOMAIN_END, 0, 0.0);
            self.remove_fixed_indices();
            self.dstop(clk::UPDATE_LOCAL_DOMAIN);
        }
    }
}

#[inline(always)]
fn max2(a: f64, b: f64) -> f64 {
    if a < b {
        b
    } else {
        a
    }
}
#[inline(always)]
fn min2(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// HighsMipSolver::run after the C++ created mipdata_
///
/// # Safety
/// `m` filled for this call (and refilled through CMipFns::refill)
pub unsafe fn run(m: *mut MipData) {
    let md = &*m;
    md.dstart(clk::PRESOLVE_TIME);
    md.dstart(clk::INIT);
    md.d(op::INIT, 0, 0.0);
    md.dstop(clk::INIT);
    md.dstart(clk::RUN_PRESOLVE);
    md.d(op::RUN_MIP_PRESOLVE, 0, 0.0);
    glue::refill(m);
    let md = &*m;
    md.dstop(clk::RUN_PRESOLVE);
    md.dstop(clk::PRESOLVE_TIME);
    md.profiling_mip_log("completed presolve");
    // identify whether the time limit has been reached (in presolve)
    if md.modelstatus() == status::NOTSET && md.timer_read() >= md.opts.time_limit {
        md.set_modelstatus(status::TIME_LIMIT);
    }
    if md.modelstatus() != status::NOTSET {
        md.d(op::LOG_PRESOLVE_STATUS, 0, 0.0);
        if md.modelstatus() == status::OPTIMAL {
            md.sc().lower_bound = 0.0;
            md.sc().upper_bound = 0.0;
            md.transform_new_integer_feasible_solution(&[], true);
            md.op(mop::SAVE_REPORT_SOLUTION, None, 0, -INF);
        }
        cleanup_solve(md);
        return;
    }
    md.dstart(clk::SOLVE_TIME);
    md.profiling_mip_log("starting  setup");
    md.dstart(clk::RUN_SETUP);
    md.d(op::RUN_SETUP, 0, 0.0);
    glue::refill(m);
    let md = &*m;
    md.dstop(clk::RUN_SETUP);
    md.profiling_mip_log("completed setup");
    if md.domain().infeasible() {
        cleanup_solve(md);
        return;
    }
    // initialise the master worker
    md.d(op::MASTER_WORKER_NEW, 0, 0.0);
    md.d(op::RUN_CTX_NEW, 0, 0.0);
    search(m);
    (*m).d(op::RUN_CTX_FREE, 0, 0.0);
}

/// From the `restart:` label of HighsMipSolver::run to its end
unsafe fn search(m: *mut MipData) {
    'restart: loop {
        // (the model changes on a restart)
        glue::refill(m);
        let md = &*m;
        if md.modelstatus() == status::NOTSET {
            // check the limits before evaluating the root node
            if md.check_limits_rs(0) {
                return cleanup_solve(md);
            }
            // possibly query the existence of an external solution
            if !md.submip {
                md.op(rop::QUERY_EXTERNAL_SOLUTION, None, 0, 0.0);
            }
            // the trivial heuristics
            md.dstart(clk::TRIVIAL_HEURISTICS);
            let returned = md.trivial_heuristics();
            md.dstop(clk::TRIVIAL_HEURISTICS);
            if md.modelstatus() == status::NOTSET && returned == status::INFEASIBLE {
                md.set_modelstatus(returned);
                return cleanup_solve(md);
            }
            // the feasibility jump heuristic (if enabled)
            if md.opts.run_feasibility_jump {
                md.dstart(clk::FEASIBILITY_JUMP);
                let returned = md.d(op::FEASIBILITY_JUMP, 0, 0.0) as i32;
                md.dstop(clk::FEASIBILITY_JUMP);
                if md.modelstatus() == status::NOTSET && returned == status::INFEASIBLE {
                    md.set_modelstatus(returned);
                    return cleanup_solve(md);
                }
            }
            md.profiling_mip_log("starting evaluate root node");
            md.dstart(clk::EVALUATE_ROOT_NODE);
            super::root::evaluate_root_node(m, glue::master_worker(md));
            glue::refill(m);
            let md = &*m;
            md.dstop(clk::EVALUATE_ROOT_NODE);
            if md.op(rop::TERMINATE, None, 0, 0.0) != 0.0 {
                md.set_modelstatus(md.op(op::TERMINATION_STATUS, None, 0, 0.0) as i32);
                return cleanup_solve(md);
            }
            md.profiling_mip_log("completed evaluate root node");
            // age 5 times to remove stored but never violated cuts after
            // root separation
            md.dstart(clk::PERFORM_AGING0);
            for _ in 0..5 {
                md.d(op::CUTPOOL_PERFORM_AGING, 0, 0.0);
            }
            md.dstop(clk::PERFORM_AGING0);
        }
        let md = &*m;
        if md.nodequeue().num_active_nodes() == 0 || md.check_limits_rs(0) {
            return cleanup_solve(md);
        }
        md.update_lower_bound_ex(md.nodequeue().best_lower_bound(), true, true);
        md.print_display_line(src::NONE);

        let max_num_workers = md.d(op::MAX_NUM_WORKERS, 0, 0.0) as i32;
        let mut num_workers: i32 = 1;

        md.d(op::RESET_MASTER_WORKER, 0, 0.0);
        md.d(op::DESTROY_OLD_WORKERS, 0, 0.0);
        md.d(op::REGISTER_DEBUG_DOMAIN, 0, 0.0);

        md.dstart(clk::SEARCH);
        let mut num_stall_nodes: i64 = 0;
        let mut last_lb_leave: i64 = 0;
        let mut num_queue_leaves: i64 = 0;
        let sc = md.sc();
        let mut run = Run {
            num_huge_tree_estim: 0,
            num_nodes_last_check: sc.num_nodes,
            next_check: sc.num_nodes,
            treeweight_last_check: 0.0,
            upper_lim_last_check: sc.upper_limit,
            lower_bound_last_check: sc.lower_bound,
        };

        // the main loop
        let mut search_indices: Vec<i32> = vec![0];
        search_indices.reserve(max_num_workers.max(0) as usize);
        let mut worker_restart_votes: Vec<Vote> = vec![Vote::NoCheck; max_num_workers.max(0) as usize];
        let mut root_node = true; // don't separate the root node again
        let mut node_lim: i32 = if max_num_workers > 1 { 1 } else { IINF }; // for ramp up
        while md.nodequeue().num_active_nodes() != 0 {
            md.op(rop::SYNC_CONCURRENT_LNS, None, 0, 0.0);
            // a graph LNS round once the tree search has had its share
            let sc = md.sc();
            if sc.lns_tree_next >= 0
                && md.d(op::NUM_WORKERS, 0, 0.0) <= 1.0
                && sc.total_lp_iterations >= sc.lns_tree_next
                && !md.rootlpsol().is_empty()
            {
                let iters = -sc.total_lp_iterations;
                let upper_bound = sc.upper_bound;
                md.d(op::TREE_GRAPH_LNS, 0, 5000i64.max(sc.lns_tree_wait) as f64);
                let sc = md.sc();
                if sc.upper_bound >= upper_bound {
                    sc.lns_tree_wait *= 2;
                }
                sc.lns_tree_next = sc.total_lp_iterations + sc.lns_tree_wait.max(iters + sc.total_lp_iterations);
                if md.check_limits_rs(0) {
                    break;
                }
            }
            // possibly query the existence of an external solution
            if !md.submip {
                md.op(rop::QUERY_EXTERNAL_SOLUTION, None, 1, 0.0);
            }
            // update the global pseudocost with the workers' information
            md.d(op::SYNC_GLOBAL_PSEUDOCOST, 0, 0.0);

            // getSearchIndicesWithNoNodes
            search_indices.clear();
            worker_restart_votes.iter_mut().for_each(|v| *v = Vote::NoCheck);
            let num_search_workers = (num_workers as i64).min(md.nodequeue().num_active_nodes()) as i32;
            let num_heuristic_workers = if md.sc().upper_bound < INF {
                1.max((num_search_workers + 3) / 4)
            } else {
                1.max((num_search_workers + 1) / 2)
            };
            for i in 0..num_search_workers {
                search_indices.push(i);
                md.d(op::SET_ALLOW_HEURISTICS, i as i64, (i < num_heuristic_workers) as i32 as f64);
            }

            // only update the pseudocosts of workers that get a node
            md.op(op::RESET_WORKER_PSEUDOCOSTS, None, search_indices.len() as i64, 0.0);

            // assign nodes to workers
            let mut limit_reached = false;
            if root_node {
                md.d(op::INSTALL_NODE, 0, 0.0);
            } else {
                // installNodes
                let nidx = search_indices.len();
                for &i in &search_indices {
                    if (nidx == 1 && num_queue_leaves - last_lb_leave >= 10) || (nidx > 1 && i == 0) {
                        md.d(op::INSTALL_NODE, i as i64, 0.0);
                        last_lb_leave = num_queue_leaves;
                    } else if md.d(op::INSTALL_NODE, i as i64, 1.0) != 0.0 {
                        last_lb_leave = num_queue_leaves;
                    }
                    num_queue_leaves += 1;
                    if md.d(op::SEARCH_CURRENT_ESTIMATE, i as i64, 0.0) >= md.sc().upper_limit {
                        num_stall_nodes += 1;
                        let msn = md.opts.mip_max_stall_nodes;
                        if msn != IINF && num_stall_nodes >= msn as i64 {
                            limit_reached = true;
                            md.set_modelstatus(status::SOLUTION_LIMIT);
                            break;
                        }
                    } else {
                        num_stall_nodes = 0;
                    }
                }
            }
            if limit_reached {
                break;
            }

            if node_lim != IINF {
                if num_workers >= max_num_workers {
                    node_lim = 20.max(2 * node_lim);
                }
                if node_lim > 100 {
                    node_lim = IINF;
                }
            }

            // process nodes (separation / heuristics / dives)
            let avgiter = md.d(op::LP_AVG_SOLVE_ITERS, 0, 0.0);
            let ctx = ProcessCtx {
                md: m,
                skip_separation: root_node,
                node_lim,
                plunge_limit: 100,
                avgiter,
                restarts: worker_restart_votes.as_mut_ptr(),
                run: &run,
            };
            glue::run_process_nodes(md, &search_indices, &ctx as *const ProcessCtx as *const std::ffi::c_void);

            root_node = false;

            // sync the statistics, check infeasibility, and flush the nodes
            // from the worker queues
            let mut infeasible = false;
            for &i in &search_indices {
                if md.d(op::FLUSH_WORKER, i as i64, 0.0) != 0.0 {
                    infeasible = true;
                }
            }
            let sc = md.sc();
            if infeasible {
                md.nodequeue().clear();
                sc.pruned_treeweight = CDouble::from(1.0);
                md.update_lower_bound_ex(min2(INF, sc.upper_bound), true, true);
                break;
            }
            md.update_lower_bound_ex(min2(sc.upper_bound, md.nodequeue().best_lower_bound()), true, true);
            md.d(op::SYNC_SOLUTIONS, 0, 0.0);
            if md.check_limits_rs(0) {
                md.print_display_line(src::NONE);
                break;
            }

            // sync the global information
            md.dstart(clk::DOMAIN_PROPAGATE);
            md.op(op::SYNC_POOLS, None, search_indices.len() as i64, 0.0);
            md.op(op::SYNC_GLOBAL_DOMAIN, None, search_indices.len() as i64, 0.0);
            md.domain().propagate();
            md.dstop(clk::DOMAIN_PROPAGATE);

            md.dstart(clk::PRUNE_INFEASIBLE_NODES);
            let pruned = md.d(op::PRUNE_INFEASIBLE_NODES, 0, 0.0);
            let sc = md.sc();
            sc.pruned_treeweight += pruned;
            md.dstop(clk::PRUNE_INFEASIBLE_NODES);

            if md.domain().infeasible() {
                md.nodequeue().clear();
                sc.pruned_treeweight = CDouble::from(1.0);
                md.update_lower_bound_ex(min2(INF, sc.upper_bound), true, true);
                md.print_display_line(src::NONE);
                break;
            }
            md.update_lower_bound_ex(min2(sc.upper_bound, md.nodequeue().best_lower_bound()), true, true);
            md.print_display_line(src::NONE);
            if md.nodequeue().num_active_nodes() == 0 {
                break;
            }

            // reset the global domain and sync the workers' global domains
            let spawn_more_workers =
                num_workers < max_num_workers && md.nodequeue().num_nodes() > num_workers as i64;
            let multiple = md.d(op::NUM_WORKERS, 0, 0.0) > 1.0;
            md.reset_global_domain(spawn_more_workers, multiple, num_workers);

            if node_lim == IINF && check_worker_restart_votes(md, &mut run, &worker_restart_votes) {
                continue 'restart;
            }

            if spawn_more_workers {
                let new_max_num_workers = (md.nodequeue().num_nodes().min(max_num_workers as i64)) as i32;
                md.d(op::PSEUDOCOST_REMOVE_CHANGED, 0, 0.0);
                if num_workers == 1 {
                    md.d(op::CONSTRUCT_ADDITIONAL_WORKER_DATA, 0, 0.0);
                }
                md.d(op::CREATE_NEW_WORKERS, (new_max_num_workers - num_workers) as i64, 0.0);
                num_workers = new_max_num_workers;
            }
        }
        md.d(op::SYNC_SOLUTIONS, 0, 0.0);
        md.dstop(clk::SEARCH);
        return cleanup_solve(md);
    }
}

/// checkWorkerRestartVotes: true if the search restarts
fn check_worker_restart_votes(md: &MipData, run: &mut Run, votes: &[Vote]) -> bool {
    let mut num_voters = 0;
    let mut num_huge_tree_votes = 0;
    let mut num_restart_votes = 0;
    for &v in votes {
        if v != Vote::NoCheck {
            num_voters += 1;
            if v == Vote::WouldRestart {
                num_restart_votes += 1;
            } else if v == Vote::HugeTree {
                num_huge_tree_votes += 1;
            }
        }
    }
    let perform_restart = |md: &MipData| {
        crate::log_user!(md.log, LogType::Info, "\nRestarting search from the root node\n");
        md.d(op::PERFORM_RESTART, 0, 0.0);
        md.dstop(clk::SEARCH);
    };
    // force a restart if enough individual workers vote for it
    if md.opts.mip_allow_restart && num_voters >= 2 && num_restart_votes as f64 / num_voters as f64 >= 0.25 {
        perform_restart(md);
        return true;
    }
    // using joint information after workers are synced, query a restart
    let joint = (num_restart_votes + num_huge_tree_votes + 2) / 2;
    let vote = md.check_restart(run, 0, joint);
    if vote == Vote::WouldRestart {
        perform_restart(md);
        return true;
    }
    let sc = md.sc();
    if vote == Vote::HugeTree {
        run.next_check = sc.num_nodes + 100;
        run.num_huge_tree_estim += joint;
    } else if vote == Vote::NoHugeTree {
        run.num_huge_tree_estim = 0;
        run.treeweight_last_check = sc.pruned_treeweight.to_f64();
        run.num_nodes_last_check = sc.num_nodes;
        run.upper_lim_last_check = sc.upper_limit;
        run.lower_bound_last_check = sc.lower_bound;
    }
    false
}

/// The results of cleanupSolve for the HighsMipSolver
#[repr(C)]
pub struct CleanupResult {
    pub dual_bound: f64,
    pub primal_bound: f64,
    pub gap: f64,
    pub node_count: i64,
    pub total_lp_iterations: i64,
    pub primal_dual_integral: f64,
}

/// utilModelStatusToString
pub fn model_status_to_string(s: i32) -> &'static str {
    const T: [&str; 20] = [
        "Not Set",
        "Load error",
        "Model error",
        "Presolve error",
        "Solve error",
        "Postsolve error",
        "Empty",
        "Optimal",
        "Infeasible",
        "Primal infeasible or unbounded",
        "Unbounded",
        "Bound on objective reached",
        "Target for objective reached",
        "Time limit reached",
        "Iteration limit reached",
        "Unknown",
        "Solution limit reached",
        "Interrupted by user",
        "Memory limit reached",
        "Interrupted by HiGHS",
    ];
    T.get(s as usize).copied().unwrap_or("Unrecognised HiGHS model status")
}

/// highsDoubleToString
pub fn double_to_string(val: f64, tolerance: f64) -> String {
    let l = if val.abs() == INF {
        1.0
    } else {
        1.0 - tolerance + (max2(tolerance, val.abs()) / tolerance).log10()
    };
    match l as i32 {
        0 => "0".to_string(),
        k @ 1..=16 => crate::util::printf::sprintf(&format!("%.{}g", k), &[val.into()]),
        _ => crate::sprintf!("%.16g", val),
    }
}

/// getGapString
pub fn gap_string(gap: f64, primal_bound: f64, rel_gap: f64, abs_gap: f64, feastol: f64) -> String {
    if gap == INF {
        return "inf".to_string();
    }
    let print_tol = max2(min2(1e-2, 1e-1 * gap), 1e-6);
    let gap_val = double_to_string(100.0 * gap, print_tol);
    let mut gap_tol = rel_gap;
    if abs_gap > feastol {
        gap_tol = if primal_bound == 0.0 { INF } else { max2(gap_tol, abs_gap / primal_bound.abs()) };
    }
    if gap_tol == 0.0 {
        crate::sprintf!("%s%%", &gap_val)
    } else if gap_tol != INF {
        let print_tol = max2(min2(1e-2, 1e-1 * gap_tol), 1e-6);
        let gap_tol_string = double_to_string(100.0 * gap_tol, print_tol);
        crate::sprintf!("%s%% (tolerance: %s%%)", &gap_val, &gap_tol_string)
    } else {
        crate::sprintf!("%s%% (tolerance: inf)", &gap_val)
    }
}

/// cleanupSolve
pub fn cleanup_solve(md: &MipData) {
    // take the helper's best solution even if no crossover took place, and
    // its root bound, valid for the whole solve
    let helper_bound = md.d(op::CONCURRENT_HELPER_BOUND, 0, 0.0);
    if helper_bound > md.sc().lower_bound {
        md.update_lower_bound_ex(helper_bound, true, true);
    }
    md.d(op::STOP_CONCURRENT_LNS, 0, 0.0);
    let sc = md.sc();
    // a solution of the helper may have closed the gap after a limit
    // stopped the search
    if md.modelstatus() == status::TIME_LIMIT && sc.upper_bound < INF && sc.lower_bound > sc.optimality_limit {
        md.set_modelstatus(status::NOTSET);
    }
    if md.d(op::TERMINATOR, 0, 0.0) != 0.0 {
        if md.d(op::TERMINATOR, 1, 0.0) != 0.0 {
            // this instance has been interrupted
            md.set_modelstatus(19);
        } else if !md.submip {
            md.d(op::TERMINATOR, 2, 0.0);
        }
    }
    // force a final logging line
    md.print_display_line(src::CLEANUP);
    // the solve clock is not running if presolve determined the status
    if md.op(rop::PROF_RUNNING, None, clk::SOLVE_TIME, 0.0) != 0.0 {
        md.dstop(clk::SOLVE_TIME);
    }
    let sc = md.sc();
    let (lb, ub) = (sc.lower_bound, sc.upper_bound);
    md.update_primal_dual_integral(lb, lb, ub, ub, false, true);
    md.dstart(clk::POSTSOLVE_TIME);
    let tol = md.opts.mip_feasibility_tolerance;
    let havesolution = md.sol_objective() != INF;
    let (bv, iv, rv) = md.violations();
    let feasible = havesolution && bv <= tol && iv <= tol && rv <= tol;
    let sc = md.sc();
    let mut dual_bound = sc.lower_bound;
    let scale = md.op(mop::OBJ_INT_SCALE, None, 0, 0.0);
    if scale != 0.0 {
        let rounded_lower_bound = sc.lower_bound.mul_add_c(scale, -sc.feastol).ceil() / scale;
        dual_bound = max2(dual_bound, rounded_lower_bound);
    }
    dual_bound += md.offset;
    let mut primal_bound = sc.upper_bound + md.offset;
    dual_bound = min2(dual_bound, primal_bound);
    // adjust the objective sense in case of a maximization problem
    if md.orig_maximize {
        dual_bound = -dual_bound;
        primal_bound = -primal_bound;
    }
    if md.modelstatus() == status::NOTSET || md.modelstatus() == status::INFEASIBLE {
        md.set_modelstatus(if feasible && havesolution { status::OPTIMAL } else { status::INFEASIBLE });
    }
    md.dstop(clk::POSTSOLVE_TIME);
    md.d(op::TIMER_STOP, 0, 0.0);
    let gap = if primal_bound == 0.0 {
        if dual_bound == 0.0 {
            0.0
        } else {
            INF
        }
    } else if primal_bound != INF {
        (primal_bound - dual_bound).abs() / primal_bound.abs()
    } else {
        INF
    };
    let r = CleanupResult {
        dual_bound,
        primal_bound,
        gap,
        node_count: sc.num_nodes,
        total_lp_iterations: sc.total_lp_iterations,
        primal_dual_integral: sc.pdi.value,
    };
    glue::set_cleanup_result(md, &r);
    if md.opts.output_flag_option {
        let solutionstatus = if havesolution {
            if feasible {
                "feasible"
            } else {
                "infeasible"
            }
        } else {
            "-"
        };
        solving_report(md, &r, solutionstatus);
    }
    md.d(op::CLOSE_IMPROVING_FILE, 0, 0.0);
}

/// solvingReport
fn solving_report(md: &MipData, r: &CleanupResult, solutionstatus: &str) {
    let o = &md.opts;
    let gap_str = gap_string(r.gap, r.primal_bound, o.mip_rel_gap, o.mip_abs_gap, o.mip_feasibility_tolerance);
    let log = &md.log;
    crate::log_user!(log, LogType::Info, "\nSolving report\n");
    let name = glue::model_name(md);
    if !name.is_empty() {
        crate::log_user!(log, LogType::Info, "  Model             %s\n", &name);
    }
    crate::log_user!(
        log,
        LogType::Info,
        "  Status            %s\n  Primal bound      %.12g\n  Dual bound        %.12g\n  Gap               %s\n",
        model_status_to_string(md.modelstatus()),
        r.primal_bound,
        r.dual_bound,
        &gap_str
    );
    if !o.timeless_log {
        crate::log_user!(log, LogType::Info, "  P-D integral      %.12g\n", md.sc().pdi.value);
    }
    crate::log_user!(log, LogType::Info, "  Solution status   %s\n", solutionstatus);
    if solutionstatus != "-" {
        let (bv, iv, rv) = md.violations();
        crate::log_user!(
            log,
            LogType::Info,
            "                    %.12g (objective)\n                    %.12g (bound viol.)\n                    %.12g (int. viol.)\n                    %.12g (row viol.)\n",
            md.sol_objective(),
            bv,
            iv,
            rv
        );
    }
    if !o.timeless_log {
        md.d(op::REPORT_TIMING, 0, 0.0);
    }
    let sc = md.sc();
    crate::log_user!(
        log,
        LogType::Info,
        "  Max sub-MIP depth %d\n  Nodes             %llu\n",
        glue::max_submip_level(md),
        sc.num_nodes
    );
    if sc.total_repair_lp != 0 {
        crate::log_user!(
            log,
            LogType::Info,
            "  Repair LPs        %llu (%llu feasible; %llu iterations)\n",
            sc.total_repair_lp,
            sc.total_repair_lp_feasible,
            sc.total_repair_lp_iterations
        );
    } else {
        crate::log_user!(log, LogType::Info, "  Repair LPs        0\n");
    }
    crate::log_user!(log, LogType::Info, "  LP iterations     %llu\n", sc.total_lp_iterations);
    if sc.total_lp_iterations != 0 {
        crate::log_user!(
            log,
            LogType::Info,
            "                    %llu (strong br.)\n                    %llu (separation)\n                    %llu (heuristics)\n",
            sc.sb_lp_iterations,
            sc.sepa_lp_iterations,
            sc.heuristic_lp_iterations
        );
    }
}

/// HighsMipSolver::run (after the C++ created mipdata_)
///
/// # Safety
/// `f` the C++ functions, `m` filled for this call
#[no_mangle]
pub unsafe extern "C" fn highs_rs_mip_run(f: *const glue::CMipFns, m: *mut MipData) {
    glue::set_fns(f);
    run(m)
}

/// HighsMipSolver::cleanupSolve
///
/// # Safety
/// as highs_rs_mip_run
#[no_mangle]
pub unsafe extern "C" fn highs_rs_mip_cleanup_solve(f: *const glue::CMipFns, m: *const MipData) {
    glue::set_fns(f);
    cleanup_solve(&*m)
}

/// processNode for worker i (a task of HighsMipSolver::runTask)
///
/// # Safety
/// as highs_rs_mip_run; `ctx` the ProcessCtx of the call
#[no_mangle]
pub unsafe extern "C" fn highs_rs_mip_process_node(f: *const glue::CMipFns, ctx: *const std::ffi::c_void, i: i32) {
    glue::set_fns(f);
    let ctx = &*(ctx as *const ProcessCtx);
    (*ctx.md).process_node(ctx, i as i64)
}
