//! HighsMipSolverData (highs/mip/HighsMipSolverData.cpp): the bookkeeping
//! of the bounds and the gap (limitsToGap, computeNewUpperLimit,
//! updateLowerBound and the primal-dual integral), the limits, the B&B
//! table (printDisplayLine), the heuristic effort rule, solutions
//! (checkSolution, trySolution, the trivial heuristics, addIncumbent and
//! transformNewIntegerFeasibleSolution), the root LP evaluation and the root
//! node. The data stays in the C++ struct, which the rest of the MIP solver
//! reads; Rust works on it through [`MipData`] and calls C++ for the C++
//! objects ([`CMipFns::op`] and the named functions of glue.rs).

use super::domain::StdVec;
use super::glue::{self, fns, lp_status, Dom, Lp, MipData, Worker};
use crate::lp_data::LogType;
use crate::util::cdouble::CDouble;
use crate::util::fma::ClangFma;

const INF: f64 = f64::INFINITY;
const IINF: i32 = i32::MAX;

/// HighsModelStatus values used here
pub mod status {
    pub const NOTSET: i32 = 0;
    pub const OPTIMAL: i32 = 7;
    pub const INFEASIBLE: i32 = 8;
    pub const UNBOUNDED_OR_INFEASIBLE: i32 = 9;
    pub const UNBOUNDED: i32 = 10;
    pub const OBJECTIVE_TARGET: i32 = 12;
    pub const TIME_LIMIT: i32 = 13;
    pub const SOLUTION_LIMIT: i32 = 16;
    pub const INTERRUPT: i32 = 17;
}

/// MipSolutionSource
pub mod src {
    pub const NONE: i32 = -1;
    pub const EVALUATE_NODE: i32 = 11;
    pub const USER_SOLUTION: i32 = 13;
    pub const EMPTY_MIP: i32 = 8;
    pub const TRIVIAL_L: i32 = 16;
    pub const TRIVIAL_P: i32 = 17;
    pub const TRIVIAL_U: i32 = 18;
    pub const TRIVIAL_Z: i32 = 19;
    pub const CLEANUP: i32 = 20;
    pub const COUNT: i32 = 21;
}

/// The operations of CMipFns::op on the solver (`m`, worker `w`, integer
/// `i`, real `x`; returns a real)
pub mod op {
    /// timer_.read()
    pub const TIMER_READ: i32 = 0;
    /// the terminator stopped this instance
    pub const LIMIT_FLAGS: i32 = 1;
    /// getCutPool().getNumCuts()
    pub const NUM_CUTS: i32 = 4;
    /// getConflictPool().getNumConflicts()
    pub const NUM_CONFLICTS: i32 = 5;
    /// getLp().numRows()
    pub const LP_NUM_ROWS: i32 = 6;
    /// objectiveFunction.integralScale() (0: not integral)
    pub const OBJ_INT_SCALE: i32 = 7;
    /// cliquetable.getSubstitutions().size()
    pub const NUM_SUBSTITUTIONS: i32 = 8;
    /// workers' upper_bound = x (i 0), upper_limit and optimality_limit
    /// (i 1)
    pub const SYNC_WORKERS: i32 = 14;
    /// debugSolution.newIncumbentFound()
    pub const DEBUG_NEW_INCUMBENT: i32 = 15;
    /// redcostfixing.propagateRootRedcost(mipsolver)
    pub const PROPAGATE_ROOT_REDCOST: i32 = 16;
    /// cliquetable.extractObjCliques(mipsolver)
    pub const EXTRACT_OBJ_CLIQUES: i32 = 17;
    /// globalOrbits->orbitalFixing(getDomain()) if there are global orbits
    pub const ORBITAL_FIXING: i32 = 18;
    /// store the scratch solution as the solver's (with the violations in
    /// MipData's solution fields)
    pub const STORE_SOLUTION: i32 = 20;
    /// getLp().getNumModelRows()
    pub const NUM_MODEL_ROWS: i32 = 21;
    /// getLp().getLpSolver().getModelStatus() == kNotset
    pub const LP_MODEL_STATUS_NOTSET: i32 = 22;
    /// redcostfixing.addRootRedcost(mipsolver, LP duals, LP objective)
    pub const ADD_ROOT_REDCOST: i32 = 23;
    /// heuristics.ziRound(worker, LP solution)
    pub const HEUR_ZI_ROUND: i32 = 24;
    /// mipsolver.solution_.empty()
    pub const SOLUTION_EMPTY: i32 = 25;
}

/// HighsMipSolverData's vectors ([`MipVecs`]): doubles (0-2, 20-22),
/// integers (3-10), bytes (30)
pub mod vec {
    pub const INCUMBENT: i32 = 0;
    pub const FIRSTLPSOL: i32 = 1;
    pub const ROOTLPSOL: i32 = 2;
    pub const INTEGRAL_COLS: i32 = 3;
    pub const INTEGER_COLS: i32 = 4;
    pub const IMPLINT_COLS: i32 = 5;
    pub const CONTINUOUS_COLS: i32 = 6;
    pub const AR_START: i32 = 7;
    pub const AR_INDEX: i32 = 8;
    pub const UPLOCKS: i32 = 9;
    pub const DOWNLOCKS: i32 = 10;
    /// doubles
    pub const AR_VALUE: i32 = 20;
    pub const MAX_ABS_ROW_COEF: i32 = 21;
    pub const ANALYTIC_CENTER: i32 = 22;
    /// bytes
    pub const ROW_INTEGRAL: i32 = 30;
}

/// HighsMipSolverData's vectors, owned by Rust; the C++ HighsMipSolverData
/// owns the struct (highs_rs::MipVecsOwner) and its old members refer to
/// the fields in place (HighsRsArray, the layout of StdVec). Set with
/// [`MipVecs::set`] (C++: highs_rs_mip_vecs_set)
#[repr(C)]
pub struct MipVecs {
    pub incumbent: StdVec<f64>,
    pub firstlpsol: StdVec<f64>,
    pub rootlpsol: StdVec<f64>,
    pub analytic_center: StdVec<f64>,
    pub ar_start: StdVec<i32>,
    pub ar_index: StdVec<i32>,
    pub ar_value: StdVec<f64>,
    pub max_abs_row_coef: StdVec<f64>,
    pub row_integral: StdVec<u8>,
    pub uplocks: StdVec<i32>,
    pub downlocks: StdVec<i32>,
    pub integer_cols: StdVec<i32>,
    pub implint_cols: StdVec<i32>,
    pub integral_cols: StdVec<i32>,
    pub continuous_cols: StdVec<i32>,
}

/// v = data (std::vector::assign)
fn assign<T: Copy>(v: &mut StdVec<T>, data: &[T]) {
    // SAFETY: a Rust-owned vector
    let mut x = unsafe { v.take_vec() };
    x.clear();
    x.extend_from_slice(data);
    *v = StdVec::from_vec(x);
}

impl MipVecs {
    fn new() -> Self {
        fn e<T: Copy>() -> StdVec<T> {
            StdVec::from_vec(Vec::new())
        }
        MipVecs {
            incumbent: e(),
            firstlpsol: e(),
            rootlpsol: e(),
            analytic_center: e(),
            ar_start: e(),
            ar_index: e(),
            ar_value: e(),
            max_abs_row_coef: e(),
            row_integral: e(),
            uplocks: e(),
            downlocks: e(),
            integer_cols: e(),
            implint_cols: e(),
            integral_cols: e(),
            continuous_cols: e(),
        }
    }
    fn dbl(&mut self, which: i32) -> &mut StdVec<f64> {
        match which {
            vec::INCUMBENT => &mut self.incumbent,
            vec::FIRSTLPSOL => &mut self.firstlpsol,
            vec::ROOTLPSOL => &mut self.rootlpsol,
            vec::AR_VALUE => &mut self.ar_value,
            vec::MAX_ABS_ROW_COEF => &mut self.max_abs_row_coef,
            _ => &mut self.analytic_center,
        }
    }
    pub fn int(&mut self, which: i32) -> &mut StdVec<i32> {
        match which {
            vec::INTEGRAL_COLS => &mut self.integral_cols,
            vec::INTEGER_COLS => &mut self.integer_cols,
            vec::IMPLINT_COLS => &mut self.implint_cols,
            vec::CONTINUOUS_COLS => &mut self.continuous_cols,
            vec::AR_START => &mut self.ar_start,
            vec::AR_INDEX => &mut self.ar_index,
            vec::UPLOCKS => &mut self.uplocks,
            _ => &mut self.downlocks,
        }
    }
    pub fn set_f64(&mut self, which: i32, data: &[f64]) {
        assign(self.dbl(which), data)
    }
    pub fn set_i32(&mut self, which: i32, data: &[i32]) {
        assign(self.int(which), data)
    }
    pub fn set_u8(&mut self, data: &[u8]) {
        assign(&mut self.row_integral, data)
    }
}

impl Drop for MipVecs {
    fn drop(&mut self) {
        // SAFETY: every vector is Rust-owned
        unsafe {
            drop(self.incumbent.take_vec());
            drop(self.firstlpsol.take_vec());
            drop(self.rootlpsol.take_vec());
            drop(self.analytic_center.take_vec());
            drop(self.ar_start.take_vec());
            drop(self.ar_index.take_vec());
            drop(self.ar_value.take_vec());
            drop(self.max_abs_row_coef.take_vec());
            drop(self.row_integral.take_vec());
            drop(self.uplocks.take_vec());
            drop(self.downlocks.take_vec());
            drop(self.integer_cols.take_vec());
            drop(self.implint_cols.take_vec());
            drop(self.integral_cols.take_vec());
            drop(self.continuous_cols.take_vec());
        }
    }
}

#[no_mangle]
pub extern "C" fn highs_rs_mip_vecs_new() -> *mut MipVecs {
    Box::into_raw(Box::new(MipVecs::new()))
}

/// # Safety
/// `v` from highs_rs_mip_vecs_new, or null
#[no_mangle]
pub unsafe extern "C" fn highs_rs_mip_vecs_free(v: *mut MipVecs) {
    if !v.is_null() {
        drop(Box::from_raw(v));
    }
}

/// Vector `which` (mod vec) = the `n` elements of `data`
///
/// # Safety
/// `v` live; `data` valid for `n` elements of the vector's type
#[no_mangle]
pub unsafe extern "C" fn highs_rs_mip_vecs_set(v: *mut MipVecs, which: i32, data: *const std::ffi::c_void, n: i32) {
    let v = &mut *v;
    if which == vec::ROW_INTEGRAL {
        v.set_u8(crate::ffi::sl(data as *const u8, n));
    } else if which <= 2 || which >= 20 {
        v.set_f64(which, crate::ffi::sl(data as *const f64, n));
    } else {
        v.set_i32(which, crate::ffi::sl(data as *const i32, n));
    }
}

/// std::min / std::max
#[inline(always)]
fn cmin(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}
#[inline(always)]
fn cmax(a: f64, b: f64) -> f64 {
    if a < b {
        b
    } else {
        a
    }
}

/// fractionality(x)
#[inline(always)]
fn fractionality(x: f64) -> f64 {
    (x - x.round()).abs()
}

/// solutionSourceToString
pub fn solution_source_to_string(source: i32, code: bool) -> &'static str {
    const T: [(&str, &str); 21] = [
        ("B", "Branching"),
        ("C", "Central rounding"),
        ("F", "Feasibility pump"),
        ("G", "Graph LNS"),
        ("H", "Heuristic"),
        ("I", "Shifting"),
        ("J", "Feasibility jump"),
        ("L", "Sub-MIP"),
        ("P", "Empty MIP"),
        ("R", "Randomized rounding"),
        ("S", "Solve LP"),
        ("T", "Evaluate node"),
        ("U", "Unbounded"),
        ("X", "User solution"),
        ("Y", "HiGHS solution"),
        ("Z", "ZI Round"),
        ("l", "Trivial lower"),
        ("p", "Trivial point"),
        ("u", "Trivial upper"),
        ("z", "Trivial zero"),
        (" ", ""),
    ];
    if source == src::NONE {
        return if code { " " } else { "None" };
    }
    match T.get(source as usize) {
        Some(&(c, n)) => {
            if code {
                c
            } else {
                n
            }
        }
        None => {
            if code {
                "*"
            } else {
                "None"
            }
        }
    }
}

/// convertToPrintString(int64_t)
fn print_i64(val: i64) -> String {
    let l = cmax(1.0, val as f64).log10();
    match l as i32 {
        0..=5 => crate::sprintf!("%lld", val),
        6..=8 => crate::sprintf!("%lldk", val / 1000),
        _ => crate::sprintf!("%lldm", val / 1000000),
    }
}

/// convertToPrintString(double, trailingStr)
fn print_f64(val: f64, trailing: &str) -> String {
    let l = if val.abs() == INF { 0.0 } else { cmax(1e-6, val.abs()).log10() };
    match l as i32 {
        0..=3 => crate::sprintf!("%.10g%s", val, trailing),
        4 => crate::sprintf!("%.11g%s", val, trailing),
        5 => crate::sprintf!("%.12g%s", val, trailing),
        6..=10 => crate::sprintf!("%.13g%s", val, trailing),
        _ => crate::sprintf!("%.9g%s", val, trailing),
    }
}

impl MipData {
    pub fn op(&self, op: i32, w: Option<&Worker>, i: i64, x: f64) -> f64 {
        // SAFETY: the C++ operation on the live solver
        unsafe { (fns().op)(self.mipsolver, op, w.map_or(std::ptr::null_mut(), |w| w.p), i, x) }
    }
    pub fn timer_read(&self) -> f64 {
        self.op(op::TIMER_READ, None, 0, 0.0)
    }

    // ---- fields written in place ----
    #[allow(clippy::mut_from_ref)]
    pub fn sc(&self) -> &mut MipScalars {
        // SAFETY: the solver's scalars; Rust holds no other reference to
        // them across a use, and C++ only changes them in calls made from
        // here, which do not overlap a use
        unsafe { &mut *self.scalars }
    }
    pub fn modelstatus(&self) -> i32 {
        // SAFETY: as sc
        unsafe { *self.modelstatus }
    }
    pub fn set_modelstatus(&self, s: i32) {
        // SAFETY: as sc
        unsafe { *self.modelstatus = s }
    }

    /// limitsToGap: (gap, lb, ub)
    pub fn limits_to_gap(&self, use_lower_bound: f64, use_upper_bound: f64) -> (f64, f64, f64) {
        let offset = self.offset;
        let epsilon = self.sc().epsilon;
        let mut lb = use_lower_bound + offset;
        if lb.abs() <= epsilon {
            lb = 0.0;
        }
        let mut ub = INF;
        let mut gap = INF;
        if use_upper_bound != INF {
            ub = use_upper_bound + offset;
            if ub.abs() <= epsilon {
                ub = 0.0;
            }
            lb = cmin(ub, lb);
            if ub == 0.0 {
                gap = if lb == 0.0 { 0.0 } else { INF };
            } else {
                gap = (ub - lb) / ub.abs();
            }
        }
        (gap, lb, ub)
    }

    /// computeNewUpperLimit
    pub fn compute_new_upper_limit(&self, ub: f64, mip_abs_gap: f64, mip_rel_gap: f64) -> f64 {
        let scale = self.op(op::OBJ_INT_SCALE, None, 0, 0.0);
        let sc = self.sc();
        let mut new_upper_limit;
        if scale != 0.0 {
            new_upper_limit = scale.mul_add_c(ub, -0.5).floor() / scale;
            if mip_rel_gap != 0.0 {
                new_upper_limit = cmin(
                    new_upper_limit,
                    ub - ((mip_rel_gap * (ub + self.offset).abs()).mul_add_c(scale, -sc.epsilon)).ceil() / scale,
                );
            }
            if mip_abs_gap != 0.0 {
                new_upper_limit = cmin(new_upper_limit, ub - mip_abs_gap.mul_add_c(scale, -sc.epsilon).ceil() / scale);
            }
            // add feasibility tolerance so that the next best integer
            // feasible solution is definitely included in the remaining
            // search
            new_upper_limit += sc.feastol;
        } else {
            new_upper_limit = cmin(ub - sc.feastol, ub.next_down());
            if mip_rel_gap != 0.0 {
                new_upper_limit = cmin(new_upper_limit, (-mip_rel_gap).mul_add_c((ub + self.offset).abs(), ub));
            }
            if mip_abs_gap != 0.0 {
                new_upper_limit = cmin(new_upper_limit, ub - mip_abs_gap);
            }
        }
        new_upper_limit
    }

    /// limitsToBounds: (dual_bound, primal_bound, mip_rel_gap)
    pub fn limits_to_bounds(&self) -> (f64, f64, f64) {
        let sc = self.sc();
        let (gap, mut dual_bound, primal_bound) = self.limits_to_gap(sc.lower_bound, sc.upper_bound);
        let mut primal_bound = cmin(self.opts.objective_bound, primal_bound);
        if self.orig_maximize {
            dual_bound = -dual_bound;
            primal_bound = -primal_bound;
        }
        (dual_bound, primal_bound, gap)
    }

    /// updateLowerBound
    pub fn update_lower_bound_ex(&self, new_lower_bound: f64, check_bound_change: bool, check_prev_data: bool) {
        let sc = self.sc();
        let prev_lower_bound = sc.lower_bound;
        sc.lower_bound = new_lower_bound;
        if !self.submip && sc.lower_bound != prev_lower_bound {
            let (lb, ub) = (sc.lower_bound, sc.upper_bound);
            self.update_primal_dual_integral(prev_lower_bound, lb, ub, ub, check_bound_change, check_prev_data);
        }
    }

    /// updatePrimalDualIntegral (its consistency checks are debug asserts
    /// in the C++)
    pub fn update_primal_dual_integral(
        &self,
        from_lower_bound: f64,
        to_lower_bound: f64,
        from_upper_bound: f64,
        to_upper_bound: f64,
        _check_bound_change: bool,
        _check_prev_data: bool,
    ) {
        let (_from_gap, _from_lb, _from_ub) = self.limits_to_gap(from_lower_bound, from_upper_bound);
        let (to_gap, to_lb, to_ub) = self.limits_to_gap(to_lower_bound, to_upper_bound);
        let from_gap = _from_gap;
        let pdi = &mut self.sc().pdi;
        if pdi.value > -INF {
            if to_gap < INF {
                let time = self.timer_read();
                if from_gap < INF {
                    // Need to update the P-D integral
                    let time_diff = time - pdi.prev_time;
                    pdi.value = time_diff.mul_add_c(pdi.prev_gap, pdi.value);
                }
                pdi.prev_time = time;
            }
        } else {
            pdi.value = 0.0;
        }
        pdi.prev_lb = to_lb;
        pdi.prev_ub = to_ub;
        pdi.prev_gap = to_gap;
    }

    /// checkLimits
    pub fn check_limits_rs(&self, node_offset: i64) -> bool {
        if self.concurrent_limit() || self.op(op::LIMIT_FLAGS, None, 0, 0.0) != 0.0 {
            return true;
        }
        // possible user interrupt
        if !self.submip && !self.parallel_lock_active() && self.user_interrupt() {
            if self.modelstatus() == status::NOTSET {
                crate::log_dev!(self.log, LogType::Info, "User interrupt\n");
                self.set_modelstatus(status::INTERRUPT);
            }
            return true;
        }
        let o = &self.opts;
        // possible termination due to the objective reaching the target
        let solution_objective = self.sol_objective();
        if !self.submip && solution_objective < INF && o.objective_target > -INF {
            let sense = if self.orig_maximize { -1.0 } else { 1.0 };
            if sense * solution_objective < sense * o.objective_target {
                if self.modelstatus() == status::NOTSET {
                    crate::log_dev!(self.log, LogType::Info, "Reached objective target\n");
                    self.set_modelstatus(status::OBJECTIVE_TARGET);
                }
                return true;
            }
        }
        let sc = self.sc();
        if o.mip_max_nodes != IINF && sc.num_nodes + node_offset >= o.mip_max_nodes as i64 {
            if self.modelstatus() == status::NOTSET {
                crate::log_dev!(self.log, LogType::Info, "Reached node limit\n");
                self.set_modelstatus(status::SOLUTION_LIMIT);
            }
            return true;
        }
        if o.mip_max_leaves != IINF && sc.num_leaves >= o.mip_max_leaves as i64 {
            if self.modelstatus() == status::NOTSET {
                crate::log_dev!(self.log, LogType::Info, "Reached leaf node limit\n");
                self.set_modelstatus(status::SOLUTION_LIMIT);
            }
            return true;
        }
        if o.mip_max_improving_sols != IINF && sc.num_improving_sols >= o.mip_max_improving_sols {
            if self.modelstatus() == status::NOTSET {
                crate::log_dev!(self.log, LogType::Info, "Reached improving solution limit\n");
                self.set_modelstatus(status::SOLUTION_LIMIT);
            }
            return true;
        }
        if o.time_limit < INF && self.timer_read() >= o.time_limit {
            if self.modelstatus() == status::NOTSET {
                crate::log_dev!(self.log, LogType::Info, "Reached time limit\n");
                self.set_modelstatus(status::TIME_LIMIT);
            }
            return true;
        }
        false
    }

    /// moreHeuristicsAllowed
    pub fn more_heuristics_allowed(&self) -> bool {
        let sc = self.sc();
        // the quick graph-LNS search, early in the root node, has a budget
        // of its own
        let heur_lp_iterations = sc.heuristic_lp_iterations - sc.lns_quick_lp_iterations;
        let pruned = sc.pruned_treeweight;
        if self.submip {
            return (heur_lp_iterations as f64) < sc.total_lp_iterations as f64 * sc.heuristic_effort;
        } else if pruned < 1e-3
            && sc.num_leaves - sc.num_leaves_before_run < 10
            && sc.num_nodes - sc.num_nodes_before_run < 1000
        {
            // in the main MIP solver allow an initial offset of 10000
            // heuristic LP iterations
            if (heur_lp_iterations as f64) < (sc.total_lp_iterations as f64).mul_add_c(sc.heuristic_effort, 10000.0) {
                return true;
            }
        } else if heur_lp_iterations
            < 100000 + ((sc.total_lp_iterations - heur_lp_iterations - sc.sb_lp_iterations) >> 1)
        {
            let heur_iters_curr_run = heur_lp_iterations - sc.heuristic_lp_iterations_before_run;
            let sb_iters_curr_run = sc.sb_lp_iterations - sc.sb_lp_iterations_before_run;
            let node_iters_curr_run =
                sc.total_lp_iterations - sc.total_lp_iterations_before_run - heur_iters_curr_run - sb_iters_curr_run;
            let total_heuristic_effort_estim = heur_lp_iterations as f64
                / ((sc.total_lp_iterations - node_iters_curr_run) as f64
                    + node_iters_curr_run as f64 / cmax(0.01, pruned.to_f64()));
            if total_heuristic_effort_estim
                < cmax(0.3 / 0.8, cmin(pruned.to_f64(), 0.8) / 0.8) * sc.heuristic_effort
            {
                return true;
            }
        }
        false
    }

    /// percentageInactiveIntegers
    pub fn percentage_inactive_integers(&self) -> f64 {
        let nsubst = self.op(op::NUM_SUBSTITUTIONS, None, 0, 0.0) as usize;
        100.0
            * (1.0
                - (self.integer_cols().len().wrapping_sub(nsubst)) as f64 / self.sc().numintegercols as f64)
    }

    /// printSolutionSourceKey
    fn print_solution_source_key(&self) {
        let last_enum = src::COUNT - 1;
        let limits = [4, 9, 14, last_enum];
        let mut ss = String::new();
        for k in 0..limits[0] {
            ss.push_str(if k == 0 { "\nSrc: " } else { "; " });
            ss.push_str(solution_source_to_string(k, true));
            ss.push_str(" => ");
            ss.push_str(solution_source_to_string(k, false));
        }
        crate::log_user!(self.log, LogType::Info, "%s;\n", &ss);
        let to_line = limits.len() - 1;
        for line in 0..to_line {
            ss.clear();
            for k in limits[line]..limits[line + 1] {
                ss.push_str(if k == limits[line] { "     " } else { "; " });
                ss.push_str(solution_source_to_string(k, true));
                ss.push_str(" => ");
                ss.push_str(solution_source_to_string(k, false));
            }
            crate::log_user!(self.log, LogType::Info, "%s%s\n", &ss, if line < to_line - 1 { ";" } else { "" });
        }
    }

    /// printDisplayLine
    pub fn print_display_line(&self, solution_source: i32) {
        // no point in computing all the logging values if logging is off
        if !self.opts.output_flag {
            return;
        }
        let sc = self.sc();
        let o = &self.opts;
        let timeless_log = o.timeless_log;
        sc.disptime = if timeless_log { sc.disptime + 1.0 } else { self.timer_read() };
        if solution_source == src::NONE && sc.disptime - sc.last_disptime < o.mip_min_logging_interval {
            return;
        }
        sc.last_disptime = sc.disptime;
        let time_string = if timeless_log { String::new() } else { crate::sprintf!(" %7.1fs", sc.disptime) };
        if sc.num_disp_lines % 20 == 0 {
            if sc.num_disp_lines == 0 {
                self.print_solution_source_key();
            }
            let work_string0 = if timeless_log { "   Work" } else { "      Work      " };
            let work_string1 = if timeless_log { "LpIters" } else { "LpIters     Time" };
            crate::log_user!(
                self.log,
                LogType::Info,
                "\n        Nodes      |    B&B Tree     |            Objective Bounds              |  Dynamic Constraints | %s\nSrc  Proc. InQueue |  Leaves   Expl. | BestBound       BestSol              Gap |   Cuts   InLp Confl. | %s\n\n",
                work_string0,
                work_string1
            );
        }
        sc.num_disp_lines += 1;
        let print_nodes = print_i64(sc.num_nodes);
        let queue_nodes = print_i64(self.nodequeue().num_active_nodes());
        let print_leaves = print_i64(sc.num_leaves - sc.num_leaves_before_run);
        let explored = 100.0 * sc.pruned_treeweight.to_f64();
        let (gap, lb, mut ub) = self.limits_to_gap(sc.lower_bound, sc.upper_bound);
        let gap = gap * 1e2;
        if o.objective_bound < ub {
            ub = o.objective_bound;
        }
        let print_lp_iters = print_i64(sc.total_lp_iterations);
        let lp_rows = self.op(op::LP_NUM_ROWS, None, 0, 0.0) as i64;
        let dynamic_constraints_in_lp =
            if lp_rows > 0 { lp_rows - self.op(op::NUM_MODEL_ROWS, None, 0, 0.0) as i64 } else { 0 };
        let sense = if self.orig_maximize { -1.0 } else { 1.0 };
        let ub_string =
            if o.objective_bound < ub { print_f64(sense * ub, "*") } else { print_f64(sense * ub, "") };
        let lb_string = print_f64(sense * lb, "");
        let ncuts = self.op(op::NUM_CUTS, None, 0, 0.0) as i64;
        let nconfl = self.op(op::NUM_CONFLICTS, None, 0, 0.0) as i64;
        let src_str = solution_source_to_string(solution_source, true);
        if sc.upper_bound != INF {
            let gap_string = if gap >= 9999.0 { "Large".to_string() } else { crate::sprintf!("%.2f%%", gap) };
            crate::log_user!(
                self.log,
                LogType::Info,
                " %s %7s %7s   %7s %6.2f%%   %-15s %-15s %8s   %6d %6d %6d   %7s%s\n",
                src_str,
                &print_nodes,
                &queue_nodes,
                &print_leaves,
                explored,
                &lb_string,
                &ub_string,
                &gap_string,
                ncuts,
                dynamic_constraints_in_lp,
                nconfl,
                &print_lp_iters,
                &time_string
            );
        } else {
            crate::log_user!(
                self.log,
                LogType::Info,
                " %s %7s %7s   %7s %6.2f%%   %-15s %-15s %8.2f   %6d %6d %6d   %7s%s\n",
                src_str,
                &print_nodes,
                &queue_nodes,
                &print_leaves,
                explored,
                &lb_string,
                &ub_string,
                gap,
                ncuts,
                dynamic_constraints_in_lp,
                nconfl,
                &print_lp_iters,
                &time_string
            );
        }
        // possibly interrupt from the MIP logging callback
        self.logging_callback();
    }

    /// checkSolution
    pub fn check_solution(&self, solution: &[f64]) -> bool {
        let feastol = self.feastol();
        let (lo, up, intg) = (self.col_lower(), self.col_upper(), self.integrality());
        for i in 0..self.num_col as usize {
            if solution[i] < lo[i] - feastol || solution[i] > up[i] + feastol {
                return false;
            }
            if intg[i] == 1 && fractionality(solution[i]) > feastol {
                return false;
            }
        }
        self.rows_feasible_double(solution)
    }

    /// The row activities of checkSolution and trySolution (in double)
    fn rows_feasible_double(&self, solution: &[f64]) -> bool {
        let feastol = self.feastol();
        let (ars, ari, arv) = (self.ar_start(), self.ar_index(), self.ar_value());
        let (rl, ru) = (self.row_lower(), self.row_upper());
        for i in 0..self.num_row as usize {
            let mut rowactivity = 0.0;
            for j in ars[i] as usize..ars[i + 1] as usize {
                rowactivity = solution[ari[j] as usize].mul_add_c(arv[j], rowactivity);
            }
            if rowactivity > ru[i] + feastol || rowactivity < rl[i] - feastol {
                return false;
            }
        }
        true
    }

    /// trySolution
    pub fn try_solution_rs(&self, solution: &[f64], solution_source: i32) -> bool {
        match self.checked_objective(solution) {
            Some(obj) => self.add_incumbent_rs(solution, obj, solution_source, true, false),
            None => false,
        }
    }

    /// The bound, integrality and row checks of trySolution: the objective
    /// if they pass
    pub fn checked_objective(&self, solution: &[f64]) -> Option<f64> {
        if solution.len() != self.num_col as usize {
            return None;
        }
        let feastol = self.feastol();
        let (lo, up, intg, cost) = (self.col_lower(), self.col_upper(), self.integrality(), self.col_cost());
        let mut obj = CDouble::from(0.0);
        for i in 0..self.num_col as usize {
            if solution[i] < lo[i] - feastol || solution[i] > up[i] + feastol {
                return None;
            }
            if intg[i] == 1 && fractionality(solution[i]) > feastol {
                return None;
            }
            obj += cost[i] * solution[i];
        }
        if !self.rows_feasible_double(solution) {
            return None;
        }
        Some(obj.to_f64())
    }

    /// solutionRowFeasible (row activities in double-double)
    pub fn solution_row_feasible(&self, solution: &[f64]) -> bool {
        let feastol = self.feastol();
        let (ars, ari, arv) = (self.ar_start(), self.ar_index(), self.ar_value());
        let (rl, ru) = (self.row_lower(), self.row_upper());
        for i in 0..self.num_row as usize {
            let mut act = CDouble::from(0.0);
            for j in ars[i] as usize..ars[i + 1] as usize {
                act += CDouble::from(solution[ari[j] as usize]) * arv[j];
            }
            let rowactivity = act.to_f64();
            if rowactivity > ru[i] + feastol || rowactivity < rl[i] - feastol {
                return false;
            }
        }
        true
    }

    /// trivialHeuristics; returns the model status (infeasible or not set)
    pub fn trivial_heuristics(&self) -> i32 {
        if !self.continuous_cols().is_empty() {
            return status::NOTSET;
        }
        let heuristic_source = [src::TRIVIAL_Z, src::TRIVIAL_L, src::TRIVIAL_U, src::TRIVIAL_P];
        let mut col_lower = self.col_lower().to_vec();
        let mut col_upper = self.col_upper().to_vec();
        let (row_lower, row_upper) = (self.row_lower().to_vec(), self.row_upper().to_vec());
        let integer_cols = self.integer_cols().to_vec();
        let numintegercols = self.sc().numintegercols;
        let mut all_integer_lower_non_positive = true;
        let mut all_integer_lower_zero = true;
        let mut all_integer_lower_finite = true;
        let mut all_integer_upper_finite = true;
        for &c in integer_cols.iter().take(numintegercols.max(0) as usize) {
            let i = c as usize;
            // round bounds in to nearest integer
            col_lower[i] = col_lower[i].ceil();
            col_upper[i] = col_upper[i].floor();
            let legal_bounds = col_lower[i] <= col_upper[i]
                && col_lower[i] < INF
                && col_upper[i] > -INF
                && !col_lower[i].is_nan()
                && !col_upper[i].is_nan();
            if !legal_bounds {
                crate::log_user!(
                    self.log,
                    LogType::Info,
                    "HighsMipSolverData::trivialHeuristics() has detected infeasible/illegal bounds [%g, %g] for column %d: MIP is infeasible\n",
                    col_lower[i],
                    col_upper[i],
                    c
                );
                return status::INFEASIBLE;
            }
            if col_lower[i] > col_upper[i] {
                return status::INFEASIBLE;
            }
            if col_lower[i] > 0.0 {
                all_integer_lower_non_positive = false;
            }
            if col_lower[i] != 0.0 {
                all_integer_lower_zero = false;
            }
            if col_lower[i] <= -INF {
                all_integer_lower_finite = false;
            }
            if col_upper[i] >= INF {
                all_integer_upper_finite = false;
            }
            // only continue if one of the properties still holds
            if !(all_integer_lower_non_positive || all_integer_lower_zero || all_integer_upper_finite) {
                break;
            }
        }
        let all_integer_boxed = all_integer_lower_finite && all_integer_upper_finite;
        let feasibility_tolerance = self.opts.mip_feasibility_tolerance;
        let n = self.num_col as usize;
        let mut solution = vec![0.0; n];
        let (a_start, a_value) = (self.a_start().to_vec(), self.a_value().to_vec());
        for try_heuristic in 0..4 {
            match try_heuristic {
                0 => {
                    // all-zero for the integer variables
                    if !all_integer_lower_non_positive {
                        continue;
                    }
                    let failed = (0..self.num_row as usize)
                        .any(|r| row_lower[r] > feasibility_tolerance || row_upper[r] < -feasibility_tolerance);
                    if failed {
                        continue;
                    }
                    solution.iter_mut().for_each(|x| *x = 0.0);
                }
                1 => {
                    // all-lower (if distinct from all-zero)
                    if all_integer_lower_zero || !self.solution_row_feasible(&col_lower) {
                        continue;
                    }
                    solution.copy_from_slice(&col_lower);
                }
                2 => {
                    // all-upper
                    if !all_integer_upper_finite || !self.solution_row_feasible(&col_upper) {
                        continue;
                    }
                    solution.copy_from_slice(&col_upper);
                }
                _ => {
                    // the lock point
                    if !all_integer_boxed {
                        continue;
                    }
                    for &c in integer_cols.iter().take(numintegercols.max(0) as usize) {
                        let i = c as usize;
                        let mut npos = 0;
                        let mut nneg = 0;
                        for el in a_start[i] as usize..a_start[i + 1] as usize {
                            if a_value[el] > 0.0 {
                                npos += 1;
                            } else {
                                nneg += 1;
                            }
                        }
                        solution[i] = if npos > nneg { col_lower[i] } else { col_upper[i] };
                    }
                    if !self.solution_row_feasible(&solution) {
                        continue;
                    }
                }
            }
            let cost = self.col_cost();
            let mut obj = CDouble::from(0.0);
            for i in 0..n {
                obj += cost[i] * solution[i];
            }
            self.add_incumbent_rs(&solution, obj.to_f64(), heuristic_source[try_heuristic], true, false);
        }
        status::NOTSET
    }

    /// HighsMipSolver::solutionFeasible on the original model, with the row
    /// values given: (feasible, bound violation, row violation,
    /// integrality violation, objective)
    pub fn solution_feasible_orig(&self, col_value: &[f64], row_value: &[f64]) -> (bool, f64, f64, f64, CDouble) {
        let o = &self.orig;
        let tol = self.opts.mip_feasibility_tolerance;
        let mut bound_violation: f64 = 0.0;
        let mut row_violation: f64 = 0.0;
        let mut integrality_violation: f64 = 0.0;
        let mut obj = CDouble::from(o.offset);
        // SAFETY: the original model's vectors, unchanged during the call
        let (cost, intg, lo, up, rl, ru) = unsafe {
            (
                (*o.col_cost).as_slice(),
                (*o.integrality).as_slice(),
                (*o.col_lower).as_slice(),
                (*o.col_upper).as_slice(),
                (*o.row_lower).as_slice(),
                (*o.row_upper).as_slice(),
            )
        };
        for i in 0..o.num_col as usize {
            let value = col_value[i];
            obj += cost[i] * value;
            if intg[i] == 1 {
                integrality_violation = cmax(fractionality(value), integrality_violation);
            }
            let primal_infeasibility = if value < lo[i] - tol {
                lo[i] - value
            } else if value > up[i] + tol {
                value - up[i]
            } else {
                continue;
            };
            bound_violation = cmax(bound_violation, primal_infeasibility);
        }
        for i in 0..o.num_row as usize {
            let value = row_value[i];
            let primal_infeasibility = if value < rl[i] - tol {
                rl[i] - value
            } else if value > ru[i] + tol {
                value - ru[i]
            } else {
                continue;
            };
            row_violation = cmax(row_violation, primal_infeasibility);
        }
        let feasible = bound_violation <= tol && integrality_violation <= tol && row_violation <= tol;
        (feasible, bound_violation, row_violation, integrality_violation, obj)
    }

    /// transformNewIntegerFeasibleSolution's repair LP: the original model
    /// with the integers fixed at their rounded values in `col` (the
    /// scratch solution), solved by simplex in C++ (CMipFns::repair_lp);
    /// if primal feasible, its solution replaces the scratch solution
    fn repair_lp(&self, col: &[f64]) -> bool {
        let o = &self.orig;
        // SAFETY: the original model's vectors, unchanged during the call
        let (intg, lo, up) = unsafe { ((*o.integrality).as_slice(), (*o.col_lower).as_slice(), (*o.col_upper).as_slice()) };
        let mut lower = lo.to_vec();
        let mut upper = up.to_vec();
        for c in 0..o.num_col as usize {
            if intg[c] == 1 {
                let solval = col[c].round();
                lower[c] = cmax(lower[c], solval);
                upper[c] = cmin(upper[c], solval);
            }
        }
        self.sc().total_repair_lp += 1;
        let time_available = cmax(self.opts.time_limit - self.timer_read(), 0.1);
        let mut iterations = 0;
        let feasible = glue::repair_lp(
            self,
            &lower,
            &upper,
            time_available,
            self.opts.mip_feasibility_tolerance,
            !self.root_presolve_only,
            &mut iterations,
        );
        let sc = self.sc();
        sc.total_repair_lp_iterations += iterations;
        if feasible {
            sc.total_repair_lp_feasible += 1;
        }
        feasible
    }

    /// transformNewIntegerFeasibleSolution: the objective in the
    /// transformed space (infinity if not to be used for bounding)
    pub fn transform_new_integer_feasible_solution(&self, sol: &[f64], possibly_store_as_new_incumbent: bool) -> f64 {
        // primal postsolve to the original space, with its row values
        let mut scratch = glue::scratch_solution(self, Some(sol));
        let mut allow_try_again = true;
        let (feasible, bound_violation, row_violation, integrality_violation, quad_obj) = loop {
            let r = self.solution_feasible_orig(scratch.col(), scratch.row());
            if !r.0 && allow_try_again {
                // repair: an LP with the integers fixed at their rounded
                // values
                if self.repair_lp(scratch.col()) {
                    allow_try_again = false;
                    scratch = glue::scratch_solution(self, None);
                    continue;
                }
            }
            break r;
        };
        let objective_value = quad_obj.to_f64();
        let sense = if self.orig_maximize { -1.0 } else { 1.0 };
        let transformed_solobj = (quad_obj * sense - self.offset).to_f64();
        // possible MIP solution callback
        if !self.submip && feasible && self.callback_active(super::setup::cb::MIP_SOLUTION) {
            self.solution_callback(objective_value);
        }
        // the repaired solution may have a worse objective than the stored
        // one
        if transformed_solobj >= self.sc().upper_bound && !sol.is_empty() {
            return transformed_solobj;
        }
        if possibly_store_as_new_incumbent {
            if feasible {
                self.set_violations(bound_violation, integrality_violation, row_violation);
                self.op(op::STORE_SOLUTION, None, 0, objective_value);
            } else {
                let tol = self.opts.mip_feasibility_tolerance;
                let (bv, iv, rv) = self.violations();
                let current_feasible = self.sol_objective() != INF && bv <= tol && iv <= tol && rv <= tol;
                crate::log_user!(
                    self.log,
                    LogType::Warning,
                    "Solution with objective %g has untransformed violations: bound = %.4g; integrality = %.4g; row = %.4g\n",
                    objective_value,
                    bound_violation,
                    integrality_violation,
                    row_violation
                );
                if !current_feasible {
                    self.set_violations(bound_violation, integrality_violation, row_violation);
                    self.op(op::STORE_SOLUTION, None, 0, objective_value);
                }
                return INF;
            }
        }
        transformed_solobj
    }

    /// addIncumbent
    pub fn add_incumbent_rs(
        &self,
        sol: &[f64],
        mut solobj: f64,
        solution_source: i32,
        print_display_line: bool,
        is_user_solution: bool,
    ) -> bool {
        let execute_mip_solution_callback =
            !is_user_solution && !self.submip && self.callback_active(super::setup::cb::MIP_SOLUTION);
        let possibly_store_as_new_incumbent = solobj < self.sc().upper_bound;
        let get_transformed_solution = possibly_store_as_new_incumbent || execute_mip_solution_callback;
        let transformed_solobj = if get_transformed_solution {
            self.transform_new_integer_feasible_solution(sol, possibly_store_as_new_incumbent)
        } else {
            0.0
        };
        if possibly_store_as_new_incumbent {
            solobj = transformed_solobj;
            let sc = self.sc();
            if solobj >= sc.upper_bound {
                return false;
            }
            let prev_upper_bound = sc.upper_bound;
            sc.upper_bound = solobj;
            self.op(op::SYNC_WORKERS, None, 0, 0.0);
            if !self.submip && sc.upper_bound != prev_upper_bound {
                let lb = sc.lower_bound;
                self.update_primal_dual_integral(lb, lb, prev_upper_bound, sc.upper_bound, true, true);
            }
            glue::set_vec(self, vec::INCUMBENT, sol);
            self.concurrent_offer(sc.upper_bound);
            let new_upper_limit = self.compute_new_upper_limit(solobj, 0.0, 0.0);
            if !is_user_solution && !self.submip {
                self.save_report_mip_solution(new_upper_limit);
            }
            if new_upper_limit < sc.upper_limit {
                sc.num_improving_sols += 1;
                sc.upper_limit = new_upper_limit;
                sc.optimality_limit =
                    self.compute_new_upper_limit(solobj, self.opts.mip_abs_gap, self.opts.mip_rel_gap);
                self.nodequeue().set_optimality_limit(sc.optimality_limit);
                // a helper's solution within the target gap of its main
                // solver's bound finishes the main solve
                self.concurrent_target(sc.optimality_limit);
                self.op(op::SYNC_WORKERS, None, 1, 0.0);
                self.op(op::DEBUG_NEW_INCUMBENT, None, 0, 0.0);
                let gd = self.domain();
                gd.propagate();
                if !gd.infeasible() {
                    self.op(op::PROPAGATE_ROOT_REDCOST, None, 0, 0.0);
                }
                if gd.infeasible() {
                    sc.pruned_treeweight = CDouble::from(1.0);
                    self.nodequeue().clear();
                    if print_display_line {
                        self.print_display_line(solution_source);
                    }
                    return true;
                }
                self.op(op::EXTRACT_OBJ_CLIQUES, None, 0, 0.0);
                if gd.infeasible() {
                    sc.pruned_treeweight = CDouble::from(1.0);
                    self.nodequeue().clear();
                    if print_display_line {
                        self.print_display_line(solution_source);
                    }
                    return true;
                }
                let ul = sc.upper_limit;
                sc.pruned_treeweight += self.nodequeue().perform_bounding(ul);
                self.print_display_line(solution_source);
            }
        } else if self.incumbent().is_empty() {
            glue::set_vec(self, vec::INCUMBENT, sol);
        }
        true
    }

    // ---- the root node ----

    /// evaluateRootLp
    pub fn evaluate_root_lp(&self, w: &Worker) -> i32 {
        let lp = Lp::borrowed(self.lp);
        let gd = self.domain();
        loop {
            gd.propagate();
            if !gd.infeasible() {
                self.op(op::ORBITAL_FIXING, None, 0, 0.0);
            }
            let sc = self.sc();
            if gd.infeasible() {
                self.update_lower_bound_ex(cmin(INF, sc.upper_bound), true, true);
                sc.pruned_treeweight = CDouble::from(1.0);
                sc.num_nodes += 1;
                sc.num_leaves += 1;
                return lp_status::INFEASIBLE;
            }
            let mut lp_bounds_changed = false;
            if gd.num_changed_cols() != 0 {
                lp_bounds_changed = true;
                self.remove_fixed_indices();
                lp.flush_domain(&gd);
            }
            let mut lp_was_solved = false;
            let status;
            if lp_bounds_changed || self.op(op::LP_MODEL_STATUS_NOTSET, None, 0, 0.0) != 0.0 {
                let mut lp_iters = -lp.num_lp_iterations();
                status = lp.resolve(Some(&gd));
                lp_iters += lp.num_lp_iterations();
                sc.total_lp_iterations += lp_iters;
                sc.avgrootlpiters = lp.avg_solve_iters();
                lp_was_solved = true;
                if status == lp_status::UNBOUNDED {
                    if self.op(op::SOLUTION_EMPTY, None, 0, 0.0) != 0.0 {
                        self.set_modelstatus(status::UNBOUNDED_OR_INFEASIBLE);
                    } else {
                        self.set_modelstatus(status::UNBOUNDED);
                    }
                    sc.pruned_treeweight = CDouble::from(1.0);
                    sc.num_nodes += 1;
                    sc.num_leaves += 1;
                    return status;
                }
                if status == lp_status::OPTIMAL && lp.frac().is_empty() {
                    let sol = lp.col_value().to_vec();
                    if self.add_incumbent_rs(&sol, lp.objective(), src::EVALUATE_NODE, true, false) {
                        self.set_modelstatus(status::OPTIMAL);
                        let ub = sc.upper_bound;
                        self.update_lower_bound_ex(ub, true, true);
                        sc.pruned_treeweight = CDouble::from(1.0);
                        sc.num_nodes += 1;
                        sc.num_leaves += 1;
                        return lp_status::INFEASIBLE;
                    }
                }
                if status == lp_status::OPTIMAL && self.opts.run_zi_round {
                    self.op(op::HEUR_ZI_ROUND, Some(w), 0, 0.0);
                }
            } else {
                status = lp.status();
            }
            if status == lp_status::INFEASIBLE {
                self.update_lower_bound_ex(cmin(INF, sc.upper_bound), true, true);
                sc.pruned_treeweight = CDouble::from(1.0);
                sc.num_nodes += 1;
                sc.num_leaves += 1;
                return status;
            }
            let lps = lp.status();
            if lps == lp_status::OPTIMAL || lps == lp_status::UNSCALED_DUAL_FEASIBLE {
                let lb = sc.lower_bound;
                self.update_lower_bound_ex(cmax(lp.objective(), lb), true, true);
                if lp_was_solved {
                    self.op(op::ADD_ROOT_REDCOST, None, 0, 0.0);
                    if sc.upper_limit != INF {
                        self.op(op::PROPAGATE_ROOT_REDCOST, None, 0, 0.0);
                    }
                }
            }
            if sc.lower_bound > sc.optimality_limit {
                sc.pruned_treeweight = CDouble::from(1.0);
                sc.num_nodes += 1;
                sc.num_leaves += 1;
                return lp_status::INFEASIBLE;
            }
            if gd.num_changed_cols() == 0 {
                return status;
            }
        }
    }

    /// The global domain
    pub fn domain(&self) -> Dom {
        Dom::borrowed(self.globaldom)
    }
    pub fn nodequeue(&self) -> &mut super::nodequeue::NodeQueue {
        // SAFETY: the solver's node queue; no other reference to it is live
        // during a use
        unsafe { &mut *self.nodequeue }
    }
    pub fn sol_objective(&self) -> f64 {
        // SAFETY: HighsMipSolver's field
        unsafe { *self.solution.objective }
    }
    /// (bound, integrality, row) violations of the solver's solution
    pub fn violations(&self) -> (f64, f64, f64) {
        let s = &self.solution;
        // SAFETY: HighsMipSolver's fields
        unsafe { (*s.bound_violation, *s.integrality_violation, *s.row_violation) }
    }
    pub fn set_violations(&self, bound: f64, integrality: f64, row: f64) {
        let s = &self.solution;
        // SAFETY: HighsMipSolver's fields, not referenced elsewhere
        unsafe {
            *s.bound_violation = bound;
            *s.integrality_violation = integrality;
            *s.row_violation = row;
        }
    }

    /// removeFixedIndices (vectors set through C++: the integral columns
    /// are read in place by the rest of the solver)
    pub fn remove_fixed_indices(&self) {
        let b = self.domain().bnd();
        for which in [vec::INTEGRAL_COLS, vec::INTEGER_COLS, vec::IMPLINT_COLS, vec::CONTINUOUS_COLS] {
            let v: Vec<i32> =
                glue::int_vec(self, which).iter().copied().filter(|&c| b.lo(c as usize) != b.up(c as usize)).collect();
            glue::set_int_vec(self, which, &v);
        }
    }
}

pub use glue::MipScalars;

// ---- C interface (HighsMipSolverData.cpp under HIGHS_RUST) ----

pub mod ffi {
    use super::*;
    use crate::ffi::sl;
    use glue::{set_fns, CMipFns};

    unsafe fn md<'a>(f: *const CMipFns, m: *const MipData) -> &'a MipData {
        set_fns(f);
        &*m
    }

    /// checkLimits
    ///
    /// # Safety
    /// `f` the C++ functions, `m` filled for this call
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_mip_check_limits(f: *const CMipFns, m: *const MipData, node_offset: i64) -> bool {
        md(f, m).check_limits_rs(node_offset)
    }

    /// limitsToGap
    ///
    /// # Safety
    /// as highs_rs_mip_check_limits
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_mip_limits_to_gap(
        f: *const CMipFns,
        m: *const MipData,
        lower: f64,
        upper: f64,
        lb: *mut f64,
        ub: *mut f64,
    ) -> f64 {
        let (gap, l, u) = md(f, m).limits_to_gap(lower, upper);
        *lb = l;
        *ub = u;
        gap
    }

    /// computeNewUpperLimit
    ///
    /// # Safety
    /// as highs_rs_mip_check_limits
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_mip_new_upper_limit(
        f: *const CMipFns,
        m: *const MipData,
        ub: f64,
        abs_gap: f64,
        rel_gap: f64,
    ) -> f64 {
        md(f, m).compute_new_upper_limit(ub, abs_gap, rel_gap)
    }

    /// limitsToBounds
    ///
    /// # Safety
    /// as highs_rs_mip_check_limits
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_mip_limits_to_bounds(
        f: *const CMipFns,
        m: *const MipData,
        dual_bound: *mut f64,
        primal_bound: *mut f64,
        gap: *mut f64,
    ) {
        let (d, p, g) = md(f, m).limits_to_bounds();
        *dual_bound = d;
        *primal_bound = p;
        *gap = g;
    }

    /// updateLowerBound
    ///
    /// # Safety
    /// as highs_rs_mip_check_limits
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_mip_update_lower_bound(
        f: *const CMipFns,
        m: *const MipData,
        lb: f64,
        check_bound_change: bool,
        check_prev_data: bool,
    ) {
        md(f, m).update_lower_bound_ex(lb, check_bound_change, check_prev_data)
    }

    /// updatePrimalDualIntegral
    ///
    /// # Safety
    /// as highs_rs_mip_check_limits
    #[no_mangle]
    #[allow(clippy::too_many_arguments)]
    pub unsafe extern "C" fn highs_rs_mip_update_pdi(
        f: *const CMipFns,
        m: *const MipData,
        from_lb: f64,
        to_lb: f64,
        from_ub: f64,
        to_ub: f64,
        check_bound_change: bool,
        check_prev_data: bool,
    ) {
        md(f, m).update_primal_dual_integral(from_lb, to_lb, from_ub, to_ub, check_bound_change, check_prev_data)
    }

    /// 0 moreHeuristicsAllowed, 1 percentageInactiveIntegers, 2
    /// trivialHeuristics (the model status), 3 removeFixedIndices
    ///
    /// # Safety
    /// as highs_rs_mip_check_limits
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_mip_query(f: *const CMipFns, m: *const MipData, which: i32) -> f64 {
        let m = md(f, m);
        match which {
            0 => m.more_heuristics_allowed() as i32 as f64,
            1 => m.percentage_inactive_integers(),
            2 => m.trivial_heuristics() as f64,
            _ => {
                m.remove_fixed_indices();
                0.0
            }
        }
    }

    /// printDisplayLine
    ///
    /// # Safety
    /// as highs_rs_mip_check_limits
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_mip_print_display_line(f: *const CMipFns, m: *const MipData, source: i32) {
        md(f, m).print_display_line(source)
    }

    /// 0 checkSolution, 1 solutionRowFeasible, 2 trySolution(source)
    ///
    /// # Safety
    /// as highs_rs_mip_check_limits; `sol` holds `n` values
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_mip_solution(
        f: *const CMipFns,
        m: *const MipData,
        which: i32,
        sol: *const f64,
        n: i32,
        source: i32,
    ) -> bool {
        let m = md(f, m);
        let sol = sl(sol, n).to_vec();
        match which {
            0 => m.check_solution(&sol),
            1 => m.solution_row_feasible(&sol),
            _ => m.try_solution_rs(&sol, source),
        }
    }

    /// addIncumbent
    ///
    /// # Safety
    /// as highs_rs_mip_solution
    #[no_mangle]
    #[allow(clippy::too_many_arguments)]
    pub unsafe extern "C" fn highs_rs_mip_add_incumbent(
        f: *const CMipFns,
        m: *const MipData,
        sol: *const f64,
        n: i32,
        obj: f64,
        source: i32,
        print_display_line: bool,
        is_user_solution: bool,
    ) -> bool {
        let m = md(f, m);
        let sol = sl(sol, n).to_vec();
        m.add_incumbent_rs(&sol, obj, source, print_display_line, is_user_solution)
    }

    /// transformNewIntegerFeasibleSolution
    ///
    /// # Safety
    /// as highs_rs_mip_solution
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_mip_transform(
        f: *const CMipFns,
        m: *const MipData,
        sol: *const f64,
        n: i32,
        store: bool,
    ) -> f64 {
        let m = md(f, m);
        let sol = sl(sol, n).to_vec();
        m.transform_new_integer_feasible_solution(&sol, store)
    }

    /// evaluateRootLp
    ///
    /// # Safety
    /// as highs_rs_mip_check_limits; `w` the worker
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_mip_evaluate_root_lp(
        f: *const CMipFns,
        m: *const MipData,
        w: *mut std::ffi::c_void,
    ) -> i32 {
        let m = md(f, m);
        let w = Worker::new(w);
        m.evaluate_root_lp(&w)
    }
}

/// evaluateRootNode
///
/// # Safety
/// as ffi::highs_rs_mip_check_limits; `w` the worker
#[no_mangle]
pub unsafe extern "C" fn highs_rs_mip_evaluate_root_node(
    f: *const glue::CMipFns,
    m: *const MipData,
    w: *mut std::ffi::c_void,
) {
    glue::set_fns(f);
    super::root::evaluate_root_node(m as *mut MipData, w)
}
