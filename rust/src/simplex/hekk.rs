//! HEkk::solve (highs/simplex/HEkk.cpp) and the HEkk methods it reaches
//! beyond the numerical kernels of ekk.rs: initialiseForSolve, the INVERT
//! with backtracking (computeFactor, getNonsingularInverse, the
//! backtracking basis), the row-wise matrix, the factor's solution error,
//! restoring saved edge weights, the proof of primal infeasibility, the
//! quad precision pivotal row, returnFromSolve, and the logging of all
//! this. The dual (dual.rs) and primal (primal.rs) simplex solvers it runs
//! call these directly.
//!
//! # Ownership
//!
//! HEkk's data stays C++-owned. `HEkk::rustHekk()` (highs/simplex/
//! HEkkRust.cpp) fills a [`CHekk`]: the [`CEkk`] view of HEkk's vectors
//! plus pointers to the scalars of HEkk, its info_ and status_ that the
//! solve reads and writes, the values of the options, and the [`Host`]
//! functions. Before calling [`solve`], C++ sizes every vector that the
//! solve may otherwise resize (the basis, the work and base arrays, the
//! row-wise matrix, the edge weights and their backtracking copy, the
//! backtracking basis, the random vectors), so that the views stay valid
//! for the whole solve. After it, C++ takes what Rust left for its
//! vectors (`BasisRecords::out`: the hot start record, the primal phase 1
//! duals), syncs the factor's build results into the C++ HFactor, and
//! does HEkk::returnFromEkkSolve.
//!
//! # What still calls C++
//!
//! Only through [`Host`]: the log messages (formatted here with C's
//! printf rules, see util/printf.rs, and printed by highsLogUser or
//! highsLogDev; the simplex reports are formatted in report.rs); the run
//! clock (read once per
//! solver when there is a time limit); a user interrupt callback; and,
//! rarely, the handling of a rank deficient initial basis and the debug
//! check of a rank deficient INVERT.

use crate::util::fma::ClangFma;

use std::cell::Cell;
use std::ffi::{c_char, c_void, CString};

use super::basis_records::{BasisRecords, HotStart, REASON_ALL};
use super::dual::Dual;
use super::report::SimplexReport;
use super::dual_row::WorkPair;
use super::ekk::{
    choose_price_technique, update_operation_result_density, CEkk, CSlice,
    CostPerturbationReport, EkkView, SimplexStatus,
};
use super::primal::Primal;
use crate::factor::BUILD_KERNEL_RETURN_TIMEOUT;
use crate::hvector::{HVec, OwnedHVec, K_HIGHS_TINY, K_HIGHS_ZERO};
use crate::sprintf;
use crate::util::cdouble::CDouble;
use crate::util::hash;
use crate::util::random::HighsRandom;
use crate::util::sparse_vector_sum::HighsSparseVectorSum;

pub const INF: f64 = f64::INFINITY;
/// kHighsIInf for 32-bit HighsInt
pub const K_HIGHS_IINF: i32 = i32::MAX;

// HighsLogType
pub const LOG_INFO: i32 = 1;
pub const LOG_DETAILED: i32 = 2;
pub const LOG_VERBOSE: i32 = 3;
pub const LOG_WARNING: i32 = 4;
pub const LOG_ERROR: i32 = 5;

// Channels of Host::log
const CHANNEL_USER: i32 = 0;
const CHANNEL_DEV: i32 = 1;
const CHANNEL_FACTOR_DEV: i32 = 2;
const CHANNEL_PRINTF: i32 = 3;

// HighsStatus
pub const STATUS_ERROR: i32 = -1;
pub const STATUS_OK: i32 = 0;
pub const STATUS_WARNING: i32 = 1;

// HighsModelStatus
pub const MS_NOTSET: i32 = 0;
pub const MS_SOLVE_ERROR: i32 = 4;
pub const MS_OPTIMAL: i32 = 7;
pub const MS_INFEASIBLE: i32 = 8;
pub const MS_UNBOUNDED_OR_INFEASIBLE: i32 = 9;
pub const MS_UNBOUNDED: i32 = 10;
pub const MS_OBJECTIVE_BOUND: i32 = 11;
pub const MS_OBJECTIVE_TARGET: i32 = 12;
pub const MS_TIME_LIMIT: i32 = 13;
pub const MS_ITERATION_LIMIT: i32 = 14;
pub const MS_UNKNOWN: i32 = 15;
pub const MS_INTERRUPT: i32 = 17;

// SimplexAlgorithm
pub const ALGORITHM_PRIMAL: i32 = 1;
pub const ALGORITHM_DUAL: i32 = 2;

// SimplexStrategy
const STRATEGY_CHOOSE: i32 = 0;
const STRATEGY_DUAL_PLAIN: i32 = 1;
const STRATEGY_DUAL_TASKS: i32 = 2;
const STRATEGY_DUAL_MULTI: i32 = 3;
const STRATEGY_PRIMAL: i32 = 4;

// Solve phases
const PHASE_UNKNOWN: i32 = -1;
const PHASE_2: i32 = 2;

// Rebuild reasons
const RR_NO: i32 = 0;
const RR_POSSIBLY_OPTIMAL: i32 = 3;
const RR_POSSIBLY_PHASE1_FEASIBLE: i32 = 4;
const RR_POSSIBLY_PRIMAL_UNBOUNDED: i32 = 5;
const RR_POSSIBLY_DUAL_UNBOUNDED: i32 = 6;
const RR_PRIMAL_INFEASIBLE_IN_PRIMAL_SIMPLEX: i32 = 8;

// HighsSolutionStatus
const SOLUTION_STATUS_NONE: i32 = 0;
const SOLUTION_STATUS_INFEASIBLE: i32 = 1;
const SOLUTION_STATUS_FEASIBLE: i32 = 2;

const K_ILLEGAL_INFEASIBILITY_COUNT: i32 = -1;
const K_HYPER_PRICE_DENSITY: f64 = 0.1;

/// A pointer to an HEkk scalar, valid for the solve, which C++ may also
/// write during host calls: read and written by value
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Shared<T>(*mut T);

impl<T: Copy> Shared<T> {
    /// # Safety (of its uses)
    /// `p` must point to a value live for the views using it
    #[inline]
    pub fn new(p: *mut T) -> Self {
        Shared(p)
    }
    #[inline]
    pub fn get(self) -> T {
        // SAFETY: C++ passes a pointer into HEkk, live for the solve, that
        // is not otherwise borrowed by Rust
        unsafe { self.0.read() }
    }
    #[inline]
    pub fn set(self, value: T) {
        // SAFETY: as for get
        unsafe { self.0.write(value) }
    }
}

type Ctx = *mut c_void;

/// The C++ that the solve calls: see the module comment. Filled by
/// highs/simplex/HEkkRust.cpp, `ctx` being the HEkk
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Host {
    pub ctx: Ctx,
    /// (channel, HighsLogType, message): channel 0 for highsLogUser, 1 for
    /// highsLogDev, 2 for highsLogDev with the factor's log options, 3 for
    /// printf
    pub log: extern "C" fn(Ctx, i32, i32, *const c_char),
    /// HEkk::timer_->read()
    pub timer_read: extern "C" fn(Ctx) -> f64,
    /// The user interrupt part of HEkk::bailout: whether the user
    /// interrupts
    pub interrupt: extern "C" fn(Ctx) -> bool,
    /// debugDualChuzcFailQuad0 (kind 1) or Quad1 (2)
    pub chuzc_fail: extern "C" fn(Ctx, i32, i32, *const WorkPair, f64, f64),
}

/// HEkk's data for the solve: see the module comment. Mirrored by
/// highs_rs::Hekk in highs/simplex/HEkkRust.h
pub struct CHekk {
    pub ekk: CEkk,
    pub host: Host,
    // HEkk
    pub iteration_count: Shared<i32>,
    pub model_status: Shared<i32>,
    pub solve_bailout: Shared<bool>,
    pub called_return_from_solve: Shared<bool>,
    pub exit_algorithm: Shared<i32>,
    pub return_primal_solution_status: Shared<i32>,
    pub return_dual_solution_status: Shared<i32>,
    pub dual_values_valid: Shared<bool>,
    pub dual_values_scaled: Shared<bool>,
    pub dual_values_basis_hash: Shared<u64>,
    pub dual_values_cost_hash: Shared<u64>,
    pub fresh_unperturbed_dual: Shared<bool>,
    pub fresh_dual: Shared<bool>,
    pub fresh_primal: Shared<bool>,
    pub edge_weight_error: Shared<f64>,
    pub dual_simplex_cleanup_level: Shared<i32>,
    pub dual_simplex_phase1_cleanup_level: Shared<i32>,
    pub previous_iteration_cycling_detected: Shared<i32>,
    pub random: Shared<u64>,
    pub basis_records: *mut BasisRecords,
    pub nla_build_synthetic_tick: Shared<f64>,
    pub num_invert: Shared<i32>,
    pub debug_solve_call_num: i32,
    pub ar_matrix_is_scaled: Shared<bool>,
    pub random_vectors_drawn: Shared<bool>,
    /// Whether initialiseForSolve is to draw the random vectors (sized by
    /// C++)
    pub draw_random_vectors: bool,
    // lp_
    pub lp_is_scaled: bool,
    pub lp_has_scaling: bool,
    pub lp_col_scale: CSlice<f64>,
    pub lp_row_scale: CSlice<f64>,
    pub model_name: CSlice<u8>,
    /// saved_dual_edge_weight_, and whether it has been taken (moved out)
    pub saved_dual_edge_weight: Cell<CSlice<f64>>,
    pub saved_dual_edge_weight_taken: Cell<bool>,
    /// The vector of saved_dual_edge_weight, which a rank deficient
    /// initial basis replaces
    pub saved_dual_edge_weight_vec: *mut Vec<f64>,
    /// basis_.debug_origin_name
    pub basis_origin: *const String,
    // dual_ray_record_ and primal_ray_record_ (index and sign); C++
    // clears the value of each whose bit (1: dual, 2: primal) is set in
    // ray_value_clear
    pub dual_ray_index: Shared<i32>,
    pub dual_ray_sign: Shared<i32>,
    pub primal_ray_index: Shared<i32>,
    pub primal_ray_sign: Shared<i32>,
    pub ray_value_clear: Cell<i32>,
    // basis_
    pub basis_debug_id: Shared<i32>,
    pub basis_debug_update_count: Shared<i32>,
    // status_ (also in ekk.status)
    pub has_invert: Shared<bool>,
    pub has_fresh_invert: Shared<bool>,
    pub has_fresh_rebuild: Shared<bool>,
    pub has_dual_objective_value: Shared<bool>,
    pub has_primal_objective_value: Shared<bool>,
    pub has_dual_steepest_edge_weights: Shared<bool>,
    pub has_ar_matrix: Shared<bool>,
    // info_: the backtracking basis
    pub valid_backtracking_basis: Shared<bool>,
    pub bt_basic_index: CSlice<i32>,
    pub bt_nonbasic_flag: CSlice<i8>,
    pub bt_nonbasic_move: CSlice<i8>,
    pub bt_hash: Shared<u64>,
    pub bt_debug_id: Shared<i32>,
    pub bt_debug_update_count: Shared<i32>,
    pub bt_costs_shifted: Shared<i32>,
    pub bt_costs_perturbed: Shared<i32>,
    pub bt_bounds_shifted: Shared<i32>,
    pub bt_bounds_perturbed: Shared<i32>,
    pub bt_work_shift: CSlice<f64>,
    pub bt_edge_weight: CSlice<f64>,
    // info_
    pub devex_index: CSlice<i32>,
    pub num_tot_permutation: CSlice<i32>,
    pub num_col_permutation: CSlice<i32>,
    pub dual_phase1_iteration_count: Shared<i32>,
    pub dual_phase2_iteration_count: Shared<i32>,
    pub allow_cost_shifting: Shared<bool>,
    pub allow_cost_perturbation: Shared<bool>,
    pub store_squared_primal_infeasibility: Shared<bool>,
    pub factor_pivot_threshold: Shared<f64>,
    pub col_bfrt_density: Shared<f64>,
    pub costly_dse_measure: Shared<f64>,
    pub costly_dse_frequency: Shared<f64>,
    pub num_costly_dse_iteration: Shared<i32>,
    pub average_log_low_dse_weight_error: Shared<f64>,
    pub average_log_high_dse_weight_error: Shared<f64>,
    pub simplex_strategy: Shared<i32>,
    pub min_concurrency: Shared<i32>,
    pub max_concurrency: Shared<i32>,
    pub num_concurrency: Shared<i32>,
    pub iteration_count0: Shared<i32>,
    pub dual_phase1_iteration_count0: Shared<i32>,
    pub dual_phase2_iteration_count0: Shared<i32>,
    pub primal_phase1_iteration_count0: Shared<i32>,
    pub primal_phase2_iteration_count0: Shared<i32>,
    pub primal_bound_swap0: Shared<i32>,
    // info_ values, constant during the solve
    pub control_iteration_count0: i32,
    pub allow_dual_steepest_edge_to_devex_switch: bool,
    pub dual_steepest_edge_weight_log_error_threshold: f64,
    pub dual_edge_weight_strategy: i32,
    pub run_quiet: bool,
    // simplex_nla_.factor_
    pub hfactor_pivot_threshold: Shared<f64>,
    pub hfactor_pivot_tolerance: f64,
    pub hfactor_time_limit: f64,
    // options_
    pub objective_bound: f64,
    pub time_limit: f64,
    pub simplex_iteration_limit: i32,
    pub simplex_update_limit: i32,
    pub max_dual_simplex_cleanup_level: i32,
    pub max_dual_simplex_phase1_cleanup_level: i32,
    pub dual_simplex_pivot_growth_tolerance: f64,
    pub simplex_dse_exact_init_max_rows: i32,
    pub small_matrix_value: f64,
    pub dual_steepest_edge_weight_error_tolerance: f64,
    pub no_unnecessary_rebuild_refactor: bool,
    pub rebuild_refactor_solution_error_tolerance: f64,
    pub option_simplex_strategy: i32,
    pub simplex_min_concurrency: i32,
    pub simplex_max_concurrency: i32,
    pub allow_unbounded_or_infeasible: bool,
    pub less_infeasible_dse_check: bool,
    pub less_infeasible_dse_choose_row: bool,
    /// highs::parallel::num_threads()
    pub num_threads: i32,
    /// The log level for highsLogDev: 0 if it cannot print
    pub dev_level: i32,
    /// options_->output_flag
    pub output_flag: bool,
    /// options_->log_dev_level
    pub log_dev_level: i32,
    /// HFactor's log level for highsLogDev (0 if it cannot print)
    pub factor_dev_level: i32,
    /// Whether highsLogDev can print anything
    pub dev_log: bool,
    /// Whether iteration reports are logged (log_dev_level >= kVerbose)
    pub iteration_report: bool,
    /// Whether a user callback for simplex interrupts is active
    pub interrupt_callback: bool,
    /// HighsSimplexAnalysis's report data and header counters
    pub report: Shared<SimplexReport>,
}

/// Whether highsLogDev prints a message of type `t` at log level `level`
#[inline]
fn dev_prints(level: i32, t: i32) -> bool {
    level > 0 && !(t == LOG_DETAILED && level < 2) && !(t == LOG_VERBOSE && level < 3)
}

impl CHekk {
    fn emit(&self, channel: i32, t: i32, msg: &str) {
        let c = CString::new(msg).unwrap_or_default();
        (self.host.log)(self.host.ctx, channel, t, c.as_ptr());
    }

    /// Whether a highsLogDev message of type `t` would print
    #[inline]
    pub fn dev_on(&self, t: i32) -> bool {
        dev_prints(self.dev_level, t)
    }

    /// highsLogDev: the message is only made if it prints
    #[inline]
    pub fn dev(&self, t: i32, msg: impl FnOnce() -> String) {
        if self.dev_on(t) {
            self.emit(CHANNEL_DEV, t, &msg());
        }
    }

    /// highsLogUser
    #[inline]
    pub fn user(&self, t: i32, msg: &str) {
        if self.output_flag {
            self.emit(CHANNEL_USER, t, msg);
        }
    }

    /// printf to stdout
    pub fn printf(&self, msg: &str) {
        self.emit(CHANNEL_PRINTF, LOG_INFO, msg);
    }

    /// highsLogDev with HFactor's log options
    fn factor_dev(&self, t: i32, msg: impl FnOnce() -> String) {
        if dev_prints(self.factor_dev_level, t) {
            self.emit(CHANNEL_FACTOR_DEV, t, &msg());
        }
    }

    pub fn records(&self) -> &mut BasisRecords {
        // SAFETY: HEkk's records, live for the solve, touched by nothing
        // else during it; callers do not hold two borrows at once
        unsafe { &mut *self.basis_records }
    }

    pub fn model_name(&self) -> String {
        // SAFETY: lp_.model_name_'s bytes
        String::from_utf8_lossy(unsafe { self.model_name.get() }).into_owned()
    }

    /// A fresh view of HEkk's data
    ///
    /// # Safety
    /// As for CEkk::view; the caller must not use another view while
    /// using this one
    pub unsafe fn view<'a>(&self) -> EkkView<'a> {
        self.ekk.view()
    }

    /// HEkk::clearFreshValues
    pub fn clear_fresh_values(&self) {
        self.fresh_unperturbed_dual.set(false);
        self.fresh_dual.set(false);
        self.fresh_primal.set(false);
    }
}

/// highsStatusToString
fn status_string(status: i32) -> &'static str {
    match status {
        STATUS_OK => "OK",
        STATUS_WARNING => "Warning",
        STATUS_ERROR => "Error",
        _ => "Unrecognised HiGHS status",
    }
}

/// utilModelStatusToString
pub fn model_status_string(status: i32) -> &'static str {
    match status {
        0 => "Not Set",
        1 => "Load error",
        2 => "Model error",
        3 => "Presolve error",
        4 => "Solve error",
        5 => "Postsolve error",
        6 => "Empty",
        7 => "Optimal",
        8 => "Infeasible",
        9 => "Primal infeasible or unbounded",
        10 => "Unbounded",
        11 => "Bound on objective reached",
        12 => "Target for objective reached",
        13 => "Time limit reached",
        14 => "Iteration limit reached",
        15 => "Unknown",
        16 => "Solution limit reached",
        17 => "Interrupted by user",
        18 => "Memory limit reached",
        19 => "Interrupted by HiGHS",
        _ => "Unrecognised HiGHS model status",
    }
}

/// HEkk::rebuildReason
pub fn rebuild_reason_string(rebuild_reason: i32) -> &'static str {
    match rebuild_reason {
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

/// worseStatus
fn worse_status(a: i32, b: i32) -> i32 {
    if a == STATUS_ERROR || b == STATUS_ERROR {
        STATUS_ERROR
    } else if a == STATUS_WARNING || b == STATUS_WARNING {
        STATUS_WARNING
    } else {
        STATUS_OK
    }
}

/// interpretCallStatus
pub fn interpret_call_status(x: &CHekk, call_status: i32, from_return_status: i32, message: &str) -> i32 {
    let to_return_status = worse_status(call_status, from_return_status);
    if call_status != STATUS_OK {
        x.dev(LOG_WARNING, || {
            sprintf!("%s return of HighsStatus::%s\n", message, status_string(call_status))
        });
    }
    to_return_status
}

/// std::max(a, b)
#[inline]
fn std_max(a: f64, b: f64) -> f64 {
    if a < b {
        b
    } else {
        a
    }
}

/// nearestPowerOfTwoScale (util/HighsUtils.cpp): 2^-e where value = x*2^e
/// with x in (0.5, 1], so that value*scale is in (0.5, 1]
pub fn nearest_power_of_two_scale(value: f64) -> f64 {
    if value == 0.0 || !value.is_finite() {
        return 1.0;
    }
    // frexp
    let mut v = value.abs();
    let mut e_adjust = 0;
    if v < f64::MIN_POSITIVE {
        v *= 2f64.powi(54);
        e_adjust = -54;
    }
    let bits = v.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    let mut exp = biased - 1022 + e_adjust;
    let mantissa_zero = bits & ((1u64 << 52) - 1) == 0;
    // |x| == 0.5: value is a power of two
    if mantissa_zero {
        exp -= 1;
    }
    let n = -exp;
    if (-1022..=1023).contains(&n) {
        f64::from_bits(((n + 1023) as u64) << 52)
    } else {
        2f64.powi(n)
    }
}

/// HEkk::getValueScale
pub fn get_value_scale(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 1.0;
    }
    nearest_power_of_two_scale(max_abs(values))
}

/// `max_abs_value = std::max(fabs(value), max_abs_value)` over the values
/// from 0. Without NaNs that is the largest |value|, which four maxima find
/// without a serial chain of compares; a NaN takes the serial loop.
fn max_abs(values: &[f64]) -> f64 {
    // std::max(a, m): m if a < m, else a
    let max = |m: f64, a: f64| if a < m { m } else { a };
    let mut lane = [0.0f64; 4];
    let mut nan = false;
    let mut blocks = values.chunks_exact(4);
    for b in &mut blocks {
        for k in 0..4 {
            let a = b[k].abs();
            nan |= a.is_nan();
            lane[k] = max(lane[k], a);
        }
    }
    if nan {
        return values.iter().fold(0.0, |m, v| max(m, v.abs()));
    }
    let m = max(max(lane[0], lane[1]), max(lane[2], lane[3]));
    blocks.remainder().iter().fold(m, |m, v| max(m, v.abs()))
}

// ---- HEkk methods on the view ----

/// HEkk::costHash
pub fn cost_hash(e: &EkkView) -> u64 {
    // SAFETY: the bytes of the costs
    let bytes = unsafe {
        std::slice::from_raw_parts(e.col_cost.as_ptr() as *const u8, std::mem::size_of_val(e.col_cost))
    };
    hash::vector_hash(bytes) ^ hash::hash_bytes(&e.offset.to_ne_bytes()).wrapping_add(e.num_row as i64 as u64)
}

/// The dual values are those of the LP's costs for this basis: recorded so
/// that a re-solve after bound changes alone need not recompute them
pub fn record_dual_values(e: &EkkView, x: &CHekk) {
    x.dual_values_valid.set(true);
    x.dual_values_scaled.set(x.lp_is_scaled);
    x.dual_values_basis_hash.set(*e.basis_hash);
    x.dual_values_cost_hash.set(cost_hash(e));
}

/// HEkk::computeDual with scratch vectors
pub fn compute_dual(e: &mut EkkView, x: &CHekk) {
    x.dual_values_valid.set(false);
    let mut col = OwnedHVec::new(e.num_row as i32);
    let mut row = OwnedHVec::new(e.num_col as i32);
    col.with(|c| row.with(|r| e.compute_dual(c, r)));
}

/// HEkk::computePrimal with a scratch vector
pub fn compute_primal(e: &mut EkkView) {
    let mut col = OwnedHVec::new(e.num_row as i32);
    col.with(|c| e.compute_primal(c));
}

/// HEkk::computeSimplexInfeasible
pub fn compute_simplex_infeasible(e: &mut EkkView) {
    e.compute_simplex_primal_infeasible();
    e.compute_simplex_dual_infeasible();
}

/// HEkk::computeDualObjectiveValue
pub fn compute_dual_objective_value(e: &mut EkkView, phase: i32) {
    e.compute_dual_objective_value(phase);
    e.status.has_dual_objective_value = true;
}

/// HEkk::computePrimalObjectiveValue
pub fn compute_primal_objective_value(e: &mut EkkView) {
    e.compute_primal_objective_value();
    e.status.has_primal_objective_value = true;
}

/// HEkk::initialiseCost, with its report of a cost perturbation
pub fn initialise_cost(e: &mut EkkView, x: &CHekk, algorithm: i32, perturb: bool) {
    let report_cost_perturbation = x.output_flag;
    let perturbing = algorithm != ALGORITHM_PRIMAL && perturb && *e.dual_simplex_cost_perturbation_multiplier != 0.0;
    if perturbing && report_cost_perturbation {
        x.dev(LOG_INFO, || sprintf!("Cost perturbation for %s\n", &x.model_name()));
    }
    let mut r = CostPerturbationReport::default();
    e.initialise_cost(algorithm, perturb, &mut r);
    if !r.perturbed || !report_cost_perturbation {
        return;
    }
    x.dev(LOG_INFO, || {
        sprintf!("   Initially have %d nonzero costs (%3d%%)", r.num_original_nonzero_cost, r.pct0)
    });
    if r.num_original_nonzero_cost != 0 {
        x.dev(LOG_INFO, || {
            sprintf!(
                " with min / average / max = %g / %g / %g\n",
                r.min_abs_cost,
                r.average_abs_cost,
                r.max_abs_cost
            )
        });
    } else {
        x.dev(LOG_INFO, || sprintf!(" but perturb as if max cost was 1\n"));
    }
    if r.large {
        x.dev(LOG_INFO, || {
            sprintf!("   Large so set max_abs_cost = sqrt(sqrt(max_abs_cost)) = %g\n", r.large_max_abs_cost)
        });
    }
    if r.small_boxed_rate {
        x.dev(LOG_INFO, || {
            sprintf!(
                "   Small boxedRate (%g) so set max_abs_cost = min(max_abs_cost, 1.0) = %g\n",
                r.boxed_rate,
                r.small_boxed_max_abs_cost
            )
        });
    }
    x.dev(LOG_INFO, || sprintf!("   Perturbation column base = %g\n", *e.cost_perturbation_base));
    x.dev(LOG_INFO, || sprintf!("   Perturbation row    base = %g\n", r.row_cost_perturbation_base));
}

/// HEkk::isUnconstrainedLp
pub fn is_unconstrained_lp(e: &EkkView, x: &CHekk) -> bool {
    let is_unconstrained_lp = e.num_row == 0;
    if is_unconstrained_lp {
        x.dev(LOG_ERROR, || {
            sprintf!("HEkkDual::solve called for LP with non-positive (%d) number of constraints\n", e.num_row)
        });
    }
    is_unconstrained_lp
}

/// HEkk::resetSyntheticClock
pub fn reset_synthetic_clock(e: &mut EkkView, x: &CHekk) {
    *e.build_synthetic_tick = x.nla_build_synthetic_tick.get();
    *e.total_synthetic_tick = 0.0;
}

/// HEkk::computeFactor: INVERT, returning the rank deficiency
pub fn compute_factor(e: &mut EkkView, x: &CHekk) -> i32 {
    if e.status.has_fresh_invert {
        return 0;
    }
    // Clear any bad basis changes
    x.records().clear_bad_basis_change(REASON_ALL);
    // Perform INVERT: HSimplexNla::invert and HFactor::build
    let rank_deficiency = e.factor.build_with_refactor_info(
        x.hfactor_pivot_threshold.get(),
        x.hfactor_pivot_tolerance,
        x.hfactor_time_limit,
        &e.factor_a,
        e.basic_index,
    );
    if rank_deficiency != BUILD_KERNEL_RETURN_TIMEOUT
        && e.factor.rank_deficiency != 0
        && e.factor.num_basic == e.factor.num_row
    {
        let rd = e.factor.rank_deficiency;
        x.factor_dev(LOG_WARNING, || sprintf!("Rank deficiency of %d identified in basis matrix\n", rd));
    }
    x.nla_build_synthetic_tick.set(e.factor.build_synthetic_tick);
    // Set up hot start information
    let f = &e.factor;
    x.records().out.hot_start = Some(HotStart {
        refactor_use: f.refactor_use,
        pivot_row: f.refactor_pivot_row.clone(),
        pivot_var: f.refactor_pivot_var.clone(),
        pivot_type: f.refactor_pivot_type.clone(),
        build_synthetic_tick: f.refactor_build_synthetic_tick,
        nonbasic_move: e.nonbasic_move.to_vec(),
    });
    if rank_deficiency != 0 {
        // Have an invertible representation, but of B with column(s)
        // replacements due to singularity. So no (fresh) representation of
        // B^{-1}
        e.status.has_invert = false;
        e.status.has_fresh_invert = false;
    } else {
        // Now have a representation of B^{-1}, and it is fresh!
        e.status.has_invert = true;
        e.status.has_fresh_invert = true;
    }
    // Set the update count to zero since the corrected invertible
    // representation may be used for an initial basis
    *e.update_count = 0;
    x.num_invert.set(x.num_invert.get() + 1);
    rank_deficiency
}

/// HEkk::getBacktrackingBasis
fn get_backtracking_basis(e: &mut EkkView, x: &CHekk) -> bool {
    x.dual_values_valid.set(false);
    if !x.valid_backtracking_basis.get() {
        return false;
    }
    // SAFETY: info_.backtracking_basis_ and its work shifts and weights,
    // sized by C++ as the basis, the work arrays and the scattered weights
    unsafe {
        e.basic_index.copy_from_slice(x.bt_basic_index.get());
        e.nonbasic_flag.copy_from_slice(x.bt_nonbasic_flag.get());
        e.nonbasic_move.copy_from_slice(x.bt_nonbasic_move.get());
        *e.basis_hash = x.bt_hash.get();
        x.basis_debug_id.set(x.bt_debug_id.get());
        x.basis_debug_update_count.set(x.bt_debug_update_count.get());
        *e.costs_shifted = x.bt_costs_shifted.get() != 0;
        *e.costs_perturbed = x.bt_costs_perturbed.get() != 0;
        *e.bounds_shifted = x.bt_bounds_shifted.get() != 0;
        *e.bounds_perturbed = x.bt_bounds_perturbed.get() != 0;
        e.work_shift.copy_from_slice(x.bt_work_shift.get());
        let num_tot = e.num_tot();
        e.scattered_dual_edge_weight[..num_tot].copy_from_slice(&x.bt_edge_weight.get()[..num_tot]);
    }
    true
}

/// HEkk::putBacktrackingBasis(basicIndex_before_compute_factor)
fn put_backtracking_basis_with(e: &mut EkkView, x: &CHekk, basic_index: &[i32]) {
    x.valid_backtracking_basis.set(true);
    // SAFETY: as for get_backtracking_basis
    unsafe {
        x.bt_basic_index.get_mut().copy_from_slice(basic_index);
        x.bt_nonbasic_flag.get_mut().copy_from_slice(e.nonbasic_flag);
        x.bt_nonbasic_move.get_mut().copy_from_slice(e.nonbasic_move);
        x.bt_hash.set(*e.basis_hash);
        x.bt_debug_id.set(x.basis_debug_id.get());
        x.bt_debug_update_count.set(x.basis_debug_update_count.get());
        x.bt_costs_shifted.set(*e.costs_shifted as i32);
        x.bt_costs_perturbed.set(*e.costs_perturbed as i32);
        x.bt_bounds_shifted.set(*e.bounds_shifted as i32);
        x.bt_bounds_perturbed.set(*e.bounds_perturbed as i32);
        x.bt_work_shift.get_mut().copy_from_slice(e.work_shift);
        let num_tot = e.num_tot();
        x.bt_edge_weight.get_mut()[..num_tot].copy_from_slice(&e.scattered_dual_edge_weight[..num_tot]);
    }
}

/// HEkk::putBacktrackingBasis()
pub fn put_backtracking_basis(e: &mut EkkView, x: &CHekk) {
    for i in 0..e.num_row {
        e.scattered_dual_edge_weight[e.basic_index[i] as usize] = e.dual_edge_weight[i];
    }
    let basic_index = e.basic_index.to_vec();
    put_backtracking_basis_with(e, x, &basic_index);
}

/// HEkk::getNonsingularInverse
pub fn get_nonsingular_inverse(e: &mut EkkView, x: &CHekk, _solve_phase: i32) -> bool {
    let num_row = e.num_row;
    // Take a copy of basicIndex from before INVERT to be used as the saved
    // ordering of basic variables - so reinvert will run identically
    let basic_index_before_compute_factor = e.basic_index.to_vec();
    // Save the number of updates performed in case it has to be used to
    // determine a limit
    let simplex_update_count = *e.update_count;
    // Dual simplex edge weights are identified with rows, so must be
    // permuted according to INVERT
    for i in 0..num_row {
        e.scattered_dual_edge_weight[e.basic_index[i] as usize] = e.dual_edge_weight[i];
    }
    // Call computeFactor to perform INVERT
    let rank_deficiency = compute_factor(e, x);
    if rank_deficiency != 0 {
        x.dev(LOG_INFO, || {
            sprintf!(
                "HEkk::getNonsingularInverse Rank_deficiency: solve %d (Iteration %d)\n",
                x.debug_solve_call_num,
                *e.iteration_count
            )
        });
        // Rank deficient basis, so backtrack to last full rank basis
        let deficient_hash = *e.basis_hash;
        if !get_backtracking_basis(e, x) {
            return false;
        }
        // Record that backtracking is taking place
        *e.backtracking = true;
        let records = x.records();
        records.visited.clear();
        records.visited.insert(*e.basis_hash);
        records.visited.insert(deficient_hash);
        // HEkk::updateStatus(LpAction::kBacktracking)
        x.dual_values_valid.set(false);
        e.status.has_ar_matrix = false;
        e.status.has_fresh_rebuild = false;
        e.status.has_dual_objective_value = false;
        e.status.has_primal_objective_value = false;
        let backtrack_rank_deficiency = compute_factor(e, x);
        // This basis has previously been inverted successfully, so it
        // shouldn't be singular
        if backtrack_rank_deficiency != 0 {
            return false;
        }
        // simplex update limit will be half of the number of updates
        // performed, so make sure that at least one update was performed
        if simplex_update_count <= 1 {
            return false;
        }
        let use_simplex_update_limit = *e.update_limit;
        let new_simplex_update_limit = simplex_update_count / 2;
        *e.update_limit = new_simplex_update_limit;
        x.dev(LOG_WARNING, || {
            sprintf!(
                "Rank deficiency of %d after %d simplex updates, so backtracking: max updates reduced from %d to %d\n",
                rank_deficiency,
                simplex_update_count,
                use_simplex_update_limit,
                new_simplex_update_limit
            )
        });
    } else {
        // Current basis is full rank so save it
        put_backtracking_basis_with(e, x, &basic_index_before_compute_factor);
        // Indicate that backtracking is not taking place
        *e.backtracking = false;
        // Reset the update limit in case this is the first successful
        // inversion after backtracking
        *e.update_limit = x.simplex_update_limit;
    }
    // Gather the edge weights according to the permutation of basicIndex
    // after INVERT
    for i in 0..num_row {
        e.dual_edge_weight[i] = e.scattered_dual_edge_weight[e.basic_index[i] as usize];
    }
    true
}

/// HighsSparseMatrix::createRowwisePartitioned of lp_.a_matrix_ into the
/// (sized) ar_matrix_
fn create_rowwise_partitioned(e: &mut EkkView) {
    let (num_col, num_row) = (e.num_col, e.num_row);
    let a_start = e.a.start;
    let a_index = e.a.index;
    let a_value = e.a.value;
    let in_partition = &e.nonbasic_flag[..];
    let ar_start = &mut *e.ar_start;
    let ar_p_end = &mut *e.ar_p_end;
    let mut ar_end = vec![0i32; num_row];
    ar_p_end[..num_row].fill(0);
    // Count the nonzeros of nonbasic and basic columns in each row
    for i_col in 0..num_col {
        let rows = &a_index[a_start[i_col] as usize..a_start[i_col + 1] as usize];
        if in_partition[i_col] != 0 {
            for &i_row in rows {
                ar_p_end[i_row as usize] += 1;
            }
        } else {
            for &i_row in rows {
                ar_end[i_row as usize] += 1;
            }
        }
    }
    // Compute the starts and turn the lengths into ends
    ar_start[0] = 0;
    for i_row in 0..num_row {
        ar_start[i_row + 1] = ar_start[i_row] + ar_p_end[i_row] + ar_end[i_row];
    }
    for i_row in 0..num_row {
        ar_end[i_row] = ar_start[i_row] + ar_p_end[i_row];
        ar_p_end[i_row] = ar_start[i_row];
    }
    // Insert the entries
    for i_col in 0..num_col {
        let (from, to) = (a_start[i_col] as usize, a_start[i_col + 1] as usize);
        let ends = if in_partition[i_col] != 0 { &mut *ar_p_end } else { &mut ar_end[..] };
        for i_el in from..to {
            let i_row = a_index[i_el] as usize;
            let i_to_el = ends[i_row] as usize;
            ends[i_row] += 1;
            e.ar_index[i_to_el] = i_col as i32;
            e.ar_value[i_to_el] = a_value[i_el];
        }
    }
}

/// HEkk::initialisePartitionedRowwiseMatrix
pub fn initialise_partitioned_rowwise_matrix(e: &mut EkkView, x: &CHekk) {
    if e.status.has_ar_matrix && x.ar_matrix_is_scaled.get() == x.lp_is_scaled {
        return;
    }
    create_rowwise_partitioned(e);
    x.ar_matrix_is_scaled.set(x.lp_is_scaled);
    e.status.has_ar_matrix = true;
}

thread_local! {
    /// factor_solve_error's dense work arrays, kept zero between calls:
    /// it runs at every rebuild, and allocating them anew (num_row and
    /// num_tot entries) cost ~1% of the dispatch MIPs
    static SOLVE_ERROR_WORK: std::cell::RefCell<(Vec<bool>, Vec<f64>)> =
        const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

/// HEkk::factorSolveError: a cheap assessment of factor accuracy, from a
/// random solution with at most 50 nonzeros
pub fn factor_solve_error(e: &mut EkkView) -> f64 {
    let (mut solution_nonzero, mut btran_scattered_rhs) =
        SOLVE_ERROR_WORK.with(|w| std::mem::take(&mut *w.borrow_mut()));
    let error = factor_solve_error_with(e, &mut solution_nonzero, &mut btran_scattered_rhs);
    SOLVE_ERROR_WORK.with(|w| *w.borrow_mut() = (solution_nonzero, btran_scattered_rhs));
    error
}

/// factor_solve_error with zero work arrays (left zero on return)
fn factor_solve_error_with(e: &mut EkkView, solution_nonzero: &mut Vec<bool>, btran_scattered_rhs: &mut Vec<f64>) -> f64 {
    let num_col = e.num_col;
    let num_row = e.num_row;
    let mut btran_rhs = OwnedHVec::new(num_row as i32);
    let mut ftran_rhs = OwnedHVec::new(num_row as i32);
    // Solve for a random solution
    let mut random = HighsRandom::new(1);
    let ideal_solution_num_nz = 50;
    let solution_num_nz = ideal_solution_num_nz.min((num_row + 1) / 2);
    let mut solution_value: Vec<f64> = Vec::with_capacity(solution_num_nz);
    let mut solution_index: Vec<usize> = Vec::with_capacity(solution_num_nz);
    solution_nonzero.resize(solution_nonzero.len().max(num_row), false);
    loop {
        let i_row = random.integer_below(num_row as i32) as usize;
        if solution_nonzero[i_row] {
            continue;
        }
        let value = random.fraction();
        solution_value.push(value);
        solution_index.push(i_row);
        solution_nonzero[i_row] = true;
        let i_col = e.basic_index[i_row] as usize;
        let a = &e.a;
        ftran_rhs.with(|v| a.collect_aj(v, i_col, value));
        if solution_value.len() == solution_num_nz {
            break;
        }
    }
    btran_scattered_rhs.resize(btran_scattered_rhs.len().max(num_col + num_row), 0.0);
    for (&i_row, &value) in solution_index.iter().zip(&solution_value) {
        for i_el in e.ar_p_end[i_row] as usize..e.ar_start[i_row + 1] as usize {
            let i_col = e.ar_index[i_el] as usize;
            btran_scattered_rhs[i_col] = e.ar_value[i_el].mul_add_c(value, btran_scattered_rhs[i_col]);
        }
        let i_col = num_col + i_row;
        if e.nonbasic_flag[i_col] == 0 {
            btran_scattered_rhs[i_col] = value;
        }
    }
    for i_row in 0..num_row {
        let i_col = e.basic_index[i_row] as usize;
        if btran_scattered_rhs[i_col] == 0.0 {
            continue;
        }
        btran_rhs.array[i_row] = btran_scattered_rhs[i_col];
        btran_rhs.index[btran_rhs.count as usize] = i_row as i32;
        btran_rhs.count += 1;
    }
    // Leave the work arrays zero: only the entries of the solution's rows
    // were set
    for &i_row in &solution_index {
        solution_nonzero[i_row] = false;
        for i_el in e.ar_p_end[i_row] as usize..e.ar_start[i_row + 1] as usize {
            btran_scattered_rhs[e.ar_index[i_el] as usize] = 0.0;
        }
        btran_scattered_rhs[num_col + i_row] = 0.0;
    }
    let expected_density = solution_num_nz as f64 * *e.col_aq_density;
    ftran_rhs.with(|v| e.ftran(v, expected_density));
    btran_rhs.with(|v| e.btran(v, expected_density));
    let mut ftran_solution_error = 0.0;
    for (&i_row, &value) in solution_index.iter().zip(&solution_value) {
        ftran_solution_error = std_max((ftran_rhs.array[i_row] - value).abs(), ftran_solution_error);
    }
    let mut btran_solution_error = 0.0;
    for (&i_row, &value) in solution_index.iter().zip(&solution_value) {
        btran_solution_error = std_max((btran_rhs.array[i_row] - value).abs(), btran_solution_error);
    }
    std_max(ftran_solution_error, btran_solution_error)
}

/// HEkk::rebuildRefactor
pub fn rebuild_refactor(e: &mut EkkView, x: &CHekk, rebuild_reason: i32) -> bool {
    // If no updates have been performed, then don't refactor!
    if *e.update_count == 0 {
        return false;
    }
    // Otherwise, refactor by default
    let mut refactor = true;
    if x.no_unnecessary_rebuild_refactor
        && matches!(
            rebuild_reason,
            RR_NO
                | RR_POSSIBLY_OPTIMAL
                | RR_POSSIBLY_PHASE1_FEASIBLE
                | RR_POSSIBLY_PRIMAL_UNBOUNDED
                | RR_POSSIBLY_DUAL_UNBOUNDED
                | RR_PRIMAL_INFEASIBLE_IN_PRIMAL_SIMPLEX
        )
    {
        // By default, don't refactor!
        refactor = false;
        // Possibly revise the decision based on accuracy when solving a
        // test system
        let error_tolerance = x.rebuild_refactor_solution_error_tolerance;
        if error_tolerance > 0.0 {
            let solution_error = factor_solve_error(e);
            refactor = solution_error > error_tolerance;
        }
    }
    refactor
}

/// The vectors for dual edge weights when the basis has none: unit
/// weights (dual_edge_weight_.assign(num_row, 1.0); the scattered weights
/// and the backtracking copy are sized by C++)
pub fn assign_unit_dual_edge_weights(e: &mut EkkView) {
    let num_row = e.num_row;
    e.dual_edge_weight[..num_row].fill(1.0);
}

/// HEkk::restoreDualEdgeWeights: set the DSE weights from the saved ones
/// if the basis is the saved one up to added/deleted logicals
pub fn restore_dual_edge_weights(e: &mut EkkView, x: &CHekk, near_optimal: bool) -> bool {
    let saved = if x.saved_dual_edge_weight_taken.get() {
        &[][..]
    } else {
        x.saved_dual_edge_weight_taken.set(true);
        // SAFETY: saved_dual_edge_weight_, unchanged during the solve
        unsafe { x.saved_dual_edge_weight.get().get() }
    };
    let num_row = e.num_row;
    // For small LPs computing the weights afresh is cheap
    if num_row as i32 <= x.simplex_dse_exact_init_max_rows {
        return false;
    }
    if saved.len() != e.num_col + num_row {
        return false;
    }
    let mut num_new = 0;
    for i_row in 0..num_row {
        let weight = saved[e.basic_index[i_row] as usize];
        if weight == -1.0 {
            return false;
        }
        num_new += (weight < 0.0) as i32;
    }
    // Near-optimal solves otherwise use Devex rather than computing all
    // weights, so only pay for a few
    if near_optimal && num_new as f64 > 0.1 * num_row as f64 {
        return false;
    }
    let mut row_ep = OwnedHVec::new(num_row as i32);
    for i_row in 0..num_row {
        let weight = saved[e.basic_index[i_row] as usize];
        e.dual_edge_weight[i_row] =
            if weight >= 0.0 { weight } else { row_ep.with(|v| e.compute_dual_steepest_edge_weight(i_row, v)) };
    }
    true
}

/// The pivot threshold raised on numerical trouble: info_ and the factor's
pub fn set_pivot_threshold(x: &CHekk, new_pivot_threshold: f64) {
    x.user(LOG_WARNING, &sprintf!("   Increasing Markowitz threshold to %g\n", new_pivot_threshold));
    x.factor_pivot_threshold.set(new_pivot_threshold);
    // HFactor::setPivotThreshold
    const K_MIN_PIVOT_THRESHOLD: f64 = 8e-4;
    const K_MAX_PIVOT_THRESHOLD: f64 = 0.5;
    if (K_MIN_PIVOT_THRESHOLD..=K_MAX_PIVOT_THRESHOLD).contains(&new_pivot_threshold) {
        x.hfactor_pivot_threshold.set(new_pivot_threshold);
    }
}

/// HEkk::isBadBasisChange
pub fn is_bad_basis_change(e: &EkkView, x: &CHekk, variable_in: i32, row_out: i32, rebuild_reason: i32) -> bool {
    if rebuild_reason != 0 {
        return false;
    }
    if variable_in == -1 || row_out == -1 {
        return false;
    }
    let mut currhash = *e.basis_hash;
    let variable_out = e.basic_index[row_out as usize];
    hash::sparse_inverse_combine_index(&mut currhash, variable_out);
    hash::sparse_combine_index(&mut currhash, variable_in);
    let mut cycling_detected = false;
    let records = x.records();
    if records.visited.contains(&currhash) {
        if x.iteration_count.get() == x.previous_iteration_cycling_detected.get().wrapping_add(1) {
            // Cycling detected on successive iterations suggests infinite
            // cycling
            cycling_detected = true;
        } else {
            x.previous_iteration_cycling_detected.set(x.iteration_count.get());
        }
    }
    if cycling_detected {
        x.dev(LOG_WARNING, || sprintf!(" basis change (%d out; %d in) is bad\n", variable_out, variable_in));
        records.add_bad_basis_change(row_out, variable_out, variable_in, super::basis_records::REASON_CYCLING, true);
        true
    } else {
        // Look to see whether this basis change is in the list of bad ones
        records.find_and_make_taboo(row_out, variable_out, variable_in)
    }
}

/// HEkk::getMaxAbsRowValue
fn get_max_abs_row_value(e: &mut EkkView, x: &CHekk, row: usize) -> f64 {
    if !e.status.has_ar_matrix {
        initialise_partitioned_rowwise_matrix(e, x);
    }
    let mut val = -1.0;
    let range = e.ar_start[row] as usize..e.ar_start[row + 1] as usize;
    if x.ar_matrix_is_scaled.get() == x.lp_is_scaled {
        for i in range {
            val = std_max(val, e.ar_value[i].abs());
        }
        return val;
    }
    // the row-wise matrix is in the other scaling (see moveLp): a scaled
    // value is the unscaled one times its column and row scale factors
    // SAFETY: lp_.scale_
    let (col_scale, row_scale) = unsafe { (x.lp_col_scale.get(), x.lp_row_scale.get()) };
    let row_scale = row_scale[row];
    for i in range {
        let factor = col_scale[e.ar_index[i] as usize] * row_scale;
        val = std_max(val, e.ar_value[i].abs() * if x.lp_is_scaled { factor } else { 1.0 / factor });
    }
    val
}

/// HighsSparseMatrix::productTransposeQuad(result_value, result_index,
/// column) of the column-wise lp_.a_matrix_
fn product_transpose_quad_colwise(e: &EkkView, column: &HVec, value: &mut Vec<f64>, index: &mut Vec<i32>) {
    let a = &e.a;
    for i_col in 0..e.num_col {
        let mut sum = CDouble::from(0.0);
        for i_el in a.start[i_col] as usize..a.start[i_col + 1] as usize {
            sum += column.array[a.index[i_el] as usize] * a.value[i_el];
        }
        if sum.abs() - K_HIGHS_TINY > 0.0 {
            value.push(sum.to_f64());
            index.push(i_col as i32);
        }
    }
}

/// HighsSparseMatrix::productTransposeQuad(result_value, result_index,
/// column) of the row-wise ar_matrix_ (all its entries)
fn product_transpose_quad_rowwise(e: &EkkView, column: &HVec, value: &mut Vec<f64>, index: &mut Vec<i32>) {
    let mut sum = HighsSparseVectorSum::new(e.num_col);
    for i_row in 0..e.num_row {
        let multiplier = column.array[i_row];
        // rows with a zero multiplier add nothing (that cleanup keeps)
        if multiplier == 0.0 {
            continue;
        }
        for i_el in e.ar_start[i_row] as usize..e.ar_start[i_row + 1] as usize {
            sum.add(e.ar_index[i_el], multiplier * e.ar_value[i_el]);
        }
    }
    sum.cleanup(|_, x| x.abs() <= K_HIGHS_TINY);
    *index = std::mem::take(&mut sum.nonzeroinds);
    value.extend(index.iter().map(|&i| sum.get_value(i)));
}

/// HEkk::proofOfPrimalInfeasibility(row_ep, move_out, row_out)
pub fn proof_of_primal_infeasibility(e: &mut EkkView, x: &CHekk, row_ep: &mut HVec, move_out: i32) -> bool {
    // the row-wise matrix may be in the other scaling (see moveLp)
    let use_row_wise_matrix = e.status.has_ar_matrix && x.ar_matrix_is_scaled.get() == x.lp_is_scaled;
    // Refine row_ep by removing relatively small values
    let mut proof_lower = CDouble::from(0.0);
    for i_x in 0..row_ep.count as usize {
        let i_row = row_ep.index[i_x] as usize;
        // Give row_ep the sign of the leaving row - as is done in
        // getDualRayInterface.
        let row_ep_value = row_ep.array[i_row];
        if (row_ep_value * get_max_abs_row_value(e, x, i_row)).abs() <= x.small_matrix_value {
            row_ep.array[i_row] = 0.0;
            continue;
        }
        row_ep.array[i_row] *= move_out as f64;
        // make sure infinite sides are not used
        let row_bound;
        if row_ep.array[i_row] > 0.0 {
            row_bound = e.row_lower[i_row];
            if -row_bound >= INF {
                row_ep.array[i_row] = 0.0;
                continue;
            }
        } else {
            row_bound = e.row_upper[i_row];
            if row_bound >= INF {
                row_ep.array[i_row] = 0.0;
                continue;
            }
        }
        // add up lower bound of proof constraint
        proof_lower += row_ep.array[i_row] * row_bound;
    }
    // Form the proof constraint coefficients
    let mut proof_value: Vec<f64> = Vec::new();
    let mut proof_index: Vec<i32> = Vec::new();
    if use_row_wise_matrix {
        product_transpose_quad_rowwise(e, row_ep, &mut proof_value, &mut proof_index);
    } else if e.status.has_ar_matrix && x.lp_has_scaling {
        // The row-wise matrix is in the other scaling (see moveLp), so
        // convert its values: a scaled value is the unscaled one times its
        // column and row scale factors. Only the rows of row_ep's nonzeros
        // are visited
        let to_scaled = x.lp_is_scaled;
        // SAFETY: lp_.scale_
        let (col_scale, row_scale) = unsafe { (x.lp_col_scale.get(), x.lp_row_scale.get()) };
        let mut sum = HighsSparseVectorSum::new(e.num_col);
        for i_x in 0..row_ep.count as usize {
            let i_row = row_ep.index[i_x] as usize;
            let mut multiplier = row_ep.array[i_row];
            if multiplier == 0.0 {
                continue;
            }
            multiplier = if to_scaled { multiplier * row_scale[i_row] } else { multiplier / row_scale[i_row] };
            for i_el in e.ar_start[i_row] as usize..e.ar_start[i_row + 1] as usize {
                sum.add(e.ar_index[i_el], multiplier * e.ar_value[i_el]);
            }
        }
        for &i_col in sum.get_nonzeros() {
            let c = col_scale[i_col as usize];
            let value = sum.get_value(i_col) * if to_scaled { c } else { 1.0 / c };
            if value.abs() <= K_HIGHS_TINY {
                continue;
            }
            proof_value.push(value);
            proof_index.push(i_col);
        }
    } else {
        product_transpose_quad_colwise(e, row_ep, &mut proof_value, &mut proof_index);
    }
    let mut implied_upper = CDouble::from(0.0);
    let mut sum_inf = CDouble::from(0.0);
    for (&i_col, &value) in proof_index.iter().zip(&proof_value) {
        let i_col = i_col as usize;
        if value > 0.0 {
            if e.col_upper[i_col] >= INF {
                sum_inf += value;
                if sum_inf > x.small_matrix_value {
                    break;
                }
                continue;
            }
            implied_upper += value * e.col_upper[i_col];
        } else {
            if -e.col_lower[i_col] >= INF {
                sum_inf += -value;
                if sum_inf > x.small_matrix_value {
                    break;
                }
                continue;
            }
            implied_upper += value * e.col_lower[i_col];
        }
    }
    let infinite_implied_upper = sum_inf > x.small_matrix_value;
    let gap = (proof_lower - implied_upper).to_f64();
    let gap_ok = gap > e.primal_feasibility_tolerance;
    !infinite_implied_upper && gap_ok
}

/// HEkk::unitBtranResidual
fn unit_btran_residual(e: &EkkView, row_out: usize, row_ep: &HVec, residual: &mut HVec) -> f64 {
    let num_row = e.num_row;
    let mut quad_residual = vec![CDouble::from(0.0); num_row];
    quad_residual[row_out] = CDouble::from(-1.0);
    let a = &e.a;
    for i_row in 0..num_row {
        let i_var = e.basic_index[i_row] as usize;
        let mut value = quad_residual[i_row];
        if i_var < e.num_col {
            for i_el in a.start[i_var] as usize..a.start[i_var + 1] as usize {
                value += a.value[i_el] * row_ep.array[a.index[i_el] as usize];
            }
        } else {
            value += row_ep.array[i_var - e.num_col];
        }
        quad_residual[i_row] = value;
    }
    residual.clear();
    residual.pack_flag = false;
    let mut residual_norm = 0.0;
    for i_row in 0..num_row {
        let value = quad_residual[i_row].to_f64();
        if value != 0.0 {
            residual.array[i_row] = value;
            residual.index[residual.count as usize] = i_row as i32;
            residual.count += 1;
        }
        residual_norm = std_max(residual.array[i_row].abs(), residual_norm);
    }
    residual_norm
}

/// HEkk::unitBtranIterativeRefinement
pub fn unit_btran_iterative_refinement(e: &mut EkkView, row_out: usize, row_ep: &mut HVec) {
    let num_row = e.num_row;
    let mut residual = OwnedHVec::new(num_row as i32);
    let expected_density = 1.0;
    let residual_norm = residual.with(|r| unit_btran_residual(e, row_out, row_ep, r));
    if residual_norm == 0.0 {
        return;
    }
    // Normalise using nearest power of 2 to ||correction_rhs|| so
    // kHighsTiny isn't used adversely
    let residual_scale = nearest_power_of_two_scale(residual_norm);
    for i_el in 0..residual.count as usize {
        residual.array[residual.index[i_el] as usize] *= residual_scale;
    }
    residual.with(|r| e.btran(r, expected_density));
    row_ep.count = 0;
    // Adding two (possibly sparse) vectors, so have to loop over all rows
    for i_row in 0..num_row {
        if residual.array[i_row] != 0.0 {
            let correction_value = residual.array[i_row] / residual_scale;
            row_ep.array[i_row] -= correction_value;
        }
        if row_ep.array[i_row].abs() < K_HIGHS_TINY {
            row_ep.array[i_row] = 0.0;
        } else {
            row_ep.index[row_ep.count as usize] = i_row as i32;
            row_ep.count += 1;
        }
    }
}

/// HighsSparseMatrix::priceByRowWithSwitch with quad_precision of the
/// row-wise (partitioned) ar_matrix_
fn price_by_row_with_switch_quad(
    e: &EkkView,
    result: &mut HVec,
    column: &HVec,
    expected_density: f64,
    from_index: usize,
    switch_density: f64,
) {
    let num_col = e.num_col;
    let mut sum = HighsSparseVectorSum::new(num_col);
    let mut next_index = from_index;
    if expected_density <= K_HYPER_PRICE_DENSITY {
        let inv_num_col = 1.0 / num_col as f64;
        for ix in next_index..column.count as usize {
            let i_row = column.index[ix] as usize;
            let to_i_el = e.ar_p_end[i_row] as usize;
            // Possibly switch to standard row-wise price
            let row_num_nz = to_i_el as i32 - e.ar_start[i_row];
            let local_density = result.count as f64 * inv_num_col;
            let switch_to_dense = result.count + row_num_nz >= num_col as i32 || local_density > switch_density;
            if switch_to_dense {
                break;
            }
            let multiplier = column.array[i_row];
            if multiplier != 0.0 {
                for i_el in e.ar_start[i_row] as usize..to_i_el {
                    sum.add(e.ar_index[i_el], multiplier * e.ar_value[i_el]);
                }
            }
            next_index = ix + 1;
        }
    }
    sum.cleanup(|_, x| x.abs() <= K_HIGHS_TINY);
    if next_index < column.count as usize {
        // PRICE is not complete: finish without maintaining nonzeros of
        // result (priceByRowDenseResult)
        let mut result_array = sum.values.clone();
        for ix in next_index..column.count as usize {
            let i_row = column.index[ix] as usize;
            let multiplier = column.array[i_row];
            for i_el in e.ar_start[i_row] as usize..e.ar_p_end[i_row] as usize {
                let i_col = e.ar_index[i_el] as usize;
                let value1 = result_array[i_col] + multiplier * e.ar_value[i_el];
                result_array[i_col] = if value1.to_f64().abs() < K_HIGHS_TINY { CDouble::from(K_HIGHS_ZERO) } else { value1 };
            }
        }
        // Determine indices of nonzeros in result
        result.count = 0;
        for i_col in 0..num_col {
            let value1 = result_array[i_col].to_f64();
            if value1.abs() < K_HIGHS_TINY {
                result.array[i_col] = 0.0;
            } else {
                result.array[i_col] = value1;
                result.index[result.count as usize] = i_col as i32;
                result.count += 1;
            }
        }
    } else {
        let num_nz = sum.nonzeroinds.len();
        result.index[..num_nz].copy_from_slice(&sum.nonzeroinds);
        result.count = num_nz as i32;
        for i in 0..num_nz {
            let i_row = result.index[i];
            result.array[i_row as usize] = sum.get_value(i_row);
        }
    }
}

/// HEkk::tableauRowPrice with quad_precision
pub fn tableau_row_price_quad(e: &mut EkkView, row_ep: &HVec, row_ap: &mut HVec) {
    let num_row = e.num_row;
    let num_col = e.num_col;
    let local_density = row_ep.count as f64 / num_row as f64;
    let (use_col_price, use_row_price_w_switch) = choose_price_technique(e.price_strategy, local_density);
    row_ap.clear();
    if use_col_price {
        // HighsSparseMatrix::priceByColumn with quad_precision
        let a = &e.a;
        row_ap.count = 0;
        for i_col in 0..num_col {
            let mut quad_value = CDouble::from(0.0);
            for i_el in a.start[i_col] as usize..a.start[i_col + 1] as usize {
                quad_value += row_ep.array[a.index[i_el] as usize] * a.value[i_el];
            }
            let value = quad_value.to_f64();
            let nonzero = value.abs() > K_HIGHS_TINY;
            row_ap.array[i_col] = if nonzero { value } else { 0.0 };
            row_ap.index[row_ap.count as usize] = i_col as i32;
            row_ap.count += nonzero as i32;
        }
    } else if use_row_price_w_switch {
        // Perform hyper-sparse row-wise PRICE, but switch if the density
        // of row_ap becomes extreme
        price_by_row_with_switch_quad(e, row_ap, row_ep, *e.row_ap_density, 0, K_HYPER_PRICE_DENSITY);
    } else {
        // Perform hyper-sparse row-wise PRICE
        price_by_row_with_switch_quad(e, row_ap, row_ep, -INF, 0, INF);
    }
    if use_col_price {
        // Column-wise PRICE computes components corresponding to basic
        // variables, so zero these
        for i_col in 0..num_col {
            row_ap.array[i_col] *= e.nonbasic_flag[i_col] as f64;
        }
    }
    // Update the record of average row_ap density
    let local_row_ap_density = row_ap.count as f64 / num_col as f64;
    update_operation_result_density(local_row_ap_density, e.row_ap_density);
}

/// HEkk::bailout, with a run clock read from C++ once and then advanced
/// by a Rust monotonic clock
pub struct Bailout {
    clock: Option<(f64, std::time::Instant)>,
}

impl Bailout {
    pub fn new() -> Self {
        Bailout { clock: None }
    }

    fn timer_read(&mut self, x: &CHekk) -> f64 {
        match self.clock {
            Some((time0, instant0)) => time0 + instant0.elapsed().as_secs_f64(),
            None => {
                let time0 = (x.host.timer_read)(x.host.ctx);
                self.clock = Some((time0, std::time::Instant::now()));
                time0
            }
        }
    }

    /// HEkk::bailout
    pub fn check(&mut self, x: &CHekk) -> bool {
        if x.solve_bailout.get() {
        } else if x.time_limit < INF && self.timer_read(x) > x.time_limit {
            x.solve_bailout.set(true);
            x.model_status.set(MS_TIME_LIMIT);
        } else if x.iteration_count.get() >= x.simplex_iteration_limit {
            x.solve_bailout.set(true);
            x.model_status.set(MS_ITERATION_LIMIT);
        } else if x.interrupt_callback && (x.host.interrupt)(x.host.ctx) {
            // The user interrupts
            x.solve_bailout.set(true);
            x.model_status.set(MS_INTERRUPT);
        }
        x.solve_bailout.get()
    }
}

impl Default for Bailout {
    fn default() -> Self {
        Self::new()
    }
}

/// HEkk::invalidatePrimalInfeasibilityRecord and
/// invalidateDualInfeasibilityRecord
fn invalidate_infeasibility_records(e: &mut EkkView) {
    *e.num_primal_infeasibilities = K_ILLEGAL_INFEASIBILITY_COUNT;
    *e.max_primal_infeasibility = INF;
    *e.sum_primal_infeasibilities = INF;
    *e.num_dual_infeasibilities = K_ILLEGAL_INFEASIBILITY_COUNT;
    *e.max_dual_infeasibility = INF;
    *e.sum_dual_infeasibilities = INF;
}

/// HEkk::returnFromSolve
pub fn return_from_solve(e: &mut EkkView, x: &CHekk, return_status: i32) -> i32 {
    // Always called before returning from HEkkPrimal/Dual::solve()
    x.called_return_from_solve.set(true);
    x.valid_backtracking_basis.set(false);
    // Initialise the status of the primal and dual solutions
    x.return_primal_solution_status.set(SOLUTION_STATUS_NONE);
    x.return_dual_solution_status.set(SOLUTION_STATUS_NONE);
    // Nothing more is known about the solve after an error return
    if return_status == STATUS_ERROR {
        return return_status;
    }
    // Determine a primal and dual solution, removing the effects of
    // perturbations and shifts
    let model_status = x.model_status.get();
    if model_status != MS_OPTIMAL {
        invalidate_infeasibility_records(e);
    }
    match model_status {
        MS_OPTIMAL => {}
        MS_INFEASIBLE => {
            if x.exit_algorithm.get() == ALGORITHM_PRIMAL {
                // Reset the simplex costs and recompute duals after primal
                // phase 1
                initialise_cost(e, x, ALGORITHM_DUAL, false);
                compute_dual(e, x);
            }
            compute_simplex_infeasible(e);
        }
        MS_UNBOUNDED_OR_INFEASIBLE => {
            // Reset the simplex bounds and recompute primals
            e.initialise_bound(ALGORITHM_DUAL, PHASE_2, false);
            compute_primal(e);
            compute_simplex_infeasible(e);
        }
        MS_UNBOUNDED => compute_simplex_infeasible(e),
        MS_OBJECTIVE_BOUND | MS_OBJECTIVE_TARGET | MS_TIME_LIMIT | MS_ITERATION_LIMIT | MS_INTERRUPT | MS_UNKNOWN => {
            // Reset the simplex bounds and recompute primals
            e.initialise_bound(ALGORITHM_DUAL, PHASE_2, false);
            e.initialise_nonbasic_value_and_move();
            compute_primal(e);
            // Reset the simplex costs and recompute duals
            initialise_cost(e, x, ALGORITHM_DUAL, false);
            compute_dual(e, x);
            compute_simplex_infeasible(e);
        }
        _ => {
            let algorithm = if x.exit_algorithm.get() == ALGORITHM_PRIMAL { "primal" } else { "dual" };
            x.dev(LOG_ERROR, || {
                sprintf!("%s simplex solver returns status %s\n", algorithm, model_status_string(model_status))
            });
            return STATUS_ERROR;
        }
    }
    if model_status == MS_OPTIMAL {
        // The infeasibility records are those of the solve
    }
    e.zero_basic_duals();
    x.return_primal_solution_status.set(if *e.num_primal_infeasibilities == 0 {
        SOLUTION_STATUS_FEASIBLE
    } else {
        SOLUTION_STATUS_INFEASIBLE
    });
    x.return_dual_solution_status.set(if *e.num_dual_infeasibilities == 0 {
        SOLUTION_STATUS_FEASIBLE
    } else {
        SOLUTION_STATUS_INFEASIBLE
    });
    compute_primal_objective_value(e);
    if x.log_dev_level == 0 {
        x.user_invert_report(true);
    }
    return_status
}

/// The dual ray: HEkkDual::saveDualRay
pub fn save_dual_ray(x: &CHekk, row_out: i32, move_out: i32) {
    x.ray_value_clear.set(x.ray_value_clear.get() | 1);
    x.dual_ray_index.set(row_out);
    x.dual_ray_sign.set(move_out);
}

/// The primal ray: HEkkPrimal::savePrimalRay
pub fn save_primal_ray(x: &CHekk, index: i32, sign: i32) {
    x.ray_value_clear.set(x.ray_value_clear.get() | 2);
    x.primal_ray_index.set(index);
    x.primal_ray_sign.set(sign);
}

/// HEkk::clearRayRecords
fn clear_ray_records(x: &CHekk) {
    x.ray_value_clear.set(3);
    x.dual_ray_index.set(-1);
    x.dual_ray_sign.set(0);
    x.primal_ray_index.set(-1);
    x.primal_ray_sign.set(0);
}

/// isLessInfeasibleDSECandidate (lp_data/HighsLpUtils.cpp)
pub fn is_less_infeasible_dse_candidate(e: &EkkView, x: &CHekk) -> bool {
    let mut max_col_num_en = -1;
    let max_allowed_col_num_en = 24;
    let max_assess_col_num_en = 9.max(max_allowed_col_num_en);
    let max_average_col_num_en = 6;
    let a = &e.a;
    for col in 0..e.num_col {
        // Check limit on number of entries in the column has not been
        // breached
        let col_num_en = a.start[col + 1] - a.start[col];
        max_col_num_en = col_num_en.max(max_col_num_en);
        if col_num_en > max_assess_col_num_en {
            return false;
        }
        // All nonzeros must be +1 or -1
        for en in a.start[col] as usize..a.start[col + 1] as usize {
            if a.value[en].abs() != 1.0 {
                return false;
            }
        }
    }
    let average_col_num_en = a.start[e.num_col] as f64 / e.num_col as f64;
    let li_dse_candidate = average_col_num_en <= max_average_col_num_en as f64;
    x.dev(LOG_INFO, || {
        sprintf!(
            "LP %s has all |entries|=1; max column count = %d (limit %d); average column count = %0.2g (limit %d): LP %s a candidate for LiDSE\n",
            &x.model_name(),
            max_col_num_en,
            max_allowed_col_num_en,
            average_col_num_en,
            max_average_col_num_en,
            if li_dse_candidate { "is" } else { "is not" }
        )
    });
    li_dse_candidate
}

/// HEkkDual::possiblyUseLiDualSteepestEdge
pub fn possibly_use_li_dual_steepest_edge(e: &EkkView, x: &CHekk) {
    x.store_squared_primal_infeasibility.set(true);
    if x.less_infeasible_dse_check && is_less_infeasible_dse_candidate(e, x) && x.less_infeasible_dse_choose_row {
        // Use LiDSE
        x.store_squared_primal_infeasibility.set(false);
    }
}

// ---- HEkk::solve ----

/// HEkk::initialiseSimplexLpRandomVectors, for vectors sized by C++
fn initialise_simplex_lp_random_vectors(x: &CHekk) {
    // SAFETY: info_'s random vectors, sized by C++
    let (col_perm, tot_perm, tot_value) =
        unsafe { (x.num_col_permutation.get_mut(), x.num_tot_permutation.get_mut(), x.ekk.num_tot_random_value_mut()) };
    let num_tot = tot_perm.len();
    if num_tot == 0 {
        return;
    }
    let mut random = HighsRandom::from_state(x.random.get());
    if !col_perm.is_empty() {
        // Generate a random permutation of the column indices
        for (i, p) in col_perm.iter_mut().enumerate() {
            *p = i as i32;
        }
        random.shuffle(col_perm);
    }
    // Generate a random permutation of all the indices
    for (i, p) in tot_perm.iter_mut().enumerate() {
        *p = i as i32;
    }
    random.shuffle(tot_perm);
    // Generate a vector of random reals
    for v in tot_value.iter_mut() {
        *v = random.fraction();
    }
    x.random.set(random.state());
}

/// HEkk::initialiseSimplexLpBasisAndFactor (after the C++ set-up of the
/// basis and the simplex NLA)
fn initialise_simplex_lp_basis_and_factor(e: &mut EkkView, x: &CHekk) {
    if !e.status.has_invert {
        let rank_deficiency = compute_factor(e, x);
        if rank_deficiency != 0 {
            // Basis is rank deficient: account for it by correcting
            // nonbasicFlag
            initial_rank_deficiency(e, x);
        }
        // Record the synthetic clock for INVERT, and zero it for UPDATE
        reset_synthetic_clock(e, x);
    }
}

/// The handling of a rank deficient initial basis in
/// HEkk::initialiseSimplexLpBasisAndFactor: handleRankDeficiency, the
/// update of the status for a new basis and setNonbasicMove
pub fn initial_rank_deficiency(e: &mut EkkView, x: &CHekk) {
    let rank_deficiency = e.factor.rank_deficiency;
    x.dev(LOG_INFO, || {
        // SAFETY: basis_.debug_origin_name, unchanged during the call
        let origin = unsafe { &*x.basis_origin };
        sprintf!(
            "HEkk::initialiseSimplexLpBasisAndFactor (%s) Rank_deficiency %d: Id = %d; UpdateCount = %d\n",
            origin,
            rank_deficiency,
            x.basis_debug_id.get(),
            x.basis_debug_update_count.get()
        )
    });
    // HEkk::handleRankDeficiency
    let num_col = e.num_col as i32;
    for k in 0..rank_deficiency.max(0) as usize {
        let row_in = e.factor.row_with_no_pivot[k];
        let variable_in = num_col + row_in;
        let variable_out = e.factor.var_with_no_pivot[k];
        e.nonbasic_flag[variable_in as usize] = 0;
        e.nonbasic_flag[variable_out as usize] = 1;
        let row_out = row_in;
        x.dev(LOG_INFO, || {
            sprintf!(
                "HEkk::handleRankDeficiency: %4d: Basic row of leaving variable (%4d is %s %4d) is %4d; Entering logical = %4d is variable %d)\n",
                k as i32,
                variable_out,
                if variable_out < num_col { " column" } else { "logical" },
                if variable_out < num_col { variable_out } else { variable_out - num_col },
                row_out,
                row_in,
                variable_in
            )
        });
        // variable_in is the logical that must not come out to be replaced
        // by the structural variable_out
        x.records().add_bad_basis_change(row_out, variable_in, variable_out, super::basis_records::REASON_SINGULAR, true);
    }
    e.status.has_ar_matrix = false;
    // HEkk::updateStatus(LpAction::kNewBasis): keep the weights of the
    // outgoing basis
    x.dual_values_valid.set(false);
    let num_row = e.basic_index.len() as i32;
    let saved = super::lp_solver::scatter_dual_edge_weights(
        e.status,
        e.basic_index,
        e.nonbasic_flag.len(),
        e.dual_edge_weight,
        num_row,
        num_row,
        None,
    );
    // invalidateBasis
    e.status.has_basis = false;
    e.status.has_ar_matrix = false;
    e.status.has_dual_steepest_edge_weights = false;
    e.status.has_invert = false;
    e.status.has_fresh_invert = false;
    e.status.has_fresh_rebuild = false;
    e.status.has_dual_objective_value = false;
    e.status.has_primal_objective_value = false;
    clear_ray_records(x);
    if !saved.is_empty() {
        // SAFETY: saved_dual_edge_weight_, not otherwise borrowed now
        unsafe {
            let v = &mut *x.saved_dual_edge_weight_vec;
            *v = saved;
            x.saved_dual_edge_weight.set(CSlice { p: v.as_mut_ptr(), n: v.len() as i32 });
        }
    }
    e.set_nonbasic_move();
    e.status.has_basis = true;
    e.status.has_invert = true;
    e.status.has_fresh_invert = true;
}

/// HEkk::initialiseForSolve
fn initialise_for_solve(e: &mut EkkView, x: &CHekk) {
    initialise_simplex_lp_basis_and_factor(e, x);
    // The random vectors only depend on the LP dimensions: C++ has
    // decided whether to draw them
    if x.draw_random_vectors {
        initialise_simplex_lp_random_vectors(x);
        x.random_vectors_drawn.set(true);
    }
    initialise_partitioned_rowwise_matrix(e, x);
    initialise_cost(e, x, ALGORITHM_PRIMAL, false);
    e.initialise_bound(ALGORITHM_PRIMAL, PHASE_UNKNOWN, false);
    e.initialise_nonbasic_value_and_move();
    compute_primal(e);
    // after bound changes alone, the dual values of the last solve still
    // hold
    if !(x.dual_values_valid.get()
        && x.dual_values_scaled.get() == x.lp_is_scaled
        && x.dual_values_basis_hash.get() == *e.basis_hash
        && x.dual_values_cost_hash.get() == cost_hash(e))
    {
        compute_dual(e, x);
    }
    x.fresh_primal.set(true);
    x.fresh_unperturbed_dual.set(true);
    compute_simplex_infeasible(e);
    compute_dual_objective_value(e, PHASE_2);
    compute_primal_objective_value(e);
    e.status.initialised_for_solve = true;
    let primal_feasible = *e.num_primal_infeasibilities == 0;
    let dual_feasible = *e.num_dual_infeasibilities == 0;
    let records = x.records();
    records.visited.clear();
    records.visited.insert(*e.basis_hash);
    x.model_status.set(MS_NOTSET);
    if primal_feasible && dual_feasible {
        x.model_status.set(MS_OPTIMAL);
    }
}

/// HEkk::chooseSimplexStrategyThreads
fn choose_simplex_strategy_threads(e: &EkkView, x: &CHekk) {
    let mut simplex_strategy = x.option_simplex_strategy;
    if simplex_strategy == STRATEGY_CHOOSE {
        // HiGHS is left to choose the simplex strategy
        simplex_strategy = if *e.num_primal_infeasibilities > 0 { STRATEGY_DUAL_PLAIN } else { STRATEGY_PRIMAL };
    } else if simplex_strategy == STRATEGY_DUAL_TASKS || simplex_strategy == STRATEGY_DUAL_MULTI {
        // SIP and PAMI are not in Crestline (Highs::run warns)
        simplex_strategy = STRATEGY_DUAL_PLAIN;
    }
    x.simplex_strategy.set(simplex_strategy);
    // Set min/max_threads to correspond to serial code
    let min_concurrency = 1;
    let max_concurrency = 1;
    let simplex_min_concurrency = x.simplex_min_concurrency;
    let simplex_max_concurrency = x.simplex_max_concurrency;
    let max_threads = x.num_threads;
    x.min_concurrency.set(min_concurrency);
    x.max_concurrency.set(max_concurrency);
    // Set the concurrency to be used to be the maximum number
    let num_concurrency = max_concurrency;
    x.num_concurrency.set(num_concurrency);
    if num_concurrency < simplex_min_concurrency {
        x.user(
            LOG_WARNING,
            &sprintf!(
                "Using concurrency of %d for parallel strategy rather than minimum number (%d) specified in options\n",
                num_concurrency,
                simplex_min_concurrency
            ),
        );
    }
    if num_concurrency > simplex_max_concurrency {
        x.user(
            LOG_WARNING,
            &sprintf!(
                "Using concurrency of %d for parallel strategy rather than maximum number (%d) specified in options\n",
                num_concurrency,
                simplex_max_concurrency
            ),
        );
    }
    if num_concurrency > max_threads {
        x.user(
            LOG_WARNING,
            &sprintf!(
                "Number of threads available = %d < %d = Simplex concurrency to be used: Parallel performance may be less than anticipated\n",
                max_threads,
                num_concurrency
            ),
        );
    }
}

/// reportSimplexPhaseIterations (simplex/HSimplexReport.cpp)
fn report_simplex_phase_iterations(e: &EkkView, x: &CHekk, initialise: bool) {
    if x.run_quiet {
        return;
    }
    let iteration_count = *e.iteration_count;
    if initialise {
        x.iteration_count0.set(iteration_count);
        x.dual_phase1_iteration_count0.set(x.dual_phase1_iteration_count.get());
        x.dual_phase2_iteration_count0.set(x.dual_phase2_iteration_count.get());
        x.primal_phase1_iteration_count0.set(*e.primal_phase1_iteration_count);
        x.primal_phase2_iteration_count0.set(*e.primal_phase2_iteration_count);
        x.primal_bound_swap0.set(*e.primal_bound_swap);
        return;
    }
    let delta_iteration_count = iteration_count - x.iteration_count0.get();
    let delta_dual_phase1 = x.dual_phase1_iteration_count.get() - x.dual_phase1_iteration_count0.get();
    let delta_dual_phase2 = x.dual_phase2_iteration_count.get() - x.dual_phase2_iteration_count0.get();
    let delta_primal_phase1 = *e.primal_phase1_iteration_count - x.primal_phase1_iteration_count0.get();
    let delta_primal_phase2 = *e.primal_phase2_iteration_count - x.primal_phase2_iteration_count0.get();
    let delta_primal_bound_swap = *e.primal_bound_swap - x.primal_bound_swap0.get();
    let check_delta_iteration_count = delta_dual_phase1 + delta_dual_phase2 + delta_primal_phase1 + delta_primal_phase2;
    if check_delta_iteration_count != delta_iteration_count {
        x.user(
            LOG_ERROR,
            &sprintf!(
                "Iteration total error %d + %d + %d + %d = %d != %d\n",
                delta_dual_phase1,
                delta_dual_phase2,
                delta_primal_phase1,
                delta_primal_phase2,
                check_delta_iteration_count,
                delta_iteration_count
            ),
        );
    }
    x.dev(LOG_INFO, || {
        let mut report = String::new();
        for (name, delta) in [
            ("DuPh1", delta_dual_phase1),
            ("DuPh2", delta_dual_phase2),
            ("PrPh1", delta_primal_phase1),
            ("PrPh2", delta_primal_phase2),
            ("PrSwap", delta_primal_bound_swap),
        ] {
            if delta != 0 {
                report += &format!("{name} {delta}; ");
            }
        }
        sprintf!("Simplex iterations: %sTotal %d\n", &report, delta_iteration_count)
    });
}

/// A dual simplex solve: HEkkDual::solve
pub fn dual_solve(x: &CHekk, force_phase2: bool) -> i32 {
    // SAFETY: the views of the solvers are used one at a time
    let mut dual = unsafe { Dual::new(x) };
    dual.solve(force_phase2)
}

/// A primal simplex solve: HEkkPrimal::solve
pub fn primal_solve(x: &CHekk, force_phase2: bool) -> i32 {
    // SAFETY: the views of the solvers are used one at a time
    let mut primal = unsafe { Primal::new(x) };
    primal.solve(force_phase2)
}

/// HEkk::solve from initialiseForSolve to just before returnFromEkkSolve:
/// returns the HighsStatus
pub fn solve(x: &CHekk, force_phase2: bool) -> i32 {
    // SAFETY: C++ has sized HEkk's vectors for the solve; the view is not
    // used while a solver's is
    let mut e = unsafe { x.view() };
    x.dual_simplex_cleanup_level.set(0);
    x.dual_simplex_phase1_cleanup_level.set(0);
    x.previous_iteration_cycling_detected.set(-K_HIGHS_IINF);
    x.clear_fresh_values();
    initialise_for_solve(&mut e, x);
    if x.model_status.get() == MS_OPTIMAL {
        return STATUS_OK;
    }
    let mut return_status = STATUS_OK;
    // Indicate that dual and primal rays are not known
    clear_ray_records(x);
    // Allow primal and dual perturbations in case a block on them is
    // hanging over from a previous call
    x.allow_cost_shifting.set(true);
    x.allow_cost_perturbation.set(true);
    *e.allow_bound_perturbation = true;
    choose_simplex_strategy_threads(&e, x);
    let simplex_strategy = x.simplex_strategy.get();
    let algorithm_name;
    if simplex_strategy == STRATEGY_PRIMAL {
        algorithm_name = "primal";
        report_simplex_phase_iterations(&e, x, true);
        x.user(LOG_INFO, "Using primal simplex solver\n");
        let call_status = primal_solve(x, force_phase2);
        return_status = interpret_call_status(x, call_status, return_status, "HEkkPrimal::solve");
    } else {
        algorithm_name = "dual";
        report_simplex_phase_iterations(&e, x, true);
        x.user(LOG_INFO, "Using dual simplex solver\n");
        let call_status = dual_solve(x, force_phase2);
        return_status = interpret_call_status(x, call_status, return_status, "HEkkDual::solve");
        // Dual simplex solver may set model_status to be
        // kUnboundedOrInfeasible, and Highs::run() may not allow that to
        // be returned, so use primal simplex to distinguish
        if x.model_status.get() == MS_UNBOUNDED_OR_INFEASIBLE && !x.allow_unbounded_or_infeasible {
            let call_status = primal_solve(x, false);
            return_status = interpret_call_status(x, call_status, return_status, "HEkkPrimal::solve");
        }
    }
    report_simplex_phase_iterations(&e, x, false);
    if return_status == STATUS_ERROR {
        return return_status;
    }
    x.dev(LOG_INFO, || {
        sprintf!(
            "%s simplex solver returns %d primal and %d dual infeasibilities: Status %s\n",
            algorithm_name,
            *e.num_primal_infeasibilities,
            *e.num_dual_infeasibilities,
            model_status_string(x.model_status.get())
        )
    });
    return_status
}

/// The status flags of a view (for the solvers)
pub fn status<'a>(e: &'a mut EkkView) -> &'a mut SimplexStatus {
    e.status
}

mod ffi {
    use super::*;

    /// The hot start record of the last INVERT of a solve: copies of the
    /// refactorization information and nonbasicMove, or false if none
    ///
    /// # Safety
    /// `p` valid; the out pointers are filled with Rust-owned arrays valid
    /// until the next call on the records
    #[no_mangle]
    #[allow(clippy::too_many_arguments)]
    pub unsafe extern "C" fn highs_rs_ekk_hot_start(
        p: *mut BasisRecords,
        refactor_use: *mut bool,
        pivot_row: *mut *const i32,
        pivot_var: *mut *const i32,
        pivot_type: *mut *const i8,
        num_pivot: *mut i32,
        build_synthetic_tick: *mut f64,
        nonbasic_move: *mut *const i8,
        num_tot: *mut i32,
    ) -> bool {
        let Some(h) = &(*p).out.hot_start else { return false };
        *refactor_use = h.refactor_use;
        *pivot_row = h.pivot_row.as_ptr();
        *pivot_var = h.pivot_var.as_ptr();
        *pivot_type = h.pivot_type.as_ptr();
        *num_pivot = h.pivot_row.len().min(h.pivot_var.len()).min(h.pivot_type.len()) as i32;
        *build_synthetic_tick = h.build_synthetic_tick;
        *nonbasic_move = h.nonbasic_move.as_ptr();
        *num_tot = h.nonbasic_move.len() as i32;
        true
    }

    /// The primal phase 1 duals saved by a solve, if any
    ///
    /// # Safety
    /// As for highs_rs_ekk_hot_start
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_ekk_primal_phase1_dual(
        p: *mut BasisRecords,
        values: *mut *const f64,
        n: *mut i32,
    ) -> bool {
        let Some(d) = &(*p).out.primal_phase1_dual else { return false };
        *values = d.as_ptr();
        *n = d.len() as i32;
        true
    }

    /// Forget what a solve left for C++
    ///
    /// # Safety
    /// `p` valid
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_ekk_clear_out(p: *mut BasisRecords) {
        (*p).out = Default::default();
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn max_abs_is_the_serial_std_max() {
        let serial = |v: &[f64]| v.iter().fold(0.0, |m: f64, x| if x.abs() < m { m } else { x.abs() });
        let base = [0.5, -3.0, 2.0, -0.0, 7.5, -7.5, 1.0, 4.0, -9.0, 0.25, 3.0];
        for n in 0..=base.len() {
            for nan_at in [None, Some(0), Some(5), Some(9)] {
                let mut v = base[..n].to_vec();
                if let Some(i) = nan_at.filter(|&i| i < n) {
                    v[i] = f64::NAN;
                }
                assert_eq!(super::max_abs(&v).to_bits(), serial(&v).to_bits());
            }
        }
    }
}
