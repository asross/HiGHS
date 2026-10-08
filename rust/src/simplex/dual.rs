//! The dual simplex driver HEkkDual (highs/simplex/HEkkDual.cpp) for the
//! serial strategy (simplex_strategy 1): solve(), the phase loops, rebuild,
//! and the iteration (CHUZR, PRICE + CHUZC, the FTRANs, the updates).
//!
//! The data is HEkk's, reached through an `EkkView` (ekk.rs) and the
//! `CHekk` of hekk.rs, whose HEkk methods (the INVERT with backtracking,
//! the proof of primal infeasibility, returnFromSolve, ...) it calls
//! directly, as it does the primal simplex (primal.rs) for clean-up.
//! HEkkDual's HVectors are owned here. What still reaches C++ goes
//! through hekk.rs's `Host`: logging (when it can print), the analysis
//! reports, the run clock and a user interrupt callback.
//!
//! The analysis records that only HighsSimplexAnalysis::summaryReport
//! reads are not kept: HEkk::solve runs in Rust only when simplex
//! analysis, timing and debugging are off.

use crate::util::fma::ClangFma;

use super::basis_records::{BasisRecords, REASON_CYCLING, REASON_FAILED_INFEASIBILITY_PROOF};
use super::dual_row::{AMatrix as RowMatrix, ChooseFail, DualRow};
use super::dual_rhs::{DualRhs, Primal};
use super::ekk::{
    reinvert_on_numerical_trouble, sparse_loop_style, update_operation_result_density, EkkView, Infeasibility,
};
use super::hekk::{self, get_value_scale, Bailout, CHekk, LOG_DETAILED, LOG_ERROR, LOG_INFO, LOG_VERBOSE, LOG_WARNING};
use crate::hvector::{HVec, OwnedHVec};
use crate::matrix;
use crate::sprintf;
use crate::util::cdouble::CDouble;
use crate::util::random::HighsRandom;

const INF: f64 = f64::INFINITY;

const ALGORITHM_DUAL: i32 = 2;

// EdgeWeightMode
const EW_DANTZIG: i32 = 0;
const EW_DEVEX: i32 = 1;
const EW_DSE: i32 = 2;

// Solve phases
const PHASE_ERROR: i32 = -3;
const PHASE_EXIT: i32 = -2;
const PHASE_UNKNOWN: i32 = -1;
const PHASE_OPTIMAL: i32 = 0;
const PHASE_1: i32 = 1;
const PHASE_2: i32 = 2;
const PHASE_PRIMAL_INFEASIBLE_CLEANUP: i32 = 3;
const PHASE_OPTIMAL_CLEANUP: i32 = 4;
const PHASE_TABOO_BASIS: i32 = 5;

// Rebuild reasons
const RR_CLEANUP: i32 = -1;
const RR_NO: i32 = 0;
const RR_POSSIBLY_OPTIMAL: i32 = 3;
const RR_POSSIBLY_DUAL_UNBOUNDED: i32 = 6;
const RR_POSSIBLY_SINGULAR_BASIS: i32 = 7;
const RR_CHOOSE_COLUMN_FAIL: i32 = 9;
const RR_EXCESSIVE_PRIMAL_VALUE: i32 = 11;

// HighsModelStatus
const MS_NOTSET: i32 = 0;
const MS_SOLVE_ERROR: i32 = 4;
const MS_OPTIMAL: i32 = 7;
const MS_INFEASIBLE: i32 = 8;
const MS_UNBOUNDED_OR_INFEASIBLE: i32 = 9;
const MS_OBJECTIVE_BOUND: i32 = 11;
const MS_UNKNOWN: i32 = 15;

// HighsStatus
const STATUS_ERROR: i32 = -1;
const STATUS_OK: i32 = 0;
const STATUS_WARNING: i32 = 1;

const NO_ROW_CHOSEN: i32 = -1;
const K_ACCEPT_DSE_WEIGHT_THRESHOLD: f64 = 0.25;
const K_NUMERICAL_TROUBLE_TOLERANCE: f64 = 1e-7;
const K_RUNNING_AVERAGE_MULTIPLIER: f64 = 0.05;
const K_ILLEGAL_INFEASIBILITY_COUNT: i32 = -1;

/// The messages logged by the driver, with ints and reals as listed: their
/// text is in Dual::message
pub mod msg {
    pub const NEAR_OPTIMAL: i32 = 0; // [num pr inf] [max, sum]
    pub const NEAR_OPTIMAL_NO_PERTURBATION: i32 = 1;
    pub const NEAR_OPTIMAL_USE_DEVEX: i32 = 2;
    pub const COMPUTE_DSE_WEIGHTS: i32 = 3;
    pub const CANNOT_CLEANUP: i32 = 4; // [level, num du inf]
    pub const PHASE1_START: i32 = 5;
    pub const PHASE1_OPTIMAL: i32 = 6;
    pub const RATIO_TEST_FAILED: i32 = 7; // user
    pub const EXCESSIVE_PRIMAL_VALUES: i32 = 8; // user
    pub const PHASE1_NOT_SOLVED: i32 = 9;
    pub const PHASE1_UNBOUNDED: i32 = 10;
    pub const CLEANING_UP_PHASE1: i32 = 11;
    pub const PHASE1_BAD_PHASE: i32 = 12; // [solve_phase]
    pub const PHASE2_NO_PERTURBATION: i32 = 13;
    pub const PHASE2_START: i32 = 14;
    pub const PHASE2_FOUND_FREE: i32 = 15;
    pub const PHASE2_OPTIMAL: i32 = 16;
    pub const PROBLEM_OPTIMAL: i32 = 17;
    pub const PHASE2_NOT_SOLVED: i32 = 18;
    pub const PROBLEM_PRIMAL_INFEASIBLE: i32 = 19;
    pub const CLEANUP_LEVEL_EXCEEDED: i32 = 20;
    pub const CLEANUP_SHIFT: i32 = 21;
    pub const DSE_WEIGHT_ERROR: i32 = 22; // [] [error]
    pub const SWITCH_DEVEX_COST: i32 = 23; // [num costly, iterations] [4 densities]
    pub const SWITCH_DEVEX_ERROR: i32 = 24; // [] [measure, threshold]
    pub const FLIPS: i32 = 25; // [num flip, num inf] [max, sum, min inf, max inf, sum inf, change]
    pub const SHIFTS: i32 = 26; // [num shift, num inf] [max, sum, max inf, sum inf, change]
    pub const SHIFT: i32 = 27; // [up] [shift, change]
    pub const PHASE1_OPTIMAL_NOT_PHASE2: i32 = 28; // [costs perturbed] [objective]
    pub const PHASE1_GO_PHASE2: i32 = 29;
    pub const PHASE1_FEASIBLE_WRT_PHASE1: i32 = 30; // [] [objective]
    pub const PHASE1_RETURN: i32 = 31; // [dual infeasibility count]
    pub const ALREADY_PERTURBED: i32 = 32;
    pub const REPERTURBING: i32 = 33;
    pub const FREE_SHIFT: i32 = 34; // [variable] [shift]
    pub const FREE_SHIFTS: i32 = 35; // [num] [sum]
    pub const POSSIBLE_LP_DUAL_INFEASIBILITY: i32 = 36; // [num] [objective, max, sum]
    pub const EXACT_DUAL_INFEASIBILITIES: i32 = 37; // [num] [max, sum]
    pub const EXACT_COL_RESIDUAL: i32 = 38; // [col] [exact, work, residual]
    pub const EXACT_ROW_RESIDUAL: i32 = 39; // [row] [exact, work, residual]
    pub const EXACT_RELATIVE_DELTA: i32 = 40; // [] [norm, norm delta, ratio]
    pub const OBJECTIVE_BOUND_EXCEEDED: i32 = 41; // [] [objective, bound]
    pub const DUAL_UB_BAILOUT: i32 = 42; // [have, iteration, frequency] [density, perturbed, exact]
    pub const BAD_BASIS_CHANGE: i32 = 43; // [variable out, variable in]
}

/// HEkkDual's scalars for reports (HEkkDual::iterationAnalysisData)
#[repr(C)]
#[derive(Default)]
pub struct DualState {
    pub solve_phase: i32,
    pub edge_weight_mode: i32,
    pub num_devex_iterations: i32,
    pub row_out: i32,
    pub variable_out: i32,
    pub variable_in: i32,
    pub rebuild_reason: i32,
    pub delta_primal: f64,
    pub theta_primal: f64,
    pub theta_dual: f64,
    pub alpha_col: f64,
    pub alpha_row: f64,
    pub numerical_trouble: f64,
}

/// The values that HEkkDual::iterationAnalysisData copies into
/// HEkk::analysis_ after each iteration, kept until C++ needs them (the
/// report at the end of the solve)
#[repr(C)]
#[derive(Default)]
pub struct AnalysisData {
    pub state: DualState,
    pub iteration_count: i32,
    pub factor_pivot_threshold: f64,
    pub edge_weight_error: f64,
    pub updated_dual_objective_value: f64,
    pub num_primal_infeasibilities: i32,
    pub sum_primal_infeasibilities: f64,
    /// Those of the LP in phase 1 (HighsSimplexAnalysis), else info_'s
    pub num_dual_infeasibilities: i32,
    pub sum_dual_infeasibilities: f64,
    pub col_aq_density: f64,
    pub row_ep_density: f64,
    pub row_ap_density: f64,
    pub row_dse_density: f64,
    pub col_bfrt_density: f64,
    pub primal_col_density: f64,
    pub dual_col_density: f64,
    pub num_costly_dse_iteration: i32,
    pub costly_dse_measure: f64,
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

/// std::min(a, b)
#[inline]
fn std_min(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// Whether a message goes to highsLogUser (else highsLogDev), and its
/// HighsLogType
fn msg_kind(id: i32) -> (bool, i32) {
    use msg::*;
    match id {
        RATIO_TEST_FAILED | EXCESSIVE_PRIMAL_VALUES => (true, LOG_ERROR),
        NEAR_OPTIMAL | NEAR_OPTIMAL_NO_PERTURBATION | NEAR_OPTIMAL_USE_DEVEX | COMPUTE_DSE_WEIGHTS | PHASE1_START
        | PHASE1_OPTIMAL | PHASE2_START | PHASE2_FOUND_FREE | PHASE2_OPTIMAL | PROBLEM_OPTIMAL | CLEANUP_SHIFT
        | FLIPS | SHIFTS | REPERTURBING | FREE_SHIFTS | OBJECTIVE_BOUND_EXCEEDED => (false, LOG_DETAILED),
        SHIFT | FREE_SHIFT => (false, LOG_VERBOSE),
        CANNOT_CLEANUP | CLEANING_UP_PHASE1 | PHASE2_NO_PERTURBATION | EXACT_COL_RESIDUAL | EXACT_ROW_RESIDUAL
        | EXACT_RELATIVE_DELTA | BAD_BASIS_CHANGE => (false, LOG_WARNING),
        CLEANUP_LEVEL_EXCEEDED => (false, LOG_ERROR),
        _ => (false, LOG_INFO),
    }
}

/// Run `f` on a view of an HVector
#[inline]
fn with_vec<R>(c: &mut OwnedHVec, f: impl FnOnce(&mut HVec) -> R) -> R {
    c.with(f)
}

/// HEkkDual's HVectors
struct DualVectors {
    row_ep: OwnedHVec,
    row_ap: OwnedHVec,
    col_aq: OwnedHVec,
    col_bfrt: OwnedHVec,
}

pub struct Dual<'a> {
    e: EkkView<'a>,
    x: &'a CHekk,
    devex_index: &'a mut [i32],
    num_tot_permutation: &'a [i32],
    v: DualVectors,
    row: DualRow,
    rhs: DualRhs,
    work_col: OwnedHVec,
    work_row: OwnedHVec,
    refactor_info_dirty: bool,
    bailout: Bailout,
    num_row: usize,
    num_col: usize,
    num_tot: usize,
    inv_num_row: f64,
    // HEkkDual
    edge_weight_mode: i32,
    td: f64,
    force_phase2: bool,
    solve_phase: i32,
    rebuild_reason: i32,
    dual_infeas_count: i32,
    row_out: i32,
    variable_out: i32,
    move_out: i32,
    variable_in: i32,
    delta_primal: f64,
    theta_dual: f64,
    theta_primal: f64,
    alpha_col: f64,
    alpha_row: f64,
    numerical_trouble: f64,
    computed_edge_weight: f64,
    num_devex_iterations: i32,
    new_devex_framework: bool,
    /// analysis_.num_correct_dual_primal_flip, for the rebuild
    num_correct_dual_primal_flip: i32,
    /// The latest phase 1 LP dual infeasibilities (HighsSimplexAnalysis)
    lp_dual_infeasibility: Infeasibility,
    /// The analysis data of the last iteration, if not yet in analysis_
    analysis_data: AnalysisData,
    analysis_data_pending: bool,
    // HEkkDualRow's scalars
    pack_count: usize,
    work_count: usize,
    work_theta: f64,
    work_delta: f64,
    work_pivot: i32,
    work_alpha: f64,
}

impl<'a> Dual<'a> {
    /// HEkkDual::HEkkDual
    ///
    /// # Safety
    /// As for CHekk::view: `x` filled for the solve, and no other view of
    /// HEkk's data used while this solver runs
    pub unsafe fn new(x: &'a CHekk) -> Self {
        let e = x.view();
        let devex_index = x.devex_index.get_mut();
        let num_tot_permutation = x.num_tot_permutation.get();
        let (num_row, num_col) = (e.num_row, e.num_col);
        let num_tot = num_row + num_col;
        let mut row = DualRow::default();
        row.setup_slice(num_tot);
        let mut rhs = DualRhs::default();
        rhs.setup(num_row as i32);
        Dual {
            e,
            x,
            devex_index,
            num_tot_permutation,
            v: DualVectors {
                row_ep: OwnedHVec::new(num_row as i32),
                row_ap: OwnedHVec::new(num_col as i32),
                col_aq: OwnedHVec::new(num_row as i32),
                col_bfrt: OwnedHVec::new(num_row as i32),
            },
            row,
            rhs,
            work_col: OwnedHVec::new(num_row as i32),
            work_row: OwnedHVec::new(num_col as i32),
            refactor_info_dirty: true,
            bailout: Bailout::new(),
            num_row,
            num_col,
            num_tot,
            inv_num_row: 1.0 / num_row as f64,
            edge_weight_mode: EW_DSE,
            td: 0.0,
            force_phase2: false,
            solve_phase: 0,
            rebuild_reason: RR_NO,
            // Uninitialised in the C++, where it is read only when the
            // initial duals are feasible and not near-optimal
            dual_infeas_count: 0,
            row_out: NO_ROW_CHOSEN,
            variable_out: -1,
            move_out: 0,
            variable_in: -1,
            delta_primal: 0.0,
            theta_dual: 0.0,
            theta_primal: 0.0,
            alpha_col: 0.0,
            alpha_row: 0.0,
            numerical_trouble: 0.0,
            computed_edge_weight: 0.0,
            num_devex_iterations: 0,
            new_devex_framework: false,
            num_correct_dual_primal_flip: 0,
            lp_dual_infeasibility: Infeasibility::default(),
            analysis_data: AnalysisData::default(),
            analysis_data_pending: false,
            pack_count: 0,
            work_count: 0,
            work_theta: 0.0,
            work_delta: 0.0,
            work_pivot: -1,
            work_alpha: 0.0,
        }
    }

    fn records(&mut self) -> &mut BasisRecords {
        self.x.records()
    }

    /// Log a message
    fn log(&self, id: i32, ints: &[i32], reals: &[f64]) {
        let (user, log_type) = msg_kind(id);
        if user {
            self.x.user(log_type, &self.message(id, ints, reals));
        } else {
            self.x.dev(log_type, || self.message(id, ints, reals));
        }
    }

    /// A highsLogDev message
    fn dev(&self, id: i32, ints: &[i32], reals: &[f64]) {
        self.log(id, ints, reals);
    }

    fn state(&self) -> DualState {
        DualState {
            solve_phase: self.solve_phase,
            edge_weight_mode: self.edge_weight_mode,
            num_devex_iterations: self.num_devex_iterations,
            row_out: self.row_out,
            variable_out: self.variable_out,
            variable_in: self.variable_in,
            rebuild_reason: self.rebuild_reason,
            delta_primal: self.delta_primal,
            theta_primal: self.theta_primal,
            theta_dual: self.theta_dual,
            alpha_col: self.alpha_col,
            alpha_row: self.alpha_row,
            numerical_trouble: self.numerical_trouble,
        }
    }

    fn return_from_solve(&mut self, status: i32) -> i32 {
        self.flush_analysis_data();
        hekk::return_from_solve(&mut self.e, self.x, status)
    }

    /// Give HEkk::analysis_ the data of the last iteration, if a report
    /// has not done so since
    fn flush_analysis_data(&mut self) {
        if self.analysis_data_pending {
            self.x.dual_report(0, &self.analysis_data, 0, self.e.sense);
            self.analysis_data_pending = false;
        }
    }

    /// HEkkDual::iterationAnalysisData, kept for C++
    fn record_analysis_data(&mut self) {
        let e = &self.e;
        let x = self.x;
        let phase1 = self.solve_phase == PHASE_1;
        self.analysis_data = AnalysisData {
            state: self.state(),
            iteration_count: x.iteration_count.get(),
            factor_pivot_threshold: x.factor_pivot_threshold.get(),
            edge_weight_error: x.edge_weight_error.get(),
            updated_dual_objective_value: *e.updated_dual_objective_value,
            num_primal_infeasibilities: *e.num_primal_infeasibilities,
            sum_primal_infeasibilities: *e.sum_primal_infeasibilities,
            num_dual_infeasibilities: if phase1 {
                self.lp_dual_infeasibility.num
            } else {
                *e.num_dual_infeasibilities
            },
            sum_dual_infeasibilities: if phase1 {
                self.lp_dual_infeasibility.sum
            } else {
                *e.sum_dual_infeasibilities
            },
            col_aq_density: *e.col_aq_density,
            row_ep_density: *e.row_ep_density,
            row_ap_density: *e.row_ap_density,
            row_dse_density: *e.row_dse_density,
            col_bfrt_density: x.col_bfrt_density.get(),
            primal_col_density: *e.primal_col_density,
            dual_col_density: *e.dual_col_density,
            num_costly_dse_iteration: x.num_costly_dse_iteration.get(),
            costly_dse_measure: x.costly_dse_measure.get(),
        };
        self.analysis_data_pending = true;
    }

    fn primal(e: &mut EkkView<'a>, x: &CHekk) -> Primal<'static> {
        // SAFETY: the base arrays are HEkk's, live for the solve; the
        // Primal is used within a statement while `e` is not otherwise
        // accessed for them
        unsafe {
            let n = e.base_value.len().min(e.base_lower.len()).min(e.base_upper.len());
            Primal {
                base_value: std::slice::from_raw_parts_mut(e.base_value.as_mut_ptr(), n),
                base_lower: std::slice::from_raw_parts(e.base_lower.as_ptr(), n),
                base_upper: std::slice::from_raw_parts(e.base_upper.as_ptr(), n),
                tp: e.primal_feasibility_tolerance,
                squared: x.store_squared_primal_infeasibility.get(),
            }
        }
    }

    // ---- HEkk methods with side effects beyond the kernels ----

    fn clear_fresh_values(&self) {
        self.x.clear_fresh_values();
    }

    /// HEkk::computeDual
    fn compute_dual(&mut self) {
        self.x.dual_values_valid.set(false);
        self.work_col.clear();
        self.work_row.clear();
        let e = &mut self.e;
        let row = &mut self.work_row;
        self.work_col.with(|c| row.with(|r| e.compute_dual(c, r)));
    }

    /// HEkk::computePrimal
    fn compute_primal(&mut self) {
        self.work_col.clear();
        let e = &mut self.e;
        self.work_col.with(|c| e.compute_primal(c));
    }

    /// HEkk::computeDualObjectiveValue
    fn compute_dual_objective_value(&mut self, phase: i32) {
        self.e.compute_dual_objective_value(phase);
        self.x.has_dual_objective_value.set(true);
    }

    /// HEkk::initialiseCost for the dual simplex
    fn initialise_cost(&mut self, perturb: bool) {
        hekk::initialise_cost(&mut self.e, self.x, ALGORITHM_DUAL, perturb);
    }

    /// HEkk::initialiseBound and HEkk::initialiseNonbasicValueAndMove
    fn initialise_bound_and_values(&mut self, phase: i32) {
        self.e.initialise_bound(ALGORITHM_DUAL, phase, false);
        self.e.initialise_nonbasic_value_and_move();
    }

    /// HEkk::computeSimplexLpDualInfeasible
    fn compute_simplex_lp_dual_infeasible(&mut self) {
        self.lp_dual_infeasibility = self.e.compute_simplex_lp_dual_infeasible();
    }

    /// HEkk::resetSyntheticClock
    fn reset_synthetic_clock(&mut self) {
        hekk::reset_synthetic_clock(&mut self.e, self.x);
    }

    /// HEkk::rebuildRefactor
    fn rebuild_refactor(&mut self, rebuild_reason: i32) -> bool {
        hekk::rebuild_refactor(&mut self.e, self.x, rebuild_reason)
    }

    /// HEkk::bailout
    fn bailout(&mut self) -> bool {
        self.bailout.check(self.x)
    }

    // ---- HEkkDual ----

    pub fn solve(&mut self, pass_force_phase2: bool) -> i32 {
        let x = self.x;
        self.initialise_solve();
        if !self.start() {
            return self.return_from_solve(STATUS_ERROR);
        }
        // Determine the duals without cost perturbation: unless just
        // computed by initialiseForSolve with these costs
        self.initialise_cost(false);
        if !x.fresh_unperturbed_dual.get() {
            self.compute_dual();
        }
        x.fresh_unperturbed_dual.set(false);
        self.e.compute_simplex_dual_infeasible();
        // Record whether the solution with unperturbed costs is dual
        // feasible
        let dual_feasible_with_unperturbed_costs = *self.e.num_dual_infeasibilities == 0;
        // Force phase 2 if dual infeasibilities without cost perturbation
        // involved fixed variables or were (at most) small
        self.force_phase2 = pass_force_phase2
            || *self.e.max_dual_infeasibility * *self.e.max_dual_infeasibility < self.e.dual_feasibility_tolerance;
        let no_simplex_dual_infeasibilities = dual_feasible_with_unperturbed_costs || self.force_phase2;
        let near_optimal = no_simplex_dual_infeasibilities
            && *self.e.num_primal_infeasibilities < 1000
            && *self.e.max_primal_infeasibility < 1e-3;
        if near_optimal {
            self.dev(
                msg::NEAR_OPTIMAL,
                &[*self.e.num_primal_infeasibilities],
                &[*self.e.max_primal_infeasibility, *self.e.sum_primal_infeasibilities],
            );
        }
        // Perturb costs according to whether the solution is near-optimal
        let perturb_costs = !near_optimal;
        if !perturb_costs {
            self.dev(msg::NEAR_OPTIMAL_NO_PERTURBATION, &[], &[]);
        }
        self.initialise_cost(perturb_costs);
        // Check whether the time/iteration limit has been reached. First
        // point at which a non-error return can occur
        if self.bailout() {
            return self.return_from_solve(STATUS_WARNING);
        }
        // Consider initialising edge weights
        let has_weights = x.has_dual_steepest_edge_weights.get();
        if !has_weights {
            // Assign unit weights: the vectors are sized by C++
            hekk::assign_unit_dual_edge_weights(&mut self.e);
            // Unit weights are assigned: correct for steepest edge when
            // B=I
            if self.edge_weight_mode == EW_DSE {
                if self.logical_basis() {
                    x.has_dual_steepest_edge_weights.set(true);
                } else if hekk::restore_dual_edge_weights(&mut self.e, x, near_optimal) {
                    x.has_dual_steepest_edge_weights.set(true);
                } else if near_optimal {
                    // Use Devex rather than compute steepest edge weights
                    self.dev(msg::NEAR_OPTIMAL_USE_DEVEX, &[], &[]);
                    self.edge_weight_mode = EW_DEVEX;
                } else if self.num_row as i32 > x.simplex_dse_exact_init_max_rows {
                    // Start from unit weights: CHUZR recomputes the weight
                    // of each chosen row
                    x.has_dual_steepest_edge_weights.set(true);
                } else {
                    self.dev(msg::COMPUTE_DSE_WEIGHTS, &[], &[]);
                    let mut row_ep = OwnedHVec::new(self.num_row as i32);
                    let e = &mut self.e;
                    row_ep.with(|v| e.compute_dual_steepest_edge_weights(v));
                    x.has_dual_steepest_edge_weights.set(true);
                }
            }
            if self.edge_weight_mode == EW_DEVEX {
                self.initialise_devex_framework();
            }
        }
        if perturb_costs {
            // Compute the dual values with perturbed costs
            self.compute_dual();
            // Determine the number of dual infeasibilities after fixed
            // variable flips
            self.compute_dual_infeasibilities_with_fixed_variable_flips();
            self.dual_infeas_count = *self.e.num_dual_infeasibilities;
        }
        // The dual values are now those for the current costs, so the
        // first rebuild of phase 2 need not recompute them
        x.fresh_dual.set(true);
        self.solve_phase = if self.force_phase2 || self.dual_infeas_count <= 0 { PHASE_2 } else { PHASE_1 };
        // The major solving loop
        while self.solve_phase != 0 {
            let it0 = x.iteration_count.get();
            // When starting a new phase the (updated) dual objective
            // function value isn't known
            x.has_dual_objective_value.set(false);
            if self.solve_phase == PHASE_UNKNOWN {
                self.clear_fresh_values();
                // Reset the phase 2 bounds so that true number of dual
                // infeasibilities can be determined
                self.initialise_bound_and_values(PHASE_UNKNOWN);
                self.compute_dual_infeasibilities_with_fixed_variable_flips();
                self.dual_infeas_count = *self.e.num_dual_infeasibilities;
                self.solve_phase = if self.dual_infeas_count > 0 { PHASE_1 } else { PHASE_2 };
                if *self.e.backtracking {
                    // Backtracking, so set the bounds and primal values
                    self.initialise_bound_and_values(self.solve_phase);
                    *self.e.backtracking = false;
                }
            }
            if self.solve_phase == PHASE_1 {
                self.clear_fresh_values();
                self.solve_phase1();
                x.dual_phase1_iteration_count.set(x.dual_phase1_iteration_count.get() + x.iteration_count.get() - it0);
            } else if self.solve_phase == PHASE_2 {
                self.solve_phase2();
                x.dual_phase2_iteration_count.set(x.dual_phase2_iteration_count.get() + x.iteration_count.get() - it0);
            } else {
                x.model_status.set(MS_SOLVE_ERROR);
                return self.return_from_solve(STATUS_ERROR);
            }
            if x.solve_bailout.get() {
                return self.return_from_solve(STATUS_WARNING);
            }
            match self.solve_phase {
                PHASE_TABOO_BASIS => {
                    x.model_status.set(MS_UNKNOWN);
                    return self.return_from_solve(STATUS_WARNING);
                }
                PHASE_ERROR => return self.return_from_solve(STATUS_ERROR),
                PHASE_EXIT | PHASE_OPTIMAL_CLEANUP | PHASE_PRIMAL_INFEASIBLE_CLEANUP => break,
                _ => {}
            }
        }
        if self.solve_phase == PHASE_OPTIMAL_CLEANUP || self.solve_phase == PHASE_PRIMAL_INFEASIBLE_CLEANUP {
            x.dual_simplex_cleanup_level.set(x.dual_simplex_cleanup_level.get() + 1);
            if self.solve_phase == PHASE_PRIMAL_INFEASIBLE_CLEANUP {
                // HEkk::computeSimplexInfeasible
                self.e.compute_simplex_primal_infeasible();
                self.e.compute_simplex_dual_infeasible();
            }
            if x.dual_simplex_cleanup_level.get() > x.max_dual_simplex_cleanup_level {
                self.dev(
                    msg::CANNOT_CLEANUP,
                    &[x.dual_simplex_cleanup_level.get(), *self.e.num_dual_infeasibilities],
                    &[],
                );
                x.model_status.set(if self.solve_phase == PHASE_OPTIMAL_CLEANUP { MS_OPTIMAL } else { MS_INFEASIBLE });
            } else {
                self.flush_analysis_data();
                let return_status = self.primal_cleanup();
                self.refactor_info_dirty = true;
                if return_status != STATUS_OK {
                    return self.return_from_solve(return_status);
                }
            }
        }
        // Optimal without a clean-up: the dual values were computed from
        // scratch with the LP's costs after the last basis change
        if x.model_status.get() == MS_OPTIMAL
            && self.solve_phase == PHASE_OPTIMAL
            && !*self.e.costs_perturbed
            && !*self.e.costs_shifted
        {
            hekk::record_dual_values(&self.e, self.x);
        }
        self.return_from_solve(STATUS_OK)
    }

    fn initialise_solve(&mut self) {
        self.td = self.e.dual_feasibility_tolerance;
        self.interpret_dual_edge_weight_strategy(self.x.dual_edge_weight_strategy);
        self.x.model_status.set(MS_NOTSET);
        self.x.solve_bailout.set(false);
        self.x.called_return_from_solve.set(false);
        self.x.exit_algorithm.set(ALGORITHM_DUAL);
        self.rebuild_reason = RR_NO;
    }

    /// The checks and settings at the start of HEkkDual::solve: false on
    /// error
    fn start(&mut self) -> bool {
        // Assumes that the LP has a positive number of rows
        if hekk::is_unconstrained_lp(&self.e, self.x) {
            return false;
        }
        // Possibly use Li dual steepest edge weights by not storing
        // squared primal infeasibilities
        hekk::possibly_use_li_dual_steepest_edge(&self.e, self.x);
        if !self.e.status.has_invert {
            self.x.dev(LOG_ERROR, || sprintf!("HDual:: Should enter solve with INVERT\n"));
            return false;
        }
        true
    }

    /// Clean up dual infeasibilities with the primal simplex: returns the
    /// status of the call
    fn primal_cleanup(&mut self) -> i32 {
        let x = self.x;
        let e = &self.e;
        x.dev(LOG_INFO, || {
            sprintf!(
                "HEkkDual:: Using primal simplex to try to clean up num / max / sum = %d / %g / %g dual infeasibilities\n",
                *e.num_dual_infeasibilities,
                *e.max_dual_infeasibility,
                *e.sum_dual_infeasibilities
            )
        });
        // Switch off any bound perturbation
        let save_primal_simplex_bound_perturbation_multiplier = *self.e.primal_simplex_bound_perturbation_multiplier;
        *self.e.primal_simplex_bound_perturbation_multiplier = 0.0;
        let call_status = hekk::primal_solve(x, true);
        // Restore any bound perturbation
        *self.e.primal_simplex_bound_perturbation_multiplier = save_primal_simplex_bound_perturbation_multiplier;
        let return_status = hekk::interpret_call_status(x, call_status, STATUS_OK, "HEkkPrimal::solve");
        // Reset called_return_from_solve_ to be false, since it's called
        // for this solve
        x.called_return_from_solve.set(false);
        if return_status != STATUS_OK {
            return return_status;
        }
        let e = &self.e;
        if x.model_status.get() == MS_OPTIMAL && *e.num_primal_infeasibilities + *e.num_dual_infeasibilities != 0 {
            x.dev(LOG_WARNING, || {
                sprintf!(
                    "HEkkDual:: Primal simplex clean up yields optimality, but with %d (max %g) primal infeasibilities and %d (max %g) dual infeasibilities\n",
                    *e.num_primal_infeasibilities,
                    *e.max_primal_infeasibility,
                    *e.num_dual_infeasibilities,
                    *e.max_dual_infeasibility
                )
            });
        }
        STATUS_OK
    }

    /// The text of a message
    fn message(&self, id: i32, i: &[i32], r: &[f64]) -> String {
        use msg::*;
        match id {
            NEAR_OPTIMAL => sprintf!(
                "Dual feasible with unperturbed costs and num / max / sum primal infeasibilities of %d / %g / %g, so near-optimal\n",
                i[0],
                r[0],
                r[1]
            ),
            NEAR_OPTIMAL_NO_PERTURBATION => "Near-optimal, so don't use cost perturbation\n".into(),
            NEAR_OPTIMAL_USE_DEVEX => {
                "Basis is not logical, but near-optimal, so use Devex rather than compute steepest edge weights\n"
                    .into()
            }
            COMPUTE_DSE_WEIGHTS => "Basis is not logical, so compute steepest edge weights\n".into(),
            CANNOT_CLEANUP => sprintf!(
                "HEkkDual:: Cannot use level %d primal simplex cleanup for %d dual infeasibilities\n",
                i[0],
                i[1]
            ),
            PHASE1_START => "dual-phase-1-start\n".into(),
            PHASE1_OPTIMAL => "dual-phase-1-optimal\n".into(),
            RATIO_TEST_FAILED => "Dual simplex ratio test failed due to excessive dual values: consider scaling down the LP objective coefficients\n".into(),
            EXCESSIVE_PRIMAL_VALUES => {
                "Dual simplex detected excessive primal values: consider scaling down the LP bounds\n".into()
            }
            PHASE1_NOT_SOLVED => "dual-phase-1-not-solved\n".into(),
            PHASE1_UNBOUNDED => "dual-phase-1-unbounded\n".into(),
            CLEANING_UP_PHASE1 => "Cleaning up cost perturbation when unbounded in phase 1\n".into(),
            PHASE1_BAD_PHASE => sprintf!(
                "HEkkDual::solvePhase1 solve_phase == %d (solve call %d; iter %d)\n",
                i[0],
                self.x.debug_solve_call_num,
                self.x.iteration_count.get()
            ),
            PHASE2_NO_PERTURBATION => "Moving to phase 2, but not allowing cost perturbation\n".into(),
            PHASE2_START => "dual-phase-2-start\n".into(),
            PHASE2_FOUND_FREE => "dual-phase-2-found-free\n".into(),
            PHASE2_OPTIMAL => "dual-phase-2-optimal\n".into(),
            PROBLEM_OPTIMAL => "problem-optimal\n".into(),
            PHASE2_NOT_SOLVED => "dual-phase-2-not-solved\n".into(),
            PROBLEM_PRIMAL_INFEASIBLE => "problem-primal-infeasible\n".into(),
            CLEANUP_LEVEL_EXCEEDED => sprintf!("Dual simplex cleanup level has exceeded limit of %d\n", i[0]),
            CLEANUP_SHIFT => "dual-cleanup-shift\n".into(),
            DSE_WEIGHT_ERROR => sprintf!("Dual steepest edge weight error is %g\n", r[0]),
            SWITCH_DEVEX_COST => sprintf!(
                "Switch from DSE to Devex after %d costly DSE iterations of %d with densities C_Aq = %11.4g; R_Ep = %11.4g; R_Ap = %11.4g; DSE = %11.4g\n",
                i[0],
                i[1],
                r[0],
                r[1],
                r[2],
                r[3]
            ),
            SWITCH_DEVEX_ERROR => sprintf!(
                "Switch from DSE to Devex with log error measure of %g > %g = threshold\n",
                r[0],
                r[1]
            ),
            FLIPS => sprintf!(
                "Performed num / max / sum = %d / %g / %g flip(s) for num / min / max / sum dual infeasibility of %d / %g / %g / %g; objective change = %g\n",
                i[0],
                r[0],
                r[1],
                i[1],
                r[2],
                r[3],
                r[4],
                r[5]
            ),
            SHIFTS => sprintf!(
                "Performed num / max / sum = %d / %g / %g shift(s) for num / max / sum dual infeasibility of %d / %g / %g; objective change = %g\n",
                i[0],
                r[0],
                r[1],
                i[1],
                r[2],
                r[3],
                r[4]
            ),
            SHIFT => sprintf!(
                "Move %s: cost shift = %g; objective change = %g\n",
                if i[0] != 0 { "  up" } else { "down" },
                r[0],
                r[1]
            ),
            PHASE1_OPTIMAL_NOT_PHASE2 => sprintf!(
                "Optimal in phase 1 but not jumping to phase 2 since dual objective is %10.4g: Costs perturbed = %d\n",
                r[0],
                i[0]
            ),
            PHASE1_GO_PHASE2 => {
                "LP is dual feasible wrt Phase 2 bounds after removing cost perturbations so go to phase 2\n".into()
            }
            PHASE1_FEASIBLE_WRT_PHASE1 => sprintf!(
                "LP is dual feasible wrt Phase 1 bounds after removing cost perturbations: dual objective is %10.4g\n",
                r[0]
            ),
            PHASE1_RETURN => sprintf!(
                "LP has %d dual feasibilities wrt Phase 1 bounds after removing cost perturbations so return to phase 1\n",
                i[0]
            ),
            ALREADY_PERTURBED => "Costs are already perturbed in exitPhase1ResetDuals\n".into(),
            REPERTURBING => "Re-perturbing costs when optimal in phase 1\n".into(),
            FREE_SHIFT => sprintf!("Variable %d is free: shift cost to zero dual of %g\n", i[0], r[0]),
            FREE_SHIFTS => sprintf!(
                "Performed %d cost shift(s) for free variables to zero dual values: total = %g\n",
                i[0],
                r[0]
            ),
            POSSIBLE_LP_DUAL_INFEASIBILITY => sprintf!(
                "LP is dual %s with dual phase 1 objective %10.4g and num / max / sum dual infeasibilities = %d / %9.4g / %9.4g\n",
                if i[0] != 0 { "infeasible" } else { "feasible" },
                r[0],
                i[0],
                r[1],
                r[2]
            ),
            EXACT_DUAL_INFEASIBILITIES => sprintf!(
                "When computing exact dual objective, the unperturbed costs yield num / max / sum dual infeasibilities = %d / %g / %g\n",
                i[0],
                r[0],
                r[1]
            ),
            EXACT_COL_RESIDUAL => sprintf!(
                "Col %4d: ExactDual = %11.4g; WorkDual = %11.4g; Residual = %11.4g\n",
                i[0],
                r[0],
                r[1],
                r[2]
            ),
            EXACT_ROW_RESIDUAL => sprintf!(
                "Row %4d: ExactDual = %11.4g; WorkDual = %11.4g; Residual = %11.4g\n",
                i[0],
                r[0],
                r[1],
                r[2]
            ),
            EXACT_RELATIVE_DELTA => sprintf!(
                "||exact dual vector|| = %g; ||delta dual vector|| = %g: ratio = %g\n",
                r[0],
                r[1],
                r[2]
            ),
            OBJECTIVE_BOUND_EXCEEDED => {
                sprintf!("HEkkDual::solvePhase2: %12g = Objective > ObjectiveUB = %12g\n", r[0], r[1])
            }
            DUAL_UB_BAILOUT => sprintf!(
                "%s on iteration %d: Density %11.4g; Frequency %d: Residual(Perturbed = %g; Exact = %g)\n",
                if i[0] != 0 { "Have DualUB bailout" } else { "No   DualUB bailout" },
                i[1],
                r[0],
                i[2],
                r[1],
                r[2]
            ),
            BAD_BASIS_CHANGE => sprintf!(" basis change (%d out; %d in) is bad\n", i[0], i[1]),
            _ => String::new(),
        }
    }

    fn interpret_dual_edge_weight_strategy(&mut self, strategy: i32) {
        self.edge_weight_mode = match strategy {
            0 => EW_DANTZIG,
            1 => EW_DEVEX,
            // Choose (-1), steepest edge (2), or unrecognised
            _ => EW_DSE,
        };
    }

    /// HEkk::logicalBasis
    fn logical_basis(&self) -> bool {
        self.e.basic_index[..self.num_row].iter().all(|&i| i as usize >= self.num_col)
    }

    fn solve_phase1(&mut self) {
        let x = self.x;
        x.has_primal_objective_value.set(false);
        x.has_dual_objective_value.set(false);
        self.rebuild_reason = RR_NO;
        if self.bailout() {
            return;
        }
        self.dev(msg::PHASE1_START, &[], &[]);
        // Switch to dual phase 1 bounds
        self.initialise_bound_and_values(self.solve_phase);
        // If there's no backtracking basis, save the initial basis in case
        // of backtracking
        if !x.valid_backtracking_basis.get() {
            hekk::put_backtracking_basis(&mut self.e, x);
        }
        loop {
            self.rebuild();
            if self.solve_phase == PHASE_ERROR {
                x.model_status.set(MS_SOLVE_ERROR);
                return;
            }
            if self.solve_phase == PHASE_UNKNOWN {
                // If backtracking, may change phase, so drop out
                return;
            }
            if self.bailout() {
                break;
            }
            loop {
                self.iterate();
                if self.bailout() {
                    break;
                }
                if self.rebuild_reason != 0 {
                    break;
                }
            }
            if x.solve_bailout.get() {
                break;
            }
            let finished = x.has_fresh_rebuild.get() && !self.rebuild_refactor(self.rebuild_reason);
            if finished && self.records().taboo() {
                self.solve_phase = PHASE_TABOO_BASIS;
                return;
            }
            if finished {
                break;
            }
        }
        if x.solve_bailout.get() {
            return;
        }
        // Assess outcome of dual phase 1
        if self.row_out == NO_ROW_CHOSEN {
            self.dev(msg::PHASE1_OPTIMAL, &[], &[]);
            if *self.e.dual_objective_value == 0.0 {
                // Zero phase 1 objective so go to phase 2
                self.solve_phase = PHASE_2;
            } else {
                self.assess_phase1_optimality();
            }
        } else if self.rebuild_reason == RR_CHOOSE_COLUMN_FAIL || self.rebuild_reason == RR_EXCESSIVE_PRIMAL_VALUE {
            self.solve_phase = PHASE_ERROR;
            self.report_ratio_test_or_excessive_primal();
            self.dev(msg::PHASE1_NOT_SOLVED, &[], &[]);
            x.model_status.set(MS_SOLVE_ERROR);
        } else if self.variable_in == -1 {
            // We got dual phase 1 unbounded - strange
            self.dev(msg::PHASE1_UNBOUNDED, &[], &[]);
            if *self.e.costs_perturbed {
                // Clean up perturbation
                self.cleanup();
                self.dev(msg::CLEANING_UP_PHASE1, &[], &[]);
                if self.dual_infeas_count == 0 {
                    self.solve_phase = PHASE_2;
                }
            } else {
                self.solve_phase = PHASE_ERROR;
                self.dev(msg::PHASE1_NOT_SOLVED, &[], &[]);
                x.model_status.set(MS_SOLVE_ERROR);
            }
        }
        let solve_phase_ok = matches!(self.solve_phase, PHASE_1 | PHASE_2 | PHASE_EXIT | PHASE_ERROR);
        if !solve_phase_ok {
            self.dev(msg::PHASE1_BAD_PHASE, &[self.solve_phase], &[]);
        }
        if self.solve_phase == PHASE_2 || self.solve_phase == PHASE_EXIT || self.solve_phase == PHASE_ERROR {
            // Moving to phase 2 or exiting, so make sure that the simplex
            // bounds and nonbasic value/move correspond to the LP
            self.initialise_bound_and_values(PHASE_2);
            if self.solve_phase == PHASE_2 {
                // Moving to phase 2 so possibly reinstate cost perturbation
                if x.dual_simplex_phase1_cleanup_level.get() < x.max_dual_simplex_phase1_cleanup_level {
                    x.allow_cost_shifting.set(true);
                    x.allow_cost_perturbation.set(true);
                }
                if !x.allow_cost_perturbation.get() {
                    self.dev(msg::PHASE2_NO_PERTURBATION, &[], &[]);
                }
            }
        }
    }

    fn report_ratio_test_or_excessive_primal(&self) {
        // Solve error is opaque to users, so put in some logging
        if self.rebuild_reason == RR_CHOOSE_COLUMN_FAIL {
            self.log(msg::RATIO_TEST_FAILED, &[], &[]);
        } else {
            self.log(msg::EXCESSIVE_PRIMAL_VALUES, &[], &[]);
        }
    }

    fn solve_phase2(&mut self) {
        let x = self.x;
        x.has_primal_objective_value.set(false);
        x.has_dual_objective_value.set(false);
        self.rebuild_reason = RR_NO;
        self.solve_phase = PHASE_2;
        x.solve_bailout.set(false);
        if self.bailout() {
            return;
        }
        self.dev(msg::PHASE2_START, &[], &[]);
        // Collect free variables
        let n = self.num_tot;
        self.row.create_freelist(&self.e.nonbasic_flag[..n], &self.e.work_lower[..n], &self.e.work_upper[..n]);
        // If there's no backtracking basis, save the initial basis in
        // case of backtracking
        if !x.valid_backtracking_basis.get() {
            hekk::put_backtracking_basis(&mut self.e, x);
        }
        loop {
            // Rebuild all values, reinverting B if updates have been
            // performed
            self.rebuild();
            if self.solve_phase == PHASE_ERROR {
                x.model_status.set(MS_SOLVE_ERROR);
                return;
            }
            if self.solve_phase == PHASE_UNKNOWN {
                return;
            }
            if self.bailout() {
                break;
            }
            if self.bailout_on_dual_objective() {
                break;
            }
            if self.dual_infeas_count > 0 {
                break;
            }
            loop {
                self.iterate();
                if self.bailout() {
                    break;
                }
                if self.bailout_on_dual_objective() {
                    break;
                }
                // If possibly dual unbounded, assess whether this implies
                // primal infeasibility
                if self.rebuild_reason == RR_POSSIBLY_DUAL_UNBOUNDED {
                    self.assess_possibly_dual_unbounded();
                }
                if self.rebuild_reason != 0 {
                    break;
                }
            }
            if x.solve_bailout.get() {
                break;
            }
            let finished = x.has_fresh_rebuild.get() && !self.rebuild_refactor(self.rebuild_reason);
            if finished && self.records().taboo() {
                self.solve_phase = PHASE_TABOO_BASIS;
                return;
            }
            if finished {
                break;
            }
        }
        if x.solve_bailout.get() {
            return;
        }
        // Assess outcome of dual phase 2
        if self.dual_infeas_count > 0 {
            self.dev(msg::PHASE2_FOUND_FREE, &[], &[]);
            self.solve_phase = PHASE_1;
        } else if self.row_out == NO_ROW_CHOSEN {
            // There is no candidate in CHUZR, even after rebuild so
            // probably optimal
            self.dev(msg::PHASE2_OPTIMAL, &[], &[]);
            // Remove any cost perturbations and see if basis is still dual
            // feasible
            self.cleanup();
            if self.dual_infeas_count > 0 {
                self.solve_phase = PHASE_OPTIMAL_CLEANUP;
            } else {
                self.solve_phase = PHASE_OPTIMAL;
                self.dev(msg::PROBLEM_OPTIMAL, &[], &[]);
                x.model_status.set(MS_OPTIMAL);
            }
        } else if self.rebuild_reason == RR_CHOOSE_COLUMN_FAIL || self.rebuild_reason == RR_EXCESSIVE_PRIMAL_VALUE {
            self.solve_phase = PHASE_ERROR;
            self.report_ratio_test_or_excessive_primal();
            self.dev(msg::PHASE2_NOT_SOLVED, &[], &[]);
            x.model_status.set(MS_SOLVE_ERROR);
        } else {
            // Can only be that primal infeasibility has been detected
            self.dev(msg::PROBLEM_PRIMAL_INFEASIBLE, &[], &[]);
        }
    }

    fn rebuild(&mut self) {
        let x = self.x;
        // Clear taboo flag from any bad basis changes
        self.records().clear_taboo_flag();
        // Decide whether refactorization should be performed
        let refactor_basis_matrix = self.rebuild_refactor(self.rebuild_reason);
        // Take a local copy of the rebuild reason and then reset the
        // global value
        let local_rebuild_reason = self.rebuild_reason;
        self.rebuild_reason = RR_NO;
        if refactor_basis_matrix {
            // Get a nonsingular inverse if possible
            let ok = hekk::get_nonsingular_inverse(&mut self.e, x, self.solve_phase);
            self.refactor_info_dirty = true;
            if !ok {
                self.solve_phase = PHASE_ERROR;
                return;
            }
            self.reset_synthetic_clock();
        }
        if !x.has_ar_matrix.get() {
            // Don't have the row-wise matrix, so reinitialise it: should
            // only happen when backtracking
            hekk::initialise_partitioned_rowwise_matrix(&mut self.e, x);
        }
        // Record whether the update objective value should be tested
        let check_updated_objective_value = x.has_dual_objective_value.get();
        let previous_dual_objective_value =
            if check_updated_objective_value { *self.e.updated_dual_objective_value } else { -INF };
        // On the first rebuild of phase 2 after the set-up, without
        // refactorization, the dual values are fresh and, unless
        // correcting dual infeasibilities flips bounds, so are the primal
        // values
        let use_fresh = self.solve_phase == PHASE_2 && !refactor_basis_matrix;
        let fresh_primal = use_fresh && x.fresh_primal.get();
        if !(use_fresh && x.fresh_dual.get()) {
            self.compute_dual();
        }
        self.clear_fresh_values();
        if *self.e.backtracking {
            // If backtracking, may change phase, so drop out
            self.solve_phase = PHASE_UNKNOWN;
            return;
        }
        let num_flip = self.num_correct_dual_primal_flip;
        self.correct_dual_infeasibilities();
        // Recompute primal solution
        if !fresh_primal || self.num_correct_dual_primal_flip != num_flip {
            self.compute_primal();
        }
        // Collect primal infeasible as a list
        let num_row = self.num_row as i32;
        let primal = Self::primal(&mut self.e, x);
        self.rhs.create_array_of_primal_infeasibilities(&primal, num_row);
        let e = &mut self.e;
        self.rhs.create_infeas_list(*e.col_aq_density, e.dual_edge_weight, e.scattered_dual_edge_weight, num_row);
        // Dual objective section
        self.compute_dual_objective_value(self.solve_phase);
        if check_updated_objective_value {
            // Apply the objective value correction due to computing duals
            // from scratch
            let dual_objective_value_correction = *self.e.dual_objective_value - previous_dual_objective_value;
            *self.e.updated_dual_objective_value += dual_objective_value_correction;
        }
        // Now that there's a new dual_objective_value, reset the updated
        // value
        *self.e.updated_dual_objective_value = *self.e.dual_objective_value;
        if !x.run_quiet {
            self.compute_infeasibilities_for_reporting();
            self.report_rebuild(local_rebuild_reason);
        }
        // Record the synthetic clock for INVERT, and zero it for UPDATE
        self.reset_synthetic_clock();
        // Dual simplex doesn't maintain the number of primal
        // infeasibilities, so set it to an illegal value now
        *self.e.num_primal_infeasibilities = K_ILLEGAL_INFEASIBILITY_COUNT;
        *self.e.max_primal_infeasibility = INF;
        *self.e.sum_primal_infeasibilities = INF;
        // Although dual simplex should always be dual feasible,
        // infeasibilities are only corrected in rebuild
        *self.e.num_dual_infeasibilities = K_ILLEGAL_INFEASIBILITY_COUNT;
        *self.e.max_dual_infeasibility = INF;
        *self.e.sum_dual_infeasibilities = INF;
        // Data are fresh from rebuild
        x.has_fresh_rebuild.set(true);
    }

    /// HEkk::computeInfeasibilitiesForReporting for the dual simplex
    fn compute_infeasibilities_for_reporting(&mut self) {
        self.e.compute_simplex_primal_infeasible();
        if self.solve_phase == PHASE_1 {
            self.compute_simplex_lp_dual_infeasible();
        } else {
            self.e.compute_simplex_dual_infeasible();
        }
    }

    fn report_rebuild(&mut self, reason: i32) {
        self.record_analysis_data();
        self.x.dual_report(2, &self.analysis_data, reason, self.e.sense);
        self.analysis_data_pending = false;
    }

    fn cleanup(&mut self) {
        let x = self.x;
        if self.solve_phase == PHASE_1 {
            // Take action to prevent infinite loop
            x.dual_simplex_phase1_cleanup_level.set(x.dual_simplex_phase1_cleanup_level.get() + 1);
            if x.dual_simplex_phase1_cleanup_level.get() > x.max_dual_simplex_phase1_cleanup_level {
                self.dev(msg::CLEANUP_LEVEL_EXCEEDED, &[x.max_dual_simplex_phase1_cleanup_level], &[]);
            }
        }
        self.dev(msg::CLEANUP_SHIFT, &[], &[]);
        // Remove perturbation and don't permit further perturbation
        self.initialise_cost(false);
        x.allow_cost_perturbation.set(false);
        self.e.initialise_bound(ALGORITHM_DUAL, self.solve_phase, false);
        // Compute the dual values
        self.compute_dual();
        // Compute the dual infeasibilities
        self.e.compute_simplex_dual_infeasible();
        self.dual_infeas_count = *self.e.num_dual_infeasibilities;
        // Compute the dual objective value
        self.compute_dual_objective_value(self.solve_phase);
        *self.e.updated_dual_objective_value = *self.e.dual_objective_value;
        if !x.run_quiet {
            // Report the primal infeasibilities
            self.e.compute_simplex_primal_infeasible();
            // In phase 1, report the simplex LP dual infeasibilities
            if self.solve_phase == PHASE_1 {
                self.compute_simplex_lp_dual_infeasible();
            }
            self.report_rebuild(RR_CLEANUP);
        }
    }

    fn iterate(&mut self) {
        self.choose_row();
        self.choose_column();
        if self.is_bad_basis_change() {
            return;
        }
        self.update_ftran_bfrt();
        // Compute the pivotal column
        self.update_ftran();
        // The DSE FTRAN is performed on pi_p
        if self.edge_weight_mode == EW_DSE {
            self.update_ftran_dse();
        }
        // Check the row-wise pivot against the column-wise pivot for
        // numerical trouble
        self.update_verify();
        self.update_dual();
        // Update the primal values and the edge weights
        self.update_primal();
        // After primal update in dual simplex the primal objective value
        // is not known
        self.x.has_primal_objective_value.set(false);
        // Update the basis representation
        self.update_pivots();
        if self.new_devex_framework {
            self.initialise_devex_framework();
        }
        // Analyse the iteration: possibly report; possibly switch strategy
        self.iteration_analysis();
    }

    fn iteration_analysis(&mut self) {
        if self.x.iteration_report {
            self.record_analysis_data();
            self.x.dual_report(1, &self.analysis_data, 0, self.e.sense);
            self.analysis_data_pending = false;
        } else {
            self.record_analysis_data();
        }
        // Possibly switch from DSE to Devex
        if self.edge_weight_mode == EW_DSE && self.switch_to_devex() {
            self.edge_weight_mode = EW_DEVEX;
            self.initialise_devex_framework();
        }
    }

    /// HEkk::switchToDevex
    fn switch_to_devex(&mut self) -> bool {
        let x = self.x;
        // Parameters controlling switch from DSE to Devex on cost
        let k_costly_dse_measure_limit = 1000.0;
        let k_costly_dse_minimum_density = 0.01;
        let k_costly_dse_fraction_num_total_iteration_before_switch = 0.1;
        let k_costly_dse_fraction_num_costly_dse_iteration_before_switch = 0.05;
        let mut switch_to_devex = false;
        // Firstly consider switching on the basis of NLA cost
        let row_dse_density = *self.e.row_dse_density;
        let costly_dse_measure_denominator =
            std_max(std_max(*self.e.row_ep_density, *self.e.col_aq_density), *self.e.row_ap_density);
        if costly_dse_measure_denominator > 0.0 {
            let m = row_dse_density / costly_dse_measure_denominator;
            x.costly_dse_measure.set(m * m);
        } else {
            x.costly_dse_measure.set(0.0);
        }
        let costly_dse_iteration =
            x.costly_dse_measure.get() > k_costly_dse_measure_limit && row_dse_density > k_costly_dse_minimum_density;
        x.costly_dse_frequency.set((1.0 - K_RUNNING_AVERAGE_MULTIPLIER) * x.costly_dse_frequency.get());
        if costly_dse_iteration {
            x.num_costly_dse_iteration.set(x.num_costly_dse_iteration.get() + 1);
            x.costly_dse_frequency.set(x.costly_dse_frequency.get() + K_RUNNING_AVERAGE_MULTIPLIER * 1.0);
            let local_iteration_count = x.iteration_count.get() - x.control_iteration_count0;
            let local_num_tot = self.num_tot as i32;
            // Switch to Devex if at least 5% of the (at least) 0.1NumTot
            // iterations have been costly
            switch_to_devex = x.allow_dual_steepest_edge_to_devex_switch
                && (x.num_costly_dse_iteration.get() as f64
                    > local_iteration_count as f64 * k_costly_dse_fraction_num_costly_dse_iteration_before_switch)
                && (local_iteration_count as f64
                    > k_costly_dse_fraction_num_total_iteration_before_switch * local_num_tot as f64);
            if switch_to_devex {
                self.dev(
                    msg::SWITCH_DEVEX_COST,
                    &[x.num_costly_dse_iteration.get(), local_iteration_count],
                    &[*self.e.col_aq_density, *self.e.row_ep_density, *self.e.row_ap_density, row_dse_density],
                );
            }
        }
        if !switch_to_devex {
            // Secondly consider switching on the basis of weight accuracy
            let local_measure = x.average_log_low_dse_weight_error.get() + x.average_log_high_dse_weight_error.get();
            let local_threshold = x.dual_steepest_edge_weight_log_error_threshold;
            switch_to_devex = x.allow_dual_steepest_edge_to_devex_switch && local_measure > local_threshold;
            if switch_to_devex {
                self.dev(msg::SWITCH_DEVEX_ERROR, &[], &[local_measure, local_threshold]);
            }
        }
        switch_to_devex
    }

    fn choose_row(&mut self) {
        // If reinversion is needed then skip this method
        if self.rebuild_reason != 0 {
            return;
        }
        // Zero the infeasibility of any taboo rows
        {
            // SAFETY: as for records()
            let records = unsafe { &mut *self.x.basis_records };
            records.apply_taboo(&mut self.rhs.work_infeasibility, 0.0, 0);
        }
        let num_row = self.num_row as i32;
        loop {
            // Choose the index of a good row to leave the basis
            let mut random = HighsRandom::from_state(self.x.random.get());
            let e = &mut self.e;
            self.row_out =
                self.rhs.choose_normal(e.dual_edge_weight, e.scattered_dual_edge_weight, &mut random, num_row);
            self.x.random.set(random.state());
            if self.row_out == NO_ROW_CHOSEN {
                // No index found so may be dual optimal
                self.rebuild_reason = RR_POSSIBLY_OPTIMAL;
                return;
            }
            let row_out = self.row_out as usize;
            // Compute pi_p = B^{-T}e_p in row_ep
            let (updated_edge_weight, computed) = {
                with_vec(&mut self.v.row_ep, |row_ep| {
                    row_ep.clear();
                    row_ep.count = 1;
                    row_ep.index[0] = row_out as i32;
                    row_ep.array[row_out] = 1.0;
                    row_ep.pack_flag = true;
                    e.btran(row_ep, *e.row_ep_density);
                    if self.edge_weight_mode == EW_DSE {
                        // For DSE, see how accurate the updated weight is
                        let updated_edge_weight = e.dual_edge_weight[row_out];
                        // norm2 is inlined into chooseRow, contracted
                        let computed = if e.simplex_in_scaled_space {
                            row_ep.norm2_fused()
                        } else {
                            row_ep_2norm_in_scaled_space(e, row_out, row_ep)
                        };
                        (updated_edge_weight, computed)
                    } else {
                        (0.0, 0.0)
                    }
                })
            };
            if self.edge_weight_mode == EW_DSE {
                self.computed_edge_weight = computed;
                self.e.dual_edge_weight[row_out] = computed;
                // If the weight error is acceptable then break out of the
                // loop
                if self.accept_dual_steepest_edge_weight(updated_edge_weight) {
                    break;
                }
            } else {
                break;
            }
        }
        // Recover the infeasibility of any taboo rows
        {
            // SAFETY: as for records()
            let records = unsafe { &*self.x.basis_records };
            records.unapply_taboo(&mut self.rhs.work_infeasibility, 0);
        }
        let row_out = self.row_out as usize;
        self.variable_out = self.e.basic_index[row_out];
        let (value, lower, upper) = (self.e.base_value[row_out], self.e.base_lower[row_out], self.e.base_upper[row_out]);
        self.delta_primal = if value < lower { value - lower } else { value - upper };
        self.move_out = if self.delta_primal < 0.0 { -1 } else { 1 };
        // Update the record of average row_ep (pi_p) density
        let count = self.v.row_ep.count;
        let local_row_ep_density = count as f64 * self.inv_num_row;
        update_operation_result_density(local_row_ep_density, self.e.row_ep_density);
    }

    fn accept_dual_steepest_edge_weight(&mut self, updated_edge_weight: f64) -> bool {
        let accept_weight = updated_edge_weight >= K_ACCEPT_DSE_WEIGHT_THRESHOLD * self.computed_edge_weight;
        self.assess_dse_weight_error(self.computed_edge_weight, updated_edge_weight);
        accept_weight
    }

    /// HEkk::assessDSEWeightError
    fn assess_dse_weight_error(&mut self, computed_edge_weight: f64, updated_edge_weight: f64) {
        let x = self.x;
        let edge_weight_error = (updated_edge_weight - computed_edge_weight).abs() / std_max(1.0, computed_edge_weight);
        x.edge_weight_error.set(edge_weight_error);
        if edge_weight_error > x.dual_steepest_edge_weight_error_tolerance {
            self.dev(msg::DSE_WEIGHT_ERROR, &[], &[edge_weight_error]);
        }
        if updated_edge_weight < computed_edge_weight {
            // Updated weight is low
            let weight_relative_deviation = computed_edge_weight / updated_edge_weight;
            let a = x.average_log_low_dse_weight_error.get();
            x.average_log_low_dse_weight_error.set(0.99f64.mul_add_c(a, 0.01 * weight_relative_deviation.ln()));
        } else {
            // Updated weight is correct or high
            let weight_relative_deviation = updated_edge_weight / computed_edge_weight;
            let a = x.average_log_high_dse_weight_error.get();
            x.average_log_high_dse_weight_error.set(0.99f64.mul_add_c(a, 0.01 * weight_relative_deviation.ln()));
        }
    }

    fn new_devex_framework_check(&self, updated_edge_weight: f64) -> bool {
        let k_min_abs_number_devex_iterations = 25;
        let k_min_rlv_number_devex_iterations = 1e-2;
        let k_max_allowed_devex_weight_ratio = 3.0;
        let devex_ratio = std_max(
            updated_edge_weight / self.computed_edge_weight,
            self.computed_edge_weight / updated_edge_weight,
        );
        let mut i_te = (self.num_row as f64 / k_min_rlv_number_devex_iterations) as i32;
        i_te = k_min_abs_number_devex_iterations.max(i_te);
        // Square kMaxAllowedDevexWeightRatio due to keeping squared weights
        let accept_ratio_threshold = k_max_allowed_devex_weight_ratio * k_max_allowed_devex_weight_ratio;
        let accept_ratio = devex_ratio <= accept_ratio_threshold;
        let accept_it = self.num_devex_iterations <= i_te;
        !accept_ratio || !accept_it
    }

    fn row_matrix(&self) -> RowMatrix<'a> {
        // SAFETY: lp_.a_matrix_, unchanged during the solve
        unsafe {
            let a = &self.e.a;
            RowMatrix {
                start: std::slice::from_raw_parts(a.start.as_ptr(), a.start.len()),
                index: std::slice::from_raw_parts(a.index.as_ptr(), a.index.len()),
                value: std::slice::from_raw_parts(a.value.as_ptr(), a.value.len()),
            }
        }
    }

    /// HEkkDualRow::clear, then workDelta and createFreemove(row_ep)
    fn row_clear_and_create_freemove(&mut self) {
        self.pack_count = 0;
        self.work_count = 0;
        self.work_delta = self.delta_primal;
        let a = self.row_matrix();
        let (row, e, num_row) = (&self.row, &mut self.e, self.num_row);
        {
            with_vec(&mut self.v.row_ep, |row_ep| {
                row.create_freemove(*e.update_count, self.work_delta, &a, &row_ep.array[..num_row], e.nonbasic_move)
            })
        };
    }

    /// Pack row_ap and row_ep (chooseMakepack)
    fn row_makepack(&mut self) {
        let row = &mut self.row;
        let num_col = self.num_col as i32;
        self.pack_count = {
            let pc = with_vec(&mut self.v.row_ap, |ap| {
                row.choose_makepack(0, &ap.index[..ap.count as usize], ap.array, 0)
            });
            with_vec(&mut self.v.row_ep, |ep| {
                row.choose_makepack(pc, &ep.index[..ep.count as usize], ep.array, num_col)
            })
        };
    }

    fn choose_column(&mut self) {
        // If reinversion is needed then skip this method
        if self.rebuild_reason != 0 {
            return;
        }
        // PRICE
        let e = &mut self.e;
        with_vec(&mut self.v.row_ep, |ep| with_vec(&mut self.v.row_ap, |ap| e.tableau_row_price(ep, ap)));
        // CHUZC
        // Section 0: Clear data and call createFreemove to set a value of
        // nonbasicMove for all free columns to prevent their dual values
        // from being changed
        self.row_clear_and_create_freemove();
        // Section 1: Pack row_ap and row_ep
        self.row_makepack();
        let row_ep_scale = get_value_scale(&self.row.pack_value[..self.pack_count]);
        // Loop until an acceptable pivot is found
        let mut chuzc_pass = 0;
        let n = self.num_tot;
        loop {
            // Section 2: Determine the possible variables - candidates for
            // CHUZC
            let (count, theta) = self.row.choose_possible(
                self.pack_count,
                self.work_delta,
                *self.e.update_count,
                self.td,
                &self.e.nonbasic_move[..n],
                &self.e.work_dual[..n],
            );
            self.work_count = count;
            self.work_theta = theta;
            // Take action if the step to an expanded bound is not
            // positive, or there are no candidates for CHUZC
            self.variable_in = -1;
            if self.work_theta <= 0.0 || self.work_count == 0 {
                self.rebuild_reason = RR_POSSIBLY_DUAL_UNBOUNDED;
                return;
            }
            // Sections 3 and 4: Perform (bound-flipping) ratio test. This
            // can fail if the dual values are excessively large
            if self.choose_final() != 0 {
                self.rebuild_reason = RR_CHOOSE_COLUMN_FAIL;
                return;
            }
            if self.work_pivot >= 0 {
                let growth_tolerance = self.x.dual_simplex_pivot_growth_tolerance;
                // A pivot has been chosen
                let alpha_row = self.work_alpha;
                let scaled_value = row_ep_scale * alpha_row;
                if scaled_value.abs() <= growth_tolerance {
                    // On the first pass, try to make the pivotal row more
                    // accurate
                    if chuzc_pass == 0 {
                        self.improve_choose_column_row();
                    } else {
                        // Remove the pivot
                        let pc = self.pack_count;
                        let (index, value) = (&mut self.row.pack_index, &mut self.row.pack_value);
                        if let Some(i) = index[..pc].iter().position(|&j| j == self.work_pivot) {
                            index[i] = index[pc - 1];
                            value[i] = value[pc - 1];
                            self.pack_count -= 1;
                        }
                    }
                    // Indicate that no pivot has been chosen
                    self.work_pivot = -1;
                }
            } else {
                // No pivot has been chosen
                break;
            }
            // If a pivot has been chosen, or there are no more packed
            // values then end CHUZC
            if self.work_pivot >= 0 || self.pack_count == 0 {
                break;
            }
            chuzc_pass += 1;
        }
        // Section 5: Reset the nonbasicMove values for free columns
        self.row.delete_freemove(self.e.nonbasic_move);
        // Record values for basis change
        self.variable_in = self.work_pivot;
        self.alpha_row = self.work_alpha;
        self.theta_dual = self.work_theta;
        if self.edge_weight_mode == EW_DEVEX && !self.new_devex_framework {
            // Determine the exact Devex weight
            let w = self.row.compute_devex_weight(self.pack_count, &self.e.nonbasic_flag[..n], self.devex_index);
            self.computed_edge_weight = std_max(1.0, w);
        }
    }

    /// HEkkDualRow::chooseFinal: 0 on success, -1 on failure
    fn choose_final(&mut self) -> i32 {
        let n = self.num_tot;
        let (mv, dual, range) = (&self.e.nonbasic_move[..n], &self.e.work_dual[..n], &self.e.work_range[..n]);
        // 1. Reduce by large step BFRT
        self.work_count = self.row.choose_final_reduce(self.work_count, self.work_theta, self.work_delta, mv, dual, range);
        // 2-4. Choose by small step BFRT, then by large alpha, and
        // determine the BFRT flips
        match self.row.choose_final(
            self.work_count,
            self.work_theta,
            self.work_delta,
            self.td,
            mv,
            dual,
            range,
            self.num_tot_permutation,
        ) {
            Ok(c) => {
                self.work_count = c.work_count;
                self.work_theta = c.work_theta;
                self.work_pivot = c.work_pivot;
                self.work_alpha = c.work_alpha;
                0
            }
            Err(fail) => {
                let (kind, work_count, select_theta, remain_theta) = match fail {
                    ChooseFail::NoChange { work_count, select_theta, remain_theta } => {
                        (1, work_count, select_theta, remain_theta)
                    }
                    ChooseFail::NoGroup { work_count, select_theta } => (2, work_count, select_theta, 0.0),
                };
                self.work_count = work_count;
                if self.x.dev_log {
                    (self.x.host.chuzc_fail)(
                        self.x.host.ctx,
                        kind,
                        work_count as i32,
                        self.row.work_data.as_ptr(),
                        select_theta,
                        remain_theta,
                    );
                }
                -1
            }
        }
    }

    fn improve_choose_column_row(&mut self) {
        self.row.delete_freemove(self.e.nonbasic_move);
        // Refine row_ep, and compute row_ap in quad precision
        let (e, row_out) = (&mut self.e, self.row_out as usize);
        let (row_ep, row_ap) = (&mut self.v.row_ep, &mut self.v.row_ap);
        row_ep.with(|ep| {
            hekk::unit_btran_iterative_refinement(e, row_out, ep);
            row_ap.with(|ap| hekk::tableau_row_price_quad(e, ep, ap));
        });
        self.row_clear_and_create_freemove();
        self.row_makepack();
    }

    fn is_bad_basis_change(&mut self) -> bool {
        if self.rebuild_reason != 0 {
            return false;
        }
        if self.variable_in == -1 || self.row_out == -1 {
            return false;
        }
        let x = self.x;
        let mut currhash = *self.e.basis_hash;
        let variable_out = self.e.basic_index[self.row_out as usize];
        crate::util::hash::sparse_inverse_combine_index(&mut currhash, variable_out);
        crate::util::hash::sparse_combine_index(&mut currhash, self.variable_in);
        let mut cycling_detected = false;
        if self.records().visited.contains(&currhash) {
            if x.iteration_count.get() == x.previous_iteration_cycling_detected.get().wrapping_add(1) {
                // Cycling detected on successive iterations suggests
                // infinite cycling
                cycling_detected = true;
            } else {
                x.previous_iteration_cycling_detected.set(x.iteration_count.get());
            }
        }
        let (row_out, variable_in) = (self.row_out, self.variable_in);
        if cycling_detected {
            self.dev(msg::BAD_BASIS_CHANGE, &[variable_out, variable_in], &[]);
            self.records().add_bad_basis_change(row_out, variable_out, variable_in, REASON_CYCLING, true);
            true
        } else {
            // Look to see whether this basis change is in the list of bad
            // ones
            self.records().find_and_make_taboo(row_out, variable_out, variable_in)
        }
    }

    fn update_ftran(&mut self) {
        if self.rebuild_reason != 0 {
            return;
        }
        let (e, variable_in, row_out, inv) = (&mut self.e, self.variable_in as usize, self.row_out as usize, self.inv_num_row);
        self.alpha_col = {
            with_vec(&mut self.v.col_aq, |col_aq| {
                // Clear the pivotal column and indicate that its values
                // should be packed
                col_aq.clear();
                col_aq.pack_flag = true;
                e.a.collect_aj(col_aq, variable_in, 1.0);
                e.ftran(col_aq, *e.col_aq_density);
                let local_col_aq_density = col_aq.count as f64 * inv;
                update_operation_result_density(local_col_aq_density, e.col_aq_density);
                col_aq.array[row_out]
            })
        };
    }

    fn update_ftran_bfrt(&mut self) {
        if self.rebuild_reason != 0 {
            return;
        }
        let a = self.row_matrix();
        let (row, e, x, work_count, inv) = (&self.row, &mut self.e, self.x, self.work_count, self.inv_num_row);
        {
            with_vec(&mut self.v.col_bfrt, |col| {
                col.clear();
                let mut count = col.count as usize;
                let n = e.num_row + e.num_col;
                *e.updated_dual_objective_value += row.update_flip(
                    work_count,
                    &a,
                    &e.work_dual[..n],
                    e.cost_scale,
                    &mut e.nonbasic_move[..n],
                    &mut e.work_value[..n],
                    &e.work_lower[..n],
                    &e.work_upper[..n],
                    col.array,
                    col.index,
                    &mut count,
                );
                col.count = count as i32;
                if col.count != 0 {
                    e.ftran(col, x.col_bfrt_density.get());
                }
                let local_col_bfrt_density = col.count as f64 * inv;
                let mut density = x.col_bfrt_density.get();
                update_operation_result_density(local_col_bfrt_density, &mut density);
                x.col_bfrt_density.set(density);
            })
        };
    }

    /// The FTRAN for the DSE update, applied to row_ep
    fn update_ftran_dse(&mut self) {
        if self.rebuild_reason != 0 {
            return;
        }
        let (e, inv) = (&mut self.e, self.inv_num_row);
        {
            with_vec(&mut self.v.row_ep, |v| {
                // Apply R^{-1}: HSimplexNla::unapplyBasisMatrixRowScale
                if let Some((_, row_scale)) = e.scale {
                    let (use_row_indices, to_entry) = sparse_loop_style(v.count, e.num_row);
                    for i_entry in 0..to_entry {
                        let i_row = if use_row_indices { v.index[i_entry] as usize } else { i_entry };
                        v.array[i_row] /= row_scale[i_row];
                    }
                }
                // Perform FTRAN DSE
                e.factor.ftran(v, *e.row_dse_density);
                let local_row_dse_density = v.count as f64 * inv;
                update_operation_result_density(local_row_dse_density, e.row_dse_density);
            })
        };
    }

    fn update_verify(&mut self) {
        if self.rebuild_reason != 0 {
            return;
        }
        let trouble = reinvert_on_numerical_trouble(
            self.alpha_col,
            self.alpha_row,
            K_NUMERICAL_TROUBLE_TOLERANCE,
            *self.e.update_count,
            self.x.factor_pivot_threshold.get(),
        );
        self.numerical_trouble = trouble.measure;
        if trouble.new_pivot_threshold != 0.0 {
            hekk::set_pivot_threshold(self.x, trouble.new_pivot_threshold);
        }
        if trouble.reinvert {
            self.rebuild_reason = RR_POSSIBLY_SINGULAR_BASIS;
        }
    }

    fn update_dual(&mut self) {
        if self.rebuild_reason != 0 {
            return;
        }
        let variable_in = self.variable_in as usize;
        let variable_out = self.variable_out as usize;
        if self.theta_dual == 0.0 {
            // Little to do if theta_dual is zero
            self.shift_cost(variable_in, -self.e.work_dual[variable_in]);
        } else {
            // Update the whole vector of dual values
            let e = &mut self.e;
            let n = e.num_row + e.num_col;
            *e.updated_dual_objective_value += self.row.update_dual(
                self.pack_count,
                self.theta_dual,
                &mut e.work_dual[..n],
                &e.work_value[..n],
                &e.nonbasic_flag[..n],
                e.cost_scale,
            );
        }
        // Identify the changes in the dual objective
        let e = &mut self.e;
        let variable_in_delta_dual = e.work_dual[variable_in];
        let variable_in_value = e.work_value[variable_in];
        let variable_in_nonbasic_flag = e.nonbasic_flag[variable_in] as i32;
        let mut dual_objective_value_change =
            variable_in_nonbasic_flag as f64 * (-variable_in_value * variable_in_delta_dual);
        dual_objective_value_change *= e.cost_scale;
        *e.updated_dual_objective_value += dual_objective_value_change;
        // Surely variable_out_nonbasicFlag is always 0 since it's basic -
        // so there's no dual objective change
        let variable_out_nonbasic_flag = e.nonbasic_flag[variable_out] as i32;
        if variable_out_nonbasic_flag != 0 {
            let variable_out_delta_dual = e.work_dual[variable_out] - self.theta_dual;
            let variable_out_value = e.work_value[variable_out];
            let mut change = variable_out_nonbasic_flag as f64 * (-variable_out_value * variable_out_delta_dual);
            change *= e.cost_scale;
            *e.updated_dual_objective_value += change;
        }
        e.work_dual[variable_in] = 0.0;
        e.work_dual[variable_out] = -self.theta_dual;
        self.shift_back(variable_out);
    }

    fn shift_cost(&mut self, i_col: usize, amount: f64) {
        *self.e.costs_shifted = true;
        self.x.dual_values_valid.set(false);
        if amount == 0.0 {
            return;
        }
        self.e.work_shift[i_col] = amount;
    }

    fn shift_back(&mut self, i_col: usize) {
        let e = &mut self.e;
        if e.work_shift[i_col] == 0.0 {
            return;
        }
        e.work_dual[i_col] -= e.work_shift[i_col];
        e.work_shift[i_col] = 0.0;
    }

    fn update_primal(&mut self) {
        if self.rebuild_reason != 0 {
            return;
        }
        let x = self.x;
        let row_out = self.row_out as usize;
        if self.edge_weight_mode == EW_DEVEX {
            let updated_edge_weight = self.e.dual_edge_weight[row_out];
            self.e.dual_edge_weight[row_out] = self.computed_edge_weight;
            self.new_devex_framework = self.new_devex_framework_check(updated_edge_weight);
        }
        // Update - primal and weight
        let num_row = self.num_row as i32;
        let mut primal = Self::primal(&mut self.e, x);
        let rhs = &mut self.rhs;
        {
            with_vec(&mut self.v.col_bfrt, |col| {
                rhs.update_primal(col.count, col.index, col.array, 1.0, &mut primal, num_row);
            })
        };
        let edge_weight = &*self.e.dual_edge_weight;
        {
            with_vec(&mut self.v.col_bfrt, |col| {
                rhs.update_infeas_list(&col.index[..col.count.max(0) as usize], edge_weight)
            })
        };
        let x_out = self.e.base_value[row_out];
        let l_out = self.e.base_lower[row_out];
        let u_out = self.e.base_upper[row_out];
        self.theta_primal = (x_out - if self.delta_primal < 0.0 { l_out } else { u_out }) / self.alpha_col;
        let theta_primal = self.theta_primal;
        let mut primal = Self::primal(&mut self.e, x);
        let rhs = &mut self.rhs;
        let ok_update_primal = {
            with_vec(&mut self.v.col_aq, |col| {
                rhs.update_primal(col.count, col.index, col.array, theta_primal, &mut primal, num_row)
            })
        };
        if !ok_update_primal {
            self.rebuild_reason = RR_EXCESSIVE_PRIMAL_VALUE;
            return;
        }
        let tp = self.e.primal_feasibility_tolerance;
        // SAFETY: as for records(); HEkkDual's col_aq
        unsafe {
            let records = &mut *self.x.basis_records;
            with_vec(&mut self.v.col_aq, |col| records.update_bad_basis_change(col.array, theta_primal, tp));
        }
        let variable_in = self.variable_in as usize;
        if self.edge_weight_mode == EW_DSE {
            let e = &mut self.e;
            let new_pivotal_edge_weight = {
                with_vec(&mut self.v.col_aq, |col_aq| {
                    with_vec(&mut self.v.row_ep, |row_ep| {
                        // HSimplexNla::pivotInScaledSpace
                        let pivot_in_scaled_space = col_aq.array[row_out] * e.variable_scale_factor(variable_in)
                            / e.variable_scale_factor(e.basic_index[row_out] as usize);
                        let new_pivotal_edge_weight =
                            e.dual_edge_weight[row_out] / (pivot_in_scaled_space * pivot_in_scaled_space);
                        let kai = -2.0 / pivot_in_scaled_space;
                        e.update_dual_steepest_edge_weights(
                            row_out,
                            variable_in,
                            col_aq,
                            new_pivotal_edge_weight,
                            kai,
                            row_ep.array,
                        );
                        new_pivotal_edge_weight
                    })
                })
            };
            self.e.dual_edge_weight[row_out] = new_pivotal_edge_weight;
        } else if self.edge_weight_mode == EW_DEVEX {
            // Pivotal row is for the current basis: weights are required
            // for the next basis so have to divide the current (exact)
            // weight by the pivotal value
            let alpha_col = self.alpha_col;
            let e = &mut self.e;
            let new_pivotal_edge_weight = std_max(1.0, e.dual_edge_weight[row_out] / (alpha_col * alpha_col));
            with_vec(&mut self.v.col_aq, |col_aq| e.update_dual_devex_weights(col_aq, new_pivotal_edge_weight));
            self.e.dual_edge_weight[row_out] = new_pivotal_edge_weight;
            self.num_devex_iterations += 1;
        }
        let edge_weight = &*self.e.dual_edge_weight;
        let rhs = &mut self.rhs;
        let ticks = {
            let aq_tick = with_vec(&mut self.v.col_aq, |col| {
                rhs.update_infeas_list(&col.index[..col.count.max(0) as usize], edge_weight);
                col.synthetic_tick
            });
            (aq_tick, self.v.row_ep.synthetic_tick)
        };
        // Add in the synthetic ticks of col_aq and of row_ep, which
        // contains the contribution from forming row_ep = B^{-T}e_p
        *self.e.total_synthetic_tick += ticks.0;
        *self.e.total_synthetic_tick += ticks.1;
    }

    fn update_pivots(&mut self) {
        if self.rebuild_reason != 0 {
            return;
        }
        let x = self.x;
        let (variable_in, row_out) = (self.variable_in as usize, self.row_out as usize);
        let e = &mut self.e;
        // Transform the vectors used in updateFactor if the simplex NLA
        // involves scaling
        if e.scale.is_some() {
            {
                with_vec(&mut self.v.col_aq, |aq| {
                    with_vec(&mut self.v.row_ep, |ep| e.transform_for_update(aq, ep, variable_in, row_out))
                })
            };
        }
        // HEkk::updatePivots
        x.dual_values_valid.set(false);
        e.update_pivots(variable_in, row_out, self.move_out);
        let hash = *e.basis_hash;
        self.records().visited.insert(hash);
        x.has_invert.set(false);
        x.has_fresh_invert.set(false);
        x.has_fresh_rebuild.set(false);
        x.iteration_count.set(x.iteration_count.get() + 1);
        // HEkk::updateFactor: the update invalidates the refactorization
        // information of the INVERT
        if self.refactor_info_dirty {
            self.e.factor.refactor_info_clear();
            self.refactor_info_dirty = false;
        }
        let (e, rebuild_reason) = (&mut self.e, &mut self.rebuild_reason);
        {
            with_vec(&mut self.v.col_aq, |aq| {
                with_vec(&mut self.v.row_ep, |ep| e.update_factor(aq, ep, row_out as i32, rebuild_reason))
            })
        };
        x.has_invert.set(true);
        // Update the row-wise representation of the nonbasic columns
        self.e.update_matrix(variable_in, self.variable_out as usize);
        // Delete Freelist entry for variable_in
        self.row.delete_freelist(self.variable_in);
        // Update the primal value for the row where the basis change has
        // occurred, and set the corresponding primal infeasibility value
        let value = self.e.work_value[variable_in] + self.theta_primal;
        let mut primal = Self::primal(&mut self.e, x);
        self.rhs.update_pivots(self.row_out, value, &mut primal);
    }

    fn initialise_devex_framework(&mut self) {
        // The devex reference set is initialised to be the current set of
        // basic variables
        for (d, &f) in self.devex_index[..self.num_tot].iter_mut().zip(&self.e.nonbasic_flag[..self.num_tot]) {
            *d = 1 - (f as i32) * (f as i32);
        }
        // Set all initial weights to 1
        hekk::assign_unit_dual_edge_weights(&mut self.e);
        self.num_devex_iterations = 0;
        self.new_devex_framework = false;
    }

    fn compute_dual_infeasibilities_with_fixed_variable_flips(&mut self) {
        let mut num_dual_infeasibility = 0;
        let mut max_dual_infeasibility = 0.0;
        let mut sum_dual_infeasibility = 0.0;
        let e = &mut self.e;
        for i_var in 0..self.num_tot {
            if e.nonbasic_flag[i_var] == 0 {
                continue;
            }
            // Nonbasic column
            let lower = e.work_lower[i_var];
            let upper = e.work_upper[i_var];
            let dual = e.work_dual[i_var];
            let dual_infeasibility = if lower == -INF && upper == INF {
                dual.abs()
            } else {
                -(e.nonbasic_move[i_var] as i32) as f64 * dual
            };
            if dual_infeasibility > 0.0 {
                if dual_infeasibility >= e.dual_feasibility_tolerance {
                    num_dual_infeasibility += 1;
                }
                max_dual_infeasibility = std_max(dual_infeasibility, max_dual_infeasibility);
                sum_dual_infeasibility += dual_infeasibility;
            }
        }
        *e.num_dual_infeasibilities = num_dual_infeasibility;
        *e.max_dual_infeasibility = max_dual_infeasibility;
        *e.sum_dual_infeasibilities = sum_dual_infeasibility;
    }

    /// Updates dual_infeas_count (free_infeasibility_count in the C++)
    fn correct_dual_infeasibilities(&mut self) {
        let x = self.x;
        let mut free_infeasibility_count = 0;
        let dual_feasibility_tolerance = self.e.dual_feasibility_tolerance;
        let mut flip_dual_objective_value_change = 0.0;
        let mut shift_dual_objective_value_change = 0.0;
        let mut num_flip = 0;
        let mut num_shift = 0;
        let mut sum_flip = 0.0;
        let mut sum_shift = 0.0;
        let mut max_flip = 0.0;
        let mut max_shift = 0.0;
        let mut min_dual_infeasibility_for_flip = INF;
        let mut max_dual_infeasibility_for_flip = 0.0;
        let mut num_dual_infeasibilities_for_flip = 0;
        let mut sum_dual_infeasibilities_for_flip = 0.0;
        let mut num_dual_infeasibilities_for_shift = 0;
        let mut max_dual_infeasibility_for_shift = 0.0;
        let mut sum_dual_infeasibilities_for_shift = 0.0;
        let mut random = HighsRandom::from_state(x.random.get());
        for i_var in 0..self.num_tot {
            let e = &mut self.e;
            if e.nonbasic_flag[i_var] == 0 {
                continue;
            }
            // Nonbasic column
            let lower = e.work_lower[i_var];
            let upper = e.work_upper[i_var];
            let current_dual = e.work_dual[i_var];
            let mv = e.nonbasic_move[i_var] as i32;
            let fixed = lower == upper;
            let boxed = lower > -INF && upper < INF;
            let free = lower == -INF && upper == INF;
            if free {
                if current_dual.abs() >= dual_feasibility_tolerance {
                    free_infeasibility_count += 1;
                }
                continue;
            }
            let dual_infeasibility = -mv as f64 * current_dual;
            if dual_infeasibility < dual_feasibility_tolerance {
                continue;
            }
            // There is a dual infeasibility to remove
            if fixed || (boxed && !self.force_phase2) {
                // Flip for fixed variables and boxed variables when not
                // forcing phase 2: HEkk::flipBound
                let new_move = -e.nonbasic_move[i_var];
                e.nonbasic_move[i_var] = new_move;
                e.work_value[i_var] = if new_move == 1 { e.work_lower[i_var] } else { e.work_upper[i_var] };
                let flip = upper - lower;
                let mut local_dual_objective_change = mv as f64 * flip * current_dual;
                local_dual_objective_change *= e.cost_scale;
                flip_dual_objective_value_change += local_dual_objective_change;
                num_flip += 1;
                max_flip = std_max(flip.abs(), max_flip);
                sum_flip += flip.abs();
                // Flipping fixed variables is trivial, so only track the
                // infeasibilities involved when flipping boxed variables
                if !fixed {
                    min_dual_infeasibility_for_flip = std_min(dual_infeasibility, min_dual_infeasibility_for_flip);
                    if dual_infeasibility >= dual_feasibility_tolerance {
                        num_dual_infeasibilities_for_flip += 1;
                    }
                    sum_dual_infeasibilities_for_flip += dual_infeasibility;
                    max_dual_infeasibility_for_flip = std_max(dual_infeasibility, max_dual_infeasibility_for_flip);
                }
                continue;
            }
            // Either boxed but not fixed, of one-sided, so shift
            if dual_infeasibility >= dual_feasibility_tolerance {
                num_dual_infeasibilities_for_shift += 1;
            }
            sum_dual_infeasibilities_for_shift += dual_infeasibility;
            max_dual_infeasibility_for_shift = std_max(dual_infeasibility, max_dual_infeasibility_for_shift);
            *e.costs_shifted = true;
            x.dual_values_valid.set(false);
            let new_dual = if mv == 1 {
                (1.0 + random.fraction()) * dual_feasibility_tolerance
            } else {
                -(1.0 + random.fraction()) * dual_feasibility_tolerance
            };
            let shift = new_dual - current_dual;
            e.work_dual[i_var] = new_dual;
            e.work_cost[i_var] += shift;
            let mut local_dual_objective_change = shift * e.work_value[i_var];
            local_dual_objective_change *= e.cost_scale;
            shift_dual_objective_value_change += local_dual_objective_change;
            num_shift += 1;
            max_shift = std_max(shift.abs(), max_shift);
            sum_shift += shift.abs();
            if self.x.dev_log {
                x.random.set(random.state());
                self.dev(msg::SHIFT, &[(mv == 1) as i32], &[shift, local_dual_objective_change]);
            }
        }
        x.random.set(random.state());
        self.num_correct_dual_primal_flip += num_flip;
        if num_flip != 0 && self.force_phase2 {
            self.dev(
                msg::FLIPS,
                &[num_flip, num_dual_infeasibilities_for_flip],
                &[
                    max_flip,
                    sum_flip,
                    min_dual_infeasibility_for_flip,
                    max_dual_infeasibility_for_flip,
                    sum_dual_infeasibilities_for_flip,
                    flip_dual_objective_value_change,
                ],
            );
        }
        if num_shift != 0 {
            self.dev(
                msg::SHIFTS,
                &[num_shift, num_dual_infeasibilities_for_shift],
                &[
                    max_shift,
                    sum_shift,
                    max_dual_infeasibility_for_shift,
                    sum_dual_infeasibilities_for_shift,
                    shift_dual_objective_value_change,
                ],
            );
        }
        self.force_phase2 = false;
        self.dual_infeas_count = free_infeasibility_count;
    }

    fn assess_phase1_optimality(&mut self) {
        // There are (possibly insignificant) LP dual infeasibilities that
        // can't be removed by dual Phase 1, so clean up any perturbations
        // before concluding dual infeasibility
        self.dev(
            msg::PHASE1_OPTIMAL_NOT_PHASE2,
            &[*self.e.costs_perturbed as i32],
            &[*self.e.dual_objective_value],
        );
        if *self.e.costs_perturbed {
            // Clean up perturbation
            self.cleanup();
        }
        self.assess_phase1_optimality_unperturbed();
        if self.dual_infeas_count <= 0 && self.solve_phase == PHASE_2 {
            // Reset the duals, if necessary shifting costs of free
            // variables so that their duals are zero
            self.exit_phase1_reset_duals();
        }
    }

    fn assess_phase1_optimality_unperturbed(&mut self) {
        if self.dual_infeas_count == 0 {
            // No dual infeasibilities with respect to phase 1 bounds
            if *self.e.dual_objective_value == 0.0 {
                // No dual infeasibilities with respect to phase 2 bounds so
                // go to phase 2
                self.dev(msg::PHASE1_GO_PHASE2, &[], &[]);
                self.solve_phase = PHASE_2;
            } else {
                // Nonzero dual objective value: could be insignificant dual
                // infeasibilities
                self.dev(msg::PHASE1_FEASIBLE_WRT_PHASE1, &[], &[*self.e.dual_objective_value]);
                self.compute_simplex_lp_dual_infeasible();
                if self.lp_dual_infeasibility.num == 0 {
                    self.dev(msg::PHASE1_GO_PHASE2, &[], &[]);
                    self.solve_phase = PHASE_2;
                } else {
                    // LP is dual infeasible if the dual objective is
                    // sufficiently negative, so no conclusions on the
                    // primal LP can be deduced
                    let l = &self.lp_dual_infeasibility;
                    self.dev(
                        msg::POSSIBLE_LP_DUAL_INFEASIBILITY,
                        &[l.num],
                        &[*self.e.dual_objective_value, l.max, l.sum],
                    );
                    self.x.model_status.set(MS_UNBOUNDED_OR_INFEASIBLE);
                    self.solve_phase = PHASE_EXIT;
                }
            }
        } else {
            self.dev(msg::PHASE1_RETURN, &[self.dual_infeas_count], &[]);
        }
    }

    fn exit_phase1_reset_duals(&mut self) {
        if *self.e.costs_perturbed {
            self.dev(msg::ALREADY_PERTURBED, &[], &[]);
        } else {
            self.dev(msg::REPERTURBING, &[], &[]);
            self.initialise_cost(true);
            self.compute_dual();
        }
        let mut num_shift = 0;
        let mut sum_shift = 0.0;
        for i_var in 0..self.num_tot {
            let e = &mut self.e;
            if e.nonbasic_flag[i_var] == 0 {
                continue;
            }
            let (lp_lower, lp_upper) = if i_var < self.num_col {
                (e.col_lower[i_var], e.col_upper[i_var])
            } else {
                let i_row = i_var - self.num_col;
                (e.row_lower[i_row], e.row_upper[i_row])
            };
            if lp_lower <= -INF && lp_upper >= INF {
                let shift = -e.work_dual[i_var];
                e.work_dual[i_var] = 0.0;
                e.work_cost[i_var] += shift;
                num_shift += 1;
                sum_shift += shift.abs();
                self.dev(msg::FREE_SHIFT, &[i_var as i32], &[shift]);
            }
        }
        if num_shift != 0 {
            self.dev(msg::FREE_SHIFTS, &[num_shift], &[sum_shift]);
            *self.e.costs_shifted = true;
            self.x.dual_values_valid.set(false);
        }
    }

    fn bailout_on_dual_objective(&mut self) -> bool {
        let x = self.x;
        if x.solve_bailout.get() {
        } else if self.e.sense == 1 && self.solve_phase == PHASE_2 {
            if *self.e.updated_dual_objective_value > x.objective_bound {
                let reached = self.reached_exact_objective_bound();
                x.solve_bailout.set(reached);
            }
        }
        x.solve_bailout.get()
    }

    fn reached_exact_objective_bound(&mut self) -> bool {
        let x = self.x;
        let mut reached_exact_objective_bound = false;
        let use_row_ap_density = std_min(std_max(*self.e.row_ap_density, 0.01), 1.0);
        let check_frequency = (1.0 / use_row_ap_density) as i32;
        let check_exact_dual_objective_value = *self.e.update_count % check_frequency == 0;
        if check_exact_dual_objective_value {
            let objective_bound = x.objective_bound;
            let perturbed_dual_objective_value = *self.e.updated_dual_objective_value;
            let perturbed_value_residual = perturbed_dual_objective_value - objective_bound;
            let mut dual_col = OwnedHVec::new(self.num_row as i32);
            let mut dual_row = OwnedHVec::new(self.num_col as i32);
            let exact_dual_objective_value = self.compute_exact_dual_objective_value(&mut dual_col, &mut dual_row);
            let exact_value_residual = exact_dual_objective_value - objective_bound;
            let have;
            if exact_dual_objective_value > objective_bound {
                self.dev(msg::OBJECTIVE_BOUND_EXCEEDED, &[], &[*self.e.updated_dual_objective_value, objective_bound]);
                have = 1;
                if *self.e.costs_perturbed || *self.e.costs_shifted {
                    // Remove cost perturbation/shifting
                    self.initialise_cost(false);
                }
                // Set the duals as computed in the
                // computeExactDualObjectiveValue call
                let e = &mut self.e;
                for i in 0..self.num_col {
                    e.work_dual[i] = e.work_cost[i] - dual_row.array[i];
                }
                for i in self.num_col..self.num_tot {
                    e.work_dual[i] = -dual_col.array[i - self.num_col];
                }
                // Since the computeExactDualObjectiveValue() call
                // succeeded, if there are any dual infeasibilities they
                // can be removed by a bound flip
                self.force_phase2 = false;
                self.correct_dual_infeasibilities();
                reached_exact_objective_bound = true;
                x.model_status.set(MS_OBJECTIVE_BOUND);
            } else {
                have = 0;
            }
            self.dev(
                msg::DUAL_UB_BAILOUT,
                &[have, x.iteration_count.get(), check_frequency],
                &[use_row_ap_density, perturbed_value_residual, exact_value_residual],
            );
        }
        reached_exact_objective_bound
    }

    fn compute_exact_dual_objective_value(&mut self, dual_col: &mut OwnedHVec, dual_row: &mut OwnedHVec) -> f64 {
        let (num_row, num_col) = (self.num_row, self.num_col);
        // Create a local buffer for the pi vector
        {
            let e = &self.e;
            for i_row in 0..num_row {
                let i_var = e.basic_index[i_row] as usize;
                if i_var < num_col {
                    let value = e.col_cost[i_var];
                    if value != 0.0 {
                        dual_col.array[i_row] = value;
                        dual_col.index[dual_col.count as usize] = i_row as i32;
                        dual_col.count += 1;
                    }
                }
            }
        }
        if dual_col.count != 0 {
            let e = &mut self.e;
            dual_col.with(|c| e.btran(c, 1.0));
            let a = &self.e.a;
            dual_row.count = matrix::price_by_column(
                a.start,
                a.index,
                a.value,
                &dual_col.array,
                &mut dual_row.array,
                &mut dual_row.index,
            ) as i32;
        }
        // Compute dual infeasibilities
        self.e.compute_simplex_dual_infeasible();
        if *self.e.num_dual_infeasibilities > 0 {
            self.dev(
                msg::EXACT_DUAL_INFEASIBILITIES,
                &[*self.e.num_dual_infeasibilities],
                &[*self.e.max_dual_infeasibility, *self.e.sum_dual_infeasibilities],
            );
        }
        let small_matrix_value = self.x.small_matrix_value;
        let mut dual_objective = CDouble::from(self.e.offset);
        let mut norm_dual = 0.0;
        let mut norm_delta_dual = 0.0;
        for i_col in 0..num_col {
            let e = &self.e;
            if e.nonbasic_flag[i_col] == 0 {
                continue;
            }
            let exact_dual = e.col_cost[i_col] - dual_row.array[i_col];
            let active_value = if exact_dual > small_matrix_value {
                e.col_lower[i_col]
            } else if exact_dual < -small_matrix_value {
                e.col_upper[i_col]
            } else {
                e.work_value[i_col]
            };
            // when the active value is infinite the dual objective lower
            // bound is -infinity
            if active_value.abs() >= INF {
                return -INF;
            }
            let residual = (exact_dual - e.work_dual[i_col]).abs();
            norm_dual += exact_dual.abs();
            norm_delta_dual += residual;
            if residual > 1e10 {
                self.dev(msg::EXACT_COL_RESIDUAL, &[i_col as i32], &[exact_dual, e.work_dual[i_col], residual]);
            }
            dual_objective += active_value * exact_dual;
        }
        for i_var in num_col..num_col + num_row {
            let e = &self.e;
            if e.nonbasic_flag[i_var] == 0 {
                continue;
            }
            let i_row = i_var - num_col;
            let exact_dual = dual_col.array[i_row];
            let active_value = if exact_dual > small_matrix_value {
                e.row_lower[i_row]
            } else if exact_dual < -small_matrix_value {
                e.row_upper[i_row]
            } else {
                -e.work_value[i_var]
            };
            if active_value.abs() >= INF {
                return -INF;
            }
            let residual = (exact_dual + e.work_dual[i_var]).abs();
            norm_dual += exact_dual.abs();
            norm_delta_dual += residual;
            if residual > 1e10 {
                self.dev(msg::EXACT_ROW_RESIDUAL, &[i_row as i32], &[exact_dual, e.work_dual[i_var], residual]);
            }
            dual_objective += active_value * exact_dual;
        }
        let relative_delta = norm_delta_dual / std_max(norm_dual, 1.0);
        if relative_delta > 1e-3 {
            self.dev(msg::EXACT_RELATIVE_DELTA, &[], &[norm_dual, norm_delta_dual, relative_delta]);
        }
        dual_objective.to_f64()
    }

    fn assess_possibly_dual_unbounded(&mut self) {
        if self.solve_phase != PHASE_2 {
            return;
        }
        if !self.x.has_fresh_rebuild.get() {
            return;
        }
        // Appears to be dual unbounded in phase 2 after fresh rebuild.
        // Normally this implies primal infeasibility, but only allow this
        // to be claimed if the proof of primal infeasibility is true
        let (e, x, move_out) = (&mut self.e, self.x, self.move_out);
        let proof_of_infeasibility =
            self.v.row_ep.with(|row_ep| hekk::proof_of_primal_infeasibility(e, x, row_ep, move_out));
        if proof_of_infeasibility {
            // There is a proof of primal infeasibility
            self.solve_phase = PHASE_EXIT;
            // Save dual ray information
            hekk::save_dual_ray(self.x, self.row_out, self.move_out);
            self.x.model_status.set(MS_INFEASIBLE);
        } else {
            // No proof of primal infeasibility, so assume dual unbounded
            // claim is spurious. Make row_out taboo, and prevent rebuild
            let (row_out, variable_out, variable_in) = (self.row_out, self.variable_out, self.variable_in);
            self.records().add_bad_basis_change(
                row_out,
                variable_out,
                variable_in,
                REASON_FAILED_INFEASIBILITY_PROOF,
                true,
            );
            self.rebuild_reason = RR_NO;
        }
    }
}

/// HSimplexNla::rowEp2NormInScaledSpace, as clang inlines it into
/// HEkkDual::chooseRow: contracted, except for the dense loop, which it
/// vectorizes by 8 without contracting before a contracted remainder
fn row_ep_2norm_in_scaled_space(e: &EkkView, i_row: usize, row_ep: &HVec) -> f64 {
    let Some((_, row_scale)) = e.scale else { return row_ep.norm2_fused() };
    // The scaling that was applied to the unit RHS before scaled BTRAN
    let col_scale_value = e.basic_col_scale_factor(i_row);
    let mut row_ep_2norm = 0.0;
    let (use_row_indices, to_entry) = sparse_loop_style(row_ep.count, e.num_row);
    let unfused = if !use_row_indices && to_entry >= 8 { to_entry & !7 } else { 0 };
    for i_entry in 0..to_entry {
        let i_row = if use_row_indices { row_ep.index[i_entry] as usize } else { i_entry };
        let value_in_scaled_space = row_ep.array[i_row] / (row_scale[i_row] * col_scale_value);
        row_ep_2norm = if i_entry < unfused {
            row_ep_2norm + value_in_scaled_space * value_in_scaled_space
        } else {
            value_in_scaled_space.mul_add_c(value_in_scaled_space, row_ep_2norm)
        };
    }
    row_ep_2norm
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simplex::hekk::nearest_power_of_two_scale;

    #[test]
    fn power_of_two_scale() {
        // nearestPowerOfTwoScale: value * scale in (0.5, 1]
        assert_eq!(nearest_power_of_two_scale(1.0), 1.0);
        assert_eq!(nearest_power_of_two_scale(0.75), 1.0);
        assert_eq!(nearest_power_of_two_scale(3.0), 0.25);
        assert_eq!(nearest_power_of_two_scale(4.0), 0.25);
        assert_eq!(nearest_power_of_two_scale(4.5), 0.125);
        assert_eq!(nearest_power_of_two_scale(0.0), 1.0);
        assert_eq!(nearest_power_of_two_scale(1e-310), 2f64.powi(1030));
        assert_eq!(get_value_scale(&[-3.0, 0.5]), 0.25);
        assert_eq!(get_value_scale(&[]), 1.0);
    }
}
