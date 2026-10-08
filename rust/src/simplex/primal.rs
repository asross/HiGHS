//! The primal simplex solver HEkkPrimal (highs/simplex/HEkkPrimal.cpp).
//!
//! The solver works on HEkk's data through an [`EkkView`] and the
//! [`CHekk`] of hekk.rs, whose HEkk methods it calls directly (through
//! [`Primal::op3`], named after the HEkk methods), as it does the dual
//! simplex (dual.rs) for clean-up. What still reaches C++ goes through
//! hekk.rs's `Host`: the log messages (formatted here), the
//! HighsSimplexAnalysis iteration and rebuild reports, the run clock and
//! a user interrupt callback.
//!
//! C++ keeps its own HEkkPrimal for the debug levels and simplex analysis
//! (timers, operation records), which this port does not support.

use crate::util::fma::ClangFma;

use crate::hvector::{HVec, OwnedHVec};
use crate::simplex::ekk::{
    choose_price_technique, compute_dual_for_tableau_column, interleaved_part, sparse_loop_style,
    update_operation_result_density, EkkView,
};
use crate::simplex::hekk::{
    self, Bailout, CHekk, ALGORITHM_PRIMAL, LOG_DETAILED, LOG_ERROR, LOG_INFO, LOG_WARNING, MS_NOTSET,
};
use crate::sprintf;
use crate::util::hset::HSet;
use crate::util::random::HighsRandom;
use crate::util::sort::{add_to_decreasing_heap, sort_decreasing_heap};

const K_HIGHS_INF: f64 = f64::INFINITY;
const K_HYPER_PRICE_DENSITY: f64 = 0.1;

// HighsStatus
const STATUS_ERROR: i32 = -1;
const STATUS_OK: i32 = 0;
const STATUS_WARNING: i32 = 1;

// HighsModelStatus
const MODEL_STATUS_SOLVE_ERROR: i32 = 4;
const MODEL_STATUS_OPTIMAL: i32 = 7;
const MODEL_STATUS_INFEASIBLE: i32 = 8;
const MODEL_STATUS_UNBOUNDED: i32 = 10;
const MODEL_STATUS_UNKNOWN: i32 = 15;

// SolvePhase
const SOLVE_PHASE_ERROR: i32 = -3;
const SOLVE_PHASE_EXIT: i32 = -2;
const SOLVE_PHASE_UNKNOWN: i32 = -1;
const SOLVE_PHASE_OPTIMAL: i32 = 0;
const SOLVE_PHASE_1: i32 = 1;
const SOLVE_PHASE_2: i32 = 2;
const SOLVE_PHASE_OPTIMAL_CLEANUP: i32 = 4;
const SOLVE_PHASE_TABOO_BASIS: i32 = 5;

// RebuildReason
const REBUILD_REASON_CLEANUP: i32 = -1;
const REBUILD_REASON_NO: i32 = 0;
const REBUILD_REASON_UPDATE_LIMIT_REACHED: i32 = 1;
const REBUILD_REASON_SYNTHETIC_CLOCK_SAYS_INVERT: i32 = 2;
const REBUILD_REASON_POSSIBLY_OPTIMAL: i32 = 3;
const REBUILD_REASON_POSSIBLY_PHASE1_FEASIBLE: i32 = 4;
const REBUILD_REASON_POSSIBLY_PRIMAL_UNBOUNDED: i32 = 5;
const REBUILD_REASON_POSSIBLY_SINGULAR_BASIS: i32 = 7;
const REBUILD_REASON_PRIMAL_INFEASIBLE_IN_PRIMAL_SIMPLEX: i32 = 8;

const NO_ROW_SOUGHT: i32 = -2;
const NO_ROW_CHOSEN: i32 = -1;

// EdgeWeightMode
const EDGE_WEIGHT_DANTZIG: i32 = 0;
const EDGE_WEIGHT_DEVEX: i32 = 1;
const EDGE_WEIGHT_STEEPEST_EDGE: i32 = 2;
// SimplexEdgeWeightStrategy
const EDGE_WEIGHT_STRATEGY_CHOOSE: i32 = -1;
const EDGE_WEIGHT_STRATEGY_DANTZIG: i32 = 0;
const EDGE_WEIGHT_STRATEGY_DEVEX: i32 = 1;

const ALLOWED_NUM_BAD_DEVEX_WEIGHT: i32 = 3;
const BAD_DEVEX_WEIGHT_FACTOR: f64 = 3.0;
const HIGHS_DEBUG_LEVEL_COSTLY: i32 = 2;
const HIGHS_DEBUG_LEVEL_EXPENSIVE: i32 = 3;

/// The HEkk operations of the solver: see [`Primal::op3`]
#[repr(i32)]
#[derive(Clone, Copy)]
pub enum Op {
    ClearFreshValues = 0,
    IsUnconstrainedLp,
    /// The C++ part of HEkkPrimal::initialiseSolve
    InitialiseSolve,
    Bailout,
    SolveBailout,
    /// a: HighsStatus; returns the HighsStatus
    ReturnFromSolve,
    /// a: solve phase, b: perturb
    InitialiseBound,
    /// a: solve phase
    InitialiseCost,
    InitialiseNonbasicValueAndMove,
    ComputePrimal,
    ComputeDual,
    ComputeSimplexPrimalInfeasible,
    ComputeSimplexDualInfeasible,
    ComputePrimalObjectiveValue,
    ComputeDualObjectiveValue,
    ResizeBacktrackingEdgeWeight,
    PutBacktrackingBasisIfInvalid,
    /// a: rebuild reason
    RebuildRefactor,
    /// a: solve phase
    GetNonsingularInverse,
    ResetSyntheticClock,
    InitialisePartitionedRowwiseMatrix,
    ClearBadBasisChangeTabooFlag,
    TabooBadBasisChange,
    ApplyTabooVariableIn,
    UnapplyTabooVariableIn,
    /// a: variable_in, b: row_out, c: rebuild_reason
    IsBadBasisChange,
    /// After updatePivots: record the basis in visited_basis_, invalidate
    /// the dual values and clear the factor's refactor info
    BasisChanged,
    /// a: HighsModelStatus
    SetModelStatus,
    GetModelStatus,
    SavePrimalPhase1Dual,
    /// a: index, b: sign
    SavePrimalRay,
    /// The dual simplex clean-up of primal infeasibilities: returns the
    /// HighsStatus
    DualCleanup,
}

// Kinds of report
const REPORT_ITERATION: i32 = 0;
const REPORT_REBUILD: i32 = 1;
const REPORT_ANALYSIS_DATA: i32 = 2;

/// Log messages: see [`Primal::message`]
#[repr(i32)]
#[derive(Clone, Copy)]
enum Log {
    /// i: num; d: max, sum dual infeasibilities
    NearOptimal = 0,
    NoBoundPerturbation,
    OnlyTaboo,
    /// i: num free columns
    FreeColumns,
    Phase1Start,
    Phase2NoPerturbation,
    Phase2Start,
    ReturnPhase1,
    Phase2Optimal,
    ProblemOptimal,
    Phase2Unbounded,
    ProblemUnbounded,
    CleanupShift,
    RebuildPhase1,
    ChooseRowFailed,
    /// i: variable_in, iteration, update count, small, sign error; d:
    /// computed, updated dual
    DontUseVariableIn,
    /// i: variable_in
    RemoveFreeFailed,
    /// i: num missed
    MissedBoundShifts,
    /// i: num; d: max, sum
    PrimalCorrections,
    /// i: iteration, from row; d: alpha_col, alpha_row, diff, measure
    NumericalCheck,
    /// i: variable, lower; d: value, old bound, random value, feasibility,
    /// infeasibility, shift, bound, new infeasibility, error
    ShiftBound,
    /// i: iteration, num checked; d: error, norm, relative error
    PseWeightError,
    WithoutInvert,
    /// d: infeasibility (printf)
    LeavingDualInfeasibility,
    /// i: row_out (printf)
    Phase2RowOut,
}

/// HEkkPrimal
pub struct Primal {
    x: &'static CHekk,
    ekk: EkkView<'static>,
    bailout_state: Bailout,

    num_col: usize,
    num_row: usize,
    num_tot: usize,
    solve_phase: i32,
    edge_weight_mode: i32,
    primal_feasibility_tolerance: f64,
    dual_feasibility_tolerance: f64,
    rebuild_reason: i32,
    // Pivot related
    variable_in: i32,
    move_in: i32,
    row_out: i32,
    variable_out: i32,
    move_out: i32,
    theta_dual: f64,
    theta_primal: f64,
    value_in: f64,
    alpha_col: f64,
    alpha_row: f64,
    numerical_trouble: f64,

    num_flip_since_rebuild: i32,
    // Primal phase 1 tools
    ph1_sorter_r: Vec<(f64, i32)>,
    ph1_sorter_t: Vec<(f64, i32)>,
    // Edge weights
    edge_weight: Vec<f64>,
    num_devex_iterations: i32,
    num_bad_devex_weight: i32,
    devex_index: Vec<i32>,
    // Nonbasic free column data
    num_free_col: i32,
    nonbasic_free_col_set: HSet,
    // Hyper-sparse CHUZC data
    use_hyper_chuzc: bool,
    initialise_hyper_chuzc: bool,
    done_next_chuzc: bool,
    num_hyper_chuzc_candidates: i32,
    hyper_chuzc_candidate: Vec<i32>,
    hyper_chuzc_measure: Vec<f64>,
    max_hyper_chuzc_non_candidate_measure: f64,
    max_changed_measure_value: f64,
    max_changed_measure_column: i32,
    // Solve buffers
    row_ep: OwnedHVec,
    row_ap: OwnedHVec,
    col_aq: OwnedHVec,
    col_basic_feasibility_change: OwnedHVec,
    row_basic_feasibility_change: OwnedHVec,
    col_steepest_edge: OwnedHVec,
    // Just for checking PSE weights
    random: HighsRandom,

    max_max_local_primal_infeasibility: f64,
    max_max_primal_correction: f64,
    debug_max_relative_primal_steepest_edge_weight_error: f64,
    /// Whether any bad basis change may be taboo (so that
    /// applyTabooVariableIn has something to do)
    any_taboo: bool,
    /// Whether the analysis data have been set by a rebuild report
    reported: bool,
    /// solve_phase at the last iteration or rebuild, which the C++ leaves
    /// in the analysis data
    analysed_phase: i32,
}

const MAX_NUM_HYPER_CHUZC_CANDIDATES: i32 = 50;

/// HVectorBase::copy
fn copy_hvec(to: &mut OwnedHVec, from: &OwnedHVec) {
    to.clear();
    to.synthetic_tick = from.synthetic_tick;
    to.count = from.count;
    for i in 0..from.count as usize {
        let i_from = from.index[i];
        to.index[i] = i_from;
        to.array[i_from as usize] = from.array[i_from as usize];
    }
}

/// Whether a value is below its lower bound (-1) or above its upper bound
/// (1) by more than the tolerance
#[inline]
fn bound_violated(value: f64, lower: f64, upper: f64, tolerance: f64) -> i32 {
    if value < lower - tolerance {
        -1
    } else if value > upper + tolerance {
        1
    } else {
        0
    }
}

impl Primal {
    // ---- The C++ side ----

    /// An HEkk operation
    fn op3(&mut self, op: Op, a: i32, b: i32, c: i32) -> i32 {
        let x = self.x;
        let e = &mut self.ekk;
        match op {
            Op::ClearFreshValues => x.clear_fresh_values(),
            Op::IsUnconstrainedLp => return hekk::is_unconstrained_lp(e, x) as i32,
            Op::InitialiseSolve => {
                e.status.has_primal_objective_value = false;
                e.status.has_dual_objective_value = false;
                x.model_status.set(MS_NOTSET);
                x.solve_bailout.set(false);
                x.called_return_from_solve.set(false);
                x.exit_algorithm.set(ALGORITHM_PRIMAL);
                if !e.status.has_dual_steepest_edge_weights {
                    // No dual weights to maintain, so ensure that the
                    // vectors are assigned since they are used around
                    // factorization and when setting up the backtracking
                    // information (C++ has sized them)
                    hekk::assign_unit_dual_edge_weights(e);
                }
            }
            Op::Bailout => return self.bailout_state.check(x) as i32,
            Op::SolveBailout => return x.solve_bailout.get() as i32,
            Op::ReturnFromSolve => return hekk::return_from_solve(e, x, a),
            Op::InitialiseBound => e.initialise_bound(ALGORITHM_PRIMAL, a, b != 0),
            Op::InitialiseCost => hekk::initialise_cost(e, x, ALGORITHM_PRIMAL, false),
            Op::InitialiseNonbasicValueAndMove => e.initialise_nonbasic_value_and_move(),
            Op::ComputePrimal => hekk::compute_primal(e),
            Op::ComputeDual => hekk::compute_dual(e, x),
            Op::ComputeSimplexPrimalInfeasible => e.compute_simplex_primal_infeasible(),
            Op::ComputeSimplexDualInfeasible => e.compute_simplex_dual_infeasible(),
            Op::ComputePrimalObjectiveValue => hekk::compute_primal_objective_value(e),
            Op::ComputeDualObjectiveValue => hekk::compute_dual_objective_value(e, 2),
            // Sized by C++
            Op::ResizeBacktrackingEdgeWeight => {}
            Op::PutBacktrackingBasisIfInvalid => {
                if !x.valid_backtracking_basis.get() {
                    hekk::put_backtracking_basis(e, x);
                }
            }
            Op::RebuildRefactor => return hekk::rebuild_refactor(e, x, a) as i32,
            Op::GetNonsingularInverse => return hekk::get_nonsingular_inverse(e, x, a) as i32,
            Op::ResetSyntheticClock => hekk::reset_synthetic_clock(e, x),
            Op::InitialisePartitionedRowwiseMatrix => hekk::initialise_partitioned_rowwise_matrix(e, x),
            Op::ClearBadBasisChangeTabooFlag => x.records().clear_taboo_flag(),
            Op::TabooBadBasisChange => return x.records().taboo() as i32,
            Op::ApplyTabooVariableIn => x.records().apply_taboo(e.work_dual, 0.0, 1),
            Op::UnapplyTabooVariableIn => x.records().unapply_taboo(e.work_dual, 1),
            Op::IsBadBasisChange => return hekk::is_bad_basis_change(e, x, a, b, c) as i32,
            Op::BasisChanged => {
                // The parts of HEkk::updatePivots and HEkk::updateFactor
                // beyond the kernels
                x.dual_values_valid.set(false);
                x.records().visited.insert(*e.basis_hash);
                e.factor.refactor_info_clear();
            }
            Op::SetModelStatus => x.model_status.set(a),
            Op::GetModelStatus => return x.model_status.get(),
            Op::SavePrimalPhase1Dual => x.records().out.primal_phase1_dual = Some(e.work_dual.to_vec()),
            Op::SavePrimalRay => hekk::save_primal_ray(x, a, b),
            Op::DualCleanup => return self.cleanup_with_dual(),
        }
        0
    }

    fn op(&mut self, op: Op) -> i32 {
        self.op3(op, 0, 0, 0)
    }

    fn op1(&mut self, op: Op, a: i32) -> i32 {
        self.op3(op, a, 0, 0)
    }

    fn op_keep(&mut self, op: Op, a: i32, b: i32, c: i32) -> i32 {
        self.op3(op, a, b, c)
    }

    /// HEkkPrimal::cleanupWithDual
    fn cleanup_with_dual(&mut self) -> i32 {
        let x = self.x;
        let e = &mut self.ekk;
        x.dev(LOG_INFO, || {
            sprintf!(
                "HEkkPrimal:: Using dual simplex to try to clean up num / max / sum = %d / %g / %g primal infeasibilities\n",
                *e.num_primal_infeasibilities,
                *e.max_primal_infeasibility,
                *e.sum_primal_infeasibilities
            )
        });
        hekk::compute_primal_objective_value(e);
        // Switch off any bound perturbation
        let save_dual_simplex_cost_perturbation_multiplier = *e.dual_simplex_cost_perturbation_multiplier;
        *e.dual_simplex_cost_perturbation_multiplier = 0.0;
        let simplex_strategy = x.simplex_strategy.get();
        x.simplex_strategy.set(1);
        let call_status = hekk::dual_solve(x, true);
        // Restore any bound perturbation
        let e = &mut self.ekk;
        *e.dual_simplex_cost_perturbation_multiplier = save_dual_simplex_cost_perturbation_multiplier;
        x.simplex_strategy.set(simplex_strategy);
        let return_status = hekk::interpret_call_status(x, call_status, STATUS_OK, "HEkkDual::solve");
        // Reset called_return_from_solve_ to be false, since it's called
        // for this solve
        x.called_return_from_solve.set(false);
        if return_status != STATUS_OK {
            return return_status;
        }
        if x.model_status.get() == MODEL_STATUS_OPTIMAL
            && *e.num_primal_infeasibilities + *e.num_dual_infeasibilities != 0
        {
            x.dev(LOG_WARNING, || {
                sprintf!(
                    "HEkkPrimal:: Dual simplex clean up yields  optimality, but with %d (max %g) primal infeasibilities and %d (max %g) dual infeasibilities\n",
                    *e.num_primal_infeasibilities,
                    *e.max_primal_infeasibility,
                    *e.num_dual_infeasibilities,
                    *e.max_dual_infeasibility
                )
            });
        }
        STATUS_OK
    }

    fn log(&self, id: Log, i: &[i32], d: &[f64]) {
        use Log::*;
        let x = self.x;
        let t = match id {
            NearOptimal | NoBoundPerturbation | Phase1Start | Phase2Start | ReturnPhase1 | Phase2Optimal
            | ProblemOptimal | CleanupShift => LOG_DETAILED,
            Phase2NoPerturbation | RebuildPhase1 => LOG_WARNING,
            ChooseRowFailed | RemoveFreeFailed | MissedBoundShifts | WithoutInvert => LOG_ERROR,
            PseWeightError | LeavingDualInfeasibility | Phase2RowOut => {
                x.printf(&self.message(id, i, d));
                return;
            }
            _ => LOG_INFO,
        };
        x.dev(t, || self.message(id, i, d));
    }

    /// The text of a message
    fn message(&self, id: Log, i: &[i32], d: &[f64]) -> String {
        use Log::*;
        match id {
            NearOptimal => sprintf!(
                "Primal feasible and num / max / sum dual infeasibilities of %d / %g / %g, so near-optimal\n",
                i[0],
                d[0],
                d[1]
            ),
            NoBoundPerturbation => "Near-optimal, so don't use bound perturbation\n".into(),
            OnlyTaboo => "HEkkPrimal::solve Only basis change is taboo\n".into(),
            FreeColumns => sprintf!("HEkkPrimal:: LP has %d free columns\n", i[0]),
            Phase1Start => "primal-phase1-start\n".into(),
            Phase2NoPerturbation => "Moving to phase 2, but not allowing bound perturbation\n".into(),
            Phase2Start => "primal-phase2-start\n".into(),
            ReturnPhase1 => "primal-return-phase1\n".into(),
            Phase2Optimal => "primal-phase-2-optimal\n".into(),
            ProblemOptimal => "problem-optimal\n".into(),
            Phase2Unbounded => "primal-phase-2-unbounded\n".into(),
            ProblemUnbounded => "problem-primal-unbounded\n".into(),
            CleanupShift => "primal-cleanup-shift\n".into(),
            RebuildPhase1 => "HEkkPrimal::rebuild switching back to phase 1 from phase 2\n".into(),
            ChooseRowFailed => "Primal phase 1 choose row failed\n".into(),
            DontUseVariableIn => sprintf!(
                "Chosen entering variable %d (Iter = %d; Update = %d) has computed (updated) dual of %10.4g (%10.4g) so don't use it%s%s\n",
                i[0],
                i[1],
                i[2],
                d[0],
                d[1],
                if i[3] != 0 { "; too small" } else { "" },
                if i[4] != 0 { "; sign error" } else { "" }
            ),
            RemoveFreeFailed => {
                sprintf!("HEkkPrimal::phase1update failed to remove nonbasic free column %d\n", i[0])
            }
            MissedBoundShifts => sprintf!("correctPrimal: Missed %d bound shifts\n", i[0]),
            PrimalCorrections => sprintf!(
                "phase2CorrectPrimal: num / max / sum primal corrections = %d / %g / %g\n",
                i[0],
                d[0],
                d[1]
            ),
            NumericalCheck => sprintf!(
                "Numerical check: Iter %4d: alpha_col = %12g, (From %3s alpha_row = %12g), aDiff = %12g: measure = %12g\n",
                i[0],
                d[0],
                if i[1] != 0 { "Row" } else { "Col" },
                d[1],
                d[2],
                d[3]
            ),
            ShiftBound => sprintf!(
                "HEkkPrimal::shiftBound Value(%4d) = %10.4g exceeds %s: random_value = %g; value = %g; feasibility = %g; infeasibility = %g; shift = %g; bound = %g; new_infeasibility = %g with error %g\n",
                i[0],
                d[0],
                if i[1] != 0 { "lower" } else { "upper" },
                d[1],
                d[2],
                d[0],
                d[3],
                d[4],
                d[5],
                d[6],
                d[7],
                d[8]
            ),
            PseWeightError => sprintf!(
                "HEkk::debugPrimalSteepestEdgeWeights Iteration %5d: Checked %2d weights: error = %10.4g; norm = %10.4g; relative error = %10.4g\n",
                i[0],
                i[1],
                d[0],
                d[1],
                d[2]
            ),
            WithoutInvert => "HEkkPrimal::solve called without INVERT\n".into(),
            LeavingDualInfeasibility => sprintf!("Dual infeasibility %g for leaving column!\n", d[0]),
            Phase2RowOut => sprintf!(
                "HEkkPrimal::solvePhase2 row_out = %d solve %d\n",
                i[0],
                self.x.debug_solve_call_num
            ),
        }
    }

    /// HEkkPrimal::iterationAnalysisData and its iteration or rebuild
    /// report (`CHekk::primal_report`)
    fn report(&self, kind: i32, reason_for_rebuild: i32) {
        let solve_phase = if kind == REPORT_ANALYSIS_DATA { self.analysed_phase } else { self.solve_phase };
        let e = &self.ekk;
        self.x.primal_report(
            kind,
            solve_phase,
            self.row_out,
            self.variable_in,
            self.rebuild_reason,
            reason_for_rebuild,
            *e.updated_primal_objective_value,
            (
                *e.num_primal_infeasibilities,
                *e.sum_primal_infeasibilities,
                *e.num_dual_infeasibilities,
                *e.sum_dual_infeasibilities,
            ),
            [*e.col_aq_density, *e.row_ep_density, *e.row_ap_density, *e.row_dse_density],
        );
    }

    fn report_rebuild(&mut self, reason_for_rebuild: i32) {
        self.analysed_phase = self.solve_phase;
        self.report(REPORT_REBUILD, reason_for_rebuild);
        self.reported = true;
    }

    fn return_from_solve(&mut self, status: i32) -> i32 {
        // Leave the analysis data as the last iteration or rebuild report
        // of the C++ would have left them (they are reported only when due)
        if self.reported {
            self.report(REPORT_ANALYSIS_DATA, 0);
        }
        self.op_keep(Op::ReturnFromSolve, status, 0, 0)
    }

    /// HEkk::bailout
    fn bailout(&mut self) -> bool {
        self.bailout_state.check(self.x)
    }

    // ---- Set-up ----

    /// HEkkPrimal::HEkkPrimal and initialiseInstance
    ///
    /// # Safety
    /// As for CHekk::view: `x` filled for the solve, and no other view of
    /// HEkk's data used while this solver runs
    pub unsafe fn new(x: &CHekk) -> Primal {
        let x: &'static CHekk = &*(x as *const CHekk);
        let ekk = x.view();
        let num_col = ekk.num_col;
        let num_row = ekk.num_row;
        let num_tot = num_col + num_row;
        let mut p = Primal {
            x,
            ekk,
            bailout_state: Bailout::new(),
            num_col,
            num_row,
            num_tot,
            solve_phase: 0,
            edge_weight_mode: EDGE_WEIGHT_DEVEX,
            primal_feasibility_tolerance: 0.0,
            dual_feasibility_tolerance: 0.0,
            rebuild_reason: REBUILD_REASON_NO,
            variable_in: 0,
            move_in: 0,
            row_out: 0,
            variable_out: 0,
            move_out: 0,
            theta_dual: 0.0,
            theta_primal: 0.0,
            value_in: 0.0,
            alpha_col: 0.0,
            alpha_row: 0.0,
            numerical_trouble: 0.0,
            num_flip_since_rebuild: 0,
            ph1_sorter_r: Vec::with_capacity(num_row),
            ph1_sorter_t: Vec::with_capacity(num_row),
            edge_weight: Vec::new(),
            num_devex_iterations: 0,
            num_bad_devex_weight: 0,
            devex_index: Vec::new(),
            num_free_col: 0,
            nonbasic_free_col_set: HSet::default(),
            use_hyper_chuzc: false,
            initialise_hyper_chuzc: false,
            done_next_chuzc: false,
            num_hyper_chuzc_candidates: 0,
            hyper_chuzc_candidate: vec![0; 1 + MAX_NUM_HYPER_CHUZC_CANDIDATES as usize],
            hyper_chuzc_measure: vec![0.0; 1 + MAX_NUM_HYPER_CHUZC_CANDIDATES as usize],
            max_hyper_chuzc_non_candidate_measure: 0.0,
            max_changed_measure_value: 0.0,
            max_changed_measure_column: 0,
            row_ep: OwnedHVec::new(num_row as i32),
            row_ap: OwnedHVec::new(num_col as i32),
            col_aq: OwnedHVec::new(num_row as i32),
            col_basic_feasibility_change: OwnedHVec::new(num_row as i32),
            row_basic_feasibility_change: OwnedHVec::new(num_col as i32),
            col_steepest_edge: OwnedHVec::new(num_row as i32),
            random: HighsRandom::new(0),
            max_max_local_primal_infeasibility: 0.0,
            max_max_primal_correction: 0.0,
            debug_max_relative_primal_steepest_edge_weight_error: 0.0,
            any_taboo: false,
            reported: false,
            analysed_phase: 0,
        };
        let (lower, upper) = (&p.ekk.work_lower[..num_tot], &p.ekk.work_upper[..num_tot]);
        p.num_free_col = (0..num_tot).filter(|&i| lower[i] == -K_HIGHS_INF && upper[i] == K_HIGHS_INF).count() as i32;
        if p.num_free_col != 0 {
            p.log(Log::FreeColumns, &[p.num_free_col], &[]);
            p.nonbasic_free_col_set.setup(p.num_free_col, num_tot as i32);
        }
        // hyper_chuzc_candidate_set is set up but not used
        p
    }

    /// HEkkPrimal::initialiseSolve
    fn initialise_solve(&mut self) {
        self.primal_feasibility_tolerance = self.ekk.primal_feasibility_tolerance;
        self.dual_feasibility_tolerance = self.ekk.dual_feasibility_tolerance;
        // Status flags, model status, bailout, exit algorithm and (if there
        // are no dual steepest edge weights) the dual edge weights
        self.op(Op::InitialiseSolve);
        self.rebuild_reason = REBUILD_REASON_NO;
        let edge_weight_strategy = self.ekk.simplex_primal_edge_weight_strategy;
        self.edge_weight_mode = if edge_weight_strategy == EDGE_WEIGHT_STRATEGY_CHOOSE
            || edge_weight_strategy == EDGE_WEIGHT_STRATEGY_DEVEX
        {
            // By default, use Devex
            EDGE_WEIGHT_DEVEX
        } else if edge_weight_strategy == EDGE_WEIGHT_STRATEGY_DANTZIG {
            EDGE_WEIGHT_DANTZIG
        } else {
            EDGE_WEIGHT_STEEPEST_EDGE
        };
        if self.edge_weight_mode == EDGE_WEIGHT_DANTZIG {
            self.edge_weight.clear();
            self.edge_weight.resize(self.num_tot, 1.0);
        } else if self.edge_weight_mode == EDGE_WEIGHT_DEVEX {
            self.initialise_devex_framework();
        } else {
            self.compute_primal_steepest_edge_weights();
        }
    }

    // ---- Solve ----

    /// HEkkPrimal::solve
    pub fn solve(&mut self, pass_force_phase2: bool) -> i32 {
        self.op(Op::ClearFreshValues);
        // Initialise control data for a particular solve
        self.initialise_solve();
        // Assumes that the LP has a positive number of rows
        if self.op(Op::IsUnconstrainedLp) != 0 {
            return self.return_from_solve(STATUS_ERROR);
        }
        if !self.ekk.status.has_invert {
            self.log(Log::WithoutInvert, &[], &[]);
            debug_assert!(self.ekk.status.has_fresh_invert);
            return self.return_from_solve(STATUS_ERROR);
        }
        // Get the nonbasic free column set
        self.get_nonbasic_free_column_set();

        let primal_feasible_with_unperturbed_bounds = *self.ekk.num_primal_infeasibilities == 0;
        let force_phase2 = pass_force_phase2
            || *self.ekk.max_primal_infeasibility * *self.ekk.max_primal_infeasibility
                < self.ekk.primal_feasibility_tolerance;
        // Consider there to be no primal infeasibilities if there are
        // none, or if phase 2 is forced, in which case any primal
        // infeasibilities will be shifted
        let no_simplex_primal_infeasibilities = primal_feasible_with_unperturbed_bounds || force_phase2;
        let near_optimal = *self.ekk.num_dual_infeasibilities < 1000
            && *self.ekk.max_dual_infeasibility < 1e-3
            && no_simplex_primal_infeasibilities;
        if near_optimal {
            self.log(
                Log::NearOptimal,
                &[*self.ekk.num_dual_infeasibilities],
                &[*self.ekk.max_dual_infeasibility, *self.ekk.sum_dual_infeasibilities],
            );
        }
        // Perturb bounds according to whether the solution is near-optimal
        let perturb_bounds = !near_optimal;
        if !perturb_bounds {
            self.log(Log::NoBoundPerturbation, &[], &[]);
        }
        if perturb_bounds && *self.ekk.primal_simplex_bound_perturbation_multiplier != 0.0 {
            self.op3(Op::InitialiseBound, SOLVE_PHASE_UNKNOWN, perturb_bounds as i32, 0);
            self.op(Op::InitialiseNonbasicValueAndMove);
            self.op(Op::ComputePrimal);
            self.op(Op::ComputeSimplexPrimalInfeasible);
        }
        // Check whether the time/iteration limit has been reached. First
        // point at which a non-error return can occur
        if self.bailout() {
            return self.return_from_solve(STATUS_WARNING);
        }

        // Now to do some iterations!
        let num_primal_infeasibility = *self.ekk.num_primal_infeasibilities;
        self.solve_phase = if num_primal_infeasibility > 0 { SOLVE_PHASE_1 } else { SOLVE_PHASE_2 };
        if force_phase2 {
            // Dual infeasibilities without cost perturbation involved
            // fixed variables or were (at most) small, so can easily be
            // removed by flips for and fixed variables shifts for the rest
            self.solve_phase = SOLVE_PHASE_2;
        }
        // Resize the copy of scattered edge weights for backtracking
        self.op(Op::ResizeBacktrackingEdgeWeight);

        // The major solving loop
        //
        // Initialise records for primal correction reporting
        self.max_max_primal_correction = 0.0;
        while self.solve_phase != 0 {
            let it0 = *self.ekk.iteration_count;
            // When starting a new phase the (updated) primal objective
            // function value isn't known
            self.ekk.status.has_primal_objective_value = false;
            if self.solve_phase == SOLVE_PHASE_UNKNOWN {
                // Determine the number of primal infeasibilities, and hence
                // the solve phase
                self.op(Op::ComputeSimplexPrimalInfeasible);
                let num_primal_infeasibility = *self.ekk.num_primal_infeasibilities;
                self.solve_phase = if num_primal_infeasibility > 0 { SOLVE_PHASE_1 } else { SOLVE_PHASE_2 };
                if *self.ekk.backtracking {
                    // Backtracking
                    self.op1(Op::InitialiseCost, self.solve_phase);
                    self.op(Op::InitialiseNonbasicValueAndMove);
                    // Can now forget that we might have been backtracking
                    *self.ekk.backtracking = false;
                }
            }
            debug_assert!(self.solve_phase == SOLVE_PHASE_1 || self.solve_phase == SOLVE_PHASE_2);
            if self.solve_phase == SOLVE_PHASE_1 {
                self.solve_phase1();
                *self.ekk.primal_phase1_iteration_count += *self.ekk.iteration_count - it0;
            } else if self.solve_phase == SOLVE_PHASE_2 {
                self.solve_phase2();
                *self.ekk.primal_phase2_iteration_count += *self.ekk.iteration_count - it0;
            } else {
                // Should only be kSolvePhase1 or kSolvePhase2
                self.op1(Op::SetModelStatus, MODEL_STATUS_SOLVE_ERROR);
                return self.return_from_solve(STATUS_ERROR);
            }
            // Return if bailing out from solve
            if self.op_keep(Op::SolveBailout, 0, 0, 0) != 0 {
                return self.return_from_solve(STATUS_WARNING);
            }
            // Look for scenarios when the major solving loop ends
            if self.solve_phase == SOLVE_PHASE_TABOO_BASIS {
                // Only basis change is taboo
                self.log(Log::OnlyTaboo, &[], &[]);
                self.op1(Op::SetModelStatus, MODEL_STATUS_UNKNOWN);
                return self.return_from_solve(STATUS_WARNING);
            }
            if self.solve_phase == SOLVE_PHASE_ERROR {
                // Solver error
                self.op1(Op::SetModelStatus, MODEL_STATUS_SOLVE_ERROR);
                return self.return_from_solve(STATUS_ERROR);
            }
            if self.solve_phase == SOLVE_PHASE_EXIT {
                // LP identified as not having an optimal solution. If
                // infeasible, save the primal phase 1 dual values before
                // they are overwritten with the duals for the original
                // objective
                if self.op_keep(Op::GetModelStatus, 0, 0, 0) == MODEL_STATUS_INFEASIBLE {
                    self.op(Op::SavePrimalPhase1Dual);
                }
                break;
            }
            if self.solve_phase == SOLVE_PHASE_OPTIMAL_CLEANUP {
                // Primal infeasibilities after phase 2. Dual feasible with
                // primal infeasibilities so use dual simplex to clean up
                break;
            }
            // If solve_phase == kSolvePhaseOptimal == 0 then major solving
            // loop ends naturally since solve_phase is false
        }
        if self.solve_phase == SOLVE_PHASE_OPTIMAL {
            self.op1(Op::SetModelStatus, MODEL_STATUS_OPTIMAL);
        }
        if self.solve_phase == SOLVE_PHASE_OPTIMAL_CLEANUP {
            // Use dual to clean up
            let return_status = self.op(Op::DualCleanup);
            // The analysis data are the dual simplex solver's
            self.reported = false;
            return self.return_from_solve(return_status);
        }
        self.return_from_solve(STATUS_OK)
    }

    /// The end of the solving loops of solvePhase1 and solvePhase2: whether
    /// to stop iterating, setting kSolvePhaseTabooBasis when only a taboo
    /// basis change could be made
    fn finished(&mut self) -> Option<bool> {
        // If the data are fresh from rebuild() and no flips have occurred,
        // possibly break out of the outer loop to see what's occurred
        let finished = self.ekk.status.has_fresh_rebuild
            && self.num_flip_since_rebuild == 0
            && self.op1(Op::RebuildRefactor, self.rebuild_reason) == 0;
        if finished && self.op(Op::TabooBadBasisChange) != 0 {
            // A bad basis change has had to be made taboo without any
            // other basis changes or flips having been performed from a
            // fresh rebuild. In other words, the only basis change that
            // could be made is not permitted, so no definitive statement
            // about the LP can be made.
            self.solve_phase = SOLVE_PHASE_TABOO_BASIS;
            return None;
        }
        Some(finished)
    }

    /// The solving loops of solvePhase1 and solvePhase2: false on returning
    /// early
    fn solve_loop(&mut self, phase: i32) -> bool {
        loop {
            // Rebuild
            //
            // solve_phase = kSolvePhaseError is set if the basis matrix is
            // singular
            self.rebuild();
            if self.solve_phase == SOLVE_PHASE_ERROR || self.solve_phase == SOLVE_PHASE_UNKNOWN {
                return false;
            }
            if self.bailout() {
                return false;
            }
            // In phase 1, solve_phase = kSolvePhase2 is set if no primal
            // infeasibilities are found in rebuild(), in which case return
            // for phase 2 (and vice versa)
            if self.solve_phase != phase {
                break;
            }
            loop {
                self.iterate();
                if self.bailout() {
                    return false;
                }
                if self.solve_phase == SOLVE_PHASE_ERROR {
                    return false;
                }
                debug_assert!(self.solve_phase == phase);
                if self.rebuild_reason != 0 {
                    break;
                }
            }
            match self.finished() {
                None => return false,
                Some(true) => break,
                Some(false) => {}
            }
        }
        true
    }

    /// HEkkPrimal::solvePhase1
    fn solve_phase1(&mut self) {
        // When starting a new phase the (updated) primal objective function
        // value isn't known
        self.ekk.status.has_primal_objective_value = false;
        self.ekk.status.has_dual_objective_value = false;
        // Possibly bail out immediately if iteration limit is current value
        if self.bailout() {
            return;
        }
        self.log(Log::Phase1Start, &[], &[]);
        // If there's no backtracking basis, save the initial basis in case
        // of backtracking
        self.op(Op::PutBacktrackingBasisIfInvalid);
        if !self.solve_loop(SOLVE_PHASE_1) {
            return;
        }
        if self.solve_phase == SOLVE_PHASE_1 {
            // Determine whether primal infeasibility has been identified
            if self.variable_in < 0 {
                // Optimal in phase 1, so should have primal infeasibilities
                debug_assert!(*self.ekk.num_primal_infeasibilities > 0);
                if *self.ekk.bounds_shifted || *self.ekk.bounds_perturbed {
                    // Remove any bound shifts or perturbations and return to
                    // phase 1
                    self.cleanup();
                } else {
                    self.op1(Op::SetModelStatus, MODEL_STATUS_INFEASIBLE);
                    self.solve_phase = SOLVE_PHASE_EXIT;
                }
            }
        }
        if self.solve_phase == SOLVE_PHASE_2 {
            // Moving to phase 2 so comment if bound perturbation is not
            // permitted
            if !*self.ekk.allow_bound_perturbation {
                self.log(Log::Phase2NoPerturbation, &[], &[]);
            }
        }
    }

    /// HEkkPrimal::solvePhase2
    fn solve_phase2(&mut self) {
        self.ekk.status.has_primal_objective_value = false;
        self.ekk.status.has_dual_objective_value = false;
        // Possibly bail out immediately if iteration limit is current value
        if self.bailout() {
            return;
        }
        self.log(Log::Phase2Start, &[], &[]);
        // phase2UpdatePrimal(true)
        self.max_max_local_primal_infeasibility = 0.0;
        // If there's no backtracking basis Save the initial basis in case
        // of backtracking
        self.op(Op::PutBacktrackingBasisIfInvalid);
        if !self.solve_loop(SOLVE_PHASE_2) {
            return;
        }
        if self.solve_phase == SOLVE_PHASE_1 {
            self.log(Log::ReturnPhase1, &[], &[]);
        } else if self.variable_in == -1 {
            // There is no candidate in CHUZC, even after rebuild so
            // probably optimal
            self.log(Log::Phase2Optimal, &[], &[]);
            // Remove any bound perturbations and see if basis is still
            // primal feasible
            self.cleanup();
            if *self.ekk.num_primal_infeasibilities > 0 {
                // There are primal infeasibilities, so consider performing
                // dual simplex iterations to get primal feasibility
                self.solve_phase = SOLVE_PHASE_OPTIMAL_CLEANUP;
            } else {
                // There are no primal infeasibilities so optimal!
                self.solve_phase = SOLVE_PHASE_OPTIMAL;
                self.log(Log::ProblemOptimal, &[], &[]);
                self.op1(Op::SetModelStatus, MODEL_STATUS_OPTIMAL);
                self.op(Op::ComputeDualObjectiveValue);
            }
        } else if self.row_out == NO_ROW_SOUGHT {
            // CHUZR has not been performed - because the chosen reduced
            // cost was unattractive when computed from scratch and no
            // rebuild was required. This is very rare and should be handled
            // otherwise
            self.log(Log::Phase2RowOut, &[self.row_out], &[]);
            debug_assert!(self.row_out != NO_ROW_SOUGHT);
        } else {
            // No candidate in CHUZR
            if self.row_out >= 0 {
                self.log(Log::Phase2RowOut, &[self.row_out], &[]);
            }
            // Ensure that CHUZR was performed and found no row
            debug_assert!(self.row_out == NO_ROW_CHOSEN);
            // There is no candidate in CHUZR, so probably primal unbounded
            self.log(Log::Phase2Unbounded, &[], &[]);
            if *self.ekk.bounds_shifted || *self.ekk.bounds_perturbed {
                // If the bounds have been shifted or perturbed, clean up
                // and return
                self.cleanup();
                // If there are primal infeasibilities, go back to phase 1
                if *self.ekk.num_primal_infeasibilities > 0 {
                    self.solve_phase = SOLVE_PHASE_1;
                }
            } else {
                // The bounds have not been perturbed, so primal unbounded
                self.solve_phase = SOLVE_PHASE_EXIT;
                // Primal unbounded, so save primal ray
                debug_assert!(self.variable_in >= 0);
                self.op3(Op::SavePrimalRay, self.variable_in, -self.move_in, 0);
                self.log(Log::ProblemUnbounded, &[], &[]);
                self.op1(Op::SetModelStatus, MODEL_STATUS_UNBOUNDED);
            }
        }
    }

    /// HEkkPrimal::cleanup
    fn cleanup(&mut self) {
        if !*self.ekk.bounds_shifted && !*self.ekk.bounds_perturbed {
            return;
        }
        self.log(Log::CleanupShift, &[], &[]);
        // Remove perturbation and don't permit further perturbation
        self.op3(Op::InitialiseBound, self.solve_phase, 0, 0);
        self.op(Op::InitialiseNonbasicValueAndMove);
        *self.ekk.allow_bound_perturbation = false;
        // Compute the primal values
        self.op(Op::ComputePrimal);
        // Compute the primal infeasibilities
        self.op(Op::ComputeSimplexPrimalInfeasible);
        // Compute the primal objective value
        self.op(Op::ComputePrimalObjectiveValue);
        // Now that there's a new primal_objective_value, reset the updated
        // value
        *self.ekk.updated_primal_objective_value = *self.ekk.primal_objective_value;
        // Report the dual infeasibilities
        self.op(Op::ComputeSimplexDualInfeasible);
        self.report_rebuild(REBUILD_REASON_CLEANUP);
    }

    /// HEkkPrimal::rebuild
    fn rebuild(&mut self) {
        // Clear taboo flag from any bad basis changes
        self.op(Op::ClearBadBasisChangeTabooFlag);
        self.any_taboo = false;

        // Record whether the update objective value should be tested
        let check_updated_objective_value = self.ekk.status.has_primal_objective_value;
        let previous_primal_objective_value =
            if check_updated_objective_value { *self.ekk.updated_primal_objective_value } else { -K_HIGHS_INF };

        // Decide whether refactorization should be performed
        let refactor_basis_matrix = self.op1(Op::RebuildRefactor, self.rebuild_reason) != 0;

        // Take a local copy of the rebuild reason and then reset the global
        // value
        let local_rebuild_reason = self.rebuild_reason;
        self.rebuild_reason = REBUILD_REASON_NO;
        if refactor_basis_matrix {
            // Get a nonsingular inverse if possible
            if self.op1(Op::GetNonsingularInverse, self.solve_phase) == 0 {
                self.solve_phase = SOLVE_PHASE_ERROR;
                return;
            }
            // Record the synthetic clock for INVERT, and zero it for UPDATE
            self.op(Op::ResetSyntheticClock);
        }
        if !self.ekk.status.has_ar_matrix {
            // Don't have the row-wise matrix, so reinitialise it
            //
            // Should only happen when backtracking
            debug_assert!(*self.ekk.backtracking);
            self.op(Op::InitialisePartitionedRowwiseMatrix);
        }
        if *self.ekk.backtracking {
            // If backtracking, may change phase, so drop out
            self.solve_phase = SOLVE_PHASE_UNKNOWN;
            return;
        }

        self.op(Op::ComputePrimal);
        if self.solve_phase == SOLVE_PHASE_2 {
            let correct_primal_ok = self.correct_primal();
            debug_assert!(correct_primal_ok);
        }
        self.get_basic_primal_infeasibility();
        if *self.ekk.num_primal_infeasibilities > 0 {
            // Primal infeasibilities so should be in phase 1
            if self.solve_phase == SOLVE_PHASE_2 {
                self.log(Log::RebuildPhase1, &[], &[]);
                self.solve_phase = SOLVE_PHASE_1;
            }
            self.phase1_compute_dual();
        } else {
            // No primal infeasibilities so in phase 2. Reset costs if was
            // previously in phase 1
            if self.solve_phase == SOLVE_PHASE_1 {
                self.op1(Op::InitialiseCost, self.solve_phase);
                self.solve_phase = SOLVE_PHASE_2;
            }
            self.op(Op::ComputeDual);
        }
        self.op(Op::ComputeSimplexDualInfeasible);
        self.op(Op::ComputePrimalObjectiveValue);
        if check_updated_objective_value {
            // Apply the objective value correction due to computing primal
            // values from scratch
            let primal_objective_value_correction =
                *self.ekk.primal_objective_value - previous_primal_objective_value;
            *self.ekk.updated_primal_objective_value += primal_objective_value_correction;
        }
        // Now that there's a new dual_objective_value, reset the updated
        // value
        *self.ekk.updated_primal_objective_value = *self.ekk.primal_objective_value;

        self.report_rebuild(local_rebuild_reason);

        // Record the synthetic clock for INVERT, and zero it for UPDATE
        self.op(Op::ResetSyntheticClock);

        // Hyper-sparse CHUZC is not used
        self.use_hyper_chuzc = false;
        self.hyper_choose_column_clear();

        self.num_flip_since_rebuild = 0;
        // Data are fresh from rebuild
        self.ekk.status.has_fresh_rebuild = true;
    }

    // ---- Iteration ----

    /// HEkkPrimal::iterate
    fn iterate(&mut self) {
        // Initialise row_out so that aborting iteration before CHUZR due to
        // numerical test of chosen reduced cost can be spotted
        self.row_out = NO_ROW_SOUGHT;
        // Perform CHUZC
        self.chuzc();
        if self.variable_in == -1 {
            self.rebuild_reason = REBUILD_REASON_POSSIBLY_OPTIMAL;
            return;
        }

        // Perform FTRAN - and dual value cross-check to decide whether to
        // use the variable
        //
        // rebuild_reason = kRebuildReasonPossiblySingularBasis is set if
        // numerical trouble is detected
        if !self.use_variable_in() {
            debug_assert!(
                self.rebuild_reason == 0 || self.rebuild_reason == REBUILD_REASON_POSSIBLY_SINGULAR_BASIS
            );
            return;
        }
        debug_assert!(self.rebuild_reason == 0);

        // Perform CHUZR
        if self.solve_phase == SOLVE_PHASE_1 {
            self.phase1_choose_row();
            debug_assert!(self.row_out != NO_ROW_SOUGHT);
            if self.row_out == NO_ROW_CHOSEN {
                self.log(Log::ChooseRowFailed, &[], &[]);
                self.solve_phase = SOLVE_PHASE_ERROR;
                return;
            }
        } else {
            self.choose_row();
        }
        debug_assert!(self.rebuild_reason == 0);

        // Consider whether to perform a bound swap - either because it's
        // shorter than the pivoting step or, in the case of Phase 1,
        // because it's cheaper than pivoting - which may be questionable
        //
        // rebuild_reason = kRebuildReasonPossiblyPrimalUnbounded is set in
        // phase 2 if there's no pivot or bound swap
        debug_assert!(self.solve_phase == SOLVE_PHASE_2 || self.row_out >= 0);
        self.consider_bound_swap();
        if self.rebuild_reason == REBUILD_REASON_POSSIBLY_PRIMAL_UNBOUNDED {
            return;
        }
        debug_assert!(self.rebuild_reason == 0);

        if self.row_out >= 0 {
            // Perform unit BTRAN and PRICE to get pivotal row - and do a
            // numerical check
            self.assess_pivot();
            if self.rebuild_reason != 0 {
                debug_assert!(self.rebuild_reason == REBUILD_REASON_POSSIBLY_SINGULAR_BASIS);
                return;
            }
        }

        if self.is_bad_basis_change() {
            return;
        }

        // Any pivoting is numerically acceptable, so perform update
        self.update();
        // Force rebuild if there are no infeasibilities in phase 1
        if *self.ekk.num_primal_infeasibilities == 0 && self.solve_phase == SOLVE_PHASE_1 {
            self.rebuild_reason = REBUILD_REASON_POSSIBLY_PHASE1_FEASIBLE;
        }
        debug_assert!(matches!(
            self.rebuild_reason,
            REBUILD_REASON_NO
                | REBUILD_REASON_POSSIBLY_PHASE1_FEASIBLE
                | REBUILD_REASON_PRIMAL_INFEASIBLE_IN_PRIMAL_SIMPLEX
                | REBUILD_REASON_SYNTHETIC_CLOCK_SAYS_INVERT
                | REBUILD_REASON_UPDATE_LIMIT_REACHED
        ));
    }

    /// HEkk::isBadBasisChange
    fn is_bad_basis_change(&mut self) -> bool {
        let bad = self.op_keep(Op::IsBadBasisChange, self.variable_in, self.row_out, self.rebuild_reason) != 0;
        // A bad basis change is made taboo
        self.any_taboo |= bad;
        bad
    }

    /// HEkkPrimal::chuzc
    fn chuzc(&mut self) {
        if self.done_next_chuzc {
            debug_assert!(self.use_hyper_chuzc);
        }
        if self.any_taboo {
            self.op(Op::ApplyTabooVariableIn);
        }
        if self.use_hyper_chuzc {
            // Perform hyper-sparse CHUZC and then check result using full
            // CHUZC
            if !self.done_next_chuzc {
                self.choose_column(true);
            }
            let hyper_sparse_variable_in = self.variable_in;
            self.choose_column(false);
            let work_dual = &self.ekk.work_dual;
            let mut hyper_sparse_measure = 0.0;
            if hyper_sparse_variable_in >= 0 {
                let d = work_dual[hyper_sparse_variable_in as usize];
                hyper_sparse_measure = d * d / self.edge_weight[hyper_sparse_variable_in as usize];
            }
            let mut measure = 0.0;
            if self.variable_in >= 0 {
                let d = work_dual[self.variable_in as usize];
                measure = d * d / self.edge_weight[self.variable_in as usize];
            }
            let abs_measure_error = (hyper_sparse_measure - measure).abs();
            debug_assert!(abs_measure_error <= 1e-12);
            self.variable_in = hyper_sparse_variable_in;
        } else {
            self.choose_column(false);
        }
        if self.any_taboo {
            self.op(Op::UnapplyTabooVariableIn);
        }
    }

    /// HEkkPrimal::chooseColumn
    fn choose_column(&mut self, hyper_sparse: bool) {
        debug_assert!(!hyper_sparse || !self.done_next_chuzc);
        let mut best_measure = 0.0;
        self.variable_in = -1;
        let num_nonbasic_free_col = self.nonbasic_free_col_set.count();
        let tolerance = self.dual_feasibility_tolerance;
        if hyper_sparse {
            if !self.initialise_hyper_chuzc {
                self.hyper_choose_column();
            }
            if self.initialise_hyper_chuzc {
                self.num_hyper_chuzc_candidates = 0;
                let work_dual = &self.ekk.work_dual;
                let nonbasic_move = &self.ekk.nonbasic_move;
                for &i_col in self.nonbasic_free_col_set.entries() {
                    let i_col = i_col as usize;
                    let dual_infeasibility = work_dual[i_col].abs();
                    if dual_infeasibility > tolerance {
                        let measure = dual_infeasibility * dual_infeasibility / self.edge_weight[i_col];
                        add_to_decreasing_heap(
                            &mut self.num_hyper_chuzc_candidates,
                            MAX_NUM_HYPER_CHUZC_CANDIDATES,
                            &mut self.hyper_chuzc_measure,
                            &mut self.hyper_chuzc_candidate,
                            measure,
                            i_col as i32,
                        );
                    }
                }
                // Now look at other columns
                for i_col in 0..self.num_tot {
                    let dual_infeasibility = -(nonbasic_move[i_col] as f64) * work_dual[i_col];
                    if dual_infeasibility > tolerance {
                        let measure = dual_infeasibility * dual_infeasibility / self.edge_weight[i_col];
                        add_to_decreasing_heap(
                            &mut self.num_hyper_chuzc_candidates,
                            MAX_NUM_HYPER_CHUZC_CANDIDATES,
                            &mut self.hyper_chuzc_measure,
                            &mut self.hyper_chuzc_candidate,
                            measure,
                            i_col as i32,
                        );
                    }
                }
                // Sort the heap
                sort_decreasing_heap(
                    self.num_hyper_chuzc_candidates,
                    &mut self.hyper_chuzc_measure,
                    &mut self.hyper_chuzc_candidate,
                );
                self.initialise_hyper_chuzc = false;
                // Choose the first entry - if there is one
                if self.num_hyper_chuzc_candidates != 0 {
                    self.variable_in = self.hyper_chuzc_candidate[1];
                    self.max_hyper_chuzc_non_candidate_measure =
                        self.hyper_chuzc_measure[self.num_hyper_chuzc_candidates as usize];
                }
            }
        } else {
            let work_dual = &self.ekk.work_dual[..self.num_tot];
            let nonbasic_move = &self.ekk.nonbasic_move[..self.num_tot];
            let edge_weight = &self.edge_weight[..self.num_tot];
            let mut variable_in = -1;
            // Choose any attractive nonbasic free column
            if num_nonbasic_free_col != 0 {
                for &i_col in self.nonbasic_free_col_set.entries() {
                    let i_col = i_col as usize;
                    let dual_infeasibility = work_dual[i_col].abs();
                    if dual_infeasibility > tolerance
                        && dual_infeasibility * dual_infeasibility > best_measure * edge_weight[i_col]
                    {
                        variable_in = i_col as i32;
                        best_measure = dual_infeasibility * dual_infeasibility / edge_weight[i_col];
                    }
                }
            }
            // Now look at other columns
            for i_col in 0..self.num_tot {
                let dual_infeasibility = -(nonbasic_move[i_col] as f64) * work_dual[i_col];
                if dual_infeasibility > tolerance
                    && dual_infeasibility * dual_infeasibility > best_measure * edge_weight[i_col]
                {
                    variable_in = i_col as i32;
                    best_measure = dual_infeasibility * dual_infeasibility / edge_weight[i_col];
                }
            }
            self.variable_in = variable_in;
        }
    }

    /// HEkkPrimal::useVariableIn
    fn use_variable_in(&mut self) -> bool {
        let variable_in = self.variable_in as usize;
        let updated_theta_dual = self.ekk.work_dual[variable_in];
        // Determine the move direction - can't use
        // nonbasicMove_[variable_in] due to free columns
        self.move_in = if updated_theta_dual > 0.0 { -1 } else { 1 };
        // Unless the variable is free, nonbasicMove[variable_in] should be
        // the same as move_in
        debug_assert!(
            self.ekk.nonbasic_move[variable_in] == 0 || self.ekk.nonbasic_move[variable_in] as i32 == self.move_in
        );
        // FTRAN: compute pivot column
        let ekk = &mut self.ekk;
        self.col_aq.with(|v| ekk.pivot_column_ftran(variable_in, v));
        // Compute the dual for the pivot column and compare it with the
        // updated value
        let computed_theta_dual = compute_dual_for_tableau_column(
            self.ekk.work_cost,
            self.ekk.basic_index,
            variable_in,
            &self.col_aq.index[..self.col_aq.count as usize],
            &self.col_aq.array,
        );
        // Feed in the computed dual value
        self.ekk.work_dual[variable_in] = computed_theta_dual;
        // Reassign theta_dual to be the computed value
        self.theta_dual = computed_theta_dual;
        // Determine whether theta_dual is too small or has changed sign
        let theta_dual_small = self.theta_dual.abs() <= self.dual_feasibility_tolerance;
        let theta_dual_sign_error = updated_theta_dual * computed_theta_dual <= 0.0;
        // If theta_dual is small, then it's no longer a dual infeasibility,
        // so reduce the number of dual infeasibilities
        if theta_dual_small {
            *self.ekk.num_dual_infeasibilities -= 1;
        }
        if theta_dual_small || theta_dual_sign_error {
            // The computed dual is small or has a sign error, so don't use it
            self.log(
                Log::DontUseVariableIn,
                &[
                    self.variable_in,
                    *self.ekk.iteration_count,
                    *self.ekk.update_count,
                    theta_dual_small as i32,
                    theta_dual_sign_error as i32,
                ],
                &[computed_theta_dual, updated_theta_dual],
            );
            // If a significant computed dual has sign error, consider
            // reinverting
            if !theta_dual_small && *self.ekk.update_count > 0 {
                self.rebuild_reason = REBUILD_REASON_POSSIBLY_SINGULAR_BASIS;
            }
            self.hyper_choose_column_clear();
            return false;
        }
        true
    }

    /// HEkkPrimal::phase1ChooseRow
    fn phase1_choose_row(&mut self) {
        let num_row = self.num_row as i32;
        let tol = self.primal_feasibility_tolerance;
        let (base_lower, base_upper, base_value) = (&self.ekk.base_lower, &self.ekk.base_upper, &self.ekk.base_value);
        // Collect phase 1 theta lists
        let update_count = *self.ekk.update_count;
        let d_pivot_tol = if update_count < 10 {
            1e-9
        } else if update_count < 20 {
            1e-8
        } else {
            1e-7
        };
        let move_in = self.move_in as f64;
        self.ph1_sorter_r.clear();
        self.ph1_sorter_t.clear();
        let col_aq = &self.col_aq;
        for &i_row in &col_aq.index[..col_aq.count as usize] {
            let r = i_row as usize;
            let d_alpha = col_aq.array[r] * move_in;
            // When the basic variable x[i] decrease
            if d_alpha > d_pivot_tol {
                // Whether it can become feasible by going below its upper
                // bound
                if base_value[r] > base_upper[r] + tol {
                    let d_feas_theta = (base_value[r] - base_upper[r] - tol) / d_alpha;
                    self.ph1_sorter_r.push((d_feas_theta, i_row));
                    self.ph1_sorter_t.push((d_feas_theta, i_row));
                }
                // Whether it can become infeasible (again) by going below
                // its lower bound
                if base_value[r] > base_lower[r] - tol && base_lower[r] > -K_HIGHS_INF {
                    let d_relax_theta = (base_value[r] - base_lower[r] + tol) / d_alpha;
                    let d_tight_theta = (base_value[r] - base_lower[r]) / d_alpha;
                    self.ph1_sorter_r.push((d_relax_theta, i_row - num_row));
                    self.ph1_sorter_t.push((d_tight_theta, i_row - num_row));
                }
            }
            // When the basic variable x[i] increase
            if d_alpha < -d_pivot_tol {
                // Whether it can become feasible by going above its lower
                // bound
                if base_value[r] < base_lower[r] - tol {
                    let d_feas_theta = (base_value[r] - base_lower[r] + tol) / d_alpha;
                    self.ph1_sorter_r.push((d_feas_theta, i_row - num_row));
                    self.ph1_sorter_t.push((d_feas_theta, i_row - num_row));
                }
                // Whether it can become infeasible (again) by going above
                // its upper bound
                if base_value[r] < base_upper[r] + tol && base_upper[r] < K_HIGHS_INF {
                    let d_relax_theta = (base_value[r] - base_upper[r] - tol) / d_alpha;
                    let d_tight_theta = (base_value[r] - base_upper[r]) / d_alpha;
                    self.ph1_sorter_r.push((d_relax_theta, i_row));
                    self.ph1_sorter_t.push((d_tight_theta, i_row));
                }
            }
        }

        // When there are no candidates at all, we can leave it here
        if self.ph1_sorter_r.is_empty() {
            self.row_out = NO_ROW_CHOSEN;
            self.variable_out = -1;
            return;
        }

        // Now sort the relaxed theta to find the final break point. The
        // pairs are ordered as std::pair<double, HighsInt>
        let pair_order = |a: &(f64, i32), b: &(f64, i32)| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1))
        };
        let row_of = |index: i32| (if index >= 0 { index } else { index + num_row }) as usize;
        self.ph1_sorter_r.sort_unstable_by(pair_order);
        let mut d_max_theta = self.ph1_sorter_r[0].0;
        let mut d_gradient = self.theta_dual.abs();
        for &(d_my_theta, index) in &self.ph1_sorter_r {
            d_gradient -= col_aq.array[row_of(index)].abs();
            // Stop when the gradient start to decrease
            if d_gradient <= 0.0 {
                break;
            }
            d_max_theta = d_my_theta;
        }

        // Find out the biggest possible alpha for pivot
        self.ph1_sorter_t.sort_unstable_by(pair_order);
        let mut d_max_alpha = 0.0;
        let mut i_last = self.ph1_sorter_t.len();
        for (i, &(d_my_theta, index)) in self.ph1_sorter_t.iter().enumerate() {
            let d_abs_alpha = col_aq.array[row_of(index)].abs();
            // Stop when the theta is too large
            if d_my_theta > d_max_theta {
                i_last = i;
                break;
            }
            // Update the maximal possible alpha
            if d_max_alpha < d_abs_alpha {
                d_max_alpha = d_abs_alpha;
            }
        }

        // Finally choose a pivot with good enough alpha, working backwards
        self.row_out = NO_ROW_CHOSEN;
        self.variable_out = -1;
        self.move_out = 0;
        for i in (1..=i_last).rev() {
            let index = self.ph1_sorter_t[i - 1].1;
            let i_row = row_of(index);
            let d_abs_alpha = col_aq.array[i_row].abs();
            if d_abs_alpha > d_max_alpha * 0.1 {
                self.row_out = i_row as i32;
                self.move_out = if index >= 0 { 1 } else { -1 };
                break;
            }
        }
    }

    /// HEkkPrimal::chooseRow
    fn choose_row(&mut self) {
        let (base_lower, base_upper, base_value) = (&self.ekk.base_lower, &self.ekk.base_upper, &self.ekk.base_value);
        let tol = self.primal_feasibility_tolerance;
        // Initialize
        self.row_out = NO_ROW_CHOSEN;
        // Choose row pass 1
        let update_count = *self.ekk.update_count;
        let alpha_tol = if update_count < 10 {
            1e-9
        } else if update_count < 20 {
            1e-8
        } else {
            1e-7
        };
        let move_in = self.move_in as f64;
        let col_aq = &self.col_aq;
        let index = &col_aq.index[..col_aq.count as usize];
        let mut relax_theta = 1e100;
        for &i_row in index {
            let r = i_row as usize;
            let alpha = col_aq.array[r] * move_in;
            if alpha > alpha_tol {
                let relax_space = base_value[r] - base_lower[r] + tol;
                if relax_space < relax_theta * alpha {
                    relax_theta = relax_space / alpha;
                }
            } else if alpha < -alpha_tol {
                let relax_space = base_value[r] - base_upper[r] - tol;
                if relax_space > relax_theta * alpha {
                    relax_theta = relax_space / alpha;
                }
            }
        }
        // Choose row pass 2
        let mut best_alpha = 0.0;
        let mut row_out = NO_ROW_CHOSEN;
        for &i_row in index {
            let r = i_row as usize;
            let alpha = col_aq.array[r] * move_in;
            if alpha > alpha_tol {
                // Positive pivotal column entry
                let tight_space = base_value[r] - base_lower[r];
                if tight_space < relax_theta * alpha && best_alpha < alpha {
                    best_alpha = alpha;
                    row_out = i_row;
                }
            } else if alpha < -alpha_tol {
                // Negative pivotal column entry
                let tight_space = base_value[r] - base_upper[r];
                if tight_space > relax_theta * alpha && best_alpha < -alpha {
                    best_alpha = -alpha;
                    row_out = i_row;
                }
            }
        }
        self.row_out = row_out;
    }

    /// HEkkPrimal::considerBoundSwap
    fn consider_bound_swap(&mut self) {
        let e = &self.ekk;
        // Compute the primal theta and see if we should have done a bound
        // flip instead
        if self.row_out == NO_ROW_CHOSEN {
            debug_assert!(self.solve_phase == SOLVE_PHASE_2);
            // No binding ratio in CHUZR, so flip or unbounded
            self.theta_primal = self.move_in as f64 * K_HIGHS_INF;
            self.move_out = 0;
        } else {
            debug_assert!(self.row_out >= 0);
            let row_out = self.row_out as usize;
            // Determine the step to the leaving bound
            self.alpha_col = self.col_aq.array[row_out];
            // In Phase 1, move_out depends on whether the leaving variable
            // is becoming feasible - moves up to lower (down to upper) - or
            // remaining feasible - moves down to lower (up to upper) - so
            // can't be set so easily as in phase 2
            if self.solve_phase == SOLVE_PHASE_2 {
                self.move_out = if self.alpha_col * self.move_in as f64 > 0.0 { -1 } else { 1 };
            }
            self.theta_primal = if self.move_out == 1 {
                (e.base_value[row_out] - e.base_upper[row_out]) / self.alpha_col
            } else {
                (e.base_value[row_out] - e.base_lower[row_out]) / self.alpha_col
            };
            debug_assert!(self.theta_primal > -K_HIGHS_INF && self.theta_primal < K_HIGHS_INF);
        }

        // Look to see if there is a bound flip
        let mut flipped = false;
        let variable_in = self.variable_in as usize;
        let lower_in = e.work_lower[variable_in];
        let upper_in = e.work_upper[variable_in];
        self.value_in = e.work_value[variable_in] + self.theta_primal;
        if self.move_in > 0 {
            if self.value_in > upper_in + self.primal_feasibility_tolerance {
                flipped = true;
                self.row_out = NO_ROW_CHOSEN;
                self.value_in = upper_in;
                self.theta_primal = upper_in - lower_in;
            }
        } else if self.value_in < lower_in - self.primal_feasibility_tolerance {
            flipped = true;
            self.row_out = NO_ROW_CHOSEN;
            self.value_in = lower_in;
            self.theta_primal = lower_in - upper_in;
        }
        let pivot_or_flipped = self.row_out >= 0 || flipped;
        if self.solve_phase == SOLVE_PHASE_2 && !pivot_or_flipped {
            // Check for possible unboundedness
            self.rebuild_reason = REBUILD_REASON_POSSIBLY_PRIMAL_UNBOUNDED;
            return;
        }
        // Check for possible error
        debug_assert!(pivot_or_flipped);
        debug_assert!(flipped == (self.row_out == NO_ROW_CHOSEN));
    }

    /// HEkkPrimal::assessPivot
    fn assess_pivot(&mut self) {
        debug_assert!(self.row_out >= 0);
        let row_out = self.row_out as usize;
        // Record the pivot entry
        self.alpha_col = self.col_aq.array[row_out];
        self.variable_out = self.ekk.basic_index[row_out];
        // Compute the tableau row: unit BTRAN and PRICE
        let ekk = &mut self.ekk;
        let row_ap = &mut self.row_ap;
        self.row_ep.with(|ep| {
            ekk.unit_btran(row_out, ep);
            row_ap.with(|ap| ekk.tableau_row_price(ep, ap));
        });
        // Checks row-wise pivot against column-wise pivot for numerical
        // trouble
        self.update_verify();
    }

    /// HEkkPrimal::updateVerify
    fn update_verify(&mut self) {
        let numerical_trouble_tolerance = 1e-7;
        let abs_alpha_from_col = self.alpha_col.abs();
        let from_row = self.variable_in as usize >= self.num_col;
        self.alpha_row = if from_row {
            self.row_ep.array[self.variable_in as usize - self.num_col]
        } else {
            self.row_ap.array[self.variable_in as usize]
        };
        let abs_alpha_from_row = self.alpha_row.abs();
        let abs_alpha_diff = (abs_alpha_from_col - abs_alpha_from_row).abs();
        let min_abs_alpha = abs_alpha_from_col.min(abs_alpha_from_row);
        self.numerical_trouble = abs_alpha_diff / min_abs_alpha;
        if self.numerical_trouble > numerical_trouble_tolerance {
            self.log(
                Log::NumericalCheck,
                &[*self.ekk.iteration_count, from_row as i32],
                &[self.alpha_col, self.alpha_row, abs_alpha_diff, self.numerical_trouble],
            );
        }
        // Reinvert if the relative difference is large enough, and updates
        // have been performed
        if self.numerical_trouble > 1e-7 && *self.ekk.update_count > 0 {
            self.rebuild_reason = REBUILD_REASON_POSSIBLY_SINGULAR_BASIS;
        }
    }

    /// HEkkPrimal::update
    fn update(&mut self) {
        // Perform update operations that are independent of phase
        debug_assert!(self.rebuild_reason == 0);
        let flipped = self.row_out < 0;
        if flipped {
            let variable_in = self.variable_in as usize;
            self.variable_out = self.variable_in;
            self.alpha_col = 0.0;
            self.numerical_trouble = 0.0;
            self.ekk.work_value[variable_in] = self.value_in;
            debug_assert!(self.ekk.nonbasic_move[variable_in] as i32 == self.move_in);
            self.ekk.nonbasic_move[variable_in] = -self.move_in as i8;
        } else {
            // Adjust perturbation if leaving equation
            self.adjust_perturbed_equation_out();
        }

        // Start hyper-sparse CHUZC, that takes place through phase1Update()
        self.hyper_choose_column_start();

        if self.solve_phase == SOLVE_PHASE_1 {
            // Update primal values
            self.phase1_update_primal();
            // Update the duals with respect to feasibility changes
            self.basic_feasibility_change_update_dual();
            // For hyper-sparse CHUZC, analyse the duals that have just
            // changed
            self.hyper_choose_column_basic_feasibility_change();
        } else {
            // Update primal values, and identify any infeasibilities
            self.phase2_update_primal();
        }

        debug_assert!(
            self.rebuild_reason == REBUILD_REASON_NO
                || self.rebuild_reason == REBUILD_REASON_PRIMAL_INFEASIBLE_IN_PRIMAL_SIMPLEX
        );

        if flipped {
            *self.ekk.primal_bound_swap += 1;
            self.invalidate_dual_infeasibility_record();
            self.iteration_analysis();
            self.num_flip_since_rebuild += 1;
            // Update the synthetic clock for UPDATE
            *self.ekk.total_synthetic_tick += self.col_aq.synthetic_tick;
            return;
        }

        debug_assert!(self.row_out >= 0);
        let row_out = self.row_out as usize;
        let variable_in = self.variable_in as usize;
        let variable_out = self.variable_out as usize;

        // Now set the value of the entering variable
        self.ekk.base_value[row_out] = self.value_in;
        // Consider whether the entering value is feasible and, if not, take
        // action
        self.consider_infeasible_value_in();

        // Update the dual values
        self.theta_dual = self.ekk.work_dual[variable_in];
        self.update_dual();

        // Update any non-unit primal edge weights
        if self.edge_weight_mode == EDGE_WEIGHT_DEVEX {
            self.update_devex();
        } else if self.edge_weight_mode == EDGE_WEIGHT_STEEPEST_EDGE {
            self.debug_primal_steepest_edge_weights(HIGHS_DEBUG_LEVEL_COSTLY);
            self.update_primal_steepest_edge_weights();
        }

        // If entering column was nonbasic free, remove it from the set
        self.remove_nonbasic_free_column();

        // For hyper-sparse CHUZC, analyse the duals and weights that have
        // just changed
        self.hyper_choose_column_dual_change();

        if self.ekk.status.has_dual_steepest_edge_weights {
            self.update_dual_steepest_edge_weights();
        }
        // Perform pivoting
        //
        // Transform the vectors used in updateFactor if the simplex NLA
        // involves scaling
        {
            let ekk = &mut self.ekk;
            let row_ep = &mut self.row_ep;
            self.col_aq.with(|aq| row_ep.with(|ep| ekk.transform_for_update(aq, ep, variable_in, row_out)));
        }
        // Update the sets of indices of basic and nonbasic variables
        self.ekk.update_pivots(variable_in, row_out, self.move_out);
        self.ekk.status.has_invert = false;
        self.ekk.status.has_fresh_invert = false;
        self.ekk.status.has_fresh_rebuild = false;
        self.op_keep(Op::BasisChanged, 0, 0, 0);
        // Update the invertible representation of the basis matrix
        {
            let ekk = &mut self.ekk;
            let row_ep = &mut self.row_ep;
            let rebuild_reason = &mut self.rebuild_reason;
            self.col_aq
                .with(|aq| row_ep.with(|ep| ekk.update_factor(aq, ep, row_out as i32, rebuild_reason)));
        }
        self.ekk.status.has_invert = true;
        if self.edge_weight_mode == EDGE_WEIGHT_STEEPEST_EDGE {
            self.debug_primal_steepest_edge_weights(HIGHS_DEBUG_LEVEL_COSTLY);
        }
        // Update the row-wise representation of the nonbasic columns
        self.ekk.update_matrix(variable_in, variable_out);
        if *self.ekk.update_count >= *self.ekk.update_limit {
            self.rebuild_reason = REBUILD_REASON_UPDATE_LIMIT_REACHED;
        }

        // Update the iteration count
        *self.ekk.iteration_count += 1;

        // Reset the devex when there are too many errors
        if self.edge_weight_mode == EDGE_WEIGHT_DEVEX && self.num_bad_devex_weight > ALLOWED_NUM_BAD_DEVEX_WEIGHT {
            self.initialise_devex_framework();
        }

        // Report on the iteration
        self.iteration_analysis();

        // Update the synthetic clock for UPDATE
        *self.ekk.total_synthetic_tick += self.col_aq.synthetic_tick;
        *self.ekk.total_synthetic_tick += self.row_ep.synthetic_tick;

        // Perform hyper-sparse CHUZC
        self.hyper_choose_column();
    }

    /// HEkkPrimal::iterationAnalysis, when the iteration is reported
    fn iteration_analysis(&mut self) {
        self.analysed_phase = self.solve_phase;
        if self.ekk.iteration_report {
            self.report(REPORT_ITERATION, 0);
        }
    }

    /// HEkk::invalidateDualInfeasibilityRecord
    fn invalidate_dual_infeasibility_record(&mut self) {
        *self.ekk.num_dual_infeasibilities = -1;
        *self.ekk.max_dual_infeasibility = K_HIGHS_INF;
        *self.ekk.sum_dual_infeasibilities = K_HIGHS_INF;
    }

    /// HEkk::invalidatePrimalMaxSumInfeasibilityRecord
    fn invalidate_primal_max_sum_infeasibility_record(&mut self) {
        *self.ekk.max_primal_infeasibility = K_HIGHS_INF;
        *self.ekk.sum_primal_infeasibilities = K_HIGHS_INF;
    }

    // ---- Hyper-sparse CHUZC (not used: use_hyper_chuzc is false) ----

    /// HEkkPrimal::hyperChooseColumn
    fn hyper_choose_column(&mut self) {
        if !self.use_hyper_chuzc || self.initialise_hyper_chuzc {
            return;
        }
        let (nonbasic_move, nonbasic_flag, work_dual) =
            (&self.ekk.nonbasic_move, &self.ekk.nonbasic_flag, &self.ekk.work_dual);
        let mut best_measure = self.max_changed_measure_value;
        self.variable_in = -1;
        if self.max_changed_measure_column >= 0 {
            // Use max_changed_measure_column if it is well defined and has
            // nonzero dual. It may have been zeroed because it is taboo
            if work_dual[self.max_changed_measure_column as usize] != 0.0 {
                self.variable_in = self.max_changed_measure_column;
            }
        }
        let consider_nonbasic_free_column = self.nonbasic_free_col_set.count() != 0;
        for i_entry in 1..=self.num_hyper_chuzc_candidates as usize {
            let i_col = self.hyper_chuzc_candidate[i_entry] as usize;
            if nonbasic_flag[i_col] == 0 {
                debug_assert!(nonbasic_move[i_col] == 0);
                continue;
            }
            // Assess any dual infeasibility
            let mut dual_infeasibility = -(nonbasic_move[i_col] as f64) * work_dual[i_col];
            if consider_nonbasic_free_column && self.nonbasic_free_col_set.contains(i_col as i32) {
                dual_infeasibility = work_dual[i_col].abs();
            }
            if dual_infeasibility > self.dual_feasibility_tolerance
                && dual_infeasibility * dual_infeasibility > best_measure * self.edge_weight[i_col]
            {
                best_measure = dual_infeasibility * dual_infeasibility / self.edge_weight[i_col];
                self.variable_in = i_col as i32;
            }
        }
        if self.variable_in != self.max_changed_measure_column {
            self.max_hyper_chuzc_non_candidate_measure =
                self.max_changed_measure_value.max(self.max_hyper_chuzc_non_candidate_measure);
        }
        if best_measure >= self.max_hyper_chuzc_non_candidate_measure {
            // Candidate is at least as good as any unknown column, so
            // accept it
            self.done_next_chuzc = true;
        } else {
            // Candidate isn't as good as best unknown column, so do a full
            // CHUZC
            debug_assert!(!self.done_next_chuzc);
            self.done_next_chuzc = false;
            self.initialise_hyper_chuzc = true;
        }
    }

    /// HEkkPrimal::hyperChooseColumnStart
    fn hyper_choose_column_start(&mut self) {
        self.max_changed_measure_value = 0.0;
        self.max_changed_measure_column = -1;
        self.done_next_chuzc = false;
    }

    /// HEkkPrimal::hyperChooseColumnClear
    fn hyper_choose_column_clear(&mut self) {
        self.initialise_hyper_chuzc = self.use_hyper_chuzc;
        self.max_hyper_chuzc_non_candidate_measure = -1.0;
        self.done_next_chuzc = false;
    }

    /// HEkkPrimal::hyperChooseColumnChangedInfeasibility
    fn hyper_choose_column_changed_infeasibility(&mut self, infeasibility: f64, i_col: usize) {
        let w = self.edge_weight[i_col];
        if infeasibility * infeasibility > self.max_changed_measure_value * w {
            self.max_hyper_chuzc_non_candidate_measure =
                self.max_changed_measure_value.max(self.max_hyper_chuzc_non_candidate_measure);
            self.max_changed_measure_value = infeasibility * infeasibility / w;
            self.max_changed_measure_column = i_col as i32;
        } else if infeasibility * infeasibility > self.max_hyper_chuzc_non_candidate_measure * w {
            self.max_hyper_chuzc_non_candidate_measure = infeasibility * infeasibility / w;
        }
    }

    /// The nonbasic dual infeasibility of a variable
    #[inline]
    fn move_dual_infeasibility(&self, i_col: usize) -> f64 {
        -(self.ekk.nonbasic_move[i_col] as f64) * self.ekk.work_dual[i_col]
    }

    /// The changed infeasibilities of the variables of a vector of
    /// values for the columns (`rows` false) or rows
    fn hyper_choose_column_vector_changes(&mut self, v: &OwnedHVec, rows: bool) {
        let (dim, offset) = if rows { (self.num_row, self.num_col) } else { (self.num_col, 0) };
        let (use_indices, to_entry) = sparse_loop_style(v.count, dim);
        for i_entry in 0..to_entry {
            let i_col = offset + if use_indices { v.index[i_entry] as usize } else { i_entry };
            let dual_infeasibility = self.move_dual_infeasibility(i_col);
            if dual_infeasibility > self.dual_feasibility_tolerance {
                self.hyper_choose_column_changed_infeasibility(dual_infeasibility, i_col);
            }
        }
    }

    /// The changed infeasibilities of the nonbasic free columns
    fn hyper_choose_column_free_changes(&mut self) {
        for k in 0..self.nonbasic_free_col_set.count() {
            let i_col = self.nonbasic_free_col_set.entries()[k] as usize;
            let dual_infeasibility = self.ekk.work_dual[i_col].abs();
            if dual_infeasibility > self.dual_feasibility_tolerance {
                self.hyper_choose_column_changed_infeasibility(dual_infeasibility, i_col);
            }
        }
    }

    /// HEkkPrimal::hyperChooseColumnBasicFeasibilityChange
    fn hyper_choose_column_basic_feasibility_change(&mut self) {
        if !self.use_hyper_chuzc {
            return;
        }
        // Any nonbasic free columns will be handled explicitly in
        // hyperChooseColumnDualChange, so only look at them here if not
        // flipping
        let row = std::mem::replace(&mut self.row_basic_feasibility_change, OwnedHVec::new(0));
        let col = std::mem::replace(&mut self.col_basic_feasibility_change, OwnedHVec::new(0));
        self.hyper_choose_column_vector_changes(&row, false);
        self.hyper_choose_column_vector_changes(&col, true);
        self.row_basic_feasibility_change = row;
        self.col_basic_feasibility_change = col;
        if self.row_out < 0 {
            self.hyper_choose_column_free_changes();
        }
    }

    /// HEkkPrimal::hyperChooseColumnDualChange
    fn hyper_choose_column_dual_change(&mut self) {
        if !self.use_hyper_chuzc {
            return;
        }
        // Look at changes in the columns and rows, and in any nonbasic free
        // columns, and assess any dual infeasibility
        let row_ap = std::mem::replace(&mut self.row_ap, OwnedHVec::new(0));
        let row_ep = std::mem::replace(&mut self.row_ep, OwnedHVec::new(0));
        self.hyper_choose_column_vector_changes(&row_ap, false);
        self.hyper_choose_column_vector_changes(&row_ep, true);
        self.row_ap = row_ap;
        self.row_ep = row_ep;
        self.hyper_choose_column_free_changes();
        // Assess any dual infeasibility for the leaving column - should be
        // dual feasible!
        let i_col = self.variable_out as usize;
        let dual_infeasibility = self.move_dual_infeasibility(i_col);
        if dual_infeasibility > self.dual_feasibility_tolerance {
            self.log(Log::LeavingDualInfeasibility, &[], &[dual_infeasibility]);
            debug_assert!(dual_infeasibility <= self.dual_feasibility_tolerance);
            self.hyper_choose_column_changed_infeasibility(dual_infeasibility, i_col);
        }
    }

    // ---- Updates ----

    /// HEkkPrimal::updateDual
    fn update_dual(&mut self) {
        debug_assert!(self.alpha_col != 0.0);
        debug_assert!(self.row_out >= 0);
        let num_col = self.num_col;
        let work_dual = &mut *self.ekk.work_dual;
        // Update the duals
        self.theta_dual = work_dual[self.variable_in as usize] / self.alpha_col;
        let theta_dual = self.theta_dual;
        let row_ap = &self.row_ap;
        for &i_col in &row_ap.index[..row_ap.count as usize] {
            let i_col = i_col as usize;
            work_dual[i_col] = (-theta_dual).mul_add_c(row_ap.array[i_col], work_dual[i_col]);
        }
        let row_ep = &self.row_ep;
        for &i_row in &row_ep.index[..row_ep.count as usize] {
            let i_col = i_row as usize + num_col;
            work_dual[i_col] = (-theta_dual).mul_add_c(row_ep.array[i_row as usize], work_dual[i_col]);
        }
        // Dual for the pivot
        work_dual[self.variable_in as usize] = 0.0;
        work_dual[self.variable_out as usize] = -theta_dual;

        self.invalidate_dual_infeasibility_record();
        // After dual update in primal simplex the dual objective value is
        // not known
        self.ekk.status.has_dual_objective_value = false;
    }

    /// The phase 1 cost of a basic variable violating a bound
    #[inline]
    fn phase1_cost(bound_violated: i32, base: f64, random_value: f64) -> f64 {
        let mut cost = bound_violated as f64;
        if base != 0.0 {
            cost *= base.mul_add_c(random_value, 1.0);
        }
        cost
    }

    /// HEkkPrimal::phase1ComputeDual
    fn phase1_compute_dual(&mut self) {
        let (num_col, num_row, num_tot) = (self.num_col, self.num_row, self.num_tot);
        let tol = self.primal_feasibility_tolerance;
        let mut buffer = OwnedHVec::new(num_row as i32);
        buffer.clear();
        buffer.count = 0;
        let e = &mut self.ekk;
        // Accumulate costs for checking
        e.work_cost[..num_tot].fill(0.0);
        // Zero the dual values
        e.work_dual[..num_tot].fill(0.0);
        // Determine the base value for cost perturbation
        let base = e.primal_simplex_phase1_cost_perturbation_multiplier * 5e-7;
        for i_row in 0..num_row {
            let bound_violated = bound_violated(e.base_value[i_row], e.base_lower[i_row], e.base_upper[i_row], tol);
            if bound_violated == 0 {
                continue;
            }
            buffer.array[i_row] = Self::phase1_cost(bound_violated, base, e.num_tot_random_value[i_row]);
            buffer.index[buffer.count as usize] = i_row as i32;
            buffer.count += 1;
        }
        if buffer.count <= 0 {
            // Strange, should be a non-trivial RHS
            debug_assert!(buffer.count > 0);
            return;
        }
        for i_row in 0..num_row {
            e.work_cost[e.basic_index[i_row] as usize] = buffer.array[i_row];
        }
        // Full BTRAN
        buffer.with(|v| e.full_btran(v));
        // Full PRICE
        let mut buffer_long = OwnedHVec::new(num_col as i32);
        buffer.with(|c| buffer_long.with(|r| e.full_price(c, r)));
        for i_col in 0..num_col {
            e.work_dual[i_col] = -(e.nonbasic_flag[i_col] as f64) * buffer_long.array[i_col];
        }
        for i_row in 0..num_row {
            let i_col = num_col + i_row;
            e.work_dual[i_col] = -(e.nonbasic_flag[i_col] as f64) * buffer.array[i_row];
        }
    }

    /// HEkkPrimal::phase1UpdatePrimal
    fn phase1_update_primal(&mut self) {
        let num_col = self.num_col;
        let tol = self.primal_feasibility_tolerance;
        self.col_basic_feasibility_change.clear();
        // Update basic primal values, identifying all the feasibility
        // changes giving a value to col_basic_feasibility_change so that
        // the duals can be updated.
        let e = &mut self.ekk;
        // Determine the base value for cost perturbation
        let base = e.primal_simplex_phase1_cost_perturbation_multiplier * 5e-7;
        let theta_primal = self.theta_primal;
        let col_aq = &self.col_aq;
        let change = &mut self.col_basic_feasibility_change;
        for &i_row in &col_aq.index[..col_aq.count as usize] {
            let r = i_row as usize;
            e.base_value[r] = (-theta_primal).mul_add_c(col_aq.array[r], e.base_value[r]);
            let i_col = e.basic_index[r] as usize;
            let was_cost = e.work_cost[i_col];
            let bound_violated = bound_violated(e.base_value[r], e.base_lower[r], e.base_upper[r], tol);
            let cost = Self::phase1_cost(bound_violated, base, e.num_tot_random_value[r]);
            e.work_cost[i_col] = cost;
            if was_cost != 0.0 {
                if cost == 0.0 {
                    *e.num_primal_infeasibilities -= 1;
                }
            } else if cost != 0.0 {
                *e.num_primal_infeasibilities += 1;
            }
            let delta_cost = cost - was_cost;
            if delta_cost != 0.0 {
                change.array[r] = delta_cost;
                change.index[change.count as usize] = i_row;
                change.count += 1;
                if i_col >= num_col {
                    e.work_dual[i_col] += delta_cost;
                }
            }
        }
        // Don't set baseValue[row_out] yet so that dual update due to
        // feasibility changes is done correctly
        self.invalidate_primal_max_sum_infeasibility_record();
    }

    /// HEkkPrimal::considerInfeasibleValueIn
    fn consider_infeasible_value_in(&mut self) {
        debug_assert!(self.row_out >= 0);
        let variable_in = self.variable_in as usize;
        let e = &mut self.ekk;
        // Determine the base value for cost perturbation
        let base = e.primal_simplex_phase1_cost_perturbation_multiplier * 5e-7;
        let lower = e.work_lower[variable_in];
        let upper = e.work_upper[variable_in];
        let bound_violated = bound_violated(self.value_in, lower, upper, self.primal_feasibility_tolerance);
        if bound_violated == 0 {
            return;
        }
        // The primal value of the entering variable is not feasible
        if self.solve_phase == SOLVE_PHASE_1 {
            *e.num_primal_infeasibilities += 1;
            let cost = Self::phase1_cost(bound_violated, base, e.num_tot_random_value[self.row_out as usize]);
            e.work_cost[variable_in] = cost;
            e.work_dual[variable_in] += cost;
        } else {
            // primal_correction_strategy is
            // kSimplexPrimalCorrectionStrategyAlways
            let random_value = e.num_tot_random_value[variable_in];
            if bound_violated > 0 {
                // Perturb the upper bound to accommodate the infeasibility
                let mut bound = e.work_upper[variable_in];
                let shift = self.shift_bound(false, variable_in, self.value_in, random_value, &mut bound);
                self.ekk.work_upper[variable_in] = bound;
                self.ekk.work_upper_shift[variable_in] += shift;
            } else {
                // Perturb the lower bound to accommodate the infeasibility
                let mut bound = e.work_lower[variable_in];
                let shift = self.shift_bound(true, variable_in, self.value_in, random_value, &mut bound);
                self.ekk.work_lower[variable_in] = bound;
                self.ekk.work_lower_shift[variable_in] += shift;
            }
            *self.ekk.bounds_perturbed = true;
        }
        self.invalidate_primal_max_sum_infeasibility_record();
    }

    /// Shift the bound of the basic variable in row `i_row` that its value
    /// violates
    fn shift_basic_bound(&mut self, i_row: usize, bound_violated: i32) -> f64 {
        let i_col = self.ekk.basic_index[i_row] as usize;
        let value = self.ekk.base_value[i_row];
        let random_value = self.ekk.num_tot_random_value[i_col];
        if bound_violated > 0 {
            // Perturb the upper bound to accommodate the infeasibility
            let mut bound = self.ekk.work_upper[i_col];
            let shift = self.shift_bound(false, i_col, value, random_value, &mut bound);
            self.ekk.work_upper[i_col] = bound;
            self.ekk.base_upper[i_row] = bound;
            self.ekk.work_upper_shift[i_col] += shift;
            shift
        } else {
            // Perturb the lower bound to accommodate the infeasibility
            let mut bound = self.ekk.work_lower[i_col];
            let shift = self.shift_bound(true, i_col, value, random_value, &mut bound);
            self.ekk.work_lower[i_col] = bound;
            self.ekk.base_lower[i_row] = bound;
            self.ekk.work_lower_shift[i_col] += shift;
            shift
        }
    }

    /// HEkkPrimal::phase2UpdatePrimal (primal_correction_strategy is
    /// kSimplexPrimalCorrectionStrategyAlways, so violated bounds are
    /// shifted)
    fn phase2_update_primal(&mut self) {
        let tol = self.primal_feasibility_tolerance;
        let theta_primal = self.theta_primal;
        let (use_col_indices, to_entry) = sparse_loop_style(self.col_aq.count, self.num_row);
        for i_entry in 0..to_entry {
            let i_row = if use_col_indices { self.col_aq.index[i_entry] as usize } else { i_entry };
            let e = &mut self.ekk;
            e.base_value[i_row] = (-theta_primal).mul_add_c(self.col_aq.array[i_row], e.base_value[i_row]);
            // Determine whether a bound is violated and take action
            let bound_violated = bound_violated(e.base_value[i_row], e.base_lower[i_row], e.base_upper[i_row], tol);
            if bound_violated == 0 {
                continue;
            }
            let bound_shift = self.shift_basic_bound(i_row, bound_violated);
            *self.ekk.bounds_shifted = true;
            debug_assert!(bound_shift > 0.0);
        }
        let e = &mut self.ekk;
        *e.updated_primal_objective_value =
            e.work_dual[self.variable_in as usize].mul_add_c(theta_primal, *e.updated_primal_objective_value);
    }

    /// HEkkPrimal::correctPrimal
    fn correct_primal(&mut self) -> bool {
        debug_assert!(self.solve_phase == SOLVE_PHASE_2);
        let tol = self.primal_feasibility_tolerance;
        let mut num_primal_correction = 0;
        let mut max_primal_correction: f64 = 0.0;
        let mut sum_primal_correction = 0.0;
        let mut num_primal_correction_skipped = 0;
        for i_row in 0..self.num_row {
            let e = &self.ekk;
            let bound_violated = bound_violated(e.base_value[i_row], e.base_lower[i_row], e.base_upper[i_row], tol);
            if bound_violated == 0 {
                continue;
            }
            if *self.ekk.allow_bound_perturbation {
                let bound_shift = self.shift_basic_bound(i_row, bound_violated);
                debug_assert!(bound_shift > 0.0);
                num_primal_correction += 1;
                max_primal_correction = if bound_shift > max_primal_correction { bound_shift } else { max_primal_correction };
                sum_primal_correction += bound_shift;
                *self.ekk.bounds_perturbed = true;
            } else {
                // Bound perturbation is not permitted
                num_primal_correction_skipped += 1;
            }
        }
        if num_primal_correction_skipped != 0 {
            self.log(Log::MissedBoundShifts, &[num_primal_correction_skipped], &[]);
            return false;
        }
        if max_primal_correction > 2.0 * self.max_max_primal_correction {
            self.log(Log::PrimalCorrections, &[num_primal_correction], &[max_primal_correction, sum_primal_correction]);
            self.max_max_primal_correction = max_primal_correction;
        }
        true
    }

    /// HEkkPrimal::basicFeasibilityChangeUpdateDual
    fn basic_feasibility_change_update_dual(&mut self) {
        // For basic logicals, the change in the basic cost will be a
        // component in col_basic_feasibility_change. This will lead to it
        // being subtracted from workDual in the loop below over the
        // nonzeros in col_basic_feasibility_change, so add it in now. For
        // basic structurals, there will be no corresponding component in
        // row_basic_feasibility_change, since only the nonbasic components
        // are computed. Hence, only add in the basic cost change for
        // logicals.
        self.basic_feasibility_change_btran();
        self.basic_feasibility_change_price();
        let num_col = self.num_col;
        let work_dual = &mut *self.ekk.work_dual;
        let row = &self.row_basic_feasibility_change;
        let (use_row_indices, to_entry) = sparse_loop_style(row.count, num_col);
        for i_entry in 0..to_entry {
            let i_col = if use_row_indices { row.index[i_entry] as usize } else { i_entry };
            work_dual[i_col] -= row.array[i_col];
        }
        let col = &self.col_basic_feasibility_change;
        let (use_col_indices, to_entry) = sparse_loop_style(col.count, self.num_row);
        for i_entry in 0..to_entry {
            let i_row = if use_col_indices { col.index[i_entry] as usize } else { i_entry };
            work_dual[num_col + i_row] -= col.array[i_row];
        }
        self.invalidate_dual_infeasibility_record();
    }

    /// HEkkPrimal::basicFeasibilityChangeBtran
    fn basic_feasibility_change_btran(&mut self) {
        let e = &mut self.ekk;
        let num_row = self.num_row;
        self.col_basic_feasibility_change.with(|v| {
            e.btran(v, *e.col_basic_feasibility_change_density);
            let local_density = v.count as f64 / num_row as f64;
            update_operation_result_density(local_density, e.col_basic_feasibility_change_density);
        });
    }

    /// HEkkPrimal::basicFeasibilityChangePrice
    fn basic_feasibility_change_price(&mut self) {
        let e = &mut self.ekk;
        let (num_col, num_row) = (self.num_col, self.num_row);
        let local_density = 1.0 * self.col_basic_feasibility_change.count as f64 / num_row as f64;
        let (use_col_price, use_row_price_w_switch) = choose_price_technique(e.price_strategy, local_density);
        let row = &mut self.row_basic_feasibility_change;
        row.clear();
        self.col_basic_feasibility_change.with(|col| {
            row.with(|row| {
                if use_col_price {
                    // Perform column-wise PRICE
                    e.price_by_column(col, row);
                } else if use_row_price_w_switch {
                    // Perform hyper-sparse row-wise PRICE, but switch if the
                    // density of row_basic_feasibility_change becomes
                    // extreme
                    e.price_by_row_with_switch(
                        col,
                        row,
                        *e.row_basic_feasibility_change_density,
                        0,
                        K_HYPER_PRICE_DENSITY,
                    );
                } else {
                    // Perform hyper-sparse row-wise PRICE
                    e.price_by_row_with_switch(col, row, -K_HIGHS_INF, 0, K_HIGHS_INF);
                }
            })
        });
        if use_col_price {
            // Column-wise PRICE computes components corresponding to basic
            // variables, so zero these by exploiting the fact that, for
            // basic variables, nonbasicFlag[*]=0
            for (x, &flag) in row.array[..num_col].iter_mut().zip(&*e.nonbasic_flag) {
                *x *= flag as f64;
            }
        }
        // Update the record of average row_basic_feasibility_change density
        let local_row_density = row.count as f64 / num_col as f64;
        update_operation_result_density(local_row_density, e.row_basic_feasibility_change_density);
    }

    // ---- Edge weights ----

    /// HEkkPrimal::initialiseDevexFramework
    fn initialise_devex_framework(&mut self) {
        let num_tot = self.num_tot;
        self.edge_weight.clear();
        self.edge_weight.resize(num_tot, 1.0);
        self.devex_index.clear();
        self.devex_index.extend(self.ekk.nonbasic_flag[..num_tot].iter().map(|&f| f as i32 * f as i32));
        self.num_devex_iterations = 0;
        self.num_bad_devex_weight = 0;
        self.hyper_choose_column_clear();
    }

    /// HEkkPrimal::updateDevex
    fn update_devex(&mut self) {
        let num_col = self.num_col;
        let basic_index = &self.ekk.basic_index;
        let devex_index = &self.devex_index;
        let edge_weight = &mut self.edge_weight;
        // Compute the pivot weight from the reference set
        let col_aq = &self.col_aq;
        let mut d_pivot_weight = 0.0;
        let (use_col_indices, to_entry) = sparse_loop_style(col_aq.count, self.num_row);
        // Clang interleaves this loop by 4 without contracting, and
        // contracts the remainder loop
        let unfused = interleaved_part(to_entry);
        for i_entry in 0..to_entry {
            let i_row = if use_col_indices { col_aq.index[i_entry] as usize } else { i_entry };
            let i_col = basic_index[i_row] as usize;
            let d_alpha = devex_index[i_col] as f64 * col_aq.array[i_row];
            d_pivot_weight = if i_entry < unfused {
                d_pivot_weight + d_alpha * d_alpha
            } else {
                d_alpha.mul_add_c(d_alpha, d_pivot_weight)
            };
        }
        let variable_in = self.variable_in as usize;
        d_pivot_weight += devex_index[variable_in] as f64;

        // Check if the saved weight is too large
        if edge_weight[variable_in] > BAD_DEVEX_WEIGHT_FACTOR * d_pivot_weight {
            self.num_bad_devex_weight += 1;
        }

        // Update the devex weight for all
        let d_pivot = col_aq.array[self.row_out as usize];
        d_pivot_weight /= d_pivot * d_pivot;

        let row_ap = &self.row_ap;
        for &i_col in &row_ap.index[..row_ap.count as usize] {
            let i_col = i_col as usize;
            let alpha = row_ap.array[i_col];
            let devex = d_pivot_weight * alpha * alpha + devex_index[i_col] as f64;
            if edge_weight[i_col] < devex {
                edge_weight[i_col] = devex;
            }
        }
        let row_ep = &self.row_ep;
        for &i_row in &row_ep.index[..row_ep.count as usize] {
            let i_col = i_row as usize + num_col;
            let alpha = row_ep.array[i_row as usize];
            let devex = d_pivot_weight * alpha * alpha + devex_index[i_col] as f64;
            if edge_weight[i_col] < devex {
                edge_weight[i_col] = devex;
            }
        }
        // Update devex weight for the pivots
        edge_weight[self.variable_out as usize] = 1.0f64.max(d_pivot_weight);
        edge_weight[variable_in] = 1.0;
        self.num_devex_iterations += 1;
    }

    /// HEkk::logicalBasis
    fn logical_basis(&self) -> bool {
        self.ekk.basic_index[..self.num_row].iter().all(|&i| i as usize >= self.num_col)
    }

    /// HEkkPrimal::computePrimalSteepestEdgeWeights
    fn compute_primal_steepest_edge_weights(&mut self) {
        self.edge_weight.resize(self.num_tot, 0.0);
        if self.logical_basis() {
            let a = &self.ekk.a;
            for i_col in 0..self.num_col {
                let mut w = 1.0;
                for &v in &a.value[a.start[i_col] as usize..a.start[i_col + 1] as usize] {
                    w = v.mul_add_c(v, w);
                }
                self.edge_weight[i_col] = w;
            }
        } else {
            let mut local_col_aq = OwnedHVec::new(self.num_row as i32);
            for i_var in 0..self.num_tot {
                if self.ekk.nonbasic_flag[i_var] != 0 {
                    self.edge_weight[i_var] = self.compute_primal_steepest_edge_weight(i_var, &mut local_col_aq);
                }
            }
        }
    }

    /// HEkkPrimal::computePrimalSteepestEdgeWeight (as inlined by clang,
    /// with the norm contracted throughout)
    fn compute_primal_steepest_edge_weight(&mut self, i_var: usize, local_col_aq: &mut OwnedHVec) -> f64 {
        local_col_aq.clear();
        let e = &mut self.ekk;
        let num_row = self.num_row;
        local_col_aq.with(|v| {
            e.a.collect_aj(v, i_var, 1.0);
            v.pack_flag = false;
            e.ftran(v, *e.col_aq_density);
            let local_col_aq_density = (1.0 * v.count as f64) / num_row as f64;
            update_operation_result_density(local_col_aq_density, e.col_aq_density);
            1.0 + v.norm2_fused()
        })
    }

    /// HEkkPrimal::updatePrimalSteepestEdgeWeights
    fn update_primal_steepest_edge_weights(&mut self) {
        copy_hvec(&mut self.col_steepest_edge, &self.col_aq);
        self.update_btran_pse();
        let col_aq_squared_2norm = self.col_aq.with(|v| v.norm2());
        let num_col = self.num_col;
        let variable_in = self.variable_in as usize;
        debug_assert!(self.ekk.nonbasic_flag[variable_in] != 0);
        let a = &self.ekk.a;
        let mu = &self.col_steepest_edge.array;
        let (row_ap, row_ep) = (&self.row_ap, &self.row_ep);
        let ap_count = row_ap.count as usize;
        for i_x in 0..ap_count + row_ep.count as usize {
            let (i_var, pivotal_row_value) = if i_x < ap_count {
                let i_var = row_ap.index[i_x] as usize;
                (i_var, row_ap.array[i_var])
            } else {
                let i_row = row_ep.index[i_x - ap_count] as usize;
                (num_col + i_row, row_ep.array[i_row])
            };
            if i_var == variable_in || self.ekk.nonbasic_flag[i_var] == 0 {
                continue;
            }
            let lambda = pivotal_row_value / self.alpha_col;
            let mut mu_aj = 0.0;
            if i_var < num_col {
                for i_el in a.start[i_var] as usize..a.start[i_var + 1] as usize {
                    mu_aj = mu[a.index[i_el] as usize].mul_add_c(a.value[i_el], mu_aj);
                }
            } else {
                mu_aj = mu[i_var - num_col];
            }
            let min_weight = lambda.mul_add_c(lambda, 1.0);
            let w = &mut self.edge_weight[i_var];
            *w += (lambda * lambda).mul_add_c(col_aq_squared_2norm, (lambda * -2.0) * mu_aj);
            *w = lambda.mul_add_c(lambda, *w);
            if *w < min_weight {
                *w = min_weight;
            }
        }
        // The tableau column for the variable leaving the basis is the
        // pivotal column, divided through by the pivot, except for the
        // value in the pivotal location, which is 1/pivot
        self.edge_weight[self.variable_out as usize] =
            (1.0 + col_aq_squared_2norm) / (self.alpha_col * self.alpha_col);
        self.edge_weight[variable_in] = 0.0;
    }

    /// HEkkPrimal::updateDualSteepestEdgeWeights
    fn update_dual_steepest_edge_weights(&mut self) {
        copy_hvec(&mut self.col_steepest_edge, &self.row_ep);
        self.update_ftran_dse();
        let row_out = self.row_out as usize;
        let variable_in = self.variable_in as usize;
        let e = &mut self.ekk;
        // Compute the weight from row_ep and over-write the updated weight
        e.dual_edge_weight[row_out] = if e.simplex_in_scaled_space {
            self.row_ep.with(|v| v.norm2())
        } else {
            let ekk = &*e;
            self.row_ep.with(|v| row_ep_2norm_in_scaled_space(ekk, row_out, v))
        };
        // HSimplexNla::pivotInScaledSpace
        let pivot_in_scaled_space = self.col_aq.array[row_out] * e.variable_scale_factor(variable_in)
            / e.variable_scale_factor(e.basic_index[row_out] as usize);
        let new_pivotal_edge_weight =
            e.dual_edge_weight[row_out] / (pivot_in_scaled_space * pivot_in_scaled_space);
        let kai = -2.0 / pivot_in_scaled_space;
        let dse_array = &self.col_steepest_edge.array;
        self.col_aq.with(|aq| {
            e.update_dual_steepest_edge_weights(row_out, variable_in, aq, new_pivotal_edge_weight, kai, dse_array)
        });
        e.dual_edge_weight[row_out] = new_pivotal_edge_weight;
    }

    /// HEkkPrimal::updateFtranDSE
    fn update_ftran_dse(&mut self) {
        let e = &mut self.ekk;
        let num_row = self.num_row;
        self.col_steepest_edge.with(|v| {
            // Apply R{-1}: HSimplexNla::unapplyBasisMatrixRowScale
            if let Some((_, row_scale)) = e.scale {
                let (use_row_indices, to_entry) = sparse_loop_style(v.count, num_row);
                for i_entry in 0..to_entry {
                    let i_row = if use_row_indices { v.index[i_entry] as usize } else { i_entry };
                    v.array[i_row] /= row_scale[i_row];
                }
            }
            // Perform FTRAN DSE
            e.factor.ftran(v, *e.row_dse_density);
            let local_row_dse_density = (1.0 * v.count as f64) / num_row as f64;
            update_operation_result_density(local_row_dse_density, e.row_dse_density);
        });
    }

    /// HEkkPrimal::updateBtranPSE
    fn update_btran_pse(&mut self) {
        let e = &mut self.ekk;
        let num_row = self.num_row;
        self.col_steepest_edge.with(|v| {
            e.btran(v, *e.col_steepest_edge_density);
            let local_density = (1.0 * v.count as f64) / num_row as f64;
            update_operation_result_density(local_density, e.col_steepest_edge_density);
        });
    }

    /// HEkkPrimal::debugPrimalSteepestEdgeWeights(alt_debug_level), which
    /// HEkkPrimal::update always calls at level kHighsDebugLevelCostly
    fn debug_primal_steepest_edge_weights(&mut self, use_debug_level: i32) {
        if use_debug_level < HIGHS_DEBUG_LEVEL_COSTLY {
            return;
        }
        let num_tot = self.num_tot;
        let mut norm = 0.0;
        let mut error = 0.0;
        let num_check_weight;
        let mut local_col_aq = OwnedHVec::new(self.num_row as i32);
        if use_debug_level < HIGHS_DEBUG_LEVEL_EXPENSIVE {
            for i_var in 0..num_tot {
                norm += (self.ekk.nonbasic_flag[i_var] as f64 * self.edge_weight[i_var]).abs();
            }
            // Just check a few weights
            num_check_weight = 1.max(10.min(num_tot as i32 / 10));
            for _ in 0..num_check_weight {
                let i_var = loop {
                    let i_var = self.random.integer_below(num_tot as i32) as usize;
                    if self.ekk.nonbasic_flag[i_var] != 0 {
                        break i_var;
                    }
                };
                let true_weight = self.compute_primal_steepest_edge_weight(i_var, &mut local_col_aq);
                error += (self.edge_weight[i_var] - true_weight).abs();
            }
        } else {
            // Check all weights
            num_check_weight = self.num_col as i32;
            let updated_primal_edge_weight = self.edge_weight.clone();
            self.compute_primal_steepest_edge_weights();
            for i_var in 0..num_tot {
                if self.ekk.nonbasic_flag[i_var] == 0 {
                    continue;
                }
                norm += self.edge_weight[i_var].abs();
                error += (updated_primal_edge_weight[i_var] - self.edge_weight[i_var]).abs();
            }
            self.edge_weight = updated_primal_edge_weight;
        }
        // Now assess the relative error
        debug_assert!(norm > 0.0);
        let relative_error = error / norm;
        if relative_error > 10.0 * self.debug_max_relative_primal_steepest_edge_weight_error {
            self.log(Log::PseWeightError, &[*self.ekk.iteration_count, num_check_weight], &[error, norm, relative_error]);
            self.debug_max_relative_primal_steepest_edge_weight_error = relative_error;
        }
    }

    // ---- Nonbasic free columns, perturbations and infeasibilities ----

    /// HEkkPrimal::getNonbasicFreeColumnSet
    fn get_nonbasic_free_column_set(&mut self) {
        if self.num_free_col == 0 {
            return;
        }
        self.nonbasic_free_col_set.clear();
        let e = &self.ekk;
        for i_col in 0..self.num_tot {
            let nonbasic_free =
                e.nonbasic_flag[i_col] == 1 && e.work_lower[i_col] <= -K_HIGHS_INF && e.work_upper[i_col] >= K_HIGHS_INF;
            if nonbasic_free {
                self.nonbasic_free_col_set.add(i_col as i32);
            }
        }
    }

    /// HEkkPrimal::removeNonbasicFreeColumn
    fn remove_nonbasic_free_column(&mut self) {
        let remove_nonbasic_free_column = self.ekk.nonbasic_move[self.variable_in as usize] == 0;
        if remove_nonbasic_free_column && !self.nonbasic_free_col_set.remove(self.variable_in) {
            self.log(Log::RemoveFreeFailed, &[self.variable_in], &[]);
            debug_assert!(false);
        }
    }

    /// HEkkPrimal::adjustPerturbedEquationOut
    fn adjust_perturbed_equation_out(&mut self) {
        if !*self.ekk.bounds_perturbed {
            return;
        }
        let e = &mut self.ekk;
        let variable_out = self.variable_out as usize;
        let (lp_lower, lp_upper) = if variable_out < self.num_col {
            (e.col_lower[variable_out], e.col_upper[variable_out])
        } else {
            (-e.row_upper[variable_out - self.num_col], -e.row_lower[variable_out - self.num_col])
        };
        if lp_lower < lp_upper {
            return;
        }
        // Leaving variable is fixed
        let true_fixed_value = lp_lower;
        // Modify theta_primal so that variable leaves at true fixed value
        self.theta_primal = (e.base_value[self.row_out as usize] - true_fixed_value) / self.alpha_col;
        e.work_lower[variable_out] = true_fixed_value;
        e.work_upper[variable_out] = true_fixed_value;
        e.work_range[variable_out] = 0.0;
        self.value_in = e.work_value[self.variable_in as usize] + self.theta_primal;
    }

    /// HEkkPrimal::getBasicPrimalInfeasibility
    fn get_basic_primal_infeasibility(&mut self) {
        // Gets the num/max/sum of basic primal infeasibilities
        let tol = self.primal_feasibility_tolerance;
        let e = &mut self.ekk;
        let updated_num_primal_infeasibility = *e.num_primal_infeasibilities;
        let mut num = 0;
        let mut max: f64 = 0.0;
        let mut sum = 0.0;
        for i_row in 0..self.num_row {
            let (value, lower, upper) = (e.base_value[i_row], e.base_lower[i_row], e.base_upper[i_row]);
            let primal_infeasibility = if value < lower - tol {
                lower - value
            } else if value > upper + tol {
                value - upper
            } else {
                0.0
            };
            if primal_infeasibility > 0.0 {
                if primal_infeasibility > tol {
                    num += 1;
                }
                max = if primal_infeasibility > max { primal_infeasibility } else { max };
                sum += primal_infeasibility;
            }
        }
        *e.num_primal_infeasibilities = num;
        *e.max_primal_infeasibility = max;
        *e.sum_primal_infeasibilities = sum;
        if updated_num_primal_infeasibility >= 0 {
            // The number of primal infeasibilities should be correct
            debug_assert!(num == updated_num_primal_infeasibility);
        }
    }

    /// HEkkPrimal::shiftBound: shift `bound` so that `value` is feasible
    /// for it; returns the shift
    fn shift_bound(&self, lower: bool, i_var: usize, value: f64, random_value: f64, bound: &mut f64) -> f64 {
        // If infeasibility is very large, then adding feasibility may not
        // yield a new value (see #1144) so new_infeasibility < 0 is false
        let tol = self.primal_feasibility_tolerance;
        let feasibility = (1.0 + random_value) * tol;
        let old_bound = *bound;
        let infeasibility;
        let new_infeasibility;
        let shift;
        if lower {
            // Bound to shift is lower
            debug_assert!(value < *bound - tol);
            infeasibility = *bound - value;
            // Determine the amount by which value will be feasible - so that
            // (ideally) it's not degenerate
            shift = infeasibility + feasibility;
            *bound -= shift;
            new_infeasibility = *bound - value;
        } else {
            // Bound to shift is upper
            debug_assert!(value > *bound + tol);
            infeasibility = value - *bound;
            shift = infeasibility + feasibility;
            *bound += shift;
            new_infeasibility = value - *bound;
        }
        if new_infeasibility > 0.0 {
            // new_infeasibility should be non-positive, and negative unless
            // bound is excessively large, whereas feasibility is positive
            let error = (new_infeasibility + feasibility).abs();
            self.log(
                Log::ShiftBound,
                &[i_var as i32, lower as i32],
                &[value, old_bound, random_value, feasibility, infeasibility, shift, *bound, new_infeasibility, error],
            );
        }
        debug_assert!(new_infeasibility <= 0.0);
        shift
    }
}

/// HSimplexNla::rowEp2NormInScaledSpace (as inlined by clang into
/// HEkkPrimal::updateDualSteepestEdgeWeights)
fn row_ep_2norm_in_scaled_space(e: &EkkView, i_row: usize, row_ep: &HVec) -> f64 {
    let Some((_, row_scale)) = e.scale else { return row_ep.norm2() };
    // Determine the scaling that was applied to the unit RHS before scaled
    // BTRAN. This must be unapplied to all components of the result
    let col_scale_value = e.basic_col_scale_factor(i_row);
    let (use_row_indices, to_entry) = sparse_loop_style(row_ep.count, e.num_row);
    // Clang interleaves the indexed loop by 4 and the dense loop by 8
    // without contracting, and contracts the remainder loops
    let unfused = if use_row_indices {
        interleaved_part(to_entry)
    } else if to_entry >= 8 {
        to_entry & !7
    } else {
        0
    };
    let mut row_ep_2norm = 0.0;
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
