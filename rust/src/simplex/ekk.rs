//! The numerical kernels of HEkk (highs/simplex/HEkk.cpp), working on HEkk's
//! data through an `EkkView`.
//!
//! # The view
//!
//! HEkk's data stays owned by C++ for now. `HEkk::rustView()`
//! (highs/simplex/HEkkRust.cpp) fills a `#[repr(C)]` [`CEkk`] with
//! pointer+length pairs for HEkk's vectors (lp_, the row-wise ar_matrix_,
//! basis_, the info_ work/base arrays, the edge weights), pointers to the
//! scalars that the kernels update (densities, update_count, objective
//! values, infeasibility records, ...), the values of the options and
//! scalars that are constant during a call, and the Rust factor handle of
//! simplex_nla_.factor_. [`CEkk::view`] turns it into an [`EkkView`] of
//! slices and `&mut` scalars, which the kernels take as `&mut self`.
//! Building a view costs some tens of loads and stores, so it is done per
//! C++ call; the Rust solve (hekk.rs) builds one per solver from the
//! `CHekk` that C++ fills, with the vectors sized for the whole solve.
//!
//! Rules for extending it:
//! - Add a field to `CEkk` here and to `highs_rs::Ekk` in
//!   highs/simplex/HEkkRust.h in the same position, filled in
//!   `HEkk::rustView()`; give every vector its length (`CSlice`).
//! - The view never resizes a vector: C++ sizes them before calling (for a
//!   whole solve, HEkk::solveRust does: see hekk.rs).
//! - Scalars that a kernel writes, or that change between kernel calls of
//!   one solve, are pointers (`&mut`); options are values.
//! - visited_basis_, bad_basis_change_, the factor's refactorization
//!   information and its saved INVERT are Rust-owned. status_ is shared;
//!   analysis_ (timers, operation records) and logging stay on the C++
//!   side (hekk.rs's Host). The ProductFormUpdate of simplex_nla_ is never
//!   set up in this version of HiGHS, so is not ported.
//!   HVectors other than HEkk's arrays come per call as `CHVec`s, sized by
//!   C++, or are owned by the Rust solvers.

use crate::util::fma::ClangFma;

use crate::factor::{AMatrix, HFactor};
use crate::ffi::{sl, sl_mut, CHVec};
use crate::hvector::{HVec, K_HIGHS_TINY, K_HIGHS_ZERO};
use crate::matrix;
use crate::util::hash;

const K_HIGHS_INF: f64 = f64::INFINITY;
const K_RUNNING_AVERAGE_MULTIPLIER: f64 = 0.05;
const K_DENSITY_FOR_INDEXING: f64 = 0.4;
const K_HYPER_PRICE_DENSITY: f64 = 0.1;
const K_MIN_DUAL_STEEPEST_EDGE_WEIGHT: f64 = 1e-4;
const K_ILLEGAL_INFEASIBILITY_COUNT: i32 = -1;
const K_ILLEGAL_INFEASIBILITY_MEASURE: f64 = K_HIGHS_INF;
const K_DEFAULT_PIVOT_THRESHOLD: f64 = 0.1;
const K_PIVOT_THRESHOLD_CHANGE_FACTOR: f64 = 5.0;
const K_MAX_PIVOT_THRESHOLD: f64 = 0.5;
const K_SYNTHETIC_TICK_REINVERSION_MIN_UPDATE_COUNT: i32 = 50;
const K_REBUILD_REASON_UPDATE_LIMIT_REACHED: i32 = 1;
const K_REBUILD_REASON_SYNTHETIC_CLOCK_SAYS_INVERT: i32 = 2;

const MOVE_UP: i8 = 1;
const MOVE_DN: i8 = -1;
const MOVE_ZE: i8 = 0;

// SimplexPriceStrategy
const PRICE_COL: i32 = 0;
const PRICE_ROW_SWITCH: i32 = 2;
const PRICE_ROW_SWITCH_COL_SWITCH: i32 = 3;

// SimplexAlgorithm and solve phases
pub const ALGORITHM_PRIMAL: i32 = 1;
pub const SOLVE_PHASE_2: i32 = 2;

/// highs_isInfinity
#[inline]
fn is_inf(x: f64) -> bool {
    x >= K_HIGHS_INF
}

/// HEkk::updateOperationResultDensity
#[inline]
pub fn update_operation_result_density(local_density: f64, density: &mut f64) {
    *density = density.mul_add_c(
        1.0 - K_RUNNING_AVERAGE_MULTIPLIER,
        K_RUNNING_AVERAGE_MULTIPLIER * local_density,
    );
}

/// HSimplexNla::sparseLoopStyle: whether to loop over the indices, and
/// how far
#[inline]
pub fn sparse_loop_style(count: i32, dim: usize) -> (bool, usize) {
    let use_indices = count >= 0 && (count as f64) < K_DENSITY_FOR_INDEXING * dim as f64;
    (use_indices, if use_indices { count as usize } else { dim })
}

/// HEkk::choosePriceTechnique: (use_col_price, use_row_price_w_switch)
pub fn choose_price_technique(price_strategy: i32, row_ep_density: f64) -> (bool, bool) {
    // By default switch to column PRICE when pi_p has at least this density
    let density_for_column_price_switch = 0.75;
    let use_col_price = price_strategy == PRICE_COL
        || (price_strategy == PRICE_ROW_SWITCH_COL_SWITCH
            && row_ep_density > density_for_column_price_switch);
    let use_row_price_w_switch =
        price_strategy == PRICE_ROW_SWITCH || price_strategy == PRICE_ROW_SWITCH_COL_SWITCH;
    (use_col_price, use_row_price_w_switch)
}

/// HEkk::computeDualForTableauColumn
pub fn compute_dual_for_tableau_column(
    work_cost: &[f64],
    basic_index: &[i32],
    i_var: usize,
    tableau_index: &[i32],
    tableau_array: &[f64],
) -> f64 {
    let mut dual = work_cost[i_var];
    // Clang interleaves this loop by 4 without contracting, and contracts
    // the remainder loop
    let unfused = interleaved_part(tableau_index.len());
    for (k, &i_row) in tableau_index.iter().enumerate() {
        let i_row = i_row as usize;
        let (a, c) = (tableau_array[i_row], work_cost[basic_index[i_row] as usize]);
        dual = if k < unfused { dual - a * c } else { (-a).mul_add_c(c, dual) };
    }
    dual
}

/// How many leading iterations of a reduction loop `sum += a * b` clang
/// runs interleaved by 4 (and so without fusing the multiply-add) when it
/// does `count` iterations: the rest run in a scalar remainder loop, fused
#[inline]
pub fn interleaved_part(count: usize) -> usize {
    if count >= 4 {
        count & !3
    } else {
        0
    }
}

/// HEkk::flipBound
#[inline]
pub fn flip_bound(
    nonbasic_move: &mut [i8],
    work_value: &mut [f64],
    work_lower: &[f64],
    work_upper: &[f64],
    i_col: usize,
) {
    let mv = -nonbasic_move[i_col];
    nonbasic_move[i_col] = mv;
    work_value[i_col] = if mv == 1 { work_lower[i_col] } else { work_upper[i_col] };
}

/// The outcome of HEkk::reinvertOnNumericalTrouble
#[repr(C)]
pub struct NumericalTrouble {
    pub measure: f64,
    /// The pivot threshold to set, or zero
    pub new_pivot_threshold: f64,
    pub reinvert: bool,
}

/// HEkk::reinvertOnNumericalTrouble (logging and the setting of the
/// threshold stay with the caller)
pub fn reinvert_on_numerical_trouble(
    alpha_from_col: f64,
    alpha_from_row: f64,
    numerical_trouble_tolerance: f64,
    update_count: i32,
    current_pivot_threshold: f64,
) -> NumericalTrouble {
    let abs_alpha_from_col = alpha_from_col.abs();
    let abs_alpha_from_row = alpha_from_row.abs();
    let min_abs_alpha = abs_alpha_from_col.min(abs_alpha_from_row);
    let abs_alpha_diff = (abs_alpha_from_col - abs_alpha_from_row).abs();
    let measure = abs_alpha_diff / min_abs_alpha;
    // Reinvert if the relative difference is large enough, and updates
    // have been performed
    let reinvert = measure > numerical_trouble_tolerance && update_count > 0;
    let mut new_pivot_threshold = 0.0;
    if reinvert {
        // Consider increasing the Markowitz multiplier
        if current_pivot_threshold < K_DEFAULT_PIVOT_THRESHOLD {
            // Threshold is below default value, so increase it
            new_pivot_threshold = (current_pivot_threshold * K_PIVOT_THRESHOLD_CHANGE_FACTOR)
                .min(K_DEFAULT_PIVOT_THRESHOLD);
        } else if current_pivot_threshold < K_MAX_PIVOT_THRESHOLD && update_count < 10 {
            // Threshold is below max value, so increase it if few updates
            // have been performed
            new_pivot_threshold = (current_pivot_threshold * K_PIVOT_THRESHOLD_CHANGE_FACTOR)
                .min(K_MAX_PIVOT_THRESHOLD);
        }
    }
    NumericalTrouble { measure, new_pivot_threshold, reinvert }
}

/// Number, max and sum of infeasibilities
#[repr(C)]
#[derive(Default)]
pub struct Infeasibility {
    pub num: i32,
    pub max: f64,
    pub sum: f64,
}

impl Infeasibility {
    #[inline]
    fn add(&mut self, infeasibility: f64, tolerance: f64, strict: bool) {
        if infeasibility > 0.0 {
            if if strict { infeasibility > tolerance } else { infeasibility >= tolerance } {
                self.num += 1;
            }
            self.max = if self.max < infeasibility { infeasibility } else { self.max };
            self.sum += infeasibility;
        }
    }
}

/// The figures that HEkk::initialiseCost reports when perturbing costs
#[repr(C)]
#[derive(Default)]
pub struct CostPerturbationReport {
    pub num_original_nonzero_cost: i32,
    pub pct0: i32,
    pub min_abs_cost: f64,
    pub average_abs_cost: f64,
    /// max_abs_cost before the large/boxed-rate adjustments
    pub max_abs_cost: f64,
    /// Whether max_abs_cost > 100, and its fourth root
    pub large: bool,
    pub large_max_abs_cost: f64,
    pub boxed_rate: f64,
    /// Whether the boxed rate is small, and max_abs_cost after
    pub small_boxed_rate: bool,
    pub small_boxed_max_abs_cost: f64,
    pub row_cost_perturbation_base: f64,
    /// Whether the costs were perturbed (and the figures are set)
    pub perturbed: bool,
}

/// A pointer and length, as std::vector's data() and size()
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CSlice<T> {
    pub(crate) p: *mut T,
    pub(crate) n: i32,
}

impl<T> CSlice<T> {
    /// # Safety
    /// `p` valid for `n` reads
    pub(crate) unsafe fn get<'a>(&self) -> &'a [T] {
        sl(self.p, self.n)
    }
    /// # Safety
    /// `p` valid for `n` reads and writes, unaliased
    pub(crate) unsafe fn get_mut<'a>(&self) -> &'a mut [T] {
        sl_mut(self.p, self.n)
    }
}

/// HEkk's data as filled by HEkk::rustView(): mirrored by highs_rs::Ekk in
/// highs/simplex/HEkkRust.h
#[repr(C)]
pub struct CEkk {
    pub(crate) num_col: i32,
    pub(crate) num_row: i32,
    // lp_
    pub(crate) a_start: CSlice<i32>,
    pub(crate) a_index: CSlice<i32>,
    pub(crate) a_value: CSlice<f64>,
    pub(crate) col_cost: CSlice<f64>,
    pub(crate) col_lower: CSlice<f64>,
    pub(crate) col_upper: CSlice<f64>,
    pub(crate) row_lower: CSlice<f64>,
    pub(crate) row_upper: CSlice<f64>,
    pub(crate) sense: i32,
    pub(crate) offset: f64,
    // Scaling of the basis matrix (simplex_nla_.scale_), if any
    pub(crate) has_scale: bool,
    pub(crate) col_scale: CSlice<f64>,
    pub(crate) row_scale: CSlice<f64>,
    // ar_matrix_
    pub(crate) ar_start: CSlice<i32>,
    pub(crate) ar_p_end: CSlice<i32>,
    pub(crate) ar_index: CSlice<i32>,
    pub(crate) ar_value: CSlice<f64>,
    // basis_
    pub(crate) basic_index: CSlice<i32>,
    pub(crate) nonbasic_flag: CSlice<i8>,
    pub(crate) nonbasic_move: CSlice<i8>,
    pub(crate) basis_hash: *mut u64,
    // info_ arrays
    pub(crate) work_cost: CSlice<f64>,
    pub(crate) work_dual: CSlice<f64>,
    pub(crate) work_shift: CSlice<f64>,
    pub(crate) work_lower: CSlice<f64>,
    pub(crate) work_upper: CSlice<f64>,
    pub(crate) work_range: CSlice<f64>,
    pub(crate) work_value: CSlice<f64>,
    pub(crate) work_lower_shift: CSlice<f64>,
    pub(crate) work_upper_shift: CSlice<f64>,
    pub(crate) base_lower: CSlice<f64>,
    pub(crate) base_upper: CSlice<f64>,
    pub(crate) base_value: CSlice<f64>,
    pub(crate) num_tot_random_value: CSlice<f64>,
    pub(crate) dual_edge_weight: CSlice<f64>,
    pub(crate) scattered_dual_edge_weight: CSlice<f64>,
    // info_ scalars
    pub(crate) col_aq_density: *mut f64,
    pub(crate) row_ep_density: *mut f64,
    pub(crate) row_ap_density: *mut f64,
    pub(crate) row_dse_density: *mut f64,
    pub(crate) primal_col_density: *mut f64,
    pub(crate) dual_col_density: *mut f64,
    pub(crate) update_count: *mut i32,
    pub(crate) num_basic_logicals: *mut i32,
    pub(crate) updated_dual_objective_value: *mut f64,
    pub(crate) primal_objective_value: *mut f64,
    pub(crate) dual_objective_value: *mut f64,
    pub(crate) num_primal_infeasibilities: *mut i32,
    pub(crate) max_primal_infeasibility: *mut f64,
    pub(crate) sum_primal_infeasibilities: *mut f64,
    pub(crate) num_dual_infeasibilities: *mut i32,
    pub(crate) max_dual_infeasibility: *mut f64,
    pub(crate) sum_dual_infeasibilities: *mut f64,
    pub(crate) costs_shifted: *mut bool,
    pub(crate) costs_perturbed: *mut bool,
    pub(crate) bounds_shifted: *mut bool,
    pub(crate) bounds_perturbed: *mut bool,
    pub(crate) price_strategy: i32,
    pub(crate) dual_simplex_cost_perturbation_multiplier: *mut f64,
    pub(crate) primal_simplex_bound_perturbation_multiplier: *mut f64,
    // options_
    pub(crate) primal_feasibility_tolerance: f64,
    pub(crate) dual_feasibility_tolerance: f64,
    pub(crate) cost_scale_factor: i32,
    pub(crate) output_flag: bool,
    // HEkk scalars
    pub(crate) cost_scale: f64,
    pub(crate) cost_perturbation_base: *mut f64,
    pub(crate) cost_perturbation_max_abs_cost: *mut f64,
    pub(crate) simplex_in_scaled_space: bool,
    pub(crate) update_limit: *mut i32,
    pub(crate) build_synthetic_tick: *mut f64,
    pub(crate) total_synthetic_tick: *mut f64,
    pub(crate) factor: *mut HFactor,
    // The factor's constraint matrix (HFactor::a_start, ...)
    pub(crate) factor_num_col: i32,
    pub(crate) factor_a_start: CSlice<i32>,
    pub(crate) factor_a_index: CSlice<i32>,
    pub(crate) factor_a_value: CSlice<f64>,
    // HEkkPrimal
    pub(crate) status: *mut SimplexStatus,
    pub(crate) iteration_count: *mut i32,
    pub(crate) updated_primal_objective_value: *mut f64,
    pub(crate) allow_bound_perturbation: *mut bool,
    pub(crate) backtracking: *mut bool,
    pub(crate) primal_phase1_iteration_count: *mut i32,
    pub(crate) primal_phase2_iteration_count: *mut i32,
    pub(crate) primal_bound_swap: *mut i32,
    pub(crate) col_basic_feasibility_change_density: *mut f64,
    pub(crate) row_basic_feasibility_change_density: *mut f64,
    pub(crate) col_steepest_edge_density: *mut f64,
    pub(crate) primal_simplex_phase1_cost_perturbation_multiplier: f64,
    pub(crate) simplex_primal_edge_weight_strategy: i32,
    pub(crate) simplex_iteration_limit: i32,
    pub(crate) bailout_in_cpp: bool,
    pub(crate) iteration_report: bool,
}

/// Column-wise matrix (lp_.a_matrix_)
pub struct Csc<'a> {
    pub start: &'a [i32],
    pub index: &'a [i32],
    pub value: &'a [f64],
}

impl Csc<'_> {
    /// HighsSparseMatrix::collectAj: column += multiplier * a_j (j a
    /// logical if j >= num_col)
    #[inline]
    pub fn collect_aj(&self, column: &mut HVec, use_col: usize, multiplier: f64) {
        #[inline(always)]
        fn add(column: &mut HVec, i_row: usize, value0: f64, value1: f64) {
            if value0 == 0.0 {
                column.index[column.count as usize] = i_row as i32;
                column.count += 1;
            }
            column.array[i_row] = if value1.abs() < K_HIGHS_TINY { K_HIGHS_ZERO } else { value1 };
        }
        let num_col = self.start.len() - 1;
        if use_col < num_col {
            let (from, to) = (self.start[use_col] as usize, self.start[use_col + 1] as usize);
            for (&i_row, &a) in self.index[from..to].iter().zip(&self.value[from..to]) {
                let i_row = i_row as usize;
                let value0 = column.array[i_row];
                add(column, i_row, value0, multiplier.mul_add_c(a, value0));
            }
        } else {
            let i_row = use_col - num_col;
            let value0 = column.array[i_row];
            add(column, i_row, value0, value0 + multiplier);
        }
    }
}

/// HEkk's data: see the module comment
pub struct EkkView<'a> {
    pub num_col: usize,
    pub num_row: usize,
    // lp_
    pub a: Csc<'a>,
    pub col_cost: &'a [f64],
    pub col_lower: &'a [f64],
    pub col_upper: &'a [f64],
    pub row_lower: &'a [f64],
    pub row_upper: &'a [f64],
    pub sense: i32,
    pub offset: f64,
    /// (col, row) scale factors of the basis matrix
    pub scale: Option<(&'a [f64], &'a [f64])>,
    // ar_matrix_ (row-wise, partitioned by nonbasicFlag)
    pub ar_start: &'a mut [i32],
    pub ar_p_end: &'a mut [i32],
    pub ar_index: &'a mut [i32],
    pub ar_value: &'a mut [f64],
    // basis_
    pub basic_index: &'a mut [i32],
    pub nonbasic_flag: &'a mut [i8],
    pub nonbasic_move: &'a mut [i8],
    pub basis_hash: &'a mut u64,
    // info_
    pub work_cost: &'a mut [f64],
    pub work_dual: &'a mut [f64],
    pub work_shift: &'a mut [f64],
    pub work_lower: &'a mut [f64],
    pub work_upper: &'a mut [f64],
    pub work_range: &'a mut [f64],
    pub work_value: &'a mut [f64],
    pub work_lower_shift: &'a mut [f64],
    pub work_upper_shift: &'a mut [f64],
    pub base_lower: &'a mut [f64],
    pub base_upper: &'a mut [f64],
    pub base_value: &'a mut [f64],
    pub num_tot_random_value: &'a [f64],
    pub dual_edge_weight: &'a mut [f64],
    pub scattered_dual_edge_weight: &'a mut [f64],
    pub col_aq_density: &'a mut f64,
    pub row_ep_density: &'a mut f64,
    pub row_ap_density: &'a mut f64,
    pub row_dse_density: &'a mut f64,
    pub primal_col_density: &'a mut f64,
    pub dual_col_density: &'a mut f64,
    pub update_count: &'a mut i32,
    pub num_basic_logicals: &'a mut i32,
    pub updated_dual_objective_value: &'a mut f64,
    pub primal_objective_value: &'a mut f64,
    pub dual_objective_value: &'a mut f64,
    pub num_primal_infeasibilities: &'a mut i32,
    pub max_primal_infeasibility: &'a mut f64,
    pub sum_primal_infeasibilities: &'a mut f64,
    pub num_dual_infeasibilities: &'a mut i32,
    pub max_dual_infeasibility: &'a mut f64,
    pub sum_dual_infeasibilities: &'a mut f64,
    pub costs_shifted: &'a mut bool,
    pub costs_perturbed: &'a mut bool,
    pub bounds_shifted: &'a mut bool,
    pub bounds_perturbed: &'a mut bool,
    pub price_strategy: i32,
    pub dual_simplex_cost_perturbation_multiplier: &'a mut f64,
    pub primal_simplex_bound_perturbation_multiplier: &'a mut f64,
    // options_
    pub primal_feasibility_tolerance: f64,
    pub dual_feasibility_tolerance: f64,
    pub cost_scale_factor: i32,
    pub output_flag: bool,
    // HEkk
    pub cost_scale: f64,
    pub cost_perturbation_base: &'a mut f64,
    pub cost_perturbation_max_abs_cost: &'a mut f64,
    pub simplex_in_scaled_space: bool,
    pub update_limit: &'a mut i32,
    pub build_synthetic_tick: &'a mut f64,
    pub total_synthetic_tick: &'a mut f64,
    pub factor: &'a mut HFactor,
    pub factor_a: AMatrix<'a>,
    // HEkkPrimal
    pub status: &'a mut SimplexStatus,
    pub iteration_count: &'a mut i32,
    pub updated_primal_objective_value: &'a mut f64,
    pub allow_bound_perturbation: &'a mut bool,
    pub backtracking: &'a mut bool,
    pub primal_phase1_iteration_count: &'a mut i32,
    pub primal_phase2_iteration_count: &'a mut i32,
    pub primal_bound_swap: &'a mut i32,
    pub col_basic_feasibility_change_density: &'a mut f64,
    pub row_basic_feasibility_change_density: &'a mut f64,
    pub col_steepest_edge_density: &'a mut f64,
    pub primal_simplex_phase1_cost_perturbation_multiplier: f64,
    pub simplex_primal_edge_weight_strategy: i32,
    pub simplex_iteration_limit: i32,
    /// Whether HEkk::bailout() has more to check than the iteration limit
    /// (a time limit or a user interrupt callback)
    pub bailout_in_cpp: bool,
    /// Whether HighsSimplexAnalysis::iterationReport() reports
    pub iteration_report: bool,
}

/// HighsSimplexStatus (simplex/SimplexStruct.h)
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct SimplexStatus {
    pub initialised_for_new_lp: bool,
    pub is_dualized: bool,
    pub is_permuted: bool,
    pub initialised_for_solve: bool,
    pub has_basis: bool,
    pub has_ar_matrix: bool,
    pub has_nla: bool,
    pub has_dual_steepest_edge_weights: bool,
    pub has_invert: bool,
    pub has_fresh_invert: bool,
    pub has_fresh_rebuild: bool,
    pub has_dual_objective_value: bool,
    pub has_primal_objective_value: bool,
}

impl CEkk {
    /// info_.numTotRandomValue_, to be written
    ///
    /// # Safety
    /// As for view; no view may be in use
    pub(crate) unsafe fn num_tot_random_value_mut<'a>(&self) -> &'a mut [f64] {
        self.num_tot_random_value.get_mut()
    }

    /// # Safety
    /// The pointers must be valid for their lengths (scalars: non-null),
    /// unaliased while the view lives
    pub unsafe fn view<'a>(&self) -> EkkView<'a> {
        EkkView {
            num_col: self.num_col as usize,
            num_row: self.num_row as usize,
            a: Csc {
                start: self.a_start.get(),
                index: self.a_index.get(),
                value: self.a_value.get(),
            },
            col_cost: self.col_cost.get(),
            col_lower: self.col_lower.get(),
            col_upper: self.col_upper.get(),
            row_lower: self.row_lower.get(),
            row_upper: self.row_upper.get(),
            sense: self.sense,
            offset: self.offset,
            scale: if self.has_scale {
                Some((self.col_scale.get(), self.row_scale.get()))
            } else {
                None
            },
            ar_start: self.ar_start.get_mut(),
            ar_p_end: self.ar_p_end.get_mut(),
            ar_index: self.ar_index.get_mut(),
            ar_value: self.ar_value.get_mut(),
            basic_index: self.basic_index.get_mut(),
            nonbasic_flag: self.nonbasic_flag.get_mut(),
            nonbasic_move: self.nonbasic_move.get_mut(),
            basis_hash: &mut *self.basis_hash,
            work_cost: self.work_cost.get_mut(),
            work_dual: self.work_dual.get_mut(),
            work_shift: self.work_shift.get_mut(),
            work_lower: self.work_lower.get_mut(),
            work_upper: self.work_upper.get_mut(),
            work_range: self.work_range.get_mut(),
            work_value: self.work_value.get_mut(),
            work_lower_shift: self.work_lower_shift.get_mut(),
            work_upper_shift: self.work_upper_shift.get_mut(),
            base_lower: self.base_lower.get_mut(),
            base_upper: self.base_upper.get_mut(),
            base_value: self.base_value.get_mut(),
            num_tot_random_value: self.num_tot_random_value.get(),
            dual_edge_weight: self.dual_edge_weight.get_mut(),
            scattered_dual_edge_weight: self.scattered_dual_edge_weight.get_mut(),
            col_aq_density: &mut *self.col_aq_density,
            row_ep_density: &mut *self.row_ep_density,
            row_ap_density: &mut *self.row_ap_density,
            row_dse_density: &mut *self.row_dse_density,
            primal_col_density: &mut *self.primal_col_density,
            dual_col_density: &mut *self.dual_col_density,
            update_count: &mut *self.update_count,
            num_basic_logicals: &mut *self.num_basic_logicals,
            updated_dual_objective_value: &mut *self.updated_dual_objective_value,
            primal_objective_value: &mut *self.primal_objective_value,
            dual_objective_value: &mut *self.dual_objective_value,
            num_primal_infeasibilities: &mut *self.num_primal_infeasibilities,
            max_primal_infeasibility: &mut *self.max_primal_infeasibility,
            sum_primal_infeasibilities: &mut *self.sum_primal_infeasibilities,
            num_dual_infeasibilities: &mut *self.num_dual_infeasibilities,
            max_dual_infeasibility: &mut *self.max_dual_infeasibility,
            sum_dual_infeasibilities: &mut *self.sum_dual_infeasibilities,
            costs_shifted: &mut *self.costs_shifted,
            costs_perturbed: &mut *self.costs_perturbed,
            bounds_shifted: &mut *self.bounds_shifted,
            bounds_perturbed: &mut *self.bounds_perturbed,
            price_strategy: self.price_strategy,
            dual_simplex_cost_perturbation_multiplier: &mut *self
                .dual_simplex_cost_perturbation_multiplier,
            primal_simplex_bound_perturbation_multiplier: &mut *self
                .primal_simplex_bound_perturbation_multiplier,
            primal_feasibility_tolerance: self.primal_feasibility_tolerance,
            dual_feasibility_tolerance: self.dual_feasibility_tolerance,
            cost_scale_factor: self.cost_scale_factor,
            output_flag: self.output_flag,
            cost_scale: self.cost_scale,
            cost_perturbation_base: &mut *self.cost_perturbation_base,
            cost_perturbation_max_abs_cost: &mut *self.cost_perturbation_max_abs_cost,
            simplex_in_scaled_space: self.simplex_in_scaled_space,
            update_limit: &mut *self.update_limit,
            build_synthetic_tick: &mut *self.build_synthetic_tick,
            total_synthetic_tick: &mut *self.total_synthetic_tick,
            factor: &mut *self.factor,
            factor_a: AMatrix {
                num_col: self.factor_num_col,
                start: self.factor_a_start.get(),
                index: self.factor_a_index.get(),
                value: self.factor_a_value.get(),
            },
            status: &mut *self.status,
            iteration_count: &mut *self.iteration_count,
            updated_primal_objective_value: &mut *self.updated_primal_objective_value,
            allow_bound_perturbation: &mut *self.allow_bound_perturbation,
            backtracking: &mut *self.backtracking,
            primal_phase1_iteration_count: &mut *self.primal_phase1_iteration_count,
            primal_phase2_iteration_count: &mut *self.primal_phase2_iteration_count,
            primal_bound_swap: &mut *self.primal_bound_swap,
            col_basic_feasibility_change_density: &mut *self.col_basic_feasibility_change_density,
            row_basic_feasibility_change_density: &mut *self.row_basic_feasibility_change_density,
            col_steepest_edge_density: &mut *self.col_steepest_edge_density,
            primal_simplex_phase1_cost_perturbation_multiplier: self
                .primal_simplex_phase1_cost_perturbation_multiplier,
            simplex_primal_edge_weight_strategy: self.simplex_primal_edge_weight_strategy,
            simplex_iteration_limit: self.simplex_iteration_limit,
            bailout_in_cpp: self.bailout_in_cpp,
            iteration_report: self.iteration_report,
        }
    }
}

impl EkkView<'_> {
    #[inline]
    pub fn num_tot(&self) -> usize {
        self.num_col + self.num_row
    }

    // ---- HSimplexNla: FTRAN/BTRAN with the basis matrix scaling ----

    /// HSimplexNla::variableScaleFactor
    #[inline]
    pub fn variable_scale_factor(&self, i_var: usize) -> f64 {
        match self.scale {
            None => 1.0,
            Some((col, row)) => {
                if i_var < self.num_col {
                    col[i_var]
                } else {
                    1.0 / row[i_var - self.num_col]
                }
            }
        }
    }

    /// HSimplexNla::basicColScaleFactor
    #[inline]
    pub fn basic_col_scale_factor(&self, i_col: usize) -> f64 {
        match self.scale {
            None => 1.0,
            Some(_) => self.variable_scale_factor(self.basic_index[i_col] as usize),
        }
    }

    /// HSimplexNla::applyBasisMatrixRowScale
    pub fn apply_basis_matrix_row_scale(&self, rhs: &mut HVec) {
        let Some((_, row_scale)) = self.scale else { return };
        let (use_row_indices, to_entry) = sparse_loop_style(rhs.count, self.num_row);
        for i_entry in 0..to_entry {
            let i_row = if use_row_indices { rhs.index[i_entry] as usize } else { i_entry };
            rhs.array[i_row] *= row_scale[i_row];
        }
    }

    /// HSimplexNla::applyBasisMatrixColScale
    pub fn apply_basis_matrix_col_scale(&self, rhs: &mut HVec) {
        let Some((col_scale, row_scale)) = self.scale else { return };
        let (use_row_indices, to_entry) = sparse_loop_style(rhs.count, self.num_row);
        for i_entry in 0..to_entry {
            let i_col = if use_row_indices { rhs.index[i_entry] as usize } else { i_entry };
            let i_var = self.basic_index[i_col] as usize;
            if i_var < self.num_col {
                rhs.array[i_col] *= col_scale[i_var];
            } else {
                rhs.array[i_col] /= row_scale[i_var - self.num_col];
            }
        }
    }

    /// HSimplexNla::ftran
    pub fn ftran(&self, rhs: &mut HVec, expected_density: f64) {
        self.apply_basis_matrix_row_scale(rhs);
        self.factor.ftran(rhs, expected_density);
        self.apply_basis_matrix_col_scale(rhs);
    }

    /// HSimplexNla::btran
    pub fn btran(&self, rhs: &mut HVec, expected_density: f64) {
        self.apply_basis_matrix_col_scale(rhs);
        self.factor.btran(rhs, expected_density);
        self.apply_basis_matrix_row_scale(rhs);
    }

    /// HSimplexNla::transformForUpdate
    pub fn transform_for_update(&self, aq: &mut HVec, ep: &mut HVec, variable_in: usize, row_out: usize) {
        if self.scale.is_none() {
            return;
        }
        let cq_scale_factor = self.variable_scale_factor(variable_in);
        for v in &mut aq.pack_value[..aq.pack_count as usize] {
            *v *= cq_scale_factor;
        }
        aq.array[row_out] *= cq_scale_factor;
        let cp_scale_factor = self.basic_col_scale_factor(row_out);
        aq.array[row_out] /= cp_scale_factor;
        for v in &mut ep.pack_value[..ep.pack_count as usize] {
            *v /= cp_scale_factor;
        }
    }

    // ---- FTRAN, BTRAN and PRICE of the simplex iteration ----

    /// HEkk::pivotColumnFtran
    pub fn pivot_column_ftran(&mut self, i_col: usize, col_aq: &mut HVec) {
        col_aq.clear();
        col_aq.pack_flag = true;
        self.a.collect_aj(col_aq, i_col, 1.0);
        self.ftran(col_aq, *self.col_aq_density);
        let local_col_aq_density = col_aq.count as f64 / self.num_row as f64;
        update_operation_result_density(local_col_aq_density, self.col_aq_density);
    }

    /// HEkk::unitBtran
    pub fn unit_btran(&mut self, i_row: usize, row_ep: &mut HVec) {
        row_ep.clear();
        row_ep.count = 1;
        row_ep.index[0] = i_row as i32;
        row_ep.array[i_row] = 1.0;
        row_ep.pack_flag = true;
        self.btran(row_ep, *self.row_ep_density);
        let local_row_ep_density = row_ep.count as f64 / self.num_row as f64;
        update_operation_result_density(local_row_ep_density, self.row_ep_density);
    }

    /// HEkk::fullBtran
    pub fn full_btran(&mut self, buffer: &mut HVec) {
        self.btran(buffer, *self.dual_col_density);
        let local_dual_col_density = buffer.count as f64 / self.num_row as f64;
        update_operation_result_density(local_dual_col_density, self.dual_col_density);
    }

    /// HEkk::fullPrice
    pub fn full_price(&self, full_col: &HVec, full_row: &mut HVec) {
        full_row.clear();
        self.price_by_column(full_col, full_row);
    }

    /// HighsSparseMatrix::priceByColumn of lp_.a_matrix_ (double precision)
    pub(crate) fn price_by_column(&self, column: &HVec, result: &mut HVec) {
        result.count =
            matrix::price_by_column(self.a.start, self.a.index, self.a.value, column.array, result.array, result.index)
                as i32;
    }

    /// HighsSparseMatrix::priceByRowWithSwitch of ar_matrix_ (double
    /// precision)
    pub(crate) fn price_by_row_with_switch(
        &self,
        column: &HVec,
        result: &mut HVec,
        expected_density: f64,
        from_index: usize,
        switch_density: f64,
    ) {
        result.count = matrix::price_by_row_with_switch(
            self.ar_start,
            self.ar_p_end,
            self.ar_index,
            self.ar_value,
            self.num_col,
            &column.index[..column.count as usize],
            column.array,
            expected_density <= K_HYPER_PRICE_DENSITY,
            from_index,
            switch_density,
            result.count as usize,
            result.array,
            result.index,
        ) as i32;
    }

    /// HEkk::tableauRowPrice (double precision)
    pub fn tableau_row_price(&mut self, row_ep: &HVec, row_ap: &mut HVec) {
        let local_density = 1.0 * row_ep.count as f64 / self.num_row as f64;
        let (use_col_price, use_row_price_w_switch) =
            choose_price_technique(self.price_strategy, local_density);
        row_ap.clear();
        if use_col_price {
            // Perform column-wise PRICE
            self.price_by_column(row_ep, row_ap);
        } else if use_row_price_w_switch {
            // Perform hyper-sparse row-wise PRICE, but switch if the
            // density of row_ap becomes extreme
            self.price_by_row_with_switch(row_ep, row_ap, *self.row_ap_density, 0, K_HYPER_PRICE_DENSITY);
        } else {
            // Perform hyper-sparse row-wise PRICE
            self.price_by_row_with_switch(row_ep, row_ap, -K_HIGHS_INF, 0, K_HIGHS_INF);
        }
        if use_col_price {
            // Column-wise PRICE computes components corresponding to basic
            // variables, so zero these by exploiting the fact that, for
            // basic variables, nonbasicFlag[*]=0
            for (x, &flag) in row_ap.array[..self.num_col].iter_mut().zip(&*self.nonbasic_flag) {
                *x *= flag as f64;
            }
        }
        // Update the record of average row_ap density
        let local_row_ap_density = row_ap.count as f64 / self.num_col as f64;
        update_operation_result_density(local_row_ap_density, self.row_ap_density);
    }

    // ---- Primal and dual values ----

    /// HEkk::computePrimal: `primal_col` is a cleared work vector of
    /// dimension num_row
    pub fn compute_primal(&mut self, primal_col: &mut HVec) {
        let num_row = self.num_row;
        let n = self.num_tot();
        let (flag, work_value) = (&self.nonbasic_flag[..n], &self.work_value[..n]);
        for i in 0..n {
            if flag[i] != 0 && work_value[i] != 0.0 {
                self.a.collect_aj(primal_col, i, work_value[i]);
            }
        }
        // It's possible that the buffer has no nonzeros, so performing
        // FTRAN is unnecessary
        if primal_col.count != 0 {
            self.ftran(primal_col, *self.primal_col_density);
            let local_primal_col_density = primal_col.count as f64 / num_row as f64;
            update_operation_result_density(local_primal_col_density, self.primal_col_density);
        }
        for i in 0..num_row {
            let i_col = self.basic_index[i] as usize;
            self.base_value[i] = -primal_col.array[i];
            self.base_lower[i] = self.work_lower[i_col];
            self.base_upper[i] = self.work_upper[i_col];
        }
        // Indicate that the primal infeasibility information isn't known
        *self.num_primal_infeasibilities = K_ILLEGAL_INFEASIBILITY_COUNT;
        *self.max_primal_infeasibility = K_ILLEGAL_INFEASIBILITY_MEASURE;
        *self.sum_primal_infeasibilities = K_ILLEGAL_INFEASIBILITY_MEASURE;
    }

    /// HEkk::computeDual: `dual_col` and `dual_row` are cleared work
    /// vectors of dimension num_row and num_col
    pub fn compute_dual(&mut self, dual_col: &mut HVec, dual_row: &mut HVec) {
        for i_row in 0..self.num_row {
            let i_var = self.basic_index[i_row] as usize;
            let value = self.work_cost[i_var] + self.work_shift[i_var];
            if value != 0.0 {
                dual_col.index[dual_col.count as usize] = i_row as i32;
                dual_col.count += 1;
                dual_col.array[i_row] = value;
            }
        }
        // Copy the costs in case the basic costs are all zero
        let num_tot = self.num_tot();
        for ((d, &c), &s) in self.work_dual[..num_tot].iter_mut().zip(&self.work_cost[..num_tot]).zip(&self.work_shift[..num_tot]) {
            *d = c + s;
        }
        if dual_col.count != 0 {
            self.full_btran(dual_col);
            self.full_price(dual_col, dual_row);
            let (col_dual, row_dual) = self.work_dual[..num_tot].split_at_mut(self.num_col);
            for (d, &x) in col_dual.iter_mut().zip(&dual_row.array[..]) {
                *d -= x;
            }
            for (d, &x) in row_dual.iter_mut().zip(&dual_col.array[..]) {
                *d -= x;
            }
        }
        // Indicate that the dual infeasibility information isn't known
        *self.num_dual_infeasibilities = K_ILLEGAL_INFEASIBILITY_COUNT;
        *self.max_dual_infeasibility = K_ILLEGAL_INFEASIBILITY_MEASURE;
        *self.sum_dual_infeasibilities = K_ILLEGAL_INFEASIBILITY_MEASURE;
    }

    /// HEkk::zeroBasicDuals
    pub fn zero_basic_duals(&mut self) {
        for &i_var in &*self.basic_index {
            self.work_dual[i_var as usize] = 0.0;
        }
    }

    /// HEkk::computePrimalObjectiveValue
    pub fn compute_primal_objective_value(&mut self) {
        let mut value = 0.0;
        for i_row in 0..self.num_row {
            let i_var = self.basic_index[i_row] as usize;
            if i_var < self.num_col {
                value = self.base_value[i_row].mul_add_c(self.col_cost[i_var], value);
            }
        }
        let n = self.num_col;
        let (flag, work_value, cost) = (&self.nonbasic_flag[..n], &self.work_value[..n], &self.col_cost[..n]);
        for i in 0..n {
            if flag[i] != 0 {
                value = work_value[i].mul_add_c(cost[i], value);
            }
        }
        value *= self.cost_scale;
        // Objective value calculation is done using primal values and
        // original costs so offset is vanilla
        value += self.offset;
        *self.primal_objective_value = value;
    }

    /// HEkk::computeDualObjectiveValue
    pub fn compute_dual_objective_value(&mut self, phase: i32) {
        let mut value = 0.0;
        let n = self.num_tot();
        let (flag, work_value, work_dual) = (&self.nonbasic_flag[..n], &self.work_value[..n], &self.work_dual[..n]);
        for i in 0..n {
            if flag[i] != 0 {
                value = work_value[i].mul_add_c(work_dual[i], value);
            }
        }
        value *= self.cost_scale;
        if phase != 1 {
            // In phase 1 the dual objective has no objective shift.
            // Otherwise the shift is added according to the sign implied
            // by sense_
            value = (self.sense as f64).mul_add_c(self.offset, value);
        }
        *self.dual_objective_value = value;
    }

    // ---- Infeasibilities ----

    /// HEkk::computeSimplexPrimalInfeasible
    pub fn compute_simplex_primal_infeasible(&mut self) {
        let tolerance = self.primal_feasibility_tolerance;
        let mut infeas = Infeasibility::default();
        let primal_infeasibility = |value: f64, lower: f64, upper: f64| {
            if value < lower - tolerance {
                lower - value
            } else if value > upper + tolerance {
                value - upper
            } else {
                0.0
            }
        };
        let n = self.num_tot();
        let (flag, value, lower, upper) =
            (&self.nonbasic_flag[..n], &self.work_value[..n], &self.work_lower[..n], &self.work_upper[..n]);
        for i in 0..n {
            if flag[i] != 0 {
                infeas.add(primal_infeasibility(value[i], lower[i], upper[i]), tolerance, true);
            }
        }
        let n = self.num_row;
        let (value, lower, upper) = (&self.base_value[..n], &self.base_lower[..n], &self.base_upper[..n]);
        for i in 0..n {
            infeas.add(primal_infeasibility(value[i], lower[i], upper[i]), tolerance, true);
        }
        *self.num_primal_infeasibilities = infeas.num;
        *self.max_primal_infeasibility = infeas.max;
        *self.sum_primal_infeasibilities = infeas.sum;
    }

    /// HEkk::computeSimplexDualInfeasible
    pub fn compute_simplex_dual_infeasible(&mut self) {
        let tolerance = self.dual_feasibility_tolerance;
        let mut infeas = Infeasibility::default();
        let n = self.num_tot();
        let (flag, mv, work_dual, lower, upper) = (
            &self.nonbasic_flag[..n],
            &self.nonbasic_move[..n],
            &self.work_dual[..n],
            &self.work_lower[..n],
            &self.work_upper[..n],
        );
        for i in 0..n {
            if flag[i] == 0 {
                continue;
            }
            let dual = work_dual[i];
            let infeasibility = if is_inf(-lower[i]) && is_inf(upper[i]) {
                // Free: any nonzero dual value is infeasible
                dual.abs()
            } else {
                // Not free: any dual infeasibility is given by the dual
                // value signed by nonbasicMove
                -(mv[i] as f64) * dual
            };
            infeas.add(infeasibility, tolerance, false);
        }
        *self.num_dual_infeasibilities = infeas.num;
        *self.max_dual_infeasibility = infeas.max;
        *self.sum_dual_infeasibilities = infeas.sum;
    }

    /// HEkk::computeSimplexLpDualInfeasible: returns the infeasibilities
    /// (recorded in HEkk::analysis_ by the caller)
    pub fn compute_simplex_lp_dual_infeasible(&self) -> Infeasibility {
        let tolerance = self.dual_feasibility_tolerance;
        let mut infeas = Infeasibility::default();
        let dual_infeasibility = |dual: f64, lower: f64, upper: f64| {
            if is_inf(upper) {
                if is_inf(-lower) {
                    // Free: any nonzero dual value is infeasible
                    dual.abs()
                } else {
                    // Only lower bounded: a negative dual is infeasible
                    -dual
                }
            } else if is_inf(-lower) {
                // Only upper bounded: a positive dual is infeasible
                dual
            } else {
                // Boxed or fixed: any dual value is feasible
                0.0
            }
        };
        for i_col in 0..self.num_col {
            if self.nonbasic_flag[i_col] == 0 {
                continue;
            }
            let infeasibility =
                dual_infeasibility(self.work_dual[i_col], self.col_lower[i_col], self.col_upper[i_col]);
            infeas.add(infeasibility, tolerance, false);
        }
        for i_row in 0..self.num_row {
            let i_var = self.num_col + i_row;
            if self.nonbasic_flag[i_var] == 0 {
                continue;
            }
            let infeasibility =
                dual_infeasibility(-self.work_dual[i_var], self.row_lower[i_row], self.row_upper[i_row]);
            infeas.add(infeasibility, tolerance, false);
        }
        infeas
    }

    // ---- Basis changes ----

    /// HEkk::updatePivots, apart from recording the new basis in
    /// visited_basis_ and the status flags
    pub fn update_pivots(&mut self, variable_in: usize, row_out: usize, move_out: i32) {
        let variable_out = self.basic_index[row_out] as usize;
        // update hash value of basis
        hash::sparse_inverse_combine_index(self.basis_hash, variable_out as i32);
        hash::sparse_combine_index(self.basis_hash, variable_in as i32);
        // Incoming variable
        self.basic_index[row_out] = variable_in as i32;
        self.nonbasic_flag[variable_in] = 0;
        self.nonbasic_move[variable_in] = 0;
        self.base_lower[row_out] = self.work_lower[variable_in];
        self.base_upper[row_out] = self.work_upper[variable_in];
        // Outgoing variable
        self.nonbasic_flag[variable_out] = 1;
        let (lower, upper) = (self.work_lower[variable_out], self.work_upper[variable_out]);
        if lower == upper {
            self.work_value[variable_out] = lower;
            self.nonbasic_move[variable_out] = 0;
        } else if move_out == -1 {
            self.work_value[variable_out] = lower;
            self.nonbasic_move[variable_out] = 1;
        } else {
            self.work_value[variable_out] = upper;
            self.nonbasic_move[variable_out] = -1;
        }
        // Update the dual objective value
        let nw_value = self.work_value[variable_out];
        let vr_dual = self.work_dual[variable_out];
        let dl_dual_objective_value = nw_value * vr_dual;
        *self.updated_dual_objective_value += dl_dual_objective_value;
        *self.update_count += 1;
        // Update the number of basic logicals
        if variable_out < self.num_col {
            *self.num_basic_logicals += 1;
        }
        if variable_in < self.num_col {
            *self.num_basic_logicals -= 1;
        }
    }

    /// HEkk::updateFactor for a single (aq, ep) pair, when HSimplexNla has
    /// no ProductFormUpdate (the caller clears the factor's refactor info
    /// and checks the INVERT when debugging)
    pub fn update_factor(&mut self, column: &HVec, row_ep: &HVec, i_row: i32, hint: &mut i32) {
        self.factor.update(
            std::slice::from_ref(column),
            std::slice::from_ref(row_ep),
            &[i_row],
            hint,
            Some(&self.factor_a),
            self.basic_index,
        );
        if *self.update_count >= *self.update_limit {
            *hint = K_REBUILD_REASON_UPDATE_LIMIT_REACHED;
        }
        // Determine whether to reinvert based on the synthetic clock
        let reinvert_synthetic_clock = *self.total_synthetic_tick >= *self.build_synthetic_tick;
        let performed_min_updates = *self.update_count >= K_SYNTHETIC_TICK_REINVERSION_MIN_UPDATE_COUNT;
        if reinvert_synthetic_clock && performed_min_updates {
            *hint = K_REBUILD_REASON_SYNTHETIC_CLOCK_SAYS_INVERT;
        }
    }

    /// HEkk::updateMatrix: HighsSparseMatrix::update of ar_matrix_
    pub fn update_matrix(&mut self, variable_in: usize, variable_out: usize) {
        let num_col = self.num_col;
        if variable_in < num_col {
            for i_el in self.a.start[variable_in] as usize..self.a.start[variable_in + 1] as usize {
                let i_row = self.a.index[i_el] as usize;
                let mut i_find = self.ar_start[i_row] as usize;
                self.ar_p_end[i_row] -= 1;
                let i_swap = self.ar_p_end[i_row] as usize;
                while self.ar_index[i_find] != variable_in as i32 {
                    i_find += 1;
                }
                self.ar_index.swap(i_find, i_swap);
                self.ar_value.swap(i_find, i_swap);
            }
        }
        if variable_out < num_col {
            for i_el in self.a.start[variable_out] as usize..self.a.start[variable_out + 1] as usize {
                let i_row = self.a.index[i_el] as usize;
                let mut i_find = self.ar_p_end[i_row] as usize;
                let i_swap = self.ar_p_end[i_row] as usize;
                self.ar_p_end[i_row] += 1;
                while self.ar_index[i_find] != variable_out as i32 {
                    i_find += 1;
                }
                self.ar_index.swap(i_find, i_swap);
                self.ar_value.swap(i_find, i_swap);
            }
        }
    }

    // ---- Dual edge weights ----

    /// HEkk::computeDualSteepestEdgeWeight: `row_ep` has dimension num_row
    pub fn compute_dual_steepest_edge_weight(&mut self, i_row: usize, row_ep: &mut HVec) -> f64 {
        self.unit_btran_in_scaled_space(i_row, row_ep);
        row_ep.norm2()
    }

    /// computeDualSteepestEdgeWeight's BTRAN of e_{i_row}
    fn unit_btran_in_scaled_space(&mut self, i_row: usize, row_ep: &mut HVec) {
        row_ep.clear();
        row_ep.count = 1;
        row_ep.index[0] = i_row as i32;
        row_ep.array[i_row] = 1.0;
        row_ep.pack_flag = false;
        self.factor.btran(row_ep, *self.row_ep_density);
        let local_row_ep_density = (1.0 * row_ep.count as f64) / self.num_row as f64;
        update_operation_result_density(local_row_ep_density, self.row_ep_density);
    }

    /// HEkk::computeDualSteepestEdgeWeights (into which clang inlines
    /// computeDualSteepestEdgeWeight with norm2 contracted throughout)
    pub fn compute_dual_steepest_edge_weights(&mut self, row_ep: &mut HVec) {
        for i_row in 0..self.num_row {
            self.unit_btran_in_scaled_space(i_row, row_ep);
            self.dual_edge_weight[i_row] = row_ep.norm2_fused();
        }
    }

    /// HEkk::updateDualSteepestEdgeWeights
    pub fn update_dual_steepest_edge_weights(
        &mut self,
        row_out: usize,
        variable_in: usize,
        column: &HVec,
        new_pivotal_edge_weight: f64,
        kai: f64,
        dual_steepest_edge_array: &[f64],
    ) {
        let col_aq_scale = self.variable_scale_factor(variable_in);
        let col_ap_scale = self.basic_col_scale_factor(row_out);
        let inv_col_ap_scale = 1.0 / col_ap_scale;
        let (use_row_indices, to_entry) = sparse_loop_style(column.count, self.num_row);
        let convert_to_scaled_space = !self.simplex_in_scaled_space;
        for i_entry in 0..to_entry {
            let i_row = if use_row_indices { column.index[i_entry] as usize } else { i_entry };
            let mut aa_i_row = column.array[i_row];
            if aa_i_row == 0.0 {
                continue;
            }
            let mut dual_steepest_edge_array_value = dual_steepest_edge_array[i_row];
            if convert_to_scaled_space {
                let basic_col_scale = self.basic_col_scale_factor(i_row);
                aa_i_row /= basic_col_scale;
                aa_i_row *= col_aq_scale;
                dual_steepest_edge_array_value *= inv_col_ap_scale;
            }
            let w = &mut self.dual_edge_weight[i_row];
            let inner = new_pivotal_edge_weight.mul_add_c(aa_i_row, kai * dual_steepest_edge_array_value);
            *w = aa_i_row.mul_add_c(inner, *w);
            *w = w.max(K_MIN_DUAL_STEEPEST_EDGE_WEIGHT);
        }
    }

    /// HEkk::updateDualDevexWeights
    pub fn update_dual_devex_weights(&mut self, column: &HVec, new_pivotal_edge_weight: f64) {
        let (use_row_indices, to_entry) = sparse_loop_style(column.count, self.num_row);
        for i_entry in 0..to_entry {
            let i_row = if use_row_indices { column.index[i_entry] as usize } else { i_entry };
            let aa_i_row = column.array[i_row];
            let w = &mut self.dual_edge_weight[i_row];
            let candidate = new_pivotal_edge_weight * aa_i_row * aa_i_row;
            // std::max(w, candidate)
            if *w < candidate {
                *w = candidate;
            }
        }
    }

    // ---- Costs, bounds and nonbasic values ----

    /// HEkk::initialiseLpColCost
    pub fn initialise_lp_col_cost(&mut self) {
        let cost_scale_factor = 2.0f64.powi(self.cost_scale_factor);
        let sense = self.sense as f64;
        for i_col in 0..self.num_col {
            self.work_cost[i_col] = sense * cost_scale_factor * self.col_cost[i_col];
            self.work_shift[i_col] = 0.0;
        }
    }

    /// HEkk::initialiseLpRowCost
    pub fn initialise_lp_row_cost(&mut self) {
        for i_var in self.num_col..self.num_tot() {
            self.work_cost[i_var] = 0.0;
            self.work_shift[i_var] = 0.0;
        }
    }

    fn set_work_bound(&mut self, i_var: usize, lower: f64, upper: f64) {
        self.work_lower[i_var] = lower;
        self.work_upper[i_var] = upper;
        self.work_range[i_var] = upper - lower;
        self.work_lower_shift[i_var] = 0.0;
        self.work_upper_shift[i_var] = 0.0;
    }

    /// HEkk::initialiseLpColBound
    pub fn initialise_lp_col_bound(&mut self) {
        for i_col in 0..self.num_col {
            self.set_work_bound(i_col, self.col_lower[i_col], self.col_upper[i_col]);
        }
    }

    /// HEkk::initialiseLpRowBound
    pub fn initialise_lp_row_bound(&mut self) {
        for i_row in 0..self.num_row {
            self.set_work_bound(self.num_col + i_row, -self.row_upper[i_row], -self.row_lower[i_row]);
        }
    }

    /// HEkk::initialiseCost; `report` gets the figures that the caller
    /// logs
    pub fn initialise_cost(&mut self, algorithm: i32, perturb: bool, report: &mut CostPerturbationReport) {
        // Copy the cost
        self.initialise_lp_col_cost();
        self.initialise_lp_row_cost();
        *self.costs_shifted = false;
        *self.costs_perturbed = false;
        // Primal simplex costs are either from the LP or set specially in
        // phase 1
        if algorithm == ALGORITHM_PRIMAL {
            return;
        }
        // Dual simplex costs are either from the LP or perturbed
        if !perturb || *self.dual_simplex_cost_perturbation_multiplier == 0.0 {
            return;
        }
        // Perturb the original costs, scale down if is too big
        let report_cost_perturbation = self.output_flag;
        let num_col = self.num_col;
        let mut num_original_nonzero_cost: i32 = 0;
        let mut min_abs_cost = K_HIGHS_INF;
        let mut max_abs_cost: f64 = 0.0;
        let mut sum_abs_cost = 0.0;
        for i in 0..num_col {
            let abs_cost = self.work_cost[i].abs();
            if report_cost_perturbation {
                if abs_cost != 0.0 {
                    num_original_nonzero_cost += 1;
                    min_abs_cost = min_abs_cost.min(abs_cost);
                }
                sum_abs_cost += abs_cost;
            }
            max_abs_cost = max_abs_cost.max(abs_cost);
        }
        // Integer division by zero gives zero on arm64
        report.pct0 = (100 * num_original_nonzero_cost).checked_div(num_col as i32).unwrap_or(0);
        report.num_original_nonzero_cost = num_original_nonzero_cost;
        if report_cost_perturbation {
            if num_original_nonzero_cost != 0 {
                report.average_abs_cost = sum_abs_cost / num_original_nonzero_cost as f64;
            } else {
                min_abs_cost = 1.0;
                max_abs_cost = 1.0;
                report.average_abs_cost = 1.0;
            }
        }
        report.min_abs_cost = min_abs_cost;
        report.max_abs_cost = max_abs_cost;
        if max_abs_cost > 100.0 {
            max_abs_cost = max_abs_cost.sqrt().sqrt();
            report.large = true;
            report.large_max_abs_cost = max_abs_cost;
        }
        // If there are few boxed variables, we will just use simple
        // perturbation
        let num_tot = self.num_tot();
        let mut boxed_rate = 0.0;
        for i in 0..num_tot {
            boxed_rate += (self.work_range[i] < 1e30) as i32 as f64;
        }
        boxed_rate /= num_tot as f64;
        report.boxed_rate = boxed_rate;
        if boxed_rate < 0.01 {
            max_abs_cost = max_abs_cost.min(1.0);
            report.small_boxed_rate = true;
            report.small_boxed_max_abs_cost = max_abs_cost;
        }
        // Determine the perturbation base
        *self.cost_perturbation_max_abs_cost = max_abs_cost;
        let base = *self.dual_simplex_cost_perturbation_multiplier * 5e-7 * max_abs_cost;
        *self.cost_perturbation_base = base;
        // Now do the perturbation
        for i in 0..num_col {
            let lower = self.col_lower[i];
            let upper = self.col_upper[i];
            let cost = &mut self.work_cost[i];
            let xpert = (1.0 + self.num_tot_random_value[i]) * (cost.abs() + 1.0) * base;
            if lower <= -K_HIGHS_INF && upper >= K_HIGHS_INF {
                // Free - no perturb
            } else if upper >= K_HIGHS_INF {
                // Lower
                *cost += xpert;
            } else if lower <= -K_HIGHS_INF {
                // Upper
                *cost += -xpert;
            } else if lower != upper {
                // Boxed
                *cost += if *cost >= 0.0 { xpert } else { -xpert };
            }
            // Fixed - no perturb
        }
        let row_cost_perturbation_base = *self.dual_simplex_cost_perturbation_multiplier * 1e-12;
        report.row_cost_perturbation_base = row_cost_perturbation_base;
        for i in num_col..num_tot {
            let perturbation2 = (0.5 - self.num_tot_random_value[i]) * row_cost_perturbation_base;
            self.work_cost[i] += perturbation2;
        }
        *self.costs_perturbed = true;
        report.perturbed = true;
    }

    /// HEkk::initialiseBound
    pub fn initialise_bound(&mut self, algorithm: i32, solve_phase: i32, perturb: bool) {
        self.initialise_lp_col_bound();
        self.initialise_lp_row_bound();
        *self.bounds_shifted = false;
        *self.bounds_perturbed = false;
        let num_tot = self.num_tot();
        // Primal simplex bounds are either from the LP or perturbed
        if algorithm == ALGORITHM_PRIMAL {
            if !perturb || *self.primal_simplex_bound_perturbation_multiplier == 0.0 {
                return;
            }
            // Perturb the bounds
            let base = *self.primal_simplex_bound_perturbation_multiplier * 5e-7;
            for i_var in 0..num_tot {
                let mut lower = self.work_lower[i_var];
                let mut upper = self.work_upper[i_var];
                let fixed = lower == upper;
                // Don't perturb bounds of nonbasic fixed variables as they
                // stay nonbasic
                if self.nonbasic_flag[i_var] == 1 && fixed {
                    continue;
                }
                let random_value = self.num_tot_random_value[i_var];
                if lower > -K_HIGHS_INF {
                    if lower < -1.0 {
                        // lower -= random_value * base * (-lower)
                        lower = (random_value * base).mul_add_c(lower, lower);
                    } else if lower < 1.0 {
                        lower = (-random_value).mul_add_c(base, lower);
                    } else {
                        lower = (-(random_value * base)).mul_add_c(lower, lower);
                    }
                    self.work_lower[i_var] = lower;
                }
                if upper < K_HIGHS_INF {
                    if upper < -1.0 {
                        // upper += random_value * base * (-upper)
                        upper = (-(random_value * base)).mul_add_c(upper, upper);
                    } else if upper < 1.0 {
                        upper = random_value.mul_add_c(base, upper);
                    } else {
                        upper = (random_value * base).mul_add_c(upper, upper);
                    }
                    self.work_upper[i_var] = upper;
                }
                self.work_range[i_var] = self.work_upper[i_var] - self.work_lower[i_var];
                if self.nonbasic_flag[i_var] == 0 {
                    continue;
                }
                // Set values of nonbasic variables
                if self.nonbasic_move[i_var] > 0 {
                    self.work_value[i_var] = lower;
                } else if self.nonbasic_move[i_var] < 0 {
                    self.work_value[i_var] = upper;
                }
            }
            for i_row in 0..self.num_row {
                let i_var = self.basic_index[i_row] as usize;
                self.base_lower[i_row] = self.work_lower[i_var];
                self.base_upper[i_row] = self.work_upper[i_var];
            }
            *self.bounds_perturbed = true;
            return;
        }
        // Dual simplex bounds are either from the LP or set to special
        // values in phase 1
        if solve_phase == SOLVE_PHASE_2 {
            return;
        }
        // In phase 1 the primal bounds are set so that the dual objective
        // is the negation of the sum of dual infeasibilities
        let inf = K_HIGHS_INF;
        for i in 0..num_tot {
            let (lower, upper) = if self.work_lower[i] == -inf && self.work_upper[i] == inf {
                (-1000.0, 1000.0) // FREE
            } else if self.work_lower[i] == -inf {
                (-1.0, 0.0) // UPPER
            } else if self.work_upper[i] == inf {
                (0.0, 1.0) // LOWER
            } else {
                (0.0, 0.0) // BOXED or FIXED
            };
            self.work_lower[i] = lower;
            self.work_upper[i] = upper;
            self.work_range[i] = upper - lower;
        }
    }

    /// HEkk::setNonbasicMove: nonbasic_move must have num_tot entries
    pub fn set_nonbasic_move(&mut self) {
        let num_col = self.num_col;
        for i_var in 0..self.num_tot() {
            if self.nonbasic_flag[i_var] == 0 {
                // Basic variable
                self.nonbasic_move[i_var] = MOVE_ZE;
                continue;
            }
            // Nonbasic variable
            let (lower, upper) = if i_var < num_col {
                (self.col_lower[i_var], self.col_upper[i_var])
            } else {
                let i_row = i_var - num_col;
                (-self.row_upper[i_row], -self.row_lower[i_row])
            };
            self.nonbasic_move[i_var] = if lower == upper {
                // Fixed
                MOVE_ZE
            } else if !is_inf(-lower) {
                // Finite lower bound so boxed or lower
                if !is_inf(upper) {
                    // Boxed: bound of original LP that is closer to zero
                    if lower.abs() < upper.abs() {
                        MOVE_UP
                    } else {
                        MOVE_DN
                    }
                } else {
                    // Lower (since upper bound is infinite)
                    MOVE_UP
                }
            } else if !is_inf(upper) {
                // Upper
                MOVE_DN
            } else {
                // FREE
                MOVE_ZE
            };
        }
    }

    /// HEkk::initialiseNonbasicValueAndMove
    pub fn initialise_nonbasic_value_and_move(&mut self) {
        for i_var in 0..self.num_tot() {
            if self.nonbasic_flag[i_var] == 0 {
                // Basic variable
                self.nonbasic_move[i_var] = MOVE_ZE;
                continue;
            }
            // Nonbasic variable
            let lower = self.work_lower[i_var];
            let upper = self.work_upper[i_var];
            let original_move = self.nonbasic_move[i_var];
            let (value, mv) = if lower == upper {
                // Fixed
                (lower, MOVE_ZE)
            } else if !is_inf(-lower) {
                // Finite lower bound so boxed or lower
                if !is_inf(upper) {
                    // Boxed: set at the bound given by the move, correcting
                    // an invalid move to lower
                    if original_move == MOVE_DN {
                        (upper, MOVE_DN)
                    } else {
                        (lower, MOVE_UP)
                    }
                } else {
                    // Lower
                    (lower, MOVE_UP)
                }
            } else if !is_inf(upper) {
                // Upper
                (upper, MOVE_DN)
            } else {
                // FREE
                (0.0, MOVE_ZE)
            };
            self.nonbasic_move[i_var] = mv;
            self.work_value[i_var] = value;
        }
    }
}

mod ffi {
    use super::*;

    /// # Safety
    /// As for CEkk::view; `v` (if any) as for CHVec::view
    unsafe fn with<R>(ekk: *const CEkk, f: impl FnOnce(&mut EkkView) -> R) -> R {
        f(&mut (*ekk).view())
    }

    /// Run `f` on a view of a CHVec, then copy its scalars back
    unsafe fn hv<R>(v: *mut CHVec, f: impl FnOnce(&mut HVec) -> R) -> R {
        let c = &mut *v;
        let mut view = c.view();
        let r = f(&mut view);
        c.store(&view);
        r
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_ekk_compute_primal(ekk: *const CEkk, primal_col: *mut CHVec) {
        with(ekk, |e| hv(primal_col, |v| e.compute_primal(v)))
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_ekk_compute_dual(ekk: *const CEkk, dual_col: *mut CHVec, dual_row: *mut CHVec) {
        with(ekk, |e| hv(dual_col, |c| hv(dual_row, |r| e.compute_dual(c, r))))
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_ekk_full_btran(ekk: *const CEkk, buffer: *mut CHVec) {
        with(ekk, |e| hv(buffer, |v| e.full_btran(v)))
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_ekk_full_price(ekk: *const CEkk, full_col: *mut CHVec, full_row: *mut CHVec) {
        with(ekk, |e| hv(full_col, |c| hv(full_row, |r| e.full_price(c, r))))
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_ekk_unit_btran(ekk: *const CEkk, i_row: i32, row_ep: *mut CHVec) {
        with(ekk, |e| hv(row_ep, |v| e.unit_btran(i_row as usize, v)))
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_ekk_pivot_column_ftran(ekk: *const CEkk, i_col: i32, col_aq: *mut CHVec) {
        with(ekk, |e| hv(col_aq, |v| e.pivot_column_ftran(i_col as usize, v)))
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_ekk_tableau_row_price(ekk: *const CEkk, row_ep: *mut CHVec, row_ap: *mut CHVec) {
        with(ekk, |e| hv(row_ep, |ep| hv(row_ap, |ap| e.tableau_row_price(ep, ap))))
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_ekk_transform_for_update(
        ekk: *const CEkk,
        aq: *mut CHVec,
        ep: *mut CHVec,
        variable_in: i32,
        row_out: i32,
    ) {
        with(ekk, |e| {
            hv(aq, |aq| hv(ep, |ep| e.transform_for_update(aq, ep, variable_in as usize, row_out as usize)))
        })
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_ekk_compute_simplex_primal_infeasible(ekk: *const CEkk) {
        with(ekk, |e| e.compute_simplex_primal_infeasible())
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_ekk_compute_simplex_dual_infeasible(ekk: *const CEkk) {
        with(ekk, |e| e.compute_simplex_dual_infeasible())
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_ekk_compute_simplex_lp_dual_infeasible(ekk: *const CEkk) -> Infeasibility {
        with(ekk, |e| e.compute_simplex_lp_dual_infeasible())
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_ekk_compute_primal_objective_value(ekk: *const CEkk) {
        with(ekk, |e| e.compute_primal_objective_value())
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_ekk_compute_dual_objective_value(ekk: *const CEkk, phase: i32) {
        with(ekk, |e| e.compute_dual_objective_value(phase))
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_ekk_zero_basic_duals(ekk: *const CEkk) {
        with(ekk, |e| e.zero_basic_duals())
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_ekk_update_pivots(ekk: *const CEkk, variable_in: i32, row_out: i32, move_out: i32) {
        with(ekk, |e| e.update_pivots(variable_in as usize, row_out as usize, move_out))
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_ekk_update_factor(
        ekk: *const CEkk,
        column: *mut CHVec,
        row_ep: *mut CHVec,
        i_row: i32,
        hint: *mut i32,
    ) {
        with(ekk, |e| hv(column, |c| hv(row_ep, |r| e.update_factor(c, r, i_row, &mut *hint))))
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_ekk_update_matrix(ekk: *const CEkk, variable_in: i32, variable_out: i32) {
        with(ekk, |e| e.update_matrix(variable_in as usize, variable_out as usize))
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_ekk_compute_dual_steepest_edge_weight(
        ekk: *const CEkk,
        i_row: i32,
        row_ep: *mut CHVec,
    ) -> f64 {
        with(ekk, |e| hv(row_ep, |v| e.compute_dual_steepest_edge_weight(i_row as usize, v)))
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_ekk_compute_dual_steepest_edge_weights(ekk: *const CEkk, row_ep: *mut CHVec) {
        with(ekk, |e| hv(row_ep, |v| e.compute_dual_steepest_edge_weights(v)))
    }

    /// # Safety
    /// `dse_array` has num_row entries (or is unused: null when the column
    /// is empty)
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_ekk_update_dual_steepest_edge_weights(
        ekk: *const CEkk,
        row_out: i32,
        variable_in: i32,
        column: *mut CHVec,
        new_pivotal_edge_weight: f64,
        kai: f64,
        dse_array: *const f64,
    ) {
        with(ekk, |e| {
            let dse = sl(dse_array, e.num_row as i32);
            hv(column, |c| {
                e.update_dual_steepest_edge_weights(
                    row_out as usize,
                    variable_in as usize,
                    c,
                    new_pivotal_edge_weight,
                    kai,
                    dse,
                )
            })
        })
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_ekk_update_dual_devex_weights(
        ekk: *const CEkk,
        column: *mut CHVec,
        new_pivotal_edge_weight: f64,
    ) {
        with(ekk, |e| hv(column, |c| e.update_dual_devex_weights(c, new_pivotal_edge_weight)))
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_ekk_initialise_cost(
        ekk: *const CEkk,
        algorithm: i32,
        perturb: bool,
        report: *mut CostPerturbationReport,
    ) {
        with(ekk, |e| e.initialise_cost(algorithm, perturb, &mut *report))
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_ekk_initialise_bound(ekk: *const CEkk, algorithm: i32, solve_phase: i32, perturb: bool) {
        with(ekk, |e| e.initialise_bound(algorithm, solve_phase, perturb))
    }

    /// which: 0 col cost, 1 row cost, 2 col bound, 3 row bound
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_ekk_initialise_lp(ekk: *const CEkk, which: i32) {
        with(ekk, |e| match which {
            0 => e.initialise_lp_col_cost(),
            1 => e.initialise_lp_row_cost(),
            2 => e.initialise_lp_col_bound(),
            _ => e.initialise_lp_row_bound(),
        })
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_ekk_set_nonbasic_move(ekk: *const CEkk) {
        with(ekk, |e| e.set_nonbasic_move())
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_ekk_initialise_nonbasic_value_and_move(ekk: *const CEkk) {
        with(ekk, |e| e.initialise_nonbasic_value_and_move())
    }

    // Kernels too small to be worth a view: their data comes directly

    /// # Safety
    /// work_cost covers i_var and the basic variables, basic_index the rows
    /// of tableau_index, tableau_array the rows
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_ekk_compute_dual_for_tableau_column(
        work_cost: *const f64,
        n_work_cost: i32,
        basic_index: *const i32,
        num_row: i32,
        i_var: i32,
        count: i32,
        tableau_index: *const i32,
        tableau_array: *const f64,
        n_tableau_array: i32,
    ) -> f64 {
        compute_dual_for_tableau_column(
            sl(work_cost, n_work_cost),
            sl(basic_index, num_row),
            i_var as usize,
            sl(tableau_index, count),
            sl(tableau_array, n_tableau_array),
        )
    }

    /// # Safety
    /// The arrays have num_tot entries
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_ekk_flip_bound(
        num_tot: i32,
        nonbasic_move: *mut i8,
        work_value: *mut f64,
        work_lower: *const f64,
        work_upper: *const f64,
        i_col: i32,
    ) {
        flip_bound(
            sl_mut(nonbasic_move, num_tot),
            sl_mut(work_value, num_tot),
            sl(work_lower, num_tot),
            sl(work_upper, num_tot),
            i_col as usize,
        )
    }

    #[no_mangle]
    pub extern "C" fn highs_rs_ekk_reinvert_on_numerical_trouble(
        alpha_from_col: f64,
        alpha_from_row: f64,
        numerical_trouble_tolerance: f64,
        update_count: i32,
        current_pivot_threshold: f64,
    ) -> NumericalTrouble {
        reinvert_on_numerical_trouble(
            alpha_from_col,
            alpha_from_row,
            numerical_trouble_tolerance,
            update_count,
            current_pivot_threshold,
        )
    }

    /// Returns use_col_price + 2 * use_row_price_w_switch
    #[no_mangle]
    pub extern "C" fn highs_rs_ekk_choose_price_technique(price_strategy: i32, row_ep_density: f64) -> i32 {
        let (col, row_switch) = choose_price_technique(price_strategy, row_ep_density);
        col as i32 + 2 * row_switch as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_kernels() {
        let mut density = 0.5;
        update_operation_result_density(1.0, &mut density);
        assert_eq!(density, 0.5f64.mul_add_c(0.95, 0.05));
        assert_eq!(choose_price_technique(PRICE_COL, 0.0), (true, false));
        assert_eq!(choose_price_technique(PRICE_ROW_SWITCH_COL_SWITCH, 0.8), (true, true));
        assert_eq!(choose_price_technique(PRICE_ROW_SWITCH_COL_SWITCH, 0.5), (false, true));
        assert_eq!(sparse_loop_style(3, 10), (true, 3));
        assert_eq!(sparse_loop_style(4, 10), (false, 10));
        // c_j - sum_i a_ij c_B(i): 5 - (2 * 1 + 1 * 3)
        let dual = compute_dual_for_tableau_column(&[5.0, 1.0, 3.0], &[1, 2], 0, &[0, 1], &[2.0, 1.0]);
        assert_eq!(dual, 0.0);
        let (mut mv, mut value) = ([1i8], [0.0]);
        flip_bound(&mut mv, &mut value, &[-1.0], &[2.0], 0);
        assert_eq!((mv[0], value[0]), (-1, 2.0));
        let t = reinvert_on_numerical_trouble(1.0, 2.0, 1e-7, 5, 0.1);
        assert!(t.reinvert && t.measure == 1.0 && t.new_pivot_threshold == 0.5);
        assert!(!reinvert_on_numerical_trouble(1.0, 2.0, 1e-7, 0, 0.1).reinvert);
    }
}
