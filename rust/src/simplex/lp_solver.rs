//! The simplex engine's data, owned by Rust: everything that HEkk
//! (highs/simplex/HEkk.h) held but the pointers to the C++ options,
//! callback, timer and analysis. The LP being solved ([`LpSolver::lp`]) is
//! a Rust copy of the C++ HighsLp (made by HEkk::moveLp, scaled and
//! dualized in place by solveLpSimplex, simplex/app.rs); every call gets a
//! view of it ([`LpsEnv::lp`], filled by [`LpSolver::env_of`]) with the
//! option values and the C++ host functions. The C++ HEkk is a shell that
//! owns an [`LpSolver`] and forwards to it (highs/simplex/HEkkRust.cpp).
//!
//! The simplex kernels (ekk.rs, hekk.rs, dual.rs, primal.rs) work on
//! [`CEkk`] / [`CHekk`] views of pointers: [`LpSolver::c_ekk`] and
//! [`LpSolver::c_hekk`] build them from the fields here, after the vectors
//! have been sized for the call, so the kernels are unchanged.
//!
//! What C++ still reads or writes in place is [`EkkShared`] (the status
//! flags, the model status, iteration count, exit algorithm, the ray
//! indices and signs, the dual-values flag and the info_ values the Highs
//! API reports): the C++ shell holds references into it with HEkk's old
//! member names. The basis, the work arrays and the edge weights are read
//! through accessors.
//!
//! The simplex NLA (HSimplexNla) is the Rust factor plus the scaling of the
//! basis matrix: the factor is owned here; which LP's scale factors apply
//! (simplex_nla_.lp_ / scale_, set by setNlaPointersForLpAndScale) is kept
//! by the C++ shell, which passes that LP's dimensions and scale in
//! [`LpsEnv`]. The factor's constraint matrix is the LP's or, if the LP
//! has scale factors but is unscaled, the scaled copy kept here
//! (getScaledAMatrixPointer).

use std::cell::Cell;
use std::ffi::c_void;

use super::basis_records::{BasisRecords, REASON_ALL};
use super::ekk::{CEkk, CSlice, SimplexStatus};
use super::hekk::{self, CHekk, Host, Shared, LOG_ERROR, LOG_INFO};
use super::report::SimplexReport;
use crate::factor::HFactor;
use crate::ffi::CHVec;
use crate::hvector::OwnedHVec;
use crate::lp_data::ffi::{CIndexCollection, CLp, RsMut};
use crate::lp_data::lp::{rs, Lp};
use crate::sprintf;
use crate::util::hash;
use crate::util::random::HighsRandom;

const INF: f64 = f64::INFINITY;
const K_HIGHS_IINF: i32 = i32::MAX;

// LpAction
const LP_SCALE: i32 = 0;
const LP_NEW_COSTS: i32 = 1;
const LP_NEW_BOUNDS: i32 = 2;
const LP_NEW_BASIS: i32 = 3;
const LP_NEW_COLS: i32 = 4;
const LP_NEW_ROWS: i32 = 5;
const LP_DEL_COLS: i32 = 6;
const LP_DEL_NONBASIC_COLS: i32 = 7;
const LP_DEL_ROWS: i32 = 8;
const LP_SCALED_COL: i32 = 10;
const LP_SCALED_ROW: i32 = 11;
const LP_BACKTRACKING: i32 = 12;

// HighsBasisStatus
const BS_LOWER: u8 = 0;
const BS_BASIC: u8 = 1;
const BS_UPPER: u8 = 2;
const BS_ZERO: u8 = 3;
const BS_NONBASIC: u8 = 4;

const MOVE_UP: i8 = 1;
const MOVE_DN: i8 = -1;
const MOVE_ZE: i8 = 0;
const ILLEGAL_MOVE: i8 = -99;
const ILLEGAL_FLAG: i8 = -99;

// MatrixFormat::kColwise
const MATRIX_FORMAT_COLWISE: i32 = 1;

const SIMPLEX_EDGE_WEIGHT_STRATEGY_CHOOSE: i32 = -1;
const SIMPLEX_CONCURRENCY_LIMIT: i32 = 8;
const UPDATE_METHOD_FT: i32 = 1;
const K_MIN_PIVOT_THRESHOLD: f64 = 8e-4;
const K_MAX_PIVOT_THRESHOLD: f64 = 0.5;
const K_MIN_PIVOT_TOLERANCE: f64 = 0.0;
const K_MAX_PIVOT_TOLERANCE: f64 = 1.0;

/// std::max(lo, std::min(x, hi))
fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    let m = if hi < x { hi } else { x };
    if m < lo {
        lo
    } else {
        m
    }
}

/// SimplexBasis (simplex/SimplexStruct.h)
#[derive(Clone)]
pub struct SimplexBasis {
    pub basic_index: Vec<i32>,
    pub nonbasic_flag: Vec<i8>,
    pub nonbasic_move: Vec<i8>,
    pub hash: u64,
    pub debug_id: i32,
    pub debug_update_count: i32,
    pub debug_origin_name: String,
}

impl Default for SimplexBasis {
    fn default() -> Self {
        SimplexBasis {
            basic_index: Vec::new(),
            nonbasic_flag: Vec::new(),
            nonbasic_move: Vec::new(),
            hash: 0,
            debug_id: -1,
            debug_update_count: -1,
            debug_origin_name: String::new(),
        }
    }
}

impl SimplexBasis {
    /// SimplexBasis::clear
    pub fn clear(&mut self) {
        self.hash = 0;
        self.basic_index.clear();
        self.nonbasic_flag.clear();
        self.nonbasic_move.clear();
        self.debug_id = -1;
        self.debug_update_count = -1;
        self.debug_origin_name = "None".into();
    }

    /// SimplexBasis::setup
    pub fn setup(&mut self, num_col: i32, num_row: i32) {
        self.hash = 0;
        self.basic_index.resize(num_row.max(0) as usize, 0);
        let num_tot = (num_col + num_row).max(0) as usize;
        self.nonbasic_flag.resize(num_tot, 0);
        self.nonbasic_move.resize(num_tot, 0);
        self.debug_id = -1;
        self.debug_update_count = -1;
        self.debug_origin_name = "None".into();
    }
}

/// The info_ values that C++ reads (HighsSimplexInfo's names): mirrored by
/// HEkkInfo in highs/simplex/HEkkRust.h
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SharedInfo {
    pub dual_objective_value: f64,
    pub primal_objective_value: f64,
    pub max_primal_infeasibility: f64,
    pub sum_primal_infeasibilities: f64,
    pub max_dual_infeasibility: f64,
    pub sum_dual_infeasibilities: f64,
    pub factor_pivot_threshold: f64,
    pub row_ep_density: f64,
    pub col_aq_density: f64,
    pub num_primal_infeasibilities: i32,
    pub num_dual_infeasibilities: i32,
    pub dual_edge_weight_strategy: i32,
}

/// HEkk's scalars that C++ reads and writes in place: mirrored by
/// HEkkShared in highs/simplex/HEkkRust.h
#[repr(C)]
pub struct EkkShared {
    pub info: SharedInfo,
    pub model_status: i32,
    pub iteration_count: i32,
    pub exit_algorithm: i32,
    pub debug_solve_call_num: i32,
    pub debug_initial_build_synthetic_tick: i32,
    pub dual_ray_index: i32,
    pub dual_ray_sign: i32,
    pub primal_ray_index: i32,
    pub primal_ray_sign: i32,
    pub status: SimplexStatus,
    pub dual_values_valid: bool,
    /// Whether the simplex NLA has an LP (simplex_nla_.lp_ != NULL): C++
    /// sets it with the LP, Rust clears it with the NLA
    pub nla_lp_set: bool,
}

const _: () = assert!(std::mem::size_of::<SharedInfo>() == 88);
const _: () = assert!(std::mem::size_of::<EkkShared>() == 144);
const _: () = assert!(std::mem::offset_of!(EkkShared, status) == 124);

/// The scalars of HighsSimplexInfo that only Rust uses
#[derive(Clone, Copy, Default)]
struct Info {
    backtracking: bool,
    valid_backtracking_basis: bool,
    bt_costs_shifted: i32,
    bt_costs_perturbed: i32,
    bt_bounds_shifted: i32,
    bt_bounds_perturbed: i32,
    simplex_strategy: i32,
    price_strategy: i32,
    dual_simplex_cost_perturbation_multiplier: f64,
    primal_simplex_phase1_cost_perturbation_multiplier: f64,
    primal_simplex_bound_perturbation_multiplier: f64,
    update_limit: i32,
    control_iteration_count0: i32,
    row_ap_density: f64,
    row_dse_density: f64,
    col_steepest_edge_density: f64,
    col_basic_feasibility_change_density: f64,
    row_basic_feasibility_change_density: f64,
    col_bfrt_density: f64,
    primal_col_density: f64,
    dual_col_density: f64,
    allow_dual_steepest_edge_to_devex_switch: bool,
    dual_steepest_edge_weight_log_error_threshold: f64,
    costly_dse_frequency: f64,
    num_costly_dse_iteration: i32,
    costly_dse_measure: f64,
    average_log_low_dse_weight_error: f64,
    average_log_high_dse_weight_error: f64,
    run_quiet: bool,
    store_squared_primal_infeasibility: bool,
    allow_cost_shifting: bool,
    allow_cost_perturbation: bool,
    allow_bound_perturbation: bool,
    costs_shifted: bool,
    costs_perturbed: bool,
    bounds_shifted: bool,
    bounds_perturbed: bool,
    dual_phase1_iteration_count: i32,
    dual_phase2_iteration_count: i32,
    primal_phase1_iteration_count: i32,
    primal_phase2_iteration_count: i32,
    primal_bound_swap: i32,
    iteration_count0: i32,
    dual_phase1_iteration_count0: i32,
    dual_phase2_iteration_count0: i32,
    primal_phase1_iteration_count0: i32,
    primal_phase2_iteration_count0: i32,
    primal_bound_swap0: i32,
    min_concurrency: i32,
    num_concurrency: i32,
    max_concurrency: i32,
    update_count: i32,
    updated_dual_objective_value: f64,
    updated_primal_objective_value: f64,
    num_basic_logicals: i32,
}

/// A HighsSparseMatrix: the row-wise matrix partitioned by nonbasicFlag
/// (HEkk::ar_matrix_) or the scaled copy of the constraint matrix
#[derive(Clone)]
struct RowMatrix {
    rowwise: bool,
    num_col: i32,
    num_row: i32,
    start: Vec<i32>,
    p_end: Vec<i32>,
    index: Vec<i32>,
    value: Vec<f64>,
}

impl Default for RowMatrix {
    /// HighsSparseMatrix() and HighsSparseMatrix::clear
    fn default() -> Self {
        RowMatrix { rowwise: false, num_col: 0, num_row: 0, start: vec![0], p_end: Vec::new(), index: Vec::new(), value: Vec::new() }
    }
}

impl RowMatrix {
    /// HighsSparseMatrix::numNz
    fn num_nz(&self) -> i32 {
        let k = if self.rowwise { self.num_row } else { self.num_col };
        self.start.get(k as usize).copied().unwrap_or(0)
    }
}

/// What HFactor kept besides the Rust factor: its dimensions, pivot
/// parameters and log level (copied at set-up), and which matrix it
/// factorizes
#[derive(Clone, Copy, Default)]
struct FactorShell {
    set_up: bool,
    num_col: i32,
    num_row: i32,
    pivot_threshold: f64,
    pivot_tolerance: f64,
    time_limit: f64,
    /// The factor's matrix is the scaled copy (else the LP's)
    uses_scaled_copy: bool,
    /// highsLogDev level of the factor's log options (0 if they cannot
    /// print)
    dev_level: i32,
}

/// SimplexIterate (HSimplexNla::simplex_iterate_): the INVERT is saved by
/// the factor
#[derive(Clone, Default)]
struct Iterate {
    valid: bool,
    basis: SimplexBasis,
    dual_edge_weight: Vec<f64>,
}

/// The option values that the simplex reads: mirrored by LpsOptions in
/// highs/simplex/HEkkRust.h
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LpsOptions {
    pub primal_feasibility_tolerance: f64,
    pub dual_feasibility_tolerance: f64,
    pub time_limit: f64,
    pub objective_bound: f64,
    pub dual_simplex_pivot_growth_tolerance: f64,
    pub small_matrix_value: f64,
    pub dual_steepest_edge_weight_error_tolerance: f64,
    pub rebuild_refactor_solution_error_tolerance: f64,
    pub factor_pivot_tolerance: f64,
    pub factor_pivot_threshold: f64,
    pub dual_simplex_cost_perturbation_multiplier: f64,
    pub primal_simplex_bound_perturbation_multiplier: f64,
    pub dual_steepest_edge_weight_log_error_threshold: f64,
    pub cost_scale_factor: i32,
    pub log_dev_level: i32,
    /// *log_options.log_dev_level, if *log_options.output_flag, else 0
    pub dev_level: i32,
    pub simplex_primal_edge_weight_strategy: i32,
    pub simplex_iteration_limit: i32,
    pub simplex_update_limit: i32,
    pub max_dual_simplex_cleanup_level: i32,
    pub max_dual_simplex_phase1_cleanup_level: i32,
    pub simplex_dse_exact_init_max_rows: i32,
    pub simplex_strategy: i32,
    pub simplex_min_concurrency: i32,
    pub simplex_max_concurrency: i32,
    pub simplex_dual_edge_weight_strategy: i32,
    pub simplex_price_strategy: i32,
    pub random_seed: i32,
    pub output_flag: bool,
    pub no_unnecessary_rebuild_refactor: bool,
    pub allow_unbounded_or_infeasible: bool,
    pub less_infeasible_dse_check: bool,
    pub less_infeasible_dse_choose_row: bool,
    pub simplex_keep_random_vectors: bool,
}

/// What a call needs from C++: a view of HEkk's LP, the LP of the simplex
/// NLA (dimensions and scale factors), the option values and the host
/// functions. Mirrored by LpsEnv in highs/simplex/HEkkRust.h
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LpsEnv {
    pub lp: CLp,
    pub model_name: RsMut<u8>,
    pub nla_num_col: i32,
    pub nla_num_row: i32,
    pub nla_has_scale: bool,
    pub nla_col_scale: RsMut<f64>,
    pub nla_row_scale: RsMut<f64>,
    pub opt: LpsOptions,
    pub host: Host,
    /// HighsSimplexAnalysis's report data
    pub report: *mut SimplexReport,
    /// HEkk::simplex_stats_.num_invert
    pub num_invert: *mut i32,
    pub num_threads: i32,
    /// Whether a user callback for simplex interrupts is active
    pub interrupt_callback: bool,
}

fn cs<T>(v: &RsMut<T>) -> CSlice<T> {
    CSlice { p: v.ptr, n: v.len as i32 }
}

fn vs<T>(v: &mut Vec<T>) -> CSlice<T> {
    CSlice { p: v.as_mut_ptr(), n: v.len() as i32 }
}

fn sh<T: Copy>(p: &mut T) -> Shared<T> {
    Shared::new(p as *mut T)
}

impl LpsEnv {
    /// highsLogDev
    fn dev(&self, t: i32, msg: impl FnOnce() -> String) {
        let level = self.opt.dev_level;
        if level > 0 && !(t == hekk::LOG_DETAILED && level < 2) && !(t == hekk::LOG_VERBOSE && level < 3) {
            self.emit(1, t, &msg());
        }
    }

    /// highsLogUser
    fn user(&self, t: i32, msg: &str) {
        if self.opt.output_flag {
            self.emit(0, t, msg);
        }
    }

    fn emit(&self, channel: i32, t: i32, msg: &str) {
        let c = std::ffi::CString::new(msg).unwrap_or_default();
        (self.host.log)(self.host.ctx, channel, t, c.as_ptr());
    }
}

/// HEkk's data but its LP: see the module comment
pub struct LpSolver {
    pub sh: EkkShared,
    info: Info,
    pub basis: SimplexBasis,
    // info_'s vectors
    work_cost: Vec<f64>,
    pub work_dual: Vec<f64>,
    work_shift: Vec<f64>,
    work_lower: Vec<f64>,
    work_upper: Vec<f64>,
    work_range: Vec<f64>,
    work_value: Vec<f64>,
    work_lower_shift: Vec<f64>,
    work_upper_shift: Vec<f64>,
    base_lower: Vec<f64>,
    base_upper: Vec<f64>,
    base_value: Vec<f64>,
    num_tot_random_value: Vec<f64>,
    num_tot_permutation: Vec<i32>,
    num_col_permutation: Vec<i32>,
    devex_index: Vec<i32>,
    bt_basis: SimplexBasis,
    bt_work_shift: Vec<f64>,
    bt_edge_weight: Vec<f64>,
    random: u64,
    pub dual_edge_weight: Vec<f64>,
    scattered_dual_edge_weight: Vec<f64>,
    saved_dual_edge_weight: Vec<f64>,
    simplex_in_scaled_space: bool,
    ar: RowMatrix,
    ar_matrix_is_scaled: bool,
    random_vectors_drawn: bool,
    dual_values_scaled: bool,
    dual_values_cost_hash: u64,
    dual_values_basis_hash: u64,
    fresh_unperturbed_dual: bool,
    fresh_dual: bool,
    fresh_primal: bool,
    /// The scaled copy of the constraint matrix (scaled_a_matrix_)
    scaled_a: RowMatrix,
    factor: HFactor,
    fs: FactorShell,
    nla_build_synthetic_tick: f64,
    iterate: Iterate,
    cost_scale: f64,
    cost_perturbation_base: f64,
    cost_perturbation_max_abs_cost: f64,
    dual_simplex_cleanup_level: i32,
    dual_simplex_phase1_cleanup_level: i32,
    previous_iteration_cycling_detected: i32,
    solve_bailout: bool,
    called_return_from_solve: bool,
    return_primal_solution_status: i32,
    return_dual_solution_status: i32,
    pub dual_ray_value: Vec<f64>,
    pub primal_ray_value: Vec<f64>,
    edge_weight_error: f64,
    build_synthetic_tick: f64,
    total_synthetic_tick: f64,
    pub records: BasisRecords,
    /// HEkk's LP (HEkk::lp_), copied from the C++ LP being solved
    pub lp: Lp,
    /// Whether the simplex NLA's LP is `lp` (else a C++ LP, whose
    /// dimensions and scale factors come with each call), and whether its
    /// scale factors apply (decided when it is set, as HSimplexNla::scale_)
    nla_rust: bool,
    nla_rust_scale: bool,
    /// What dualize keeps to undualize
    pub dz: Dualized,
}

/// What HEkk::dualize keeps (original_* and upper_bound_*)
#[derive(Clone, Default)]
pub struct Dualized {
    pub original_num_col: i32,
    pub original_num_row: i32,
    pub original_num_nz: i32,
    pub original_offset: f64,
    pub original_col_cost: Vec<f64>,
    pub original_col_lower: Vec<f64>,
    pub original_col_upper: Vec<f64>,
    pub original_row_lower: Vec<f64>,
    pub original_row_upper: Vec<f64>,
    pub upper_bound_col: Vec<i32>,
    pub upper_bound_row: Vec<i32>,
}

/// The result of a solve
#[repr(C)]
pub struct LpsSolveOut {
    pub status: i32,
    pub invert_num_el: i32,
    pub basis_matrix_num_el: i32,
}

/// Scatter the DSE weights of the first num_weighted_row rows over the
/// variables of an LP with new_num_row rows (HEkk::scatterDualEdgeWeights):
/// the weight of a row is a property of its basic variable. new_row_index
/// maps old rows to new ones (-1 if deleted), and is None if rows are only
/// appended. Empty if there are no weights.
pub(crate) fn scatter_dual_edge_weights(
    status: &SimplexStatus,
    basic_index: &[i32],
    num_tot: usize,
    dual_edge_weight: &[f64],
    num_weighted_row: i32,
    new_num_row: i32,
    new_row_index: Option<&[i32]>,
) -> Vec<f64> {
    let mut saved = Vec::new();
    if !status.has_dual_steepest_edge_weights || !status.has_basis {
        return saved;
    }
    // The basis may already have been extended by basic logicals of new
    // rows, which come after the num_weighted_row rows with weights
    let num_row = basic_index.len() as i32;
    let num_col = num_tot as i32 - num_row;
    if num_weighted_row <= 0 || num_weighted_row > num_row || num_col < 0 || (dual_edge_weight.len() as i32) < num_weighted_row {
        return saved;
    }
    // -1: no weight (variable was nonbasic); -2: logical of a new row
    saved = vec![-1.0; (num_col + new_num_row) as usize];
    if new_row_index.is_none() {
        for i_row in num_weighted_row..new_num_row {
            saved[(num_col + i_row) as usize] = -2.0;
        }
    }
    for i_row in 0..num_weighted_row as usize {
        let mut i_var = basic_index[i_row];
        if i_var >= num_col {
            let mut row = i_var - num_col;
            if let Some(m) = new_row_index {
                row = m[row as usize];
            }
            if row < 0 || row >= new_num_row {
                continue;
            }
            i_var = num_col + row;
        }
        saved[i_var as usize] = dual_edge_weight[i_row];
    }
    saved
}

impl Default for LpSolver {
    fn default() -> Self {
        let mut s = LpSolver {
            sh: EkkShared {
                info: SharedInfo {
                    dual_objective_value: 0.0,
                    primal_objective_value: 0.0,
                    max_primal_infeasibility: INF,
                    sum_primal_infeasibilities: INF,
                    max_dual_infeasibility: INF,
                    sum_dual_infeasibilities: INF,
                    factor_pivot_threshold: 0.0,
                    row_ep_density: 0.0,
                    col_aq_density: 0.0,
                    num_primal_infeasibilities: -1,
                    num_dual_infeasibilities: -1,
                    dual_edge_weight_strategy: 0,
                },
                model_status: 0,
                iteration_count: 0,
                exit_algorithm: 0,
                debug_solve_call_num: 0,
                debug_initial_build_synthetic_tick: 0,
                dual_ray_index: -1,
                dual_ray_sign: 0,
                primal_ray_index: -1,
                primal_ray_sign: 0,
                status: SimplexStatus::default(),
                dual_values_valid: false,
                nla_lp_set: false,
            },
            info: Info::default(),
            basis: SimplexBasis::default(),
            work_cost: Vec::new(),
            work_dual: Vec::new(),
            work_shift: Vec::new(),
            work_lower: Vec::new(),
            work_upper: Vec::new(),
            work_range: Vec::new(),
            work_value: Vec::new(),
            work_lower_shift: Vec::new(),
            work_upper_shift: Vec::new(),
            base_lower: Vec::new(),
            base_upper: Vec::new(),
            base_value: Vec::new(),
            num_tot_random_value: Vec::new(),
            num_tot_permutation: Vec::new(),
            num_col_permutation: Vec::new(),
            devex_index: Vec::new(),
            bt_basis: SimplexBasis::default(),
            bt_work_shift: Vec::new(),
            bt_edge_weight: Vec::new(),
            random: HighsRandom::new(0).state(),
            dual_edge_weight: Vec::new(),
            scattered_dual_edge_weight: Vec::new(),
            saved_dual_edge_weight: Vec::new(),
            simplex_in_scaled_space: false,
            ar: RowMatrix::default(),
            ar_matrix_is_scaled: false,
            random_vectors_drawn: false,
            dual_values_scaled: false,
            dual_values_cost_hash: 0,
            dual_values_basis_hash: 0,
            fresh_unperturbed_dual: false,
            fresh_dual: false,
            fresh_primal: false,
            scaled_a: RowMatrix::default(),
            factor: HFactor::default(),
            fs: FactorShell::default(),
            nla_build_synthetic_tick: 0.0,
            iterate: Iterate::default(),
            cost_scale: 1.0,
            cost_perturbation_base: 0.0,
            cost_perturbation_max_abs_cost: 0.0,
            dual_simplex_cleanup_level: 0,
            dual_simplex_phase1_cleanup_level: 0,
            previous_iteration_cycling_detected: -K_HIGHS_IINF,
            solve_bailout: false,
            called_return_from_solve: false,
            return_primal_solution_status: 0,
            return_dual_solution_status: 0,
            dual_ray_value: Vec::new(),
            primal_ray_value: Vec::new(),
            edge_weight_error: 0.0,
            build_synthetic_tick: 0.0,
            total_synthetic_tick: 0.0,
            records: BasisRecords::default(),
            lp: Lp::default(),
            nla_rust: false,
            nla_rust_scale: false,
            dz: Dualized::default(),
        };
        s.clear_ekk_data_info();
        s
    }
}

impl LpSolver {
    /// A call's environment with this solver's LP (and, if it is the NLA's
    /// LP, its dimensions and the scale factors that apply)
    pub fn env_of(&mut self, e: &LpsEnv) -> LpsEnv {
        let mut env = *e;
        env.lp = self.lp.view();
        env.model_name = rs(&mut self.lp.model_name);
        if self.nla_rust {
            let lp = &mut self.lp;
            env.nla_num_col = lp.num_col;
            env.nla_num_row = lp.num_row;
            env.nla_has_scale = self.nla_rust_scale;
            let none = RsMut { ptr: std::ptr::null_mut(), len: 0 };
            env.nla_col_scale = if env.nla_has_scale { rs(&mut lp.scale.col) } else { none };
            env.nla_row_scale = if env.nla_has_scale { rs(&mut lp.scale.row) } else { none };
        }
        env
    }

    /// The simplex NLA's LP is this solver's (`rust`) or a C++ one
    /// (HEkk::setNlaPointersForLpAndScale)
    pub fn set_nla_rust(&mut self, rust: bool) {
        self.nla_rust = rust;
        self.nla_rust_scale = rust && self.lp.scale.has_scaling && !self.lp.is_scaled;
        self.sh.nla_lp_set = true;
    }

    // ---- Clearing and invalidation ----

    /// HEkk::clear but its C++ parts (the LP's name and the pointers)
    pub fn clear(&mut self) {
        self.lp.clear();
        self.dz = Dualized::default();
        self.nla_rust = false;
        self.clear_ekk_data();
        self.clear_ekk_dual_edge_weight_data();
        self.basis.clear();
        self.clear_nla();
        self.clear_ekk_all_status();
        self.clear_ray_records();
    }

    /// HSimplexNla::clear (the factor and the saved iterate are kept)
    fn clear_nla(&mut self) {
        self.sh.nla_lp_set = false;
        self.nla_rust = false;
        self.nla_build_synthetic_tick = 0.0;
    }

    /// HEkk::clearEkkAllStatus
    fn clear_ekk_all_status(&mut self) {
        self.sh.status.initialised_for_new_lp = false;
        self.sh.status.initialised_for_solve = false;
        self.clear_nla_status();
        self.clear_ekk_data_status();
    }

    /// HEkk::clearEkkDataStatus
    fn clear_ekk_data_status(&mut self) {
        self.sh.dual_values_valid = false;
        let s = &mut self.sh.status;
        s.has_ar_matrix = false;
        s.has_dual_steepest_edge_weights = false;
        s.has_fresh_rebuild = false;
        s.has_dual_objective_value = false;
        s.has_primal_objective_value = false;
    }

    /// HEkk::clearNlaStatus
    fn clear_nla_status(&mut self) {
        self.sh.status.has_basis = false;
        self.sh.status.has_nla = false;
        self.sh.status.has_invert = false;
        self.sh.status.has_fresh_invert = false;
    }

    /// HEkk::clearRayRecords
    pub fn clear_ray_records(&mut self) {
        self.sh.dual_ray_index = -1;
        self.sh.dual_ray_sign = 0;
        self.dual_ray_value.clear();
        self.sh.primal_ray_index = -1;
        self.sh.primal_ray_sign = 0;
        self.primal_ray_value.clear();
    }

    /// HEkk::clearEkkDualEdgeWeightData
    fn clear_ekk_dual_edge_weight_data(&mut self) {
        self.dual_edge_weight.clear();
        self.scattered_dual_edge_weight.clear();
        self.saved_dual_edge_weight.clear();
    }

    /// HEkk::clearEkkData
    fn clear_ekk_data(&mut self) {
        self.clear_ekk_data_info();
        self.sh.model_status = 0;
        self.simplex_in_scaled_space = false;
        self.ar = RowMatrix::default();
        self.scaled_a = RowMatrix::default();
        self.cost_scale = 1.0;
        self.sh.iteration_count = 0;
        self.dual_simplex_cleanup_level = 0;
        self.dual_simplex_phase1_cleanup_level = 0;
        self.previous_iteration_cycling_detected = -K_HIGHS_IINF;
        self.solve_bailout = false;
        self.called_return_from_solve = false;
        self.sh.exit_algorithm = 0;
        self.return_primal_solution_status = 0;
        self.return_dual_solution_status = 0;
        self.clear_ray_records();
        self.build_synthetic_tick = 0.0;
        self.total_synthetic_tick = 0.0;
        self.sh.debug_solve_call_num = 0;
        self.sh.debug_initial_build_synthetic_tick = 0;
        self.records.clear_bad_basis_change(REASON_ALL);
        self.records.out.primal_phase1_dual = None;
    }

    /// HEkk::clearEkkDataInfo
    fn clear_ekk_data_info(&mut self) {
        for v in [
            &mut self.work_cost,
            &mut self.work_dual,
            &mut self.work_shift,
            &mut self.work_lower,
            &mut self.work_upper,
            &mut self.work_range,
            &mut self.work_value,
            &mut self.work_lower_shift,
            &mut self.work_upper_shift,
            &mut self.base_lower,
            &mut self.base_upper,
            &mut self.base_value,
            &mut self.num_tot_random_value,
            &mut self.bt_work_shift,
            &mut self.bt_edge_weight,
        ] {
            v.clear();
        }
        self.num_tot_permutation.clear();
        self.num_col_permutation.clear();
        self.devex_index.clear();
        let info = &mut self.info;
        info.backtracking = false;
        info.valid_backtracking_basis = false;
        self.bt_basis.clear();
        info.bt_costs_shifted = 0;
        info.bt_costs_perturbed = 0;
        info.bt_bounds_shifted = 0;
        info.bt_bounds_perturbed = 0;
        info.simplex_strategy = 0;
        self.sh.info.dual_edge_weight_strategy = 0;
        info.price_strategy = 0;
        info.dual_simplex_cost_perturbation_multiplier = 1.0;
        info.primal_simplex_phase1_cost_perturbation_multiplier = 1.0;
        info.primal_simplex_bound_perturbation_multiplier = 1.0;
        info.allow_dual_steepest_edge_to_devex_switch = false;
        info.dual_steepest_edge_weight_log_error_threshold = 0.0;
        info.run_quiet = false;
        info.store_squared_primal_infeasibility = false;
        info.allow_cost_shifting = true;
        info.allow_cost_perturbation = true;
        info.allow_bound_perturbation = true;
        info.costs_shifted = false;
        info.costs_perturbed = false;
        info.bounds_shifted = false;
        info.bounds_perturbed = false;
        let si = &mut self.sh.info;
        si.num_primal_infeasibilities = -1;
        si.max_primal_infeasibility = INF;
        si.sum_primal_infeasibilities = INF;
        si.num_dual_infeasibilities = -1;
        si.max_dual_infeasibility = INF;
        si.sum_dual_infeasibilities = INF;
        info.dual_phase1_iteration_count = 0;
        info.dual_phase2_iteration_count = 0;
        info.primal_phase1_iteration_count = 0;
        info.primal_phase2_iteration_count = 0;
        info.primal_bound_swap = 0;
        info.min_concurrency = 1;
        info.num_concurrency = 1;
        info.max_concurrency = SIMPLEX_CONCURRENCY_LIMIT;
        info.update_count = 0;
        si.dual_objective_value = 0.0;
        si.primal_objective_value = 0.0;
        info.updated_dual_objective_value = 0.0;
        info.updated_primal_objective_value = 0.0;
        info.num_basic_logicals = 0;
    }

    /// HEkk::invalidate but simplex_stats_ (C++)
    pub fn invalidate(&mut self) {
        self.sh.status.initialised_for_new_lp = false;
        self.sh.status.initialised_for_solve = false;
        self.invalidate_basis_matrix();
    }

    /// HEkk::invalidateBasisMatrix
    fn invalidate_basis_matrix(&mut self) {
        self.sh.status.has_nla = false;
        self.invalidate_basis();
    }

    /// HEkk::invalidateBasis
    fn invalidate_basis(&mut self) {
        self.sh.status.has_basis = false;
        self.sh.dual_values_valid = false;
        self.invalidate_basis_artifacts();
    }

    /// HEkk::invalidateBasisArtifacts
    fn invalidate_basis_artifacts(&mut self) {
        let s = &mut self.sh.status;
        s.has_ar_matrix = false;
        s.has_dual_steepest_edge_weights = false;
        s.has_invert = false;
        s.has_fresh_invert = false;
        s.has_fresh_rebuild = false;
        s.has_dual_objective_value = false;
        s.has_primal_objective_value = false;
        self.clear_ray_records();
    }

    fn scatter(&self, num_weighted_row: i32, new_num_row: i32, new_row_index: Option<&[i32]>) -> Vec<f64> {
        scatter_dual_edge_weights(
            &self.sh.status,
            &self.basis.basic_index,
            self.basis.nonbasic_flag.len(),
            &self.dual_edge_weight,
            num_weighted_row,
            new_num_row,
            new_row_index,
        )
    }

    /// HEkk::updateStatus: returns whether HEkk::clear() was done (C++ then
    /// clears its parts)
    pub fn update_status(&mut self, action: i32) -> bool {
        // only bound changes leave the dual values of the basis valid
        if action != LP_NEW_BOUNDS {
            self.sh.dual_values_valid = false;
        }
        match action {
            LP_SCALE | LP_SCALED_COL | LP_SCALED_ROW => self.invalidate_basis_matrix(),
            LP_NEW_COSTS | LP_NEW_BOUNDS => {
                let s = &mut self.sh.status;
                s.has_fresh_rebuild = false;
                s.has_dual_objective_value = false;
                s.has_primal_objective_value = false;
            }
            LP_NEW_BASIS => {
                // Keep the weights of the outgoing basis: rows whose basic
                // variable stays basic start from them
                let num_row = self.basis.basic_index.len() as i32;
                let saved = self.scatter(num_row, num_row, None);
                self.invalidate_basis();
                if !saved.is_empty() {
                    self.saved_dual_edge_weight = saved;
                }
            }
            LP_NEW_COLS | LP_NEW_ROWS | LP_DEL_COLS | LP_DEL_NONBASIC_COLS | LP_DEL_ROWS => {
                self.clear();
                return true;
            }
            LP_BACKTRACKING => {
                let s = &mut self.sh.status;
                s.has_ar_matrix = false;
                s.has_fresh_rebuild = false;
                s.has_dual_objective_value = false;
                s.has_primal_objective_value = false;
            }
            _ => {}
        }
        false
    }

    // ---- Moving the LP in and the set-up for a new LP ----

    /// HEkk::moveLp after the C++ move: returns whether Ekk was initialised
    /// (initialiseEkk, which clears the simplex NLA)
    pub fn move_lp(&mut self, env: &LpsEnv) -> bool {
        // Changes to the matrix or basis invalidate the row-wise matrix via
        // updateStatus, and new scaling does in solveLpSimplex, so it only
        // needs rebuilding here if it doesn't fit the LP. If only its
        // scaling differs (the LP is moved in unscaled to check a proof of
        // infeasibility), it is kept: a solve rebuilds it, and the proof
        // converts its values
        let lp = &env.lp;
        // HighsSparseMatrix::numNz of lp_.a_matrix_
        let k = if lp.a.format == MATRIX_FORMAT_COLWISE { lp.a.num_col } else { lp.a.num_row };
        // SAFETY: lp_.a_matrix_.start_
        let num_nz = unsafe { lp.a.start.get() }.get(k as usize).copied().unwrap_or(0);
        if self.ar.num_col != lp.num_col
            || self.ar.num_row != lp.num_row
            || self.ar.num_nz() != num_nz
            || (self.ar_matrix_is_scaled != lp.is_scaled && !lp.scale_has_scaling)
        {
            self.sh.status.has_ar_matrix = false;
        }
        self.simplex_in_scaled_space = lp.is_scaled;
        self.initialise_ekk(env)
    }

    /// HEkk::initialiseEkk: returns whether it did
    fn initialise_ekk(&mut self, env: &LpsEnv) -> bool {
        if self.sh.status.initialised_for_new_lp {
            return false;
        }
        self.set_simplex_options(env);
        self.initialise_control(env);
        self.initialise_simplex_lp_random_vectors(env);
        self.random_vectors_drawn = false;
        self.clear_nla();
        self.records.clear_bad_basis_change(REASON_ALL);
        self.sh.status.initialised_for_new_lp = true;
        true
    }

    /// HEkk::setSimplexOptions
    fn set_simplex_options(&mut self, env: &LpsEnv) {
        let o = &env.opt;
        self.sh.info.dual_edge_weight_strategy = o.simplex_dual_edge_weight_strategy;
        self.info.price_strategy = o.simplex_price_strategy;
        self.info.dual_simplex_cost_perturbation_multiplier = o.dual_simplex_cost_perturbation_multiplier;
        self.info.primal_simplex_bound_perturbation_multiplier = o.primal_simplex_bound_perturbation_multiplier;
        self.sh.info.factor_pivot_threshold = o.factor_pivot_threshold;
        self.info.update_limit = o.simplex_update_limit;
        self.random = HighsRandom::new(o.random_seed as u32).state();
        // Set values of internal options
        self.info.store_squared_primal_infeasibility = true;
    }

    /// HEkk::updateSimplexOptions
    fn update_simplex_options(&mut self, env: &LpsEnv) {
        self.info.dual_simplex_cost_perturbation_multiplier = env.opt.dual_simplex_cost_perturbation_multiplier;
        self.info.primal_simplex_bound_perturbation_multiplier = env.opt.primal_simplex_bound_perturbation_multiplier;
    }

    /// HEkk::initialiseControl
    pub fn initialise_control(&mut self, env: &LpsEnv) {
        let info = &mut self.info;
        info.allow_dual_steepest_edge_to_devex_switch =
            env.opt.simplex_dual_edge_weight_strategy == SIMPLEX_EDGE_WEIGHT_STRATEGY_CHOOSE;
        info.dual_steepest_edge_weight_log_error_threshold = env.opt.dual_steepest_edge_weight_log_error_threshold;
        info.control_iteration_count0 = self.sh.iteration_count;
        self.sh.info.col_aq_density = 0.0;
        self.sh.info.row_ep_density = 0.0;
        info.row_ap_density = 0.0;
        info.row_dse_density = 0.0;
        info.col_steepest_edge_density = 0.0;
        info.col_basic_feasibility_change_density = 0.0;
        info.row_basic_feasibility_change_density = 0.0;
        info.col_bfrt_density = 0.0;
        info.primal_col_density = 0.0;
        // Set the row_dual_density to 1 since it's assumed all costs are
        // at least perturbed from zero, if not initially nonzero
        info.dual_col_density = 1.0;
        info.costly_dse_frequency = 0.0;
        info.num_costly_dse_iteration = 0;
        info.costly_dse_measure = 0.0;
        info.average_log_low_dse_weight_error = 0.0;
        info.average_log_high_dse_weight_error = 0.0;
    }

    /// HEkk::initialiseSimplexLpRandomVectors
    fn initialise_simplex_lp_random_vectors(&mut self, env: &LpsEnv) {
        let num_col = env.lp.num_col;
        let num_tot = (num_col + env.lp.num_row) as usize;
        if num_tot == 0 {
            return;
        }
        let mut random = HighsRandom::from_state(self.random);
        if num_col > 0 {
            // Generate a random permutation of the column indices
            let p = &mut self.num_col_permutation;
            p.resize(num_col as usize, 0);
            for (i, v) in p.iter_mut().enumerate() {
                *v = i as i32;
            }
            random.shuffle(p);
        }
        // Generate a random permutation of all the indices
        let p = &mut self.num_tot_permutation;
        p.resize(num_tot, 0);
        for (i, v) in p.iter_mut().enumerate() {
            *v = i as i32;
        }
        random.shuffle(p);
        // Generate a vector of random reals
        self.num_tot_random_value.resize(num_tot, 0.0);
        for v in self.num_tot_random_value.iter_mut() {
            *v = random.fraction();
        }
        self.random = random.state();
    }

    // ---- The basis ----

    /// HEkk::setBasis(): a logical basis for the LP
    pub fn set_basis_logical(&mut self, env: &LpsEnv) {
        self.sh.dual_values_valid = false;
        let (num_col, num_row) = (env.lp.num_col, env.lp.num_row);
        // SAFETY: lp_'s column bounds
        let (lower, upper) = unsafe { (env.lp.col_lower.get(), env.lp.col_upper.get()) };
        let b = &mut self.basis;
        b.setup(num_col, num_row);
        b.debug_origin_name = "HEkk::setBasis - logical".into();
        for i_col in 0..num_col as usize {
            b.nonbasic_flag[i_col] = 1;
            let (l, u) = (lower[i_col], upper[i_col]);
            let mv = if l == u {
                MOVE_ZE
            } else if -l < INF {
                if u < INF {
                    if l.abs() < u.abs() {
                        MOVE_UP
                    } else {
                        MOVE_DN
                    }
                } else {
                    MOVE_UP
                }
            } else if u < INF {
                MOVE_DN
            } else {
                MOVE_ZE
            };
            b.nonbasic_move[i_col] = mv;
        }
        for i_row in 0..num_row {
            let i_var = num_col + i_row;
            b.nonbasic_flag[i_var as usize] = 0;
            hash::sparse_combine_index(&mut b.hash, i_var);
            b.basic_index[i_row as usize] = i_var;
        }
        self.info.num_basic_logicals = num_row;
        self.sh.status.has_basis = true;
    }

    /// HEkk::setBasis(const HighsBasis&): the statuses, the basis' debug
    /// identifiers and origin
    pub fn set_basis(
        &mut self,
        env: &LpsEnv,
        col_status: &[u8],
        row_status: &[u8],
        debug_id: i32,
        debug_update_count: i32,
        origin: &str,
    ) {
        self.sh.dual_values_valid = false;
        let (num_col, num_row) = (env.lp.num_col, env.lp.num_row);
        // SAFETY: lp_'s bounds
        let (col_lower, col_upper, row_lower, row_upper) = unsafe {
            (env.lp.col_lower.get(), env.lp.col_upper.get(), env.lp.row_lower.get(), env.lp.row_upper.get())
        };
        let b = &mut self.basis;
        b.setup(num_col, num_row);
        b.debug_id = debug_id;
        b.debug_update_count = debug_update_count;
        b.debug_origin_name = origin.into();
        let mut num_basic = 0;
        for i_col in 0..num_col as usize {
            if col_status[i_col] == BS_BASIC {
                b.nonbasic_flag[i_col] = 0;
                b.nonbasic_move[i_col] = 0;
                b.basic_index[num_basic] = i_col as i32;
                num_basic += 1;
                hash::sparse_combine_index(&mut b.hash, i_col as i32);
            } else {
                b.nonbasic_flag[i_col] = 1;
                b.nonbasic_move[i_col] = if col_lower[i_col] == col_upper[i_col] {
                    MOVE_ZE
                } else if col_status[i_col] == BS_LOWER {
                    MOVE_UP
                } else if col_status[i_col] == BS_UPPER {
                    MOVE_DN
                } else {
                    MOVE_ZE
                };
            }
        }
        for i_row in 0..num_row as usize {
            let i_var = num_col as usize + i_row;
            if row_status[i_row] == BS_BASIC {
                b.nonbasic_flag[i_var] = 0;
                b.nonbasic_move[i_var] = 0;
                b.basic_index[num_basic] = i_var as i32;
                num_basic += 1;
                hash::sparse_combine_index(&mut b.hash, i_var as i32);
            } else {
                b.nonbasic_flag[i_var] = 1;
                b.nonbasic_move[i_var] = if row_lower[i_row] == row_upper[i_row] {
                    MOVE_ZE
                } else if row_status[i_row] == BS_LOWER {
                    MOVE_DN
                } else if row_status[i_row] == BS_UPPER {
                    MOVE_UP
                } else {
                    MOVE_ZE
                };
            }
        }
        self.sh.status.has_basis = true;
    }

    /// HEkk::getHighsBasis(use_lp): the statuses (the caller sized them)
    /// and the debug identifiers; the origin is basis.debug_origin_name.
    /// `sense` is that of HEkk's LP
    pub fn get_highs_basis(&self, use_lp: &CLp, sense: i32, col_status: &mut [u8], row_status: &mut [u8]) -> (i32, i32) {
        let (num_col, num_row) = (use_lp.num_col as usize, use_lp.num_row as usize);
        // SAFETY: use_lp's bounds
        let (col_lower, col_upper, row_lower, row_upper) = unsafe {
            (use_lp.col_lower.get(), use_lp.col_upper.get(), use_lp.row_lower.get(), use_lp.row_upper.get())
        };
        let b = &self.basis;
        let status = |i_var: usize, lower: f64, upper: f64, up: u8, dn: u8| -> u8 {
            if b.nonbasic_flag[i_var] == 0 {
                BS_BASIC
            } else if b.nonbasic_move[i_var] == MOVE_UP {
                up
            } else if b.nonbasic_move[i_var] == MOVE_DN {
                dn
            } else if b.nonbasic_move[i_var] == MOVE_ZE {
                if lower == upper {
                    let dual = sense as f64 * self.work_dual[i_var];
                    if dual >= 0.0 {
                        BS_LOWER
                    } else {
                        BS_UPPER
                    }
                } else {
                    BS_ZERO
                }
            } else {
                BS_NONBASIC
            }
        };
        for i_col in 0..num_col {
            col_status[i_col] = status(i_col, col_lower[i_col], col_upper[i_col], BS_LOWER, BS_UPPER);
        }
        for i_row in 0..num_row {
            row_status[i_row] = status(num_col + i_row, row_lower[i_row], row_upper[i_row], BS_UPPER, BS_LOWER);
        }
        ((self.build_synthetic_tick + self.total_synthetic_tick) as i32, self.info.update_count)
    }

    /// HEkk::getSolution: the solution vectors (sized by the caller)
    pub fn get_solution(
        &mut self,
        env: &LpsEnv,
        col_value: &mut [f64],
        col_dual: &mut [f64],
        row_value: &mut [f64],
        row_dual: &mut [f64],
    ) {
        let (num_col, num_row) = (env.lp.num_col as usize, env.lp.num_row as usize);
        // Scatter the basic primal values
        for i_row in 0..num_row {
            self.work_value[self.basis.basic_index[i_row] as usize] = self.base_value[i_row];
        }
        // Zero the basic dual values
        for i_row in 0..num_row {
            self.work_dual[self.basis.basic_index[i_row] as usize] = 0.0;
        }
        let sense = env.lp.sense as f64;
        for i_col in 0..num_col {
            col_value[i_col] = self.work_value[i_col];
            col_dual[i_col] = sense * self.work_dual[i_col];
        }
        for i_row in 0..num_row {
            row_value[i_row] = -self.work_value[num_col + i_row];
            // @FlipRowDual negate RHS
            row_dual[i_row] = -sense * self.work_dual[num_col + i_row];
        }
    }

    /// HEkk::unscaleSimplex(incumbent_lp)
    pub fn unscale_simplex(&mut self, lp: &CLp) {
        if !self.simplex_in_scaled_space {
            return;
        }
        let (num_col, num_row) = (lp.num_col as usize, lp.num_row as usize);
        // SAFETY: the LP's scale factors
        let (col_scale, row_scale) = unsafe { (lp.scale_col.get(), lp.scale_row.get()) };
        for i_col in 0..num_col {
            let factor = col_scale[i_col];
            self.work_cost[i_col] /= factor;
            self.work_dual[i_col] /= factor;
            self.work_shift[i_col] /= factor;
            self.work_lower[i_col] *= factor;
            self.work_upper[i_col] *= factor;
            self.work_range[i_col] *= factor;
            self.work_value[i_col] *= factor;
            self.work_lower_shift[i_col] *= factor;
            self.work_upper_shift[i_col] *= factor;
        }
        for i_row in 0..num_row {
            let i_var = num_col + i_row;
            let factor = row_scale[i_row];
            self.work_cost[i_var] *= factor;
            self.work_dual[i_var] *= factor;
            self.work_shift[i_var] *= factor;
            self.work_lower[i_var] /= factor;
            self.work_upper[i_var] /= factor;
            self.work_range[i_var] /= factor;
            self.work_value[i_var] /= factor;
            self.work_lower_shift[i_var] /= factor;
            self.work_upper_shift[i_var] /= factor;
        }
        for i_row in 0..num_row {
            let i_var = self.basis.basic_index[i_row] as usize;
            let factor = if i_var < num_col { col_scale[i_var] } else { 1.0 / row_scale[i_var - num_col] };
            self.base_lower[i_row] *= factor;
            self.base_upper[i_row] *= factor;
            self.base_value[i_row] *= factor;
        }
        self.simplex_in_scaled_space = false;
    }

    /// getUnscaledInfeasibilities (simplex/HSimplex.cpp): the counts, maxima
    /// and sums of the unscaled primal and dual infeasibilities, in
    /// HighsInfo's order (C++ then sets the solution status)
    pub fn get_unscaled_infeasibilities(&self, env: &LpsEnv, lp: &CLp, out: &mut UnscaledInfeasibilities) {
        let primal_feasibility_tolerance = env.opt.primal_feasibility_tolerance;
        let dual_feasibility_tolerance = env.opt.dual_feasibility_tolerance;
        *out = UnscaledInfeasibilities::default();
        let (num_col, num_row) = (lp.scale_num_col as usize, lp.scale_num_row as usize);
        // SAFETY: the scale factors
        let (col_scale, row_scale) = unsafe { (lp.scale_col.get(), lp.scale_row.get()) };
        let cost = lp.scale_cost;
        let max = |a: f64, b: f64| if a < b { b } else { a };
        for i_var in 0..num_col + num_row {
            // Look at the dual infeasibilities of nonbasic variables
            if self.basis.nonbasic_flag[i_var] == 0 {
                continue;
            }
            // No dual infeasibility for fixed rows and columns
            if self.work_lower[i_var] == self.work_upper[i_var] {
                continue;
            }
            let scale_mu = if i_var < num_col {
                1.0 / (col_scale[i_var] / cost)
            } else {
                row_scale[i_var - num_col] * cost
            };
            let dual = self.work_dual[i_var];
            let lower = self.work_lower[i_var];
            let upper = self.work_upper[i_var];
            let unscaled_dual = dual * scale_mu;
            let dual_infeasibility = if -lower >= INF && upper >= INF {
                // Free: any nonzero dual value is infeasible
                unscaled_dual.abs()
            } else {
                // Not fixed: any dual infeasibility is given by value
                // signed by nonbasicMove
                -(self.basis.nonbasic_move[i_var] as f64) * unscaled_dual
            };
            if dual_infeasibility > 0.0 {
                if dual_infeasibility >= dual_feasibility_tolerance {
                    out.num_dual_infeasibilities += 1;
                }
                out.max_dual_infeasibility = max(dual_infeasibility, out.max_dual_infeasibility);
                out.sum_dual_infeasibilities += dual_infeasibility;
            }
        }
        // Look at the primal infeasibilities of basic variables
        for ix in 0..num_row {
            let i_var = self.basis.basic_index[ix] as usize;
            let scale_mu = if i_var < num_col { col_scale[i_var] } else { 1.0 / row_scale[i_var - num_col] };
            let unscaled_lower = self.base_lower[ix] * scale_mu;
            let unscaled_value = self.base_value[ix] * scale_mu;
            let unscaled_upper = self.base_upper[ix] * scale_mu;
            let mut primal_infeasibility = 0.0;
            if unscaled_value < unscaled_lower - primal_feasibility_tolerance {
                primal_infeasibility = unscaled_lower - unscaled_value;
            } else if unscaled_value > unscaled_upper + primal_feasibility_tolerance {
                primal_infeasibility = unscaled_value - unscaled_upper;
            }
            if primal_infeasibility > 0.0 {
                out.num_primal_infeasibilities += 1;
                out.max_primal_infeasibility = max(primal_infeasibility, out.max_primal_infeasibility);
                out.sum_primal_infeasibilities += primal_infeasibility;
            }
        }
    }

    // ---- LP edits ----

    /// HEkk::addRows: the LP has new_num_row rows, of which num_weighted_row
    /// had weights (C++ sets lp_.num_row_)
    pub fn add_rows(&mut self, num_weighted_row: i32, new_num_row: i32) {
        // New rows come in with basic logicals, which leaves the DSE
        // weights of the existing rows unchanged
        let saved = self.scatter(num_weighted_row, new_num_row, None);
        self.update_status(LP_NEW_ROWS);
        self.saved_dual_edge_weight = saved;
    }

    /// HEkk::deleteRows
    pub fn delete_rows(&mut self, ic: &crate::lp_data::lp_utils::IndexCollection) {
        // Deleting rows with basic logicals leaves the DSE weights of the
        // remaining rows unchanged, so keep them under the new row indices
        let mut saved = Vec::new();
        let num_row = self.basis.basic_index.len() as i32;
        if self.sh.status.has_dual_steepest_edge_weights && num_row > 0 {
            let mut new_row_index = vec![0i32; num_row as usize];
            if ic.is_interval {
                let mut i_row = ic.from;
                while i_row <= ic.to && i_row < num_row {
                    new_row_index[i_row as usize] = -1;
                    i_row += 1;
                }
            } else if ic.is_set {
                for &i_row in ic.set {
                    if i_row < num_row {
                        new_row_index[i_row as usize] = -1;
                    }
                }
            } else if ic.is_mask {
                for i_row in 0..num_row as usize {
                    if ic.mask[i_row] != 0 {
                        new_row_index[i_row] = -1;
                    }
                }
            }
            let mut new_num_row = 0;
            for v in new_row_index.iter_mut() {
                if *v >= 0 {
                    *v = new_num_row;
                    new_num_row += 1;
                }
            }
            saved = self.scatter(num_row, new_num_row, Some(&new_row_index));
        }
        self.update_status(LP_DEL_ROWS);
        self.saved_dual_edge_weight = saved;
    }

    /// The simplex basis for new nonbasic columns (Highs::
    /// appendNonbasicColsToBasisInterface sizes it, then sets the statuses)
    pub fn resize_basis(&mut self, num_tot: usize) {
        self.basis.nonbasic_flag.resize(num_tot, 0);
        self.basis.nonbasic_move.resize(num_tot, 0);
    }

    /// The simplex basis part of Highs::appendBasicRowsToBasisInterface
    pub fn append_basic_rows(&mut self, num_col: i32, num_row: i32, new_num_row: i32) {
        let new_num_tot = (num_col + new_num_row) as usize;
        let b = &mut self.basis;
        b.nonbasic_flag.resize(new_num_tot, 0);
        b.nonbasic_move.resize(new_num_tot, 0);
        b.basic_index.resize(new_num_row as usize, 0);
        for i_row in num_row..new_num_row {
            let i_var = num_col + i_row;
            b.nonbasic_flag[i_var as usize] = 0;
            b.nonbasic_move[i_var as usize] = 0;
            b.basic_index[i_row as usize] = i_var;
        }
    }

    /// The flip of a nonbasic move for a negative scale factor
    /// (Highs::scaleColInterface, scaleRowInterface)
    pub fn flip_nonbasic_move(&mut self, var: usize) {
        let m = &mut self.basis.nonbasic_move[var];
        if *m == MOVE_UP {
            *m = MOVE_DN;
        } else if *m == MOVE_DN {
            *m = MOVE_UP;
        }
    }

    /// The basis part of HEkk::undualize: the primal basis from the basis
    /// of the dual LP, given the original bounds
    pub fn undualize_basis(
        &mut self,
        dual_num_col: i32,
        original_col_lower: &[f64],
        original_col_upper: &[f64],
        original_row_lower: &[f64],
        original_row_upper: &[f64],
    ) -> i32 {
        let original_num_col = original_col_lower.len() as i32;
        let original_num_row = original_row_lower.len() as i32;
        let primal_num_tot = (original_num_col + original_num_row) as usize;
        let dual_nonbasic_flag = self.basis.nonbasic_flag.clone();
        let b = &mut self.basis;
        b.nonbasic_flag = vec![ILLEGAL_FLAG; primal_num_tot];
        b.nonbasic_move = vec![ILLEGAL_MOVE; primal_num_tot];
        b.basic_index.clear();
        let mut upper_bound_col = original_num_row;
        let dual_basic = |v: i32| dual_nonbasic_flag[v as usize] == 0;
        for i_col in 0..original_num_col {
            let lower = original_col_lower[i_col as usize];
            let upper = original_col_upper[i_col as usize];
            let mut mv = ILLEGAL_MOVE;
            let mut is_basic = dual_basic(dual_num_col + i_col);
            if lower == upper {
                if is_basic {
                    mv = MOVE_ZE;
                }
            } else if -lower < INF {
                if upper < INF {
                    if is_basic {
                        mv = MOVE_UP;
                    } else {
                        is_basic = dual_basic(upper_bound_col);
                        if is_basic {
                            mv = MOVE_DN;
                        }
                    }
                    upper_bound_col += 1;
                } else if is_basic {
                    mv = MOVE_UP;
                }
            } else if upper < INF {
                if is_basic {
                    mv = MOVE_DN;
                }
            } else if is_basic {
                mv = MOVE_ZE;
            }
            if is_basic {
                b.nonbasic_flag[i_col as usize] = 1;
                b.nonbasic_move[i_col as usize] = mv;
            } else {
                b.basic_index.push(i_col);
                b.nonbasic_flag[i_col as usize] = 0;
                b.nonbasic_move[i_col as usize] = 0;
            }
        }
        for i_row in 0..original_num_row {
            let lower = original_row_lower[i_row as usize];
            let upper = original_row_upper[i_row as usize];
            let mut mv = ILLEGAL_MOVE;
            let mut is_basic = dual_basic(i_row);
            if lower == upper {
                if is_basic {
                    mv = MOVE_ZE;
                }
            } else if -lower < INF {
                if upper < INF {
                    if is_basic {
                        mv = MOVE_DN;
                    } else {
                        is_basic = dual_basic(upper_bound_col);
                        if is_basic {
                            mv = MOVE_UP;
                        }
                    }
                    upper_bound_col += 1;
                } else if is_basic {
                    mv = MOVE_DN;
                }
            } else if upper < INF {
                if is_basic {
                    mv = MOVE_UP;
                }
            } else if is_basic {
                mv = MOVE_ZE;
            }
            let i_var = (original_num_col + i_row) as usize;
            if is_basic {
                b.nonbasic_flag[i_var] = 1;
                b.nonbasic_move[i_var] = mv;
            } else {
                b.basic_index.push(i_var as i32);
                b.nonbasic_flag[i_var] = 0;
                b.nonbasic_move[i_var] = 0;
            }
        }
        b.basic_index.len() as i32
    }

    // ---- The simplex NLA and INVERT ----

    /// HEkk::getScaledAMatrixPointer: whether the factor's matrix is the
    /// scaled copy, made here if so
    fn get_scaled_a_matrix(&mut self, lp: &CLp) -> bool {
        if lp.scale_has_scaling && !lp.is_scaled {
            // SAFETY: lp_'s matrix and scale factors
            let (start, index, value, col_scale, row_scale) =
                unsafe { (lp.a.start.get(), lp.a.index.get(), lp.a.value.get(), lp.scale_col.get(), lp.scale_row.get()) };
            let a = &mut self.scaled_a;
            a.rowwise = lp.a.format != MATRIX_FORMAT_COLWISE;
            a.num_col = lp.a.num_col;
            a.num_row = lp.a.num_row;
            a.start.clear();
            a.start.extend_from_slice(start);
            a.p_end.clear();
            // SAFETY: as above
            a.p_end.extend_from_slice(unsafe { lp.a.p_end.get() });
            a.index.clear();
            a.index.extend_from_slice(index);
            a.value.clear();
            a.value.extend_from_slice(value);
            // HighsSparseMatrix::applyScale (column-wise)
            for i_col in 0..lp.a.num_col as usize {
                for i_el in start[i_col] as usize..start[i_col + 1] as usize {
                    let i_row = index[i_el] as usize;
                    a.value[i_el] *= col_scale[i_col] * row_scale[i_row];
                }
            }
            true
        } else {
            false
        }
    }

    /// The set-up of the simplex NLA for HEkk's LP before INVERT, in
    /// initialiseSimplexLpBasisAndFactor and the solve: HSimplexNla::
    /// setPointers or HSimplexNla::setup (C++ has set the NLA's LP)
    fn set_up_nla(&mut self, env: &LpsEnv) {
        self.fs.uses_scaled_copy = self.get_scaled_a_matrix(&env.lp);
        if self.sh.status.has_nla {
            debug_assert!(self.fs.num_row == env.lp.num_row);
            return;
        }
        // HSimplexNla::setup: HFactor::setupGeneral
        let lp = &env.lp;
        let fs = &mut self.fs;
        fs.set_up = true;
        fs.num_col = lp.num_col;
        fs.num_row = lp.num_row;
        fs.pivot_threshold = clamp(self.sh.info.factor_pivot_threshold, K_MIN_PIVOT_THRESHOLD, K_MAX_PIVOT_THRESHOLD);
        fs.pivot_tolerance = clamp(env.opt.factor_pivot_tolerance, K_MIN_PIVOT_TOLERANCE, K_MAX_PIVOT_TOLERANCE);
        fs.time_limit = INF;
        fs.dev_level = env.opt.dev_level;
        let a_start: Vec<i32> = if fs.uses_scaled_copy {
            self.scaled_a.start.clone()
        } else {
            // SAFETY: lp_'s matrix
            unsafe { lp.a.start.get().to_vec() }
        };
        self.factor.setup(lp.num_col, lp.num_row, lp.num_row, &a_start, UPDATE_METHOD_FT);
        self.sh.status.has_nla = true;
    }

    /// HEkk::lpFactorRowCompatible(expected_num_row)
    pub fn lp_factor_row_compatible(&self, env: &LpsEnv, expected_num_row: i32) -> bool {
        let consistent_num_row = self.fs.num_row == expected_num_row;
        if !consistent_num_row {
            env.dev(LOG_ERROR, || {
                sprintf!(
                    "HEkk::initialiseSimplexLpBasisAndFactor: LP(%6d, %6d) has factor_num_row = %d\n",
                    env.lp.num_col,
                    expected_num_row,
                    self.fs.num_row
                )
            });
        }
        consistent_num_row
    }

    /// HEkk::initialiseSimplexLpBasisAndFactor: the HighsStatus
    pub fn initialise_simplex_lp_basis_and_factor(&mut self, env: &LpsEnv, only_from_known_basis: bool) -> i32 {
        if !self.sh.status.has_basis {
            self.set_basis_logical(env);
        }
        self.set_up_nla(env);
        if self.sh.status.has_invert {
            return hekk::STATUS_OK;
        }
        let mut x = self.c_hekk(env, false);
        // SAFETY: the vectors of the view are those of this solver, sized
        let mut e = unsafe { x.view() };
        let rank_deficiency = hekk::compute_factor(&mut e, &x);
        if rank_deficiency != 0 {
            if only_from_known_basis {
                x.dev(LOG_INFO, || {
                    sprintf!(
                        "HEkk::initialiseSimplexLpBasisAndFactor (%s) Rank_deficiency %d: Id = %d; UpdateCount = %d\n",
                        &self.basis.debug_origin_name,
                        rank_deficiency,
                        self.basis.debug_id,
                        self.basis.debug_update_count
                    )
                });
                x.dev(LOG_ERROR, || "Supposed to be a full-rank basis, but incorrect\n".into());
                return hekk::STATUS_ERROR;
            }
            hekk::initial_rank_deficiency(&mut e, &x);
        }
        hekk::reset_synthetic_clock(&mut e, &x);
        drop(e);
        self.after_c_hekk(&mut x);
        hekk::STATUS_OK
    }

    /// Take what a CHekk call left: the ray values to clear and whether
    /// the saved weights were taken
    fn after_c_hekk(&mut self, x: &mut CHekk) {
        let clear = x.ray_value_clear.get();
        if clear & 1 != 0 {
            self.dual_ray_value.clear();
        }
        if clear & 2 != 0 {
            self.primal_ray_value.clear();
        }
        if x.saved_dual_edge_weight_taken.get() {
            self.saved_dual_edge_weight.clear();
        }
    }

    /// The NLA's view of the basis matrix scaling
    fn nla_scale(&self, env: &LpsEnv) -> (bool, CSlice<f64>, CSlice<f64>) {
        (env.nla_has_scale, cs(&env.nla_col_scale), cs(&env.nla_row_scale))
    }

    /// HSimplexNla::btran or ftran (with the NLA's scaling) of a C++ HVector
    pub fn nla_solve(&mut self, env: &LpsEnv, rhs: &mut CHVec, expected_density: f64, transposed: bool) {
        let c = self.c_ekk_nla(env);
        // SAFETY: this solver's vectors and the C++ HVector's arrays
        unsafe {
            let e = c.view();
            let mut v = rhs.view();
            if transposed {
                e.btran(&mut v, expected_density);
            } else {
                e.ftran(&mut v, expected_density);
            }
            rhs.store(&v);
        }
    }

    /// HEkk::putIterate
    pub fn put_iterate(&mut self) {
        // HSimplexNla::putInvert
        self.iterate.valid = true;
        // SAFETY: this solver's factor
        unsafe { crate::ffi::highs_rs_factor_put_invert(&mut self.factor) };
        self.iterate.basis = self.basis.clone();
        if self.sh.status.has_dual_steepest_edge_weights {
            self.iterate.dual_edge_weight = self.dual_edge_weight.clone();
        } else {
            self.iterate.dual_edge_weight.clear();
        }
    }

    /// HEkk::getIterate: whether there was one
    pub fn get_iterate(&mut self) -> bool {
        self.sh.dual_values_valid = false;
        if !self.iterate.valid {
            return false;
        }
        // SAFETY: this solver's factor
        unsafe { crate::ffi::highs_rs_factor_get_invert(&mut self.factor) };
        self.basis = self.iterate.basis.clone();
        // The row-wise matrix is partitioned by the outgoing basis
        self.sh.status.has_ar_matrix = false;
        if !self.iterate.dual_edge_weight.is_empty() {
            self.dual_edge_weight = self.iterate.dual_edge_weight.clone();
        } else {
            self.sh.status.has_dual_steepest_edge_weights = false;
        }
        self.sh.status.has_invert = true;
        true
    }

    /// HEkk::computeBasisCondition(lp, exact, report): the condition
    /// estimate of the basis matrix, `lp` being the LP whose matrix is
    /// used (the NLA's scaling applies)
    pub fn compute_basis_condition(&mut self, env: &LpsEnv, lp: &CLp, model_name: &str, exact: bool, report: bool) -> f64 {
        let solver_num_row = lp.num_row as usize;
        let solver_num_col = lp.num_col as usize;
        // SAFETY: the LP's matrix
        let (a_start, a_value) = unsafe { (lp.a.start.get(), lp.a.value.get()) };
        let c = self.c_ekk_nla(env);
        // SAFETY: this solver's vectors
        let e = unsafe { c.view() };
        let mut row_ep = OwnedHVec::new(solver_num_row as i32);
        let mut exact_norm_binv: f64 = 0.0;
        let max = |a: f64, b: f64| if a < b { b } else { a };
        if exact {
            // Compute the exact norm of B^{-1}
            for r_n in 0..solver_num_row {
                row_ep.clear();
                row_ep.index[row_ep.count as usize] = r_n as i32;
                row_ep.array[r_n] = 1.0;
                row_ep.count += 1;
                row_ep.pack_flag = false;
                row_ep.with(|v| e.ftran(v, 0.1));
                let mut c_norm = 0.0;
                for i_x in 0..row_ep.count as usize {
                    c_norm += row_ep.array[row_ep.index[i_x] as usize].abs();
                }
                exact_norm_binv = max(c_norm, exact_norm_binv);
            }
        }
        // Compute the Hager condition number estimate for the basis matrix
        let expected_density = 1.0;
        let mut bs_cond_x = vec![0.0; solver_num_row];
        let mut bs_cond_y = vec![0.0; solver_num_row];
        let mut bs_cond_z = vec![0.0; solver_num_row];
        let mut bs_cond_w = vec![0.0; solver_num_row];
        let mu = 1.0 / solver_num_row as f64;
        let mut norm_binv = 0.0;
        bs_cond_x.fill(mu);
        row_ep.clear();
        for r_n in 0..solver_num_row {
            let value = bs_cond_x[r_n];
            if value != 0.0 {
                row_ep.index[row_ep.count as usize] = r_n as i32;
                row_ep.array[r_n] = value;
                row_ep.count += 1;
            }
        }
        for _ps_n in 1..=5 {
            row_ep.pack_flag = false;
            row_ep.with(|v| e.ftran(v, expected_density));
            // zeta = sign(y);
            for r_n in 0..solver_num_row {
                bs_cond_y[r_n] = row_ep.array[r_n];
                bs_cond_w[r_n] = if bs_cond_y[r_n] > 0.0 {
                    1.0
                } else if bs_cond_y[r_n] < 0.0 {
                    -1.0
                } else {
                    0.0
                };
            }
            // z=A'\zeta;
            row_ep.clear();
            for r_n in 0..solver_num_row {
                let value = bs_cond_w[r_n];
                if value != 0.0 {
                    row_ep.index[row_ep.count as usize] = r_n as i32;
                    row_ep.array[r_n] = value;
                    row_ep.count += 1;
                }
            }
            row_ep.pack_flag = false;
            row_ep.with(|v| e.btran(v, expected_density));
            let mut norm_z = 0.0;
            let mut ztx = 0.0;
            norm_binv = 0.0;
            let mut argmax_z: i32 = -1;
            for r_n in 0..solver_num_row {
                bs_cond_z[r_n] = row_ep.array[r_n];
                let abs_z_v = bs_cond_z[r_n].abs();
                if abs_z_v > norm_z {
                    norm_z = abs_z_v;
                    argmax_z = r_n as i32;
                }
                ztx += bs_cond_z[r_n] * bs_cond_x[r_n];
                norm_binv += bs_cond_y[r_n].abs();
            }
            if norm_z <= ztx {
                break;
            }
            // x = zeros(n,1); x(fd_i) = 1;
            bs_cond_x.fill(0.0);
            row_ep.clear();
            row_ep.count = 1;
            row_ep.index[0] = argmax_z;
            row_ep.array[argmax_z as usize] = 1.0;
            bs_cond_x[argmax_z as usize] = 1.0;
        }
        let mut norm_b: f64 = 0.0;
        for r_n in 0..solver_num_row {
            let vr_n = e.basic_index[r_n] as usize;
            let mut c_norm = 0.0;
            if vr_n < solver_num_col {
                for &v in &a_value[a_start[vr_n] as usize..a_start[vr_n + 1] as usize] {
                    c_norm += v.abs();
                }
            } else {
                c_norm += 1.0;
            }
            norm_b = max(c_norm, norm_b);
        }
        let cond_b = norm_binv * norm_b;
        let exact_cond_b = exact_norm_binv * norm_b;
        if exact {
            if report {
                env.user(
                    LOG_INFO,
                    &sprintf!(
                        "HEkk::computeBasisCondition: grep_kappa model,||B||_1,approx ||B^{-1}||_1,approx_kappa,||B^{-1}||_1,kappa = ,%s,%g,%g,%g,%g,%g\n",
                        model_name,
                        norm_b,
                        norm_binv,
                        cond_b,
                        exact_norm_binv,
                        exact_cond_b
                    ),
                );
            }
            return exact_cond_b;
        }
        cond_b
    }

    /// HEkk::proofOfPrimalInfeasibility(): with the basis inverse row of
    /// the dual ray
    pub fn proof_of_primal_infeasibility(&mut self, env: &LpsEnv) -> bool {
        let move_out = self.sh.dual_ray_sign;
        let row_out = self.sh.dual_ray_index as usize;
        let mut x = self.c_hekk(env, false);
        // SAFETY: the vectors of the view are those of this solver
        let mut e = unsafe { x.view() };
        let mut row_ep = OwnedHVec::new(env.lp.num_row);
        let proof = row_ep.with(|v| {
            e.unit_btran(row_out, v);
            hekk::proof_of_primal_infeasibility(&mut e, &x, v, move_out)
        });
        drop(e);
        self.after_c_hekk(&mut x);
        proof
    }

    // ---- The solve ----

    /// HEkk::solve after the C++ analysis set-up, up to returnFromEkkSolve
    /// (initialiseControl, then solveRust's set-up, the Rust solve and the
    /// clean-up of what it left)
    pub fn solve(&mut self, env: &LpsEnv, force_phase2: bool) -> LpsSolveOut {
        self.initialise_control(env);
        // The part of initialiseSimplexLpBasisAndFactor before INVERT: the
        // basis and the simplex NLA
        if !self.sh.status.has_basis {
            self.set_basis_logical(env);
        }
        self.set_up_nla(env);
        self.update_simplex_options(env);
        // Size the vectors that the solve may otherwise resize, so that
        // the kernels can work on views of them
        let num_col = env.lp.num_col as usize;
        let num_row = env.lp.num_row as usize;
        let num_tot = num_col + num_row;
        // The random vectors only depend on the LP dimensions: if asked to
        // (as for the LP relaxation of a MIP), or for large LPs, keep them
        // over re-solves
        let draw_random_vectors = (!env.opt.simplex_keep_random_vectors
            && num_row as i32 <= env.opt.simplex_dse_exact_init_max_rows)
            || !self.random_vectors_drawn
            || self.num_tot_random_value.len() != num_tot
            || self.num_col_permutation.len() != num_col;
        if draw_random_vectors && num_tot > 0 {
            if num_col > 0 {
                self.num_col_permutation.resize(num_col, 0);
            }
            self.num_tot_permutation.resize(num_tot, 0);
            self.num_tot_random_value.resize(num_tot, 0.0);
        }
        // SAFETY: lp_'s matrix
        let num_nz = if num_col > 0 { unsafe { env.lp.a.start.get()[num_col] as usize } } else { 0 };
        if !(self.sh.status.has_ar_matrix && self.ar_matrix_is_scaled == env.lp.is_scaled) {
            self.ar.rowwise = true;
            self.ar.num_col = env.lp.num_col;
            self.ar.num_row = env.lp.num_row;
        }
        self.ar.start.resize(num_row + 1, 0);
        self.ar.p_end.resize(num_row, 0);
        self.ar.index.resize(num_nz, 0);
        self.ar.value.resize(num_nz, 0.0);
        self.allocate_work_and_base_arrays(num_tot, num_row);
        self.basis.nonbasic_move.resize(num_tot, 0);
        if !self.sh.status.has_dual_steepest_edge_weights {
            self.dual_edge_weight.resize(num_row, 0.0);
            self.scattered_dual_edge_weight.resize(num_tot, 0.0);
        }
        self.bt_edge_weight.resize(num_tot, 0.0);
        self.bt_basis.basic_index.resize(num_row, 0);
        self.bt_basis.nonbasic_flag.resize(num_tot, 0);
        self.bt_basis.nonbasic_move.resize(num_tot, 0);
        self.bt_basis.debug_origin_name = self.basis.debug_origin_name.clone();
        self.bt_work_shift.resize(num_tot, 0.0);
        let mut x = self.c_hekk(env, draw_random_vectors);
        let status = hekk::solve(&x, force_phase2);
        self.after_c_hekk(&mut x);
        LpsSolveOut { status, invert_num_el: self.factor.invert_num_el, basis_matrix_num_el: self.factor.basis_matrix_num_el }
    }

    /// The part of HEkk::returnFromEkkSolve on this data
    pub fn return_from_ekk_solve(&mut self) {
        // Saved weights not used by this solve are stale for the next one
        self.saved_dual_edge_weight.clear();
        self.fresh_unperturbed_dual = false;
        self.fresh_dual = false;
        self.fresh_primal = false;
    }

    /// HEkk::allocateWorkAndBaseArrays
    fn allocate_work_and_base_arrays(&mut self, num_tot: usize, num_row: usize) {
        for v in [
            &mut self.work_cost,
            &mut self.work_dual,
            &mut self.work_shift,
            &mut self.work_lower,
            &mut self.work_upper,
            &mut self.work_range,
            &mut self.work_value,
            &mut self.work_lower_shift,
            &mut self.work_upper_shift,
        ] {
            v.resize(num_tot, 0.0);
        }
        self.devex_index.resize(num_tot, 0);
        self.base_lower.resize(num_row, 0.0);
        self.base_upper.resize(num_row, 0.0);
        self.base_value.resize(num_row, 0.0);
    }

    // ---- The views for the kernels ----

    /// The CEkk view for the simplex NLA's solves, with the dimensions of
    /// the NLA's LP (HSimplexNla::lp_)
    fn c_ekk_nla(&mut self, env: &LpsEnv) -> CEkk {
        let mut c = self.c_ekk(env, true);
        c.num_col = env.nla_num_col;
        c.num_row = env.nla_num_row;
        c
    }

    /// The CEkk view of HEkk's data (HEkk::rustView), `nla` choosing the
    /// NLA's scaling (else none)
    pub fn c_ekk(&mut self, env: &LpsEnv, nla: bool) -> CEkk {
        let lp = &env.lp;
        let (has_scale, col_scale, row_scale) = if nla {
            self.nla_scale(env)
        } else {
            (false, CSlice { p: std::ptr::null_mut(), n: 0 }, CSlice { p: std::ptr::null_mut(), n: 0 })
        };
        let (fa_start, fa_index, fa_value) = if !self.fs.set_up {
            (CSlice { p: std::ptr::null_mut(), n: 0 }, CSlice { p: std::ptr::null_mut(), n: 0 }, CSlice { p: std::ptr::null_mut(), n: 0 })
        } else if self.fs.uses_scaled_copy {
            let a = &mut self.scaled_a;
            let nnz = a.start.get(self.fs.num_col as usize).copied().unwrap_or(0) as usize;
            (
                CSlice { p: a.start.as_mut_ptr(), n: (self.fs.num_col + 1).min(a.start.len() as i32) },
                CSlice { p: a.index.as_mut_ptr(), n: nnz.min(a.index.len()) as i32 },
                CSlice { p: a.value.as_mut_ptr(), n: nnz.min(a.value.len()) as i32 },
            )
        } else {
            let start_n = ((self.fs.num_col + 1) as usize).min(lp.a.start.len);
            // SAFETY: lp_'s matrix
            let nnz = if start_n > self.fs.num_col as usize { unsafe { lp.a.start.get()[self.fs.num_col as usize] as usize } } else { 0 };
            (
                CSlice { p: lp.a.start.ptr, n: start_n as i32 },
                CSlice { p: lp.a.index.ptr, n: nnz.min(lp.a.index.len) as i32 },
                CSlice { p: lp.a.value.ptr, n: nnz.min(lp.a.value.len) as i32 },
            )
        };
        let o = &env.opt;
        CEkk {
            num_col: lp.num_col,
            num_row: lp.num_row,
            a_start: cs(&lp.a.start),
            a_index: cs(&lp.a.index),
            a_value: cs(&lp.a.value),
            col_cost: cs(&lp.col_cost),
            col_lower: cs(&lp.col_lower),
            col_upper: cs(&lp.col_upper),
            row_lower: cs(&lp.row_lower),
            row_upper: cs(&lp.row_upper),
            sense: lp.sense,
            offset: lp.offset,
            has_scale,
            col_scale,
            row_scale,
            ar_start: vs(&mut self.ar.start),
            ar_p_end: vs(&mut self.ar.p_end),
            ar_index: vs(&mut self.ar.index),
            ar_value: vs(&mut self.ar.value),
            basic_index: vs(&mut self.basis.basic_index),
            nonbasic_flag: vs(&mut self.basis.nonbasic_flag),
            nonbasic_move: vs(&mut self.basis.nonbasic_move),
            basis_hash: &mut self.basis.hash,
            work_cost: vs(&mut self.work_cost),
            work_dual: vs(&mut self.work_dual),
            work_shift: vs(&mut self.work_shift),
            work_lower: vs(&mut self.work_lower),
            work_upper: vs(&mut self.work_upper),
            work_range: vs(&mut self.work_range),
            work_value: vs(&mut self.work_value),
            work_lower_shift: vs(&mut self.work_lower_shift),
            work_upper_shift: vs(&mut self.work_upper_shift),
            base_lower: vs(&mut self.base_lower),
            base_upper: vs(&mut self.base_upper),
            base_value: vs(&mut self.base_value),
            num_tot_random_value: vs(&mut self.num_tot_random_value),
            dual_edge_weight: vs(&mut self.dual_edge_weight),
            scattered_dual_edge_weight: vs(&mut self.scattered_dual_edge_weight),
            col_aq_density: &mut self.sh.info.col_aq_density,
            row_ep_density: &mut self.sh.info.row_ep_density,
            row_ap_density: &mut self.info.row_ap_density,
            row_dse_density: &mut self.info.row_dse_density,
            primal_col_density: &mut self.info.primal_col_density,
            dual_col_density: &mut self.info.dual_col_density,
            update_count: &mut self.info.update_count,
            num_basic_logicals: &mut self.info.num_basic_logicals,
            updated_dual_objective_value: &mut self.info.updated_dual_objective_value,
            primal_objective_value: &mut self.sh.info.primal_objective_value,
            dual_objective_value: &mut self.sh.info.dual_objective_value,
            num_primal_infeasibilities: &mut self.sh.info.num_primal_infeasibilities,
            max_primal_infeasibility: &mut self.sh.info.max_primal_infeasibility,
            sum_primal_infeasibilities: &mut self.sh.info.sum_primal_infeasibilities,
            num_dual_infeasibilities: &mut self.sh.info.num_dual_infeasibilities,
            max_dual_infeasibility: &mut self.sh.info.max_dual_infeasibility,
            sum_dual_infeasibilities: &mut self.sh.info.sum_dual_infeasibilities,
            costs_shifted: &mut self.info.costs_shifted,
            costs_perturbed: &mut self.info.costs_perturbed,
            bounds_shifted: &mut self.info.bounds_shifted,
            bounds_perturbed: &mut self.info.bounds_perturbed,
            price_strategy: self.info.price_strategy,
            dual_simplex_cost_perturbation_multiplier: &mut self.info.dual_simplex_cost_perturbation_multiplier,
            primal_simplex_bound_perturbation_multiplier: &mut self.info.primal_simplex_bound_perturbation_multiplier,
            primal_feasibility_tolerance: o.primal_feasibility_tolerance,
            dual_feasibility_tolerance: o.dual_feasibility_tolerance,
            cost_scale_factor: o.cost_scale_factor,
            output_flag: o.output_flag,
            cost_scale: self.cost_scale,
            cost_perturbation_base: &mut self.cost_perturbation_base,
            cost_perturbation_max_abs_cost: &mut self.cost_perturbation_max_abs_cost,
            simplex_in_scaled_space: self.simplex_in_scaled_space,
            update_limit: &mut self.info.update_limit,
            build_synthetic_tick: &mut self.build_synthetic_tick,
            total_synthetic_tick: &mut self.total_synthetic_tick,
            factor: &mut self.factor,
            factor_num_col: self.fs.num_col,
            factor_a_start: fa_start,
            factor_a_index: fa_index,
            factor_a_value: fa_value,
            status: &mut self.sh.status,
            iteration_count: &mut self.sh.iteration_count,
            updated_primal_objective_value: &mut self.info.updated_primal_objective_value,
            allow_bound_perturbation: &mut self.info.allow_bound_perturbation,
            backtracking: &mut self.info.backtracking,
            primal_phase1_iteration_count: &mut self.info.primal_phase1_iteration_count,
            primal_phase2_iteration_count: &mut self.info.primal_phase2_iteration_count,
            primal_bound_swap: &mut self.info.primal_bound_swap,
            col_basic_feasibility_change_density: &mut self.info.col_basic_feasibility_change_density,
            row_basic_feasibility_change_density: &mut self.info.row_basic_feasibility_change_density,
            col_steepest_edge_density: &mut self.info.col_steepest_edge_density,
            primal_simplex_phase1_cost_perturbation_multiplier: self.info.primal_simplex_phase1_cost_perturbation_multiplier,
            simplex_primal_edge_weight_strategy: o.simplex_primal_edge_weight_strategy,
            simplex_iteration_limit: o.simplex_iteration_limit,
            bailout_in_cpp: o.time_limit < INF || env.interrupt_callback,
            iteration_report: o.log_dev_level >= LOG_VERBOSE_LEVEL,
        }
    }

    /// The CHekk view for a solve and the HEkk methods it reaches
    /// (HEkk::rustHekk), with the NLA's scaling: the solver's vectors must
    /// be sized for the call
    pub fn c_hekk(&mut self, env: &LpsEnv, draw_random_vectors: bool) -> CHekk {
        let ekk = self.c_ekk(env, true);
        let o = &env.opt;
        let lp = &env.lp;
        let num_col_permutation = CSlice {
            p: self.num_col_permutation.as_mut_ptr(),
            n: (lp.num_col as usize).min(self.num_col_permutation.len()) as i32,
        };
        CHekk {
            ekk,
            host: env.host,
            iteration_count: sh(&mut self.sh.iteration_count),
            model_status: sh(&mut self.sh.model_status),
            solve_bailout: sh(&mut self.solve_bailout),
            called_return_from_solve: sh(&mut self.called_return_from_solve),
            exit_algorithm: sh(&mut self.sh.exit_algorithm),
            return_primal_solution_status: sh(&mut self.return_primal_solution_status),
            return_dual_solution_status: sh(&mut self.return_dual_solution_status),
            dual_values_valid: sh(&mut self.sh.dual_values_valid),
            dual_values_scaled: sh(&mut self.dual_values_scaled),
            dual_values_basis_hash: sh(&mut self.dual_values_basis_hash),
            dual_values_cost_hash: sh(&mut self.dual_values_cost_hash),
            fresh_unperturbed_dual: sh(&mut self.fresh_unperturbed_dual),
            fresh_dual: sh(&mut self.fresh_dual),
            fresh_primal: sh(&mut self.fresh_primal),
            edge_weight_error: sh(&mut self.edge_weight_error),
            dual_simplex_cleanup_level: sh(&mut self.dual_simplex_cleanup_level),
            dual_simplex_phase1_cleanup_level: sh(&mut self.dual_simplex_phase1_cleanup_level),
            previous_iteration_cycling_detected: sh(&mut self.previous_iteration_cycling_detected),
            random: sh(&mut self.random),
            basis_records: &mut self.records,
            nla_build_synthetic_tick: sh(&mut self.nla_build_synthetic_tick),
            num_invert: Shared::new(env.num_invert),
            debug_solve_call_num: self.sh.debug_solve_call_num,
            ar_matrix_is_scaled: sh(&mut self.ar_matrix_is_scaled),
            random_vectors_drawn: sh(&mut self.random_vectors_drawn),
            draw_random_vectors,
            lp_is_scaled: lp.is_scaled,
            lp_has_scaling: lp.scale_has_scaling,
            lp_col_scale: cs(&lp.scale_col),
            lp_row_scale: cs(&lp.scale_row),
            model_name: CSlice { p: env.model_name.ptr, n: env.model_name.len as i32 },
            saved_dual_edge_weight: Cell::new(vs(&mut self.saved_dual_edge_weight)),
            saved_dual_edge_weight_taken: Cell::new(false),
            saved_dual_edge_weight_vec: &mut self.saved_dual_edge_weight,
            basis_origin: &self.basis.debug_origin_name,
            dual_ray_index: sh(&mut self.sh.dual_ray_index),
            dual_ray_sign: sh(&mut self.sh.dual_ray_sign),
            primal_ray_index: sh(&mut self.sh.primal_ray_index),
            primal_ray_sign: sh(&mut self.sh.primal_ray_sign),
            ray_value_clear: Cell::new(0),
            basis_debug_id: sh(&mut self.basis.debug_id),
            basis_debug_update_count: sh(&mut self.basis.debug_update_count),
            has_invert: sh(&mut self.sh.status.has_invert),
            has_fresh_invert: sh(&mut self.sh.status.has_fresh_invert),
            has_fresh_rebuild: sh(&mut self.sh.status.has_fresh_rebuild),
            has_dual_objective_value: sh(&mut self.sh.status.has_dual_objective_value),
            has_primal_objective_value: sh(&mut self.sh.status.has_primal_objective_value),
            has_dual_steepest_edge_weights: sh(&mut self.sh.status.has_dual_steepest_edge_weights),
            has_ar_matrix: sh(&mut self.sh.status.has_ar_matrix),
            valid_backtracking_basis: sh(&mut self.info.valid_backtracking_basis),
            bt_basic_index: vs(&mut self.bt_basis.basic_index),
            bt_nonbasic_flag: vs(&mut self.bt_basis.nonbasic_flag),
            bt_nonbasic_move: vs(&mut self.bt_basis.nonbasic_move),
            bt_hash: sh(&mut self.bt_basis.hash),
            bt_debug_id: sh(&mut self.bt_basis.debug_id),
            bt_debug_update_count: sh(&mut self.bt_basis.debug_update_count),
            bt_costs_shifted: sh(&mut self.info.bt_costs_shifted),
            bt_costs_perturbed: sh(&mut self.info.bt_costs_perturbed),
            bt_bounds_shifted: sh(&mut self.info.bt_bounds_shifted),
            bt_bounds_perturbed: sh(&mut self.info.bt_bounds_perturbed),
            bt_work_shift: vs(&mut self.bt_work_shift),
            bt_edge_weight: vs(&mut self.bt_edge_weight),
            devex_index: vs(&mut self.devex_index),
            num_tot_permutation: vs(&mut self.num_tot_permutation),
            num_col_permutation,
            dual_phase1_iteration_count: sh(&mut self.info.dual_phase1_iteration_count),
            dual_phase2_iteration_count: sh(&mut self.info.dual_phase2_iteration_count),
            allow_cost_shifting: sh(&mut self.info.allow_cost_shifting),
            allow_cost_perturbation: sh(&mut self.info.allow_cost_perturbation),
            store_squared_primal_infeasibility: sh(&mut self.info.store_squared_primal_infeasibility),
            factor_pivot_threshold: sh(&mut self.sh.info.factor_pivot_threshold),
            col_bfrt_density: sh(&mut self.info.col_bfrt_density),
            costly_dse_measure: sh(&mut self.info.costly_dse_measure),
            costly_dse_frequency: sh(&mut self.info.costly_dse_frequency),
            num_costly_dse_iteration: sh(&mut self.info.num_costly_dse_iteration),
            average_log_low_dse_weight_error: sh(&mut self.info.average_log_low_dse_weight_error),
            average_log_high_dse_weight_error: sh(&mut self.info.average_log_high_dse_weight_error),
            simplex_strategy: sh(&mut self.info.simplex_strategy),
            min_concurrency: sh(&mut self.info.min_concurrency),
            max_concurrency: sh(&mut self.info.max_concurrency),
            num_concurrency: sh(&mut self.info.num_concurrency),
            iteration_count0: sh(&mut self.info.iteration_count0),
            dual_phase1_iteration_count0: sh(&mut self.info.dual_phase1_iteration_count0),
            dual_phase2_iteration_count0: sh(&mut self.info.dual_phase2_iteration_count0),
            primal_phase1_iteration_count0: sh(&mut self.info.primal_phase1_iteration_count0),
            primal_phase2_iteration_count0: sh(&mut self.info.primal_phase2_iteration_count0),
            primal_bound_swap0: sh(&mut self.info.primal_bound_swap0),
            control_iteration_count0: self.info.control_iteration_count0,
            allow_dual_steepest_edge_to_devex_switch: self.info.allow_dual_steepest_edge_to_devex_switch,
            dual_steepest_edge_weight_log_error_threshold: self.info.dual_steepest_edge_weight_log_error_threshold,
            dual_edge_weight_strategy: self.sh.info.dual_edge_weight_strategy,
            run_quiet: self.info.run_quiet,
            hfactor_pivot_threshold: sh(&mut self.fs.pivot_threshold),
            hfactor_pivot_tolerance: self.fs.pivot_tolerance,
            hfactor_time_limit: self.fs.time_limit,
            objective_bound: o.objective_bound,
            time_limit: o.time_limit,
            simplex_iteration_limit: o.simplex_iteration_limit,
            simplex_update_limit: o.simplex_update_limit,
            max_dual_simplex_cleanup_level: o.max_dual_simplex_cleanup_level,
            max_dual_simplex_phase1_cleanup_level: o.max_dual_simplex_phase1_cleanup_level,
            dual_simplex_pivot_growth_tolerance: o.dual_simplex_pivot_growth_tolerance,
            simplex_dse_exact_init_max_rows: o.simplex_dse_exact_init_max_rows,
            small_matrix_value: o.small_matrix_value,
            dual_steepest_edge_weight_error_tolerance: o.dual_steepest_edge_weight_error_tolerance,
            no_unnecessary_rebuild_refactor: o.no_unnecessary_rebuild_refactor,
            rebuild_refactor_solution_error_tolerance: o.rebuild_refactor_solution_error_tolerance,
            option_simplex_strategy: o.simplex_strategy,
            simplex_min_concurrency: o.simplex_min_concurrency,
            simplex_max_concurrency: o.simplex_max_concurrency,
            allow_unbounded_or_infeasible: o.allow_unbounded_or_infeasible,
            less_infeasible_dse_check: o.less_infeasible_dse_check,
            less_infeasible_dse_choose_row: o.less_infeasible_dse_choose_row,
            num_threads: env.num_threads,
            dev_level: o.dev_level,
            output_flag: o.output_flag,
            log_dev_level: o.log_dev_level,
            factor_dev_level: self.fs.dev_level,
            dev_log: o.dev_level != 0,
            iteration_report: o.log_dev_level >= LOG_VERBOSE_LEVEL,
            interrupt_callback: env.interrupt_callback,
            report: Shared::new(env.report),
        }
    }
}

/// kIterationReportLogType (HighsLogType::kVerbose)
const LOG_VERBOSE_LEVEL: i32 = 3;

/// The unscaled infeasibilities of getUnscaledInfeasibilities, in
/// HighsInfo's field order
#[repr(C)]
#[derive(Default)]
pub struct UnscaledInfeasibilities {
    pub num_primal_infeasibilities: i32,
    pub max_primal_infeasibility: f64,
    pub sum_primal_infeasibilities: f64,
    pub num_dual_infeasibilities: i32,
    pub max_dual_infeasibility: f64,
    pub sum_dual_infeasibilities: f64,
}

/// The slices of HEkk's data that the ranging reads (HighsRanging.cpp)
#[repr(C)]
pub struct RangingSlices {
    pub work_value: RsMut<f64>,
    pub work_dual: RsMut<f64>,
    pub work_cost: RsMut<f64>,
    pub work_lower: RsMut<f64>,
    pub work_upper: RsMut<f64>,
    pub base_value: RsMut<f64>,
    pub base_lower: RsMut<f64>,
    pub base_upper: RsMut<f64>,
    pub nonbasic_flag: RsMut<i8>,
    pub nonbasic_move: RsMut<i8>,
    pub basic_index: RsMut<i32>,
}

fn rm<T>(v: &mut Vec<T>) -> RsMut<T> {
    RsMut { ptr: v.as_mut_ptr(), len: v.len() }
}

mod ffi {
    use super::*;

    fn s<'a>(p: *mut LpSolver) -> &'a mut LpSolver {
        // SAFETY: a solver made by highs_rs_lps_new, used by one thread at
        // a time (the C++ HEkk that owns it)
        unsafe { &mut *p }
    }

    fn en<'a>(e: *const LpsEnv) -> &'a LpsEnv {
        // SAFETY: filled by HEkk::rsEnv for the call
        unsafe { &*e }
    }

    #[no_mangle]
    pub extern "C" fn highs_rs_lps_new() -> *mut LpSolver {
        Box::into_raw(Box::default())
    }

    /// # Safety
    /// `p` from highs_rs_lps_new, not used afterwards
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lps_free(p: *mut LpSolver) {
        drop(Box::from_raw(p));
    }

    #[no_mangle]
    pub extern "C" fn highs_rs_lps_shared(p: *mut LpSolver) -> *mut EkkShared {
        &mut s(p).sh
    }

    #[no_mangle]
    pub extern "C" fn highs_rs_lps_records(p: *mut LpSolver) -> *mut BasisRecords {
        &mut s(p).records
    }

    #[no_mangle]
    pub extern "C" fn highs_rs_lps_clear(p: *mut LpSolver) {
        s(p).clear();
    }

    #[no_mangle]
    pub extern "C" fn highs_rs_lps_invalidate(p: *mut LpSolver) {
        s(p).invalidate();
    }

    #[no_mangle]
    pub extern "C" fn highs_rs_lps_update_status(p: *mut LpSolver, action: i32) -> bool {
        s(p).update_status(action)
    }

    #[no_mangle]
    pub extern "C" fn highs_rs_lps_clear_ray_records(p: *mut LpSolver) {
        s(p).clear_ray_records();
    }

    #[no_mangle]
    pub extern "C" fn highs_rs_lps_move_lp(p: *mut LpSolver, env: *const LpsEnv) -> bool {
        let solver = s(p);
        let env = solver.env_of(en(env));
        solver.move_lp(&env)
    }

    #[no_mangle]
    pub extern "C" fn highs_rs_lps_set_basis_logical(p: *mut LpSolver, env: *const LpsEnv) {
        let solver = s(p);
        let env = solver.env_of(en(env));
        solver.set_basis_logical(&env);
    }

    /// # Safety
    /// The statuses sized as the LP; `origin` valid for `origin_len` bytes
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lps_set_basis(
        p: *mut LpSolver,
        env: *const LpsEnv,
        col_status: *const u8,
        row_status: *const u8,
        debug_id: i32,
        debug_update_count: i32,
        origin: *const u8,
        origin_len: usize,
    ) {
        let solver = s(p);
        let env = solver.env_of(en(env));
        let col = crate::ffi::sl(col_status, env.lp.num_col);
        let row = crate::ffi::sl(row_status, env.lp.num_row);
        let origin = String::from_utf8_lossy(crate::ffi::sl(origin, origin_len as i32));
        solver.set_basis(&env, col, row, debug_id, debug_update_count, &origin);
    }

    /// The statuses of HEkk::getHighsBasis(use_lp) (sized by the caller),
    /// its debug identifiers and its origin (valid until the basis
    /// changes)
    ///
    /// # Safety
    /// The statuses sized as `use_lp`
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lps_get_highs_basis(
        p: *mut LpSolver,
        use_lp: *const CLp,
        sense: i32,
        col_status: *mut u8,
        row_status: *mut u8,
        debug_id: *mut i32,
        debug_update_count: *mut i32,
        origin: *mut *const u8,
        origin_len: *mut usize,
    ) {
        let solver = s(p);
        let lp = if use_lp.is_null() { solver.lp.view() } else { *use_lp };
        let col = crate::ffi::sl_mut(col_status, lp.num_col);
        let row = crate::ffi::sl_mut(row_status, lp.num_row);
        let _ = sense;
        let sense = solver.lp.sense;
        let (id, count) = solver.get_highs_basis(&lp, sense, col, row);
        *debug_id = id;
        *debug_update_count = count;
        *origin = solver.basis.debug_origin_name.as_ptr();
        *origin_len = solver.basis.debug_origin_name.len();
    }

    /// # Safety
    /// The vectors sized as the LP
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lps_get_solution(
        p: *mut LpSolver,
        env: *const LpsEnv,
        col_value: *mut f64,
        col_dual: *mut f64,
        row_value: *mut f64,
        row_dual: *mut f64,
    ) {
        let solver = s(p);
        let env = solver.env_of(en(env));
        let (nc, nr) = (env.lp.num_col, env.lp.num_row);
        solver.get_solution(
            &env,
            crate::ffi::sl_mut(col_value, nc),
            crate::ffi::sl_mut(col_dual, nc),
            crate::ffi::sl_mut(row_value, nr),
            crate::ffi::sl_mut(row_dual, nr),
        );
    }

    /// # Safety
    /// `lp` a valid view
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lps_unscale_simplex(p: *mut LpSolver, lp: *const CLp) {
        s(p).unscale_simplex(&*lp);
    }

    /// # Safety
    /// `lp` a valid view
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lps_unscaled_infeasibilities(
        p: *mut LpSolver,
        env: *const LpsEnv,
        lp: *const CLp,
        out: *mut UnscaledInfeasibilities,
    ) {
        let solver = s(p);
        let env = solver.env_of(en(env));
        let lp = if lp.is_null() { env.lp } else { *lp };
        solver.get_unscaled_infeasibilities(&env, &lp, &mut *out);
    }

    #[no_mangle]
    pub extern "C" fn highs_rs_lps_add_rows(p: *mut LpSolver, num_weighted_row: i32, new_num_row: i32) {
        s(p).add_rows(num_weighted_row, new_num_row);
    }

    /// # Safety
    /// `ic` a valid index collection
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lps_delete_rows(p: *mut LpSolver, ic: *const CIndexCollection) {
        s(p).delete_rows(&(*ic).view());
    }

    #[no_mangle]
    pub extern "C" fn highs_rs_lps_resize_basis(p: *mut LpSolver, num_tot: i32) {
        s(p).resize_basis(num_tot as usize);
    }

    #[no_mangle]
    pub extern "C" fn highs_rs_lps_append_basic_rows(p: *mut LpSolver, num_col: i32, num_row: i32, new_num_row: i32) {
        s(p).append_basic_rows(num_col, num_row, new_num_row);
    }

    #[no_mangle]
    pub extern "C" fn highs_rs_lps_flip_nonbasic_move(p: *mut LpSolver, var: i32) {
        s(p).flip_nonbasic_move(var as usize);
    }

    #[no_mangle]
    pub extern "C" fn highs_rs_lps_initialise_basis_and_factor(
        p: *mut LpSolver,
        env: *const LpsEnv,
        only_from_known_basis: bool,
    ) -> i32 {
        let solver = s(p);
        let env = solver.env_of(en(env));
        solver.initialise_simplex_lp_basis_and_factor(&env, only_from_known_basis)
    }

    #[no_mangle]
    pub extern "C" fn highs_rs_lps_lp_factor_row_compatible(
        p: *mut LpSolver,
        env: *const LpsEnv,
        expected_num_row: i32,
    ) -> bool {
        let solver = s(p);
        let env = solver.env_of(en(env));
        solver.lp_factor_row_compatible(&env, expected_num_row)
    }

    /// # Safety
    /// `rhs` a valid C++ HVector of the LP's row count
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lps_nla_solve(
        p: *mut LpSolver,
        env: *const LpsEnv,
        rhs: *mut CHVec,
        expected_density: f64,
        transposed: bool,
    ) {
        let solver = s(p);
        let env = solver.env_of(en(env));
        solver.nla_solve(&env, &mut *rhs, expected_density, transposed);
    }

    #[no_mangle]
    pub extern "C" fn highs_rs_lps_put_iterate(p: *mut LpSolver) {
        s(p).put_iterate();
    }

    #[no_mangle]
    pub extern "C" fn highs_rs_lps_get_iterate(p: *mut LpSolver) -> bool {
        s(p).get_iterate()
    }

    /// # Safety
    /// `lp` a valid view, `name` valid for `name_len` bytes
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lps_basis_condition(
        p: *mut LpSolver,
        env: *const LpsEnv,
        lp: *const CLp,
        name: *const u8,
        name_len: usize,
        exact: bool,
        report: bool,
    ) -> f64 {
        let solver = s(p);
        let env = solver.env_of(en(env));
        let (lp, name) = if lp.is_null() {
            (env.lp, String::from_utf8_lossy(&solver.lp.model_name).into_owned())
        } else {
            (*lp, String::from_utf8_lossy(crate::ffi::sl(name, name_len as i32)).into_owned())
        };
        solver.compute_basis_condition(&env, &lp, &name, exact, report)
    }

    #[no_mangle]
    pub extern "C" fn highs_rs_lps_proof_of_primal_infeasibility(p: *mut LpSolver, env: *const LpsEnv) -> bool {
        let solver = s(p);
        let env = solver.env_of(en(env));
        solver.proof_of_primal_infeasibility(&env)
    }

    #[no_mangle]
    pub extern "C" fn highs_rs_lps_solve(p: *mut LpSolver, env: *const LpsEnv, force_phase2: bool) -> LpsSolveOut {
        let solver = s(p);
        let env = solver.env_of(en(env));
        solver.solve(&env, force_phase2)
    }

    #[no_mangle]
    pub extern "C" fn highs_rs_lps_return_from_solve(p: *mut LpSolver) {
        s(p).return_from_ekk_solve();
    }

    /// basis_.basicIndex_ (valid until the basis changes)
    ///
    /// # Safety
    /// `n` valid
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lps_basic_index(p: *mut LpSolver, n: *mut i32) -> *mut i32 {
        let v = &mut s(p).basis.basic_index;
        *n = v.len() as i32;
        v.as_mut_ptr()
    }

    /// basis_.nonbasicFlag_ (`which` 0) or nonbasicMove_ (1)
    ///
    /// # Safety
    /// `n` valid
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lps_nonbasic(p: *mut LpSolver, which: i32, n: *mut i32) -> *mut i8 {
        let b = &mut s(p).basis;
        let v = if which == 0 { &mut b.nonbasic_flag } else { &mut b.nonbasic_move };
        *n = v.len() as i32;
        v.as_mut_ptr()
    }

    /// dual_edge_weight_ (valid until it changes)
    #[no_mangle]
    pub extern "C" fn highs_rs_lps_dual_edge_weight(p: *mut LpSolver) -> *const f64 {
        s(p).dual_edge_weight.as_ptr()
    }

    /// info_.workDual_
    ///
    /// # Safety
    /// `n` valid
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lps_work_dual(p: *mut LpSolver, n: *mut i32) -> *const f64 {
        let v = &s(p).work_dual;
        *n = v.len() as i32;
        v.as_ptr()
    }

    /// The value of the primal (`primal`) or dual ray record
    ///
    /// # Safety
    /// `n` valid
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lps_ray_value(p: *mut LpSolver, primal: bool, n: *mut usize) -> *const f64 {
        let solver = s(p);
        let v = if primal { &solver.primal_ray_value } else { &solver.dual_ray_value };
        *n = v.len();
        v.as_ptr()
    }

    /// Set the value of the primal (`primal`) or dual ray record
    ///
    /// # Safety
    /// `value` valid for `n` reads
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lps_set_ray_value(p: *mut LpSolver, primal: bool, value: *const f64, n: usize) {
        let solver = s(p);
        let v = if primal { &mut solver.primal_ray_value } else { &mut solver.dual_ray_value };
        v.clear();
        v.extend_from_slice(crate::ffi::sl(value, n as i32));
    }

    /// # Safety
    /// `out` valid
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lps_ranging_slices(p: *mut LpSolver, out: *mut RangingSlices) {
        let solver = s(p);
        *out = RangingSlices {
            work_value: rm(&mut solver.work_value),
            work_dual: rm(&mut solver.work_dual),
            work_cost: rm(&mut solver.work_cost),
            work_lower: rm(&mut solver.work_lower),
            work_upper: rm(&mut solver.work_upper),
            base_value: rm(&mut solver.base_value),
            base_lower: rm(&mut solver.base_lower),
            base_upper: rm(&mut solver.base_upper),
            nonbasic_flag: rm(&mut solver.basis.nonbasic_flag),
            nonbasic_move: rm(&mut solver.basis.nonbasic_move),
            basic_index: rm(&mut solver.basis.basic_index),
        };
    }

    /// Copy a C++ LP into the solver's (HEkk::moveLp's move)
    ///
    /// # Safety
    /// `lp` a valid view, `name` valid for `name_len` bytes
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lps_import_lp(p: *mut LpSolver, lp: *const CLp, name: *const u8, name_len: usize) {
        s(p).lp.import(&*lp, crate::ffi::sl(name, name_len as i32));
    }

    /// A view of the solver's LP, valid until it changes
    ///
    /// # Safety
    /// `out` valid
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lps_lp_view(p: *mut LpSolver, out: *mut CLp) {
        *out = s(p).lp.view();
    }

    /// The model name of the solver's LP
    ///
    /// # Safety
    /// `out` valid
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lps_model_name(p: *mut LpSolver, out: *mut RsMut<u8>) {
        *out = rs(&mut s(p).lp.model_name);
    }

    /// HEkk::clearEkkLp's LP part
    #[no_mangle]
    pub extern "C" fn highs_rs_lps_clear_lp(p: *mut LpSolver) {
        s(p).lp.clear();
    }

    /// The simplex NLA's LP is the solver's (`rust`) or a C++ LP
    #[no_mangle]
    pub extern "C" fn highs_rs_lps_set_nla_rust(p: *mut LpSolver, rust: bool) {
        s(p).set_nla_rust(rust);
    }

    /// HEkk::lp_.num_row_ = num_row (the rows' data are not kept)
    #[no_mangle]
    pub extern "C" fn highs_rs_lps_set_lp_num_row(p: *mut LpSolver, num_row: i32) {
        s(p).lp.num_row = num_row;
    }

    /// The factor's row count (HFactor::num_row)
    #[no_mangle]
    pub extern "C" fn highs_rs_lps_factor_num_row(p: *mut LpSolver) -> i32 {
        s(p).fs.num_row
    }

    #[allow(dead_code)]
    fn _unused(_: *mut c_void) {}
}
