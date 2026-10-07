//! HighsSolution.cpp: the KKT failures of a solution (getKktFailures,
//! getVariableKktFailures, the basis and glpsol error measures,
//! complementarity violations), the primal and dual objective values,
//! lpKktCheck and reportKktFailures.
//!
//! clang fuses `obj += c * x` and `dobj += bound * dual`; the LTO build
//! vectorizes the objective sums, so HighsLp::objectiveValue and
//! computeObjectiveValue add their first n/4*4 terms rounded and the
//! rest fused (`dot_blocked` by 4), and the quadratic term of the dual
//! objective likewise by 8.

use super::ffi::{CLp, RsMut};
use super::lp_utils::cmax;
use super::{var_type, Log, LogType, INF};
use crate::ipx::utils::dot_blocked;
use crate::log_dev;
use crate::log_user;
use crate::util::cdouble::CDouble;
use crate::util::fma::ClangFma;

pub const SOLUTION_STATUS_NONE: i32 = 0;
pub const SOLUTION_STATUS_INFEASIBLE: i32 = 1;
pub const SOLUTION_STATUS_FEASIBLE: i32 = 2;

const ILLEGAL_COUNT: i32 = -1;
const ILLEGAL_MEASURE: f64 = INF;
const DEFAULT_KKT_TOLERANCE: f64 = 1e-7;

// HighsModelStatus
pub const MODEL_STATUS_OPTIMAL: i32 = 7;
pub const MODEL_STATUS_UNBOUNDED_OR_INFEASIBLE: i32 = 9;
pub const MODEL_STATUS_UNBOUNDED: i32 = 10;
pub const MODEL_STATUS_UNKNOWN: i32 = 15;

// HighsBasisStatus
pub const BASIS_LOWER: u8 = 0;
pub const BASIS_BASIC: u8 = 1;
pub const BASIS_UPPER: u8 = 2;

const SOL_NO: u8 = 0;
const SOL_LO: u8 = 1;
const SOL_UP: u8 = 2;

/// HighsInfoStruct (the data of HighsInfo, a standard-layout base)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Info {
    pub valid: bool,
    pub mip_node_count: i64,
    pub simplex_iteration_count: i32,
    pub ipm_iteration_count: i32,
    pub crossover_iteration_count: i32,
    pub pdlp_iteration_count: i32,
    pub qp_iteration_count: i32,
    pub primal_solution_status: i32,
    pub dual_solution_status: i32,
    pub basis_validity: i32,
    pub objective_function_value: f64,
    pub mip_dual_bound: f64,
    pub mip_gap: f64,
    pub max_integrality_violation: f64,
    pub num_primal_infeasibilities: i32,
    pub max_primal_infeasibility: f64,
    pub sum_primal_infeasibilities: f64,
    pub num_dual_infeasibilities: i32,
    pub max_dual_infeasibility: f64,
    pub sum_dual_infeasibilities: f64,
    pub num_semi_infeasibilities: i32,
    pub max_semi_infeasibility: f64,
    pub sum_semi_infeasibilities: f64,
    pub num_relative_primal_infeasibilities: i32,
    pub max_relative_primal_infeasibility: f64,
    pub num_relative_dual_infeasibilities: i32,
    pub max_relative_dual_infeasibility: f64,
    pub num_primal_residual_errors: i32,
    pub max_primal_residual_error: f64,
    pub num_dual_residual_errors: i32,
    pub max_dual_residual_error: f64,
    pub num_relative_primal_residual_errors: i32,
    pub max_relative_primal_residual_error: f64,
    pub num_relative_dual_residual_errors: i32,
    pub max_relative_dual_residual_error: f64,
    pub num_complementarity_violations: i32,
    pub max_complementarity_violation: f64,
    pub primal_dual_objective_error: f64,
    pub primal_dual_integral: f64,
}

impl Info {
    /// HighsInfo::invalidatePrimalKkt
    pub fn invalidate_primal_kkt(&mut self) {
        self.primal_solution_status = SOLUTION_STATUS_NONE;
        self.num_primal_infeasibilities = ILLEGAL_COUNT;
        self.max_primal_infeasibility = ILLEGAL_MEASURE;
        self.sum_primal_infeasibilities = ILLEGAL_MEASURE;
        self.num_semi_infeasibilities = ILLEGAL_COUNT;
        self.max_semi_infeasibility = ILLEGAL_MEASURE;
        self.sum_semi_infeasibilities = ILLEGAL_MEASURE;
        self.num_relative_primal_infeasibilities = ILLEGAL_COUNT;
        self.max_relative_primal_infeasibility = ILLEGAL_MEASURE;
        self.num_primal_residual_errors = ILLEGAL_COUNT;
        self.max_primal_residual_error = ILLEGAL_MEASURE;
        self.num_relative_primal_residual_errors = ILLEGAL_COUNT;
        self.max_relative_primal_residual_error = ILLEGAL_MEASURE;
        self.num_complementarity_violations = ILLEGAL_COUNT;
        self.max_complementarity_violation = ILLEGAL_MEASURE;
        self.primal_dual_objective_error = ILLEGAL_MEASURE;
    }
    /// HighsInfo::invalidateDualKkt
    pub fn invalidate_dual_kkt(&mut self) {
        self.dual_solution_status = SOLUTION_STATUS_NONE;
        self.num_dual_infeasibilities = ILLEGAL_COUNT;
        self.max_dual_infeasibility = ILLEGAL_MEASURE;
        self.sum_dual_infeasibilities = ILLEGAL_MEASURE;
        self.num_relative_dual_infeasibilities = ILLEGAL_COUNT;
        self.max_relative_dual_infeasibility = ILLEGAL_MEASURE;
        self.num_dual_residual_errors = ILLEGAL_COUNT;
        self.max_dual_residual_error = ILLEGAL_MEASURE;
        self.num_relative_dual_residual_errors = ILLEGAL_COUNT;
        self.max_relative_dual_residual_error = ILLEGAL_MEASURE;
        self.num_complementarity_violations = ILLEGAL_COUNT;
        self.max_complementarity_violation = ILLEGAL_MEASURE;
        self.primal_dual_objective_error = ILLEGAL_MEASURE;
    }
    /// HighsInfo::invalidateKkt
    pub fn invalidate_kkt(&mut self) {
        self.invalidate_primal_kkt();
        self.invalidate_dual_kkt();
    }
}

/// HighsError
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct HError {
    pub absolute_value: f64,
    pub absolute_index: i32,
    pub relative_value: f64,
    pub relative_index: i32,
}

impl HError {
    fn reset(&mut self) {
        *self = HError { absolute_value: 0.0, absolute_index: 0, relative_value: 0.0, relative_index: 0 };
    }
    fn invalidate(&mut self) {
        *self = HError { absolute_value: INF, absolute_index: -1, relative_value: INF, relative_index: -1 };
    }
}

/// HighsPrimalDualErrors
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PrimalDualErrors {
    pub num_nonzero_basic_duals: i32,
    pub max_nonzero_basic_dual: f64,
    pub sum_nonzero_basic_duals: f64,
    pub num_off_bound_nonbasic: i32,
    pub max_off_bound_nonbasic: f64,
    pub sum_off_bound_nonbasic: f64,
    pub glpsol_num_primal_residual_errors: i32,
    pub glpsol_num_dual_residual_errors: i32,
    pub glpsol_max_primal_residual: HError,
    pub glpsol_max_primal_infeasibility: HError,
    pub glpsol_max_dual_residual: HError,
    pub glpsol_max_dual_infeasibility: HError,
}

/// HighsSolution
#[repr(C)]
pub struct CSolution {
    pub value_valid: bool,
    pub dual_valid: bool,
    pub col_value: RsMut<f64>,
    pub col_dual: RsMut<f64>,
    pub row_value: RsMut<f64>,
    pub row_dual: RsMut<f64>,
}

/// HighsBasis
#[repr(C)]
pub struct CBasis {
    pub valid: bool,
    pub col_status: RsMut<u8>,
    pub row_status: RsMut<u8>,
}

/// The options of the KKT checks
#[repr(C)]
pub struct CKktOptions {
    pub log: Log,
    pub primal_feasibility_tolerance: f64,
    pub dual_feasibility_tolerance: f64,
    pub mip_feasibility_tolerance: f64,
    pub primal_residual_tolerance: f64,
    pub dual_residual_tolerance: f64,
    pub optimality_tolerance: f64,
    pub kkt_tolerance: f64,
    pub log_dev_level: i32,
    pub full_lp_kkt_check: bool,
}

/// Borrowed arrays of an LP
pub struct LpRef<'a> {
    pub num_col: usize,
    pub num_row: usize,
    pub col_cost: &'a [f64],
    pub col_lower: &'a [f64],
    pub col_upper: &'a [f64],
    pub row_lower: &'a [f64],
    pub row_upper: &'a [f64],
    pub colwise: bool,
    pub start: &'a [i32],
    pub index: &'a [i32],
    pub value: &'a [f64],
    pub sense: f64,
    pub offset: f64,
    pub integrality: &'a [u8],
}

impl LpRef<'_> {
    /// # Safety
    /// The view's arrays must be valid while the reference lives
    pub unsafe fn new<'a>(lp: &CLp) -> LpRef<'a> {
        LpRef {
            num_col: lp.num_col as usize,
            num_row: lp.num_row as usize,
            col_cost: lp.col_cost.get(),
            col_lower: lp.col_lower.get(),
            col_upper: lp.col_upper.get(),
            row_lower: lp.row_lower.get(),
            row_upper: lp.row_upper.get(),
            colwise: lp.a.format == super::matrix_format::COLWISE,
            start: lp.a.start.get(),
            index: lp.a.index.get(),
            value: lp.a.value.get(),
            sense: lp.sense as f64,
            offset: lp.offset,
            integrality: lp.integrality.get(),
        }
    }
    /// HighsLp::isMip
    pub fn is_mip(&self) -> bool {
        self.integrality[..self.num_col.min(self.integrality.len())].iter().any(|&t| t != var_type::CONTINUOUS)
    }
    /// HighsLp::objectiveValue
    pub fn objective_value(&self, x: &[f64]) -> f64 {
        dot_blocked(&self.col_cost[..self.num_col], &x[..self.num_col], self.offset, 4)
    }
    /// HighsSparseMatrix::productQuad
    pub fn product_quad(&self, x: &[f64]) -> Vec<f64> {
        let mut v = vec![CDouble::from(0.0); self.num_row];
        if self.colwise {
            for c in 0..self.num_col {
                for k in self.start[c] as usize..self.start[c + 1] as usize {
                    v[self.index[k] as usize] += x[c] * self.value[k];
                }
            }
        } else {
            for (r, vr) in v.iter_mut().enumerate() {
                for k in self.start[r] as usize..self.start[r + 1] as usize {
                    *vr += x[self.index[k] as usize] * self.value[k];
                }
            }
        }
        v.into_iter().map(f64::from).collect()
    }
    /// HighsSparseMatrix::productTransposeQuad
    pub fn product_transpose_quad(&self, y: &[f64]) -> Vec<f64> {
        let mut v = vec![CDouble::from(0.0); self.num_col];
        if self.colwise {
            for (c, vc) in v.iter_mut().enumerate() {
                for k in self.start[c] as usize..self.start[c + 1] as usize {
                    *vc += y[self.index[k] as usize] * self.value[k];
                }
            }
        } else {
            for r in 0..self.num_row {
                for k in self.start[r] as usize..self.start[r + 1] as usize {
                    v[self.index[k] as usize] += y[r] * self.value[k];
                }
            }
        }
        v.into_iter().map(f64::from).collect()
    }
    /// (lower, upper) of variable i (columns then rows)
    #[inline]
    fn bounds(&self, i: usize) -> (f64, f64) {
        if i < self.num_col {
            (self.col_lower[i], self.col_upper[i])
        } else {
            (self.row_lower[i - self.num_col], self.row_upper[i - self.num_col])
        }
    }
}

/// Borrowed solution
pub struct SolRef<'a> {
    pub value_valid: bool,
    pub dual_valid: bool,
    pub col_value: &'a [f64],
    pub col_dual: &'a [f64],
    pub row_value: &'a [f64],
    pub row_dual: &'a [f64],
}

impl SolRef<'_> {
    /// # Safety
    /// The view's arrays must be valid while the reference lives
    pub unsafe fn new<'a>(s: &CSolution) -> SolRef<'a> {
        SolRef {
            value_valid: s.value_valid,
            dual_valid: s.dual_valid,
            col_value: s.col_value.get(),
            col_dual: s.col_dual.get(),
            row_value: s.row_value.get(),
            row_dual: s.row_dual.get(),
        }
    }
    #[inline]
    fn value(&self, num_col: usize, i: usize) -> f64 {
        if i < num_col {
            self.col_value[i]
        } else {
            self.row_value[i - num_col]
        }
    }
    #[inline]
    fn dual(&self, num_col: usize, i: usize) -> f64 {
        if i < num_col {
            self.col_dual[i]
        } else {
            self.row_dual[i - num_col]
        }
    }
}

/// infeasibility() of HighsUtils.h: (infeasibility, residual)
#[inline]
pub fn infeasibility(lower: f64, value: f64, upper: f64, tolerance: f64) -> (f64, f64) {
    let mut residual = 0.0;
    let mut infeas = 0.0;
    if value < lower - tolerance {
        infeas = lower - value;
    }
    if value > upper + tolerance {
        infeas = value - upper;
    }
    if tolerance > 0.0 {
        if value < lower {
            residual = lower - value;
        }
        if value > upper {
            residual = value - upper;
        }
    } else {
        residual = infeas;
    }
    if infeas == 0.0 {
        residual = if tolerance < residual { tolerance } else { residual };
    }
    (infeas, residual)
}

/// The results of getVariableKktFailures
pub struct VarKkt {
    pub primal_infeasibility: f64,
    pub dual_infeasibility: f64,
    pub semi_infeasibility: f64,
    pub at_status: u8,
    pub mid_status: u8,
}

/// getVariableKktFailures (dual_infeasibility keeps `prev_dual_infeas`
/// where the C++ leaves its reference unassigned: never, here)
#[allow(clippy::too_many_arguments)]
pub fn get_variable_kkt_failures(
    primal_feasibility_tolerance: f64,
    mip_feasibility_tolerance: f64,
    lower: f64,
    upper: f64,
    value: f64,
    dual: f64,
    integrality: u8,
) -> VarKkt {
    let (_, residual) = infeasibility(lower, value, upper, primal_feasibility_tolerance);
    let mut primal_infeasibility = residual;
    let mut semi_infeasibility = 0.0;
    let mut at_status = SOL_NO;
    let mut bound_residual = (lower - value).abs();
    if bound_residual * bound_residual <= primal_feasibility_tolerance {
        at_status = SOL_LO;
    } else {
        bound_residual = (value - upper).abs();
        if bound_residual * bound_residual <= primal_feasibility_tolerance {
            at_status = SOL_UP;
        }
    }
    let mut mid_status = SOL_NO;
    let dual_infeasibility;
    if lower < upper {
        let length = upper - lower;
        if lower <= -INF && upper >= INF {
            dual_infeasibility = dual.abs();
        } else if length * length > primal_feasibility_tolerance {
            let middle = (lower + upper) * 0.5;
            if value < middle {
                mid_status = SOL_LO;
                dual_infeasibility = cmax(-dual, 0.0);
            } else {
                mid_status = SOL_UP;
                dual_infeasibility = cmax(dual, 0.0);
            }
        } else {
            dual_infeasibility = 0.0;
        }
    } else {
        dual_infeasibility = 0.0;
    }
    let semi_variable = integrality == var_type::SEMI_CONTINUOUS || integrality == var_type::SEMI_INTEGER;
    let mid = cmax(primal_feasibility_tolerance, lower * 0.5);
    if semi_variable {
        primal_infeasibility = 0.0;
        semi_infeasibility = 0.0;
        if value < mid {
            semi_infeasibility = value.abs();
            if semi_infeasibility < mip_feasibility_tolerance {
                semi_infeasibility = 0.0;
            }
        } else {
            if value < lower {
                semi_infeasibility = lower - value;
            } else if value > upper {
                semi_infeasibility = value - upper;
            }
            if semi_infeasibility < primal_feasibility_tolerance {
                semi_infeasibility = 0.0;
            }
        }
    }
    VarKkt { primal_infeasibility, dual_infeasibility, semi_infeasibility, at_status, mid_status }
}

/// The tolerances of getKktFailures (kkt_tolerance overrides)
struct Tols {
    primal_feasibility: f64,
    dual_feasibility: f64,
    mip_feasibility: f64,
    primal_residual: f64,
    dual_residual: f64,
    optimality: f64,
}

fn tols(o: &CKktOptions) -> Tols {
    let mut t = Tols {
        primal_feasibility: o.primal_feasibility_tolerance,
        dual_feasibility: o.dual_feasibility_tolerance,
        mip_feasibility: o.mip_feasibility_tolerance,
        primal_residual: o.primal_residual_tolerance,
        dual_residual: o.dual_residual_tolerance,
        optimality: o.optimality_tolerance,
    };
    if o.kkt_tolerance != DEFAULT_KKT_TOLERANCE {
        t.primal_feasibility = o.kkt_tolerance;
        t.dual_feasibility = o.kkt_tolerance;
        t.primal_residual = o.kkt_tolerance;
        t.dual_residual = o.kkt_tolerance;
        t.optimality = o.kkt_tolerance;
    }
    t
}

/// getKktFailures(options, is_qp, lp, gradient, solution, info,
/// get_residuals)
pub fn get_kkt_failures(
    o: &CKktOptions,
    is_qp: bool,
    lp: &LpRef,
    gradient: &[f64],
    sol: &SolRef,
    info: &mut Info,
    get_residuals: bool,
) {
    let t = tols(o);
    let have_primal = sol.value_valid;
    let have_dual = sol.dual_valid;
    let have_integrality = !lp.integrality.is_empty();
    info.primal_solution_status = SOLUTION_STATUS_NONE;
    info.dual_solution_status = SOLUTION_STATUS_NONE;
    info.invalidate_kkt();
    if have_primal {
        info.num_primal_infeasibilities = 0;
        info.max_primal_infeasibility = 0.0;
        info.sum_primal_infeasibilities = 0.0;
        info.num_semi_infeasibilities = 0;
        info.max_semi_infeasibility = 0.0;
        info.sum_semi_infeasibilities = 0.0;
        info.num_relative_primal_infeasibilities = 0;
        info.max_relative_primal_infeasibility = 0.0;
        if get_residuals {
            info.num_primal_residual_errors = 0;
            info.max_primal_residual_error = 0.0;
            info.num_relative_primal_residual_errors = 0;
            info.max_relative_primal_residual_error = 0.0;
        }
    }
    if have_dual {
        info.num_dual_infeasibilities = 0;
        info.max_dual_infeasibility = 0.0;
        info.sum_dual_infeasibilities = 0.0;
        info.num_relative_dual_infeasibilities = 0;
        info.max_relative_dual_infeasibility = 0.0;
        if get_residuals {
            info.num_dual_residual_errors = 0;
            info.max_dual_residual_error = 0.0;
            info.num_relative_dual_residual_errors = 0;
            info.max_relative_dual_residual_error = 0.0;
        }
    }
    if !have_primal {
        return;
    }
    let primal_activity = if get_residuals { lp.product_quad(sol.col_value) } else { Vec::new() };
    let mut dual_activity =
        if get_residuals && have_dual { lp.product_transpose_quad(sol.row_dual) } else { Vec::new() };

    let mut max_col_primal_infeasibility = 0.0;
    let mut max_col_dual_infeasibility = 0.0;
    let mut max_relative_col_primal_infeasibility = 0.0;
    let mut max_relative_col_dual_infeasibility = 0.0;
    let mut cost = 0.0;
    let mut dual = 0.0;
    let mut integrality = var_type::CONTINUOUS;
    let mut highs_norm_bounds = 0.0;
    let mut highs_norm_costs = 0.0;
    let num_col = lp.num_col;
    let num_var = num_col + lp.num_row;
    let visit_all_vars = get_residuals;
    let mut infeasible_var: Vec<usize> = Vec::new();
    for pass in 0..2 {
        let num_pass_var = if pass == 0 || visit_all_vars { num_var } else { infeasible_var.len() };
        let mut have_col_maxima = false;
        macro_rules! save_col_maxima {
            () => {
                have_col_maxima = true;
                max_col_primal_infeasibility = info.max_primal_infeasibility;
                max_col_dual_infeasibility = info.max_dual_infeasibility;
                max_relative_col_primal_infeasibility = info.max_relative_primal_infeasibility;
                max_relative_col_dual_infeasibility = info.max_relative_dual_infeasibility;
                info.max_primal_infeasibility = 0.0;
                info.max_dual_infeasibility = 0.0;
                info.max_relative_primal_infeasibility = 0.0;
                info.max_relative_dual_infeasibility = 0.0;
            };
        }
        for k in 0..num_pass_var {
            let var = if pass == 0 || visit_all_vars { k } else { infeasible_var[k] };
            let is_col = var < num_col;
            if pass == 1 && !is_col && !have_col_maxima {
                save_col_maxima!();
            }
            let (lower, upper) = lp.bounds(var);
            let value = sol.value(num_col, var);
            if is_col {
                cost = gradient[var];
                if have_dual {
                    dual = sol.col_dual[var];
                }
                if have_integrality {
                    integrality = lp.integrality[var];
                }
                if pass == 0 {
                    if dual * dual < t.dual_feasibility {
                        highs_norm_costs = cmax(cost.abs(), highs_norm_costs);
                    }
                    if get_residuals && have_dual {
                        let mut q = CDouble::from(dual_activity[var]);
                        q -= gradient[var];
                        dual_activity[var] = f64::from(q);
                    }
                }
            } else {
                if have_dual {
                    dual = sol.row_dual[var - num_col];
                }
                integrality = var_type::CONTINUOUS;
            }
            dual *= lp.sense;
            let v = get_variable_kkt_failures(
                t.primal_feasibility,
                t.mip_feasibility,
                lower,
                upper,
                value,
                dual,
                integrality,
            );
            if pass == 0 {
                if v.at_status == SOL_LO {
                    highs_norm_bounds = cmax(lower.abs(), highs_norm_bounds);
                } else if v.at_status == SOL_UP {
                    highs_norm_bounds = cmax(upper.abs(), highs_norm_bounds);
                }
                if !visit_all_vars
                    && (v.primal_infeasibility > 0.0
                        || v.semi_infeasibility > 0.0
                        || (have_dual && v.dual_infeasibility > 0.0))
                {
                    infeasible_var.push(var);
                }
                continue;
            }
            if v.primal_infeasibility > 0.0 {
                let p = v.primal_infeasibility;
                if p > t.primal_feasibility {
                    info.num_primal_infeasibilities += 1;
                }
                if info.max_primal_infeasibility < p {
                    info.max_primal_infeasibility = p;
                }
                info.sum_primal_infeasibilities += p;
                let mut relative_bound_measure = highs_norm_bounds;
                if v.at_status == SOL_NO {
                    if v.mid_status == SOL_NO || v.mid_status == SOL_LO {
                        relative_bound_measure = cmax(lower.abs(), relative_bound_measure);
                    } else {
                        relative_bound_measure = cmax(upper.abs(), relative_bound_measure);
                    }
                }
                let rel = p / (1.0 + relative_bound_measure);
                if rel > t.primal_feasibility {
                    info.num_relative_primal_infeasibilities += 1;
                }
                if info.max_relative_primal_infeasibility < rel {
                    info.max_relative_primal_infeasibility = rel;
                }
            }
            if v.semi_infeasibility > 0.0 {
                let s = v.semi_infeasibility;
                info.num_semi_infeasibilities += 1;
                if info.max_semi_infeasibility < s {
                    info.max_semi_infeasibility = s;
                }
                info.sum_semi_infeasibilities += s;
            }
            if have_dual && v.dual_infeasibility > 0.0 {
                let d = v.dual_infeasibility;
                if d > t.dual_feasibility {
                    info.num_dual_infeasibilities += 1;
                }
                if info.max_dual_infeasibility < d {
                    info.max_dual_infeasibility = d;
                }
                info.sum_dual_infeasibilities += d;
                let mut relative_cost_measure = highs_norm_costs;
                if is_col && cost != 0.0 && dual * dual >= t.dual_feasibility {
                    relative_cost_measure = cmax(cost.abs(), relative_cost_measure);
                }
                let rel = d / (1.0 + relative_cost_measure);
                if rel > t.dual_feasibility {
                    info.num_relative_dual_infeasibilities += 1;
                }
                if info.max_relative_dual_infeasibility < rel {
                    info.max_relative_dual_infeasibility = rel;
                }
            }
            if !is_col && get_residuals {
                let r = var - num_col;
                let err = (primal_activity[r] - sol.row_value[r]).abs();
                let rel = err / (1.0 + highs_norm_bounds);
                if err > t.primal_residual {
                    info.num_primal_residual_errors += 1;
                }
                if info.max_primal_residual_error < err {
                    info.max_primal_residual_error = err;
                }
                if rel > t.primal_residual {
                    info.num_relative_primal_residual_errors += 1;
                }
                if info.max_relative_primal_residual_error < rel {
                    info.max_relative_primal_residual_error = rel;
                }
            }
            if is_col && get_residuals && have_dual {
                let err = (dual_activity[var] + sol.col_dual[var]).abs();
                let rel = err / (1.0 + highs_norm_costs);
                if err > t.dual_residual {
                    info.num_dual_residual_errors += 1;
                }
                if info.max_dual_residual_error < err {
                    info.max_dual_residual_error = err;
                }
                if rel > t.dual_residual {
                    info.num_relative_dual_residual_errors += 1;
                }
                if info.max_relative_dual_residual_error < rel {
                    info.max_relative_dual_residual_error = rel;
                }
            }
        }
        if pass == 1 && !have_col_maxima {
            save_col_maxima!();
        }
        let _ = have_col_maxima;
    }
    let max_row_primal_infeasibility = info.max_primal_infeasibility;
    let max_row_dual_infeasibility = info.max_dual_infeasibility;
    let max_relative_row_primal_infeasibility = info.max_relative_primal_infeasibility;
    let max_relative_row_dual_infeasibility = info.max_relative_dual_infeasibility;
    info.max_primal_infeasibility = cmax(max_col_primal_infeasibility, max_row_primal_infeasibility);
    info.max_dual_infeasibility = cmax(max_col_dual_infeasibility, max_row_dual_infeasibility);
    info.max_relative_primal_infeasibility =
        cmax(max_relative_col_primal_infeasibility, max_relative_row_primal_infeasibility);
    info.max_relative_dual_infeasibility =
        cmax(max_relative_col_dual_infeasibility, max_relative_row_dual_infeasibility);
    if have_dual {
        let (n, m) = get_complementarity_violations(lp, sol, t.optimality);
        info.num_complementarity_violations = n;
        info.max_complementarity_violation = m;
        let dual_objective_value = compute_dual_objective_value(if is_qp { Some(gradient) } else { None }, lp, sol);
        let abs_objective_difference = (info.objective_function_value - dual_objective_value).abs();
        let denominator = 1.0 + info.objective_function_value.abs() + dual_objective_value.abs();
        info.primal_dual_objective_error = abs_objective_difference / denominator;
    }
    if o.log_dev_level > 0 {
        let log = &o.log;
        log_dev!(
            log,
            LogType::Info,
            "getKktFailures:: cost norm = %8.3g; bound norm = %8.3g\n",
            highs_norm_costs,
            highs_norm_bounds
        );
        log_dev!(
            log,
            LogType::Info,
            "getKktFailures:                      LP  (abs / rel)         Col (abs / rel)         Row (abs / rel)\n"
        );
        log_dev!(
            log,
            LogType::Info,
            "getKktFailures: primal infeasibility %8.3g / %8.3g     %8.3g / %8.3g     %8.3g / %8.3g\n",
            info.max_primal_infeasibility,
            info.max_relative_primal_infeasibility,
            max_col_primal_infeasibility,
            max_relative_col_primal_infeasibility,
            max_row_primal_infeasibility,
            max_relative_row_primal_infeasibility
        );
        if have_dual {
            log_dev!(
                log,
                LogType::Info,
                "getKktFailures:   dual infeasibility %8.3g / %8.3g     %8.3g / %8.3g     %8.3g / %8.3g\n",
                info.max_dual_infeasibility,
                info.max_relative_dual_infeasibility,
                max_col_dual_infeasibility,
                max_relative_col_dual_infeasibility,
                max_row_dual_infeasibility,
                max_relative_row_dual_infeasibility
            );
        }
        if get_residuals {
            log_dev!(
                log,
                LogType::Info,
                "getKktFailures: primal residual      %8.3g / %8.3g\n",
                info.max_primal_residual_error,
                info.max_relative_primal_residual_error
            );
            if have_dual {
                log_dev!(
                    log,
                    LogType::Info,
                    "getKktFailures:   dual residual      %8.3g / %8.3g\n",
                    info.max_dual_residual_error,
                    info.max_relative_dual_residual_error
                );
            }
        }
        if !is_qp && have_dual {
            log_dev!(
                log,
                LogType::Info,
                "getKktFailures: objective gap        %8.3g\n",
                info.primal_dual_objective_error
            );
        }
    }
    info.primal_solution_status = if info.num_primal_infeasibilities + info.num_semi_infeasibilities != 0 {
        SOLUTION_STATUS_INFEASIBLE
    } else {
        SOLUTION_STATUS_FEASIBLE
    };
    if have_dual {
        info.dual_solution_status =
            if info.num_dual_infeasibilities != 0 { SOLUTION_STATUS_INFEASIBLE } else { SOLUTION_STATUS_FEASIBLE };
    }
}

/// getComplementarityViolations (with a dual solution): (num, max)
pub fn get_complementarity_violations(lp: &LpRef, sol: &SolRef, optimality_tolerance: f64) -> (i32, f64) {
    let mut num = 0;
    let mut max = 0.0;
    let n = lp.num_col;
    for i in 0..n + lp.num_row {
        let primal = sol.value(n, i);
        let dual = sol.dual(n, i);
        let (lower, upper) = lp.bounds(i);
        let primal_residual = if lower <= -INF && upper >= INF {
            1.0
        } else {
            let mid = (lower + upper) * 0.5;
            if primal < mid {
                (lower - primal).abs()
            } else {
                (upper - primal).abs()
            }
        };
        let v = primal_residual * dual.abs();
        if v > optimality_tolerance {
            num += 1;
        }
        max = cmax(v, max);
    }
    (num, max)
}

/// computeDualObjectiveValue (with a dual solution)
pub fn compute_dual_objective_value(gradient: Option<&[f64]>, lp: &LpRef, sol: &SolRef) -> f64 {
    let mut d = lp.offset;
    let n = lp.num_col;
    if let Some(g) = gradient {
        // Vectorized by 8 in the C++, the tail fused
        let nb = if n >= 8 { n - n % 8 } else { 0 };
        let mut q = 0.0;
        for i in 0..nb {
            q += (lp.col_cost[i] - g[i]) * sol.col_value[i];
        }
        for i in nb..n {
            q = (lp.col_cost[i] - g[i]).mul_add_c(sol.col_value[i], q);
        }
        d = 0.5f64.mul_add_c(q, d);
    }
    for i in 0..n + lp.num_row {
        let primal = sol.value(n, i);
        let dual = sol.dual(n, i);
        let (lower, upper) = lp.bounds(i);
        let bound = if lower <= -INF && upper >= INF {
            1.0
        } else {
            let mid = (lower + upper) * 0.5;
            if primal < mid {
                lower
            } else {
                upper
            }
        };
        d = bound.mul_add_c(dual, d);
    }
    d
}

/// computeObjectiveValue
pub fn compute_objective_value(lp: &LpRef, sol: &SolRef) -> f64 {
    dot_blocked(&lp.col_cost[..lp.num_col], &sol.col_value[..lp.num_col], 0.0, 4) + lp.offset
}

/// getPrimalDualBasisErrors
pub fn get_primal_dual_basis_errors(
    o: &CKktOptions,
    lp: &LpRef,
    sol: &SolRef,
    basis_valid: bool,
    col_status: &[u8],
    row_status: &[u8],
    e: &mut PrimalDualErrors,
) {
    let tol = o.primal_feasibility_tolerance;
    if basis_valid {
        e.num_nonzero_basic_duals = 0;
        e.max_nonzero_basic_dual = 0.0;
        e.sum_nonzero_basic_duals = 0.0;
        e.num_off_bound_nonbasic = 0;
        e.max_off_bound_nonbasic = 0.0;
        e.sum_off_bound_nonbasic = 0.0;
    } else {
        e.num_nonzero_basic_duals = ILLEGAL_COUNT;
        e.max_nonzero_basic_dual = ILLEGAL_MEASURE;
        e.sum_nonzero_basic_duals = ILLEGAL_MEASURE;
        e.num_off_bound_nonbasic = ILLEGAL_COUNT;
        e.max_off_bound_nonbasic = ILLEGAL_MEASURE;
        e.sum_off_bound_nonbasic = ILLEGAL_MEASURE;
    }
    if !sol.value_valid || !basis_valid {
        return;
    }
    let n = lp.num_col;
    for i in 0..n + lp.num_row {
        let (lower, upper) = lp.bounds(i);
        let value = sol.value(n, i);
        let mut dual = sol.dual(n, i);
        let status = if i < n { col_status[i] } else { row_status[i - n] };
        let a = (lower - value).abs();
        let b = (value - upper).abs();
        let value_residual = if b < a { b } else { a };
        dual *= lp.sense;
        let mut status_value_ok = true;
        if status == BASIS_LOWER {
            if lower.abs() / tol < 1e-16 {
                status_value_ok = value >= lower - tol && value <= lower + tol;
            }
        } else if status == BASIS_UPPER && upper.abs() / tol < 1e-16 {
            status_value_ok = value >= upper - tol && value <= upper + tol;
        }
        if !status_value_ok {
            log_user!(
                o.log,
                LogType::Error,
                "getPrimalDualBasisErrors: %s %d status-value error: [%23.18g; %23.18g; %23.18g] has residual %g\n",
                if i < n { "Column" } else { "Row   " },
                if i < n { i as i32 } else { (i - n) as i32 },
                lower,
                value,
                upper,
                value_residual
            );
        }
        if status == BASIS_BASIC {
            let abs_basic_dual = dual.abs();
            if abs_basic_dual > 0.0 {
                e.num_nonzero_basic_duals += 1;
                e.max_nonzero_basic_dual = cmax(abs_basic_dual, e.max_nonzero_basic_dual);
                e.sum_nonzero_basic_duals += abs_basic_dual;
            }
        } else {
            let off = value_residual;
            if off > 0.0 {
                e.num_off_bound_nonbasic += 1;
            }
            e.max_off_bound_nonbasic = cmax(off, e.max_off_bound_nonbasic);
            e.sum_off_bound_nonbasic += off;
        }
    }
}

/// getPrimalDualGlpsolErrors (the C++ also forms residual sums that it
/// never uses)
pub fn get_primal_dual_glpsol_errors(o: &CKktOptions, lp: &LpRef, sol: &SolRef, e: &mut PrimalDualErrors) {
    e.glpsol_max_primal_infeasibility.invalidate();
    e.glpsol_max_dual_infeasibility.invalidate();
    let have_primal = sol.value_valid;
    let have_dual = sol.dual_valid;
    let have_integrality = !lp.integrality.is_empty();
    if have_primal {
        e.glpsol_max_primal_infeasibility.absolute_value = 0.0;
        e.glpsol_max_primal_infeasibility.reset();
        if have_dual {
            e.glpsol_max_dual_infeasibility.absolute_value = 0.0;
            e.glpsol_max_dual_infeasibility.reset();
        }
    }
    if have_primal {
        e.glpsol_num_primal_residual_errors = 0;
        e.glpsol_max_primal_residual.reset();
    } else {
        e.glpsol_num_primal_residual_errors = ILLEGAL_COUNT;
        e.glpsol_max_primal_residual.invalidate();
    }
    if have_dual {
        e.glpsol_num_dual_residual_errors = 0;
        e.glpsol_max_dual_residual.reset();
    } else {
        e.glpsol_num_dual_residual_errors = ILLEGAL_COUNT;
        e.glpsol_max_dual_residual.invalidate();
    }
    if !have_primal {
        return;
    }
    let tol = o.primal_feasibility_tolerance;
    let n = lp.num_col;
    let mut dual = 0.0;
    let mut integrality = var_type::CONTINUOUS;
    for i in 0..n + lp.num_row {
        let (lower, upper) = lp.bounds(i);
        let value = sol.value(n, i);
        if have_dual {
            dual = sol.dual(n, i);
        }
        if i < n {
            if have_integrality {
                integrality = lp.integrality[i];
            }
        } else {
            integrality = var_type::CONTINUOUS;
        }
        dual *= lp.sense;
        let v = get_variable_kkt_failures(tol, o.mip_feasibility_tolerance, lower, upper, value, dual, integrality);
        let p = cmax(v.primal_infeasibility, v.semi_infeasibility);
        let mut rel = 0.0;
        if v.mid_status == SOL_LO {
            rel = p / (1.0 + lower.abs());
        } else if v.mid_status == SOL_UP {
            rel = p / (1.0 + upper.abs());
        } else if lower > -INF {
            rel = p / (1.0 + lower.abs());
        }
        let pi = &mut e.glpsol_max_primal_infeasibility;
        if pi.absolute_value < p {
            pi.absolute_value = p;
            pi.absolute_index = i as i32;
        }
        if pi.relative_value < rel {
            pi.relative_value = rel;
            pi.relative_index = i as i32;
        }
        if have_dual {
            let di = &mut e.glpsol_max_dual_infeasibility;
            if di.absolute_value < v.dual_infeasibility {
                di.absolute_value = v.dual_infeasibility;
                di.absolute_index = i as i32;
            }
        }
    }
    let di = &mut e.glpsol_max_dual_infeasibility;
    di.relative_value = di.absolute_value;
    di.relative_index = di.absolute_index;
}

/// lpBasicInfeasibilities
fn lp_basic_infeasibilities(model_status: &mut i32, info: &mut Info, lp: &LpRef, sol: &SolRef, o: &CKktOptions) {
    let mut primal_tol = o.primal_feasibility_tolerance;
    let mut dual_tol = o.dual_feasibility_tolerance;
    if o.kkt_tolerance != DEFAULT_KKT_TOLERANCE {
        primal_tol = o.kkt_tolerance;
        dual_tol = o.kkt_tolerance;
    }
    info.objective_function_value = lp.objective_value(sol.col_value);
    info.invalidate_kkt();
    let have_dual = sol.dual_valid;
    let have_integrality = !lp.integrality.is_empty();
    let (mut num_primal, mut num_dual, mut num_semi) = (0, 0, 0);
    let (mut max_primal, mut sum_primal, mut max_dual, mut sum_dual) = (0.0, 0.0, 0.0, 0.0);
    let (mut max_semi, mut sum_semi) = (0.0, 0.0);
    let n = lp.num_col;
    for i in 0..n + lp.num_row {
        let is_col = i < n;
        let (lower, upper) = lp.bounds(i);
        let value = sol.value(n, i);
        let mut dual = 0.0;
        if have_dual {
            dual = sol.dual(n, i);
        }
        dual *= lp.sense;
        let integrality = if is_col && have_integrality { lp.integrality[i] } else { var_type::CONTINUOUS };
        let v = get_variable_kkt_failures(primal_tol, o.mip_feasibility_tolerance, lower, upper, value, dual, integrality);
        if v.primal_infeasibility > 0.0 {
            if v.primal_infeasibility > primal_tol {
                num_primal += 1;
            }
            max_primal = cmax(v.primal_infeasibility, max_primal);
            sum_primal += v.primal_infeasibility;
        }
        if v.semi_infeasibility > 0.0 {
            num_semi += 1;
            max_semi = cmax(v.semi_infeasibility, max_semi);
            sum_semi += v.semi_infeasibility;
        }
        if have_dual && v.dual_infeasibility > 0.0 {
            if v.dual_infeasibility > dual_tol {
                num_dual += 1;
            }
            max_dual = cmax(v.dual_infeasibility, max_dual);
            sum_dual += v.dual_infeasibility;
        }
    }
    info.num_primal_infeasibilities = num_primal;
    info.max_primal_infeasibility = max_primal;
    info.sum_primal_infeasibilities = sum_primal;
    info.num_semi_infeasibilities = num_semi;
    info.max_semi_infeasibility = max_semi;
    info.sum_semi_infeasibilities = sum_semi;
    info.primal_solution_status = if num_primal != 0 { SOLUTION_STATUS_INFEASIBLE } else { SOLUTION_STATUS_FEASIBLE };
    info.dual_solution_status = SOLUTION_STATUS_NONE;
    if have_dual {
        info.num_dual_infeasibilities = num_dual;
        info.max_dual_infeasibility = max_dual;
        info.sum_dual_infeasibilities = sum_dual;
        info.dual_solution_status = if num_dual != 0 { SOLUTION_STATUS_INFEASIBLE } else { SOLUTION_STATUS_FEASIBLE };
    }
    if *model_status == MODEL_STATUS_UNBOUNDED_OR_INFEASIBLE && num_primal == 0 {
        *model_status = MODEL_STATUS_UNBOUNDED;
    }
}

/// lpKktCheck
pub fn lp_kkt_check(
    model_status: &mut i32,
    info: &mut Info,
    lp: &LpRef,
    sol: &SolRef,
    basis_valid: bool,
    o: &CKktOptions,
    message: &str,
) {
    if !sol.value_valid {
        return;
    }
    if basis_valid && !o.full_lp_kkt_check {
        lp_basic_infeasibilities(model_status, info, lp, sol, o);
        return;
    }
    let log = &o.log;
    let t = tols(o);
    info.objective_function_value = lp.objective_value(sol.col_value);
    let get_residuals = !basis_valid;
    get_kkt_failures(o, false, lp, lp.col_cost, sol, info, get_residuals);
    if *model_status == MODEL_STATUS_OPTIMAL {
        report_kkt_failures(lp, o, info, message);
    }
    if *model_status == MODEL_STATUS_UNBOUNDED_OR_INFEASIBLE
        && info.num_primal_infeasibilities == 0
        && (!get_residuals || info.num_primal_residual_errors == 0)
    {
        *model_status = MODEL_STATUS_UNBOUNDED;
    }
    let was_optimal = *model_status == MODEL_STATUS_OPTIMAL;
    let mut written_header = false;
    let mut found_error = || {
        if !was_optimal || written_header {
            return;
        }
        log_user!(log, LogType::Warning, "LP solver claims optimality, but with\n");
        written_header = true;
    };
    let mut max_primal_trv = 0.0;
    let mut max_dual_trv = 0.0;
    let mut pd_objective_trv = 0.0;
    let max_allowed = 1e2;
    if basis_valid {
        if info.num_primal_infeasibilities > 0 {
            max_primal_trv = cmax(info.max_primal_infeasibility / t.primal_feasibility, max_primal_trv);
            found_error();
            if was_optimal {
                log_user!(
                    log,
                    LogType::Warning,
                    "   num/max/sum %6d / %8.3g / %8.3g primal infeasibilities       (tolerance = %4.0e)\n",
                    info.num_primal_infeasibilities,
                    info.max_primal_infeasibility,
                    info.sum_primal_infeasibilities,
                    t.primal_feasibility
                );
            }
        }
        if info.num_dual_infeasibilities > 0 {
            max_dual_trv = cmax(info.max_dual_infeasibility / t.dual_feasibility, max_dual_trv);
            found_error();
            if was_optimal {
                log_user!(
                    log,
                    LogType::Warning,
                    "   num/max/sum %6d / %8.3g / %8.3g   dual infeasibilities       (tolerance = %4.0e)\n",
                    info.num_dual_infeasibilities,
                    info.max_dual_infeasibility,
                    info.sum_dual_infeasibilities,
                    t.dual_feasibility
                );
            }
        }
        let mut unexpected = info.num_complementarity_violations != 0;
        let mut local_dual_objective = 0.0;
        if info.primal_dual_objective_error > t.optimality {
            if sol.dual_valid {
                local_dual_objective = compute_dual_objective_value(None, lp, sol);
            }
            if info.objective_function_value * info.objective_function_value > t.optimality
                && local_dual_objective * local_dual_objective > t.optimality
            {
                unexpected = true;
            }
        }
        let have_residual_errors = info.num_primal_residual_errors != ILLEGAL_COUNT;
        if have_residual_errors {
            unexpected = unexpected
                || info.num_relative_primal_residual_errors != 0
                || info.num_relative_dual_residual_errors != 0;
            max_primal_trv = cmax(info.max_relative_primal_residual_error / t.primal_residual, max_primal_trv);
            max_dual_trv = cmax(info.max_relative_dual_residual_error / t.dual_residual, max_dual_trv);
        }
        pd_objective_trv = info.primal_dual_objective_error / t.optimality;
        if was_optimal && unexpected {
            log_user!(
                log,
                LogType::Warning,
                "Optimal basic solution has %d complementarity violations and %g primal dual objective error from primal (dual) objective = %g (%g)\n",
                info.num_complementarity_violations,
                info.primal_dual_objective_error,
                info.objective_function_value,
                local_dual_objective
            );
            if have_residual_errors {
                log_user!(
                    log,
                    LogType::Warning,
                    "   num/max %6d / %8.3g  relative primal residual errors         (tolerance = %4.0e)\n",
                    info.num_relative_primal_residual_errors,
                    info.max_relative_primal_residual_error,
                    t.primal_residual
                );
                log_user!(
                    log,
                    LogType::Warning,
                    "   num/max %6d / %8.3g  relative   dual residual errors         (tolerance = %4.0e)\n",
                    info.num_relative_dual_residual_errors,
                    info.max_relative_dual_residual_error,
                    t.dual_residual
                );
            }
        }
        if info.num_primal_infeasibilities == 0 {
            info.primal_solution_status = SOLUTION_STATUS_FEASIBLE;
        }
        if info.num_dual_infeasibilities == 0 {
            info.dual_solution_status = SOLUTION_STATUS_FEASIBLE;
        }
        if max_primal_trv > max_allowed {
            info.primal_solution_status = SOLUTION_STATUS_INFEASIBLE;
        }
        if max_dual_trv > max_allowed {
            info.dual_solution_status = SOLUTION_STATUS_INFEASIBLE;
        }
    } else {
        let mut trv = info.max_relative_primal_infeasibility / t.primal_feasibility;
        max_primal_trv = cmax(trv, max_primal_trv);
        if info.num_relative_primal_infeasibilities > 0 {
            found_error();
            if was_optimal {
                log_user!(
                    log,
                    LogType::Warning,
                    "   num/max %6d / %8.3g relative primal infeasibilities (tolerance = %4.0e)\n",
                    info.num_relative_primal_infeasibilities,
                    info.max_relative_primal_infeasibility,
                    t.primal_feasibility
                );
            }
        }
        trv = info.max_relative_dual_infeasibility / t.dual_feasibility;
        max_dual_trv = cmax(trv, max_dual_trv);
        if info.num_relative_dual_infeasibilities > 0 {
            found_error();
            if was_optimal {
                log_user!(
                    log,
                    LogType::Warning,
                    "   num/max %6d / %8.3g relative   dual infeasibilities (tolerance = %4.0e)\n",
                    info.num_relative_dual_infeasibilities,
                    info.max_relative_dual_infeasibility,
                    t.dual_feasibility
                );
            }
        }
        trv = info.max_relative_primal_residual_error / t.primal_residual;
        max_primal_trv = cmax(trv, max_primal_trv);
        if info.num_relative_primal_residual_errors > 0 {
            found_error();
            if was_optimal {
                log_user!(
                    log,
                    LogType::Warning,
                    "   num/max %6d / %8.3g relative primal residual errors (tolerance = %4.0e)\n",
                    info.num_relative_primal_residual_errors,
                    info.max_relative_primal_residual_error,
                    t.primal_residual
                );
            }
        }
        trv = info.max_relative_dual_residual_error / t.dual_residual;
        max_dual_trv = cmax(trv, max_dual_trv);
        if info.num_relative_dual_residual_errors > 0 {
            found_error();
            if was_optimal {
                log_user!(
                    log,
                    LogType::Warning,
                    "   num/max %6d / %8.3g relative   dual residual errors (tolerance = %4.0e)\n",
                    info.num_relative_dual_residual_errors,
                    info.max_relative_dual_residual_error,
                    t.dual_residual
                );
            }
        }
        if info.primal_dual_objective_error > t.optimality {
            pd_objective_trv = info.primal_dual_objective_error / t.optimality;
            found_error();
            if was_optimal {
                log_user!(
                    log,
                    LogType::Warning,
                    "                    %8.3g relative P-D objective error    (tolerance = %4.0e)\n",
                    info.primal_dual_objective_error,
                    t.optimality
                );
            }
        }
        info.primal_solution_status =
            if max_primal_trv > max_allowed { SOLUTION_STATUS_INFEASIBLE } else { SOLUTION_STATUS_FEASIBLE };
        info.dual_solution_status =
            if max_dual_trv > max_allowed { SOLUTION_STATUS_INFEASIBLE } else { SOLUTION_STATUS_FEASIBLE };
    }
    let mut max_trv = pd_objective_trv;
    max_trv = cmax(max_primal_trv, max_trv);
    max_trv = cmax(max_dual_trv, max_trv);
    if *model_status == MODEL_STATUS_OPTIMAL {
        if max_trv > max_allowed {
            *model_status = MODEL_STATUS_UNKNOWN;
            log_user!(
                log,
                LogType::Warning,
                "Model status changed from \"Optimal\" to \"Unknown\" since relative violation of tolerances is %8.3g\n",
                max_trv
            );
        } else if max_allowed > 1.0 && max_trv > 1.0 {
            log_user!(
                log,
                LogType::Info,
                "Model status is \"Optimal\" since relative violation of tolerances is no more than %8.3g\n",
                max_trv
            );
        }
    } else if *model_status == MODEL_STATUS_UNKNOWN && max_trv <= max_allowed {
        *model_status = MODEL_STATUS_OPTIMAL;
        log_user!(log, LogType::Warning, "Model status changed from \"Unknown\" to \"Optimal\"\n");
    }
    log_user!(log, LogType::Info, "\n");
}

/// reportKktFailures
pub fn report_kkt_failures(lp: &LpRef, o: &CKktOptions, info: &Info, message: &str) -> bool {
    let log = &o.log;
    let mut mip_feasibility_tolerance = o.mip_feasibility_tolerance;
    let mut primal_feasibility_tolerance = o.primal_feasibility_tolerance;
    let mut dual_feasibility_tolerance = o.dual_feasibility_tolerance;
    let mut primal_residual_tolerance = o.primal_residual_tolerance;
    let mut dual_residual_tolerance = o.dual_residual_tolerance;
    let mut optimality_tolerance = o.optimality_tolerance;
    let is_mip = lp.is_mip();
    if is_mip {
        primal_feasibility_tolerance = mip_feasibility_tolerance;
    } else if o.kkt_tolerance != DEFAULT_KKT_TOLERANCE {
        mip_feasibility_tolerance = o.kkt_tolerance;
        primal_feasibility_tolerance = o.kkt_tolerance;
        dual_feasibility_tolerance = o.kkt_tolerance;
        primal_residual_tolerance = o.kkt_tolerance;
        dual_residual_tolerance = o.kkt_tolerance;
        optimality_tolerance = o.kkt_tolerance;
    }
    let force_report = o.log_dev_level >= 1;
    let complementarity_error = !is_mip && info.primal_dual_objective_error > optimality_tolerance;
    let integrality_error = is_mip && info.max_integrality_violation >= mip_feasibility_tolerance;
    let has_kkt_failures = integrality_error
        || info.num_primal_infeasibilities > 0
        || info.num_dual_infeasibilities > 0
        || info.num_primal_residual_errors > 0
        || info.num_dual_residual_errors > 0
        || complementarity_error;
    if !has_kkt_failures && !force_report {
        return has_kkt_failures;
    }
    let log_type = if has_kkt_failures { LogType::Warning } else { LogType::Info };
    log_user!(
        log,
        log_type,
        "Solution optimality conditions%s%s\n",
        if message.is_empty() { "" } else { ": " },
        message
    );
    if is_mip && info.max_integrality_violation >= 0.0 {
        log_user!(
            log,
            LogType::Info,
            "    max      %8.3g                                  integrality violations     (tolerance = %4.0e)\n",
            info.max_integrality_violation,
            mip_feasibility_tolerance
        );
    }
    if info.num_primal_infeasibilities >= 0 {
        log_user!(
            log,
            LogType::Info,
            "num/max %6d / %8.3g (relative %6d / %8.3g) primal infeasibilities     (tolerance = %4.0e)\n",
            info.num_primal_infeasibilities,
            info.max_primal_infeasibility,
            info.num_relative_primal_infeasibilities,
            info.max_relative_primal_infeasibility,
            primal_feasibility_tolerance
        );
    }
    if info.num_dual_infeasibilities >= 0 {
        log_user!(
            log,
            LogType::Info,
            "num/max %6d / %8.3g (relative %6d / %8.3g)   dual infeasibilities     (tolerance = %4.0e)\n",
            info.num_dual_infeasibilities,
            info.max_dual_infeasibility,
            info.num_relative_dual_infeasibilities,
            info.max_relative_dual_infeasibility,
            dual_feasibility_tolerance
        );
    }
    if info.num_primal_residual_errors >= 0 {
        log_user!(
            log,
            LogType::Info,
            "num/max %6d / %8.3g (relative %6d / %8.3g) primal residual errors     (tolerance = %4.0e)\n",
            info.num_primal_residual_errors,
            info.max_primal_residual_error,
            info.num_relative_primal_residual_errors,
            info.max_relative_primal_residual_error,
            primal_residual_tolerance
        );
    }
    if info.num_dual_residual_errors >= 0 {
        log_user!(
            log,
            LogType::Info,
            "num/max %6d / %8.3g (relative %6d / %8.3g)   dual residual errors     (tolerance = %4.0e)\n",
            info.num_dual_residual_errors,
            info.max_dual_residual_error,
            info.num_relative_dual_residual_errors,
            info.max_relative_dual_residual_error,
            dual_residual_tolerance
        );
    }
    if info.primal_dual_objective_error != INF {
        log_user!(
            log,
            LogType::Info,
            "                                         %1d / %8.3g  P-D objective error        (tolerance = %4.0e)\n",
            if info.primal_dual_objective_error > optimality_tolerance { 1 } else { 0 },
            info.primal_dual_objective_error,
            optimality_tolerance
        );
    }
    has_kkt_failures
}

/// The `extern "C"` entry points (highs/lp_data/HighsSolutionRust.cpp)
pub mod ffi {
    use super::*;
    use std::slice::from_raw_parts;

    unsafe fn text<'a>(p: *const u8, n: usize) -> &'a str {
        if n == 0 {
            ""
        } else {
            std::str::from_utf8_unchecked(from_raw_parts(p, n))
        }
    }

    /// getKktFailures(options, is_qp, lp, gradient, solution, info,
    /// get_residuals)
    ///
    /// # Safety
    /// The views must be valid
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_get_kkt_failures(
        o: *const CKktOptions,
        is_qp: bool,
        lp: *const CLp,
        gradient: RsMut<f64>,
        sol: *const CSolution,
        info: *mut Info,
        get_residuals: bool,
    ) {
        get_kkt_failures(&*o, is_qp, &LpRef::new(&*lp), gradient.get(), &SolRef::new(&*sol), &mut *info, get_residuals)
    }

    /// getPrimalDualBasisErrors
    ///
    /// # Safety
    /// The views must be valid
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_get_primal_dual_basis_errors(
        o: *const CKktOptions,
        lp: *const CLp,
        sol: *const CSolution,
        basis: *const CBasis,
        e: *mut PrimalDualErrors,
    ) {
        let b = &*basis;
        get_primal_dual_basis_errors(
            &*o,
            &LpRef::new(&*lp),
            &SolRef::new(&*sol),
            b.valid,
            b.col_status.get(),
            b.row_status.get(),
            &mut *e,
        )
    }

    /// getPrimalDualGlpsolErrors
    ///
    /// # Safety
    /// The views must be valid
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_get_primal_dual_glpsol_errors(
        o: *const CKktOptions,
        lp: *const CLp,
        sol: *const CSolution,
        e: *mut PrimalDualErrors,
    ) {
        get_primal_dual_glpsol_errors(&*o, &LpRef::new(&*lp), &SolRef::new(&*sol), &mut *e)
    }

    /// getComplementarityViolations (with a dual solution)
    ///
    /// # Safety
    /// The views must be valid
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_get_complementarity_violations(
        lp: *const CLp,
        sol: *const CSolution,
        optimality_tolerance: f64,
        num: *mut i32,
        max: *mut f64,
    ) {
        let (n, m) = get_complementarity_violations(&LpRef::new(&*lp), &SolRef::new(&*sol), optimality_tolerance);
        *num = n;
        *max = m;
    }

    /// computeDualObjectiveValue (with a dual solution; an empty gradient
    /// for none)
    ///
    /// # Safety
    /// The views must be valid
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_compute_dual_objective_value(
        gradient: RsMut<f64>,
        lp: *const CLp,
        sol: *const CSolution,
    ) -> f64 {
        let g = if gradient.ptr.is_null() { None } else { Some(gradient.get()) };
        compute_dual_objective_value(g, &LpRef::new(&*lp), &SolRef::new(&*sol))
    }

    /// computeObjectiveValue
    ///
    /// # Safety
    /// The views must be valid
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_compute_objective_value(lp: *const CLp, sol: *const CSolution) -> f64 {
        compute_objective_value(&LpRef::new(&*lp), &SolRef::new(&*sol))
    }

    /// HighsLp::objectiveValue
    ///
    /// # Safety
    /// The views must be valid
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lp_objective_value(lp: *const CLp, x: RsMut<f64>) -> f64 {
        LpRef::new(&*lp).objective_value(x.get())
    }

    /// lpKktCheck
    ///
    /// # Safety
    /// The views must be valid
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lp_kkt_check(
        model_status: *mut i32,
        info: *mut Info,
        lp: *const CLp,
        sol: *const CSolution,
        basis_valid: bool,
        o: *const CKktOptions,
        message: *const u8,
        message_len: usize,
    ) {
        lp_kkt_check(
            &mut *model_status,
            &mut *info,
            &LpRef::new(&*lp),
            &SolRef::new(&*sol),
            basis_valid,
            &*o,
            text(message, message_len),
        )
    }

    /// reportKktFailures
    ///
    /// # Safety
    /// The views must be valid
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_report_kkt_failures(
        lp: *const CLp,
        o: *const CKktOptions,
        info: *const Info,
        message: *const u8,
        message_len: usize,
    ) -> bool {
        report_kkt_failures(&LpRef::new(&*lp), &*o, &*info, text(message, message_len))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    #[test]
    fn layouts() {
        // HighsInfoStruct and HighsPrimalDualErrors as the C++ lays them out
        assert_eq!(offset_of!(Info, mip_node_count), 8);
        assert_eq!(offset_of!(Info, objective_function_value), 48);
        assert_eq!(size_of::<Info>(), 280);
        assert_eq!(size_of::<HError>(), 32);
        assert_eq!(size_of::<PrimalDualErrors>(), 184);
    }

    #[test]
    fn variable_kkt() {
        let v = get_variable_kkt_failures(1e-7, 1e-6, 0.0, 1.0, -1e-3, 2.0, var_type::CONTINUOUS);
        assert_eq!(v.primal_infeasibility, 1e-3);
        assert_eq!(v.dual_infeasibility, 0.0);
        assert_eq!(v.mid_status, SOL_LO);
        let v = get_variable_kkt_failures(1e-7, 1e-6, -INF, INF, 0.0, -2.0, var_type::CONTINUOUS);
        assert_eq!(v.dual_infeasibility, 2.0);
    }
}
