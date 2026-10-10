//! Highs methods driven by Rust over the C++ `Highs` object (the `Run`
//! of run.rs, whose `Op` steps are in HighsRunRust.cpp): callSolveMip's
//! handling of the MIP solver's result, checkOptimality,
//! completeSolutionFromDiscreteAssignment, getDualRay / getPrimalRay's
//! re-solves (getDualRayInterface, getPrimalRayInterface), presolve and
//! crossover. Each step on a C++ object is one `Op`.

use super::ffi::{CLp, RsMut};
use super::report::assess_col_primal_solution;
use super::run::{
    status_from_model_status, CHighs, Op, Run, MS_INFEASIBLE, MS_NOTSET, MS_OPTIMAL, MS_POSTSOLVE_ERROR,
    MS_PRESOLVE_ERROR, MS_SOLVE_ERROR, MS_UNKNOWN, PS_INFEASIBLE, PS_NOT_PRESOLVED, PS_NOT_REDUCED,
    PS_OUT_OF_MEMORY, PS_REDUCED, PS_REDUCED_TO_EMPTY, PS_TIMEOUT, PS_UNBOUNDED_OR_INFEASIBLE,
};
use super::var_type::CONTINUOUS;
use super::{LogType, Status, INF};
use crate::simplex::hekk::model_status_string;
use crate::util::printf::sprintf;
use crate::{log_dev, log_user};
use std::ffi::c_void;

const SOLUTION_STATUS_NONE: i32 = 0;
const SOLUTION_STATUS_INFEASIBLE: i32 = 1;
/// kHighsIInf
const HIGHS_I_INF: i64 = i32::MAX as i64;

/// What the MIP solver of callSolveMip returns (HighsRunRust.cpp: RsMipResult)
#[repr(C)]
#[derive(Default)]
pub struct MipResult {
    pub model_status: i32,
    pub solution_objective: f64,
    pub node_count: i64,
    pub total_lp_iterations: i64,
    pub dual_bound: f64,
    pub gap: f64,
    pub primal_dual_integral: f64,
    pub row_violation: f64,
    pub bound_violation: f64,
    pub integrality_violation: f64,
}

/// The views of solution_ (HighsRust.h: RsSolution)
#[repr(C)]
pub struct SolutionView {
    pub value_valid: bool,
    pub dual_valid: bool,
    pub col_value: RsMut<f64>,
    pub col_dual: RsMut<f64>,
    pub row_value: RsMut<f64>,
    pub row_dual: RsMut<f64>,
}

/// The dual or primal ray record of the simplex solver
/// (HighsRunRust.cpp: RsRayRecord)
#[repr(C)]
#[derive(Default)]
pub struct RayRecord {
    pub index: i32,
    pub sign: i32,
    pub value_size: i64,
    pub has_invert: bool,
}

/// The sizes of the solution and basis passed to Highs::postsolve
/// (HighsRunRust.cpp: RsPostsolveArgs)
#[repr(C)]
#[derive(Default)]
pub struct PostsolveArgs {
    pub col_value_size: i64,
    pub col_dual_size: i64,
    pub row_dual_size: i64,
    pub dual_valid: bool,
    pub basis_col_size: i64,
    pub basis_row_size: i64,
    pub basis_valid: bool,
}

/// The debug fields of basis_ (HighsRunRust.cpp: RsBasisDebug)
#[repr(C)]
#[derive(Default)]
pub struct BasisDebug {
    pub id: i32,
    pub update_count: i32,
    pub origin: super::options::RsStr,
}

/// HighsPostsolveStatus::kSolutionRecovered
const POSTSOLVE_SOLUTION_RECOVERED: i32 = 1;

/// kNoRayIndex
const NO_RAY_INDEX: i32 = -1;

impl Run<'_> {
    fn lp_view(&self) -> CLp {
        let mut v = std::mem::MaybeUninit::<CLp>::uninit();
        self.op(Op::LpView, 0, v.as_mut_ptr() as *mut c_void);
        // SAFETY: the C++ step fills the view
        unsafe { v.assume_init() }
    }
    fn solution_view(&self) -> SolutionView {
        let mut v = std::mem::MaybeUninit::<SolutionView>::uninit();
        self.op(Op::SolutionView, 0, v.as_mut_ptr() as *mut c_void);
        // SAFETY: the C++ step fills the view
        unsafe { v.assume_init() }
    }
    fn st(&self, v: i64) -> Status {
        match v {
            0 => Status::Ok,
            1 => Status::Warning,
            _ => Status::Error,
        }
    }

    /// callRunPostsolve's clean-up solve of the model LP, timed by the
    /// solve clock, on the engine's data (as the LP part of a run, but the
    /// Highs object's presolve data are kept)
    fn postsolve_cleanup_solve(&self) -> Status {
        let mut handle: *mut super::lp_handle::LpHandle = std::ptr::null_mut();
        self.op(Op::LpRustBegin, 0, &mut handle as *mut _ as *mut c_void);
        if self.ab() {
            return Status::Error;
        }
        let mode = super::lp_run::LpMode::new(self.c, handle);
        let c2 = mode.view();
        // SAFETY: c2's pointers live for the call
        let run2 = unsafe { Run::new(&c2) };
        run2.op0(Op::EkkInvalidate);
        run2.op_msg(Op::SetEkkLpName, 0, "Postsolve LP");
        self.clock(super::run::Clock::Solve, 1);
        let call_status =
            self.st(run2.op_msg(Op::CallSolveLp, 0, "Solving the original LP from the solution after postsolve"));
        let aborted = run2.ab() || mode.aborted.get();
        if !aborted {
            self.clock(super::run::Clock::Solve, 2);
        }
        self.op(Op::LpRustEnd, 1, std::ptr::null_mut());
        if aborted {
            self.abort();
            return Status::Error;
        }
        call_status
    }

    /// Highs::checkOptimality
    pub fn check_optimality(&self, solver_type: &str) -> Status {
        let info = self.info();
        if info.num_primal_infeasibilities == 0 && info.num_dual_infeasibilities <= 0 {
            if info.num_semi_infeasibilities > 0 {
                log_user!(
                    self.log(),
                    LogType::Error,
                    "%s solver claims optimality, but with num/max/sum %d/%g/%g semi-variable infeasibilities: consider solving with smaller mip_feasibility_tolerance\n",
                    solver_type,
                    info.num_semi_infeasibilities,
                    info.max_semi_infeasibility,
                    info.sum_semi_infeasibilities
                );
                self.set_ms(MS_SOLVE_ERROR);
                log_user!(self.log(), LogType::Error, "Setting model status to %s\n", model_status_string(MS_SOLVE_ERROR));
                return Status::Error;
            }
            return Status::Ok;
        }
        self.set_ms(MS_SOLVE_ERROR);
        let mut report = sprintf(
            "%s solver claims optimality, but with num/max/sum primal(%d/%g/%g)",
            &[
                solver_type.into(),
                info.num_primal_infeasibilities.into(),
                info.max_primal_infeasibility.into(),
                info.sum_primal_infeasibilities.into(),
            ],
        );
        if info.num_dual_infeasibilities > 0 {
            report += &sprintf(
                "and dual(%d/%g/%g)",
                &[
                    info.num_dual_infeasibilities.into(),
                    info.max_dual_infeasibility.into(),
                    info.sum_dual_infeasibilities.into(),
                ],
            );
        }
        report += " infeasibilities\n";
        log_user!(self.log(), LogType::Error, "%s", &report);
        log_user!(self.log(), LogType::Error, "Setting model status to %s\n", model_status_string(MS_SOLVE_ERROR));
        Status::Error
    }

    /// Highs::callSolveMip: the MIP solver is one C++ step, its result
    /// is handled here
    pub fn call_solve_mip(&self) -> Status {
        let mut r = MipResult::default();
        self.op(Op::MipRun, 0, &mut r as *mut MipResult as *mut c_void);
        if self.ab() {
            return Status::Error;
        }
        let mut return_status = status_from_model_status(r.model_status);
        self.set_ms(r.model_status);
        let has_solution = r.solution_objective != INF;
        if has_solution {
            self.op0(Op::MipTakeSolution);
        }
        if self.get(self.c.value_valid) && self.op0(Op::ActiveModifiedUpperBounds) != 0 {
            self.set(self.c.value_valid, false);
            self.set_ms(MS_SOLVE_ERROR);
            return_status = Status::Error;
        }
        self.info().objective_function_value = r.solution_objective;
        // Primal feasibility is judged by mip_feasibility_tolerance
        let mut mft = self.c.o.mip_feasibility_tolerance;
        self.op(Op::SwapPrimalTolerance, 0, &mut mft as *mut f64 as *mut c_void);
        self.op0(Op::KktFailures);
        {
            let info = self.info();
            info.mip_node_count = r.node_count;
            info.mip_dual_bound = r.dual_bound;
            info.mip_gap = r.gap;
            info.primal_dual_integral = r.primal_dual_integral;
            info.simplex_iteration_count =
                if r.total_lp_iterations > HIGHS_I_INF { -1 } else { r.total_lp_iterations as i32 };
            info.valid = true;
        }
        if self.ms() == MS_OPTIMAL {
            return_status = self.check_optimality("MIP");
        }
        if has_solution {
            let info = self.info();
            let mip_max_bound_violation = super::lp_utils::cmax(r.row_violation, r.bound_violation);
            let delta = (mip_max_bound_violation - info.max_primal_infeasibility).abs();
            if delta > 1e-12 && self.dev_on() {
                log_dev!(
                    self.log(),
                    LogType::Warning,
                    "Inconsistent max bound violation: MIP solver (%10.4g); LP (%10.4g); Difference of %10.4g\n",
                    mip_max_bound_violation,
                    info.max_primal_infeasibility,
                    delta
                );
            }
            info.max_integrality_violation = r.integrality_violation;
            if info.max_integrality_violation > self.c.o.mip_feasibility_tolerance {
                info.primal_solution_status = SOLUTION_STATUS_INFEASIBLE;
            }
        }
        self.op(Op::SwapPrimalTolerance, 1, std::ptr::null_mut());
        self.op0(Op::MipFinish);
        return_status
    }

    /// Highs::completeSolutionFromDiscreteAssignment: fixes the discrete
    /// variables at integer values of the user's solution and solves for
    /// the rest
    pub fn complete_solution_from_discrete_assignment(&self) -> Status {
        let contains_undefined_values = self.op0(Op::SolutionHasUndefined) != 0;
        if !contains_undefined_values && self.op0(Op::SolutionFeasible) != 0 {
            return Status::Ok;
        }
        self.op(Op::SaveColBounds, 0, std::ptr::null_mut());
        let (pft, mft) = (self.c.o.primal_feasibility_tolerance, self.c.o.mip_feasibility_tolerance);
        let mut num_fixed = 0;
        let mut num_unfixed = 0;
        let num_col;
        {
            let lp = self.lp_view();
            let s = self.solution_view();
            num_col = lp.num_col;
            // SAFETY: the views of model_.lp_ and solution_, unaliased
            // until the next step
            let (lower, upper, integrality, col_value) =
                unsafe { (lp.col_lower.get_mut(), lp.col_upper.get_mut(), lp.integrality.get_mut(), s.col_value.get_mut()) };
            for j in 0..num_col as usize {
                let primal = col_value[j];
                col_value[j] = lower[j];
                if integrality[j] == CONTINUOUS {
                    continue;
                }
                if primal == INF {
                    num_unfixed += 1;
                } else {
                    let (_, integer_infeasibility) =
                        assess_col_primal_solution(pft, mft, primal, lower[j], upper[j], integrality[j]);
                    if integer_infeasibility > mft {
                        num_unfixed += 1;
                    } else {
                        num_fixed += 1;
                        lower[j] = primal;
                        upper[j] = primal;
                        integrality[j] = CONTINUOUS;
                    }
                }
            }
        }
        if num_fixed > 0 {
            // The fixings changed the model through its view
            self.op(Op::LpView, 2, std::ptr::null_mut());
        }
        let num_discrete = num_unfixed + num_fixed;
        let num_continuous = num_col - num_discrete;
        let mut call_run = true;
        let few_fixed = 10 * num_fixed < num_discrete;
        if num_unfixed == 0 {
            if num_continuous == 0 {
                log_user!(
                    self.log(),
                    LogType::Info,
                    "User-supplied values of discrete variables cannot yield feasible solution\n"
                );
                call_run = false;
            } else {
                self.op0(Op::ClearIntegrality);
                log_user!(
                    self.log(),
                    LogType::Info,
                    "Attempting to find feasible solution by solving LP for user-supplied values of discrete variables\n"
                );
            }
        } else if few_fixed {
            log_user!(
                self.log(),
                LogType::Warning,
                "User-supplied values fix only %d / %d discrete variables, so attempt to complete a feasible solution may be expensive\n",
                num_fixed,
                num_discrete
            );
        } else {
            log_user!(
                self.log(),
                LogType::Info,
                "Attempting to find feasible solution by solving MIP for user-supplied values of %d / %d discrete variables\n",
                num_fixed,
                num_discrete
            );
        }
        let mut return_status = Status::Ok;
        self.op0(Op::SolutionClear);
        if call_run {
            self.op(Op::SwapMipMaxNodes, 0, std::ptr::null_mut());
            self.op0(Op::BasisClear);
            return_status = self.st(self.op0(Op::OptimizeModel));
            if self.ab() {
                return Status::Error;
            }
            self.op(Op::SwapMipMaxNodes, 1, std::ptr::null_mut());
        }
        self.op(Op::SaveColBounds, 1, std::ptr::null_mut());
        if return_status == Status::Error {
            log_user!(self.log(), LogType::Error, "Highs::optimizeModel() error trying to find feasible solution\n");
            return Status::Error;
        }
        Status::Ok
    }

    fn ray_record(&self, primal: bool) -> RayRecord {
        let mut r = RayRecord::default();
        self.op(Op::RayRecord, primal as i64, &mut r as *mut RayRecord as *mut c_void);
        r
    }

    /// Highs::getDualRayInterface (getDualRay with a ray to fill): solves
    /// the feasibility problem (zero costs and Hessian, no presolve, the
    /// relaxation) if no ray or INVERT is known
    pub fn get_dual_ray(&self, has_dual_ray: &mut bool, value: Option<&mut [f64]>) -> Status {
        let mut return_status = Status::Ok;
        let num_row = self.facts(0).num_row;
        if num_row == 0 {
            return return_status;
        }
        let mut rec = self.ray_record(false);
        let mut has_invert = rec.has_invert;
        *has_dual_ray = rec.index != NO_RAY_INDEX;
        let mut solve_feasibility_problem = false;
        let is_qp = self.facts(0).is_qp;
        if let Some(value) = value {
            if !*has_dual_ray || !has_invert {
                if self.ms() == MS_OPTIMAL {
                    log_user!(self.log(), LogType::Info, "Model status is optimal, so no dual ray is available\n");
                    return return_status;
                }
                log_user!(self.log(), LogType::Info, "Solving LP to try to compute dual ray\n");
                // Saves the costs, any Hessian and the presolve and
                // solve_relaxation options, zeroes the costs (keeping the
                // primal ray record) and any Hessian, and sets the options
                self.op(Op::FeasibilityProblem, is_qp as i64, std::ptr::null_mut());
                solve_feasibility_problem = true;
                let call_status = self.st(self.op0(Op::HighsRun));
                if self.ab() {
                    return Status::Error;
                }
                if call_status != Status::Ok {
                    return_status = call_status;
                }
                rec = self.ray_record(false);
                *has_dual_ray = rec.index != NO_RAY_INDEX;
                has_invert = rec.has_invert;
            }
            if *has_dual_ray {
                if rec.value_size != 0 {
                    log_user!(self.log(), LogType::Info, "Copying known dual ray\n");
                    self.op(Op::CopyRay, 0, value.as_mut_ptr() as *mut c_void);
                } else if has_invert {
                    log_user!(self.log(), LogType::Info, "Solving linear system to compute dual ray\n");
                    self.op(Op::ComputeDualRay, 0, value.as_mut_ptr() as *mut c_void);
                } else {
                    log_user!(self.log(), LogType::Error, "No LP invertible representation to compute dual ray\n");
                    return_status = Status::Error;
                }
            } else {
                log_user!(self.log(), LogType::Info, "No dual ray found\n");
                return_status = Status::Ok;
            }
        }
        if solve_feasibility_problem {
            // Restores the costs, Hessian and options
            self.op(Op::FeasibilityProblem, 2 + is_qp as i64, std::ptr::null_mut());
            let info = self.info();
            info.primal_solution_status = SOLUTION_STATUS_NONE;
            info.dual_solution_status = SOLUTION_STATUS_NONE;
            info.objective_function_value = 0.0;
            info.invalidate_dual_kkt();
            if !*has_dual_ray {
                info.invalidate_primal_kkt();
                self.set_ms(MS_NOTSET);
            }
        }
        return_status
    }

    /// Highs::getPrimalRayInterface: solves the unboundedness problem (no
    /// presolve, the relaxation, unbounded-or-infeasible not allowed) if
    /// no ray or INVERT is known
    pub fn get_primal_ray(&self, has_primal_ray: &mut bool, value: Option<&mut [f64]>) -> Status {
        let mut return_status = Status::Ok;
        let facts = self.facts(0);
        if facts.num_row == 0 {
            return return_status;
        }
        if facts.is_qp {
            log_user!(self.log(), LogType::Info, "Cannot find primal ray for unbounded QP\n");
            return Status::Error;
        }
        let mut rec = self.ray_record(true);
        let mut has_invert = rec.has_invert;
        *has_primal_ray = rec.index != NO_RAY_INDEX;
        let mut solve_unboundedness_problem = false;
        if let Some(value) = value {
            if !*has_primal_ray || !has_invert {
                if self.ms() == MS_OPTIMAL {
                    log_user!(self.log(), LogType::Info, "Model status is optimal, so no primal ray is available\n");
                    return return_status;
                }
                log_user!(self.log(), LogType::Info, "Solving LP to try to compute primal ray\n");
                self.op(Op::UnboundednessProblem, 0, std::ptr::null_mut());
                solve_unboundedness_problem = true;
                let call_status = self.st(self.op0(Op::HighsRun));
                if self.ab() {
                    return Status::Error;
                }
                if call_status != Status::Ok {
                    return_status = call_status;
                }
                rec = self.ray_record(true);
                *has_primal_ray = rec.index != NO_RAY_INDEX;
                has_invert = rec.has_invert;
            }
            if *has_primal_ray {
                if rec.value_size != 0 {
                    log_user!(self.log(), LogType::Info, "Copying known primal ray\n");
                    self.op(Op::CopyRay, 1, value.as_mut_ptr() as *mut c_void);
                    return return_status;
                } else if has_invert {
                    log_user!(self.log(), LogType::Info, "Solving linear system to compute primal ray\n");
                    self.op(Op::ComputePrimalRay, 0, value.as_mut_ptr() as *mut c_void);
                }
            } else {
                log_user!(self.log(), LogType::Info, "No primal ray found\n");
                return_status = Status::Ok;
            }
        }
        if solve_unboundedness_problem {
            if self.facts(0).is_mip {
                let info = self.info();
                info.dual_solution_status = SOLUTION_STATUS_NONE;
                info.invalidate_dual_kkt();
            }
            self.op(Op::UnboundednessProblem, 1, std::ptr::null_mut());
        }
        return_status
    }

    /// Highs::presolve
    pub fn presolve(&self) -> Status {
        if self.op0(Op::NeedsMods) != 0 {
            log_user!(
                self.log(),
                LogType::Error,
                "Model contains infinite costs or semi-variables, so cannot be presolved independently\n"
            );
            return Status::Error;
        }
        let facts = self.facts(0);
        if facts.is_qp {
            log_user!(self.log(), LogType::Error, "Model is a QP, for which no presolve techniques are implemented\n");
            return Status::Error;
        }
        let mut return_status;
        self.op0(Op::ReportModelStats);
        self.op0(Op::ClearPresolve);
        // SAFETY: the C++ presolve status lives for the call
        let set_ps = |s: i32| unsafe { *self.c.presolve_status = s };
        if facts.is_empty {
            set_ps(PS_NOT_REDUCED);
        } else {
            return_status = self.st(self.op0(Op::InitializeMultiThreading));
            if return_status != Status::Ok {
                return return_status;
            }
            // runPresolve with the profiling set up if there is none
            let ps = self.op(Op::PresolveProfiled, 0, std::ptr::null_mut());
            if self.ab() {
                return Status::Error;
            }
            set_ps(ps as i32);
        }
        // SAFETY: as set_ps
        let ps = unsafe { *self.c.presolve_status };
        self.op0(Op::ReportPresolveReductions);
        let mut using_reduced_lp = false;
        match ps {
            PS_NOT_PRESOLVED => return_status = Status::Error,
            PS_NOT_REDUCED | PS_INFEASIBLE | PS_REDUCED | PS_REDUCED_TO_EMPTY | PS_UNBOUNDED_OR_INFEASIBLE => {
                if ps == PS_INFEASIBLE {
                    self.set_status_and_clear(MS_INFEASIBLE);
                } else if ps == PS_NOT_REDUCED {
                    self.op(Op::PresolvedModel, 0, std::ptr::null_mut());
                } else if ps == PS_REDUCED || ps == PS_REDUCED_TO_EMPTY {
                    using_reduced_lp = true;
                }
                return_status = Status::Ok;
            }
            PS_TIMEOUT => {
                using_reduced_lp = true;
                return_status = Status::Warning;
            }
            _ => {
                debug_assert_eq!(ps, PS_OUT_OF_MEMORY);
                log_user!(self.log(), LogType::Error, "Presolve fails due to memory allocation error\n");
                self.set_status_and_clear(MS_PRESOLVE_ERROR);
                return_status = Status::Error;
            }
        }
        if using_reduced_lp {
            self.op(Op::PresolvedModel, 1, std::ptr::null_mut());
        }
        log_user!(self.log(), LogType::Info, "Presolve status: %s\n", super::run::presolve_status_string(ps));
        self.return_from_highs(return_status)
    }

    /// Highs::crossover from a user solution
    pub fn crossover(&self) -> Status {
        let facts = self.facts(0);
        let return_status;
        if facts.is_mip {
            log_user!(self.log(), LogType::Error, "Cannot apply crossover to solve MIP\n");
            return_status = Status::Error;
        } else if facts.is_qp {
            log_user!(self.log(), LogType::Error, "Cannot apply crossover to solve QP\n");
            return_status = Status::Error;
        } else {
            self.op0(Op::ClearSolver);
            // solution_ = user_solution; callCrossover
            let s = self.st(self.op(Op::Crossover, 0, std::ptr::null_mut()));
            if self.ab() {
                return Status::Error;
            }
            if s == Status::Error {
                return s;
            }
            return_status = s;
            // The objective and the KKT failures
            self.op(Op::Crossover, 1, std::ptr::null_mut());
        }
        self.return_from_highs(return_status)
    }

    /// Highs::callRunPostsolve of a solution (and basis) of the presolved
    /// model
    pub fn call_run_postsolve(&self) -> Status {
        let mut return_status = Status::Ok;
        let reduced = self.facts(1);
        let mut a = PostsolveArgs::default();
        self.op(Op::PostsolveArgs, 0, &mut a as *mut PostsolveArgs as *mut c_void);
        if a.col_value_size != reduced.num_col as i64 {
            log_user!(
                self.log(),
                LogType::Error,
                "Primal solution provided to postsolve is of size %d rather than %d\n",
                a.col_value_size,
                reduced.num_col
            );
            return Status::Error;
        }
        let basis_supplied = a.basis_col_size > 0 || a.basis_row_size > 0 || a.basis_valid;
        if basis_supplied && self.op0(Op::PostsolveBasisConsistent) == 0 {
            log_user!(self.log(), LogType::Error, "Basis provided to postsolve is incorrect size or inconsistent\n");
            return Status::Error;
        }
        // recovered_solution_ = solution, with zero row values
        self.op(Op::PostsolveSetSolution, 0, std::ptr::null_mut());
        let facts = self.facts(0);
        if facts.is_mip && !a.basis_valid {
            // No duals and no basis
            self.op(Op::PostsolveSetSolution, 1, std::ptr::null_mut());
            let postsolve_status = self.run_postsolve();
            if self.ab() {
                return Status::Error;
            }
            if postsolve_status == POSTSOLVE_SOLUTION_RECOVERED {
                self.op0(Op::TakeRecoveredSolution);
                self.set_ms(MS_UNKNOWN);
                self.invalidate_info();
                self.op(Op::PostsolveKkt, 1, std::ptr::null_mut());
                let lp = self.lp_view();
                let s = self.solution_view();
                // SAFETY: the views of model_.lp_ and solution_
                let (integrality, col_value) = unsafe { (lp.integrality.get(), s.col_value.get()) };
                let mut max_integrality_violation = 0.0;
                for j in 0..lp.num_col as usize {
                    if integrality[j] == super::var_type::INTEGER {
                        max_integrality_violation =
                            super::lp_utils::cmax((col_value[j] - col_value[j].round()).abs(), max_integrality_violation);
                    }
                }
                self.info().max_integrality_violation = max_integrality_violation;
                log_user!(self.log(), LogType::Warning, "Postsolve performed for MIP, but model status cannot be known\n");
            } else {
                log_user!(self.log(), LogType::Error, "Postsolve return status is %d\n", postsolve_status);
                self.set_status_and_clear(MS_POSTSOLVE_ERROR);
            }
        } else {
            let dual_supplied = a.col_dual_size > 0 || a.row_dual_size > 0 || a.dual_valid;
            if dual_supplied {
                if a.row_dual_size != reduced.num_row as i64 {
                    log_user!(
                        self.log(),
                        LogType::Error,
                        "Row dual solution provided to postsolve is of size %d rather than %d\n",
                        a.row_dual_size,
                        reduced.num_row
                    );
                    return Status::Error;
                }
                if a.col_dual_size != reduced.num_col as i64 {
                    log_user!(
                        self.log(),
                        LogType::Error,
                        "Column dual solution provided to postsolve is of size %d rather than %d\n",
                        a.col_dual_size,
                        reduced.num_col
                    );
                    return Status::Error;
                }
            }
            // dual_valid, then recovered_basis_ = basis, valid if supplied
            self.op(Op::PostsolveSetSolution, 2 + dual_supplied as i64 + 2 * basis_supplied as i64, std::ptr::null_mut());
            let postsolve_status = self.run_postsolve();
            if self.ab() {
                return Status::Error;
            }
            if postsolve_status == POSTSOLVE_SOLUTION_RECOVERED {
                log_dev!(self.log(), LogType::Verbose, "Postsolve finished\n");
                // solution_ and basis_ from the recovered ones, zero duals
                // if there are none
                self.op0(Op::PostsolveTakeRecovered);
                if self.get(self.c.basis_valid) {
                    self.op0(Op::SaveOptions);
                    self.op0(Op::OptionsPostsolveCleanup);
                    self.op0(Op::RefineBasis);
                    let call_status = self.postsolve_cleanup_solve();
                    if self.ab() {
                        return Status::Error;
                    }
                    return_status = self.interpret(call_status, return_status, "callSolveLp");
                    self.op0(Op::RestoreOptions);
                    self.op(Op::PostsolveKkt, 0, std::ptr::null_mut());
                    if return_status == Status::Error {
                        return self.return_from_optimize_model(return_status, false);
                    }
                } else {
                    self.op0(Op::BasisClear);
                    self.set_ms(if facts.is_mip { MS_NOTSET } else { MS_UNKNOWN });
                    self.op_msg(Op::KktCheck, 0, "");
                    self.info().valid = true;
                    let dual_valid = self.get(self.c.dual_valid);
                    log_user!(
                        self.log(),
                        LogType::Info,
                        "\nPure postsolve yields primal %s basis: model status is %s\n",
                        if dual_valid { "and dual solution, but no" } else { "but no dual solution or" },
                        model_status_string(self.ms())
                    );
                }
            } else {
                log_user!(self.log(), LogType::Error, "Postsolve return status is %d\n", postsolve_status);
                self.set_status_and_clear(MS_POSTSOLVE_ERROR);
                return self.return_from_optimize_model(Status::Error, false);
            }
        }
        let call_status = status_from_model_status(self.ms());
        self.interpret(call_status, return_status, "highsStatusFromHighsModelStatus")
    }

    /// Highs::setBasis(basis, origin): an alien basis is checked and
    /// completed by forming a simplex basis and its factor
    pub fn set_basis(&self, alien: bool, origin: &str) -> Status {
        if alien {
            if self.facts(0).num_row == 0 {
                // No rows: basic columns are made nonbasic
                self.op(Op::SetBasis, 0, std::ptr::null_mut());
            } else {
                let mut sizes = [0i64; 4];
                if self.op(Op::SetBasis, 1, sizes.as_mut_ptr() as *mut c_void) == 0 {
                    log_user!(
                        self.log(),
                        LogType::Error,
                        "setBasis: User basis is rejected due to mismatch between size of column and row status vectors (%d, %d) and number of columns and rows in the model (%d, %d)\n",
                        sizes[0],
                        sizes[1],
                        sizes[2],
                        sizes[3]
                    );
                    return Status::Error;
                }
                // formSimplexLpBasisAndFactor of the basis marked
                // was_alien, which becomes basis_
                let s = self.st(self.op(Op::SetBasis, 2, std::ptr::null_mut()));
                if self.ab() || s != Status::Ok {
                    return Status::Error;
                }
            }
        } else {
            if self.op(Op::SetBasis, 3, std::ptr::null_mut()) == 0 {
                log_user!(self.log(), LogType::Error, "setBasis: invalid basis\n");
                return Status::Error;
            }
            // basis_ = basis
            self.op(Op::SetBasis, 4, std::ptr::null_mut());
        }
        self.set(self.c.basis_valid, true);
        self.set(self.c.basis_useful, true);
        if !origin.is_empty() {
            self.op_msg(Op::SetBasisOrigin, 0, origin);
        }
        if self.get(self.c.basis_was_alien) && self.dev_on() {
            let mut d = BasisDebug::default();
            self.op(Op::BasisDebug, 0, &mut d as *mut BasisDebug as *mut c_void);
            // SAFETY: the C++ string lives for the call
            let name = String::from_utf8_lossy(unsafe { d.origin.get() });
            log_dev!(
                self.log(),
                LogType::Info,
                "Highs::setBasis Was alien = %-5s; Id = %9d; UpdateCount = %4d; Origin (%s)\n",
                "true",
                d.id,
                d.update_count,
                name.as_ref()
            );
        }
        self.op0(Op::NewHighsBasis);
        Status::Ok
    }

    /// The dimension and number of nonzeros of model_.hessian_
    fn hessian_dims(&self) -> (i32, i32) {
        let mut d = [0i32; 2];
        self.op(Op::HessianDims, 0, d.as_mut_ptr() as *mut c_void);
        (d[0], d[1])
    }

    /// The Hessian part of passModel and passHessian: assess it, drop a
    /// zero Hessian, complete it to the number of columns
    fn assess_passed_hessian(&self, mut return_status: Status) -> Status {
        return_status = self.interpret(self.st(self.op0(Op::AssessHessian)), return_status, "assessHessian");
        if return_status == Status::Error {
            return return_status;
        }
        let (dim, num_nz) = self.hessian_dims();
        if dim != 0 && num_nz == 0 {
            log_user!(self.log(), LogType::Info, "Hessian has dimension %d but no nonzeros, so is ignored\n", dim);
            self.op0(Op::HessianClear);
        }
        if self.hessian_dims().0 != 0 {
            self.op0(Op::CompleteHessian);
        }
        return_status
    }

    /// Highs::passModel(HighsModel): the model's LP and Hessian become
    /// model_ after their checks
    pub fn pass_model(&self) -> Status {
        self.op0(Op::LogHeader);
        // (analyseLp of highs_analysis_level is left out)
        let mut return_status = Status::Ok;
        self.op0(Op::ClearModel);
        self.op0(Op::TakeModel);
        let f = self.facts(0);
        if f.num_col == 0 || f.num_row == 0 {
            log_user!(
                self.log(),
                LogType::Info,
                "Model has either no columns or no rows, so ignoring user constraint matrix data and initialising empty matrix\n"
            );
            self.op0(Op::EmptyMatrix);
        } else if self.op(Op::FormatOk, 0, std::ptr::null_mut()) == 0 {
            return Status::Error;
        }
        // Matrix dimensions from the LP, no scaling
        self.op0(Op::PrepareModelLp);
        if !super::lp_utils::lp_dimensions_ok(self.log(), "passModel", &self.lp_view()) {
            return Status::Error;
        }
        if self.op(Op::FormatOk, 1, std::ptr::null_mut()) == 0 {
            return Status::Error;
        }
        return_status = self.interpret(self.st(self.op0(Op::AssessLp)), return_status, "assessLp");
        if return_status == Status::Error {
            return return_status;
        }
        self.op0(Op::EnsureColwise);
        return_status = self.assess_passed_hessian(return_status);
        if return_status == Status::Error {
            return return_status;
        }
        self.op0(Op::MatrixImages);
        return_status = self.interpret(self.st(self.op0(Op::ClearSolver2)), return_status, "clearSolver");
        self.return_from_highs(return_status)
    }

    /// Highs::passHessian(HighsHessian)
    pub fn pass_hessian(&self) -> Status {
        self.op0(Op::LogHeader);
        self.op0(Op::TakeHessian);
        let mut return_status = self.assess_passed_hessian(Status::Ok);
        if return_status == Status::Error {
            return return_status;
        }
        return_status = self.interpret(self.st(self.op0(Op::ClearSolver2)), return_status, "clearSolver");
        self.return_from_highs(return_status)
    }

    /// Highs::readModel: the file read into a model by the C++ reader of
    /// its type, then passed
    pub fn read_model(&self, filename: &str) -> Status {
        self.op0(Op::LogHeader);
        let mut return_status = Status::Ok;
        let code = self.op_msg(Op::ReadModelFile, 0, filename);
        if self.ab() {
            return Status::Error;
        }
        if code < 0 {
            log_user!(self.log(), LogType::Error, "Model file %s not supported\n", filename);
            return Status::Error;
        }
        if code != 0 {
            super::model_utils::interpret_filereader_retcode(self.log(), filename, code as i32);
            let call_status = if code == 1 { Status::Warning } else { Status::Error };
            return_status = self.interpret(call_status, return_status, "readModelFromFile");
            if return_status == Status::Error {
                return return_status;
            }
        }
        let name = super::model_utils::extract_model_name(filename);
        self.op_msg(Op::ReadModelPass, 0, &name);
        let pass_status = self.st(self.op(Op::ReadModelPass, 1, std::ptr::null_mut()));
        if self.ab() {
            return Status::Error;
        }
        return_status = self.interpret(pass_status, return_status, "passModel");
        self.return_from_highs(return_status)
    }

    /// Highs::readBasis
    pub fn read_basis(&self, filename: &str) -> Status {
        self.op0(Op::LogHeader);
        // readBasisFile into a copy of basis_
        let call_status = self.st(self.op_msg(Op::ReadBasis, 0, filename));
        let return_status = self.interpret(call_status, Status::Ok, "readBasis");
        if return_status != Status::Ok {
            return return_status;
        }
        if self.op(Op::ReadBasis, 1, std::ptr::null_mut()) == 0 {
            log_user!(self.log(), LogType::Error, "readBasis: invalid basis\n");
            return Status::Error;
        }
        // basis_ = the basis read, valid and useful; newHighsBasis
        self.op(Op::ReadBasis, 2, std::ptr::null_mut());
        Status::Ok
    }

    /// Highs::writeLocalModel of model_ (which 0), presolved_model_ (1) or
    /// another model (2)
    pub fn write_local_model(&self, filename: &str) -> Status {
        let mut return_status = Status::Ok;
        let file_type = super::model_utils::file_type(filename.as_bytes());
        // setMatrixDimensions, normaliseNames, ensureColwise
        let call_status = self.st(self.op(Op::WriteModelPrepare, file_type as i64, std::ptr::null_mut()));
        return_status = self.interpret(call_status, return_status, "normaliseNames");
        let mut lp = std::mem::MaybeUninit::<CLp>::uninit();
        self.op(Op::WriteModelLpView, 0, lp.as_mut_ptr() as *mut c_void);
        // SAFETY: the C++ step fills the view
        let lp = unsafe { lp.assume_init() };
        if !super::lp_utils::lp_dimensions_ok(self.log(), "writeLocalModel", &lp) {
            return Status::Error;
        }
        // assessHessianDimensions, the matrix's assessStart and
        // assessIndexBounds
        for check in 0..3 {
            if self.st(self.op(Op::WriteModelCheck, check, std::ptr::null_mut())) == Status::Error {
                return Status::Error;
            }
        }
        if self.op(Op::WriteModelCheck, 3, std::ptr::null_mut()) != 0 {
            log_user!(self.log(), LogType::Error, "Model has repeated column names\n");
            return self.return_from_highs(Status::Error);
        }
        if self.op(Op::WriteModelCheck, 4, std::ptr::null_mut()) != 0 {
            log_user!(self.log(), LogType::Error, "Model has repeated row names\n");
            return self.return_from_highs(Status::Error);
        }
        if filename.is_empty() {
            self.op0(Op::ReportWrittenModel);
        } else {
            if self.op_msg(Op::WriteModelFile, 0, filename) == 0 {
                log_user!(self.log(), LogType::Error, "Model file %s not supported\n", filename);
                return Status::Error;
            }
            log_user!(self.log(), LogType::Info, "Writing the model to %s\n", filename);
            let call_status = self.st(self.op_msg(Op::WriteModelFile, 1, filename));
            return_status = self.interpret(call_status, return_status, "writeModelToFile");
        }
        self.return_from_highs(return_status)
    }

    /// Highs::writeBasis
    pub fn write_basis(&self, filename: &str) -> Status {
        let mut return_status = Status::Ok;
        // openWriteFile
        let call_status = self.st(self.op_msg(Op::WriteBasis, 0, filename));
        return_status = self.interpret(call_status, return_status, "openWriteFile");
        if return_status == Status::Error {
            return return_status;
        }
        // normaliseNames of model_.lp_
        let call_status = self.st(self.op(Op::WriteBasis, 1, std::ptr::null_mut()));
        return_status = self.interpret(call_status, return_status, "normaliseNames");
        if !filename.is_empty() {
            if !self.get(self.c.basis_valid) {
                log_user!(self.log(), LogType::Warning, "No basis to write: generated null basis file %s\n", filename);
                return_status = Status::Warning;
            } else {
                log_user!(self.log(), LogType::Info, "Writing the basis to %s\n", filename);
            }
        }
        // writeBasisFile and fclose
        self.op(Op::WriteBasis, 2, std::ptr::null_mut());
        return_status
    }

    /// Highs::forceHighsSolutionBasisSize: the solution and basis of the
    /// model's size, invalid if they were smaller (or, for the basis,
    /// different)
    pub fn force_solution_basis_size(&self) {
        let f = self.facts(0);
        let (num_col, num_row) = (f.num_col as i64, f.num_row as i64);
        let mut sizes = [0i64; 6];
        self.op(Op::SolutionBasisSizes, 0, sizes.as_mut_ptr() as *mut c_void);
        if sizes[0] < num_col || sizes[1] < num_row {
            self.set(self.c.value_valid, false);
            self.info().primal_solution_status = SOLUTION_STATUS_NONE;
        }
        if sizes[2] < num_col || sizes[3] < num_row {
            self.set(self.c.dual_valid, false);
            self.info().dual_solution_status = SOLUTION_STATUS_NONE;
        }
        if sizes[4] != num_col || sizes[5] != num_row {
            self.set(self.c.basis_valid, false);
            self.set(self.c.basis_useful, false);
            // kBasisValidityInvalid
            self.info().basis_validity = 0;
        }
        self.op(Op::SolutionBasisSizes, 1, std::ptr::null_mut());
    }
}

// ------------------------------------------------------------ entry points

/// Highs::getDualRayInterface (primal false) / getPrimalRayInterface;
/// `value` holds num_row (num_col) entries, or is NULL
///
/// # Safety
/// `c` is a valid CHighs
#[no_mangle]
pub unsafe extern "C" fn highs_rs_get_ray(c: *const CHighs, primal: bool, has_ray: *mut bool, value: *mut f64, len: usize) -> i32 {
    let run = Run::new(&*c);
    let value = if value.is_null() { None } else { Some(std::slice::from_raw_parts_mut(value, len)) };
    if primal {
        run.get_primal_ray(&mut *has_ray, value) as i32
    } else {
        run.get_dual_ray(&mut *has_ray, value) as i32
    }
}

/// aFormatOk (hessian false) / qFormatOk of a matrix passed by a user
///
/// # Safety
/// `log` is valid
#[no_mangle]
pub unsafe extern "C" fn highs_rs_format_ok(log: *const super::Log, hessian: bool, num_nz: i32, format: i32) -> bool {
    if num_nz == 0 {
        return true;
    }
    let ok = if hessian {
        format == super::hessian::TRIANGULAR || format == super::hessian::SQUARE
    } else {
        format == super::matrix_format::COLWISE || format == super::matrix_format::ROWWISE
    };
    if !ok {
        let fmt = if hessian {
            "Non-empty Hessian matrix has illegal format = %d\n"
        } else {
            "Non-empty Constraint matrix has illegal format = %d\n"
        };
        log_user!(&*log, LogType::Error, fmt, format);
    }
    ok
}

/// The integrality values passed to passModel: the index of the first
/// illegal one (with its error), or -1 if all are legal
///
/// # Safety
/// `integrality` holds `num_col` values
#[no_mangle]
pub unsafe extern "C" fn highs_rs_check_integrality(log: *const super::Log, num_col: i32, integrality: *const i32) -> i32 {
    for j in 0..num_col.max(0) as usize {
        let t = *integrality.add(j);
        if !(0..=4).contains(&t) {
            log_user!(
                &*log,
                LogType::Error,
                "Model has illegal integer value of %d (type %s) for integrality[%d]\n",
                t,
                super::report::var_type_string(t),
                j
            );
            return j as i32;
        }
    }
    -1
}

/// Highs::readModel (0), readBasis (1), writeLocalModel (2), writeBasis
/// (3) of a file name
///
/// # Safety
/// `c` is a valid CHighs; `filename` holds `len` bytes
#[no_mangle]
pub unsafe extern "C" fn highs_rs_highs_file(c: *const CHighs, which: i32, filename: *const u8, len: usize) -> i32 {
    let bytes = if len == 0 { &[][..] } else { std::slice::from_raw_parts(filename, len) };
    let filename = String::from_utf8_lossy(bytes);
    let run = Run::new(&*c);
    match which {
        0 => run.read_model(&filename),
        1 => run.read_basis(&filename),
        2 => run.write_local_model(&filename),
        _ => run.write_basis(&filename),
    }
    .into()
}

/// Highs::forceHighsSolutionBasisSize
///
/// # Safety
/// `c` is a valid CHighs
#[no_mangle]
pub unsafe extern "C" fn highs_rs_force_solution_basis_size(c: *const CHighs) {
    Run::new(&*c).force_solution_basis_size()
}

/// analyseSetCreateError of the Highs methods taking a set
///
/// # Safety
/// `log` is valid; `method` holds `len` bytes; `set` holds the entry
/// a negative create_error points to
#[no_mangle]
pub unsafe extern "C" fn highs_rs_set_create_error(
    log: *const super::Log,
    method: *const u8,
    len: usize,
    create_error: i32,
    ordered: bool,
    num_set_entries: i32,
    set: *const i32,
    dimension: i32,
) -> i32 {
    let log = &*log;
    let method = std::str::from_utf8_unchecked(std::slice::from_raw_parts(method, len));
    // kIndexCollectionCreateIllegalSetSize, kIndexCollectionCreateIllegalSetOrder
    if create_error == 1 {
        log_user!(log, LogType::Error, "Set supplied to Highs::%s has illegal size of %d\n", method, num_set_entries);
    } else if create_error == 3 {
        if ordered {
            log_user!(log, LogType::Error, "Set supplied to Highs::%s contains duplicate entries\n", method);
        } else {
            log_user!(log, LogType::Error, "Set supplied to Highs::%s not ordered\n", method);
        }
    } else if create_error < 0 {
        let illegal_set_index = -1 - create_error;
        let illegal_set_entry = *set.add(illegal_set_index as usize);
        log_user!(
            log,
            LogType::Error,
            "Set supplied to Highs::%s has entry %d of %d out of range [0, %d)\n",
            method,
            illegal_set_index,
            illegal_set_entry,
            dimension
        );
    }
    Status::Error as i32
}
