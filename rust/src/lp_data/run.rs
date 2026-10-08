//! Highs::run's control flow (Highs.cpp): calledOptimizeModel (the
//! choice between the QP, MIP and LP solvers, and for an LP the
//! presolve decision, the solve of the reduced LP, postsolve and the
//! clean-up solve), runPresolve, runPostsolve, returnFromOptimizeModel,
//! returnFromHighs, reportSolvedLpQpStats and reportPresolveReductions.
//!
//! The `Highs` object stays C++ (highs/lp_data/HighsRunRust.cpp): Rust
//! reads and writes its scalars in place (model status, HighsInfo,
//! HighsRunData, the validity flags of the solution and basis) and calls
//! back for each step on a C++ object (`Op`): the solvers, presolve and
//! postsolve calls, the timer, copies of solutions, bases and options.
//! Options are read once per call (`ROptions`), before any step changes
//! them; the steps that change options restore them before returning.
//!
//! MIP solves call calledOptimizeModel for every LP of the relaxation, so
//! nothing here allocates unless it logs, and messages are formatted only
//! when the C++ logger would print them (`on`, `dev_on`).

use super::ffi::CLp;
use super::options::RsStr;
use super::solution::Info;
use super::{Log, LogType, Status, INF};
use crate::simplex::hekk::model_status_string;
use crate::{log_dev, log_user};
use std::cell::Cell;
use std::ffi::c_void;

// HighsModelStatus
pub const MS_NOTSET: i32 = 0;
pub const MS_LOAD_ERROR: i32 = 1;
pub const MS_MODEL_ERROR: i32 = 2;
pub const MS_PRESOLVE_ERROR: i32 = 3;
pub const MS_SOLVE_ERROR: i32 = 4;
pub const MS_POSTSOLVE_ERROR: i32 = 5;
pub const MS_MODEL_EMPTY: i32 = 6;
pub const MS_OPTIMAL: i32 = 7;
pub const MS_INFEASIBLE: i32 = 8;
pub const MS_UNBOUNDED_OR_INFEASIBLE: i32 = 9;
pub const MS_UNBOUNDED: i32 = 10;
pub const MS_OBJECTIVE_BOUND: i32 = 11;
pub const MS_OBJECTIVE_TARGET: i32 = 12;
pub const MS_TIME_LIMIT: i32 = 13;
pub const MS_ITERATION_LIMIT: i32 = 14;
pub const MS_UNKNOWN: i32 = 15;
pub const MS_SOLUTION_LIMIT: i32 = 16;
pub const MS_INTERRUPT: i32 = 17;
pub const MS_MEMORY_LIMIT: i32 = 18;
pub const MS_HIGHS_INTERRUPT: i32 = 19;

// HighsPresolveStatus
pub const PS_NOT_PRESOLVED: i32 = -1;
pub const PS_NOT_REDUCED: i32 = 0;
pub const PS_INFEASIBLE: i32 = 1;
pub const PS_UNBOUNDED_OR_INFEASIBLE: i32 = 2;
pub const PS_REDUCED: i32 = 3;
pub const PS_REDUCED_TO_EMPTY: i32 = 4;
pub const PS_TIMEOUT: i32 = 5;
pub const PS_NULL_ERROR: i32 = 6;
pub const PS_OUT_OF_MEMORY: i32 = 9;

// HighsPostsolveStatus
const POSTSOLVE_NO_PRIMAL_SOLUTION_ERROR: i32 = 0;
const POSTSOLVE_SOLUTION_RECOVERED: i32 = 1;

const SOLUTION_STATUS_NONE: i32 = 0;
const ILLEGAL_INFEASIBILITY_COUNT: i32 = -1;
const ILLEGAL_INFEASIBILITY_MEASURE: f64 = INF;
const ILLEGAL_INT_MEASURE: i32 = -1;
const ILLEGAL_DOUBLE_MEASURE: f64 = -INF;

/// highsStatusFromHighsModelStatus
pub fn status_from_model_status(model_status: i32) -> Status {
    match model_status {
        MS_MODEL_EMPTY | MS_OPTIMAL | MS_INFEASIBLE | MS_UNBOUNDED_OR_INFEASIBLE | MS_UNBOUNDED
        | MS_OBJECTIVE_BOUND | MS_OBJECTIVE_TARGET => Status::Ok,
        MS_TIME_LIMIT | MS_ITERATION_LIMIT | MS_SOLUTION_LIMIT | MS_INTERRUPT | MS_HIGHS_INTERRUPT
        | MS_UNKNOWN => Status::Warning,
        _ => Status::Error,
    }
}

/// Highs::presolveStatusToString
pub fn presolve_status_string(status: i32) -> &'static str {
    match status {
        PS_NOT_PRESOLVED => "Not presolved",
        PS_NOT_REDUCED => "Not reduced",
        PS_INFEASIBLE => "Infeasible",
        PS_UNBOUNDED_OR_INFEASIBLE => "Unbounded or infeasible",
        PS_REDUCED => "Reduced",
        PS_REDUCED_TO_EMPTY => "Reduced to empty",
        PS_TIMEOUT => "Timeout",
        PS_OUT_OF_MEMORY => "Memory allocation error",
        _ => "Unrecognised presolve status",
    }
}

fn status_from_i64(v: i64) -> Status {
    match v {
        0 => Status::Ok,
        1 => Status::Warning,
        _ => Status::Error,
    }
}

/// HighsRunDataStruct
#[repr(C)]
pub struct RunData {
    pub valid: bool,
    pub presolved_model_num_col: i32,
    pub presolved_model_num_row: i32,
    pub presolved_model_num_nz: i32,
    pub num_simplex_iterations_after_postsolve: i32,
    pub presolve_time: f64,
    pub solve_time: f64,
    pub postsolve_time: f64,
}

/// The option values read by the run, and the options it changes in place
#[repr(C)]
pub struct ROptions {
    pub solver: RsStr,
    pub run_crossover: RsStr,
    pub presolve: RsStr,
    pub use_warm_start: bool,
    pub icrash: bool,
    pub solve_relaxation: bool,
    pub allow_unbounded_or_infeasible: bool,
    pub timeless_log: bool,
    pub large_matrix_value: f64,
    pub time_limit: f64,
    pub primal_feasibility_tolerance: f64,
    pub mip_feasibility_tolerance: f64,
    pub highs_debug_level: *mut i32,
    pub objective_bound: *mut f64,
    pub lp_presolve_requires_basis_postsolve: *mut bool,
    /// HighsLogOptions::output_flag and log_dev_level
    pub output_flag: *const bool,
    pub log_dev_level: *const i32,
    pub simplex_strategy: i32,
}

/// Dimensions and properties of model_ (0) or the reduced LP (1)
#[repr(C)]
#[derive(Default)]
pub struct Facts {
    pub num_col: i32,
    pub num_row: i32,
    pub num_nz: i32,
    pub is_mip: bool,
    pub is_qp: bool,
    pub is_empty: bool,
    pub has_infinite_cost: bool,
    pub model_name: RsStr,
}

impl Default for RsStr {
    fn default() -> Self {
        RsStr::of(&[])
    }
}

/// The steps on C++ objects. `arg` and `p` as listed; a returned status
/// is a HighsStatus.
#[repr(i32)]
#[derive(Clone, Copy)]
pub enum Op {
    /// clearSolver()
    ClearSolver = 1,
    /// handleInfCost() -> status
    HandleInfCost,
    /// exactResizeModel()
    ExactResizeModel,
    /// completeSolutionFromDiscreteAssignment() -> status
    CompleteSolution,
    /// invalidateInfo() / invalidateRunData() / invalidateBasis()
    InvalidateInfo,
    InvalidateRunData,
    InvalidateBasis,
    /// Facts of model_ (arg 0) or the reduced LP (arg 1) into p
    Facts,
    /// model_.lp_.ensureColwise()
    EnsureColwise,
    /// model_.lp_.a_matrix_.hasLargeValue(large_matrix_value) -> bool
    HasLargeValue,
    /// assessLp and checkOptions (highs_debug_level > min) -> status
    DebugAssess,
    /// assessSemiVariables -> status, made mods to p (bool)
    AssessSemiVariables,
    /// relaxSemiVariables -> made mods
    RelaxSemiVariables,
    /// okHessianDiagonal -> bool
    OkHessianDiagonal,
    /// callSolveQp() / callSolveMip() -> status
    CallSolveQp,
    CallSolveMip,
    /// basisForSolution() -> status
    BasisForSolution,
    /// basis_.clear()
    BasisClear,
    /// refineBasis(model_.lp_, solution_, basis_)
    RefineBasis,
    /// callSolveLp(model_.lp_ (arg 0) or the reduced LP (1), message) -> status
    CallSolveLp,
    /// ekk_instance_.lp_name_ = message
    SetEkkLpName,
    /// The reduced LP after presolve: origin name, matrix dimensions,
    /// small values, cleanBounds -> status
    PrepareReducedLp,
    /// ekk_instance_.clear() / invalidate()
    EkkClear,
    EkkInvalidate,
    /// ekk_instance_.status_.initialised_for_solve, factor pivot
    /// threshold to p (f64) -> bool
    EkkPivotThreshold,
    /// The trivial optimal solution and basis of a reduced-to-empty LP
    ReducedToEmpty,
    /// Save options_ / restore it from the copy
    SaveOptions,
    RestoreOptions,
    /// options_.solver = "simplex", simplex_strategy = primal
    OptionsPrimalSimplex,
    /// The options of the clean-up solve after postsolve; the factor
    /// pivot threshold in p (f64) if arg
    OptionsCleanup,
    /// presolve_.data_.recovered_solution_ / basis_ = solution_ / basis_
    CopyToPresolve,
    /// callLpKktCheck(model_.lp_ (arg 0) or the reduced LP (1), message)
    KktCheck,
    /// presolve_.data_.recovered_solution_ value_valid | dual_valid << 1
    RecoveredValidity,
    /// postSolveStack.undo, calculateRowValuesQuad, and the column duals
    /// negated (arg: have dual solution and maximize)
    PostsolveUndo,
    /// presolve_.postsolve_status_ = arg
    SetPostsolveStatus,
    /// solution_ = presolve_.data_.recovered_solution_ (after clear())
    TakeRecoveredSolution,
    /// basis_'s statuses from the recovered basis, origin name
    TakeRecoveredBasis,
    /// debugHighsSolution("After returning from postsolve") -> logical error
    DebugPostsolveSolution,
    /// restoreInfCost(status) and unapplyMods -> status
    UndoMods,
    /// The debug checks of returnFromOptimizeModel -> logical error
    DebugReturn,
    /// forceHighsSolutionBasisSize()
    ForceSolutionBasisSize,
    /// debugHighsBasisConsistent -> consistent
    BasisConsistent,
    /// ekk_instance_.debugRetainedDataOk -> ok
    RetainedEkkDataOk,
    /// lpDimensionsOk("returnFromHighs", model_.lp_) -> ok
    LpDimensionsOk,
    /// -1 no NLA, else lpFactorRowCompatible
    EkkFactorCompatible,
    /// presolve_.clear()
    PresolveClear,
    /// MIP presolve -> presolve status
    MipPresolve,
    /// presolve_.init(model_.lp_, timer_), options_
    PresolveInit,
    /// presolve_.run() -> presolve status
    PresolveRun,
    /// presolve_log_ = presolve_.getPresolveLog(); presolve_.presolve_status_ -> status
    PresolveLog,
    /// presolve_.info_ removed counts from p ([i32; 3]); arg: clear the
    /// reduced LP's scaling
    PresolveRemoved,
    /// presolve_.data_.reduced_lp_.integrality_.clear()
    ClearReducedIntegrality,
    /// presolve_.info_.presolve_time (arg 0) / postsolve_time (1) = *p
    PresolveTime,
    /// The view of model_.lp_ (rsLp) into p (CLp)
    LpView,
    // drivers.rs
    /// The MIP solver of callSolveMip (the user's solution kept, solver
    /// data invalidated, semi-variables replaced) run, its result into p
    /// (MipResult); the solver is kept until MipFinish
    MipRun,
    /// solution_.col_value = the solver's solution, the saved solutions,
    /// the row values (productQuad), value_valid
    MipTakeSolution,
    /// activeModifiedUpperBounds(options_, model_.lp_, col_value) -> bool
    ActiveModifiedUpperBounds,
    /// options_.primal_feasibility_tolerance saved and set to *p (arg 0)
    /// or restored (arg 1)
    SwapPrimalTolerance,
    /// getKktFailures(options_, model_, solution_, basis_, info_)
    KktFailures,
    /// Drops the MIP solver
    MipFinish,
    /// solution_.hasUndefined() -> bool
    SolutionHasUndefined,
    /// assessLpPrimalSolution("", options_, model_.lp_, solution_) -> feasible
    SolutionFeasible,
    /// Saves (arg 0) or restores (1) model_.lp_'s column bounds and integrality
    SaveColBounds,
    /// model_.lp_.integrality_.clear()
    ClearIntegrality,
    /// solution_.clear()
    SolutionClear,
    /// options_.mip_max_nodes saved and set to mip_max_start_nodes (arg 0)
    /// or restored (1)
    SwapMipMaxNodes,
    /// optimizeModel(), then resetProfiling() if profiling -> status
    OptimizeModel,
    /// The view of solution_ into p (SolutionView)
    SolutionView,
    /// The simplex dual (arg 0) or primal (1) ray record into p (RayRecord)
    RayRecord,
    /// The feasibility problem of getDualRay: set up (arg 0 or 1, 1 for a
    /// QP) or undone (2 or 3)
    FeasibilityProblem,
    /// The unboundedness problem of getPrimalRay: set up (arg 0) or
    /// undone (1)
    UnboundednessProblem,
    /// run() -> status
    HighsRun,
    /// The known dual (arg 0) or primal (1) ray into p (f64 array)
    CopyRay,
    /// The dual / primal ray solved for into p and the record
    ComputeDualRay,
    ComputePrimalRay,
    /// model_.needsMods(options_.infinite_cost) -> bool
    NeedsMods,
    /// reportModelStats()
    ReportModelStats,
    /// clearPresolve()
    ClearPresolve,
    /// initializeMultiThreading() -> status
    InitializeMultiThreading,
    /// runPresolve(solve_relaxation, true) with profiling -> presolve status
    PresolveProfiled,
    /// reportPresolveReductions of model_.lp_ and the reduced LP
    ReportPresolveReductions,
    /// presolved_model_ = model_ (arg 0) or the reduced LP (1)
    PresolvedModel,
    /// solution_ = the user solution and callCrossover -> status (arg 0);
    /// the objective and KKT failures (1)
    Crossover,
    /// The sizes of postsolve's solution and basis into p (PostsolveArgs)
    PostsolveArgs,
    /// isBasisConsistent(reduced LP, postsolve's basis) -> bool
    PostsolveBasisConsistent,
    /// recovered_solution_ = solution with zero row values and value_valid
    /// (arg 0), no duals and no recovered basis (arg 1), or (arg 2 + d +
    /// 2 b) dual_valid = d and recovered_basis_ = basis with valid = b
    PostsolveSetSolution,
    /// getKktFailures of model_.lp_ with residuals, after objective =
    /// computeObjectiveValue (arg 1)
    PostsolveKkt,
    /// solution_ = recovered_solution_ (zero duals if there are none),
    /// basis_ = recovered_basis_ with ": after postsolve"
    PostsolveTakeRecovered,
    /// simplex_strategy = choose, simplex_min/max_concurrency = 1
    OptionsPostsolveCleanup,
    /// setBasis steps with the user's basis: (arg 0) no rows, basis_'s
    /// basic columns nonbasic; (1) isBasisRightSize, basis_'s and the
    /// model's sizes into p ([i64; 4]) -> bool; (2) the alien basis
    /// formed and factored -> status; (3) isBasisConsistent -> bool; (4)
    /// basis_ = basis
    SetBasis,
    /// basis_.debug_origin_name = message
    SetBasisOrigin,
    /// basis_'s debug fields into p (BasisDebug)
    BasisDebug,
    /// newHighsBasis()
    NewHighsBasis,
    /// model_.hessian_'s dimension and nonzeros into p ([i32; 2])
    HessianDims,
    /// assessHessian(model_.hessian_, options_) -> status
    AssessHessian,
    /// model_.hessian_.clear()
    HessianClear,
    /// completeHessian(model_.lp_.num_col_, model_.hessian_)
    CompleteHessian,
    /// logHeader()
    LogHeader,
    /// clearModel()
    ClearModel,
    /// model_ = the passed model, origin "Original"
    TakeModel,
    /// The column-wise empty matrix of an LP without rows or columns
    EmptyMatrix,
    /// formatOk of model_.lp_'s matrix (arg 0) or Hessian (1) -> bool
    FormatOk,
    /// setMatrixDimensions() and resetScale() of model_.lp_
    PrepareModelLp,
    /// assessLp(model_.lp_, options_) -> status
    AssessLp,
    /// The matrix and Hessian images (write_matrix_image,
    /// write_hessian_image)
    MatrixImages,
    /// clearSolver() -> status
    ClearSolver2,
    /// model_.hessian_ = the passed Hessian
    TakeHessian,
    /// The C++ reader of the file (message) reads it into a model -> -1
    /// if there is none, else its FilereaderRetcode
    ReadModelFile,
    /// The model read: its name = message (arg 0); passModel -> status (1)
    ReadModelPass,
    /// readBasisFile(message) into a copy of basis_ -> status (arg 0);
    /// isBasisConsistent of it -> bool (1); basis_ = it, valid, useful,
    /// newHighsBasis (2)
    ReadBasis,
    /// The model written: setMatrixDimensions, normaliseNames for the
    /// file type arg -> status, ensureColwise
    WriteModelPrepare,
    /// The view of the written model's LP into p (CLp)
    WriteModelLpView,
    /// Checks of the written model: assessHessianDimensions (arg 0, if a
    /// Hessian), assessStart (1), assessIndexBounds (2) -> status;
    /// repeated column (3) or row (4) names -> bool
    WriteModelCheck,
    /// reportModel of the written model
    ReportWrittenModel,
    /// The file writer of message: exists -> bool (arg 0); writes ->
    /// status (1)
    WriteModelFile,
    /// writeBasis: openWriteFile(message) -> status (arg 0),
    /// normaliseNames -> status (1), writeBasisFile and close (2)
    WriteBasis,
    /// The sizes of solution_'s and basis_'s vectors into p ([i64; 6])
    /// (arg 0); resized to the model (1)
    SolutionBasisSizes,
}

/// HighsTimer clocks and actions
#[repr(i32)]
#[derive(Clone, Copy)]
pub enum Clock {
    Run = 0,
    Solve,
    Presolve,
    Postsolve,
}
const READ: i32 = 0;
const START: i32 = 1;
const STOP: i32 = 2;
const RUNNING: i32 = 3;

/// The Highs object
#[repr(C)]
pub struct CHighs {
    pub log: Log,
    pub ctx: *mut c_void,
    pub op: unsafe extern "C" fn(*mut c_void, i32, i64, *mut c_void, *const u8, usize) -> i64,
    pub clock: unsafe extern "C" fn(*mut c_void, i32, i32) -> f64,
    pub model_status: *mut i32,
    pub presolve_status: *mut i32,
    pub info: *mut Info,
    pub run_data: *mut RunData,
    pub value_valid: *mut bool,
    pub dual_valid: *mut bool,
    pub basis_valid: *mut bool,
    pub basis_alien: *mut bool,
    pub basis_useful: *mut bool,
    pub basis_was_alien: *mut bool,
    pub called_return: *mut bool,
    pub o: ROptions,
}

const SIMPLEX: &[u8] = b"simplex";
const CHOOSE: &[u8] = b"choose";
const PDLP: &[u8] = b"pdlp";
const OFF: &[u8] = b"off";
const ON: &[u8] = b"on";

/// useIpm
pub fn use_ipm(solver: &[u8]) -> bool {
    solver == b"ipm" || solver == b"hipo" || solver == b"ipx"
}

/// The option values that decide the run, read at its start
struct Flags {
    solver_will_use_basis: bool,
    ipm_no_crossover: bool,
    solver_pdlp: bool,
    ipm_crossover_on: bool,
    presolve_off: bool,
    solver_ok: [bool; 3],
}

pub struct Run<'a> {
    pub(crate) c: &'a CHighs,
    f: Flags,
    /// A step threw a C++ exception, which C++ rethrows once Rust has
    /// returned: no further step (or log message) is made
    aborted: Cell<bool>,
}

/// What a step returns when it threw
const ABORT: i64 = i64::MIN;

impl<'a> Run<'a> {
    /// # Safety
    /// `c` must be a valid CHighs whose pointers live for the call
    pub unsafe fn new(c: &'a CHighs) -> Run<'a> {
        let solver = c.o.solver.get();
        let run_crossover = c.o.run_crossover.get();
        let f = Flags {
            solver_will_use_basis: solver == SIMPLEX || solver == CHOOSE,
            ipm_no_crossover: use_ipm(solver) && run_crossover == OFF,
            solver_pdlp: solver == PDLP,
            ipm_crossover_on: use_ipm(solver) && run_crossover == ON,
            presolve_off: c.o.presolve.get() == OFF,
            solver_ok: [
                super::options::solver_valid(solver, 0),
                super::options::solver_valid(solver, 1),
                super::options::solver_valid(solver, 2),
            ],
        };
        Run { c, f, aborted: Cell::new(false) }
    }

    pub(crate) fn op(&self, op: Op, arg: i64, p: *mut c_void) -> i64 {
        // SAFETY: the C++ steps of the run, called with their context
        let r = unsafe { (self.c.op)(self.c.ctx, op as i32, arg, p, std::ptr::null(), 0) };
        if r == ABORT {
            self.aborted.set(true);
        }
        r
    }
    pub(crate) fn ab(&self) -> bool {
        self.aborted.get()
    }
    pub(crate) fn op0(&self, op: Op) -> i64 {
        self.op(op, 0, std::ptr::null_mut())
    }
    pub(crate) fn op_msg(&self, op: Op, arg: i64, msg: &str) -> i64 {
        // SAFETY: as op; the message lives for the call
        let r = unsafe { (self.c.op)(self.c.ctx, op as i32, arg, std::ptr::null_mut(), msg.as_ptr(), msg.len()) };
        if r == ABORT {
            self.aborted.set(true);
        }
        r
    }
    pub(crate) fn status_op(&self, op: Op) -> Status {
        status_from_i64(self.op0(op))
    }
    pub(crate) fn clock(&self, clock: Clock, action: i32) -> f64 {
        // SAFETY: as op
        unsafe { (self.c.clock)(self.c.ctx, clock as i32, action) }
    }
    pub(crate) fn read(&self, clock: Clock) -> f64 {
        self.clock(clock, READ)
    }
    pub(crate) fn facts(&self, which: i64) -> Facts {
        let mut f = Facts::default();
        self.op(Op::Facts, which, &mut f as *mut Facts as *mut c_void);
        f
    }

    // The Highs scalars, read and written in place
    pub(crate) fn ms(&self) -> i32 {
        // SAFETY: the C++ model status lives for the call
        unsafe { *self.c.model_status }
    }
    pub(crate) fn set_ms(&self, s: i32) {
        // SAFETY: as ms
        unsafe { *self.c.model_status = s }
    }
    #[allow(clippy::mut_from_ref)]
    pub(crate) fn info(&self) -> &mut Info {
        // SAFETY: HighsInfo lives for the call; no other Rust reference
        // to it is held across a call into C++
        unsafe { &mut *self.c.info }
    }
    #[allow(clippy::mut_from_ref)]
    pub(crate) fn run_data(&self) -> &mut RunData {
        // SAFETY: as info
        unsafe { &mut *self.c.run_data }
    }
    pub(crate) fn get(&self, p: *mut bool) -> bool {
        // SAFETY: a flag of the Highs object
        unsafe { *p }
    }
    pub(crate) fn set(&self, p: *mut bool, v: bool) {
        // SAFETY: as get
        unsafe { *p = v }
    }
    /// highsLogUser would print
    pub(crate) fn on(&self) -> bool {
        // SAFETY: the C++ log options' flag
        !self.ab() && unsafe { *self.c.o.output_flag }
    }
    /// highsLogDev would print
    pub(crate) fn dev_on(&self) -> bool {
        // SAFETY: as on
        self.on() && unsafe { *self.c.o.log_dev_level } != 0
    }
    pub(crate) fn log(&self) -> &Log {
        &self.c.log
    }
    pub(crate) fn interpret(&self, call: Status, from: Status, message: &str) -> Status {
        if call != Status::Ok && self.dev_on() {
            self.log().interpret(call, from, message)
        } else {
            call.worse(from)
        }
    }

    /// Highs::invalidateSolution
    pub(crate) fn invalidate_solution(&self) {
        let info = self.info();
        info.primal_solution_status = SOLUTION_STATUS_NONE;
        info.dual_solution_status = SOLUTION_STATUS_NONE;
        info.num_primal_infeasibilities = ILLEGAL_INFEASIBILITY_COUNT;
        info.max_primal_infeasibility = ILLEGAL_INFEASIBILITY_MEASURE;
        info.sum_primal_infeasibilities = ILLEGAL_INFEASIBILITY_MEASURE;
        info.num_dual_infeasibilities = ILLEGAL_INFEASIBILITY_COUNT;
        info.max_dual_infeasibility = ILLEGAL_INFEASIBILITY_MEASURE;
        info.sum_dual_infeasibilities = ILLEGAL_INFEASIBILITY_MEASURE;
        self.set(self.c.value_valid, false);
        self.set(self.c.dual_valid, false);
    }
    pub(crate) fn invalidate_info(&self) {
        self.info().invalidate();
    }
    /// HighsRunData::invalidate
    pub(crate) fn invalidate_run_data(&self) {
        let r = self.run_data();
        r.valid = false;
        r.presolved_model_num_col = ILLEGAL_INT_MEASURE;
        r.presolved_model_num_row = ILLEGAL_INT_MEASURE;
        r.presolved_model_num_nz = ILLEGAL_INT_MEASURE;
        r.num_simplex_iterations_after_postsolve = ILLEGAL_INT_MEASURE;
        r.presolve_time = ILLEGAL_DOUBLE_MEASURE;
        r.solve_time = ILLEGAL_DOUBLE_MEASURE;
        r.postsolve_time = ILLEGAL_DOUBLE_MEASURE;
    }
    /// Highs::setHighsModelStatusAndClearSolutionAndBasis
    pub(crate) fn set_status_and_clear(&self, s: i32) {
        self.set_ms(s);
        self.invalidate_solution();
        self.op0(Op::InvalidateBasis);
        self.info().valid = true;
    }
    pub(crate) fn warn_solver_invalid(&self, problem: &[u8]) {
        // The options are unchanged since the start of the run
        // SAFETY: the C++ string lives for the call
        let solver = unsafe { self.c.o.solver.get() };
        super::options::warn_solver_invalid(self.log(), solver, problem);
    }

    /// The solveLp lambda of calledOptimizeModel: callSolveLp timed by
    /// the solve clock
    pub(crate) fn solve_lp(&self, which: i64, message: &str, time: &mut f64) -> Status {
        *time = -self.read(Clock::Solve);
        self.clock(Clock::Solve, START);
        let call_status = status_from_i64(self.op_msg(Op::CallSolveLp, which, message));
        if self.ab() {
            return Status::Error;
        }
        self.clock(Clock::Solve, STOP);
        *time += self.read(Clock::Solve);
        call_status
    }

    /// Highs::calledOptimizeModel
    pub fn called_optimize_model(&self) -> Status {
        let c = self.c;
        let log = self.log();
        // SAFETY: an option of the Highs object
        unsafe {
            if *c.o.highs_debug_level < 0 {
                *c.o.highs_debug_level = 0;
            }
        }
        // SIP and PAMI are not in Crestline: the simplex solver uses the
        // serial dual simplex for them (simplex/hekk.rs)
        if (c.o.simplex_strategy == 2 || c.o.simplex_strategy == 3) && self.on() {
            log_user!(
                log,
                LogType::Warning,
                "simplex_strategy = %d (%s) is not available in this build: using the serial dual simplex\n",
                c.o.simplex_strategy,
                if c.o.simplex_strategy == 2 { "SIP" } else { "PAMI" }
            );
        }
        if !c.o.use_warm_start {
            self.op0(Op::ClearSolver);
        }
        if !self.get(c.called_return) {
            if self.dev_on() {
                log_dev!(
                    log,
                    LogType::Error,
                    "Highs::optimizeModel() called with called_return_from_optimize_model false\n"
                );
            }
            return Status::Error;
        }
        let mut undo_mods = false;
        if self.facts(0).has_infinite_cost {
            let return_status = self.status_op(Op::HandleInfCost);
            if return_status != Status::Ok {
                self.set_status_and_clear(MS_UNKNOWN);
                return return_status;
            }
            undo_mods = true;
        }
        self.op0(Op::ExactResizeModel);
        if self.facts(0).is_mip && self.get(c.value_valid) {
            let call_status = self.status_op(Op::CompleteSolution);
            if self.ab() {
                return Status::Error;
            }
            if call_status != Status::Ok {
                return Status::Error;
            }
        }
        // From here all returns execute returnFromOptimizeModel()
        self.set(c.called_return, false);
        let mut return_status = Status::Ok;
        self.set_ms(MS_NOTSET);
        self.invalidate_info();
        self.invalidate_run_data();
        // zeroIterationCounts
        {
            let info = self.info();
            info.simplex_iteration_count = 0;
            info.ipm_iteration_count = 0;
            info.crossover_iteration_count = 0;
            info.pdlp_iteration_count = 0;
            info.qp_iteration_count = 0;
        }
        self.clock(Clock::Run, START);
        let facts = self.facts(0);
        if facts.num_col == 0 {
            self.set_status_and_clear(MS_MODEL_EMPTY);
            return self.return_from_optimize_model(Status::Ok, undo_mods);
        }
        if !self.infeasible_bounds_ok() {
            self.set_status_and_clear(MS_INFEASIBLE);
            return self.return_from_optimize_model(return_status, undo_mods);
        }
        self.op0(Op::EnsureColwise);
        if self.op0(Op::HasLargeValue) != 0 {
            if self.on() {
                log_user!(
                    log,
                    LogType::Error,
                    "Cannot solve a model with a |value| exceeding %g in constraint matrix\n",
                    c.o.large_matrix_value
                );
            }
            return self.return_from_optimize_model(Status::Error, undo_mods);
        }
        // SAFETY: an option of the Highs object
        if unsafe { *c.o.highs_debug_level } > 0 {
            return_status = self.status_op(Op::DebugAssess);
            if return_status == Status::Error {
                return self.return_from_optimize_model(return_status, undo_mods);
            }
        }
        if facts.model_name.len > 0 && self.dev_on() {
            // SAFETY: the C++ model name lives for the call
            let name = String::from_utf8_lossy(unsafe { facts.model_name.get() });
            log_dev!(log, LogType::Verbose, "Solving model: %s\n", &*name);
        }
        if !c.o.solve_relaxation {
            let mut made = false;
            let call_status = status_from_i64(self.op(
                Op::AssessSemiVariables,
                0,
                &mut made as *mut bool as *mut c_void,
            ));
            undo_mods = undo_mods || made;
            if call_status == Status::Error {
                self.set_status_and_clear(MS_SOLVE_ERROR);
                return self.return_from_optimize_model(Status::Error, undo_mods);
            }
        }
        if facts.is_qp {
            if facts.is_mip {
                if c.o.solve_relaxation {
                    undo_mods = self.op0(Op::RelaxSemiVariables) != 0 || undo_mods;
                } else {
                    if self.on() {
                        log_user!(log, LogType::Error, "Cannot solve MIQP problems with HiGHS\n");
                    }
                    return self.return_from_optimize_model(Status::Error, undo_mods);
                }
            }
            if self.op0(Op::OkHessianDiagonal) == 0 {
                if self.on() {
                    log_user!(log, LogType::Error, "Cannot solve non-convex QP problems with HiGHS\n");
                }
                return self.return_from_optimize_model(Status::Error, undo_mods);
            }
            if !self.f.solver_ok[2] {
                self.warn_solver_invalid(b"QP");
            }
            let call_status = self.status_op(Op::CallSolveQp);
            if self.ab() {
                return Status::Error;
            }
            return_status = self.interpret(call_status, return_status, "callSolveQp");
            return self.return_from_optimize_model(return_status, undo_mods);
        } else if facts.is_mip {
            if c.o.solve_relaxation {
                undo_mods = self.op0(Op::RelaxSemiVariables) != 0 || undo_mods;
                if self.on() {
                    log_user!(log, LogType::Info, "Solving LP relaxation since solve_relaxation is true\n");
                }
            } else {
                if !self.f.solver_ok[1] {
                    self.warn_solver_invalid(b"MIP");
                }
                let call_status = self.status_op(Op::CallSolveMip);
                if self.ab() {
                    return Status::Error;
                }
                return_status = self.interpret(call_status, return_status, "callSolveMip");
                return self.return_from_optimize_model(return_status, undo_mods);
            }
        }
        self.optimize_lp(return_status, undo_mods)
    }

    /// The LP part of calledOptimizeModel
    pub(crate) fn optimize_lp(&self, mut return_status: Status, undo_mods: bool) -> Status {
        let c = self.c;
        let log = self.log();
        let mut no_incumbent_lp_solution_or_basis = false;
        if !self.f.solver_ok[0] {
            self.warn_solver_invalid(b"LP");
        }
        let initial_time = self.read(Clock::Run);
        let mut this_presolve_time = -1.0;
        let mut this_solve_presolved_lp_time = -1.0;
        let mut this_postsolve_time = -1.0;
        let mut this_solve_original_lp_time = -1.0;
        let mut postsolve_iteration_count: i32 = -1;
        let lp_no_solution_basis = self.f.ipm_no_crossover || self.f.solver_pdlp;
        // iCrash is not in Crestline
        if c.o.icrash && self.on() {
            log_user!(log, LogType::Warning, "icrash = true: iCrash is not available in this build and is ignored\n");
        }
        let solver_will_use_basis = self.f.solver_will_use_basis;
        if solver_will_use_basis {
            if !self.get(c.basis_valid) && self.get(c.value_valid) {
                let call_status = self.status_op(Op::BasisForSolution);
                if self.ab() {
                    return Status::Error;
                }
                return_status = self.interpret(call_status, return_status, "basisForSolution");
                if return_status == Status::Error {
                    return self.return_from_optimize_model(return_status, undo_mods);
                }
            }
        } else {
            self.op0(Op::BasisClear);
        }
        let incumbent = self.facts(0);
        let unconstrained_lp = incumbent.num_nz == 0;
        let has_basis = self.get(c.basis_useful);
        let without_presolve = self.f.presolve_off;
        if (unconstrained_lp || has_basis || without_presolve) && solver_will_use_basis {
            let lp_solve = if unconstrained_lp {
                "Solving unconstrained LP"
            } else if has_basis {
                if without_presolve {
                    "Solving LP with useful basis"
                } else {
                    "Solving LP with useful basis so presolve not used"
                }
            } else {
                "Solving LP without presolve or useful basis"
            };
            self.op_msg(Op::SetEkkLpName, 0, lp_solve);
            if self.get(c.basis_useful) {
                self.op0(Op::RefineBasis);
            }
            let call_status = self.solve_lp(0, lp_solve, &mut this_solve_original_lp_time);
            if self.ab() {
                return Status::Error;
            }
            return_status = self.interpret(call_status, return_status, "callSolveLp");
            if return_status == Status::Error {
                return self.return_from_optimize_model(return_status, undo_mods);
            }
        } else {
            // SAFETY: an option of the Highs object
            let lp_presolve_requires_basis_postsolve = unsafe { *c.o.lp_presolve_requires_basis_postsolve };
            if lp_no_solution_basis {
                // SAFETY: as above
                unsafe { *c.o.lp_presolve_requires_basis_postsolve = false };
            }
            let from_presolve_time = self.read(Clock::Presolve);
            this_presolve_time = -from_presolve_time;
            self.clock(Clock::Presolve, START);
            let presolve_status = self.run_presolve(true, false);
            if self.ab() {
                return Status::Error;
            }
            // SAFETY: model_presolve_status_ of the Highs object
            unsafe { *c.presolve_status = presolve_status };
            self.clock(Clock::Presolve, STOP);
            let to_presolve_time = self.read(Clock::Presolve);
            this_presolve_time += to_presolve_time;
            self.op(Op::PresolveTime, 0, &mut this_presolve_time as *mut f64 as *mut c_void);
            self.run_data().presolve_time = this_presolve_time;
            // SAFETY: as above
            unsafe { *c.o.lp_presolve_requires_basis_postsolve = lp_presolve_requires_basis_postsolve };

            let mut factor_pivot_threshold = -1.0;
            let mut presolved_lp_pdlp_iteration_count = 0;
            let reduced = self.facts(1);
            let incumbent = self.facts(0);
            report_presolve_reductions(
                log,
                self.on(),
                presolve_status,
                [incumbent.num_col, incumbent.num_row, incumbent.num_nz],
                [reduced.num_col, reduced.num_row, reduced.num_nz],
            );
            {
                let r = self.run_data();
                r.presolved_model_num_col = reduced.num_col;
                r.presolved_model_num_row = reduced.num_row;
                r.presolved_model_num_nz = reduced.num_nz;
            }
            match presolve_status {
                PS_NOT_PRESOLVED | PS_NOT_REDUCED => {
                    let (name, message) = if presolve_status == PS_NOT_PRESOLVED {
                        ("Original LP", "Not presolved: solving the LP")
                    } else {
                        ("Unreduced LP", "Problem not reduced by presolve: solving the LP")
                    };
                    self.op_msg(Op::SetEkkLpName, 0, name);
                    let call_status = self.solve_lp(0, message, &mut this_solve_original_lp_time);
                    if self.ab() {
                        return Status::Error;
                    }
                    return_status = self.interpret(call_status, return_status, "callSolveLp");
                    if return_status == Status::Error {
                        return self.return_from_optimize_model(return_status, undo_mods);
                    }
                }
                PS_REDUCED => {
                    let call_status = self.status_op(Op::PrepareReducedLp);
                    // Ignore any warning from clean bounds
                    if self.interpret(call_status, return_status, "cleanBounds") == Status::Error {
                        return Status::Error;
                    }
                    self.op0(Op::EkkClear);
                    self.op_msg(Op::SetEkkLpName, 0, "Presolved LP");
                    // SAFETY: an option of the Highs object
                    let save_objective_bound = unsafe { *c.o.objective_bound };
                    unsafe { *c.o.objective_bound = INF };
                    let call_status =
                        self.solve_lp(1, "Solving the presolved LP", &mut this_solve_presolved_lp_time);
                        if self.ab() {
                            return Status::Error;
                        }
                    self.run_data().solve_time = this_solve_presolved_lp_time;
                    let mut threshold = 0.0;
                    if self.op(Op::EkkPivotThreshold, 0, &mut threshold as *mut f64 as *mut c_void) != 0 {
                        factor_pivot_threshold = threshold;
                    }
                    // SAFETY: as above
                    unsafe { *c.o.objective_bound = save_objective_bound };
                    return_status = self.interpret(call_status, return_status, "callSolveLp");
                    if return_status == Status::Error {
                        return self.return_from_optimize_model(return_status, undo_mods);
                    }
                    presolved_lp_pdlp_iteration_count = self.info().pdlp_iteration_count;
                    let ms = self.ms();
                    no_incumbent_lp_solution_or_basis = matches!(
                        ms,
                        MS_INFEASIBLE
                            | MS_UNBOUNDED
                            | MS_UNBOUNDED_OR_INFEASIBLE
                            | MS_TIME_LIMIT
                            | MS_ITERATION_LIMIT
                            | MS_INTERRUPT
                    );
                    if no_incumbent_lp_solution_or_basis {
                        self.op0(Op::EkkClear);
                        self.set_status_and_clear(ms);
                    }
                }
                PS_REDUCED_TO_EMPTY => {
                    self.op0(Op::ReducedToEmpty);
                    self.set_ms(MS_OPTIMAL);
                    self.run_data().solve_time = 0.0;
                }
                PS_INFEASIBLE => {
                    self.set_status_and_clear(MS_INFEASIBLE);
                    if self.on() {
                        log_user!(
                            log,
                            LogType::Info,
                            "Problem status detected on presolve: %s\n",
                            model_status_string(self.ms())
                        );
                    }
                    return self.return_from_optimize_model(return_status, undo_mods);
                }
                PS_UNBOUNDED_OR_INFEASIBLE => {
                    if self.on() {
                        log_user!(
                            log,
                            LogType::Info,
                            "Problem status detected on presolve: %s\n",
                            model_status_string(MS_UNBOUNDED_OR_INFEASIBLE)
                        );
                    }
                    if c.o.allow_unbounded_or_infeasible {
                        self.set_status_and_clear(MS_UNBOUNDED_OR_INFEASIBLE);
                        return self.return_from_optimize_model(return_status, undo_mods);
                    }
                    self.op0(Op::SaveOptions);
                    self.op0(Op::OptionsPrimalSimplex);
                    // The C++ ignores the status of this solve
                    self.solve_lp(
                        0,
                        "Solving the original LP with primal simplex to determine infeasible or unbounded",
                        &mut this_solve_original_lp_time,
                    );
                    if self.ab() {
                        return Status::Error;
                    }
                    self.op0(Op::RestoreOptions);
                    if return_status == Status::Error {
                        return self.return_from_optimize_model(return_status, undo_mods);
                    }
                    self.info().valid = true;
                    return self.return_from_optimize_model(return_status, undo_mods);
                }
                PS_TIMEOUT => {
                    self.set_status_and_clear(MS_TIME_LIMIT);
                    if self.dev_on() {
                        log_dev!(log, LogType::Warning, "Presolve reached timeout\n");
                    }
                    return self.return_from_optimize_model(Status::Warning, undo_mods);
                }
                PS_OUT_OF_MEMORY => {
                    self.set_status_and_clear(MS_MEMORY_LIMIT);
                    if self.on() {
                        log_user!(log, LogType::Error, "Presolve fails due to memory allocation error\n");
                    }
                    return self.return_from_optimize_model(Status::Error, undo_mods);
                }
                _ => {
                    self.set_status_and_clear(MS_PRESOLVE_ERROR);
                    if self.dev_on() {
                        log_dev!(log, LogType::Error, "Presolve returned status %d\n", presolve_status);
                    }
                    return self.return_from_optimize_model(Status::Error, undo_mods);
                }
            }
            // Postsolve
            if lp_no_solution_basis {
                self.op0(Op::InvalidateBasis);
            }
            let ms = self.ms();
            let have_optimal_reduced_solution = presolve_status == PS_REDUCED_TO_EMPTY
                || (presolve_status == PS_REDUCED && ms == MS_OPTIMAL);
            let have_unknown_reduced_solution = presolve_status == PS_REDUCED && ms == MS_UNKNOWN;
            if have_optimal_reduced_solution || have_unknown_reduced_solution {
                if have_unknown_reduced_solution && self.on() {
                    log_user!(
                        log,
                        LogType::Warning,
                        "Running postsolve on non-optimal solution of reduced LP\n\n"
                    );
                }
                self.op0(Op::CopyToPresolve);
                if presolve_status == PS_REDUCED {
                    self.op_msg(Op::KktCheck, 1, "Before postsolve");
                }
                this_postsolve_time = -self.read(Clock::Postsolve);
                self.clock(Clock::Postsolve, START);
                let postsolve_status = self.run_postsolve();
                if self.ab() {
                    return Status::Error;
                }
                self.clock(Clock::Postsolve, STOP);
                this_postsolve_time += self.read(Clock::Postsolve);
                self.op(Op::PresolveTime, 1, &mut this_postsolve_time as *mut f64 as *mut c_void);
                self.run_data().postsolve_time = this_postsolve_time;

                if postsolve_status == POSTSOLVE_SOLUTION_RECOVERED {
                    if self.on() {
                        log_user!(log, LogType::Info, "Performed postsolve\n");
                    }
                    self.op0(Op::TakeRecoveredSolution);
                    self.set(c.value_valid, true);
                    if !self.get(c.basis_valid) {
                        self.set(c.dual_valid, true);
                        self.op0(Op::InvalidateBasis);
                    } else {
                        self.set(c.dual_valid, true);
                        self.set(c.basis_valid, true);
                        self.set(c.basis_useful, true);
                        self.op0(Op::TakeRecoveredBasis);
                        if self.op0(Op::DebugPostsolveSolution) != 0 {
                            return self.return_from_optimize_model(Status::Error, undo_mods);
                        }
                        self.op0(Op::SaveOptions);
                        let mut threshold = factor_pivot_threshold;
                        self.op(
                            Op::OptionsCleanup,
                            (factor_pivot_threshold > 0.0) as i64,
                            &mut threshold as *mut f64 as *mut c_void,
                        );
                        self.op0(Op::RefineBasis);
                        self.op0(Op::EkkInvalidate);
                        self.op_msg(Op::SetEkkLpName, 0, "Postsolve LP");
                        postsolve_iteration_count = -self.info().simplex_iteration_count;
                        let call_status = self.solve_lp(
                            0,
                            "Solving the original LP from the solution after postsolve",
                            &mut this_solve_original_lp_time,
                        );
                        if self.ab() {
                            return Status::Error;
                        }
                        postsolve_iteration_count += self.info().simplex_iteration_count;
                        return_status = self.interpret(call_status, Status::Ok, "callSolveLp");
                        self.op0(Op::RestoreOptions);
                        if return_status == Status::Error {
                            return self.return_from_optimize_model(return_status, undo_mods);
                        }
                        self.run_data().num_simplex_iterations_after_postsolve = postsolve_iteration_count;
                        if postsolve_iteration_count > 0 && self.on() {
                            log_user!(
                                log,
                                LogType::Info,
                                "Required %d simplex iterations after postsolve\n",
                                postsolve_iteration_count
                            );
                        }
                    }
                } else {
                    if self.on() {
                        log_user!(log, LogType::Error, "Postsolve return status is %d\n", postsolve_status);
                    }
                    self.set_status_and_clear(MS_POSTSOLVE_ERROR);
                    return self.return_from_optimize_model(Status::Error, undo_mods);
                }
            } else {
                self.run_data().postsolve_time = 0.0;
            }
            // The PDLP clean-up (tryPdlpCleanup) is switched off in the C++
            let _ = presolved_lp_pdlp_iteration_count;
        }
        if !no_incumbent_lp_solution_or_basis {
            self.op_msg(Op::KktCheck, 0, "");
            self.info().valid = true;
        }
        if self.dev_on() {
            self.log_times(
                initial_time,
                this_presolve_time,
                this_solve_presolved_lp_time,
                this_postsolve_time,
                this_solve_original_lp_time,
                postsolve_iteration_count,
            );
        }
        return_status = status_from_model_status(self.ms());
        self.return_from_optimize_model(return_status, undo_mods)
    }

    /// The timing report at the end of calledOptimizeModel (dev log)
    pub(crate) fn log_times(&self, initial_time: f64, pre: f64, pre_lp: f64, post: f64, orig_lp: f64, post_iter: i32) {
        let log = self.log();
        let this_solve_time = self.read(Clock::Run) - initial_time;
        if post_iter < 0 {
            log_dev!(log, LogType::Info, "Postsolve  : \n");
        } else {
            log_dev!(log, LogType::Info, "Postsolve  : %d\n", post_iter);
        }
        if this_solve_time > 0.0 {
            log_dev!(log, LogType::Info, "Time           : %8.2f\n", this_solve_time);
        }
        if pre > 0.0 {
            log_dev!(log, LogType::Info, "Time Pre       : %8.2f\n", pre);
        }
        if pre_lp > 0.0 {
            log_dev!(log, LogType::Info, "Time PreLP     : %8.2f\n", pre_lp);
        }
        if orig_lp > 0.0 {
            log_dev!(log, LogType::Info, "Time OriginalLP: %8.2f\n", orig_lp);
        }
        if this_solve_time > 0.0 {
            let facts = self.facts(0);
            // SAFETY: the C++ model name lives for the call
            let name = String::from_utf8_lossy(unsafe { facts.model_name.get() });
            log_dev!(log, LogType::Info, "For LP %16s", &*name);
            let mut sum_time = 0.0;
            let pct = |t: f64| ((100.0 * t) / this_solve_time) as i32;
            if pre > 0.0 {
                sum_time += pre;
                log_dev!(log, LogType::Info, "    : Presolve %8.2f (%3d%%)", pre, pct(pre));
            }
            if pre_lp > 0.0 {
                sum_time += pre_lp;
                log_dev!(log, LogType::Info, "    : Solve presolved LP %8.2f (%3d%%)", pre_lp, pct(pre_lp));
            }
            if post > 0.0 {
                sum_time += post;
                log_dev!(log, LogType::Info, "    : Postsolve %8.2f (%3d%%)", post, pct(post));
            }
            if orig_lp > 0.0 {
                sum_time += orig_lp;
                log_dev!(log, LogType::Info, "    : Solve original LP %8.2f (%3d%%)", orig_lp, pct(orig_lp));
            }
            log_dev!(log, LogType::Info, "\n");
            let rlv_time_difference = (sum_time - this_solve_time).abs() / this_solve_time;
            if rlv_time_difference > 0.1 {
                log_dev!(
                    log,
                    LogType::Info,
                    "Strange: Solve time = %g; Sum times = %g: relative difference = %g\n",
                    this_solve_time,
                    sum_time,
                    rlv_time_difference
                );
            }
        }
    }

    /// Highs::infeasibleBoundsOk
    pub(crate) fn infeasible_bounds_ok(&self) -> bool {
        let mut lp = std::mem::MaybeUninit::<CLp>::uninit();
        self.op(Op::LpView, 0, lp.as_mut_ptr() as *mut c_void);
        // SAFETY: filled by C++ with model_.lp_'s arrays, which nothing
        // else touches during the call
        let lp = unsafe { lp.assume_init() };
        let o = &self.c.o;
        unsafe {
            infeasible_bounds_ok(
                self.log(),
                self.on(),
                [lp.col_lower.get_mut(), lp.col_upper.get_mut(), lp.row_lower.get_mut(), lp.row_upper.get_mut()],
                lp.integrality.get(),
                o.primal_feasibility_tolerance,
                o.mip_feasibility_tolerance,
                o.solve_relaxation,
            )
        }
    }

    /// Highs::runPresolve
    pub fn run_presolve(&self, force_lp_presolve: bool, force_presolve: bool) -> i32 {
        let c = self.c;
        let log = self.log();
        self.op0(Op::PresolveClear);
        if self.f.presolve_off && !force_presolve {
            return PS_NOT_PRESOLVED;
        }
        let facts = self.facts(0);
        if facts.is_empty {
            return PS_NOT_REDUCED;
        }
        self.op0(Op::EnsureColwise);
        if facts.num_col == 0 && facts.num_row == 0 {
            return PS_NULL_ERROR;
        }
        if self.clock(Clock::Run, RUNNING) == 0.0 {
            self.clock(Clock::Run, START);
        }
        let start_presolve = self.read(Clock::Run);
        let time_limit = c.o.time_limit;
        let limited = time_limit > 0.0 && time_limit < INF;
        if limited {
            let left = time_limit - start_presolve;
            if left <= 0.0 {
                if self.dev_on() {
                    log_dev!(log, LogType::Error, "Time limit reached while reading in matrix\n");
                }
                return PS_TIMEOUT;
            }
            if self.dev_on() {
                log_dev!(
                    log,
                    LogType::Verbose,
                    "Time limit set: reading matrix took %.2g, presolve time left: %.2g\n",
                    start_presolve,
                    left
                );
            }
        }
        let presolve_return_status = if facts.is_mip && !force_lp_presolve {
            self.op0(Op::MipPresolve) as i32
        } else {
            self.op0(Op::PresolveInit);
            if limited {
                let current = self.read(Clock::Run);
                let time_init = current - start_presolve;
                let left = time_limit - time_init;
                if left <= 0.0 {
                    if self.dev_on() {
                        log_dev!(log, LogType::Error, "Time limit reached while copying matrix into presolve.\n");
                    }
                    return PS_TIMEOUT;
                }
                if self.dev_on() {
                    log_dev!(
                        log,
                        LogType::Verbose,
                        "Time limit set: copying matrix took %.2g, presolve time left: %.2g\n",
                        time_init,
                        left
                    );
                }
            }
            self.op0(Op::PresolveRun) as i32
        };
        if self.ab() {
            return PS_NULL_ERROR;
        }
        if self.dev_on() {
            log_dev!(
                log,
                LogType::Verbose,
                "presolve_.run() returns status: %s\n",
                presolve_status_string(presolve_return_status)
            );
        }
        let status = self.op0(Op::PresolveLog) as i32;
        let original = self.facts(0);
        match status {
            PS_REDUCED => {
                let reduced = self.facts(1);
                let mut removed = [
                    original.num_col - reduced.num_col,
                    original.num_row - reduced.num_row,
                    original.num_nz - reduced.num_nz,
                ];
                self.op(Op::PresolveRemoved, 1, removed.as_mut_ptr() as *mut c_void);
            }
            PS_REDUCED_TO_EMPTY => {
                let mut removed = [original.num_col, original.num_row, original.num_nz];
                self.op(Op::PresolveRemoved, 0, removed.as_mut_ptr() as *mut c_void);
            }
            _ => {}
        }
        if !original.is_mip {
            self.op0(Op::ClearReducedIntegrality);
        }
        presolve_return_status
    }

    /// Highs::runPostsolve
    pub fn run_postsolve(&self) -> i32 {
        let validity = self.op0(Op::RecoveredValidity);
        if validity & 1 == 0 {
            return POSTSOLVE_NO_PRIMAL_SOLUTION_ERROR;
        }
        let have_dual_solution = validity & 2 != 0;
        self.op(Op::PostsolveUndo, have_dual_solution as i64, std::ptr::null_mut());
        if self.ab() {
            return POSTSOLVE_NO_PRIMAL_SOLUTION_ERROR;
        }
        self.op(Op::SetPostsolveStatus, POSTSOLVE_SOLUTION_RECOVERED as i64, std::ptr::null_mut());
        POSTSOLVE_SOLUTION_RECOVERED
    }

    /// Highs::returnFromOptimizeModel
    pub fn return_from_optimize_model(&self, run_return_status: Status, undo_mods: bool) -> Status {
        let c = self.c;
        let log = self.log();
        let ms = self.ms();
        let mut return_status = status_from_model_status(ms);
        if return_status != run_return_status && self.dev_on() {
            log_dev!(
                log,
                LogType::Error,
                "Highs::returnFromOptimizeModel: run_return_status = %d != %d = return_status = highsStatusFromHighsModelStatus(model_status_ = %s)\n",
                run_return_status as i32,
                return_status as i32,
                model_status_string(ms)
            );
        }
        match ms {
            MS_NOTSET | MS_LOAD_ERROR | MS_MODEL_ERROR | MS_PRESOLVE_ERROR | MS_SOLVE_ERROR
            | MS_POSTSOLVE_ERROR | MS_MEMORY_LIMIT | MS_MODEL_EMPTY => {
                self.invalidate_info();
                self.invalidate_run_data();
                self.invalidate_solution();
                self.op0(Op::InvalidateBasis);
            }
            MS_UNBOUNDED_OR_INFEASIBLE => {
                let facts_mip = self.facts(0).is_mip;
                if !(c.o.allow_unbounded_or_infeasible
                    || self.f.ipm_crossover_on
                    || self.f.solver_pdlp
                    || facts_mip)
                {
                    if self.on() {
                        log_user!(
                            log,
                            LogType::Error,
                            "returnFromHighs: HighsModelStatus::kUnboundedOrInfeasible is not permitted\n"
                        );
                    }
                    return_status = Status::Error;
                }
            }
            _ => {}
        }
        if self.op(Op::DebugReturn, 0, std::ptr::null_mut()) != 0 {
            return_status = Status::Error;
        }
        self.set(c.called_return, true);
        if undo_mods {
            return_status = status_from_i64(self.op(Op::UndoMods, return_status as i64, std::ptr::null_mut()));
        }
        let facts = self.facts(0);
        let solved_as_mip = facts.is_mip && !c.o.solve_relaxation;
        if !solved_as_mip {
            self.report_solved_lp_qp_stats(&facts);
        }
        self.return_from_highs(return_status)
    }

    /// Highs::reportSolvedLpQpStats
    pub(crate) fn report_solved_lp_qp_stats(&self, facts: &Facts) {
        if !self.on() {
            return;
        }
        let log = self.log();
        if facts.model_name.len > 0 {
            // SAFETY: the C++ model name lives for the call
            let name = String::from_utf8_lossy(unsafe { facts.model_name.get() });
            log_user!(log, LogType::Info, "Model name          : %s\n", &*name);
        }
        log_user!(log, LogType::Info, "Model status        : %s\n", model_status_string(self.ms()));
        let info = *self.info();
        if info.valid {
            if info.simplex_iteration_count != 0 {
                log_user!(log, LogType::Info, "Simplex   iterations: %d\n", info.simplex_iteration_count);
            }
            if info.ipm_iteration_count != 0 {
                log_user!(log, LogType::Info, "IPM       iterations: %d\n", info.ipm_iteration_count);
            }
            if info.crossover_iteration_count != 0 {
                log_user!(log, LogType::Info, "Crossover iterations: %d\n", info.crossover_iteration_count);
            }
            if info.pdlp_iteration_count != 0 {
                log_user!(log, LogType::Info, "PDLP      iterations: %d\n", info.pdlp_iteration_count);
            }
            if info.qp_iteration_count != 0 {
                log_user!(log, LogType::Info, "QP ASM    iterations: %d\n", info.qp_iteration_count);
            }
            log_user!(log, LogType::Info, "Objective value     : %17.10e\n", info.objective_function_value);
        }
        if self.get(self.c.dual_valid) {
            log_user!(log, LogType::Info, "P-D objective error : %17.10e\n", info.primal_dual_objective_error);
        }
        if !self.c.o.timeless_log {
            let run_time = self.read(Clock::Run);
            log_user!(log, LogType::Info, "HiGHS run time      : %13.2f\n", run_time);
        }
    }

    /// Highs::returnFromHighs
    pub fn return_from_highs(&self, highs_return_status: Status) -> Status {
        let log = self.log();
        let mut return_status = highs_return_status;
        self.op0(Op::ForceSolutionBasisSize);
        if self.op0(Op::BasisConsistent) == 0 {
            if self.on() {
                log_user!(
                    log,
                    LogType::Error,
                    "returnFromHighs: Supposed to be a HiGHS basis, but not consistent\n"
                );
            }
            return_status = Status::Error;
        }
        if self.op0(Op::RetainedEkkDataOk) == 0 {
            if self.on() {
                log_user!(log, LogType::Error, "returnFromHighs: Retained Ekk data not OK\n");
            }
            return_status = Status::Error;
        }
        if !self.get(self.c.called_return) && self.dev_on() {
            log_dev!(
                log,
                LogType::Error,
                "Highs::returnFromHighs() called with called_return_from_optimize_model false\n"
            );
        }
        if self.clock(Clock::Run, RUNNING) != 0.0 {
            self.clock(Clock::Run, STOP);
        }
        if self.op0(Op::LpDimensionsOk) == 0 {
            if self.dev_on() {
                log_dev!(log, LogType::Error, "LP Dimension error in returnFromHighs()\n");
            }
            return_status = Status::Error;
        }
        if self.op0(Op::EkkFactorCompatible) == 0 {
            if self.dev_on() {
                log_dev!(
                    log,
                    LogType::Warning,
                    "Highs::returnFromHighs(): LP and HFactor have inconsistent numbers of rows\n"
                );
            }
            self.op0(Op::EkkClear);
        }
        return_status
    }
}

/// reportPresolveReductions, from the (columns, rows, nonzeros) of the
/// LP and of the presolved LP
pub fn report_presolve_reductions(log: &Log, on: bool, presolve_status: i32, from: [i32; 3], presolved: [i32; 3]) {
    let (to, message) = match presolve_status {
        PS_NOT_REDUCED => (from, "- Not reduced"),
        PS_REDUCED => (presolved, ""),
        PS_TIMEOUT => (presolved, "- Timeout"),
        PS_REDUCED_TO_EMPTY => ([0, 0, 0], "- Reduced to empty"),
        _ => return,
    };
    if !on {
        return;
    }
    let [num_col_from, num_row_from, num_nz_from] = from;
    let [num_col_to, num_row_to, num_nz_to] = to;
    let mut nz_sign = "-";
    let mut delta_nz = num_nz_from.wrapping_sub(num_nz_to);
    if num_nz_to > num_nz_from {
        delta_nz = delta_nz.wrapping_neg();
        nz_sign = "+";
    }
    log_user!(
        log,
        LogType::Info,
        "Presolve reductions: rows %d(-%d); columns %d(-%d); nonzeros %d(%s%d) %s\n",
        num_row_to,
        num_row_from - num_row_to,
        num_col_to,
        num_col_from - num_col_to,
        num_nz_to,
        nz_sign,
        delta_nz,
        message
    );
}

struct BoundCounts {
    num_ok: i32,
    num_true: i32,
}

/// The infeasibleBoundOk lambda of Highs::infeasibleBoundsOk
#[allow(clippy::too_many_arguments)]
fn infeasible_bound_ok(
    log: &Log,
    on: bool,
    n: &mut BoundCounts,
    kind: &str,
    ix: usize,
    lower: &mut f64,
    upper: &mut f64,
    pft: f64,
    performed_inward_integer_rounding: bool,
) -> bool {
    let range = *upper - *lower;
    if range >= 0.0 {
        return true;
    }
    if range > -pft {
        n.num_ok += 1;
        let report = n.num_ok <= 10 && on;
        let integer_lower = *lower == (*lower + 0.5).floor();
        let integer_upper = *upper == (*upper + 0.5).floor();
        if integer_lower {
            if report {
                log_user!(
                    log,
                    LogType::Info,
                    "%s %d bounds [%g, %g] have infeasibility = %g so set upper bound to %g\n",
                    kind,
                    ix,
                    *lower,
                    *upper,
                    range,
                    *lower
                );
            }
            *upper = *lower;
        } else if integer_upper {
            if report {
                log_user!(
                    log,
                    LogType::Info,
                    "%s %d bounds [%g, %g] have infeasibility = %g so set lower bound to %g\n",
                    kind,
                    ix,
                    *lower,
                    *upper,
                    range,
                    *upper
                );
            }
            *lower = *upper;
        } else {
            let mid = 0.5 * (*lower + *upper);
            if report {
                log_user!(
                    log,
                    LogType::Info,
                    "%s %d bounds [%g, %g] have infeasibility = %g so set both bounds to %g\n",
                    kind,
                    ix,
                    *lower,
                    *upper,
                    range,
                    mid
                );
            }
            *lower = mid;
            *upper = mid;
        }
        return true;
    }
    n.num_true += 1;
    if n.num_true <= 10 && on {
        log_user!(
            log,
            LogType::Info,
            "%s %d bounds [%g, %g] have excessive infeasibility = %g%s\n",
            kind,
            ix,
            *lower,
            *upper,
            range,
            if performed_inward_integer_rounding { " due to inward integer rounding" } else { "" }
        );
    }
    false
}

/// Highs::infeasibleBoundsOk on model_.lp_'s column and row bounds
/// (lower, upper, lower, upper) and integrality: bounds inconsistent by
/// less than the primal feasibility tolerance are made consistent
pub fn infeasible_bounds_ok(
    log: &Log,
    on: bool,
    bounds: [&mut [f64]; 4],
    integrality: &[u8],
    pft: f64,
    mft: f64,
    solve_relaxation: bool,
) -> bool {
    use super::var_type::{INTEGER, SEMI_CONTINUOUS, SEMI_INTEGER};
    let [col_lower, col_upper, row_lower, row_upper] = bounds;
    let mut n = BoundCounts { num_ok: 0, num_true: 0 };
    let has_integrality = !integrality.is_empty();
    let perform_inward_integer_rounding = !solve_relaxation;
    for i in 0..col_lower.len() {
        let mut performed = false;
        let mut lower = col_lower[i];
        let mut upper = col_upper[i];
        if has_integrality {
            if integrality[i] == SEMI_CONTINUOUS || integrality[i] == SEMI_INTEGER {
                continue;
            }
            if perform_inward_integer_rounding && integrality[i] == INTEGER {
                let integer_lower = (lower - mft).ceil();
                let integer_upper = (upper + mft).floor();
                performed = integer_lower > lower || integer_upper < upper;
                lower = integer_lower;
                upper = integer_upper;
            }
        }
        if lower > upper && infeasible_bound_ok(log, on, &mut n, "Column", i, &mut lower, &mut upper, pft, performed)
        {
            col_lower[i] = lower;
            col_upper[i] = upper;
        }
    }
    for i in 0..row_lower.len() {
        if row_lower[i] > row_upper[i] {
            infeasible_bound_ok(log, on, &mut n, "Row", i, &mut row_lower[i], &mut row_upper[i], pft, false);
        }
    }
    if n.num_ok > 0 && on {
        log_user!(log, LogType::Info, "Model has %d small inconsistent bound(s): rectified\n", n.num_ok);
    }
    if n.num_true > 0 && on {
        log_user!(log, LogType::Info, "Model has %d significant inconsistent bound(s): infeasible\n", n.num_true);
    }
    n.num_true == 0
}

// The C++ entry points (highs/lp_data/HighsRunRust.cpp)

/// # Safety
/// `c` must be a valid CHighs
#[no_mangle]
pub unsafe extern "C" fn highs_rs_called_optimize_model(c: *const CHighs) -> i32 {
    Run::new(&*c).called_optimize_model() as i32
}

/// # Safety
/// As highs_rs_called_optimize_model
#[no_mangle]
pub unsafe extern "C" fn highs_rs_return_from_optimize_model(c: *const CHighs, status: i32, undo_mods: bool) -> i32 {
    Run::new(&*c).return_from_optimize_model(status_from_i64(status as i64), undo_mods) as i32
}

/// # Safety
/// As highs_rs_called_optimize_model
#[no_mangle]
pub unsafe extern "C" fn highs_rs_return_from_highs(c: *const CHighs, status: i32) -> i32 {
    Run::new(&*c).return_from_highs(status_from_i64(status as i64)) as i32
}

/// # Safety
/// As highs_rs_called_optimize_model
#[no_mangle]
pub unsafe extern "C" fn highs_rs_run_presolve(c: *const CHighs, force_lp_presolve: bool, force_presolve: bool) -> i32 {
    Run::new(&*c).run_presolve(force_lp_presolve, force_presolve)
}

/// # Safety
/// As highs_rs_called_optimize_model
#[no_mangle]
pub unsafe extern "C" fn highs_rs_run_postsolve(c: *const CHighs) -> i32 {
    Run::new(&*c).run_postsolve()
}

/// # Safety
/// `log` valid, `from` and `presolved` three values each
#[no_mangle]
pub unsafe extern "C" fn highs_rs_report_presolve_reductions(
    log: *const Log,
    on: bool,
    presolve_status: i32,
    from: *const i32,
    presolved: *const i32,
) {
    let a = |p: *const i32| [*p, *p.add(1), *p.add(2)];
    report_presolve_reductions(&*log, on, presolve_status, a(from), a(presolved));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_maps() {
        assert_eq!(status_from_model_status(MS_OPTIMAL), Status::Ok);
        assert_eq!(status_from_model_status(MS_UNKNOWN), Status::Warning);
        assert_eq!(status_from_model_status(MS_NOTSET), Status::Error);
        assert_eq!(status_from_model_status(99), Status::Error);
        assert_eq!(presolve_status_string(PS_OUT_OF_MEMORY), "Memory allocation error");
    }

    #[test]
    fn bounds_rectified() {
        let mut cl = [1.0, 0.3, 2.0, 0.0, 0.5];
        let mut cu = [1.0 - 1e-9, 0.3 - 1e-9, 1.0, 0.0, 0.7];
        let mut rl = [1.0 + 1e-9];
        let mut ru = [1.0];
        let integrality = [0, 0, 0, 3, 1];
        let ok = infeasible_bounds_ok(
            &Log::none(),
            false,
            [&mut cl, &mut cu, &mut rl, &mut ru],
            &integrality,
            1e-7,
            1e-6,
            false,
        );
        // Column 2 is infeasible; column 4 only after inward rounding,
        // which leaves the model's bounds
        assert!(!ok);
        assert_eq!((cl[0], cu[0]), (1.0, 1.0));
        assert_eq!(cl[1], cu[1]);
        assert_eq!((cl[2], cu[2]), (2.0, 1.0));
        assert_eq!((cl[4], cu[4]), (0.5, 0.7));
        assert_eq!((rl[0], ru[0]), (1.0, 1.0));
    }
}
