/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file lp_data/HighsRunRust.cpp
 * @brief Highs::run's control flow done by Rust (rust/src/lp_data/run.rs):
 * calledOptimizeModel, runPresolve, runPostsolve, returnFromOptimizeModel
 * and returnFromHighs. Rust calls back here for each step on a C++ object.
 */
#include "lp_data/HighsRust.h"

#ifdef HIGHS_RUST
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>

#include "Highs.h"
#include "io/Filereader.h"
#include "lp_data/HighsInfoDebug.h"
#include "lp_data/HighsModelUtils.h"
#include "lp_data/HighsSolutionDebug.h"
#include "lp_data/HighsSolution.h"
#include "mip/HighsMipHost.h"
#include "mip/MipTimer.h"
#include "parallel/HighsParallel.h"
#include "model/HighsHessianUtils.h"
#include "simplex/HSimplex.h"
#include "util/HighsMatrixPic.h"

static_assert(sizeof(HighsRunDataStruct) == 48, "HighsRunDataStruct layout");
static_assert(sizeof(HighsModelStatus) == 4, "HighsModelStatus is an int");
static_assert(sizeof(HighsPresolveStatus) == 4,
              "HighsPresolveStatus is an int");

namespace {

struct RsRunStr {
  const char* ptr;
  size_t len;
};

// run.rs: ROptions
struct RsRunOptions {
  RsRunStr solver, run_crossover, presolve;
  bool use_warm_start, icrash, solve_relaxation, allow_unbounded_or_infeasible,
      timeless_log;
  double large_matrix_value, time_limit, primal_feasibility_tolerance,
      mip_feasibility_tolerance;
  HighsInt* highs_debug_level;
  double* objective_bound;
  bool* lp_presolve_requires_basis_postsolve;
  const bool* output_flag;
  const HighsInt* log_dev_level;
  HighsInt simplex_strategy;
};

// run.rs: CHighs
struct RsHighs {
  RsLog log;
  void* ctx;
  int64_t (*op)(void*, int, int64_t, void*, const char*, size_t);
  double (*clock)(void*, int, int);
  HighsModelStatus* model_status;
  HighsPresolveStatus* presolve_status;
  HighsInfoStruct* info;
  HighsRunDataStruct* run_data;
  bool *value_valid, *dual_valid, *basis_valid, *basis_alien, *basis_useful,
      *basis_was_alien, *called_return;
  RsRunOptions o;
};

// run.rs: Facts
struct RsFacts {
  HighsInt num_col, num_row, num_nz;
  bool is_mip, is_qp, is_empty, has_infinite_cost;
  RsRunStr model_name;
};

// run.rs: Op
enum class RunOp {
  kClearSolver = 1,
  kHandleInfCost,
  kExactResizeModel,
  kCompleteSolution,
  kInvalidateInfo,
  kInvalidateRunData,
  kInvalidateBasis,
  kFacts,
  kEnsureColwise,
  kHasLargeValue,
  kDebugAssess,
  kAssessSemiVariables,
  kRelaxSemiVariables,
  kOkHessianDiagonal,
  kCallSolveQp,
  kCallSolveMip,
  kBasisForSolution,
  kBasisClear,
  kRefineBasis,
  kCallSolveLp,
  kSetEkkLpName,
  kPrepareReducedLp,
  kEkkClear,
  kEkkInvalidate,
  kEkkPivotThreshold,
  kReducedToEmpty,
  kSaveOptions,
  kRestoreOptions,
  kOptionsPrimalSimplex,
  kOptionsCleanup,
  kCopyToPresolve,
  kKktCheck,
  kRecoveredValidity,
  kPostsolveUndo,
  kSetPostsolveStatus,
  kTakeRecoveredSolution,
  kTakeRecoveredBasis,
  kDebugPostsolveSolution,
  kUndoMods,
  kDebugReturn,
  kForceSolutionBasisSize,
  kBasisConsistent,
  kRetainedEkkDataOk,
  kLpDimensionsOk,
  kEkkFactorCompatible,
  kPresolveClear,
  kMipPresolve,
  kPresolveInit,
  kPresolveRun,
  kPresolveLog,
  kPresolveRemoved,
  kClearReducedIntegrality,
  kPresolveTime,
  kLpView,
  // drivers.rs
  kMipRun,
  kMipTakeSolution,
  kActiveModifiedUpperBounds,
  kSwapPrimalTolerance,
  kKktFailures,
  kMipFinish,
  kSolutionHasUndefined,
  kSolutionFeasible,
  kSaveColBounds,
  kClearIntegrality,
  kSolutionClear,
  kSwapMipMaxNodes,
  kOptimizeModel,
  kSolutionView,
  kRayRecord,
  kFeasibilityProblem,
  kUnboundednessProblem,
  kHighsRun,
  kCopyRay,
  kComputeDualRay,
  kComputePrimalRay,
  kNeedsMods,
  kReportModelStats,
  kClearPresolve,
  kInitializeMultiThreading,
  kPresolveProfiled,
  kReportPresolveReductions,
  kPresolvedModel,
  kCrossover,
  kPostsolveArgs,
  kPostsolveBasisConsistent,
  kPostsolveSetSolution,
  kPostsolveKkt,
  kPostsolveTakeRecovered,
  kOptionsPostsolveCleanup,
  kSetBasis,
  kSetBasisOrigin,
  kBasisDebug,
  kNewHighsBasis,
  kHessianDims,
  kAssessHessian,
  kHessianClear,
  kCompleteHessian,
  kLogHeader,
  kClearModel,
  kTakeModel,
  kEmptyMatrix,
  kFormatOk,
  kPrepareModelLp,
  kAssessLp,
  kMatrixImages,
  kClearSolver2,
  kTakeHessian,
  kReadModelFile,
  kReadModelPass,
  kReadBasis,
  kWriteModelPrepare,
  kWriteModelLpView,
  kWriteModelCheck,
  kReportWrittenModel,
  kWriteModelFile,
  kWriteBasis,
  kSolutionBasisSizes,
  // lp_run.rs
  kLpRustBegin,
  kLpRustEnd,
  kKktOptions,
  kSolveTemplate,
  kUnconstrainedTemplate,
  kIpxTemplate,
  kPdlpTemplate,
  kPdlpProfiling,
  kSimplexTemplate,
  kSimplexShell,
  kSetInterrupt,
  kPresolveOptions,
  kAssessSmallValues,
  kLpOptions,
};

// lp_presolve.rs: CPresolveExport
struct RsPresolveExport {
  HighsInt status;
  bool prepared;
  const HighsInt (*log)[3];
  size_t num_log;
  const char* data;
  size_t data_len;
  const void* reductions;
  size_t num_reductions;
  const HighsInt* orig_col_index;
  size_t num_col;
  const HighsInt* orig_row_index;
  size_t num_row;
  const uint8_t* linearly_transformable;
  size_t num_lt;
  HighsInt orig_num_col, orig_num_row;
};


// lp_run.rs: CRunData, the Highs object's solution, basis, info and model
// status in place
struct RsRunData {
  RsVec<double> col_value, col_dual, row_value, row_dual;
  bool *value_valid, *dual_valid;
  RsBasisVec* basis;
  RsMut<uint8_t> origin;
  void* origin_ctx;
  void (*set_origin)(void*, const uint8_t*, size_t);
  HighsInfoStruct* info;
  HighsModelStatus* model_status;
};

// lp_run.rs: UnconTemplate
struct RsUnconTemplate {
  bool on;
  double primal_feasibility_tolerance, dual_feasibility_tolerance;
};

void setOrigin(void* ctx, const uint8_t* p, size_t n) {
  static_cast<std::string*>(ctx)->assign(reinterpret_cast<const char*>(p), n);
}

// drivers.rs: BasisDebug
struct RsBasisDebug {
  HighsInt id, update_count;
  RsRunStr origin;
};

// drivers.rs: PostsolveArgs
struct RsPostsolveArgs {
  int64_t col_value_size, col_dual_size, row_dual_size;
  bool dual_valid;
  int64_t basis_col_size, basis_row_size;
  bool basis_valid;
};

// drivers.rs: MipResult
struct RsMipResult {
  HighsInt model_status;
  double solution_objective;
  int64_t node_count, total_lp_iterations;
  double dual_bound, gap, primal_dual_integral, row_violation,
      bound_violation, integrality_violation;
};

// drivers.rs: RayRecord
struct RsRayRecord {
  HighsInt index, sign;
  int64_t value_size;
  bool has_invert;
};

RsRunStr rsRunStr(const std::string& s) { return {s.data(), s.size()}; }

}  // namespace

extern "C" {
int highs_rs_return_from_highs(const RsHighs* h, int status);
bool highs_rs_lps_presolve_export(void* lps, RsLpVec* lp, RsPresolveExport* out);
}

namespace {
// top.rs: CTopData, the Highs object's data in place
struct RsTopData {
  RsRunData run;
  HighsRunDataStruct* run_data;
  HighsPresolveStatus* presolve_status;
  bool* called_return;
  RsHessian hessian;
  void* profiling;
};
// top.rs: PresolveRecords
struct RsPresolveRecords {
  int postsolve_status;
  HighsInt n_cols_removed, n_rows_removed, n_nnz_removed;
  double presolve_time, postsolve_time;
};
// top.rs: what changed for the C++ copies
enum : uint32_t {
  kXModel = 1,
  kXHessian = 2,
  kXClearIis = 4,
  kXClearDerived = 8,
  kXPresolveClear = 16,
  kXPresolve = 32,
  kXSaved = 64,
  kXPresolveRecords = 128,
  kXClearPresolve = 256,
  kXPresolvedModel = 512,
  kXClearModel = 1024,
  kXTakeModel = 2048,
  kXPresolveMip = 4096,
};
}  // namespace

extern "C" {
void highs_rs_lph_top_import(highs_rs::LpHandle* p, const RsTopData* d,
                             bool run);
bool highs_rs_lph_top_run_matches(highs_rs::LpHandle* p, const RsTopData* d);
uint32_t highs_rs_lph_top_changed(highs_rs::LpHandle* p);
void highs_rs_lph_top_export(highs_rs::LpHandle* p, RsTopData* d);
RsPresolveRecords highs_rs_lph_top_records(highs_rs::LpHandle* p);
size_t highs_rs_lph_top_saved(highs_rs::LpHandle* p, size_t k,
                              double* objective, size_t* n);
void highs_rs_lph_top_clear_saved(highs_rs::LpHandle* p);
int highs_rs_lph_called_optimize_model(highs_rs::LpHandle* p,
                                       bool* interrupted);
void highs_rs_lph_top_hessian(highs_rs::LpHandle* p, RsHessian* h);
int highs_rs_lph_top_presolved_which(highs_rs::LpHandle* p);
int highs_rs_lph_top_presolve(highs_rs::LpHandle* p, bool* interrupted);
int highs_rs_lph_top_postsolve(highs_rs::LpHandle* p, const RsSolution* s,
                               const RsBasisVec* b, const char* origin,
                               size_t origin_len, bool* interrupted);
int highs_rs_lph_top_crossover(highs_rs::LpHandle* p, const RsSolution* s,
                               bool* interrupted);
int highs_rs_lph_top_set_basis(highs_rs::LpHandle* p, const RsBasisVec* b,
                               const char* basis_origin, size_t basis_origin_len,
                               const char* origin, size_t origin_len);
int highs_rs_lph_top_pass_model(highs_rs::LpHandle* p, int which,
                                const RsLp* lp, const char* name,
                                size_t name_len, const RsHessian* hessian);
}

// ---- The simplex engine of a Highs object (lp_handle.rs on a host)

// The profiling steps of a simplex solve on a HighsProfiling (code 1
// start, 2 stop, arg whether there is a basis) and of PDLP (3 start, 4
// stop)
void rsSimplexProfiling(void* profiling_p, int code,
                               HighsInt simplex_strategy, int64_t arg) {
  HighsProfiling* profiling = static_cast<HighsProfiling*>(profiling_p);
  switch (code) {
    case 1:
      // arg: whether the HiGHS basis is valid
      if (profiling) {
        HighsInt profiling_clock = -1;
        if (simplex_strategy == kSimplexStrategyPrimal) {
          profiling_clock =
              arg ? kSubSolverPrSimplexBasis : kSubSolverPrSimplexNoBasis;
        } else {
          profiling_clock =
              arg ? kSubSolverDuSimplexBasis : kSubSolverDuSimplexNoBasis;
        }
        profiling->start(profiling_clock);
      }
      return;
    case 2:
      if (profiling->sub_solver_) {
        HighsInt profiling_clock = -1;
        HighsProfilingRecord* thread_record =
            profiling->getHighsProfilingRecord();
        if (std::signbit(thread_record->start_time[kSubSolverDuSimplexBasis]))
          profiling_clock = kSubSolverDuSimplexBasis;
        if (std::signbit(
                thread_record->start_time[kSubSolverDuSimplexNoBasis]))
          profiling_clock = kSubSolverDuSimplexNoBasis;
        if (std::signbit(thread_record->start_time[kSubSolverPrSimplexBasis]))
          profiling_clock = kSubSolverPrSimplexBasis;
        if (std::signbit(
                thread_record->start_time[kSubSolverPrSimplexNoBasis]))
          profiling_clock = kSubSolverPrSimplexNoBasis;
        profiling->stop(profiling_clock);
      }
      return;
    case 3:
      profiling->start(kSubSolverPdlp);
      return;
    case 4:
      profiling->stop(kSubSolverPdlp);
      return;
  }
}

// The top level's host functions (rust/src/lp_data/top.rs CTop), after
// HighsRunRust below
static int64_t rsTopOp(void* ctx, int code, int64_t arg, void* p);
static double rsTopClock(void* ctx, int which, int action);

highs_rs::LphHost rsEngineHost(Highs* highs) {
  // The profiling steps of the engine's (and the MIP's) simplex solves
  highs_rs::highs_rs_lph_register(rsSimplexProfiling);
  highs_rs::LphHost host;
  host.ctx = highs;
  host.log_options = &highs->options_.log_options;
  host.timer_read = [](void* ctx) {
    return static_cast<Highs*>(ctx)->timer_.read();
  };
  // The user interrupt part of HEkk::bailout (iteration_count < 0: whether
  // the callback is active)
  host.simplex_interrupt = [](void* ctx, HighsInt iteration_count) {
    Highs& h = *static_cast<Highs*>(ctx);
    HighsCallback& callback = h.callback_;
    if (iteration_count < 0)
      return bool(callback.user_callback &&
                  callback.active[kCallbackSimplexInterrupt]);
    callback.clearHighsCallbackOutput();
    callback.data_out.simplex_iteration_count = iteration_count;
    if (callback.callbackAction(kCallbackSimplexInterrupt,
                                "Simplex interrupt")) {
      highsLogDev(h.options_.log_options, HighsLogType::kInfo,
                  "User interrupt\n");
      return true;
    }
    return false;
  };
  host.ipm_interrupt = [](void* ctx, HighsInt ipm_iteration_count) {
    HighsCallback& callback = static_cast<Highs*>(ctx)->callback_;
    if (callback.user_callback && callback.active[kCallbackIpmInterrupt]) {
      callback.clearHighsCallbackOutput();
      callback.data_out.ipm_iteration_count = ipm_iteration_count;
      if (callback.callbackAction(kCallbackIpmInterrupt, "IPM interrupt"))
        return HighsInt{1};
    }
    return HighsInt{0};
  };
  host.top_op = rsTopOp;
  host.clock = rsTopClock;
  return host;
}

// The engine's model is a copy of `lp`
// `lp` takes the engine model's scale factors (and its matrix, if the run
// rebuilt it: HEkk::lpBack)
static void rsModelBack(highs_rs::LpHandle* e, HighsLp& lp) {
  RsLp v;
  const char* name;
  size_t len;
  highs_rs::highs_rs_lph_model(e, &v, &name, &len);
  HighsScale& scale = lp.scale_;
  scale.strategy = v.scale_strategy;
  scale.has_scaling = v.scale_has_scaling;
  scale.num_col = v.scale_num_col;
  scale.num_row = v.scale_num_row;
  scale.cost = v.scale_cost;
  scale.col.assign(v.scale_col.ptr, v.scale_col.ptr + v.scale_col.len);
  scale.row.assign(v.scale_row.ptr, v.scale_row.ptr + v.scale_row.len);
  lp.is_scaled_ = v.is_scaled;
  if (highs_rs::highs_rs_lph_take_matrix_back(e)) {
    HighsSparseMatrix& a = lp.a_matrix_;
    a.format_ = MatrixFormat(v.a.format);
    a.num_col_ = v.a.num_col;
    a.num_row_ = v.a.num_row;
    a.start_.assign(v.a.start.ptr, v.a.start.ptr + v.a.start.len);
    a.p_end_.assign(v.a.p_end.ptr, v.a.p_end.ptr + v.a.p_end.len);
    a.index_.assign(v.a.index.ptr, v.a.index.ptr + v.a.index.len);
    a.value_.assign(v.a.value.ptr, v.a.value.ptr + v.a.value.len);
  }
}

HighsStatus rsFormBasis(Highs& h, HighsBasis& basis,
                        const bool only_from_known_basis) {
  HighsEngine& ekk = h.ekk_instance_;
  h.model_w().lp_.ensureColwise();
  rsSyncOptions(ekk.p, h.options_);
  h.lpToRust();
  highs_rs::LphFormBasis b;
  b.valid = basis.valid;
  b.useful = basis.useful;
  b.alien = &basis.alien;
  b.col_status = {reinterpret_cast<uint8_t*>(basis.col_status.data()),
                  basis.col_status.size()};
  b.row_status = {reinterpret_cast<uint8_t*>(basis.row_status.data()),
                  basis.row_status.size()};
  b.debug_id = basis.debug_id;
  b.debug_update_count = basis.debug_update_count;
  b.origin = {const_cast<char*>(basis.debug_origin_name.data()),
              basis.debug_origin_name.size()};
  const HighsStatus status = HighsStatus(
      highs_rs::highs_rs_lph_form_basis(ekk.p, &b, only_from_known_basis));
  // Both copies of the model take the scale factors
  rsModelBack(ekk.p, h.model_cache_.lp_);
  return status;
}

void Highs::lpSetScalars(const ObjSense sense, const double offset) {
  model_cache_.lp_.sense_ = sense;
  model_cache_.lp_.offset_ = offset;
  if (!lp_cpp_newer_)
    highs_rs::highs_rs_lph_set_model_scalars(ekk_instance_.p, int(sense),
                                             offset);
}

void Highs::lpSetName(const std::string& name) {
  model_cache_.lp_.model_name_ = name;
  if (!lp_cpp_newer_)
    highs_rs::highs_rs_lph_set_model_name(ekk_instance_.p, name.data(),
                                          name.size());
}

void Highs::lpFromRust() {
  HighsLp& lp = model_cache_.lp_;
  RsLpVec v = rsLpVec(lp);
  const char* name;
  size_t len;
  highs_rs::highs_rs_lph_export_model(ekk_instance_.p, &v, &name, &len);
  rsLpVecBack(v, lp);
  lp.model_name_.assign(name, len);
  lp_cpp_newer_ = false;
}

// HIGHS_RS_CHECK_SYNC set: lpToRust checks that an engine model it does
// not import is the C++ model's copy
static const bool kRsCheckSync = std::getenv("HIGHS_RS_CHECK_SYNC") != nullptr;

void Highs::lpToRust() {
  const HighsLp& lp = model_cache_.lp_;
  if (!lp_cpp_newer_) {
    if (kRsCheckSync) {
      const RsLp v = rsLp(lp);
      if (!highs_rs::highs_rs_lph_model_matches(ekk_instance_.p, &v,
                                                lp.model_name_.data(),
                                                lp.model_name_.size()))
        std::abort();
    }
    return;
  }
  const RsLp v = rsLp(lp);
  highs_rs::highs_rs_lph_import_model(ekk_instance_.p, &v, lp.model_name_.data(),
                                      lp.model_name_.size());
  lp_cpp_newer_ = false;
}

void HighsEngine::setNlaPointersForLpAndScale(const HighsLp& lp) {
  const RsLp v = rsLp(lp);
  highs_rs::highs_rs_lph_set_nla_lp(p, &v);
}

double HighsEngine::computeBasisCondition(const HighsLp& lp, const bool exact,
                                          const bool report) const {
  const RsLp v = rsLp(lp);
  return highs_rs::highs_rs_lph_basis_condition(
      p, &v, lp.model_name_.data(), lp.model_name_.size(), exact, report);
}

HighsBasis HighsEngine::getHighsBasis(const HighsLp& use_lp) const {
  HighsBasis highs_basis;
  highs_basis.col_status.resize(use_lp.num_col_);
  highs_basis.row_status.resize(use_lp.num_row_);
  assert(status_.has_basis);
  const RsLp lp = rsLp(use_lp);
  const char* origin;
  size_t origin_len;
  highs_rs::highs_rs_lps_get_highs_basis(
      lps, &lp, 0, reinterpret_cast<uint8_t*>(highs_basis.col_status.data()),
      reinterpret_cast<uint8_t*>(highs_basis.row_status.data()),
      &highs_basis.debug_id, &highs_basis.debug_update_count, &origin,
      &origin_len);
  highs_basis.debug_origin_name.assign(origin, origin_len);
  highs_basis.valid = true;
  highs_basis.alien = false;
  highs_basis.useful = true;
  highs_basis.was_alien = false;
  return highs_basis;
}

const HotStart& HighsEngine::hotStart() const {
  bool use;
  const HighsInt *pivot_row, *pivot_var;
  const int8_t *pivot_type, *nonbasic_move;
  int num_pivot, num_move;
  double build_synthetic_tick;
  if (highs_rs::highs_rs_lph_hot_start(p, &use, &pivot_row, &pivot_var,
                                       &pivot_type, &num_pivot,
                                       &build_synthetic_tick, &nonbasic_move,
                                       &num_move)) {
    RefactorInfo& refactor_info = hot_start_.refactor_info;
    refactor_info.use = use;
    refactor_info.pivot_row.assign(pivot_row, pivot_row + num_pivot);
    refactor_info.pivot_var.assign(pivot_var, pivot_var + num_pivot);
    refactor_info.pivot_type.assign(pivot_type, pivot_type + num_pivot);
    refactor_info.build_synthetic_tick = build_synthetic_tick;
    hot_start_.nonbasicMove.assign(nonbasic_move, nonbasic_move + num_move);
    hot_start_.valid = true;
  }
  return hot_start_;
}

const std::vector<double>& HighsEngine::primalPhase1Dual() const {
  size_t n;
  const double* v = highs_rs::highs_rs_lph_primal_phase1_dual(p, &n);
  primal_phase1_dual_.assign(v, v + n);
  return primal_phase1_dual_;
}

void HighsSimplexStats::report(FILE* file, std::string message) const {
  fprintf(file, "\nSimplex stats: %s\n", message.c_str());
  fprintf(file, "   valid                      = %d\n", this->valid);
  fprintf(file, "   iteration_count            = %d\n",
          static_cast<int>(this->iteration_count));
  fprintf(file, "   num_invert                 = %d\n",
          static_cast<int>(this->num_invert));
  fprintf(file, "   last_invert_num_el         = %d\n",
          static_cast<int>(this->last_invert_num_el));
  fprintf(file, "   last_factored_basis_num_el = %d\n",
          static_cast<int>(this->last_factored_basis_num_el));
  fprintf(file, "   col_aq_density             = %g\n", this->col_aq_density);
  fprintf(file, "   row_ep_density             = %g\n", this->row_ep_density);
  fprintf(file, "   row_ap_density             = %g\n", this->row_ap_density);
  fprintf(file, "   row_DSE_density            = %g\n", this->row_DSE_density);
}

static_assert(sizeof(HighsSimplexStats) == 56,
              "HighsSimplexStats is SimplexStats in lp_handle.rs");
static_assert(sizeof(highs_rs::HEkkShared) == 144,
              "EkkShared in lp_solver.rs");

// The steps of the run on the Highs object (a friend)
struct HighsRunRust {
  explicit HighsRunRust(Highs& highs) : h(highs) {}
  Highs& h;
  // An exception thrown by a step (a cancelled task's HighsTask::Interrupt),
  // rethrown once Rust has returned; no step is made after it
  std::exception_ptr pending;
  // The state of the drivers (drivers.rs) between steps
  std::vector<double> saved_cost;
  HighsHessian saved_hessian;
  std::string saved_presolve;
  bool saved_solve_relaxation = false;
  bool saved_allow_unbounded_or_infeasible = false;
  HighsModel* user_model = nullptr;
  HighsModel read_model;
  HighsBasis read_basis;

  // A call on the engine with the Highs object's data synced in and out
  template <typename F>
  HighsStatus topCall(F f) {
    topIn();
    bool interrupted = false;
    const HighsStatus status = HighsStatus(f(&interrupted));
    topOut();
    if (interrupted) throw HighsTask::Interrupt();
    return status;
  }
  FILE* write_file = nullptr;
  // The presolve data of the LP run on Rust data into presolve_ (its
  // reduced LP's other members, from the model, are kept: the names follow
  // the index maps)
  HighsPresolveStatus presolveExport() {
    HighsLp& lp = h.presolve_.data_.reduced_lp_;
    RsLpVec v = rsLpVec(lp);
    RsPresolveExport e;
    if (!highs_rs_lps_presolve_export(h.ekk_instance_.lps, &v, &e))
      return HighsPresolveStatus::kNotPresolved;
    rsLpVecBack(v, lp);
    if (lp.col_names_.size() > 0) {
      std::vector<std::string> names(e.num_col);
      for (size_t i = 0; i != e.num_col; ++i)
        names[i] = std::move(lp.col_names_[e.orig_col_index[i]]);
      lp.col_names_ = std::move(names);
    }
    if (lp.row_names_.size() > 0) {
      std::vector<std::string> names(e.num_row);
      for (size_t i = 0; i != e.num_row; ++i)
        names[i] = std::move(lp.row_names_[e.orig_row_index[i]]);
      lp.row_names_ = std::move(names);
    }
    if (e.prepared) lp.origin_name_ = "Reduced LP";
    h.presolve_.data_.postSolveStack.rustSet(
        e.data, e.data_len, e.reductions, e.num_reductions, e.orig_col_index,
        e.num_col, e.orig_row_index, e.num_row, e.linearly_transformable,
        e.num_lt, e.orig_num_col, e.orig_num_row);
    if (e.num_log > 0) {
      h.presolve_.presolve_status_ = HighsPresolveStatus(e.status);
      HighsPresolveLog& log = h.presolve_.data_.presolve_log_;
      log.rule.resize(e.num_log);
      for (size_t r = 0; r != e.num_log; ++r) {
        log.rule[r].call = e.log[r][0];
        log.rule[r].col_removed = e.log[r][1];
        log.rule[r].row_removed = e.log[r][2];
      }
      h.presolve_log_ = h.presolve_.getPresolveLog();
    }
    return HighsPresolveStatus(e.status);
  }

  // The Highs object's data for the LP run on Rust data
  RsRunData runData(RsBasisVec& basis) {
    basis = rsBasisVec(h.basis_w());
    RsRunData d;
    d.col_value = rsVec(h.solution_w().col_value);
    d.col_dual = rsVec(h.solution_w().col_dual);
    d.row_value = rsVec(h.solution_w().row_value);
    d.row_dual = rsVec(h.solution_w().row_dual);
    d.value_valid = &h.solution_w().value_valid;
    d.dual_valid = &h.solution_w().dual_valid;
    d.basis = &basis;
    d.origin = {reinterpret_cast<uint8_t*>(
                    const_cast<char*>(h.basis_r().debug_origin_name.data())),
                h.basis_r().debug_origin_name.size()};
    d.origin_ctx = &h.basis_w().debug_origin_name;
    d.set_origin = setOrigin;
    d.info = static_cast<HighsInfoStruct*>(&h.info_w());
    d.model_status = &h.model_status_w();
    return d;
  }

  // ---- The top level on the engine (rust/src/lp_data/top.rs)

  RsTopData topData(RsBasisVec& basis) {
    RsTopData d;
    d.run = runData(basis);
    d.run_data = static_cast<HighsRunDataStruct*>(&h.run_data_);
    d.presolve_status = &h.model_presolve_status_;
    d.called_return = &h.called_return_from_optimize_model;
    d.hessian = rsHessian(h.model_cache_.hessian_);
    d.profiling = h.profiling_;
    return d;
  }

  // The Highs object's options, model, solution, basis, info, model
  // status, run data and Hessian into its engine before a call
  void topIn() {
    highs_rs::LpHandle* e = h.ekk_instance_.p;
    if (h.options_cpp_newer_) {
      rsSyncOptions(e, h.options_);
      h.options_cpp_newer_ = false;
    } else if (kRsCheckSync) {
      HighsLogOptions no_log;
      if (rsHighsOptions(e, 5, no_log, h.options_, "", 0, false, 0, 0,
                         nullptr, nullptr))
        std::abort();
    }
    h.lpToRust();
    // (the views below are made through the writers' accessors)
    const bool run_newer = h.run_cpp_newer_;
    RsBasisVec basis;
    const RsTopData d = topData(basis);
    if (!run_newer && kRsCheckSync && !highs_rs_lph_top_run_matches(e, &d))
      std::abort();
    highs_rs_lph_top_import(e, &d, run_newer);
    h.run_cpp_newer_ = false;
  }

  // The engine's data back into the Highs object's copies after a call
  void topOut() {
    highs_rs::LpHandle* e = h.ekk_instance_.p;
    RsBasisVec basis;
    RsTopData d = topData(basis);
    const uint32_t changed = highs_rs_lph_top_changed(e);
    // The option values the run changed (and restored but for these)
    HighsLogOptions no_log;
    rsHighsOptions(e, 4, no_log, h.options_, "", 0, false, 0, 0, nullptr,
                   nullptr);
    // What the engine cleared, then its data
    if (changed & kXClearModel) {
      h.model_cache_.clear();
      h.multi_linear_objective_.clear();
      h.saved_objective_and_solution_.clear();
    }
    if (changed & kXClearIis) {
      h.clearIis();
      h.invalidateRanging();
    }
    if (changed & kXClearDerived) {
      h.clearPresolve();
      h.clearStandardFormLp();
    }
    if (changed & kXClearPresolve) h.clearPresolve();
    highs_rs_lph_top_export(e, &d);
    rsBasisVecBack(basis, h.basis_w());
    // The mirror is the engine's
    h.run_cpp_newer_ = false;
    if (changed & kXTakeModel) {
      // The passed model's names and the members the engine's model does
      // not hold
      h.model_cache_.lp_ = std::move(user_model->lp_);
      h.model_cache_.lp_.origin_name_ = "Original";
    }
    if (changed & kXModel) {
      h.lpFromRust();
      highs_rs::highs_rs_lph_take_matrix_back(e);
    } else {
      // The scale factors the simplex gave the model (and its matrix)
      rsModelBack(e, h.model_cache_.lp_);
    }
    if (changed & kXHessian) {
      RsHessian v = rsHessian(h.model_cache_.hessian_);
      highs_rs_lph_top_hessian(e, &v);
    }
    if (changed & kXPresolveClear) h.presolve_.clear();
    if (changed & kXPresolve) {
      h.presolve_.init(h.model_r().lp_, h.timer_);
      h.presolve_.options_ = &h.options_;
      const HighsPresolveStatus status = presolveExport();
      if (changed & kXPresolveMip) {
        h.presolve_.presolve_status_ = status;
        h.presolve_log_ = h.presolve_.getPresolveLog();
      }
    }
    if (changed & kXPresolveRecords) {
      const RsPresolveRecords r = highs_rs_lph_top_records(e);
      h.presolve_.postsolve_status_ = HighsPostsolveStatus(r.postsolve_status);
      h.presolve_.info_.n_cols_removed = r.n_cols_removed;
      h.presolve_.info_.n_rows_removed = r.n_rows_removed;
      h.presolve_.info_.n_nnz_removed = r.n_nnz_removed;
      h.presolve_.info_.presolve_time = r.presolve_time;
      h.presolve_.info_.postsolve_time = r.postsolve_time;
    }
    if (changed & kXPresolvedModel) {
      if (highs_rs_lph_top_presolved_which(e) == 0) {
        h.presolved_model_ = h.model_r();
      } else {
        h.presolved_model_.lp_ = h.presolve_.getReducedProblem();
        h.presolved_model_.lp_.setMatrixDimensions();
      }
    }
    if (changed & kXSaved) {
      std::vector<HighsObjectiveSolution>& saved =
          h.saved_objective_and_solution_;
      saved.clear();
      double objective;
      size_t n;
      const size_t num = highs_rs_lph_top_saved(e, SIZE_MAX, &objective, &n);
      for (size_t k = 0; k < num; ++k) {
        HighsObjectiveSolution record;
        const double* v = reinterpret_cast<const double*>(
            highs_rs_lph_top_saved(e, k, &record.objective, &n));
        record.col_value.assign(v, v + n);
        saved.push_back(std::move(record));
      }
    }
  }

  // The top level's steps on C++ objects (top.rs H_*)
  int64_t topOp(int code, int64_t arg, void* p) {
    switch (code) {
      case 1: {
        const bool already = h.profiling_ != nullptr;
        if (!already) h.initializeProfiling(new HighsProfiling);
        *static_cast<void**>(p) = h.profiling_;
        return already;
      }
      case 2:
        if (!arg) {
          HighsProfiling* profiling = h.profiling_;
          h.reportProfiling();
          h.clearProfiling();
          delete profiling;
        }
        return 0;
      case 3:
        if (h.profiling_) h.resetProfiling();
        return 0;
      case 4:
        return int64_t(h.initializeMultiThreading());
      case 5:
        if (h.profiling_) {
          const HighsInt k = (arg >> 1) ? kSubSolverQpAsm : kSubSolverMip;
          if (arg & 1)
            h.profiling_->start(k);
          else
            h.profiling_->stop(k);
        }
        return 0;
      case 6:
        if (arg & 1) {
          if (arg & 4) h.lpFromRust();
          *static_cast<void**>(p) = highsMipHostNew(
              h.callback_, h.options_, h.model_r().lp_, (arg & 2) != 0);
        } else {
          highsMipHostFree(p);
        }
        return 0;
      case 7: {
        const RsMut<double>& v = *static_cast<const RsMut<double>*>(p);
        const std::vector<double> values(v.ptr, v.ptr + v.len);
        analyseVectorValues(&h.options_.log_options, "Small values in matrix",
                            HighsInt(values.size()), values, false, "");
        return 0;
      }
      case 9: {
        const bool already = h.profiling_ != nullptr;
        if (!already) h.initializeSingleThreadedProfiling(new HighsProfiling);
        *static_cast<void**>(p) = h.profiling_;
        return already;
      }
      case 10:
        if (!arg) {
          HighsProfiling* profiling = h.profiling_;
          h.clearProfiling();
          delete profiling;
        }
        return 0;
      case 11:
        h.logHeader();
        return 0;
      case 12:
        if (h.options_.write_matrix_image)
          writeLpMatrixPicToFile(h.options_, "LpMatrix", h.model_r().lp_);
        if (h.options_.write_hessian_image)
          writeHessianPicToFile(h.options_, "Hessian", h.model_cache_.hessian_);
        return 0;
      case 8: {
        static thread_local std::unique_ptr<RsNameList> col, row;
        col.reset(new RsNameList(h.lpCpp().col_names_));
        row.reset(new RsNameList(h.lpCpp().row_names_));
        RsMut<RsName>* out = static_cast<RsMut<RsName>*>(p);
        out[0] = col->view();
        out[1] = row->view();
        return 0;
      }
    }
    assert(false);
    return 0;
  }

  static int64_t op(void* ctx, int which, int64_t arg, void* p,
                    const char* msg, size_t len) {
    HighsRunRust& r = *static_cast<HighsRunRust*>(ctx);
    if (r.pending) return 0;
    try {
      return r.step(RunOp(which), arg, p, msg, len);
    } catch (...) {
      r.pending = std::current_exception();
      return kAbort;
    }
  }

  static double clock(void* ctx, int which, int action) {
    if (static_cast<HighsRunRust*>(ctx)->pending) return 0;
    Highs& h = static_cast<HighsRunRust*>(ctx)->h;
    HighsTimer& t = h.timer_;
    const HighsInt c = which == 0   ? 0
                       : which == 1 ? t.solve_clock
                       : which == 2 ? t.presolve_clock
                                    : t.postsolve_clock;
    switch (action) {
      case 0:
        return t.read(c);
      case 1:
        t.start(c);
        return 0;
      case 2:
        t.stop(c);
        return 0;
      default:
        return t.running(c) ? 1 : 0;
    }
  }

  RsHighs view() {
    HighsOptions& o = h.options_;
    RsHighs v;
    v.log = rsLog(o.log_options);
    v.ctx = this;
    v.op = op;
    v.clock = clock;
    v.model_status = &h.model_status_w();
    v.presolve_status = &h.model_presolve_status_;
    v.info = static_cast<HighsInfoStruct*>(&h.info_w());
    v.run_data = static_cast<HighsRunDataStruct*>(&h.run_data_);
    v.value_valid = &h.solution_w().value_valid;
    v.dual_valid = &h.solution_w().dual_valid;
    v.basis_valid = &h.basis_w().valid;
    v.basis_alien = &h.basis_w().alien;
    v.basis_useful = &h.basis_w().useful;
    v.basis_was_alien = &h.basis_w().was_alien;
    v.called_return = &h.called_return_from_optimize_model;
    v.o.solver = rsRunStr(o.solver);
    v.o.run_crossover = rsRunStr(o.run_crossover);
    v.o.presolve = rsRunStr(o.presolve);
    v.o.use_warm_start = o.use_warm_start;
    v.o.icrash = o.icrash;
    v.o.solve_relaxation = o.solve_relaxation;
    v.o.allow_unbounded_or_infeasible = o.allow_unbounded_or_infeasible;
    v.o.timeless_log = o.timeless_log;
    v.o.large_matrix_value = o.large_matrix_value;
    v.o.time_limit = o.time_limit;
    v.o.primal_feasibility_tolerance = o.primal_feasibility_tolerance;
    v.o.mip_feasibility_tolerance = o.mip_feasibility_tolerance;
    v.o.highs_debug_level = &o.highs_debug_level;
    v.o.objective_bound = &o.objective_bound;
    v.o.lp_presolve_requires_basis_postsolve =
        &o.lp_presolve_requires_basis_postsolve;
    v.o.output_flag = o.log_options.output_flag;
    v.o.log_dev_level = o.log_options.log_dev_level;
    v.o.simplex_strategy = o.simplex_strategy;
    return v;
  }

  static int st(HighsStatus s) { return int(s); }
  // run.rs: ABORT
  static constexpr int64_t kAbort = INT64_MIN;

  void rethrow() {
    if (pending) std::rethrow_exception(pending);
  }

  const HighsLp& lpOf(int64_t which) {
    return which ? h.presolve_.getReducedProblem() : h.model_r().lp_;
  }

  int64_t step(RunOp which, int64_t arg, void* p, const char* m, size_t len) {
    HighsOptions& options = h.options_;
    switch (which) {
      case RunOp::kFacts: {
        RsFacts& f = *static_cast<RsFacts*>(p);
        const HighsLp& lp = lpOf(arg);
        f.num_col = lp.num_col_;
        f.num_row = lp.num_row_;
        f.num_nz = lp.a_matrix_.numNz();
        f.is_mip = arg ? lp.isMip() : h.model_r().isMip();
        f.is_qp = arg ? false : h.model_r().isQp();
        f.is_empty = arg ? lp.num_col_ == 0 && lp.num_row_ == 0
                         : h.model_r().isEmpty();
        f.has_infinite_cost = lp.has_infinite_cost_;
        f.model_name = rsRunStr(lp.model_name_);
        return 0;
      }
      case RunOp::kEkkClear:
        h.ekk_instance_.clear();
        return 0;
      case RunOp::kForceSolutionBasisSize:
        h.forceHighsSolutionBasisSize();
        return 0;
      case RunOp::kBasisConsistent:
        // debugHighsBasisConsistent is not checked in this build
        return 1;
      case RunOp::kRetainedEkkDataOk:
        // debugRetainedDataOk is not checked in this build
        return 1;
      case RunOp::kLpDimensionsOk:
        return lpDimensionsOk("returnFromHighs", h.model_r().lp_,
                              options.log_options);
      case RunOp::kEkkFactorCompatible:
        return highs_rs::highs_rs_lph_factor_row_compatible(
            h.ekk_instance_.p, h.lpNumRow());
      default:
        return driverStep(which, arg, p, m, len);
    }
    assert(false);
    return 0;
  }

  // The steps of drivers.rs
  int64_t driverStep(RunOp which, int64_t arg, void* p, const char* m,
                     size_t len) {
    HighsOptions& options = h.options_;
    // The model's LP, read (lpR) or to change (lpW: the engine's copy
    // takes it at the next lpToRust)
#define lpR (h.model_r().lp_)
#define lpW (h.model_w().lp_)
    switch (which) {
      case RunOp::kRayRecord: {
        const HighsEngine& ekk = h.ekk_instance_;
        RsRayRecord& r = *static_cast<RsRayRecord*>(p);
        r.index = arg ? ekk.sh_.primal_ray_index : ekk.sh_.dual_ray_index;
        r.sign = arg ? ekk.sh_.primal_ray_sign : ekk.sh_.dual_ray_sign;
        r.value_size = ekk.rayValue(arg).size();
        r.has_invert = ekk.status_.has_invert;
        return 0;
      }
      case RunOp::kFeasibilityProblem: {
        const bool is_qp = arg & 1;
        if (arg < 2) {
          saved_cost = lpR.col_cost_;
          if (is_qp) saved_hessian = h.model_cache_.hessian_;
          h.getOptionValue("presolve", saved_presolve);
          h.getOptionValue("solve_relaxation", saved_solve_relaxation);
          std::vector<double> zero_costs;
          zero_costs.assign(lpR.num_col_, 0);
          HighsEngine& ekk = h.ekk_instance_;
          const HighsInt ray_index = ekk.sh_.primal_ray_index;
          const HighsInt ray_sign = ekk.sh_.primal_ray_sign;
          const std::vector<double> ray_value = ekk.rayValue(true);
          HighsStatus status =
              h.changeColsCost(0, lpR.num_col_ - 1, zero_costs.data());
          assert(status == HighsStatus::kOk);
          (void)status;
          ekk.sh_.primal_ray_index = ray_index;
          ekk.sh_.primal_ray_sign = ray_sign;
          ekk.setRayValue(true, ray_value);
          if (is_qp) {
            HighsHessian zero_hessian;
            h.passHessian(zero_hessian);
          }
          h.setOptionValue("presolve", kHighsOffString);
          h.setOptionValue("solve_relaxation", true);
        } else {
          lpW.col_cost_ = saved_cost;
          if (is_qp) h.model_cache_.hessian_ = saved_hessian;
          h.setOptionValue("presolve", saved_presolve);
          h.setOptionValue("solve_relaxation", saved_solve_relaxation);
        }
        return 0;
      }
      case RunOp::kUnboundednessProblem:
        if (arg == 0) {
          h.getOptionValue("presolve", saved_presolve);
          h.getOptionValue("solve_relaxation", saved_solve_relaxation);
          h.getOptionValue("allow_unbounded_or_infeasible",
                           saved_allow_unbounded_or_infeasible);
          h.setOptionValue("presolve", kHighsOffString);
          h.setOptionValue("solve_relaxation", true);
          h.setOptionValue("allow_unbounded_or_infeasible", false);
        } else {
          h.setOptionValue("presolve", saved_presolve);
          h.setOptionValue("solve_relaxation", saved_solve_relaxation);
          h.setOptionValue("allow_unbounded_or_infeasible",
                           saved_allow_unbounded_or_infeasible);
        }
        return 0;
      case RunOp::kHighsRun:
        return st(h.run());
      case RunOp::kCopyRay: {
        double* value = static_cast<double*>(p);
        const std::vector<double> ray = h.ekk_instance_.rayValue(arg);
        const HighsInt n = arg ? lpR.num_col_ : lpR.num_row_;
        for (HighsInt i = 0; i < n; i++) value[i] = ray[i];
        return 0;
      }
      case RunOp::kComputeDualRay: {
        double* dual_ray_value = static_cast<double*>(p);
        const HighsInt num_row = lpR.num_row_;
        std::vector<double> rhs;
        HighsInt iRow = h.ekk_instance_.sh_.dual_ray_index;
        rhs.assign(num_row, 0);
        rhs[iRow] = h.ekk_instance_.sh_.dual_ray_sign;
        HighsInt* dual_ray_num_nz = 0;
        h.basisSolveInterface(rhs, dual_ray_value, dual_ray_num_nz, NULL, true);
        h.ekk_instance_.setRayValue(
            false,
            std::vector<double>(dual_ray_value, dual_ray_value + num_row));
        return 0;
      }
      case RunOp::kComputePrimalRay: {
        double* primal_ray_value = static_cast<double*>(p);
        const HighsInt num_row = lpR.num_row_;
        const HighsInt num_col = lpR.num_col_;
        HighsInt col = h.ekk_instance_.sh_.primal_ray_index;
        HighsInt num_tot;
        assert(highs_rs::highs_rs_lps_nonbasic(h.ekk_instance_.lps, 0,
                                               &num_tot)[col] ==
               kNonbasicFlagTrue);
        (void)num_tot;
        std::vector<double> rhs;
        std::vector<double> column;
        column.assign(num_row, 0);
        rhs.assign(num_row, 0);
        if (!lpR.a_matrix_.isColwise()) lpW.ensureColwise();
        HighsInt primal_ray_sign = h.ekk_instance_.sh_.primal_ray_sign;
        if (col < num_col) {
          for (HighsInt iEl = lpR.a_matrix_.start_[col];
               iEl < lpR.a_matrix_.start_[col + 1]; iEl++)
            rhs[lpR.a_matrix_.index_[iEl]] =
                primal_ray_sign * lpR.a_matrix_.value_[iEl];
        } else {
          rhs[col - num_col] = primal_ray_sign;
        }
        HighsInt* column_num_nz = 0;
        h.basisSolveInterface(rhs, column.data(), column_num_nz, NULL, false);
        for (HighsInt iCol = 0; iCol < num_col; iCol++)
          primal_ray_value[iCol] = 0;
        for (HighsInt iRow = 0; iRow < num_row; iRow++) {
          HighsInt iCol = h.ekk_instance_.basicIndex()[iRow];
          if (iCol < num_col) primal_ray_value[iCol] = column[iRow];
        }
        if (col < num_col) primal_ray_value[col] = -primal_ray_sign;
        h.ekk_instance_.setRayValue(
            true,
            std::vector<double>(primal_ray_value, primal_ray_value + num_col));
        return 0;
      }
      case RunOp::kLogHeader:
        h.logHeader();
        return 0;
      case RunOp::kReadModelFile: {
        const std::string filename(m, len);
        Filereader* reader =
            Filereader::getFilereader(options.log_options, filename);
        if (reader == NULL) return -1;
        FilereaderRetcode call_code =
            reader->readModelFromFile(options, filename, read_model);
        delete reader;
        return int64_t(call_code);
      }
      case RunOp::kReadModelPass:
        if (arg == 0) {
          read_model.lp_.model_name_.assign(m, len);
          return 0;
        }
        return st(h.passModel(std::move(read_model)));
      case RunOp::kReadBasis:
        if (arg == 0) {
          read_basis = h.basis_r();
          return st(readBasisFile(options.log_options, lpW, read_basis,
                                  std::string(m, len)));
        } else if (arg == 1) {
          return isBasisConsistent(lpR, read_basis);
        }
        h.basis_w() = read_basis;
        h.basis_w().valid = true;
        h.basis_w().useful = true;
        h.newHighsBasis();
        return 0;
      case RunOp::kWriteModelPrepare: {
        HighsLp& model_lp = user_model->lp_;
        model_lp.setMatrixDimensions();
        const HighsStatus call_status = normaliseNames(
            options.log_options, model_lp, HighsFileType(arg));
        assert(call_status != HighsStatus::kError);
        model_lp.ensureColwise();
        return st(call_status);
      }
      case RunOp::kWriteModelLpView:
        *static_cast<RsLp*>(p) = rsLp(user_model->lp_);
        return 0;
      case RunOp::kWriteModelCheck: {
        HighsModel& model = *user_model;
        switch (arg) {
          case 0:
            return model.hessian_.dim_ > 0
                       ? st(assessHessianDimensions(options, model.hessian_))
                       : 0;
          case 1:
            return st(model.lp_.a_matrix_.assessStart(options.log_options));
          case 2:
            return st(
                model.lp_.a_matrix_.assessIndexBounds(options.log_options));
          case 3:
            return model.lp_.col_hash_.hasDuplicate(model.lp_.col_names_);
          default:
            return model.lp_.row_hash_.hasDuplicate(model.lp_.row_names_);
        }
      }
      case RunOp::kReportWrittenModel:
        h.reportModel(*user_model);
        return 0;
      case RunOp::kWriteModelFile: {
        const std::string filename(m, len);
        Filereader* writer =
            Filereader::getFilereader(options.log_options, filename);
        if (arg == 0) {
          delete writer;
          return writer != NULL;
        }
        const HighsStatus status =
            writer->writeModelToFile(options, filename, *user_model);
        delete writer;
        return st(status);
      }
      case RunOp::kWriteBasis:
        if (arg == 0) {
          HighsFileType file_type;
          return st(h.openWriteFile(std::string(m, len), "writeBasis",
                                    write_file, file_type));
        } else if (arg == 1) {
          const HighsStatus call_status =
              normaliseNames(options.log_options, lpW);
          assert(call_status != HighsStatus::kError);
          return st(call_status);
        }
        writeBasisFile(write_file, options, lpR, h.basis_r());
        if (write_file != stdout) fclose(write_file);
        return 0;
      case RunOp::kSolutionBasisSizes:
        if (arg == 0) {
          int64_t* sizes = static_cast<int64_t*>(p);
          sizes[0] = h.solution_r().col_value.size();
          sizes[1] = h.solution_r().row_value.size();
          sizes[2] = h.solution_r().col_dual.size();
          sizes[3] = h.solution_r().row_dual.size();
          sizes[4] = h.basis_r().col_status.size();
          sizes[5] = h.basis_r().row_status.size();
        } else {
          const HighsInt num_col = h.lpNumCol();
          const HighsInt num_row = h.lpNumRow();
          h.solution_w().col_value.resize(num_col, 0);
          h.solution_w().row_value.resize(num_row, 0);
          h.solution_w().col_dual.resize(num_col, 0);
          h.solution_w().row_dual.resize(num_row, 0);
          h.basis_w().col_status.resize(num_col, HighsBasisStatus::kNonbasic);
          h.basis_w().row_status.resize(num_row, HighsBasisStatus::kBasic);
        }
        return 0;
      default:
        break;
    }
    (void)m;
    (void)len;
    assert(false);
    return 0;
  }

#undef lpR
#undef lpW

  // The debug checks of returnFromOptimizeModel: 1 for a logical error
};

static int64_t rsTopOp(void* ctx, int code, int64_t arg, void* p) {
  HighsRunRust r(*static_cast<Highs*>(ctx));
  return r.topOp(code, arg, p);
}

static double rsTopClock(void* ctx, int which, int action) {
  HighsRunRust r(*static_cast<Highs*>(ctx));
  return HighsRunRust::clock(&r, which, action);
}

extern "C" int highs_rs_get_ray(const RsHighs* h, bool primal, bool* has_ray,
                                double* value, size_t len);

HighsStatus Highs::getDualRayInterface(bool& has_dual_ray,
                                       double* dual_ray_value) {
  assert(!model_r().lp_.is_moved_);
  HighsRunRust r(*this);
  const RsHighs v = r.view();
  const HighsStatus status = HighsStatus(highs_rs_get_ray(
      &v, false, &has_dual_ray, dual_ray_value, lpNumRow()));
  r.rethrow();
  return status;
}

HighsStatus Highs::getPrimalRayInterface(bool& has_primal_ray,
                                         double* primal_ray_value) {
  assert(!model_r().lp_.is_moved_);
  HighsRunRust r(*this);
  const RsHighs v = r.view();
  const HighsStatus status = HighsStatus(highs_rs_get_ray(
      &v, true, &has_primal_ray, primal_ray_value, lpNumCol()));
  r.rethrow();
  return status;
}

HighsStatus Highs::presolve() {
  HighsRunRust r(*this);
  return r.topCall([&](bool* interrupted) {
    return highs_rs_lph_top_presolve(ekk_instance_.p, interrupted);
  });
}

HighsStatus Highs::crossover(const HighsSolution& user_solution) {
  HighsRunRust r(*this);
  const RsSolution s = rsSolution(user_solution);
  return r.topCall([&](bool* interrupted) {
    return highs_rs_lph_top_crossover(ekk_instance_.p, &s, interrupted);
  });
}

extern "C" bool highs_rs_format_ok(const RsLog* log, bool hessian,
                                   HighsInt num_nz, HighsInt format);

extern "C" int highs_rs_highs_file(const RsHighs* h, int which,
                                   const char* filename, size_t len);
extern "C" void highs_rs_force_solution_basis_size(const RsHighs* h);

HighsStatus Highs::readModel(const std::string& filename) {
  HighsRunRust r(*this);
  const RsHighs v = r.view();
  const HighsStatus status = HighsStatus(
      highs_rs_highs_file(&v, 0, filename.data(), filename.size()));
  r.rethrow();
  return status;
}

HighsStatus Highs::readBasis(const std::string& filename) {
  HighsRunRust r(*this);
  const RsHighs v = r.view();
  const HighsStatus status = HighsStatus(
      highs_rs_highs_file(&v, 1, filename.data(), filename.size()));
  r.rethrow();
  return status;
}

HighsStatus Highs::writeLocalModel(HighsModel& model,
                                   const std::string& filename) {
  HighsRunRust r(*this);
  r.user_model = &model;
  const RsHighs v = r.view();
  const HighsStatus status = HighsStatus(
      highs_rs_highs_file(&v, 2, filename.data(), filename.size()));
  r.rethrow();
  return status;
}

HighsStatus Highs::writeBasis(const std::string& filename) {
  HighsRunRust r(*this);
  const RsHighs v = r.view();
  const HighsStatus status = HighsStatus(
      highs_rs_highs_file(&v, 3, filename.data(), filename.size()));
  r.rethrow();
  return status;
}

void Highs::forceHighsSolutionBasisSize() {
  HighsRunRust r(*this);
  const RsHighs v = r.view();
  highs_rs_force_solution_basis_size(&v);
}

HighsStatus Highs::passModel(HighsModel model) {
  HighsRunRust r(*this);
  r.user_model = &model;
  const RsLp lp = rsLp(model.lp_);
  const RsHessian hessian = rsHessian(model.hessian_);
  return r.topCall([&](bool*) {
    return highs_rs_lph_top_pass_model(ekk_instance_.p, 0, &lp,
                                       model.lp_.model_name_.data(),
                                       model.lp_.model_name_.size(), &hessian);
  });
}

HighsStatus Highs::passHessian(HighsHessian hessian_) {
  HighsRunRust r(*this);
  const RsHessian hessian = rsHessian(hessian_);
  return r.topCall([&](bool*) {
    return highs_rs_lph_top_pass_model(ekk_instance_.p, 1, nullptr, nullptr, 0,
                                       &hessian);
  });
}

bool Highs::aFormatOk(const HighsInt num_nz, const HighsInt format) {
  const RsLog log = rsLog(options_.log_options);
  return highs_rs_format_ok(&log, false, num_nz, format);
}

bool Highs::qFormatOk(const HighsInt num_nz, const HighsInt format) {
  const RsLog log = rsLog(options_.log_options);
  return highs_rs_format_ok(&log, true, num_nz, format);
}

HighsStatus Highs::setBasis(const HighsBasis& basis,
                            const std::string& origin) {
  HighsRunRust r(*this);
  const RsBasisVec b = rsBasisVec(const_cast<HighsBasis&>(basis));
  const HighsStatus status = r.topCall([&](bool*) {
    return highs_rs_lph_top_set_basis(
        ekk_instance_.p, &b, basis.debug_origin_name.data(),
        basis.debug_origin_name.size(), origin.data(), origin.size());
  });
  assert(basis_r().debug_origin_name != "");
  assert(!basis_r().alien || status != HighsStatus::kOk);
  return status;
}

HighsStatus Highs::callRunPostsolve(const HighsSolution& solution,
                                    const HighsBasis& basis) {
  HighsRunRust r(*this);
  const RsSolution s = rsSolution(solution);
  const RsBasisVec b = rsBasisVec(const_cast<HighsBasis&>(basis));
  return r.topCall([&](bool* interrupted) {
    return highs_rs_lph_top_postsolve(
        ekk_instance_.p, &s, &b, basis.debug_origin_name.data(),
        basis.debug_origin_name.size(), interrupted);
  });
}

HighsStatus Highs::calledOptimizeModel() {
  // The run is the engine's (rust/src/lp_data/top.rs)
  HighsRunRust r(*this);
  return r.topCall([&](bool* interrupted) {
    return highs_rs_lph_called_optimize_model(ekk_instance_.p, interrupted);
  });
}

HighsStatus Highs::returnFromHighs(HighsStatus highs_return_status) {
  HighsRunRust r(*this);
  const RsHighs v = r.view();
  const HighsStatus status =
      HighsStatus(highs_rs_return_from_highs(&v, int(highs_return_status)));
  r.rethrow();
  return status;
}

// The basis and tableau queries and setSolution (rust/src/lp_data/query.rs)
extern "C" {
int highs_rs_check_query(const RsLog* log, const char* method,
                         size_t method_len, const char* null_arg,
                         size_t null_len, const char* index_kind,
                         size_t index_kind_len, HighsInt index, HighsInt dim,
                         bool has_invert);
HighsInt highs_rs_reduced_row(RsMut<HighsInt> start, RsMut<HighsInt> index,
                              RsMut<double> value, RsMut<double> binv_row,
                              RsMut<double> row, RsMut<HighsInt> indices);
HighsInt highs_rs_extract_solve(HighsInt count, RsMut<HighsInt> index,
                                RsMut<double> array, RsMut<double> solution,
                                RsMut<HighsInt> indices);
int highs_rs_check_sparse_solution(const RsLog* log, RsMut<HighsInt> index,
                                   RsMut<double> value, RsMut<double> lower,
                                   RsMut<double> upper, double pft);
int highs_rs_new_solution_parts(const RsLog* log, HighsInt num_col,
                                HighsInt num_row, size_t col_value_size,
                                size_t row_dual_size);
}

namespace {
// The checks of a query: a NULL argument, an index out of range, no INVERT
HighsStatus checkQuery(const HighsOptions& options, const char* method,
                       const char* null_arg, const char* index_kind,
                       const HighsInt index, const HighsInt dim,
                       const bool has_invert) {
  const RsLog log = rsLog(options.log_options);
  auto n = [](const char* s) { return s ? strlen(s) : 0; };
  return HighsStatus(highs_rs_check_query(
      &log, method, n(method), null_arg, n(null_arg), index_kind,
      n(index_kind), index, dim, has_invert));
}

template <typename T>
RsMut<T> rsOut(T* p, const size_t n) {
  return {p, p ? n : 0};
}
}  // namespace

HighsStatus Highs::getBasisInverseRow(const HighsInt row, double* row_vector,
                                      HighsInt* row_num_nz,
                                      HighsInt* row_indices) {
  const HighsInt num_row = lpNumRow();
  if (checkQuery(options_, "getBasisInverseRow",
                 row_vector ? nullptr : "row_vector", "Row", row, num_row,
                 ekk_instance_.status_.has_invert) != HighsStatus::kOk)
    return HighsStatus::kError;
  vector<double> rhs(num_row, 0);
  rhs[row] = 1;
  basisSolveInterface(rhs, row_vector, row_num_nz, row_indices, true);
  return HighsStatus::kOk;
}

HighsStatus Highs::getBasisInverseCol(const HighsInt col, double* col_vector,
                                      HighsInt* col_num_nz,
                                      HighsInt* col_indices) {
  const HighsInt num_row = lpNumRow();
  if (checkQuery(options_, "getBasisInverseCol",
                 col_vector ? nullptr : "col_vector", "Column", col, num_row,
                 ekk_instance_.status_.has_invert) != HighsStatus::kOk)
    return HighsStatus::kError;
  vector<double> rhs(num_row, 0);
  rhs[col] = 1;
  basisSolveInterface(rhs, col_vector, col_num_nz, col_indices, false);
  return HighsStatus::kOk;
}

HighsStatus Highs::getBasisSolve(const double* Xrhs, double* solution_vector,
                                 HighsInt* solution_num_nz,
                                 HighsInt* solution_indices) {
  const char* null_arg = !Xrhs              ? "Xrhs"
                         : !solution_vector ? "solution_vector"
                                            : nullptr;
  if (checkQuery(options_, "getBasisSolve", null_arg, nullptr, 0, 0,
                 ekk_instance_.status_.has_invert) != HighsStatus::kOk)
    return HighsStatus::kError;
  const HighsInt num_row = lpNumRow();
  vector<double> rhs(Xrhs, Xrhs + num_row);
  basisSolveInterface(rhs, solution_vector, solution_num_nz, solution_indices,
                      false);
  return HighsStatus::kOk;
}

HighsStatus Highs::getBasisTransposeSolve(const double* Xrhs,
                                          double* solution_vector,
                                          HighsInt* solution_num_nz,
                                          HighsInt* solution_indices) {
  const char* null_arg = !Xrhs              ? "Xrhs"
                         : !solution_vector ? "solution_vector"
                                            : nullptr;
  if (checkQuery(options_, "getBasisTransposeSolve", null_arg, nullptr, 0, 0,
                 ekk_instance_.status_.has_invert) != HighsStatus::kOk)
    return HighsStatus::kError;
  const HighsInt num_row = lpNumRow();
  vector<double> rhs(Xrhs, Xrhs + num_row);
  basisSolveInterface(rhs, solution_vector, solution_num_nz, solution_indices,
                      true);
  return HighsStatus::kOk;
}

HighsStatus Highs::getReducedRow(const HighsInt row, double* row_vector,
                                 HighsInt* row_num_nz, HighsInt* row_indices,
                                 const double* pass_basis_inverse_row_vector) {
  if (!model_r().lp_.a_matrix_.isColwise()) model_w().lp_.ensureColwise();
  const HighsLp& lp = model_r().lp_;
  if (checkQuery(options_, "getReducedRow",
                 row_vector ? nullptr : "row_vector", "Row", row, lp.num_row_,
                 ekk_instance_.status_.has_invert) != HighsStatus::kOk)
    return HighsStatus::kError;
  const HighsInt num_row = lp.num_row_;
  vector<double> basis_inverse_row;
  const double* basis_inverse_row_vector = pass_basis_inverse_row_vector;
  if (basis_inverse_row_vector == NULL) {
    vector<double> rhs(num_row, 0);
    rhs[row] = 1;
    basis_inverse_row.resize(num_row, 0);
    basisSolveInterface(rhs, basis_inverse_row.data(), NULL, NULL, true);
    basis_inverse_row_vector = basis_inverse_row.data();
  }
  const HighsInt num_nz = highs_rs_reduced_row(
      rsMut(lp.a_matrix_.start_), rsMut(lp.a_matrix_.index_),
      rsMut(lp.a_matrix_.value_),
      {const_cast<double*>(basis_inverse_row_vector), size_t(num_row)},
      {row_vector, size_t(lp.num_col_)},
      rsOut(row_num_nz ? row_indices : nullptr, size_t(lp.num_col_)));
  if (row_num_nz) *row_num_nz = num_nz;
  return HighsStatus::kOk;
}

HighsStatus Highs::getReducedColumn(const HighsInt col, double* col_vector,
                                    HighsInt* col_num_nz,
                                    HighsInt* col_indices) {
  if (!model_r().lp_.a_matrix_.isColwise()) model_w().lp_.ensureColwise();
  const HighsLp& lp = model_r().lp_;
  if (checkQuery(options_, "getReducedColumn",
                 col_vector ? nullptr : "col_vector", "Column", col,
                 lp.num_col_,
                 ekk_instance_.status_.has_invert) != HighsStatus::kOk)
    return HighsStatus::kError;
  vector<double> rhs(lp.num_row_, 0);
  for (HighsInt el = lp.a_matrix_.start_[col];
       el < lp.a_matrix_.start_[col + 1]; el++)
    rhs[lp.a_matrix_.index_[el]] = lp.a_matrix_.value_[el];
  basisSolveInterface(rhs, col_vector, col_num_nz, col_indices, false);
  return HighsStatus::kOk;
}

HighsStatus Highs::basisSolveInterface(const vector<double>& rhs,
                                       double* solution_vector,
                                       HighsInt* solution_num_nz,
                                       HighsInt* solution_indices,
                                       bool transpose) {
  const HighsLp& lp = model_r().lp_;
  const HighsInt num_row = lp.num_row_;
  if (num_row == 0) return HighsStatus::kOk;
  assert(ekk_instance_.status_.has_invert);
  ekk_instance_.setNlaPointersForLpAndScale(lp);
  assert(!lp.is_moved_);
  HVector solve_vector;
  solve_vector.setup(num_row);
  solve_vector.clear();
  HighsInt rhs_num_nz = 0;
  for (HighsInt iRow = 0; iRow < num_row; iRow++) {
    if (rhs[iRow]) {
      solve_vector.index[rhs_num_nz++] = iRow;
      solve_vector.array[iRow] = rhs[iRow];
    }
  }
  solve_vector.count = rhs_num_nz;
  const double expected_density = 1;
  if (transpose) {
    ekk_instance_.btran(solve_vector, expected_density);
  } else {
    ekk_instance_.ftran(solve_vector, expected_density);
  }
  const HighsInt num_nz = highs_rs_extract_solve(
      solve_vector.count, rsMut(solve_vector.index), rsMut(solve_vector.array),
      {solution_vector, size_t(num_row)},
      rsOut(solution_indices, size_t(num_row)));
  if (num_nz >= 0) *solution_num_nz = num_nz;
  return HighsStatus::kOk;
}

HighsStatus Highs::setSolution(const HighsSolution& solution) {
  HighsStatus return_status = HighsStatus::kOk;
  const RsLog log = rsLog(options_.log_options);
  const int parts = highs_rs_new_solution_parts(
      &log, lpNumCol(), lpNumRow(),
      solution.col_value.size(), solution.row_dual.size());
  const bool new_primal_solution = parts & 1;
  const bool new_dual_solution = parts & 2;
  if (parts) {
    invalidateSolverData();
  } else {
    return_status = HighsStatus::kError;
  }
  if (new_primal_solution) {
    solution_w().col_value = solution.col_value;
    if (lpNumRow() > 0) {
      solution_w().row_value.resize(lpNumRow());
      if (!model_r().lp_.a_matrix_.isColwise())
        model_w().lp_.a_matrix_.ensureColwise();
      return_status = interpretCallStatus(
          options_.log_options, calculateRowValuesQuad(model_r().lp_, solution_w()),
          return_status, "calculateRowValuesQuad");
      if (return_status == HighsStatus::kError) return return_status;
    }
    solution_w().value_valid = true;
  }
  if (new_dual_solution) {
    solution_w().row_dual = solution.row_dual;
    if (lpNumCol() > 0) {
      solution_w().col_dual.resize(lpNumCol());
      if (!model_r().lp_.a_matrix_.isColwise())
        model_w().lp_.a_matrix_.ensureColwise();
      return_status = interpretCallStatus(
          options_.log_options, calculateColDualsQuad(model_r().lp_, solution_w()),
          return_status, "calculateColDuals");
      if (return_status == HighsStatus::kError) return return_status;
    }
    solution_w().dual_valid = true;
  }
  return returnFromHighs(return_status);
}

HighsStatus Highs::setSolution(const HighsInt num_entries,
                               const HighsInt* index, const double* value) {
  if (lpNumCol() == 0) return HighsStatus::kOk;
  const RsLog log = rsLog(options_.log_options);
  const HighsStatus return_status =
      HighsStatus(highs_rs_check_sparse_solution(
          &log, {const_cast<HighsInt*>(index), size_t(num_entries)},
          {const_cast<double*>(value), size_t(num_entries)},
          rsMut(model_r().lp_.col_lower_), rsMut(model_r().lp_.col_upper_),
          options_.primal_feasibility_tolerance));
  if (return_status == HighsStatus::kError) return return_status;
  HighsSolution new_solution;
  new_solution.col_value.assign(lpNumCol(), kHighsUndefined);
  for (HighsInt iX = 0; iX < num_entries; iX++)
    new_solution.col_value[index[iX]] = value[iX];
  return interpretCallStatus(options_.log_options, setSolution(new_solution),
                             return_status, "setSolution");
}

extern "C" double highs_rs_user_scale_solution(
    const HighsUserScaleData* d, RsMut<uint8_t> integrality, bool primal,
    bool dual, RsMut<double> col_value, RsMut<double> row_value,
    RsMut<double> col_dual, RsMut<double> row_dual, double objective,
    double offset);

HighsStatus Highs::userScaleSolution(HighsUserScaleData& data,
                                     bool update_kkt) {
  HighsStatus return_status = HighsStatus::kOk;
  if (!data.user_objective_scale && !data.user_bound_scale)
    return HighsStatus::kOk;
  const HighsLp& lp = this->model_r().lp_;
  const bool primal = info_r().primal_solution_status != kSolutionStatusNone;
  const bool dual = info_r().dual_solution_status != kSolutionStatusNone;
  auto part = [](std::vector<double>& v, const bool use, const HighsInt n) {
    return use ? RsMut<double>{v.data(), size_t(n)} : RsMut<double>{nullptr, 0};
  };
  const double objective_function_value = highs_rs_user_scale_solution(
      &data, rsMut(lp.integrality_), primal, dual,
      part(solution_w().col_value, primal, lp.num_col_),
      part(solution_w().row_value, primal, lp.num_row_),
      part(solution_w().col_dual, dual, lp.num_col_),
      part(solution_w().row_dual, dual, lp.num_row_),
      info_r().objective_function_value, lp.offset_);
  if (!update_kkt) return return_status;
  info_w().objective_function_value = objective_function_value;
  getKktFailures(options_, model_r(), solution_w(), basis_w(), info_w());
  return reportKktFailures(model_r().lp_, options_, info_r(),
                           "After removing user scaling")
             ? HighsStatus::kWarning
             : return_status;
}

// Infinite costs, basisForSolution and reportModelStats
// (rust/src/lp_data/model.rs)
struct RsInfCostMods {
  RsMut<HighsInt> index;
  RsMut<double> cost, lower, upper;
  HighsInt num;
};

extern "C" {
int highs_rs_handle_inf_cost(const RsLog* log, double inf_cost, bool minimize,
                             bool is_mip, RsMut<uint8_t> integrality,
                             RsMut<double> cost, RsMut<double> col_lower,
                             RsMut<double> col_upper, RsInfCostMods* m);
void highs_rs_restore_inf_cost(RsMut<HighsInt> index, RsMut<double> saved_cost,
                               RsMut<double> saved_lower,
                               RsMut<double> saved_upper,
                               RsMut<double> col_value,
                               RsMut<uint8_t> col_status, RsMut<double> cost,
                               RsMut<double> col_lower,
                               RsMut<double> col_upper, double* objective);
HighsInt highs_rs_basis_for_solution(
    const RsLog* log, double tol, RsMut<double> col_lower,
    RsMut<double> col_upper, RsMut<double> col_value, RsMut<double> row_lower,
    RsMut<double> row_upper, RsMut<double> row_value,
    RsMut<uint8_t> col_status, RsMut<uint8_t> row_status);
void highs_rs_report_model_stats(const RsLog* log, bool dev, const char* name,
                                 size_t name_len, HighsInt num_col,
                                 HighsInt num_row, HighsInt a_num_nz,
                                 HighsInt hessian_dim, HighsInt q_num_nz,
                                 RsMut<uint8_t> integrality,
                                 RsMut<double> col_lower,
                                 RsMut<double> col_upper);
}

static RsMut<uint8_t> rsBasisStatusOf(std::vector<HighsBasisStatus>& s) {
  return {reinterpret_cast<uint8_t*>(s.data()), s.size()};
}

void Highs::reportModelStats() const {
  const HighsLp& lp = this->model_r().lp_;
  const HighsHessian& hessian = this->model_r().hessian_;
  const HighsLogOptions& log_options = this->options_.log_options;
  if (!*log_options.output_flag) return;
  const RsLog log = rsLog(log_options);
  highs_rs_report_model_stats(
      &log, *log_options.log_dev_level != 0, lp.model_name_.data(),
      lp.model_name_.size(), lp.num_col_, lp.num_row_, lp.a_matrix_.numNz(),
      hessian.dim_, hessian.dim_ > 0 ? hessian.numNz() : 0,
      rsMut(lp.integrality_), rsMut(lp.col_lower_), rsMut(lp.col_upper_));
}
#endif
