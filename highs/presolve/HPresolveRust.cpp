/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file presolve/HPresolveRust.cpp
 * @brief HPresolve of the Rust port (rust/src/presolve/hpresolve): okSetInput
 * prepares the model as the C++ does, run hands the model and the postsolve
 * stack's index maps to Rust, which presolves and writes the model back.
 * The callbacks below are what the Rust calls in C++: logging, the timer,
 * the HighsLp and postsolve stack
 * updates, the dependent equations' HFactor, and the parts of the MIP solver
 * that are C++: the setup of the domain and clique table for probing,
 * HighsImplications::runProbing's C++ glue and its lifting opportunities,
 * the cut pool and the views of the domain and clique table the Rust
 * probing and enumeration loops work on.
 */
#include "presolve/HPresolve.h"

#ifdef HIGHS_RUST

#include <numeric>

#include "io/HighsIO.h"
#include "lp_data/HighsLpUtils.h"
#include "presolve/HighsPostsolveStack.h"
#include "util/HFactor.h"

static_assert(sizeof(HighsInt) == 4, "the Rust port uses 32-bit HighsInt");
static_assert(sizeof(HighsVarType) == 1, "integrality is read as bytes");

namespace presolve {

// The factorization of the dependent equations (HPresolve's
// removeDependentEquations): the build return, its time, the variables
// with no pivot in `ints`
HighsInt rsDependentEquations(HighsTimer& timer, std::vector<HighsInt>& ints,
                              size_t num_col, HighsInt num_row,
                              const HighsInt* start, const HighsInt* index,
                              const double* value, size_t nnz,
                              double time_limit, double* time_taken) {
  HighsSparseMatrix matrix;
  matrix.num_col_ = num_col;
  matrix.num_row_ = num_row;
  matrix.start_.assign(start, start + num_col + 1);
  matrix.index_.assign(index, index + nnz);
  matrix.value_.assign(value, value + nnz);
  std::vector<HighsInt> colSet(matrix.num_col_);
  std::iota(colSet.begin(), colSet.end(), 0);
  HFactor factor;
  factor.setup(matrix, colSet);
  factor.setTimeLimit(time_limit);
  *time_taken = -timer.read();
  HighsInt build_return = factor.build();
  *time_taken += timer.read();
  ints = factor.var_with_no_pivot;
  return build_return;
}

namespace {

struct RsOptions {
  double primal_feasibility_tolerance;
  double dual_feasibility_tolerance;
  double mip_feasibility_tolerance;
  double small_matrix_value;
  double time_limit;
  double presolve_pivot_threshold;
  HighsInt presolve_substitution_maxfillin;
  HighsInt presolve_rule_test;
  HighsInt presolve_rule_off;
  HighsInt log_dev_level;
  HighsInt random_seed;
  HighsInt mip_lifting_for_probing;
  bool presolve_off;
  bool lp_presolve_requires_basis_postsolve;
  bool presolve_remove_slacks;
  bool output_flag;
  bool timeless_log;
  bool use_implied_bounds_from_presolve;
  bool presolve_rule_logging;
};


struct RsModel {
  HighsInt num_col;
  HighsInt num_row;
  const double* col_cost;
  const double* col_lower;
  const double* col_upper;
  const double* row_lower;
  const double* row_upper;
  const uint8_t* integrality;
  double offset;
  bool maximize;
};

struct RsLiftOpp {
  HighsInt row;
  HighsInt col;
  HighsInt val;
  double coef;
};

template <typename T>
struct RsSlice {
  const T* ptr;
  size_t len;
};

struct RsInput {
  HighsInt num_col;
  HighsInt num_row;
  const double* col_cost;
  const double* col_lower;
  const double* col_upper;
  const double* row_lower;
  const double* row_upper;
  const uint8_t* integrality;
  double offset;
  bool maximize;
  const HighsInt* a_start;
  const HighsInt* a_index;
  const double* a_value;
  size_t num_nz;
  const HighsInt* orig_col_index;
  size_t num_orig_col_index;
  const HighsInt* orig_row_index;
  size_t num_orig_row_index;
  size_t stack_data_size;
  size_t num_reductions;
  HighsInt presolve_reduction_limit;
  const char* model_name;
};

struct RsOut {
  HighsInt presolve_status;
  HighsInt log[kPresolveRuleCount][3];
};

/// The C++ state the callbacks work on
struct Ctx {
  HPresolve* presolve;
  HighsLp* model;
  const HighsOptions* options;
  HighsTimer* timer;
  HighsPostsolveStack* stack;
  // scratch arrays handed to Rust
  std::vector<HighsInt> ints;
  std::vector<HighsInt> ints2;
  std::vector<RsLiftOpp> lifting;
};

using Cb = void*;


struct RsHost {
  Cb ctx;
  void (*log)(Cb, HighsInt, HighsInt, const char*);
  double (*timer_read)(Cb);
  void (*time_string)(Cb, double, char*, size_t);
  void (*sync_model)(Cb, const RsModel*, bool);
  void (*set_matrix)(Cb, const HighsInt*, size_t, const HighsInt*,
                     const double*, size_t);
  void (*flush)(Cb, const char*, size_t, const uint8_t*, const size_t*, size_t,
                const HighsInt*, size_t);
  void (*shrink)(Cb, const HighsInt*, size_t, const HighsInt*, size_t);
  HighsInt (*dependent_equations)(Cb, size_t, HighsInt, const HighsInt*,
                                  const HighsInt*, const double*, size_t,
                                  double, double*, RsSlice<HighsInt>*);
  void (*profiling)(Cb, bool, HighsInt);
  bool (*probing_prepare)(Cb, HighsInt, bool*);
  void (*mip_env)(Cb, void*);
  bool (*probe)(Cb, HighsInt, HighsInt*);
  void (*set_lifting)(Cb, bool);
  void (*lifting_opps)(Cb, RsSlice<RsLiftOpp>*);
  void (*finalise_begin)(Cb, bool, RsSlice<HighsInt>*, RsSlice<HighsInt>*);
  void (*domain_bounds)(Cb, RsSlice<double>*, RsSlice<double>*);
  void (*mip_finish_presolve)(Cb, HighsInt);
  void (*add_cut)(Cb, const HighsInt*, const double*, size_t, double, bool);
  double (*upper_limit)(Cb);
  void (*set_lower_bound_zero)(Cb);
};

Ctx& C(Cb c) { return *static_cast<Ctx*>(c); }

void cbLog(Cb c, HighsInt channel, HighsInt type, const char* msg) {
  const HighsLogOptions& lo = C(c).options->log_options;
  switch (channel) {
    case 0:
      highsLogUser(lo, HighsLogType(type), "%s", msg);
      break;
    case 1:
      highsLogDev(lo, HighsLogType(type), "%s", msg);
      break;
    default:
      printf("%s", msg);
      fflush(stdout);
  }
}

double cbTimerRead(Cb c) { return C(c).timer->read(); }

void cbTimeString(Cb, double t, char* buf, size_t len) {
  std::string s = highsTimeSecondToString(t);
  snprintf(buf, len, "%s", s.c_str());
}

void cbSyncModel(Cb c, const RsModel* m, bool resize_row_names) {
  HighsLp& lp = *C(c).model;
  lp.num_col_ = m->num_col;
  lp.num_row_ = m->num_row;
  lp.col_cost_.assign(m->col_cost, m->col_cost + m->num_col);
  lp.col_lower_.assign(m->col_lower, m->col_lower + m->num_col);
  lp.col_upper_.assign(m->col_upper, m->col_upper + m->num_col);
  lp.row_lower_.assign(m->row_lower, m->row_lower + m->num_row);
  lp.row_upper_.assign(m->row_upper, m->row_upper + m->num_row);
  lp.integrality_.resize(m->num_col);
  for (HighsInt i = 0; i < m->num_col; i++)
    lp.integrality_[i] = HighsVarType(m->integrality[i]);
  lp.offset_ = m->offset;
  lp.sense_ = m->maximize ? ObjSense::kMaximize : ObjSense::kMinimize;
  if (resize_row_names) lp.row_names_.resize(lp.num_row_);
}

void cbSetMatrix(Cb c, const HighsInt* start, size_t nstart,
                 const HighsInt* index, const double* value, size_t nnz) {
  HighsSparseMatrix& a = C(c).model->a_matrix_;
  a.start_.assign(start, start + nstart);
  a.index_.assign(index, index + nnz);
  a.value_.assign(value, value + nnz);
}

void cbFlush(Cb c, const char* data, size_t len, const uint8_t* types,
             const size_t* pos, size_t n, const HighsInt* nt, size_t nnt) {
  C(c).stack->rustAppend(data, len, types, pos, n, nt, nnt);
}

void cbShrink(Cb c, const HighsInt* new_col, size_t nc, const HighsInt* new_row,
              size_t nr) {
  Ctx& x = C(c);
  HighsLp& lp = *x.model;
  std::vector<HighsInt> newColIndex(new_col, new_col + nc);
  std::vector<HighsInt> newRowIndex(new_row, new_row + nr);
  if (lp.col_names_.size() > 0) {
    for (size_t i = 0; i != nc; ++i)
      if (newColIndex[i] != -1 && newColIndex[i] < HighsInt(i))
        lp.col_names_[newColIndex[i]] = std::move(lp.col_names_[i]);
    lp.col_names_.resize(lp.num_col_);
  }
  if (lp.row_names_.size() > 0) {
    for (size_t i = 0; i != nr; ++i)
      if (newRowIndex[i] != -1 && newRowIndex[i] < HighsInt(i))
        lp.row_names_[newRowIndex[i]] = std::move(lp.row_names_[i]);
    lp.row_names_.resize(lp.num_row_);
  }
  x.stack->compressIndexMaps(newRowIndex, newColIndex);
  lp.setMatrixDimensions();
}

RsOptions rsOptions(const HighsOptions& options) {
  RsOptions o;
  rsOptionsTemplate(options, 4, &o);
  return o;
}

HighsInt cbDependentEquations(Cb c, size_t num_col, HighsInt num_row,
                              const HighsInt* start, const HighsInt* index,
                              const double* value, size_t nnz,
                              double time_limit, double* time_taken,
                              RsSlice<HighsInt>* var_with_no_pivot) {
  Ctx& x = C(c);
  const HighsInt build_return =
      rsDependentEquations(*x.timer, x.ints, num_col, num_row, start, index,
                           value, nnz, time_limit, time_taken);
  *var_with_no_pivot = {x.ints.data(), x.ints.size()};
  return build_return;
}

// The MIP presolve's callbacks: the MIP presolve is Rust's
// (rust/src/mip/host/presolve.rs), so an LP presolve never calls them
[[noreturn]] void mipOnly() { abort(); }
void cbProfiling(Cb, bool, HighsInt) { mipOnly(); }
bool cbProbingPrepare(Cb, HighsInt, bool*) { mipOnly(); }
void cbMipEnv(Cb, void*) { mipOnly(); }
bool cbProbe(Cb, HighsInt, HighsInt*) { mipOnly(); }
void cbSetLifting(Cb, bool) { mipOnly(); }
void cbLiftingOpps(Cb, RsSlice<RsLiftOpp>*) { mipOnly(); }
void cbFinaliseBegin(Cb, bool, RsSlice<HighsInt>*, RsSlice<HighsInt>*) {
  mipOnly();
}
void cbDomainBounds(Cb, RsSlice<double>*, RsSlice<double>*) { mipOnly(); }
void cbMipFinishPresolve(Cb, HighsInt) { mipOnly(); }
void cbAddCut(Cb, const HighsInt*, const double*, size_t, double, bool) {
  mipOnly();
}
double cbUpperLimit(Cb) { mipOnly(); }
void cbSetLowerBoundZero(Cb) { mipOnly(); }
}  // namespace

extern "C" HighsInt highs_rs_presolve_run(const RsHost* host,
                                          const RsOptions* opt,
                                          const void* mip,
                                          const RsInput* inp, RsOut* out);

bool HPresolve::okSetInput(HighsLp& model_, const HighsOptions& options_,
                           const HighsInt presolve_reduction_limit,
                           HighsTimer* timer) {
  model = &model_;
  options = &options_;
  this->timer = timer;
  if (mipsolver == nullptr) {
    primal_feastol = options->primal_feasibility_tolerance;
    model->integrality_.assign(model->num_col_, HighsVarType::kContinuous);
  } else
    primal_feastol = options->mip_feasibility_tolerance;
  reductionLimit =
      presolve_reduction_limit < 0 ? kHighsSize_tInf : presolve_reduction_limit;
  return true;
}

HighsModelStatus HPresolve::run(HighsPostsolveStack& postsolve_stack) {
  postsolve_stack.debug_prev_numreductions = 0;
  postsolve_stack.debug_prev_col_lower = 0;
  postsolve_stack.debug_prev_col_upper = 0;
  postsolve_stack.debug_prev_row_lower = 0;
  postsolve_stack.debug_prev_row_upper = 0;
  if (model->a_matrix_.isRowwise()) model->a_matrix_.ensureColwise();
  assert(model->a_matrix_.numNz() || model->num_row_ == 0);

  Ctx ctx{this, model, options, timer, &postsolve_stack, {}, {}, {}};
  RsHost host{&ctx,
              cbLog,
              cbTimerRead,
              cbTimeString,
              cbSyncModel,
              cbSetMatrix,
              cbFlush,
              cbShrink,
              cbDependentEquations,
              cbProfiling,
              cbProbingPrepare,
              cbMipEnv,
              cbProbe,
              cbSetLifting,
              cbLiftingOpps,
              cbFinaliseBegin,
              cbDomainBounds,
              cbMipFinishPresolve,
              cbAddCut,
              cbUpperLimit,
              cbSetLowerBoundZero};
  const RsOptions o = rsOptions(*options);
  const PostsolveRsStack s = postsolve_stack.rustStack();
  static_assert(sizeof(HighsVarType) == sizeof(uint8_t), "");
  RsInput in{model->num_col_,
             model->num_row_,
             model->col_cost_.data(),
             model->col_lower_.data(),
             model->col_upper_.data(),
             model->row_lower_.data(),
             model->row_upper_.data(),
             reinterpret_cast<const uint8_t*>(model->integrality_.data()),
             model->offset_,
             model->sense_ == ObjSense::kMaximize,
             model->a_matrix_.start_.data(),
             model->a_matrix_.index_.data(),
             model->a_matrix_.value_.data(),
             model->a_matrix_.value_.size(),
             s.orig_col_index,
             s.num_col,
             s.orig_row_index,
             s.num_row,
             s.data_len,
             s.num_reductions,
             reductionLimit == kHighsSize_tInf ? HighsInt{-1}
                                               : HighsInt(reductionLimit),
             model->model_name_.c_str()};
  RsOut result{};
  const HighsInt status = highs_rs_presolve_run(
      &host, &o, nullptr, &in, &result);
  presolve_status_ = HighsPresolveStatus(result.presolve_status);
  analysis_.presolve_log_.rule.resize(kPresolveRuleCount);
  for (HighsInt r = 0; r < kPresolveRuleCount; r++) {
    analysis_.presolve_log_.rule[r].call = result.log[r][0];
    analysis_.presolve_log_.rule[r].col_removed = result.log[r][1];
    analysis_.presolve_log_.rule[r].row_removed = result.log[r][2];
  }
  return HighsModelStatus(status);
}

}  // namespace presolve

#endif  // HIGHS_RUST
