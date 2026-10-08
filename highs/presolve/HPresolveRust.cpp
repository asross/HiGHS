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
#include "mip/HighsCliqueTable.h"
#include "mip/HighsCliqueTableRust.h"
#include "mip/HighsDomainRust.h"
#include "mip/HighsImplications.h"
#include "mip/HighsMipSolverData.h"
#include "mip/HighsObjectiveFunction.h"
#include "mip/MipTimer.h"
#include "presolve/HighsPostsolveStack.h"
#include "util/HFactor.h"

static_assert(sizeof(HighsInt) == 4, "the Rust port uses 32-bit HighsInt");
static_assert(sizeof(HighsVarType) == 1, "integrality is read as bytes");
static_assert(sizeof(HighsSubstitution) == 24, "HighsSubstitution layout");

namespace presolve {

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

struct RsMipInfo {
  double epsilon;
  double feastol;
  highs_rs::CliqueTable* cliquetable;
  highs_rs::Implications* implications;
  HighsInt orig_num_row;
  HighsInt num_restarts;
  bool submip;
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
  HighsMipSolver* mipsolver;
  HighsPostsolveStack* stack;
  // scratch arrays handed to Rust
  std::vector<HighsInt> ints;
  std::vector<HighsInt> ints2;
  std::vector<RsLiftOpp> lifting;
};

using Cb = void*;

struct RsMipEnv {
  const highs_rs::Domain* domain;
  highs_rs::CliqueDom cdom;
  highs_rs::CliqueMip cmip;
};

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
  void (*mip_env)(Cb, RsMipEnv*);
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
  HighsMipSolver* mipsolver = x.mipsolver;
  if (mipsolver != nullptr) {
    mipsolver->mipdata_->rowMatrixSet = false;
    mipsolver->mipdata_->objectiveFunction = HighsObjectiveFunction(*mipsolver);
    mipsolver->mipdata_->getDomain() = HighsDomain(*mipsolver);
    mipsolver->mipdata_->cliquetable.rebuild(lp.num_col_, *x.stack,
                                             mipsolver->mipdata_->getDomain(),
                                             newColIndex, newRowIndex);
    mipsolver->mipdata_->implications.rebuild(lp.num_col_, newColIndex,
                                              newRowIndex);
    mipsolver->mipdata_->getCutPool() =
        HighsCutPool(mipsolver->model_->num_col_,
                     mipsolver->options_mip_->mip_pool_age_limit,
                     mipsolver->options_mip_->mip_pool_soft_limit, 0);
    mipsolver->mipdata_->getConflictPool() =
        HighsConflictPool(5 * mipsolver->options_mip_->mip_pool_age_limit,
                          mipsolver->options_mip_->mip_pool_soft_limit);
    mipsolver->mipdata_->debugSolution.shrink(newColIndex);
    lp.setMatrixDimensions();
  }
  lp.setMatrixDimensions();
}

HighsInt cbDependentEquations(Cb c, size_t num_col, HighsInt num_row,
                              const HighsInt* start, const HighsInt* index,
                              const double* value, size_t nnz,
                              double time_limit, double* time_taken,
                              RsSlice<HighsInt>* var_with_no_pivot) {
  Ctx& x = C(c);
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
  *time_taken = -x.timer->read();
  HighsInt build_return = factor.build();
  *time_taken += x.timer->read();
  x.ints = factor.var_with_no_pivot;
  *var_with_no_pivot = {x.ints.data(), x.ints.size()};
  return build_return;
}

void cbProfiling(Cb c, bool start, HighsInt clock) {
  HighsMipSolver* mipsolver = C(c).mipsolver;
  const HighsInt k =
      clock == 0 ? kMipClockProbingPresolve : kMipClockEnumerationPresolve;
  if (start)
    mipsolver->profiling_->start(k);
  else
    mipsolver->profiling_->stop(k);
}

// the C++ part of prepareProbing; true if infeasible
bool cbProbingPrepare(Cb c, HighsInt nnz, bool* first_call) {
  Ctx& x = C(c);
  HighsMipSolver* mipsolver = x.mipsolver;
  HighsDomain& domain = mipsolver->mipdata_->getDomain();
  HighsCliqueTable& cliquetable = mipsolver->mipdata_->cliquetable;
  (void)nnz;
  mipsolver->mipdata_->setupDomainPropagation();
  const bool firstCall = !mipsolver->mipdata_->cliquesExtracted;
  *first_call = firstCall;
  domain.propagate();
  if (domain.infeasible()) return true;
  if (firstCall) {
    mipsolver->mipdata_->cliquesExtracted = true;
    cliquetable.extractCliques(*mipsolver);
    if (domain.infeasible()) return true;
    if (mipsolver->mipdata_->upper_limit != kHighsInf) {
      double tmpLimit = mipsolver->mipdata_->upper_limit;
      mipsolver->mipdata_->upper_limit = tmpLimit - x.model->offset_;
      cliquetable.extractObjCliques(*mipsolver);
      mipsolver->mipdata_->upper_limit = tmpLimit;
      if (domain.infeasible()) return true;
    }
    domain.propagate();
    if (domain.infeasible()) return true;
  }
  cliquetable.cleanupFixed(domain);
  if (domain.infeasible()) return true;
  return false;
}

// the domain view, clique table and MIP solver contexts of the probing and
// enumeration loops (rust/src/presolve/hpresolve/probing.rs, enumeration.rs)
void cbMipEnv(Cb c, RsMipEnv* env) {
  HighsMipSolver* mipsolver = C(c).mipsolver;
  HighsDomain& domain = mipsolver->mipdata_->getDomain();
  env->domain = highs_rs::DomainAccess::view(domain);
  env->cdom = highs_rs::cliqueDom(domain);
  env->cmip = highs_rs::cliqueMip(*mipsolver);
}

bool cbProbe(Cb c, HighsInt col, HighsInt* num_bound_chgs) {
  return C(c).mipsolver->mipdata_->implications.runProbing(col,
                                                           *num_bound_chgs);
}

// collect the lifting opportunities of probing (on) or stop (off)
void cbSetLifting(Cb c, bool on) {
  Ctx& x = C(c);
  HighsImplications& implications = x.mipsolver->mipdata_->implications;
  if (!on) {
    implications.storeLiftingOpportunity = nullptr;
    return;
  }
  x.lifting.clear();
  implications.storeLiftingOpportunity = [&x](HighsInt row, HighsInt col,
                                              HighsInt val, double coef) {
    x.lifting.push_back({row, col, val, coef});
  };
}

void cbLiftingOpps(Cb c, RsSlice<RsLiftOpp>* out) {
  Ctx& x = C(c);
  *out = {x.lifting.data(), x.lifting.size()};
}

void cbFinaliseBegin(Cb c, bool first_call, RsSlice<HighsInt>* deleted,
                     RsSlice<HighsInt>* extensions) {
  Ctx& x = C(c);
  HighsMipSolver* mipsolver = x.mipsolver;
  HighsDomain& domain = mipsolver->mipdata_->getDomain();
  HighsCliqueTable& cliquetable = mipsolver->mipdata_->cliquetable;
  cliquetable.cleanupFixed(domain);
  if (!first_call) cliquetable.extractCliques(*mipsolver, false);
  cliquetable.runCliqueMerging(domain);
  x.ints.assign(cliquetable.getDeletedRows().begin(),
                cliquetable.getDeletedRows().end());
  cliquetable.getDeletedRows().clear();
  x.ints2.clear();
  for (const auto& ext : cliquetable.getCliqueExtensions()) {
    x.ints2.push_back(ext.first);
    x.ints2.push_back(ext.second.col);
    x.ints2.push_back(ext.second.val);
  }
  cliquetable.getCliqueExtensions().clear();
  *deleted = {x.ints.data(), x.ints.size()};
  *extensions = {x.ints2.data(), x.ints2.size()};
}

void cbDomainBounds(Cb c, RsSlice<double>* lower, RsSlice<double>* upper) {
  const HighsDomain& domain = C(c).mipsolver->mipdata_->getDomain();
  *lower = {domain.col_lower_.data(), domain.col_lower_.size()};
  *upper = {domain.col_upper_.data(), domain.col_upper_.size()};
}

void cbMipFinishPresolve(Cb c, HighsInt nnz) {
  HighsMipSolver* mipsolver = C(c).mipsolver;
  mipsolver->mipdata_->cliquetable.setPresolveFlag(false);
  mipsolver->mipdata_->cliquetable.setMaxEntries(nnz);
  mipsolver->mipdata_->getDomain().addCutpool(
      mipsolver->mipdata_->getCutPool());
  mipsolver->mipdata_->getDomain().addConflictPool(
      mipsolver->mipdata_->getConflictPool());
}

void cbAddCut(Cb c, const HighsInt* inds, const double* vals, size_t n,
              double rhs, bool integral) {
  HighsMipSolver* mipsolver = C(c).mipsolver;
  std::vector<HighsInt> cutinds(inds, inds + n);
  std::vector<double> cutvals(vals, vals + n);
  mipsolver->mipdata_->getCutPool().addCut(*mipsolver, cutinds.data(),
                                           cutvals.data(), n, rhs, integral,
                                           true, false, false);
}

double cbUpperLimit(Cb c) { return C(c).mipsolver->mipdata_->upper_limit; }

void cbSetLowerBoundZero(Cb c) { C(c).mipsolver->mipdata_->lower_bound = 0; }

}  // namespace

extern "C" HighsInt highs_rs_presolve_run(const RsHost* host,
                                          const RsOptions* opt,
                                          const RsMipInfo* mip,
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

bool HPresolve::okSetInput(HighsMipSolver& mipsolver,
                           const HighsInt presolve_reduction_limit) {
  this->mipsolver = &mipsolver;
  if (mipsolver.model_ != &mipsolver.mipdata_->presolvedModel) {
    mipsolver.mipdata_->presolvedModel = *mipsolver.model_;
    mipsolver.model_ = &mipsolver.mipdata_->presolvedModel;
  } else {
    mipsolver.mipdata_->presolvedModel.col_lower_ =
        mipsolver.mipdata_->getDomain().col_lower_;
    mipsolver.mipdata_->presolvedModel.col_upper_ =
        mipsolver.mipdata_->getDomain().col_upper_;
  }
  return okSetInput(mipsolver.mipdata_->presolvedModel, *mipsolver.options_mip_,
                    presolve_reduction_limit, &mipsolver.timer_);
}

HighsModelStatus HPresolve::run(HighsPostsolveStack& postsolve_stack) {
  postsolve_stack.debug_prev_numreductions = 0;
  postsolve_stack.debug_prev_col_lower = 0;
  postsolve_stack.debug_prev_col_upper = 0;
  postsolve_stack.debug_prev_row_lower = 0;
  postsolve_stack.debug_prev_row_upper = 0;
  if (model->a_matrix_.isRowwise()) model->a_matrix_.ensureColwise();
  assert(model->a_matrix_.numNz() || model->num_row_ == 0);

  Ctx ctx{this, model, options, timer, mipsolver, &postsolve_stack, {}, {},
          {}};
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
  RsOptions o{options->primal_feasibility_tolerance,
              options->dual_feasibility_tolerance,
              options->mip_feasibility_tolerance,
              options->small_matrix_value,
              options->time_limit,
              options->presolve_pivot_threshold,
              options->presolve_substitution_maxfillin,
              options->presolve_rule_test,
              options->presolve_rule_off,
              options->log_dev_level,
              options->random_seed,
              options->mip_lifting_for_probing,
              options->presolve == kHighsOffString,
              options->lp_presolve_requires_basis_postsolve,
              options->presolve_remove_slacks,
              options->output_flag,
              options->timeless_log,
              options->use_implied_bounds_from_presolve,
              options->presolve_rule_logging};
  RsMipInfo mi{};
  if (mipsolver != nullptr) {
    mi.epsilon = mipsolver->mipdata_->epsilon;
    mi.feastol = mipsolver->mipdata_->feastol;
    mi.cliquetable = mipsolver->mipdata_->cliquetable.rust();
    mi.implications = mipsolver->mipdata_->implications.rust();
    mi.orig_num_row = mipsolver->orig_model_->num_row_;
    mi.num_restarts = mipsolver->mipdata_->numRestarts;
    mi.submip = mipsolver->submip;
  }
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
      &host, &o, mipsolver != nullptr ? &mi : nullptr, &in, &result);
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
