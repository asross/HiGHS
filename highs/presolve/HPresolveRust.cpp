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
 * the presolve rule analysis setup, the HighsLp and postsolve stack
 * updates, the dependent equations' HFactor, and the MIP solver's clique
 * table, implications, domain and pools (with the probing and enumeration
 * loops on them).
 */
#include "presolve/HPresolve.h"

#ifdef HIGHS_RUST

#include <numeric>

#include "../extern/pdqsort/pdqsort.h"
#include "io/HighsIO.h"
#include "lp_data/HighsLpUtils.h"
#include "mip/HighsCliqueTable.h"
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
};

struct RsMipInfo {
  double epsilon;
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

struct RsProbingIo {
  uint16_t* num_probes;
  HighsInt num_col;
  HighsInt num_row;
  HighsInt num_nonzeros;
  const uint8_t* col_deleted;
  const uint8_t* integrality;
  size_t colsize_len;
  int64_t* probing_contingent;
  HighsInt* num_probed;
  HighsInt* probing_num_del_col;
  bool* probing_early_abort;
  const RsLiftOpp* lifting;
  size_t num_lifting;
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
  std::vector<size_t> sizes;
  std::vector<RsLiftOpp> lifting;
};

using Cb = void*;

struct RsHost {
  Cb ctx;
  void (*log)(Cb, HighsInt, HighsInt, const char*);
  double (*timer_read)(Cb);
  void (*time_string)(Cb, double, char*, size_t);
  bool (*analysis_setup)(Cb, bool, uint8_t*);
  void (*sync_model)(Cb, const RsModel*, bool);
  void (*set_matrix)(Cb, const HighsInt*, size_t, const HighsInt*,
                     const double*, size_t);
  void (*flush)(Cb, const char*, size_t, const uint8_t*, const size_t*, size_t,
                const HighsInt*, size_t);
  void (*shrink)(Cb, const HighsInt*, size_t, const HighsInt*, size_t);
  HighsInt (*dependent_equations)(Cb, size_t, HighsInt, const HighsInt*,
                                  const HighsInt*, const double*, size_t,
                                  double, double*, RsSlice<HighsInt>*);
  bool (*clique_have_common_clique)(Cb, HighsInt, HighsInt, HighsInt,
                                    HighsInt);
  HighsInt (*clique_num_cliques_col)(Cb, HighsInt, HighsInt);
  HighsInt (*clique_num_cliques)(Cb);
  void (*clique_set_presolve_flag)(Cb, bool);
  void (*clique_set_max_entries)(Cb, HighsInt);
  void (*implications_column_transformed)(Cb, HighsInt, double, double);
  void (*implications_add_vb)(Cb, bool, HighsInt, HighsInt, double, double,
                              double, bool);
  void (*profiling)(Cb, bool, HighsInt);
  bool (*probing_prepare)(Cb, HighsInt, bool*);
  HighsInt (*probing_loop)(Cb, RsProbingIo*);
  void (*finalise_begin)(Cb, bool, RsSlice<HighsInt>*, RsSlice<HighsInt>*);
  void (*domain_bounds)(Cb, RsSlice<double>*, RsSlice<double>*);
  void (*impl_substitutions)(Cb, RsSlice<HighsSubstitution>*);
  void (*clear_impl_substitutions)(Cb);
  void (*clique_substitutions)(Cb, RsSlice<HighsInt>*);
  void (*clear_clique_substitutions)(Cb);
  void (*compute_maximal_cliques)(Cb, const HighsInt*, size_t, double,
                                  RsSlice<HighsInt>*, RsSlice<size_t>*);
  bool (*enumerate)(Cb, RsProbingIo*);
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

bool cbAnalysisSetup(Cb c, bool silent, uint8_t* allow) {
  Ctx& x = C(c);
  HPresolveAnalysis& a = x.presolve->rustAnalysis();
  a.setup(x.model, x.options, x.presolve->rustNumDeletedRows(),
          x.presolve->rustNumDeletedCols(), silent);
  for (HighsInt i = 0; i < kPresolveRuleCount; i++)
    allow[i] = a.allow_rule_[i] ? 1 : 0;
  return a.allow_logging_;
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

bool cbHaveCommonClique(Cb c, HighsInt c1, HighsInt v1, HighsInt c2,
                        HighsInt v2) {
  return C(c).mipsolver->mipdata_->cliquetable.haveCommonClique(
      HighsCliqueTable::CliqueVar(c1, v1), HighsCliqueTable::CliqueVar(c2, v2));
}

HighsInt cbNumCliquesCol(Cb c, HighsInt col, HighsInt val) {
  return C(c).mipsolver->mipdata_->cliquetable.numCliques(col, val);
}

HighsInt cbNumCliques(Cb c) {
  return C(c).mipsolver->mipdata_->cliquetable.numCliques();
}

void cbSetPresolveFlag(Cb c, bool f) {
  C(c).mipsolver->mipdata_->cliquetable.setPresolveFlag(f);
}

void cbSetMaxEntries(Cb c, HighsInt n) {
  C(c).mipsolver->mipdata_->cliquetable.setMaxEntries(n);
}

void cbColumnTransformed(Cb c, HighsInt col, double scale, double constant) {
  C(c).mipsolver->mipdata_->implications.columnTransformed(col, scale,
                                                           constant);
}

void cbAddVb(Cb c, bool is_vub, HighsInt col, HighsInt bin_col, double coef,
             double constant, double bound, bool is_int) {
  HighsImplications& implications = C(c).mipsolver->mipdata_->implications;
  if (is_vub)
    implications.addVUB(col, bin_col, coef, constant, bound, is_int);
  else
    implications.addVLB(col, bin_col, coef, constant, bound, is_int);
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

std::pair<int64_t, HighsInt> probingScore(HighsMipSolver* mipsolver,
                                          const uint16_t* numProbes,
                                          HighsInt col) {
  HighsInt implicsUp =
      mipsolver->mipdata_->cliquetable.getNumImplications(col, 1);
  HighsInt implicsDown =
      mipsolver->mipdata_->cliquetable.getNumImplications(col, 0);
  return std::make_pair(
      std::min(int64_t{5000}, static_cast<int64_t>(implicsUp) * implicsDown) /
          (int64_t{1} + static_cast<int64_t>(numProbes[col])),
      std::min(HighsInt{100}, implicsUp + implicsDown));
}

// the loop of runProbing: 0 ok, 1 infeasible (profiling stopped), 2 time
// limit, 3 no binaries
HighsInt cbProbingLoop(Cb c, RsProbingIo* io) {
  Ctx& x = C(c);
  HighsMipSolver* mipsolver = x.mipsolver;
  const HighsOptions* options = x.options;
  HighsDomain& domain = mipsolver->mipdata_->getDomain();
  HighsCliqueTable& cliquetable = mipsolver->mipdata_->cliquetable;
  HighsImplications& implications = mipsolver->mipdata_->implications;
  x.lifting.clear();
  io->lifting = nullptr;
  io->num_lifting = 0;

  const HighsInt oldNumProbed = *io->num_probed;
  std::vector<std::tuple<int64_t, HighsInt, HighsInt, HighsInt>> binaries;
  if (!cliquetable.isFull()) {
    binaries.reserve(io->num_col);
    HighsRandom random(options->random_seed);
    for (HighsInt i = 0; i != io->num_col; ++i) {
      if (domain.isBinary(i)) {
        auto probingScore_ = probingScore(mipsolver, io->num_probes, i);
        binaries.emplace_back(-probingScore_.first, -probingScore_.second,
                              random.integer(), i);
      }
    }
  }
  if (binaries.empty()) return 3;

  pdqsort(binaries.begin(), binaries.end());

  size_t numChangedCols = 0;
  while (domain.getChangedCols().size() != numChangedCols) {
    if (domain.isFixed(domain.getChangedCols()[numChangedCols++]))
      ++*io->probing_num_del_col;
  }

  HighsInt numCliquesStart = cliquetable.numCliques();
  HighsInt numImplicsStart = implications.getNumImplications();
  HighsInt numDelStart = *io->probing_num_del_col;

  auto calcNumDel = [&]() {
    return *io->probing_num_del_col - numDelStart +
           static_cast<HighsInt>(implications.substitutions.size() +
                                 cliquetable.getSubstitutions().size());
  };

  HighsInt numDel = calcNumDel();
  const HighsInt numNonzeros = io->num_nonzeros;
  int64_t splayContingent =
      cliquetable.numNeighbourhoodQueries +
      std::max(mipsolver->submip ? HighsInt{0} : HighsInt{100000},
               10 * numNonzeros);
  HighsInt numFail = 0;

  auto modelHasPercentageContVars = [&](size_t percentage) {
    size_t num_cols = 0, num_cont_cols = 0;
    for (size_t col = 0; col < io->colsize_len; col++) {
      if (io->col_deleted[col]) continue;
      num_cols++;
      if (HighsVarType(io->integrality[col]) == HighsVarType::kContinuous)
        num_cont_cols++;
    }
    return size_t{100} * num_cont_cols >= percentage * num_cols;
  };

  const size_t maxNumLiftOpps = std::max(
      size_t{100000}, size_t{10} * static_cast<size_t>(io->num_row));
  size_t numLiftOpps = 0;
  if (mipsolver->options_mip_->mip_lifting_for_probing != -1 &&
      modelHasPercentageContVars(size_t{2})) {
    implications.storeLiftingOpportunity = [&x, &numLiftOpps](
                                               HighsInt row, HighsInt col,
                                               HighsInt val, double coef) {
      x.lifting.push_back({row, col, val, coef});
      numLiftOpps++;
    };
  }

  const bool silent =
      mipsolver != nullptr && mipsolver->mipdata_->numRestarts > 0;
  HighsInt iBin = -1;
  HighsInt iBin_probed = -1;
  HighsInt num_binary = binaries.size();
  double tt = x.timer->read();
  double tt0 = tt;
  double log_tt = tt0;
  HighsInt log_iBin_probed = iBin_probed;
  auto probingLog = [&]() {
    if (silent || options->timeless_log) return;
    const double log_tt_interval = 5.0;
    if (tt > log_tt + log_tt_interval && iBin_probed > log_iBin_probed) {
      assert(iBin_probed > 0);
      double rate0 = (tt - tt0) / double(iBin_probed);
      HighsInt dl_iBin_probed = iBin_probed - log_iBin_probed;
      assert(dl_iBin_probed > 0);
      double rate1 = (tt - log_tt) / double(dl_iBin_probed);
      double rate = std::max(rate0, rate1);
      std::string rate_str =
          " (rate " + highsTimeToString(1e3 * rate) + "/ms";
      double expected_probing_finish_time =
          tt + rate * (num_binary - iBin_probed);
      std::string expected_probing_finish_time_str =
          " => expected probing finish time " +
          highsTimeSecondToString(expected_probing_finish_time) + ")";
      std::string time_str = highsTimeSecondToString(tt);
      highsLogUser(options->log_options, HighsLogType::kInfo,
                   "   Considered %d / %d binaries; %d probed %s%s %s\n",
                   int(iBin), int(num_binary), int(iBin_probed),
                   rate_str.c_str(), expected_probing_finish_time_str.c_str(),
                   time_str.c_str());
      log_tt = tt;
      log_iBin_probed = iBin_probed;
    }
  };

  HighsInt result = 0;
  for (const auto& binvar : binaries) {
    iBin++;
    HighsInt i = std::get<3>(binvar);
    if (cliquetable.getSubstitution(i) != nullptr || !domain.isBinary(i))
      continue;
    iBin_probed++;
    tt = x.timer->read();
    if (tt > options->time_limit) {
      highsLogUser(options->log_options, HighsLogType::kInfo,
                   "Time limit reached in probing: "
                   "consider not using probing by setting option "
                   "presolve_rule_off to 2^%-d = %d\n",
                   int(kPresolveRuleProbing),
                   int(std::pow(int(2), int(kPresolveRuleProbing))));
      result = 2;
      break;
    }
    probingLog();
    bool tightenLimits = (*io->num_probed - oldNumProbed) >= 2500;
    if (!tightenLimits) {
      *io->probing_early_abort =
          numDel >
          std::max(HighsInt{1000}, (io->num_row + io->num_col) / 20);
    } else {
      *io->probing_early_abort =
          numDel >
          std::min(HighsInt{1000}, (io->num_row + io->num_col) / 20);
    }
    if (*io->probing_early_abort) break;
    if (cliquetable.isFull() ||
        cliquetable.numCliques() - numCliquesStart >
            std::max(HighsInt{1000000}, 2 * numNonzeros) ||
        implications.getNumImplications() - numImplicsStart >
            std::max(HighsInt{1000000}, 2 * numNonzeros))
      break;
    if (cliquetable.numNeighbourhoodQueries > splayContingent) break;
    if (*io->probing_contingent - *io->num_probed < 0) break;

    HighsInt numBoundChgs = 0;
    HighsInt numNewCliques = -cliquetable.numCliques();
    const bool probing_result = implications.runProbing(i, numBoundChgs);
    if (!probing_result) continue;
    *io->probing_contingent += numBoundChgs;
    numNewCliques += cliquetable.numCliques();
    numNewCliques = std::max(numNewCliques, HighsInt{0});
    while (domain.getChangedCols().size() != numChangedCols) {
      if (domain.isFixed(domain.getChangedCols()[numChangedCols++]))
        ++*io->probing_num_del_col;
    }
    HighsInt newNumDel = calcNumDel();
    if (newNumDel > numDel) {
      *io->probing_contingent += numDel;
      if (!mipsolver->submip) {
        splayContingent += 100 * (newNumDel + numDelStart);
        splayContingent += 1000 * numNewCliques;
      }
      numDel = newNumDel;
      numFail = 0;
    } else if (mipsolver->submip || numNewCliques == 0) {
      splayContingent -= (tightenLimits ? 250 : 100) * numFail;
      ++numFail;
    } else {
      splayContingent += 1000 * numNewCliques;
      numFail = 0;
    }
    ++*io->num_probed;
    io->num_probes[i] += 1;
    if (numLiftOpps >= maxNumLiftOpps)
      implications.storeLiftingOpportunity = nullptr;
    if (domain.infeasible()) {
      mipsolver->profiling_->stop(kMipClockProbingPresolve);
      result = 1;
      break;
    }
  }
  implications.storeLiftingOpportunity = nullptr;
  io->lifting = x.lifting.data();
  io->num_lifting = x.lifting.size();
  return result;
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

void cbImplSubstitutions(Cb c, RsSlice<HighsSubstitution>* s) {
  const auto& subs = C(c).mipsolver->mipdata_->implications.substitutions;
  *s = {subs.begin(), subs.size()};
}

void cbClearImplSubstitutions(Cb c) {
  C(c).mipsolver->mipdata_->implications.substitutions.clear();
}

void cbCliqueSubstitutions(Cb c, RsSlice<HighsInt>* s) {
  Ctx& x = C(c);
  x.ints.clear();
  for (const auto& subst :
       x.mipsolver->mipdata_->cliquetable.getSubstitutions()) {
    x.ints.push_back(subst.substcol);
    x.ints.push_back(subst.replace.col);
    x.ints.push_back(subst.replace.val);
  }
  *s = {x.ints.data(), x.ints.size()};
}

void cbClearCliqueSubstitutions(Cb c) {
  C(c).mipsolver->mipdata_->cliquetable.getSubstitutions().clear();
}

void cbComputeMaximalCliques(Cb c, const HighsInt* pairs, size_t n,
                             double feastol, RsSlice<HighsInt>* vars,
                             RsSlice<size_t>* starts) {
  Ctx& x = C(c);
  std::vector<HighsCliqueTable::CliqueVar> candidates;
  candidates.reserve(n);
  for (size_t i = 0; i != n; ++i)
    candidates.emplace_back(pairs[2 * i], pairs[2 * i + 1]);
  auto cliques = x.mipsolver->mipdata_->cliquetable.computeMaximalCliques(
      candidates, feastol);
  x.ints.clear();
  x.sizes.assign(1, 0);
  for (const auto& clique : cliques) {
    for (const auto& v : clique) {
      x.ints.push_back(v.col);
      x.ints.push_back(v.val);
    }
    x.sizes.push_back(x.ints.size() / 2);
  }
  *vars = {x.ints.data(), x.ints.size()};
  *starts = {x.sizes.data(), x.sizes.size()};
}

// the solution enumeration of enumerateSolutions, between prepareProbing and
// finaliseProbing; true if infeasible (profiling stopped)
bool cbEnumerate(Cb c, RsProbingIo* io) {
  Ctx& x = C(c);
  HighsMipSolver* mipsolver = x.mipsolver;
  const HighsOptions* options = x.options;
  HighsDomain& domain = mipsolver->mipdata_->getDomain();
  HighsCliqueTable& cliquetable = mipsolver->mipdata_->cliquetable;

  typedef std::tuple<double, double, HighsInt, HighsInt, uint32_t> candidateRow;
  const size_t maxRowSize = 8;
  const HighsInt maxNumRowsChecked = 400;
  const size_t maxNumSolutions = 1 << maxRowSize;
  const size_t maxPercentageRowOverlap = 50;
  const HighsInt maxNumFails = 6;

  auto getBinaryRow = [&](HighsInt row, std::vector<HighsInt>& binvars,
                          size_t& numnzs) {
    numnzs = 0;
    for (HighsInt j = mipsolver->mipdata_->ARstart_[row];
         j < mipsolver->mipdata_->ARstart_[row + 1]; j++) {
      HighsInt col = mipsolver->mipdata_->ARindex_[j];
      if (domain.isFixed(col)) continue;
      if (!domain.isBinary(col) || numnzs >= maxRowSize) return false;
      binvars[numnzs++] = col;
    }
    if (numnzs == 0) return false;
    pdqsort(binvars.begin(), binvars.begin() + numnzs);
    return true;
  };

  auto computeRowScore = [&](const std::vector<HighsInt>& binvars,
                             size_t numnzs) {
    int64_t score = 0;
    HighsInt score2 = 0;
    for (size_t i = 0; i < numnzs; i++) {
      auto probingScore_ = probingScore(mipsolver, io->num_probes, binvars[i]);
      score += probingScore_.first;
      score2 += probingScore_.second;
    }
    return std::make_pair<double, double>(score / static_cast<double>(numnzs),
                                          score2 / static_cast<double>(numnzs));
  };

  auto computeRowSignature = [&](const std::vector<HighsInt>& binvars,
                                 size_t numnzs) {
    uint32_t signature = 0;
    for (size_t i = 0; i < numnzs; i++) {
      HighsInt colHashedPos = (HighsHashHelpers::hash(binvars[i]) >> 59);
      assert(colHashedPos < 32);
      signature |= 1 << colHashedPos;
    }
    return signature;
  };

  auto computeRowOverlap = [&](const std::vector<HighsInt>& binvars,
                               const std::vector<HighsInt>& binvars2,
                               size_t numnzs, size_t numnzs2) {
    size_t overlap = 0;
    size_t ir = 0;
    size_t ir2 = 0;
    while (ir < numnzs && ir2 < numnzs2) {
      if (binvars[ir] < binvars2[ir2])
        ir++;
      else if (binvars[ir] > binvars2[ir2])
        ir2++;
      else {
        ir++;
        ir2++;
        overlap++;
      }
    }
    return overlap;
  };

  auto compileRows = [&](std::vector<candidateRow>& rows) {
    std::vector<HighsInt> binvars(maxRowSize);
    size_t numnzs = 0;
    HighsRandom random(options->random_seed);
    for (HighsInt i = 0; i < mipsolver->numRow(); i++) {
      if (domain.isRedundantRow(i)) continue;
      if (!getBinaryRow(i, binvars, numnzs)) continue;
      auto score = computeRowScore(binvars, numnzs);
      rows.emplace_back(-score.first, -score.second, random.integer(), i,
                        computeRowSignature(binvars, numnzs));
    }
  };

  auto removeSimilarRows = [&](std::vector<candidateRow>& rows) {
    if (rows.size() <= 1) return;
    std::vector<HighsInt> binvars(maxRowSize);
    std::vector<HighsInt> binvars2(maxRowSize);
    size_t numnzs;
    size_t numnzs2;
    HighsInt numRowsAccepted = 0;
    HighsInt numRowsRemoved = 0;
    size_t numComparisons = 0;
    for (size_t i = 0; i < rows.size() - 1; i++) {
      HighsInt r = std::get<3>(rows[i]);
      if (r == -1) continue;
      if ((++numRowsAccepted) >= maxNumRowsChecked) break;
      getBinaryRow(r, binvars, numnzs);
      HighsInt numRowsActive = 0;
      HighsInt oldNumRowsRemoved = numRowsRemoved;
      for (size_t ii = i + 1; ii < rows.size(); ii++) {
        HighsInt& r2 = std::get<3>(rows[ii]);
        if (r2 == -1) continue;
        numRowsActive++;
        if ((std::get<4>(rows[i]) & std::get<4>(rows[ii])) == 0) continue;
        getBinaryRow(r2, binvars2, numnzs2);
        numComparisons++;
        size_t overlap = computeRowOverlap(binvars, binvars2, numnzs, numnzs2);
        if ((100 * overlap) / std::min(numnzs, numnzs2) >
            maxPercentageRowOverlap) {
          numRowsRemoved++;
          r2 = -1;
        }
      }
      if (numRowsActive - numRowsRemoved + oldNumRowsRemoved <= 1) break;
    }
    (void)numComparisons;
    if (numRowsRemoved > 0)
      rows.erase(std::remove_if(rows.begin(), rows.end(),
                                [](const candidateRow& p) {
                                  return std::get<3>(p) == -1;
                                }),
                 rows.end());
    if (rows.size() > static_cast<size_t>(maxNumRowsChecked))
      rows.resize(maxNumRowsChecked);
  };

  std::vector<candidateRow> rows;
  rows.reserve(io->num_row);
  compileRows(rows);
  pdqsort(rows.begin(), rows.end());
  removeSimilarRows(rows);

  struct branch {
    size_t numDomainChanges;
    size_t numChangedCols;
  };
  std::vector<std::array<HighsInt, maxNumSolutions>> solutions(maxRowSize);
  std::vector<HighsInt> vars(maxRowSize);
  std::vector<branch> branches(maxRowSize);
  std::vector<HighsInt> worstCaseBounds(io->num_col);
  std::vector<double> worstCaseLowerBound(io->num_col, kHighsInf);
  std::vector<double> worstCaseUpperBound(io->num_col, -kHighsInf);
  std::vector<double> col_lower(domain.col_lower_);
  std::vector<double> col_upper(domain.col_upper_);

  auto findBranchVar = [&](size_t numVars) {
    for (size_t i = 0; i < numVars; i++)
      if (!domain.isFixed(vars[i])) return vars[i];
    return HighsInt{-1};
  };

  auto doBranch = [&](size_t numVars, HighsInt& numBranches) {
    HighsInt branchvar = findBranchVar(numVars);
    assert(branchvar >= 0);
    branches[++numBranches] = {domain.getDomainChangeStack().size(),
                               domain.getChangedCols().size()};
    domain.changeBound(HighsBoundType::kUpper, branchvar, 0);
  };

  auto doBacktrack = [&](HighsInt& numBranches) {
    while (numBranches >= 0) {
      const auto& domchg =
          domain.getDomainChangeStack()[branches[numBranches].numDomainChanges];
      HighsInt col = domchg.column;
      HighsBoundType bndtype = domchg.boundtype;
      domain.backtrack();
      domain.clearChangedCols(
          static_cast<HighsInt>(branches[numBranches].numChangedCols));
      if (bndtype == HighsBoundType::kUpper) {
        domain.changeBound(HighsBoundType::kLower, col, 1);
        break;
      } else {
        branches[numBranches--] = {0, 0};
      }
    }
    return (numBranches >= 0);
  };

  auto identicalVars = [&](size_t numSolutions, size_t index1, size_t index2) {
    for (size_t sol = 0; sol < numSolutions; sol++) {
      if (solutions[index1][sol] != solutions[index2][sol]) return false;
    }
    return true;
  };

  auto complementaryVars = [&](size_t numSolutions, size_t index1,
                               size_t index2) {
    for (size_t sol = 0; sol < numSolutions; sol++) {
      if (solutions[index1][sol] != 1 - solutions[index2][sol]) return false;
    }
    return true;
  };

  auto handleInfeasibility = [&](bool infeasible) {
    if (infeasible) {
      mipsolver->profiling_->stop(kMipClockEnumerationPresolve);
      return true;
    }
    return false;
  };

  auto removeWorstCaseBounds = [&](size_t pos, size_t& numWorstCaseBounds) {
    worstCaseLowerBound[worstCaseBounds[pos]] = kHighsInf;
    worstCaseUpperBound[worstCaseBounds[pos]] = -kHighsInf;
    worstCaseBounds[pos] = worstCaseBounds[numWorstCaseBounds - 1];
    worstCaseBounds[numWorstCaseBounds - 1] = 0;
    numWorstCaseBounds--;
  };

  auto updateWorstCaseBounds = [&](HighsInt col) {
    worstCaseLowerBound[col] =
        std::min(worstCaseLowerBound[col], domain.col_lower_[col]);
    worstCaseUpperBound[col] =
        std::max(worstCaseUpperBound[col], domain.col_upper_[col]);
    return (worstCaseLowerBound[col] <= col_lower[col] &&
            worstCaseUpperBound[col] >= col_upper[col]);
  };

  auto handleSolution = [&](size_t numVars, size_t& numSolutions,
                            size_t& numWorstCaseBounds,
                            size_t& minNumActiveCols,
                            size_t& maxNumActiveCols) {
    domain.propagate();
    if (domain.infeasible()) return;
    if (numSolutions == 0) {
      for (HighsInt col : domain.getChangedCols()) {
        worstCaseBounds[numWorstCaseBounds++] = col;
        updateWorstCaseBounds(col);
      }
    } else {
      size_t i = 0;
      while (i < numWorstCaseBounds) {
        HighsInt col = worstCaseBounds[i];
        if (!domain.isChangedCol(col)) {
          removeWorstCaseBounds(i, numWorstCaseBounds);
        } else {
          if (updateWorstCaseBounds(col))
            removeWorstCaseBounds(i, numWorstCaseBounds);
          else
            i++;
        }
      }
    }
    size_t numActiveCols = 0;
    for (size_t i = 0; i < numVars; i++) {
      HighsInt solValue = domain.col_lower_[vars[i]] == 0.0 ? 0 : 1;
      solutions[i][numSolutions] = solValue;
      if (solValue != 0) numActiveCols++;
    }
    minNumActiveCols = std::min(minNumActiveCols, numActiveCols);
    maxNumActiveCols = std::max(maxNumActiveCols, numActiveCols);
    numSolutions++;
  };

  HighsInt numCliquesFound = 0;
  HighsInt numFails = 0;
  for (const auto& r : rows) {
    HighsInt row = std::get<3>(r);
    if (domain.isRedundantRow(row)) continue;
    size_t numVars = 0;
    if (!getBinaryRow(row, vars, numVars)) continue;

    HighsInt numBranches = -1;
    size_t numWorstCaseBounds = 0;
    size_t numSolutions = 0;
    size_t minNumActiveCols = numVars;
    size_t maxNumActiveCols = 0;
    while (true) {
      bool backtrack = domain.infeasible();
      if (!backtrack) {
        backtrack = findBranchVar(numVars) < 0;
        if (backtrack)
          handleSolution(numVars, numSolutions, numWorstCaseBounds,
                         minNumActiveCols, maxNumActiveCols);
      }
      if (!backtrack)
        doBranch(numVars, numBranches);
      else if (!doBacktrack(numBranches))
        break;
    }

    if (handleInfeasibility(numSolutions == 0)) return true;

    size_t oldNumChangedCols = domain.getChangedCols().size();
    HighsInt oldNumCliques = cliquetable.numCliques();
    size_t oldNumSubstitutions = cliquetable.getSubstitutions().size();

    if (maxNumActiveCols == 1 || minNumActiveCols == numVars - 1) {
      numCliquesFound++;
      std::vector<HighsCliqueTable::CliqueVar> clique(numVars);
      for (size_t i = 0; i < numVars; i++)
        clique[i] =
            HighsCliqueTable::CliqueVar(vars[i], maxNumActiveCols == 1 ? 1 : 0);
      cliquetable.addClique(*mipsolver, clique.data(),
                            static_cast<HighsInt>(numVars),
                            minNumActiveCols == maxNumActiveCols);
      if (handleInfeasibility(domain.infeasible())) return true;
    }

    for (size_t i = 0; i < numWorstCaseBounds; i++) {
      HighsInt col = worstCaseBounds[i];
      if (worstCaseLowerBound[col] > domain.col_lower_[col]) {
        domain.changeBound(HighsBoundType::kLower, col,
                           worstCaseLowerBound[col],
                           HighsDomain::Reason::unspecified());
        if (handleInfeasibility(domain.infeasible())) return true;
      }
      if (worstCaseUpperBound[col] < domain.col_upper_[col]) {
        domain.changeBound(HighsBoundType::kUpper, col,
                           worstCaseUpperBound[col],
                           HighsDomain::Reason::unspecified());
        if (handleInfeasibility(domain.infeasible())) return true;
      }
      worstCaseLowerBound[col] = kHighsInf;
      worstCaseUpperBound[col] = -kHighsInf;
      worstCaseBounds[i] = 0;
    }

    for (size_t i = 0; i < numVars - 1; i++) {
      HighsInt col = vars[i];
      for (size_t ii = i + 1; ii < numVars; ii++) {
        HighsInt col2 = vars[ii];
        if (identicalVars(numSolutions, i, ii)) {
          numCliquesFound++;
          std::array<HighsCliqueTable::CliqueVar, 2> clique;
          clique[0] = HighsCliqueTable::CliqueVar(col, 0);
          clique[1] = HighsCliqueTable::CliqueVar(col2, 1);
          cliquetable.addClique(*mipsolver, clique.data(), 2, true);
          if (handleInfeasibility(domain.infeasible())) return true;
        } else if (complementaryVars(numSolutions, i, ii)) {
          numCliquesFound++;
          std::array<HighsCliqueTable::CliqueVar, 2> clique;
          clique[0] = HighsCliqueTable::CliqueVar(col, 0);
          clique[1] = HighsCliqueTable::CliqueVar(col2, 0);
          cliquetable.addClique(*mipsolver, clique.data(), 2, true);
          if (handleInfeasibility(domain.infeasible())) return true;
        }
      }
    }

    size_t numChangedCols = domain.getChangedCols().size();
    for (HighsInt col : domain.getChangedCols()) {
      col_lower[col] = domain.col_lower_[col];
      col_upper[col] = domain.col_upper_[col];
    }
    domain.clearChangedCols();

    if (numChangedCols != oldNumChangedCols ||
        cliquetable.numCliques() != oldNumCliques ||
        cliquetable.getSubstitutions().size() != oldNumSubstitutions)
      numFails = 0;
    else {
      numFails++;
      if (numFails > maxNumFails) break;
    }
  }
  (void)numCliquesFound;
  return false;
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
          {}, {}};
  RsHost host{&ctx,
              cbLog,
              cbTimerRead,
              cbTimeString,
              cbAnalysisSetup,
              cbSyncModel,
              cbSetMatrix,
              cbFlush,
              cbShrink,
              cbDependentEquations,
              cbHaveCommonClique,
              cbNumCliquesCol,
              cbNumCliques,
              cbSetPresolveFlag,
              cbSetMaxEntries,
              cbColumnTransformed,
              cbAddVb,
              cbProfiling,
              cbProbingPrepare,
              cbProbingLoop,
              cbFinaliseBegin,
              cbDomainBounds,
              cbImplSubstitutions,
              cbClearImplSubstitutions,
              cbCliqueSubstitutions,
              cbClearCliqueSubstitutions,
              cbComputeMaximalCliques,
              cbEnumerate,
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
              options->use_implied_bounds_from_presolve};
  RsMipInfo mi{};
  if (mipsolver != nullptr) {
    mi.epsilon = mipsolver->mipdata_->epsilon;
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
