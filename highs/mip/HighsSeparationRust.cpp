/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file mip/HighsSeparationRust.cpp
 * @brief The cut separation of the Rust port (rust/src/mip/cuts):
 * HighsTransformedLp holds the Rust round (bound substitution data, LP rows,
 * aggregation); the path, tableau and mod-k separators and the conflict
 * generation of HighsCutGeneration run in Rust, calling back here for the
 * implications, the slack bounds, the LP rows and the cut pool.
 */
#include "mip/HighsTransformedLp.h"

#ifdef HIGHS_RUST

#include "mip/HighsCutGeneration.h"
#include "mip/HighsCutPool.h"
#include "mip/HighsDomain.h"
#include "mip/HighsLpRelaxation.h"
#include "mip/HighsMipSolverData.h"
#include "mip/HighsModkSeparator.h"
#include "mip/HighsPathSeparator.h"
#include "mip/HighsTableauSeparator.h"
#include "util/HVector.h"

static_assert(sizeof(HighsInt) == 4, "the Rust port uses 32-bit HighsInt");
static_assert(sizeof(HighsVarType) == 1, "integrality is read as bytes");

namespace highs_rs {
struct SepaVarBound {
  double coef;
  double constant;
};

// rust/src/mip/cuts/round.rs: Host
struct SepaHost {
  void* ctx;
  void (*cleanup_varbounds)(void*, HighsInt);
  bool (*dom_infeasible)(void*);
  HighsInt (*best_vub)(void*, HighsInt, double*, SepaVarBound*);
  HighsInt (*best_vlb)(void*, HighsInt, double*, SepaVarBound*);
  double (*slack_lower)(void*, HighsInt);
  double (*slack_upper)(void*, HighsInt);
  void (*get_row)(void*, HighsInt, HighsInt*, const HighsInt**, const double**,
                  uint8_t*, double*);
  HighsInt (*add_cut)(void*, HighsInt*, double*, HighsInt, double, bool,
                      bool);
  HighsInt (*num_cuts)(void*);
  HighsInt (*num_available_cuts)(void*);
  int64_t (*num_nodes_down)(void*, HighsInt);
  int64_t (*num_nodes_up)(void*, HighsInt);
  int64_t (*num_lp_iterations)(void*);
};

// rust/src/mip/cuts/round.rs: CSepaLp
struct CSepaLp {
  HighsInt num_col;
  HighsInt num_row;
  const double* col_lower;
  const double* col_upper;
  const double* col_value;
  const double* row_value;
  const double* row_dual;
  const double* row_lower;
  const double* row_upper;
  const HighsInt* a_start;
  const HighsInt* a_index;
  const double* a_value;
  const uint8_t* integrality;
  const HighsInt* continuous_cols;
  HighsInt num_continuous_cols;
  const HighsInt* integral_cols;
  HighsInt num_integral_cols;
  double feastol;
  double epsilon;
  double small_matrix_value;
  bool parallel_lock_active;
  HighsInt mip_pool_soft_limit;
  SepaHost host;
  LpHandle* lph;
};

// rust/src/mip/cuts/ffi.rs: CConflict
struct CConflict {
  HighsInt num_col;
  const double* glb;
  const double* gub;
  const double* llb;
  const double* lub;
  const uint8_t* integrality;
  HighsInt num_lp_cols;
  double feastol;
  double epsilon;
  uint32_t seed;
  SepaHost host;
};

struct SepaRound;
struct PathSeparator;
struct TableauSeparator;

extern "C" {
SepaRound* highs_rs_sepa_round_new(const CSepaLp* c);
void highs_rs_sepa_round_free(SepaRound* r);
PathSeparator* highs_rs_path_new(uint32_t seed);
void highs_rs_path_free(PathSeparator* p);
void highs_rs_path_separate(PathSeparator* p, SepaRound* r,
                            uint32_t cutgen_seed);
TableauSeparator* highs_rs_tableau_new();
void highs_rs_tableau_free(TableauSeparator* t);
void highs_rs_tableau_separate(TableauSeparator* t, SepaRound* r,
                               uint32_t cutgen_seed, HighsInt num_calls,
                               const HighsInt* basisinds,
                               int64_t lp_iterations);
void highs_rs_modk_separate(SepaRound* r, uint32_t cutgen_seed);
bool highs_rs_generate_conflict(const CConflict* c, const HighsInt* inds,
                                const double* vals, HighsInt len, double rhs);
}
}  // namespace highs_rs

/// The C++ state of a round that the Rust callbacks reach
struct HighsSepaRoundCtx {
  const HighsLpRelaxation* lp;
  const HighsDomain* globaldom;
  HighsImplications* implications;
  const HighsMipSolver* mipsolver;
  HighsCutPool* cutpool = nullptr;
  const HighsDomain* localdom = nullptr;
};

namespace {
HighsSepaRoundCtx& sepaCtxOf(void* p) { return *static_cast<HighsSepaRoundCtx*>(p); }

void sepaCbCleanupVarbounds(void* p, HighsInt col) {
  sepaCtxOf(p).mipsolver->mipdata_->implications.cleanupVarbounds(col);
}

bool sepaCbDomInfeasible(void* p) { return sepaCtxOf(p).globaldom->infeasible(); }

HighsInt sepaCbBestVub(void* p, HighsInt col, double* bound,
                   highs_rs::SepaVarBound* vb) {
  HighsSepaRoundCtx& c = sepaCtxOf(p);
  const highs_rs::LphView v = c.lp->lpView();
  auto best = c.implications->getBestVb(false, col, v.col_value, v.col_dual,
                                        v.n_col_value, *bound, *c.globaldom);
  vb->coef = best.second.coef;
  vb->constant = best.second.constant;
  return best.first;
}

HighsInt sepaCbBestVlb(void* p, HighsInt col, double* bound,
                   highs_rs::SepaVarBound* vb) {
  HighsSepaRoundCtx& c = sepaCtxOf(p);
  const highs_rs::LphView v = c.lp->lpView();
  auto best = c.implications->getBestVb(true, col, v.col_value, v.col_dual,
                                        v.n_col_value, *bound, *c.globaldom);
  vb->coef = best.second.coef;
  vb->constant = best.second.constant;
  return best.first;
}

double sepaCbSlackLower(void* p, HighsInt row) {
  return sepaCtxOf(p).lp->slackLower(row, *sepaCtxOf(p).globaldom);
}

double sepaCbSlackUpper(void* p, HighsInt row) {
  return sepaCtxOf(p).lp->slackUpper(row, *sepaCtxOf(p).globaldom);
}

void sepaCbGetRow(void* p, HighsInt row, HighsInt* len, const HighsInt** inds,
              const double** vals, uint8_t* integral, double* maxabs) {
  const HighsLpRelaxation& lp = *sepaCtxOf(p).lp;
  lp.getRow(row, *len, *inds, *vals);
  *integral = lp.isRowIntegral(row);
  *maxabs = lp.getMaxAbsRowVal(row);
}

HighsInt sepaCbAddCut(void* p, HighsInt* inds, double* vals, HighsInt len,
                  double rhs, bool integral, bool isConflict) {
  HighsSepaRoundCtx& c = sepaCtxOf(p);
  return c.cutpool->addCut(*c.mipsolver, inds, vals, len, rhs, integral, true,
                           true, isConflict);
}

HighsInt sepaCbNumCuts(void* p) { return sepaCtxOf(p).cutpool->getNumCuts(); }

HighsInt sepaCbNumAvailableCuts(void* p) {
  return sepaCtxOf(p).cutpool->getNumAvailableCuts();
}

int64_t sepaCbNumNodesDown(void* p, HighsInt col) {
  return sepaCtxOf(p).mipsolver->mipdata_->nodequeue.numNodesDown(col);
}

int64_t sepaCbNumNodesUp(void* p, HighsInt col) {
  return sepaCtxOf(p).mipsolver->mipdata_->nodequeue.numNodesUp(col);
}

int64_t sepaCbNumLpIterations(void* p) {
  return sepaCtxOf(p).lp->getNumLpIterations();
}

highs_rs::SepaHost sepaMakeHost(HighsSepaRoundCtx* ctx) {
  return highs_rs::SepaHost{ctx,
                        sepaCbCleanupVarbounds,
                        sepaCbDomInfeasible,
                        sepaCbBestVub,
                        sepaCbBestVlb,
                        sepaCbSlackLower,
                        sepaCbSlackUpper,
                        sepaCbGetRow,
                        sepaCbAddCut,
                        sepaCbNumCuts,
                        sepaCbNumAvailableCuts,
                        sepaCbNumNodesDown,
                        sepaCbNumNodesUp,
                        sepaCbNumLpIterations};
}

/// HighsCutGeneration's seed: random_seed + LP iterations + cuts in the pool
uint32_t sepaCutGenSeed(const HighsLpRelaxation& lp, const HighsCutPool& cutpool) {
  return uint32_t(lp.getMipSolver().options_mip_->random_seed +
                  lp.getNumLpIterations() + cutpool.getNumCuts());
}
}  // namespace

HighsTransformedLp::HighsTransformedLp(const HighsLpRelaxation& lprelaxation,
                                       HighsImplications& implications,
                                       const HighsDomain& globaldom)
    : globaldom_(globaldom), ctx_(new HighsSepaRoundCtx) {
  assert(lprelaxation.scaledOptimal(lprelaxation.getStatus()));
  const HighsMipSolver& mipsolver = implications.mipsolver;
  const HighsMipSolverData& mipdata = *mipsolver.mipdata_;
  const highs_rs::LphView lp = lprelaxation.lpView();
  ctx_->lp = &lprelaxation;
  ctx_->globaldom = &globaldom;
  ctx_->implications = &implications;
  ctx_->mipsolver = &mipsolver;

  highs_rs::CSepaLp c;
  c.num_col = lp.num_col;
  c.num_row = lp.num_row;
  c.col_lower = globaldom.col_lower_.data();
  c.col_upper = globaldom.col_upper_.data();
  c.col_value = lp.col_value;
  c.row_value = lp.row_value;
  c.row_dual = lp.row_dual;
  c.row_lower = lp.row_lower;
  c.row_upper = lp.row_upper;
  c.a_start = lp.a_start;
  c.a_index = lp.a_index;
  c.a_value = lp.a_value;
  c.integrality =
      reinterpret_cast<const uint8_t*>(mipsolver.model_->integrality_.data());
  c.continuous_cols = mipdata.continuous_cols.data();
  c.num_continuous_cols = mipdata.continuous_cols.size();
  c.integral_cols = mipdata.integral_cols.data();
  c.num_integral_cols = mipdata.integral_cols.size();
  c.feastol = mipdata.feastol;
  c.epsilon = mipdata.epsilon;
  c.small_matrix_value = mipsolver.options_mip_->small_matrix_value;
  c.parallel_lock_active = mipdata.parallelLockActive();
  c.mip_pool_soft_limit = mipsolver.options_mip_->mip_pool_soft_limit;
  c.host = sepaMakeHost(ctx_.get());
  c.lph = lprelaxation.lpHandle();
  rs_ = highs_rs::highs_rs_sepa_round_new(&c);
}

HighsTransformedLp::~HighsTransformedLp() {
  highs_rs::highs_rs_sepa_round_free(rs_);
}

highs_rs::SepaRound* HighsTransformedLp::rust(HighsCutPool& cutpool) {
  ctx_->cutpool = &cutpool;
  return rs_;
}

HighsPathSeparator::HighsPathSeparator(const HighsMipSolver& mipsolver)
    : HighsSeparator(mipsolver, kPathAggrSepaString),
      rs_(highs_rs::highs_rs_path_new(mipsolver.options_mip_->random_seed)) {}

HighsPathSeparator::~HighsPathSeparator() { highs_rs::highs_rs_path_free(rs_); }

void HighsPathSeparator::separateLpSolution(HighsLpRelaxation& lpRelaxation,
                                            HighsLpAggregator&,
                                            HighsTransformedLp& transLp,
                                            HighsCutPool& cutpool) {
  highs_rs::highs_rs_path_separate(rs_, transLp.rust(cutpool),
                                   sepaCutGenSeed(lpRelaxation, cutpool));
}

HighsTableauSeparator::HighsTableauSeparator(const HighsMipSolver& mipsolver)
    : HighsSeparator(mipsolver, kTableauSepaString),
      rs_(highs_rs::highs_rs_tableau_new()) {}

HighsTableauSeparator::~HighsTableauSeparator() {
  highs_rs::highs_rs_tableau_free(rs_);
}

void HighsTableauSeparator::separateLpSolution(HighsLpRelaxation& lpRelaxation,
                                               HighsLpAggregator&,
                                               HighsTransformedLp& transLp,
                                               HighsCutPool& cutpool) {
  if (!lpRelaxation.lpHasInvert()) return;
  const HighsMipSolverData& mipdata = *lpRelaxation.getMipSolver().mipdata_;
  highs_rs::SepaRound* r = transLp.rust(cutpool);
  highs_rs::highs_rs_tableau_separate(
      rs_, r, sepaCutGenSeed(lpRelaxation, cutpool), getNumCalls(),
      lpRelaxation.lpBasicIndex(),
      mipdata.total_lp_iterations - mipdata.heuristic_lp_iterations);
}

void HighsModkSeparator::separateLpSolution(HighsLpRelaxation& lpRelaxation,
                                            HighsLpAggregator&,
                                            HighsTransformedLp& transLp,
                                            HighsCutPool& cutpool) {
  highs_rs::highs_rs_modk_separate(transLp.rust(cutpool),
                                   sepaCutGenSeed(lpRelaxation, cutpool));
}

bool HighsCutGeneration::generateConflict(const HighsDomain& localdom,
                                          const HighsDomain& globaldom,
                                          std::vector<HighsInt>& proofinds,
                                          std::vector<double>& proofvals,
                                          double& proofrhs) {
  const HighsMipSolver& mipsolver = lpRelaxation.getMipSolver();
  HighsSepaRoundCtx ctx;
  ctx.lp = &lpRelaxation;
  ctx.globaldom = &globaldom;
  ctx.mipsolver = &mipsolver;
  ctx.cutpool = &cutpool;
  highs_rs::CConflict c;
  c.num_col = mipsolver.numCol();
  c.glb = globaldom.col_lower_.data();
  c.gub = globaldom.col_upper_.data();
  c.llb = localdom.col_lower_.data();
  c.lub = localdom.col_upper_.data();
  c.integrality =
      reinterpret_cast<const uint8_t*>(mipsolver.model_->integrality_.data());
  c.num_lp_cols = lpRelaxation.numCols();
  c.feastol = mipsolver.mipdata_->feastol;
  c.epsilon = mipsolver.mipdata_->epsilon;
  c.seed = sepaCutGenSeed(lpRelaxation, cutpool);
  c.host = sepaMakeHost(&ctx);
  return highs_rs::highs_rs_generate_conflict(
      &c, proofinds.data(), proofvals.data(), proofinds.size(), proofrhs);
}

#endif  // HIGHS_RUST
