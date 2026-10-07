/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#ifndef HIGHS_CUTPOOL_H_
#define HIGHS_CUTPOOL_H_

#include <atomic>
#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>

#include "lp_data/HConst.h"
#include "mip/HighsDomain.h"
#include "mip/HighsDynamicRowMatrix.h"
#include "mip/HighsRsSpan.h"

class HighsLpRelaxation;

struct HighsCutSet {
  std::vector<HighsInt> cutindices;
  std::vector<HighsInt> cutpools;
  std::vector<HighsInt> ARstart_;
  std::vector<HighsInt> ARindex_;
  std::vector<double> ARvalue_;
  std::vector<double> lower_;  // Currently only ever contains -kHighsInf
  std::vector<double> upper_;

  HighsInt numCuts() const { return cutindices.size(); }

  void resize(HighsInt nnz) {
    HighsInt ncuts = numCuts();
    lower_.resize(ncuts, -kHighsInf);
    upper_.resize(ncuts);
    ARstart_.resize(ncuts + 1);
    ARindex_.resize(nnz);
    ARvalue_.resize(nnz);
  }

  void clear() {
    cutindices.clear();
    cutpools.clear();
    upper_.clear();
    ARstart_.clear();
    ARindex_.clear();
    ARvalue_.clear();
  }

  bool empty() const { return cutindices.empty(); }
};

#ifdef HIGHS_RUST
namespace highs_rs {
struct CutPool;
struct CutPoolModelSize {
  HighsInt num_nonzero;
  HighsInt num_row;
};
extern "C" {
CutPool* highs_rs_cutpool_new(HighsInt ncols, HighsInt agelim,
                              HighsInt softlimit, HighsInt index,
                              void (*added)(void*, HighsInt, bool),
                              void (*deleted)(void*, HighsInt, bool));
void highs_rs_cutpool_free(CutPool* p);
void highs_rs_cutpool_view(const CutPool* p, CutPoolView* v);
HighsInt highs_rs_cutpool_geti(const CutPool* p, int which, HighsInt i);
double highs_rs_cutpool_getd(const CutPool* p, int which, HighsInt i);
void highs_rs_cutpool_op(CutPool* p, int which, HighsInt i, HighsInt j,
                         void* d);
double highs_rs_cutpool_parallelism(const CutPool* p, HighsInt row1,
                                    HighsInt row2, const CutPool* pool2);
HighsInt highs_rs_cutpool_add_cut(CutPool* p, const CutPool* global,
                                  const CutPoolModelSize* model,
                                  HighsInt* index, double* value, HighsInt len,
                                  double rhs, bool integral, bool propagate,
                                  bool isConflict);
const HighsInt* highs_rs_cutpool_separate(
    CutPool* p, const double* sol, HighsInt ncol, const double* colLower,
    const double* colUpper, const HighsInt* cutIndices,
    const HighsInt* cutPools, HighsInt ncuts, double feastol,
    const CutPool* const* pools, HighsInt npools, bool threadSafe,
    HighsInt* num);
void highs_rs_cutpool_free_buf(const HighsInt* buf, HighsInt num);
HighsInt* highs_rs_cutpool_cuts_to_sync(CutPool* p, HighsInt* num);
}
}  // namespace highs_rs

// The pool is Rust's (rust/src/mip/cutpool.rs); this class is a handle
class HighsCutPool {
  highs_rs::CutPool* rs_;

  void op(int which, HighsInt i = 0, HighsInt j = 0, void* d = nullptr) {
    highs_rs::highs_rs_cutpool_op(rs_, which, i, j, d);
  }
  highs_rs::CutPoolView view() const {
    highs_rs::CutPoolView v;
    highs_rs::highs_rs_cutpool_view(rs_, &v);
    return v;
  }

 public:
  HighsInt index_;

  HighsCutPool(HighsInt ncols, HighsInt agelim, HighsInt softlimit,
               HighsInt index);
  HighsCutPool(const HighsCutPool&) = delete;
  HighsCutPool& operator=(const HighsCutPool&) = delete;
  HighsCutPool(HighsCutPool&& other) : rs_(other.rs_), index_(other.index_) {
    other.rs_ = nullptr;
  }
  HighsCutPool& operator=(HighsCutPool&& other) {
    std::swap(rs_, other.rs_);
    std::swap(index_, other.index_);
    return *this;
  }
  ~HighsCutPool() { highs_rs::highs_rs_cutpool_free(rs_); }

  highs_rs::CutPool* rust() const { return rs_; }

  HighsDynamicRowMatrix getMatrix() const {
    return HighsDynamicRowMatrix(view());
  }

  HighsRsSpan<double> getRhs() const {
    highs_rs::CutPoolView v = view();
    return HighsRsSpan<double>{v.rhs, (size_t)v.num_rhs};
  }

  void resetAge(HighsInt cut, bool thread_safe = false) {
    op(0, cut, thread_safe);
  }

  double getParallelism(HighsInt row1, HighsInt row2) const {
    return highs_rs::highs_rs_cutpool_parallelism(rs_, row1, row2, rs_);
  }

  double getParallelism(HighsInt row1, HighsInt row2,
                        const HighsCutPool& pool2) const {
    return highs_rs::highs_rs_cutpool_parallelism(rs_, row1, row2, pool2.rs_);
  }

  void performAging() { op(4); }

  void lpCutRemoved(HighsInt cut, bool thread_safe = false) {
    op(1, cut, thread_safe);
  }

  void addPropagationDomain(HighsDomain::CutpoolPropagation* domain) {
    op(5, 0, 0, domain);
  }

  void removePropagationDomain(HighsDomain::CutpoolPropagation* domain) {
    op(6, 0, 0, domain);
  }

  void setAgeLimit(HighsInt agelim) { op(3, agelim); }

  void increaseNumLps(HighsInt cut, HighsInt n) { op(2, cut, n); }

  void separate(const std::vector<double>& sol, const HighsDomain& domprop,
                HighsCutSet& cutset, double feastol,
                const std::deque<HighsCutPool>& cutpools,
                bool thread_safe = false);

  void separateLpCutsAfterRestart(HighsCutSet& cutset);

  bool cutIsIntegral(HighsInt cut) const {
    return highs_rs::highs_rs_cutpool_geti(rs_, 3, cut) != 0;
  }

  HighsInt getNumCuts() const { return highs_rs::highs_rs_cutpool_geti(rs_, 0, 0); }

  HighsInt getNumAvailableCuts() const {
    return highs_rs::highs_rs_cutpool_geti(rs_, 1, 0);
  }

  double getMaxAbsCutCoef(HighsInt cut) const {
    return highs_rs::highs_rs_cutpool_getd(rs_, 0, cut);
  }

  double getRowNormalization(HighsInt cut) const {
    return highs_rs::highs_rs_cutpool_getd(rs_, 1, cut);
  }

  HighsInt addCut(const HighsMipSolver& mipsolver, HighsInt* Rindex,
                  double* Rvalue, HighsInt Rlen, double rhs,
                  bool integral = false, bool propagate = true,
                  bool extractCliques = true, bool isConflict = false);

  HighsInt getRowLength(HighsInt row) const {
    return highs_rs::highs_rs_cutpool_geti(rs_, 2, row);
  }

  void getCut(HighsInt cut, HighsInt& cutlen, const HighsInt*& cutinds,
              const double*& cutvals) const {
    highs_rs::CutPoolView v = view();
    HighsInt start = v.ar_range[cut].first;
    cutlen = v.ar_range[cut].second - start;
    cutinds = v.ar_index + start;
    cutvals = v.ar_value + start;
  }

  void syncCutPool(const HighsMipSolver& mipsolver, HighsCutPool& syncpool);
};
#else
class HighsCutPool {
 private:
  HighsDynamicRowMatrix matrix_;
  std::vector<double> rhs_;
  std::vector<int16_t> ages_;
  std::deque<std::atomic<int16_t>> numLps_;
  std::deque<std::atomic<uint8_t>>
      ageResetWhileLocked_;      // Was the cut propagated?
  std::vector<bool> hasSynced_;  // Has the cut been globally synced?
  std::vector<double> rownormalization_;
  std::vector<double> maxabscoef_;
  std::vector<uint8_t> rowintegral;
  std::unordered_multimap<uint64_t, HighsInt> hashToCutMap;
  std::vector<HighsDomain::CutpoolPropagation*> propagationDomains;
  std::set<std::pair<HighsInt, HighsInt>> propRows;

  double bestObservedScore;
  double minScoreFactor;
  double minDensityLim;

  HighsInt agelim_;
  HighsInt softlimit_;
  HighsInt numLpCuts;
  HighsInt numPropNzs;
  HighsInt numPropRows;
  std::vector<HighsInt> ageDistribution;
  std::vector<std::pair<HighsInt, double>> sortBuffer;

 public:
  HighsInt index_;

  HighsCutPool(HighsInt ncols, HighsInt agelim, HighsInt softlimit,
               HighsInt index)
      : matrix_(ncols),
        agelim_(agelim),
        softlimit_(softlimit),
        numLpCuts(0),
        numPropNzs(0),
        numPropRows(0),
        index_(index) {
    ageDistribution.resize(agelim_ + 1);
    minScoreFactor = 0.9;
    bestObservedScore = 0.0;
    minDensityLim = 0.1 * ncols;
  }
  const HighsDynamicRowMatrix& getMatrix() const { return matrix_; }

  const std::vector<double>& getRhs() const { return rhs_; }

  bool isDuplicate(size_t hash, double norm, const HighsInt* Rindex,
                   const double* Rvalue, HighsInt Rlen, double rhs);

  void resetAge(HighsInt cut, bool thread_safe = false) {
    if (ages_[cut] > 0) {
      if (thread_safe) {
        ageResetWhileLocked_[cut].store(1, std::memory_order_relaxed);
        return;
      }
      if (matrix_.columnsLinked(cut)) {
        propRows.erase(std::make_pair(ages_[cut], cut));
        propRows.emplace(0, cut);
      }
      ageDistribution[ages_[cut]] -= 1;
      ageDistribution[0] += 1;
      ages_[cut] = 0;
      ageResetWhileLocked_[cut].store(0, std::memory_order_relaxed);
    }
  }

  double getParallelism(HighsInt row1, HighsInt row2) const;

  double getParallelism(HighsInt row1, HighsInt row2,
                        const HighsCutPool& pool2) const;

  void performAging();

  void lpCutRemoved(HighsInt cut, bool thread_safe = false);

  void addPropagationDomain(HighsDomain::CutpoolPropagation* domain) {
    propagationDomains.push_back(domain);
  }

  void removePropagationDomain(HighsDomain::CutpoolPropagation* domain) {
    for (HighsInt k = propagationDomains.size() - 1; k >= 0; --k) {
      if (propagationDomains[k] == domain) {
        propagationDomains.erase(propagationDomains.begin() + k);
        return;
      }
    }
  }

  void setAgeLimit(HighsInt agelim) {
    agelim_ = agelim;
    ageDistribution.resize(agelim_ + 1);
  }

  void increaseNumLps(HighsInt cut, HighsInt n) {
    assert(numLps_[cut] >= 1);
    numLps_[cut].fetch_add(n, std::memory_order_relaxed);
  };

  void separate(const std::vector<double>& sol, const HighsDomain& domprop,
                HighsCutSet& cutset, double feastol,
                const std::deque<HighsCutPool>& cutpools,
                bool thread_safe = false);

  void separateLpCutsAfterRestart(HighsCutSet& cutset);

  bool cutIsIntegral(HighsInt cut) const { return (rowintegral[cut] != 0); }

  HighsInt getNumCuts() const {
    return matrix_.getNumRows() - matrix_.getNumDelRows();
  }

  HighsInt getNumAvailableCuts() const { return getNumCuts() - numLpCuts; }

  double getMaxAbsCutCoef(HighsInt cut) const { return maxabscoef_[cut]; }

  double getRowNormalization(HighsInt cut) const {
    return rownormalization_[cut];
  }

  HighsInt addCut(const HighsMipSolver& mipsolver, HighsInt* Rindex,
                  double* Rvalue, HighsInt Rlen, double rhs,
                  bool integral = false, bool propagate = true,
                  bool extractCliques = true, bool isConflict = false);

  HighsInt getRowLength(HighsInt row) const {
    return matrix_.getRowEnd(row) - matrix_.getRowStart(row);
  }

  void getCut(HighsInt cut, HighsInt& cutlen, const HighsInt*& cutinds,
              const double*& cutvals) const {
    HighsInt start = matrix_.getRowStart(cut);
    cutlen = matrix_.getRowEnd(cut) - start;
    cutinds = matrix_.getARindex() + start;
    cutvals = matrix_.getARvalue() + start;
  }

  void syncCutPool(const HighsMipSolver& mipsolver, HighsCutPool& syncpool);
};

#endif  // HIGHS_RUST

#endif
