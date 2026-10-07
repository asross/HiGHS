/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#ifndef HIGHS_CONFLICTPOOL_H_
#define HIGHS_CONFLICTPOOL_H_

#include <atomic>
#include <set>
#include <vector>

#include "HConfig.h"
#include "mip/HighsDomain.h"
#include "mip/HighsRsSpan.h"
#include "util/HighsInt.h"

#ifdef HIGHS_RUST
namespace highs_rs {
struct ConflictPool;
extern "C" {
ConflictPool* highs_rs_conflictpool_new(HighsInt agelim, HighsInt softlimit,
                                        void (*added)(void*, HighsInt),
                                        void (*deleted)(void*, HighsInt));
void highs_rs_conflictpool_free(ConflictPool* p);
const void* highs_rs_conflictpool_data(const ConflictPool* p, int which,
                                       size_t* len);
unsigned highs_rs_conflictpool_get(const ConflictPool* p, int which,
                                   HighsInt i);
void highs_rs_conflictpool_op(ConflictPool* p, int which, HighsInt i,
                              void* d);
void highs_rs_conflictpool_add(ConflictPool* p,
                               const HighsDomainChange* entries, HighsInt len,
                               const HighsDomainChange* flipped);
void highs_rs_conflictpool_add_from_other(ConflictPool* p,
                                          const HighsDomainChange* entries,
                                          HighsInt len);
void highs_rs_conflictpool_sync(ConflictPool* p, ConflictPool* sync);
}
}  // namespace highs_rs

// The pool is Rust's (rust/src/mip/conflictpool.rs); this class is a handle
class HighsConflictPool {
  highs_rs::ConflictPool* rs_;

  void op(int which, HighsInt i = 0, void* d = nullptr) {
    highs_rs::highs_rs_conflictpool_op(rs_, which, i, d);
  }
  template <typename T>
  HighsRsSpan<T> data(int which) const {
    size_t n;
    const void* p = highs_rs::highs_rs_conflictpool_data(rs_, which, &n);
    return HighsRsSpan<T>{static_cast<const T*>(p), n};
  }

 public:
  HighsConflictPool(HighsInt agelim, HighsInt softlimit);
  HighsConflictPool(const HighsConflictPool&) = delete;
  HighsConflictPool& operator=(const HighsConflictPool&) = delete;
  HighsConflictPool(HighsConflictPool&& other) : rs_(other.rs_) {
    other.rs_ = nullptr;
  }
  HighsConflictPool& operator=(HighsConflictPool&& other) {
    std::swap(rs_, other.rs_);
    return *this;
  }
  ~HighsConflictPool() { highs_rs::highs_rs_conflictpool_free(rs_); }

  highs_rs::ConflictPool* rust() const { return rs_; }

  void addConflictCut(const HighsDomain& domain,
                      const std::set<HighsDomain::ConflictSet::LocalDomChg>&
                          reasonSideFrontier);

  void addReconvergenceCut(
      const HighsDomain& domain,
      const std::set<HighsDomain::ConflictSet::LocalDomChg>&
          reconvergenceFrontier,
      const HighsDomainChange& reconvergenceDomchg);

  void addConflictCut(const HighsDomain& domain,
                      const HighsDomainChange* entries, HighsInt len);

  void addReconvergenceCut(const HighsDomain& domain,
                           const HighsDomainChange* entries, HighsInt len,
                           const HighsDomainChange& reconvergenceDomchg);

  void removeConflict(HighsInt conflict) { op(4, conflict); }

  void performAging(bool thread_safe = false) { op(3, thread_safe); }

  void addConflictFromOtherPool(const HighsDomainChange* conflictEntries,
                                HighsInt conflictLen) {
    highs_rs::highs_rs_conflictpool_add_from_other(rs_, conflictEntries, conflictLen);
  }

  void syncConflictPool(HighsConflictPool& syncpool) {
    highs_rs::highs_rs_conflictpool_sync(rs_, syncpool.rs_);
  }

  void resetAge(HighsInt conflict) { op(0, conflict); }

  void setAgeLimit(HighsInt agelim) { op(1, agelim); }

  unsigned getModificationCount(HighsInt cut) const {
    return highs_rs::highs_rs_conflictpool_get(rs_, 1, cut);
  }

  void addPropagationDomain(HighsDomain::ConflictPoolPropagation* domain) {
    op(5, 0, domain);
  }

  void removePropagationDomain(HighsDomain::ConflictPoolPropagation* domain) {
    op(6, 0, domain);
  }

  HighsRsSpan<HighsDomainChange> getConflictEntryVector() const {
    return data<HighsDomainChange>(0);
  }

  HighsRsSpan<std::pair<HighsInt, HighsInt>> getConflictRanges() const {
    return data<std::pair<HighsInt, HighsInt>>(1);
  }

  HighsInt getNumConflicts() const {
    return highs_rs::highs_rs_conflictpool_get(rs_, 0, 0);
  }

  void setAgeLock(const bool ageLock) { op(2, ageLock); }
};
#else
class HighsConflictPool {
 private:
  HighsInt agelim_;
  HighsInt softlimit_;
  bool age_lock_;
  std::vector<HighsInt> ageDistribution_;
  std::vector<int16_t> ages_;
  std::vector<unsigned> modification_;
  std::deque<std::atomic<uint8_t>> ageResetWhileLocked_;

  std::vector<HighsDomainChange> conflictEntries_;
  std::vector<std::pair<HighsInt, HighsInt>> conflictRanges_;

  /// keep an ordered set of free spaces in the row arrays so that they can be
  /// reused efficiently
  std::set<std::pair<HighsInt, HighsInt>> freeSpaces_;

  /// vector of deleted conflicts so that their indices can be reused
  std::vector<HighsInt> deletedConflicts_;

  std::vector<HighsDomain::ConflictPoolPropagation*> propagationDomains;

 public:
  HighsConflictPool(HighsInt agelim, HighsInt softlimit)
      : agelim_(agelim),
        softlimit_(softlimit),
        age_lock_(false),
        ageDistribution_(),
        ages_(),
        modification_(),
        ageResetWhileLocked_(),
        conflictEntries_(),
        conflictRanges_(),
        freeSpaces_(),
        deletedConflicts_(),
        propagationDomains() {
    ageDistribution_.resize(agelim_ + 1);
  }

  void addConflictCut(const HighsDomain& domain,
                      const std::set<HighsDomain::ConflictSet::LocalDomChg>&
                          reasonSideFrontier);

  void addReconvergenceCut(
      const HighsDomain& domain,
      const std::set<HighsDomain::ConflictSet::LocalDomChg>&
          reconvergenceFrontier,
      const HighsDomainChange& reconvergenceDomchg);

  // the same with the frontier's domain changes in the order of the stack
  void addConflictCut(const HighsDomain& domain,
                      const HighsDomainChange* entries, HighsInt len);

  void addReconvergenceCut(const HighsDomain& domain,
                           const HighsDomainChange* entries, HighsInt len,
                           const HighsDomainChange& reconvergenceDomchg);

  void removeConflict(HighsInt conflict);

  void performAging(bool thread_safe = false);

  void addConflictFromOtherPool(const HighsDomainChange* conflictEntries,
                                HighsInt conflictLen);

  void syncConflictPool(HighsConflictPool& syncpool);

  void resetAge(HighsInt conflict) {
    if (ages_[conflict] > 0) {
      if (age_lock_) {
        ageResetWhileLocked_[conflict].store(1, std::memory_order_relaxed);
        return;
      }
      ageDistribution_[ages_[conflict]] -= 1;
      ageDistribution_[0] += 1;
      ages_[conflict] = 0;
    }
  }

  void setAgeLimit(HighsInt agelim) {
    agelim_ = agelim;
    ageDistribution_.resize(agelim_ + 1);
  }

  unsigned getModificationCount(HighsInt cut) const {
    return modification_[cut];
  }

  void addPropagationDomain(HighsDomain::ConflictPoolPropagation* domain) {
    propagationDomains.push_back(domain);
  }

  void removePropagationDomain(HighsDomain::ConflictPoolPropagation* domain) {
    for (HighsInt k = propagationDomains.size() - 1; k >= 0; --k) {
      if (propagationDomains[k] == domain) {
        propagationDomains.erase(propagationDomains.begin() + k);
        return;
      }
    }
  }

  const std::vector<HighsDomainChange>& getConflictEntryVector() const {
    return conflictEntries_;
  }

  const std::vector<std::pair<HighsInt, HighsInt>>& getConflictRanges() const {
    return conflictRanges_;
  }

  HighsInt getNumConflicts() const {
    return conflictRanges_.size() - deletedConflicts_.size();
  }

  void setAgeLock(const bool ageLock) { age_lock_ = ageLock; }
};

#endif  // HIGHS_RUST

#endif
