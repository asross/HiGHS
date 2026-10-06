/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "mip/HighsConflictPool.h"

#include "mip/HighsDomain.h"

#ifdef HIGHS_RUST
namespace {
void conflictAdded(void* d, HighsInt conflict) {
  static_cast<HighsDomain::ConflictPoolPropagation*>(d)->conflictAdded(
      conflict);
}
void conflictDeleted(void* d, HighsInt conflict) {
  static_cast<HighsDomain::ConflictPoolPropagation*>(d)->conflictDeleted(
      conflict);
}
}  // namespace

HighsConflictPool::HighsConflictPool(HighsInt agelim, HighsInt softlimit)
    : rs_(highs_rs::highs_rs_conflictpool_new(agelim, softlimit, conflictAdded,
                                    conflictDeleted)) {}

// the entries of a conflict: continuous bounds relaxed by the tolerance
static std::vector<HighsDomainChange> relaxedEntries(
    const HighsDomain& domain, const HighsDomainChange* entries, HighsInt len) {
  std::vector<HighsDomainChange> relaxed(entries, entries + len);
  double feastol = domain.feastol();
  for (HighsDomainChange& e : relaxed) {
    if (domain.variableType(e.column) == HighsVarType::kContinuous) {
      if (e.boundtype == HighsBoundType::kLower)
        e.boundval += feastol;
      else
        e.boundval -= feastol;
    }
  }
  return relaxed;
}

void HighsConflictPool::addConflictCut(const HighsDomain& domain,
                                       const HighsDomainChange* entries,
                                       HighsInt len) {
  std::vector<HighsDomainChange> relaxed = relaxedEntries(domain, entries, len);
  highs_rs::highs_rs_conflictpool_add(rs_, relaxed.data(), len, nullptr);
}

void HighsConflictPool::addReconvergenceCut(
    const HighsDomain& domain, const HighsDomainChange* entries, HighsInt len,
    const HighsDomainChange& reconvergenceDomchg) {
  HighsDomainChange flipped = domain.flip(reconvergenceDomchg);
  std::vector<HighsDomainChange> relaxed = relaxedEntries(domain, entries, len);
  highs_rs::highs_rs_conflictpool_add(rs_, relaxed.data(), len, &flipped);
}

void HighsConflictPool::addConflictCut(
    const HighsDomain& domain,
    const std::set<HighsDomain::ConflictSet::LocalDomChg>& reasonSideFrontier) {
  std::vector<HighsDomainChange> entries;
  entries.reserve(reasonSideFrontier.size());
  for (const HighsDomain::ConflictSet::LocalDomChg& domchg :
       reasonSideFrontier)
    entries.push_back(domchg.domchg);
  addConflictCut(domain, entries.data(), entries.size());
}

void HighsConflictPool::addReconvergenceCut(
    const HighsDomain& domain,
    const std::set<HighsDomain::ConflictSet::LocalDomChg>&
        reconvergenceFrontier,
    const HighsDomainChange& reconvergenceDomchg) {
  std::vector<HighsDomainChange> entries;
  entries.reserve(reconvergenceFrontier.size());
  for (const HighsDomain::ConflictSet::LocalDomChg& domchg :
       reconvergenceFrontier)
    entries.push_back(domchg.domchg);
  addReconvergenceCut(domain, entries.data(), entries.size(),
                      reconvergenceDomchg);
}
#else
void HighsConflictPool::addConflictCut(
    const HighsDomain& domain,
    const std::set<HighsDomain::ConflictSet::LocalDomChg>& reasonSideFrontier) {
  std::vector<HighsDomainChange> entries;
  entries.reserve(reasonSideFrontier.size());
  for (const HighsDomain::ConflictSet::LocalDomChg& domchg :
       reasonSideFrontier) {
    assert(domchg.pos >= 0);
    assert(domchg.pos < (HighsInt)domain.getDomainChangeStack().size());
    entries.push_back(domchg.domchg);
  }
  addConflictCut(domain, entries.data(), entries.size());
}

void HighsConflictPool::addConflictCut(const HighsDomain& domain,
                                       const HighsDomainChange* entries,
                                       HighsInt conflictLen) {
  HighsInt conflictIndex;
  HighsInt start;
  HighsInt end;
  std::set<std::pair<HighsInt, HighsInt>>::iterator it;
  if (freeSpaces_.empty() ||
      (it = freeSpaces_.lower_bound(
           std::make_pair(conflictLen, HighsInt{-1}))) == freeSpaces_.end()) {
    start = conflictEntries_.size();
    end = start + conflictLen;

    conflictEntries_.resize(end);
  } else {
    std::pair<HighsInt, HighsInt> freeslot = *it;
    freeSpaces_.erase(it);

    start = freeslot.second;
    end = start + conflictLen;
    // if the space was not completely occupied, we register the remainder of
    // it again in the priority queue
    if (freeslot.first > conflictLen) {
      freeSpaces_.emplace(freeslot.first - conflictLen, end);
    }
  }

  // register the range of entries for this conflict with a reused or a new
  // index
  if (deletedConflicts_.empty()) {
    conflictIndex = conflictRanges_.size();
    conflictRanges_.emplace_back(start, end);
    ages_.resize(conflictRanges_.size());
    modification_.resize(conflictRanges_.size());
    ageResetWhileLocked_.resize(conflictRanges_.size());
  } else {
    conflictIndex = deletedConflicts_.back();
    deletedConflicts_.pop_back();
    conflictRanges_[conflictIndex].first = start;
    conflictRanges_[conflictIndex].second = end;
  }

  ageResetWhileLocked_[conflictIndex].store(0, std::memory_order_relaxed);
  modification_[conflictIndex] += 1;
  ages_[conflictIndex] = 0;
  ageDistribution_[ages_[conflictIndex]] += 1;

  HighsInt i = start;
  double feastol = domain.feastol();
  for (HighsInt k = 0; k < conflictLen; ++k) {
    assert(i < end);
    conflictEntries_[i] = entries[k];
    if (domain.variableType(conflictEntries_[i].column) ==
        HighsVarType::kContinuous) {
      if (conflictEntries_[i].boundtype == HighsBoundType::kLower)
        conflictEntries_[i].boundval += feastol;
      else
        conflictEntries_[i].boundval -= feastol;
    }
    ++i;
  }

  for (HighsDomain::ConflictPoolPropagation* conflictProp : propagationDomains)
    conflictProp->conflictAdded(conflictIndex);
}

void HighsConflictPool::addReconvergenceCut(
    const HighsDomain& domain,
    const std::set<HighsDomain::ConflictSet::LocalDomChg>&
        reconvergenceFrontier,
    const HighsDomainChange& reconvergenceDomchg) {
  std::vector<HighsDomainChange> entries;
  entries.reserve(reconvergenceFrontier.size());
  for (const HighsDomain::ConflictSet::LocalDomChg& domchg :
       reconvergenceFrontier) {
    assert(domchg.pos >= 0);
    assert(domchg.pos < (HighsInt)domain.getDomainChangeStack().size());
    entries.push_back(domchg.domchg);
  }
  addReconvergenceCut(domain, entries.data(), entries.size(),
                      reconvergenceDomchg);
}

void HighsConflictPool::addReconvergenceCut(
    const HighsDomain& domain, const HighsDomainChange* entries,
    HighsInt frontierLen, const HighsDomainChange& reconvergenceDomchg) {
  HighsInt conflictIndex;
  HighsInt start;
  HighsInt end;
  HighsInt conflictLen = frontierLen + 1;
  std::set<std::pair<HighsInt, HighsInt>>::iterator it;
  if (freeSpaces_.empty() ||
      (it = freeSpaces_.lower_bound(
           std::make_pair(conflictLen, HighsInt{-1}))) == freeSpaces_.end()) {
    start = conflictEntries_.size();
    end = start + conflictLen;

    conflictEntries_.resize(end);
  } else {
    std::pair<HighsInt, HighsInt> freeslot = *it;
    freeSpaces_.erase(it);

    start = freeslot.second;
    end = start + conflictLen;
    // if the space was not completely occupied, we register the remainder of
    // it again in the priority queue
    if (freeslot.first > conflictLen) {
      freeSpaces_.emplace(freeslot.first - conflictLen, end);
    }
  }

  // register the range of entries for this conflict with a reused or a new
  // index
  if (deletedConflicts_.empty()) {
    conflictIndex = conflictRanges_.size();
    conflictRanges_.emplace_back(start, end);
    ages_.resize(conflictRanges_.size());
    modification_.resize(conflictRanges_.size());
    ageResetWhileLocked_.resize(conflictRanges_.size());
  } else {
    conflictIndex = deletedConflicts_.back();
    deletedConflicts_.pop_back();
    conflictRanges_[conflictIndex].first = start;
    conflictRanges_[conflictIndex].second = end;
  }

  ageResetWhileLocked_[conflictIndex].store(0, std::memory_order_relaxed);
  modification_[conflictIndex] += 1;
  ages_[conflictIndex] = 0;
  ageDistribution_[ages_[conflictIndex]] += 1;

  HighsInt i = start;
  assert(i < end);
  conflictEntries_[i++] = domain.flip(reconvergenceDomchg);
  double feastol = domain.feastol();
  for (HighsInt k = 0; k < frontierLen; ++k) {
    assert(i < end);
    conflictEntries_[i] = entries[k];
    if (domain.variableType(conflictEntries_[i].column) ==
        HighsVarType::kContinuous) {
      if (conflictEntries_[i].boundtype == HighsBoundType::kLower)
        conflictEntries_[i].boundval += feastol;
      else
        conflictEntries_[i].boundval -= feastol;
    }
    ++i;
  }

  for (HighsDomain::ConflictPoolPropagation* conflictProp : propagationDomains)
    conflictProp->conflictAdded(conflictIndex);
}

void HighsConflictPool::removeConflict(HighsInt conflict) {
  for (HighsDomain::ConflictPoolPropagation* conflictProp : propagationDomains)
    conflictProp->conflictDeleted(conflict);

  if (ages_[conflict] >= 0) {
    ageDistribution_[ages_[conflict]] -= 1;
    ages_[conflict] = -1;
  }

  HighsInt start = conflictRanges_[conflict].first;
  HighsInt end = conflictRanges_[conflict].second;

  // register the space of the deleted row and the index so that it can be
  // reused
  deletedConflicts_.push_back(conflict);
  freeSpaces_.emplace(end - start, start);

  // set the range to -1,-1 to indicate a deleted row
  conflictRanges_[conflict].first = -1;
  conflictRanges_[conflict].second = -1;
  ++modification_[conflict];
}

void HighsConflictPool::performAging(const bool thread_safe) {
  if (age_lock_) return;
  HighsInt conflictMaxIndex = conflictRanges_.size();
  HighsInt agelim = agelim_;
  HighsInt numActiveConflicts = getNumConflicts();
  while (agelim > 5 && numActiveConflicts > softlimit_) {
    numActiveConflicts -= ageDistribution_[agelim];
    --agelim;
  }

  for (HighsInt i = 0; i != conflictMaxIndex; ++i) {
    if (ages_[i] < 0) continue;
    if (thread_safe &&
        ageResetWhileLocked_[i].load(std::memory_order_relaxed) == 1)
      resetAge(i);

    ageDistribution_[ages_[i]] -= 1;
    ages_[i] += 1;
    ageResetWhileLocked_[i].store(0, std::memory_order_relaxed);

    if (ages_[i] > agelim) {
      ages_[i] = -1;
      removeConflict(i);
    } else
      ageDistribution_[ages_[i]] += 1;
  }
}

void HighsConflictPool::addConflictFromOtherPool(
    const HighsDomainChange* conflictEntries, const HighsInt conflictLen) {
  HighsInt conflictIndex;
  HighsInt start;
  HighsInt end;
  std::set<std::pair<HighsInt, HighsInt>>::iterator it;
  if (freeSpaces_.empty() ||
      (it = freeSpaces_.lower_bound(
           std::make_pair(conflictLen, HighsInt{-1}))) == freeSpaces_.end()) {
    start = conflictEntries_.size();
    end = start + conflictLen;

    conflictEntries_.resize(end);
  } else {
    std::pair<HighsInt, HighsInt> freeslot = *it;
    freeSpaces_.erase(it);

    start = freeslot.second;
    end = start + conflictLen;
    // if the space was not completely occupied, we register the remainder of
    // it again in the priority queue
    if (freeslot.first > conflictLen) {
      freeSpaces_.emplace(freeslot.first - conflictLen, end);
    }
  }

  // register the range of entries for this conflict with a reused or a new
  // index
  if (deletedConflicts_.empty()) {
    conflictIndex = conflictRanges_.size();
    conflictRanges_.emplace_back(start, end);
    ages_.resize(conflictRanges_.size());
    modification_.resize(conflictRanges_.size());
    ageResetWhileLocked_.resize(conflictRanges_.size());
  } else {
    conflictIndex = deletedConflicts_.back();
    deletedConflicts_.pop_back();
    conflictRanges_[conflictIndex].first = start;
    conflictRanges_[conflictIndex].second = end;
  }

  ageResetWhileLocked_[conflictIndex].store(0, std::memory_order_relaxed);
  modification_[conflictIndex] += 1;
  ages_[conflictIndex] = 0;
  ageDistribution_[ages_[conflictIndex]] += 1;

  for (HighsInt i = 0; i != conflictLen; ++i) {
    assert(start + i < end);
    conflictEntries_[start + i] = conflictEntries[i];
  }

  for (HighsDomain::ConflictPoolPropagation* conflictProp : propagationDomains)
    conflictProp->conflictAdded(conflictIndex);
}

void HighsConflictPool::syncConflictPool(HighsConflictPool& syncpool) {
  HighsInt conflictMaxIndex = conflictRanges_.size();
  for (HighsInt i = 0; i != conflictMaxIndex; ++i) {
    if (ages_[i] < 0) continue;
    HighsInt start = conflictRanges_[i].first;
    HighsInt end = conflictRanges_[i].second;
    assert(start >= 0 && end >= 0);
    syncpool.addConflictFromOtherPool(&conflictEntries_[start], end - start);
    removeConflict(i);
  }
  deletedConflicts_.clear();
  freeSpaces_.clear();
  conflictRanges_.clear();
  conflictEntries_.clear();
  modification_.clear();
  ages_.clear();
  ageResetWhileLocked_.clear();
}
#endif  // HIGHS_RUST
