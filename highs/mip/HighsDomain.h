/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#ifndef HIGHS_DOMAIN_H_
#define HIGHS_DOMAIN_H_

#include <cstdint>
#include <deque>
#include <memory>
#include <set>
#include <vector>

#include "HConfig.h"
#include "HighsPseudocost.h"
#include "mip/HighsDomainChange.h"
#include "mip/HighsMipSolver.h"
#include "util/HighsCDouble.h"
#include "util/HighsRbTree.h"

class HighsCutPool;
#ifdef HIGHS_RUST
#include "mip/HighsDomainRustView.h"
#include "mip/HighsRsSpan.h"
namespace highs_rs {
struct DomainVecs;
struct CutPropState;
struct ConfPropState;
struct ObjPropState;
struct DomainAccess;
struct CliqueAccess;
struct SymmetryAccess;
}
#endif
class HighsConflictPool;
class HighsObjectiveFunction;

class HighsDomain {
#ifdef HIGHS_RUST
  friend struct highs_rs::DomainAccess;
  friend struct highs_rs::ObjPropState;
  friend struct highs_rs::CliqueAccess;
  friend struct highs_rs::SymmetryAccess;
#endif

 public:
  struct Reason {
    HighsInt type;
    HighsInt index;

    enum {
      kBranching = -1,
      kUnknown = -2,
      kModelRowUpper = -3,
      kModelRowLower = -4,
      kCliqueTable = -5,
      kConflictingBounds = -6,
      kObjective = -7,
    };
    static Reason branching() { return Reason{kBranching, 0}; }
    static Reason unspecified() { return Reason{kUnknown, 0}; }
    static Reason cliqueTable(HighsInt col, HighsInt val) {
      return Reason{kCliqueTable, 2 * col + val};
    }
    static Reason modelRowUpper(HighsInt row) {
      return Reason{kModelRowUpper, row};
    }
    static Reason modelRowLower(HighsInt row) {
      return Reason{kModelRowLower, row};
    }
    static Reason cut(HighsInt cutpool, HighsInt cut) {
      return Reason{cutpool, cut};
    }
    static Reason conflictingBounds(HighsInt pos) {
      return Reason{kConflictingBounds, pos};
    }
    static Reason objective() { return Reason{kObjective, 0}; }
  };

  class ConflictSet {
    friend class HighsDomain;
    HighsDomain& localdom;
    const HighsDomain& globaldom;

   public:
    struct LocalDomChg {
      HighsInt pos;
      mutable HighsDomainChange domchg;

      bool operator<(const LocalDomChg& other) const { return pos < other.pos; }
    };

    ConflictSet(HighsDomain& localdom, const HighsDomain& globaldom);

    void conflictAnalysis(HighsConflictPool& conflictPool,
                          HighsPseudocost& pseudocost);
    void conflictAnalysis(const HighsInt* proofinds, const double* proofvals,
                          HighsInt prooflen, double proofrhs,
                          HighsConflictPool& conflictPool,
                          HighsPseudocost& pseudocost);

   private:
    std::set<LocalDomChg> reasonSideFrontier;
    std::set<LocalDomChg> reconvergenceFrontier;
    std::vector<std::set<LocalDomChg>::iterator> resolveQueue;
    std::vector<LocalDomChg> resolvedDomainChanges;

    struct ResolveCandidate {
      double delta;
      double baseBound;
      double prio;
      HighsInt boundPos;
      HighsInt valuePos;

      bool operator<(const ResolveCandidate& other) const {
        if (prio > other.prio) return true;
        if (other.prio > prio) return false;

        return boundPos < other.boundPos;
      }
    };

    std::vector<ResolveCandidate> resolveBuffer;

    void pushQueue(std::set<LocalDomChg>::iterator domchgPos);
    std::set<LocalDomChg>::iterator popQueue();
    void clearQueue();
    HighsInt queueSize() const;
    bool resolvable(HighsInt domChgPos) const;

    HighsInt resolveDepth(std::set<LocalDomChg>& frontier, HighsInt depthLevel,
                          HighsInt stopSize, HighsPseudocost& pseudocost,
                          HighsInt minResolve = 0,
                          bool increaseConflictScore = false);

    HighsInt computeCuts(HighsInt depthLevel, HighsConflictPool& conflictPool,
                         HighsPseudocost& pseudocost);

    bool explainInfeasibility();

    bool explainInfeasibilityConflict(const HighsDomainChange* conflict,
                                      HighsInt len);

    bool explainInfeasibilityLeq(const HighsInt* inds, const double* vals,
                                 HighsInt len, double rhs, double minActivity);

    bool explainInfeasibilityGeq(const HighsInt* inds, const double* vals,
                                 HighsInt len, double rhs, double maxActivity);

    bool explainBoundChange(const std::set<LocalDomChg>& currentFrontier,
                            LocalDomChg domchg);

    // bool explainBoundChange(HighsInt pos) {
    //   return explainBoundChange(LocalDomChg{pos,
    //   localdom.domchgstack_[pos]});
    // }

    bool explainBoundChangeConflict(const LocalDomChg& domchg,
                                    const HighsDomainChange* conflict,
                                    HighsInt len);

    bool explainBoundChangeLeq(const std::set<LocalDomChg>& currentFrontier,
                               const LocalDomChg& domChg, const HighsInt* inds,
                               const double* vals, HighsInt len, double rhs,
                               double minActivity);

    bool explainBoundChangeGeq(const std::set<LocalDomChg>& currentFrontier,
                               const LocalDomChg& domChg, const HighsInt* inds,
                               const double* vals, HighsInt len, double rhs,
                               double maxActivity);

    bool resolveLinearLeq(HighsCDouble M, double Mlower, const double* vals);

    bool resolveLinearGeq(HighsCDouble M, double Mupper, const double* vals);
  };

  struct CutpoolPropagation {
    HighsInt cutpoolindex;
    HighsDomain* domain;
    HighsCutPool* cutpool;
#ifdef HIGHS_RUST
    // Rust's (domain.rs CutPropState), owned, referred to in place
    highs_rs::CutPropState* rs_;
    HighsRsArray<HighsCDouble>& activitycuts_;
    HighsRsArray<HighsInt>& activitycutsinf_;
    HighsRsArray<uint8_t>& propagatecutflags_;
    HighsRsArray<HighsInt>& propagatecutinds_;
    HighsRsArray<double>& capacityThreshold_;
#else
    std::vector<HighsCDouble> activitycuts_;
    std::vector<HighsInt> activitycutsinf_;
    std::vector<uint8_t> propagatecutflags_;
    std::vector<HighsInt> propagatecutinds_;
    std::vector<double> capacityThreshold_;
#endif

    CutpoolPropagation(HighsInt cutpoolindex, HighsDomain* domain,
                       HighsCutPool& cutpool);

    CutpoolPropagation(const CutpoolPropagation& other);

    CutpoolPropagation& operator=(const CutpoolPropagation& other);

    ~CutpoolPropagation();

    void recomputeCapacityThreshold(HighsInt cut);

    void cutAdded(HighsInt cut, bool propagate);

    void cutDeleted(HighsInt cut, bool deletedOnlyForPropagation = false);

    void markPropagateCut(HighsInt cut);

    void updateActivityLbChange(HighsInt col, double oldbound, double newbound,
                                bool threshold, bool activity,
                                bool infeasdomain);

    void updateActivityUbChange(HighsInt col, double oldbound, double newbound,
                                bool threshold, bool activity,
                                bool infeasdomain);
  };

  struct ConflictPoolPropagation {
    HighsInt conflictpoolindex;
    HighsDomain* domain;
    HighsConflictPool* conflictpool_;

    struct WatchedLiteral {
      HighsDomainChange domchg = {0.0, -1, HighsBoundType::kLower};
      HighsInt prev = -1;
      HighsInt next = -1;
    };

#ifdef HIGHS_RUST
    // Rust's (domain.rs ConfPropState), owned, referred to in place
    highs_rs::ConfPropState* rs_;
    HighsRsArray<HighsInt>& colLowerWatched_;
    HighsRsArray<HighsInt>& colUpperWatched_;
    HighsRsArray<uint8_t>& conflictFlag_;
    HighsRsArray<HighsInt>& propagateConflictInds_;
    HighsRsArray<WatchedLiteral>& watchedLiterals_;
#else
    std::vector<HighsInt> colLowerWatched_;
    std::vector<HighsInt> colUpperWatched_;
    std::vector<uint8_t> conflictFlag_;
    std::vector<HighsInt> propagateConflictInds_;
    std::vector<WatchedLiteral> watchedLiterals_;
#endif

    ConflictPoolPropagation(HighsInt conflictpoolindex, HighsDomain* domain,
                            HighsConflictPool& cutpool);

    ConflictPoolPropagation(const ConflictPoolPropagation& other);

    ConflictPoolPropagation& operator=(const ConflictPoolPropagation& other);

    ~ConflictPoolPropagation();

    void linkWatchedLiteral(HighsInt linkPos);

    void unlinkWatchedLiteral(HighsInt linkPos);

    void conflictAdded(HighsInt conflict);

    void conflictDeleted(HighsInt conflict);

    void markPropagateConflict(HighsInt conflict);

    void updateActivityLbChange(HighsInt col, double oldbound, double newbound);

    void updateActivityUbChange(HighsInt col, double oldbound, double newbound);

    void propagateConflict(HighsInt conflict);
  };

 private:
  struct ObjectivePropagation {
#ifdef HIGHS_RUST
    // the state is Rust's (rust/src/mip/objprop.rs ObjPropState), owned
    HighsDomain* domain = nullptr;
    const HighsObjectiveFunction* objFunc = nullptr;
    const double* cost = nullptr;
    highs_rs::ObjPropState* rs_ = nullptr;

    struct ObjectiveContribution {
      double contribution;
      HighsInt col;
      HighsInt partition;
      highs::RbTreeLinks<HighsInt> links;
    };
    struct PartitionCliqueData {
      double multiplier;
      HighsInt rhs;
      bool changed;
    };

    ObjectivePropagation() = default;
    ObjectivePropagation(HighsDomain* domain);
    ObjectivePropagation(const ObjectivePropagation& other);
    ObjectivePropagation& operator=(const ObjectivePropagation& other);
    ~ObjectivePropagation();

    bool isActive() const { return domain != nullptr; }

    // the objective's lower bound if it is finite
    double objectiveLowerBound() const;

    // construct the proot constraint at the time when the domain change stack
    // had the given size
    void getPropagationConstraint(HighsInt domchgStackSize, const double*& vals,
                                  const HighsInt*& inds, HighsInt& len,
                                  double& rhs, HighsInt domchgCol = -1);
#else
    HighsDomain* domain = nullptr;
    const HighsObjectiveFunction* objFunc;
    const double* cost;
    HighsCDouble objectiveLower;
    HighsInt numInfObjLower;
    double capacityThreshold;
    bool isPropagated;

    struct ObjectiveContribution {
      double contribution;
      HighsInt col;
      HighsInt partition;
      highs::RbTreeLinks<HighsInt> links;
    };

    class ObjectiveContributionTree;

    std::vector<ObjectiveContribution> objectiveLowerContributions;
    std::vector<std::pair<HighsInt, HighsInt>> contributionPartitionSets;
    std::vector<double> propagationConsBuffer;
    struct PartitionCliqueData {
      double multiplier;
      HighsInt rhs;
      bool changed;
    };

    std::vector<PartitionCliqueData> partitionCliqueData;

    ObjectivePropagation() {
      objFunc = nullptr;
      cost = nullptr;
      objectiveLower = 0.0;
      numInfObjLower = 0;
      capacityThreshold = 0.0;
      isPropagated = false;
    }
    ObjectivePropagation(HighsDomain* domain);

    bool isActive() const { return domain != nullptr; }

    void updateActivityLbChange(HighsInt col, double oldbound, double newbound);

    void updateActivityUbChange(HighsInt col, double oldbound, double newbound);

    bool shouldBePropagated() const;

    void propagate();

    void debugCheckObjectiveLower() const;

    // construct the proot constraint at the time when the domain change stack
    // had the given size
    void getPropagationConstraint(HighsInt domchgStackSize, const double*& vals,
                                  const HighsInt*& inds, HighsInt& len,
                                  double& rhs, HighsInt domchgCol = -1);

   private:
    void recomputeCapacityThreshold();
#endif
  };

#ifdef HIGHS_RUST
  // the vectors are Rust's (rust/src/mip/domain.rs DomainVecs), owned by
  // this domain; the members below refer to them in place
  highs_rs::DomainVecs* rsv_;

  HighsRsArray<uint8_t>& changedcolsflags_;
  HighsRsArray<HighsInt>& changedcols_;

  HighsRsArray<HighsDomainChange>& domchgstack_;
  HighsRsArray<Reason>& domchgreason_;
  HighsRsArray<std::pair<double, HighsInt>>& prevboundval_;

  HighsRsArray<HighsCDouble>& activitymin_;
  HighsRsArray<HighsCDouble>& activitymax_;
  HighsRsArray<HighsInt>& activitymininf_;
  HighsRsArray<HighsInt>& activitymaxinf_;
  HighsRsArray<double>& capacityThreshold_;
  HighsRsArray<uint8_t>& propagateflags_;
  HighsRsArray<HighsInt>& propagateinds_;
#else
  std::vector<uint8_t> changedcolsflags_;
  std::vector<HighsInt> changedcols_;

  std::vector<std::pair<HighsInt, HighsInt>> propRowNumChangedBounds_;

  std::vector<HighsDomainChange> domchgstack_;
  std::vector<Reason> domchgreason_;
  std::vector<std::pair<double, HighsInt>> prevboundval_;

  std::vector<HighsCDouble> activitymin_;
  std::vector<HighsCDouble> activitymax_;
  std::vector<HighsInt> activitymininf_;
  std::vector<HighsInt> activitymaxinf_;
  std::vector<double> capacityThreshold_;
  std::vector<uint8_t> propagateflags_;
  std::vector<HighsInt> propagateinds_;
#endif
  ObjectivePropagation objProp_;

  HighsMipSolver* mipsolver;

 private:
  std::deque<CutpoolPropagation> cutpoolpropagation;
  std::deque<ConflictPoolPropagation> conflictPoolPropagation;

  bool infeasible_ = false;
  Reason infeasible_reason;
  HighsInt infeasible_pos;

#ifdef HIGHS_RUST
  // the view of this domain passed to Rust
  highs_rs::DomainCache rsView_;
#endif
  void invalidateRustView() {
#ifdef HIGHS_RUST
    rsView_.valid = false;
#endif
  }

  void updateActivityLbChange(HighsInt col, double oldbound, double newbound);

  void updateActivityUbChange(HighsInt col, double oldbound, double newbound);

  void updateThresholdLbChange(HighsInt col, double newbound, double val,
                               double& threshold) const;

  void updateThresholdUbChange(HighsInt col, double newbound, double val,
                               double& threshold) const;

  void recomputeCapacityThreshold(HighsInt row);

  void updateRedundantRows(HighsInt row);

  double doChangeBound(const HighsDomainChange& boundchg);

#ifdef HIGHS_RUST
  HighsRsArray<HighsInt>& colLowerPos_;
  HighsRsArray<HighsInt>& colUpperPos_;
  HighsRsArray<HighsInt>& branchPos_;
#else
  std::vector<HighsInt> colLowerPos_;
  std::vector<HighsInt> colUpperPos_;
  std::vector<HighsInt> branchPos_;
#endif
  HighsHashTable<HighsInt> redundantRows_;
  bool recordRedundantRows_ = false;

 public:
#ifdef HIGHS_RUST
  HighsRsArray<double>& col_lower_;
  HighsRsArray<double>& col_upper_;

  HighsDomain(HighsMipSolver& mipsolver);
  HighsDomain(const HighsDomain& other);
  HighsDomain& operator=(const HighsDomain& other);
  ~HighsDomain();
#else
  std::vector<double> col_lower_;
  std::vector<double> col_upper_;

  HighsDomain(HighsMipSolver& mipsolver);

  HighsDomain(const HighsDomain& other)
      : changedcolsflags_(other.changedcolsflags_),
        changedcols_(other.changedcols_),
        domchgstack_(other.domchgstack_),
        domchgreason_(other.domchgreason_),
        prevboundval_(other.prevboundval_),
        activitymin_(other.activitymin_),
        activitymax_(other.activitymax_),
        activitymininf_(other.activitymininf_),
        activitymaxinf_(other.activitymaxinf_),
        capacityThreshold_(other.capacityThreshold_),
        propagateflags_(other.propagateflags_),
        propagateinds_(other.propagateinds_),
        objProp_(other.objProp_),
        mipsolver(other.mipsolver),
        cutpoolpropagation(other.cutpoolpropagation),
        conflictPoolPropagation(other.conflictPoolPropagation),
        infeasible_(other.infeasible_),
        infeasible_reason(other.infeasible_reason),
        infeasible_pos(other.infeasible_pos),
        colLowerPos_(other.colLowerPos_),
        colUpperPos_(other.colUpperPos_),
        branchPos_(other.branchPos_),
        col_lower_(other.col_lower_),
        col_upper_(other.col_upper_) {
    for (CutpoolPropagation& cutpoolprop : cutpoolpropagation)
      cutpoolprop.domain = this;
    for (ConflictPoolPropagation& conflictprop : conflictPoolPropagation)
      conflictprop.domain = this;
    if (objProp_.domain) objProp_.domain = this;
  }

  HighsDomain& operator=(const HighsDomain& other) {
    changedcolsflags_ = other.changedcolsflags_;
    changedcols_ = other.changedcols_;
    domchgstack_ = other.domchgstack_;
    domchgreason_ = other.domchgreason_;
    prevboundval_ = other.prevboundval_;
    activitymin_ = other.activitymin_;
    activitymax_ = other.activitymax_;
    activitymininf_ = other.activitymininf_;
    activitymaxinf_ = other.activitymaxinf_;
    capacityThreshold_ = other.capacityThreshold_;
    propagateflags_ = other.propagateflags_;
    propagateinds_ = other.propagateinds_;
    objProp_ = other.objProp_;
    mipsolver = other.mipsolver;
    cutpoolpropagation = other.cutpoolpropagation;
    conflictPoolPropagation = other.conflictPoolPropagation;
    infeasible_ = other.infeasible_;
    infeasible_reason = other.infeasible_reason;
    invalidateRustView();
    colLowerPos_ = other.colLowerPos_;
    colUpperPos_ = other.colUpperPos_;
    branchPos_ = other.branchPos_;
    col_lower_ = other.col_lower_;
    col_upper_ = other.col_upper_;
    for (CutpoolPropagation& cutpoolprop : cutpoolpropagation)
      cutpoolprop.domain = this;
    for (ConflictPoolPropagation& conflictprop : conflictPoolPropagation)
      conflictprop.domain = this;
    if (objProp_.domain) objProp_.domain = this;
    return *this;
  }
#endif

  void computeMinActivity(HighsInt start, HighsInt end, const HighsInt* ARindex,
                          const double* ARvalue, HighsInt& ninfmin,
                          HighsCDouble& activitymin) const;

  void computeMaxActivity(HighsInt start, HighsInt end, const HighsInt* ARindex,
                          const double* ARvalue, HighsInt& ninfmax,
                          HighsCDouble& activitymax) const;

  double adjustedUb(HighsInt col, HighsCDouble boundVal, bool& accept) const;

  // Whether a bound implied for col (an upper bound if upper, else a lower
  // bound), estimated in double as estimate with terms of magnitude up to
  // scale, is surely not strictly inside the current bound, so that
  // adjustedUb or adjustedLb would not accept it. The margin is thousands
  // of times the rounding error of the estimate
  bool impliedBoundNoTighter(HighsInt col, bool upper, double estimate,
                             double scale) const {
    const double margin = 1e-12 * (scale + std::fabs(estimate));
    return upper ? estimate - margin >= col_upper_[col]
                 : estimate + margin <= col_lower_[col];
  }

  double adjustedLb(HighsInt col, HighsCDouble boundVal, bool& accept) const;

  HighsInt propagateRowUpper(const HighsInt* Rindex, const double* Rvalue,
                             HighsInt Rlen, double Rupper,
                             const HighsCDouble& minactivity, HighsInt ninfmin,
                             HighsDomainChange* boundchgs) const;

  HighsInt propagateRowLower(const HighsInt* Rindex, const double* Rvalue,
                             HighsInt Rlen, double Rlower,
                             const HighsCDouble& maxactivity, HighsInt ninfmax,
                             HighsDomainChange* boundchgs) const;

#ifdef HIGHS_RUST
  using IntArray = HighsRsArray<HighsInt>;
  using DomChgArray = HighsRsArray<HighsDomainChange>;
  using ReasonArray = HighsRsArray<Reason>;
  using PrevBoundArray = HighsRsArray<std::pair<double, HighsInt>>;
#else
  using IntArray = std::vector<HighsInt>;
  using DomChgArray = std::vector<HighsDomainChange>;
  using ReasonArray = std::vector<Reason>;
  using PrevBoundArray = std::vector<std::pair<double, HighsInt>>;
#endif

  const IntArray& getChangedCols() const { return changedcols_; }

  void addCutpool(HighsCutPool& cutpool);

  void addConflictPool(HighsConflictPool& conflictPool);

  void clearChangedCols() {
    for (HighsInt i : changedcols_) changedcolsflags_[i] = 0;
    changedcols_.clear();
  }

  void removeContinuousChangedCols() {
    for (HighsInt i : changedcols_)
      changedcolsflags_[i] = mipsolver->isColIntegral(i);

    changedcols_.erase(
        std::remove_if(changedcols_.begin(), changedcols_.end(),
                       [&](HighsInt i) { return !isChangedCol(i); }),
        changedcols_.end());
  }

  void clearChangedCols(size_t start) {
    for (size_t i = start; i != changedcols_.size(); ++i)
      changedcolsflags_[changedcols_[i]] = 0;

    changedcols_.resize(start);
  }

  bool isChangedCol(HighsInt col) const { return changedcolsflags_[col] != 0; }

  void markPropagate(HighsInt row);

  bool isActive(const HighsDomainChange& domchg) const {
    return domchg.boundtype == HighsBoundType::kLower
               ? domchg.boundval <= col_lower_[domchg.column]
               : domchg.boundval >= col_upper_[domchg.column];
  }

  void markPropagateCut(Reason reason);

  void setupObjectivePropagation() {
    invalidateRustView();
    objProp_ = ObjectivePropagation(this);
  }

  void computeRowActivities();

  void markInfeasible(Reason reason = Reason::unspecified()) {
    infeasible_ = true;
    infeasible_pos = domchgstack_.size();
    infeasible_reason = reason;
  }

  bool infeasible() const { return infeasible_; }

  void changeBound(HighsDomainChange boundchg,
                   Reason reason = Reason::branching());

  void changeBound(HighsBoundType boundtype, HighsInt col, double boundval,
                   Reason reason = Reason::branching()) {
    changeBound({boundval, col, boundtype}, reason);
  }

  bool checkChangeBound(HighsBoundType boundtype, HighsInt col,
                        HighsCDouble boundval, Reason reason);

  void fixCol(HighsInt col, double val, Reason reason = Reason::unspecified()) {
    if (kAllowDeveloperAssert) {
      assert(infeasible_ == 0);
    }
    if (col_lower_[col] < val) {
      changeBound({val, col, HighsBoundType::kLower}, reason);
      if (infeasible_ == 0) propagate();
    }

    if (infeasible_ == 0 && col_upper_[col] > val)
      changeBound({val, col, HighsBoundType::kUpper}, reason);
  }

  void backtrackToGlobal();

  // stop propagating the cut and conflict pools (for a domain that only
  // checks fixings against the model rows)
  void clearPoolPropagation() {
    invalidateRustView();
    cutpoolpropagation.clear();
    conflictPoolPropagation.clear();
  }

  HighsDomainChange backtrack();

  const IntArray& getBranchingPositions() const { return branchPos_; }

  const PrevBoundArray& getPreviousBounds() const { return prevboundval_; }

  const DomChgArray& getDomainChangeStack() const { return domchgstack_; }

  const ReasonArray& getDomainChangeReason() const { return domchgreason_; }

#ifdef HIGHS_RUST
  double getObjectiveLowerBound() const {
    return objProp_.isActive() ? objProp_.objectiveLowerBound() : -kHighsInf;
  }
#else
  double getObjectiveLowerBound() const {
    if (objProp_.isActive() && objProp_.numInfObjLower == 0)
      return double(objProp_.objectiveLower);

    return -kHighsInf;
  }
#endif

  void getCutoffConstraint(const double*& vals, const HighsInt*& inds,
                           HighsInt& len, double& rhs) {
    objProp_.getPropagationConstraint(domchgstack_.size(), vals, inds, len,
                                      rhs);
  }

  HighsInt getNumDomainChanges() const { return domchgstack_.size(); }

  bool colBoundsAreGlobal(HighsInt col) const {
    return colLowerPos_[col] == -1 && colUpperPos_[col] == -1;
  }

  HighsInt getBranchDepth() const { return branchPos_.size(); }

  std::vector<HighsDomainChange> getReducedDomainChangeStack(
      std::vector<HighsInt>& branchingPositions) const {
    std::vector<HighsDomainChange> reducedstack;
    reducedstack.reserve(domchgstack_.size());
    branchingPositions.reserve(branchPos_.size());
    for (HighsInt i = 0; i < (HighsInt)domchgstack_.size(); ++i) {
      // keep only the tightest bound change for each variable
      if ((domchgstack_[i].boundtype == HighsBoundType::kLower &&
           colLowerPos_[domchgstack_[i].column] != i) ||
          (domchgstack_[i].boundtype == HighsBoundType::kUpper &&
           colUpperPos_[domchgstack_[i].column] != i))
        continue;

      if (domchgreason_[i].type == Reason::kBranching)
        branchingPositions.push_back(reducedstack.size());
      else {
        HighsInt k = i;
        while (prevboundval_[k].second != -1) {
          k = prevboundval_[k].second;
          if (domchgreason_[k].type == Reason::kBranching) {
            branchingPositions.push_back(reducedstack.size());
            break;
          }
        }
      }

      reducedstack.push_back(domchgstack_[i]);
    }

    reducedstack.shrink_to_fit();
    return reducedstack;
  }

  void setDomainChangeStack(const std::vector<HighsDomainChange>& domchgstack);

  void setDomainChangeStack(const std::vector<HighsDomainChange>& domchgstack,
                            const std::vector<HighsInt>& branchingPositions);

  bool propagate();

  double getColLowerPos(HighsInt col, HighsInt stackpos, HighsInt& pos) const;

  double getColUpperPos(HighsInt col, HighsInt stackpos, HighsInt& pos) const;

  void conflictAnalysis(HighsConflictPool& conflictPool, HighsDomain& globaldom,
                        HighsPseudocost& pseudocost);

  void conflictAnalysis(const HighsInt* proofinds, const double* proofvals,
                        HighsInt prooflen, double proofrhs,
                        HighsConflictPool& conflictPool, HighsDomain& globaldom,
                        HighsPseudocost& pseudocost);

  void conflictAnalyzeReconvergence(const HighsDomainChange& domchg,
                                    const HighsInt* proofinds,
                                    const double* proofvals, HighsInt prooflen,
                                    double proofrhs,
                                    HighsConflictPool& conflictPool,
                                    HighsDomain& globaldom,
                                    HighsPseudocost& pseudocost);

  void tightenCoefficients(HighsInt* inds, double* vals, HighsInt len,
                           double& rhs) const;

  double getMinActivity(HighsInt row) const {
    return activitymininf_[row] == 0 ? double(activitymin_[row]) : -kHighsInf;
  }

  double getMaxActivity(HighsInt row) const {
    return activitymaxinf_[row] == 0 ? double(activitymax_[row]) : kHighsInf;
  }

  double getMinCutActivity(const HighsCutPool& cutpool, HighsInt cut) const;

  bool isBinary(HighsInt col) const {
    return mipsolver->isColIntegral(col) && col_lower_[col] == 0.0 &&
           col_upper_[col] == 1.0;
  }

  bool isGlobalBinary(HighsInt col) const {
    return mipsolver->isColIntegral(col) &&
           mipsolver->model_->col_lower_[col] == 0.0 &&
           mipsolver->model_->col_upper_[col] == 1.0;
  }

  HighsVarType variableType(HighsInt col) const {
    return mipsolver->variableType(col);
  }

  bool isFixed(HighsInt col) const {
    return col_lower_[col] == col_upper_[col];
  }

  bool isFixing(const HighsDomainChange& domchg) const;

  HighsDomainChange flip(const HighsDomainChange& domchg) const;

  double feastol() const;

  HighsInt numModelNonzeros() const { return mipsolver->numNonzero(); }

  bool inSubmip() const { return mipsolver->submip; }

  void clearRedundantRows() { redundantRows_.clear(); };

  const HighsHashTable<HighsInt>& getRedundantRows() const {
    return redundantRows_;
  };

  double getRedundantRowValue(HighsInt row) const;

  void setRecordRedundantRows(bool val) { recordRedundantRows_ = val; };

  bool isRedundantRow(HighsInt row) const;
};

#ifdef HIGHS_RUST
namespace highs_rs {
// rust/src/mip/domain.rs DomainVecs
struct DomainVecs {
  HighsRsArray<double> col_lower;
  HighsRsArray<double> col_upper;
  HighsRsArray<HighsInt> col_lower_pos;
  HighsRsArray<HighsInt> col_upper_pos;
  HighsRsArray<HighsInt> branch_pos;
  HighsRsArray<uint8_t> changedcolsflags;
  HighsRsArray<HighsInt> changedcols;
  HighsRsArray<HighsDomainChange> domchgstack;
  HighsRsArray<HighsDomain::Reason> domchgreason;
  HighsRsArray<std::pair<double, HighsInt>> prevboundval;
  HighsRsArray<HighsCDouble> activitymin;
  HighsRsArray<HighsCDouble> activitymax;
  HighsRsArray<HighsInt> activitymininf;
  HighsRsArray<HighsInt> activitymaxinf;
  HighsRsArray<double> capacity_threshold;
  HighsRsArray<uint8_t> propagateflags;
  HighsRsArray<HighsInt> propagateinds;
  HighsRsArray<std::pair<HighsInt, HighsInt>> scratch_counts;
  HighsRsArray<HighsInt> scratch_inds;
  HighsRsArray<HighsDomainChange> scratch_bounds;
};
// rust/src/mip/domain.rs CutPropState and ConfPropState
struct CutPropState {
  HighsRsArray<HighsCDouble> activitycuts;
  HighsRsArray<HighsInt> activitycutsinf;
  HighsRsArray<uint8_t> propagatecutflags;
  HighsRsArray<HighsInt> propagatecutinds;
  HighsRsArray<double> capacity_threshold;
};
struct ConfPropState {
  HighsRsArray<HighsInt> col_lower_watched;
  HighsRsArray<HighsInt> col_upper_watched;
  HighsRsArray<uint8_t> conflict_flag;
  HighsRsArray<HighsInt> propagate_conflict_inds;
  HighsRsArray<HighsDomain::ConflictPoolPropagation::WatchedLiteral> watched;
};
// rust/src/mip/objprop.rs ObjPropState
struct ObjPropState {
  HighsRsArray<HighsDomain::ObjectivePropagation::ObjectiveContribution>
      contributions;
  HighsRsArray<std::pair<HighsInt, HighsInt>> partition_sets;
  HighsRsArray<double> cons_buffer;
  HighsRsArray<HighsDomain::ObjectivePropagation::PartitionCliqueData>
      clique_data;
  HighsCDouble objective_lower;
  HighsInt num_inf_obj_lower;
  double capacity_threshold;
  bool is_propagated;
};
struct CutPool;
struct ConflictPool;
struct Bounds;
struct Domain;
extern "C" {
ObjPropState* highs_rs_objprop_new(const Bounds* b, const double* cost,
                                   HighsInt ncol, const HighsInt* obj_nonzeros,
                                   HighsInt nnz,
                                   const HighsInt* partition_starts,
                                   HighsInt nstarts, const double* packed,
                                   HighsInt npacked);
ObjPropState* highs_rs_objprop_clone(const ObjPropState* s);
void highs_rs_objprop_free(ObjPropState* s);
void highs_rs_domain_obj_propagation_constraint(const Domain* d,
                                                HighsInt stacksize,
                                                HighsInt domchg_col,
                                                const double** vals,
                                                const HighsInt** inds,
                                                HighsInt* len, double* rhs);
CutPropState* highs_rs_cutprop_new();
CutPropState* highs_rs_cutprop_clone(const CutPropState* s);
void highs_rs_cutprop_assign(CutPropState* d, const CutPropState* s);
void highs_rs_cutprop_free(CutPropState* s);
void highs_rs_cutprop_cut_added(CutPropState* s, const CutPool* pool,
                                HighsInt cut, const Bounds* b, bool propagate,
                                bool global);
void highs_rs_cutprop_cut_deleted(CutPropState* s, HighsInt cut, bool keep);
ConfPropState* highs_rs_confprop_new(HighsInt ncol);
ConfPropState* highs_rs_confprop_clone(const ConfPropState* s);
void highs_rs_confprop_assign(ConfPropState* d, const ConfPropState* s);
void highs_rs_confprop_free(ConfPropState* s);
void highs_rs_confprop_conflict_added(ConfPropState* s,
                                      const ConflictPool* pool,
                                      HighsInt conflict, const Bounds* b);
void highs_rs_confprop_conflict_deleted(ConfPropState* s, HighsInt conflict);
DomainVecs* highs_rs_domain_vecs_new(HighsInt ncol, const double* lower,
                                     const double* upper);
DomainVecs* highs_rs_domain_vecs_clone(const DomainVecs* v);
void highs_rs_domain_vecs_assign(DomainVecs* dst, const DomainVecs* src);
void highs_rs_domain_vecs_free(DomainVecs* v);
void highs_rs_domain_vecs_size_rows(DomainVecs* v, HighsInt nrow);
void highs_rs_reserve_i32(void* v, size_t n);
void highs_rs_reserve_domchg(void* v, size_t n);
void highs_rs_reserve_reason(void* v, size_t n);
void highs_rs_reserve_prev(void* v, size_t n);
void highs_rs_reserve_pair(void* v, size_t n);
}
}  // namespace highs_rs
#endif

#endif
