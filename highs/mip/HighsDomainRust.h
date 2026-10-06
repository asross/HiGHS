/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file mip/HighsDomainRust.h
 * @brief The view of HighsDomain's data passed to the Rust port of
 * HighsDomain (rust/src/mip/domain.rs, whose module comment describes the
 * design)
 */
#ifndef MIP_HIGHSDOMAINRUST_H_
#define MIP_HIGHSDOMAINRUST_H_

#include "HConfig.h"

#ifdef HIGHS_RUST

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>

#include "mip/HighsConflictPool.h"
#include "mip/HighsCutPool.h"
#include "mip/HighsDomain.h"
#include "mip/HighsDomainRustView.h"
#include "mip/HighsMipSolverData.h"

#ifdef HIGHS_DEBUGSOL
#error "the Rust domain does not call the debug solution's checks"
#endif

namespace highs_rs {

// a slice of a const array (which Rust only reads)
template <typename T>
DSlice<T> cslice(const T* p, HighsInt n) {
  return {nonNull(const_cast<T*>(p)), (int)n};
}

// The layouts the Rust side assumes
static_assert(sizeof(HighsInt) == 4, "HighsInt is i32 in Rust");
// (hi and lo are private, declared in this order)
static_assert(sizeof(HighsCDouble) == 16 &&
                  std::is_standard_layout<HighsCDouble>::value,
              "HighsCDouble is CDouble {hi, lo}");
static_assert(sizeof(HighsDomainChange) == 16 &&
                  offsetof(HighsDomainChange, boundval) == 0 &&
                  offsetof(HighsDomainChange, column) == 8 &&
                  offsetof(HighsDomainChange, boundtype) == 12 &&
                  sizeof(HighsBoundType) == 4,
              "HighsDomainChange is DomChg");
static_assert(sizeof(HighsDomain::Reason) == 8, "Reason is {i32, i32}");
static_assert(sizeof(std::pair<double, HighsInt>) == 16 &&
                  sizeof(std::pair<HighsInt, HighsInt>) == 8,
              "pairs are PrevBound and [i32; 2]");
static_assert(sizeof(HighsDomain::ConflictPoolPropagation::WatchedLiteral) ==
                  24,
              "WatchedLiteral is {DomChg, i32, i32}");
static_assert(sizeof(HighsVarType) == 1 && sizeof(bool) == 1,
              "integrality is u8");
static_assert(sizeof(std::vector<HighsInt>) == 3 * sizeof(void*),
              "std::vector is StdVec {begin, end, capacity end}");

extern "C" {
void highs_rs_domain_change_bound(const Domain* d, HighsDomainChange chg,
                                  HighsDomain::Reason reason);
bool highs_rs_domain_propagate(const Domain* d);
HighsDomainChange highs_rs_domain_backtrack(const Domain* d, bool toGlobal);
void highs_rs_domain_set_domain_change_stack(const Domain* d,
                                             const HighsDomainChange* stack,
                                             int len, const int* branching,
                                             int nbranching);
void highs_rs_domain_tighten_coefficients(const Bounds* b, const int* inds,
                                          double* vals, int len, double* rhs);
void highs_rs_domain_compute_row_activities(const Domain* d);
void highs_rs_domain_compute_activity(const Bounds* b, const int* index,
                                      const double* value, int len, bool max,
                                      int* ninf, HighsCDouble* activity);
int highs_rs_domain_propagate_row(const Bounds* b, const int* index,
                                  const double* value, int len, double rhs,
                                  const HighsCDouble* activity, int ninf,
                                  bool lower, HighsDomainChange* out);
}

// Fills a domain's view: view() the cached one (HighsDomain::rsView_) of a
// domain owned by the calling thread, bounds() the data of the const
// methods, which other threads may call at the same time on the global
// domain. The static functions are the calls from Rust into C++
struct DomainAccess {
  using Contribution = HighsDomain::ObjectivePropagation::ObjectiveContribution;
  static_assert(
      sizeof(HighsDomain::ObjectivePropagation::PartitionCliqueData) == 16 &&
          offsetof(HighsDomain::ObjectivePropagation::PartitionCliqueData,
                   rhs) == 8,
      "PartitionCliqueData is CliqueData");
  static_assert(sizeof(Contribution) == 32 &&
                    offsetof(Contribution, col) == 8 &&
                    offsetof(Contribution, partition) == 12 &&
                    offsetof(Contribution, links) == 16 &&
                    sizeof(highs::RbTreeLinks<HighsInt>) == 12,
                "ObjectiveContribution is Contribution");

  template <typename T>
  static void reserve(void* v, size_t n) {
    std::vector<T>& x = *static_cast<std::vector<T>*>(v);
    x.reserve(std::max(n, 2 * x.capacity()));
  }

  // whether std::vector is {begin, end, capacity end}, as Rust's StdVec
  static bool stdVecLayout() {
    std::vector<HighsInt> v;
    v.reserve(4);
    v.push_back(1);
    HighsInt* const* p = reinterpret_cast<HighsInt* const*>(&v);
    return p[0] == v.data() && p[1] == v.data() + 1 && p[2] == v.data() + 4;
  }

  // the fixings implied by fixing binary col to val
  static void implications(void* d, int col, int val) {
    HighsDomain& dom = *static_cast<HighsDomain*>(d);
    HighsMipSolverData& mipdata = *dom.mipsolver->mipdata_;
    mipdata.cliquetable.addImplications(dom, col, val);
    if (!dom.infeasible_) mipdata.implications.applyImplications(dom, col, val);
  }

  static void redundantRow(void* d, int row) {
    static_cast<HighsDomain*>(d)->redundantRows_.insert(row);
  }

  static void cutResetAge(void* d, int pool, int cut) {
    HighsDomain& dom = *static_cast<HighsDomain*>(d);
    dom.cutpoolpropagation[pool].cutpool->resetAge(
        cut, dom.mipsolver->mipdata_->parallelLockActive());
  }

  static void conflictResetAge(void* d, int pool, int conflict) {
    HighsDomain& dom = *static_cast<HighsDomain*>(d);
    dom.conflictPoolPropagation[pool].conflictpool_->resetAge(conflict);
  }

  static void fillCutProp(HighsDomain::CutpoolPropagation& cp, CutProp& v) {
    v.cutpoolindex = cp.cutpoolindex;
    v.cutpool = cp.cutpool;
    v.activitycuts = dslice(cp.activitycuts_);
    v.activitycutsinf = dslice(cp.activitycutsinf_);
    v.propagatecutflags = dslice(cp.propagatecutflags_);
    v.capacity_threshold = dslice(cp.capacityThreshold_);
    v.propagatecutinds = &cp.propagatecutinds_;
    cp.cutpool->getMatrix().rustView(v);
  }

  static void fillConfProp(HighsDomain::ConflictPoolPropagation& cp,
                           ConfProp& v) {
    v.col_lower_watched = dslice(cp.colLowerWatched_);
    v.col_upper_watched = dslice(cp.colUpperWatched_);
    v.watched = {nonNull(cp.watchedLiterals_.data()),
                 (int)cp.watchedLiterals_.size()};
    v.conflict_flag = dslice(cp.conflictFlag_);
    v.propagate_conflict_inds = &cp.propagateConflictInds_;
    v.pool = cp.conflictpool_->rust();
  }

  // the arrays of a cut pool's propagation (or its matrix) may have moved:
  // update its part of a valid view
  static void cutPoolChanged(HighsDomain& dom, HighsInt pool) {
    if (dom.rsView_.valid)
      fillCutProp(dom.cutpoolpropagation[pool], dom.rsView_.cuts[pool]);
  }

  // as cutPoolChanged for a conflict pool
  static void conflictPoolChanged(HighsDomain& dom, HighsInt pool) {
    if (dom.rsView_.valid)
      fillConfProp(dom.conflictPoolPropagation[pool],
                   dom.rsView_.conflicts[pool]);
  }

  static void fill(HighsDomain& dom, DomainCache& c) {
    static const bool layoutOk = stdVecLayout();
    if (!layoutOk) {
      fprintf(stderr, "HighsDomainRust: unexpected std::vector layout\n");
      abort();
    }
    Domain& d = c.d;
    c.valid = true;
    HighsMipSolverData& mipdata = *dom.mipsolver->mipdata_;
    const HighsLp& model = *dom.mipsolver->model_;
    d.feastol = &mipdata.feastol;
    d.epsilon = &mipdata.epsilon;
    d.upper_limit = &mipdata.upper_limit;
    d.a_start = dslice(model.a_matrix_.start_);
    d.a_index = dslice(model.a_matrix_.index_);
    d.a_value = dslice(model.a_matrix_.value_);
    d.ar_start = dslice(mipdata.ARstart_);
    d.ar_index = dslice(mipdata.ARindex_);
    d.ar_value = dslice(mipdata.ARvalue_);
    d.row_lower = dslice(model.row_lower_);
    d.row_upper = dslice(model.row_upper_);
    d.integrality = dslice(model.integrality_);
    d.col_lower = dslice(dom.col_lower_);
    d.col_upper = dslice(dom.col_upper_);
    d.activitymin = dslice(dom.activitymin_);
    d.activitymax = dslice(dom.activitymax_);
    d.activitymininf = dslice(dom.activitymininf_);
    d.activitymaxinf = dslice(dom.activitymaxinf_);
    d.capacity_threshold = dslice(dom.capacityThreshold_);
    d.propagateflags = dslice(dom.propagateflags_);
    d.col_lower_pos = dslice(dom.colLowerPos_);
    d.col_upper_pos = dslice(dom.colUpperPos_);
    d.changedcolsflags = dslice(dom.changedcolsflags_);
    d.propagateinds = &dom.propagateinds_;
    d.changedcols = &dom.changedcols_;
    d.branchpos = &dom.branchPos_;
    d.domchgstack = &dom.domchgstack_;
    d.domchgreason = &dom.domchgreason_;
    d.prevboundval = &dom.prevboundval_;
    d.scratch_inds = &dom.rsScratchInds_;
    d.scratch_bounds = &dom.rsScratchBounds_;
    d.scratch_counts = &dom.propRowNumChangedBounds_;
    d.infeasible = &dom.infeasible_;
    d.infeasible_reason = &dom.infeasible_reason;
    d.infeasible_pos = &dom.infeasible_pos;
    d.record_redundant_rows = &dom.recordRedundantRows_;
    c.cuts.resize(dom.cutpoolpropagation.size());
    c.conflicts.resize(dom.conflictPoolPropagation.size());
    d.cutpools = {nonNull(c.cuts.data()), 0};
    d.conflictpools = {nonNull(c.conflicts.data()), 0};
    for (HighsDomain::CutpoolPropagation& cp : dom.cutpoolpropagation)
      fillCutProp(cp, c.cuts[d.cutpools.n++]);
    for (HighsDomain::ConflictPoolPropagation& cp :
         dom.conflictPoolPropagation)
      fillConfProp(cp, c.conflicts[d.conflictpools.n++]);
    HighsDomain::ObjectivePropagation& op = dom.objProp_;
    ObjProp& o = d.objprop;
    o = ObjProp();
    o.active = op.isActive();
    if (o.active) {
      const HighsObjectiveFunction& f = *op.objFunc;
      o.cost = dslice(model.col_cost_);
      o.obj_nonzeros = dslice(f.getObjectiveNonzeros());
      o.partition_starts = dslice(f.getCliquePartitionStarts());
      o.col_to_partition = dslice(f.getColToPartition());
      o.num_binaries = f.getNumBinariesInObjective();
      o.contributions = {nonNull(op.objectiveLowerContributions.data()),
                         (int)op.objectiveLowerContributions.size()};
      o.partition_sets = dslice(op.contributionPartitionSets);
      o.objective_lower = &op.objectiveLower;
      o.num_inf_obj_lower = &op.numInfObjLower;
      o.capacity_threshold = &op.capacityThreshold;
      o.is_propagated = &op.isPropagated;
      o.obj_vals = dslice(f.getObjectiveValuesPacked());
      o.clique_data = {nonNull(op.partitionCliqueData.data()),
                       (int)op.partitionCliqueData.size()};
      o.cons_buffer = dslice(op.propagationConsBuffer);
    }
    d.dom = &dom;
    d.implications = implications;
    d.redundant_row = redundantRow;
    d.cut_reset_age = cutResetAge;
    d.conflict_reset_age = conflictResetAge;
    d.reserve_i32 = reserve<HighsInt>;
    d.reserve_domchg = reserve<HighsDomainChange>;
    d.reserve_reason = reserve<HighsDomain::Reason>;
    d.reserve_prev = reserve<std::pair<double, HighsInt>>;
    d.reserve_pair = reserve<std::pair<HighsInt, HighsInt>>;
  }

  template <typename T>
  static bool same(const DSlice<T>& a, const DSlice<T>& b) {
    return a.p == b.p && a.n == b.n;
  }

  static Bounds bounds(const HighsDomain& dom) {
    const HighsMipSolverData& mipdata = *dom.mipsolver->mipdata_;
    Bounds b;
    b.feastol = mipdata.feastol;
    b.epsilon = mipdata.epsilon;
    b.col_lower = dslice(dom.col_lower_);
    b.col_upper = dslice(dom.col_upper_);
    b.integrality = dslice(dom.mipsolver->model_->integrality_);
    b.col_lower_pos = dslice(dom.colLowerPos_);
    b.col_upper_pos = dslice(dom.colUpperPos_);
    b.prevboundval = dslice(dom.prevboundval_);
    b.infeasible = dom.infeasible_;
    b.infeasible_pos = dom.infeasible_pos;
    return b;
  }

  static const Domain* view(HighsDomain& dom) {
    DomainCache& c = dom.rsView_;
    if (!c.valid) fill(dom, c);
#ifndef NDEBUG
    {
      // the cache must equal a fresh fill
      DomainCache f;
      fill(dom, f);
      const Domain &a = c.d, &b = f.d;
#define HIGHS_RS_SAME(x) assert(same(a.x, b.x))
      HIGHS_RS_SAME(a_start);
      HIGHS_RS_SAME(a_index);
      HIGHS_RS_SAME(a_value);
      HIGHS_RS_SAME(ar_start);
      HIGHS_RS_SAME(ar_index);
      HIGHS_RS_SAME(ar_value);
      HIGHS_RS_SAME(row_lower);
      HIGHS_RS_SAME(row_upper);
      HIGHS_RS_SAME(integrality);
      HIGHS_RS_SAME(col_lower);
      HIGHS_RS_SAME(col_upper);
      HIGHS_RS_SAME(activitymin);
      HIGHS_RS_SAME(activitymax);
      HIGHS_RS_SAME(activitymininf);
      HIGHS_RS_SAME(activitymaxinf);
      HIGHS_RS_SAME(capacity_threshold);
      HIGHS_RS_SAME(propagateflags);
      HIGHS_RS_SAME(col_lower_pos);
      HIGHS_RS_SAME(col_upper_pos);
      HIGHS_RS_SAME(changedcolsflags);
      HIGHS_RS_SAME(objprop.cost);
      HIGHS_RS_SAME(objprop.obj_nonzeros);
      HIGHS_RS_SAME(objprop.partition_starts);
      HIGHS_RS_SAME(objprop.col_to_partition);
      HIGHS_RS_SAME(objprop.contributions);
      HIGHS_RS_SAME(objprop.partition_sets);
#undef HIGHS_RS_SAME
      assert(a.objprop.active == b.objprop.active &&
             a.objprop.objective_lower == b.objprop.objective_lower &&
             c.cuts.size() == f.cuts.size() &&
             c.conflicts.size() == f.conflicts.size());
      for (size_t i = 0; i < c.cuts.size(); ++i) {
        const CutProp &x = c.cuts[i], &y = f.cuts[i];
        assert(x.cutpoolindex == y.cutpoolindex &&
               same(x.activitycuts, y.activitycuts) &&
               same(x.activitycutsinf, y.activitycutsinf) &&
               same(x.propagatecutflags, y.propagatecutflags) &&
               same(x.capacity_threshold, y.capacity_threshold) &&
               x.propagatecutinds == y.propagatecutinds &&
               same(x.ar_range, y.ar_range) && same(x.ar_index, y.ar_index) &&
               same(x.ar_value, y.ar_value) &&
               same(x.ar_rowindex, y.ar_rowindex) &&
               same(x.next_pos, y.next_pos) && same(x.next_neg, y.next_neg) &&
               same(x.head_pos, y.head_pos) && same(x.head_neg, y.head_neg) &&
               same(x.rhs, y.rhs));
      }
      for (size_t i = 0; i < c.conflicts.size(); ++i) {
        const ConfProp &x = c.conflicts[i], &y = f.conflicts[i];
        assert(same(x.col_lower_watched, y.col_lower_watched) &&
               same(x.col_upper_watched, y.col_upper_watched) &&
               same(x.watched, y.watched) &&
               same(x.conflict_flag, y.conflict_flag) &&
               x.propagate_conflict_inds == y.propagate_conflict_inds &&
               x.pool == y.pool);
      }
    }
#endif
    return &c.d;
  }
};

// Mirror of CConflict: the conflict analysis of a local domain
struct Conflict {
  const Domain* local;
  const Domain* global;
  HighsConflictPool* pool;
  highs_rs::Pseudocost* pseudocost;
  const highs_rs::NodeQueue* nodequeue;
  HighsInt num_integral;
  void (*add_cut)(const Conflict*, const HighsDomainChange*, int,
                  const HighsDomainChange*);

  static void addCut(const Conflict* c, const HighsDomainChange* entries,
                     int len, const HighsDomainChange* domchg) {
    HighsDomain& local = *static_cast<HighsDomain*>(c->local->dom);
    HighsDomain& global = *static_cast<HighsDomain*>(c->global->dom);
    if (domchg)
      c->pool->addReconvergenceCut(local, entries, len, *domchg);
    else
      c->pool->addConflictCut(local, entries, len);
    // the pool's propagation domains resized their arrays: refill the views
    // (in place)
    DomainAccess::view(local);
    DomainAccess::view(global);
  }

  Conflict(HighsDomain& local, HighsDomain& global, HighsConflictPool& pool,
           HighsPseudocost& pseudocost, const HighsMipSolverData& mipdata)
      : local(DomainAccess::view(local)),
        global(DomainAccess::view(global)),
        pool(&pool),
        pseudocost(pseudocost.rust()),
        nodequeue(mipdata.nodequeue.rust()),
        num_integral((HighsInt)mipdata.integral_cols.size()),
        add_cut(addCut) {}
};

extern "C" {
void highs_rs_conflict_analysis(const Conflict* c);
void highs_rs_conflict_analysis_proof(const Conflict* c, const int* inds,
                                      const double* vals, int len, double rhs);
void highs_rs_conflict_reconvergence(const Conflict* c, HighsDomainChange domchg,
                                     const int* inds, const double* vals,
                                     int len, double rhs);
}

}  // namespace highs_rs

inline void HighsDynamicRowMatrix::rustView(highs_rs::CutProp& c) const {
  using highs_rs::cslice;
  c.ar_range = cslice(v_.ar_range, v_.num_rows);
  c.ar_index = cslice(v_.ar_index, v_.num_nz);
  c.ar_value = cslice(v_.ar_value, v_.num_nz);
  c.ar_rowindex = cslice(v_.ar_rowindex, v_.num_nz);
  c.next_pos = cslice(v_.next_pos, v_.num_nz);
  c.next_neg = cslice(v_.next_neg, v_.num_nz);
  c.head_pos = cslice(v_.head_pos, v_.num_cols);
  c.head_neg = cslice(v_.head_neg, v_.num_cols);
  c.rhs = cslice(v_.rhs, v_.num_rhs);
}

#endif
#endif
