/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file mip/HighsDomainRust.h
 * @brief The view of HighsDomain's data passed to the Rust port of its
 * propagation engine (rust/src/mip/domain.rs, whose module comment
 * describes the design)
 */
#ifndef MIP_HIGHSDOMAINRUST_H_
#define MIP_HIGHSDOMAINRUST_H_

#include "HConfig.h"

#ifdef HIGHS_RUST

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>

#include "mip/HighsCutPool.h"
#include "mip/HighsDomain.h"
#include "mip/HighsDomainRustView.h"
#include "mip/HighsMipSolverData.h"

namespace highs_rs {

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

// Fills a domain's view: view() the cached one (HighsDomain::rsView_) of a
// domain owned by the calling thread, ConstView a temporary one without
// the pools for the const methods, which other threads may call at the same
// time on the global domain
struct DomainAccess {
  static void push(void* v, int x) {
    static_cast<std::vector<HighsInt>*>(v)->push_back(x);
  }

  static void fill(const HighsDomain& constdom, DomainCache& c, bool pools) {
    // the kernels write only to the propagation arrays, which the C++ const
    // methods do not touch, and to what their callers own
    HighsDomain& dom = const_cast<HighsDomain&>(constdom);
    Domain& d = c.d;
    c.valid = true;
    const HighsMipSolverData& mipdata = *dom.mipsolver->mipdata_;
    const HighsLp& model = *dom.mipsolver->model_;
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
    d.propagateinds = &dom.propagateinds_;
    d.col_lower_pos = dslice(dom.colLowerPos_);
    d.col_upper_pos = dslice(dom.colUpperPos_);
    d.infeasible = &dom.infeasible_;
    d.infeasible_reason = &dom.infeasible_reason;
    d.infeasible_pos = &dom.infeasible_pos;
    d.push = push;
    d.cutpools = {nullptr, 0};
    d.conflictpools = {nullptr, 0};
    if (!pools) return;
    c.cuts.resize(dom.cutpoolpropagation.size());
    c.conflicts.resize(dom.conflictPoolPropagation.size());
    d.cutpools = {c.cuts.data(), 0};
    d.conflictpools = {c.conflicts.data(), 0};
    for (HighsDomain::CutpoolPropagation& cp : dom.cutpoolpropagation) {
      CutProp& v = c.cuts[d.cutpools.n++];
      v.cutpoolindex = cp.cutpoolindex;
      v.activitycuts = dslice(cp.activitycuts_);
      v.activitycutsinf = dslice(cp.activitycutsinf_);
      v.propagatecutflags = dslice(cp.propagatecutflags_);
      v.capacity_threshold = dslice(cp.capacityThreshold_);
      v.propagatecutinds = &cp.propagatecutinds_;
      cp.cutpool->getMatrix().rustView(v);
      v.rhs = dslice(cp.cutpool->getRhs());
    }
    for (HighsDomain::ConflictPoolPropagation& cp :
         dom.conflictPoolPropagation) {
      ConfProp& v = c.conflicts[d.conflictpools.n++];
      v.col_lower_watched = dslice(cp.colLowerWatched_);
      v.col_upper_watched = dslice(cp.colUpperWatched_);
      v.watched = {cp.watchedLiterals_.data(), (int)cp.watchedLiterals_.size()};
      v.conflict_flag = dslice(cp.conflictFlag_);
      v.propagate_conflict_inds = &cp.propagateConflictInds_;
    }
  }

  template <typename T>
  static bool same(const DSlice<T>& a, const DSlice<T>& b) {
    return a.p == b.p && a.n == b.n;
  }

  static void setDynamic(const HighsDomain& dom, Domain& d) {
    const HighsMipSolverData& mipdata = *dom.mipsolver->mipdata_;
    d.feastol = mipdata.feastol;
    d.epsilon = mipdata.epsilon;
    d.prevboundval = dslice(dom.prevboundval_);
    d.domchgstack_size = dom.domchgstack_.size();
  }

  static const Domain* view(HighsDomain& dom) {
    DomainCache& c = dom.rsView_;
    if (!c.valid) fill(dom, c, true);
#ifndef NDEBUG
    {
      // the cache must equal a fresh fill
      const Domain a = c.d;
      const std::vector<CutProp> cuts = c.cuts;
      const std::vector<ConfProp> conflicts = c.conflicts;
      fill(dom, c, true);
      const Domain& b = c.d;
      assert(same(a.a_start, b.a_start) && same(a.a_index, b.a_index) &&
             same(a.a_value, b.a_value) && same(a.ar_start, b.ar_start) &&
             same(a.ar_index, b.ar_index) && same(a.ar_value, b.ar_value) &&
             same(a.row_lower, b.row_lower) && same(a.row_upper, b.row_upper) &&
             same(a.integrality, b.integrality) &&
             same(a.col_lower, b.col_lower) && same(a.col_upper, b.col_upper) &&
             same(a.activitymin, b.activitymin) &&
             same(a.activitymax, b.activitymax) &&
             same(a.activitymininf, b.activitymininf) &&
             same(a.activitymaxinf, b.activitymaxinf) &&
             same(a.capacity_threshold, b.capacity_threshold) &&
             same(a.propagateflags, b.propagateflags) &&
             a.propagateinds == b.propagateinds &&
             same(a.col_lower_pos, b.col_lower_pos) &&
             same(a.col_upper_pos, b.col_upper_pos) &&
             a.infeasible == b.infeasible &&
             a.infeasible_reason == b.infeasible_reason &&
             a.infeasible_pos == b.infeasible_pos &&
             cuts.size() == c.cuts.size() &&
             conflicts.size() == c.conflicts.size());
      for (size_t i = 0; i < cuts.size(); ++i) {
        const CutProp &x = cuts[i], &y = c.cuts[i];
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
      for (size_t i = 0; i < conflicts.size(); ++i) {
        const ConfProp &x = conflicts[i], &y = c.conflicts[i];
        assert(same(x.col_lower_watched, y.col_lower_watched) &&
               same(x.col_upper_watched, y.col_upper_watched) &&
               same(x.watched, y.watched) &&
               same(x.conflict_flag, y.conflict_flag) &&
               x.propagate_conflict_inds == y.propagate_conflict_inds);
      }
    }
#endif
    setDynamic(dom, c.d);
    return &c.d;
  }
};

struct ConstView {
  DomainCache c;
  explicit ConstView(const HighsDomain& dom) {
    DomainAccess::fill(dom, c, false);
    DomainAccess::setDynamic(dom, c.d);
  }
  const Domain* get() const { return &c.d; }
};

extern "C" {
void highs_rs_domain_update_activity(const Domain* d, int col, double oldbound,
                                     double newbound, bool upper);
void highs_rs_domain_compute_row_activities(const Domain* d);
void highs_rs_domain_compute_activity(const Domain* d, const int* index,
                                      const double* value, int len, bool max,
                                      int* ninf, HighsCDouble* activity);
int highs_rs_domain_propagate_row(const Domain* d, const int* index,
                                  const double* value, int len, double rhs,
                                  const HighsCDouble* activity, int ninf,
                                  bool lower, HighsDomainChange* out);
void highs_rs_domain_propagate_model_rows(const Domain* d, const int* rows,
                                          int nrows,
                                          std::pair<HighsInt, HighsInt>* counts,
                                          HighsDomainChange* changedbounds,
                                          int nchangedbounds);
void highs_rs_domain_propagate_cuts(const Domain* d, int pool, const int* cuts,
                                    int ncuts,
                                    std::pair<HighsInt, HighsInt>* counts,
                                    HighsDomainChange* changedbounds,
                                    int nchangedbounds);
void highs_rs_domain_mark_propagate(const Domain* d, int row);
void highs_rs_domain_cut_recompute_capacity_threshold(const Domain* d, int pool,
                                                      int cut);
}

}  // namespace highs_rs

inline void HighsDynamicRowMatrix::rustView(highs_rs::CutProp& c) const {
  using highs_rs::dslice;
  c.ar_range = dslice(ARrange_);
  c.ar_index = dslice(ARindex_);
  c.ar_value = dslice(ARvalue_);
  c.ar_rowindex = dslice(ARrowindex_);
  c.next_pos = dslice(AnextPos_);
  c.next_neg = dslice(AnextNeg_);
  c.head_pos = dslice(AheadPos_);
  c.head_neg = dslice(AheadNeg_);
}

#endif
#endif
