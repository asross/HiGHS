/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file mip/HighsDomainRustView.h
 * @brief The plain-data view of a HighsDomain passed to the Rust port of its
 * propagation engine (rust/src/mip/domain.rs, whose module comment describes
 * the design); filled in mip/HighsDomainRust.h
 */
#ifndef MIP_HIGHSDOMAINRUSTVIEW_H_
#define MIP_HIGHSDOMAINRUSTVIEW_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "lp_data/HConst.h"
#include "util/HighsCDouble.h"
#include "util/HighsInt.h"

namespace highs_rs {

// Mirror of CSlice in rust/src/mip/domain.rs
template <typename T>
struct DSlice {
  T* p;
  int n;
};

template <typename T>
DSlice<T> dslice(std::vector<T>& v) {
  return {v.data(), (int)v.size()};
}

template <typename T>
DSlice<T> dslice(const std::vector<T>& v) {
  // The Rust side only reads the vectors it gets as const
  return {const_cast<T*>(v.data()), (int)v.size()};
}

using PushFn = void (*)(void*, int);

// Mirror of CCutProp: a CutpoolPropagation and its cut pool's matrix
struct CutProp {
  int cutpoolindex;
  DSlice<HighsCDouble> activitycuts;
  DSlice<HighsInt> activitycutsinf;
  DSlice<uint8_t> propagatecutflags;
  DSlice<double> capacity_threshold;
  void* propagatecutinds;  // std::vector<HighsInt>*
  DSlice<std::pair<HighsInt, HighsInt>> ar_range;
  DSlice<HighsInt> ar_index;
  DSlice<double> ar_value;
  DSlice<HighsInt> ar_rowindex;
  DSlice<HighsInt> next_pos;
  DSlice<HighsInt> next_neg;
  DSlice<HighsInt> head_pos;
  DSlice<HighsInt> head_neg;
  DSlice<double> rhs;
};

// Mirror of CConfProp: a ConflictPoolPropagation
struct ConfProp {
  DSlice<HighsInt> col_lower_watched;
  DSlice<HighsInt> col_upper_watched;
  DSlice<void> watched;  // WatchedLiteral
  DSlice<uint8_t> conflict_flag;
  void* propagate_conflict_inds;  // std::vector<HighsInt>*
};

// Mirror of CDomain
struct Domain {
  double feastol;
  double epsilon;
  DSlice<HighsInt> a_start;
  DSlice<HighsInt> a_index;
  DSlice<double> a_value;
  DSlice<HighsInt> ar_start;
  DSlice<HighsInt> ar_index;
  DSlice<double> ar_value;
  DSlice<double> row_lower;
  DSlice<double> row_upper;
  DSlice<HighsVarType> integrality;
  DSlice<double> col_lower;
  DSlice<double> col_upper;
  DSlice<HighsCDouble> activitymin;
  DSlice<HighsCDouble> activitymax;
  DSlice<HighsInt> activitymininf;
  DSlice<HighsInt> activitymaxinf;
  DSlice<double> capacity_threshold;
  DSlice<uint8_t> propagateflags;
  void* propagateinds;  // std::vector<HighsInt>*
  DSlice<HighsInt> col_lower_pos;
  DSlice<HighsInt> col_upper_pos;
  DSlice<std::pair<double, HighsInt>> prevboundval;
  bool* infeasible;
  void* infeasible_reason;  // HighsDomain::Reason*
  HighsInt* infeasible_pos;
  HighsInt domchgstack_size;
  DSlice<CutProp> cutpools;
  DSlice<ConfProp> conflictpools;
  PushFn push;
};

// A HighsDomain's view, kept until a vector it points to may have moved:
// HighsDomain resets `valid` when it is copied or assigned, when it
// (re)sizes its row arrays, and when its pools change or add a cut or
// conflict (which resizes their arrays and may move the cut matrix). The
// sizes of prevboundval_ and domchgstack_, which change all the time, are
// refreshed on every use
struct DomainCache {
  Domain d;
  std::vector<CutProp> cuts;
  std::vector<ConfProp> conflicts;
  bool valid = false;

  DomainCache() = default;
  // a copy points to the other domain's data
  DomainCache(const DomainCache&) {}
  DomainCache& operator=(const DomainCache&) {
    valid = false;
    return *this;
  }
};

}  // namespace highs_rs

#endif
