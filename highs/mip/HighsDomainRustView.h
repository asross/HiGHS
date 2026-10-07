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

#include <cstddef>
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

// Rust's slices need a non-null pointer, also when empty
template <typename T>
T* nonNull(T* p) {
  return p ? p : reinterpret_cast<T*>(alignof(T));
}

template <typename T>
DSlice<T> dslice(std::vector<T>& v) {
  return {nonNull(v.data()), (int)v.size()};
}

template <typename T>
DSlice<T> dslice(const std::vector<T>& v) {
  // The Rust side only reads the vectors it gets as const
  return {nonNull(const_cast<T*>(v.data())), (int)v.size()};
}

// Makes a std::vector's capacity at least n (and at least twice the old)
using ReserveFn = void (*)(void*, size_t);

// Mirror of CCutProp: a CutpoolPropagation and its cut pool's matrix
struct CutProp {
  int cutpoolindex;
  const void* cutpool;  // HighsCutPool*
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

// Mirror of CConfProp: a ConflictPoolPropagation and its pool's conflicts
struct ConfProp {
  DSlice<HighsInt> col_lower_watched;
  DSlice<HighsInt> col_upper_watched;
  DSlice<void> watched;  // WatchedLiteral
  DSlice<uint8_t> conflict_flag;
  void* propagate_conflict_inds;  // std::vector<HighsInt>*
  void* pool;                     // highs_rs::ConflictPool*
};

// Mirror of CObjProp: the ObjectivePropagation
struct ObjProp {
  bool active;
  DSlice<double> cost;
  DSlice<HighsInt> obj_nonzeros;
  DSlice<HighsInt> partition_starts;
  DSlice<HighsInt> col_to_partition;
  HighsInt num_binaries;
  DSlice<void> contributions;  // ObjectiveContribution
  DSlice<std::pair<HighsInt, HighsInt>> partition_sets;
  HighsCDouble* objective_lower;
  HighsInt* num_inf_obj_lower;
  double* capacity_threshold;
  bool* is_propagated;
  // getPropagationConstraint
  DSlice<double> obj_vals;
  DSlice<void> clique_data;  // PartitionCliqueData
  DSlice<double> cons_buffer;
};

// Mirror of CDomain
struct Domain {
  const double* feastol;
  const double* epsilon;
  const double* upper_limit;
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
  DSlice<HighsInt> col_lower_pos;
  DSlice<HighsInt> col_upper_pos;
  DSlice<uint8_t> changedcolsflags;
  // std::vector objects
  void* propagateinds;
  void* changedcols;
  void* branchpos;
  void* domchgstack;
  void* domchgreason;
  void* prevboundval;
  void* scratch_inds;
  void* scratch_bounds;
  void* scratch_counts;
  bool* infeasible;
  void* infeasible_reason;  // HighsDomain::Reason*
  HighsInt* infeasible_pos;
  const bool* record_redundant_rows;
  DSlice<CutProp> cutpools;
  DSlice<ConfProp> conflictpools;
  ObjProp objprop;
  void* dom;  // HighsDomain*
  void (*implications)(void*, int, int);
  void (*redundant_row)(void*, int);
  void (*cut_reset_age)(void*, int, int);
  void (*conflict_reset_age)(void*, int, int);
  ReserveFn reserve_i32;
  ReserveFn reserve_domchg;
  ReserveFn reserve_reason;
  ReserveFn reserve_prev;
  ReserveFn reserve_pair;
};

// Mirror of CBounds: what the const methods read
struct Bounds {
  double feastol;
  double epsilon;
  DSlice<double> col_lower;
  DSlice<double> col_upper;
  DSlice<HighsVarType> integrality;
  DSlice<HighsInt> col_lower_pos;
  DSlice<HighsInt> col_upper_pos;
  DSlice<std::pair<double, HighsInt>> prevboundval;
  bool infeasible;
  HighsInt infeasible_pos;
};

// A HighsDomain's view, kept until a vector it points to by data() and
// size() may have moved or resized: HighsDomain resets `valid` when it is
// copied or assigned, when it (re)sizes its row arrays, sets up its
// objective propagation, and when its pools change or add a cut or conflict
// (which resizes their arrays and may move the cut matrix). The vectors that
// grow during propagation are passed as std::vector objects
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
