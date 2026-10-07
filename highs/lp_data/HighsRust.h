/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file lp_data/HighsRust.h
 * @brief The views of HiGHS's top-level data passed to the Rust port of
 * lp_data (rust/src/lp_data/ffi.rs mirrors these repr(C) structs)
 */
#ifndef LP_DATA_HIGHS_RUST_H_
#define LP_DATA_HIGHS_RUST_H_

#include "HConfig.h"

#ifdef HIGHS_RUST
#include <cstddef>
#include <vector>

#include "lp_data/HStruct.h"
#include "lp_data/HighsLp.h"
#include "lp_data/HighsOptions.h"
#include "util/HighsUtils.h"

// A C++ array (a vector's data() and size()) seen by Rust
template <typename T>
struct RsMut {
  T* ptr;
  size_t len;
};
template <typename T>
RsMut<T> rsMut(std::vector<T>& v) {
  return {v.data(), v.size()};
}
// Rust only reads arrays passed from a const vector
template <typename T>
RsMut<T> rsMut(const std::vector<T>& v) {
  return {const_cast<T*>(v.data()), v.size()};
}
inline RsMut<uint8_t> rsMut(const std::vector<HighsVarType>& v) {
  static_assert(sizeof(HighsVarType) == 1, "HighsVarType is a byte");
  return {reinterpret_cast<uint8_t*>(const_cast<HighsVarType*>(v.data())),
          v.size()};
}
inline RsMut<uint8_t> rsMut(std::vector<HighsVarType>& v) {
  return rsMut(static_cast<const std::vector<HighsVarType>&>(v));
}

// HighsLogOptions and the function through which Rust logs
struct RsLog {
  const void* opts;
  void (*log)(const void* opts, int dev, int type, const char* msg,
              size_t len);
};
RsLog rsLog(const HighsLogOptions& log_options);

// HighsSolution and HighsBasis
struct RsSolution {
  bool value_valid, dual_valid;
  RsMut<double> col_value, col_dual, row_value, row_dual;
};
inline RsSolution rsSolution(const HighsSolution& s) {
  return {s.value_valid,      s.dual_valid,       rsMut(s.col_value),
          rsMut(s.col_dual),  rsMut(s.row_value), rsMut(s.row_dual)};
}
struct RsBasis {
  bool valid;
  RsMut<uint8_t> col_status, row_status;
};
inline RsMut<uint8_t> rsMut(const std::vector<HighsBasisStatus>& v) {
  static_assert(sizeof(HighsBasisStatus) == 1, "HighsBasisStatus is a byte");
  return {reinterpret_cast<uint8_t*>(const_cast<HighsBasisStatus*>(v.data())),
          v.size()};
}
inline RsBasis rsBasis(const HighsBasis& b) {
  return {b.valid, rsMut(b.col_status), rsMut(b.row_status)};
}

struct RsMatrix {
  int format, num_col, num_row;
  RsMut<HighsInt> start, p_end, index;
  RsMut<double> value;
};

// HighsLp's numerical data and scaling
struct RsLp {
  HighsInt num_col, num_row;
  RsMut<double> col_cost, col_lower, col_upper, row_lower, row_upper;
  RsMatrix a;
  int sense;
  double offset;
  RsMut<uint8_t> integrality;
  HighsInt scale_strategy;
  bool scale_has_scaling;
  HighsInt scale_num_col, scale_num_row;
  double scale_cost;
  RsMut<double> scale_col, scale_row;
  bool is_scaled, is_moved, has_infinite_cost;
};
RsLp rsLp(const HighsLp& lp);
// Copy back the scalars Rust may change
void rsLpBack(const RsLp& v, HighsLp& lp);

struct RsIndexCollection {
  HighsInt dimension;
  bool is_interval;
  HighsInt from, to;
  bool is_set;
  HighsInt set_num_entries;
  RsMut<HighsInt> set;
  bool is_mask;
  RsMut<HighsInt> mask;
};
RsIndexCollection rsIndexCollection(const HighsIndexCollection& ic);

// calculateRowValuesQuad without its debugging report
void highsRsCalculateRowValuesQuad(const HighsLp& lp,
                                   const std::vector<double>& col_value,
                                   std::vector<double>& row_value);

// The options of LP validation and scaling
struct RsLpOptions {
  RsLog log;
  double infinite_cost, infinite_bound, small_matrix_value,
      large_matrix_value;
  HighsInt simplex_scale_strategy, allowed_matrix_scale_factor,
      highs_analysis_level, log_dev_level;
  double primal_feasibility_tolerance;
};
RsLpOptions rsLpOptions(const HighsOptions& options);

extern "C" {
int highs_rs_assess_lp(RsLp* lp, const RsLpOptions* o);
int highs_rs_assess_costs(const RsLpOptions* o, HighsInt ml_col_os,
                          const RsIndexCollection* ic, RsMut<double> cost,
                          bool* has_infinite_cost, double infinite_cost);
int highs_rs_assess_bounds(const RsLpOptions* o, const char* kind,
                           size_t kind_len, HighsInt ml_ix_os,
                           const RsIndexCollection* ic, RsMut<double> lower,
                           RsMut<double> upper, double infinite_bound,
                           RsMut<uint8_t> integrality);
int highs_rs_assess_matrix(const RsLog* log, const char* name,
                           size_t name_len, HighsInt vec_dim, HighsInt num_vec,
                           bool partitioned, RsMut<HighsInt> start,
                           RsMut<HighsInt> p_end, RsMut<HighsInt> index,
                           RsMut<double> value, double small_matrix_value,
                           double large_matrix_value, bool sum_duplicates);
bool highs_rs_scale_lp(RsLp* lp, const RsLpOptions* o, bool force_scaling);
void highs_rs_lp_apply_scale(RsLp* lp, bool apply);
int highs_rs_clean_bounds(RsLp* lp, const RsLpOptions* o);
bool highs_rs_lp_dimensions_ok(const RsLog* log, const char* message,
                               size_t message_len, const RsLp* lp);
int highs_rs_assess_matrix_dimensions(const RsLog* log, HighsInt num_vec,
                                      bool partitioned, RsMut<HighsInt> start,
                                      RsMut<HighsInt> p_end, size_t index_size,
                                      size_t value_size);
}

#endif
#endif
