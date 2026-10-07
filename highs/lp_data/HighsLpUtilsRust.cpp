/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file lp_data/HighsLpUtilsRust.cpp
 * @brief The views of lp_data passed to Rust, and the LP validation and
 * scaling of HighsLpUtils.cpp and HighsMatrixUtils.cpp done by Rust
 * (rust/src/lp_data/lp_utils.rs)
 */
#include "lp_data/HighsRust.h"

#ifdef HIGHS_RUST
#include <string>

#include "io/HighsIO.h"
#include "lp_data/HighsLpUtils.h"
#include "util/HighsMatrixUtils.h"

static_assert(sizeof(HighsInt) == 4, "Rust's lp_data uses 32-bit ints");

static void rsLogFn(const void* opts, int dev, int type, const char* msg,
                    size_t len) {
  const std::string s(msg, len);
  const HighsLogOptions& log_options =
      *static_cast<const HighsLogOptions*>(opts);
  if (dev)
    highsLogDev(log_options, HighsLogType(type), "%s", s.c_str());
  else
    highsLogUser(log_options, HighsLogType(type), "%s", s.c_str());
}

RsLog rsLog(const HighsLogOptions& log_options) {
  return {&log_options, rsLogFn};
}

RsLp rsLp(const HighsLp& lp) {
  RsLp v;
  v.num_col = lp.num_col_;
  v.num_row = lp.num_row_;
  v.col_cost = rsMut(lp.col_cost_);
  v.col_lower = rsMut(lp.col_lower_);
  v.col_upper = rsMut(lp.col_upper_);
  v.row_lower = rsMut(lp.row_lower_);
  v.row_upper = rsMut(lp.row_upper_);
  const HighsSparseMatrix& a = lp.a_matrix_;
  v.a = {int(a.format_),   a.num_col_,        a.num_row_,
         rsMut(a.start_),  rsMut(a.p_end_),   rsMut(a.index_),
         rsMut(a.value_)};
  v.sense = int(lp.sense_);
  v.offset = lp.offset_;
  v.integrality = rsMut(lp.integrality_);
  v.scale_strategy = lp.scale_.strategy;
  v.scale_has_scaling = lp.scale_.has_scaling;
  v.scale_num_col = lp.scale_.num_col;
  v.scale_num_row = lp.scale_.num_row;
  v.scale_cost = lp.scale_.cost;
  v.scale_col = rsMut(lp.scale_.col);
  v.scale_row = rsMut(lp.scale_.row);
  v.is_scaled = lp.is_scaled_;
  v.is_moved = lp.is_moved_;
  v.has_infinite_cost = lp.has_infinite_cost_;
  return v;
}

void rsLpBack(const RsLp& v, HighsLp& lp) {
  lp.scale_.strategy = v.scale_strategy;
  lp.scale_.has_scaling = v.scale_has_scaling;
  lp.scale_.num_col = v.scale_num_col;
  lp.scale_.num_row = v.scale_num_row;
  lp.scale_.cost = v.scale_cost;
  lp.is_scaled_ = v.is_scaled;
  lp.has_infinite_cost_ = v.has_infinite_cost;
}

RsIndexCollection rsIndexCollection(const HighsIndexCollection& ic) {
  return {ic.dimension_,       ic.is_interval_, ic.from_,
          ic.to_,              ic.is_set_,      ic.set_num_entries_,
          rsMut(ic.set_),      ic.is_mask_,     rsMut(ic.mask_)};
}

RsLpOptions rsLpOptions(const HighsOptions& options) {
  RsLpOptions o;
  o.log = rsLog(options.log_options);
  o.infinite_cost = options.infinite_cost;
  o.infinite_bound = options.infinite_bound;
  o.small_matrix_value = options.small_matrix_value;
  o.large_matrix_value = options.large_matrix_value;
  o.simplex_scale_strategy = options.simplex_scale_strategy;
  o.allowed_matrix_scale_factor = options.allowed_matrix_scale_factor;
  o.highs_analysis_level = options.highs_analysis_level;
  o.log_dev_level = options.log_dev_level;
  o.primal_feasibility_tolerance = options.primal_feasibility_tolerance;
  return o;
}

HighsStatus assessLp(HighsLp& lp, const HighsOptions& options) {
  RsLp v = rsLp(lp);
  const RsLpOptions o = rsLpOptions(options);
  const HighsStatus return_status = HighsStatus(highs_rs_assess_lp(&v, &o));
  rsLpBack(v, lp);
  if (return_status != HighsStatus::kError && lp.num_col_ > 0) {
    // Entries may have been removed from the matrix
    const HighsInt lp_num_nz = lp.a_matrix_.numNz();
    if ((HighsInt)lp.a_matrix_.index_.size() > lp_num_nz)
      lp.a_matrix_.index_.resize(lp_num_nz);
    if ((HighsInt)lp.a_matrix_.value_.size() > lp_num_nz)
      lp.a_matrix_.value_.resize(lp_num_nz);
  }
  return return_status;
}

bool lpDimensionsOk(const std::string& message, const HighsLp& lp,
                    const HighsLogOptions& log_options) {
  const RsLog log = rsLog(log_options);
  const RsLp v = rsLp(lp);
  return highs_rs_lp_dimensions_ok(&log, message.data(), message.size(), &v);
}

HighsStatus assessCosts(const HighsOptions& options, const HighsInt ml_col_os,
                        const HighsIndexCollection& index_collection,
                        std::vector<double>& cost, bool& has_infinite_cost,
                        const double infinite_cost) {
  assert(ok(index_collection));
  const RsLpOptions o = rsLpOptions(options);
  const RsIndexCollection ic = rsIndexCollection(index_collection);
  return HighsStatus(highs_rs_assess_costs(&o, ml_col_os, &ic, rsMut(cost),
                                           &has_infinite_cost, infinite_cost));
}

HighsStatus assessBounds(const HighsOptions& options, const char* type,
                         const HighsInt ml_ix_os,
                         const HighsIndexCollection& index_collection,
                         std::vector<double>& lower,
                         std::vector<double>& upper,
                         const double infinite_bound,
                         const HighsVarType* integrality) {
  assert(ok(index_collection));
  const RsLpOptions o = rsLpOptions(options);
  const RsIndexCollection ic = rsIndexCollection(index_collection);
  // The integrality is indexed like lower and upper
  RsMut<uint8_t> rs_integrality = {
      reinterpret_cast<uint8_t*>(const_cast<HighsVarType*>(integrality)),
      integrality ? lower.size() : 0};
  const std::string kind(type);
  return HighsStatus(highs_rs_assess_bounds(
      &o, kind.data(), kind.size(), ml_ix_os, &ic, rsMut(lower), rsMut(upper),
      infinite_bound, rs_integrality));
}

HighsStatus cleanBounds(const HighsOptions& options, HighsLp& lp) {
  RsLp v = rsLp(lp);
  const RsLpOptions o = rsLpOptions(options);
  return HighsStatus(highs_rs_clean_bounds(&v, &o));
}

void scaleLp(const HighsOptions& options, HighsLp& lp,
             const bool force_scaling) {
  lp.clearScaling();
  // Scaling not well defined for models with no columns
  assert(lp.num_col_ > 0);
  lp.scale_.col.assign(lp.num_col_, 1);
  lp.scale_.row.assign(lp.num_row_, 1);
  RsLp v = rsLp(lp);
  const RsLpOptions o = rsLpOptions(options);
  const bool scaled = highs_rs_scale_lp(&v, &o, force_scaling);
  rsLpBack(v, lp);
  if (!scaled) {
    const HighsInt strategy = lp.scale_.strategy;
    lp.clearScale();
    lp.scale_.strategy = strategy;
  }
}

void HighsLp::applyScale() {
  RsLp v = rsLp(*this);
  highs_rs_lp_apply_scale(&v, true);
  rsLpBack(v, *this);
}

void HighsLp::unapplyScale() {
  RsLp v = rsLp(*this);
  highs_rs_lp_apply_scale(&v, false);
  rsLpBack(v, *this);
}

HighsStatus assessMatrix(
    const HighsLogOptions& log_options, const std::string& matrix_name,
    const HighsInt vec_dim, const HighsInt num_vec, const bool partitioned,
    std::vector<HighsInt>& matrix_start, std::vector<HighsInt>& matrix_p_end,
    std::vector<HighsInt>& matrix_index, std::vector<double>& matrix_value,
    const double small_matrix_value, const double large_matrix_value,
    const bool sum_duplicates) {
  const RsLog log = rsLog(log_options);
  return HighsStatus(highs_rs_assess_matrix(
      &log, matrix_name.data(), matrix_name.size(), vec_dim, num_vec,
      partitioned, rsMut(matrix_start), rsMut(matrix_p_end),
      rsMut(matrix_index), rsMut(matrix_value), small_matrix_value,
      large_matrix_value, sum_duplicates));
}

HighsStatus assessMatrixDimensions(const HighsLogOptions& log_options,
                                   const HighsInt num_vec,
                                   const bool partitioned,
                                   const std::vector<HighsInt>& matrix_start,
                                   const std::vector<HighsInt>& matrix_p_end,
                                   const std::vector<HighsInt>& matrix_index,
                                   const std::vector<double>& matrix_value) {
  const RsLog log = rsLog(log_options);
  return HighsStatus(highs_rs_assess_matrix_dimensions(
      &log, num_vec, partitioned, rsMut(matrix_start), rsMut(matrix_p_end),
      matrix_index.size(), matrix_value.size()));
}
#endif
