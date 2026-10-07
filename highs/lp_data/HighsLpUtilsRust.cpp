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

extern "C" void highs_rs_report_presolve_reductions(const RsLog* log, bool on,
                                                    int presolve_status,
                                                    const HighsInt* from,
                                                    const HighsInt* presolved);

void reportPresolveReductions(const HighsLogOptions& log_options,
                              HighsPresolveStatus presolve_status,
                              const HighsLp& lp, const HighsLp& presolved_lp) {
  const RsLog log = rsLog(log_options);
  const HighsInt from[3] = {lp.num_col_, lp.num_row_, lp.a_matrix_.numNz()};
  HighsInt to[3] = {0, 0, 0};
  if (presolve_status == HighsPresolveStatus::kReduced ||
      presolve_status == HighsPresolveStatus::kTimeout) {
    to[0] = presolved_lp.num_col_;
    to[1] = presolved_lp.num_row_;
    to[2] = presolved_lp.a_matrix_.numNz();
  }
  highs_rs_report_presolve_reductions(&log, *log_options.output_flag,
                                      int(presolve_status), from, to);
}

// Model modification internals (rust/src/lp_data/edit.rs)
extern "C" {
void highs_rs_change_values(int which, const RsIndexCollection* ic,
                            RsMut<double> a, RsMut<double> b,
                            RsMut<double> new_a, RsMut<double> new_b);
void highs_rs_change_integrality(const RsIndexCollection* ic,
                                 RsMut<uint8_t> integrality,
                                 RsMut<uint8_t> new_integrality);
void highs_rs_delete_scale(const RsIndexCollection* ic, RsMut<double> scale);
int64_t highs_rs_change_matrix_coefficient(RsMut<HighsInt> start,
                                           RsMut<HighsInt> index,
                                           RsMut<double> value,
                                           HighsInt num_col, HighsInt row,
                                           HighsInt col, double new_value,
                                           bool zero_new_value);
double highs_rs_get_coefficient(RsMut<HighsInt> start, RsMut<HighsInt> index,
                                RsMut<double> value, HighsInt major,
                                HighsInt minor);
void highs_rs_calculate_row_values_quad(RsMut<HighsInt> start,
                                        RsMut<HighsInt> index,
                                        RsMut<double> value,
                                        RsMut<double> col_value,
                                        RsMut<double> row_value);
void highs_rs_calculate_col_duals_quad(RsMut<HighsInt> start,
                                       RsMut<HighsInt> index,
                                       RsMut<double> value, RsMut<double> cost,
                                       RsMut<double> row_dual,
                                       RsMut<double> col_dual);
}

static const RsMut<double> kRsNone = {nullptr, 0};

void deleteScale(vector<double>& scale,
                 const HighsIndexCollection& index_collection) {
  assert(ok(index_collection));
  const RsIndexCollection ic = rsIndexCollection(index_collection);
  highs_rs_delete_scale(&ic, rsMut(scale));
}

void changeLpMatrixCoefficient(HighsLp& lp, const HighsInt row,
                               const HighsInt col, const double new_value,
                               const bool zero_new_value) {
  assert(0 <= row && row < lp.num_row_);
  assert(0 <= col && col < lp.num_col_);
  HighsSparseMatrix& a = lp.a_matrix_;
  // Room for an inserted entry; the C++ only resizes when inserting
  const size_t old_size = a.index_.size();
  const size_t num_nz = a.start_[lp.num_col_];
  a.index_.resize(std::max(old_size, num_nz + 1));
  a.value_.resize(std::max(old_size, num_nz + 1));
  const int64_t new_num_nz = highs_rs_change_matrix_coefficient(
      rsMut(a.start_), rsMut(a.index_), rsMut(a.value_), lp.num_col_, row, col,
      new_value, zero_new_value);
  const size_t size = new_num_nz < 0 ? old_size : size_t(new_num_nz);
  a.index_.resize(size);
  a.value_.resize(size);
}

HighsStatus changeLpIntegrality(HighsLp& lp,
                                const HighsIndexCollection& index_collection,
                                const vector<HighsVarType>& new_integrality,
                                const HighsOptions options) {
  assert(ok(index_collection));
  HighsInt from_k;
  HighsInt to_k;
  limits(index_collection, from_k, to_k);
  if (from_k > to_k) return HighsStatus::kOk;
  if (lp.integrality_.size() == 0)
    lp.integrality_.assign(lp.num_col_, HighsVarType::kContinuous);
  assert(HighsInt(lp.integrality_.size()) == lp.num_col_);
  const RsIndexCollection ic = rsIndexCollection(index_collection);
  highs_rs_change_integrality(&ic, rsMut(lp.integrality_),
                              rsMut(new_integrality));
  if (!lp.isMip()) lp.integrality_.clear();
  return HighsStatus::kOk;
}

void changeLpCosts(HighsLp& lp, const HighsIndexCollection& index_collection,
                   const vector<double>& new_col_cost,
                   const double infinite_cost) {
  assert(ok(index_collection));
  HighsInt from_k;
  HighsInt to_k;
  limits(index_collection, from_k, to_k);
  if (from_k > to_k) return;
  const RsIndexCollection ic = rsIndexCollection(index_collection);
  highs_rs_change_values(0, &ic, rsMut(lp.col_cost_), kRsNone,
                         rsMut(new_col_cost), kRsNone);
  if (lp.has_infinite_cost_)
    lp.has_infinite_cost_ = lp.hasInfiniteCost(infinite_cost);
}

void changeBounds(vector<double>& lower, vector<double>& upper,
                  const HighsIndexCollection& index_collection,
                  const vector<double>& new_lower,
                  const vector<double>& new_upper) {
  assert(ok(index_collection));
  const RsIndexCollection ic = rsIndexCollection(index_collection);
  highs_rs_change_values(1, &ic, rsMut(lower), rsMut(upper), rsMut(new_lower),
                         rsMut(new_upper));
}

void getLpMatrixCoefficient(const HighsLp& lp, const HighsInt Xrow,
                            const HighsInt Xcol, double* val) {
  assert(0 <= Xrow && Xrow < lp.num_row_);
  assert(0 <= Xcol && Xcol < lp.num_col_);
  const HighsSparseMatrix& a = lp.a_matrix_;
  *val = highs_rs_get_coefficient(rsMut(a.start_), rsMut(a.index_),
                                  rsMut(a.value_), Xcol, Xrow);
}

HighsStatus calculateColDualsQuad(const HighsLp& lp, HighsSolution& solution) {
  const bool correct_size = int(solution.row_dual.size()) == lp.num_row_;
  const bool is_colwise = lp.a_matrix_.isColwise();
  const bool data_error = !correct_size || !is_colwise;
  assert(!data_error);
  if (data_error) return HighsStatus::kError;
  solution.col_dual.resize(lp.num_col_);
  const HighsSparseMatrix& a = lp.a_matrix_;
  highs_rs_calculate_col_duals_quad(
      rsMut(a.start_), rsMut(a.index_), rsMut(a.value_), rsMut(lp.col_cost_),
      rsMut(solution.row_dual), rsMut(solution.col_dual));
  return HighsStatus::kOk;
}

void highsRsCalculateRowValuesQuad(const HighsLp& lp,
                                   const std::vector<double>& col_value,
                                   std::vector<double>& row_value) {
  const HighsSparseMatrix& a = lp.a_matrix_;
  highs_rs_calculate_row_values_quad(rsMut(a.start_), rsMut(a.index_),
                                     rsMut(a.value_), rsMut(col_value),
                                     rsMut(row_value));
}
#endif
