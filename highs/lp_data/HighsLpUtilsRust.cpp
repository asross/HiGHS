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
#include <cstddef>
#include <string>

#include "io/HighsIO.h"
#include "lp_data/HighsLpUtils.h"
#include "util/HighsMatrixUtils.h"

static_assert(sizeof(HighsInt) == 4, "Rust's lp_data uses 32-bit ints");

// The Rust log sink (rust/src/io/log.rs), called by Rust directly
extern "C" void highs_rs_log(const void* opts, int dev, int type,
                             const char* msg, size_t len);

RsLog rsLog(const HighsLogOptions& log_options) {
  return {&log_options, highs_rs_log};
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

static_assert(sizeof(RsMatVec) == 144, "Mat<RsVec, RsVec> in sparse.rs");
static_assert(sizeof(RsScaleVec) == 88, "ScaleG<RsVec> in lp.rs");
static_assert(sizeof(RsLpVec) == 456, "CppLp in lp.rs");

static_assert(sizeof(RsBasisVec) == 80, "BasisG<RsVec> in interface.rs");

RsBasisVec rsBasisVec(HighsBasis& b) {
  return {b.valid,
          b.alien,
          b.useful,
          b.was_alien,
          b.debug_id,
          b.debug_update_count,
          rsByteVec(b.col_status),
          rsByteVec(b.row_status)};
}

void rsBasisVecBack(const RsBasisVec& v, HighsBasis& b) {
  b.valid = v.valid;
  b.alien = v.alien;
  b.useful = v.useful;
  b.was_alien = v.was_alien;
  b.debug_id = v.debug_id;
  b.debug_update_count = v.debug_update_count;
}

RsMatVec rsMatVec(HighsSparseMatrix& a) {
  return {int(a.format_),   a.num_col_,       a.num_row_,
          rsVec(a.start_),  rsVec(a.p_end_),  rsVec(a.index_),
          rsVec(a.value_)};
}

void rsMatVecBack(const RsMatVec& v, HighsSparseMatrix& a) {
  a.format_ = MatrixFormat(v.format);
  a.num_col_ = v.num_col;
  a.num_row_ = v.num_row;
}

RsLpVec rsLpVec(HighsLp& lp) {
  RsLpVec v;
  v.num_col = lp.num_col_;
  v.num_row = lp.num_row_;
  v.col_cost = rsVec(lp.col_cost_);
  v.col_lower = rsVec(lp.col_lower_);
  v.col_upper = rsVec(lp.col_upper_);
  v.row_lower = rsVec(lp.row_lower_);
  v.row_upper = rsVec(lp.row_upper_);
  v.a = rsMatVec(lp.a_matrix_);
  v.sense = int(lp.sense_);
  v.offset = lp.offset_;
  v.integrality = rsByteVec(lp.integrality_);
  HighsScale& s = lp.scale_;
  v.scale = {s.strategy, s.has_scaling, s.num_col,     s.num_row,
             s.cost,     rsVec(s.col),  rsVec(s.row)};
  v.is_scaled = lp.is_scaled_;
  v.is_moved = lp.is_moved_;
  v.has_infinite_cost = lp.has_infinite_cost_;
  return v;
}

void rsLpVecBack(const RsLpVec& v, HighsLp& lp) {
  lp.num_col_ = v.num_col;
  lp.num_row_ = v.num_row;
  rsMatVecBack(v.a, lp.a_matrix_);
  lp.sense_ = ObjSense(v.sense);
  lp.offset_ = v.offset;
  HighsScale& s = lp.scale_;
  s.strategy = v.scale.strategy;
  s.has_scaling = v.scale.has_scaling;
  s.num_col = v.scale.num_col;
  s.num_row = v.scale.num_row;
  s.cost = v.scale.cost;
  lp.is_scaled_ = v.is_scaled;
  lp.is_moved_ = v.is_moved;
  lp.has_infinite_cost_ = v.has_infinite_cost;
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
  rsOptionsTemplate(options, 2, &o);
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

HighsStatus cleanBounds(const HighsOptions& options, HighsLp& lp) {
  RsLp v = rsLp(lp);
  const RsLpOptions o = rsLpOptions(options);
  return HighsStatus(highs_rs_clean_bounds(&v, &o));
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

// User objective and bound scaling (rust/src/lp_data/user_scale.rs)
static_assert(sizeof(HighsUserScaleData) == 80, "HighsUserScaleData layout");
static_assert(offsetof(HighsUserScaleData, applied) == 72,
              "HighsUserScaleData layout");

extern "C" {
void highs_rs_user_scale_lp(const RsLp* lp, HighsUserScaleData* d,
                            bool apply);
int highs_rs_user_scale_status(const RsLog* log, const HighsUserScaleData* d);
bool highs_rs_user_scale_message(const HighsUserScaleData* d, int which,
                                 void* ctx,
                                 void (*set)(void*, const char*, size_t));
}

void userScaleLp(HighsLp& lp, HighsUserScaleData& data, const bool apply) {
  const RsLp v = rsLp(lp);
  highs_rs_user_scale_lp(&v, &data, apply);
}

HighsStatus userScaleStatus(const HighsLogOptions& log_options,
                            const HighsUserScaleData& data) {
  const RsLog log = rsLog(log_options);
  return HighsStatus(highs_rs_user_scale_status(&log, &data));
}

static void setUserScaleMessage(void* ctx, const char* p, size_t n) {
  static_cast<std::string*>(ctx)->assign(p, n);
}

bool HighsUserScaleData::scaleError(std::string& message) const {
  return highs_rs_user_scale_message(this, 0, &message, setUserScaleMessage);
}

bool HighsUserScaleData::scaleWarning(std::string& message) const {
  return highs_rs_user_scale_message(this, 1, &message, setUserScaleMessage);
}

// Semi-variables and the unapplying of model modifications
// (rust/src/lp_data/semi.rs)
struct RsSemiMods {
  RsMut<HighsInt> inconsistent_index;
  RsMut<double> inconsistent_lower, inconsistent_upper;
  RsMut<uint8_t> inconsistent_type;
  RsMut<HighsInt> non_semi_index, tightened_index;
  RsMut<double> tightened_value;
  HighsInt num_inconsistent, num_non_semi, num_tightened;
  bool made_mods;
};
struct RsLpMods {
  RsMut<HighsInt> non_semi_index, inconsistent_index;
  RsMut<double> inconsistent_lower, inconsistent_upper;
  RsMut<uint8_t> inconsistent_type;
  RsMut<HighsInt> relaxed_index;
  RsMut<double> relaxed_value;
  RsMut<HighsInt> tightened_index;
  RsMut<double> tightened_value;
};

extern "C" {
int highs_rs_assess_semi_variables(const RsLog* log, RsMut<double> col_lower,
                                   RsMut<double> col_upper,
                                   RsMut<uint8_t> integrality, RsSemiMods* m);
size_t highs_rs_relax_semi_variables(RsMut<double> col_lower,
                                     RsMut<uint8_t> integrality,
                                     RsMut<HighsInt> index,
                                     RsMut<double> value);
bool highs_rs_active_modified_upper_bounds(const RsLog* log,
                                           RsMut<HighsInt> tightened_index,
                                           RsMut<double> col_upper,
                                           RsMut<double> col_value,
                                           double pft);
void highs_rs_unapply_mods(const RsLpMods* m, RsMut<double> col_lower,
                           RsMut<double> col_upper,
                           RsMut<uint8_t> integrality);
}

// Appends the first n entries of `from` to `to`, or clears `to` if n < 0
template <typename T>
static void appendMods(std::vector<T>& to, const std::vector<T>& from,
                       const HighsInt n) {
  if (n < 0) {
    to.clear();
  } else {
    to.insert(to.end(), from.begin(), from.begin() + n);
  }
}

HighsStatus assessSemiVariables(HighsLp& lp, const HighsOptions& options,
                                bool& made_semi_variable_mods) {
  made_semi_variable_mods = false;
  if (!lp.integrality_.size()) return HighsStatus::kOk;
  assert((HighsInt)lp.integrality_.size() == lp.num_col_);
  assert(int(lp.mods_.save_inconsistent_semi_variable_index.size()) == 0);
  const size_t n = lp.num_col_;
  std::vector<HighsInt> inconsistent_index(n), non_semi_index(n),
      tightened_index(n);
  std::vector<double> inconsistent_lower(n), inconsistent_upper(n),
      tightened_value(n);
  std::vector<HighsVarType> inconsistent_type(n);
  RsSemiMods m;
  m.inconsistent_index = rsMut(inconsistent_index);
  m.inconsistent_lower = rsMut(inconsistent_lower);
  m.inconsistent_upper = rsMut(inconsistent_upper);
  m.inconsistent_type = rsMut(inconsistent_type);
  m.non_semi_index = rsMut(non_semi_index);
  m.tightened_index = rsMut(tightened_index);
  m.tightened_value = rsMut(tightened_value);
  const RsLog log = rsLog(options.log_options);
  const HighsStatus return_status = HighsStatus(highs_rs_assess_semi_variables(
      &log, rsMut(lp.col_lower_), rsMut(lp.col_upper_), rsMut(lp.integrality_),
      &m));
  HighsLpMods& mods = lp.mods_;
  appendMods(mods.save_non_semi_variable_index, non_semi_index, m.num_non_semi);
  appendMods(mods.save_inconsistent_semi_variable_index, inconsistent_index,
             m.num_inconsistent);
  appendMods(mods.save_inconsistent_semi_variable_lower_bound_value,
             inconsistent_lower, m.num_inconsistent);
  appendMods(mods.save_inconsistent_semi_variable_upper_bound_value,
             inconsistent_upper, m.num_inconsistent);
  appendMods(mods.save_inconsistent_semi_variable_type, inconsistent_type,
             m.num_inconsistent);
  appendMods(mods.save_tightened_semi_variable_upper_bound_index,
             tightened_index, m.num_tightened);
  appendMods(mods.save_tightened_semi_variable_upper_bound_value,
             tightened_value, m.num_tightened);
  made_semi_variable_mods = m.made_mods;
  return return_status;
}

void relaxSemiVariables(HighsLp& lp, bool& made_semi_variable_mods) {
  made_semi_variable_mods = false;
  if (!lp.integrality_.size()) return;
  assert((HighsInt)lp.integrality_.size() == lp.num_col_);
  std::vector<HighsInt>& index =
      lp.mods_.save_relaxed_semi_variable_lower_bound_index;
  std::vector<double>& value =
      lp.mods_.save_relaxed_semi_variable_lower_bound_value;
  assert(index.size() == 0);
  std::vector<HighsInt> new_index(lp.num_col_);
  std::vector<double> new_value(lp.num_col_);
  const HighsInt num_relaxed = highs_rs_relax_semi_variables(
      rsMut(lp.col_lower_), rsMut(lp.integrality_), rsMut(new_index),
      rsMut(new_value));
  appendMods(index, new_index, num_relaxed);
  appendMods(value, new_value, num_relaxed);
  made_semi_variable_mods = index.size() > 0;
}

bool activeModifiedUpperBounds(const HighsOptions& options, const HighsLp& lp,
                               const std::vector<double>& col_value) {
  const RsLog log = rsLog(options.log_options);
  return highs_rs_active_modified_upper_bounds(
      &log, rsMut(lp.mods_.save_tightened_semi_variable_upper_bound_index),
      rsMut(lp.col_upper_), rsMut(col_value),
      options.primal_feasibility_tolerance);
}

void HighsLp::unapplyMods() {
  const HighsLpMods& mods = this->mods_;
  const RsLpMods m = {
      rsMut(mods.save_non_semi_variable_index),
      rsMut(mods.save_inconsistent_semi_variable_index),
      rsMut(mods.save_inconsistent_semi_variable_lower_bound_value),
      rsMut(mods.save_inconsistent_semi_variable_upper_bound_value),
      rsMut(mods.save_inconsistent_semi_variable_type),
      rsMut(mods.save_relaxed_semi_variable_lower_bound_index),
      rsMut(mods.save_relaxed_semi_variable_lower_bound_value),
      rsMut(mods.save_tightened_semi_variable_upper_bound_index),
      rsMut(mods.save_tightened_semi_variable_upper_bound_value)};
  highs_rs_unapply_mods(&m, rsMut(this->col_lower_), rsMut(this->col_upper_),
                        rsMut(this->integrality_));
  this->mods_.clear();
}

// getSubVectors and getSubVectorsTranspose (rust/src/lp_data/edit.rs)
extern "C" void highs_rs_get_sub_vectors(
    bool transpose, const RsIndexCollection* ic, HighsInt data_dim,
    const double* const* data,
    RsMut<HighsInt> m_start, RsMut<HighsInt> m_index, RsMut<double> m_value,
    double* const* out_data, HighsInt* out_start, HighsInt* out_index,
    double* out_value, HighsInt* num_sub_vector, HighsInt* num_nz);

static void rsGetSubVectors(
    const bool transpose, const HighsIndexCollection& index_collection,
    const HighsInt data_dim, const double* data0, const double* data1, const double* data2,
    const HighsSparseMatrix& matrix, HighsInt& num_sub_vector,
    double* sub_vector_data0, double* sub_vector_data1,
    double* sub_vector_data2, HighsInt& sub_matrix_num_nz,
    HighsInt* sub_matrix_start, HighsInt* sub_matrix_index,
    double* sub_matrix_value) {
  if (data0 == nullptr) assert(sub_vector_data0 == nullptr);
  assert(ok(index_collection));
  const RsIndexCollection ic = rsIndexCollection(index_collection);
  const double* data[3] = {data0, data1, data2};
  double* out[3] = {sub_vector_data0, sub_vector_data1, sub_vector_data2};
  highs_rs_get_sub_vectors(transpose, &ic, data_dim, data, rsMut(matrix.start_),
                           rsMut(matrix.index_), rsMut(matrix.value_), out,
                           sub_matrix_start, sub_matrix_index,
                           sub_matrix_value, &num_sub_vector,
                           &sub_matrix_num_nz);
}

void getSubVectors(const HighsIndexCollection& index_collection,
                   const HighsInt data_dim, const double* data0,
                   const double* data1, const double* data2,
                   const HighsSparseMatrix& matrix, HighsInt& num_sub_vector,
                   double* sub_vector_data0, double* sub_vector_data1,
                   double* sub_vector_data2, HighsInt& sub_matrix_num_nz,
                   HighsInt* sub_matrix_start, HighsInt* sub_matrix_index,
                   double* sub_matrix_value) {
  rsGetSubVectors(false, index_collection, data_dim, data0, data1, data2, matrix,
                  num_sub_vector, sub_vector_data0, sub_vector_data1,
                  sub_vector_data2, sub_matrix_num_nz, sub_matrix_start,
                  sub_matrix_index, sub_matrix_value);
}

void getSubVectorsTranspose(const HighsIndexCollection& index_collection,
                            const HighsInt data_dim, const double* data0,
                            const double* data1, const double* data2,
                            const HighsSparseMatrix& matrix,
                            HighsInt& num_sub_vector, double* sub_vector_data0,
                            double* sub_vector_data1, double* sub_vector_data2,
                            HighsInt& sub_matrix_num_nz,
                            HighsInt* sub_matrix_start,
                            HighsInt* sub_matrix_index,
                            double* sub_matrix_value) {
  rsGetSubVectors(true, index_collection, data_dim, data0, data1, data2, matrix,
                  num_sub_vector, sub_vector_data0, sub_vector_data1,
                  sub_vector_data2, sub_matrix_num_nz, sub_matrix_start,
                  sub_matrix_index, sub_matrix_value);
}

// Solution and basis file reading (rust/src/lp_data/readers.rs)
namespace {
// readers.rs: CNames
struct RsNames {
  void* ctx;
  int (*op)(void* ctx, int code, const char* name, size_t len);
  bool have_col, have_row;
};
// readers.rs: CRead
struct RsRead {
  RsLog log;
  RsNames names;
  RsMut<uint8_t> filename;
  bool* basis_valid;
  RsMut<uint8_t> col_status, row_status;
};
// readers.rs: CReadSolution
struct RsReadSolution {
  bool style_sparse, colwise;
  RsMut<HighsInt> a_start, a_index;
  RsMut<double> a_value;
  bool* value_valid;
  RsMut<double> col_value, col_dual, row_value, row_dual;
};

// Forms the name hashes there are names for (op 0), or looks up a column
// (1) or row (2) name: its index, kHashIsDuplicate, or -2 if not found
int rsNamesOp(void* ctx, int code, const char* name, size_t len) {
  HighsLp& lp = *static_cast<HighsLp*>(ctx);
  const bool have_col_names =
      lp.col_names_.size() == static_cast<size_t>(lp.num_col_);
  const bool have_row_names =
      lp.row_names_.size() == static_cast<size_t>(lp.num_row_);
  if (code == 0) {
    if (have_col_names && !lp.col_hash_.name2index.size())
      lp.col_hash_.form(lp.col_names_);
    if (have_row_names && !lp.row_hash_.name2index.size())
      lp.row_hash_.form(lp.row_names_);
    return 0;
  }
  const auto& name2index =
      code == 1 ? lp.col_hash_.name2index : lp.row_hash_.name2index;
  auto search = name2index.find(std::string(name, len));
  return search == name2index.end() ? -2 : search->second;
}

RsRead rsRead(const HighsLogOptions& log_options, HighsLp& lp,
              HighsBasis& basis, const std::string& filename) {
  return {rsLog(log_options),
          {&lp, rsNamesOp,
           lp.col_names_.size() == static_cast<size_t>(lp.num_col_),
           lp.row_names_.size() == static_cast<size_t>(lp.num_row_)},
          {reinterpret_cast<uint8_t*>(const_cast<char*>(filename.data())),
           filename.size()},
          &basis.valid,
          rsMut(static_cast<const std::vector<HighsBasisStatus>&>(
              basis.col_status)),
          rsMut(static_cast<const std::vector<HighsBasisStatus>&>(
              basis.row_status))};
}
}  // namespace

extern "C" {
int highs_rs_read_basis_file(const RsRead* r);
int highs_rs_read_solution_file(const RsRead* r, const RsReadSolution* s);
}

HighsStatus readBasisFile(const HighsLogOptions& log_options, HighsLp& lp,
                          HighsBasis& basis, const std::string& filename) {
  const RsRead r = rsRead(log_options, lp, basis, filename);
  return HighsStatus(highs_rs_read_basis_file(&r));
}

HighsStatus readSolutionFile(const std::string& filename,
                             const HighsOptions& options, HighsLp& lp,
                             HighsBasis& basis, HighsSolution& solution,
                             const HighsInt style) {
  const HighsLogOptions& log_options = options.log_options;
  if (style != kSolutionStyleRaw && style != kSolutionStyleSparse) {
    highsLogUser(log_options, HighsLogType::kError,
                 "readSolutionFile: Cannot read file of style %d\n",
                 (int)style);
    return HighsStatus::kError;
  }
  HighsSolution read_solution = solution;
  HighsBasis read_basis = basis;
  read_solution.clear();
  read_basis.clear();
  read_solution.col_value.resize(lp.num_col_);
  read_solution.row_value.resize(lp.num_row_);
  read_solution.col_dual.resize(lp.num_col_);
  read_solution.row_dual.resize(lp.num_row_);
  read_basis.col_status.resize(lp.num_col_);
  read_basis.row_status.resize(lp.num_row_);
  const RsRead r = rsRead(log_options, lp, read_basis, filename);
  const HighsSparseMatrix& a = lp.a_matrix_;
  const RsReadSolution s = {style == kSolutionStyleSparse,
                            a.isColwise(),
                            rsMut(a.start_),
                            rsMut(a.index_),
                            rsMut(a.value_),
                            &read_solution.value_valid,
                            rsMut(read_solution.col_value),
                            rsMut(read_solution.col_dual),
                            rsMut(read_solution.row_value),
                            rsMut(read_solution.row_dual)};
  // -2: readSolutionFileErrorReturn; otherwise readSolutionFileReturn's
  // status, the read solution and basis taken when kOk
  const int status = highs_rs_read_solution_file(&r, &s);
  if (status == -2) return HighsStatus::kError;
  if (HighsStatus(status) != HighsStatus::kOk) return HighsStatus(status);
  solution = read_solution;
  basis = read_basis;
  return HighsStatus::kOk;
}
#endif
