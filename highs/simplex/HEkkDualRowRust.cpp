/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file simplex/HEkkDualRowRust.cpp
 * @brief HEkkDualRow delegating to the Rust port
 * (rust/src/simplex/dual_row.rs)
 */
#include "simplex/HEkkDualRow.h"

#ifdef HIGHS_RUST

#include <cassert>

#include "simplex/HSimplexDebug.h"
#include "simplex/SimplexTimer.h"

static_assert(sizeof(HighsInt) == 4, "the Rust CHUZC takes 32-bit ints");
static_assert(sizeof(std::pair<HighsInt, double>) == 16,
              "workData is shared with Rust as #[repr(C)] (i32, f64)");

namespace {
// Mirrors of the #[repr(C)] structs in rust/src/simplex/dual_row.rs
struct RsAMatrix {
  int num_col;
  const int* start;
  const int* index;
  const double* value;
};

struct RsArrays {
  int* pack_index;
  double* pack_value;
  std::pair<HighsInt, double>* work_data;
};

RsAMatrix rsAMatrix(const HighsSparseMatrix& a) {
  assert(a.isColwise());
  return {a.num_col_, a.start_.data(), a.index_.data(), a.value_.data()};
}
}  // namespace

extern "C" {
void* highs_rs_dual_row_new();
void highs_rs_dual_row_free(void* p);
void highs_rs_dual_row_setup_slice(void* p, int size, RsArrays* out);
int highs_rs_dual_row_makepack(void* p, int pack_count, int count,
                               const int* index, const double* array,
                               int n_array, int offset);
int highs_rs_dual_row_possible(void* p, int pack_count, double work_delta,
                               int update_count, double td, int num_tot,
                               const int8_t* work_move, const double* work_dual,
                               double* work_theta);
int highs_rs_dual_row_joinpack(void* p, int work_count, const void* other,
                               int other_count);
int highs_rs_dual_row_final_reduce(void* p, int work_count, double work_theta,
                                   double work_delta, int num_tot,
                                   const int8_t* work_move,
                                   const double* work_dual,
                                   const double* work_range);
int highs_rs_dual_row_final(void* p, int* work_count, double* work_theta,
                            double work_delta, double td, int num_tot,
                            const int8_t* work_move, const double* work_dual,
                            const double* work_range,
                            const int* num_tot_permutation, int* work_pivot,
                            double* work_alpha, double* select_theta,
                            double* remain_theta);
double highs_rs_dual_row_update_flip(
    const void* p, int work_count, const RsAMatrix* a, int num_tot,
    const double* work_dual, double cost_scale, int8_t* nonbasic_move,
    double* work_value, const double* work_lower, const double* work_upper,
    int num_row, double* column, int* column_index, int* column_count);
double highs_rs_dual_row_update_dual(const void* p, int pack_count,
                                     double theta, int num_tot,
                                     double* work_dual,
                                     const double* work_value,
                                     const int8_t* nonbasic_flag,
                                     double cost_scale);
void highs_rs_dual_row_create_freelist(void* p, int num_tot,
                                       const int8_t* nonbasic_flag,
                                       const double* work_lower,
                                       const double* work_upper);
void highs_rs_dual_row_create_freemove(const void* p, int update_count,
                                       double work_delta, const RsAMatrix* a,
                                       int num_tot, const double* row_ep,
                                       int8_t* nonbasic_move);
void highs_rs_dual_row_delete_freemove(const void* p, int num_tot,
                                       int8_t* nonbasic_move);
void highs_rs_dual_row_clear_freelist(void* p);
void highs_rs_dual_row_delete_freelist(void* p, int i_var);
double highs_rs_dual_row_devex_weight(const void* p, int pack_count,
                                      int num_tot, const int8_t* nonbasic_flag,
                                      const int* devex_index);
}

HEkkDualRow::RustDualRow::RustDualRow() : p(highs_rs_dual_row_new()) {}
HEkkDualRow::RustDualRow::RustDualRow(RustDualRow&& other) noexcept
    : RustDualRow() {
  std::swap(p, other.p);
}
HEkkDualRow::RustDualRow& HEkkDualRow::RustDualRow::operator=(
    RustDualRow&& other) noexcept {
  std::swap(p, other.p);
  return *this;
}
HEkkDualRow::RustDualRow::~RustDualRow() { highs_rs_dual_row_free(p); }

namespace {
HighsInt numTot(const HEkk& ekk) { return ekk.lp_.num_col_ + ekk.lp_.num_row_; }
}  // namespace

void HEkkDualRow::setupSlice(HighsInt size) {
  workSize = size;
  workMove = ekk_instance_.basis_.nonbasicMove_.data();
  workDual = ekk_instance_.info_.workDual_.data();
  workRange = ekk_instance_.info_.workRange_.data();
  work_devex_index = ekk_instance_.info_.devex_index_.data();

  // Allocate spaces
  packCount = 0;
  workCount = 0;
  RsArrays arrays;
  highs_rs_dual_row_setup_slice(rs_.p, workSize, &arrays);
  packIndex = arrays.pack_index;
  packValue = arrays.pack_value;
  workData = arrays.work_data;
  analysis = &ekk_instance_.analysis_;
}

void HEkkDualRow::setup() {
  // Setup common vectors
  setupSlice(numTot(ekk_instance_));
  workNumTotPermutation = ekk_instance_.info_.numTotPermutation_.data();
  // deleteFreelist() is being called in Phase 1 and Phase 2 since
  // it's in updatePivots(), but create_Freelist() is only called in
  // Phase 2. Hence freeList is not initialised when freeList.empty()
  // is used in deleteFreelist(), clear freeList now.
  highs_rs_dual_row_clear_freelist(rs_.p);
}

void HEkkDualRow::chooseMakepack(const HVector* row, const HighsInt offset) {
  packCount = highs_rs_dual_row_makepack(
      rs_.p, packCount, row->count, row->index.data(), row->array.data(),
      row->array.size(), offset);
}

void HEkkDualRow::choosePossible() {
  workCount = highs_rs_dual_row_possible(
      rs_.p, packCount, workDelta, ekk_instance_.info_.update_count,
      ekk_instance_.options_->dual_feasibility_tolerance,
      numTot(ekk_instance_), workMove, workDual, &workTheta);
}

void HEkkDualRow::chooseJoinpack(const HEkkDualRow* otherRow) {
  workCount = highs_rs_dual_row_joinpack(rs_.p, workCount, otherRow->rs_.p,
                                         otherRow->workCount);
  workTheta = std::min(workTheta, otherRow->workTheta);
}

HighsInt HEkkDualRow::chooseFinal() {
  const HighsInt num_tot = numTot(ekk_instance_);
  // 1. Reduce by large step BFRT
  analysis->simplexTimerStart(Chuzc3Clock);
  workCount = highs_rs_dual_row_final_reduce(rs_.p, workCount, workTheta,
                                             workDelta, num_tot, workMove,
                                             workDual, workRange);
  analysis->simplexTimerStop(Chuzc3Clock);
  // 2-4. Choose by small step BFRT (quadratic sort), then by large
  // alpha, and determine the BFRT flips
  analysis->num_quad_chuzc++;
  analysis->sum_quad_chuzc_size += workCount;
  analysis->max_quad_chuzc_size =
      std::max(workCount, analysis->max_quad_chuzc_size);
  analysis->simplexTimerStart(Chuzc4Clock);
  const double Td = ekk_instance_.options_->dual_feasibility_tolerance;
  double selectTheta = 0;
  double remainTheta = 0;
  const HighsInt fail = highs_rs_dual_row_final(
      rs_.p, &workCount, &workTheta, workDelta, Td, num_tot, workMove,
      workDual, workRange, workNumTotPermutation, &workPivot, &workAlpha,
      &selectTheta, &remainTheta);
  analysis->simplexTimerStop(Chuzc4Clock);
  if (fail) {
    const std::vector<std::pair<HighsInt, double>> data(workData,
                                                        workData + workCount);
    if (fail == 1) {
      debugDualChuzcFailQuad0(*ekk_instance_.options_, workCount, data,
                              num_tot, workDual, selectTheta, remainTheta,
                              true);
    } else {
      debugDualChuzcFailQuad1(*ekk_instance_.options_, workCount, data,
                              num_tot, workDual, selectTheta, true);
    }
    return -1;
  }
  return 0;
}

void HEkkDualRow::updateFlip(HVector* bfrtColumn) {
  bfrtColumn->clear();
  const RsAMatrix a = rsAMatrix(ekk_instance_.lp_.a_matrix_);
  HEkk& ekk = ekk_instance_;
  ekk.info_.updated_dual_objective_value += highs_rs_dual_row_update_flip(
      rs_.p, workCount, &a, numTot(ekk), ekk.info_.workDual_.data(),
      ekk.cost_scale_, ekk.basis_.nonbasicMove_.data(),
      ekk.info_.workValue_.data(), ekk.info_.workLower_.data(),
      ekk.info_.workUpper_.data(), bfrtColumn->array.size(),
      bfrtColumn->array.data(), bfrtColumn->index.data(), &bfrtColumn->count);
}

void HEkkDualRow::updateDual(double theta) {
  analysis->simplexTimerStart(UpdateDualClock);
  HEkk& ekk = ekk_instance_;
  ekk.info_.updated_dual_objective_value += highs_rs_dual_row_update_dual(
      rs_.p, packCount, theta, numTot(ekk), ekk.info_.workDual_.data(),
      ekk.info_.workValue_.data(), ekk.basis_.nonbasicFlag_.data(),
      ekk.cost_scale_);
  analysis->simplexTimerStop(UpdateDualClock);
}

void HEkkDualRow::createFreelist() {
  HEkk& ekk = ekk_instance_;
  highs_rs_dual_row_create_freelist(rs_.p, numTot(ekk),
                                    ekk.basis_.nonbasicFlag_.data(),
                                    ekk.info_.workLower_.data(),
                                    ekk.info_.workUpper_.data());
}

void HEkkDualRow::createFreemove(HVector* row_ep) {
  HEkk& ekk = ekk_instance_;
  const RsAMatrix a = rsAMatrix(ekk.lp_.a_matrix_);
  assert((HighsInt)row_ep->array.size() >= ekk.lp_.num_row_);
  highs_rs_dual_row_create_freemove(rs_.p, ekk.info_.update_count, workDelta,
                                    &a, numTot(ekk), row_ep->array.data(),
                                    ekk.basis_.nonbasicMove_.data());
}

void HEkkDualRow::deleteFreemove() {
  highs_rs_dual_row_delete_freemove(
      rs_.p, numTot(ekk_instance_),
      ekk_instance_.basis_.nonbasicMove_.data());
}

void HEkkDualRow::deleteFreelist(HighsInt iVar) {
  highs_rs_dual_row_delete_freelist(rs_.p, iVar);
}

void HEkkDualRow::computeDevexWeight(const HighsInt /*slice*/) {
  computed_edge_weight = highs_rs_dual_row_devex_weight(
      rs_.p, packCount, numTot(ekk_instance_),
      ekk_instance_.basis_.nonbasicFlag_.data(), work_devex_index);
}

#endif  // HIGHS_RUST
