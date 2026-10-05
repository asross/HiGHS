/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file simplex/HEkkDualRHSRust.cpp
 * @brief HEkkDualRHS delegating to the Rust port
 * (rust/src/simplex/dual_rhs.rs)
 */
#include "simplex/HEkkDualRHS.h"

#ifdef HIGHS_RUST

#include <algorithm>
#include <type_traits>

static_assert(sizeof(HighsInt) == 4, "the Rust dual RHS takes 32-bit ints");
// The Rust draws advance the state of HEkk::random_ in place
static_assert(std::is_standard_layout<HighsRandom>::value &&
                  sizeof(HighsRandom) == sizeof(uint64_t),
              "HighsRandom is just its 64-bit state");

namespace {
// Mirror of CPrimal in rust/src/simplex/dual_rhs.rs
struct RsPrimal {
  double* base_value;
  const double* base_lower;
  const double* base_upper;
  int n;
  double tp;
  int squared;
};
}  // namespace

extern "C" {
void* highs_rs_dual_rhs_new();
void highs_rs_dual_rhs_free(void* p);
double* highs_rs_dual_rhs_setup(void* p, int num_row);
void highs_rs_dual_rhs_scalars(const void* p, int* work_count,
                               double* work_cutoff);
int highs_rs_dual_rhs_choose_normal(void* p, const double* edge_weight,
                                    int n_edge_weight, double* dwork,
                                    int n_dwork, uint64_t* random, int num_row);
int highs_rs_dual_rhs_choose_multi(void* p, int part, int* ch_index,
                                   int ch_limit, const double* edge_weight,
                                   int n_edge_weight, uint64_t* random);
int highs_rs_dual_rhs_update_primal(void* p, int column_count,
                                    const int* column_index, int n_column_index,
                                    const double* column_array,
                                    int n_column_array, double theta,
                                    const RsPrimal* primal, int num_row);
void highs_rs_dual_rhs_update_pivots(void* p, int i_row, double value,
                                     const RsPrimal* primal);
void highs_rs_dual_rhs_update_infeas_list(void* p, const int* column_index,
                                          int column_count,
                                          const double* edge_weight,
                                          int n_edge_weight);
void highs_rs_dual_rhs_create_array(void* p, const RsPrimal* primal,
                                    int num_row);
void highs_rs_dual_rhs_create_infeas_list(void* p, double column_density,
                                          const double* edge_weight,
                                          int n_edge_weight, double* dwork,
                                          int n_dwork, int num_row);
}

namespace {
RsPrimal rsPrimal(HEkk& ekk) {
  HighsSimplexInfo& info = ekk.info_;
  return {info.baseValue_.data(),
          info.baseLower_.data(),
          info.baseUpper_.data(),
          (int)std::min({info.baseValue_.size(), info.baseLower_.size(),
                         info.baseUpper_.size()}),
          ekk.options_->primal_feasibility_tolerance,
          info.store_squared_primal_infeasibility};
}

uint64_t* randomState(HEkk& ekk) {
  return reinterpret_cast<uint64_t*>(&ekk.random_);
}
}  // namespace

HEkkDualRHS::RustDualRhs::RustDualRhs() : p(highs_rs_dual_rhs_new()) {}
HEkkDualRHS::RustDualRhs::~RustDualRhs() { highs_rs_dual_rhs_free(p); }

void HEkkDualRHS::mirror() {
  int count;
  highs_rs_dual_rhs_scalars(rs_.p, &count, &workCutoff);
  workCount = count;
}

void HEkkDualRHS::setup() {
  work_infeasibility.p =
      highs_rs_dual_rhs_setup(rs_.p, ekk_instance_.lp_.num_row_);
}

void HEkkDualRHS::chooseNormal(HighsInt* chIndex) {
  std::vector<double>& edge_weight = ekk_instance_.dual_edge_weight_;
  std::vector<double>& dwork = ekk_instance_.scattered_dual_edge_weight_;
  *chIndex = highs_rs_dual_rhs_choose_normal(
      rs_.p, edge_weight.data(), edge_weight.size(), dwork.data(), dwork.size(),
      randomState(ekk_instance_), ekk_instance_.lp_.num_row_);
  mirror();
}

static HighsInt chooseMulti(void* p, int part, HEkk& ekk, HighsInt* chIndex,
                            HighsInt chLimit) {
  std::vector<double>& edge_weight = ekk.dual_edge_weight_;
  return highs_rs_dual_rhs_choose_multi(p, part, chIndex, chLimit,
                                        edge_weight.data(), edge_weight.size(),
                                        randomState(ekk));
}

void HEkkDualRHS::chooseMultiGlobal(HighsInt* chIndex, HighsInt* chCount,
                                    HighsInt chLimit) {
  *chCount = chooseMulti(rs_.p, 0, ekk_instance_, chIndex, chLimit);
}

void HEkkDualRHS::chooseMultiHyperGraphAuto(HighsInt* chIndex,
                                            HighsInt* chCount,
                                            HighsInt chLimit) {
  *chCount = chooseMulti(rs_.p, 1, ekk_instance_, chIndex, chLimit);
}

void HEkkDualRHS::chooseMultiHyperGraphPart(HighsInt* chIndex,
                                            HighsInt* chCount,
                                            HighsInt chLimit) {
  *chCount = chooseMulti(rs_.p, 2, ekk_instance_, chIndex, chLimit);
}

bool HEkkDualRHS::updatePrimal(HVector* column, double theta) {
  const RsPrimal primal = rsPrimal(ekk_instance_);
  return highs_rs_dual_rhs_update_primal(
      rs_.p, column->count, column->index.data(), column->index.size(),
      column->array.data(), column->array.size(), theta, &primal,
      ekk_instance_.lp_.num_row_);
}

void HEkkDualRHS::updatePivots(const HighsInt iRow, const double value) {
  const RsPrimal primal = rsPrimal(ekk_instance_);
  highs_rs_dual_rhs_update_pivots(rs_.p, iRow, value, &primal);
}

void HEkkDualRHS::updateInfeasList(HVector* column) {
  std::vector<double>& edge_weight = ekk_instance_.dual_edge_weight_;
  highs_rs_dual_rhs_update_infeas_list(rs_.p, column->index.data(),
                                       std::max(column->count, HighsInt{0}),
                                       edge_weight.data(), edge_weight.size());
  mirror();
}

void HEkkDualRHS::createArrayOfPrimalInfeasibilities() {
  const RsPrimal primal = rsPrimal(ekk_instance_);
  highs_rs_dual_rhs_create_array(rs_.p, &primal, ekk_instance_.lp_.num_row_);
}

void HEkkDualRHS::createInfeasList(double columnDensity) {
  std::vector<double>& edge_weight = ekk_instance_.dual_edge_weight_;
  std::vector<double>& dwork = ekk_instance_.scattered_dual_edge_weight_;
  highs_rs_dual_rhs_create_infeas_list(
      rs_.p, columnDensity, edge_weight.data(), edge_weight.size(),
      dwork.data(), dwork.size(), ekk_instance_.lp_.num_row_);
  mirror();
}

#endif  // HIGHS_RUST
