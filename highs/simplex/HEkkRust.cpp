/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file simplex/HEkkRust.cpp
 * @brief The view of HEkk's data for the Rust port of its kernels
 * (rust/src/simplex/ekk.rs). The HEkk methods delegating to them are in
 * HEkk.cpp, under HIGHS_RUST
 */
#include "simplex/HEkkRust.h"

#ifdef HIGHS_RUST

#include <type_traits>

#include "simplex/HEkk.h"

static_assert(sizeof(HighsInt) == 4, "the Rust HEkk kernels take 32-bit ints");
static_assert(sizeof(bool) == 1, "Rust bools are bytes");

highs_rs::Ekk HEkk::rustView() {
  using highs_rs::slice;
  const HighsScale* scale = simplex_nla_.scale_;
  const HighsSparseMatrix& a = lp_.a_matrix_;
  highs_rs::Ekk v;
  v.num_col = lp_.num_col_;
  v.num_row = lp_.num_row_;
  v.a_start = slice(a.start_);
  v.a_index = slice(a.index_);
  v.a_value = slice(a.value_);
  v.col_cost = slice(lp_.col_cost_);
  v.col_lower = slice(lp_.col_lower_);
  v.col_upper = slice(lp_.col_upper_);
  v.row_lower = slice(lp_.row_lower_);
  v.row_upper = slice(lp_.row_upper_);
  v.sense = (int)lp_.sense_;
  v.offset = lp_.offset_;
  v.has_scale = scale != nullptr;
  v.col_scale = scale ? slice(scale->col) : highs_rs::Slice<double>{nullptr, 0};
  v.row_scale = scale ? slice(scale->row) : highs_rs::Slice<double>{nullptr, 0};
  v.ar_start = slice(ar_matrix_.start_);
  v.ar_p_end = slice(ar_matrix_.p_end_);
  v.ar_index = slice(ar_matrix_.index_);
  v.ar_value = slice(ar_matrix_.value_);
  v.basic_index = slice(basis_.basicIndex_);
  v.nonbasic_flag = slice(basis_.nonbasicFlag_);
  v.nonbasic_move = slice(basis_.nonbasicMove_);
  v.basis_hash = &basis_.hash;
  v.work_cost = slice(info_.workCost_);
  v.work_dual = slice(info_.workDual_);
  v.work_shift = slice(info_.workShift_);
  v.work_lower = slice(info_.workLower_);
  v.work_upper = slice(info_.workUpper_);
  v.work_range = slice(info_.workRange_);
  v.work_value = slice(info_.workValue_);
  v.work_lower_shift = slice(info_.workLowerShift_);
  v.work_upper_shift = slice(info_.workUpperShift_);
  v.base_lower = slice(info_.baseLower_);
  v.base_upper = slice(info_.baseUpper_);
  v.base_value = slice(info_.baseValue_);
  v.num_tot_random_value = slice(info_.numTotRandomValue_);
  v.dual_edge_weight = slice(dual_edge_weight_);
  v.scattered_dual_edge_weight = slice(scattered_dual_edge_weight_);
  v.col_aq_density = &info_.col_aq_density;
  v.row_ep_density = &info_.row_ep_density;
  v.row_ap_density = &info_.row_ap_density;
  v.row_dse_density = &info_.row_DSE_density;
  v.primal_col_density = &info_.primal_col_density;
  v.dual_col_density = &info_.dual_col_density;
  v.update_count = &info_.update_count;
  v.num_basic_logicals = &info_.num_basic_logicals;
  v.updated_dual_objective_value = &info_.updated_dual_objective_value;
  v.primal_objective_value = &info_.primal_objective_value;
  v.dual_objective_value = &info_.dual_objective_value;
  v.num_primal_infeasibilities = &info_.num_primal_infeasibilities;
  v.max_primal_infeasibility = &info_.max_primal_infeasibility;
  v.sum_primal_infeasibilities = &info_.sum_primal_infeasibilities;
  v.num_dual_infeasibilities = &info_.num_dual_infeasibilities;
  v.max_dual_infeasibility = &info_.max_dual_infeasibility;
  v.sum_dual_infeasibilities = &info_.sum_dual_infeasibilities;
  v.costs_shifted = &info_.costs_shifted;
  v.costs_perturbed = &info_.costs_perturbed;
  v.bounds_shifted = &info_.bounds_shifted;
  v.bounds_perturbed = &info_.bounds_perturbed;
  v.price_strategy = info_.price_strategy;
  v.dual_simplex_cost_perturbation_multiplier =
      info_.dual_simplex_cost_perturbation_multiplier;
  v.primal_simplex_bound_perturbation_multiplier =
      info_.primal_simplex_bound_perturbation_multiplier;
  v.primal_feasibility_tolerance = options_->primal_feasibility_tolerance;
  v.dual_feasibility_tolerance = options_->dual_feasibility_tolerance;
  v.cost_scale_factor = options_->cost_scale_factor;
  v.output_flag = options_->output_flag;
  v.cost_scale = cost_scale_;
  v.cost_perturbation_base = &cost_perturbation_base_;
  v.cost_perturbation_max_abs_cost = &cost_perturbation_max_abs_cost_;
  v.simplex_in_scaled_space = simplex_in_scaled_space_;
  v.update_limit = info_.update_limit;
  v.build_synthetic_tick = &build_synthetic_tick_;
  v.total_synthetic_tick = &total_synthetic_tick_;
  const HFactor& factor = simplex_nla_.factor_;
  v.factor = factor.rs_.p;
  v.factor_num_col = factor.num_col;
  const bool has_a = factor.a_start != nullptr;
  const int a_nnz = has_a ? factor.a_start[factor.num_col] : 0;
  v.factor_a_start = {factor.a_start, has_a ? factor.num_col + 1 : 0};
  v.factor_a_index = {factor.a_index, a_nnz};
  v.factor_a_value = {factor.a_value, a_nnz};
  return v;
}

HEkk::RustBasisRecords::RustBasisRecords() : p(highs_rs_basis_records_new()) {}
HEkk::RustBasisRecords::RustBasisRecords(const RustBasisRecords& other)
    : RustBasisRecords() {
  highs_rs_basis_records_copy(p, other.p);
}
HEkk::RustBasisRecords& HEkk::RustBasisRecords::operator=(
    const RustBasisRecords& other) {
  if (this != &other) highs_rs_basis_records_copy(p, other.p);
  return *this;
}
HEkk::RustBasisRecords::~RustBasisRecords() { highs_rs_basis_records_free(p); }

#endif  // HIGHS_RUST
