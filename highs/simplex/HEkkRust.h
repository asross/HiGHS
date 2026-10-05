/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file simplex/HEkkRust.h
 * @brief The view of HEkk's data passed to the Rust port of its kernels
 * (rust/src/simplex/ekk.rs, whose module comment describes the design)
 */
#ifndef SIMPLEX_HEKKRUST_H_
#define SIMPLEX_HEKKRUST_H_

#include "HConfig.h"

#ifdef HIGHS_RUST

#include <cstdint>
#include <vector>

#include "util/HVector.h"

struct HighsSimplexStatus;

namespace highs_rs {

// Mirror of CSlice in rust/src/simplex/ekk.rs
template <typename T>
struct Slice {
  T* p;
  int n;
};

template <typename T>
Slice<T> slice(std::vector<T>& v) {
  return {v.data(), (int)v.size()};
}

template <typename T>
Slice<T> slice(const std::vector<T>& v) {
  // The Rust side only reads the vectors it gets as const
  return {const_cast<T*>(v.data()), (int)v.size()};
}

// Mirror of CHVec in rust/src/ffi.rs
struct HVec {
  int size;
  int count;
  int* index;
  int n_index;
  double* array;
  int n_array;
  char* cwork;
  int n_cwork;
  int* iwork;
  int n_iwork;
  double synthetic_tick;
  unsigned char pack_flag;
  int pack_count;
  int* pack_index;
  int n_pack_index;
  double* pack_value;
  int n_pack_value;
};

// An HVector's data for a call to Rust, whose changes to the scalars are
// copied back on destruction (the arrays are shared)
struct HVecCall {
  HVector& v;
  HVec rs;
  explicit HVecCall(HVector& vector)
      : v(vector),
        rs{v.size,
           v.count,
           v.index.data(),
           (int)v.index.size(),
           v.array.data(),
           (int)v.array.size(),
           v.cwork.data(),
           (int)v.cwork.size(),
           v.iwork.data(),
           (int)v.iwork.size(),
           v.synthetic_tick,
           (unsigned char)v.packFlag,
           v.packCount,
           v.packIndex.data(),
           (int)v.packIndex.size(),
           v.packValue.data(),
           (int)v.packValue.size()} {}
  ~HVecCall() {
    v.count = rs.count;
    v.synthetic_tick = rs.synthetic_tick;
    v.packFlag = rs.pack_flag != 0;
    v.packCount = rs.pack_count;
  }
  HVec* get() { return &rs; }
  HVecCall(const HVecCall&) = delete;
  HVecCall& operator=(const HVecCall&) = delete;
};

// Mirror of CEkk in rust/src/simplex/ekk.rs: filled by HEkk::rustView()
struct Ekk {
  int num_col;
  int num_row;
  // lp_
  Slice<HighsInt> a_start;
  Slice<HighsInt> a_index;
  Slice<double> a_value;
  Slice<double> col_cost;
  Slice<double> col_lower;
  Slice<double> col_upper;
  Slice<double> row_lower;
  Slice<double> row_upper;
  int sense;
  double offset;
  // Scaling of the basis matrix (simplex_nla_.scale_), if any
  bool has_scale;
  Slice<double> col_scale;
  Slice<double> row_scale;
  // ar_matrix_
  Slice<HighsInt> ar_start;
  Slice<HighsInt> ar_p_end;
  Slice<HighsInt> ar_index;
  Slice<double> ar_value;
  // basis_
  Slice<HighsInt> basic_index;
  Slice<int8_t> nonbasic_flag;
  Slice<int8_t> nonbasic_move;
  uint64_t* basis_hash;
  // info_ arrays
  Slice<double> work_cost;
  Slice<double> work_dual;
  Slice<double> work_shift;
  Slice<double> work_lower;
  Slice<double> work_upper;
  Slice<double> work_range;
  Slice<double> work_value;
  Slice<double> work_lower_shift;
  Slice<double> work_upper_shift;
  Slice<double> base_lower;
  Slice<double> base_upper;
  Slice<double> base_value;
  Slice<double> num_tot_random_value;
  Slice<double> dual_edge_weight;
  Slice<double> scattered_dual_edge_weight;
  // info_ scalars
  double* col_aq_density;
  double* row_ep_density;
  double* row_ap_density;
  double* row_dse_density;
  double* primal_col_density;
  double* dual_col_density;
  HighsInt* update_count;
  HighsInt* num_basic_logicals;
  double* updated_dual_objective_value;
  double* primal_objective_value;
  double* dual_objective_value;
  HighsInt* num_primal_infeasibilities;
  double* max_primal_infeasibility;
  double* sum_primal_infeasibilities;
  HighsInt* num_dual_infeasibilities;
  double* max_dual_infeasibility;
  double* sum_dual_infeasibilities;
  bool* costs_shifted;
  bool* costs_perturbed;
  bool* bounds_shifted;
  bool* bounds_perturbed;
  int price_strategy;
  double dual_simplex_cost_perturbation_multiplier;
  double primal_simplex_bound_perturbation_multiplier;
  // options_
  double primal_feasibility_tolerance;
  double dual_feasibility_tolerance;
  int cost_scale_factor;
  bool output_flag;
  // HEkk scalars
  double cost_scale;
  double* cost_perturbation_base;
  double* cost_perturbation_max_abs_cost;
  bool simplex_in_scaled_space;
  int update_limit;
  double* build_synthetic_tick;
  double* total_synthetic_tick;
  void* factor;
  // The factor's constraint matrix (HFactor::a_start, ...)
  int factor_num_col;
  Slice<const HighsInt> factor_a_start;
  Slice<const HighsInt> factor_a_index;
  Slice<const double> factor_a_value;
  // HEkkPrimal
  HighsSimplexStatus* status;
  HighsInt* iteration_count;
  double* updated_primal_objective_value;
  bool* allow_bound_perturbation;
  bool* backtracking;
  HighsInt* primal_phase1_iteration_count;
  HighsInt* primal_phase2_iteration_count;
  HighsInt* primal_bound_swap;
  double* col_basic_feasibility_change_density;
  double* row_basic_feasibility_change_density;
  double* col_steepest_edge_density;
  double primal_simplex_phase1_cost_perturbation_multiplier;
  int simplex_primal_edge_weight_strategy;
  int simplex_iteration_limit;
  // Whether HEkk::bailout() has more to check than the iteration limit
  bool bailout_in_cpp;
  // Whether HighsSimplexAnalysis::iterationReport() reports
  bool iteration_report;
};

// Mirrors of the results of some kernels
struct Infeasibility {
  int num;
  double max;
  double sum;
};

struct NumericalTrouble {
  double measure;
  double new_pivot_threshold;
  bool reinvert;
};

struct CostPerturbationReport {
  int num_original_nonzero_cost;
  int pct0;
  double min_abs_cost;
  double average_abs_cost;
  double max_abs_cost;
  bool large;
  double large_max_abs_cost;
  double boxed_rate;
  bool small_boxed_rate;
  double small_boxed_max_abs_cost;
  double row_cost_perturbation_base;
  bool perturbed;
};

}  // namespace highs_rs

extern "C" {
void highs_rs_ekk_compute_primal(const highs_rs::Ekk* ekk,
                                 highs_rs::HVec* primal_col);
void highs_rs_ekk_compute_dual(const highs_rs::Ekk* ekk,
                               highs_rs::HVec* dual_col,
                               highs_rs::HVec* dual_row);
void highs_rs_ekk_full_btran(const highs_rs::Ekk* ekk, highs_rs::HVec* buffer);
void highs_rs_ekk_full_price(const highs_rs::Ekk* ekk, highs_rs::HVec* full_col,
                             highs_rs::HVec* full_row);
void highs_rs_ekk_unit_btran(const highs_rs::Ekk* ekk, int i_row,
                             highs_rs::HVec* row_ep);
void highs_rs_ekk_pivot_column_ftran(const highs_rs::Ekk* ekk, int i_col,
                                     highs_rs::HVec* col_aq);
void highs_rs_ekk_tableau_row_price(const highs_rs::Ekk* ekk,
                                    highs_rs::HVec* row_ep,
                                    highs_rs::HVec* row_ap);
void highs_rs_ekk_transform_for_update(const highs_rs::Ekk* ekk,
                                       highs_rs::HVec* aq, highs_rs::HVec* ep,
                                       int variable_in, int row_out);
void highs_rs_ekk_compute_simplex_primal_infeasible(const highs_rs::Ekk* ekk);
void highs_rs_ekk_compute_simplex_dual_infeasible(const highs_rs::Ekk* ekk);
highs_rs::Infeasibility highs_rs_ekk_compute_simplex_lp_dual_infeasible(
    const highs_rs::Ekk* ekk);
void highs_rs_ekk_compute_primal_objective_value(const highs_rs::Ekk* ekk);
void highs_rs_ekk_compute_dual_objective_value(const highs_rs::Ekk* ekk,
                                               int phase);
void highs_rs_ekk_zero_basic_duals(const highs_rs::Ekk* ekk);
void highs_rs_ekk_update_pivots(const highs_rs::Ekk* ekk, int variable_in,
                                int row_out, int move_out);
void highs_rs_ekk_update_factor(const highs_rs::Ekk* ekk,
                                highs_rs::HVec* column, highs_rs::HVec* row_ep,
                                int i_row, int* hint);
void highs_rs_ekk_update_matrix(const highs_rs::Ekk* ekk, int variable_in,
                                int variable_out);
double highs_rs_ekk_compute_dual_steepest_edge_weight(const highs_rs::Ekk* ekk,
                                                      int i_row,
                                                      highs_rs::HVec* row_ep);
void highs_rs_ekk_compute_dual_steepest_edge_weights(const highs_rs::Ekk* ekk,
                                                     highs_rs::HVec* row_ep);
void highs_rs_ekk_update_dual_steepest_edge_weights(
    const highs_rs::Ekk* ekk, int row_out, int variable_in,
    highs_rs::HVec* column, double new_pivotal_edge_weight, double kai,
    const double* dse_array);
void highs_rs_ekk_update_dual_devex_weights(const highs_rs::Ekk* ekk,
                                            highs_rs::HVec* column,
                                            double new_pivotal_edge_weight);
void highs_rs_ekk_initialise_cost(const highs_rs::Ekk* ekk, int algorithm,
                                  bool perturb,
                                  highs_rs::CostPerturbationReport* report);
void highs_rs_ekk_initialise_bound(const highs_rs::Ekk* ekk, int algorithm,
                                   int solve_phase, bool perturb);
void highs_rs_ekk_initialise_lp(const highs_rs::Ekk* ekk, int which);
void highs_rs_ekk_set_nonbasic_move(const highs_rs::Ekk* ekk);
void highs_rs_ekk_initialise_nonbasic_value_and_move(const highs_rs::Ekk* ekk);
double highs_rs_ekk_compute_dual_for_tableau_column(
    const double* work_cost, int n_work_cost, const int* basic_index,
    int num_row, int i_var, int count, const int* tableau_index,
    const double* tableau_array, int n_tableau_array);
void highs_rs_ekk_flip_bound(int num_tot, int8_t* nonbasic_move,
                             double* work_value, const double* work_lower,
                             const double* work_upper, int i_col);
highs_rs::NumericalTrouble highs_rs_ekk_reinvert_on_numerical_trouble(
    double alpha_from_col, double alpha_from_row,
    double numerical_trouble_tolerance, int update_count,
    double current_pivot_threshold);
int highs_rs_ekk_choose_price_technique(int price_strategy,
                                        double row_ep_density);
}

#endif  // HIGHS_RUST

#endif /* SIMPLEX_HEKKRUST_H_ */
