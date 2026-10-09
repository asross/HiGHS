/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file simplex/HEkkRust.h
 * @brief The Rust-owned simplex engine (rust/src/simplex/lp_solver.rs,
 * whose module comment describes the design): the mirrors of what the C++
 * HEkk shell shares with it, and its calls
 */
#ifndef SIMPLEX_HEKKRUST_H_
#define SIMPLEX_HEKKRUST_H_

#include "HConfig.h"

#ifdef HIGHS_RUST

#include <cstdint>
#include <utility>
#include <vector>

#include "lp_data/HighsRust.h"
#include "simplex/HighsSimplexAnalysis.h"
#include "simplex/SimplexStruct.h"
#include "util/HVector.h"

namespace highs_rs {

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

// Mirror of Host in rust/src/simplex/hekk.rs: the C++ that the simplex
// calls
struct Host {
  void* ctx;
  void (*log)(void*, int channel, int type, const char* msg);
  double (*timer_read)(void*);
  bool (*interrupt)(void*);
  void (*chuzc_fail)(void*, int kind, int work_count,
                     const std::pair<HighsInt, double>* work_data,
                     double select_theta, double remain_theta);
};

// Mirror of SharedInfo in rust/src/simplex/lp_solver.rs: the info_ values
// that C++ reads (HighsSimplexInfo's names)
struct HEkkInfo {
  double dual_objective_value;
  double primal_objective_value;
  double max_primal_infeasibility;
  double sum_primal_infeasibilities;
  double max_dual_infeasibility;
  double sum_dual_infeasibilities;
  double factor_pivot_threshold;
  double row_ep_density;
  double col_aq_density;
  HighsInt num_primal_infeasibilities;
  HighsInt num_dual_infeasibilities;
  HighsInt dual_edge_weight_strategy;
};

// Mirror of EkkShared in rust/src/simplex/lp_solver.rs: HEkk's scalars
// that C++ reads and writes in place
struct HEkkShared {
  HEkkInfo info;
  HighsModelStatus model_status;
  HighsInt iteration_count;
  SimplexAlgorithm exit_algorithm;
  HighsInt debug_solve_call_num;
  HighsInt debug_initial_build_synthetic_tick;
  HighsInt dual_ray_index;
  HighsInt dual_ray_sign;
  HighsInt primal_ray_index;
  HighsInt primal_ray_sign;
  HighsSimplexStatus status;
  bool dual_values_valid;
  bool nla_lp_set;
};

// Mirror of LpsOptions in rust/src/simplex/lp_solver.rs
struct LpsOptions {
  double primal_feasibility_tolerance;
  double dual_feasibility_tolerance;
  double time_limit;
  double objective_bound;
  double dual_simplex_pivot_growth_tolerance;
  double small_matrix_value;
  double dual_steepest_edge_weight_error_tolerance;
  double rebuild_refactor_solution_error_tolerance;
  double factor_pivot_tolerance;
  double factor_pivot_threshold;
  double dual_simplex_cost_perturbation_multiplier;
  double primal_simplex_bound_perturbation_multiplier;
  double dual_steepest_edge_weight_log_error_threshold;
  HighsInt cost_scale_factor;
  HighsInt log_dev_level;
  HighsInt dev_level;
  HighsInt simplex_primal_edge_weight_strategy;
  HighsInt simplex_iteration_limit;
  HighsInt simplex_update_limit;
  HighsInt max_dual_simplex_cleanup_level;
  HighsInt max_dual_simplex_phase1_cleanup_level;
  HighsInt simplex_dse_exact_init_max_rows;
  HighsInt simplex_strategy;
  HighsInt simplex_min_concurrency;
  HighsInt simplex_max_concurrency;
  HighsInt simplex_dual_edge_weight_strategy;
  HighsInt simplex_price_strategy;
  HighsInt random_seed;
  bool output_flag;
  bool no_unnecessary_rebuild_refactor;
  bool allow_unbounded_or_infeasible;
  bool less_infeasible_dse_check;
  bool less_infeasible_dse_choose_row;
  bool simplex_keep_random_vectors;
};

// Mirror of LpsEnv in rust/src/simplex/lp_solver.rs: what a call needs
// from C++
struct LpsEnv {
  RsLp lp;
  RsMut<char> model_name;
  HighsInt nla_num_col;
  HighsInt nla_num_row;
  bool nla_has_scale;
  RsMut<double> nla_col_scale;
  RsMut<double> nla_row_scale;
  LpsOptions opt;
  Host host;
  SimplexReport* report;
  HighsInt* num_invert;
  HighsInt num_threads;
  bool interrupt_callback;
};

struct LpsSolveOut {
  int status;
  HighsInt invert_num_el;
  HighsInt basis_matrix_num_el;
};

// Mirror of UnscaledInfeasibilities in rust/src/simplex/lp_solver.rs
struct UnscaledInfeasibilities {
  HighsInt num_primal_infeasibilities;
  double max_primal_infeasibility;
  double sum_primal_infeasibilities;
  HighsInt num_dual_infeasibilities;
  double max_dual_infeasibility;
  double sum_dual_infeasibilities;
};

// Mirror of RangingSlices in rust/src/simplex/lp_solver.rs
struct RangingSlices {
  RsMut<double> work_value, work_dual, work_cost, work_lower, work_upper,
      base_value, base_lower, base_upper;
  RsMut<int8_t> nonbasic_flag, nonbasic_move;
  RsMut<HighsInt> basic_index;
};

}  // namespace highs_rs

extern "C" {
void* highs_rs_lps_new();
void highs_rs_lps_free(void* p);
highs_rs::HEkkShared* highs_rs_lps_shared(void* p);
void* highs_rs_lps_records(void* p);
void highs_rs_lps_clear(void* p);
void highs_rs_lps_invalidate(void* p);
bool highs_rs_lps_update_status(void* p, int action);
void highs_rs_lps_clear_ray_records(void* p);
bool highs_rs_lps_move_lp(void* p, const highs_rs::LpsEnv* env);
void highs_rs_lps_set_basis_logical(void* p, const highs_rs::LpsEnv* env);
void highs_rs_lps_set_basis(void* p, const highs_rs::LpsEnv* env,
                            const uint8_t* col_status,
                            const uint8_t* row_status, HighsInt debug_id,
                            HighsInt debug_update_count, const char* origin,
                            size_t origin_len);
void highs_rs_lps_get_highs_basis(void* p, const RsLp* use_lp, int sense,
                                  uint8_t* col_status, uint8_t* row_status,
                                  HighsInt* debug_id,
                                  HighsInt* debug_update_count,
                                  const char** origin, size_t* origin_len);
void highs_rs_lps_get_solution(void* p, const highs_rs::LpsEnv* env,
                               double* col_value, double* col_dual,
                               double* row_value, double* row_dual);
void highs_rs_lps_unscale_simplex(void* p, const RsLp* lp);
void highs_rs_lps_unscaled_infeasibilities(
    void* p, const highs_rs::LpsEnv* env, const RsLp* lp,
    highs_rs::UnscaledInfeasibilities* out);
void highs_rs_lps_add_rows(void* p, HighsInt num_weighted_row,
                           HighsInt new_num_row);
void highs_rs_lps_delete_rows(void* p, const RsIndexCollection* ic);
void highs_rs_lps_resize_basis(void* p, HighsInt num_tot);
void highs_rs_lps_append_basic_rows(void* p, HighsInt num_col,
                                    HighsInt num_row, HighsInt new_num_row);
void highs_rs_lps_flip_nonbasic_move(void* p, HighsInt var);
// The engine's LP: a copy of a C++ LP, a view, its model name, clearing it
void highs_rs_lps_import_lp(void* p, const RsLp* lp, const char* name,
                            size_t name_len);
void highs_rs_lps_lp_view(void* p, RsLp* out);
void highs_rs_lps_model_name(void* p, RsMut<char>* out);
void highs_rs_lps_clear_lp(void* p);
void highs_rs_lps_set_lp_num_row(void* p, HighsInt num_row);
// The simplex NLA's LP is the engine's (else a C++ LP passed with each call)
void highs_rs_lps_set_nla_rust(void* p, bool rust);
int highs_rs_lps_initialise_basis_and_factor(void* p,
                                             const highs_rs::LpsEnv* env,
                                             bool only_from_known_basis);
bool highs_rs_lps_lp_factor_row_compatible(void* p,
                                           const highs_rs::LpsEnv* env,
                                           HighsInt expected_num_row);
void highs_rs_lps_nla_solve(void* p, const highs_rs::LpsEnv* env,
                            highs_rs::HVec* rhs, double expected_density,
                            bool transposed);
void highs_rs_lps_put_iterate(void* p);
bool highs_rs_lps_get_iterate(void* p);
double highs_rs_lps_basis_condition(void* p, const highs_rs::LpsEnv* env,
                                    const RsLp* lp, const char* name,
                                    size_t name_len, bool exact, bool report);
bool highs_rs_lps_proof_of_primal_infeasibility(void* p,
                                                const highs_rs::LpsEnv* env);
highs_rs::LpsSolveOut highs_rs_lps_solve(void* p, const highs_rs::LpsEnv* env,
                                         bool force_phase2);
void highs_rs_lps_return_from_solve(void* p);
HighsInt* highs_rs_lps_basic_index(void* p, HighsInt* n);
int8_t* highs_rs_lps_nonbasic(void* p, int which, HighsInt* n);
const double* highs_rs_lps_dual_edge_weight(void* p);
const double* highs_rs_lps_work_dual(void* p, HighsInt* n);
const double* highs_rs_lps_ray_value(void* p, bool primal, size_t* n);
void highs_rs_lps_set_ray_value(void* p, bool primal, const double* value,
                                size_t n);
void highs_rs_lps_ranging_slices(void* p, highs_rs::RangingSlices* out);
HighsInt highs_rs_lps_factor_num_row(void* p);
// What a solve or INVERT left in the basis records
bool highs_rs_ekk_hot_start(void* p, bool* refactor_use,
                            const HighsInt** pivot_row,
                            const HighsInt** pivot_var,
                            const int8_t** pivot_type, int* num_pivot,
                            double* build_synthetic_tick,
                            const int8_t** nonbasic_move, int* num_tot);
bool highs_rs_ekk_primal_phase1_dual(void* p, const double** values, int* n);
void highs_rs_ekk_clear_out(void* p);
}

#endif  // HIGHS_RUST

#endif /* SIMPLEX_HEKKRUST_H_ */
