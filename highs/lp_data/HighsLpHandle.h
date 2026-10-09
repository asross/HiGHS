/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file lp_data/HighsLpHandle.h
 * @brief The Rust LP solver (rust/src/lp_data/lp_handle.rs): the LP
 * solvers of the MIP, and the simplex engine of a Highs object
 */
#ifndef LP_DATA_HIGHS_LP_HANDLE_H_
#define LP_DATA_HIGHS_LP_HANDLE_H_

#include "HConfig.h"

#ifdef HIGHS_RUST

#include <cstdint>
#include <utility>
#include <vector>

#include "lp_data/HStruct.h"
#include "lp_data/HighsRust.h"
#include "lp_data/HighsStatus.h"
#include "simplex/SimplexStruct.h"
#include "util/HVector.h"
// Highs.h included these through the C++ simplex engine (HEkk.h)
#include "util/HFactor.h"
#include "util/HighsHash.h"
#include "util/HighsRandom.h"

namespace highs_rs {

struct LpHandle;

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

// Mirror of SharedInfo in rust/src/simplex/lp_solver.rs: the simplex
// info's values that C++ reads (HighsSimplexInfo's names)
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

// Mirror of EkkShared in rust/src/simplex/lp_solver.rs: the engine's
// scalars that C++ reads and writes in place
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

// Mirror of RangingSlices in rust/src/simplex/lp_solver.rs
struct RangingSlices {
  RsMut<double> work_value, work_dual, work_cost, work_lower, work_upper,
      base_value, base_lower, base_upper;
  RsMut<int8_t> nonbasic_flag, nonbasic_move;
  RsMut<HighsInt> basic_index;
};

// lp_handle.rs: LphView, what C++ reads of the LP solver (valid until it
// changes)
struct LphView {
  HighsInt num_col, num_row, num_nz;
  int model_status;
  const double *col_cost, *col_lower, *col_upper, *row_lower, *row_upper;
  const HighsInt *a_start, *a_index;
  const double* a_value;
  const double *col_value, *col_dual, *row_value, *row_dual;
  HighsInt n_col_value, n_col_dual, n_row_value, n_row_dual;
  const uint8_t *col_status, *row_status;
  HighsInt n_col_status, n_row_status;
  bool value_valid, dual_valid, basis_valid;
  const HighsInfoStruct* info;
};

// lp_handle.rs: LphBasis, a HighsBasis' fields
struct LphBasis {
  bool valid, alien, useful, was_alien;
  HighsInt debug_id, debug_update_count;
  const uint8_t* col_status;
  HighsInt n_col;
  const uint8_t* row_status;
  HighsInt n_row;
  const char* origin;
  size_t origin_len;
};

// lp_handle.rs: CHost, what the engine of a Highs object calls in C++
struct LphHost {
  void* ctx;
  const void* log_options;
  double (*timer_read)(void* ctx);
  bool (*simplex_interrupt)(void* ctx, HighsInt iteration_count);
  HighsInt (*ipm_interrupt)(void* ctx, HighsInt ipm_iteration_count);
};

// lp_handle.rs: FormBasis, the views of a basis that
// formSimplexLpBasisAndFactor reads and changes
struct LphFormBasis {
  bool valid, useful;
  bool* alien;
  RsMut<uint8_t> col_status, row_status;
  HighsInt debug_id, debug_update_count;
  RsMut<char> origin;
};

extern "C" {
LpHandle* highs_rs_lph_new();
void highs_rs_lph_free(LpHandle* h);
void highs_rs_lph_register(void (*profiling)(void*, int, HighsInt, int64_t));
void highs_rs_lph_view(LpHandle* h, LphView* v);
void highs_rs_lph_basis(LpHandle* h, LphBasis* out);
int highs_rs_lph_set_basis(LpHandle* h, const LphBasis* b, const char* origin,
                           size_t len);
bool highs_rs_lph_set_option(LpHandle* h, const char* name, size_t name_len,
                             int kind, int64_t i, double d, const char* s,
                             size_t len);
bool highs_rs_lph_get_option(LpHandle* h, const char* name, size_t name_len,
                             double* d, const char** s, size_t* len);
void highs_rs_lph_pass_options(LpHandle* h, LpHandle* from);
int highs_rs_lph_pass_model(LpHandle* h, const RsLp* lp, const char* name,
                            size_t len);
int highs_rs_lph_pass_model_of(LpHandle* h, LpHandle* from);
void highs_rs_lph_model(LpHandle* h, RsLp* out, const char** name,
                        size_t* len);
int highs_rs_lph_clear_solver(LpHandle* h);
int highs_rs_lph_clear_model(LpHandle* h);
int highs_rs_lph_change_col_bounds_interval(LpHandle* h, HighsInt from,
                                            HighsInt to, const double* lower,
                                            const double* upper);
int highs_rs_lph_change_col_bounds_set(LpHandle* h, HighsInt num,
                                       const HighsInt* set,
                                       const double* lower,
                                       const double* upper);
int highs_rs_lph_change_col_costs_mask(LpHandle* h, const HighsInt* mask,
                                       const double* cost);
int highs_rs_lph_add_rows(LpHandle* h, HighsInt num, const double* lower,
                          const double* upper, HighsInt num_nz,
                          const HighsInt* start, const HighsInt* index,
                          const double* value);
int highs_rs_lph_delete_rows_interval(LpHandle* h, HighsInt from,
                                      HighsInt to);
int highs_rs_lph_delete_rows_mask(LpHandle* h, HighsInt* mask);
int highs_rs_lph_optimize_lp(LpHandle* h, bool* interrupted);
int highs_rs_lph_put_iterate(LpHandle* h);
int highs_rs_lph_get_iterate(LpHandle* h);
void highs_rs_lph_basis_inverse_row(LpHandle* h, HighsInt row, HVec* v);
bool highs_rs_lph_dual_ray(LpHandle* h, HVec* v);
bool highs_rs_lph_has_invert(LpHandle* h);
const HighsInt* highs_rs_lph_basic_index(LpHandle* h);
const double* highs_rs_lph_dual_edge_weights(LpHandle* h);
double highs_rs_lph_run_time(LpHandle* h);
void highs_rs_lph_set_profiling(LpHandle* h, void* profiling);
int highs_rs_lph_race_ipx(LpHandle* h, HighsInt seed, int64_t* extra,
                          bool* ipx_won);
void highs_rs_lph_ipm_basis(LpHandle* h, bool use_presolve, void* profiling);

// The engine of a Highs object
LpHandle* highs_rs_lph_new_host(const LphHost* host);
void* highs_rs_lph_lps(LpHandle* h);
void highs_rs_lph_import_model(LpHandle* h, const RsLp* lp, const char* name,
                               size_t len);
bool highs_rs_lph_take_matrix_back(LpHandle* h);
void highs_rs_lph_ekk_clear(LpHandle* h);
void highs_rs_lph_ekk_invalidate(LpHandle* h);
void highs_rs_lph_update_status(LpHandle* h, int action);
void highs_rs_lph_clear_shell(LpHandle* h);
void highs_rs_lph_set_lp_name(LpHandle* h, const char* name, size_t len);
void highs_rs_lph_set_nla_lp(LpHandle* h, const RsLp* lp);
void highs_rs_lph_nla_solve(LpHandle* h, const RsLp* lp, HVec* rhs,
                            double expected_density, bool transposed);
double highs_rs_lph_basis_condition(LpHandle* h, const RsLp* lp,
                                    const char* name, size_t len, bool exact,
                                    bool report);
int highs_rs_lph_factor_row_compatible(LpHandle* h, HighsInt expected_num_row);
int highs_rs_lph_form_basis(LpHandle* h, const LphFormBasis* b,
                            bool only_from_known_basis);
HighsSimplexStats* highs_rs_lph_simplex_stats(LpHandle* h);
void highs_rs_lph_initialise_simplex_stats(LpHandle* h);
bool highs_rs_lph_hot_start(LpHandle* h, bool* refactor_use,
                            const HighsInt** pivot_row,
                            const HighsInt** pivot_var,
                            const int8_t** pivot_type, int* num_pivot,
                            double* build_synthetic_tick,
                            const int8_t** nonbasic_move, int* num_tot);
const double* highs_rs_lph_primal_phase1_dual(LpHandle* h, size_t* n);

// The engine's calls that need no environment (lp_solver.rs)
HEkkShared* highs_rs_lps_shared(void* p);
void highs_rs_lps_clear_ray_records(void* p);
void highs_rs_lps_get_highs_basis(void* p, const RsLp* use_lp, int sense,
                                  uint8_t* col_status, uint8_t* row_status,
                                  HighsInt* debug_id,
                                  HighsInt* debug_update_count,
                                  const char** origin, size_t* origin_len);
void highs_rs_lps_unscale_simplex(void* p, const RsLp* lp);
void highs_rs_lps_put_iterate(void* p);
bool highs_rs_lps_get_iterate(void* p);
HighsInt* highs_rs_lps_basic_index(void* p, HighsInt* n);
int8_t* highs_rs_lps_nonbasic(void* p, int which, HighsInt* n);
const double* highs_rs_lps_ray_value(void* p, bool primal, size_t* n);
void highs_rs_lps_set_ray_value(void* p, bool primal, const double* value,
                                size_t n);
void highs_rs_lps_ranging_slices(void* p, RangingSlices* out);
}
}  // namespace highs_rs

/// An LP solver of the MIP (rust/src/lp_data/lp_handle.rs) that C++ owns
struct HighsLpHandle {
  highs_rs::LpHandle* p;
  HighsLpHandle() : p(highs_rs::highs_rs_lph_new()) {}
  ~HighsLpHandle() { highs_rs::highs_rs_lph_free(p); }
  HighsLpHandle(const HighsLpHandle&) = delete;
  HighsLpHandle& operator=(const HighsLpHandle&) = delete;
};

class HighsLp;
class Highs;

class HighsOptions;
struct HighsEngine;

// The host functions of a Highs object's engine (HighsRunRust.cpp)
highs_rs::LphHost rsEngineHost(Highs* highs);
// The profiling steps of the simplex solves of handles on a
// HighsProfiling (registered with highs_rs_lph_register; HighsRunRust.cpp)
void rsSimplexProfiling(void* profiling, int code, HighsInt simplex_strategy,
                        int64_t arg);
// The option values of a handle from a HighsOptions (HighsOptionsRust.cpp)
void rsSyncOptions(highs_rs::LpHandle* h, const HighsOptions& options);
// formSimplexLpBasisAndFactor of `basis` for `lp` on a Highs object's
// engine (HighsRunRust.cpp)
HighsStatus rsFormBasis(HighsEngine& ekk, const HighsOptions& options,
                        HighsLp& lp, HighsBasis& basis,
                        const bool only_from_known_basis);

/// The simplex engine of a Highs object (HEkk's place, with the names of
/// HEkk's members that the Highs object uses): a Rust LP solver whose
/// host is the Highs object (its log, run clock and callbacks), on which
/// its LP runs are made (HighsRunRust.cpp). The scalars C++ reads are the
/// engine's in place.
struct HighsEngine {
  highs_rs::LpHandle* p;
  void* lps;
  highs_rs::HEkkShared& sh_;
  HighsSimplexStatus& status_;
  highs_rs::HEkkInfo& info_;
  // Copies of what the API returns by reference
  mutable HotStart hot_start_;
  mutable std::vector<double> primal_phase1_dual_;

  explicit HighsEngine(const highs_rs::LphHost& host)
      : p(highs_rs::highs_rs_lph_new_host(&host)),
        lps(highs_rs::highs_rs_lph_lps(p)),
        sh_(*highs_rs::highs_rs_lps_shared(lps)),
        status_(sh_.status),
        info_(sh_.info) {}
  ~HighsEngine() { highs_rs::highs_rs_lph_free(p); }
  HighsEngine(const HighsEngine&) = delete;
  HighsEngine& operator=(const HighsEngine&) = delete;

  void clear() { highs_rs::highs_rs_lph_ekk_clear(p); }
  void invalidate() { highs_rs::highs_rs_lph_ekk_invalidate(p); }
  void updateStatus(LpAction action) {
    highs_rs::highs_rs_lph_update_status(p, int(action));
  }
  void setLpName(const char* name, size_t len) {
    highs_rs::highs_rs_lph_set_lp_name(p, name, len);
  }
  HighsInt dualRayIndex() const { return sh_.dual_ray_index; }
  HighsInt dualRaySign() const { return sh_.dual_ray_sign; }
  // The simplex NLA's LP is `lp` (a view of it, valid until it changes)
  void setNlaPointersForLpAndScale(const HighsLp& lp);
  void btran(HVector& rhs, const double expected_density) {
    highs_rs::HVecCall v(rhs);
    highs_rs::highs_rs_lph_nla_solve(p, nullptr, v.get(), expected_density,
                                     true);
  }
  void ftran(HVector& rhs, const double expected_density) {
    highs_rs::HVecCall v(rhs);
    highs_rs::highs_rs_lph_nla_solve(p, nullptr, v.get(), expected_density,
                                     false);
  }
  HighsInt* basicIndex() const {
    HighsInt n;
    return highs_rs::highs_rs_lps_basic_index(lps, &n);
  }
  double computeBasisCondition(const HighsLp& lp, const bool exact,
                               const bool report) const;
  void putIterate() { highs_rs::highs_rs_lps_put_iterate(lps); }
  HighsStatus getIterate() {
    return highs_rs::highs_rs_lps_get_iterate(lps) ? HighsStatus::kOk
                                                   : HighsStatus::kError;
  }
  HighsBasis getHighsBasis(const HighsLp& use_lp) const;
  bool lpFactorRowCompatible(const HighsInt expected_num_row) const {
    return highs_rs::highs_rs_lph_factor_row_compatible(
               p, expected_num_row) == 1;
  }
  std::vector<double> rayValue(const bool primal) const {
    size_t n;
    const double* v = highs_rs::highs_rs_lps_ray_value(lps, primal, &n);
    return std::vector<double>(v, v + n);
  }
  void setRayValue(const bool primal, const std::vector<double>& value) {
    highs_rs::highs_rs_lps_set_ray_value(lps, primal, value.data(),
                                         value.size());
  }
  const HighsSimplexStats& getSimplexStats() const {
    return *highs_rs::highs_rs_lph_simplex_stats(p);
  }
  const HotStart& hotStart() const;
  const std::vector<double>& primalPhase1Dual() const;
};

#endif  // HIGHS_RUST

#endif  // LP_DATA_HIGHS_LP_HANDLE_H_
