/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file mip/HighsMipRust.h
 * @brief The C++ objects of the MIP solver as Rust drives them
 * (rust/src/mip/glue.rs mirrors these repr(C) structs): the operations on
 * domains, LP relaxations, searches, workers and the solver, and the
 * solver's data read in place
 */
#ifndef MIP_HIGHS_MIP_RUST_H_
#define MIP_HIGHS_MIP_RUST_H_

#include "HConfig.h"

#ifdef HIGHS_RUST
#include <cstdint>
#include <vector>

#include "lp_data/HighsRust.h"
#include "mip/HighsDomainChange.h"
#include "mip/HighsDomain.h"
#include "mip/HighsLpRelaxation.h"
#include "mip/HighsRsSpan.h"

class HighsMipSolver;
class HighsMipWorker;

namespace highs_rs {
struct Search;
struct NodeQueue;
struct Heuristics;
struct CliqueTable;
struct RedcostFixing;
struct ConcurrentPool;
struct MipVecs;

// glue.rs SearchParts
struct MipSearchParts {
  void* cpp;
  Search* rs;
  void* ps;
  const NodeQueue* nq;
  void* localdom;
};

// glue.rs SubMipSpec
struct MipSubMipSpec {
  void* lp;
  const double* col_lower;
  const double* col_upper;
  const double* start_cols;
  const double* start_rows;
  HighsInt num_start_rows;
  HighsInt mip_max_leaves;
  HighsInt mip_max_nodes;
  HighsInt mip_max_stall_nodes;
  HighsInt mip_pscost_minreliable;
  double time_limit;
  double objective_bound;
  double mip_rel_gap;
  double mip_abs_gap;
  double mip_heuristic_effort;
  int heur_flags;
  bool presolve;
  bool output_flag;
  bool mip_detect_symmetry;
  const ConcurrentPool* lns_target;
};

// glue.rs SubMipResult
struct MipSubMipResult {
  int termination_status;
  int model_status;
  int64_t node_count;
  int64_t total_lp_iterations;
  int64_t total_repair_lp;
  int64_t total_repair_lp_feasible;
  int64_t total_repair_lp_iterations;
  int max_submip_level;
  bool has_solution;
};

// glue.rs HeurStats (= HighsMipWorker::HeurStatistics)
struct MipHeurStats {
  int64_t total_repair_lp;
  int64_t total_repair_lp_feasible;
  int64_t total_repair_lp_iterations;
  int64_t lp_iterations;
  double success_observations;
  HighsInt num_success_observations;
  double infeas_observations;
  HighsInt num_infeas_observations;
  HighsInt max_submip_level;
  int termination_status;
};

// glue.rs WorkerData
struct MipWorkerData {
  double* upper_limit;
  void* heur;
  void* randgen;
  void* globaldom;
  void* lp;
  double* upper_bound;
  double* optimality_limit;
  void* state;
};

// glue.rs MipOptions
struct MipOptions {
  double objective_bound;
  double objective_target;
  double mip_abs_gap;
  double mip_rel_gap;
  double mip_feasibility_tolerance;
  double time_limit;
  double mip_min_logging_interval;
  HighsInt mip_max_nodes;
  HighsInt mip_max_leaves;
  HighsInt mip_max_improving_sols;
  bool output_flag;
  bool timeless_log;
  bool run_zi_round;
  bool run_shifting;
  bool run_graph_lns;
  bool run_root_reduced_cost;
  bool run_rens;
  bool run_rins;
  bool mip_allow_restart;
  bool presolve_off;
  bool run_feasibility_jump;
  bool output_flag_option;
  HighsInt mip_max_stall_nodes;
  double small_matrix_value;
  double mip_heuristic_effort;
  HighsInt mip_report_level;
  HighsInt restart_presolve_reduction_limit;
  HighsInt presolve_reduction_limit;
  bool mip_detect_symmetry;
  bool mip_improving_solution_save;
  bool mip_concurrent_crossover;
};

// glue.rs OrigModel
struct MipOrigModel {
  HighsInt num_col;
  HighsInt num_row;
  double offset;
  const std::vector<double>* col_cost;
  const std::vector<double>* col_lower;
  const std::vector<double>* col_upper;
  const std::vector<double>* row_lower;
  const std::vector<double>* row_upper;
  const std::vector<HighsVarType>* integrality;
  const std::vector<HighsInt>* a_start;
  const std::vector<HighsInt>* a_index;
  const std::vector<double>* a_value;
};

// glue.rs SolutionPtrs
struct MipSolutionPtrs {
  double* objective;
  double* bound_violation;
  double* integrality_violation;
  double* row_violation;
  const std::vector<double>* solution;
};

// glue.rs ScratchView
struct MipScratchView {
  const double* col;
  HighsInt ncol;
  const double* row;
  HighsInt nrow;
};

// setup.rs CallbackOut
struct MipCallbackOut {
  double running_time;
  double objective_function_value;
  int64_t mip_node_count;
  int64_t mip_total_lp_iterations;
  double mip_primal_bound;
  double mip_dual_bound;
  double mip_gap;
  int external_solution_query_origin;
  bool clear_output;
  bool clear_input;
  int solution;
};

// glue.rs MipData: HighsMipSolverData and its model in place
struct MipData {
  const void* mipsolver;
  RsLog log;
  int num_col;
  int num_row;
  bool colwise;
  bool minimize;
  bool orig_maximize;
  bool submip;
  bool concurrent_helper;
  bool root_presolve_only;
  double offset;
  const std::vector<HighsInt>* a_start;
  const std::vector<HighsInt>* a_index;
  const std::vector<double>* a_value;
  const std::vector<double>* col_cost;
  const std::vector<double>* col_lower;
  const std::vector<double>* col_upper;
  const std::vector<double>* row_lower;
  const std::vector<double>* row_upper;
  const std::vector<HighsVarType>* integrality;
  const HighsRsArray<HighsInt>* ar_start;
  const HighsRsArray<HighsInt>* ar_index;
  const HighsRsArray<double>* ar_value;
  const HighsRsArray<HighsInt>* uplocks;
  const HighsRsArray<HighsInt>* downlocks;
  const HighsRsArray<HighsInt>* integer_cols;
  const HighsRsArray<HighsInt>* integral_cols;
  const HighsRsArray<HighsInt>* continuous_cols;
  const HighsRsArray<double>* rootlpsol;
  const HighsRsArray<double>* firstlpsol;
  const HighsRsArray<double>* analytic_center;
  const HighsRsArray<double>* incumbent;
  void* scalars;  // HighsMipScalars
  const CliqueTable* clique;
  const RedcostFixing* redcost;
  NodeQueue* nodequeue;
  void* globaldom;
  void* lp;
  int* modelstatus;
  MipSolutionPtrs solution;
  MipOrigModel orig;
  MipOptions opts;
  const ConcurrentPool* helper_pool;
  const ConcurrentPool* lns_target;
  const Heuristics* heur;
  MipVecs* vecs;
};

// glue.rs CMipFns
struct MipFns {
  void* (*dom_copy)(void*);
  void (*dom_free)(void*);
  void (*dom_assign)(void*, void*);
  void (*dom_bounds)(void*, const double**, const double**);
  void (*dom_change_bound)(void*, HighsDomainChange, HighsDomain::Reason);
  void (*dom_fix_col)(void*, HighsInt, double, HighsDomain::Reason);
  bool (*dom_propagate)(void*);
  bool (*dom_infeasible)(void*);
  HighsDomainChange (*dom_backtrack)(void*);
  void (*dom_conflict_analysis)(void*, void*);
  const HighsDomainChange* (*dom_stack)(void*, HighsInt*);
  HighsInt (*dom_branch_depth)(void*);
  void (*dom_clear_changed_cols)(void*);
  void (*dom_clear_pool_propagation)(void*);
  HighsInt (*dom_num_changed_cols)(void*);
  void* (*lp_copy)(void*, void*);
  void* (*lp_new)(void*, void*);
  void (*lp_free)(void*);
  LpShared* (*lp_shared)(void*);
  void (*lp_set_iteration_limit)(void*, HighsInt);
  void (*lp_change_cols_bounds)(void*, const double*, const double*);
  void (*lp_change_col_bounds)(void*, HighsInt, double, double);
  void (*lp_change_cols_cost)(void*, const HighsInt*, const double*);
  void (*lp_set_option)(void*, int);
  void (*lp_set_root_basis)(void*, const char*);
  int (*lp_resolve)(void*, void*);
  const double* (*lp_solution)(void*, int, HighsInt*);
  void (*lp_set_objective_limit)(void*, double);
  void (*lp_flush_domain)(void*, void*);
  void (*lp_remove_obsolete_rows)(void*, bool);
  void (*lp_infeasible_conflict)(void*, void*, void*);
  bool (*lp_put_iterate)(void*);
  void (*lp_get_iterate)(void*);
  void (*search_new)(void*, MipSearchParts*);
  void (*search_free)(void*);
  void (*search_set_lp)(void*, void*);
  bool (*check_limits)(void*);
  void (*update_lower_bound)(void*, double);
  bool (*parallel_lock_active)(void*);
  HighsInt (*num_workers)(void*);
  void (*worker_view)(void*, MipWorkerData*);
  void (*sub_mip)(void*, void*, const MipSubMipSpec*, MipSubMipResult*,
                  double*);
  double (*op)(void*, int, void*, int64_t, double);
  void (*scratch_solution)(void*, const double*, HighsInt, MipScratchView*);
  void (*refill)(void*, MipData*);
  void* (*master_worker)(void*);
  void (*run_process_nodes)(void*, const HighsInt*, HighsInt, const void*);
  void (*set_cleanup_result)(void*, const void*);
  const char* (*model_name)(void*, HighsInt*);
  HighsInt (*max_submip_level)(void*);
  void* (*helper_new)(void*, double);
  void (*helper_run)(void*, const ConcurrentPool*);
  void (*add_root_cut)(void*, const HighsInt*, const double*, HighsInt, double,
                       bool);
  const void* (*vec_ptr)(void*, int, HighsInt*);
  void (*set_basis)(void*, int, const uint8_t*, HighsInt, const uint8_t*,
                    HighsInt, bool, bool, bool);
  bool (*callback)(void*, int, const MipCallbackOut*, const char*, HighsInt);
  void* (*worker)(void*, HighsInt);
  void (*worker_scratch)(void*, void*, const double*, HighsInt,
                         MipScratchView*);
  bool (*repair_lp)(void*, const double*, const double*, double, double, bool,
                    int64_t*);
};

// The functions (HighsPrimalHeuristics.cpp) and the solver's data
const MipFns* mipFns();
MipData mipData(const HighsMipSolver& mipsolver);
// the driver's operations (HighsMipSolver.cpp)
double mipDriverOp(void* m, int which, void* w, int64_t i, double x);
void* mipMasterWorker(void* m);
void mipRunProcessNodes(void* m, const HighsInt* idx, HighsInt n,
                        const void* ctx);
void mipSetCleanupResult(void* m, const void* r);
const char* mipModelName(void* m, HighsInt* n);
HighsInt mipMaxSubmipLevel(void* m);
// the setup's operations (HighsMipSolverData.cpp)
double mipSetupOp(void* m, int which, void* w, int64_t i, double x);
const void* mipVecPtr(void* m, int which, HighsInt* n);
void mipSetBasis(void* m, int which, const uint8_t* col, HighsInt ncol,
                 const uint8_t* row, HighsInt nrow, bool valid, bool alien,
                 bool useful);
bool mipCallback(void* m, int type, const MipCallbackOut* out,
                 const char* message, HighsInt len);
void* mipWorker(void* m, HighsInt k);
void mipWorkerScratch(void* m, void* w, const double* x, HighsInt n,
                      MipScratchView* v);

extern "C" {
void highs_rs_concurrent_lns_set_root_cuts(const ConcurrentPool* pool,
                                           const HighsInt* start,
                                           HighsInt num_cuts,
                                           const HighsInt* index,
                                           const double* value, HighsInt nnz,
                                           const double* rhs,
                                           const uint8_t* integral);
Heuristics* highs_rs_heur_new(HighsInt seed);
void highs_rs_heur_free(Heuristics* h);
void highs_rs_heur_run(Heuristics* h, const MipFns* f, const MipData* m,
                       void* worker, int which, const double* x, HighsInt n);
HighsInt highs_rs_heur_crossover(Heuristics* h, const MipFns* f,
                                 const MipData* m, void* worker,
                                 const double* other, HighsInt n,
                                 double other_objective, double time_cap);
bool highs_rs_heur_rounding(Heuristics* h, const MipFns* f, const MipData* m,
                            void* worker, const double* x, const double* y,
                            HighsInt n, int solution_source);
void highs_rs_heur_add_observations(Heuristics* h, double s, HighsInt ns,
                                    double i, HighsInt ni);
HighsInt highs_rs_heur_random(Heuristics* h, HighsInt sup);
bool highs_rs_mip_check_limits(const MipFns* f, const MipData* m,
                               int64_t node_offset);
double highs_rs_mip_limits_to_gap(const MipFns* f, const MipData* m,
                                  double lower, double upper, double* lb,
                                  double* ub);
double highs_rs_mip_new_upper_limit(const MipFns* f, const MipData* m,
                                    double ub, double abs_gap, double rel_gap);
void highs_rs_mip_limits_to_bounds(const MipFns* f, const MipData* m,
                                   double* dual_bound, double* primal_bound,
                                   double* gap);
void highs_rs_mip_update_lower_bound(const MipFns* f, const MipData* m,
                                     double lb, bool check_bound_change,
                                     bool check_prev_data);
void highs_rs_mip_update_pdi(const MipFns* f, const MipData* m,
                             double from_lb, double to_lb, double from_ub,
                             double to_ub, bool check_bound_change,
                             bool check_prev_data);
double highs_rs_mip_query(const MipFns* f, const MipData* m, int which);
void highs_rs_mip_print_display_line(const MipFns* f, const MipData* m,
                                     int source);
bool highs_rs_mip_solution(const MipFns* f, const MipData* m, int which,
                           const double* sol, HighsInt n, int source);
bool highs_rs_mip_add_incumbent(const MipFns* f, const MipData* m,
                                const double* sol, HighsInt n, double obj,
                                int source, bool print_display_line,
                                bool is_user_solution);
double highs_rs_mip_transform(const MipFns* f, const MipData* m,
                              const double* sol, HighsInt n, bool store);
int highs_rs_mip_evaluate_root_lp(const MipFns* f, const MipData* m,
                                  void* worker);
void highs_rs_mip_evaluate_root_node(const MipFns* f, MipData* m,
                                     void* worker);
void highs_rs_mip_run(const MipFns* f, MipData* m);
void highs_rs_mip_cleanup_solve(const MipFns* f, const MipData* m);
void highs_rs_mip_process_node(const MipFns* f, const void* ctx, int i);
void highs_rs_mip_setup_domain_propagation(const MipFns* f, const MipData* m);
bool highs_rs_worker_solution(const MipFns* f, const MipData* m, void* w,
                              const double* sol, HighsInt n, double obj,
                              int source, bool try_solution);
void highs_rs_mip_presolve_only(const MipFns* f, const MipData* m,
                                HighsInt limit);
void highs_rs_heur_graph_lns(Heuristics* h, const MipFns* f, const MipData* m,
                             void* worker, const double* x, HighsInt n,
                             bool deep, int64_t max_lp_iters);
}
}  // namespace highs_rs

#endif  // HIGHS_RUST
#endif
