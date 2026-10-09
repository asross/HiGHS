/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file pdlp/CupdlpWrapperRs.cpp
 * @brief With HIGHS_RUST, solveLpCupdlp calls the Rust port of cuPDLP-C
 * (rust/src/pdlp). This file reads the HiGHS options (as
 * getUserParamsFromOptions of CupdlpWrapper.cpp) and maps the
 * termination code to a model status; Rust builds the cuPDLP-C LP, solves
 * it and returns the HiGHS solution. Its output goes through printf, as
 * cupdlp_printf does.
 */
#include <cstdio>

#include "pdlp/CupdlpWrapper.h"

extern "C" {
// rust/src/pdlp/ffi.rs
struct PdlpRsLp {
  int num_col;
  int num_row;
  const int* a_start;
  const int* a_index;
  const double* a_value;
  const double* col_cost;
  const double* col_lower;
  const double* col_upper;
  const double* row_lower;
  const double* row_upper;
  double offset;
  double sense;
};

// rust/src/pdlp/mod.rs
struct PdlpRsParams {
  double primal_tol;
  double dual_tol;
  double gap_tol;
  double time_lim;
  int iter_lim;
  int log_level;
  int scaling;
  int line_search;
  int restart;
};

int pdlp_rs_solve(const PdlpRsLp* lp, const PdlpRsParams* params,
                  void (*print)(const char*), double* col_value,
                  double* col_dual, double* row_value, double* row_dual,
                  int* value_valid, int* dual_valid, int* num_iter);
}

static_assert(sizeof(HighsInt) == sizeof(int),
              "the Rust cuPDLP-C takes 32-bit indices");

static void pdlpPrint(const char* msg) { printf("%s", msg); }

// getUserParamsFromOptions of CupdlpWrapper.cpp
static PdlpRsParams getParamsFromOptions(const HighsOptions& options) {
  PdlpRsParams params;
  params.iter_lim = cupdlp_int(options.pdlp_iteration_limit > kHighsIInf32
                                   ? kHighsIInf32
                                   : options.pdlp_iteration_limit);
  params.log_level = getCupdlpLogLevel(options);
  params.scaling = (options.pdlp_features_off & kPdlpScalingOff) == 0 ? 1 : 0;
  if (params.scaling == 0)
    highsLogUser(options.log_options, HighsLogType::kInfo,
                 "PDLP: Scaling off\n");
  const bool adaptive_linesearch =
      (options.pdlp_features_off & kPdlpAdaptiveStepSizeOff) == 0;
  params.line_search =
      adaptive_linesearch ? PDHG_ADAPTIVE_LINESEARCH : PDHG_FIXED_LINESEARCH;
  if (!adaptive_linesearch)
    highsLogUser(options.log_options, HighsLogType::kInfo,
                 "PDLP: Adaptive line search off\n");
  params.primal_tol = options.primal_feasibility_tolerance;
  params.dual_tol = options.dual_feasibility_tolerance;
  params.gap_tol = options.pdlp_optimality_tolerance;
  if (options.kkt_tolerance != kDefaultKktTolerance) {
    params.primal_tol = options.kkt_tolerance;
    params.dual_tol = options.kkt_tolerance;
    params.gap_tol = options.kkt_tolerance;
  }
  // As in CupdlpWrapper.cpp: the HiGHS time limit, not the time remaining
  params.time_lim = options.time_limit;
  int restart_on = (options.pdlp_features_off & kPdlpRestartOff) == 0 ? 1 : 0;
  if (options.pdlp_cupdlpc_restart_method == 0) restart_on = 0;
  params.restart = restart_on;
  if (restart_on == 0)
    highsLogUser(options.log_options, HighsLogType::kInfo,
                 "PDLP: Restart off\n");
  return params;
}

// The parameters and print function of solveLpCupdlp (logging as
// getUserParamsFromOptions does)
struct RsPdlpTemplate {
  PdlpRsParams params;
  void (*print)(const char*);
};

void rsPdlpTemplate(const HighsOptions& options, void* out) {
  RsPdlpTemplate& t = *static_cast<RsPdlpTemplate*>(out);
  t.params = getParamsFromOptions(options);
  t.print = pdlpPrint;
}

// From cupdlp_utils.c, for the restarts of HiPDLP (hipdlp/restart.cc)
void debugPdlpRestartLog(FILE* file, const int iter_num,
                         const double current_score,
                         const double average_score) {
  if (!file) return;
  fprintf(file,
          "Restart at iter %6d: Current Score = %.6g, Average Score = %.6g\n",
          iter_num, current_score, average_score);
}

cupdlp_int getCupdlpLogLevel(const HighsOptions& options) {
  if (options.output_flag) return options.log_dev_level ? 2 : 1;
  return 0;
}
