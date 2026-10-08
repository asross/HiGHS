/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file ipm/IpxWrapperRust.cpp
 * @brief solveLpIpx and fillInIpxData done by Rust
 * (rust/src/lp_data/ipx_glue.rs): the options, the LP view, the timer, the
 * sizing of the solution and basis, and the IPX hooks
 */
#include "lp_data/HighsRust.h"

#ifdef HIGHS_RUST
#include <iostream>

#include "ipm/IpxWrapper.h"
#include "lp_data/HighsSolution.h"
#include "parallel/HighsParallel.h"

namespace {
struct RsIpxSolutionOut {
  RsMut<double> col_value, col_dual, row_value, row_dual;
  RsMut<uint8_t> col_status, row_status;
};

struct RsIpxHooks {
  void (*log)(const void* log_options, const char* msg);
  void (*print)(const char* msg);
  ipx::Int (*task_interrupt)(void* ctx);
  ipx::Int (*user_interrupt)(void* ctx, ipx::Int iter);
  void* ctx;
};

struct RsIpxOptions {
  RsLog log;
  const void* log_options;
  bool output_flag, log_to_console, timeless_log, run_centring;
  HighsInt log_dev_level, ipx_dualize_strategy, highs_analysis_level,
      ipm_iteration_limit, run_crossover, max_centring_steps;
  double primal_feasibility_tolerance, dual_feasibility_tolerance,
      ipm_optimality_tolerance, start_crossover_tolerance, kkt_tolerance,
      time_limit, centring_ratio_tolerance;
};

struct RsIpxHost {
  void* ctx;
  double (*timer_read)(void* ctx);
  void (*resize)(void* ctx, bool with_basis, RsIpxSolutionOut* out);
  RsIpxHooks hooks;
  RsLp lp;
  RsIpxOptions options;
  HighsInfoStruct* info;
  int* model_status;
  bool *value_valid, *dual_valid, *basis_valid, *basis_useful;
};

static_assert(sizeof(RsIpxOptions) == 112, "RsIpxOptions layout");
static_assert(sizeof(RsIpxHost) == 64 + sizeof(RsLp) + 112 + 48,
              "RsIpxHost layout");

struct RsIpxData {
  HighsInt num_col, num_row;
  double offset;
  RsMut<double> f[5];  // obj, col_lb, col_ub, ax, rhs
  RsMut<HighsInt> i[2];  // ap, ai
  RsMut<uint8_t> constraint_type;
};

struct IpxGlueCtx {
  const HighsLp& lp;
  HighsTimer& timer;
  HighsSolution& solution;
  HighsBasis& basis;
};

// As the hooks of ipx::LpSolver (ipx/lp_solver_rs.cc), with the callback
// as context
void ipxGlueLog(const void* log_options, const char* msg) {
  highsLogUser(*static_cast<const HighsLogOptions*>(log_options),
               HighsLogType::kInfo, "%s", msg);
}

void ipxGluePrint(const char* msg) { std::cout << msg; }

ipx::Int ipxGlueTaskInterrupt(void*) {
  try {
    HighsTaskExecutor::getThisWorkerDeque()->checkInterrupt();
  } catch (const HighsTask::Interrupt&) {
    return 1;
  }
  return 0;
}

ipx::Int ipxGlueUserInterrupt(void* ctx, ipx::Int ipm_iteration_count) {
  HighsCallback* callback = static_cast<HighsCallback*>(ctx);
  if (callback->user_callback && callback->active[kCallbackIpmInterrupt]) {
    callback->clearHighsCallbackOutput();
    callback->data_out.ipm_iteration_count = ipm_iteration_count;
    if (callback->callbackAction(kCallbackIpmInterrupt, "IPM interrupt"))
      return 1;
  }
  return 0;
}

double ipxGlueTimerRead(void* ctx) {
  return static_cast<IpxGlueCtx*>(ctx)->timer.read();
}

RsMut<uint8_t> ipxGlueStatus(std::vector<HighsBasisStatus>& s) {
  return {reinterpret_cast<uint8_t*>(s.data()), s.size()};
}

void ipxGlueResize(void* ctx, bool with_basis, RsIpxSolutionOut* out) {
  IpxGlueCtx& c = *static_cast<IpxGlueCtx*>(ctx);
  HighsSolution& s = c.solution;
  s.col_value.resize(c.lp.num_col_);
  s.row_value.resize(c.lp.num_row_);
  s.col_dual.resize(c.lp.num_col_);
  s.row_dual.resize(c.lp.num_row_);
  if (with_basis) {
    c.basis.col_status.resize(c.lp.num_col_);
    c.basis.row_status.resize(c.lp.num_row_);
  }
  out->col_value = rsMut(s.col_value);
  out->col_dual = rsMut(s.col_dual);
  out->row_value = rsMut(s.row_value);
  out->row_dual = rsMut(s.row_dual);
  out->col_status = with_basis ? ipxGlueStatus(c.basis.col_status)
                               : RsMut<uint8_t>{nullptr, 0};
  out->row_status = with_basis ? ipxGlueStatus(c.basis.row_status)
                               : RsMut<uint8_t>{nullptr, 0};
}
}  // namespace

extern "C" {
int highs_rs_solve_lp_ipx(const RsIpxHost* host);
void* highs_rs_fill_in_ipx_data(const RsLp* lp);
void highs_rs_ipx_data_get(void* d, RsIpxData* out);
void highs_rs_ipx_data_free(void* d);
}

HighsStatus solveLpIpx(const HighsOptions& options, HighsTimer& timer,
                       const HighsLp& lp, HighsBasis& highs_basis,
                       HighsSolution& highs_solution,
                       HighsModelStatus& model_status, HighsInfo& highs_info,
                       HighsCallback& callback) {
  IpxGlueCtx ctx{lp, timer, highs_solution, highs_basis};
  RsIpxHost h;
  h.ctx = &ctx;
  h.timer_read = ipxGlueTimerRead;
  h.resize = ipxGlueResize;
  h.hooks = {ipxGlueLog, ipxGluePrint, ipxGlueTaskInterrupt,
             ipxGlueUserInterrupt, &callback};
  h.lp = rsLp(lp);
  RsIpxOptions& o = h.options;
  o.log = rsLog(options.log_options);
  o.log_options = &options.log_options;
  o.output_flag = options.output_flag;
  o.log_to_console = options.log_to_console;
  o.timeless_log = options.timeless_log;
  o.run_centring = options.run_centring;
  o.log_dev_level = options.log_dev_level;
  o.ipx_dualize_strategy = options.ipx_dualize_strategy;
  o.highs_analysis_level = options.highs_analysis_level;
  o.ipm_iteration_limit = options.ipm_iteration_limit;
  o.run_crossover = options.run_crossover == kHighsOnString    ? 1
                    : options.run_crossover == kHighsOffString ? 0
                                                               : -1;
  o.max_centring_steps = options.max_centring_steps;
  o.primal_feasibility_tolerance = options.primal_feasibility_tolerance;
  o.dual_feasibility_tolerance = options.dual_feasibility_tolerance;
  o.ipm_optimality_tolerance = options.ipm_optimality_tolerance;
  o.start_crossover_tolerance = options.start_crossover_tolerance;
  o.kkt_tolerance = options.kkt_tolerance;
  o.time_limit = options.time_limit;
  o.centring_ratio_tolerance = options.centring_ratio_tolerance;
  h.info = static_cast<HighsInfoStruct*>(&highs_info);
  h.model_status = reinterpret_cast<int*>(&model_status);
  h.value_valid = &highs_solution.value_valid;
  h.dual_valid = &highs_solution.dual_valid;
  h.basis_valid = &highs_basis.valid;
  h.basis_useful = &highs_basis.useful;
  const int status = highs_rs_solve_lp_ipx(&h);
  // A cancelled task: IPX stopped, rethrow as ipx::LpSolver::Solve does
  if (status == 2) throw HighsTask::Interrupt();
  return HighsStatus(status);
}

void fillInIpxData(const HighsLp& lp, ipx::Int& num_col, ipx::Int& num_row,
                   double& offset, std::vector<double>& obj,
                   std::vector<double>& col_lb, std::vector<double>& col_ub,
                   std::vector<ipx::Int>& Ap, std::vector<ipx::Int>& Ai,
                   std::vector<double>& Ax, std::vector<double>& rhs,
                   std::vector<char>& constraint_type) {
  const RsLp v = rsLp(lp);
  void* d = highs_rs_fill_in_ipx_data(&v);
  RsIpxData r;
  highs_rs_ipx_data_get(d, &r);
  num_col = r.num_col;
  num_row = r.num_row;
  offset = r.offset;
  obj.assign(r.f[0].ptr, r.f[0].ptr + r.f[0].len);
  col_lb.assign(r.f[1].ptr, r.f[1].ptr + r.f[1].len);
  col_ub.assign(r.f[2].ptr, r.f[2].ptr + r.f[2].len);
  Ax.assign(r.f[3].ptr, r.f[3].ptr + r.f[3].len);
  rhs.assign(r.f[4].ptr, r.f[4].ptr + r.f[4].len);
  Ap.assign(r.i[0].ptr, r.i[0].ptr + r.i[0].len);
  Ai.assign(r.i[1].ptr, r.i[1].ptr + r.i[1].len);
  constraint_type.assign(r.constraint_type.ptr,
                         r.constraint_type.ptr + r.constraint_type.len);
  highs_rs_ipx_data_free(d);
}
#endif
