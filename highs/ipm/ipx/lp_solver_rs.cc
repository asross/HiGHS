// With HIGHS_RUST, ipx::LpSolver wraps the Rust port of IPX
// (rust/src/ipx/ffi.rs). Logging and the interrupt checks of HiGHS (task
// cancellation and the user callback) are called back from Rust through
// the hooks below.

#include <cassert>
#include <iostream>

#include "io/HighsIO.h"
#include "ipm/ipx/info.h"
#include "ipm/ipx/lp_solver.h"
#include "parallel/HighsParallel.h"

extern "C" {
struct IpxRsHooks {
  void (*log)(const void* log_options, const char* msg);
  void (*print)(const char* msg);
  ipxint (*task_interrupt)(void* ctx);
  ipxint (*user_interrupt)(void* ctx, ipxint iter);
  void* ctx;
};

void* ipx_rs_new();
void ipx_rs_free(void* p);
void ipx_rs_set_hooks(void* p, const IpxRsHooks* hooks);
void ipx_rs_set_parameters(void* p, const ipx_parameters* params);
void ipx_rs_get_parameters(void* p, ipx_parameters* params);
void ipx_rs_set_timer_offset(void* p, double offset);
ipxint ipx_rs_load_model(void* p, ipxint num_var, double offset,
                         const double* obj, const double* lb,
                         const double* ub, ipxint num_constr,
                         const ipxint* Ap, const ipxint* Ai, const double* Ax,
                         const double* rhs, const char* constr_type);
ipxint ipx_rs_load_ipm_starting_point(void* p, const double* x,
                                      const double* xl, const double* xu,
                                      const double* slack, const double* y,
                                      const double* zl, const double* zu);
ipxint ipx_rs_solve(void* p);
void ipx_rs_get_info(void* p, ipx_info* info);
ipxint ipx_rs_cancelled(void* p);
ipxint ipx_rs_get_interior_solution(void* p, double* x, double* xl,
                                    double* xu, double* slack, double* y,
                                    double* zl, double* zu);
ipxint ipx_rs_get_basic_solution(void* p, double* x, double* slack,
                                 double* y, double* z, ipxint* cbasis,
                                 ipxint* vbasis);
void ipx_rs_clear_model(void* p);
void ipx_rs_clear_ipm_starting_point(void* p);
ipxint ipx_rs_crossover_from_starting_point(void* p, const double* x,
                                            const double* slack,
                                            const double* y, const double* z);
ipxint ipx_rs_get_iterate(void* p, double* x, double* y, double* zl,
                          double* zu, double* xl, double* xu);
ipxint ipx_rs_get_basis(void* p, ipxint* cbasis, ipxint* vbasis);
ipxint ipx_rs_get_kkt_matrix(void* p, ipxint* AIp, ipxint* AIi, double* AIx,
                             double* g);
ipxint ipx_rs_symbolic_invert(void* p, ipxint* rowcounts, ipxint* colcounts);
const char* ipx_rs_status_string(ipxint status);
}

// The Rust structs mirror the C layouts (rust/src/ipx/mod.rs)
static_assert(sizeof(ipx_parameters) == 240, "ipx_parameters layout");
static_assert(sizeof(ipx_info) == 464, "ipx_info layout");

namespace ipx {

static void hookLog(const void* log_options, const char* msg) {
  highsLogUser(*static_cast<const HighsLogOptions*>(log_options),
               HighsLogType::kInfo, "%s", msg);
}

static void hookPrint(const char* msg) { std::cout << msg; }

// Control::InterruptCheck: a cancelled task throws HighsTask::Interrupt,
// which must not unwind through Rust; Rust stops the solver and Solve()
// rethrows.
static ipxint hookTaskInterrupt(void* /*ctx*/) {
  try {
    HighsTaskExecutor::getThisWorkerDeque()->checkInterrupt();
  } catch (const HighsTask::Interrupt&) {
    return 1;
  }
  return 0;
}

static ipxint hookUserInterrupt(void* ctx, ipxint ipm_iteration_count) {
  // The callback should not be null, since that indicates that it's not
  // been set
  HighsCallback* callback = static_cast<LpSolver*>(ctx)->callback_;
  assert(callback);
  if (callback && callback->user_callback &&
      callback->active[kCallbackIpmInterrupt]) {
    callback->clearHighsCallbackOutput();
    callback->data_out.ipm_iteration_count = ipm_iteration_count;
    if (callback->callbackAction(kCallbackIpmInterrupt, "IPM interrupt"))
      return 1;
  }
  return 0;
}

LpSolver::LpSolver() : rs_(ipx_rs_new()) {
  const IpxRsHooks hooks = {hookLog, hookPrint, hookTaskInterrupt,
                            hookUserInterrupt, this};
  ipx_rs_set_hooks(rs_, &hooks);
}

LpSolver::~LpSolver() { ipx_rs_free(rs_); }

void LpSolver::RethrowInterrupt() {
  if (ipx_rs_cancelled(rs_)) throw HighsTask::Interrupt();
}

Int LpSolver::LoadModel(Int num_var, const double offset, const double* obj,
                        const double* lb, const double* ub, Int num_constr,
                        const Int* Ap, const Int* Ai, const double* Ax,
                        const double* rhs, const char* constr_type) {
  return ipx_rs_load_model(rs_, num_var, offset, obj, lb, ub, num_constr, Ap,
                           Ai, Ax, rhs, constr_type);
}

Int LpSolver::LoadIPMStartingPoint(const double* x, const double* xl,
                                   const double* xu, const double* slack,
                                   const double* y, const double* zl,
                                   const double* zu) {
  return ipx_rs_load_ipm_starting_point(rs_, x, xl, xu, slack, y, zl, zu);
}

Int LpSolver::Solve() {
  Int status = ipx_rs_solve(rs_);
  RethrowInterrupt();
  return status;
}

Info LpSolver::GetInfo() const {
  Info info;
  ipx_rs_get_info(rs_, &info);
  return info;
}

Int LpSolver::GetInteriorSolution(double* x, double* xl, double* xu,
                                  double* slack, double* y, double* zl,
                                  double* zu) const {
  return ipx_rs_get_interior_solution(rs_, x, xl, xu, slack, y, zl, zu);
}

Int LpSolver::GetBasicSolution(double* x, double* slack, double* y, double* z,
                               Int* cbasis, Int* vbasis) const {
  return ipx_rs_get_basic_solution(rs_, x, slack, y, z, cbasis, vbasis);
}

Parameters LpSolver::GetParameters() const {
  Parameters parameters;
  ipx_rs_get_parameters(rs_, &parameters);
  return parameters;
}

void LpSolver::SetParameters(Parameters new_parameters) {
  ipx_rs_set_parameters(rs_, &new_parameters);
}

void LpSolver::SetCallback(HighsCallback* callback) { callback_ = callback; }

void LpSolver::ClearModel() { ipx_rs_clear_model(rs_); }

void LpSolver::ClearIPMStartingPoint() {
  ipx_rs_clear_ipm_starting_point(rs_);
}

Int LpSolver::CrossoverFromStartingPoint(const double* x_start,
                                         const double* slack_start,
                                         const double* y_start,
                                         const double* z_start) {
  Int status = ipx_rs_crossover_from_starting_point(rs_, x_start, slack_start,
                                                    y_start, z_start);
  RethrowInterrupt();
  return status;
}

Int LpSolver::GetIterate(double* x, double* y, double* zl, double* zu,
                         double* xl, double* xu) {
  return ipx_rs_get_iterate(rs_, x, y, zl, zu, xl, xu);
}

Int LpSolver::GetBasis(Int* cbasis, Int* vbasis) {
  return ipx_rs_get_basis(rs_, cbasis, vbasis);
}

Int LpSolver::GetKKTMatrix(Int* AIp, Int* AIi, double* AIx, double* g) {
  return ipx_rs_get_kkt_matrix(rs_, AIp, AIi, AIx, g);
}

Int LpSolver::SymbolicInvert(Int* rowcounts, Int* colcounts) {
  return ipx_rs_symbolic_invert(rs_, rowcounts, colcounts);
}

void LpSolver::setTimerOffset(const double offset) {
  ipx_rs_set_timer_offset(rs_, offset);
}

std::string StatusString(Int status) { return ipx_rs_status_string(status); }

}  // namespace ipx
