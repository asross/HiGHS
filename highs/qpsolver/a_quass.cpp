/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "qpsolver/a_quass.hpp"

#include "qpsolver/a_asm.hpp"
#include "qpsolver/feasibility_bounded.hpp"
#include "qpsolver/feasibility_highs.hpp"

static QpAsmStatus quass2highs(Instance& instance, Settings& settings,
                               Statistics& stats,
                               QpModelStatus& qp_model_status,
                               QpSolution& qp_solution,
                               HighsModelStatus& highs_model_status,
                               HighsBasis& highs_basis,
                               HighsSolution& highs_solution) {
  settings.qp_model_status_log.fire(qp_model_status);
  QpAsmStatus qp_asm_return_status = QpAsmStatus::kError;
  switch (qp_model_status) {
    case QpModelStatus::kOptimal:
      highs_model_status = HighsModelStatus::kOptimal;
      qp_asm_return_status = QpAsmStatus::kOk;
      break;
    case QpModelStatus::kUnbounded:
      highs_model_status = HighsModelStatus::kUnbounded;
      qp_asm_return_status = QpAsmStatus::kOk;
      break;
    case QpModelStatus::kInfeasible:
      highs_model_status = HighsModelStatus::kInfeasible;
      qp_asm_return_status = QpAsmStatus::kOk;
      break;
    case QpModelStatus::kIterationLimit:
      highs_model_status = HighsModelStatus::kIterationLimit;
      qp_asm_return_status = QpAsmStatus::kWarning;
      break;
    case QpModelStatus::kTimeLimit:
      highs_model_status = HighsModelStatus::kTimeLimit;
      qp_asm_return_status = QpAsmStatus::kWarning;
      break;
    case QpModelStatus::kInterrupt:
      highs_model_status = HighsModelStatus::kInterrupt;
      qp_asm_return_status = QpAsmStatus::kWarning;
      break;
    case QpModelStatus::kUndetermined:
      highs_model_status = HighsModelStatus::kSolveError;
      qp_asm_return_status = QpAsmStatus::kError;
      return QpAsmStatus::kError;
    case QpModelStatus::kLargeNullspace:
      highs_model_status = HighsModelStatus::kSolveError;
      return QpAsmStatus::kError;
    case QpModelStatus::kError:
      highs_model_status = HighsModelStatus::kSolveError;
      return QpAsmStatus::kError;
    case QpModelStatus::kNotset:
      highs_model_status = HighsModelStatus::kNotset;
      return QpAsmStatus::kError;
    default:
      highs_model_status = HighsModelStatus::kNotset;
      return QpAsmStatus::kError;
  }

  assert(qp_asm_return_status != QpAsmStatus::kError);
  // extract variable values
  highs_solution.col_value.resize(instance.num_var);
  highs_solution.col_dual.resize(instance.num_var);
  for (HighsInt iCol = 0; iCol < instance.num_var; iCol++) {
    highs_solution.col_value[iCol] = qp_solution.primal.value[iCol];
    highs_solution.col_dual[iCol] =
        instance.sense * qp_solution.dualvar.value[iCol];
  }
  // extract constraint activity
  highs_solution.row_value.resize(instance.num_con);
  highs_solution.row_dual.resize(instance.num_con);
  // Negate the vector and Hessian
  for (HighsInt iRow = 0; iRow < instance.num_con; iRow++) {
    highs_solution.row_value[iRow] = qp_solution.rowactivity.value[iRow];
    highs_solution.row_dual[iRow] =
        instance.sense * qp_solution.dualcon.value[iRow];
  }
  highs_solution.value_valid = true;
  highs_solution.dual_valid = true;

  // extract basis status
  highs_basis.col_status.resize(instance.num_var);
  highs_basis.row_status.resize(instance.num_con);

  const bool debug_report =
      false;  // instance.num_var + instance.num_con < 100;
  for (HighsInt i = 0; i < instance.num_var; i++) {
    if (debug_report)
      printf("Column %2d: status %s\n", int(i),
             qpBasisStatusToString(qp_solution.status_var[i]).c_str());
    if (qp_solution.status_var[i] == BasisStatus::kActiveAtLower) {
      highs_basis.col_status[i] = HighsBasisStatus::kLower;
    } else if (qp_solution.status_var[i] == BasisStatus::kActiveAtUpper) {
      highs_basis.col_status[i] = HighsBasisStatus::kUpper;
    } else if (qp_solution.status_var[i] == BasisStatus::kInactiveInBasis) {
      highs_basis.col_status[i] = HighsBasisStatus::kNonbasic;
    } else {
      assert(qp_solution.status_var[i] == BasisStatus::kInactive);
      highs_basis.col_status[i] = HighsBasisStatus::kBasic;
    }
  }

  for (HighsInt i = 0; i < instance.num_con; i++) {
    if (debug_report)
      printf("Row    %2d: status %s\n", int(i),
             qpBasisStatusToString(qp_solution.status_con[i]).c_str());
    if (qp_solution.status_con[i] == BasisStatus::kActiveAtLower) {
      highs_basis.row_status[i] = HighsBasisStatus::kLower;
    } else if (qp_solution.status_con[i] == BasisStatus::kActiveAtUpper) {
      highs_basis.row_status[i] = HighsBasisStatus::kUpper;
    } else if (qp_solution.status_con[i] == BasisStatus::kInactiveInBasis) {
      highs_basis.row_status[i] = HighsBasisStatus::kNonbasic;
    } else {
      assert(qp_solution.status_con[i] == BasisStatus::kInactive);
      highs_basis.row_status[i] = HighsBasisStatus::kBasic;
    }
  }
  highs_basis.valid = true;
  highs_basis.alien = false;
  highs_basis.useful = true;
  return qp_asm_return_status;
}

#ifdef HIGHS_RUST
// The solver runs in Rust (rust/src/qp); phase 1, the timer and the
// logging are called back from there
static_assert(sizeof(HighsInt) == 4, "the Rust QP solver takes 32-bit ints");

extern "C" {
struct RsQpModel {
  int num_var;
  int num_con;
  double offset;
  const double* c;
  const int* a_start;
  const int* a_index;
  const double* a_value;
  const int* q_start;
  const int* q_index;
  const double* q_value;
  const double* con_lo;
  const double* con_up;
  const double* var_lo;
  const double* var_up;
};

// Mirror of qp::Settings
struct RsQpSettings {
  int ratiotest;
  int pricing;
  int reportingfequency;
  int nullspace_limit;
  int reinvertfrequency;
  int gradientrecomputefrequency;
  int iteration_limit;
  double ratiotest_t;
  double ratiotest_d;
  double pnorm_zero_threshold;
  double d_zero_threshold;
  double lambda_zero_threshold;
  double pqp_zero_threshold;
  double hessian_regularization_value;
  double time_limit;
};

struct RsQpCallbacks {
  void* ctx;
  double (*time)(void* ctx);
  void (*iteration_log)(void* ctx, int iteration, double objval,
                        int nullspace_dim, double time);
  void (*nullspace_limit_log)(void* ctx, int nullspace_limit);
  void (*degeneracy_fail_log)(void* ctx, int maxabsd, double log_d);
  int (*phase1)(void* ctx, int* active, int* status, int* num_active,
                int* inactive, int* num_inactive, double* x0, double* ra);
};

struct RsQpSolution {
  int num_iterations;
  double* primal;
  double* rowact;
  double* dualvar;
  double* dualcon;
  int* status_var;
  int* status_con;
};

int highs_rs_qp_solve(const RsQpModel* model, const RsQpSettings* settings,
                      const RsQpCallbacks* callbacks, RsQpSolution* sol);
}

namespace {
struct QpContext {
  Instance& instance;
  Settings& settings;
  Statistics& stats;
  HighsModelStatus& highs_model_status;
  HighsBasis& highs_basis;
  HighsSolution& highs_solution;
  HighsTimer& timer;
};

double qpTime(void* ctx) { return static_cast<QpContext*>(ctx)->timer.read(); }

void qpIterationLog(void* ctx, int iteration, double objval, int nullspace_dim,
                    double time) {
  QpContext& c = *static_cast<QpContext*>(ctx);
  c.stats.iteration.push_back(iteration);
  c.stats.nullspacedimension.push_back(nullspace_dim);
  c.stats.objval.push_back(objval);
  c.stats.time.push_back(time);
  c.settings.iteration_log.fire(c.stats);
}

void qpNullspaceLimitLog(void* ctx, int nullspace_limit) {
  HighsInt limit = nullspace_limit;
  static_cast<QpContext*>(ctx)->settings.nullspace_limit_log.fire(limit);
}

void qpDegeneracyFailLog(void* ctx, int maxabsd, double log_d) {
  std::pair<HighsInt, double> data = std::make_pair(HighsInt(maxabsd), log_d);
  static_cast<QpContext*>(ctx)->settings.degeneracy_fail_log.fire(data);
}

int qpPhase1(void* ctx, int* active, int* status, int* num_active,
             int* inactive, int* num_inactive, double* x0, double* ra) {
  QpContext& c = *static_cast<QpContext*>(ctx);
  QpModelStatus qp_model_status = QpModelStatus::kUndetermined;
  QpHotstartInformation startinfo(c.instance.num_var, c.instance.num_con);
  computeStartingPointHighs(c.instance, c.settings, c.stats, qp_model_status,
                            startinfo, c.highs_model_status, c.highs_basis,
                            c.highs_solution, c.timer);
  *num_active = startinfo.active.size();
  for (size_t i = 0; i < startinfo.active.size(); i++) {
    active[i] = startinfo.active[i];
    status[i] = int(startinfo.status[i]);
  }
  *num_inactive = startinfo.inactive.size();
  for (size_t i = 0; i < startinfo.inactive.size(); i++)
    inactive[i] = startinfo.inactive[i];
  std::copy(startinfo.primal.value.begin(), startinfo.primal.value.end(), x0);
  std::copy(startinfo.rowact.value.begin(), startinfo.rowact.value.end(), ra);
  return int(qp_model_status);
}
}  // namespace

QpAsmStatus solveqp(Instance& instance, Settings& settings, Statistics& stats,
                    HighsModelStatus& highs_model_status,
                    HighsBasis& highs_basis, HighsSolution& highs_solution,
                    HighsTimer& qp_timer) {
  QpSolution qp_solution(instance);
  const RsQpModel model{instance.num_var,
                        instance.num_con,
                        instance.offset,
                        instance.c.value.data(),
                        instance.A.mat.start.data(),
                        instance.A.mat.index.data(),
                        instance.A.mat.value.data(),
                        instance.Q.mat.start.data(),
                        instance.Q.mat.index.data(),
                        instance.Q.mat.value.data(),
                        instance.con_lo.data(),
                        instance.con_up.data(),
                        instance.var_lo.data(),
                        instance.var_up.data()};
  const RsQpSettings rs_settings{int(settings.ratiotest),
                                 int(settings.pricing),
                                 settings.reportingfequency,
                                 settings.nullspace_limit,
                                 settings.reinvertfrequency,
                                 settings.gradientrecomputefrequency,
                                 settings.iteration_limit,
                                 settings.ratiotest_t,
                                 settings.ratiotest_d,
                                 settings.pnorm_zero_threshold,
                                 settings.d_zero_threshold,
                                 settings.lambda_zero_threshold,
                                 settings.pQp_zero_threshold,
                                 settings.hessian_regularization_value,
                                 settings.time_limit};
  QpContext ctx{instance,    settings,       stats,   highs_model_status,
                highs_basis, highs_solution, qp_timer};
  const RsQpCallbacks callbacks{&ctx,           qpTime,
                                qpIterationLog, qpNullspaceLimitLog,
                                qpDegeneracyFailLog, qpPhase1};
  std::vector<int> status_var(instance.num_var), status_con(instance.num_con);
  RsQpSolution sol{0,
                   qp_solution.primal.value.data(),
                   qp_solution.rowactivity.value.data(),
                   qp_solution.dualvar.value.data(),
                   qp_solution.dualcon.value.data(),
                   status_var.data(),
                   status_con.data()};
  QpModelStatus qp_model_status = QpModelStatus(
      highs_rs_qp_solve(&model, &rs_settings, &callbacks, &sol));
  stats.num_iterations = sol.num_iterations;
  for (HighsInt i = 0; i < instance.num_var; i++)
    qp_solution.status_var[i] = BasisStatus(status_var[i]);
  for (HighsInt i = 0; i < instance.num_con; i++)
    qp_solution.status_con[i] = BasisStatus(status_con[i]);
  return quass2highs(instance, settings, stats, qp_model_status, qp_solution,
                     highs_model_status, highs_basis, highs_solution);
}
#else
QpAsmStatus solveqp(Instance& instance, Settings& settings, Statistics& stats,
                    HighsModelStatus& highs_model_status,
                    HighsBasis& highs_basis, HighsSolution& highs_solution,
                    HighsTimer& qp_timer) {
  QpModelStatus qp_model_status = QpModelStatus::kUndetermined;

  QpSolution qp_solution(instance);

  // presolve

  // scale instance, store scaling factors

  // perturb instance, store perturbance information

  // regularize
  for (HighsInt i = 0; i < instance.num_var; i++) {
    for (HighsInt index = instance.Q.mat.start[i];
         index < instance.Q.mat.start[i + 1]; index++) {
      if (instance.Q.mat.index[index] == i) {
        instance.Q.mat.value[index] += settings.hessian_regularization_value;
      }
    }
  }

  // compute initial feasible point
  QpHotstartInformation startinfo(instance.num_var, instance.num_con);
  if (instance.num_con == 0 && instance.num_var <= 15000) {
    computeStartingPointBounded(instance, settings, stats, qp_model_status,
                                startinfo, qp_timer);
    if (qp_model_status == QpModelStatus::kOptimal) {
      qp_solution.primal = startinfo.primal;
      return quass2highs(instance, settings, stats, qp_model_status,
                         qp_solution, highs_model_status, highs_basis,
                         highs_solution);
    }
    if (qp_model_status == QpModelStatus::kUnbounded) {
      return quass2highs(instance, settings, stats, qp_model_status,
                         qp_solution, highs_model_status, highs_basis,
                         highs_solution);
    }
  } else {
    computeStartingPointHighs(instance, settings, stats, qp_model_status,
                              startinfo, highs_model_status, highs_basis,
                              highs_solution, qp_timer);
    if (qp_model_status != QpModelStatus::kNotset) {
      return quass2highs(instance, settings, stats, qp_model_status,
                         qp_solution, highs_model_status, highs_basis,
                         highs_solution);
    }
  }

  // solve
  solveqp_actual(instance, settings, startinfo, stats, qp_model_status,
                 qp_solution, qp_timer);

  // undo perturbation and resolve

  // undo scaling and resolve

  // postsolve

  // Transform QP status and qp_solution to HiGHS highs_basis and highs_solution
  return quass2highs(instance, settings, stats, qp_model_status, qp_solution,
                     highs_model_status, highs_basis, highs_solution);
}
#endif
