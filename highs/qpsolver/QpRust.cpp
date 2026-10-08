/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file qpsolver/QpRust.cpp
 * @brief Highs::callSolveQp with the QP glue in Rust (rust/src/qp/glue.rs):
 * the views, the profiling clock, the timer, sizing the solution and
 * basis, and the phase 1 LP solve
 */
#include "lp_data/HighsRust.h"

#ifdef HIGHS_RUST
#include "Highs.h"

namespace {
struct RsQpSolutionOut {
  RsMut<double> col_value, col_dual, row_value, row_dual;
  RsMut<uint8_t> col_status, row_status;
};

struct RsQpPhase1 {
  double time_limit;
  bool run_error;
  int model_status;
  HighsInt simplex_iteration_count;
  RsMut<uint8_t> col_status, row_status;
  RsMut<double> col_value, row_value;
  size_t num_col_status, num_row_status;
};

struct RsQpHost {
  void* ctx;
  double (*op)(void* ctx, int code, void* out);
  RsLog log;
  RsLp lp;
  HighsInt hessian_dim;
  RsMut<HighsInt> hessian_start, hessian_index;
  RsMut<double> hessian_value;
  HighsInt qp_iteration_limit, qp_nullspace_limit,
      simplex_primal_edge_weight_strategy;
  bool qp_allow_hot_start, timeless_log;
  double qp_regularization_value, time_limit, dual_feasibility_tolerance;
  RsMut<double> col_value, row_value;
  RsMut<uint8_t> col_status, row_status;
  bool *value_valid, *dual_valid, *basis_valid, *basis_alien, *basis_useful;
  int* model_status;
  HighsInfoStruct* info;
};

struct QpGlueCtx {
  HighsProfiling* profiling;
  HighsTimer& timer;
  const HighsLp& lp;
  HighsSolution& solution;
  HighsBasis& basis;
};

RsMut<uint8_t> qpGlueStatus(const std::vector<HighsBasisStatus>& s) {
  return {reinterpret_cast<uint8_t*>(const_cast<HighsBasisStatus*>(s.data())),
          s.size()};
}

template <typename T, typename F>
void qpGluePut(RsMut<T> to, const std::vector<F>& from) {
  for (size_t i = 0; i < to.len && i < from.size(); i++)
    to.ptr[i] = static_cast<T>(from[i]);
}

// The feasibility LP of computeStartingPointHighs, solved by a silent Highs
void qpGluePhase1(QpGlueCtx& c, RsQpPhase1& p) {
  Highs highs;
  highs.setOptionValue("output_flag", false);
  highs.setOptionValue("presolve", kHighsOnString);
  highs.setOptionValue("time_limit", p.time_limit);
  HighsLp lp;
  lp.a_matrix_.index_ = c.lp.a_matrix_.index_;
  lp.a_matrix_.start_ = c.lp.a_matrix_.start_;
  lp.a_matrix_.value_ = c.lp.a_matrix_.value_;
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.col_cost_.assign(c.lp.num_col_, 0.0);
  lp.col_lower_ = c.lp.col_lower_;
  lp.col_upper_ = c.lp.col_upper_;
  lp.row_lower_ = c.lp.row_lower_;
  lp.row_upper_ = c.lp.row_upper_;
  lp.num_col_ = c.lp.num_col_;
  lp.num_row_ = c.lp.num_row_;
  highs.passModel(lp);
  if (highs.run() == HighsStatus::kError) {
    p.run_error = true;
    return;
  }
  p.model_status = int(highs.getModelStatus());
  p.simplex_iteration_count = highs.getInfo().simplex_iteration_count;
  if (highs.getModelStatus() != HighsModelStatus::kOptimal) return;
  const HighsBasis& basis = highs.getBasis();
  const HighsSolution& solution = highs.getSolution();
  qpGluePut(p.col_status, basis.col_status);
  qpGluePut(p.row_status, basis.row_status);
  qpGluePut(p.col_value, solution.col_value);
  qpGluePut(p.row_value, solution.row_value);
  p.num_col_status = basis.col_status.size();
  p.num_row_status = basis.row_status.size();
}

double qpGlueOp(void* ctx, int code, void* out) {
  QpGlueCtx& c = *static_cast<QpGlueCtx*>(ctx);
  switch (code) {
    case 0:
      if (c.profiling) c.profiling->start(kSubSolverQpAsm);
      break;
    case 1:
      if (c.profiling) c.profiling->stop(kSubSolverQpAsm);
      break;
    case 2:
      return c.timer.read();
    case 3: {
      RsQpSolutionOut& o = *static_cast<RsQpSolutionOut*>(out);
      HighsSolution& s = c.solution;
      s.col_value.resize(c.lp.num_col_);
      s.col_dual.resize(c.lp.num_col_);
      s.row_value.resize(c.lp.num_row_);
      s.row_dual.resize(c.lp.num_row_);
      c.basis.col_status.resize(c.lp.num_col_);
      c.basis.row_status.resize(c.lp.num_row_);
      o.col_value = rsMut(s.col_value);
      o.col_dual = rsMut(s.col_dual);
      o.row_value = rsMut(s.row_value);
      o.row_dual = rsMut(s.row_dual);
      o.col_status = qpGlueStatus(c.basis.col_status);
      o.row_status = qpGlueStatus(c.basis.row_status);
      break;
    }
    case 4:
      qpGluePhase1(c, *static_cast<RsQpPhase1*>(out));
      break;
  }
  return 0;
}
}  // namespace

extern "C" int highs_rs_call_solve_qp(const RsQpHost* host);

HighsStatus Highs::callSolveQp() {
  HighsLp& lp = model_.lp_;
  assert(lp.a_matrix_.isColwise());
  const HighsHessian& hessian = model_.hessian_;
  assert(hessian.format_ == HessianFormat::kTriangular);
  QpGlueCtx ctx{this->profiling_, timer_, lp, solution_, basis_};
  RsQpHost h;
  h.ctx = &ctx;
  h.op = qpGlueOp;
  h.log = rsLog(options_.log_options);
  h.lp = rsLp(lp);
  h.hessian_dim = hessian.dim_;
  h.hessian_start = rsMut(hessian.start_);
  h.hessian_index = rsMut(hessian.index_);
  h.hessian_value = rsMut(hessian.value_);
  h.qp_iteration_limit = options_.qp_iteration_limit;
  h.qp_nullspace_limit = options_.qp_nullspace_limit;
  h.simplex_primal_edge_weight_strategy =
      options_.simplex_primal_edge_weight_strategy;
  h.qp_allow_hot_start = options_.qp_allow_hot_start;
  h.timeless_log = options_.timeless_log;
  h.qp_regularization_value = options_.qp_regularization_value;
  h.time_limit = options_.time_limit;
  h.dual_feasibility_tolerance = options_.dual_feasibility_tolerance;
  h.col_value = rsMut(solution_.col_value);
  h.row_value = rsMut(solution_.row_value);
  h.col_status = qpGlueStatus(basis_.col_status);
  h.row_status = qpGlueStatus(basis_.row_status);
  h.value_valid = &solution_.value_valid;
  h.dual_valid = &solution_.dual_valid;
  h.basis_valid = &basis_.valid;
  h.basis_alien = &basis_.alien;
  h.basis_useful = &basis_.useful;
  h.model_status = reinterpret_cast<int*>(&model_status_);
  h.info = static_cast<HighsInfoStruct*>(&info_);
  const HighsStatus return_status = HighsStatus(highs_rs_call_solve_qp(&h));
  if (return_status == HighsStatus::kError) return return_status;
  // Get the objective and any KKT failures
  info_.objective_function_value = model_.objectiveValue(solution_.col_value);
  getKktFailures(options_, model_, solution_, basis_, info_);
  info_.valid = true;
  if (model_status_ == HighsModelStatus::kOptimal) return checkOptimality("QP");
  return return_status;
}
#endif
