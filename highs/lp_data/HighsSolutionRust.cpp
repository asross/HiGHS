/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file lp_data/HighsSolutionRust.cpp
 * @brief The KKT checks of HighsSolution.cpp done by Rust
 * (rust/src/lp_data/solution.rs)
 */
#include "lp_data/HighsRust.h"

#ifdef HIGHS_RUST
#include <cstddef>

#include "ipm/IpxSolution.h"
#include "lp_data/HighsLpSolverObject.h"
#include "lp_data/HighsLpUtils.h"
#include "lp_data/HighsSolution.h"

// Rust reads and writes HighsInfoStruct and HighsPrimalDualErrors in
// place (solution.rs: Info, PrimalDualErrors)
static_assert(sizeof(HighsInfoStruct) == 280, "HighsInfoStruct layout");
static_assert(offsetof(HighsInfoStruct, objective_function_value) == 48,
              "HighsInfoStruct layout");
static_assert(offsetof(HighsInfoStruct, primal_dual_integral) == 272,
              "HighsInfoStruct layout");
static_assert(sizeof(HighsPrimalDualErrors) == 184,
              "HighsPrimalDualErrors layout");
static_assert(sizeof(HighsBasisStatus) == 1, "HighsBasisStatus is a byte");

struct RsKktOptions {
  RsLog log;
  double primal_feasibility_tolerance, dual_feasibility_tolerance,
      mip_feasibility_tolerance, primal_residual_tolerance,
      dual_residual_tolerance, optimality_tolerance, kkt_tolerance;
  HighsInt log_dev_level;
  bool full_lp_kkt_check;
};

extern "C" {
void highs_rs_get_kkt_failures(const RsKktOptions* o, bool is_qp,
                               const RsLp* lp, RsMut<double> gradient,
                               const RsSolution* sol, HighsInfoStruct* info,
                               bool get_residuals);
void highs_rs_get_primal_dual_basis_errors(const RsKktOptions* o,
                                           const RsLp* lp,
                                           const RsSolution* sol,
                                           const RsBasis* basis,
                                           HighsPrimalDualErrors* e);
void highs_rs_get_primal_dual_glpsol_errors(const RsKktOptions* o,
                                            const RsLp* lp,
                                            const RsSolution* sol,
                                            HighsPrimalDualErrors* e);
void highs_rs_get_complementarity_violations(const RsLp* lp,
                                             const RsSolution* sol,
                                             double optimality_tolerance,
                                             HighsInt* num, double* max);
double highs_rs_compute_dual_objective_value(RsMut<double> gradient,
                                             const RsLp* lp,
                                             const RsSolution* sol);
double highs_rs_compute_objective_value(const RsLp* lp, const RsSolution* sol);
double highs_rs_lp_objective_value(const RsLp* lp, RsMut<double> x);
void highs_rs_lp_kkt_check(int* model_status, HighsInfoStruct* info,
                           const RsLp* lp, const RsSolution* sol,
                           bool basis_valid, const RsKktOptions* o,
                           const char* message, size_t message_len);
bool highs_rs_report_kkt_failures(const RsLp* lp, const RsKktOptions* o,
                                  const HighsInfoStruct* info,
                                  const char* message, size_t message_len);
}

static RsKktOptions rsKktOptions(const HighsOptions& options) {
  RsKktOptions o;
  rsOptionsTemplate(options, 1, &o);
  return o;
}

void rsKktOptionsInto(const HighsOptions& options, void* out) {
  rsOptionsTemplate(options, 1, out);
}

void getKktFailures(const HighsOptions& options, const bool is_qp,
                    const HighsLp& lp, const std::vector<double>& gradient,
                    const HighsSolution& solution, HighsInfo& highs_info,
                    const bool get_residuals) {
  assert(solution.value_valid || !solution.dual_valid);
  const RsKktOptions o = rsKktOptions(options);
  const RsLp v = rsLp(lp);
  const RsSolution s = rsSolution(solution);
  highs_rs_get_kkt_failures(&o, is_qp, &v, rsMut(gradient), &s,
                            static_cast<HighsInfoStruct*>(&highs_info),
                            get_residuals);
}

void getPrimalDualBasisErrors(const HighsOptions& options, const HighsLp& lp,
                              const HighsSolution& solution,
                              const HighsBasis& basis,
                              HighsPrimalDualErrors& primal_dual_errors) {
  const RsKktOptions o = rsKktOptions(options);
  const RsLp v = rsLp(lp);
  const RsSolution s = rsSolution(solution);
  const RsBasis b = rsBasis(basis);
  highs_rs_get_primal_dual_basis_errors(&o, &v, &s, &b, &primal_dual_errors);
}

void getPrimalDualGlpsolErrors(const HighsOptions& options, const HighsLp& lp,
                               const std::vector<double>& gradient,
                               const HighsSolution& solution,
                               HighsPrimalDualErrors& primal_dual_errors) {
  const RsKktOptions o = rsKktOptions(options);
  const RsLp v = rsLp(lp);
  const RsSolution s = rsSolution(solution);
  highs_rs_get_primal_dual_glpsol_errors(&o, &v, &s, &primal_dual_errors);
}

bool getComplementarityViolations(const HighsLp& lp,
                                  const HighsSolution& solution,
                                  const double optimality_tolerance,
                                  HighsInt& num_complementarity_violation,
                                  double& max_complementarity_violation) {
  num_complementarity_violation = kHighsIllegalComplementarityCount;
  max_complementarity_violation = kHighsIllegalComplementarityViolation;
  if (!solution.dual_valid) return false;
  const RsLp v = rsLp(lp);
  const RsSolution s = rsSolution(solution);
  highs_rs_get_complementarity_violations(&v, &s, optimality_tolerance,
                                          &num_complementarity_violation,
                                          &max_complementarity_violation);
  return true;
}

bool computeDualObjectiveValue(const double* gradient, const HighsLp& lp,
                               const HighsSolution& solution,
                               double& dual_objective_value) {
  dual_objective_value = 0;
  if (!solution.dual_valid) return false;
  // #2184 Make sure that the solution corresponds to this LP
  assert(solution.col_value.size() == static_cast<size_t>(lp.num_col_));
  assert(solution.col_dual.size() == static_cast<size_t>(lp.num_col_));
  assert(solution.row_value.size() == static_cast<size_t>(lp.num_row_));
  assert(solution.row_dual.size() == static_cast<size_t>(lp.num_row_));
  const RsLp v = rsLp(lp);
  const RsSolution s = rsSolution(solution);
  RsMut<double> g = {const_cast<double*>(gradient),
                     gradient ? size_t(lp.num_col_) : 0};
  dual_objective_value = highs_rs_compute_dual_objective_value(g, &v, &s);
  return true;
}

double computeObjectiveValue(const HighsLp& lp, const HighsSolution& solution) {
  const RsLp v = rsLp(lp);
  const RsSolution s = rsSolution(solution);
  return highs_rs_compute_objective_value(&v, &s);
}

double HighsLp::objectiveValue(const std::vector<double>& solution) const {
  assert((int)solution.size() >= this->num_col_);
  const RsLp v = rsLp(*this);
  return highs_rs_lp_objective_value(&v, rsMut(solution));
}

void lpKktCheck(HighsModelStatus& model_status, HighsInfo& info,
                const HighsLp& lp, const HighsSolution& solution,
                const HighsBasis& basis, const HighsOptions& options,
                const std::string& message) {
  const RsKktOptions o = rsKktOptions(options);
  const RsLp v = rsLp(lp);
  const RsSolution s = rsSolution(solution);
  int status = int(model_status);
  highs_rs_lp_kkt_check(&status, static_cast<HighsInfoStruct*>(&info), &v, &s,
                        basis.valid, &o, message.data(), message.size());
  model_status = HighsModelStatus(status);
}

bool reportKktFailures(const HighsLp& lp, const HighsOptions& options,
                       const HighsInfo& info, const std::string& message) {
  const RsKktOptions o = rsKktOptions(options);
  const RsLp v = rsLp(lp);
  return highs_rs_report_kkt_failures(
      &v, &o, static_cast<const HighsInfoStruct*>(&info), message.data(),
      message.size());
}

// The basis functions and IPX conversions (rust/src/lp_data/basis.rs)
struct RsSolutionOut {
  RsMut<double> col_value, col_dual, row_value, row_dual;
  RsMut<uint8_t> col_status, row_status;
};
struct RsIpxSolution {
  HighsInt num_col, num_row;
  RsMut<double> col_value, row_value, col_dual, row_dual;
  RsMut<HighsInt> col_status, row_status;
};

extern "C" {
void highs_rs_refine_basis(RsMut<double> lower, RsMut<double> upper,
                           RsMut<double> value, RsMut<uint8_t> status);
bool highs_rs_basis_consistent(RsMut<uint8_t> col_status,
                               RsMut<uint8_t> row_status);
void highs_rs_ipx_solution_to_highs_solution(
    const RsLp* lp, RsMut<double> rhs, HighsInt ipx_num_row,
    RsMut<double> ipx_x, RsMut<double> ipx_slack_vars, RsMut<double> ipx_y,
    RsMut<double> ipx_zl, RsMut<double> ipx_zu, const RsSolutionOut* out);
int highs_rs_ipx_basic_solution_to_highs_basic_solution(
    const RsLog* log, const RsLp* lp, RsMut<double> rhs,
    RsMut<uint8_t> constraint_type, const RsIpxSolution* ipx,
    const RsSolutionOut* out);
}

static RsMut<uint8_t> rsStatus(const std::vector<HighsBasisStatus>& s) {
  return {reinterpret_cast<uint8_t*>(const_cast<HighsBasisStatus*>(s.data())),
          s.size()};
}

static const RsMut<double> kRsNoValues = {nullptr, 0};

void refineBasis(const HighsLp& lp, const HighsSolution& solution,
                 HighsBasis& basis) {
  assert(basis.useful);
  assert(isBasisRightSize(lp, basis));
  const bool have_highs_solution = solution.value_valid;
  highs_rs_refine_basis(
      rsMut(lp.col_lower_), rsMut(lp.col_upper_),
      have_highs_solution ? rsMut(solution.col_value) : kRsNoValues,
      rsStatus(basis.col_status));
  highs_rs_refine_basis(
      rsMut(lp.row_lower_), rsMut(lp.row_upper_),
      have_highs_solution ? rsMut(solution.row_value) : kRsNoValues,
      rsStatus(basis.row_status));
}

bool isBasisConsistent(const HighsLp& lp, const HighsBasis& basis) {
  if (!isBasisRightSize(lp, basis)) return false;
  return highs_rs_basis_consistent(rsStatus(basis.col_status),
                                   rsStatus(basis.row_status));
}

static RsSolutionOut rsSolutionOut(HighsSolution& solution,
                                   HighsBasis* basis) {
  RsSolutionOut out;
  out.col_value = rsMut(solution.col_value);
  out.col_dual = rsMut(solution.col_dual);
  out.row_value = rsMut(solution.row_value);
  out.row_dual = rsMut(solution.row_dual);
  out.col_status =
      basis ? rsStatus(basis->col_status) : RsMut<uint8_t>{nullptr, 0};
  out.row_status =
      basis ? rsStatus(basis->row_status) : RsMut<uint8_t>{nullptr, 0};
  return out;
}

HighsStatus ipxSolutionToHighsSolution(
    const HighsOptions& options, const HighsLp& lp,
    const std::vector<double>& rhs, const std::vector<char>& constraint_type,
    const HighsInt ipx_num_col, const HighsInt ipx_num_row,
    const std::vector<double>& ipx_x, const std::vector<double>& ipx_slack_vars,
    const std::vector<double>& ipx_y, const std::vector<double>& ipx_zl,
    const std::vector<double>& ipx_zu, HighsSolution& highs_solution) {
  highs_solution.col_value.resize(lp.num_col_);
  highs_solution.row_value.resize(lp.num_row_);
  highs_solution.col_dual.resize(lp.num_col_);
  highs_solution.row_dual.resize(lp.num_row_);
  const RsLp v = rsLp(lp);
  const RsSolutionOut out = rsSolutionOut(highs_solution, nullptr);
  highs_rs_ipx_solution_to_highs_solution(
      &v, rsMut(rhs), ipx_num_row, rsMut(ipx_x), rsMut(ipx_slack_vars),
      rsMut(ipx_y), rsMut(ipx_zl), rsMut(ipx_zu), &out);
  highs_solution.value_valid = true;
  highs_solution.dual_valid = true;
  return HighsStatus::kOk;
}

HighsStatus ipxBasicSolutionToHighsBasicSolution(
    const HighsLogOptions& log_options, const HighsLp& lp,
    const std::vector<double>& rhs, const std::vector<char>& constraint_type,
    const IpxSolution& ipx_solution, HighsBasis& highs_basis,
    HighsSolution& highs_solution) {
  highs_solution.col_value.resize(lp.num_col_);
  highs_solution.row_value.resize(lp.num_row_);
  highs_solution.col_dual.resize(lp.num_col_);
  highs_solution.row_dual.resize(lp.num_row_);
  highs_basis.col_status.resize(lp.num_col_);
  highs_basis.row_status.resize(lp.num_row_);
  const RsLog log = rsLog(log_options);
  const RsLp v = rsLp(lp);
  const RsSolutionOut out = rsSolutionOut(highs_solution, &highs_basis);
  const RsIpxSolution ipx = {
      ipx_solution.num_col,             ipx_solution.num_row,
      rsMut(ipx_solution.ipx_col_value), rsMut(ipx_solution.ipx_row_value),
      rsMut(ipx_solution.ipx_col_dual),  rsMut(ipx_solution.ipx_row_dual),
      rsMut(ipx_solution.ipx_col_status), rsMut(ipx_solution.ipx_row_status)};
  const RsMut<uint8_t> types = {
      reinterpret_cast<uint8_t*>(const_cast<char*>(constraint_type.data())),
      constraint_type.size()};
  if (HighsStatus(highs_rs_ipx_basic_solution_to_highs_basic_solution(
          &log, &v, rsMut(rhs), types, &ipx, &out)) == HighsStatus::kError)
    return HighsStatus::kError;
  highs_solution.value_valid = true;
  highs_solution.dual_valid = true;
  highs_basis.valid = true;
  highs_basis.useful = true;
  return HighsStatus::kOk;
}

// formSimplexLpBasisAndFactor and accommodateAlienBasis
// (rust/src/lp_data/form_basis.rs)
struct RsFormHost {
  void* ctx;
  int (*op)(void* ctx, int code, int arg, void* out);
  RsLog log, factor_log;
  void* lps;
  RsLp incumbent;
  RsMut<char> model_name;
  RsLpOptions lp_options;
  bool basis_valid, basis_useful;
  bool* basis_alien;
  RsMut<uint8_t> col_status, row_status;
};

extern "C" {
int highs_rs_form_simplex_lp_basis_and_factor(const RsFormHost* host,
                                              bool only_from_known_basis);
}

namespace {
// The log of an HFactor set up with these log options: no callbacks
struct FactorLog {
  bool output_flag, log_to_console;
  HighsInt log_dev_level;
  HighsLogOptions log_options;
  explicit FactorLog(const HighsLogOptions& from) {
    output_flag = *from.output_flag;
    log_to_console = *from.log_to_console;
    log_dev_level = *from.log_dev_level;
    log_options.output_flag = &output_flag;
    log_options.log_to_console = &log_to_console;
    log_options.log_dev_level = &log_dev_level;
    log_options.log_stream = from.log_stream;
  }
};

int formBasisOp(void* ctx, int code, int arg, void* out) {
  HighsLpSolverObject& so = *static_cast<HighsLpSolverObject*>(ctx);
  switch (code) {
    case 4:
      so.ekk_instance_.movedLp(so);
      break;
    case 6:
      return int(so.ekk_instance_.setBasis(so.basis_));
    case 7:
      return int(so.ekk_instance_.initialiseSimplexLpBasisAndFactor(arg != 0));
    case 8:
      so.ekk_instance_.lpBack(so.lp_, false);
      break;
  }
  return 0;
}
}  // namespace

HighsStatus formSimplexLpBasisAndFactor(HighsLpSolverObject& solver_object,
                                        const bool only_from_known_basis) {
  const FactorLog factor_log(solver_object.options_.log_options);
  HighsBasis& basis = solver_object.basis_;
  HighsLp& lp = solver_object.lp_;
  lp.ensureColwise();
  RsFormHost h;
  h.ctx = &solver_object;
  h.op = formBasisOp;
  h.log = rsLog(solver_object.options_.log_options);
  h.factor_log = rsLog(factor_log.log_options);
  h.lps = solver_object.ekk_instance_.rs_;
  h.incumbent = rsLp(lp);
  h.model_name = {const_cast<char*>(lp.model_name_.data()),
                  lp.model_name_.size()};
  h.lp_options = rsLpOptions(solver_object.options_);
  h.basis_valid = basis.valid;
  h.basis_useful = basis.useful;
  h.basis_alien = &basis.alien;
  h.col_status = rsStatus(basis.col_status);
  h.row_status = rsStatus(basis.row_status);
  return HighsStatus(
      highs_rs_form_simplex_lp_basis_and_factor(&h, only_from_known_basis));
}

#endif
