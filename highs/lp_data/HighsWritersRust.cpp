/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file lp_data/HighsWritersRust.cpp
 * @brief The file writers done by Rust: solution files in every style and
 * basis files (rust/src/lp_data/writers.rs), MPS and LP model files
 * (rust/src/io/model_write.rs). The C++ originals are in
 * HighsModelUtils.cpp, HighsLpUtils.cpp, HMPSIO.cpp and FilereaderLp.cpp.
 */
#include "lp_data/HighsRust.h"

#ifdef HIGHS_RUST
#include <cstring>

#include "io/FilereaderLp.h"
#include "io/HMPSIO.h"
#include "io/HighsIO.h"
#include "lp_data/HighsLpUtils.h"
#include "lp_data/HighsModelUtils.h"
#include "lp_data/HighsRanging.h"
#include "lp_data/HighsSolution.h"

namespace {

// A string as "%s" prints it
struct RsStr {
  const char* ptr;
  size_t len;
};

RsStr rsStr(const std::string& s) { return {s.c_str(), strlen(s.c_str())}; }

std::vector<RsStr> rsStrs(const std::vector<std::string>& names) {
  std::vector<RsStr> v;
  v.reserve(names.size());
  for (const std::string& s : names) v.push_back(rsStr(s));
  return v;
}

// Where Rust sends file text (kind -1) and messages (0 for highsLogDev,
// otherwise the HighsLogType of highsLogUser). With chunked, the file is
// stdout and each piece of text is one highsFprintfString.
struct RsOut {
  FILE* file;
  const HighsLogOptions* log_options;
  bool chunked;
  void (*emit)(const RsOut* out, int kind, const char* text, size_t len);
};

void rsOutEmit(const RsOut* out, int kind, const char* text, size_t len) {
  if (kind < 0) {
    if (out->chunked)
      highsFprintfString(out->file, *out->log_options, std::string(text, len));
    else if (out->file)
      fwrite(text, 1, len, out->file);
    return;
  }
  const std::string s(text, len);
  if (kind == 0)
    highsLogDev(*out->log_options, HighsLogType::kInfo, "%s", s.c_str());
  else
    highsLogUser(*out->log_options, HighsLogType(kind), "%s", s.c_str());
}

// For text C++ prints with highsFprintfString (fprintf_string) or fprintf
RsOut rsOut(FILE* file, const HighsLogOptions& log_options,
            bool fprintf_string = true) {
  return {file, &log_options, fprintf_string && file == stdout, rsOutEmit};
}

// The model data a writer reads (io/write.rs: CWriteModel)
struct RsWriteModel {
  HighsInt num_col, num_row;
  RsMut<double> col_cost, col_lower, col_upper, row_lower, row_upper;
  RsMut<HighsInt> a_start, a_index;
  RsMut<double> a_value;
  int sense;
  double offset;
  RsMut<uint8_t> integrality;
  HighsInt q_dim;
  RsMut<HighsInt> q_start, q_index;
  RsMut<double> q_value;
  RsMut<RsStr> col_names, row_names;
  RsStr model_name, objective_name;
  HighsInt cost_row_location;
};

// The names stay alive with the view
struct WriteModel {
  std::vector<RsStr> col_names, row_names;
  RsWriteModel v;
  // Without a, the matrix is not passed
  WriteModel(const HighsLp& lp, const HighsHessian* hessian,
             const HighsSparseMatrix* a, const std::string& objective_name)
      : col_names(rsStrs(lp.col_names_)), row_names(rsStrs(lp.row_names_)) {
    v.num_col = lp.num_col_;
    v.num_row = lp.num_row_;
    v.col_cost = rsMut(lp.col_cost_);
    v.col_lower = rsMut(lp.col_lower_);
    v.col_upper = rsMut(lp.col_upper_);
    v.row_lower = rsMut(lp.row_lower_);
    v.row_upper = rsMut(lp.row_upper_);
    v.a_start = a ? rsMut(a->start_) : RsMut<HighsInt>{nullptr, 0};
    v.a_index = a ? rsMut(a->index_) : RsMut<HighsInt>{nullptr, 0};
    v.a_value = a ? rsMut(a->value_) : RsMut<double>{nullptr, 0};
    v.sense = int(lp.sense_);
    v.offset = lp.offset_;
    v.integrality = rsMut(lp.integrality_);
    v.q_dim = hessian ? hessian->dim_ : 0;
    v.q_start = hessian ? rsMut(hessian->start_) : RsMut<HighsInt>{nullptr, 0};
    v.q_index = hessian ? rsMut(hessian->index_) : RsMut<HighsInt>{nullptr, 0};
    v.q_value = hessian ? rsMut(hessian->value_) : RsMut<double>{nullptr, 0};
    v.col_names = {col_names.data(), col_names.size()};
    v.row_names = {row_names.data(), row_names.size()};
    v.model_name = rsStr(lp.model_name_);
    v.objective_name = rsStr(objective_name);
    v.cost_row_location = lp.cost_row_location_;
  }
};

// writers.rs: CSolutionFile
struct RsSolutionFile {
  HighsInt style;
  const HighsInfoStruct* info;
  int model_status;
  RsStr model_status_string;
  double objective;
  HighsInt num_nz;
  HighsInt glpsol_cost_row_location;
};

// Parts of writeSolutionFile (writers.rs: PART_*)
const HighsInt kPartModelSolution = 5;
const HighsInt kPartModelSolutionSparse = 6;

}  // namespace

extern "C" {
void highs_rs_write_mps(const RsOut* out, const RsWriteModel* model);
void highs_rs_write_lp(const RsOut* out, const RsWriteModel* model);
bool highs_rs_write_solution_file(const RsOut* out, const RsWriteModel* model,
                                  const RsSolution* sol, const RsBasis* basis,
                                  const RsSolutionFile* s);
void highs_rs_write_glpsol_kkt(const RsOut* out,
                               const HighsPrimalDualErrors* e,
                               HighsInt num_col, bool is_mip, bool have_dual);
void highs_rs_write_model_bound_solution(
    const RsOut* out, bool columns, RsMut<double> lower, RsMut<double> upper,
    RsMut<RsStr> names, const RsMut<double>* primal, const RsMut<double>* dual,
    const RsMut<uint8_t>* status, const RsMut<uint8_t>* integrality);
void highs_rs_write_primal_solution(const RsOut* out, HighsInt num_col,
                                    RsMut<RsStr> col_names,
                                    RsMut<double> primal, bool sparse);
void highs_rs_write_objective_value(const RsOut* out, double v);
void highs_rs_write_basis_file(const RsOut* out, const RsWriteModel* model,
                               const RsBasis* basis);
// writers.rs: CRangingFile
struct RsRangingFile {
  bool valid, pretty;
  double objective;
  RsMut<double> rec[12];
};
void highs_rs_write_ranging_file(const RsOut* out, const RsWriteModel* model,
                                 const RsSolution* sol, const RsBasis* basis,
                                 const RsRangingFile* r);
}

void writeRangingFile(FILE* file, const HighsLp& lp,
                      const double objective_function_value,
                      const HighsBasis& basis, const HighsSolution& solution,
                      const HighsRanging& ranging, const HighsInt style) {
  assert(!ranging.valid ||
         lp.col_names_.size() == static_cast<size_t>(lp.num_col_));
  assert(!ranging.valid ||
         lp.row_names_.size() == static_cast<size_t>(lp.num_row_));
  RsRangingFile r;
  r.valid = ranging.valid;
  r.pretty = style == kSolutionStylePretty;
  r.objective = objective_function_value;
  const HighsRangingRecord* recs[6] = {
      &ranging.col_cost_up,  &ranging.col_cost_dn,  &ranging.col_bound_up,
      &ranging.col_bound_dn, &ranging.row_bound_up, &ranging.row_bound_dn};
  for (int k = 0; k < 6; k++) {
    r.rec[2 * k] = rsMut(recs[k]->value_);
    r.rec[2 * k + 1] = rsMut(recs[k]->objective_);
  }
  // No messages: the log options are not read
  static const HighsLogOptions no_log{};
  const WriteModel m(lp, nullptr, nullptr, lp.objective_name_);
  const RsOut out = rsOut(file, no_log, false);
  const RsSolution sol = rsSolution(solution);
  const RsBasis b = rsBasis(basis);
  highs_rs_write_ranging_file(&out, &m.v, &sol, &b, &r);
}

// Writes a solution file style, or a part of one (kPart*); returns
// whether the KKT report of a pretty Glpsol file follows
static bool rsWriteSolution(FILE* file, const HighsLogOptions& log_options,
                            const HighsLp& lp, const HighsHessian& hessian,
                            const HighsBasis& basis,
                            const HighsSolution& solution,
                            const HighsInfo& info,
                            const HighsModelStatus model_status,
                            const HighsInt style,
                            const HighsInt glpsol_cost_row_location) {
  assert(lp.col_names_.size() == static_cast<size_t>(lp.num_col_));
  assert(lp.row_names_.size() == static_cast<size_t>(lp.num_row_));
  const bool model_solution = style == kSolutionStyleRaw ||
                              style == kSolutionStyleSparse ||
                              style == kPartModelSolution ||
                              style == kPartModelSolutionSparse;
  // The objective value that writeModelObjective writes
  double objective = 0;
  if (model_solution && solution.value_valid &&
      info.primal_solution_status != kSolutionStatusNone) {
    HighsCDouble objective_value = lp.objectiveCDoubleValue(solution.col_value);
    objective_value += hessian.objectiveCDoubleValue(solution.col_value);
    objective = double(objective_value);
  }
  const std::string status_string = utilModelStatusToString(model_status);
  const RsSolutionFile s = {style,
                            &info,
                            int(model_status),
                            rsStr(status_string),
                            objective,
                            lp.a_matrix_.numNz(),
                            glpsol_cost_row_location};
  // The matrix is only counted
  const WriteModel m(lp, &hessian, nullptr, lp.objective_name_);
  const RsOut out = rsOut(file, log_options);
  const RsSolution sol = rsSolution(solution);
  const RsBasis b = rsBasis(basis);
  return highs_rs_write_solution_file(&out, &m.v, &sol, &b, &s);
}

void writeSolutionFile(FILE* file, const HighsOptions& options,
                       const HighsModel& model, const HighsBasis& basis,
                       const HighsSolution& solution, const HighsInfo& info,
                       const HighsModelStatus model_status,
                       const HighsInt style) {
  if (!rsWriteSolution(file, options.log_options, model.lp_, model.hessian_,
                       basis, solution, info, model_status, style,
                       options.glpsol_cost_row_location))
    return;
  // The KKT report of a pretty Glpsol file
  HighsPrimalDualErrors errors;
  HighsInfo local_info;
  getKktFailures(options, model, solution, basis, local_info, errors, true);
  const RsOut out = rsOut(file, options.log_options);
  highs_rs_write_glpsol_kkt(&out, &errors, model.lp_.num_col_,
                            model.lp_.isMip(), solution.dual_valid);
}

void writeGlpsolSolution(FILE* file, const HighsOptions& options,
                         const HighsModel& model, const HighsBasis& basis,
                         const HighsSolution& solution,
                         const HighsModelStatus model_status,
                         const HighsInfo& info, const bool raw) {
  writeSolutionFile(file, options, model, basis, solution, info, model_status,
                    raw ? kSolutionStyleGlpsolRaw : kSolutionStyleGlpsolPretty);
}

void writeOldRawSolution(FILE* file, const HighsLogOptions& log_options,
                         const HighsLp& lp, const HighsBasis& basis,
                         const HighsSolution& solution) {
  rsWriteSolution(file, log_options, lp, HighsHessian(), basis, solution,
                  HighsInfo(), HighsModelStatus::kNotset, kSolutionStyleOldRaw,
                  0);
}

void writeModelSolution(FILE* file, const HighsLogOptions& log_options,
                        const HighsModel& model, const HighsSolution& solution,
                        const HighsInfo& info, const bool sparse) {
  rsWriteSolution(file, log_options, model.lp_, model.hessian_, HighsBasis(),
                  solution, info, HighsModelStatus::kNotset,
                  sparse ? kPartModelSolutionSparse : kPartModelSolution, 0);
}

void writeModelBoundSolution(
    FILE* file, const HighsLogOptions& log_options, const bool columns,
    const HighsInt dim, const std::vector<double>& lower,
    const std::vector<double>& upper, const std::vector<std::string>& names,
    const bool have_primal, const std::vector<double>& primal,
    const bool have_dual, const std::vector<double>& dual,
    const bool have_basis, const std::vector<HighsBasisStatus>& status,
    const HighsVarType* integrality) {
  assert(names.size() == static_cast<size_t>(dim));
  std::vector<RsStr> rs_names = rsStrs(names);
  const RsMut<double> p = rsMut(primal), d = rsMut(dual);
  const RsMut<uint8_t> s = rsMut(status);
  const RsMut<uint8_t> t = {
      reinterpret_cast<uint8_t*>(const_cast<HighsVarType*>(integrality)),
      size_t(dim)};
  const RsOut out = rsOut(file, log_options);
  highs_rs_write_model_bound_solution(
      &out, columns, rsMut(lower), rsMut(upper),
      {rs_names.data(), rs_names.size()}, have_primal ? &p : nullptr,
      have_dual ? &d : nullptr, have_basis ? &s : nullptr,
      integrality ? &t : nullptr);
}

void writeObjectiveValue(FILE* file, const HighsLogOptions& log_options,
                         const double objective_value) {
  const RsOut out = rsOut(file, log_options);
  highs_rs_write_objective_value(&out, objective_value);
}

void writePrimalSolution(FILE* file, const HighsLogOptions& log_options,
                         const HighsLp& lp,
                         const std::vector<double>& primal_solution,
                         const bool sparse) {
  if (lp.col_names_.size() > 0)
    assert(lp.col_names_.size() == static_cast<size_t>(lp.num_col_));
  std::vector<RsStr> names = rsStrs(lp.col_names_);
  const RsOut out = rsOut(file, log_options);
  highs_rs_write_primal_solution(&out, lp.num_col_,
                                 {names.data(), names.size()},
                                 rsMut(primal_solution), sparse);
  fflush(file);
}

void writeBasisFile(FILE*& file, const HighsOptions& options, const HighsLp& lp,
                    const HighsBasis& basis) {
  if (basis.valid) {
    assert(basis.col_status.size() == static_cast<size_t>(lp.num_col_));
    assert(basis.row_status.size() == static_cast<size_t>(lp.num_row_));
    assert(lp.col_names_.size() == static_cast<size_t>(lp.num_col_));
    assert(lp.row_names_.size() == static_cast<size_t>(lp.num_row_));
  }
  const WriteModel m(lp, nullptr, nullptr, lp.objective_name_);
  const RsOut out = rsOut(file, options.log_options);
  const RsBasis b = rsBasis(basis);
  highs_rs_write_basis_file(&out, &m.v, &b);
}

HighsStatus writeMps(
    const HighsLogOptions& log_options, const std::string& filename,
    const std::string& model_name, const HighsInt& num_row,
    const HighsInt& num_col, const HighsInt& q_dim, const ObjSense& sense,
    const double& offset, const vector<double>& col_cost,
    const vector<double>& col_lower, const vector<double>& col_upper,
    const vector<double>& row_lower, const vector<double>& row_upper,
    const vector<HighsInt>& a_start, const vector<HighsInt>& a_index,
    const vector<double>& a_value, const vector<HighsInt>& q_start,
    const vector<HighsInt>& q_index, const vector<double>& q_value,
    const vector<HighsVarType>& integrality, const std::string& objective_name,
    const vector<std::string>& col_names, const vector<std::string>& row_names,
    const bool use_free_format) {
  highsLogDev(log_options, HighsLogType::kInfo,
              "writeMPS: Trying to open file %s\n", filename.c_str());
  FILE* file = fopen(filename.c_str(), "w");
  if (file == 0) {
    highsLogUser(log_options, HighsLogType::kError, "Cannot open file %s\n",
                 filename.c_str());
    return HighsStatus::kError;
  }
  highsLogDev(log_options, HighsLogType::kInfo, "writeMPS: Opened file  OK\n");
  // Check that the names are no longer than 8 characters for fixed format
  HighsInt max_name_length =
      std::max(maxNameLength(col_names), maxNameLength(row_names));
  if (!use_free_format && max_name_length > 8) {
    highsLogUser(
        log_options, HighsLogType::kError,
        "Cannot write fixed MPS with names of length (up to) %" HIGHSINT_FORMAT
        "\n",
        max_name_length);
    fclose(file);
    return HighsStatus::kError;
  }
  assert(objective_name != "");
  std::vector<RsStr> cols = rsStrs(col_names), rows = rsStrs(row_names);
  RsWriteModel v;
  v.num_col = num_col;
  v.num_row = num_row;
  v.col_cost = rsMut(col_cost);
  v.col_lower = rsMut(col_lower);
  v.col_upper = rsMut(col_upper);
  v.row_lower = rsMut(row_lower);
  v.row_upper = rsMut(row_upper);
  v.a_start = rsMut(a_start);
  v.a_index = rsMut(a_index);
  v.a_value = rsMut(a_value);
  v.sense = int(sense);
  v.offset = offset;
  v.integrality = rsMut(integrality);
  v.q_dim = q_dim;
  v.q_start = rsMut(q_start);
  v.q_index = rsMut(q_index);
  v.q_value = rsMut(q_value);
  v.col_names = {cols.data(), cols.size()};
  v.row_names = {rows.data(), rows.size()};
  v.model_name = rsStr(model_name);
  v.objective_name = rsStr(objective_name);
  v.cost_row_location = -1;
  const RsOut out = rsOut(file, log_options, false);
  highs_rs_write_mps(&out, &v);
  fclose(file);
  return HighsStatus::kOk;
}

HighsStatus FilereaderLp::writeModelToFile(const HighsOptions& options,
                                           const std::string filename,
                                           const HighsModel& model) {
  const HighsLp& lp = model.lp_;

  const bool ok_names = lp.okNames();
  assert(ok_names);
  if (!ok_names) return HighsStatus::kError;

  // Create a row-wise copy of the matrix
  HighsSparseMatrix ar_matrix = lp.a_matrix_;
  ar_matrix.ensureRowwise();

  FILE* file = fopen(filename.c_str(), "w");
  const WriteModel m(lp, model.isQp() ? &model.hessian_ : nullptr, &ar_matrix,
                     lp.objective_name_);
  const RsOut out = rsOut(file, options.log_options, false);
  highs_rs_write_lp(&out, &m.v);
  fclose(file);
  return HighsStatus::kOk;
}

#endif
