/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file io/FilereaderMps.cpp
 * @brief
 */
#include "io/FilereaderMps.h"

#include "io/HMPSIO.h"
#include "io/HMpsFF.h"
#include "lp_data/HighsLp.h"
#include "lp_data/HighsLpUtils.h"
#include "lp_data/HighsModelUtils.h"

using free_format_parser::HMpsFF;

#ifdef HIGHS_RUST
#ifdef ZLIB_FOUND
#include "../extern/zstr/zstr.hpp"
#endif

#include "io/RustFfi.h"

// The free format MPS parser is in rust/src/io/mps.rs. This mirrors its
// repr(C) view.
struct RsMpsView {
  int status;  // FreeFormatParserReturnCode
  bool warning_issued;
  bool maximize;
  int num_row, num_col, cost_row_location, q_dim;
  double offset;
  RsSlice<int> a_start, a_index;
  RsSlice<double> a_value, col_cost, col_lower, col_upper, row_lower,
      row_upper;
  RsSlice<HighsVarType> integrality;
  RsSlice<int> q_start, q_index;
  RsSlice<double> q_value;
  RsSlice<char> objective_name;
  RsSlice<RsSlice<char>> row_names, col_names;
  RsSlice<RsMessage> messages;
};
extern "C" void* highs_rs_mps_read(const char* buf, size_t len,
                                   double time_limit, RsMpsView* view);
extern "C" void highs_rs_mps_free(void* handle);

// HMpsFF::loadProblem with the parsing done by Rust
static FreeFormatParserReturnCode loadProblemRust(
    const HighsLogOptions& log_options, const std::string& filename,
    const double time_limit, HighsModel& model, bool& warning_issued) {
  static_assert(sizeof(HighsInt) == 4, "the Rust reader returns 32-bit ints");
  static_assert(sizeof(HighsVarType) == 1, "the Rust reader returns bytes");
  highsLogDev(log_options, HighsLogType::kInfo,
              "readMPS: Trying to open file %s\n", filename.c_str());
#ifdef ZLIB_FOUND
  zstr::ifstream f;
  try {
    f.open(filename.c_str(), std::ios::in);
  } catch (const strict_fstream::Exception& e) {
    highsLogDev(log_options, HighsLogType::kInfo, e.what());
    return FreeFormatParserReturnCode::kFileNotFound;
  }
#else
  std::ifstream f;
  f.open(filename.c_str(), std::ios::in);
#endif
  if (!f.is_open()) {
    highsLogDev(log_options, HighsLogType::kInfo,
                "readMPS: Not opened file OK\n");
    return FreeFormatParserReturnCode::kFileNotFound;
  }
  std::string buf;
  std::vector<char> chunk(1 << 16);
  while (f.read(chunk.data(), chunk.size()) || f.gcount())
    buf.append(chunk.data(), f.gcount());
  f.close();

  RsMpsView v;
  void* handle = highs_rs_mps_read(buf.data(), buf.size(), time_limit, &v);
  for (size_t i = 0; i < v.messages.len; i++) {
    const RsMessage& m = v.messages.ptr[i];
    const std::string text = m.text.str();
    if (m.kind == 0)
      highsLogDev(log_options, HighsLogType::kInfo, "%s", text.c_str());
    else
      highsLogUser(log_options, HighsLogType(m.kind), "%s", text.c_str());
  }
  warning_issued = v.warning_issued;
  const auto result = FreeFormatParserReturnCode(v.status);
  if (result == FreeFormatParserReturnCode::kSuccess) {
    HighsLp& lp = model.lp_;
    HighsHessian& hessian = model.hessian_;
    lp.num_row_ = v.num_row;
    lp.num_col_ = v.num_col;
    lp.sense_ = v.maximize ? ObjSense::kMaximize : ObjSense::kMinimize;
    lp.offset_ = v.offset;
    lp.a_matrix_.format_ = MatrixFormat::kColwise;
    lp.a_matrix_.start_ = v.a_start.vec();
    lp.a_matrix_.index_ = v.a_index.vec();
    lp.a_matrix_.value_ = v.a_value.vec();
    if (lp.a_matrix_.start_.size() == 0) lp.a_matrix_.clear();
    lp.col_cost_ = v.col_cost.vec();
    lp.col_lower_ = v.col_lower.vec();
    lp.col_upper_ = v.col_upper.vec();
    lp.row_lower_ = v.row_lower.vec();
    lp.row_upper_ = v.row_upper.vec();
    lp.objective_name_ = v.objective_name.str();
    lp.row_names_.resize(v.row_names.len);
    for (size_t i = 0; i < v.row_names.len; i++)
      lp.row_names_[i] = v.row_names.ptr[i].str();
    lp.col_names_.resize(v.col_names.len);
    for (size_t i = 0; i < v.col_names.len; i++)
      lp.col_names_[i] = v.col_names.ptr[i].str();
    // Empty unless the model is a MIP
    if (v.integrality.len) lp.integrality_ = v.integrality.vec();
    hessian.dim_ = v.q_dim;
    hessian.format_ = HessianFormat::kSquare;
    hessian.start_ = v.q_start.vec();
    hessian.index_ = v.q_index.vec();
    hessian.value_ = v.q_value.vec();
    if (hessian.start_.size() == 0) hessian.clear();
    lp.objective_name_ = findModelObjectiveName(&lp, &hessian);
    lp.cost_row_location_ = v.cost_row_location;
  }
  highs_rs_mps_free(handle);
  return result;
}
#endif

FilereaderRetcode FilereaderMps::readModelFromFile(const HighsOptions& options,
                                                   const std::string filename,
                                                   HighsModel& model) {
  HighsLp& lp = model.lp_;
  HighsHessian& hessian = model.hessian_;
  // if free format parser
  // Parse file and return status.
#ifdef HIGHS_RUST
  // The fixed format reader is not in Crestline: always the free one
  if (!options.mps_parser_type_free)
    highsLogUser(options.log_options, HighsLogType::kWarning,
                 "The fixed format MPS reader is not available in this "
                 "build: using the free format reader\n");
  {
#else
  if (options.mps_parser_type_free) {
#endif
    HMpsFF parser{};
    if (options.time_limit < kHighsInf && options.time_limit > 0)
      parser.time_limit_ = options.time_limit;

#ifdef HIGHS_RUST
    bool warning_issued = false;
    FreeFormatParserReturnCode result =
        loadProblemRust(options.log_options, filename, parser.time_limit_,
                        model, warning_issued);
    parser.warning_issued_ = warning_issued;
#else
    FreeFormatParserReturnCode result =
        parser.loadProblem(options.log_options, filename, model);
#endif
    switch (result) {
      case FreeFormatParserReturnCode::kSuccess:
        lp.ensureColwise();
        assert(model.lp_.objective_name_ != "");
        return parser.warning_issued_ ? FilereaderRetcode::kWarning
                                      : FilereaderRetcode::kOk;
      case FreeFormatParserReturnCode::kParserError:
        return FilereaderRetcode::kParserError;
      case FreeFormatParserReturnCode::kFileNotFound:
        return FilereaderRetcode::kFileNotFound;
      case FreeFormatParserReturnCode::kFixedFormat:
#ifdef HIGHS_RUST
        highsLogUser(options.log_options, HighsLogType::kError,
                     "Free format reader has detected row/col names with "
                     "spaces: the fixed format MPS reader is not available "
                     "in this build\n");
        return FilereaderRetcode::kParserError;
#else
        highsLogUser(options.log_options, HighsLogType::kWarning,
                     "Free format reader has detected row/col names with "
                     "spaces: switching to fixed format parser\n");
        break;
#endif
      case FreeFormatParserReturnCode::kTimeout:
        highsLogUser(options.log_options, HighsLogType::kWarning,
                     "Free format reader reached time_limit while parsing "
                     "the input file\n");
        return FilereaderRetcode::kTimeout;
    }
  }

#ifdef HIGHS_RUST
  assert(false);
  return FilereaderRetcode::kParserError;
#else
  // else use fixed format parser
  //
  // If the fixed format parser has had to be used, then a warning was
  // issued, otherwise no warning has yet been issued
  bool warning_issued = options.mps_parser_type_free;
  FilereaderRetcode return_code =
      readMps(options.log_options, filename, -1, -1, lp.num_row_, lp.num_col_,
              lp.sense_, lp.offset_, lp.a_matrix_.start_, lp.a_matrix_.index_,
              lp.a_matrix_.value_, lp.col_cost_, lp.col_lower_, lp.col_upper_,
              lp.row_lower_, lp.row_upper_, lp.integrality_, lp.objective_name_,
              lp.col_names_, lp.row_names_, hessian.dim_, hessian.start_,
              hessian.index_, hessian.value_, lp.cost_row_location_,
              warning_issued, options.keep_n_rows);
  if (return_code == FilereaderRetcode::kOk) lp.ensureColwise();
  // Comment on existence of names with spaces
  hasNamesWithSpaces(options.log_options, lp);
  assert(model.lp_.objective_name_ != "");
  if (return_code == FilereaderRetcode::kOk && warning_issued)
    return_code = FilereaderRetcode::kWarning;
  return return_code;
#endif
}

HighsStatus FilereaderMps::writeModelToFile(const HighsOptions& options,
                                            const std::string filename,
                                            const HighsModel& model) {
  assert(model.lp_.a_matrix_.isColwise());
  return writeModelAsMps(options, filename, model,
                         options.mps_parser_type_free);
}
