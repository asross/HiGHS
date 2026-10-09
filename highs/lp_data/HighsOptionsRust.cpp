/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file lp_data/HighsOptionsRust.cpp
 * @brief HighsOptions.cpp, HighsInfo.cpp and LoadOptions.cpp done by Rust
 * (rust/src/lp_data/options.rs, info.rs). The records stay here; each
 * call passes Rust a table of views of them.
 */
#include "lp_data/HighsRust.h"

#ifdef HIGHS_RUST
#include <cstdio>
#include <string>
#include <vector>

#include "io/LoadOptions.h"
#include "lp_data/HighsInfo.h"
#include "lp_data/HighsLpHandle.h"

namespace {

struct RsStr {
  const char* ptr;
  size_t len;
};
RsStr rsStr(const std::string& s) { return {s.data(), s.size()}; }

// options.rs: COptionRecord
struct RsOptionRecord {
  int type;
  bool advanced;
  RsStr name, description;
  void* value;
  RsStr str_value, str_default;
  bool bool_default;
  HighsInt int_lower, int_default, int_upper;
  double dbl_lower, dbl_default, dbl_upper;
};

RsOptionRecord rsOptionRecord(const OptionRecord& r) {
  RsOptionRecord v{};
  v.type = int(r.type);
  v.advanced = r.advanced;
  v.name = rsStr(r.name);
  v.description = rsStr(r.description);
  v.str_value = {nullptr, 0};
  v.str_default = {nullptr, 0};
  switch (r.type) {
    case HighsOptionType::kBool: {
      const auto& o = static_cast<const OptionRecordBool&>(r);
      v.value = o.value;
      v.bool_default = o.default_value;
      break;
    }
    case HighsOptionType::kInt: {
      const auto& o = static_cast<const OptionRecordInt&>(r);
      v.value = o.value;
      v.int_lower = o.lower_bound;
      v.int_default = o.default_value;
      v.int_upper = o.upper_bound;
      break;
    }
    case HighsOptionType::kDouble: {
      const auto& o = static_cast<const OptionRecordDouble&>(r);
      v.value = o.value;
      v.dbl_lower = o.lower_bound;
      v.dbl_default = o.default_value;
      v.dbl_upper = o.upper_bound;
      break;
    }
    default: {
      const auto& o = static_cast<const OptionRecordString&>(r);
      v.value = o.value;
      v.str_value = rsStr(*o.value);
      v.str_default = rsStr(o.default_value);
    }
  }
  return v;
}

std::vector<RsOptionRecord> rsOptionRecords(
    const std::vector<OptionRecord*>& records) {
  std::vector<RsOptionRecord> v;
  v.reserve(records.size());
  for (const OptionRecord* r : records) v.push_back(rsOptionRecord(*r));
  return v;
}

}  // namespace

extern "C" {
void highs_rs_opts_template(const RsOptionRecord* recs, size_t n, int which,
                            const RsLog* log, const void* log_options,
                            void* out);
int highs_rs_opts_default_diff(const RsOptionRecord* recs, size_t n);
}

void rsOptionsTemplate(const HighsOptions& options, int which, void* out) {
  const std::vector<RsOptionRecord> recs = rsOptionRecords(options.records);
  const RsLog log = rsLog(options.log_options);
  highs_rs_opts_template(recs.data(), recs.size(), which, &log,
                         &options.log_options, out);
}

extern "C" void highs_rs_lph_sync_options(highs_rs::LpHandle* h,
                                          const RsOptionRecord* recs,
                                          size_t n);

void rsSyncOptions(highs_rs::LpHandle* h, const HighsOptions& options) {
  const std::vector<RsOptionRecord> recs = rsOptionRecords(options.records);
  highs_rs_lph_sync_options(h, recs.data(), recs.size());
}

HighsInt rsOptionsDefaultsDiffer(const HighsOptions& options) {
  const std::vector<RsOptionRecord> recs = rsOptionRecords(options.records);
  return highs_rs_opts_default_diff(recs.data(), recs.size());
}

namespace {

// What highsOpenLogFile needs
struct OptionCtx {
  HighsLogOptions* log_options;
  std::vector<OptionRecord*>* records;
};

// options.rs: COptionHost
struct RsOptionHost {
  RsLog log;
  void* ctx;
  void (*set_string)(void*, const char*, size_t);
  void (*open_log_file)(void*, const char*, size_t);
  void (*write)(void*, const char*, size_t);
};

void setString(void* s, const char* p, size_t n) {
  static_cast<std::string*>(s)->assign(p, n);
}
void openLogFile(void* ctx, const char* p, size_t n) {
  OptionCtx& c = *static_cast<OptionCtx*>(ctx);
  highsOpenLogFile(*c.log_options, *c.records, std::string(p, n));
}
void writeFile(void* file, const char* p, size_t n) {
  fwrite(p, 1, n, static_cast<FILE*>(file));
}

RsOptionHost rsOptionHost(const HighsLogOptions& log_options,
                          OptionCtx* ctx = nullptr) {
  return {rsLog(log_options), ctx, setString, openLogFile, writeFile};
}

// info.rs: CInfoRecord
struct RsInfoRecord {
  int type;
  bool advanced;
  RsStr name, description;
  void* value;
};

RsInfoRecord rsInfoRecord(const InfoRecord& r) {
  void* value;
  if (r.type == HighsInfoType::kInt64)
    value = static_cast<const InfoRecordInt64&>(r).value;
  else if (r.type == HighsInfoType::kInt)
    value = static_cast<const InfoRecordInt&>(r).value;
  else
    value = static_cast<const InfoRecordDouble&>(r).value;
  return {int(r.type), r.advanced, rsStr(r.name), rsStr(r.description),
          value};
}

std::vector<RsInfoRecord> rsInfoRecords(
    const std::vector<InfoRecord*>& records) {
  std::vector<RsInfoRecord> v;
  v.reserve(records.size());
  for (const InfoRecord* r : records) v.push_back(rsInfoRecord(*r));
  return v;
}

}  // namespace

extern "C" {
int highs_rs_option_index(const RsLog* log, const char* name, size_t name_len,
                          const RsOptionRecord* recs, size_t n,
                          HighsInt* index);
int highs_rs_check_options(const RsLog* log, const RsOptionRecord* recs,
                           size_t n);
int highs_rs_check_option(const RsLog* log, const RsOptionRecord* rec);
int highs_rs_check_option_value_number(const RsLog* log,
                                       const RsOptionRecord* rec,
                                       HighsInt int_value,
                                       double double_value);
int highs_rs_check_option_value_string(const RsOptionHost* host,
                                       const RsOptionRecord* rec,
                                       const char* value, size_t len);
bool highs_rs_option_value_ok(const RsOptionHost* host, int which,
                              const char* name, size_t name_len,
                              const char* value, size_t value_len);
bool highs_rs_bool_from_string(const char* value, size_t len, bool* b);
void highs_rs_possible_lower_case(const char* name, size_t name_len,
                                  char* value, size_t len);
int highs_rs_set_option(const RsOptionHost* host, const char* name,
                        size_t name_len, const RsOptionRecord* recs, size_t n,
                        int kind, bool bool_value, HighsInt int_value,
                        double double_value, const char* value,
                        size_t value_len);
int highs_rs_set_option_record(const RsOptionHost* host,
                               const RsOptionRecord* rec, bool bool_value,
                               HighsInt int_value, double double_value,
                               const char* value, size_t value_len);
int highs_rs_pass_options(const RsOptionHost* host,
                          const RsOptionRecord* from, const RsOptionRecord* to,
                          size_t n);
int highs_rs_get_option_values(const RsOptionHost* host, const char* name,
                               size_t name_len, const RsOptionRecord* recs,
                               size_t n, int want, void* current, void* min,
                               void* max, void* default_value);
int highs_rs_get_option_type(const RsLog* log, const char* name,
                             size_t name_len, const RsOptionRecord* recs,
                             size_t n, int* type);
void highs_rs_reset_options(const RsOptionHost* host,
                            const RsOptionRecord* recs, size_t n);
void highs_rs_report_options(const RsOptionHost* host, void* file,
                             bool file_is_stdout, const RsOptionRecord* recs,
                             size_t n, bool all, bool only_deviations,
                             int file_type);
int highs_rs_load_options_from_file(const RsOptionHost* host,
                                    const RsOptionRecord* recs, size_t n,
                                    const char* filename, size_t len);
void highs_rs_warn_solver_invalid(const RsLog* log, const char* solver,
                                  size_t solver_len, const char* problem,
                                  size_t problem_len);
bool highs_rs_solver_valid(const char* solver, size_t len, int problem);

void highs_rs_info_invalidate(HighsInfoStruct* info, int what);
bool highs_rs_info_equal(const HighsInfoStruct* a, const HighsInfoStruct* b);
int highs_rs_info_index(const RsLog* log, const char* name, size_t name_len,
                        const RsInfoRecord* recs, size_t n, HighsInt* index);
int highs_rs_check_info(const RsLog* log, const RsInfoRecord* recs, size_t n);
int highs_rs_get_info_value(const RsLog* log, const char* name,
                            size_t name_len, bool valid,
                            const RsInfoRecord* recs, size_t n, int want,
                            void* value);
int highs_rs_get_info_type(const RsLog* log, const char* name,
                           size_t name_len, const RsInfoRecord* recs, size_t n,
                           int* type);
int highs_rs_write_info(void* file,
                        void (*write)(void*, const char*, size_t),
                        bool check_valid, bool valid, const RsInfoRecord* recs,
                        size_t n, int file_type);
}

// HighsOptions.cpp

bool optionOffChooseOnOk(const HighsLogOptions& report_log_options,
                         const string& name, const string& value) {
  const RsOptionHost host = rsOptionHost(report_log_options);
  return highs_rs_option_value_ok(&host, 0, name.data(), name.size(),
                                  value.data(), value.size());
}

bool optionOffOnOk(const HighsLogOptions& report_log_options,
                   const string& name, const string& value) {
  const RsOptionHost host = rsOptionHost(report_log_options);
  return highs_rs_option_value_ok(&host, 1, name.data(), name.size(),
                                  value.data(), value.size());
}

static bool valueOk(const HighsLogOptions& report_log_options, int which,
                    const string& value) {
  const RsOptionHost host = rsOptionHost(report_log_options);
  return highs_rs_option_value_ok(&host, which, nullptr, 0, value.data(),
                                  value.size());
}

bool optionSolverOk(const HighsLogOptions& report_log_options,
                    const string& value) {
  return valueOk(report_log_options, 2, value);
}
bool optionMipLpSolverOk(const HighsLogOptions& report_log_options,
                         const string& value) {
  return valueOk(report_log_options, 3, value);
}
bool optionMipIpmSolverOk(const HighsLogOptions& report_log_options,
                          const string& value) {
  return valueOk(report_log_options, 4, value);
}
bool optionHipoParallelTypeOk(const HighsLogOptions& report_log_options,
                              const string& value) {
  return valueOk(report_log_options, 5, value);
}
bool optionHipoSystemOk(const HighsLogOptions& report_log_options,
                        const string& value) {
  return valueOk(report_log_options, 6, value);
}
bool optionHipoOrderingOk(const HighsLogOptions& report_log_options,
                          const string& value) {
  return valueOk(report_log_options, 7, value);
}

bool boolFromString(std::string value, bool& bool_value) {
  return highs_rs_bool_from_string(value.data(), value.size(), &bool_value);
}

OptionStatus getOptionIndex(const HighsLogOptions& report_log_options,
                            const std::string& name,
                            const std::vector<OptionRecord*>& option_records,
                            HighsInt& index) {
  const RsLog log = rsLog(report_log_options);
  const auto recs = rsOptionRecords(option_records);
  return OptionStatus(highs_rs_option_index(
      &log, name.data(), name.size(), recs.data(), recs.size(), &index));
}

OptionStatus checkOptions(const HighsLogOptions& report_log_options,
                          const std::vector<OptionRecord*>& option_records) {
  const RsLog log = rsLog(report_log_options);
  const auto recs = rsOptionRecords(option_records);
  return OptionStatus(
      highs_rs_check_options(&log, recs.data(), recs.size()));
}

OptionStatus checkOption(const HighsLogOptions& report_log_options,
                         const OptionRecordInt& option) {
  const RsLog log = rsLog(report_log_options);
  const RsOptionRecord rec = rsOptionRecord(option);
  return OptionStatus(highs_rs_check_option(&log, &rec));
}

OptionStatus checkOption(const HighsLogOptions& report_log_options,
                         const OptionRecordDouble& option) {
  const RsLog log = rsLog(report_log_options);
  const RsOptionRecord rec = rsOptionRecord(option);
  return OptionStatus(highs_rs_check_option(&log, &rec));
}

OptionStatus checkOptionValue(const HighsLogOptions& report_log_options,
                              OptionRecordInt& option, const HighsInt value) {
  const RsLog log = rsLog(report_log_options);
  const RsOptionRecord rec = rsOptionRecord(option);
  return OptionStatus(
      highs_rs_check_option_value_number(&log, &rec, value, 0));
}

OptionStatus checkOptionValue(const HighsLogOptions& report_log_options,
                              OptionRecordDouble& option, const double value) {
  const RsLog log = rsLog(report_log_options);
  const RsOptionRecord rec = rsOptionRecord(option);
  return OptionStatus(
      highs_rs_check_option_value_number(&log, &rec, 0, value));
}

OptionStatus checkOptionValue(const HighsLogOptions& report_log_options,
                              OptionRecordString& option,
                              const std::string& value) {
  const RsOptionHost host = rsOptionHost(report_log_options);
  const RsOptionRecord rec = rsOptionRecord(option);
  return OptionStatus(highs_rs_check_option_value_string(
      &host, &rec, value.data(), value.size()));
}

static OptionStatus setByName(const HighsLogOptions& report_log_options,
                              const std::string& name, OptionCtx* ctx,
                              std::vector<OptionRecord*>& option_records,
                              int kind, bool b, HighsInt i, double d,
                              const std::string* s) {
  const RsOptionHost host = rsOptionHost(report_log_options, ctx);
  const auto recs = rsOptionRecords(option_records);
  return OptionStatus(highs_rs_set_option(
      &host, name.data(), name.size(), recs.data(), recs.size(), kind, b, i,
      d, s ? s->data() : nullptr, s ? s->size() : 0));
}

OptionStatus setLocalOptionValue(const HighsLogOptions& report_log_options,
                                 const std::string& name,
                                 std::vector<OptionRecord*>& option_records,
                                 const bool value) {
  return setByName(report_log_options, name, nullptr, option_records, 0,
                   value, 0, 0, nullptr);
}

OptionStatus setLocalOptionValue(const HighsLogOptions& report_log_options,
                                 const std::string& name,
                                 std::vector<OptionRecord*>& option_records,
                                 const HighsInt value) {
  return setByName(report_log_options, name, nullptr, option_records, 1,
                   false, value, 0, nullptr);
}

OptionStatus setLocalOptionValue(const HighsLogOptions& report_log_options,
                                 const std::string& name,
                                 std::vector<OptionRecord*>& option_records,
                                 const double value) {
  return setByName(report_log_options, name, nullptr, option_records, 2,
                   false, 0, value, nullptr);
}

OptionStatus setLocalOptionValue(const HighsLogOptions& report_log_options,
                                 const std::string& name,
                                 HighsLogOptions& log_options,
                                 std::vector<OptionRecord*>& option_records,
                                 const std::string& value) {
  OptionCtx ctx{&log_options, &option_records};
  return setByName(report_log_options, name, &ctx, option_records, 3, false,
                   0, 0, &value);
}

OptionStatus setLocalOptionValue(const HighsLogOptions& report_log_options,
                                 const std::string& name,
                                 HighsLogOptions& log_options,
                                 std::vector<OptionRecord*>& option_records,
                                 const char* value) {
  std::string value_as_string(value);
  return setLocalOptionValue(report_log_options, name, log_options,
                             option_records, value_as_string);
}

static OptionStatus setRecord(const HighsLogOptions& report_log_options,
                              OptionRecord& option, bool b, HighsInt i,
                              double d, const std::string* s) {
  const RsOptionHost host = rsOptionHost(report_log_options);
  const RsOptionRecord rec = rsOptionRecord(option);
  return OptionStatus(highs_rs_set_option_record(
      &host, &rec, b, i, d, s ? s->data() : nullptr, s ? s->size() : 0));
}

OptionStatus setLocalOptionValue(OptionRecordBool& option, const bool value) {
  HighsLogOptions no_log;
  return setRecord(no_log, option, value, 0, 0, nullptr);
}

OptionStatus setLocalOptionValue(const HighsLogOptions& report_log_options,
                                 OptionRecordInt& option,
                                 const HighsInt value) {
  return setRecord(report_log_options, option, false, value, 0, nullptr);
}

OptionStatus setLocalOptionValue(const HighsLogOptions& report_log_options,
                                 OptionRecordDouble& option,
                                 const double value) {
  return setRecord(report_log_options, option, false, 0, value, nullptr);
}

OptionStatus setLocalOptionValue(const HighsLogOptions& report_log_options,
                                 OptionRecordString& option,
                                 const std::string& value) {
  return setRecord(report_log_options, option, false, 0, 0, &value);
}

void possibleLowerCaseOptionValue(const std::string& name, std::string& value) {
  highs_rs_possible_lower_case(name.data(), name.size(), &value[0],
                               value.size());
}

OptionStatus passLocalOptions(const HighsLogOptions& report_log_options,
                              const HighsOptions& from_options,
                              HighsOptions& to_options) {
  const RsOptionHost host = rsOptionHost(report_log_options);
  const auto from = rsOptionRecords(from_options.records);
  const auto to = rsOptionRecords(to_options.records);
  return OptionStatus(
      highs_rs_pass_options(&host, from.data(), to.data(), to.size()));
}

static OptionStatus getValues(const HighsLogOptions& report_log_options,
                              const std::string& option,
                              const std::vector<OptionRecord*>& option_records,
                              HighsOptionType want, void* current, void* min,
                              void* max, void* default_value) {
  const RsOptionHost host = rsOptionHost(report_log_options);
  const auto recs = rsOptionRecords(option_records);
  return OptionStatus(highs_rs_get_option_values(
      &host, option.data(), option.size(), recs.data(), recs.size(),
      int(want), current, min, max, default_value));
}

OptionStatus getLocalOptionValues(
    const HighsLogOptions& report_log_options, const std::string& option,
    const std::vector<OptionRecord*>& option_records, bool* current_value,
    bool* default_value) {
  return getValues(report_log_options, option, option_records,
                   HighsOptionType::kBool, current_value, nullptr, nullptr,
                   default_value);
}

OptionStatus getLocalOptionValues(
    const HighsLogOptions& report_log_options, const std::string& option,
    const std::vector<OptionRecord*>& option_records, HighsInt* current_value,
    HighsInt* min_value, HighsInt* max_value, HighsInt* default_value) {
  return getValues(report_log_options, option, option_records,
                   HighsOptionType::kInt, current_value, min_value, max_value,
                   default_value);
}

OptionStatus getLocalOptionValues(
    const HighsLogOptions& report_log_options, const std::string& option,
    const std::vector<OptionRecord*>& option_records, double* current_value,
    double* min_value, double* max_value, double* default_value) {
  return getValues(report_log_options, option, option_records,
                   HighsOptionType::kDouble, current_value, min_value,
                   max_value, default_value);
}

OptionStatus getLocalOptionValues(
    const HighsLogOptions& report_log_options, const std::string& option,
    const std::vector<OptionRecord*>& option_records,
    std::string* current_value, std::string* default_value) {
  return getValues(report_log_options, option, option_records,
                   HighsOptionType::kString, current_value, nullptr, nullptr,
                   default_value);
}

OptionStatus getLocalOptionType(
    const HighsLogOptions& report_log_options, const std::string& option,
    const std::vector<OptionRecord*>& option_records, HighsOptionType* type) {
  const RsLog log = rsLog(report_log_options);
  const auto recs = rsOptionRecords(option_records);
  int t = 0;
  const OptionStatus status = OptionStatus(highs_rs_get_option_type(
      &log, option.data(), option.size(), recs.data(), recs.size(), &t));
  if (status == OptionStatus::kOk && type) *type = HighsOptionType(t);
  return status;
}

void resetLocalOptions(std::vector<OptionRecord*>& option_records) {
  HighsLogOptions no_log;
  const RsOptionHost host = rsOptionHost(no_log);
  const auto recs = rsOptionRecords(option_records);
  highs_rs_reset_options(&host, recs.data(), recs.size());
}

HighsStatus writeOptionsToFile(FILE* file, const HighsLogOptions& log_options,
                               const std::vector<OptionRecord*>& option_records,
                               const bool report_only_deviations,
                               const HighsFileType file_type) {
  reportOptions(file, log_options, option_records, report_only_deviations,
                file_type);
  return HighsStatus::kOk;
}

static void report(FILE* file, const HighsLogOptions& log_options,
                   const RsOptionRecord* recs, size_t n, bool all,
                   bool report_only_deviations, HighsFileType file_type) {
  const RsOptionHost host = rsOptionHost(log_options);
  highs_rs_report_options(&host, file, file == stdout, recs, n, all,
                          report_only_deviations, int(file_type));
}

void reportOptions(FILE* file, const HighsLogOptions& log_options,
                   const std::vector<OptionRecord*>& option_records,
                   const bool report_only_deviations,
                   const HighsFileType file_type) {
  const auto recs = rsOptionRecords(option_records);
  report(file, log_options, recs.data(), recs.size(), true,
         report_only_deviations, file_type);
}

static void reportOne(FILE* file, const HighsLogOptions& log_options,
                      const OptionRecord& option, bool report_only_deviations,
                      HighsFileType file_type) {
  const RsOptionRecord rec = rsOptionRecord(option);
  report(file, log_options, &rec, 1, false, report_only_deviations,
         file_type);
}

void reportOption(FILE* file, const HighsLogOptions& log_options,
                  const OptionRecordBool& option,
                  const bool report_only_deviations,
                  const HighsFileType file_type) {
  reportOne(file, log_options, option, report_only_deviations, file_type);
}
void reportOption(FILE* file, const HighsLogOptions& log_options,
                  const OptionRecordInt& option,
                  const bool report_only_deviations,
                  const HighsFileType file_type) {
  reportOne(file, log_options, option, report_only_deviations, file_type);
}
void reportOption(FILE* file, const HighsLogOptions& log_options,
                  const OptionRecordDouble& option,
                  const bool report_only_deviations,
                  const HighsFileType file_type) {
  reportOne(file, log_options, option, report_only_deviations, file_type);
}
void reportOption(FILE* file, const HighsLogOptions& log_options,
                  const OptionRecordString& option,
                  const bool report_only_deviations,
                  const HighsFileType file_type) {
  reportOne(file, log_options, option, report_only_deviations, file_type);
}

void warnSolverInvalid(const HighsOptions& options,
                       const std::string& problem_type) {
  const RsLog log = rsLog(options.log_options);
  highs_rs_warn_solver_invalid(&log, options.solver.data(),
                               options.solver.size(), problem_type.data(),
                               problem_type.size());
}
bool solverValidForLp(const std::string& solver) {
  return highs_rs_solver_valid(solver.data(), solver.size(), 0);
}
bool solverValidForMip(const std::string& solver) {
  return highs_rs_solver_valid(solver.data(), solver.size(), 1);
}
bool solverValidForQp(const std::string& solver) {
  return highs_rs_solver_valid(solver.data(), solver.size(), 2);
}

// LoadOptions.cpp

HighsLoadOptionsStatus loadOptionsFromFile(
    const HighsLogOptions& report_log_options, HighsOptions& options,
    const std::string& filename) {
  OptionCtx ctx{&options.log_options, &options.records};
  const RsOptionHost host = rsOptionHost(report_log_options, &ctx);
  const auto recs = rsOptionRecords(options.records);
  return HighsLoadOptionsStatus(highs_rs_load_options_from_file(
      &host, recs.data(), recs.size(), filename.data(), filename.size()));
}

// HighsInfo.cpp

void HighsInfo::invalidate() { highs_rs_info_invalidate(this, 0); }
void HighsInfo::invalidateKkt() { highs_rs_info_invalidate(this, 1); }
void HighsInfo::invalidatePrimalKkt() { highs_rs_info_invalidate(this, 2); }
void HighsInfo::invalidateDualKkt() { highs_rs_info_invalidate(this, 3); }
bool HighsInfo::equal(const HighsInfo& info_) const {
  return highs_rs_info_equal(this, &info_);
}

InfoStatus getInfoIndex(const HighsLogOptions& report_log_options,
                        const std::string& name,
                        const std::vector<InfoRecord*>& info_records,
                        HighsInt& index) {
  const RsLog log = rsLog(report_log_options);
  const auto recs = rsInfoRecords(info_records);
  return InfoStatus(highs_rs_info_index(&log, name.data(), name.size(),
                                        recs.data(), recs.size(), &index));
}

InfoStatus checkInfo(const HighsLogOptions& report_log_options,
                     const std::vector<InfoRecord*>& info_records) {
  const RsLog log = rsLog(report_log_options);
  const auto recs = rsInfoRecords(info_records);
  return InfoStatus(highs_rs_check_info(&log, recs.data(), recs.size()));
}

static InfoStatus getInfo(const HighsLogOptions& report_log_options,
                          const std::string& name, const bool valid,
                          const std::vector<InfoRecord*>& info_records,
                          HighsInfoType want, void* value) {
  const RsLog log = rsLog(report_log_options);
  const auto recs = rsInfoRecords(info_records);
  return InfoStatus(highs_rs_get_info_value(&log, name.data(), name.size(),
                                            valid, recs.data(), recs.size(),
                                            int(want), value));
}

InfoStatus getLocalInfoValue(const HighsLogOptions& report_log_options,
                             const std::string& name, const bool valid,
                             const std::vector<InfoRecord*>& info_records,
                             int64_t& value) {
  return getInfo(report_log_options, name, valid, info_records,
                 HighsInfoType::kInt64, &value);
}

InfoStatus getLocalInfoValue(const HighsLogOptions& report_log_options,
                             const std::string& name, const bool valid,
                             const std::vector<InfoRecord*>& info_records,
                             HighsInt& value) {
  return getInfo(report_log_options, name, valid, info_records,
                 HighsInfoType::kInt, &value);
}

InfoStatus getLocalInfoValue(const HighsLogOptions& report_log_options,
                             const std::string& name, const bool valid,
                             const std::vector<InfoRecord*>& info_records,
                             double& value) {
  return getInfo(report_log_options, name, valid, info_records,
                 HighsInfoType::kDouble, &value);
}

InfoStatus getLocalInfoType(const HighsLogOptions& report_log_options,
                            const std::string& name,
                            const std::vector<InfoRecord*>& info_records,
                            HighsInfoType& type) {
  const RsLog log = rsLog(report_log_options);
  const auto recs = rsInfoRecords(info_records);
  int t = 0;
  const InfoStatus status = InfoStatus(highs_rs_get_info_type(
      &log, name.data(), name.size(), recs.data(), recs.size(), &t));
  if (status == InfoStatus::kOk) type = HighsInfoType(t);
  return status;
}

HighsStatus writeInfoToFile(FILE* file, const bool valid, const HighsInfo& info,
                            const HighsFileType file_type) {
  return writeInfoToFile(file, valid, info.records, file_type);
}

HighsStatus writeInfoToFile(FILE* file, const bool valid,
                            const std::vector<InfoRecord*>& info_records,
                            const HighsFileType file_type) {
  const auto recs = rsInfoRecords(info_records);
  return HighsStatus(highs_rs_write_info(file, writeFile, true, valid,
                                         recs.data(), recs.size(),
                                         int(file_type)));
}

void reportInfo(FILE* file, const std::vector<InfoRecord*>& info_records,
                const HighsFileType file_type) {
  const auto recs = rsInfoRecords(info_records);
  highs_rs_write_info(file, writeFile, false, true, recs.data(), recs.size(),
                      int(file_type));
}

static void reportInfoRecord(FILE* file, const InfoRecord& info,
                             const HighsFileType file_type) {
  const RsInfoRecord rec = rsInfoRecord(info);
  highs_rs_write_info(file, writeFile, false, true, &rec, 1, int(file_type));
}

void reportInfo(FILE* file, const InfoRecordInt64& info,
                const HighsFileType file_type) {
  reportInfoRecord(file, info, file_type);
}
void reportInfo(FILE* file, const InfoRecordInt& info,
                const HighsFileType file_type) {
  reportInfoRecord(file, info, file_type);
}
void reportInfo(FILE* file, const InfoRecordDouble& info,
                const HighsFileType file_type) {
  reportInfoRecord(file, info, file_type);
}

#endif
