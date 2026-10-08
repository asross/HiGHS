/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file lp_data/HighsAppRust.cpp
 * @brief The objects of the highs app, whose main and loadOptions are Rust
 * (rust/src/lp_data/app.rs): a Highs instance and the loaded options, and
 * one function per step on them. Used by app/RunHighs.cpp and by the Rust
 * `crest` binary, which links this library.
 */
#include "lp_data/HighsRust.h"

#ifdef HIGHS_RUST
#include <cstdio>
#include <string>

#include "../../app/HighsAppExternalDeps.h"
#include "Highs.h"
#include "io/LoadOptions.h"

namespace {

struct AppCtx {
  Highs highs;
  HighsOptions loaded;
};

}  // namespace

extern "C" {

struct RsAppHost {
  void* ctx;
  int64_t (*op)(void* ctx, int code, int64_t arg, double x, const char* s,
                size_t n, const char* s2, size_t n2, void* out);
  RsLog log;
};

void highs_rs_app_put(void* out, const char* p, size_t n);
int highs_rs_app_main(int argc, const char* const* argv,
                      const RsAppHost* host);

static int64_t appOp(void* p, int code, int64_t arg, double x, const char* s,
                     size_t n, const char* s2, size_t n2, void* out) {
  AppCtx& a = *static_cast<AppCtx*>(p);
  Highs& highs = a.highs;
  const HighsLogOptions& report = highs.getOptions().log_options;
  const std::string str(s, n);
  auto put = [&](const std::string& t) {
    highs_rs_app_put(out, t.data(), t.size());
  };
  switch (code) {
    case 0: {
      FILE* f = arg == 2 ? stderr : stdout;
      fwrite(s, 1, n, f);
      fflush(f);
      return 0;
    }
    case 1:
      highs.logHeader();
      return 0;
    case 2:
      highs.closeLogFile();
      return 0;
    case 3:
      highs.openLogFile(a.loaded.log_file);
      return 0;
    case 4:
      highs.passOptions(a.loaded);
      return 0;
    case 5:
      highs.writeOptions("", true);
      return 0;
    case 6:
      return int64_t(highs.readModel(str));
    case 7:
      return int64_t(highs.presolve());
    case 8:
      return int64_t(highs.getModelPresolveStatus());
    case 9:
      return int64_t(highs.writePresolvedModel(
          highs.getOptions().write_presolved_model_file));
    case 10:
      return highs.getOptions().write_presolved_model_file != "";
    case 11:
      return int64_t(highs.run());
    case 12:
      // Shut down task executor for explicit release of memory
      Highs::resetGlobalScheduler(true);
      return 0;
    case 13:
      return a.loaded.output_flag;
    case 14:
      return int64_t(loadOptionsFromFile(report, a.loaded, str));
    case 15:
      writeOptionsToFile(stdout, a.loaded.log_options, a.loaded.records);
      return 0;
    case 16:
      return int64_t(setLocalOptionValue(report, str, a.loaded.log_options,
                                         a.loaded.records,
                                         std::string(s2, n2)));
    case 17:
      return int64_t(
          setLocalOptionValue(report, str, a.loaded.records, HighsInt(arg)));
    case 18:
      return int64_t(setLocalOptionValue(report, str, a.loaded.records, x));
    case 19:
      switch (arg) {
        case 0:
          put(std::to_string(HIGHS_VERSION_MAJOR) + "." +
              std::to_string(HIGHS_VERSION_MINOR) + "." +
              std::to_string(HIGHS_VERSION_PATCH));
          break;
        case 1:
          put(HIGHS_GITHASH);
          break;
        case 2:
          put(kHighsCopyrightStatement);
          break;
        case 3:
          put(HighsExternalApi::thirdPartyNoticeHeader());
          break;
        case 4:
          put(HighsExternalApi::getThirdPartyNotice<HighsExtras::appAll>());
          break;
        case 5:
          put(CLI11_VERSION);
          break;
      }
      return 0;
  }
  return 0;
}

// Creates the app's Highs instance and loaded options (with log_file
// kHighsRunLogFile, the app's default), and fills host
void* highs_app_create(RsAppHost* host) {
  AppCtx* a = new AppCtx;
  a->loaded.log_file = kHighsRunLogFile;
  host->ctx = a;
  host->op = appOp;
  host->log = rsLog(a->highs.getOptions().log_options);
  return a;
}

void highs_app_destroy(void* ctx) { delete static_cast<AppCtx*>(ctx); }

// The highs app's main
int highs_app_main(int argc, char** argv) {
  RsAppHost host;
  void* ctx = highs_app_create(&host);
  const int status = highs_rs_app_main(argc, argv, &host);
  highs_app_destroy(ctx);
  return status;
}

}  // extern "C"

#endif
