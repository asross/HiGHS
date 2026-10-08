/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "io/Filereader.h"

#include <cctype>

#include "io/FilereaderLp.h"
#include "io/FilereaderMps.h"
#include "io/HighsIO.h"

// convert string to lower-case, modifies string
static inline void tolower(std::string& s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
}

#ifndef HIGHS_RUST
Filereader* Filereader::getFilereader(const HighsLogOptions& log_options,
                                      const std::string filename) {
  Filereader* reader;
  std::string extension = getFilenameExt(filename);
  if (extension == "gz") {
#ifdef ZLIB_FOUND
    extension = getFilenameExt(filename.substr(0, filename.size() - 3));
#else
    highsLogUser(log_options, HighsLogType::kError,
                 "HiGHS build without zlib support. Cannot read .gz file.\n",
                 filename.c_str());
    reader = NULL;
#endif
  }
  std::string lower_case_extension = extension;
  tolower(lower_case_extension);
  if (lower_case_extension.compare("mps") == 0) {
    reader = new FilereaderMps();
  } else if (lower_case_extension.compare("lp") == 0) {
    reader = new FilereaderLp();
  } else {
    reader = NULL;
  }
  return reader;
}

void interpretFilereaderRetcode(const HighsLogOptions& log_options,
                                const std::string& filename,
                                const FilereaderRetcode code) {
  switch (code) {
    case FilereaderRetcode::kOk:
    case FilereaderRetcode::kWarning:
      break;
    case FilereaderRetcode::kFileNotFound:
      highsLogUser(log_options, HighsLogType::kError, "File %s not found\n",
                   filename.c_str());
      break;
    case FilereaderRetcode::kParserError:
      highsLogUser(log_options, HighsLogType::kError,
                   "Parser error reading %s\n", filename.c_str());
      break;
    case FilereaderRetcode::kNotImplemented:
      highsLogUser(log_options, HighsLogType::kError,
                   "Parser not implemented for %s", filename.c_str());
      break;
    case FilereaderRetcode::kTimeout:
      highsLogUser(log_options, HighsLogType::kError,
                   "Parser reached timeout\n", filename.c_str());
      break;
  }
}

std::string extractModelName(const std::string& filename) {
  // Extract model name
  std::string name = filename;
  std::size_t found = name.find_last_of("/\\");
  if (found < name.size()) name = name.substr(found + 1);
  found = name.find_last_of(".");
  if (name.substr(found + 1) == "gz") {
    name.erase(found, name.size() - found);
    found = name.find_last_of(".");
  }
  if (found < name.size()) name.erase(found, name.size() - found);
  return name;
}
#else
// The file name logic is Rust's (rust/src/lp_data/model_utils.rs)
#include "lp_data/HighsRust.h"

extern "C" int highs_rs_filereader(const RsLog* log, int which,
                                   const char* name, size_t len, int arg,
                                   void (*set)(void*, const char*, size_t),
                                   void* ctx);

static void setString(void* s, const char* p, size_t n) {
  static_cast<std::string*>(s)->assign(p, n);
}

Filereader* Filereader::getFilereader(const HighsLogOptions& log_options,
                                      const std::string filename) {
  const RsLog log = rsLog(log_options);
#ifdef ZLIB_FOUND
  const int zlib = 1;
#else
  const int zlib = 0;
#endif
  switch (highs_rs_filereader(&log, 0, filename.data(), filename.size(), zlib,
                              setString, nullptr)) {
    case 1:
      return new FilereaderMps();
    case 2:
      return new FilereaderLp();
    default:
      return NULL;
  }
}

void interpretFilereaderRetcode(const HighsLogOptions& log_options,
                                const std::string& filename,
                                const FilereaderRetcode code) {
  const RsLog log = rsLog(log_options);
  highs_rs_filereader(&log, 1, filename.data(), filename.size(), int(code),
                      setString, nullptr);
}

std::string extractModelName(const std::string& filename) {
  std::string name;
  highs_rs_filereader(nullptr, 2, filename.data(), filename.size(), 0,
                      setString, &name);
  return name;
}
#endif
