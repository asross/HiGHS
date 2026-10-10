// Prints rust/src/lp_data/option_records.rs from a HighsOptions
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <string>

#include "Highs.h"

static std::string lit(const std::string& s) {
  std::string o = "\"";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\')
      o += '\\', o += char(c);
    else if (c == '\n')
      o += "\\n";
    else if (c < 32 || c >= 127) {
      char b[8];
      snprintf(b, sizeof b, "\\x%02x", c);
      o += b;
    } else
      o += char(c);
  }
  return o + "\"";
}

static std::string dbl(double x) {
  if (std::isinf(x)) return x > 0 ? "f64::INFINITY" : "f64::NEG_INFINITY";
  char b[64];
  snprintf(b, sizeof b, "%.17g", x);
  std::string s = b;
  if (s.find_first_of(".eEn") == std::string::npos) s += ".0";
  return s;
}

int main() {
  HighsOptions o;
  printf(
      "//! The option records' metadata (HighsOptions.h: names, types, "
      "descriptions,\n//! bounds and defaults), generated from a C++ "
      "HighsOptions by\n//! rust/bench/gen_option_records.cpp; a test "
      "checks the defaults against\n//! `Opts`. Crest's options API works "
      "on these (the C++ builds use their own\n//! records).\n\n");
  printf("use super::options::{BOOL, DOUBLE, INT, STRING};\n\n");
  printf("/// An option record's metadata\npub struct Meta {\n    pub type_: i32,\n    pub advanced: bool,\n    pub name: &'static str,\n    pub description: &'static str,\n    pub bool_default: bool,\n    pub int: (i32, i32, i32),\n    pub dbl: (f64, f64, f64),\n    pub str_default: &'static str,\n}\n\n");
  printf("/// HighsOptions::records' metadata, in their order\npub const RECORDS: &[Meta] = &[\n");
  for (OptionRecord* r : o.records) {
    printf("    Meta {\n        type_: %s,\n        advanced: %s,\n        name: %s,\n        description: %s,\n",
           r->type == HighsOptionType::kBool     ? "BOOL"
           : r->type == HighsOptionType::kInt    ? "INT"
           : r->type == HighsOptionType::kDouble ? "DOUBLE"
                                                 : "STRING",
           r->advanced ? "true" : "false", lit(r->name).c_str(),
           lit(r->description).c_str());
    bool b = false;
    int il = 0, id = 0, iu = 0;
    double dl = 0, dd = 0, du = 0;
    std::string sd;
    switch (r->type) {
      case HighsOptionType::kBool:
        b = ((OptionRecordBool*)r)->default_value;
        break;
      case HighsOptionType::kInt: {
        auto* x = (OptionRecordInt*)r;
        il = x->lower_bound, id = x->default_value, iu = x->upper_bound;
        break;
      }
      case HighsOptionType::kDouble: {
        auto* x = (OptionRecordDouble*)r;
        dl = x->lower_bound, dd = x->default_value, du = x->upper_bound;
        break;
      }
      default:
        sd = ((OptionRecordString*)r)->default_value;
    }
    printf("        bool_default: %s,\n        int: (%d, %d, %d),\n        dbl: (%s, %s, %s),\n        str_default: %s,\n    },\n",
           b ? "true" : "false", il, id, iu, dbl(dl).c_str(), dbl(dd).c_str(),
           dbl(du).c_str(), lit(sd).c_str());
  }
  printf("];\n");
  return 0;
}
