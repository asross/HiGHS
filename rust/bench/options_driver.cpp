// Exercises option and info handling through the Highs API, for comparing
// a pure C++ and a HIGHS_RUST build (rust/bench/options_compare.sh): run
// in an empty directory with the path of check/instances as argument.
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "Highs.h"
#include "io/LoadOptions.h"

static void file(const std::string& name, const std::string& text) {
  std::ofstream f(name);
  f << text;
}

int main(int argc, char** argv) {
  const std::string instances = argc > 1 ? argv[1] : "check/instances";
  Highs h;
  h.setOptionValue("timeless_log", true);
  const std::vector<std::string> names = {
      "output_flag",   "threads",          "simplex_strategy",
      "time_limit",    "mip_rel_gap",      "presolve",
      "solver",        "mip_lp_solver",    "mip_ipm_solver",
      "hipo_ordering", "hipo_system",      "hipo_parallel_type",
      "ranging",       "run_crossover",    "parallel",
      "model_file",    "solution_file",    "no_such_option",
      "random_seed",   "objective_bound",  "user_bound_scale",
      "write_iis_model_file", "icrash_strategy", "log_dev_level"};
  const std::vector<std::string> values = {
      "  on ", "TRUE", "yes", "F", "0", "1", "1e3", "12 ", "+-3", "",
      "inf", "-INF", "1e400", "nan", "0x10", "1.5.2", "5.", "-0", "-1",
      "99999999999", "2147483648", "1e-12", "3.25", "Choose", "IPX",
      "simplex", "hipo", "AMD", "augmented", "tree", "MiXed.SOL", "mip",
      "Off", "pdlp", "  7  ", "1e", ".5", "-.5e1"};
  for (int dev : {0, 3}) {
    printf("=== log_dev_level %d\n", dev);
    h.setOptionValue("log_dev_level", dev);
    for (const auto& n : names)
      for (const auto& v : values) {
        HighsStatus s = h.setOptionValue(n, v);
        printf("set %s \"%s\" -> %d\n", n.c_str(), v.c_str(), int(s));
        // Keep the messages coming
        h.setOptionValue("output_flag", true);
        h.setOptionValue("log_dev_level", dev);
      }
    h.resetOptions();
    h.setOptionValue("timeless_log", true);
  }
  // Typed setters, also of the wrong type
  for (const auto& n : names) {
    h.setOptionValue("output_flag", true);
    printf("%s: bool %d int %d int-big %d double %d double-neg %d\n",
           n.c_str(), int(h.setOptionValue(n, true)),
           int(h.setOptionValue(n, HighsInt{3})),
           int(h.setOptionValue(n, HighsInt{1 << 30})),
           int(h.setOptionValue(n, 0.125)), int(h.setOptionValue(n, -2.5)));
  }
  h.setOptionValue("output_flag", true);
  h.setOptionValue("log_dev_level", 0);
  // Getters, also of the wrong type
  for (const auto& n : names) {
    bool b;
    HighsInt i, imin, imax, idef;
    double d, dmin, dmax, ddef;
    std::string s, sdef;
    HighsOptionType t = HighsOptionType::kBool;
    HighsStatus sb = h.getOptionValue(n, b);
    HighsStatus si = h.getOptionValue(n, i);
    HighsStatus sd = h.getOptionValue(n, d);
    HighsStatus ss = h.getOptionValue(n, s);
    HighsStatus st = h.getOptionType(n, t);
    printf("get %s: %d %d %d %d type %d %d\n", n.c_str(), int(sb), int(si),
           int(sd), int(ss), int(st), int(t));
    if (sb == HighsStatus::kOk) printf("  %d\n", b);
    if (si == HighsStatus::kOk) printf("  %d\n", int(i));
    if (sd == HighsStatus::kOk) printf("  %.17g\n", d);
    if (ss == HighsStatus::kOk) printf("  \"%s\"\n", s.c_str());
    if (getLocalOptionValues(h.getOptions().log_options, n,
                             h.getOptions().records, &i, &imin, &imax,
                             &idef) == OptionStatus::kOk)
      printf("  int %d [%d, %d] default %d\n", int(i), int(imin), int(imax),
             int(idef));
    if (getLocalOptionValues(h.getOptions().log_options, n,
                             h.getOptions().records, &d, &dmin, &dmax,
                             &ddef) == OptionStatus::kOk)
      printf("  double %g [%g, %g] default %g\n", d, dmin, dmax, ddef);
    if (getLocalOptionValues(h.getOptions().log_options, n,
                             h.getOptions().records, &s,
                             &sdef) == OptionStatus::kOk)
      printf("  string \"%s\" default \"%s\"\n", s.c_str(), sdef.c_str());
  }
  std::string name;
  for (HighsInt k = -1; k < 1000; k += 37)
    if (h.getOptionName(k, &name) == HighsStatus::kOk)
      printf("option %d: %s\n", int(k), name.c_str());
  // Options files
  file("good.set",
       "# comment\n\n   \npresolve = Off\nthreads=2\n  time_limit = 1e3 \n"
       "solver = 'simplex'\nmip_rel_gap = \"0.5\"\r\noutput_flag = T\n"
       "solution_file = Sol.TXT\nsimplex_strategy=4");
  file("bad_line.set", "presolve = off\nthreads\n");
  file("bad_eq.set", "threads =\n");
  file("bad_name.set", "presolve = on\nnot_an_option = 3\n");
  file("bad_value.set", "threads = -4\n");
  file("bad_type.set", "time_limit = abc\n");
  file("empty.set", "");
  file("comment_only.set", "# x = 1\n#\n");
  for (const std::string f :
       {"good.set", "bad_line.set", "bad_eq.set", "bad_name.set",
        "bad_value.set", "bad_type.set", "empty.set", "comment_only.set",
        "missing.set", "", "."}) {
    HighsStatus s = h.readOptions(f);
    printf("readOptions(%s) -> %d\n", f.c_str(), int(s));
  }
  HighsOptions loaded;
  printf("load empty name -> %d\n",
         int(loadOptionsFromFile(h.getOptions().log_options, loaded, "")));
  // Writing options in each format
  h.writeOptions("", true);
  h.writeOptions("");
  h.writeOptions("opts_full.set");
  h.writeOptions("opts_dev.set", true);
  h.writeOptions("opts.md");
  h.writeOptions("opts.html");
  h.writeOptions("opts_dev.md", true);
  reportOptions(stdout, h.getOptions().log_options, h.getOptions().records,
                true, HighsFileType::kMps);
  FILE* fo = fopen("report_minimal.txt", "w");
  reportOptions(fo, h.getOptions().log_options, h.getOptions().records,
                false, HighsFileType::kMinimal);
  fclose(fo);
  printf("checkOptions %d\n", int(checkOptions(h.getOptions().log_options,
                                               h.getOptions().records)));
  // passOptions: an illegal value, then legal ones
  HighsOptions o;
  o.threads = -5;
  printf("passOptions bad -> %d\n", int(h.passOptions(o)));
  o.threads = 1;
  o.presolve = "MaYbE";
  printf("passOptions bad string -> %d\n", int(h.passOptions(o)));
  o.presolve = "OFF";
  o.mip_rel_gap = 0.25;
  o.solution_file = "X.Y";
  printf("passOptions ok -> %d\n", int(h.passOptions(o)));
  h.writeOptions("", true);
  // Log file
  h.setOptionValue("log_file", "run.log");
  h.setOptionValue("log_file", "run.log");
  h.setOptionValue("log_file", " run2.log ");
  // Info before and after a solve
  printf("writeInfo invalid -> %d\n", int(h.writeInfo("")));
  h.writeInfo("info_invalid.md");
  h.resetOptions();
  h.setOptionValue("timeless_log", true);
  h.setOptionValue("presolve", "off");
  h.readModel(instances + "/afiro.mps");
  h.run();
  h.writeInfo("");
  h.writeInfo("info_full.txt");
  h.writeInfo("info.md");
  for (const std::string n :
       {"mip_node_count", "simplex_iteration_count", "objective_function_value",
        "mip_gap", "no_such_info", "primal_solution_status"}) {
    int64_t l = 0;
    HighsInt i = 0;
    double d = 0;
    HighsInfoType t = HighsInfoType::kInt;
    printf("info %s: %d %d %d type %d\n", n.c_str(),
           int(h.getInfoValue(n, l)), int(h.getInfoValue(n, i)),
           int(h.getInfoValue(n, d)), int(h.getInfoType(n, t)));
    printf("  %lld %d %.17g %d\n", (long long)l, int(i), d, int(t));
  }
  const HighsInfo& info = h.getInfo();
  HighsInfo copy = info;
  printf("equal %d\n", copy.equal(info));
  copy.invalidate();
  printf("equal after invalidate %d %d %g\n", copy.equal(info),
         int(copy.num_primal_infeasibilities), copy.mip_gap);
  printf("checkInfo %d\n",
         int(checkInfo(h.getOptions().log_options, info.records)));
  h.clearSolver();
  printf("info invalid get %d\n",
         int(h.getInfoValue("mip_gap", copy.mip_gap)));
  return 0;
}
