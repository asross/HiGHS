// Exercises the Highs API (runs with option variations, model
// modification, rays, basis inverse and tableau rows, ranging, setSolution,
// setBasis, presolve and postsolve), printing results at full precision,
// for comparing a pure C++ and a HIGHS_RUST build (rust/bench/api_compare.sh):
// run in an empty directory with the path of check/instances as argument.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "Highs.h"

static std::string g_instances;

static void vec(const char* name, const std::vector<double>& v) {
  printf("%s[%d]", name, int(v.size()));
  for (double x : v) printf(" %.17g", x);
  printf("\n");
}

static void ivec(const char* name, const std::vector<HighsInt>& v) {
  printf("%s[%d]", name, int(v.size()));
  for (HighsInt x : v) printf(" %d", int(x));
  printf("\n");
}

// A checksum of a long vector, with a few entries
static void sum(const char* name, const std::vector<double>& v) {
  double s = 0, a = 0;
  for (size_t i = 0; i < v.size(); i++) {
    s += v[i] * double(1 + i % 7);
    a += std::fabs(v[i]);
  }
  printf("%s[%d] sum %.17g abs %.17g", name, int(v.size()), s, a);
  for (size_t i = 0; i < v.size() && i < 3; i++) printf(" %.17g", v[i]);
  printf("\n");
}

static void report(Highs& h, const char* what, HighsStatus status) {
  const HighsInfo& info = h.getInfo();
  printf("== %s: status %d model %s\n", what, int(status),
         h.modelStatusToString(h.getModelStatus()).c_str());
  printf("info valid %d obj %.17g simplex %d ipm %d crossover %d pdlp %d "
         "qp %d\n",
         int(info.valid), info.objective_function_value,
         int(info.simplex_iteration_count), int(info.ipm_iteration_count),
         int(info.crossover_iteration_count), int(info.pdlp_iteration_count),
         int(info.qp_iteration_count));
  printf("info pss %d dss %d bv %d npi %d mpi %.17g spi %.17g ndi %d mdi "
         "%.17g sdi %.17g pdoe %.17g mip %lld %.17g %.17g %.17g\n",
         int(info.primal_solution_status), int(info.dual_solution_status),
         int(info.basis_validity), int(info.num_primal_infeasibilities),
         info.max_primal_infeasibility, info.sum_primal_infeasibilities,
         int(info.num_dual_infeasibilities), info.max_dual_infeasibility,
         info.sum_dual_infeasibilities, info.primal_dual_objective_error,
         (long long)info.mip_node_count, info.mip_dual_bound, info.mip_gap,
         info.max_integrality_violation);
  const HighsSolution& s = h.getSolution();
  printf("solution %d %d\n", int(s.value_valid), int(s.dual_valid));
  sum("col_value", s.col_value);
  sum("row_value", s.row_value);
  sum("col_dual", s.col_dual);
  sum("row_dual", s.row_dual);
  const HighsBasis& b = h.getBasis();
  printf("basis valid %d alien %d useful %d was_alien %d", int(b.valid),
         int(b.alien), int(b.useful), int(b.was_alien));
  long long bs = 0;
  for (size_t i = 0; i < b.col_status.size(); i++)
    bs += (i % 13 + 1) * int(b.col_status[i]);
  for (size_t i = 0; i < b.row_status.size(); i++)
    bs += (i % 11 + 3) * int(b.row_status[i]);
  printf(" %d %d %lld\n", int(b.col_status.size()), int(b.row_status.size()),
         bs);
  const HighsRunData& r = h.getRunData();
  printf("run data %d %d %d %d %d\n", int(r.valid),
         int(r.presolved_model_num_col), int(r.presolved_model_num_row),
         int(r.presolved_model_num_nz),
         int(r.num_simplex_iterations_after_postsolve));
}

static void rays(Highs& h) {
  const HighsLp& lp = h.getLp();
  bool has = false;
  std::vector<double> ray(lp.num_row_ + lp.num_col_ + 1, 0);
  HighsStatus s = h.getDualRay(has, ray.data());
  printf("dual ray %d %d\n", int(s), int(has));
  if (has) sum("dual_ray", std::vector<double>(ray.begin(), ray.begin() + lp.num_row_));
  std::vector<double> pray(lp.num_col_ + 1, 0);
  s = h.getPrimalRay(has, pray.data());
  printf("primal ray %d %d\n", int(s), int(has));
  if (has) sum("primal_ray", std::vector<double>(pray.begin(), pray.begin() + lp.num_col_));
  s = h.getDualUnboundednessDirection(has, pray.data());
  printf("dual unboundedness direction %d %d\n", int(s), int(has));
  if (has) sum("dud", std::vector<double>(pray.begin(), pray.begin() + lp.num_col_));
}

static void tableau(Highs& h) {
  const HighsLp& lp = h.getLp();
  if (!h.hasInvert()) {
    printf("no invert\n");
    return;
  }
  std::vector<HighsInt> basic(lp.num_row_);
  HighsStatus s = h.getBasicVariables(basic.data());
  printf("basic variables %d\n", int(s));
  ivec("basic", basic);
  std::vector<double> v(std::max(lp.num_row_, lp.num_col_));
  std::vector<HighsInt> idx(std::max(lp.num_row_, lp.num_col_));
  for (HighsInt r = 0; r < lp.num_row_; r += std::max(HighsInt(1), lp.num_row_ / 4)) {
    HighsInt nz = -1;
    s = h.getBasisInverseRow(r, v.data(), &nz, idx.data());
    printf("binv row %d: %d nz %d\n", int(r), int(s), int(nz));
    sum("binv_row", std::vector<double>(v.begin(), v.begin() + lp.num_row_));
    s = h.getBasisInverseCol(r, v.data(), &nz, idx.data());
    printf("binv col %d: %d nz %d\n", int(r), int(s), int(nz));
    sum("binv_col", std::vector<double>(v.begin(), v.begin() + lp.num_row_));
    std::vector<double> rv(lp.num_col_);
    s = h.getReducedRow(r, rv.data(), &nz, idx.data());
    printf("reduced row %d: %d nz %d\n", int(r), int(s), int(nz));
    sum("reduced_row", rv);
  }
  for (HighsInt c = 0; c < lp.num_col_; c += std::max(HighsInt(1), lp.num_col_ / 4)) {
    HighsInt nz = -1;
    s = h.getReducedColumn(c, v.data(), &nz, idx.data());
    printf("reduced col %d: %d nz %d\n", int(c), int(s), int(nz));
    sum("reduced_col", std::vector<double>(v.begin(), v.begin() + lp.num_row_));
  }
  std::vector<double> rhs(lp.num_row_);
  for (HighsInt i = 0; i < lp.num_row_; i++) rhs[i] = 1.0 / (1 + i);
  s = h.getBasisSolve(rhs.data(), v.data());
  printf("basis solve %d\n", int(s));
  sum("bsolve", std::vector<double>(v.begin(), v.begin() + lp.num_row_));
  s = h.getBasisTransposeSolve(rhs.data(), v.data());
  printf("basis transpose solve %d\n", int(s));
  sum("btsolve", std::vector<double>(v.begin(), v.begin() + lp.num_row_));
  // Out of range
  s = h.getBasisInverseRow(-1, v.data());
  printf("binv row -1: %d\n", int(s));
  s = h.getReducedColumn(lp.num_col_, v.data());
  printf("reduced col n: %d\n", int(s));
}

static void ranging(Highs& h) {
  HighsRanging r;
  HighsStatus s = h.getRanging(r);
  printf("ranging %d valid %d\n", int(s), int(r.valid));
  if (!r.valid) return;
  auto rec = [](const char* n, const HighsRangingRecord& x) {
    std::string a = std::string(n) + ".value";
    sum(a.c_str(), x.value_);
    a = std::string(n) + ".objective";
    sum(a.c_str(), x.objective_);
    a = std::string(n) + ".in";
    long long si = 0, so = 0;
    for (size_t i = 0; i < x.in_var_.size(); i++)
      si += (i % 5 + 1) * x.in_var_[i], so += (i % 3 + 1) * x.ou_var_[i];
    printf("%s %lld %lld\n", a.c_str(), si, so);
  };
  rec("col_cost_up", r.col_cost_up);
  rec("col_cost_dn", r.col_cost_dn);
  rec("col_bound_up", r.col_bound_up);
  rec("col_bound_dn", r.col_bound_dn);
  rec("row_bound_up", r.row_bound_up);
  rec("row_bound_dn", r.row_bound_dn);
}

static Highs* fresh() {
  Highs* h = new Highs;
  h->setOptionValue("timeless_log", true);
  return h;
}

static void solveFile(const std::string& file,
                      const std::vector<std::pair<std::string, std::string>>& opts,
                      bool extras) {
  Highs* h = fresh();
  printf("\n######## %s", file.c_str());
  for (auto& o : opts) printf(" %s=%s", o.first.c_str(), o.second.c_str());
  printf("\n");
  fflush(stdout);
  for (auto& o : opts) h->setOptionValue(o.first, o.second);
  if (h->readModel(file) == HighsStatus::kError) {
    printf("read error\n");
    delete h;
    return;
  }
  HighsStatus s = h->run();
  report(*h, "run", s);
  if (extras) {
    rays(*h);
    tableau(*h);
    ranging(*h);
  }
  // Warm start after a modification
  if (h->getLp().num_col_ > 1 && !h->getLp().isMip()) {
    const HighsLp& lp = h->getLp();
    double lo = lp.col_lower_[0], up = lp.col_upper_[0];
    h->changeColBounds(0, lo, up < kHighsInf ? 0.5 * (lo + up) : lo + 1);
    s = h->run();
    report(*h, "rerun after bound change", s);
    h->changeColCost(1, lp.col_cost_[1] + 1);
    s = h->run();
    report(*h, "rerun after cost change", s);
  }
  fflush(stdout);
  delete h;
}

static void modelEdits(const std::string& instances) {
  printf("\n######## model edits\n");
  Highs* h = fresh();
  h->readModel(instances + "/adlittle.mps");
  HighsStatus s = h->run();
  report(*h, "adlittle", s);
  const double cost[2] = {1.5, -2.0};
  const double lower[2] = {0, -1};
  const double upper[2] = {10, kHighsInf};
  s = h->addCols(2, cost, lower, upper, 0, nullptr, nullptr, nullptr);
  printf("addCols %d\n", int(s));
  const double rl[2] = {-kHighsInf, 1};
  const double ru[2] = {4, 3};
  const HighsInt start[2] = {0, 2};
  const HighsInt index[4] = {0, 5, 3, 1};
  const double value[4] = {1, -1, 2, 0.5};
  s = h->addRows(2, rl, ru, 4, start, index, value);
  printf("addRows %d\n", int(s));
  s = h->changeCoeff(0, 1, 3.25);
  printf("changeCoeff %d\n", int(s));
  const double lower3[3] = {0, -1, 0.5};
  const double upper3[3] = {10, kHighsInf, 0.5};
  s = h->changeColsBounds(1, 3, lower3, upper3);
  printf("changeColsBounds %d\n", int(s));
  s = h->changeRowBounds(2, -1, 7);
  printf("changeRowBounds %d\n", int(s));
  s = h->run();
  report(*h, "after edits", s);
  ranging(*h);
  s = h->deleteCols(2, 4);
  printf("deleteCols %d\n", int(s));
  HighsInt set[3] = {0, 3, 5};
  s = h->deleteRows(3, set);
  printf("deleteRows %d\n", int(s));
  std::vector<HighsInt> mask(h->getLp().num_col_, 0);
  mask[1] = 1;
  s = h->deleteCols(mask.data());
  printf("deleteCols mask %d", int(s));
  for (HighsInt m : mask) printf(" %d", int(m));
  printf("\n");
  s = h->run();
  report(*h, "after deletes", s);
  // Bad edits
  s = h->changeColBounds(-1, 0, 1);
  printf("bad changeColBounds %d\n", int(s));
  s = h->changeColCost(h->getLp().num_col_, 1);
  printf("bad changeColCost %d\n", int(s));
  s = h->changeColBounds(0, 2, 1);
  printf("inconsistent changeColBounds %d\n", int(s));
  s = h->run();
  report(*h, "after inconsistent bounds", s);
  s = h->changeColBounds(0, 1 + 1e-9, 1);
  printf("slightly inconsistent changeColBounds %d\n", int(s));
  s = h->run();
  report(*h, "after slightly inconsistent bounds", s);
  delete h;
}

static void solutionAndBasis(const std::string& instances) {
  printf("\n######## setSolution / setBasis\n");
  Highs* h = fresh();
  h->readModel(instances + "/afiro.mps");
  HighsStatus s = h->run();
  report(*h, "afiro", s);
  const HighsSolution solution = h->getSolution();
  const HighsBasis basis = h->getBasis();
  h->clearSolver();
  s = h->setSolution(solution);
  printf("setSolution %d\n", int(s));
  s = h->run();
  report(*h, "from solution", s);
  h->clearSolver();
  s = h->setBasis(basis, "driver");
  printf("setBasis %d\n", int(s));
  s = h->run();
  report(*h, "from basis", s);
  HighsBasis bad = basis;
  bad.col_status[0] = HighsBasisStatus::kBasic;
  bad.col_status[1] = HighsBasisStatus::kBasic;
  s = h->setBasis(bad);
  printf("setBasis bad %d\n", int(s));
  bad.row_status.pop_back();
  s = h->setBasis(bad);
  printf("setBasis wrong size %d\n", int(s));
  s = h->setBasis();
  printf("setBasis logical %d\n", int(s));
  s = h->run();
  report(*h, "from logical basis", s);
  HighsInt idx[3] = {0, 2, 4};
  double val[3] = {1, 2, 3};
  h->clearSolver();
  s = h->setSolution(3, idx, val);
  printf("setSolution sparse %d\n", int(s));
  s = h->run();
  report(*h, "from sparse solution", s);
  delete h;
}

static void presolvePostsolve(const std::string& instances) {
  printf("\n######## presolve / postsolve\n");
  for (const char* f : {"adlittle.mps", "afiro.mps", "egout.mps", "woodinfe.mps"}) {
    Highs* h = fresh();
    h->readModel(instances + "/" + f);
    HighsStatus s = h->presolve();
    printf("%s presolve %d status %s\n", f, int(s),
           h->presolveStatusToString(h->getModelPresolveStatus()).c_str());
    const HighsLp& p = h->getPresolvedLp();
    printf("presolved %d %d %d\n", int(p.num_col_), int(p.num_row_),
           int(p.a_matrix_.numNz()));
    if (h->getModelPresolveStatus() == HighsPresolveStatus::kReduced &&
        !h->getLp().isMip()) {
      Highs g;
      g.setOptionValue("output_flag", false);
      g.passModel(p);
      g.run();
      s = h->postsolve(g.getSolution(), g.getBasis());
      report(*h, "postsolve with basis", s);
      s = h->postsolve(g.getSolution());
      report(*h, "postsolve without basis", s);
    } else if (h->getModelPresolveStatus() == HighsPresolveStatus::kReduced) {
      Highs g;
      g.setOptionValue("output_flag", false);
      g.passModel(p);
      g.run();
      s = h->postsolve(g.getSolution());
      report(*h, "postsolve MIP", s);
    }
    delete h;
  }
}

static void special() {
  printf("\n######## special LPs\n");
  // Maximization: ranging and tableau with flipped signs
  for (const char* f : {"adlittle.mps", "scrs8.mps"}) {
    Highs* h = fresh();
    h->readModel(std::string(g_instances) + "/" + f);
    std::vector<double> cost = h->getLp().col_cost_;
    for (double& c : cost) c = -c;
    h->changeColsCost(0, h->getLp().num_col_ - 1, cost.data());
    h->changeObjectiveSense(ObjSense::kMaximize);
    HighsStatus s = h->run();
    report(*h, "maximize", s);
    rays(*h);
    tableau(*h);
    ranging(*h);
    delete h;
  }
  // Unconstrained LP, with bound cases
  {
    Highs* h = fresh();
    HighsLp lp;
    lp.num_col_ = 6;
    lp.num_row_ = 0;
    lp.col_cost_ = {0.1, -0.7, 0.3, 1e-9, 0, -2.5};
    lp.col_lower_ = {0.3, -kHighsInf, -1, -kHighsInf, 2, 1};
    lp.col_upper_ = {0.7, 3.3, kHighsInf, kHighsInf, 2, kHighsInf};
    lp.offset_ = 0.1;
    lp.a_matrix_.start_ = {0, 0, 0, 0, 0, 0, 0};
    h->passModel(lp);
    HighsStatus s = h->run();
    report(*h, "unconstrained", s);
    vec("col_value", h->getSolution().col_value);
    vec("col_dual", h->getSolution().col_dual);
    h->changeColBounds(5, 1, 4);
    s = h->run();
    report(*h, "unconstrained bounded", s);
    h->changeObjectiveSense(ObjSense::kMaximize);
    s = h->run();
    report(*h, "unconstrained max", s);
    delete h;
  }
  // LP with zero matrix and rows
  {
    Highs* h = fresh();
    HighsLp lp;
    lp.num_col_ = 2;
    lp.num_row_ = 2;
    lp.col_cost_ = {1, -1};
    lp.col_lower_ = {0, 0};
    lp.col_upper_ = {1, 1};
    lp.row_lower_ = {-1, 0.5};
    lp.row_upper_ = {1, 2};
    lp.a_matrix_.start_ = {0, 0, 0};
    h->passModel(lp);
    HighsStatus s = h->run();
    report(*h, "zero matrix", s);
    delete h;
  }
  // Empty model
  {
    Highs* h = fresh();
    HighsStatus s = h->run();
    report(*h, "empty", s);
    delete h;
  }
  // Infinite costs
  {
    Highs* h = fresh();
    HighsLp lp;
    lp.num_col_ = 2;
    lp.num_row_ = 1;
    lp.col_cost_ = {kHighsInf, 1};
    lp.col_lower_ = {0, 0};
    lp.col_upper_ = {1, 1};
    lp.row_lower_ = {1};
    lp.row_upper_ = {2};
    lp.a_matrix_.start_ = {0, 1, 2};
    lp.a_matrix_.index_ = {0, 0};
    lp.a_matrix_.value_ = {1, 1};
    h->passModel(lp);
    HighsStatus s = h->run();
    report(*h, "infinite cost", s);
    delete h;
  }
  // Large matrix value
  {
    Highs* h = fresh();
    HighsLp lp;
    lp.num_col_ = 1;
    lp.num_row_ = 1;
    lp.col_cost_ = {1};
    lp.col_lower_ = {0};
    lp.col_upper_ = {1};
    lp.row_lower_ = {1};
    lp.row_upper_ = {2};
    lp.a_matrix_.start_ = {0, 1};
    lp.a_matrix_.index_ = {0};
    lp.a_matrix_.value_ = {1e10};
    h->setOptionValue("large_matrix_value", 1e12);
    h->passModel(lp);
    h->setOptionValue("large_matrix_value", 1e9);
    HighsStatus s = h->run();
    report(*h, "large value", s);
    delete h;
  }
}

// getCols / getRows by interval, set and mask, with the matrix column-wise
// and row-wise (getSubVectors and getSubVectorsTranspose), and without
// the matrix or data
static void getColsRows(const std::string& instances) {
  for (const char* f : {"adlittle.mps", "afiro.mps"}) {
    Highs h;
    h.setOptionValue("output_flag", false);
    h.readModel(instances + "/" + f);
    for (int rowwise = 0; rowwise < 2; rowwise++) {
      HighsLp lp = h.getLp();
      if (rowwise) {
        lp.a_matrix_.ensureRowwise();
        h.passModel(lp);
      }
      printf("== getColsRows %s rowwise %d (%d)\n", f, rowwise,
             int(h.getLp().a_matrix_.isRowwise()));
      for (int what = 0; what < 2; what++) {
        const HighsInt dim = what ? lp.num_row_ : lp.num_col_;
        std::vector<HighsInt> set = {1, 2, 3, 7, dim - 2, dim - 1};
        std::vector<HighsInt> mask(dim, 0);
        for (HighsInt i = 0; i < dim; i += 3) mask[i] = 1;
        for (int how = 0; how < 4; how++) {
          HighsInt num = 0, nnz = 0;
          std::vector<double> c(dim), lo(dim), up(dim);
          std::vector<HighsInt> start(dim + 1), index(lp.a_matrix_.numNz());
          std::vector<double> value(lp.a_matrix_.numNz());
          HighsStatus s;
          if (what == 0) {
            if (how == 0)
              s = h.getCols(2, dim - 3, num, c.data(), lo.data(), up.data(),
                            nnz, start.data(), index.data(), value.data());
            else if (how == 1)
              s = h.getCols(HighsInt(set.size()), set.data(), num, c.data(),
                            lo.data(), up.data(), nnz, start.data(),
                            index.data(), value.data());
            else if (how == 2)
              s = h.getCols(mask.data(), num, c.data(), lo.data(), up.data(),
                            nnz, start.data(), index.data(), value.data());
            else
              s = h.getCols(0, dim - 1, num, nullptr, nullptr, up.data(), nnz,
                            start.data(), nullptr, nullptr);
          } else {
            if (how == 0)
              s = h.getRows(2, dim - 3, num, lo.data(), up.data(), nnz,
                            start.data(), index.data(), value.data());
            else if (how == 1)
              s = h.getRows(HighsInt(set.size()), set.data(), num, lo.data(),
                            up.data(), nnz, start.data(), index.data(),
                            value.data());
            else if (how == 2)
              s = h.getRows(mask.data(), num, lo.data(), up.data(), nnz,
                            start.data(), index.data(), value.data());
            else
              s = h.getRows(0, dim - 1, num, nullptr, lo.data(), nnz, nullptr,
                            nullptr, nullptr);
          }
          printf("%s how %d status %d num %d nnz %d\n", what ? "rows" : "cols",
                 how, int(s), int(num), int(nnz));
          c.resize(num);
          lo.resize(num);
          up.resize(num);
          start.resize(num);
          index.resize(nnz);
          value.resize(nnz);
          sum("cost", c);
          sum("lower", lo);
          sum("upper", up);
          ivec("start", start);
          ivec("index", index);
          sum("value", value);
        }
      }
    }
  }
}

// Completing a MIP solution from a (partial) user assignment, crossover
// from a user solution, rays of infeasible and unbounded LPs and a QP,
// presolve and postsolve errors
static void drivers(const std::string& instances) {
  printf("\n######## drivers\n");
  for (const char* f : {"flugpl.mps", "egout.mps", "bell5.mps"}) {
    for (int variant = 0; variant < 4; variant++) {
      Highs h;
      h.setOptionValue("output_flag", false);
      if (h.readModel(instances + "/" + f) == HighsStatus::kError) break;
      h.setOptionValue("output_flag", true);
      h.setOptionValue("timeless_log", true);
      h.setOptionValue("mip_max_start_nodes", 50);
      Highs g;
      g.setOptionValue("output_flag", false);
      g.passModel(h.getLp());
      g.run();
      HighsSolution s = g.getSolution();
      const HighsLp& lp = h.getLp();
      printf("-- %s variant %d\n", f, variant);
      for (HighsInt j = 0; j < lp.num_col_; j++) {
        const bool discrete = lp.integrality_.size() &&
                              lp.integrality_[j] != HighsVarType::kContinuous;
        if (variant == 1 && discrete && j % 2) s.col_value[j] = kHighsUndefined;
        if (variant == 2 && discrete) s.col_value[j] += 0.3;
        if (variant == 3 && !discrete) s.col_value[j] = kHighsUndefined;
      }
      HighsStatus st = h.setSolution(s);
      printf("setSolution %d\n", int(st));
      st = h.run();
      report(h, "run from user solution", st);
    }
  }
  // Crossover from IPX's interior solution, and for a MIP and QP
  for (const char* f : {"afiro.mps", "adlittle.mps", "flugpl.mps", "qjh.mps"}) {
    Highs h;
    h.setOptionValue("output_flag", false);
    h.readModel(instances + "/" + f);
    h.setOptionValue("solver", "ipm");
    h.setOptionValue("run_crossover", "off");
    h.setOptionValue("solve_relaxation", true);
    h.run();
    HighsSolution s = h.getSolution();
    h.setOptionValue("output_flag", true);
    h.setOptionValue("timeless_log", true);
    HighsStatus st = h.crossover(s);
    report(h, (std::string("crossover ") + f).c_str(), st);
  }
  // Rays: infeasible and unbounded LPs, an infeasible MIP's relaxation, a QP
  {
    const double inf = kHighsInf;
    for (int which = 0; which < 4; which++) {
      Highs h;
      h.setOptionValue("timeless_log", true);
      HighsLp lp;
      lp.num_col_ = 2;
      lp.num_row_ = 2;
      lp.col_cost_ = {which == 1 ? -1.0 : 1.0, 1};
      lp.col_lower_ = {0, 0};
      lp.col_upper_ = {inf, inf};
      lp.row_lower_ = {which == 1 ? -inf : 3.0, -inf};
      lp.row_upper_ = {inf, which == 1 ? 1.0 : 2.0};
      lp.a_matrix_.start_ = {0, 2, 4};
      lp.a_matrix_.index_ = {0, 1, 0, 1};
      lp.a_matrix_.value_ = {1, which == 1 ? -1.0 : 1.0, 1, 1};
      if (which == 2) lp.integrality_ = {HighsVarType::kInteger,
                                         HighsVarType::kInteger};
      h.passModel(lp);
      if (which == 3) {
        HighsHessian q;
        q.dim_ = 2;
        q.start_ = {0, 1, 2};
        q.index_ = {0, 1};
        q.value_ = {1, 1};
        h.passHessian(q);
      }
      for (int pass = 0; pass < 2; pass++) {
        HighsStatus st = h.run();
        report(h, "ray model", st);
        bool has = false;
        std::vector<double> ray(2);
        st = h.getDualRay(has, nullptr);
        printf("dual ray (no values) %d %d\n", int(st), int(has));
        st = h.getDualRay(has, ray.data());
        printf("dual ray %d %d %.17g %.17g\n", int(st), int(has), ray[0],
               ray[1]);
        report(h, "after dual ray", st);
        st = h.getPrimalRay(has, ray.data());
        printf("primal ray %d %d %.17g %.17g\n", int(st), int(has), ray[0],
               ray[1]);
        report(h, "after primal ray", st);
        // The second time from a cleared solver
        h.clearSolver();
      }
    }
  }
  // Presolve of a QP and of a model with semi-variables; postsolve errors
  for (const char* f : {"qjh.mps", "semi-continuous.mps", "afiro.mps"}) {
    Highs h;
    h.setOptionValue("timeless_log", true);
    h.setOptionValue("output_flag", false);
    if (h.readModel(instances + "/" + f) == HighsStatus::kError) continue;
    h.setOptionValue("output_flag", true);
    HighsStatus st = h.presolve();
    printf("%s presolve %d\n", f, int(st));
    // (Postsolve after a failed presolve may throw in the C++)
    if (st == HighsStatus::kError) continue;
    HighsSolution s;
    s.col_value.assign(3, 0);
    st = h.postsolve(s);
    printf("postsolve wrong size %d\n", int(st));
    s.col_value.assign(h.getPresolvedLp().num_col_, 0);
    s.row_dual.assign(1, 0);
    st = h.postsolve(s);
    printf("postsolve wrong dual size %d\n", int(st));
    HighsBasis b;
    b.valid = true;
    st = h.postsolve(s, b);
    printf("postsolve bad basis %d\n", int(st));
  }
}

// Passing models (arrays, Hessians, bad formats and integrality), reading
// and writing models and bases (unsupported and missing files, repeated
// names), set errors and a model change after a solve
static void modelPassing(const std::string& instances) {
  printf("\n######## model passing\n");
  Highs h;
  h.setOptionValue("timeless_log", true);
  const double inf = kHighsInf;
  const HighsInt start[3] = {0, 2, 4};
  const HighsInt index[4] = {0, 1, 0, 1};
  const double value[4] = {1, 2, 3, 4};
  const double cost[2] = {1, -1}, lower[2] = {0, 0}, upper[2] = {inf, 4};
  const double rlower[2] = {-inf, 1}, rupper[2] = {6, inf};
  HighsInt integrality[2] = {1, 0};
  for (int variant = 0; variant < 8; variant++) {
    HighsStatus s;
    const HighsInt q_start[3] = {0, 2, 3};
    const HighsInt q_index[3] = {0, 1, 1};
    const HighsInt q_square_start[3] = {0, 2, 4};
    const HighsInt q_square_index[4] = {0, 1, 0, 1};
    const double q_value[4] = {2, 1, 1.5, 2};
    switch (variant) {
      case 0:
        s = h.passModel(2, 2, 4, 1, 1, 0.5, cost, lower, upper, rlower,
                        rupper, start, index, value, integrality);
        break;
      case 1:
        integrality[1] = 7;
        s = h.passModel(2, 2, 4, 1, 1, 0.5, cost, lower, upper, rlower,
                        rupper, start, index, value, integrality);
        integrality[1] = 0;
        break;
      case 2:
        s = h.passModel(2, 2, 4, 5, 1, 0.5, cost, lower, upper, rlower,
                        rupper, start, index, value, integrality);
        break;
      case 3:
        s = h.passModel(2, 2, 4, 3, 2, 1, 1, 0, cost, lower, upper, rlower,
                        rupper, start, index, value, q_start, q_index,
                        q_value, nullptr);
        break;
      case 4:
        // Square, slightly asymmetric
        s = h.passModel(2, 2, 4, 4, 1, 2, 1, 0, cost, lower, upper, rlower,
                        rupper, start, index, value, q_square_start,
                        q_square_index, q_value, nullptr);
        break;
      case 5:
        s = h.passHessian(3, 3, 1, q_start, q_index, q_value);
        printf("passHessian wrong dim %d\n", int(s));
        s = h.passHessian(2, 3, 9, q_start, q_index, q_value);
        printf("passHessian bad format %d\n", int(s));
        s = h.passHessian(2, 0, 1, q_start, q_index, q_value);
        break;
      case 6:
        // No rows: the matrix is ignored
        s = h.passModel(2, 0, 0, 1, 1, 0.5, cost, lower, upper, nullptr,
                        nullptr, start, index, value, nullptr);
        break;
      default: {
        HighsLp lp;
        lp.num_col_ = 2;
        lp.num_row_ = 2;
        lp.col_cost_ = {1, 1};
        lp.col_lower_ = {0, 0};
        lp.col_upper_ = {1, 1};
        lp.row_lower_ = {0, 0};
        lp.row_upper_ = {1, 1};
        lp.a_matrix_.start_ = {0, 1};
        lp.a_matrix_.index_ = {0};
        lp.a_matrix_.value_ = {1};
        s = h.passModel(lp);
      }
    }
    printf("pass variant %d: %d\n", variant, int(s));
    if (s != HighsStatus::kError) report(h, "after pass", h.run());
  }
  // Sets with errors
  h.readModel(instances + "/afiro.mps");
  const HighsInt dup[3] = {1, 1, 2}, unordered[3] = {3, 1, 2},
                 out[3] = {1, 2, 99};
  const double c3[3] = {1, 2, 3};
  printf("dup %d\n", int(h.changeColsCost(3, dup, c3)));
  printf("unordered %d\n", int(h.changeColsBounds(3, unordered, c3, c3)));
  printf("out %d\n", int(h.changeRowsBounds(3, out, c3, c3)));
  printf("size %d\n", int(h.deleteCols(-1, dup)));
  // A model change after a solve: the solution and basis are resized
  report(h, "afiro", h.run());
  h.addCol(1, 0, 1, 0, nullptr, nullptr);
  h.addRow(0, 1, 0, nullptr, nullptr);
  report(h, "after adding a column and row", h.run());
  // Reading and writing
  for (const char* f : {"/nosuchfile.mps", "/afiro.txt", "/afiro.mps.gz",
                        "/nosuch.lp"}) {
    HighsStatus s = h.readModel(instances + f);
    printf("read %s %d\n", f, int(s));
  }
  h.readModel(instances + "/adlittle.mps");
  h.run();
  for (const char* f : {"m.mps", "m.lp", "m.xyz", ""}) {
    HighsStatus s = h.writeModel(f);
    printf("write model '%s' %d\n", f, int(s));
  }
  h.passColName(0, "same");
  h.passColName(1, "same");
  printf("passColName bad %d %d\n", int(h.passColName(-1, "x")),
         int(h.passColName(0, "")));
  printf("write repeated %d\n", int(h.writeModel("r.mps")));
  printf("write basis %d\n", int(h.writeBasis("b.bas")));
  printf("read basis %d\n", int(h.readBasis("b.bas")));
  printf("read basis missing %d\n", int(h.readBasis("nosuch.bas")));
  printf("read basis wrong %d\n", int(h.readBasis("m.mps")));
  h.clearSolver();
  printf("write basis invalid %d\n", int(h.writeBasis("b2.bas")));
  printf("write basis stdout %d\n", int(h.writeBasis("")));
  printf("write basis unwritable %d\n", int(h.writeBasis("/nosuchdir/b.bas")));
}

// getStandardFormLp of LPs with boxed, free, fixed and one-sided rows
// and columns, before and after a model change
static void standardForm(const std::string& instances) {
  printf("\n######## standard form\n");
  for (const char* f : {"afiro.mps", "adlittle.mps", "box1.mps",
                        "shell.mps", "25fv47.mps", "egout.mps"}) {
    Highs h;
    h.setOptionValue("output_flag", false);
    if (h.readModel(instances + "/" + f) == HighsStatus::kError) continue;
    h.setOptionValue("output_flag", true);
    for (int pass = 0; pass < 2; pass++) {
      HighsInt num_col, num_row, num_nz;
      double offset;
      HighsStatus s =
          h.getStandardFormLp(num_col, num_row, num_nz, offset);
      printf("%s pass %d: status %d %d %d %d offset %.17g\n", f, pass, int(s),
             int(num_col), int(num_row), int(num_nz), offset);
      std::vector<double> cost(num_col), rhs(num_row), value(num_nz);
      std::vector<HighsInt> start(num_col + 1), index(num_nz);
      s = h.getStandardFormLp(num_col, num_row, num_nz, offset, cost.data(),
                              rhs.data(), start.data(), index.data(),
                              value.data());
      sum("cost", cost);
      sum("rhs", rhs);
      sum("value", value);
      ivec("start", start);
      long long isum = 0;
      for (size_t k = 0; k < index.size(); k++) isum += index[k] * (k % 13 + 1);
      printf("index sum %lld\n", isum);
      // The incumbent matrix after the round trip through row-wise
      const HighsLp& lp = h.getLp();
      sum("a_value", lp.a_matrix_.value_);
      ivec("a_start", lp.a_matrix_.start_);
      h.changeColBounds(0, -kHighsInf, kHighsInf);
      h.changeObjectiveSense(ObjSense::kMaximize);
      h.changeObjectiveOffset(2.5);
    }
  }
}

// Timing-dependent messages (dev level timings, thread numbers) are not
// printed by the callbacks
static const char* untimed(const char* message) {
  const std::string m(message);
  for (const char* t : {"Strange", "Thread", "MIP  ", "time", "Time"})
    if (m.find(t) != std::string::npos) return nullptr;
  return message;
}

// Log and solver callbacks: the deprecated log callback, the C++ and C
// callbacks, messages longer than the callback buffer, logging to a file
static void logCallback(HighsLogType type, const char* message, void* data) {
  if (!untimed(message)) return;
  printf("logcb[%s] %d: %s", static_cast<const char*>(data), int(type),
         untimed(message));
}

static void cCallback(int type, const char* message,
                      const HighsCallbackDataOut* out, HighsCallbackDataIn* in,
                      void* data) {
  (void)in;
  if (!untimed(message)) return;
  printf("ccb[%s] %d log %d: %s", static_cast<const char*>(data), type,
         out->log_type, untimed(message));
}

static void callbacks(const std::string& instances) {
  printf("\n######## callbacks\n");
  const std::string long_name(1500, 'x');
  for (int variant = 0; variant < 5; variant++) {
    printf("-- variant %d\n", variant);
    fflush(stdout);
    Highs h;
    h.setOptionValue("timeless_log", true);
    if (variant == 1) h.setOptionValue("log_to_console", false);
    if (variant == 2) h.setOptionValue("log_dev_level", 3);
    if (variant == 3) h.setOptionValue("log_file", "callbacks.log");
    if (variant == 4) h.setOptionValue("output_flag", false);
    static char tag[] = "user";
    h.setLogCallback(logCallback, tag);
    if (variant != 1) {
      h.setCallback(
          [](int type, const std::string& message,
             const HighsCallbackOutput* out, HighsCallbackInput* in,
             void* data) {
            (void)data;
            if (type == kCallbackLogging) {
              if (!untimed(message.c_str())) return;
              printf("cb %d log %d: %s", type, int(out->log_type),
                     untimed(message.c_str()));
            } else if (type == kCallbackSimplexInterrupt) {
              printf("cb simplex %d\n", int(out->simplex_iteration_count));
              if (out->simplex_iteration_count > 30) in->user_interrupt = true;
            } else if (type == kCallbackMipImprovingSolution) {
              printf("cb mip improving %.17g\n", out->objective_function_value);
            } else {
              printf("cb %d\n", type);
            }
          },
          nullptr);
      h.startCallback(kCallbackLogging);
      if (variant == 2) h.startCallback(kCallbackSimplexInterrupt);
      h.startCallback(kCallbackMipImprovingSolution);
    } else {
      static char ctag[] = "c";
      h.setCallback(cCallback, ctag);
      h.startCallback(kCallbackLogging);
    }
    // An unknown option with a long name: a message beyond the 1024-byte
    // callback buffer
    h.setOptionValue(long_name, true);
    h.readModel(instances + "/adlittle.mps");
    report(h, "adlittle", h.run());
    h.readModel(instances + "/flugpl.mps");
    report(h, "flugpl", h.run());
    h.stopCallback(kCallbackLogging);
    h.readModel(instances + "/afiro.mps");
    report(h, "afiro, logging callback stopped", h.run());
    fflush(stdout);
  }
  FILE* f = fopen("callbacks.log", "r");
  if (f) {
    printf("-- callbacks.log\n");
    char line[4096];
    while (fgets(line, sizeof line, f)) {
      if (std::string(line).find("time") == std::string::npos) printf("%s", line);
    }
    fclose(f);
  }
}


static void iisReport(Highs& h, const char* what, HighsStatus s,
                      const HighsIis& iis) {
  printf("%s %d model status %s iis valid %d status %d strategy %d\n", what,
         int(s), h.modelStatusToString(h.getModelStatus()).c_str(),
         int(iis.valid_), int(iis.status_), int(iis.strategy_));
  ivec("col_index", iis.col_index_);
  ivec("row_index", iis.row_index_);
  ivec("col_bound", iis.col_bound_);
  ivec("row_bound", iis.row_bound_);
  ivec("col_status", iis.col_status_);
  ivec("row_status", iis.row_status_);
  printf("lps %d iterations %d min %d max %d\n", int(iis.info_.num_lp_solved),
         int(iis.info_.sum_simplex_iteration_counts),
         int(iis.info_.min_simplex_iteration_count),
         int(iis.info_.max_simplex_iteration_count));
  const HighsLp& lp = h.getIisLp();
  printf("iis lp %s %d x %d nnz %d names %d %d\n", lp.model_name_.c_str(),
         int(lp.num_row_), int(lp.num_col_), int(lp.a_matrix_.numNz()),
         int(lp.col_names_.size()), int(lp.row_names_.size()));
  fflush(stdout);
}

static int g_log_calls[16];
static void countCallback(int type, const std::string&,
                          const HighsCallbackOutput*, HighsCallbackInput*,
                          void*) {
  if (type >= 0 && type < 16) g_log_calls[type]++;
}

static void iisCases(const std::string& instances) {
  const char* models[] = {"galenet", "woodinfe", "forest6", "klein1",
                          "ex72a",   "avgas",    "infeasible-mip0",
                          "infeasible-mip1", "refinery"};
  const int strategies[] = {0, 1, 2, 4, 6, 10, 12, 18, 22, 30};
  for (const char* m : models) {
    for (int strategy : strategies) {
      for (int solve_first = 0; solve_first < 2; solve_first++) {
        printf("\n######## iis %s strategy %d solve first %d\n", m, strategy,
               solve_first);
        Highs* h = fresh();
        h->setOptionValue("output_flag", strategy == 6 && solve_first);
        h->readModel(instances + "/" + m + ".mps");
        h->setOptionValue("iis_strategy", strategy);
        if (solve_first) report(*h, "run", h->run());
        HighsIis iis;
        HighsStatus s = h->getIis(iis);
        iisReport(*h, "getIis", s, iis);
        // Again: the IIS is kept
        s = h->getIis(iis);
        iisReport(*h, "getIis again", s, iis);
        std::string file = std::string("iis_") + m + "_" +
                           std::to_string(strategy) + "_" +
                           std::to_string(solve_first) + ".lp";
        printf("writeIisModel %d\n", int(h->writeIisModel(file)));
        delete h;
      }
    }
  }
  // Time limit, callbacks, logging to the console
  for (const char* m : {"forest6", "avgas", "woodinfe"}) {
    printf("\n######## iis %s time limit 0\n", m);
    Highs* h = fresh();
    h->setOptionValue("output_flag", false);
    h->readModel(instances + "/" + m + ".mps");
    h->setOptionValue("iis_time_limit", 0.0);
    h->setOptionValue("iis_strategy", 6);
    HighsIis iis;
    HighsStatus s = h->getIis(iis);
    iisReport(*h, "getIis", s, iis);
    delete h;
  }
  {
    printf("\n######## iis callback\n");
    Highs* h = fresh();
    h->setOptionValue("log_to_console", false);
    h->readModel(instances + "/galenet.mps");
    h->setOptionValue("iis_strategy", 6);
    h->setCallback(countCallback, nullptr);
    h->startCallback(kCallbackLogging);
    h->startCallback(kCallbackMipImprovingSolution);
    HighsIis iis;
    HighsStatus s = h->getIis(iis);
    iisReport(*h, "getIis", s, iis);
    for (int i = 0; i < 10; i++) printf(" %d", g_log_calls[i]);
    printf("\n");
    delete h;
  }
  // Feasibility relaxation
  for (const char* m : {"galenet", "woodinfe", "avgas", "infeasible-mip0",
                        "klein1"}) {
    for (int k = 0; k < 4; k++) {
      printf("\n######## feasibilityRelaxation %s %d\n", m, k);
      Highs* h = fresh();
      h->setOptionValue("output_flag", k == 1);
      h->readModel(instances + "/" + m + ".mps");
      if (k == 3) h->setOptionValue("iis_strategy", 16);
      const HighsLp& lp = h->getLp();
      std::vector<double> lo(lp.num_col_), up(lp.num_col_), rhs(lp.num_row_);
      for (HighsInt i = 0; i < lp.num_col_; i++) {
        lo[i] = i % 3 == 0 ? -1 : 1 + i % 2;
        up[i] = i % 4 == 0 ? -1 : 2;
      }
      for (HighsInt i = 0; i < lp.num_row_; i++) rhs[i] = i % 5 == 0 ? -1 : i % 3;
      HighsStatus s =
          k == 0   ? h->feasibilityRelaxation(1, 1, 1)
          : k == 1 ? h->feasibilityRelaxation(-1, 1, 1)
          : k == 2 ? h->feasibilityRelaxation(1, 1, 0, lo.data(), up.data(),
                                              rhs.data())
                   : h->feasibilityRelaxation(1, -1, 1, nullptr, nullptr,
                                              rhs.data());
      report(*h, "feasibilityRelaxation", s);
      printf("objective %.17g\n", h->getInfo().objective_function_value);
      HighsIis iis;
      s = h->getIis(iis);
      iisReport(*h, "getIis after", s, iis);
      delete h;
    }
  }
}

// QPs through the API (passModel with a Hessian, passHessian, hot start)
// and the ill-conditioning analysis of an optimal basis
static void qpAndIllConditioning(const std::string& instances) {
  printf("\n######## QP through the API\n");
  {
    Highs* h = fresh();
    // min x0^2 + x1^2 - x0 x1 + x0 - 2 x1, x0 + x1 >= 1, 0 <= x <= 3
    const double cost[] = {1, -2}, lower[] = {0, 0}, upper[] = {3, 3};
    const double row_lower[] = {1}, row_upper[] = {kHighsInf};
    const HighsInt a_start[] = {0, 1}, a_index[] = {0, 0};
    const double a_value[] = {1, 1};
    const HighsInt q_start[] = {0, 2}, q_index[] = {0, 1, 1};
    const double q_value[] = {2, -1, 2};
    HighsStatus s = h->passModel(2, 1, 2, 3, 1, 1, 1, 0.5, cost, lower,
                                 upper, row_lower, row_upper, a_start,
                                 a_index, a_value, q_start, q_index, q_value,
                                 nullptr);
    printf("passModel %d\n", int(s));
    s = h->run();
    report(*h, "qp", s);
    h->changeObjectiveSense(ObjSense::kMaximize);
    const double neg[] = {-2, 1, -2};
    s = h->passHessian(2, 3, 1, q_start, q_index, neg);
    printf("passHessian %d\n", int(s));
    s = h->run();
    report(*h, "qp max", s);
    delete h;
  }
  for (const char* f : {"qjh.mps", "primal1.mps", "qptestnw.lp"}) {
    Highs* h = fresh();
    printf("\n######## QP %s hot start\n", f);
    h->readModel(instances + "/" + f);
    HighsStatus s = h->run();
    report(*h, "qp", s);
    h->setOptionValue("qp_allow_hot_start", true);
    s = h->run();
    report(*h, "qp hot start", s);
    h->changeColBounds(0, -1, 1);
    s = h->run();
    report(*h, "qp hot start after bound change", s);
    delete h;
  }
  for (const char* f : {"afiro.mps", "adlittle.mps", "israel.mps"}) {
    Highs* h = fresh();
    printf("\n######## ill-conditioning %s\n", f);
    h->readModel(instances + "/" + f);
    HighsIllConditioning ic;
    HighsStatus s = h->getIllConditioning(ic, true);
    printf("before solve %d\n", int(s));
    h->run();
    for (int constraint = 0; constraint < 2; constraint++)
      for (int method = 0; method < 2; method++)
        for (double bound : {1e-4, 1.0, 1e3}) {
          if (method == 0 && bound > 1e-4) continue;
          s = h->getIllConditioning(ic, constraint, method, bound);
          printf("constraint %d method %d bound %g: %d, %d records\n",
                 constraint, method, bound, int(s), int(ic.record.size()));
          for (const auto& r : ic.record)
            printf("  %d %.17g\n", int(r.index), r.multiplier);
        }
    delete h;
  }
  fflush(stdout);
}

int main(int argc, char** argv) {
  const std::string instances = argc > 1 ? argv[1] : "check/instances";
  g_instances = instances;
  typedef std::vector<std::pair<std::string, std::string>> Opts;
  const std::vector<std::string> lps = {"afiro.mps", "adlittle.mps",
                                        "gas11.mps", "woodinfe.mps",
                                        "25fv47.mps", "shell.mps",
                                        "israel.mps", "scrs8.mps"};
  const std::vector<Opts> variants = {
      {},
      {{"presolve", "off"}},
      {{"solver", "ipm"}},
      {{"solver", "ipm"}, {"run_crossover", "off"}},
      {{"solver", "pdlp"}},
      {{"simplex_strategy", "4"}},
      {{"simplex_iteration_limit", "20"}},
      {{"allow_unbounded_or_infeasible", "true"}},
      {{"user_bound_scale", "-3"}, {"user_objective_scale", "4"}},
      {{"user_bound_scale", "700"}},
      {{"user_objective_scale", "-2"}, {"solver", "ipm"}},
  };
  for (const auto& f : lps)
    for (size_t v = 0; v < variants.size(); v++)
      solveFile(instances + "/" + f, variants[v], v < 2);
  for (const char* f : {"egout.mps", "flugpl.mps", "semi-continuous.mps",
                        "qjh.mps", "qptestnw.lp"}) {
    solveFile(instances + "/" + f, {}, false);
    solveFile(instances + "/" + f, {{"solve_relaxation", "true"}}, false);
    solveFile(instances + "/" + f,
              {{"user_bound_scale", "2"}, {"user_objective_scale", "-1"}},
              false);
  }
  modelEdits(instances);
  solutionAndBasis(instances);
  presolvePostsolve(instances);
  special();
  getColsRows(instances);
  callbacks(instances);
  standardForm(instances);
  drivers(instances);
  modelPassing(instances);
  iisCases(instances);
  qpAndIllConditioning(instances);
  return 0;
}
