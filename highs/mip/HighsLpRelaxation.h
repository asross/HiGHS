/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#ifndef HIGHS_LP_RELAXATION_H_
#define HIGHS_LP_RELAXATION_H_

#include <cstdint>
#include <memory>

#include "Highs.h"
#include "mip/HighsConflictPool.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsRsSpan.h"

class HighsDomain;
struct HighsCutSet;
class HighsPseudocost;
class HighsMipWorker;

#ifdef HIGHS_RUST
#include "lp_data/HighsLpHandle.h"

namespace highs_rs {
struct LpRelax;
struct CutPool;
struct LpHandle;
// rust/src/mip/lp_relaxation.rs: LpShared, read in place
struct LpShared {
  const void* rows;
  HighsInt num_rows;
  std::pair<HighsInt, double>* frac;
  HighsInt num_frac;
  const HighsInt* proof_inds;
  const double* proof_vals;
  HighsInt proof_len;
  double proof_rhs;
  bool has_proof;
  bool adjust_sym;
  int status;
  double objective;
  int64_t numlpiters;
  double avg_solve_iters;
};

extern "C" {
LpHandle* highs_rs_lprelax_lp(LpRelax* p);
}
}  // namespace highs_rs

// Highs::setOptionValue of an LP solver (asserted to succeed)
void rsLpSetOption(highs_rs::LpHandle* lp, const std::string& name,
                   bool value);
void rsLpSetOption(highs_rs::LpHandle* lp, const std::string& name,
                   HighsInt value);
void rsLpSetOption(highs_rs::LpHandle* lp, const std::string& name,
                   double value);
void rsLpSetOption(highs_rs::LpHandle* lp, const std::string& name,
                   const std::string& value);
// Highs::passModel of a C++ LP
HighsStatus rsLpPassModel(highs_rs::LpHandle* lp, const HighsLp& model);
// Highs::optimizeLp (a cancelled task throws HighsTask::Interrupt)
HighsStatus rsLpOptimize(highs_rs::LpHandle* lp);
// Highs::getSolution, a copy
HighsSolution rsLpSolution(highs_rs::LpHandle* lp);

/// The fractional integers of the Rust LP relaxation (valid until the next
/// solve); the heuristics sort them in place
struct HighsFracInts {
  std::pair<HighsInt, double>* p;
  size_t n;
  std::pair<HighsInt, double>& operator[](size_t i) const {
    assert(i < n);
    return p[i];
  }
  size_t size() const { return n; }
  bool empty() const { return n == 0; }
  std::pair<HighsInt, double>* begin() const { return p; }
  std::pair<HighsInt, double>* end() const { return p + n; }
  std::pair<HighsInt, double>* data() const { return p; }
};

class HighsLpRelaxation {
 public:
  enum class Status {
    kNotSet,
    kOptimal,
    kInfeasible,
    kUnscaledDualFeasible,
    kUnscaledPrimalFeasible,
    kUnscaledInfeasible,
    kUnbounded,
    kError,
  };

 private:
  // a row of the Rust LP relaxation (same layout)
  struct LpRow {
    enum Origin {
      kModel,
      kCutPool,
    };

    Origin origin;
    HighsInt index;
    HighsInt age;
    HighsInt cutpoolindex;

    void get(const HighsMipSolver& mipsolver, HighsInt& len,
             const HighsInt*& inds, const double*& vals) const;

    HighsInt getRowLen(const HighsMipSolver& mipsolver) const;

    bool isIntegral(const HighsMipSolver& mipsolver) const;

    double getMaxAbsVal(const HighsMipSolver& mipsolver) const;
  };

  const HighsMipSolver& mipsolver;
  highs_rs::LpRelax* rs_;
  highs_rs::LpShared* sh_;
  // The LP solver (rust/src/lp_data/lp_handle.rs), owned by rs_
  highs_rs::LpHandle* lp_;

  std::vector<double> colLbBuffer;
  std::vector<double> colUbBuffer;
  std::shared_ptr<const HighsBasis> basischeckpoint;
  bool currentbasisstored;
  bool solved_first_lp;
  bool raceIpx = false;
  HighsMipWorker* worker_;
  // for the Rust callbacks: the cut pools, and the arguments of
  // computeBasicDegenerateDuals' conflict analysis
  std::vector<highs_rs::CutPool*> cutpoolPtrs_;
  void* degenArgs_ = nullptr;
  friend struct HighsLpRelaxationAccess;

  const LpRow& lprow(HighsInt row) const {
    assert(row >= 0 && row < sh_->num_rows);
    return static_cast<const LpRow*>(sh_->rows)[row];
  }

  // run(): the solve (solver choice, IPM, race or simplex)
  HighsStatus runSolve(bool& use_simplex, int64_t& extraIterations);

  // run(): the IPM basis after the simplex hit its iteration limit
  void ipmBasisAfterIterationLimit();

 public:
  HighsLpRelaxation(const HighsMipSolver& mip);

  HighsLpRelaxation(const HighsLpRelaxation& other);

  HighsLpRelaxation& operator=(const HighsLpRelaxation&) = delete;

  ~HighsLpRelaxation();

  void setProfiling(HighsProfiling* profiling);

  void getCutPool(HighsInt& num_col, HighsInt& num_cut,
                  std::vector<double>& cut_lower,
                  std::vector<double>& cut_upper,
                  HighsSparseMatrix& cut_matrix) const;

  class Playground {
    friend class HighsLpRelaxation;
    HighsLpRelaxation* lp;
    bool iterateStored;

    Playground(HighsLpRelaxation* lp) : lp(lp), iterateStored(false) {}

   public:
    Playground(Playground&& other)
        : lp(other.lp), iterateStored(other.iterateStored) {
      other.iterateStored = false;
    }

    Playground& operator=(Playground&& other) {
      std::swap(lp, other.lp);
      std::swap(iterateStored, other.iterateStored);
      return *this;
    }

    HighsLpRelaxation::Status solveLp(HighsDomain& localdom) {
      if (iterateStored) {
        lp->flushDomain(localdom);
        lp->getIterate();
      } else {
        assert(lp->getSolverInfo().valid);
        lp->putIterate();
        lp->flushDomain(localdom);
        iterateStored = true;
      }

      return lp->run(false);
    }

    Playground(const Playground& other) = delete;
    Playground& operator=(const Playground& other) = delete;

    ~Playground() {
      if (iterateStored) {
        lp->getIterate();
        lp->run();
        // If desired, here is the place to clear the stored iterate
      }
    }
  };

  Playground playground() { return Playground(this); }

  void loadModel();

  void getRow(HighsInt row, HighsInt& len, const HighsInt*& inds,
              const double*& vals) const {
    if (row < mipsolver.numRow())
      assert(lprow(row).origin == LpRow::Origin::kModel);
    else
      assert(lprow(row).origin == LpRow::Origin::kCutPool);
    lprow(row).get(mipsolver, len, inds, vals);
  }

  bool isRowIntegral(HighsInt row) const {
    return lprow(row).isIntegral(mipsolver);
  }

  void setAdjustSymmetricBranchingCol(bool adjustSymBranchingCol) {
    sh_->adjust_sym = adjustSymBranchingCol;
  }

  void resetToGlobalDomain(const HighsDomain& globaldom);

  void computeBasicDegenerateDuals(double threshold, HighsDomain& localdom,
                                   HighsDomain& globaldom,
                                   HighsConflictPool& conflictpol,
                                   HighsPseudocost& pseudocost,
                                   bool getdualproof);

  double getAvgSolveIters() { return sh_->avg_solve_iters; }

  HighsInt getRowLen(HighsInt row) const {
    return lprow(row).getRowLen(mipsolver);
  }

  double getMaxAbsRowVal(HighsInt row) const {
    return lprow(row).getMaxAbsVal(mipsolver);
  }

  // The LP solver's model, solution, basis and info (valid until it
  // changes)
  highs_rs::LphView lpView() const {
    highs_rs::LphView v;
    highs_rs::highs_rs_lph_view(lp_, &v);
    return v;
  }

  // The LP solver's model LP as a C++ HighsLp
  HighsLp getLpCopy() const;

  // isBasisConsistent(LP, basis): of the LP's size, with a basic variable
  // per row
  bool isBasisConsistent(const HighsBasis& basis) const {
    const highs_rs::LphView v = lpView();
    if (basis.col_status.size() != size_t(v.num_col) ||
        basis.row_status.size() != size_t(v.num_row))
      return false;
    HighsInt num_basic = 0;
    for (HighsBasisStatus s : basis.col_status)
      num_basic += s == HighsBasisStatus::kBasic;
    for (HighsBasisStatus s : basis.row_status)
      num_basic += s == HighsBasisStatus::kBasic;
    return num_basic == v.num_row;
  }

  // The LP solution's column values and duals and row values (valid until
  // the LP solver changes)
  HighsRsSpan<double> lpColValue() const {
    const highs_rs::LphView v = lpView();
    return {v.col_value, size_t(v.n_col_value)};
  }
  HighsRsSpan<double> lpColDual() const {
    const highs_rs::LphView v = lpView();
    return {v.col_dual, size_t(v.n_col_dual)};
  }
  // copies of them
  std::vector<double> lpColValueVec() const {
    const HighsRsSpan<double> v = lpColValue();
    return std::vector<double>(v.begin(), v.end());
  }
  std::vector<double> lpColDualVec() const {
    const HighsRsSpan<double> v = lpColDual();
    return std::vector<double>(v.begin(), v.end());
  }

  double slackUpper(HighsInt row, const HighsDomain& globaldom) const;

  double slackLower(HighsInt row, const HighsDomain& globaldom) const;

  double rowLower(HighsInt row) const { return lpView().row_lower[row]; }

  double rowUpper(HighsInt row) const { return lpView().row_upper[row]; }

  double colLower(HighsInt col, const HighsDomain& globaldom) const {
    const highs_rs::LphView v = lpView();
    return col < v.num_col ? v.col_lower[col]
                           : slackLower(col - v.num_col, globaldom);
  }

  double colUpper(HighsInt col, const HighsDomain& globaldom) const {
    const highs_rs::LphView v = lpView();
    return col < v.num_col ? v.col_upper[col]
                           : slackUpper(col - v.num_col, globaldom);
  }

  bool isColIntegral(HighsInt col) const {
    const HighsInt num_col = lpView().num_col;
    return col < num_col ? mipsolver.isColIntegral(col)
                         : isRowIntegral(col - num_col);
  }

  double solutionValue(HighsInt col) const {
    const highs_rs::LphView v = lpView();
    return col < v.num_col ? v.col_value[col] : v.row_value[col - v.num_col];
  }

  Status getStatus() const { return Status(sh_->status); }

  const HighsInfoStruct& getSolverInfo() const { return *lpView().info; }

  HighsModelStatus getLpModelStatus() const {
    return HighsModelStatus(lpView().model_status);
  }

  int64_t getNumLpIterations() const { return sh_->numlpiters; }

  bool integerFeasible() const {
    Status status = getStatus();
    if ((status == Status::kOptimal ||
         status == Status::kUnscaledPrimalFeasible) &&
        sh_->num_frac == 0)
      return true;

    return false;
  }

  void setMipWorker(HighsMipWorker& worker) { worker_ = &worker; };

  double computeBestEstimate(const HighsPseudocost& ps) const;

  double computeLPDegneracy(const HighsDomain& localdomain) const;

  static bool scaledOptimal(Status status) {
    switch (status) {
      case Status::kOptimal:
      case Status::kUnscaledDualFeasible:
      case Status::kUnscaledPrimalFeasible:
      case Status::kUnscaledInfeasible:
        return true;
      default:
        return false;
    }
  }

  static bool unscaledPrimalFeasible(Status status) {
    switch (status) {
      case Status::kOptimal:
      case Status::kUnscaledPrimalFeasible:
        return true;
      default:
        return false;
    }
  }

  static bool unscaledDualFeasible(Status status) {
    switch (status) {
      case Status::kOptimal:
      case Status::kUnscaledDualFeasible:
        return true;
      default:
        return false;
    }
  }

  void recoverBasis();

  void setObjectiveLimit(double objlim = kHighsInf);

  // The LP solver's basis: whether it is valid, and a copy
  bool lpBasisValid() const { return lpView().basis_valid; }
  HighsBasis getLpBasis() const;

  // Highs::setBasis of the LP solver
  HighsStatus setLpBasis(const HighsBasis& basis,
                         const std::string& origin = "");

  // Highs::setOptionValue of the LP solver
  void setLpOption(const std::string& name, bool value);
  void setLpOption(const std::string& name, HighsInt value);
  void setLpOption(const std::string& name, double value);
  void setLpOption(const std::string& name, const std::string& value);
  void setLpOption(const std::string& name, const char* value) {
    setLpOption(name, std::string(value));
  }

  // The LP solver's Highs methods
  HighsStatus clearLpSolver();
  HighsStatus changeColsBounds(HighsInt from, HighsInt to, const double* lower,
                               const double* upper);
  HighsStatus changeColBounds(HighsInt col, double lower, double upper);
  HighsStatus changeColsCost(const HighsInt* mask, const double* cost);
  HighsStatus putIterate();
  HighsStatus getIterate();
  bool lpHasInvert() const { return highs_rs::highs_rs_lph_has_invert(lp_); }
  const HighsInt* lpBasicIndex() const {
    return highs_rs::highs_rs_lph_basic_index(lp_);
  }
  const double* lpDualEdgeWeights() const {
    return highs_rs::highs_rs_lph_dual_edge_weights(lp_);
  }
  // Highs::optimizeLp (a cancelled task throws HighsTask::Interrupt)
  HighsStatus optimizeLp();

  void storeBasis() {
    if (!currentbasisstored && lpBasisValid()) {
      basischeckpoint = std::make_shared<HighsBasis>(getLpBasis());
      currentbasisstored = true;
    }
  }

  std::shared_ptr<const HighsBasis> getStoredBasis() const {
    return basischeckpoint;
  }

  void setStoredBasis(std::shared_ptr<const HighsBasis> basis) {
    basischeckpoint = std::move(basis);
    currentbasisstored = false;
  }

  const HighsMipSolver& getMipSolver() const { return mipsolver; }

  HighsInt getNumModelRows() const { return mipsolver.numRow(); }

  HighsInt numRows() const { return lpView().num_row; }

  HighsInt numCols() const { return lpView().num_col; }

  HighsInt numNonzeros() const { return lpView().num_nz; }

  void addCuts(HighsCutSet& cutset);

  void performAging(bool deleteRows = false);

  void resetAges();

  void notifyCutPoolsLpCopied(HighsInt n);

  void removeObsoleteRows(bool notifyPool = true);

  void removeWorkerSpecificRows();

  void removeCuts();

  void flushDomain(HighsDomain& domain, bool continuous = false);

  void getDualProof(const HighsInt*& inds, const double*& vals, double& rhs,
                    HighsInt& len) {
    inds = sh_->proof_inds;
    vals = sh_->proof_vals;
    rhs = sh_->proof_rhs;
    len = sh_->proof_len;
  }

  bool computeDualProof(const HighsDomain& globaldomain, double upperbound,
                        std::vector<HighsInt>& inds, std::vector<double>& vals,
                        double& rhs, bool extractCliques = true) const;

  bool computeDualInfProof(const HighsDomain& globaldomain,
                           std::vector<HighsInt>& inds,
                           std::vector<double>& vals, double& rhs) const;

  Status resolveLp(HighsDomain* domain = nullptr);

  Status run(bool resolve_on_error = true);

  // The Rust LP solver (rust/src/lp_data/lp_handle.rs)
  highs_rs::LpHandle* lpHandle() const { return lp_; }

  HighsFracInts getFractionalIntegers() const {
    return HighsFracInts{sh_->frac, size_t(sh_->num_frac)};
  }

  double getObjective() const { return sh_->objective; }

  highs_rs::LpRelax* rust() const { return rs_; }
  highs_rs::LpShared* rustShared() const { return sh_; }

  void setIterationLimit(HighsInt limit = kHighsIInf) {
    setLpOption("simplex_iteration_limit", limit);
  }
  void setSolvedFirstLp(const bool solved_first_lp_) {
    this->solved_first_lp = solved_first_lp_;
  }

  // Race IPX against the dual simplex in solves without a basis
  void setRaceIpx(const bool race) { raceIpx = race; }
};
#else
class HighsLpRelaxation {
 public:
  enum class Status {
    kNotSet,
    kOptimal,
    kInfeasible,
    kUnscaledDualFeasible,
    kUnscaledPrimalFeasible,
    kUnscaledInfeasible,
    kUnbounded,
    kError,
  };

 private:
  struct LpRow {
    enum Origin {
      kModel,
      kCutPool,
    };

    Origin origin;
    HighsInt index;
    HighsInt age;
    HighsInt cutpoolindex;

    void get(const HighsMipSolver& mipsolver, HighsInt& len,
             const HighsInt*& inds, const double*& vals) const;

    HighsInt getRowLen(const HighsMipSolver& mipsolver) const;

    bool isIntegral(const HighsMipSolver& mipsolver) const;

    double getMaxAbsVal(const HighsMipSolver& mipsolver) const;

    static LpRow cut(HighsInt index, HighsInt cutpoolindex) {
      return LpRow{kCutPool, index, 0, cutpoolindex};
    }
    static LpRow model(HighsInt index) { return LpRow{kModel, index, 0, -1}; }
  };

  const HighsMipSolver& mipsolver;
  Highs lpsolver;

  std::vector<LpRow> lprows;

  std::vector<std::pair<HighsInt, double>> fractionalints;
  std::vector<double> dualproofvals;
  std::vector<HighsInt> dualproofinds;
  std::vector<double> dualproofbuffer;
  std::vector<double> colLbBuffer;
  std::vector<double> colUbBuffer;
  HVector row_ep;
  HighsSparseVectorSum row_ap;
  double dualproofrhs;
  bool hasdualproof;
  double objective;
  std::shared_ptr<const HighsBasis> basischeckpoint;
  bool currentbasisstored;
  int64_t numlpiters;
  int64_t lastAgeCall;
  double avgSolveIters;
  int64_t numSolved;
  size_t epochs;
  HighsInt maxNumFractional;
  Status status;
  bool adjustSymBranchingCol;
  bool solved_first_lp;
  bool raceIpx = false;
  HighsMipWorker* worker_;

  HighsStatus optimizeRacingIpx(int64_t& extraIterations);

  // run(): the solve (solver choice, IPM, race or simplex)
  HighsStatus runSolve(bool& use_simplex, int64_t& extraIterations);

  // run(): the IPM basis after the simplex hit its iteration limit
  void ipmBasisAfterIterationLimit();

  const LpRow& lprow(HighsInt row) const { return lprows[row]; }

  void storeDualInfProof();

  void storeDualUBProof();

  bool checkDualProof() const;

 public:
  HighsLpRelaxation(const HighsMipSolver& mip);

  HighsLpRelaxation(const HighsLpRelaxation& other);

  void setProfiling(HighsProfiling* profiling);

  void getCutPool(HighsInt& num_col, HighsInt& num_cut,
                  std::vector<double>& cut_lower,
                  std::vector<double>& cut_upper,
                  HighsSparseMatrix& cut_matrix) const;

  class Playground {
    friend class HighsLpRelaxation;
    HighsLpRelaxation* lp;
    bool iterateStored;

    Playground(HighsLpRelaxation* lp) : lp(lp), iterateStored(false) {}

   public:
    Playground(Playground&& other)
        : lp(other.lp), iterateStored(other.iterateStored) {
      other.iterateStored = false;
    }

    Playground& operator=(Playground&& other) {
      std::swap(lp, other.lp);
      std::swap(iterateStored, other.iterateStored);
      return *this;
    }

    HighsLpRelaxation::Status solveLp(HighsDomain& localdom) {
      if (iterateStored) {
        lp->flushDomain(localdom);
        lp->getLpSolver().getIterate();
      } else {
        assert(lp->getLpSolver().getInfo().valid);
        lp->getLpSolver().putIterate();
        lp->flushDomain(localdom);
        iterateStored = true;
      }

      return lp->run(false);
    }

    Playground(const Playground& other) = delete;
    Playground& operator=(const Playground& other) = delete;

    ~Playground() {
      if (iterateStored) {
        lp->getLpSolver().getIterate();
        lp->run();
        // If desired, here is the place to clear the stored iterate
      }
    }
  };

  Playground playground() { return Playground(this); }

  void loadModel();

  void getRow(HighsInt row, HighsInt& len, const HighsInt*& inds,
              const double*& vals) const {
    if (row < mipsolver.numRow())
      assert(lprows[row].origin == LpRow::Origin::kModel);
    else
      assert(lprows[row].origin == LpRow::Origin::kCutPool);
    lprows[row].get(mipsolver, len, inds, vals);
  }

  bool isRowIntegral(HighsInt row) const {
    assert(row < (HighsInt)lprows.size());
    return lprows[row].isIntegral(mipsolver);
  }

  void setAdjustSymmetricBranchingCol(bool adjustSymBranchingCol) {
    this->adjustSymBranchingCol = adjustSymBranchingCol;
  }

  void resetToGlobalDomain(const HighsDomain& globaldom);

  void computeBasicDegenerateDuals(double threshold, HighsDomain& localdom,
                                   HighsDomain& globaldom,
                                   HighsConflictPool& conflictpol,
                                   HighsPseudocost& pseudocost,
                                   bool getdualproof);

  double getAvgSolveIters() { return avgSolveIters; }

  HighsInt getRowLen(HighsInt row) const {
    return lprows[row].getRowLen(mipsolver);
  }

  double getMaxAbsRowVal(HighsInt row) const {
    return lprows[row].getMaxAbsVal(mipsolver);
  }

  const HighsLp& getLp() const { return lpsolver.getLp(); }

  const HighsSolution& getSolution() const { return lpsolver.getSolution(); }

  double slackUpper(HighsInt row, const HighsDomain& globaldom) const;

  double slackLower(HighsInt row, const HighsDomain& globaldom) const;

  double rowLower(HighsInt row) const {
    return lpsolver.getLp().row_lower_[row];
  }

  double rowUpper(HighsInt row) const {
    return lpsolver.getLp().row_upper_[row];
  }

  double colLower(HighsInt col, const HighsDomain& globaldom) const {
    return col < lpsolver.getLp().num_col_
               ? lpsolver.getLp().col_lower_[col]
               : slackLower(col - lpsolver.getLp().num_col_, globaldom);
  }

  double colUpper(HighsInt col, const HighsDomain& globaldom) const {
    return col < lpsolver.getLp().num_col_
               ? lpsolver.getLp().col_upper_[col]
               : slackUpper(col - lpsolver.getLp().num_col_, globaldom);
  }

  bool isColIntegral(HighsInt col) const {
    return col < lpsolver.getLp().num_col_
               ? mipsolver.isColIntegral(col)
               : isRowIntegral(col - lpsolver.getLp().num_col_);
  }

  double solutionValue(HighsInt col) const {
    return col < lpsolver.getLp().num_col_
               ? getSolution().col_value[col]
               : getSolution().row_value[col - lpsolver.getLp().num_col_];
  }

  Status getStatus() const { return status; }

  const HighsInfo& getSolverInfo() const { return lpsolver.getInfo(); }

  HighsModelStatus getLpModelStatus() const {
    return lpsolver.getModelStatus();
  }

  int64_t getNumLpIterations() const { return numlpiters; }

  bool integerFeasible() const {
    if ((status == Status::kOptimal ||
         status == Status::kUnscaledPrimalFeasible) &&
        fractionalints.empty())
      return true;

    return false;
  }

  void setMipWorker(HighsMipWorker& worker) { worker_ = &worker; };

  double computeBestEstimate(const HighsPseudocost& ps) const;

  double computeLPDegneracy(const HighsDomain& localdomain) const;

  static bool scaledOptimal(Status status) {
    switch (status) {
      case Status::kOptimal:
      case Status::kUnscaledDualFeasible:
      case Status::kUnscaledPrimalFeasible:
      case Status::kUnscaledInfeasible:
        return true;
      default:
        return false;
    }
  }

  static bool unscaledPrimalFeasible(Status status) {
    switch (status) {
      case Status::kOptimal:
      case Status::kUnscaledPrimalFeasible:
        return true;
      default:
        return false;
    }
  }

  static bool unscaledDualFeasible(Status status) {
    switch (status) {
      case Status::kOptimal:
      case Status::kUnscaledDualFeasible:
        return true;
      default:
        return false;
    }
  }

  void recoverBasis();

  void setObjectiveLimit(double objlim = kHighsInf);

  void storeBasis() {
    if (!currentbasisstored && lpsolver.getBasis().valid) {
      basischeckpoint = std::make_shared<HighsBasis>(lpsolver.getBasis());
      currentbasisstored = true;
    }
  }

  std::shared_ptr<const HighsBasis> getStoredBasis() const {
    return basischeckpoint;
  }

  void setStoredBasis(std::shared_ptr<const HighsBasis> basis) {
    basischeckpoint = std::move(basis);
    currentbasisstored = false;
  }

  const HighsMipSolver& getMipSolver() const { return mipsolver; }

  HighsInt getNumModelRows() const { return mipsolver.numRow(); }

  HighsInt numRows() const { return lpsolver.getNumRow(); }

  HighsInt numCols() const { return lpsolver.getNumCol(); }

  HighsInt numNonzeros() const { return lpsolver.getNumNz(); }

  void addCuts(HighsCutSet& cutset);

  void performAging(bool deleteRows = false);

  void resetAges();

  void notifyCutPoolsLpCopied(HighsInt n);

  void removeObsoleteRows(bool notifyPool = true);

  void removeWorkerSpecificRows();

  void removeCuts(HighsInt ndelcuts, std::vector<HighsInt>& deletemask);

  void removeCuts();

  void flushDomain(HighsDomain& domain, bool continuous = false);

  void getDualProof(const HighsInt*& inds, const double*& vals, double& rhs,
                    HighsInt& len) {
    inds = dualproofinds.data();
    vals = dualproofvals.data();
    rhs = dualproofrhs;
    len = dualproofinds.size();
  }

  bool computeDualProof(const HighsDomain& globaldomain, double upperbound,
                        std::vector<HighsInt>& inds, std::vector<double>& vals,
                        double& rhs, bool extractCliques = true) const;

  bool computeDualInfProof(const HighsDomain& globaldomain,
                           std::vector<HighsInt>& inds,
                           std::vector<double>& vals, double& rhs) const;

  Status resolveLp(HighsDomain* domain = nullptr);

  Status run(bool resolve_on_error = true);

  Highs& getLpSolver() { return lpsolver; }
  const Highs& getLpSolver() const { return lpsolver; }

  const std::vector<std::pair<HighsInt, double>>& getFractionalIntegers()
      const {
    return fractionalints;
  }

  std::vector<std::pair<HighsInt, double>>& getFractionalIntegers() {
    return fractionalints;
  }

  double getObjective() const { return objective; }

  void setIterationLimit(HighsInt limit = kHighsIInf) {
    lpsolver.setOptionValue("simplex_iteration_limit", limit);
  }
  void setSolvedFirstLp(const bool solved_first_lp_) {
    this->solved_first_lp = solved_first_lp_;
  }

  // Race IPX against the dual simplex in solves without a basis
  void setRaceIpx(const bool race) { raceIpx = race; }
};
#endif  // HIGHS_RUST

#endif
