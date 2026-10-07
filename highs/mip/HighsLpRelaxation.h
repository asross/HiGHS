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

class HighsDomain;
struct HighsCutSet;
class HighsPseudocost;
class HighsMipWorker;

#ifdef HIGHS_RUST
namespace highs_rs {
struct LpRelax;
struct CutPool;
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
}  // namespace highs_rs

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
  Highs lpsolver;
  highs_rs::LpRelax* rs_;
  highs_rs::LpShared* sh_;

  std::vector<double> colLbBuffer;
  std::vector<double> colUbBuffer;
  HVector row_ep;
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

  HighsStatus optimizeRacingIpx(int64_t& extraIterations);

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

  Status getStatus() const { return Status(sh_->status); }

  const HighsInfo& getSolverInfo() const { return lpsolver.getInfo(); }

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

  Highs& getLpSolver() { return lpsolver; }
  const Highs& getLpSolver() const { return lpsolver; }

  HighsFracInts getFractionalIntegers() const {
    return HighsFracInts{sh_->frac, size_t(sh_->num_frac)};
  }

  double getObjective() const { return sh_->objective; }

  highs_rs::LpRelax* rust() const { return rs_; }
  highs_rs::LpShared* rustShared() const { return sh_; }

  void setIterationLimit(HighsInt limit = kHighsIInf) {
    lpsolver.setOptionValue("simplex_iteration_limit", limit);
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
