/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file simplex/HEkk.h
 * @brief Primal simplex solver for HiGHS
 */
#ifndef SIMPLEX_HEKK_H_
#define SIMPLEX_HEKK_H_

#include "lp_data/HighsCallback.h"
#include "simplex/HSimplexNla.h"
#include "simplex/HighsSimplexAnalysis.h"
#include "util/HSet.h"
#include "util/HighsHash.h"
#include "util/HighsRandom.h"

class HighsLpSolverObject;

#ifdef HIGHS_RUST
#include "simplex/HEkkRust.h"

// HEkk is a shell of the Rust-owned simplex engine (rust/src/simplex/
// lp_solver.rs), which owns the LP being solved (a copy of the C++ LP,
// made by moveLp; solveLpSimplex and dualization are Rust too,
// rust/src/simplex/app.rs). The shell keeps the pointers to the options,
// callback and timer, the analysis, a C++ LP of the simplex NLA and the
// records the API returns by reference. The scalars that C++ reads and
// writes (status_, info_, model_status_, ...) are references into the
// Rust object.
class HEkk {
 public:
  HEkk();
  ~HEkk();
  HEkk(const HEkk&) = delete;
  HEkk& operator=(const HEkk&) = delete;

  void clear();
  void clearEkkLp();
  void clearRayRecords();
  void invalidate();
  void updateStatus(LpAction action);
  void setNlaPointersForLpAndScale(const HighsLp& lp);
  void btran(HVector& rhs, const double expected_density);
  void ftran(HVector& rhs, const double expected_density);
  // Copy the solver object's LP into the engine, then movedLp
  void moveLp(HighsLpSolverObject& solver_object);
  // The pointers to the solver object's options, callback and timer, and
  // the engine's checks of its (copied) LP
  void movedLp(HighsLpSolverObject& solver_object);
  void movedLp(HighsCallback& callback, HighsOptions& options,
               HighsTimer& timer);
  // What clear() clears on the C++ side (after the engine cleared itself)
  void clearShell() { clearCpp(); }
  // The environment of a call into the engine (for Rust)
  highs_rs::LpsEnv callEnv() const;
  // A C++ LP takes the engine LP's scale (and with `matrix` its
  // constraint matrix, which an undualized LP rebuilt)
  void lpBack(HighsLp& lp, const bool matrix) const;
  // The simplex NLA's LP is the engine's
  void setNlaEngineLp();
  // The engine LP's dimensions
  HighsInt lpNumCol() const;
  HighsInt lpNumRow() const;
  void setPointers(HighsCallback* callback, HighsOptions* options,
                   HighsTimer* timer);

  HighsStatus solve(const bool force_phase2 = false);
  HighsStatus setBasis();
  HighsStatus setBasis(const HighsBasis& highs_basis);

  void putIterate();
  HighsStatus getIterate();

  void addCols(const HighsLp& lp, const HighsSparseMatrix& scaled_a_matrix);
  void addRows(const HighsLp& lp, const HighsSparseMatrix& scaled_ar_matrix);
  void deleteCols(const HighsIndexCollection& index_collection);
  void deleteRows(const HighsIndexCollection& index_collection);
  void unscaleSimplex(const HighsLp& incumbent_lp);
  bool proofOfPrimalInfeasibility();

  HighsBasis getHighsBasis(HighsLp& use_lp) const;
  double computeBasisCondition(const HighsLp& lp, const bool exact = false,
                               const bool report = false) const;
  // Of the engine's LP
  double computeBasisCondition() const;
  HighsStatus initialiseSimplexLpBasisAndFactor(
      const bool only_from_known_basis = false);
  bool lpFactorRowCompatible() const;
  bool lpFactorRowCompatible(const HighsInt expectedNumRow) const;
  std::string simplexStrategyToString(const HighsInt simplex_strategy) const;
  // getUnscaledInfeasibilities (simplex/HSimplex.cpp) of the scaled
  // simplex data, with `lp`'s scale factors
  void getUnscaledInfeasibilities(const HighsLp& lp,
                                  HighsInfo& highs_info) const;

  // The simplex basis (valid until it changes); the slices are empty
  // unless `use`
  HighsInt* basicIndex() const;
  RsMut<HighsInt> basicIndexSlice(const bool use = true) const;
  RsMut<int8_t> nonbasicSlice(const bool move, const bool use = true) const;
  HighsInt dualRayIndex() const { return sh_.dual_ray_index; }
  HighsInt dualRaySign() const { return sh_.dual_ray_sign; }
  // For new nonbasic columns: resize the simplex basis to num_tot
  void resizeBasis(const HighsInt num_tot);
  void appendBasicRows(const HighsInt num_col, const HighsInt num_row,
                       const HighsInt new_num_row);
  void flipNonbasicMove(const HighsInt var);
  // The DSE weights, if any
  const double* dualEdgeWeights() const;
  // The values of the dual and primal ray records
  std::vector<double> rayValue(const bool primal) const;
  void setRayValue(const bool primal, const std::vector<double>& value);
  highs_rs::RangingSlices rangingSlices();

  const HighsSimplexStats& getSimplexStats() const { return simplex_stats_; }
  void initialiseSimplexStats() { simplex_stats_.initialise(iteration_count_); }
  void reportSimplexStats(FILE* file, const std::string message = "") const {
    simplex_stats_.report(file, message);
  }
  // Debugging is left out of this build
  HighsDebugStatus debugRetainedDataOk(const HighsLp&) const {
    return HighsDebugStatus::kNotChecked;
  }
  HighsDebugStatus debugNlaCheckInvert(const std::string,
                                       const HighsInt = -1) const {
    return HighsDebugStatus::kNotChecked;
  }
  bool debugNlaScalingOk(const HighsLp&) const { return true; }

  // Data members
  HighsCallback* callback_;
  HighsOptions* options_;
  HighsTimer* timer_;
  HighsSimplexAnalysis analysis_;

  std::string lp_name_;

  // The Rust simplex engine and its shared scalars
  void* rs_;
  highs_rs::HEkkShared& sh_;
  HighsSimplexStatus& status_;
  highs_rs::HEkkInfo& info_;
  HighsModelStatus& model_status_;
  HighsInt& iteration_count_;
  SimplexAlgorithm& exit_algorithm_;
  bool& dual_values_valid_;
  HighsInt& debug_solve_call_num_;
  HighsInt& debug_initial_build_synthetic_tick_;

  // The LP of the simplex NLA and its scale factors if they are to be
  // applied (HSimplexNla::lp_, scale_)
  const HighsLp* nla_lp_;
  const HighsScale* nla_scale_;

  // Unused, but retained since there is a const reference to this in
  // a deprecated method
  HotStart hot_start_;
  std::vector<double> primal_phase1_dual_;
  HighsSimplexStats simplex_stats_;

 private:
  struct RustHost;
  // The factor's log options: a copy of the options' log flags made when
  // the simplex NLA is set up, without callbacks (HFactor::log_options)
  struct FactorLogData {
    bool output_flag = false;
    bool log_to_console = true;
    HighsInt log_dev_level = 0;
  };
  FactorLogData factor_log_data_;
  HighsLogOptions factor_log_options_;
  void snapshotFactorLog();
  // What clear() clears on the C++ side
  void clearCpp();
  // The view of the LP, the NLA's LP, the options and the host functions
  // for a call
  highs_rs::LpsEnv rsEnv(RustHost& host) const;
  // Take what a solve or INVERT left in the Rust records
  void takeRustOut();
  void setNlaLp(const HighsLp& lp);
  HighsStatus returnFromEkkSolve(const HighsStatus return_status,
                                 const highs_rs::LpsSolveOut& out);
};

// The factor's log options for a solveLpSimplex host: a copy of the
// options' log flags without callbacks
struct RsFactorLogStore {
  bool output_flag = false, log_to_console = true;
  HighsInt log_dev_level = 0;
  HighsLogOptions log_options;
  void set(const HighsLogOptions& from);
};
// The options and logs of solveLpSimplex on Rust data (lp_run.rs) into
// out (a CSimplexApp), the incumbent `lp`
void rsSimplexAppTemplate(HighsOptions& options, HighsLp& lp,
                          RsFactorLogStore& factor_log, void* out);
// A step of solveLpSimplex on the HEkk shell (app.rs ops)
int64_t rsSimplexShellOp(HEkk& ekk, HighsProfiling* profiling,
                         HighsOptions& options, HighsCallback& callback,
                         HighsTimer& timer, HighsLp& lp, int code,
                         int64_t arg, void* p);

#else

class HEkk {
 public:
  HEkk()
      : callback_(nullptr),
        options_(nullptr),
        timer_(nullptr),
        lp_name_(""),
        model_status_(HighsModelStatus::kNotset),
        simplex_in_scaled_space_(false),
        cost_scale_(1.0),
        cost_perturbation_base_(0.0),
        cost_perturbation_max_abs_cost_(0.0),
        iteration_count_(0),
        dual_simplex_cleanup_level_(0),
        dual_simplex_phase1_cleanup_level_(0),
        previous_iteration_cycling_detected(-kHighsIInf),
        solve_bailout_(false),
        called_return_from_solve_(false),
        exit_algorithm_(SimplexAlgorithm::kNone),
        return_primal_solution_status_(0),
        return_dual_solution_status_(0),
        original_num_col_(0),
        original_num_row_(0),
        original_num_nz_(0),
        original_offset_(0.0),
        edge_weight_error_(0.0),
        build_synthetic_tick_(0.0),
        total_synthetic_tick_(0.0),
        debug_solve_call_num_(0),
        debug_basis_id_(0),
        time_report_(false),
        debug_initial_build_synthetic_tick_(0),
        debug_solve_report_(false),
        debug_iteration_report_(false),
        debug_basis_report_(false),
        debug_dual_feasible(false),
        debug_max_relative_dual_steepest_edge_weight_error(0) {}
  /**
   * @brief Interface to simplex solvers
   */
  void clear();
  void clearEkkLp();
  void clearEkkData();
  void clearEkkDualize();
  void clearEkkDualEdgeWeightData();
  void clearEkkPointers();
  void clearEkkDataInfo();
  void clearEkkControlInfo();
  void clearEkkNlaInfo();
  void clearEkkAllStatus();
  void clearEkkDataStatus();
  void clearNlaStatus();
  void clearNlaInvertStatus();
  void clearRayRecords();

  void invalidate();
  void invalidateBasisMatrix();
  void invalidateBasis();
  void invalidateBasisArtifacts();

  void updateStatus(LpAction action);
  void setNlaPointersForLpAndScale(const HighsLp& lp);
  void setNlaPointersForTrans(const HighsLp& lp);
  void setNlaRefactorInfo();
  void btran(HVector& rhs, const double expected_density);
  void ftran(HVector& rhs, const double expected_density);

  void moveLp(HighsLpSolverObject& solver_object);
  void setPointers(HighsCallback* callback, HighsOptions* options,
                   HighsTimer* timer);
  HighsSparseMatrix* getScaledAMatrixPointer();
  HighsScale* getScalePointer();

  void initialiseEkk();
  HighsStatus dualize();
  HighsStatus undualize();
  HighsStatus permute();
  HighsStatus unpermute();
  HighsStatus solve(const bool force_phase2 = false);
  HighsStatus setBasis();
  HighsStatus setBasis(const HighsBasis& highs_basis);

  void putIterate();
  HighsStatus getIterate();

  void addCols(const HighsLp& lp, const HighsSparseMatrix& scaled_a_matrix);
  void addRows(const HighsLp& lp, const HighsSparseMatrix& scaled_ar_matrix);
  void deleteCols(const HighsIndexCollection& index_collection);
  void deleteRows(const HighsIndexCollection& index_collection);
  void unscaleSimplex(const HighsLp& incumbent_lp);
  double factorSolveError();

  bool proofOfPrimalInfeasibility();
  bool proofOfPrimalInfeasibility(HVector& row_ep, const HighsInt move_out,
                                  const HighsInt row_out);

  double getValueScale(const HighsInt count, const double* value) const;
  double getMaxAbsRowValue(HighsInt row);

  void unitBtranIterativeRefinement(const HighsInt row_out, HVector& row_ep);
  void unitBtranResidual(const HighsInt row_out, const HVector& row_ep,
                         HVector& residual, double& residual_norm);

  HighsSolution getSolution();
  HighsBasis getHighsBasis(HighsLp& use_lp) const;

  const SimplexBasis& getSimplexBasis() { return basis_; }
  HighsInt* basicIndex() const {
    return const_cast<HighsInt*>(basis_.basicIndex_.data());
  }
  HighsInt dualRayIndex() const { return dual_ray_record_.index; }
  HighsInt dualRaySign() const { return dual_ray_record_.sign; }
  double computeBasisCondition(const HighsLp& lp, const bool exact = false,
                               const bool report = false) const;
  double computeBasisCondition() const {
    return computeBasisCondition(this->lp_, false, false);
  }

  HighsStatus initialiseSimplexLpBasisAndFactor(
      const bool only_from_known_basis = false);
  void handleRankDeficiency();
  void initialisePartitionedRowwiseMatrix();
  bool lpFactorRowCompatible() const;
  bool lpFactorRowCompatible(const HighsInt expectedNumRow) const;
  std::string simplexStrategyToString(const HighsInt simplex_strategy) const;

  void zeroBasicDuals();

  // Interface methods
  void appendColsToVectors(const HighsInt num_new_col,
                           const vector<double>& colCost,
                           const vector<double>& colLower,
                           const vector<double>& colUpper);
  void appendRowsToVectors(const HighsInt num_new_row,
                           const vector<double>& rowLower,
                           const vector<double>& rowUpper);

  const HighsSimplexStats& getSimplexStats() const { return simplex_stats_; }
  void initialiseSimplexStats() { simplex_stats_.initialise(iteration_count_); }
  void reportSimplexStats(FILE* file, const std::string message = "") const {
    simplex_stats_.report(file, message);
  }

  // Make this private later
  void chooseSimplexStrategyThreads(const HighsOptions& options,
                                    HighsSimplexInfo& info);
  // Debug methods
  void debugInitialise();
  void debugReportInitialBasis();
  void debugReporting(
      const HighsInt save_mod_recover,
      const HighsInt log_dev_level_ = kHighsLogDevLevelDetailed);
  void timeReporting(const HighsInt save_mod_recover);
  HighsDebugStatus debugRetainedDataOk(const HighsLp& lp) const;
  HighsDebugStatus debugNlaCheckInvert(
      const std::string message, const HighsInt alt_debug_level = -1) const;
  bool debugNlaScalingOk(const HighsLp& lp) const;


  // Data members
  HighsCallback* callback_;
  HighsOptions* options_;
  HighsTimer* timer_;
  HighsSimplexAnalysis analysis_;

  HighsLp lp_;
  std::string lp_name_;
  HighsSimplexStatus status_;
  HighsSimplexInfo info_;
  HighsModelStatus model_status_;
  SimplexBasis basis_;
  HighsHashTable<uint64_t> visited_basis_;
  HighsRandom random_;
  std::vector<double> dual_edge_weight_;
  std::vector<double> scattered_dual_edge_weight_;
  // DSE weights keyed by variable (-1 if unknown), carried over LP
  // changes that keep the basis apart from added/deleted logicals
  // (cuts) and over basis resets; restored by the next dual solve
  std::vector<double> saved_dual_edge_weight_;
  // Work vectors of computePrimal, computeDual and factorSolveError, kept
  // so that each call does not set up (allocate and zero) new ones
  HVector work_col_;
  HVector work_col2_;
  HVector work_row_;
  HVector& workVector(HVector& vector, const HighsInt size) {
    if (vector.size != size || HighsInt(vector.array.size()) != size)
      vector.setup(size);
    else
      vector.clear();
    return vector;
  }

  bool simplex_in_scaled_space_;
  HighsSparseMatrix ar_matrix_;
  bool ar_matrix_is_scaled_ = false;
  bool random_vectors_drawn_for_solve_ = false;
  // Whether info_.workDual_ holds the reduced costs of the current basis for
  // the LP's own costs (no perturbation or shift): set at the end of a dual
  // simplex solve that ends optimal, cleared by anything that changes the
  // basis, the costs, the scaling or the dual values. A solve after bound
  // changes alone then need not compute them again
  bool dual_values_valid_ = false;
  // whether those reduced costs are of the scaled LP, and hashes of the
  // costs and the basis they are for: some changes of the costs (e.g. user
  // objective scaling, blending objectives) reach the LP without notice
  bool dual_values_scaled_ = false;
  uint64_t dual_values_cost_hash_ = 0;
  uint64_t dual_values_basis_hash_ = 0;
  uint64_t costHash() const {
    return HighsHashHelpers::vector_hash(lp_.col_cost_.data(),
                                         lp_.col_cost_.size()) ^
           (HighsHashHelpers::hash(lp_.offset_) + lp_.num_row_);
  }
  // Values computed by initialiseForSolve or the dual simplex set-up that
  // nothing has changed since, so the dual simplex need not recompute them
  bool fresh_unperturbed_dual_ = false;
  bool fresh_dual_ = false;
  bool fresh_primal_ = false;
  void clearFreshValues() {
    fresh_unperturbed_dual_ = false;
    fresh_dual_ = false;
    fresh_primal_ = false;
  }
  HighsSparseMatrix scaled_a_matrix_;
  HSimplexNla simplex_nla_;

  // Unused, but retained since there is a const reference to this in
  // a deprecated method
  HotStart hot_start_;

  double cost_scale_;
  double cost_perturbation_base_;
  double cost_perturbation_max_abs_cost_;
  HighsInt iteration_count_;
  HighsInt dual_simplex_cleanup_level_;
  HighsInt dual_simplex_phase1_cleanup_level_;

  HighsInt previous_iteration_cycling_detected;

  bool solve_bailout_;
  bool called_return_from_solve_;
  SimplexAlgorithm exit_algorithm_;
  HighsInt return_primal_solution_status_;
  HighsInt return_dual_solution_status_;

  // Data to be retained after proving primal infeasibility
  vector<HighsInt> proof_index_;
  vector<double> proof_value_;

  // Data to be retained after computing primal or dual ray
  HighsRayRecord dual_ray_record_;
  HighsRayRecord primal_ray_record_;

  // Data to be retained when dualizing
  HighsInt original_num_col_;
  HighsInt original_num_row_;
  HighsInt original_num_nz_;
  double original_offset_;
  vector<double> original_col_cost_;
  vector<double> original_col_lower_;
  vector<double> original_col_upper_;
  vector<double> original_row_lower_;
  vector<double> original_row_upper_;
  //
  // The upper_bound_col vector accumulates the indices of boxed
  // variables, whose upper bounds are treated as additional
  // constraints.
  //
  // The upper_bound_row vector accumulates the indices of boxed
  // constraints, whose upper bounds are treated as additional
  // constraints.
  vector<HighsInt> upper_bound_col_;
  vector<HighsInt> upper_bound_row_;

  double edge_weight_error_;

  double build_synthetic_tick_;
  double total_synthetic_tick_;
  HighsInt debug_solve_call_num_;
  HighsInt debug_basis_id_;
  bool time_report_;
  HighsInt debug_initial_build_synthetic_tick_;
  bool debug_solve_report_;
  bool debug_iteration_report_;
  bool debug_basis_report_;
  bool debug_dual_feasible;
  double debug_max_relative_dual_steepest_edge_weight_error;

  std::vector<HighsSimplexBadBasisChangeRecord> bad_basis_change_;
  std::vector<double> primal_phase1_dual_;

  HighsSimplexStats simplex_stats_;

 private:
  bool isUnconstrainedLp() const;
  void initialiseForSolve();
  void setSimplexOptions();
  void updateSimplexOptions();
  void initialiseSimplexLpRandomVectors();
  void setNonbasicMove();
  bool getNonsingularInverse(const HighsInt solve_phase = 0);
  bool getBacktrackingBasis();
  void putBacktrackingBasis();
  void putBacktrackingBasis(
      const vector<HighsInt>& basicIndex_before_compute_factor);
  void computePrimalObjectiveValue();
  void computeDualObjectiveValue(const HighsInt phase = 2);
  bool rebuildRefactor(HighsInt rebuild_reason);
  HighsInt computeFactor();
  void computeDualSteepestEdgeWeights(const bool initial = false);
  double computeDualSteepestEdgeWeight(const HighsInt iRow, HVector& row_ep);
  std::vector<double> scatterDualEdgeWeights(
      const HighsInt num_weighted_row, const HighsInt new_num_row,
      const std::vector<HighsInt>* new_row_index) const;
  bool restoreDualEdgeWeights(const bool near_optimal);
  void updateDualSteepestEdgeWeights(const HighsInt row_out,
                                     const HighsInt variable_in,
                                     const HVector* column,
                                     const double new_pivotal_edge_weight,
                                     const double Kai,
                                     const double* dual_steepest_edge_array);
  void updateDualDevexWeights(const HVector* column,
                              const double new_pivotal_edge_weight);
  void resetSyntheticClock();
  void allocateWorkAndBaseArrays();
  void initialiseCost(const SimplexAlgorithm algorithm,
                      const HighsInt solve_phase, const bool perturb = false);
  void initialiseBound(const SimplexAlgorithm algorithm,
                       const HighsInt solve_phase, const bool perturb = false);
  void initialiseLpColCost();
  void initialiseLpRowCost();
  void initialiseLpColBound();
  void initialiseLpRowBound();
  void initialiseNonbasicValueAndMove();
  void pivotColumnFtran(const HighsInt iCol, HVector& col_aq);
  void unitBtran(const HighsInt iRow, HVector& row_ep);
  void fullBtran(HVector& buffer);
  void choosePriceTechnique(const HighsInt price_strategy,
                            const double row_ep_density, bool& use_col_price,
                            bool& use_row_price_w_switch) const;
  void tableauRowPrice(const bool quad_precision, const HVector& row_ep,
                       HVector& row_ap,
                       const HighsInt debug_report = kDebugReportOff);
  void fullPrice(const HVector& full_col, HVector& full_row);
  void computePrimal();
  void computeDual();
  double computeDualForTableauColumn(const HighsInt iVar,
                                     const HVector& tableau_column) const;
  bool reinvertOnNumericalTrouble(const std::string method_name,
                                  double& numerical_trouble_measure,
                                  const double alpha_from_col,
                                  const double alpha_from_row,
                                  const double numerical_trouble_tolerance);

  void flipBound(const HighsInt iCol);
  void updateFactor(HVector* column, HVector* row_ep, HighsInt* iRow,
                    HighsInt* hint);

  void transformForUpdate(HVector* column, HVector* row_ep,
                          const HighsInt variable_in, HighsInt* row_out);

  void updatePivots(const HighsInt variable_in, const HighsInt row_out,
                    const HighsInt move_out);
  bool isBadBasisChange(const SimplexAlgorithm algorithm,
                        const HighsInt variable_in, const HighsInt row_out,
                        const HighsInt rebuild_reason);
  void updateMatrix(const HighsInt variable_in, const HighsInt variable_out);

  void computeInfeasibilitiesForReporting(
      const SimplexAlgorithm algorithm,
      const HighsInt solve_phase = kSolvePhase2);
  void computeSimplexInfeasible();
  void computeSimplexPrimalInfeasible();
  void computeSimplexDualInfeasible();
  void computeSimplexLpDualInfeasible();

  void invalidatePrimalInfeasibilityRecord();
  void invalidatePrimalMaxSumInfeasibilityRecord();
  void invalidateDualInfeasibilityRecord();
  void invalidateDualMaxSumInfeasibilityRecord();
  bool bailout();
  HighsStatus returnFromEkkSolve(const HighsStatus return_status);
  HighsStatus returnFromSolve(const HighsStatus return_status);

  void initialiseAnalysis();
  std::string rebuildReason(const HighsInt rebuild_reason) const;

  void clearBadBasisChange(
      const BadBasisChangeReason reason = BadBasisChangeReason::kAll);
  void updateBadBasisChange(const HVector& col_aq, double theta_primal);

  HighsInt addBadBasisChange(const HighsInt row_out,
                             const HighsInt variable_out,
                             const HighsInt variable_in,
                             const BadBasisChangeReason reason,
                             const bool taboo = false);
  void clearBadBasisChangeTabooFlag();
  bool tabooBadBasisChange() const;
  void applyTabooRowOut(double* values, const double overwrite_with);
  void unapplyTabooRowOut(double* values);
  void applyTabooVariableIn(vector<double>& values,
                            const double overwrite_with);
  void unapplyTabooVariableIn(vector<double>& values);
  bool logicalBasis() const;
  // Methods in HEkkControl
  void initialiseControl();
  void assessDSEWeightError(const double computed_edge_weight,
                            const double updated_edge_weight);
  void updateOperationResultDensity(const double local_density,
                                    double& density) const;
  bool switchToDevex();

  // private debug methods
  HighsDebugStatus debugSimplex(const std::string message,
                                const SimplexAlgorithm algorithm,
                                const HighsInt phase,
                                const bool initialise = false) const;
  void debugReportReinvertOnNumericalTrouble(
      const std::string method_name, const double numerical_trouble_measure,
      const double alpha_from_col, const double alpha_from_row,
      const double numerical_trouble_tolerance, const bool reinvert) const;

  HighsDebugStatus debugUpdatedDual(const double updated_dual,
                                    const double computed_dual) const;

  HighsDebugStatus debugBasisCorrect(const HighsLp* lp = NULL) const;
  HighsDebugStatus debugBasisConsistent() const;
  HighsDebugStatus debugNonbasicFlagConsistent() const;
  HighsDebugStatus debugNonbasicMove(const HighsLp* lp = NULL) const;
  HighsDebugStatus debugOkForSolve(const SimplexAlgorithm algorithm,
                                   const HighsInt phase) const;
  bool debugWorkArraysOk(const SimplexAlgorithm algorithm,
                         const HighsInt phase) const;
  bool debugOneNonbasicMoveVsWorkArraysOk(const HighsInt var) const;

  HighsDebugStatus debugNonbasicFreeColumnSet(
      const HighsInt num_free_col, const HSet nonbasic_free_col_set) const;
  HighsDebugStatus debugRowMatrix() const;
  HighsDebugStatus devDebugDualSteepestEdgeWeights(const std::string message);
  HighsDebugStatus debugDualSteepestEdgeWeights(
      const HighsInt alt_debug_level = -1);
  HighsDebugStatus debugSimplexDualInfeasible(const std::string message,
                                              const bool force_report = false);
  HighsDebugStatus debugComputeDual(const bool initialise = false) const;
  bool debugNoShiftsOrPerturbations() const;
  friend class HEkkPrimal;
  friend class HEkkDual;
  friend class HEkkDualRow;
  friend class HEkkDualRHS;  // For  HEkkDualRHS::assessOptimality
};

#endif  // HIGHS_RUST

#endif /* SIMPLEX_HEKK_H_ */
