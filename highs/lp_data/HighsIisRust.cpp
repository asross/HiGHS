/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file lp_data/HighsIisRust.cpp
 * @brief The IIS done by Rust (rust/src/lp_data/iis.rs): getIisInterface
 * and elasticityFilter, with HighsIis's methods. Rust calls back here for
 * each step on a Highs object: the incumbent (handle 0) or one the IIS
 * search creates for its LP solves.
 */
#include "lp_data/HighsRust.h"

#ifdef HIGHS_RUST
#include <cstdint>
#include <exception>
#include <memory>

#include "Highs.h"

static_assert(sizeof(HighsIisInfo) == 56, "HighsIisInfo layout");

namespace {

struct RsIisStr {
  const char* ptr;
  size_t len;
};

// iis.rs: IisState
struct RsIisState {
  bool valid;
  HighsInt status, strategy;
  RsMut<HighsInt> col_index, row_index, col_bound, row_bound, col_status,
      row_status;
  HighsIisInfo info;
};

// iis.rs: LpArrays
struct RsIisLp {
  HighsInt num_col, num_row, format, model_name;
  RsMut<double> col_cost, col_lower, col_upper, row_lower, row_upper;
  RsMut<HighsInt> start, index;
  RsMut<double> value;
  RsMut<HighsInt> col_map, row_map;
};

// iis.rs: Args
struct RsIisArgs {
  HighsInt h, i, j;
  double x, y;
  RsIisStr s;
  const void* p[6];
};

// iis.rs: IisHost
struct RsIisHost {
  RsLog log;
  void* ctx;
  double (*op)(void*, int, const RsIisArgs*);
  void (*lp)(void*, RsLp*, size_t*, size_t*);
  RsIisStr (*name)(void*, bool, HighsInt);
  RsMut<double> (*col_value)(void*);
};

// iis.rs: Op
enum class IisOp {
  kLoadIis = 1,
  kStoreIis,
  kClearIisModel,
  kSetIisLp,
  kNewHighs,
  kDeleteHighs,
  kPassOptions,
  kSetOptionBool,
  kSetOptionInt,
  kSetOptionDouble,
  kSetOptionString,
  kPropagateCallbacks,
  kPassModelArrays,
  kPassIisModel,
  kChangeColsCost,
  kOptimizeModel,
  kRunTime,
  kSimplexIterations,
  kModelStatus,
  kChangeColBounds,
  kChangeRowBounds,
  kWriteModel,
  kZeroAllClocks,
  kCallbackActive,
  kStopCallback,
  kStartCallback,
  kSaveOptions,
  kRestoreOptions,
  kEnsureColwise,
  kGetOption,
  kSetOutputFlag,
  kInvalidateSolverData,
  kPassModelName,
  kChangeColsIntegrality,
  kChangeColsBounds,
  kAddCols,
  kAddRows,
  kPassColName,
  kPassRowName,
  kDeleteRows,
  kDeleteCols,
  kBasisInvalid,
  kElasticSolution,
  kSetModelStatus,
  kObjectiveValue,
  kEngine,
  kCallbacksToPropagate,
};

template <typename T>
void assignFrom(std::vector<T>& v, const RsMut<T>& m) {
  v.assign(m.ptr, m.ptr + m.len);
}

std::string str(const RsIisStr& s) { return std::string(s.ptr, s.len); }

}  // namespace

extern "C" {
int highs_rs_get_iis(const RsIisHost* host);
int highs_rs_elasticity_filter(const RsIisHost* host,
                               double global_lower_penalty,
                               double global_upper_penalty,
                               double global_rhs_penalty,
                               const double* local_lower_penalty,
                               const double* local_upper_penalty,
                               const double* local_rhs_penalty, bool get_iis);
}

// The steps of the IIS on Highs objects (a friend)
struct HighsIisRust {
  Highs& h;
  // The IIS LP (HighsIis::model_), kept aside until the end of the call
  HighsModel model;
  // The options saved by kSaveOptions
  std::unique_ptr<HighsOptions> saved_options;
  // An exception thrown by a step, rethrown once Rust has returned; no
  // step is made after it
  std::exception_ptr pending;


  static double op(void* ctx, int which, const RsIisArgs* a) {
    HighsIisRust& r = *static_cast<HighsIisRust*>(ctx);
    if (r.pending) return -1;
    try {
      return r.step(IisOp(which), *a);
    } catch (...) {
      r.pending = std::current_exception();
      return -1;
    }
  }

  static void lpView(void* ctx, RsLp* out, size_t* num_col_names,
                     size_t* num_row_names) {
    const HighsLp& lp = static_cast<HighsIisRust*>(ctx)->h.model_r().lp_;
    *out = rsLp(lp);
    *num_col_names = lp.col_names_.size();
    *num_row_names = lp.row_names_.size();
  }

  static RsIisStr name(void* ctx, bool is_col, HighsInt i) {
    const HighsLp& lp = static_cast<HighsIisRust*>(ctx)->h.model_r().lp_;
    const std::string& s = i < 0      ? lp.model_name_
                           : is_col ? lp.col_names_[i]
                                    : lp.row_names_[i];
    return {s.data(), s.size()};
  }

  static RsMut<double> colValue(void* ctx) {
    return rsMut(static_cast<HighsIisRust*>(ctx)->h.solution_r().col_value);
  }

  RsIisHost host() {
    return {rsLog(h.options_.log_options), this, op, lpView, name, colValue};
  }

  void rethrow() {
    if (pending) std::rethrow_exception(pending);
  }

  // An LP from Rust's arrays, with the incumbent's names of its columns
  // and rows
  void buildLp(const RsIisLp& a, HighsLp& lp) const {
    const HighsLp& from = h.model_r().lp_;
    lp.clear();
    lp.num_col_ = a.num_col;
    lp.num_row_ = a.num_row;
    assignFrom(lp.col_cost_, a.col_cost);
    assignFrom(lp.col_lower_, a.col_lower);
    assignFrom(lp.col_upper_, a.col_upper);
    assignFrom(lp.row_lower_, a.row_lower);
    assignFrom(lp.row_upper_, a.row_upper);
    lp.a_matrix_.format_ = MatrixFormat(a.format);
    lp.a_matrix_.num_col_ = a.num_col;
    lp.a_matrix_.num_row_ = a.num_row;
    assignFrom(lp.a_matrix_.start_, a.start);
    assignFrom(lp.a_matrix_.index_, a.index);
    assignFrom(lp.a_matrix_.value_, a.value);
    if (from.col_names_.size())
      for (size_t k = 0; k < a.col_map.len; k++)
        lp.col_names_.push_back(from.col_names_[a.col_map.ptr[k]]);
    if (from.row_names_.size())
      for (size_t k = 0; k < a.row_map.len; k++)
        lp.row_names_.push_back(from.row_names_[a.row_map.ptr[k]]);
    if (a.model_name) lp.model_name_ = from.model_name_ + "_IIS";
  }

  static double st(HighsStatus s) { return int(s); }

  double step(IisOp which, const RsIisArgs& a) {
    // The incumbent's steps (the IIS search's LP solvers are Rust's
    // LpHandles: iis.rs)
    Highs& x = h;
    HighsOptions& options = h.options_;
    const double* d0 = static_cast<const double*>(a.p[0]);
    const double* d1 = static_cast<const double*>(a.p[1]);
    const double* d2 = static_cast<const double*>(a.p[2]);
    switch (which) {
      case IisOp::kLoadIis: {
        HighsIis& iis = h.iis_;
        RsIisState& s = *static_cast<RsIisState*>(const_cast<void*>(a.p[0]));
        s = {iis.valid_,
             iis.status_,
             iis.strategy_,
             rsMut(iis.col_index_),
             rsMut(iis.row_index_),
             rsMut(iis.col_bound_),
             rsMut(iis.row_bound_),
             rsMut(iis.col_status_),
             rsMut(iis.row_status_),
             iis.info_};
        model = iis.model_;
        return 0;
      }
      case IisOp::kStoreIis: {
        HighsIis& iis = h.iis_;
        const RsIisState& s = *static_cast<const RsIisState*>(a.p[0]);
        iis.valid_ = s.valid;
        iis.status_ = s.status;
        iis.strategy_ = s.strategy;
        assignFrom(iis.col_index_, s.col_index);
        assignFrom(iis.row_index_, s.row_index);
        assignFrom(iis.col_bound_, s.col_bound);
        assignFrom(iis.row_bound_, s.row_bound);
        assignFrom(iis.col_status_, s.col_status);
        assignFrom(iis.row_status_, s.row_status);
        iis.info_ = s.info;
        iis.model_ = std::move(model);
        return 0;
      }
      case IisOp::kClearIisModel:
        model.clear();
        return 0;
      case IisOp::kSetIisLp:
        buildLp(*static_cast<const RsIisLp*>(a.p[0]), model.lp_);
        return 0;
      case IisOp::kSetOptionBool:
        return st(x.setOptionValue(str(a.s), bool(a.i)));
      case IisOp::kSetOptionInt:
        return st(x.setOptionValue(str(a.s), a.i));
      case IisOp::kSetOptionDouble:
        return st(x.setOptionValue(str(a.s), a.x));
      case IisOp::kSetOptionString:
        return st(x.setOptionValue(
            str(a.s), str(*static_cast<const RsIisStr*>(a.p[0]))));
      case IisOp::kChangeColBounds:
        return st(x.changeColBounds(a.i, a.x, a.y));
      case IisOp::kChangeRowBounds:
        return st(x.changeRowBounds(a.i, a.x, a.y));
      case IisOp::kChangeColsCost:
        return st(x.changeColsCost(a.i, a.j, d0));
      case IisOp::kOptimizeModel:
        return st(x.optimizeModel());
      case IisOp::kRunTime:
        return x.getRunTime();
      case IisOp::kSimplexIterations:
        return x.getInfo().simplex_iteration_count;
      case IisOp::kModelStatus:
        return int(x.getModelStatus());
      case IisOp::kZeroAllClocks:
        h.zeroAllClocks();
        return 0;
      case IisOp::kCallbackActive:
        return h.callback_.active[a.i] ? 1 : 0;
      case IisOp::kStopCallback:
        return st(h.stopCallback(int(a.i)));
      case IisOp::kStartCallback:
        return st(h.startCallback(int(a.i)));
      case IisOp::kSaveOptions:
        saved_options.reset(new HighsOptions(options));
        return 0;
      case IisOp::kRestoreOptions:
        h.options_ = *saved_options;
        h.options_cpp_newer_ = true;
        return 0;
      case IisOp::kEnsureColwise:
        if (!h.model_r().lp_.a_matrix_.isColwise())
          h.model_w().lp_.a_matrix_.ensureColwise();
        return 0;
      case IisOp::kGetOption:
        switch (a.i) {
          case 0:
            return options.iis_strategy;
          case 1:
            return options.primal_feasibility_tolerance;
          case 2:
            return options.iis_time_limit;
          case 3:
            return *options.log_options.output_flag ? 1 : 0;
          default:
            return options.log_dev_level;
        }
      case IisOp::kSetOutputFlag:
        options.output_flag = a.i != 0;
        h.options_cpp_newer_ = true;
        return 0;
      case IisOp::kInvalidateSolverData:
        h.invalidateSolverData();
        return 0;
      case IisOp::kPassModelName:
        return st(h.passModelName(str(a.s)));
      case IisOp::kChangeColsIntegrality:
        return st(h.changeColsIntegrality(
            a.i, a.j, static_cast<const HighsVarType*>(a.p[0])));
      case IisOp::kChangeColsBounds:
        return st(h.changeColsBounds(a.i, a.j, d0, d1));
      case IisOp::kAddCols:
        return st(h.addCols(a.i, d0, d1, d2, a.j,
                            static_cast<const HighsInt*>(a.p[3]),
                            static_cast<const HighsInt*>(a.p[4]),
                            static_cast<const double*>(a.p[5])));
      case IisOp::kAddRows:
        return st(h.addRows(a.i, d0, d1, a.j,
                            static_cast<const HighsInt*>(a.p[2]),
                            static_cast<const HighsInt*>(a.p[3]),
                            static_cast<const double*>(a.p[4])));
      case IisOp::kPassColName:
        return st(h.passColName(a.i, str(a.s)));
      case IisOp::kPassRowName:
        return st(h.passRowName(a.i, str(a.s)));
      case IisOp::kDeleteRows:
        return st(h.deleteRows(a.i, a.j));
      case IisOp::kDeleteCols:
        return st(h.deleteCols(a.i, a.j));
      case IisOp::kBasisInvalid:
        h.basis_w().valid = false;
        return 0;
      case IisOp::kElasticSolution:
        // Deleting rows and columns invalidates the solution, but the
        // primal values are right: recompute the row activities
        h.model_r().lp_.a_matrix_.productQuad(h.solution_w().row_value,
                                           h.solution_r().col_value);
        h.solution_w().value_valid = true;
        h.info_w().objective_function_value = a.x;
        getKktFailures(options, h.model_r(), h.solution_w(), h.basis_w(), h.info_w());
        h.info_w().valid = true;
        return 0;
      case IisOp::kSetModelStatus:
        h.model_status_w() = HighsModelStatus(a.i);
        return 0;
      case IisOp::kObjectiveValue:
        return h.info_r().objective_function_value;
      case IisOp::kEngine:
        h.optionsToRust();
        *static_cast<highs_rs::LpHandle**>(const_cast<void*>(a.p[0])) =
            h.ekk_instance_.p;
        return 0;
      case IisOp::kCallbacksToPropagate: {
        const HighsCallback& callback = h.callback_;
        return options.log_options.user_log_callback ||
                       callback.active[kCallbackLogging] ||
                       callback.active[kCallbackSimplexInterrupt]
                   ? 1
                   : 0;
      }
      default:
        // The IIS search's LP solvers' steps are Rust's (iis.rs)
        fprintf(stderr, "HighsIisRust: op %d is not the incumbent's\n",
                int(which));
        std::abort();
    }
    return 0;
  }
};

HighsStatus Highs::getIisInterface() {
  HighsIisRust r{*this, {}, {}, {}};
  const RsIisHost host = r.host();
  const int status = highs_rs_get_iis(&host);
  r.rethrow();
  return HighsStatus(status);
}

HighsStatus Highs::elasticityFilter(
    const double global_lower_penalty, const double global_upper_penalty,
    const double global_rhs_penalty, const double* local_lower_penalty,
    const double* local_upper_penalty, const double* local_rhs_penalty,
    const bool get_iis) {
  HighsIisRust r{*this, {}, {}, {}};
  const RsIisHost host = r.host();
  const int status = highs_rs_elasticity_filter(
      &host, global_lower_penalty, global_upper_penalty, global_rhs_penalty,
      local_lower_penalty, local_upper_penalty, local_rhs_penalty, get_iis);
  r.rethrow();
  return HighsStatus(status);
}

#endif
