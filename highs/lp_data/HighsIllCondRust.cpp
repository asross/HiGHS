/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file lp_data/HighsIllCondRust.cpp
 * @brief Highs::computeIllConditioning done by Rust
 * (rust/src/lp_data/ill_cond.rs): the views, the solve of the analysis LP
 * and the records
 */
#include "lp_data/HighsRust.h"

#ifdef HIGHS_RUST
#include "Highs.h"

namespace {
struct RsIllLp {
  HighsInt num_col, num_row;
  RsMut<double> col_cost, col_lower, col_upper, row_lower, row_upper;
  RsMut<HighsInt> start, index;
  RsMut<double> value;
  RsMut<const char*> names;
  int run_status, model_status;
  double objective;
  RsMut<double> col_value;
  double last_row_value;
};

struct RsIllHost {
  void* ctx;
  void (*op)(void* ctx, int code, void* p, HighsInt index, double x);
  RsLog log;
  RsLp lp;
  RsMut<uint8_t> col_status, row_status;
  RsMut<const char*> col_names, row_names;
  bool constraint;
  HighsInt method;
  double bound;
};

std::vector<const char*> illCondNames(const std::vector<std::string>& names) {
  std::vector<const char*> p;
  for (const std::string& name : names) p.push_back(name.c_str());
  return p;
}

RsMut<uint8_t> illCondStatus(const std::vector<HighsBasisStatus>& s) {
  return {reinterpret_cast<uint8_t*>(const_cast<HighsBasisStatus*>(s.data())),
          s.size()};
}

template <typename T>
std::vector<T> illCondVec(const RsMut<T>& v) {
  return std::vector<T>(v.ptr, v.ptr + v.len);
}
}  // namespace

extern "C" int highs_rs_compute_ill_conditioning(const RsIllHost* host);

HighsStatus Highs::computeIllConditioning(
    HighsIllConditioning& ill_conditioning, const bool constraint,
    const HighsInt method, const double ill_conditioning_bound) {
  ill_conditioning.clear();
  HighsLp& incumbent_lp = this->model_w().lp_;
  incumbent_lp.a_matrix_.ensureColwise();
  const std::vector<const char*> col_names =
      illCondNames(incumbent_lp.col_names_);
  const std::vector<const char*> row_names =
      illCondNames(incumbent_lp.row_names_);
  RsIllHost h;
  h.ctx = &ill_conditioning;
  // Solve the analysis LP (code 0) or store a record (code 1)
  h.op = [](void* ctx, int code, void* p, HighsInt index, double x) {
    if (code == 1) {
      HighsIllConditioningRecord record;
      record.index = index;
      record.multiplier = x;
      static_cast<HighsIllConditioning*>(ctx)->record.push_back(record);
      return;
    }
    RsIllLp& a = *static_cast<RsIllLp*>(p);
    Highs conditioning;
    conditioning.setOptionValue("output_flag", false);
    HighsLp& lp = conditioning.model_w().lp_;
    lp.num_col_ = a.num_col;
    lp.num_row_ = a.num_row;
    lp.col_cost_ = illCondVec(a.col_cost);
    lp.col_lower_ = illCondVec(a.col_lower);
    lp.col_upper_ = illCondVec(a.col_upper);
    lp.row_lower_ = illCondVec(a.row_lower);
    lp.row_upper_ = illCondVec(a.row_upper);
    lp.a_matrix_.start_ = illCondVec(a.start);
    lp.a_matrix_.index_ = illCondVec(a.index);
    lp.a_matrix_.value_ = illCondVec(a.value);
    lp.a_matrix_.num_col_ = a.num_col;
    lp.a_matrix_.num_row_ = a.num_row;
    for (size_t k = 0; k < a.names.len; k++)
      lp.col_names_.push_back(a.names.ptr[k]);
    a.run_status = int(conditioning.run());
    a.model_status = int(conditioning.getModelStatus());
    a.objective = conditioning.getInfo().objective_function_value;
    const HighsSolution& solution = conditioning.solution_;
    for (size_t k = 0; k < a.col_value.len && k < solution.col_value.size();
         k++)
      a.col_value.ptr[k] = solution.col_value[k];
    if (solution.row_value.size() == size_t(conditioning.getNumRow()) &&
        conditioning.getNumRow() > 0)
      a.last_row_value = solution.row_value[conditioning.getNumRow() - 1];
  };
  h.log = rsLog(options_.log_options);
  h.lp = rsLp(incumbent_lp);
  h.col_status = illCondStatus(basis_.col_status);
  h.row_status = illCondStatus(basis_.row_status);
  h.col_names = {const_cast<const char**>(col_names.data()), col_names.size()};
  h.row_names = {const_cast<const char**>(row_names.data()), row_names.size()};
  h.constraint = constraint;
  h.method = method;
  h.bound = ill_conditioning_bound;
  return HighsStatus(highs_rs_compute_ill_conditioning(&h));
}
#endif
