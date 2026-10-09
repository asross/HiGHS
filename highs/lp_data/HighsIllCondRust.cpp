/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file lp_data/HighsIllCondRust.cpp
 * @brief Highs::computeIllConditioning done by Rust
 * (rust/src/lp_data/ill_cond.rs): the views and the records
 */
#include "lp_data/HighsRust.h"

#ifdef HIGHS_RUST
#include "Highs.h"

namespace {
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

}  // namespace

extern "C" int highs_rs_compute_ill_conditioning(const RsIllHost* host);

HighsStatus Highs::computeIllConditioning(
    HighsIllConditioning& ill_conditioning, const bool constraint,
    const HighsInt method, const double ill_conditioning_bound) {
  ill_conditioning.clear();
  if (!this->model_r().lp_.a_matrix_.isColwise())
    this->model_w().lp_.a_matrix_.ensureColwise();
  const HighsLp& incumbent_lp = this->model_r().lp_;
  const std::vector<const char*> col_names =
      illCondNames(incumbent_lp.col_names_);
  const std::vector<const char*> row_names =
      illCondNames(incumbent_lp.row_names_);
  RsIllHost h;
  h.ctx = &ill_conditioning;
  // Store a record
  h.op = [](void* ctx, int code, void* p, HighsInt index, double x) {
    HighsIllConditioningRecord record;
    record.index = index;
    record.multiplier = x;
    static_cast<HighsIllConditioning*>(ctx)->record.push_back(record);
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
