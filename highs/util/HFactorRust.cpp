/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file util/HFactorRust.cpp
 * @brief HFactor delegating to the Rust port (rust/src/factor.rs)
 */
#include "util/HFactor.h"

#ifdef HIGHS_RUST

#include <cassert>

static_assert(sizeof(HighsInt) == 4, "the Rust factor takes 32-bit ints");

namespace {
// Mirrors of the #[repr(C)] structs in rust/src/ffi.rs
struct RsHVec {
  int size;
  int count;
  int* index;
  int n_index;
  double* array;
  int n_array;
  char* cwork;
  int n_cwork;
  int* iwork;
  int n_iwork;
  double synthetic_tick;
  unsigned char pack_flag;
  int pack_count;
  int* pack_index;
  int n_pack_index;
  double* pack_value;
  int n_pack_value;
};

struct RsAMatrix {
  int num_col;
  const int* start;
  const int* index;
  const double* value;
};

struct RsInfo {
  double build_synthetic_tick;
  double refactor_build_synthetic_tick;
  int rank_deficiency;
  int basis_matrix_num_el;
  int invert_num_el;
  int kernel_dim;
  int kernel_num_el;
  int num_row;
};

// Vector identifiers: see ivec and dvec in rust/src/ffi.rs
enum RsIvec {
  kRowWithNoPivot = 0,
  kColWithNoPivot,
  kVarWithNoPivot,
  kRefactorPivotRow,
  kRefactorPivotVar,
  kLPivotIndex,
  kLPivotLookup,
  kLStart,
  kLIndex,
  kLrStart,
  kLrIndex,
  kUPivotLookup,
  kUPivotIndex,
  kUStart,
  kULastP,
  kUIndex,
  kUrStart,
  kUrLastp,
  kUrSpace,
  kUrIndex,
  kPfStart,
  kPfIndex,
  kPfPivotIndex
};
enum RsDvec {
  kLValue = 0,
  kLrValue,
  kUPivotValue,
  kUValue,
  kUrValue,
  kPfValue,
  kPfPivotValue
};
}  // namespace

extern "C" {
void* highs_rs_factor_new();
void highs_rs_factor_free(void* p);
void highs_rs_factor_setup(void* p, int num_col, int num_row, int num_basic,
                           const int* a_start, int update_method);
int highs_rs_factor_build(void* p, double pivot_threshold,
                          double pivot_tolerance, double time_limit,
                          const RsAMatrix* a, int* basic_index, int n_basic);
void highs_rs_factor_refactor_clear(void* p);
void highs_rs_factor_put_invert(void* p);
void highs_rs_factor_get_invert(void* p);
bool highs_rs_factor_refactor_use(const void* p);
void highs_rs_factor_refactor_set(void* p, bool use, const int* pivot_row,
                                  const int* pivot_var,
                                  const int8_t* pivot_type, int n,
                                  double build_synthetic_tick);
void highs_rs_factor_ftran(const void* p, RsHVec* v, double expected_density);
void highs_rs_factor_btran(const void* p, RsHVec* v, double expected_density);
void highs_rs_factor_update(void* p, const RsHVec* aq, const RsHVec* ep, int n,
                            const int* i_row, int* hint, const RsAMatrix* a,
                            const int* basic_index, int n_basic);
void highs_rs_factor_add_rows(void* p, int num_col, const int* basic_index,
                              int n_basic, const int* ar_start,
                              const int* ar_index, const double* ar_value,
                              int num_new_row);
void highs_rs_factor_info(const void* p, RsInfo* out);
const int* highs_rs_factor_ivec(void* p, int which, int* len);
const double* highs_rs_factor_dvec(void* p, int which, int* len);
const int8_t* highs_rs_factor_refactor_type(const void* p, int* len);
void highs_rs_factor_set_ivec(void* p, int which, const int* data, int len);
void highs_rs_factor_set_dvec(void* p, int which, const double* data,
                              int len);
void highs_rs_factor_check_indices(const void* p);
}

namespace {
RsHVec rsHVec(HVector& v) {
  return {v.size,
          v.count,
          v.index.data(),
          (int)v.index.size(),
          v.array.data(),
          (int)v.array.size(),
          v.cwork.data(),
          (int)v.cwork.size(),
          v.iwork.data(),
          (int)v.iwork.size(),
          v.synthetic_tick,
          (unsigned char)v.packFlag,
          v.packCount,
          v.packIndex.data(),
          (int)v.packIndex.size(),
          v.packValue.data(),
          (int)v.packValue.size()};
}

void storeHVec(const RsHVec& r, HVector& v) {
  v.count = r.count;
  v.synthetic_tick = r.synthetic_tick;
  v.packFlag = r.pack_flag != 0;
  v.packCount = r.pack_count;
}

std::vector<HighsInt> getIvec(void* p, RsIvec which) {
  int len;
  const int* data = highs_rs_factor_ivec(p, which, &len);
  return std::vector<HighsInt>(data, data + len);
}

std::vector<double> getDvec(void* p, RsDvec which) {
  int len;
  const double* data = highs_rs_factor_dvec(p, which, &len);
  return std::vector<double>(data, data + len);
}

void setIvec(void* p, RsIvec which, const std::vector<HighsInt>& v) {
  highs_rs_factor_set_ivec(p, which, v.data(), v.size());
}

void setDvec(void* p, RsDvec which, const std::vector<double>& v) {
  highs_rs_factor_set_dvec(p, which, v.data(), v.size());
}
}  // namespace

HFactor::RustFactor::RustFactor() : p(highs_rs_factor_new()) {}
HFactor::RustFactor::RustFactor(RustFactor&& other) noexcept : RustFactor() {
  std::swap(p, other.p);
}
HFactor::RustFactor& HFactor::RustFactor::operator=(
    RustFactor&& other) noexcept {
  std::swap(p, other.p);
  return *this;
}
HFactor::RustFactor::~RustFactor() { highs_rs_factor_free(p); }

void HFactor::setupRust() {
  highs_rs_factor_setup(rs_.p, num_col, num_row, num_basic, a_start,
                        update_method);
}

HighsInt HFactor::build(HighsTimerClock* /*factor_timer_clock_pointer*/) {
  // Ensure that the A matrix is valid for factorization
  assert(this->a_matrix_valid);
  const RsAMatrix a{num_col, a_start, a_index, a_value};
  // The refactorization information is held by the Rust factor
  const HighsInt build_return =
      highs_rs_factor_build(rs_.p, pivot_threshold, pivot_tolerance,
                            time_limit_, &a, basic_index, num_basic);
  pullRustBuildInfo();
  if (build_return == kBuildKernelReturnTimeout) return build_return;
  if (rank_deficiency && num_basic == num_row)
    highsLogDev(log_options, HighsLogType::kWarning,
                "Rank deficiency of %" HIGHSINT_FORMAT
                " identified in basis matrix\n",
                rank_deficiency);
  return build_return;
}

void HFactor::pullRustBuildInfo() {
  RsInfo info;
  highs_rs_factor_info(rs_.p, &info);
  build_synthetic_tick = info.build_synthetic_tick;
  rank_deficiency = info.rank_deficiency;
  basis_matrix_num_el = info.basis_matrix_num_el;
  invert_num_el = info.invert_num_el;
  kernel_dim = info.kernel_dim;
  kernel_num_el = info.kernel_num_el;
  row_with_no_pivot = getIvec(rs_.p, kRowWithNoPivot);
  col_with_no_pivot = getIvec(rs_.p, kColWithNoPivot);
  var_with_no_pivot = getIvec(rs_.p, kVarWithNoPivot);
}

RefactorInfo HFactor::getRefactorInfo() const {
  RefactorInfo refactor_info;
  refactor_info.use = highs_rs_factor_refactor_use(rs_.p);
  refactor_info.pivot_row = getIvec(rs_.p, kRefactorPivotRow);
  refactor_info.pivot_var = getIvec(rs_.p, kRefactorPivotVar);
  int len;
  const int8_t* type = highs_rs_factor_refactor_type(rs_.p, &len);
  refactor_info.pivot_type.assign(type, type + len);
  RsInfo info;
  highs_rs_factor_info(rs_.p, &info);
  refactor_info.build_synthetic_tick = info.refactor_build_synthetic_tick;
  return refactor_info;
}

#ifndef HIGHS_RUST
void HFactor::setRefactorInfo(const RefactorInfo& refactor_info) {
  const int n = std::min({refactor_info.pivot_row.size(),
                          refactor_info.pivot_var.size(),
                          refactor_info.pivot_type.size()});
  highs_rs_factor_refactor_set(
      rs_.p, refactor_info.use, refactor_info.pivot_row.data(),
      refactor_info.pivot_var.data(), refactor_info.pivot_type.data(), n,
      refactor_info.build_synthetic_tick);
}
#endif

void HFactor::clearRefactorInfo() { highs_rs_factor_refactor_clear(rs_.p); }

void HFactor::saveInvert() { highs_rs_factor_put_invert(rs_.p); }

void HFactor::restoreInvert() { highs_rs_factor_get_invert(rs_.p); }

void HFactor::ftranCall(HVector& vector, const double expected_density,
                        HighsTimerClock* /*factor_timer_clock_pointer*/) const {
  RsHVec v = rsHVec(vector);
  highs_rs_factor_ftran(rs_.p, &v, expected_density);
  storeHVec(v, vector);
}

void HFactor::btranCall(HVector& vector, const double expected_density,
                        HighsTimerClock* /*factor_timer_clock_pointer*/) const {
  RsHVec v = rsHVec(vector);
  highs_rs_factor_btran(rs_.p, &v, expected_density);
  storeHVec(v, vector);
}

void HFactor::update(HVector* aq, HVector* ep, HighsInt* iRow, HighsInt* hint) {
  // Updating implies a change of basis. Since the refactorizaion info
  // no longer corresponds to the current basis, it must be
  // invalidated
  clearRefactorInfo();
  // Only APF uses the A matrix
  const RsAMatrix a{num_col, a_start, a_index, a_value};
  const RsAMatrix* use_a = update_method == kUpdateMethodApf ? &a : nullptr;
  if (!aq->next) {
    RsHVec aq_work = rsHVec(*aq);
    RsHVec ep_work = rsHVec(*ep);
    highs_rs_factor_update(rs_.p, &aq_work, &ep_work, 1, iRow, hint, use_a,
                           basic_index, num_basic);
    return;
  }
  // Linked vectors mean a multiple (CFT) update
  std::vector<RsHVec> aq_work, ep_work;
  for (; aq != nullptr; aq = aq->next, ep = ep->next) {
    aq_work.push_back(rsHVec(*aq));
    ep_work.push_back(rsHVec(*ep));
  }
  highs_rs_factor_update(rs_.p, aq_work.data(), ep_work.data(),
                         aq_work.size(), iRow, hint, use_a, basic_index,
                         num_basic);
}

#ifndef HIGHS_RUST
void HFactor::addRows(const HighsSparseMatrix* ar_matrix) {
  invalidAMatrixAction();
  assert(kExtendInvertWhenAddingRows);
  HighsInt num_new_row = ar_matrix->num_row_;
  highs_rs_factor_add_rows(rs_.p, num_col, basic_index, num_row + num_new_row,
                           ar_matrix->start_.data(), ar_matrix->index_.data(),
                           ar_matrix->value_.data(), num_new_row);
  num_row += num_new_row;
}
#endif

InvertibleRepresentation HFactor::getInvert() const {
  void* p = rs_.p;
  InvertibleRepresentation invert;
  invert.l_pivot_index = getIvec(p, kLPivotIndex);
  invert.l_pivot_lookup = getIvec(p, kLPivotLookup);
  invert.l_start = getIvec(p, kLStart);
  invert.l_index = getIvec(p, kLIndex);
  invert.l_value = getDvec(p, kLValue);
  invert.lr_start = getIvec(p, kLrStart);
  invert.lr_index = getIvec(p, kLrIndex);
  invert.lr_value = getDvec(p, kLrValue);

  invert.u_pivot_lookup = getIvec(p, kUPivotLookup);
  invert.u_pivot_index = getIvec(p, kUPivotIndex);
  invert.u_pivot_value = getDvec(p, kUPivotValue);
  invert.u_start = getIvec(p, kUStart);
  invert.u_last_p = getIvec(p, kULastP);
  invert.u_index = getIvec(p, kUIndex);
  invert.u_value = getDvec(p, kUValue);

  invert.ur_start = getIvec(p, kUrStart);
  invert.ur_lastp = getIvec(p, kUrLastp);
  invert.ur_space = getIvec(p, kUrSpace);
  invert.ur_index = getIvec(p, kUrIndex);
  invert.ur_value = getDvec(p, kUrValue);
  invert.pf_start = getIvec(p, kPfStart);
  invert.pf_index = getIvec(p, kPfIndex);
  invert.pf_value = getDvec(p, kPfValue);
  invert.pf_pivot_index = getIvec(p, kPfPivotIndex);
  invert.pf_pivot_value = getDvec(p, kPfPivotValue);
  return invert;
}

void HFactor::setInvert(const InvertibleRepresentation& invert) {
  void* p = rs_.p;
  setIvec(p, kLPivotIndex, invert.l_pivot_index);
  setIvec(p, kLPivotLookup, invert.l_pivot_lookup);
  setIvec(p, kLStart, invert.l_start);
  setIvec(p, kLIndex, invert.l_index);
  setDvec(p, kLValue, invert.l_value);
  setIvec(p, kLrStart, invert.lr_start);
  setIvec(p, kLrIndex, invert.lr_index);
  setDvec(p, kLrValue, invert.lr_value);

  setIvec(p, kUPivotLookup, invert.u_pivot_lookup);
  setIvec(p, kUPivotIndex, invert.u_pivot_index);
  setDvec(p, kUPivotValue, invert.u_pivot_value);
  setIvec(p, kUStart, invert.u_start);
  setIvec(p, kULastP, invert.u_last_p);
  setIvec(p, kUIndex, invert.u_index);
  setDvec(p, kUValue, invert.u_value);

  setIvec(p, kUrStart, invert.ur_start);
  setIvec(p, kUrLastp, invert.ur_lastp);
  setIvec(p, kUrSpace, invert.ur_space);
  setIvec(p, kUrIndex, invert.ur_index);
  setDvec(p, kUrValue, invert.ur_value);
  setIvec(p, kPfStart, invert.pf_start);
  setIvec(p, kPfIndex, invert.pf_index);
  setDvec(p, kPfValue, invert.pf_value);
  setIvec(p, kPfPivotIndex, invert.pf_pivot_index);
  setDvec(p, kPfPivotValue, invert.pf_pivot_value);
  highs_rs_factor_check_indices(p);
}

#ifndef HIGHS_RUST
// The factor's data is in Rust: no reports
void HFactor::reportLu(const HighsInt, const bool) const {}
void HFactor::reportAsm() const {}
#endif

#endif  // HIGHS_RUST
