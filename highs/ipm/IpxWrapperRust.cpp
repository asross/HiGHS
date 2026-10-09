/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file ipm/IpxWrapperRust.cpp
 * @brief fillInIpxData done by Rust (rust/src/lp_data/ipx_glue.rs), for
 * callCrossover; the LP solves' IPX glue is Rust (lp_run.rs)
 */
#include "lp_data/HighsRust.h"

#ifdef HIGHS_RUST
#include "ipm/IpxWrapper.h"

namespace {
struct RsIpxData {
  HighsInt num_col, num_row;
  double offset;
  RsMut<double> f[5];  // obj, col_lb, col_ub, ax, rhs
  RsMut<HighsInt> i[2];  // ap, ai
  RsMut<uint8_t> constraint_type;
};
}  // namespace

extern "C" {
void* highs_rs_fill_in_ipx_data(const RsLp* lp);
void highs_rs_ipx_data_get(void* d, RsIpxData* out);
void highs_rs_ipx_data_free(void* d);
}

void fillInIpxData(const HighsLp& lp, ipx::Int& num_col, ipx::Int& num_row,
                   double& offset, std::vector<double>& obj,
                   std::vector<double>& col_lb, std::vector<double>& col_ub,
                   std::vector<ipx::Int>& Ap, std::vector<ipx::Int>& Ai,
                   std::vector<double>& Ax, std::vector<double>& rhs,
                   std::vector<char>& constraint_type) {
  const RsLp v = rsLp(lp);
  void* d = highs_rs_fill_in_ipx_data(&v);
  RsIpxData r;
  highs_rs_ipx_data_get(d, &r);
  num_col = r.num_col;
  num_row = r.num_row;
  offset = r.offset;
  obj.assign(r.f[0].ptr, r.f[0].ptr + r.f[0].len);
  col_lb.assign(r.f[1].ptr, r.f[1].ptr + r.f[1].len);
  col_ub.assign(r.f[2].ptr, r.f[2].ptr + r.f[2].len);
  Ax.assign(r.f[3].ptr, r.f[3].ptr + r.f[3].len);
  rhs.assign(r.f[4].ptr, r.f[4].ptr + r.f[4].len);
  Ap.assign(r.i[0].ptr, r.i[0].ptr + r.i[0].len);
  Ai.assign(r.i[1].ptr, r.i[1].ptr + r.i[1].len);
  constraint_type.assign(r.constraint_type.ptr,
                         r.constraint_type.ptr + r.constraint_type.len);
  highs_rs_ipx_data_free(d);
}
#endif
