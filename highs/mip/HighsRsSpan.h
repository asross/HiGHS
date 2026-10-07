/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#ifndef MIP_HIGHSRSSPAN_H_
#define MIP_HIGHSRSSPAN_H_

#include <cassert>
#include <cstddef>

/// A read-only view of a vector of a Rust-owned object (HIGHS_RUST),
/// valid until the object changes
template <typename T>
struct HighsRsSpan {
  const T* p;
  size_t n;
  const T& operator[](size_t i) const {
    assert(i < n);
    return p[i];
  }
  size_t size() const { return n; }
  const T* data() const { return p; }
  const T* begin() const { return p; }
  const T* end() const { return p + n; }
};

#endif
