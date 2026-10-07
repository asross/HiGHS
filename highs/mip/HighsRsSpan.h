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
#include <vector>

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

/// A vector owned by a Rust object (HIGHS_RUST), in place: the
/// begin/end/capacity layout of Rust's StdVec (rust/src/mip/domain.rs).
/// C++ reads and writes its elements and may shrink it; only Rust grows
/// it, frees it or makes it
template <typename T>
class HighsRsArray {
  T* begin_;
  T* end_;
  T* cap_;

 public:
  HighsRsArray() = delete;
  HighsRsArray(const HighsRsArray&) = delete;
  HighsRsArray& operator=(const HighsRsArray&) = delete;
  size_t size() const { return end_ - begin_; }
  bool empty() const { return end_ == begin_; }
  T* data() { return begin_; }
  const T* data() const { return begin_; }
  T* begin() { return begin_; }
  T* end() { return end_; }
  const T* begin() const { return begin_; }
  const T* end() const { return end_; }
  T& operator[](size_t i) {
    assert(i < size());
    return begin_[i];
  }
  const T& operator[](size_t i) const {
    assert(i < size());
    return begin_[i];
  }
  const T& back() const {
    assert(!empty());
    return end_[-1];
  }
  void clear() { end_ = begin_; }
  // shrinks to n elements
  void resize(size_t n) {
    assert(n <= size());
    end_ = begin_ + n;
  }
  // erases the tail [first, end())
  void erase(T* first, T* last) {
    assert(last == end_);
    (void)last;
    end_ = first;
  }
  operator std::vector<T>() const { return std::vector<T>(begin_, end_); }
};

#endif
