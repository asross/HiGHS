/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#ifndef HIGHS_DYNAMIC_ROW_MATRIX_H_
#define HIGHS_DYNAMIC_ROW_MATRIX_H_

#include <cstdint>
#include <set>
#include <utility>
#include <vector>

#include "HConfig.h"
#include "util/HighsInt.h"

#ifdef HIGHS_RUST
namespace highs_rs {
struct CutProp;
// Mirror of CMatrixView (rust/src/mip/cutpool.rs): a Rust cut pool's
// matrix and right-hand sides
struct CutPoolView {
  const std::pair<HighsInt, HighsInt>* ar_range;
  HighsInt num_rows;
  HighsInt num_del_rows;
  const HighsInt* ar_index;
  const double* ar_value;
  HighsInt num_nz;
  const HighsInt* ar_rowindex;
  const HighsInt* next_pos;
  const HighsInt* next_neg;
  const HighsInt* head_pos;
  const HighsInt* head_neg;
  HighsInt num_cols;
  const uint8_t* cols_linked;
  const double* rhs;
  HighsInt num_rhs;
};
}  // namespace highs_rs

// The matrix of a Rust cut pool (rust/src/mip/cutpool.rs), read through a
// view valid until the pool changes
class HighsDynamicRowMatrix {
  highs_rs::CutPoolView v_;

 public:
  explicit HighsDynamicRowMatrix(const highs_rs::CutPoolView& v) : v_(v) {}

  const highs_rs::CutPoolView& view() const { return v_; }

  bool columnsLinked(HighsInt rowindex) const {
    return v_.cols_linked[rowindex] != 0;
  }

  std::size_t nonzeroCapacity() const { return v_.num_nz; }

  template <typename Func>
  void forEachPositiveColumnEntry(HighsInt col, Func&& f) const {
    HighsInt iter = v_.head_pos[col];
    while (iter != -1) {
      if (!f(v_.ar_rowindex[iter], v_.ar_value[iter])) break;
      iter = v_.next_pos[iter];
    }
  }

  template <typename Func>
  void forEachNegativeColumnEntry(HighsInt col, Func&& f) const {
    HighsInt iter = v_.head_neg[col];
    while (iter != -1) {
      if (!f(v_.ar_rowindex[iter], v_.ar_value[iter])) break;
      iter = v_.next_neg[iter];
    }
  }

  HighsInt getNumRows() const { return v_.num_rows; }

  HighsInt getNumDelRows() const { return v_.num_del_rows; }

  HighsInt getRowStart(HighsInt row) const { return v_.ar_range[row].first; }

  HighsInt getRowEnd(HighsInt row) const { return v_.ar_range[row].second; }

  const HighsInt* getARindex() const { return v_.ar_index; }

  const double* getARvalue() const { return v_.ar_value; }

  // fills the matrix part of a highs_rs::CutProp (mip/HighsDomainRust.h)
  void rustView(highs_rs::CutProp& v) const;
};
#else

class HighsDynamicRowMatrix {
 private:
  /// vector of index ranges in the index and value arrays of AR for each row
  std::vector<std::pair<HighsInt, HighsInt>> ARrange_;

  /// column indices for each nonzero in AR
  std::vector<HighsInt> ARindex_;
  /// values for each nonzero in AR
  std::vector<double> ARvalue_;

  std::vector<HighsInt> ARrowindex_;
  std::vector<HighsInt> AnextPos_;
  std::vector<HighsInt> AprevPos_;
  std::vector<HighsInt> AnextNeg_;
  std::vector<HighsInt> AprevNeg_;

  /// vector of pointers to the head/tail of the nonzero block list for each
  /// column
  std::vector<HighsInt> AheadPos_;
  std::vector<HighsInt> AheadNeg_;

  std::vector<uint8_t> colsLinked;

  /// vector of column sizes

  /// keep an ordered set of free spaces in the row arrays so that they can be
  /// reused efficiently
  std::set<std::pair<HighsInt, HighsInt>> freespaces_;

  /// vector of deleted rows so that their indices can be reused
  std::vector<HighsInt> deletedrows_;

 public:
  HighsDynamicRowMatrix(HighsInt ncols);

  bool columnsLinked(HighsInt rowindex) const {
    return (colsLinked[rowindex] != 0);
  }

  void unlinkColumns(HighsInt rowindex);

  /// adds a row to the matrix with the given values and returns its index
  HighsInt addRow(HighsInt* Rindex, double* Rvalue, HighsInt Rlen,
                  bool linkCols = true);

  /// removes the row with the given index from the matrix, afterwards the index
  /// can be reused for new rows
  void removeRow(HighsInt rowindex);

  std::size_t nonzeroCapacity() const { return ARvalue_.size(); }

  /// calls the given function object for each entry in the given column.
  /// The function object should accept the row index as first argument and
  /// the nonzero value of the column in that row as the second argument.
  template <typename Func>
  void forEachPositiveColumnEntry(HighsInt col, Func&& f) const {
    HighsInt iter = AheadPos_[col];

    while (iter != -1) {
      if (!f(ARrowindex_[iter], ARvalue_[iter])) break;
      iter = AnextPos_[iter];
    }
  }

  template <typename Func>
  void forEachNegativeColumnEntry(HighsInt col, Func&& f) const {
    HighsInt iter = AheadNeg_[col];

    while (iter != -1) {
      if (!f(ARrowindex_[iter], ARvalue_[iter])) break;
      iter = AnextNeg_[iter];
    }
  }

  HighsInt getNumRows() const { return ARrange_.size(); }

  HighsInt getNumDelRows() const { return deletedrows_.size(); }

  HighsInt getRowStart(HighsInt row) const { return ARrange_[row].first; }

  HighsInt getRowEnd(HighsInt row) const { return ARrange_[row].second; }

  const HighsInt* getARindex() const { return ARindex_.data(); }

  const double* getARvalue() const { return ARvalue_.data(); }
};
#endif  // HIGHS_RUST

#endif
