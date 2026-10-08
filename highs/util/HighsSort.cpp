/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file util/HighsSort.cpp
 * @brief Sorting routines for HiGHS
 */
#include "util/HighsSort.h"

#include <algorithm>
#include <cstddef>

using std::vector;

#ifdef HIGHS_RUST
// The sorts are Rust (rust/src/util/sort.rs); heap arrays are 1-based
extern "C" {
void highs_rs_add_to_decreasing_heap(HighsInt* n, HighsInt mx_n, double* heap_v,
                                     HighsInt* heap_ix, HighsInt len, double v,
                                     HighsInt ix);
void highs_rs_sort_decreasing_heap(HighsInt n, double* heap_v,
                                   HighsInt* heap_ix, HighsInt len);
void highs_rs_heap_int(HighsInt* heap_v, HighsInt* heap_i, HighsInt i,
                       HighsInt n, int what);
void highs_rs_heap_double(double* heap_v, HighsInt* heap_i, HighsInt i,
                          HighsInt n, int what);
bool highs_rs_increasing_set_ok_int(const HighsInt* set, HighsInt n,
                                    HighsInt lower, HighsInt upper,
                                    bool strict);
bool highs_rs_increasing_set_ok_double(const double* set, HighsInt n,
                                       double lower, double upper, bool strict);
void highs_rs_sort_set_data(HighsInt n, HighsInt* set, const void* const* data,
                            void* const* sorted, int num_data, int width);
}
static_assert(sizeof(HighsVarType) == 1, "HighsVarType is a byte");

void addToDecreasingHeap(HighsInt& n, HighsInt mx_n, vector<double>& heap_v,
                         vector<HighsInt>& heap_ix, const double v,
                         const HighsInt ix) {
  highs_rs_add_to_decreasing_heap(
      &n, mx_n, heap_v.data(), heap_ix.data(),
      HighsInt(std::min(heap_v.size(), heap_ix.size())), v, ix);
}

void sortDecreasingHeap(const HighsInt n, vector<double>& heap_v,
                        vector<HighsInt>& heap_ix) {
  highs_rs_sort_decreasing_heap(
      n, heap_v.data(), heap_ix.data(),
      HighsInt(std::min(heap_v.size(), heap_ix.size())));
}

// what: 0 maxheapsort, 1 buildMaxheap, 2 maxHeapsort, 3 maxHeapify
void maxheapsort(HighsInt* heap_v, HighsInt n) {
  highs_rs_heap_int(heap_v, nullptr, 0, n, 0);
}
void maxheapsort(HighsInt* heap_v, HighsInt* heap_i, HighsInt n) {
  highs_rs_heap_int(heap_v, heap_i, 0, n, 0);
}
void maxheapsort(double* heap_v, HighsInt* heap_i, HighsInt n) {
  highs_rs_heap_double(heap_v, heap_i, 0, n, 0);
}
void buildMaxheap(HighsInt* heap_v, HighsInt n) {
  highs_rs_heap_int(heap_v, nullptr, 0, n, 1);
}
void buildMaxheap(HighsInt* heap_v, HighsInt* heap_i, HighsInt n) {
  highs_rs_heap_int(heap_v, heap_i, 0, n, 1);
}
void buildMaxheap(double* heap_v, HighsInt* heap_i, HighsInt n) {
  highs_rs_heap_double(heap_v, heap_i, 0, n, 1);
}
void maxHeapsort(HighsInt* heap_v, HighsInt n) {
  highs_rs_heap_int(heap_v, nullptr, 0, n, 2);
}
void maxHeapsort(HighsInt* heap_v, HighsInt* heap_i, HighsInt n) {
  highs_rs_heap_int(heap_v, heap_i, 0, n, 2);
}
void maxHeapsort(double* heap_v, HighsInt* heap_i, HighsInt n) {
  highs_rs_heap_double(heap_v, heap_i, 0, n, 2);
}
void maxHeapify(HighsInt* heap_v, HighsInt i, HighsInt n) {
  highs_rs_heap_int(heap_v, nullptr, i, n, 3);
}
void maxHeapify(HighsInt* heap_v, HighsInt* heap_i, HighsInt i, HighsInt n) {
  highs_rs_heap_int(heap_v, heap_i, i, n, 3);
}
void maxHeapify(double* heap_v, HighsInt* heap_i, HighsInt i, HighsInt n) {
  highs_rs_heap_double(heap_v, heap_i, i, n, 3);
}

bool increasingSetOk(const vector<HighsInt>& set,
                     const HighsInt set_entry_lower,
                     const HighsInt set_entry_upper, bool strict) {
  return highs_rs_increasing_set_ok_int(set.data(), HighsInt(set.size()),
                                        set_entry_lower, set_entry_upper,
                                        strict);
}

bool increasingSetOk(const vector<double>& set, const double set_entry_lower,
                     const double set_entry_upper, bool strict) {
  return highs_rs_increasing_set_ok_double(set.data(), HighsInt(set.size()),
                                           set_entry_lower, set_entry_upper,
                                           strict);
}

void sortSetData(const HighsInt num_entries, vector<HighsInt>& set,
                 const double* data0, const double* data1, const double* data2,
                 double* sorted_data0, double* sorted_data1,
                 double* sorted_data2) {
  const void* data[3] = {data0, data1, data2};
  void* sorted[3] = {sorted_data0, sorted_data1, sorted_data2};
  highs_rs_sort_set_data(num_entries, set.data(), data, sorted, 3,
                         sizeof(double));
}

void sortSetData(const HighsInt num_entries, vector<HighsInt>& set,
                 const HighsVarType* data0, HighsVarType* sorted_data0) {
  const void* data[1] = {data0};
  void* sorted[1] = {sorted_data0};
  highs_rs_sort_set_data(num_entries, set.data(), data, sorted, 1,
                         sizeof(HighsVarType));
}
#else

void addToDecreasingHeap(HighsInt& n, HighsInt mx_n, vector<double>& heap_v,
                         vector<HighsInt>& heap_ix, const double v,
                         const HighsInt ix) {
  HighsInt cd_p, pa_p;
  if (n < mx_n) {
    // The heap is not full so put the new value at the bottom of the
    // heap and let it rise up to its correct level.
    n++;
    cd_p = n;
    pa_p = cd_p / 2;
    for (;;) {
      if (pa_p > 0) {
        if (v < heap_v[pa_p]) {
          heap_v[cd_p] = heap_v[pa_p];
          heap_ix[cd_p] = heap_ix[pa_p];
          cd_p = pa_p;
          pa_p = pa_p / 2;
          continue;
        }
      }
      break;
    }
    heap_v[cd_p] = v;
    heap_ix[cd_p] = ix;
  } else if (v > heap_v[1]) {
    // The heap is full so replace the least value with the new value
    // and let it sink down to its correct level.
    pa_p = 1;
    cd_p = pa_p + pa_p;
    for (;;) {
      if (cd_p <= n) {
        if (cd_p < n) {
          if (heap_v[cd_p] > heap_v[cd_p + 1]) cd_p++;
        }
        if (v > heap_v[cd_p]) {
          heap_v[pa_p] = heap_v[cd_p];
          heap_ix[pa_p] = heap_ix[cd_p];
          pa_p = cd_p;
          cd_p = cd_p + cd_p;
          continue;
        }
      }
      break;
    }
    heap_v[pa_p] = v;
    heap_ix[pa_p] = ix;
  }
  // Set heap_ix[0]=1 to indicate that the values form a heap.
  heap_ix[0] = 1;
  return;
}

void sortDecreasingHeap(const HighsInt n, vector<double>& heap_v,
                        vector<HighsInt>& heap_ix) {
  HighsInt fo_p, srt_p;
  HighsInt cd_p, pa_p;
  HighsInt ix;
  double v;
  if (n <= 1) return;
  if (heap_ix[0] != 1) {
    // The data are assumed to be completely unordered. A heap will be formed
    // and sorted.
    fo_p = n / 2 + 1;
    srt_p = n;
  } else {
    // The data are assumed to form a heap which is to be sorted.
    fo_p = 1;
    srt_p = n;
  }
  for (;;) {
    if (fo_p > 1) {
      fo_p = fo_p - 1;
      v = heap_v[fo_p];
      ix = heap_ix[fo_p];
    } else {
      v = heap_v[srt_p];
      ix = heap_ix[srt_p];
      heap_v[srt_p] = heap_v[1];
      heap_ix[srt_p] = heap_ix[1];
      srt_p--;
      if (srt_p == 1) {
        heap_v[1] = v;
        heap_ix[1] = ix;
        return;
      }
    }
    pa_p = fo_p;
    cd_p = fo_p + fo_p;
    for (;;) {
      if (cd_p <= srt_p) {
        if (cd_p < srt_p) {
          if (heap_v[cd_p] > heap_v[cd_p + 1]) cd_p = cd_p + 1;
        }
        if (v > heap_v[cd_p]) {
          heap_v[pa_p] = heap_v[cd_p];
          heap_ix[pa_p] = heap_ix[cd_p];
          pa_p = cd_p;
          cd_p = cd_p + cd_p;
          continue;
        }
      }
      break;
    }
    heap_v[pa_p] = v;
    heap_ix[pa_p] = ix;
  }
  return;
}

void maxheapsort(HighsInt* heap_v, HighsInt n) {
  buildMaxheap(heap_v, n);
  maxHeapsort(heap_v, n);
}

void maxheapsort(HighsInt* heap_v, HighsInt* heap_i, HighsInt n) {
  buildMaxheap(heap_v, heap_i, n);
  maxHeapsort(heap_v, heap_i, n);
}

void maxheapsort(double* heap_v, HighsInt* heap_i, HighsInt n) {
  buildMaxheap(heap_v, heap_i, n);
  maxHeapsort(heap_v, heap_i, n);
}

void buildMaxheap(HighsInt* heap_v, HighsInt n) {
  HighsInt i;
  for (i = n / 2; i >= 1; i--) {
    maxHeapify(heap_v, i, n);
  }
}

void buildMaxheap(HighsInt* heap_v, HighsInt* heap_i, HighsInt n) {
  HighsInt i;
  for (i = n / 2; i >= 1; i--) {
    maxHeapify(heap_v, heap_i, i, n);
  }
}

void buildMaxheap(double* heap_v, HighsInt* heap_i, HighsInt n) {
  HighsInt i;
  for (i = n / 2; i >= 1; i--) {
    maxHeapify(heap_v, heap_i, i, n);
  }
}

void maxHeapsort(HighsInt* heap_v, HighsInt n) {
  HighsInt temp_v;
  HighsInt i;
  for (i = n; i >= 2; i--) {
    temp_v = heap_v[i];
    heap_v[i] = heap_v[1];
    heap_v[1] = temp_v;
    maxHeapify(heap_v, 1, i - 1);
  }
}

void maxHeapsort(HighsInt* heap_v, HighsInt* heap_i, HighsInt n) {
  HighsInt temp_v;
  HighsInt i, temp_i;
  for (i = n; i >= 2; i--) {
    temp_v = heap_v[i];
    heap_v[i] = heap_v[1];
    heap_v[1] = temp_v;
    temp_i = heap_i[i];
    heap_i[i] = heap_i[1];
    heap_i[1] = temp_i;
    maxHeapify(heap_v, heap_i, 1, i - 1);
  }
}

void maxHeapsort(double* heap_v, HighsInt* heap_i, HighsInt n) {
  double temp_v;
  HighsInt i, temp_i;
  for (i = n; i >= 2; i--) {
    temp_v = heap_v[i];
    heap_v[i] = heap_v[1];
    heap_v[1] = temp_v;
    temp_i = heap_i[i];
    heap_i[i] = heap_i[1];
    heap_i[1] = temp_i;
    maxHeapify(heap_v, heap_i, 1, i - 1);
  }
}

void maxHeapify(HighsInt* heap_v, HighsInt i, HighsInt n) {
  HighsInt temp_v;
  HighsInt j;
  temp_v = heap_v[i];
  j = 2 * i;
  while (j <= n) {
    if (j < n && heap_v[j + 1] > heap_v[j]) j = j + 1;
    if (temp_v > heap_v[j])
      break;
    else if (temp_v <= heap_v[j]) {
      heap_v[j / 2] = heap_v[j];
      j = 2 * j;
    }
  }
  heap_v[j / 2] = temp_v;
  return;
}

void maxHeapify(HighsInt* heap_v, HighsInt* heap_i, HighsInt i, HighsInt n) {
  HighsInt temp_v;
  HighsInt j, temp_i;
  temp_v = heap_v[i];
  temp_i = heap_i[i];
  j = 2 * i;
  while (j <= n) {
    if (j < n && heap_v[j + 1] > heap_v[j]) j = j + 1;
    if (temp_v > heap_v[j])
      break;
    else if (temp_v <= heap_v[j]) {
      heap_v[j / 2] = heap_v[j];
      heap_i[j / 2] = heap_i[j];
      j = 2 * j;
    }
  }
  heap_v[j / 2] = temp_v;
  heap_i[j / 2] = temp_i;
  return;
}

void maxHeapify(double* heap_v, HighsInt* heap_i, HighsInt i, HighsInt n) {
  double temp_v;
  HighsInt j, temp_i;
  temp_v = heap_v[i];
  temp_i = heap_i[i];
  j = 2 * i;
  while (j <= n) {
    if (j < n && heap_v[j + 1] > heap_v[j]) j = j + 1;
    if (temp_v > heap_v[j])
      break;
    else if (temp_v <= heap_v[j]) {
      heap_v[j / 2] = heap_v[j];
      heap_i[j / 2] = heap_i[j];
      j = 2 * j;
    }
  }
  heap_v[j / 2] = temp_v;
  heap_i[j / 2] = temp_i;
  return;
}

bool increasingSetOk(const vector<HighsInt>& set,
                     const HighsInt set_entry_lower,
                     const HighsInt set_entry_upper, bool strict) {
  HighsInt set_num_entries = set.size();
  bool check_bounds = set_entry_lower <= set_entry_upper;
  HighsInt previous_entry;
  if (check_bounds) {
    if (strict) {
      previous_entry = set_entry_lower - 1;
    } else {
      previous_entry = set_entry_lower;
    }
  } else {
    previous_entry = -kHighsIInf;
  }
  for (HighsInt k = 0; k < set_num_entries; k++) {
    HighsInt entry = set[k];
    if (strict) {
      if (entry <= previous_entry) return false;
    } else {
      if (entry < previous_entry) return false;
    }
    if (check_bounds && entry > set_entry_upper) return false;
    previous_entry = entry;
  }
  return true;
}

bool increasingSetOk(const vector<double>& set, const double set_entry_lower,
                     const double set_entry_upper, bool strict) {
  HighsInt set_num_entries = set.size();
  bool check_bounds = set_entry_lower <= set_entry_upper;
  double previous_entry;
  if (check_bounds) {
    if (strict) {
      if (set_entry_lower < 0) {
        previous_entry = (1 + kHighsTiny) * set_entry_lower;
      } else if (set_entry_lower > 0) {
        previous_entry = (1 - kHighsTiny) * set_entry_lower;
      } else {
        previous_entry = -kHighsTiny;
      }
    } else {
      previous_entry = set_entry_lower;
    }
  } else {
    previous_entry = -kHighsInf;
  }
  for (HighsInt k = 0; k < set_num_entries; k++) {
    double entry = set[k];
    if (strict) {
      if (entry <= previous_entry) return false;
    } else {
      if (entry < previous_entry) return false;
    }
    if (check_bounds && entry > set_entry_upper) return false;
    previous_entry = entry;
  }
  return true;
}

void sortSetData(const HighsInt num_entries, vector<HighsInt>& set,
                 const double* data0, const double* data1, const double* data2,
                 double* sorted_data0, double* sorted_data1,
                 double* sorted_data2) {
  if (num_entries <= 0) return;
  vector<HighsInt> sort_set_vec(1 + num_entries);
  vector<HighsInt> perm_vec(1 + num_entries);

  HighsInt* sort_set = sort_set_vec.data();
  HighsInt* perm = perm_vec.data();

  for (HighsInt ix = 0; ix < num_entries; ix++) {
    sort_set[1 + ix] = set[ix];
    perm[1 + ix] = ix;
  }
  maxheapsort(sort_set, perm, num_entries);
  for (HighsInt ix = 0; ix < num_entries; ix++) {
    set[ix] = sort_set[1 + ix];
    if (data0 != NULL) sorted_data0[ix] = data0[perm[1 + ix]];
    if (data1 != NULL) sorted_data1[ix] = data1[perm[1 + ix]];
    if (data2 != NULL) sorted_data2[ix] = data2[perm[1 + ix]];
  }
}

void sortSetData(const HighsInt num_entries, vector<HighsInt>& set,
                 const HighsVarType* data0, HighsVarType* sorted_data0) {
  if (num_entries <= 0) return;
  vector<HighsInt> sort_set_vec(1 + num_entries);
  vector<HighsInt> perm_vec(1 + num_entries);

  HighsInt* sort_set = sort_set_vec.data();
  HighsInt* perm = perm_vec.data();

  for (HighsInt ix = 0; ix < num_entries; ix++) {
    sort_set[1 + ix] = set[ix];
    perm[1 + ix] = ix;
  }
  maxheapsort(sort_set, perm, num_entries);
  for (HighsInt ix = 0; ix < num_entries; ix++) {
    set[ix] = sort_set[1 + ix];
    if (data0 != NULL) sorted_data0[ix] = data0[perm[1 + ix]];
  }
}
#endif
