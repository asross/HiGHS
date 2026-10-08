// Checksum of the HighsSort heap sorts on arrays with many ties; the Rust
// test sort::tests::golden computes the same and compares. Build against
// the C++ sources (not a HIGHS_RUST build):
//   clang++ -O2 -std=c++11 -Ihighs -Ibuild golden_sort.cpp highs/util/HighsSort.cpp
#include <cstdint>
#include <cstdio>
#include <vector>

#include "util/HighsSort.h"
static uint64_t h = 1469598103934665603ull;
static void mix(uint64_t v) { h = (h ^ v) * 1099511628211ull; }
static uint64_t s = 1;
static int next(int m) {
  s = s * 6364136223846793005ull + 1442695040888963407ull;
  return int((s >> 33) % uint64_t(m));
}
int main() {
  for (int t = 0; t < 2000; t++) {
    int n = next(40);
    int m = 1 + next(10);
    std::vector<int> a(n + 1), ai(n + 1), b(n + 1);
    std::vector<double> d(n + 1), dv(n + 2, 0);
    std::vector<int> di(n + 2, 0);
    for (int k = 1; k <= n; k++) {
      a[k] = b[k] = next(m);
      ai[k] = k;
      d[k] = next(m) * 0.5;
    }
    std::vector<int> ci = ai;
    maxheapsort(a.data(), ai.data(), n);
    maxheapsort(b.data(), n);
    maxheapsort(d.data(), ci.data(), n);
    int hn = 0;
    for (int k = 1; k <= n; k++)
      addToDecreasingHeap(hn, n / 2 + 1, dv, di, next(m) * 0.25, k);
    sortDecreasingHeap(hn, dv, di);
    for (int k = 1; k <= n; k++) {
      mix(a[k]); mix(ai[k]); mix(b[k]); mix(uint64_t(d[k] * 4)); mix(ci[k]);
    }
    for (int k = 1; k <= hn; k++) { mix(uint64_t(dv[k] * 4)); mix(di[k]); }
  }
  printf("%llu\n", (unsigned long long)h);
}
