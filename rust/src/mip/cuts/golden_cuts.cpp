// Checksum of the orders that pdqsort, pdqsort_branchless, std::partial_sort,
// std::partition and std::priority_queue give pairs with many equal keys;
// rust/src/mip/cuts/tests.rs (sorts_match_cpp) computes the same with the
// Rust ports. Build from the repository root:
//   clang++ -O2 -std=c++17 -Ihighs -Ibuild-cpp rust/src/mip/cuts/golden_cuts.cpp
#include <algorithm>
#include <cstdio>
#include <queue>
#include <vector>

#include "../extern/pdqsort/pdqsort.h"
#include "util/HighsRandom.h"

static uint64_t h = 1469598103934665603ull;
static void mix(uint64_t v) { h = (h ^ v) * 1099511628211ull; }

int main() {
  HighsRandom r(99);
  using P = std::pair<int, int>;
  for (int round = 0; round < 300; ++round) {
    int n = (round * 7) % 400 + 1;
    int keys = round % 13 + 1;
    std::vector<P> base(n);
    for (int i = 0; i < n; ++i) base[i] = P(r.integer(keys), i);
    auto v = base;
    pdqsort(v.begin(), v.end(), [](const P& a, const P& b) { return a.first < b.first; });
    for (auto& x : v) mix(x.second);
    v = base;
    pdqsort_branchless(v.begin(), v.end(), [](const P& a, const P& b) { return a.first > b.first; });
    for (auto& x : v) mix(x.second);
    v = base;
    std::partial_sort(v.begin(), v.begin() + n / 3, v.end(),
                      [](const P& a, const P& b) { return a.first < b.first; });
    for (auto& x : v) mix(x.second);
    v = base;
    auto p = std::partition(v.begin(), v.end(), [](const P& a) { return a.first % 2 == 0; }) - v.begin();
    mix(p);
    for (auto& x : v) mix(x.second);
    auto cmp = [](const P& a, const P& b) { return a.first > b.first; };
    std::priority_queue<P, std::vector<P>, decltype(cmp)> q(cmp);
    for (auto& x : base) {
      q.push(x);
      if (x.second % 3 == 0) {
        mix(q.top().second);
        q.pop();
      }
    }
    while (!q.empty()) {
      mix(q.top().second);
      q.pop();
    }
  }
  printf("sorts %llu\n", (unsigned long long)h);
}
