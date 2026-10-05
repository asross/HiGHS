// Checksums of the hash helpers, HighsHashTable, HighsHashTree,
// HighsSparseVectorSum, HSet and HighsDisjointSets over random operation
// sequences, mixing in results and iteration orders; the Rust tests in
// rust/src/util/tests.rs compute the same and compare. From the repo root:
//   clang++ -O3 -std=c++11 -Ihighs -I<build dir with HConfig.h> \
//     rust/src/util/golden2.cpp highs/util/HighsHash.cpp highs/util/HSet.cpp \
//     -o golden2 && ./golden2
// (HighsInt 32 bits; libc++, whose std::tuple stores members in order.)
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <tuple>
#include <vector>

#include "util/HSet.h"
#include "util/HighsCDouble.h"
#include "util/HighsDisjointSets.h"
#include "util/HighsHash.h"
#include "util/HighsHashTree.h"
#include "util/HighsRandom.h"
#include "util/HighsSparseVectorSum.h"

static uint64_t h;
static void mix(uint64_t v) { h = (h ^ v) * 1099511628211ull; }
static void mixd(double d) {
  uint64_t v;
  memcpy(&v, &d, 8);
  mix(v);
}
static void start() { h = 1469598103934665603ull; }
static void report(const char* name) {
  printf("%s %llu\n", name, (unsigned long long)h);
}

using HH = HighsHashHelpers;

// hash of std::array<uint8_t, n> for n = 1..N
template <int N>
struct ArrHash {
  static void run(const uint8_t* buf) {
    ArrHash<N - 1>::run(buf);
    std::array<uint8_t, N> a;
    memcpy(a.data(), buf, N);
    mix(HH::hash(a));
  }
};
template <>
struct ArrHash<0> {
  static void run(const uint8_t*) {}
};

static double randDouble(HighsRandom& r, int range) {
  double f = r.fraction() - 0.5;
  return f * std::ldexp(1.0, r.integer(2 * range) - range);
}

static void hashes() {
  start();
  HighsRandom r(1);
  uint8_t buf[1024];
  for (int it = 0; it < 2000; ++it) {
    for (int b = 0; b < 1024; ++b) buf[b] = r.integer(256);
    ArrHash<64>::run(buf);
    HighsInt len = r.integer(1025);
    mix(HH::vector_hash(buf, len));

    HighsInt a = r.integer();
    HighsInt b = r.integer();
    mix(HH::hash(a));
    mix(HH::hash((uint16_t)a));
    mix(HH::hash((uint64_t(a) << 32) | uint64_t(b)));
    mix(HH::hash(std::make_pair(a, b)));
    mix(HH::hash(std::make_tuple(a, b, (HighsUInt)a)));
    double d = randDouble(r, 100);
    mix(HH::hash(d));
    mix(HH::double_hash_code(d));
    std::vector<HighsInt> v(r.integer(100));
    for (auto& x : v) x = r.integer();
    mix(HH::hash(v));

    uint64_t s = 0;
    uint32_t s32 = 0;
    for (int k = 0; k < 20; ++k) {
      HighsInt idx = r.integer(5000);
      uint64_t hi = r.integer();
      uint64_t val = (hi << 33) ^ uint64_t(r.integer());
      HH::sparse_combine(s, idx, val);
      mix(s);
      HH::sparse_combine(s, idx);
      mix(s);
      HH::sparse_combine32(s32, idx, val);
      mix(s32);
      if (r.bit()) {
        HH::sparse_inverse_combine(s, idx, val);
        mix(s);
        HH::sparse_inverse_combine(s, idx);
        mix(s);
        HH::sparse_inverse_combine32(s32, idx, val);
        mix(s32);
      }
      mix(HH::multiply_modM61(val, hi * 0x9e3779b97f4a7c15ull));
      mix(HH::modexp_M61(val & HH::M61(), r.integer(1000) + 1));
    }
  }
  report("hash");
}

static void hashTable() {
  start();
  HighsRandom r(2);
  HighsHashTable<HighsInt, HighsInt> t;
  HighsHashTable<std::tuple<HighsInt, HighsInt, HighsUInt>> s;
  auto setKey = [](HighsInt k) {
    return std::make_tuple(k, k ^ 0x55, (HighsUInt)k * 2654435761u);
  };
  const HighsInt ranges[] = {50, 2000, 20000};
  for (int phase = 0; phase < 12; ++phase) {
    HighsInt range = ranges[phase % 3];
    HighsInt insProb = phase % 2 ? 70 : 30;
    for (int i = 0; i < 15000; ++i) {
      HighsInt op = r.integer(100);
      HighsInt k = r.integer(range);
      if (op < insProb) {
        HighsInt v = r.integer(1000000);
        mix(t.insert(k, v));
        mix(s.insert(setKey(k)));
      } else if (op < insProb + 15) {
        mix(t.erase(k));
        mix(s.erase(setKey(k)));
      } else if (op < insProb + 25) {
        t[k] += 1;
        mix(t[k]);
      } else {
        const HighsInt* p = t.find(k);
        mix(p ? *p : -1);
        mix(s.find(setKey(k)) != nullptr);
      }
      mix(t.size());
      mix(s.size());
    }
    for (const auto& e : t) {
      mix(e.key());
      mix(e.value());
    }
    for (const auto& e : s) mix(std::get<2>(e.key()));
    if (phase % 4 == 3) {
      // erase most keys, in iteration order, to shrink the tables (this
      // thrashes: the remaining keys crowd the top slots, which overflow
      // the halved table, so it grows back; quadratic, hence small sizes)
      std::vector<HighsInt> keys;
      for (const auto& e : t) keys.push_back(e.key());
      for (HighsInt k : keys)
        if (r.integer(10) != 0) {
          mix(t.erase(k));
          mix(s.erase(setKey(k)));
        }
      for (const auto& e : t) mix(e.key());
      mix(t.size());
    }
    if (phase == 6) {
      t.clear();
      s.clear();
    }
  }

  HighsHashTable<HighsInt, HighsInt> c(1000);
  for (int i = 0; i < 900; ++i) c.insert(r.integer(), i);
  for (const auto& e : c) mix(e.key());

  HighsHashTable<std::vector<HighsInt>, HighsInt> vt;
  for (int i = 0; i < 20000; ++i) {
    std::vector<HighsInt> v(r.integer(2, 6));
    for (auto& x : v) x = r.integer(4);
    mix(vt.insert(v, i));
    const HighsInt* p = vt.find(v);
    mix(*p);
  }
  for (const auto& e : vt) mix(e.value());
  report("table");
}

static void hashTree() {
  start();
  HighsRandom r(3);
  std::vector<HighsHashTree<HighsInt, HighsInt>> trees(8);
  std::vector<HighsHashTree<HighsInt>> sets(8);
  const HighsInt ranges[] = {100, 3000, 1000000};
  for (int phase = 0; phase < 10; ++phase) {
    HighsInt range = ranges[phase % 3];
    HighsInt insProb = phase % 2 ? 60 : 35;
    for (int i = 0; i < 40000; ++i) {
      HighsInt w = r.integer(8);
      HighsInt op = r.integer(100);
      HighsInt k = r.integer(range);
      if (op < insProb) {
        HighsInt v = r.integer(1000000);
        auto res = trees[w].insert_or_get(k, v);
        mix(*res.first);
        mix(res.second);
        *res.first += 1;
        mix(sets[w].insert(k));
      } else if (op < insProb + 20) {
        trees[w].erase(k);
        sets[w].erase(k);
      } else if (op < insProb + 30) {
        const HighsInt* p = static_cast<const HighsHashTree<HighsInt, HighsInt>&>(trees[w]).find(k);
        mix(p ? *p : -1);
        mix(sets[w].contains(k));
      } else {
        HighsInt w2 = r.integer(8);
        auto e = trees[w].find_common(trees[w2]);
        mix(e ? e->key() : -1);
        auto e2 = sets[w].find_common(sets[w2]);
        mix(e2 ? e2->key() : -1);
      }
    }
    for (int w = 0; w < 8; ++w) {
      trees[w].for_each([](HighsInt k, HighsInt v) {
        mix(k);
        mix(v);
      });
      sets[w].for_each([](HighsInt k) { mix(k); });
      mix(trees[w].empty());
      if (phase % 4 == 3) {
        std::vector<HighsInt> keys;
        trees[w].for_each([&](HighsInt k, HighsInt) { keys.push_back(k); });
        for (HighsInt k : keys)
          if (r.integer(10) != 0) {
            trees[w].erase(k);
            sets[w].erase(k);
          }
        sets[w].for_each([](HighsInt k) { mix(k); });
      }
    }
  }
  report("tree");

  // Random keys always burst a leaf into children sized exactly. Force the
  // other bursts: keys with distinct leading 6 hash bits (all children
  // small) and keys with equal leading bits (a single child).
  start();
  for (int variant = 0; variant < 2; ++variant) {
    std::vector<HighsInt> keys;
    uint64_t used = 0;
    for (HighsInt k = 0; keys.size() < (variant ? 300u : 60u); ++k) {
      uint64_t chunk = HH::hash(k) >> 58;
      if (variant ? chunk == 0 : !((used >> chunk) & 1)) {
        used |= uint64_t{1} << chunk;
        keys.push_back(k);
      }
    }
    HighsHashTree<HighsInt, HighsInt> t;
    for (HighsInt k : keys) mix(t.insert(k, k));
    t.for_each([](HighsInt k, HighsInt) { mix(k); });
    for (size_t i = 0; i < keys.size(); i += 2) t.erase(keys[i]);
    t.for_each([](HighsInt k, HighsInt) { mix(k); });
    for (HighsInt k : keys) mix(t.insert(k, k));
    t.for_each([](HighsInt k, HighsInt) { mix(k); });
  }
  report("treeburst");
}

static void sparseVectorSum() {
  start();
  HighsRandom r(4);
  HighsSparseVectorSum s(1000);
  for (int it = 0; it < 3000; ++it) {
    HighsInt n = r.integer(2, 400);
    for (int j = 0; j < n; ++j) {
      HighsInt idx = r.integer(1000);
      double d = randDouble(r, 20);
      HighsInt kind = r.integer(3);
      if (kind == 0)
        s.add(idx, d);
      else if (kind == 1)
        s.add(idx, HighsCDouble(d) * 3.0 + 1e-17);
      else
        s.add(idx, -s.getValue(idx));
    }
    HighsInt op = r.integer(3);
    if (op == 0)
      s.cleanup([](HighsInt, double v) { return std::fabs(v) < 1e-3; });
    else if (op == 1)
      mix(s.partition([](HighsInt i) { return i % 3 == 0; }));
    for (HighsInt i : s.getNonzeros()) {
      mix(i);
      double hl[2];
      memcpy(hl, &s.values[i], 16);
      mixd(hl[0]);
      mixd(hl[1]);
    }
    s.clear();
  }
  report("sparsesum");
}

static void hset() {
  start();
  HighsRandom r(5);
  HSet set;
  set.setup(100, 50);
  for (int i = 0; i < 200000; ++i) {
    HighsInt op = r.integer(3);
    HighsInt e = r.integer(-2, 300);
    if (op == 0)
      mix(set.add(e));
    else if (op == 1)
      mix(set.remove(e));
    else
      mix(set.in(e));
    mix(set.count());
    if (i % 1000 == 999)
      for (HighsInt j = 0; j < set.count(); ++j) mix(set.entry()[j]);
    if (i % 50000 == 49999) set.clear();
  }
  report("hset");
}

static void disjointSets() {
  start();
  HighsRandom r(6);
  HighsDisjointSets<false> a(5000);
  HighsDisjointSets<true> b(5000);
  for (int i = 0; i < 20000; ++i) {
    HighsInt x = r.integer(5000);
    HighsInt y = r.integer(5000);
    a.merge(x, y);
    b.merge(x, y);
    HighsInt z = r.integer(5000);
    mix(a.getSet(z));
    mix(b.getSet(z));
    mix(a.getSetSize(a.getSet(z)));
    mix(b.getSetSize(b.getSet(z)));
  }
  report("disjointsets");
}

int main() {
  hashes();
  hashTable();
  hashTree();
  sparseVectorSum();
  hset();
  disjointSets();
}
