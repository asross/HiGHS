// Checksums of HighsRandom draws and HighsCDouble operations; the Rust test
// rust/src/util/tests.rs computes the same and compares.
#include <cmath>
#include <cstdio>
#include <cstring>
#include "util/HighsCDouble.h"
#include "util/HighsRandom.h"
static uint64_t h = 1469598103934665603ull;
static void mix(uint64_t v) { h = (h ^ v) * 1099511628211ull; }
static void mixd(double d) { uint64_t v; memcpy(&v, &d, 8); mix(v); }
static void mixc(HighsCDouble c) { double hl[2]; memcpy(hl, &c, 16); mixd(hl[0]); mixd(hl[1]); }
int main() {
  HighsRandom r(12345);
  for (int i = 0; i < 100000; i++) {
    mix(r.integer()); mix(r.integer(i % 1000 + 1)); mix(r.integer(-5, i % 77 + 3));
    mixd(r.fraction()); mixd(r.closedFraction()); mixd(r.real(-3.5, 1e3)); mix(r.bit());
  }
  printf("random %llu\n", (unsigned long long)h);
  h = 1469598103934665603ull;
  HighsRandom s(7);
  auto val = [&]() { return (s.fraction() - 0.5) * std::ldexp(1.0, s.integer(80) - 40); };
  for (int i = 0; i < 100000; i++) {
    double a = val(), b = val(), c = val();
    HighsCDouble x = HighsCDouble(a) + b, y = HighsCDouble(c) * a;
    mixc(x); mixc(y); mixc(x * y); mixc(x * c); mixc(x / y); mixc(x / b); mixc(x - y); mixc(x - c);
    mixc(a - y); mixc(a + y); mixc(a * y); mixc(b / x); mixc(-x);
    HighsCDouble z = x; z += y; z -= c; z *= y; z /= x; z *= b; z /= c; z -= x; mixc(z);
    mixc(sqrt(abs(x))); mixc(floor(x * 1e3)); mixc(ceil(y * 1e3)); mixc(round(x));
    z = x; z.renormalize(); mixc(z);
    mix(x < y); mix(x == a); mix(x >= b);
  }
  printf("cdouble %llu\n", (unsigned long long)h);
}
