//! Bit-identity with the C++: checksums produced by the same sequence of
//! operations in golden.cpp (clang++ -O3 -std=c++11 -Ihighs -Ibuild).

use super::cdouble::CDouble;
use super::random::HighsRandom;

struct Fnv(u64);
impl Fnv {
    fn new() -> Self {
        Fnv(1469598103934665603)
    }
    fn mix(&mut self, v: u64) {
        self.0 = (self.0 ^ v).wrapping_mul(1099511628211);
    }
    fn mixd(&mut self, d: f64) {
        self.mix(d.to_bits());
    }
    fn mixc(&mut self, c: CDouble) {
        self.mixd(c.hi);
        self.mixd(c.lo);
    }
}

#[test]
fn random_matches_cpp() {
    let mut h = Fnv::new();
    let mut r = HighsRandom::new(12345);
    for i in 0..100000 {
        h.mix(r.integer() as u64);
        h.mix(r.integer_below(i % 1000 + 1) as u64);
        h.mix(r.integer_between(-5, i % 77 + 3) as i64 as u64);
        h.mixd(r.fraction());
        h.mixd(r.closed_fraction());
        h.mixd(r.real(-3.5, 1e3));
        h.mix(r.bit() as u64);
    }
    assert_eq!(h.0, 16995212409544007823);
}

#[test]
fn cdouble_matches_cpp() {
    let mut h = Fnv::new();
    let mut s = HighsRandom::new(7);
    let mut val = || {
        let f = s.fraction() - 0.5;
        f * 2f64.powi(s.integer_below(80) - 40)
    };
    for _ in 0..100000 {
        let (a, b, c) = (val(), val(), val());
        let x = CDouble::from(a) + b;
        let y = CDouble::from(c) * a;
        for v in [x, y, x * y, x * c, x / y, x / b, x - y, x - c, a - y, a + y, a * y, b / x, -x] {
            h.mixc(v);
        }
        let mut z = x;
        z += y;
        z -= c;
        z *= y;
        z /= x;
        z *= b;
        z /= c;
        z -= x;
        h.mixc(z);
        h.mixc(x.abs().sqrt());
        h.mixc((x * 1e3).floor());
        h.mixc((y * 1e3).ceil());
        h.mixc(x.round());
        let mut z = x;
        z.renormalize();
        h.mixc(z);
        h.mix((x < y) as u64);
        h.mix((x == a) as u64);
        h.mix((x >= b) as u64);
    }
    assert_eq!(h.0, 11651025907869496546);
}
