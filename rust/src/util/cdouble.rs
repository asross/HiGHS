//! HighsCDouble: double-double arithmetic (error-free transformations).
//! Every operation mirrors the C++ step by step, including where clang fuses
//! a product into an add on arm64, so results are bit-identical.

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Clone, Copy, Debug, Default)]
pub struct CDouble {
    pub hi: f64,
    pub lo: f64,
}

#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    let z = x - a;
    (x, (a - (x - z)) + (b - z))
}

#[inline]
fn split(a: f64) -> (f64, f64) {
    const FACTOR: f64 = ((1 << 27) + 1) as f64;
    let c = FACTOR * a;
    let x = c - (c - a);
    (x, a - x)
}

/// y = a2 * b2 - (((x - a1 * b1) - a2 * b1) - a1 * b2), each product fused.
#[inline]
fn two_product(a: f64, b: f64) -> (f64, f64) {
    let x = a * b;
    let (a1, a2) = split(a);
    let (b1, b2) = split(b);
    let t = (-a1).mul_add(b1, x);
    let t = (-a2).mul_add(b1, t);
    let t = (-a1).mul_add(b2, t);
    (x, a2.mul_add(b2, -t))
}

impl CDouble {
    pub const fn new(hi: f64, lo: f64) -> Self {
        CDouble { hi, lo }
    }

    #[inline]
    pub fn to_f64(self) -> f64 {
        self.hi + self.lo
    }

    pub fn renormalize(&mut self) {
        (self.hi, self.lo) = two_sum(self.hi, self.lo);
    }

    pub fn abs(self) -> Self {
        if self < 0.0 { -self } else { self }
    }

    pub fn sqrt(self) -> Self {
        let c = (self.hi + self.lo).sqrt();
        if c == 0.0 {
            return 0.0.into();
        }
        let mut res = self / c;
        res += c;
        res.hi *= 0.5;
        res.lo *= 0.5;
        res
    }

    pub fn floor(self) -> Self {
        if self.abs() < 1.0 {
            return if self == 0.0 || self > 0.0 { 0.0 } else { -1.0 }.into();
        }
        let floor_x = self.to_f64().floor();
        let (hi, lo) = two_sum(floor_x, (self - floor_x).to_f64().floor());
        CDouble { hi, lo }
    }

    pub fn ceil(self) -> Self {
        if self.abs() < 1.0 {
            return if self == 0.0 || self < 0.0 { 0.0 } else { 1.0 }.into();
        }
        let ceil_x = self.to_f64().ceil();
        let (hi, lo) = two_sum(ceil_x, (self - ceil_x).to_f64().ceil());
        CDouble { hi, lo }
    }

    pub fn round(self) -> Self {
        (self + 0.5).floor()
    }
}

impl From<f64> for CDouble {
    #[inline]
    fn from(v: f64) -> Self {
        CDouble { hi: v, lo: 0.0 }
    }
}

impl From<CDouble> for f64 {
    #[inline]
    fn from(v: CDouble) -> f64 {
        v.to_f64()
    }
}

impl AddAssign<f64> for CDouble {
    #[inline]
    fn add_assign(&mut self, v: f64) {
        let (hi, c) = two_sum(v, self.hi);
        self.hi = hi;
        self.lo += c;
    }
}

impl AddAssign for CDouble {
    #[inline]
    fn add_assign(&mut self, v: CDouble) {
        *self += v.hi;
        self.lo += v.lo;
    }
}

impl SubAssign<f64> for CDouble {
    #[inline]
    fn sub_assign(&mut self, v: f64) {
        *self += -v;
    }
}

impl SubAssign for CDouble {
    #[inline]
    fn sub_assign(&mut self, v: CDouble) {
        *self -= v.hi;
        self.lo -= v.lo;
    }
}

impl MulAssign<f64> for CDouble {
    #[inline]
    fn mul_assign(&mut self, v: f64) {
        let c = self.lo * v;
        (self.hi, self.lo) = two_product(self.hi, v);
        *self += c;
    }
}

impl MulAssign for CDouble {
    #[inline]
    fn mul_assign(&mut self, v: CDouble) {
        let c1 = self.hi * v.lo;
        let c2 = self.lo * v.hi;
        (self.hi, self.lo) = two_product(self.hi, v.hi);
        *self += c1;
        *self += c2;
    }
}

impl DivAssign<f64> for CDouble {
    #[inline]
    fn div_assign(&mut self, v: f64) {
        let d = CDouble::new(self.hi / v, self.lo / v);
        let mut c = d * v - *self;
        c.hi /= v;
        c.lo /= v;
        *self = d - c;
    }
}

impl DivAssign for CDouble {
    #[inline]
    fn div_assign(&mut self, v: CDouble) {
        let vdbl = v.hi + v.lo;
        let d = CDouble::new(self.hi / vdbl, self.lo / vdbl);
        let mut c = d * v - *self;
        c.hi /= vdbl;
        c.lo /= vdbl;
        *self = d - c;
    }
}

impl Neg for CDouble {
    type Output = CDouble;
    #[inline]
    fn neg(self) -> CDouble {
        CDouble::new(-self.hi, -self.lo)
    }
}

impl Add<f64> for CDouble {
    type Output = CDouble;
    #[inline]
    fn add(self, v: f64) -> CDouble {
        let (hi, lo) = two_sum(self.hi, v);
        CDouble { hi, lo: lo + self.lo }
    }
}

impl Add for CDouble {
    type Output = CDouble;
    #[inline]
    fn add(self, v: CDouble) -> CDouble {
        let mut res = self + v.hi;
        res.lo += v.lo;
        res
    }
}

impl Add<CDouble> for f64 {
    type Output = CDouble;
    #[inline]
    fn add(self, b: CDouble) -> CDouble {
        b + self
    }
}

impl Sub<f64> for CDouble {
    type Output = CDouble;
    #[inline]
    fn sub(self, v: f64) -> CDouble {
        let (hi, lo) = two_sum(self.hi, -v);
        CDouble { hi, lo: lo + self.lo }
    }
}

impl Sub for CDouble {
    type Output = CDouble;
    #[inline]
    fn sub(self, v: CDouble) -> CDouble {
        let mut res = self - v.hi;
        res.lo -= v.lo;
        res
    }
}

impl Sub<CDouble> for f64 {
    type Output = CDouble;
    #[inline]
    fn sub(self, b: CDouble) -> CDouble {
        -b + self
    }
}

impl Mul<f64> for CDouble {
    type Output = CDouble;
    #[inline]
    fn mul(self, v: f64) -> CDouble {
        let (hi, lo) = two_product(self.hi, v);
        let mut res = CDouble { hi, lo };
        res += self.lo * v;
        res
    }
}

impl Mul for CDouble {
    type Output = CDouble;
    #[inline]
    fn mul(self, v: CDouble) -> CDouble {
        let mut res = self * v.hi;
        res += self.hi * v.lo;
        res
    }
}

impl Mul<CDouble> for f64 {
    type Output = CDouble;
    #[inline]
    fn mul(self, b: CDouble) -> CDouble {
        b * self
    }
}

impl Div<f64> for CDouble {
    type Output = CDouble;
    #[inline]
    fn div(mut self, v: f64) -> CDouble {
        self /= v;
        self
    }
}

impl Div for CDouble {
    type Output = CDouble;
    #[inline]
    fn div(mut self, v: CDouble) -> CDouble {
        self /= v;
        self
    }
}

impl Div<CDouble> for f64 {
    type Output = CDouble;
    #[inline]
    fn div(self, b: CDouble) -> CDouble {
        CDouble::from(self) / b
    }
}

// Comparisons are on the rounded value, as in the C++.
impl PartialEq<f64> for CDouble {
    fn eq(&self, other: &f64) -> bool {
        self.to_f64() == *other
    }
}

impl PartialOrd<f64> for CDouble {
    fn partial_cmp(&self, other: &f64) -> Option<std::cmp::Ordering> {
        self.to_f64().partial_cmp(other)
    }
}

impl PartialOrd for CDouble {
    fn partial_cmp(&self, other: &CDouble) -> Option<std::cmp::Ordering> {
        self.to_f64().partial_cmp(&other.to_f64())
    }
}

impl PartialEq for CDouble {
    fn eq(&self, other: &CDouble) -> bool {
        self.to_f64() == other.to_f64()
    }
}
