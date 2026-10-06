//! Fused multiply-add where the C++ build has one.
//!
//! The port mirrors clang: on arm64 clang contracts `a ± b*c` within an
//! expression into an FMA instruction (and the Rust uses `mul_add` there);
//! on x86_64 the default C++ build has no FMA instructions, so nothing is
//! fused. `f64::mul_add` without hardware FMA is a slow software `fma()`
//! with different rounding, so every mirrored FMA goes through `mul_add_c`,
//! which is `mul_add` on aarch64 and `self * a + b` elsewhere (or with the
//! `no_fma` feature, to compare with a C++ build without contraction).

pub trait ClangFma {
    fn mul_add_c(self, a: Self, b: Self) -> Self;
}

impl ClangFma for f64 {
    #[inline(always)]
    fn mul_add_c(self, a: f64, b: f64) -> f64 {
        #[cfg(all(target_arch = "aarch64", not(feature = "no_fma")))]
        {
            self.mul_add(a, b)
        }
        #[cfg(not(all(target_arch = "aarch64", not(feature = "no_fma"))))]
        {
            self * a + b
        }
    }
}
