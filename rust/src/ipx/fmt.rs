//! The output formatting of control.h (Format, Scientific, Fixed, Textline)
//! with iostream semantics. Rust's float formatting rounds the exact binary
//! value half-to-even like printf; only the exponent is spelled differently
//! ("e-5" vs "e-05").

use std::fmt::Display;

pub(crate) use crate::io::mps::g;

/// Format(i, width) / Format(const char*, width): right-aligned
pub(crate) fn fmt<T: Display>(v: T, width: usize) -> String {
    format!("{v:>width$}")
}

/// printf's "%.{prec}e"
pub(crate) fn sci_raw(d: f64, prec: usize) -> String {
    if !d.is_finite() {
        return nonfinite(d);
    }
    let s = format!("{d:.prec$e}");
    let (mantissa, exp) = s.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let sign = if exp < 0 { '-' } else { '+' };
    format!("{mantissa}e{sign}{:02}", exp.abs())
}

fn nonfinite(d: f64) -> String {
    if d.is_nan() {
        if d.is_sign_negative() { "-nan" } else { "nan" }.into()
    } else if d > 0.0 {
        "inf".into()
    } else {
        "-inf".into()
    }
}

/// Scientific(d, width, prec)
pub(crate) fn sci(d: f64, width: usize, prec: usize) -> String {
    fmt(sci_raw(d, prec), width)
}

/// Fixed(d, width, prec)
pub(crate) fn fixed(d: f64, width: usize, prec: usize) -> String {
    let s = if d.is_finite() {
        format!("{d:.prec$}")
    } else {
        nonfinite(d)
    };
    fmt(s, width)
}

pub(crate) fn sci2(d: f64) -> String {
    sci(d, 0, 2)
}
pub(crate) fn sci8(d: f64) -> String {
    sci(d, 0, 8)
}
pub(crate) fn fix2(d: f64) -> String {
    fixed(d, 0, 2)
}
pub(crate) fn time(d: f64) -> String {
    fixed(d, 8, 1)
}

/// Textline(text): indented by 4 and left-aligned in 52 columns
pub(crate) fn textline(text: &str) -> String {
    format!("    {text:<52}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats() {
        assert_eq!(sci(1714.65449, 15, 8), " 1.71465449e+03");
        assert_eq!(sci(-0.03125, 9, 2), "-3.12e-02");
        assert_eq!(sci(0.0, 0, 0), "0e+00");
        assert_eq!(fixed(0.25, 8, 1), "     0.2");
        assert_eq!(fmt(27, 3), " 27");
        assert_eq!(g(100.0), "100");
        assert_eq!(g(2e-5), "2e-05");
        assert_eq!(textline("ab").len(), 56);
    }
}
