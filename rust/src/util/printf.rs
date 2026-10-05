//! C's printf formatting, so that messages formatted in Rust are
//! byte-identical to those HiGHS formats in C++: the C++ format strings
//! are used verbatim. Rust rounds the exact binary value of a float
//! half-to-even, as the C library does; only the spelling of special
//! values and exponents differs, and is adjusted here.
//!
//! Supported: the conversions d i u x X c s e E f F g G % with the flags
//! `-+ 0#`, a width and a precision (not `*`); length modifiers are
//! skipped.

/// An argument of [`sprintf`]
#[derive(Clone, Copy, Debug)]
pub enum Arg<'a> {
    I(i64),
    F(f64),
    S(&'a str),
}

impl From<i32> for Arg<'_> {
    fn from(v: i32) -> Self {
        Arg::I(v as i64)
    }
}
impl From<i64> for Arg<'_> {
    fn from(v: i64) -> Self {
        Arg::I(v)
    }
}
impl From<usize> for Arg<'_> {
    fn from(v: usize) -> Self {
        Arg::I(v as i64)
    }
}
impl From<f64> for Arg<'_> {
    fn from(v: f64) -> Self {
        Arg::F(v)
    }
}
impl<'a> From<&'a str> for Arg<'a> {
    fn from(v: &'a str) -> Self {
        Arg::S(v)
    }
}
impl<'a> From<&'a String> for Arg<'a> {
    fn from(v: &'a String) -> Self {
        Arg::S(v)
    }
}

/// `sprintf!(fmt, args...)`: C's sprintf of the arguments
#[macro_export]
macro_rules! sprintf {
    ($fmt:expr) => { $crate::util::printf::sprintf($fmt, &[]) };
    ($fmt:expr, $($a:expr),+ $(,)?) => {
        $crate::util::printf::sprintf($fmt, &[$($crate::util::printf::Arg::from($a)),+])
    };
}

#[derive(Default)]
struct Spec {
    left: bool,
    plus: bool,
    space: bool,
    zero: bool,
    alt: bool,
    width: usize,
    prec: Option<usize>,
}

/// C's sprintf(fmt, args...)
pub fn sprintf(fmt: &str, args: &[Arg]) -> String {
    let mut out = String::with_capacity(fmt.len() + 16);
    let b = fmt.as_bytes();
    let mut args = args.iter();
    let mut i = 0;
    while i < b.len() {
        if b[i] != b'%' {
            let j = b[i..].iter().position(|&c| c == b'%').map_or(b.len(), |k| i + k);
            out.push_str(&fmt[i..j]);
            i = j;
            continue;
        }
        i += 1;
        let mut s = Spec::default();
        while i < b.len() {
            match b[i] {
                b'-' => s.left = true,
                b'+' => s.plus = true,
                b' ' => s.space = true,
                b'0' => s.zero = true,
                b'#' => s.alt = true,
                _ => break,
            }
            i += 1;
        }
        while i < b.len() && b[i].is_ascii_digit() {
            s.width = s.width * 10 + (b[i] - b'0') as usize;
            i += 1;
        }
        if i < b.len() && b[i] == b'.' {
            i += 1;
            let mut p = 0;
            while i < b.len() && b[i].is_ascii_digit() {
                p = p * 10 + (b[i] - b'0') as usize;
                i += 1;
            }
            s.prec = Some(p);
        }
        while i < b.len() && matches!(b[i], b'h' | b'l' | b'L' | b'q' | b'j' | b'z' | b't') {
            i += 1;
        }
        let conv = if i < b.len() { b[i] } else { b'%' };
        i += 1;
        if conv == b'%' {
            out.push('%');
            continue;
        }
        let arg = args.next().copied().unwrap_or(Arg::I(0));
        let (sign, body, numeric) = match (conv, arg) {
            (b'd' | b'i', Arg::I(v)) => {
                let mut digits = v.unsigned_abs().to_string();
                if let Some(p) = s.prec {
                    if p == 0 && v == 0 {
                        digits.clear();
                    }
                    while digits.len() < p {
                        digits.insert(0, '0');
                    }
                }
                (sign_of(v < 0, &s), digits, s.prec.is_none())
            }
            (b'u', Arg::I(v)) => (String::new(), unsigned(v).to_string(), s.prec.is_none()),
            (b'x', Arg::I(v)) => (String::new(), format!("{:x}", unsigned(v)), s.prec.is_none()),
            (b'X', Arg::I(v)) => (String::new(), format!("{:X}", unsigned(v)), s.prec.is_none()),
            (b'c', Arg::I(v)) => (String::new(), ((v as u8) as char).to_string(), false),
            (b's', Arg::S(v)) => {
                let v = match s.prec {
                    Some(p) if p < v.len() => &v[..p],
                    _ => v,
                };
                (String::new(), v.to_string(), false)
            }
            (b'e' | b'E' | b'f' | b'F' | b'g' | b'G', a) => {
                let v = match a {
                    Arg::F(v) => v,
                    Arg::I(v) => v as f64,
                    Arg::S(_) => 0.0,
                };
                let neg = v.is_sign_negative() && !(v.is_nan());
                let upper = conv.is_ascii_uppercase();
                let body = if v.is_nan() {
                    "nan".to_string()
                } else if v.is_infinite() {
                    "inf".to_string()
                } else {
                    let p = s.prec.unwrap_or(6);
                    match conv.to_ascii_lowercase() {
                        b'e' => fmt_e(v.abs(), p, s.alt),
                        b'f' => fmt_f(v.abs(), p, s.alt),
                        _ => fmt_g(v.abs(), p, s.alt),
                    }
                };
                let finite = v.is_finite();
                let body = if upper { body.to_ascii_uppercase() } else { body };
                // The C library gives NaN no sign
                let sign = if v.is_nan() { String::new() } else { sign_of(neg, &s) };
                (sign, body, finite)
            }
            // A mismatched argument: format it plainly
            (_, Arg::I(v)) => (String::new(), v.to_string(), false),
            (_, Arg::F(v)) => (String::new(), v.to_string(), false),
            (_, Arg::S(v)) => (String::new(), v.to_string(), false),
        };
        let len = sign.len() + body.len();
        if len >= s.width {
            out.push_str(&sign);
            out.push_str(&body);
        } else if s.left {
            out.push_str(&sign);
            out.push_str(&body);
            out.extend(std::iter::repeat(' ').take(s.width - len));
        } else if s.zero && numeric {
            out.push_str(&sign);
            out.extend(std::iter::repeat('0').take(s.width - len));
            out.push_str(&body);
        } else {
            out.extend(std::iter::repeat(' ').take(s.width - len));
            out.push_str(&sign);
            out.push_str(&body);
        }
    }
    out
}

/// An int argument as C's unsigned conversions see it: 32 bits if it fits
fn unsigned(v: i64) -> u64 {
    if i32::try_from(v).is_ok() {
        v as u32 as u64
    } else {
        v as u64
    }
}

fn sign_of(neg: bool, s: &Spec) -> String {
    if neg {
        "-".into()
    } else if s.plus {
        "+".into()
    } else if s.space {
        " ".into()
    } else {
        String::new()
    }
}

/// %.{p}e of a finite non-negative value
fn fmt_e(v: f64, p: usize, alt: bool) -> String {
    let r = format!("{v:.p$e}");
    let (mantissa, exp) = r.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let mut m = mantissa.to_string();
    if alt && p == 0 {
        m.push('.');
    }
    format!("{m}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
}

/// %.{p}f of a finite non-negative value
fn fmt_f(v: f64, p: usize, alt: bool) -> String {
    let mut r = format!("{v:.p$}");
    if alt && p == 0 {
        r.push('.');
    }
    r
}

/// %.{p}g of a finite non-negative value
fn fmt_g(v: f64, p: usize, alt: bool) -> String {
    let p = if p == 0 { 1 } else { p };
    // The exponent after rounding to p significant digits
    let x: i32 = if v == 0.0 {
        0
    } else {
        let r = format!("{:.*e}", p - 1, v);
        r.split_once('e').unwrap().1.parse().unwrap()
    };
    let mut r = if (p as i32) > x && x >= -4 {
        fmt_f(v, (p as i32 - 1 - x) as usize, alt)
    } else {
        fmt_e(v, p - 1, alt)
    };
    if !alt {
        // Remove trailing zeros of the fraction, and a trailing point
        let (num, exp) = match r.find('e') {
            Some(k) => (r[..k].to_string(), r[k..].to_string()),
            None => (r.clone(), String::new()),
        };
        let num = if num.contains('.') {
            num.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            num
        };
        r = num + &exp;
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::{c_char, CStr, CString};

    extern "C" {
        fn snprintf(buf: *mut c_char, n: usize, fmt: *const c_char, ...) -> i32;
    }

    fn c_f(fmt: &str, v: f64) -> String {
        let mut buf = [0 as c_char; 512];
        let f = CString::new(fmt).unwrap();
        unsafe {
            snprintf(buf.as_mut_ptr(), buf.len(), f.as_ptr(), v);
            CStr::from_ptr(buf.as_ptr()).to_str().unwrap().to_string()
        }
    }

    fn c_d(fmt: &str, v: i32) -> String {
        let mut buf = [0 as c_char; 512];
        let f = CString::new(fmt).unwrap();
        unsafe {
            snprintf(buf.as_mut_ptr(), buf.len(), f.as_ptr(), v);
            CStr::from_ptr(buf.as_ptr()).to_str().unwrap().to_string()
        }
    }

    #[test]
    fn floats_match_libc() {
        let fmts = [
            "%g", "%11.4g", "%10.4g", "%9.4g", "%12g", "%.4g", "%#g", "%-12.3g|", "%+g", "% g", "%e",
            "%.2e", "%15.8e", "%f", "%.1f", "%8.2f", "%012.4f", "%-8.1f|", "%g%%", "%.0f", "%.0e",
            "%.10g", "%.17g", "%G", "%E", "%010.3g",
        ];
        let mut vals = vec![
            0.0, -0.0, 1.0, -1.0, 0.5, 0.25, 2.5, 1e-5, 1.5e-5, 123456.0, 1234567.0, 999999.5, 0.0001,
            0.00009999995, 1e100, -1e-300, 4.9e-324, 1.7976931348623157e308, 3.14159265358979, 100.0,
            1e6, 1e5, 99999.95, 0.1, 0.3, 2.0 / 3.0, -7.25e-7, f64::INFINITY, f64::NEG_INFINITY, f64::NAN,
        ];
        let mut seed = 12345u64;
        for _ in 0..2000 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let m = (seed >> 11) as f64 / (1u64 << 53) as f64;
            let e = ((seed >> 3) % 40) as i32 - 20;
            vals.push((m - 0.5) * 10f64.powi(e));
        }
        for f in fmts {
            for &v in &vals {
                assert_eq!(sprintf(f, &[Arg::F(v)]), c_f(f, v), "format {f} value {v:e}");
            }
        }
    }

    #[test]
    fn ints_and_strings() {
        for f in ["%d", "%4d", "%-4d|", "%04d", "%+d", "%.3d", "%x", "%2d"] {
            for v in [0, 1, -1, 42, -42, 123456, i32::MIN, i32::MAX] {
                assert_eq!(sprintf(f, &[Arg::I(v as i64)]), c_d(f, v), "format {f} value {v}");
            }
        }
        assert_eq!(sprintf!("%s = %d; %-6s|%5s", "x", 3, "ab", "cd"), "x = 3; ab    |   cd");
        assert_eq!(sprintf!("%.2s %%", "abc"), "ab %");
    }
}
