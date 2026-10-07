//! What the file writers share: C's number formatting written straight
//! into a byte buffer (`%.Ng`, `%+.Ng`, widths, HiGHS's
//! highsDoubleToString), the output sink through which C++ receives the
//! bytes (`Out`), and the views of the C++ data the writers read.
//!
//! `g` is C's `%.{p}g`: Rust's `{:.*e}` rounds the exact binary value
//! half-to-even to p significant digits, as the C library does, and the
//! digits are then laid out as %g would. One formatting call per number
//! and no allocation, so writing is faster than through printf.

use crate::lp_data::ffi::RsMut;
use std::ffi::c_void;
use std::io::Write as _;

/// A C++ string: c_str() and strlen(c_str()), so text after a NUL is
/// dropped as "%s" drops it
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RsStr {
    pub ptr: *const u8,
    pub len: usize,
}

impl RsStr {
    /// # Safety
    /// `ptr` must be valid for `len` reads while the slice lives
    pub unsafe fn get<'a>(&self) -> &'a [u8] {
        if self.len == 0 {
            &[]
        } else {
            std::slice::from_raw_parts(self.ptr, self.len)
        }
    }
}

/// Where C++ wants the text: `emit(out, kind, text, len)` with kind -1 for
/// file text and otherwise a message: 0 for highsLogDev(kInfo), or the
/// HighsLogType of highsLogUser. With `chunked` (the file is stdout,
/// which highsFprintfString logs), file text is sent in the pieces C++
/// prints with one highsFprintfString each.
#[repr(C)]
pub struct COut {
    pub file: *mut c_void,
    pub log_options: *const c_void,
    pub chunked: bool,
    pub emit: Option<unsafe extern "C" fn(*const COut, i32, *const u8, usize)>,
}

pub const FILE_TEXT: i32 = -1;
pub const LOG_DEV: i32 = 0;
pub const LOG_INFO: i32 = 1;
pub const LOG_WARNING: i32 = 4;

/// The output of a writer: buffered file text and messages in order. With
/// no sink (tests), text and messages are kept.
pub struct Out<'a> {
    pub buf: Vec<u8>,
    pub messages: Vec<(i32, String)>,
    sink: Option<&'a COut>,
}

const FLUSH_AT: usize = 1 << 16;

impl<'a> Out<'a> {
    pub fn new(sink: Option<&'a COut>) -> Self {
        Out {
            buf: Vec::with_capacity(FLUSH_AT + 1024),
            messages: Vec::new(),
            sink,
        }
    }
    fn send(&self, kind: i32, text: &[u8]) {
        if let Some(c) = self.sink {
            if let Some(emit) = c.emit {
                // SAFETY: C++ gave the function and its context
                unsafe { emit(c, kind, text.as_ptr(), text.len()) }
            }
        }
    }
    /// Pass on the buffered text
    pub fn flush(&mut self) {
        if self.sink.is_some() && !self.buf.is_empty() {
            self.send(FILE_TEXT, &self.buf);
            self.buf.clear();
        }
    }
    /// The end of what C++ prints with one highsFprintfString
    pub fn chunk(&mut self) {
        if let Some(c) = self.sink {
            if c.chunked || self.buf.len() >= FLUSH_AT {
                self.flush();
            }
        }
    }
    /// The end of a line of a writer that fprintf's directly
    pub fn line(&mut self) {
        if self.buf.len() >= FLUSH_AT {
            self.flush();
        }
    }
    /// highsLogUser / highsLogDev of `text`
    pub fn msg(&mut self, kind: i32, text: &str) {
        if self.sink.is_some() {
            self.flush();
            self.send(kind, text.as_bytes());
        } else {
            self.messages.push((kind, text.to_string()));
        }
    }
    pub fn s(&mut self, s: &[u8]) -> &mut Self {
        self.buf.extend_from_slice(s);
        self
    }
    pub fn d(&mut self, v: i64) -> &mut Self {
        int(&mut self.buf, v);
        self
    }
}

impl Drop for Out<'_> {
    fn drop(&mut self) {
        self.flush();
    }
}

/// `%d`
pub fn int(out: &mut Vec<u8>, v: i64) {
    let _ = write!(out, "{v}");
}

/// `%{w}s` (right) or `%-{w}s` (left)
pub fn pad(out: &mut Vec<u8>, s: &[u8], w: usize, left: bool) {
    let fill = w.saturating_sub(s.len());
    if !left {
        out.resize(out.len() + fill, b' ');
    }
    out.extend_from_slice(s);
    if left {
        out.resize(out.len() + fill, b' ');
    }
}

/// glibc signs a NaN by its sign bit (and `+`); the macOS libc never does
fn nan(out: &mut Vec<u8>, v: f64, plus: bool) {
    if cfg!(target_os = "linux") {
        if v.is_sign_negative() {
            out.push(b'-');
        } else if plus {
            out.push(b'+');
        }
    }
    out.extend_from_slice(b"nan");
}

/// `%.{p}g`
pub fn g(out: &mut Vec<u8>, v: f64, p: usize) {
    g_sign(out, v, p, false)
}

/// `%{w}.{p}g`
pub fn g_w(out: &mut Vec<u8>, v: f64, p: usize, w: usize) {
    let start = out.len();
    g(out, v, p);
    // right-aligned
    let n = out.len() - start;
    if n < w {
        out.splice(start..start, std::iter::repeat(b' ').take(w - n));
    }
}

/// `%.{p}g`, or `%+.{p}g` with `plus`
pub fn g_sign(out: &mut Vec<u8>, v: f64, p: usize, plus: bool) {
    let p = p.max(1);
    if v.is_nan() {
        return nan(out, v, plus);
    }
    if v.is_sign_negative() {
        out.push(b'-');
    } else if plus {
        out.push(b'+');
    }
    let a = v.abs();
    if a.is_infinite() {
        out.extend_from_slice(b"inf");
        return;
    }
    if a == 0.0 {
        out.push(b'0');
        return;
    }
    // d.ddde[-]x with p digits, rounded as C rounds
    let mut tmp = [0u8; 48];
    let n = {
        let mut c = &mut tmp[..];
        let _ = write!(c, "{:.*e}", p - 1, a);
        48 - c.len()
    };
    let s = &tmp[..n];
    let e = s.iter().position(|&b| b == b'e').unwrap();
    let mut x: i32 = 0;
    let neg_x = s[e + 1] == b'-';
    for &b in &s[e + 1 + neg_x as usize..] {
        x = x * 10 + (b - b'0') as i32;
    }
    if neg_x {
        x = -x;
    }
    let mut digits = [0u8; 40];
    let mut nd = 0;
    for &b in &s[..e] {
        if b != b'.' {
            digits[nd] = b;
            nd += 1;
        }
    }
    // Trailing zeros are dropped, from the fraction only
    let trim = |from: usize| {
        let mut end = nd;
        while end > from && digits[end - 1] == b'0' {
            end -= 1;
        }
        end
    };
    if x < p as i32 && x >= -4 {
        if x >= 0 {
            let ip = (x + 1) as usize;
            out.extend_from_slice(&digits[..ip]);
            let end = trim(ip);
            if end > ip {
                out.push(b'.');
                out.extend_from_slice(&digits[ip..end]);
            }
        } else {
            out.extend_from_slice(b"0.");
            out.resize(out.len() + (-x - 1) as usize, b'0');
            out.extend_from_slice(&digits[..trim(0)]);
        }
    } else {
        out.push(digits[0]);
        let end = if libc_keeps_zeros(a, p) { nd } else { trim(1) };
        if end > 1 {
            out.push(b'.');
            out.extend_from_slice(&digits[1..end]);
        }
        out.push(b'e');
        out.push(if x < 0 { b'-' } else { b'+' });
        let ax = x.unsigned_abs();
        if ax < 10 {
            out.push(b'0');
        }
        let _ = write!(out, "{ax}");
    }
}

const TENS: [f64; 15] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14,
];

/// Whether the C library keeps the trailing zeros of the p digits of
/// %g's exponent form. The macOS libc (gdtoa's dtoa) does when an integer
/// below 1e15 is rounded down to p <= 14 digits by its "small integer"
/// path, which it takes when its floating-point "quick" path cannot
/// decide the rounding (a tie, or within its error bound of one):
/// %.4g of 55005 is "5.500e+04". Both paths are simulated here.
fn libc_keeps_zeros(a: f64, p: usize) -> bool {
    if !cfg!(target_os = "macos") || p > 14 || a >= 1e15 || a.fract() != 0.0 {
        return false;
    }
    let mut k = 0;
    while k < 14 && a >= TENS[k + 1] {
        k += 1;
    }
    if k < p {
        // All digits fit: no rounding
        return false;
    }
    // The quick path
    let mut d = a;
    if k > 0 {
        d /= TENS[k];
    }
    let mut eps = (2.0 * d + 7.0) * f64::powi(2.0, -52);
    eps *= TENS[p - 1];
    let mut ilim = p;
    let mut i = 1;
    loop {
        d -= d.trunc();
        if d == 0.0 {
            ilim = i;
        }
        if i == ilim {
            if d > 0.5 + eps || d < 0.5 - eps {
                return false;
            }
            break;
        }
        i += 1;
        d *= 10.0;
    }
    // The small integer path: exact, and keeps the digits when rounding
    // down
    let ds = TENS[k];
    let mut d = a;
    for i in 1.. {
        let l = (d / ds).trunc();
        d -= l * ds;
        if d == 0.0 {
            return false;
        }
        if i == p {
            d += d;
            return !(d > ds || (d == ds && l % 2.0 == 1.0));
        }
        d *= 10.0;
    }
    false
}

/// highsDoubleToString: `%.{n}g`, n the number of significant digits
/// above `tolerance` (between 1 and 16), or "0" when |v| is below it
pub fn double_to_string(out: &mut Vec<u8>, v: f64, tolerance: f64) {
    let a = v.abs();
    // std::max(tolerance, a): tolerance when a is NaN
    let m = if tolerance < a { a } else { tolerance };
    let l = if a == f64::INFINITY {
        1.0
    } else {
        1.0 - tolerance + (m / tolerance).log10()
    };
    match l as i32 {
        0 => out.push(b'0'),
        n @ 1..=15 => g(out, v, n as usize),
        _ => g(out, v, 16),
    }
}

/// The data of a HighsModel a writer reads (HighsRust.h: RsWriteModel)
#[repr(C)]
pub struct CWriteModel {
    pub num_col: i32,
    pub num_row: i32,
    pub col_cost: RsMut<f64>,
    pub col_lower: RsMut<f64>,
    pub col_upper: RsMut<f64>,
    pub row_lower: RsMut<f64>,
    pub row_upper: RsMut<f64>,
    /// Column-wise, or row-wise for the LP writer
    pub a_start: RsMut<i32>,
    pub a_index: RsMut<i32>,
    pub a_value: RsMut<f64>,
    pub sense: i32,
    pub offset: f64,
    pub integrality: RsMut<u8>,
    pub q_dim: i32,
    pub q_start: RsMut<i32>,
    pub q_index: RsMut<i32>,
    pub q_value: RsMut<f64>,
    pub col_names: RsMut<RsStr>,
    pub row_names: RsMut<RsStr>,
    pub model_name: RsStr,
    pub objective_name: RsStr,
    pub cost_row_location: i32,
}

/// `CWriteModel` as slices
pub struct Model<'a> {
    pub num_col: usize,
    pub num_row: usize,
    pub col_cost: &'a [f64],
    pub col_lower: &'a [f64],
    pub col_upper: &'a [f64],
    pub row_lower: &'a [f64],
    pub row_upper: &'a [f64],
    pub a_start: &'a [i32],
    pub a_index: &'a [i32],
    pub a_value: &'a [f64],
    /// ObjSense: 1 minimize, -1 maximize
    pub sense: i32,
    pub offset: f64,
    pub integrality: &'a [u8],
    pub q_dim: usize,
    pub q_start: &'a [i32],
    pub q_index: &'a [i32],
    pub q_value: &'a [f64],
    pub col_names: Vec<&'a [u8]>,
    pub row_names: Vec<&'a [u8]>,
    pub model_name: &'a [u8],
    pub objective_name: &'a [u8],
    pub cost_row_location: i32,
}

impl CWriteModel {
    /// # Safety
    /// The arrays must be valid and not written while the view lives
    pub unsafe fn view<'a>(&self) -> Model<'a> {
        Model {
            num_col: self.num_col as usize,
            num_row: self.num_row as usize,
            col_cost: self.col_cost.get(),
            col_lower: self.col_lower.get(),
            col_upper: self.col_upper.get(),
            row_lower: self.row_lower.get(),
            row_upper: self.row_upper.get(),
            a_start: self.a_start.get(),
            a_index: self.a_index.get(),
            a_value: self.a_value.get(),
            sense: self.sense,
            offset: self.offset,
            integrality: self.integrality.get(),
            q_dim: self.q_dim as usize,
            q_start: self.q_start.get(),
            q_index: self.q_index.get(),
            q_value: self.q_value.get(),
            col_names: self.col_names.get().iter().map(|s| s.get()).collect(),
            row_names: self.row_names.get().iter().map(|s| s.get()).collect(),
            model_name: self.model_name.get(),
            objective_name: self.objective_name.get(),
            cost_row_location: self.cost_row_location,
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::ffi::{c_char, CStr, CString};

    extern "C" {
        fn snprintf(buf: *mut c_char, n: usize, fmt: *const c_char, ...) -> i32;
    }

    pub(crate) fn c_f(fmt: &str, v: f64) -> String {
        let mut buf = [0 as c_char; 512];
        let f = CString::new(fmt).unwrap();
        unsafe {
            snprintf(buf.as_mut_ptr(), buf.len(), f.as_ptr(), v);
            CStr::from_ptr(buf.as_ptr()).to_str().unwrap().to_string()
        }
    }

    fn values() -> Vec<f64> {
        let mut vals = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            0.25,
            2.5,
            1e-5,
            1.5e-5,
            123456.0,
            1234567.0,
            999999.5,
            0.0001,
            0.00009999995,
            1e100,
            -1e-300,
            4.9e-324,
            1.7976931348623157e308,
            3.14159265358979,
            100.0,
            1e6,
            1e5,
            99999.95,
            0.1,
            0.3,
            2.0 / 3.0,
            -7.25e-7,
            1e15,
            1e16,
            999999999999999.9,
            9.9999999999999999e14,
            0.00001,
            1e-13,
            1e-14,
            5e-14,
            1.0000000000000002,
            1e30,
            -1e30,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
        let mut seed = 12345u64;
        for _ in 0..20000 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let m = (seed >> 11) as f64 / (1u64 << 53) as f64;
            let e = ((seed >> 3) % 60) as i32 - 30;
            vals.push((m - 0.5) * 10f64.powi(e));
            // short decimals, as in model files
            vals.push(((seed >> 20) % 100000) as f64 / 10f64.powi(((seed >> 40) % 8) as i32));
            vals.push(f64::from_bits(seed));
            // integers, and ties of them
            let i = (seed >> 14) % 1_000_000_000_000_000;
            vals.push(i as f64);
            vals.push(((i / 10) * 10 + 5) as f64);
            vals.push(((i >> 7) * 100 + 50) as f64);
            vals.push(((i >> 9) * 1000 + 500) as f64);
        }
        vals.retain(|v| !v.is_nan());
        vals
    }

    #[test]
    fn g_matches_libc() {
        for v in values() {
            for p in 1..=17 {
                let mut b = Vec::new();
                g(&mut b, v, p);
                assert_eq!(
                    String::from_utf8(b).unwrap(),
                    c_f(&format!("%.{p}g"), v),
                    "%.{p}g of {v:e}"
                );
            }
            let mut b = Vec::new();
            g_sign(&mut b, v, 15, true);
            assert_eq!(
                String::from_utf8(b).unwrap(),
                c_f("%+.15g", v),
                "%+.15g of {v:e}"
            );
            for (p, w) in [(6, 12), (6, 13)] {
                let mut b = Vec::new();
                g_w(&mut b, v, p, w);
                assert_eq!(String::from_utf8(b).unwrap(), c_f(&format!("%{w}.{p}g"), v));
            }
        }
        let mut b = Vec::new();
        g(&mut b, f64::NAN, 15);
        assert_eq!(String::from_utf8(b).unwrap(), c_f("%.15g", f64::NAN));
    }

    /// highsDoubleToString, as in HighsIO.cpp
    fn c_double_to_string(v: f64, tol: f64) -> String {
        let a = v.abs();
        let m = if tol < a { a } else { tol };
        let l = if a == f64::INFINITY {
            1.0
        } else {
            1.0 - tol + (m / tol).log10()
        };
        match l as i32 {
            0 => "0".into(),
            n @ 1..=15 => c_f(&format!("%.{n}g"), v),
            _ => c_f("%.16g", v),
        }
    }

    #[test]
    fn double_to_string_matches() {
        for v in values()
            .into_iter()
            .chain([f64::NAN, 1e-13, 9.99e-14, 1.0001e-13])
        {
            for tol in [1e-13, 1e-12] {
                let mut b = Vec::new();
                double_to_string(&mut b, v, tol);
                assert_eq!(
                    String::from_utf8(b).unwrap(),
                    c_double_to_string(v, tol),
                    "{v:e}"
                );
            }
        }
    }
}
