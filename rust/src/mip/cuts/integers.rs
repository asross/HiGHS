//! The parts of HighsIntegers (highs/util/HighsIntegers.h) that cut
//! separation uses.

use crate::util::cdouble::CDouble;

#[inline]
pub fn nearest_integer(x: f64) -> i64 {
    (x + 0.5f64.copysign(x)) as i64
}

#[inline]
pub fn is_integral(x: f64, eps: f64) -> bool {
    let y = (x - x.trunc()).abs();
    y.min(1.0 - y) <= eps
}

pub fn gcd(mut a: i64, mut b: i64) -> i64 {
    if a < 0 {
        a = -a;
    }
    if b < 0 {
        b = -b;
    }
    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }
    loop {
        let h = a % b;
        a = b;
        b = h;
        if b == 0 {
            return a;
        }
    }
}

/// A rational approximation of x with denominator at most maxdenom.
pub fn denominator(x: f64, eps: f64, maxdenom: i64) -> i64 {
    let mut ai = x as i64;
    let mut m = [ai, 1i64, 1, 0];
    let mut xi = CDouble::from(x);
    let mut fraction = xi - ai as f64;
    while fraction > eps {
        xi = 1.0 / fraction;
        if xi.to_f64() > (1i64 << 53) as f64 {
            break;
        }
        ai = xi.to_f64() as i64;
        let mut t = m[2].wrapping_mul(ai).wrapping_add(m[3]);
        if t > maxdenom {
            break;
        }
        m[3] = m[2];
        m[2] = t;
        t = m[0].wrapping_mul(ai).wrapping_add(m[1]);
        m[1] = m[0];
        m[0] = t;
        fraction = xi - ai as f64;
    }
    ai = (maxdenom - m[3]) / m[2];
    m[1] = m[1].wrapping_add(m[0].wrapping_mul(ai));
    m[3] = m[3].wrapping_add(m[2].wrapping_mul(ai));
    let x0 = m[0] as f64 / m[2] as f64;
    let x1 = m[1] as f64 / m[3] as f64;
    let x = x.abs();
    let err0 = (x - x0).abs();
    let err1 = (x - x1).abs();
    if err0 < err1 {
        m[2]
    } else {
        m[3]
    }
}

extern "C" {
    #[link_name = "frexp"]
    fn c_frexp(x: f64, e: *mut i32) -> f64;
    #[link_name = "ldexp"]
    fn c_ldexp(x: f64, e: i32) -> f64;
}

/// C frexp
#[inline]
pub fn frexp(x: f64) -> (f64, i32) {
    let mut e = 0;
    // SAFETY: e is a valid out pointer
    let m = unsafe { c_frexp(x, &mut e) };
    (m, e)
}

#[inline]
fn frexp_exp(x: f64) -> i32 {
    frexp(x).1
}

/// C ldexp
#[inline]
pub fn ldexp(x: f64, e: i32) -> f64 {
    // SAFETY: a pure libm function
    unsafe { c_ldexp(x, e) }
}

pub fn integral_scale(vals: &[f64], deltadown: f64, deltaup: f64) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    // std::minmax_element: the first smallest and the last largest
    let (mut imin, mut imax) = (0usize, 0usize);
    // libc++ minmax_element compares in pairs; for |.| keys the result is
    // the first minimum and the last maximum
    for i in 1..vals.len() {
        if vals[i].abs() < vals[imin].abs() {
            imin = i;
        }
        if !(vals[i].abs() < vals[imax].abs()) {
            imax = i;
        }
    }
    let minval = vals[imin];
    let maxval = vals[imax];

    let mut expshift = 0;
    if minval < -deltadown || minval > deltaup {
        expshift = frexp_exp(minval);
    }
    expshift = (-expshift).max(0) + 3;

    let mut exp_max_val = frexp_exp(maxval);
    exp_max_val = exp_max_val.min(32);
    if exp_max_val + expshift > 32 {
        expshift = 32 - exp_max_val;
    }

    let mut denom: u64 = 75u64 << expshift;
    let mut startdenom: i64 = denom as i64;
    let mut val = startdenom as f64 * CDouble::from(vals[0]);
    let mut downval = (val + deltaup).floor();
    let mut fraction = val - downval;

    if fraction > deltadown {
        denom = denom.wrapping_mul(denominator(fraction.to_f64(), deltaup, 1000) as u64);
        val = denom as f64 * CDouble::from(vals[0]);
        downval = (val + deltaup).floor();
        fraction = val - downval;
        if fraction > deltadown {
            return 0.0;
        }
    }

    let mut currgcd: u64 = downval.to_f64().abs() as u64;

    for &v in &vals[1..] {
        val = denom as f64 * CDouble::from(v);
        downval = (val + deltaup).floor();
        fraction = val - downval;

        if fraction > deltadown {
            val = startdenom as f64 * CDouble::from(v);
            fraction = val - val.floor();
            denom = denom.wrapping_mul(denominator(fraction.to_f64(), deltaup, 1000) as u64);
            val = denom as f64 * CDouble::from(v);
            downval = (val + deltaup).floor();
            fraction = val - downval;
            if fraction > deltadown {
                return 0.0;
            }
        }

        if currgcd != 1 {
            currgcd = gcd(currgcd as i64, downval.to_f64() as i64) as u64;
            if denom > u32::MAX as u64 {
                denom /= currgcd;
                if startdenom != 1 {
                    startdenom /= gcd(currgcd as i64, startdenom);
                }
                currgcd = 1;
            }
        }
    }

    denom as f64 / currgcd as f64
}
