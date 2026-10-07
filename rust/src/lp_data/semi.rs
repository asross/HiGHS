//! Semi-variables (HighsLpUtils.cpp: assessSemiVariables,
//! relaxSemiVariables, activeModifiedUpperBounds) and HighsLp::unapplyMods.
//! The modification records (HighsLpMods) are C++ vectors: C++ passes
//! buffers of the model's size, Rust returns how many entries to append
//! (or that a record is to be cleared), and C++ clears the records after
//! unapplyMods.

use super::ffi::RsMut;
use super::var_type::{CONTINUOUS, INTEGER, SEMI_CONTINUOUS, SEMI_INTEGER};
use super::{Log, LogType, Status, INF};
use crate::log_user;

/// kMaxSemiVariableUpper
const MAX_SEMI_VARIABLE_UPPER: f64 = 1e5;

fn is_semi(t: u8) -> bool {
    t == SEMI_CONTINUOUS || t == SEMI_INTEGER
}

/// The records of assessSemiVariables to append to HighsLpMods
#[repr(C)]
pub struct CSemiMods {
    pub inconsistent_index: RsMut<i32>,
    pub inconsistent_lower: RsMut<f64>,
    pub inconsistent_upper: RsMut<f64>,
    pub inconsistent_type: RsMut<u8>,
    pub non_semi_index: RsMut<i32>,
    pub tightened_index: RsMut<i32>,
    pub tightened_value: RsMut<f64>,
    /// Entries to append, or -1 to clear the record
    pub num_inconsistent: i32,
    pub num_non_semi: i32,
    pub num_tightened: i32,
    pub made_mods: bool,
}

/// assessSemiVariables for an LP with integrality
///
/// # Safety
/// The buffers of `m` hold num_col entries
pub unsafe fn assess_semi_variables(
    log: &Log,
    col_lower: &mut [f64],
    col_upper: &mut [f64],
    integrality: &mut [u8],
    m: &mut CSemiMods,
) -> Status {
    let mut return_status = Status::Ok;
    let (ii, il, iu, it) = (
        m.inconsistent_index.get_mut(),
        m.inconsistent_lower.get_mut(),
        m.inconsistent_upper.get_mut(),
        m.inconsistent_type.get_mut(),
    );
    let ns = m.non_semi_index.get_mut();
    let (ti, tv) = (m.tightened_index.get_mut(), m.tightened_value.get_mut());
    let mut num_illegal_lower = 0;
    let mut num_illegal_upper = 0;
    let mut num_tightened_upper = 0usize;
    let mut num_inconsistent_semi = 0usize;
    let mut num_non_semi = 0usize;
    let mut num_non_continuous_variables = 0;
    const LOWER_BOUND_MU: f64 = 10.0;
    for j in 0..integrality.len() {
        let t = integrality[j];
        if is_semi(t) {
            if col_lower[j] > col_upper[j] {
                ii[num_inconsistent_semi] = j as i32;
                il[num_inconsistent_semi] = col_lower[j];
                iu[num_inconsistent_semi] = col_upper[j];
                it[num_inconsistent_semi] = t;
                num_inconsistent_semi += 1;
                continue;
            }
            if col_lower[j] == 0.0 {
                ns[num_non_semi] = j as i32;
                num_non_semi += 1;
                if t == SEMI_INTEGER {
                    num_non_continuous_variables += 1;
                }
                continue;
            }
            if col_lower[j] < 0.0 {
                num_illegal_lower += 1;
            } else if col_upper[j] > MAX_SEMI_VARIABLE_UPPER {
                if LOWER_BOUND_MU * col_lower[j] > MAX_SEMI_VARIABLE_UPPER {
                    num_illegal_upper += 1;
                } else {
                    ti[num_tightened_upper] = j as i32;
                    tv[num_tightened_upper] = MAX_SEMI_VARIABLE_UPPER;
                    num_tightened_upper += 1;
                }
            }
            num_non_continuous_variables += 1;
        } else if t == INTEGER {
            num_non_continuous_variables += 1;
        }
    }
    if num_inconsistent_semi > 0 {
        log_user!(
            log,
            LogType::Warning,
            "%d semi-continuous/integer variable(s) have inconsistent bounds so are fixed at zero\n",
            num_inconsistent_semi
        );
        return_status = Status::Warning;
    }
    if num_non_semi > 0 {
        log_user!(
            log,
            LogType::Warning,
            "%d semi-continuous/integer variable(s) have zero lower bound so are continuous/integer\n",
            num_non_semi
        );
        return_status = Status::Warning;
    }
    if num_non_continuous_variables == 0 {
        log_user!(
            log,
            LogType::Warning,
            "No semi-integer/integer variables in model with non-empty integrality\n"
        );
        return_status = Status::Warning;
    }
    let has_illegal_bounds = num_illegal_lower > 0 || num_illegal_upper > 0;
    m.num_tightened = num_tightened_upper as i32;
    m.num_inconsistent = num_inconsistent_semi as i32;
    m.num_non_semi = num_non_semi as i32;
    if num_tightened_upper > 0 {
        log_user!(
            log,
            LogType::Warning,
            "%d semi-continuous/integer variable(s) have upper bounds exceeding %g that can be tightened to %g > %g*lower)\n",
            num_tightened_upper,
            MAX_SEMI_VARIABLE_UPPER,
            MAX_SEMI_VARIABLE_UPPER,
            LOWER_BOUND_MU
        );
        return_status = Status::Warning;
        if has_illegal_bounds {
            m.num_tightened = -1;
            num_tightened_upper = 0;
        } else {
            for k in 0..num_tightened_upper {
                let use_upper_bound = tv[k];
                let j = ti[k] as usize;
                tv[k] = col_upper[j];
                col_upper[j] = use_upper_bound;
            }
        }
    }
    if num_inconsistent_semi > 0 {
        if has_illegal_bounds {
            m.num_inconsistent = -1;
            num_inconsistent_semi = 0;
        } else {
            for &j in &ii[..num_inconsistent_semi] {
                let j = j as usize;
                col_lower[j] = 0.0;
                col_upper[j] = 0.0;
                integrality[j] = CONTINUOUS;
            }
        }
    }
    if num_non_semi > 0 {
        if has_illegal_bounds {
            // As in the C++, num_non_semi is not reset
            m.num_non_semi = -1;
        } else {
            for &j in &ns[..num_non_semi] {
                let j = j as usize;
                integrality[j] = if integrality[j] == SEMI_CONTINUOUS { CONTINUOUS } else { INTEGER };
            }
        }
    }
    if num_illegal_lower > 0 {
        log_user!(
            log,
            LogType::Error,
            "%d semi-continuous/integer variable(s) have negative lower bounds\n",
            num_illegal_lower
        );
        return_status = Status::Error;
    }
    if num_illegal_upper > 0 {
        log_user!(
            log,
            LogType::Error,
            "%d semi-continuous/integer variables have upper bounds exceeding %g that cannot be modified due to large lower bounds\n",
            num_illegal_upper,
            MAX_SEMI_VARIABLE_UPPER
        );
        return_status = Status::Error;
    }
    m.made_mods = num_non_semi > 0 || num_inconsistent_semi > 0 || num_tightened_upper > 0;
    return_status
}

/// relaxSemiVariables: the semi-variables' indices and lower bounds put
/// in the buffers, their lower bounds zeroed; returns how many
pub fn relax_semi_variables(col_lower: &mut [f64], integrality: &[u8], index: &mut [i32], value: &mut [f64]) -> usize {
    let mut n = 0;
    for j in 0..integrality.len() {
        if is_semi(integrality[j]) {
            index[n] = j as i32;
            value[n] = col_lower[j];
            col_lower[j] = 0.0;
            n += 1;
        }
    }
    n
}

/// activeModifiedUpperBounds
pub fn active_modified_upper_bounds(log: &Log, tightened_index: &[i32], col_upper: &[f64], col_value: &[f64], pft: f64) -> bool {
    let mut num_active_modified_upper = 0;
    let mut min_semi_variable_margin = INF;
    for &j in tightened_index {
        let j = j as usize;
        let value = col_value[j];
        let upper = col_upper[j];
        let semi_variable_margin = upper - value;
        if value > upper - pft {
            min_semi_variable_margin = 0.0;
            num_active_modified_upper += 1;
        } else {
            // std::min(semi_variable_margin, min_semi_variable_margin)
            min_semi_variable_margin = super::lp_utils::cmin(semi_variable_margin, min_semi_variable_margin);
        }
    }
    if num_active_modified_upper > 0 {
        log_user!(
            log,
            LogType::Error,
            "%d semi-variables are active at modified upper bounds\n",
            num_active_modified_upper
        );
    } else if !tightened_index.is_empty() {
        log_user!(
            log,
            LogType::Warning,
            "No semi-variables are active at modified upper bounds: a large minimum margin (%g) suggests optimality, but there is no guarantee\n",
            min_semi_variable_margin
        );
    }
    num_active_modified_upper != 0
}

/// The HighsLpMods records read by unapplyMods
#[repr(C)]
pub struct CLpMods {
    pub non_semi_index: RsMut<i32>,
    pub inconsistent_index: RsMut<i32>,
    pub inconsistent_lower: RsMut<f64>,
    pub inconsistent_upper: RsMut<f64>,
    pub inconsistent_type: RsMut<u8>,
    pub relaxed_index: RsMut<i32>,
    pub relaxed_value: RsMut<f64>,
    pub tightened_index: RsMut<i32>,
    pub tightened_value: RsMut<f64>,
}

/// HighsLp::unapplyMods before the records are cleared
///
/// # Safety
/// The records valid
pub unsafe fn unapply_mods(m: &CLpMods, col_lower: &mut [f64], col_upper: &mut [f64], integrality: &mut [u8]) {
    for &j in m.non_semi_index.get() {
        let j = j as usize;
        integrality[j] = if integrality[j] == CONTINUOUS { SEMI_CONTINUOUS } else { SEMI_INTEGER };
    }
    let (il, iu, it) = (m.inconsistent_lower.get(), m.inconsistent_upper.get(), m.inconsistent_type.get());
    for (k, &j) in m.inconsistent_index.get().iter().enumerate() {
        let j = j as usize;
        col_lower[j] = il[k];
        col_upper[j] = iu[k];
        integrality[j] = it[k];
    }
    let rv = m.relaxed_value.get();
    for (k, &j) in m.relaxed_index.get().iter().enumerate() {
        col_lower[j as usize] = rv[k];
    }
    let tv = m.tightened_value.get();
    for (k, &j) in m.tightened_index.get().iter().enumerate() {
        col_upper[j as usize] = tv[k];
    }
}

// The C++ entry points (HighsLpUtilsRust.cpp)

/// # Safety
/// The arrays valid; the buffers of `m` hold num_col entries
#[no_mangle]
pub unsafe extern "C" fn highs_rs_assess_semi_variables(
    log: *const Log,
    col_lower: RsMut<f64>,
    col_upper: RsMut<f64>,
    integrality: RsMut<u8>,
    m: *mut CSemiMods,
) -> i32 {
    assess_semi_variables(&*log, col_lower.get_mut(), col_upper.get_mut(), integrality.get_mut(), &mut *m) as i32
}

/// # Safety
/// The arrays valid; the buffers hold num_col entries
#[no_mangle]
pub unsafe extern "C" fn highs_rs_relax_semi_variables(
    col_lower: RsMut<f64>,
    integrality: RsMut<u8>,
    index: RsMut<i32>,
    value: RsMut<f64>,
) -> usize {
    relax_semi_variables(col_lower.get_mut(), integrality.get(), index.get_mut(), value.get_mut())
}

/// # Safety
/// The arrays valid
#[no_mangle]
pub unsafe extern "C" fn highs_rs_active_modified_upper_bounds(
    log: *const Log,
    tightened_index: RsMut<i32>,
    col_upper: RsMut<f64>,
    col_value: RsMut<f64>,
    pft: f64,
) -> bool {
    active_modified_upper_bounds(&*log, tightened_index.get(), col_upper.get(), col_value.get(), pft)
}

/// # Safety
/// The arrays valid
#[no_mangle]
pub unsafe extern "C" fn highs_rs_unapply_mods(
    m: *const CLpMods,
    col_lower: RsMut<f64>,
    col_upper: RsMut<f64>,
    integrality: RsMut<u8>,
) {
    unapply_mods(&*m, col_lower.get_mut(), col_upper.get_mut(), integrality.get_mut());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semi() {
        let mut lower = [1.0, 0.0, 2.0, 3.0];
        let mut upper = [0.5, 4.0, 1e6, 5.0];
        let mut integrality = [SEMI_CONTINUOUS, SEMI_INTEGER, SEMI_CONTINUOUS, INTEGER];
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g) =
            ([0i32; 4], [0f64; 4], [0f64; 4], [0u8; 4], [0i32; 4], [0i32; 4], [0f64; 4]);
        let mut m = CSemiMods {
            inconsistent_index: RsMut { ptr: a.as_mut_ptr(), len: 4 },
            inconsistent_lower: RsMut { ptr: b.as_mut_ptr(), len: 4 },
            inconsistent_upper: RsMut { ptr: c.as_mut_ptr(), len: 4 },
            inconsistent_type: RsMut { ptr: d.as_mut_ptr(), len: 4 },
            non_semi_index: RsMut { ptr: e.as_mut_ptr(), len: 4 },
            tightened_index: RsMut { ptr: f.as_mut_ptr(), len: 4 },
            tightened_value: RsMut { ptr: g.as_mut_ptr(), len: 4 },
            num_inconsistent: 0,
            num_non_semi: 0,
            num_tightened: 0,
            made_mods: false,
        };
        let s = unsafe { assess_semi_variables(&Log::none(), &mut lower, &mut upper, &mut integrality, &mut m) };
        assert_eq!(s, Status::Warning);
        assert_eq!((m.num_inconsistent, m.num_non_semi, m.num_tightened, m.made_mods), (1, 1, 1, true));
        assert_eq!((lower[0], upper[0], integrality[0]), (0.0, 0.0, CONTINUOUS));
        assert_eq!(integrality[1], INTEGER);
        assert_eq!((upper[2], g[0]), (1e5, 1e6));
        assert!(active_modified_upper_bounds(&Log::none(), &f[..1], &upper, &[0.0, 0.0, 1e5, 0.0], 1e-7));
    }
}
