//! HEkk's records of the bases visited in a solve (visited_basis_) and of
//! bad basis changes (bad_basis_change_), owned here so that the simplex
//! iterations in Rust need not call C++ for them. The C++ HEkk reaches them
//! through `HEkk::RustBasisRecords` (highs/simplex/HEkk.h), whose methods
//! call the shims below.

use std::collections::HashSet;
use std::hash::{BuildHasherDefault, Hasher};

/// The basis hashes are already well mixed: spread them by one multiply
#[derive(Default)]
pub struct U64Hasher(u64);

impl Hasher for U64Hasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ b as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
    }
    fn write_u64(&mut self, x: u64) {
        self.0 = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

/// BadBasisChangeReason
pub const REASON_ALL: i32 = 0;
pub const REASON_SINGULAR: i32 = 1;
pub const REASON_CYCLING: i32 = 2;
pub const REASON_FAILED_INFEASIBILITY_PROOF: i32 = 3;

/// HighsSimplexBadBasisChangeRecord
#[derive(Clone, Debug, PartialEq)]
pub struct BadBasisChange {
    pub taboo: bool,
    pub row_out: i32,
    pub variable_out: i32,
    pub variable_in: i32,
    pub reason: i32,
    pub save_value: f64,
}

#[derive(Default)]
pub struct BasisRecords {
    pub visited: HashSet<u64, BuildHasherDefault<U64Hasher>>,
    pub bad: Vec<BadBasisChange>,
    /// Results of a solve in Rust that C++ takes when it returns
    pub out: SolveOut,
}

/// What a solve in Rust (hekk.rs) leaves for HEkk's C++-owned vectors,
/// taken by HEkk::solve when the Rust returns
#[derive(Default)]
pub struct SolveOut {
    /// HEkk::hot_start_ as at the last INVERT: the refactorization
    /// information (use, pivot rows, variables and types, synthetic tick)
    /// and nonbasicMove
    pub hot_start: Option<HotStart>,
    /// HEkk::primal_phase1_dual_, if set
    pub primal_phase1_dual: Option<Vec<f64>>,
}

pub struct HotStart {
    pub refactor_use: bool,
    pub pivot_row: Vec<i32>,
    pub pivot_var: Vec<i32>,
    pub pivot_type: Vec<i8>,
    pub build_synthetic_tick: f64,
    pub nonbasic_move: Vec<i8>,
}

impl BasisRecords {
    /// HEkk::clearBadBasisChange
    pub fn clear_bad_basis_change(&mut self, reason: i32) {
        if reason == REASON_ALL {
            self.bad.clear();
        } else {
            self.bad.retain(|r| r.reason != reason);
        }
    }

    /// HEkk::updateBadBasisChange: drop the changes whose rows' primal
    /// values have moved
    pub fn update_bad_basis_change(&mut self, col_aq_array: &[f64], theta_primal: f64, tolerance: f64) {
        if !self.bad.is_empty() {
            self.bad.retain(|r| !((col_aq_array[r.row_out as usize] * theta_primal).abs() >= tolerance));
        }
    }

    /// HEkk::addBadBasisChange
    pub fn add_bad_basis_change(&mut self, row_out: i32, variable_out: i32, variable_in: i32, reason: i32, taboo: bool) -> i32 {
        let found = self.bad.iter().position(|r| {
            r.row_out == row_out && r.variable_out == variable_out && r.variable_in == variable_in && r.reason == reason
        });
        match found {
            Some(i) => {
                self.bad[i].taboo = taboo;
                i as i32
            }
            None => {
                self.bad.push(BadBasisChange { taboo, row_out, variable_out, variable_in, reason, save_value: 0.0 });
                self.bad.len() as i32 - 1
            }
        }
    }

    /// HEkk::clearBadBasisChangeTabooFlag
    pub fn clear_taboo_flag(&mut self) {
        for r in &mut self.bad {
            r.taboo = false;
        }
    }

    /// HEkk::tabooBadBasisChange
    pub fn taboo(&self) -> bool {
        self.bad.iter().any(|r| r.taboo)
    }

    /// HEkk::applyTabooRowOut (which = 0) and applyTabooVariableIn (1)
    pub fn apply_taboo(&mut self, values: &mut [f64], overwrite_with: f64, which: i32) {
        for r in &mut self.bad {
            if r.taboo {
                let i = if which == 0 { r.row_out } else { r.variable_in } as usize;
                r.save_value = values[i];
                values[i] = overwrite_with;
            }
        }
    }

    /// HEkk::unapplyTabooRowOut (which = 0) and unapplyTabooVariableIn (1):
    /// in reverse, so that a repeated index gets its first saved value
    pub fn unapply_taboo(&self, values: &mut [f64], which: i32) {
        for r in self.bad.iter().rev() {
            if r.taboo {
                let i = if which == 0 { r.row_out } else { r.variable_in } as usize;
                values[i] = r.save_value;
            }
        }
    }

    /// The loop of HEkk::isBadBasisChange over the bad basis changes: is
    /// this change among them (and then made taboo)?
    pub fn find_and_make_taboo(&mut self, row_out: i32, variable_out: i32, variable_in: i32) -> bool {
        for r in &mut self.bad {
            if r.variable_out == variable_out && r.variable_in == variable_in && r.row_out == row_out {
                r.taboo = true;
                return true;
            }
        }
        false
    }
}

/// `extern "C"` shims for HEkk::RustBasisRecords
mod ffi {
    use super::BasisRecords;
    use crate::ffi::sl_mut;

    #[no_mangle]
    pub extern "C" fn highs_rs_basis_records_new() -> *mut BasisRecords {
        Box::into_raw(Box::default())
    }

    /// # Safety
    /// `p` from highs_rs_basis_records_new (or null)
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_basis_records_free(p: *mut BasisRecords) {
        if !p.is_null() {
            drop(Box::from_raw(p));
        }
    }

    /// # Safety
    /// `p` and `from` valid
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_basis_records_copy(p: *mut BasisRecords, from: *const BasisRecords) {
        (*p).visited = (*from).visited.clone();
        (*p).bad = (*from).bad.clone();
    }

    /// # Safety
    /// `p` valid
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_visited_basis_clear(p: *mut BasisRecords) {
        (*p).visited.clear();
    }

    /// # Safety
    /// `p` valid
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_visited_basis_insert(p: *mut BasisRecords, hash: u64) {
        (*p).visited.insert(hash);
    }

    /// # Safety
    /// `p` valid
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_visited_basis_find(p: *const BasisRecords, hash: u64) -> bool {
        (*p).visited.contains(&hash)
    }

    /// # Safety
    /// `p` valid
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_bad_basis_clear(p: *mut BasisRecords, reason: i32) {
        (*p).clear_bad_basis_change(reason);
    }

    /// # Safety
    /// `p` valid; `col_aq_array` covers the rows of the records
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_bad_basis_update(
        p: *mut BasisRecords,
        col_aq_array: *const f64,
        n: i32,
        theta_primal: f64,
        tolerance: f64,
    ) {
        (*p).update_bad_basis_change(crate::ffi::sl(col_aq_array, n), theta_primal, tolerance);
    }

    /// # Safety
    /// `p` valid
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_bad_basis_add(
        p: *mut BasisRecords,
        row_out: i32,
        variable_out: i32,
        variable_in: i32,
        reason: i32,
        taboo: bool,
    ) -> i32 {
        (*p).add_bad_basis_change(row_out, variable_out, variable_in, reason, taboo)
    }

    /// # Safety
    /// `p` valid
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_bad_basis_clear_taboo_flag(p: *mut BasisRecords) {
        (*p).clear_taboo_flag();
    }

    /// # Safety
    /// `p` valid
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_bad_basis_taboo(p: *const BasisRecords) -> bool {
        (*p).taboo()
    }

    /// # Safety
    /// `p` valid
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_bad_basis_find_and_make_taboo(
        p: *mut BasisRecords,
        row_out: i32,
        variable_out: i32,
        variable_in: i32,
    ) -> bool {
        (*p).find_and_make_taboo(row_out, variable_out, variable_in)
    }

    /// # Safety
    /// `p` valid; `values` covers the indices of the records
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_bad_basis_apply_taboo(
        p: *mut BasisRecords,
        values: *mut f64,
        n: i32,
        overwrite_with: f64,
        which: i32,
    ) {
        (*p).apply_taboo(sl_mut(values, n), overwrite_with, which);
    }

    /// # Safety
    /// `p` valid; `values` covers the indices of the records
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_bad_basis_unapply_taboo(p: *const BasisRecords, values: *mut f64, n: i32, which: i32) {
        (*p).unapply_taboo(sl_mut(values, n), which);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records() {
        let mut r = BasisRecords::default();
        r.visited.insert(7);
        assert!(r.visited.contains(&7) && !r.visited.contains(&8));
        assert_eq!(r.add_bad_basis_change(1, 5, 6, REASON_CYCLING, true), 0);
        assert_eq!(r.add_bad_basis_change(2, 5, 6, REASON_CYCLING, false), 1);
        assert_eq!(r.add_bad_basis_change(1, 5, 6, REASON_CYCLING, true), 0);
        let mut values = [10.0, 11.0, 12.0];
        r.apply_taboo(&mut values, 0.0, 0);
        assert_eq!(values, [10.0, 0.0, 12.0]);
        r.unapply_taboo(&mut values, 0);
        assert_eq!(values, [10.0, 11.0, 12.0]);
        assert!(r.find_and_make_taboo(2, 5, 6) && r.bad[1].taboo);
        r.clear_taboo_flag();
        assert!(!r.taboo());
        // Row 1 moves, row 2 does not
        r.update_bad_basis_change(&[0.0, 1.0, 0.0], 1.0, 1e-7);
        assert_eq!(r.bad.len(), 1);
        assert_eq!(r.bad[0].row_out, 2);
        r.clear_bad_basis_change(REASON_ALL);
        assert!(r.bad.is_empty());
    }
}
