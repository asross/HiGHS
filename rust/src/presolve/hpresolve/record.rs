//! The recording side of HighsPostsolveStack (the inline templates of
//! highs/presolve/HighsPostsolveStack.h): each reduction pushes its record
//! struct and vectors of nonzeros (with original indices) as bytes in the
//! layout of HighsDataStack, which postsolve.rs reads back. The bytes are
//! buffered here and appended to the C++ stack by `flush` (the C++ keeps
//! owning the stack). The index maps are mirrored: copied at the start and
//! compressed together with the C++ ones.

use crate::presolve::postsolve::*;
use std::mem::{offset_of, size_of};

/// Builds the bytes of a record struct: every field written at its C++
/// offset, padding zeroed (the struct literal checks names and types)
macro_rules! record_bytes {
    ($ty:ident { $($f:ident : $v:expr),* $(,)? }) => {{
        let r = $ty { $($f: $v),* };
        let mut b = [0u8; size_of::<$ty>()];
        $(
            let bytes = r.$f.to_ne_bytes();
            let o = offset_of!($ty, $f);
            b[o..o + bytes.len()].copy_from_slice(&bytes);
        )*
        b
    }};
}

#[derive(Default)]
pub struct Recorder {
    pub orig_col_index: Vec<i32>,
    pub orig_row_index: Vec<i32>,
    /// bytes not yet appended to the C++ stack
    pub buf: Vec<u8>,
    /// (type, position) of the reductions not yet appended; positions count
    /// from the start of the C++ stack
    pub reductions: Vec<(u8, usize)>,
    /// size of the C++ stack data when buf is empty
    pub base: usize,
    /// total number of reductions, the C++ ones included
    pub num_reductions: usize,
    /// original columns that DuplicateColumn marks as not linearly
    /// transformable (applied by the C++ at the flush)
    pub not_transformable: Vec<i32>,
}

/// A nonzero of a slice: (index, value)
pub type Nz = (i32, f64);

impl Recorder {
    #[inline]
    fn added(&mut self, kind: u8) {
        self.reductions.push((kind, self.base + self.buf.len()));
        self.num_reductions += 1;
    }

    #[inline]
    fn push_vec(&mut self, it: impl Iterator<Item = Nz>, map: Map) {
        let mut n = 0usize;
        for (i, v) in it {
            let orig = match map {
                Map::Col => self.orig_col_index[i as usize],
                Map::Row => self.orig_row_index[i as usize],
                Map::None => i,
            };
            let mut b = [0u8; 16];
            b[0..4].copy_from_slice(&orig.to_ne_bytes());
            b[8..16].copy_from_slice(&v.to_ne_bytes());
            self.buf.extend_from_slice(&b);
            n += 1;
        }
        self.buf.extend_from_slice(&n.to_ne_bytes());
    }

    pub fn linear_transform(&mut self, col: i32, scale: f64, constant: f64) {
        let b = record_bytes!(LinearTransform { scale: scale, constant: constant, col: self.orig_col_index[col as usize] });
        self.buf.extend_from_slice(&b);
        self.added(LINEAR_TRANSFORM);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn free_col_substitution(
        &mut self,
        row: i32,
        col: i32,
        rhs: f64,
        col_cost: f64,
        row_type: i32,
        row_vec: impl Iterator<Item = Nz>,
        col_vec: impl Iterator<Item = Nz>,
    ) {
        let b = record_bytes!(FreeColSubstitution {
            rhs: rhs,
            col_cost: col_cost,
            row: self.orig_row_index[row as usize],
            col: self.orig_col_index[col as usize],
            row_type: row_type,
        });
        self.buf.extend_from_slice(&b);
        self.push_vec(row_vec, Map::Col);
        self.push_vec(col_vec, Map::Row);
        self.added(FREE_COL_SUBSTITUTION);
    }

    pub fn slack_col_substitution(&mut self, row: i32, col: i32, rhs: f64, row_vec: impl Iterator<Item = Nz>) {
        let b = record_bytes!(SlackColSubstitution {
            rhs: rhs,
            row: self.orig_row_index[row as usize],
            col: self.orig_col_index[col as usize],
        });
        self.buf.extend_from_slice(&b);
        self.push_vec(row_vec, Map::Col);
        self.added(SLACK_COL_SUBSTITUTION);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn doubleton_equation(
        &mut self,
        row: i32,
        col_subst: i32,
        col: i32,
        coef_subst: f64,
        coef: f64,
        rhs: f64,
        subst_lower: f64,
        subst_upper: f64,
        subst_cost: f64,
        lower_tightened: bool,
        upper_tightened: bool,
        row_type: i32,
        col_vec: impl Iterator<Item = Nz>,
    ) {
        let b = record_bytes!(DoubletonEquation {
            coef: coef,
            coef_subst: coef_subst,
            rhs: rhs,
            subst_lower: subst_lower,
            subst_upper: subst_upper,
            subst_cost: subst_cost,
            row: if row == -1 { -1 } else { self.orig_row_index[row as usize] },
            col_subst: self.orig_col_index[col_subst as usize],
            col: self.orig_col_index[col as usize],
            lower_tightened: lower_tightened as u8,
            upper_tightened: upper_tightened as u8,
            row_type: row_type,
        });
        self.buf.extend_from_slice(&b);
        self.push_vec(col_vec, Map::Row);
        self.added(DOUBLETON_EQUATION);
    }

    pub fn equality_row_addition(
        &mut self,
        row: i32,
        added_eq_row: i32,
        eq_row_scale: f64,
        eq_row_vec: impl Iterator<Item = Nz>,
    ) {
        let b = record_bytes!(EqualityRowAddition {
            row: self.orig_row_index[row as usize],
            added_eq_row: self.orig_row_index[added_eq_row as usize],
            eq_row_scale: eq_row_scale,
        });
        self.buf.extend_from_slice(&b);
        self.push_vec(eq_row_vec, Map::Col);
        self.added(EQUALITY_ROW_ADDITION);
    }

    /// target rows are (row, scale) with presolved row indices: the C++
    /// pushes them as given
    pub fn equality_row_additions(
        &mut self,
        added_eq_row: i32,
        eq_row_vec: impl Iterator<Item = Nz>,
        target_rows: &[(i32, f64)],
    ) {
        let b = record_bytes!(EqualityRowAdditions { added_eq_row: self.orig_row_index[added_eq_row as usize] });
        self.buf.extend_from_slice(&b);
        self.push_vec(eq_row_vec, Map::Col);
        self.push_vec(target_rows.iter().copied(), Map::None);
        self.added(EQUALITY_ROW_ADDITIONS);
    }

    pub fn singleton_row(&mut self, row: i32, col: i32, coef: f64, tightened_lower: bool, tightened_upper: bool) {
        let b = record_bytes!(SingletonRow {
            coef: coef,
            row: self.orig_row_index[row as usize],
            col: self.orig_col_index[col as usize],
            col_lower_tightened: tightened_lower as u8,
            col_upper_tightened: tightened_upper as u8,
        });
        self.buf.extend_from_slice(&b);
        self.added(SINGLETON_ROW);
    }

    fn fixed_col(&mut self, col: i32, fix_value: f64, col_cost: f64, fix_type: u8, col_vec: impl Iterator<Item = Nz>) {
        let b = record_bytes!(FixedCol {
            fix_value: fix_value,
            col_cost: col_cost,
            col: self.orig_col_index[col as usize],
            fix_type: fix_type,
        });
        self.buf.extend_from_slice(&b);
        self.push_vec(col_vec, Map::Row);
        self.added(FIXED_COL);
    }

    pub fn fixed_col_at_lower(&mut self, col: i32, fix_value: f64, col_cost: f64, col_vec: impl Iterator<Item = Nz>) {
        self.fixed_col(col, fix_value, col_cost, LOWER, col_vec);
    }

    pub fn fixed_col_at_upper(&mut self, col: i32, fix_value: f64, col_cost: f64, col_vec: impl Iterator<Item = Nz>) {
        self.fixed_col(col, fix_value, col_cost, UPPER, col_vec);
    }

    pub fn fixed_col_at_zero(&mut self, col: i32, col_cost: f64, col_vec: impl Iterator<Item = Nz>) {
        self.fixed_col(col, 0.0, col_cost, ZERO, col_vec);
    }

    pub fn removed_fixed_col(&mut self, col: i32, fix_value: f64, col_cost: f64, col_vec: impl Iterator<Item = Nz>) {
        self.fixed_col(col, fix_value, col_cost, NONBASIC, col_vec);
    }

    pub fn redundant_row(&mut self, row: i32) {
        let b = record_bytes!(RedundantRow { row: self.orig_row_index[row as usize] });
        self.buf.extend_from_slice(&b);
        self.added(REDUNDANT_ROW);
    }

    pub fn forcing_row(&mut self, row: i32, row_vec: impl Iterator<Item = Nz>, side: f64, row_type: i32) {
        let b = record_bytes!(ForcingRow { side: side, row: self.orig_row_index[row as usize], row_type: row_type });
        self.buf.extend_from_slice(&b);
        self.push_vec(row_vec, Map::Col);
        self.added(FORCING_ROW);
    }

    pub fn forcing_column(
        &mut self,
        col: i32,
        col_vec: impl Iterator<Item = Nz>,
        cost: f64,
        bound_val: f64,
        at_infinite_upper: bool,
        col_integral: bool,
    ) {
        let b = record_bytes!(ForcingColumn {
            col_cost: cost,
            col_bound: bound_val,
            col: self.orig_col_index[col as usize],
            at_infinite_upper: at_infinite_upper as u8,
            col_integral: col_integral as u8,
        });
        self.buf.extend_from_slice(&b);
        self.push_vec(col_vec, Map::Row);
        self.added(FORCING_COLUMN);
    }

    pub fn forcing_column_removed_row(
        &mut self,
        forcing_col: i32,
        row: i32,
        rhs: f64,
        row_vec: impl Iterator<Item = Nz>,
    ) {
        let b = record_bytes!(ForcingColumnRemovedRow { rhs: rhs, row: self.orig_row_index[row as usize] });
        self.buf.extend_from_slice(&b);
        self.push_vec(row_vec.filter(|&(i, _)| i != forcing_col), Map::Col);
        self.added(FORCING_COLUMN_REMOVED_ROW);
    }

    pub fn duplicate_row(
        &mut self,
        row: i32,
        row_upper_tightened: bool,
        row_lower_tightened: bool,
        duplicate_row: i32,
        duplicate_row_scale: f64,
    ) {
        let b = record_bytes!(DuplicateRow {
            duplicate_row_scale: duplicate_row_scale,
            duplicate_row: self.orig_row_index[duplicate_row as usize],
            row: self.orig_row_index[row as usize],
            row_lower_tightened: row_lower_tightened as u8,
            row_upper_tightened: row_upper_tightened as u8,
        });
        self.buf.extend_from_slice(&b);
        self.added(DUPLICATE_ROW);
    }

    /// HighsPostsolveStack::duplicateColumn: false (and nothing recorded) if
    /// the merge would be illegal
    #[allow(clippy::too_many_arguments)]
    pub fn duplicate_column(
        &mut self,
        col_scale: f64,
        col_lower: f64,
        col_upper: f64,
        duplicate_col_lower: f64,
        duplicate_col_upper: f64,
        col: i32,
        duplicate_col: i32,
        col_integral: bool,
        duplicate_col_integral: bool,
        ok_merge_tolerance: f64,
    ) -> bool {
        let orig_col = self.orig_col_index[col as usize];
        let orig_dup = self.orig_col_index[duplicate_col as usize];
        let r = DuplicateColumn {
            col_scale,
            col_lower,
            col_upper,
            duplicate_col_lower,
            duplicate_col_upper,
            col: orig_col,
            duplicate_col: orig_dup,
            col_integral: col_integral as u8,
            duplicate_col_integral: duplicate_col_integral as u8,
        };
        if !r.ok_merge(ok_merge_tolerance) {
            return false;
        }
        let b = record_bytes!(DuplicateColumn {
            col_scale: col_scale,
            col_lower: col_lower,
            col_upper: col_upper,
            duplicate_col_lower: duplicate_col_lower,
            duplicate_col_upper: duplicate_col_upper,
            col: orig_col,
            duplicate_col: orig_dup,
            col_integral: col_integral as u8,
            duplicate_col_integral: duplicate_col_integral as u8,
        });
        self.buf.extend_from_slice(&b);
        self.added(DUPLICATE_COLUMN);
        self.not_transformable.push(orig_col);
        self.not_transformable.push(orig_dup);
        true
    }
}

#[derive(Clone, Copy)]
enum Map {
    Col,
    Row,
    None,
}
