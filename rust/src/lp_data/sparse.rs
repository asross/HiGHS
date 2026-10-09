//! HighsSparseMatrix's layout changes and edits (util/HighsSparseMatrix.cpp)
//! on a matrix whose vectors are Rust's ([`Vec`], the Rust LP's matrix) or
//! C++ std::vectors that Rust resizes ([`RsVec`], a HighsSparseMatrix
//! passed by C++): [`Mat`] is generic over the vector type ([`Buf`]), so
//! both run the same code. Every step is the C++'s, including how the
//! vectors are resized, so a C++ matrix ends up as the C++ left it.

use super::edit::{update_out_in_index, OutIn};
use super::ffi::RsVec;
use super::lp_utils::IndexCollection;
use super::matrix_format;

/// A vector that the matrix code resizes: std::vector's resize (new
/// elements are zero), assign and push_back
pub trait Buf<T: Copy + Default> {
    fn len(&self) -> usize;
    fn resize(&mut self, n: usize);
    fn sl(&self) -> &[T];
    fn sl_mut(&mut self) -> &mut [T];
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// std::vector::assign(n, x)
    fn assign(&mut self, n: usize, x: T) {
        self.resize(0);
        self.resize(n);
        for v in self.sl_mut() {
            *v = x;
        }
    }
    fn push(&mut self, x: T) {
        let n = self.len();
        self.resize(n + 1);
        self.sl_mut()[n] = x;
    }
    fn clear(&mut self) {
        self.resize(0);
    }
}

impl<T: Copy + Default> Buf<T> for Vec<T> {
    fn len(&self) -> usize {
        Vec::len(self)
    }
    fn resize(&mut self, n: usize) {
        Vec::resize(self, n, T::default());
    }
    fn sl(&self) -> &[T] {
        self
    }
    fn sl_mut(&mut self) -> &mut [T] {
        self
    }
    fn push(&mut self, x: T) {
        Vec::push(self, x);
    }
}

impl<T: Copy + Default> Buf<T> for RsVec<T> {
    fn len(&self) -> usize {
        RsVec::len(self)
    }
    fn resize(&mut self, n: usize) {
        RsVec::resize(self, n);
    }
    fn sl(&self) -> &[T] {
        self.as_slice()
    }
    fn sl_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

/// HighsSparseMatrix over vectors of type `I` (indices) and `F` (values)
#[repr(C)]
#[derive(Clone, Debug, PartialEq)]
pub struct Mat<I, F> {
    pub format: i32,
    pub num_col: i32,
    pub num_row: i32,
    pub start: I,
    pub p_end: I,
    pub index: I,
    pub value: F,
}

/// A Rust-owned HighsSparseMatrix
pub type SparseMatrix = Mat<Vec<i32>, Vec<f64>>;

impl Default for SparseMatrix {
    /// HighsSparseMatrix() (clear)
    fn default() -> Self {
        Mat {
            format: matrix_format::COLWISE,
            num_col: 0,
            num_row: 0,
            start: vec![0],
            p_end: Vec::new(),
            index: Vec::new(),
            value: Vec::new(),
        }
    }
}

impl<I: Buf<i32>, F: Buf<f64>> Mat<I, F> {
    pub fn is_colwise(&self) -> bool {
        self.format == matrix_format::COLWISE
    }
    pub fn is_rowwise(&self) -> bool {
        self.format == matrix_format::ROWWISE || self.format == matrix_format::ROWWISE_PARTITIONED
    }
    /// HighsSparseMatrix::numNz
    pub fn num_nz(&self) -> i32 {
        let n = if self.is_colwise() { self.num_col } else { self.num_row };
        self.start.sl().get(n as usize).copied().unwrap_or(0)
    }

    /// HighsSparseMatrix::clear
    pub fn clear(&mut self) {
        self.num_col = 0;
        self.num_row = 0;
        self.start.clear();
        self.p_end.clear();
        self.index.clear();
        self.value.clear();
        self.format = matrix_format::COLWISE;
        self.start.assign(1, 0);
    }

    /// HighsSparseMatrix::exactResize
    pub fn exact_resize(&mut self) {
        let n = if self.is_colwise() { self.num_col } else { self.num_row } as usize;
        self.start.resize(n + 1);
        let num_nz = self.start.sl()[n] as usize;
        if self.format == matrix_format::ROWWISE_PARTITIONED {
            self.p_end.resize(self.num_row as usize);
        } else {
            self.p_end.clear();
        }
        self.index.resize(num_nz);
        self.value.resize(num_nz);
    }

    /// The transpose of the current orientation into `num_vec` vectors
    /// of the other (the body of ensureColwise / ensureRowwise)
    fn transpose_in_place(&mut self, num_vec: usize, num_from: usize) {
        let num_nz = self.num_nz() as usize;
        if num_nz == 0 {
            self.start.assign(num_vec + 1, 0);
            self.index.clear();
            self.value.clear();
            return;
        }
        let from_start: Vec<i32> = self.start.sl().to_vec();
        let from_index: Vec<i32> = self.index.sl().to_vec();
        let from_value: Vec<f64> = self.value.sl().to_vec();
        self.start.resize(num_vec + 1);
        self.index.resize(num_nz);
        self.value.resize(num_nz);
        let mut length = vec![0i32; num_vec];
        for &i in &from_index[from_start[0] as usize..num_nz] {
            length[i as usize] += 1;
        }
        let start = self.start.sl_mut();
        start[0] = 0;
        for v in 0..num_vec {
            start[v + 1] = start[v] + length[v];
        }
        {
            let index = self.index.sl_mut();
            let value = self.value.sl_mut();
            for f in 0..num_from {
                for el in from_start[f] as usize..from_start[f + 1] as usize {
                    let v = from_index[el] as usize;
                    let to = start[v] as usize;
                    index[to] = f as i32;
                    value[to] = from_value[el];
                    start[v] += 1;
                }
            }
        }
        start[0] = 0;
        for v in 0..num_vec {
            start[v + 1] = start[v] + length[v];
        }
    }

    /// HighsSparseMatrix::ensureColwise
    pub fn ensure_colwise(&mut self) {
        if self.is_colwise() {
            return;
        }
        self.transpose_in_place(self.num_col as usize, self.num_row as usize);
        self.format = matrix_format::COLWISE;
    }

    /// HighsSparseMatrix::ensureRowwise
    pub fn ensure_rowwise(&mut self) {
        if self.is_rowwise() {
            return;
        }
        self.transpose_in_place(self.num_row as usize, self.num_col as usize);
        self.format = matrix_format::ROWWISE;
    }

    /// HighsSparseMatrix::addVec
    pub fn add_vec(&mut self, index: &[i32], value: &[f64], multiple: f64) {
        let num_vec = if self.is_colwise() { self.num_col } else { self.num_row } as usize;
        for (&i, &v) in index.iter().zip(value) {
            self.index.push(i);
            self.value.push(multiple * v);
        }
        let s = self.start.sl()[num_vec] + index.len() as i32;
        self.start.push(s);
        if self.is_colwise() {
            self.num_col += 1;
        } else {
            self.num_row += 1;
        }
    }

    /// HighsSparseMatrix::addCols of a column-wise matrix (not to a
    /// partitioned one)
    pub fn add_cols(&mut self, new_start: &[i32], new_index: &[i32], new_value: &[f64], num_new_col: i32) {
        debug_assert!(self.format != matrix_format::ROWWISE_PARTITIONED);
        if num_new_col == 0 {
            return;
        }
        let num_new_nz = if num_new_col > 0 && !new_start.is_empty() { new_start[num_new_col as usize] } else { 0 };
        let num_col = self.num_col as usize;
        let num_row = self.num_row as usize;
        let num_nz = self.num_nz();
        if self.format == matrix_format::ROWWISE && num_new_nz > num_nz {
            self.ensure_colwise();
        }
        let new_num_col = num_col + num_new_col as usize;
        let new_num_nz = (num_nz + num_new_nz) as usize;
        if self.is_colwise() {
            self.start.resize(new_num_col + 1);
            let start = self.start.sl_mut();
            for k in 0..num_new_col as usize {
                start[num_col + k] = if num_new_nz != 0 { num_nz + new_start[k] } else { num_nz };
            }
            start[new_num_col] = new_num_nz as i32;
            self.num_col += num_new_col;
            if num_new_nz <= 0 {
                return;
            }
            self.index.resize(new_num_nz);
            self.value.resize(new_num_nz);
            let nz = num_nz as usize;
            let n = num_new_nz as usize;
            self.index.sl_mut()[nz..nz + n].copy_from_slice(&new_index[..n]);
            self.value.sl_mut()[nz..nz + n].copy_from_slice(&new_value[..n]);
        } else {
            if num_new_nz != 0 {
                self.index.resize(new_num_nz);
                self.value.resize(new_num_nz);
                let mut new_row_length = vec![0i32; num_row];
                for &i in &new_index[..num_new_nz as usize] {
                    new_row_length[i as usize] += 1;
                }
                let start = self.start.sl_mut();
                let index = self.index.sl_mut();
                let value = self.value.sl_mut();
                let mut entry_offset = num_new_nz;
                let mut to_original_el = start[num_row];
                start[num_row] = new_num_nz as i32;
                for row in (0..num_row).rev() {
                    entry_offset -= new_row_length[row];
                    let from_original_el = start[row];
                    new_row_length[row] = to_original_el + entry_offset;
                    let mut el = to_original_el - 1;
                    while el >= from_original_el {
                        index[(el + entry_offset) as usize] = index[el as usize];
                        value[(el + entry_offset) as usize] = value[el as usize];
                        el -= 1;
                    }
                    to_original_el = from_original_el;
                    start[row] = entry_offset + from_original_el;
                }
                for c in 0..num_new_col as usize {
                    for el in new_start[c] as usize..new_start[c + 1] as usize {
                        let row = new_index[el] as usize;
                        let to = new_row_length[row] as usize;
                        index[to] = (num_col + c) as i32;
                        value[to] = new_value[el];
                        new_row_length[row] += 1;
                    }
                }
            }
            self.num_col += num_new_col;
        }
    }

    /// HighsSparseMatrix::addRows of a row-wise matrix (not to a
    /// partitioned one)
    pub fn add_rows(&mut self, new_start: &[i32], new_index: &[i32], new_value: &[f64], num_new_row: i32) {
        debug_assert!(self.format != matrix_format::ROWWISE_PARTITIONED);
        if num_new_row == 0 {
            return;
        }
        let num_new_nz = if num_new_row > 0 && !new_start.is_empty() { new_start[num_new_row as usize] } else { 0 };
        let num_col = self.num_col as usize;
        let num_row = self.num_row as usize;
        let num_nz = self.num_nz();
        if self.is_colwise() && num_new_nz > num_nz {
            self.ensure_rowwise();
        }
        let new_num_nz = (num_nz + num_new_nz) as usize;
        let new_num_row = num_row + num_new_row as usize;
        if self.is_rowwise() {
            self.start.resize(new_num_row + 1);
            let start = self.start.sl_mut();
            for k in 0..num_new_row as usize {
                start[num_row + k] = if num_new_nz != 0 { num_nz + new_start[k] } else { num_nz };
            }
            start[new_num_row] = new_num_nz as i32;
            if num_new_nz > 0 {
                self.index.resize(new_num_nz);
                self.value.resize(new_num_nz);
                let nz = num_nz as usize;
                let n = num_new_nz as usize;
                self.index.sl_mut()[nz..nz + n].copy_from_slice(&new_index[..n]);
                self.value.sl_mut()[nz..nz + n].copy_from_slice(&new_value[..n]);
            }
        } else if num_new_nz != 0 {
            let mut length = vec![0i32; num_col];
            for &i in &new_index[..num_new_nz as usize] {
                length[i as usize] += 1;
            }
            self.index.resize(new_num_nz);
            self.value.resize(new_num_nz);
            let start = self.start.sl_mut();
            let index = self.index.sl_mut();
            let value = self.value.sl_mut();
            let mut new_el = new_num_nz as i32;
            for col in (0..num_col).rev() {
                let start_col_plus_1 = new_el;
                new_el -= length[col];
                let mut el = start[col + 1] - 1;
                while el >= start[col] {
                    new_el -= 1;
                    index[new_el as usize] = index[el as usize];
                    value[new_el as usize] = value[el as usize];
                    el -= 1;
                }
                start[col + 1] = start_col_plus_1;
            }
            for r in 0..num_new_row as usize {
                let first = new_start[r] as usize;
                let last = if r + 1 < num_new_row as usize { new_start[r + 1] as usize } else { num_new_nz as usize };
                for el in first..last {
                    let col = new_index[el] as usize;
                    let to = (start[col + 1] - length[col]) as usize;
                    length[col] -= 1;
                    index[to] = (num_row + r) as i32;
                    value[to] = new_value[el];
                }
            }
        }
        self.num_row += num_new_row;
    }

    /// HighsSparseMatrix::getRow: the number of entries written
    pub fn get_row(&self, row: i32, index: &mut [i32], value: &mut [f64]) -> i32 {
        let mut num_nz = 0usize;
        let (start, idx, val) = (self.start.sl(), self.index.sl(), self.value.sl());
        if self.is_rowwise() {
            let r = row as usize;
            for el in start[r] as usize..start[r + 1] as usize {
                index[num_nz] = idx[el];
                value[num_nz] = val[el];
                num_nz += 1;
            }
        } else {
            for col in 0..self.num_col as usize {
                for el in start[col] as usize..start[col + 1] as usize {
                    if idx[el] == row {
                        index[num_nz] = col as i32;
                        value[num_nz] = val[el];
                        num_nz += 1;
                        break;
                    }
                }
            }
        }
        num_nz as i32
    }

    /// HighsSparseMatrix::deleteCols (column-wise)
    pub fn delete_cols(&mut self, ic: &IndexCollection) {
        let (from_k, to_k) = ic.limits();
        if from_k > to_k {
            return;
        }
        let col_dim = self.num_col;
        let mut s = OutIn::new();
        let mut new_num_col = 0i32;
        let mut new_num_nz = 0i32;
        {
            let start = self.start.sl_mut();
            let index = self.index.sl_mut();
            let value = self.value.sl_mut();
            for k in from_k..=to_k {
                update_out_in_index(ic, &mut s);
                if k == from_k {
                    new_num_col = s.out_from;
                    new_num_nz = start[s.out_from as usize];
                }
                for col in s.out_from..=s.out_to {
                    start[col as usize] = 0;
                }
                let keep_from_el = start[s.in_from as usize];
                for col in s.in_from..=s.in_to {
                    start[new_num_col as usize] = new_num_nz + start[col as usize] - keep_from_el;
                    new_num_col += 1;
                }
                let keep_to_el = start[(s.in_to + 1) as usize];
                for el in keep_from_el..keep_to_el {
                    index[new_num_nz as usize] = index[el as usize];
                    value[new_num_nz as usize] = value[el as usize];
                    new_num_nz += 1;
                }
                if s.in_to >= col_dim - 1 {
                    break;
                }
            }
            start[self.num_col as usize] = 0;
            start[new_num_col as usize] = new_num_nz;
        }
        self.start.resize(new_num_col as usize + 1);
        self.index.resize(new_num_nz as usize);
        self.value.resize(new_num_nz as usize);
        self.num_col = new_num_col;
    }

    /// HighsSparseMatrix::deleteRows (column-wise)
    pub fn delete_rows(&mut self, ic: &IndexCollection) {
        let (from_k, to_k) = ic.limits();
        if from_k > to_k {
            return;
        }
        let row_dim = self.num_row;
        let mut new_index = vec![0i32; self.num_row as usize];
        let mut new_num_row = 0i32;
        if !ic.is_mask {
            let mut s = OutIn::new();
            for k in from_k..=to_k {
                update_out_in_index(ic, &mut s);
                if k == from_k {
                    for row in 0..s.out_from {
                        new_index[row as usize] = new_num_row;
                        new_num_row += 1;
                    }
                }
                for row in s.out_from..=s.out_to {
                    new_index[row as usize] = -1;
                }
                for row in s.in_from..=s.in_to {
                    new_index[row as usize] = new_num_row;
                    new_num_row += 1;
                }
                if s.in_to >= row_dim - 1 {
                    break;
                }
            }
        } else {
            for row in 0..self.num_row as usize {
                if ic.mask[row] != 0 {
                    new_index[row] = -1;
                } else {
                    new_index[row] = new_num_row;
                    new_num_row += 1;
                }
            }
        }
        let mut new_num_nz = 0i32;
        {
            let start = self.start.sl_mut();
            let index = self.index.sl_mut();
            let value = self.value.sl_mut();
            for col in 0..self.num_col as usize {
                let from_el = start[col];
                start[col] = new_num_nz;
                for el in from_el..start[col + 1] {
                    let new_row = new_index[index[el as usize] as usize];
                    if new_row >= 0 {
                        index[new_num_nz as usize] = new_row;
                        value[new_num_nz as usize] = value[el as usize];
                        new_num_nz += 1;
                    }
                }
            }
            start[self.num_col as usize] = new_num_nz;
        }
        self.start.resize(self.num_col as usize + 1);
        self.index.resize(new_num_nz as usize);
        self.value.resize(new_num_nz as usize);
        self.num_row = new_num_row;
    }

    /// HighsSparseMatrix::createRowwise of a column-wise matrix
    pub fn create_rowwise<J: Buf<i32>, G: Buf<f64>>(&mut self, m: &Mat<J, G>) {
        self.create_rowwise_from(m.num_col, m.num_row, m.start.sl(), m.index.sl(), m.value.sl());
    }

    /// HighsSparseMatrix::createRowwise of a column-wise matrix's arrays
    pub fn create_rowwise_from(&mut self, num_col: i32, num_row: i32, a_start: &[i32], a_index: &[i32], a_value: &[f64]) {
        let (nc, nr) = (num_col as usize, num_row as usize);
        let num_nz = a_start[nc] as usize;
        self.start.resize(nr + 1);
        let mut ar_end = vec![0i32; nr];
        for col in 0..nc {
            for el in a_start[col] as usize..a_start[col + 1] as usize {
                ar_end[a_index[el] as usize] += 1;
            }
        }
        {
            let ar_start = self.start.sl_mut();
            ar_start[0] = 0;
            for row in 0..nr {
                ar_start[row + 1] = ar_start[row] + ar_end[row];
                ar_end[row] = ar_start[row];
            }
        }
        self.index.resize(num_nz);
        self.value.resize(num_nz);
        let ar_index = self.index.sl_mut();
        let ar_value = self.value.sl_mut();
        for col in 0..nc {
            for el in a_start[col] as usize..a_start[col + 1] as usize {
                let row = a_index[el] as usize;
                let to = ar_end[row] as usize;
                ar_end[row] += 1;
                ar_index[to] = col as i32;
                ar_value[to] = a_value[el];
            }
        }
        self.format = matrix_format::ROWWISE;
        self.num_col = num_col;
        self.num_row = num_row;
    }

    /// HighsSparseMatrix::createRowwisePartitioned of a column-wise
    /// matrix's arrays: the entries of columns in the partition (all, if
    /// none is given) first in each row
    pub fn create_rowwise_partitioned_from(
        &mut self,
        num_col: i32,
        num_row: i32,
        a_start: &[i32],
        a_index: &[i32],
        a_value: &[f64],
        in_partition: Option<&[i8]>,
    ) {
        let (nc, nr) = (num_col as usize, num_row as usize);
        let num_nz = a_start[nc] as usize;
        let in_part = |col: usize| in_partition.is_none_or(|p| p[col] != 0);
        self.start.resize(nr + 1);
        self.p_end.assign(nr, 0);
        let mut ar_end = vec![0i32; nr];
        {
            let ar_p_end = self.p_end.sl_mut();
            for col in 0..nc {
                let count = if in_part(col) { &mut *ar_p_end } else { &mut ar_end[..] };
                for el in a_start[col] as usize..a_start[col + 1] as usize {
                    count[a_index[el] as usize] += 1;
                }
            }
            let ar_start = self.start.sl_mut();
            ar_start[0] = 0;
            for row in 0..nr {
                ar_start[row + 1] = ar_start[row] + ar_p_end[row] + ar_end[row];
            }
            for row in 0..nr {
                ar_end[row] = ar_start[row] + ar_p_end[row];
                ar_p_end[row] = ar_start[row];
            }
        }
        self.index.resize(num_nz);
        self.value.resize(num_nz);
        let ar_p_end = self.p_end.sl_mut();
        let ar_index = self.index.sl_mut();
        let ar_value = self.value.sl_mut();
        for col in 0..nc {
            let next = if in_part(col) { &mut *ar_p_end } else { &mut ar_end[..] };
            for el in a_start[col] as usize..a_start[col + 1] as usize {
                let row = a_index[el] as usize;
                let to = next[row] as usize;
                next[row] += 1;
                ar_index[to] = col as i32;
                ar_value[to] = a_value[el];
            }
        }
        self.format = matrix_format::ROWWISE_PARTITIONED;
        self.num_col = num_col;
        self.num_row = num_row;
    }

    /// Each entry with its (column, row), for the scalings
    fn for_each_entry(&mut self, mut f: impl FnMut(usize, usize, &mut f64)) {
        let colwise = self.is_colwise();
        let num_vec = if colwise { self.num_col } else { self.num_row } as usize;
        let start = self.start.sl();
        let index = self.index.sl();
        let value = self.value.sl_mut();
        for v in 0..num_vec {
            for el in start[v] as usize..start[v + 1] as usize {
                let o = index[el] as usize;
                let (c, r) = if colwise { (v, o) } else { (o, v) };
                f(c, r, &mut value[el]);
            }
        }
    }

    /// HighsSparseMatrix::applyScale
    pub fn apply_scale(&mut self, col: &[f64], row: &[f64]) {
        self.for_each_entry(|c, r, x| *x *= col[c] * row[r]);
    }

    /// HighsSparseMatrix::applyColScale
    pub fn apply_col_scale(&mut self, col: &[f64]) {
        self.for_each_entry(|c, _, x| *x *= col[c]);
    }

    /// HighsSparseMatrix::applyRowScale
    pub fn apply_row_scale(&mut self, row: &[f64]) {
        self.for_each_entry(|_, r, x| *x *= row[r]);
    }

    /// HighsSparseMatrix::scaleCol (`is_col`) or scaleRow
    pub fn scale_vec(&mut self, is_col: bool, ix: i32, scale: f64) {
        let ix = ix as usize;
        self.for_each_entry(|c, r, x| {
            if (if is_col { c } else { r }) == ix {
                *x *= scale;
            }
        });
    }

    /// HighsSparseMatrix::product (`transpose` false: result = A x) or
    /// productTranspose (result = A^T x), the products fused into the sums
    /// as clang compiles them
    pub fn product(&self, transpose: bool, result: &mut [f64], x: &[f64]) {
        use crate::util::fma::ClangFma;
        let (start, index, value) = (self.start.sl(), self.index.sl(), self.value.sl());
        let colwise = self.is_colwise();
        let num_vec = if colwise { self.num_col } else { self.num_row } as usize;
        // Along the vectors (gathering) or across them (scattering)
        let gather = colwise == transpose;
        for v in 0..num_vec {
            for el in start[v] as usize..start[v + 1] as usize {
                let o = index[el] as usize;
                if gather {
                    result[v] = x[o].mul_add_c(value[el], result[v]);
                } else {
                    result[o] = x[v].mul_add_c(value[el], result[o]);
                }
            }
        }
    }

    /// HighsSparseMatrix::alphaProductPlusY: y += alpha A x (`transpose`
    /// false) or alpha A^T x, as `(alpha * value) * x + y` fused
    pub fn alpha_product_plus_y(&self, alpha: f64, x: &[f64], y: &mut [f64], transpose: bool) {
        use crate::util::fma::ClangFma;
        let (start, index, value) = (self.start.sl(), self.index.sl(), self.value.sl());
        let colwise = self.is_colwise();
        let num_vec = if colwise { self.num_col } else { self.num_row } as usize;
        let gather = colwise == transpose;
        for v in 0..num_vec {
            for el in start[v] as usize..start[v + 1] as usize {
                let o = index[el] as usize;
                let av = alpha * value[el];
                if gather {
                    y[v] = av.mul_add_c(x[o], y[v]);
                } else {
                    y[o] = av.mul_add_c(x[v], y[o]);
                }
            }
        }
    }

    /// HighsSparseMatrix::computeDot of a column-wise matrix: the column's
    /// dot product with `array` (clang vectorizes it by 4: the first
    /// len/4*4 products rounded and summed in order, the rest fused), or
    /// the array's entry of a logical
    pub fn compute_dot(&self, array: &[f64], use_col: i32) -> f64 {
        use crate::util::fma::ClangFma;
        if use_col >= self.num_col {
            return array[(use_col - self.num_col) as usize];
        }
        let (start, index, value) = (self.start.sl(), self.index.sl(), self.value.sl());
        let c = use_col as usize;
        let (from, to) = (start[c] as usize, start[c + 1] as usize);
        let blocked = from + (to - from) / 4 * 4;
        let mut result = 0.0f64;
        for el in from..blocked {
            result += array[index[el] as usize] * value[el];
        }
        for el in blocked..to {
            result = array[index[el] as usize].mul_add_c(value[el], result);
        }
        result
    }

    /// HighsSparseMatrix::productQuad: result = A x in double-double
    pub fn product_quad(&self, result: &mut [f64], x: &[f64]) {
        use crate::util::cdouble::CDouble;
        let (start, index, value) = (self.start.sl(), self.index.sl(), self.value.sl());
        if self.is_colwise() {
            let mut v = vec![CDouble::from(0.0); self.num_row as usize];
            for col in 0..self.num_col as usize {
                for el in start[col] as usize..start[col + 1] as usize {
                    v[index[el] as usize] += x[col] * value[el];
                }
            }
            for (r, q) in result.iter_mut().zip(&v) {
                *r = q.to_f64();
            }
        } else {
            for row in 0..self.num_row as usize {
                let mut v = CDouble::from(0.0);
                for el in start[row] as usize..start[row + 1] as usize {
                    v += x[index[el] as usize] * value[el];
                }
                result[row] = v.to_f64();
            }
        }
    }

    /// HighsSparseMatrix::productTransposeQuad (dense): result = A^T x in
    /// double-double
    pub fn product_transpose_quad(&self, result: &mut [f64], x: &[f64]) {
        use crate::util::cdouble::CDouble;
        let (start, index, value) = (self.start.sl(), self.index.sl(), self.value.sl());
        if self.is_colwise() {
            for col in 0..self.num_col as usize {
                let mut v = CDouble::from(0.0);
                for el in start[col] as usize..start[col + 1] as usize {
                    v += x[index[el] as usize] * value[el];
                }
                result[col] = v.to_f64();
            }
        } else {
            let mut v = vec![CDouble::from(0.0); self.num_col as usize];
            for row in 0..self.num_row as usize {
                for el in start[row] as usize..start[row + 1] as usize {
                    v[index[el] as usize] += x[row] * value[el];
                }
            }
            for (r, q) in result.iter_mut().zip(&v) {
                *r = q.to_f64();
            }
        }
    }

    /// HighsSparseMatrix::hasLargeValue
    pub fn has_large_value(&self, large_matrix_value: f64) -> bool {
        let n = self.num_nz() as usize;
        self.value.sl()[..n].iter().any(|v| v.abs() >= large_matrix_value)
    }
}

/// A C++ HighsSparseMatrix edited in place (HighsRust.h: RsMatVec; C++
/// copies the scalars back)
pub type CppMat = Mat<RsVec<i32>, RsVec<f64>>;

/// The methods of the C++ HighsSparseMatrix (util/HighsSparseMatrix.cpp
/// under HIGHS_RUST)
pub mod ffi {
    use super::*;
    use crate::ffi::{sl, sl_mut};
    use crate::lp_data::ffi::{CIndexCollection, CMatrix};

    /// ensureColwise (0), ensureRowwise (1), exactResize (2)
    ///
    /// # Safety
    /// `m` a valid view
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_mat_layout(m: *mut CppMat, what: i32) {
        let m = &mut *m;
        match what {
            0 => m.ensure_colwise(),
            1 => m.ensure_rowwise(),
            _ => m.exact_resize(),
        }
    }

    /// addVec
    ///
    /// # Safety
    /// `m` a valid view, the arrays of `num_nz` entries
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_mat_add_vec(
        m: *mut CppMat,
        num_nz: i32,
        index: *const i32,
        value: *const f64,
        multiple: f64,
    ) {
        (*m).add_vec(sl(index, num_nz), sl(value, num_nz), multiple);
    }

    /// addCols (`cols`, the new columns column-wise) or addRows (the new
    /// rows row-wise)
    ///
    /// # Safety
    /// `m` a valid view, `new` a valid matrix
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_mat_add(m: *mut CppMat, cols: bool, new: *const CMatrix) {
        let new = &*new;
        let num_new = if cols { new.num_col } else { new.num_row };
        let (start, index, value) = (new.start.get(), new.index.get(), new.value.get());
        if cols {
            (*m).add_cols(start, index, value, num_new);
        } else {
            (*m).add_rows(start, index, value, num_new);
        }
    }

    /// getRow: the number of entries written
    ///
    /// # Safety
    /// `m` a valid view, the outputs large enough for a row
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_mat_get_row(m: *const CppMat, row: i32, index: *mut i32, value: *mut f64) -> i32 {
        let m = &*m;
        let n = m.num_col.max(0);
        m.get_row(row, sl_mut(index, n), sl_mut(value, n))
    }

    /// deleteCols (`cols`) or deleteRows
    ///
    /// # Safety
    /// `m` and `ic` valid
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_mat_delete(m: *mut CppMat, cols: bool, ic: *const CIndexCollection) {
        let ic = (*ic).view();
        if cols {
            (*m).delete_cols(&ic);
        } else {
            (*m).delete_rows(&ic);
        }
    }

    /// createRowwise (not `partitioned`) or createRowwisePartitioned
    /// (`partition` null: all columns) of a column-wise matrix
    ///
    /// # Safety
    /// `m` and `from` valid, `partition` null or of from's column count
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_mat_create_rowwise(
        m: *mut CppMat,
        from: *const CMatrix,
        partitioned: bool,
        partition: *const i8,
    ) {
        let a = &*from;
        let (start, index, value) = (a.start.get(), a.index.get(), a.value.get());
        if partitioned {
            let p = if partition.is_null() { None } else { Some(sl(partition, a.num_col)) };
            (*m).create_rowwise_partitioned_from(a.num_col, a.num_row, start, index, value, p);
        } else {
            (*m).create_rowwise_from(a.num_col, a.num_row, start, index, value);
        }
    }

    /// hasLargeValue
    ///
    /// # Safety
    /// `m` valid
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_mat_has_large_value(m: *const CppMat, large_matrix_value: f64) -> bool {
        (*m).has_large_value(large_matrix_value)
    }

    /// applyScale (0), applyColScale (1), applyRowScale (2), scaleCol (3:
    /// `ix` by `s`), scaleRow (4)
    ///
    /// # Safety
    /// `m` valid, the scale factors of its dimensions
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_mat_scale(m: *mut CppMat, what: i32, col: *const f64, row: *const f64, ix: i32, s: f64) {
        let m = &mut *m;
        let (nc, nr) = (m.num_col, m.num_row);
        match what {
            0 => m.apply_scale(sl(col, nc), sl(row, nr)),
            1 => m.apply_col_scale(sl(col, nc)),
            2 => m.apply_row_scale(sl(row, nr)),
            3 => m.scale_vec(true, ix, s),
            _ => m.scale_vec(false, ix, s),
        }
    }

    /// collectAj: `multiplier` times column `use_col` (a logical if past
    /// the columns) of a column-wise matrix added to an HVector's array,
    /// index and count (tiny results become kHighsZero)
    ///
    /// # Safety
    /// `m` valid, the HVector's arrays of its row count
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_mat_collect_aj(
        m: *const CppMat,
        use_col: i32,
        multiplier: f64,
        array: *mut f64,
        index: *mut i32,
        count: *mut i32,
    ) {
        use crate::util::fma::ClangFma;
        const TINY: f64 = 1e-14;
        const ZERO: f64 = 1e-50;
        let m = &*m;
        let nr = m.num_row;
        let (array, index) = (sl_mut(array, nr), sl_mut(index, nr));
        let mut add = |row: usize, v: f64| {
            let value0 = array[row];
            let value1 = multiplier.mul_add_c(v, value0);
            if value0 == 0.0 {
                index[*count as usize] = row as i32;
                *count += 1;
            }
            array[row] = if value1.abs() < TINY { ZERO } else { value1 };
        };
        if use_col < m.num_col {
            let (start, idx, value) = (m.start.sl(), m.index.sl(), m.value.sl());
            let c = use_col as usize;
            for el in start[c] as usize..start[c + 1] as usize {
                add(idx[el] as usize, value[el]);
            }
        } else {
            let row = (use_col - m.num_col) as usize;
            let value0 = array[row];
            let value1 = value0 + multiplier;
            if value0 == 0.0 {
                index[*count as usize] = row as i32;
                *count += 1;
            }
            array[row] = if value1.abs() < TINY { ZERO } else { value1 };
        }
    }

    /// product (`what` 0), productTranspose (1), alphaProductPlusY (2: y
    /// in `result`, 3: transposed)
    ///
    /// # Safety
    /// `m` valid, `x` and `result` of the matrix's dimensions
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_mat_product(m: *const CppMat, what: i32, alpha: f64, x: *const f64, result: *mut f64) {
        let m = &*m;
        let (nc, nr) = (m.num_col, m.num_row);
        let transpose = what & 1 != 0;
        let (nx, nres) = if transpose { (nr, nc) } else { (nc, nr) };
        if what < 2 {
            m.product(transpose, sl_mut(result, nres), sl(x, nx));
        } else {
            m.alpha_product_plus_y(alpha, sl(x, nx), sl_mut(result, nres), transpose);
        }
    }

    /// computeDot
    ///
    /// # Safety
    /// `m` valid, `array` of its row count
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_mat_compute_dot(m: *const CppMat, array: *const f64, n: usize, use_col: i32) -> f64 {
        (*m).compute_dot(sl(array, n as i32), use_col)
    }

    /// productQuad (not `transpose`) or the dense productTransposeQuad
    ///
    /// # Safety
    /// `m` valid, `x` and `result` of the matrix's dimensions
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_mat_product_quad(m: *const CppMat, transpose: bool, x: *const f64, result: *mut f64) {
        let m = &*m;
        let (nc, nr) = (m.num_col, m.num_row);
        if transpose {
            m.product_transpose_quad(sl_mut(result, nc), sl(x, nr));
        } else {
            m.product_quad(sl_mut(result, nr), sl(x, nc));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn colwise() -> SparseMatrix {
        // [1 0 2; 0 3 4]
        Mat {
            format: matrix_format::COLWISE,
            num_col: 3,
            num_row: 2,
            start: vec![0, 1, 2, 4],
            p_end: vec![],
            index: vec![0, 1, 0, 1],
            value: vec![1.0, 3.0, 2.0, 4.0],
        }
    }

    #[test]
    fn transpose_round_trip() {
        let mut m = colwise();
        m.ensure_rowwise();
        assert_eq!(m.start, vec![0, 2, 4]);
        assert_eq!(m.index, vec![0, 2, 1, 2]);
        assert_eq!(m.value, vec![1.0, 2.0, 3.0, 4.0]);
        m.ensure_colwise();
        assert_eq!(m, colwise());
    }

    #[test]
    fn add_and_delete() {
        let mut m = colwise();
        // a row [5 0 6]
        m.add_rows(&[0, 2], &[0, 2], &[5.0, 6.0], 1);
        assert_eq!(m.num_row, 3);
        assert_eq!(m.start, vec![0, 2, 3, 6]);
        assert_eq!(m.index, vec![0, 2, 1, 0, 1, 2]);
        let mask = [0, 0, 1];
        let ic = IndexCollection {
            dimension: 3,
            is_interval: false,
            from: -1,
            to: -2,
            is_set: false,
            set_num_entries: -1,
            set: &[],
            is_mask: true,
            mask: &mask,
        };
        m.delete_rows(&ic);
        assert_eq!(m, colwise());
        m.add_cols(&[0, 1], &[1], &[7.0], 1);
        assert_eq!(m.num_col, 4);
        m.delete_cols(&IndexCollection::interval(4, 3, 3));
        assert_eq!(m, colwise());
        let mut r = SparseMatrix::default();
        r.create_rowwise(&m);
        let mut t = colwise();
        t.ensure_rowwise();
        assert_eq!(r, t);
    }
}
