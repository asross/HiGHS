//! The views of the C++ data (highs/lp_data/HighsRust.h) and the
//! `extern "C"` entry points of lp_data. Every array comes with its
//! length; C++ sizes the vectors before a call, Rust never resizes them
//! (a call returns a new length where C++ must shrink a vector).

use super::lp_utils::{self, IndexCollection, MatrixMut};
use super::Log;
use std::slice::{from_raw_parts, from_raw_parts_mut};

/// A C++ array: a std::vector's data() and size()
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RsMut<T> {
    pub ptr: *mut T,
    pub len: usize,
}

impl<T> RsMut<T> {
    /// # Safety
    /// `ptr` must be valid for `len` reads (or `len == 0`), not written
    /// while the slice lives
    pub unsafe fn get<'a>(&self) -> &'a [T] {
        if self.len == 0 || self.ptr.is_null() {
            &[]
        } else {
            from_raw_parts(self.ptr, self.len)
        }
    }
    /// # Safety
    /// `ptr` must be valid for `len` reads and writes (or `len == 0`),
    /// unaliased while the slice lives
    pub unsafe fn get_mut<'a>(&self) -> &'a mut [T] {
        if self.len == 0 || self.ptr.is_null() {
            &mut []
        } else {
            from_raw_parts_mut(self.ptr, self.len)
        }
    }
}

/// A C++ std::vector that Rust may resize (HighsRust.h: rsVec): its
/// address, a function resizing it as std::vector::resize does (new
/// elements are zero) and returning its data(), and its data and size
#[repr(C)]
pub struct RsVec<T> {
    vec: *mut std::ffi::c_void,
    resize_fn: unsafe extern "C" fn(*mut std::ffi::c_void, usize) -> *mut T,
    ptr: *mut T,
    len: usize,
}

impl<T: Copy + Default> RsVec<T> {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn as_slice(&self) -> &[T] {
        if self.len == 0 {
            &[]
        } else {
            // SAFETY: the C++ vector holds len elements at ptr
            unsafe { from_raw_parts(self.ptr, self.len) }
        }
    }
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.len == 0 {
            &mut []
        } else {
            // SAFETY: as as_slice, and &mut self makes it unaliased
            unsafe { from_raw_parts_mut(self.ptr, self.len) }
        }
    }
    /// std::vector::resize
    pub fn resize(&mut self, n: usize) {
        // SAFETY: the C++ function resizes the vector it was given with
        self.ptr = unsafe { (self.resize_fn)(self.vec, n) };
        self.len = n;
    }
    /// std::vector::assign of a range
    pub fn assign(&mut self, v: &[T]) {
        self.resize(0);
        self.resize(v.len());
        self.as_mut_slice().copy_from_slice(v);
    }
    pub fn push(&mut self, x: T) {
        let n = self.len;
        self.resize(n + 1);
        self.as_mut_slice()[n] = x;
    }
    pub fn to_vec(&self) -> Vec<T> {
        self.as_slice().to_vec()
    }
}

#[cfg(test)]
pub mod rs_vec_test {
    //! RsVec over a Rust Vec, for tests
    use super::RsVec;
    unsafe extern "C" fn resize<T: Copy + Default>(v: *mut std::ffi::c_void, n: usize) -> *mut T {
        let v = &mut *(v as *mut Vec<T>);
        v.resize(n, T::default());
        v.as_mut_ptr()
    }
    pub fn rs_vec<T: Copy + Default>(v: &mut Vec<T>) -> RsVec<T> {
        RsVec { vec: v as *mut Vec<T> as *mut _, resize_fn: resize::<T>, ptr: v.as_mut_ptr(), len: v.len() }
    }
}

/// A name as "%s" prints it (HighsRust.h: RsNameList)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RsName {
    pub ptr: *const u8,
    pub len: usize,
}

impl RsName {
    pub fn bytes(&self) -> &[u8] {
        if self.len == 0 {
            &[]
        } else {
            // SAFETY: the C++ string lives with its RsNameList
            unsafe { from_raw_parts(self.ptr, self.len) }
        }
    }
    /// The name as Rust's printf takes it (non-UTF-8 bytes replaced)
    pub fn text(&self) -> std::borrow::Cow<'_, str> {
        String::from_utf8_lossy(self.bytes())
    }
}

/// HighsSparseMatrix
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CMatrix {
    pub format: i32,
    pub num_col: i32,
    pub num_row: i32,
    pub start: RsMut<i32>,
    pub p_end: RsMut<i32>,
    pub index: RsMut<i32>,
    pub value: RsMut<f64>,
}

impl CMatrix {
    /// # Safety
    /// The arrays must be valid and unaliased while the view lives
    pub unsafe fn view<'a>(&self) -> MatrixMut<'a> {
        MatrixMut {
            format: self.format,
            num_col: self.num_col,
            num_row: self.num_row,
            start: self.start.get_mut(),
            p_end: self.p_end.get_mut(),
            index: self.index.get_mut(),
            value: self.value.get_mut(),
        }
    }
}

/// HighsLp's numerical data and scaling (HighsRust.h: rsLp). Scalars that
/// Rust changes are copied back by C++ (rsLpBack).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CLp {
    pub num_col: i32,
    pub num_row: i32,
    pub col_cost: RsMut<f64>,
    pub col_lower: RsMut<f64>,
    pub col_upper: RsMut<f64>,
    pub row_lower: RsMut<f64>,
    pub row_upper: RsMut<f64>,
    pub a: CMatrix,
    pub sense: i32,
    pub offset: f64,
    pub integrality: RsMut<u8>,
    pub scale_strategy: i32,
    pub scale_has_scaling: bool,
    pub scale_num_col: i32,
    pub scale_num_row: i32,
    pub scale_cost: f64,
    pub scale_col: RsMut<f64>,
    pub scale_row: RsMut<f64>,
    pub is_scaled: bool,
    pub is_moved: bool,
    pub has_infinite_cost: bool,
}

/// HighsIndexCollection
#[repr(C)]
pub struct CIndexCollection {
    pub dimension: i32,
    pub is_interval: bool,
    pub from: i32,
    pub to: i32,
    pub is_set: bool,
    pub set_num_entries: i32,
    pub set: RsMut<i32>,
    pub is_mask: bool,
    pub mask: RsMut<i32>,
}

impl CIndexCollection {
    /// # Safety
    /// The arrays must be valid while the view lives
    pub unsafe fn view<'a>(&self) -> IndexCollection<'a> {
        IndexCollection {
            dimension: self.dimension,
            is_interval: self.is_interval,
            from: self.from,
            to: self.to,
            is_set: self.is_set,
            set_num_entries: self.set_num_entries,
            set: self.set.get(),
            is_mask: self.is_mask,
            mask: self.mask.get(),
        }
    }
}

/// The options lp_utils reads
#[repr(C)]
pub struct CLpOptions {
    pub log: Log,
    pub infinite_cost: f64,
    pub infinite_bound: f64,
    pub small_matrix_value: f64,
    pub large_matrix_value: f64,
    pub simplex_scale_strategy: i32,
    pub allowed_matrix_scale_factor: i32,
    pub highs_analysis_level: i32,
    pub log_dev_level: i32,
    pub primal_feasibility_tolerance: f64,
}

/// assessLp; C++ shrinks the matrix's index and value to the returned
/// number of nonzeros
///
/// # Safety
/// The views must be valid
#[no_mangle]
pub unsafe extern "C" fn highs_rs_assess_lp(lp: *mut CLp, o: *const CLpOptions) -> i32 {
    lp_utils::assess_lp(&mut *lp, &*o) as i32
}

/// assessCosts
///
/// # Safety
/// The views must be valid
#[no_mangle]
pub unsafe extern "C" fn highs_rs_assess_costs(
    o: *const CLpOptions,
    ml_col_os: i32,
    ic: *const CIndexCollection,
    cost: RsMut<f64>,
    has_infinite_cost: *mut bool,
    infinite_cost: f64,
) -> i32 {
    let _ = ml_col_os;
    lp_utils::assess_costs(&(*o).log, &(*ic).view(), cost.get_mut(), &mut *has_infinite_cost, infinite_cost)
        as i32
}

/// assessBounds; `integrality` may be empty (no integrality)
///
/// # Safety
/// The views must be valid, `kind` NUL-free of length `kind_len`
#[no_mangle]
pub unsafe extern "C" fn highs_rs_assess_bounds(
    o: *const CLpOptions,
    kind: *const u8,
    kind_len: usize,
    ml_ix_os: i32,
    ic: *const CIndexCollection,
    lower: RsMut<f64>,
    upper: RsMut<f64>,
    infinite_bound: f64,
    integrality: RsMut<u8>,
) -> i32 {
    let kind = std::str::from_utf8_unchecked(from_raw_parts(kind, kind_len));
    let integrality = integrality.get();
    lp_utils::assess_bounds(
        &(*o).log,
        kind,
        ml_ix_os,
        &(*ic).view(),
        lower.get_mut(),
        upper.get_mut(),
        infinite_bound,
        if integrality.is_empty() { None } else { Some(integrality) },
    ) as i32
}

/// assessMatrix; the new number of nonzeros is start[num_vec]
///
/// # Safety
/// The views must be valid
#[no_mangle]
pub unsafe extern "C" fn highs_rs_assess_matrix(
    log: *const Log,
    name: *const u8,
    name_len: usize,
    vec_dim: i32,
    num_vec: i32,
    partitioned: bool,
    start: RsMut<i32>,
    p_end: RsMut<i32>,
    index: RsMut<i32>,
    value: RsMut<f64>,
    small_matrix_value: f64,
    large_matrix_value: f64,
    sum_duplicates: bool,
) -> i32 {
    let name = std::str::from_utf8_unchecked(from_raw_parts(name, name_len));
    lp_utils::assess_matrix(
        &*log,
        name,
        vec_dim,
        num_vec,
        partitioned,
        start.get_mut(),
        p_end.get(),
        index.get_mut(),
        value.get_mut(),
        small_matrix_value,
        large_matrix_value,
        sum_duplicates,
    ) as i32
}

/// scaleLp after C++'s lp.clearScaling() and with scale.col/row sized
/// num_col/num_row; returns whether the LP is scaled (else C++ clears the
/// scaling again)
///
/// # Safety
/// The views must be valid
#[no_mangle]
pub unsafe extern "C" fn highs_rs_scale_lp(lp: *mut CLp, o: *const CLpOptions, force_scaling: bool) -> bool {
    lp_utils::scale_lp(&mut *lp, &*o, force_scaling)
}

/// HighsLp::applyScale (apply) or unapplyScale
///
/// # Safety
/// The view must be valid
#[no_mangle]
pub unsafe extern "C" fn highs_rs_lp_apply_scale(lp: *mut CLp, apply: bool) {
    if apply {
        lp_utils::apply_scale(&mut *lp)
    } else {
        lp_utils::unapply_scale(&mut *lp)
    }
}

/// cleanBounds
///
/// # Safety
/// The views must be valid
#[no_mangle]
pub unsafe extern "C" fn highs_rs_clean_bounds(lp: *mut CLp, o: *const CLpOptions) -> i32 {
    lp_utils::clean_bounds(&mut *lp, &*o) as i32
}

/// lpDimensionsOk
///
/// # Safety
/// The views must be valid
#[no_mangle]
pub unsafe extern "C" fn highs_rs_lp_dimensions_ok(
    log: *const Log,
    message: *const u8,
    message_len: usize,
    lp: *const CLp,
) -> bool {
    let message = std::str::from_utf8_unchecked(from_raw_parts(message, message_len));
    lp_utils::lp_dimensions_ok(&*log, message, &*lp)
}

/// assessMatrixDimensions
///
/// # Safety
/// The views must be valid
#[no_mangle]
pub unsafe extern "C" fn highs_rs_assess_matrix_dimensions(
    log: *const Log,
    num_vec: i32,
    partitioned: bool,
    start: RsMut<i32>,
    p_end: RsMut<i32>,
    index_size: usize,
    value_size: usize,
) -> i32 {
    lp_utils::assess_matrix_dimensions(&*log, num_vec, partitioned, start.get(), p_end.get(), index_size, value_size)
        as i32
}
