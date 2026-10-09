//! `extern "C"` shims for HFactor (highs/util/HFactor.cpp under
//! HIGHS_RUST). Every pointer comes with its length, so the Rust side is
//! bounds-checked.

use crate::factor::{AMatrix, HFactor};
use crate::hvector::HVec;
use std::slice::{from_raw_parts, from_raw_parts_mut};

/// # Safety
/// `p` must be valid for `n` reads (or `n <= 0`)
pub(crate) unsafe fn sl<'a, T>(p: *const T, n: i32) -> &'a [T] {
    if n <= 0 || p.is_null() {
        &[]
    } else {
        from_raw_parts(p, n as usize)
    }
}

/// # Safety
/// `p` must be valid for `n` reads and writes (or `n <= 0`), unaliased
pub(crate) unsafe fn sl_mut<'a, T>(p: *mut T, n: i32) -> &'a mut [T] {
    if n <= 0 || p.is_null() {
        &mut []
    } else {
        from_raw_parts_mut(p, n as usize)
    }
}

/// An HVector's arrays (with lengths) and scalars
#[repr(C)]
pub struct CHVec {
    size: i32,
    count: i32,
    index: *mut i32,
    n_index: i32,
    array: *mut f64,
    n_array: i32,
    cwork: *mut u8,
    n_cwork: i32,
    iwork: *mut i32,
    n_iwork: i32,
    synthetic_tick: f64,
    pack_flag: u8,
    pack_count: i32,
    pack_index: *mut i32,
    n_pack_index: i32,
    pack_value: *mut f64,
    n_pack_value: i32,
}

impl CHVec {
    /// The view of a Rust-owned HVector (its scalars copied back with
    /// `store_into`)
    pub(crate) fn of(v: &mut crate::hvector::OwnedHVec) -> CHVec {
        CHVec {
            size: v.size,
            count: v.count,
            index: v.index.as_mut_ptr(),
            n_index: v.index.len() as i32,
            array: v.array.as_mut_ptr(),
            n_array: v.array.len() as i32,
            cwork: v.cwork.as_mut_ptr(),
            n_cwork: v.cwork.len() as i32,
            iwork: v.iwork.as_mut_ptr(),
            n_iwork: v.iwork.len() as i32,
            synthetic_tick: v.synthetic_tick,
            pack_flag: v.pack_flag as u8,
            pack_count: v.pack_count,
            pack_index: v.pack_index.as_mut_ptr(),
            n_pack_index: v.pack_index.len() as i32,
            pack_value: v.pack_value.as_mut_ptr(),
            n_pack_value: v.pack_value.len() as i32,
        }
    }

    /// The scalars back into the Rust-owned HVector it views
    pub(crate) fn store_into(&self, v: &mut crate::hvector::OwnedHVec) {
        v.count = self.count;
        v.synthetic_tick = self.synthetic_tick;
        v.pack_flag = self.pack_flag != 0;
        v.pack_count = self.pack_count;
    }

    /// # Safety
    /// The pointers must be valid for their lengths and unaliased
    pub(crate) unsafe fn view<'a>(&self) -> HVec<'a> {
        HVec {
            size: self.size,
            count: self.count,
            index: sl_mut(self.index, self.n_index),
            array: sl_mut(self.array, self.n_array),
            cwork: sl_mut(self.cwork, self.n_cwork),
            iwork: sl_mut(self.iwork, self.n_iwork),
            synthetic_tick: self.synthetic_tick,
            pack_flag: self.pack_flag != 0,
            pack_count: self.pack_count,
            pack_index: sl_mut(self.pack_index, self.n_pack_index),
            pack_value: sl_mut(self.pack_value, self.n_pack_value),
        }
    }

    /// Copy the scalars back from a view
    pub(crate) fn store(&mut self, v: &HVec) {
        self.count = v.count;
        self.synthetic_tick = v.synthetic_tick;
        self.pack_flag = v.pack_flag as u8;
        self.pack_count = v.pack_count;
    }
}

/// The column-wise constraint matrix
#[repr(C)]
pub struct CAMatrix {
    num_col: i32,
    start: *const i32,
    index: *const i32,
    value: *const f64,
}

impl CAMatrix {
    /// # Safety
    /// `start` must hold num_col + 1 entries, and index/value
    /// start[num_col]
    unsafe fn view(&self) -> AMatrix<'_> {
        let start = sl(self.start, self.num_col + 1);
        let nnz = start.last().copied().unwrap_or(0);
        AMatrix {
            num_col: self.num_col,
            start,
            index: sl(self.index, nnz),
            value: sl(self.value, nnz),
        }
    }
}

#[repr(C)]
pub struct CInfo {
    build_synthetic_tick: f64,
    refactor_build_synthetic_tick: f64,
    rank_deficiency: i32,
    basis_matrix_num_el: i32,
    invert_num_el: i32,
    kernel_dim: i32,
    kernel_num_el: i32,
    num_row: i32,
}

#[no_mangle]
pub extern "C" fn highs_rs_factor_new() -> *mut HFactor {
    Box::into_raw(Box::default())
}

/// # Safety
/// `p` must come from highs_rs_factor_new (or be null)
#[no_mangle]
pub unsafe extern "C" fn highs_rs_factor_free(p: *mut HFactor) {
    if !p.is_null() {
        drop(Box::from_raw(p));
    }
}

/// # Safety
/// `a_start` must hold num_col + 1 entries
#[no_mangle]
pub unsafe extern "C" fn highs_rs_factor_setup(
    p: *mut HFactor,
    num_col: i32,
    num_row: i32,
    num_basic: i32,
    a_start: *const i32,
    update_method: i32,
) {
    let a_start = sl(a_start, num_col + 1);
    (*p).setup(num_col, num_row, num_basic, a_start, update_method);
}

/// HFactor::build, with the refactorization information held by the
/// factor (used if it is to be)
///
/// # Safety
/// As for CAMatrix::view; basic_index holds n_basic entries
#[no_mangle]
pub unsafe extern "C" fn highs_rs_factor_build(
    p: *mut HFactor,
    pivot_threshold: f64,
    pivot_tolerance: f64,
    time_limit: f64,
    a: *const CAMatrix,
    basic_index: *mut i32,
    n_basic: i32,
) -> i32 {
    let a = (*a).view();
    let basic_index = sl_mut(basic_index, n_basic);
    (*p).build_with_refactor_info(pivot_threshold, pivot_tolerance, time_limit, &a, basic_index)
}

/// RefactorInfo::clear for the factor's information
///
/// # Safety
/// `p` must be a live factor
#[no_mangle]
pub unsafe extern "C" fn highs_rs_factor_refactor_clear(p: *mut HFactor) {
    (*p).refactor_info_clear();
}

/// Whether the factor's refactorization information is to be used
///
/// # Safety
/// `p` must be a live factor
#[no_mangle]
pub unsafe extern "C" fn highs_rs_factor_refactor_use(p: *const HFactor) -> bool {
    (*p).refactor_use
}

/// Set the factor's refactorization information (RefactorInfo)
///
/// # Safety
/// The arrays must hold n entries
#[no_mangle]
pub unsafe extern "C" fn highs_rs_factor_refactor_set(
    p: *mut HFactor,
    use_: bool,
    pivot_row: *const i32,
    pivot_var: *const i32,
    pivot_type: *const i8,
    n: i32,
    build_synthetic_tick: f64,
) {
    let f = &mut *p;
    f.refactor_use = use_;
    f.refactor_pivot_row = sl(pivot_row, n).to_vec();
    f.refactor_pivot_var = sl(pivot_var, n).to_vec();
    f.refactor_pivot_type = sl(pivot_type, n).to_vec();
    f.refactor_build_synthetic_tick = build_synthetic_tick;
}

/// # Safety
/// `v`'s pointers must be valid for their lengths
#[no_mangle]
pub unsafe extern "C" fn highs_rs_factor_ftran(
    p: *const HFactor,
    v: *mut CHVec,
    expected_density: f64,
) {
    let c = &mut *v;
    let mut h = c.view();
    (*p).ftran(&mut h, expected_density);
    c.store(&h);
}

/// # Safety
/// `v`'s pointers must be valid for their lengths
#[no_mangle]
pub unsafe extern "C" fn highs_rs_factor_btran(
    p: *const HFactor,
    v: *mut CHVec,
    expected_density: f64,
) {
    let c = &mut *v;
    let mut h = c.view();
    (*p).btran(&mut h, expected_density);
    c.store(&h);
}

/// # Safety
/// aq, ep and i_row hold n entries; a may be null (only APF uses it)
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn highs_rs_factor_update(
    p: *mut HFactor,
    aq: *const CHVec,
    ep: *const CHVec,
    n: i32,
    i_row: *const i32,
    hint: *mut i32,
    a: *const CAMatrix,
    basic_index: *const i32,
    n_basic: i32,
) {
    let (aq, ep, i_row) = (sl(aq, n), sl(ep, n), sl(i_row, n));
    let a = a.as_ref().map(|a| a.view());
    let basic_index = sl(basic_index, n_basic);
    if n == 1 {
        // The usual single update: no allocation
        let (aq, ep) = ([aq[0].view()], [ep[0].view()]);
        (*p).update(&aq, &ep, i_row, &mut *hint, a.as_ref(), basic_index);
    } else {
        let aq: Vec<HVec> = aq.iter().map(|c| c.view()).collect();
        let ep: Vec<HVec> = ep.iter().map(|c| c.view()).collect();
        (*p).update(&aq, &ep, i_row, &mut *hint, a.as_ref(), basic_index);
    }
}

/// # Safety
/// ar_start holds num_new_row + 1 entries, ar_index/ar_value
/// ar_start[num_new_row]; basic_index n_basic
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn highs_rs_factor_add_rows(
    p: *mut HFactor,
    num_col: i32,
    basic_index: *const i32,
    n_basic: i32,
    ar_start: *const i32,
    ar_index: *const i32,
    ar_value: *const f64,
    num_new_row: i32,
) {
    let ar_start = sl(ar_start, num_new_row + 1);
    let nnz = ar_start.last().copied().unwrap_or(0);
    (*p).add_rows(
        num_col,
        sl(basic_index, n_basic),
        ar_start,
        sl(ar_index, nnz),
        sl(ar_value, nnz),
    );
}

/// # Safety
/// `p` must be a live factor
#[no_mangle]
pub unsafe extern "C" fn highs_rs_factor_info(p: *const HFactor, out: *mut CInfo) {
    let f = &*p;
    *out = CInfo {
        build_synthetic_tick: f.build_synthetic_tick,
        refactor_build_synthetic_tick: f.refactor_build_synthetic_tick,
        rank_deficiency: f.rank_deficiency,
        basis_matrix_num_el: f.basis_matrix_num_el,
        invert_num_el: f.invert_num_el,
        kernel_dim: f.kernel_dim,
        kernel_num_el: f.kernel_num_el,
        num_row: f.num_row,
    };
}

fn ivec(f: &mut HFactor, which: i32) -> &mut Vec<i32> {
    match which {
        0 => &mut f.row_with_no_pivot,
        1 => &mut f.col_with_no_pivot,
        2 => &mut f.var_with_no_pivot,
        3 => &mut f.refactor_pivot_row,
        4 => &mut f.refactor_pivot_var,
        5 => &mut f.l_pivot_index,
        6 => &mut f.l_pivot_lookup,
        7 => &mut f.l_start,
        8 => &mut f.l_index,
        9 => &mut f.lr_start,
        10 => &mut f.lr_index,
        11 => &mut f.u_pivot_lookup,
        12 => &mut f.u_pivot_index,
        13 => &mut f.u_start,
        14 => &mut f.u_last_p,
        15 => &mut f.u_index,
        16 => &mut f.ur_start,
        17 => &mut f.ur_lastp,
        18 => &mut f.ur_space,
        19 => &mut f.ur_index,
        20 => &mut f.pf_start,
        21 => &mut f.pf_index,
        22 => &mut f.pf_pivot_index,
        _ => panic!("no integer vector {which}"),
    }
}

fn dvec(f: &mut HFactor, which: i32) -> &mut Vec<f64> {
    match which {
        0 => &mut f.l_value,
        1 => &mut f.lr_value,
        2 => &mut f.u_pivot_value,
        3 => &mut f.u_value,
        4 => &mut f.ur_value,
        5 => &mut f.pf_value,
        6 => &mut f.pf_pivot_value,
        _ => panic!("no double vector {which}"),
    }
}

/// Integer vector `which` (see ivec): its data, with its length in `len`
///
/// # Safety
/// `p` must be a live factor; the data is valid until the factor changes
#[no_mangle]
pub unsafe extern "C" fn highs_rs_factor_ivec(
    p: *mut HFactor,
    which: i32,
    len: *mut i32,
) -> *const i32 {
    let v = ivec(&mut *p, which);
    *len = v.len() as i32;
    v.as_ptr()
}

/// # Safety
/// As for highs_rs_factor_ivec
#[no_mangle]
pub unsafe extern "C" fn highs_rs_factor_dvec(
    p: *mut HFactor,
    which: i32,
    len: *mut i32,
) -> *const f64 {
    let v = dvec(&mut *p, which);
    *len = v.len() as i32;
    v.as_ptr()
}

/// # Safety
/// As for highs_rs_factor_ivec
#[no_mangle]
pub unsafe extern "C" fn highs_rs_factor_refactor_type(
    p: *const HFactor,
    len: *mut i32,
) -> *const i8 {
    let v = &(*p).refactor_pivot_type;
    *len = v.len() as i32;
    v.as_ptr()
}

/// Replace integer vector `which` by data[..len]
///
/// # Safety
/// `data` must be valid for `len` reads
#[no_mangle]
pub unsafe extern "C" fn highs_rs_factor_set_ivec(
    p: *mut HFactor,
    which: i32,
    data: *const i32,
    len: i32,
) {
    let v = ivec(&mut *p, which);
    v.clear();
    v.extend_from_slice(sl(data, len));
}

/// HFactor::getInvert into the factor's own saved copy (for
/// HSimplexNla::putInvert): the vectors that getInvert copies out
///
/// # Safety
/// `p` must be a live factor
#[no_mangle]
pub unsafe extern "C" fn highs_rs_factor_put_invert(p: *mut HFactor) {
    let f = &mut *p;
    let mut saved = std::mem::take(&mut f.saved_invert);
    saved.0.resize(INVERT_IVECS.len(), Vec::new());
    saved.1.resize(INVERT_DVECS.len(), Vec::new());
    for (k, &which) in INVERT_IVECS.iter().enumerate() {
        saved.0[k].clone_from(ivec(f, which));
    }
    for (k, &which) in INVERT_DVECS.iter().enumerate() {
        saved.1[k].clone_from(dvec(f, which));
    }
    f.saved_invert = saved;
}

/// HFactor::setInvert from the factor's saved copy (for
/// HSimplexNla::getInvert)
///
/// # Safety
/// `p` must be a live factor, with a saved copy
#[no_mangle]
pub unsafe extern "C" fn highs_rs_factor_get_invert(p: *mut HFactor) {
    let f = &mut *p;
    let saved = std::mem::take(&mut f.saved_invert);
    for (k, &which) in INVERT_IVECS.iter().enumerate() {
        ivec(f, which).clone_from(&saved.0[k]);
    }
    for (k, &which) in INVERT_DVECS.iter().enumerate() {
        dvec(f, which).clone_from(&saved.1[k]);
    }
    f.saved_invert = saved;
    f.check_indices();
}

/// The vectors of InvertibleRepresentation (see ivec and dvec)
const INVERT_IVECS: [i32; 18] = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22];
const INVERT_DVECS: [i32; 7] = [0, 1, 2, 3, 4, 5, 6];

/// Check the row index invariant after setting vectors
///
/// # Safety
/// `p` must be a live factor
#[no_mangle]
pub unsafe extern "C" fn highs_rs_factor_check_indices(p: *const HFactor) {
    (*p).check_indices();
}

/// # Safety
/// `data` must be valid for `len` reads
#[no_mangle]
pub unsafe extern "C" fn highs_rs_factor_set_dvec(
    p: *mut HFactor,
    which: i32,
    data: *const f64,
    len: i32,
) {
    let v = dvec(&mut *p, which);
    v.clear();
    v.extend_from_slice(sl(data, len));
}
