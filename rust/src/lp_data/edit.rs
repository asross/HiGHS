//! Model modification internals of HighsInterface.cpp and HighsLpUtils.cpp:
//! changing costs, bounds, integrality and matrix coefficients over an
//! index collection, deleting entries of scale and basis status vectors,
//! the basis statuses of nonbasic variables whose bounds changed or that
//! are appended, feasibility with respect to bounds, and the row values
//! and column duals of a solution in double-double arithmetic.
//!
//! C++ (HighsLpUtilsRust.cpp) sizes the vectors before a call and shrinks
//! them to the length a call returns.

use super::ffi::{CIndexCollection, RsMut};
use super::lp_utils::IndexCollection;
use super::INF;
use crate::util::cdouble::CDouble;

// HighsBasisStatus
const LOWER: u8 = 0;
const BASIC: u8 = 1;
const UPPER: u8 = 2;
const ZERO: u8 = 3;
const NONBASIC: u8 = 4;
// nonbasicMove
const MOVE_UP: i8 = 1;
const MOVE_DN: i8 = -1;
const MOVE_ZE: i8 = 0;

/// The state of updateOutInIndex between calls
pub struct OutIn {
    pub out_from: i32,
    pub out_to: i32,
    pub in_from: i32,
    pub in_to: i32,
    pub current_set_entry: i32,
}

impl OutIn {
    pub fn new() -> OutIn {
        OutIn { out_from: 0, out_to: 0, in_from: 0, in_to: -1, current_set_entry: 0 }
    }
}

impl Default for OutIn {
    fn default() -> Self {
        Self::new()
    }
}

/// updateOutInIndex (util/HighsUtils.cpp)
pub fn update_out_in_index(ic: &IndexCollection, s: &mut OutIn) {
    if ic.is_interval {
        s.out_from = ic.from;
        s.out_to = ic.to;
        s.in_from = ic.to + 1;
        s.in_to = ic.dimension - 1;
    } else if ic.is_set {
        let set = ic.set;
        s.out_from = set[s.current_set_entry as usize];
        s.out_to = s.out_from;
        s.current_set_entry += 1;
        let current_set_entry0 = s.current_set_entry;
        for set_entry in current_set_entry0..ic.set_num_entries {
            let ix = set[set_entry as usize];
            if ix > s.out_to + 1 {
                break;
            }
            s.out_to = set[s.current_set_entry as usize];
            s.current_set_entry += 1;
        }
        s.in_from = s.out_to + 1;
        s.in_to = if s.current_set_entry < ic.set_num_entries {
            set[s.current_set_entry as usize] - 1
        } else {
            ic.dimension - 1
        };
    } else {
        let mask = ic.mask;
        s.out_from = s.in_to + 1;
        s.out_to = ic.dimension - 1;
        for ix in s.in_to + 1..ic.dimension {
            if mask[ix as usize] == 0 {
                s.out_to = ix - 1;
                break;
            }
        }
        s.in_from = s.out_to + 1;
        s.in_to = ic.dimension - 1;
        for ix in s.out_to + 1..ic.dimension {
            if mask[ix as usize] != 0 {
                s.in_to = ix - 1;
                break;
            }
        }
    }
}

/// The loop of changeLpCosts, changeBounds and changeLpIntegrality: each
/// (model index, user data index) to change
fn for_each_change(ic: &IndexCollection, mut f: impl FnMut(usize, usize)) {
    let (from_k, to_k) = ic.limits();
    if from_k > to_k {
        return;
    }
    let mut usr_ix: i32 = -1;
    for k in from_k..=to_k {
        let ix = if ic.is_interval || ic.is_mask { k } else { ic.set[k as usize] };
        if ic.is_interval {
            usr_ix += 1;
        } else {
            usr_ix = k;
        }
        if ic.is_mask && ic.mask[ix as usize] == 0 {
            continue;
        }
        f(ix as usize, usr_ix as usize);
    }
}

/// changeLpCosts (without the infinite cost check)
pub fn change_costs(cost: &mut [f64], ic: &IndexCollection, new_cost: &[f64]) {
    for_each_change(ic, |ix, u| cost[ix] = new_cost[u]);
}

/// changeBounds
pub fn change_bounds(lower: &mut [f64], upper: &mut [f64], ic: &IndexCollection, new_lower: &[f64], new_upper: &[f64]) {
    for_each_change(ic, |ix, u| {
        lower[ix] = new_lower[u];
        upper[ix] = new_upper[u];
    });
}

/// changeLpIntegrality on an integrality vector of the right size
pub fn change_integrality(integrality: &mut [u8], ic: &IndexCollection, new_integrality: &[u8]) {
    for_each_change(ic, |ix, u| integrality[ix] = new_integrality[u]);
}

/// The compaction of deleteScale and deleteBasisEntries: keeps the
/// entries not in the collection, returning the new length and whether
/// an entry of `is_basic` / not of it was deleted
fn delete_entries<T: Copy>(v: &mut [T], ic: &IndexCollection, is_basic: impl Fn(&T) -> bool) -> (usize, bool, bool) {
    let (from_k, to_k) = ic.limits();
    let dim = ic.dimension;
    let mut deleted_basic = false;
    let mut deleted_nonbasic = false;
    if from_k > to_k {
        return (v.len(), false, false);
    }
    let mut s = OutIn::new();
    let mut new_num: i32 = 0;
    for k in from_k..=to_k {
        update_out_in_index(ic, &mut s);
        if k == from_k {
            new_num = s.out_from;
        }
        for e in s.out_from..=s.out_to {
            if (e as usize) < v.len() {
                if is_basic(&v[e as usize]) {
                    deleted_basic = true;
                } else {
                    deleted_nonbasic = true;
                }
            }
        }
        if s.out_to >= dim - 1 {
            break;
        }
        for e in s.in_from..=s.in_to {
            v[new_num as usize] = v[e as usize];
            new_num += 1;
        }
        if s.in_to >= dim - 1 {
            break;
        }
    }
    (new_num as usize, deleted_basic, deleted_nonbasic)
}

/// HighsLp::deleteColsFromVectors / deleteRowsFromVectors: the kept
/// entries of the (cost,) lower and upper bounds and any integrality
/// moved to the front, and `kept[new] = old` for the names (moved by C++).
/// Returns the new dimension.
pub fn delete_from_vectors(
    ic: &IndexCollection,
    vectors: &mut [&mut [f64]],
    integrality: &mut [u8],
    kept: &mut [i32],
) -> usize {
    let (from_k, to_k) = ic.limits();
    let dim = ic.dimension;
    if from_k > to_k {
        return dim as usize;
    }
    let have_integrality = !integrality.is_empty();
    let mut s = OutIn::new();
    let mut new_num: i32 = 0;
    for k in from_k..=to_k {
        update_out_in_index(ic, &mut s);
        if k == from_k {
            new_num = s.out_from;
            for (i, x) in kept[..new_num as usize].iter_mut().enumerate() {
                *x = i as i32;
            }
        }
        if s.out_to >= dim - 1 {
            break;
        }
        for e in s.in_from..=s.in_to {
            let (n, e) = (new_num as usize, e as usize);
            for v in vectors.iter_mut() {
                v[n] = v[e];
            }
            if have_integrality {
                integrality[n] = integrality[e];
            }
            kept[n] = e as i32;
            new_num += 1;
        }
        if s.in_to >= dim - 1 {
            break;
        }
    }
    new_num as usize
}

/// deleteScale: the kept scale factors moved to the front (C++ resizes)
pub fn delete_scale(scale: &mut [f64], ic: &IndexCollection) {
    delete_entries(scale, ic, |_| false);
}

/// deleteBasisEntries: returns the new length, whether a basic and
/// whether a nonbasic entry was deleted
pub fn delete_basis_entries(status: &mut [u8], ic: &IndexCollection) -> (usize, bool, bool) {
    let (from_k, to_k) = ic.limits();
    if from_k > to_k {
        // The C++ returns before its resize and flag resets
        return (status.len(), false, false);
    }
    delete_entries(status, ic, |s| *s == BASIC)
}

/// changeLpMatrixCoefficient on a column-wise matrix whose index and value
/// have room for one more entry: returns the new number of nonzeros if
/// the C++ resizes (an inserted entry), else -1
#[allow(clippy::too_many_arguments)]
pub fn change_matrix_coefficient(
    start: &mut [i32],
    index: &mut [i32],
    value: &mut [f64],
    num_col: usize,
    row: i32,
    col: usize,
    new_value: f64,
    zero_new_value: bool,
) -> i64 {
    let mut change_el: Option<usize> = None;
    for el in start[col] as usize..start[col + 1] as usize {
        if index[el] == row {
            change_el = Some(el);
            break;
        }
    }
    let mut resized = -1;
    let change_el = match change_el {
        None => {
            if zero_new_value {
                return -1;
            }
            let change_el = start[col + 1] as usize;
            let new_num_nz = start[num_col] as usize + 1;
            for s in &mut start[col + 1..=num_col] {
                *s += 1;
            }
            for el in (change_el + 1..new_num_nz).rev() {
                index[el] = index[el - 1];
                value[el] = value[el - 1];
            }
            resized = new_num_nz as i64;
            change_el
        }
        Some(el) if zero_new_value => {
            let new_num_nz = start[num_col] as usize - 1;
            for s in &mut start[col + 1..=num_col] {
                *s -= 1;
            }
            for e in el..new_num_nz {
                index[e] = index[e + 1];
                value[e] = value[e + 1];
            }
            return -1;
        }
        Some(el) => el,
    };
    index[change_el] = row;
    value[change_el] = new_value;
    resized
}

/// getCoefficientInterface / getLpMatrixCoefficient: the value of
/// (`minor`, `major`) in a column-wise (major = column) or row-wise matrix
pub fn get_coefficient(start: &[i32], index: &[i32], value: &[f64], major: usize, minor: i32) -> f64 {
    for el in start[major] as usize..start[major + 1] as usize {
        if index[el] == minor {
            return value[el];
        }
    }
    0.0
}

/// Highs::feasibleWrtBounds, given that the primal solution is feasible
pub fn feasible_wrt_bounds(value: &[f64], lower: &[f64], upper: &[f64], tolerance: f64) -> bool {
    for i in 0..lower.len() {
        if value[i] < lower[i] - tolerance {
            return false;
        }
        if value[i] > upper[i] + tolerance {
            return false;
        }
    }
    true
}

/// The status and move of a nonbasic variable with these bounds, keeping
/// a definitive status of a boxed one (setNonbasicStatusInterface,
/// appendNonbasicColsToBasisInterface). A row's move is the opposite of a
/// column's.
fn nonbasic_status(lower: f64, upper: f64, status: u8, column: bool) -> (u8, i8) {
    let (up, dn) = if column { (MOVE_UP, MOVE_DN) } else { (MOVE_DN, MOVE_UP) };
    if lower == upper {
        (if status == NONBASIC { LOWER } else { status }, MOVE_ZE)
    } else if -lower < INF {
        if upper < INF {
            if status == NONBASIC {
                if lower.abs() < upper.abs() {
                    (LOWER, up)
                } else {
                    (UPPER, dn)
                }
            } else if status == LOWER {
                (status, up)
            } else {
                (status, dn)
            }
        } else {
            (LOWER, up)
        }
    } else if upper < INF {
        (UPPER, dn)
    } else {
        (ZERO, MOVE_ZE)
    }
}

/// Highs::setNonbasicStatusInterface for a valid basis: the simplex
/// flags and moves (offset by the number of columns for rows) are set
/// when `flag` and `mv` are not empty
#[allow(clippy::too_many_arguments)]
pub fn set_nonbasic_status(
    ic: &IndexCollection,
    columns: bool,
    status: &mut [u8],
    lower: &[f64],
    upper: &[f64],
    flag: &mut [i8],
    mv: &mut [i8],
    offset: usize,
) {
    let (from_k, to_k) = ic.limits();
    let ix_dim = status.len() as i32;
    let has_simplex_basis = !flag.is_empty();
    let mut s = OutIn::new();
    for _k in from_k..=to_k {
        update_out_in_index(ic, &mut s);
        for i in s.out_from..=s.out_to {
            let i = i as usize;
            if status[i] == BASIC {
                continue;
            }
            let (st, m) = nonbasic_status(lower[i], upper[i], status[i], columns);
            status[i] = st;
            if has_simplex_basis {
                flag[offset + i] = 1;
                mv[offset + i] = m;
            }
        }
        if s.in_to >= ix_dim - 1 {
            break;
        }
    }
}

/// Highs::appendNonbasicColsToBasisInterface after C++ has resized the
/// column statuses (to `num_col + num_new`) and, with a simplex basis,
/// nonbasicFlag and nonbasicMove
#[allow(clippy::too_many_arguments)]
pub fn append_nonbasic_cols(
    num_col: usize,
    num_row: usize,
    num_new: usize,
    col_status: &mut [u8],
    lower: &[f64],
    upper: &[f64],
    flag: &mut [i8],
    mv: &mut [i8],
    basic_index: &mut [i32],
) {
    let new_num_col = num_col + num_new;
    let has_simplex_basis = !flag.is_empty();
    if has_simplex_basis {
        for i_row in (0..num_row).rev() {
            if basic_index[i_row] as usize >= num_col {
                basic_index[i_row] += num_new as i32;
            }
            flag[new_num_col + i_row] = flag[num_col + i_row];
            mv[new_num_col + i_row] = mv[num_col + i_row];
        }
    }
    for j in num_col..new_num_col {
        let (st, m) = nonbasic_status(lower[j], upper[j], NONBASIC, true);
        col_status[j] = st;
        if has_simplex_basis {
            flag[j] = 1;
            mv[j] = m;
        }
    }
}

/// calculateRowValuesQuad: row values of a column-wise matrix in
/// double-double arithmetic
pub fn calculate_row_values_quad(start: &[i32], index: &[i32], value: &[f64], col_value: &[f64], row_value: &mut [f64]) {
    let mut quad = vec![CDouble::from(0.0); row_value.len()];
    for (col, &x) in col_value.iter().enumerate() {
        for el in start[col] as usize..start[col + 1] as usize {
            quad[index[el] as usize] += CDouble::from(x) * value[el];
        }
    }
    for (r, q) in row_value.iter_mut().zip(quad) {
        *r = q.to_f64();
    }
}

/// calculateColDualsQuad
pub fn calculate_col_duals_quad(
    start: &[i32],
    index: &[i32],
    value: &[f64],
    cost: &[f64],
    row_dual: &[f64],
    col_dual: &mut [f64],
) {
    for (col, d) in col_dual.iter_mut().enumerate() {
        let mut q = CDouble::from(0.0);
        for el in start[col] as usize..start[col + 1] as usize {
            q += CDouble::from(row_dual[index[el] as usize]) * value[el];
        }
        q += cost[col];
        *d = q.to_f64();
    }
}

/// Where getSubVectors(Transpose) writes: the three data vectors and the
/// sub-matrix, each optional
#[derive(Default)]
pub struct SubOut<'a> {
    pub data: [Option<&'a mut [f64]>; 3],
    pub start: Option<&'a mut [i32]>,
    pub index: Option<&'a mut [i32]>,
    pub value: Option<&'a mut [f64]>,
}

/// getSubVectors: the vectors of `ic` (and their part of the matrix,
/// whose vectors are the same as the data's); returns (num_sub_vector,
/// sub_matrix_num_nz)
pub fn get_sub_vectors(
    ic: &IndexCollection,
    dim: i32,
    data: [&[f64]; 3],
    m_start: &[i32],
    m_index: &[i32],
    m_value: &[f64],
    out: &mut SubOut,
) -> (usize, usize) {
    let (from_k, to_k) = ic.limits();
    let mut s = OutIn::new();
    let (mut n, mut nnz) = (0usize, 0usize);
    for _ in from_k..=to_k {
        update_out_in_index(ic, &mut s);
        for v in s.out_from..=s.out_to {
            let v = v as usize;
            for (o, d) in out.data.iter_mut().zip(data) {
                if let Some(o) = o {
                    o[n] = d[v];
                }
            }
            if let Some(o) = &mut out.start {
                o[n] = nnz as i32 + m_start[v] - m_start[s.out_from as usize];
            }
            n += 1;
        }
        for el in m_start[s.out_from as usize] as usize..m_start[s.out_to as usize + 1] as usize {
            if let Some(o) = &mut out.index {
                o[nnz] = m_index[el];
            }
            if let Some(o) = &mut out.value {
                o[nnz] = m_value[el];
            }
            nnz += 1;
        }
        if s.out_to == dim - 1 || s.in_to == dim - 1 {
            break;
        }
    }
    (n, nnz)
}

/// getSubVectorsTranspose: as get_sub_vectors, of the vectors that are the
/// matrix's indices
pub fn get_sub_vectors_transpose(
    ic: &IndexCollection,
    dim: i32,
    data: [&[f64]; 3],
    m_start: &[i32],
    m_index: &[i32],
    m_value: &[f64],
    out: &mut SubOut,
) -> (usize, usize) {
    let (from_k, to_k) = ic.limits();
    let mut new_index = vec![0i32; dim.max(0) as usize];
    let mut n = 0i32;
    if !ic.is_mask {
        // "In" and "out" swap: the vectors to get are the "in" ones
        let mut s = OutIn::new();
        s.out_to = -1;
        for k in from_k..=to_k {
            update_out_in_index(ic, &mut s);
            let (in_from, in_to, out_from, out_to) = (s.out_from, s.out_to, s.in_from, s.in_to);
            if k == from_k {
                for v in 0..in_from {
                    new_index[v as usize] = -1;
                }
            }
            for v in in_from..=in_to {
                new_index[v as usize] = n;
                n += 1;
            }
            for v in out_from..=out_to {
                new_index[v as usize] = -1;
            }
            if out_to >= dim - 1 {
                break;
            }
        }
    } else {
        for (v, x) in new_index.iter_mut().enumerate() {
            if ic.mask[v] != 0 {
                *x = n;
                n += 1;
            } else {
                *x = -1;
            }
        }
    }
    let n = n as usize;
    if n == 0 {
        return (0, 0);
    }
    for (v, &nv) in new_index.iter().enumerate() {
        if nv >= 0 {
            for (o, d) in out.data.iter_mut().zip(data) {
                if let Some(o) = o {
                    o[nv as usize] = d[v];
                }
            }
        }
    }
    let mut length = vec![0i32; n];
    let num_vector = m_start.len() - 1;
    for vector in 0..num_vector {
        for el in m_start[vector] as usize..m_start[vector + 1] as usize {
            let nv = new_index[m_index[el] as usize];
            if nv >= 0 {
                length[nv as usize] += 1;
            }
        }
    }
    let Some(start) = &mut out.start else {
        return (n, length.iter().map(|&l| l as usize).sum());
    };
    start[0] = 0;
    for v in 0..n - 1 {
        start[v + 1] = start[v] + length[v];
        length[v] = start[v];
    }
    let nnz = (start[n - 1] + length[n - 1]) as usize;
    if out.index.is_none() && out.value.is_none() {
        return (n, nnz);
    }
    length[n - 1] = start[n - 1];
    for vector in 0..num_vector {
        for el in m_start[vector] as usize..m_start[vector + 1] as usize {
            let nv = new_index[m_index[el] as usize];
            if nv >= 0 {
                let row_el = length[nv as usize] as usize;
                if let Some(o) = &mut out.index {
                    o[row_el] = vector as i32;
                }
                if let Some(o) = &mut out.value {
                    o[row_el] = m_value[el];
                }
                length[nv as usize] += 1;
            }
        }
    }
    (n, nnz)
}

// The C++ entry points (highs/lp_data/HighsLpUtilsRust.cpp)

/// getSubVectors (transpose false) or getSubVectorsTranspose; the data
/// vectors have `ic`'s dimension (null if absent), the outputs are null
/// or as large as the C++ caller made them for what is got: sized here by
/// a first pass that writes nothing
///
/// # Safety
/// `ic` valid; data and matrix arrays valid; the non-null outputs hold
/// num_sub_vector (data, start) or num_nz (index, value) entries
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn highs_rs_get_sub_vectors(
    transpose: bool,
    ic: *const CIndexCollection,
    data_dim: i32,
    data: *const *const f64,
    m_start: RsMut<i32>,
    m_index: RsMut<i32>,
    m_value: RsMut<f64>,
    out_data: *const *mut f64,
    out_start: *mut i32,
    out_index: *mut i32,
    out_value: *mut f64,
    num_sub_vector: *mut i32,
    num_nz: *mut i32,
) {
    let ic = (*ic).view();
    let dim = data_dim.max(0) as usize;
    let data: [&[f64]; 3] = std::array::from_fn(|k| {
        let p = *data.add(k);
        if p.is_null() {
            &[][..]
        } else {
            std::slice::from_raw_parts(p, dim)
        }
    });
    let f = if transpose { get_sub_vectors_transpose } else { get_sub_vectors };
    let (ms, mi, mv) = (m_start.get(), m_index.get(), m_value.get());
    let (n, nnz) = f(&ic, data_dim, data, ms, mi, mv, &mut SubOut::default());
    unsafe fn sl<'a, T>(p: *mut T, n: usize) -> Option<&'a mut [T]> {
        (!p.is_null()).then(|| std::slice::from_raw_parts_mut(p, n))
    }
    let mut out = SubOut {
        data: std::array::from_fn(|k| sl(*out_data.add(k), n)),
        start: sl(out_start, n),
        index: sl(out_index, nnz),
        value: sl(out_value, nnz),
    };
    let (n, nnz) = f(&ic, data_dim, data, ms, mi, mv, &mut out);
    *num_sub_vector = n as i32;
    *num_nz = nnz as i32;
}

/// # Safety
/// The arrays valid; `ic` a valid collection
#[no_mangle]
pub unsafe extern "C" fn highs_rs_change_values(
    which: i32,
    ic: *const CIndexCollection,
    a: RsMut<f64>,
    b: RsMut<f64>,
    new_a: RsMut<f64>,
    new_b: RsMut<f64>,
) {
    let ic = (*ic).view();
    if which == 0 {
        change_costs(a.get_mut(), &ic, new_a.get());
    } else {
        change_bounds(a.get_mut(), b.get_mut(), &ic, new_a.get(), new_b.get());
    }
}

/// # Safety
/// As highs_rs_change_values
#[no_mangle]
pub unsafe extern "C" fn highs_rs_change_integrality(
    ic: *const CIndexCollection,
    integrality: RsMut<u8>,
    new_integrality: RsMut<u8>,
) {
    change_integrality(integrality.get_mut(), &(*ic).view(), new_integrality.get());
}

/// # Safety
/// As highs_rs_change_values; `cost` may be empty (rows)
#[no_mangle]
pub unsafe extern "C" fn highs_rs_delete_from_vectors(
    ic: *const CIndexCollection,
    cost: RsMut<f64>,
    lower: RsMut<f64>,
    upper: RsMut<f64>,
    integrality: RsMut<u8>,
    kept: RsMut<i32>,
) -> usize {
    let ic = (*ic).view();
    let (lower, upper) = (lower.get_mut(), upper.get_mut());
    if cost.len == 0 {
        delete_from_vectors(&ic, &mut [lower, upper], integrality.get_mut(), kept.get_mut())
    } else {
        delete_from_vectors(&ic, &mut [cost.get_mut(), lower, upper], integrality.get_mut(), kept.get_mut())
    }
}

/// # Safety
/// As highs_rs_change_values
#[no_mangle]
pub unsafe extern "C" fn highs_rs_delete_scale(ic: *const CIndexCollection, scale: RsMut<f64>) {
    delete_scale(scale.get_mut(), &(*ic).view());
}

/// # Safety
/// As highs_rs_change_values
#[no_mangle]
pub unsafe extern "C" fn highs_rs_delete_basis_entries(
    ic: *const CIndexCollection,
    status: RsMut<u8>,
    deleted_basic: *mut bool,
    deleted_nonbasic: *mut bool,
) -> usize {
    let (n, b, nb) = delete_basis_entries(status.get_mut(), &(*ic).view());
    *deleted_basic = b;
    *deleted_nonbasic = nb;
    n
}

/// # Safety
/// The arrays valid, index and value with room for one more entry
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn highs_rs_change_matrix_coefficient(
    start: RsMut<i32>,
    index: RsMut<i32>,
    value: RsMut<f64>,
    num_col: i32,
    row: i32,
    col: i32,
    new_value: f64,
    zero_new_value: bool,
) -> i64 {
    change_matrix_coefficient(
        start.get_mut(),
        index.get_mut(),
        value.get_mut(),
        num_col as usize,
        row,
        col as usize,
        new_value,
        zero_new_value,
    )
}

/// # Safety
/// The arrays valid
#[no_mangle]
pub unsafe extern "C" fn highs_rs_get_coefficient(
    start: RsMut<i32>,
    index: RsMut<i32>,
    value: RsMut<f64>,
    major: i32,
    minor: i32,
) -> f64 {
    get_coefficient(start.get(), index.get(), value.get(), major as usize, minor)
}

/// # Safety
/// The arrays valid
#[no_mangle]
pub unsafe extern "C" fn highs_rs_feasible_wrt_bounds(
    value: RsMut<f64>,
    lower: RsMut<f64>,
    upper: RsMut<f64>,
    tolerance: f64,
) -> bool {
    feasible_wrt_bounds(value.get(), lower.get(), upper.get(), tolerance)
}

/// # Safety
/// The arrays valid
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn highs_rs_set_nonbasic_status(
    ic: *const CIndexCollection,
    columns: bool,
    status: RsMut<u8>,
    lower: RsMut<f64>,
    upper: RsMut<f64>,
    flag: RsMut<i8>,
    mv: RsMut<i8>,
    offset: i32,
) {
    set_nonbasic_status(
        &(*ic).view(),
        columns,
        status.get_mut(),
        lower.get(),
        upper.get(),
        flag.get_mut(),
        mv.get_mut(),
        offset as usize,
    );
}

/// # Safety
/// The arrays valid and sized as documented
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn highs_rs_append_nonbasic_cols(
    num_col: i32,
    num_row: i32,
    num_new: i32,
    col_status: RsMut<u8>,
    lower: RsMut<f64>,
    upper: RsMut<f64>,
    flag: RsMut<i8>,
    mv: RsMut<i8>,
    basic_index: RsMut<i32>,
) {
    append_nonbasic_cols(
        num_col as usize,
        num_row as usize,
        num_new as usize,
        col_status.get_mut(),
        lower.get(),
        upper.get(),
        flag.get_mut(),
        mv.get_mut(),
        basic_index.get_mut(),
    );
}

/// # Safety
/// The arrays valid; out sized by C++
#[no_mangle]
pub unsafe extern "C" fn highs_rs_calculate_row_values_quad(
    start: RsMut<i32>,
    index: RsMut<i32>,
    value: RsMut<f64>,
    col_value: RsMut<f64>,
    row_value: RsMut<f64>,
) {
    calculate_row_values_quad(start.get(), index.get(), value.get(), col_value.get(), row_value.get_mut());
}

/// # Safety
/// The arrays valid; out sized by C++
#[no_mangle]
pub unsafe extern "C" fn highs_rs_calculate_col_duals_quad(
    start: RsMut<i32>,
    index: RsMut<i32>,
    value: RsMut<f64>,
    cost: RsMut<f64>,
    row_dual: RsMut<f64>,
    col_dual: RsMut<f64>,
) {
    calculate_col_duals_quad(start.get(), index.get(), value.get(), cost.get(), row_dual.get(), col_dual.get_mut());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(dim: i32, s: &[i32]) -> IndexCollection<'_> {
        IndexCollection {
            dimension: dim,
            is_interval: false,
            from: -1,
            to: -2,
            is_set: true,
            set_num_entries: s.len() as i32,
            set: s,
            is_mask: false,
            mask: &[],
        }
    }

    #[test]
    fn delete_and_change() {
        let mut v = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let ic = set(6, &[1, 2, 4]);
        assert_eq!(delete_entries(&mut v, &ic, |_| false).0, 3);
        assert_eq!(&v[..3], &[0.0, 3.0, 5.0]);
        let mut st = [BASIC, LOWER, BASIC, UPPER];
        let ic = IndexCollection::interval(4, 1, 2);
        assert_eq!(delete_basis_entries(&mut st, &ic), (2, true, true));
        assert_eq!(&st[..2], &[BASIC, UPPER]);
        let mut cost = [1.0, 2.0, 3.0];
        let mask = [1, 0, 1];
        let ic = IndexCollection { is_interval: false, is_mask: true, mask: &mask, ..IndexCollection::interval(3, 0, 2) };
        change_costs(&mut cost, &ic, &[7.0, 8.0, 9.0]);
        assert_eq!(cost, [7.0, 2.0, 9.0]);
    }

    #[test]
    fn coefficient() {
        // 2 columns: col 0 rows {0, 1}, col 1 row {1}
        let mut start = [0, 2, 3];
        let mut index = [0, 1, 1, 0];
        let mut value = [1.0, 2.0, 3.0, 0.0];
        assert_eq!(change_matrix_coefficient(&mut start, &mut index, &mut value, 2, 0, 1, 5.0, false), 4);
        assert_eq!((start, &index[..], &value[..]), ([0, 2, 4], &[0, 1, 1, 0][..], &[1.0, 2.0, 3.0, 5.0][..]));
        assert_eq!(get_coefficient(&start, &index, &value, 1, 0), 5.0);
        assert_eq!(change_matrix_coefficient(&mut start, &mut index, &mut value, 2, 1, 0, 0.0, true), -1);
        assert_eq!(start, [0, 1, 3]);
        assert_eq!(get_coefficient(&start, &index, &value, 0, 1), 0.0);
    }
}
