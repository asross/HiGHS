//! The BASICLU C interface (basiclu_*.c, basiclu_object.c), exported under
//! the C names so that IPX calls these when the C files are left out of the
//! build (HIGHS_RUST). Array lengths are derived as in the C code: the
//! stores from the dimension in xstore, the factor files from their sizes
//! in xstore, the right-hand sides from nzrhs.

use super::get_factors::Csc;
use super::*;
use crate::ffi::{sl, sl_mut};
use std::ffi::c_void;

/// The store arrays, if both are present and initialized
///
/// # Safety
/// istore, xstore must be null or hold istore_len(m), xstore_len(m)
/// elements for the m in xstore
unsafe fn stores<'a>(istore: *mut Int, xstore: *mut f64) -> Option<(&'a mut [Int], &'a mut [f64])> {
    if istore.is_null() || xstore.is_null() || *istore != HASH || *xstore != HASH as f64 {
        return None;
    }
    let m = *xstore.add(DIM) as Int;
    Some((
        sl_mut(istore, istore_len(m) as Int),
        sl_mut(xstore, xstore_len(m) as Int),
    ))
}

/// lu_load on the C arguments; Err(status) if the store is invalid
///
/// # Safety
/// As for stores(); the factor arrays must be null or hold the sizes given
/// in xstore
#[allow(clippy::too_many_arguments)]
unsafe fn load<'a>(
    istore: *mut Int,
    xstore: *mut f64,
    li: *mut Int,
    lx: *mut f64,
    ui: *mut Int,
    ux: *mut f64,
    wi: *mut Int,
    wx: *mut f64,
) -> Result<Lu<'a>, Int> {
    let (istore, xstore) = stores(istore, xstore).ok_or(ERROR_INVALID_STORE)?;
    let lmem = xstore[MEMORYL] as Int;
    let umem = xstore[MEMORYU] as Int;
    let wmem = xstore[MEMORYW] as Int;
    Lu::load(
        istore,
        xstore,
        sl_mut(li, lmem),
        sl_mut(lx, lmem),
        sl_mut(ui, umem),
        sl_mut(ux, umem),
        sl_mut(wi, wmem),
        sl_mut(wx, wmem),
    )
}

/// # Safety
/// istore, xstore must hold istore_len(m), xstore_len(m) elements (or be
/// null)
#[no_mangle]
pub unsafe extern "C" fn basiclu_initialize(m: Int, istore: *mut Int, xstore: *mut f64) -> Int {
    if istore.is_null() || xstore.is_null() {
        return ERROR_ARGUMENT_MISSING;
    }
    if m <= 0 {
        return ERROR_INVALID_ARGUMENT;
    }
    initialize(
        m,
        sl_mut(istore, istore_len(m) as Int),
        sl_mut(xstore, xstore_len(m) as Int),
    );
    OK
}

/// # Safety
/// As for load(); Bbegin, Bend hold m entries, Bi, Bx max(Bend) entries
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn basiclu_factorize(
    istore: *mut Int,
    xstore: *mut f64,
    li: *mut Int,
    lx: *mut f64,
    ui: *mut Int,
    ux: *mut f64,
    wi: *mut Int,
    wx: *mut f64,
    bbegin: *const Int,
    bend: *const Int,
    bi: *const Int,
    bx: *const f64,
    c0ntinue: Int,
) -> Int {
    let mut this = match load(istore, xstore, li, lx, ui, ux, wi, wx) {
        Ok(this) => this,
        Err(status) => return status,
    };
    if li.is_null()
        || lx.is_null()
        || ui.is_null()
        || ux.is_null()
        || wi.is_null()
        || wx.is_null()
        || bbegin.is_null()
        || bend.is_null()
        || bi.is_null()
        || bx.is_null()
    {
        return this.save(ERROR_ARGUMENT_MISSING);
    }
    let m = this.m;
    let bbegin = sl(bbegin, m);
    let bend = sl(bend, m);
    let bnz = bend.iter().copied().max().unwrap_or(0);
    let status = this.factorize(bbegin, bend, sl(bi, bnz), sl(bx, bnz), c0ntinue != 0);
    this.save(status)
}

/// # Safety
/// As for load(); the output arrays must be null or hold m (rowperm,
/// colperm), m+1 (colptr), Lnz+m (Lrowidx, Lvalue) and Unz+m (Urowidx,
/// Uvalue) entries
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn basiclu_get_factors(
    istore: *mut Int,
    xstore: *mut f64,
    li: *mut Int,
    lx: *mut f64,
    ui: *mut Int,
    ux: *mut f64,
    wi: *mut Int,
    wx: *mut f64,
    rowperm: *mut Int,
    colperm: *mut Int,
    lcolptr: *mut Int,
    lrowidx: *mut Int,
    lvalue: *mut f64,
    ucolptr: *mut Int,
    urowidx: *mut Int,
    uvalue: *mut f64,
) -> Int {
    let mut this = match load(istore, xstore, li, lx, ui, ux, wi, wx) {
        Ok(this) => this,
        Err(status) => return status,
    };
    if this.nupdate != 0 {
        return this.save(ERROR_INVALID_CALL);
    }
    let m = this.m;
    let opt = |p: *mut Int| (!p.is_null()).then(|| sl_mut(p, m));
    let l: Option<Csc> =
        (!lcolptr.is_null() && !lrowidx.is_null() && !lvalue.is_null()).then(|| {
            let n = this.lnz + m;
            (
                sl_mut(lcolptr, m + 1),
                sl_mut(lrowidx, n),
                sl_mut(lvalue, n),
            )
        });
    let u: Option<Csc> =
        (!ucolptr.is_null() && !urowidx.is_null() && !uvalue.is_null()).then(|| {
            let n = this.unz + m;
            (
                sl_mut(ucolptr, m + 1),
                sl_mut(urowidx, n),
                sl_mut(uvalue, n),
            )
        });
    this.get_factors(opt(rowperm), opt(colperm), l, u);
    OK // the C code does not save on success
}

/// # Safety
/// As for load(); rhs, lhs hold m entries and may be the same array
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn basiclu_solve_dense(
    istore: *mut Int,
    xstore: *mut f64,
    li: *mut Int,
    lx: *mut f64,
    ui: *mut Int,
    ux: *mut f64,
    wi: *mut Int,
    wx: *mut f64,
    rhs: *const f64,
    lhs: *mut f64,
    trans: u8,
) -> Int {
    let mut this = match load(istore, xstore, li, lx, ui, ux, wi, wx) {
        Ok(this) => this,
        Err(status) => return status,
    };
    let status = if li.is_null()
        || lx.is_null()
        || ui.is_null()
        || ux.is_null()
        || wi.is_null()
        || wx.is_null()
        || rhs.is_null()
        || lhs.is_null()
    {
        ERROR_ARGUMENT_MISSING
    } else if this.nupdate < 0 {
        ERROR_INVALID_CALL
    } else {
        let m = this.m;
        let rhs = (!std::ptr::eq(rhs, lhs)).then(|| sl(rhs, m));
        this.solve_dense(rhs, sl_mut(lhs, m), trans);
        OK
    };
    this.save(status)
}

/// Check the indices of a sparse right-hand side
fn rhs_ok(irhs: &[Int], m: Int) -> bool {
    irhs.iter().all(|&i| i >= 0 && i < m)
}

/// # Safety
/// As for load(); irhs, xrhs hold nzrhs entries (irhs at least 1 for
/// trans 'T'; xrhs may be null for 'T'); ilhs, lhs hold m entries (or are
/// null, then no solution is computed, as is p_nzlhs)
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn basiclu_solve_for_update(
    istore: *mut Int,
    xstore: *mut f64,
    li: *mut Int,
    lx: *mut f64,
    ui: *mut Int,
    ux: *mut f64,
    wi: *mut Int,
    wx: *mut f64,
    nzrhs: Int,
    irhs: *const Int,
    xrhs: *const f64,
    p_nzlhs: *mut Int,
    ilhs: *mut Int,
    lhs: *mut f64,
    trans: u8,
) -> Int {
    let mut this = match load(istore, xstore, li, lx, ui, ux, wi, wx) {
        Ok(this) => this,
        Err(status) => return status,
    };
    let m = this.m;
    let is_t = trans == b't' || trans == b'T';
    let mut status = OK;
    if li.is_null()
        || lx.is_null()
        || ui.is_null()
        || ux.is_null()
        || wi.is_null()
        || wx.is_null()
        || irhs.is_null()
        || (!is_t && xrhs.is_null())
    {
        status = ERROR_ARGUMENT_MISSING;
    } else if this.nupdate < 0 {
        status = ERROR_INVALID_CALL;
    } else if this.nforrest == m {
        status = ERROR_MAXIMUM_UPDATES;
    } else {
        // check RHS indices
        let ok = if is_t {
            rhs_ok(sl(irhs, 1), m)
        } else {
            (0..=m).contains(&nzrhs) && rhs_ok(sl(irhs, nzrhs), m)
        };
        if !ok {
            status = ERROR_INVALID_ARGUMENT;
        }
    }

    if status == OK {
        // may request reallocation
        let (irhs, xrhs) = if is_t {
            (sl(irhs, 1), &[][..])
        } else {
            (sl(irhs, nzrhs), sl(xrhs, nzrhs))
        };
        let out = (!p_nzlhs.is_null() && !ilhs.is_null() && !lhs.is_null())
            .then(|| (&mut *p_nzlhs, sl_mut(ilhs, m), sl_mut(lhs, m)));
        status = this.solve_for_update(irhs, xrhs, out, trans);
    }
    this.save(status)
}

/// # Safety
/// As for load(); irhs, xrhs hold nzrhs entries, ilhs, lhs m entries
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn basiclu_solve_sparse(
    istore: *mut Int,
    xstore: *mut f64,
    li: *mut Int,
    lx: *mut f64,
    ui: *mut Int,
    ux: *mut f64,
    wi: *mut Int,
    wx: *mut f64,
    nzrhs: Int,
    irhs: *const Int,
    xrhs: *const f64,
    p_nzlhs: *mut Int,
    ilhs: *mut Int,
    lhs: *mut f64,
    trans: u8,
) -> Int {
    let mut this = match load(istore, xstore, li, lx, ui, ux, wi, wx) {
        Ok(this) => this,
        Err(status) => return status,
    };
    let m = this.m;
    let mut status = OK;
    if li.is_null()
        || lx.is_null()
        || ui.is_null()
        || ux.is_null()
        || wi.is_null()
        || wx.is_null()
        || irhs.is_null()
        || xrhs.is_null()
        || p_nzlhs.is_null()
        || ilhs.is_null()
        || lhs.is_null()
    {
        status = ERROR_ARGUMENT_MISSING;
    } else if this.nupdate < 0 {
        status = ERROR_INVALID_CALL;
    } else if !((0..=m).contains(&nzrhs) && rhs_ok(sl(irhs, nzrhs), m)) {
        // check RHS indices
        status = ERROR_INVALID_ARGUMENT;
    }

    if status == OK {
        this.solve_sparse(
            sl(irhs, nzrhs),
            sl(xrhs, nzrhs),
            &mut *p_nzlhs,
            sl_mut(ilhs, m),
            sl_mut(lhs, m),
            trans,
        );
    }
    this.save(status)
}

/// # Safety
/// As for load()
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn basiclu_update(
    istore: *mut Int,
    xstore: *mut f64,
    li: *mut Int,
    lx: *mut f64,
    ui: *mut Int,
    ux: *mut f64,
    wi: *mut Int,
    wx: *mut f64,
    xtbl: f64,
) -> Int {
    let mut this = match load(istore, xstore, li, lx, ui, ux, wi, wx) {
        Ok(this) => this,
        Err(status) => return status,
    };
    let status = if li.is_null()
        || lx.is_null()
        || ui.is_null()
        || ux.is_null()
        || wi.is_null()
        || wx.is_null()
    {
        ERROR_ARGUMENT_MISSING
    } else if this.nupdate < 0 || this.ftran_for_update < 0 || this.btran_for_update < 0 {
        ERROR_INVALID_CALL
    } else {
        this.update(xtbl)
    };
    this.save(status)
}

// The object interface (basiclu_object.c). The arrays are allocated with
// the C allocator since the struct is owned by C code.

extern "C" {
    fn malloc(size: usize) -> *mut c_void;
    fn calloc(n: usize, size: usize) -> *mut c_void;
    fn realloc(p: *mut c_void, size: usize) -> *mut c_void;
    fn free(p: *mut c_void);
}

/// struct basiclu_object
#[repr(C)]
pub struct BasicluObject {
    pub(super) istore: *mut Int,
    pub(super) xstore: *mut f64,
    pub(super) li: *mut Int,
    pub(super) ui: *mut Int,
    pub(super) wi: *mut Int,
    pub(super) lx: *mut f64,
    pub(super) ux: *mut f64,
    pub(super) wx: *mut f64,
    pub(super) lhs: *mut f64,
    pub(super) ilhs: *mut Int,
    pub(super) nzlhs: Int,
    pub(super) realloc_factor: f64,
}

/// lu_free: deallocate p if not null, return null
unsafe fn lu_free<T>(p: *mut T) -> *mut T {
    if !p.is_null() {
        free(p as *mut c_void);
    }
    std::ptr::null_mut()
}

/// lu_reallocix: reallocate two arrays to nz elements each; on failure the
/// old pointer is kept
unsafe fn lu_reallocix(nz: Int, p_ai: &mut *mut Int, p_ax: &mut *mut f64) -> Int {
    let nz = nz as usize;
    let ainew = realloc(*p_ai as *mut c_void, nz * size_of::<Int>()) as *mut Int;
    if !ainew.is_null() {
        *p_ai = ainew;
    }
    let axnew = realloc(*p_ax as *mut c_void, nz * size_of::<f64>()) as *mut f64;
    if !axnew.is_null() {
        *p_ax = axnew;
    }
    if !ainew.is_null() && !axnew.is_null() {
        OK
    } else {
        ERROR_OUT_OF_MEMORY
    }
}

/// lu_realloc_obj: reallocate Li,Lx and/or Ui,Ux and/or Wi,Wx as requested
/// in xstore
unsafe fn lu_realloc_obj(obj: &mut BasicluObject) -> Int {
    let xstore = obj.xstore;
    let addmem_l = *xstore.add(ADD_MEMORYL) as Int;
    let addmem_u = *xstore.add(ADD_MEMORYU) as Int;
    let addmem_w = *xstore.add(ADD_MEMORYW) as Int;
    let realloc_factor = 1.0f64.max(obj.realloc_factor);
    let mut status = OK;

    for (addmem, mem, pi, px) in [
        (addmem_l, MEMORYL, &mut obj.li, &mut obj.lx),
        (addmem_u, MEMORYU, &mut obj.ui, &mut obj.ux),
        (addmem_w, MEMORYW, &mut obj.wi, &mut obj.wx),
    ] {
        if status == OK && addmem > 0 {
            let mut nelem = (*xstore.add(mem) + addmem as f64) as Int;
            nelem = (nelem as f64 * realloc_factor) as Int;
            status = lu_reallocix(nelem, pi, px);
            if status == OK {
                *xstore.add(mem) = nelem as f64;
            }
        }
    }
    status
}

/// isvalid: test if obj is an allocated BASICLU object
unsafe fn isvalid(obj: *const BasicluObject) -> bool {
    !obj.is_null() && !(*obj).istore.is_null() && !(*obj).xstore.is_null()
}

/// lu_clear_lhs: reset contents of lhs to zero
unsafe fn lu_clear_lhs(obj: &mut BasicluObject) {
    let m = *obj.xstore.add(DIM) as Int;
    let nzsparse = (*obj.xstore.add(SPARSE_THRESHOLD) * m as f64) as Int;
    let nz = obj.nzlhs;
    if nz != 0 {
        let lhs = sl_mut(obj.lhs, m);
        if nz <= nzsparse {
            for &i in sl(obj.ilhs, nz) {
                lhs[i as usize] = 0.0;
            }
        } else {
            lhs.fill(0.0);
        }
        obj.nzlhs = 0;
    }
}

/// # Safety
/// obj must be null or point to a struct basiclu_object
#[no_mangle]
pub unsafe extern "C" fn basiclu_obj_initialize(obj: *mut BasicluObject, m: Int) -> Int {
    let Some(obj) = obj.as_mut() else {
        return ERROR_ARGUMENT_MISSING;
    };
    if m < 0 {
        return ERROR_INVALID_ARGUMENT;
    }

    if m == 0 {
        obj.istore = std::ptr::null_mut();
        obj.xstore = std::ptr::null_mut();
        obj.li = std::ptr::null_mut();
        obj.lx = std::ptr::null_mut();
        obj.ui = std::ptr::null_mut();
        obj.ux = std::ptr::null_mut();
        obj.wi = std::ptr::null_mut();
        obj.wx = std::ptr::null_mut();
        obj.lhs = std::ptr::null_mut();
        obj.ilhs = std::ptr::null_mut();
        obj.nzlhs = 0;
        return OK;
    }

    let imemsize = istore_len(m);
    let xmemsize = xstore_len(m);
    let fmemsize = m as usize; // initial length of Li, Lx, Ui, Ux, Wi, Wx
    let isz = size_of::<Int>();
    let xsz = size_of::<f64>();

    obj.istore = malloc(imemsize * isz) as *mut Int;
    obj.xstore = malloc(xmemsize * xsz) as *mut f64;
    obj.li = malloc(fmemsize * isz) as *mut Int;
    obj.lx = malloc(fmemsize * xsz) as *mut f64;
    obj.ui = malloc(fmemsize * isz) as *mut Int;
    obj.ux = malloc(fmemsize * xsz) as *mut f64;
    obj.wi = malloc(fmemsize * isz) as *mut Int;
    obj.wx = malloc(fmemsize * xsz) as *mut f64;
    obj.lhs = calloc(m as usize, xsz) as *mut f64;
    obj.ilhs = malloc(m as usize * isz) as *mut Int;
    obj.nzlhs = 0;
    obj.realloc_factor = 1.5;

    if obj.istore.is_null()
        || obj.xstore.is_null()
        || obj.li.is_null()
        || obj.lx.is_null()
        || obj.ui.is_null()
        || obj.ux.is_null()
        || obj.wi.is_null()
        || obj.wx.is_null()
        || obj.lhs.is_null()
        || obj.ilhs.is_null()
    {
        basiclu_obj_free(obj);
        return ERROR_OUT_OF_MEMORY;
    }

    // The stores come from malloc: zero them, so that they are initialized
    // memory for Rust (the C code reads some of the uninitialized slots in
    // lu_load before lu_reset overwrites them; the values do not matter)
    std::ptr::write_bytes(obj.istore, 0, imemsize);
    std::ptr::write_bytes(obj.xstore, 0, xmemsize);
    initialize(
        m,
        sl_mut(obj.istore, imemsize as Int),
        sl_mut(obj.xstore, xmemsize as Int),
    );
    *obj.xstore.add(MEMORYL) = fmemsize as f64;
    *obj.xstore.add(MEMORYU) = fmemsize as f64;
    *obj.xstore.add(MEMORYW) = fmemsize as f64;
    OK
}

/// # Safety
/// obj must be null or point to an initialized struct basiclu_object
#[no_mangle]
pub unsafe extern "C" fn basiclu_obj_free(obj: *mut BasicluObject) {
    if let Some(obj) = obj.as_mut() {
        obj.istore = lu_free(obj.istore);
        obj.xstore = lu_free(obj.xstore);
        obj.li = lu_free(obj.li);
        obj.lx = lu_free(obj.lx);
        obj.ui = lu_free(obj.ui);
        obj.ux = lu_free(obj.ux);
        obj.wi = lu_free(obj.wi);
        obj.wx = lu_free(obj.wx);
        obj.lhs = lu_free(obj.lhs);
        obj.ilhs = lu_free(obj.ilhs);
        obj.nzlhs = -1;
    }
}

/// # Safety
/// obj as for basiclu_obj_free; B as for basiclu_factorize
#[no_mangle]
pub unsafe extern "C" fn basiclu_obj_factorize(
    obj: *mut BasicluObject,
    bbegin: *const Int,
    bend: *const Int,
    bi: *const Int,
    bx: *const f64,
) -> Int {
    if !isvalid(obj) {
        return ERROR_INVALID_OBJECT;
    }
    let obj = &mut *obj;
    let mut status = basiclu_factorize(
        obj.istore, obj.xstore, obj.li, obj.lx, obj.ui, obj.ux, obj.wi, obj.wx, bbegin, bend, bi,
        bx, 0,
    );
    while status == REALLOCATE {
        status = lu_realloc_obj(obj);
        if status != OK {
            break;
        }
        status = basiclu_factorize(
            obj.istore, obj.xstore, obj.li, obj.lx, obj.ui, obj.ux, obj.wi, obj.wx, bbegin, bend,
            bi, bx, 1,
        );
    }
    status
}

/// # Safety
/// obj as for basiclu_obj_free; outputs as for basiclu_get_factors
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn basiclu_obj_get_factors(
    obj: *mut BasicluObject,
    rowperm: *mut Int,
    colperm: *mut Int,
    lcolptr: *mut Int,
    lrowidx: *mut Int,
    lvalue: *mut f64,
    ucolptr: *mut Int,
    urowidx: *mut Int,
    uvalue: *mut f64,
) -> Int {
    if !isvalid(obj) {
        return ERROR_INVALID_OBJECT;
    }
    let obj = &mut *obj;
    basiclu_get_factors(
        obj.istore, obj.xstore, obj.li, obj.lx, obj.ui, obj.ux, obj.wi, obj.wx, rowperm, colperm,
        lcolptr, lrowidx, lvalue, ucolptr, urowidx, uvalue,
    )
}

/// # Safety
/// obj as for basiclu_obj_free; rhs, lhs as for basiclu_solve_dense
#[no_mangle]
pub unsafe extern "C" fn basiclu_obj_solve_dense(
    obj: *mut BasicluObject,
    rhs: *const f64,
    lhs: *mut f64,
    trans: u8,
) -> Int {
    if !isvalid(obj) {
        return ERROR_INVALID_OBJECT;
    }
    let obj = &mut *obj;
    basiclu_solve_dense(
        obj.istore, obj.xstore, obj.li, obj.lx, obj.ui, obj.ux, obj.wi, obj.wx, rhs, lhs, trans,
    )
}

/// # Safety
/// obj as for basiclu_obj_free; irhs, xrhs hold nzrhs entries
#[no_mangle]
pub unsafe extern "C" fn basiclu_obj_solve_sparse(
    obj: *mut BasicluObject,
    nzrhs: Int,
    irhs: *const Int,
    xrhs: *const f64,
    trans: u8,
) -> Int {
    if !isvalid(obj) {
        return ERROR_INVALID_OBJECT;
    }
    let obj = &mut *obj;
    lu_clear_lhs(obj);
    basiclu_solve_sparse(
        obj.istore,
        obj.xstore,
        obj.li,
        obj.lx,
        obj.ui,
        obj.ux,
        obj.wi,
        obj.wx,
        nzrhs,
        irhs,
        xrhs,
        &mut obj.nzlhs,
        obj.ilhs,
        obj.lhs,
        trans,
    )
}

/// # Safety
/// obj as for basiclu_obj_free; irhs, xrhs as for basiclu_solve_for_update
#[no_mangle]
pub unsafe extern "C" fn basiclu_obj_solve_for_update(
    obj: *mut BasicluObject,
    nzrhs: Int,
    irhs: *const Int,
    xrhs: *const f64,
    trans: u8,
    want_solution: Int,
) -> Int {
    if !isvalid(obj) {
        return ERROR_INVALID_OBJECT;
    }
    let obj = &mut *obj;
    lu_clear_lhs(obj);
    let mut status = OK;
    while status == OK {
        let p_nzlhs: *mut Int = if want_solution != 0 {
            &mut obj.nzlhs
        } else {
            std::ptr::null_mut()
        };
        status = basiclu_solve_for_update(
            obj.istore, obj.xstore, obj.li, obj.lx, obj.ui, obj.ux, obj.wi, obj.wx, nzrhs, irhs,
            xrhs, p_nzlhs, obj.ilhs, obj.lhs, trans,
        );
        if status != REALLOCATE {
            break;
        }
        status = lu_realloc_obj(obj);
    }
    status
}

/// # Safety
/// obj as for basiclu_obj_free
#[no_mangle]
pub unsafe extern "C" fn basiclu_obj_update(obj: *mut BasicluObject, xtbl: f64) -> Int {
    if !isvalid(obj) {
        return ERROR_INVALID_OBJECT;
    }
    let obj = &mut *obj;
    let mut status = OK;
    while status == OK {
        status = basiclu_update(
            obj.istore, obj.xstore, obj.li, obj.lx, obj.ui, obj.ux, obj.wi, obj.wx, xtbl,
        );
        if status != REALLOCATE {
            break;
        }
        status = lu_realloc_obj(obj);
    }
    status
}
