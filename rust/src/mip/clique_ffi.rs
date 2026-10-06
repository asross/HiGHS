//! `extern "C"` shims for the clique table (highs/mip/HighsCliqueTable.cpp
//! under HIGHS_RUST). The table pointer comes from highs_rs_clique_new; a
//! method that may call into the domain gets it as a raw pointer in a
//! [`Ctx`] (see clique.rs).

use super::clique::{CDom, CMip, CRows, CSepaCliques, CliqueTable, CliqueVar, Ctx};
use crate::ffi::{sl, sl_mut};
use crate::util::cdouble::CDouble;
use crate::util::random::HighsRandom;
use std::ffi::c_void;

#[no_mangle]
pub extern "C" fn highs_rs_clique_new(ncols: i32) -> *mut CliqueTable {
    Box::into_raw(Box::new(CliqueTable::new(ncols)))
}

/// # Safety
/// `t` from highs_rs_clique_new, or null
#[no_mangle]
pub unsafe extern "C" fn highs_rs_clique_free(t: *mut CliqueTable) {
    if !t.is_null() {
        drop(Box::from_raw(t));
    }
}

/// The scalar settings and counters (`which`: 0 inPresolve, 1 numEntries,
/// 2 nfixings, 3 numCliques, 4 isFull, 5 allowParallel)
///
/// # Safety
/// `t` a live table
#[no_mangle]
pub unsafe extern "C" fn highs_rs_clique_get(t: *const CliqueTable, which: i32) -> i32 {
    let t = &*t;
    match which {
        0 => t.in_presolve as i32,
        1 => t.num_entries,
        2 => t.nfixings,
        3 => t.num_cliques_total(),
        4 => t.is_full() as i32,
        _ => t.allow_parallel as i32,
    }
}

/// `which`: 0 inPresolve, 1 maxEntries from a nonzero count (setMaxEntries),
/// 2 minEntriesForParallelism, 3 allowParallel
///
/// # Safety
/// `t` a live table
#[no_mangle]
pub unsafe extern "C" fn highs_rs_clique_set(t: *mut CliqueTable, which: i32, value: i32) {
    let t = &mut *t;
    match which {
        0 => t.in_presolve = value != 0,
        1 => t.set_max_entries(value),
        2 => t.min_entries_for_parallelism = value,
        _ => t.allow_parallel = value != 0,
    }
}

/// The table's generator and neighbourhood query counter, which C++ uses
/// in place (the layouts match)
///
/// # Safety
/// `t` a live table
#[no_mangle]
pub unsafe extern "C" fn highs_rs_clique_randgen(t: *mut CliqueTable) -> *mut HighsRandom {
    std::ptr::addr_of_mut!((*t).randgen)
}

/// # Safety
/// `t` a live table
#[no_mangle]
pub unsafe extern "C" fn highs_rs_clique_num_queries(t: *mut CliqueTable) -> *mut i64 {
    std::ptr::addr_of_mut!((*t).num_neighbourhood_queries)
}

/// The vectors C++ reads and clears (`which`: 0 substitutions, 1 deleted
/// rows, 2 clique extensions): the data, length in `*len`
///
/// # Safety
/// `t` a live table
#[no_mangle]
pub unsafe extern "C" fn highs_rs_clique_vec(t: *mut CliqueTable, which: i32, len: *mut usize) -> *mut c_void {
    let t = &mut *t;
    match which {
        0 => {
            *len = t.substitutions.len();
            t.substitutions.as_mut_ptr() as *mut c_void
        }
        1 => {
            *len = t.deletedrows.len();
            t.deletedrows.as_mut_ptr() as *mut c_void
        }
        _ => {
            *len = t.cliqueextensions.len();
            t.cliqueextensions.as_mut_ptr() as *mut c_void
        }
    }
}

/// # Safety
/// `t` a live table
#[no_mangle]
pub unsafe extern "C" fn highs_rs_clique_vec_clear(t: *mut CliqueTable, which: i32) {
    let t = &mut *t;
    match which {
        0 => t.substitutions.clear(),
        1 => t.deletedrows.clear(),
        _ => t.cliqueextensions.clear(),
    }
}

/// getSubstitution: null if col is not substituted
///
/// # Safety
/// `t` a live table
#[no_mangle]
pub unsafe extern "C" fn highs_rs_clique_substitution(t: *const CliqueTable, col: i32) -> *const c_void {
    match (*t).get_substitution(col) {
        Some(s) => s as *const _ as *const c_void,
        None => std::ptr::null(),
    }
}

/// numCliques(v)
///
/// # Safety
/// `t` a live table
#[no_mangle]
pub unsafe extern "C" fn highs_rs_clique_num_cliques_var(t: *const CliqueTable, v: CliqueVar) -> i32 {
    (*t).num_cliques(v)
}

/// haveCommonClique: with the table's counter if `num_queries` is null
///
/// # Safety
/// `t` a live table, `num_queries` valid or null
#[no_mangle]
pub unsafe extern "C" fn highs_rs_clique_have_common(
    t: *mut CliqueTable,
    num_queries: *mut i64,
    v1: CliqueVar,
    v2: CliqueVar,
) -> bool {
    if num_queries.is_null() {
        (*t).have_common_clique(v1, v2)
    } else {
        (*t).have_common_clique_q(&mut *num_queries, v1, v2)
    }
}

/// findCommonClique: the clique's entries (null if none), length in *len
///
/// # Safety
/// `t` a live table
#[no_mangle]
pub unsafe extern "C" fn highs_rs_clique_find_common(
    t: *mut CliqueTable,
    v1: CliqueVar,
    v2: CliqueVar,
    len: *mut i32,
) -> *const CliqueVar {
    *len = 0;
    let t = &mut *t;
    if v1 == v2 {
        return std::ptr::null();
    }
    let clq = t.find_common_clique_id(v1, v2);
    if clq == -1 {
        return std::ptr::null();
    }
    let c = t.cliques[clq as usize];
    *len = c.end - c.start;
    t.entries.as_ptr().add(c.start as usize)
}

/// resolveSubstitution(col, val, offset)
///
/// # Safety
/// `t` a live table, the pointers valid
#[no_mangle]
pub unsafe extern "C" fn highs_rs_clique_resolve_subst_val(
    t: *const CliqueTable,
    col: *mut i32,
    val: *mut f64,
    offset: *mut f64,
) {
    (*t).resolve_substitution_val(&mut *col, &mut *val, &mut *offset);
}

/// resolveSubstitution(v)
///
/// # Safety
/// `t` a live table
#[no_mangle]
pub unsafe extern "C" fn highs_rs_clique_resolve_subst(t: *const CliqueTable, v: *mut CliqueVar) {
    (*t).resolve_substitution(&mut *v);
}

/// getNumImplications(col) if val < 0, else getNumImplications(col, val)
///
/// # Safety
/// `t` a live table
#[no_mangle]
pub unsafe extern "C" fn highs_rs_clique_num_implications(t: *const CliqueTable, col: i32, val: i32) -> i32 {
    if val < 0 {
        (*t).get_num_implications(col)
    } else {
        (*t).get_num_implications_val(col, val != 0)
    }
}

/// addImplications: only reads the table, so it may run inside a callback
/// of another call on the table, and concurrently on other threads
///
/// # Safety
/// `t` a live table, `dom` valid
#[no_mangle]
pub unsafe extern "C" fn highs_rs_clique_add_implications(t: *const CliqueTable, dom: *const CDom, col: i32, val: i32) {
    (*t).add_implications(&*dom, col, val);
}

/// addClique; changes `vars` in place
///
/// # Safety
/// live table, valid pointers
#[no_mangle]
pub unsafe extern "C" fn highs_rs_clique_add_clique(
    t: *mut CliqueTable,
    dom: *const CDom,
    mip: *const CMip,
    vars: *mut CliqueVar,
    n: i32,
    equality: bool,
    origin: i32,
) {
    Ctx::new(t, *dom).add_clique(&*mip, sl_mut(vars, n), equality, origin);
}

/// removeClique
///
/// # Safety
/// live table
#[no_mangle]
pub unsafe extern "C" fn highs_rs_clique_remove_clique(t: *mut CliqueTable, cliqueid: i32) {
    (*t).remove_clique(cliqueid);
}

/// doAddClique
///
/// # Safety
/// live table, valid pointers
#[no_mangle]
pub unsafe extern "C" fn highs_rs_clique_do_add_clique(
    t: *mut CliqueTable,
    vars: *const CliqueVar,
    n: i32,
    equality: bool,
    origin: i32,
) {
    (*t).do_add_clique(sl(vars, n), equality, origin);
}

/// cliquePartition (with the objective if `objective` is not null): the
/// partition starts are written to `starts` (room for n + 2), their number
/// returned
///
/// # Safety
/// live table, valid pointers
#[no_mangle]
pub unsafe extern "C" fn highs_rs_clique_partition(
    t: *mut CliqueTable,
    objective: *const f64,
    num_col: i32,
    vars: *mut CliqueVar,
    n: i32,
    starts: *mut i32,
) -> i32 {
    let mut ps = Vec::new();
    let vars = sl_mut(vars, n);
    if objective.is_null() {
        (*t).clique_partition(vars, &mut ps);
    } else {
        (*t).clique_partition_obj(sl(objective, num_col), vars, &mut ps);
    }
    sl_mut(starts, n + 2)[..ps.len()].copy_from_slice(&ps);
    ps.len() as i32
}

/// foundCover
///
/// # Safety
/// live table, valid pointers
#[no_mangle]
pub unsafe extern "C" fn highs_rs_clique_found_cover(
    t: *mut CliqueTable,
    dom: *const CDom,
    v1: CliqueVar,
    v2: CliqueVar,
) -> bool {
    Ctx::new(t, *dom).found_cover(v1, v2)
}

/// extractCliques(mipsolver, transformRows)
///
/// # Safety
/// live table, valid pointers
#[no_mangle]
pub unsafe extern "C" fn highs_rs_clique_extract_cliques(
    t: *mut CliqueTable,
    dom: *const CDom,
    mip: *const CMip,
    rows: *const CRows,
    transform_rows: bool,
) {
    Ctx::new(t, *dom).extract_cliques(&*mip, &*rows, transform_rows);
}

/// extractCliquesFromCut
///
/// # Safety
/// live table, valid pointers
#[no_mangle]
pub unsafe extern "C" fn highs_rs_clique_extract_from_cut(
    t: *mut CliqueTable,
    dom: *const CDom,
    mip: *const CMip,
    inds: *const i32,
    vals: *const f64,
    len: i32,
    rhs: f64,
) {
    Ctx::new(t, *dom).extract_cliques_from_cut(&*mip, sl(inds, len), sl(vals, len), rhs);
}

/// extractObjCliques after the C++ part (see clique.rs)
///
/// # Safety
/// live table, valid pointers
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn highs_rs_clique_extract_obj(
    t: *mut CliqueTable,
    dom: *const CDom,
    mip: *const CMip,
    nbin: i32,
    vals: *const f64,
    inds: *const i32,
    len: i32,
    rhs: f64,
    minact_hi: f64,
    minact_lo: f64,
) {
    let minact = CDouble { hi: minact_hi, lo: minact_lo };
    Ctx::new(t, *dom).extract_obj_cliques(&*mip, nbin as usize, sl(vals, len), sl(inds, len), rhs, minact);
}

/// vertexInfeasible
///
/// # Safety
/// live table, valid pointers
#[no_mangle]
pub unsafe extern "C" fn highs_rs_clique_vertex_infeasible(t: *mut CliqueTable, dom: *const CDom, col: i32, val: i32) {
    Ctx::new(t, *dom).vertex_infeasible(col, val);
}

/// cleanupFixed
///
/// # Safety
/// live table, valid pointers
#[no_mangle]
pub unsafe extern "C" fn highs_rs_clique_cleanup_fixed(t: *mut CliqueTable, dom: *const CDom) {
    Ctx::new(t, *dom).cleanup_fixed();
}

/// runCliqueMerging(globaldomain)
///
/// # Safety
/// live table, valid pointers
#[no_mangle]
pub unsafe extern "C" fn highs_rs_clique_run_merging(t: *mut CliqueTable, dom: *const CDom) {
    Ctx::new(t, *dom).run_clique_merging();
}

/// separateCliques
///
/// # Safety
/// live table, valid pointers (see Ctx::separate_cliques)
#[no_mangle]
pub unsafe extern "C" fn highs_rs_clique_separate(
    t: *mut CliqueTable,
    dom: *const CDom,
    s: *const CSepaCliques,
    randgen: *mut HighsRandom,
    local_num_queries: *mut i64,
) {
    Ctx::new(t, *dom).separate_cliques(&*s, randgen, local_num_queries);
}

/// computeMaximalCliques: each clique handed to `push(out, data, len)`
///
/// # Safety
/// live table, valid pointers
#[no_mangle]
pub unsafe extern "C" fn highs_rs_clique_maximal_cliques(
    t: *const CliqueTable,
    vars: *const CliqueVar,
    n: i32,
    feastol: f64,
    out: *mut c_void,
    push: unsafe extern "C" fn(*mut c_void, *const CliqueVar, i32),
) {
    for c in (*t).compute_maximal_cliques(sl(vars, n), feastol) {
        push(out, c.as_ptr(), c.len() as i32);
    }
}

/// rebuild
///
/// # Safety
/// live table, valid pointers (orig2reducedcol for every column of the
/// table, keep for the ncols reduced columns)
#[no_mangle]
pub unsafe extern "C" fn highs_rs_clique_rebuild(
    t: *mut CliqueTable,
    ncols: i32,
    orig2reducedcol: *const i32,
    norig: i32,
    keep: *const u8,
) {
    (*t).rebuild(ncols, sl(orig2reducedcol, norig), sl(keep, ncols));
}

/// buildFrom
///
/// # Safety
/// live tables, valid pointers
#[no_mangle]
pub unsafe extern "C" fn highs_rs_clique_build_from(
    t: *mut CliqueTable,
    orig_lower: *const f64,
    orig_upper: *const f64,
    num_col: i32,
    init: *const CliqueTable,
) {
    (*t).build_from(sl(orig_lower, num_col), sl(orig_upper, num_col), &*init);
}
