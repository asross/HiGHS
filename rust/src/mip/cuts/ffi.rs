//! `extern "C"` shims for the cut separation (highs/mip/HighsSeparationRust.cpp
//! under HIGHS_RUST).

use super::cut_generation::{CutEnv, CutGeneration};
use super::path::PathSeparator;
use super::round::{CSepaLp, Host, Live, SepaRound};
use super::tableau::TableauSeparator;
use crate::ffi::sl;
use crate::util::random::HighsRandom;

/// # Safety
/// `c` must be valid as described for [`CSepaLp`], for the life of the
/// returned round
#[no_mangle]
pub unsafe extern "C" fn highs_rs_sepa_round_new(c: *const CSepaLp) -> *mut SepaRound {
    Box::into_raw(SepaRound::new(&*c))
}

/// # Safety
/// `r` from highs_rs_sepa_round_new, or null
#[no_mangle]
pub unsafe extern "C" fn highs_rs_sepa_round_free(r: *mut SepaRound) {
    if !r.is_null() {
        drop(Box::from_raw(r));
    }
}

/// HighsTransformedLp::boundDistance
///
/// # Safety
/// `r` a live round
#[no_mangle]
pub unsafe extern "C" fn highs_rs_sepa_round_bound_distance(r: *const SepaRound, col: i32) -> f64 {
    let r = &*r;
    r.cols[col as usize].bound_dist()
}

#[no_mangle]
pub extern "C" fn highs_rs_path_new(seed: u32) -> *mut PathSeparator {
    Box::into_raw(Box::new(PathSeparator::new(seed)))
}

/// # Safety
/// `p` from highs_rs_path_new, or null
#[no_mangle]
pub unsafe extern "C" fn highs_rs_path_free(p: *mut PathSeparator) {
    if !p.is_null() {
        drop(Box::from_raw(p));
    }
}

/// HighsPathSeparator::separateLpSolution
///
/// # Safety
/// live handles
#[no_mangle]
pub unsafe extern "C" fn highs_rs_path_separate(p: *mut PathSeparator, r: *mut SepaRound, cutgen_seed: u32) {
    (*p).separate(&mut *r, cutgen_seed);
}

#[no_mangle]
pub extern "C" fn highs_rs_tableau_new() -> *mut TableauSeparator {
    Box::into_raw(Box::default())
}

/// # Safety
/// `t` from highs_rs_tableau_new, or null
#[no_mangle]
pub unsafe extern "C" fn highs_rs_tableau_free(t: *mut TableauSeparator) {
    if !t.is_null() {
        drop(Box::from_raw(t));
    }
}

/// HighsTableauSeparator::separateLpSolution after the hasInvert check
///
/// # Safety
/// live handles
#[no_mangle]
pub unsafe extern "C" fn highs_rs_tableau_separate(
    t: *mut TableauSeparator,
    r: *mut SepaRound,
    cutgen_seed: u32,
    num_calls: i32,
    basisinds: *const i32,
    lp_iterations: i64,
) {
    let r = &mut *r;
    let mut cutgen = CutGeneration::new(cutgen_seed, r.feastol, r.epsilon);
    let basisinds = sl(basisinds, r.num_row as i32);
    (*t).separate(r, &mut cutgen, num_calls, basisinds, lp_iterations);
}

/// HighsModkSeparator::separateLpSolution
///
/// # Safety
/// a live round
#[no_mangle]
pub unsafe extern "C" fn highs_rs_modk_separate(r: *mut SepaRound, cutgen_seed: u32) {
    let r = &mut *r;
    let mut cutgen = CutGeneration::new(cutgen_seed, r.feastol, r.epsilon);
    super::modk::separate(r, &mut cutgen);
}

/// What generateConflict needs
#[repr(C)]
pub struct CConflict {
    pub num_col: i32,
    /// globaldom and localdom column bounds (num_col)
    pub glb: *const f64,
    pub gub: *const f64,
    pub llb: *const f64,
    pub lub: *const f64,
    /// mipsolver.model_->integrality_ (num_col)
    pub integrality: *const u8,
    /// lpRelaxation.numCols()
    pub num_lp_cols: i32,
    pub feastol: f64,
    pub epsilon: f64,
    /// random_seed + numLpIterations + cutpool.getNumCuts()
    pub seed: u32,
    /// add_cut, num_nodes_down/up are used
    pub host: Host,
}

struct ConflictEnv<'a> {
    c: &'a CConflict,
    glb: Live,
    gub: Live,
    integrality: &'a [u8],
}

impl CutEnv for ConflictEnv<'_> {
    fn is_integral(&self, idx: usize) -> bool {
        self.integrality[idx] != 0
    }
    fn glb(&self, col: usize) -> f64 {
        self.glb.at(col)
    }
    fn gub(&self, col: usize) -> f64 {
        self.gub.at(col)
    }
    fn sol(&self, _col: usize) -> f64 {
        unreachable!("generateConflict computes no violation")
    }
    fn num_lp_cols(&self) -> usize {
        self.c.num_lp_cols as usize
    }
    fn add_cut(&mut self, inds: &mut [i32], vals: &mut [f64], rhs: f64, integral: bool, conflict: bool) -> i32 {
        let h = &self.c.host;
        // SAFETY: inds and vals have len entries; the global domain does not
        // change in addCut before the conflict's last read of it
        unsafe { (h.add_cut)(h.ctx, inds.as_mut_ptr(), vals.as_mut_ptr(), inds.len() as i32, rhs, integral, conflict) }
    }
    fn num_nodes_down(&self, col: i32) -> i64 {
        // SAFETY: a C++ query
        unsafe { (self.c.host.num_nodes_down)(self.c.host.ctx, col) }
    }
    fn num_nodes_up(&self, col: i32) -> i64 {
        // SAFETY: a C++ query
        unsafe { (self.c.host.num_nodes_up)(self.c.host.ctx, col) }
    }
}

/// HighsCutGeneration::generateConflict on copies of the proof
///
/// # Safety
/// `c` valid as described; inds/vals with len entries
#[no_mangle]
pub unsafe extern "C" fn highs_rs_generate_conflict(
    c: *const CConflict,
    inds: *const i32,
    vals: *const f64,
    len: i32,
    rhs: f64,
) -> bool {
    let c = &*c;
    let n = c.num_col;
    let mut env = ConflictEnv { c, glb: Live::new(c.glb, n as usize),
        gub: Live::new(c.gub, n as usize),
         integrality: sl(c.integrality, n) };
    let (llb, lub) = (sl(c.llb, n), sl(c.lub, n));
    let mut proofinds = sl(inds, len).to_vec();
    let mut proofvals = sl(vals, len).to_vec();
    let mut proofrhs = rhs;
    let mut cutgen = CutGeneration::new(c.seed, c.feastol, c.epsilon);
    cutgen.generate_conflict(&mut env, |col| (llb[col], lub[col]), &mut proofinds, &mut proofvals, &mut proofrhs)
}

/// HighsRandom(seed)'s state, for a C++ HighsRandom's initial state
#[no_mangle]
pub extern "C" fn highs_rs_random_state(seed: u32) -> u64 {
    HighsRandom::new(seed).state()
}
