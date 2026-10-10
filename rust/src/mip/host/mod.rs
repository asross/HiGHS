//! The MIP solver's objects, owned by Rust: what the C++ shells of
//! highs/mip held (HighsMipSolver and HighsMipSolverData with their
//! containers, the workers, HighsDomain's scalars and pool propagation
//! shells, HighsLpRelaxation's glue, HighsSearch's local domain,
//! HighsSeparation and its separators, HighsObjectiveFunction, the
//! presolve's host and the postsolve stack), and the functions the ported
//! code calls through its callback tables ([`fns`]: glue.rs `CMipFns`,
//! search.rs `CSearchFns`, lp_relaxation.rs `CLpFns`, separation.rs
//! `CSepaFns`, the domain's, clique table's, implications', reduced cost
//! fixing's and separators' hosts), implemented on these objects in the
//! order the C++ made the same steps. The objects refer to each other
//! through raw pointers, as the C++ did: every object that another points
//! to is boxed and outlives the pointer (the solver's data owns them).
//!
//! The only C++ left of a MIP solve is the Highs object's: its
//! HighsProfiling, the user callback (HighsCallback), the improving
//! solution file and the log options, reached through [`HighsFns`]
//! (HighsMipHost.cpp, registered once). A Highs object's MIP solve and
//! presolve are its engine's (lp_data/top.rs: the solver built from the
//! engine's model and option values).

use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};

pub mod dom;
pub mod fns;
pub mod lp;
pub mod pools;
pub mod presolve;
pub mod search;
pub mod sepa;
pub mod solver;
pub mod tables;
pub mod worker;

pub type P = *mut c_void;

/// The data_out of a callback (setup.rs CallbackOut) with the solution and
/// message to pass
pub use super::setup::CallbackOut;

/// The C++ functions of the Highs object a MIP solve calls
/// (HighsRunRust.cpp: the MipHost context `ctx` of the solve)
#[repr(C)]
pub struct HighsFns {
    /// HighsProfiling `p`: code (see [`prof`]), clock index (into the
    /// C++ table of clock ids, rsMipClocks), argument
    pub profiling: unsafe extern "C" fn(P, i32, i32, i64) -> f64,
    /// The HighsCallback: with `out` null, `which` 0 callbackActive(type),
    /// 1 whether a user callback is set, 2 data_in.user_has_solution;
    /// otherwise data_out set from `out` (with the solution of the given
    /// length, if not null, as mip_solution) and callbackAction(type, the
    /// message of the given length), returning the interrupt
    pub callback: unsafe extern "C" fn(P, i32, i32, *const CallbackOut, *const f64, i32, *const u8, i32) -> bool,
    /// data_in.user_solution (data, length)
    pub user_solution: unsafe extern "C" fn(P, *mut i32) -> *const f64,
    /// clearHighsCallbackOutput and the cut pool fields of data_out
    /// (num_col, num_cut, lower, upper, start, index, value, nnz)
    #[allow(clippy::type_complexity)]
    pub cut_pool_output: unsafe extern "C" fn(P, i32, i32, *const f64, *const f64, *const i32, *const i32, *const f64, i32),
    /// The improving solution file: 0 open, 1 write the solution (the
    /// objective and values), 2 close
    pub improving_file: unsafe extern "C" fn(P, i32, *const f64, i32),
}

static HIGHS_FNS: AtomicUsize = AtomicUsize::new(0);

/// Registers the Highs object's functions (HighsRunRust.cpp, once)
///
/// # Safety
/// `f` a static table
#[no_mangle]
pub unsafe extern "C" fn highs_rs_mip_register(f: *const HighsFns) {
    HIGHS_FNS.store(f as usize, Ordering::Relaxed);
}

/// The registered functions, if any (the crate's tests have none)
pub fn highs_fns() -> Option<&'static HighsFns> {
    match HIGHS_FNS.load(Ordering::Relaxed) {
        0 => None,
        // SAFETY: a static C++ table registered by highs_rs_mip_register
        f => Some(unsafe { &*(f as *const HighsFns) }),
    }
}

/// The profiling operations of HighsFns::profiling
pub mod prof {
    pub const START: i32 = 0;
    pub const STOP: i32 = 1;
    pub const RUNNING: i32 = 2;
    /// read(clock, record_type arg)
    pub const READ: i32 = 3;
    /// numCall(clock, record_type arg)
    pub const NUM_CALL: i32 = 4;
    /// mip_
    pub const MIP: i32 = 5;
    pub const IS_SUBMIP: i32 = 6;
    /// setSubMip(arg != 0)
    pub const SET_SUBMIP: i32 = 7;
    /// sub_solver_
    pub const SUB_SOLVER: i32 = 8;
    /// myThread()
    pub const MY_THREAD: i32 = 9;
    /// start(clock, restart = true)
    pub const RESTART: i32 = 10;
    /// solveCall(model arg: 0 "LP0", 1 "LP1", 2 "LP2", 3 "LP3", 4 "MIP",
    /// submip clock != 0)
    pub const SOLVE_CALL: i32 = 11;
    /// HighsProfiling record types
    pub const MIP_RECORD: i64 = 0;
    pub const SUB_MIP_RECORD: i64 = 1;

    /// The clock indices of the C++ table (rsMipClocks): root.rs and
    /// driver.rs use 0 to 47 (their mod clk), these the rest
    pub const EVALUATE_NODE1: i32 = 34;
    pub const NODE_PRUNED_LOOP: i32 = 35;
    pub const NODE_SEARCH_SEPARATION: i32 = 36;
    pub const BACKTRACK_PLUNGE: i32 = 37;
    pub const DIVE_EVALUATE_NODE: i32 = 38;
    pub const DIVE_PRIMAL_HEURISTICS: i32 = 39;
    pub const DIVE_RANDOMIZED_ROUNDING: i32 = 40;
    pub const DIVE_RENS: i32 = 41;
    pub const DIVE_RINS: i32 = 42;
    pub const THE_DIVE: i32 = 43;
    pub const OPEN_NODES_TO_QUEUE0: i32 = 44;
    pub const PRESOLVE_TIME: i32 = 23;
    pub const SOLVE_TIME: i32 = 26;
    pub const POSTSOLVE_TIME: i32 = 47;
    pub const SUB_MIP_SOLVE: i32 = 48;
    pub const SUB_SOLVER_SUB_MIP: i32 = 49;
    pub const SUB_SOLVER_IPX_AC: i32 = 50;
    pub const PROBING_IMPLICATIONS: i32 = 51;
    pub const PROBING_PRESOLVE: i32 = 52;
    pub const ENUMERATION_PRESOLVE: i32 = 53;
}

/// A HighsProfiling (the Highs object's, shared by the sub-MIPs), or none
/// (the concurrent helper's single-thread profiling, which nothing reads)
#[derive(Clone, Copy)]
pub struct Prof {
    pub p: P,
}

impl Prof {
    pub fn none() -> Prof {
        Prof { p: std::ptr::null_mut() }
    }
    fn call(&self, code: i32, clock: i32, arg: i64) -> f64 {
        match (self.p.is_null(), highs_fns()) {
            // SAFETY: the Highs object's profiling, alive during the solve
            (false, Some(f)) => unsafe { (f.profiling)(self.p, code, clock, arg) },
            _ => 0.0,
        }
    }
    pub fn start(&self, clock: i32) {
        self.call(prof::START, clock, 0);
    }
    pub fn restart(&self, clock: i32) {
        self.call(prof::RESTART, clock, 0);
    }
    pub fn stop(&self, clock: i32) {
        self.call(prof::STOP, clock, 0);
    }
    pub fn running(&self, clock: i32) -> bool {
        self.call(prof::RUNNING, clock, 0) != 0.0
    }
    pub fn read(&self, clock: i32, record: i64) -> f64 {
        self.call(prof::READ, clock, record)
    }
    pub fn num_call(&self, clock: i32, record: i64) -> i32 {
        self.call(prof::NUM_CALL, clock, record) as i32
    }
    pub fn mip(&self) -> bool {
        self.call(prof::MIP, 0, 0) != 0.0
    }
    pub fn is_submip(&self) -> bool {
        self.call(prof::IS_SUBMIP, 0, 0) != 0.0
    }
    pub fn set_submip(&self, submip: bool) {
        self.call(prof::SET_SUBMIP, 0, submip as i64);
    }
    pub fn sub_solver(&self) -> bool {
        self.call(prof::SUB_SOLVER, 0, 0) != 0.0
    }
    pub fn my_thread(&self) -> i32 {
        self.call(prof::MY_THREAD, 0, 0) as i32
    }
    /// solveCall: model 0 "LP0" .. 3 "LP3", 4 "MIP"
    pub fn solve_call(&self, model: i64, submip: bool) {
        self.call(prof::SOLVE_CALL, submip as i32, model);
    }
}

/// A raw pointer that a task may carry to another thread (the C++ lambdas
/// captured references; the task group is synced before the pointee goes)
pub struct SendPtr<T>(pub *mut T);
impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SendPtr<T> {}
// SAFETY: see above
unsafe impl<T> Send for SendPtr<T> {}

/// An owned object that others point to (the C++ containers' elements):
/// a raw pointer from a Box, freed on drop. The objects of the solver
/// refer to each other through raw pointers, as in the C++, so their
/// accessors hand out references from the pointer, not from `&self`
pub struct Own<T>(pub(crate) *mut T);

impl<T> Own<T> {
    pub fn new(b: Box<T>) -> Own<T> {
        Own(Box::into_raw(b))
    }
    pub fn ptr(&self) -> *mut T {
        self.0
    }
    /// The object, as the C++ used its containers' elements
    #[allow(clippy::mut_from_ref)]
    pub fn get<'a>(&self) -> &'a mut T {
        // SAFETY: the owned object, alive while self is
        unsafe { &mut *self.0 }
    }
}

impl<T> Drop for Own<T> {
    fn drop(&mut self) {
        // SAFETY: from Box::into_raw
        unsafe { drop(Box::from_raw(self.0)) }
    }
}

impl<T> std::ops::Deref for Own<T> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: the owned object
        unsafe { &*self.0 }
    }
}

impl<T> std::ops::DerefMut for Own<T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: the owned object
        unsafe { &mut *self.0 }
    }
}

// SAFETY: as the objects' own Send
unsafe impl<T: Send> Send for Own<T> {}

/// std::max / std::min of doubles as the C++ (the first if unordered)
#[inline(always)]
pub fn max2(a: f64, b: f64) -> f64 {
    if a < b {
        b
    } else {
        a
    }
}
#[inline(always)]
pub fn min2(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

pub const INF: f64 = f64::INFINITY;
pub const IINF: i32 = i32::MAX;
