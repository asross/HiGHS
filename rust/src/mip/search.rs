//! HighsSearch (highs/mip/HighsSearch.cpp): the branch-and-bound dive and
//! branching driver: node evaluation, reliability branching with strong
//! branching, child selection, backtracking (with the plunge heuristic),
//! and moving open nodes to the node queue.
//!
//! The node stack, the statistics and the branching state are Rust's; the
//! C++ class keeps the local domain (external code uses it) and is reached
//! through [`CSearchFns`]: the domain operations, the LP relaxation (also
//! strong branching's playground and the fallback LP of branch), the
//! symmetries, conflicts from LP proofs, reduced cost fixing, incumbents and
//! limits. The node bases and stabilizer orbits are C++ shared pointers held
//! by [`Shared`]. The pseudocosts and the node queue are Rust and used
//! directly. No callback re-enters the search.
//!
//! clang fuses `minrel - r * (minrel - 1)` of branch (and the pseudocost
//! scores, see pseudocost.rs).

use super::domain::{DomChg, LOWER, UPPER};
use super::nodequeue::{ldexp1, NodeQueue};
use super::lp_relaxation::{self, LpRelax};
use super::pseudocost::Pseudocost;
use crate::util::cdouble::CDouble;
use crate::util::fma::ClangFma;
use crate::util::random::HighsRandom;
use std::ffi::c_void;

const INF: f64 = f64::INFINITY;

/// NodeResult
pub const BOUND_EXCEEDING: i32 = 0;
pub const DOMAIN_INFEASIBLE: i32 = 1;
pub const LP_INFEASIBLE: i32 = 2;
pub const BRANCHED: i32 = 3;
pub const SUB_OPTIMAL: i32 = 4;
pub const OPEN: i32 = 5;

/// ChildSelectionRule
const CHILD_UP: i32 = 0;
const CHILD_DOWN: i32 = 1;
const CHILD_ROOT_SOL: i32 = 2;
const CHILD_OBJ: i32 = 3;
const CHILD_RANDOM: i32 = 4;
const CHILD_BEST_COST: i32 = 5;
const CHILD_WORST_COST: i32 = 6;
const CHILD_DISJUNCTION: i32 = 7;
pub const CHILD_HYBRID_INFERENCE_COST: i32 = 8;

const UP_RELIABLE: i32 = 1;
const DOWN_RELIABLE: i32 = 2;

/// The solution sources of addIncumbent
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Sources {
    pub heuristic: i32,
    pub branching: i32,
    pub evaluate_node: i32,
}

/// A C++ std::shared_ptr (to a HighsBasis or StabilizerOrbits), boxed on
/// the C++ heap; null for an empty pointer
pub struct Shared {
    p: *mut c_void,
    fns: *const CSearchFns,
}

impl Shared {
    fn null(fns: *const CSearchFns) -> Self {
        Shared { p: std::ptr::null_mut(), fns }
    }
    fn is_some(&self) -> bool {
        !self.p.is_null()
    }
    /// std::move: leaves this empty
    fn take(&mut self) -> Shared {
        Shared { p: std::mem::replace(&mut self.p, std::ptr::null_mut()), fns: self.fns }
    }
    /// The box, handed over to the C++
    fn into_raw(mut self) -> *mut c_void {
        std::mem::replace(&mut self.p, std::ptr::null_mut())
    }
}

impl Clone for Shared {
    fn clone(&self) -> Self {
        if self.p.is_null() {
            return Shared::null(self.fns);
        }
        // SAFETY: the C++ copies the shared pointer
        Shared { p: unsafe { ((*self.fns).shared_clone)(self.p) }, fns: self.fns }
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        if !self.p.is_null() {
            // SAFETY: the C++ frees its box
            unsafe { ((*self.fns).shared_free)(self.p) };
        }
    }
}

/// A fractional integer column of the LP solution
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FracInt {
    pub col: i32,
    pub val: f64,
}

/// The C++ side of the search, called with the C++ HighsSearch
#[repr(C)]
pub struct CSearchFns {
    pub shared_clone: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    pub shared_free: unsafe extern "C" fn(*mut c_void),
    // the local domain
    pub change_bound: unsafe extern "C" fn(*mut c_void, DomChg),
    pub propagate: unsafe extern "C" fn(*mut c_void),
    pub infeasible: unsafe extern "C" fn(*mut c_void) -> bool,
    pub backtrack: unsafe extern "C" fn(*mut c_void) -> DomChg,
    pub backtrack_to_global: unsafe extern "C" fn(*mut c_void),
    /// the domain change stack (data, length)
    pub stack: unsafe extern "C" fn(*mut c_void, *mut i32) -> *const DomChg,
    pub num_changed_cols: unsafe extern "C" fn(*mut c_void) -> i32,
    /// clearChangedCols(start), all for start < 0
    pub clear_changed_cols: unsafe extern "C" fn(*mut c_void, i32),
    /// conflictAnalysis with the worker's conflict pool, global domain and
    /// the search's pseudocosts
    pub conflict_analysis: unsafe extern "C" fn(*mut c_void),
    /// the column bounds (lower, upper; numCol entries)
    pub bounds: unsafe extern "C" fn(*mut c_void, *mut *const f64, *mut *const f64),
    pub objective_lower_bound: unsafe extern "C" fn(*mut c_void) -> f64,
    /// getColLowerPos (upper false) / getColUpperPos of col at the stack end
    pub col_pos: unsafe extern "C" fn(*mut c_void, i32, bool) -> i32,
    /// isGlobalBinary (global false: model bounds) / the worker's global
    /// domain's isBinary
    pub is_binary: unsafe extern "C" fn(*mut c_void, i32, bool) -> bool,
    /// setDomainChangeStack(stack, branchings)
    pub set_stack: unsafe extern "C" fn(*mut c_void, *const DomChg, i32, *const i32, i32),
    /// getBranchingPositions (data, length)
    pub branching_positions: unsafe extern "C" fn(*mut c_void, *mut i32) -> *const i32,
    /// the reduced domain change stack to the node queue (a HighsNodeQueue)
    pub node_to_queue: unsafe extern "C" fn(*mut c_void, *mut c_void, f64, f64, i32) -> f64,
    // the LP relaxation
    pub lp_flush_domain: unsafe extern "C" fn(*mut c_void),
    pub lp_set_objective_limit: unsafe extern "C" fn(*mut c_void, f64),
    pub lp_resolve: unsafe extern "C" fn(*mut c_void) -> i32,
    /// the current LP relaxation (Rust; the fallback LP while one is swapped in)
    pub lp_rust: unsafe extern "C" fn(*mut c_void) -> *mut LpRelax,
    /// storeBasis (get false, returns null) / getStoredBasis (get true)
    pub lp_store_basis: unsafe extern "C" fn(*mut c_void, bool) -> *mut c_void,
    /// setStoredBasis (taking the box)
    pub lp_set_stored_basis: unsafe extern "C" fn(*mut c_void, *mut c_void),
    pub lp_recover_basis: unsafe extern "C" fn(*mut c_void),
    /// the number of rows of a basis box
    pub basis_rows: unsafe extern "C" fn(*mut c_void) -> i32,
    pub lp_perform_aging: unsafe extern "C" fn(*mut c_void),
    pub lp_degenerate_duals: unsafe extern "C" fn(*mut c_void, f64),
    pub lp_degeneracy: unsafe extern "C" fn(*mut c_void) -> f64,
    pub playground_new: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    pub playground_solve: unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32,
    pub playground_free: unsafe extern "C" fn(*mut c_void),
    /// the steps of branch's fallback LP: 0 iteration limit, fresh LP
    /// swapped in, presolve on; 1 clear, primal simplex; 2 dual simplex; 3
    /// clear, IPM; 4 the warning; 5 the old LP swapped back
    pub lp_fallback: unsafe extern "C" fn(*mut c_void, i32),
    // symmetries
    pub num_perms: unsafe extern "C" fn(*mut c_void) -> i32,
    pub global_orbits: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    pub column_position: unsafe extern "C" fn(*mut c_void, i32) -> i32,
    pub compute_stabilizer_orbits: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    /// orbitCols.empty() (col < 0) / isStabilized(col) of an orbits box
    pub orbits_query: unsafe extern "C" fn(*mut c_void, i32) -> bool,
    pub orbital_fixing: unsafe extern "C" fn(*mut c_void, *mut c_void),
    pub propagate_orbitopes: unsafe extern "C" fn(*mut c_void),
    // the MIP solver
    /// 0 feastol, 1 epsilon, 2 upper limit, 3 optimality limit
    pub mip_value: unsafe extern "C" fn(*mut c_void, i32) -> f64,
    /// 0 heuristic, 1 total, 2 strong branching LP iterations of mipdata;
    /// 3 whether graph LNS alternates with the tree (lns_tree_next >= 0)
    pub mip_stat: unsafe extern "C" fn(*mut c_void, i32) -> i64,
    pub check_limits: unsafe extern "C" fn(*mut c_void, i64) -> bool,
    pub add_incumbent: unsafe extern "C" fn(*mut c_void, *const f64, i32, f64, i32),
    pub add_bound_exceeding_conflict: unsafe extern "C" fn(*mut c_void),
    pub add_infeasible_conflict: unsafe extern "C" fn(*mut c_void),
    pub propagate_redcost: unsafe extern "C" fn(*mut c_void),
    /// the error log of selectBranchingCandidate (upper: the upper bound
    /// residual)
    pub log_frac_error: unsafe extern "C" fn(*mut c_void, bool, f64, f64, f64, f64, f64),
    /// the model data (refreshed at each call into the search)
    pub model: unsafe extern "C" fn(*mut c_void, *mut CModel),
}

/// The model data of the search (C++-owned, valid during a call)
#[repr(C)]
pub struct CModel {
    pub num_col: i32,
    pub col_cost: *const f64,
    /// HighsVarType (0 continuous, 1 integer)
    pub integrality: *const u8,
    pub root_lp_sol: *const f64,
    pub num_root_lp_sol: i32,
    pub integral_cols: *const i32,
    pub num_integral_cols: i32,
    pub sources: Sources,
}

/// The statistics and settings C++ reads and writes in place
#[repr(C)]
pub struct Stats {
    pub nnodes: i64,
    pub nleaves: i64,
    pub lpiterations: i64,
    pub heurlpiterations: i64,
    pub sblpiterations: i64,
    pub upper_limit: f64,
    pub treeweight: CDouble,
    pub depthoffset: i32,
    pub inbranching: bool,
    pub inheuristic: bool,
    pub count_tree_weight: bool,
    pub childselrule: i32,
}

struct NodeData {
    lower_bound: f64,
    estimate: f64,
    branching_point: f64,
    lp_objective: f64,
    other_child_lb: f64,
    node_basis: Shared,
    stabilizer_orbits: Shared,
    branchingdecision: DomChg,
    domgchg_stack_pos: i32,
    skip_depth_count: u8,
    opensubtrees: u8,
}

impl NodeData {
    fn new(parentlb: f64, parentestimate: f64, basis: Shared, orbits: Shared) -> Self {
        NodeData {
            lower_bound: parentlb,
            estimate: parentestimate,
            branching_point: 0.0,
            lp_objective: -INF,
            other_child_lb: parentlb,
            node_basis: basis,
            stabilizer_orbits: orbits,
            branchingdecision: DomChg { boundval: 0.0, column: -1, boundtype: LOWER },
            domgchg_stack_pos: -1,
            skip_depth_count: 0,
            opensubtrees: 2,
        }
    }
}

pub struct Search {
    pub stats: Stats,
    fns: *const CSearchFns,
    ctx: *mut c_void,
    model: CModel,
    random: HighsRandom,
    subrootsol: Vec<f64>,
    nodestack: Vec<NodeData>,
    /// reliableatnode: the flags by column and the columns set
    reliable_flags: Vec<u8>,
    reliable_cols: Vec<i32>,
    // per call: the pseudocosts and the global node queue
    ps: *mut Pseudocost,
    queue: *const NodeQueue,
}

/// std::max(a, b)
#[inline(always)]
fn max2(a: f64, b: f64) -> f64 {
    if a < b {
        b
    } else {
        a
    }
}

/// std::min(a, b)
#[inline(always)]
fn min2(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// fractionality(x): the distance to std::round(x)
#[inline(always)]
fn fractionality(x: f64) -> f64 {
    (x - x.round()).abs()
}

macro_rules! cb {
    ($s:expr, $f:ident $(, $a:expr)*) => {
        // SAFETY: the C++ callbacks of the search, called with its object
        unsafe { ((*$s.fns).$f)($s.ctx $(, $a)*) }
    };
}

impl Search {
    /// # Safety
    /// `fns` static, `ctx` the C++ HighsSearch, `model` valid while the
    /// search lives
    pub unsafe fn new(fns: *const CSearchFns, ctx: *mut c_void, model: CModel, submip: bool) -> Self {
        let ncol = model.num_col.max(0) as usize;
        Search {
            stats: Stats {
                nnodes: 0,
                nleaves: 0,
                lpiterations: 0,
                heurlpiterations: 0,
                sblpiterations: 0,
                upper_limit: INF,
                treeweight: CDouble::from(0.0),
                depthoffset: 0,
                inbranching: false,
                inheuristic: false,
                count_tree_weight: true,
                childselrule: if submip { CHILD_HYBRID_INFERENCE_COST } else { CHILD_ROOT_SOL },
            },
            fns,
            ctx,
            model,
            random: HighsRandom::new(0),
            subrootsol: Vec::new(),
            nodestack: Vec::new(),
            reliable_flags: vec![0; ncol],
            reliable_cols: Vec::new(),
            ps: std::ptr::null_mut(),
            queue: std::ptr::null(),
        }
    }

    /// The start of a call from C++: the pseudocosts, the global node queue
    /// and the model data
    pub(crate) fn enter(&mut self, ps: *mut Pseudocost, nq: *const NodeQueue) {
        self.ps = ps;
        self.queue = nq;
        let mut m = CModel {
            num_col: 0,
            col_cost: std::ptr::null(),
            integrality: std::ptr::null(),
            root_lp_sol: std::ptr::null(),
            num_root_lp_sol: 0,
            integral_cols: std::ptr::null(),
            num_integral_cols: 0,
            sources: self.model.sources,
        };
        // SAFETY: the C++ fills the model data
        unsafe { ((*self.fns).model)(self.ctx, &mut m) };
        if m.num_col as usize != self.reliable_flags.len() {
            self.reliable_flags = vec![0; m.num_col.max(0) as usize];
            self.reliable_cols.clear();
        }
        self.model = m;
    }

    #[inline(always)]
    fn ps(&self) -> &mut Pseudocost {
        // SAFETY: set by the entry points; no callback touches the search's
        // pseudocosts while this reference is used (conflict analysis gets
        // them only through the C++, between uses)
        unsafe { &mut *self.ps }
    }

    #[inline(always)]
    fn queue(&self) -> &NodeQueue {
        // SAFETY: the global node queue, set by the entry points
        unsafe { &*self.queue }
    }

    fn shared(&self, p: *mut c_void) -> Shared {
        Shared { p, fns: self.fns }
    }

    // ----- small accessors -----

    fn feastol(&self) -> f64 {
        cb!(self, mip_value, 0)
    }
    fn epsilon(&self) -> f64 {
        cb!(self, mip_value, 1)
    }
    fn upper_limit(&self) -> f64 {
        cb!(self, mip_value, 2)
    }
    fn optimality_limit(&self) -> f64 {
        cb!(self, mip_value, 3)
    }
    pub fn cutoff_bound(&self) -> f64 {
        min2(self.upper_limit(), self.stats.upper_limit)
    }
    fn infeasible(&self) -> bool {
        cb!(self, infeasible)
    }
    fn propagate(&self) {
        cb!(self, propagate)
    }
    fn change_bound(&self, d: DomChg) {
        cb!(self, change_bound, d)
    }
    fn backtrack_dom(&self) -> DomChg {
        cb!(self, backtrack)
    }
    fn stack_len(&self) -> i32 {
        let mut n = 0;
        cb!(self, stack, &mut n);
        n
    }
    fn stack_entry(&self, k: i32) -> DomChg {
        let mut n = 0;
        let p = cb!(self, stack, &mut n);
        assert!(k >= 0 && k < n);
        // SAFETY: the stack's data with n entries
        unsafe { *p.add(k as usize) }
    }
    fn num_changed_cols(&self) -> i32 {
        cb!(self, num_changed_cols)
    }
    fn clear_changed_cols(&self, start: i32) {
        cb!(self, clear_changed_cols, start)
    }
    fn conflict_analysis(&self) {
        cb!(self, conflict_analysis)
    }
    fn col_bounds(&self, col: i32) -> (f64, f64) {
        let (mut l, mut u) = (std::ptr::null(), std::ptr::null());
        cb!(self, bounds, &mut l, &mut u);
        assert!(col >= 0 && col < self.model.num_col);
        // SAFETY: the domain's numCol bounds
        unsafe { (*l.add(col as usize), *u.add(col as usize)) }
    }
    fn lp_flush(&self) {
        cb!(self, lp_flush_domain)
    }
    fn lp(&self) -> &LpRelax {
        // SAFETY: the C++ search's current LP relaxation, live during the
        // call; only read here between callbacks that may change it
        let p = cb!(self, lp_rust);
        unsafe { &*p }
    }
    /// 0 scaledOptimal(status), 1 unscaledPrimalFeasible, 2
    /// unscaledDualFeasible, 3 status == kInfeasible, 4 status == kOptimal,
    /// 5 the LP solver's model status is kObjectiveBound
    fn lp_query(&self, which: i32, status: i32) -> bool {
        match which {
            0 => lp_relaxation::scaled_optimal(status),
            1 => lp_relaxation::unscaled_primal_feasible(status),
            2 => lp_relaxation::unscaled_dual_feasible(status),
            3 => status == lp_relaxation::INFEASIBLE,
            4 => status == lp_relaxation::OPTIMAL,
            _ => self.lp().lph().model_status() == crate::lp_data::run::MS_OBJECTIVE_BOUND,
        }
    }
    fn lp_objective(&self) -> f64 {
        self.lp().sh.objective
    }
    /// The LP solution's column values (0) or duals (1)
    fn lp_solution(&self, which: i32) -> &[f64] {
        let s = self.lp().lph().solution();
        if which == 1 {
            &s.col_dual
        } else {
            &s.col_value
        }
    }

    /// The number of rows of the LP
    fn lp_rows(&self) -> i32 {
        self.lp().lph().model.num_row
    }
    fn frac_ints(&self) -> Vec<FracInt> {
        self.lp().frac.clone()
    }
    fn num_frac_ints(&self) -> i32 {
        self.lp().frac.len() as i32
    }
    fn col_cost(&self, col: i32) -> f64 {
        // SAFETY: the model's costs
        unsafe { *self.model.col_cost.add(col as usize) }
    }
    fn root_lp_sol(&self) -> &[f64] {
        // SAFETY: mipdata's root LP solution (fixed during the search)
        unsafe { crate::ffi::sl(self.model.root_lp_sol, self.model.num_root_lp_sol) }
    }
    fn integral_cols(&self) -> &[i32] {
        // SAFETY: mipdata's integral columns
        unsafe { crate::ffi::sl(self.model.integral_cols, self.model.num_integral_cols) }
    }
    fn orbits_empty(&self, o: &Shared) -> bool {
        // SAFETY: a live orbits box
        unsafe { ((*self.fns).orbits_query)(o.p, -1) }
    }
    fn orbits_stabilized(&self, o: &Shared, col: i32) -> bool {
        // SAFETY: a live orbits box
        unsafe { ((*self.fns).orbits_query)(o.p, col) }
    }
    fn orbital_fixing(&self, o: &Shared) {
        cb!(self, orbital_fixing, o.p)
    }
    fn propagate_orbitopes(&self) {
        cb!(self, propagate_orbitopes)
    }
    fn add_incumbent(&self, sol_which: i32, obj: f64, source: i32) {
        let sol = self.lp_solution(sol_which);
        cb!(self, add_incumbent, sol.as_ptr(), sol.len() as i32, obj, source);
    }

    pub fn current_depth(&self) -> i32 {
        self.nodestack.len() as i32 + self.stats.depthoffset
    }

    fn strong_branching_lp_iterations(&self) -> i64 {
        self.stats.sblpiterations + cb!(self, mip_stat, 2)
    }
    fn total_lp_iterations(&self) -> i64 {
        self.stats.lpiterations + cb!(self, mip_stat, 1)
    }
    fn heuristic_lp_iterations(&self) -> i64 {
        self.stats.heurlpiterations + cb!(self, mip_stat, 0)
    }
    fn check_limits(&self, offset: i64) -> bool {
        cb!(self, check_limits, offset)
    }

    fn reliable_flags(&self, col: i32) -> i32 {
        self.reliable_flags[col as usize] as i32
    }
    fn mark_reliable(&mut self, col: i32, flag: i32) {
        let f = &mut self.reliable_flags[col as usize];
        if *f == 0 {
            self.reliable_cols.push(col);
        }
        *f |= flag as u8;
    }

    /// checkSol
    fn check_sol(&self, sol: &[f64], integerfeasible: &mut bool) -> f64 {
        let mut objval = CDouble::from(0.0);
        *integerfeasible = true;
        let feastol = self.feastol();
        for i in 0..self.model.num_col {
            objval += sol[i as usize] * self.col_cost(i);
            // SAFETY: the model's integrality
            let integer = unsafe { *self.model.integrality.add(i as usize) == 1 };
            if !*integerfeasible || !integer {
                continue;
            }
            if fractionality(sol[i as usize]) > feastol {
                *integerfeasible = false;
            }
        }
        objval.to_f64()
    }

    /// orbitsValidInChildNode
    fn orbits_valid_in_child(&self, chg: &DomChg) -> bool {
        let curr = self.nodestack.last().unwrap();
        if !curr.stabilizer_orbits.is_some()
            || self.orbits_empty(&curr.stabilizer_orbits)
            || self.orbits_stabilized(&curr.stabilizer_orbits, chg.column)
        {
            return true;
        }
        if chg.boundtype == UPPER && cb!(self, is_binary, chg.column, false) {
            return true;
        }
        false
    }

    /// The child node of the current node's branching decision
    fn push_child(&mut self, lb: f64, pass_orbits: bool, domchg_pos: i32) {
        let curr = self.nodestack.last().unwrap();
        let basis = curr.node_basis.clone();
        let orbits = if pass_orbits { curr.stabilizer_orbits.clone() } else { Shared::null(self.fns) };
        let estimate = curr.estimate;
        let mut child = NodeData::new(lb, estimate, basis, orbits);
        child.domgchg_stack_pos = domchg_pos;
        self.nodestack.push(child);
    }

    pub fn create_new_node(&mut self) {
        let mut node = NodeData::new(-INF, -INF, Shared::null(self.fns), Shared::null(self.fns));
        node.domgchg_stack_pos = self.stack_len();
        self.nodestack.push(node);
    }

    pub fn cutoff_node(&mut self) {
        self.nodestack.last_mut().unwrap().opensubtrees = 0;
    }

    /// branchDownwards (up false) / branchUpwards
    pub fn branch_dir(&mut self, col: i32, newbound: f64, branchpoint: f64, up: bool) {
        let curr = self.nodestack.last_mut().unwrap();
        debug_assert_eq!(curr.opensubtrees, 2);
        curr.opensubtrees = 1;
        curr.branching_point = branchpoint;
        curr.branchingdecision = DomChg { boundval: newbound, column: col, boundtype: if up { LOWER } else { UPPER } };
        let d = curr.branchingdecision;
        let pos = self.stack_len();
        let pass = self.orbits_valid_in_child(&d);
        self.change_bound(d);
        let lb = self.nodestack.last().unwrap().lower_bound;
        self.push_child(lb, pass, pos);
    }

    pub fn set_heuristic(&mut self, inheuristic: bool) {
        self.stats.inheuristic = inheuristic;
        if inheuristic {
            self.stats.childselrule = CHILD_HYBRID_INFERENCE_COST;
        }
    }

    pub fn has_node(&self) -> bool {
        !self.nodestack.is_empty()
    }
    pub fn current_node_pruned(&self) -> bool {
        self.nodestack.last().unwrap().opensubtrees == 0
    }
    pub fn current_estimate(&self) -> f64 {
        self.nodestack.last().unwrap().estimate
    }
    pub fn current_lower_bound(&self) -> f64 {
        self.nodestack.last().unwrap().lower_bound
    }

    /// The current node to the queue (or pruned): the common part of
    /// currentNodeToQueue and openNodesToQueue
    fn node_to_queue(&mut self, queue: *mut c_void) {
        let oldchangedcols = self.num_changed_cols();
        let mut prune = self.nodestack.last().unwrap().lower_bound > self.cutoff_bound();
        if !prune {
            self.propagate();
            self.clear_changed_cols(oldchangedcols);
            prune = self.infeasible();
            if prune {
                self.conflict_analysis();
            }
        }
        if !prune {
            let back = self.nodestack.last().unwrap();
            let lb = max2(back.lower_bound, cb!(self, objective_lower_bound));
            let w = cb!(self, node_to_queue, queue, lb, back.estimate, self.current_depth());
            if self.stats.count_tree_weight {
                self.stats.treeweight += w;
            }
        } else if self.stats.count_tree_weight {
            self.stats.treeweight += ldexp1(1 - self.current_depth());
        }
        self.nodestack.last_mut().unwrap().opensubtrees = 0;
    }

    pub fn current_node_to_queue(&mut self, queue: *mut c_void) {
        self.node_to_queue(queue);
    }

    pub fn open_nodes_to_queue(&mut self, queue: *mut c_void) {
        if self.nodestack.is_empty() {
            return;
        }
        // the basis of the node highest up in the tree
        let mut basis = Shared::null(self.fns);
        for n in &mut self.nodestack {
            if n.node_basis.is_some() {
                basis = n.node_basis.take();
                break;
            }
        }
        if self.nodestack.last().unwrap().opensubtrees == 0 {
            self.backtrack(false);
        }
        while !self.nodestack.is_empty() {
            self.node_to_queue(queue);
            self.backtrack(false);
        }
        self.lp_flush();
        if basis.is_some() {
            // SAFETY: a live basis box
            let rows = unsafe { ((*self.fns).basis_rows)(basis.p) };
            if rows == self.lp_rows() {
                let p = basis.take().into_raw();
                cb!(self, lp_set_stored_basis, p);
            }
            cb!(self, lp_recover_basis);
        }
    }

    /// installNode
    pub fn install_node(&mut self, domchgs: &[DomChg], branchings: &[i32], lower_bound: f64, estimate: f64, depth: i32) {
        cb!(
            self,
            set_stack,
            domchgs.as_ptr(),
            domchgs.len() as i32,
            branchings.as_ptr(),
            branchings.len() as i32
        );
        let mut valid = true;
        let global = self.shared(cb!(self, global_orbits));
        if global.is_some() {
            let mut n = 0;
            let p = cb!(self, branching_positions, &mut n);
            // SAFETY: the domain's branching positions
            let positions = unsafe { crate::ffi::sl(p, n) }.to_vec();
            for i in positions {
                let d = self.stack_entry(i);
                if cb!(self, column_position, d.column) == -1 {
                    continue;
                }
                if !cb!(self, is_binary, d.column, true) || (d.boundtype == LOWER && d.boundval == 1.0) {
                    valid = false;
                    break;
                }
            }
        }
        let orbits = if valid { global } else { Shared::null(self.fns) };
        self.nodestack.push(NodeData::new(lower_bound, estimate, Shared::null(self.fns), orbits));
        self.subrootsol.clear();
        self.stats.depthoffset = depth - 1;
    }

    /// The pseudocost cutoff observation of the parent's branching, if it
    /// had an LP objective and branched on a fractional value
    fn parent_cutoff_observation(&mut self) {
        let n = self.nodestack.len();
        if n < 2 {
            return;
        }
        let p = &self.nodestack[n - 2];
        if p.lp_objective != -INF && p.branching_point != p.branchingdecision.boundval {
            let (col, up) = (p.branchingdecision.column, p.branchingdecision.boundtype == LOWER);
            self.ps().add_cutoff_observation(col, up);
        }
    }

    /// evaluateNode
    pub fn evaluate_node(&mut self) -> i32 {
        let n = self.nodestack.len();
        let has_parent = n > 1;
        let inheuristic = self.stats.inheuristic;
        if !inheuristic && self.nodestack[n - 1].lower_bound > self.optimality_limit() {
            return SUB_OPTIMAL;
        }
        self.propagate();

        if !inheuristic && !self.infeasible() {
            let curr_orbits = self.nodestack[n - 1].stabilizer_orbits.is_some();
            let parent_ok = !has_parent || {
                let po = &self.nodestack[n - 2].stabilizer_orbits;
                !po.is_some() || !self.orbits_empty(po)
            };
            if cb!(self, num_perms) > 0 && !curr_orbits && parent_ok {
                let o = self.shared(cb!(self, compute_stabilizer_orbits));
                self.nodestack[n - 1].stabilizer_orbits = o;
            }
            let o = &self.nodestack[n - 1].stabilizer_orbits;
            if o.is_some() {
                self.orbital_fixing(o);
            } else {
                self.propagate_orbitopes();
            }
        }
        if has_parent {
            let inferences = self.stack_len() as i64 - (self.nodestack[n - 1].domgchg_stack_pos as i64 + 1);
            let p = &self.nodestack[n - 2];
            let (col, up) = (p.branchingdecision.column, p.branchingdecision.boundtype == LOWER);
            self.ps().add_inference_observation(col, inferences as i32, up);
        }

        let mut result = OPEN;
        if self.infeasible() {
            result = DOMAIN_INFEASIBLE;
            self.clear_changed_cols(-1);
            self.parent_cutoff_observation();
            self.conflict_analysis();
        } else {
            self.lp_flush();
            cb!(self, lp_set_objective_limit, self.upper_limit());
            let oldnumiters = self.lp().sh.numlpiters;
            let status = cb!(self, lp_resolve);
            self.stats.lpiterations += self.lp().sh.numlpiters - oldnumiters;

            let olb = cb!(self, objective_lower_bound);
            let curr = &mut self.nodestack[n - 1];
            curr.lower_bound = max2(olb, curr.lower_bound);

            if self.infeasible() {
                result = DOMAIN_INFEASIBLE;
                self.clear_changed_cols(-1);
                self.parent_cutoff_observation();
                self.conflict_analysis();
            } else if self.lp_query(0, status) {
                cb!(self, lp_store_basis, false);
                cb!(self, lp_perform_aging);
                let basis = self.shared(cb!(self, lp_store_basis, true));
                let estimate = self.lp().compute_best_estimate(self.ps());
                let lpobj = self.lp_objective();
                {
                    let curr = &mut self.nodestack[n - 1];
                    curr.node_basis = basis;
                    curr.estimate = estimate;
                    curr.lp_objective = lpobj;
                }
                if has_parent {
                    let p = &self.nodestack[n - 2];
                    if p.lp_objective != -INF && p.branching_point != p.branchingdecision.boundval {
                        let delta = p.branchingdecision.boundval - p.branching_point;
                        let objdelta = max2(0.0, lpobj - p.lp_objective);
                        let col = p.branchingdecision.column;
                        self.ps().add_observation(col, delta, objdelta);
                    }
                }

                if self.lp_query(1, status) && self.num_frac_ints() == 0 {
                    let cutoffbnd = self.cutoff_bound();
                    let src = if inheuristic { self.model.sources.heuristic } else { self.model.sources.evaluate_node };
                    self.add_incumbent(2, self.lp_objective(), src);
                    if self.upper_limit() < cutoffbnd {
                        cb!(self, lp_set_objective_limit, self.upper_limit());
                    }
                    if self.lp_query(2, status) {
                        cb!(self, add_bound_exceeding_conflict);
                        result = BOUND_EXCEEDING;
                    }
                }

                if result == OPEN {
                    if self.lp_query(2, status) {
                        let curr = &mut self.nodestack[n - 1];
                        curr.lower_bound = max2(curr.lp_objective, curr.lower_bound);
                        if self.nodestack[n - 1].lower_bound > self.cutoff_bound() {
                            result = BOUND_EXCEEDING;
                            cb!(self, add_bound_exceeding_conflict);
                        } else if self.upper_limit() != INF {
                            if !inheuristic {
                                let gap = self.upper_limit() - self.lp_objective();
                                let t = gap + max2(10.0 * self.feastol(), self.epsilon() * gap);
                                cb!(self, lp_degenerate_duals, t);
                            }
                            cb!(self, propagate_redcost);
                            self.propagate();
                            if self.infeasible() {
                                result = DOMAIN_INFEASIBLE;
                                self.clear_changed_cols(-1);
                                self.parent_cutoff_observation();
                                self.conflict_analysis();
                            } else if self.num_changed_cols() != 0 {
                                return self.evaluate_node();
                            }
                        } else if !inheuristic {
                            cb!(self, lp_degenerate_duals, INF);
                            self.propagate();
                            if self.infeasible() {
                                result = DOMAIN_INFEASIBLE;
                                self.clear_changed_cols(-1);
                                self.parent_cutoff_observation();
                                self.conflict_analysis();
                            } else if self.num_changed_cols() != 0 {
                                return self.evaluate_node();
                            }
                        }
                    } else if self.lp_objective() > self.cutoff_bound() {
                        cb!(self, add_bound_exceeding_conflict);
                        self.propagate();
                        if self.infeasible() {
                            result = BOUND_EXCEEDING;
                        }
                    }
                }
            } else if self.lp_query(3, status) {
                result = if self.lp_query(5, status) { BOUND_EXCEEDING } else { LP_INFEASIBLE };
                cb!(self, add_infeasible_conflict);
                self.parent_cutoff_observation();
            }
        }

        if result != OPEN {
            self.stats.treeweight += ldexp1(1 - self.current_depth());
            self.nodestack[n - 1].opensubtrees = 0;
        } else if !inheuristic && self.nodestack[n - 1].lower_bound > self.optimality_limit() {
            result = SUB_OPTIMAL;
            cb!(self, add_bound_exceeding_conflict);
        }
        result
    }

    /// selectBranchingCandidate: the index of the candidate among the
    /// fractional integers, -1 if strong branching decided a branch
    pub fn select_branching_candidate(&mut self, max_sb_iters: i64, down_lb: &mut f64, up_lb: &mut f64) -> i32 {
        let fracints = self.frac_ints();
        let numfrac = fracints.len();
        debug_assert!(numfrac > 0);
        let curr_lb = self.current_lower_bound();
        let mut upscore = vec![INF; numfrac];
        let mut downscore = vec![INF; numfrac];
        let mut upbound = vec![curr_lb; numfrac];
        let mut downbound = vec![curr_lb; numfrac];
        let mut upreliable = vec![false; numfrac];
        let mut downreliable = vec![false; numfrac];
        let feastol = self.feastol();

        for k in 0..numfrac {
            let (col, fracval) = (fracints[k].col, fracints[k].val);
            let (lb, ub) = self.col_bounds(col);
            let lower_residual = (fracval - lb) - feastol;
            if lower_residual <= 0.0 {
                cb!(self, log_frac_error, false, fracval, lb + feastol, lb, feastol, lower_residual);
            }
            let upper_residual = (ub - fracval) - feastol;
            if upper_residual <= 0.0 {
                cb!(self, log_frac_error, true, fracval, ub - feastol, ub, feastol, upper_residual);
            }
            let ps = self.ps();
            if ps.is_reliable(col) {
                upscore[k] = ps.get_pseudocost_up(col, fracval);
                downscore[k] = ps.get_pseudocost_down(col, fracval);
                upreliable[k] = true;
                downreliable[k] = true;
            } else {
                let flags = self.reliable_flags(col);
                if flags & UP_RELIABLE != 0 {
                    upscore[k] = ps.get_pseudocost_up(col, fracval);
                    upreliable[k] = true;
                }
                if flags & DOWN_RELIABLE != 0 {
                    downscore[k] = ps.get_pseudocost_down(col, fracval);
                    downreliable[k] = true;
                }
            }
        }

        let mut min_score = feastol;
        let playground = cb!(self, playground_new);
        let result = loop {
            let must_stop = self.strong_branching_lp_iterations() >= max_sb_iters || self.check_limits(0);

            // selectBestScore
            let candidate = {
                let ps = self.ps();
                let queue = self.queue();
                let mut best: i32 = -1;
                let mut bestscore = -1.0;
                let mut bestnodes = -1.0;
                let mut bestnumnodes: i64 = 0;
                let oldminscore = min_score;
                for k in 0..numfrac {
                    if upscore[k] <= oldminscore {
                        upreliable[k] = true;
                    }
                    if downscore[k] <= oldminscore {
                        downreliable[k] = true;
                    }
                    let s = 1e-3
                        * min2(
                            if upreliable[k] { upscore[k] } else { 0.0 },
                            if downreliable[k] { downscore[k] } else { 0.0 },
                        );
                    min_score = max2(s, min_score);
                    let col = fracints[k].col;
                    let score = if upscore[k] <= oldminscore || downscore[k] <= oldminscore {
                        ps.get_score_costs(col, min2(upscore[k], oldminscore), min2(downscore[k], oldminscore))
                    } else if upscore[k] == INF || downscore[k] == INF {
                        if must_stop {
                            ps.get_score(col, fracints[k].val)
                        } else {
                            INF
                        }
                    } else {
                        ps.get_score_costs(col, upscore[k], downscore[k])
                    };
                    let upnodes = queue.num_nodes_up(col);
                    let downnodes = queue.num_nodes_down(col);
                    let mut nodes = 0.0;
                    let numnodes = upnodes + downnodes;
                    if upnodes != 0 || downnodes != 0 {
                        nodes = (downnodes as f64 / numnodes as f64) * (upnodes as f64 / numnodes as f64);
                    }
                    if score > bestscore
                        || (score > bestscore - feastol
                            && (nodes > bestnodes || (nodes == bestnodes && numnodes > bestnumnodes)))
                    {
                        bestscore = score;
                        best = k as i32;
                        bestnodes = nodes;
                        bestnumnodes = numnodes;
                    }
                }
                best
            };
            let cand = candidate as usize;

            if (upreliable[cand] && downreliable[cand]) || must_stop {
                *down_lb = downbound[cand];
                *up_lb = upbound[cand];
                break candidate;
            }

            cb!(self, lp_set_objective_limit, self.upper_limit());
            let col = fracints[cand].col;
            let fracval = fracints[cand].val;
            let upval = fracval.ceil();
            let downval = fracval.floor();

            let ps = self.ps();
            let down_first = !downreliable[cand]
                && (upreliable[cand] || {
                    let (d, di) = (downscore[cand], ps.inferencesdown[col as usize]);
                    let (u, ui) = (upscore[cand], ps.inferencesup[col as usize]);
                    // std::make_pair(d, di) >= std::make_pair(u, ui)
                    d > u || (!(u > d) && !(di < ui))
                });
            let mut sb = StrongBranch {
                upscore: &mut upscore,
                downscore: &mut downscore,
                upbound: &mut upbound,
                downbound: &mut downbound,
                upreliable: &mut upreliable,
                downreliable: &mut downreliable,
                fracints: &fracints,
            };
            if self.strong_branch(&mut sb, playground, cand, col, fracval, upval, downval, !down_first) {
                break -1;
            }
        };
        // SAFETY: the playground of this call
        unsafe { ((*self.fns).playground_free)(playground) };
        result
    }

    /// The bound changes of other fractional columns that the strong
    /// branching LP solution `sol` satisfies (analyzeSolution)
    fn analyze_solution(&mut self, sb: &mut StrongBranch, col: i32, objdelta: f64, sol: &[f64]) {
        let num_changed = self.num_changed_cols();
        let stack_size = self.stack_len();
        let feastol = self.feastol();
        for k in 0..sb.fracints.len() {
            let fcol = sb.fracints[k].col;
            if fcol == col {
                continue;
            }
            let otherfracval = sb.fracints[k].val;
            let otherdownval = otherfracval.floor();
            let otherupval = otherfracval.ceil();
            let s = sol[fcol as usize];
            let (down, bound) = if s <= otherdownval + feastol {
                (true, otherdownval)
            } else if s >= otherupval - feastol {
                (false, otherupval)
            } else {
                continue;
            };
            let (lb, ub) = self.col_bounds(fcol);
            let needs_change = if down { ub > otherdownval + feastol } else { lb < otherupval - feastol };
            if needs_change {
                self.change_bound(DomChg { boundval: bound, column: fcol, boundtype: if down { UPPER } else { LOWER } });
                if self.infeasible() {
                    self.conflict_analysis();
                    self.backtrack_dom();
                    self.clear_changed_cols(num_changed);
                    continue;
                }
                self.propagate();
                if self.infeasible() {
                    self.conflict_analysis();
                    self.backtrack_dom();
                    self.clear_changed_cols(num_changed);
                    continue;
                }
                let new_size = self.stack_len();
                let mut valid = true;
                for j in stack_size + 1..new_size {
                    let d = self.stack_entry(j);
                    if d.boundtype == LOWER {
                        if d.boundval > sol[d.column as usize] + feastol {
                            valid = false;
                            break;
                        }
                    } else if d.boundval < sol[d.column as usize] - feastol {
                        valid = false;
                        break;
                    }
                }
                self.backtrack_dom();
                self.clear_changed_cols(num_changed);
                if !valid {
                    continue;
                }
            }
            if objdelta <= feastol {
                let delta = if down { otherdownval - otherfracval } else { otherupval - otherfracval };
                self.ps().add_observation(fcol, delta, objdelta);
                self.mark_reliable(fcol, if down { DOWN_RELIABLE } else { UP_RELIABLE });
            }
            if down {
                sb.downscore[k] = min2(sb.downscore[k], objdelta);
            } else {
                sb.upscore[k] = min2(sb.upscore[k], objdelta);
            }
        }
    }

    /// The other child after strong branching closed one (`pruned` the
    /// parent's remaining subtree)
    fn sb_branch_other(&mut self, col: i32, upbranch: bool, downval: f64, upval: f64, fracval: f64) {
        if upbranch {
            self.branch_dir(col, downval, fracval, false);
        } else {
            self.branch_dir(col, upval, fracval, true);
        }
    }

    /// strongBranch: true if the candidate's branching was decided
    #[allow(clippy::too_many_arguments)]
    fn strong_branch(
        &mut self,
        sb: &mut StrongBranch,
        playground: *mut c_void,
        cand: usize,
        col: i32,
        fracval: f64,
        upval: f64,
        downval: f64,
        upbranch: bool,
    ) -> bool {
        let mut inferences = -(self.stack_len() as i64) - 1;
        let boundval = if upbranch { upval } else { downval };
        let domchg = DomChg { boundval, column: col, boundtype: if upbranch { LOWER } else { UPPER } };
        let n = self.nodestack.len();
        let orbital_fixing = self.nodestack[n - 1].stabilizer_orbits.is_some() && self.orbits_valid_in_child(&domchg);
        self.change_bound(domchg);
        self.propagate();
        if !self.infeasible() {
            if orbital_fixing {
                self.orbital_fixing(&self.nodestack[n - 1].stabilizer_orbits);
            } else {
                self.propagate_orbitopes();
            }
        }
        inferences += self.stack_len() as i64;
        if self.infeasible() {
            self.conflict_analysis();
            self.ps().add_cutoff_observation(col, upbranch);
            self.backtrack_dom();
            self.clear_changed_cols(-1);
            self.sb_branch_other(col, upbranch, downval, upval, fracval);
            let m = self.nodestack.len();
            self.nodestack[m - 2].opensubtrees = 0;
            self.nodestack[m - 2].skip_depth_count = 1;
            self.stats.depthoffset -= 1;
            return true;
        }
        self.ps().add_inference_observation(col, inferences as i32, upbranch);

        let numiters = self.lp().sh.numlpiters;
        let status = cb!(self, playground_solve, playground);
        let numiters = self.lp().sh.numlpiters - numiters;
        self.stats.lpiterations += numiters;
        self.stats.sblpiterations += numiters;

        if self.lp_query(0, status) {
            cb!(self, lp_perform_aging);
            let delta = if upbranch { upval - fracval } else { downval - fracval };
            let mut integerfeasible = false;
            let sol = self.lp_solution(0).to_vec();
            let solobj = self.check_sol(&sol, &mut integerfeasible);
            let mut objdelta = max2(solobj - self.lp_objective(), 0.0);
            if objdelta <= self.epsilon() {
                objdelta = 0.0;
            }
            if upbranch {
                sb.upscore[cand] = objdelta;
                sb.upreliable[cand] = true;
                self.mark_reliable(col, UP_RELIABLE);
            } else {
                sb.downscore[cand] = objdelta;
                sb.downreliable[cand] = true;
                self.mark_reliable(col, DOWN_RELIABLE);
            }
            self.ps().add_observation(col, delta, objdelta);
            self.analyze_solution(sb, col, objdelta, &sol);

            if self.lp_query(1, status) && integerfeasible {
                let cutoffbnd = self.cutoff_bound();
                let src = if self.stats.inheuristic { self.model.sources.heuristic } else { self.model.sources.branching };
                self.add_incumbent(2, solobj, src);
                if self.upper_limit() < cutoffbnd {
                    cb!(self, lp_set_objective_limit, self.upper_limit());
                }
            }

            if self.lp_query(2, status) {
                if upbranch {
                    sb.upbound[cand] = solobj;
                } else {
                    sb.downbound[cand] = solobj;
                }
                if solobj > self.optimality_limit() {
                    cb!(self, add_bound_exceeding_conflict);
                    let pruned = solobj > self.cutoff_bound();
                    self.backtrack_dom();
                    self.lp_flush();
                    self.sb_branch_other(col, upbranch, downval, upval, fracval);
                    let m = self.nodestack.len();
                    let parent = &mut self.nodestack[m - 2];
                    parent.opensubtrees = if pruned { 0 } else { 1 };
                    parent.other_child_lb = solobj;
                    parent.skip_depth_count = 1;
                    self.stats.depthoffset -= 1;
                    return true;
                }
            } else if solobj > self.cutoff_bound() {
                cb!(self, add_bound_exceeding_conflict);
                self.propagate();
                if self.infeasible() {
                    self.backtrack_dom();
                    self.lp_flush();
                    self.sb_branch_other(col, upbranch, downval, upval, fracval);
                    let m = self.nodestack.len();
                    self.nodestack[m - 2].opensubtrees = 0;
                    self.nodestack[m - 2].skip_depth_count = 1;
                    self.stats.depthoffset -= 1;
                    return true;
                }
            }
        } else if self.lp_query(3, status) {
            cb!(self, add_infeasible_conflict);
            self.ps().add_cutoff_observation(col, upbranch);
            self.backtrack_dom();
            self.lp_flush();
            self.sb_branch_other(col, upbranch, downval, upval, fracval);
            let m = self.nodestack.len();
            self.nodestack[m - 2].opensubtrees = 0;
            self.nodestack[m - 2].skip_depth_count = 1;
            self.stats.depthoffset -= 1;
            return true;
        } else {
            // an LP error: score zero, so that the column is not chosen if
            // possible
            sb.downscore[cand] = 0.0;
            sb.upscore[cand] = 0.0;
            sb.downreliable[cand] = true;
            sb.upreliable[cand] = true;
            self.mark_reliable(col, UP_RELIABLE);
            self.mark_reliable(col, DOWN_RELIABLE);
        }
        self.backtrack_dom();
        self.lp_flush();
        false
    }

    /// The child selection of branch for the candidate (col, branching
    /// point), setting the current node's decision and other child's lower
    /// bound: the child's lower bound
    fn select_child(&mut self, col: i32, down_lb: f64, up_lb: f64) -> f64 {
        let n = self.nodestack.len();
        let bp = self.nodestack[n - 1].branching_point;
        let feastol = self.feastol();
        let eps = self.epsilon();
        let up = match self.stats.childselrule {
            CHILD_UP => true,
            CHILD_DOWN => false,
            CHILD_ROOT_SOL => {
                let ps = self.ps();
                let mut down_prio = ps.inferencesdown[col as usize] + eps;
                let mut up_prio = ps.inferencesup[col as usize] + eps;
                let down_val = bp.floor();
                let up_val = bp.ceil();
                let rootsol = if !self.subrootsol.is_empty() {
                    Some(self.subrootsol[col as usize])
                } else {
                    if self.nodestack[n - 1].lp_objective != -INF {
                        self.subrootsol = self.lp_solution(0).to_vec();
                    }
                    let r = self.root_lp_sol();
                    if r.is_empty() {
                        None
                    } else {
                        Some(r[col as usize])
                    }
                };
                if let Some(mut rootsol) = rootsol {
                    if rootsol < down_val {
                        rootsol = down_val;
                    } else if rootsol > up_val {
                        rootsol = up_val;
                    }
                    up_prio *= 1.0 + (bp - rootsol);
                    down_prio *= 1.0 + (rootsol - bp);
                }
                up_prio + eps >= down_prio
            }
            CHILD_OBJ => self.col_cost(col) >= 0.0,
            CHILD_RANDOM => self.random.bit(),
            CHILD_BEST_COST => {
                let ps = self.ps();
                !(ps.get_pseudocost_up_offset(col, bp, feastol) > ps.get_pseudocost_down_offset(col, bp, feastol))
            }
            CHILD_WORST_COST => {
                let ps = self.ps();
                ps.get_pseudocost_up(col, bp) >= ps.get_pseudocost_down(col, bp)
            }
            CHILD_DISJUNCTION => {
                let q = self.queue();
                let (nup, ndown) = (q.num_nodes_up(col), q.num_nodes_down(col));
                if nup > ndown {
                    true
                } else if ndown > nup {
                    false
                } else {
                    self.col_cost(col) >= 0.0
                }
            }
            _ => {
                let ps = self.ps();
                let up_score =
                    (1.0 + ps.inferencesup[col as usize]) / ps.get_pseudocost_up_offset(col, bp, feastol);
                let down_score =
                    (1.0 + ps.inferencesdown[col as usize]) / ps.get_pseudocost_down_offset(col, bp, feastol);
                up_score >= down_score
            }
        };
        let curr = &mut self.nodestack[n - 1];
        if up {
            curr.branchingdecision.boundtype = LOWER;
            curr.branchingdecision.boundval = bp.ceil();
            curr.other_child_lb = down_lb;
            up_lb
        } else {
            curr.branchingdecision.boundtype = UPPER;
            curr.branchingdecision.boundval = bp.floor();
            curr.other_child_lb = up_lb;
            down_lb
        }
    }

    /// branch
    pub fn branch(&mut self) -> i32 {
        debug_assert_eq!(self.nodestack.last().unwrap().opensubtrees, 2);
        self.nodestack.last_mut().unwrap().branchingdecision.column = -1;
        self.stats.inbranching = true;

        let minrel = self.ps().minreliable;
        let mut child_lb = self.current_lower_bound();
        let mut result = OPEN;
        while self.nodestack.last().unwrap().opensubtrees == 2
            && self.lp_query(0, self.lp().sh.status)
            && self.num_frac_ints() != 0
        {
            let mut sbmaxiters: i64 = 0;
            if minrel > 0 {
                let sbiters = self.strong_branching_lp_iterations();
                let sb_base: i64 = if cb!(self, mip_stat, 3) != 0 { 10000 } else { 100000 };
                sbmaxiters = sb_base
                    + ((self.total_lp_iterations() - self.heuristic_lp_iterations() - self.strong_branching_lp_iterations())
                        >> 1);
                if sbiters > sbmaxiters {
                    self.ps().minreliable = 0;
                } else if sbiters > (sbmaxiters >> 1) {
                    let reductionratio =
                        (sbiters - (sbmaxiters >> 1)) as f64 / (sbmaxiters - (sbmaxiters >> 1)) as f64;
                    let minrelreduced = (-reductionratio).mul_add_c((minrel - 1) as f64, minrel as f64) as i32;
                    self.ps().minreliable = minrel.min(minrelreduced);
                }
            }
            let degeneracy = cb!(self, lp_degeneracy);
            self.ps().degeneracy_factor = degeneracy;
            if degeneracy >= 10.0 {
                self.ps().minreliable = 0;
            }
            let mut down_lb = self.current_lower_bound();
            let mut up_lb = self.current_lower_bound();
            let cand = self.select_branching_candidate(sbmaxiters, &mut down_lb, &mut up_lb);
            child_lb = self.nodestack.last().unwrap().lower_bound;
            if cand != -1 {
                let fracints = self.frac_ints();
                let b = fracints[cand as usize];
                let curr = self.nodestack.last_mut().unwrap();
                curr.branchingdecision.column = b.col;
                curr.branching_point = b.val;
                child_lb = self.select_child(b.col, down_lb, up_lb);
                result = BRANCHED;
                break;
            }
            result = self.evaluate_node();
            if result == SUB_OPTIMAL {
                break;
            }
        }
        self.stats.inbranching = false;
        self.ps().minreliable = minrel;
        self.ps().degeneracy_factor = 1.0;

        let opensubtrees = self.nodestack.last().unwrap().opensubtrees;
        if opensubtrees != 2 || result == SUB_OPTIMAL {
            return result;
        }

        if self.nodestack.last().unwrap().branchingdecision.column == -1 {
            self.fallback_branching();
        }

        if self.nodestack.last().unwrap().branchingdecision.column == -1 {
            return self.fallback_lp();
        }

        // the child of the branching decision
        let domchg_pos = self.stack_len();
        let d = self.nodestack.last().unwrap().branchingdecision;
        let pass = self.orbits_valid_in_child(&d);
        self.change_bound(d);
        let curr = self.nodestack.last_mut().unwrap();
        curr.opensubtrees = 1;
        let lb = max2(child_lb, curr.lower_bound);
        self.push_child(lb, pass, domchg_pos);
        BRANCHED
    }

    /// Branching on any integer column when the solution branching failed
    fn fallback_branching(&mut self) {
        let mut bestscore = -1.0;
        self.ps().degeneracy_factor = 1e6;
        let feastol = self.feastol();
        let cols = self.integral_cols().to_vec();
        for i in cols {
            let (lb, ub) = self.col_bounds(i);
            if ub - lb < 0.5 {
                continue;
            }
            let fracval = if lb != -INF && ub != INF {
                (0.5 * (lb + ub + 0.5)).floor() + 0.5
            } else if lb != -INF {
                lb + 0.5
            } else if ub != INF {
                ub - 0.5
            } else {
                0.5
            };
            let score = self.ps().get_score(i, fracval);
            if score > bestscore {
                bestscore = score;
                let cost = if self.lp_query(2, self.lp().sh.status) { self.lp_solution(1)[i as usize] } else { self.col_cost(i) };
                let ps = self.ps();
                let (iu, id) = (ps.inferencesup[i as usize], ps.inferencesdown[i as usize]);
                let up = if cost.abs() > feastol && self.cutoff_bound() < INF {
                    cost > 0.0
                } else if iu > id + feastol {
                    true
                } else if iu < id - feastol {
                    false
                } else {
                    cb!(self, col_pos, i, false) <= cb!(self, col_pos, i, true)
                };
                let curr = self.nodestack.last_mut().unwrap();
                let v = if up { fracval.ceil() } else { fracval.floor() };
                curr.branching_point = v;
                curr.branchingdecision = DomChg { boundval: v, column: i, boundtype: if up { LOWER } else { UPPER } };
            }
        }
        self.ps().degeneracy_factor = 1.0;
    }

    /// All integer columns fixed: prune, or evaluate the node with a fresh
    /// LP of the model rows (presolve, then primal simplex, then IPM)
    fn fallback_lp(&mut self) -> i32 {
        if self.lp_query(4, self.lp().sh.status) {
            self.nodestack.last_mut().unwrap().opensubtrees = 0;
            return LP_INFEASIBLE;
        }
        cb!(self, lp_fallback, 0);
        let mut result = self.evaluate_node();
        if result == OPEN {
            cb!(self, lp_fallback, 1);
            result = self.evaluate_node();
            cb!(self, lp_fallback, 2);
            if result == OPEN {
                cb!(self, lp_fallback, 3);
                result = self.evaluate_node();
                if result == OPEN {
                    cb!(self, lp_fallback, 4);
                    self.nodestack.last_mut().unwrap().opensubtrees = 0;
                    result = LP_INFEASIBLE;
                }
            }
        }
        cb!(self, lp_fallback, 5);
        result
    }

    /// Pops the closed nodes (opensubtrees == 0) of the stack, repropagating
    /// the remaining node: the common start of backtrack and
    /// backtrackPlunge. False if the stack became empty (back at the root)
    fn pop_closed(&mut self, recover_basis: bool) -> bool {
        while self.nodestack.last().unwrap().opensubtrees == 0 {
            self.stats.count_tree_weight = true;
            self.stats.depthoffset += self.nodestack.last().unwrap().skip_depth_count as i32;
            if self.nodestack.len() == 1 {
                let mut back = self.nodestack.pop().unwrap();
                if recover_basis && back.node_basis.is_some() {
                    let p = back.node_basis.take().into_raw();
                    cb!(self, lp_set_stored_basis, p);
                }
                drop(back);
                cb!(self, backtrack_to_global);
                self.lp_flush();
                if recover_basis {
                    cb!(self, lp_recover_basis);
                }
                return false;
            }
            self.nodestack.pop();
            self.backtrack_dom();
            if self.nodestack.last().unwrap().opensubtrees != 0 {
                self.stats.count_tree_weight = self.nodestack.last().unwrap().skip_depth_count == 0;
                let old_num_domchgs = self.stack_len();
                let old_changed = self.num_changed_cols();
                self.propagate();
                if !self.infeasible() && old_num_domchgs != self.stack_len() {
                    let o = &self.nodestack.last().unwrap().stabilizer_orbits;
                    if o.is_some() {
                        self.orbital_fixing(o);
                    } else {
                        self.propagate_orbitopes();
                    }
                }
                if self.infeasible() {
                    self.clear_changed_cols(old_changed);
                    if self.stats.count_tree_weight {
                        self.stats.treeweight += ldexp1(-self.current_depth());
                    }
                    self.nodestack.last_mut().unwrap().opensubtrees = 0;
                }
            }
        }
        true
    }

    /// Switches the current node's branching decision to the other child,
    /// returning whether the branch point was the fallback (boundval)
    fn flip_decision(&mut self) -> bool {
        let curr = self.nodestack.last_mut().unwrap();
        debug_assert_eq!(curr.opensubtrees, 1);
        curr.opensubtrees = 0;
        let fallback = curr.branchingdecision.boundval == curr.branching_point;
        if curr.branchingdecision.boundtype == LOWER {
            curr.branchingdecision.boundtype = UPPER;
            curr.branchingdecision.boundval = (curr.branchingdecision.boundval - 0.5).floor();
        } else {
            curr.branchingdecision.boundtype = LOWER;
            curr.branchingdecision.boundval = (curr.branchingdecision.boundval + 0.5).ceil();
        }
        fallback
    }

    /// The other child of the current node: its bound change, propagation
    /// and symmetry fixing; true if it is pruned (and undone)
    fn other_child_pruned(&mut self, pass: bool, num_changed: i32) -> bool {
        let curr = self.nodestack.last().unwrap();
        let nodelb = max2(curr.lower_bound, curr.other_child_lb);
        let mut prune = nodelb > self.cutoff_bound() || self.infeasible();
        if !prune {
            self.propagate();
            prune = self.infeasible();
            if prune {
                self.conflict_analysis();
            }
        }
        if !prune {
            self.propagate_orbitopes();
            prune = self.infeasible();
        }
        if !prune && pass && self.nodestack.last().unwrap().stabilizer_orbits.is_some() {
            self.orbital_fixing(&self.nodestack.last().unwrap().stabilizer_orbits);
            prune = self.infeasible();
        }
        if prune {
            self.backtrack_dom();
            self.clear_changed_cols(num_changed);
            if self.stats.count_tree_weight {
                self.stats.treeweight += ldexp1(-self.current_depth());
            }
        }
        prune
    }

    fn recover_node_basis(&mut self) {
        let b = &self.nodestack.last().unwrap().node_basis;
        if b.is_some() {
            let p = b.clone().take().into_raw();
            cb!(self, lp_set_stored_basis, p);
            cb!(self, lp_recover_basis);
        }
    }

    /// backtrack
    pub fn backtrack(&mut self, recover_basis: bool) -> bool {
        if self.nodestack.is_empty() {
            return false;
        }
        loop {
            if !self.pop_closed(recover_basis) {
                return false;
            }
            let fallback = self.flip_decision();
            let domchg_pos = self.stack_len();
            let curr = self.nodestack.last_mut().unwrap();
            if fallback {
                curr.branching_point = curr.branchingdecision.boundval;
            }
            let d = curr.branchingdecision;
            let num_changed = self.num_changed_cols();
            let pass = self.orbits_valid_in_child(&d);
            self.change_bound(d);
            if self.other_child_pruned(pass, num_changed) {
                continue;
            }
            let curr = self.nodestack.last().unwrap();
            let nodelb = max2(curr.lower_bound, curr.other_child_lb);
            self.push_child(nodelb, pass, -1);
            self.lp_flush();
            self.nodestack.last_mut().unwrap().domgchg_stack_pos = domchg_pos;
            break;
        }
        if recover_basis {
            self.recover_node_basis();
        }
        true
    }

    /// backtrackPlunge
    pub fn backtrack_plunge(&mut self, queue: *mut c_void) -> bool {
        if self.nodestack.is_empty() {
            return false;
        }
        loop {
            if !self.pop_closed(true) {
                return false;
            }
            let fallback = self.flip_decision();
            let ps: *const Pseudocost = self.ps;
            let curr = self.nodestack.last_mut().unwrap();
            let bp = if fallback { 0.5 } else { curr.branching_point };
            let col = curr.branchingdecision.column;
            // the decision is flipped already: the score of the new side
            // SAFETY: as Search::ps
            let node_score = unsafe { &*ps }.get_score_dir(col, bp, curr.branchingdecision.boundtype == LOWER);
            if fallback {
                curr.branching_point = curr.branchingdecision.boundval;
            }
            let d = curr.branchingdecision;
            let domchg_pos = self.stack_len();
            let num_changed = self.num_changed_cols();
            let pass = self.orbits_valid_in_child(&d);
            self.change_bound(d);
            if self.other_child_pruned(pass, num_changed) {
                continue;
            }
            let curr = self.nodestack.last().unwrap();
            let mut nodelb = max2(curr.lower_bound, curr.other_child_lb);
            nodelb = max2(nodelb, cb!(self, objective_lower_bound));
            let mut to_queue = nodelb > self.optimality_limit();
            if !to_queue {
                let feastol = self.feastol();
                let ps = self.ps();
                let m = self.nodestack.len();
                for i in (0..m - 1).rev() {
                    let a = &self.nodestack[i];
                    if a.opensubtrees == 0 {
                        continue;
                    }
                    let fb = a.branchingdecision.boundval == a.branching_point;
                    let bp = if fb { 0.5 } else { a.branching_point };
                    let col = a.branchingdecision.column;
                    let (active, inactive) = if a.branchingdecision.boundtype == LOWER {
                        (ps.get_score_dir(col, bp, true), ps.get_score_dir(col, bp, false))
                    } else {
                        (ps.get_score_dir(col, bp, false), ps.get_score_dir(col, bp, true))
                    };
                    to_queue = inactive - active > node_score + feastol;
                    break;
                }
            }
            if to_queue {
                let est = self.nodestack.last().unwrap().estimate;
                let w = cb!(self, node_to_queue, queue, nodelb, est, self.current_depth() + 1);
                if self.stats.count_tree_weight {
                    self.stats.treeweight += w;
                }
                self.backtrack_dom();
                self.clear_changed_cols(num_changed);
                continue;
            }
            self.push_child(nodelb, pass, -1);
            self.lp_flush();
            self.nodestack.last_mut().unwrap().domgchg_stack_pos = domchg_pos;
            break;
        }
        self.recover_node_basis();
        true
    }

    /// backtrackUntilDepth
    pub fn backtrack_until_depth(&mut self, target_depth: i32) -> bool {
        if self.nodestack.is_empty() {
            return false;
        }
        if self.current_depth() >= target_depth {
            self.nodestack.last_mut().unwrap().opensubtrees = 0;
        }
        while self.nodestack.last().unwrap().opensubtrees == 0 {
            self.stats.depthoffset += self.nodestack.last().unwrap().skip_depth_count as i32;
            self.nodestack.pop();
            self.backtrack_dom();
            if self.nodestack.is_empty() {
                self.lp_flush();
                return false;
            }
            if self.current_depth() >= target_depth {
                self.nodestack.last_mut().unwrap().opensubtrees = 0;
            }
        }
        let fallback = self.flip_decision();
        let curr = self.nodestack.last_mut().unwrap();
        if fallback {
            curr.branching_point = curr.branchingdecision.boundval;
        }
        let d = curr.branchingdecision;
        let domchg_pos = self.stack_len();
        let pass = self.orbits_valid_in_child(&d);
        self.change_bound(d);
        let lb = self.nodestack.last().unwrap().lower_bound;
        self.push_child(lb, pass, -1);
        self.lp_flush();
        self.nodestack.last_mut().unwrap().domgchg_stack_pos = domchg_pos;
        let b = &self.nodestack.last().unwrap().node_basis;
        // SAFETY: a live basis box
        if b.is_some() && unsafe { ((*self.fns).basis_rows)(b.p) } == self.lp_rows() {
            let p = b.clone().take().into_raw();
            cb!(self, lp_set_stored_basis, p);
        }
        cb!(self, lp_recover_basis);
        true
    }

    /// dive
    pub fn dive(&mut self, node_lim: i64) -> i32 {
        for c in self.reliable_cols.drain(..) {
            self.reliable_flags[c as usize] = 0;
        }
        loop {
            self.stats.nnodes += 1;
            let result = self.evaluate_node();
            if self.check_limits(self.stats.nnodes) {
                return result;
            }
            if result != OPEN {
                return result;
            }
            let result = self.branch();
            if result != BRANCHED {
                return result;
            }
            if self.stats.nnodes >= node_lim {
                return result;
            }
        }
    }

    /// solveDepthFirst
    pub fn solve_depth_first(&mut self, mut maxbacktracks: i64) {
        loop {
            if maxbacktracks == 0 {
                break;
            }
            let result = self.dive(i64::MAX);
            if result == OPEN {
                break;
            }
            maxbacktracks -= 1;
            if !self.backtrack(true) {
                break;
            }
        }
    }
}

/// The score arrays of a strong branching round
struct StrongBranch<'a> {
    upscore: &'a mut Vec<f64>,
    downscore: &'a mut Vec<f64>,
    upbound: &'a mut Vec<f64>,
    downbound: &'a mut Vec<f64>,
    upreliable: &'a mut Vec<bool>,
    downreliable: &'a mut Vec<bool>,
    fracints: &'a [FracInt],
}

pub(crate) mod ffi {
    use super::*;
    use crate::ffi::sl;

    /// # Safety
    /// see Search::new
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_search_new(
        fns: *const CSearchFns,
        ctx: *mut c_void,
        model: *const CModel,
        submip: bool,
    ) -> *mut Search {
        let model = std::ptr::read(model);
        Box::into_raw(Box::new(Search::new(fns, ctx, model, submip)))
    }

    /// # Safety
    /// `s` from highs_rs_search_new, or null
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_search_free(s: *mut Search) {
        if !s.is_null() {
            drop(Box::from_raw(s));
        }
    }

    /// The statistics C++ uses in place
    ///
    /// # Safety
    /// live search
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_search_stats(s: *mut Search) -> *mut Stats {
        std::ptr::addr_of_mut!((*s).stats)
    }

    /// The operations without results: 0 createNewNode, 1 cutoffNode, 2
    /// branchDownwards(i, x, y), 3 branchUpwards(i, x, y), 4
    /// setHeuristic(i), 5 currentNodeToQueue(q), 6 openNodesToQueue(q), 7
    /// solveDepthFirst(n)
    ///
    /// # Safety
    /// live search, `ps` its pseudocosts, `nq` the global node queue, `q`
    /// a HighsNodeQueue
    #[no_mangle]
    #[allow(clippy::too_many_arguments)]
    pub unsafe extern "C" fn highs_rs_search_op(
        s: *mut Search,
        ps: *mut Pseudocost,
        nq: *const NodeQueue,
        which: i32,
        i: i32,
        x: f64,
        y: f64,
        n: i64,
        q: *mut c_void,
    ) {
        let s = &mut *s;
        s.enter(ps, nq);
        match which {
            0 => s.create_new_node(),
            1 => s.cutoff_node(),
            2 => s.branch_dir(i, x, y, false),
            3 => s.branch_dir(i, x, y, true),
            4 => s.set_heuristic(i != 0),
            5 => s.current_node_to_queue(q),
            6 => s.open_nodes_to_queue(q),
            _ => s.solve_depth_first(n),
        }
    }

    /// The operations with an integer result: 0 evaluateNode, 1 branch, 2
    /// backtrack(i), 3 backtrackPlunge(q), 4 backtrackUntilDepth(i), 5
    /// dive(n), 6 hasNode, 7 currentNodePruned, 8 getCurrentDepth
    ///
    /// # Safety
    /// as highs_rs_search_op
    #[no_mangle]
    #[allow(clippy::too_many_arguments)]
    pub unsafe extern "C" fn highs_rs_search_run(
        s: *mut Search,
        ps: *mut Pseudocost,
        nq: *const NodeQueue,
        which: i32,
        i: i32,
        n: i64,
        q: *mut c_void,
    ) -> i32 {
        let s = &mut *s;
        s.enter(ps, nq);
        match which {
            0 => s.evaluate_node(),
            1 => s.branch(),
            2 => s.backtrack(i != 0) as i32,
            3 => s.backtrack_plunge(q) as i32,
            4 => s.backtrack_until_depth(i) as i32,
            5 => s.dive(n),
            6 => s.has_node() as i32,
            7 => s.current_node_pruned() as i32,
            _ => s.current_depth(),
        }
    }

    /// 0 getCurrentEstimate, 1 getCurrentLowerBound, 2 getCutoffBound
    ///
    /// # Safety
    /// live search
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_search_value(s: *const Search, which: i32) -> f64 {
        let s = &*s;
        match which {
            0 => s.current_estimate(),
            1 => s.current_lower_bound(),
            _ => s.cutoff_bound(),
        }
    }

    /// selectBranchingCandidate
    ///
    /// # Safety
    /// as highs_rs_search_op
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_search_select(
        s: *mut Search,
        ps: *mut Pseudocost,
        nq: *const NodeQueue,
        max_sb_iters: i64,
        down_lb: *mut f64,
        up_lb: *mut f64,
    ) -> i32 {
        let s = &mut *s;
        s.enter(ps, nq);
        s.select_branching_candidate(max_sb_iters, &mut *down_lb, &mut *up_lb)
    }

    /// installNode
    ///
    /// # Safety
    /// as highs_rs_search_op, arrays valid for their lengths
    #[no_mangle]
    #[allow(clippy::too_many_arguments)]
    pub unsafe extern "C" fn highs_rs_search_install(
        s: *mut Search,
        ps: *mut Pseudocost,
        nq: *const NodeQueue,
        domchgs: *const DomChg,
        ndomchgs: i32,
        branchings: *const i32,
        nbranchings: i32,
        lower_bound: f64,
        estimate: f64,
        depth: i32,
    ) {
        let s = &mut *s;
        s.enter(ps, nq);
        s.install_node(sl(domchgs, ndomchgs), sl(branchings, nbranchings), lower_bound, estimate, depth);
    }
}
