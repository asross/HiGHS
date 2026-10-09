//! HighsNodeQueue (highs/mip/HighsNodeQueue.cpp): the open nodes of the
//! branch-and-bound tree, ordered by lower bound, by a hybrid of lower
//! bound and estimate, and (nodes above the optimality limit) by lower
//! bound in the suboptimal set; and per column the nodes with a changed
//! lower / upper bound, by bound value. Rust-owned; the C++ class is a
//! handle (HighsNodeQueue.cpp under HIGHS_RUST).
//!
//! The C++ red-black trees order the nodes by keys that end with the node
//! index, so they are unique: BTreeSets of the keys give the same orders
//! (minimum, maximum, predecessor). Bound values compare as doubles (so
//! -0 and 0 are equal, as in the C++). The free slots are a min-heap of
//! unique indices, so their order is the C++ priority_queue's.

use super::domain::{DomChg, LOWER};
use crate::util::cdouble::CDouble;
use crate::util::fma::ClangFma;
use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeSet, BinaryHeap};

const INF: f64 = f64::INFINITY;
const IINF: i64 = i32::MAX as i64;

/// A double ordered as by `<` (never NaN here)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ordf(pub f64);

impl Eq for Ordf {}

impl PartialOrd for Ordf {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ordf {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

/// HighsNodeQueue::OpenNode
#[derive(Default)]
pub struct OpenNode {
    pub domchgstack: Vec<DomChg>,
    pub branchings: Vec<i32>,
    pub lower_bound: f64,
    pub estimate: f64,
    pub depth: i32,
}

type LowerKey = (Ordf, i32, Ordf, i64);
type HybridKey = (Ordf, i32, i64);
type SubKey = (Ordf, i64);
type NodeSet = BTreeSet<(Ordf, i64)>;

#[derive(Default)]
pub struct NodeQueue {
    nodes: Vec<OpenNode>,
    freeslots: BinaryHeap<Reverse<i64>>,
    col_lower_nodes: Vec<NodeSet>,
    col_upper_nodes: Vec<NodeSet>,
    lower: BTreeSet<LowerKey>,
    hybrid: BTreeSet<HybridKey>,
    suboptimal: BTreeSet<SubKey>,
    num_suboptimal: i64,
    optimality_limit: f64,
    num_col: i32,
    /// the last popped node, read by the C++
    pub popped: OpenNode,
}

/// std::ldexp(1.0, e): 2^e, exact (subnormal or 0 below 2^-1022)
#[inline]
pub fn ldexp1(e: i32) -> f64 {
    if e > 1023 {
        INF
    } else if e >= -1022 {
        f64::from_bits(((e + 1023) as u64) << 52)
    } else if e >= -1074 {
        f64::from_bits(1u64 << (e + 1074))
    } else {
        0.0
    }
}

impl NodeQueue {
    pub fn new() -> Self {
        NodeQueue { optimality_limit: INF, ..Default::default() }
    }

    fn lower_key(&self, n: i64) -> LowerKey {
        let x = &self.nodes[n as usize];
        (Ordf(x.lower_bound), x.domchgstack.len() as i32, Ordf(x.estimate), n)
    }

    fn hybrid_key(&self, n: i64) -> HybridKey {
        let x = &self.nodes[n as usize];
        // clang fuses 0.5 * lb + 0.5 * estimate
        (Ordf(0.5f64.mul_add_c(x.lower_bound, 0.5 * x.estimate)), -(x.domchgstack.len() as i32), n)
    }

    fn sub_key(&self, n: i64) -> SubKey {
        (Ordf(self.nodes[n as usize].lower_bound), n)
    }

    fn link_domchgs(&mut self, n: i64) {
        let node = &self.nodes[n as usize];
        for d in &node.domchgstack {
            let set = if d.boundtype == LOWER { &mut self.col_lower_nodes } else { &mut self.col_upper_nodes };
            set[d.column as usize].insert((Ordf(d.boundval), n));
        }
    }

    fn unlink_domchgs(&mut self, n: i64) {
        let node = &self.nodes[n as usize];
        for d in &node.domchgstack {
            let set = if d.boundtype == LOWER { &mut self.col_lower_nodes } else { &mut self.col_upper_nodes };
            set[d.column as usize].remove(&(Ordf(d.boundval), n));
        }
    }

    fn link_suboptimal(&mut self, n: i64) {
        self.suboptimal.insert(self.sub_key(n));
        self.num_suboptimal += 1;
    }

    fn unlink_estim_lower(&mut self, n: i64) {
        self.hybrid.remove(&self.hybrid_key(n));
        self.lower.remove(&self.lower_key(n));
    }

    /// link: the tree weight of the node if it goes to the suboptimal set
    fn link(&mut self, n: i64) -> f64 {
        if self.nodes[n as usize].lower_bound > self.optimality_limit {
            self.nodes[n as usize].estimate = INF;
            self.link_suboptimal(n);
            self.link_domchgs(n);
            return ldexp1(1 - self.nodes[n as usize].depth);
        }
        self.hybrid.insert(self.hybrid_key(n));
        self.lower.insert(self.lower_key(n));
        self.link_domchgs(n);
        0.0
    }

    fn unlink(&mut self, n: i64) {
        if self.nodes[n as usize].estimate == INF {
            self.suboptimal.remove(&self.sub_key(n));
            self.num_suboptimal -= 1;
        } else {
            self.unlink_estim_lower(n);
        }
        self.unlink_domchgs(n);
        self.freeslots.push(Reverse(n));
    }

    pub fn set_optimality_limit(&mut self, limit: f64) {
        self.optimality_limit = limit;
    }

    pub fn set_num_col(&mut self, num_col: i32) {
        if self.num_col == num_col {
            return;
        }
        self.num_col = num_col;
        let n = num_col.max(0) as usize;
        self.col_lower_nodes = vec![NodeSet::new(); n];
        self.col_upper_nodes = vec![NodeSet::new(); n];
    }

    /// checkGlobalBounds: prunes the nodes whose bound change on col
    /// contradicts the global bounds
    pub fn check_global_bounds(&mut self, col: i32, lb: f64, ub: f64, feastol: f64, treeweight: &mut CDouble) {
        let c = col as usize;
        let mut delnodes = BTreeSet::new();
        for &(_, n) in self.col_lower_nodes[c].range((Ordf(ub + feastol), -1)..) {
            delnodes.insert(n);
        }
        for &(_, n) in self.col_upper_nodes[c].range(..=(Ordf(lb - feastol), IINF)) {
            delnodes.insert(n);
        }
        for n in delnodes {
            if self.nodes[n as usize].estimate != INF {
                *treeweight += ldexp1(1 - self.nodes[n as usize].depth);
            }
            self.unlink(n);
        }
    }

    /// The bound all open nodes have on col (lower: the smallest of their
    /// lower bounds, else the largest upper bound), if they all change it
    pub fn common_bound(&self, col: i32, lower: bool) -> Option<f64> {
        let num = self.num_nodes() as usize;
        let c = col as usize;
        if lower {
            let set = &self.col_lower_nodes[c];
            if set.len() == num {
                return set.first().map(|x| x.0 .0);
            }
        } else {
            let set = &self.col_upper_nodes[c];
            if set.len() == num {
                return set.last().map(|x| x.0 .0);
            }
        }
        None
    }

    pub fn prune_node(&mut self, n: i64) -> f64 {
        let w = if self.nodes[n as usize].estimate != INF { ldexp1(1 - self.nodes[n as usize].depth) } else { 0.0 };
        self.unlink(n);
        w
    }

    /// performBounding
    pub fn perform_bounding(&mut self, upper_limit: f64) -> f64 {
        if self.lower.is_empty() {
            return 0.0;
        }
        let mut treeweight = CDouble::from(0.0);
        while let Some(&(lb, _, _, n)) = self.lower.last() {
            if lb.0 < upper_limit {
                break;
            }
            treeweight += self.prune_node(n);
        }
        if self.optimality_limit < upper_limit {
            while let Some(&(lb, _, _, n)) = self.lower.last() {
                if lb.0 < self.optimality_limit {
                    break;
                }
                self.unlink_estim_lower(n);
                treeweight += ldexp1(1 - self.nodes[n as usize].depth);
                self.nodes[n as usize].estimate = INF;
                self.link_suboptimal(n);
            }
        }
        if self.num_suboptimal != 0 {
            while let Some(&(lb, n)) = self.suboptimal.last() {
                if lb.0 < upper_limit {
                    break;
                }
                self.unlink(n);
            }
        }
        treeweight.to_f64()
    }

    /// emplaceNode
    pub fn emplace_node(&mut self, domchgs: &[DomChg], branchings: &[i32], lower_bound: f64, estimate: f64, depth: i32) -> f64 {
        let pos = match self.freeslots.pop() {
            None => {
                self.nodes.push(OpenNode::default());
                self.nodes.len() as i64 - 1
            }
            Some(Reverse(pos)) => pos,
        };
        let node = &mut self.nodes[pos as usize];
        node.domchgstack.clear();
        node.domchgstack.extend_from_slice(domchgs);
        node.branchings.clear();
        node.branchings.extend_from_slice(branchings);
        node.lower_bound = lower_bound;
        node.estimate = estimate;
        node.depth = depth;
        self.link(pos)
    }

    /// popBestNode (best estimate) / popBestBoundNode into `popped`
    pub fn pop(&mut self, best_bound: bool) {
        let n = if best_bound { self.lower.first().map(|x| x.3) } else { self.hybrid.first().map(|x| x.2) };
        let n = n.expect("pop from an empty node queue");
        self.unlink(n);
        std::mem::swap(&mut self.popped, &mut self.nodes[n as usize]);
    }

    pub fn best_lower_bound(&self) -> f64 {
        let lb = self.lower.first().map_or(INF, |x| x.0 .0);
        match self.suboptimal.first() {
            None => lb,
            Some(s) => {
                // std::min(suboptimal lb, lb)
                if lb < s.0 .0 {
                    lb
                } else {
                    s.0 .0
                }
            }
        }
    }

    pub fn best_bound_domchg_stack_size(&self) -> i32 {
        let size = self.lower.first().map_or(i32::MAX, |x| x.1);
        match self.suboptimal.first() {
            None => size,
            Some(s) => (self.nodes[s.1 as usize].domchgstack.len() as i32).min(size),
        }
    }

    /// clear: an empty queue for the same columns (the optimality limit
    /// reset)
    pub fn clear(&mut self) {
        let num_col = self.num_col;
        *self = NodeQueue::new();
        self.set_num_col(num_col);
    }

    pub fn num_nodes(&self) -> i64 {
        (self.nodes.len() - self.freeslots.len()) as i64
    }

    pub fn num_active_nodes(&self) -> i64 {
        self.num_nodes() - self.num_suboptimal
    }

    #[inline]
    pub fn num_nodes_up(&self, col: i32) -> i64 {
        self.col_lower_nodes[col as usize].len() as i64
    }

    #[inline]
    pub fn num_nodes_down(&self, col: i32) -> i64 {
        self.col_upper_nodes[col as usize].len() as i64
    }

    /// The node queue part of the clique table's new edge (v1, v2): prunes
    /// the open nodes in both ranges [(val, IInf), (val, IInf)] of the
    /// columns' up (val 1) / down node sets, as the C++ (whose ranges hold
    /// no node)
    pub fn prune_edge(&mut self, col1: i32, val1: i32, col2: i32, val2: i32, treeweight: &mut CDouble) {
        let range = |q: &NodeQueue, col: i32, val: i32| -> Vec<i64> {
            let set = if val == 1 { &q.col_lower_nodes[col as usize] } else { &q.col_upper_nodes[col as usize] };
            let k = (Ordf(val as f64), IINF);
            set.range(k..=k).map(|x| x.1).collect()
        };
        let (a, b) = (range(self, col1, val1), range(self, col2, val2));
        if a.is_empty() || b.is_empty() || !(a[0] <= b[b.len() - 1] || b[0] <= a[a.len() - 1]) {
            return;
        }
        let (mut i, mut j) = (0, 0);
        while i < a.len() && j < b.len() {
            if a[i] < b[j] {
                i += 1;
            } else if b[j] < a[i] {
                j += 1;
            } else {
                let n = b[j];
                i += 1;
                j += 1;
                *treeweight += self.prune_node(n);
            }
        }
    }

    /// numNodesUp(col, val): the nodes with a lower bound above val
    pub fn num_nodes_up_val(&self, col: i32, val: f64) -> i64 {
        self.col_lower_nodes[col as usize].range((Ordf(val), IINF + 1)..).count() as i64
    }

    /// numNodesDown(col, val): the nodes with an upper bound below val
    pub fn num_nodes_down_val(&self, col: i32, val: f64) -> i64 {
        self.col_upper_nodes[col as usize].range(..(Ordf(val), -1)).count() as i64
    }
}

/// The last popped node's data for the C++
#[repr(C)]
pub struct CPopped {
    pub domchgstack: *const DomChg,
    pub num_domchgs: i32,
    pub branchings: *const i32,
    pub num_branchings: i32,
    pub lower_bound: f64,
    pub estimate: f64,
    pub depth: i32,
}

pub(crate) mod ffi {
    use super::*;
    use crate::ffi::sl;

    #[no_mangle]
    pub extern "C" fn highs_rs_nodequeue_new() -> *mut NodeQueue {
        Box::into_raw(Box::new(NodeQueue::new()))
    }

    /// # Safety
    /// `q` from highs_rs_nodequeue_new, or null
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_nodequeue_free(q: *mut NodeQueue) {
        if !q.is_null() {
            drop(Box::from_raw(q));
        }
    }

    /// 0 setOptimalityLimit(x), 1 setNumCol(i), 2 clear
    ///
    /// # Safety
    /// live queue
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_nodequeue_set(q: *mut NodeQueue, which: i32, i: i32, x: f64) {
        let q = &mut *q;
        match which {
            0 => q.set_optimality_limit(x),
            1 => q.set_num_col(i),
            _ => q.clear(),
        }
    }

    /// 0 numNodes, 1 numActiveNodes, 2 numNodesUp(col), 3
    /// numNodesDown(col), 4 numNodesUp(col, x), 5 numNodesDown(col, x), 6
    /// getBestBoundDomchgStackSize
    ///
    /// # Safety
    /// live queue
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_nodequeue_count(q: *const NodeQueue, which: i32, col: i32, x: f64) -> i64 {
        let q = &*q;
        match which {
            0 => q.num_nodes(),
            1 => q.num_active_nodes(),
            2 => q.num_nodes_up(col),
            3 => q.num_nodes_down(col),
            4 => q.num_nodes_up_val(col, x),
            5 => q.num_nodes_down_val(col, x),
            _ => q.best_bound_domchg_stack_size() as i64,
        }
    }

    /// 0 getBestLowerBound, 1 performBounding(x), 2 pruneNode(i)
    ///
    /// # Safety
    /// live queue
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_nodequeue_bound(q: *mut NodeQueue, which: i32, i: i64, x: f64) -> f64 {
        let q = &mut *q;
        match which {
            0 => q.best_lower_bound(),
            1 => q.perform_bounding(x),
            _ => q.prune_node(i),
        }
    }

    /// emplaceNode
    ///
    /// # Safety
    /// live queue, arrays valid for their lengths
    #[no_mangle]
    #[allow(clippy::too_many_arguments)]
    pub unsafe extern "C" fn highs_rs_nodequeue_emplace(
        q: *mut NodeQueue,
        domchgs: *const DomChg,
        ndomchgs: i32,
        branchings: *const i32,
        nbranchings: i32,
        lower_bound: f64,
        estimate: f64,
        depth: i32,
    ) -> f64 {
        (*q).emplace_node(sl(domchgs, ndomchgs), sl(branchings, nbranchings), lower_bound, estimate, depth)
    }

    /// popBestBoundNode (best_bound) / popBestNode: the node's data, valid
    /// until the next call
    ///
    /// # Safety
    /// live non-empty queue
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_nodequeue_pop(q: *mut NodeQueue, best_bound: bool, out: *mut CPopped) {
        let q = &mut *q;
        q.pop(best_bound);
        let p = &q.popped;
        *out = CPopped {
            domchgstack: p.domchgstack.as_ptr(),
            num_domchgs: p.domchgstack.len() as i32,
            branchings: p.branchings.as_ptr(),
            num_branchings: p.branchings.len() as i32,
            lower_bound: p.lower_bound,
            estimate: p.estimate,
            depth: p.depth,
        };
    }

    /// The checkGlobalBounds loop of pruneInfeasibleNodes over all columns
    /// (adding to `treeweight`)
    ///
    /// # Safety
    /// live queue, bounds valid for the queue's columns
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_nodequeue_check_global_bounds(
        q: *mut NodeQueue,
        col_lower: *const f64,
        col_upper: *const f64,
        feastol: f64,
        treeweight: *mut CDouble,
    ) {
        let q = &mut *q;
        let n = q.num_col;
        let (lower, upper) = (sl(col_lower, n), sl(col_upper, n));
        for i in 0..n {
            q.check_global_bounds(i, lower[i as usize], upper[i as usize], feastol, &mut *treeweight);
        }
    }

    /// checkGlobalBounds of one column
    ///
    /// # Safety
    /// live queue
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_nodequeue_check_global_bound(
        q: *mut NodeQueue,
        col: i32,
        lb: f64,
        ub: f64,
        feastol: f64,
        treeweight: *mut CDouble,
    ) {
        (*q).check_global_bounds(col, lb, ub, feastol, &mut *treeweight);
    }

    /// prune_edge
    ///
    /// # Safety
    /// live queue
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_nodequeue_prune_edge(
        q: *mut NodeQueue,
        col1: i32,
        val1: i32,
        col2: i32,
        val2: i32,
        treeweight: *mut CDouble,
    ) {
        (*q).prune_edge(col1, val1, col2, val2, &mut *treeweight);
    }

    /// The bound all open nodes change on col (lower / upper), in `*val`
    ///
    /// # Safety
    /// live queue
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_nodequeue_common_bound(q: *const NodeQueue, col: i32, lower: bool, val: *mut f64) -> bool {
        match (*q).common_bound(col, lower) {
            Some(v) => {
                *val = v;
                true
            }
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::domain::UPPER;
    use super::*;

    fn chg(col: i32, val: f64, lower: bool) -> DomChg {
        DomChg { boundval: val, column: col, boundtype: if lower { LOWER } else { UPPER } }
    }

    #[test]
    fn orders_and_bounding() {
        let mut q = NodeQueue::new();
        q.set_num_col(2);
        q.emplace_node(&[chg(0, 1.0, true)], &[0], 5.0, 7.0, 2);
        q.emplace_node(&[chg(0, 0.0, false), chg(1, 1.0, true)], &[0], 3.0, 9.0, 3);
        q.emplace_node(&[chg(1, 0.0, false)], &[0], 4.0, 4.0, 2);
        assert_eq!(q.num_nodes(), 3);
        assert_eq!(q.best_lower_bound(), 3.0);
        assert_eq!(q.num_nodes_up(1), 1);
        assert_eq!(q.num_nodes_down_val(0, 0.5), 1);
        // the best estimate: 0.5 lb + 0.5 estimate = 4 (the third node)
        q.pop(false);
        assert_eq!(q.popped.lower_bound, 4.0);
        // bounding prunes the node with lower bound 5
        let w = q.perform_bounding(4.5);
        assert_eq!(w, ldexp1(1 - 2));
        assert_eq!(q.num_nodes(), 1);
        // the smallest free slot is reused
        q.emplace_node(&[], &[], 1.0, 1.0, 1);
        assert_eq!(q.freeslots.len(), 1);
        q.pop(true);
        assert_eq!(q.popped.lower_bound, 1.0);
        assert_eq!(ldexp1(-1074), f64::from_bits(1));
    }
}
