//! HighsDomain::ObjectivePropagation (highs/mip/HighsDomain.cpp): the
//! objective's minimal value over the domain, kept up to date with the
//! bound changes, and the bounds it implies under the cutoff
//! (mipdata.upper_limit). The binaries of each clique partition of the
//! objective contribute through a red-black tree of their contributions
//! (highs/util/HighsRbTree.h, ported exactly so that the trees have the
//! same shape and order). The state is Rust's ([`ObjPropState`]), built
//! here and owned by the C++ ObjectivePropagation shell.

use super::domain::{bound_range, max2, Bounds, CBounds, CDomain, CSlice, Ctx, Dom, DomChg, StdVec, LOWER};
use super::domain::{Reason, INF, REASON_OBJECTIVE, UPPER};
use crate::ffi::sl;
use crate::util::cdouble::CDouble;

/// ObjectivePropagation::ObjectiveContribution
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Contribution {
    pub contribution: f64,
    pub col: i32,
    pub partition: i32,
    /// RbTreeLinks<HighsInt>: children, parent + 1 with the color in bit 31
    pub child: [i32; 2],
    pub parent_and_color: u32,
}

/// The ObjectivePropagation of a domain, mirrored by highs_rs::ObjProp in
/// highs/mip/HighsDomainRustView.h
#[repr(C)]
pub struct CObjProp {
    pub active: bool,
    pub(crate) cost: CSlice<f64>,
    pub(crate) obj_nonzeros: CSlice<i32>,
    pub(crate) partition_starts: CSlice<i32>,
    pub(crate) col_to_partition: CSlice<i32>,
    pub(crate) num_binaries: i32,
    pub(crate) contributions: CSlice<Contribution>,
    /// (root, first) of each partition's tree
    pub(crate) partition_sets: CSlice<[i32; 2]>,
    pub(crate) objective_lower: *mut CDouble,
    pub(crate) num_inf_obj_lower: *mut i32,
    pub(crate) capacity_threshold: *mut f64,
    pub(crate) is_propagated: *mut bool,
    // getPropagationConstraint
    pub(crate) obj_vals: CSlice<f64>,
    pub(crate) clique_data: CSlice<CliqueData>,
    pub(crate) cons_buffer: CSlice<f64>,
}

/// ObjectivePropagation::PartitionCliqueData
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CliqueData {
    multiplier: f64,
    rhs: i32,
    changed: bool,
}

const COLOR: u32 = 1 << 31;
const NO_LINK: i32 = -1;
const LEFT: usize = 0;
const RIGHT: usize = 1;

/// highs::CacheMinRbTree<ObjectiveContributionTree> over one partition
struct Tree<'a> {
    nodes: &'a mut [Contribution],
    root: &'a mut i32,
    first: &'a mut i32,
}

impl Tree<'_> {
    /// getKey: (-contribution, col), compared as std::pair
    #[inline]
    fn less(&self, a: i32, b: i32) -> bool {
        let (x, y) = (&self.nodes[a as usize], &self.nodes[b as usize]);
        let (xk, yk) = (-x.contribution, -y.contribution);
        xk < yk || (!(yk < xk) && x.col < y.col)
    }

    #[inline]
    fn is_red(&self, n: i32) -> bool {
        n != NO_LINK && self.nodes[n as usize].parent_and_color & COLOR != 0
    }
    #[inline]
    fn is_black(&self, n: i32) -> bool {
        n == NO_LINK || self.nodes[n as usize].parent_and_color & COLOR == 0
    }
    #[inline]
    fn make_red(&mut self, n: i32) {
        self.nodes[n as usize].parent_and_color |= COLOR;
    }
    #[inline]
    fn make_black(&mut self, n: i32) {
        self.nodes[n as usize].parent_and_color &= !COLOR;
    }
    #[inline]
    fn color(&self, n: i32) -> u32 {
        self.nodes[n as usize].parent_and_color >> 31
    }
    #[inline]
    fn set_color(&mut self, n: i32, color: u32) {
        self.make_black(n);
        self.nodes[n as usize].parent_and_color |= (color != 0) as u32 * COLOR;
    }
    #[inline]
    fn parent(&self, n: i32) -> i32 {
        (self.nodes[n as usize].parent_and_color & !COLOR) as i32 - 1
    }
    #[inline]
    fn set_parent(&mut self, n: i32, p: i32) {
        let c = &mut self.nodes[n as usize].parent_and_color;
        *c = (*c & COLOR) | (p + 1) as u32;
    }
    #[inline]
    fn child(&self, n: i32, dir: usize) -> i32 {
        self.nodes[n as usize].child[dir]
    }
    #[inline]
    fn set_child(&mut self, n: i32, dir: usize, c: i32) {
        self.nodes[n as usize].child[dir] = c;
    }

    fn rotate(&mut self, x: i32, dir: usize) {
        let y = self.child(x, 1 - dir);
        let y_dir = self.child(y, dir);
        self.set_child(x, 1 - dir, y_dir);
        if y_dir != NO_LINK {
            self.set_parent(y_dir, x);
        }
        let p_x = self.parent(x);
        self.set_parent(y, p_x);
        if p_x == NO_LINK {
            *self.root = y;
        } else {
            let d = ((x != self.child(p_x, dir)) as usize) ^ dir;
            self.set_child(p_x, d, y);
        }
        self.set_child(y, dir, x);
        self.set_parent(x, y);
    }

    fn insert_fixup(&mut self, mut z: i32) {
        let mut p_z = self.parent(z);
        while self.is_red(p_z) {
            let mut z_grand_parent = self.parent(p_z);
            let dir = (self.child(z_grand_parent, LEFT) == p_z) as usize;
            let y = self.child(z_grand_parent, dir);
            if self.is_red(y) {
                self.make_black(p_z);
                self.make_black(y);
                self.make_red(z_grand_parent);
                z = z_grand_parent;
            } else {
                if z == self.child(p_z, dir) {
                    z = p_z;
                    self.rotate(z, 1 - dir);
                    p_z = self.parent(z);
                    z_grand_parent = self.parent(p_z);
                }
                self.make_black(p_z);
                self.make_red(z_grand_parent);
                self.rotate(z_grand_parent, dir);
            }
            p_z = self.parent(z);
        }
        let r = *self.root;
        self.make_black(r);
    }

    fn transplant(&mut self, u: i32, v: i32, nil_parent: &mut i32) {
        let p = self.parent(u);
        if p == NO_LINK {
            *self.root = v;
        } else {
            let d = (u != self.child(p, LEFT)) as usize;
            self.set_child(p, d, v);
        }
        if v == NO_LINK {
            *nil_parent = p;
        } else {
            self.set_parent(v, p);
        }
    }

    fn delete_fixup(&mut self, mut x: i32, nil_parent: i32) {
        while x != *self.root && self.is_black(x) {
            let p = if x == NO_LINK { nil_parent } else { self.parent(x) };
            let dir = (x == self.child(p, LEFT)) as usize;
            let mut w = self.child(p, dir);
            if self.is_red(w) {
                self.make_black(w);
                self.make_red(p);
                self.rotate(p, 1 - dir);
                w = self.child(p, dir);
            }
            if self.is_black(self.child(w, LEFT)) && self.is_black(self.child(w, RIGHT)) {
                self.make_red(w);
                x = p;
            } else {
                if self.is_black(self.child(w, dir)) {
                    let c = self.child(w, 1 - dir);
                    self.make_black(c);
                    self.make_red(w);
                    self.rotate(w, dir);
                    w = self.child(p, dir);
                }
                let pc = self.color(p);
                self.set_color(w, pc);
                self.make_black(p);
                let c = self.child(w, dir);
                self.make_black(c);
                self.rotate(p, 1 - dir);
                x = *self.root;
            }
        }
        if x != NO_LINK {
            self.make_black(x);
        }
    }

    /// The cached first (smallest) node
    #[inline]
    fn first(&self) -> i32 {
        *self.first
    }

    fn first_of(&self, mut x: i32) -> i32 {
        if x == NO_LINK {
            return NO_LINK;
        }
        loop {
            let l = self.child(x, LEFT);
            if l == NO_LINK {
                return x;
            }
            x = l;
        }
    }

    fn last(&self) -> i32 {
        let mut x = *self.root;
        if x == NO_LINK {
            return NO_LINK;
        }
        loop {
            let r = self.child(x, RIGHT);
            if r == NO_LINK {
                return x;
            }
            x = r;
        }
    }

    fn successor(&self, mut x: i32) -> i32 {
        let mut y = self.child(x, RIGHT);
        if y != NO_LINK {
            return self.first_of(y);
        }
        y = self.parent(x);
        while y != NO_LINK && x == self.child(y, RIGHT) {
            x = y;
            y = self.parent(x);
        }
        y
    }

    /// CacheMinRbTree::link(z, parent), then RbTree::link(z, parent)
    fn link_at(&mut self, z: i32, parent: i32) {
        if *self.first == parent && (parent == NO_LINK || self.less(z, parent)) {
            *self.first = z;
        }
        self.set_parent(z, parent);
        if parent == NO_LINK {
            *self.root = z;
        } else {
            let d = self.less(parent, z) as usize;
            self.set_child(parent, d, z);
        }
        self.set_child(z, LEFT, NO_LINK);
        self.set_child(z, RIGHT, NO_LINK);
        self.make_red(z);
        self.insert_fixup(z);
    }

    /// RbTree::link(z)
    fn link(&mut self, z: i32) {
        let mut y = NO_LINK;
        let mut x = *self.root;
        while x != NO_LINK {
            y = x;
            x = self.child(y, self.less(x, z) as usize);
        }
        self.link_at(z, y);
    }

    /// CacheMinRbTree::unlink, then RbTree::unlink
    fn unlink(&mut self, z: i32) {
        if z == *self.first {
            *self.first = self.successor(z);
        }
        let mut nil_parent = NO_LINK;
        let mut y = z;
        let mut y_was_black = self.is_black(y);
        let x;
        if self.child(z, LEFT) == NO_LINK {
            x = self.child(z, RIGHT);
            self.transplant(z, x, &mut nil_parent);
        } else if self.child(z, RIGHT) == NO_LINK {
            x = self.child(z, LEFT);
            self.transplant(z, x, &mut nil_parent);
        } else {
            y = self.first_of(self.child(z, RIGHT));
            y_was_black = self.is_black(y);
            x = self.child(y, RIGHT);
            if self.parent(y) == z {
                if x == NO_LINK {
                    nil_parent = y;
                } else {
                    self.set_parent(x, y);
                }
            } else {
                let yr = self.child(y, RIGHT);
                self.transplant(y, yr, &mut nil_parent);
                let z_right = self.child(z, RIGHT);
                self.set_child(y, RIGHT, z_right);
                self.set_parent(z_right, y);
            }
            self.transplant(z, y, &mut nil_parent);
            let z_left = self.child(z, LEFT);
            self.set_child(y, LEFT, z_left);
            self.set_parent(z_left, y);
            let zc = self.color(z);
            self.set_color(y, zc);
        }
        if y_was_black {
            self.delete_fixup(x, nil_parent);
        }
    }
}

/// The mutable state of the objective propagation, borrowed from a view
struct Obj<'a> {
    cost: &'a [f64],
    obj_nonzeros: &'a [i32],
    partition_starts: &'a [i32],
    col_to_partition: &'a [i32],
    num_binaries: i32,
    contributions: &'a mut [Contribution],
    partition_sets: &'a mut [[i32; 2]],
    objective_lower: &'a mut CDouble,
    num_inf: &'a mut i32,
    capacity_threshold: &'a mut f64,
    is_propagated: &'a mut bool,
}

/// ObjectivePropagation::recomputeCapacityThreshold on the domain's bounds
fn capacity_threshold(
    contributions: &mut [Contribution],
    partition_sets: &mut [[i32; 2]],
    cost: &[f64],
    obj_nonzeros: &[i32],
    partition_starts: &[i32],
    b: &Bounds,
) -> f64 {
    let feastol = b.feastol;
    let np = partition_starts.len() - 1;
    let mut cap = -feastol;
    for i in 0..np {
        let (worst, best) = {
            let [root, first] = &mut partition_sets[i];
            let t = Tree { nodes: contributions, root, first };
            (t.first(), t.last())
        };
        if worst == -1 {
            continue;
        }
        let col = contributions[worst as usize].col as usize;
        if b.col_lower[col] == b.col_upper[col] {
            continue;
        }
        let mut contribution = contributions[worst as usize].contribution;
        if best != worst {
            contribution -= contributions[best as usize].contribution;
        }
        cap = max2(cap, contribution * (1.0 - feastol));
    }
    for &c in &obj_nonzeros[partition_starts[np] as usize..] {
        let col = c as usize;
        cap = max2(cap, cost[col].abs() * bound_range(b.col_upper[col], b.col_lower[col], feastol, b.continuous(col)));
    }
    cap
}

/// The ObjectivePropagation's state, owned by the C++ shell (mirrored by
/// highs_rs::ObjPropState)
#[repr(C)]
pub struct ObjPropState {
    pub contributions: StdVec<Contribution>,
    /// (root, first) of each partition's tree
    pub partition_sets: StdVec<[i32; 2]>,
    pub cons_buffer: StdVec<f64>,
    pub clique_data: StdVec<CliqueData>,
    pub objective_lower: CDouble,
    pub num_inf_obj_lower: i32,
    pub capacity_threshold: f64,
    pub is_propagated: bool,
}

impl Clone for ObjPropState {
    fn clone(&self) -> Self {
        ObjPropState {
            contributions: self.contributions.to_owned_vec(),
            partition_sets: self.partition_sets.to_owned_vec(),
            cons_buffer: self.cons_buffer.to_owned_vec(),
            clique_data: self.clique_data.to_owned_vec(),
            ..*self
        }
    }
}

impl Drop for ObjPropState {
    fn drop(&mut self) {
        // SAFETY: Rust-owned vectors
        unsafe {
            drop(self.contributions.take_vec());
            drop(self.partition_sets.take_vec());
            drop(self.cons_buffer.take_vec());
            drop(self.clique_data.take_vec());
        }
    }
}

impl ObjPropState {
    /// ObjectivePropagation(domain): for each clique partition all columns
    /// contribute with their largest value, then the largest contribution
    /// of each partition is removed (the trees keep the columns not fixed
    /// to their value with the larger contribution); then the other
    /// objective nonzeros
    pub fn build(b: &Bounds, cost: &[f64], obj_nonzeros: &[i32], partition_starts: &[i32], packed: &[f64]) -> Self {
        let np = partition_starts.len() - 1;
        let mut contributions = vec![Contribution::default(); partition_starts[np] as usize];
        let mut partition_sets = vec![[-1, -1]; np];
        let (cons_buffer, mut clique_data) =
            if np != 0 { (packed.to_vec(), vec![CliqueData::default(); np]) } else { (Vec::new(), Vec::new()) };
        let mut objective_lower = CDouble::from(0.0);
        let mut num_inf = 0;
        for i in 0..np {
            clique_data[i].rhs = 1;
            for j in partition_starts[i] as usize..partition_starts[i + 1] as usize {
                let col = obj_nonzeros[j];
                let c = col as usize;
                contributions[j].col = col;
                contributions[j].partition = i as i32;
                let link = if cost[c] > 0.0 {
                    objective_lower += cost[c];
                    contributions[j].contribution = cost[c];
                    clique_data[i].rhs -= 1;
                    b.col_lower[c] == 0.0
                } else {
                    contributions[j].contribution = -cost[c];
                    b.col_upper[c] == 1.0
                };
                if link {
                    let [root, first] = &mut partition_sets[i];
                    Tree { nodes: &mut contributions, root, first }.link(j as i32);
                }
            }
            let worst = partition_sets[i][1];
            if worst != -1 {
                objective_lower -= contributions[worst as usize].contribution;
            }
        }
        for &col in &obj_nonzeros[partition_starts[np] as usize..] {
            let c = col as usize;
            if cost[c] > 0.0 {
                if b.col_lower[c] == -INF {
                    num_inf += 1;
                } else {
                    objective_lower += b.col_lower[c] * cost[c];
                }
            } else if b.col_upper[c] == INF {
                num_inf += 1;
            } else {
                objective_lower += b.col_upper[c] * cost[c];
            }
        }
        let capacity_threshold =
            capacity_threshold(&mut contributions, &mut partition_sets, cost, obj_nonzeros, partition_starts, b);
        ObjPropState {
            contributions: StdVec::from_vec(contributions),
            partition_sets: StdVec::from_vec(partition_sets),
            cons_buffer: StdVec::from_vec(cons_buffer),
            clique_data: StdVec::from_vec(clique_data),
            objective_lower,
            num_inf_obj_lower: num_inf,
            capacity_threshold,
            is_propagated: false,
        }
    }
}

/// ObjectivePropagation(domain): the state built on the domain's bounds
///
/// # Safety
/// The arrays valid for their lengths; `partition_starts` has
/// numPartitions + 1 entries
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn highs_rs_objprop_new(
    b: *const CBounds,
    cost: *const f64,
    ncol: i32,
    obj_nonzeros: *const i32,
    nnz: i32,
    partition_starts: *const i32,
    nstarts: i32,
    packed: *const f64,
    npacked: i32,
) -> *mut ObjPropState {
    let s = ObjPropState::build(
        &(*b).view(),
        sl(cost, ncol),
        sl(obj_nonzeros, nnz),
        sl(partition_starts, nstarts),
        sl(packed, npacked),
    );
    Box::into_raw(Box::new(s))
}

/// # Safety
/// `s` live
#[no_mangle]
pub unsafe extern "C" fn highs_rs_objprop_clone(s: *const ObjPropState) -> *mut ObjPropState {
    Box::into_raw(Box::new((*s).clone()))
}

/// # Safety
/// `s` from new/clone, or null
#[no_mangle]
pub unsafe extern "C" fn highs_rs_objprop_free(s: *mut ObjPropState) {
    if !s.is_null() {
        drop(Box::from_raw(s));
    }
}

/// getPropagationConstraint of the domain's (active) objective propagation
///
/// # Safety
/// `d` a domain's view; the outputs writable
#[no_mangle]
pub unsafe extern "C" fn highs_rs_domain_obj_propagation_constraint(
    d: *const CDomain,
    stacksize: i32,
    domchg_col: i32,
    vals: *mut *const f64,
    inds: *mut *const i32,
    len: *mut i32,
    rhs: *mut f64,
) {
    let mut dom = CDomain::view(d as *mut CDomain);
    let (i, v, r) = dom.obj_propagation_constraint(stacksize, domchg_col);
    *inds = i.as_ptr();
    *vals = v.as_ptr();
    *len = i.len() as i32;
    *rhs = r;
}

impl CObjProp {
    /// The view of an inactive objective propagation
    pub fn inactive() -> CObjProp {
        CObjProp {
            active: false,
            cost: CSlice::of(&[]),
            obj_nonzeros: CSlice::of(&[]),
            partition_starts: CSlice::of(&[]),
            col_to_partition: CSlice::of(&[]),
            num_binaries: 0,
            contributions: CSlice::of(&[]),
            partition_sets: CSlice::of(&[]),
            objective_lower: std::ptr::null_mut(),
            num_inf_obj_lower: std::ptr::null_mut(),
            capacity_threshold: std::ptr::null_mut(),
            is_propagated: std::ptr::null_mut(),
            obj_vals: CSlice::of(&[]),
            clique_data: CSlice::of(&[]),
            cons_buffer: CSlice::of(&[]),
        }
    }
    /// # Safety
    /// As for CDomain::view; one view at a time
    #[inline(always)]
    unsafe fn view<'a>(&self) -> Obj<'a> {
        Obj {
            cost: self.cost.get(),
            obj_nonzeros: self.obj_nonzeros.get(),
            partition_starts: self.partition_starts.get(),
            col_to_partition: self.col_to_partition.get(),
            num_binaries: self.num_binaries,
            contributions: self.contributions.get_mut(),
            partition_sets: self.partition_sets.get_mut(),
            objective_lower: &mut *self.objective_lower,
            num_inf: &mut *self.num_inf_obj_lower,
            capacity_threshold: &mut *self.capacity_threshold,
            is_propagated: &mut *self.is_propagated,
        }
    }
}

impl<'a> Obj<'a> {
    fn num_partitions(&self) -> usize {
        self.partition_starts.len() - 1
    }

    fn tree(&mut self, partition: usize) -> Tree<'_> {
        let [root, first] = &mut self.partition_sets[partition];
        Tree { nodes: self.contributions, root, first }
    }
}

impl<'a> Dom<'a> {
    #[inline(always)]
    fn obj(&self) -> Obj<'a> {
        // SAFETY: the objective's data is distinct from the rest of the
        // view, and Dom never holds two Obj at a time
        unsafe { self.objprop.view() }
    }

    /// ObjectivePropagation::updateActivityLbChange (upper = false) /
    /// updateActivityUbChange (upper = true)
    pub(crate) fn obj_update_activity(&mut self, col: usize, oldbound: f64, newbound: f64, upper: bool) {
        let mut o = self.obj();
        let cost = o.cost[col];
        let feastol = self.feastol;
        // the bound that gives the objective's lower bound
        if if upper { cost >= 0.0 } else { cost <= 0.0 } {
            let relaxed = if upper { newbound > oldbound } else { newbound < oldbound };
            if cost != 0.0 && relaxed {
                let t = if upper {
                    cost * bound_range(newbound, self.col_lower[col], feastol, self.is_continuous(col))
                } else {
                    -cost * bound_range(self.col_upper[col], newbound, feastol, self.is_continuous(col))
                };
                *o.capacity_threshold = max2(*o.capacity_threshold, t);
                *o.is_propagated = false;
            }
            return;
        }

        *o.is_propagated = false;

        let partition_pos = o.col_to_partition[col];
        if partition_pos == -1 {
            let inf = if upper { INF } else { -INF };
            if oldbound == inf {
                *o.num_inf -= 1;
            } else {
                *o.objective_lower -= oldbound * cost;
            }
            if newbound == inf {
                *o.num_inf += 1;
            } else {
                *o.objective_lower += newbound * cost;
            }

            let relaxed = if upper { newbound > oldbound } else { newbound < oldbound };
            if relaxed {
                let c = if upper { -cost } else { cost };
                let t = c * bound_range(self.col_upper[col], self.col_lower[col], feastol, self.is_continuous(col));
                *o.capacity_threshold = max2(*o.capacity_threshold, t);
            } else if *o.num_inf == 0 && o.objective_lower.to_f64() > self.upper_limit {
                self.set_infeasible(REASON_OBJECTIVE, 0);
                self.obj_update_activity(col, newbound, oldbound, upper);
            }
        } else {
            let pp = partition_pos as usize;
            let partition = o.contributions[pp].partition as usize;
            // relaxing the bound links the column into its partition's tree
            if newbound == if upper { 1.0 } else { 0.0 } {
                let (curr_first, first, last) = {
                    let mut t = o.tree(partition);
                    let curr_first = t.first();
                    t.link(partition_position(pp));
                    (curr_first, t.first(), t.last())
                };
                let old_contribution = if curr_first != -1 { o.contributions[curr_first as usize].contribution } else { 0.0 };
                let c = o.contributions[pp].contribution;
                if partition_position(pp) == first && c != old_contribution {
                    *o.objective_lower += old_contribution;
                    *o.objective_lower -= c;
                    let mut delta = c;
                    if last != partition_position(pp) {
                        delta -= o.contributions[last as usize].contribution;
                    }
                    *o.capacity_threshold = max2(delta * (1.0 - feastol), *o.capacity_threshold);
                } else {
                    *o.capacity_threshold = max2((old_contribution - c) * (1.0 - feastol), *o.capacity_threshold);
                }
            } else {
                let c = o.contributions[pp].contribution;
                let was_first = o.tree(partition).first() == partition_position(pp);
                if was_first {
                    *o.objective_lower += c;
                }
                let new_worst = {
                    let mut t = o.tree(partition);
                    t.unlink(partition_position(pp));
                    t.first()
                };
                if was_first && new_worst != -1 {
                    *o.objective_lower -= o.contributions[new_worst as usize].contribution;
                }
                if *o.num_inf == 0 && o.objective_lower.to_f64() > self.upper_limit {
                    self.set_infeasible(REASON_OBJECTIVE, 0);
                    self.obj_update_activity(col, newbound, oldbound, upper);
                }
            }
        }
    }

    /// Whether checkChangeBound surely rejects the bound (capacity + b * cost)
    /// / cost on col (an upper bound if upper), as shown by a double estimate
    /// with a margin thousands of times its rounding error: the exact
    /// (double-double) computation is skipped then. Not in the C++
    #[inline]
    fn obj_bound_hopeless(&self, col: usize, upper: bool, capacity: f64, b: f64, cost: f64) -> bool {
        let bc = b * cost;
        let est = (capacity + bc) / cost;
        let margin = 1e-12 * ((capacity.abs() + bc.abs()) / cost.abs() + est.abs());
        if !(est.is_finite() && margin.is_finite()) {
            return false;
        }
        let (lb, ub) = (self.col_lower[col], self.col_upper[col]);
        let integral = !self.is_continuous(col);
        if upper {
            if integral {
                // floor(v + feastol) >= ceil(ub) >= ub
                est - margin + self.feastol >= ub.ceil()
            } else {
                // the bound is v itself (not snapped to lb) and v >= ub
                est - margin >= ub && (est - margin) - lb > self.epsilon
            }
        } else if integral {
            // ceil(v - feastol) <= floor(lb) <= lb
            est + margin - self.feastol <= lb.floor()
        } else {
            est + margin <= lb && ub - (est + margin) > self.epsilon
        }
    }

    /// ObjectivePropagation::getPropagationConstraint: the objective cutoff
    /// as a constraint (indices, values, rhs) at the time the domain change
    /// stack had the given size, with the partitions' largest contributions
    /// left out (skipping column domchg_col)
    pub(crate) fn obj_propagation_constraint(&mut self, stacksize: i32, domchg_col: i32) -> (&'a [i32], &'a [f64], f64) {
        let o = &self.objprop;
        // SAFETY: the objective's data, valid for the view's lifetime; the
        // buffer and clique data are written only here
        let (inds, cost, starts) = unsafe { (o.obj_nonzeros.get(), o.cost.get(), o.partition_starts.get()) };
        let (vals, clique, buf) = unsafe { (o.obj_vals.get(), o.clique_data.get_mut(), o.cons_buffer.get_mut()) };
        let np = starts.len() - 1;
        if np == 0 {
            return (inds, vals, self.upper_limit);
        }
        let b = self.bounds();
        let mut tmp_rhs = CDouble::from(self.upper_limit);
        for i in 0..np {
            let (start, end) = (starts[i] as usize, starts[i + 1] as usize);
            let mut largest = 0.0;
            for &c in &inds[start..end] {
                // skip the column we might want to explain a bound change for
                // and take the second largest column instead
                if c == domchg_col {
                    continue;
                }
                let cu = c as usize;
                if cost[cu] > 0.0 {
                    if b.col_bound_at(cu, stacksize, false).0 < 1.0 {
                        largest = max2(largest, cost[cu]);
                    }
                } else if b.col_bound_at(cu, stacksize, true).0 > 0.0 {
                    largest = max2(largest, -cost[cu]);
                }
            }
            tmp_rhs += largest * clique[i].rhs as f64;
            if clique[i].multiplier != largest {
                clique[i].multiplier = largest;
                for j in start..end {
                    buf[j] = vals[j] - largest.copysign(vals[j]);
                }
            }
        }
        (inds, buf, tmp_rhs.to_f64())
    }

    /// ObjectivePropagation::shouldBePropagated
    pub(crate) fn obj_should_be_propagated(&self) -> bool {
        let o = self.obj();
        if *o.is_propagated {
            return false;
        }
        if *o.num_inf > 1 {
            return false;
        }
        if *self.infeasible {
            return false;
        }
        let upper_limit = self.upper_limit;
        if upper_limit == INF {
            return false;
        }
        if upper_limit - o.objective_lower.to_f64() > *o.capacity_threshold {
            return false;
        }
        true
    }

    /// ObjectivePropagation::recomputeCapacityThreshold
    fn obj_recompute_capacity_threshold(&mut self) {
        let o = self.obj();
        let b = self.bounds();
        *o.capacity_threshold =
            capacity_threshold(o.contributions, o.partition_sets, o.cost, o.obj_nonzeros, o.partition_starts, &b);
    }
}

/// The node of the contribution at a position of the clique partitions
#[inline(always)]
fn partition_position(pp: usize) -> i32 {
    pp as i32
}

impl Ctx {
    /// ObjectivePropagation::propagate
    pub(crate) fn obj_propagate(&mut self) {
        let (upper_limit, mut capacity, num_inf) = {
            let mut d = self.dom();
            if !d.obj_should_be_propagated() {
                return;
            }
            let o = d.obj();
            let upper_limit = d.upper_limit;
            if *o.num_inf == 0 && o.objective_lower.to_f64() > upper_limit {
                d.set_infeasible(REASON_OBJECTIVE, 0);
                return;
            }
            (upper_limit, upper_limit - *o.objective_lower, *o.num_inf)
        };
        let objective = Reason { kind: REASON_OBJECTIVE, index: 0 };

        if num_inf == 1 {
            // Scan non-binary columns for infinite bound contribution until
            // the one column that contributes with an infinite bound is found
            // which is the only column that can be propagated
            let (start, n) = {
                let d = self.dom();
                let o = d.obj();
                (o.num_binaries as usize, o.obj_nonzeros.len())
            };
            for i in start..n {
                let (col, cost) = {
                    let d = self.dom();
                    let o = d.obj();
                    let col = o.obj_nonzeros[i] as usize;
                    let cost = o.cost[col];
                    if (cost > 0.0 && d.col_lower[col] != -INF) || (cost < 0.0 && d.col_upper[col] != INF) {
                        continue;
                    }
                    (col, cost)
                };
                self.check_change_bound(if cost > 0.0 { UPPER } else { LOWER }, col, capacity / cost, objective);
                break;
            }
        } else {
            let np = self.dom().obj().num_partitions();
            loop {
                let mut num_bound_changes = 0;
                for i in 0..np {
                    // the worst and second worst contributions
                    let (worst, second_worst, contribution) = {
                        let d = self.dom();
                        let mut o = d.obj();
                        let t = o.tree(i);
                        let worst = t.first();
                        if worst == -1 {
                            continue;
                        }
                        let second_worst = t.successor(worst);
                        let mut contribution = o.contributions[worst as usize].contribution;
                        if second_worst != -1 {
                            contribution -= o.contributions[second_worst as usize].contribution;
                        }
                        (worst, second_worst, contribution)
                    };

                    // the upper limit already uses a tolerance, so we can do a
                    // hard cutoff
                    if contribution > capacity.to_f64() {
                        let chg = {
                            let d = self.dom();
                            let o = d.obj();
                            let col = o.contributions[worst as usize].col;
                            let c = col as usize;
                            if o.cost[c] > 0.0 {
                                (d.col_upper[c] > 0.0).then_some(DomChg { boundval: 0.0, column: col, boundtype: UPPER })
                            } else {
                                (d.col_lower[c] < 1.0).then_some(DomChg { boundval: 1.0, column: col, boundtype: LOWER })
                            }
                        };
                        if let Some(chg) = chg {
                            num_bound_changes += 1;
                            self.change_bound(chg, objective);
                            if self.infeasible() {
                                break;
                            }
                        }
                    } else if second_worst != -1 {
                        // it might be that we can fix the column with the
                        // lowest possible contribution to its bound value that
                        // yields the highest objective contribution
                        loop {
                            let chg = {
                                let d = self.dom();
                                let mut o = d.obj();
                                let (first, best) = {
                                    let t = o.tree(i);
                                    (t.first(), t.last())
                                };
                                if best == first {
                                    break;
                                }
                                if o.contributions[first as usize].contribution
                                    - o.contributions[best as usize].contribution
                                    > capacity.to_f64()
                                {
                                    let col = o.contributions[best as usize].col;
                                    if o.cost[col as usize] > 0.0 {
                                        DomChg { boundval: 1.0, column: col, boundtype: LOWER }
                                    } else {
                                        DomChg { boundval: 0.0, column: col, boundtype: UPPER }
                                    }
                                } else {
                                    break;
                                }
                            };
                            num_bound_changes += 1;
                            self.change_bound(chg, objective);
                            if self.infeasible() {
                                break;
                            }
                        }
                        if self.infeasible() {
                            break;
                        }
                    }
                }

                if self.infeasible() {
                    break;
                }

                let (start, n) = {
                    let d = self.dom();
                    let o = d.obj();
                    (o.partition_starts[np] as usize, o.obj_nonzeros.len())
                };
                for i in start..n {
                    let (col, boundtype, val) = {
                        let d = self.dom();
                        let o = d.obj();
                        let col = o.obj_nonzeros[i] as usize;
                        let cost = o.cost[col];
                        let upper = cost > 0.0;
                        let b = if upper { d.col_lower[col] } else { d.col_upper[col] };
                        if d.obj_bound_hopeless(col, upper, capacity.to_f64(), b, cost) {
                            continue;
                        }
                        (col, if upper { UPPER } else { LOWER }, (capacity + b * cost) / cost)
                    };
                    if self.check_change_bound(boundtype, col, val, objective) {
                        num_bound_changes += 1;
                    }
                    if self.infeasible() {
                        break;
                    }
                }
                if self.infeasible() {
                    break;
                }

                if num_bound_changes == 0 {
                    break;
                }
                capacity = upper_limit - *self.dom().obj().objective_lower;
            }
        }

        let mut d = self.dom();
        d.obj_recompute_capacity_threshold();
        *d.obj().is_propagated = true;
    }
}
