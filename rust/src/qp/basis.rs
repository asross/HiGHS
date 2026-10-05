//! Basis (qpsolver/basis.cpp): the working set of the active-set method,
//! factorized with HFactor as the basis matrix of A' (columns: the
//! constraints; logicals: the variable bounds).

use super::pricing::Pricing;
use super::vector::{MatrixBase, QpVector};
use super::{BasisStatus, Settings, SolverStatus};
use crate::factor::{AMatrix, HFactor, UPDATE_FT};
use crate::hvector::OwnedHVec;

const K_DEFAULT_PIVOT_THRESHOLD: f64 = 0.1;
const K_DEFAULT_PIVOT_TOLERANCE: f64 = 1e-10;

pub struct Basis {
    num_var: usize,
    /// A' (num_row = num_var, num_col = num_con)
    atran: MatrixBase,
    a_start: Vec<i32>,
    a_index: Vec<i32>,
    factor: HFactor,
    updatessinceinvert: i32,
    /// Constraints active in the basis
    active: Vec<usize>,
    /// Constraints in the basis but not active (the null space)
    nonactive: Vec<usize>,
    baseindex: Vec<i32>,
    status: Vec<BasisStatus>,
    /// Position in the factor of each constraint, or -1
    index_in_factor: Vec<i32>,
    work: OwnedHVec,
    col_aq: OwnedHVec,
    row_ep: OwnedHVec,
    ztprod_res: QpVector,
    buffer_zprod: QpVector,
    reinversion_hint: bool,
}

impl Basis {
    pub fn new(
        atran: MatrixBase,
        num_con: usize,
        active: &[usize],
        status: &[BasisStatus],
        inactive: &[usize],
    ) -> Self {
        let num_var = atran.num_row;
        let mut st = vec![BasisStatus::Inactive; num_var + num_con];
        for (&a, &s) in active.iter().zip(status) {
            st[a] = s;
        }
        for &i in inactive {
            st[i] = BasisStatus::InactiveInBasis;
        }
        let n = num_var as i32;
        let mut basis = Basis {
            num_var,
            a_start: atran.start.iter().map(|&s| s as i32).collect(),
            a_index: atran.index.iter().map(|&s| s as i32).collect(),
            atran,
            factor: HFactor::default(),
            updatessinceinvert: 0,
            active: active.to_vec(),
            nonactive: inactive.to_vec(),
            baseindex: vec![],
            status: st,
            index_in_factor: vec![],
            work: OwnedHVec::new(n),
            col_aq: OwnedHVec::new(n),
            row_ep: OwnedHVec::new(n),
            ztprod_res: QpVector::new(num_var),
            buffer_zprod: QpVector::new(num_var),
            reinversion_hint: false,
        };
        basis.build();
        basis
    }

    fn amatrix(&self) -> AMatrix<'_> {
        AMatrix {
            num_col: self.atran.num_col as i32,
            start: &self.a_start,
            index: &self.a_index,
            value: &self.atran.value,
        }
    }

    fn factorize(&mut self) {
        let mut refactored = false;
        let mut baseindex = std::mem::take(&mut self.baseindex);
        let mut factor = std::mem::take(&mut self.factor);
        factor.build(
            K_DEFAULT_PIVOT_THRESHOLD,
            K_DEFAULT_PIVOT_TOLERANCE,
            f64::INFINITY,
            &self.amatrix(),
            &mut baseindex,
            None,
            &mut refactored,
        );
        self.factor = factor;
        self.baseindex = baseindex;
        self.updatessinceinvert = 0;
        self.index_in_factor = vec![-1; self.atran.num_row + self.atran.num_col];
        for (i, &b) in self.baseindex.iter().enumerate() {
            self.index_in_factor[b as usize] = i as i32;
        }
    }

    fn build(&mut self) {
        assert_eq!(self.nonactive.len() + self.active.len(), self.num_var);
        self.baseindex = self.nonactive.iter().chain(&self.active).map(|&i| i as i32).collect();
        self.factor = HFactor::default();
        let (nc, nr) = (self.atran.num_col as i32, self.num_var as i32);
        self.factor.setup(nc, nr, nr, &self.a_start, UPDATE_FT);
        self.factorize();
    }

    pub fn rebuild(&mut self) {
        self.factorize();
        self.reinversion_hint = false;
    }

    pub fn atran(&self) -> &MatrixBase {
        &self.atran
    }

    pub fn reinversion_hint(&self) -> bool {
        self.reinversion_hint
    }

    pub fn num_active(&self) -> usize {
        self.active.len()
    }

    pub fn active(&self) -> &[usize] {
        &self.active
    }

    pub fn inactive(&self) -> &[usize] {
        &self.nonactive
    }

    pub fn index_in_factor(&self) -> &[i32] {
        &self.index_in_factor
    }

    pub fn status(&self, con: usize) -> BasisStatus {
        self.status[con]
    }

    /// Move a constraint into the null space part of the basis
    pub fn deactivate(&mut self, con: usize) {
        debug_assert!(self.active.contains(&con));
        self.status[con] = BasisStatus::InactiveInBasis;
        self.active.retain(|&a| a != con);
        self.nonactive.push(con);
    }

    pub fn activate(
        &mut self,
        settings: &Settings,
        con: usize,
        newstatus: BasisStatus,
        nonactivetoremove: usize,
        pricing: &mut Pricing,
    ) -> SolverStatus {
        if self.active.contains(&con) {
            println!("Degeneracy? constraint {con} already in basis");
            return SolverStatus::Degenerate;
        }
        self.status[nonactivetoremove] = BasisStatus::Inactive;
        self.status[con] = newstatus;
        self.active.push(con);

        let rowtoremove = self.index_in_factor[nonactivetoremove];
        self.baseindex[rowtoremove as usize] = con as i32;
        self.nonactive.retain(|&a| a != nonactivetoremove);
        self.updatebasis(settings, con, nonactivetoremove, pricing);

        if self.updatessinceinvert != 0 {
            self.index_in_factor[nonactivetoremove] = -1;
            self.index_in_factor[con] = rowtoremove;
        }
        SolverStatus::Ok
    }

    fn updatebasis(&mut self, settings: &Settings, newactivecon: usize, droppedcon: usize, pricing: &mut Pricing) {
        if newactivecon == droppedcon {
            return;
        }
        const K_HINT_NOT_CHANGED: i32 = 99999;
        let mut hint = K_HINT_NOT_CHANGED;

        let row = self.index_in_factor[droppedcon];
        // (The buffered row_ep is never valid: C++ records it as buffered_q)
        self.row_ep.clear();
        self.row_ep.pack_flag = true;
        self.row_ep.index[0] = row;
        self.row_ep.array[row as usize] = 1.0;
        self.row_ep.count = 1;
        let factor = &self.factor;
        self.row_ep.with(|v| factor.btran(v, 1.0));

        let aq = hvec2vec(&self.col_aq);
        let ep = hvec2vec(&self.row_ep);
        pricing.update_weights(self, &aq, &ep, droppedcon);

        let aq = self.col_aq.view();
        let ep = self.row_ep.view();
        self.factor.update(&[aq], &[ep], &[row], &mut hint, None, &self.baseindex);

        self.updatessinceinvert += 1;
        if self.updatessinceinvert >= settings.reinvertfrequency || hint != K_HINT_NOT_CHANGED {
            self.reinversion_hint = true;
        }
    }

    /// Solve with B (ftran) or B' (btran) for `rhs` into `target`; an
    /// FTRAN result is kept as col_aq for the update if `buffer`
    fn solve(&mut self, rhs: &QpVector, target: &mut QpVector, ftran: bool, buffer: bool) {
        // vec2hvec into a fresh copy of the (cleared) buffer
        let w = &mut self.work;
        w.array.fill(0.0);
        for (i, &j) in rhs.nz().iter().enumerate() {
            w.index[i] = j as i32;
            w.array[j] = rhs.value[j];
        }
        w.count = rhs.num_nz as i32;
        w.pack_flag = true;
        w.synthetic_tick = 0.0;
        let factor = &self.factor;
        w.with(|v| if ftran { factor.ftran(v, 1.0) } else { factor.btran(v, 1.0) });
        if buffer {
            copy_hvec(w, &mut self.col_aq);
        }
        // hvec2vec
        target.reset();
        for i in 0..w.count as usize {
            let j = w.index[i] as usize;
            target.index[i] = j;
            target.value[j] = w.array[j];
        }
        target.num_nz = w.count as usize;
    }

    pub fn ftran(&mut self, rhs: &QpVector, target: &mut QpVector, buffer: bool) {
        self.solve(rhs, target, true, buffer);
    }

    /// (A buffered BTRAN is never used, see updatebasis)
    pub fn btran(&mut self, rhs: &QpVector, target: &mut QpVector) {
        self.solve(rhs, target, false, false);
    }

    pub fn recomputex(&mut self, con_lo: &[f64], con_up: &[f64], var_lo: &[f64], var_up: &[f64]) -> QpVector {
        let n = self.num_var;
        let num_con = con_lo.len();
        let mut rhs = QpVector::new(n);
        for i in 0..n {
            let con = self.active[i];
            let at = self.index_in_factor[con] as usize;
            let lower = self.status[con] == BasisStatus::ActiveAtLower;
            rhs.value[at] = match (con < num_con, lower) {
                (true, true) => con_lo[con],
                (true, false) => con_up[con],
                (false, true) => var_lo[con - num_con],
                (false, false) => var_up[con - num_con],
            };
            rhs.index[i] = i;
            rhs.num_nz += 1;
        }
        let mut x = QpVector::new(n);
        self.btran(&rhs, &mut x);
        x
    }

    /// target = Z' rhs
    pub fn ztprod(&mut self, rhs: &QpVector, target: &mut QpVector, buffer: bool) {
        let mut res = std::mem::replace(&mut self.ztprod_res, QpVector::new(0));
        self.ftran(rhs, &mut res, buffer);
        target.reset();
        for (i, &nonactive) in self.nonactive.iter().enumerate() {
            let idx = self.index_in_factor[nonactive] as usize;
            target.index[i] = i;
            target.value[i] = res.value[idx];
        }
        target.resparsify();
        self.ztprod_res = res;
    }

    /// target = Z rhs
    pub fn zprod(&mut self, rhs: &QpVector, target: &mut QpVector) {
        let mut b = std::mem::replace(&mut self.buffer_zprod, QpVector::new(0));
        b.reset();
        b.dim = target.dim;
        for i in 0..rhs.num_nz {
            let nz = rhs.index[i];
            let idx = self.index_in_factor[self.nonactive[nz]] as usize;
            b.index[i] = idx;
            b.value[idx] = rhs.value[nz];
        }
        b.num_nz = rhs.num_nz;
        self.btran(&b, target);
        self.buffer_zprod = b;
    }
}

/// HVectorBase::copy plus the pack data, as Basis::ftran buffers col_aq
fn copy_hvec(from: &OwnedHVec, to: &mut OwnedHVec) {
    to.clear();
    to.synthetic_tick = from.synthetic_tick;
    to.count = from.count;
    for i in 0..from.count as usize {
        let j = from.index[i];
        to.index[i] = j;
        to.array[j as usize] = from.array[j as usize];
    }
    let pc = from.pack_count as usize;
    to.pack_index[..pc].copy_from_slice(&from.pack_index[..pc]);
    to.pack_value[..pc].copy_from_slice(&from.pack_value[..pc]);
    to.pack_count = from.pack_count;
    to.pack_flag = from.pack_flag;
}

/// Basis::hvec2vec into a new QpVector of the HVector's size
fn hvec2vec(h: &OwnedHVec) -> QpVector {
    let mut v = QpVector::new(h.size as usize);
    for i in 0..h.count as usize {
        let j = h.index[i] as usize;
        v.index[i] = j;
        v.value[j] = h.array[j];
    }
    v.num_nz = h.count as usize;
    v
}
