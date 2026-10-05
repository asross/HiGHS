//! cupdlp_scaling.c: Ruiz (10 passes, infinity norm) and then
//! Pock-Chambolle (alpha = 1) scaling, the fixed choice of Init_Scaling.
//! The L2 scaling is never switched on through HiGHS and is not ported.

use super::linalg::{nrm2, nrminf, Sparse};
use super::Log;

const RUIZ_TIMES: usize = 10;
const PC_ALPHA: f64 = 1.0;

/// CUPDLPscaling
pub(super) struct Scaling {
    pub(super) scaled: bool,
    pub(super) col_scale: Vec<f64>,
    pub(super) row_scale: Vec<f64>,
    /// 2-norms of the unscaled cost and rhs
    pub(super) norm_cost: f64,
    pub(super) norm_rhs: f64,
}

/// The LP data that scaling changes
pub(super) struct ScaledLp<'a> {
    pub(super) csc: &'a mut Sparse,
    pub(super) cost: &'a mut [f64],
    pub(super) lower: &'a mut [f64],
    pub(super) upper: &'a mut [f64],
    pub(super) rhs: &'a mut [f64],
}

impl Scaling {
    /// Init_Scaling
    pub(super) fn new(log: &Log, cost: &[f64], rhs: &[f64]) -> Self {
        let s = Scaling {
            scaled: false,
            col_scale: vec![1.0; cost.len()],
            row_scale: vec![1.0; rhs.len()],
            norm_cost: nrm2(cost),
            norm_rhs: nrm2(rhs),
        };
        if log.level > 0 {
            log.out(&format!(
                "Using cost norm = {:>9} and RHS norm = {:>9}\n",
                super::g(s.norm_cost, 3),
                super::g(s.norm_rhs, 3)
            ));
        }
        s
    }

    /// (row, column) scale factors when the LP is scaled
    pub(super) fn active(&self) -> (Option<&[f64]>, Option<&[f64]>) {
        if self.scaled {
            (Some(&self.row_scale), Some(&self.col_scale))
        } else {
            (None, None)
        }
    }

    /// PDHG_Scale_Data
    pub(super) fn scale(&mut self, log: &Log, if_scaling: bool, lp: &mut ScaledLp) {
        if !if_scaling {
            return;
        }
        let dashes = "--------------------------------------------------\n";
        log.info(dashes);
        log.info("running scaling\n");
        log.info("- use Ruiz scaling\n");
        self.ruiz(lp);
        log.info("- use PC scaling\n");
        self.pc(lp);
        self.scaled = true;
        log.info(dashes);
    }

    /// cupdlp_ruiz_scaling
    fn ruiz(&mut self, lp: &mut ScaledLp) {
        let (ncols, nrows) = (lp.cost.len(), lp.rhs.len());
        let mut col = vec![0.0; ncols];
        let mut row = vec![0.0; nrows];
        for _ in 0..RUIZ_TIMES {
            row.fill(0.0);
            for (j, c) in col.iter_mut().enumerate() {
                let (b, e) = (lp.csc.start[j], lp.csc.start[j + 1]);
                *c = if b == e {
                    0.0
                } else {
                    nrminf(&lp.csc.value[b..e]).sqrt()
                };
                if *c == 0.0 {
                    *c = 1.0;
                }
            }
            if nrows > 0 {
                for (&i, &v) in lp.csc.index.iter().zip(&lp.csc.value) {
                    let r = &mut row[i as usize];
                    if *r < v.abs() {
                        *r = v.abs();
                    }
                }
                for r in row.iter_mut() {
                    *r = if *r == 0.0 { 1.0 } else { r.sqrt() };
                }
            }
            self.apply(lp, &col, &row);
        }
    }

    /// cupdlp_pc_scaling
    fn pc(&mut self, lp: &mut ScaledLp) {
        let (ncols, nrows) = (lp.cost.len(), lp.rhs.len());
        let mut col = vec![0.0; ncols];
        let mut row = vec![0.0; nrows];
        if nrows > 0 {
            for (j, c) in col.iter_mut().enumerate() {
                for p in lp.csc.start[j]..lp.csc.start[j + 1] {
                    *c += lp.csc.value[p].abs().powf(PC_ALPHA);
                }
                *c = c.powf(1.0 / PC_ALPHA).sqrt();
                if *c == 0.0 {
                    *c = 1.0;
                }
            }
            for (&i, &v) in lp.csc.index.iter().zip(&lp.csc.value) {
                row[i as usize] += v.abs().powf(2.0 - PC_ALPHA);
            }
            for r in row.iter_mut() {
                *r = r.powf(1.0 / (2.0 - PC_ALPHA)).sqrt();
                if *r == 0.0 {
                    *r = 1.0;
                }
            }
        }
        self.apply(lp, &col, &row);
    }

    /// scale_problem, and the update of the accumulated scaling
    fn apply(&mut self, lp: &mut ScaledLp, col: &[f64], row: &[f64]) {
        for (j, &c) in col.iter().enumerate() {
            lp.cost[j] /= c;
            lp.lower[j] *= c;
            lp.upper[j] *= c;
        }
        for (r, &s) in lp.rhs.iter_mut().zip(row) {
            *r /= s;
        }
        for (v, &i) in lp.csc.value.iter_mut().zip(&lp.csc.index) {
            *v /= row[i as usize];
        }
        for (j, &c) in col.iter().enumerate() {
            let (b, e) = (lp.csc.start[j], lp.csc.start[j + 1]);
            lp.csc.value[b..e].iter_mut().for_each(|v| *v /= c);
        }
        self.col_scale
            .iter_mut()
            .zip(col)
            .for_each(|(s, &c)| *s *= c);
        self.row_scale
            .iter_mut()
            .zip(row)
            .for_each(|(s, &r)| *s *= r);
    }
}
