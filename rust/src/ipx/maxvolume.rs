//! Maxvolume (maxvolume.h/.cc): basis updates that increase the volume of
//! the scaled basis matrix (the "Russian algorithm"), sequentially over all
//! columns or with the slice heuristic of column weights.

use super::basis::{BasicStatus, Basis};
use super::control::Control;
use super::fmt::sci2;
use super::indexed_vector::IndexedVector;
use super::lu::LuResult;
use super::sparse_matrix::dot_column;
use super::utils::sortperm;
use super::{cmax, Int};
use std::time::Instant;

const PIVOT_ZERO_TOL: f64 = 1e-7;

pub struct Maxvolume<'a> {
    control: &'a Control,
    updates: Int,
    skipped: Int,
    passes: Int,
    slices: Int,
    volinc: f64,
    time: f64,
    tblnnz: Int,
    tblmax: f64,
    frobnorm_squared: f64,
}

struct Slice {
    colscale: Vec<f64>,
    invscale_basic: Vec<f64>,
    tblrow_used: Vec<bool>,
    colweights: Vec<f64>,
    lhs: IndexedVector,
    row: IndexedVector,
    work: Vec<f64>,
}

/// The indices of the second largest and the largest entry of |weights|
/// (the largest at the back).
fn find_largest(weights: &[f64]) -> Vec<usize> {
    let mut jmax = 0; // index of largest element
    let mut jmax2 = 0; // index of second largest element
    let mut wmax = 0.0;
    let mut wmax2 = 0.0;
    for (j, w) in weights.iter().enumerate() {
        let w = w.abs();
        if w > wmax {
            wmax2 = wmax;
            wmax = w;
            jmax2 = jmax;
            jmax = j;
        } else if w > wmax2 {
            wmax2 = w;
            jmax2 = j;
        }
    }
    vec![jmax2, jmax]
}

impl<'a> Maxvolume<'a> {
    pub fn new(control: &'a Control) -> Self {
        Maxvolume {
            control,
            updates: 0,
            skipped: 0,
            passes: 0,
            slices: 0,
            volinc: 0.0,
            time: 0.0,
            tblnnz: 0,
            tblmax: 0.0,
            frobnorm_squared: 0.0,
        }
    }

    pub fn updates(&self) -> Int {
        self.updates
    }
    pub fn time(&self) -> f64 {
        self.time
    }

    fn reset(&mut self) {
        self.updates = 0;
        self.skipped = 0;
        self.passes = 0;
        self.slices = 0;
        self.volinc = 0.0;
        self.time = 0.0;
        self.tblnnz = 0;
        self.tblmax = 0.0;
        self.frobnorm_squared = 0.0;
    }

    /// Passes over all nonbasic columns in decreasing order of scaling
    /// factor, exchanging when the volume increases by more than
    /// volume_tol, until a pass makes no update.
    pub fn run_sequential(&mut self, colscale: Option<&[f64]>, basis: &mut Basis) -> LuResult<Int> {
        let m = basis.model().rows();
        let n = basis.model().cols();
        let mut ftran = IndexedVector::new(m);
        let timer = Instant::now();
        let mut errflag = 0;

        let maxpasses = self.control.maxpasses();
        let volumetol = cmax(self.control.volume_tol(), 1.0);

        // Inverse scaling factors of basic variables; zero for BASIC_FREE,
        // so that these are never pivoted out of the basis.
        let mut invscale_basic = vec![0.0; m];
        for p in 0..m {
            let j = basis.at(p);
            if basis.status_of(j) == BasicStatus::Basic {
                invscale_basic[p] = colscale.map_or(1.0, |c| 1.0 / c[j]);
            }
        }

        self.reset();
        while self.passes < maxpasses || maxpasses < 0 {
            self.tblnnz = 0;
            self.tblmax = 0.0;
            self.frobnorm_squared = 0.0;
            let mut updates_last = 0; // # basis updates in this pass
            let mut candidates = sortperm(n + m, colscale, false);
            while let Some(&j) = candidates.last() {
                let j = j as usize;
                let dj = colscale.map_or(1.0, |c| c[j]);
                if dj == 0.0 {
                    // all remaining columns have scaling factor 0
                    break;
                }
                if basis.status_of(j) != BasicStatus::Nonbasic {
                    candidates.pop();
                    continue;
                }
                errflag = self.control.interrupt_check(-1);
                if errflag != 0 {
                    break;
                }
                basis.solve_for_update(j, Some(&mut ftran))?;
                let mut pmax = 0usize;
                let mut vmax = 0.0;
                let (mut tblnnz, mut frob) = (self.tblnnz, self.frobnorm_squared);
                ftran.for_each_nonzero(|p, x| {
                    let v = x.abs() * invscale_basic[p] * dj;
                    if v > vmax {
                        vmax = v;
                        pmax = p;
                    }
                    tblnnz += (v != 0.0) as Int;
                    frob = v.mul_add(v, frob);
                });
                self.tblnnz = tblnnz;
                self.frobnorm_squared = frob;
                self.tblmax = cmax(self.tblmax, vmax);
                if vmax <= volumetol {
                    self.skipped += 1;
                    candidates.pop();
                    continue;
                }

                let jb = basis.at(pmax);
                let (err, exchanged) = basis.exchange_if_stable(jb, j, ftran[pmax], -1)?;
                errflag = err;
                if errflag != 0 {
                    break;
                }
                if !exchanged {
                    // factorization was unstable, try again
                    continue;
                }
                invscale_basic[pmax] = 1.0 / dj;
                updates_last += 1;
                self.volinc += vmax.log2();
                candidates.pop();
            }
            self.updates += updates_last;
            self.passes += 1;
            if updates_last == 0 || errflag != 0 {
                break;
            }
        }
        self.time = timer.elapsed().as_secs_f64();
        Ok(errflag)
    }

    /// The heuristic: the tableau matrix is split into row slices, and in
    /// each slice the column with maximum weight is exchanged until the
    /// volume does not increase enough.
    pub fn run_heuristic(&mut self, colscale: Option<&[f64]>, basis: &mut Basis) -> LuResult<Int> {
        let m = basis.model().rows();
        let n = basis.model().cols();
        let mut slice = Slice {
            colscale: vec![0.0; n + m],
            invscale_basic: vec![0.0; m],
            tblrow_used: vec![false; m],
            colweights: vec![0.0; n + m],
            lhs: IndexedVector::new(m),
            row: IndexedVector::new(n + m),
            work: vec![0.0; m],
        };
        let mut errflag = 0;
        let timer = Instant::now();

        self.reset();
        let mut num_slices = 5 + std::cmp::max(m as Int / self.control.rows_per_slice(), 0);
        num_slices = std::cmp::min(num_slices, m as Int);

        // Maintain a copy of the inverse scaling factors of basic variables.
        for p in 0..m {
            let j = basis.at(p);
            if basis.status_of(j) == BasicStatus::Basic {
                slice.invscale_basic[p] = colscale.map_or(1.0, |c| 1.0 / c[j]);
            }
        }

        // Copy of the column scaling factors, in which skipped columns are
        // set to zero (and not scanned again).
        for j in 0..n + m {
            if basis.status_of(j) == BasicStatus::Nonbasic {
                slice.colscale[j] = colscale.map_or(1.0, |c| c[j]);
            }
        }

        // Split tableau matrix into num_slices row slices. In each call to
        // driver() exactly one row from each slice has tblrow_used set.
        let perm = sortperm(m, Some(&slice.invscale_basic), false);
        for s in 0..num_slices {
            for i in 0..m {
                slice.tblrow_used[perm[i] as usize] = i as Int % num_slices == s;
            }
            errflag = self.driver(basis, &mut slice)?;
            if errflag != 0 {
                break;
            }
        }

        self.time = timer.elapsed().as_secs_f64();
        self.passes = -1;
        self.slices = num_slices;
        Ok(errflag)
    }

    fn driver(&mut self, basis: &mut Basis, slice: &mut Slice) -> LuResult<Int> {
        let m = basis.model().rows();
        let n = basis.model().cols();
        let mut errflag = 0;

        let volumetol = cmax(self.control.volume_tol(), 1.0);
        let maxskip = self.control.maxskip_updates();

        let Slice {
            colscale,
            invscale_basic,
            tblrow_used,
            colweights,
            lhs,
            row,
            work,
        } = slice;

        // Compute column weights.
        for p in 0..m {
            work[p] = if tblrow_used[p] { invscale_basic[p] } else { 0.0 };
        }
        basis.solve_dense_inplace(work, b'T')?;
        {
            let ai = basis.model().ai();
            for j in 0..n + m {
                if colscale[j] != 0.0 {
                    let sum = dot_column(ai, j, work);
                    colweights[j] = sum * colscale[j];
                } else {
                    colweights[j] = 0.0;
                }
            }
        }

        let mut candidates: Vec<usize> = Vec::new();
        let mut skipped = 0;
        loop {
            // Pick column with maximum weight.
            if candidates.is_empty() {
                candidates = find_largest(colweights);
            }
            let jn = *candidates.last().unwrap();
            let weight = colweights[jn];
            if weight == 0.0 {
                break;
            }

            errflag = self.control.interrupt_check(-1);
            if errflag != 0 {
                break;
            }

            // Find maximum scaled FTRAN entry.
            basis.solve_for_update(jn, Some(lhs))?;
            let pmax = scale_ftran(colscale[jn], invscale_basic, lhs);
            let scaled_pivot = lhs[pmax];
            let vmax = scaled_pivot.abs();

            // Skip the column if exchange does not increase volume enough.
            if vmax <= volumetol {
                colweights[jn] = 0.0;
                colscale[jn] = 0.0;
                candidates.pop();
                skipped += 1;
                if skipped > maxskip && maxskip >= 0 {
                    break;
                }
                continue;
            }

            // Recompute column weight from FTRAN.
            let mut weight_recomp = 0.0;
            lhs.for_each_nonzero(|p, x| {
                if tblrow_used[p] {
                    weight_recomp += x;
                }
            });

            // Update basis.
            let jb = basis.at(pmax);
            basis.tableau_row(jb, lhs, row, true)?;
            let pivot = row[jn];
            if pivot.abs() < 1e-3 {
                self.control
                    .debug_out(3, &format!(" |pivot| {}(maxvolume)\n", sci2(pivot.abs())));
            }
            let (err, exchanged) = basis.exchange_if_stable(jb, jn, pivot, 0)?;
            errflag = err;
            if errflag != 0 {
                break;
            }
            if !exchanged {
                // factorization was unstable, try again
                continue;
            }
            self.updates += 1;
            self.volinc += vmax.log2();

            // Update colscale and invscale_basic.
            let dn = colscale[jn];
            let dbinv = invscale_basic[pmax];
            colscale[jb] = 1.0 / invscale_basic[pmax];
            invscale_basic[pmax] = 1.0 / colscale[jn];
            colscale[jn] = 0.0;

            // Update column weights.
            let used = if tblrow_used[pmax] { 1.0 } else { 0.0 };
            let alpha = (used - weight_recomp) / (dn * pivot);
            row.for_each_nonzero(|j, x| {
                colweights[j] = (alpha * x).mul_add(colscale[j], colweights[j]);
            });
            colweights[jb] = used + alpha / dbinv;
            colweights[jn] = 0.0;
            candidates.clear();
        }

        self.skipped += skipped;
        Ok(errflag)
    }
}

/// Scales the FTRAN entries by colscale_jn * invscale_basic[p] in place and
/// returns the position of the maximum among those with |pivot| >
/// kPivotZeroTol (0 if none).
fn scale_ftran(colscale_jn: f64, invscale_basic: &[f64], ftran: &mut IndexedVector) -> usize {
    let mut vmax = 0.0;
    let mut pmax = 0;
    ftran.for_each_nonzero_mut(|p, pivot| {
        let scaled_pivot = *pivot * colscale_jn * invscale_basic[p];
        let v = scaled_pivot.abs();
        if v > vmax && pivot.abs() > PIVOT_ZERO_TOL {
            vmax = v;
            pmax = p;
        }
        *pivot = scaled_pivot;
    });
    pmax
}
