//! basiclu_factorize.c and lu_factorize_bump.c: the factorization driver.

use crate::util::fma::ClangFma;

use super::condest::condest;
use super::file::list_remove;
use super::{
    Int, Lu, BUILD_FACTORS, ERROR_INVALID_CALL, FACTORIZE_BUMP, NO_TASK, OK, SETUP_BUMP,
    SINGLETONS, WARNING_SINGULAR_MATRIX,
};

impl Lu<'_> {
    /// lu_factorize_bump: eliminate pivots from the bump until it is done
    /// or reallocation is required
    pub(crate) fn factorize_bump(&mut self) -> Int {
        let m = self.m;
        let mut status = OK;
        while self.rank + self.rankdef < m {
            // Find pivot element. Markowitz search need not be called if the
            // previous call to pivot() returned for reallocation. In this
            // case pivot_col is valid.
            if self.pivot_col < 0 {
                self.markowitz();
            }
            debug_assert!(self.pivot_col >= 0);

            if self.pivot_row < 0 {
                // Eliminate empty column without choosing a pivot.
                list_remove(self.colcount_flink, self.colcount_blink, self.pivot_col);
                self.pivot_col = -1;
                self.rankdef += 1;
            } else {
                // Eliminate pivot. This may require reallocation.
                debug_assert!(self.pinv[self.pivot_row as usize] == -1);
                debug_assert!(self.qinv[self.pivot_col as usize] == -1);
                status = self.pivot();
                if status != OK {
                    break;
                }
                self.pinv[self.pivot_row as usize] = self.rank;
                self.qinv[self.pivot_col as usize] = self.rank;
                self.pivot_col = -1;
                self.pivot_row = -1;
                self.rank += 1;
            }
        }
        status
    }

    /// The body of basiclu_factorize after the store and arguments have been
    /// checked. Each of the four parts may request reallocation; then return
    /// to the caller, keeping the entry point in `task`.
    pub(crate) fn factorize(
        &mut self,
        bbegin: &[Int],
        bend: &[Int],
        bi: &[Int],
        bx: &[f64],
        c0ntinue: bool,
    ) -> Int {
        if !c0ntinue {
            self.reset();
            self.task = SINGLETONS;
        }

        // continue factorization
        let start = self.task;
        if !(SINGLETONS..=BUILD_FACTORS).contains(&start) {
            return ERROR_INVALID_CALL;
        }
        if start <= SINGLETONS {
            self.task = SINGLETONS;
            let status = self.singletons(bbegin, bend, bi, bx);
            if status != OK {
                return status;
            }
        }
        if start <= SETUP_BUMP {
            self.task = SETUP_BUMP;
            let status = self.setup_bump(bbegin, bend, bi, bx);
            if status != OK {
                return status;
            }
        }
        if start <= FACTORIZE_BUMP {
            self.task = FACTORIZE_BUMP;
            let status = self.factorize_bump();
            if status != OK {
                return status;
            }
        }
        self.task = BUILD_FACTORS;
        let mut status = self.build_factors();
        if status != OK {
            return status;
        }

        // factorization successfully finished
        self.task = NO_TASK;
        self.nupdate = 0; // make factorization valid
        self.ftran_for_update = -1;
        self.btran_for_update = -1;
        self.nfactorize += 1;

        let m = self.m as usize;
        let (lbegin, p) = (&self.wbegin[m + 1..], &self.wblink[m + 1..]);
        let (c, norm, norminv) = condest(
            self.m,
            lbegin,
            self.lindex,
            self.lvalue,
            None,
            p,
            false,
            self.work1,
        );
        self.condest_l = c;
        self.norm_l = norm;
        self.normest_linv = norminv;
        let (c, norm, norminv) = condest(
            self.m,
            self.ubegin,
            self.uindex,
            self.uvalue,
            Some(self.row_pivot),
            p,
            true,
            self.work1,
        );
        self.condest_u = c;
        self.norm_u = norm;
        self.normest_uinv = norminv;

        // measure numerical stability of the factorization
        self.residual_test(bbegin, bend, bi, bx);

        // factor_cost is a deterministic measure of the factorization cost.
        // The parameters have been adjusted such that (on the author's
        // computer) 1e-6 * factor_cost =~ time_factorize.
        //
        // update_cost measures the accumulated cost of updates/solves
        // compared to the last factorization:
        // update_cost = update_cost_numer / update_cost_denom, with
        // update_cost_denom fixed here and update_cost_numer zero here and
        // increased by solves/updates.
        let factor_cost = 0.008f64.mul_add_c(
            self.factor_flops as f64,
            0.20f64.mul_add_c(
                self.nsearch_pivot as f64,
                0.20f64.mul_add_c(
                    self.bump_nz as f64,
                    0.07f64.mul_add_c(self.matrix_nz as f64, 0.04 * self.m as f64),
                ),
            ),
        );
        self.update_cost_denom = factor_cost * 250.0;

        if self.rank < self.m {
            status = WARNING_SINGULAR_MATRIX;
        }
        status
    }
}
