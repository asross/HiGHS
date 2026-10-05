//! lu_build_factors.c: build rowwise and columnwise form of L and U.
//!
//! BASICLU maintains the factorization in the form
//!
//!   B = L * R^1 * R^2 * ... * R^{nforrest} * U,
//!
//! where L[p,p] is unit lower triangular and U[pivotrow,pivotcol] is upper
//! triangular. After refactorization nforrest = 0 and p and pivotrow hold
//! the same permutation. pivotrow and pivotcol are modified by updates, p
//! is not.
//!
//! Permutations: p[0..m-1]; pivotrow[0..pivotlen-1], pivotcol[0..pivotlen-1]
//! with m <= pivotlen < 2*m may contain duplicates, the last occurrence of
//! each index being its position in the pivot sequence (see
//! garbage_perm()); pmap, qmap with i = pmap[j], j = qmap[i] when element
//! (i,j) of U is a pivot element.
//!
//! L: Lindex/Lvalue[0..Lnz+m-1] hold L columnwise without the unit
//! diagonal, each column terminated by index -1, row indices of B.
//! Lbegin[i] points to column i, Lbegin_p[k] to column p[k].
//! Lindex/Lvalue[Lnz+m..2*(Lnz+m)-1] hold L rowwise without the unit
//! diagonal, rows terminated by -1; column i holds the elimination factors
//! from the pivot step in which row i was pivot row. Ltbegin[i] points to
//! row i, Ltbegin_p[k] to row p[k].
//!
//! R^k: Lindex/Lvalue[Rbegin[k]..Rbegin[k+1]-1] hold the nontrivial column
//! of R^k without the unit diagonal (row indices of B); Rbegin[0] is one
//! past the L storage. eta_row[k] is the row index of the diagonal element
//! of that column.
//!
//! U: Uindex/Uvalue[1..Unz+m] hold U columnwise without the pivots, each
//! column terminated by -1, row indices of B; updates introduce gaps.
//! Uindex[0] = -1, and all empty columns point there. Ubegin[i] points to
//! column qmap[i]. Windex/Wvalue hold U rowwise without the pivots (column
//! indices of B) in a file with gaps, out of order: Wbegin[j], Wend[j]
//! delimit row pmap[j], Wflink/Wblink link the rows in memory order.
//! col_pivot, row_pivot hold the pivots by column and by row index.

use super::file::{file_empty, list_move};
use super::{Int, Lu, OK, REALLOCATE};

impl Lu<'_> {
    /// Returns REALLOCATE (require more memory in L, U, and/or W) or OK
    pub(crate) fn build_factors(&mut self) -> Int {
        let m = self.m;
        let mu = m as usize;
        let rank = self.rank;
        let ru = rank as usize;
        let lmem = self.lmem;
        let umem = self.umem;
        let wmem = self.wmem;
        let pad = self.pad;
        let stretch = self.stretch;
        let pinv = &mut *self.pinv;
        let qinv = &mut *self.qinv;
        let pivotcol = &mut *self.colcount_flink;
        let pivotrow = &mut *self.colcount_blink;
        let lbegin_p = &mut *self.lbegin_p;
        let ubegin = &mut *self.ubegin;
        let (wbegin, lbegin) = self.wbegin.split_at_mut(mu + 1);
        let (wend, ltbegin) = self.wend.split_at_mut(mu + 1);
        let (wflink, ltbegin_p) = self.wflink.split_at_mut(mu + 1);
        let (wblink, p) = self.wblink.split_at_mut(mu + 1);
        let rbegin = &mut *self.rowcount_flink;
        let col_pivot = &mut *self.col_pivot;
        let row_pivot = &mut *self.row_pivot;
        let lindex = &mut *self.lindex;
        let lvalue = &mut *self.lvalue;
        let uindex = &mut *self.uindex;
        let uvalue = &mut *self.uvalue;
        let windex = &mut *self.windex;
        let wvalue = &mut *self.wvalue;
        let iwork1 = &mut self.rowcount_blink[..mu];
        let mut status = OK;

        // So far L is stored columnwise in Lindex, Lvalue and U rowwise in
        // Uindex, Uvalue. The factorization has computed rank columns of L
        // and rank rows of U. If rank < m, then the columns which have not
        // been pivotal will be removed from U.
        let lnz = lbegin_p[ru] - rank; // each column is terminated by -1
        let mut unz = ubegin[ru]; // might be decreased when rank < m

        // Calculate memory and reallocate. The rowwise and columnwise storage
        // of L both need space for Lnz nonzeros + m terminators. The same for
        // the columnwise storage of U except that Uindex[0] = -1 is reserved
        // to accommodate pointers to empty columns. In the rowwise storage of
        // U each row with nz nonzeros is padded by stretch*nz + pad elements.
        let need = 2 * (lnz + m);
        if lmem < need {
            self.addmem_l = need - lmem;
            status = REALLOCATE;
        }
        let need = unz + m + 1;
        if umem < need {
            self.addmem_u = need - umem;
            status = REALLOCATE;
        }
        let need = (stretch.mul_add(unz as f64, unz as f64) + (m * pad) as f64) as Int;
        if wmem < need {
            self.addmem_w = need - wmem;
            status = REALLOCATE;
        }
        if status != OK {
            return status;
        }

        // Build permutations

        // Append columns/rows which have not been pivotal to the end of the
        // pivot sequence. Build pivotrow, pivotcol as inverse of pinv, qinv.
        let mut lrank = rank;
        for i in 0..mu {
            if pinv[i] < 0 {
                pinv[i] = lrank;
                lrank += 1;
            }
            pivotrow[pinv[i] as usize] = i as Int;
        }
        debug_assert!(lrank == m);
        let mut lrank = rank;
        for j in 0..mu {
            if qinv[j] < 0 {
                qinv[j] = lrank;
                lrank += 1;
            }
            pivotcol[qinv[j] as usize] = j as Int;
        }
        debug_assert!(lrank == m);

        // Dependent columns get unit pivot elements.
        for k in ru..mu {
            col_pivot[pivotcol[k] as usize] = 1.0;
        }

        // Lower triangular factor

        // L columnwise. If rank < m, then complete with unit columns (no
        // off-diagonals, so nothing to store here).
        let mut put = lbegin_p[ru];
        for k in ru..mu {
            lindex[put as usize] = -1;
            put += 1;
            lbegin_p[k + 1] = put;
        }
        debug_assert!(lbegin_p[mu] == lnz + m);
        for i in 0..mu {
            lbegin[i] = lbegin_p[pinv[i] as usize];
        }

        // L rowwise.
        iwork1.fill(0); // row counts
        for get in 0..(lnz + m) as usize {
            let i = lindex[get];
            if i >= 0 {
                iwork1[i as usize] += 1;
            }
        }
        let mut put = lnz + m; // L rowwise starts here
        for k in 0..mu {
            let i = pivotrow[k] as usize;
            ltbegin_p[k] = put;
            ltbegin[i] = put;
            put += iwork1[i];
            lindex[put as usize] = -1; // terminate row
            put += 1;
            iwork1[i] = ltbegin_p[k];
        }
        debug_assert!(put == 2 * (lnz + m));
        for k in 0..mu {
            // fill rows
            let ipivot = pivotrow[k];
            let mut get = lbegin_p[k] as usize;
            loop {
                let i = lindex[get];
                if i < 0 {
                    break;
                }
                let put = iwork1[i as usize] as usize; // put into row i
                iwork1[i as usize] += 1;
                lindex[put] = ipivot;
                lvalue[put] = lvalue[get];
                get += 1;
            }
        }
        rbegin[0] = 2 * (lnz + m); // beginning of update etas

        // Upper triangular factor

        // U rowwise.
        file_empty(m, wbegin, wend, wflink, wblink, wmem);
        iwork1.fill(0); // column counts
        let mut put: Int = 0;

        // Use separate loops for full rank and rank deficient
        // factorizations. In the first case no elements are removed from U,
        // so skip the test.
        if rank == m {
            for k in 0..mu {
                let jpivot = pivotcol[k] as usize;
                wbegin[jpivot] = put;
                let mut nz = 0;
                for pos in ubegin[k] as usize..ubegin[k + 1] as usize {
                    let j = uindex[pos];
                    windex[put as usize] = j;
                    wvalue[put as usize] = uvalue[pos];
                    put += 1;
                    iwork1[j as usize] += 1;
                    nz += 1;
                }
                wend[jpivot] = put;
                put = (put as f64 + stretch.mul_add(nz as f64, pad as f64)) as Int;
                list_move(jpivot as Int, 0, wflink, wblink, m, None);
            }
        } else {
            unz = 0; // actual number of nonzeros
            for k in 0..ru {
                let jpivot = pivotcol[k] as usize;
                wbegin[jpivot] = put;
                let mut nz = 0;
                for pos in ubegin[k] as usize..ubegin[k + 1] as usize {
                    let j = uindex[pos];
                    if qinv[j as usize] < rank {
                        windex[put as usize] = j;
                        wvalue[put as usize] = uvalue[pos];
                        put += 1;
                        iwork1[j as usize] += 1;
                        nz += 1;
                    }
                }
                wend[jpivot] = put;
                put = (put as f64 + stretch.mul_add(nz as f64, pad as f64)) as Int;
                list_move(jpivot as Int, 0, wflink, wblink, m, None);
                unz += nz;
            }
            for k in ru..mu {
                let jpivot = pivotcol[k] as usize;
                wbegin[jpivot] = put;
                wend[jpivot] = put;
                put += pad;
                list_move(jpivot as Int, 0, wflink, wblink, m, None);
            }
        }
        debug_assert!(put <= wend[mu]);
        wbegin[mu] = put; // beginning of free space

        // U columnwise.
        uindex[0] = -1;
        let mut put: Int = 1;
        for k in 0..mu {
            // set column pointers
            let j = pivotcol[k] as usize;
            let i = pivotrow[k] as usize;
            let nz = iwork1[j];
            if nz == 0 {
                ubegin[i] = 0; // empty columns all in position 0
            } else {
                ubegin[i] = put;
                put += nz;
                uindex[put as usize] = -1; // terminate column
                put += 1;
            }
            iwork1[j] = ubegin[i];
        }
        ubegin[mu] = put;
        for k in 0..mu {
            // fill columns
            let jpivot = pivotcol[k] as usize;
            let i = pivotrow[k];
            for pos in wbegin[jpivot] as usize..wend[jpivot] as usize {
                let j = windex[pos] as usize;
                let put = iwork1[j] as usize;
                iwork1[j] += 1;
                uindex[put] = i;
                uvalue[put] = wvalue[pos];
            }
        }

        // Build pivot sequence

        // Build row-column mappings, overwriting pinv, qinv.
        let (pmap, qmap) = (pinv, qinv);
        for k in 0..mu {
            let i = pivotrow[k];
            let j = pivotcol[k];
            pmap[j as usize] = i;
            qmap[i as usize] = j;
        }

        // Build pivots by row index.
        let mut max_pivot: f64 = 0.0;
        let mut min_pivot = f64::INFINITY;
        for i in 0..mu {
            row_pivot[i] = col_pivot[qmap[i] as usize];
            let pivot = row_pivot[i].abs();
            max_pivot = pivot.max(max_pivot);
            min_pivot = pivot.min(min_pivot);
        }

        p[..mu].copy_from_slice(&pivotrow[..mu]);

        self.min_pivot = min_pivot;
        self.max_pivot = max_pivot;
        self.pivotlen = m;
        self.lnz = lnz;
        self.unz = unz;
        self.rnz = 0;
        status
    }
}
