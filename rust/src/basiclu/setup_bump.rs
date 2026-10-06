//! lu_setup_bump.c: set up the data structures for the bump factorization.
//!
//! The bump is composed of rows i and columns j with pinv[i] < 0 and
//! qinv[j] < 0. It is stored in Windex, Wvalue columnwise and additionally
//! its pattern rowwise: Wbegin[j]/Wend[j] delimit column j, Wbegin[m+i]/
//! Wend[m+i] row i (empty when begin == end; row values are undefined).
//! Wflink, Wblink hold the 2*m lines in a double linked list in memory
//! order.
//!
//! The Markowitz search requires double linked lists of columns (rows) with
//! equal column (row) counts, colcount_flink/blink and rowcount_flink/blink:
//! m elements in m+2 lists. Column j is in list 0 <= nz <= m when it has nz
//! nonzeros in the active submatrix; row i can alternatively be in list m+1
//! to exclude it temporarily from the search. The maximum of each active
//! column j is kept in col_pivot[j], replaced by the pivot element when j
//! becomes pivotal.

use crate::util::fma::ClangFma;

use super::file::{file_empty, list_add, list_init, list_move};
use super::{Int, Lu, OK, REALLOCATE};

impl Lu<'_> {
    /// Returns REALLOCATE (require more memory in W) or OK
    pub(crate) fn setup_bump(
        &mut self,
        bbegin: &[Int],
        bend: &[Int],
        bi: &[Int],
        bx: &[f64],
    ) -> Int {
        let m = self.m;
        let mu = m as usize;
        let rank = self.rank;
        let wmem = self.wmem;
        let bnz = self.matrix_nz;
        let lnz = self.lbegin_p[rank as usize] - rank;
        let unz = self.ubegin[rank as usize];
        let abstol = self.abstol;
        let pad = self.pad;
        let stretch = self.stretch;
        let colcount_flink = &mut *self.colcount_flink;
        let colcount_blink = &mut *self.colcount_blink;
        let rowcount_flink = &mut *self.rowcount_flink;
        let rowcount_blink = &mut *self.rowcount_blink;
        let pinv = &*self.pinv;
        let qinv = &*self.qinv;
        let wbegin = &mut *self.wbegin;
        let wend = &mut *self.wend;
        let wflink = &mut *self.wflink;
        let wblink = &mut *self.wblink;
        let windex = &mut *self.windex;
        let wvalue = &mut *self.wvalue;
        let colmax = &mut *self.col_pivot;
        let iwork0 = &mut *self.iwork0;

        let mut bump_nz = bnz - lnz - unz - rank; // changes if columns are dropped
        debug_assert!(lnz >= 0 && unz >= 0 && bump_nz >= 0);
        debug_assert!(iwork0.iter().all(|&x| x == 0));

        // Calculate memory and reallocate. For each row/column with nz
        // nonzeros add stretch*nz+pad elements extra space for fill-in.
        let mut need =
            (stretch.mul_add_c(bump_nz as f64, bump_nz as f64) + ((m - rank) * pad) as f64) as Int;
        need *= 2; // rowwise + columnwise
        if need > wmem {
            self.addmem_w = need - wmem;
            return REALLOCATE;
        }

        file_empty(2 * m, wbegin, wend, wflink, wblink, wmem);

        // Build columnwise storage. Build row counts in iwork0.
        let mut min_colnz = list_init(colcount_flink, colcount_blink, m, m + 2);
        let mut put: Int = 0;
        for j in 0..mu {
            if qinv[j] >= 0 {
                continue;
            }
            let mut cnz = 0; // count nz per column
            let mut cmx = 0.0f64; // find column maximum
            for pos in bbegin[j] as usize..bend[j] as usize {
                if pinv[bi[pos] as usize] >= 0 {
                    continue;
                }
                cmx = cmx.max(bx[pos].abs());
                cnz += 1;
            }
            if cmx == 0.0 || cmx < abstol {
                // Leave column of active submatrix empty.
                colmax[j] = 0.0;
                list_add(
                    j as Int,
                    0,
                    colcount_flink,
                    colcount_blink,
                    m,
                    Some(&mut min_colnz),
                );
                bump_nz -= cnz;
            } else {
                // Copy column into active submatrix.
                colmax[j] = cmx;
                list_add(
                    j as Int,
                    cnz,
                    colcount_flink,
                    colcount_blink,
                    m,
                    Some(&mut min_colnz),
                );
                wbegin[j] = put;
                for pos in bbegin[j] as usize..bend[j] as usize {
                    let i = bi[pos];
                    if pinv[i as usize] >= 0 {
                        continue;
                    }
                    windex[put as usize] = i;
                    wvalue[put as usize] = bx[pos];
                    put += 1;
                    iwork0[i as usize] += 1;
                }
                wend[j] = put;
                put = (put as f64 + stretch.mul_add_c(cnz as f64, pad as f64)) as Int;
                // reappend line to list end
                list_move(j as Int, 0, wflink, wblink, 2 * m, None);
            }
        }

        // Build rowwise storage (pattern only).
        let mut min_rownz = list_init(rowcount_flink, rowcount_blink, m, m + 2);
        for i in 0..mu {
            // set row pointers
            if pinv[i] >= 0 {
                continue;
            }
            let rnz = iwork0[i];
            iwork0[i] = 0;
            list_add(
                i as Int,
                rnz,
                rowcount_flink,
                rowcount_blink,
                m,
                Some(&mut min_rownz),
            );
            wbegin[mu + i] = put;
            wend[mu + i] = put;
            put += rnz;
            // reappend line to list end
            list_move(m + i as Int, 0, wflink, wblink, 2 * m, None);
            put = (put as f64 + stretch.mul_add_c(rnz as f64, pad as f64)) as Int;
        }
        for j in 0..mu {
            // fill rows
            for pos in wbegin[j] as usize..wend[j] as usize {
                let i = windex[pos] as usize;
                windex[wend[mu + i] as usize] = j as Int;
                wend[mu + i] += 1;
            }
        }
        wbegin[2 * mu] = put; // set beginning of free space
        debug_assert!(wbegin[2 * mu] <= wend[2 * mu]);

        self.bump_nz = bump_nz;
        self.bump_size = m - rank;
        self.min_colnz = min_colnz;
        self.min_rownz = min_rownz;
        OK
    }
}
