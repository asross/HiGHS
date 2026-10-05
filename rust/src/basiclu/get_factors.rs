//! basiclu_get_factors.c: extract the factors of a fresh factorization.

use super::{Int, Lu};

/// Compressed column output arrays: (colptr, rowidx, value)
pub(crate) type Csc<'b> = (&'b mut [Int], &'b mut [Int], &'b mut [f64]);

impl Lu<'_> {
    /// Requires nupdate == 0 (checked by the caller). L gets the unit
    /// diagonal at the front of each column, U the pivot at the end; row
    /// indices are sorted.
    pub(crate) fn get_factors(
        &mut self,
        rowperm: Option<&mut [Int]>,
        colperm: Option<&mut [Int]>,
        l: Option<Csc>,
        u: Option<Csc>,
    ) {
        let m = self.m as usize;
        let pivotcol = &*self.colcount_flink;
        let pivotrow = &*self.colcount_blink;
        let colptr = &mut self.rowcount_blink[..m]; // iwork1, size m workspace

        if let Some(rowperm) = rowperm {
            rowperm[..m].copy_from_slice(&pivotrow[..m]);
        }
        if let Some(colperm) = colperm {
            colperm[..m].copy_from_slice(&pivotcol[..m]);
        }

        if let Some((lcolptr, lrowidx, lvalue_)) = l {
            let lbegin_p = &*self.lbegin_p;
            let ltbegin_p = &self.wflink[m + 1..];
            let lindex = &*self.lindex;
            let lvalue = &*self.lvalue;
            let p = &self.wblink[m + 1..];

            // L[:,k] will hold the elimination factors from the k-th pivot
            // step. First set the column pointers and store the unit diagonal
            // elements at the front of each column. Then scatter each row of
            // L' into the columnwise L so that the row indices become sorted.
            let mut put: Int = 0;
            for k in 0..m {
                lcolptr[k] = put;
                lrowidx[put as usize] = k as Int;
                lvalue_[put as usize] = 1.0;
                put += 1;
                colptr[p[k] as usize] = put; // next free position in column
                                             // subtract 1 because internal storage uses (-1) terminators
                put += lbegin_p[k + 1] - lbegin_p[k] - 1;
            }
            lcolptr[m] = put;
            debug_assert!(put == self.lnz + m as Int);

            for k in 0..m {
                let mut pos = ltbegin_p[k] as usize;
                loop {
                    let i = lindex[pos];
                    if i < 0 {
                        break;
                    }
                    let put = colptr[i as usize] as usize;
                    colptr[i as usize] += 1;
                    lrowidx[put] = k as Int;
                    lvalue_[put] = lvalue[pos];
                    pos += 1;
                }
            }
        }

        if let Some((ucolptr, urowidx, uvalue_)) = u {
            let wbegin = &*self.wbegin;
            let wend = &*self.wend;
            let windex = &*self.windex;
            let wvalue = &*self.wvalue;
            let col_pivot = &*self.col_pivot;

            // U[:,k] will hold the column of B from the k-th pivot step.
            // First set the column pointers and store the pivot element at
            // the end of each column. Then scatter each row of U' into the
            // columnwise U so that the row indices become sorted.
            colptr.fill(0); // column counts
            for j in 0..m {
                for pos in wbegin[j] as usize..wend[j] as usize {
                    colptr[windex[pos] as usize] += 1;
                }
            }
            let mut put: Int = 0;
            for k in 0..m {
                // set column pointers
                let j = pivotcol[k] as usize;
                ucolptr[k] = put;
                put += colptr[j];
                colptr[j] = ucolptr[k]; // next free position in column
                urowidx[put as usize] = k as Int;
                uvalue_[put as usize] = col_pivot[j];
                put += 1;
            }
            ucolptr[m] = put;
            debug_assert!(put == self.unz + m as Int);
            for k in 0..m {
                // scatter row k
                let j = pivotcol[k] as usize;
                for pos in wbegin[j] as usize..wend[j] as usize {
                    let c = windex[pos] as usize;
                    let put = colptr[c] as usize;
                    colptr[c] += 1;
                    urowidx[put] = k as Int;
                    uvalue_[put] = wvalue[pos];
                }
            }
        }
    }
}
