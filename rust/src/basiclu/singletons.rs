//! lu_singletons.c: build initial triangular factors.
//!
//! Initialize the data structures which store the LU factors during
//! factorization and eliminate pivots with Markowitz cost zero.
//!
//! During factorization the inverse pivot sequence is recorded in pinv,
//! qinv: pinv[i] >= 0 if row i was pivot row in stage pinv[i], -1 if not
//! pivotal yet (likewise qinv for columns).
//!
//! L is composed columnwise in Lindex, Lvalue: after rank steps,
//! Lbegin_p[rank] is the next unused position and Lindex[Lbegin_p[k]..]
//! holds the column of L computed in stage k without the unit diagonal,
//! terminated by a negative index. U is composed rowwise in Uindex, Uvalue:
//! Uindex[Ubegin[k]..Ubegin[k+1]-1] holds the row of U computed in stage k
//! without the pivot element.
//!
//! When nzbias >= 0, singleton columns are eliminated before singleton rows
//! (to keep L sparse), otherwise rows first. Off-diagonals from singleton
//! columns are stored in U, off-diagonals from singleton rows in L (divided
//! by the diagonal). Diagonals are stored in col_pivot. Pivots that are zero
//! or less than abstol in magnitude are not taken; the bump factorization
//! detects the singularity.

use super::{Int, Lu, ERROR_INVALID_ARGUMENT, OK, REALLOCATE};

impl Lu<'_> {
    /// Returns REALLOCATE (less than nnz(B) memory in L, U or W),
    /// ERROR_INVALID_ARGUMENT (B invalid: negative column count, index out
    /// of range, duplicates) or OK
    pub(crate) fn singletons(
        &mut self,
        bbegin: &[Int],
        bend: &[Int],
        bi: &[Int],
        bx: &[f64],
    ) -> Int {
        let m = self.m;
        let mu = m as usize;
        let lmem = self.lmem;
        let umem = self.umem;
        let wmem = self.wmem;
        let abstol = self.abstol;
        let nzbias = self.nzbias;
        let pinv = &mut *self.pinv;
        let qinv = &mut *self.qinv;
        let lbegin_p = &mut *self.lbegin_p;
        let ubegin = &mut *self.ubegin;
        let col_pivot = &mut *self.col_pivot;
        let lindex = &mut *self.lindex;
        let lvalue = &mut *self.lvalue;
        let uindex = &mut *self.uindex;
        let uvalue = &mut *self.uvalue;
        let (iwork1, iwork2) = self.rowcount_blink.split_at_mut(mu);

        // build B rowwise in W
        let btp = &mut *self.wbegin;
        let bti = &mut *self.windex;
        let btx = &mut *self.wvalue;

        // Check matrix and build transpose

        // Check pointers and count nnz(B).
        let mut bnz: Int = 0;
        for j in 0..mu {
            if bend[j] < bbegin[j] {
                return ERROR_INVALID_ARGUMENT;
            }
            bnz += bend[j] - bbegin[j];
        }

        // Check if sufficient memory in L, U, W.
        let mut ok = true;
        if lmem < bnz {
            self.addmem_l = bnz - lmem;
            ok = false;
        }
        if umem < bnz {
            self.addmem_u = bnz - umem;
            ok = false;
        }
        if wmem < bnz {
            self.addmem_w = bnz - wmem;
            ok = false;
        }
        if !ok {
            return REALLOCATE;
        }

        // Count nz per row, check indices.
        iwork1.fill(0); // row counts
        for j in 0..mu {
            for pos in bbegin[j] as usize..bend[j] as usize {
                let i = bi[pos];
                if i < 0 || i >= m {
                    return ERROR_INVALID_ARGUMENT;
                }
                iwork1[i as usize] += 1;
            }
        }

        // Pack matrix rowwise, check for duplicates.
        let mut put: Int = 0;
        for i in 0..mu {
            // set row pointers
            btp[i] = put;
            put += iwork1[i];
            iwork1[i] = btp[i];
        }
        btp[mu] = put;
        debug_assert!(put == bnz);
        let mut ok = true;
        for j in 0..mu {
            // fill rows
            for pos in bbegin[j] as usize..bend[j] as usize {
                let i = bi[pos] as usize;
                let put = iwork1[i];
                iwork1[i] += 1;
                bti[put as usize] = j as Int;
                btx[put as usize] = bx[pos];
                if put > btp[i] && bti[put as usize - 1] == j as Int {
                    ok = false;
                }
            }
        }
        if !ok {
            return ERROR_INVALID_ARGUMENT;
        }

        // Pivot singletons

        // No pivot rows or pivot columns so far.
        pinv.fill(-1);
        qinv.fill(-1);

        let btp = &*btp;
        let bti = &*bti;
        let btx = &*btx;
        lbegin_p[0] = 0;
        ubegin[0] = 0;
        let mut rank: Int = 0;
        if nzbias >= 0 {
            // put more in U
            rank = singleton_cols(
                m, bbegin, bend, bi, btp, bti, btx, ubegin, uindex, uvalue, lbegin_p, lindex,
                col_pivot, pinv, qinv, iwork1, iwork2, rank, abstol,
            );
            rank = singleton_rows(
                m, bbegin, bend, bi, bx, btp, bti, ubegin, lbegin_p, lindex, lvalue, col_pivot,
                pinv, qinv, iwork1, iwork2, rank, abstol,
            );
        } else {
            // put more in L
            rank = singleton_rows(
                m, bbegin, bend, bi, bx, btp, bti, ubegin, lbegin_p, lindex, lvalue, col_pivot,
                pinv, qinv, iwork1, iwork2, rank, abstol,
            );
            rank = singleton_cols(
                m, bbegin, bend, bi, btp, bti, btx, ubegin, uindex, uvalue, lbegin_p, lindex,
                col_pivot, pinv, qinv, iwork1, iwork2, rank, abstol,
            );
        }

        // pinv, qinv were used as nonzero counters. Reset to -1 if not
        // pivoted.
        for x in pinv.iter_mut().chain(qinv.iter_mut()) {
            if *x < 0 {
                *x = -1;
            }
        }

        self.matrix_nz = bnz;
        self.rank = rank;
        OK
    }
}

/// Successively remove singleton columns from the active submatrix (columns
/// j with qinv[j] < 0 and rows i with pinv[i] < 0). Removing a singleton
/// column and its row may create new singleton columns, which are appended
/// to a queue. Stops when the active submatrix has no singleton columns.
///
/// For each active column j, iset[j] is the XOR of the row indices in the
/// column in the active submatrix; for a singleton column this is its
/// single row index (J. Gilbert, see T. Davis, "Direct methods for sparse
/// linear systems", ex 3.7).
///
/// For each eliminated column its row is stored in U without the pivot,
/// the pivot in col_pivot, and an empty column is appended to L. Zero or
/// tiny pivots and empty columns are not eliminated (we want singularities
/// at the end of the pivot sequence).
#[allow(clippy::too_many_arguments)]
fn singleton_cols(
    m: Int,
    bbegin: &[Int],
    bend: &[Int],
    bi: &[Int],
    btp: &[Int],
    bti: &[Int],
    btx: &[f64],
    up: &mut [Int],
    ui: &mut [Int],
    ux: &mut [f64],
    lp: &mut [Int],
    li: &mut [Int],
    col_pivot: &mut [f64],
    pinv: &mut [Int],
    qinv: &mut [Int],
    iset: &mut [Int],
    queue: &mut [Int],
    mut rank: Int,
    abstol: f64,
) -> Int {
    let mut rk = rank;

    // Build index sets and initialize queue.
    let mut tail = 0usize;
    for j in 0..m as usize {
        if qinv[j] < 0 {
            let nz = bend[j] - bbegin[j];
            let mut i = 0;
            for pos in bbegin[j] as usize..bend[j] as usize {
                i ^= bi[pos]; // put row into set j
            }
            iset[j] = i;
            qinv[j] = -nz - 1; // use as nonzero counter
            if nz == 1 {
                queue[tail] = j as Int;
                tail += 1;
            }
        }
    }

    // Eliminate singleton columns.
    let mut put = up[rank as usize];
    let mut front = 0;
    while front < tail {
        let j = queue[front] as usize;
        front += 1;
        debug_assert!(qinv[j] == -2 || qinv[j] == -1);
        if qinv[j] == -1 {
            continue; // empty column in active submatrix
        }
        let i = iset[j] as usize;
        debug_assert!(pinv[i] < 0);
        let end = btp[i + 1] as usize;
        let mut pos = btp[i] as usize;
        while bti[pos] != j as Int {
            // find pivot
            pos += 1;
        }
        let piv = btx[pos];
        if piv == 0.0 || piv.abs() < abstol {
            continue; // skip singularity
        }

        // Eliminate pivot.
        qinv[j] = rank;
        pinv[i] = rank;
        for pos in btp[i] as usize..end {
            let j2 = bti[pos] as usize;
            // test is mandatory because the initial active submatrix may not
            // be the entire matrix (rows eliminated before)
            if qinv[j2] < 0 {
                ui[put as usize] = j2 as Int;
                ux[put as usize] = btx[pos];
                put += 1;
                iset[j2] ^= i as Int; // remove i from set j2
                qinv[j2] += 1;
                if qinv[j2] == -2 {
                    queue[tail] = j2 as Int; // new singleton
                    tail += 1;
                }
            }
        }
        up[rank as usize + 1] = put;
        col_pivot[j] = piv;
        rank += 1;
    }

    // Put empty columns into L.
    let mut pos = lp[rk as usize];
    while rk < rank {
        li[pos as usize] = -1;
        pos += 1;
        lp[rk as usize + 1] = pos;
        rk += 1;
    }
    rank
}

/// As singleton_cols, except that for each singleton row the associated
/// column is stored in L and divided by the pivot element.
#[allow(clippy::too_many_arguments)]
fn singleton_rows(
    m: Int,
    bbegin: &[Int],
    bend: &[Int],
    bi: &[Int],
    bx: &[f64],
    btp: &[Int],
    bti: &[Int],
    up: &mut [Int],
    lp: &mut [Int],
    li: &mut [Int],
    lx: &mut [f64],
    col_pivot: &mut [f64],
    pinv: &mut [Int],
    qinv: &mut [Int],
    iset: &mut [Int],
    queue: &mut [Int],
    mut rank: Int,
    abstol: f64,
) -> Int {
    let mut rk = rank;

    // Build index sets and initialize queue.
    let mut tail = 0usize;
    for i in 0..m as usize {
        if pinv[i] < 0 {
            let nz = btp[i + 1] - btp[i];
            let mut j = 0;
            for pos in btp[i] as usize..btp[i + 1] as usize {
                j ^= bti[pos]; // put column into set i
            }
            iset[i] = j;
            pinv[i] = -nz - 1; // use as nonzero counter
            if nz == 1 {
                queue[tail] = i as Int;
                tail += 1;
            }
        }
    }

    // Eliminate singleton rows.
    let mut put = lp[rank as usize];
    let mut front = 0;
    while front < tail {
        let i = queue[front] as usize;
        front += 1;
        debug_assert!(pinv[i] == -2 || pinv[i] == -1);
        if pinv[i] == -1 {
            continue; // empty column in active submatrix
        }
        let j = iset[i] as usize;
        debug_assert!(qinv[j] < 0);
        let end = bend[j] as usize;
        let mut pos = bbegin[j] as usize;
        while bi[pos] != i as Int {
            // find pivot
            pos += 1;
        }
        let piv = bx[pos];
        if piv == 0.0 || piv.abs() < abstol {
            continue; // skip singularity
        }

        // Eliminate pivot.
        qinv[j] = rank;
        pinv[i] = rank;
        for pos in bbegin[j] as usize..end {
            let i2 = bi[pos] as usize;
            // test is mandatory because the initial active submatrix may not
            // be the entire matrix (columns eliminated before)
            if pinv[i2] < 0 {
                li[put as usize] = i2 as Int;
                lx[put as usize] = bx[pos] / piv;
                put += 1;
                iset[i2] ^= j as Int; // remove j from set i2
                pinv[i2] += 1;
                if pinv[i2] == -2 {
                    queue[tail] = i2 as Int; // new singleton
                    tail += 1;
                }
            }
        }
        li[put as usize] = -1; // terminate column
        put += 1;
        lp[rank as usize + 1] = put;
        col_pivot[j] = piv;
        rank += 1;
    }

    // Put empty rows into U.
    let pos = up[rk as usize];
    while rk < rank {
        up[rk as usize + 1] = pos;
        rk += 1;
    }
    rank
}
