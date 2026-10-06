//! lu_pivot.c: pivot elimination from the active submatrix.
//!
//! The pivot operation removes row pivot_row and column pivot_col from the
//! active submatrix and applies a rank-1 update to the remaining active
//! submatrix. It updates the row and column counts and the column maxima.
//!
//! Each pivot elimination adds one column to L and one row to U. On entry
//! Lbegin_p[rank] and Ubegin[rank] point to the next position in Lindex,
//! Lvalue respectively Uindex, Uvalue. On return the column in L is
//! terminated by index -1 and Lbegin_p[rank+1], Ubegin[rank+1] point to the
//! next free position. Memory is checked (and reallocation requested)
//! before any data structure is changed.
//!
//! Columns of the active submatrix are updated like in Clp (J. Forrest):
//! unmodified entries are compressed and the entries updated or filled in
//! by the pivot column are appended. Compared to the Suhl/Suhl technique
//! this often gives sparser factors (presumably through tie breaking in the
//! Markowitz search, since updated elements move to the end of the column,
//! and likewise for rows).

use crate::util::fma::ClangFma;

use super::file::{file_compress, file_reappend, list_move, list_remove};
use super::{Int, Lu, OK, REALLOCATE};

/// Maximum number of off-diagonals in the pivot column handled by
/// pivot_small(), which uses one bit of an int64 mask per updated row. A
/// fixed threshold keeps pivot operations identical on all architectures.
const MAXROW_SMALL: Int = 64;

/// `x += a + stretch*a + pad` for the room estimates (C: lu_int arithmetic
/// with a double expression, contracted to an FMA)
#[inline]
fn add_room(x: Int, a: Int, stretch: f64, pad: Int) -> Int {
    (x as f64 + (stretch.mul_add_c(a as f64, a as f64) + pad as f64)) as Int
}

/// `a + stretch*b + pad` truncated to lu_int
#[inline]
fn room_for(a: Int, b: Int, stretch: f64, pad: Int) -> Int {
    (stretch.mul_add_c(b as f64, a as f64) + pad as f64) as Int
}

impl Lu<'_> {
    pub(crate) fn pivot(&mut self) -> Int {
        let m = self.m as usize;
        let rank = self.rank as usize;
        let pivot_col = self.pivot_col as usize;
        let pivot_row = self.pivot_row as usize;
        let nz_col = self.wend[pivot_col] - self.wbegin[pivot_col];
        let nz_row = self.wend[m + pivot_row] - self.wbegin[m + pivot_row];
        let mut status = OK;
        debug_assert!(nz_row >= 1 && nz_col >= 1);

        // Check if room is available in L and U.
        let room = self.lmem - self.lbegin_p[rank];
        let need = nz_col; // # off-diagonals in pivot col + end marker (-1)
        if room < need {
            self.addmem_l = need - room;
            status = REALLOCATE;
        }
        let room = self.umem - self.ubegin[rank];
        let need = nz_row - 1; // # off-diagonals in pivot row
        if room < need {
            self.addmem_u = need - room;
            status = REALLOCATE;
        }
        if status != OK {
            return status;
        }

        // Branch out implementation of pivot operation.
        status = if nz_row == 1 {
            self.pivot_singleton_row()
        } else if nz_col == 1 {
            self.pivot_singleton_col()
        } else if nz_col == 2 {
            self.pivot_doubleton_col()
        } else if nz_col - 1 <= MAXROW_SMALL {
            self.pivot_small()
        } else {
            self.pivot_any()
        };

        // Remove all entries in columns whose maximum entry has dropped below
        // absolute pivot tolerance.
        if status == OK {
            for pos in self.ubegin[rank] as usize..self.ubegin[rank + 1] as usize {
                let j = self.uindex[pos];
                let cmx = self.col_pivot[j as usize];
                if cmx == 0.0 || cmx < self.abstol {
                    self.remove_col(j);
                }
            }
        }

        self.factor_flops += (nz_col - 1) * (nz_row - 1);
        status
    }

    /// Room check shared by pivot_any and pivot_small: at most each updated
    /// row and column is reappended and filled in with rnz1 respectively
    /// cnz1 elements. Moves the pivot to the front of pivot row and column.
    /// Returns (cbeg, cend, rbeg, rend, pivot) after a possible file
    /// compression, or Err(REALLOCATE).
    fn pivot_prepare(&mut self) -> Result<(usize, usize, usize, usize, f64), Int> {
        let m = self.m as usize;
        let pad = self.pad;
        let stretch = self.stretch;
        let pivot_col = self.pivot_col;
        let pivot_row = self.pivot_row;
        let wbegin = &mut *self.wbegin;
        let wend = &mut *self.wend;
        let windex = &mut *self.windex;
        let wvalue = &mut *self.wvalue;

        let mut cbeg = wbegin[pivot_col as usize] as usize; // changed by file compression
        let mut cend = wend[pivot_col as usize] as usize;
        let mut rbeg = wbegin[m + pivot_row as usize] as usize;
        let mut rend = wend[m + pivot_row as usize] as usize;
        let cnz1 = (cend - cbeg - 1) as Int; // nz in pivot column except pivot
        let rnz1 = (rend - rbeg - 1) as Int; // nz in pivot row except pivot

        let mut grow: Int = 0;
        let mut where_ = usize::MAX;
        for pos in cbeg..cend {
            let i = windex[pos];
            if i == pivot_row {
                where_ = pos;
            } else {
                let nz = wend[m + i as usize] - wbegin[m + i as usize];
                grow = add_room(grow, nz + rnz1, stretch, pad);
            }
        }
        windex.swap(cbeg, where_);
        wvalue.swap(cbeg, where_);
        let pivot = wvalue[cbeg];
        let mut where_ = usize::MAX;
        for rpos in rbeg..rend {
            let j = windex[rpos];
            if j == pivot_col {
                where_ = rpos;
            } else {
                let nz = wend[j as usize] - wbegin[j as usize];
                grow = add_room(grow, nz + cnz1, stretch, pad);
            }
        }
        windex.swap(rbeg, where_);
        let mut room = wend[2 * m] - wbegin[2 * m];
        if grow > room {
            file_compress(
                2 * m as Int,
                wbegin,
                wend,
                self.wflink,
                windex,
                wvalue,
                stretch,
                pad,
            );
            cbeg = wbegin[pivot_col as usize] as usize;
            cend = wend[pivot_col as usize] as usize;
            rbeg = wbegin[m + pivot_row as usize] as usize;
            rend = wend[m + pivot_row as usize] as usize;
            room = wend[2 * m] - wbegin[2 * m];
            self.ngarbage += 1;
        }
        if grow > room {
            self.addmem_w = grow - room;
            return Err(REALLOCATE);
        }
        Ok((cbeg, cend, rbeg, rend, pivot))
    }

    /// Column file update shared by pivot_any and pivot_small. For each
    /// column j of the pivot row: compress unmodified entries, gather the
    /// entries to be updated in work0, move the pivot row entry to the
    /// front, update and append. With `small`, entries <= droptol are
    /// dropped and recorded per column as bit masks in row_pivot (as int64,
    /// like the C code). Returns the next position in U.
    fn pivot_update_cols(
        &mut self,
        cbeg: usize,
        cend: usize,
        rbeg: usize,
        rend: usize,
        pivot: f64,
        small: bool,
    ) -> Int {
        let m = self.m;
        let mu = m as usize;
        let droptol = self.droptol;
        let pad = self.pad;
        let stretch = self.stretch;
        let pivot_row = self.pivot_row;
        let colcount_flink = &mut *self.colcount_flink;
        let colcount_blink = &mut *self.colcount_blink;
        let colmax = &mut *self.col_pivot;
        let wbegin = &mut *self.wbegin;
        let wend = &mut *self.wend;
        let wflink = &mut *self.wflink;
        let wblink = &mut *self.wblink;
        let uindex = &mut *self.uindex;
        let uvalue = &mut *self.uvalue;
        let windex = &mut *self.windex;
        let wvalue = &mut *self.wvalue;
        let marked = &mut *self.iwork0;
        let work = &mut *self.work0;
        let cancelled = &mut *self.row_pivot;
        let cnz1 = cend - cbeg - 1;

        // get pointer to U
        let mut uput = self.ubegin[self.rank as usize];

        // For each row i to be updated set marked[i] > 0 to its position in
        // the (packed) pivot column.
        let mut position = 1;
        for pos in cbeg + 1..cend {
            marked[windex[pos] as usize] = position;
            position += 1;
        }

        // wi, wx: the pivot column (from cbeg)
        for (col_number, rpos) in (rbeg + 1..rend).enumerate() {
            let j = windex[rpos] as usize;
            let mut cmx = 0.0; // column maximum

            // Compress unmodified column entries. Store entries to be updated
            // in workspace. Move pivot row entry to the front of column.
            let mut where_ = usize::MAX;
            let pos1 = wbegin[j] as usize;
            let mut put = pos1;
            for pos in pos1..wend[j] as usize {
                let i = windex[pos];
                let position = marked[i as usize];
                if position > 0 {
                    work[position as usize] = wvalue[pos];
                } else {
                    if i == pivot_row {
                        where_ = put;
                    } else {
                        let x = wvalue[pos].abs();
                        if x > cmx {
                            cmx = x;
                        }
                    }
                    windex[put] = windex[pos];
                    wvalue[put] = wvalue[pos];
                    put += 1;
                }
            }
            wend[j] = put as Int;
            windex.swap(pos1, where_);
            wvalue.swap(pos1, where_);
            let xrj = wvalue[pos1]; // pivot row entry

            // Reappend column if no room for update.
            let room = wbegin[wflink[j] as usize] - put as Int;
            if room < cnz1 as Int {
                let nz = wend[j] - wbegin[j];
                let room = room_for(cnz1 as Int, nz + cnz1 as Int, stretch, pad);
                file_reappend(
                    j as Int,
                    2 * m,
                    wbegin,
                    wend,
                    wflink,
                    wblink,
                    windex,
                    wvalue,
                    room,
                );
                put = wend[j] as usize;
                self.nexpand += 1;
            }

            // Compute update in workspace and append to column.
            let a = xrj / pivot;
            for pos in 1..=cnz1 {
                work[pos] = (-a).mul_add_c(wvalue[cbeg + pos], work[pos]);
            }
            if small {
                let mut mask: u64 = 0;
                for pos in 1..=cnz1 {
                    let x = work[pos].abs();
                    if x > droptol {
                        windex[put] = windex[cbeg + pos];
                        wvalue[put] = work[pos];
                        put += 1;
                        if x > cmx {
                            cmx = x;
                        }
                    } else {
                        // cancellation in row wi[pos]
                        mask |= 1u64 << (pos - 1);
                    }
                    work[pos] = 0.0;
                }
                cancelled[col_number] = f64::from_bits(mask);
            } else {
                for pos in 1..=cnz1 {
                    windex[put] = windex[cbeg + pos];
                    wvalue[put] = work[pos];
                    put += 1;
                    let x = work[pos].abs();
                    if x > cmx {
                        cmx = x;
                    }
                    work[pos] = 0.0;
                }
            }
            wend[j] = put as Int;

            // Write pivot row entry to U and remove from file.
            if xrj.abs() > droptol {
                uindex[uput as usize] = j as Int;
                uvalue[uput as usize] = xrj;
                uput += 1;
            }
            debug_assert!(windex[wbegin[j] as usize] == pivot_row);
            wbegin[j] += 1;

            // Move column to new list and update min_colnz.
            let nz = wend[j] - wbegin[j];
            list_move(
                j as Int,
                nz,
                colcount_flink,
                colcount_blink,
                m,
                Some(&mut self.min_colnz),
            );

            colmax[j] = cmx;
        }
        for pos in cbeg + 1..cend {
            marked[windex[pos] as usize] = 0;
        }
        let _ = mu;
        uput
    }

    /// Row file update shared by pivot_any and pivot_small: for each row i
    /// of the pivot column remove the overlap with the pivot row (including
    /// the pivot column entry) and append the pattern of the pivot row
    /// (without the entries cancelled in the column update when `small`).
    fn pivot_update_rows(
        &mut self,
        cbeg: usize,
        cend: usize,
        rbeg: usize,
        rend: usize,
        small: bool,
    ) {
        let m = self.m;
        let mu = m as usize;
        let pad = self.pad;
        let stretch = self.stretch;
        let pivot_col = self.pivot_col;
        let rowcount_flink = &mut *self.rowcount_flink;
        let rowcount_blink = &mut *self.rowcount_blink;
        let wbegin = &mut *self.wbegin;
        let wend = &mut *self.wend;
        let wflink = &mut *self.wflink;
        let wblink = &mut *self.wblink;
        let windex = &mut *self.windex;
        let wvalue = &mut *self.wvalue;
        let marked = &mut *self.iwork0;
        let cancelled = &*self.row_pivot;
        let rnz1 = (rend - rbeg - 1) as Int;

        for rpos in rbeg..rend {
            marked[windex[rpos] as usize] = 1;
        }

        let mut mask: u64 = 1;
        for pos in cbeg + 1..cend {
            let i = windex[pos] as usize;

            // Compress unmodified row entries (not marked). Remove overlap
            // with pivot row, including pivot column entry.
            let mut found = false;
            let mut put = wbegin[mu + i] as usize;
            for rpos in wbegin[mu + i] as usize..wend[mu + i] as usize {
                let j = windex[rpos];
                if j == pivot_col {
                    found = true;
                }
                if marked[j as usize] == 0 {
                    windex[put] = j;
                    put += 1;
                }
            }
            debug_assert!(found);
            wend[mu + i] = put as Int;

            // Reappend row if no room for update. Append pattern of pivot row.
            let room = wbegin[wflink[mu + i] as usize] - put as Int;
            if room < rnz1 {
                let nz = wend[mu + i] - wbegin[mu + i];
                let room = room_for(rnz1, nz + rnz1, stretch, pad);
                file_reappend(
                    m + i as Int,
                    2 * m,
                    wbegin,
                    wend,
                    wflink,
                    wblink,
                    windex,
                    wvalue,
                    room,
                );
                put = wend[mu + i] as usize;
                self.nexpand += 1;
            }
            if small {
                for (col_number, rpos) in (rbeg + 1..rend).enumerate() {
                    if cancelled[col_number].to_bits() & mask == 0 {
                        windex[put] = windex[rpos];
                        put += 1;
                    }
                }
            } else {
                for rpos in rbeg + 1..rend {
                    windex[put] = windex[rpos];
                    put += 1;
                }
            }
            wend[mu + i] = put as Int;

            // Move to new list. The row must be reinserted even if nz are
            // unchanged since it might have been taken out in Markowitz
            // search.
            let nz = wend[mu + i] - wbegin[mu + i];
            list_move(
                i as Int,
                nz,
                rowcount_flink,
                rowcount_blink,
                m,
                Some(&mut self.min_rownz),
            );
            mask <<= 1;
        }
        for rpos in rbeg..rend {
            marked[windex[rpos] as usize] = 0;
        }
    }

    /// Store the pivot column in L and clean up (shared by pivot_any,
    /// pivot_small): store the pivot element; remove pivot column from the
    /// column file, pivot row from the row file, and both from the counts.
    fn pivot_finish(&mut self, cbeg: usize, cend: usize, rbeg: usize, pivot: f64, uput: Int) {
        let mu = self.m as usize;
        let rank = self.rank as usize;
        let droptol = self.droptol;
        let windex = &*self.windex;
        let wvalue = &*self.wvalue;
        let lindex = &mut *self.lindex;
        let lvalue = &mut *self.lvalue;

        // Store column in L.
        let mut put = self.lbegin_p[rank] as usize;
        for pos in cbeg + 1..cend {
            let x = wvalue[pos] / pivot;
            if x.abs() > droptol {
                lindex[put] = windex[pos];
                lvalue[put] = x;
                put += 1;
            }
        }
        lindex[put] = -1; // terminate column
        put += 1;
        self.lbegin_p[rank + 1] = put as Int;
        self.ubegin[rank + 1] = uput;

        let pivot_col = self.pivot_col;
        let pivot_row = self.pivot_row;
        self.col_pivot[pivot_col as usize] = pivot;
        self.wend[pivot_col as usize] = cbeg as Int;
        self.wend[mu + pivot_row as usize] = rbeg as Int;
        list_remove(self.colcount_flink, self.colcount_blink, pivot_col);
        list_remove(self.rowcount_flink, self.rowcount_blink, pivot_row);
    }

    fn pivot_any(&mut self) -> Int {
        let (cbeg, cend, rbeg, rend, pivot) = match self.pivot_prepare() {
            Ok(x) => x,
            Err(status) => return status,
        };
        let uput = self.pivot_update_cols(cbeg, cend, rbeg, rend, pivot, false);
        self.pivot_update_rows(cbeg, cend, rbeg, rend, false);
        self.pivot_finish(cbeg, cend, rbeg, pivot, uput);
        OK
    }

    fn pivot_small(&mut self) -> Int {
        let (cbeg, cend, rbeg, rend, pivot) = match self.pivot_prepare() {
            Ok(x) => x,
            Err(status) => return status,
        };
        debug_assert!((cend - cbeg - 1) as Int <= MAXROW_SMALL);
        let uput = self.pivot_update_cols(cbeg, cend, rbeg, rend, pivot, true);
        self.pivot_update_rows(cbeg, cend, rbeg, rend, true);
        self.pivot_finish(cbeg, cend, rbeg, pivot, uput);
        OK
    }

    fn pivot_singleton_row(&mut self) -> Int {
        let m = self.m;
        let mu = m as usize;
        let rank = self.rank as usize;
        let droptol = self.droptol;
        let pivot_col = self.pivot_col;
        let pivot_row = self.pivot_row;
        let wbegin = &mut *self.wbegin;
        let wend = &mut *self.wend;
        let windex = &mut *self.windex;
        let wvalue = &*self.wvalue;
        let lindex = &mut *self.lindex;
        let lvalue = &mut *self.lvalue;

        let cbeg = wbegin[pivot_col as usize] as usize;
        let cend = wend[pivot_col as usize] as usize;
        let rbeg = wbegin[mu + pivot_row as usize];
        debug_assert!(wend[mu + pivot_row as usize] - rbeg == 1);

        // Find pivot.
        let mut where_ = cbeg;
        while windex[where_] != pivot_row {
            where_ += 1;
        }
        let pivot = wvalue[where_];

        // Store column in L.
        let mut put = self.lbegin_p[rank] as usize;
        for pos in cbeg..cend {
            let x = wvalue[pos] / pivot;
            if pos != where_ && x.abs() > droptol {
                lindex[put] = windex[pos];
                lvalue[put] = x;
                put += 1;
            }
        }
        lindex[put] = -1; // terminate column
        put += 1;
        self.lbegin_p[rank + 1] = put as Int;
        self.ubegin[rank + 1] = self.ubegin[rank];

        // Remove pivot column from row file. Update row lists.
        for pos in cbeg..cend {
            let i = windex[pos];
            if i == pivot_row {
                continue;
            }
            let iu = i as usize;
            let mut where_ = wbegin[mu + iu] as usize;
            while windex[where_] != pivot_col {
                where_ += 1;
            }
            wend[mu + iu] -= 1;
            windex[where_] = windex[wend[mu + iu] as usize];
            let nz = wend[mu + iu] - wbegin[mu + iu];
            list_move(
                i,
                nz,
                self.rowcount_flink,
                self.rowcount_blink,
                m,
                Some(&mut self.min_rownz),
            );
        }

        // Cleanup: store pivot element; remove pivot column from column
        // file, pivot row from row file; remove both from counts.
        self.col_pivot[pivot_col as usize] = pivot;
        wend[pivot_col as usize] = cbeg as Int;
        wend[mu + pivot_row as usize] = rbeg;
        list_remove(self.colcount_flink, self.colcount_blink, pivot_col);
        list_remove(self.rowcount_flink, self.rowcount_blink, pivot_row);
        OK
    }

    fn pivot_singleton_col(&mut self) -> Int {
        let m = self.m;
        let mu = m as usize;
        let rank = self.rank as usize;
        let droptol = self.droptol;
        let pivot_col = self.pivot_col;
        let pivot_row = self.pivot_row;
        let colmax = &mut *self.col_pivot;
        let wbegin = &*self.wbegin;
        let wend = &mut *self.wend;
        let uindex = &mut *self.uindex;
        let uvalue = &mut *self.uvalue;
        let windex = &mut *self.windex;
        let wvalue = &mut *self.wvalue;

        let cbeg = wbegin[pivot_col as usize];
        debug_assert!(wend[pivot_col as usize] - cbeg == 1);
        let rbeg = wbegin[mu + pivot_row as usize];
        let rend = wend[mu + pivot_row as usize];

        // Remove pivot row from column file and store in U. Update column
        // lists.
        let mut put = self.ubegin[rank] as usize;
        let pivot = wvalue[cbeg as usize];
        let mut xrj = 0.0;
        for rpos in rbeg as usize..rend as usize {
            let j = windex[rpos];
            if j == pivot_col {
                continue;
            }
            let ju = j as usize;
            let mut where_ = usize::MAX;
            let mut cmx = 0.0; // column maximum
            for pos in wbegin[ju] as usize..wend[ju] as usize {
                if windex[pos] == pivot_row {
                    where_ = pos;
                    xrj = wvalue[pos];
                } else {
                    let x = wvalue[pos].abs();
                    if x > cmx {
                        cmx = x;
                    }
                }
            }
            if xrj.abs() > droptol {
                uindex[put] = j;
                uvalue[put] = xrj;
                put += 1;
            }
            wend[ju] -= 1;
            let end = wend[ju] as usize;
            windex[where_] = windex[end];
            wvalue[where_] = wvalue[end];
            let nz = wend[ju] - wbegin[ju];
            list_move(
                j,
                nz,
                self.colcount_flink,
                self.colcount_blink,
                m,
                Some(&mut self.min_colnz),
            );
            colmax[ju] = cmx;
        }
        self.ubegin[rank + 1] = put as Int;

        // Store empty column in L.
        let put = self.lbegin_p[rank];
        self.lindex[put as usize] = -1; // terminate column
        self.lbegin_p[rank + 1] = put + 1;

        // Cleanup
        colmax[pivot_col as usize] = pivot;
        wend[pivot_col as usize] = cbeg;
        wend[mu + pivot_row as usize] = rbeg;
        list_remove(self.colcount_flink, self.colcount_blink, pivot_col);
        list_remove(self.rowcount_flink, self.rowcount_blink, pivot_row);
        OK
    }

    fn pivot_doubleton_col(&mut self) -> Int {
        let m = self.m;
        let mu = m as usize;
        let rank = self.rank as usize;
        let droptol = self.droptol;
        let pad = self.pad;
        let stretch = self.stretch;
        let pivot_col = self.pivot_col;
        let pivot_row = self.pivot_row;
        let colcount_flink = &mut *self.colcount_flink;
        let colcount_blink = &mut *self.colcount_blink;
        let colmax = &mut *self.col_pivot;
        let wbegin = &mut *self.wbegin;
        let wend = &mut *self.wend;
        let wflink = &mut *self.wflink;
        let wblink = &mut *self.wblink;
        let uindex = &mut *self.uindex;
        let uvalue = &mut *self.uvalue;
        let windex = &mut *self.windex;
        let wvalue = &mut *self.wvalue;
        let marked = &mut *self.iwork0;

        let mut cbeg = wbegin[pivot_col as usize] as usize; // changed by file compression
        let mut rbeg = wbegin[mu + pivot_row as usize] as usize;
        let mut rend = wend[mu + pivot_row as usize] as usize;
        let rnz1 = (rend - rbeg - 1) as Int; // nz in pivot row except pivot
        debug_assert!(wend[pivot_col as usize] as usize - cbeg - 1 == 1);

        // Move pivot element to front of pivot column and pivot row.
        if windex[cbeg] != pivot_row {
            windex.swap(cbeg, cbeg + 1);
            wvalue.swap(cbeg, cbeg + 1);
        }
        let pivot = wvalue[cbeg];
        let other_row = windex[cbeg + 1];
        let other_value = wvalue[cbeg + 1];
        let orow = mu + other_row as usize;
        let mut where_ = rbeg;
        while windex[where_] != pivot_col {
            where_ += 1;
        }
        windex.swap(rbeg, where_);

        // Check if room is available in W. Columns can be updated in place
        // but the updated row may need to be expanded.
        let nz = wend[orow] - wbegin[orow];
        let grow = add_room(0, nz + rnz1, stretch, pad);
        let mut room = wend[2 * mu] - wbegin[2 * mu];
        if grow > room {
            file_compress(2 * m, wbegin, wend, wflink, windex, wvalue, stretch, pad);
            cbeg = wbegin[pivot_col as usize] as usize;
            rbeg = wbegin[mu + pivot_row as usize] as usize;
            rend = wend[mu + pivot_row as usize] as usize;
            room = wend[2 * mu] - wbegin[2 * mu];
            self.ngarbage += 1;
        }
        if grow > room {
            self.addmem_w = grow - room;
            return REALLOCATE;
        }

        // Column file update

        let mut uput = self.ubegin[rank] as usize;
        let mut put = rbeg + 1;
        let mut ncancelled = 0;
        for rpos in rbeg + 1..rend {
            let j = windex[rpos];
            let ju = j as usize;
            let mut cmx = 0.0; // column maximum

            // Find position of pivot row entry and possibly other row entry
            // in column j.
            let mut where_pivot = usize::MAX;
            let mut where_other = usize::MAX;
            let end = wend[ju] as usize;
            for pos in wbegin[ju] as usize..end {
                if windex[pos] == pivot_row {
                    where_pivot = pos;
                } else if windex[pos] == other_row {
                    where_other = pos;
                } else {
                    let x = wvalue[pos].abs();
                    if x > cmx {
                        cmx = x;
                    }
                }
            }
            let xrj = wvalue[where_pivot];

            // Store pivot row entry in U.
            if wvalue[where_pivot].abs() > droptol {
                uindex[uput] = j;
                uvalue[uput] = wvalue[where_pivot];
                uput += 1;
            }

            if where_other == usize::MAX {
                // Compute fill-in element.
                let x = -xrj * (other_value / pivot);
                let xabs = x.abs();
                if xabs > droptol {
                    // Store fill-in where pivot row entry was.
                    windex[where_pivot] = other_row;
                    wvalue[where_pivot] = x;
                    windex[put] = j;
                    put += 1;
                    if xabs > cmx {
                        cmx = xabs;
                    }
                } else {
                    // Remove pivot row entry.
                    wend[ju] -= 1;
                    let end = wend[ju] as usize;
                    windex[where_pivot] = windex[end];
                    wvalue[where_pivot] = wvalue[end];

                    // Decrease column count.
                    let nz = end as Int - wbegin[ju];
                    list_move(
                        j,
                        nz,
                        colcount_flink,
                        colcount_blink,
                        m,
                        Some(&mut self.min_colnz),
                    );
                }
            } else {
                // Remove pivot row entry and update other row entry.
                wend[ju] -= 1;
                let end = wend[ju] as usize;
                windex[where_pivot] = windex[end];
                wvalue[where_pivot] = wvalue[end];
                if where_other == end {
                    where_other = where_pivot;
                }
                wvalue[where_other] = (-xrj).mul_add_c(other_value / pivot, wvalue[where_other]);

                // If we have numerical cancellation, then remove the entry and
                // mark the column.
                let x = wvalue[where_other].abs();
                if x <= droptol {
                    wend[ju] -= 1;
                    let end = wend[ju] as usize;
                    windex[where_other] = windex[end];
                    wvalue[where_other] = wvalue[end];
                    marked[ju] = 1;
                    ncancelled += 1;
                } else if x > cmx {
                    cmx = x;
                }

                // Decrease column count.
                let nz = wend[ju] - wbegin[ju];
                list_move(
                    j,
                    nz,
                    colcount_flink,
                    colcount_blink,
                    m,
                    Some(&mut self.min_colnz),
                );
            }
            colmax[ju] = cmx;
        }
        rend = put;
        self.ubegin[rank + 1] = uput as Int;

        // Row file update

        // If we have numerical cancellation, then we have to remove these
        // entries (marked) from the row pattern. In any case remove pivot
        // column entry.
        if ncancelled > 0 {
            marked[pivot_col as usize] = 1; // treat as cancelled
            let mut put = wbegin[orow] as usize; // compress remaining entries
            let end = wend[orow] as usize;
            for pos in put..end {
                let j = windex[pos] as usize;
                if marked[j] != 0 {
                    marked[j] = 0;
                } else {
                    windex[put] = j as Int;
                    put += 1;
                }
            }
            debug_assert!(end - put == ncancelled + 1);
            wend[orow] = put as Int;
        } else {
            let mut where_ = wbegin[orow] as usize;
            while windex[where_] != pivot_col {
                where_ += 1;
            }
            wend[orow] -= 1;
            windex[where_] = windex[wend[orow] as usize];
        }

        // Reappend row if no room for update.
        let nfill = (rend - (rbeg + 1)) as Int;
        let room = wbegin[wflink[orow] as usize] - wend[orow];
        if nfill > room {
            let nz = wend[orow] - wbegin[orow];
            let space = room_for(nfill, nz + nfill, stretch, pad);
            file_reappend(
                orow as Int,
                2 * m,
                wbegin,
                wend,
                wflink,
                wblink,
                windex,
                wvalue,
                space,
            );
            self.nexpand += 1;
        }

        // Append fill-in to row pattern.
        let mut put = wend[orow] as usize;
        for pos in rbeg + 1..rend {
            windex[put] = windex[pos];
            put += 1;
        }
        wend[orow] = put as Int;

        // Reinsert other row into row counts.
        let nz = wend[orow] - wbegin[orow];
        list_move(
            other_row,
            nz,
            self.rowcount_flink,
            self.rowcount_blink,
            m,
            Some(&mut self.min_rownz),
        );

        // Store column in L.
        let mut put = self.lbegin_p[rank] as usize;
        let x = other_value / pivot;
        if x.abs() > droptol {
            self.lindex[put] = other_row;
            self.lvalue[put] = x;
            put += 1;
        }
        self.lindex[put] = -1; // terminate column
        put += 1;
        self.lbegin_p[rank + 1] = put as Int;

        // Cleanup
        colmax[pivot_col as usize] = pivot;
        wend[pivot_col as usize] = cbeg as Int;
        wend[mu + pivot_row as usize] = rbeg as Int;
        list_remove(colcount_flink, colcount_blink, pivot_col);
        list_remove(self.rowcount_flink, self.rowcount_blink, pivot_row);
        OK
    }

    /// lu_remove_col: remove column j from the active submatrix
    fn remove_col(&mut self, j: Int) {
        let m = self.m;
        let mu = m as usize;
        let ju = j as usize;
        let wbegin = &*self.wbegin;
        let wend = &mut *self.wend;
        let windex = &mut *self.windex;
        let cbeg = wbegin[ju];
        let cend = wend[ju];

        // Remove column j from row file.
        for pos in cbeg as usize..cend as usize {
            let i = windex[pos];
            let iu = mu + i as usize;
            let mut where_ = wbegin[iu] as usize;
            while windex[where_] != j {
                where_ += 1;
            }
            wend[iu] -= 1;
            windex[where_] = windex[wend[iu] as usize];
            let nz = wend[iu] - wbegin[iu];
            list_move(
                i,
                nz,
                self.rowcount_flink,
                self.rowcount_blink,
                m,
                Some(&mut self.min_rownz),
            );
        }

        // Remove column j from column file.
        self.col_pivot[ju] = 0.0;
        wend[ju] = cbeg;
        list_move(
            j,
            0,
            self.colcount_flink,
            self.colcount_blink,
            m,
            Some(&mut self.min_colnz),
        );
    }
}
