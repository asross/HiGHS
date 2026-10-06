//! Solves with the factorization: lu_solve_dense.c, lu_solve_sparse.c,
//! lu_solve_for_update.c and their kernels lu_solve_symbolic.c,
//! lu_solve_triangular.c, lu_dfs.c, plus lu_garbage_perm.c.

use crate::util::fma::ClangFma;

use super::{as_int_mut, Int, Lu, OK, REALLOCATE};

/// lu_dfs: compute reach(i) in a graph by depth first search (adapted from
/// T. Davis, CSPARSE).
///
/// begin, end, index define the graph: node j has neighbours
/// index[begin[j]..end[j]-1], or, when `end` is None, index[begin[j]..] up
/// to a negative index. On return xi[newtop..top-1] hold reach(i) in
/// topological order (newtop is returned); nodes already marked are
/// excluded. xi[0..] is also used as the dfs stack. pstack is size m
/// workspace. Node j is marked iff marked[j] == mk; on return the nodes in
/// the reach are marked. If node i is marked on entry, nothing is done.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dfs(
    i: Int,
    begin: &[Int],
    end: Option<&[Int]>,
    index: &[Int],
    mut top: Int,
    xi: &mut [Int],
    pstack: &mut [Int],
    marked: &mut [Int],
    mk: Int,
) -> Int {
    if marked[i as usize] == mk {
        return top;
    }
    let mut head: isize = 0;
    xi[0] = i;
    while head >= 0 {
        let h = head as usize;
        let i = xi[h];
        let iu = i as usize;
        if marked[iu] != mk {
            // node i has not been visited
            marked[iu] = mk;
            pstack[h] = begin[iu];
        }
        let mut done = true;
        // continue dfs at node i
        let mut p = pstack[h] as usize;
        match end {
            Some(end) => {
                let pend = end[iu] as usize;
                while p < pend {
                    let inext = index[p];
                    if marked[inext as usize] != mk {
                        pstack[h] = p as Int + 1;
                        head += 1;
                        xi[head as usize] = inext; // start dfs at node inext
                        done = false;
                        break;
                    }
                    p += 1; // skip visited node
                }
            }
            None => loop {
                let inext = index[p];
                if inext < 0 {
                    break;
                }
                if marked[inext as usize] != mk {
                    pstack[h] = p as Int + 1;
                    head += 1;
                    xi[head as usize] = inext; // start dfs at node inext
                    done = false;
                    break;
                }
                p += 1; // skip visited node
            },
        }
        if done {
            // node i has no unvisited neighbours
            head -= 1;
            top -= 1;
            xi[top as usize] = i;
        }
    }
    top
}

/// lu_solve_symbolic: the pattern of the right-hand side is irhs; the
/// pattern of the solution is returned in ilhs[top..m-1] in topological
/// order, top is returned (J. Gilbert and T. Peierls, "Sparse partial
/// pivoting in time proportional to arithmetic operations", 1988). Matrix
/// columns as for dfs(). marked[i] != mk on entry.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_symbolic(
    m: Int,
    begin: &[Int],
    end: Option<&[Int]>,
    index: &[Int],
    irhs: &[Int],
    ilhs: &mut [Int],
    pstack: &mut [Int],
    marked: &mut [Int],
    mk: Int,
) -> Int {
    let mut top = m;
    for &i in irhs {
        if marked[i as usize] != mk {
            top = dfs(i, begin, end, index, top, ilhs, pstack, marked, mk);
        }
    }
    top
}

/// lu_solve_triangular: substitution with a triangular matrix.
///
/// The symbolic pattern of the solution must be given in topological order
/// in pattern_symb. On return pattern[0..nz-1] holds the pattern of the
/// solution after dropping numerical zeros; nz is returned. Entries <=
/// droptol are set to zero (none when droptol <= 0); the pattern never
/// includes exact zeros. The pivots are stored separately (None: unit
/// pivots); columns as for dfs().
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_triangular(
    pattern_symb: &[Int],
    begin: &[Int],
    end: Option<&[Int]>,
    index: &[Int],
    value: &[f64],
    pivot: Option<&[f64]>,
    droptol: f64,
    lhs: &mut [f64],
    pattern: &mut [Int],
    flops: &mut Int,
) -> Int {
    let mut nz = 0usize;
    let mut flop_count: Int = 0;

    // The four variants are separate loops, as in the C code.
    macro_rules! column {
        ($ipivot:expr, $x:expr, $end:expr) => {
            match $end {
                Some(end) => {
                    for pos in begin[$ipivot] as usize..end[$ipivot] as usize {
                        let i = index[pos] as usize;
                        lhs[i] = (-$x).mul_add_c(value[pos], lhs[i]);
                        flop_count += 1;
                    }
                }
                None => {
                    let mut pos = begin[$ipivot] as usize;
                    loop {
                        let i = index[pos];
                        if i < 0 {
                            break;
                        }
                        let i = i as usize;
                        lhs[i] = (-$x).mul_add_c(value[pos], lhs[i]);
                        flop_count += 1;
                        pos += 1;
                    }
                }
            }
        };
    }
    macro_rules! solve {
        ($end:expr) => {
            for &ip in pattern_symb {
                let ipivot = ip as usize;
                if lhs[ipivot] != 0.0 {
                    let x = match pivot {
                        Some(pivot) => {
                            lhs[ipivot] /= pivot[ipivot];
                            flop_count += 1;
                            lhs[ipivot]
                        }
                        None => lhs[ipivot],
                    };
                    column!(ipivot, x, $end);
                    if x.abs() > droptol {
                        pattern[nz] = ip;
                        nz += 1;
                    } else {
                        lhs[ipivot] = 0.0;
                    }
                }
            }
        };
    }
    match end {
        Some(end) => solve!(Some(end)),
        None => solve!(None::<&[Int]>),
    }

    *flops = flops.wrapping_add(flop_count);
    nz as Int
}

fn is_trans(trans: u8) -> bool {
    trans == b't' || trans == b'T'
}

impl Lu<'_> {
    /// lu_garbage_perm: pivotcol[0..pivotlen-1], pivotrow[0..pivotlen-1]
    /// (pivotlen >= m) may contain duplicates, the last occurrence of each
    /// index being its position in the pivot sequence. Remove duplicates and
    /// compress such that pivotlen == m.
    pub(crate) fn garbage_perm(&mut self) {
        let m = self.m as usize;
        let pivotlen = self.pivotlen as usize;
        let pivotcol = &mut *self.colcount_flink;
        let pivotrow = &mut *self.colcount_blink;
        let marked = &mut *self.iwork0;

        if pivotlen > m {
            self.marker += 1;
            let marker = self.marker;
            let mut put = pivotlen;
            for get in (0..pivotlen).rev() {
                let j = pivotcol[get];
                if marked[j as usize] != marker {
                    marked[j as usize] = marker;
                    put -= 1;
                    pivotcol[put] = j;
                    pivotrow[put] = pivotrow[get];
                }
            }
            debug_assert!(put + m == pivotlen);
            pivotcol.copy_within(put..put + m, 0);
            pivotrow.copy_within(put..put + m, 0);
            self.pivotlen = m as Int;
        }
    }

    /// lu_solve_dense; `rhs` None means the right-hand side is in `lhs`
    /// (the C interface allows rhs == lhs)
    pub(crate) fn solve_dense(&mut self, rhs: Option<&[f64]>, lhs: &mut [f64], trans: u8) {
        self.garbage_perm();
        let m = self.m as usize;
        let nforrest = self.nforrest as usize;
        let p = &self.wblink[m + 1..];
        let (rbegin, eta_row) = self.rowcount_flink.split_at(m + 1);
        let pivotcol = &*self.colcount_flink;
        let pivotrow = &*self.colcount_blink;
        let lbegin_p = &*self.lbegin_p;
        let ltbegin_p = &self.wflink[m + 1..];
        let ubegin = &*self.ubegin;
        let wbegin = &*self.wbegin;
        let wend = &*self.wend;
        let col_pivot = &*self.col_pivot;
        let row_pivot = &*self.row_pivot;
        let lindex = &*self.lindex;
        let lvalue = &*self.lvalue;
        let uindex = &*self.uindex;
        let uvalue = &*self.uvalue;
        let windex = &*self.windex;
        let wvalue = &*self.wvalue;
        let work1 = &mut *self.work1;

        work1.copy_from_slice(&rhs.unwrap_or(lhs)[..m]);
        if is_trans(trans) {
            // Solve transposed system

            // Solve with U'.
            for k in 0..m {
                let jpivot = pivotcol[k] as usize;
                let ipivot = pivotrow[k] as usize;
                let x = work1[jpivot] / col_pivot[jpivot];
                for pos in wbegin[jpivot] as usize..wend[jpivot] as usize {
                    let i = windex[pos] as usize;
                    work1[i] = (-x).mul_add_c(wvalue[pos], work1[i]);
                }
                lhs[ipivot] = x;
            }

            // Solve with update ETAs backwards.
            for t in (0..nforrest).rev() {
                let x = lhs[eta_row[t] as usize];
                for pos in rbegin[t] as usize..rbegin[t + 1] as usize {
                    let i = lindex[pos] as usize;
                    lhs[i] = (-x).mul_add_c(lvalue[pos], lhs[i]);
                }
            }

            // Solve with L'.
            for k in (0..m).rev() {
                let mut x: f64 = 0.0;
                let mut pos = lbegin_p[k] as usize;
                loop {
                    let i = lindex[pos];
                    if i < 0 {
                        break;
                    }
                    x = lhs[i as usize].mul_add_c(lvalue[pos], x);
                    pos += 1;
                }
                lhs[p[k] as usize] -= x;
            }
        } else {
            // Solve forward system

            // Solve with L.
            for k in 0..m {
                let mut x: f64 = 0.0;
                let mut pos = ltbegin_p[k] as usize;
                loop {
                    let i = lindex[pos];
                    if i < 0 {
                        break;
                    }
                    x = work1[i as usize].mul_add_c(lvalue[pos], x);
                    pos += 1;
                }
                work1[p[k] as usize] -= x;
            }

            // Solve with update ETAs.
            let mut pos = rbegin[0] as usize;
            for t in 0..nforrest {
                let ipivot = eta_row[t] as usize;
                let mut x: f64 = 0.0;
                while pos < rbegin[t + 1] as usize {
                    x = work1[lindex[pos] as usize].mul_add_c(lvalue[pos], x);
                    pos += 1;
                }
                work1[ipivot] -= x;
            }

            // Solve with U.
            for k in (0..m).rev() {
                let jpivot = pivotcol[k] as usize;
                let ipivot = pivotrow[k] as usize;
                let x = work1[ipivot] / row_pivot[ipivot];
                let mut pos = ubegin[ipivot] as usize;
                loop {
                    let i = uindex[pos];
                    if i < 0 {
                        break;
                    }
                    work1[i as usize] = (-x).mul_add_c(uvalue[pos], work1[i as usize]);
                    pos += 1;
                }
                lhs[jpivot] = x;
            }
        }
    }

    /// lu_solve_sparse: solve with sparse right-hand side irhs, xrhs. The
    /// solution is scattered into xlhs (zero on entry), its pattern returned
    /// in ilhs[0..*p_nlhs-1].
    pub(crate) fn solve_sparse(
        &mut self,
        irhs: &[Int],
        xrhs: &[f64],
        p_nlhs: &mut Int,
        ilhs: &mut [Int],
        xlhs: &mut [f64],
        trans: u8,
    ) {
        let m = self.m;
        let mu = m as usize;
        let nforrest = self.nforrest as usize;
        let pivotlen = self.pivotlen as usize;
        let nz_sparse = (self.sparse_thres * m as f64) as Int;
        let droptol = self.droptol;
        let p = &self.wblink[mu + 1..];
        let pmap = &*self.pinv;
        let qmap = &*self.qinv;
        let (rbegin, eta_row) = self.rowcount_flink.split_at(mu + 1);
        let pivotcol = &*self.colcount_flink;
        let pivotrow = &*self.colcount_blink;
        let (wbegin, lbegin) = self.wbegin.split_at(mu + 1);
        let (wend, ltbegin) = self.wend.split_at(mu + 1);
        let ltbegin_p = &self.wflink[mu + 1..];
        let ubegin = &*self.ubegin;
        let col_pivot = &*self.col_pivot;
        let row_pivot = &*self.row_pivot;
        let lindex = &*self.lindex;
        let lvalue = &*self.lvalue;
        let uindex = &*self.uindex;
        let uvalue = &*self.uvalue;
        let windex = &*self.windex;
        let wvalue = &*self.wvalue;
        let marked = &mut *self.iwork0;
        let marker = &mut self.marker;
        let (pattern_symb, pattern) = self.rowcount_blink.split_at_mut(mu);
        let work = &mut *self.work0;
        let pstack = as_int_mut(self.work1);

        let mut lflops: Int = 0;
        let mut uflops: Int = 0;
        let mut rflops: Int = 0;

        if is_trans(trans) {
            // Sparse triangular solve with U'. Solution scattered into work,
            // indices in pattern[0..nz-1].
            *marker += 1;
            let mk = *marker;
            let top = solve_symbolic(
                m,
                wbegin,
                Some(wend),
                windex,
                irhs,
                pattern_symb,
                pstack,
                marked,
                mk,
            );
            for (&i, &x) in irhs.iter().zip(xrhs) {
                work[i as usize] = x;
            }
            let mut nz = solve_triangular(
                &pattern_symb[top as usize..],
                wbegin,
                Some(wend),
                windex,
                wvalue,
                Some(col_pivot),
                droptol,
                work,
                pattern,
                &mut uflops,
            ) as usize;

            // Permute solution into xlhs. Map pattern from column indices to
            // row indices.
            *marker += 1;
            let mk = *marker;
            for n in 0..nz {
                let j = pattern[n] as usize;
                let i = pmap[j];
                pattern[n] = i;
                xlhs[i as usize] = work[j];
                work[j] = 0.0;
                marked[i as usize] = mk;
            }

            // Solve with update etas. Append fill-in to pattern.
            solve_etas_t(
                nforrest,
                eta_row,
                rbegin,
                lindex,
                lvalue,
                mk,
                marked,
                pattern,
                &mut nz,
                xlhs,
                &mut rflops,
            );

            *p_nlhs = solve_lt(
                nz,
                nz_sparse,
                marker,
                m,
                ltbegin,
                ltbegin_p,
                p,
                lindex,
                lvalue,
                droptol,
                pattern_symb,
                pattern,
                pstack,
                marked,
                xlhs,
                ilhs,
                &mut lflops,
            );
        } else {
            // Sparse triangular solve with L. Solution scattered into work,
            // indices in pattern[0..nz-1].
            *marker += 1;
            let mk = *marker;
            let top = solve_symbolic(
                m,
                lbegin,
                None,
                lindex,
                irhs,
                pattern_symb,
                pstack,
                marked,
                mk,
            );
            for (&i, &x) in irhs.iter().zip(xrhs) {
                work[i as usize] = x;
            }
            let mut nz = solve_triangular(
                &pattern_symb[top as usize..],
                lbegin,
                None,
                lindex,
                lvalue,
                None,
                droptol,
                work,
                pattern,
                &mut lflops,
            ) as usize;
            unmark_cancellation(nz, top as usize, mu, pattern_symb, pattern, marked);

            // Solve with update etas. Append fill-in to pattern.
            solve_etas(
                nforrest,
                eta_row,
                rbegin,
                lindex,
                lvalue,
                mk,
                marked,
                pattern,
                &mut nz,
                work,
                &mut rflops,
            );

            *p_nlhs = solve_u(
                nz,
                nz_sparse,
                marker,
                m,
                pivotlen,
                pivotcol,
                pivotrow,
                qmap,
                ubegin,
                uindex,
                uvalue,
                row_pivot,
                droptol,
                pattern_symb,
                pattern,
                pstack,
                marked,
                work,
                ilhs,
                xlhs,
                &mut uflops,
            );
        }

        self.lflops = self.lflops.wrapping_add(lflops);
        self.uflops = self.uflops.wrapping_add(uflops);
        self.rflops = self.rflops.wrapping_add(rflops);
        self.update_cost_numer += rflops as f64;
    }

    /// lu_solve_for_update: like solve_sparse, but also prepare the update.
    /// Forward ('N'): store the spike (the solution after L and the etas)
    /// at the end of U. Transposed ('T', irhs[0] the column to leave):
    /// compute the row eta and store it in L. The solution is computed only
    /// when `out` = (nlhs, ilhs, xlhs) is given. Returns OK or REALLOCATE.
    pub(crate) fn solve_for_update(
        &mut self,
        irhs: &[Int],
        xrhs: &[f64],
        out: Option<(&mut Int, &mut [Int], &mut [f64])>,
        trans: u8,
    ) -> Int {
        let m = self.m;
        let mu = m as usize;
        let nforrest = self.nforrest as usize;
        let pivotlen = self.pivotlen as usize;
        let nz_sparse = (self.sparse_thres * m as f64) as Int;
        let droptol = self.droptol;
        let p = &self.wblink[mu + 1..];
        let pmap = &*self.pinv;
        let qmap = &*self.qinv;
        let (rbegin, eta_row) = self.rowcount_flink.split_at_mut(mu + 1);
        let pivotcol = &*self.colcount_flink;
        let pivotrow = &*self.colcount_blink;
        let (wbegin, lbegin) = self.wbegin.split_at(mu + 1);
        let (wend, ltbegin) = self.wend.split_at(mu + 1);
        let ltbegin_p = &self.wflink[mu + 1..];
        let ubegin = &*self.ubegin;
        let col_pivot = &*self.col_pivot;
        let row_pivot = &*self.row_pivot;
        let lindex = &mut *self.lindex;
        let lvalue = &mut *self.lvalue;
        let uindex = &mut *self.uindex;
        let uvalue = &mut *self.uvalue;
        let windex = &*self.windex;
        let wvalue = &*self.wvalue;
        let marked = &mut *self.iwork0;
        let marker = &mut self.marker;
        let (pattern_symb, pattern) = self.rowcount_blink.split_at_mut(mu);
        let work = &mut *self.work0;
        let pstack = as_int_mut(self.work1);

        let mut lflops: Int = 0;
        let mut uflops: Int = 0;
        let mut rflops: Int = 0;

        if is_trans(trans) {
            let jpivot = irhs[0];
            let ipivot = pmap[jpivot as usize];
            let jbegin = wbegin[jpivot as usize] as usize;
            let jend = wend[jpivot as usize] as usize;

            // Compute row eta vector. Symbolic pattern in
            // pattern_symb[top..m-1], indices of (actual) nonzeros in
            // pattern[0..nz-1], values scattered into work. We do not drop
            // small elements to zero, but the symbolic and the numeric
            // pattern will still be different when we have exact
            // cancellation.
            *marker += 1;
            let mk = *marker;
            let top = solve_symbolic(
                m,
                wbegin,
                Some(wend),
                windex,
                &windex[jbegin..jend],
                pattern_symb,
                pstack,
                marked,
                mk,
            );
            let nz_symb = m - top;

            // reallocate if not enough memory in Li, Lx (where we store R)
            let room = self.lmem - rbegin[nforrest];
            if room < nz_symb {
                self.addmem_l = nz_symb - room;
                return REALLOCATE;
            }

            for pos in jbegin..jend {
                work[windex[pos] as usize] = wvalue[pos];
            }
            solve_triangular(
                &pattern_symb[top as usize..],
                wbegin,
                Some(wend),
                windex,
                wvalue,
                Some(col_pivot),
                0.0,
                work,
                pattern,
                &mut uflops,
            );

            // Compress row eta into L, pattern mapped from column to row
            // indices. The triangularity test in update() requires the
            // symbolic pattern.
            let mut put = rbegin[nforrest] as usize;
            for &j in &pattern_symb[top as usize..] {
                let j = j as usize;
                lindex[put] = pmap[j];
                lvalue[put] = work[j];
                put += 1;
                work[j] = 0.0;
            }
            rbegin[nforrest + 1] = put as Int;
            eta_row[nforrest] = ipivot;
            self.btran_for_update = jpivot;

            if let Some((p_nlhs, ilhs, xlhs)) = out {
                // Scatter the row eta into xlhs and scale it to become the
                // solution to U^{-1}*[unit vector]. Now we can drop small
                // entries to zero and recompute the numerical pattern.
                *marker += 1;
                let mk = *marker;
                pattern[0] = ipivot;
                marked[ipivot as usize] = mk;
                let pivot = col_pivot[jpivot as usize];
                xlhs[ipivot as usize] = 1.0 / pivot;

                let xdrop = droptol * pivot.abs();
                let mut nz = 1;
                for pos in rbegin[nforrest] as usize..rbegin[nforrest + 1] as usize {
                    if lvalue[pos].abs() > xdrop {
                        let i = lindex[pos];
                        pattern[nz] = i;
                        nz += 1;
                        marked[i as usize] = mk;
                        xlhs[i as usize] = -lvalue[pos] / pivot;
                    }
                }

                // Solve with update etas. Append fill-in to pattern.
                solve_etas_t(
                    nforrest,
                    eta_row,
                    rbegin,
                    lindex,
                    lvalue,
                    mk,
                    marked,
                    pattern,
                    &mut nz,
                    xlhs,
                    &mut rflops,
                );

                *p_nlhs = solve_lt(
                    nz,
                    nz_sparse,
                    marker,
                    m,
                    ltbegin,
                    ltbegin_p,
                    p,
                    lindex,
                    lvalue,
                    droptol,
                    pattern_symb,
                    pattern,
                    pstack,
                    marked,
                    xlhs,
                    ilhs,
                    &mut lflops,
                );
            }
        } else {
            // Sparse triangular solve with L. Solution scattered into work,
            // indices in pattern[0..nz-1].
            *marker += 1;
            let mk = *marker;
            let top = solve_symbolic(
                m,
                lbegin,
                None,
                lindex,
                irhs,
                pattern_symb,
                pstack,
                marked,
                mk,
            );
            for (&i, &x) in irhs.iter().zip(xrhs) {
                work[i as usize] = x;
            }
            let mut nz = solve_triangular(
                &pattern_symb[top as usize..],
                lbegin,
                None,
                lindex,
                lvalue,
                None,
                droptol,
                work,
                pattern,
                &mut lflops,
            ) as usize;
            unmark_cancellation(nz, top as usize, mu, pattern_symb, pattern, marked);

            // Solve with update etas. Append fill-in to pattern.
            solve_etas(
                nforrest,
                eta_row,
                rbegin,
                lindex,
                lvalue,
                mk,
                marked,
                pattern,
                &mut nz,
                work,
                &mut rflops,
            );

            // reallocate if not enough memory in U
            let room = self.umem - ubegin[mu];
            let need = nz as Int + 1;
            if room < need {
                for &i in &pattern[..nz] {
                    work[i as usize] = 0.0;
                }
                self.addmem_u = need - room;
                return REALLOCATE;
            }

            // Compress spike into U.
            let want_solution = out.is_some();
            let mut put = ubegin[mu] as usize;
            for &i in &pattern[..nz] {
                uindex[put] = i;
                uvalue[put] = work[i as usize];
                put += 1;
                if !want_solution {
                    work[i as usize] = 0.0;
                }
            }
            uindex[put] = -1; // terminate column
            self.ftran_for_update = 0;

            if let Some((p_nlhs, ilhs, xlhs)) = out {
                *p_nlhs = solve_u(
                    nz,
                    nz_sparse,
                    marker,
                    m,
                    pivotlen,
                    pivotcol,
                    pivotrow,
                    qmap,
                    ubegin,
                    uindex,
                    uvalue,
                    row_pivot,
                    droptol,
                    pattern_symb,
                    pattern,
                    pstack,
                    marked,
                    work,
                    ilhs,
                    xlhs,
                    &mut uflops,
                );
            }
        }

        self.lflops = self.lflops.wrapping_add(lflops);
        self.uflops = self.uflops.wrapping_add(uflops);
        self.rflops = self.rflops.wrapping_add(rflops);
        self.update_cost_numer += rflops as f64;
        OK
    }
}

/// Unmark the entries of the symbolic pattern pattern_symb[top..m-1] that
/// cancelled in the numerical pattern pattern[0..nz-1] (after the solve
/// with L)
fn unmark_cancellation(
    nz: usize,
    top: usize,
    m: usize,
    pattern_symb: &[Int],
    pattern: &[Int],
    marked: &mut [Int],
) {
    if nz < m - top {
        let mut t = top;
        let mut n = 0;
        while n < nz {
            let i = pattern_symb[t];
            if i == pattern[n] {
                n += 1;
            } else {
                marked[i as usize] -= 1;
            }
            t += 1;
        }
        for &i in &pattern_symb[t..m] {
            marked[i as usize] -= 1;
        }
    }
}

/// Forward solve with the update etas R^0..R^{nforrest-1} on `work`;
/// entries that become nonzero and are not marked `mk` are appended to
/// pattern[*nz..]
#[allow(clippy::too_many_arguments)]
fn solve_etas(
    nforrest: usize,
    eta_row: &[Int],
    rbegin: &[Int],
    lindex: &[Int],
    lvalue: &[f64],
    mk: Int,
    marked: &mut [Int],
    pattern: &mut [Int],
    nz: &mut usize,
    work: &mut [f64],
    rflops: &mut Int,
) {
    let mut pos = rbegin[0] as usize;
    for t in 0..nforrest {
        let ipivot = eta_row[t] as usize;
        let mut x: f64 = 0.0;
        while pos < rbegin[t + 1] as usize {
            x = work[lindex[pos] as usize].mul_add_c(lvalue[pos], x);
            pos += 1;
        }
        work[ipivot] -= x;
        if x != 0.0 && marked[ipivot] != mk {
            marked[ipivot] = mk;
            pattern[*nz] = ipivot as Int;
            *nz += 1;
        }
    }
    *rflops = rflops.wrapping_add(rbegin[nforrest] - rbegin[0]);
}

/// Transposed solve with the update etas backwards on xlhs, appending
/// fill-in (entries not marked `mk`) to pattern[*nz..]
#[allow(clippy::too_many_arguments)]
fn solve_etas_t(
    nforrest: usize,
    eta_row: &[Int],
    rbegin: &[Int],
    lindex: &[Int],
    lvalue: &[f64],
    mk: Int,
    marked: &mut [Int],
    pattern: &mut [Int],
    nz: &mut usize,
    xlhs: &mut [f64],
    rflops: &mut Int,
) {
    for t in (0..nforrest).rev() {
        let ipivot = eta_row[t] as usize;
        if xlhs[ipivot] != 0.0 {
            let x = xlhs[ipivot];
            for pos in rbegin[t] as usize..rbegin[t + 1] as usize {
                let i = lindex[pos];
                if marked[i as usize] != mk {
                    marked[i as usize] = mk;
                    pattern[*nz] = i;
                    *nz += 1;
                }
                xlhs[i as usize] = (-x).mul_add_c(lvalue[pos], xlhs[i as usize]);
                *rflops += 1;
            }
        }
    }
}

/// Solve with L' on xlhs (right-hand side pattern in pattern[0..nz-1]):
/// sparse triangular solve if nz <= nz_sparse, sequential otherwise.
/// Solution indices go to ilhs; returns their number.
#[allow(clippy::too_many_arguments)]
fn solve_lt(
    nz: usize,
    nz_sparse: Int,
    marker: &mut Int,
    m: Int,
    ltbegin: &[Int],
    ltbegin_p: &[Int],
    p: &[Int],
    lindex: &[Int],
    lvalue: &[f64],
    droptol: f64,
    pattern_symb: &mut [Int],
    pattern: &[Int],
    pstack: &mut [Int],
    marked: &mut [Int],
    xlhs: &mut [f64],
    ilhs: &mut [Int],
    lflops: &mut Int,
) -> Int {
    if nz as Int <= nz_sparse {
        // Sparse triangular solve with L'. Solution scattered into xlhs,
        // indices in ilhs[0..nz-1].
        *marker += 1;
        let mk = *marker;
        let top = solve_symbolic(
            m,
            ltbegin,
            None,
            lindex,
            &pattern[..nz],
            pattern_symb,
            pstack,
            marked,
            mk,
        );
        solve_triangular(
            &pattern_symb[top as usize..],
            ltbegin,
            None,
            lindex,
            lvalue,
            None,
            droptol,
            xlhs,
            ilhs,
            lflops,
        )
    } else {
        // Sequential triangular solve with L'. Solution scattered into xlhs,
        // indices in ilhs[0..nz-1].
        let mut nz = 0;
        for k in (0..m as usize).rev() {
            let ipivot = p[k];
            let x = xlhs[ipivot as usize];
            if x != 0.0 {
                let mut pos = ltbegin_p[k] as usize;
                loop {
                    let i = lindex[pos];
                    if i < 0 {
                        break;
                    }
                    xlhs[i as usize] = (-x).mul_add_c(lvalue[pos], xlhs[i as usize]);
                    *lflops += 1;
                    pos += 1;
                }
                if x.abs() > droptol {
                    ilhs[nz] = ipivot;
                    nz += 1;
                } else {
                    xlhs[ipivot as usize] = 0.0;
                }
            }
        }
        nz as Int
    }
}

/// Solve with U on work (right-hand side pattern in pattern[0..nz-1]) and
/// permute the solution into xlhs, its pattern (column indices) into ilhs:
/// sparse triangular solve if nz <= nz_sparse, sequential otherwise.
/// Returns the number of solution entries.
#[allow(clippy::too_many_arguments)]
fn solve_u(
    nz: usize,
    nz_sparse: Int,
    marker: &mut Int,
    m: Int,
    pivotlen: usize,
    pivotcol: &[Int],
    pivotrow: &[Int],
    qmap: &[Int],
    ubegin: &[Int],
    uindex: &[Int],
    uvalue: &[f64],
    row_pivot: &[f64],
    droptol: f64,
    pattern_symb: &mut [Int],
    pattern: &[Int],
    pstack: &mut [Int],
    marked: &mut [Int],
    work: &mut [f64],
    ilhs: &mut [Int],
    xlhs: &mut [f64],
    uflops: &mut Int,
) -> Int {
    if nz as Int <= nz_sparse {
        // Sparse triangular solve with U. Solution scattered into work,
        // indices in ilhs[0..nz-1].
        *marker += 1;
        let mk = *marker;
        let top = solve_symbolic(
            m,
            ubegin,
            None,
            uindex,
            &pattern[..nz],
            pattern_symb,
            pstack,
            marked,
            mk,
        );
        let nz = solve_triangular(
            &pattern_symb[top as usize..],
            ubegin,
            None,
            uindex,
            uvalue,
            Some(row_pivot),
            droptol,
            work,
            ilhs,
            uflops,
        );

        // Permute solution into xlhs. Map pattern from row indices to column
        // indices.
        for x in &mut ilhs[..nz as usize] {
            let i = *x as usize;
            let j = qmap[i];
            *x = j;
            xlhs[j as usize] = work[i];
            work[i] = 0.0;
        }
        nz
    } else {
        // Sequential triangular solve with U. Solution computed in work and
        // permuted into xlhs. Pattern (in column indices) stored in
        // ilhs[0..nz-1].
        let mut nz = 0;
        for k in (0..pivotlen).rev() {
            let ipivot = pivotrow[k] as usize;
            let jpivot = pivotcol[k];
            if work[ipivot] != 0.0 {
                let x = work[ipivot] / row_pivot[ipivot];
                work[ipivot] = 0.0;
                let mut pos = ubegin[ipivot] as usize;
                loop {
                    let i = uindex[pos];
                    if i < 0 {
                        break;
                    }
                    work[i as usize] = (-x).mul_add_c(uvalue[pos], work[i as usize]);
                    *uflops += 1;
                    pos += 1;
                }
                if x.abs() > droptol {
                    ilhs[nz] = jpivot;
                    nz += 1;
                    xlhs[jpivot as usize] = x;
                }
            }
        }
        nz as Int
    }
}
