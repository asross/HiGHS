//! lu_update.c: Forrest-Tomlin update with reordering.

use crate::util::fma::ClangFma;

use super::file::{file_compress, file_reappend, list_swap};
use super::solve::dfs;
use super::{as_int_mut, Int, Lu, ERROR_SINGULAR_UPDATE, OK, REALLOCATE};

const GAP: Int = -1;

#[inline]
fn flip(i: Int) -> Int {
    -i - 1
}

/// find: position of index j in index[start..end-1], or end if not found
fn find(j: Int, index: &[Int], mut start: usize, end: usize) -> usize {
    while start < end && index[start] != j {
        start += 1;
    }
    start
}

/// find with end < 0: the search stops at the first negative index;
/// None if not found
fn find_neg(j: Int, index: &[Int], mut start: usize) -> Option<usize> {
    while index[start] != j && index[start] >= 0 {
        start += 1;
    }
    (index[start] == j).then_some(start)
}

/// bfs_path: find a path from j0 to j0 by a breadth first search. When
/// top < m is returned, the indices in the path (excluding the final j0) are
/// jlist[top..m-1]; top == m means no such path exists. The neighbours of
/// node j are index[begin[j]..end[j]-1]. On entry marked[j] >= 0 for all
/// nodes j; on return some elements of marked are set to zero.
#[allow(clippy::too_many_arguments)]
fn bfs_path(
    m: Int,
    j0: Int,
    begin: &[Int],
    end: &[Int],
    index: &[Int],
    jlist: &mut [Int],
    marked: &mut [Int],
    queue: &mut [Int],
) -> Int {
    let mut tail = 1;
    let mut top = m;
    let mut found = false;
    let mut j = j0;

    queue[0] = j0;
    let mut front = 0;
    while front < tail && !found {
        j = queue[front];
        for pos in begin[j as usize] as usize..end[j as usize] as usize {
            let k = index[pos];
            if k == j0 {
                found = true;
                break;
            }
            if marked[k as usize] >= 0 {
                // not in queue yet
                marked[k as usize] = flip(j); // parent[k] = j
                queue[tail] = k; // append to queue
                tail += 1;
            }
        }
        front += 1;
    }
    if found {
        // build path (j0,..,j)
        while j != j0 {
            top -= 1;
            jlist[top as usize] = j;
            j = flip(marked[j as usize]); // go to parent
        }
        top -= 1;
        jlist[top as usize] = j0;
    }
    for &q in &queue[..tail] {
        marked[q as usize] = 0; // reset
    }
    top
}

/// compress_packed: compress the matrix file to reuse memory gaps. Line
/// 0 <= i < m begins at begin[i] and ends before the first slot with
/// index == GAP; begin[m] points to the unused space at the file end.
/// Unused slots have index == GAP, all others index > GAP; index[0] must be
/// unused. On return index[1..begin[m]-1] contains the data of nonempty
/// lines, all empty lines begin at slot 0, and subsequent lines are
/// separated by one gap. Returns the number of entries.
fn compress_packed(m: Int, begin: &mut [Int], index: &mut [Int], value: &mut [f64]) -> Int {
    let mu = m as usize;
    let end = begin[mu] as usize;
    let mut nz = 0;

    // Mark the beginning of each nonempty line.
    for i in 0..mu {
        let p = begin[i] as usize;
        if index[p] == GAP {
            begin[i] = 0;
        } else {
            begin[i] = index[p]; // temporarily store index here
            index[p] = GAP - i as Int - 1; // mark beginning of line i
        }
    }

    // Compress nonempty lines.
    let mut i: Int = -1;
    let mut put = 1usize;
    for get in 1..end {
        if index[get] > GAP {
            // shift entry of line i
            index[put] = index[get];
            value[put] = value[get];
            put += 1;
            nz += 1;
        } else if index[get] < GAP {
            // beginning of line i
            i = GAP - index[get] - 1;
            index[put] = begin[i as usize]; // store back
            begin[i as usize] = put as Int;
            value[put] = value[get];
            put += 1;
            nz += 1;
        } else if i >= 0 {
            // line i ended at a gap
            i = -1;
            index[put] = GAP;
            put += 1;
        }
    }
    begin[mu] = put as Int;
    nz
}

/// The parts of struct lu that permute() changes
struct Permute<'b> {
    pmap: &'b mut [Int],
    qmap: &'b mut [Int],
    ubegin: &'b mut [Int],
    wbegin: &'b mut [Int],
    wend: &'b mut [Int],
    wflink: &'b mut [Int],
    wblink: &'b mut [Int],
    col_pivot: &'b mut [f64],
    row_pivot: &'b mut [f64],
    uindex: &'b mut [Int],
    uvalue: &'b mut [f64],
    windex: &'b mut [Int],
    wvalue: &'b mut [f64],
    min_pivot: &'b mut f64,
    max_pivot: &'b mut f64,
}

/// permute: change row-column mappings for columns jlist[0..nswap]. When
/// row i was mapped to column jlist[n], it will be mapped to jlist[n+1];
/// the row mapped to jlist[nswap] will be mapped to jlist[0]. This updates
/// pmap, qmap, the rowwise and columnwise storage of U and the pivots.
/// (Looks inefficient, in particular the list swaps, but nswap is usually
/// small.)
fn permute(s: Permute, jlist: &[Int], nswap: usize) {
    let Permute {
        pmap,
        qmap,
        ubegin,
        wbegin,
        wend,
        wflink,
        wblink,
        col_pivot,
        row_pivot,
        uindex,
        uvalue,
        windex,
        wvalue,
        min_pivot,
        max_pivot,
    } = s;
    let j0 = jlist[0];
    let jn = jlist[nswap];
    let i0 = pmap[j0 as usize];
    let in_ = pmap[jn as usize];
    debug_assert!(nswap >= 1);

    // Update row file

    let begin = wbegin[jn as usize]; // keep for later
    let end = wend[jn as usize];
    let piv = col_pivot[jn as usize];

    for n in (1..=nswap).rev() {
        let j = jlist[n];
        let jprev = jlist[n - 1];
        let (ju, jp) = (j as usize, jprev as usize);

        // When row i was indexed by jprev in the row file before, then it is
        // indexed by j now.
        wbegin[ju] = wbegin[jp];
        wend[ju] = wend[jp];
        list_swap(wflink, wblink, j, jprev);

        // That row must have an entry in column j because (jprev,j) is an
        // edge in the augmenting path. This entry becomes a pivot element.
        // If jprev is not the first node in the path, then it has an entry
        // in the row (the old pivot) which becomes an off-diagonal entry
        // now.
        let where_ = find(j, windex, wbegin[ju] as usize, wend[ju] as usize);
        if n > 1 {
            windex[where_] = jprev;
            col_pivot[ju] = wvalue[where_];
            wvalue[where_] = col_pivot[jp];
        } else {
            col_pivot[ju] = wvalue[where_];
            wend[ju] -= 1;
            let e = wend[ju] as usize;
            windex[where_] = windex[e];
            wvalue[where_] = wvalue[e];
        }
        *min_pivot = min_pivot.min(col_pivot[ju].abs());
        *max_pivot = max_pivot.max(col_pivot[ju].abs());
    }

    wbegin[j0 as usize] = begin;
    wend[j0 as usize] = end;
    let where_ = find(j0, windex, begin as usize, end as usize);
    windex[where_] = jn;
    col_pivot[j0 as usize] = wvalue[where_];
    wvalue[where_] = piv;
    *min_pivot = min_pivot.min(col_pivot[j0 as usize].abs());
    *max_pivot = max_pivot.max(col_pivot[j0 as usize].abs());

    // Update column file

    let begin = ubegin[i0 as usize]; // keep for later

    for n in 0..nswap {
        let i = pmap[jlist[n] as usize];
        let inext = pmap[jlist[n + 1] as usize];

        // When column j was indexed by inext in the column file before, then
        // it is indexed by i now.
        ubegin[i as usize] = ubegin[inext as usize];

        // That column must have an entry in row i because there is an edge
        // in the augmenting path. This entry becomes a pivot element. There
        // is also an entry in row inext (the old pivot), which now becomes
        // an off-diagonal entry.
        let where_ = find_neg(i, uindex, ubegin[i as usize] as usize).expect("path edge");
        uindex[where_] = inext;
        row_pivot[i as usize] = uvalue[where_];
        uvalue[where_] = row_pivot[inext as usize];
    }

    ubegin[in_ as usize] = begin;
    let where_ = find_neg(in_, uindex, begin as usize).expect("path edge");
    row_pivot[in_ as usize] = uvalue[where_];
    let mut end = where_;
    while uindex[end] >= 0 {
        end += 1;
    }
    uindex[where_] = uindex[end - 1];
    uvalue[where_] = uvalue[end - 1];
    uindex[end - 1] = -1;

    // Update row-column mappings

    for n in (1..=nswap).rev() {
        let j = jlist[n];
        let i = pmap[jlist[n - 1] as usize];
        pmap[j as usize] = i;
        qmap[i as usize] = j;
    }
    pmap[j0 as usize] = in_;
    qmap[in_ as usize] = j0;
}

impl Lu<'_> {
    /// lu_update: insert the spike into U and restore triangularity. If the
    /// spiked matrix is permuted triangular, only the permutations are
    /// updated; otherwise the Forrest-Tomlin update adds a row eta.
    ///
    /// Returns OK, REALLOCATE (require more memory in W) or
    /// ERROR_SINGULAR_UPDATE (new pivot element is zero or < abstol).
    ///
    /// If the singularity test fails or memory is insufficient, the update
    /// is aborted and the user may call this routine a second time; changes
    /// made in the first call must not violate the logic in the second.
    pub(crate) fn update(&mut self, xtbl: f64) -> Int {
        let m = self.m;
        let mu = m as usize;
        let nforrest = self.nforrest as usize;
        let mut unz = self.unz;
        let pad = self.pad;
        let stretch = self.stretch;
        let pmap = &mut *self.pinv;
        let qmap = &mut *self.qinv;
        let ubegin = &mut *self.ubegin;
        let rbegin = &mut self.rowcount_flink[..mu + 1];
        let wbegin = &mut self.wbegin[..mu + 1];
        let wend = &mut self.wend[..mu + 1];
        let wflink = &mut self.wflink[..mu + 1];
        let wblink = &mut self.wblink[..mu + 1];
        let col_pivot = &mut *self.col_pivot;
        let row_pivot = &mut *self.row_pivot;
        let lindex = &mut *self.lindex;
        let lvalue = &mut *self.lvalue;
        let uindex = &mut *self.uindex;
        let uvalue = &mut *self.uvalue;
        let windex = &mut *self.windex;
        let wvalue = &mut *self.wvalue;
        let marked = &mut *self.iwork0;
        let iwork1 = &mut *self.rowcount_blink;
        let work1 = &mut *self.work1;

        let jpivot = self.btran_for_update;
        let ipivot = pmap[jpivot as usize];
        let oldpiv = col_pivot[jpivot as usize];
        debug_assert!((nforrest as Int) < m);

        // Prepare

        // if present, move diagonal element to end of spike
        let mut spike_diag = 0.0;
        let mut have_diag = false;
        let mut put = ubegin[mu] as usize;
        let mut pos = put;
        loop {
            let i = uindex[pos];
            if i < 0 {
                break;
            }
            if i != ipivot {
                uindex[put] = i;
                uvalue[put] = uvalue[pos];
                put += 1;
            } else {
                spike_diag = uvalue[pos];
                have_diag = true;
            }
            pos += 1;
        }
        if have_diag {
            uindex[put] = ipivot;
            uvalue[put] = spike_diag;
        }
        let nz_spike = put - ubegin[mu] as usize; // nz excluding diagonal

        let nz_roweta = rbegin[nforrest + 1] - rbegin[nforrest];

        // Compute pivot

        // newpiv is the diagonal element in the spike column after the
        // Forrest-Tomlin update has been applied. It can be computed as
        //
        //    newpiv = spike_diag - dot(spike,row eta)                (1)
        // or
        //    newpiv = xtbl * oldpiv,                                 (2)
        //
        // where spike_diag is the diagonal element in the spike column
        // before the update and oldpiv was the pivot element in column
        // jpivot before inserting the spike. Use (1) and report the
        // difference to (2) to monitor numerical stability.
        //
        // While computing (1), count the intersection of the patterns of
        // spike and row eta.

        // scatter row eta into work1 and mark positions
        self.marker += 1;
        let mk = self.marker;
        for pos in rbegin[nforrest] as usize..rbegin[nforrest + 1] as usize {
            let i = lindex[pos] as usize;
            marked[i] = mk;
            work1[i] = lvalue[pos];
        }

        // compute newpiv and count intersection
        let spike = ubegin[mu] as usize..ubegin[mu] as usize + nz_spike;
        let mut newpiv = spike_diag;
        let mut intersect = 0;
        for pos in spike.clone() {
            let i = uindex[pos] as usize;
            if marked[i] == mk {
                newpiv = (-uvalue[pos]).mul_add_c(work1[i], newpiv);
                intersect += 1;
            }
        }

        // singularity test
        if newpiv == 0.0 || newpiv.abs() < self.abstol {
            return ERROR_SINGULAR_UPDATE;
        }

        // stability measure
        let piverr = (-xtbl).mul_add_c(oldpiv, newpiv).abs();

        // Insert spike

        // calculate bound on file growth
        let mut grow: Int = 0;
        for pos in spike.clone() {
            let j = qmap[uindex[pos] as usize] as usize;
            let jnext = wflink[j] as usize;
            if wend[j] == wbegin[jnext] {
                let nz = wend[j] - wbegin[j];
                grow += nz + 1; // row including spike entry
                grow = (grow as f64 + stretch.mul_add_c((nz + 1) as f64, pad as f64)) as Int;
                // extra room
            }
        }

        // reallocate if necessary
        let room = wend[mu] - wbegin[mu];
        if grow > room {
            self.addmem_w = grow - room;
            return REALLOCATE;
        }

        // remove column jpivot from row file
        let mut nz = 0;
        let mut pos = ubegin[ipivot as usize] as usize;
        while uindex[pos] >= 0 {
            let j = qmap[uindex[pos] as usize] as usize;
            let end = wend[j] as usize;
            wend[j] -= 1;
            let where_ = find(jpivot, windex, wbegin[j] as usize, end);
            windex[where_] = windex[end - 1];
            wvalue[where_] = wvalue[end - 1];
            nz += 1;
            pos += 1;
        }
        unz -= nz;

        // erase column jpivot in column file
        let mut pos = ubegin[ipivot as usize] as usize;
        while uindex[pos] >= 0 {
            uindex[pos] = GAP;
            pos += 1;
        }

        // set column pointers to spike, chop off diagonal
        ubegin[ipivot as usize] = ubegin[mu];
        ubegin[mu] += nz_spike as Int;
        uindex[ubegin[mu] as usize] = GAP;
        ubegin[mu] += 1;

        // insert spike into row file
        let mut pos = ubegin[ipivot as usize] as usize;
        loop {
            let i = uindex[pos];
            if i < 0 {
                break;
            }
            let j = qmap[i as usize];
            let ju = j as usize;
            let jnext = wflink[ju] as usize;
            if wend[ju] == wbegin[jnext] {
                let nz = wend[ju] - wbegin[ju];
                let room = (stretch.mul_add_c((nz + 1) as f64, 1.0) + pad as f64) as Int;
                file_reappend(j, m, wbegin, wend, wflink, wblink, windex, wvalue, room);
            }
            let end = wend[ju] as usize;
            wend[ju] += 1;
            windex[end] = jpivot;
            wvalue[end] = uvalue[pos];
            pos += 1;
        }
        unz += nz_spike as Int;

        // insert diagonal
        col_pivot[jpivot as usize] = spike_diag;
        row_pivot[ipivot as usize] = spike_diag;

        // Test triangularity

        // Offset of row_reach in iwork1 and of col_reach in iwork2 = iwork1
        // + m, and their length, if the spiked matrix is permuted triangular
        let mut reach: Option<(usize, usize)> = None;
        let istriangular;

        if have_diag {
            // When the spike has a nonzero diagonal element, then the spiked
            // matrix is (symmetrically) permuted triangular if and only if
            // reach(ipivot) does not intersect with the spike pattern except
            // for ipivot. Since reach(ipivot) \ {ipivot} is the structural
            // pattern of the row eta, the matrix is permuted triangular iff
            // the patterns of the row eta and the spike do not intersect.
            //
            // To update the permutations below, we have to provide
            // reach(ipivot) and the associated column indices in topological
            // order as arrays row_reach[0..nreach-1] and
            // col_reach[0..nreach-1]. Because the pattern of the row eta was
            // computed by a dfs, we obtain row_reach simply by adding ipivot
            // to the front. col_reach can then be obtained through qmap.
            istriangular = intersect == 0;
            if istriangular {
                self.min_pivot = self.min_pivot.min(newpiv.abs());
                self.max_pivot = self.max_pivot.max(newpiv.abs());

                // build row_reach and col_reach in topological order
                let nreach = nz_roweta as usize + 1;
                let (row_reach, col_reach) = iwork1.split_at_mut(mu);
                row_reach[0] = ipivot;
                col_reach[0] = jpivot;
                let mut pos = rbegin[nforrest] as usize;
                for n in 1..nreach {
                    let i = lindex[pos];
                    pos += 1;
                    row_reach[n] = i;
                    col_reach[n] = qmap[i as usize];
                }
                reach = Some((0, nreach));
                self.nsymperm_total += 1;
            }
        } else {
            // The spike has a zero diagonal element, so the spiked matrix may
            // only be the *un*symmetric permutation of an upper triangular
            // matrix.
            //
            // Part 1:
            //
            // Find an augmenting path in U[pmap,:] starting from jpivot. An
            // augmenting path is a sequence of column indices such that there
            // is an edge from each node to the next, and an edge from the
            // final node back to jpivot. bfs_path computes such a path in
            // path[top..m-1].
            //
            // Because jpivot has no self-edge, the path must have at least
            // two nodes. The path must exist because otherwise the spiked
            // matrix was structurally singular and the singularity test above
            // had failed.
            let (path, reach_arr) = iwork1.split_at_mut(mu);
            let pstack = as_int_mut(work1);

            let top = bfs_path(m, jpivot, wbegin, wend, windex, path, marked, reach_arr) as usize;
            debug_assert!(top < mu - 1 && path[top] == jpivot);

            // Part 2a:
            //
            // For each path index j (except the final one) mark the nodes in
            // reach(j), where the reach is computed in U[pmap,:] without the
            // path edges. If a path index is contained in the reach of an
            // index that comes before it in the path, then U is not permuted
            // triangular.
            //
            // At the same time assemble the combined reach of all path nodes
            // (except the final one) in U[pmap_new,:], where pmap_new is the
            // column-row mapping after applying the permutation associated
            // with the augmenting path. We only have to replace each index
            // where the dfs starts by the next index in the path. The
            // combined reach is then assembled in topological order in
            // reach[rtop..m-1].
            let mut tri = true;
            let mut rtop = m;
            self.marker += 1;
            let mk = self.marker;
            let mut t = top;
            while t < mu - 1 && tri {
                let j = path[t];
                let jnext = path[t + 1];
                let ju = j as usize;
                let where_ = find(jnext, windex, wbegin[ju] as usize, wend[ju] as usize);
                windex[where_] = j; // take out for a moment
                rtop = dfs(
                    j,
                    wbegin,
                    Some(wend),
                    windex,
                    rtop,
                    reach_arr,
                    pstack,
                    marked,
                    mk,
                );
                reach_arr[rtop as usize] = jnext;
                windex[where_] = jnext; // restore
                tri = marked[jnext as usize] != mk;
                t += 1;
            }

            // Part 2b:
            //
            // If the matrix looks triangular so far, then also mark the reach
            // of the final path node, which is reach(jpivot) in
            // U[pmap_new,:]. U is then permuted triangular iff the combined
            // reach does not intersect the spike pattern except in the final
            // path index.
            if tri {
                let j = path[mu - 1];
                rtop = dfs(
                    j,
                    wbegin,
                    Some(wend),
                    windex,
                    rtop,
                    reach_arr,
                    pstack,
                    marked,
                    mk,
                );
                reach_arr[rtop as usize] = jpivot;
                marked[j as usize] -= 1; // unmark for a moment
                let mut pos = ubegin[ipivot as usize] as usize;
                loop {
                    let i = uindex[pos];
                    if i < 0 {
                        break;
                    }
                    if marked[qmap[i as usize] as usize] == mk {
                        tri = false;
                    }
                    pos += 1;
                }
                marked[j as usize] += 1; // restore
            }

            // If U is permuted triangular, then permute to zero-free
            // diagonal. Set up row_reach[0..nreach-1] and
            // col_reach[0..nreach-1] for updating the permutations below. The
            // column reach is the combined reach of the path nodes. The row
            // reach is given through pmap.
            if tri {
                let nswap = mu - top - 1;
                permute(
                    Permute {
                        pmap,
                        qmap,
                        ubegin,
                        wbegin,
                        wend,
                        wflink,
                        wblink,
                        col_pivot,
                        row_pivot,
                        uindex,
                        uvalue,
                        windex,
                        wvalue,
                        min_pivot: &mut self.min_pivot,
                        max_pivot: &mut self.max_pivot,
                    },
                    &path[top..],
                    nswap,
                );
                unz -= 1;
                let rtop = rtop as usize;
                // row_reach = iwork1 + rtop (overwrites the path, which is
                // no longer needed), col_reach = reach + rtop
                for n in rtop..mu {
                    path[n] = pmap[reach_arr[n] as usize];
                }
                reach = Some((rtop, mu - rtop));
            }
            istriangular = tri;
        }

        // Forrest-Tomlin update

        if !istriangular {
            // remove row ipivot from column file
            for pos in wbegin[jpivot as usize] as usize..wend[jpivot as usize] as usize {
                let j = windex[pos];
                let mut where_ = usize::MAX;
                let mut end = ubegin[pmap[j as usize] as usize] as usize;
                loop {
                    let i = uindex[end];
                    if i < 0 {
                        break;
                    }
                    if i == ipivot {
                        where_ = end;
                    }
                    end += 1;
                }
                uindex[where_] = uindex[end - 1];
                uvalue[where_] = uvalue[end - 1];
                uindex[end - 1] = -1;
                unz -= 1;
            }

            // remove row ipivot from row file
            wend[jpivot as usize] = wbegin[jpivot as usize];

            // replace pivot
            col_pivot[jpivot as usize] = newpiv;
            row_pivot[ipivot as usize] = newpiv;
            self.min_pivot = self.min_pivot.min(newpiv.abs());
            self.max_pivot = self.max_pivot.max(newpiv.abs());

            // drop zeros from row eta; update max entry of row etas
            let mut nz = 0;
            let mut put = rbegin[nforrest] as usize;
            let mut max_eta: f64 = 0.0;
            for pos in put..rbegin[nforrest + 1] as usize {
                if lvalue[pos] != 0.0 {
                    max_eta = max_eta.max(lvalue[pos].abs());
                    lindex[put] = lindex[pos];
                    lvalue[put] = lvalue[pos];
                    put += 1;
                    nz += 1;
                }
            }
            rbegin[nforrest + 1] = put as Int;
            self.rnz += nz;
            self.max_eta = self.max_eta.max(max_eta);

            // prepare permutation update
            self.nforrest += 1;
            self.nforrest_total += 1;
        }

        // Update permutations

        let nreach = reach.map_or(1, |(_, n)| n);
        if self.pivotlen as usize + nreach > 2 * mu {
            self.garbage_perm();
        }

        // append row indices row_reach[0..nreach-1] and col indices
        // col_reach[0..nreach-1] to end of pivot sequence
        let put = self.pivotlen as usize;
        match reach {
            Some((off, n)) => {
                let iwork1 = &*self.rowcount_blink;
                self.colcount_blink[put..put + n].copy_from_slice(&iwork1[off..off + n]);
                self.colcount_flink[put..put + n].copy_from_slice(&iwork1[mu + off..mu + off + n]);
            }
            None => {
                self.colcount_blink[put] = ipivot;
                self.colcount_flink[put] = jpivot;
            }
        }
        self.pivotlen += nreach as Int;

        // Clean up

        // compress U if used memory is shrinked sufficiently
        let used = self.ubegin[mu];
        if (used - unz - m) as f64 > self.compress_thres * used as f64 {
            let _nz = compress_packed(m, self.ubegin, self.uindex, self.uvalue);
            debug_assert!(_nz == unz);
        }

        // compress W if used memory is shrinked sufficiently
        let used = self.wbegin[mu];
        let need = (stretch.mul_add_c(unz as f64, unz as f64) + (m * pad) as f64) as Int;
        if (used - need) as f64 > self.compress_thres * used as f64 {
            let _nz = file_compress(
                m,
                &mut self.wbegin[..mu + 1],
                &mut self.wend[..mu + 1],
                &self.wflink[..mu + 1],
                self.windex,
                self.wvalue,
                stretch,
                pad,
            );
            debug_assert!(_nz == unz);
        }

        self.pivot_error = piverr / (1.0 + newpiv.abs());
        self.unz = unz;
        self.btran_for_update = -1;
        self.ftran_for_update = -1;
        self.update_cost_numer += nz_roweta as f64;
        self.nupdate += 1;
        self.nupdate_total += 1;
        OK
    }
}
