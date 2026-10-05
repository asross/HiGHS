//! sparse_utils.h/.cc: depth-first search and augmenting paths, adapted
//! from CSparse (cs_dfs.c, cs_augment.c).

use super::Int;

/// Depth-first search in the graph of matrix A from node istart (unmarked
/// on entry). The neighbours of node i are the entries in column colmap[i]
/// of A (none if negative; identity if colmap is None). The newly reached
/// nodes are stored in istack[newtop..top-1] and marked; returns newtop.
/// work: size # rows of A.
pub fn depth_first_search(
    istart: Int,
    ap: &[Int],
    ai: &[Int],
    colmap: Option<&[Int]>,
    mut top: Int,
    istack: &mut [Int],
    marked: &mut [Int],
    marker: Int,
    work: &mut [Int],
) -> Int {
    debug_assert!(marked[istart as usize] != marker);
    let pstack = work;
    let mut head: Int = 0;
    istack[0] = istart;
    while head >= 0 {
        let h = head as usize;
        let i = istack[h] as usize;
        let j = colmap.map_or(i as Int, |c| c[i]);
        if marked[i] != marker {
            marked[i] = marker;
            pstack[h] = if j >= 0 { ap[j as usize] } else { 0 };
        }
        let mut done = true;
        let pbeg = pstack[h];
        let pend = if j >= 0 { ap[j as usize + 1] } else { 0 };
        for p in pbeg..pend {
            let inext = ai[p as usize];
            if marked[inext as usize] == marker {
                continue;
            }
            pstack[h] = p + 1;
            head += 1;
            istack[head as usize] = inext;
            done = false;
            break;
        }
        if done {
            head -= 1;
            top -= 1;
            istack[top as usize] = i as Int;
        }
    }
    top
}

/// Alternating augmenting path for extending a matching for the m-by-n
/// matrix A by column jstart. jmatch[i] = j >= 0 if row i is matched to
/// column j, -1 if unmatched, < -1 if excluded. cheap must hold Ap[0..n-1]
/// on the first call and be unchanged in between; marked[j] != jstart on
/// entry. work: size m, work2, work3: size m+1. Returns true if the
/// matching was extended.
pub fn augmenting_path(
    jstart: Int,
    ap: &[Int],
    ai: &[Int],
    jmatch: &mut [Int],
    cheap: &mut [Int],
    marked: &mut [Int],
    work: &mut [Int],
    work2: &mut [Int],
    work3: &mut [Int],
) -> bool {
    // istack holds row indices without duplicates.
    // jstack holds jstart and up to m indices from jmatch.
    // pstack holds the corresponding column pointers.
    let istack = work;
    let jstack = work2;
    let pstack = work3;
    let mut found = false;
    let mut head: Int = 0;
    jstack[0] = jstart;
    while head >= 0 {
        let h = head as usize;
        let j = jstack[h] as usize;
        if marked[j] != jstart {
            // first time j visited in this path
            marked[j] = jstart;
            let mut i = 0;
            let mut p = cheap[j];
            // C loop: the increment also runs after the match is found
            while p < ap[j + 1] && !found {
                i = ai[p as usize];
                found = jmatch[i as usize] == -1;
                p += 1;
            }
            cheap[j] = p;
            if found {
                istack[h] = i;
                break;
            }
            pstack[h] = ap[j]; // start depth-first search from j
        }
        let pbeg = pstack[h];
        let pend = ap[j + 1];
        let mut p = pbeg;
        while p < pend {
            let i = ai[p as usize];
            let jm = jmatch[i as usize];
            if jm < -1 {
                // row to be ignored
                p += 1;
                continue;
            }
            if marked[jm as usize] == jstart {
                p += 1;
                continue;
            }
            pstack[h] = p + 1;
            istack[h] = i;
            head += 1;
            jstack[head as usize] = jm; // continue augmenting path here
            break;
        }
        if p == ap[j + 1] {
            head -= 1;
        }
    }
    if found {
        for p in (0..=head).rev() {
            let p = p as usize;
            jmatch[istack[p] as usize] = jstack[p];
        }
    }
    found
}
