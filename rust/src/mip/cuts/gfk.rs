//! HighsGFkSolve (highs/mip/HighsGFkSolve.h): Gaussian elimination over the
//! field of integers mod a small prime k, iterating basic solutions, with
//! the same pivot order (libc++ priority queue) and splay trees for rows.

use super::sort::PriorityQueue;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct SolutionEntry {
    pub index: i32,
    pub weight: u32,
}

fn inverse(k: u32, a: u32) -> u32 {
    match k {
        2 => 1,
        3 => a,
        // HighsGFk<k>::inverse: a^(k-2) % k with unsigned overflow as in C++
        _ => {
            let mut p: u32 = 1;
            for _ in 0..k - 2 {
                p = p.wrapping_mul(a);
            }
            p % k
        }
    }
}

#[derive(Default)]
pub struct GfkSolve {
    num_col: usize,
    num_row: usize,
    arow: Vec<i32>,
    acol: Vec<i32>,
    avalue: Vec<u32>,
    rowsize: Vec<i32>,
    colsize: Vec<i32>,
    colhead: Vec<i32>,
    anext: Vec<i32>,
    aprev: Vec<i32>,
    rowroot: Vec<i32>,
    arleft: Vec<i32>,
    arright: Vec<i32>,
    rhs: Vec<u32>,
    factor_col_perm: Vec<i32>,
    factor_row_perm: Vec<i32>,
    col_basis_status: Vec<i8>,
    row_used: Vec<i8>,
    iterstack: Vec<i32>,
    rowpositions: Vec<i32>,
    rowpos_colsizes: Vec<i32>,
    /// min-heap of free slots (distinct values: any heap gives the same
    /// order)
    freeslots: std::collections::BinaryHeap<std::cmp::Reverse<i32>>,
}

/// highs_splay on the row trees (keys: Acol)
fn splay(key: i32, mut root: i32, left: &mut [i32], right: &mut [i32], keyof: &[i32]) -> i32 {
    if root == -1 {
        return -1;
    }
    // the C++ keeps pointers to the links to fill; here: which link of
    // which node (-1: the local Nleft/Nright)
    let mut n_left = -1i32;
    let mut n_right = -1i32;
    // lright: a right link (of node, or N_RIGHT); rleft: a left link
    let mut lright_node = -1i32;
    let mut rleft_node = -1i32;
    macro_rules! set_lright {
        ($v:expr) => {
            if lright_node == -1 {
                n_right = $v;
            } else {
                right[lright_node as usize] = $v;
            }
        };
    }
    macro_rules! set_rleft {
        ($v:expr) => {
            if rleft_node == -1 {
                n_left = $v;
            } else {
                left[rleft_node as usize] = $v;
            }
        };
    }
    loop {
        let r = root as usize;
        if key < keyof[r] {
            let l = left[r];
            if l == -1 {
                break;
            }
            if key < keyof[l as usize] {
                let y = l as usize;
                left[r] = right[y];
                right[y] = root;
                root = y as i32;
                if left[y] == -1 {
                    break;
                }
            }
            set_rleft!(root);
            rleft_node = root;
            root = left[root as usize];
        } else if key > keyof[r] {
            let rr = right[r];
            if rr == -1 {
                break;
            }
            if key > keyof[rr as usize] {
                let y = rr as usize;
                right[r] = left[y];
                left[y] = root;
                root = y as i32;
                if right[y] == -1 {
                    break;
                }
            }
            set_lright!(root);
            lright_node = root;
            root = right[root as usize];
        } else {
            break;
        }
    }
    let r = root as usize;
    set_lright!(left[r]);
    set_rleft!(right[r]);
    left[r] = n_right;
    right[r] = n_left;
    root
}

impl GfkSolve {
    fn link(&mut self, pos: usize) {
        let col = self.acol[pos] as usize;
        self.anext[pos] = self.colhead[col];
        self.aprev[pos] = -1;
        self.colhead[col] = pos as i32;
        if self.anext[pos] != -1 {
            self.aprev[self.anext[pos] as usize] = pos as i32;
        }
        self.colsize[col] += 1;

        // highs_splay_link
        let row = self.arow[pos] as usize;
        let root = self.rowroot[row];
        if root == -1 {
            self.arleft[pos] = -1;
            self.arright[pos] = -1;
            self.rowroot[row] = pos as i32;
        } else {
            let key = self.acol[pos];
            let root = splay(key, root, &mut self.arleft, &mut self.arright, &self.acol);
            let r = root as usize;
            if key < self.acol[r] {
                self.arleft[pos] = self.arleft[r];
                self.arright[pos] = root;
                self.arleft[r] = -1;
            } else {
                self.arright[pos] = self.arright[r];
                self.arleft[pos] = root;
                self.arright[r] = -1;
            }
            self.rowroot[row] = pos as i32;
        }
        self.rowsize[row] += 1;
    }

    /// highs_splay_unlink on the tree rooted at *root
    fn splay_unlink(&mut self, node: usize, root: i32) -> i32 {
        let key = self.acol[node];
        let root = splay(key, root, &mut self.arleft, &mut self.arright, &self.acol);
        if root != node as i32 {
            let sub = self.arright[root as usize];
            let newsub = self.splay_unlink(node, sub);
            self.arright[root as usize] = newsub;
            return root;
        }
        if self.arleft[node] == -1 {
            self.arright[node]
        } else {
            let r = splay(key, self.arleft[node], &mut self.arleft, &mut self.arright, &self.acol);
            self.arright[r as usize] = self.arright[node];
            r
        }
    }

    fn unlink(&mut self, pos: usize) {
        let next = self.anext[pos];
        let prev = self.aprev[pos];
        if next != -1 {
            self.aprev[next as usize] = prev;
        }
        let col = self.acol[pos] as usize;
        if prev != -1 {
            self.anext[prev as usize] = next;
        } else {
            self.colhead[col] = next;
        }
        self.colsize[col] -= 1;
        let row = self.arow[pos] as usize;
        let root = self.rowroot[row];
        self.rowroot[row] = self.splay_unlink(pos, root);
        self.rowsize[row] -= 1;
        self.avalue[pos] = 0;
        self.freeslots.push(std::cmp::Reverse(pos as i32));
    }

    fn store_row_positions(&mut self, pos: i32) {
        if pos == -1 {
            return;
        }
        self.iterstack.push(pos);
        while let Some(pos) = self.iterstack.pop() {
            self.rowpositions.push(pos);
            self.rowpos_colsizes.push(self.colsize[self.acol[pos as usize] as usize]);
            if self.arleft[pos as usize] != -1 {
                self.iterstack.push(self.arleft[pos as usize]);
            }
            if self.arright[pos as usize] != -1 {
                self.iterstack.push(self.arright[pos as usize]);
            }
        }
    }

    fn find_nonzero(&mut self, row: usize, col: i32) -> i32 {
        let root = self.rowroot[row];
        if root == -1 {
            return -1;
        }
        let root = splay(col, root, &mut self.arleft, &mut self.arright, &self.acol);
        self.rowroot[row] = root;
        if self.acol[root as usize] == col {
            root
        } else {
            -1
        }
    }

    fn add_nonzero(&mut self, row: usize, col: i32, val: u32) {
        let pos = match self.freeslots.pop() {
            None => {
                let pos = self.avalue.len();
                self.avalue.push(val);
                self.arow.push(row as i32);
                self.acol.push(col);
                self.anext.push(-1);
                self.aprev.push(-1);
                self.arleft.push(-1);
                self.arright.push(-1);
                pos
            }
            Some(std::cmp::Reverse(pos)) => {
                let pos = pos as usize;
                self.avalue[pos] = val;
                self.arow[pos] = row as i32;
                self.acol[pos] = col;
                self.aprev[pos] = -1;
                pos
            }
        };
        self.link(pos);
    }

    /// fromCSC<k>(Aval, Aindex, Astart, numRow)
    pub fn from_csc(&mut self, k: u32, aval: &[i64], aindex: &[i32], astart: &[i32], num_row: usize) {
        self.avalue.clear();
        self.acol.clear();
        self.arow.clear();
        self.freeslots.clear();
        self.num_col = astart.len() - 1;
        self.num_row = num_row;
        self.colhead.clear();
        self.colhead.resize(self.num_col, -1);
        self.colsize.clear();
        self.colsize.resize(self.num_col, 0);
        self.rhs.clear();
        self.rhs.resize(num_row, 0);
        self.rowroot.clear();
        self.rowroot.resize(num_row, -1);
        self.rowsize.clear();
        self.rowsize.resize(num_row, 0);
        for i in 0..self.num_col {
            for j in astart[i] as usize..astart[i + 1] as usize {
                let mut val = aval[j] % k as i64;
                if val == 0 {
                    continue;
                }
                if val < 0 {
                    val += k as i64;
                }
                self.avalue.push(val as u32);
                self.acol.push(i as i32);
                self.arow.push(aindex[j]);
            }
        }
        let nnz = self.avalue.len();
        self.anext.clear();
        self.anext.resize(nnz, 0);
        self.aprev.clear();
        self.aprev.resize(nnz, 0);
        self.arleft.clear();
        self.arleft.resize(nnz, 0);
        self.arright.clear();
        self.arright.resize(nnz, 0);
        for pos in 0..nnz {
            self.link(pos);
        }
    }

    /// setRhs<k>(row, val)
    pub fn set_rhs(&mut self, k: u32, row: usize, val: i32) {
        self.rhs[row] = val.unsigned_abs() % k;
    }

    /// solve<k>(reportSolution)
    pub fn solve(&mut self, k: u32, mut report_solution: impl FnMut(&mut Vec<SolutionEntry>)) {
        let mut pqueue = PriorityQueue::new(|a: &(i32, i32), b: &(i32, i32)| a.0 > b.0);
        for i in 0..self.num_col {
            pqueue.push((self.colsize[i], i as i32));
        }
        let max_pivot = self.num_row.min(self.num_col);
        self.factor_col_perm.clear();
        self.factor_row_perm.clear();
        self.col_basis_status.clear();
        self.col_basis_status.resize(self.num_col, 0);
        self.row_used.clear();
        self.row_used.resize(self.num_row, 0);
        let mut num_pivot = 0;

        while !pqueue.is_empty() {
            let (old_col_size, pivot_col) = pqueue.top();
            pqueue.pop();
            let pc = pivot_col as usize;
            if self.colsize[pc] == 0 {
                continue;
            }
            if self.colsize[pc] != old_col_size {
                pqueue.push((self.colsize[pc], pivot_col));
                continue;
            }
            let mut pivot = -1i32;
            let mut pivot_row = -1i32;
            let mut pivot_row_len = i32::MAX;
            let mut coliter = self.colhead[pc];
            while coliter != -1 {
                let row = self.arow[coliter as usize];
                if self.row_used[row as usize] == 0 && self.rowsize[row as usize] < pivot_row_len {
                    pivot_row_len = self.rowsize[row as usize];
                    pivot_row = row;
                    pivot = coliter;
                }
                coliter = self.anext[coliter as usize];
            }
            let pivot_inverse = inverse(k, self.avalue[pivot as usize]);
            self.rowpositions.clear();
            self.rowpos_colsizes.clear();
            self.store_row_positions(self.rowroot[pivot_row as usize]);
            let mut coliter = self.colhead[pc];
            while coliter != -1 {
                let next = self.anext[coliter as usize];
                if coliter == pivot {
                    coliter = next;
                    continue;
                }
                let row = self.arow[coliter as usize] as usize;
                if self.row_used[row] != 0 {
                    coliter = next;
                    continue;
                }
                let pivot_row_scale = pivot_inverse.wrapping_mul(k - self.avalue[coliter as usize]);
                self.rhs[row] =
                    self.rhs[row].wrapping_add(pivot_row_scale.wrapping_mul(self.rhs[pivot_row as usize])) % k;
                for p in 0..self.rowpositions.len() {
                    let pivot_row_pos = self.rowpositions[p] as usize;
                    let c = self.acol[pivot_row_pos];
                    let nonzero_pos = self.find_nonzero(row, c);
                    if nonzero_pos == -1 {
                        let val = pivot_row_scale.wrapping_mul(self.avalue[pivot_row_pos]) % k;
                        if val != 0 {
                            self.add_nonzero(row, c, val);
                        }
                    } else {
                        let np = nonzero_pos as usize;
                        self.avalue[np] = self.avalue[np]
                            .wrapping_add(pivot_row_scale.wrapping_mul(self.avalue[pivot_row_pos]))
                            % k;
                        if self.avalue[np] == 0 {
                            self.unlink(np);
                        }
                    }
                }
                coliter = next;
            }

            num_pivot += 1;
            self.factor_col_perm.push(pivot_col);
            self.factor_row_perm.push(pivot_row);
            self.col_basis_status[pc] = 1;
            self.row_used[pivot_row as usize] = 1;
            if num_pivot == max_pivot {
                break;
            }
            for i in 0..pivot_row_len as usize {
                let col = self.acol[self.rowpositions[i] as usize] as usize;
                let oldsize = self.rowpos_colsizes[i];
                self.colsize[col] -= 1;
                if self.colsize[col] == 0 {
                    continue;
                }
                if self.colsize[col] < oldsize {
                    pqueue.push((self.colsize[col], col as i32));
                }
            }
        }

        for i in 0..self.num_row {
            if self.row_used[i] == 1 {
                continue;
            }
            if self.rhs[i] != 0 {
                return;
            }
        }

        let mut solution: Vec<SolutionEntry> = Vec::with_capacity(self.num_col);
        let num_factor_rows = self.factor_row_perm.len();
        let mut basis_swaps: Vec<(usize, i32)> = Vec::new();
        for i in (0..num_factor_rows).rev() {
            let row = self.factor_row_perm[i] as usize;
            self.iterstack.push(self.rowroot[row]);
            while let Some(rowpos) = self.iterstack.pop() {
                let rp = rowpos as usize;
                if self.arleft[rp] != -1 {
                    self.iterstack.push(self.arleft[rp]);
                }
                if self.arright[rp] != -1 {
                    self.iterstack.push(self.arright[rp]);
                }
                let col = self.acol[rp] as usize;
                if self.col_basis_status[col] != 0 {
                    continue;
                }
                self.col_basis_status[col] = -1;
                basis_swaps.push((i, col as i32));
            }
        }

        let mut basis_swap_pos = 0;
        loop {
            let mut performed_basis_swap = false;
            solution.clear();
            for i in (0..num_factor_rows).rev() {
                let row = self.factor_row_perm[i] as usize;
                let mut solval: u32 = 0;
                for e in 0..solution.len() {
                    let SolutionEntry { index, weight } = solution[e];
                    let pos = self.find_nonzero(row, index);
                    if pos != -1 {
                        solval = solval.wrapping_add(self.avalue[pos as usize].wrapping_mul(weight));
                    }
                }
                solval = self.rhs[row].wrapping_add(k).wrapping_sub(solval % k);
                let col = self.factor_col_perm[i];
                let pos = self.find_nonzero(row, col);
                let col_val_inverse = inverse(k, self.avalue[pos as usize]);
                solval = solval.wrapping_mul(col_val_inverse) % k;
                if solval != 0 {
                    solution.push(SolutionEntry { index: col, weight: solval });
                }
            }
            report_solution(&mut solution);
            if basis_swap_pos < basis_swaps.len() {
                let (basis_index, entering_col) = basis_swaps[basis_swap_pos];
                let leaving_col = self.factor_col_perm[basis_index];
                self.factor_col_perm[basis_index] = entering_col;
                self.col_basis_status[entering_col as usize] = 1;
                self.col_basis_status[leaving_col as usize] = 0;
                performed_basis_swap = true;
                basis_swap_pos += 1;
            }
            if !performed_basis_swap {
                break;
            }
        }
    }
}
