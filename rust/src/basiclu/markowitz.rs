//! lu_markowitz.c: search for a pivot element with small Markowitz cost.
//!
//! An eligible pivot must be nonzero and satisfy abs(piv) >= abstol and
//! abs(piv) >= reltol * max[pivot column]. From all eligible pivots search
//! for one that minimizes mc := (nnz[pivot row]-1) * (nnz[pivot column]-1).
//! The search terminates when maxsearch rows or columns with eligible pivots
//! have been searched. The cheapest pivot found is stored in pivot_row,
//! pivot_col.
//!
//! If the active submatrix has a column with column count 0, it is chosen
//! immediately with pivot_row = -1. If no pivot that is nonzero and >=
//! abstol is found, then pivot_col = pivot_row = -1 (cannot happen since
//! lu_pivot erases columns whose maximum drops below abstol).
//!
//! U. Suhl, L. Suhl, "Computing Sparse LU Factorizations for Large-Scale
//! Linear Programming Bases", ORSA Journal on Computing (1990).

use super::file::list_move;
use super::{Int, Lu};

impl Lu<'_> {
    pub(crate) fn markowitz(&mut self) {
        let m = self.m;
        let mu = m as usize;
        let wbegin = &*self.wbegin;
        let wend = &*self.wend;
        let windex = &*self.windex;
        let wvalue = &*self.wvalue;
        let colcount_flink = &*self.colcount_flink;
        let rowcount_flink = &mut *self.rowcount_flink;
        let rowcount_blink = &mut *self.rowcount_blink;
        let colmax = &*self.col_pivot;
        let abstol = self.abstol;
        let reltol = self.reltol;
        let maxsearch = self.maxsearch;
        let search_rows = self.search_rows != 0;
        let nz_start = if search_rows {
            self.min_colnz.min(self.min_rownz)
        } else {
            self.min_colnz
        };

        // integers for Markowitz cost must be 64 bit to prevent overflow
        let mm = m as i64;

        let mut pivot_row: Int = -1; // row of best pivot so far
        let mut pivot_col: Int = -1; // col of best pivot so far
        let mut mc_best: i64 = mm * mm; // Markowitz cost of best pivot so far
        let mut nsearch: Int = 0; // count rows/columns searched
        let mut min_colnz: Int = -1; // minimum col count in active submatrix
        let mut min_rownz: Int = -1; // minimum row count in active submatrix
        debug_assert!(nz_start >= 1);

        'done: {
            // If the active submatrix contains empty columns, choose one and
            // return with pivot_row = -1.
            if colcount_flink[mu] != m {
                pivot_col = colcount_flink[mu];
                break 'done;
            }

            for nz in nz_start..=m {
                // Search columns with nz nonzeros.
                let mut j = colcount_flink[mu + nz as usize];
                while j < m {
                    let ju = j as usize;
                    if min_colnz == -1 {
                        min_colnz = nz;
                    }
                    let cmx = colmax[ju];
                    if cmx == 0.0 || cmx < abstol {
                        j = colcount_flink[ju];
                        continue;
                    }
                    let tol = abstol.max(reltol * cmx);
                    for pos in wbegin[ju] as usize..wend[ju] as usize {
                        let x = wvalue[pos].abs();
                        if x == 0.0 || x < tol {
                            continue;
                        }
                        let i = windex[pos] as usize;
                        let nz1 = nz as i64;
                        let nz2 = (wend[mu + i] - wbegin[mu + i]) as i64;
                        let mc = (nz1 - 1) * (nz2 - 1);
                        if mc < mc_best {
                            mc_best = mc;
                            pivot_row = i as Int;
                            pivot_col = j;
                            if search_rows && mc_best <= (nz1 - 1) * (nz1 - 1) {
                                break 'done;
                            }
                        }
                    }
                    // We have seen at least one eligible pivot in column j.
                    nsearch += 1;
                    if nsearch >= maxsearch {
                        break 'done;
                    }
                    j = colcount_flink[ju];
                }

                if !search_rows {
                    continue;
                }

                // Search rows with nz nonzeros.
                let mut i = rowcount_flink[mu + nz as usize];
                while i < m {
                    let iu = i as usize;
                    if min_rownz == -1 {
                        min_rownz = nz;
                    }
                    // rowcount_flink[i] might be changed below, so keep a copy
                    let inext = rowcount_flink[iu];
                    let mut cheap = false; // row has entries with Markowitz cost < MC?
                    let mut found = false; // eligible pivot found?
                    for pos in wbegin[mu + iu] as usize..wend[mu + iu] as usize {
                        let j = windex[pos] as usize;
                        let nz1 = nz as i64;
                        let nz2 = (wend[j] - wbegin[j]) as i64;
                        let mc = (nz1 - 1) * (nz2 - 1);
                        if mc >= mc_best {
                            continue;
                        }
                        cheap = true;
                        let cmx = colmax[j];
                        if cmx == 0.0 || cmx < abstol {
                            continue;
                        }
                        // find position of pivot in column file
                        let mut where_ = wbegin[j] as usize;
                        while windex[where_] != i {
                            where_ += 1;
                        }
                        let x = wvalue[where_].abs();
                        if x >= abstol && x >= reltol * cmx {
                            found = true;
                            mc_best = mc;
                            pivot_row = i;
                            pivot_col = j as Int;
                            if mc_best <= nz1 * (nz1 - 1) {
                                break 'done;
                            }
                        }
                    }
                    // If row i has cheap entries but none of them is
                    // numerically acceptable, then don't search the row again
                    // until updated.
                    if cheap && !found {
                        list_move(i, m + 1, rowcount_flink, rowcount_blink, m, None);
                    } else {
                        nsearch += 1;
                        if nsearch >= maxsearch {
                            break 'done;
                        }
                    }
                    i = inext;
                }
            }
        }

        self.pivot_row = pivot_row;
        self.pivot_col = pivot_col;
        self.nsearch_pivot += nsearch;
        if min_colnz >= 0 {
            self.min_colnz = min_colnz;
        }
        if min_rownz >= 0 {
            self.min_rownz = min_rownz;
        }
    }
}
