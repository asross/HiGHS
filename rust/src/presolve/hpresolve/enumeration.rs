//! The solution enumeration of HPresolve::enumerateSolutions: for rows of
//! at most 8 unfixed binaries, enumerate the feasible assignments by
//! branching and propagation on the global domain, then add the cliques
//! and bounds valid in every solution.
//!
//! The clique table is the Rust one, the global domain is changed through
//! its Rust view (mip/domain.rs); bound changes that fix a binary call into
//! the C++ glue of the clique table and implications.

use super::*;
use crate::mip::clique::{CliqueVar, Ctx as CliqueCtx};
use crate::mip::domain::{Ctx as DomCtx, DomChg, Reason, LOWER, REASON_BRANCHING, UPPER};
use crate::util::hash::HighsHash;
use crate::util::random::HighsRandom;

const MAX_ROW_SIZE: usize = 8;
const MAX_NUM_ROWS_CHECKED: i32 = 400;
const MAX_NUM_SOLUTIONS: usize = 1 << MAX_ROW_SIZE;
const MAX_PERCENTAGE_ROW_OVERLAP: usize = 50;
const MAX_NUM_FAILS: i32 = 6;

const BRANCHING: Reason = Reason { kind: REASON_BRANCHING, index: 0 };

/// A candidate row: (-score, -score2, random, row, signature)
#[derive(Clone, Copy)]
struct CandidateRow {
    score: f64,
    score2: f64,
    random: i32,
    row: i32,
    signature: u32,
}

/// getBinaryRow: the unfixed columns of a row (sorted) if they are at most
/// 8 binaries; the count found so far otherwise
fn binary_row(d: &mut DomCtx, row: i32, binvars: &mut [i32; MAX_ROW_SIZE]) -> (bool, usize) {
    let dom = d.dom();
    let mut numnzs = 0;
    for &col in dom.row(row as usize).0 {
        if dom.is_fixed(col as usize) {
            continue;
        }
        if !dom.is_binary(col as usize) || numnzs >= MAX_ROW_SIZE {
            return (false, numnzs);
        }
        binvars[numnzs] = col;
        numnzs += 1;
    }
    if numnzs == 0 {
        return (false, 0);
    }
    // distinct columns: any sort gives pdqsort's order
    binvars[..numnzs].sort_unstable();
    (true, numnzs)
}

/// The number of common entries of two sorted rows
fn row_overlap(a: &[i32], b: &[i32]) -> usize {
    let (mut i, mut j, mut overlap) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        if a[i] < b[j] {
            i += 1;
        } else if a[i] > b[j] {
            j += 1;
        } else {
            i += 1;
            j += 1;
            overlap += 1;
        }
    }
    overlap
}

impl Presolve<'_> {
    /// The rows to enumerate, by score (compileRows, sorted,
    /// removeSimilarRows)
    fn enumeration_rows(&self, d: &mut DomCtx) -> Vec<CandidateRow> {
        let mut rows: Vec<CandidateRow> = Vec::with_capacity(self.num_row as usize);
        let mut binvars = [0i32; MAX_ROW_SIZE];
        let mut random = HighsRandom::new(self.opt.random_seed as u32);
        for i in 0..self.num_row {
            if d.dom().is_redundant_row(i as usize) {
                continue;
            }
            let (ok, numnzs) = binary_row(d, i, &mut binvars);
            if !ok {
                continue;
            }
            let (mut score, mut score2) = (0i64, 0i32);
            let mut signature = 0u32;
            for &col in &binvars[..numnzs] {
                let s = self.probing_score(self.cliquetable(), col);
                score += s.0;
                score2 += s.1;
                signature |= 1 << (col.highs_hash() >> 59);
            }
            rows.push(CandidateRow {
                score: -(score as f64 / numnzs as f64),
                score2: -(score2 as f64 / numnzs as f64),
                random: random.integer(),
                row: i,
                signature,
            });
        }
        // std::tuple's operator<; the keys are distinct (by row)
        rows.sort_unstable_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap()
                .then(a.score2.partial_cmp(&b.score2).unwrap())
                .then(a.random.cmp(&b.random))
                .then(a.row.cmp(&b.row))
                .then(a.signature.cmp(&b.signature))
        });

        // remove rows that overlap more than half with a better one
        if rows.len() <= 1 {
            return rows;
        }
        let mut binvars2 = [0i32; MAX_ROW_SIZE];
        let mut num_rows_accepted = 0;
        let mut num_rows_removed = 0;
        for i in 0..rows.len() - 1 {
            let r = rows[i].row;
            if r == -1 {
                continue;
            }
            num_rows_accepted += 1;
            if num_rows_accepted >= MAX_NUM_ROWS_CHECKED {
                break;
            }
            let (_, numnzs) = binary_row(d, r, &mut binvars);
            let mut num_rows_active = 0;
            let old_num_rows_removed = num_rows_removed;
            for ii in i + 1..rows.len() {
                if rows[ii].row == -1 {
                    continue;
                }
                num_rows_active += 1;
                if rows[i].signature & rows[ii].signature == 0 {
                    continue;
                }
                let (_, numnzs2) = binary_row(d, rows[ii].row, &mut binvars2);
                let overlap = row_overlap(&binvars[..numnzs], &binvars2[..numnzs2]);
                if (100 * overlap) / numnzs.min(numnzs2) > MAX_PERCENTAGE_ROW_OVERLAP {
                    num_rows_removed += 1;
                    rows[ii].row = -1;
                }
            }
            if num_rows_active - num_rows_removed + old_num_rows_removed <= 1 {
                break;
            }
        }
        if num_rows_removed > 0 {
            rows.retain(|r| r.row != -1);
        }
        rows.truncate(MAX_NUM_ROWS_CHECKED as usize);
        rows
    }

    /// The enumeration of enumerateSolutions, between prepareProbing and
    /// finaliseProbing; infeasibility (profiling stopped) is an error
    pub(crate) fn enumeration_loop(&mut self) -> R {
        let env = self.host.mip_env();
        // SAFETY: the view of the global domain, valid during the loop (no
        // presolve reduction is made)
        let mut d = unsafe { DomCtx::new(env.domain) };
        let num_col = self.num_col as usize;
        let rows = self.enumeration_rows(&mut d);

        let mut solutions = vec![[0u8; MAX_NUM_SOLUTIONS]; MAX_ROW_SIZE];
        let mut vars = [0i32; MAX_ROW_SIZE];
        // (domain change stack size, changed columns) at each branching
        let mut branches = [(0usize, 0usize); MAX_ROW_SIZE];
        let mut worst_case_bounds = vec![0i32; num_col];
        let mut worst_case_lower = vec![INF; num_col];
        let mut worst_case_upper = vec![-INF; num_col];
        let (mut col_lower, mut col_upper) = {
            let dom = d.dom();
            (dom.col_lower[..].to_vec(), dom.col_upper[..].to_vec())
        };

        let find_branch_var = |d: &mut DomCtx, vars: &[i32]| {
            let dom = d.dom();
            vars.iter().copied().find(|&v| !dom.is_fixed(v as usize)).unwrap_or(-1)
        };
        let infeasible = |p: &Self| -> R {
            p.host.profiling(false, 1);
            Err(Stop::PrimalInfeasible)
        };

        let mut num_fails = 0;
        for r in &rows {
            let row = r.row;
            if d.dom().is_redundant_row(row as usize) {
                continue;
            }
            let (ok, num_vars) = binary_row(&mut d, row, &mut vars);
            if !ok {
                continue;
            }
            let vars = &vars[..num_vars];

            // enumerate the solutions of the row by depth-first branching
            let mut num_branches: i32 = -1;
            let mut num_worst_case_bounds = 0usize;
            let mut num_solutions = 0usize;
            let mut min_num_active_cols = num_vars;
            let mut max_num_active_cols = 0usize;
            loop {
                let mut backtrack = d.infeasible();
                if !backtrack {
                    backtrack = find_branch_var(&mut d, vars) < 0;
                    if backtrack {
                        // handleSolution
                        d.propagate();
                        if !d.infeasible() {
                            let dom = d.dom();
                            // updateWorstCaseBounds: true if no tighter than
                            // the bounds before the row
                            let update = |col: usize, wl: &mut [f64], wu: &mut [f64]| {
                                wl[col] = std_min(wl[col], dom.col_lower[col]);
                                wu[col] = std_max(wu[col], dom.col_upper[col]);
                                wl[col] <= col_lower[col] && wu[col] >= col_upper[col]
                            };
                            if num_solutions == 0 {
                                for &col in dom.changed_cols() {
                                    worst_case_bounds[num_worst_case_bounds] = col;
                                    num_worst_case_bounds += 1;
                                    update(col as usize, &mut worst_case_lower, &mut worst_case_upper);
                                }
                            } else {
                                let mut i = 0;
                                while i < num_worst_case_bounds {
                                    let col = worst_case_bounds[i] as usize;
                                    if !dom.is_changed_col(col)
                                        || update(col, &mut worst_case_lower, &mut worst_case_upper)
                                    {
                                        // removeWorstCaseBounds
                                        worst_case_lower[col] = INF;
                                        worst_case_upper[col] = -INF;
                                        worst_case_bounds[i] = worst_case_bounds[num_worst_case_bounds - 1];
                                        worst_case_bounds[num_worst_case_bounds - 1] = 0;
                                        num_worst_case_bounds -= 1;
                                    } else {
                                        i += 1;
                                    }
                                }
                            }
                            let mut num_active_cols = 0;
                            for (k, &v) in vars.iter().enumerate() {
                                let sol_value = if dom.col_lower[v as usize] == 0.0 { 0 } else { 1 };
                                solutions[k][num_solutions] = sol_value;
                                if sol_value != 0 {
                                    num_active_cols += 1;
                                }
                            }
                            min_num_active_cols = min_num_active_cols.min(num_active_cols);
                            max_num_active_cols = max_num_active_cols.max(num_active_cols);
                            num_solutions += 1;
                        }
                    }
                }
                if !backtrack {
                    // doBranch
                    let branchvar = find_branch_var(&mut d, vars);
                    num_branches += 1;
                    let dom = d.dom();
                    branches[num_branches as usize] = (dom.stack().len(), dom.changed_cols().len());
                    d.change_bound(DomChg { boundval: 0.0, column: branchvar, boundtype: UPPER }, BRANCHING);
                } else {
                    // doBacktrack
                    while num_branches >= 0 {
                        let (num_domain_changes, num_changed_cols) = branches[num_branches as usize];
                        let mut dom = d.dom();
                        let domchg = dom.stack()[num_domain_changes];
                        dom.backtrack(false);
                        dom.clear_changed_cols(num_changed_cols);
                        if domchg.boundtype == UPPER {
                            d.change_bound(DomChg { boundval: 1.0, column: domchg.column, boundtype: LOWER }, BRANCHING);
                            break;
                        }
                        branches[num_branches as usize] = (0, 0);
                        num_branches -= 1;
                    }
                    if num_branches < 0 {
                        break;
                    }
                }
            }

            if num_solutions == 0 {
                return infeasible(self);
            }

            let old_num_changed_cols = d.dom().changed_cols().len();
            let old_num_cliques = self.cliquetable().num_cliques_total();
            let old_num_substitutions = self.cliquetable().substitutions.len();

            let add_clique = |p: &Self, clique: &mut [CliqueVar], equality: bool| {
                // SAFETY: the live table; no other borrow of it is held
                let mut c = unsafe { CliqueCtx::new(p.mip.expect("MIP presolve").cliquetable, env.cdom) };
                c.add_clique(&env.cmip, clique, equality, IINF);
            };

            if max_num_active_cols == 1 || min_num_active_cols == num_vars - 1 {
                let val = (max_num_active_cols == 1) as i32;
                let mut clique: Vec<CliqueVar> = vars.iter().map(|&v| CliqueVar::new(v, val)).collect();
                add_clique(self, &mut clique, min_num_active_cols == max_num_active_cols);
                if d.infeasible() {
                    return infeasible(self);
                }
            }

            for &col in &worst_case_bounds[..num_worst_case_bounds] {
                let c = col as usize;
                if worst_case_lower[c] > d.dom().col_lower[c] {
                    d.change_bound(DomChg { boundval: worst_case_lower[c], column: col, boundtype: LOWER }, Reason::UNSPECIFIED);
                    if d.infeasible() {
                        return infeasible(self);
                    }
                }
                if worst_case_upper[c] < d.dom().col_upper[c] {
                    d.change_bound(DomChg { boundval: worst_case_upper[c], column: col, boundtype: UPPER }, Reason::UNSPECIFIED);
                    if d.infeasible() {
                        return infeasible(self);
                    }
                }
                worst_case_lower[c] = INF;
                worst_case_upper[c] = -INF;
            }
            worst_case_bounds[..num_worst_case_bounds].fill(0);

            // identical and complementary columns
            for i in 0..num_vars - 1 {
                for ii in i + 1..num_vars {
                    let (s1, s2) = (&solutions[i][..num_solutions], &solutions[ii][..num_solutions]);
                    let val2 = if s1 == s2 {
                        1
                    } else if s1.iter().zip(s2).all(|(&a, &b)| a == 1 - b) {
                        0
                    } else {
                        continue;
                    };
                    let mut clique = [CliqueVar::new(vars[i], 0), CliqueVar::new(vars[ii], val2)];
                    add_clique(self, &mut clique, true);
                    if d.infeasible() {
                        return infeasible(self);
                    }
                }
            }

            let num_changed_cols = {
                let mut dom = d.dom();
                let changed = dom.changed_cols();
                for &col in changed {
                    col_lower[col as usize] = dom.col_lower[col as usize];
                    col_upper[col as usize] = dom.col_upper[col as usize];
                }
                let n = changed.len();
                dom.clear_changed_cols(0);
                n
            };

            if num_changed_cols != old_num_changed_cols
                || self.cliquetable().num_cliques_total() != old_num_cliques
                || self.cliquetable().substitutions.len() != old_num_substitutions
            {
                num_fails = 0;
            } else {
                num_fails += 1;
                if num_fails > MAX_NUM_FAILS {
                    break;
                }
            }
        }
        Ok(())
    }
}
