//! The dual simplex ratio test CHUZC of highs/simplex/HEkkDualRow.cpp: the
//! bound-flipping ratio test (BFRT) with EXPAND, and the dual/flip updates.
//!
//! The packed row, the ratio test data and the free list are owned here.
//! The scalars (workCount, workTheta, ...) stay in the C++ HEkkDualRow,
//! where HEkkDual reads and writes them, and are passed in per call along
//! with the HEkk arrays.

use crate::util::fma::ClangFma;

use std::collections::BTreeSet;

use crate::hvector::{K_HIGHS_TINY, K_HIGHS_ZERO};

const K_HIGHS_INF: f64 = f64::INFINITY;
const K_INITIAL_TOTAL_CHANGE: f64 = 1e-12;
const K_INITIAL_REMAIN_THETA: f64 = 1e100;
const K_MAX_SELECT_THETA: f64 = 1e18;

/// An (iCol, value) pair of workData, laid out as std::pair<int, double>
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct WorkPair {
    pub col: i32,
    pub value: f64,
}

/// The constraint matrix (column-wise) as slices
pub struct AMatrix<'a> {
    pub start: &'a [i32],
    pub index: &'a [i32],
    pub value: &'a [f64],
}

impl AMatrix<'_> {
    fn num_col(&self) -> usize {
        self.start.len() - 1
    }
}

/// Why chooseFinal failed, with the workCount and thetas for the C++ debug
/// report
#[derive(Debug, PartialEq)]
pub enum ChooseFail {
    /// No change in a pass of the quadratic BFRT search
    NoChange { work_count: usize, select_theta: f64, remain_theta: f64 },
    /// No group identified
    NoGroup { work_count: usize, select_theta: f64 },
}

/// The outcome of a successful chooseFinal
#[derive(Debug, PartialEq)]
pub struct Chosen {
    pub work_pivot: i32,
    pub work_alpha: f64,
    pub work_theta: f64,
    /// Number of BFRT flips, held as (iCol, move * range) in work_data
    pub work_count: usize,
}

#[derive(Default)]
pub struct DualRow {
    pub pack_index: Vec<i32>,
    pub pack_value: Vec<f64>,
    /// Index-value pairs for the ratio test
    pub work_data: Vec<WorkPair>,
    /// Pointers into work_data for the degenerate nodes in BFRT
    work_group: Vec<usize>,
    /// The nonbasic free columns
    free_list: BTreeSet<i32>,
}

fn small_alpha_tolerance(update_count: i32) -> f64 {
    if update_count < 10 {
        1e-9
    } else if update_count < 20 {
        3e-8
    } else {
        1e-6
    }
}

impl DualRow {
    /// Allocate for a slice (or the whole row) of `size` variables
    pub fn setup_slice(&mut self, size: usize) {
        self.pack_index.resize(size, 0);
        self.pack_value.resize(size, 0.0);
        self.work_data.resize(size, WorkPair::default());
    }

    pub fn clear_freelist(&mut self) {
        self.free_list.clear();
    }

    /// Pack the indices and values of a row (offset by num_col for row_ep)
    /// after the first `pack_count`; returns the new pack count
    pub fn choose_makepack(
        &mut self,
        pack_count: usize,
        row_index: &[i32],
        row_array: &[f64],
        offset: i32,
    ) -> usize {
        let n = row_index.len();
        let pack_index = &mut self.pack_index[pack_count..pack_count + n];
        let pack_value = &mut self.pack_value[pack_count..pack_count + n];
        for ((pi, pv), &index) in pack_index.iter_mut().zip(pack_value).zip(row_index) {
            *pi = index + offset;
            *pv = row_array[index as usize];
        }
        pack_count + n
    }

    /// Determine the candidates for CHUZC: returns (workCount, workTheta)
    pub fn choose_possible(
        &mut self,
        pack_count: usize,
        work_delta: f64,
        update_count: i32,
        td: f64,
        work_move: &[i8],
        work_dual: &[f64],
    ) -> (usize, f64) {
        let ta = small_alpha_tolerance(update_count);
        let move_out: i32 = if work_delta < 0.0 { -1 } else { 1 };
        let mut work_theta = K_HIGHS_INF;
        let mut work_count = 0;
        for (&i_col, &value) in self.pack_index[..pack_count].iter().zip(&self.pack_value[..pack_count]) {
            let mv = work_move[i_col as usize] as i32;
            // move_out * move is +/-1 or 0, so its product is exact in
            // either order
            let alpha = value * (move_out * mv) as f64;
            // Whether alpha passes is unpredictable, so the pair is stored
            // and counted by a select (work_count <= the entry's position
            // < pack_count), leaving one rarely taken branch for the theta
            // update, which black_box keeps from becoming selects
            let candidate = alpha > ta;
            self.work_data[work_count] = WorkPair { col: i_col, value: alpha };
            work_count += candidate as usize;
            let relax = work_dual[i_col as usize].mul_add_c(mv as f64, td);
            if candidate & (work_theta * alpha > relax) {
                work_theta = std::hint::black_box(relax / alpha);
            }
        }
        (work_count, work_theta)
    }

    /// Join the candidates of `other` to those of this row; returns the
    /// new workCount
    pub fn choose_joinpack(&mut self, work_count: usize, other: &DualRow, other_count: usize) -> usize {
        self.work_data[work_count..work_count + other_count].copy_from_slice(&other.work_data[..other_count]);
        work_count + other_count
    }

    /// Section 1 of chooseFinal: reduce the candidates by large step BFRT;
    /// returns the reduced workCount
    pub fn choose_final_reduce(
        &mut self,
        full_count: usize,
        work_theta: f64,
        work_delta: f64,
        work_move: &[i8],
        work_dual: &[f64],
        work_range: &[f64],
    ) -> usize {
        let work_data = &mut self.work_data[..full_count];
        let mut work_count = 0;
        let mut total_change = 0.0;
        let total_delta = work_delta.abs();
        let mut select_theta = 10f64.mul_add_c(work_theta, 1e-7);
        loop {
            for i in work_count..full_count {
                let WorkPair { col, value: alpha } = work_data[i];
                let i_col = col as usize;
                let tight = work_move[i_col] as f64 * work_dual[i_col];
                if alpha * select_theta >= tight {
                    work_data.swap(work_count, i);
                    work_count += 1;
                    total_change = work_range[i_col].mul_add_c(alpha, total_change);
                }
            }
            select_theta *= 10.0;
            if total_change >= total_delta || work_count == full_count {
                break;
            }
        }
        work_count
    }

    /// Sections 2-4 of chooseFinal: choose by small step BFRT, then by large
    /// alpha, and determine the BFRT flips, sorted by column
    #[allow(clippy::too_many_arguments)]
    pub fn choose_final(
        &mut self,
        work_count: usize,
        work_theta: f64,
        work_delta: f64,
        td: f64,
        work_move: &[i8],
        work_dual: &[f64],
        work_range: &[f64],
        num_tot_permutation: &[i32],
    ) -> Result<Chosen, ChooseFail> {
        // 2. Choose by small step BFRT, with the quadratic cost sort
        let work_count = self.choose_final_work_group_quad(work_count, work_theta, work_delta, td, work_move, work_dual, work_range)?;
        // 3. Choose large alpha
        let (break_index, break_group) = self.choose_final_large_alpha(work_count, num_tot_permutation);
        let move_out: i32 = if work_delta < 0.0 { -1 } else { 1 };
        let work_pivot = self.work_data[break_index].col;
        let pivot = work_pivot as usize;
        let work_alpha = self.work_data[break_index].value * (move_out * work_move[pivot] as i32) as f64;
        let work_theta = if work_dual[pivot] * work_move[pivot] as f64 > 0.0 {
            work_dual[pivot] / work_alpha
        } else {
            0.0
        };
        // 4. Determine BFRT flip index: flip all
        let mut work_count = self.work_group[break_group];
        for p in &mut self.work_data[..work_count] {
            let i_col = p.col as usize;
            p.value = work_move[i_col] as f64 * work_range[i_col];
        }
        if work_theta == 0.0 {
            work_count = 0;
        }
        // Sort by column so that the columns of A are accessed in
        // order. The columns are distinct, so the order is that of the
        // C++ pdqsort of the pairs
        self.work_data[..work_count].sort_unstable_by_key(|p| p.col);
        Ok(Chosen { work_pivot, work_alpha, work_theta, work_count })
    }

    /// Identify the groups of degenerate nodes in BFRT by repeated passes
    /// over the candidates; returns the workCount
    #[allow(clippy::too_many_arguments)]
    fn choose_final_work_group_quad(
        &mut self,
        full_count: usize,
        work_theta: f64,
        work_delta: f64,
        td: f64,
        work_move: &[i8],
        work_dual: &[f64],
        work_range: &[f64],
    ) -> Result<usize, ChooseFail> {
        let work_data = &mut self.work_data[..full_count];
        let mut work_count = 0;
        let mut total_change = K_INITIAL_TOTAL_CHANGE;
        let mut select_theta = work_theta;
        let total_delta = work_delta.abs();
        self.work_group.clear();
        self.work_group.push(0);
        let mut prev_work_count = work_count;
        let mut prev_remain_theta = K_INITIAL_REMAIN_THETA;
        let mut prev_select_theta = select_theta;

        while select_theta < K_MAX_SELECT_THETA {
            let mut remain_theta = K_INITIAL_REMAIN_THETA;
            for i in work_count..full_count {
                let WorkPair { col, value } = work_data[i];
                let i_col = col as usize;
                let dual = work_move[i_col] as f64 * work_dual[i_col];
                // Tight satisfy
                if dual <= select_theta * value {
                    work_data.swap(work_count, i);
                    work_count += 1;
                    total_change = value.mul_add_c(work_range[i_col], total_change);
                } else if dual + td < remain_theta * value {
                    remain_theta = (dual + td) / value;
                }
            }
            self.work_group.push(work_count);

            // Update selectTheta with the value of remainTheta;
            select_theta = remain_theta;
            // Check for no change in this loop - to prevent infinite loop
            if work_count == prev_work_count && prev_select_theta == select_theta && prev_remain_theta == remain_theta {
                return Err(ChooseFail::NoChange { work_count, select_theta, remain_theta });
            }
            // Record the values of workCount, remainTheta and selectTheta
            // for the next pass through the loop - to check for the
            // infinite loop condition
            prev_work_count = work_count;
            prev_remain_theta = remain_theta;
            prev_select_theta = select_theta;
            if total_change >= total_delta || work_count == full_count {
                break;
            }
        }
        // Check that at least one group has been identified
        if self.work_group.len() <= 1 {
            return Err(ChooseFail::NoGroup { work_count, select_theta });
        }
        Ok(work_count)
    }

    /// Choose the last group with a large enough alpha, and its largest
    /// alpha (ties to the earlier column in the random permutation);
    /// returns (breakIndex, breakGroup)
    fn choose_final_large_alpha(&self, work_count: usize, num_tot_permutation: &[i32]) -> (usize, usize) {
        let work_data = &self.work_data;
        // std::max and std::min, as written in the C++
        let mut final_compare = 0.0;
        for p in &work_data[..work_count] {
            if final_compare < p.value {
                final_compare = p.value;
            }
        }
        final_compare = if 1.0 < 0.1 * final_compare { 1.0 } else { 0.1 * final_compare };
        let count_group = self.work_group.len() - 1;
        for i_group in (0..count_group).rev() {
            let mut d_max_final = 0.0;
            let mut i_max_final: Option<usize> = None;
            for i in self.work_group[i_group]..self.work_group[i_group + 1] {
                let value = work_data[i].value;
                if d_max_final < value {
                    d_max_final = value;
                    i_max_final = Some(i);
                } else if d_max_final == value {
                    // Candidate alphas are positive, so there is an
                    // incumbent here
                    let j_col = work_data[i_max_final.unwrap()].col as usize;
                    let i_col = work_data[i].col as usize;
                    if num_tot_permutation[i_col] < num_tot_permutation[j_col] {
                        i_max_final = Some(i);
                    }
                }
            }
            // An empty group has no candidate (the C++ would read before
            // workData)
            if let Some(i_max_final) = i_max_final {
                if work_data[i_max_final].value > final_compare {
                    return (i_max_final, i_group);
                }
            }
        }
        // The group holding the largest alpha passes, since finalCompare is
        // at most a tenth of it
        unreachable!("CHUZC: no large alpha")
    }

    /// Flip the bounds of the BFRT columns, accumulating the RHS of the
    /// FTRAN needed to update the primal values in (column, column_index,
    /// column_count); returns the change in the dual objective value
    #[allow(clippy::too_many_arguments)]
    pub fn update_flip(
        &self,
        work_count: usize,
        a: &AMatrix,
        work_dual: &[f64],
        cost_scale: f64,
        nonbasic_move: &mut [i8],
        work_value: &mut [f64],
        work_lower: &[f64],
        work_upper: &[f64],
        column: &mut [f64],
        column_index: &mut [i32],
        column_count: &mut usize,
    ) -> f64 {
        let num_col = a.num_col();
        let mut dual_objective_value_change = 0.0;
        for &WorkPair { col, value: change } in &self.work_data[..work_count] {
            let i_col = col as usize;
            let mut local_dual_objective_change = change * work_dual[i_col];
            local_dual_objective_change *= cost_scale;
            dual_objective_value_change += local_dual_objective_change;
            // HEkk::flipBound
            let mv = -nonbasic_move[i_col];
            nonbasic_move[i_col] = mv;
            work_value[i_col] = if mv == 1 { work_lower[i_col] } else { work_upper[i_col] };
            // HighsSparseMatrix::collectAj. The slack's entry is 1, and
            // fma(change, 1, value0) is value0 + change
            let slack_row = [i_col.wrapping_sub(num_col) as i32];
            let (index, value) = if i_col < num_col {
                let (from, to) = (a.start[i_col] as usize, a.start[i_col + 1] as usize);
                (&a.index[from..to], &a.value[from..to])
            } else {
                (&slack_row[..], &[1.0][..])
            };
            for (&i_row, &value) in index.iter().zip(value) {
                let i_row = i_row as usize;
                let value0 = column[i_row];
                let value1 = change.mul_add_c(value, value0);
                if value0 == 0.0 {
                    column_index[*column_count] = i_row as i32;
                    *column_count += 1;
                }
                column[i_row] = if value1.abs() < K_HIGHS_TINY { K_HIGHS_ZERO } else { value1 };
            }
        }
        dual_objective_value_change
    }

    /// Update the dual values by theta times the packed row; returns the
    /// change in the dual objective value
    pub fn update_dual(
        &self,
        pack_count: usize,
        theta: f64,
        work_dual: &mut [f64],
        work_value: &[f64],
        nonbasic_flag: &[i8],
        cost_scale: f64,
    ) -> f64 {
        let mut dual_objective_value_change = 0.0;
        for (&i_col, &value) in self.pack_index[..pack_count].iter().zip(&self.pack_value[..pack_count]) {
            let i_col = i_col as usize;
            work_dual[i_col] = (-theta).mul_add_c(value, work_dual[i_col]);
            // Identify the change to the dual objective
            let delta_dual = theta * value;
            let local_value = work_value[i_col];
            let mut local_dual_objective_change = nonbasic_flag[i_col] as f64 * (-local_value * delta_dual);
            local_dual_objective_change *= cost_scale;
            dual_objective_value_change += local_dual_objective_change;
        }
        dual_objective_value_change
    }

    /// Create the list of nonbasic free columns
    pub fn create_freelist(&mut self, nonbasic_flag: &[i8], work_lower: &[f64], work_upper: &[f64]) {
        self.free_list.clear();
        for (i, ((&flag, &lower), &upper)) in nonbasic_flag.iter().zip(work_lower).zip(work_upper).enumerate() {
            if flag != 0 && -lower >= K_HIGHS_INF && upper >= K_HIGHS_INF {
                self.free_list.insert(i as i32);
            }
        }
    }

    /// Set nonbasicMove for the free columns with a nonzero entry in the
    /// pivotal row, preventing their dual values from being changed
    pub fn create_freemove(
        &self,
        update_count: i32,
        work_delta: f64,
        a: &AMatrix,
        row_ep: &[f64],
        nonbasic_move: &mut [i8],
    ) {
        if self.free_list.is_empty() {
            return;
        }
        let ta = small_alpha_tolerance(update_count);
        let move_out = if work_delta < 0.0 { -1.0 } else { 1.0 };
        let num_col = a.num_col();
        for &i_var in &self.free_list {
            let i_var = i_var as usize;
            // HighsSparseMatrix::computeDot
            let alpha = if i_var < num_col {
                let (from, to) = (a.start[i_var] as usize, a.start[i_var + 1] as usize);
                let mut result = 0.0;
                for (&i_row, &value) in a.index[from..to].iter().zip(&a.value[from..to]) {
                    result = row_ep[i_row as usize].mul_add_c(value, result);
                }
                result
            } else {
                row_ep[i_var - num_col]
            };
            if alpha.abs() > ta {
                nonbasic_move[i_var] = if alpha * move_out > 0.0 { 1 } else { -1 };
            }
        }
    }

    /// Reset nonbasicMove for the free columns
    pub fn delete_freemove(&self, nonbasic_move: &mut [i8]) {
        for &i_var in &self.free_list {
            nonbasic_move[i_var as usize] = 0;
        }
    }

    /// Remove a column from the free list
    pub fn delete_freelist(&mut self, i_var: i32) {
        self.free_list.remove(&i_var);
    }

    /// The (contribution to the) Devex weight from the packed row
    pub fn compute_devex_weight(&self, pack_count: usize, nonbasic_flag: &[i8], devex_index: &[i32]) -> f64 {
        let mut computed_edge_weight = 0.0;
        for (&vr_n, &value) in self.pack_index[..pack_count].iter().zip(&self.pack_value[..pack_count]) {
            let vr_n = vr_n as usize;
            if nonbasic_flag[vr_n] == 0 {
                continue;
            }
            let pv = devex_index[vr_n] as f64 * value;
            if pv != 0.0 {
                computed_edge_weight = pv.mul_add_c(pv, computed_edge_weight);
            }
        }
        computed_edge_weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three candidates with ratios 1, 2 and 3 (and ranges 1): the
    /// step to the first breakpoint does not exhaust delta, so the first
    /// is flipped and the second, with the larger alpha, enters
    #[test]
    fn bfrt_flips_then_enters() {
        let mut row = DualRow::default();
        row.setup_slice(3);
        let work_move = [1i8, 1, 1];
        let work_dual = [1.0, 4.0, 3.0];
        let work_range = [1.0, 1.0, 1.0];
        let perm = [0, 1, 2];
        let mut pack = row.choose_makepack(0, &[0, 1, 2], &[1.0, 2.0, 1.0], 0);
        assert_eq!(pack, 3);
        let (count, theta) = row.choose_possible(pack, 1.5, 0, 1e-7, &work_move, &work_dual);
        assert_eq!(count, 3);
        assert_eq!(theta, (1.0 + 1e-7) / 1.0);
        let count = row.choose_final_reduce(count, theta, 1.5, &work_move, &work_dual, &work_range);
        let chosen = row.choose_final(count, theta, 1.5, 1e-7, &work_move, &work_dual, &work_range, &perm).unwrap();
        assert_eq!(chosen.work_pivot, 1);
        assert_eq!(chosen.work_alpha, 2.0);
        assert_eq!(chosen.work_theta, 2.0);
        assert_eq!(chosen.work_count, 1);
        assert_eq!(row.work_data[0], WorkPair { col: 0, value: 1.0 });
        // Dual update along the packed row
        let mut dual = work_dual;
        let change = row.update_dual(pack, 2.0, &mut dual, &[0.0; 3], &[1; 3], 1.0);
        assert_eq!(dual, [-1.0, 0.0, 1.0]);
        assert_eq!(change, 0.0);
        pack -= 1;
        assert_eq!(row.compute_devex_weight(pack, &[1; 3], &[1; 3]), 5.0);
    }
}
