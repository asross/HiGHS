//! HEkkDualRHS (highs/simplex/HEkkDualRHS.cpp): the dual simplex CHUZR
//! and the primal value/infeasibility updates. The list of rows with
//! the greatest primal infeasibilities is owned here; HEkk's arrays come
//! in as slices for each call.

use crate::util::fma::ClangFma;

use crate::hvector::K_HIGHS_ZERO;
use crate::util::random::HighsRandom;
use std::cmp::Ordering;

const K_EXCESSIVE_PRIMAL_VALUE: f64 = 1e25;

/// HEkk's basic primal values and bounds
pub struct Primal<'a> {
    pub base_value: &'a mut [f64],
    pub base_lower: &'a [f64],
    pub base_upper: &'a [f64],
    /// primal_feasibility_tolerance
    pub tp: f64,
    /// info_.store_squared_primal_infeasibility
    pub squared: bool,
}

impl Primal<'_> {
    /// The primal infeasibility of row i, as stored in work_infeasibility
    #[inline]
    fn infeasibility(&self, i: usize) -> f64 {
        let value = self.base_value[i];
        let lower = self.base_lower[i];
        let upper = self.base_upper[i];
        // Selects rather than branches: which case holds is unpredictable
        let (below, above) = (lower - value, value - upper);
        let primal_infeasibility = if value < lower - self.tp {
            below
        } else if value > upper + self.tp {
            above
        } else {
            0.0
        };
        if self.squared {
            primal_infeasibility * primal_infeasibility
        } else {
            primal_infeasibility.abs()
        }
    }
}

#[derive(Default)]
pub struct DualRhs {
    /// Limit for row to be in list with greatest primal infeasibilities
    pub work_cutoff: f64,
    /// Number of rows in the list (negative: dense mode, -num_row)
    pub work_count: i32,
    work_mark: Vec<u8>,
    work_index: Vec<i32>,
    pub work_infeasibility: Vec<f64>,
    part_num: i32,
    part_switch: i32,
    work_partition: Vec<i32>,
}

/// The best row by merit infeasibility / weight over rows, starting at a
/// random position and wrapping round
#[inline]
fn wrapped(count: usize, random_start: usize, mut f: impl FnMut(usize)) {
    for i in random_start..count {
        f(i);
    }
    for i in 0..random_start {
        f(i);
    }
}

impl DualRhs {
    pub fn setup(&mut self, num_row: i32) {
        let n = num_row as usize;
        self.work_mark.resize(n, 0);
        self.work_index.resize(n, 0);
        self.work_infeasibility.resize(n, 0.0);
        self.part_num = 0;
        self.part_switch = 0;
    }

    pub fn choose_normal(
        &mut self,
        edge_weight: &[f64],
        dwork: &mut [f64],
        random: &mut HighsRandom,
        num_row: i32,
    ) -> i32 {
        if self.work_count == 0 {
            return -1;
        }
        let infeas = &self.work_infeasibility;
        let mut best_merit = 0.0;
        let mut best_index = -1;
        let mut consider = |i_row: usize| {
            let my_infeas = infeas[i_row];
            let my_weight = edge_weight[i_row];
            // The tests of the C++ in the other order (they have no side
            // effects): whether a row is infeasible is unpredictable, but
            // its merit rarely beats the best. black_box keeps LLVM from
            // turning the rare update into selects, which would chain the
            // division through every row
            if best_merit * my_weight < my_infeas {
                if my_infeas > K_HIGHS_ZERO {
                    best_merit = std::hint::black_box(my_infeas / my_weight);
                    best_index = i_row as i32;
                }
            }
        };
        if self.work_count < 0 {
            // DENSE mode
            let n = -self.work_count;
            let random_start = random.integer_below(n) as usize;
            wrapped(n as usize, random_start, &mut consider);
            best_index
        } else {
            // SPARSE mode
            let n = self.work_count;
            let random_start = random.integer_below(n) as usize;
            let index = &self.work_index;
            wrapped(n as usize, random_start, |i| consider(index[i] as usize));
            let create_list_again = if best_index == -1 {
                self.work_cutoff > 0.0
            } else {
                best_merit <= self.work_cutoff * 0.99
            };
            if create_list_again {
                self.create_infeas_list(0.0, edge_weight, dwork, num_row);
                best_index = self.choose_normal(edge_weight, dwork, random, num_row);
            }
            best_index
        }
    }

    /// Returns the number of rows chosen
    pub fn choose_multi_global(
        &self,
        ch_index: &mut [i32],
        edge_weight: &[f64],
        random: &mut HighsRandom,
    ) -> i32 {
        let ch_limit = ch_index.len();
        ch_index.fill(-1);
        let choose_check = ch_limit * 2;
        // (merit, row) pairs are distinct, so any correct sort gives
        // pdqsort's order
        let less = |a: &(f64, i32), b: &(f64, i32)| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(Ordering::Equal)
                .then(a.1.cmp(&b.1))
        };
        let mut set_p: Vec<(f64, i32)> = Vec::with_capacity(choose_check);
        let infeas = &self.work_infeasibility;
        let mut cutoff_merit = 0.0;
        let mut consider = |i_row: usize| {
            if infeas[i_row] > K_HIGHS_ZERO {
                let my_infeas = infeas[i_row];
                let my_weight = edge_weight[i_row];
                if cutoff_merit * my_weight < my_infeas {
                    set_p.push((-my_infeas / my_weight, i_row as i32));
                    if set_p.len() >= choose_check {
                        set_p.sort_unstable_by(less);
                        set_p.truncate(ch_limit);
                        cutoff_merit = -set_p.last().unwrap().0;
                    }
                }
            }
        };
        if self.work_count < 0 {
            // DENSE mode
            let n = -self.work_count;
            let random_start = random.integer_below(n) as usize;
            wrapped(n as usize, random_start, &mut consider);
        } else {
            // SPARSE mode
            let n = self.work_count;
            let random_start = if n != 0 {
                random.integer_below(n) as usize
            } else {
                0
            };
            let index = &self.work_index;
            wrapped(n as usize, random_start, |i| consider(index[i] as usize));
        }
        set_p.sort_unstable_by(less);
        set_p.truncate(ch_limit);
        for (c, p) in ch_index.iter_mut().zip(&set_p) {
            *c = p.1;
        }
        set_p.len() as i32
    }

    pub fn choose_multi_hyper_graph_auto(
        &mut self,
        ch_index: &mut [i32],
        edge_weight: &[f64],
        random: &mut HighsRandom,
    ) -> i32 {
        if self.part_switch != 0 {
            self.choose_multi_hyper_graph_part(ch_index, edge_weight, random)
        } else {
            self.choose_multi_global(ch_index, edge_weight, random)
        }
    }

    pub fn choose_multi_hyper_graph_part(
        &mut self,
        ch_index: &mut [i32],
        edge_weight: &[f64],
        random: &mut HighsRandom,
    ) -> i32 {
        let ch_limit = ch_index.len();
        // Force to use partition method, unless doesn't exist
        if self.part_num as usize != ch_limit {
            self.part_switch = 0;
            return self.choose_multi_global(ch_index, edge_weight, random);
        }
        ch_index.fill(-1);
        if self.work_count == 0 {
            return 0;
        }
        let infeas = &self.work_infeasibility;
        let partition = &self.work_partition;
        let mut best_merit = vec![0.0; ch_limit];
        let mut best_index = vec![-1; ch_limit];
        let mut consider = |i_row: usize| {
            if infeas[i_row] > K_HIGHS_ZERO {
                let i_part = partition[i_row] as usize;
                let my_infeas = infeas[i_row];
                let my_weight = edge_weight[i_row];
                if best_merit[i_part] * my_weight < my_infeas {
                    best_merit[i_part] = my_infeas / my_weight;
                    best_index[i_part] = i_row as i32;
                }
            }
        };
        if self.work_count < 0 {
            let n = -self.work_count;
            let random_start = random.integer_below(n) as usize;
            wrapped(n as usize, random_start, &mut consider);
        } else {
            let n = self.work_count;
            let random_start = random.integer_below(n) as usize;
            let index = &self.work_index;
            wrapped(n as usize, random_start, |i| consider(index[i] as usize));
        }
        let mut count = 0;
        for &i_row in best_index.iter().filter(|&&i| i != -1) {
            ch_index[count] = i_row;
            count += 1;
        }
        count as i32
    }

    /// Subtract theta times the column from the primal values, returning
    /// false if excessive values are created
    pub fn update_primal(
        &mut self,
        column_count: i32,
        column_index: &[i32],
        column_array: &[f64],
        theta: f64,
        p: &mut Primal,
        num_row: i32,
    ) -> bool {
        let mut num_excessive_primal = 0;
        let in_dense = column_count < 0 || column_count as f64 > 0.4 * num_row as f64;
        let to_entry = if in_dense { num_row } else { column_count } as usize;
        for i_entry in 0..to_entry {
            let i_row = if in_dense {
                i_entry
            } else {
                column_index[i_entry] as usize
            };
            // Fused by clang
            p.base_value[i_row] = (-theta).mul_add_c(column_array[i_row], p.base_value[i_row]);
            self.work_infeasibility[i_row] = p.infeasibility(i_row);
            if p.base_value[i_row] <= -K_EXCESSIVE_PRIMAL_VALUE
                || p.base_value[i_row] >= K_EXCESSIVE_PRIMAL_VALUE
            {
                num_excessive_primal += 1;
            }
        }
        num_excessive_primal == 0
    }

    pub fn update_pivots(&mut self, i_row: i32, value: f64, p: &mut Primal) {
        let i_row = i_row as usize;
        p.base_value[i_row] = value;
        self.work_infeasibility[i_row] = p.infeasibility(i_row);
    }

    pub fn update_infeas_list(&mut self, column_index: &[i32], edge_weight: &[f64]) {
        // DENSE mode: disabled
        if self.work_count < 0 {
            return;
        }
        let infeas = &self.work_infeasibility;
        for &i_row in column_index {
            let r = i_row as usize;
            if self.work_mark[r] == 0
                && (if self.work_cutoff <= 0.0 {
                    infeas[r] != 0.0
                } else {
                    infeas[r] > edge_weight[r] * self.work_cutoff
                })
            {
                self.work_index[self.work_count as usize] = i_row;
                self.work_count += 1;
                self.work_mark[r] = 1;
            }
        }
    }

    pub fn create_array_of_primal_infeasibilities(&mut self, p: &Primal, num_row: i32) {
        for i in 0..num_row as usize {
            self.work_infeasibility[i] = p.infeasibility(i);
        }
    }

    /// dwork is HEkk's scattered_dual_edge_weight_, used as workspace. Only
    /// its value at icutoff after the partial sort is used, so it does not
    /// matter that the order of the rest differs from std::nth_element
    pub fn create_infeas_list(
        &mut self,
        column_density: f64,
        edge_weight: &[f64],
        dwork: &mut [f64],
        num_row: i32,
    ) {
        let n = num_row as usize;
        let infeas = &self.work_infeasibility;
        let mark = &mut self.work_mark[..n];
        let index = &mut self.work_index;

        // 1. Build the full list
        mark.fill(0);
        let mut count = 0usize;
        self.work_cutoff = 0.0;
        for i_row in 0..n {
            if infeas[i_row] != 0.0 {
                mark[i_row] = 1;
                index[count] = i_row as i32;
                count += 1;
            }
        }

        // 2. See if it worth to try to go sparse
        //    (Many candidates, really sparse RHS)
        if count as f64 > f64::max(num_row as f64 * 0.01, 500.0) && column_density < 0.05 {
            let icutoff = f64::max(count as f64 * 0.001, 500.0) as usize;
            let mut max_merit = 0.0;
            let mut i_put = 0;
            for i_row in 0..n {
                if mark[i_row] != 0 {
                    let my_merit = infeas[i_row] / edge_weight[i_row];
                    if max_merit < my_merit {
                        max_merit = my_merit;
                    }
                    dwork[i_put] = -my_merit;
                    i_put += 1;
                }
            }
            dwork[..count].select_nth_unstable_by(icutoff, f64::total_cmp);
            let cut_merit = -dwork[icutoff];
            let (a, b) = (max_merit * 0.99999, cut_merit * 1.00001);
            self.work_cutoff = if b < a { b } else { a };

            // Create again
            mark.fill(0);
            count = 0;
            for i_row in 0..n {
                if infeas[i_row] >= edge_weight[i_row] * self.work_cutoff {
                    index[count] = i_row as i32;
                    count += 1;
                    mark[i_row] = 1;
                }
            }

            // Reduce by drop smaller
            if count as f64 > icutoff as f64 * 1.5 {
                // Firstly take up "icutoff" number of elements
                let full_count = count;
                count = icutoff;
                for i in icutoff..full_count {
                    let i_row = index[i] as usize;
                    if infeas[i_row] > edge_weight[i_row] * cut_merit {
                        index[count] = i_row as i32;
                        count += 1;
                    } else {
                        mark[i_row] = 0;
                    }
                }
            }
        }
        self.work_count = count as i32;

        // 3. If there are still too many candidates: disable them
        if self.work_count as f64 > 0.2 * num_row as f64 {
            self.work_count = -num_row;
            self.work_cutoff = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infeas_list_and_choose() {
        let n = 3000;
        let mut rhs = DualRhs::default();
        rhs.setup(n);
        let mut value: Vec<f64> = (0..n).map(|i| ((i * 7919) % 1000) as f64 - 500.0).collect();
        let lower = vec![0.0; n as usize];
        let upper = vec![1.0; n as usize];
        let weight = vec![1.0; n as usize];
        let mut dwork = vec![0.0; n as usize];
        let mut p = Primal {
            base_value: &mut value,
            base_lower: &lower,
            base_upper: &upper,
            tp: 1e-7,
            squared: false,
        };
        rhs.create_array_of_primal_infeasibilities(&p, n);
        rhs.create_infeas_list(0.0, &weight, &mut dwork, n);
        // Sparse list of the ~500 largest of the 2997 infeasibilities
        assert!(rhs.work_count > 0 && rhs.work_cutoff > 0.0);
        let max = rhs.work_infeasibility.iter().cloned().fold(0.0, f64::max);
        let mut random = HighsRandom::new(0);
        let r = rhs.choose_normal(&weight, &mut dwork, &mut random, n);
        assert_eq!(rhs.work_infeasibility[r as usize], max);
        // Making the chosen row feasible
        rhs.update_pivots(r, 0.5, &mut p);
        assert_eq!(rhs.work_infeasibility[r as usize], 0.0);
        let mut ch = vec![0; 8];
        let count = rhs.choose_multi_global(&mut ch, &weight, &mut random);
        assert_eq!(count, 8);
        for w in ch.windows(2) {
            assert!(rhs.work_infeasibility[w[0] as usize] >= rhs.work_infeasibility[w[1] as usize]);
        }
    }
}
