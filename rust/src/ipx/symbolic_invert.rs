//! symbolic_invert.h/.cc: the # structural nonzeros per row and column of
//! inverse(B), from the block triangular form of B (only used for debugging
//! output and statistics).

use super::model::Model;
use super::sparse_matrix::{copy_columns, transpose, SparseMatrix};
use super::sparse_utils::{augmenting_path, depth_first_search};
use super::Int;

/// std::default_random_engine of libc++ (minstd_rand0, seed 1) with
/// std::uniform_int_distribution<Int>(0, m-1), so that the permutation is
/// the one the C++ code computes.
struct MinstdRand0(u64);

impl MinstdRand0 {
    fn next(&mut self) -> u32 {
        self.0 = self.0 * 16807 % 2147483647;
        self.0 as u32
    }

    /// uniform_int_distribution(0, m-1): libc++'s independent bits engine
    /// over the engine range [1, 2^31-2], which for w <= 30 bits draws one
    /// engine value below a multiple of 2^w and keeps its low w bits
    fn uniform(&mut self, m: u32) -> u32 {
        let rp = m; // b - a + 1
        if rp == 1 {
            return 0;
        }
        let mut w = 32 - rp.leading_zeros() - 1;
        if rp & (u32::MAX >> (32 - w)) != 0 {
            w += 1;
        }
        const ENGINE_RANGE: u32 = 2147483646; // max - min + 1
        let y0 = (ENGINE_RANGE >> w) << w;
        let mask0 = u32::MAX >> (32 - w);
        loop {
            let u = loop {
                let u = self.next() - 1;
                if u < y0 {
                    break u;
                }
            };
            let s = u & mask0;
            if s < rp {
                return s;
            }
        }
    }
}

fn random_permute(basis: &[Int]) -> Vec<Int> {
    let m = basis.len();
    let mut re = MinstdRand0(1);
    let mut permuted = basis.to_vec();
    for k in 0..m {
        let r = re.uniform(m as u32) as usize;
        permuted.swap(k, r);
    }
    permuted
}

fn matching(model: &Model, basis: &[Int]) -> Vec<Int> {
    let m = model.rows();
    let n = model.cols();
    let ai = model.ai();
    let ap = &ai.colptr;
    let mut jmatch = vec![-1; m];
    let mut cheap = ap[..n + m].to_vec();
    let mut marked = vec![-1; n + m];
    let mut work = vec![0; m];
    let mut work2 = vec![0; m + 1];
    let mut work3 = vec![0; m + 1];
    // singleton columns first
    for pass in 0..2 {
        for &j in basis {
            let ju = j as usize;
            let singleton = ap[ju + 1] == ap[ju] + 1;
            if singleton == (pass == 0) {
                augmenting_path(
                    j,
                    ap,
                    &ai.rowidx,
                    &mut jmatch,
                    &mut cheap,
                    &mut marked,
                    &mut work,
                    &mut work2,
                    &mut work3,
                );
            }
        }
    }
    jmatch
}

fn blockperm(ai: &SparseMatrix, jmatch: &[Int], bt: &SparseMatrix) -> Vec<Vec<Int>> {
    let m = ai.rows() as usize;
    let mut istack = vec![0; m];
    let mut marked = vec![0; m];
    let mut work = vec![0; m];
    let mut top = m as Int;
    for i in 0..m {
        if marked[i] != 1 {
            top = depth_first_search(
                i as Int,
                &ai.colptr,
                &ai.rowidx,
                Some(jmatch),
                top,
                &mut istack,
                &mut marked,
                1,
                &mut work,
            );
        }
    }

    let mut blocks = Vec::new();
    let mut rowperm = vec![0; m];
    top = m as Int;
    for &i in &istack {
        if marked[i as usize] != 2 {
            let end = top;
            top = depth_first_search(
                i,
                &bt.colptr,
                &bt.rowidx,
                None,
                top,
                &mut rowperm,
                &mut marked,
                2,
                &mut work,
            );
            blocks.push(rowperm[top as usize..end as usize].to_vec());
        }
    }
    blocks.reverse();
    blocks
}

fn coarsened_graph(bt: &SparseMatrix, blocks: &[Vec<Int>]) -> SparseMatrix {
    let m = bt.rows() as usize;
    let nb = blocks.len();
    let mut map2block = vec![-1; m];
    for (b, block) in blocks.iter().enumerate() {
        for &i in block {
            map2block[i as usize] = b as Int;
        }
    }
    let mut c = SparseMatrix::new(nb as Int, 0);
    let mut marked = vec![-1; m];
    for (k, block) in blocks.iter().enumerate() {
        for &i in block {
            for p in bt.begin(i as usize)..bt.end(i as usize) {
                let b = map2block[bt.index(p)];
                if marked[b as usize] != k as Int {
                    marked[b as usize] = k as Int;
                    c.push_back(b, 1.0);
                }
            }
        }
        c.add_column();
    }
    c
}

/// Counts the structural nonzeros per row (rowcounts, by basis position)
/// and column of inverse(B).
pub fn symbolic_invert(
    model: &Model,
    basis: &[Int],
    rowcounts: Option<&mut [Int]>,
    colcounts: Option<&mut [Int]>,
) {
    let ai = model.ai();
    let m = ai.rows() as usize;

    let jmatch = matching(model, &random_permute(basis));
    let bt = transpose(&copy_columns(ai, &jmatch));

    let blocks = blockperm(ai, &jmatch, &bt);
    let mut c = coarsened_graph(&bt, &blocks);

    let nb = blocks.len();
    let mut stack = vec![0; nb];
    let mut marked = vec![-1; nb];
    let mut work = vec![0; nb];

    // # entries reachable from block b in graph c
    let mut reach = |c: &SparseMatrix, b: usize, marked: &mut [Int]| -> Int {
        let top = depth_first_search(
            b as Int,
            &c.colptr,
            &c.rowidx,
            None,
            nb as Int,
            &mut stack,
            marked,
            b as Int,
            &mut work,
        );
        stack[top as usize..nb]
            .iter()
            .map(|&s| blocks[s as usize].len() as Int)
            .sum()
    };

    if let Some(rowcounts) = rowcounts {
        let mut jcount = vec![-1; ai.cols() as usize];
        for b in 0..nb {
            let nz = reach(&c, b, &mut marked);
            for &i in &blocks[b] {
                jcount[jmatch[i as usize] as usize] = nz;
            }
        }
        for p in 0..m {
            rowcounts[p] = jcount[basis[p] as usize];
        }
    }

    if let Some(colcounts) = colcounts {
        c = transpose(&c);
        marked.fill(-1);
        for b in 0..nb {
            let nz = reach(&c, b, &mut marked);
            for &i in &blocks[b] {
                colcounts[i as usize] = nz;
            }
        }
    }
}
