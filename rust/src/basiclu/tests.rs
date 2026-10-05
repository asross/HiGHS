//! Factorize, solve and update small matrices; check residuals.

use super::*;

/// A BASICLU instance with its arrays, grown on REALLOCATE like IPX does
struct Inst {
    istore: Vec<Int>,
    xstore: Vec<f64>,
    li: Vec<Int>,
    lx: Vec<f64>,
    ui: Vec<Int>,
    ux: Vec<f64>,
    wi: Vec<Int>,
    wx: Vec<f64>,
}

impl Inst {
    fn new(m: Int) -> Inst {
        let mut s = Inst {
            istore: vec![0; istore_len(m)],
            xstore: vec![0.0; xstore_len(m)],
            li: vec![0; 1],
            lx: vec![0.0; 1],
            ui: vec![0; 1],
            ux: vec![0.0; 1],
            wi: vec![0; 1],
            wx: vec![0.0; 1],
        };
        initialize(m, &mut s.istore, &mut s.xstore);
        s.xstore[MEMORYL] = 1.0;
        s.xstore[MEMORYU] = 1.0;
        s.xstore[MEMORYW] = 1.0;
        s
    }

    /// Run `f` on the loaded instance until it does not ask for memory
    fn run(&mut self, mut f: impl FnMut(&mut Lu, bool) -> Int) -> Int {
        let mut again = false;
        loop {
            let mut lu = Lu::load(
                &mut self.istore,
                &mut self.xstore,
                &mut self.li,
                &mut self.lx,
                &mut self.ui,
                &mut self.ux,
                &mut self.wi,
                &mut self.wx,
            )
            .unwrap();
            let status = f(&mut lu, again);
            lu.save(status);
            if status != REALLOCATE {
                return status;
            }
            again = true;
            for (add, mem, i, x) in [
                (ADD_MEMORYL, MEMORYL, &mut self.li, &mut self.lx),
                (ADD_MEMORYU, MEMORYU, &mut self.ui, &mut self.ux),
                (ADD_MEMORYW, MEMORYW, &mut self.wi, &mut self.wx),
            ] {
                if self.xstore[add] > 0.0 {
                    let n = (self.xstore[mem] + self.xstore[add]) as usize;
                    i.resize(n, 0);
                    x.resize(n, 0.0);
                    self.xstore[mem] = n as f64;
                }
            }
        }
    }
}

/// Columnwise matrix
struct Mat {
    m: usize,
    begin: Vec<Int>,
    end: Vec<Int>,
    index: Vec<Int>,
    value: Vec<f64>,
}

impl Mat {
    fn from_dense(a: &[Vec<f64>]) -> Mat {
        let m = a.len();
        let mut s = Mat {
            m,
            begin: vec![],
            end: vec![],
            index: vec![],
            value: vec![],
        };
        for j in 0..m {
            s.begin.push(s.index.len() as Int);
            for (i, row) in a.iter().enumerate() {
                if row[j] != 0.0 {
                    s.index.push(i as Int);
                    s.value.push(row[j]);
                }
            }
            s.end.push(s.index.len() as Int);
        }
        s
    }

    fn mul(&self, x: &[f64], trans: bool) -> Vec<f64> {
        let mut y = vec![0.0; self.m];
        for j in 0..self.m {
            for p in self.begin[j] as usize..self.end[j] as usize {
                let i = self.index[p] as usize;
                if trans {
                    y[j] += self.value[p] * x[i];
                } else {
                    y[i] += self.value[p] * x[j];
                }
            }
        }
        y
    }
}

/// Deterministic pseudo random numbers in [0,1)
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Sparse nonsingular test matrix with a permuted strong diagonal
fn test_matrix(m: usize, density: f64, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = Rng(seed);
    let mut a = vec![vec![0.0; m]; m];
    for (i, row) in a.iter_mut().enumerate() {
        for x in row.iter_mut() {
            if rng.next() < density {
                *x = rng.next() * 2.0 - 1.0;
            }
        }
        row[(i * 7 + 3) % m] = 10.0 + rng.next();
    }
    a
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn factorize(inst: &mut Inst, b: &Mat) -> Int {
    inst.run(|lu, again| lu.factorize(&b.begin, &b.end, &b.index, &b.value, again))
}

fn check_solves(inst: &mut Inst, b: &Mat, seed: u64) {
    let mut rng = Rng(seed);
    for &trans in b"NT" {
        let rhs: Vec<f64> = (0..b.m).map(|_| rng.next()).collect();
        let mut x = vec![0.0; b.m];
        inst.run(|lu, _| {
            lu.solve_dense(Some(&rhs), &mut x, trans);
            OK
        });
        let r = b.mul(&x, trans == b'T');
        assert!(
            max_diff(&r, &rhs) < 1e-10,
            "dense residual {}",
            max_diff(&r, &rhs)
        );

        // sparse solve with the same right-hand side
        let irhs: Vec<Int> = (0..b.m as Int).collect();
        let mut xs = vec![0.0; b.m];
        let mut ilhs = vec![0; b.m];
        let mut nz = 0;
        inst.run(|lu, _| {
            lu.solve_sparse(&irhs, &rhs, &mut nz, &mut ilhs, &mut xs, trans);
            OK
        });
        assert!(
            max_diff(&x, &xs) < 1e-9,
            "sparse vs dense {}",
            max_diff(&x, &xs)
        );
    }
}

#[test]
fn factorize_and_solve() {
    for (m, density, seed) in [(1, 0.0, 1), (5, 0.5, 2), (60, 0.05, 3), (200, 0.4, 4)] {
        let a = test_matrix(m, density, seed);
        let b = Mat::from_dense(&a);
        let mut inst = Inst::new(m as Int);
        assert_eq!(factorize(&mut inst, &b), OK);
        assert_eq!(inst.xstore[RANK] as usize, m);
        assert!(inst.xstore[RESIDUAL_TEST] < 1e-12);
        check_solves(&mut inst, &b, seed);
    }
}

#[test]
fn get_factors_reproduce_matrix() {
    let m = 40;
    let a = test_matrix(m, 0.1, 5);
    let b = Mat::from_dense(&a);
    let mut inst = Inst::new(m as Int);
    assert_eq!(factorize(&mut inst, &b), OK);
    let lnz = inst.xstore[LNZ] as usize + m;
    let unz = inst.xstore[UNZ] as usize + m;
    let (mut rowperm, mut colperm) = (vec![0; m], vec![0; m]);
    let (mut lp, mut li, mut lx) = (vec![0; m + 1], vec![0; lnz], vec![0.0; lnz]);
    let (mut up, mut ui, mut ux) = (vec![0; m + 1], vec![0; unz], vec![0.0; unz]);
    inst.run(|lu, _| {
        lu.get_factors(
            Some(&mut rowperm),
            Some(&mut colperm),
            Some((&mut lp, &mut li, &mut lx)),
            Some((&mut up, &mut ui, &mut ux)),
        );
        OK
    });
    // L*U == B[rowperm, colperm]
    let dense = |p: &[Int], i: &[Int], x: &[f64]| {
        let mut d = vec![vec![0.0; m]; m];
        for j in 0..m {
            for k in p[j] as usize..p[j + 1] as usize {
                d[i[k] as usize][j] = x[k];
            }
        }
        d
    };
    let (l, u) = (dense(&lp, &li, &lx), dense(&up, &ui, &ux));
    for i in 0..m {
        for j in 0..m {
            let lu: f64 = (0..m).map(|k| l[i][k] * u[k][j]).sum();
            let bij = a[rowperm[i] as usize][colperm[j] as usize];
            assert!((lu - bij).abs() < 1e-12, "({i},{j}): {lu} vs {bij}");
        }
    }
}

#[test]
fn updates() {
    let m = 50;
    let a = test_matrix(m, 0.08, 6);
    let mut b = Mat::from_dense(&a);
    let mut inst = Inst::new(m as Int);
    assert_eq!(factorize(&mut inst, &b), OK);
    let mut rng = Rng(7);
    for t in 0..30 {
        // replace column j by a random sparse column
        let j = (t * 13 + 5) % m;
        // keep the strong entry where the old column has it (well
        // conditioned), but change the pattern otherwise
        let old = b.mul(&unit(m, j), false);
        let big = (0..m)
            .max_by(|&x, &y| old[x].abs().total_cmp(&old[y].abs()))
            .unwrap();
        let mut col: Vec<(Int, f64)> = vec![(big as Int, 10.0 + rng.next())];
        for i in 0..m as Int {
            if i != col[0].0 && rng.next() < 0.1 {
                col.push((i, rng.next() - 0.5));
            }
        }
        let (bi, bx): (Vec<Int>, Vec<f64>) = col.iter().copied().unzip();
        let (mut nz, mut ilhs, mut lhs) = (0, vec![0; m], vec![0.0; m]);
        assert_eq!(
            inst.run(|lu, _| lu.solve_for_update(
                &bi,
                &bx,
                Some((&mut nz, &mut ilhs, &mut lhs)),
                b'N'
            )),
            OK
        );
        let xtbl = lhs[j];
        assert_eq!(
            inst.run(|lu, _| lu.solve_for_update(&[j as Int], &[], None, b'T')),
            OK
        );
        if inst.run(|lu, _| lu.update(xtbl)) != OK {
            break; // singular update: stop
        }
        assert!(inst.xstore[PIVOT_ERROR] < 1e-8);

        // the new matrix: column j replaced
        let mut d: Vec<Vec<f64>> = (0..m)
            .map(|i| (0..m).map(|c| b.mul(&unit(m, c), false)[i]).collect())
            .collect();
        for row in d.iter_mut() {
            row[j] = 0.0;
        }
        for &(i, x) in &col {
            d[i as usize][j] = x;
        }
        b = Mat::from_dense(&d);
        check_solves(&mut inst, &b, t as u64);
    }
    assert!(inst.xstore[NUPDATE] > 10.0);
}

fn unit(m: usize, j: usize) -> Vec<f64> {
    let mut e = vec![0.0; m];
    e[j] = 1.0;
    e
}

#[test]
fn singular_matrix() {
    let mut a = test_matrix(20, 0.1, 8);
    for row in a.iter_mut() {
        row[4] = 0.0; // empty column
    }
    let b = Mat::from_dense(&a);
    let mut inst = Inst::new(20);
    assert_eq!(factorize(&mut inst, &b), WARNING_SINGULAR_MATRIX);
    assert_eq!(inst.xstore[RANK], 19.0);
}

#[test]
fn object_interface() {
    use super::ffi::*;
    let m = 30;
    let a = test_matrix(m, 0.1, 9);
    let b = Mat::from_dense(&a);
    // SAFETY: the object is initialized before use and freed at the end;
    // all arrays have the lengths the C interface requires
    unsafe {
        let mut obj: BasicluObject = std::mem::zeroed();
        assert_eq!(basiclu_obj_initialize(&mut obj, m as Int), OK);
        assert_eq!(
            basiclu_obj_factorize(
                &mut obj,
                b.begin.as_ptr(),
                b.end.as_ptr(),
                b.index.as_ptr(),
                b.value.as_ptr()
            ),
            OK
        );

        // dense solve, in place (rhs == lhs)
        let rhs: Vec<f64> = (0..m).map(|i| i as f64 + 1.0).collect();
        let mut x = rhs.clone();
        assert_eq!(
            basiclu_obj_solve_dense(&mut obj, x.as_ptr(), x.as_mut_ptr(), b'N'),
            OK
        );
        assert!(max_diff(&b.mul(&x, false), &rhs) < 1e-10);

        // sparse solve: solution in obj.lhs
        let irhs: Vec<Int> = (0..m as Int).collect();
        assert_eq!(
            basiclu_obj_solve_sparse(&mut obj, m as Int, irhs.as_ptr(), rhs.as_ptr(), b'T'),
            OK
        );
        let y = std::slice::from_raw_parts(obj.lhs, m);
        assert!(max_diff(&b.mul(y, true), &rhs) < 1e-10);

        // replace column 0 by column 0 scaled by 2
        let col: Vec<(Int, f64)> = (b.begin[0]..b.end[0])
            .map(|p| (b.index[p as usize], 2.0 * b.value[p as usize]))
            .collect();
        let (bi, bx): (Vec<Int>, Vec<f64>) = col.into_iter().unzip();
        assert_eq!(
            basiclu_obj_solve_for_update(
                &mut obj,
                bi.len() as Int,
                bi.as_ptr(),
                bx.as_ptr(),
                b'N',
                1
            ),
            OK
        );
        let xtbl = *obj.lhs;
        assert!((xtbl - 2.0).abs() < 1e-12);
        let j: Int = 0;
        assert_eq!(
            basiclu_obj_solve_for_update(&mut obj, 0, &j, std::ptr::null(), b'T', 0),
            OK
        );
        assert_eq!(basiclu_obj_update(&mut obj, xtbl), OK);
        let mut x = rhs.clone();
        assert_eq!(
            basiclu_obj_solve_dense(&mut obj, rhs.as_ptr(), x.as_mut_ptr(), b'N'),
            OK
        );
        let mut bx2 = b.mul(&x, false);
        for p in b.begin[0]..b.end[0] {
            bx2[b.index[p as usize] as usize] += b.value[p as usize] * x[0];
        }
        assert!(max_diff(&bx2, &rhs) < 1e-10);

        // get_factors needs a fresh factorization
        let mut rowperm = vec![0; m];
        assert_eq!(
            basiclu_obj_get_factors(
                &mut obj,
                rowperm.as_mut_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut()
            ),
            ERROR_INVALID_CALL
        );
        basiclu_obj_free(&mut obj);
        assert_eq!(
            basiclu_obj_factorize(
                &mut obj,
                b.begin.as_ptr(),
                b.end.as_ptr(),
                b.index.as_ptr(),
                b.value.as_ptr()
            ),
            ERROR_INVALID_OBJECT
        );
    }
}
