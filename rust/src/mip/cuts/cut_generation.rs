//! HighsCutGeneration (highs/mip/HighsCutGeneration.cpp): cuts from single
//! row relaxations by lifted cover inequalities and the complemented MIR
//! heuristic, and their postprocessing. Step for step as the C++,
//! including where clang fuses `a + b * c`.

use super::integers::{frexp, integral_scale, is_integral, ldexp, nearest_integer};
use super::round::{MinCpp, SepaRound, K_HIGHS_INF};
use super::sort::{partial_sort, partition, pdqsort, pdqsort_branchless, upper_bound};
use crate::util::cdouble::CDouble;
use crate::util::hash::pair_hash;
use crate::util::random::HighsRandom;

const K_HIGHS_TINY: f64 = 1e-14;

/// What cut generation needs from the MIP solver.
pub trait CutEnv {
    /// lpRelaxation.isColIntegral (a column or a row slack)
    fn is_integral(&self, idx: usize) -> bool;
    /// the global domain's bounds of a column
    fn glb(&self, col: usize) -> f64;
    fn gub(&self, col: usize) -> f64;
    /// the LP solution value of a column
    fn sol(&self, col: usize) -> f64;
    /// lpRelaxation.numCols()
    fn num_lp_cols(&self) -> usize;
    /// HighsCutPool::addCut
    fn add_cut(&mut self, inds: &mut [i32], vals: &mut [f64], rhs: f64, integral: bool, conflict: bool) -> i32;
    /// nodequeue.numNodesDown/Up
    fn num_nodes_down(&self, col: i32) -> i64;
    fn num_nodes_up(&self, col: i32) -> i64;
}

impl CutEnv for SepaRound {
    #[inline]
    fn is_integral(&self, idx: usize) -> bool {
        self.cols[idx].integral
    }
    #[inline]
    fn glb(&self, col: usize) -> f64 {
        self.col_lower.at(col)
    }
    #[inline]
    fn gub(&self, col: usize) -> f64 {
        self.col_upper.at(col)
    }
    #[inline]
    fn sol(&self, col: usize) -> f64 {
        self.col_value.get()[col]
    }
    fn num_lp_cols(&self) -> usize {
        self.num_col
    }
    fn add_cut(&mut self, inds: &mut [i32], vals: &mut [f64], rhs: f64, integral: bool, conflict: bool) -> i32 {
        SepaRound::add_cut(self, inds, vals, rhs, integral, conflict)
    }
    fn num_nodes_down(&self, col: i32) -> i64 {
        // SAFETY: a C++ query
        unsafe { (self.host.num_nodes_down)(self.host.ctx, col) }
    }
    fn num_nodes_up(&self, col: i32) -> i64 {
        // SAFETY: a C++ query
        unsafe { (self.host.num_nodes_up)(self.host.ctx, col) }
    }
}

/// HighsHashHelpers::hash(std::pair<HighsInt, HighsInt>)
#[inline]
fn hash_pair(a: i32, b: i32) -> u64 {
    let (a, b) = (a as u32, b as u32);
    pair_hash::<1>(a, b) ^ (pair_hash::<0>(a, b) >> 32)
}

/// fast_floor of HighsCutGeneration.cpp
#[inline]
fn fast_floor(x: f64) -> f64 {
    let t = x as i64;
    (t - ((x < t as f64) as i64)) as f64
}

pub struct CutGeneration {
    pub randgen: HighsRandom,
    cover: Vec<usize>,
    coverweight: CDouble,
    lambda: CDouble,
    pub upper: Vec<f64>,
    pub solval: Vec<f64>,
    complementation: Vec<u8>,
    isintegral: Vec<u8>,
    feastol: f64,
    epsilon: f64,

    /// the row being worked on (the caller's vectors, swapped in)
    inds: Vec<i32>,
    vals: Vec<f64>,
    rhs: CDouble,
    integral_support: bool,
    integral_coefficients: bool,
    rowlen: usize,
    initial_scale: f64,

    integerinds: Vec<usize>,
    /// (vals, solval, upper) of integerinds, packed for cmir_efficacy
    intdata: Vec<[f64; 3]>,
    deltas: Vec<f64>,

    tmp_vals: Vec<f64>,
    tmp_inds: Vec<i32>,
    tmp_complementation: Vec<u8>,
    tmp_solval: Vec<f64>,

    // work space of the lifting functions
    s: Vec<f64>,
    coverflag: Vec<i8>,
    ca: Vec<CDouble>,
    cu: Vec<CDouble>,
    cm: Vec<CDouble>,
    cancel_nzs: Vec<usize>,
}

impl Default for CutGeneration {
    fn default() -> Self {
        CutGeneration {
            randgen: HighsRandom::from_state(1),
            cover: Vec::new(),
            coverweight: CDouble::default(),
            lambda: CDouble::default(),
            upper: Vec::new(),
            solval: Vec::new(),
            complementation: Vec::new(),
            isintegral: Vec::new(),
            feastol: 0.0,
            epsilon: 0.0,
            inds: Vec::new(),
            vals: Vec::new(),
            rhs: CDouble::default(),
            integral_support: false,
            integral_coefficients: false,
            rowlen: 0,
            initial_scale: 0.0,
            integerinds: Vec::new(),
            intdata: Vec::new(),
            deltas: Vec::new(),
            tmp_vals: Vec::new(),
            tmp_inds: Vec::new(),
            tmp_complementation: Vec::new(),
            tmp_solval: Vec::new(),
            s: Vec::new(),
            coverflag: Vec::new(),
            ca: Vec::new(),
            cu: Vec::new(),
            cm: Vec::new(),
            cancel_nzs: Vec::new(),
        }
    }
}

impl CutGeneration {
    pub fn new(seed: u32, feastol: f64, epsilon: f64) -> Self {
        CutGeneration { randgen: HighsRandom::new(seed), feastol, epsilon, ..Default::default() }
    }

    /// A new cut generator with the C++ constructor's seed:
    /// random_seed + numLpIterations + cutpool.getNumCuts()
    pub fn reset(&mut self, seed: u32, feastol: f64, epsilon: f64) {
        self.randgen = HighsRandom::new(seed);
        self.feastol = feastol;
        self.epsilon = epsilon;
    }

    fn determine_cover<E: CutEnv>(&mut self, env: &E, lp_sol: bool) -> bool {
        let feastol = self.feastol;
        if self.rhs <= 10.0 * feastol {
            return false;
        }
        let mut cover = std::mem::take(&mut self.cover);
        cover.clear();
        cover.reserve(self.rowlen);
        for j in 0..self.rowlen {
            if self.isintegral[j] == 0 {
                continue;
            }
            if lp_sol && self.solval[j] <= feastol {
                continue;
            }
            cover.push(j);
        }
        let max_cover_size = cover.len();
        let mut coversize = 0;
        let r = self.randgen.integer();
        self.coverweight = CDouble::from(0.0);
        let (vals, upper, solval, inds) = (&self.vals, &self.upper, &self.solval, &self.inds);
        if lp_sol {
            coversize = partition(&mut cover, |&j| solval[j] >= upper[j] - feastol);
            for &j in &cover[..coversize] {
                self.coverweight += vals[j] * upper[j];
            }
            pdqsort(&mut cover[coversize..max_cover_size], |&i, &j| {
                if upper[i] < 1.5 && upper[j] > 1.5 {
                    return true;
                }
                if upper[i] > 1.5 && upper[j] < 1.5 {
                    return false;
                }
                let contribution_a = solval[i] * vals[i];
                let contribution_b = solval[j] * vals[j];
                if contribution_a > contribution_b + feastol {
                    return true;
                }
                if contribution_a < contribution_b - feastol {
                    return false;
                }
                if (vals[i] - vals[j]).abs() <= feastol {
                    return hash_pair(inds[i], r) > hash_pair(inds[j], r);
                }
                vals[i] > vals[j]
            });
        } else {
            let comp = &self.complementation;
            pdqsort(&mut cover[coversize..max_cover_size], |&i, &j| {
                if solval[i] > feastol && solval[j] <= feastol {
                    return true;
                }
                if solval[i] <= feastol && solval[j] > feastol {
                    return false;
                }
                let num_nodes_a =
                    if comp[i] != 0 { env.num_nodes_down(inds[i]) } else { env.num_nodes_up(inds[i]) };
                let num_nodes_b =
                    if comp[j] != 0 { env.num_nodes_down(inds[j]) } else { env.num_nodes_up(inds[j]) };
                if num_nodes_a > num_nodes_b {
                    return true;
                }
                if num_nodes_a < num_nodes_b {
                    return false;
                }
                hash_pair(inds[i], r) > hash_pair(inds[j], r)
            });
        }

        let minlambda = (10.0 * feastol).max_cpp(feastol * self.rhs.to_f64().abs());
        while coversize != max_cover_size {
            let lambda = (self.coverweight - self.rhs).to_f64();
            if lambda > minlambda {
                break;
            }
            let j = cover[coversize];
            self.coverweight += vals[j] * upper[j];
            coversize += 1;
        }
        if coversize == 0 {
            self.cover = cover;
            return false;
        }
        self.coverweight.renormalize();
        self.lambda = self.coverweight - self.rhs;
        if self.lambda <= minlambda {
            self.cover = cover;
            return false;
        }
        cover.truncate(coversize);
        self.cover = cover;
        true
    }

    fn separate_lifted_knapsack_cover(&mut self) {
        let feastol = self.feastol;
        let epsilon = self.epsilon;
        let coversize = self.cover.len();
        let rowlen = self.rowlen;
        self.s.clear();
        self.s.resize(coversize, 0.0);
        self.coverflag.clear();
        self.coverflag.resize(rowlen, 0);
        let vals = &mut self.vals;
        let cover = &mut self.cover;
        pdqsort_branchless(cover, |&a, &b| vals[a] > vals[b]);

        let mut abartmp = CDouble::from(vals[cover[0]]);
        let mut sigma = self.lambda;
        for i in 1..coversize {
            let delta = abartmp - vals[cover[i]];
            let kdelta = i as f64 * delta;
            if kdelta.to_f64() < sigma.to_f64() {
                abartmp = CDouble::from(vals[cover[i]]);
                sigma -= kdelta;
            } else {
                abartmp -= sigma * (1.0 / i as f64);
                sigma = CDouble::from(0.0);
                break;
            }
        }
        if sigma.to_f64() > 0.0 {
            abartmp = self.rhs / coversize as f64;
        }
        let abar = abartmp.to_f64();

        let mut sum = CDouble::from(0.0);
        let mut cplussize = 0i32;
        for i in 0..coversize {
            sum += abar.min_cpp(vals[cover[i]]);
            self.s[i] = sum.to_f64();
            if vals[cover[i]] > abar + feastol {
                cplussize += 1;
                self.coverflag[cover[i]] = 1;
            } else {
                self.coverflag[cover[i]] = -1;
            }
        }
        let mut halfintegral = false;
        let s = &self.s;
        let mut g = |z: f64| -> f64 {
            let hfrac = z / abar;
            let mut coef = 0.0;
            let mut h = (hfrac + 0.5).floor() as i32;
            if h != 0 && (hfrac - h as f64).abs() * 1.0f64.max_cpp(abar) <= epsilon && h < cplussize {
                halfintegral = true;
                coef = 0.5;
            }
            h = (h - 1).max(0);
            while (h as usize) < coversize {
                if z <= s[h as usize] + feastol {
                    break;
                }
                h += 1;
            }
            coef + h as f64
        };
        self.rhs = CDouble::from((coversize as i32 - 1) as f64);
        for i in 0..rowlen {
            if vals[i] == 0.0 {
                continue;
            }
            if self.coverflag[i] == -1 {
                vals[i] = 1.0;
            } else {
                vals[i] = g(vals[i]);
            }
        }
        if halfintegral {
            self.rhs *= 2.0;
            for v in &mut vals[..rowlen] {
                *v *= 2.0;
            }
        }
        self.integral_support = true;
        self.integral_coefficients = true;
    }

    fn separate_lifted_mixed_binary_cover(&mut self) -> bool {
        self.integral_support = false;
        self.integral_coefficients = false;
        let coversize = self.cover.len();
        let rowlen = self.rowlen;
        self.s.clear();
        self.s.resize(coversize, 0.0);
        self.coverflag.clear();
        self.coverflag.resize(rowlen, 0);
        if coversize == 0 {
            return false;
        }
        for &c in &self.cover {
            self.coverflag[c] = 1;
        }
        let vals = &mut self.vals;
        let epsilon = self.epsilon;
        pdqsort_branchless(&mut self.cover, |&a, &b| vals[a] > vals[b]);
        let cover = &self.cover;
        let lambda = self.lambda;
        let mut sum = CDouble::from(0.0);
        let mut p = coversize;
        for i in 0..coversize {
            if vals[cover[i]] - lambda <= epsilon {
                p = i;
                break;
            }
            sum += vals[cover[i]];
            self.s[i] = sum.to_f64();
        }
        if p == 0 {
            return false;
        }
        let s = &self.s;
        let phi = |a: f64| -> f64 {
            for i in 0..p {
                if a <= (s[i] - lambda).to_f64() {
                    return (i as f64 * lambda).to_f64();
                }
                if a <= s[i] {
                    return ((i + 1) as f64 * lambda + (CDouble::from(a) - s[i])).to_f64();
                }
            }
            (p as f64 * lambda + (CDouble::from(a) - s[p - 1])).to_f64()
        };
        self.rhs = -lambda;
        self.integral_coefficients = false;
        self.integral_support = true;
        for i in 0..rowlen {
            if self.isintegral[i] == 0 {
                if vals[i] < 0.0 {
                    self.integral_support = false;
                } else {
                    vals[i] = 0.0;
                }
                continue;
            }
            if self.coverflag[i] != 0 {
                vals[i] = vals[i].min_cpp(lambda.to_f64());
                self.rhs += vals[i];
            } else {
                vals[i] = phi(vals[i]);
            }
        }
        true
    }

    fn separate_lifted_mixed_integer_cover(&mut self) -> bool {
        self.integral_support = false;
        self.integral_coefficients = false;
        let feastol = self.feastol;
        let epsilon = self.epsilon;
        let rowlen = self.rowlen;
        let coversize = self.cover.len();
        self.coverflag.clear();
        self.coverflag.resize(rowlen, 0);
        for &c in &self.cover {
            self.coverflag[c] = 1;
        }
        {
            let vals = &self.vals;
            pdqsort_branchless(&mut self.cover, |&a, &b| vals[a] > vals[b]);
        }
        let zero = CDouble::from(0.0);
        let (a, u, m) = (&mut self.ca, &mut self.cu, &mut self.cm);
        a.clear();
        a.resize(coversize, zero);
        u.clear();
        u.resize(coversize + 1, zero);
        m.clear();
        m.resize(coversize + 1, zero);
        let mut usum = zero;
        let mut msum = zero;
        let vals = &self.vals;
        let upper = &self.upper;
        let solval = &self.solval;
        let lambda = self.lambda;
        for c in 0..coversize {
            let i = self.cover[c];
            u[c] = usum;
            m[c] = msum;
            a[c] = CDouble::from(vals[i]);
            let ub = upper[i];
            usum += ub;
            msum += ub * a[c];
        }
        u[coversize] = usum;
        m[coversize] = msum;

        let mut lpos: isize = -1;
        let mut bestl_cplusend = 0usize;
        let mut bestl_val = 0.0;
        let mut bestl_at_upper = true;
        for i in 0..coversize {
            let j = self.cover[i];
            let ub = upper[j];
            let at_upper = solval[j] >= ub - feastol;
            if at_upper && !bestl_at_upper {
                continue;
            }
            let mju = ub * vals[j];
            let mu = mju - lambda;
            if mu <= 10.0 * feastol {
                continue;
            }
            if vals[j].abs() < 1000.0 * feastol {
                continue;
            }
            let mudival = (mu / vals[j]).to_f64();
            if is_integral(mudival, feastol) {
                continue;
            }
            let eta = mudival.ceil();
            let ulminusetaplusone = CDouble::from(ub) - eta + 1.0;
            let cplusthreshold = (ulminusetaplusone * vals[j]).to_f64();
            let cplusend = upper_bound(&self.cover, |&i| cplusthreshold > vals[i]);
            let mut mcplus = m[cplusend];
            if i < cplusend {
                mcplus -= mju;
            }
            let jl_val = (mcplus + eta * vals[j]).to_f64();
            if jl_val > bestl_val || (!at_upper && bestl_at_upper) {
                lpos = i as isize;
                bestl_cplusend = cplusend;
                bestl_val = jl_val;
                bestl_at_upper = at_upper;
            }
        }
        if lpos == -1 {
            return false;
        }
        let lpos = lpos as usize;
        let l = self.cover[lpos];
        let al = CDouble::from(vals[l]);
        let upperl = upper[l];
        let mlu = upperl * al;
        let mu = mlu - lambda;

        a.truncate(bestl_cplusend);
        self.cover.truncate(bestl_cplusend);
        u.truncate(bestl_cplusend + 1);
        m.truncate(bestl_cplusend + 1);
        if lpos < bestl_cplusend {
            a.remove(lpos);
            self.cover.remove(lpos);
            u.remove(lpos + 1);
            m.remove(lpos + 1);
            for i in lpos + 1..bestl_cplusend {
                u[i] -= upperl;
                m[i] -= mlu;
            }
        }
        let cplussize = a.len();
        let mudival = (mu / al).to_f64();
        let eta = mudival.ceil();
        let mut r = mu - mudival.floor() * al;
        if r < 0.0 {
            r = CDouble::from(0.0);
        }
        let ulminusetaplusone = CDouble::from(upperl) - eta + 1.0;
        let cplusthreshold = ulminusetaplusone * al;
        let kmin = (eta - upperl - 0.5).floor() as i32;

        let phi_l = |a: f64| -> f64 {
            let mut k = ((a / al.to_f64()) as i64).min(-1);
            while k >= kmin as i64 {
                if a >= (k as f64 * al + r).to_f64() {
                    return (a - (k + 1) as f64 * r).to_f64();
                }
                if a >= (k as f64 * al).to_f64() {
                    return (k as f64 * (al - r)).to_f64();
                }
                k -= 1;
            }
            (kmin as f64 * (al - r)).to_f64()
        };

        let kmax = (upperl - eta + 0.5).floor() as i64;
        let cover = &self.cover;
        let (a, u, m) = (&*a, &*u, &*m);
        let gamma_l = |z: f64| -> f64 {
            for i in 0..cplussize {
                let upperi = upper[cover[i]] as i32;
                for h in 0..=upperi {
                    let mih = m[i] + h as f64 * a[i];
                    let uih = u[i] + h as f64;
                    let mihplusdeltai = mih + a[i] - cplusthreshold;
                    if z <= mihplusdeltai.to_f64() {
                        return (uih * ulminusetaplusone * (al - r)).to_f64();
                    }
                    let mut k = (((z - mihplusdeltai) / al).to_f64() as i64) - 1;
                    while k <= kmax {
                        if z <= (mihplusdeltai + k as f64 * al + r).to_f64() {
                            return ((uih * ulminusetaplusone + k as f64) * (al - r)).to_f64();
                        }
                        if z <= (mihplusdeltai + (k + 1) as f64 * al).to_f64() {
                            return ((uih * ulminusetaplusone) * (al - r) + z - mih - a[i] + cplusthreshold
                                - (k + 1) as f64 * r)
                                .to_f64();
                        }
                        k += 1;
                    }
                }
            }
            let mut p = (((z - m[cplussize]) / al).to_f64() as i64) - 1;
            loop {
                if z <= (m[cplussize] + p as f64 * al + r).to_f64() {
                    return ((u[cplussize] * ulminusetaplusone + p as f64) * (al - r)).to_f64();
                }
                if z <= (m[cplussize] + (p + 1) as f64 * al).to_f64() {
                    return ((u[cplussize] * ulminusetaplusone) * (al - r) + z - m[cplussize] - (p + 1) as f64 * r)
                        .to_f64();
                }
                p += 1;
            }
        };

        let mut rhs = (CDouble::from(upperl) - eta) * r - lambda;
        let mut integral_support = true;
        let vals = &mut self.vals;
        for i in 0..rowlen {
            if vals[i] == 0.0 {
                continue;
            }
            if self.isintegral[i] == 0 {
                if vals[i] < 0.0 {
                    integral_support = false;
                } else {
                    vals[i] = 0.0;
                }
                continue;
            }
            if self.coverflag[i] != 0 {
                vals[i] = -phi_l(-vals[i]);
                rhs += vals[i] * upper[i];
            } else {
                vals[i] = gamma_l(vals[i]);
            }
        }
        let _ = epsilon;
        self.rhs = rhs;
        self.integral_support = integral_support;
        self.integral_coefficients = false;
        true
    }

    #[inline]
    fn update_violation_and_norm(&self, index: usize, aj: f64, violation: &mut f64, norm: &mut f64) {
        let s = self.solval[index];
        *violation = aj.mul_add(s, *violation);
        if aj > 0.0 && s <= self.feastol {
            return;
        }
        if aj < 0.0 && s >= self.upper[index] - self.feastol {
            return;
        }
        *norm = aj.mul_add(aj, *norm);
    }

    /// The efficacy of the MIR cut for delta (NaN-free inputs); None if
    /// f0 or the scale are out of range
    #[inline]
    fn cmir_efficacy(&self, delta: f64, contcontribution: f64, contsqrnorm: f64) -> Option<f64> {
        const MAX_CMIR_SCALE: f64 = 1e6;
        const F0MIN: f64 = 0.005;
        const F0MAX: f64 = 0.995;
        let scale = 1.0 / delta;
        let scalrhs = self.rhs.to_f64() * scale;
        let downrhs = fast_floor(scalrhs);
        let f0 = scalrhs - downrhs;
        if f0 < F0MIN || f0 > F0MAX {
            return None;
        }
        let oneoveroneminusf0 = 1.0 / (1.0 - f0);
        if oneoveroneminusf0 > MAX_CMIR_SCALE {
            return None;
        }
        let contscale = scale * oneoveroneminusf0;
        let mut sqrnorm = contscale * contscale * contsqrnorm;
        let mut viol = contscale.mul_add(contcontribution, -downrhs);
        let feastol = self.feastol;
        for &[v, s, u] in &self.intdata {
            let scalaj = v * scale;
            let downaj = fast_floor(scalaj + K_HIGHS_TINY);
            let fj = scalaj - downaj;
            let aj = downaj + 0.0f64.max_cpp((fj - f0) * oneoveroneminusf0);
            // updateViolationAndNorm
            viol = aj.mul_add(s, viol);
            if aj > 0.0 && s <= feastol {
                continue;
            }
            if aj < 0.0 && s >= u - feastol {
                continue;
            }
            sqrnorm = aj.mul_add(aj, sqrnorm);
        }
        Some(viol / sqrnorm.sqrt())
    }

    fn cmir_cut_generation_heuristic(&mut self, min_efficacy: f64, only_initial_cmir_scale: bool) -> bool {
        const F0MIN: f64 = 0.005;
        const F0MAX: f64 = 0.995;
        let feastol = self.feastol;
        self.integral_support = false;
        self.integral_coefficients = false;
        let mut contcontribution = 0.0;
        let mut contsqrnorm = 0.0;
        self.deltas.clear();
        self.deltas.reserve(self.rowlen + 3);
        self.integerinds.clear();
        self.integerinds.reserve(self.rowlen);
        let mut maxabsdelta = 0.0f64;
        self.complementation.resize(self.rowlen, 0);

        for i in 0..self.rowlen {
            if self.isintegral[i] != 0 {
                self.integerinds.push(i);
                if self.upper[i] < 2.0 * self.solval[i] {
                    self.flip_complementation(i);
                }
                if only_initial_cmir_scale {
                    continue;
                }
                if self.solval[i] > feastol {
                    let delta = self.vals[i].abs();
                    if delta <= 1e-4 || delta == maxabsdelta {
                        continue;
                    }
                    maxabsdelta = maxabsdelta.max_cpp(delta);
                    self.deltas.push(delta);
                }
            } else {
                let v = self.vals[i];
                self.update_violation_and_norm(i, v, &mut contcontribution, &mut contsqrnorm);
            }
        }

        if contsqrnorm == 0.0 && self.deltas.len() > 1 {
            let int_scale = integral_scale(&self.deltas, feastol, K_HIGHS_TINY);
            if int_scale != 0.0 && int_scale <= 1e4 {
                let scalrhs = self.rhs.to_f64() * int_scale;
                let downrhs = fast_floor(scalrhs);
                let f0 = scalrhs - downrhs;
                if (F0MIN..=F0MAX).contains(&f0) {
                    self.deltas.push(1.0 / int_scale);
                }
            }
        }

        self.deltas.push(1.0f64.min_cpp(self.initial_scale));
        if !only_initial_cmir_scale {
            self.deltas.push(maxabsdelta + 1.0f64.min_cpp(self.initial_scale));
        }
        pdqsort_branchless(&mut self.deltas, |a, b| a < b);
        let mut curdelta = self.deltas[0];
        for i in 1..self.deltas.len() {
            if self.deltas[i] - curdelta <= 10.0 * feastol {
                self.deltas[i] = 0.0;
            } else {
                curdelta = self.deltas[i];
            }
        }
        self.deltas.retain(|&d| d != 0.0);
        let mut bestdelta = -1.0;
        let mut bestefficacy = min_efficacy;

        self.intdata.clear();
        for &j in &self.integerinds {
            self.intdata.push([self.vals[j], self.solval[j], self.upper[j]]);
        }
        for di in 0..self.deltas.len() {
            let delta = self.deltas[di];
            if let Some(efficacy) = self.cmir_efficacy(delta, contcontribution, contsqrnorm) {
                if efficacy > bestefficacy {
                    bestdelta = delta;
                    bestefficacy = efficacy;
                }
            }
        }
        if bestdelta == -1.0 {
            return false;
        }

        let mut k = 1;
        while !only_initial_cmir_scale && k <= 3 {
            let delta = bestdelta * (1 << k) as f64;
            if let Some(efficacy) = self.cmir_efficacy(delta, contcontribution, contsqrnorm) {
                if efficacy > bestefficacy {
                    bestdelta = delta;
                    bestefficacy = efficacy;
                }
            }
            k += 1;
        }

        for kk in 0..self.integerinds.len() {
            let k = self.integerinds[kk];
            if self.upper[k] == K_HIGHS_INF {
                continue;
            }
            if self.solval[k] <= feastol {
                continue;
            }
            self.flip_complementation(k);
            self.intdata[kk] = [self.vals[k], self.solval[k], self.upper[k]];
            let keep = match self.cmir_efficacy(bestdelta, contcontribution, contsqrnorm) {
                None => false,
                Some(efficacy) => {
                    if efficacy > bestefficacy {
                        bestefficacy = efficacy;
                        true
                    } else {
                        false
                    }
                }
            };
            if !keep {
                self.flip_complementation(k);
                self.intdata[kk] = [self.vals[k], self.solval[k], self.upper[k]];
            }
        }

        let scale = 1.0 / CDouble::from(bestdelta);
        let scalrhs = self.rhs * scale;
        let downrhs = scalrhs.to_f64().floor();
        let f0 = scalrhs - downrhs;
        let oneoveroneminusf0 = 1.0 / (1.0 - f0);
        self.rhs = CDouble::from(downrhs * bestdelta);
        self.integral_support = true;
        self.integral_coefficients = false;
        for j in 0..self.rowlen {
            if self.vals[j] == 0.0 {
                continue;
            }
            if self.isintegral[j] == 0 {
                if self.vals[j] > 0.0 {
                    self.vals[j] = 0.0;
                } else {
                    self.vals[j] = (self.vals[j] * oneoveroneminusf0).to_f64();
                    self.integral_support = false;
                }
            } else {
                let scalaj = scale * self.vals[j];
                let downaj = (scalaj + K_HIGHS_TINY).to_f64().floor();
                let fj = scalaj - downaj;
                let mut aj = CDouble::from(downaj);
                if fj > f0 {
                    aj += (fj - f0) * oneoveroneminusf0;
                }
                self.vals[j] = (aj * bestdelta).to_f64();
            }
        }
        true
    }

    fn scale(&mut self, val: f64) -> f64 {
        let (_, e) = frexp(val);
        let expshift = (-e).min(10);
        self.rhs = CDouble::new(ldexp(self.rhs.hi, expshift), ldexp(self.rhs.lo, expshift));
        for v in &mut self.vals[..self.rowlen] {
            *v = ldexp(*v, expshift);
        }
        ldexp(1.0, expshift)
    }

    /// Removes the zeros of inds/vals in place, from the back
    fn remove_zeros(&mut self) {
        let mut i = self.rowlen;
        while i > 0 {
            i -= 1;
            if self.vals[i] == 0.0 {
                self.rowlen -= 1;
                self.inds[i] = self.inds[self.rowlen];
                self.vals[i] = self.vals[self.rowlen];
            }
        }
    }

    fn postprocess_cut<E: CutEnv>(&mut self, env: &E) -> bool {
        let feastol = self.feastol;
        let epsilon = self.epsilon;
        if self.rhs < 0.0 && self.rhs > -epsilon {
            self.rhs = CDouble::from(0.0);
        }
        if self.integral_support && self.integral_coefficients {
            self.remove_zeros();
            return true;
        }
        let mut max_abs_value = 0.0f64;
        for &v in &self.vals[..self.rowlen] {
            max_abs_value = v.abs().max_cpp(max_abs_value);
        }
        let min_coefficient_value = 100.0 * feastol * max_abs_value.max_cpp(1e-3);
        self.integral_support = true;
        let mut i = self.rowlen;
        while i > 0 {
            i -= 1;
            let v = self.vals[i];
            if v == 0.0 {
                continue;
            }
            if v.abs() <= min_coefficient_value {
                let col = self.inds[i] as usize;
                if v < 0.0 {
                    let ub = env.gub(col);
                    if ub == K_HIGHS_INF {
                        return false;
                    }
                    self.rhs -= ub * v;
                } else {
                    let lb = env.glb(col);
                    if lb == -K_HIGHS_INF {
                        return false;
                    }
                    self.rhs -= lb * v;
                }
                self.vals[i] = 0.0;
                continue;
            }
            if self.integral_support && !env.is_integral(self.inds[i] as usize) {
                self.integral_support = false;
            }
        }
        self.remove_zeros();
        if self.rowlen == 0 {
            return false;
        }
        if self.integral_support {
            let intscale = integral_scale(&self.vals[..self.rowlen], feastol, epsilon);
            let mut scale_smallest_val_to_one = true;
            if intscale != 0.0 && intscale * 1.0f64.max_cpp(max_abs_value) <= (1u64 << 52) as f64 {
                self.rhs.renormalize();
                self.rhs *= intscale;
                max_abs_value = nearest_integer(max_abs_value * intscale) as f64;
                for i in 0..self.rowlen {
                    let scaleval = intscale * CDouble::from(self.vals[i]);
                    let intval = nearest_integer(scaleval.to_f64()) as f64;
                    let delta = (scaleval - intval).to_f64();
                    self.vals[i] = intval;
                    let col = self.inds[i] as usize;
                    if delta < 0.0 {
                        let ub = env.gub(col);
                        if ub == K_HIGHS_INF {
                            return false;
                        }
                        self.rhs -= delta * ub;
                    } else {
                        let lb = env.glb(col);
                        if lb == -K_HIGHS_INF {
                            return false;
                        }
                        self.rhs -= delta * lb;
                    }
                }
                self.rhs = (self.rhs + feastol).floor();
                if intscale * max_abs_value * feastol < 0.5 {
                    scale_smallest_val_to_one = false;
                    self.integral_coefficients = true;
                }
            }
            if scale_smallest_val_to_one {
                let mut min_abs_value = K_HIGHS_INF;
                for &v in &self.vals[..self.rowlen] {
                    min_abs_value = v.abs().min_cpp(min_abs_value);
                }
                self.scale(min_abs_value - epsilon);
            }
        } else {
            self.scale(max_abs_value - epsilon);
        }
        true
    }

    /// Returns (ok, has_unbounded_ints, has_general_ints, has_continuous)
    fn preprocess_base_inequality<E: CutEnv>(&mut self, env: &E) -> (bool, bool, bool, bool) {
        let feastol = self.feastol;
        let mut has_unbounded_ints = false;
        let mut has_continuous = false;
        let mut has_general_ints = false;
        let mut num_zeros = 0usize;
        let mut maxact = -feastol;
        let mut max_abs_val = 0.0f64;
        for &v in &self.vals[..self.rowlen] {
            max_abs_val = v.abs().max_cpp(max_abs_val);
        }
        self.initial_scale = self.scale(max_abs_val);

        self.isintegral.resize(self.rowlen, 0);
        for i in 0..self.rowlen {
            self.isintegral[i] =
                (env.is_integral(self.inds[i] as usize) && self.vals[i].abs() > 10.0 * feastol) as u8;
            if self.isintegral[i] == 0 {
                if self.upper[i] < 2.0 * self.solval[i] {
                    if self.complementation.is_empty() {
                        self.complementation.resize(self.rowlen, 0);
                    }
                    self.flip_complementation(i);
                }
                if self.vals[i] > 0.0 || self.vals[i].abs() * self.upper[i] <= 10.0 * feastol {
                    if self.vals[i] < 0.0 {
                        if self.upper[i] == K_HIGHS_INF {
                            return (false, has_unbounded_ints, has_general_ints, has_continuous);
                        }
                        self.rhs -= self.vals[i] * self.upper[i];
                    }
                    num_zeros += 1;
                    self.vals[i] = 0.0;
                    continue;
                }
                has_continuous = true;
            } else {
                if self.upper[i] == K_HIGHS_INF {
                    has_unbounded_ints = true;
                    has_general_ints = true;
                } else if self.upper[i] != 1.0 {
                    has_general_ints = true;
                }
                if self.vals[i] > 0.0 {
                    maxact = self.vals[i].mul_add(self.upper[i], maxact);
                }
            }
        }

        // 100 + 0.15 * numCols, fused by clang
        let max_len = (env.num_lp_cols() as i32 as f64).mul_add(0.15, 100.0) as i32 as usize;
        if self.rowlen - num_zeros > max_len {
            let num_cancel = self.rowlen - num_zeros - max_len;
            let mut cancel_nzs = std::mem::take(&mut self.cancel_nzs);
            cancel_nzs.clear();
            for i in 0..self.rowlen {
                let cancel_slack = if self.vals[i] > 0.0 { self.solval[i] } else { self.upper[i] - self.solval[i] };
                if cancel_slack <= feastol {
                    cancel_nzs.push(i);
                }
            }
            if cancel_nzs.len() < num_cancel {
                self.cancel_nzs = cancel_nzs;
                return (false, has_unbounded_ints, has_general_ints, has_continuous);
            }
            if cancel_nzs.len() > num_cancel {
                let vals = &self.vals;
                partial_sort(&mut cancel_nzs, num_cancel, |&a, &b| vals[a].abs() < vals[b].abs());
            }
            for &j in &cancel_nzs[..num_cancel] {
                if self.vals[j] < 0.0 {
                    self.rhs -= self.vals[j] * self.upper[j];
                } else {
                    maxact = (-self.vals[j]).mul_add(self.upper[j], maxact);
                }
                self.vals[j] = 0.0;
            }
            num_zeros += num_cancel;
            self.cancel_nzs = cancel_nzs;
        }

        if num_zeros != 0 {
            let with_comp = !self.complementation.is_empty();
            let mut i = self.rowlen;
            while i > 0 {
                i -= 1;
                if self.vals[i] == 0.0 {
                    self.rowlen -= 1;
                    let rl = self.rowlen;
                    self.inds[i] = self.inds[rl];
                    self.vals[i] = self.vals[rl];
                    self.upper[i] = self.upper[rl];
                    self.solval[i] = self.solval[rl];
                    self.isintegral[i] = self.isintegral[rl];
                    if with_comp {
                        self.complementation[i] = self.complementation[rl];
                    }
                    num_zeros -= 1;
                    if num_zeros == 0 {
                        break;
                    }
                }
            }
        }
        (maxact > self.rhs.to_f64(), has_unbounded_ints, has_general_ints, has_continuous)
    }

    #[inline]
    fn flip_complementation(&mut self, index: usize) {
        self.complementation[index] = 1 - self.complementation[index];
        self.solval[index] = self.upper[index] - self.solval[index];
        self.rhs -= self.upper[index] * self.vals[index];
        self.vals[index] = -self.vals[index];
    }

    fn remove_complementation(&mut self) {
        if self.complementation.is_empty() {
            return;
        }
        for i in 0..self.rowlen {
            if self.complementation[i] != 0 {
                self.flip_complementation(i);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn try_generate_cut<E: CutEnv>(
        &mut self,
        env: &E,
        has_unbounded_ints: bool,
        has_general_ints: bool,
        has_continuous: bool,
        min_efficacy: f64,
        only_initial_cmir_scale: bool,
        allow_reject_cut: bool,
        lp_sol: bool,
    ) -> bool {
        if has_unbounded_ints {
            return self.cmir_cut_generation_heuristic(min_efficacy, only_initial_cmir_scale);
        }
        self.tmp_vals.clear();
        self.tmp_vals.extend_from_slice(&self.vals[..self.rowlen]);
        self.tmp_inds.clear();
        self.tmp_inds.extend_from_slice(&self.inds[..self.rowlen]);
        self.tmp_complementation.clone_from(&self.complementation);
        self.tmp_solval.clone_from(&self.solval);
        let mut tmp_rhs = self.rhs;

        let mut success = false;
        let mut save_integral_support = false;
        let mut save_integral_coefficients = false;
        if self.determine_cover(env, lp_sol) {
            if !has_continuous && !has_general_ints {
                self.separate_lifted_knapsack_cover();
                success = true;
            } else if has_general_ints {
                success = self.separate_lifted_mixed_integer_cover();
            } else {
                success = self.separate_lifted_mixed_binary_cover();
            }
        }

        let mut min_mir_efficacy = min_efficacy;
        if success {
            save_integral_support = self.integral_support;
            save_integral_coefficients = self.integral_coefficients;
            let mut violation = -self.rhs.to_f64();
            let mut sqrnorm = 0.0;
            for i in 0..self.rowlen {
                let v = self.vals[i];
                self.update_violation_and_norm(i, v, &mut violation, &mut sqrnorm);
            }
            let efficacy = violation / sqrnorm.sqrt();
            if allow_reject_cut && efficacy <= min_efficacy {
                success = false;
                self.rhs = tmp_rhs;
            } else {
                min_mir_efficacy += efficacy;
                std::mem::swap(&mut tmp_rhs, &mut self.rhs);
            }
        }

        // continue on the saved row; the lifted one goes to tmp
        std::mem::swap(&mut self.inds, &mut self.tmp_inds);
        std::mem::swap(&mut self.vals, &mut self.tmp_vals);

        if self.cmir_cut_generation_heuristic(min_mir_efficacy, only_initial_cmir_scale) {
            true
        } else if success {
            self.rhs = tmp_rhs;
            std::mem::swap(&mut self.complementation, &mut self.tmp_complementation);
            std::mem::swap(&mut self.solval, &mut self.tmp_solval);
            std::mem::swap(&mut self.inds, &mut self.tmp_inds);
            std::mem::swap(&mut self.vals, &mut self.tmp_vals);
            self.integral_support = save_integral_support;
            self.integral_coefficients = save_integral_coefficients;
            true
        } else {
            std::mem::swap(&mut self.inds, &mut self.tmp_inds);
            std::mem::swap(&mut self.vals, &mut self.tmp_vals);
            false
        }
    }

    /// HighsDomain::tightenCoefficients on the global domain
    fn tighten_coefficients<E: CutEnv>(&self, env: &E, inds: &[i32], vals: &mut [f64], rhs: &mut f64) {
        let mut maxactivity = CDouble::from(0.0);
        for (&c, &v) in inds.iter().zip(vals.iter()) {
            let c = c as usize;
            if v > 0.0 {
                let ub = env.gub(c);
                if ub == K_HIGHS_INF {
                    return;
                }
                maxactivity += ub * v;
            } else {
                let lb = env.glb(c);
                if lb == -K_HIGHS_INF {
                    return;
                }
                maxactivity += lb * v;
            }
        }
        let maxabscoef = maxactivity - *rhs;
        if maxabscoef > self.feastol {
            let mut upper = CDouble::from(*rhs);
            let mut tightened = 0;
            for (i, &c) in inds.iter().enumerate() {
                let c = c as usize;
                if !env.is_integral(c) {
                    continue;
                }
                if vals[i] > maxabscoef.to_f64() {
                    let delta = vals[i] - maxabscoef;
                    upper -= delta * env.gub(c);
                    vals[i] = maxabscoef.to_f64();
                    tightened += 1;
                } else if vals[i] < (-maxabscoef).to_f64() {
                    let delta = -vals[i] - maxabscoef;
                    upper += delta * env.glb(c);
                    vals[i] = -maxabscoef.to_f64();
                    tightened += 1;
                }
            }
            if tightened != 0 {
                *rhs = upper.to_f64();
            }
        }
    }

    /// The violation check, coefficient tightening and addCut that end
    /// generateCut and finalizeAndAddCut
    fn violated_add<E: CutEnv>(&mut self, env: &mut E, inds_: &mut Vec<i32>, vals_: &mut Vec<f64>, rhs_: &mut f64) -> bool {
        *rhs_ = self.rhs.to_f64();
        vals_.truncate(self.rowlen);
        inds_.truncate(self.rowlen);
        let mut violation = CDouble::from(-*rhs_);
        for (&c, &v) in inds_.iter().zip(vals_.iter()) {
            violation += env.sol(c as usize) * v;
        }
        if violation <= 10.0 * self.feastol {
            return false;
        }
        self.tighten_coefficients(env, inds_, vals_, rhs_);
        let integral = self.integral_support && self.integral_coefficients;
        env.add_cut(inds_, vals_, *rhs_, integral, false) != -1
    }

    /// HighsCutGeneration::generateCut
    pub fn generate_cut(
        &mut self,
        round: &mut SepaRound,
        inds_: &mut Vec<i32>,
        vals_: &mut Vec<f64>,
        rhs_: &mut f64,
        only_initial_cmir_scale: bool,
    ) -> bool {
        let mut ints_positive = true;
        if !round.transform(vals_, &mut self.upper, &mut self.solval, inds_, rhs_, &mut ints_positive, false) {
            return false;
        }
        std::mem::swap(&mut self.inds, inds_);
        std::mem::swap(&mut self.vals, vals_);
        let ok = self.generate_cut_transformed(round, rhs_, ints_positive, only_initial_cmir_scale);
        std::mem::swap(&mut self.inds, inds_);
        std::mem::swap(&mut self.vals, vals_);
        if !ok {
            return false;
        }
        self.violated_add(round, inds_, vals_, rhs_)
    }

    fn generate_cut_transformed(
        &mut self,
        round: &mut SepaRound,
        rhs_: &mut f64,
        ints_positive: bool,
        only_initial_cmir_scale: bool,
    ) -> bool {
        self.rowlen = self.inds.len();
        self.rhs = CDouble::from(*rhs_);
        self.complementation.clear();
        let (ok, has_unbounded_ints, has_general_ints, has_continuous) = self.preprocess_base_inequality(round);
        if !ok {
            return false;
        }
        if !has_unbounded_ints && !ints_positive {
            self.complementation.resize(self.rowlen, 0);
            for i in 0..self.rowlen {
                if self.vals[i] > 0.0 || self.isintegral[i] == 0 {
                    continue;
                }
                self.flip_complementation(i);
            }
        }
        if !self.try_generate_cut(
            round,
            has_unbounded_ints,
            has_general_ints,
            has_continuous,
            10.0 * self.feastol,
            only_initial_cmir_scale,
            true,
            true,
        ) {
            return false;
        }
        self.remove_complementation();
        self.remove_zeros();
        *rhs_ = self.rhs.to_f64();
        self.vals.truncate(self.rowlen);
        self.inds.truncate(self.rowlen);
        if !round.untransform(&mut self.vals, &mut self.inds, rhs_, false) {
            return false;
        }
        self.rowlen = self.inds.len();
        self.rhs = CDouble::from(*rhs_);
        self.postprocess_cut(round)
    }

    /// HighsCutGeneration::finalizeAndAddCut
    pub fn finalize_and_add_cut<E: CutEnv>(
        &mut self,
        env: &mut E,
        inds_: &mut Vec<i32>,
        vals_: &mut Vec<f64>,
        rhs_: &mut f64,
    ) -> bool {
        std::mem::swap(&mut self.inds, inds_);
        std::mem::swap(&mut self.vals, vals_);
        self.complementation.clear();
        self.rowlen = self.inds.len();
        self.rhs = CDouble::from(*rhs_);
        self.integral_support = true;
        self.integral_coefficients = false;
        let mut i = self.rowlen;
        while i > 0 {
            i -= 1;
            if self.vals[i] == 0.0 {
                self.rowlen -= 1;
                self.inds[i] = self.inds[self.rowlen];
                self.vals[i] = self.vals[self.rowlen];
            } else {
                self.integral_support &= env.is_integral(self.inds[i] as usize);
            }
        }
        self.vals.truncate(self.rowlen);
        self.inds.truncate(self.rowlen);
        let ok = self.postprocess_cut(env);
        std::mem::swap(&mut self.inds, inds_);
        std::mem::swap(&mut self.vals, vals_);
        if !ok {
            return false;
        }
        self.violated_add(env, inds_, vals_, rhs_)
    }

    /// HighsCutGeneration::generateConflict: the local domain's bounds
    /// come as (lower, upper) per proof entry
    pub fn generate_conflict<E: CutEnv>(
        &mut self,
        env: &mut E,
        local_bounds: impl Fn(usize) -> (f64, f64),
        proofinds: &mut Vec<i32>,
        proofvals: &mut Vec<f64>,
        proofrhs: &mut f64,
    ) -> bool {
        self.rhs = CDouble::from(*proofrhs);
        std::mem::swap(&mut self.inds, proofinds);
        std::mem::swap(&mut self.vals, proofvals);
        let ok = self.generate_conflict_inner(env, local_bounds);
        std::mem::swap(&mut self.inds, proofinds);
        std::mem::swap(&mut self.vals, proofvals);
        if !ok {
            return false;
        }
        proofvals.truncate(self.rowlen);
        proofinds.truncate(self.rowlen);
        *proofrhs = self.rhs.to_f64();
        let cutintegral = self.integral_support && self.integral_coefficients;
        self.tighten_coefficients(env, proofinds, proofvals, proofrhs);
        env.add_cut(proofinds, proofvals, *proofrhs, cutintegral, true) != -1
    }

    fn generate_conflict_inner<E: CutEnv>(&mut self, env: &mut E, local_bounds: impl Fn(usize) -> (f64, f64)) -> bool {
        self.rowlen = self.inds.len();
        let rowlen = self.rowlen;
        self.complementation.clear();
        self.complementation.resize(rowlen, 0);
        self.upper.resize(rowlen, 0.0);
        self.solval.resize(rowlen, 0.0);
        let mut activity = 0.0f64;
        for i in 0..rowlen {
            let col = self.inds[i] as usize;
            let (glb, gub) = (env.glb(col), env.gub(col));
            let (llb, lub) = local_bounds(col);
            self.upper[i] = gub - glb;
            self.solval[i] = if self.vals[i] < 0.0 { gub.min_cpp(lub) } else { glb.max_cpp(llb) };
            if self.vals[i] < 0.0 && gub != K_HIGHS_INF {
                self.rhs -= gub * self.vals[i];
                self.vals[i] = -self.vals[i];
                self.complementation[i] = 1;
                self.solval[i] = gub - self.solval[i];
            } else {
                self.rhs -= glb * self.vals[i];
                self.complementation[i] = 0;
                self.solval[i] -= glb;
            }
            activity = self.solval[i].mul_add(self.vals[i], activity);
        }
        if activity > self.rhs.to_f64() {
            let sol_scale = self.rhs.to_f64() / activity;
            for s in &mut self.solval[..rowlen] {
                *s *= sol_scale;
            }
        }
        let (ok, has_unbounded_ints, has_general_ints, has_continuous) = self.preprocess_base_inequality(env);
        if !ok {
            return false;
        }
        if !self.try_generate_cut(
            env,
            has_unbounded_ints,
            has_general_ints,
            has_continuous,
            self.feastol,
            false,
            false,
            false,
        ) {
            return false;
        }
        if !self.complementation.is_empty() {
            for i in 0..self.rowlen {
                let col = self.inds[i] as usize;
                if self.complementation[i] != 0 {
                    self.rhs -= env.gub(col) * self.vals[i];
                    self.vals[i] = -self.vals[i];
                } else {
                    self.rhs += env.glb(col) * self.vals[i];
                }
            }
        }
        self.postprocess_cut(env)
    }
}
