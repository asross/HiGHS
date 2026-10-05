//! The QP driver: solveqp (a_quass.cpp) and Quass::solve (quass.cpp) with
//! its helpers (gradient.hpp, reducedcosts.hpp, reducedgradient.hpp,
//! ratiotest.cpp, feasibility_bounded.hpp)

use super::basis::Basis;
use super::cholesky::CholeskyFactor;
use super::pricing::Pricing;
use super::vector::{fma, QpVector};
use super::{BasisStatus, Instance, ModelStatus, Settings, SolverStatus};

/// What the solver asks of its caller
pub trait Callbacks {
    /// Seconds on the HiGHS run clock
    fn time(&mut self) -> f64;
    fn iteration_log(&mut self, iteration: i32, objval: f64, nullspace_dim: i32, time: f64);
    fn nullspace_limit_log(&mut self, nullspace_limit: i32);
    fn degeneracy_fail_log(&mut self, maxabsd: i32, log_d: f64);
    /// Phase 1 (computeStartingPointHighs): a feasible start
    fn phase1(&mut self) -> Phase1Start;
}

/// QpHotstartInformation plus the status of phase 1
pub struct Phase1Start {
    pub status: ModelStatus,
    pub active: Vec<usize>,
    pub status_active: Vec<BasisStatus>,
    pub inactive: Vec<usize>,
    /// Dense starting point and row activities
    pub primal: Vec<f64>,
    pub rowact: Vec<f64>,
}

/// The QP model status and QpSolution
pub struct QpOutcome {
    pub status: ModelStatus,
    pub num_iterations: i32,
    pub primal: Vec<f64>,
    pub rowact: Vec<f64>,
    pub dualvar: Vec<f64>,
    pub dualcon: Vec<f64>,
    pub status_var: Vec<BasisStatus>,
    pub status_con: Vec<BasisStatus>,
}

impl QpOutcome {
    fn new(status: ModelStatus, n: usize, m: usize) -> Self {
        QpOutcome {
            status,
            num_iterations: 0,
            primal: vec![0.0; n],
            rowact: vec![0.0; m],
            dualvar: vec![0.0; n],
            dualcon: vec![0.0; m],
            status_var: vec![BasisStatus::Inactive; n],
            status_con: vec![BasisStatus::Inactive; m],
        }
    }
}

/// solveqp: regularize, find a feasible start, and solve
pub fn solve(mut inst: Instance, settings: &Settings, cb: &mut dyn Callbacks) -> QpOutcome {
    let (n, m) = (inst.num_var, inst.num_con);
    // regularize
    for i in 0..n {
        for e in inst.q.start[i]..inst.q.start[i + 1] {
            if inst.q.index[e] == i {
                inst.q.value[e] += settings.hessian_regularization_value;
            }
        }
    }
    let start = if m == 0 && n <= 15000 {
        let start = start_bounded(&inst, settings);
        match start.status {
            ModelStatus::Optimal => {
                let mut out = QpOutcome::new(start.status, n, m);
                out.primal = start.primal;
                return out;
            }
            ModelStatus::Unbounded => return QpOutcome::new(start.status, n, m),
            _ => start,
        }
    } else {
        let start = cb.phase1();
        if start.status != ModelStatus::NotSet {
            return QpOutcome::new(start.status, n, m);
        }
        start
    };
    Quass::new(inst, settings, &start).solve(cb)
}

/// computeStartingPointBounded: for bounds only, the unconstrained
/// minimizer projected onto the bounds
fn start_bounded(inst: &Instance, settings: &Settings) -> Phase1Start {
    let n = inst.num_var;
    let q = &inst.q;
    let mut l = vec![0.0; n * n];
    for col in 0..n {
        for idx in q.start[col]..q.start[col + 1] {
            let row = q.index[idx];
            let mut sum = 0.0;
            if row == col {
                for k in 0..row {
                    sum = fma(l[k * n + row], l[k * n + row], sum);
                }
                l[row * n + row] = (q.value[idx] - sum).sqrt();
            } else {
                for k in 0..row {
                    sum = fma(l[k * n + col], l[k * n + row], sum);
                }
                l[row * n + col] = (q.value[idx] - sum) / l[row * n + row];
            }
        }
    }
    let mut res = inst.c.neg();
    for r in 0..res.dim {
        for j in 0..r {
            res.value[r] = fma(-res.value[j], l[j * n + r], res.value[r]);
        }
        res.value[r] /= l[r * n + r];
    }
    for i in (0..res.dim).rev() {
        let mut sum = 0.0;
        for j in (i + 1..res.dim).rev() {
            sum = fma(res.value[j], l[i * n + j], sum);
        }
        res.value[i] = (res.value[i] - sum) / l[i * n + i];
    }

    let mut start = Phase1Start {
        status: ModelStatus::Undetermined,
        active: vec![],
        status_active: vec![],
        inactive: vec![],
        primal: vec![0.0; n],
        rowact: vec![0.0; inst.num_con],
    };
    let big = 0.5 / settings.hessian_regularization_value;
    for i in 0..n {
        let (lo, up, c) = (inst.var_lo[i], inst.var_up[i], inst.c.value[i]);
        let x = &mut res.value[i];
        // (sic: lo == +inf)
        if (*x > big && up == f64::INFINITY && c < 0.0) || (*x < big && lo == f64::INFINITY && c > 0.0) {
            start.status = ModelStatus::Unbounded;
            return start;
        } else if *x <= lo {
            *x = lo;
            start.active.push(i + inst.num_con);
            start.status_active.push(BasisStatus::ActiveAtLower);
        } else if *x >= up {
            *x = up;
            start.active.push(i + inst.num_con);
            start.status_active.push(BasisStatus::ActiveAtUpper);
        } else {
            start.inactive.push(i + inst.num_con);
        }
        if x.abs() > 1e-4 {
            start.primal[i] = *x;
        }
    }
    if start.active.is_empty() {
        start.status = ModelStatus::Optimal;
    }
    start
}

struct Gradient {
    g: QpVector,
    uptodate: bool,
    numupdates: i32,
}

impl Gradient {
    fn get(&mut self, inst: &Instance, primal: &QpVector, frequency: i32) -> &QpVector {
        if !self.uptodate || self.numupdates >= frequency {
            self.recompute(inst, primal);
        }
        &self.g
    }

    fn recompute(&mut self, inst: &Instance, primal: &QpVector) {
        inst.q.vec_mat(primal, &mut self.g);
        self.g.add_assign(&inst.c);
        self.uptodate = true;
        self.numupdates = 0;
    }
}

#[derive(Clone, Copy)]
struct RatiotestResult {
    alpha: f64,
    limitingconstraint: Option<usize>,
    nowactiveatlower: bool,
}

fn step(x: f64, p: f64, l: f64, u: f64, t: f64) -> f64 {
    if p < -t && l > f64::NEG_INFINITY {
        (l - x) / p
    } else if p > t && u < f64::INFINITY {
        (u - x) / p
    } else {
        f64::INFINITY
    }
}

/// Bounds relaxed by ratiotest_d (ratiotest_relax_instance)
struct Bounds {
    con_lo: Vec<f64>,
    con_up: Vec<f64>,
    var_lo: Vec<f64>,
    var_up: Vec<f64>,
}

impl Bounds {
    fn relaxed(inst: &Instance, d: f64) -> Self {
        let lo = |v: &[f64]| v.iter().map(|&b| if b != f64::NEG_INFINITY { b - d } else { b }).collect();
        let up = |v: &[f64]| v.iter().map(|&b| if b != f64::INFINITY { b + d } else { b }).collect();
        Bounds { con_lo: lo(&inst.con_lo), con_up: up(&inst.con_up), var_lo: lo(&inst.var_lo), var_up: up(&inst.var_up) }
    }
}

struct Quass {
    inst: Instance,
    settings: Settings,
    relaxed: Bounds,
    primal: QpVector,
    rowactivity: QpVector,
    basis: Basis,
    gradient: Gradient,
    /// ReducedCosts
    redcosts: QpVector,
    redcosts_uptodate: bool,
    /// ReducedGradient
    redgrad: QpVector,
    redgrad_uptodate: bool,
    factor: CholeskyFactor,
    pricing: Pricing,
    num_iterations: i32,
    status: ModelStatus,
}

impl Quass {
    fn new(inst: Instance, settings: &Settings, start: &Phase1Start) -> Self {
        let (n, m) = (inst.num_var, inst.num_con);
        let mut basis = Basis::new(inst.a.transpose(), m, &start.active, &start.status_active, &start.inactive);
        let factor = CholeskyFactor::new(n, basis.inactive().len());
        let primal = QpVector::from_dense(&start.primal);
        let mut rowactivity = QpVector::new(m);
        inst.a.mat_vec(&primal, &mut rowactivity);
        let pricing = Pricing::new(settings.pricing_strategy(), n, &mut basis);
        let relaxed = Bounds::relaxed(&inst, settings.ratiotest_d);
        Quass {
            settings: *settings,
            relaxed,
            primal,
            rowactivity,
            basis,
            gradient: Gradient { g: QpVector::new(n), uptodate: false, numupdates: 0 },
            redcosts: QpVector::new(n),
            redcosts_uptodate: false,
            redgrad: QpVector::new(n),
            redgrad_uptodate: false,
            factor,
            pricing,
            num_iterations: 0,
            status: ModelStatus::Undetermined,
            inst,
        }
    }

    fn grad(&mut self) -> &QpVector {
        self.gradient.get(&self.inst, &self.primal, self.settings.gradientrecomputefrequency)
    }

    fn recompute_redcosts(&mut self) {
        self.gradient.get(&self.inst, &self.primal, self.settings.gradientrecomputefrequency);
        self.basis.ftran(&self.gradient.g, &mut self.redcosts, false);
        self.redcosts_uptodate = true;
    }

    fn recompute_redgrad(&mut self) {
        self.redgrad.dim = self.basis.inactive().len();
        self.gradient.get(&self.inst, &self.primal, self.settings.gradientrecomputefrequency);
        self.basis.ztprod(&self.gradient.g, &mut self.redgrad, false);
        self.redgrad_uptodate = true;
    }

    fn reinvert(&mut self) -> SolverStatus {
        self.basis.rebuild();
        let status = self.factor.recompute(&self.inst.q, self.inst.num_var, &mut self.basis);
        if status != SolverStatus::Ok {
            return status;
        }
        self.gradient.recompute(&self.inst, &self.primal);
        self.recompute_redcosts();
        self.recompute_redgrad();
        SolverStatus::Ok
    }

    fn not_ok(&mut self, status: SolverStatus) {
        self.status = if status == SolverStatus::NotPositiveDefinite {
            ModelStatus::NonConvex
        } else {
            ModelStatus::Error
        };
    }

    fn log(&mut self, cb: &mut dyn Callbacks) {
        let nullspace = (self.inst.num_var - self.basis.num_active()) as i32;
        let objval = self.inst.objval(&self.primal);
        let time = cb.time();
        cb.iteration_log(self.num_iterations, objval, nullspace, time);
    }

    /// computesearchdirection_minor
    fn search_direction_minor(&mut self, p: &mut QpVector) -> SolverStatus {
        if !self.redgrad_uptodate {
            self.recompute_redgrad();
        }
        let mut g2 = self.redgrad.neg();
        g2.sanitize(1e-14);
        let status = self.factor.solve(&self.inst.q, self.inst.num_var, &mut self.basis, &mut g2);
        if status != SolverStatus::Ok {
            return status;
        }
        g2.sanitize(1e-14);
        self.basis.zprod(&g2, p);
        SolverStatus::Ok
    }

    /// computesearchdirection_major
    fn search_direction_major(
        &mut self,
        yp: &QpVector,
        gyp: &mut QpVector,
        l: &mut QpVector,
        m: &mut QpVector,
        p: &mut QpVector,
    ) -> SolverStatus {
        let yyp = yp.clone();
        self.inst.q.mat_vec(&yyp, gyp);
        if self.basis.num_active() < self.inst.num_var {
            self.basis.ztprod(gyp, m, false);
            l.clone_from(m);
            let status = self.factor.solve_l(&self.inst.q, self.inst.num_var, &mut self.basis, l);
            if status != SolverStatus::Ok {
                return status;
            }
            let mut v = l.clone();
            self.factor.solve_lt(&mut v);
            self.basis.zprod(&v, p);
            let b = if self.grad().dot(&yyp) < 0.0 { 1.0 } else { -1.0 };
            p.saxpy2(-1.0, b, &yyp);
        } else {
            p.repopulate(yp);
            let s = -self.grad().dot(yp);
            p.scale(s);
        }
        SolverStatus::Ok
    }

    /// computemaxsteplength: (step length, zero curvature direction)
    fn max_step_length(&mut self, p: &QpVector, qp: &mut QpVector) -> (f64, bool) {
        self.inst.q.mat_vec(p, qp);
        let denominator = p.dot(qp);
        if denominator.abs() > self.settings.pqp_zero_threshold {
            let numerator = -p.dot(self.grad());
            if numerator < 0.0 {
                (0.0, false)
            } else {
                (numerator / denominator, false)
            }
        } else {
            (f64::INFINITY, true)
        }
    }

    /// reduce: the constraint to drop from the null space for the newly
    /// active one, with d = Z'a. Err(log10|d|) if degenerate.
    fn reduce(&mut self, newactivecon: usize, d: &mut QpVector) -> Result<(usize, usize), (usize, f64)> {
        if let Some(idx) = self.basis.inactive().iter().position(|&c| c == newactivecon) {
            d.set_unit(idx);
            return Ok((idx, newactivecon));
        }
        let mut aq = QpVector::new(self.inst.num_var);
        self.basis.atran().extractcol(newactivecon, &mut aq);
        self.basis.ztprod(&aq, d, true);
        let mut maxabsd = 0;
        for &j in d.nz() {
            if d.value[j].abs() > d.value[maxabsd].abs() {
                maxabsd = j;
            }
        }
        if d.value[maxabsd].abs() < self.settings.d_zero_threshold {
            // (the null space may be empty here)
            return Err((maxabsd, d.value[maxabsd].abs().log10()));
        }
        Ok((maxabsd, self.basis.inactive()[maxabsd]))
    }

    fn ratiotest(&self, p: &QpVector, rowmove: &QpVector, alphastart: f64) -> RatiotestResult {
        if self.settings.ratiotest == 1 {
            let b = &self.inst;
            return self.ratiotest_textbook(p, rowmove, (&b.con_lo, &b.con_up, &b.var_lo, &b.var_up), alphastart);
        }
        let r = &self.relaxed;
        let res1 = self.ratiotest_textbook(p, rowmove, (&r.con_lo, &r.con_up, &r.var_lo, &r.var_up), alphastart);
        let Some(lc) = res1.limitingconstraint else {
            return res1;
        };
        let inst = &self.inst;
        let t = self.settings.ratiotest_t;
        let mut result = res1;
        let mut max_pivot = if lc < inst.num_con { rowmove.value[lc] } else { p.value[lc - inst.num_con] };
        for i in 0..inst.num_con {
            let step_i = step(self.rowactivity.value[i], rowmove.value[i], inst.con_lo[i], inst.con_up[i], t);
            if rowmove.value[i].abs() >= max_pivot.abs() && step_i <= res1.alpha {
                max_pivot = rowmove.value[i];
                result.limitingconstraint = Some(i);
                result.alpha = step_i;
                result.nowactiveatlower = rowmove.value[i] < 0.0;
            }
        }
        for i in 0..inst.num_var {
            let step_i = step(self.primal.value[i], p.value[i], inst.var_lo[i], inst.var_up[i], t);
            if p.value[i].abs() >= max_pivot.abs() && step_i <= res1.alpha {
                max_pivot = p.value[i];
                result.limitingconstraint = Some(inst.num_con + i);
                result.alpha = step_i;
                result.nowactiveatlower = p.value[i] < 0.0;
            }
        }
        result.alpha = result.alpha.max(0.0);
        result
    }

    fn ratiotest_textbook(
        &self,
        p: &QpVector,
        rowmove: &QpVector,
        (con_lo, con_up, var_lo, var_up): (&[f64], &[f64], &[f64], &[f64]),
        alphastart: f64,
    ) -> RatiotestResult {
        let t = self.settings.ratiotest_t;
        let mut result = RatiotestResult { alpha: alphastart, limitingconstraint: None, nowactiveatlower: false };
        for &i in p.nz() {
            let alpha_i = step(self.primal.value[i], p.value[i], var_lo[i], var_up[i], t);
            if alpha_i < result.alpha {
                result.alpha = alpha_i;
                result.limitingconstraint = Some(self.inst.num_con + i);
                result.nowactiveatlower = p.value[i] < 0.0;
            }
        }
        for &i in rowmove.nz() {
            let alpha_i = step(self.rowactivity.value[i], rowmove.value[i], con_lo[i], con_up[i], t);
            if alpha_i < result.alpha {
                result.alpha = alpha_i;
                result.limitingconstraint = Some(i);
                result.nowactiveatlower = rowmove.value[i] < 0.0;
            }
        }
        result
    }

    /// computerowmove and tidyup
    fn row_move(&self, p: &mut QpVector, rowmove: &mut QpVector) {
        self.inst.a.mat_vec(p, rowmove);
        let num_con = self.inst.num_con;
        for &acon in self.basis.active() {
            if acon >= num_con {
                p.value[acon - num_con] = 0.0;
            } else {
                rowmove.value[acon] = 0.0;
            }
        }
    }

    /// The main loop; returns early (without duals) on failure
    fn iterate(&mut self, cb: &mut dyn Callbacks) -> bool {
        let n = self.inst.num_var;
        let mut p = QpVector::new(n);
        let mut rowmove = QpVector::new(self.inst.num_con);
        let mut buffer_yp = QpVector::new(n);
        let mut buffer_gyp = QpVector::new(n);
        let mut buffer_l = QpVector::new(n);
        let mut buffer_m = QpVector::new(n);
        let mut buffer_qp = QpVector::new(n);
        let mut buffer_d = QpVector::new(n);

        let mut last_logging_iteration = self.num_iterations - 1;
        let mut last_logging_time = 0.0;
        let mut logging_time_interval = 10.0;
        let mut atfsep = self.basis.num_active() == n;
        loop {
            if self.num_iterations >= self.settings.iteration_limit {
                self.status = ModelStatus::IterationLimit;
                return true;
            }
            if cb.time() >= self.settings.time_limit {
                self.status = ModelStatus::TimeLimit;
                return true;
            }
            if self.basis.inactive().len() as i64 > self.settings.nullspace_limit as i64 {
                cb.nullspace_limit_log(self.settings.nullspace_limit);
                self.status = ModelStatus::LargeNullspace;
                return false;
            }

            // Logging
            let run_time = cb.time();
            let freq = self.settings.reportingfequency;
            if (self.num_iterations % freq == 0 || run_time - last_logging_time > logging_time_interval)
                && self.num_iterations > last_logging_iteration
            {
                let mut log_report = true;
                if self.num_iterations > 10 * freq {
                    self.settings.reportingfequency *= 10;
                    log_report = false;
                }
                if run_time > 10.0 * logging_time_interval {
                    logging_time_interval *= 2.0;
                }
                if log_report {
                    last_logging_time = run_time;
                    last_logging_iteration = self.num_iterations;
                    self.log(cb);
                }
            }

            if self.basis.reinversion_hint() {
                let status = self.reinvert();
                if status != SolverStatus::Ok {
                    self.not_ok(status);
                    return false;
                }
            }

            let mut zero_curvature_direction = false;
            let mut maxsteplength = 1.0;
            if atfsep {
                // Determine a constraint to relax; if none, optimal. (C++
                // passes the gradient to price)
                self.grad();
                if !self.redcosts_uptodate {
                    self.recompute_redcosts();
                }
                let Some(minidx) =
                    self.pricing.price(&self.inst, &self.basis, &self.redcosts, self.settings.lambda_zero_threshold)
                else {
                    self.status = ModelStatus::Optimal;
                    return true;
                };
                self.num_iterations += 1;

                let unit = self.basis.index_in_factor()[minidx] as usize;
                buffer_yp.set_unit(unit);
                let unit_vec = buffer_yp.clone();
                self.basis.btran(&unit_vec, &mut buffer_yp);

                let ns = self.basis.inactive().len();
                buffer_l.dim = ns;
                buffer_m.dim = ns;
                let status =
                    self.search_direction_major(&buffer_yp, &mut buffer_gyp, &mut buffer_l, &mut buffer_m, &mut p);
                if status != SolverStatus::Ok {
                    self.not_ok(status);
                    return false;
                }
                self.basis.deactivate(minidx);
                self.row_move(&mut p, &mut rowmove);
                let (len, zcd) = self.max_step_length(&p, &mut buffer_qp);
                maxsteplength = len;
                zero_curvature_direction = zcd;
                if !zero_curvature_direction {
                    let status = self.factor.expand(&buffer_yp, &buffer_gyp, &mut buffer_l);
                    if status != SolverStatus::Ok {
                        self.not_ok(status);
                        return false;
                    }
                }
                // ReducedGradient::expand
                if self.redgrad_uptodate {
                    let newval = buffer_yp.dot(self.grad());
                    let rg = &mut self.redgrad;
                    rg.value.push(newval);
                    rg.index.push(0);
                    rg.index[rg.num_nz] = rg.dim;
                    rg.num_nz += 1;
                    rg.dim += 1;
                }
            } else {
                let status = self.search_direction_minor(&mut p);
                if status != SolverStatus::Ok {
                    self.not_ok(status);
                    return false;
                }
                self.row_move(&mut p, &mut rowmove);
                self.inst.q.mat_vec(&p, &mut buffer_qp);
            }

            if p.norm2() < self.settings.pnorm_zero_threshold || maxsteplength == 0.0 {
                atfsep = true;
                continue;
            }
            self.num_iterations += 1;
            let stepres = self.ratiotest(&p, &rowmove, maxsteplength);
            if let Some(limiting) = stepres.limitingconstraint {
                let (maxabsd, constrainttodrop) = match self.reduce(limiting, &mut buffer_d) {
                    Ok(r) => r,
                    Err((maxabsd, log_d)) => {
                        cb.degeneracy_fail_log(maxabsd as i32, log_d);
                        self.status = ModelStatus::Undetermined;
                        return false;
                    }
                };
                if !zero_curvature_direction {
                    let p_in_v = self.basis.inactive().contains(&limiting);
                    self.factor.reduce(&buffer_d, maxabsd, p_in_v);
                }
                // ReducedGradient::reduce, then update(alpha, false)
                if self.redgrad_uptodate {
                    let rg = &mut self.redgrad;
                    for &idx in buffer_d.nz() {
                        if idx != maxabsd {
                            rg.value[idx] -= rg.value[maxabsd] * buffer_d.value[idx] / buffer_d.value[maxabsd];
                        }
                    }
                    rg.resparsify();
                }
                self.redgrad_uptodate = false;

                let newstatus =
                    if stepres.nowactiveatlower { BasisStatus::ActiveAtLower } else { BasisStatus::ActiveAtUpper };
                let status =
                    self.basis.activate(&self.settings, limiting, newstatus, constrainttodrop, &mut self.pricing);
                if status != SolverStatus::Ok {
                    self.status = ModelStatus::Undetermined;
                    return false;
                }
                if self.basis.num_active() != n {
                    atfsep = false;
                }
            } else {
                if stepres.alpha == f64::INFINITY {
                    self.status = ModelStatus::Unbounded;
                    return false;
                }
                atfsep = false;
                self.redgrad_uptodate = false;
            }

            self.primal.saxpy(stepres.alpha, &p);
            self.rowactivity.saxpy(stepres.alpha, &rowmove);
            self.gradient.g.saxpy(stepres.alpha, &buffer_qp);
            self.gradient.numupdates += 1;
            self.redcosts_uptodate = false;
        }
    }

    /// Quass::solve and the copy to QpSolution (solveqp_actual)
    fn solve(mut self, cb: &mut dyn Callbacks) -> QpOutcome {
        let (n, m) = (self.inst.num_var, self.inst.num_con);
        let mut out = QpOutcome::new(ModelStatus::Undetermined, n, m);
        if self.iterate(cb) {
            self.log(cb);
            if !self.redcosts_uptodate {
                self.recompute_redcosts();
            }
            for &e in self.basis.active() {
                let lambda = self.redcosts.value[self.basis.index_in_factor()[e] as usize];
                if e >= m {
                    out.dualvar[e - m] = lambda;
                } else {
                    out.dualcon[e] = lambda;
                }
            }
            for i in 0..n {
                out.status_var[i] = self.basis.status(m + i);
            }
            for i in 0..m {
                out.status_con[i] = self.basis.status(i);
            }
            if self.basis.num_active() == n {
                let inst = &self.inst;
                self.primal = self.basis.recomputex(&inst.con_lo, &inst.con_up, &inst.var_lo, &inst.var_up);
            }
        }
        out.status = self.status;
        out.num_iterations = self.num_iterations;
        out.primal.copy_from_slice(&self.primal.value[..n]);
        out.rowact.copy_from_slice(&self.rowactivity.value[..m]);
        out
    }
}
