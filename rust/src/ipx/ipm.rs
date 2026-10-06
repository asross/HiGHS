//! IPM (ipm.h/.cc): Mehrotra's predictor-corrector method on the iterate,
//! with the KKT systems solved by a KktSolver; plus the starting point and
//! the optional centring steps.

use crate::util::fma::ClangFma;

use super::control::Control;
use super::fmt::{fixed, fmt, sci, time};
use super::iterate::{Iterate, State};
use super::kkt::KktSolver;
use super::lu::LuResult;
use super::sparse_matrix::{dot_column, multiply_add, scatter_column};
use super::utils::{infnorm, twonorm};
use super::{
    cmax, cmin, Info, Int, ERROR_TIME_INTERRUPT, ERROR_USER_INTERRUPT, STATUS_DUAL_INFEAS,
    STATUS_FAILED, STATUS_ITER_LIMIT, STATUS_NOT_RUN, STATUS_NO_PROGRESS, STATUS_OPTIMAL,
    STATUS_PRIMAL_INFEAS, STATUS_TIME_LIMIT, STATUS_USER_INTERRUPT,
};
use std::time::Instant;

const DIVERGE_TOL: f64 = 1e6;

struct Step {
    x: Vec<f64>,
    xl: Vec<f64>,
    xu: Vec<f64>,
    y: Vec<f64>,
    zl: Vec<f64>,
    zu: Vec<f64>,
}

impl Step {
    fn new(m: usize, n: usize) -> Self {
        Step {
            x: vec![0.0; n + m],
            xl: vec![0.0; n + m],
            xu: vec![0.0; n + m],
            y: vec![0.0; m],
            zl: vec![0.0; n + m],
            zu: vec![0.0; n + m],
        }
    }
}

pub struct Ipm<'a> {
    control: &'a Control,
    step_primal: f64,
    step_dual: f64,
    num_bad_iter: Int,
    best_complementarity: f64,
    maxiter: Int,
    centring_ratio: f64,
    bad_products: Int,
}

/// Maximum alpha <= alpha0 such that x + alpha*dx >= 0, and the blocking
/// index (-1 if none).
fn step_to_boundary(x: &[f64], dx: &[f64], alpha0: f64) -> (f64, Int) {
    let damp = 1.0 - f64::EPSILON;
    let mut alpha = alpha0;
    let mut iblock = -1;
    for i in 0..x.len() {
        if alpha.mul_add_c(dx[i], x[i]) < 0.0 {
            alpha = -(x[i] * damp) / dx[i];
            iblock = i as Int;
        }
    }
    (alpha, iblock)
}

/// Converts errflag into status_ipm (as at the end of StartingPoint() and
/// Driver()).
fn interrupt_status(info: &mut Info) -> bool {
    if info.errflag == ERROR_USER_INTERRUPT {
        info.errflag = 0;
        info.status_ipm = STATUS_USER_INTERRUPT;
        true
    } else if info.errflag == ERROR_TIME_INTERRUPT {
        info.errflag = 0;
        info.status_ipm = STATUS_TIME_LIMIT;
        true
    } else {
        false
    }
}

impl<'a> Ipm<'a> {
    pub fn new(control: &'a Control) -> Self {
        Ipm {
            control,
            step_primal: 0.0,
            step_dual: 0.0,
            num_bad_iter: 0,
            best_complementarity: 0.0,
            maxiter: -1,
            centring_ratio: 0.0,
            bad_products: 0,
        }
    }

    pub fn set_maxiter(&mut self, maxiter: Int) {
        self.maxiter = maxiter;
    }

    /// Computes the starting point and initializes the iterate.
    pub fn starting_point(&mut self, kkt: &mut dyn KktSolver, iterate: &mut Iterate, info: &mut Info) -> LuResult<()> {
        self.print_header();
        self.compute_starting_point(kkt, iterate, info)?;
        if info.errflag == 0 {
            self.print_output(kkt, iterate, info)?;
        }
        // Set status_ipm.
        if !interrupt_status(info) {
            info.status_ipm = if info.errflag != 0 {
                STATUS_FAILED
            } else {
                STATUS_NOT_RUN
            };
        }
        Ok(())
    }

    /// Runs IPM iterations until termination.
    pub fn driver(&mut self, kkt: &mut dyn KktSolver, iterate: &mut Iterate, info: &mut Info) -> LuResult<()> {
        let m = iterate.model().rows();
        let n = iterate.model().cols();
        let mut step = Step::new(m, n);
        self.num_bad_iter = 0;

        loop {
            if iterate.term_crit_reached() {
                info.status_ipm = STATUS_OPTIMAL;
                break;
            }
            if self.num_bad_iter >= 5
                || (self.best_complementarity > 0.0
                    && iterate.complementarity() > DIVERGE_TOL * self.best_complementarity)
            {
                // No progress in reducing the complementarity gap.
                // Check if model seems to be primal or dual infeasible.
                let dualized = iterate.model().dualized();
                let pobjective = iterate.pobjective_after_postproc();
                let dobjective = iterate.dobjective_after_postproc();
                info.status_ipm = if dobjective > cmax(10.0 * pobjective.abs(), 1.0) {
                    // Dual objective tends to positive infinity. Looks like
                    // the model is dual unbounded, i.e. primal infeasible.
                    if dualized {
                        STATUS_DUAL_INFEAS
                    } else {
                        STATUS_PRIMAL_INFEAS
                    }
                } else if pobjective < -cmax(10.0 * dobjective.abs(), 1.0) {
                    // Primal objective tends to negative infinity. Looks like
                    // the model is primal unbounded, i.e. dual infeasible.
                    if dualized {
                        STATUS_PRIMAL_INFEAS
                    } else {
                        STATUS_DUAL_INFEAS
                    }
                } else {
                    STATUS_NO_PROGRESS
                };
                break;
            }
            if info.iter >= self.maxiter {
                info.status_ipm = STATUS_ITER_LIMIT;
                break;
            }
            info.errflag = self.control.interrupt_check(info.iter);
            if info.errflag != 0 {
                break;
            }
            kkt.factorize(Some(iterate), info)?;
            if info.errflag != 0 {
                break;
            }
            self.predictor(kkt, iterate, info, &mut step)?;
            if info.errflag != 0 {
                break;
            }
            self.add_corrector(kkt, iterate, info, &mut step)?;
            if info.errflag != 0 {
                break;
            }
            self.make_step(iterate, &step, false);
            info.iter += 1;
            self.print_output(kkt, iterate, info)?;
        }

        // Set status_ipm if errflag terminated IPM.
        if info.errflag != 0 && !interrupt_status(info) {
            info.status_ipm = STATUS_FAILED;
        }

        if self.control.run_centring() != 0 && info.status_ipm == STATUS_OPTIMAL && info.centring_tried == 0 {
            // Centrality of a point is evaluated by min (xj*zj)/mu and max
            // (xj*zj)/mu (ideally in [0.1,10.0]). As soon as max/min is below
            // centringRatioTolerance, the point is considered centred. A
            // centring step is accepted if the new ratio is lower than the
            // previous ratio times centringRatioReduction; otherwise no more
            // centring steps are performed.
            self.control.log("Performing centring steps...\n");

            // freeze mu to its current value
            let mu_frozen = iterate.mu();

            // assess and print centrality of current point
            self.assess_centrality(iterate, None, iterate.mu(), true);
            let mut prev_ratio = self.centring_ratio;
            let mut prev_bad_products = self.bad_products;

            info.centring_success = 0;
            // if ratio is below tolerance, point is centred
            if prev_ratio < self.control.centring_ratio_tolerance() {
                self.control.log("\tPoint is now centred\n");
                info.centring_success = 1;
            } else {
                // perform centring steps
                let mut centring_complete = false;
                for _ in 0..self.control.max_centring_steps() {
                    // compute centring step
                    self.centring(kkt, iterate, info, &mut step, mu_frozen)?;

                    // assess whether to take the step
                    let accept = self.evaluate_centring_step(iterate, &step, prev_ratio, prev_bad_products);
                    if !accept {
                        self.control.log("\tPoint cannot be centred further\n");
                        centring_complete = true;
                        break;
                    }

                    // take the step and print output
                    self.make_step(iterate, &step, true);
                    info.iter += 1;
                    self.print_output(kkt, iterate, info)?;
                    self.assess_centrality(iterate, None, iterate.mu(), true);
                    prev_ratio = self.centring_ratio;
                    prev_bad_products = self.bad_products;

                    // if ratio is below tolerance, point is centred
                    if prev_ratio < self.control.centring_ratio_tolerance() {
                        self.control.log("\tPoint is now centred\n");
                        info.centring_success = 1;
                        centring_complete = true;
                        break;
                    }
                }
                if !centring_complete {
                    self.control.log(&format!(
                        "\tPoint could not be centred within {} iterations\n",
                        self.control.max_centring_steps()
                    ));
                }
            }
            info.centring_tried = 1;
        }
        Ok(())
    }

    fn compute_starting_point(&mut self, kkt: &mut dyn KktSolver, iterate: &mut Iterate, info: &mut Info) -> LuResult<()> {
        let model = iterate.model();
        let m = model.rows();
        let n = model.cols();
        let (ai, b, c, lb, ub) = (model.ai(), model.b(), model.c(), model.lb(), model.ub());
        let mut x = vec![0.0; n + m];
        let mut xl = vec![0.0; n + m];
        let mut xu = vec![0.0; n + m];
        let mut y = vec![0.0; m];
        let mut zl = vec![0.0; n + m];
        let mut zu = vec![0.0; n + m];
        let mut rb = vec![0.0; m]; // workspace

        // Factorize the KKT matrix with the identity matrix in the (1,1)
        // block.
        kkt.factorize(None, info)?;
        if info.errflag != 0 {
            return Ok(());
        }

        // Set x within its bounds and compute the minimum norm solution dx
        // to AI*dx = (b-AI*x). Then update x := x + dx to obtain a feasible
        // point.
        rb.copy_from_slice(b);
        for j in 0..n + m {
            let mut xj = 0.0;
            if xj < lb[j] {
                xj = lb[j];
            }
            if xj > ub[j] {
                xj = ub[j];
            }
            x[j] = xj;
            if xj != 0.0 {
                scatter_column(ai, j, -xj, &mut rb);
            }
        }
        let tol = 0.1 * infnorm(&rb);
        zl.fill(0.0);
        kkt.solve(&zl, &rb, tol, &mut xl, &mut y, info)?;
        if info.errflag != 0 {
            return Ok(());
        }
        for j in 0..n + m {
            x[j] += xl[j];
        }

        // Compute xl, xu and shift to become positive.
        let mut xinfeas = 0.0;
        for j in 0..n + m {
            xl[j] = x[j] - lb[j];
            xinfeas = cmax(xinfeas, -xl[j]);
            xu[j] = ub[j] - x[j];
            xinfeas = cmax(xinfeas, -xu[j]);
        }
        let xshift1 = 1.5f64.mul_add_c(xinfeas, 1.0);
        for j in 0..n + m {
            xl[j] += xshift1;
            xu[j] += xshift1;
        }

        let cnorm = twonorm(c);
        if cnorm == 0.0 {
            // Special treatment for zero objective.
            for j in 0..n + m {
                zl[j] = if lb[j].is_finite() { 1.0 } else { 0.0 };
                zu[j] = if ub[j].is_finite() { 1.0 } else { 0.0 };
            }
        } else {
            // Compute y as the least-squares solution to AI'*y=c. Recompute
            // zl = c-AI'*y because the KKT system is solved approximately
            // with a residual in the first block equation.
            rb.fill(0.0);
            let tol = 0.1 * infnorm(c);
            kkt.solve(c, &rb, tol, &mut zl, &mut y, info)?;
            if info.errflag != 0 {
                return Ok(());
            }
            zl.copy_from_slice(c);
            multiply_add(ai, &y, -1.0, &mut zl, b'T');

            // When c lies in range(AI'), then the dual slack variables are
            // (close to) zero, and the initial point would be almost
            // complementary but usually not primal feasible. To prevent this,
            // add a fraction of the objective to zl and adjust y.
            let znorm = twonorm(&zl);
            const RHO: f64 = 0.05;
            if znorm < RHO * cnorm {
                for j in 0..n + m {
                    zl[j] += RHO * c[j];
                }
                for yi in y.iter_mut() {
                    *yi *= 1.0 - RHO;
                }
            }

            // Split dual slack solution into zl, zu and shift to become
            // positive.
            let mut zinfeas = 0.0;
            for j in 0..n + m {
                let zval = zl[j];
                zl[j] = 0.0;
                zu[j] = 0.0;
                if lb[j].is_finite() && ub[j].is_finite() {
                    zl[j] = 0.5 * zval;
                    zu[j] = -0.5 * zval;
                } else if lb[j].is_finite() {
                    zl[j] = zval;
                } else if ub[j].is_finite() {
                    zu[j] = -zval;
                }
                zinfeas = cmax(zinfeas, -zl[j]);
                zinfeas = cmax(zinfeas, -zu[j]);
            }
            let zshift1 = 1.5f64.mul_add_c(zinfeas, 1.0);
            for j in 0..n + m {
                if lb[j].is_finite() {
                    zl[j] += zshift1;
                }
                if ub[j].is_finite() {
                    zu[j] += zshift1;
                }
            }
        }

        // Level pairwise complementarity products.
        let mut xsum = 1.0;
        let mut zsum = 1.0;
        let mut mu = 1.0f64;
        for j in 0..n + m {
            if lb[j].is_finite() {
                xsum += xl[j];
                zsum += zl[j];
                mu = xl[j].mul_add_c(zl[j], mu);
            }
            if ub[j].is_finite() {
                xsum += xu[j];
                zsum += zu[j];
                mu = xu[j].mul_add_c(zu[j], mu);
            }
        }
        let xshift2 = 0.5 * mu / zsum;
        let zshift2 = 0.5 * mu / xsum;
        for j in 0..n + m {
            xl[j] += xshift2;
            xu[j] += xshift2;
        }
        for j in 0..n + m {
            if lb[j].is_finite() {
                zl[j] += zshift2;
            }
            if ub[j].is_finite() {
                zu[j] += zshift2;
            }
        }
        iterate.initialize(&x, &xl, &xu, &y, &zl, &zu);
        self.best_complementarity = iterate.complementarity();
        Ok(())
    }

    fn predictor(&mut self, kkt: &mut dyn KktSolver, iterate: &Iterate, info: &mut Info, step: &mut Step) -> LuResult<()> {
        let nm = iterate.x().len();
        let (xl, xu, zl, zu) = (iterate.xl(), iterate.xu(), iterate.zl(), iterate.zu());

        // sl = -xl.*zl
        let mut sl = vec![0.0; nm];
        for j in 0..nm {
            if iterate.has_barrier_lb(j) {
                sl[j] = -xl[j] * zl[j];
            }
        }
        // su = -xu.*zu
        let mut su = vec![0.0; nm];
        for j in 0..nm {
            if iterate.has_barrier_ub(j) {
                su[j] = -xu[j] * zu[j];
            }
        }
        self.solve_newton_system(kkt, iterate, info, &sl, &su, step)
    }

    fn add_corrector(&mut self, kkt: &mut dyn KktSolver, iterate: &Iterate, info: &mut Info, step: &mut Step) -> LuResult<()> {
        let nm = iterate.x().len();
        let (xl, xu, zl, zu) = (iterate.xl(), iterate.xu(), iterate.zl(), iterate.zu());
        let mu = iterate.mu();

        // Choose centering parameter.
        let sigma = {
            let (dxl, dxu, dzl, dzu) = (&step.xl, &step.xu, &step.zl, &step.zu);
            let step_xl = step_to_boundary(xl, dxl, 1.0).0;
            let step_xu = step_to_boundary(xu, dxu, 1.0).0;
            let step_zl = step_to_boundary(zl, dzl, 1.0).0;
            let step_zu = step_to_boundary(zu, dzu, 1.0).0;
            let maxp = cmin(step_xl, step_xu);
            let maxd = cmin(step_zl, step_zu);
            let mut muaff = 0.0f64;
            let mut num_finite = 0;
            for j in 0..nm {
                if iterate.has_barrier_lb(j) {
                    let a = maxp.mul_add_c(dxl[j], xl[j]);
                    let b = maxd.mul_add_c(dzl[j], zl[j]);
                    muaff = a.mul_add_c(b, muaff);
                    num_finite += 1;
                }
                if iterate.has_barrier_ub(j) {
                    let a = maxp.mul_add_c(dxu[j], xu[j]);
                    let b = maxd.mul_add_c(dzu[j], zu[j]);
                    muaff = a.mul_add_c(b, muaff);
                    num_finite += 1;
                }
            }
            muaff /= num_finite as f64;
            let ratio = muaff / mu;
            ratio * ratio * ratio
        };

        // sl = -xl.*zl + sigma*mu - dxl.*dzl
        let mut sl = vec![0.0; nm];
        for j in 0..nm {
            if iterate.has_barrier_lb(j) {
                sl[j] = (-step.xl[j]).mul_add_c(step.zl[j], (-xl[j]).mul_add_c(zl[j], sigma * mu));
            }
        }
        // su = -xu.*zu + sigma*mu - dxu.*dzu
        let mut su = vec![0.0; nm];
        for j in 0..nm {
            if iterate.has_barrier_ub(j) {
                su[j] = (-step.xu[j]).mul_add_c(step.zu[j], (-xu[j]).mul_add_c(zu[j], sigma * mu));
            }
        }
        self.solve_newton_system(kkt, iterate, info, &sl, &su, step)
    }

    fn centring(&mut self, kkt: &mut dyn KktSolver, iterate: &Iterate, info: &mut Info, step: &mut Step, mu: f64) -> LuResult<()> {
        let nm = iterate.x().len();
        let (xl, xu, zl, zu) = (iterate.xl(), iterate.xu(), iterate.zl(), iterate.zu());
        // Set sigma to 1 for pure centring
        let sigma = 1.0;
        let mut sl = vec![0.0; nm];
        let mut su = vec![0.0; nm];
        // sl = -xl.*zl + sigma*mu
        for j in 0..nm {
            if iterate.has_barrier_lb(j) {
                sl[j] = (-xl[j]).mul_add_c(zl[j], sigma * mu);
            }
        }
        // su = -xu.*zu + sigma*mu
        for j in 0..nm {
            if iterate.has_barrier_ub(j) {
                su[j] = (-xu[j]).mul_add_c(zu[j], sigma * mu);
            }
        }
        self.solve_newton_system(kkt, iterate, info, &sl, &su, step)
    }

    /// Computes the ratio max(xj*zj)/min(xj*zj) (including mu) and the #
    /// products outside [0.1*mu, 10*mu], of the iterate or of the given
    /// point [xl, xu, zl, zu]; prints them if print.
    fn assess_centrality(&mut self, iterate: &Iterate, point: Option<[&[f64]; 4]>, mu: f64, print: bool) {
        let [xl, xu, zl, zu] = point.unwrap_or([iterate.xl(), iterate.xu(), iterate.zl(), iterate.zu()]);
        let nm = xl.len();
        let mut minxz = f64::INFINITY;
        let mut maxxz = 0.0;
        const GAMMA: f64 = 0.1;
        self.bad_products = 0;

        for j in 0..nm {
            if iterate.has_barrier_lb(j) {
                let product = xl[j] * zl[j];
                if product < GAMMA * mu || product > mu / GAMMA {
                    self.bad_products += 1;
                }
                minxz = cmin(minxz, product);
                maxxz = cmax(maxxz, product);
            }
        }
        for j in 0..nm {
            if iterate.has_barrier_ub(j) {
                let product = xu[j] * zu[j];
                if product < GAMMA * mu || product > mu / GAMMA {
                    self.bad_products += 1;
                }
                minxz = cmin(minxz, product);
                maxxz = cmax(maxxz, product);
            }
        }
        maxxz = cmax(maxxz, mu);
        minxz = cmin(minxz, mu);
        self.centring_ratio = maxxz / minxz;

        if print {
            self.control.log(&format!(
                "\txj*zj in [ {}, {}]; Ratio = {}; (xj*zj / mu) not_in [0.1, 10]: {}\n",
                sci(minxz / mu, 8, 2),
                sci(maxxz / mu, 8, 2),
                sci(self.centring_ratio, 8, 2),
                self.bad_products
            ));
        }
    }

    /// True if the centring step is to be accepted: the ratio of the new
    /// point is below centringRatioReduction times the previous one or the
    /// number of outlier products is reduced.
    fn evaluate_centring_step(&mut self, iterate: &Iterate, step: &Step, prev_ratio: f64, prev_bad: Int) -> bool {
        self.step_sizes(iterate, step, true);
        let nm = iterate.x().len();
        let mut xl_temp = iterate.xl().to_vec();
        let mut xu_temp = iterate.xu().to_vec();
        let mut zl_temp = iterate.zl().to_vec();
        let mut zu_temp = iterate.zu().to_vec();
        let (sp, sd) = (self.step_primal, self.step_dual);

        // perform temporary step
        for j in 0..nm {
            if iterate.has_barrier_lb(j) {
                xl_temp[j] = sp.mul_add_c(step.xl[j], xl_temp[j]);
            }
            if iterate.has_barrier_ub(j) {
                xu_temp[j] = sp.mul_add_c(step.xu[j], xu_temp[j]);
            }
            if iterate.has_barrier_lb(j) {
                zl_temp[j] = sd.mul_add_c(step.zl[j], zl_temp[j]);
            }
            if iterate.has_barrier_ub(j) {
                zu_temp[j] = sd.mul_add_c(step.zu[j], zu_temp[j]);
            }
        }

        // compute temporary mu
        let mut mu_temp = 0.0f64;
        let mut num_finite = 0;
        for j in 0..nm {
            if iterate.has_barrier_lb(j) {
                mu_temp = xl_temp[j].mul_add_c(zl_temp[j], mu_temp);
                num_finite += 1;
            }
            if iterate.has_barrier_ub(j) {
                mu_temp = xu_temp[j].mul_add_c(zu_temp[j], mu_temp);
                num_finite += 1;
            }
        }
        mu_temp /= num_finite as f64;

        // assess quality of temporary point
        self.assess_centrality(iterate, Some([&xl_temp, &xu_temp, &zl_temp, &zu_temp]), mu_temp, false);

        self.centring_ratio < self.control.centring_ratio_reduction() * prev_ratio
            || self.bad_products < prev_bad
    }

    fn step_sizes(&mut self, iterate: &Iterate, step: &Step, is_centring: bool) {
        let (xl, xu, zl, zu) = (iterate.xl(), iterate.xu(), iterate.zl(), iterate.zu());
        let (dxl, dxu, dzl, dzu) = (&step.xl, &step.xu, &step.zl, &step.zu);
        const GAMMAF: f64 = 0.9;
        let gammaa = 1.0 / (1.0 - GAMMAF);

        let (step_xl, block_xl) = step_to_boundary(xl, dxl, 1.0);
        let (step_xu, block_xu) = step_to_boundary(xu, dxu, 1.0);
        let (step_zl, block_zl) = step_to_boundary(zl, dzl, 1.0);
        let (step_zu, block_zu) = step_to_boundary(zu, dzu, 1.0);
        // std::fmin
        let maxp = step_xl.min(step_xu);
        let maxd = step_zl.min(step_zu);
        let mut mufull = 0.0f64;
        let mut num_finite = 0;
        for j in 0..xl.len() {
            if iterate.has_barrier_lb(j) {
                let a = maxp.mul_add_c(dxl[j], xl[j]);
                let b = maxd.mul_add_c(dzl[j], zl[j]);
                mufull = a.mul_add_c(b, mufull);
                num_finite += 1;
            }
            if iterate.has_barrier_ub(j) {
                let a = maxp.mul_add_c(dxu[j], xu[j]);
                let b = maxd.mul_add_c(dzu[j], zu[j]);
                mufull = a.mul_add_c(b, mufull);
                num_finite += 1;
            }
        }
        mufull /= num_finite as f64;
        mufull /= gammaa;

        let mut alphap = 1.0;
        let mut alphad = 1.0;
        if maxp < 1.0 {
            if step_xl <= step_xu {
                let blockp = block_xl as usize;
                let buffer = mufull / maxd.mul_add_c(dzl[blockp], zl[blockp]);
                alphap = (xl[blockp] - buffer) / (-dxl[blockp]);
            } else {
                let blockp = block_xu as usize;
                let buffer = mufull / maxd.mul_add_c(dzu[blockp], zu[blockp]);
                alphap = (xu[blockp] - buffer) / (-dxu[blockp]);
            }
            alphap = cmax(alphap, GAMMAF * maxp);
            alphap = cmin(alphap, 1.0);
        }
        if maxd < 1.0 {
            if step_zl <= step_zu {
                let blockd = block_zl as usize;
                let buffer = mufull / maxp.mul_add_c(dxl[blockd], xl[blockd]);
                alphad = (zl[blockd] - buffer) / (-dzl[blockd]);
            } else {
                let blockd = block_zu as usize;
                let buffer = mufull / maxp.mul_add_c(dxu[blockd], xu[blockd]);
                alphad = (zu[blockd] - buffer) / (-dzu[blockd]);
            }
            alphad = cmax(alphad, GAMMAF * maxd);
            alphad = cmin(alphad, 1.0);
        }
        self.step_primal = cmin(alphap, 1.0 - 1e-6);
        self.step_dual = cmin(alphad, 1.0 - 1e-6);

        if is_centring {
            // When computing stepsizes for a centring step, reduce them by
            // centringAlphaScaling, so that the point is well centred and
            // does not get too close to the boundary.
            self.step_primal = alphap * self.control.centring_alpha_scaling();
            self.step_dual = alphad * self.control.centring_alpha_scaling();
        }
    }

    fn make_step(&mut self, iterate: &mut Iterate, step: &Step, is_centring: bool) {
        self.step_sizes(iterate, step, is_centring);
        iterate.update(
            self.step_primal,
            Some(&step.x),
            Some(&step.xl),
            Some(&step.xu),
            self.step_dual,
            Some(&step.y),
            Some(&step.zl),
            Some(&step.zu),
        );
        if !is_centring {
            if cmin(self.step_primal, self.step_dual) < 0.05 {
                self.num_bad_iter += 1;
            } else {
                self.num_bad_iter = 0;
            }
            self.best_complementarity = cmin(self.best_complementarity, iterate.complementarity());
        }
    }

    /// Solves the Newton system with right-hand sides rb, rc, rl, ru from
    /// the iterate and sl, su.
    fn solve_newton_system(
        &mut self,
        kkt: &mut dyn KktSolver,
        iterate: &Iterate,
        info: &mut Info,
        sl: &[f64],
        su: &[f64],
        step: &mut Step,
    ) -> LuResult<()> {
        let model = iterate.model();
        let m = model.rows();
        let n = model.cols();
        let ai = model.ai();
        let (xl, xu, zl, zu) = (iterate.xl(), iterate.xu(), iterate.zl(), iterate.zu());
        let ev = iterate.eval();
        let (rb, rc, rl, ru) = (&ev.rb, &ev.rc, &ev.rl, &ev.ru);
        let Step {
            x: dx,
            xl: dxl,
            xu: dxu,
            y: dy,
            zl: dzl,
            zu: dzu,
        } = step;

        // Build RHS for KKT system.
        let mut rhs1 = vec![0.0; n + m];
        for j in 0..n + m {
            rhs1[j] = -rc[j];
        }
        for j in 0..n + m {
            let rlj = rl[j];
            let ruj = ru[j];
            if iterate.has_barrier_lb(j) {
                rhs1[j] += zl[j].mul_add_c(rlj, sl[j]) / xl[j];
            }
            if iterate.has_barrier_ub(j) {
                rhs1[j] -= (-zu[j]).mul_add_c(ruj, su[j]) / xu[j];
            }
            if iterate.state_of(j) == State::Fixed {
                rhs1[j] = 0.0;
            }
        }
        let rhs2 = rb.clone();

        // Solve KKT system.
        let tol = self.control.kkt_tol() * ev.mu.sqrt();
        kkt.solve(&rhs1, &rhs2, tol, dx, dy, info)?;
        if info.errflag != 0 {
            return Ok(());
        }

        // Recover solution to Newton system.
        for v in dy.iter_mut() {
            *v *= -1.0;
        }
        for j in 0..n + m {
            match iterate.state_of(j) {
                State::Fixed | State::Free => {
                    dxl[j] = 0.0;
                    dzl[j] = 0.0;
                }
                State::Barrier => {
                    dxl[j] = dx[j] - rl[j];
                    dzl[j] = (-zl[j]).mul_add_c(dxl[j], sl[j]) / xl[j];
                }
            }
        }
        for j in 0..n + m {
            match iterate.state_of(j) {
                State::Fixed | State::Free => {
                    dxu[j] = 0.0;
                    dzu[j] = 0.0;
                }
                State::Barrier => {
                    dxu[j] = ru[j] - dx[j];
                    dzu[j] = (-zu[j]).mul_add_c(dxu[j], su[j]) / xu[j];
                }
            }
        }

        // Shift residual to the last two block equations.
        for j in 0..n + m {
            if iterate.state_of(j) == State::Barrier {
                let atdy = dot_column(ai, j, dy);
                let rcj = rc[j];
                if xl[j].is_finite() && xu[j].is_finite() {
                    if zl[j] * xu[j] >= zu[j] * xl[j] {
                        dzl[j] = rcj + dzu[j] - atdy;
                    } else {
                        dzu[j] = -rcj + dzl[j] + atdy;
                    }
                } else if xl[j].is_finite() {
                    dzl[j] = rcj + dzu[j] - atdy;
                } else {
                    dzu[j] = -rcj + dzl[j] + atdy;
                }
            }
        }
        Ok(())
    }

    pub fn print_header(&self) {
        let c = self.control;
        let mut s = format!(
            " {}  {}  {}  {}  {}  {}",
            fmt("Iter", 4),
            fmt("primal obj", 15),
            fmt("dual obj", 15),
            fmt("pinf", 9),
            fmt("dinf", 9),
            fmt("gap", 8)
        );
        if !c.timeless_log() {
            s += &format!("   {}", fmt("time", 7));
        }
        c.log(&s);
        c.debug_out(
            1,
            &format!(
                "  {}  {} {}  {} {}",
                fmt("stepsizes", 9),
                fmt("pivots", 7),
                fmt("kktiter", 7),
                fmt("P.fixed", 7),
                fmt("D.fixed", 7)
            ),
        );
        c.debug_out(4, &format!("  {}", fmt("svdmin(B)", 9)));
        c.debug_out(4, &format!("  {}", fmt("density", 8)));
        c.log("\n");
    }

    fn print_output(&self, kkt: &dyn KktSolver, iterate: &Iterate, info: &mut Info) -> LuResult<()> {
        let c = self.control;
        let ipm_optimal = iterate.feasible() && iterate.optimal();

        let logging_pobj = iterate.pobjective_after_postproc();
        let logging_dobj = iterate.dobjective_after_postproc();
        // relative primal and dual infeasibility, relative objective gap
        let logging_presidual = iterate.presidual() / iterate.bounds_measure;
        let logging_dresidual = iterate.dresidual() / iterate.costs_measure;
        let logging_gap = (logging_pobj - logging_dobj).abs()
            / 0.5f64.mul_add_c((logging_pobj + logging_dobj).abs(), 1.0);

        let mut s = format!(
            " {}{}  {}  {}  {}  {}  {}",
            fmt(info.iter, 3),
            if ipm_optimal { "*" } else { " " },
            sci(logging_pobj, 15, 8),
            sci(logging_dobj, 15, 8),
            sci(logging_presidual, 9, 2),
            sci(logging_dresidual, 9, 2),
            sci(logging_gap, 8, 2)
        );
        if !c.timeless_log() {
            s += &format!("  {}", time(c.elapsed()));
        }
        c.log(&s);
        c.debug_out(
            1,
            &format!(
                "  {} {}  {} {}",
                fixed(self.step_primal, 4, 2),
                fixed(self.step_dual, 4, 2),
                fmt(kkt.basis_changes(), 7),
                fmt(kkt.iter(), 7)
            ),
        );
        c.debug_out(
            1,
            &format!("  {} {}", fmt(info.dual_dropped, 7), fmt(info.primal_dropped, 7)),
        );

        match kkt.basis() {
            Some(basis) => {
                if c.debug(4) {
                    c.debug_out(4, &format!("  {}", sci(basis.min_singular_value()?, 9, 2)));
                    let timer = Instant::now();
                    let density = basis.density_inverse();
                    info.time_symb_invert += timer.elapsed().as_secs_f64();
                    c.debug_out(4, &format!("  {}", sci(density, 8, 2)));
                }
            }
            None => {
                c.debug_out(4, &format!("  {}", fmt("-", 9)));
                c.debug_out(4, &format!("  {}", fmt("-", 8)));
            }
        }
        c.log("\n");
        Ok(())
    }
}
