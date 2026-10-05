//! Model (model.h/.cc): the interface between the user LP
//!
//!   minimize   obj'x
//!   subject to A*x {=,<,>} rhs, lbuser <= x <= ubuser,
//!
//! and the computational form
//!
//!   minimize   c'x
//!   subject to AI*x = b,                              (dual: y)
//!              x-xl = lb, xl >= 0,                    (dual: zl >= 0)
//!              x+xu = ub, xu >= 0.                    (dual: zu >= 0)
//!
//! where AI has m rows and n+m columns, the last m forming the identity.
//! The user model is (a) scaled (equilibration, and "flipping" variables
//! with only a finite upper bound) and (b) dualized if appropriate.

use super::control::Control;
use super::fmt::{g, sci, textline};
use super::sparse_matrix::{dot_column, scale_column, scatter_column, transpose, SparseMatrix};
use super::utils::{dot, infnorm};
use super::{
    cmax, cmin, frexp_exp, ldexp1, Info, Int, ERROR_ARGUMENT_NULL, ERROR_INVALID_DIMENSION,
    ERROR_INVALID_MATRIX, ERROR_INVALID_VECTOR, ERROR_NOT_IMPLEMENTED, BASIC, NONBASIC,
    NONBASIC_LB, NONBASIC_UB, SUPERBASIC,
};

const INF: f64 = f64::INFINITY;

/// The user LP as given to LpSolver::LoadModel (slices of the C arrays;
/// None for a NULL pointer)
pub struct UserLp<'a> {
    pub num_constr: Int,
    pub num_var: Int,
    pub ap: Option<&'a [Int]>,
    pub ai: Option<&'a [Int]>,
    pub ax: Option<&'a [f64]>,
    pub rhs: Option<&'a [f64]>,
    pub constr_type: Option<&'a [u8]>,
    pub offset: f64,
    pub obj: Option<&'a [f64]>,
    pub lbuser: Option<&'a [f64]>,
    pub ubuser: Option<&'a [f64]>,
}

#[derive(Default)]
pub struct Model {
    // Computational form model.
    dualized: bool,
    num_rows: usize,
    num_cols: usize,
    num_dense_cols: Int,
    nz_dense: Int,
    ai: SparseMatrix,  // matrix AI columnwise
    ait: SparseMatrix, // matrix AI rowwise
    b: Vec<f64>,
    c: Vec<f64>,
    lb: Vec<f64>,
    ub: Vec<f64>,
    norm_bounds: f64, // infinity norm of [b;lb;ub]
    norm_c: f64,      // infinity norm of c

    // User model after scaling.
    num_constr: usize,
    num_eqconstr: usize,
    num_var: usize,
    num_free_var: usize,
    num_entries: Int,
    boxed_vars: Vec<usize>,
    constr_type: Vec<u8>,
    norm_obj: f64, // Infnorm(obj) as given by user
    norm_rhs: f64, // Infnorm(rhs,lb,ub) as given by user
    offset: f64,
    scaled_obj: Vec<f64>,
    scaled_rhs: Vec<f64>,
    scaled_lbuser: Vec<f64>,
    scaled_ubuser: Vec<f64>,
    a: SparseMatrix, // is cleared after preprocessing

    // Data from ScaleModel() that is required by ScaleBack*().
    flipped_vars: Vec<usize>,
    colscale: Vec<f64>,
    rowscale: Vec<f64>,
}

/// CheckVectors: 0 if valid LP data vectors, negative otherwise
fn check_vectors(rhs: &[f64], constr_type: &[u8], obj: &[f64], lb: &[f64], ub: &[f64]) -> i32 {
    if !rhs.iter().all(|x| x.is_finite()) {
        return -1;
    }
    if !obj.iter().all(|x| x.is_finite()) {
        return -2;
    }
    for j in 0..obj.len() {
        if !lb[j].is_finite() && lb[j] != -INF {
            return -3;
        }
        if !ub[j].is_finite() && ub[j] != INF {
            return -3;
        }
        if lb[j] > ub[j] {
            return -3;
        }
    }
    if !constr_type.iter().all(|&t| t == b'=' || t == b'<' || t == b'>') {
        return -4;
    }
    0
}

/// CheckMatrix: 0 if A is a valid m-by-n matrix in CSC format
fn check_matrix(m: usize, n: usize, ap: &[Int], ai: &[Int], ax: &[f64]) -> i32 {
    if ap[0] != 0 {
        return -5;
    }
    for j in 0..n {
        if ap[j] > ap[j + 1] {
            return -5;
        }
    }
    if !ax[..ap[n] as usize].iter().all(|x| x.is_finite()) {
        return -6;
    }
    // Test for out of bound indices and duplicates.
    let mut marked = vec![-1i64; m];
    for j in 0..n {
        for p in ap[j] as usize..ap[j + 1] as usize {
            let i = ai[p];
            if i < 0 || i as usize >= m {
                return -7;
            }
            if marked[i as usize] == j as i64 {
                return -8;
            }
            marked[i as usize] = j as i64;
        }
    }
    0
}

/// Returns a power-of-2 factor s such that s*2^exp becomes closer to the
/// interval [2^expmin, 2^expmax].
fn equilibration_factor(expmin: i32, expmax: i32, exp: i32) -> f64 {
    if exp < expmin {
        return ldexp1((expmin - exp + 1) / 2);
    }
    if exp > expmax {
        return ldexp1(-((exp - expmax + 1) / 2));
    }
    1.0
}

impl Model {
    pub fn empty(&self) -> bool {
        self.num_cols == 0
    }
    pub fn rows(&self) -> usize {
        self.num_rows
    }
    pub fn cols(&self) -> usize {
        self.num_cols
    }
    pub fn dualized(&self) -> bool {
        self.dualized
    }
    pub fn ai(&self) -> &SparseMatrix {
        &self.ai
    }
    pub fn ait(&self) -> &SparseMatrix {
        &self.ait
    }
    pub fn offset(&self) -> f64 {
        self.offset
    }
    pub fn b(&self) -> &[f64] {
        &self.b
    }
    pub fn c(&self) -> &[f64] {
        &self.c
    }
    pub fn lb(&self) -> &[f64] {
        &self.lb
    }
    pub fn ub(&self) -> &[f64] {
        &self.ub
    }
    pub fn norm_bounds(&self) -> f64 {
        self.norm_bounds
    }
    pub fn norm_c(&self) -> f64 {
        self.norm_c
    }

    /// Model::Load: initializes from the user LP; on invalid input returns
    /// an error code and the model stays empty.
    pub fn load(control: &Control, lp: &UserLp) -> (Model, Int) {
        let mut model = Model::default();
        let errflag = model.copy_input(lp);
        if errflag != 0 {
            return (Model::default(), errflag);
        }
        control.log(&format!(
            "Input\n{}{}\n{}{}\n{}{}\n{}{}\n{}{}\n",
            textline("Number of variables:"),
            model.num_var,
            textline("Number of free variables:"),
            model.num_free_var,
            textline("Number of constraints:"),
            model.num_constr,
            textline("Number of equality constraints:"),
            model.num_eqconstr,
            textline("Number of matrix entries:"),
            model.num_entries
        ));
        model.print_coefficient_range(control);
        model.scale_model(control);

        // Make an automatic decision for dualization if not specified by
        // user: -2 Filippo style, -1 Lukas style, 0 no, 1 yes.
        let mut dualize = control.dualize();
        let dualize_lukas = lp.num_constr > 2 * lp.num_var;
        let dualize_filippo = model.filippo_dualization_test();
        if dualize == -1 {
            dualize = dualize_lukas as Int;
        } else if dualize == -2 {
            dualize = dualize_filippo as Int;
        }
        if dualize != 0 {
            model.load_dual();
        } else {
            model.load_primal();
        }

        model.a.clear();
        model.ait = transpose(&model.ai);
        model.find_dense_columns();
        model.norm_c = infnorm(&model.c);
        model.norm_bounds = infnorm(&model.b);
        for &x in model.lb.iter().chain(model.ub.iter()) {
            if x.is_finite() {
                model.norm_bounds = cmax(model.norm_bounds, x.abs());
            }
        }
        model.print_preprocessing_log(control);
        (model, 0)
    }

    fn filippo_dualization_test(&self) -> bool {
        false
    }

    /// Writes statistics of input data and preprocessing to info.
    pub fn get_info(&self, info: &mut Info) {
        info.num_var = self.num_var as Int;
        info.num_constr = self.num_constr as Int;
        info.num_entries = self.num_entries;
        info.num_rows_solver = self.num_rows as Int;
        info.num_cols_solver = (self.num_cols + self.num_rows) as Int; // including slack columns
        info.num_entries_solver = self.ai.entries();
        info.dualized = self.dualized as Int;
        info.dense_cols = self.num_dense_cols;
    }

    /// Transforms a point from the user model to the solver model; None
    /// components are zero.
    pub fn presolve_starting_point(
        &self,
        x_user: Option<&[f64]>,
        slack_user: Option<&[f64]>,
        y_user: Option<&[f64]>,
        z_user: Option<&[f64]>,
        x_solver: &mut [f64],
        y_solver: &mut [f64],
        z_solver: &mut [f64],
    ) {
        let copy = |v: Option<&[f64]>, n: usize| v.map_or(vec![0.0; n], |v| v[..n].to_vec());
        let mut x_temp = copy(x_user, self.num_var);
        let mut slack_temp = copy(slack_user, self.num_constr);
        let mut y_temp = copy(y_user, self.num_constr);
        let mut z_temp = copy(z_user, self.num_var);
        self.scale_point(&mut x_temp, &mut slack_temp, &mut y_temp, &mut z_temp);
        self.dualize_basic_solution(&x_temp, &slack_temp, &y_temp, &z_temp, x_solver, y_solver, z_solver);
    }

    /// Inverse of PostsolveInteriorSolution for an IPM starting point
    /// (sign conditions checked; not implemented for dualized models).
    #[allow(clippy::type_complexity)]
    pub fn presolve_ipm_starting_point(
        &self,
        user: [Option<&[f64]>; 7],
        x_solver: &mut [f64],
        xl_solver: &mut [f64],
        xu_solver: &mut [f64],
        y_solver: &mut [f64],
        zl_solver: &mut [f64],
        zu_solver: &mut [f64],
    ) -> Int {
        let [Some(x_user), Some(xl_user), Some(xu_user), Some(slack_user), Some(y_user), Some(zl_user), Some(zu_user)] =
            user
        else {
            return ERROR_ARGUMENT_NULL;
        };
        if self.dualized {
            return ERROR_NOT_IMPLEMENTED;
        }
        let nv = self.num_var;
        let nc = self.num_constr;
        // Copy user point into workspace and apply model scaling.
        let mut x_temp = x_user[..nv].to_vec();
        let mut xl_temp = xl_user[..nv].to_vec();
        let mut xu_temp = xu_user[..nv].to_vec();
        let mut slack_temp = slack_user[..nc].to_vec();
        let mut y_temp = y_user[..nc].to_vec();
        let mut zl_temp = zl_user[..nv].to_vec();
        let mut zu_temp = zu_user[..nv].to_vec();
        self.scale_point7(
            &mut x_temp,
            &mut xl_temp,
            &mut xu_temp,
            &mut slack_temp,
            &mut y_temp,
            &mut zl_temp,
            &mut zu_temp,
        );

        // Check that point is compatible with bounds.
        if !x_temp.iter().all(|x| x.is_finite()) {
            return ERROR_INVALID_VECTOR;
        }
        let lbu = &self.scaled_lbuser;
        let ubu = &self.scaled_ubuser;
        for j in 0..nv {
            if !(xl_temp[j] >= 0.0)
                || (lbu[j] == -INF && xl_temp[j] != INF)
                || (lbu[j] != -INF && xl_temp[j] == INF)
            {
                return ERROR_INVALID_VECTOR;
            }
        }
        for j in 0..nv {
            if !(xu_temp[j] >= 0.0)
                || (ubu[j] == INF && xu_temp[j] != INF)
                || (ubu[j] != INF && xu_temp[j] == INF)
            {
                return ERROR_INVALID_VECTOR;
            }
        }
        for i in 0..nc {
            let ct = self.constr_type[i];
            if !slack_temp[i].is_finite()
                || (ct == b'=' && !(slack_temp[i] == 0.0))
                || (ct == b'<' && !(slack_temp[i] >= 0.0))
                || (ct == b'>' && !(slack_temp[i] <= 0.0))
            {
                return ERROR_INVALID_VECTOR;
            }
        }
        for i in 0..nc {
            let ct = self.constr_type[i];
            if !y_temp[i].is_finite()
                || (ct == b'<' && !(y_temp[i] <= 0.0))
                || (ct == b'>' && !(y_temp[i] >= 0.0))
            {
                return ERROR_INVALID_VECTOR;
            }
        }
        for j in 0..nv {
            if !(zl_temp[j] >= 0.0 && zl_temp[j] < INF) || (lbu[j] == -INF && zl_temp[j] != 0.0) {
                return ERROR_INVALID_VECTOR;
            }
        }
        for j in 0..nv {
            if !(zu_temp[j] >= 0.0 && zu_temp[j] < INF) || (ubu[j] == INF && zu_temp[j] != 0.0) {
                return ERROR_INVALID_VECTOR;
            }
        }

        // DualizeIPMStartingPoint (not dualized)
        let n = self.num_cols;
        let m = self.num_rows;
        x_solver[..nv].copy_from_slice(&x_temp);
        x_solver[n..n + nc].copy_from_slice(&slack_temp);
        xl_solver[..nv].copy_from_slice(&xl_temp);
        xu_solver[..nv].copy_from_slice(&xu_temp);
        y_solver[..nc].copy_from_slice(&y_temp);
        zl_solver[..nv].copy_from_slice(&zl_temp);
        zu_solver[..nv].copy_from_slice(&zu_temp);
        for i in 0..m {
            match self.constr_type[i] {
                b'=' => {
                    // For a fixed slack variable xl, xu, zl and zu won't be
                    // used by the IPM. Just put them to zero.
                    xl_solver[n + i] = 0.0;
                    xu_solver[n + i] = 0.0;
                    zl_solver[n + i] = 0.0;
                    zu_solver[n + i] = 0.0;
                }
                b'<' => {
                    xl_solver[n + i] = slack_temp[i];
                    xu_solver[n + i] = INF;
                    zl_solver[n + i] = -y_temp[i];
                    zu_solver[n + i] = 0.0;
                }
                b'>' => {
                    xl_solver[n + i] = INF;
                    xu_solver[n + i] = -slack_temp[i];
                    zl_solver[n + i] = 0.0;
                    zu_solver[n + i] = y_temp[i];
                }
                _ => {}
            }
        }
        0
    }

    /// Recovers the solution to the user model from an IPM iterate.
    pub fn postsolve_interior_solution(
        &self,
        solver: [&[f64]; 6],
        out: [Option<&mut [f64]>; 7],
    ) {
        let mut u = self.user_interior(solver);
        self.scale_back_interior_solution(&mut u);
        for (o, v) in out.into_iter().zip(u.iter()) {
            if let Some(o) = o {
                o[..v.len()].copy_from_slice(v);
            }
        }
    }

    /// DualizeBackInteriorSolution into fresh user vectors [x, xl, xu,
    /// slack, y, zl, zu]
    fn user_interior(&self, s: [&[f64]; 6]) -> [Vec<f64>; 7] {
        let nv = self.num_var;
        let nc = self.num_constr;
        let mut u = [
            vec![0.0; nv],
            vec![0.0; nv],
            vec![0.0; nv],
            vec![0.0; nc],
            vec![0.0; nc],
            vec![0.0; nv],
            vec![0.0; nv],
        ];
        self.dualize_back_interior_solution(s, &mut u);
        u
    }

    /// Evaluates the solution to the user model obtained from postsolving
    /// the IPM iterate.
    pub fn evaluate_interior_solution(&self, solver: [&[f64]; 6], info: &mut Info) {
        let nv = self.num_var;
        let nc = self.num_constr;
        // Build solution to scaled user model.
        let mut u = self.user_interior(solver);
        {
            let [x, xl, xu, slack, y, zl, zu] = &u;
            let lbu = &self.scaled_lbuser;
            let ubu = &self.scaled_ubuser;

            // Build residuals for scaled model.
            // rl = lb-x+xl
            let mut rl = vec![0.0; nv];
            for j in 0..nv {
                if lbu[j].is_finite() {
                    rl[j] = lbu[j] - x[j] + xl[j];
                }
            }
            // ru = ub-x-xu
            let mut ru = vec![0.0; nv];
            for j in 0..nv {
                if ubu[j].is_finite() {
                    ru[j] = ubu[j] - x[j] - xu[j];
                }
            }
            // rb = rhs-slack-A*x
            // Add rhs at the end to avoid losing digits when x is huge.
            let mut rb = vec![0.0; nc];
            self.multiply_with_scaled_matrix(x, -1.0, &mut rb, b'N');
            for i in 0..nc {
                rb[i] -= slack[i];
            }
            for i in 0..nc {
                rb[i] += self.scaled_rhs[i];
            }
            // rc = obj-zl+zu-A'y
            // Add obj at the end to avoid losing digits when y, z are huge.
            let mut rc = vec![0.0; nv];
            self.multiply_with_scaled_matrix(y, -1.0, &mut rc, b'T');
            for j in 0..nv {
                rc[j] -= zl[j] - zu[j];
            }
            for j in 0..nv {
                rc[j] += self.scaled_obj[j];
            }

            self.scale_back_residuals(&mut rb, &mut rc, &mut rl, &mut ru);
            let mut presidual = infnorm(&rb);
            presidual = cmax(presidual, infnorm(&rl));
            presidual = cmax(presidual, infnorm(&ru));
            let dresidual = infnorm(&rc);

            let pobjective = self.offset + dot(&self.scaled_obj, x);
            let mut dobjective = self.offset + dot(&self.scaled_rhs, y);
            for j in 0..nv {
                if lbu[j].is_finite() {
                    dobjective = lbu[j].mul_add(zl[j], dobjective);
                }
                if ubu[j].is_finite() {
                    dobjective = (-ubu[j]).mul_add(zu[j], dobjective);
                }
            }
            let objective_gap =
                (pobjective - dobjective) / 0.5f64.mul_add((pobjective + dobjective).abs(), 1.0);

            let mut complementarity = 0.0f64;
            for j in 0..nv {
                if lbu[j].is_finite() {
                    complementarity = xl[j].mul_add(zl[j], complementarity);
                }
                if ubu[j].is_finite() {
                    complementarity = xu[j].mul_add(zu[j], complementarity);
                }
            }
            // vectorized by 8 in C++: unfused for the first nc/8*8 terms
            // (utils::dot_blocked)
            let nb = if nc >= 8 { nc - nc % 8 } else { 0 };
            for i in 0..nb {
                complementarity -= y[i] * slack[i];
            }
            for i in nb..nc {
                complementarity = (-y[i]).mul_add(slack[i], complementarity);
            }

            info.abs_presidual = presidual;
            info.abs_dresidual = dresidual;
            info.rel_presidual = presidual / (1.0 + self.norm_rhs);
            info.rel_dresidual = dresidual / (1.0 + self.norm_obj);
            info.pobjval = pobjective;
            info.dobjval = dobjective;
            info.rel_objgap = objective_gap;
            info.complementarity = complementarity;
        }
        // For computing the norms of the user variables, we have to scale
        // back.
        self.scale_back_interior_solution(&mut u);
        let [x, _, _, _, y, zl, zu] = &u;
        info.normx = infnorm(x);
        info.normy = infnorm(y);
        info.normz = cmax(infnorm(zl), infnorm(zu));
    }

    /// Basic solution to the scaled user model, corrected to the bounds
    /// given by the basis: (x, slack, y, z, cbasis, vbasis)
    #[allow(clippy::type_complexity)]
    fn user_basic(
        &self,
        x_solver: &[f64],
        y_solver: &[f64],
        z_solver: &[f64],
        basic_status_solver: &[Int],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<Int>, Vec<Int>) {
        let nv = self.num_var;
        let nc = self.num_constr;
        let mut x = vec![0.0; nv];
        let mut slack = vec![0.0; nc];
        let mut y = vec![0.0; nc];
        let mut z = vec![0.0; nv];
        let mut cbasis = vec![0; nc];
        let mut vbasis = vec![0; nv];
        self.dualize_back_basic_solution(x_solver, y_solver, z_solver, &mut x, &mut slack, &mut y, &mut z);
        self.dualize_back_basis(basic_status_solver, &mut cbasis, &mut vbasis);
        self.correct_scaled_basic_solution(&mut x, &mut slack, &mut y, &mut z, &cbasis, &vbasis);
        (x, slack, y, z, cbasis, vbasis)
    }

    /// Recovers the basic solution to the user model.
    pub fn postsolve_basic_solution(
        &self,
        x_solver: &[f64],
        y_solver: &[f64],
        z_solver: &[f64],
        basic_status_solver: &[Int],
        x_user: Option<&mut [f64]>,
        slack_user: Option<&mut [f64]>,
        y_user: Option<&mut [f64]>,
        z_user: Option<&mut [f64]>,
    ) {
        let (mut x, mut slack, mut y, mut z, _, _) =
            self.user_basic(x_solver, y_solver, z_solver, basic_status_solver);
        self.scale_back_basic_solution(&mut x, &mut slack, &mut y, &mut z);
        for (o, v) in [x_user, slack_user, y_user, z_user].into_iter().zip([&x, &slack, &y, &z]) {
            if let Some(o) = o {
                o[..v.len()].copy_from_slice(v);
            }
        }
    }

    /// Sets info.primal_infeas, dual_infeas, objval of the postsolved basic
    /// solution.
    pub fn evaluate_basic_solution(
        &self,
        x_solver: &[f64],
        y_solver: &[f64],
        z_solver: &[f64],
        basic_status_solver: &[Int],
        info: &mut Info,
    ) {
        let nv = self.num_var;
        let nc = self.num_constr;
        let (mut x, mut slack, mut y, mut z, _cbasis, vbasis) =
            self.user_basic(x_solver, y_solver, z_solver, basic_status_solver);
        let pobj = dot(&self.scaled_obj, &x);

        // Build infeasibilities in scaled user model.
        let mut xinfeas = vec![0.0; nv];
        let mut sinfeas = vec![0.0; nc];
        let mut yinfeas = vec![0.0; nc];
        let mut zinfeas = vec![0.0; nv];
        for j in 0..nv {
            if x[j] < self.scaled_lbuser[j] {
                xinfeas[j] = x[j] - self.scaled_lbuser[j];
            }
            if x[j] > self.scaled_ubuser[j] {
                xinfeas[j] = x[j] - self.scaled_ubuser[j];
            }
            if vbasis[j] != NONBASIC_LB && z[j] > 0.0 {
                zinfeas[j] = z[j];
            }
            if vbasis[j] != NONBASIC_UB && z[j] < 0.0 {
                zinfeas[j] = z[j];
            }
        }
        for i in 0..nc {
            if self.constr_type[i] == b'<' {
                if slack[i] < 0.0 {
                    sinfeas[i] = slack[i];
                }
                if y[i] > 0.0 {
                    yinfeas[i] = y[i];
                }
            }
            if self.constr_type[i] == b'>' {
                if slack[i] > 0.0 {
                    sinfeas[i] = slack[i];
                }
                if y[i] < 0.0 {
                    yinfeas[i] = y[i];
                }
            }
        }

        // Scale back basic solution and infeasibilities.
        self.scale_back_basic_solution(&mut x, &mut slack, &mut y, &mut z);
        self.scale_back_basic_solution(&mut xinfeas, &mut sinfeas, &mut yinfeas, &mut zinfeas);

        info.primal_infeas = cmax(infnorm(&xinfeas), infnorm(&sinfeas));
        info.dual_infeas = cmax(infnorm(&zinfeas), infnorm(&yinfeas));
        info.objval = pobj;
    }

    /// Recovers the basic statuses of the user model.
    pub fn postsolve_basis(
        &self,
        basic_status_solver: &[Int],
        cbasis_user: Option<&mut [Int]>,
        vbasis_user: Option<&mut [Int]>,
    ) {
        let mut cbasis = vec![0; self.num_constr];
        let mut vbasis = vec![0; self.num_var];
        self.dualize_back_basis(basic_status_solver, &mut cbasis, &mut vbasis);
        self.scale_back_basis(&mut cbasis, &mut vbasis);
        if let Some(c) = cbasis_user {
            c[..cbasis.len()].copy_from_slice(&cbasis);
        }
        if let Some(v) = vbasis_user {
            v[..vbasis.len()].copy_from_slice(&vbasis);
        }
    }

    fn copy_input(&mut self, lp: &UserLp) -> Int {
        let (
            Some(ap),
            Some(ai),
            Some(ax),
            Some(rhs),
            Some(constr_type),
            Some(obj),
            Some(lbuser),
            Some(ubuser),
        ) = (lp.ap, lp.ai, lp.ax, lp.rhs, lp.constr_type, lp.obj, lp.lbuser, lp.ubuser)
        else {
            return ERROR_ARGUMENT_NULL;
        };
        if lp.num_constr < 0 || lp.num_var <= 0 {
            return ERROR_INVALID_DIMENSION;
        }
        let nc = lp.num_constr as usize;
        let nv = lp.num_var as usize;
        if check_vectors(&rhs[..nc], &constr_type[..nc], &obj[..nv], lbuser, ubuser) != 0 {
            return ERROR_INVALID_VECTOR;
        }
        if check_matrix(nc, nv, ap, ai, ax) != 0 {
            return ERROR_INVALID_MATRIX;
        }
        self.num_constr = nc;
        self.num_eqconstr = constr_type[..nc].iter().filter(|&&t| t == b'=').count();
        self.num_var = nv;
        self.num_entries = ap[nv];
        self.num_free_var = 0;
        self.boxed_vars.clear();
        for j in 0..nv {
            if lbuser[j].is_infinite() && ubuser[j].is_infinite() {
                self.num_free_var += 1;
            }
            if lbuser[j].is_finite() && ubuser[j].is_finite() {
                self.boxed_vars.push(j);
            }
        }
        self.constr_type = constr_type[..nc].to_vec();
        self.offset = lp.offset;
        self.scaled_obj = obj[..nv].to_vec();
        self.scaled_rhs = rhs[..nc].to_vec();
        self.scaled_lbuser = lbuser[..nv].to_vec();
        self.scaled_ubuser = ubuser[..nv].to_vec();
        self.a
            .load_from_arrays(nc as Int, nv as Int, &ap[..nv], &ap[1..nv + 1], ai, ax);
        self.norm_obj = infnorm(&self.scaled_obj);
        self.norm_rhs = infnorm(&self.scaled_rhs);
        for &x in self.scaled_lbuser.iter().chain(self.scaled_ubuser.iter()) {
            if x.is_finite() {
                self.norm_rhs = cmax(self.norm_rhs, x.abs());
            }
        }
        0
    }

    fn scale_model(&mut self, control: &Control) {
        self.flipped_vars.clear();
        for j in 0..self.num_var {
            if self.scaled_ubuser[j].is_finite() && self.scaled_lbuser[j].is_infinite() {
                self.scaled_lbuser[j] = -self.scaled_ubuser[j];
                self.scaled_ubuser[j] = INF;
                scale_column(&mut self.a, j, -1.0);
                self.scaled_obj[j] *= -1.0;
                self.flipped_vars.push(j);
            }
        }
        self.colscale.clear();
        self.rowscale.clear();

        // Choose scaling method.
        if control.scale() >= 1 {
            self.equilibrate_matrix();
        }

        // Apply scaling to vectors.
        if !self.colscale.is_empty() {
            for j in 0..self.num_var {
                self.scaled_obj[j] *= self.colscale[j];
                self.scaled_lbuser[j] /= self.colscale[j];
                self.scaled_ubuser[j] /= self.colscale[j];
            }
        }
        if !self.rowscale.is_empty() {
            for i in 0..self.num_constr {
                self.scaled_rhs[i] *= self.rowscale[i];
            }
        }
    }

    /// Computational form without dualization:
    /// AI = [A eye(nc)], b = rhs, c = [obj; 0],
    /// lb = [lbuser; constr_type .== '>' ? -Inf : 0],
    /// ub = [ubuser; constr_type .== '<' ? +Inf : 0].
    fn load_primal(&mut self) {
        let nc = self.num_constr;
        let nv = self.num_var;
        self.num_rows = nc;
        self.num_cols = nv;
        self.dualized = false;

        // Copy A and append identity matrix.
        self.ai = self.a.clone();
        for i in 0..nc {
            self.ai.push_back(i as Int, 1.0);
            self.ai.add_column();
        }

        // Copy vectors and set bounds on slack variables.
        self.b = self.scaled_rhs.clone();
        self.c = vec![0.0; nv + nc];
        self.c[..nv].copy_from_slice(&self.scaled_obj);
        self.lb = vec![0.0; nv + nc];
        self.lb[..nv].copy_from_slice(&self.scaled_lbuser);
        self.ub = vec![0.0; nv + nc];
        self.ub[..nv].copy_from_slice(&self.scaled_ubuser);
        for i in 0..nc {
            let (l, u) = match self.constr_type[i] {
                b'=' => (0.0, 0.0),
                b'<' => (0.0, INF),
                _ => (-INF, 0.0),
            };
            self.lb[nv + i] = l;
            self.ub[nv + i] = u;
        }
    }

    /// Computational form with dualization:
    /// AI = [A' -eye(nv)[:,jboxed] eye(nv)], b = obj,
    /// c = [-rhs; ubuser[jb]; -lbuser],
    /// lb = [constr_type .== '>' ? 0 : -Inf; zeros(nb); zeros(nv)],
    /// ub = [constr_type .== '<' ? 0 : +Inf; Inf*ones(nb); Inf*ones(nv)].
    /// If variable j is free, the j-th slack variable gets a zero upper
    /// bound (fixed at zero) and a zero objective coefficient.
    fn load_dual(&mut self) {
        let nc = self.num_constr;
        let nv = self.num_var;
        self.num_rows = nv;
        self.num_cols = nc + self.boxed_vars.len();
        self.dualized = true;
        let n = self.num_cols;

        // Build AI.
        self.ai = transpose(&self.a);
        for j in 0..nv {
            if self.scaled_ubuser[j].is_finite() {
                self.ai.push_back(j as Int, -1.0);
                self.ai.add_column();
            }
        }
        for i in 0..nv {
            self.ai.push_back(i as Int, 1.0);
            self.ai.add_column();
        }

        // Build vectors.
        self.b = self.scaled_obj.clone();
        self.c = vec![0.0; n + nv];
        let mut put = 0;
        for &x in &self.scaled_rhs {
            self.c[put] = -x;
            put += 1;
        }
        for &x in &self.scaled_ubuser {
            if x.is_finite() {
                self.c[put] = x;
                put += 1;
            }
        }
        for &x in &self.scaled_lbuser {
            // If x is negative infinity, then the variable will be fixed and
            // we can give it any (finite) cost.
            self.c[put] = if x.is_finite() { -x } else { 0.0 };
            put += 1;
        }
        self.lb = vec![0.0; n + nv];
        self.ub = vec![0.0; n + nv];
        for i in 0..nc {
            let (l, u) = match self.constr_type[i] {
                b'=' => (-INF, INF),
                b'<' => (-INF, 0.0),
                _ => (0.0, INF),
            };
            self.lb[i] = l;
            self.ub[i] = u;
        }
        for j in nc..n {
            self.lb[j] = 0.0;
            self.ub[j] = INF;
        }
        for j in 0..nv {
            self.lb[n + j] = 0.0;
            self.ub[n + j] = if self.scaled_lbuser[j].is_finite() { INF } else { 0.0 };
        }
    }

    /// Recursively equilibrates A in infinity norm (Knight, Ruiz, Ucar),
    /// with factors truncated to powers of 2, until the entries are within
    /// [0.5,8).
    fn equilibrate_matrix(&mut self) {
        let m = self.a.rows() as usize;
        let n = self.a.cols() as usize;
        self.colscale.clear();
        self.rowscale.clear();

        const EXPMIN: i32 = 0;
        const EXPMAX: i32 = 3;
        const MAXROUND: usize = 10;

        // Quick return if entries are within the target range.
        let nz = self.a.colptr[n] as usize;
        let out_of_range = self.a.values[..nz].iter().any(|&x| {
            let exp = frexp_exp(x.abs());
            !(EXPMIN..=EXPMAX).contains(&exp)
        });
        if !out_of_range {
            return;
        }

        self.colscale = vec![1.0; n];
        self.rowscale = vec![1.0; m];
        let mut colmax = vec![0.0; n];
        let mut rowmax = vec![0.0; m];
        let a = &mut self.a;

        for _round in 0..MAXROUND {
            // Compute infinity norm of each row and column.
            rowmax.fill(0.0);
            for j in 0..n {
                colmax[j] = 0.0;
                for p in a.begin(j)..a.end(j) {
                    let i = a.index(p);
                    let xa = a.values[p].abs();
                    colmax[j] = cmax(colmax[j], xa);
                    rowmax[i] = cmax(rowmax[i], xa);
                }
            }
            // Replace rowmax and colmax entries by scaling factors from this
            // round.
            let mut out_of_range = false;
            for i in 0..m {
                rowmax[i] = equilibration_factor(EXPMIN, EXPMAX, frexp_exp(rowmax[i]));
                if rowmax[i] != 1.0 {
                    out_of_range = true;
                    self.rowscale[i] *= rowmax[i];
                }
            }
            for j in 0..n {
                colmax[j] = equilibration_factor(EXPMIN, EXPMAX, frexp_exp(colmax[j]));
                if colmax[j] != 1.0 {
                    out_of_range = true;
                    self.colscale[j] *= colmax[j];
                }
            }
            if !out_of_range {
                break;
            }
            // Rescale A.
            for j in 0..n {
                for p in a.begin(j)..a.end(j) {
                    a.values[p] *= colmax[j]; // column scaling
                    a.values[p] *= rowmax[a.index(p)]; // row scaling
                }
            }
        }
    }

    /// Classifies as "dense" the maximum # columns which have more than 40
    /// nonzeros and more than 10 times the # nonzeros of any column that is
    /// not dense; none if that gives more than 1000.
    fn find_dense_columns(&mut self) {
        self.num_dense_cols = 0;
        self.nz_dense = self.num_rows as Int + 1;

        let mut colcount: Vec<Int> = (0..self.num_cols).map(|j| self.ai.col_entries(j)).collect();
        colcount.sort_unstable();

        for j in 1..self.num_cols {
            if colcount[j] > std::cmp::max(40, 10 * colcount[j - 1]) {
                // j is the first dense column
                self.num_dense_cols = (self.num_cols - j) as Int;
                self.nz_dense = colcount[j];
                break;
            }
        }

        if self.num_dense_cols > 1000 {
            self.num_dense_cols = 0;
            self.nz_dense = self.num_rows as Int + 1;
        }
    }

    fn print_coefficient_range(&self, control: &Control) {
        // [min, max] of the nonzero (finite) absolute values
        fn range<'a>(it: impl Iterator<Item = &'a f64>) -> (f64, f64) {
            let mut vmin = INF;
            let mut vmax = 0.0;
            for &x in it {
                if x != 0.0 && x.is_finite() {
                    vmin = cmin(vmin, x.abs());
                    vmax = cmax(vmax, x.abs());
                }
            }
            if vmin == INF {
                vmin = 0.0;
            }
            (vmin, vmax)
        }
        let line = |name: &str, (lo, hi): (f64, f64)| {
            control.log(&format!(
                "{}[{}, {}]\n",
                textline(name),
                sci(lo, 5, 0),
                sci(hi, 5, 0)
            ));
        };
        let nz = self.a.entries() as usize;
        line("Matrix range:", range(self.a.values[..nz].iter()));
        line("RHS range:", range(self.scaled_rhs.iter()));
        line("Objective range:", range(self.scaled_obj.iter()));
        line(
            "Bounds range:",
            range(self.scaled_lbuser.iter().chain(self.scaled_ubuser.iter())),
        );
    }

    fn print_preprocessing_log(&self, control: &Control) {
        // Find the minimum and maximum scaling factor.
        let mut minscale = INF;
        let mut maxscale = 0.0;
        for s in [&self.colscale, &self.rowscale] {
            if !s.is_empty() {
                // std::minmax_element: first smallest, last largest
                let (mut lo, mut hi) = (s[0], s[0]);
                for &x in &s[1..] {
                    if x < lo {
                        lo = x;
                    }
                    if !(x < hi) {
                        hi = x;
                    }
                }
                minscale = cmin(minscale, lo);
                maxscale = cmax(maxscale, hi);
            }
        }
        if minscale == INF {
            minscale = 1.0;
        }
        if maxscale == 0.0 {
            maxscale = 1.0;
        }

        control.log(&format!(
            "Preprocessing\n{}{}\n{}{}\n",
            textline("Dualized model:"),
            if self.dualized { "yes" } else { "no" },
            textline("Number of dense columns:"),
            self.num_dense_cols
        ));
        if control.scale() > 0 {
            control.log(&format!(
                "{}[{}, {}]\n",
                textline("Range of scaling factors:"),
                sci(minscale, 8, 2),
                sci(maxscale, 8, 2)
            ));
        }
        control.log(&format!(
            "{}{}\n{}{}\n",
            textline("Scaled cost norm:   "),
            g(self.norm_c),
            textline("Scaled bounds norm: "),
            g(self.norm_bounds)
        ));
    }

    /// Applies the operations from ScaleModel() to a primal-dual point.
    fn scale_point(&self, x: &mut [f64], slack: &mut [f64], y: &mut [f64], z: &mut [f64]) {
        if !self.colscale.is_empty() {
            for j in 0..x.len() {
                x[j] /= self.colscale[j];
                z[j] *= self.colscale[j];
            }
        }
        if !self.rowscale.is_empty() {
            for i in 0..y.len() {
                y[i] /= self.rowscale[i];
                slack[i] *= self.rowscale[i];
            }
        }
        for &j in &self.flipped_vars {
            x[j] *= -1.0;
            z[j] *= -1.0;
        }
    }

    fn scale_point7(
        &self,
        x: &mut [f64],
        xl: &mut [f64],
        xu: &mut [f64],
        slack: &mut [f64],
        y: &mut [f64],
        zl: &mut [f64],
        zu: &mut [f64],
    ) {
        if !self.colscale.is_empty() {
            for j in 0..x.len() {
                let s = self.colscale[j];
                x[j] /= s;
                xl[j] /= s;
                xu[j] /= s;
                zl[j] *= s;
                zu[j] *= s;
            }
        }
        if !self.rowscale.is_empty() {
            for i in 0..y.len() {
                y[i] /= self.rowscale[i];
                slack[i] *= self.rowscale[i];
            }
        }
        for &j in &self.flipped_vars {
            x[j] *= -1.0;
            xl[j] = xu[j];
            xu[j] = INF;
            zl[j] = zu[j];
            zu[j] = 0.0;
        }
    }

    /// u = [x, xl, xu, slack, y, zl, zu]
    fn scale_back_interior_solution(&self, u: &mut [Vec<f64>; 7]) {
        let [x, xl, xu, slack, y, zl, zu] = u;
        if !self.colscale.is_empty() {
            for j in 0..x.len() {
                let s = self.colscale[j];
                x[j] *= s;
                xl[j] *= s;
                xu[j] *= s;
                zl[j] /= s;
                zu[j] /= s;
            }
        }
        if !self.rowscale.is_empty() {
            for i in 0..y.len() {
                y[i] *= self.rowscale[i];
                slack[i] /= self.rowscale[i];
            }
        }
        for &j in &self.flipped_vars {
            x[j] *= -1.0;
            xu[j] = xl[j];
            xl[j] = INF;
            zu[j] = zl[j];
            zl[j] = 0.0;
        }
    }

    fn scale_back_residuals(&self, rb: &mut [f64], rc: &mut [f64], rl: &mut [f64], ru: &mut [f64]) {
        if !self.colscale.is_empty() {
            for j in 0..rc.len() {
                rc[j] /= self.colscale[j];
                rl[j] *= self.colscale[j];
                ru[j] *= self.colscale[j];
            }
        }
        if !self.rowscale.is_empty() {
            for i in 0..rb.len() {
                rb[i] /= self.rowscale[i];
            }
        }
        for &j in &self.flipped_vars {
            rc[j] *= -1.0;
            ru[j] = -rl[j];
            rl[j] = 0.0;
        }
    }

    fn scale_back_basic_solution(&self, x: &mut [f64], slack: &mut [f64], y: &mut [f64], z: &mut [f64]) {
        if !self.colscale.is_empty() {
            for j in 0..x.len() {
                x[j] *= self.colscale[j];
                z[j] /= self.colscale[j];
            }
        }
        if !self.rowscale.is_empty() {
            for i in 0..y.len() {
                y[i] *= self.rowscale[i];
                slack[i] /= self.rowscale[i];
            }
        }
        for &j in &self.flipped_vars {
            x[j] *= -1.0;
            z[j] *= -1.0;
        }
    }

    fn scale_back_basis(&self, _cbasis: &mut [Int], vbasis: &mut [Int]) {
        for &j in &self.flipped_vars {
            if vbasis[j] == NONBASIC_LB {
                vbasis[j] = NONBASIC_UB;
            }
        }
    }

    /// Applies the operations of LoadPrimal() or LoadDual() to a primal-dual
    /// point.
    fn dualize_basic_solution(
        &self,
        x_user: &[f64],
        slack_user: &[f64],
        y_user: &[f64],
        z_user: &[f64],
        x_solver: &mut [f64],
        y_solver: &mut [f64],
        z_solver: &mut [f64],
    ) {
        let m = self.num_rows;
        let n = self.num_cols;
        let nc = self.num_constr;
        if self.dualized {
            // Build dual solver variables from primal user variables.
            for i in 0..m {
                y_solver[i] = -x_user[i];
            }
            for i in 0..nc {
                z_solver[i] = -slack_user[i];
            }
            for (k, &j) in self.boxed_vars.iter().enumerate() {
                z_solver[nc + k] = self.c[nc + k] + y_solver[j];
            }
            for i in 0..m {
                z_solver[n + i] = self.c[n + i] - y_solver[i];
            }
            // Build primal solver variables from dual user variables.
            x_solver[..nc].copy_from_slice(&y_user[..nc]);
            x_solver[n..n + self.num_var].copy_from_slice(&z_user[..self.num_var]);
            for (k, &j) in self.boxed_vars.iter().enumerate() {
                if x_solver[n + j] < 0.0 {
                    // j is a boxed variable and z_user[j] < 0
                    x_solver[nc + k] = -x_solver[n + j];
                    x_solver[n + j] = 0.0;
                } else {
                    x_solver[nc + k] = 0.0;
                }
            }
        } else {
            x_solver[..n].copy_from_slice(&x_user[..n]);
            x_solver[n..n + m].copy_from_slice(&slack_user[..m]);
            y_solver[..m].copy_from_slice(&y_user[..m]);
            z_solver[..n].copy_from_slice(&z_user[..n]);
            for i in 0..m {
                z_solver[n + i] = self.c[n + i] - y_solver[i];
            }
        }
    }

    /// Recovers the solution to the scaled user model u = [x, xl, xu,
    /// slack, y, zl, zu] from s = [x, xl, xu, y, zl, zu] of the solver.
    fn dualize_back_interior_solution(&self, s: [&[f64]; 6], u: &mut [Vec<f64>; 7]) {
        let [x_solver, xl_solver, xu_solver, y_solver, zl_solver, zu_solver] = s;
        let [x_user, xl_user, xu_user, slack_user, y_user, zl_user, zu_user] = u;
        let m = self.num_rows;
        let n = self.num_cols;
        let nc = self.num_constr;
        let nv = self.num_var;

        if self.dualized {
            for i in 0..m {
                x_user[i] = -y_solver[i];
            }
            // To satisfy the sign condition on y_user even if the solution is
            // not exact, use the xl_solver and xu_solver entries for
            // inequality constraints.
            for i in 0..nc {
                y_user[i] = match self.constr_type[i] {
                    b'=' => x_solver[i],
                    b'<' => -xu_solver[i],
                    _ => xl_solver[i],
                };
            }
            // Dual variables associated with lbuser <= x are the slack
            // variables from the solver; using xl_solver guarantees zl_user
            // >= 0. If variable j is free, the j-th slack variable was fixed
            // at zero, which the IPM solution may not satisfy; hence set
            // zl_user[j] = 0 explicitly.
            zl_user[..nv].copy_from_slice(&xl_solver[n..n + nv]);
            for j in 0..nv {
                if !self.scaled_lbuser[j].is_finite() {
                    zl_user[j] = 0.0;
                }
            }
            // Dual variables associated with x <= ubuser are the primal
            // variables that were added for boxed variables.
            zu_user.fill(0.0);
            let mut k = nc;
            for &j in &self.boxed_vars {
                zu_user[j] = xl_solver[k];
                k += 1;
            }
            // xl in the scaled user model is zl[n+1:n+m] or infinity.
            for i in 0..m {
                xl_user[i] = if self.scaled_lbuser[i].is_finite() {
                    zl_solver[n + i]
                } else {
                    INF
                };
            }
            // xu in the scaled user model are the entries in zl for columns
            // of the negative identity matrix (added for boxed variables).
            xu_user.fill(INF);
            k = nc;
            for &j in &self.boxed_vars {
                xu_user[j] = zl_solver[k];
                k += 1;
            }
            for i in 0..nc {
                slack_user[i] = match self.constr_type[i] {
                    b'=' => 0.0,
                    b'<' => zu_solver[i],
                    _ => -zl_solver[i],
                };
            }
        } else {
            x_user[..nv].copy_from_slice(&x_solver[..nv]);
            // Instead of copying y_solver into y_user, use the entries from
            // zl_solver and zu_solver for inequality constraints, so that the
            // sign condition on y_user is satisfied.
            for i in 0..m {
                y_user[i] = match self.constr_type[i] {
                    b'=' => y_solver[i],
                    b'<' => -zl_solver[n + i],
                    _ => zu_solver[n + i],
                };
            }
            zl_user[..nv].copy_from_slice(&zl_solver[..nv]);
            zu_user[..nv].copy_from_slice(&zu_solver[..nv]);
            xl_user[..nv].copy_from_slice(&xl_solver[..nv]);
            xu_user[..nv].copy_from_slice(&xu_solver[..nv]);
            // Build the slack for inequality constraints from xl_solver and
            // xu_solver (sign condition) and set it to zero for equality
            // constraints.
            for i in 0..m {
                slack_user[i] = match self.constr_type[i] {
                    b'=' => 0.0,
                    b'<' => xl_solver[n + i],
                    _ => -xu_solver[n + i],
                };
            }
        }
    }

    fn dualize_back_basic_solution(
        &self,
        x_solver: &[f64],
        y_solver: &[f64],
        z_solver: &[f64],
        x_user: &mut [f64],
        slack_user: &mut [f64],
        y_user: &mut [f64],
        z_user: &mut [f64],
    ) {
        let m = self.num_rows;
        let n = self.num_cols;
        let nc = self.num_constr;
        let nv = self.num_var;
        if self.dualized {
            for i in 0..m {
                x_user[i] = -y_solver[i];
            }
            for i in 0..nc {
                slack_user[i] = -z_solver[i];
            }
            y_user[..nc].copy_from_slice(&x_solver[..nc]);
            z_user[..nv].copy_from_slice(&x_solver[n..n + nv]);
            let mut k = nc;
            for &j in &self.boxed_vars {
                z_user[j] -= x_solver[k];
                k += 1;
            }
        } else {
            x_user[..nv].copy_from_slice(&x_solver[..nv]);
            slack_user[..nc].copy_from_slice(&x_solver[n..n + nc]);
            y_user[..nc].copy_from_slice(&y_solver[..nc]);
            z_user[..nv].copy_from_slice(&z_solver[..nv]);
        }
    }

    fn dualize_back_basis(&self, basic_status_solver: &[Int], cbasis_user: &mut [Int], vbasis_user: &mut [Int]) {
        let n = self.num_cols;
        let nc = self.num_constr;
        let nv = self.num_var;
        if self.dualized {
            for i in 0..nc {
                cbasis_user[i] = if basic_status_solver[i] == BASIC { NONBASIC } else { BASIC };
            }
            for j in 0..nv {
                vbasis_user[j] = if basic_status_solver[n + j] == 0 {
                    if self.scaled_lbuser[j].is_finite() {
                        NONBASIC_LB
                    } else {
                        SUPERBASIC
                    }
                } else {
                    BASIC
                };
            }
            let mut k = nc;
            for &j in &self.boxed_vars {
                if basic_status_solver[k] == BASIC {
                    vbasis_user[j] = NONBASIC_UB;
                }
                k += 1;
            }
        } else {
            for i in 0..nc {
                cbasis_user[i] = if basic_status_solver[n + i] == BASIC { BASIC } else { NONBASIC };
            }
            vbasis_user[..nv].copy_from_slice(&basic_status_solver[..nv]);
        }
    }

    fn correct_scaled_basic_solution(
        &self,
        x: &mut [f64],
        slack: &mut [f64],
        y: &mut [f64],
        z: &mut [f64],
        cbasis: &[Int],
        vbasis: &[Int],
    ) {
        for j in 0..self.num_var {
            if vbasis[j] == NONBASIC_LB {
                x[j] = self.scaled_lbuser[j];
            }
            if vbasis[j] == NONBASIC_UB {
                x[j] = self.scaled_ubuser[j];
            }
            if vbasis[j] == BASIC {
                z[j] = 0.0;
            }
        }
        for i in 0..self.num_constr {
            if cbasis[i] == NONBASIC {
                slack[i] = 0.0;
            }
            if cbasis[i] == BASIC {
                y[i] = 0.0;
            }
        }
    }

    /// lhs += alpha*A*rhs or lhs += alpha*A'rhs ('t'/'T'), A the scaled user
    /// matrix (used implicitly through AI).
    fn multiply_with_scaled_matrix(&self, rhs: &[f64], alpha: f64, lhs: &mut [f64], trans: u8) {
        let ai = &self.ai;
        if trans == b't' || trans == b'T' {
            if self.dualized {
                for i in 0..self.num_constr {
                    scatter_column(ai, i, alpha * rhs[i], lhs);
                }
            } else {
                for j in 0..self.num_var {
                    lhs[j] = alpha.mul_add(dot_column(ai, j, rhs), lhs[j]);
                }
            }
        } else if self.dualized {
            for i in 0..self.num_constr {
                lhs[i] = alpha.mul_add(dot_column(ai, i, rhs), lhs[i]);
            }
        } else {
            for j in 0..self.num_var {
                scatter_column(ai, j, alpha * rhs[j], lhs);
            }
        }
    }
}

/// Maximum violation of lb <= x <= ub.
pub fn primal_infeasibility(model: &Model, x: &[f64]) -> f64 {
    let (lb, ub) = (model.lb(), model.ub());
    let mut infeas = 0.0;
    for j in 0..x.len() {
        infeas = cmax(infeas, lb[j] - x[j]);
        infeas = cmax(infeas, x[j] - ub[j]);
    }
    infeas
}

/// Maximum violation of z[j] <= 0 if x[j] > lb[j], z[j] >= 0 if x[j] <
/// ub[j].
pub fn dual_infeasibility(model: &Model, x: &[f64], z: &[f64]) -> f64 {
    let (lb, ub) = (model.lb(), model.ub());
    let mut infeas = 0.0;
    for j in 0..x.len() {
        if x[j] > lb[j] {
            infeas = cmax(infeas, z[j]);
        }
        if x[j] < ub[j] {
            infeas = cmax(infeas, -z[j]);
        }
    }
    infeas
}

/// Maximum violation of Ax=b.
pub fn primal_residual(model: &Model, x: &[f64]) -> f64 {
    let ait = model.ait();
    let b = model.b();
    let mut res = 0.0;
    for i in 0..b.len() {
        let r = b[i] - dot_column(ait, i, x);
        res = cmax(res, r.abs());
    }
    res
}

/// Maximum violation of A'y+z=c.
pub fn dual_residual(model: &Model, y: &[f64], z: &[f64]) -> f64 {
    let ai = model.ai();
    let c = model.c();
    let mut res = 0.0;
    for j in 0..c.len() {
        let r = c[j] - z[j] - dot_column(ai, j, y);
        res = cmax(res, r.abs());
    }
    res
}
