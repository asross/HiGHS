use super::vector::{MatrixBase, QpVector};
use super::*;

struct NoCallbacks(Option<Phase1Start>);

impl Callbacks for NoCallbacks {
    fn time(&mut self) -> f64 {
        0.0
    }
    fn iteration_log(&mut self, _: i32, _: f64, _: i32, _: f64) {}
    fn nullspace_limit_log(&mut self, _: i32) {}
    fn degeneracy_fail_log(&mut self, _: i32, _: f64) {}
    fn phase1(&mut self) -> Phase1Start {
        self.0.take().expect("one phase 1")
    }
}

fn settings() -> Settings {
    Settings {
        ratiotest: 0,
        pricing: 2,
        reportingfequency: 100,
        nullspace_limit: 4000,
        reinvertfrequency: 1000,
        gradientrecomputefrequency: 100,
        iteration_limit: 1000,
        ratiotest_t: 1e-9,
        ratiotest_d: 1e-8,
        pnorm_zero_threshold: 1e-11,
        d_zero_threshold: 1e-12,
        lambda_zero_threshold: 1e-9,
        pqp_zero_threshold: 1e-7,
        hessian_regularization_value: 1e-7,
        time_limit: f64::INFINITY,
    }
}

fn csc(num_row: usize, cols: &[&[(usize, f64)]]) -> MatrixBase {
    let mut m = MatrixBase { num_row, num_col: cols.len(), start: vec![0], ..Default::default() };
    for col in cols {
        for &(i, v) in *col {
            m.index.push(i);
            m.value.push(v);
        }
        m.start.push(m.index.len());
    }
    m
}

/// min (x-1)^2 + (y-2)^2 with x <= 0.5: the bounded start, then QUASS
#[test]
fn bounded_qp() {
    let inst = Instance {
        num_var: 2,
        num_con: 0,
        offset: 5.0,
        c: QpVector::from_dense(&[-2.0, -4.0]),
        q: csc(2, &[&[(0, 2.0)], &[(1, 2.0)]]),
        a: csc(0, &[&[], &[]]),
        con_lo: vec![],
        con_up: vec![],
        var_lo: vec![0.0, 0.0],
        var_up: vec![0.5, 10.0],
    };
    let out = solve(inst, &settings(), &mut NoCallbacks(None));
    assert_eq!(out.status, ModelStatus::Optimal);
    assert!((out.primal[0] - 0.5).abs() < 1e-6 && (out.primal[1] - 2.0).abs() < 1e-6);
    assert_eq!(out.status_var[0], BasisStatus::ActiveAtUpper);
}

/// min x^2 + y^2 s.t. x + y >= 1, from the vertex (1, 0)
fn constrained() -> (Instance, Phase1Start) {
    let inst = Instance {
        num_var: 2,
        num_con: 1,
        offset: 0.0,
        c: QpVector::from_dense(&[0.0, 0.0]),
        q: csc(2, &[&[(0, 2.0)], &[(1, 2.0)]]),
        a: csc(1, &[&[(0, 1.0)], &[(0, 1.0)]]),
        con_lo: vec![1.0],
        con_up: vec![f64::INFINITY],
        var_lo: vec![f64::NEG_INFINITY; 2],
        var_up: vec![f64::INFINITY; 2],
    };
    let start = Phase1Start {
        status: ModelStatus::NotSet,
        active: vec![0],
        status_active: vec![BasisStatus::ActiveAtLower],
        inactive: vec![2],
        primal: vec![1.0, 0.0],
        rowact: vec![1.0],
    };
    (inst, start)
}

#[test]
fn constrained_qp() {
    for pricing in 0..3 {
        let (inst, start) = constrained();
        let s = Settings { pricing, ..settings() };
        let out = solve(inst, &s, &mut NoCallbacks(Some(start)));
        assert_eq!(out.status, ModelStatus::Optimal);
        assert!((out.primal[0] - 0.5).abs() < 1e-6 && (out.primal[1] - 0.5).abs() < 1e-6, "{:?}", out.primal);
        assert!((out.dualcon[0] - 1.0).abs() < 1e-6, "{:?}", out.dualcon);
    }
}
