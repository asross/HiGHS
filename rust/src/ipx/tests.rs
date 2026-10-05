//! The LP of check/TestIpx.cpp, solved end-to-end (IPM and crossover).

use super::model::UserLp;
use super::{LpSolver, Parameters, STATUS_OPTIMAL, STATUS_SOLVED};

const INF: f64 = f64::INFINITY;

#[test]
fn test_ipx() {
    let obj = [-0.2194, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.32, -0.5564, 0.6, -0.48];
    let lb = [0.0; 12];
    let ub = [80.0, 283.303, 283.303, 312.813, 349.187, INF, INF, INF, 57.201, 500.0, 500.501, 357.501];
    let ap = [0, 2, 6, 10, 14, 18, 20, 22, 24, 26, 28, 30, 32];
    let ai = [0, 5, 1, 6, 7, 8, 2, 6, 7, 8, 3, 6, 7, 8, 4, 6, 7, 8, 1, 2, 2, 3, 2, 4, 0, 6, 0, 5, 2, 5, 5, 7];
    let ax = [
        -1.0, 0.301, 1.0, -1.0, 0.301, 1.06, 1.0, -1.0, 0.313, 1.06, 1.0, -1.0, 0.313, 0.96, 1.0, -1.0, 0.326,
        0.86, -1.0, 0.99078, 1.00922, -1.0, 1.01802, -1.0, 1.4, 1.0, 0.109, -1.0, -0.419111, 1.0, 1.4, -1.0,
    ];
    let rhs = [0.0, 80.0, 0.0, 0.0, 0.0, 0.0, 0.0, 44.0, 300.0];
    let constr_type = *b"<<=<<=<<<";

    let mut lps = LpSolver::new();
    lps.set_parameters(Parameters {
        display: 0,
        ..Default::default()
    });
    let lp = UserLp {
        num_constr: 9,
        num_var: 12,
        ap: Some(&ap),
        ai: Some(&ai),
        ax: Some(&ax),
        rhs: Some(&rhs),
        constr_type: Some(&constr_type),
        offset: 0.0,
        obj: Some(&obj),
        lbuser: Some(&lb),
        ubuser: Some(&ub),
    };
    assert_eq!(lps.load_model(&lp), 0);
    assert_eq!(lps.solve(), STATUS_SOLVED);
    let info = lps.get_info();
    assert_eq!(info.status_ipm, STATUS_OPTIMAL);
    assert_eq!(info.status_crossover, STATUS_OPTIMAL);

    let mut x = [0.0; 12];
    let mut xl = [0.0; 12];
    let mut xu = [0.0; 12];
    let mut slack = [0.0; 9];
    assert_eq!(
        lps.get_interior_solution([Some(&mut x), Some(&mut xl), Some(&mut xu), Some(&mut slack), None, None, None]),
        0
    );
    assert!((x[11] - 339.9).abs() < 1.0);
    assert!((xl[11] - 339.94).abs() < 1.0);
    assert!((xu[11] - 17.55).abs() < 1.0);
    assert!((slack[8] - 234.76).abs() < 1.0);

    let mut xb = [0.0; 12];
    let mut vbasis = [0; 12];
    assert_eq!(lps.get_basic_solution(Some(&mut xb), None, None, None, None, Some(&mut vbasis)), 0);
    assert!((xb[11] - 339.9).abs() < 1.0);
}
