use super::*;

#[test]
fn printf_g() {
    assert_eq!(g(1.4655, 3), "1.47");
    assert_eq!(g(314.0, 3), "314");
    assert_eq!(g(123456.0, 3), "1.23e+05");
    assert_eq!(g(0.0001234, 6), "0.0001234");
    assert_eq!(g(0.0, 4), "0");
    assert_eq!(plus(e(-0.5, 2)), "-5.00e-01");
    assert_eq!(plus(e(2.0, 8)), "+2.00000000e+00");
}

/// max x + y s.t. x + 2y <= 4 (row 0), 3x + y in [1, 6] (row 1),
/// x - y = 0 (row 2), x, y >= 0: optimum x = y = 4/3
#[test]
fn small_lp() {
    let lp = Lp {
        start: &[0, 3, 6],
        index: &[0, 1, 2, 0, 1, 2],
        value: &[1.0, 3.0, 1.0, 2.0, 1.0, -1.0],
        col_cost: &[1.0, 1.0],
        col_lower: &[0.0, 0.0],
        col_upper: &[f64::INFINITY, f64::INFINITY],
        row_lower: &[-f64::INFINITY, 1.0, 0.0],
        row_upper: &[4.0, 6.0, 0.0],
        offset: 0.0,
        sense: -1.0,
    };
    let params = Params {
        primal_tol: 1e-7,
        dual_tol: 1e-7,
        gap_tol: 1e-7,
        time_lim: f64::INFINITY,
        iter_lim: 100000,
        log_level: 0,
        scaling: 1,
        line_search: 2,
        restart: 1,
    };
    let (mut cv, mut cd) = (vec![0.0; 2], vec![0.0; 2]);
    let (mut rv, mut rd) = (vec![0.0; 3], vec![0.0; 3]);
    let mut sol = Solution {
        col_value: &mut cv,
        col_dual: &mut cd,
        row_value: &mut rv,
        row_dual: &mut rd,
        value_valid: false,
        dual_valid: false,
    };
    let log = Log {
        level: 0,
        print: None,
    };
    let (code, iters) = solve(&lp, &params, &log, &mut sol);
    assert_eq!(code, TermCode::Optimal);
    assert!(iters > 0);
    assert!((cv[0] - 4.0 / 3.0).abs() < 1e-4 && (cv[1] - 4.0 / 3.0).abs() < 1e-4);
    assert!((rv[0] - 4.0).abs() < 1e-4);
}
