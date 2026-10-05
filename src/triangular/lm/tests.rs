//! The operators against each other and against finite differences of the
//! residuals, on a small map with every transformer layer.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

use super::conditioner::Conditioner;
use super::FisherResiduals;
use crate::triangular::layers::{
    Contract2Spec, LayerSpec, Param, PositiveAffineSpec, TangentSasSpec,
};
use crate::triangular::pattern::{Pattern, Ragged};

const N_DRAW: usize = 16;

pub(super) fn normal(rng: &mut ChaCha8Rng, n: usize, scale: f64) -> Vec<f64> {
    (0..n)
        .map(|_| {
            let z: f64 = StandardNormal.sample(&mut *rng);
            scale * z
        })
        .collect()
}

/// Parents closed under elimination, with rows out of order so that edge
/// order is not parent order.
fn graph() -> Pattern {
    let rows: [&[usize]; 6] = [&[], &[0], &[1, 0], &[2, 1], &[3, 1], &[0]];
    Pattern::new(Ragged::from_rows(rows.iter().map(|r| r.iter().copied()))).unwrap()
}

fn param(index: usize, offset: f64) -> Option<Param> {
    Some(Param { index, offset })
}

/// `Contract2` (bounded), `TangentSAS` and `PositiveAffine`, eleven
/// parameters, the location in `Contract2`'s `mu`.
fn problem(regularization: Option<f64>) -> FisherResiduals {
    problem_with(regularization, N_DRAW)
}

pub(super) fn problem_with(regularization: Option<f64>, n_draw: usize) -> FisherResiduals {
    let specs = vec![
        LayerSpec::Contract2(Contract2Spec {
            alpha: param(0, 0.1),
            beta: param(1, 0.0),
            sigma: param(2, -0.2),
            mu: param(3, 0.0),
            nu: param(4, 0.3),
            log_gamma_bounds: Some((-1.5, 2.0)),
        }),
        LayerSpec::TangentSas(TangentSasSpec {
            nu: param(5, 0.0),
            eps: param(6, 0.1),
            b: param(7, 0.2),
            r: param(8, -0.1),
        }),
        LayerSpec::PositiveAffine(PositiveAffineSpec {
            loc: param(9, 0.0),
            scale: param(10, 0.1),
        }),
    ];
    let conditioner = Conditioner::new(3, 11, 3, specs).unwrap();
    let mut problem = FisherResiduals::new(graph(), conditioner, regularization).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let n_var = problem.n_var();
    let y = normal(&mut rng, n_draw * n_var, 1.0);
    let g: Vec<f64> = y
        .iter()
        .zip(normal(&mut rng, n_draw * n_var, 0.3))
        .map(|(y, noise)| -y + noise)
        .collect();
    problem.set_data(y, g).unwrap();
    problem
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

fn assert_close(actual: &[f64], expected: &[f64], tol: f64) {
    let scale = expected.iter().fold(1.0f64, |m, v| m.max(v.abs()));
    for (k, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!((a - e).abs() <= tol * scale, "entry {k}: {a} != {e}");
    }
}

const REGULARIZATIONS: [Option<f64>; 2] = [None, Some(0.3)];

#[test]
fn pushforward_matches_finite_differences() {
    for rho in REGULARIZATIONS {
        let problem = problem(rho);
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let theta = normal(&mut rng, problem.n_params(), 0.3);
        let v = normal(&mut rng, problem.n_params(), 1.0);
        let (_, lin) = problem.residuals(&theta).unwrap();
        let jv = problem.pushforward(&lin, &v).unwrap();

        let h = 1e-5;
        let shifted = |sign: f64| {
            let theta: Vec<f64> = theta
                .iter()
                .zip(&v)
                .map(|(t, v)| t + sign * h * v)
                .collect();
            problem.residuals(&theta).unwrap().0
        };
        let (plus, minus) = (shifted(1.0), shifted(-1.0));
        let fd: Vec<f64> = plus
            .iter()
            .zip(&minus)
            .map(|(p, m)| (p - m) / (2.0 * h))
            .collect();
        assert_close(&jv, &fd, 1e-6);
    }
}

#[test]
fn pullback_is_the_adjoint_and_products_compose() {
    for rho in REGULARIZATIONS {
        let problem = problem(rho);
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let theta = normal(&mut rng, problem.n_params(), 0.3);
        let v = normal(&mut rng, problem.n_params(), 1.0);
        let u = normal(&mut rng, N_DRAW * problem.n_residuals(), 1.0);
        let (_, lin) = problem.residuals(&theta).unwrap();

        let jv = problem.pushforward(&lin, &v).unwrap();
        let jtu = problem.pullback(&lin, &u).unwrap();
        let (lhs, rhs) = (dot(&jv, &u), dot(&v, &jtu));
        assert!(
            (lhs - rhs).abs() <= 1e-10 * lhs.abs().max(1.0),
            "{lhs} != {rhs}"
        );

        let jtjv = problem.gauss_newton_product(&lin, &v).unwrap();
        let composed = problem.pullback(&lin, &jv).unwrap();
        assert_close(&jtjv, &composed, 1e-12);
    }
}

#[test]
fn blocks_match_the_dense_gram() {
    for rho in REGULARIZATIONS {
        let problem = problem(rho);
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let theta = normal(&mut rng, problem.n_params(), 0.3);
        let (_, lin) = problem.residuals(&theta).unwrap();
        let column = |k: usize| {
            let mut e = vec![0.0; problem.n_params()];
            e[k] = 1.0;
            problem.pushforward(&lin, &e).unwrap()
        };
        let columns: Vec<Vec<f64>> = (0..problem.n_params()).map(column).collect();

        for max_block_size in [5, 1000] {
            let blocks = problem.gauss_newton_blocks(&lin, max_block_size).unwrap();
            let covered: usize = blocks.iter().map(|b| b.matrix.nrows()).sum();
            assert_eq!(covered, problem.n_params());
            for block in &blocks {
                let n = block.matrix.nrows();
                let actual: Vec<f64> = (0..n * n)
                    .map(|rc| block.matrix[(rc / n, rc % n)])
                    .collect();
                let expected: Vec<f64> = (0..n * n)
                    .map(|rc| {
                        let (r, c) = (block.start + rc / n, block.start + rc % n);
                        dot(&columns[r], &columns[c])
                    })
                    .collect();
                assert_close(&actual, &expected, 1e-10);
            }
        }
    }
}

#[test]
fn reductions_do_not_depend_on_the_threads() {
    // Many tiles, so that the work is split between threads.
    let problem = problem_with(Some(0.3), 512);
    let mut rng = ChaCha8Rng::seed_from_u64(4);
    let theta = normal(&mut rng, problem.n_params(), 0.3);
    let v = normal(&mut rng, problem.n_params(), 1.0);
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let (r, lin) = problem.residuals(&theta).unwrap();
            let grad = problem.pullback(&lin, &r).unwrap();
            let gn = problem.gauss_newton_product(&lin, &v).unwrap();
            (grad, gn)
        })
    };
    let reference = run(1);
    for threads in [2, 3, 8] {
        for _ in 0..5 {
            assert!(run(threads) == reference, "{threads} threads");
        }
    }
}
