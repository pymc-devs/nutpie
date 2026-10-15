//! One harness, generic over [`Transformer`], run for every kind: the
//! identity at zero, the two directions against each other, and every
//! derivative against finite differences and against the other modes.

use fearless_simd::{dispatch, prelude::*, Level};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

use super::{Transformer, TransformerConfig, TransformerDerivatives};
use crate::triangular::transformer::jet::{packed_len, JetArena};

/// Every kind, with and without its options.
fn kinds() -> Vec<TransformerConfig> {
    use TransformerConfig::*;
    vec![
        Affine {
            scale_squash: Some(1.0),
        },
        Affine { scale_squash: None },
        Affine {
            scale_squash: Some(3.0),
        },
        Sas {
            layers: 1,
            log_gamma_bounds: None,
            scale_squash: Some(1.0),
        },
        Sas {
            layers: 2,
            log_gamma_bounds: Some((-1.5, 2.0)),
            scale_squash: Some(1.0),
        },
        Sas {
            layers: 3,
            log_gamma_bounds: None,
            scale_squash: Some(1.0),
        },
        TangentSas {
            layers: 1,
            fix_b: false,
            scale_squash: Some(1.0),
        },
        TangentSas {
            layers: 2,
            fix_b: true,
            scale_squash: Some(1.0),
        },
        AsymmetricSas {
            layers: 1,
            log_gamma_bounds: None,
            scale_squash: Some(1.0),
        },
        AsymmetricSas {
            layers: 3,
            log_gamma_bounds: Some((-1.0, 1.0)),
            scale_squash: Some(1.0),
        },
        Sas {
            layers: 2,
            log_gamma_bounds: None,
            scale_squash: None,
        },
        TangentSas {
            layers: 1,
            fix_b: false,
            scale_squash: None,
        },
        AsymmetricSas {
            layers: 2,
            log_gamma_bounds: None,
            scale_squash: None,
        },
        Sas {
            layers: 1,
            log_gamma_bounds: None,
            scale_squash: Some(2.5),
        },
    ]
}

/// Base values and data values the checks run at. None is zero, where
/// `asymmetric_sas` switches sides and its second derivatives jump.
const POINTS: [f64; 5] = [-2.1, -0.4, 0.3, 0.7, 1.9];
/// Central differences: step, and tolerance relative to the largest entry.
const STEP: f64 = 1e-5;
const TOL: f64 = 1e-6;

fn normal(rng: &mut ChaCha8Rng, n: usize, scale: f64) -> Vec<f64> {
    (0..n)
        .map(|_| {
            let z: f64 = StandardNormal.sample(&mut *rng);
            scale * z
        })
        .collect()
}

fn assert_close(what: &str, actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len(), "{what}: lengths differ");
    let scale = expected.iter().fold(1.0f64, |m, v| m.max(v.abs()));
    for (k, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol * scale,
            "{what}, entry {k}: {a} != {e} (tolerance {})",
            tol * scale
        );
    }
}

/// `d f / d input k` by central differences, for every `k`.
fn finite_differences(n: usize, f: impl Fn(usize, f64) -> f64) -> Vec<f64> {
    (0..n)
        .map(|k| (f(k, STEP) - f(k, -STEP)) / (2.0 * STEP))
        .collect()
}

fn shifted(values: &[f64], k: usize, h: f64) -> Vec<f64> {
    let mut out = values.to_vec();
    out[k] += h;
    out
}

/// A packed lower triangle as a dense row-major matrix.
fn unpack(packed: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    for r in 0..n {
        for c in 0..=r {
            let v = packed[packed_len(r) + c];
            out[r * n + c] = v;
            out[c * n + r] = v;
        }
    }
    out
}

/// [`Transformer::derivatives_tile`] in lane 0, on `f64`.
struct Derivatives {
    x: f64,
    log_det: f64,
    grad_x: Vec<f64>,
    grad_log_det: Vec<f64>,
    /// Dense, row-major.
    hess_x: Vec<f64>,
    hess_log_det: Vec<f64>,
}

fn derivatives<T: Transformer>(t: &T, coefs: &[f64], y: f64) -> Derivatives {
    fn on<T: Transformer, S: Simd>(t: &T, simd: S, coefs: &[f64], y: f64) -> Derivatives {
        let splat = |v| S::f64s::splat(simd, v);
        let coefs: Vec<_> = coefs.iter().map(|&c| splat(c)).collect();
        let mut out = TransformerDerivatives::new(simd, coefs.len());
        t.derivatives_tile(simd, &mut JetArena::new(simd), &coefs, splat(y), &mut out);
        let lane = |v: &[S::f64s]| v.iter().map(|v| v[0]).collect::<Vec<_>>();
        let n_z = 1 + coefs.len();
        Derivatives {
            x: out.x[0],
            log_det: out.log_det[0],
            grad_x: lane(&out.grad_x),
            grad_log_det: lane(&out.grad_log_det),
            hess_x: unpack(&lane(&out.hess_x), n_z),
            hess_log_det: unpack(&lane(&out.hess_log_det), n_z),
        }
    }
    dispatch!(Level::new(), simd => on(t, simd, coefs, y))
}

/// [`Transformer::inverse_jet_tile`] along `dirs` in lane 0: the gradients,
/// and per direction the Hessian products, of `x` and of `log_det`.
#[allow(clippy::type_complexity)]
fn along<T: Transformer>(
    t: &T,
    coefs: &[f64],
    y: f64,
    dirs: &[f64],
    k: usize,
) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
    #[allow(clippy::type_complexity)]
    fn on<T: Transformer, S: Simd>(
        t: &T,
        simd: S,
        coefs: &[f64],
        y: f64,
        dirs: &[f64],
        k: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let splat = |v| S::f64s::splat(simd, v);
        let coefs: Vec<_> = coefs.iter().map(|&c| splat(c)).collect();
        let dirs: Vec<_> = dirs.iter().map(|&d| splat(d)).collect();
        let mut ws = JetArena::new(simd);
        let (x, log_det) = t.inverse_jet_tile(simd, &mut ws, &coefs, splat(y), &dirs, k);
        let lane = |v: &[S::f64s]| v.iter().map(|v| v[0]).collect::<Vec<_>>();
        (
            lane(ws.grad(&x)),
            lane(ws.grad(&log_det)),
            (0..k).map(|d| lane(ws.hess_dir(&x, d))).collect(),
            (0..k).map(|d| lane(ws.hess_dir(&log_det, d))).collect(),
        )
    }
    dispatch!(Level::new(), simd => on(t, simd, coefs, y, dirs, k))
}

fn mat_vec(m: &[f64], v: &[f64]) -> Vec<f64> {
    m.chunks_exact(v.len())
        .map(|row| row.iter().zip(v).map(|(a, b)| a * b).sum())
        .collect()
}

/// Everything the harness checks for one transformer.
fn check<T: Transformer>(t: &T, name: &str) {
    let n = t.n_coefs();
    let n_z = 1 + n;

    // Roles.
    let location = t.location_coef().expect("every kind has a location");
    let scale = t.scale_coef().expect("every kind has a scale");
    assert!(
        location < n && scale < n && location != scale,
        "{name}: roles"
    );
    for k in t.shape_coefs() {
        assert!(k < n && k != location && k != scale, "{name}: shape {k}");
    }

    // All-zero coefficients are the identity.
    let zeros = vec![0.0; n];
    for &x in POINTS.iter().chain(&[0.0]) {
        let what = format!("{name}: identity at {x}");
        let (y, log_det) = t.forward_draw(&zeros, x);
        assert_close(&what, &[y, log_det], &[x, 0.0], 1e-14);
        let (back, log_det) = t.inverse_draw(&zeros, x);
        assert_close(&what, &[back, log_det], &[x, 0.0], 1e-14);
    }

    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let mut scratch = t.new_scratch();
    for trial in 0..4 {
        let coefs = normal(&mut rng, n, 0.5);
        let dirs = normal(&mut rng, 2 * n_z, 1.0);
        for &x in &POINTS {
            let what = |check: &str| format!("{name}, trial {trial}, x = {x}: {check}");

            // The generative direction and its gradients.
            let (y, log_det) = t.forward_draw(&coefs, x);
            let (mut dy_dcoefs, mut dld_dcoefs) = (vec![0.0; n], vec![0.0; n]);
            let (y2, log_det2, dy_dx, dld_dx) =
                t.forward_draw_with_grads(&mut scratch, &coefs, x, &mut dy_dcoefs, &mut dld_dcoefs);
            assert_close(
                &what("forward values"),
                &[y2, log_det2],
                &[y, log_det],
                1e-14,
            );
            let fd = |f: &dyn Fn(f64, &[f64]) -> f64| {
                let mut out = vec![(f(x + STEP, &coefs) - f(x - STEP, &coefs)) / (2.0 * STEP)];
                out.extend(finite_differences(n, |k, h| f(x, &shifted(&coefs, k, h))));
                out
            };
            let y_of = |x: f64, c: &[f64]| t.forward_draw(c, x).0;
            let ld_of = |x: f64, c: &[f64]| t.forward_draw(c, x).1;
            let mut grad_y = vec![dy_dx];
            grad_y.extend(&dy_dcoefs);
            let mut grad_ld = vec![dld_dx];
            grad_ld.extend(&dld_dcoefs);
            assert_close(&what("forward dy"), &grad_y, &fd(&y_of), TOL);
            assert_close(&what("forward dlog_det"), &grad_ld, &fd(&ld_of), TOL);

            // Moving the location and the scale moves the output.
            let (a, b) = (1.7, -0.4);
            let mut moved = coefs.clone();
            moved[location] = t.shifted_location_coef(coefs[location], a, b);
            moved[scale] = t.rescaled_scale_coef(coefs[scale], a);
            let (y_moved, log_det_moved) = t.forward_draw(&moved, x);
            assert_close(
                &what("location and scale"),
                &[y_moved, log_det_moved],
                &[a * y + b, log_det + a.ln()],
                1e-12,
            );

            // The density direction inverts it.
            let (back, inv_log_det) = t.inverse_draw(&coefs, y);
            assert_close(
                &what("round trip"),
                &[back, inv_log_det],
                &[x, -log_det],
                1e-10,
            );

            // Its derivatives: values, gradients against finite differences
            // and against the generative direction's by the implicit function
            // theorem, Hessians against finite differences of the gradients.
            let d = derivatives(t, &coefs, y);
            assert_close(
                &what("derivatives values"),
                &[d.x, d.log_det],
                &[back, inv_log_det],
                1e-12,
            );
            let at = |y: f64, c: &[f64]| t.inverse_draw(c, y);
            let fd_inv = |f: &dyn Fn((f64, f64)) -> f64| {
                let mut out =
                    vec![(f(at(y + STEP, &coefs)) - f(at(y - STEP, &coefs))) / (2.0 * STEP)];
                out.extend(finite_differences(n, |k, h| {
                    f(at(y, &shifted(&coefs, k, h)))
                }));
                out
            };
            assert_close(&what("inverse dx"), &d.grad_x, &fd_inv(&|v| v.0), TOL);
            assert_close(
                &what("inverse dlog_det"),
                &d.grad_log_det,
                &fd_inv(&|v| v.1),
                TOL,
            );

            let delta = 1.0 / dy_dx;
            let mut implicit_x = vec![delta];
            implicit_x.extend(dy_dcoefs.iter().map(|g| -g * delta));
            let mut implicit_ld = vec![-dld_dx * delta];
            implicit_ld.extend(
                dld_dcoefs
                    .iter()
                    .zip(&implicit_x[1..])
                    .map(|(g, t)| -(g + dld_dx * t)),
            );
            assert_close(&what("implicit dx"), &d.grad_x, &implicit_x, 1e-9);
            assert_close(
                &what("implicit dlog_det"),
                &d.grad_log_det,
                &implicit_ld,
                1e-9,
            );

            let z = {
                let mut z = vec![y];
                z.extend(&coefs);
                z
            };
            let grads_at = |z: &[f64]| {
                let d = derivatives(t, &z[1..], z[0]);
                (d.grad_x, d.grad_log_det)
            };
            for r in 0..n_z {
                let (plus, minus) = (
                    grads_at(&shifted(&z, r, STEP)),
                    grads_at(&shifted(&z, r, -STEP)),
                );
                let column = |p: &[f64], m: &[f64]| -> Vec<f64> {
                    p.iter()
                        .zip(m)
                        .map(|(p, m)| (p - m) / (2.0 * STEP))
                        .collect()
                };
                let row = r * n_z..(r + 1) * n_z;
                assert_close(
                    &what(&format!("hessian of x, row {r}")),
                    &d.hess_x[row.clone()],
                    &column(&plus.0, &minus.0),
                    TOL,
                );
                assert_close(
                    &what(&format!("hessian of log_det, row {r}")),
                    &d.hess_log_det[row],
                    &column(&plus.1, &minus.1),
                    TOL,
                );
            }

            // The directional jets agree with the full Hessians.
            let (grad_x, grad_ld, hd_x, hd_ld) = along(t, &coefs, y, &dirs, 2);
            assert_close(&what("directional dx"), &grad_x, &d.grad_x, 1e-14);
            assert_close(
                &what("directional dlog_det"),
                &grad_ld,
                &d.grad_log_det,
                1e-14,
            );
            for (k, dir) in dirs.chunks_exact(n_z).enumerate() {
                let label = format!("hessian product {k}");
                assert_close(&what(&label), &hd_x[k], &mat_vec(&d.hess_x, dir), 1e-12);
                assert_close(
                    &what(&label),
                    &hd_ld[k],
                    &mat_vec(&d.hess_log_det, dir),
                    1e-12,
                );
            }
        }
    }
}

#[test]
fn every_kind_passes_the_harness() {
    for config in kinds() {
        let chain = config.build().unwrap();
        check(&chain, &format!("{config:?}"));
    }
}

#[test]
fn kinds_have_their_documented_coefficients() {
    use TransformerConfig::*;
    let n_coefs = |config: TransformerConfig| config.build().unwrap().n_coefs;
    assert_eq!(
        n_coefs(Affine {
            scale_squash: Some(1.0)
        }),
        2
    );
    // alpha, beta, sigma, mu, nu; interior layers drop mu.
    let sas = |layers| Sas {
        layers,
        log_gamma_bounds: None,
        scale_squash: Some(1.0),
    };
    assert_eq!(n_coefs(sas(1)), 5);
    assert_eq!(n_coefs(sas(3)), 13);
    // nu, eps, b, r per layer, and the affine.
    let tangent = |layers, fix_b| TangentSas {
        layers,
        fix_b,
        scale_squash: Some(1.0),
    };
    assert_eq!(n_coefs(tangent(2, false)), 10);
    assert_eq!(n_coefs(tangent(2, true)), 8);
    // alpha_pos, alpha_neg, c and a after the first, and the affine.
    let asymmetric = |layers| AsymmetricSas {
        layers,
        log_gamma_bounds: None,
        scale_squash: Some(1.0),
    };
    assert_eq!(n_coefs(asymmetric(1)), 4);
    assert_eq!(n_coefs(asymmetric(3)), 12);
    assert!(sas(0).build().is_err());
}

#[test]
fn configs_deserialize_from_their_kind() {
    let parse = |json: &str| serde_json::from_str::<TransformerConfig>(json);
    assert_eq!(
        parse(r#"{"kind": "affine"}"#).unwrap(),
        TransformerConfig::Affine {
            scale_squash: Some(1.0)
        }
    );
    assert_eq!(
        parse(r#"{"kind": "affine", "scale_squash": null}"#).unwrap(),
        TransformerConfig::Affine { scale_squash: None }
    );
    assert_eq!(
        parse(r#"{"kind": "sas", "scale_squash": 2.5}"#).unwrap(),
        TransformerConfig::Sas {
            layers: 1,
            log_gamma_bounds: None,
            scale_squash: Some(2.5),
        }
    );
    assert_eq!(
        parse(r#"{"kind": "sas"}"#).unwrap(),
        TransformerConfig::Sas {
            layers: 1,
            log_gamma_bounds: None,
            scale_squash: Some(1.0),
        }
    );
    assert_eq!(
        parse(r#"{"kind": "tangent_sas", "layers": 2, "fix_b": true}"#).unwrap(),
        TransformerConfig::TangentSas {
            layers: 2,
            fix_b: true,
            scale_squash: Some(1.0),
        }
    );
    assert!(parse(r#"{"kind": "sas", "layer": 2}"#).is_err());
    assert!(parse(r#"{"kind": "spline"}"#).is_err());
}
