//! One harness, generic over [`Conditioner`], run for every kind with
//! different targets, with and without parents: each pushforward against its
//! pullback, against finite differences of the forward pass, and the
//! overridden defaults against the defaults.

use fearless_simd::{dispatch, prelude::*, Level};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

use super::{
    default_parent_jacobian, default_pullback_weights_stack, At, Conditioner, ConditionerConfig,
    Conditioners, SparseCotangents, Stack, StackMut,
};
use crate::triangular::transformer::{Transformer, TransformerConfig};

const STEP: f64 = 1e-6;
const TOL: f64 = 1e-6;
/// Pulled cotangents per stack, as the LM's `(t, l)`.
const K: usize = 2;

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

/// Random per-lane vectors.
fn tile<S: Simd>(simd: S, rng: &mut ChaCha8Rng, n: usize, scale: f64) -> Vec<S::f64s> {
    (0..n)
        .map(|_| S::f64s::from_slice(simd, &normal(rng, S::f64s::LEN, scale)))
        .collect()
}

fn lanes<S: Simd>(v: &[S::f64s]) -> Vec<f64> {
    v.iter()
        .flat_map(|v| (0..S::f64s::LEN).map(move |l| v[l]))
        .collect()
}

fn zeros<S: Simd>(simd: S, n: usize) -> Vec<S::f64s> {
    vec![S::f64s::splat(simd, 0.0); n]
}

/// `sum_i a_i b_i` per lane.
fn dot_lanes<S: Simd>(simd: S, a: &[S::f64s], b: &[S::f64s]) -> Vec<f64> {
    let mut acc = S::f64s::splat(simd, 0.0);
    for (&a, &b) in a.iter().zip(b) {
        acc += a * b;
    }
    lanes::<S>(&[acc])
}

/// `sum_k v_k w_k` per lane, with `w` per lane and `v` shared.
fn dot_shared<S: Simd>(simd: S, v: &[f64], w: &[S::f64s]) -> Vec<f64> {
    let mut acc = S::f64s::splat(simd, 0.0);
    for (&v, &w) in v.iter().zip(w) {
        acc += w * v;
    }
    lanes::<S>(&[acc])
}

fn shifted(values: &[f64], direction: &[f64], h: f64) -> Vec<f64> {
    values
        .iter()
        .zip(direction)
        .map(|(v, d)| v + h * d)
        .collect()
}

fn shifted_tile<S: Simd>(values: &[S::f64s], direction: &[S::f64s], h: f64) -> Vec<S::f64s> {
    values
        .iter()
        .zip(direction)
        .map(|(&v, &d)| v + d * h)
        .collect()
}

fn central<S: Simd>(plus: &[S::f64s], minus: &[S::f64s]) -> Vec<f64> {
    lanes::<S>(plus)
        .iter()
        .zip(lanes::<S>(minus))
        .map(|(p, m)| (p - m) / (2.0 * STEP))
        .collect()
}

struct Case<'a, C: Conditioner> {
    c: &'a C,
    n_coefs: usize,
    n_parent: usize,
    name: &'a str,
}

impl<C: Conditioner> Case<'_, C> {
    fn forward<S: Simd>(&self, simd: S, weights: &[f64], parents: &[S::f64s]) -> Vec<S::f64s> {
        let mut state = self.c.new_tile_state(simd, self.n_parent);
        let mut coefs = zeros(simd, self.n_coefs);
        self.c
            .forward_tile(simd, weights, parents, &mut state, &mut coefs);
        coefs
    }

    fn pullback_parents<S: Simd>(
        &self,
        simd: S,
        weights: &[f64],
        parents: &[S::f64s],
        coefs_bar: &[S::f64s],
    ) -> Vec<S::f64s> {
        let mut state = self.c.new_tile_state(simd, self.n_parent);
        let mut scratch = self.c.new_scratch(simd, self.n_parent);
        let mut coefs = zeros(simd, self.n_coefs);
        self.c
            .forward_tile(simd, weights, parents, &mut state, &mut coefs);
        let at = At {
            weights,
            parents,
            state: &state,
        };
        let mut out = zeros(simd, K * self.n_parent);
        if self.n_parent > 0 {
            self.c.pullback_parents_tile(
                simd,
                &at,
                Stack::new(coefs_bar, K),
                StackMut::new(&mut out, K),
                &mut scratch,
            );
        }
        out
    }

    fn run<S: Simd>(&self, simd: S, rng: &mut ChaCha8Rng) {
        let (c, n_coefs, n_parent) = (self.c, self.n_coefs, self.n_parent);
        let what = |check: &str| format!("{}, {n_parent} parents: {check}", self.name);
        let n_w = c.n_weights(n_parent);

        let mut strengths = Vec::new();
        c.ridge_strengths(n_parent, 2.0, &mut strengths);
        assert_eq!(strengths.len(), n_w, "{}", what("ridge strengths"));
        let mut init = vec![f64::NAN; n_w];
        c.init_weights(n_parent, rng, &mut init);
        assert!(init.iter().all(|w| w.is_finite()), "{}", what("init"));

        let parents = tile(simd, rng, n_parent, 1.0);
        let zero_coefs = self.forward(simd, &vec![0.0; n_w], &parents);
        let expected = vec![0.0; n_coefs * S::f64s::LEN];
        assert_close(
            &what("zero weights"),
            &lanes::<S>(&zero_coefs),
            &expected,
            0.0,
        );

        let weights = normal(rng, n_w, 0.7);
        let v = normal(rng, n_w, 1.0);
        let mut state = c.new_tile_state(simd, n_parent);
        let mut scratch = c.new_scratch(simd, n_parent);
        let mut coefs = zeros(simd, n_coefs);
        c.forward_tile(simd, &weights, &parents, &mut state, &mut coefs);
        // Only the targets move.
        for (k, value) in coefs.iter().enumerate() {
            if !c.outputs().coefs().contains(&k) {
                assert_eq!(
                    lanes::<S>(&[*value]),
                    vec![0.0; S::f64s::LEN],
                    "{}",
                    what("targets")
                );
            }
        }
        let at = At {
            weights: &weights,
            parents: &parents,
            state: &state,
        };

        // Weights: the pushforward against finite differences.
        let mut coefs_dot = zeros(simd, n_coefs);
        c.pushforward_weights_tile(simd, &at, &v, &mut coefs_dot, &mut scratch);
        let fd = central::<S>(
            &self.forward(simd, &shifted(&weights, &v, STEP), &parents),
            &self.forward(simd, &shifted(&weights, &v, -STEP), &parents),
        );
        assert_close(
            &what("pushforward weights"),
            &lanes::<S>(&coefs_dot),
            &fd,
            TOL,
        );

        // Parents: the pushforward against finite differences, and against
        // the pullback.
        let parents_dot = tile(simd, rng, K * n_parent, 1.0);
        let mut pushed = zeros(simd, K * n_coefs);
        if n_parent > 0 {
            c.pushforward_parents_tile(
                simd,
                &at,
                Stack::new(&parents_dot, K),
                StackMut::new(&mut pushed, K),
                &mut scratch,
            );
        }
        for i in 0..K {
            let direction = &parents_dot[i * n_parent..(i + 1) * n_parent];
            let fd = central::<S>(
                &self.forward(
                    simd,
                    &weights,
                    &shifted_tile::<S>(&parents, direction, STEP),
                ),
                &self.forward(
                    simd,
                    &weights,
                    &shifted_tile::<S>(&parents, direction, -STEP),
                ),
            );
            let got = lanes::<S>(&pushed[i * n_coefs..(i + 1) * n_coefs]);
            assert_close(&what("pushforward parents"), &got, &fd, TOL);
        }
        let coefs_bar = tile(simd, rng, K * n_coefs, 1.0);
        let pulled = self.pullback_parents(simd, &weights, &parents, &coefs_bar);
        for i in 0..K {
            let (cb, pd) = (
                &coefs_bar[i * n_coefs..(i + 1) * n_coefs],
                &parents_dot[i * n_parent..(i + 1) * n_parent],
            );
            assert_close(
                &what("parents adjoint"),
                &dot_lanes(simd, &pushed[i * n_coefs..(i + 1) * n_coefs], cb),
                &dot_lanes(simd, pd, &pulled[i * n_parent..(i + 1) * n_parent]),
                1e-12,
            );
        }

        // One draw, through the per-draw kernels, agrees with every lane.
        let packed = c.pack(n_parent, &weights);
        let mut draw = c.new_draw_state(n_parent);
        let (coefs_lanes, pulled_lanes) = (lanes::<S>(&coefs), lanes::<S>(&pulled));
        let cb_lanes = lanes::<S>(&coefs_bar);
        let width = S::f64s::LEN;
        for lane in 0..width {
            let at_lane = |values: &[f64], n: usize| -> Vec<f64> {
                (0..n).map(|k| values[k * width + lane]).collect()
            };
            let parents_lane = at_lane(&lanes::<S>(&parents), n_parent);
            let mut coefs_draw = vec![0.0; n_coefs];
            c.forward_draw(&packed, &parents_lane, &mut draw, true, &mut coefs_draw);
            assert_close(
                &what("forward draw"),
                &coefs_draw,
                &at_lane(&coefs_lanes, n_coefs),
                1e-12,
            );
            let mut parents_bar = vec![0.0; K * n_parent];
            c.pullback_parents_draw(
                &packed,
                &mut draw,
                Stack::new(&at_lane(&cb_lanes, K * n_coefs), K),
                StackMut::new(&mut parents_bar, K),
            );
            assert_close(
                &what("pullback parents draw"),
                &parents_bar,
                &at_lane(&pulled_lanes, K * n_parent),
                1e-12,
            );
        }

        // The tangent of the parent pullback, along `v` and `coefs_bar_dot`.
        let coefs_bar_dot = tile(simd, rng, K * n_coefs, 1.0);
        let mut tangent = zeros(simd, K * n_parent);
        if n_parent > 0 {
            c.pushforward_weights_tile(simd, &at, &v, &mut zeros(simd, n_coefs), &mut scratch);
            c.pullback_parents_tangent_tile(
                simd,
                &at,
                &v,
                Stack::new(&coefs_bar, K),
                Stack::new(&coefs_bar_dot, K),
                StackMut::new(&mut tangent, K),
                &mut scratch,
            );
        }
        let fd = central::<S>(
            &self.pullback_parents(
                simd,
                &shifted(&weights, &v, STEP),
                &parents,
                &shifted_tile::<S>(&coefs_bar, &coefs_bar_dot, STEP),
            ),
            &self.pullback_parents(
                simd,
                &shifted(&weights, &v, -STEP),
                &parents,
                &shifted_tile::<S>(&coefs_bar, &coefs_bar_dot, -STEP),
            ),
        );
        assert_close(
            &what("pullback parents tangent"),
            &lanes::<S>(&tangent),
            &fd,
            TOL,
        );

        // The weight pullback is the transpose of both weight tangents:
        // `<weights_bar, v> = <coefs_dot, coefs_bar> + sum_i <(dC^T v)
        // pulled_i, parents_bar_bar_i>`.
        let pulled_cb = tile(simd, rng, K * n_coefs, 1.0);
        let parents_bar_bar = tile(simd, rng, K * n_parent, 1.0);
        let cb = tile(simd, rng, n_coefs, 1.0);
        let mut weights_bar = zeros(simd, n_w);
        if n_parent > 0 {
            c.pushforward_parents_tile(
                simd,
                &at,
                Stack::new(&parents_bar_bar, K),
                StackMut::new(&mut zeros(simd, K * n_coefs), K),
                &mut scratch,
            );
        }
        c.pullback_weights_tile(
            simd,
            &at,
            &cb,
            Stack::new(&pulled_cb, K),
            Stack::new(&parents_bar_bar, K),
            &mut weights_bar,
            &mut scratch,
        );
        let mut curvature = zeros(simd, K * n_parent);
        if n_parent > 0 {
            c.pushforward_weights_tile(simd, &at, &v, &mut zeros(simd, n_coefs), &mut scratch);
            c.pullback_parents_tangent_tile(
                simd,
                &at,
                &v,
                Stack::new(&pulled_cb, K),
                Stack::new(&zeros(simd, K * n_coefs), K),
                StackMut::new(&mut curvature, K),
                &mut scratch,
            );
        }
        let mut expected = dot_lanes(simd, &coefs_dot, &cb);
        let extra = dot_lanes(simd, &curvature, &parents_bar_bar);
        for (e, x) in expected.iter_mut().zip(extra) {
            *e += x;
        }
        assert_close(
            &what("weights adjoint"),
            &dot_shared(simd, &v, &weights_bar),
            &expected,
            1e-12,
        );

        if n_parent == 0 {
            return;
        }

        // The overridden defaults.
        let mut jacobian = zeros(simd, n_coefs * n_parent);
        c.add_parent_jacobian_tile(simd, &at, &mut jacobian, &mut scratch);
        let mut reference = zeros(simd, n_coefs * n_parent);
        default_parent_jacobian(c, simd, &at, &mut reference, &mut scratch);
        assert_close(
            &what("parent jacobian"),
            &lanes::<S>(&jacobian),
            &lanes::<S>(&reference),
            1e-12,
        );

        let n_r = 2 + n_parent;
        let edges: Vec<Option<usize>> = (0..n_r).map(|r| (r >= 2).then(|| r - 2)).collect();
        let coefs_bars = tile(simd, rng, n_r * n_coefs, 1.0);
        let edge_values = tile(simd, rng, n_r * K, 1.0);
        let cotangents = SparseCotangents {
            coefs_bar: Stack::new(&coefs_bars, n_r),
            pulled_coefs_bar: Stack::new(&pulled_cb, K),
            edges: &edges,
            edge_values: Stack::new(&edge_values, n_r),
        };
        let mut rows = zeros(simd, n_r * n_w);
        c.pullback_weights_stack_tile(
            simd,
            &at,
            &cotangents,
            StackMut::new(&mut rows, n_r),
            &mut scratch,
        );
        let mut reference = zeros(simd, n_r * n_w);
        default_pullback_weights_stack(
            c,
            simd,
            &at,
            &cotangents,
            StackMut::new(&mut reference, n_r),
            &mut scratch,
        );
        assert_close(
            &what("weights stack"),
            &lanes::<S>(&rows),
            &lanes::<S>(&reference),
            1e-12,
        );
    }
}

fn check<C: Conditioner>(c: &C, n_coefs: usize, name: &str) {
    for n_parent in [0, 1, 3] {
        let case = Case {
            c,
            n_coefs,
            n_parent,
            name,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(n_parent as u64);
        dispatch!(Level::new(), simd => case.run(simd, &mut rng));
    }
}

fn configs() -> Vec<ConditionerConfig> {
    let parse = |json: &str| serde_json::from_str::<ConditionerConfig>(json).unwrap();
    vec![
        parse(r#"{"kind": "constant", "target": "all", "ridge": {"shape": 10.0}}"#),
        parse(r#"{"kind": "constant", "target": ["scale", "location"]}"#),
        parse(r#"{"kind": "linear", "target": ["location", "scale"], "ridge": 0.1}"#),
        parse(r#"{"kind": "linear", "target": "shape"}"#),
        parse(r#"{"kind": "mlp", "width": 4, "target": "all"}"#),
        parse(
            r#"{"kind": "mlp", "width": 3, "target": ["location", "shape"],
                "activation": "gelu_tanh", "input_squash": 0.7, "init_output_scale": 1.0}"#,
        ),
    ]
}

#[test]
fn every_kind_passes_the_harness() {
    let transformer = TransformerConfig::TangentSas {
        layers: 2,
        fix_b: false,
        scale_squash: Some(1.0),
    }
    .build()
    .unwrap();
    let n_coefs = transformer.n_coefs();
    for config in configs() {
        let conditioners = Conditioners::new(std::slice::from_ref(&config), &transformer).unwrap();
        let name = format!("{config:?}");
        if let Some(c) = &conditioners.constant {
            check(c, n_coefs, &name);
        }
        if let Some(c) = &conditioners.linear {
            check(c, n_coefs, &name);
        }
        for c in &conditioners.mlps {
            check(c, n_coefs, &name);
        }
    }
}

#[test]
fn configs_are_checked() {
    let affine = TransformerConfig::Affine {
        scale_squash: Some(1.0),
    }
    .build()
    .unwrap();
    let parse = |json: &str| serde_json::from_str::<Vec<ConditionerConfig>>(json);
    let build = |json: &str| Conditioners::new(&parse(json).unwrap(), &affine);
    // An affine transformer has no shape coefficients.
    assert!(build(r#"[{"kind": "constant", "target": "shape"}]"#).is_err());
    assert!(build(
        r#"[{"kind": "constant", "target": "location"},
                      {"kind": "constant", "target": "scale"}]"#
    )
    .is_err());
    assert!(build(
        r#"[{"kind": "linear", "target": "location"},
                      {"kind": "linear", "target": "scale"}]"#
    )
    .is_err());
    assert!(build(r#"[{"kind": "constant", "target": "all", "ridge": -1.0}]"#).is_err());
    assert!(parse(r#"[{"kind": "constant", "target": "all", "ridge": {"loc": 1.0}}]"#).is_err());
    assert!(parse(r#"[{"kind": "mlp", "target": "all"}]"#).is_err());
    let ok = build(
        r#"[{"kind": "mlp", "width": 2, "target": "all"},
            {"kind": "linear", "target": "location"},
            {"kind": "constant", "target": "all"},
            {"kind": "mlp", "width": 3, "target": "scale"}]"#,
    )
    .unwrap();
    // Constant, linear, then the MLPs in list order.
    let n_parent = 2;
    let ranges = ok.weight_ranges(n_parent);
    assert_eq!(ranges, vec![0..2, 2..4, 4..4 + 2 * 5, 14..14 + 3 * 4]);
    assert_eq!(ok.n_weights(n_parent), 26);
    // Without parents, only the constant.
    assert_eq!(ok.n_weights(0), 2);
}

fn build(json: &str, transformer: &impl Transformer) -> Conditioners {
    let configs: Vec<ConditionerConfig> = serde_json::from_str(json).unwrap();
    Conditioners::new(&configs, transformer).unwrap()
}

/// The coefficients of `conditioners` at one draw, `parents` (one lane).
fn coefs_at(conditioners: &Conditioners, weights: &[f64], parents: &[f64]) -> Vec<f64> {
    fn on<S: Simd>(simd: S, c: &Conditioners, weights: &[f64], parents: &[f64]) -> Vec<f64> {
        let parents: Vec<_> = parents.iter().map(|&p| S::f64s::splat(simd, p)).collect();
        let mut work = c.workspace(simd, parents.len());
        let mut coefs = zeros(simd, c.n_coefs());
        c.forward_tile(simd, weights, &parents, &mut work, &mut coefs);
        coefs.iter().map(|v| v[0]).collect()
    }
    dispatch!(Level::new(), simd => on(simd, conditioners, weights, parents))
}

#[test]
fn weights_carry_over() {
    let transformer = TransformerConfig::Sas {
        layers: 2,
        log_gamma_bounds: None,
        scale_squash: Some(1.0),
    }
    .build()
    .unwrap();
    let full = build(
        r#"[{"kind": "constant", "target": "all"},
            {"kind": "linear", "target": ["location", "scale"]},
            {"kind": "mlp", "width": 3, "target": "all"},
            {"kind": "mlp", "width": 2, "target": "shape"}]"#,
        &transformer,
    );
    let n_parent = 3;
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let weights = normal(&mut rng, full.n_weights(n_parent), 1.0);
    let carried = |to: &Conditioners| {
        let mut out = normal(&mut rng.clone(), to.n_weights(n_parent), 1.0);
        let plan = to.carry_plan(&full).unwrap();
        to.carry_weights(&full, &plan, n_parent, &weights, &mut out);
        out
    };

    // The same list carries everything.
    assert_eq!(carried(&full), weights);

    // The linear stage keeps the constant off the shape and the linear
    // location weights, and so the coefficients there without the MLPs.
    let linear = build(
        r#"[{"kind": "constant", "target": ["location", "scale", "other"]},
            {"kind": "linear", "target": "location"}]"#,
        &transformer,
    );
    let mut linear_part = weights.clone();
    for range in &full.weight_ranges(n_parent)[2..] {
        linear_part[range.clone()].fill(0.0);
    }
    let parents = [0.3, -1.2, 0.8];
    let reference = coefs_at(&full, &linear_part, &parents);
    let ours = coefs_at(&linear, &carried(&linear), &parents);
    for (k, (&ours, &reference)) in ours.iter().zip(&reference).enumerate() {
        let shape = transformer.shape_coefs().any(|s| s == k);
        let scale = Some(k) == transformer.scale_coef();
        if shape {
            assert_eq!(ours, 0.0, "coefficient {k}");
        } else if scale {
            // Without the linear scale weights.
            let constant_only = linear_part[full.weight_ranges(n_parent)[0].clone()][k];
            assert!((ours - constant_only).abs() < 1e-14, "coefficient {k}");
        } else {
            assert!((ours - reference).abs() < 1e-12, "coefficient {k}");
        }
    }

    // An MLP with other targets keeps its units and the shared outputs;
    // the new outputs start at zero.
    let moved = build(
        r#"[{"kind": "mlp", "width": 3, "target": ["location", "scale"]}]"#,
        &transformer,
    );
    let out = carried(&moved);
    let mlp = &moved.mlps[0];
    let from = &full.mlps[0];
    let from_weights = &weights[full.weight_ranges(n_parent)[2].clone()];
    for u in 0..3 {
        let (w1, b1, w2) = mlp.unit(&out, n_parent, u);
        let (from_w1, from_b1, from_w2) = from.unit(from_weights, n_parent, u);
        assert_eq!((w1, b1), (from_w1, from_b1));
        for (j, &k) in mlp.outputs().coefs().iter().enumerate() {
            assert_eq!(w2[j], from_w2[k]);
        }
    }
    // A different width is a new MLP, and two candidates are an error.
    let wider = build(
        r#"[{"kind": "mlp", "width": 4, "target": "all"}]"#,
        &transformer,
    );
    assert!(wider.carry_plan(&full).unwrap().mlps == vec![None]);
    let twice = build(
        r#"[{"kind": "mlp", "width": 3, "target": "location"},
            {"kind": "mlp", "width": 3, "target": "scale"}]"#,
        &transformer,
    );
    assert!(twice.carry_plan(&full).is_err());
}

#[test]
fn coordinates_change_the_location_exactly() {
    for config in [
        TransformerConfig::Affine {
            scale_squash: Some(1.0),
        },
        TransformerConfig::Sas {
            layers: 2,
            log_gamma_bounds: None,
            scale_squash: Some(2.5),
        },
    ] {
        let transformer = config.build().unwrap();
        let conditioners = build(
            r#"[{"kind": "constant", "target": "all"},
                {"kind": "linear", "target": "location"}]"#,
            &transformer,
        );
        let n_parent = 2;
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let weights = normal(&mut rng, conditioners.n_weights(n_parent), 1.0);
        let (own, parents) = ((1.7, -0.3), [(0.6, 0.4), (2.5, -1.1)]);
        let mut moved = weights.clone();
        conditioners.change_coordinates(&transformer, own, &parents, &mut moved);

        let v = [0.9, -0.2];
        let v_new: Vec<f64> = v
            .iter()
            .zip(&parents)
            .map(|(v, (a, b))| a * v + b)
            .collect();
        let before = coefs_at(&conditioners, &weights, &v);
        let after = coefs_at(&conditioners, &moved, &v_new);
        let (location, scale) = (
            transformer.location_coef().unwrap(),
            transformer.scale_coef().unwrap(),
        );
        let expected = transformer.shifted_location_coef(before[location], own.0, own.1);
        assert!((after[location] - expected).abs() < 1e-12, "{config:?}");
        let expected = transformer.rescaled_scale_coef(before[scale], own.0);
        assert!((after[scale] - expected).abs() < 1e-12, "{config:?}");
    }
}

#[test]
fn coordinates_change_a_linear_log_scale_exactly() {
    for config in [
        TransformerConfig::Affine { scale_squash: None },
        TransformerConfig::Sas {
            layers: 2,
            log_gamma_bounds: None,
            scale_squash: None,
        },
    ] {
        let transformer = config.build().unwrap();
        let conditioners = build(
            r#"[{"kind": "constant", "target": "all"},
                {"kind": "linear", "target": ["location", "scale"]}]"#,
            &transformer,
        );
        let n_parent = 2;
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let weights = normal(&mut rng, conditioners.n_weights(n_parent), 1.0);
        let (own, parents) = ((1.7, -0.3), [(0.6, 0.4), (2.5, -1.1)]);
        let mut moved = weights.clone();
        conditioners.change_coordinates(&transformer, own, &parents, &mut moved);

        let v = [0.9, -0.2];
        let v_new: Vec<f64> = v
            .iter()
            .zip(&parents)
            .map(|(v, (a, b))| a * v + b)
            .collect();
        let before = coefs_at(&conditioners, &weights, &v);
        let after = coefs_at(&conditioners, &moved, &v_new);
        let scale = transformer.scale_coef().unwrap();
        let expected = transformer.rescaled_scale_coef(before[scale], own.0);
        assert!((after[scale] - expected).abs() < 1e-12, "{config:?}");
    }
}
