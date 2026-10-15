//! The MLP conditioner: one hidden layer of `n_unit` units and no output
//! bias,
//!
//! ```text
//! a = W1 z + b1,   h = act(a),   coefs[targets] += W2 h,
//! ```
//!
//! with `z` the parents, or `c asinh(y / c)` of them with an input squash
//! `c`, and `dz = z'(y)`. So `C = W2 diag(h') W1 diag(dz)`. The weights are
//! laid out by unit: `W1[u, :]`, `b1[u]`, `W2[:, u]`, so every kernel walks
//! the units with each unit's weights contiguous. Without parents it has no
//! weights: its output would be a constant, which is the constant
//! conditioner's.
//!
//! Names: `h1`, `h2` are `h'`, `h''`; `zeta` is `W1[u, :] . (dz *
//! parents_bar_bar)`, the units' share of the pullback of
//! `parents_bar_bar`; `a_dot` the units' inputs' tangent.

use anyhow::{bail, Result};
use fearless_simd::prelude::*;
use rand::RngExt;
use serde::{Deserialize, Serialize};

use super::{At, Conditioner, Outputs, SparseCotangents, Stack, StackMut};
use crate::simd_math;
use crate::triangular::simd::{axpy, axpy_lanes, dot};

/// The hidden units' activation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Activation {
    #[default]
    Softplus,
    /// `jax.nn.gelu`'s default, the tanh approximation.
    GeluTanh,
}

impl Activation {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Softplus => "softplus",
            Self::GeluTanh => "gelu_tanh",
        }
    }

    /// `(h, h', h'')` at `a`.
    #[inline(always)]
    fn eval<S: Simd>(self, simd: S, a: S::f64s) -> (S::f64s, S::f64s, S::f64s) {
        match self {
            Self::Softplus => simd_math::softplus(simd, a),
            Self::GeluTanh => simd_math::gelu_tanh(simd, a),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Mlp {
    outputs: Outputs,
    n_unit: usize,
    activation: Activation,
    squash: Option<f64>,
    init_output_scale: f64,
    hidden_ridge: f64,
}

/// What [`Mlp::forward_tile`] records.
pub(crate) struct MlpTile<S: Simd> {
    /// What the units see of the parents, and its derivative.
    z: Vec<S::f64s>,
    dz: Vec<S::f64s>,
    /// The activation and its first two derivatives, per unit.
    h: Vec<S::f64s>,
    h1: Vec<S::f64s>,
    h2: Vec<S::f64s>,
    /// The outputs, before they are added to the coefficients.
    out: Vec<S::f64s>,
}

/// Per-thread workspace; `a_dot`, `zeta` and `edge_z` carry work from one
/// method to the next (see [`Conditioner`]).
pub(crate) struct MlpScratch<S: Simd> {
    /// Coefficient-space vectors gathered to the outputs, `n_out` each.
    gathered: Vec<S::f64s>,
    gathered_dot: Vec<S::f64s>,
    /// Outputs, `n_out` each.
    out: Vec<S::f64s>,
    /// Parent-space vectors, `n_parent` each.
    edge: Vec<S::f64s>,
    /// `dz * parents_bar_bar`, by `pushforward_parents_tile`.
    edge_z: Vec<S::f64s>,
    /// Per unit, `k` values, by `pushforward_parents_tile`.
    zeta: Vec<S::f64s>,
    /// Per unit, by `pushforward_weights_tile`.
    a_dot: Vec<S::f64s>,
    /// Per unit, `k` values: `W2[:, u] . pulled_coefs_bar[i]`.
    unit_p: Vec<S::f64s>,
}

/// `v` resized to `n` zeros.
#[inline(always)]
fn zeros<S: Simd>(simd: S, v: &mut Vec<S::f64s>, n: usize) {
    v.clear();
    v.resize(n, S::f64s::splat(simd, 0.0));
}

impl Mlp {
    pub(crate) fn new(
        outputs: Outputs,
        n_unit: usize,
        activation: Activation,
        squash: Option<f64>,
        init_output_scale: f64,
        hidden_ridge: f64,
    ) -> Result<Self> {
        if let Some(c) = squash {
            if !(c.is_finite() && c > 0.0) {
                bail!("input_squash must be positive and finite, got {c}");
            }
        }
        if !(init_output_scale.is_finite() && init_output_scale >= 0.0) {
            bail!("init_output_scale must be finite and non-negative, got {init_output_scale}");
        }
        Ok(Self {
            outputs,
            n_unit,
            activation,
            squash,
            init_output_scale,
            hidden_ridge,
        })
    }

    pub(crate) fn n_unit(&self) -> usize {
        self.n_unit
    }

    pub(crate) fn activation(&self) -> Activation {
        self.activation
    }

    pub(crate) fn squash(&self) -> Option<f64> {
        self.squash
    }

    #[inline(always)]
    fn stride(&self, n_parent: usize) -> usize {
        n_parent + 1 + self.outputs.len()
    }

    /// Unit `u`'s `(W1[u, :], b1[u], W2[:, u])` in `weights`.
    #[inline(always)]
    pub(crate) fn unit<'a>(
        &self,
        weights: &'a [f64],
        n_parent: usize,
        u: usize,
    ) -> (&'a [f64], f64, &'a [f64]) {
        let unit = &weights[u * self.stride(n_parent)..][..self.stride(n_parent)];
        let (w1, rest) = unit.split_at(n_parent);
        (w1, rest[0], &rest[1..])
    }

    /// Unit `u`'s `(W1[u, :], b1[u], W2[:, u])` in `weights_bar`.
    #[inline(always)]
    fn unit_mut<'a, T>(
        &self,
        weights_bar: &'a mut [T],
        n_parent: usize,
        u: usize,
    ) -> (&'a mut [T], &'a mut T, &'a mut [T]) {
        let stride = self.stride(n_parent);
        let unit = &mut weights_bar[u * stride..][..stride];
        let (w1, rest) = unit.split_at_mut(n_parent);
        let (b1, w2) = rest.split_first_mut().expect("a unit has a bias");
        (w1, b1, w2)
    }

    /// `out[i]` = `src[i]` gathered to the outputs, for every `i`.
    #[inline(always)]
    fn gather_stack<S: Simd>(&self, simd: S, src: Stack<S::f64s>, out: &mut Vec<S::f64s>) {
        let n_out = self.outputs.len();
        zeros(simd, out, src.k() * n_out);
        for (i, out) in out.chunks_exact_mut(n_out).enumerate() {
            self.outputs.gather(src.get(i), out);
        }
    }
}

impl Conditioner for Mlp {
    type TileState<S: Simd> = MlpTile<S>;
    type Scratch<S: Simd> = MlpScratch<S>;

    fn outputs(&self) -> &Outputs {
        &self.outputs
    }

    fn n_weights(&self, n_parent: usize) -> usize {
        if n_parent == 0 {
            0
        } else {
            self.n_unit * self.stride(n_parent)
        }
    }

    fn ridge_strengths(&self, n_parent: usize, loc_scale: f64, out: &mut Vec<f64>) {
        if n_parent == 0 {
            return;
        }
        for _ in 0..self.n_unit {
            out.extend(std::iter::repeat_n(self.hidden_ridge, n_parent));
            out.push(0.0);
            out.extend((0..self.outputs.len()).map(|j| self.outputs.strength(j, loc_scale)));
        }
    }

    /// Like `equinox.nn.Linear`: uniform in `±1 / sqrt(fan_in)`, the output
    /// weights scaled by `init_output_scale`.
    fn init_weights<R: rand::Rng + ?Sized>(&self, n_parent: usize, rng: &mut R, out: &mut [f64]) {
        if n_parent == 0 {
            return;
        }
        let in_limit = 1.0 / (n_parent as f64).sqrt();
        let out_limit = 1.0 / (self.n_unit as f64).sqrt();
        for u in 0..self.n_unit {
            let (w1, b1, w2) = self.unit_mut(out, n_parent, u);
            for w in w1.iter_mut().chain(std::iter::once(b1)) {
                *w = rng.random_range(-in_limit..in_limit);
            }
            for w in w2 {
                *w = self.init_output_scale * rng.random_range(-out_limit..out_limit);
            }
        }
    }

    /// The same units: width, activation and input squash.
    fn same_structure(&self, other: &Self) -> bool {
        self.n_unit == other.n_unit
            && self.activation == other.activation
            && self.squash == other.squash
    }

    fn carry_weights(&self, from: &Self, n_parent: usize, from_weights: &[f64], out: &mut [f64]) {
        if n_parent == 0 {
            return;
        }
        for u in 0..self.n_unit {
            let (w1, b1, w2) = from.unit(from_weights, n_parent, u);
            let (out_w1, out_b1, out_w2) = self.unit_mut(out, n_parent, u);
            out_w1.copy_from_slice(w1);
            *out_b1 = b1;
            out_w2.fill(0.0);
            for (j, j_from) in self.outputs.shared(&from.outputs) {
                out_w2[j] = w2[j_from];
            }
        }
    }

    fn new_tile_state<S: Simd>(&self, simd: S, max_parent: usize) -> MlpTile<S> {
        let zero = S::f64s::splat(simd, 0.0);
        MlpTile {
            z: Vec::with_capacity(max_parent),
            dz: Vec::with_capacity(max_parent),
            h: vec![zero; self.n_unit],
            h1: vec![zero; self.n_unit],
            h2: vec![zero; self.n_unit],
            out: vec![zero; self.outputs.len()],
        }
    }

    fn new_scratch<S: Simd>(&self, _: S, _: usize) -> MlpScratch<S> {
        MlpScratch {
            gathered: Vec::new(),
            gathered_dot: Vec::new(),
            out: Vec::new(),
            edge: Vec::new(),
            edge_z: Vec::new(),
            zeta: Vec::new(),
            a_dot: Vec::new(),
            unit_p: Vec::new(),
        }
    }

    #[inline(always)]
    fn forward_tile<S: Simd>(
        &self,
        simd: S,
        weights: &[f64],
        parents: &[S::f64s],
        state: &mut MlpTile<S>,
        coefs: &mut [S::f64s],
    ) {
        let n_parent = parents.len();
        if n_parent == 0 {
            return;
        }
        state.z.clear();
        state.dz.clear();
        match self.squash {
            None => {
                state.z.extend_from_slice(parents);
                state.dz.resize(n_parent, S::f64s::splat(simd, 1.0));
            }
            Some(c) => {
                let one = S::f64s::splat(simd, 1.0);
                for &y in parents {
                    let v = y * (1.0 / c);
                    state.z.push(simd_math::asinh(simd, v) * c);
                    state.dz.push(one / simd_math::hypot_one(simd, v));
                }
            }
        }
        state.out.fill(S::f64s::splat(simd, 0.0));
        for u in 0..self.n_unit {
            let (w1, b1, w2) = self.unit(weights, n_parent, u);
            let a = dot(simd, w1, &state.z) + b1;
            let (h, h1, h2) = self.activation.eval(simd, a);
            state.h[u] = h;
            state.h1[u] = h1;
            state.h2[u] = h2;
            axpy(simd, h, w2, &mut state.out);
        }
        self.outputs.scatter_add::<S>(&state.out, coefs);
    }

    #[inline(always)]
    fn pullback_parents_tile<S: Simd>(
        &self,
        simd: S,
        at: &At<Self, S>,
        coefs_bar: Stack<S::f64s>,
        mut parents_bar: StackMut<S::f64s>,
        scratch: &mut MlpScratch<S>,
    ) {
        let n_parent = at.parents.len();
        if n_parent == 0 {
            return;
        }
        let (k, n_out, st) = (coefs_bar.k(), self.outputs.len(), at.state);
        self.gather_stack(simd, coefs_bar, &mut scratch.gathered);
        zeros(simd, &mut scratch.edge, k * n_parent);
        for u in 0..self.n_unit {
            let (w1, _, w2) = self.unit(at.weights, n_parent, u);
            let edges = scratch.edge.chunks_exact_mut(n_parent);
            for (edge, g) in edges.zip(scratch.gathered.chunks_exact(n_out)) {
                axpy(simd, dot(simd, w2, g) * st.h1[u], w1, edge);
            }
        }
        for (i, edge) in scratch.edge.chunks_exact(n_parent).enumerate() {
            for ((out, &e), &dz) in parents_bar.get(i).iter_mut().zip(edge).zip(&st.dz) {
                *out += e * dz;
            }
        }
    }

    #[inline(always)]
    fn pushforward_weights_tile<S: Simd>(
        &self,
        simd: S,
        at: &At<Self, S>,
        v: &[f64],
        coefs_dot: &mut [S::f64s],
        scratch: &mut MlpScratch<S>,
    ) {
        let n_parent = at.parents.len();
        if n_parent == 0 {
            return;
        }
        let st = at.state;
        zeros(simd, &mut scratch.out, self.outputs.len());
        zeros(simd, &mut scratch.a_dot, self.n_unit);
        for u in 0..self.n_unit {
            let (v1, vb1, v2) = self.unit(v, n_parent, u);
            let (_, _, w2) = self.unit(at.weights, n_parent, u);
            let a_dot = dot(simd, v1, &st.z) + vb1;
            scratch.a_dot[u] = a_dot;
            axpy(simd, st.h[u], v2, &mut scratch.out);
            axpy(simd, st.h1[u] * a_dot, w2, &mut scratch.out);
        }
        self.outputs.scatter_add::<S>(&scratch.out, coefs_dot);
    }

    #[inline(always)]
    fn pullback_parents_tangent_tile<S: Simd>(
        &self,
        simd: S,
        at: &At<Self, S>,
        v: &[f64],
        coefs_bar: Stack<S::f64s>,
        coefs_bar_dot: Stack<S::f64s>,
        mut parents_bar_dot: StackMut<S::f64s>,
        scratch: &mut MlpScratch<S>,
    ) {
        let n_parent = at.parents.len();
        if n_parent == 0 {
            return;
        }
        let (k, n_out, st) = (coefs_bar.k(), self.outputs.len(), at.state);
        self.gather_stack(simd, coefs_bar, &mut scratch.gathered);
        self.gather_stack(simd, coefs_bar_dot, &mut scratch.gathered_dot);
        zeros(simd, &mut scratch.edge, k * n_parent);
        for u in 0..self.n_unit {
            let (w1, _, w2) = self.unit(at.weights, n_parent, u);
            let (v1, _, v2) = self.unit(v, n_parent, u);
            let (h1, curvature) = (st.h1[u], st.h2[u] * scratch.a_dot[u]);
            let stacks = scratch
                .edge
                .chunks_exact_mut(n_parent)
                .zip(scratch.gathered.chunks_exact(n_out))
                .zip(scratch.gathered_dot.chunks_exact(n_out));
            for ((edge, g), g_dot) in stacks {
                let p = dot(simd, w2, g);
                let alpha = (dot(simd, w2, g_dot) + dot(simd, v2, g)) * h1 + p * curvature;
                axpy(simd, alpha, w1, edge);
                axpy(simd, p * h1, v1, edge);
            }
        }
        for (i, edge) in scratch.edge.chunks_exact(n_parent).enumerate() {
            for ((out, &e), &dz) in parents_bar_dot.get(i).iter_mut().zip(edge).zip(&st.dz) {
                *out += e * dz;
            }
        }
    }

    #[inline(always)]
    fn pushforward_parents_tile<S: Simd>(
        &self,
        simd: S,
        at: &At<Self, S>,
        parents_dot: Stack<S::f64s>,
        mut coefs_dot: StackMut<S::f64s>,
        scratch: &mut MlpScratch<S>,
    ) {
        let n_parent = at.parents.len();
        if n_parent == 0 {
            return;
        }
        let (k, n_out, st) = (parents_dot.k(), self.outputs.len(), at.state);
        scratch.edge_z.clear();
        for i in 0..k {
            let edge = parents_dot
                .get(i)
                .iter()
                .zip(&st.dz)
                .map(|(&p, &dz)| p * dz);
            scratch.edge_z.extend(edge);
        }
        zeros(simd, &mut scratch.zeta, self.n_unit * k);
        zeros(simd, &mut scratch.out, k * n_out);
        for u in 0..self.n_unit {
            let (w1, _, w2) = self.unit(at.weights, n_parent, u);
            let stacks = scratch
                .edge_z
                .chunks_exact(n_parent)
                .zip(scratch.out.chunks_exact_mut(n_out));
            for (i, (edge_z, out)) in stacks.enumerate() {
                let zeta = dot(simd, w1, edge_z);
                scratch.zeta[u * k + i] = zeta;
                axpy(simd, st.h1[u] * zeta, w2, out);
            }
        }
        for (i, out) in scratch.out.chunks_exact(n_out).enumerate() {
            self.outputs.scatter_add::<S>(out, coefs_dot.get(i));
        }
    }

    #[inline(always)]
    fn pullback_weights_tile<S: Simd>(
        &self,
        simd: S,
        at: &At<Self, S>,
        coefs_bar: &[S::f64s],
        pulled_coefs_bar: Stack<S::f64s>,
        _: Stack<S::f64s>,
        weights_bar: &mut [S::f64s],
        scratch: &mut MlpScratch<S>,
    ) {
        let n_parent = at.parents.len();
        if n_parent == 0 {
            return;
        }
        let (k, n_out, st) = (pulled_coefs_bar.k(), self.outputs.len(), at.state);
        zeros(simd, &mut scratch.out, n_out);
        self.outputs.gather(coefs_bar, &mut scratch.out);
        self.gather_stack(simd, pulled_coefs_bar, &mut scratch.gathered);
        for u in 0..self.n_unit {
            let (_, _, w2) = self.unit(at.weights, n_parent, u);
            let (out_w1, out_b1, out_w2) = self.unit_mut(weights_bar, n_parent, u);
            let (h, h1, h2) = (st.h[u], st.h1[u], st.h2[u]);
            let mut a_bar = h1 * dot(simd, w2, &scratch.out);
            axpy_lanes(simd, h, &scratch.out, out_w2);
            let stacks = scratch
                .gathered
                .chunks_exact(n_out)
                .zip(scratch.edge_z.chunks_exact(n_parent));
            for (i, (pulled, edge_z)) in stacks.enumerate() {
                let p = dot(simd, w2, pulled);
                let zeta = scratch.zeta[u * k + i];
                a_bar += h2 * p * zeta;
                axpy_lanes(simd, h1 * zeta, pulled, out_w2);
                axpy_lanes(simd, h1 * p, edge_z, out_w1);
            }
            axpy_lanes(simd, a_bar, &st.z, out_w1);
            *out_b1 += a_bar;
        }
    }

    #[inline(always)]
    fn add_parent_jacobian_tile<S: Simd>(
        &self,
        _: S,
        at: &At<Self, S>,
        c: &mut [S::f64s],
        _: &mut MlpScratch<S>,
    ) {
        let n_parent = at.parents.len();
        if n_parent == 0 {
            return;
        }
        let (n_coefs, st) = (c.len() / n_parent, at.state);
        for u in 0..self.n_unit {
            let (w1, _, w2) = self.unit(at.weights, n_parent, u);
            let columns = c.chunks_exact_mut(n_coefs).zip(w1).zip(&st.dz);
            for ((column, &w1), &dz) in columns {
                let factor = st.h1[u] * dz * w1;
                for (&w2, &k) in w2.iter().zip(self.outputs.coefs()) {
                    column[k] += factor * w2;
                }
            }
        }
    }

    /// With a one-hot `parents_bar_bar` on parent `j`, `zeta` is `dz[j]
    /// W1[u, j]` times its values, so the units read `W1[:, j]` directly.
    #[inline(always)]
    fn pullback_weights_stack_tile<S: Simd>(
        &self,
        simd: S,
        at: &At<Self, S>,
        cotangents: &SparseCotangents<S>,
        mut weights_bar: StackMut<S::f64s>,
        scratch: &mut MlpScratch<S>,
    ) {
        let n_parent = at.parents.len();
        if n_parent == 0 {
            return;
        }
        let pulled = cotangents.pulled_coefs_bar;
        let (k, n_out, st) = (pulled.k(), self.outputs.len(), at.state);
        self.gather_stack(simd, pulled, &mut scratch.gathered);
        zeros(simd, &mut scratch.unit_p, self.n_unit * k);
        for u in 0..self.n_unit {
            let (_, _, w2) = self.unit(at.weights, n_parent, u);
            for (i, pulled) in scratch.gathered.chunks_exact(n_out).enumerate() {
                scratch.unit_p[u * k + i] = dot(simd, w2, pulled);
            }
        }
        zeros(simd, &mut scratch.out, n_out);
        zeros(simd, &mut scratch.gathered_dot, n_out);
        for (r, &edge) in cotangents.edges.iter().enumerate() {
            let values = cotangents.edge_values.get(r);
            self.outputs
                .gather(cotangents.coefs_bar.get(r), &mut scratch.out);
            // The edge's direction in the outputs, `sum_i values[i]
            // pulled[i]`.
            if edge.is_some() {
                scratch.gathered_dot.fill(S::f64s::splat(simd, 0.0));
                for (&value, pulled) in values.iter().zip(scratch.gathered.chunks_exact(n_out)) {
                    axpy_lanes(simd, value, pulled, &mut scratch.gathered_dot);
                }
            }
            let row = weights_bar.get(r);
            for u in 0..self.n_unit {
                let (w1, _, w2) = self.unit(at.weights, n_parent, u);
                let (out_w1, out_b1, out_w2) = self.unit_mut(row, n_parent, u);
                let (h, h1, h2) = (st.h[u], st.h1[u], st.h2[u]);
                let mut a_bar = h1 * dot(simd, w2, &scratch.out);
                axpy_lanes(simd, h, &scratch.out, out_w2);
                let mut direct = None;
                if let Some(j) = edge {
                    let w1_dz = st.dz[j] * w1[j];
                    let mut p = S::f64s::splat(simd, 0.0);
                    for (&value, &unit_p) in values.iter().zip(&scratch.unit_p[u * k..][..k]) {
                        p += value * unit_p;
                    }
                    a_bar += h2 * p * w1_dz;
                    axpy_lanes(simd, h1 * w1_dz, &scratch.gathered_dot, out_w2);
                    direct = Some((j, h1 * p * st.dz[j]));
                }
                axpy_lanes(simd, a_bar, &st.z, out_w1);
                if let Some((j, value)) = direct {
                    out_w1[j] += value;
                }
                *out_b1 += a_bar;
            }
        }
    }
}
