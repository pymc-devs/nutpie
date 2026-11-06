//! The linear conditioner: `coefs[k] += w_k . parents` per target
//! coefficient `k`, so `C = sum_k e_k w_k^T`. Its weights are the `w_k`, one
//! after the other in target order, one per parent.

use fearless_simd::prelude::*;

use super::{At, Conditioner, Outputs, SparseCotangents, Stack, StackMut};
use crate::triangular::simd::{axpy, axpy_lanes, dot};

#[derive(Debug, Clone)]
pub(crate) struct Linear {
    outputs: Outputs,
}

impl Linear {
    pub(crate) fn new(outputs: Outputs) -> Self {
        Self { outputs }
    }

    /// Output `j`'s and its coefficient's `(w_j, k)` pairs, `w_j` from
    /// `weights`.
    #[inline(always)]
    fn rows<'a>(
        &'a self,
        weights: &'a [f64],
        n_parent: usize,
    ) -> impl Iterator<Item = (&'a [f64], usize)> + 'a {
        weights
            .chunks_exact(n_parent.max(1))
            .zip(self.outputs.coefs().iter().copied())
    }
}

impl Conditioner for Linear {
    type TileState<S: Simd> = ();
    type Scratch<S: Simd> = ();

    fn outputs(&self) -> &Outputs {
        &self.outputs
    }

    fn n_weights(&self, n_parent: usize) -> usize {
        self.outputs.len() * n_parent
    }

    fn ridge_strengths(&self, n_parent: usize, loc_scale: f64, out: &mut Vec<f64>) {
        for j in 0..self.outputs.len() {
            out.extend(std::iter::repeat_n(
                self.outputs.strength(j, loc_scale),
                n_parent,
            ));
        }
    }

    /// Zero: the coefficients do not depend on the parents.
    fn init_weights<R: rand::Rng + ?Sized>(&self, _: usize, _: &mut R, out: &mut [f64]) {
        out.fill(0.0);
    }

    fn same_structure(&self, _: &Self) -> bool {
        true
    }

    fn carry_weights(&self, from: &Self, n_parent: usize, from_weights: &[f64], out: &mut [f64]) {
        for (j, j_from) in self.outputs.shared(&from.outputs) {
            out[j * n_parent..(j + 1) * n_parent]
                .copy_from_slice(&from_weights[j_from * n_parent..(j_from + 1) * n_parent]);
        }
    }

    type Packed = Vec<f64>;
    type DrawState = ();

    fn pack(&self, _: usize, weights: &[f64]) -> Vec<f64> {
        weights.to_vec()
    }

    fn new_draw_state(&self, _: usize) {}

    #[inline(always)]
    fn forward_draw(
        &self,
        packed: &Vec<f64>,
        parents: &[f64],
        _: &mut (),
        _: bool,
        coefs: &mut [f64],
    ) {
        if parents.is_empty() {
            return;
        }
        for (w, k) in self.rows(packed, parents.len()) {
            coefs[k] += w.iter().zip(parents).map(|(w, p)| w * p).sum::<f64>();
        }
    }

    #[inline(always)]
    fn pullback_parents_draw(
        &self,
        packed: &Vec<f64>,
        _: &mut (),
        coefs_bar: Stack<f64>,
        mut parents_bar: StackMut<f64>,
    ) {
        for i in 0..coefs_bar.k() {
            let (coefs_bar, out) = (coefs_bar.get(i), parents_bar.get(i));
            if out.is_empty() {
                return;
            }
            for (w, k) in self.rows(packed, out.len()) {
                let c = coefs_bar[k];
                for (out, &w) in out.iter_mut().zip(w) {
                    *out += c * w;
                }
            }
        }
    }

    fn new_tile_state<S: Simd>(&self, _: S, _: usize) {}

    fn new_scratch<S: Simd>(&self, _: S, _: usize) {}

    #[inline(always)]
    fn forward_tile<S: Simd>(
        &self,
        simd: S,
        weights: &[f64],
        parents: &[S::f64s],
        _: &mut (),
        coefs: &mut [S::f64s],
    ) {
        if parents.is_empty() {
            return;
        }
        for (w, k) in self.rows(weights, parents.len()) {
            coefs[k] += dot(simd, w, parents);
        }
    }

    #[inline(always)]
    fn pullback_parents_tile<S: Simd>(
        &self,
        simd: S,
        at: &At<Self, S>,
        coefs_bar: Stack<S::f64s>,
        mut parents_bar: StackMut<S::f64s>,
        _: &mut (),
    ) {
        if at.parents.is_empty() {
            return;
        }
        for i in 0..coefs_bar.k() {
            let (coefs_bar, parents_bar) = (coefs_bar.get(i), parents_bar.get(i));
            for (w, k) in self.rows(at.weights, at.parents.len()) {
                axpy(simd, coefs_bar[k], w, parents_bar);
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
        _: &mut (),
    ) {
        if at.parents.is_empty() {
            return;
        }
        for (v, k) in self.rows(v, at.parents.len()) {
            coefs_dot[k] += dot(simd, v, at.parents);
        }
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
        _: &mut (),
    ) {
        let n_parent = at.parents.len();
        if n_parent == 0 {
            return;
        }
        for i in 0..coefs_bar.k() {
            let (coefs_bar, coefs_bar_dot) = (coefs_bar.get(i), coefs_bar_dot.get(i));
            let out = parents_bar_dot.get(i);
            let rows = self
                .rows(at.weights, n_parent)
                .zip(v.chunks_exact(n_parent));
            for ((w, k), v) in rows {
                axpy(simd, coefs_bar_dot[k], w, out);
                axpy(simd, coefs_bar[k], v, out);
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
        _: &mut (),
    ) {
        if at.parents.is_empty() {
            return;
        }
        for i in 0..parents_dot.k() {
            let (parents_dot, coefs_dot) = (parents_dot.get(i), coefs_dot.get(i));
            for (w, k) in self.rows(at.weights, at.parents.len()) {
                coefs_dot[k] += dot(simd, w, parents_dot);
            }
        }
    }

    #[inline(always)]
    fn pullback_weights_tile<S: Simd>(
        &self,
        simd: S,
        at: &At<Self, S>,
        coefs_bar: &[S::f64s],
        pulled_coefs_bar: Stack<S::f64s>,
        parents_bar_bar: Stack<S::f64s>,
        weights_bar: &mut [S::f64s],
        _: &mut (),
    ) {
        let n_parent = at.parents.len();
        if n_parent == 0 {
            return;
        }
        let rows = weights_bar
            .chunks_exact_mut(n_parent)
            .zip(self.outputs.coefs());
        for (out, &k) in rows {
            axpy_lanes(simd, coefs_bar[k], at.parents, out);
            for i in 0..pulled_coefs_bar.k() {
                axpy_lanes(
                    simd,
                    pulled_coefs_bar.get(i)[k],
                    parents_bar_bar.get(i),
                    out,
                );
            }
        }
    }

    #[inline(always)]
    fn add_parent_jacobian_tile<S: Simd>(
        &self,
        simd: S,
        at: &At<Self, S>,
        c: &mut [S::f64s],
        _: &mut (),
    ) {
        let n_parent = at.parents.len();
        if n_parent == 0 {
            return;
        }
        let n_coefs = c.len() / n_parent;
        for (w, k) in self.rows(at.weights, n_parent) {
            for (column, &w) in c.chunks_exact_mut(n_coefs).zip(w) {
                column[k] += S::f64s::splat(simd, w);
            }
        }
    }

    #[inline(always)]
    fn pullback_weights_stack_tile<S: Simd>(
        &self,
        simd: S,
        at: &At<Self, S>,
        cotangents: &SparseCotangents<S>,
        mut weights_bar: StackMut<S::f64s>,
        _: &mut (),
    ) {
        let n_parent = at.parents.len();
        if n_parent == 0 {
            return;
        }
        let pulled = cotangents.pulled_coefs_bar;
        for (r, &edge) in cotangents.edges.iter().enumerate() {
            let coefs_bar = cotangents.coefs_bar.get(r);
            let rows = weights_bar
                .get(r)
                .chunks_exact_mut(n_parent)
                .zip(self.outputs.coefs());
            for (out, &k) in rows {
                axpy_lanes(simd, coefs_bar[k], at.parents, out);
                if let Some(j) = edge {
                    for (i, &value) in cotangents.edge_values.get(r).iter().enumerate() {
                        out[j] += value * pulled.get(i)[k];
                    }
                }
            }
        }
    }
}
