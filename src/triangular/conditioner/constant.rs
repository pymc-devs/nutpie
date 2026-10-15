//! The constant conditioner: one weight per target coefficient, whatever the
//! parents. `C = 0`.

use fearless_simd::prelude::*;

use super::{At, Conditioner, Outputs, SparseCotangents, Stack, StackMut};

#[derive(Debug, Clone)]
pub(crate) struct Constant {
    outputs: Outputs,
}

impl Constant {
    pub(crate) fn new(outputs: Outputs) -> Self {
        Self { outputs }
    }
}

impl Conditioner for Constant {
    type TileState<S: Simd> = ();
    type Scratch<S: Simd> = ();

    fn outputs(&self) -> &Outputs {
        &self.outputs
    }

    fn n_weights(&self, _: usize) -> usize {
        self.outputs.len()
    }

    fn ridge_strengths(&self, _: usize, loc_scale: f64, out: &mut Vec<f64>) {
        out.extend((0..self.outputs.len()).map(|j| self.outputs.strength(j, loc_scale)));
    }

    /// Zero: the transformer's identity.
    fn init_weights<R: rand::Rng + ?Sized>(&self, _: usize, _: &mut R, out: &mut [f64]) {
        out.fill(0.0);
    }

    fn same_structure(&self, _: &Self) -> bool {
        true
    }

    fn carry_weights(&self, from: &Self, _: usize, from_weights: &[f64], out: &mut [f64]) {
        for (j, j_from) in self.outputs.shared(&from.outputs) {
            out[j] = from_weights[j_from];
        }
    }

    fn new_tile_state<S: Simd>(&self, _: S, _: usize) {}

    fn new_scratch<S: Simd>(&self, _: S, _: usize) {}

    #[inline(always)]
    fn forward_tile<S: Simd>(
        &self,
        simd: S,
        weights: &[f64],
        _: &[S::f64s],
        _: &mut (),
        coefs: &mut [S::f64s],
    ) {
        for (&w, &k) in weights.iter().zip(self.outputs.coefs()) {
            coefs[k] += S::f64s::splat(simd, w);
        }
    }

    #[inline(always)]
    fn pullback_parents_tile<S: Simd>(
        &self,
        _: S,
        _: &At<Self, S>,
        _: Stack<S::f64s>,
        _: StackMut<S::f64s>,
        _: &mut (),
    ) {
    }

    #[inline(always)]
    fn pushforward_weights_tile<S: Simd>(
        &self,
        simd: S,
        _: &At<Self, S>,
        v: &[f64],
        coefs_dot: &mut [S::f64s],
        _: &mut (),
    ) {
        for (&v, &k) in v.iter().zip(self.outputs.coefs()) {
            coefs_dot[k] += S::f64s::splat(simd, v);
        }
    }

    #[inline(always)]
    fn pullback_parents_tangent_tile<S: Simd>(
        &self,
        _: S,
        _: &At<Self, S>,
        _: &[f64],
        _: Stack<S::f64s>,
        _: Stack<S::f64s>,
        _: StackMut<S::f64s>,
        _: &mut (),
    ) {
    }

    #[inline(always)]
    fn pushforward_parents_tile<S: Simd>(
        &self,
        _: S,
        _: &At<Self, S>,
        _: Stack<S::f64s>,
        _: StackMut<S::f64s>,
        _: &mut (),
    ) {
    }

    #[inline(always)]
    fn pullback_weights_tile<S: Simd>(
        &self,
        _: S,
        _: &At<Self, S>,
        coefs_bar: &[S::f64s],
        _: Stack<S::f64s>,
        _: Stack<S::f64s>,
        weights_bar: &mut [S::f64s],
        _: &mut (),
    ) {
        for (w, &k) in weights_bar.iter_mut().zip(self.outputs.coefs()) {
            *w += coefs_bar[k];
        }
    }

    #[inline(always)]
    fn add_parent_jacobian_tile<S: Simd>(
        &self,
        _: S,
        _: &At<Self, S>,
        _: &mut [S::f64s],
        _: &mut (),
    ) {
    }

    #[inline(always)]
    fn pullback_weights_stack_tile<S: Simd>(
        &self,
        _: S,
        _: &At<Self, S>,
        cotangents: &SparseCotangents<S>,
        mut weights_bar: StackMut<S::f64s>,
        _: &mut (),
    ) {
        for r in 0..weights_bar.k() {
            let coefs_bar = cotangents.coefs_bar.get(r);
            for (w, &k) in weights_bar.get(r).iter_mut().zip(self.outputs.coefs()) {
                *w += coefs_bar[k];
            }
        }
    }
}
