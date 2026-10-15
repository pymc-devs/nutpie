//! Conditioners: a variable's transformer coefficients as functions of its
//! parents, `coefs = sum_c f_c(parents; weights_c)`, each conditioner with
//! its own slice of the variable's weights and its own targets among the
//! coefficients.
//!
//! A conditioner is *what it computes* (its kind: [`Constant`], [`Linear`],
//! [`Mlp`]) plus *where it goes* (its [`Target`]s). Every operation the LM
//! needs is its forward pass, or a pushforward or pullback of it, with
//! respect to the parents or the weights; with `C = dcoefs/dparents`,
//! `(n_coefs, n_parent)`:
//!
//! - `pullback_parents`: `parents_bar = C^T coefs_bar`;
//! - `pushforward_parents`: `coefs_dot = C parents_dot`;
//! - `pushforward_weights`: `coefs_dot = (dcoefs/dweights) v`;
//! - `pullback_parents_tangent`: the tangent of `parents_bar = C^T coefs_bar`
//!   along `v` in the weights and `coefs_bar_dot`, which depends on the
//!   weights through `C`: the LM's residuals contain first derivatives in
//!   the parents;
//! - `pullback_weights`: the transpose of both weight tangents, with
//!   `parents_bar_bar` the cotangent of `parents_bar`.
//!
//! Stacked arguments ([`Stack`]) hold several (co)tangents at once, e.g. the
//! two for `x` and `Lambda`; every formula holds for each index `i` of a
//! stack. Vectors in coefficient space have the transformer's `n_coefs`
//! entries, whatever the conditioner's targets.
//!
//! Method suffixes as for the transformer: `_tile` for one SIMD tile of
//! draws.
mod conditioners;
pub(crate) mod config;
mod constant;
mod linear;
mod mlp;
#[cfg(test)]
mod tests;

use fearless_simd::prelude::*;

pub(crate) use conditioners::{Conditioners, DrawWorkspace, PackedVariable, Workspace};
pub(crate) use config::{ConditionerConfig, Outputs};
pub(crate) use constant::Constant;
pub(crate) use linear::Linear;
pub(crate) use mlp::{Activation, Mlp};

/// `k` vectors of length `n`, vector `i` at `values[i * stride..][..n]`.
/// `T` is `S::f64s` for a tile.
#[derive(Clone, Copy)]
pub(crate) struct Stack<'a, T> {
    values: &'a [T],
    n: usize,
    stride: usize,
    k: usize,
}

impl<'a, T> Stack<'a, T> {
    /// `values` split into `k` contiguous vectors.
    pub(crate) fn new(values: &'a [T], k: usize) -> Self {
        assert!(k > 0 && values.len().is_multiple_of(k));
        let n = values.len() / k;
        Self {
            values,
            n,
            stride: n,
            k,
        }
    }

    pub(crate) fn k(&self) -> usize {
        self.k
    }

    #[inline(always)]
    pub(crate) fn get(&self, i: usize) -> &'a [T] {
        &self.values[i * self.stride..][..self.n]
    }
}

/// [`Stack`], to write.
pub(crate) struct StackMut<'a, T> {
    values: &'a mut [T],
    n: usize,
    stride: usize,
    k: usize,
}

impl<'a, T> StackMut<'a, T> {
    /// `values` split into `k` contiguous vectors.
    pub(crate) fn new(values: &'a mut [T], k: usize) -> Self {
        assert!(k > 0 && values.len().is_multiple_of(k));
        let n = values.len() / k;
        Self {
            values,
            n,
            stride: n,
            k,
        }
    }

    pub(crate) fn k(&self) -> usize {
        self.k
    }

    #[inline(always)]
    pub(crate) fn get(&mut self, i: usize) -> &mut [T] {
        &mut self.values[i * self.stride..][..self.n]
    }

    /// Entries `range` of every vector.
    pub(crate) fn sub(&mut self, range: std::ops::Range<usize>) -> StackMut<'_, T> {
        assert!(range.end <= self.n);
        let len = (self.k - 1) * self.stride + range.end;
        StackMut {
            values: &mut self.values[range.start..len],
            n: range.len(),
            stride: self.stride,
            k: self.k,
        }
    }
}

/// A conditioner at one variable and tile: its weights, the parents, and its
/// state from `forward_tile`.
pub(crate) struct At<'a, C: Conditioner + ?Sized, S: Simd> {
    pub(crate) weights: &'a [f64],
    pub(crate) parents: &'a [S::f64s],
    pub(crate) state: &'a C::TileState<S>,
}

/// A stack of cotangents for `pullback_weights_stack_tile`: per cotangent
/// `r`, its `coefs_bar` and a `parents_bar_bar` that is zero but on at most
/// one parent, `parents_bar_bar[i] = edge_values[r][i] e_j` with `j =
/// edges[r]`, one value per pulled cotangent.
pub(crate) struct SparseCotangents<'a, S: Simd> {
    pub(crate) coefs_bar: Stack<'a, S::f64s>,
    pub(crate) pulled_coefs_bar: Stack<'a, S::f64s>,
    pub(crate) edges: &'a [Option<usize>],
    pub(crate) edge_values: Stack<'a, S::f64s>,
}

pub(crate) trait Conditioner {
    /// What `forward_tile` records for the other tile methods.
    type TileState<S: Simd>: Send;
    /// Per-thread workspace of the tile methods. Also carries work from one
    /// method to the next: see `pullback_parents_tangent_tile` and
    /// `pullback_weights_tile`.
    type Scratch<S: Simd>: Send;

    fn outputs(&self) -> &Outputs;
    fn n_weights(&self, n_parent: usize) -> usize;
    /// Appends one ridge strength per weight, before the division by
    /// `n_draw`; `loc_scale` scales what feeds the location.
    fn ridge_strengths(&self, n_parent: usize, loc_scale: f64, out: &mut Vec<f64>);
    /// A new conditioner's weights.
    fn init_weights<R: rand::Rng + ?Sized>(&self, n_parent: usize, rng: &mut R, out: &mut [f64]);

    /// Whether `other`'s weights mean the same as this one's, so that
    /// [`Self::carry_weights`] can copy them: the same kind of conditioner
    /// with the same structural settings, whatever their targets.
    fn same_structure(&self, other: &Self) -> bool;
    /// Copies `from`'s weights (`from_weights`, of a conditioner with the
    /// same structure) into `out` where they feed a coefficient both
    /// target, and makes the outputs only this one targets start at zero.
    /// Leaves the rest of `out`, a new conditioner's weights, as it is.
    fn carry_weights(&self, from: &Self, n_parent: usize, from_weights: &[f64], out: &mut [f64]);

    // ---- Sampler: one draw.

    /// A variable's weights in the per-draw kernels' layout; built once per
    /// transform.
    type Packed: Send + Sync;
    /// What `forward_draw` records for `pullback_parents_draw`, and the
    /// kernels' working buffers.
    type DrawState: Send;
    fn pack(&self, n_parent: usize, weights: &[f64]) -> Self::Packed;
    fn new_draw_state(&self, max_parent: usize) -> Self::DrawState;
    /// `coefs += f(parents; weights)`. With `record`, keeps what
    /// `pullback_parents_draw` needs.
    fn forward_draw(
        &self,
        packed: &Self::Packed,
        parents: &[f64],
        state: &mut Self::DrawState,
        record: bool,
        coefs: &mut [f64],
    );
    /// `parents_bar[i] += C^T coefs_bar[i]` for a stack of two (`dy/dcoefs`
    /// and `dlog_det/dcoefs`), after a recording `forward_draw` on `state`.
    fn pullback_parents_draw(
        &self,
        packed: &Self::Packed,
        state: &mut Self::DrawState,
        coefs_bar: Stack<f64>,
        parents_bar: StackMut<f64>,
    );

    // ---- LM: one tile of draws.

    fn new_tile_state<S: Simd>(&self, simd: S, max_parent: usize) -> Self::TileState<S>;
    fn new_scratch<S: Simd>(&self, simd: S, max_parent: usize) -> Self::Scratch<S>;

    /// `coefs += f(parents; weights)`.
    fn forward_tile<S: Simd>(
        &self,
        simd: S,
        weights: &[f64],
        parents: &[S::f64s],
        state: &mut Self::TileState<S>,
        coefs: &mut [S::f64s],
    );

    /// `parents_bar[i] += C^T coefs_bar[i]`.
    fn pullback_parents_tile<S: Simd>(
        &self,
        simd: S,
        at: &At<Self, S>,
        coefs_bar: Stack<S::f64s>,
        parents_bar: StackMut<S::f64s>,
        scratch: &mut Self::Scratch<S>,
    );

    /// `coefs_dot += (dcoefs/dweights) v`.
    fn pushforward_weights_tile<S: Simd>(
        &self,
        simd: S,
        at: &At<Self, S>,
        v: &[f64],
        coefs_dot: &mut [S::f64s],
        scratch: &mut Self::Scratch<S>,
    );

    /// The tangent of `parents_bar[i] = C^T coefs_bar[i]` along `v` in the
    /// weights and `coefs_bar_dot`: `parents_bar_dot[i] += C^T
    /// coefs_bar_dot[i] + (d C^T / dweights v) coefs_bar[i]`. After
    /// `pushforward_weights_tile` along the same `v`.
    #[allow(clippy::too_many_arguments)]
    fn pullback_parents_tangent_tile<S: Simd>(
        &self,
        simd: S,
        at: &At<Self, S>,
        v: &[f64],
        coefs_bar: Stack<S::f64s>,
        coefs_bar_dot: Stack<S::f64s>,
        parents_bar_dot: StackMut<S::f64s>,
        scratch: &mut Self::Scratch<S>,
    );

    /// `coefs_dot[i] += C parents_dot[i]`. Transposing `C^T c` in `c` gives
    /// `C` again, so this is also the pullback of `parents_bar_bar` to the
    /// pulled cotangents.
    fn pushforward_parents_tile<S: Simd>(
        &self,
        simd: S,
        at: &At<Self, S>,
        parents_dot: Stack<S::f64s>,
        coefs_dot: StackMut<S::f64s>,
        scratch: &mut Self::Scratch<S>,
    );

    /// The transpose of the weight tangents: `weights_bar +=
    /// (dcoefs/dweights)^T coefs_bar + sum_i (d (C^T
    /// pulled_coefs_bar[i]) / dweights)^T parents_bar_bar[i]`. After
    /// `pushforward_parents_tile` of the same `parents_bar_bar`, whose work
    /// it reuses.
    #[allow(clippy::too_many_arguments)]
    fn pullback_weights_tile<S: Simd>(
        &self,
        simd: S,
        at: &At<Self, S>,
        coefs_bar: &[S::f64s],
        pulled_coefs_bar: Stack<S::f64s>,
        parents_bar_bar: Stack<S::f64s>,
        weights_bar: &mut [S::f64s],
        scratch: &mut Self::Scratch<S>,
    );

    /// `c += C`, column-major `(n_coefs, n_parent)`. Default: one
    /// `pushforward_parents_tile` per parent; allocates.
    fn add_parent_jacobian_tile<S: Simd>(
        &self,
        simd: S,
        at: &At<Self, S>,
        c: &mut [S::f64s],
        scratch: &mut Self::Scratch<S>,
    ) {
        default_parent_jacobian(self, simd, at, c, scratch);
    }

    /// `weights_bar[r] +=` `pullback_weights_tile` of cotangent `r`.
    /// Default: one `pushforward_parents_tile` and `pullback_weights_tile`
    /// per cotangent; allocates.
    fn pullback_weights_stack_tile<S: Simd>(
        &self,
        simd: S,
        at: &At<Self, S>,
        cotangents: &SparseCotangents<S>,
        weights_bar: StackMut<S::f64s>,
        scratch: &mut Self::Scratch<S>,
    ) {
        default_pullback_weights_stack(self, simd, at, cotangents, weights_bar, scratch);
    }
}

/// The default [`Conditioner::add_parent_jacobian_tile`].
pub(crate) fn default_parent_jacobian<C: Conditioner + ?Sized, S: Simd>(
    conditioner: &C,
    simd: S,
    at: &At<C, S>,
    c: &mut [S::f64s],
    scratch: &mut C::Scratch<S>,
) {
    let n_parent = at.parents.len();
    if n_parent == 0 {
        return;
    }
    let n_coefs = c.len() / n_parent;
    let zero = S::f64s::splat(simd, 0.0);
    let mut one_hot = vec![zero; n_parent];
    for (j, column) in c.chunks_exact_mut(n_coefs).enumerate() {
        one_hot[j] = S::f64s::splat(simd, 1.0);
        conditioner.pushforward_parents_tile(
            simd,
            at,
            Stack::new(&one_hot, 1),
            StackMut::new(column, 1),
            scratch,
        );
        one_hot[j] = zero;
    }
}

/// The default [`Conditioner::pullback_weights_stack_tile`].
pub(crate) fn default_pullback_weights_stack<C: Conditioner + ?Sized, S: Simd>(
    conditioner: &C,
    simd: S,
    at: &At<C, S>,
    cotangents: &SparseCotangents<S>,
    mut weights_bar: StackMut<S::f64s>,
    scratch: &mut C::Scratch<S>,
) {
    let n_parent = at.parents.len();
    let pulled = cotangents.pulled_coefs_bar;
    let k = pulled.k();
    let n_coefs = pulled.get(0).len();
    let zero = S::f64s::splat(simd, 0.0);
    let mut parents_bar_bar = vec![zero; k * n_parent];
    let mut discard = vec![zero; k * n_coefs];
    for (r, &edge) in cotangents.edges.iter().enumerate() {
        parents_bar_bar.fill(zero);
        if let Some(j) = edge {
            for (i, &value) in cotangents.edge_values.get(r).iter().enumerate() {
                parents_bar_bar[i * n_parent + j] = value;
            }
        }
        let parents_bar_bar = Stack::new(&parents_bar_bar, k);
        conditioner.pushforward_parents_tile(
            simd,
            at,
            parents_bar_bar,
            StackMut::new(&mut discard, k),
            scratch,
        );
        conditioner.pullback_weights_tile(
            simd,
            at,
            cotangents.coefs_bar.get(r),
            pulled,
            parents_bar_bar,
            weights_bar.get(r),
            scratch,
        );
    }
}
