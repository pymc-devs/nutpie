//! Residuals of the Levenberg-Marquardt fit of a `SparseTriangularMap`, and
//! the derivatives `lmopt` needs from them: the pushforward `J v`, the
//! pullback `J^T r`, and the exact Gauss-Newton blocks. The derivation and the
//! notation are in `notes/lm_derivatives.md`.
//!
//! Only the conditioners are fitted. The map's inputs `y` and the gradients
//! `g` are fixed data, already mapped through the frozen affine and
//! permutation, so the residual of one draw is
//!
//! ```text
//! r = (x + w) / sqrt(n_draw),   J^T w = g - grad_y log_det,
//! ```
//!
//! followed, with a Fisher regularization `rho`, by one parent score per edge,
//! `sqrt(rho) (L[i,j] - x_i A[i,j]) / sqrt(n_draw)`.
//!
//! Supported conditioners: one hidden layer with softplus, and the linear
//! location skip. The transformer is a chain of `Contract2`, `TangentSAS` and
//! `PositiveAffine` layers, evaluated in the density direction.
//!
//! Parameters are laid out per variable, with no bucketing or padding.
//! Variable `i`'s slice holds, for each hidden unit `u`, the unit's input
//! weights `W1[u, :]`, its bias `b1[u]` and its output weights `W2[:, u]`,
//! followed by `b2` and the skip weights `s`. Every kernel walks the
//! conditioner unit by unit, so this keeps each unit's weights contiguous.
//!
//! Draws are processed in tiles of one native `f64` SIMD vector each
//! (`fearless_simd`): every draw walks the same variables, parents and
//! weights, so a tile runs the per-draw algorithm once with each value
//! widened to a vector. `n_draw` must be a multiple of the vector width.
//!
//! The [`Linearization`] at `theta` keeps only what the triangular solves
//! need across variables (`A`, `delta`, `x`, `w`), and the data. Everything
//! local to a variable -- the hidden units and the transformer's derivatives
//! -- is recomputed by each operator, with one jet pass of the transformer per
//! variable and tile.
//!
//! # Notation
//!
//! The names follow `notes/lm_derivatives.md`; the code's spelling is on the
//! left. Everything is for one variable `i` and one draw (one lane of a
//! tile), with parents `P(i)`, `n_p = |P(i)|`.
//!
//! | name | meaning |
//! |---|---|
//! | `y`, `g` | the map's inputs and the log density's gradient there (data) |
//! | `y_parents` | `y[P(i)]` |
//! | `W1`, `b1`, `W2`, `b2`, `s` | variable `i`'s conditioner parameters, `theta_i`; `s` is the location skip ([`conditioner::Params`]) |
//! | `a` | the hidden units' inputs, `W1 y_parents + b1` |
//! | `h`, `h1`, `h2` | softplus and its first two derivatives at `a` (`h`, `h'`, `h''`) |
//! | `pi` | the transformer parameters, `W2 h + b2 + e_loc s . y_parents`, `n_par` of them |
//! | `loc` | the index of the transformer parameter the skip adds to |
//! | `T`, `Lambda` | the transformer `x_i = T(y_i; pi)` and `Lambda = log dT/dy_i` |
//! | `x`, `delta`, `mu` | `T`, `dT/dy_i` and `dLambda/dy_i` at `y_i` |
//! | `t`, `l` | `dT/dpi` and `dLambda/dpi`, `n_par` each |
//! | `n_z`, `z` in jets | `1 + n_par`, the transformer's inputs `(y_i, pi)` |
//! | `C` | `dpi/dy_parents`, `(n_par, n_p)` |
//! | `A`, `L` (`edge_a`, `edge_l`) | per edge `(i, j)`: `dx_i/dy_j = t . C[:, j]` and `dLambda_i/dy_j = l . C[:, j]` |
//! | `w` | the solution of `J^T w = g - grad_y log_det` |
//! | `z` in the pullback | `J^{-1} r_bar` |
//! | `p_t`, `p_l` | per unit `u`, `W2[:, u] . t` and `W2[:, u] . l` |
//!
//! A `_t`/`_l` pair is the same quantity for `T` and for `Lambda`, and an
//! `_a`/`_l` pair the same for `A` and for `L`. A `_dot` suffix is a tangent
//! (the derivative along `v` in the pushforward), a `_bar` suffix a
//! cotangent (the pullback's adjoint).

mod blocks;
mod conditioner;
pub(crate) mod optimizer;
mod python;
mod selected_inverse;
mod simd;
mod sweeps;
#[cfg(test)]
mod tests;
mod tiles;

use std::sync::Arc;

use anyhow::{bail, Result};
use fearless_simd::{dispatch, prelude::*, Level};
use rayon::prelude::*;

use crate::triangular::pattern::{Offsets, Pattern};
pub(crate) use blocks::GnBlock;
use conditioner::{Conditioner, Params, ParamsMut, Shape, UnitState};
pub(crate) use python::PyFisherResiduals;
use selected_inverse::SelectedInverseLayout;
use simd::{f64_width, store};
use sweeps::TileScratch;
pub(crate) use tiles::Linearization;
use tiles::{Data, Tiled};

#[derive(Clone)]
pub(crate) struct FisherResiduals {
    level: Level,
    /// Lanes of `level`'s native `f64` vector: the draws in a tile.
    width: usize,
    graph: Pattern,
    /// Each variable's slice of the flat parameter vector.
    pub(crate) param_offset: Offsets,
    conditioner: Conditioner,
    /// `sqrt(fisher_regularization)`, if regularized.
    regularization: Option<f64>,
    inverse: SelectedInverseLayout,
    /// Shared with every [`Linearization`] made from it.
    data: Option<Arc<Data>>,
}

impl FisherResiduals {
    fn new(
        graph: Pattern,
        conditioner: Conditioner,
        fisher_regularization: Option<f64>,
    ) -> Result<Self> {
        if let Some(rho) = fisher_regularization {
            if rho.is_nan() || rho < 0.0 {
                bail!("fisher_regularization must be non-negative, got {rho}");
            }
        }
        let param_offset = Offsets::from_lens(
            (0..graph.n_var()).map(|i| conditioner.shape(graph.n_parent(i)).size()),
        );
        let inverse = SelectedInverseLayout::new(&graph)?;
        let level = Level::new();
        Ok(Self {
            level,
            width: dispatch!(level, simd => f64_width(simd)),
            graph,
            param_offset,
            conditioner,
            regularization: fisher_regularization.map(f64::sqrt),
            inverse,
            data: None,
        })
    }

    pub(crate) fn n_var(&self) -> usize {
        self.graph.n_var()
    }

    fn n_edge(&self) -> usize {
        self.graph.n_edge()
    }

    pub(crate) fn n_params(&self) -> usize {
        self.param_offset.total()
    }

    pub(crate) fn n_residuals(&self) -> usize {
        self.n_var()
            + if self.regularization.is_some() {
                self.n_edge()
            } else {
                0
            }
    }

    /// The number of draws, zero before [`Self::set_data`].
    pub(crate) fn n_draw(&self) -> usize {
        self.data.as_ref().map_or(0, |data| data.n_draw)
    }

    fn unit_state<S: Simd>(&self, simd: S) -> UnitState<S> {
        self.conditioner.unit_state(simd, self.graph.max_parent())
    }

    fn shape(&self, i: usize) -> Shape {
        self.conditioner.shape(self.graph.n_parent(i))
    }

    fn params(&self, i: usize) -> std::ops::Range<usize> {
        self.param_offset.range(i)
    }

    /// Variable `i`'s slice of `values`.
    fn params_of<'a, T: Copy>(&self, values: &'a [T], i: usize) -> Params<'a, T> {
        Params::new(&values[self.params(i)], self.shape(i))
    }

    fn params_of_mut<'a, T>(&self, values: &'a mut [T], i: usize) -> ParamsMut<'a, T> {
        ParamsMut::new(&mut values[self.params(i)], self.shape(i))
    }

    pub(crate) fn set_data(&mut self, y: Vec<f64>, g: Vec<f64>) -> Result<()> {
        let n_var = self.n_var();
        if n_var == 0 || !y.len().is_multiple_of(n_var) || y.len() != g.len() {
            bail!("y and g must both have shape (n_draw, {n_var})");
        }
        let n_draw = y.len() / n_var;
        if n_draw == 0 {
            bail!("need at least one draw");
        }
        if !n_draw.is_multiple_of(self.width) {
            bail!(
                "the number of draws ({n_draw}) must be a multiple of the SIMD width \
                 ({}); a multiple of 8 works on every machine",
                self.width
            );
        }
        self.data = Some(Arc::new(Data {
            n_draw,
            y: Tiled::from_rows(&y, n_draw, self.width),
            g: Tiled::from_rows(&g, n_draw, self.width),
        }));
        Ok(())
    }

    #[inline(always)]
    fn gather_parents<T: Copy>(&self, i: usize, y: &[T], out: &mut Vec<T>) {
        out.clear();
        out.extend(self.graph.parents(i).iter().map(|&p| y[p]));
    }

    // ------------------------------------------------------------ operators

    fn check_params(&self, values: &[f64], name: &str) -> Result<()> {
        if values.len() != self.n_params() {
            bail!(
                "{name} has length {}, expected {}",
                values.len(),
                self.n_params()
            );
        }
        Ok(())
    }

    /// `(n_draw, n_residuals)` residuals at `theta`, and the linearization
    /// every derivative at `theta` reads.
    pub(crate) fn residuals(&self, theta: &[f64]) -> Result<(Vec<f64>, Linearization)> {
        self.check_params(theta, "theta")?;
        let Some(data) = &self.data else {
            bail!("no data: call `set_data` first");
        };
        Ok(dispatch!(self.level, simd => self.residuals_with(simd, theta, data)))
    }

    fn residuals_with<S: Simd>(
        &self,
        simd: S,
        theta: &[f64],
        data: &Arc<Data>,
    ) -> (Vec<f64>, Linearization) {
        let mut lin = Linearization::zeros(theta, data.clone(), self.n_var(), self.n_edge());
        let mut out = Tiled::zeros(data.n_tile(), self.n_residuals(), self.width);
        let tiles: Vec<_> = lin.tiles_mut().into_iter().zip(out.tiles_mut()).collect();
        tiles.into_par_iter().enumerate().for_each_init(
            || TileScratch::new(simd, self),
            |s, (tile, (lin, out))| self.primal_tile(simd, theta, data, tile, lin, out, s),
        );
        (out.to_rows(), lin)
    }

    pub(crate) fn pushforward(&self, lin: &Linearization, v: &[f64]) -> Result<Vec<f64>> {
        self.check_params(v, "v")?;
        Ok(dispatch!(self.level, simd => self.pushforward_with(simd, lin, v)))
    }

    fn pushforward_with<S: Simd>(&self, simd: S, lin: &Linearization, v: &[f64]) -> Vec<f64> {
        let mut out = Tiled::zeros(lin.data.n_tile(), self.n_residuals(), self.width);
        out.tiles_mut().into_par_iter().enumerate().for_each_init(
            || TileScratch::new(simd, self),
            |s, (tile, out)| {
                self.load_tile(simd, lin, tile, s);
                self.pushforward_tile(simd, lin, v, s);
                store(simd, &s.residual, out);
            },
        );
        out.to_rows()
    }

    pub(crate) fn pullback(&self, lin: &Linearization, r_bar: &[f64]) -> Result<Vec<f64>> {
        let (n_draw, n_res) = (lin.data.n_draw, self.n_residuals());
        if r_bar.len() != n_draw * n_res {
            bail!("r_bar must have shape ({n_draw}, {n_res})");
        }
        Ok(dispatch!(self.level, simd => self.pullback_with(simd, lin, r_bar)))
    }

    fn pullback_with<S: Simd>(&self, simd: S, lin: &Linearization, r_bar: &[f64]) -> Vec<f64> {
        let r_bar = Tiled::from_rows(r_bar, lin.data.n_draw, self.width);
        self.sum_over_tiles(simd, lin, |tile, grad, s| {
            self.load_tile(simd, lin, tile, s);
            r_bar.load(simd, tile, &mut s.residual);
            self.pullback_tile(simd, lin, s, grad);
        })
    }

    /// `J^T J v`, one tile at a time, without forming `J v` for all draws.
    pub(crate) fn gauss_newton_product(&self, lin: &Linearization, v: &[f64]) -> Result<Vec<f64>> {
        self.check_params(v, "v")?;
        Ok(dispatch!(self.level, simd => self.gauss_newton_product_with(simd, lin, v)))
    }

    fn gauss_newton_product_with<S: Simd>(
        &self,
        simd: S,
        lin: &Linearization,
        v: &[f64],
    ) -> Vec<f64> {
        self.sum_over_tiles(simd, lin, |tile, grad, s| {
            self.load_tile(simd, lin, tile, s);
            self.pushforward_tile(simd, lin, v, s);
            self.pullback_tile(simd, lin, s, grad);
        })
    }

    /// `sum_tiles f(tile)`, each `f` accumulating its tile's lanes into a
    /// per-thread gradient.
    fn sum_over_tiles<S: Simd>(
        &self,
        simd: S,
        lin: &Linearization,
        f: impl Fn(usize, &mut [S::f64s], &mut TileScratch<S>) + Sync,
    ) -> Vec<f64> {
        let n_params = self.n_params();
        let zero = S::f64s::splat(simd, 0.0);
        (0..lin.data.n_tile())
            .into_par_iter()
            .fold(
                || (vec![zero; n_params], TileScratch::new(simd, self)),
                |(mut grad, mut s), tile| {
                    f(tile, &mut grad, &mut s);
                    (grad, s)
                },
            )
            .map(|(grad, _)| grad.iter().map(|g| g.reduce_sum()).collect::<Vec<f64>>())
            .reduce(
                || vec![0.0; n_params],
                |mut a, b| {
                    for (a, b) in a.iter_mut().zip(&b) {
                        *a += b;
                    }
                    a
                },
            )
    }
}
