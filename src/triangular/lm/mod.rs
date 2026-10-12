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
//! Supported conditioners: one hidden layer with softplus or `gelu_tanh`,
//! optionally on squashed inputs, and the linear location skip. The transformer is a chain of `Contract2`, `TangentSAS` and
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
//! | `z_parents`, `dz` | what the units see of the parents, `c asinh(y / c)` with an input squash `c` and `y` without, and its derivative in `y` |
//! | `a` | the hidden units' inputs, `W1 z_parents + b1` |
//! | `h`, `h1`, `h2` | the activation and its first two derivatives at `a` (`h`, `h'`, `h''`) |
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
mod flow;
pub(crate) mod optimizer;
mod python;
mod selected_inverse;
mod simd;
mod sweeps;
#[cfg(test)]
mod tests;
mod tiles;

use std::sync::{Arc, Mutex};

use anyhow::{bail, Result};
use fearless_simd::{dispatch, prelude::*, Level};
use rayon::prelude::*;

use crate::triangular::pattern::{Offsets, Pattern};
pub(crate) use blocks::GnBlock;
use conditioner::{Conditioner, Params, Shape, UnitState};
pub(crate) use flow::PyTriangularFlow;
pub(crate) use python::PyFisherResiduals;
use selected_inverse::SelectedInverseLayout;
use simd::{f64_width, store};
use sweeps::{TileScratch, VariableScratch};
pub(crate) use tiles::Linearization;
use tiles::{Data, Tiled};

/// Bytes of residual cotangents [`FisherResiduals::sum_pullbacks`] keeps for
/// one block of tiles, but at least [`MIN_BLOCK_TILES`] tiles' worth.
const PULLBACK_BLOCK_BYTES: usize = 512 << 20;
const MIN_BLOCK_TILES: usize = 16;

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
    pullback_buffers: PullbackBuffers,
}

/// [`FisherResiduals::sum_pullbacks`]'s `r_bar` and `z` of one block of
/// tiles, kept between calls so that each GN product doesn't fault in fresh
/// pages. A clone starts without them.
#[derive(Default)]
struct PullbackBuffers(Mutex<Option<(Tiled, Tiled)>>);

impl Clone for PullbackBuffers {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl PullbackBuffers {
    /// The kept buffers if they hold `block_len` tiles, else new ones. A
    /// concurrent call finds none and makes its own.
    fn take(&self, block_len: usize, n_res: usize, n_var: usize, width: usize) -> (Tiled, Tiled) {
        match self.0.lock().expect("never poisoned").take() {
            Some((r_bar, z)) if r_bar.n_tile() == block_len => (r_bar, z),
            _ => (
                Tiled::zeros(block_len, n_res, width),
                Tiled::zeros(block_len, n_var, width),
            ),
        }
    }

    fn put(&self, buffers: (Tiled, Tiled)) {
        *self.0.lock().expect("never poisoned") = Some(buffers);
    }
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
        Ok(Self {
            level: Level::new(),
            width: Self::native_width(),
            graph,
            param_offset,
            conditioner,
            regularization: fisher_regularization.map(f64::sqrt),
            inverse,
            data: None,
            pullback_buffers: PullbackBuffers::default(),
        })
    }

    /// Lanes of this machine's native `f64` vector; [`Self::set_data`] needs
    /// a multiple of this many draws.
    pub(crate) fn native_width() -> usize {
        let level = Level::new();
        dispatch!(level, simd => f64_width(simd))
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

    /// Per parameter, whether it is a weight of the units, `W1` or `W2`:
    /// what the optimizer's MLP ridge penalizes. With all of them zero, each
    /// conditioner is its biases and the skip alone.
    pub(crate) fn unit_weight_mask(&self) -> Vec<bool> {
        self.unit_mask(false)
    }

    /// Per parameter, whether it belongs to the hidden units (`W1`, `b1` or
    /// `W2`): with these frozen at `W2 = 0`, a fit only moves the linear
    /// part of each conditioner, its biases `b2` and the skip.
    pub(crate) fn unit_param_mask(&self) -> Vec<bool> {
        self.unit_mask(true)
    }

    fn unit_mask(&self, with_bias: bool) -> Vec<bool> {
        let mut mask = vec![false; self.n_params()];
        for i in 0..self.n_var() {
            let (start, shape) = (self.params(i).start, self.shape(i));
            for u in 0..shape.n_unit {
                for k in shape.w1(u).chain(shape.w2(u)) {
                    mask[start + k] = true;
                }
                if with_bias {
                    mask[start + shape.b1(u)] = true;
                }
            }
        }
        mask
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
        let mut lin = Linearization::zeros(theta, data.clone(), self.n_var(), self.n_edge());
        let r =
            dispatch!(self.level, simd => self.residuals_with(simd, theta, data, Some(&mut lin)));
        Ok((r, lin))
    }

    /// The residuals at `theta` alone, without the memory of a
    /// linearization.
    pub(crate) fn residual_values(&self, theta: &[f64]) -> Result<Vec<f64>> {
        self.check_params(theta, "theta")?;
        let Some(data) = &self.data else {
            bail!("no data: call `set_data` first");
        };
        Ok(dispatch!(self.level, simd => self.residuals_with(simd, theta, data, None)))
    }

    fn residuals_with<S: Simd>(
        &self,
        simd: S,
        theta: &[f64],
        data: &Data,
        lin: Option<&mut Linearization>,
    ) -> Vec<f64> {
        let mut out = Tiled::zeros(data.n_tile(), self.n_residuals(), self.width);
        let lin_tiles: Vec<_> = match lin {
            Some(lin) => lin.tiles_mut().into_iter().map(Some).collect(),
            None => (0..data.n_tile()).map(|_| None).collect(),
        };
        let tiles: Vec<_> = lin_tiles.into_iter().zip(out.tiles_mut()).collect();
        tiles.into_par_iter().enumerate().for_each_init(
            || TileScratch::new(simd, self),
            |s, (tile, (lin, out))| self.primal_tile(simd, theta, data, tile, lin, out, s),
        );
        out.to_rows()
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
        self.sum_pullbacks(simd, lin, |tile, s| {
            self.load_tile(simd, lin, tile, s);
            r_bar.load(simd, tile, &mut s.residual);
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
        self.sum_pullbacks(simd, lin, |tile, s| {
            self.load_tile(simd, lin, tile, s);
            self.pushforward_tile(simd, lin, v, s);
        })
    }

    /// `J^T r_bar` summed over the tiles, with `cotangent(tile, s)` loading
    /// tile `tile` into `s` and leaving its `r_bar` in `s.residual`.
    ///
    /// In blocks of tiles: first, in parallel over the block's tiles, the
    /// triangular solve of each ([`Self::pullback_solve`]), then, in parallel
    /// over the variables, each variable's local pullback over the block's
    /// tiles in order, its lanes summed straight into its slice of the
    /// gradient ([`Self::pullback_variable`]). So a variable's parameters stay
    /// in cache across the tiles, the gradient is the only parameter-sized
    /// buffer, and the order of every addition depends only on the problem's
    /// sizes, not on the threads.
    fn sum_pullbacks<S: Simd>(
        &self,
        simd: S,
        lin: &Linearization,
        cotangent: impl Fn(usize, &mut TileScratch<S>) + Sync,
    ) -> Vec<f64> {
        let (n_var, n_res, n_tile) = (self.n_var(), self.n_residuals(), lin.data.n_tile());
        let tile_bytes = (n_res + n_var) * self.width * std::mem::size_of::<f64>();
        let block_len = (PULLBACK_BLOCK_BYTES / tile_bytes)
            .max(MIN_BLOCK_TILES)
            .min(n_tile);
        // Every tile a block uses is overwritten, so old values don't matter.
        let (mut r_bar, mut z) = self
            .pullback_buffers
            .take(block_len, n_res, n_var, self.width);
        let scratch = PerThread::new();

        let mut grad = vec![0.0; self.n_params()];
        let mut grad_of_var = Vec::with_capacity(n_var);
        let mut rest = grad.as_mut_slice();
        for i in 0..n_var {
            let (head, tail) = rest.split_at_mut(self.param_offset.row_len(i));
            grad_of_var.push(head);
            rest = tail;
        }

        for start in (0..n_tile).step_by(block_len) {
            let tiles = start..(start + block_len).min(n_tile);
            r_bar
                .tiles_mut()
                .into_par_iter()
                .zip(z.tiles_mut())
                .take(tiles.len())
                .enumerate()
                .for_each(|(k, (r_bar, z))| {
                    scratch.with(
                        || TileScratch::new(simd, self),
                        |s| {
                            cotangent(start + k, s);
                            self.pullback_solve(simd, lin, s, r_bar, z);
                        },
                    );
                });
            grad_of_var.par_iter_mut().enumerate().for_each_init(
                || VariableScratch::new(simd, self),
                |ws, (i, out)| {
                    self.pullback_variable(simd, lin, &r_bar, &z, tiles.clone(), i, ws, out)
                },
            );
        }
        drop(grad_of_var);
        self.pullback_buffers.put((r_bar, z));
        grad
    }
}

/// One value per thread of the current rayon pool, made on its first use
/// there, so that big scratch is allocated once per thread rather than per
/// task.
struct PerThread<T>(Vec<Mutex<Option<T>>>);

impl<T> PerThread<T> {
    fn new() -> Self {
        Self(
            (0..rayon::current_num_threads())
                .map(|_| Mutex::new(None))
                .collect(),
        )
    }

    /// `f` of this thread's value. `f` must not wait on other rayon tasks,
    /// which could run on this thread and ask for the same value.
    fn with<R>(&self, init: impl FnOnce() -> T, f: impl FnOnce(&mut T) -> R) -> R {
        let slot = rayon::current_thread_index().unwrap_or(0) % self.0.len();
        let mut value = self.0[slot].lock().expect("a task panicked");
        f(value.get_or_insert_with(init))
    }
}
