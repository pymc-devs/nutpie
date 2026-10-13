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

/// Which of the optimizer's ridge strengths a parameter carries, see
/// [`FisherResiduals::ridge_classes`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RidgeClass {
    None,
    /// The location skip `s`.
    Skip,
    /// The scale skip.
    ScaleSkip,
    /// `W1`, or `W2` into a field that is neither the location nor a shape
    /// field.
    Mlp,
    /// `W2` into the location.
    MlpLoc,
    /// `b2` of a shape field.
    Shape,
    /// `W2` into a shape field.
    MlpShape,
}

/// The ridge strengths of the optimizer's objective, per [`RidgeClass`]:
/// `mlp` on the units' weights, `skip` on the location skip, `scale_skip` on
/// the scale skip (unscaled: whatever the scale field's parameterization, it
/// is already relative), and `shape` on what feeds the shape fields
/// directly, `b2` and the units' `W2`.
///
/// What moves the location is weighed against the conditional's width: a
/// shift `Delta` of the location moves `x_i` by `delta_i Delta`, so the skip
/// and the units' `W2` into the location carry a factor `E[delta_i^2]`, the
/// variable's [`FisherResiduals::loc_scales`]. For a Gaussian conditional
/// that makes the skip's ridge one on the precision's Cholesky factor,
/// `s / sigma_i`.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RidgeStrengths {
    pub(crate) mlp: f64,
    pub(crate) skip: f64,
    pub(crate) scale_skip: f64,
    pub(crate) shape: f64,
}

impl RidgeStrengths {
    pub(crate) fn check(&self) -> Result<()> {
        for (name, value) in [
            ("mlp_ridge", self.mlp),
            ("skip_ridge", self.skip),
            ("scale_skip_ridge", self.scale_skip),
            ("shape_ridge", self.shape),
        ] {
            if !(value.is_finite() && value >= 0.0) {
                bail!("{name} must be finite and non-negative, got {value}");
            }
        }
        Ok(())
    }

    /// The strength on a parameter of class `class` of a variable with loc
    /// scale `loc_scale`.
    pub(crate) fn of(&self, class: RidgeClass, loc_scale: f64) -> f64 {
        match class {
            RidgeClass::None => 0.0,
            RidgeClass::Skip => self.skip * loc_scale,
            RidgeClass::ScaleSkip => self.scale_skip,
            RidgeClass::Mlp => self.mlp,
            RidgeClass::MlpLoc => self.mlp * loc_scale,
            RidgeClass::Shape => self.shape,
            RidgeClass::MlpShape => self.mlp + self.shape,
        }
    }
}

#[derive(Clone)]
pub(crate) struct FisherResiduals {
    level: Level,
    /// Lanes of `level`'s native `f64` vector: the draws in a tile.
    width: usize,
    graph: Pattern,
    /// Each variable's slice of the flat parameter vector.
    pub(crate) param_offset: Offsets,
    conditioner: Conditioner,
    inverse: SelectedInverseLayout,
    /// Shared with every [`Linearization`] made from it.
    data: Option<Arc<Data>>,
}

impl FisherResiduals {
    fn new(graph: Pattern, conditioner: Conditioner) -> Result<Self> {
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
            inverse,
            data: None,
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
    }

    /// Per parameter, whether it makes a conditioner or its transformer
    /// nonlinear: the hidden units (`W1`, `b1`, `W2`), the biases `b2` of
    /// the shape fields and the scale skip. With these frozen at `W2 = 0`, the
    /// shape fields at their affine start and the scale skip at zero, a fit
    /// only moves the Gaussian part of each conditional: the location skip
    /// and the affine fields' `b2`.
    pub(crate) fn nonlinear_param_mask(&self) -> Vec<bool> {
        let shape_param = self.conditioner.shape_param_mask();
        let mut mask = vec![false; self.n_params()];
        for i in 0..self.n_var() {
            let (start, shape) = (self.params(i).start, self.shape(i));
            for u in 0..shape.n_unit {
                for k in shape.w1(u).chain(shape.w2(u)) {
                    mask[start + k] = true;
                }
                mask[start + shape.b1(u)] = true;
            }
            for (m, k) in shape.b2().enumerate() {
                mask[start + k] = shape_param[m];
            }
            // A scale that depends on the parents is not linear either.
            if self.conditioner.scale.is_some() {
                for k in shape.skip(1) {
                    mask[start + k] = true;
                }
            }
        }
        mask
    }

    /// Per parameter, which of the optimizer's ridge strengths it carries.
    pub(crate) fn ridge_classes(&self) -> Vec<RidgeClass> {
        let shape_param = self.conditioner.shape_param_mask();
        let loc = self.conditioner.location;
        let mut classes = vec![RidgeClass::None; self.n_params()];
        for i in 0..self.n_var() {
            let (start, shape) = (self.params(i).start, self.shape(i));
            for u in 0..shape.n_unit {
                for k in shape.w1(u) {
                    classes[start + k] = RidgeClass::Mlp;
                }
                for (m, k) in shape.w2(u).enumerate() {
                    classes[start + k] = if m == loc {
                        RidgeClass::MlpLoc
                    } else if shape_param[m] {
                        RidgeClass::MlpShape
                    } else {
                        RidgeClass::Mlp
                    };
                }
            }
            for (m, k) in shape.b2().enumerate() {
                if shape_param[m] {
                    classes[start + k] = RidgeClass::Shape;
                }
            }
            for k in shape.skip(0) {
                classes[start + k] = RidgeClass::Skip;
            }
            if self.conditioner.scale.is_some() {
                for k in shape.skip(1) {
                    classes[start + k] = RidgeClass::ScaleSkip;
                }
            }
        }
        classes
    }

    /// Per variable, `E[delta_i^2]` over the draws at `lin`: the squared
    /// inverse width of its conditional, which scales the ridge on what moves
    /// its location (see [`RidgeStrengths`]).
    pub(crate) fn loc_scales(&self, lin: &Linearization) -> Vec<f64> {
        lin.delta.mean_squares()
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
    /// First, in parallel over the tiles, the triangular solve of each
    /// ([`Self::pullback_solve`]), keeping `r_bar` and `z` for all draws (only
    /// `n_var` values per draw each). Then, in parallel over the variables,
    /// each variable's local pullback over the tiles in order, its lanes
    /// summed straight into its slice of the gradient
    /// ([`Self::pullback_variable`]). So a variable's parameters stay in cache
    /// across the tiles, the gradient is the only parameter-sized buffer, and
    /// the order of every addition depends only on the number of tiles, not
    /// on the threads.
    fn sum_pullbacks<S: Simd>(
        &self,
        simd: S,
        lin: &Linearization,
        cotangent: impl Fn(usize, &mut TileScratch<S>) + Sync,
    ) -> Vec<f64> {
        let (n_var, n_tile) = (self.n_var(), lin.data.n_tile());
        let mut r_bar = Tiled::zeros(n_tile, self.n_residuals(), self.width);
        let mut z = Tiled::zeros(n_tile, n_var, self.width);
        let scratch = PerThread::new();
        r_bar
            .tiles_mut()
            .into_par_iter()
            .zip(z.tiles_mut())
            .enumerate()
            .for_each(|(tile, (r_bar, z))| {
                scratch.with(
                    || TileScratch::new(simd, self),
                    |s| {
                        cotangent(tile, s);
                        self.pullback_solve(simd, lin, s, r_bar, z);
                    },
                );
            });

        let mut grad = vec![0.0; self.n_params()];
        let mut grad_of_var = Vec::with_capacity(n_var);
        let mut rest = grad.as_mut_slice();
        for i in 0..n_var {
            let (head, tail) = rest.split_at_mut(self.param_offset.row_len(i));
            grad_of_var.push(head);
            rest = tail;
        }

        grad_of_var.into_par_iter().enumerate().for_each_init(
            || VariableScratch::new(simd, self),
            |ws, (i, out)| self.pullback_variable(simd, lin, &r_bar, &z, i, ws, out),
        );
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
