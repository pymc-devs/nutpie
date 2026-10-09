//! The local kernels: one variable's conditioner and transformer, and their
//! tangents and cotangents, for one tile of draws. For the names, see the
//! notation in [`super`].

use anyhow::{bail, Result};
use fearless_simd::prelude::*;
use fearless_simd_macros::simd;

use super::simd::{axpy, axpy_lanes, dot, dot_lanes};
use crate::simd_math;
use crate::triangular::jet::{packed_len, JetArena};
use crate::triangular::layers::jet::{transformer, transformer_hessian};
use crate::triangular::layers::{Layer, LayerSpec};

/// The hidden units' activation, by its `nutpie.triangular_layout` name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Activation {
    Softplus,
    /// `jax.nn.gelu`'s default, the tanh approximation.
    GeluTanh,
}

impl Activation {
    pub(super) fn from_name(name: &str) -> Result<Self> {
        match name {
            "softplus" => Ok(Self::Softplus),
            "gelu_tanh" => Ok(Self::GeluTanh),
            other => bail!("unsupported activation {other:?}, expected 'softplus' or 'gelu_tanh'"),
        }
    }
}

/// The architecture every variable's conditioner and transformer share: one
/// hidden layer of `n_unit` units (`activation`), `n_par` transformer parameters
/// with the location skip into parameter `location` (`loc`), and the
/// transformer layers.
///
/// With `squash = Some(c)`, the units see `z = c asinh(y / c)` of each parent
/// instead of `y`, while the skip stays linear in `y`. The units' part of
/// every edge derivative then carries `z'(y_j)` (`dz`).
#[derive(Debug, Clone)]
pub(super) struct Conditioner {
    pub(super) n_unit: usize,
    pub(super) n_par: usize,
    pub(super) location: usize,
    pub(super) layers: Vec<Layer>,
    pub(super) squash: Option<f64>,
    pub(super) activation: Activation,
}

/// Where one variable's conditioner parameters sit in its slice.
#[derive(Clone, Copy)]
pub(super) struct Shape {
    pub(super) n_parent: usize,
    pub(super) n_unit: usize,
    pub(super) n_par: usize,
}

impl Shape {
    #[inline(always)]
    fn stride(&self) -> usize {
        self.n_parent + 1 + self.n_par
    }

    pub(super) fn size(&self) -> usize {
        self.n_unit * self.stride() + self.n_par + self.n_parent
    }

    #[inline(always)]
    pub(super) fn w1(&self, u: usize) -> std::ops::Range<usize> {
        let start = u * self.stride();
        start..start + self.n_parent
    }

    #[inline(always)]
    pub(super) fn b1(&self, u: usize) -> usize {
        u * self.stride() + self.n_parent
    }

    #[inline(always)]
    pub(super) fn w2(&self, u: usize) -> std::ops::Range<usize> {
        let start = u * self.stride() + self.n_parent + 1;
        start..start + self.n_par
    }

    #[inline(always)]
    fn b2(&self) -> std::ops::Range<usize> {
        let start = self.n_unit * self.stride();
        start..start + self.n_par
    }

    #[inline(always)]
    fn skip(&self) -> std::ops::Range<usize> {
        let start = self.n_unit * self.stride() + self.n_par;
        start..start + self.n_parent
    }
}

/// One variable's slice of a parameter-shaped vector (`theta`, a tangent or
/// a gradient), read by role.
#[derive(Clone, Copy)]
pub(super) struct Params<'a, T> {
    values: &'a [T],
    pub(super) shape: Shape,
}

impl<'a, T: Copy> Params<'a, T> {
    pub(super) fn new(values: &'a [T], shape: Shape) -> Self {
        debug_assert_eq!(values.len(), shape.size());
        Self { values, shape }
    }

    /// Unit `u`'s input weights, one per parent.
    #[inline(always)]
    pub(super) fn w1(&self, u: usize) -> &'a [T] {
        &self.values[self.shape.w1(u)]
    }

    #[inline(always)]
    pub(super) fn b1(&self, u: usize) -> T {
        self.values[self.shape.b1(u)]
    }

    /// Unit `u`'s output weights, one per transformer parameter.
    #[inline(always)]
    pub(super) fn w2(&self, u: usize) -> &'a [T] {
        &self.values[self.shape.w2(u)]
    }

    #[inline(always)]
    pub(super) fn b2(&self) -> &'a [T] {
        &self.values[self.shape.b2()]
    }

    /// The location skip's weights, one per parent.
    #[inline(always)]
    pub(super) fn skip(&self) -> &'a [T] {
        &self.values[self.shape.skip()]
    }
}

/// [`Params`], to write.
pub(super) struct ParamsMut<'a, T> {
    values: &'a mut [T],
    shape: Shape,
}

impl<'a, T> ParamsMut<'a, T> {
    pub(super) fn new(values: &'a mut [T], shape: Shape) -> Self {
        debug_assert_eq!(values.len(), shape.size());
        Self { values, shape }
    }

    #[inline(always)]
    pub(super) fn w1(&mut self, u: usize) -> &mut [T] {
        &mut self.values[self.shape.w1(u)]
    }

    #[inline(always)]
    pub(super) fn b1(&mut self, u: usize) -> &mut T {
        &mut self.values[self.shape.b1(u)]
    }

    #[inline(always)]
    pub(super) fn w2(&mut self, u: usize) -> &mut [T] {
        &mut self.values[self.shape.w2(u)]
    }

    #[inline(always)]
    pub(super) fn b2(&mut self) -> &mut [T] {
        &mut self.values[self.shape.b2()]
    }

    #[inline(always)]
    pub(super) fn skip(&mut self) -> &mut [T] {
        &mut self.values[self.shape.skip()]
    }
}

/// The transformer's gradients in `z = (y_i, pi)` and its full Hessians as
/// packed lower triangles (see [`crate::triangular::jet`]), of `T` and of
/// `Lambda`.
pub(super) struct DenseCurvature<S: Simd> {
    /// `(delta, t)` and `(mu, l)`.
    pub(super) grad_t: Vec<S::f64s>,
    pub(super) grad_l: Vec<S::f64s>,
    /// `d^2 T / dz^2` and `d^2 Lambda / dz^2`.
    pub(super) hess_t: Vec<S::f64s>,
    pub(super) hess_l: Vec<S::f64s>,
}

impl<S: Simd> DenseCurvature<S> {
    pub(super) fn new(simd: S, n_par: usize) -> Self {
        let zero = S::f64s::splat(simd, 0.0);
        let n_z = 1 + n_par;
        Self {
            grad_t: vec![zero; n_z],
            grad_l: vec![zero; n_z],
            hess_t: vec![zero; packed_len(n_z)],
            hess_l: vec![zero; packed_len(n_z)],
        }
    }
}

/// A cotangent on one variable's local outputs `(x_i, delta_i, mu_i, A[i, :],
/// L[i, :])`: `x_bar`, `delta_bar`, `mu_bar`, `A_bar[i, :]` and
/// `L_bar[i, :]`, the last two one per parent.
pub(super) struct Cotangent<'a, S: Simd> {
    pub(super) x: S::f64s,
    pub(super) delta: S::f64s,
    pub(super) mu: S::f64s,
    pub(super) edge_a: &'a [S::f64s],
    pub(super) edge_l: &'a [S::f64s],
}

/// One variable's conditioner at its parents, see [`Conditioner::evaluate`].
pub(super) struct UnitState<S: Simd> {
    /// The parents' values, filled before [`Conditioner::evaluate`].
    pub(super) y_parents: Vec<S::f64s>,
    /// What the units see of them, `z`, and `dz = z'(y)`; `y` and one
    /// without a squash.
    pub(super) z_parents: Vec<S::f64s>,
    pub(super) dz: Vec<S::f64s>,
    /// The activation and its first two derivatives at the hidden units.
    pub(super) h: Vec<S::f64s>,
    pub(super) h1: Vec<S::f64s>,
    pub(super) h2: Vec<S::f64s>,
    /// The transformer parameters.
    pub(super) pi: Vec<S::f64s>,
}

/// Per-thread work space of the local kernels.
pub(super) struct Scratch<S: Simd> {
    /// Per unit: `a_dot` in the pushforward; `zeta_a = W1[u, :] . A_bar_z`
    /// and `zeta_l = W1[u, :] . L_bar_z` in the pullback.
    unit_a: Vec<S::f64s>,
    unit_b: Vec<S::f64s>,
    /// Per parent, in the pullback: `A_bar_z = dz * A_bar[i, :]` and
    /// `L_bar_z = dz * L_bar[i, :]`, the edge cotangents the units see.
    edge_za: Vec<S::f64s>,
    edge_zl: Vec<S::f64s>,
    pi_dot: Vec<S::f64s>,
    /// The cotangents of `t` and `l` from the edges' `A_bar` and `L_bar`.
    t_bar: Vec<S::f64s>,
    l_bar: Vec<S::f64s>,
    pi_bar: Vec<S::f64s>,
    pub(super) jets: JetArena<S>,
    /// Up to two jet directions in `z = (y_i, pi)`, `n_z` entries each.
    dirs: Vec<S::f64s>,
    /// `(delta, t)` and `(mu, l)`, and a pullback's Hessian products along
    /// `dirs` (see [`Conditioner::curvature`]).
    grad_t: Vec<S::f64s>,
    grad_l: Vec<S::f64s>,
    prod_t: Vec<S::f64s>,
    prod_l: Vec<S::f64s>,
}

impl Conditioner {
    pub(super) fn new(
        n_unit: usize,
        n_par: usize,
        location: usize,
        specs: Vec<LayerSpec>,
        squash: Option<f64>,
        activation: Activation,
    ) -> Result<Self> {
        if location >= n_par {
            bail!("location index {location} out of range for {n_par} parameters");
        }
        if let Some(c) = squash {
            if !(c.is_finite() && c > 0.0) {
                bail!("input_squash must be positive and finite, got {c}");
            }
        }
        for param in specs.iter().flat_map(LayerSpec::params) {
            if param.index >= n_par {
                bail!("transformer parameter index {} out of range", param.index);
            }
        }
        let layers = specs
            .into_iter()
            .map(Layer::new)
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            n_unit,
            n_par,
            location,
            layers,
            squash,
            activation,
        })
    }

    /// Where the parameters of a variable with `n_parent` parents sit in its
    /// slice.
    pub(super) fn shape(&self, n_parent: usize) -> Shape {
        Shape {
            n_parent,
            n_unit: self.n_unit,
            n_par: self.n_par,
        }
    }

    pub(super) fn unit_state<S: Simd>(&self, simd: S, max_parent: usize) -> UnitState<S> {
        let zero = S::f64s::splat(simd, 0.0);
        UnitState {
            y_parents: Vec::with_capacity(max_parent),
            z_parents: Vec::with_capacity(max_parent),
            dz: Vec::with_capacity(max_parent),
            h: vec![zero; self.n_unit],
            h1: vec![zero; self.n_unit],
            h2: vec![zero; self.n_unit],
            pi: vec![zero; self.n_par],
        }
    }

    pub(super) fn scratch<S: Simd>(&self, simd: S) -> Scratch<S> {
        let zero = S::f64s::splat(simd, 0.0);
        let (n_unit, n_par, n_z) = (self.n_unit, self.n_par, 1 + self.n_par);
        Scratch {
            unit_a: vec![zero; n_unit],
            unit_b: vec![zero; n_unit],
            edge_za: Vec::new(),
            edge_zl: Vec::new(),
            pi_dot: vec![zero; n_par],
            t_bar: vec![zero; n_par],
            l_bar: vec![zero; n_par],
            pi_bar: vec![zero; n_par],
            jets: JetArena::new(simd),
            dirs: vec![zero; 2 * n_z],
            grad_t: vec![zero; n_z],
            grad_l: vec![zero; n_z],
            prod_t: vec![zero; n_z],
            prod_l: vec![zero; n_z],
        }
    }

    /// One variable's conditioner at its parents, already in
    /// `units.y_parents`: the units' inputs `z` and `dz`, the hidden units
    /// `h, h', h''` and the transformer parameters `pi`. Every local kernel
    /// reads its result.
    #[simd]
    pub(super) fn evaluate<S: Simd>(&self, simd: S, theta: Params<f64>, units: &mut UnitState<S>) {
        units.z_parents.clear();
        units.dz.clear();
        match self.squash {
            None => {
                units.z_parents.extend_from_slice(&units.y_parents);
                units
                    .dz
                    .resize(units.y_parents.len(), S::f64s::splat(simd, 1.0));
            }
            Some(c) => {
                let one = S::f64s::splat(simd, 1.0);
                for &y in &units.y_parents {
                    let v = y * (1.0 / c);
                    units.z_parents.push(simd_math::asinh(simd, v) * c);
                    units.dz.push(one / simd_math::hypot_one(simd, v));
                }
            }
        }
        for (pi, &b) in units.pi.iter_mut().zip(theta.b2()) {
            *pi = S::f64s::splat(simd, b);
        }
        for u in 0..self.n_unit {
            let a = dot(simd, theta.w1(u), &units.z_parents) + theta.b1(u);
            let (h, h1, h2) = match self.activation {
                Activation::Softplus => simd_math::softplus(simd, a),
                Activation::GeluTanh => simd_math::gelu_tanh(simd, a),
            };
            units.h[u] = h;
            units.h1[u] = h1;
            units.h2[u] = h2;
            axpy(simd, h, theta.w2(u), &mut units.pi);
        }
        units.pi[self.location] += dot(simd, theta.skip(), &units.y_parents);
    }

    /// One variable's primal. Writes its edge values to `edge_a` and
    /// `edge_l` and returns `(x, delta, mu)`.
    #[simd]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn local_primal<S: Simd>(
        &self,
        simd: S,
        theta: Params<f64>,
        units: &UnitState<S>,
        y_own: S::f64s,
        edge_a: &mut [S::f64s],
        edge_l: &mut [S::f64s],
        s: &mut Scratch<S>,
    ) -> (S::f64s, S::f64s, S::f64s) {
        let loc = self.location;
        let (x, log_det) = transformer(simd, &self.layers, y_own, &units.pi, &[], 0, &mut s.jets);
        let ws = &s.jets;
        let (t, l) = (&ws.grad(&x)[1..], &ws.grad(&log_det)[1..]);

        let zero = S::f64s::splat(simd, 0.0);
        edge_a.fill(zero);
        edge_l.fill(zero);
        for u in 0..self.n_unit {
            let (w1, w2) = (theta.w1(u), theta.w2(u));
            axpy(simd, dot(simd, w2, t) * units.h1[u], w1, edge_a);
            axpy(simd, dot(simd, w2, l) * units.h1[u], w1, edge_l);
        }
        let edges = edge_a.iter_mut().zip(edge_l.iter_mut());
        for (((a, l_out), &skip), &dz) in edges.zip(theta.skip()).zip(&units.dz) {
            *a = *a * dz + t[loc] * skip;
            *l_out = *l_out * dz + l[loc] * skip;
        }
        (ws.value(&x), ws.grad(&x)[0], ws.grad(&log_det)[0])
    }

    /// One variable's tangents along `v`. Writes `A_dot[i, :]` and
    /// `L_dot[i, :]` to `edge_a` and `edge_l`; returns `(x_dot, delta_dot,
    /// mu_dot)`.
    ///
    /// `pi_dot` comes through the conditioner, `t_dot` and `l_dot` (and
    /// `delta_dot`, `mu_dot`) from one jet pass along `(0, pi_dot)`. Then per
    /// unit, `alpha_c = (W2 . c_dot + W2_dot . c) h' + p_c h'' a_dot` for `c`
    /// in `t`, `l` (`curvature = h'' a_dot`), and `A_dot[i, j] = sum_u
    /// alpha_t W1[u, j] + p_t h' W1_dot[u, j]`, plus the skip.
    #[simd]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn local_pushforward<S: Simd>(
        &self,
        simd: S,
        theta: Params<f64>,
        v: Params<f64>,
        units: &UnitState<S>,
        y_own: S::f64s,
        edge_a: &mut [S::f64s],
        edge_l: &mut [S::f64s],
        s: &mut Scratch<S>,
    ) -> (S::f64s, S::f64s, S::f64s) {
        let n_z = 1 + self.n_par;
        let loc = self.location;
        for (pi_dot, &b) in s.pi_dot.iter_mut().zip(v.b2()) {
            *pi_dot = S::f64s::splat(simd, b);
        }
        for u in 0..self.n_unit {
            let a_dot = dot(simd, v.w1(u), &units.z_parents) + v.b1(u);
            s.unit_a[u] = a_dot;
            axpy(simd, units.h[u], v.w2(u), &mut s.pi_dot);
            axpy(simd, units.h1[u] * a_dot, theta.w2(u), &mut s.pi_dot);
        }
        s.pi_dot[loc] += dot(simd, v.skip(), &units.y_parents);

        s.dirs[0] = S::f64s::splat(simd, 0.0);
        s.dirs[1..n_z].copy_from_slice(&s.pi_dot);
        let (x, log_det) = transformer(
            simd,
            &self.layers,
            y_own,
            &units.pi,
            &s.dirs[..n_z],
            1,
            &mut s.jets,
        );
        let ws = &s.jets;
        let (t, l) = (&ws.grad(&x)[1..], &ws.grad(&log_det)[1..]);
        let (t_dot, l_dot) = (&ws.hess_dir(&x, 0)[1..], &ws.hess_dir(&log_det, 0)[1..]);

        let zero = S::f64s::splat(simd, 0.0);
        edge_a.fill(zero);
        edge_l.fill(zero);
        for u in 0..self.n_unit {
            let (w2, w2_dot) = (theta.w2(u), v.w2(u));
            let (p_t, p_l) = (dot(simd, w2, t), dot(simd, w2, l));
            let h1 = units.h1[u];
            let curvature = units.h2[u] * s.unit_a[u];
            let alpha_t = (dot(simd, w2, t_dot) + dot(simd, w2_dot, t)) * h1 + p_t * curvature;
            let alpha_l = (dot(simd, w2, l_dot) + dot(simd, w2_dot, l)) * h1 + p_l * curvature;
            let (w1, w1_dot) = (theta.w1(u), v.w1(u));
            axpy(simd, alpha_t, w1, edge_a);
            axpy(simd, p_t * h1, w1_dot, edge_a);
            axpy(simd, alpha_l, w1, edge_l);
            axpy(simd, p_l * h1, w1_dot, edge_l);
        }
        let (skip, skip_dot) = (theta.skip(), v.skip());
        for j in 0..edge_a.len() {
            let dz = units.dz[j];
            edge_a[j] = edge_a[j] * dz + t_dot[loc] * skip[j] + t[loc] * skip_dot[j];
            edge_l[j] = edge_l[j] * dz + l_dot[loc] * skip[j] + l[loc] * skip_dot[j];
        }
        (
            dot_lanes(simd, t, &s.pi_dot),
            ws.hess_dir(&x, 0)[0],
            ws.hess_dir(&log_det, 0)[0],
        )
    }

    /// The transformer's gradients into `s.grad_t` and `s.grad_l`, and its
    /// Hessian products along the directions in `s.dirs` -- of `T` along the
    /// first, of `Lambda` along the second -- into `s.prod_t` and `s.prod_l`.
    #[inline(always)]
    fn curvature<S: Simd>(&self, simd: S, y: S::f64s, pi: &[S::f64s], s: &mut Scratch<S>) {
        let n_z = 1 + self.n_par;
        let (x, log_det) = transformer(
            simd,
            &self.layers,
            y,
            pi,
            &s.dirs[..2 * n_z],
            2,
            &mut s.jets,
        );
        let ws = &s.jets;
        s.grad_t.copy_from_slice(ws.grad(&x));
        s.grad_l.copy_from_slice(ws.grad(&log_det));
        s.prod_t.copy_from_slice(ws.hess_dir(&x, 0));
        s.prod_l.copy_from_slice(ws.hess_dir(&log_det, 1));
    }

    /// One variable's local pullback of `cot`, accumulated into `grad`, its
    /// slice of the gradient.
    ///
    /// The edge cotangents reach `t` and `l` through `C`: `t_bar = C A_bar`,
    /// `l_bar = C L_bar`, collapsed per unit through `zeta_a` and `zeta_l`.
    /// One jet pass gives `pi_bar = x_bar t + H_T (delta_bar, t_bar) +
    /// H_Lambda (mu_bar, l_bar)` (restricted to `pi`), which goes back
    /// through the conditioner with `a_bar`, the cotangent of `a`.
    #[simd]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn local_pullback<S: Simd>(
        &self,
        simd: S,
        theta: Params<f64>,
        units: &UnitState<S>,
        cot: &Cotangent<S>,
        y_own: S::f64s,
        s: &mut Scratch<S>,
        mut grad: ParamsMut<S::f64s>,
    ) {
        let n_z = 1 + self.n_par;
        let loc = self.location;
        let zero = S::f64s::splat(simd, 0.0);

        // The units see the edge cotangents through `dz`.
        s.edge_za.clear();
        s.edge_zl.clear();
        for ((&a, &l), &dz) in cot.edge_a.iter().zip(cot.edge_l).zip(&units.dz) {
            s.edge_za.push(a * dz);
            s.edge_zl.push(l * dz);
        }

        // The cotangents of `t` and `l`, from the edge cotangents, keeping
        // `zeta` for `A` and `L` in `unit_a` and `unit_b`.
        s.t_bar.fill(zero);
        s.l_bar.fill(zero);
        for u in 0..self.n_unit {
            let (w1, w2) = (theta.w1(u), theta.w2(u));
            let zeta_a = dot(simd, w1, &s.edge_za);
            let zeta_l = dot(simd, w1, &s.edge_zl);
            s.unit_a[u] = zeta_a;
            s.unit_b[u] = zeta_l;
            axpy(simd, units.h1[u] * zeta_a, w2, &mut s.t_bar);
            axpy(simd, units.h1[u] * zeta_l, w2, &mut s.l_bar);
        }
        s.t_bar[loc] += dot(simd, theta.skip(), cot.edge_a);
        s.l_bar[loc] += dot(simd, theta.skip(), cot.edge_l);

        // `T` along `(delta_bar, t_bar)` and `Lambda` along `(mu_bar,
        // l_bar)`: every second-order path into `pi` at once.
        s.dirs[0] = cot.delta;
        s.dirs[1..n_z].copy_from_slice(&s.t_bar);
        s.dirs[n_z] = cot.mu;
        s.dirs[n_z + 1..2 * n_z].copy_from_slice(&s.l_bar);
        self.curvature(simd, y_own, &units.pi, s);
        let (t, l) = (&s.grad_t[1..], &s.grad_l[1..]);
        for m in 0..self.n_par {
            s.pi_bar[m] = cot.x * t[m] + s.prod_t[1 + m] + s.prod_l[1 + m];
        }

        for u in 0..self.n_unit {
            let w2 = theta.w2(u);
            let (p_t, p_l) = (dot(simd, w2, t), dot(simd, w2, l));
            let (zeta_a, zeta_l) = (s.unit_a[u], s.unit_b[u]);
            let (h, h1) = (units.h[u], units.h1[u]);
            let a_bar = units.h2[u] * (p_t * zeta_a + p_l * zeta_l) + h1 * dot(simd, w2, &s.pi_bar);

            let g_w2 = grad.w2(u);
            axpy_lanes(simd, h1 * zeta_a, t, g_w2);
            axpy_lanes(simd, h1 * zeta_l, l, g_w2);
            axpy_lanes(simd, h, &s.pi_bar, g_w2);

            let g_w1 = grad.w1(u);
            axpy_lanes(simd, h1 * p_t, &s.edge_za, g_w1);
            axpy_lanes(simd, h1 * p_l, &s.edge_zl, g_w1);
            axpy_lanes(simd, a_bar, &units.z_parents, g_w1);
            *grad.b1(u) += a_bar;
        }
        for (g, &p) in grad.b2().iter_mut().zip(&s.pi_bar) {
            *g += p;
        }
        let g_skip = grad.skip();
        axpy_lanes(simd, t[loc], cot.edge_a, g_skip);
        axpy_lanes(simd, l[loc], cot.edge_l, g_skip);
        axpy_lanes(simd, s.pi_bar[loc], &units.y_parents, g_skip);
    }

    /// The transformer's gradients in `z = (y, pi)`, at `y` and `pi`, and its
    /// full Hessians as packed lower triangles, from one jet pass.
    #[simd]
    pub(super) fn dense_curvature<S: Simd>(
        &self,
        simd: S,
        y: S::f64s,
        pi: &[S::f64s],
        jets: &mut JetArena<S>,
        out: &mut DenseCurvature<S>,
    ) {
        let (x, log_det) = transformer_hessian(simd, &self.layers, y, pi, jets);
        out.grad_t.copy_from_slice(jets.grad(&x));
        out.grad_l.copy_from_slice(jets.grad(&log_det));
        out.hess_t.copy_from_slice(jets.hessian(&x));
        out.hess_l.copy_from_slice(jets.hessian(&log_det));
    }
}
