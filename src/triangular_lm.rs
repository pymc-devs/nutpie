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
//! The tape keeps only what the triangular solves need across variables
//! (`A`, `delta`, `x`, `w`). Everything local to a variable -- the hidden
//! units and the transformer's derivatives -- is recomputed by each
//! operator, with one jet pass of the transformer per variable and tile.

use anyhow::{bail, Result};
use fearless_simd::{dispatch, prelude::*, Level};
use fearless_simd_macros::simd;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, Par};

use crate::simd_math;
use crate::triangular::{Contract2Spec, LayerSpec, Param, PositiveAffineSpec, TangentSasSpec};

/// Capacity of a [`Jet`]: the transformer input `y` and up to 15 transformer
/// parameters (three `Contract2` layers, or three `TangentSAS` layers and
/// their `PositiveAffine`).
const N: usize = 16;
/// Draws per matrix product when accumulating the exact blocks.
const BLOCK_DRAW_BATCH: usize = 32;

/// `K` directions in `z = (y, pi)`, one vector per entry.
type Dirs<S, const K: usize> = [[<S as Simd>::f64s; N]; K];

/// A scalar function of `z = (y, pi)`, one lane per draw, with its gradient
/// and the products of its Hessian with `K` fixed directions:
/// `e[k] = g . d_k`, `h[k] = H d_k`.
///
/// Forward-over-forward, but sharing the value and gradient across the
/// directions, so every operation is O(N) instead of the O(N^2) a full
/// Hessian would cost.
#[derive(Clone, Copy)]
struct Jet<S: Simd, const K: usize> {
    simd: S,
    v: S::f64s,
    g: [S::f64s; N],
    e: [S::f64s; K],
    h: [[S::f64s; N]; K],
}

impl<S: Simd, const K: usize> Jet<S, K> {
    #[inline(always)]
    fn constant(simd: S, v: S::f64s) -> Self {
        let zero = S::f64s::splat(simd, 0.0);
        Self {
            simd,
            v,
            g: [zero; N],
            e: [zero; K],
            h: [[zero; N]; K],
        }
    }

    #[inline(always)]
    fn variable(simd: S, v: S::f64s, index: usize, dirs: &Dirs<S, K>) -> Self {
        let mut out = Self::constant(simd, v);
        out.g[index] = S::f64s::splat(simd, 1.0);
        for k in 0..K {
            out.e[k] = dirs[k][index];
        }
        out
    }

    #[inline(always)]
    fn one(&self) -> S::f64s {
        S::f64s::splat(self.simd, 1.0)
    }

    /// `f(self)`, given `f`, `f'` and `f''` at `self.v`.
    #[inline(always)]
    fn chain(self, f0: S::f64s, f1: S::f64s, f2: S::f64s) -> Self {
        let mut out = Self::constant(self.simd, f0);
        for n in 0..N {
            out.g[n] = f1 * self.g[n];
        }
        for k in 0..K {
            out.e[k] = f1 * self.e[k];
            let curvature = f2 * self.e[k];
            for n in 0..N {
                out.h[k][n] = f1 * self.h[k][n] + curvature * self.g[n];
            }
        }
        out
    }

    #[inline(always)]
    fn asinh(self) -> Self {
        let a = self.v;
        let root = simd_math::hypot_one(self.simd, a);
        let value = simd_math::asinh(self.simd, a);
        self.chain(value, self.one() / root, -a / (root * root * root))
    }

    #[inline(always)]
    fn sinh(self) -> Self {
        let (s, c) = simd_math::sinh_cosh(self.simd, self.v);
        self.chain(s, c, s)
    }

    #[inline(always)]
    fn cosh(self) -> Self {
        let (s, c) = simd_math::sinh_cosh(self.simd, self.v);
        self.chain(c, s, c)
    }

    /// `v + sqrt(1 + v*v)`, in `exp_asinh`'s cancellation-free form.
    #[inline(always)]
    fn positive(self) -> Self {
        let a = self.v;
        let one = self.one();
        let root = simd_math::hypot_one(self.simd, a);
        let value = a.simd_ge(0.0).select(a + root, one / (root - a));
        // d/da (value / root) = (value (root - a)) / root^3 = 1 / root^3.
        self.chain(value, value / root, one / (root * root * root))
    }

    #[inline(always)]
    fn exp(self) -> Self {
        let e = simd_math::exp(self.simd, self.v);
        self.chain(e, e, e)
    }

    #[inline(always)]
    fn ln_1p(self) -> Self {
        let inv = self.one() / (self.v + 1.0);
        let value = simd_math::ln_1p(self.simd, self.v);
        self.chain(value, inv, -inv * inv)
    }

    /// `log(cosh(v))`, in `_log_cosh`'s stable form.
    #[inline(always)]
    fn log_cosh(self) -> Self {
        let (value, t) = simd_math::log_cosh_tanh(self.simd, self.v);
        self.chain(value, t, self.one() - t * t)
    }

    #[inline(always)]
    fn sigmoid(self) -> Self {
        let one = self.one();
        let s = one / (simd_math::exp(self.simd, -self.v) + 1.0);
        let ds = s * (one - s);
        self.chain(s, ds, ds * (one - s * 2.0))
    }

    #[inline(always)]
    fn square(self) -> Self {
        self * self
    }
}

impl<S: Simd, const K: usize> std::ops::Add for Jet<S, K> {
    type Output = Self;
    #[inline(always)]
    fn add(mut self, other: Self) -> Self {
        self.v += other.v;
        for n in 0..N {
            self.g[n] += other.g[n];
        }
        for k in 0..K {
            self.e[k] += other.e[k];
            for n in 0..N {
                self.h[k][n] += other.h[k][n];
            }
        }
        self
    }
}

impl<S: Simd, const K: usize> std::ops::Neg for Jet<S, K> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        self * -1.0
    }
}

impl<S: Simd, const K: usize> std::ops::Sub for Jet<S, K> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        self + (-other)
    }
}

impl<S: Simd, const K: usize> std::ops::Mul for Jet<S, K> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        let (a, b) = (self, other);
        let mut out = Self::constant(a.simd, a.v * b.v);
        for n in 0..N {
            out.g[n] = a.v * b.g[n] + b.v * a.g[n];
        }
        for k in 0..K {
            out.e[k] = a.v * b.e[k] + b.v * a.e[k];
            for n in 0..N {
                out.h[k][n] = a.v * b.h[k][n] + b.v * a.h[k][n] + a.e[k] * b.g[n] + b.e[k] * a.g[n];
            }
        }
        out
    }
}

impl<S: Simd, const K: usize> std::ops::Mul<f64> for Jet<S, K> {
    type Output = Self;
    #[inline(always)]
    fn mul(mut self, c: f64) -> Self {
        self.v *= c;
        for n in 0..N {
            self.g[n] *= c;
        }
        for k in 0..K {
            self.e[k] *= c;
            for n in 0..N {
                self.h[k][n] *= c;
            }
        }
        self
    }
}

impl<S: Simd, const K: usize> std::ops::Add<f64> for Jet<S, K> {
    type Output = Self;
    #[inline(always)]
    fn add(mut self, c: f64) -> Self {
        self.v += c;
        self
    }
}

/// The transformer parameters `pi` of one variable as jet inputs.
struct Fields<'a, S: Simd, const K: usize> {
    simd: S,
    pi: &'a [S::f64s],
    dirs: &'a Dirs<S, K>,
}

impl<S: Simd, const K: usize> Fields<'_, S, K> {
    #[inline(always)]
    fn get(&self, param: Option<Param>) -> Option<Jet<S, K>> {
        param.map(|p| {
            Jet::variable(
                self.simd,
                self.pi[p.index] + p.offset,
                1 + p.index,
                self.dirs,
            )
        })
    }

    #[inline(always)]
    fn get_or_zero(&self, param: Option<Param>) -> Jet<S, K> {
        self.get(param)
            .unwrap_or_else(|| Jet::constant(self.simd, S::f64s::splat(self.simd, 0.0)))
    }
}

/// `_bounded_log_gamma`'s constants for one pair of bounds.
#[derive(Debug, Clone, Copy)]
struct GammaBound {
    low: f64,
    width: f64,
    slope: f64,
    offset: f64,
}

#[derive(Debug, Clone, Copy)]
struct Contract2Layer {
    alpha: Option<Param>,
    beta: Option<Param>,
    sigma: Option<Param>,
    mu: Option<Param>,
    nu: Option<Param>,
    bound: Option<GammaBound>,
}

impl Contract2Layer {
    fn new(spec: Contract2Spec) -> Result<Self> {
        let bound = match spec.log_gamma_bounds {
            None => None,
            Some((low, high)) => {
                if !(low < 0.0 && 0.0 < high) {
                    bail!("log_gamma_bounds must satisfy low < 0 < high, got ({low}, {high})");
                }
                let width = high - low;
                let at_zero = -low / width;
                Some(GammaBound {
                    low,
                    width,
                    slope: width / (-low * high),
                    offset: (at_zero / (1.0 - at_zero)).ln(),
                })
            }
        };
        Ok(Self {
            alpha: spec.alpha,
            beta: spec.beta,
            sigma: spec.sigma,
            mu: spec.mu,
            nu: spec.nu,
            bound,
        })
    }

    /// `Contract2.inverse_and_log_det`.
    #[inline(always)]
    fn inverse<S: Simd, const K: usize>(
        &self,
        fields: &Fields<S, K>,
        x: Jet<S, K>,
    ) -> (Jet<S, K>, Jet<S, K>) {
        let log_gamma = fields.get(self.alpha).map(|alpha| {
            let log_gamma = alpha.asinh();
            match self.bound {
                None => log_gamma,
                Some(b) => (log_gamma * b.slope + b.offset).sigmoid() * b.width + b.low,
            }
        });
        let log_delta = fields.get(self.beta).map(Jet::asinh);
        let log_sigma = fields.get(self.sigma).map(Jet::asinh);

        let centred = match fields.get(self.mu) {
            Some(mu) => x - mu,
            None => x,
        };
        let log_scale = match (log_gamma, log_sigma) {
            (Some(g), Some(s)) => Some(g - s),
            (Some(g), None) => Some(g),
            (None, Some(s)) => Some(-s),
            (None, None) => None,
        };
        let half_a = match log_scale {
            Some(scale) => scale.exp() * centred * 0.5,
            None => centred * 0.5,
        };
        let arg = half_a.asinh();
        let shifted = match log_delta {
            Some(delta) => arg - delta * 2.0,
            None => arg,
        };
        let u = match log_gamma {
            Some(g) => shifted * (-g).exp(),
            None => shifted,
        };

        let mut out = u.sinh() * 2.0;
        if let Some(nu) = fields.get(self.nu) {
            out = out + nu;
        }
        let mut ld = u.log_cosh() - half_a.square().ln_1p() * 0.5;
        if let Some(s) = log_sigma {
            ld = ld - s;
        }
        (out, ld)
    }
}

/// `TangentSAS.inverse_and_log_det`: with
/// `q = r b cosh(eps) (y - nu) + sinh(eps)` and `a = (asinh(q) - eps) / r`,
/// `x = nu + sinh(a) / b` and `log_det = logcosh(a) + logcosh(eps) -
/// 0.5 log1p(q^2)`. `1 / b = 1 + exp(-b_raw)` and
/// `1 / r = positive(-r_raw)`.
#[inline(always)]
fn tangent_sas_inverse<S: Simd, const K: usize>(
    layer: &TangentSasSpec,
    fields: &Fields<S, K>,
    x: Jet<S, K>,
) -> (Jet<S, K>, Jet<S, K>) {
    let nu = fields.get_or_zero(layer.nu);
    let eps = fields.get_or_zero(layer.eps);
    let b_raw = fields.get_or_zero(layer.b);
    let r_raw = fields.get_or_zero(layer.r);

    let q = (x - nu) * b_raw.sigmoid() * r_raw.positive() * eps.cosh() + eps.sinh();
    let a = (q.asinh() - eps) * (-r_raw).positive();
    let out = a.sinh() * ((-b_raw).exp() + 1.0) + nu;
    let ld = a.log_cosh() + eps.log_cosh() - q.square().ln_1p() * 0.5;
    (out, ld)
}

/// `PositiveAffine.inverse_and_log_det`: `x = (y - loc) / scale_mod`, and
/// `1 / scale_mod = positive(-scale)`.
#[inline(always)]
fn positive_affine_inverse<S: Simd, const K: usize>(
    layer: &PositiveAffineSpec,
    fields: &Fields<S, K>,
    x: Jet<S, K>,
) -> (Jet<S, K>, Jet<S, K>) {
    let centred = match fields.get(layer.loc) {
        Some(loc) => x - loc,
        None => x,
    };
    match fields.get(layer.scale) {
        Some(scale) => (centred * (-scale).positive(), -scale.asinh()),
        None => (centred, fields.get_or_zero(None)),
    }
}

#[derive(Debug, Clone, Copy)]
enum Layer {
    Contract2(Contract2Layer),
    TangentSas(TangentSasSpec),
    PositiveAffine(PositiveAffineSpec),
}

impl Layer {
    fn new(spec: LayerSpec) -> Result<Self> {
        Ok(match spec {
            LayerSpec::Contract2(spec) => Layer::Contract2(Contract2Layer::new(spec)?),
            LayerSpec::TangentSas(spec) => Layer::TangentSas(spec),
            LayerSpec::PositiveAffine(spec) => Layer::PositiveAffine(spec),
        })
    }
}

/// `T(y; pi)` and `Lambda(y; pi) = log |dT/dy|` of the inverted transformer
/// chain (each layer's `inverse_and_log_det`, last layer first), as jets in
/// `z = (y, pi)` along the directions `dirs`.
#[simd]
fn transformer<S: Simd, const K: usize>(
    simd: S,
    layers: &[Layer],
    y: S::f64s,
    pi: &[S::f64s],
    dirs: &Dirs<S, K>,
) -> (Jet<S, K>, Jet<S, K>) {
    let fields = Fields { simd, pi, dirs };
    let mut x = Jet::variable(simd, y, 0, dirs);
    let mut log_det = fields.get_or_zero(None);
    for layer in layers.iter().rev() {
        let (out, ld) = match layer {
            Layer::Contract2(layer) => layer.inverse(&fields, x),
            Layer::TangentSas(layer) => tangent_sas_inverse(layer, &fields, x),
            Layer::PositiveAffine(layer) => positive_affine_inverse(layer, &fields, x),
        };
        x = out;
        log_det = log_det + ld;
    }
    (x, log_det)
}

/// `sum_k w[k] x[k]`: weights shared by the draws, against per-draw values.
#[inline(always)]
fn dot<S: Simd>(simd: S, w: &[f64], x: &[S::f64s]) -> S::f64s {
    let mut acc = S::f64s::splat(simd, 0.0);
    for (&w, &x) in w.iter().zip(x) {
        acc += x * w;
    }
    acc
}

#[inline(always)]
fn dot_lanes<S: Simd>(simd: S, a: &[S::f64s], b: &[S::f64s]) -> S::f64s {
    let mut acc = S::f64s::splat(simd, 0.0);
    for (&a, &b) in a.iter().zip(b) {
        acc += a * b;
    }
    acc
}

/// `y += alpha x` for shared `x`.
#[inline(always)]
fn axpy<S: Simd>(_: S, alpha: S::f64s, x: &[f64], y: &mut [S::f64s]) {
    for (y, &x) in y.iter_mut().zip(x) {
        *y += alpha * x;
    }
}

#[inline(always)]
fn axpy_lanes<S: Simd>(_: S, alpha: S::f64s, x: &[S::f64s], y: &mut [S::f64s]) {
    for (y, &x) in y.iter_mut().zip(x) {
        *y += alpha * x;
    }
}

/// One tile's `dst.len()` vectors from their `(count, width)` block.
#[simd]
fn load<S: Simd>(simd: S, src: &[f64], dst: &mut [S::f64s]) {
    for (dst, src) in dst.iter_mut().zip(src.chunks_exact(S::f64s::LEN)) {
        *dst = S::f64s::from_slice(simd, src);
    }
}

#[simd]
fn store<S: Simd>(_: S, src: &[S::f64s], dst: &mut [f64]) {
    for (src, dst) in src.iter().zip(dst.chunks_exact_mut(S::f64s::LEN)) {
        src.store_slice(dst);
    }
}

fn f64_width<S: Simd>(_: S) -> usize {
    S::f64s::LEN
}

/// Where one variable's conditioner parameters sit in its slice.
#[derive(Clone, Copy)]
struct Shape {
    n_parent: usize,
    n_unit: usize,
    n_par: usize,
}

impl Shape {
    #[inline(always)]
    fn stride(&self) -> usize {
        self.n_parent + 1 + self.n_par
    }

    fn size(&self) -> usize {
        self.n_unit * self.stride() + self.n_par + self.n_parent
    }

    #[inline(always)]
    fn w1(&self, u: usize) -> std::ops::Range<usize> {
        let start = u * self.stride();
        start..start + self.n_parent
    }

    #[inline(always)]
    fn b1(&self, u: usize) -> usize {
        u * self.stride() + self.n_parent
    }

    #[inline(always)]
    fn w2(&self, u: usize) -> std::ops::Range<usize> {
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

/// A cotangent on one variable's local outputs `(x_i, delta_i, mu_i, A[i, :],
/// L[i, :])`.
struct Cotangent<'a, S: Simd> {
    x: S::f64s,
    delta: S::f64s,
    mu: S::f64s,
    edge_a: &'a [S::f64s],
    edge_l: &'a [S::f64s],
}

/// Per-thread scratch for the local kernels.
struct Scratch<S: Simd> {
    y_parents: Vec<S::f64s>,
    /// Softplus and its first two derivatives at the hidden units.
    h: Vec<S::f64s>,
    h1: Vec<S::f64s>,
    h2: Vec<S::f64s>,
    unit_a: Vec<S::f64s>,
    unit_b: Vec<S::f64s>,
    pi: Vec<S::f64s>,
    pi_dot: Vec<S::f64s>,
    t_bar: Vec<S::f64s>,
    l_bar: Vec<S::f64s>,
    pi_bar: Vec<S::f64s>,
}

impl<S: Simd> Scratch<S> {
    fn new(simd: S, n_unit: usize, n_par: usize, max_parent: usize) -> Self {
        let zero = S::f64s::splat(simd, 0.0);
        Self {
            y_parents: Vec::with_capacity(max_parent),
            h: vec![zero; n_unit],
            h1: vec![zero; n_unit],
            h2: vec![zero; n_unit],
            unit_a: vec![zero; n_unit],
            unit_b: vec![zero; n_unit],
            pi: vec![zero; n_par],
            pi_dot: vec![zero; n_par],
            t_bar: vec![zero; n_par],
            l_bar: vec![zero; n_par],
            pi_bar: vec![zero; n_par],
        }
    }
}

/// Gradients of `T` and `Lambda` in `z = (y, pi)`, and their Hessians'
/// products with the directions `dir_t` and `dir_l` of a pullback.
struct Curvature<S: Simd> {
    grad_t: [S::f64s; N],
    grad_l: [S::f64s; N],
    hess_t: [S::f64s; N],
    hess_l: [S::f64s; N],
}

/// Where a local pullback gets its [`Curvature`] from.
enum CurvatureSource<'a, S: Simd> {
    /// A jet pass of the transformer at the variable's own `y`.
    Recompute { y: S::f64s },
    /// Gradients and packed Hessians from [`FisherResiduals::dense_curvature`],
    /// for the many seeds of the exact blocks.
    Dense {
        grad_t: &'a [S::f64s; N],
        grad_l: &'a [S::f64s; N],
        hess_t: &'a [S::f64s],
        hess_l: &'a [S::f64s],
    },
}

/// Length of a packed lower triangle of an `n x n` matrix.
fn packed_len(n: usize) -> usize {
    n * (n + 1) / 2
}

/// `out = H d` for a symmetric `H` stored as a packed lower triangle.
#[inline(always)]
fn packed_product<S: Simd>(simd: S, packed: &[S::f64s], d: &[S::f64s], out: &mut [S::f64s]) {
    out.fill(S::f64s::splat(simd, 0.0));
    let mut index = 0;
    for r in 0..d.len() {
        for c in 0..r {
            let value = packed[index];
            out[r] += value * d[c];
            out[c] += value * d[r];
            index += 1;
        }
        out[r] += packed[index] * d[r];
        index += 1;
    }
}

/// One exact Gauss-Newton sub-block: parameters `start..start + n` of the
/// flat parameter vector, `n = matrix.nrows()`.
pub(crate) struct GnBlock {
    pub(crate) start: usize,
    pub(crate) matrix: Mat<f64>,
}

/// What every operator at `theta` reads from the primal: the quantities the
/// triangular solves couple across variables. Each array is laid out
/// `(n_tile, count, width)`, so a tile's values for one variable or edge are
/// one SIMD vector.
#[derive(Default)]
pub(crate) struct Tape {
    theta: Vec<f64>,
    edge_a: Vec<f64>,
    delta: Vec<f64>,
    x: Vec<f64>,
    w: Vec<f64>,
}

/// One tile of the [`Tape`], being written by the primal.
struct TileTapeMut<'a> {
    edge_a: &'a mut [f64],
    delta: &'a mut [f64],
    x: &'a mut [f64],
    w: &'a mut [f64],
}

/// Splits `values` into `n` consecutive chunks of `size`, which may be zero.
fn chunks<T>(values: &mut [T], size: usize, n: usize) -> Vec<&mut [T]> {
    let mut out = Vec::with_capacity(n);
    let mut rest = values;
    for _ in 0..n {
        let (head, tail) = rest.split_at_mut(size);
        out.push(head);
        rest = tail;
    }
    out
}

/// Per-thread scratch for the sweeps over one tile, and that tile's data and
/// tape as vectors.
struct TileScratch<S: Simd> {
    local: Scratch<S>,
    y: Vec<S::f64s>,
    g: Vec<S::f64s>,
    edge_a: Vec<S::f64s>,
    edge_l: Vec<S::f64s>,
    delta: Vec<S::f64s>,
    x: Vec<S::f64s>,
    w: Vec<S::f64s>,
    by_var: [Vec<S::f64s>; 2],
    by_edge: [Vec<S::f64s>; 2],
    residual: Vec<S::f64s>,
}

impl<S: Simd> TileScratch<S> {
    fn new(simd: S, problem: &FisherResiduals) -> Self {
        let zero = S::f64s::splat(simd, 0.0);
        let by_var = || vec![zero; problem.n_var];
        let by_edge = || vec![zero; problem.n_edge()];
        Self {
            local: problem.scratch(simd),
            y: by_var(),
            g: by_var(),
            edge_a: by_edge(),
            edge_l: by_edge(),
            delta: by_var(),
            x: by_var(),
            w: by_var(),
            by_var: [by_var(), by_var()],
            by_edge: [by_edge(), by_edge()],
            residual: vec![zero; problem.n_residuals()],
        }
    }
}

/// Per-thread scratch for [`FisherResiduals::variable_blocks`].
struct BlockScratch<S: Simd> {
    local: Scratch<S>,
    /// The Jacobian rows of one tile, `n_theta` per column.
    rows: Vec<S::f64s>,
    edge_a: Vec<S::f64s>,
    cot_a: Vec<S::f64s>,
    cot_l: Vec<S::f64s>,
    hess_t: Vec<S::f64s>,
    hess_l: Vec<S::f64s>,
}

#[derive(Clone)]
pub(crate) struct FisherResiduals {
    level: Level,
    /// Lanes of `level`'s native `f64` vector: the draws in a tile.
    width: usize,
    pub(crate) n_var: usize,
    n_unit: usize,
    n_par: usize,
    location: usize,
    parent_indptr: Vec<usize>,
    parent_index: Vec<usize>,
    max_parent: usize,
    pub(crate) param_offset: Vec<usize>,
    layers: Vec<Layer>,
    /// `sqrt(fisher_regularization)`, if regularized.
    regularization: Option<f64>,
    /// The parent sets closed under elimination, on which the selected
    /// inverse is stored. Equal to the parents for a chordal pattern.
    filled_indptr: Vec<usize>,
    filled_index: Vec<usize>,
    pub(crate) n_draw: usize,
    n_tile: usize,
    /// `(n_tile, n_var, width)`, see [`Tape`].
    y: Vec<f64>,
    g: Vec<f64>,
}

impl FisherResiduals {
    #[allow(clippy::too_many_arguments)]
    fn new(
        parent_indptr: Vec<usize>,
        parent_index: Vec<usize>,
        n_unit: usize,
        n_par: usize,
        location: usize,
        specs: Vec<LayerSpec>,
        fisher_regularization: Option<f64>,
    ) -> Result<Self> {
        let Some(n_var) = parent_indptr.len().checked_sub(1) else {
            bail!("parent_indptr must not be empty");
        };
        if parent_indptr[0] != 0
            || parent_indptr.windows(2).any(|w| w[0] > w[1])
            || parent_indptr[n_var] != parent_index.len()
        {
            bail!("parent_indptr is not a valid CSR index pointer");
        }
        for i in 0..n_var {
            for &p in &parent_index[parent_indptr[i]..parent_indptr[i + 1]] {
                if p >= i {
                    bail!("variable {i} has parent {p}, which does not precede it");
                }
            }
        }
        if n_par + 1 > N {
            bail!(
                "the transformer has {n_par} parameters; at most {} are supported",
                N - 1
            );
        }
        if location >= n_par {
            bail!("location index {location} out of range for {n_par} parameters");
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
        if let Some(rho) = fisher_regularization {
            if rho.is_nan() || rho < 0.0 {
                bail!("fisher_regularization must be non-negative, got {rho}");
            }
        }

        let max_parent = (0..n_var)
            .map(|i| parent_indptr[i + 1] - parent_indptr[i])
            .max()
            .unwrap_or(0);
        let mut param_offset = vec![0usize; n_var + 1];
        for i in 0..n_var {
            let shape = Shape {
                n_parent: parent_indptr[i + 1] - parent_indptr[i],
                n_unit,
                n_par,
            };
            param_offset[i + 1] = param_offset[i] + shape.size();
        }

        // Symbolic fill, as `_build_selected_inverse`: each `p` in `P*(i)`
        // must see the earlier part of `P*(i)`. Walking backwards finalizes
        // `P*(i)` before it propagates.
        let mut filled: Vec<std::collections::BTreeSet<usize>> = (0..n_var)
            .map(|i| {
                parent_index[parent_indptr[i]..parent_indptr[i + 1]]
                    .iter()
                    .copied()
                    .collect()
            })
            .collect();
        for i in (0..n_var).rev() {
            let members: Vec<usize> = filled[i].iter().copied().collect();
            for (k, &p) in members.iter().enumerate() {
                filled[p].extend(members[..k].iter().copied());
            }
        }
        let mut filled_indptr = vec![0usize; n_var + 1];
        let mut filled_index = Vec::new();
        for i in 0..n_var {
            filled_index.extend(filled[i].iter().copied());
            filled_indptr[i + 1] = filled_index.len();
        }

        let level = Level::new();
        Ok(Self {
            level,
            width: dispatch!(level, simd => f64_width(simd)),
            n_var,
            n_unit,
            n_par,
            location,
            parent_indptr,
            parent_index,
            max_parent,
            param_offset,
            layers,
            regularization: fisher_regularization.map(f64::sqrt),
            filled_indptr,
            filled_index,
            n_draw: 0,
            n_tile: 0,
            y: Vec::new(),
            g: Vec::new(),
        })
    }

    fn n_edge(&self) -> usize {
        self.parent_index.len()
    }

    pub(crate) fn n_params(&self) -> usize {
        self.param_offset[self.n_var]
    }

    pub(crate) fn n_residuals(&self) -> usize {
        self.n_var
            + if self.regularization.is_some() {
                self.n_edge()
            } else {
                0
            }
    }

    fn scale(&self) -> f64 {
        1.0 / (self.n_draw as f64).sqrt()
    }

    fn scratch<S: Simd>(&self, simd: S) -> Scratch<S> {
        Scratch::new(simd, self.n_unit, self.n_par, self.max_parent)
    }

    fn shape(&self, i: usize) -> Shape {
        Shape {
            n_parent: self.parent_indptr[i + 1] - self.parent_indptr[i],
            n_unit: self.n_unit,
            n_par: self.n_par,
        }
    }

    fn edges(&self, i: usize) -> std::ops::Range<usize> {
        self.parent_indptr[i]..self.parent_indptr[i + 1]
    }

    fn params(&self, i: usize) -> std::ops::Range<usize> {
        self.param_offset[i]..self.param_offset[i + 1]
    }

    pub(crate) fn set_data(&mut self, y: Vec<f64>, g: Vec<f64>) -> Result<()> {
        if self.n_var == 0 || !y.len().is_multiple_of(self.n_var) || y.len() != g.len() {
            bail!("y and g must both have shape (n_draw, {})", self.n_var);
        }
        let n_draw = y.len() / self.n_var;
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
        self.n_draw = n_draw;
        self.n_tile = n_draw / self.width;
        self.y = self.to_tiles(&y, self.n_var);
        self.g = self.to_tiles(&g, self.n_var);
        Ok(())
    }

    /// Row-major `(n_draw, count)` values in the tiled `(n_tile, count,
    /// width)` layout.
    fn to_tiles(&self, values: &[f64], count: usize) -> Vec<f64> {
        let width = self.width;
        let mut out = vec![0.0; values.len()];
        for (draw, row) in values.chunks_exact(count).enumerate() {
            let (tile, lane) = (draw / width, draw % width);
            for (k, &value) in row.iter().enumerate() {
                out[(tile * count + k) * width + lane] = value;
            }
        }
        out
    }

    /// The inverse of [`Self::to_tiles`].
    fn from_tiles(&self, values: &[f64], count: usize) -> Vec<f64> {
        let width = self.width;
        let mut out = vec![0.0; values.len()];
        for (draw, row) in out.chunks_exact_mut(count).enumerate() {
            let (tile, lane) = (draw / width, draw % width);
            for (k, value) in row.iter_mut().enumerate() {
                *value = values[(tile * count + k) * width + lane];
            }
        }
        out
    }

    /// Tile `tile` of a tiled array with `count` values per draw.
    fn tile<'a>(&self, values: &'a [f64], count: usize, tile: usize) -> &'a [f64] {
        let len = count * self.width;
        &values[tile * len..(tile + 1) * len]
    }

    /// Value `k` of tile `tile` of a tiled array with `count` values per draw.
    #[inline(always)]
    fn tile_value<S: Simd>(
        &self,
        simd: S,
        values: &[f64],
        count: usize,
        tile: usize,
        k: usize,
    ) -> S::f64s {
        let start = (tile * count + k) * self.width;
        S::f64s::from_slice(simd, &values[start..start + self.width])
    }

    /// Loads tile `tile`'s draws and tape into `s`.
    #[simd]
    fn load_tile<S: Simd>(&self, simd: S, tile: usize, tape: &Tape, s: &mut TileScratch<S>) {
        let (n_var, n_edge) = (self.n_var, self.n_edge());
        load(simd, self.tile(&self.y, n_var, tile), &mut s.y);
        load(simd, self.tile(&tape.edge_a, n_edge, tile), &mut s.edge_a);
        load(simd, self.tile(&tape.delta, n_var, tile), &mut s.delta);
        load(simd, self.tile(&tape.x, n_var, tile), &mut s.x);
        load(simd, self.tile(&tape.w, n_var, tile), &mut s.w);
    }

    #[inline(always)]
    fn gather_parents<T: Copy>(&self, i: usize, y: &[T], out: &mut Vec<T>) {
        out.clear();
        out.extend(self.parent_index[self.edges(i)].iter().map(|&p| y[p]));
    }

    // ------------------------------------------------------------ local kernels

    /// Variable `i`'s conditioner at its parents, already in `s.y_parents`:
    /// the hidden units `h, h', h''` and the transformer parameters `pi`.
    #[simd]
    fn conditioner<S: Simd>(&self, simd: S, shape: Shape, theta: &[f64], s: &mut Scratch<S>) {
        for (pi, &b) in s.pi.iter_mut().zip(&theta[shape.b2()]) {
            *pi = S::f64s::splat(simd, b);
        }
        for u in 0..shape.n_unit {
            let a = dot(simd, &theta[shape.w1(u)], &s.y_parents) + theta[shape.b1(u)];
            let (h, h1, h2) = simd_math::softplus(simd, a);
            s.h[u] = h;
            s.h1[u] = h1;
            s.h2[u] = h2;
            axpy(simd, h, &theta[shape.w2(u)], &mut s.pi);
        }
        s.pi[self.location] += dot(simd, &theta[shape.skip()], &s.y_parents);
    }

    /// Variable `i`'s primal. Writes its edge values to `edge_a` and `edge_l`
    /// and returns `(x, delta, mu)`.
    #[simd]
    #[allow(clippy::too_many_arguments)]
    fn local_primal<S: Simd>(
        &self,
        simd: S,
        shape: Shape,
        theta: &[f64],
        y_own: S::f64s,
        edge_a: &mut [S::f64s],
        edge_l: &mut [S::f64s],
        s: &mut Scratch<S>,
    ) -> (S::f64s, S::f64s, S::f64s) {
        let n_z = 1 + self.n_par;
        let loc = self.location;
        self.conditioner(simd, shape, theta, s);
        let (x, log_det) = transformer::<S, 0>(simd, &self.layers, y_own, &s.pi, &[]);
        let (t, l) = (&x.g[1..n_z], &log_det.g[1..n_z]);

        let skip = &theta[shape.skip()];
        for j in 0..shape.n_parent {
            edge_a[j] = t[loc] * skip[j];
            edge_l[j] = l[loc] * skip[j];
        }
        for u in 0..shape.n_unit {
            let w2 = &theta[shape.w2(u)];
            let w1 = &theta[shape.w1(u)];
            axpy(simd, dot(simd, w2, t) * s.h1[u], w1, edge_a);
            axpy(simd, dot(simd, w2, l) * s.h1[u], w1, edge_l);
        }
        (x.v, x.g[0], log_det.g[0])
    }

    /// Variable `i`'s tangents along `v` (its slice). Writes `A`'s and `L`'s
    /// to `edge_a` and `edge_l`; returns those of `(x, delta, mu)`.
    #[simd]
    #[allow(clippy::too_many_arguments)]
    fn local_pushforward<S: Simd>(
        &self,
        simd: S,
        shape: Shape,
        theta: &[f64],
        v: &[f64],
        y_own: S::f64s,
        edge_a: &mut [S::f64s],
        edge_l: &mut [S::f64s],
        s: &mut Scratch<S>,
    ) -> (S::f64s, S::f64s, S::f64s) {
        let n_z = 1 + self.n_par;
        let loc = self.location;
        self.conditioner(simd, shape, theta, s);
        for (pi_dot, &b) in s.pi_dot.iter_mut().zip(&v[shape.b2()]) {
            *pi_dot = S::f64s::splat(simd, b);
        }
        for u in 0..shape.n_unit {
            let a_dot = dot(simd, &v[shape.w1(u)], &s.y_parents) + v[shape.b1(u)];
            s.unit_a[u] = a_dot;
            axpy(simd, s.h[u], &v[shape.w2(u)], &mut s.pi_dot);
            axpy(simd, s.h1[u] * a_dot, &theta[shape.w2(u)], &mut s.pi_dot);
        }
        s.pi_dot[loc] += dot(simd, &v[shape.skip()], &s.y_parents);

        let mut dir = [S::f64s::splat(simd, 0.0); N];
        dir[1..n_z].copy_from_slice(&s.pi_dot);
        let (x, log_det) = transformer::<S, 1>(simd, &self.layers, y_own, &s.pi, &[dir]);
        let (t, l) = (&x.g[1..n_z], &log_det.g[1..n_z]);
        let (t_dot, l_dot) = (&x.h[0][1..n_z], &log_det.h[0][1..n_z]);

        let skip = &theta[shape.skip()];
        let skip_dot = &v[shape.skip()];
        for j in 0..shape.n_parent {
            edge_a[j] = t_dot[loc] * skip[j] + t[loc] * skip_dot[j];
            edge_l[j] = l_dot[loc] * skip[j] + l[loc] * skip_dot[j];
        }
        for u in 0..shape.n_unit {
            let w2 = &theta[shape.w2(u)];
            let w2_dot = &v[shape.w2(u)];
            let (p_t, p_l) = (dot(simd, w2, t), dot(simd, w2, l));
            let h1 = s.h1[u];
            let curvature = s.h2[u] * s.unit_a[u];
            let alpha_t = (dot(simd, w2, t_dot) + dot(simd, w2_dot, t)) * h1 + p_t * curvature;
            let alpha_l = (dot(simd, w2, l_dot) + dot(simd, w2_dot, l)) * h1 + p_l * curvature;
            let w1 = &theta[shape.w1(u)];
            let w1_dot = &v[shape.w1(u)];
            axpy(simd, alpha_t, w1, edge_a);
            axpy(simd, p_t * h1, w1_dot, edge_a);
            axpy(simd, alpha_l, w1, edge_l);
            axpy(simd, p_l * h1, w1_dot, edge_l);
        }
        (dot_lanes(simd, t, &s.pi_dot), x.h[0][0], log_det.h[0][0])
    }

    /// The transformer's gradients, and Hessian products along `dir_t` (of
    /// `T`) and `dir_l` (of `Lambda`).
    #[inline(always)]
    fn curvature<S: Simd>(
        &self,
        simd: S,
        source: &CurvatureSource<S>,
        pi: &[S::f64s],
        dir_t: &[S::f64s; N],
        dir_l: &[S::f64s; N],
    ) -> Curvature<S> {
        match source {
            CurvatureSource::Recompute { y } => {
                let (x, log_det) =
                    transformer::<S, 2>(simd, &self.layers, *y, pi, &[*dir_t, *dir_l]);
                Curvature {
                    grad_t: x.g,
                    grad_l: log_det.g,
                    hess_t: x.h[0],
                    hess_l: log_det.h[1],
                }
            }
            CurvatureSource::Dense {
                grad_t,
                grad_l,
                hess_t,
                hess_l,
            } => {
                let n_z = 1 + self.n_par;
                let zero = S::f64s::splat(simd, 0.0);
                let mut out = Curvature {
                    grad_t: **grad_t,
                    grad_l: **grad_l,
                    hess_t: [zero; N],
                    hess_l: [zero; N],
                };
                packed_product(simd, hess_t, &dir_t[..n_z], &mut out.hess_t[..n_z]);
                packed_product(simd, hess_l, &dir_l[..n_z], &mut out.hess_l[..n_z]);
                out
            }
        }
    }

    /// Variable `i`'s local pullback of `cot`, accumulated into `grad`, its
    /// slice of the gradient. Needs [`Self::conditioner`] run on `s` first.
    #[simd]
    #[allow(clippy::too_many_arguments)]
    fn local_pullback<S: Simd>(
        &self,
        simd: S,
        shape: Shape,
        theta: &[f64],
        cot: &Cotangent<S>,
        source: &CurvatureSource<S>,
        s: &mut Scratch<S>,
        grad: &mut [S::f64s],
    ) {
        let n_z = 1 + self.n_par;
        let loc = self.location;
        let skip = &theta[shape.skip()];
        let zero = S::f64s::splat(simd, 0.0);

        // The cotangents of `t` and `l`, from the edge cotangents, keeping
        // `zeta` for `A` and `L` in `unit_a` and `unit_b`.
        s.t_bar.fill(zero);
        s.l_bar.fill(zero);
        for u in 0..shape.n_unit {
            let w1 = &theta[shape.w1(u)];
            let zeta_a = dot(simd, w1, cot.edge_a);
            let zeta_l = dot(simd, w1, cot.edge_l);
            s.unit_a[u] = zeta_a;
            s.unit_b[u] = zeta_l;
            let w2 = &theta[shape.w2(u)];
            axpy(simd, s.h1[u] * zeta_a, w2, &mut s.t_bar);
            axpy(simd, s.h1[u] * zeta_l, w2, &mut s.l_bar);
        }
        s.t_bar[loc] += dot(simd, skip, cot.edge_a);
        s.l_bar[loc] += dot(simd, skip, cot.edge_l);

        // `T` along `(delta_bar, t_bar)` and `Lambda` along `(mu_bar,
        // l_bar)`: every second-order path into `pi` at once.
        let (mut dir_t, mut dir_l) = ([zero; N], [zero; N]);
        dir_t[0] = cot.delta;
        dir_t[1..n_z].copy_from_slice(&s.t_bar);
        dir_l[0] = cot.mu;
        dir_l[1..n_z].copy_from_slice(&s.l_bar);
        let curvature = self.curvature(simd, source, &s.pi, &dir_t, &dir_l);
        let (t, l) = (&curvature.grad_t[1..n_z], &curvature.grad_l[1..n_z]);
        for m in 0..self.n_par {
            s.pi_bar[m] = cot.x * t[m] + curvature.hess_t[1 + m] + curvature.hess_l[1 + m];
        }

        for u in 0..shape.n_unit {
            let w2 = &theta[shape.w2(u)];
            let (p_t, p_l) = (dot(simd, w2, t), dot(simd, w2, l));
            let (zeta_a, zeta_l) = (s.unit_a[u], s.unit_b[u]);
            let (h, h1) = (s.h[u], s.h1[u]);
            let a_bar = s.h2[u] * (p_t * zeta_a + p_l * zeta_l) + h1 * dot(simd, w2, &s.pi_bar);

            let g_w2 = &mut grad[shape.w2(u)];
            axpy_lanes(simd, h1 * zeta_a, t, g_w2);
            axpy_lanes(simd, h1 * zeta_l, l, g_w2);
            axpy_lanes(simd, h, &s.pi_bar, g_w2);

            let g_w1 = &mut grad[shape.w1(u)];
            axpy_lanes(simd, h1 * p_t, cot.edge_a, g_w1);
            axpy_lanes(simd, h1 * p_l, cot.edge_l, g_w1);
            axpy_lanes(simd, a_bar, &s.y_parents, g_w1);
            grad[shape.b1(u)] += a_bar;
        }
        for (g, &p) in grad[shape.b2()].iter_mut().zip(&s.pi_bar) {
            *g += p;
        }
        let g_skip = &mut grad[shape.skip()];
        axpy_lanes(simd, t[loc], cot.edge_a, g_skip);
        axpy_lanes(simd, l[loc], cot.edge_l, g_skip);
        axpy_lanes(simd, s.pi_bar[loc], &s.y_parents, g_skip);
    }

    /// The transformer's gradients in `z = (y, pi)`, and its full Hessians as
    /// packed lower triangles, from `ceil((1 + n_par) / 2)` jet passes.
    #[simd]
    fn dense_curvature<S: Simd>(
        &self,
        simd: S,
        y: S::f64s,
        pi: &[S::f64s],
        hess_t: &mut [S::f64s],
        hess_l: &mut [S::f64s],
    ) -> ([S::f64s; N], [S::f64s; N]) {
        let n_z = 1 + self.n_par;
        let zero = S::f64s::splat(simd, 0.0);
        let mut grads = ([zero; N], [zero; N]);
        for first in (0..n_z).step_by(2) {
            let mut dirs = [[zero; N]; 2];
            for (k, dir) in dirs.iter_mut().enumerate() {
                if first + k < n_z {
                    dir[first + k] = S::f64s::splat(simd, 1.0);
                }
            }
            let (x, log_det) = transformer::<S, 2>(simd, &self.layers, y, pi, &dirs);
            for k in 0..2 {
                let column = first + k;
                if column >= n_z {
                    continue;
                }
                // Row `r >= column` of the lower triangle.
                for r in column..n_z {
                    let index = packed_len(r) + column;
                    hess_t[index] = x.h[k][r];
                    hess_l[index] = log_det.h[k][r];
                }
            }
            grads = (x.g, log_det.g);
        }
        grads
    }

    // ------------------------------------------------------------ global sweeps

    /// Solves `J^T w = b` in place (back substitution, children first).
    #[simd]
    fn solve_transpose<S: Simd>(
        &self,
        _: S,
        edge_a: &[S::f64s],
        delta: &[S::f64s],
        b: &mut [S::f64s],
    ) {
        for i in (0..self.n_var).rev() {
            let w_i = b[i] / delta[i];
            b[i] = w_i;
            for e in self.edges(i) {
                b[self.parent_index[e]] -= edge_a[e] * w_i;
            }
        }
    }

    /// Solves `J z = b` in place (forward substitution, parents first).
    #[simd]
    fn solve<S: Simd>(&self, _: S, edge_a: &[S::f64s], delta: &[S::f64s], b: &mut [S::f64s]) {
        for i in 0..self.n_var {
            let mut acc = b[i];
            for e in self.edges(i) {
                acc -= edge_a[e] * b[self.parent_index[e]];
            }
            b[i] = acc / delta[i];
        }
    }

    /// The primal of one tile: its residuals into `out` and its tape.
    #[simd]
    fn primal_tile<S: Simd>(
        &self,
        simd: S,
        theta: &[f64],
        tile: usize,
        tape: TileTapeMut<'_>,
        out: &mut [f64],
        s: &mut TileScratch<S>,
    ) {
        let n_var = self.n_var;
        load(simd, self.tile(&self.y, n_var, tile), &mut s.y);
        load(simd, self.tile(&self.g, n_var, tile), &mut s.g);
        let TileScratch {
            local,
            y,
            g,
            edge_a,
            edge_l,
            delta,
            x,
            w,
            by_var,
            residual,
            ..
        } = s;
        let [grad_log_det, _] = by_var;
        grad_log_det.fill(S::f64s::splat(simd, 0.0));

        for i in 0..n_var {
            self.gather_parents(i, y, &mut local.y_parents);
            let edges = self.edges(i);
            let (x_i, delta_i, mu_i) = self.local_primal(
                simd,
                self.shape(i),
                &theta[self.params(i)],
                y[i],
                &mut edge_a[edges.clone()],
                &mut edge_l[edges.clone()],
                local,
            );
            x[i] = x_i;
            delta[i] = delta_i;
            grad_log_det[i] += mu_i;
            for e in edges {
                grad_log_det[self.parent_index[e]] += edge_l[e];
            }
        }

        for i in 0..n_var {
            w[i] = g[i] - grad_log_det[i];
        }
        self.solve_transpose(simd, edge_a, delta, w);

        let scale = self.scale();
        for i in 0..n_var {
            residual[i] = (x[i] + w[i]) * scale;
        }
        if let Some(reg) = self.regularization {
            for i in 0..n_var {
                for e in self.edges(i) {
                    residual[n_var + e] = (edge_l[e] - x[i] * edge_a[e]) * (scale * reg);
                }
            }
        }

        store(simd, edge_a, tape.edge_a);
        store(simd, delta, tape.delta);
        store(simd, x, tape.x);
        store(simd, w, tape.w);
        store(simd, residual, out);
    }

    /// `J v` for one tile, already loaded into `s`, into `s.residual`.
    #[simd]
    fn pushforward_tile<S: Simd>(&self, simd: S, theta: &[f64], v: &[f64], s: &mut TileScratch<S>) {
        let n_var = self.n_var;
        let TileScratch {
            local,
            y,
            edge_a,
            delta,
            x,
            w,
            by_var,
            by_edge,
            residual,
            ..
        } = s;
        let [x_dot, rhs] = by_var;
        let [a_dot, l_dot] = by_edge;
        rhs.fill(S::f64s::splat(simd, 0.0));

        for i in 0..n_var {
            self.gather_parents(i, y, &mut local.y_parents);
            let edges = self.edges(i);
            let params = self.params(i);
            let (xd, delta_dot, mu_dot) = self.local_pushforward(
                simd,
                self.shape(i),
                &theta[params.clone()],
                &v[params],
                y[i],
                &mut a_dot[edges.clone()],
                &mut l_dot[edges.clone()],
                local,
            );
            x_dot[i] = xd;
            // J^T w_dot = -(grad_y log_det)_dot - J_dot^T w
            rhs[i] -= mu_dot + delta_dot * w[i];
            for e in edges {
                rhs[self.parent_index[e]] -= l_dot[e] + a_dot[e] * w[i];
            }
        }
        self.solve_transpose(simd, edge_a, delta, rhs);

        let scale = self.scale();
        for i in 0..n_var {
            residual[i] = (x_dot[i] + rhs[i]) * scale;
        }
        if let Some(reg) = self.regularization {
            for i in 0..n_var {
                for e in self.edges(i) {
                    residual[n_var + e] =
                        (l_dot[e] - x_dot[i] * edge_a[e] - x[i] * a_dot[e]) * (scale * reg);
                }
            }
        }
    }

    /// Accumulates `J^T r_bar` for one tile, already loaded into `s` with
    /// `r_bar` in `s.residual`, into `grad`.
    #[simd]
    fn pullback_tile<S: Simd>(
        &self,
        simd: S,
        theta: &[f64],
        s: &mut TileScratch<S>,
        grad: &mut [S::f64s],
    ) {
        let n_var = self.n_var;
        let scale = self.scale();
        let TileScratch {
            local,
            y,
            edge_a,
            delta,
            x,
            w,
            by_var,
            by_edge,
            residual,
            ..
        } = s;
        let r_bar = &*residual;
        let [z, _] = by_var;
        let [cot_a, cot_l] = by_edge;

        for i in 0..n_var {
            z[i] = r_bar[i] * scale;
        }
        self.solve(simd, edge_a, delta, z);

        for i in 0..n_var {
            self.gather_parents(i, y, &mut local.y_parents);
            let edges = self.edges(i);
            let mut x_bar = r_bar[i] * scale;
            for e in edges.clone() {
                let z_p = z[self.parent_index[e]];
                cot_a[e] = -(w[i] * z_p);
                cot_l[e] = -z_p;
                if let Some(reg) = self.regularization {
                    let o = r_bar[n_var + e] * (scale * reg);
                    x_bar -= o * edge_a[e];
                    cot_a[e] -= x[i] * o;
                    cot_l[e] += o;
                }
            }
            let cot = Cotangent {
                x: x_bar,
                delta: -(w[i] * z[i]),
                mu: -z[i],
                edge_a: &cot_a[edges.clone()],
                edge_l: &cot_l[edges],
            };
            let shape = self.shape(i);
            let params = self.params(i);
            let theta_i = &theta[params.clone()];
            self.conditioner(simd, shape, theta_i, local);
            self.local_pullback(
                simd,
                shape,
                theta_i,
                &cot,
                &CurvatureSource::Recompute { y: y[i] },
                local,
                &mut grad[params],
            );
        }
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

    /// `(n_draw, n_residuals)` residuals at `theta`, and the tape every
    /// derivative at `theta` reads.
    pub(crate) fn residuals(&self, theta: &[f64]) -> Result<(Vec<f64>, Tape)> {
        self.check_params(theta, "theta")?;
        if self.n_draw == 0 {
            bail!("no data: call `set_data` first");
        }
        Ok(dispatch!(self.level, simd => self.residuals_with(simd, theta)))
    }

    fn residuals_with<S: Simd>(&self, simd: S, theta: &[f64]) -> (Vec<f64>, Tape) {
        let (n_draw, n_tile, width) = (self.n_draw, self.n_tile, self.width);
        let (n_var, n_edge, n_res) = (self.n_var, self.n_edge(), self.n_residuals());
        let mut tape = Tape {
            theta: theta.to_vec(),
            edge_a: vec![0.0; n_draw * n_edge],
            delta: vec![0.0; n_draw * n_var],
            x: vec![0.0; n_draw * n_var],
            w: vec![0.0; n_draw * n_var],
        };
        let mut out = vec![0.0; n_draw * n_res];
        {
            let views: Vec<_> = chunks(&mut tape.edge_a, width * n_edge, n_tile)
                .into_iter()
                .zip(chunks(&mut tape.delta, width * n_var, n_tile))
                .zip(chunks(&mut tape.x, width * n_var, n_tile))
                .zip(chunks(&mut tape.w, width * n_var, n_tile))
                .zip(chunks(&mut out, width * n_res, n_tile))
                .enumerate()
                .map(|(tile, ((((edge_a, delta), x), w), out))| {
                    (
                        tile,
                        TileTapeMut {
                            edge_a,
                            delta,
                            x,
                            w,
                        },
                        out,
                    )
                })
                .collect();
            views.into_par_iter().for_each_init(
                || TileScratch::new(simd, self),
                |s, (tile, tape, out)| self.primal_tile(simd, theta, tile, tape, out, s),
            );
        }
        (self.from_tiles(&out, n_res), tape)
    }

    pub(crate) fn pushforward(&self, tape: &Tape, v: &[f64]) -> Result<Vec<f64>> {
        self.check_params(v, "v")?;
        Ok(dispatch!(self.level, simd => self.pushforward_with(simd, tape, v)))
    }

    fn pushforward_with<S: Simd>(&self, simd: S, tape: &Tape, v: &[f64]) -> Vec<f64> {
        let n_res = self.n_residuals();
        let mut out = vec![0.0; self.n_draw * n_res];
        out.par_chunks_mut(self.width * n_res)
            .enumerate()
            .for_each_init(
                || TileScratch::new(simd, self),
                |s, (tile, out)| {
                    self.load_tile(simd, tile, tape, s);
                    self.pushforward_tile(simd, &tape.theta, v, s);
                    store(simd, &s.residual, out);
                },
            );
        self.from_tiles(&out, n_res)
    }

    pub(crate) fn pullback(&self, tape: &Tape, r_bar: &[f64]) -> Result<Vec<f64>> {
        let n_res = self.n_residuals();
        if r_bar.len() != self.n_draw * n_res {
            bail!("r_bar must have shape ({}, {n_res})", self.n_draw);
        }
        Ok(dispatch!(self.level, simd => self.pullback_with(simd, tape, r_bar)))
    }

    fn pullback_with<S: Simd>(&self, simd: S, tape: &Tape, r_bar: &[f64]) -> Vec<f64> {
        let n_res = self.n_residuals();
        let r_bar = self.to_tiles(r_bar, n_res);
        self.sum_over_tiles(simd, |tile, grad, s| {
            self.load_tile(simd, tile, tape, s);
            load(simd, self.tile(&r_bar, n_res, tile), &mut s.residual);
            self.pullback_tile(simd, &tape.theta, s, grad);
        })
    }

    /// `J^T J v`, one tile at a time, without forming `J v` for all draws.
    pub(crate) fn gauss_newton_product(&self, tape: &Tape, v: &[f64]) -> Result<Vec<f64>> {
        self.check_params(v, "v")?;
        Ok(dispatch!(self.level, simd => self.gauss_newton_product_with(simd, tape, v)))
    }

    fn gauss_newton_product_with<S: Simd>(&self, simd: S, tape: &Tape, v: &[f64]) -> Vec<f64> {
        self.sum_over_tiles(simd, |tile, grad, s| {
            self.load_tile(simd, tile, tape, s);
            self.pushforward_tile(simd, &tape.theta, v, s);
            self.pullback_tile(simd, &tape.theta, s, grad);
        })
    }

    /// `sum_tiles f(tile)`, each `f` accumulating its tile's lanes into a
    /// per-thread gradient.
    fn sum_over_tiles<S: Simd>(
        &self,
        simd: S,
        f: impl Fn(usize, &mut [S::f64s], &mut TileScratch<S>) + Sync,
    ) -> Vec<f64> {
        let n_params = self.n_params();
        let zero = S::f64s::splat(simd, 0.0);
        (0..self.n_tile)
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

    // ------------------------------------------------------------ exact blocks

    /// `Sigma = (J^T J)^{-1}` on the diagonal and the filled pattern, by
    /// Takahashi's recurrence: the diagonal, then one slot per entry of
    /// `filled_index`.
    #[simd]
    fn selected_inverse<S: Simd>(
        &self,
        simd: S,
        edge_a: &[S::f64s],
        delta: &[S::f64s],
        store: &mut [S::f64s],
    ) {
        let zero = S::f64s::splat(simd, 0.0);
        let one = S::f64s::splat(simd, 1.0);
        for i in 0..self.n_var {
            let filled = self.filled_indptr[i]..self.filled_indptr[i + 1];
            for slot in filled {
                let j = self.filled_index[slot];
                let mut acc = zero;
                for e in self.edges(i) {
                    acc += edge_a[e] * self.sigma(store, self.parent_index[e], j);
                }
                store[self.n_var + slot] = -acc / delta[i];
            }
            let mut acc = zero;
            for e in self.edges(i) {
                acc += edge_a[e] * self.sigma(store, i, self.parent_index[e]);
            }
            store[i] = (one / delta[i] - acc) / delta[i];
        }
    }

    #[inline(always)]
    fn sigma<V: Copy>(&self, store: &[V], a: usize, b: usize) -> V {
        if a == b {
            return store[a];
        }
        let (row, col) = if a > b { (a, b) } else { (b, a) };
        let start = self.filled_indptr[row];
        let filled = &self.filled_index[start..self.filled_indptr[row + 1]];
        let k = filled
            .binary_search(&col)
            .expect("selected inverse entry outside the filled pattern");
        store[self.n_var + start + k]
    }

    /// Exact Gauss-Newton blocks, `sum_draws (dr/dtheta_i)^T (dr/dtheta_i) /
    /// n_draw`, for each variable restricted to consecutive sub-blocks of at
    /// most `max_block_size` parameters of its slice. Ordered by variable,
    /// then position.
    pub(crate) fn gauss_newton_blocks(
        &self,
        tape: &Tape,
        max_block_size: usize,
    ) -> Result<Vec<GnBlock>> {
        if max_block_size == 0 {
            bail!("max_block_size must be positive");
        }
        Ok(dispatch!(self.level, simd => self.gauss_newton_blocks_with(simd, tape, max_block_size)))
    }

    fn gauss_newton_blocks_with<S: Simd>(
        &self,
        simd: S,
        tape: &Tape,
        max_block_size: usize,
    ) -> Vec<GnBlock> {
        let (n_var, n_edge) = (self.n_var, self.n_edge());
        let zero = S::f64s::splat(simd, 0.0);
        let n_store = n_var + self.filled_index.len();
        let mut stores = vec![zero; self.n_tile * n_store];
        stores.par_chunks_mut(n_store).enumerate().for_each_init(
            || (vec![zero; n_edge], vec![zero; n_var]),
            |(edge_a, delta), (tile, store)| {
                load(simd, self.tile(&tape.edge_a, n_edge, tile), edge_a);
                load(simd, self.tile(&tape.delta, n_var, tile), delta);
                self.selected_inverse(simd, edge_a, delta, store);
            },
        );

        let n_z = 1 + self.n_par;
        let per_var: Vec<Vec<GnBlock>> = (0..n_var)
            .into_par_iter()
            .map_init(
                || BlockScratch {
                    local: self.scratch(simd),
                    rows: Vec::new(),
                    edge_a: Vec::with_capacity(self.max_parent),
                    cot_a: Vec::with_capacity(self.max_parent),
                    cot_l: Vec::with_capacity(self.max_parent),
                    hess_t: vec![zero; packed_len(n_z)],
                    hess_l: vec![zero; packed_len(n_z)],
                },
                |b, i| self.variable_blocks(simd, tape, &stores, n_store, i, max_block_size, b),
            )
            .collect();
        per_var.into_iter().flatten().collect()
    }

    /// Variable `i`'s Jacobian rows for one tile, as columns of `b.rows`
    /// (`n_theta` each): `a = dx_i/dtheta_i`, then `B' = B + alpha a` (see
    /// `variable_blocks`), then the scores scaled by `sqrt(rho)`.
    #[simd]
    fn block_rows<S: Simd>(
        &self,
        simd: S,
        i: usize,
        theta: &[f64],
        tile: usize,
        tape: &Tape,
        b: &mut BlockScratch<S>,
    ) {
        let shape = self.shape(i);
        let n_theta = shape.size();
        let (n_var, n_edge) = (self.n_var, self.n_edge());
        let (n_p, n_s) = (shape.n_parent, shape.n_parent + 1);
        let n_score = if self.regularization.is_some() {
            n_p
        } else {
            0
        };
        let zero = S::f64s::splat(simd, 0.0);
        let one = S::f64s::splat(simd, 1.0);
        let edges = self.edges(i);

        let BlockScratch {
            local: s,
            rows,
            edge_a,
            cot_a,
            cot_l,
            hess_t,
            hess_l,
        } = b;
        s.y_parents.clear();
        edge_a.clear();
        for e in edges.clone() {
            let p = self.parent_index[e];
            s.y_parents
                .push(self.tile_value(simd, &self.y, n_var, tile, p));
            edge_a.push(self.tile_value(simd, &tape.edge_a, n_edge, tile, e));
        }
        let y_own = self.tile_value(simd, &self.y, n_var, tile, i);
        let w_i = self.tile_value(simd, &tape.w, n_var, tile, i);
        let x_i = self.tile_value(simd, &tape.x, n_var, tile, i);
        let delta_i = self.tile_value(simd, &tape.delta, n_var, tile, i);

        self.conditioner(simd, shape, theta, s);
        let (grad_t, grad_l) = self.dense_curvature(simd, y_own, &s.pi, hess_t, hess_l);
        let source = CurvatureSource::Dense {
            grad_t: &grad_t,
            grad_l: &grad_l,
            hess_t,
            hess_l,
        };

        // Each Jacobian row is a local pullback of one seed.
        let n_col = 1 + n_s + n_score;
        rows.clear();
        rows.resize(n_col * n_theta, zero);
        for col in 0..n_col {
            cot_a.clear();
            cot_a.resize(n_p, zero);
            cot_l.clear();
            cot_l.resize(n_p, zero);
            let (x, delta, mu) = if col == 0 {
                // a = dx_i / dtheta_i
                (one, zero, zero)
            } else if col == 1 {
                // q_own = -mu_i - delta_i w_i
                (zero, -w_i, -one)
            } else if col < 1 + n_s {
                // q_j = -L[i,j] - A[i,j] w_i
                let j = col - 2;
                cot_a[j] = -w_i;
                cot_l[j] = -one;
                (zero, zero, zero)
            } else {
                // score_j = L[i,j] - x_i A[i,j]
                let j = col - 1 - n_s;
                cot_a[j] = -x_i;
                cot_l[j] = one;
                (-edge_a[j], zero, zero)
            };
            let cot = Cotangent {
                x,
                delta,
                mu,
                edge_a: cot_a,
                edge_l: cot_l,
            };
            let out = &mut rows[col * n_theta..(col + 1) * n_theta];
            self.local_pullback(simd, shape, theta, &cot, &source, s, out);
        }

        // Row `i` of `J` is `alpha = (delta_i, A[i, P(i)])` on `S_i`, so
        // `e_i = J^{-T} E_i alpha` and the direct part folds into the
        // solve: `dr/dtheta_i = J^{-T} E_i B'` with `B' = B + alpha a`.
        let (a, rest) = rows.split_at_mut(n_theta);
        axpy_lanes(simd, delta_i, a, &mut rest[..n_theta]);
        for j in 0..n_p {
            axpy_lanes(
                simd,
                edge_a[j],
                a,
                &mut rest[(1 + j) * n_theta..(2 + j) * n_theta],
            );
        }
        let reg = self.regularization.unwrap_or(0.0);
        for value in &mut rest[n_s * n_theta..] {
            *value *= reg;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn variable_blocks<S: Simd>(
        &self,
        simd: S,
        tape: &Tape,
        stores: &[S::f64s],
        n_store: usize,
        i: usize,
        max_block_size: usize,
        b: &mut BlockScratch<S>,
    ) -> Vec<GnBlock> {
        let shape = self.shape(i);
        let n_theta = shape.size();
        let n_p = shape.n_parent;
        let n_s = n_p + 1;
        let theta = &tape.theta[self.params(i)];
        let n_score = if self.regularization.is_some() {
            n_p
        } else {
            0
        };

        let mut blocks: Vec<GnBlock> = (0..n_theta)
            .step_by(max_block_size)
            .map(|start| GnBlock {
                start: self.param_offset[i] + start,
                matrix: Mat::zeros(
                    max_block_size.min(n_theta - start),
                    max_block_size.min(n_theta - start),
                ),
            })
            .collect();

        let mut k_mat = Mat::<f64>::zeros(n_s, n_s);
        let mut index = Vec::with_capacity(n_s);
        index.push(i);
        index.extend_from_slice(&self.parent_index[self.edges(i)]);

        // `G += P Q^T` over batches of draws, with the columns of `P` each
        // draw's `B'` and scaled score rows, and those of `Q` the matching
        // `K B'` and score rows: one matrix product per batch and sub-block,
        // instead of streaming every block once per row.
        let per_draw = n_s + n_score;
        let batch = BLOCK_DRAW_BATCH.min(self.n_draw);
        let mut p_mat = Mat::<f64>::zeros(n_theta, batch * per_draw);
        let mut q_mat = Mat::<f64>::zeros(n_theta, batch * per_draw);
        let mut in_batch = 0;

        for tile in 0..self.n_tile {
            self.block_rows(simd, i, theta, tile, tape, b);
            let store = &stores[tile * n_store..(tile + 1) * n_store];
            for lane in 0..self.width {
                let draw = tile * self.width + lane;

                // `B'` and the scores of this draw, skipping `a`.
                let first = in_batch * per_draw;
                for c in 0..per_draw {
                    let rows = &b.rows[(1 + c) * n_theta..(2 + c) * n_theta];
                    for (out, row) in p_mat.col_as_slice_mut(first + c).iter_mut().zip(rows) {
                        *out = row[lane];
                    }
                }

                // K = Sigma on {i} + P(i), and K B' (as columns, B' K).
                for r in 0..n_s {
                    for c in 0..n_s {
                        k_mat[(r, c)] = self.sigma(store, index[r], index[c])[lane];
                    }
                }
                matmul(
                    q_mat.as_mut().subcols_mut(first, n_s),
                    Accum::Replace,
                    p_mat.as_ref().subcols(first, n_s),
                    k_mat.as_ref(),
                    1.0,
                    Par::Seq,
                );
                q_mat
                    .as_mut()
                    .subcols_mut(first + n_s, n_score)
                    .copy_from(p_mat.as_ref().subcols(first + n_s, n_score));
                in_batch += 1;

                if in_batch == batch || draw + 1 == self.n_draw {
                    let n_cols = in_batch * per_draw;
                    let local_start = self.param_offset[i];
                    for block in &mut blocks {
                        let start = block.start - local_start;
                        let size = block.matrix.nrows();
                        matmul(
                            block.matrix.as_mut(),
                            Accum::Add,
                            p_mat.as_ref().subrows(start, size).subcols(0, n_cols),
                            q_mat
                                .as_ref()
                                .subrows(start, size)
                                .subcols(0, n_cols)
                                .transpose(),
                            1.0,
                            Par::Seq,
                        );
                    }
                    in_batch = 0;
                }
            }
        }

        // `P Q^T` is symmetric only up to rounding.
        let half_inv_n = 0.5 / self.n_draw as f64;
        for block in &mut blocks {
            let m = &mut block.matrix;
            for r in 0..m.nrows() {
                for c in 0..=r {
                    let value = (m[(r, c)] + m[(c, r)]) * half_inv_n;
                    m[(r, c)] = value;
                    m[(c, r)] = value;
                }
            }
        }
        blocks
    }
}

fn as_usize(values: &[i64]) -> Result<Vec<usize>> {
    values
        .iter()
        .map(|&v| usize::try_from(v).map_err(|_| anyhow::anyhow!("negative index {v}")))
        .collect()
}

/// Residuals of the LM fit of a `SparseTriangularMap` and their derivatives,
/// see `nutpie.triangular_lm`.
#[pyclass(name = "FisherResiduals")]
pub struct PyFisherResiduals {
    pub(crate) inner: FisherResiduals,
    tape: Option<Tape>,
}

impl PyFisherResiduals {
    fn tape(&self) -> Result<&Tape> {
        match &self.tape {
            Some(tape) => Ok(tape),
            None => bail!("no tape: call `residuals(theta, record=True)` first"),
        }
    }
}

#[pymethods]
impl PyFisherResiduals {
    #[new]
    #[pyo3(signature = (
        *,
        parent_indptr,
        parent_index,
        n_unit,
        n_par,
        location_index,
        transformer,
        fisher_regularization = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        parent_indptr: PyReadonlyArray1<'_, i64>,
        parent_index: PyReadonlyArray1<'_, i64>,
        n_unit: usize,
        n_par: usize,
        location_index: usize,
        transformer: &Bound<'_, PyAny>,
        fisher_regularization: Option<f64>,
    ) -> Result<Self> {
        let specs: Vec<LayerSpec> = pythonize::depythonize(transformer)?;
        Ok(Self {
            inner: FisherResiduals::new(
                as_usize(parent_indptr.as_slice()?)?,
                as_usize(parent_index.as_slice()?)?,
                n_unit,
                n_par,
                location_index,
                specs,
                fisher_regularization,
            )?,
            tape: None,
        })
    }

    #[getter]
    fn n_params(&self) -> usize {
        self.inner.n_params()
    }

    #[getter]
    fn n_residuals(&self) -> usize {
        self.inner.n_residuals()
    }

    #[getter]
    fn n_draw(&self) -> usize {
        self.inner.n_draw
    }

    /// Start of each variable's parameter slice, plus the total.
    #[getter]
    fn param_offsets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_vec(
            py,
            self.inner.param_offset.iter().map(|&o| o as i64).collect(),
        )
    }

    /// Set the map-space draws and gradients, each ``(n_draw, n_var)``
    /// flattened. Drops the tape.
    fn set_data(
        &mut self,
        y: PyReadonlyArray1<'_, f64>,
        g: PyReadonlyArray1<'_, f64>,
    ) -> Result<()> {
        self.tape = None;
        self.inner
            .set_data(y.as_slice()?.to_vec(), g.as_slice()?.to_vec())
    }

    /// Flattened ``(n_draw, n_residuals)`` residuals at `theta`; with
    /// `record`, keeps the tape the derivatives below use.
    #[pyo3(signature = (theta, record = true))]
    fn residuals<'py>(
        &mut self,
        py: Python<'py>,
        theta: PyReadonlyArray1<'py, f64>,
        record: bool,
    ) -> Result<Bound<'py, PyArray1<f64>>> {
        let theta = theta.as_slice()?.to_vec();
        let (out, tape) = py.detach(|| self.inner.residuals(&theta))?;
        if record {
            self.tape = Some(tape);
        }
        Ok(PyArray1::from_vec(py, out))
    }

    /// ``J v``, flattened ``(n_draw, n_residuals)``, at the recorded `theta`.
    fn pushforward<'py>(
        &self,
        py: Python<'py>,
        v: PyReadonlyArray1<'py, f64>,
    ) -> Result<Bound<'py, PyArray1<f64>>> {
        let v = v.as_slice()?.to_vec();
        let tape = self.tape()?;
        let out = py.detach(|| self.inner.pushforward(tape, &v))?;
        Ok(PyArray1::from_vec(py, out))
    }

    /// ``J^T r_bar`` for a flattened ``(n_draw, n_residuals)`` `r_bar`.
    fn pullback<'py>(
        &self,
        py: Python<'py>,
        r_bar: PyReadonlyArray1<'py, f64>,
    ) -> Result<Bound<'py, PyArray1<f64>>> {
        let r_bar = r_bar.as_slice()?.to_vec();
        let tape = self.tape()?;
        let out = py.detach(|| self.inner.pullback(tape, &r_bar))?;
        Ok(PyArray1::from_vec(py, out))
    }

    /// ``J^T J v``.
    fn gauss_newton_product<'py>(
        &self,
        py: Python<'py>,
        v: PyReadonlyArray1<'py, f64>,
    ) -> Result<Bound<'py, PyArray1<f64>>> {
        let v = v.as_slice()?.to_vec();
        let tape = self.tape()?;
        let out = py.detach(|| self.inner.gauss_newton_product(tape, &v))?;
        Ok(PyArray1::from_vec(py, out))
    }

    /// Exact Gauss-Newton sub-blocks: ``(starts, sizes, data)``, where block
    /// ``k`` covers parameters ``starts[k]:starts[k] + sizes[k]`` and is
    /// stored row-major in `data`, after the blocks before it.
    fn gauss_newton_blocks<'py>(
        &self,
        py: Python<'py>,
        max_block_size: usize,
    ) -> Result<(
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        let tape = self.tape()?;
        let blocks = py.detach(|| self.inner.gauss_newton_blocks(tape, max_block_size))?;
        let starts = blocks.iter().map(|b| b.start as i64).collect();
        let sizes = blocks.iter().map(|b| b.matrix.nrows() as i64).collect();
        // Symmetric, so row- and column-major agree.
        let data = blocks
            .iter()
            .flat_map(|b| {
                (0..b.matrix.ncols()).flat_map(move |c| b.matrix.col_as_slice(c).iter().copied())
            })
            .collect();
        Ok((
            PyArray1::from_vec(py, starts),
            PyArray1::from_vec(py, sizes),
            PyArray1::from_vec(py, data),
        ))
    }
}
