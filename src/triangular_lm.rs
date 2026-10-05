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

/// Draws per matrix product when accumulating the exact blocks.
const BLOCK_DRAW_BATCH: usize = 32;

/// A jet in a [`JetArena`]: a scalar function of `z = (y, pi)`, one lane per
/// draw, with its gradient and the products of its Hessian with the arena's
/// `k` fixed directions: `e[d] = g . dir_d`, `h[d] = H dir_d`.
///
/// Forward-over-forward, but sharing the value and gradient across the
/// directions, so every operation is O(n_z) instead of the O(n_z^2) a full
/// Hessian would cost. For the full Hessian, see [`JetArena::reset_hessian`].
///
/// Not `Copy`: an operation that takes a jet by value writes its result into
/// that jet's slot, so a value needed twice is copied explicitly with
/// [`JetArena::copy`].
#[must_use]
struct Jet(u32);

/// Jet slots of one size, `[v, g[n_z], e[k], h[k][n_z]]` each, or `[v,
/// g[n_z], H]` with `H` the full Hessian as a packed lower triangle.
/// [`Self::reset`] and [`Self::reset_hessian`] set the size for one
/// transformer evaluation and hand every slot back; the storage only grows,
/// so after the first evaluation nothing allocates.
struct JetArena<S: Simd> {
    simd: S,
    data: Vec<S::f64s>,
    n_z: usize,
    k: usize,
    /// Whether the slots hold full Hessians instead of directions.
    hessian: bool,
    slot_len: usize,
    n_slots: usize,
    /// Slots handed back by consuming binary operations.
    free: Vec<u32>,
}

impl<S: Simd> JetArena<S> {
    fn new(simd: S) -> Self {
        Self {
            simd,
            data: Vec::new(),
            n_z: 0,
            k: 0,
            hessian: false,
            slot_len: 0,
            n_slots: 0,
            free: Vec::new(),
        }
    }

    /// Starts an evaluation with `n_z` inputs and `k` directions.
    #[inline(always)]
    fn reset(&mut self, n_z: usize, k: usize) {
        self.n_z = n_z;
        self.k = k;
        self.hessian = false;
        self.slot_len = 1 + n_z * (k + 1) + k;
        self.n_slots = 0;
        self.free.clear();
    }

    /// Starts an evaluation with `n_z` inputs and full Hessians. Each row of
    /// a Hessian update reads only the same row of the operands' Hessians and
    /// their gradients, so the lower triangle is all that is computed.
    #[inline(always)]
    fn reset_hessian(&mut self, n_z: usize) {
        self.reset(n_z, 0);
        self.hessian = true;
        self.slot_len = 1 + n_z + packed_len(n_z);
    }

    #[inline(always)]
    fn splat(&self, value: f64) -> S::f64s {
        S::f64s::splat(self.simd, value)
    }

    #[inline(always)]
    fn range(&self, j: &Jet) -> std::ops::Range<usize> {
        let start = j.0 as usize * self.slot_len;
        start..start + self.slot_len
    }

    #[inline(always)]
    fn alloc(&mut self) -> Jet {
        if let Some(index) = self.free.pop() {
            return Jet(index);
        }
        let index = self.n_slots;
        self.n_slots += 1;
        let end = self.n_slots * self.slot_len;
        if self.data.len() < end {
            let zero = self.splat(0.0);
            self.data.resize(end, zero);
        }
        Jet(index as u32)
    }

    #[inline(always)]
    fn release(&mut self, j: Jet) {
        self.free.push(j.0);
    }

    #[inline(always)]
    fn value(&self, j: &Jet) -> S::f64s {
        self.data[self.range(j).start]
    }

    /// The gradient in `z`.
    #[inline(always)]
    fn grad(&self, j: &Jet) -> &[S::f64s] {
        let start = self.range(j).start + 1;
        &self.data[start..start + self.n_z]
    }

    /// The Hessian's product with direction `d`.
    #[inline(always)]
    fn hess_dir(&self, j: &Jet, d: usize) -> &[S::f64s] {
        let start = self.range(j).start + 1 + self.n_z + self.k + d * self.n_z;
        &self.data[start..start + self.n_z]
    }

    /// The full Hessian as a packed lower triangle, after [`Self::reset_hessian`].
    #[inline(always)]
    fn hessian(&self, j: &Jet) -> &[S::f64s] {
        let range = self.range(j);
        &self.data[range.start + 1 + self.n_z..range.end]
    }

    #[inline(always)]
    fn constant(&mut self, v: S::f64s) -> Jet {
        let j = self.alloc();
        let zero = self.splat(0.0);
        let range = self.range(&j);
        let slot = &mut self.data[range];
        slot.fill(zero);
        slot[0] = v;
        j
    }

    /// Input `index` of `z` at `v`, with `dirs` the `k` directions, `n_z`
    /// entries each.
    #[inline(always)]
    fn variable(&mut self, v: S::f64s, index: usize, dirs: &[S::f64s]) -> Jet {
        let j = self.constant(v);
        let (n_z, k) = (self.n_z, self.k);
        let one = self.splat(1.0);
        let range = self.range(&j);
        let slot = &mut self.data[range];
        slot[1 + index] = one;
        for d in 0..k {
            slot[1 + n_z + d] = dirs[d * n_z + index];
        }
        j
    }

    #[inline(always)]
    fn copy(&mut self, j: &Jet) -> Jet {
        let out = self.alloc();
        let (source, start) = (self.range(j), self.range(&out).start);
        self.data.copy_within(source, start);
        out
    }

    /// The slots of a binary operation: `a`'s to write, `b`'s to read.
    #[inline(always)]
    fn pair(&mut self, a: &Jet, b: &Jet) -> (&mut [S::f64s], &[S::f64s]) {
        let (range_a, range_b) = (self.range(a), self.range(b));
        let [a, b] = self
            .data
            .get_disjoint_mut([range_a, range_b])
            .expect("jets have distinct slots");
        (a, b)
    }

    /// `f(j)`, given `f`, `f'` and `f''` at its value.
    #[inline(always)]
    fn chain(&mut self, j: Jet, f0: S::f64s, f1: S::f64s, f2: S::f64s) -> Jet {
        if self.hessian {
            return self.chain_hessian(j, f0, f1, f2);
        }
        let (n_z, k) = (self.n_z, self.k);
        let range = self.range(&j);
        let (head, h) = self.data[range].split_at_mut(1 + n_z + k);
        let (v, rest) = head.split_at_mut(1);
        let (g, e) = rest.split_at_mut(n_z);
        // `h` reads the old `g` and `e`, so it goes first.
        for (e, h) in e.iter_mut().zip(h.chunks_exact_mut(n_z)) {
            let curvature = f2 * *e;
            for (h, &g) in h.iter_mut().zip(&*g) {
                *h = f1 * *h + curvature * g;
            }
            *e = f1 * *e;
        }
        for g in g.iter_mut() {
            *g = f1 * *g;
        }
        v[0] = f0;
        j
    }

    /// [`Self::chain`] on full Hessians: `H' = f' H + f'' g g^T`.
    #[inline(always)]
    fn chain_hessian(&mut self, j: Jet, f0: S::f64s, f1: S::f64s, f2: S::f64s) -> Jet {
        let n_z = self.n_z;
        let range = self.range(&j);
        let (head, h) = self.data[range].split_at_mut(1 + n_z);
        let (v, g) = head.split_at_mut(1);
        let mut rows = h;
        for r in 0..n_z {
            let (row, rest) = rows.split_at_mut(r + 1);
            let curvature = f2 * g[r];
            for (h, &g) in row.iter_mut().zip(&*g) {
                *h = f1 * *h + curvature * g;
            }
            rows = rest;
        }
        for g in g.iter_mut() {
            *g = f1 * *g;
        }
        v[0] = f0;
        j
    }

    #[inline(always)]
    fn add(&mut self, a: Jet, b: Jet) -> Jet {
        let (a_slot, b_slot) = self.pair(&a, &b);
        for (a, &b) in a_slot.iter_mut().zip(b_slot) {
            *a += b;
        }
        self.release(b);
        a
    }

    #[inline(always)]
    fn sub(&mut self, a: Jet, b: Jet) -> Jet {
        let (a_slot, b_slot) = self.pair(&a, &b);
        for (a, &b) in a_slot.iter_mut().zip(b_slot) {
            *a -= b;
        }
        self.release(b);
        a
    }

    #[inline(always)]
    fn mul(&mut self, a: Jet, b: Jet) -> Jet {
        if self.hessian {
            return self.mul_hessian(a, b);
        }
        let (n_z, k) = (self.n_z, self.k);
        let (a_slot, b_slot) = self.pair(&a, &b);
        let (a_head, a_h) = a_slot.split_at_mut(1 + n_z + k);
        let (b_head, b_h) = b_slot.split_at(1 + n_z + k);
        let (a_v, a_rest) = a_head.split_at_mut(1);
        let (a_g, a_e) = a_rest.split_at_mut(n_z);
        let (b_g, b_e) = b_head[1..].split_at(n_z);
        let (av, bv) = (a_v[0], b_head[0]);
        // `h` reads the old `g` and `e` of both, so it goes first.
        for ((&ae, &be), (a_h, b_h)) in a_e
            .iter()
            .zip(b_e)
            .zip(a_h.chunks_exact_mut(n_z).zip(b_h.chunks_exact(n_z)))
        {
            for ((ah, &bh), (&ag, &bg)) in a_h.iter_mut().zip(b_h).zip(a_g.iter().zip(b_g)) {
                *ah = av * bh + bv * *ah + ae * bg + be * ag;
            }
        }
        for (ae, &be) in a_e.iter_mut().zip(b_e) {
            *ae = av * be + bv * *ae;
        }
        for (ag, &bg) in a_g.iter_mut().zip(b_g) {
            *ag = av * bg + bv * *ag;
        }
        a_v[0] = av * bv;
        self.release(b);
        a
    }

    /// [`Self::mul`] on full Hessians: `H = a H_b + b H_a + g_a g_b^T + g_b
    /// g_a^T`.
    #[inline(always)]
    fn mul_hessian(&mut self, a: Jet, b: Jet) -> Jet {
        let n_z = self.n_z;
        let (a_slot, b_slot) = self.pair(&a, &b);
        let (a_head, a_h) = a_slot.split_at_mut(1 + n_z);
        let (b_head, b_h) = b_slot.split_at(1 + n_z);
        let (a_v, a_g) = a_head.split_at_mut(1);
        let b_g = &b_head[1..];
        let (av, bv) = (a_v[0], b_head[0]);
        let mut index = 0;
        for r in 0..n_z {
            let (ag_r, bg_r) = (a_g[r], b_g[r]);
            let (a_row, b_row) = (&mut a_h[index..=index + r], &b_h[index..=index + r]);
            for ((ah, &bh), (&ag, &bg)) in a_row.iter_mut().zip(b_row).zip(a_g.iter().zip(b_g)) {
                *ah = av * bh + bv * *ah + ag_r * bg + bg_r * ag;
            }
            index += r + 1;
        }
        for (ag, &bg) in a_g.iter_mut().zip(b_g) {
            *ag = av * bg + bv * *ag;
        }
        a_v[0] = av * bv;
        self.release(b);
        a
    }

    #[inline(always)]
    fn scale(&mut self, j: Jet, c: f64) -> Jet {
        let range = self.range(&j);
        for value in &mut self.data[range] {
            *value *= c;
        }
        j
    }

    #[inline(always)]
    fn neg(&mut self, j: Jet) -> Jet {
        self.scale(j, -1.0)
    }

    #[inline(always)]
    fn add_const(&mut self, j: Jet, c: f64) -> Jet {
        let start = self.range(&j).start;
        self.data[start] += c;
        j
    }

    #[inline(always)]
    fn asinh(&mut self, j: Jet) -> Jet {
        let a = self.value(&j);
        let root = simd_math::hypot_one(self.simd, a);
        let value = simd_math::asinh(self.simd, a);
        let f1 = self.splat(1.0) / root;
        self.chain(j, value, f1, -a / (root * root * root))
    }

    #[inline(always)]
    fn sinh(&mut self, j: Jet) -> Jet {
        let (s, c) = simd_math::sinh_cosh(self.simd, self.value(&j));
        self.chain(j, s, c, s)
    }

    #[inline(always)]
    fn cosh(&mut self, j: Jet) -> Jet {
        let (s, c) = simd_math::sinh_cosh(self.simd, self.value(&j));
        self.chain(j, c, s, c)
    }

    /// `v + sqrt(1 + v*v)`, in `exp_asinh`'s cancellation-free form.
    #[inline(always)]
    fn positive(&mut self, j: Jet) -> Jet {
        let a = self.value(&j);
        let one = self.splat(1.0);
        let root = simd_math::hypot_one(self.simd, a);
        let value = a.simd_ge(0.0).select(a + root, one / (root - a));
        // d/da (value / root) = (value (root - a)) / root^3 = 1 / root^3.
        self.chain(j, value, value / root, one / (root * root * root))
    }

    #[inline(always)]
    fn exp(&mut self, j: Jet) -> Jet {
        let e = simd_math::exp(self.simd, self.value(&j));
        self.chain(j, e, e, e)
    }

    #[inline(always)]
    fn ln_1p(&mut self, j: Jet) -> Jet {
        let v = self.value(&j);
        let inv = self.splat(1.0) / (v + 1.0);
        let value = simd_math::ln_1p(self.simd, v);
        self.chain(j, value, inv, -inv * inv)
    }

    /// `log(cosh(v))`, in `_log_cosh`'s stable form.
    #[inline(always)]
    fn log_cosh(&mut self, j: Jet) -> Jet {
        let (value, t) = simd_math::log_cosh_tanh(self.simd, self.value(&j));
        let f2 = self.splat(1.0) - t * t;
        self.chain(j, value, t, f2)
    }

    #[inline(always)]
    fn sigmoid(&mut self, j: Jet) -> Jet {
        let one = self.splat(1.0);
        let s = one / (simd_math::exp(self.simd, -self.value(&j)) + 1.0);
        let ds = s * (one - s);
        self.chain(j, s, ds, ds * (one - s * 2.0))
    }

    #[inline(always)]
    fn square(&mut self, j: Jet) -> Jet {
        let v = self.value(&j);
        let two = self.splat(2.0);
        self.chain(j, v * v, v * 2.0, two)
    }
}

/// The transformer parameters `pi` of one variable as jet inputs.
struct Fields<'a, S: Simd> {
    pi: &'a [S::f64s],
    dirs: &'a [S::f64s],
}

impl<S: Simd> Fields<'_, S> {
    #[inline(always)]
    fn get(&self, ws: &mut JetArena<S>, param: Option<Param>) -> Option<Jet> {
        param.map(|p| ws.variable(self.pi[p.index] + p.offset, 1 + p.index, self.dirs))
    }

    #[inline(always)]
    fn get_or_zero(&self, ws: &mut JetArena<S>, param: Option<Param>) -> Jet {
        match self.get(ws, param) {
            Some(jet) => jet,
            None => {
                let zero = ws.splat(0.0);
                ws.constant(zero)
            }
        }
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
    fn inverse<S: Simd>(&self, ws: &mut JetArena<S>, fields: &Fields<S>, x: Jet) -> (Jet, Jet) {
        let log_gamma = fields.get(ws, self.alpha).map(|alpha| {
            let log_gamma = ws.asinh(alpha);
            match self.bound {
                None => log_gamma,
                Some(b) => {
                    let t = ws.scale(log_gamma, b.slope);
                    let t = ws.add_const(t, b.offset);
                    let t = ws.sigmoid(t);
                    let t = ws.scale(t, b.width);
                    ws.add_const(t, b.low)
                }
            }
        });
        let log_delta = fields.get(ws, self.beta).map(|beta| ws.asinh(beta));
        let log_sigma = fields.get(ws, self.sigma).map(|sigma| ws.asinh(sigma));

        let centred = match fields.get(ws, self.mu) {
            Some(mu) => ws.sub(x, mu),
            None => x,
        };
        // `log_gamma` and `log_sigma` are used again below.
        let log_scale = match (&log_gamma, &log_sigma) {
            (Some(g), Some(s)) => {
                let (g, s) = (ws.copy(g), ws.copy(s));
                Some(ws.sub(g, s))
            }
            (Some(g), None) => Some(ws.copy(g)),
            (None, Some(s)) => {
                let s = ws.copy(s);
                Some(ws.neg(s))
            }
            (None, None) => None,
        };
        let half_a = match log_scale {
            Some(scale) => {
                let scale = ws.exp(scale);
                let scaled = ws.mul(scale, centred);
                ws.scale(scaled, 0.5)
            }
            None => ws.scale(centred, 0.5),
        };
        let arg = ws.copy(&half_a);
        let arg = ws.asinh(arg);
        let shifted = match log_delta {
            Some(delta) => {
                let delta = ws.scale(delta, 2.0);
                ws.sub(arg, delta)
            }
            None => arg,
        };
        let u = match log_gamma {
            Some(g) => {
                let g = ws.neg(g);
                let factor = ws.exp(g);
                ws.mul(shifted, factor)
            }
            None => shifted,
        };

        let sinh_u = ws.copy(&u);
        let sinh_u = ws.sinh(sinh_u);
        let mut out = ws.scale(sinh_u, 2.0);
        if let Some(nu) = fields.get(ws, self.nu) {
            out = ws.add(out, nu);
        }
        let log_cosh_u = ws.log_cosh(u);
        let correction = ws.square(half_a);
        let correction = ws.ln_1p(correction);
        let correction = ws.scale(correction, 0.5);
        let mut ld = ws.sub(log_cosh_u, correction);
        if let Some(s) = log_sigma {
            ld = ws.sub(ld, s);
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
fn tangent_sas_inverse<S: Simd>(
    layer: &TangentSasSpec,
    ws: &mut JetArena<S>,
    fields: &Fields<S>,
    x: Jet,
) -> (Jet, Jet) {
    let nu = fields.get_or_zero(ws, layer.nu);
    let eps = fields.get_or_zero(ws, layer.eps);
    let b_raw = fields.get_or_zero(ws, layer.b);
    let r_raw = fields.get_or_zero(ws, layer.r);

    // q = (y - nu) * sigmoid(b_raw) * positive(r_raw) * cosh(eps) + sinh(eps)
    let centred = ws.copy(&nu);
    let q = ws.sub(x, centred);
    let factor = ws.copy(&b_raw);
    let factor = ws.sigmoid(factor);
    let q = ws.mul(q, factor);
    let factor = ws.copy(&r_raw);
    let factor = ws.positive(factor);
    let q = ws.mul(q, factor);
    let factor = ws.copy(&eps);
    let factor = ws.cosh(factor);
    let q = ws.mul(q, factor);
    let shift = ws.copy(&eps);
    let shift = ws.sinh(shift);
    let q = ws.add(q, shift);

    // a = (asinh(q) - eps) * positive(-r_raw)
    let a = ws.copy(&q);
    let a = ws.asinh(a);
    let shift = ws.copy(&eps);
    let a = ws.sub(a, shift);
    let r = ws.neg(r_raw);
    let r = ws.positive(r);
    let a = ws.mul(a, r);

    // out = sinh(a) * (exp(-b_raw) + 1) + nu
    let out = ws.copy(&a);
    let out = ws.sinh(out);
    let b = ws.neg(b_raw);
    let b = ws.exp(b);
    let b = ws.add_const(b, 1.0);
    let out = ws.mul(out, b);
    let out = ws.add(out, nu);

    // ld = log_cosh(a) + log_cosh(eps) - 0.5 log1p(q^2)
    let ld = ws.log_cosh(a);
    let eps = ws.log_cosh(eps);
    let ld = ws.add(ld, eps);
    let q = ws.square(q);
    let q = ws.ln_1p(q);
    let q = ws.scale(q, 0.5);
    let ld = ws.sub(ld, q);
    (out, ld)
}

/// `PositiveAffine.inverse_and_log_det`: `x = (y - loc) / scale_mod`, and
/// `1 / scale_mod = positive(-scale)`.
#[inline(always)]
fn positive_affine_inverse<S: Simd>(
    layer: &PositiveAffineSpec,
    ws: &mut JetArena<S>,
    fields: &Fields<S>,
    x: Jet,
) -> (Jet, Jet) {
    let centred = match fields.get(ws, layer.loc) {
        Some(loc) => ws.sub(x, loc),
        None => x,
    };
    match fields.get(ws, layer.scale) {
        Some(scale) => {
            let factor = ws.copy(&scale);
            let factor = ws.neg(factor);
            let factor = ws.positive(factor);
            let out = ws.mul(centred, factor);
            let ld = ws.asinh(scale);
            (out, ws.neg(ld))
        }
        None => (centred, fields.get_or_zero(ws, None)),
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
/// `z = (y, pi)` along the `k` directions `dirs`, `1 + pi.len()` entries each.
/// Resets `ws`, so earlier jets in it are invalid afterwards.
#[simd]
fn transformer<S: Simd>(
    simd: S,
    layers: &[Layer],
    y: S::f64s,
    pi: &[S::f64s],
    dirs: &[S::f64s],
    k: usize,
    ws: &mut JetArena<S>,
) -> (Jet, Jet) {
    ws.reset(1 + pi.len(), k);
    evaluate_transformer(simd, layers, y, pi, dirs, ws)
}

/// [`transformer`] with full Hessians, see [`JetArena::reset_hessian`].
#[simd]
fn transformer_hessian<S: Simd>(
    simd: S,
    layers: &[Layer],
    y: S::f64s,
    pi: &[S::f64s],
    ws: &mut JetArena<S>,
) -> (Jet, Jet) {
    ws.reset_hessian(1 + pi.len());
    evaluate_transformer(simd, layers, y, pi, &[], ws)
}

#[inline(always)]
fn evaluate_transformer<S: Simd>(
    simd: S,
    layers: &[Layer],
    y: S::f64s,
    pi: &[S::f64s],
    dirs: &[S::f64s],
    ws: &mut JetArena<S>,
) -> (Jet, Jet) {
    let fields = Fields { pi, dirs };
    let mut x = ws.variable(y, 0, dirs);
    let mut log_det = ws.constant(S::f64s::splat(simd, 0.0));
    for layer in layers.iter().rev() {
        let (out, ld) = match layer {
            Layer::Contract2(layer) => layer.inverse(ws, &fields, x),
            Layer::TangentSas(layer) => tangent_sas_inverse(layer, ws, &fields, x),
            Layer::PositiveAffine(layer) => positive_affine_inverse(layer, ws, &fields, x),
        };
        x = out;
        log_det = ws.add(log_det, ld);
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
    jets: JetArena<S>,
    /// Up to two jet directions, `n_z` entries each.
    dirs: Vec<S::f64s>,
    /// The transformer's gradients in `z`, and a pullback's Hessian products
    /// (see [`FisherResiduals::curvature`]).
    grad_t: Vec<S::f64s>,
    grad_l: Vec<S::f64s>,
    prod_t: Vec<S::f64s>,
    prod_l: Vec<S::f64s>,
}

impl<S: Simd> Scratch<S> {
    fn new(simd: S, n_unit: usize, n_par: usize, max_parent: usize) -> Self {
        let zero = S::f64s::splat(simd, 0.0);
        let n_z = 1 + n_par;
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
            jets: JetArena::new(simd),
            dirs: vec![zero; 2 * n_z],
            grad_t: vec![zero; n_z],
            grad_l: vec![zero; n_z],
            prod_t: vec![zero; n_z],
            prod_l: vec![zero; n_z],
        }
    }
}

/// Length of a packed lower triangle of an `n x n` matrix.
fn packed_len(n: usize) -> usize {
    n * (n + 1) / 2
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

/// Per-thread scratch for [`FisherResiduals::variable_blocks`]. The matrices
/// are column-major.
struct BlockScratch<S: Simd> {
    local: Scratch<S>,
    /// The Jacobian rows of one tile, `n_theta` per column.
    rows: Vec<S::f64s>,
    edge_a: Vec<S::f64s>,
    /// `M = dpi/dy_parents`, `(n_par, n_parent)`.
    pi_jac: Vec<S::f64s>,
    /// `H_T[pi, pi] M` and `H_Lambda[pi, pi] M`, `(n_par, n_parent)`.
    hess_t_jac: Vec<S::f64s>,
    hess_l_jac: Vec<S::f64s>,
    /// Every seed's `pi_bar`, `(n_par, n_col)`, and `W2^T pi_bar`,
    /// `(n_unit, n_col)`.
    pi_bar: Vec<S::f64s>,
    unit_bar: Vec<S::f64s>,
    /// `w2_u . t` and `w2_u . l` per unit, and one seed's `alpha t + beta l`.
    unit_t: Vec<S::f64s>,
    unit_l: Vec<S::f64s>,
    edge_dir: Vec<S::f64s>,
    grad_t: Vec<S::f64s>,
    grad_l: Vec<S::f64s>,
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
        let loc = self.location;
        self.conditioner(simd, shape, theta, s);
        let (x, log_det) = transformer(simd, &self.layers, y_own, &s.pi, &[], 0, &mut s.jets);
        let ws = &s.jets;
        let (t, l) = (&ws.grad(&x)[1..], &ws.grad(&log_det)[1..]);

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
        (ws.value(&x), ws.grad(&x)[0], ws.grad(&log_det)[0])
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

        s.dirs[0] = S::f64s::splat(simd, 0.0);
        s.dirs[1..n_z].copy_from_slice(&s.pi_dot);
        let (x, log_det) = transformer(
            simd,
            &self.layers,
            y_own,
            &s.pi,
            &s.dirs[..n_z],
            1,
            &mut s.jets,
        );
        let ws = &s.jets;
        let (t, l) = (&ws.grad(&x)[1..], &ws.grad(&log_det)[1..]);
        let (t_dot, l_dot) = (&ws.hess_dir(&x, 0)[1..], &ws.hess_dir(&log_det, 0)[1..]);

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
    fn curvature<S: Simd>(&self, simd: S, y: S::f64s, s: &mut Scratch<S>) {
        let n_z = 1 + self.n_par;
        let (x, log_det) = transformer(
            simd,
            &self.layers,
            y,
            &s.pi,
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
        y_own: S::f64s,
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
        s.dirs[0] = cot.delta;
        s.dirs[1..n_z].copy_from_slice(&s.t_bar);
        s.dirs[n_z] = cot.mu;
        s.dirs[n_z + 1..2 * n_z].copy_from_slice(&s.l_bar);
        self.curvature(simd, y_own, s);
        let (t, l) = (&s.grad_t[1..], &s.grad_l[1..]);
        for m in 0..self.n_par {
            s.pi_bar[m] = cot.x * t[m] + s.prod_t[1 + m] + s.prod_l[1 + m];
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

    /// The transformer's gradients in `z = (y, pi)`, at `y` and `s.pi`, and its
    /// full Hessians as packed lower triangles, from one jet pass.
    #[simd]
    #[allow(clippy::too_many_arguments)]
    fn dense_curvature<S: Simd>(
        &self,
        simd: S,
        y: S::f64s,
        s: &mut Scratch<S>,
        grad_t: &mut [S::f64s],
        grad_l: &mut [S::f64s],
        hess_t: &mut [S::f64s],
        hess_l: &mut [S::f64s],
    ) {
        let (x, log_det) = transformer_hessian(simd, &self.layers, y, &s.pi, &mut s.jets);
        let ws = &s.jets;
        grad_t.copy_from_slice(ws.grad(&x));
        grad_l.copy_from_slice(ws.grad(&log_det));
        hess_t.copy_from_slice(ws.hessian(&x));
        hess_l.copy_from_slice(ws.hessian(&log_det));
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
            self.local_pullback(simd, shape, theta_i, &cot, y[i], local, &mut grad[params]);
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
                    pi_jac: Vec::new(),
                    hess_t_jac: Vec::new(),
                    hess_l_jac: Vec::new(),
                    pi_bar: Vec::new(),
                    unit_bar: Vec::new(),
                    unit_t: vec![zero; self.n_unit],
                    unit_l: vec![zero; self.n_unit],
                    edge_dir: vec![zero; self.n_par],
                    grad_t: vec![zero; n_z],
                    grad_l: vec![zero; n_z],
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
            pi_jac,
            hess_t_jac,
            hess_l_jac,
            pi_bar,
            unit_bar,
            unit_t,
            unit_l,
            edge_dir,
            grad_t,
            grad_l,
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
        self.dense_curvature(simd, y_own, s, grad_t, grad_l, hess_t, hess_l);
        let (n_unit, n_par, loc) = (shape.n_unit, shape.n_par, self.location);
        let (t, l) = (&grad_t[1..], &grad_l[1..]);

        // Each Jacobian row is the local pullback (see `local_pullback`) of
        // one seed `(x_bar, delta_bar, mu_bar)` plus, for an edge `j`, the
        // one-hot cotangents `alpha e_j` on `A[i, :]` and `beta e_j` on
        // `L[i, :]`. Through the units these collapse onto the columns of
        // `M = dpi/dy_parents = W2 diag(h') W1 + e_loc skip^T`: `t_bar =
        // alpha M e_j` and `l_bar = beta M e_j`. So all seeds together need
        // the products `M`, `H[pi, pi] M` and `W2^T pi_bar`.
        let seed = |col: usize| {
            if col == 0 {
                // a = dx_i / dtheta_i
                (one, zero, zero, None)
            } else if col == 1 {
                // q_own = -mu_i - delta_i w_i
                (zero, -w_i, -one, None)
            } else if col < 1 + n_s {
                // q_j = -L[i,j] - A[i,j] w_i
                (zero, zero, zero, Some((col - 2, -w_i, -one)))
            } else {
                // score_j = L[i,j] - x_i A[i,j]
                let j = col - 1 - n_s;
                (-edge_a[j], zero, zero, Some((j, -x_i, one)))
            }
        };

        pi_jac.clear();
        pi_jac.resize(n_par * n_p, zero);
        for u in 0..n_unit {
            let (w1, w2) = (&theta[shape.w1(u)], &theta[shape.w2(u)]);
            for (column, &w1) in pi_jac.chunks_exact_mut(n_par).zip(w1) {
                axpy(simd, s.h1[u] * w1, w2, column);
            }
        }
        for (column, &skip) in pi_jac.chunks_exact_mut(n_par).zip(&theta[shape.skip()]) {
            column[loc] += skip;
        }

        // `H[pi, pi] M`, from the packed lower triangles.
        for jac in [&mut *hess_t_jac, &mut *hess_l_jac] {
            jac.clear();
            jac.resize(n_par * n_p, zero);
        }
        for r in 0..n_par {
            let row = packed_len(1 + r) + 1;
            for c in 0..=r {
                let (h_t, h_l) = (hess_t[row + c], hess_l[row + c]);
                for j in 0..n_p {
                    let (rj, cj) = (j * n_par + r, j * n_par + c);
                    hess_t_jac[rj] += h_t * pi_jac[cj];
                    hess_l_jac[rj] += h_l * pi_jac[cj];
                    if c < r {
                        hess_t_jac[cj] += h_t * pi_jac[rj];
                        hess_l_jac[cj] += h_l * pi_jac[rj];
                    }
                }
            }
        }

        // pi_bar = x_bar t + H[pi, z] (delta_bar, t_bar) + H_Lambda[pi, z]
        // (mu_bar, l_bar)
        let n_col = 1 + n_s + n_score;
        pi_bar.clear();
        pi_bar.resize(n_col * n_par, zero);
        for (col, pi) in pi_bar.chunks_exact_mut(n_par).enumerate() {
            let (x_bar, delta_bar, mu_bar, edge) = seed(col);
            for (m, value) in pi.iter_mut().enumerate() {
                let r = packed_len(1 + m);
                *value = x_bar * t[m] + delta_bar * hess_t[r] + mu_bar * hess_l[r];
            }
            if let Some((j, alpha, beta)) = edge {
                let column = j * n_par..(j + 1) * n_par;
                axpy_lanes(simd, alpha, &hess_t_jac[column.clone()], pi);
                axpy_lanes(simd, beta, &hess_l_jac[column], pi);
            }
        }

        unit_bar.clear();
        unit_bar.resize(n_col * n_unit, zero);
        for u in 0..n_unit {
            let w2 = &theta[shape.w2(u)];
            unit_t[u] = dot(simd, w2, t);
            unit_l[u] = dot(simd, w2, l);
            for (col, pi) in pi_bar.chunks_exact(n_par).enumerate() {
                unit_bar[col * n_unit + u] = dot(simd, w2, pi);
            }
        }

        rows.clear();
        rows.resize(n_col * n_theta, zero);
        let columns = rows
            .chunks_exact_mut(n_theta)
            .zip(pi_bar.chunks_exact(n_par));
        for (col, (out, pi)) in columns.enumerate() {
            let edge = seed(col).3;
            if let Some((_, alpha, beta)) = edge {
                for ((dir, &t), &l) in edge_dir.iter_mut().zip(t).zip(l) {
                    *dir = alpha * t + beta * l;
                }
            }
            for u in 0..n_unit {
                let (h, h1) = (s.h[u], s.h1[u]);
                let mut a_bar = h1 * unit_bar[col * n_unit + u];
                let out_w2 = &mut out[shape.w2(u)];
                for (out, &pi) in out_w2.iter_mut().zip(pi) {
                    *out = h * pi;
                }
                // The edge cotangents' paths through `W1[u, j]`.
                let mut w1_direct = None;
                if let Some((j, alpha, beta)) = edge {
                    let w1 = theta[shape.w1(u).start + j];
                    let p = alpha * unit_t[u] + beta * unit_l[u];
                    a_bar += s.h2[u] * p * w1;
                    axpy_lanes(simd, h1 * w1, &edge_dir[..], out_w2);
                    w1_direct = Some((j, h1 * p));
                }
                let out_w1 = &mut out[shape.w1(u)];
                for (out, &y) in out_w1.iter_mut().zip(&s.y_parents) {
                    *out = a_bar * y;
                }
                if let Some((j, value)) = w1_direct {
                    out_w1[j] += value;
                }
                out[shape.b1(u)] = a_bar;
            }
            out[shape.b2()].copy_from_slice(pi);
            let out_skip = &mut out[shape.skip()];
            for (out, &y) in out_skip.iter_mut().zip(&s.y_parents) {
                *out = pi[loc] * y;
            }
            if let Some((j, alpha, beta)) = edge {
                out_skip[j] += alpha * t[loc] + beta * l[loc];
            }
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
