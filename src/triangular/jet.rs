//! Forward-mode second-order automatic differentiation on SIMD vectors.
//!
//! The LM fit needs the transformer's value, gradient and some second-order
//! information in `z = (y, pi)`: the map input `y` and the `n_par`
//! transformer parameters, so `n_z = 1 + n_par` inputs. The transformer is a
//! short chain of scalar operations, so the derivatives are carried forward
//! through it: every intermediate value is a [`Jet`], and every operation
//! updates the value and its derivatives at once by the chain rule. Each
//! entry is one `S::f64s` vector, one lane per draw, so a pass evaluates a
//! whole tile of draws.
//!
//! # Two modes
//!
//! A jet's slot in the [`JetArena`] holds, for a function `f(z)`:
//!
//! | mode | set by | slot | cost per operation |
//! |---|---|---|---|
//! | directions | [`JetArena::reset`]`(n_z, k)` | `[f, g[n_z], e[k], h[k][n_z]]` | `O(n_z (k + 1))` |
//! | full Hessian | [`JetArena::reset_hessian`]`(n_z)` | `[f, g[n_z], H]` | `O(n_z^2 / 2)` |
//!
//! with `g = grad f`, and for fixed directions `dir_d`, `e[d] = g . dir_d`
//! and `h[d] = H dir_d`. `H` is the full Hessian as a packed lower triangle,
//! row `r` at [`packed_len`]`(r)`. Directions give Hessian-vector products
//! for `k` directions known before the pass (the pushforward needs one, the
//! pullback two); the full Hessian is for when every product is needed (the
//! exact Gauss-Newton blocks).
//!
//! Most operations treat the slot as one flat vector (`add`, `sub`, `scale`,
//! `copy`) or touch only the value (`add_const`), so they are the same in
//! both modes. Only the two second-order rules differ: [`JetArena::chain`]
//! for `F(f)`, `H' = F' H + F'' g g^T`, and [`JetArena::mul`] for `f1 f2`,
//! `H = f1 H2 + f2 H1 + g1 g2^T + g2 g1^T`. In the full mode, row `r` of
//! either update reads only row `r` of the inputs' Hessians and entries of
//! their gradients, so the lower triangle is all that is computed.
//!
//! # Using the arena
//!
//! Jets are handles to slots, and the arena is their only storage, so an
//! evaluation allocates nothing once the arena has grown to fit it.
//!
//! - A reset starts an evaluation and invalidates every earlier jet.
//! - [`JetArena::variable`] creates input `index` of `z` (with its entries of
//!   the directions), [`JetArena::constant`] a value with no derivatives.
//! - Operations consume their arguments: a unary operation rewrites its
//!   argument's slot in place, a binary one writes into its first argument's
//!   slot and frees the second. [`Jet`] is not `Copy`, so a value used twice
//!   is duplicated with [`JetArena::copy`] first.
//! - The results are read with [`JetArena::value`], [`JetArena::grad`],
//!   [`JetArena::hess_dir`] and [`JetArena::hessian`].
//!
//! For example, `f = x exp(p)` with `z = (x, p)` and its full Hessian:
//!
//! ```ignore
//! ws.reset_hessian(2);
//! let x = ws.variable(x_value, 0, &[]);
//! let p = ws.variable(p_value, 1, &[]);
//! let e = ws.exp(p);
//! let f = ws.mul(x, e);
//! // grad = (e^p, x e^p), packed H = [0, e^p, x e^p]
//! let (g, h) = (ws.grad(&f), ws.hessian(&f));
//! ```

use fearless_simd::prelude::*;

use crate::simd_math;

/// Length of a packed lower triangle of an `n x n` matrix.
pub(crate) fn packed_len(n: usize) -> usize {
    n * (n + 1) / 2
}

/// A handle to a slot of a [`JetArena`], see the module docs.
///
/// Not `Copy`: an operation that takes a jet by value writes its result into
/// that jet's slot, so a value needed twice is copied explicitly with
/// [`JetArena::copy`].
#[must_use]
pub(crate) struct Jet(u32);

/// Jet slots of one size, laid out by the mode (see the module docs).
/// [`Self::reset`] and [`Self::reset_hessian`] set the size for one
/// evaluation and hand every slot back; the storage only grows, so after the
/// first evaluation nothing allocates.
pub(crate) struct JetArena<S: Simd> {
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
    pub(crate) fn new(simd: S) -> Self {
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
    pub(crate) fn reset(&mut self, n_z: usize, k: usize) {
        self.n_z = n_z;
        self.k = k;
        self.hessian = false;
        self.slot_len = 1 + n_z * (k + 1) + k;
        self.n_slots = 0;
        self.free.clear();
    }

    /// Starts an evaluation with `n_z` inputs and full Hessians.
    #[inline(always)]
    pub(crate) fn reset_hessian(&mut self, n_z: usize) {
        self.reset(n_z, 0);
        self.hessian = true;
        self.slot_len = 1 + n_z + packed_len(n_z);
    }

    #[inline(always)]
    pub(crate) fn splat(&self, value: f64) -> S::f64s {
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
    pub(crate) fn value(&self, j: &Jet) -> S::f64s {
        self.data[self.range(j).start]
    }

    /// The gradient in `z`.
    #[inline(always)]
    pub(crate) fn grad(&self, j: &Jet) -> &[S::f64s] {
        let start = self.range(j).start + 1;
        &self.data[start..start + self.n_z]
    }

    /// The Hessian's product with direction `d`.
    #[inline(always)]
    pub(crate) fn hess_dir(&self, j: &Jet, d: usize) -> &[S::f64s] {
        let start = self.range(j).start + 1 + self.n_z + self.k + d * self.n_z;
        &self.data[start..start + self.n_z]
    }

    /// The full Hessian as a packed lower triangle, after [`Self::reset_hessian`].
    #[inline(always)]
    pub(crate) fn hessian(&self, j: &Jet) -> &[S::f64s] {
        let range = self.range(j);
        &self.data[range.start + 1 + self.n_z..range.end]
    }

    #[inline(always)]
    pub(crate) fn constant(&mut self, v: S::f64s) -> Jet {
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
    pub(crate) fn variable(&mut self, v: S::f64s, index: usize, dirs: &[S::f64s]) -> Jet {
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
    pub(crate) fn copy(&mut self, j: &Jet) -> Jet {
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
    pub(crate) fn add(&mut self, a: Jet, b: Jet) -> Jet {
        let (a_slot, b_slot) = self.pair(&a, &b);
        for (a, &b) in a_slot.iter_mut().zip(b_slot) {
            *a += b;
        }
        self.release(b);
        a
    }

    #[inline(always)]
    pub(crate) fn sub(&mut self, a: Jet, b: Jet) -> Jet {
        let (a_slot, b_slot) = self.pair(&a, &b);
        for (a, &b) in a_slot.iter_mut().zip(b_slot) {
            *a -= b;
        }
        self.release(b);
        a
    }

    #[inline(always)]
    pub(crate) fn mul(&mut self, a: Jet, b: Jet) -> Jet {
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
    pub(crate) fn scale(&mut self, j: Jet, c: f64) -> Jet {
        let range = self.range(&j);
        for value in &mut self.data[range] {
            *value *= c;
        }
        j
    }

    #[inline(always)]
    pub(crate) fn neg(&mut self, j: Jet) -> Jet {
        self.scale(j, -1.0)
    }

    #[inline(always)]
    pub(crate) fn add_const(&mut self, j: Jet, c: f64) -> Jet {
        let start = self.range(&j).start;
        self.data[start] += c;
        j
    }

    #[inline(always)]
    pub(crate) fn asinh(&mut self, j: Jet) -> Jet {
        let a = self.value(&j);
        let root = simd_math::hypot_one(self.simd, a);
        let value = simd_math::asinh(self.simd, a);
        let f1 = self.splat(1.0) / root;
        self.chain(j, value, f1, -a / (root * root * root))
    }

    #[inline(always)]
    pub(crate) fn sinh(&mut self, j: Jet) -> Jet {
        let (s, c) = simd_math::sinh_cosh(self.simd, self.value(&j));
        self.chain(j, s, c, s)
    }

    #[inline(always)]
    pub(crate) fn cosh(&mut self, j: Jet) -> Jet {
        let (s, c) = simd_math::sinh_cosh(self.simd, self.value(&j));
        self.chain(j, c, s, c)
    }

    /// `v + sqrt(1 + v*v)`, in `exp_asinh`'s cancellation-free form.
    #[inline(always)]
    pub(crate) fn positive(&mut self, j: Jet) -> Jet {
        let a = self.value(&j);
        let one = self.splat(1.0);
        let root = simd_math::hypot_one(self.simd, a);
        let value = a.simd_ge(0.0).select(a + root, one / (root - a));
        // d/da (value / root) = (value (root - a)) / root^3 = 1 / root^3.
        self.chain(j, value, value / root, one / (root * root * root))
    }

    #[inline(always)]
    pub(crate) fn exp(&mut self, j: Jet) -> Jet {
        let e = simd_math::exp(self.simd, self.value(&j));
        self.chain(j, e, e, e)
    }

    #[inline(always)]
    pub(crate) fn ln_1p(&mut self, j: Jet) -> Jet {
        let v = self.value(&j);
        let inv = self.splat(1.0) / (v + 1.0);
        let value = simd_math::ln_1p(self.simd, v);
        self.chain(j, value, inv, -inv * inv)
    }

    /// `log(cosh(v))`, in `_log_cosh`'s stable form.
    #[inline(always)]
    pub(crate) fn log_cosh(&mut self, j: Jet) -> Jet {
        let (value, t) = simd_math::log_cosh_tanh(self.simd, self.value(&j));
        let f2 = self.splat(1.0) - t * t;
        self.chain(j, value, t, f2)
    }

    #[inline(always)]
    pub(crate) fn sigmoid(&mut self, j: Jet) -> Jet {
        let one = self.splat(1.0);
        let s = one / (simd_math::exp(self.simd, -self.value(&j)) + 1.0);
        let ds = s * (one - s);
        self.chain(j, s, ds, ds * (one - s * 2.0))
    }

    #[inline(always)]
    pub(crate) fn square(&mut self, j: Jet) -> Jet {
        let v = self.value(&j);
        let two = self.splat(2.0);
        self.chain(j, v * v, v * 2.0, two)
    }
}

#[cfg(test)]
mod tests {
    use fearless_simd::{dispatch, Level};

    use super::*;

    /// `f = x exp(p)` at `(x, p) = (2, 0.5)`: `(value, grad, packed H)` from
    /// the full mode, and `(e, h)` along `dir = (1, -3)` from the directions.
    #[allow(clippy::type_complexity)]
    fn example<S: Simd>(simd: S) -> ((f64, Vec<f64>, Vec<f64>), (f64, Vec<f64>)) {
        let first = |v: S::f64s| v.as_slice()[0];
        let splat = |v: f64| S::f64s::splat(simd, v);
        let mut ws = JetArena::new(simd);

        ws.reset_hessian(2);
        let x = ws.variable(splat(2.0), 0, &[]);
        let p = ws.variable(splat(0.5), 1, &[]);
        let e = ws.exp(p);
        let f = ws.mul(x, e);
        let full = (
            first(ws.value(&f)),
            ws.grad(&f).iter().map(|&v| first(v)).collect(),
            ws.hessian(&f).iter().map(|&v| first(v)).collect(),
        );

        let dir = [splat(1.0), splat(-3.0)];
        ws.reset(2, 1);
        let x = ws.variable(splat(2.0), 0, &dir);
        let p = ws.variable(splat(0.5), 1, &dir);
        let e = ws.exp(p);
        let f = ws.mul(x, e);
        let g: Vec<f64> = ws.grad(&f).iter().map(|&v| first(v)).collect();
        let along = (
            g[0] - 3.0 * g[1],
            ws.hess_dir(&f, 0).iter().map(|&v| first(v)).collect(),
        );
        (full, along)
    }

    #[test]
    fn modes_agree_with_the_closed_form() {
        let ((value, grad, hess), (e, h)) = dispatch!(Level::new(), simd => example(simd));
        let ep = 0.5f64.exp();
        let close = |a: f64, b: f64| assert!((a - b).abs() < 1e-12, "{a} != {b}");
        close(value, 2.0 * ep);
        close(grad[0], ep);
        close(grad[1], 2.0 * ep);
        // H = [[0, e^p], [e^p, x e^p]]
        for (a, b) in hess.iter().zip([0.0, ep, 2.0 * ep]) {
            close(*a, b);
        }
        close(e, ep - 6.0 * ep);
        // H (1, -3) = (-3 e^p, e^p - 6 e^p)
        close(h[0], -3.0 * ep);
        close(h[1], ep - 6.0 * ep);
    }
}
