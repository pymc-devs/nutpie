//! Small SIMD kernels. A `&[f64]` argument is shared by the draws (weights),
//! a `&[S::f64s]` one per draw, one lane each; the `_lanes` variants take
//! per-draw values on both sides.

use fearless_simd::prelude::*;
use fearless_simd_macros::simd;

/// `sum_k w[k] x[k]`: weights shared by the draws, against per-draw values.
#[inline(always)]
pub(super) fn dot<S: Simd>(simd: S, w: &[f64], x: &[S::f64s]) -> S::f64s {
    let mut acc = S::f64s::splat(simd, 0.0);
    for (&w, &x) in w.iter().zip(x) {
        acc += x * w;
    }
    acc
}

#[inline(always)]
pub(super) fn dot_lanes<S: Simd>(simd: S, a: &[S::f64s], b: &[S::f64s]) -> S::f64s {
    let mut acc = S::f64s::splat(simd, 0.0);
    for (&a, &b) in a.iter().zip(b) {
        acc += a * b;
    }
    acc
}

/// `y += alpha x` for shared `x`.
#[inline(always)]
pub(super) fn axpy<S: Simd>(_: S, alpha: S::f64s, x: &[f64], y: &mut [S::f64s]) {
    for (y, &x) in y.iter_mut().zip(x) {
        *y += alpha * x;
    }
}

#[inline(always)]
pub(super) fn axpy_lanes<S: Simd>(_: S, alpha: S::f64s, x: &[S::f64s], y: &mut [S::f64s]) {
    for (y, &x) in y.iter_mut().zip(x) {
        *y += alpha * x;
    }
}

/// One tile's `dst.len()` vectors from their `(count, width)` block.
#[simd]
pub(super) fn load<S: Simd>(simd: S, src: &[f64], dst: &mut [S::f64s]) {
    for (dst, src) in dst.iter_mut().zip(src.chunks_exact(S::f64s::LEN)) {
        *dst = S::f64s::from_slice(simd, src);
    }
}

#[simd]
pub(super) fn store<S: Simd>(_: S, src: &[S::f64s], dst: &mut [f64]) {
    for (src, dst) in src.iter().zip(dst.chunks_exact_mut(S::f64s::LEN)) {
        src.store_slice(dst);
    }
}

pub(super) fn f64_width<S: Simd>(_: S) -> usize {
    S::f64s::LEN
}
