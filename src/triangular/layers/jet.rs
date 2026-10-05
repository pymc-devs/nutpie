//! The inverted transformer chain on jets, see [`crate::triangular::jet`].
//!
//! Each layer's `*_inverse` mirrors its Python `inverse_and_log_det`, with
//! the same variable names, and returns the jets of its output and log
//! determinant. `ws` is the [`JetArena`] every jet lives in, and `fields`
//! makes the layer's parameters, entries of `pi`, into jet inputs.

use fearless_simd::prelude::*;
use fearless_simd_macros::simd;

use super::{Contract2, Layer, Param, PositiveAffineSpec, TangentSasSpec};
use crate::triangular::jet::{Jet, JetArena};

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

impl Contract2 {
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

/// `T(y; pi)` and `Lambda(y; pi) = log |dT/dy|` of the inverted transformer
/// chain (each layer's `inverse_and_log_det`, last layer first), as jets in
/// `z = (y, pi)` along the `k` directions `dirs`, `1 + pi.len()` entries each.
/// Resets `ws`, so earlier jets in it are invalid afterwards.
#[simd]
pub(crate) fn transformer<S: Simd>(
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
pub(crate) fn transformer_hessian<S: Simd>(
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
