//! The inverted layer chain on jets, see [`crate::triangular::transformer::jet`].
//!
//! Each layer's `*_inverse` mirrors its Python `inverse_and_log_det`, with
//! the same variable names, and returns the jets of its output and log
//! determinant. `ws` is the [`JetArena`] every jet lives in, and `coefs`
//! makes the layer's fields, entries of the coefficients, into jet inputs.

use fearless_simd::prelude::*;

use super::{
    AsymmetricSas, Contract2, Layer, LogGammaBound, Param, PositiveAffineConfig, ScaleSquash,
    TangentSasConfig,
};
use crate::triangular::transformer::jet::{Jet, JetArena};

/// A transformer's coefficients as jet inputs: coefficient `k` is input `1 +
/// k` of `z = (y, coefs)`, along the directions `dirs` (`1 + values.len()`
/// entries each, none for full Hessians).
pub(crate) struct CoefJets<'a, S: Simd> {
    values: &'a [S::f64s],
    dirs: &'a [S::f64s],
}

impl<'a, S: Simd> CoefJets<'a, S> {
    #[inline(always)]
    pub(crate) fn new(values: &'a [S::f64s], dirs: &'a [S::f64s]) -> Self {
        Self { values, dirs }
    }

    #[inline(always)]
    fn get(&self, ws: &mut JetArena<S>, param: Option<Param>) -> Option<Jet> {
        param.map(|p| ws.variable(self.values[p.index] + p.offset, 1 + p.index, self.dirs))
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

/// `log gamma = asinh(alpha)`, squashed into `bound` if given
/// (`_bounded_log_gamma`).
#[inline(always)]
fn log_gamma<S: Simd>(ws: &mut JetArena<S>, alpha: Jet, bound: Option<LogGammaBound>) -> Jet {
    let log_gamma = ws.asinh(alpha);
    match bound {
        None => log_gamma,
        Some(b) => {
            let t = ws.scale(log_gamma, b.slope);
            let t = ws.add_const(t, b.offset);
            let t = ws.sigmoid(t);
            let t = ws.scale(t, b.width);
            ws.add_const(t, b.low)
        }
    }
}

/// A scale field's `log scale = c asinh(s / c)`, see [`ScaleSquash`].
#[inline(always)]
fn log_scale<S: Simd>(ws: &mut JetArena<S>, s: Jet, squash: ScaleSquash) -> Jet {
    match squash {
        None => s,
        Some(1.0) => ws.asinh(s),
        Some(c) => {
            let t = ws.scale(s, 1.0 / c);
            let t = ws.asinh(t);
            ws.scale(t, c)
        }
    }
}

impl Contract2 {
    /// `Contract2.inverse_and_log_det`.
    #[inline(always)]
    fn inverse<S: Simd>(&self, ws: &mut JetArena<S>, fields: &CoefJets<S>, x: Jet) -> (Jet, Jet) {
        let log_gamma = fields
            .get(ws, self.alpha)
            .map(|alpha| log_gamma(ws, alpha, self.bound));
        let log_delta = fields.get(ws, self.beta).map(|beta| ws.asinh(beta));
        let log_sigma = fields
            .get(ws, self.sigma)
            .map(|sigma| log_scale(ws, sigma, self.scale_squash));

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
    layer: &TangentSasConfig,
    ws: &mut JetArena<S>,
    fields: &CoefJets<S>,
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

impl AsymmetricSas {
    /// `AsymmetricSAS.inverse_and_log_det`: with `gamma` that of `y`'s side,
    /// `q = gamma y` and `t = asinh(q) / gamma`, `x = c + sinh(t) /
    /// scale_mod` and `log_det = logcosh(t) - log scale_mod - 0.5 log1p(q^2)`.
    #[inline(always)]
    fn inverse<S: Simd>(&self, ws: &mut JetArena<S>, fields: &CoefJets<S>, y: Jet) -> (Jet, Jet) {
        let alpha_pos = fields.get_or_zero(ws, self.alpha_pos);
        let log_gamma_pos = log_gamma(ws, alpha_pos, self.bound);
        let alpha_neg = fields.get_or_zero(ws, self.alpha_neg);
        let log_gamma_neg = log_gamma(ws, alpha_neg, self.bound);
        let positive = ws.value(&y).simd_ge(0.0);
        let log_gamma = ws.select(positive, log_gamma_pos, log_gamma_neg);

        let gamma = ws.copy(&log_gamma);
        let gamma = ws.exp(gamma);
        let q = ws.mul(y, gamma);
        let t = ws.copy(&q);
        let t = ws.asinh(t);
        let inv_gamma = ws.neg(log_gamma);
        let inv_gamma = ws.exp(inv_gamma);
        let t = ws.mul(t, inv_gamma);

        let mut out = ws.copy(&t);
        out = ws.sinh(out);
        let log_scale = fields.get(ws, self.a).map(|a| ws.asinh(a));
        if let Some(log_scale) = &log_scale {
            let factor = ws.copy(log_scale);
            let factor = ws.neg(factor);
            let factor = ws.exp(factor);
            out = ws.mul(out, factor);
        }
        if let Some(c) = fields.get(ws, self.c) {
            out = ws.add(out, c);
        }

        let mut ld = ws.log_cosh(t);
        let q = ws.square(q);
        let q = ws.ln_1p(q);
        let q = ws.scale(q, 0.5);
        ld = ws.sub(ld, q);
        if let Some(log_scale) = log_scale {
            ld = ws.sub(ld, log_scale);
        }
        (out, ld)
    }
}

/// `PositiveAffine.inverse_and_log_det`: `x = (y - loc) / scale_mod`, and
/// `1 / scale_mod = positive(-scale)` for the default squash `c = 1`, else
/// `exp(-log scale_mod)`.
#[inline(always)]
fn positive_affine_inverse<S: Simd>(
    layer: &PositiveAffineConfig,
    ws: &mut JetArena<S>,
    fields: &CoefJets<S>,
    x: Jet,
) -> (Jet, Jet) {
    let centred = match fields.get(ws, layer.loc) {
        Some(loc) => ws.sub(x, loc),
        None => x,
    };
    match fields.get(ws, layer.scale) {
        Some(scale) => {
            let (factor, ld) = if layer.scale_squash == Some(1.0) {
                let factor = ws.copy(&scale);
                let factor = ws.neg(factor);
                (ws.positive(factor), ws.asinh(scale))
            } else {
                let ld = log_scale(ws, scale, layer.scale_squash);
                let factor = ws.copy(&ld);
                let factor = ws.neg(factor);
                (ws.exp(factor), ld)
            };
            let out = ws.mul(centred, factor);
            (out, ws.neg(ld))
        }
        None => (centred, fields.get_or_zero(ws, None)),
    }
}

/// `x = T(y; coefs)` and `Lambda = log |dT/dy|` of the inverted chain (each
/// layer's `inverse_and_log_det`, last layer first), as jets.
#[inline(always)]
pub(crate) fn inverse_chain<S: Simd>(
    layers: &[Layer],
    ws: &mut JetArena<S>,
    coefs: &CoefJets<S>,
    y: Jet,
) -> (Jet, Jet) {
    let mut x = y;
    let zero = ws.splat(0.0);
    let mut log_det = ws.constant(zero);
    for layer in layers.iter().rev() {
        let (out, ld) = match layer {
            Layer::Contract2(layer) => layer.inverse(ws, coefs, x),
            Layer::TangentSas(layer) => tangent_sas_inverse(layer, ws, coefs, x),
            Layer::AsymmetricSas(layer) => layer.inverse(ws, coefs, x),
            Layer::PositiveAffine(layer) => positive_affine_inverse(layer, ws, coefs, x),
        };
        x = out;
        log_det = ws.add(log_det, ld);
    }
    (x, log_det)
}
