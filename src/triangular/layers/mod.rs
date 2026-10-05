//! The elementwise transformer layers: their specs as Python describes them,
//! and the prepared layers both evaluations share.
//!
//! [`scalar`] runs the chain forward on `f64` with first derivatives, for the
//! transform. [`jet`] runs it inverted on jets with second-order derivatives in
//! the transformer parameters, for the LM fit.

pub(crate) mod jet;
pub(crate) mod scalar;

use anyhow::{bail, Result};
use serde::Deserialize;

/// `field = conditioner_output[index] + offset`.
#[derive(Debug, Clone, Copy, Deserialize)]
pub(crate) struct Param {
    pub(crate) index: usize,
    pub(crate) offset: f64,
}

impl Param {
    #[inline(always)]
    fn get(self, params: &[f64]) -> f64 {
        params[self.index] + self.offset
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub(crate) struct Contract2Spec {
    pub(crate) alpha: Option<Param>,
    pub(crate) beta: Option<Param>,
    pub(crate) sigma: Option<Param>,
    pub(crate) mu: Option<Param>,
    pub(crate) nu: Option<Param>,
    pub(crate) log_gamma_bounds: Option<(f64, f64)>,
}

/// `nutpie.normalizing_flow.TangentSAS`, with its raw (unconstrained) fields.
#[derive(Debug, Clone, Copy, Deserialize)]
pub(crate) struct TangentSasSpec {
    pub(crate) nu: Option<Param>,
    pub(crate) eps: Option<Param>,
    pub(crate) b: Option<Param>,
    pub(crate) r: Option<Param>,
}

/// `nutpie.normalizing_flow.PositiveAffine`, with its raw fields.
#[derive(Debug, Clone, Copy, Deserialize)]
pub(crate) struct PositiveAffineSpec {
    pub(crate) loc: Option<Param>,
    pub(crate) scale: Option<Param>,
}

/// One transformer layer, tagged by `kind` (see
/// `nutpie.triangular_layout.transformer_dicts`).
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum LayerSpec {
    Contract2(Contract2Spec),
    TangentSas(TangentSasSpec),
    PositiveAffine(PositiveAffineSpec),
}

impl LayerSpec {
    /// The parameters the layer reads.
    pub(crate) fn params(&self) -> impl Iterator<Item = Param> {
        let fields = match *self {
            LayerSpec::Contract2(c) => [c.alpha, c.beta, c.sigma, c.mu, c.nu],
            LayerSpec::TangentSas(t) => [t.nu, t.eps, t.b, t.r, None],
            LayerSpec::PositiveAffine(a) => [a.loc, a.scale, None, None, None],
        };
        fields.into_iter().flatten()
    }
}

/// `_bounded_log_gamma` from `nutpie/normalizing_flow.py`, with the constants
/// derived from the bounds once instead of per call.
#[derive(Debug, Clone, Copy)]
struct LogGammaBound {
    low: f64,
    width: f64,
    slope: f64,
    offset: f64,
}

impl LogGammaBound {
    fn new(low: f64, high: f64) -> Result<Self> {
        if !(low < 0.0 && 0.0 < high) {
            bail!("log_gamma_bounds must satisfy low < 0 < high, got ({low}, {high})");
        }
        let width = high - low;
        let at_zero = -low / width;
        Ok(Self {
            low,
            width,
            slope: width / (-low * high),
            offset: (at_zero / (1.0 - at_zero)).ln(),
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Contract2 {
    alpha: Option<Param>,
    beta: Option<Param>,
    sigma: Option<Param>,
    mu: Option<Param>,
    nu: Option<Param>,
    bound: Option<LogGammaBound>,
}

impl Contract2 {
    fn new(spec: Contract2Spec) -> Result<Self> {
        let bound = match spec.log_gamma_bounds {
            None => None,
            Some((low, high)) => Some(LogGammaBound::new(low, high)?),
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
}

/// A transformer layer, ready to evaluate.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Layer {
    Contract2(Contract2),
    TangentSas(TangentSasSpec),
    PositiveAffine(PositiveAffineSpec),
}

impl Layer {
    pub(crate) fn new(spec: LayerSpec) -> Result<Self> {
        Ok(match spec {
            LayerSpec::Contract2(spec) => Layer::Contract2(Contract2::new(spec)?),
            LayerSpec::TangentSas(spec) => Layer::TangentSas(spec),
            LayerSpec::PositiveAffine(spec) => Layer::PositiveAffine(spec),
        })
    }
}
