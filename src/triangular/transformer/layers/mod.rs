//! The elementwise transformer layers: their configs, and the prepared layers
//! both evaluations share. The transformers of [`crate::triangular::transformer`]
//! are chains of them.
//!
//! [`scalar`] runs a chain forward on `f64` with first derivatives, for the
//! sampler. [`jet`] runs it inverted on jets with second-order derivatives in
//! the coefficients, for the LM fit.

pub(crate) mod jet;
pub(crate) mod scalar;

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

/// `field = coefs[index] + offset`.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
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

#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub(crate) struct Contract2Config {
    pub(crate) alpha: Option<Param>,
    pub(crate) beta: Option<Param>,
    pub(crate) sigma: Option<Param>,
    pub(crate) mu: Option<Param>,
    pub(crate) nu: Option<Param>,
    pub(crate) log_gamma_bounds: Option<(f64, f64)>,
}

/// `nutpie.normalizing_flow.TangentSAS`, with its raw (unconstrained) fields.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub(crate) struct TangentSasConfig {
    pub(crate) nu: Option<Param>,
    pub(crate) eps: Option<Param>,
    pub(crate) b: Option<Param>,
    pub(crate) r: Option<Param>,
}

/// `nutpie.normalizing_flow.AsymmetricSAS`, with its raw fields.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub(crate) struct AsymmetricSasConfig {
    pub(crate) c: Option<Param>,
    pub(crate) a: Option<Param>,
    pub(crate) alpha_pos: Option<Param>,
    pub(crate) alpha_neg: Option<Param>,
    pub(crate) log_gamma_bounds: Option<(f64, f64)>,
}

/// `nutpie.normalizing_flow.PositiveAffine`, with its raw fields.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub(crate) struct PositiveAffineConfig {
    pub(crate) loc: Option<Param>,
    pub(crate) scale: Option<Param>,
}

/// One transformer layer, tagged by `kind` (see
/// `nutpie.triangular_layout.transformer_dicts`).
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum LayerConfig {
    Contract2(Contract2Config),
    TangentSas(TangentSasConfig),
    AsymmetricSas(AsymmetricSasConfig),
    PositiveAffine(PositiveAffineConfig),
}

impl LayerConfig {
    /// The parameters the layer reads.
    pub(crate) fn params(&self) -> impl Iterator<Item = Param> {
        let fields = match *self {
            LayerConfig::Contract2(c) => [c.alpha, c.beta, c.sigma, c.mu, c.nu],
            LayerConfig::TangentSas(t) => [t.nu, t.eps, t.b, t.r, None],
            LayerConfig::AsymmetricSas(t) => [t.c, t.a, t.alpha_pos, t.alpha_neg, None],
            LayerConfig::PositiveAffine(a) => [a.loc, a.scale, None, None, None],
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
    fn new(config: Contract2Config) -> Result<Self> {
        let bound = match config.log_gamma_bounds {
            None => None,
            Some((low, high)) => Some(LogGammaBound::new(low, high)?),
        };
        Ok(Self {
            alpha: config.alpha,
            beta: config.beta,
            sigma: config.sigma,
            mu: config.mu,
            nu: config.nu,
            bound,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct AsymmetricSas {
    c: Option<Param>,
    a: Option<Param>,
    alpha_pos: Option<Param>,
    alpha_neg: Option<Param>,
    bound: Option<LogGammaBound>,
}

impl AsymmetricSas {
    fn new(config: AsymmetricSasConfig) -> Result<Self> {
        let bound = match config.log_gamma_bounds {
            None => None,
            Some((low, high)) => Some(LogGammaBound::new(low, high)?),
        };
        Ok(Self {
            c: config.c,
            a: config.a,
            alpha_pos: config.alpha_pos,
            alpha_neg: config.alpha_neg,
            bound,
        })
    }
}

/// A transformer layer, ready to evaluate.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Layer {
    Contract2(Contract2),
    TangentSas(TangentSasConfig),
    AsymmetricSas(AsymmetricSas),
    PositiveAffine(PositiveAffineConfig),
}

impl Layer {
    pub(crate) fn new(config: LayerConfig) -> Result<Self> {
        Ok(match config {
            LayerConfig::Contract2(config) => Layer::Contract2(Contract2::new(config)?),
            LayerConfig::TangentSas(config) => Layer::TangentSas(config),
            LayerConfig::AsymmetricSas(config) => Layer::AsymmetricSas(AsymmetricSas::new(config)?),
            LayerConfig::PositiveAffine(config) => Layer::PositiveAffine(config),
        })
    }

    /// The conditioner outputs of the fields that shape the layer beyond an
    /// affine map: `Contract2`'s `alpha` and `beta` (with both zero it is
    /// affine, `sigma`, `mu` and `nu` being its scale and shifts), and all of
    /// `TangentSAS`'s and `AsymmetricSAS`'s, which are the identity at
    /// `eps = 0`, `r = 1` and `gamma_pm = 1` whatever their shift and scale.
    pub(crate) fn shape_params(&self) -> impl Iterator<Item = usize> {
        let fields = match *self {
            Layer::Contract2(c) => [c.alpha, c.beta, None, None],
            Layer::TangentSas(t) => [t.nu, t.eps, t.b, t.r],
            Layer::AsymmetricSas(t) => [t.c, t.a, t.alpha_pos, t.alpha_neg],
            Layer::PositiveAffine(_) => [None; 4],
        };
        fields.into_iter().flatten().map(|p| p.index)
    }
}
