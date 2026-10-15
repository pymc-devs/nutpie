//! The transformer: the elementwise bijection of one variable, given its
//! coefficients `coefs` (`pi` in `notes/lm_derivatives.md`).
//!
//! The generative direction maps a base value `x` to `y`, the density
//! direction back, `x = T(y; coefs)`. With all coefficients zero, every kind
//! is the identity. The coefficients have roles: one location, one scale and
//! the shape coefficients that make the transformer nonlinear; others (e.g.
//! the input shifts of a stacked `sas`) have none.
//!
//! The kinds Python describes ([`TransformerConfig`]) are all chains of layers
//! (see [`crate::triangular::transformer::layers`]), so one type implements the trait,
//! [`LayerChain`]; each kind's constructor owns its stacking rules.
//!
//! Method suffixes: `_draw` for one draw on `f64` (the sampler), `_tile` for
//! one SIMD tile of draws (the LM's local kernels), `_jet` for jets.

pub(crate) mod jet;
pub(crate) mod layers;
#[cfg(test)]
mod tests;

use anyhow::{bail, Result};
use fearless_simd::{dispatch, prelude::*, Level};
use fearless_simd_macros::simd;
use serde::{Deserialize, Serialize};

use crate::triangular::transformer::jet::{packed_len, Jet, JetArena};
use crate::triangular::transformer::layers::jet::{inverse_chain, CoefJets};
use crate::triangular::transformer::layers::scalar::{
    forward_chain, forward_chain_with_grads, LayerTape,
};
use crate::triangular::transformer::layers::{
    AsymmetricSasConfig, Contract2Config, Layer, LayerConfig, Param, PositiveAffineConfig,
    TangentSasConfig,
};

pub(crate) trait Transformer {
    /// Per-thread workspace of [`Self::forward_draw_with_grads`].
    type Scratch: Send;
    fn new_scratch(&self) -> Self::Scratch;

    /// Roles of the coefficients, for the whole transformer.
    fn n_coefs(&self) -> usize;
    fn location_coef(&self) -> Option<usize>;
    fn scale_coef(&self) -> Option<usize>;
    fn shape_coefs(&self) -> impl Iterator<Item = usize> + '_;

    /// The location coefficient that turns the output `y` into `a y + b`,
    /// from its value `coef`, for a transformer whose location and scale
    /// act last: `y = loc + scale f`.
    fn shifted_location_coef(&self, coef: f64, a: f64, b: f64) -> f64;
    /// The scale coefficient that multiplies the output's scale by `factor`
    /// (positive), from its value `coef`.
    fn rescaled_scale_coef(&self, coef: f64, factor: f64) -> f64;

    /// Generative direction: `(y, log_det)`.
    fn forward_draw(&self, coefs: &[f64], x: f64) -> (f64, f64);

    /// `(y, log_det, dy/dx, dlog_det/dx)`, and `dy/dcoefs`,
    /// `dlog_det/dcoefs` for the pullback through the conditioners.
    fn forward_draw_with_grads(
        &self,
        scratch: &mut Self::Scratch,
        coefs: &[f64],
        x: f64,
        dy_dcoefs: &mut [f64],
        dlog_det_dcoefs: &mut [f64],
    ) -> (f64, f64, f64, f64);

    /// Density direction on jets in `z = (y, coefs)`: `(x, log_det)`, with
    /// `log_det = log dx/dy`.
    fn inverse_jet<S: Simd>(&self, ws: &mut JetArena<S>, coefs: &CoefJets<S>, y: Jet)
        -> (Jet, Jet);

    /// Density direction: `(x, log_det)`. Not for hot loops: the default
    /// runs [`Self::inverse_jet`] on a whole tile.
    #[allow(dead_code, reason = "only the tests need it so far")]
    fn inverse_draw(&self, coefs: &[f64], y: f64) -> (f64, f64) {
        dispatch!(Level::new(), simd => inverse_draw_on(self, simd, coefs, y))
    }

    /// Density direction for one tile, as jets along the `k` directions
    /// `dirs` in `z` (`1 + n_coefs` entries each). Resets `ws`.
    #[simd]
    fn inverse_jet_tile<S: Simd>(
        &self,
        simd: S,
        ws: &mut JetArena<S>,
        coefs: &[S::f64s],
        y: S::f64s,
        dirs: &[S::f64s],
        k: usize,
    ) -> (Jet, Jet) {
        ws.reset(1 + coefs.len(), k);
        let y = ws.variable(y, 0, dirs);
        self.inverse_jet(ws, &CoefJets::new(coefs, dirs), y)
    }

    /// Density direction for one tile, with full derivatives.
    #[simd]
    fn derivatives_tile<S: Simd>(
        &self,
        simd: S,
        ws: &mut JetArena<S>,
        coefs: &[S::f64s],
        y: S::f64s,
        out: &mut TransformerDerivatives<S>,
    ) {
        ws.reset_hessian(1 + coefs.len());
        let y = ws.variable(y, 0, &[]);
        let (x, log_det) = self.inverse_jet(ws, &CoefJets::new(coefs, &[]), y);
        out.x = ws.value(&x);
        out.log_det = ws.value(&log_det);
        out.grad_x.copy_from_slice(ws.grad(&x));
        out.grad_log_det.copy_from_slice(ws.grad(&log_det));
        out.hess_x.copy_from_slice(ws.hessian(&x));
        out.hess_log_det.copy_from_slice(ws.hessian(&log_det));
    }
}

#[allow(dead_code, reason = "only the tests need it so far")]
fn inverse_draw_on<T: Transformer + ?Sized, S: Simd>(
    transformer: &T,
    simd: S,
    coefs: &[f64],
    y: f64,
) -> (f64, f64) {
    let coefs: Vec<_> = coefs.iter().map(|&c| S::f64s::splat(simd, c)).collect();
    let mut ws = JetArena::new(simd);
    let y = S::f64s::splat(simd, y);
    let (x, log_det) = transformer.inverse_jet_tile(simd, &mut ws, &coefs, y, &[], 0);
    (ws.value(&x)[0], ws.value(&log_det)[0])
}

/// The transformer at one variable and tile in the density direction, as a
/// function of `z = (y, coefs)`. The Hessians are packed lower triangles,
/// see [`crate::triangular::transformer::jet`].
pub(crate) struct TransformerDerivatives<S: Simd> {
    /// `x = T(y; coefs)`.
    pub(crate) x: S::f64s,
    /// `Lambda = log dT/dy`.
    pub(crate) log_det: S::f64s,
    /// `(delta, t)`: `delta = dx/dy`, `t = dx/dcoefs`.
    pub(crate) grad_x: Vec<S::f64s>,
    /// `(mu, l)`: `mu = dLambda/dy`, `l = dLambda/dcoefs`.
    pub(crate) grad_log_det: Vec<S::f64s>,
    pub(crate) hess_x: Vec<S::f64s>,
    pub(crate) hess_log_det: Vec<S::f64s>,
}

impl<S: Simd> TransformerDerivatives<S> {
    pub(crate) fn new(simd: S, n_coefs: usize) -> Self {
        let zero = S::f64s::splat(simd, 0.0);
        let n_z = 1 + n_coefs;
        Self {
            x: zero,
            log_det: zero,
            grad_x: vec![zero; n_z],
            grad_log_det: vec![zero; n_z],
            hess_x: vec![zero; packed_len(n_z)],
            hess_log_det: vec![zero; packed_len(n_z)],
        }
    }
}

/// The transformer every flow holds. Every kind is a [`LayerChain`] so far;
/// a kind that is not becomes a variant of an enum here.
pub(crate) type AnyTransformer = LayerChain;

/// A transformer as Python describes it, by `kind`.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum TransformerConfig {
    /// `loc + scale_mod * x`.
    Affine,
    /// `layers` stacked `Contract2` (sinh-arcsinh) layers. Every interior
    /// output shift is redundant with the next layer's input shift, so only
    /// the last layer has one.
    Sas {
        #[serde(default = "one")]
        layers: usize,
        #[serde(default)]
        log_gamma_bounds: Option<(f64, f64)>,
    },
    /// `layers` stacked `TangentSAS` layers and a trailing affine. With
    /// `fix_b`, every layer's `b` stays at its start.
    TangentSas {
        #[serde(default = "one")]
        layers: usize,
        #[serde(default)]
        fix_b: bool,
    },
    /// `layers` stacked `AsymmetricSAS` layers and a trailing affine. The
    /// first acts on the standard-normal base, so it has no shift or scale.
    AsymmetricSas {
        #[serde(default = "one")]
        layers: usize,
        #[serde(default)]
        log_gamma_bounds: Option<(f64, f64)>,
    },
}

fn one() -> usize {
    1
}

/// Hands out the coefficients in order, each feeding one field at offset
/// zero, so that all-zero coefficients are each layer's identity.
struct Coefs(usize);

impl Coefs {
    fn next(&mut self) -> Option<Param> {
        self.0 += 1;
        Some(Param {
            index: self.0 - 1,
            offset: 0.0,
        })
    }
}

impl TransformerConfig {
    /// The kind's layer chain. The coefficients are numbered in the order
    /// `nutpie.normalizing_flow.make_transformer`'s arrays flatten in.
    pub(crate) fn build(&self) -> Result<LayerChain> {
        let mut coefs = Coefs(0);
        let mut configs = Vec::new();
        let n_layers = match *self {
            Self::Affine => 1,
            Self::Sas { layers, .. }
            | Self::TangentSas { layers, .. }
            | Self::AsymmetricSas { layers, .. } => layers,
        };
        if n_layers == 0 {
            bail!("a transformer needs at least one layer");
        }
        let (location, scale) = match *self {
            Self::Sas {
                layers,
                log_gamma_bounds,
            } => {
                let mut roles = (None, None);
                for index in 0..layers {
                    let (alpha, beta, sigma) = (coefs.next(), coefs.next(), coefs.next());
                    let mu = if index == layers - 1 {
                        coefs.next()
                    } else {
                        None
                    };
                    configs.push(LayerConfig::Contract2(Contract2Config {
                        alpha,
                        beta,
                        sigma,
                        mu,
                        nu: coefs.next(),
                        log_gamma_bounds,
                    }));
                    roles = (mu, sigma);
                }
                roles
            }
            Self::TangentSas { layers, fix_b } => {
                for _ in 0..layers {
                    let (nu, eps) = (coefs.next(), coefs.next());
                    let b = if fix_b { None } else { coefs.next() };
                    configs.push(LayerConfig::TangentSas(TangentSasConfig {
                        nu,
                        eps,
                        b,
                        r: coefs.next(),
                    }));
                }
                affine(&mut coefs, &mut configs)
            }
            Self::AsymmetricSas {
                layers,
                log_gamma_bounds,
            } => {
                for index in 0..layers {
                    let (c, a) = if index > 0 {
                        (coefs.next(), coefs.next())
                    } else {
                        (None, None)
                    };
                    configs.push(LayerConfig::AsymmetricSas(AsymmetricSasConfig {
                        c,
                        a,
                        alpha_pos: coefs.next(),
                        alpha_neg: coefs.next(),
                        log_gamma_bounds,
                    }));
                }
                affine(&mut coefs, &mut configs)
            }
            Self::Affine => affine(&mut coefs, &mut configs),
        };
        LayerChain::new(
            configs,
            coefs.0,
            location.map(|p| p.index),
            scale.map(|p| p.index),
        )
    }
}

/// Appends the trailing `PositiveAffine`, returning its `(loc, scale)`.
fn affine(coefs: &mut Coefs, configs: &mut Vec<LayerConfig>) -> (Option<Param>, Option<Param>) {
    let (loc, scale) = (coefs.next(), coefs.next());
    configs.push(LayerConfig::PositiveAffine(PositiveAffineConfig {
        loc,
        scale,
    }));
    (loc, scale)
}

/// A transformer that is a chain of layers: the generative direction runs
/// them first to last, the density direction inverts them last to first.
#[derive(Debug, Clone)]
pub(crate) struct LayerChain {
    configs: Vec<LayerConfig>,
    layers: Vec<Layer>,
    n_coefs: usize,
    location: Option<usize>,
    scale: Option<usize>,
    /// The offsets of the fields the location and the scale feed.
    location_offset: f64,
    scale_offset: f64,
    /// The coefficients of the fields that shape a layer beyond an affine
    /// map, see [`Layer::shape_params`].
    shape: Vec<usize>,
}

impl LayerChain {
    /// The chain of `configs` on `n_coefs` coefficients, with the given roles.
    /// For a layout probed from a JAX transformer; the kinds use
    /// [`TransformerConfig::build`].
    pub(crate) fn new(
        configs: Vec<LayerConfig>,
        n_coefs: usize,
        location: Option<usize>,
        scale: Option<usize>,
    ) -> Result<Self> {
        for param in configs.iter().flat_map(LayerConfig::params) {
            if param.index >= n_coefs {
                bail!("transformer coefficient index {} out of range", param.index);
            }
        }
        for role in [location, scale].into_iter().flatten() {
            if role >= n_coefs {
                bail!("transformer role index {role} out of range for {n_coefs} coefficients");
            }
        }
        if location.is_some() && location == scale {
            bail!("the location and the scale must be different coefficients");
        }
        let layers = configs
            .iter()
            .copied()
            .map(Layer::new)
            .collect::<Result<Vec<_>>>()?;
        let offset = |role: Option<usize>| {
            configs
                .iter()
                .flat_map(LayerConfig::params)
                .find(|p| Some(p.index) == role)
                .map_or(0.0, |p| p.offset)
        };
        let (location_offset, scale_offset) = (offset(location), offset(scale));
        let mut shape: Vec<usize> = layers.iter().flat_map(Layer::shape_params).collect();
        shape.sort_unstable();
        shape.dedup();
        Ok(Self {
            configs,
            layers,
            n_coefs,
            location,
            scale,
            location_offset,
            scale_offset,
            shape,
        })
    }

    pub(crate) fn configs(&self) -> &[LayerConfig] {
        &self.configs
    }

    pub(crate) fn n_layers(&self) -> usize {
        self.layers.len()
    }
}

impl Transformer for LayerChain {
    type Scratch = Vec<LayerTape>;

    fn new_scratch(&self) -> Self::Scratch {
        Vec::with_capacity(self.layers.len())
    }

    fn n_coefs(&self) -> usize {
        self.n_coefs
    }

    fn location_coef(&self) -> Option<usize> {
        self.location
    }

    fn scale_coef(&self) -> Option<usize> {
        self.scale
    }

    fn shape_coefs(&self) -> impl Iterator<Item = usize> + '_ {
        self.shape.iter().copied()
    }

    /// The location field is the last layer's output shift.
    fn shifted_location_coef(&self, coef: f64, a: f64, b: f64) -> f64 {
        let field = coef + self.location_offset;
        a * field + b - self.location_offset
    }

    /// The scale field `s` scales the last layer's output by `exp(asinh(s))`
    /// (`Contract2`'s `sigma`, `PositiveAffine`'s `scale`).
    fn rescaled_scale_coef(&self, coef: f64, factor: f64) -> f64 {
        let field = coef + self.scale_offset;
        (field.asinh() + factor.ln()).sinh() - self.scale_offset
    }

    #[inline(always)]
    fn forward_draw(&self, coefs: &[f64], x: f64) -> (f64, f64) {
        forward_chain(&self.layers, coefs, x)
    }

    #[inline(always)]
    fn forward_draw_with_grads(
        &self,
        scratch: &mut Self::Scratch,
        coefs: &[f64],
        x: f64,
        dy_dcoefs: &mut [f64],
        dlog_det_dcoefs: &mut [f64],
    ) -> (f64, f64, f64, f64) {
        forward_chain_with_grads(&self.layers, scratch, coefs, x, dy_dcoefs, dlog_det_dcoefs)
    }

    #[inline(always)]
    fn inverse_jet<S: Simd>(
        &self,
        ws: &mut JetArena<S>,
        coefs: &CoefJets<S>,
        y: Jet,
    ) -> (Jet, Jet) {
        inverse_chain(&self.layers, ws, coefs, y)
    }
}

/// The layout of a transformer kind, `{"kind": ...}` as in
/// `nutpie.triangular_flow`: its number of coefficients, their roles, and
/// its layers as `nutpie.triangular_layout.transformer_dicts` writes them.
#[pyo3::pyfunction]
#[pyo3(name = "transformer_layout")]
pub(crate) fn py_transformer_layout<'py>(
    py: pyo3::Python<'py>,
    config: &pyo3::Bound<'py, pyo3::PyAny>,
) -> Result<pyo3::Bound<'py, pyo3::types::PyDict>> {
    use pyo3::types::{PyDict, PyDictMethods};

    let config: TransformerConfig = crate::pyjson::from_py(config)?;
    let chain = config.build()?;
    let out = PyDict::new(py);
    out.set_item("n_coefs", chain.n_coefs())?;
    out.set_item("location", chain.location_coef())?;
    out.set_item("scale", chain.scale_coef())?;
    out.set_item("shape", chain.shape_coefs().collect::<Vec<_>>())?;
    let layers = chain
        .configs()
        .iter()
        .map(|layer| layer.to_pydict(py))
        .collect::<pyo3::PyResult<Vec<_>>>()?;
    out.set_item("layers", layers)?;
    Ok(out)
}
