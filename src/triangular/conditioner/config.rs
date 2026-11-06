//! What Python describes: a conditioner's kind, its targets and its ridge,
//! and what they resolve to against a transformer ([`Outputs`]).

use anyhow::{bail, Result};
use fearless_simd::prelude::*;
use serde::{Deserialize, Serialize};

use super::Activation;
use crate::triangular::transformer::Transformer;

/// Where a conditioner's outputs go, by the transformer's roles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Target {
    Location,
    Scale,
    Shape,
    /// The coefficients without a role, see [`Role::Other`].
    Other,
    All,
}

/// One target or several.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(untagged)]
pub(crate) enum Targets {
    One(Target),
    Many(Vec<Target>),
}

impl Targets {
    fn iter(&self) -> impl Iterator<Item = Target> + '_ {
        let many: &[Target] = match self {
            Targets::One(target) => std::slice::from_ref(target),
            Targets::Many(targets) => targets,
        };
        many.iter().copied()
    }
}

/// The role of one transformer coefficient, which picks its ridge strength.
/// `Other` are the coefficients without a role, e.g. a `sas`'s input shifts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Role {
    Location,
    Scale,
    Shape,
    Other,
}

/// Each of the transformer's coefficients' role.
pub(crate) fn roles(transformer: &impl Transformer) -> Vec<Role> {
    let mut roles = vec![Role::Other; transformer.n_coefs()];
    for k in transformer.shape_coefs() {
        roles[k] = Role::Shape;
    }
    if let Some(k) = transformer.scale_coef() {
        roles[k] = Role::Scale;
    }
    if let Some(k) = transformer.location_coef() {
        roles[k] = Role::Location;
    }
    roles
}

/// Ridge strengths by group: what feeds the coefficients of each role, and
/// for an MLP its hidden units' input weights (`hidden`). Missing groups are
/// zero.
#[derive(Debug, Clone, Copy, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RidgeGroups {
    #[serde(default)]
    pub(crate) location: f64,
    #[serde(default)]
    pub(crate) scale: f64,
    #[serde(default)]
    pub(crate) shape: f64,
    #[serde(default)]
    pub(crate) other: f64,
    #[serde(default)]
    pub(crate) hidden: f64,
}

/// A conditioner's ridge: one strength for everything, or one per group.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
#[serde(untagged)]
pub(crate) enum RidgeConfig {
    All(f64),
    Groups(RidgeGroups),
}

impl Default for RidgeConfig {
    fn default() -> Self {
        RidgeConfig::All(0.0)
    }
}

impl RidgeConfig {
    pub(super) fn groups(&self) -> Result<RidgeGroups> {
        let groups = match *self {
            RidgeConfig::All(value) => RidgeGroups {
                location: value,
                scale: value,
                shape: value,
                other: value,
                hidden: value,
            },
            RidgeConfig::Groups(groups) => groups,
        };
        let RidgeGroups {
            location,
            scale,
            shape,
            other,
            hidden,
        } = groups;
        for value in [location, scale, shape, other, hidden] {
            if !(value.is_finite() && value >= 0.0) {
                bail!("ridge strengths must be finite and non-negative, got {value}");
            }
        }
        Ok(groups)
    }
}

fn one_thousandth() -> f64 {
    1e-3
}

/// A conditioner as Python describes it, by `kind`.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum ConditionerConfig {
    /// Its weights, one per target coefficient, whatever the parents.
    Constant {
        target: Targets,
        #[serde(default)]
        ridge: RidgeConfig,
    },
    /// `w_k . parents` per target coefficient `k`.
    Linear {
        target: Targets,
        #[serde(default)]
        ridge: RidgeConfig,
    },
    /// One hidden layer of `width` units and no output bias. With
    /// `input_squash = c` the units see `c asinh(y / c)` of each parent.
    /// New output weights start at `init_output_scale` times their usual
    /// range, so that a new MLP starts close to zero.
    Mlp {
        width: usize,
        target: Targets,
        #[serde(default)]
        activation: Activation,
        #[serde(default)]
        input_squash: Option<f64>,
        #[serde(default = "one_thousandth")]
        init_output_scale: f64,
        #[serde(default)]
        ridge: RidgeConfig,
    },
}

/// The coefficients a conditioner writes, in increasing order, and the
/// ridge strength on what feeds each.
#[derive(Debug, Clone)]
pub(crate) struct Outputs {
    coefs: Vec<usize>,
    strengths: Vec<f64>,
    roles: Vec<Role>,
}

impl Outputs {
    pub(super) fn new(targets: &Targets, groups: &RidgeGroups, roles: &[Role]) -> Result<Self> {
        let mut coefs = Vec::new();
        for target in targets.iter() {
            let role = match target {
                Target::All => {
                    coefs.extend(0..roles.len());
                    continue;
                }
                Target::Location => Role::Location,
                Target::Scale => Role::Scale,
                Target::Shape => Role::Shape,
                Target::Other => Role::Other,
            };
            let before = coefs.len();
            coefs.extend((0..roles.len()).filter(|&k| roles[k] == role));
            if coefs.len() == before {
                bail!("target {target:?}: the transformer has no such coefficient");
            }
        }
        coefs.sort_unstable();
        coefs.dedup();
        if coefs.is_empty() {
            bail!("a conditioner needs a target");
        }
        let roles: Vec<Role> = coefs.iter().map(|&k| roles[k]).collect();
        let strengths = roles
            .iter()
            .map(|role| match role {
                Role::Location => groups.location,
                Role::Scale => groups.scale,
                Role::Shape => groups.shape,
                Role::Other => groups.other,
            })
            .collect();
        Ok(Self {
            coefs,
            strengths,
            roles,
        })
    }

    /// The transformer coefficients written, in increasing order.
    pub(crate) fn coefs(&self) -> &[usize] {
        &self.coefs
    }

    pub(crate) fn len(&self) -> usize {
        self.coefs.len()
    }

    pub(crate) fn role(&self, j: usize) -> Role {
        self.roles[j]
    }

    /// The ridge strength on what feeds output `j`: its group's, times
    /// `loc_scale` for the location.
    pub(crate) fn strength(&self, j: usize, loc_scale: f64) -> f64 {
        match self.roles[j] {
            Role::Location => self.strengths[j] * loc_scale,
            _ => self.strengths[j],
        }
    }

    /// `out[j] = src[coefs[j]]`.
    #[inline(always)]
    pub(crate) fn gather<T: Copy>(&self, src: &[T], out: &mut [T]) {
        for (out, &k) in out.iter_mut().zip(&self.coefs) {
            *out = src[k];
        }
    }

    /// The pairs `(j, j_other)` of outputs both write to the same
    /// coefficient.
    pub(crate) fn shared<'a>(
        &'a self,
        other: &'a Outputs,
    ) -> impl Iterator<Item = (usize, usize)> + 'a {
        self.coefs.iter().enumerate().filter_map(move |(j, k)| {
            other
                .coefs
                .binary_search(k)
                .ok()
                .map(|j_other| (j, j_other))
        })
    }

    /// `dst[coefs[j]] += values[j]`.
    #[inline(always)]
    pub(crate) fn scatter_add<S: Simd>(&self, values: &[S::f64s], dst: &mut [S::f64s]) {
        for (&value, &k) in values.iter().zip(&self.coefs) {
            dst[k] += value;
        }
    }
}
