//! The sweep's model for a map flattened from a JAX `SparseTriangularMap`
//! (`nutpie.triangular_layout`): conditioner MLPs of any depth on the raw,
//! squashed or featurized parents, linear skips into the location and the
//! scale, and the transformer. It goes with the JAX flow.

use anyhow::{bail, Result};
use pulp::Arch;
use pyo3::prelude::*;
use smallvec::SmallVec;

use super::{as_u32, as_usize, Cell, Schedule, Scratch, Tape, TriangularTransform, VariableModel};
use crate::triangular::mlp_kernels::{from_tag, Activation, BackpropMlp, EvalMlp};
use crate::triangular::pattern::{Offsets, Pattern, Ragged};
use crate::triangular::transformer::layers::LayerConfig;
use crate::triangular::transformer::{AnyTransformer, LayerChain, Transformer};

/// The fixed marginal map of a parent feature, `u = g(y)` and `g'(y)`, with
/// `params = (log gamma, eps, log sigma, mu, nu)`; the inverse of one
/// sinh-arcsinh layer, as `nutpie.triangular.marginal_to_normal`.
#[inline]
fn marginal_to_normal(params: &[f64; 5], y: f64) -> (f64, f64) {
    let [log_gamma, eps, log_sigma, mu, nu] = *params;
    let half = (log_gamma - log_sigma).exp() * (y - mu) / 2.0;
    let w = (half.asinh() - eps) * (-log_gamma).exp();
    let u = 2.0 * w.sinh() + nu;
    let du = (-log_sigma).exp() * w.cosh() / (1.0 + half * half).sqrt();
    (u, du)
}

/// Per-task working buffers for one variable's evaluation.
///
/// Held on the stack (spilling to the heap only for unusually wide maps) so the
/// parallel schedules can make one per task without touching the allocator.
pub struct LayoutPad {
    inputs: Scratch,
    acc: Scratch,
    act_derivs: Scratch,
    dy_dtheta: Scratch,
    dld_dtheta: Scratch,
    seed_y: Scratch,
    seed_l: Scratch,
    next_y: Scratch,
    next_l: Scratch,
    edge_y: Scratch,
    edge_l: Scratch,
    /// With parent features: the MLP's input cotangents, `K` per parent,
    /// before they are chained to one per parent.
    feature_y: Scratch,
    feature_l: Scratch,
    transformer: <AnyTransformer as Transformer>::Scratch,
}

impl LayoutPad {
    fn new(transform: &LayoutModel) -> Self {
        let width = transform.buffer_size;
        let params = transform.num_params;
        let parents = transform.max_parents.max(1);
        Self {
            inputs: SmallVec::from_elem(0.0, width),
            acc: SmallVec::from_elem(0.0, width),
            act_derivs: SmallVec::from_elem(0.0, transform.act_deriv_len.max(1)),
            dy_dtheta: SmallVec::from_elem(0.0, params),
            dld_dtheta: SmallVec::from_elem(0.0, params),
            seed_y: SmallVec::from_elem(0.0, width),
            seed_l: SmallVec::from_elem(0.0, width),
            next_y: SmallVec::from_elem(0.0, width),
            next_l: SmallVec::from_elem(0.0, width),
            edge_y: SmallVec::from_elem(0.0, parents),
            edge_l: SmallVec::from_elem(0.0, parents),
            feature_y: SmallVec::from_elem(0.0, parents * transform.feature_degree.max(1)),
            feature_l: SmallVec::from_elem(0.0, parents * transform.feature_degree.max(1)),
            transformer: transform.transformer.new_scratch(),
        }
    }
}

/// A `SparseTriangularMap` flattened for evaluation, as a sweep's model. See
/// `nutpie.triangular_layout.TriangularLayout` for the field-by-field meaning;
/// the arrays are the same ones, taken by value.
pub struct LayoutModel {
    /// Each variable's slice of the weight blob.
    weights: Ragged<f64>,
    layer_out: Vec<usize>,
    /// Linear skip from the parents to conditioner output `skip_index`, one
    /// weight per edge. All zeros when the conditioners have none.
    skip_weight: Vec<f64>,
    skip_index: usize,
    /// The same to conditioner output `scale_skip_index`, the transformer's
    /// scale field, if the conditioners have a scale skip.
    scale_skip: Option<(Vec<f64>, usize)>,
    /// Parent features: with `feature_degree = K > 0` the MLPs see, for each
    /// parent, the orthonormal Hermite polynomials of degree `1..=K` of the
    /// parent through its marginal map (`marginal_to_normal` with
    /// `feature_params[parent]`). Zero: the raw parents.
    feature_degree: usize,
    feature_params: Vec<[f64; 5]>,
    /// With `Some(c)` (and no features), the MLPs see `c asinh(y / c)` of
    /// each parent instead of `y`; the skip stays linear in `y`.
    input_squash: Option<f64>,
    /// Each variable's `K` features and their derivatives in its value,
    /// written when the variable is evaluated and read by its children.
    features: Vec<Cell>,
    feature_derivs: Vec<Cell>,
    activation: Activation,
    transformer: AnyTransformer,
    buffer_size: usize,
    num_params: usize,
    max_parents: usize,
    /// Total width of the hidden-layer activation derivatives, which is what
    /// `BackpropMlp` reads.
    act_deriv_len: usize,
    /// Detected once; `dispatch` is then a match on a feature enum, so the
    /// per-variable cost of picking a SIMD path is a predictable branch.
    arch: Arch,
}

impl VariableModel for LayoutModel {
    type Pad = LayoutPad;

    fn new_pad(&self) -> LayoutPad {
        LayoutPad::new(self)
    }

    /// One variable: gather parents, run its MLP, apply the transformer.
    ///
    /// Reads only cells belonging to variables at strictly earlier levels,
    /// which every schedule guarantees are already written.
    ///
    /// With `TAPE`, the same pass also records this variable's row of the
    /// sparse Jacobian. The backward pass through the conditioner runs here
    /// rather than in the reverse sweep deliberately: the weights are still in
    /// L1 from the forward pass, whereas by the time a reverse sweep reached
    /// this variable they would have to come back from memory -- a second pass
    /// over the whole weight blob, which at high parent counts is the dominant
    /// cost.
    #[inline]
    fn eval_variable<const TAPE: bool>(
        &self,
        variable: usize,
        parents: &[u32],
        edges: std::ops::Range<usize>,
        y: &[Cell],
        x: f64,
        pad: &mut LayoutPad,
        tape: &Tape,
    ) -> (f64, f64) {
        let (start, n_in) = (edges.start, edges.len());
        // Taken before the MLP runs, which reuses `inputs` as a layer buffer.
        let linear = |weight: &[f64]| -> f64 {
            weight[edges.clone()]
                .iter()
                .zip(parents)
                .map(|(weight, &parent)| weight * y[parent as usize].get())
                .sum()
        };
        let skip = linear(&self.skip_weight);
        let scale_skip = self
            .scale_skip
            .as_ref()
            .map(|(weight, index)| (linear(weight), *index));
        let degree = self.feature_degree;
        let n_mlp_in = if degree == 0 {
            for (slot, &parent) in pad.inputs[..n_in].iter_mut().zip(parents) {
                let value = y[parent as usize].get();
                *slot = match self.input_squash {
                    None => value,
                    Some(c) => c * (value / c).asinh(),
                };
            }
            n_in
        } else {
            for (j, &parent) in parents.iter().enumerate() {
                let source = &self.features[parent as usize * degree..][..degree];
                for (slot, cell) in pad.inputs[j * degree..][..degree].iter_mut().zip(source) {
                    *slot = cell.get();
                }
            }
            n_in * degree
        };

        let weights = self.weights.row(variable);

        // One dispatch per variable, covering every layer: the whole MLP runs
        // inside a single `#[target_feature]` body rather than paying for the
        // feature dispatch per layer.
        self.arch.dispatch(EvalMlp {
            weights,
            layer_out: &self.layer_out,
            activation: self.activation,
            n_in: n_mlp_in,
            inputs: &mut pad.inputs,
            acc: &mut pad.acc,
            act_derivs: if TAPE {
                Some(&mut pad.act_derivs)
            } else {
                None
            },
        });
        pad.acc[self.skip_index] += skip;
        if let Some((value, index)) = scale_skip {
            pad.acc[index] += value;
        }

        if !TAPE {
            let result = self
                .transformer
                .forward_draw(&pad.acc[..self.num_params], x);
            self.store_features(variable, result.0);
            return result;
        }

        let (value, log_det, dy_dx, dld_dx) = self.transformer.forward_draw_with_grads(
            &mut pad.transformer,
            &pad.acc[..self.num_params],
            x,
            &mut pad.dy_dtheta,
            &mut pad.dld_dtheta,
        );
        tape.diag[variable].set(dy_dx);
        tape.ld_dx[variable].set(dld_dx);

        if n_in > 0 {
            pad.seed_y[..self.num_params].copy_from_slice(&pad.dy_dtheta);
            pad.seed_l[..self.num_params].copy_from_slice(&pad.dld_dtheta);
            let (out_y, out_l) = if degree == 0 {
                (&mut pad.edge_y[..n_in], &mut pad.edge_l[..n_in])
            } else {
                (&mut pad.feature_y[..n_mlp_in], &mut pad.feature_l[..n_mlp_in])
            };
            self.arch.dispatch(BackpropMlp {
                weights,
                layer_out: &self.layer_out,
                n_in: n_mlp_in,
                act_derivs: &pad.act_derivs,
                seed_y: &mut pad.seed_y,
                seed_l: &mut pad.seed_l,
                next_y: &mut pad.next_y,
                next_l: &mut pad.next_l,
                out_y,
                out_l,
            });
            if degree > 0 {
                // Chain the feature cotangents to one per parent.
                for (j, &parent) in parents.iter().enumerate() {
                    let derivs = &self.feature_derivs[parent as usize * degree..][..degree];
                    let (mut sum_y, mut sum_l) = (0.0, 0.0);
                    for (k, derivative) in derivs.iter().enumerate() {
                        let derivative = derivative.get();
                        sum_y += pad.feature_y[j * degree + k] * derivative;
                        sum_l += pad.feature_l[j * degree + k] * derivative;
                    }
                    pad.edge_y[j] = sum_y;
                    pad.edge_l[j] = sum_l;
                }
            } else if let Some(c) = self.input_squash {
                // `d/dy c asinh(y / c) = 1 / sqrt(1 + (y / c)^2)`.
                for (j, &parent) in parents.iter().enumerate() {
                    let v = y[parent as usize].get() / c;
                    let dz = 1.0 / v.hypot(1.0);
                    pad.edge_y[j] *= dz;
                    pad.edge_l[j] *= dz;
                }
            }
            let skips = std::iter::once((&self.skip_weight, self.skip_index)).chain(
                self.scale_skip
                    .as_ref()
                    .map(|(weight, index)| (weight, *index)),
            );
            for (weight, index) in skips {
                let (skip_dy, skip_dld) = (pad.dy_dtheta[index], pad.dld_dtheta[index]);
                for (k, &weight) in weight[edges.clone()].iter().enumerate() {
                    pad.edge_y[k] += skip_dy * weight;
                    pad.edge_l[k] += skip_dld * weight;
                }
            }
            for k in 0..n_in {
                tape.edges[2 * (start + k)].set(pad.edge_y[k]);
                tape.edges[2 * (start + k) + 1].set(pad.edge_l[k]);
            }
        }

        self.store_features(variable, value);
        (value, log_det)
    }
}

impl LayoutModel {
    /// Write `variable`'s parent features, and their derivatives, for its
    /// children to read.
    #[inline]
    fn store_features(&self, variable: usize, value: f64) {
        let degree = self.feature_degree;
        if degree == 0 {
            return;
        }
        let (u, du) = marginal_to_normal(&self.feature_params[variable], value);
        let features = &self.features[variable * degree..][..degree];
        let derivs = &self.feature_derivs[variable * degree..][..degree];
        // He_k(u) / sqrt(k!) and its derivative k He_{k-1}(u) / sqrt(k!) * du.
        let (mut previous, mut current) = (1.0, u);
        let mut norm = 1.0_f64;
        for k in 1..=degree {
            if k > 1 {
                (previous, current) = (current, u * current - (k - 1) as f64 * previous);
            }
            norm *= (k as f64).sqrt();
            features[k - 1].set(current / norm);
            derivs[k - 1].set(k as f64 * previous / norm * du);
        }
    }
}

/// Build a transform from `nutpie.triangular_layout.extract_layout`'s arrays.
/// Shared by the Python constructor and by [`FlowTransform::from_layout`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_transform(
    parent_indptr: &[i64],
    parent_index: &[i64],
    blob: &[f64],
    blob_offset: &[i64],
    layer_out: &[i64],
    skip_weight: &[f64],
    skip_index: i64,
    scale_skip: Option<(&[f64], i64)>,
    feature_degree: i64,
    feature_params: &[f64],
    input_squash: Option<f64>,
    activation: &str,
    transformer: &Bound<'_, PyAny>,
    level_ptr: &[i64],
    level_vars: &[i64],
    level_work: &[i64],
    min_parallel_work: i64,
    schedule: &str,
) -> Result<TriangularTransform<LayoutModel>> {
    // A layout probed from a JAX map; the skips carry their own indices.
    let configs: Vec<LayerConfig> = pythonize::depythonize(transformer)?;
    let n_coefs = layer_out
        .last()
        .and_then(|&n| usize::try_from(n).ok())
        .ok_or_else(|| anyhow::anyhow!("conditioners must have at least one layer"))?;
    build_layout_transform(
        parent_indptr,
        parent_index,
        blob,
        blob_offset,
        layer_out,
        skip_weight,
        skip_index,
        scale_skip,
        feature_degree,
        feature_params,
        input_squash,
        activation,
        LayerChain::new(configs, n_coefs, None, None)?,
        level_ptr,
        level_vars,
        level_work,
        min_parallel_work,
        schedule,
    )
}

/// [`build_transform`], with the transformer already built: for maps built
/// in Rust (`nutpie.triangular_flow`).
#[allow(clippy::too_many_arguments)]
fn build_layout_transform(
    parent_indptr: &[i64],
    parent_index: &[i64],
    blob: &[f64],
    blob_offset: &[i64],
    layer_out: &[i64],
    skip_weight: &[f64],
    skip_index: i64,
    scale_skip: Option<(&[f64], i64)>,
    feature_degree: i64,
    feature_params: &[f64],
    input_squash: Option<f64>,
    activation: &str,
    transformer: AnyTransformer,
    level_ptr: &[i64],
    level_vars: &[i64],
    level_work: &[i64],
    min_parallel_work: i64,
    schedule: &str,
) -> Result<TriangularTransform<LayoutModel>> {
    if let Some(c) = input_squash {
        if !(c.is_finite() && c > 0.0) {
            bail!("input_squash must be positive and finite, got {c}");
        }
        if feature_degree != 0 {
            bail!("input_squash and parent features are mutually exclusive");
        }
    }
    let activation: Activation =
        from_tag(activation).map_err(|_| anyhow::anyhow!("unknown activation {activation:?}"))?;
    let schedule: Schedule =
        from_tag(schedule).map_err(|_| anyhow::anyhow!("unknown schedule {schedule:?}"))?;

    let parents: Pattern<u32> = Pattern::from_i64(parent_indptr, parent_index)?;
    let weights = Ragged::new(Offsets::from_i64(blob_offset)?, blob.to_vec())?;
    let layer_out: Vec<usize> = as_usize(layer_out)?;
    let levels = Ragged::new(Offsets::from_i64(level_ptr)?, as_u32(level_vars)?)?;
    let skip_weight = skip_weight.to_vec();

    let n_variables = parents.n_var();
    if weights.n_rows() != n_variables {
        bail!("blob_offset must have one more entry than there are variables");
    }
    if layer_out.is_empty() {
        bail!("conditioners must have at least one layer");
    }

    let num_params = *layer_out.last().expect("checked non-empty");
    if transformer.n_coefs() != num_params {
        bail!(
            "the conditioners have {num_params} outputs, the transformer {} coefficients",
            transformer.n_coefs()
        );
    }
    if skip_weight.len() != parents.n_edge() {
        bail!("skip_weight must have one entry per parent_index entry");
    }
    let skip_index = usize::try_from(skip_index)
        .ok()
        .filter(|&index| index < num_params)
        .ok_or_else(|| anyhow::anyhow!("skip_index {skip_index} is out of range"))?;
    let scale_skip = match scale_skip {
        None => None,
        Some((weight, index)) => {
            if weight.len() != parents.n_edge() {
                bail!("scale_skip_weight must have one entry per parent_index entry");
            }
            let index = usize::try_from(index)
                .ok()
                .filter(|&i| i < num_params && i != skip_index)
                .ok_or_else(|| anyhow::anyhow!("scale_skip_index {index} is out of range"))?;
            Some((weight.to_vec(), index))
        }
    };
    let max_parents = parents.max_parent();
    let act_deriv_len: usize = layer_out[..layer_out.len() - 1].iter().sum();

    let feature_degree = usize::try_from(feature_degree)
        .map_err(|_| anyhow::anyhow!("feature_degree must not be negative"))?;
    let feature_params: Vec<[f64; 5]> = if feature_degree == 0 {
        vec![[0.0; 5]; n_variables]
    } else {
        if feature_params.len() != 5 * n_variables {
            bail!("feature_params must have five entries per variable");
        }
        feature_params
            .chunks_exact(5)
            .map(|chunk| chunk.try_into().expect("chunks of five"))
            .collect()
    };

    let buffer_size = layer_out
        .iter()
        .copied()
        .chain(std::iter::once(max_parents * feature_degree.max(1)))
        .max()
        .unwrap_or(1)
        .max(1);

    let model = LayoutModel {
        weights,
        layer_out,
        skip_weight,
        skip_index,
        scale_skip,
        feature_degree,
        feature_params,
        input_squash,
        features: Cell::zeros(n_variables * feature_degree),
        feature_derivs: Cell::zeros(n_variables * feature_degree),
        activation,
        transformer,
        buffer_size,
        num_params,
        max_parents,
        act_deriv_len,
        arch: Arch::new(),
    };
    TriangularTransform::new(
        parents,
        levels,
        level_work,
        min_parallel_work,
        schedule,
        model,
    )
}
