//! A triangular flow held in Rust: the map's parents and [`Conditional`], its
//! conditioners' parameters in the LM layout (see [`crate::triangular::lm`]),
//! and the flow's permutation and diagonal affine. Built by
//! `nutpie.triangular_flow`.
//!
//! The flow is, as in [`FlowTransform`],
//!
//! ```text
//! u = z[p],   v = map(u),   w = v[p^-1],   y = loc + scale * w,
//! ```
//!
//! with `p` the `permutation`. The LM residuals are defined on the
//! triangular map's inputs `v`: a draw `y` with gradient `g`, with the
//! affine undone and in the triangular order (`undo_affine`), `v[j] = (y[p[j]]
//! - loc[p[j]]) / scale[p[j]]` with gradient `g[p[j]] * scale[p[j]]`.

use anyhow::{bail, Result};
use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::triangular::conditional::Conditional;
use crate::triangular::conditioner::{ConditionerConfig, Conditioners};
use crate::triangular::lm::{FisherResiduals, FisherSpeeds, PyFisherResiduals};
use crate::triangular::mlp_kernels::from_tag;
use crate::triangular::pattern::{Offsets, Pattern, Ragged};
use crate::triangular::sampler::{
    ConditionalModel, FlowTransform, Map, PyFlowTransform, Schedule, TriangularTransform,
};
use crate::triangular::transformer::TransformerConfig;

/// Rough cost of one transformer layer in multiply-adds, for scheduling the
/// native transform's levels, as `nutpie.triangular_layout`.
const TRANSFORMER_WORK_PER_LAYER: i64 = 100;

#[derive(Clone)]
pub(crate) struct TriangularFlow {
    /// The map's parents, in the triangular order.
    graph: Pattern,
    conditional: Conditional,
    /// Each variable's slice of `theta`.
    param_offset: Offsets,
    /// The conditioners' parameters, in the LM layout.
    theta: Vec<f64>,
    permutation: Vec<usize>,
    loc: Vec<f64>,
    scale: Vec<f64>,
}

impl TriangularFlow {
    /// The flow with conditioners' parameters `theta`, or new ones drawn
    /// from `seed`.
    fn new(
        graph: Pattern,
        conditional: Conditional,
        theta: Option<Vec<f64>>,
        seed: u64,
        permutation: Vec<usize>,
        loc: Vec<f64>,
        scale: Vec<f64>,
    ) -> Result<Self> {
        let n = graph.n_var();
        let param_offset =
            Offsets::from_lens((0..n).map(|i| conditional.n_weights(graph.n_parent(i))));
        let theta = match theta {
            Some(theta) => theta,
            None => {
                let mut rng = ChaCha8Rng::seed_from_u64(seed);
                let mut theta = vec![0.0; param_offset.total()];
                for i in 0..n {
                    conditional.conditioners.init_weights(
                        graph.n_parent(i),
                        &mut rng,
                        &mut theta[param_offset.range(i)],
                    );
                }
                theta
            }
        };
        if theta.len() != param_offset.total() {
            bail!(
                "theta has length {}, expected {}",
                theta.len(),
                param_offset.total()
            );
        }
        if permutation.len() != n || loc.len() != n || scale.len() != n {
            bail!("permutation, loc and scale must each have one entry per variable");
        }
        let mut seen = vec![false; n];
        for &p in &permutation {
            if p >= n || std::mem::replace(&mut seen[p], true) {
                bail!("permutation is not a permutation of 0..{n}");
            }
        }
        if scale.iter().any(|s| !(s.is_finite() && *s != 0.0)) {
            bail!("scale must be finite and nonzero");
        }
        Ok(Self {
            graph,
            conditional,
            param_offset,
            theta,
            permutation,
            loc,
            scale,
        })
    }

    fn n_var(&self) -> usize {
        self.graph.n_var()
    }

    /// Variable `i`'s slice of `theta`.
    fn theta_of(&self, i: usize) -> &[f64] {
        &self.theta[self.param_offset.range(i)]
    }

    /// Each variable's level: the longest parent chain ending at it.
    fn levels(&self) -> Vec<usize> {
        let graph = &self.graph;
        let mut level = vec![0usize; self.n_var()];
        for i in 0..level.len() {
            level[i] = graph
                .entries(i)
                .map(|(_, p)| level[p] + 1)
                .max()
                .unwrap_or(0);
        }
        level
    }

    /// The same flow with the diagonal affine `loc + scale * w`: the
    /// standardized coordinates change, and the conditioners' weights with
    /// them where that is exact (see [`Conditioners::change_coordinates`]).
    fn with_affine(&self, loc: Vec<f64>, scale: Vec<f64>) -> Result<Self> {
        let n = self.n_var();
        if loc.len() != n || scale.len() != n {
            bail!("loc and scale must each have one entry per variable");
        }
        if scale.iter().any(|s| !(s.is_finite() && *s != 0.0)) {
            bail!("scale must be finite and nonzero");
        }
        // Variable `j` of the map, `v_j = (y_k - loc_k) / scale_k` with `k =
        // permutation[j]`, becomes `a_j v_j + b_j`.
        let change: Vec<(f64, f64)> = self
            .permutation
            .iter()
            .map(|&k| {
                let a = self.scale[k] / scale[k];
                (a, (self.loc[k] - loc[k]) / scale[k])
            })
            .collect();
        if change.iter().any(|&(a, _)| a <= 0.0) {
            bail!("scale must keep its sign");
        }
        let mut out = self.clone();
        let mut parents = Vec::with_capacity(self.graph.max_parent());
        for i in 0..n {
            parents.clear();
            parents.extend(self.graph.parents(i).iter().map(|&p| change[p]));
            out.conditional.conditioners.change_coordinates(
                &self.conditional.transformer,
                change[i],
                &parents,
                &mut out.theta[self.param_offset.range(i)],
            );
        }
        out.loc = loc;
        out.scale = scale;
        Ok(out)
    }

    /// The same flow with the conditioners `configs`: what both have
    /// carries over (see [`Conditioners::carry_plan`]), the rest starts as
    /// new conditioners do, drawn from `seed`.
    fn with_conditioners(&self, configs: &[ConditionerConfig], seed: u64) -> Result<Self> {
        let transformer = self.conditional.transformer.clone();
        let conditioners = Conditioners::new(configs, &transformer)?;
        let plan = conditioners.carry_plan(&self.conditional.conditioners)?;
        let mut out = Self::new(
            self.graph.clone(),
            Conditional::new(conditioners, transformer)?,
            None,
            seed,
            self.permutation.clone(),
            self.loc.clone(),
            self.scale.clone(),
        )?;
        for i in 0..self.n_var() {
            out.conditional.conditioners.carry_weights(
                &self.conditional.conditioners,
                &plan,
                self.graph.n_parent(i),
                self.theta_of(i),
                &mut out.theta[out.param_offset.range(i)],
            );
        }
        Ok(out)
    }

    /// Draws and their gradients, row-major `(n_draw, n_var)` each, with the
    /// diagonal affine undone and reordered into the triangular order: what
    /// the triangular map and its LM residuals see.
    fn undo_affine(&self, positions: &[f64], gradients: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
        let n = self.n_var();
        if positions.len() != gradients.len() || !positions.len().is_multiple_of(n) {
            bail!("positions and gradients must both have shape (n_draw, {n})");
        }
        let mut y = vec![0.0; positions.len()];
        let mut g = vec![0.0; positions.len()];
        for ((pos, grad), (y, g)) in positions
            .chunks_exact(n)
            .zip(gradients.chunks_exact(n))
            .zip(y.chunks_exact_mut(n).zip(g.chunks_exact_mut(n)))
        {
            for (j, &k) in self.permutation.iter().enumerate() {
                y[j] = (pos[k] - self.loc[k]) / self.scale[k];
                g[j] = grad[k] * self.scale[k];
            }
        }
        Ok((y, g))
    }

    /// The LM residuals of the draws, see [`FisherResiduals`].
    fn problem(&self, positions: &[f64], gradients: &[f64]) -> Result<FisherResiduals> {
        let mut problem = FisherResiduals::new(self.graph.clone(), self.conditional.clone())?;
        let (y, g) = self.undo_affine(positions, gradients)?;
        problem.set_data(y, g)?;
        Ok(problem)
    }

    /// The mean Fisher divergence of the draws.
    fn fisher_divergence(&self, positions: &[f64], gradients: &[f64]) -> Result<f64> {
        let (r, _) = self.problem(positions, gradients)?.residuals(&self.theta)?;
        Ok(r.iter().map(|r| r * r).sum())
    }

    /// The squared Fisher speeds of every edge over the draws, see
    /// [`FisherResiduals::fisher_speeds`].
    fn fisher_speeds(
        &self,
        positions: &[f64],
        nodes: &[f64],
        weights: &[f64],
    ) -> Result<FisherSpeeds> {
        let gradients = vec![0.0; positions.len()];
        self.problem(positions, &gradients)?
            .fisher_speeds(&self.theta, nodes, weights)
    }

    /// The gate eigenvalues over the draws, row-major `(n_var,
    /// hidden_width)`, and `hidden_width`, see
    /// [`FisherResiduals::gate_eigenvalues`].
    fn gate_eigenvalues(&self, positions: &[f64], gradients: &[f64]) -> Result<(Vec<f64>, usize)> {
        let problem = self.problem(positions, gradients)?;
        let (_, lin) = problem.residuals(&self.theta)?;
        Ok((problem.gate_eigenvalues(&lin)?, problem.hidden_width()))
    }

    /// One draw `y` and its gradient mapped back to the base space: `(z,
    /// grad_z, log_det)`, with `log_det` the forward transform's at `z`, so
    /// that `log p(y) + log_det` is the base density at `z` and `grad_z` its
    /// gradient. As `transform_adapter._inv_transform` on the JAX flow.
    fn inverse(&self, position: &[f64], gradient: &[f64]) -> Result<(Vec<f64>, Vec<f64>, f64)> {
        let n = self.n_var();
        if position.len() != n || gradient.len() != n {
            bail!("position and gradient must have length {n}");
        }
        // The residuals run on whole SIMD tiles of draws: fill one with
        // copies of the draw.
        let width = FisherResiduals::native_width();
        let positions = position.repeat(width);
        let gradients = gradient.repeat(width);
        let (_, lin) = self
            .problem(&positions, &gradients)?
            .residuals(&self.theta)?;
        let (x, w, delta) = lin.x_w_delta_rows();

        let mut z = vec![0.0; n];
        let mut grad_z = vec![0.0; n];
        for (j, &k) in self.permutation.iter().enumerate() {
            z[k] = x[j];
            grad_z[k] = w[j];
        }
        // `delta` is the map's inverse's Jacobian diagonal, `dx_i / dv_i`.
        let log_det = self.scale.iter().map(|s| s.abs().ln()).sum::<f64>()
            - delta[..n].iter().map(|d| d.abs().ln()).sum::<f64>();
        Ok((z, grad_z, log_det))
    }

    /// The native transform of this flow, for the sampler's leapfrog.
    fn flow_transform(&self, schedule: &str, min_parallel_work: i64) -> Result<FlowTransform> {
        let n = self.n_var();
        let schedule: Schedule =
            from_tag(schedule).map_err(|_| anyhow::anyhow!("unknown schedule {schedule:?}"))?;

        let mut parent_indptr = vec![0i64];
        let mut parent_index = Vec::with_capacity(self.graph.n_edge());
        for i in 0..n {
            parent_index.extend(self.graph.entries(i).map(|(_, p)| p as i64));
            parent_indptr.push(parent_index.len() as i64);
        }
        let parents: Pattern<u32> = Pattern::from_i64(&parent_indptr, &parent_index)?;

        let level = self.levels();
        let n_levels = level.iter().max().map_or(0, |l| l + 1);
        let mut level_vars: Vec<u32> = (0..n as u32).collect();
        level_vars.sort_by_key(|&i| level[i as usize]);
        let mut level_len = vec![0usize; n_levels];
        let mut level_work = vec![0i64; n_levels];
        let layer_work =
            TRANSFORMER_WORK_PER_LAYER * self.conditional.transformer.n_layers() as i64;
        for i in 0..n {
            level_len[level[i]] += 1;
            level_work[level[i]] += self.param_offset.row_len(i) as i64 + layer_work;
        }
        let levels = Ragged::new(Offsets::from_lens(level_len), level_vars)?;

        let model = ConditionalModel::new(
            self.conditional.clone(),
            n,
            |i| self.graph.n_parent(i),
            |i| self.theta_of(i),
        );
        let map = TriangularTransform::new(
            parents,
            levels,
            &level_work,
            min_parallel_work,
            schedule,
            model,
        )?;
        FlowTransform::new(
            Some(Map::Conditional(map)),
            self.permutation.clone(),
            self.loc.clone(),
            self.scale.clone(),
        )
    }
}

/// `y` and `g`, `(n_draw, n_var)`, as contiguous slices.
fn rows<'a>(array: &'a PyReadonlyArray2<'_, f64>, name: &str) -> Result<&'a [f64]> {
    match array.as_slice() {
        Ok(slice) => Ok(slice),
        Err(_) => bail!("{name} must be a C-contiguous (n_draw, n_var) array"),
    }
}

/// A triangular flow built in Rust; see `nutpie.triangular_flow`.
#[pyclass(name = "TriangularFlow")]
pub struct PyTriangularFlow {
    inner: TriangularFlow,
}

impl PyTriangularFlow {
    /// The sampler's native transform, one chain's: serial.
    pub(crate) fn native_transform(&self) -> Result<FlowTransform> {
        self.inner.flow_transform("serial", 20_000)
    }
}

#[pymethods]
impl PyTriangularFlow {
    /// A flow with the given parents (in the triangular order), transformer
    /// (``{"kind": ...}``) and conditioners (a list of ``{"kind": ...}``),
    /// its conditioners' parameters `theta` in the LM layout, or new ones
    /// drawn from `seed`.
    #[new]
    #[pyo3(signature = (
        *,
        parent_indptr,
        parent_index,
        transformer,
        conditioners,
        permutation,
        loc,
        scale,
        theta = None,
        seed = 0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        parent_indptr: PyReadonlyArray1<'_, i64>,
        parent_index: PyReadonlyArray1<'_, i64>,
        transformer: &Bound<'_, PyAny>,
        conditioners: &Bound<'_, PyAny>,
        permutation: PyReadonlyArray1<'_, i64>,
        loc: PyReadonlyArray1<'_, f64>,
        scale: PyReadonlyArray1<'_, f64>,
        theta: Option<PyReadonlyArray1<'_, f64>>,
        seed: u64,
    ) -> Result<Self> {
        let config: TransformerConfig = crate::pyjson::from_py(transformer)?;
        let transformer = config.build()?;
        let configs: Vec<ConditionerConfig> = crate::pyjson::from_py(conditioners)?;
        let conditional =
            Conditional::new(Conditioners::new(&configs, &transformer)?, transformer)?;
        let graph = Pattern::from_i64(parent_indptr.as_slice()?, parent_index.as_slice()?)?;
        let theta = theta
            .map(|theta| theta.as_slice().map(<[f64]>::to_vec))
            .transpose()?;
        let permutation = permutation
            .as_slice()?
            .iter()
            .map(|&p| usize::try_from(p))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(Self {
            inner: TriangularFlow::new(
                graph,
                conditional,
                theta,
                seed,
                permutation,
                loc.as_slice()?.to_vec(),
                scale.as_slice()?.to_vec(),
            )?,
        })
    }

    /// Variable `i`'s conditioners' ranges of `theta`, ``(start, stop)``
    /// each: the constant, the linear conditioner, then the MLPs in list
    /// order, each only if there is one.
    fn weight_ranges(&self, i: usize) -> Result<Vec<(usize, usize)>> {
        let s = &self.inner;
        if i >= s.n_var() {
            bail!("variable {i} out of range");
        }
        let start = s.param_offset.start(i);
        Ok(s.conditional
            .conditioners
            .weight_ranges(s.graph.n_parent(i))
            .into_iter()
            .map(|r| (start + r.start, start + r.end))
            .collect())
    }

    #[getter]
    fn n_var(&self) -> usize {
        self.inner.n_var()
    }

    #[getter]
    fn n_params(&self) -> usize {
        self.inner.param_offset.total()
    }

    /// The largest number of parents of any variable.
    #[getter]
    fn max_parents(&self) -> usize {
        self.inner.graph.max_parent()
    }

    /// The number of levels, the longest parent chain plus one.
    #[getter]
    fn n_levels(&self) -> usize {
        self.inner.levels().iter().max().map_or(0, |l| l + 1)
    }

    /// The conditioners' parameters, in the LM layout.
    #[getter]
    fn theta<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.theta.clone())
    }

    #[getter]
    fn permutation<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_vec(
            py,
            self.inner.permutation.iter().map(|&p| p as i64).collect(),
        )
    }

    #[getter]
    fn loc<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.loc.clone())
    }

    #[getter]
    fn scale<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.scale.clone())
    }

    /// The same flow with the conditioners' parameters `theta`.
    fn with_theta(&self, theta: PyReadonlyArray1<'_, f64>) -> Result<Self> {
        let mut inner = self.inner.clone();
        let theta = theta.as_slice()?;
        if theta.len() != inner.theta.len() {
            bail!(
                "theta has length {}, expected {}",
                theta.len(),
                inner.theta.len()
            );
        }
        inner.theta = theta.to_vec();
        Ok(Self { inner })
    }

    /// The same flow with the diagonal affine `loc + scale * w`, one entry
    /// per variable. The conditioners' constant and linear weights on the
    /// location, and their constant on the scale, follow the new
    /// standardized coordinates; the rest keeps its values.
    fn with_affine(
        &self,
        loc: PyReadonlyArray1<'_, f64>,
        scale: PyReadonlyArray1<'_, f64>,
    ) -> Result<Self> {
        Ok(Self {
            inner: self
                .inner
                .with_affine(loc.as_slice()?.to_vec(), scale.as_slice()?.to_vec())?,
        })
    }

    /// The same flow with the conditioners `conditioners` (a list of
    /// ``{"kind": ...}``): each one's weights carry over from the matching
    /// current conditioner where they feed the same coefficients (an MLP
    /// matches one with the same width, activation and input squash whose
    /// targets overlap), new targets start at zero, and new conditioners as
    /// in a new flow, drawn from `seed`.
    #[pyo3(signature = (conditioners, seed = 0))]
    fn with_conditioners(&self, conditioners: &Bound<'_, PyAny>, seed: u64) -> Result<Self> {
        let configs: Vec<ConditionerConfig> = crate::pyjson::from_py(conditioners)?;
        Ok(Self {
            inner: self.inner.with_conditioners(&configs, seed)?,
        })
    }

    /// Draws and gradients, `(n_draw, n_var)` each, with the diagonal affine
    /// undone and reordered into the triangular order.
    fn undo_affine<'py>(
        &self,
        py: Python<'py>,
        positions: PyReadonlyArray2<'py, f64>,
        gradients: PyReadonlyArray2<'py, f64>,
    ) -> Result<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let shape = [positions.shape()[0], positions.shape()[1]];
        let (y, g) = self.inner.undo_affine(
            rows(&positions, "positions")?,
            rows(&gradients, "gradients")?,
        )?;
        Ok((
            PyArray1::from_vec(py, y).reshape(shape)?,
            PyArray1::from_vec(py, g).reshape(shape)?,
        ))
    }

    /// The LM residuals of the draws (`(n_draw, n_var)` each, `n_draw` a
    /// multiple of `FisherResiduals.simd_width()`), for `triangular_lm.fit`.
    fn residuals(
        &self,
        positions: PyReadonlyArray2<'_, f64>,
        gradients: PyReadonlyArray2<'_, f64>,
    ) -> Result<PyFisherResiduals> {
        Ok(PyFisherResiduals::from_inner(self.inner.problem(
            rows(&positions, "positions")?,
            rows(&gradients, "gradients")?,
        )?))
    }

    /// The mean Fisher divergence of the draws.
    fn fisher_divergence(
        &self,
        py: Python<'_>,
        positions: PyReadonlyArray2<'_, f64>,
        gradients: PyReadonlyArray2<'_, f64>,
    ) -> Result<f64> {
        let positions = rows(&positions, "positions")?;
        let gradients = rows(&gradients, "gradients")?;
        py.detach(|| self.inner.fisher_divergence(positions, gradients))
    }

    /// `(log_det, z, grad_z)` of one draw, as the JAX adapter's
    /// `inv_transform`.
    fn inv_transform<'py>(
        &self,
        py: Python<'py>,
        position: PyReadonlyArray1<'py, f64>,
        gradient: PyReadonlyArray1<'py, f64>,
    ) -> Result<(f64, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let (position, gradient) = (position.as_slice()?, gradient.as_slice()?);
        let (z, grad_z, log_det) = py.detach(|| self.inner.inverse(position, gradient))?;
        Ok((
            log_det,
            PyArray1::from_vec(py, z),
            PyArray1::from_vec(py, grad_z),
        ))
    }

    /// Per edge `(child, parent)`, in the triangular order, the squared
    /// Fisher speeds of the child's conditional per unit of the parent,
    /// averaged over the draws `positions` (`(n_draw, n_var)`, `n_draw` a
    /// multiple of `FisherResiduals.simd_width()`): ``(child, parent,
    /// location_skew, scale_tails)``. `nodes` and `weights` are a
    /// Gauss-Hermite rule for the standard normal, the nodes symmetric about
    /// zero and the weights summing to one. See `nutpie.flow_diagnostics`.
    #[allow(clippy::type_complexity)]
    fn fisher_speeds<'py>(
        &self,
        py: Python<'py>,
        positions: PyReadonlyArray2<'py, f64>,
        nodes: PyReadonlyArray1<'py, f64>,
        weights: PyReadonlyArray1<'py, f64>,
    ) -> Result<(
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        let positions = rows(&positions, "positions")?;
        let (nodes, weights) = (nodes.as_slice()?, weights.as_slice()?);
        let speeds = py.detach(|| self.inner.fisher_speeds(positions, nodes, weights))?;
        let graph = &self.inner.graph;
        let (child, parent): (Vec<i64>, Vec<i64>) = (0..graph.n_var())
            .flat_map(|i| graph.entries(i).map(move |(_, p)| (i as i64, p as i64)))
            .unzip();
        Ok((
            PyArray1::from_vec(py, child),
            PyArray1::from_vec(py, parent),
            PyArray1::from_vec(py, speeds.location_skew),
            PyArray1::from_vec(py, speeds.scale_tails),
        ))
    }

    /// How much of its MLPs' hidden units each variable's fit uses, over
    /// the draws (`(n_draw, n_var)` each, `n_draw` a multiple of
    /// `FisherResiduals.simd_width()`): ``(n_var, hidden_width)``, a row per
    /// variable in the triangular order, decreasing. See
    /// `nutpie.flow_diagnostics.conditioner_capacity`.
    fn gate_eigenvalues<'py>(
        &self,
        py: Python<'py>,
        positions: PyReadonlyArray2<'py, f64>,
        gradients: PyReadonlyArray2<'py, f64>,
    ) -> Result<Bound<'py, PyArray2<f64>>> {
        let positions = rows(&positions, "positions")?;
        let gradients = rows(&gradients, "gradients")?;
        let (values, width) = py.detach(|| self.inner.gate_eigenvalues(positions, gradients))?;
        Ok(PyArray1::from_vec(py, values).reshape([self.inner.n_var(), width])?)
    }

    /// Each variable's number of parents, in the triangular order.
    #[getter]
    fn n_parents<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        let graph = &self.inner.graph;
        PyArray1::from_vec(
            py,
            (0..graph.n_var())
                .map(|i| graph.n_parent(i) as i64)
                .collect(),
        )
    }

    /// The native transform for the sampler, see `FlowTransform`.
    #[pyo3(signature = (schedule = "serial", min_parallel_work = 20_000))]
    fn flow_transform(&self, schedule: &str, min_parallel_work: i64) -> Result<PyFlowTransform> {
        Ok(PyFlowTransform::from_flow(
            self.inner.flow_transform(schedule, min_parallel_work)?,
        ))
    }
}
