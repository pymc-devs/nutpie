//! A triangular flow held in Rust: the map's parents and conditioner, its
//! parameters in the LM layout (see [`super`]), and the flow's permutation
//! and diagonal affine, as `make_flow(kind="triangular")` builds them in JAX.
//! Built and initialized by `nutpie.triangular_flow`.
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

use super::conditioner::{Activation, Conditioner};
use super::{FisherResiduals, PyFisherResiduals};
use crate::triangular::layers::LayerSpec;
use crate::triangular::pattern::Pattern;
use crate::triangular::transform::{build_transform_from_specs, FlowTransform, PyFlowTransform};

/// Rough cost of one transformer layer in multiply-adds, for scheduling the
/// native transform's levels, as `nutpie.triangular_layout`.
const TRANSFORMER_WORK_PER_LAYER: i64 = 100;

#[derive(Clone)]
pub(crate) struct TriangularFlow {
    /// The map's structure, without data.
    structure: FisherResiduals,
    /// The transformer's layers, for the native transform.
    specs: Vec<LayerSpec>,
    theta: Vec<f64>,
    permutation: Vec<usize>,
    loc: Vec<f64>,
    scale: Vec<f64>,
}

impl TriangularFlow {
    fn new(
        structure: FisherResiduals,
        specs: Vec<LayerSpec>,
        theta: Vec<f64>,
        permutation: Vec<usize>,
        loc: Vec<f64>,
        scale: Vec<f64>,
    ) -> Result<Self> {
        let n = structure.n_var();
        if theta.len() != structure.n_params() {
            bail!(
                "theta has length {}, expected {}",
                theta.len(),
                structure.n_params()
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
            structure,
            specs,
            theta,
            permutation,
            loc,
            scale,
        })
    }

    fn n_var(&self) -> usize {
        self.structure.n_var()
    }

    /// Each variable's level: the longest parent chain ending at it.
    fn levels(&self) -> Vec<usize> {
        let graph = &self.structure.graph;
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
    fn problem(
        &self,
        positions: &[f64],
        gradients: &[f64],
        fisher_regularization: Option<f64>,
    ) -> Result<FisherResiduals> {
        let mut problem = FisherResiduals::new(
            self.structure.graph.clone(),
            self.structure.conditioner.clone(),
            fisher_regularization,
        )?;
        let (y, g) = self.undo_affine(positions, gradients)?;
        problem.set_data(y, g)?;
        Ok(problem)
    }

    /// The mean Fisher divergence of the draws.
    fn fisher_divergence(&self, positions: &[f64], gradients: &[f64]) -> Result<f64> {
        let (r, _) = self
            .problem(positions, gradients, None)?
            .residuals(&self.theta)?;
        Ok(r.iter().map(|r| r * r).sum())
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
            .problem(&positions, &gradients, None)?
            .residuals(&self.theta)?;
        let (x, w, delta) = (lin.x.to_rows(), lin.w.to_rows(), lin.delta.to_rows());

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
        let s = &self.structure;
        let (n, conditioner) = (s.n_var(), &s.conditioner);
        let (n_unit, n_par) = (conditioner.n_unit, conditioner.n_par);

        let mut parent_indptr = vec![0i64];
        let mut parent_index = Vec::with_capacity(s.n_edge());
        let mut skip_weight = Vec::with_capacity(s.n_edge());
        let mut blob = Vec::new();
        let mut blob_offset = vec![0i64];
        for i in 0..n {
            let theta = s.params_of(&self.theta, i);
            let n_parent = theta.shape.n_parent;
            parent_index.extend(s.graph.entries(i).map(|(_, p)| p as i64));
            parent_indptr.push(parent_index.len() as i64);
            skip_weight.extend_from_slice(theta.skip());
            // Per layer, the transposed `(n_in, n_out)` weight, then the
            // bias: see `TriangularLayout.blob`.
            for p in 0..n_parent {
                blob.extend((0..n_unit).map(|u| theta.w1(u)[p]));
            }
            blob.extend((0..n_unit).map(|u| theta.b1(u)));
            for u in 0..n_unit {
                blob.extend_from_slice(theta.w2(u));
            }
            blob.extend_from_slice(theta.b2());
            blob_offset.push(blob.len() as i64);
        }

        let level = self.levels();
        let n_levels = level.iter().max().map_or(0, |l| l + 1);
        let mut level_vars: Vec<i64> = (0..n as i64).collect();
        level_vars.sort_by_key(|&i| level[i as usize]);
        let mut level_ptr = vec![0i64; n_levels + 1];
        let mut level_work = vec![0i64; n_levels];
        for i in 0..n {
            level_ptr[level[i] + 1] += 1;
            level_work[level[i]] += blob_offset[i + 1] - blob_offset[i]
                + TRANSFORMER_WORK_PER_LAYER * self.specs.len() as i64;
        }
        for l in 0..n_levels {
            level_ptr[l + 1] += level_ptr[l];
        }

        let map = build_transform_from_specs(
            &parent_indptr,
            &parent_index,
            &blob,
            &blob_offset,
            &[n_unit as i64, n_par as i64],
            &skip_weight,
            conditioner.location as i64,
            0,
            &[],
            conditioner.squash,
            conditioner.activation.name(),
            self.specs.clone(),
            &level_ptr,
            &level_vars,
            &level_work,
            min_parallel_work,
            schedule,
        )?;
        FlowTransform::new(
            Some(map),
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
    #[new]
    #[pyo3(signature = (
        *,
        parent_indptr,
        parent_index,
        n_unit,
        n_par,
        location_index,
        transformer,
        theta,
        permutation,
        loc,
        scale,
        input_squash = None,
        activation = "softplus",
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        parent_indptr: PyReadonlyArray1<'_, i64>,
        parent_index: PyReadonlyArray1<'_, i64>,
        n_unit: usize,
        n_par: usize,
        location_index: usize,
        transformer: &Bound<'_, PyAny>,
        theta: PyReadonlyArray1<'_, f64>,
        permutation: PyReadonlyArray1<'_, i64>,
        loc: PyReadonlyArray1<'_, f64>,
        scale: PyReadonlyArray1<'_, f64>,
        input_squash: Option<f64>,
        activation: &str,
    ) -> Result<Self> {
        let specs: Vec<LayerSpec> = pythonize::depythonize(transformer)?;
        let conditioner = Conditioner::new(
            n_unit,
            n_par,
            location_index,
            specs.clone(),
            input_squash,
            Activation::from_name(activation)?,
        )?;
        let graph = Pattern::from_i64(parent_indptr.as_slice()?, parent_index.as_slice()?)?;
        let permutation = permutation
            .as_slice()?
            .iter()
            .map(|&p| usize::try_from(p))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(Self {
            inner: TriangularFlow::new(
                FisherResiduals::new(graph, conditioner, None)?,
                specs,
                theta.as_slice()?.to_vec(),
                permutation,
                loc.as_slice()?.to_vec(),
                scale.as_slice()?.to_vec(),
            )?,
        })
    }

    #[getter]
    fn n_var(&self) -> usize {
        self.inner.n_var()
    }

    #[getter]
    fn n_params(&self) -> usize {
        self.inner.structure.n_params()
    }

    /// The largest number of parents of any variable.
    #[getter]
    fn max_parents(&self) -> usize {
        self.inner.structure.graph.max_parent()
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
    #[pyo3(signature = (positions, gradients, fisher_regularization = None))]
    fn residuals(
        &self,
        positions: PyReadonlyArray2<'_, f64>,
        gradients: PyReadonlyArray2<'_, f64>,
        fisher_regularization: Option<f64>,
    ) -> Result<PyFisherResiduals> {
        Ok(PyFisherResiduals::from_inner(self.inner.problem(
            rows(&positions, "positions")?,
            rows(&gradients, "gradients")?,
            fisher_regularization,
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

    /// The native transform for the sampler, see `FlowTransform`.
    #[pyo3(signature = (schedule = "serial", min_parallel_work = 20_000))]
    fn flow_transform(&self, schedule: &str, min_parallel_work: i64) -> Result<PyFlowTransform> {
        Ok(PyFlowTransform::from_flow(
            self.inner.flow_transform(schedule, min_parallel_work)?,
        ))
    }
}
