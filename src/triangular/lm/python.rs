//! The Python class.

use anyhow::{bail, Result};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::{FisherResiduals, Linearization};
use crate::triangular::conditional::Conditional;
use crate::triangular::conditioner::{ConditionerConfig, Conditioners};
use crate::triangular::pattern::Pattern;
use crate::triangular::transformer::layers::LayerConfig;
use crate::triangular::transformer::LayerChain;

/// Residuals of the LM fit of a `SparseTriangularMap` and their derivatives,
/// see `nutpie.triangular_lm`.
#[pyclass(name = "FisherResiduals")]
pub struct PyFisherResiduals {
    pub(crate) inner: FisherResiduals,
    lin: Option<Linearization>,
}

impl PyFisherResiduals {
    pub(crate) fn from_inner(inner: FisherResiduals) -> Self {
        Self { inner, lin: None }
    }

    fn linearization(&self) -> Result<&Linearization> {
        match &self.lin {
            Some(lin) => Ok(lin),
            None => bail!("no linearization: call `residuals(theta, linearize=True)` first"),
        }
    }
}

#[pymethods]
impl PyFisherResiduals {
    /// The residuals of a map with the given parents, a transformer of
    /// `transformer`'s layers (as `nutpie.triangular_layout.transformer_dicts`
    /// writes them) on `n_par` coefficients with the location and the scale
    /// at the given indices, and `conditioners` (``{"kind": ...}`` dicts).
    #[new]
    #[pyo3(signature = (
        *,
        parent_indptr,
        parent_index,
        transformer,
        n_par,
        location_index,
        conditioners,
        scale_index = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        parent_indptr: PyReadonlyArray1<'_, i64>,
        parent_index: PyReadonlyArray1<'_, i64>,
        transformer: &Bound<'_, PyAny>,
        n_par: usize,
        location_index: usize,
        conditioners: &Bound<'_, PyAny>,
        scale_index: Option<usize>,
    ) -> Result<Self> {
        let configs: Vec<LayerConfig> = crate::pyjson::from_py(transformer)?;
        let transformer = LayerChain::new(configs, n_par, Some(location_index), scale_index)?;
        let conditioners: Vec<ConditionerConfig> = crate::pyjson::from_py(conditioners)?;
        let conditional =
            Conditional::new(Conditioners::new(&conditioners, &transformer)?, transformer)?;
        Ok(Self {
            inner: FisherResiduals::new(
                Pattern::from_i64(parent_indptr.as_slice()?, parent_index.as_slice()?)?,
                conditional,
            )?,
            lin: None,
        })
    }

    /// Lanes of this machine's native ``f64`` vector: `set_data` needs a
    /// multiple of this many draws.
    #[staticmethod]
    fn simd_width() -> usize {
        FisherResiduals::native_width()
    }

    #[getter]
    fn n_params(&self) -> usize {
        self.inner.n_params()
    }

    #[getter]
    fn n_residuals(&self) -> usize {
        self.inner.n_residuals()
    }

    #[getter]
    fn n_draw(&self) -> usize {
        self.inner.n_draw()
    }

    /// Per parameter, the strength `LmOptimizer`'s ridge puts on it at
    /// `theta`, before the division by ``n_draw``. What feeds a variable's
    /// location is scaled by the mean of its ``(dx_i/dy_i)^2`` over the
    /// draws at `theta`.
    fn ridge_strengths<'py>(
        &self,
        py: Python<'py>,
        theta: PyReadonlyArray1<'py, f64>,
    ) -> Result<Bound<'py, PyArray1<f64>>> {
        let theta = theta.as_slice()?.to_vec();
        let (_, lin) = py.detach(|| self.inner.residuals(&theta))?;
        let values = self.inner.ridge_strengths(&self.inner.loc_scales(&lin));
        Ok(PyArray1::from_vec(py, values))
    }

    /// Per parameter, whether it makes its conditional nonlinear or
    /// non-Gaussian: the MLPs' weights, the constant's into shape
    /// coefficients and the linear conditioner's into anything but the
    /// location. `LmOptimizer`'s ``frozen`` for a linear fit.
    #[getter]
    fn nonlinear_param_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        PyArray1::from_vec(py, self.inner.nonlinear_param_mask())
    }

    /// Variable `i`'s conditioners' ranges of the parameters, ``(start,
    /// stop)`` each: the constant, the linear conditioner, then the MLPs in
    /// list order, each only if there is one.
    fn weight_ranges(&self, i: usize) -> Result<Vec<(usize, usize)>> {
        if i >= self.inner.n_var() {
            bail!("variable {i} out of range");
        }
        let start = self.inner.param_offset.start(i);
        Ok(self
            .inner
            .conditional
            .conditioners
            .weight_ranges(self.inner.graph.n_parent(i))
            .into_iter()
            .map(|r| (start + r.start, start + r.end))
            .collect())
    }

    /// Start of each variable's parameter slice, plus the total.
    #[getter]
    fn param_offsets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_vec(
            py,
            self.inner
                .param_offset
                .as_slice()
                .iter()
                .map(|&o| o as i64)
                .collect(),
        )
    }

    /// Set the map-space draws and gradients, each ``(n_draw, n_var)``
    /// flattened. Drops the linearization, which belongs to the old data.
    fn set_data(
        &mut self,
        y: PyReadonlyArray1<'_, f64>,
        g: PyReadonlyArray1<'_, f64>,
    ) -> Result<()> {
        self.lin = None;
        self.inner
            .set_data(y.as_slice()?.to_vec(), g.as_slice()?.to_vec())
    }

    /// Flattened ``(n_draw, n_residuals)`` residuals at `theta`; with
    /// `linearize`, keeps the linearization there that the derivatives below
    /// use.
    #[pyo3(signature = (theta, linearize = true))]
    fn residuals<'py>(
        &mut self,
        py: Python<'py>,
        theta: PyReadonlyArray1<'py, f64>,
        linearize: bool,
    ) -> Result<Bound<'py, PyArray1<f64>>> {
        let theta = theta.as_slice()?.to_vec();
        let (out, lin) = py.detach(|| self.inner.residuals(&theta))?;
        if linearize {
            self.lin = Some(lin);
        }
        Ok(PyArray1::from_vec(py, out))
    }

    /// ``J v``, flattened ``(n_draw, n_residuals)``, at the linearization's `theta`.
    fn pushforward<'py>(
        &self,
        py: Python<'py>,
        v: PyReadonlyArray1<'py, f64>,
    ) -> Result<Bound<'py, PyArray1<f64>>> {
        let v = v.as_slice()?.to_vec();
        let lin = self.linearization()?;
        let out = py.detach(|| self.inner.pushforward(lin, &v))?;
        Ok(PyArray1::from_vec(py, out))
    }

    /// ``J^T r_bar`` for a flattened ``(n_draw, n_residuals)`` `r_bar`.
    fn pullback<'py>(
        &self,
        py: Python<'py>,
        r_bar: PyReadonlyArray1<'py, f64>,
    ) -> Result<Bound<'py, PyArray1<f64>>> {
        let r_bar = r_bar.as_slice()?.to_vec();
        let lin = self.linearization()?;
        let out = py.detach(|| self.inner.pullback(lin, &r_bar))?;
        Ok(PyArray1::from_vec(py, out))
    }

    /// ``J^T J v``.
    fn gauss_newton_product<'py>(
        &self,
        py: Python<'py>,
        v: PyReadonlyArray1<'py, f64>,
    ) -> Result<Bound<'py, PyArray1<f64>>> {
        let v = v.as_slice()?.to_vec();
        let lin = self.linearization()?;
        let out = py.detach(|| self.inner.gauss_newton_product(lin, &v))?;
        Ok(PyArray1::from_vec(py, out))
    }

    /// Exact Gauss-Newton sub-blocks: ``(starts, sizes, data)``, where block
    /// ``k`` covers parameters ``starts[k]:starts[k] + sizes[k]`` and is
    /// stored row-major in `data`, after the blocks before it.
    fn gauss_newton_blocks<'py>(
        &self,
        py: Python<'py>,
        max_block_size: usize,
    ) -> Result<(
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        let lin = self.linearization()?;
        let blocks = py.detach(|| self.inner.gauss_newton_blocks(lin, max_block_size))?;
        let starts = blocks.iter().map(|b| b.start as i64).collect();
        let sizes = blocks.iter().map(|b| b.matrix.nrows() as i64).collect();
        // Symmetric, so row- and column-major agree.
        let data = blocks
            .iter()
            .flat_map(|b| {
                (0..b.matrix.ncols()).flat_map(move |c| b.matrix.col_as_slice(c).iter().copied())
            })
            .collect();
        Ok((
            PyArray1::from_vec(py, starts),
            PyArray1::from_vec(py, sizes),
            PyArray1::from_vec(py, data),
        ))
    }
}
