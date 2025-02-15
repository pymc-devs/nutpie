use std::sync::Arc;
use std::{ffi::CString, path::PathBuf};

use anyhow::Context;
use arrow::array::{Array, FixedSizeListArray, Float64Array, StructArray};
use arrow::datatypes::{DataType, Field};
use bridgestan::open_library;
use itertools::{izip, Itertools};
use nuts_rs::{CpuLogpFunc, CpuMath, DrawStorage, LogpError, Model, Settings};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3::{exceptions::PyValueError, pyclass, pymethods, PyResult};
use rand::prelude::Distribution;
use rand::{rng, RngCore};
use rand_distr::StandardNormal;
use smallvec::{SmallVec, ToSmallVec};

use thiserror::Error;

use crate::wrapper::PyTransformAdapt;

type InnerModel = bridgestan::Model<Arc<bridgestan::StanLibrary>>;

#[pyclass]
#[derive(Clone)]
pub struct StanLibrary(Arc<bridgestan::StanLibrary>);

#[derive(Clone)]
struct Parameter {
    name: String,
    shape: Vec<usize>,
    size: usize,
    start_idx: usize,
    end_idx: usize,
}

#[pymethods]
impl StanLibrary {
    #[new]
    fn new(path: PathBuf) -> PyResult<Self> {
        let lib = open_library(path)
            .map_err(|e| PyValueError::new_err(format!("Could not open stan libray: {}", e)))?;
        Ok(Self(Arc::new(lib)))
    }
}

#[pyclass]
pub struct StanVariable(Parameter);

#[pymethods]
impl StanVariable {
    #[getter]
    fn name(&self) -> String {
        self.0.name.clone()
    }

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.0.shape.iter())
    }

    #[getter]
    fn size(&self) -> usize {
        self.0.size
    }
}

#[pyclass]
#[derive(Clone)]
pub struct StanModel {
    model: Arc<InnerModel>,
    variables: Vec<Parameter>,
    transform_adapter: Option<PyTransformAdapt>,
}

/// Return meta information about the constrained parameters of the model
fn params(
    model: &InnerModel,
    include_tp: bool,
    include_gq: bool,
) -> anyhow::Result<Vec<Parameter>> {
    let var_string = model.param_names(include_tp, include_gq);
    let name_idxs: anyhow::Result<Vec<(&str, Vec<usize>)>> = var_string
        .split(',')
        .map(|var| {
            let mut parts = var.split('.');
            let name = parts
                .next()
                .ok_or_else(|| anyhow::Error::msg("Invalid parameter name"))?;
            let idxs: anyhow::Result<Vec<usize>> = parts
                .map(|mut idx| {
                    if idx == "real" {
                        idx = "1";
                    }
                    if idx == "imag" {
                        idx = "2";
                    }
                    let idx: usize = idx
                        .parse()
                        .map_err(|_| anyhow::Error::msg("Invalid parameter name"))?;
                    Ok(idx - 1)
                })
                .collect();
            Ok((name, idxs?))
        })
        .collect();

    let mut variables = Vec::new();
    let mut start_idx = 0;
    for (name, idxs) in &name_idxs?.iter().chunk_by(|(name, _)| name) {
        let mut shape: Vec<usize> = idxs
            .map(|(_name, idx)| idx)
            .fold(None, |acc, elem| {
                let mut shape = acc.unwrap_or(elem.clone());
                shape
                    .iter_mut()
                    .zip_eq(elem.iter())
                    .for_each(|(old, &new)| {
                        *old = new.max(*old);
                    });
                Some(shape)
            })
            .unwrap_or(vec![]);
        shape.iter_mut().for_each(|max_idx| *max_idx += 1);
        let size = shape.iter().product();
        let end_idx = start_idx + size;
        variables.push(Parameter {
            name: name.to_string(),
            shape,
            size,
            start_idx,
            end_idx,
        });
        start_idx = end_idx;
    }
    Ok(variables)
}

#[pymethods]
impl StanModel {
    #[new]
    #[pyo3(signature = (lib, seed=None, data=None, transform_adapter=None))]
    pub fn new(
        lib: StanLibrary,
        seed: Option<u32>,
        data: Option<String>,
        transform_adapter: Option<Py<PyAny>>,
    ) -> anyhow::Result<Self> {
        let seed = match seed {
            Some(seed) => seed,
            None => rng().next_u32(),
        };
        let data: Option<CString> = data.map(CString::new).transpose()?;
        let model = Arc::new(
            bridgestan::Model::new(lib.0, data.as_ref(), seed).map_err(anyhow::Error::new)?,
        );
        let variables = params(&model, true, true)?;
        let transform_adapter = transform_adapter.map(PyTransformAdapt::new);
        Ok(StanModel {
            model,
            variables,
            transform_adapter,
        })
    }

    pub fn variables<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        let results: Result<Vec<_>, _> = self
            .variables
            .iter()
            .map(|var| {
                out.set_item(
                    var.name.clone(),
                    StanVariable(var.clone()).into_pyobject(py)?,
                )
            })
            .collect();
        results?;
        Ok(out)
    }

    pub fn ndim(&self) -> usize {
        self.model.param_unc_num()
    }

    pub fn param_unc_names(&mut self) -> anyhow::Result<Vec<String>> {
        Ok(Arc::get_mut(&mut self.model)
            .ok_or_else(|| anyhow::format_err!("Model is currently in use"))
            .context("Failed to access the names of unconstrained parameters")?
            .param_unc_names()
            .split(',')
            .map(|name| name.to_string())
            .collect())
    }

    /*
    fn benchmark_logp<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray1<'py, f64>,
        cores: usize,
        evals: usize,
    ) -> PyResult<&'py PyList> {
        let point = point.as_slice()?;
        let durations = py.allow_threads(|| Model::benchmark_logp(self, &point, cores, evals))?;
        let out = PyList::new(
            py,
            durations
                .into_iter()
                .map(|inner| PyList::new(py, inner.into_iter().map(|d| d.as_secs_f64()))),
        );
        Ok(out)
    }
    */
}

pub struct StanDensity<'model> {
    inner: &'model InnerModel,
    transform_adapter: Option<PyTransformAdapt>,
}

#[derive(Debug, Error)]
pub enum StanLogpError {
    #[error("Error during logp evaluation: {0}")]
    BridgeStan(#[from] bridgestan::BridgeStanError),
    #[error("Bad logp value: {0}")]
    BadLogp(f64),
    #[error("Python exception: {0}")]
    PyErr(#[from] PyErr),
    #[error("Unspecified Error: {0}")]
    Anyhow(#[from] anyhow::Error),
}

impl LogpError for StanLogpError {
    fn is_recoverable(&self) -> bool {
        true
    }
}

impl<'model> CpuLogpFunc for StanDensity<'model> {
    type LogpError = StanLogpError;
    type TransformParams = Py<PyAny>;

    fn logp(&mut self, position: &[f64], grad: &mut [f64]) -> Result<f64, Self::LogpError> {
        let logp = self
            .inner
            .log_density_gradient(position, true, true, grad)?;
        if !logp.is_finite() {
            return Err(StanLogpError::BadLogp(logp));
        }
        Ok(logp)
    }

    fn dim(&self) -> usize {
        self.inner.param_unc_num()
    }

    fn inv_transform_normalize(
        &mut self,
        params: &Py<PyAny>,
        untransformed_position: &[f64],
        untransformed_gradient: &[f64],
        transformed_position: &mut [f64],
        transformed_gradient: &mut [f64],
    ) -> std::result::Result<f64, Self::LogpError> {
        let logdet = self
            .transform_adapter
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("No transformation adapter specified"))?
            .inv_transform_normalize(
                params,
                untransformed_position,
                untransformed_gradient,
                transformed_position,
                transformed_gradient,
            )
            .context("failed inv_transform_normalize")?;
        Ok(logdet)
    }

    fn init_from_transformed_position(
        &mut self,
        params: &Py<PyAny>,
        untransformed_position: &mut [f64],
        untransformed_gradient: &mut [f64],
        transformed_position: &[f64],
        transformed_gradient: &mut [f64],
    ) -> std::result::Result<(f64, f64), Self::LogpError> {
        let adapter = self
            .transform_adapter
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("No transformation adapter specified"))?;

        let part1 = adapter
            .init_from_transformed_position_part1(
                params,
                untransformed_position,
                transformed_position,
            )
            .context("Failed init_from_transformed_position_part1")?;

        let logp = self.logp(untransformed_position, untransformed_gradient)?;

        let adapter = self
            .transform_adapter
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("No transformation adapter specified"))?;

        let logdet = adapter
            .init_from_transformed_position_part2(
                params,
                part1,
                untransformed_gradient,
                transformed_gradient,
            )
            .context("Failed init_from_transformed_position_part2")?;
        Ok((logp, logdet))
    }

    fn init_from_untransformed_position(
        &mut self,
        params: &Py<PyAny>,
        untransformed_position: &[f64],
        untransformed_gradient: &mut [f64],
        transformed_position: &mut [f64],
        transformed_gradient: &mut [f64],
    ) -> std::result::Result<(f64, f64), Self::LogpError> {
        let logp = self
            .logp(untransformed_position, untransformed_gradient)
            .context("Failed to call stan logp function")?;

        let logdet = self
            .transform_adapter
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("No transformation adapter specified"))?
            .inv_transform_normalize(
                params,
                untransformed_position,
                untransformed_gradient,
                transformed_position,
                transformed_gradient,
            )
            .context("Failed inv_transform_normalize in stan init_from_untransformed_position")?;
        Ok((logp, logdet))
    }

    fn update_transformation<'a, R: rand::Rng + ?Sized>(
        &'a mut self,
        rng: &mut R,
        untransformed_positions: impl ExactSizeIterator<Item = &'a [f64]>,
        untransformed_gradients: impl ExactSizeIterator<Item = &'a [f64]>,
        untransformed_logp: impl ExactSizeIterator<Item = &'a f64>,
        params: &'a mut Py<PyAny>,
    ) -> std::result::Result<(), Self::LogpError> {
        self.transform_adapter
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("No transformation adapter specified"))?
            .update_transformation(
                rng,
                untransformed_positions,
                untransformed_gradients,
                untransformed_logp,
                params,
            )
            .context("Failed to update the transformation")?;
        Ok(())
    }

    fn new_transformation<R: rand::Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        untransformed_position: &[f64],
        untransformed_gradient: &[f64],
        chain: u64,
    ) -> std::result::Result<Py<PyAny>, Self::LogpError> {
        let trafo = self
            .transform_adapter
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("No transformation adapter specified"))?
            .new_transformation(rng, untransformed_position, untransformed_gradient, chain)
            .context("Could not create transformation adapter")?;
        Ok(trafo)
    }

    fn transformation_id(&self, params: &Py<PyAny>) -> std::result::Result<i64, Self::LogpError> {
        let id = self
            .transform_adapter
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("No transformation adapter specified"))?
            .transformation_id(params)?;
        Ok(id)
    }
}

fn fortran_to_c_order(data: &[f64], shape: &[usize], out: &mut Vec<f64>) {
    let rank = shape.len();
    let strides = {
        let mut strides: SmallVec<[usize; 8]> = SmallVec::with_capacity(rank);
        let mut current: usize = 1;
        for &length in shape.iter() {
            strides.push(current);
            current = current
                .checked_mul(length)
                .expect("Overflow in stride computation");
        }
        strides.reverse();
        strides
    };

    let mut shape: SmallVec<[usize; 8]> = shape.to_smallvec();
    shape.reverse();

    let mut idx: SmallVec<[usize; 8]> = shape.iter().map(|_| 0usize).collect();
    let mut position: usize = 0;
    'iterate: loop {
        out.push(data[position]);

        let mut axis: usize = 0;
        'nextidx: loop {
            idx[axis] += 1;
            position += strides[axis];

            if idx[axis] < shape[axis] {
                break 'nextidx;
            }

            idx[axis] = 0;
            position -= shape[axis] * strides[axis];
            axis += 1;
            if axis == rank {
                break 'iterate;
            }
        }
    }
}

pub struct StanTrace<'model> {
    inner: &'model InnerModel,
    model: &'model StanModel,
    trace: Vec<Vec<f64>>,
    expanded_buffer: Box<[f64]>,
    rng: bridgestan::Rng<&'model bridgestan::StanLibrary>,
}

impl<'model> Clone for StanTrace<'model> {
    fn clone(&self) -> Self {
        // TODO We should avoid this Clone implementation.
        // We only need it for `StanTrace.inspect`, which
        // doesn't need rng, so we could avoid this strange
        // seed of zeros.
        let rng = self
            .model
            .model
            .new_rng(0)
            .expect("Could not create stan rng");
        Self {
            inner: self.inner,
            model: self.model,
            trace: self.trace.clone(),
            expanded_buffer: self.expanded_buffer.clone(),
            rng,
        }
    }
}

impl<'model> DrawStorage for StanTrace<'model> {
    fn append_value(&mut self, point: &[f64]) -> anyhow::Result<()> {
        self.inner
            .param_constrain(
                point,
                true,
                true,
                &mut self.expanded_buffer,
                Some(&mut self.rng),
            )
            .context("Failed to constrain the parameters of the draw")?;
        for (var, trace) in self.model.variables.iter().zip_eq(self.trace.iter_mut()) {
            let slice = &self.expanded_buffer[var.start_idx..var.end_idx];
            assert!(slice.len() == var.size);

            if var.size == 0 {
                continue;
            }

            // The slice is in fortran order. This doesn't matter if it low dim
            if var.shape.len() < 2 {
                trace.extend_from_slice(slice);
                continue;
            }

            // We need to transpose
            fortran_to_c_order(slice, &var.shape, trace);
        }
        Ok(())
    }

    fn finalize(self) -> anyhow::Result<Arc<dyn Array>> {
        let (fields, arrays): (Vec<_>, Vec<_>) = izip!(self.trace, &self.model.variables)
            .map(|(data, variable)| {
                let data = Float64Array::from(data);
                let item_field = Arc::new(Field::new("item", DataType::Float64, false));
                let array = FixedSizeListArray::new(
                    item_field.clone(),
                    variable.size as _,
                    Arc::new(data),
                    None,
                );
                let dtype = DataType::FixedSizeList(item_field, variable.size as i32);
                let field = Arc::new(Field::new(variable.name.clone(), dtype.clone(), false));
                let list: Arc<dyn Array> = Arc::new(array);
                (field, list)
            })
            .unzip();

        Ok(Arc::new(
            StructArray::try_new(fields.into(), arrays, None)
                .context("Could not create arrow StructArray")?,
        ))
    }

    fn inspect(&self) -> anyhow::Result<Arc<dyn Array>> {
        self.clone().finalize()
    }
}

impl Model for StanModel {
    type Math<'model> = CpuMath<StanDensity<'model>>;

    type DrawStorage<'model, S: nuts_rs::Settings> = StanTrace<'model>;

    fn new_trace<'a, S: Settings, R: rand::Rng + ?Sized>(
        &'a self,
        _rng: &mut R,
        chain: u64,
        settings: &S,
    ) -> anyhow::Result<Self::DrawStorage<'a, S>> {
        let draws = settings.hint_num_tune() + settings.hint_num_draws();
        let trace = self
            .variables
            .iter()
            .map(|var| Vec::with_capacity(var.size * draws))
            .collect();
        let rng = self.model.new_rng(chain as u32)?;
        let buffer = vec![0f64; self.model.param_num(true, true)];
        Ok(StanTrace {
            model: self,
            inner: &self.model,
            trace,
            rng,
            expanded_buffer: buffer.into(),
        })
    }

    fn math(&self) -> anyhow::Result<Self::Math<'_>> {
        Ok(CpuMath::new(StanDensity {
            inner: &self.model,
            transform_adapter: self.transform_adapter.as_ref().map(|v| v.clone()),
        }))
    }

    fn init_position<R: rand::Rng + ?Sized>(
        &self,
        rng: &mut R,
        position: &mut [f64],
    ) -> anyhow::Result<()> {
        let dist = StandardNormal;
        dist.sample_iter(rng)
            .zip(position.iter_mut())
            .for_each(|(val, pos)| *pos = val);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use super::fortran_to_c_order;

    #[test]
    fn transpose() {
        // Generate the expected values using code like
        // np.arange(2 * 3 * 5, dtype=float).reshape((2, 3, 5), order="F").ravel()

        let data = vec![0., 1., 2., 3., 4., 5.];
        let mut out = vec![];
        fortran_to_c_order(&data, &[2, 3], &mut out);
        let expect = [0., 2., 4., 1., 3., 5.];
        assert!(expect.iter().zip_eq(out.iter()).all(|(a, b)| a == b));

        let data = vec![0., 1., 2., 3., 4., 5.];
        let mut out = vec![];
        fortran_to_c_order(&data, &[3, 2], &mut out);
        let expect = [0., 3., 1., 4., 2., 5.];
        assert!(expect.iter().zip_eq(out.iter()).all(|(a, b)| a == b));

        let data = vec![
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
            19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29.,
        ];
        let mut out = vec![];
        fortran_to_c_order(&data, &[2, 3, 5], &mut out);
        let expect = vec![
            0., 6., 12., 18., 24., 2., 8., 14., 20., 26., 4., 10., 16., 22., 28., 1., 7., 13., 19.,
            25., 3., 9., 15., 21., 27., 5., 11., 17., 23., 29.,
        ];
        dbg!(&out);
        assert!(expect.iter().zip_eq(out.iter()).all(|(a, b)| a == b));

        let data = vec![
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
            19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29.,
        ];
        let mut out = vec![];
        fortran_to_c_order(&data, &[2, 3, 5], &mut out);
        let expect = vec![
            0., 6., 12., 18., 24., 2., 8., 14., 20., 26., 4., 10., 16., 22., 28., 1., 7., 13., 19.,
            25., 3., 9., 15., 21., 27., 5., 11., 17., 23., 29.,
        ];
        dbg!(&out);
        assert!(expect.iter().zip_eq(out.iter()).all(|(a, b)| a == b));

        let data = vec![
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
            19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29.,
        ];
        let mut out = vec![];
        fortran_to_c_order(&data, &[5, 3, 2], &mut out);
        let expect = vec![
            0., 15., 5., 20., 10., 25., 1., 16., 6., 21., 11., 26., 2., 17., 7., 22., 12., 27., 3.,
            18., 8., 23., 13., 28., 4., 19., 9., 24., 14., 29.,
        ];
        dbg!(&out);
        assert!(expect.iter().zip_eq(out.iter()).all(|(a, b)| a == b));
    }
}
