use std::collections::HashMap;
use std::sync::Arc;
use std::{ffi::CString, path::PathBuf};

use anyhow::{bail, Context, Result};
use bridgestan::open_library;
use itertools::Itertools;
use nuts_rs::{CpuLogpFunc, CpuMath, HasDims, LogpError, Model, Storable, Value};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3::{exceptions::PyValueError, pyclass, pymethods, PyResult};
use rand::prelude::Distribution;
use rand::{rng, Rng, RngCore};
use rand_distr::StandardNormal;
use smallvec::{SmallVec, ToSmallVec};

use thiserror::Error;

use crate::common::{ItemType, PyValue, PyVariable};
use crate::wrapper::PyTransformAdapt;

type InnerModel = bridgestan::Model<Arc<bridgestan::StanLibrary>>;

#[pyclass]
#[derive(Clone)]
pub struct StanLibrary(Arc<bridgestan::StanLibrary>);

#[derive(Clone, Debug)]
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
            .map_err(|e| PyValueError::new_err(format!("Could not open stan libray: {e}")))?;
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

    #[getter]
    fn start_idx(&self) -> usize {
        self.0.start_idx
    }

    #[getter]
    fn end_idx(&self) -> usize {
        self.0.end_idx
    }
}

#[pyclass]
#[derive(Clone)]
pub struct StanModel {
    inner: Arc<InnerModel>,
    variables: Vec<PyVariable>,
    transform_adapter: Option<PyTransformAdapt>,
    dim_sizes: HashMap<String, u64>,
    coords: HashMap<String, Value>,
    #[pyo3(get)]
    dims: HashMap<String, Vec<String>>,
    unc_names: Value,
}

/// Return meta information about the constrained parameters of the model
fn params(
    var_string: &str,
    all_dims: &mut HashMap<String, Vec<String>>,
    dim_sizes: &mut HashMap<String, u64>,
) -> anyhow::Result<Vec<PyVariable>> {
    if var_string.is_empty() {
        return Ok(vec![]);
    }
    // Parse each variable string into (name, is_complex, indices)
    let parsed_variables: anyhow::Result<Vec<(String, bool, Vec<usize>)>> = var_string
        .split(',')
        .map(|var| {
            let mut indices = vec![];
            let mut remaining = var;
            let mut complex_suffix = None;

            // Parse from right to left, extracting indices and checking for complex type
            while let Some(idx) = remaining.rfind('.') {
                let suffix = &remaining[(idx + 1)..];

                // Handle complex number suffixes
                if suffix == "real" || suffix == "imag" {
                    complex_suffix = Some(suffix);
                    remaining = &remaining[..idx];
                    continue;
                }

                // Try to parse as index
                if let Ok(index) = suffix.parse::<usize>() {
                    // Convert from 1-based to 0-based indexing
                    let zero_based_idx = index.checked_sub(1).ok_or_else(|| {
                        anyhow::Error::msg("Invalid parameter index (must be > 0)")
                    })?;

                    indices.push(zero_based_idx);
                    remaining = &remaining[..idx];
                } else {
                    // Not a number - this is part of the variable name
                    break;
                }
            }

            // Variable name is what remains
            let name = remaining.trim().to_string();

            // Reverse indices since we parsed right-to-left
            indices.reverse();

            Ok((name, complex_suffix.is_some(), indices))
        })
        .collect();

    // Group variables by name and build Parameter objects
    let mut variables = Vec::new();
    let mut start_idx = 0;

    for (name, group) in &parsed_variables?.iter().chunk_by(|(name, _, _)| name) {
        // Find maximum shape and check if this is a complex variable
        let (shape, is_complex) = determine_variable_shape(group)
            .context(format!("Error while parsing stan variable {name}"))?;

        // Calculate total size of this variable
        let size: usize = shape.iter().product();
        let mut end_idx = start_idx + size;

        // Create Parameter objects (one for real and one for imag if complex)
        if is_complex {
            variables.push(PyVariable::new(
                format!("{name}.real"),
                ItemType(nuts_rs::ItemType::F64),
                Some(shape.iter().map(|&d| d as u64).collect()),
                all_dims,
                dim_sizes,
                Some(start_idx),
            )?);
            start_idx = end_idx;
            end_idx = start_idx + size;
            variables.push(PyVariable::new(
                format!("{name}.imag"),
                ItemType(nuts_rs::ItemType::F64),
                Some(shape.iter().map(|&d| d as u64).collect()),
                all_dims,
                dim_sizes,
                Some(start_idx),
            )?);
        } else {
            variables.push(PyVariable::new(
                name.to_string(),
                ItemType(nuts_rs::ItemType::F64),
                Some(shape.iter().map(|&d| d as u64).collect()),
                all_dims,
                dim_sizes,
                Some(start_idx),
            )?);
        }

        // Move to the next variable
        start_idx = end_idx;
    }

    Ok(variables)
}

// Helper function to determine the shape and complex flag for a group of variables
fn determine_variable_shape<'a, I>(group: I) -> anyhow::Result<(Vec<usize>, bool)>
where
    I: Iterator<Item = &'a (String, bool, Vec<usize>)>,
{
    let group = group.collect_vec();

    let (mut shape, is_complex) = group
        .iter()
        .map(|&(_, is_complex, ref idx)| (idx, is_complex))
        .fold(None, |acc, (elem_index, &elem_is_complex)| {
            let (mut shape, is_complex) = acc.unwrap_or((elem_index.clone(), elem_is_complex));
            assert!(
                is_complex == elem_is_complex,
                "Inconsistent complex flags for same variable"
            );

            // Find maximum index in each dimension
            shape
                .iter_mut()
                .zip_eq(elem_index.iter())
                .for_each(|(old, &new)| {
                    *old = new.max(*old);
                });

            Some((shape, is_complex))
        })
        .expect("List of variable entries cannot be empty");

    shape.iter_mut().for_each(|max_idx| *max_idx += 1);

    // Check if the indices are in Fortran order
    let mut expected_index: Vec<usize> = vec![0; shape.len()];
    let mut expect_imag = false;
    for (_, _, idx) in group.iter() {
        if idx != &expected_index {
            bail!("Stan returned data that was not in the expected order.")
        }
        if is_complex {
            expect_imag = !expect_imag;
        }
        if !expect_imag {
            // increment expected index
            for i in 0..shape.len() {
                if expected_index[i] < shape[i] - 1 {
                    expected_index[i] += 1;
                    break;
                } else {
                    expected_index[i] = 0;
                }
            }
        }
    }

    Ok((shape, is_complex))
}
#[pymethods]
impl StanModel {
    #[new]
    #[pyo3(signature = (lib, dim_sizes, dims, coords, seed=None, data=None, transform_adapter=None))]
    pub fn new(
        py: Python<'_>,
        lib: StanLibrary,
        dim_sizes: Py<PyDict>,
        dims: Py<PyDict>,
        coords: Py<PyDict>,
        seed: Option<u32>,
        data: Option<String>,
        transform_adapter: Option<Py<PyAny>>,
    ) -> anyhow::Result<Self> {
        let mut dim_sizes = dim_sizes
            .bind(py)
            .iter()
            .map(|(key, value)| {
                let key: String = key.extract().context("Dimension key is not a string")?;
                let value: u64 = value
                    .extract()
                    .context("Dimension size value is not an integer")?;
                Ok((key, value))
            })
            .collect::<Result<HashMap<_, _>>>()?;

        let mut dims = dims
            .bind(py)
            .iter()
            .map(|(key, value)| {
                let key: String = key.extract().context("Dimension key is not a string")?;
                let value: Vec<String> = value
                    .extract()
                    .context("Dimension value is not a list of strings")?;
                Ok((key, value))
            })
            .collect::<Result<HashMap<_, _>>>()?;

        let coords = coords
            .bind(py)
            .iter()
            .map(|(key, value)| {
                let key: String = key.extract().context("Coordinate key is not a string")?;
                let value: PyValue = value
                    .extract()
                    .with_context(|| format!("Coordinate {} value has unsupported type", key))?;
                Ok((key, value.into_value()))
            })
            .collect::<Result<HashMap<_, _>>>()?;

        let seed = match seed {
            Some(seed) => seed,
            None => rng().next_u32(),
        };
        let data: Option<CString> = data.map(CString::new).transpose()?;
        let mut model =
            bridgestan::Model::new(lib.0, data.as_ref(), seed).map_err(anyhow::Error::new)?;

        // TODO: bridgestan should not require mut self here
        let names = model.param_unc_names();
        let mut names: Vec<_> = names.split(',').map(|v| v.to_string()).collect();
        if let Some(first) = names.first() {
            if first.is_empty() {
                names = vec![];
            }
        };
        let unc_names = Value::Strings(names);

        let model = Arc::new(model);

        let var_string = model.param_names(true, true);
        let variables = params(var_string, &mut dims, &mut dim_sizes)?;
        let transform_adapter = transform_adapter.map(PyTransformAdapt::new);

        Ok(StanModel {
            inner: model,
            variables,
            transform_adapter,
            dim_sizes,
            coords,
            dims,
            unc_names,
        })
    }

    pub fn variables<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        let results: Result<Vec<_>, _> = self
            .variables
            .iter()
            .map(|var| out.set_item(var.name.clone(), var.clone()))
            .collect();
        results?;
        Ok(out)
    }

    pub fn ndim(&self) -> usize {
        self.inner.param_unc_num()
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
    model: &'model StanModel,
    rng: bridgestan::Rng<&'model bridgestan::StanLibrary>,
    transform_adapter: Option<PyTransformAdapt>,
    expanded_buffer: Vec<f64>,
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

pub struct ExpandedVector(Vec<Option<nuts_rs::Value>>);

impl<'model> Storable<StanDensity<'model>> for ExpandedVector {
    fn names<'a>(parent: &'a StanDensity<'model>) -> Vec<&'a str> {
        parent
            .model
            .variables
            .iter()
            .map(|var| var.name.as_str())
            .collect()
    }

    fn item_type(parent: &StanDensity<'model>, item: &str) -> nuts_rs::ItemType {
        parent
            .model
            .variables
            .iter()
            .find(|var| var.name == item)
            .map(|var| var.item_type.as_inner().clone())
            .expect("Item not found")
    }

    fn dims<'a>(parent: &'a StanDensity<'model>, item: &str) -> Vec<&'a str> {
        parent
            .model
            .variables
            .iter()
            .find(|var| var.name == item)
            .map(|var| var.dims.as_slice().iter().map(|s| s.as_str()).collect())
            .expect("Item not found")
    }

    fn get_all<'a>(&'a mut self, parent: &'a StanDensity<'model>) -> Vec<(&'a str, Option<Value>)> {
        self.0
            .iter_mut()
            .zip(parent.model.variables.iter())
            .map(|(val, var)| (var.name.as_str(), val.take()))
            .collect()
    }
}

impl<'model> HasDims for StanDensity<'model> {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        self.model.dim_sizes.clone()
    }

    fn coords(&self) -> HashMap<String, Value> {
        self.model.coords.clone()
    }
}

impl<'model> CpuLogpFunc for StanDensity<'model> {
    type LogpError = StanLogpError;
    type FlowParameters = Py<PyAny>;
    type ExpandedVector = ExpandedVector;

    fn logp(&mut self, position: &[f64], grad: &mut [f64]) -> Result<f64, Self::LogpError> {
        let logp = self
            .model
            .inner
            .log_density_gradient(position, true, true, grad)?;
        if !logp.is_finite() {
            return Err(StanLogpError::BadLogp(logp));
        }
        Ok(logp)
    }

    fn dim(&self) -> usize {
        self.model.inner.param_unc_num()
    }

    fn vector_coord(&self) -> Option<Value> {
        Some(self.model.unc_names.clone())
    }

    fn expand_vector<R>(
        &mut self,
        _rng: &mut R,
        array: &[f64],
    ) -> Result<Self::ExpandedVector, nuts_rs::CpuMathError>
    where
        R: rand::Rng + ?Sized,
    {
        self.model
            .inner
            .param_constrain(
                array,
                true,
                true,
                &mut self.expanded_buffer,
                Some(&mut self.rng),
            )
            .context("Failed to constrain the parameters of the draw")
            .map_err(|e| nuts_rs::CpuMathError::ExpandError(format!("{}", e)))?;

        let mut vars = Vec::new();

        for var in self.model.variables.iter() {
            let mut out = Vec::with_capacity(var.num_elements);
            let start = var.start_idx.expect("Variable start index not set");
            let end = var.end_idx.expect("Variable end index not set");
            let slice = &self.expanded_buffer[start..end];
            assert!(slice.len() == var.num_elements);

            if var.num_elements == 0 {
                vars.push(Some(Value::F64(out)));
                continue;
            }

            // The slice is in fortran order. This doesn't matter if it low dim
            if var.shape.as_slice().len() < 2 {
                out.extend_from_slice(slice);
                vars.push(Some(Value::F64(out)));
                continue;
            }

            // We need to transpose
            fortran_to_c_order(slice, var.shape.as_slice(), &mut out);
            vars.push(Some(Value::F64(out)));
        }

        Ok(ExpandedVector(vars))
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

fn fortran_to_c_order(data: &[f64], shape: &[u64], out: &mut Vec<f64>) {
    let rank = shape.len();
    let strides = {
        let mut strides: SmallVec<[u64; 8]> = SmallVec::with_capacity(rank);
        let mut current: u64 = 1;
        for &length in shape.iter() {
            strides.push(current);
            current = current
                .checked_mul(length)
                .expect("Overflow in stride computation");
        }
        strides.reverse();
        strides
    };

    let mut shape: SmallVec<[u64; 8]> = shape.to_smallvec();
    shape.reverse();

    let mut idx: SmallVec<[u64; 8]> = shape.iter().map(|_| 0u64).collect();
    let mut position: u64 = 0;
    'iterate: loop {
        out.push(data[position as usize]);

        let mut axis: u64 = 0;
        'nextidx: loop {
            idx[axis as usize] += 1;
            position += strides[axis as usize];

            if idx[axis as usize] < shape[axis as usize] {
                break 'nextidx;
            }

            idx[axis as usize] = 0;
            position -= shape[axis as usize] * strides[axis as usize];
            axis += 1;
            if axis == rank as u64 {
                break 'iterate;
            }
        }
    }
}

/*
pub struct StanTrace<'model> {
    inner: &'model InnerModel,
    model: &'model StanModel,
    trace: Vec<Vec<f64>>,
    expanded_buffer: Box<[f64]>,
    rng: bridgestan::Rng<&'model bridgestan::StanLibrary>,
    count: usize,
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
        self.count += 1;
        Ok(())
    }
}
*/

impl Model for StanModel {
    type Math<'model> = CpuMath<StanDensity<'model>>;

    /*
    fn new_trace<'a, S: Settings, R: rand::Rng + ?Sized>(
        &'a self,
        rng: &mut R,
        _chain: u64,
        settings: &S,
    ) -> anyhow::Result<Self::DrawStorage<'a, S>> {
        let draws = settings.hint_num_tune() + settings.hint_num_draws();
        let trace = self
            .variables
            .iter()
            .map(|var| Vec::with_capacity(var.size * draws))
            .collect();
        let seed = rng.next_u32();
        let rng = self.model.new_rng(seed)?;
        let buffer = vec![0f64; self.model.param_num(true, true)];
        Ok(StanTrace {
            model: self,
            inner: &self.model,
            trace,
            rng,
            expanded_buffer: buffer.into(),
            count: 0,
        })
    }
    */

    fn math<R: Rng + ?Sized>(&self, rng: &mut R) -> anyhow::Result<Self::Math<'_>> {
        let rng = self.inner.new_rng(rng.next_u32())?;
        let num_expanded = self.inner.param_num(true, true);
        Ok(CpuMath::new(StanDensity {
            model: &self,
            rng,
            transform_adapter: self.transform_adapter.clone(),
            expanded_buffer: vec![0f64; num_expanded],
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
    use std::collections::HashMap;

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
        assert!(expect.iter().zip_eq(out.iter()).all(|(a, b)| a == b));
    }

    #[test]
    fn parse_vars() {
        let mut dims = HashMap::new();
        let mut dim_sizes = HashMap::new();

        let vars = "";
        let parsed = super::params(vars, &mut dims, &mut dim_sizes).unwrap();
        assert!(parsed.len() == 0);

        let vars = "x.1.1,x.2.1,x.3.1,x.1.2,x.2.2,x.3.2";
        let parsed = super::params(vars, &mut dims, &mut dim_sizes).unwrap();
        assert!(parsed.len() == 1);
        let parsed = parsed[0].clone();
        assert!(parsed.name == "x");
        assert!(parsed.shape.as_slice() == vec![3, 2]);

        // Incorrect order
        let vars = "x.1.2,x.1.1,x.2.1,x.2.2,x.3.1,x.3.2";
        assert!(super::params(vars, &mut dims, &mut dim_sizes).is_err());

        // Incorrect order
        let vars = "x.1.2.real,x.1.2.imag";
        assert!(super::params(vars, &mut dims, &mut dim_sizes).is_err());

        let vars = "x.1.1.real,x.1.1.imag,x.2.1.real,x.2.1.imag,x.3.1.real,x.3.1.imag";
        let parsed = super::params(vars, &mut dims, &mut dim_sizes).unwrap();
        assert!(parsed.len() == 2);
        let var = parsed[0].clone();
        assert!(var.name == "x.real");
        assert!(var.shape.as_slice() == vec![3, 1]);

        let var = parsed[1].clone();
        assert!(var.name == "x.imag");
        assert!(var.shape.as_slice() == vec![3, 1]);

        // Test single variable
        let vars = "alpha";
        let parsed = super::params(vars, &mut dims, &mut dim_sizes).unwrap();
        assert_eq!(parsed.len(), 1);
        let var = &parsed[0];
        assert_eq!(var.name, "alpha");
        assert_eq!(var.shape.as_slice(), vec![0; 0]);
        assert_eq!(var.num_elements, 1);

        // Test multiple scalar variables
        let vars = "alpha,beta,gamma";
        let parsed = super::params(vars, &mut dims, &mut dim_sizes).unwrap();
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed[0].name, "alpha");
        assert_eq!(parsed[1].name, "beta");
        assert_eq!(parsed[2].name, "gamma");

        // Test 1D array
        let vars = "theta.1,theta.2,theta.3,theta.4";
        let parsed = super::params(vars, &mut dims, &mut dim_sizes).unwrap();
        assert_eq!(parsed.len(), 1);
        let var = &parsed[0];
        assert_eq!(var.name, "theta");
        assert_eq!(var.shape.as_slice(), vec![4]);
        assert_eq!(var.num_elements, 4);

        // Test variable name with colons and dots
        let vars = "x:1:2.4:1.1,x:1:2.4:1.2,x:1:2.4:1.3";
        let parsed = super::params(vars, &mut dims, &mut dim_sizes).unwrap();
        assert_eq!(parsed.len(), 1);
        let var = &parsed[0];
        assert_eq!(var.name, "x:1:2.4:1");
        assert_eq!(var.shape.as_slice(), vec![3]);
        assert_eq!(var.num_elements, 3);

        let vars = "
            a,
            base,
            base_i,
            pair:1,
            pair:2,
            nested:1,
            nested:2:1,
            nested:2:2.real,
            nested:2:2.imag,
            arr_pair.1:1,
            arr_pair.1:2,
            arr_pair.2:1,
            arr_pair.2:2,
            arr_very_nested.1:1:1,
            arr_very_nested.1:1:2:1,
            arr_very_nested.1:1:2:2.real,
            arr_very_nested.1:1:2:2.imag,
            arr_very_nested.1:2,
            arr_very_nested.2:1:1,
            arr_very_nested.2:1:2:1,
            arr_very_nested.2:1:2:2.real,
            arr_very_nested.2:1:2:2.imag,
            arr_very_nested.2:2,
            arr_very_nested.3:1:1,
            arr_very_nested.3:1:2:1,
            arr_very_nested.3:1:2:2.real,
            arr_very_nested.3:1:2:2.imag,
            arr_very_nested.3:2,
            arr_2d_pair.1.1:1,
            arr_2d_pair.1.1:2,
            arr_2d_pair.2.1:1,
            arr_2d_pair.2.1:2,
            arr_2d_pair.3.1:1,
            arr_2d_pair.3.1:2,
            arr_2d_pair.1.2:1,
            arr_2d_pair.1.2:2,
            arr_2d_pair.2.2:1,
            arr_2d_pair.2.2:2,
            arr_2d_pair.3.2:1,
            arr_2d_pair.3.2:2,
            basep1,
            basep2,
            basep3,
            basep4,
            basep5,
            ultimate.1.1:1.1:1,
            ultimate.1.1:1.1:2.1,
            ultimate.1.1:1.1:2.2,
            ultimate.1.1:1.2:1,
            ultimate.1.1:1.2:2.1,
            ultimate.1.1:1.2:2.2,
            ultimate.1.1:2.1.1,
            ultimate.1.1:2.2.1,
            ultimate.1.1:2.3.1,
            ultimate.1.1:2.4.1,
            ultimate.1.1:2.1.2,
            ultimate.1.1:2.2.2,
            ultimate.1.1:2.3.2,
            ultimate.1.1:2.4.2,
            ultimate.1.1:2.1.3,
            ultimate.1.1:2.2.3,
            ultimate.1.1:2.3.3,
            ultimate.1.1:2.4.3,
            ultimate.1.1:2.1.4,
            ultimate.1.1:2.2.4,
            ultimate.1.1:2.3.4,
            ultimate.1.1:2.4.4,
            ultimate.1.1:2.1.5,
            ultimate.1.1:2.2.5,
            ultimate.1.1:2.3.5,
            ultimate.1.1:2.4.5,
            ultimate.2.1:1.1:1,
            ultimate.2.1:1.1:2.1,
            ultimate.2.1:1.1:2.2,
            ultimate.2.1:1.2:1,
            ultimate.2.1:1.2:2.1,
            ultimate.2.1:1.2:2.2,
            ultimate.2.1:2.1.1,
            ultimate.2.1:2.2.1,
            ultimate.2.1:2.3.1,
            ultimate.2.1:2.4.1,
            ultimate.2.1:2.1.2,
            ultimate.2.1:2.2.2,
            ultimate.2.1:2.3.2,
            ultimate.2.1:2.4.2,
            ultimate.2.1:2.1.3,
            ultimate.2.1:2.2.3,
            ultimate.2.1:2.3.3,
            ultimate.2.1:2.4.3,
            ultimate.2.1:2.1.4,
            ultimate.2.1:2.2.4,
            ultimate.2.1:2.3.4,
            ultimate.2.1:2.4.4,
            ultimate.2.1:2.1.5,
            ultimate.2.1:2.2.5,
            ultimate.2.1:2.3.5,
            ultimate.2.1:2.4.5,
            ultimate.1.2:1.1:1,
            ultimate.1.2:1.1:2.1,
            ultimate.1.2:1.1:2.2,
            ultimate.1.2:1.2:1,
            ultimate.1.2:1.2:2.1,
            ultimate.1.2:1.2:2.2,
            ultimate.1.2:2.1.1,
            ultimate.1.2:2.2.1,
            ultimate.1.2:2.3.1,
            ultimate.1.2:2.4.1,
            ultimate.1.2:2.1.2,
            ultimate.1.2:2.2.2,
            ultimate.1.2:2.3.2,
            ultimate.1.2:2.4.2,
            ultimate.1.2:2.1.3,
            ultimate.1.2:2.2.3,
            ultimate.1.2:2.3.3,
            ultimate.1.2:2.4.3,
            ultimate.1.2:2.1.4,
            ultimate.1.2:2.2.4,
            ultimate.1.2:2.3.4,
            ultimate.1.2:2.4.4,
            ultimate.1.2:2.1.5,
            ultimate.1.2:2.2.5,
            ultimate.1.2:2.3.5,
            ultimate.1.2:2.4.5,
            ultimate.2.2:1.1:1,
            ultimate.2.2:1.1:2.1,
            ultimate.2.2:1.1:2.2,
            ultimate.2.2:1.2:1,
            ultimate.2.2:1.2:2.1,
            ultimate.2.2:1.2:2.2,
            ultimate.2.2:2.1.1,
            ultimate.2.2:2.2.1,
            ultimate.2.2:2.3.1,
            ultimate.2.2:2.4.1,
            ultimate.2.2:2.1.2,
            ultimate.2.2:2.2.2,
            ultimate.2.2:2.3.2,
            ultimate.2.2:2.4.2,
            ultimate.2.2:2.1.3,
            ultimate.2.2:2.2.3,
            ultimate.2.2:2.3.3,
            ultimate.2.2:2.4.3,
            ultimate.2.2:2.1.4,
            ultimate.2.2:2.2.4,
            ultimate.2.2:2.3.4,
            ultimate.2.2:2.4.4,
            ultimate.2.2:2.1.5,
            ultimate.2.2:2.2.5,
            ultimate.2.2:2.3.5,
            ultimate.2.2:2.4.5,
            ultimate.1.3:1.1:1,
            ultimate.1.3:1.1:2.1,
            ultimate.1.3:1.1:2.2,
            ultimate.1.3:1.2:1,
            ultimate.1.3:1.2:2.1,
            ultimate.1.3:1.2:2.2,
            ultimate.1.3:2.1.1,
            ultimate.1.3:2.2.1,
            ultimate.1.3:2.3.1,
            ultimate.1.3:2.4.1,
            ultimate.1.3:2.1.2,
            ultimate.1.3:2.2.2,
            ultimate.1.3:2.3.2,
            ultimate.1.3:2.4.2,
            ultimate.1.3:2.1.3,
            ultimate.1.3:2.2.3,
            ultimate.1.3:2.3.3,
            ultimate.1.3:2.4.3,
            ultimate.1.3:2.1.4,
            ultimate.1.3:2.2.4,
            ultimate.1.3:2.3.4,
            ultimate.1.3:2.4.4,
            ultimate.1.3:2.1.5,
            ultimate.1.3:2.2.5,
            ultimate.1.3:2.3.5,
            ultimate.1.3:2.4.5,
            ultimate.2.3:1.1:1,
            ultimate.2.3:1.1:2.1,
            ultimate.2.3:1.1:2.2,
            ultimate.2.3:1.2:1,
            ultimate.2.3:1.2:2.1,
            ultimate.2.3:1.2:2.2,
            ultimate.2.3:2.1.1,
            ultimate.2.3:2.2.1,
            ultimate.2.3:2.3.1,
            ultimate.2.3:2.4.1,
            ultimate.2.3:2.1.2,
            ultimate.2.3:2.2.2,
            ultimate.2.3:2.3.2,
            ultimate.2.3:2.4.2,
            ultimate.2.3:2.1.3,
            ultimate.2.3:2.2.3,
            ultimate.2.3:2.3.3,
            ultimate.2.3:2.4.3,
            ultimate.2.3:2.1.4,
            ultimate.2.3:2.2.4,
            ultimate.2.3:2.3.4,
            ultimate.2.3:2.4.4,
            ultimate.2.3:2.1.5,
            ultimate.2.3:2.2.5,
            ultimate.2.3:2.3.5,
            ultimate.2.3:2.4.5
        ";
        let parsed = super::params(vars, &mut dims, &mut dim_sizes).unwrap();
        assert_eq!(parsed[0].name, "a");
        assert_eq!(parsed[0].shape.as_slice(), vec![0; 0]);

        assert_eq!(parsed[1].name, "base");
        assert_eq!(parsed[1].shape.as_slice(), vec![0; 0]);

        assert_eq!(parsed[2].name, "base_i");
        assert_eq!(parsed[2].shape.as_slice(), vec![0; 0]);

        assert_eq!(parsed[3].name, "pair:1");
        assert_eq!(parsed[3].shape.as_slice(), vec![0; 0]);

        assert_eq!(parsed[4].name, "pair:2");
        assert_eq!(parsed[4].shape.as_slice(), vec![0; 0]);

        assert_eq!(parsed[5].name, "nested:1");
        assert_eq!(parsed[5].shape.as_slice(), vec![0; 0]);

        assert_eq!(parsed[6].name, "nested:2:1");
        assert_eq!(parsed[6].shape.as_slice(), vec![0; 0]);

        assert_eq!(parsed[7].name, "nested:2:2.real");
        assert_eq!(parsed[7].shape.as_slice(), vec![0; 0]);

        assert_eq!(parsed[8].name, "nested:2:2.imag");
        assert_eq!(parsed[8].shape.as_slice(), vec![0; 0]);

        assert_eq!(parsed[9].name, "arr_pair.1:1");
        assert_eq!(parsed[9].shape.as_slice(), vec![0; 0]);

        assert_eq!(parsed[10].name, "arr_pair.1:2");
        assert_eq!(parsed[10].shape.as_slice(), vec![0; 0]);

        assert_eq!(parsed[11].name, "arr_pair.2:1");
        assert_eq!(parsed[11].shape.as_slice(), vec![0; 0]);

        assert_eq!(parsed[12].name, "arr_pair.2:2");
        assert_eq!(parsed[12].shape.as_slice(), vec![0; 0]);

        assert_eq!(parsed[13].name, "arr_very_nested.1:1:1");
        assert_eq!(parsed[13].shape.as_slice(), vec![0; 0]);

        assert_eq!(parsed[14].name, "arr_very_nested.1:1:2:1");
        assert_eq!(parsed[14].shape.as_slice(), vec![0; 0]);

        assert_eq!(parsed[15].name, "arr_very_nested.1:1:2:2.real");
        assert_eq!(parsed[15].shape.as_slice(), vec![0; 0]);

        assert_eq!(parsed[16].name, "arr_very_nested.1:1:2:2.imag");
        assert_eq!(parsed[16].shape.as_slice(), vec![0; 0]);

        assert_eq!(parsed[17].name, "arr_very_nested.1:2");
        assert_eq!(parsed[17].shape.as_slice(), vec![0; 0]);

        assert_eq!(parsed[18].name, "arr_very_nested.2:1:1");
        assert_eq!(parsed[18].shape.as_slice(), vec![0; 0]);

        assert_eq!(parsed[19].name, "arr_very_nested.2:1:2:1");
        assert_eq!(parsed[19].shape.as_slice(), vec![0; 0]);

        assert_eq!(parsed[20].name, "arr_very_nested.2:1:2:2.real");
        assert_eq!(parsed[20].shape.as_slice(), vec![0; 0]);

        assert_eq!(parsed[21].name, "arr_very_nested.2:1:2:2.imag");
        assert_eq!(parsed[21].shape.as_slice(), vec![0; 0]);

        assert_eq!(parsed[22].name, "arr_very_nested.2:2");
        assert_eq!(parsed[22].shape.as_slice(), vec![0; 0]);

        assert_eq!(parsed[23].name, "arr_very_nested.3:1:1");
        assert_eq!(parsed[23].shape.as_slice(), vec![0; 0]);

        assert_eq!(parsed[24].name, "arr_very_nested.3:1:2:1");
        assert_eq!(parsed[24].shape.as_slice(), vec![0; 0]);

        assert_eq!(parsed[25].name, "arr_very_nested.3:1:2:2.real");
        assert_eq!(parsed[25].shape.as_slice(), vec![0; 0]);

        assert_eq!(parsed[26].name, "arr_very_nested.3:1:2:2.imag");
        assert_eq!(parsed[26].shape.as_slice(), vec![0; 0]);

        assert_eq!(parsed[27].name, "arr_very_nested.3:2");
        assert_eq!(parsed[27].shape.as_slice(), vec![0; 0]);
    }
}
