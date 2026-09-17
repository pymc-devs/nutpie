use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::{ffi::CString, path::PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use bridgestan::open_library;
use itertools::Itertools;
use numpy::{PyArray1, PyReadonlyArray1};
use nuts_rs::{
    CpuLogpFunc, CpuMath, HasDims, InitPositionError, LogpError, Model, Storable, Value,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::{PyDict, PyNone};
use pyo3::{exceptions::PyValueError, pyclass, pymethods, PyResult};
use pyo3::{prelude::*, BoundObject};
use rand::prelude::Distribution;
use rand::{rng, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;
use smallvec::{SmallVec, ToSmallVec};

use thiserror::Error;

use crate::common::{copy_init_point, ItemType, PyValue, PyVariable};
use crate::hessian_sparsity::{hessian_sparsity, HessianVectorProduct, SparsityOptions};
use crate::wrapper::{soft_clip, NativeFlow, PyTransformAdapt};

type InnerModel = bridgestan::Model<Arc<bridgestan::StanLibrary>>;

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct StanLibrary(Arc<bridgestan::StanLibrary>);

#[pymethods]
impl StanLibrary {
    #[new]
    fn new(path: PathBuf) -> PyResult<Self> {
        let lib = open_library(path)
            .map_err(|e| PyValueError::new_err(format!("Could not open stan libray: {e}")))?;
        Ok(Self(Arc::new(lib)))
    }
}

#[pyclass(from_py_object)]
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
    init_point_func: Option<Arc<Py<PyAny>>>,
    parameters: Vec<Parameter>,
}

/// A stan variable in the flat output of `param_constrain`.
///
/// Stan stores values in fortran order. For complex variables the real
/// and imaginary parts are interleaved, so `start_idx..end_idx` has
/// fortran shape `[2, *shape]`. In the trace, complex variables are split
/// into two variables `name.real` and `name.imag`.
#[derive(Clone, Debug)]
struct Parameter {
    name: String,
    shape: Vec<usize>,
    /// Number of elements, not counting real and imaginary parts separately
    size: usize,
    is_complex: bool,
    start_idx: usize,
    end_idx: usize,
}

impl Parameter {
    /// Append the values of the trace variables of this parameter in C
    /// order to `out`, one vector per variable.
    fn unpack(&self, flat: &[f64], out: &mut Vec<Vec<f64>>) {
        let slice = &flat[self.start_idx..self.end_idx];
        let mut values = Vec::with_capacity(slice.len());
        let mut stan_shape: SmallVec<[u64; 8]> = self.shape.iter().map(|&d| d as u64).collect();
        if self.is_complex {
            stan_shape.insert(0, 2);
        }
        // For rank < 2 fortran and C order are the same
        if slice.is_empty() || stan_shape.len() < 2 {
            values.extend_from_slice(slice);
        } else {
            fortran_to_c_order(slice, &stan_shape, &mut values);
        }
        if self.is_complex {
            // In C order with shape `[2, *shape]`, all real parts come first
            let imag = values.split_off(self.size);
            out.push(values);
            out.push(imag);
        } else {
            out.push(values);
        }
    }
}

/// Create the variables of the trace for the stan parameters.
fn trace_variables(
    parameters: &[Parameter],
    all_dims: &mut HashMap<String, Vec<String>>,
    dim_sizes: &mut HashMap<String, u64>,
) -> anyhow::Result<Vec<PyVariable>> {
    let mut variables = Vec::new();
    for param in parameters {
        let shape: Vec<u64> = param.shape.iter().map(|&d| d as u64).collect();
        let names = if param.is_complex {
            vec![
                format!("{}.real", param.name),
                format!("{}.imag", param.name),
            ]
        } else {
            vec![param.name.clone()]
        };
        // The indices of the variables are only nominal, the values are
        // extracted with `Parameter::unpack`.
        for (i, name) in names.into_iter().enumerate() {
            variables.push(PyVariable::new(
                name,
                ItemType(nuts_rs::ItemType::F64),
                Some(shape.clone()),
                all_dims,
                dim_sizes,
                Some(param.start_idx + i * param.size),
            )?);
        }
    }
    Ok(variables)
}

/// Parse the comma separated parameter names returned by stan.
fn params(var_string: &str) -> anyhow::Result<Vec<Parameter>> {
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
    let mut parameters = Vec::new();
    let mut start_idx = 0;

    for (name, group) in &parsed_variables?.iter().chunk_by(|(name, _, _)| name) {
        // Find maximum shape and check if this is a complex variable
        let (shape, is_complex) = determine_variable_shape(group)
            .context(format!("Error while parsing stan variable {name}"))?;

        let size: usize = shape.iter().product();
        let end_idx = start_idx + if is_complex { 2 * size } else { size };

        parameters.push(Parameter {
            name: name.to_string(),
            shape,
            size,
            is_complex,
            start_idx,
            end_idx,
        });

        start_idx = end_idx;
    }

    Ok(parameters)
}

// Helper function to determine the shape and complex flag for a group of variables
fn determine_variable_shape<'a, I>(group: I) -> anyhow::Result<(Vec<usize>, bool)>
where
    I: Iterator<Item = &'a (String, bool, Vec<usize>)>,
{
    let group = group.collect_vec();

    let (mut shape, is_complex) = group
        .iter()
        .map(|&(_, is_complex, idx)| (idx, is_complex))
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
        let model =
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
        let parameters = params(var_string)?;
        let variables = trace_variables(&parameters, &mut dims, &mut dim_sizes)?;
        let transform_adapter = transform_adapter.map(PyTransformAdapt::new);

        Ok(StanModel {
            inner: model,
            variables,
            transform_adapter,
            dim_sizes,
            coords,
            dims,
            unc_names,
            init_point_func: None,
            parameters,
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

    /// Return a copy of the model that uses `transform_adapter` for the
    /// normalizing flow adaptation.
    pub fn with_transform_adapter(&self, transform_adapter: Py<PyAny>) -> Self {
        Self {
            transform_adapter: Some(PyTransformAdapt::new(transform_adapter)),
            ..self.clone()
        }
    }

    /// Names of the unconstrained parameters.
    pub fn unconstrained_names(&self) -> Vec<String> {
        match &self.unc_names {
            Value::Strings(names) => names.clone(),
            _ => unreachable!("Unconstrained names are strings"),
        }
    }

    #[pyo3(signature = (include_tp=false, include_gq=false))]
    pub fn param_num(&self, include_tp: bool, include_gq: bool) -> usize {
        self.inner.param_num(include_tp, include_gq)
    }

    /// Whether the model was compiled with autodiff Hessians
    /// (`BRIDGESTAN_AD_HESSIAN=true`). Otherwise bridgestan uses finite
    /// differences for Hessians.
    #[getter]
    pub fn ad_hessian(&self) -> bool {
        self.inner
            .info()
            .to_string_lossy()
            .contains("BRIDGESTAN_AD_HESSIAN=true")
    }

    /// Return the log density and the product of its Hessian with `v`.
    #[pyo3(signature = (theta_unc, v, propto=true, jacobian=true))]
    pub fn log_density_hessian_vector_product<'py>(
        &self,
        py: Python<'py>,
        theta_unc: PyReadonlyArray1<'py, f64>,
        v: PyReadonlyArray1<'py, f64>,
        propto: bool,
        jacobian: bool,
    ) -> anyhow::Result<(f64, Bound<'py, PyArray1<f64>>)> {
        self.check_ad_hessian()?;
        let theta_unc = self.check_unc_len(theta_unc.as_slice()?)?;
        let v = self.check_unc_len(v.as_slice()?)?;
        let mut hvp = vec![0f64; self.inner.param_unc_num()];
        let logp = self
            .inner
            .log_density_hessian_vector_product(theta_unc, v, propto, jacobian, &mut hvp)?;
        Ok((logp, PyArray1::from_vec(py, hvp)))
    }

    /// Detect the sparsity pattern of the Hessian of the log density on the
    /// unconstrained space.
    ///
    /// The pattern is the union of the patterns at `num_points` initial
    /// points of the model, computed from autodiff Hessian-vector products.
    /// See `crate::hessian_sparsity` for the algorithm. Returns the symmetric
    /// pattern with a true diagonal in CSR format (`indptr`, `indices`), the
    /// number of Hessian-vector products and the number of colours of the
    /// verification stage.
    #[pyo3(signature = (num_points=4, seed=None, bloom_size=None, num_hashes=3, max_tries=100))]
    pub fn hessian_sparsity<'py>(
        &self,
        py: Python<'py>,
        num_points: usize,
        seed: Option<u64>,
        bloom_size: Option<usize>,
        num_hashes: usize,
        max_tries: usize,
    ) -> anyhow::Result<(
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
        usize,
        usize,
    )> {
        self.check_ad_hessian()?;
        let seed = seed.unwrap_or_else(|| rng().next_u64());
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let points = self.hessian_points(py, &mut rng, num_points, max_tries)?;
        let options = SparsityOptions {
            bloom_size,
            num_hashes,
            seed: rng.next_u64(),
        };
        let inner = &self.inner;
        let pattern = py.detach(|| {
            let mut hessian = StanHessian {
                inner,
                last_signal_check: Instant::now(),
            };
            hessian_sparsity(&mut hessian, &points, &options)
        })?;

        let (indptr, indices) = pattern.to_csr();
        Ok((
            PyArray1::from_vec(py, indptr),
            PyArray1::from_vec(py, indices),
            pattern.num_hvps,
            pattern.num_colors,
        ))
    }

    /// Return a copy of the model that generates initial points with
    /// `init_point_func(seed, chain_id)`. It must return either a flat
    /// unconstrained point, or a Stan JSON string with values for some or
    /// all parameters.
    pub fn with_init_point_func(&self, init_point_func: Py<PyAny>) -> Self {
        Self {
            init_point_func: Some(Arc::new(init_point_func)),
            ..self.clone()
        }
    }

    /// Map a point on the unconstrained space to the flat (column-major)
    /// constrained parameter vector.
    #[pyo3(signature = (theta_unc, include_tp=false, include_gq=false, seed=None))]
    pub fn param_constrain<'py>(
        &self,
        py: Python<'py>,
        theta_unc: PyReadonlyArray1<'py, f64>,
        include_tp: bool,
        include_gq: bool,
        seed: Option<u32>,
    ) -> anyhow::Result<Bound<'py, PyArray1<f64>>> {
        let theta_unc = theta_unc.as_slice()?;
        if theta_unc.len() != self.inner.param_unc_num() {
            bail!(
                "Unconstrained point has length {} (expected {})",
                theta_unc.len(),
                self.inner.param_unc_num()
            );
        }
        let mut out = vec![0f64; self.inner.param_num(include_tp, include_gq)];
        let mut rng = if include_gq {
            let seed = seed.unwrap_or_else(|| rng().next_u32());
            Some(bridgestan::Rng::new(self.inner.clone_library_ref(), seed)?)
        } else {
            None
        };
        self.inner
            .param_constrain(theta_unc, include_tp, include_gq, &mut out, rng.as_mut())?;
        Ok(PyArray1::from_vec(py, out))
    }

    /// Map a flat (column-major) constrained parameter vector to the
    /// unconstrained space.
    pub fn param_unconstrain<'py>(
        &self,
        py: Python<'py>,
        theta: PyReadonlyArray1<'py, f64>,
    ) -> anyhow::Result<Bound<'py, PyArray1<f64>>> {
        let theta = theta.as_slice()?;
        if theta.len() != self.inner.param_num(false, false) {
            bail!(
                "Constrained point has length {} (expected {})",
                theta.len(),
                self.inner.param_num(false, false)
            );
        }
        let mut out = vec![0f64; self.inner.param_unc_num()];
        self.inner.param_unconstrain(theta, &mut out)?;
        Ok(PyArray1::from_vec(py, out))
    }

    /// Map constrained parameter values in Stan JSON format to the
    /// unconstrained space.
    pub fn param_unconstrain_json<'py>(
        &self,
        py: Python<'py>,
        json: String,
    ) -> anyhow::Result<Bound<'py, PyArray1<f64>>> {
        let json = CString::new(json)?;
        let mut out = vec![0f64; self.inner.param_unc_num()];
        self.inner.param_unconstrain_json(&json, &mut out)?;
        Ok(PyArray1::from_vec(py, out))
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

pub struct StanDensity {
    model: Arc<StanModel>,
    rng: bridgestan::Rng<Arc<bridgestan::StanLibrary>>,
    transform_adapter: Option<PyTransformAdapt>,
    expanded_buffer: Vec<f64>,
    native_flow: NativeFlow,
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

impl Storable<StanDensity> for ExpandedVector {
    fn names<'a>(parent: &'a StanDensity) -> Vec<&'a str> {
        parent
            .model
            .variables
            .iter()
            .map(|var| var.name.as_str())
            .collect()
    }

    fn item_type(parent: &StanDensity, item: &str) -> nuts_rs::ItemType {
        parent
            .model
            .variables
            .iter()
            .find(|var| var.name == item)
            .map(|var| var.item_type.as_inner().clone())
            .expect("Item not found")
    }

    fn dims<'a>(parent: &'a StanDensity, item: &str) -> Vec<&'a str> {
        parent
            .model
            .variables
            .iter()
            .find(|var| var.name == item)
            .map(|var| var.dims.as_slice().iter().map(|s| s.as_str()).collect())
            .expect("Item not found")
    }

    fn get_all<'a>(&'a mut self, parent: &'a StanDensity) -> Vec<(&'a str, Option<Value>)> {
        self.0
            .iter_mut()
            .zip(parent.model.variables.iter())
            .map(|(val, var)| (var.name.as_str(), val.take()))
            .collect()
    }
}

impl HasDims for StanDensity {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        self.model.dim_sizes.clone()
    }

    fn coords(&self) -> HashMap<String, Value> {
        self.model.coords.clone()
    }
}

impl CpuLogpFunc for StanDensity {
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

        let mut values = Vec::with_capacity(self.model.variables.len());
        for param in self.model.parameters.iter() {
            param.unpack(&self.expanded_buffer, &mut values);
        }
        assert!(values.len() == self.model.variables.len());
        let vars = values
            .into_iter()
            .map(|values| Some(Value::F64(values)))
            .collect();

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
        clip: Option<f64>,
    ) -> std::result::Result<(f64, f64), Self::LogpError> {
        // Native path: no Python at all. `native_flow` is moved out for the
        // call so the logp closure can borrow `self`.
        let adapter = self
            .transform_adapter
            .clone()
            .ok_or_else(|| PyRuntimeError::new_err("No transformation adapter specified"))?;
        let mut native = std::mem::take(&mut self.native_flow);
        let result = native.init_from_transformed_position(
            &adapter,
            params,
            untransformed_position,
            untransformed_gradient,
            transformed_position,
            transformed_gradient,
            clip,
            |y, grad| self.logp(y, grad),
        );
        self.native_flow = native;
        if let Some(out) = result? {
            return Ok(out);
        }

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
        soft_clip(untransformed_gradient, clip);

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
        clip: Option<f64>,
    ) -> std::result::Result<(f64, f64), Self::LogpError> {
        let logp = self
            .logp(untransformed_position, untransformed_gradient)
            .context("Failed to call stan logp function")?;
        soft_clip(untransformed_gradient, clip);

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
        self.native_flow.invalidate();
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

    fn init_transformation<R: rand::Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        untransformed_position: &[f64],
        untransformed_gradient: &[f64],
        chain: u64,
    ) -> std::result::Result<Py<PyAny>, Self::LogpError> {
        self.native_flow.invalidate();
        let trafo = self
            .transform_adapter
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("No transformation adapter specified"))?
            .new_transformation(rng, untransformed_position, untransformed_gradient, chain)
            .context("Could not create transformation adapter")?;
        Ok(trafo)
    }

    fn new_transformation<R: rand::Rng + ?Sized>(
        &mut self,
        _rng: &mut R,
        _dim: usize,
        _chain: u64,
    ) -> std::result::Result<Self::FlowParameters, Self::LogpError> {
        Python::attach(|py| {
            let params = PyNone::get(py);
            Ok(params.unbind().into())
        })
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

impl StanModel {
    fn check_ad_hessian(&self) -> Result<()> {
        if !self.ad_hessian() {
            bail!(
                "Automatic hessian sparsity detection requires hessian information.
                 Compile with `nutpie.compile_stan_model(..., ad_hessian=True)`."
            );
        }
        Ok(())
    }

    /// Generate `num_points` initial points with finite log density.
    fn hessian_points<R: Rng + ?Sized>(
        &self,
        py: Python<'_>,
        rng: &mut R,
        num_points: usize,
        max_tries: usize,
    ) -> Result<Vec<Vec<f64>>> {
        let n = self.inner.param_unc_num();
        let mut points = Vec::with_capacity(num_points);
        let mut last_error = None;
        for _ in 0..num_points.saturating_mul(max_tries) {
            if points.len() == num_points {
                break;
            }
            py.check_signals()?;
            let mut position = vec![0f64; n];
            match self.init_position(rng, points.len() as u64, &mut position) {
                Ok(()) => {}
                Err(InitPositionError::Retry(err)) => {
                    last_error = Some(err);
                    continue;
                }
                Err(InitPositionError::Fatal(err)) => {
                    return Err(err.context("Could not generate a point to evaluate the Hessian"));
                }
            }
            let inner = &self.inner;
            match py.detach(|| inner.log_density(&position, true, true)) {
                Ok(logp) if logp.is_finite() => points.push(position),
                Ok(logp) => last_error = Some(anyhow!("Log density is {logp}")),
                Err(err) => last_error = Some(err.into()),
            }
        }
        if points.len() < num_points {
            let err = last_error.unwrap_or_else(|| anyhow!("No points were tried"));
            return Err(err.context(format!(
                "Found only {} of {num_points} points with finite log density \
                 to evaluate the Hessian",
                points.len()
            )));
        }
        Ok(points)
    }

    fn check_unc_len<'a>(&self, values: &'a [f64]) -> Result<&'a [f64]> {
        if values.len() != self.inner.param_unc_num() {
            bail!(
                "Array has length {} (expected {})",
                values.len(),
                self.inner.param_unc_num()
            );
        }
        Ok(values)
    }
}

/// How often the Hessian sparsity detection checks for a KeyboardInterrupt.
const SIGNAL_CHECK_INTERVAL: Duration = Duration::from_millis(100);

struct StanHessian<'a> {
    inner: &'a InnerModel,
    last_signal_check: Instant,
}

impl StanHessian<'_> {
    /// Let Python handle signals, since the detection runs without the GIL.
    /// A KeyboardInterrupt is returned as the plain `PyErr`, so that pyo3
    /// raises it unchanged.
    fn check_signals(&mut self) -> Result<()> {
        if self.last_signal_check.elapsed() < SIGNAL_CHECK_INTERVAL {
            return Ok(());
        }
        self.last_signal_check = Instant::now();
        Python::attach(|py| py.check_signals())?;
        Ok(())
    }
}

impl HessianVectorProduct for StanHessian<'_> {
    fn dim(&self) -> usize {
        self.inner.param_unc_num()
    }

    fn hvp(&mut self, point: &[f64], vector: &[f64], out: &mut [f64]) -> Result<()> {
        self.check_signals()?;
        self.inner
            .log_density_hessian_vector_product(point, vector, true, true, out)
            .context("Failed to compute Hessian-vector product")?;
        Ok(())
    }
}

impl Model for StanModel {
    type Math = CpuMath<StanDensity>;

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

    fn math<R: Rng + ?Sized>(self: Arc<StanModel>, rng: &mut R) -> anyhow::Result<Self::Math> {
        let rng = bridgestan::Rng::new(self.inner.clone_library_ref(), rng.next_u32())?;
        let num_expanded = self.inner.param_num(true, true);
        Ok(CpuMath::new(StanDensity {
            model: self.clone(),
            rng,
            transform_adapter: self.transform_adapter.clone(),
            expanded_buffer: vec![0f64; num_expanded],
            native_flow: NativeFlow::default(),
        }))
    }

    fn init_position<R: rand::Rng + ?Sized>(
        &self,
        rng: &mut R,
        chain_id: u64,
        position: &mut [f64],
    ) -> Result<(), InitPositionError> {
        if let Some(init_func) = self.init_point_func.as_ref() {
            let seed = rng.next_u64();
            let stan_seed = rng.next_u32();
            // The init function returns either an unconstrained array or
            // constrained values as a Stan JSON string.
            return Python::attach(|py| {
                let init_point = init_func
                    .call1(py, (seed, chain_id))
                    .context("Failed to initialize point")?;
                let init_point = init_point.bind(py);
                let Ok(json) = init_point.extract::<String>() else {
                    return Ok(copy_init_point(init_point, position)?);
                };
                let json = CString::new(json).context("Invalid initial point json")?;
                // Parameters missing from the json are drawn uniformly from
                // [-2, 2] on the unconstrained space. Stan checks that the
                // log density is finite, and the sampler retries if it isn't.
                let mut stan_rng = bridgestan::Rng::new(self.inner.clone_library_ref(), stan_seed)
                    .context("Could not create stan rng")?;
                self.inner
                    .param_initialize(&mut stan_rng, &json, 2.0, 1, true, position)
                    .map_err(|err| InitPositionError::Retry(err.into()))
            });
        }
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

    fn parse(
        vars: &str,
        dims: &mut HashMap<String, Vec<String>>,
        dim_sizes: &mut HashMap<String, u64>,
    ) -> anyhow::Result<Vec<crate::common::PyVariable>> {
        super::trace_variables(&super::params(vars)?, dims, dim_sizes)
    }

    #[test]
    fn unpack_complex() {
        let mut dims = HashMap::new();
        let mut dim_sizes = HashMap::new();

        // A real scalar, a complex vector of length 2 and a complex
        // matrix of shape (2, 2).
        let vars = "a,\
            z.1.real,z.1.imag,z.2.real,z.2.imag,\
            m.1.1.real,m.1.1.imag,m.2.1.real,m.2.1.imag,\
            m.1.2.real,m.1.2.imag,m.2.2.real,m.2.2.imag";
        let parameters = super::params(vars).unwrap();
        let variables = super::trace_variables(&parameters, &mut dims, &mut dim_sizes).unwrap();
        let names: Vec<_> = variables.iter().map(|var| var.name.as_str()).collect();
        assert_eq!(names, ["a", "z.real", "z.imag", "m.real", "m.imag"]);

        // Values as stan returns them: interleaved and in fortran order
        let flat = [
            0., //
            1., 10., 2., 20., //
            11., -11., 21., -21., 12., -12., 22., -22.,
        ];
        let mut out = vec![];
        for param in parameters.iter() {
            param.unpack(&flat, &mut out);
        }
        assert_eq!(
            out,
            vec![
                vec![0.],
                vec![1., 2.],
                vec![10., 20.],
                vec![11., 12., 21., 22.],
                vec![-11., -12., -21., -22.],
            ]
        );
    }

    #[test]
    fn parse_vars() {
        let mut dims = HashMap::new();
        let mut dim_sizes = HashMap::new();

        let vars = "";
        let parsed = parse(vars, &mut dims, &mut dim_sizes).unwrap();
        assert!(parsed.len() == 0);

        let vars = "x.1.1,x.2.1,x.3.1,x.1.2,x.2.2,x.3.2";
        let parsed = parse(vars, &mut dims, &mut dim_sizes).unwrap();
        assert!(parsed.len() == 1);
        let parsed = parsed[0].clone();
        assert!(parsed.name == "x");
        assert!(parsed.shape.as_slice() == vec![3, 2]);

        // Incorrect order
        let vars = "x.1.2,x.1.1,x.2.1,x.2.2,x.3.1,x.3.2";
        assert!(parse(vars, &mut dims, &mut dim_sizes).is_err());

        // Incorrect order
        let vars = "x.1.2.real,x.1.2.imag";
        assert!(parse(vars, &mut dims, &mut dim_sizes).is_err());

        let vars = "x.1.1.real,x.1.1.imag,x.2.1.real,x.2.1.imag,x.3.1.real,x.3.1.imag";
        let parameters = super::params(vars).unwrap();
        assert_eq!(parameters.len(), 1);
        let param = &parameters[0];
        assert_eq!(param.name, "x");
        assert!(param.is_complex);
        assert_eq!(param.shape, vec![3, 1]);
        assert_eq!(param.size, 3);
        assert_eq!((param.start_idx, param.end_idx), (0, 6));

        let parsed = parse(vars, &mut dims, &mut dim_sizes).unwrap();
        assert!(parsed.len() == 2);
        let var = parsed[0].clone();
        assert!(var.name == "x.real");
        assert!(var.shape.as_slice() == vec![3, 1]);

        let var = parsed[1].clone();
        assert!(var.name == "x.imag");
        assert!(var.shape.as_slice() == vec![3, 1]);

        // Test single variable
        let vars = "alpha";
        let parsed = parse(vars, &mut dims, &mut dim_sizes).unwrap();
        assert_eq!(parsed.len(), 1);
        let var = &parsed[0];
        assert_eq!(var.name, "alpha");
        assert_eq!(var.shape.as_slice(), vec![0; 0]);
        assert_eq!(var.num_elements, 1);

        // Test multiple scalar variables
        let vars = "alpha,beta,gamma";
        let parsed = parse(vars, &mut dims, &mut dim_sizes).unwrap();
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed[0].name, "alpha");
        assert_eq!(parsed[1].name, "beta");
        assert_eq!(parsed[2].name, "gamma");

        // Test 1D array
        let vars = "theta.1,theta.2,theta.3,theta.4";
        let parsed = parse(vars, &mut dims, &mut dim_sizes).unwrap();
        assert_eq!(parsed.len(), 1);
        let var = &parsed[0];
        assert_eq!(var.name, "theta");
        assert_eq!(var.shape.as_slice(), vec![4]);
        assert_eq!(var.num_elements, 4);

        // Test variable name with colons and dots
        let vars = "x:1:2.4:1.1,x:1:2.4:1.2,x:1:2.4:1.3";
        let parsed = parse(vars, &mut dims, &mut dim_sizes).unwrap();
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
        let parsed = parse(vars, &mut dims, &mut dim_sizes).unwrap();
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
