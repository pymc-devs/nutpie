use std::{collections::HashMap, ffi::c_void, sync::Arc};

use anyhow::{anyhow, bail, Context, Result};
use numpy::{NotContiguousError, PyReadonlyArray1};
use nuts_rs::{CpuLogpFunc, CpuMath, HasDims, LogpError, Model, Storable, Value};
use pyo3::{
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyDictMethods},
    Py, PyAny, PyErr, PyResult, Python,
};

use rand::Rng;
use thiserror::Error;

use crate::{
    common::{PyValue, PyVariable},
    wrapper::PyTransformAdapt,
};

type UserData = *const std::ffi::c_void;

type RawLogpFunc = unsafe extern "C" fn(
    usize,
    *const f64,
    *mut f64,
    *mut f64,
    *const std::ffi::c_void,
) -> std::os::raw::c_int;

type RawExpandFunc = unsafe extern "C" fn(
    usize,
    usize,
    *const f64,
    *mut f64,
    *const std::ffi::c_void,
) -> std::os::raw::c_int;

#[pyclass]
#[derive(Clone)]
pub(crate) struct LogpFunc {
    func: RawLogpFunc,
    _keep_alive: Arc<Py<PyAny>>,
    user_data_ptr: UserData,
}

unsafe impl Send for LogpFunc {}
unsafe impl Sync for LogpFunc {}

#[pymethods]
impl LogpFunc {
    #[new]
    fn new(ptr: usize, user_data_ptr: usize, keep_alive: Py<PyAny>) -> Result<Self> {
        let func =
            unsafe { std::mem::transmute::<*const c_void, RawLogpFunc>(ptr as *const c_void) };

        Ok(Self {
            func,
            _keep_alive: Arc::new(keep_alive),
            user_data_ptr: user_data_ptr as UserData,
        })
    }
}

#[pyclass]
#[derive(Clone)]
pub(crate) struct ExpandFunc {
    func: RawExpandFunc,
    _keep_alive: Arc<Py<PyAny>>,
    user_data_ptr: UserData,
    dim: usize,
    expanded_dim: usize,
}

#[pymethods]
impl ExpandFunc {
    #[new]
    fn new(
        dim: usize,
        expanded_dim: usize,
        ptr: usize,
        user_data_ptr: usize,
        keep_alive: Py<PyAny>,
    ) -> Self {
        let func =
            unsafe { std::mem::transmute::<*const c_void, RawExpandFunc>(ptr as *const c_void) };
        Self {
            dim,
            expanded_dim,
            _keep_alive: Arc::new(keep_alive),
            user_data_ptr: user_data_ptr as UserData,
            func,
        }
    }
}

unsafe impl Send for ExpandFunc {}
unsafe impl Sync for ExpandFunc {}

impl HasDims for PyMcModelRef<'_> {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        self.model.dim_sizes.clone()
    }

    fn coords(&self) -> HashMap<String, Value> {
        self.model.coords.clone()
    }
}

pub struct ExpandedVector(Vec<Option<nuts_rs::Value>>);

impl<'f> Storable<PyMcModelRef<'f>> for ExpandedVector {
    fn names<'a>(parent: &'a PyMcModelRef<'f>) -> Vec<&'a str> {
        parent
            .model
            .variables
            .iter()
            .map(|var| var.name.as_str())
            .collect()
    }

    fn item_type(parent: &PyMcModelRef<'f>, item: &str) -> nuts_rs::ItemType {
        parent
            .model
            .variables
            .iter()
            .find(|var| var.name == item)
            .map(|var| var.item_type.as_inner().clone())
            .expect("Item not found")
    }

    fn dims<'a>(parent: &'a PyMcModelRef<'f>, item: &str) -> Vec<&'a str> {
        parent
            .model
            .variables
            .iter()
            .find(|var| var.name == item)
            .map(|var| var.dims.as_slice().iter().map(|s| s.as_str()).collect())
            .expect("Item not found")
    }

    fn get_all<'a>(
        &'a mut self,
        parent: &'a PyMcModelRef<'f>,
    ) -> Vec<(&'a str, Option<nuts_rs::Value>)> {
        self.0
            .iter_mut()
            .zip(parent.model.variables.iter())
            .map(|(val, var)| (var.name.as_str(), val.take()))
            .collect()
    }
}

#[derive(Debug, Error)]
pub enum PyMcLogpError {
    #[error("Python error: {0}")]
    PyError(#[from] PyErr),
    #[error("Python retured a non-contigous array")]
    NotContiguousError(#[from] NotContiguousError),
    #[error("Unknown error: {0}")]
    Anyhow(#[from] anyhow::Error),
    #[error("Logp function returned error code: {0}")]
    ErrorCode(std::os::raw::c_int),
}

impl LogpError for PyMcLogpError {
    fn is_recoverable(&self) -> bool {
        match self {
            Self::PyError(err) => Python::attach(|py| {
                let Ok(attr) = err.value(py).getattr("is_recoverable") else {
                    return false;
                };
                attr.is_truthy()
                    .expect("Could not access is_recoverable in error check")
            }),
            Self::NotContiguousError(_) => false,
            Self::Anyhow(_) => false,
            Self::ErrorCode(code) => *code > (0 as std::os::raw::c_int),
        }
    }
}

pub struct PyMcModelRef<'a> {
    model: &'a PyMcModel,
    transform_adapter: Option<PyTransformAdapt>,
}

impl CpuLogpFunc for PyMcModelRef<'_> {
    type LogpError = PyMcLogpError;
    type FlowParameters = Py<PyAny>;
    type ExpandedVector = ExpandedVector;

    fn dim(&self) -> usize {
        self.model.dim
    }

    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, Self::LogpError> {
        let mut logp = 0f64;
        let logp_ptr = (&mut logp) as *mut f64;
        assert!(position.len() == self.model.dim);
        assert!(gradient.len() == self.model.dim);
        let retcode = unsafe {
            (self.model.density.func)(
                self.model.dim,
                position.as_ptr(),
                gradient.as_mut_ptr(),
                logp_ptr,
                self.model.density.user_data_ptr,
            )
        };
        if retcode == 0 {
            return Ok(logp);
        }
        Err(PyMcLogpError::ErrorCode(retcode))
    }

    fn expand_vector<R>(
        &mut self,
        _rng: &mut R,
        array: &[f64],
    ) -> std::result::Result<Self::ExpandedVector, nuts_rs::CpuMathError>
    where
        R: rand::Rng + ?Sized,
    {
        let mut out = vec![0f64; self.model.expand.expanded_dim].into_boxed_slice();
        let retcode = unsafe {
            (self.model.expand.func)(
                self.model.expand.dim,
                self.model.expand.expanded_dim,
                array.as_ptr(),
                out.as_mut_ptr(),
                self.model.expand.user_data_ptr,
            )
        };

        let mut values = Vec::new();
        for var in self.model.variables.iter() {
            let start = var.start_idx.expect("Variable has no start index");
            let end = var.end_idx.expect("Variable has no end index");
            let slice = &out[start..end];

            let value = match var.item_type.as_inner() {
                nuts_rs::ItemType::U64 => {
                    let vec: Vec<u64> = slice.iter().map(|&x| x as u64).collect();
                    nuts_rs::Value::U64(vec.into())
                }
                nuts_rs::ItemType::I64 => {
                    let vec: Vec<i64> = slice.iter().map(|&x| x as i64).collect();
                    nuts_rs::Value::I64(vec.into())
                }
                nuts_rs::ItemType::F64 => {
                    let vec: Vec<f64> = slice.iter().map(|&x| x as f64).collect();
                    nuts_rs::Value::F64(vec.into())
                }
                nuts_rs::ItemType::F32 => {
                    let vec: Vec<f32> = slice.iter().map(|&x| x as f32).collect();
                    nuts_rs::Value::F32(vec.into())
                }
                nuts_rs::ItemType::Bool => {
                    let vec: Vec<bool> = slice.iter().map(|&x| x != 0.0).collect();
                    nuts_rs::Value::Bool(vec.into())
                }
                nuts_rs::ItemType::String => {
                    return Err(nuts_rs::CpuMathError::ExpandError(
                        "String type not supported in expansion".into(),
                    ));
                }
                nuts_rs::ItemType::DateTime64(_) | nuts_rs::ItemType::TimeDelta64(_) => {
                    return Err(nuts_rs::CpuMathError::ExpandError(
                        "DateTime64 and TimeDelta64 types not supported in expansion".into(),
                    ));
                }
            };

            values.push(Some(value));
        }

        if retcode == 0 {
            Ok(ExpandedVector(values))
        } else {
            Err(nuts_rs::CpuMathError::ExpandError(format!(
                "Expand function returned error code {}",
                retcode
            )))
        }
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
            )?;
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
        let (logp, logdet) = self
            .transform_adapter
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("No transformation adapter specified"))?
            .init_from_transformed_position(
                params,
                untransformed_position,
                untransformed_gradient,
                transformed_position,
                transformed_gradient,
            )?;
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
        let (logp, logdet) = self
            .transform_adapter
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("No transformation adapter specified"))?
            .init_from_untransformed_position(
                params,
                untransformed_position,
                untransformed_gradient,
                transformed_position,
                transformed_gradient,
            )?;
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
            )?;
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
            .new_transformation(rng, untransformed_position, untransformed_gradient, chain)?;
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

#[pyclass]
#[derive(Clone)]
pub(crate) struct PyMcModel {
    dim: usize,
    density: LogpFunc,
    expand: ExpandFunc,
    init_func: Arc<Py<PyAny>>,
    transform_adapter: Option<PyTransformAdapt>,
    variables: Arc<Vec<PyVariable>>,
    dim_sizes: HashMap<String, u64>,
    coords: HashMap<String, Value>,
}

#[pymethods]
impl PyMcModel {
    #[new]
    fn new<'py>(
        py: Python<'py>,
        density: LogpFunc,
        expand: ExpandFunc,
        variables: Vec<PyVariable>,
        dim: usize,
        dim_sizes: Py<PyDict>,
        coords: Py<PyDict>,
        init_func: Py<PyAny>,
        transform_adapter: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let dim_sizes = dim_sizes
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

        Ok(Self {
            dim,
            density,
            expand,
            init_func: init_func.into(),
            coords,
            dim_sizes,
            transform_adapter: transform_adapter.map(PyTransformAdapt::new),
            variables: Arc::new(variables),
        })
    }

    /*
    fn benchmark_logp<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray1<'py, f64>,
        cores: usize,
        evals: usize,
    ) -> PyResult<&'py PyList> {
        let point = point.to_vec()?;
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

impl Model for PyMcModel {
    type Math<'model> = CpuMath<PyMcModelRef<'model>>;

    fn math<R: Rng + ?Sized>(&self, _rng: &mut R) -> Result<Self::Math<'_>> {
        Ok(CpuMath::new(PyMcModelRef {
            model: self,
            transform_adapter: self.transform_adapter.clone(),
        }))
    }

    fn init_position<R: rand::Rng + ?Sized>(
        &self,
        rng: &mut R,
        position: &mut [f64],
    ) -> Result<()> {
        let seed = rng.next_u64();

        Python::attach(|py| {
            let init_point = self
                .init_func
                .call1(py, (seed,))
                .context("Failed to initialize point")?;

            let init_point: PyReadonlyArray1<f64> = init_point
                .extract(py)
                .map_err(|_| anyhow!("Initialization array returned incorrect argument"))?;

            let init_point = init_point
                .as_slice()
                .context("Initial point must be contiguous")?;

            if init_point.len() != position.len() {
                bail!("Initial point has incorrect length");
            }

            position.copy_from_slice(init_point);
            Ok(())
        })?;
        Ok(())
    }
}
