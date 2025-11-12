use std::{collections::HashMap, sync::Arc};

use anyhow::{anyhow, bail, Context, Result};
use numpy::{
    NotContiguousError, PyArray1, PyReadonlyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use nuts_rs::{CpuLogpFunc, CpuMath, HasDims, LogpError, Model, Storable, Value};
use pyo3::{
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyDictMethods, PyList, PyListMethods},
    Bound, Py, PyAny, PyErr, Python,
};
use rand::Rng;
use rand_distr::{Distribution, Uniform};
use thiserror::Error;

use crate::{
    common::{PyValue, PyVariable},
    wrapper::PyTransformAdapt,
};

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyModel {
    make_logp_func: Arc<Py<PyAny>>,
    make_expand_func: Arc<Py<PyAny>>,
    init_point_func: Option<Arc<Py<PyAny>>>,
    variables: Arc<Vec<PyVariable>>,
    transform_adapter: Option<PyTransformAdapt>,
    ndim: usize,
    dim_sizes: HashMap<String, u64>,
    coords: HashMap<String, Value>,
}

#[pymethods]
impl PyModel {
    #[new]
    #[pyo3(signature = (make_logp_func, make_expand_func, variables, ndim, dim_sizes, coords, *, init_point_func=None, transform_adapter=None))]
    fn new<'py>(
        py: Python<'py>,
        make_logp_func: Py<PyAny>,
        make_expand_func: Py<PyAny>,
        variables: Vec<PyVariable>,
        ndim: usize,
        dim_sizes: Py<PyDict>,
        coords: Py<PyDict>,
        init_point_func: Option<Py<PyAny>>,
        transform_adapter: Option<Py<PyAny>>,
    ) -> Result<Self> {
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
            make_logp_func: Arc::new(make_logp_func),
            make_expand_func: Arc::new(make_expand_func),
            init_point_func: init_point_func.map(|x| x.into()),
            variables: Arc::new(variables),
            ndim,
            transform_adapter: transform_adapter.map(PyTransformAdapt::new),
            dim_sizes,
            coords,
        })
    }
}

#[derive(Debug, Error)]
pub enum PyLogpError {
    #[error("Bad logp value: {0}")]
    BadLogp(f64),
    #[error("Python error: {0}")]
    PyError(#[from] PyErr),
    #[error("logp function must return float.")]
    ReturnTypeError(),
    #[error("Python retured a non-contigous array")]
    NotContiguousError(#[from] NotContiguousError),
    #[error("Unknown error: {0}")]
    Anyhow(#[from] anyhow::Error),
}

impl LogpError for PyLogpError {
    fn is_recoverable(&self) -> bool {
        match self {
            Self::BadLogp(_) => true,
            Self::PyError(err) => Python::attach(|py| {
                let Ok(attr) = err.value(py).getattr("is_recoverable") else {
                    return false;
                };
                attr.is_truthy()
                    .expect("Could not access is_recoverable in error check")
            }),
            Self::ReturnTypeError() => false,
            Self::NotContiguousError(_) => false,
            Self::Anyhow(_) => false,
        }
    }
}

pub struct PyDensity {
    logp: Py<PyAny>,
    expand_func: Py<PyAny>,
    transform_adapter: Option<PyTransformAdapt>,
    dim: usize,
    variables: Arc<Vec<PyVariable>>,
    dim_sizes: HashMap<String, u64>,
    coords: HashMap<String, Value>,
}

impl PyDensity {
    fn new(
        logp_clone_func: &Py<PyAny>,
        expand_clone_func: &Py<PyAny>,
        dim: usize,
        transform_adapter: Option<&PyTransformAdapt>,
        variables: Arc<Vec<PyVariable>>,
        dim_sizes: HashMap<String, u64>,
        coords: HashMap<String, Value>,
    ) -> Result<Self> {
        let logp_func = Python::attach(|py| logp_clone_func.call0(py))?;
        let expand_func = Python::attach(|py| expand_clone_func.call1(py, (0u64, 0u64, 0u64)))?;
        let transform_adapter = transform_adapter.cloned();
        Ok(Self {
            logp: logp_func,
            expand_func,
            transform_adapter,
            dim,
            variables,
            dim_sizes,
            coords,
        })
    }
}

impl HasDims for PyDensity {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        self.dim_sizes.clone()
    }

    fn coords(&self) -> HashMap<String, Value> {
        self.coords.clone()
    }
}

pub struct ExpandedVector(Vec<Option<Value>>);

impl Storable<PyDensity> for ExpandedVector {
    fn names(parent: &PyDensity) -> Vec<&str> {
        parent
            .variables
            .iter()
            .map(|var| var.name.as_str())
            .collect()
    }

    fn item_type(parent: &PyDensity, item: &str) -> nuts_rs::ItemType {
        parent
            .variables
            .iter()
            .find(|var| var.name == item)
            .map(|var| var.item_type.as_inner().clone())
            .expect("Item not found")
    }

    fn dims<'a>(parent: &'a PyDensity, item: &str) -> Vec<&'a str> {
        parent
            .variables
            .iter()
            .find(|var| var.name == item)
            .map(|var| var.dims.as_slice().iter().map(|s| s.as_str()).collect())
            .expect("Item not found")
    }

    fn get_all<'a>(&'a mut self, parent: &'a PyDensity) -> Vec<(&'a str, Option<Value>)> {
        self.0
            .iter_mut()
            .zip(parent.variables.iter())
            .map(|(val, var)| (var.name.as_str(), val.take()))
            .collect()
    }
}

impl CpuLogpFunc for PyDensity {
    type LogpError = PyLogpError;
    type FlowParameters = Py<PyAny>;
    type ExpandedVector = ExpandedVector;

    fn logp(&mut self, position: &[f64], grad: &mut [f64]) -> Result<f64, Self::LogpError> {
        Python::attach(|py| {
            let pos_array = PyArray1::from_slice(py, position);
            let result = self.logp.call1(py, (pos_array,));
            match result {
                Ok(val) => {
                    let val: Result<(f64, PyReadonlyArray1<f64>), _> = val.extract(py);
                    let Ok(val) = val else {
                        return Err(PyLogpError::ReturnTypeError());
                    };
                    let (logp_val, grad_array) = val;
                    if !logp_val.is_finite() {
                        return Err(PyLogpError::BadLogp(logp_val));
                    }
                    grad.copy_from_slice(
                        // unsafty: We just allocated this array, and this is the only location
                        // where we use the array.
                        grad_array.as_slice().expect("Grad array is not contiguous"),
                    );
                    Ok(logp_val)
                }
                Err(err) => Err(PyLogpError::PyError(err)),
            }
        })
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn expand_vector<R>(
        &mut self,
        _rng: &mut R,
        array: &[f64],
    ) -> std::result::Result<Self::ExpandedVector, nuts_rs::CpuMathError>
    where
        R: rand::Rng + ?Sized,
    {
        Python::attach(|py| {
            let expanded = self
                .expand_func
                .call1(py, (PyArray1::from_slice(py, array),));
            let Ok(expanded) = expanded else {
                return Err(nuts_rs::CpuMathError::ExpandError(
                    "Expanding function raised an error".into(),
                ));
            };
            let expanded: Bound<PyDict> = expanded.extract(py).map_err(|_| {
                nuts_rs::CpuMathError::ExpandError("Expand function did not return a dict".into())
            })?;
            let values = expanded.iter();
            let vars = self.variables.iter();

            let mut expanded = Vec::with_capacity(self.variables.len());
            for (var, (name2, val)) in vars.zip(values) {
                let name2 = name2.extract::<&str>().map_err(|_| {
                    nuts_rs::CpuMathError::ExpandError("expand key was not a string".into())
                })?;
                if var.name != name2 {
                    return Err(nuts_rs::CpuMathError::ExpandError(format!(
                        "Unexpected expand key: expected {} but found {}",
                        var.name, name2
                    )));
                }

                if val.is_none() {
                    expanded.push(None);
                    continue;
                }

                fn as_value<'py, 'a, T>(
                    var: &'a PyVariable,
                    val: &'a Bound<'py, PyAny>,
                ) -> Result<PyReadonlyArrayDyn<'py, T>, nuts_rs::CpuMathError>
                where
                    T: numpy::Element + Clone,
                {
                    let arr: PyReadonlyArrayDyn<T> = val.extract().map_err(|_| {
                        nuts_rs::CpuMathError::ExpandError(format!(
                            "variable {} had incorrect type",
                            var.name
                        ))
                    })?;
                    if !arr.is_c_contiguous() {
                        return Err(nuts_rs::CpuMathError::ExpandError(format!(
                            "not c contiguous: {}",
                            var.name
                        )));
                    }
                    if arr.shape().len() != var.shape.as_slice().len() {
                        return Err(nuts_rs::CpuMathError::ExpandError(format!(
                            "unexpected number of dimensions for variable {}",
                            var.name
                        )));
                    }
                    if !arr
                        .shape()
                        .iter()
                        .zip(var.shape.as_slice())
                        .all(|(a, &b)| *a as u64 == b)
                    {
                        return Err(nuts_rs::CpuMathError::ExpandError(format!(
                            "unexpected shape for variable {}",
                            var.name
                        )));
                    }
                    Ok(arr)
                }

                let val_array = match var.item_type.as_inner() {
                    nuts_rs::ItemType::F64 => {
                        let arr = as_value::<f64>(var, &val)?;
                        let slice = arr.as_slice().map_err(|_| {
                            nuts_rs::CpuMathError::ExpandError("Could not read as slice".into())
                        })?;
                        Some(Value::F64(slice.to_vec()))
                    }
                    nuts_rs::ItemType::F32 => {
                        let arr = as_value::<f32>(var, &val)?;
                        let slice = arr.as_slice().map_err(|_| {
                            nuts_rs::CpuMathError::ExpandError("Could not read as slice".into())
                        })?;
                        Some(Value::F32(slice.to_vec()))
                    }
                    nuts_rs::ItemType::I64 => {
                        let arr = as_value::<i64>(var, &val)?;
                        let slice = arr.as_slice().map_err(|_| {
                            nuts_rs::CpuMathError::ExpandError("Could not read as slice".into())
                        })?;
                        Some(Value::I64(slice.to_vec()))
                    }
                    nuts_rs::ItemType::Bool => {
                        let arr = as_value::<bool>(var, &val)?;
                        let slice = arr.as_slice().map_err(|_| {
                            nuts_rs::CpuMathError::ExpandError("Could not read as slice".into())
                        })?;
                        Some(Value::Bool(slice.to_vec()))
                    }
                    nuts_rs::ItemType::U64 => {
                        let arr = as_value::<u64>(var, &val)?;
                        let slice = arr.as_slice().map_err(|_| {
                            nuts_rs::CpuMathError::ExpandError("Could not read as slice".into())
                        })?;
                        Some(Value::U64(slice.to_vec()))
                    }
                    nuts_rs::ItemType::String => {
                        let list: Bound<PyList> = val.extract().map_err(|_| {
                            nuts_rs::CpuMathError::ExpandError("did not return list".into())
                        })?;
                        if list.len() != var.shape.as_slice().iter().product::<u64>() as usize {
                            return Err(nuts_rs::CpuMathError::ExpandError(
                                "Incorrect number of items".into(),
                            ));
                        }
                        let vec: Vec<String> = list
                            .iter()
                            .map(|item| {
                                item.extract::<String>().map_err(|_| {
                                    nuts_rs::CpuMathError::ExpandError(
                                        "items were not all strings".into(),
                                    )
                                })
                            })
                            .collect::<Result<_, _>>()?;
                        Some(Value::Strings(vec))
                    }
                    nuts_rs::ItemType::DateTime64(date_time_unit) => {
                        let arr = as_value::<i64>(var, &val)?;
                        let slice = arr.as_slice().map_err(|_| {
                            nuts_rs::CpuMathError::ExpandError("Could not read as slice".into())
                        })?;
                        Some(Value::DateTime64(*date_time_unit, slice.to_vec()))
                    }
                    nuts_rs::ItemType::TimeDelta64(date_time_unit) => {
                        let arr = as_value::<i64>(var, &val)?;
                        let slice = arr.as_slice().map_err(|_| {
                            nuts_rs::CpuMathError::ExpandError("Could not read as slice".into())
                        })?;
                        Some(Value::TimeDelta64(*date_time_unit, slice.to_vec()))
                    }
                };
                expanded.push(val_array);
            }
            Ok(ExpandedVector(expanded))
        })
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

impl Model for PyModel {
    type Math<'model>
        = CpuMath<PyDensity>
    where
        Self: 'model;

    fn math<R: Rng + ?Sized>(&self, _rng: &mut R) -> Result<Self::Math<'_>> {
        Ok(CpuMath::new(PyDensity::new(
            &self.make_logp_func,
            &self.make_expand_func,
            self.ndim,
            self.transform_adapter.as_ref(),
            self.variables.clone(),
            self.dim_sizes.clone(),
            self.coords.clone(),
        )?))
    }

    fn init_position<R: rand::prelude::Rng + ?Sized>(
        &self,
        rng: &mut R,
        position: &mut [f64],
    ) -> Result<()> {
        let Some(init_func) = self.init_point_func.as_ref() else {
            let dist = Uniform::new(-2f64, 2f64).expect("Could not create uniform distribution");
            position.iter_mut().for_each(|x| *x = dist.sample(rng));
            return Ok(());
        };

        let seed = rng.next_u64();

        Python::attach(|py| {
            let init_point = init_func
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
