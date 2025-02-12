use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use arrow::{
    array::{
        Array, ArrayBuilder, BooleanBuilder, Float32Builder, Float64Builder, Int64Builder,
        LargeListBuilder, PrimitiveBuilder, StructBuilder,
    },
    datatypes::{DataType, Field, Float32Type, Float64Type, Int64Type},
};
use numpy::{NotContiguousError, PyArray1, PyReadonlyArray1};
use nuts_rs::{CpuLogpFunc, CpuMath, DrawStorage, LogpError, Model};
use pyo3::{
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyDictMethods},
    Bound, Py, PyAny, PyErr, Python,
};
use rand::Rng;
use rand_distr::{Distribution, Uniform};
use smallvec::SmallVec;
use thiserror::Error;

use crate::wrapper::PyTransformAdapt;

#[pyclass]
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PyVariable {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub dtype: ExpandDtype,
}

impl PyVariable {
    fn arrow_dtype(&self) -> DataType {
        match &self.dtype {
            ExpandDtype::Boolean {} => DataType::Boolean,
            ExpandDtype::Float64 {} => DataType::Float64,
            ExpandDtype::Float32 {} => DataType::Float32,
            ExpandDtype::Int64 {} => DataType::Int64,
            ExpandDtype::BooleanArray { tensor_type: _ } => {
                let field = Arc::new(Field::new("item", DataType::Boolean, false));
                DataType::LargeList(field)
            }
            ExpandDtype::ArrayFloat64 { tensor_type: _ } => {
                let field = Arc::new(Field::new("item", DataType::Float64, true));
                DataType::LargeList(field)
            }
            ExpandDtype::ArrayFloat32 { tensor_type: _ } => {
                let field = Arc::new(Field::new("item", DataType::Float32, false));
                DataType::LargeList(field)
            }
            ExpandDtype::ArrayInt64 { tensor_type: _ } => {
                let field = Arc::new(Field::new("item", DataType::Int64, false));
                DataType::LargeList(field)
            }
        }
    }
}

#[pymethods]
impl PyVariable {
    #[new]
    fn new(name: String, value_type: ExpandDtype) -> Self {
        Self {
            name,
            dtype: value_type,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyModel {
    make_logp_func: Arc<Py<PyAny>>,
    make_expand_func: Arc<Py<PyAny>>,
    init_point_func: Option<Arc<Py<PyAny>>>,
    variables: Arc<Vec<PyVariable>>,
    transform_adapter: Option<PyTransformAdapt>,
    ndim: usize,
}

#[pymethods]
impl PyModel {
    #[new]
    #[pyo3(signature = (make_logp_func, make_expand_func, variables, ndim, *, init_point_func=None, transform_adapter=None))]
    fn new<'py>(
        make_logp_func: Py<PyAny>,
        make_expand_func: Py<PyAny>,
        variables: Vec<PyVariable>,
        ndim: usize,
        init_point_func: Option<Py<PyAny>>,
        transform_adapter: Option<Py<PyAny>>,
    ) -> Self {
        Self {
            make_logp_func: Arc::new(make_logp_func),
            make_expand_func: Arc::new(make_expand_func),
            init_point_func: init_point_func.map(|x| x.into()),
            variables: Arc::new(variables),
            ndim,
            transform_adapter: transform_adapter.map(PyTransformAdapt::new),
        }
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
            Self::PyError(err) => Python::with_gil(|py| {
                let Ok(attr) = err.value(py).getattr("is_recoverable") else {
                    return false;
                };
                return attr
                    .is_truthy()
                    .expect("Could not access is_recoverable in error check");
            }),
            Self::ReturnTypeError() => false,
            Self::NotContiguousError(_) => false,
            Self::Anyhow(_) => false,
        }
    }
}

pub struct PyDensity {
    logp: Py<PyAny>,
    transform_adapter: Option<PyTransformAdapt>,
    dim: usize,
}

impl PyDensity {
    fn new(
        logp_clone_func: &Py<PyAny>,
        dim: usize,
        transform_adapter: Option<&PyTransformAdapt>,
    ) -> Result<Self> {
        let logp_func = Python::with_gil(|py| logp_clone_func.call0(py))?;
        let transform_adapter = transform_adapter.map(|val| val.clone());
        Ok(Self {
            logp: logp_func,
            transform_adapter,
            dim,
        })
    }
}

impl CpuLogpFunc for PyDensity {
    type LogpError = PyLogpError;
    type TransformParams = Py<PyAny>;

    fn logp(&mut self, position: &[f64], grad: &mut [f64]) -> Result<f64, Self::LogpError> {
        Python::with_gil(|py| {
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
                Err(err) => return Err(PyLogpError::PyError(err)),
            }
        })
    }

    fn dim(&self) -> usize {
        self.dim
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

pub struct PyTrace {
    expand: Py<PyAny>,
    variables: Arc<Vec<PyVariable>>,
    builder: StructBuilder,
}

impl PyTrace {
    pub fn new<R: Rng + ?Sized>(
        rng: &mut R,
        chain: u64,
        variables: Arc<Vec<PyVariable>>,
        make_expand_func: &Py<PyAny>,
        capacity: usize,
    ) -> Result<Self> {
        let seed1 = rng.next_u64();
        let seed2 = rng.next_u64();
        let expand = Python::with_gil(|py| {
            make_expand_func
                .call1(py, (seed1, seed2, chain))
                .context("Failed to call expand function factory")
        })?;

        let fields: Vec<Field> = variables
            .iter()
            .map(|variable| Field::new(variable.name.clone(), variable.arrow_dtype(), false))
            .collect();
        let builder = StructBuilder::from_fields(fields, capacity);

        Ok(Self {
            expand,
            variables,
            builder,
        })
    }
}

pub type ShapeVec = SmallVec<[usize; 4]>;

#[derive(Debug, Clone)]
#[non_exhaustive]
#[pyclass]
pub struct TensorShape {
    pub shape: ShapeVec,
    pub dims: Vec<Option<String>>,
    size: usize,
}

impl TensorShape {
    pub fn new(shape: ShapeVec, dims: Vec<Option<String>>) -> Self {
        let size = shape.iter().product();
        Self { shape, dims, size }
    }
    pub fn size(&self) -> usize {
        return self.size;
    }
}

#[pymethods]
impl TensorShape {
    #[new]
    #[pyo3(signature = (shape, dims=None))]
    fn py_new(shape: Vec<usize>, dims: Option<Vec<Option<String>>>) -> Result<Self> {
        let dims = dims.unwrap_or(shape.iter().map(|_| None).collect());
        if dims.len() != shape.len() {
            bail!("Number of dimensions must be the same as the shape");
        }

        let size = shape.iter().product();
        Ok(Self {
            shape: shape.into(),
            dims,
            size,
        })
    }
}

#[non_exhaustive]
#[pyclass]
#[derive(Debug, Clone)]
pub enum ExpandDtype {
    Boolean {},
    Float64 {},
    Float32 {},
    Int64 {},
    BooleanArray { tensor_type: TensorShape },
    ArrayFloat64 { tensor_type: TensorShape },
    ArrayFloat32 { tensor_type: TensorShape },
    ArrayInt64 { tensor_type: TensorShape },
}

#[pymethods]
impl ExpandDtype {
    #[staticmethod]
    fn boolean() -> Self {
        Self::Boolean {}
    }

    #[staticmethod]
    fn float64() -> Self {
        Self::Float64 {}
    }

    #[staticmethod]
    fn float32() -> Self {
        Self::Float32 {}
    }

    #[staticmethod]
    fn int64() -> Self {
        Self::Int64 {}
    }

    #[staticmethod]
    fn boolean_array(shape: TensorShape) -> Self {
        Self::BooleanArray { tensor_type: shape }
    }

    #[staticmethod]
    fn float64_array(shape: TensorShape) -> Self {
        Self::ArrayFloat64 { tensor_type: shape }
    }
    #[staticmethod]
    fn float32_array(shape: TensorShape) -> Self {
        Self::ArrayFloat32 { tensor_type: shape }
    }
    #[staticmethod]
    fn int64_array(shape: TensorShape) -> Self {
        Self::ArrayInt64 { tensor_type: shape }
    }

    #[getter]
    fn shape(&self) -> Option<Vec<usize>> {
        match self {
            Self::BooleanArray { tensor_type } => Some(tensor_type.shape.iter().cloned().collect()),
            Self::ArrayFloat64 { tensor_type } => Some(tensor_type.shape.iter().cloned().collect()),
            Self::ArrayFloat32 { tensor_type } => Some(tensor_type.shape.iter().cloned().collect()),
            Self::ArrayInt64 { tensor_type } => Some(tensor_type.shape.iter().cloned().collect()),
            _ => None,
        }
    }
}

impl DrawStorage for PyTrace {
    fn append_value(&mut self, point: &[f64]) -> Result<()> {
        Python::with_gil(|py| {
            let point = PyArray1::from_slice(py, point);
            let full_point = self
                .expand
                .call1(py, (point,))
                .context("Failed to call expand function")?
                .into_bound(py);
            let point: &Bound<PyDict> = full_point
                .downcast()
                .map_err(|_| anyhow!("expand function must return a dict"))
                .context("Expand function must return dict")?;
            point
                .iter()
                .zip(self.variables.iter())
                .enumerate()
                .try_for_each(|(i, ((key, value), variable))| {
                    let key: &str = key.extract()?;
                    if key != variable.name {
                        return Err(anyhow!("Incorrectly ordered expanded point"));
                    }

                    match &variable.dtype {
                        ExpandDtype::Boolean {} => {
                            let builder: &mut BooleanBuilder =
                                self.builder.field_builder(i).context(
                                    "Builder has incorrect type",
                                )?;
                            let value = value
                                .extract()
                                .expect("Return value from expand function could not be converted to boolean");
                            builder.append_value(value)
                        },
                        ExpandDtype::Float64 {} => {
                            let builder: &mut Float64Builder =
                                self.builder.field_builder(i).context(
                                    "Builder has incorrect type",
                                )?;
                            builder.append_value(
                                value
                                .extract()
                                .expect("Return value from expand function could not be converted to float64")
                            )
                        },
                        ExpandDtype::Float32 {} => {
                            let builder: &mut Float32Builder =
                                self.builder.field_builder(i).context(
                                    "Builder has incorrect type",
                                )?;
                            builder.append_value(
                                value
                                .extract()
                                .expect("Return value from expand function could not be converted to float32")
                            )
                        },
                        ExpandDtype::Int64 {} => {
                            let builder: &mut Int64Builder =
                                self.builder.field_builder(i).context(
                                    "Builder has incorrect type",
                                )?;
                            let value = value.extract().expect("Return value from expand function could not be converted to int64");
                            builder.append_value(value)
                        },
                        ExpandDtype::BooleanArray { tensor_type } => {
                            let builder: &mut LargeListBuilder<Box<dyn ArrayBuilder>> =
                                self.builder.field_builder(i).context(
                                    "Builder has incorrect type. Expected LargeListBuilder of Bool",
                                )?;
                            let value_builder = builder
                                .values()
                                .as_any_mut()
                                .downcast_mut::<BooleanBuilder>()
                                .context("Could not downcast builder to boolean type")?;
                            let values: PyReadonlyArray1<bool> = value.extract().context("Could not convert object to array")?;
                            if values.len()? != tensor_type.size() {
                                bail!("Extracted array has incorrect shape");
                            }
                            value_builder.append_slice(values.as_slice().context("Extracted array is not contiguous")?);
                            builder.append(true);
                        },
                        ExpandDtype::ArrayFloat64 { tensor_type } => {
                            //let builder: &mut FixedSizeListBuilder<Box<dyn ArrayBuilder>> =
                            let builder: &mut LargeListBuilder<Box<dyn ArrayBuilder>> =
                                self.builder.field_builder(i).context(
                                    "Builder has incorrect type. Expected LargeListBuilder of Float64",
                                )?;
                            let value_builder = builder
                                .values()
                                .as_any_mut()
                                .downcast_mut::<PrimitiveBuilder<Float64Type>>()
                                .context("Could not downcast builder to float64 type")?;
                            let values: PyReadonlyArray1<f64> = value.extract().context("Could not convert object to array")?;
                            if values.len()? != tensor_type.size() {
                                bail!("Extracted array has incorrect shape");
                            }
                            value_builder.append_slice(values.as_slice().context("Extracted array is not contiguous")?);
                            builder.append(true);
                        },
                        ExpandDtype::ArrayFloat32 { tensor_type } => {
                            let builder: &mut LargeListBuilder<Box<dyn ArrayBuilder>> =
                                self.builder.field_builder(i).context(
                                    "Builder has incorrect type. Expected LargeListBuilder of Float32",
                                )?;
                            let value_builder = builder
                                .values()
                                .as_any_mut()
                                .downcast_mut::<PrimitiveBuilder<Float32Type>>()
                                .context("Could not downcast builder to float32 type")?;
                            let values: PyReadonlyArray1<f32> = value.extract().context("Could not convert object to array")?;
                            if values.len()? != tensor_type.size() {
                                bail!("Extracted array has incorrect shape");
                            }
                            value_builder.append_slice(values.as_slice().context("Extracted array is not contiguous")?);
                            builder.append(true);
                        },
                        ExpandDtype::ArrayInt64 {tensor_type} => {
                            let builder: &mut LargeListBuilder<Box<dyn ArrayBuilder>> =
                                self.builder.field_builder(i).context(
                                    "Builder has incorrect type. Expected LargeListBuilder of Int64",
                                )?;
                            let value_builder = builder
                                .values()
                                .as_any_mut()
                                .downcast_mut::<PrimitiveBuilder<Int64Type>>()
                                .context("Could not downcast builder to i64 type")?;
                            let values: PyReadonlyArray1<i64> = value.extract().context("Could not convert object to array")?;
                            if values.len()? != tensor_type.size() {
                                bail!("Extracted array has incorrect shape");
                            }
                            value_builder.append_slice(values.as_slice().context("Extracted array is not contiguous")?);
                            builder.append(true);
                        },
                    }

                    Ok(())
                }).context("Could not save output of expand function to trace")?;
            self.builder.append(true);
            Ok(())
        })
    }

    fn finalize(mut self) -> Result<Arc<dyn Array>> {
        Ok(Arc::new(self.builder.finish()))
    }

    fn inspect(&self) -> Result<Arc<dyn Array>> {
        Ok(Arc::new(self.builder.finish_cloned()))
    }
}

impl Model for PyModel {
    type Math<'model>
        = CpuMath<PyDensity>
    where
        Self: 'model;

    type DrawStorage<'model, S: nuts_rs::Settings>
        = PyTrace
    where
        Self: 'model;

    fn new_trace<'model, S: nuts_rs::Settings, R: rand::prelude::Rng + ?Sized>(
        &'model self,
        rng: &mut R,
        chain_id: u64,
        settings: &'model S,
    ) -> Result<Self::DrawStorage<'model, S>> {
        let draws = settings.hint_num_tune() + settings.hint_num_draws();
        Ok(PyTrace::new(
            rng,
            chain_id,
            self.variables.clone(),
            &self.make_expand_func,
            draws,
        )
        .context("Could not create PyTrace object")?)
    }

    fn math(&self) -> Result<Self::Math<'_>> {
        Ok(CpuMath::new(PyDensity::new(
            &self.make_logp_func,
            self.ndim,
            self.transform_adapter.as_ref(),
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

        Python::with_gil(|py| {
            let init_point = init_func
                .call1(py, (seed,))
                .context("Failed to initialize point")?;

            let init_point: PyReadonlyArray1<f64> = init_point
                .extract(py)
                .context("Initializition array returned incorrect argument")?;

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
