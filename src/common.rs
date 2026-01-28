use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use numpy::{PyArray1, PyReadonlyArray1};
use nuts_rs::Value;
use pyo3::{
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyDictMethods, PyList, PyListMethods, PyType, PyTypeMethods},
    Borrowed, Bound, BoundObject, FromPyObject, IntoPyObject, IntoPyObjectExt, Py, PyAny, PyErr,
    Python,
};
use smallvec::SmallVec;

#[derive(Debug, Clone)]
pub struct Dims(pub SmallVec<[String; 4]>);

impl Dims {
    pub fn as_slice(&self) -> &[String] {
        &self.0
    }
}

#[derive(Debug, Clone)]
pub struct Shape(pub SmallVec<[u64; 4]>);

impl Shape {
    pub fn as_slice(&self) -> &[u64] {
        &self.0
    }
}

#[derive(Debug, Clone)]
pub struct ItemType(pub nuts_rs::ItemType);

impl ItemType {
    pub fn into_inner(self) -> nuts_rs::ItemType {
        self.0
    }

    pub fn as_inner(&self) -> &nuts_rs::ItemType {
        &self.0
    }
}

impl<'py> IntoPyObject<'py> for &Dims {
    type Target = PyList;
    type Output = Bound<'py, PyList>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> std::result::Result<Self::Output, Self::Error> {
        PyList::new(py, self.0.iter())
    }
}

impl<'py> IntoPyObject<'py> for &Shape {
    type Target = PyList;
    type Output = Bound<'py, PyList>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> std::result::Result<Self::Output, Self::Error> {
        PyList::new(py, self.0.iter())
    }
}

impl<'py> IntoPyObject<'py> for &ItemType {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> std::result::Result<Self::Output, Self::Error> {
        let dtype_str = match self.0 {
            nuts_rs::ItemType::U64 => "uint64",
            nuts_rs::ItemType::I64 => "int64",
            nuts_rs::ItemType::F64 => "float64",
            nuts_rs::ItemType::F32 => "float32",
            nuts_rs::ItemType::Bool => "bool",
            nuts_rs::ItemType::String => "object",
            nuts_rs::ItemType::DateTime64(unit) => match unit {
                nuts_rs::DateTimeUnit::Seconds => "datetime64[s]",
                nuts_rs::DateTimeUnit::Milliseconds => "datetime64[ms]",
                nuts_rs::DateTimeUnit::Microseconds => "datetime64[us]",
                nuts_rs::DateTimeUnit::Nanoseconds => "datetime64[ns]",
            },
            nuts_rs::ItemType::TimeDelta64(unit) => match unit {
                nuts_rs::DateTimeUnit::Seconds => "timedelta64[s]",
                nuts_rs::DateTimeUnit::Milliseconds => "timedelta64[ms]",
                nuts_rs::DateTimeUnit::Microseconds => "timedelta64[us]",
                nuts_rs::DateTimeUnit::Nanoseconds => "timedelta64[ns]",
            },
        };
        let numpy = py.import("numpy")?;
        let dtype = numpy.getattr("dtype")?.call1((dtype_str,))?;
        Ok(dtype)
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for ItemType {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> std::result::Result<Self, PyErr> {
        let dtype_str: &str = ob.extract()?;
        let item_type = match dtype_str {
            "uint64" => nuts_rs::ItemType::U64,
            "int64" => nuts_rs::ItemType::I64,
            "float64" => nuts_rs::ItemType::F64,
            "float32" => nuts_rs::ItemType::F32,
            "bool" => nuts_rs::ItemType::Bool,
            "object" => nuts_rs::ItemType::String,
            _ => {
                return Err(PyRuntimeError::new_err(format!(
                    "Unsupported item type: {}",
                    dtype_str
                )))
            }
        };
        Ok(ItemType(item_type))
    }
}

#[pyclass]
pub struct PyValue(Value);

impl<'a, 'py> FromPyObject<'a, 'py> for PyValue {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> std::result::Result<Self, PyErr> {
        let ob = if ob.hasattr("values")? {
            ob.getattr("values")?
        } else {
            ob.into_bound()
        };
        if let Ok(arr) = ob.extract::<PyReadonlyArray1<f64>>() {
            let vec = arr
                .as_slice()
                .map_err(|_| PyRuntimeError::new_err("Array is not contiguous"))?;
            return Ok(PyValue(Value::F64(vec.to_vec())));
        }
        if let Ok(arr) = ob.extract::<PyReadonlyArray1<f32>>() {
            let vec = arr
                .as_slice()
                .map_err(|_| PyRuntimeError::new_err("Array is not contiguous"))?;
            return Ok(PyValue(Value::F32(vec.to_vec())));
        }
        if let Ok(arr) = ob.extract::<PyReadonlyArray1<i64>>() {
            let vec = arr
                .as_slice()
                .map_err(|_| PyRuntimeError::new_err("Array is not contiguous"))?;
            return Ok(PyValue(Value::I64(vec.to_vec())));
        }
        if let Ok(arr) = ob.extract::<PyReadonlyArray1<u64>>() {
            let vec = arr
                .as_slice()
                .map_err(|_| PyRuntimeError::new_err("Array is not contiguous"))?;
            return Ok(PyValue(Value::U64(vec.to_vec())));
        }
        if let Ok(arr) = ob.extract::<PyReadonlyArray1<bool>>() {
            let vec = arr
                .as_slice()
                .map_err(|_| PyRuntimeError::new_err("Array is not contiguous"))?;
            return Ok(PyValue(Value::Bool(vec.to_vec())));
        }
        if let Ok(list) = ob.extract::<Bound<PyList>>() {
            let vec: Vec<String> = list
                .iter()
                .map(|item| {
                    item.extract::<String>()
                        .map_err(|_| PyRuntimeError::new_err("List item is not a string"))
                })
                .collect::<Result<_, _>>()?;
            return Ok(PyValue(Value::Strings(vec)));
        }
        if let Ok(arr) = ob.extract::<PyReadonlyArray1<Py<PyAny>>>() {
            let vec = arr
                .as_slice()
                .map_err(|_| PyRuntimeError::new_err("Array is not contiguous"))?;
            let vals_as_str = vec
                .iter()
                .map(|item| {
                    item.extract::<String>(ob.py())
                        .map_err(|_| PyRuntimeError::new_err("Array item is not a string"))
                })
                .collect::<Result<_, _>>()?;
            return Ok(PyValue(Value::Strings(vals_as_str)));
        }
        if ob.get_type().name()? == "ArrowStringArray" {
            let list: Bound<PyList> = ob.call_method0("tolist")?.extract().map_err(|_| {
                PyRuntimeError::new_err("Could not convert ArrowStringArray to list")
            })?;
            let vec: Vec<String> = list
                .iter()
                .map(|item| {
                    item.extract::<String>()
                        .map_err(|_| PyRuntimeError::new_err("List item is not a string"))
                })
                .collect::<Result<_, _>>()?;
            return Ok(PyValue(Value::Strings(vec)));
        }

        macro_rules! extract_time {
            ($unit:ident, $type:ident, $value:ident) => {
                if let Ok(arr) = ob.extract::<PyReadonlyArray1<numpy::datetime::$type<numpy::datetime::units::$unit>>>() {
                    let vec = arr
                        .as_slice()
                        .map_err(|_| PyRuntimeError::new_err("Array is not contiguous"))?;
                    let vals_as_i64 = vec.iter().map(|&dt| dt.into()).collect();
                    return Ok(PyValue(Value::$value(
                        nuts_rs::DateTimeUnit::$unit,
                        vals_as_i64,
                    )));
                }
            };
        }

        extract_time!(Seconds, Datetime, DateTime64);
        extract_time!(Milliseconds, Datetime, DateTime64);
        extract_time!(Microseconds, Datetime, DateTime64);
        extract_time!(Nanoseconds, Datetime, DateTime64);
        extract_time!(Seconds, Timedelta, TimeDelta64);
        extract_time!(Milliseconds, Timedelta, TimeDelta64);
        extract_time!(Microseconds, Timedelta, TimeDelta64);
        extract_time!(Nanoseconds, Timedelta, TimeDelta64);

        Err(PyRuntimeError::new_err(format!(
            "Could not convert to Value. Unsupported type: {}",
            ob.get_type().name()?
        )))
    }
}

impl PyValue {
    pub fn into_value(self) -> Value {
        self.0
    }

    pub fn into_array(self, py: Python) -> Result<Bound<PyAny>> {
        macro_rules! from_time {
            ($unit:ident, $items:expr, $type:ident) => {
                Ok(
                    PyArray1::<numpy::datetime::$type<numpy::datetime::units::$unit>>::from_vec(
                        py,
                        $items
                            .into_iter()
                            .map(|ts| {
                                numpy::datetime::$type::<numpy::datetime::units::$unit>::from(ts)
                            })
                            .collect(),
                    )
                    .into_any(),
                )
            };
        }

        match self.0 {
            Value::F64(vec) => Ok(PyArray1::from_vec(py, vec).into_any()),
            Value::F32(vec) => Ok(PyArray1::from_vec(py, vec).into_any()),
            Value::I64(vec) => Ok(PyArray1::from_vec(py, vec).into_any()),
            Value::U64(vec) => Ok(PyArray1::from_vec(py, vec).into_any()),
            Value::Bool(vec) => Ok(PyArray1::from_vec(py, vec).into_any()),
            Value::Strings(vec) => Ok(PyList::new(py, vec)?.into_any()),
            Value::ScalarString(val) => Ok(val.into_bound_py_any(py)?),
            Value::ScalarU64(val) => Ok(val.into_bound_py_any(py)?),
            Value::ScalarI64(val) => Ok(val.into_bound_py_any(py)?),
            Value::ScalarF64(val) => Ok(val.into_bound_py_any(py)?),
            Value::ScalarF32(val) => Ok(val.into_bound_py_any(py)?),
            Value::ScalarBool(val) => Ok(val.into_bound_py_any(py)?),
            Value::DateTime64(date_time_unit, items) => match date_time_unit {
                nuts_rs::DateTimeUnit::Seconds => from_time!(Seconds, items, Datetime),
                nuts_rs::DateTimeUnit::Milliseconds => from_time!(Milliseconds, items, Datetime),
                nuts_rs::DateTimeUnit::Microseconds => from_time!(Microseconds, items, Datetime),
                nuts_rs::DateTimeUnit::Nanoseconds => from_time!(Nanoseconds, items, Datetime),
            },
            Value::TimeDelta64(date_time_unit, items) => match date_time_unit {
                nuts_rs::DateTimeUnit::Seconds => from_time!(Seconds, items, Timedelta),
                nuts_rs::DateTimeUnit::Milliseconds => from_time!(Milliseconds, items, Timedelta),
                nuts_rs::DateTimeUnit::Microseconds => from_time!(Microseconds, items, Timedelta),
                nuts_rs::DateTimeUnit::Nanoseconds => from_time!(Nanoseconds, items, Timedelta),
            },
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PyVariable {
    #[pyo3(get)]
    pub name: String,
    pub item_type: ItemType,
    #[pyo3(get)]
    pub dims: Dims,
    #[pyo3(get)]
    pub shape: Shape,
    #[pyo3(get)]
    pub num_elements: usize,
    #[pyo3(get)]
    pub start_idx: Option<usize>,
    #[pyo3(get)]
    pub end_idx: Option<usize>,
}

impl PyVariable {
    pub fn new(
        name: String,
        item_type: ItemType,
        shape: Option<Vec<u64>>,
        all_dims: &mut HashMap<String, Vec<String>>,
        dim_sizes: &mut HashMap<String, u64>,
        start_idx: Option<usize>,
    ) -> anyhow::Result<Self> {
        let dims = all_dims.get(&name);

        let (dims, shape) = match (dims, shape) {
            (Some(dims), Some(shape)) => {
                if dims.len() != shape.len() {
                    bail!(
                        "Variable '{}': number of dims ({}) does not match number of shape entries ({})",
                        name,
                        dims.len(),
                        shape.len(),
                    );
                }
                for (dim, size) in dims.iter().zip(shape.iter()) {
                    if let Some(existing_size) = dim_sizes.get(dim) {
                        if *existing_size != *size {
                            bail!("Variable '{}': dimension '{}' has inconsistent size. Expected {}, but previously defined as {}",
                                name, dim, size, existing_size);
                        }
                    }
                }
                (dims.clone(), shape)
            }
            (Some(dims), None) => {
                let mut inferred_shape = Vec::new();
                for dim in dims.iter() {
                    if let Some(size) = dim_sizes.get(dim) {
                        inferred_shape.push(*size);
                    } else {
                        bail!(
                            "Variable '{}': dimension '{}' size unknown and no shape provided",
                            name,
                            dim
                        );
                    }
                }
                (dims.clone(), inferred_shape)
            }
            (None, Some(shape)) => {
                let mut inferred_dims = Vec::new();
                for (i, size) in shape.iter().enumerate() {
                    let generated_name = format!("{}_dim_{}", name, i);
                    if dim_sizes.contains_key(&generated_name) {
                        bail!("Variable '{}': generated anonymous dimension name '{}' already exists.",
                              name, generated_name);
                    }
                    dim_sizes.insert(generated_name.clone(), *size);
                    inferred_dims.push(generated_name);
                }
                all_dims.insert(name.clone(), inferred_dims.clone());
                (inferred_dims, shape)
            }
            (None, None) => {
                bail!("Variable '{}': no dims or shape provided", name);
            }
        };

        let num_elements = shape.iter().product::<u64>() as usize;

        Ok(PyVariable {
            name,
            item_type,
            dims: Dims(dims.into()),
            shape: Shape(shape.into()),
            num_elements,
            start_idx,
            end_idx: start_idx.map(|idx| idx + num_elements),
        })
    }
}

#[pymethods]
impl PyVariable {
    #[classmethod]
    fn new_variables<'py>(
        cls: &Bound<'py, PyType>,
        names: Vec<String>,
        item_types: Vec<String>,
        shapes: Vec<Option<Vec<u64>>>,
        dim_sizes: Py<PyDict>,
        dims: Py<PyDict>,
    ) -> Result<Vec<Self>> {
        let mut rust_all_dims = HashMap::new();
        let mut rust_dim_sizes = HashMap::new();

        let py = cls.py();

        for (key, value) in dims.bind(py).iter() {
            let key: String = key.extract().context("Dimension key is not a string")?;
            let value: Vec<String> = value
                .extract()
                .context("Dimension value is not a list of strings")?;
            rust_all_dims.insert(key, value);
        }

        for (key, value) in dim_sizes.bind(py).iter() {
            let key: String = key
                .extract()
                .context("Dimension size key is not a string")?;
            let value: u64 = value
                .extract()
                .context("Dimension size value is not an integer")?;
            rust_dim_sizes.insert(key, value);
        }

        let mut current_idx = 0;

        let variables = names
            .into_iter()
            .zip(item_types)
            .zip(shapes)
            .map(|((name, item_type), shape)| {
                let item_type = match item_type.as_str() {
                    "uint64" => ItemType(nuts_rs::ItemType::U64),
                    "int64" => ItemType(nuts_rs::ItemType::I64),
                    "float64" => ItemType(nuts_rs::ItemType::F64),
                    "float32" => ItemType(nuts_rs::ItemType::F32),
                    "bool" => ItemType(nuts_rs::ItemType::Bool),
                    "string" => ItemType(nuts_rs::ItemType::String),
                    _ => bail!("Unsupported item type: {}", item_type),
                };

                let start_idx = Some(current_idx);
                let var = Self::new(
                    name,
                    item_type,
                    shape,
                    &mut rust_all_dims,
                    &mut rust_dim_sizes,
                    start_idx,
                )
                .context("Could not create variable")?;
                current_idx += var.num_elements;
                Ok(var)
            })
            .collect::<Result<Vec<_>>>()?;

        let dim_sizes = dim_sizes.bind(py);
        for key in rust_dim_sizes.keys() {
            if !dim_sizes.contains(key).unwrap_or(false) {
                dim_sizes
                    .set_item(key, rust_dim_sizes[key])
                    .context("Could not update dimension sizes")?;
            }
        }

        let all_dims = dims.bind(py);
        for key in rust_all_dims.keys() {
            if !all_dims.contains(key).unwrap_or(false) {
                all_dims
                    .set_item(key, rust_all_dims[key].clone())
                    .context("Could not update all_dims")?;
            }
        }
        Ok(variables)
    }
}
