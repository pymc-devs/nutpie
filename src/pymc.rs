use std::{ffi::c_void, fmt::Display};

use anyhow::{Context, Result};
use arrow2::{
    array::{Array, FixedSizeListArray, Float64Array, StructArray},
    datatypes::{DataType, Field},
};
use itertools::{izip, Itertools};
use numpy::PyReadonlyArray1;
use nuts_rs::{CpuLogpFunc, LogpError, SamplerArgs};
use pyo3::{pyclass, pymethods, types::PyList, PyObject, PyResult, Python};
use rand::{distributions::Uniform, prelude::Distribution};

use thiserror::Error;

use crate::sampler::{Model, Trace};

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
    _keep_alive: PyObject,
    user_data_ptr: UserData,
    dim: usize,
}

unsafe impl Send for LogpFunc {}
unsafe impl Sync for LogpFunc {}

#[pymethods]
impl LogpFunc {
    #[new]
    fn new(dim: usize, ptr: usize, user_data_ptr: usize, keep_alive: PyObject) -> Self {
        let func =
            unsafe { std::mem::transmute::<*const c_void, RawLogpFunc>(ptr as *const c_void) };
        Self {
            func,
            _keep_alive: keep_alive,
            user_data_ptr: user_data_ptr as UserData,
            dim,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub(crate) struct ExpandFunc {
    func: RawExpandFunc,
    _keep_alive: PyObject,
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
        keep_alive: PyObject,
    ) -> Self {
        let func =
            unsafe { std::mem::transmute::<*const c_void, RawExpandFunc>(ptr as *const c_void) };
        Self {
            dim,
            expanded_dim,
            _keep_alive: keep_alive,
            user_data_ptr: user_data_ptr as UserData,
            func,
        }
    }
}

unsafe impl Send for ExpandFunc {}
unsafe impl Sync for ExpandFunc {}

#[derive(Error, Debug)]
pub(crate) struct ErrorCode(std::os::raw::c_int);

impl Display for ErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Logp function returned error code {}", self.0)
    }
}

impl LogpError for ErrorCode {
    fn is_recoverable(&self) -> bool {
        self.0 > 0
    }
}

impl<'a> CpuLogpFunc for &'a LogpFunc {
    type Err = ErrorCode;

    fn dim(&self) -> usize {
        self.dim
    }

    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, Self::Err> {
        let mut logp = 0f64;
        let logp_ptr = (&mut logp) as *mut f64;
        assert!(position.len() == self.dim);
        assert!(gradient.len() == self.dim);
        let retcode = unsafe {
            (self.func)(
                self.dim,
                &position[0] as *const f64,
                &mut gradient[0] as *mut f64,
                logp_ptr,
                self.user_data_ptr,
            )
        };
        if retcode == 0 {
            return Ok(logp);
        }
        Err(ErrorCode(retcode))
    }
}

pub(crate) struct PyMcTrace<'model> {
    dim: usize,
    data: Vec<Vec<f64>>,
    var_sizes: Vec<usize>,
    var_names: Vec<String>,
    expand: &'model ExpandFunc,
}

impl<'model> Trace for PyMcTrace<'model> {
    fn append_value(&mut self, point: &[f64]) -> Result<()> {
        assert!(point.len() == self.dim);

        let point = self
            .expand_draw(point)
            .context("Could not compute deterministic variables")?;

        let mut start: usize = 0;
        for (&size, data) in self.var_sizes.iter().zip_eq(self.data.iter_mut()) {
            let end = start.checked_add(size).unwrap();
            let vals = &point[start..end];
            data.extend_from_slice(vals);
            start = end;
        }
        Ok(())
    }

    fn finalize(self) -> Result<Box<dyn Array>> {
        let (fields, arrays) = izip!(self.data, self.var_names, self.var_sizes)
            .map(|(data, name, size)| {
                let data = Float64Array::from_vec(data);
                let inner_field = Field::new("item", DataType::Float64, false);
                let dtype = DataType::FixedSizeList(Box::new(inner_field), size);
                let field = Field::new(name, dtype.clone(), false);
                (
                    field,
                    FixedSizeListArray::new(dtype, data.boxed(), None).boxed(),
                )
            })
            .unzip();

        let dtype = DataType::Struct(fields);
        Ok(StructArray::try_new(dtype, arrays, None)?.boxed())
    }
}

impl<'model> PyMcTrace<'model> {
    fn new(model: &'model PyMcModel, settings: &SamplerArgs) -> Self {
        let draws = (settings.num_tune + settings.num_draws) as usize;
        Self {
            dim: model.dim,
            data: model
                .var_sizes
                .iter()
                .map(|&size| Vec::with_capacity(size * draws))
                .collect(),
            var_sizes: model.var_sizes.clone(),
            var_names: model.var_names.clone(),
            expand: &model.expand,
        }
    }

    fn expand_draw(&mut self, point: &[f64]) -> Result<Box<[f64]>> {
        let mut out = vec![0f64; self.expand.expanded_dim].into_boxed_slice();
        let retcode = unsafe {
            (self.expand.func)(
                self.expand.dim,
                self.expand.expanded_dim,
                point.as_ptr(),
                out.as_mut_ptr(),
                self.expand.user_data_ptr,
            )
        };
        if retcode == 0 {
            Ok(out)
        } else {
            Err(anyhow::Error::msg("Failed to expand a draw."))
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub(crate) struct PyMcModel {
    dim: usize,
    density: LogpFunc,
    expand: ExpandFunc,
    mu: Box<[f64]>,
    var_sizes: Vec<usize>,
    var_names: Vec<String>,
}

#[pymethods]
impl PyMcModel {
    #[new]
    fn new<'py>(
        _py: Python<'py>,
        dim: usize,
        density: LogpFunc,
        expand: ExpandFunc,
        var_sizes: &'py PyList,
        var_names: &'py PyList,
        start_point: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Self> {
        Ok(Self {
            dim,
            density,
            expand,
            mu: start_point.to_vec()?.into(),
            var_names: var_names.extract()?,
            var_sizes: var_sizes.extract()?,
        })
    }
}

impl Model for PyMcModel {
    type Density<'a> = &'a LogpFunc;

    type Trace<'a> = PyMcTrace<'a>;

    fn new_trace<'a, R: rand::Rng + ?Sized>(
        &'a self,
        _rng: &mut R,
        _chain: u64,
        settings: &SamplerArgs,
    ) -> Result<Self::Trace<'a>> {
        Ok(PyMcTrace::new(self, settings))
    }

    fn density(&self) -> Result<Self::Density<'_>> {
        Ok(&self.density)
    }

    fn init_position<R: rand::Rng + ?Sized>(
        &self,
        rng: &mut R,
        position: &mut [f64],
    ) -> Result<()> {
        let dist = Uniform::new(-1f64, 1f64);
        position
            .iter_mut()
            .zip_eq(self.mu.iter())
            .for_each(|(x, mu)| *x = dist.sample(rng) + mu);
        Ok(())
    }
}
