use std::{ffi::c_void, fmt::Display, sync::Arc};

use anyhow::{Context, Result};
use arrow::{
    array::{Array, FixedSizeListArray, Float64Array, StructArray},
    datatypes::{DataType, Field, Fields},
};
use itertools::{izip, Itertools};
use numpy::PyReadonlyArray1;
use nuts_rs::{CpuLogpFunc, CpuMath, DrawStorage, LogpError, Model, Settings};
use pyo3::{
    pyclass, pymethods,
    types::{PyAnyMethods, PyList},
    Bound, PyObject, PyResult,
};
use rand::{distributions::Uniform, prelude::Distribution};

use thiserror::Error;

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
    type LogpError = ErrorCode;

    fn dim(&self) -> usize {
        self.dim
    }

    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, Self::LogpError> {
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

#[derive(Clone)]
pub(crate) struct PyMcTrace<'model> {
    dim: usize,
    data: Vec<Vec<f64>>,
    var_sizes: Vec<usize>,
    var_names: Vec<String>,
    expand: &'model ExpandFunc,
}

impl<'model> DrawStorage for PyMcTrace<'model> {
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

    fn finalize(self) -> Result<Arc<dyn Array>> {
        let (fields, arrays): (Vec<_>, _) = izip!(self.data, self.var_names, self.var_sizes)
            .map(|(data, name, size)| {
                let data = Float64Array::from(data);
                let item_field = Arc::new(Field::new("item", DataType::Float64, false));
                let array =
                    FixedSizeListArray::new(item_field.clone(), size as _, Arc::new(data), None);
                let field = Field::new(name, DataType::FixedSizeList(item_field, size as _), false);
                (Arc::new(field), Arc::new(array) as Arc<dyn Array>)
            })
            .unzip();

        let fields = Fields::from(fields);
        Ok(Arc::new(
            StructArray::try_new(fields, arrays, None).context("Could not create arrow struct")?,
        ))
    }

    fn inspect(&self) -> Result<Arc<dyn Array>> {
        self.clone().finalize()
    }
}

impl<'model> PyMcTrace<'model> {
    fn new(model: &'model PyMcModel, settings: &impl Settings) -> Self {
        let draws = settings.hint_num_draws() + settings.hint_num_tune();
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
        dim: usize,
        density: LogpFunc,
        expand: ExpandFunc,
        var_sizes: &Bound<'py, PyList>,
        var_names: &Bound<'py, PyList>,
        start_point: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Self> {
        Ok(Self {
            dim,
            density,
            expand,
            mu: start_point.as_slice()?.into(),
            var_names: var_names.extract()?,
            var_sizes: var_sizes.extract()?,
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
    type Math<'model> = CpuMath<&'model LogpFunc>;

    type DrawStorage<'model, S: Settings> = PyMcTrace<'model>;

    fn math(&self) -> Result<Self::Math<'_>> {
        Ok(CpuMath::new(&self.density))
    }

    fn init_position<R: rand::Rng + ?Sized>(
        &self,
        rng: &mut R,
        position: &mut [f64],
    ) -> Result<()> {
        let dist = Uniform::new(-2f64, 2f64);
        position
            .iter_mut()
            .zip_eq(self.mu.iter())
            .for_each(|(x, mu)| *x = dist.sample(rng) + mu);
        Ok(())
    }

    fn new_trace<'model, S: Settings, R: rand::prelude::Rng + ?Sized>(
        &'model self,
        _rng: &mut R,
        _chain_id: u64,
        settings: &'model S,
    ) -> Result<Self::DrawStorage<'model, S>> {
        Ok(PyMcTrace::new(self, settings))
    }
}
