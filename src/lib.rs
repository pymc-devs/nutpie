use std::fmt::Display;
use std::thread::JoinHandle;
use std::time::Duration;

use numpy::{PyArray1, PyReadonlyArray1};
use nuts_rs::LogpError;
use nuts_rs::{JitterInitFunc, sample_parallel, SampleStats, SamplerArgs, SampleStatValue, CpuLogpFunc, NutsError, sample_sequentially};
use pyo3::exceptions::PyValueError;
use pyo3::{prelude::*, intern};
use pyo3::types::{PyFunction, PyDict};
use thiserror::Error;
use crossbeam::channel::Receiver;

type GradFunc =
    unsafe extern "C" fn(usize, *const f64, *mut f64, *mut f64, *const std::ffi::c_void) -> i64;
type UserData = *const std::ffi::c_void;

#[pyclass]
struct PySampleStats {
    stats: Box<dyn SampleStats>,
}

#[pymethods]
impl PySampleStats {
    #[getter]
    fn depth(&self) -> u64 {
        self.stats.depth()
    }

    #[getter]
    fn is_diverging(&self) -> bool {
        self.stats.divergence_info().is_some()
    }

    #[getter]
    fn divergence_trajectory_idx(&self) -> Option<i64> {
        self.stats.divergence_info().as_ref()?.end_idx_in_trajectory()
    }

    #[getter]
    fn logp(&self) -> f64 {
        self.stats.logp()
    }

    #[getter]
    fn index_in_trajectory(&self) -> i64 {
        self.stats.index_in_trajectory()
    }

    #[getter]
    fn chain(&self) -> u64 {
        self.stats.chain()
    }

    #[getter]
    fn draw(&self) -> u64 {
        self.stats.draw()
    }

    fn as_dict<'py>(&self, py: Python<'py>) -> PyResult<&'py PyDict> {
        let dict = PyDict::new(py);
        let results: Result<Vec<_>, _> = self.stats.to_vec().drain(..).map(|(key, val)| {
            match val {
                SampleStatValue::Array(val) => {
                    dict.set_item(key, PyArray1::from_slice(py, &val))
                },
                SampleStatValue::OptionArray(val) => {
                    match val {
                        Some(val) => {
                            dict.set_item(key, PyArray1::from_slice(py, &val))
                        },
                        None => {
                            dict.set_item(key, py.None())
                        }
                    }
                },
                SampleStatValue::U64(val) => {
                    dict.set_item(key, val)
                },
                SampleStatValue::I64(val) => {
                    dict.set_item(key, val)
                },
                SampleStatValue::F64(val) => {
                    dict.set_item(key, val)
                },
                SampleStatValue::Bool(val) => {
                    dict.set_item(key, val)
                },
            }
        }).collect();
        results?;
        Ok(dict)
    }
}

struct PtrLogpFunc {
    func: GradFunc,
    user_data: UserData,
    dim: usize,
    user_data_init_fn: Py<PyFunction>,
}


unsafe impl Send for PtrLogpFunc {}

impl Clone for PtrLogpFunc {
    fn clone(&self) -> Self {
        Python::with_gil(|py| {
            let user_data = self.user_data_init_fn
                    .call0(py)
                    .expect("Calling the user data generation function failed.")
                    .extract::<isize>(py)
                    .expect("User data generation function returned invalid results.");

            Self {
                func: self.func,//.clone(),
                user_data: user_data as _,
                dim: self.dim,
                user_data_init_fn: self.user_data_init_fn.clone(),
            }
        })
    }
}

#[derive(Error, Debug)]
struct ErrorCode(i64);

impl LogpError for ErrorCode {
    fn is_recoverable(&self) -> bool {
        self.0 > 0
    }
}

impl Display for ErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Logp function returned error code {}", self.0)
    }
}

impl CpuLogpFunc for PtrLogpFunc {
    type Err = ErrorCode;

    fn dim(&self) -> usize {
        self.dim
    }

    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, Self::Err> {
        let mut logp = 0f64;
        let logp_ptr = (&mut logp) as *mut f64;
        let func = self.func;
        assert!(position.len() == self.dim);
        assert!(gradient.len() == self.dim);
        let retcode = unsafe { func(self.dim, &position[0]as *const f64, &mut gradient[0] as *mut f64, logp_ptr, self.user_data) };
        if retcode == 0 {
            return Ok(logp);
        }
        Err(ErrorCode(retcode))
    }
}

impl PtrLogpFunc {
    unsafe fn new(func: GradFunc, user_data: UserData, dim: usize, user_data_init_fn: Py<PyFunction>) -> PtrLogpFunc {
        PtrLogpFunc {
            func,
            user_data,
            dim,
            user_data_init_fn,
        }
    }
}

#[derive(Error, Debug)]
enum PyLogpErr {
    #[error("Recoverable error in logp evaluation")]
    Recoverable {
        #[source]
        source: PyErr,
    },
    #[error("Non-recoverable error in logp evaluation")]
    NonRecoverable {
        #[source]
        source: PyErr,
    },
}

impl LogpError for PyLogpErr {
    fn is_recoverable(&self) -> bool {
        match self {
            PyLogpErr::Recoverable { source: _ } => { true },
            PyLogpErr::NonRecoverable { source: _ } => { false },
        }
    }
}

impl From<PyErr> for PyLogpErr {
    fn from(err: PyErr) -> Self {
        Python::with_gil(|py| {
            let recov: bool = match err.value(py).getattr(intern!(py, "is_recoverable")) {
                Result::Ok(recoverable) => {
                    match recoverable.extract::<bool>() {
                        Result::Ok(val) => { val },
                        Result::Err(_) => { false },
                    }
                },
                Result::Err(_) => { false },  // TODO
            };
            if recov {
                PyLogpErr::Recoverable { source: err }
            }
            else {
                PyLogpErr::NonRecoverable { source: err }
            }
        })
    }
}

#[pyclass]
struct PyLogpFunc {
    pyfunc: PyObject,
    dim: usize,
}

impl CpuLogpFunc for PyLogpFunc {
    type Err = PyLogpErr;

    fn dim(&self) -> usize {
        self.dim
    }

    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, Self::Err> {
        Python::with_gil(|py| {
            let out = numpy::PyArray1::from_slice(py, gradient);
            let pos = numpy::PyArray1::from_slice(py, position);
            let logp = self.pyfunc.call1(py, (pos, out))?;
            py.check_signals().map_err(|e| PyLogpErr::NonRecoverable { source: e })?;
            gradient.iter_mut().zip(out.to_owned_array().iter()).for_each(|(out, &val)| *out = val);
            Ok(logp.extract::<f64>(py)?)
        })
    }
}

#[pymethods]
impl PyLogpFunc {
    #[new]
    fn new(pyfunc: PyObject, dim: usize) -> PyLogpFunc {
        PyLogpFunc { pyfunc, dim }
    }
}

#[pyclass(unsendable)]
struct PySampler {
    sampler: Box<
        dyn Iterator<
            Item = Result<
                (Box<[f64]>, Box<dyn SampleStats>),
                NutsError
            >
        >
    >,
}

#[pymethods]
impl PySampler {
    #[staticmethod]
    fn from_pyfunc<'py>(
        func: PyObject,
        start_point: PyReadonlyArray1<'py, f64>,
        dim: usize,
        settings: PySamplerArgs,
        draws: u64,
        chain: u64,
        seed: u64,
    ) -> PyResult<PySampler> {
        let func = PyLogpFunc::new(func, dim);
        let draws = sample_sequentially(
            func,
            settings.inner,
            start_point.as_slice()?,
            draws,
            chain,
            seed
        ).map_err(|_| PyValueError::new_err(format!("Logp failed at initial location")))?;
        let sampler = Box::new(
            draws.map(|draw| {
                draw.map(|(pos, stats)| {
                    (pos, Box::new(stats) as Box<dyn SampleStats>)
                })
            })
        );
        Ok(
            PySampler {
                sampler
            }
        )
    }

    #[new]
    unsafe fn new<'py>(
        py: Python<'py>,
        func: usize,
        user_data_init_fn: Py<PyFunction>,
        start_point: PyReadonlyArray1<'py, f64>,
        dim: usize,
        settings: PySamplerArgs,
        draws: u64,
        chain: u64,
        seed: u64,
    ) -> PyResult<PySampler> {
        let func: GradFunc = std::mem::transmute(func as *const std::ffi::c_void);
        let user_data: isize = user_data_init_fn.call0(py)?.extract(py)?;

        let func = PtrLogpFunc::new(func, user_data as UserData, dim, user_data_init_fn);
        //let sampler = new_sampler(func, settings.inner, chain, seed);
        //sample_sequentially start_point

        let draws = sample_sequentially(
            func,
            settings.inner,
            start_point.as_slice()?,
            draws,
            chain,
            seed
        ).map_err(|_| PyValueError::new_err(format!("Logp failed at initial location")))?;
        let sampler = Box::new(
            draws.map(|draw| {
                draw.map(|(pos, stats)| {
                    (pos, Box::new(stats) as Box<dyn SampleStats>)
                })
            })
        );
        Ok(
            PySampler {
                sampler
            }
        )
    }

    fn __iter__(self_: PyRef<Self>) -> Py<PySampler> {
        self_.into()
    }

    fn __next__(mut self_: PyRefMut<Self>) -> PyResult<Option<(Py<PyArray1<f64>>, PySampleStats)>> {
        match self_.sampler.next() {
            Some(val) => {
                let (pos, stats) = val.map_err(|e| PyValueError::new_err(format!("Could not retrieve next draw: {}", e)))?;
                Python::with_gil(|py| {
                    //py.check_signals()?;
                    Ok(Some((numpy::PyArray1::from_vec(py, pos.into()).into(), PySampleStats { stats })))
                })
            },
            None => Ok(None),
        }
    }
}

#[pyclass]
#[derive(Clone, Default)]
pub struct PySamplerArgs {
    inner: SamplerArgs
}

#[pymethods]
impl PySamplerArgs {
    #[new]
    fn new() -> PySamplerArgs {
        PySamplerArgs { inner: SamplerArgs::default() }
    }

    #[getter]
    fn num_tune(&self) -> u64 {
        self.inner.num_tune
    }

    #[setter(num_tune)]
    fn set_num_tune(&mut self, val: u64) {
        self.inner.num_tune = val
    }

    #[getter]
    fn variance_decay(&self) -> f64 {
        self.inner.mass_matrix_adapt.variance_decay
    }

    #[setter(variance_decay)]
    fn set_variance_decay(&mut self, val: f64) {
        self.inner.mass_matrix_adapt.variance_decay = val;
    }

    #[getter]
    fn early_variance_decay(&self) -> f64 {
        self.inner.mass_matrix_adapt.early_variance_decay
    }

    #[setter(early_variance_decay)]
    fn set_early_variance_decay(&mut self, val: f64) {
        self.inner.mass_matrix_adapt.early_variance_decay = val;
    }

    #[getter]
    fn use_grad_init(&self) -> bool {
        self.inner.mass_matrix_adapt.grad_init
    }

    #[setter(use_grad_init)]
    fn set_use_grad_init(&mut self, val: bool) {
        self.inner.mass_matrix_adapt.grad_init = val;
    }

    #[getter]
    fn window_switch_freq(&self) -> u64 {
        self.inner.mass_matrix_adapt.window_switch_freq
    }

    #[setter(window_switch_freq)]
    fn set_window_switch_freq(&mut self, val: u64) {
        self.inner.mass_matrix_adapt.window_switch_freq = val;
    }

    #[getter]
    fn initial_step(&self) -> f64 {
        self.inner.step_size_adapt.params.initial_step
    }

    #[setter(initial_step)]
    fn set_initial_step(&mut self, val: f64) {
        self.inner.step_size_adapt.params.initial_step = val
    }

    #[getter]
    fn maxdepth(&self) -> u64 {
        self.inner.maxdepth
    }

    #[setter(maxdepth)]
    fn set_maxdepth(&mut self, val: u64) {
        self.inner.maxdepth = val
    }

    #[getter]
    fn discard_window(&self) -> u64 {
        self.inner.mass_matrix_adapt.discard_window
    }

    #[setter(discard_window)]
    fn set_discard_window(&mut self, val: u64) {
        self.inner.mass_matrix_adapt.discard_window = val;
    }

    #[getter]
    fn max_energy_error(&self) -> f64 {
        self.inner.max_energy_error
    }

    #[setter(energy_error)]
    fn set_max_energy_error(&mut self, val: f64) {
        self.inner.max_energy_error = val
    }

    #[getter]
    fn mass_matrix_final_window(&self) -> u64 {
        self.inner.mass_matrix_adapt.final_window
    }

    #[setter(mass_matrix_final_window)]
    fn set_mass_matrix_final_window(&mut self, val: u64) {
        self.inner.mass_matrix_adapt.final_window = val;
    }

    #[setter(target_accept)]
    fn set_target_accept(&mut self, val: f64) {
        self.inner.step_size_adapt.target_accept = val;
    }

    #[getter]
    fn target_accept(&self) -> f64 {
        self.inner.step_size_adapt.target_accept
    }

    #[getter]
    fn early_target_accept(&self) -> f64 {
        self.inner.step_size_adapt.early_target_accept
    }

    #[setter(early_target_accept)]
    fn set_early_target_accept(&mut self, val: f64) {
        self.inner.step_size_adapt.early_target_accept = val;
    }

    #[getter]
    fn save_mass_matrix(&self) -> bool {
        self.inner.mass_matrix_adapt.save_mass_matrix
    }

    #[setter(save_mass_matrix)]
    fn set_save_mass_matrix(&mut self, val: bool) {
        self.inner.mass_matrix_adapt.save_mass_matrix = val;
    }
}

#[pyclass(unsendable)]
struct PyParallelSampler {
    handle: Option<JoinHandle<Vec<nuts_rs::ParallelChainResult>>>,
    channel: Option<Receiver<(Box<[f64]>, Box<dyn nuts_rs::SampleStats>)>>,
}

#[pymethods]
impl PyParallelSampler {
    #[new]
    fn new<'py>(
        func: usize,
        user_data_init_fn: Py<PyFunction>,
        dim: usize,
        start_point: PyReadonlyArray1<'py, f64>,
        settings: PySamplerArgs,
        n_chains: u64,
        n_draws: u64,
        seed: u64,
        n_try_init: u64,
    ) -> PyResult<PyParallelSampler>
    {
        let func: GradFunc = unsafe { std::mem::transmute(func as *const std::ffi::c_void) };


        let user_data: isize = Python::with_gil(|py| {
            user_data_init_fn.call0(py)?.extract(py)
        })?;

        let func = unsafe { PtrLogpFunc::new(func, user_data as UserData, dim, user_data_init_fn) };
        let mut init_point_func = JitterInitFunc::new();  // TODO use start_point

        let (handle, channel) = sample_parallel(
            func,
            &mut init_point_func,
            settings.inner,
            n_chains,
            n_draws,
            seed,
            n_try_init,
        ).map_err(|e| PyValueError::new_err(format!("Logp function returned error {}", e)))?;

        Ok(PyParallelSampler {
            handle: Some(handle),
            channel: Some(channel),
        })
    }

    fn __iter__(self_: PyRef<Self>) -> Py<PyParallelSampler> {
        self_.into()
    }

    fn __next__(mut self_: PyRefMut<Self>, py: Python<'_>) -> PyResult<Option<(Py<PyArray1<f64>>, PySampleStats)>> {
        let channel = match self_.channel {
            Some(ref val) => { val },
            None => { return Ok(None) },
        };
        let mut call_finalize = false;
        let result = py.allow_threads(|| {
            loop {
                match channel.recv_timeout(Duration::from_millis(50)) {
                    Err(crossbeam::channel::RecvTimeoutError::Timeout) => {
                        Python::with_gil(|py| {
                            py.check_signals()
                        })?;
                        continue;
                    },
                    Err(crossbeam::channel::RecvTimeoutError::Disconnected) => {
                        call_finalize = true;
                        return Ok(None)
                    },
                    Ok((ref draw, stats)) => {
                        let draw: PyResult<Py<PyArray1<f64>>> = Python::with_gil(|py| {
                            py.check_signals()?;
                            Ok(PyArray1::from_slice(py, &draw).into_py(py))
                        });
                        let stats = PySampleStats { stats };
                        return Ok(Some((draw?, stats)));
                    }
                };
            };
        });
        if call_finalize {
            self_.finalize(py)?;
        }
        result
    }

    fn finalize(&mut self, py: Python<'_>) -> PyResult<()> {
        drop(self.channel.take());
        if let Some(handle) = self.handle.take() {
            let result: PyResult<_> = py.allow_threads(|| {
                let result: Result<Vec<_>, _> = handle
                    .join()
                    .map_err(|_| PyValueError::new_err("Worker process paniced."))?
                    .into_iter()
                    .collect();
                result.map_err(|e| PyValueError::new_err(format!("Worker thread failed: {}", e)))?;
                Ok(())
            });
            result?;
        };

        Ok(())
    }

    /*
    fn __next__<'py>(mut self_: PyRefMut<Self>, py: Python<'py>) -> PyResult<Option<(&'py PyArray1<f64>, Stats)>> {
        match self_.channel.recv() {
            Err(_) => {
                if let Some(handle) = self_.handle.take() {
                    let result = handle.join();
                    result.map_err(|_| PyValueError::new_err("Worker thread failed."))?
                }
                return Ok(None)
            },
            Ok((ref draw, stats)) => {
                let draw = PyArray1::from_slice(py, &draw);
                let stats = Stats { stats };
                return Ok(Some((draw, stats)));
            }
        };
    }
    */
}


/// A Python module implemented in Rust.
#[pymodule]
fn nuts_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySampler>()?;
    m.add_class::<PyParallelSampler>()?;
    m.add_class::<PySamplerArgs>()?;
    Ok(())
}
