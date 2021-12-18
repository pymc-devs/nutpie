use std::thread::JoinHandle;
use std::time::Duration;

use crossbeam::channel::Receiver;
use numpy::{PyArray1, PyReadonlyArray1, IntoPyArray};
use nuts_rs::cpu_sampler::JitterInitFunc;
use pyo3::exceptions::PyTypeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyFunction;

use nuts_rs::cpu_sampler::{AdaptiveSampler, CpuLogpFunc};

type GradFunc =
    unsafe extern "C" fn(usize, *const f64, *mut f64, *mut f64, *const std::ffi::c_void) -> i64;
type UserData = *const std::ffi::c_void;

#[pyclass]
struct Stats {
    stats: nuts_rs::cpu_sampler::SampleStats,
}

#[pymethods]
impl Stats {
    #[getter]
    fn mean_acceptance_rate(&self) -> f64 {
        self.stats.mean_acceptance_rate
    }

    #[getter]
    fn depth(&self) -> u64 {
        self.stats.depth
    }

    #[getter]
    fn is_diverging(&self) -> bool {
        self.stats.divergence_info.is_some()
    }
    
    #[getter]
    fn divergence_trajectory_idx(&self) -> Option<i64> {
        self.stats.divergence_info.as_ref()?.end_idx_in_trajectory()
    }

    #[getter]
    fn step_size(&self) -> f64 {
        self.stats.step_size
    }

    #[getter]
    fn step_size_bar(&self) -> f64 {
        self.stats.step_size_bar
    }

    #[getter]
    fn logp(&self) -> f64 {
        self.stats.logp
    }

    #[getter]
    fn idx_in_trajectory(&self) -> i64 {
        self.stats.idx_in_trajectory
    }

    #[getter]
    fn chain(&self) -> u64 {
        self.stats.chain
    }

    #[getter]
    fn draw(&self) -> u64 {
        self.stats.draw
    }

    #[getter]
    fn tree_size(&self) -> u64 {
        self.stats.tree_size
    }

    #[getter]
    fn first_diag_mass_matrix(&self) -> f64 {
        self.stats.first_diag_mass_matrix
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


impl CpuLogpFunc for PtrLogpFunc {
    type Err = i64;

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
        Err(retcode)
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

#[pyclass(unsendable)]
struct PtrSampler {
    sampler: AdaptiveSampler<PtrLogpFunc>,
}

#[pymethods]
impl PtrSampler {
    #[new]
    unsafe fn new(
        py: Python,
        func: usize,
        user_data_init_fn: Py<PyFunction>,
        dim: usize,
        settings: SamplerArgs,
        chain: u64,
        seed: u64,
    ) -> PyResult<PtrSampler> {
        let func: GradFunc = std::mem::transmute(func as *const std::ffi::c_void);
        let user_data: isize = user_data_init_fn.call0(py)?.extract(py)?;

        let func = PtrLogpFunc::new(func, user_data as UserData, dim, user_data_init_fn);

        Ok(PtrSampler {
            sampler: AdaptiveSampler::new(func, settings.inner, chain, seed),
        })
    }

    fn set_position<'py>(&mut self, _py: Python<'py>, init: PyReadonlyArray1<'py, f64>) -> PyResult<()> {
        self.sampler.set_position(init.as_slice()?).map_err(|e| PyValueError::new_err(format!("Could not evaluate logp. Return code was {}.", e)))?;
        Ok(())
    }

    fn draw<'py>(&mut self, py: Python<'py>) -> PyResult<(&'py PyArray1<f64>, Stats)> {
        let (out, stats) = self.sampler.draw();
        Ok((out.into_pyarray(py), Stats { stats }))
    }
}

struct PyLogpFunc {
    pyfunc: PyObject,
    dim: usize,
}

impl CpuLogpFunc for PyLogpFunc {
    type Err = PyErr;

    fn dim(&self) -> usize {
        self.dim
    }

    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, Self::Err> {
        Python::with_gil(|py| {
            let out = numpy::PyArray1::from_slice(py, gradient);
            let pos = numpy::PyArray1::from_slice(py, position);
            let logp = self.pyfunc.call1(py, (pos, out))?;
            Ok(logp.extract::<f64>(py)?)
        })
    }
}

impl PyLogpFunc {
    fn new(pyfunc: PyObject, dim: usize) -> PyLogpFunc {
        PyLogpFunc { pyfunc, dim }
    }
}

#[pyclass(unsendable)]
struct PySampler {
    sampler: AdaptiveSampler<PyLogpFunc>,
}

#[pymethods]
impl PySampler {
    #[new]
    fn new(
        py: Python,
        pyfunc: PyObject,
        settings: SamplerArgs,
        dim: usize,
        chain: u64,
        seed: u64,
    ) -> PyResult<PySampler> {
        if !pyfunc.cast_as::<PyAny>(py)?.is_callable() {
            return Err(PyErr::new::<PyTypeError, _>("func must be callable."));
        }
        let func = PyLogpFunc::new(pyfunc, dim);
        let args = settings.inner;

        Ok(PySampler {
            sampler: AdaptiveSampler::new(func, args, chain, seed),
        })
    }

    fn set_position<'py>(
        &mut self, _py: Python<'py>,
        init: PyReadonlyArray1<'py, f64>
    ) -> PyResult<()> {
        self.sampler.set_position(init.as_slice()?)?;
        Ok(())
    }

    fn draw<'py>(&mut self, py: Python<'py>) -> PyResult<(&'py PyArray1<f64>, Stats)> {
        let (out, stats) = self.sampler.draw();
        Ok((out.into_pyarray(py), Stats { stats }))
    }
}

#[pyclass]
#[derive(Clone, Default)]
pub struct SamplerArgs {
    inner: nuts_rs::cpu_sampler::SamplerArgs
}

#[pymethods]
impl SamplerArgs {
    #[new]
    fn new() -> SamplerArgs {
        SamplerArgs::default()
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
    fn initial_step(&self) -> f64 {
        self.inner.initial_step
    }

    #[setter(initial_step)]
    fn set_initial_step(&mut self, val: f64) {
        self.inner.initial_step = val
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
    fn max_energy_error(&self) -> f64 {
        self.inner.max_energy_error
    }

    #[setter(energy_error)]
    fn set_max_energy_error(&mut self, val: f64) {
        self.inner.max_energy_error = val
    }

    #[setter(target_accept)]
    fn set_target_accept(&mut self, val: f64) {
        self.inner.step_size_adapt.target = val;
    }

    #[getter]
    fn target_accept(&self) -> f64 {
        self.inner.step_size_adapt.target
    }
}

#[pyclass]
pub struct ParallelSampler {
    handle: Option<JoinHandle<Result<Vec<()>, ()>>>,
    channel: Option<Receiver<(Box<[f64]>, nuts_rs::cpu_sampler::SampleStats)>>,
}


#[pymethods]
impl ParallelSampler {
    #[new]
    fn new<'py>(
        py: Python<'py>,
        func: usize,
        user_data_init_fn: Py<PyFunction>,
        dim: usize,
        start_point: PyReadonlyArray1<'py, f64>,
        settings: SamplerArgs,
        n_chains: u64,
        n_draws: u64,
        seed: u64,
        n_try_init: u64,
    ) -> PyResult<ParallelSampler>
    {
        let func: GradFunc = unsafe { std::mem::transmute(func as *const std::ffi::c_void) };
        let user_data: isize = user_data_init_fn.call0(py)?.extract(py)?;

        let func = unsafe { PtrLogpFunc::new(func, user_data as UserData, dim, user_data_init_fn) };
        let mut init_point_func = JitterInitFunc::new();  // TODO use start_point

        let (handle, channel) = nuts_rs::cpu_sampler::sample_parallel(
            func,
            &mut init_point_func,
            settings.inner,
            n_chains,
            n_draws,
            seed,
            n_try_init,
        ).map_err(|e| PyValueError::new_err(format!("Logp function returned error {}", e)))?;

        Ok(ParallelSampler {
            handle: Some(handle),
            channel: Some(channel),
        })
    }

    fn __iter__(self_: PyRef<Self>) -> Py<ParallelSampler> {
        self_.into()
    }

    fn __next__(mut self_: PyRefMut<Self>) -> PyResult<Option<(Py<PyArray1<f64>>, Stats)>> {
        let channel = match self_.channel {
            Some(ref val) => { val },
            None => { return Ok(None) },
        };
        loop {
            match channel.recv_timeout(Duration::from_millis(10)) {
                Err(crossbeam::channel::RecvTimeoutError::Timeout) => {
                    Python::with_gil(|py| {
                        py.check_signals()
                    })?;
                    continue;
                },
                Err(crossbeam::channel::RecvTimeoutError::Disconnected) => {
                    self_.finalize()?;
                    return Ok(None)
                },
                Ok((ref draw, stats)) => {
                    let draw: PyResult<Py<PyArray1<f64>>> = Python::with_gil(|py| {
                        py.check_signals()?;
                        Ok(PyArray1::from_slice(py, &draw).into_py(py))
                    });
                    let stats = Stats { stats };
                    return Ok(Some((draw?, stats)));
                }
            };
        };
    }

    fn finalize(&mut self) -> PyResult<()> {
        drop(self.channel.take());
        if let Some(handle) = self.handle.take() {
            let result = handle
                .join()
                .map_err(|_| PyValueError::new_err("Worker process paniced."))?;
            result.map_err(|_| PyValueError::new_err("Worker thread failed."))?;
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
    m.add_class::<PtrSampler>()?;
    m.add_class::<ParallelSampler>()?;
    m.add_class::<SamplerArgs>()?;
    Ok(())
}
