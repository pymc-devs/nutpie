use std::time::{Duration, Instant};

use crate::{
    pymc::{ExpandFunc, LogpFunc, PyMcModel},
    sampler::{Sampler, SamplerControl, SamplerResult, SamplerWaitResult},
    stan::{StanLibrary, StanModel},
};

use anyhow::{Context, Result};
use arrow2::{array::Array, datatypes::Field};
use nuts_rs::{SampleStats, SamplerArgs};
use pyo3::{
    exceptions::{PyTimeoutError, PyValueError},
    ffi::Py_uintptr_t,
    prelude::*,
    types::{PyList, PyTuple},
};

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
        self.stats.divergence_info().as_ref()?.end_idx_in_trajectory
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
}

#[pyclass]
#[derive(Clone, Default)]
pub struct PySamplerArgs {
    inner: SamplerArgs,
}

#[pymethods]
impl PySamplerArgs {
    #[new]
    fn new() -> PySamplerArgs {
        PySamplerArgs {
            inner: SamplerArgs::default(),
        }
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
    fn num_draws(&self) -> u64 {
        self.inner.num_draws
    }

    #[setter(num_draws)]
    fn set_num_draws(&mut self, val: u64) {
        self.inner.num_draws = val;
    }

    #[getter]
    fn window_switch_freq(&self) -> u64 {
        self.inner.mass_matrix_adapt.mass_matrix_switch_freq
    }

    #[setter(window_switch_freq)]
    fn set_window_switch_freq(&mut self, val: u64) {
        self.inner.mass_matrix_adapt.mass_matrix_switch_freq = val;
    }

    #[getter]
    fn early_window_switch_freq(&self) -> u64 {
        self.inner.mass_matrix_adapt.early_mass_matrix_switch_freq
    }

    #[setter(early_window_switch_freq)]
    fn set_early_window_switch_freq(&mut self, val: u64) {
        self.inner.mass_matrix_adapt.early_mass_matrix_switch_freq = val;
    }
    #[getter]
    fn initial_step(&self) -> f64 {
        self.inner
            .mass_matrix_adapt
            .dual_average_options
            .initial_step
    }

    #[setter(initial_step)]
    fn set_initial_step(&mut self, val: f64) {
        self.inner
            .mass_matrix_adapt
            .dual_average_options
            .initial_step = val
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
    fn store_gradient(&self) -> bool {
        self.inner.store_gradient
    }

    #[setter(store_gradient)]
    fn set_store_gradient(&mut self, val: bool) {
        self.inner.store_gradient = val;
    }

    #[getter]
    fn store_unconstrained(&self) -> bool {
        self.inner.store_unconstrained
    }

    #[setter(store_unconstrained)]
    fn set_store_unconstrained(&mut self, val: bool) {
        self.inner.store_unconstrained = val;
    }

    #[getter]
    fn store_divergences(&self) -> bool {
        self.inner.store_divergences
    }

    #[setter(store_divergences)]
    fn set_store_divergences(&mut self, val: bool) {
        self.inner.store_divergences = val;
    }

    #[getter]
    fn max_energy_error(&self) -> f64 {
        self.inner.max_energy_error
    }

    #[setter(max_energy_error)]
    fn set_max_energy_error(&mut self, val: f64) {
        self.inner.max_energy_error = val
    }

    #[setter(target_accept)]
    fn set_target_accept(&mut self, val: f64) {
        self.inner
            .mass_matrix_adapt
            .dual_average_options
            .target_accept = val;
    }

    #[getter]
    fn target_accept(&self) -> f64 {
        self.inner
            .mass_matrix_adapt
            .dual_average_options
            .target_accept
    }

    #[getter]
    fn store_mass_matrix(&self) -> bool {
        self.inner
            .mass_matrix_adapt
            .mass_matrix_options
            .store_mass_matrix
    }

    #[setter(store_mass_matrix)]
    fn set_store_mass_matrix(&mut self, val: bool) {
        self.inner
            .mass_matrix_adapt
            .mass_matrix_options
            .store_mass_matrix = val;
    }
}

pub(crate) enum SamplerState {
    Running(SamplerControl),
    Finished(Result<SamplerResult>),
    Empty,
}

#[pyclass]
struct PySampler {
    state: SamplerState,
}

fn make_callback(callback: Option<Py<PyAny>>) -> Box<dyn FnMut(Box<dyn SampleStats>) + Send> {
    if let Some(callback) = callback {
        Box::new(move |stats| {
            let _ = Python::with_gil(|py| callback.call1(py, (PySampleStats { stats },)));
        })
    } else {
        Box::new(|_| {})
    }
}

#[pymethods]
impl PySampler {
    #[staticmethod]
    fn from_pymc(
        settings: PySamplerArgs,
        chains: u64,
        cores: usize,
        model: PyMcModel,
        seed: Option<u64>,
        callback: Option<Py<PyAny>>,
    ) -> PyResult<PySampler> {
        let sampler = Sampler::new(model, settings.inner, cores, chains, seed);
        let control = SamplerControl::new(sampler, make_callback(callback));
        Ok(PySampler {
            state: SamplerState::Running(control),
        })
    }

    #[staticmethod]
    fn from_stan(
        settings: PySamplerArgs,
        chains: u64,
        cores: usize,
        model: StanModel,
        seed: Option<u64>,
        callback: Option<Py<PyAny>>,
    ) -> PyResult<PySampler> {
        let sampler = Sampler::new(model, settings.inner, cores, chains, seed);
        let control = SamplerControl::new(sampler, make_callback(callback));
        Ok(PySampler {
            state: SamplerState::Running(control),
        })
    }

    fn is_finished(&mut self) -> PyResult<bool> {
        let state = std::mem::replace(&mut self.state, SamplerState::Empty);

        let SamplerState::Running(control) = state else {
            let _ = std::mem::replace(&mut self.state, state);
            return Ok(true);
        };

        match control.try_finalize() {
            SamplerWaitResult::Result(result) => {
                let _ = std::mem::replace(&mut self.state, SamplerState::Finished(result));
                Ok(true)
            }
            SamplerWaitResult::Timeout(control) => {
                let _ = std::mem::replace(&mut self.state, SamplerState::Running(control));
                Ok(false)
            }
        }
    }

    fn pause(&mut self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            if let SamplerState::Running(ref mut control) = self.state {
                control.pause()?
            }
            Ok(())
        })
    }

    fn resume(&mut self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            if let SamplerState::Running(ref mut control) = self.state {
                control.resume()?
            }
            Ok(())
        })
    }

    fn wait(&mut self, py: Python<'_>, timeout_seconds: Option<f64>) -> PyResult<()> {
        py.allow_threads(|| {
            let timeout = match timeout_seconds {
                Some(val) => Some(Duration::try_from_secs_f64(val).context("Invalid timeout")?),
                None => None,
            };

            let state = std::mem::replace(&mut self.state, SamplerState::Empty);

            let SamplerState::Running(mut control) = state else {
                let _ = std::mem::replace(&mut self.state, state);
                return Ok(());
            };

            let start_time = Instant::now();
            let step = Duration::from_millis(100);

            let (final_state, retval) = loop {
                let time_so_far = Instant::now().saturating_duration_since(start_time);
                let next_timeout = match timeout {
                    Some(timeout) => {
                        let Some(remaining) = timeout.checked_sub(time_so_far) else {
                            break (
                                SamplerState::Running(control),
                                Err(PyTimeoutError::new_err(
                                    "Timeout while waiting for sampler to finish",
                                )),
                            );
                        };
                        remaining.min(step)
                    }
                    None => step,
                };

                match control.wait_timeout(next_timeout) {
                    SamplerWaitResult::Result(result) => {
                        break (SamplerState::Finished(result), Ok(()))
                    }
                    SamplerWaitResult::Timeout(new_control) => {
                        control = new_control;
                    }
                }

                if let Err(err) = Python::with_gil(|py| py.check_signals()) {
                    break (SamplerState::Running(control), Err(err));
                }
            };

            let _ = std::mem::replace(&mut self.state, final_state);
            retval
        })
    }

    fn abort(&mut self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            let state = std::mem::replace(&mut self.state, SamplerState::Empty);

            let SamplerState::Running(control) = state else {
                let _ = std::mem::replace(&mut self.state, state);
                return Ok(());
            };

            let result = control.abort();
            let _ = std::mem::replace(&mut self.state, SamplerState::Finished(result));
            Ok(())
        })
    }

    fn finalize(&mut self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            let state = std::mem::replace(&mut self.state, SamplerState::Empty);

            let SamplerState::Running(control) = state else {
                let _ = std::mem::replace(&mut self.state, state);
                return Ok(());
            };

            let result = control.finalize();
            let _ = std::mem::replace(&mut self.state, SamplerState::Finished(result));
            Ok(())
        })
    }

    fn extract_results(&mut self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let state = std::mem::replace(&mut self.state, SamplerState::Empty);

        let SamplerState::Finished(values) = state else {
            let _ = std::mem::replace(&mut self.state, state);
            // Must have called finalize or abort
            return Err(anyhow::anyhow!("Sampler is not finalized"))?;
        };

        let values =
            values.map_err(|e| PyValueError::new_err(format!("Sampling failed: {:?}", e)))?;

        let list = PyList::new(
            py,
            values
                .into_iter()
                .map(|(stats, draws)| {
                    Ok(PyTuple::new(
                        py,
                        [
                            export_array(py, "sampler_stats".into(), stats)?,
                            draws
                                .map(|draws| export_array(py, "draws".into(), draws))
                                .transpose()?
                                .into_py(py),
                        ]
                        .into_iter(),
                    ))
                })
                .collect::<Result<Vec<_>>>()?,
        );
        Ok(list.into_py(py))
    }

    fn is_empty(&self) -> bool {
        match self.state {
            SamplerState::Running(_) => false,
            SamplerState::Finished(_) => false,
            SamplerState::Empty => true,
        }
    }
}

fn export_array(py: Python<'_>, name: String, data: Box<dyn Array>) -> PyResult<PyObject> {
    let pa = py.import("pyarrow")?;
    let array = pa.getattr("Array")?;

    let schema = arrow2::ffi::export_field_to_c(&Field::new(name, data.data_type().clone(), false));

    let data = arrow2::ffi::export_array_to_c(data);

    let data = array.call_method1(
        "_import_from_c",
        (
            (&data as *const _ as Py_uintptr_t).into_py(py),
            (&schema as *const _ as Py_uintptr_t).into_py(py),
        ),
    )?;
    Ok(data.into_py(py))
}

/// A Python module implemented in Rust.
#[pymodule]
pub fn _lib(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySamplerArgs>()?;
    m.add_class::<PySampleStats>()?;
    m.add_class::<PySampler>()?;
    m.add_class::<PyMcModel>()?;
    m.add_class::<LogpFunc>()?;
    m.add_class::<ExpandFunc>()?;
    m.add_class::<StanLibrary>()?;
    m.add_class::<StanModel>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
