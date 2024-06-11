use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use crate::{
    progress::{IndicatifHandler, ProgressHandler},
    pymc::{ExpandFunc, LogpFunc, PyMcModel},
    stan::{StanLibrary, StanModel},
};

use anyhow::{Context, Result};
use arrow::array::Array;
use nuts_rs::{
    ChainProgress, DiagGradNutsSettings, ProgressCallback, Sampler, SamplerWaitResult, Trace,
};
use pyo3::{
    exceptions::PyTimeoutError,
    ffi::Py_uintptr_t,
    prelude::*,
    types::{PyList, PyTuple},
};
use rand::{thread_rng, RngCore};

#[pyclass]
struct PyChainProgress(ChainProgress);

#[pymethods]
impl PyChainProgress {
    #[getter]
    fn finished_draws(&self) -> usize {
        self.0.finished_draws
    }

    #[getter]
    fn total_draws(&self) -> usize {
        self.0.total_draws
    }

    #[getter]
    fn divergences(&self) -> usize {
        self.0.divergences
    }

    #[getter]
    fn started(&self) -> bool {
        self.0.started
    }

    #[getter]
    fn tuning(&self) -> bool {
        self.0.tuning
    }

    #[getter]
    fn num_steps(&self) -> usize {
        self.0.latest_num_steps
    }

    #[getter]
    fn step_size(&self) -> f64 {
        self.0.step_size
    }
}

#[pyclass]
#[derive(Clone, Default)]
pub struct PyDiagGradNutsSettings(DiagGradNutsSettings);

#[pymethods]
impl PyDiagGradNutsSettings {
    #[new]
    fn new(seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or_else(|| {
            let mut rng = thread_rng();
            rng.next_u64()
        });
        let settings = DiagGradNutsSettings {
            seed,
            ..Default::default()
        };
        PyDiagGradNutsSettings(settings)
    }

    #[getter]
    fn num_tune(&self) -> u64 {
        self.0.num_tune
    }

    #[setter(num_tune)]
    fn set_num_tune(&mut self, val: u64) {
        self.0.num_tune = val
    }

    #[getter]
    fn num_chains(&self) -> usize {
        self.0.num_chains
    }

    #[setter(num_chains)]
    fn set_num_chains(&mut self, val: usize) {
        self.0.num_chains = val;
    }

    #[getter]
    fn num_draws(&self) -> u64 {
        self.0.num_draws
    }

    #[setter(num_draws)]
    fn set_num_draws(&mut self, val: u64) {
        self.0.num_draws = val;
    }

    #[getter]
    fn window_switch_freq(&self) -> u64 {
        self.0.mass_matrix_adapt.mass_matrix_switch_freq
    }

    #[setter(window_switch_freq)]
    fn set_window_switch_freq(&mut self, val: u64) {
        self.0.mass_matrix_adapt.mass_matrix_switch_freq = val;
    }

    #[getter]
    fn early_window_switch_freq(&self) -> u64 {
        self.0.mass_matrix_adapt.early_mass_matrix_switch_freq
    }

    #[setter(early_window_switch_freq)]
    fn set_early_window_switch_freq(&mut self, val: u64) {
        self.0.mass_matrix_adapt.early_mass_matrix_switch_freq = val;
    }
    #[getter]
    fn initial_step(&self) -> f64 {
        self.0.mass_matrix_adapt.dual_average_options.initial_step
    }

    #[setter(initial_step)]
    fn set_initial_step(&mut self, val: f64) {
        self.0.mass_matrix_adapt.dual_average_options.initial_step = val
    }

    #[getter]
    fn maxdepth(&self) -> u64 {
        self.0.maxdepth
    }

    #[setter(maxdepth)]
    fn set_maxdepth(&mut self, val: u64) {
        self.0.maxdepth = val
    }

    #[getter]
    fn store_gradient(&self) -> bool {
        self.0.store_gradient
    }

    #[setter(store_gradient)]
    fn set_store_gradient(&mut self, val: bool) {
        self.0.store_gradient = val;
    }

    #[getter]
    fn store_unconstrained(&self) -> bool {
        self.0.store_unconstrained
    }

    #[setter(store_unconstrained)]
    fn set_store_unconstrained(&mut self, val: bool) {
        self.0.store_unconstrained = val;
    }

    #[getter]
    fn store_divergences(&self) -> bool {
        self.0.store_divergences
    }

    #[setter(store_divergences)]
    fn set_store_divergences(&mut self, val: bool) {
        self.0.store_divergences = val;
    }

    #[getter]
    fn max_energy_error(&self) -> f64 {
        self.0.max_energy_error
    }

    #[setter(max_energy_error)]
    fn set_max_energy_error(&mut self, val: f64) {
        self.0.max_energy_error = val
    }

    #[setter(target_accept)]
    fn set_target_accept(&mut self, val: f64) {
        self.0.mass_matrix_adapt.dual_average_options.target_accept = val;
    }

    #[getter]
    fn target_accept(&self) -> f64 {
        self.0.mass_matrix_adapt.dual_average_options.target_accept
    }

    #[getter]
    fn store_mass_matrix(&self) -> bool {
        self.0
            .mass_matrix_adapt
            .mass_matrix_options
            .store_mass_matrix
    }

    #[setter(store_mass_matrix)]
    fn set_store_mass_matrix(&mut self, val: bool) {
        self.0
            .mass_matrix_adapt
            .mass_matrix_options
            .store_mass_matrix = val;
    }

    #[getter]
    fn use_grad_based_mass_matrix(&self) -> bool {
        self.0
            .mass_matrix_adapt
            .mass_matrix_options
            .use_grad_based_estimate
    }

    #[setter(use_grad_based_mass_matrix)]
    fn set_use_grad_based_mass_matrix(&mut self, val: bool) {
        self.0
            .mass_matrix_adapt
            .mass_matrix_options
            .use_grad_based_estimate = val
    }
}

pub(crate) enum SamplerState {
    Running(Sampler),
    Finished(Option<Trace>),
    Empty,
}

#[derive(Clone)]
#[pyclass]
pub enum ProgressType {
    Callback {
        rate: Duration,
        n_cores: usize,
        template: String,
        callback: Py<PyAny>,
    },
    Indicatif {
        rate: Duration,
    },
    None {},
}

impl ProgressType {
    fn into_callback(self) -> Result<Option<ProgressCallback>> {
        match self {
            ProgressType::Callback {
                callback,
                rate,
                n_cores,
                template,
            } => {
                let handler = ProgressHandler::new(callback, rate, template, n_cores);
                let callback = handler.into_callback()?;

                Ok(Some(callback))
            }
            ProgressType::Indicatif { rate } => {
                let handler = IndicatifHandler::new(rate);
                Ok(Some(handler.into_callback()?))
            }
            ProgressType::None {} => Ok(None),
        }
    }
}

#[pymethods]
impl ProgressType {
    #[staticmethod]
    fn indicatif(rate: u64) -> Self {
        let rate = Duration::from_millis(rate);
        ProgressType::Indicatif { rate }
    }

    #[staticmethod]
    fn none() -> Self {
        ProgressType::None {}
    }

    #[staticmethod]
    fn template_callback(rate: u64, template: String, n_cores: usize, callback: Py<PyAny>) -> Self {
        let rate = Duration::from_millis(rate);
        ProgressType::Callback {
            callback,
            template,
            n_cores,
            rate,
        }
    }
}

#[pyclass]
struct PySampler(SamplerState);

#[pymethods]
impl PySampler {
    #[staticmethod]
    fn from_pymc(
        settings: PyDiagGradNutsSettings,
        cores: usize,
        model: PyMcModel,
        progress_type: ProgressType,
    ) -> PyResult<PySampler> {
        let callback = progress_type.into_callback()?;
        let sampler = Sampler::new(model, settings.0, cores, callback)?;
        Ok(PySampler(SamplerState::Running(sampler)))
    }

    #[staticmethod]
    fn from_stan(
        settings: PyDiagGradNutsSettings,
        cores: usize,
        model: StanModel,
        progress_type: ProgressType,
    ) -> PyResult<PySampler> {
        let callback = progress_type.into_callback()?;
        let sampler = Sampler::new(model, settings.0, cores, callback)?;
        Ok(PySampler(SamplerState::Running(sampler)))
    }

    fn is_finished(&mut self, py: Python<'_>) -> PyResult<bool> {
        py.allow_threads(|| {
            let state = std::mem::replace(&mut self.0, SamplerState::Empty);

            let SamplerState::Running(sampler) = state else {
                let _ = std::mem::replace(&mut self.0, state);
                return Ok(true);
            };

            match sampler.wait_timeout(Duration::from_millis(1)) {
                SamplerWaitResult::Trace(trace) => {
                    let _ = std::mem::replace(&mut self.0, SamplerState::Finished(Some(trace)));
                    Ok(true)
                }
                SamplerWaitResult::Timeout(sampler) => {
                    let _ = std::mem::replace(&mut self.0, SamplerState::Running(sampler));
                    Ok(false)
                }
                SamplerWaitResult::Err(err, trace) => {
                    let _ = std::mem::replace(&mut self.0, SamplerState::Finished(trace));
                    Err(err.into())
                }
            }
        })
    }

    fn pause(&mut self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            if let SamplerState::Running(ref mut control) = self.0 {
                control.pause()?
            }
            Ok(())
        })
    }

    fn resume(&mut self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            if let SamplerState::Running(ref mut control) = self.0 {
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

            let state = std::mem::replace(&mut self.0, SamplerState::Empty);

            let SamplerState::Running(mut control) = state else {
                let _ = std::mem::replace(&mut self.0, state);
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
                    SamplerWaitResult::Trace(trace) => {
                        break (SamplerState::Finished(Some(trace)), Ok(()))
                    }
                    SamplerWaitResult::Timeout(new_control) => {
                        control = new_control;
                    }
                    SamplerWaitResult::Err(err, trace) => {
                        break (SamplerState::Finished(trace), Err(err.into()))
                    }
                }

                if let Err(err) = Python::with_gil(|py| py.check_signals()) {
                    break (SamplerState::Running(control), Err(err));
                }
            };

            let _ = std::mem::replace(&mut self.0, final_state);
            retval
        })
    }

    fn abort(&mut self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            let state = std::mem::replace(&mut self.0, SamplerState::Empty);

            let SamplerState::Running(control) = state else {
                let _ = std::mem::replace(&mut self.0, state);
                return Ok(());
            };

            let (result, trace) = control.abort();
            let _ = std::mem::replace(&mut self.0, SamplerState::Finished(trace));
            result?;
            Ok(())
        })
    }

    /*
    fn finalize(&mut self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            let state = std::mem::replace(&mut self.0, SamplerState::Empty);

            let SamplerState::Running(sampler) = state else {
                let _ = std::mem::replace(&mut self.0, state);
                return Ok(());
            };

            let result = sampler.finalize();
            let _ = std::mem::replace(&mut self.0, SamplerState::Finished(result));
            Ok(())
        })
    }
    */

    fn extract_results<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let state = std::mem::replace(&mut self.0, SamplerState::Empty);

        let SamplerState::Finished(trace) = state else {
            let _ = std::mem::replace(&mut self.0, state);
            return Err(anyhow::anyhow!("Sampler is not finished"))?;
        };

        let Some(trace) = trace else {
            return Err(anyhow::anyhow!(
                "Sampler failed and did not produce a trace"
            ))?;
        };

        trace_to_list(trace, py)
    }

    fn is_empty(&self) -> bool {
        match self.0 {
            SamplerState::Running(_) => false,
            SamplerState::Finished(_) => false,
            SamplerState::Empty => true,
        }
    }

    fn inspect<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let trace = py.allow_threads(|| {
            let SamplerState::Running(ref mut sampler) = self.0 else {
                return Err(anyhow::anyhow!("Sampler is not running"))?;
            };

            sampler.inspect_trace()
        })?;
        trace_to_list(trace, py)
    }
}

fn trace_to_list(trace: Trace, py: Python<'_>) -> PyResult<Bound<'_, PyList>> {
    let list = PyList::new_bound(
        py,
        trace
            .chains
            .into_iter()
            .map(|chain| {
                Ok(PyTuple::new_bound(
                    py,
                    [
                        export_array(py, chain.draws)?,
                        export_array(py, chain.stats)?,
                    ]
                    .into_iter(),
                ))
            })
            .collect::<Result<Vec<_>>>()?,
    );
    Ok(list)
}

fn export_array(py: Python<'_>, data: Arc<dyn Array>) -> PyResult<PyObject> {
    let pa = py.import_bound("pyarrow")?;
    let array = pa.getattr("Array")?;

    let data = data.into_data();

    let (data, schema) = arrow::ffi::to_ffi(&data).context("Could not convert to arrow ffi")?;

    let data = array
        .call_method1(
            "_import_from_c",
            (
                (&data as *const _ as Py_uintptr_t).into_py(py),
                (&schema as *const _ as Py_uintptr_t).into_py(py),
            ),
        )
        .context("Could not import arrow trace in python")?;
    Ok(data.into_py(py))
}

/// A Python module implemented in Rust.
#[pymodule]
pub fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySampler>()?;
    m.add_class::<PyMcModel>()?;
    m.add_class::<LogpFunc>()?;
    m.add_class::<ExpandFunc>()?;
    m.add_class::<StanLibrary>()?;
    m.add_class::<StanModel>()?;
    m.add_class::<PyDiagGradNutsSettings>()?;
    m.add_class::<PyChainProgress>()?;
    m.add_class::<ProgressType>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
