use std::{
    fmt::Debug,
    sync::Arc,
    time::{Duration, Instant},
};

use crate::{
    progress::{IndicatifHandler, ProgressHandler},
    pyfunc::{ExpandDtype, PyModel, PyVariable, TensorShape},
    pymc::{ExpandFunc, LogpFunc, PyMcModel},
    stan::{StanLibrary, StanModel},
};

use anyhow::{bail, Context, Result};
use arrow::array::Array;
use nuts_rs::{
    AdaptOptions, ChainProgress, DiagAdaptExpSettings, DiagGradNutsSettings, LowRankNutsSettings,
    LowRankSettings, NutsSettings, ProgressCallback, Sampler, SamplerWaitResult, Trace,
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

#[derive(Clone)]
enum InnerSettings {
    LowRank(LowRankSettings),
    Diag(DiagAdaptExpSettings),
}

#[pyclass]
#[derive(Clone)]
pub struct PyNutsSettings {
    settings: NutsSettings<()>,
    adapt: InnerSettings,
}

enum Settings {
    Diag(DiagGradNutsSettings),
    LowRank(LowRankNutsSettings),
}

// Would be much nicer with
// https://doc.rust-lang.org/nightly/unstable-book/language-features/type-changing-struct-update.html
fn combine_settings<T: Debug + Copy + Default>(
    inner: T,
    settings: NutsSettings<()>,
) -> NutsSettings<T> {
    let adapt = AdaptOptions {
        dual_average_options: settings.adapt_options.dual_average_options,
        mass_matrix_options: inner,
        early_window: settings.adapt_options.early_window,
        step_size_window: settings.adapt_options.step_size_window,
        mass_matrix_switch_freq: settings.adapt_options.mass_matrix_switch_freq,
        early_mass_matrix_switch_freq: settings.adapt_options.early_mass_matrix_switch_freq,
        mass_matrix_update_freq: settings.adapt_options.mass_matrix_update_freq,
    };
    NutsSettings {
        num_tune: settings.num_tune,
        num_draws: settings.num_draws,
        maxdepth: settings.maxdepth,
        store_gradient: settings.store_gradient,
        store_unconstrained: settings.store_unconstrained,
        max_energy_error: settings.max_energy_error,
        store_divergences: settings.store_divergences,
        adapt_options: adapt,
        check_turning: settings.check_turning,
        num_chains: settings.num_chains,
        seed: settings.seed,
    }
}

fn split_settings<T: Debug + Copy + Default>(settings: NutsSettings<T>) -> (NutsSettings<()>, T) {
    let adapt_settings = settings.adapt_options;
    let mass_matrix_settings = adapt_settings.mass_matrix_options;

    let remaining: AdaptOptions<()> = AdaptOptions {
        dual_average_options: adapt_settings.dual_average_options,
        mass_matrix_options: (),
        early_window: adapt_settings.early_window,
        step_size_window: adapt_settings.step_size_window,
        mass_matrix_switch_freq: adapt_settings.mass_matrix_switch_freq,
        early_mass_matrix_switch_freq: adapt_settings.early_mass_matrix_switch_freq,
        mass_matrix_update_freq: adapt_settings.mass_matrix_update_freq,
    };

    let settings = NutsSettings {
        adapt_options: remaining,
        num_tune: settings.num_tune,
        num_draws: settings.num_draws,
        maxdepth: settings.maxdepth,
        store_gradient: settings.store_gradient,
        store_unconstrained: settings.store_unconstrained,
        max_energy_error: settings.max_energy_error,
        store_divergences: settings.store_divergences,
        check_turning: settings.check_turning,
        num_chains: settings.num_chains,
        seed: settings.seed,
    };

    (settings, mass_matrix_settings)
}

impl PyNutsSettings {
    fn into_settings(self) -> Settings {
        match self.adapt {
            InnerSettings::LowRank(mass_matrix) => {
                Settings::LowRank(combine_settings(mass_matrix, self.settings))
            }
            InnerSettings::Diag(mass_matrix) => {
                Settings::Diag(combine_settings(mass_matrix, self.settings))
            }
        }
    }

    fn new_diag(seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or_else(|| {
            let mut rng = thread_rng();
            rng.next_u64()
        });
        let settings = DiagGradNutsSettings {
            seed,
            ..Default::default()
        };

        let (settings, inner) = split_settings(settings);

        Self {
            settings,
            adapt: InnerSettings::Diag(inner),
        }
    }

    fn new_low_rank(seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or_else(|| {
            let mut rng = thread_rng();
            rng.next_u64()
        });
        let settings = LowRankNutsSettings {
            seed,
            ..Default::default()
        };

        let (settings, inner) = split_settings(settings);

        Self {
            settings,
            adapt: InnerSettings::LowRank(inner),
        }
    }
}

#[pymethods]
impl PyNutsSettings {
    #[staticmethod]
    #[allow(non_snake_case)]
    fn Diag(seed: Option<u64>) -> Self {
        PyNutsSettings::new_diag(seed)
    }

    #[staticmethod]
    #[allow(non_snake_case)]
    fn LowRank(seed: Option<u64>) -> Self {
        PyNutsSettings::new_low_rank(seed)
    }

    #[getter]
    fn num_tune(&self) -> u64 {
        self.settings.num_tune
    }

    #[setter(num_tune)]
    fn set_num_tune(&mut self, val: u64) {
        self.settings.num_tune = val
    }

    #[getter]
    fn num_chains(&self) -> usize {
        self.settings.num_chains
    }

    #[setter(num_chains)]
    fn set_num_chains(&mut self, val: usize) {
        self.settings.num_chains = val;
    }

    #[getter]
    fn num_draws(&self) -> u64 {
        self.settings.num_draws
    }

    #[setter(num_draws)]
    fn set_num_draws(&mut self, val: u64) {
        self.settings.num_draws = val;
    }

    #[getter]
    fn window_switch_freq(&self) -> u64 {
        self.settings.adapt_options.mass_matrix_switch_freq
    }

    #[setter(window_switch_freq)]
    fn set_window_switch_freq(&mut self, val: u64) {
        self.settings.adapt_options.mass_matrix_switch_freq = val;
    }

    #[getter]
    fn early_window_switch_freq(&self) -> u64 {
        self.settings.adapt_options.early_mass_matrix_switch_freq
    }

    #[setter(early_window_switch_freq)]
    fn set_early_window_switch_freq(&mut self, val: u64) {
        self.settings.adapt_options.early_mass_matrix_switch_freq = val;
    }
    #[getter]
    fn initial_step(&self) -> f64 {
        self.settings
            .adapt_options
            .dual_average_options
            .initial_step
    }

    #[setter(initial_step)]
    fn set_initial_step(&mut self, val: f64) {
        self.settings
            .adapt_options
            .dual_average_options
            .initial_step = val
    }

    #[getter]
    fn maxdepth(&self) -> u64 {
        self.settings.maxdepth
    }

    #[setter(maxdepth)]
    fn set_maxdepth(&mut self, val: u64) {
        self.settings.maxdepth = val
    }

    #[getter]
    fn store_gradient(&self) -> bool {
        self.settings.store_gradient
    }

    #[setter(store_gradient)]
    fn set_store_gradient(&mut self, val: bool) {
        self.settings.store_gradient = val;
    }

    #[getter]
    fn store_unconstrained(&self) -> bool {
        self.settings.store_unconstrained
    }

    #[setter(store_unconstrained)]
    fn set_store_unconstrained(&mut self, val: bool) {
        self.settings.store_unconstrained = val;
    }

    #[getter]
    fn store_divergences(&self) -> bool {
        self.settings.store_divergences
    }

    #[setter(store_divergences)]
    fn set_store_divergences(&mut self, val: bool) {
        self.settings.store_divergences = val;
    }

    #[getter]
    fn max_energy_error(&self) -> f64 {
        self.settings.max_energy_error
    }

    #[setter(max_energy_error)]
    fn set_max_energy_error(&mut self, val: f64) {
        self.settings.max_energy_error = val
    }

    #[setter(target_accept)]
    fn set_target_accept(&mut self, val: f64) {
        self.settings
            .adapt_options
            .dual_average_options
            .target_accept = val;
    }

    #[getter]
    fn target_accept(&self) -> f64 {
        self.settings
            .adapt_options
            .dual_average_options
            .target_accept
    }

    #[getter]
    fn store_mass_matrix(&self) -> bool {
        match &self.adapt {
            InnerSettings::LowRank(low_rank) => low_rank.store_mass_matrix,
            InnerSettings::Diag(diag) => diag.store_mass_matrix,
        }
    }

    #[setter(store_mass_matrix)]
    fn set_store_mass_matrix(&mut self, val: bool) {
        match &mut self.adapt {
            InnerSettings::LowRank(low_rank) => {
                low_rank.store_mass_matrix = val;
            }
            InnerSettings::Diag(diag) => {
                diag.store_mass_matrix = val;
            }
        }
    }

    #[getter]
    fn use_grad_based_mass_matrix(&self) -> Result<bool> {
        match &self.adapt {
            InnerSettings::LowRank(_) => {
                bail!("grad based mass matrix not available for low-rank adaptation")
            }
            InnerSettings::Diag(diag) => Ok(diag.use_grad_based_estimate),
        }
    }

    #[setter(use_grad_based_mass_matrix)]
    fn set_use_grad_based_mass_matrix(&mut self, val: bool) -> Result<()> {
        match &mut self.adapt {
            InnerSettings::LowRank(_) => {
                bail!("grad based mass matrix not available for low-rank adaptation")
            }
            InnerSettings::Diag(diag) => {
                diag.use_grad_based_estimate = val;
            }
        }
        Ok(())
    }

    #[getter]
    fn mass_matrix_switch_freq(&self) -> u64 {
        self.settings.adapt_options.mass_matrix_switch_freq
    }

    #[setter(mass_matrix_switch_freq)]
    fn set_mass_matrix_switch_freq(&mut self, val: u64) {
        self.settings.adapt_options.mass_matrix_switch_freq = val;
    }

    #[getter]
    fn mass_matrix_eigval_cutoff(&self) -> Result<f64> {
        match &self.adapt {
            InnerSettings::LowRank(inner) => Ok(inner.eigval_cutoff),
            InnerSettings::Diag(_) => {
                bail!("eigenvalue cutoff not available for diag mass matrix adaptation")
            }
        }
    }

    #[setter(mass_matrix_eigval_cutoff)]
    fn set_mass_matrix_eigval_cutoff(&mut self, val: f64) -> Result<()> {
        match &mut self.adapt {
            InnerSettings::LowRank(inner) => inner.eigval_cutoff = val,
            InnerSettings::Diag(_) => {
                bail!("eigenvalue cutoff not available for diag mass matrix adaptation")
            }
        }
        Ok(())
    }

    #[getter]
    fn mass_matrix_gamma(&self) -> Result<f64> {
        match &self.adapt {
            InnerSettings::LowRank(inner) => Ok(inner.gamma),
            InnerSettings::Diag(_) => {
                bail!("gamma not available for diag mass matrix adaptation")
            }
        }
    }

    #[setter(mass_matrix_gamma)]
    fn set_mass_matrix_gamma(&mut self, val: f64) -> Result<()> {
        match &mut self.adapt {
            InnerSettings::LowRank(inner) => inner.gamma = val,
            InnerSettings::Diag(_) => {
                bail!("gamma not available for diag mass matrix adaptation")
            }
        }
        Ok(())
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
        settings: PyNutsSettings,
        cores: usize,
        model: PyMcModel,
        progress_type: ProgressType,
    ) -> PyResult<PySampler> {
        let callback = progress_type.into_callback()?;
        match settings.into_settings() {
            Settings::LowRank(settings) => {
                let sampler = Sampler::new(model, settings, cores, callback)?;
                Ok(PySampler(SamplerState::Running(sampler)))
            }
            Settings::Diag(settings) => {
                let sampler = Sampler::new(model, settings, cores, callback)?;
                Ok(PySampler(SamplerState::Running(sampler)))
            }
        }
    }

    #[staticmethod]
    fn from_stan(
        settings: PyNutsSettings,
        cores: usize,
        model: StanModel,
        progress_type: ProgressType,
    ) -> PyResult<PySampler> {
        let callback = progress_type.into_callback()?;
        match settings.into_settings() {
            Settings::LowRank(settings) => {
                let sampler = Sampler::new(model, settings, cores, callback)?;
                Ok(PySampler(SamplerState::Running(sampler)))
            }
            Settings::Diag(settings) => {
                let sampler = Sampler::new(model, settings, cores, callback)?;
                Ok(PySampler(SamplerState::Running(sampler)))
            }
        }
    }

    #[staticmethod]
    fn from_pyfunc(
        settings: PyNutsSettings,
        cores: usize,
        model: PyModel,
        progress_type: ProgressType,
    ) -> PyResult<PySampler> {
        let callback = progress_type.into_callback()?;
        match settings.into_settings() {
            Settings::LowRank(settings) => {
                let sampler = Sampler::new(model, settings, cores, callback)?;
                Ok(PySampler(SamplerState::Running(sampler)))
            }
            Settings::Diag(settings) => {
                let sampler = Sampler::new(model, settings, cores, callback)?;
                Ok(PySampler(SamplerState::Running(sampler)))
            }
        }
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
    m.add_class::<PyNutsSettings>()?;
    m.add_class::<PyChainProgress>()?;
    m.add_class::<ProgressType>()?;
    m.add_class::<TensorShape>()?;
    m.add_class::<PyModel>()?;
    m.add_class::<PyVariable>()?;
    m.add_class::<ExpandDtype>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
