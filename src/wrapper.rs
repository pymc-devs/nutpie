use std::{
    fmt::Debug,
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use crate::{
    common::PyVariable,
    progress::{IndicatifHandler, ProgressHandler, RawCallbackHandler},
    pyfunc::PyModel,
    pymc::{ExpandFunc, LogpFunc, PyMcModel},
    stan::{StanLibrary, StanModel},
};

use anyhow::{anyhow, bail, Context, Result};
use numpy::{PyArray1, PyReadonlyArray1};
use nuts_rs::{
    ArrowConfig, ArrowTrace, ChainProgress, DiagMclmcSettings, DiagNutsSettings, FlowMclmcSettings,
    FlowNutsSettings, KineticEnergyKind, LowRankMclmcSettings, LowRankNutsSettings, Model,
    ProgressCallback, Sampler, SamplerWaitResult, StepSizeAdaptMethod, ZarrAsyncConfig,
};
use pyo3::{
    exceptions::{PyAttributeError, PyTimeoutError, PyValueError},
    intern,
    prelude::*,
    types::{PyDict, PyList},
};
use pyo3_arrow::PyRecordBatch;
use pyo3_object_store::AnyObjectStore;
use pythonize::{depythonize, pythonize};
use rand::{rng, Rng};
use serde_json::Value as JsonValue;
use tokio::runtime::Runtime;
use zarrs_object_store::{object_store::limit::LimitStore, AsyncObjectStore};

#[pyclass]
pub struct PyChainProgress(ChainProgress);

impl PyChainProgress {
    pub(crate) fn new(progress: ChainProgress) -> Self {
        Self(progress)
    }
}

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
    fn latest_num_steps(&self) -> usize {
        self.0.latest_num_steps
    }

    // Keep the old name as an alias for backward compatibility
    #[getter]
    fn num_steps(&self) -> usize {
        self.0.latest_num_steps
    }

    #[getter]
    fn total_num_steps(&self) -> usize {
        self.0.total_num_steps
    }

    #[getter]
    fn step_size(&self) -> f64 {
        self.0.step_size
    }

    #[getter]
    fn runtime_ms(&self) -> u64 {
        self.0.runtime.as_millis() as u64
    }

    #[getter]
    fn divergent_draws(&self) -> Vec<usize> {
        self.0.divergent_draws.clone()
    }
}

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct PyNutsSettings {
    inner: NutsSettingsKind,
}

#[derive(Clone, FromPyObject)]
enum PySamplerSettings {
    Nuts(PyNutsSettings),
    Mclmc(PyMclmcSettings),
}

#[derive(Clone, Debug)]
enum NutsSettingsKind {
    Diag(DiagNutsSettings),
    LowRank(LowRankNutsSettings),
    Flow(FlowNutsSettings),
}

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct PyMclmcSettings {
    inner: MclmcSettingsKind,
}

#[derive(Clone, Debug)]
enum MclmcSettingsKind {
    Diag(DiagMclmcSettings),
    LowRank(LowRankMclmcSettings),
    Flow(FlowMclmcSettings),
}

macro_rules! unsupported_option_error {
    ($option:expr, $adaptation:expr) => {
        PyValueError::new_err(format!(
            "Option {} not available for {} adaptation",
            $option, $adaptation
        ))
    };
}

macro_rules! with_all_settings_mut {
    ($self:expr, $enum_name:ident, $settings:ident => $body:block) => {{
        match &mut $self.inner {
            $enum_name::Diag($settings) => $body,
            $enum_name::LowRank($settings) => $body,
            $enum_name::Flow($settings) => $body,
        }
    }};
}

macro_rules! set_all_settings_field {
    ($self:expr, $enum_name:ident, $field:ident = $value:expr) => {{
        with_all_settings_mut!($self, $enum_name, settings => {
            settings.$field = $value;
        });
    }};
    ($self:expr, $enum_name:ident, $field:ident $(. $rest:ident)+ = $value:expr) => {{
        with_all_settings_mut!($self, $enum_name, settings => {
            settings.$field$(.$rest)+ = $value;
        });
    }};
}

macro_rules! with_diag_or_low_rank_settings_mut {
    ($self:expr, $enum_name:ident, $option:expr, $settings:ident => $body:block) => {{
        match &mut $self.inner {
            $enum_name::Diag($settings) => $body,
            $enum_name::LowRank($settings) => $body,
            $enum_name::Flow(_) => return Err(unsupported_option_error!($option, "flow")),
        }
    }};
}

macro_rules! with_diag_settings_mut {
    ($self:expr, $enum_name:ident, $option:expr, $settings:ident => $body:block) => {{
        match &mut $self.inner {
            $enum_name::Diag($settings) => $body,
            $enum_name::LowRank(_) => return Err(unsupported_option_error!($option, "low-rank")),
            $enum_name::Flow(_) => return Err(unsupported_option_error!($option, "flow")),
        }
    }};
}

macro_rules! with_low_rank_settings_mut {
    ($self:expr, $enum_name:ident, $option:expr, $settings:ident => $body:block) => {{
        match &mut $self.inner {
            $enum_name::LowRank($settings) => $body,
            $enum_name::Diag(_) => return Err(unsupported_option_error!($option, "diag")),
            $enum_name::Flow(_) => return Err(unsupported_option_error!($option, "flow")),
        }
    }};
}

macro_rules! with_flow_settings_mut {
    ($self:expr, $enum_name:ident, $option:expr, $settings:ident => $body:block) => {{
        match &mut $self.inner {
            $enum_name::Flow($settings) => $body,
            $enum_name::Diag(_) => return Err(unsupported_option_error!($option, "diag")),
            $enum_name::LowRank(_) => return Err(unsupported_option_error!($option, "low-rank")),
        }
    }};
}

macro_rules! try_shared_euclidean_adapt_update {
    ($self:expr, $enum_name:ident, $name:expr, $value:expr) => {{
        match $name {
            "window_switch_freq" => {
                let value: u64 = $value.extract()?;
                match &mut $self.inner {
                    $enum_name::Diag(settings) => {
                        settings.adapt_options.mass_matrix_switch_freq = value
                    }
                    $enum_name::LowRank(settings) => {
                        settings.adapt_options.mass_matrix_switch_freq = value
                    }
                    $enum_name::Flow(settings) => {
                        settings.adapt_options.transform_update_freq = value
                    }
                }
                true
            }
            "early_window_switch_freq" => {
                let value: u64 = $value.extract()?;
                with_diag_or_low_rank_settings_mut!(
                    $self,
                    $enum_name,
                    "early_window_switch_freq",
                    settings => {
                        settings.adapt_options.early_mass_matrix_switch_freq = value;
                    }
                );
                true
            }
            "initial_step" => {
                let value: f64 = $value.extract()?;
                set_all_settings_field!(
                    $self,
                    $enum_name,
                    adapt_options.step_size_settings.initial_step = value
                );
                true
            }
            "target_accept" => {
                let value: f64 = $value.extract()?;
                set_all_settings_field!(
                    $self,
                    $enum_name,
                    adapt_options.step_size_settings.target_accept = value
                );
                true
            }
            "max_step_size" => {
                let value: f64 = $value.extract()?;
                set_all_settings_field!(
                    $self,
                    $enum_name,
                    adapt_options
                        .step_size_settings
                        .adapt_options
                        .dual_average
                        .max_step_size = value
                );
                true
            }
            "store_mass_matrix" => {
                let value: bool = $value.extract()?;
                with_diag_or_low_rank_settings_mut!(
                    $self,
                    $enum_name,
                    "store_mass_matrix",
                    settings => {
                        settings.adapt_options.mass_matrix_options.store_mass_matrix = value;
                    }
                );
                true
            }
            "use_grad_based_mass_matrix" => {
                let value: bool = $value.extract()?;
                with_diag_settings_mut!(
                    $self,
                    $enum_name,
                    "use_grad_based_mass_matrix",
                    settings => {
                        settings.adapt_options.mass_matrix_options.use_grad_based_estimate = value;
                    }
                );
                true
            }
            "mass_matrix_switch_freq" => {
                let value: u64 = $value.extract()?;
                with_diag_or_low_rank_settings_mut!(
                    $self,
                    $enum_name,
                    "mass_matrix_switch_freq",
                    settings => {
                        settings.adapt_options.mass_matrix_switch_freq = value;
                    }
                );
                true
            }
            "mass_matrix_eigval_cutoff" => {
                let value: Option<f64> = $value.extract()?;
                if let Some(value) = value {
                    with_low_rank_settings_mut!(
                        $self,
                        $enum_name,
                        "mass_matrix_eigval_cutoff",
                        settings => {
                            settings.adapt_options.mass_matrix_options.eigval_cutoff = value;
                        }
                    );
                }
                true
            }
            "mass_matrix_gamma" => {
                let value: Option<f64> = $value.extract()?;
                if let Some(value) = value {
                    with_low_rank_settings_mut!(
                        $self,
                        $enum_name,
                        "mass_matrix_gamma",
                        settings => {
                            settings.adapt_options.mass_matrix_options.gamma = value;
                        }
                    );
                }
                true
            }
            "train_on_orbit" => {
                let value: bool = $value.extract()?;
                with_flow_settings_mut!(
                    $self,
                    $enum_name,
                    "train_on_orbit",
                    settings => {
                        settings.adapt_options.use_orbit_for_training = value;
                    }
                );
                true
            }
            "step_size_adapt_method" => {
                let method = match $value.extract::<String>() {
                    Ok(method) => match method.as_str() {
                        "dual_average" => StepSizeAdaptMethod::DualAverage,
                        "adam" => StepSizeAdaptMethod::Adam,
                        _ => {
                            if let Ok(step_size) = method.parse::<f64>() {
                                StepSizeAdaptMethod::Fixed(step_size)
                            } else {
                                return Err(PyValueError::new_err(
                                    "step_size_adapt_method must be a positive float when using fixed step size",
                                ));
                            }
                        }
                    },
                    _ => {
                        return Err(PyValueError::new_err(
                            "step_size_adapt_method must be a string",
                        ));
                    }
                };

                set_all_settings_field!(
                    $self,
                    $enum_name,
                    adapt_options.step_size_settings.adapt_options.method = method
                );
                true
            }
            "step_size_adam_learning_rate" => {
                let value: Option<f64> = $value.extract()?;
                if let Some(value) = value {
                    set_all_settings_field!(
                        $self,
                        $enum_name,
                        adapt_options
                            .step_size_settings
                            .adapt_options
                            .adam
                            .learning_rate = value
                    );
                }
                true
            }
            "step_size_jitter" => {
                let mut value: Option<f64> = $value.extract()?;
                if let Some(jitter) = value {
                    if jitter < 0.0 {
                        return Err(PyValueError::new_err("step_size_jitter must be positive"));
                    }
                    if jitter == 0.0 {
                        value = None;
                    }
                }
                set_all_settings_field!(
                    $self,
                    $enum_name,
                    adapt_options.step_size_settings.jitter = value
                );
                true
            }
            "store_unconstrained" => {
                let value: bool = $value.extract()?;
                set_all_settings_field!($self, $enum_name, store_unconstrained = value);
                true
            }
            "store_gradient" => {
                let value: bool = $value.extract()?;
                set_all_settings_field!($self, $enum_name, store_gradient = value);
                true
            }
            "num_tune" => {
                let value: u64 = $value.extract()?;
                set_all_settings_field!($self, $enum_name, num_tune = value);
                true
            }
            "num_chains" => {
                let value: usize = $value.extract()?;
                set_all_settings_field!($self, $enum_name, num_chains = value);
                true
            }
            "num_draws" => {
                let value: u64 = $value.extract()?;
                set_all_settings_field!($self, $enum_name, num_draws = value);
                true
            }
            "store_transformed" => {
                let value: bool = $value.extract()?;
                set_all_settings_field!($self, $enum_name, store_transformed = value);
                true
            }
            "store_divergences" => {
                let value: bool = $value.extract()?;
                set_all_settings_field!($self, $enum_name, store_divergences = value);
                true
            }
            "max_energy_error" => {
                let value: f64 = $value.extract()?;
                set_all_settings_field!($self, $enum_name, max_energy_error = value);
                true
            }
            _ => false,
        }
    }};
}

fn random_seed(seed: Option<u64>) -> u64 {
    seed.unwrap_or_else(|| {
        let mut rng = rng();
        rng.next_u64()
    })
}

fn update_nuts_from_nested_dict(
    inner: &mut NutsSettingsKind,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match inner {
        NutsSettingsKind::Diag(settings) => {
            *settings = depythonize(value).map_err(|err| PyValueError::new_err(err.to_string()))?;
        }
        NutsSettingsKind::LowRank(settings) => {
            *settings = depythonize(value).map_err(|err| PyValueError::new_err(err.to_string()))?;
        }
        NutsSettingsKind::Flow(settings) => {
            *settings = depythonize(value).map_err(|err| PyValueError::new_err(err.to_string()))?;
        }
    }
    Ok(())
}

fn update_mclmc_from_nested_dict(
    inner: &mut MclmcSettingsKind,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match inner {
        MclmcSettingsKind::Diag(settings) => {
            *settings = depythonize(value).map_err(|err| PyValueError::new_err(err.to_string()))?;
        }
        MclmcSettingsKind::LowRank(settings) => {
            *settings = depythonize(value).map_err(|err| PyValueError::new_err(err.to_string()))?;
        }
        MclmcSettingsKind::Flow(settings) => {
            *settings = depythonize(value).map_err(|err| PyValueError::new_err(err.to_string()))?;
        }
    }
    Ok(())
}

fn nuts_to_nested_json(inner: &NutsSettingsKind) -> PyResult<JsonValue> {
    match inner {
        NutsSettingsKind::Diag(settings) => {
            serde_json::to_value(settings).map_err(|err| PyValueError::new_err(err.to_string()))
        }
        NutsSettingsKind::LowRank(settings) => {
            serde_json::to_value(settings).map_err(|err| PyValueError::new_err(err.to_string()))
        }
        NutsSettingsKind::Flow(settings) => {
            serde_json::to_value(settings).map_err(|err| PyValueError::new_err(err.to_string()))
        }
    }
}

fn mclmc_to_nested_json(inner: &MclmcSettingsKind) -> PyResult<JsonValue> {
    match inner {
        MclmcSettingsKind::Diag(settings) => {
            serde_json::to_value(settings).map_err(|err| PyValueError::new_err(err.to_string()))
        }
        MclmcSettingsKind::LowRank(settings) => {
            serde_json::to_value(settings).map_err(|err| PyValueError::new_err(err.to_string()))
        }
        MclmcSettingsKind::Flow(settings) => {
            serde_json::to_value(settings).map_err(|err| PyValueError::new_err(err.to_string()))
        }
    }
}

impl PyNutsSettings {
    fn new_diag(seed: Option<u64>) -> Self {
        let settings = DiagNutsSettings {
            seed: random_seed(seed),
            ..Default::default()
        };
        Self {
            inner: NutsSettingsKind::Diag(settings),
        }
    }

    fn new_low_rank(seed: Option<u64>) -> Self {
        let settings = LowRankNutsSettings {
            seed: random_seed(seed),
            ..Default::default()
        };
        Self {
            inner: NutsSettingsKind::LowRank(settings),
        }
    }

    fn new_flow(seed: Option<u64>) -> Self {
        let settings = FlowNutsSettings {
            seed: random_seed(seed),
            ..Default::default()
        };
        Self {
            inner: NutsSettingsKind::Flow(settings),
        }
    }

    fn update_from_nested_dict(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        update_nuts_from_nested_dict(&mut self.inner, value)
    }

    fn to_nested_json(&self) -> PyResult<JsonValue> {
        nuts_to_nested_json(&self.inner)
    }

    fn apply_update(&mut self, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        match name {
            "maxdepth" => {
                let value: u64 = value.extract()?;
                set_all_settings_field!(self, NutsSettingsKind, maxdepth = value);
            }
            "mindepth" => {
                let value: u64 = value.extract()?;
                set_all_settings_field!(self, NutsSettingsKind, mindepth = value);
            }
            "check_turning" => {
                let value: bool = value.extract()?;
                set_all_settings_field!(self, NutsSettingsKind, check_turning = value);
            }
            "target_integration_time" => {
                let value: Option<f64> = value.extract()?;
                set_all_settings_field!(self, NutsSettingsKind, target_integration_time = value);
            }
            "extra_doublings" => {
                let value: u64 = value.extract()?;
                set_all_settings_field!(self, NutsSettingsKind, extra_doublings = value);
            }
            _ => {
                if try_shared_euclidean_adapt_update!(self, NutsSettingsKind, name, value) {
                    // handled above
                } else {
                    match name {
                        "microcanonical_trajectory" => {
                            let value: bool = value.extract()?;
                            if value {
                                set_all_settings_field!(
                                    self,
                                    NutsSettingsKind,
                                    trajectory_kind = KineticEnergyKind::Microcanonical
                                );
                            }
                        }
                        "exact_normal_trajectory" => {
                            let value: bool = value.extract()?;
                            if value {
                                set_all_settings_field!(
                                    self,
                                    NutsSettingsKind,
                                    trajectory_kind = KineticEnergyKind::ExactNormal
                                );
                            }
                        }
                        _ => {
                            return Err(PyAttributeError::new_err(format!(
                                "Unknown settings attribute: {name}",
                            )));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl PyMclmcSettings {
    fn new_diag(seed: Option<u64>) -> Self {
        let settings = DiagMclmcSettings {
            seed: random_seed(seed),
            ..Default::default()
        };
        Self {
            inner: MclmcSettingsKind::Diag(settings),
        }
    }

    fn new_low_rank(seed: Option<u64>) -> Self {
        let settings = LowRankMclmcSettings {
            seed: random_seed(seed),
            ..Default::default()
        };
        Self {
            inner: MclmcSettingsKind::LowRank(settings),
        }
    }

    fn new_flow(seed: Option<u64>) -> Self {
        let settings = FlowMclmcSettings {
            seed: random_seed(seed),
            ..Default::default()
        };
        Self {
            inner: MclmcSettingsKind::Flow(settings),
        }
    }

    fn update_from_nested_dict(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        update_mclmc_from_nested_dict(&mut self.inner, value)
    }

    fn to_nested_json(&self) -> PyResult<JsonValue> {
        mclmc_to_nested_json(&self.inner)
    }

    fn apply_update(&mut self, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        match name {
            "step_size" => {
                let value: f64 = value.extract()?;
                set_all_settings_field!(self, MclmcSettingsKind, step_size = value);
            }
            "momentum_decoherence_length" => {
                let value: f64 = value.extract()?;
                set_all_settings_field!(
                    self,
                    MclmcSettingsKind,
                    momentum_decoherence_length = value
                );
            }
            "subsample_frequency" => {
                let value: f64 = value.extract()?;
                set_all_settings_field!(self, MclmcSettingsKind, subsample_frequency = value);
            }
            "dynamic_step_size" => {
                let value: bool = value.extract()?;
                set_all_settings_field!(self, MclmcSettingsKind, dynamic_step_size = value);
            }
            _ => {
                if try_shared_euclidean_adapt_update!(self, MclmcSettingsKind, name, value) {
                    // handled above
                } else {
                    return Err(PyAttributeError::new_err(format!(
                        "Unknown settings attribute: {name}",
                    )));
                }
            }
        }
        Ok(())
    }
}

// TODO switch to serde to expose all the options...
#[pymethods]
impl PyNutsSettings {
    #[staticmethod]
    #[allow(non_snake_case)]
    #[pyo3(signature = (seed=None))]
    fn Diag(seed: Option<u64>) -> Self {
        PyNutsSettings::new_diag(seed)
    }

    #[staticmethod]
    #[allow(non_snake_case)]
    #[pyo3(signature = (seed=None))]
    fn LowRank(seed: Option<u64>) -> Self {
        PyNutsSettings::new_low_rank(seed)
    }

    #[staticmethod]
    #[allow(non_snake_case)]
    #[pyo3(signature = (seed=None))]
    fn Flow(seed: Option<u64>) -> Self {
        PyNutsSettings::new_flow(seed)
    }

    fn update(&mut self, kwargs: &Bound<'_, PyDict>) -> PyResult<()> {
        for (key, value) in kwargs.iter() {
            let key: String = key.extract()?;
            self.apply_update(&key, &value)?;
        }
        Ok(())
    }

    fn __setattr__(&mut self, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.apply_update(name, value)
    }

    fn update_settings(&mut self, settings: &Bound<'_, PyDict>) -> PyResult<()> {
        self.update_from_nested_dict(settings.as_any())
    }

    fn as_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let settings = self.to_nested_json()?;
        let adaptation = match self.inner {
            NutsSettingsKind::Diag(_) => "diag",
            NutsSettingsKind::LowRank(_) => "low_rank",
            NutsSettingsKind::Flow(_) => "flow",
        };
        let value = serde_json::json!({
            "sampler": "nuts",
            "adaptation": adaptation,
            "settings": settings,
        });
        let obj = pythonize(py, &value).map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(obj.unbind())
    }
}

#[pymethods]
impl PyMclmcSettings {
    #[staticmethod]
    #[allow(non_snake_case)]
    #[pyo3(signature = (seed=None))]
    fn Diag(seed: Option<u64>) -> Self {
        PyMclmcSettings::new_diag(seed)
    }

    #[staticmethod]
    #[allow(non_snake_case)]
    #[pyo3(signature = (seed=None))]
    fn LowRank(seed: Option<u64>) -> Self {
        PyMclmcSettings::new_low_rank(seed)
    }

    #[staticmethod]
    #[allow(non_snake_case)]
    #[pyo3(signature = (seed=None))]
    fn Flow(seed: Option<u64>) -> Self {
        PyMclmcSettings::new_flow(seed)
    }

    fn update(&mut self, kwargs: &Bound<'_, PyDict>) -> PyResult<()> {
        for (key, value) in kwargs.iter() {
            let key: String = key.extract()?;
            self.apply_update(&key, &value)?;
        }
        Ok(())
    }

    fn __setattr__(&mut self, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.apply_update(name, value)
    }

    fn update_settings(&mut self, settings: &Bound<'_, PyDict>) -> PyResult<()> {
        self.update_from_nested_dict(settings.as_any())
    }

    fn as_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let settings = self.to_nested_json()?;
        let adaptation = match self.inner {
            MclmcSettingsKind::Diag(_) => "diag",
            MclmcSettingsKind::LowRank(_) => "low_rank",
            MclmcSettingsKind::Flow(_) => "flow",
        };
        let value = serde_json::json!({
            "sampler": "mclmc",
            "adaptation": adaptation,
            "settings": settings,
        });
        let obj = pythonize(py, &value).map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(obj.unbind())
    }
}

pub(crate) enum SamplerState {
    RunningZarr(Sampler<()>),
    RunningArrow(Sampler<Vec<ArrowTrace>>),
    FinishedZarr,
    FinishedArrow(Vec<ArrowTrace>),
    Empty,
}

#[derive(Clone)]
enum InnerProgressType {
    Callback {
        rate: Duration,
        n_cores: usize,
        template: String,
        callback: Arc<Py<PyAny>>,
    },
    Indicatif {
        rate: Duration,
    },
    None {},
}

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct ProgressType(InnerProgressType);

impl ProgressType {
    fn into_callback(
        self,
        extra_callback: Option<Arc<Py<PyAny>>>,
        extra_rate: Duration,
    ) -> Result<Option<ProgressCallback>> {
        // Build the primary callback (if any).
        let primary: Option<ProgressCallback> = match self.0 {
            InnerProgressType::Callback {
                callback,
                rate,
                n_cores,
                template,
            } => {
                let handler = ProgressHandler::new(callback, rate, template, n_cores);
                Some(handler.into_callback()?)
            }
            InnerProgressType::Indicatif { rate } => {
                let handler = IndicatifHandler::new(rate);
                Some(handler.into_callback()?)
            }
            InnerProgressType::None {} => None,
        };

        // Build the optional user-supplied raw callback.
        let raw: Option<ProgressCallback> = match extra_callback {
            Some(cb) => {
                let handler = RawCallbackHandler::new(cb, extra_rate);
                Some(handler.into_callback()?)
            }
            None => None,
        };

        // Combine them.
        match (primary, raw) {
            (None, None) => Ok(None),
            (Some(p), None) => Ok(Some(p)),
            (None, Some(r)) => Ok(Some(r)),
            (Some(mut p), Some(mut r)) => {
                let rate = p.rate.min(r.rate);
                let combined = move |time: Duration, progress: Box<[ChainProgress]>| {
                    (p.callback)(time, progress.clone());
                    (r.callback)(time, progress);
                };
                Ok(Some(ProgressCallback {
                    callback: Box::new(combined),
                    rate,
                }))
            }
        }
    }
}

#[pymethods]
impl ProgressType {
    #[staticmethod]
    fn indicatif(rate: u64) -> Self {
        let rate = Duration::from_millis(rate);
        ProgressType(InnerProgressType::Indicatif { rate })
    }

    #[staticmethod]
    fn none() -> Self {
        ProgressType(InnerProgressType::None {})
    }

    #[staticmethod]
    fn template_callback(rate: u64, template: String, n_cores: usize, callback: Py<PyAny>) -> Self {
        let rate = Duration::from_millis(rate);
        ProgressType(InnerProgressType::Callback {
            callback: Arc::new(callback),
            template,
            n_cores,
            rate,
        })
    }
}

enum InnerPyStorage {
    Zarr(Option<AnyObjectStore>),
    Arrow,
}

#[pyclass]
struct PyStorage(InnerPyStorage);

#[pymethods]
impl PyStorage {
    #[staticmethod]
    fn zarr(object_store: AnyObjectStore) -> Self {
        Self(InnerPyStorage::Zarr(Some(object_store)))
    }

    #[staticmethod]
    fn arrow() -> Self {
        Self(InnerPyStorage::Arrow)
    }
}

#[pyclass]
struct PySampler(Mutex<(SamplerState, Runtime)>);

impl PySampler {
    fn new<M: Model>(
        settings: PySamplerSettings,
        cores: usize,
        model: M,
        progress_type: ProgressType,
        extra_callback: Option<Py<PyAny>>,
        extra_callback_rate: u64,
        store: &mut PyStorage,
    ) -> PyResult<Self> {
        let extra_callback = extra_callback.map(Arc::new);
        let extra_rate = Duration::from_millis(extra_callback_rate);
        let callback = progress_type.into_callback(extra_callback, extra_rate)?;
        let tokio_rt = Runtime::new().context("Failed to create Tokio runtime")?;
        match &mut store.0 {
            InnerPyStorage::Arrow => {
                let storage_config = ArrowConfig::new();
                match settings {
                    PySamplerSettings::Nuts(settings) => match settings.inner {
                        NutsSettingsKind::LowRank(settings) => {
                            let sampler =
                                Sampler::new(model, settings, storage_config, cores, callback)?;
                            Ok(PySampler(Mutex::new((
                                SamplerState::RunningArrow(sampler).into(),
                                tokio_rt,
                            ))))
                        }
                        NutsSettingsKind::Diag(settings) => {
                            let sampler =
                                Sampler::new(model, settings, storage_config, cores, callback)?;
                            Ok(PySampler(Mutex::new((
                                SamplerState::RunningArrow(sampler).into(),
                                tokio_rt,
                            ))))
                        }
                        NutsSettingsKind::Flow(settings) => {
                            let sampler =
                                Sampler::new(model, settings, storage_config, cores, callback)?;
                            Ok(PySampler(Mutex::new((
                                SamplerState::RunningArrow(sampler).into(),
                                tokio_rt,
                            ))))
                        }
                    },
                    PySamplerSettings::Mclmc(settings) => match settings.inner {
                        MclmcSettingsKind::LowRank(settings) => {
                            let sampler =
                                Sampler::new(model, settings, storage_config, cores, callback)?;
                            Ok(PySampler(Mutex::new((
                                SamplerState::RunningArrow(sampler).into(),
                                tokio_rt,
                            ))))
                        }
                        MclmcSettingsKind::Diag(settings) => {
                            let sampler =
                                Sampler::new(model, settings, storage_config, cores, callback)?;
                            Ok(PySampler(Mutex::new((
                                SamplerState::RunningArrow(sampler).into(),
                                tokio_rt,
                            ))))
                        }
                        MclmcSettingsKind::Flow(settings) => {
                            let sampler =
                                Sampler::new(model, settings, storage_config, cores, callback)?;
                            Ok(PySampler(Mutex::new((
                                SamplerState::RunningArrow(sampler).into(),
                                tokio_rt,
                            ))))
                        }
                    },
                }
            }
            InnerPyStorage::Zarr(store) => {
                zarrs::config::global_config_mut().set_include_zarrs_metadata(false);
                let object_store = store
                    .take()
                    .ok_or_else(|| anyhow!("Can not use storage configuration twice"))?
                    .into_dyn();
                let object_store = LimitStore::new(object_store, 8);
                let store = AsyncObjectStore::new(object_store);
                let store = Arc::new(store);
                let storage_config = ZarrAsyncConfig::new(tokio_rt.handle().clone(), store);
                let storage_config = storage_config.with_chunk_size(16);
                match settings {
                    PySamplerSettings::Nuts(settings) => match settings.inner {
                        NutsSettingsKind::LowRank(settings) => {
                            let sampler =
                                Sampler::new(model, settings, storage_config, cores, callback)?;
                            Ok(PySampler(Mutex::new((
                                SamplerState::RunningZarr(sampler).into(),
                                tokio_rt,
                            ))))
                        }
                        NutsSettingsKind::Diag(settings) => {
                            let sampler =
                                Sampler::new(model, settings, storage_config, cores, callback)?;
                            Ok(PySampler(Mutex::new((
                                SamplerState::RunningZarr(sampler).into(),
                                tokio_rt,
                            ))))
                        }
                        NutsSettingsKind::Flow(settings) => {
                            let sampler =
                                Sampler::new(model, settings, storage_config, cores, callback)?;
                            Ok(PySampler(Mutex::new((
                                SamplerState::RunningZarr(sampler).into(),
                                tokio_rt,
                            ))))
                        }
                    },
                    PySamplerSettings::Mclmc(settings) => match settings.inner {
                        MclmcSettingsKind::LowRank(settings) => {
                            let sampler =
                                Sampler::new(model, settings, storage_config, cores, callback)?;
                            Ok(PySampler(Mutex::new((
                                SamplerState::RunningZarr(sampler).into(),
                                tokio_rt,
                            ))))
                        }
                        MclmcSettingsKind::Diag(settings) => {
                            let sampler =
                                Sampler::new(model, settings, storage_config, cores, callback)?;
                            Ok(PySampler(Mutex::new((
                                SamplerState::RunningZarr(sampler).into(),
                                tokio_rt,
                            ))))
                        }
                        MclmcSettingsKind::Flow(settings) => {
                            let sampler =
                                Sampler::new(model, settings, storage_config, cores, callback)?;
                            Ok(PySampler(Mutex::new((
                                SamplerState::RunningZarr(sampler).into(),
                                tokio_rt,
                            ))))
                        }
                    },
                }
            }
        }
    }
}

impl PySampler {
    fn wait_inner_arrow(
        &self,
        mut control: Sampler<Vec<ArrowTrace>>,
        timeout: Option<Duration>,
    ) -> (PyResult<()>, SamplerState) {
        let start_time = Instant::now();
        let step = Duration::from_millis(100);

        loop {
            let time_so_far = Instant::now().saturating_duration_since(start_time);
            let next_timeout = match timeout {
                Some(timeout) => {
                    let Some(remaining) = timeout.checked_sub(time_so_far) else {
                        return (
                            Err(PyTimeoutError::new_err(
                                "Timeout while waiting for sampler to finish",
                            )),
                            SamplerState::RunningArrow(control),
                        );
                    };
                    remaining.min(step)
                }
                None => step,
            };

            match control.wait_timeout(next_timeout) {
                SamplerWaitResult::Trace(trace) => {
                    return (Ok(()), SamplerState::FinishedArrow(trace))
                }
                SamplerWaitResult::Timeout(new_control) => {
                    control = new_control;
                }
                SamplerWaitResult::Err(err, trace) => {
                    return (
                        Err(err.into()),
                        SamplerState::FinishedArrow(trace.unwrap_or_default()),
                    )
                }
            }

            if let Err(err) = Python::attach(|py| py.check_signals()) {
                return (Err(err), SamplerState::RunningArrow(control));
            }
        }
    }

    fn wait_inner_zarr(
        &self,
        mut control: Sampler<()>,
        timeout: Option<Duration>,
    ) -> (PyResult<()>, SamplerState) {
        let start_time = Instant::now();
        let step = Duration::from_millis(100);

        loop {
            let time_so_far = Instant::now().saturating_duration_since(start_time);
            let next_timeout = match timeout {
                Some(timeout) => {
                    let Some(remaining) = timeout.checked_sub(time_so_far) else {
                        return (
                            Err(PyTimeoutError::new_err(
                                "Timeout while waiting for sampler to finish",
                            )),
                            SamplerState::RunningZarr(control),
                        );
                    };
                    remaining.min(step)
                }
                None => step,
            };

            match control.wait_timeout(next_timeout) {
                SamplerWaitResult::Trace(_trace) => return (Ok(()), SamplerState::FinishedZarr),
                SamplerWaitResult::Timeout(new_control) => {
                    control = new_control;
                }
                SamplerWaitResult::Err(err, _trace) => {
                    return (Err(err.into()), SamplerState::FinishedZarr)
                }
            }

            if let Err(err) = Python::attach(|py| py.check_signals()) {
                return (Err(err), SamplerState::RunningZarr(control));
            }
        }
    }
}

#[pymethods]
impl PySampler {
    #[staticmethod]
    fn from_pymc(
        settings: PySamplerSettings,
        cores: usize,
        model: PyMcModel,
        progress_type: ProgressType,
        extra_callback: Option<Py<PyAny>>,
        extra_callback_rate: u64,
        store: &mut PyStorage,
    ) -> PyResult<PySampler> {
        PySampler::new(
            settings,
            cores,
            model,
            progress_type,
            extra_callback,
            extra_callback_rate,
            store,
        )
    }

    #[staticmethod]
    fn from_stan(
        settings: PySamplerSettings,
        cores: usize,
        model: StanModel,
        progress_type: ProgressType,
        extra_callback: Option<Py<PyAny>>,
        extra_callback_rate: u64,
        store: &mut PyStorage,
    ) -> PyResult<PySampler> {
        PySampler::new(
            settings,
            cores,
            model,
            progress_type,
            extra_callback,
            extra_callback_rate,
            store,
        )
    }

    #[staticmethod]
    fn from_pyfunc(
        settings: PySamplerSettings,
        cores: usize,
        model: PyModel,
        progress_type: ProgressType,
        extra_callback: Option<Py<PyAny>>,
        extra_callback_rate: u64,
        store: &mut PyStorage,
    ) -> PyResult<PySampler> {
        PySampler::new(
            settings,
            cores,
            model,
            progress_type,
            extra_callback,
            extra_callback_rate,
            store,
        )
    }

    fn is_finished(&mut self, py: Python<'_>) -> PyResult<bool> {
        self.wait(py, Some(0.001))?;
        py.detach(|| {
            let guard = &mut self.0.lock().expect("Poisond sampler state mutex");
            Ok(matches!(
                guard.deref_mut().0,
                SamplerState::FinishedZarr | SamplerState::FinishedArrow(_) | SamplerState::Empty
            ))
        })
    }

    fn pause(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            match self
                .0
                .lock()
                .expect("Poisond sampler state mutex")
                .deref_mut()
            {
                (SamplerState::RunningZarr(control), _) => {
                    control.pause()?;
                    return Ok(());
                }
                (SamplerState::RunningArrow(control), _) => {
                    control.pause()?;
                    return Ok(());
                }
                _ => return Ok(()),
            }
        })
    }

    fn resume(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            match self
                .0
                .lock()
                .expect("Poisond sampler state mutex")
                .deref_mut()
            {
                (SamplerState::RunningZarr(control), _) => {
                    control.resume()?;
                    return Ok(());
                }
                (SamplerState::RunningArrow(control), _) => {
                    control.resume()?;
                    return Ok(());
                }
                _ => return Ok(()),
            }
        })
    }

    #[pyo3(signature = (timeout_seconds=None))]
    fn wait(&mut self, py: Python<'_>, timeout_seconds: Option<f64>) -> PyResult<()> {
        py.detach(|| {
            let guard = &mut self.0.lock().expect("Poisond sampler state mutex");
            let slot = guard.deref_mut();
            let slot = &mut slot.0;

            let timeout = match timeout_seconds {
                Some(val) => Some(Duration::try_from_secs_f64(val).context("Invalid timeout")?),
                None => None,
            };

            let state = std::mem::replace(slot, SamplerState::Empty);

            let (retval, final_state) = match state {
                SamplerState::FinishedZarr
                | SamplerState::FinishedArrow(_)
                | SamplerState::Empty => (Ok(()), state),
                SamplerState::RunningZarr(control) => self.wait_inner_zarr(control, timeout),
                SamplerState::RunningArrow(control) => self.wait_inner_arrow(control, timeout),
            };

            let _ = std::mem::replace(slot, final_state);
            retval
        })
    }

    fn abort(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            let guard = &mut self.0.lock().expect("Poisond sampler state mutex");
            let slot = guard.deref_mut();
            let slot = &mut slot.0;

            let state = std::mem::replace(slot, SamplerState::Empty);

            match state {
                SamplerState::FinishedZarr
                | SamplerState::FinishedArrow(_)
                | SamplerState::Empty => {
                    let _ = std::mem::replace(slot, state);
                    return Ok(());
                }
                SamplerState::RunningZarr(control) => {
                    let (result, _) = control.abort()?;
                    let _ = std::mem::replace(slot, SamplerState::FinishedZarr);
                    if let Some(err) = result {
                        Err(err)?;
                    }
                    Ok(())
                }
                SamplerState::RunningArrow(control) => {
                    let (result, trace) = control.abort()?;
                    let _ = std::mem::replace(slot, SamplerState::FinishedArrow(trace));
                    if let Some(err) = result {
                        Err(err)?;
                    }
                    Ok(())
                }
            }
        })
    }

    #[pyo3(signature = (ignore_error=false))]
    fn is_empty(&self, ignore_error: bool) -> Result<bool> {
        let out = self.0.lock();
        match (ignore_error, out) {
            (false, Err(e)) => return Err(anyhow!("The sampler panicked with error {}", e)),
            (true, Err(_)) => return Ok(true),
            (_, Ok(v)) => {
                return Ok(matches!(v.deref(), (SamplerState::Empty, _)));
            }
        }
    }

    fn flush<'py>(&mut self, py: Python<'py>) -> PyResult<()> {
        match self
            .0
            .lock()
            .expect("Poisond sampler state mutex")
            .deref_mut()
            .0
        {
            SamplerState::FinishedZarr => Ok(()),
            SamplerState::FinishedArrow(_) => Ok(()),
            SamplerState::Empty => Ok(()),
            SamplerState::RunningZarr(ref mut control) => {
                py.detach(|| control.flush())?;
                Ok(())
            }
            SamplerState::RunningArrow(ref mut control) => {
                py.detach(|| control.flush())?;
                Ok(())
            }
        }
    }

    fn inspect<'py>(&self, py: Python<'py>) -> PyResult<Option<PyTrace>> {
        match &mut self
            .0
            .lock()
            .expect("Poisond sampler state mutex")
            .deref_mut()
            .0
        {
            SamplerState::FinishedZarr => Ok(Some(PyTrace(InnerPyTrace::Zarr))),
            SamplerState::FinishedArrow(trace) => {
                Ok(Some(PyTrace(InnerPyTrace::Arrow(Some(trace.clone())))))
            }
            SamplerState::Empty => Ok(None),
            SamplerState::RunningZarr(control) => {
                let (res, _) = py.detach(|| control.inspect())?;
                if let Some(err) = res {
                    return Err(err.into());
                }
                Ok(Some(PyTrace(InnerPyTrace::Zarr)))
            }
            SamplerState::RunningArrow(control) => {
                let (res, trace) = py.detach(|| control.inspect())?;
                if let Some(err) = res {
                    return Err(err.into());
                }
                Ok(Some(PyTrace(InnerPyTrace::Arrow(Some(trace)))))
            }
        }
    }

    fn take_results(&mut self) -> PyResult<PyTrace> {
        let state = &mut self.0.lock().expect("Poisond sampler state mutex");

        match &state.0 {
            SamplerState::FinishedZarr => {
                let _ = std::mem::replace(&mut state.0, SamplerState::Empty);
                Ok(PyTrace(InnerPyTrace::Zarr))
            }
            SamplerState::FinishedArrow(_) => {
                let state = std::mem::replace(&mut state.0, SamplerState::Empty);
                let SamplerState::FinishedArrow(trace) = state else {
                    unreachable!();
                };
                Ok(PyTrace(InnerPyTrace::Arrow(Some(trace))))
            }
            SamplerState::Empty => Err(PyErr::new::<PyValueError, _>(
                "Sampler has no results to take",
            )),
            SamplerState::RunningZarr(_) => Err(PyErr::new::<PyValueError, _>(
                "Sampler is still running, can only take results after it has finished",
            )),
            SamplerState::RunningArrow(_) => Err(PyErr::new::<PyValueError, _>(
                "Sampler is still running, can only take results after it has finished",
            )),
        }
    }
}

enum InnerPyTrace {
    Zarr,
    Arrow(Option<Vec<ArrowTrace>>),
}

#[pyclass]
pub struct PyTrace(InnerPyTrace);

#[pymethods]
impl PyTrace {
    fn is_zarr(&self) -> bool {
        matches!(self.0, InnerPyTrace::Zarr)
    }

    fn is_arrow(&self) -> bool {
        matches!(self.0, InnerPyTrace::Arrow(_))
    }

    fn get_arrow_trace(&mut self) -> PyResult<(Vec<PyRecordBatch>, Vec<PyRecordBatch>)> {
        match &mut self.0 {
            InnerPyTrace::Zarr => Err(PyErr::new::<PyValueError, _>(
                "Trace is not stored in Arrow format",
            )),
            InnerPyTrace::Arrow(trace) => Ok(trace
                .take()
                .ok_or_else(|| PyValueError::new_err("The trace was already taken"))?
                .into_iter()
                .map(|array| {
                    (
                        PyRecordBatch::new(array.posterior),
                        PyRecordBatch::new(array.sample_stats),
                    )
                })
                .collect()),
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct PyTransformAdapt(Arc<Py<PyAny>>);

#[pymethods]
impl PyTransformAdapt {
    #[new]
    pub fn new(adapter: Py<PyAny>) -> Self {
        Self(Arc::new(adapter))
    }
}

impl PyTransformAdapt {
    pub fn inv_transform_normalize(
        &mut self,
        params: &Py<PyAny>,
        untransformed_position: &[f64],
        untransformed_gradient: &[f64],
        transformed_position: &mut [f64],
        transformed_gradient: &mut [f64],
    ) -> Result<f64> {
        Python::attach(|py| {
            let untransformed_position = PyArray1::from_slice(py, untransformed_position);
            let untransformed_gradient = PyArray1::from_slice(py, untransformed_gradient);

            let output = params
                .getattr(py, intern!(py, "inv_transform"))
                .context("Could not access attribute inv_transform")?
                .call1(py, (untransformed_position, untransformed_gradient))
                .context("Failed to call adapter.inv_transform")?;
            let (logdet, transformed_position_out, transformed_gradient_out): (
                f64,
                PyReadonlyArray1<f64>,
                PyReadonlyArray1<f64>,
            ) = output
                .extract(py)
                .context("Execpected results from adapter.inv_transform")?;

            if !transformed_position_out
                .as_slice()?
                .iter()
                .all(|&x| x.is_finite())
            {
                bail!("Transformed position is not finite");
            }
            if !transformed_gradient_out
                .as_slice()?
                .iter()
                .all(|&x| x.is_finite())
            {
                bail!("Transformed position is not finite");
            }

            transformed_position.copy_from_slice(
                transformed_position_out
                    .as_slice()
                    .context("Could not copy transformed_position")?,
            );

            transformed_gradient.copy_from_slice(
                transformed_gradient_out
                    .as_slice()
                    .context("Could not copy transformed_gradient")?,
            );
            Ok(logdet)
        })
    }

    pub fn init_from_transformed_position(
        &mut self,
        params: &Py<PyAny>,
        untransformed_position: &mut [f64],
        untransformed_gradient: &mut [f64],
        transformed_position: &[f64],
        transformed_gradient: &mut [f64],
    ) -> Result<(f64, f64)> {
        Python::attach(|py| {
            let transformed_position = PyArray1::from_slice(py, transformed_position);

            let output = params
                .getattr(py, intern!(py, "init_from_transformed_position"))?
                .call1(py, (transformed_position,))?;
            let (
                logp,
                logdet,
                untransformed_position_out,
                untransformed_gradient_out,
                transformed_gradient_out,
            ): (
                f64,
                f64,
                PyReadonlyArray1<f64>,
                PyReadonlyArray1<f64>,
                PyReadonlyArray1<f64>,
            ) = output.extract(py)?;

            untransformed_position.copy_from_slice(untransformed_position_out.as_slice()?);
            untransformed_gradient.copy_from_slice(untransformed_gradient_out.as_slice()?);
            transformed_gradient.copy_from_slice(transformed_gradient_out.as_slice()?);
            Ok((logp, logdet))
        })
    }

    pub fn init_from_transformed_position_part1(
        &mut self,
        params: &Py<PyAny>,
        untransformed_position: &mut [f64],
        transformed_position: &[f64],
    ) -> Result<Py<PyAny>> {
        Python::attach(|py| {
            let transformed_position = PyArray1::from_slice(py, transformed_position);

            let output = params
                .getattr(py, intern!(py, "init_from_transformed_position_part1"))?
                .call1(py, (transformed_position,))?;
            let (untransformed_position_out, part1): (PyReadonlyArray1<f64>, Py<PyAny>) =
                output.extract(py)?;

            untransformed_position.copy_from_slice(untransformed_position_out.as_slice()?);
            Ok(part1)
        })
    }

    pub fn init_from_transformed_position_part2(
        &mut self,
        params: &Py<PyAny>,
        part1: Py<PyAny>,
        untransformed_gradient: &[f64],
        transformed_gradient: &mut [f64],
    ) -> Result<f64> {
        Python::attach(|py| {
            let untransformed_gradient = PyArray1::from_slice(py, untransformed_gradient);

            let output = params
                .getattr(py, intern!(py, "init_from_transformed_position_part2"))?
                .call1(py, (part1, untransformed_gradient))?;
            let (logdet, transformed_gradient_out): (f64, PyReadonlyArray1<f64>) =
                output.extract(py)?;

            transformed_gradient.copy_from_slice(transformed_gradient_out.as_slice()?);
            Ok(logdet)
        })
    }

    pub fn init_from_untransformed_position(
        &mut self,
        params: &Py<PyAny>,
        untransformed_position: &[f64],
        untransformed_gradient: &mut [f64],
        transformed_position: &mut [f64],
        transformed_gradient: &mut [f64],
    ) -> Result<(f64, f64)> {
        Python::attach(|py| {
            let untransformed_position = PyArray1::from_slice(py, untransformed_position);

            let output = params
                .getattr(py, intern!(py, "init_from_untransformed_position"))
                .context("No attribute init_from_untransformed_position")?
                .call1(py, (untransformed_position,))
                .context("Failed adapter.init_from_untransformed_position")?;
            let (
                logp,
                logdet,
                untransformed_gradient_out,
                transformed_position_out,
                transformed_gradient_out,
            ): (
                f64,
                f64,
                PyReadonlyArray1<f64>,
                PyReadonlyArray1<f64>,
                PyReadonlyArray1<f64>,
            ) = output
                .extract(py)
                .context("Unexpected return value of init_from_untransformed_position")?;

            untransformed_gradient.copy_from_slice(untransformed_gradient_out.as_slice()?);
            transformed_position.copy_from_slice(transformed_position_out.as_slice()?);
            transformed_gradient.copy_from_slice(transformed_gradient_out.as_slice()?);
            Ok((logp, logdet))
        })
    }

    pub fn update_transformation<'a, R: rand::Rng + ?Sized>(
        &'a mut self,
        rng: &mut R,
        untransformed_positions: impl ExactSizeIterator<Item = &'a [f64]>,
        untransformed_gradients: impl ExactSizeIterator<Item = &'a [f64]>,
        untransformed_logp: impl ExactSizeIterator<Item = &'a f64>,
        params: &'a mut Py<PyAny>,
    ) -> Result<()> {
        Python::attach(|py| {
            let positions = PyList::new(
                py,
                untransformed_positions.map(|pos| PyArray1::from_slice(py, pos)),
            )?;
            let gradients = PyList::new(
                py,
                untransformed_gradients.map(|grad| PyArray1::from_slice(py, grad)),
            )?;

            let logps = PyArray1::from_iter(py, untransformed_logp.copied());
            let seed = rng.next_u64();

            params
                .getattr(py, intern!(py, "update"))?
                .call1(py, (seed, positions, gradients, logps))?;
            Ok(())
        })
    }

    pub fn new_transformation<R: rand::Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        untransformed_position: &[f64],
        untransformed_gradient: &[f64],
        chain: u64,
    ) -> Result<Py<PyAny>> {
        Python::attach(|py| {
            let position = PyArray1::from_slice(py, untransformed_position);
            let gradient = PyArray1::from_slice(py, untransformed_gradient);

            let seed = rng.next_u64();

            let transformer = self.0.call1(py, (seed, position, gradient, chain))?;

            Ok(transformer)
        })
    }

    pub fn transformation_id(&self, params: &Py<PyAny>) -> Result<i64> {
        Python::attach(|py| {
            let id: i64 = params
                .getattr(py, intern!(py, "transformation_id"))?
                .extract(py)?;
            Ok(id)
        })
    }
}

/// A Python module implemented in Rust.
#[pymodule(gil_used = false)]
pub fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySampler>()?;
    m.add_class::<PyMcModel>()?;
    m.add_class::<LogpFunc>()?;
    m.add_class::<ExpandFunc>()?;
    m.add_class::<StanLibrary>()?;
    m.add_class::<StanModel>()?;
    m.add_class::<PyNutsSettings>()?;
    m.add_class::<PyMclmcSettings>()?;
    m.add_class::<PyChainProgress>()?;
    m.add_class::<ProgressType>()?;
    m.add_class::<PyModel>()?;
    m.add_class::<PyVariable>()?;
    m.add_class::<PyStorage>()?;
    m.add_class::<PyTrace>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    pyo3_object_store::register_store_module(m.py(), m, "_lib", "store")?;
    pyo3_object_store::register_exceptions_module(m.py(), m, "_lib", "exceptions")?;
    Ok(())
}
