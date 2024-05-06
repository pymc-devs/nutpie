use anyhow::{anyhow, Context};
use anyhow::{bail, Result};
use arrow2::{
    array::{MutableArray, MutableFixedSizeListArray, MutablePrimitiveArray, StructArray, TryPush},
    datatypes::{DataType, Field},
};
use itertools::Itertools;
use nuts_rs::{CpuLogpFunc, CpuMath, DrawStorage, LogpError, Model};
use ort::{
    inputs, CPUExecutionProvider, CUDAExecutionProvider, ExecutionProviderDispatch,
    InMemorySession, Session, SessionBuilder, SessionInputValue, SessionInputs,
    TensorRTExecutionProvider, Value,
};
use pyo3::{
    pyclass, pymethods,
    types::{PyBytes, PyBytesMethods},
    Bound,
};
use rand_distr::{Distribution, Uniform};
use thiserror::Error;

#[pyclass]
#[derive(Clone, Debug)]
pub struct OnnxModel {
    ndim: usize,
    logp_model: Box<[u8]>,
    providers: Vec<ExecutionProviderDispatch>,
}

impl OnnxModel {
    fn make_logp_session<'a>(&'a self) -> Result<InMemorySession<'a>> {
        let logp_session = Session::builder()?
            .with_execution_providers(self.providers.iter().cloned())?
            .with_memory_pattern(true)?
            .commit_from_memory_directly(&self.logp_model)?;

        Ok(logp_session)
    }
}

#[pymethods]
impl OnnxModel {
    #[new]
    pub fn new_py<'py>(
        ndim: usize,
        logp_model: Bound<'py, PyBytes>,
        providers: &OnnxProviders,
    ) -> Result<Self> {
        Ok(Self {
            ndim,
            providers: providers.providers.iter().cloned().collect(),
            logp_model: logp_model.as_bytes().into(),
        })
    }
}

#[derive(Clone)]
pub struct OnnxTrace {
    trace: MutableFixedSizeListArray<MutablePrimitiveArray<f64>>,
}

impl DrawStorage for OnnxTrace {
    fn append_value(&mut self, point: &[f64]) -> Result<()> {
        self.trace.try_push(Some(point.iter().map(|&x| Some(x))))?;
        Ok(())
    }

    fn finalize(mut self) -> Result<Box<dyn arrow2::array::Array>> {
        let field = Field::new("unconstrained_draw", self.trace.data_type().clone(), false);
        let fields = vec![field];
        let data_type = DataType::Struct(fields);
        let struct_array = StructArray::new(data_type, vec![self.trace.as_box()], None);
        Ok(Box::new(struct_array))
    }

    fn inspect(&mut self) -> Result<Box<dyn arrow2::array::Array>> {
        self.clone().finalize()
    }
}

#[derive(Error, Debug)]
pub enum OnnxLogpError {
    #[error("Error while computing logp and gradient: {0:?}")]
    Iree(#[from] anyhow::Error),
    #[error("Bad logp value in gradient evaluation")]
    BadLogp(),
}

impl LogpError for OnnxLogpError {
    fn is_recoverable(&self) -> bool {
        match self {
            Self::BadLogp() => true,
            _ => false,
        }
    }
}

pub struct OnnxLogpFunc<'model> {
    session: InMemorySession<'model>,
    ndim: usize,
}

impl<'model> OnnxLogpFunc<'model> {
    fn new(ndim: usize, session: InMemorySession<'model>) -> Result<Self> {
        Ok(Self { session, ndim })
    }
}

impl<'model> CpuLogpFunc for OnnxLogpFunc<'model> {
    type LogpError = OnnxLogpError;

    fn dim(&self) -> usize {
        self.ndim
    }

    fn logp(
        &mut self,
        position: &[f64],
        gradient: &mut [f64],
    ) -> std::result::Result<f64, Self::LogpError> {
        let position = position.iter().map(|&x| x as f32).collect_vec();
        let position =
            Value::from_array(([position.len()], position)).context("Could not create input")?;
        let inputs = SessionInputs::ValueArray([position.into()]);
        let mut outputs = self
            .session
            .run(inputs)
            .context("Could not run logp function")?;
        let logp = outputs
            .pop_first()
            .context("Could not extract first output")?;
        let grad = outputs
            .pop_first()
            .context("Could not extract second output")?;
        let logp: f32 = logp
            .1
            .try_extract_raw_tensor()
            .context("Could not read logp value")?
            .1[0];
        let vals = grad
            .1
            .try_extract_raw_tensor::<f32>()
            .context("Could not read grad value")?
            .1;
        if vals.len() != gradient.len() {
            Err(anyhow!("Logp return gradient with incorrect length"))?;
        }
        gradient
            .iter_mut()
            .zip(vals.iter())
            .for_each(|(mut out, &val)| *out = val as f64);
        Ok(logp as f64)
    }
}

impl Model for OnnxModel {
    type Math<'model> = CpuMath<OnnxLogpFunc<'model>>
    where
        Self: 'model;

    type DrawStorage<'model, S: nuts_rs::Settings> = OnnxTrace
    where
        Self: 'model;

    fn new_trace<'model, S: nuts_rs::Settings, R: rand::prelude::Rng + ?Sized>(
        &'model self,
        rng: &mut R,
        chain_id: u64,
        settings: &'model S,
    ) -> Result<Self::DrawStorage<'model, S>> {
        let items = MutablePrimitiveArray::new();
        let trace = MutableFixedSizeListArray::new(items, self.ndim);

        Ok(OnnxTrace { trace })
    }

    fn math(&self) -> Result<Self::Math<'_>> {
        let session = self.make_logp_session()?;
        Ok(CpuMath::new(OnnxLogpFunc::new(self.ndim, session)?))
    }

    fn init_position<R: rand::prelude::Rng + ?Sized>(
        &self,
        rng: &mut R,
        position: &mut [f64],
    ) -> Result<()> {
        let dist = Uniform::new(-2., 2.);
        dist.sample_iter(rng)
            .zip(position.iter_mut())
            .for_each(|(val, pos)| *pos = val);
        Ok(())
    }
}

#[pyclass]
pub struct OnnxProviders {
    providers: Vec<ExecutionProviderDispatch>,
}

#[pymethods]
impl OnnxProviders {
    #[new]
    pub fn new() -> Self {
        Self { providers: vec![] }
    }

    pub fn add_cpu(&mut self) -> Result<()> {
        self.providers.push(CPUExecutionProvider::default().into());
        Ok(())
    }

    pub fn add_cuda(&mut self) -> Result<()> {
        self.providers.push(CUDAExecutionProvider::default().into());
        Ok(())
    }

    pub fn add_tensorrt(&mut self) -> Result<()> {
        self.providers
            .push(TensorRTExecutionProvider::default().into());
        Ok(())
    }
}
