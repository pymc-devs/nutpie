use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use anyhow::Result;
use anyhow::{anyhow, Context};
use arrow::array::{Array, FixedSizeListBuilder, PrimitiveBuilder, StructArray};
use arrow::datatypes::{Field, Float64Type};
use itertools::Itertools;
use nuts_rs::{CpuLogpFunc, CpuMath, DrawStorage, LogpError, Math, Model};
use ort::{
    AllocationDevice, Allocator, CPUExecutionProvider, CUDAExecutionProvider,
    ExecutionProviderDispatch, InMemorySession, IoBinding, MemoryInfo, MemoryType,
    OpenVINOExecutionProvider, Session, SessionInputs, TVMExecutionProvider, Tensor,
    TensorRTExecutionProvider, Value,
};
use pyo3::{
    pyclass, pymethods,
    types::{PyBytes, PyBytesMethods},
    Bound,
};
use rand_distr::{Distribution, Uniform};
use thiserror::Error;

#[derive(Debug, Clone)]
#[pyclass]
pub struct OnnxModel {
    ndim: usize,
    logp_model: Box<[u8]>,
    providers: OnnxProviders,
    sessions: Arc<Vec<Session>>,
    count: Arc<AtomicUsize>,
}

impl OnnxModel {
    fn make_plain_logp_session<'a>(&'a self) -> Result<Session> {
        let logp_session = Session::builder()?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_memory_pattern(true)?
            //.commit_from_memory_directly(&self.logp_model)?;
            .commit_from_memory(&self.logp_model)?;
        //

        Ok(logp_session)
    }

    fn make_logp_session<'a>(&'a self) -> Result<Session> {
        let logp_session = Session::builder()?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_execution_providers(
                self.providers
                    .clone()
                    .providers
                    .into_iter()
                    .map(|val| val.into()),
            )?
            .with_memory_pattern(true)?
            //.commit_from_memory_directly(&self.logp_model)?;
            .commit_from_memory(&self.logp_model)?;
        //

        Ok(logp_session)
    }
}

#[pymethods]
impl OnnxModel {
    #[new]
    pub fn new_py<'py>(
        ndim: usize,
        logp_model: Bound<'py, PyBytes>,
        providers: OnnxProviders,
    ) -> Result<Self> {
        let mut model = Self {
            ndim,
            providers,
            logp_model: logp_model.as_bytes().into(),
            sessions: Arc::new(vec![]),
            count: Arc::new(0usize.into()),
        };
        for _ in 0..6 {
            let session = model.make_logp_session()?;
            Arc::get_mut(&mut model.sessions).unwrap().push(session);
        }

        let session = model.make_plain_logp_session()?;

        let pos = vec![0f32; ndim];
        let input = Tensor::from_array(([ndim], pos))?;

        session.run(ort::inputs![input]?)?;

        Ok(model)
    }
}

pub struct OnnxTrace {
    trace: FixedSizeListBuilder<PrimitiveBuilder<Float64Type>>,
}

impl DrawStorage for OnnxTrace {
    fn append_value(&mut self, point: &[f64]) -> Result<()> {
        self.trace.values().append_slice(point);
        self.trace.append(true);
        Ok(())
    }

    fn finalize(mut self) -> Result<Arc<dyn Array>> {
        //let data_type = DataType::Struct(fields.into());
        let data: Arc<dyn Array> = Arc::new(self.trace.finish());
        let field = Field::new("unconstrained_draw", data.data_type().clone(), false);
        let fields = vec![field];
        let struct_array = StructArray::new(fields.into(), vec![data], None);
        Ok(Arc::new(struct_array))
    }

    fn inspect(&self) -> Result<Arc<dyn Array>> {
        let data: Arc<dyn Array> = Arc::new(self.trace.finish_cloned());
        let field = Field::new("unconstrained_draw", data.data_type().clone(), false);
        let fields = vec![field];
        let struct_array = StructArray::new(fields.into(), vec![data], None);
        Ok(Arc::new(struct_array))
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
    //session: &'model InMemorySession<'model>,
    input: Tensor<f32>,
    binding: IoBinding<'model>,
    session: &'model Session,
    ndim: usize,
    input_allocator: Allocator,
    output_allocator: Allocator,
}

impl<'model> OnnxLogpFunc<'model> {
    //fn new(ndim: usize, session: &'model InMemorySession<'model>) -> Result<Self> {
    fn new(
        ndim: usize,
        binding: IoBinding<'model>,
        session: &'model Session,
        input: Tensor<f32>,
        input_allocator: Allocator,
        output_allocator: Allocator,
    ) -> Result<Self> {
        Ok(Self {
            session,
            binding,
            ndim,
            input,
            input_allocator,
            output_allocator,
        })
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
        /*
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
            .for_each(|(out, &val)| *out = val as f64);
        */

        let (_, input_vals) = self.input.extract_raw_tensor_mut();
        position
            .iter()
            .zip(input_vals.iter_mut())
            .for_each(|(val, loc)| *loc = *val as _);

        self.binding
            .bind_input(&self.session.inputs[0].name, &self.input)
            .context("Coud not bind input to logp function")?;

        let outputs = self.binding.run().context("Could not run logp function")?;
        let first = &outputs[0];
        let logp: f32 = first
            .try_extract_scalar()
            .context("First output wnot a scalar")?;

        let grad = &outputs[1];
        let (_, grad): (_, &[f32]) = grad
            .try_extract_raw_tensor()
            .context("First output wnot a scalar")?;

        gradient
            .iter_mut()
            .zip(grad.iter())
            .for_each(|(out, &val)| *out = val as f64);

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
        let items = PrimitiveBuilder::new();
        let trace = FixedSizeListBuilder::new(items, self.ndim.try_into().unwrap());

        Ok(OnnxTrace { trace })
    }

    fn math(&self) -> Result<Self::Math<'_>> {
        //let session = self.make_logp_session()?;
        let count = self.count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let count = count % self.sessions.len();

        let session = &self.sessions[count];

        let input_allocator = Allocator::new(
            session,
            MemoryInfo::new(
                AllocationDevice::CUDAPinned,
                0,
                ort::AllocatorType::Device,
                MemoryType::CPUInput,
            )?,
        )?;
        let output_allocator = Allocator::new(
            session,
            MemoryInfo::new(
                AllocationDevice::CUDAPinned,
                0,
                ort::AllocatorType::Device,
                MemoryType::CPUOutput,
            )?,
        )?;

        let mut binding = session.create_binding()?;

        let input = Tensor::<f32>::new(&input_allocator, [self.ndim])?;

        binding.bind_input(&session.inputs[0].name, &input)?;

        let scalar_shape: [usize; 0] = [];
        let logp_output = Tensor::<f32>::new(&output_allocator, scalar_shape)?;
        let grad_output = Tensor::<f32>::new(&output_allocator, [self.ndim])?;

        binding.bind_output(&session.outputs[0].name, logp_output)?;
        binding.bind_output(&session.outputs[1].name, grad_output)?;

        Ok(CpuMath::new(OnnxLogpFunc::new(
            self.ndim,
            binding,
            session,
            input,
            input_allocator,
            output_allocator,
        )?))
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

#[derive(Debug, Clone)]
enum Provider {
    Cpu(CPUExecutionProvider),
    Cuda(CUDAExecutionProvider),
    TensorRt(TensorRTExecutionProvider),
    Tvm(TVMExecutionProvider),
    OpenVINO(OpenVINOExecutionProvider),
}

impl Into<ExecutionProviderDispatch> for Provider {
    fn into(self) -> ExecutionProviderDispatch {
        match self {
            Self::Cpu(val) => val.build().error_on_failure().into(),
            Self::Cuda(val) => val.build().error_on_failure().into(),
            Self::TensorRt(val) => val.build().error_on_failure().into(),
            Self::Tvm(val) => val.build().error_on_failure().into(),
            Self::OpenVINO(val) => val.build().error_on_failure().into(),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct OnnxProviders {
    providers: Vec<Provider>,
}

#[pymethods]
impl OnnxProviders {
    #[new]
    pub fn new() -> Self {
        Self { providers: vec![] }
    }

    pub fn add_cpu(&mut self) -> Result<()> {
        self.providers
            .push(Provider::Cpu(CPUExecutionProvider::default()));
        Ok(())
    }

    pub fn add_cuda(&mut self) -> Result<()> {
        self.providers.push(Provider::Cuda(
            CUDAExecutionProvider::default().with_cuda_graph(),
        ));
        Ok(())
    }

    pub fn add_tvm(&mut self) -> Result<()> {
        let provider = TVMExecutionProvider::default();
        self.providers.push(Provider::Tvm(provider));
        Ok(())
    }

    pub fn add_openvino(&mut self) -> Result<()> {
        let provider = OpenVINOExecutionProvider::default();
        self.providers.push(Provider::OpenVINO(provider));
        Ok(())
    }

    pub fn add_tensorrt(&mut self) -> Result<()> {
        self.providers
            .push(Provider::TensorRt(TensorRTExecutionProvider::default()));
        Ok(())
    }
}
