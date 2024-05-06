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
    InMemorySession, Session, SessionBuilder, SessionInputValue, SessionInputs, Value,
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
pub struct TorchModel {
    ndim: usize,
    logp_model: Box<[u8]>,
    providers: Vec<ExecutionProviderDispatch>,
}

impl TorchModel {
    fn make_logp_session<'a>(&'a self) -> Result<()> {
        todo!()
    }
}

#[pymethods]
impl TorchModel {
    #[new]
    pub fn new_py<'py>(ndim: usize, logp_model: Bound<'py, PyBytes>) -> Result<Self> {
        todo!()
    }
}

#[derive(Clone)]
pub struct TorchTrace {
    trace: MutableFixedSizeListArray<MutablePrimitiveArray<f64>>,
}

impl DrawStorage for TorchTrace {
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
pub enum TorchLogpError {
    #[error("Error while computing logp and gradient: {0:?}")]
    Iree(#[from] anyhow::Error),
    #[error("Bad logp value in gradient evaluation")]
    BadLogp(),
}

impl LogpError for TorchLogpError {
    fn is_recoverable(&self) -> bool {
        match self {
            Self::BadLogp() => true,
            _ => false,
        }
    }
}

pub struct TorchLogpFunc<'model> {
    ndim: usize,
}

impl<'model> TorchLogpFunc<'model> {
    fn new(ndim: usize) -> Result<Self> {
        todo!()
    }
}

impl<'model> CpuLogpFunc for TorchLogpFunc<'model> {
    type LogpError = TorchLogpError;

    fn dim(&self) -> usize {
        self.ndim
    }

    fn logp(
        &mut self,
        position: &[f64],
        gradient: &mut [f64],
    ) -> std::result::Result<f64, Self::LogpError> {
        todo!()
    }
}

impl Model for TorchModel {
    type Math<'model> = CpuMath<TorchLogpFunc<'model>>
    where
        Self: 'model;

    type DrawStorage<'model, S: nuts_rs::Settings> = TorchTrace
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
