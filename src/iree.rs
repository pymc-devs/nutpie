use std::{
    io::{stderr, stdout, Write},
    mem::{forget, transmute, ManuallyDrop},
    sync::{
        mpsc::{sync_channel, Receiver, SyncSender},
        Arc, Mutex, OnceLock,
    },
    thread::{spawn, JoinHandle},
};

use anyhow::{anyhow, Context, Result};
use arrow2::{
    array::{MutableArray, MutableFixedSizeListArray, MutablePrimitiveArray, StructArray, TryPush},
    datatypes::{DataType, Field},
};
use eerie::runtime::{
    api::{Call, Instance, InstanceOptions, Session, SessionOptions},
    hal::{BufferMapping, BufferView, Device, DriverRegistry, EncodingType},
    vm::{DynamicList, Function, List, Ref, ToRef, Undefined, Value},
};
use numpy::{PyArray1, PyReadonlyArray1, PyReadwriteArray1};
use nuts_rs::{CpuLogpFunc, CpuMath, DrawStorage, LogpError, Math, Model};
use pyo3::{
    pyclass, pymethods,
    types::{PyBytes, PyBytesMethods},
    Bound, Py, Python,
};
use rand_distr::{num_traits::ToPrimitive, Distribution, StandardNormal};
use thiserror::Error;

static INSTANCE: OnceLock<Result<Instance>> = OnceLock::new();

fn get_instance() -> Result<&'static Instance> {
    match INSTANCE.get_or_init(|| {
        let mut registry = DriverRegistry::new();
        let options = InstanceOptions::new(&mut registry).use_all_available_drivers();
        let instance = Instance::new(&options)?;

        Ok(instance)
    }) {
        &Ok(ref instance) => Ok(instance),
        &Err(ref err) => Err(anyhow!("Could not access iree instance: {}", err)),
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct IreeModel {
    //logp_module: Box<[u8]>,
    //expand_module: Box<[u8]>,
    //devices: Arc<Box<[Device<'static>]>>,
    //devices: Box<[String]>,
    ndim: usize,
    session_maker: Arc<Mutex<Receiver<Result<CpuMath<LogpFunc<'static>>>>>>,
    maker_thread: Arc<JoinHandle<Result<()>>>,
}

#[pymethods]
impl IreeModel {
    #[new]
    pub fn new_py<'py>(
        device: String,
        logp_module: Bound<'py, PyBytes>,
        expand_module: Bound<'py, PyBytes>,
        ndim: usize,
    ) -> Result<Self> {
        let logp_module: Box<[u8]> = logp_module.as_bytes().into();
        let expand_module: Box<[u8]> = expand_module.as_bytes().into();

        Self::new(device, logp_module, expand_module, ndim)
    }

    pub fn call_logp(
        &self,
        position: PyReadonlyArray1<f64>,
        mut gradient: PyReadwriteArray1<f64>,
    ) -> Result<f64> {
        let mut math = self.math()?;
        let logp = math.logp(&position.as_slice()?, gradient.as_slice_mut()?)?;
        Ok(logp)
    }
}

impl IreeModel {
    fn new(
        device: String,
        logp_module: Box<[u8]>,
        expand_module: Box<[u8]>,
        ndim: usize,
    ) -> Result<Self> {
        let (session_maker_sender, session_maker) = sync_channel(0);

        let maker_thread = spawn(move || {
            let run_loop = move || {
                let instance = get_instance()?;
                let device = instance.try_create_default_device(&device)?;

                // FIXME
                let device: Device<'static> = unsafe { transmute(device) };
                let devices = vec![device];

                let logp_module: Arc<[u8]> = Arc::from(logp_module);
                let expand_module: Arc<[u8]> = Arc::from(expand_module);

                for device in devices.iter().cycle() {
                    let make_math = || {
                        let logp_func = LogpFunc::new(
                            ndim,
                            logp_module.clone(),
                            expand_module.clone(),
                            device,
                        )?;
                        Ok(CpuMath::new(logp_func))
                    };

                    let math_result = make_math();
                    session_maker_sender
                        .send(math_result)
                        .map_err(|_| anyhow!("Could not send iree math"))?;
                }
                Ok(())
            };
            let res = run_loop();
            dbg!(res)
        });

        let session_maker = Arc::new(Mutex::new(session_maker));
        let maker_thread = Arc::new(maker_thread);

        Ok(IreeModel {
            ndim,
            //devices: vec![device].into(),
            //logp_module: logp_module.as_bytes().into(),
            //expand_module: expand_module.as_bytes().into(),
            maker_thread,
            session_maker,
        })
    }
}

#[derive(Debug)]
pub struct LogpFunc<'model> {
    pub outputs: DynamicList<'model, Undefined>,
    //pub inputs: DynamicList<'model, Ref<'model, BufferView<'model, f32>>>,
    pub inputs: DynamicList<'model, Undefined>,
    pub logp_func: ManuallyDrop<Function<'model>>,
    pub session: ManuallyDrop<Session<'model>>,
    //pub device: ManuallyDrop<Device<'model>>,
    //pub device: &'model Device<'model>,
    logp_compiled: Arc<[u8]>,
    expand_compiled: Arc<[u8]>,
    pub ndim: usize,
    pub buffer: Box<[f32]>,
}

impl<'model> LogpFunc<'model> {
    pub fn new(
        ndim: usize,
        //device: &'model Device<'model>,
        logp_compiled: Arc<[u8]>,
        expand_compiled: Arc<[u8]>,
        //session: Session<'model>,
        //device: &'model str,
        device: &Device<'static>,
    ) -> Result<Self> {
        let instance = get_instance()?;
        //let device = instance.try_create_default_device(device)?;

        let options = SessionOptions::default();
        let session = Session::create_with_device(instance, &options, &device)
            .context("Could not create session")?;

        // TODO iree things are ref counted internall, so this is probably fine, but I hate it...
        //let session: Session<'static> = unsafe { transmute(session) };

        // TODO fix the lifetime of this reference
        unsafe { session.append_module_from_memory(&logp_compiled) }
            .context("Could not load iree logp function")?;
        //unsafe { session.append_module_from_memory(expand_compiled) }.context("Coxd not load iree expand function")?;

        let logp_func = session
            .lookup_function("jit_jax_funcified_fgraph.logp")
            .context("Could not find gradient function in module")?;

        //let call = Call::new(&session, &logp_func)?;

        // TODO iree things are ref counted internall, so this is probably fine, but I hate it...
        //let call: Call<'model> = unsafe { transmute(call) };

        // TODO iree things are ref counted internall, so this is probably fine, but I hate it...
        let logp_func: Function<'model> = unsafe { transmute(logp_func) };

        let inputs = DynamicList::new(2, instance)?;
        let outputs = DynamicList::new(2, instance)?;

        Ok(Self {
            //device: ManuallyDrop::new(device),
            //device,
            inputs,
            outputs,
            logp_compiled,
            expand_compiled,
            ndim,
            session: ManuallyDrop::new(session),
            logp_func: ManuallyDrop::new(logp_func),
            buffer: vec![0.; ndim].into(),
            //call,
        })
    }
}

impl<'model> Drop for LogpFunc<'model> {
    fn drop(&mut self) {
        unsafe {
            drop(ManuallyDrop::take(&mut self.logp_func));
            drop(ManuallyDrop::take(&mut self.session));
            //drop(ManuallyDrop::take(&mut self.device));
        }
    }
}

#[derive(Error, Debug)]
pub enum IreeLogpError {
    #[error("Error while computing logp and gradient: {0:?}")]
    Iree(#[from] anyhow::Error),
    #[error("Bad logp value in gradient evaluation")]
    BadLogp(),
}

impl LogpError for IreeLogpError {
    fn is_recoverable(&self) -> bool {
        match self {
            Self::BadLogp() => true,
            _ => false,
        }
    }
}

impl<'model> CpuLogpFunc for LogpFunc<'model> {
    type LogpError = IreeLogpError;

    fn dim(&self) -> usize {
        self.ndim
    }

    fn logp(
        &mut self,
        position: &[f64],
        gradient: &mut [f64],
    ) -> std::result::Result<f64, Self::LogpError> {
        let instance = get_instance()?;

        self.buffer
            .iter_mut()
            .zip(position.iter())
            .for_each(|(out, &val)| *out = val as f32);

        let input_buffer = BufferView::<f32>::new(
            &self.session,
            &[position.len()],
            EncodingType::DenseRowMajor,
            &self.buffer,
        )
        .context("Could not create buffer view")?;

        let input_buffer_ref = input_buffer
            .to_ref(instance)
            .context("Could not create iree ref")?;

        //dbg!(&input_buffer_ref);

        self.inputs
            .push_ref(&input_buffer_ref)
            .context("Could not push input buffer to inputs")?;

        let logp_func = self
            .session
            .lookup_function("jit_jax_funcified_fgraph.logp")
            .context("Could not find gradient function in module")?;

        //dbg!(&self.inputs.get_ref::<BufferView<f32>>(0));
        //stderr().lock().flush();
        //stdout().lock().flush();

        logp_func
            .invoke(&self.inputs, &self.outputs)
            .context("Could not invoke logp function")?;
        //let mut call = Call::new(&self.session, &self.logp_func).context("Could not create iree Call")?;

        //let inputs = call.input_list();

        //inputs.push_ref(&input_buffer_ref).context("Could not push input")?;
        //drop(input_buffer_ref);
        //drop(input_buffer);
        //drop(inputs);

        //call.invoke().context("Could not invoke iree function")?;

        let output_val: Value<f32> = self
            .outputs
            .get_value(0)
            .context("Could not extract logp value")?;
        let logp: f64 = output_val.from_value().into();

        /*
        let logp_buffer_ref: Ref<BufferView<f32>> = self
            .outputs
            .get_ref(0)
            .context("Could not get logp buffer")?;
        let logp_buffer = logp_buffer_ref.to_buffer_view(&self.session);
        */

        let gradient_buffer_ref: Ref<BufferView<f32>> = self
            .outputs
            .get_ref(1)
            .context("Could not get output buffer")?;
        let gradient_buffer = gradient_buffer_ref.to_buffer_view(&self.session);

        gradient_buffer
            .copy_to_host(&mut self.buffer)
            .context("Could not copy gradient buffer from iree device")?;

        //let mut logp_array = [0f32];
        //logp_buffer.copy_to_host(&self.device, &mut logp_array).context("Could not copy logp value")?;
        //let logp = logp_array[0];

        drop(input_buffer_ref);
        drop(input_buffer);

        drop(gradient_buffer_ref);
        drop(gradient_buffer);

        //drop(logp_buffer_ref);
        //drop(logp_buffer);

        self.inputs.clear();
        self.outputs.clear();

        let mut has_bad_grad = false;
        gradient
            .iter_mut()
            .zip(self.buffer.iter())
            .for_each(|(out, &val)| {
                *out = val as f64;
                if !val.is_finite() {
                    has_bad_grad = true;
                }
            });

        if (!logp.is_finite()) | has_bad_grad {
            return Err(IreeLogpError::BadLogp());
        }

        Ok(logp as f64)
    }
}

#[derive(Clone)]
pub struct IreeTrace {
    trace: MutableFixedSizeListArray<MutablePrimitiveArray<f64>>,
}

impl DrawStorage for IreeTrace {
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

impl Model for IreeModel {
    type Math<'model> = CpuMath<LogpFunc<'model>>
    where
        Self: 'model;

    type DrawStorage<'model, S: nuts_rs::Settings> = IreeTrace
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

        Ok(IreeTrace { trace })
    }

    fn math(&self) -> Result<Self::Math<'_>> {
        self.session_maker
            .lock()
            .expect("Poisoned mutex")
            .recv()
            .context("Could not create iree session")?
    }

    fn init_position<R: rand::prelude::Rng + ?Sized>(
        &self,
        rng: &mut R,
        position: &mut [f64],
    ) -> Result<()> {
        let dist = StandardNormal;
        dist.sample_iter(rng)
            .zip(position.iter_mut())
            .for_each(|(val, pos)| *pos = val);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs::File,
        io::Read,
        mem::{transmute, ManuallyDrop},
        path::Path,
    };

    use anyhow::{Context, Result};
    use eerie::runtime::{
        api::{Call, Session, SessionOptions},
        hal::{BufferView, EncodingType},
        vm::{DynamicList, Function, List, Ref, ToRef, Undefined, Value},
    };
    use nuts_rs::{Math, Model};

    use super::{get_instance, IreeModel, LogpFunc};

    #[test]
    fn run_logp_manual1() -> Result<()> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("example-iree")
            .join("example-logp.fbvm");
        let mut logp_compiled = Vec::new();
        File::open(path)?.read_to_end(&mut logp_compiled)?;

        let instance = get_instance()?;
        let device = instance.try_create_default_device("local-task")?;

        let options = SessionOptions::default();
        let session = Session::create_with_device(instance, &options, &device)
            .context("Could not create session")?;

        // TODO iree things are ref counted internall, so this is probably fine, but I hate it...
        let session: Session<'static> = unsafe { transmute(session) };

        unsafe { session.append_module_from_memory(&logp_compiled) }
            .context("Could not load iree logp function")?;
        //unsafe { session.append_module_from_memory(expand_compiled) }.context("Coxd not load iree expand function")?;

        let logp_func = session
            .lookup_function("jit_jax_funcified_fgraph.logp")
            .context("Could not find gradient function in module")?;

        //let inputs: DynamicList<Ref<BufferView<f32>>> = DynamicList::new(2, instance)?;
        //let inputs: DynamicList<Undefined> = DynamicList::new(2, instance)?;
        //let outputs: DynamicList<Undefined> = DynamicList::new(2, instance)?;

        // TODO iree things are ref counted internall, so this is probably fine, but I hate it...
        let logp_func: Function<'static> = unsafe { transmute(logp_func) };

        let mut call = Call::new(&session, &logp_func)?;

        let mut buffer: Box<[f32]> = vec![0.; 2].into();
        let position = vec![1., 2.];
        let mut gradient: Box<[f64]> = vec![-1., -1.].into();

        buffer
            .iter_mut()
            .zip(position.iter())
            .for_each(|(out, &val)| *out = val as _);

        let input_buffer = BufferView::<f32>::new(
            &session,
            &[position.len()],
            EncodingType::DenseRowMajor,
            &buffer,
        )
        .context("Could not create buffer view")?;

        let input_buffer_ref = input_buffer
            .to_ref(instance)
            .context("Could not create iree ref")?;

        let inputs = call.input_list();
        inputs
            .push_ref(&input_buffer_ref)
            .context("Could not push input buffer to inputs")?;

        //dbg!(&inputs.get_ref::<BufferView<f32>>(0));
        //dbg!(&input_buffer_ref);

        /*
        logp_func
            .invoke(&inputs, &outputs)
            .context("Could not invoke logp function")?;
        */
        drop(inputs);
        call.invoke().context("Could not invoke iree function")?;

        drop(input_buffer_ref);
        drop(input_buffer);

        // TODO For some reason it seems we need to keep this alive until after the call...
        // Maybe a missing refcount increase somewhere?

        let outputs = call.output_list();

        let output_val: Value<f32> = outputs
            .get_value(0)
            .context("Could not extract logp value")?;
        let logp: f64 = output_val.from_value().into();
        dbg!(logp);

        let gradient_buffer: Ref<BufferView<f32>> =
            outputs.get_ref(1).context("Could not get output buffer")?;
        let gradient_buffer = gradient_buffer.to_buffer_view(&session);

        gradient_buffer
            .copy_to_host(&mut buffer)
            .context("Could not copy gradient buffer from iree device")?;

        gradient
            .iter_mut()
            .zip(buffer.iter())
            .for_each(|(out, &val)| *out = val as _);

        dbg!(gradient);

        Ok(())
    }

    #[test]
    fn run_logp_seg() -> Result<()> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("example-iree")
            .join("example-logp.fbvm");
        let mut logp_compiled = Vec::new();
        File::open(path)?.read_to_end(&mut logp_compiled)?;

        let logp_expand = vec![];

        let model = IreeModel::new(
            "local-task".into(),
            logp_compiled.into(),
            logp_expand.into(),
            2,
        )?;

        let mut math = model.math()?;

        let position = vec![1., 2.];
        let mut gradient = vec![-1., -1.];
        math.logp(&position, &mut gradient)?;

        drop(math);
        drop(model);

        Ok(())
    }

    #[test]
    fn run_logp_manual2() -> Result<()> {
        /*
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("example-iree")
            .join("example-logp.fbvm");
        let mut logp_compiled = Vec::new();
        File::open(path)?.read_to_end(&mut logp_compiled)?;

        let logp_expand = vec![];

        let model = IreeModel::new(
            "cuda".into(),
            logp_compiled.into(),
            logp_expand.into(),
            2,
        );

        let instance = get_instance()?;

        let device = instance.try_create_default_device(&model.devices[0])?;

        //let mut math_obj = LogpFunc::new(model.ndim, &model.logp_module, &model.expand_module, device)?;

        let mut math_obj = {

            let instance = get_instance()?;

            let options = SessionOptions::default();
            let session = Session::create_with_device(instance, &options, &device)
                .context("Could not create session")?;

            // TODO iree things are ref counted internall, so this is probably fine, but I hate it...
            //let session: Session<'static> = unsafe { transmute(session) };

            unsafe { session.append_module_from_memory(&model.logp_module) }
                .context("Could not load iree logp function")?;
            //unsafe { session.append_module_from_memory(expand_compiled) }.context("Coxd not load iree expand function")?;

            let logp_func = session
                .lookup_function("jit_jax_funcified_fgraph.logp")
                .context("Could not find gradient function in module")?;

            //let call = Call::new(&session, &logp_func)?;

            // TODO iree things are ref counted internall, so this is probably fine, but I hate it...
            //let call: Call<'model> = unsafe { transmute(call) };

            // TODO iree things are ref counted internall, so this is probably fine, but I hate it...
            let logp_func: Function<'static> = unsafe { transmute(logp_func) };

            let inputs = DynamicList::new(2, instance)?;
            let outputs = DynamicList::new(2, instance)?;

            LogpFunc {
                device: ManuallyDrop::new(device),
                inputs,
                outputs,
                logp_compiled: &model.logp_module,
                expand_compiled: &model.expand_module,
                ndim: 2,
                session: ManuallyDrop::new(session),
                //logp_func: ManuallyDrop::new(logp_func),
                buffer: vec![0.; 2].into(),
            }

        };

        let math = &mut math_obj;

        let position = vec![1., 2.];
        let mut gradient = vec![-1., -1.];

        let instance = get_instance()?;

        math.buffer
            .iter_mut()
            .zip(position.iter())
            .for_each(|(out, &val)| *out = val as _);

        let input_buffer = BufferView::<f32>::new(
            &math.session,
            &[position.len()],
            EncodingType::DenseRowMajor,
            &math.buffer,
        )
        .context("Could not create buffer view")?;

        (math.inputs)
            .push_ref(
                &input_buffer
                    .to_ref(instance)
                    .context("Could not dereference input buffer")?,
            )
            .context("Could not push input buffer to inputs")?;

        (math.logp_func)
            .invoke(&math.inputs, &math.outputs)
            .context("Could not invoke logp function")?;

        drop(input_buffer);

        let output_val: Value<f32> = (math.outputs)
            .get_value(0)
            .context("Could not extract logp value")?;
        let logp: f64 = output_val.from_value().into();

        let gradient_buffer_ref: Ref<BufferView<f32>> = (&math.outputs)
            .get_ref(1)
            .context("Could not get output buffer")?;
        let gradient_buffer = gradient_buffer_ref.to_buffer_view(&math.session);

        gradient_buffer
            .copy_into(&math.device, &mut math.buffer)
            .context("Could not copy gradient buffer from iree device")?;

        let mut has_bad_grad = false;
        gradient
            .iter_mut()
            .zip(math.buffer.iter())
            .for_each(|(out, &val)| {
                *out = val as f64;
                if !val.is_finite() {
                    has_bad_grad = true;
                }
            });

        drop(gradient_buffer_ref);
        drop(gradient_buffer);

        math.inputs.clear();
        math.outputs.clear();

        drop(math);
        drop(math_obj);

        /*
        drop(gradient_buffer);
        drop(gradient_buffer_ref);
        drop(math);
        drop(math_obj);
        drop(model);
        */

        */
        Ok(())
    }
}
