use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use nuts_rs::cpu::{LogpFunc, State, StaticIntegrator};
use nuts_rs::nuts::{draw, Integrator};

type GradFunc =
    unsafe extern "C" fn(usize, *const f64, *mut f64, *mut f64, *const std::ffi::c_void) -> i64;
type UserData = *const std::ffi::c_void;

#[pyclass]
struct SampleInfo {}

struct PtrLogpFunc {
    func: GradFunc,
    user_data: UserData,
    dim: usize,
}

impl LogpFunc for PtrLogpFunc {
    type Err = i64;

    fn dim(&self) -> usize {
        self.dim
    }

    fn logp(&self, state: &mut State) -> Result<(), Self::Err> {
        let grad = (&mut state.grad).as_mut_ptr();
        let pos = (&state.q).as_ptr();
        let mut logp = 0f64;
        let logp_ptr = (&mut logp) as *mut f64;
        let func = self.func;
        let retcode = unsafe { func(self.dim, pos, grad, logp_ptr, self.user_data) };
        if retcode == 0 {
            state.potential_energy = -logp;
            return Ok(());
        }
        Err(retcode)
    }
}

impl PtrLogpFunc {
    unsafe fn new(func: GradFunc, user_data: UserData, dim: usize) -> PtrLogpFunc {
        PtrLogpFunc {
            func,
            user_data,
            dim,
        }
    }
}

#[pyclass(unsendable)]
struct PtrIntegrator {
    integrator: StaticIntegrator<PtrLogpFunc>,
    rng: rand::rngs::StdRng,
    maxdepth: u64,
    dim: usize,
}

#[pymethods]
impl PtrIntegrator {
    #[new]
    unsafe fn new(
        _py: Python,
        func: usize,
        user_data: usize,
        dim: usize,
        seed: u64,
        maxdepth: u64,
    ) -> PyResult<PtrIntegrator> {
        use rand::SeedableRng;

        let func: GradFunc = std::mem::transmute(func as *const std::ffi::c_void);
        let user_data = user_data as UserData;

        let func = PtrLogpFunc::new(func, user_data, dim);
        Ok(PtrIntegrator {
            integrator: StaticIntegrator::new(func, dim),
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            maxdepth,
            dim,
        })
    }

    fn draw(&mut self, init: usize, out: usize) -> PyResult<()> {
        let init: &[f64] = unsafe { std::slice::from_raw_parts(init as *const f64, self.dim) };
        let out: &mut [f64] = unsafe { std::slice::from_raw_parts_mut(out as *mut f64, self.dim) };
        let state = self
            .integrator
            .new_state(init)
            .map_err(|_| PyErr::new::<PyValueError, _>("Error initializing state"))?;
        let (state, _info) = draw(state, &mut self.rng, &mut self.integrator, self.maxdepth);
        self.integrator.write_position(&state, out);
        Ok(())
    }
}

struct PyLogpFunc {
    pyfunc: PyObject,
    dim: usize,
}

impl LogpFunc for PyLogpFunc {
    type Err = PyErr;

    fn dim(&self) -> usize {
        self.dim
    }

    fn logp(&self, state: &mut State) -> Result<(), Self::Err> {
        Python::with_gil(|py| {
            let out = numpy::PyArray1::from_slice(py, &mut state.grad);
            let pos = numpy::PyArray1::from_slice(py, &state.q);
            let logp = self.pyfunc.call1(py, (pos, out))?;
            state.potential_energy = -logp.extract::<f64>(py)?;
            Ok(())
        })
    }
}

impl PyLogpFunc {
    fn new(pyfunc: PyObject, dim: usize) -> PyLogpFunc {
        PyLogpFunc { pyfunc, dim }
    }
}

#[pyclass(unsendable)]
struct PyIntegrator {
    integrator: StaticIntegrator<PyLogpFunc>,
    rng: rand::rngs::StdRng,
    maxdepth: u64,
}

#[pymethods]
impl PyIntegrator {
    #[new]
    fn new(
        py: Python,
        pyfunc: PyObject,
        dim: usize,
        seed: u64,
        maxdepth: u64,
    ) -> PyResult<PyIntegrator> {
        use rand::SeedableRng;

        if !pyfunc.cast_as::<PyAny>(py)?.is_callable() {
            return Err(PyErr::new::<PyTypeError, _>("func must be callable."));
        }
        let func = PyLogpFunc::new(pyfunc, dim);
        Ok(PyIntegrator {
            integrator: StaticIntegrator::new(func, dim),
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            maxdepth,
        })
    }

    fn draw(
        &mut self,
        _py: Python,
        init: numpy::PyReadonlyArray1<f64>,
        out: &numpy::PyArray1<f64>,
    ) -> PyResult<()> {
        let state = self
            .integrator
            .new_state(&init.as_slice()?)
            .map_err(|_| PyErr::new::<PyValueError, _>("Error initializing state"))?;
        let (state, _info) = draw(state, &mut self.rng, &mut self.integrator, self.maxdepth);
        self.integrator
            .write_position(&state, unsafe { out.as_slice_mut() }?);
        Ok(())
    }
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn nuts_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_class::<PyIntegrator>()?;
    m.add_class::<PtrIntegrator>()?;
    Ok(())
}
