use anyhow::{bail, Context, Result};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::{intern, prelude::*, types::PyList};
use std::sync::Arc;

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
