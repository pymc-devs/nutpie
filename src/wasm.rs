//! Bounded synchronous PyMC probe. Reuses Nutpie's model and nuts-rs chain.
use crate::{
    common::PyVariable,
    pymc::{ExpandFunc, LogpFunc, PyMcModel},
};
use anyhow::{bail, Result};
use nuts_rs::{Chain, DiagNutsSettings, Math, Model, Settings, Storable, Value};
use pyo3::prelude::*;
use rand::{rngs::StdRng, SeedableRng};
use std::collections::HashMap;

#[pyfunction]
fn evaluate_pymc(
    model: &PyMcModel,
    position: Vec<f64>,
) -> Result<(f64, Vec<f64>, HashMap<String, Vec<f64>>)> {
    let mut rng = StdRng::seed_from_u64(0);
    let mut math = model.math(&mut rng)?;
    if position.len() != math.dim() {
        bail!("Position dimension does not match model");
    }
    let mut gradient = vec![0.; position.len()];
    let logp = math.logp(&position, &mut gradient)?;
    let mut array = math.new_array();
    math.read_from_slice(&mut array, &position);
    let mut expanded = math.expand_vector(&mut rng, &array)?;
    let mut values = HashMap::new();
    for (name, value) in expanded.get_all(&math) {
        match value {
            Some(Value::F64(value)) => {
                values.insert(name.to_owned(), value.to_vec());
            }
            _ => bail!("The bounded WASM probe only supports float64 outputs"),
        }
    }
    Ok((logp, gradient, values))
}

#[pyfunction]
#[pyo3(signature = (model, *, draws=500, tune=500, chains=2, seed=42))]
fn sample_pymc(
    model: &PyMcModel,
    draws: u64,
    tune: u64,
    chains: u64,
    seed: u64,
) -> Result<Vec<Vec<HashMap<String, Vec<f64>>>>> {
    if draws == 0 || tune == 0 || chains == 0 {
        bail!("draws, tune and chains must be positive");
    }
    let mut result = Vec::new();
    for chain_id in 0..chains {
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(chain_id));
        let math = model.math(&mut rng)?;
        let settings = DiagNutsSettings {
            num_tune: tune,
            num_draws: draws,
            ..Default::default()
        };
        let mut chain = settings.new_chain(chain_id, math, &mut rng);
        let mut initial = vec![0.; chain.dim()];
        model.init_position(&mut rng, &mut initial)?;
        chain.set_position(&initial)?;
        for _ in 0..tune {
            chain.draw()?;
        }
        let mut values = Vec::new();
        for _ in 0..draws {
            let (_, mut expanded, _, _) = chain.expanded_draw()?;
            let math = chain.math();
            let mut row = HashMap::new();
            for (name, value) in expanded.get_all(&*math) {
                match value {
                    Some(Value::F64(value)) => {
                        row.insert(name.to_owned(), value.to_vec());
                    }
                    _ => bail!("The bounded WASM probe only supports float64 outputs"),
                }
            }
            values.push(row);
        }
        result.push(values);
    }
    Ok(result)
}

#[pymodule]
pub fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LogpFunc>()?;
    m.add_class::<ExpandFunc>()?;
    m.add_class::<PyMcModel>()?;
    m.add_class::<PyVariable>()?;
    m.add_function(wrap_pyfunction!(sample_pymc, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_pymc, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
