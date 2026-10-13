//! Bounded synchronous PyMC probe. Reuses Nutpie's model and nuts-rs chain.
use crate::{
    common::PyVariable,
    pymc::{ExpandFunc, LogpFunc, PyMcModel},
};
use anyhow::{bail, Result};
use nuts_rs::{Chain, DiagNutsSettings, Math, Model, Settings, Storable, Value};
use pyo3::prelude::*;
use rand::{rngs::StdRng, SeedableRng};
use std::{cell::Cell, collections::HashMap};
thread_local! { static EVALUATIONS: Cell<u64> = const { Cell::new(0) }; }
pub(crate) fn record_evaluation() {
    EVALUATIONS.with(|v| v.set(v.get() + 1));
}
type Draws = Vec<Vec<HashMap<String, Vec<f64>>>>;
type Stats = Vec<Vec<(bool, u64, f64)>>;

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
fn sample_pymc(model: &PyMcModel, draws: u64, tune: u64, chains: u64, seed: u64) -> Result<Draws> {
    Ok(run(model, draws, tune, chains, seed, 0.8, 10, None)?.0)
}

#[pyfunction]
#[pyo3(signature = (model, *, draws=500, tune=500, chains=2, seed=42, target_accept=0.9, max_depth=10, initial_positions=None))]
fn sample_pymc_stats(
    model: &PyMcModel,
    draws: u64,
    tune: u64,
    chains: u64,
    seed: u64,
    target_accept: f64,
    max_depth: u64,
    initial_positions: Option<Vec<Vec<f64>>>,
) -> Result<(Draws, Stats, u64, u64)> {
    run(
        model,
        draws,
        tune,
        chains,
        seed,
        target_accept,
        max_depth,
        initial_positions,
    )
}

fn run(
    model: &PyMcModel,
    draws: u64,
    tune: u64,
    chains: u64,
    seed: u64,
    target_accept: f64,
    max_depth: u64,
    initial_positions: Option<Vec<Vec<f64>>>,
) -> Result<(Draws, Stats, u64, u64)> {
    if draws == 0
        || tune == 0
        || chains == 0
        || !target_accept.is_finite()
        || !(0.0..1.0).contains(&target_accept)
        || target_accept == 0.0
        || !(1..=20).contains(&max_depth)
    {
        bail!("Invalid sampler settings");
    }
    if initial_positions
        .as_ref()
        .is_some_and(|v| v.len() != chains as usize)
    {
        bail!("One initial position is required per chain");
    }
    EVALUATIONS.with(|v| v.set(0));
    let mut result = Vec::new();
    let mut stats = Vec::new();
    let mut leapfrogs = 0;
    for chain_id in 0..chains {
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(chain_id));
        let math = model.math(&mut rng)?;
        let mut settings = DiagNutsSettings {
            num_tune: tune,
            num_draws: draws,
            maxdepth: max_depth,
            ..Default::default()
        };
        settings.adapt_options.step_size_settings.target_accept = target_accept;
        let mut chain = settings.new_chain(chain_id, math, &mut rng);
        let mut initial = vec![0.; chain.dim()];
        if let Some(positions) = &initial_positions {
            let point = &positions[chain_id as usize];
            if point.len() != initial.len() || point.iter().any(|v| !v.is_finite()) {
                bail!("Invalid initial position");
            }
            initial.copy_from_slice(point);
        } else {
            model.init_position(&mut rng, &mut initial)?;
        }
        chain.set_position(&initial)?;
        for _ in 0..tune {
            let (_, p) = chain.draw()?;
            leapfrogs += p.num_steps;
        }
        let mut values = Vec::new();
        let mut chain_stats = Vec::new();
        for _ in 0..draws {
            let (_, mut expanded, _, p) = chain.expanded_draw()?;
            leapfrogs += p.num_steps;
            chain_stats.push((p.diverging, p.num_steps, p.step_size));
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
        stats.push(chain_stats);
    }
    Ok((result, stats, EVALUATIONS.with(Cell::get), leapfrogs))
}

#[pymodule]
pub fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LogpFunc>()?;
    m.add_class::<ExpandFunc>()?;
    m.add_class::<PyMcModel>()?;
    m.add_class::<PyVariable>()?;
    m.add_function(wrap_pyfunction!(sample_pymc, m)?)?;
    m.add_function(wrap_pyfunction!(sample_pymc_stats, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_pymc, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
