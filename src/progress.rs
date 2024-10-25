use std::{collections::BTreeMap, sync::Arc, time::Duration};

use anyhow::{Context, Result};
use indicatif::ProgressBar;
use nuts_rs::{ChainProgress, ProgressCallback};
use pyo3::{Py, PyAny, Python};
use time_humanize::{Accuracy, Tense};
use upon::{Engine, Value};

pub struct ProgressHandler {
    engine: Engine<'static>,
    template: String,
    callback: Arc<Py<PyAny>>,
    rate: Duration,
    n_cores: usize,
}

impl ProgressHandler {
    pub fn new(callback: Arc<Py<PyAny>>, rate: Duration, template: String, n_cores: usize) -> Self {
        let engine = Engine::new();
        Self {
            engine,
            callback,
            rate,
            template,
            n_cores,
        }
    }

    pub fn into_callback(self) -> Result<ProgressCallback> {
        let template = self
            .engine
            .compile(self.template)
            .context("Could not compile progress template")?;

        let mut finished = false;
        let mut progress_update_count = 0;

        let callback = move |time_sampling, progress: Box<[ChainProgress]>| {
            if finished {
                return;
            }
            if progress
                .iter()
                .all(|chain| chain.finished_draws == chain.total_draws)
            {
                finished = true;
            }
            let progress =
                progress_to_value(progress_update_count, self.n_cores, time_sampling, progress);
            let rendered = template.render_from(&self.engine, &progress).to_string();
            let rendered = rendered.unwrap_or_else(|err| format!("{}", err));
            let _ = Python::with_gil(|py| self.callback.call1(py, (rendered,)));
            progress_update_count += 1;
        };

        Ok(ProgressCallback {
            callback: Box::new(callback),
            rate: self.rate,
        })
    }
}

fn progress_to_value(
    progress_update_count: usize,
    n_cores: usize,
    time_sampling: Duration,
    progress: Box<[ChainProgress]>,
) -> Value {
    let chains: Vec<_> = progress
        .iter()
        .enumerate()
        .map(|(i, chain)| {
            let mut values = BTreeMap::new();
            values.insert("chain_index".into(), Value::Integer(i as i64));
            values.insert(
                "finished_draws".into(),
                Value::Integer(chain.finished_draws as i64),
            );
            values.insert(
                "total_draws".into(),
                Value::Integer(chain.total_draws as i64),
            );
            values.insert(
                "divergences".into(),
                Value::Integer(chain.divergences as i64),
            );
            values.insert("tuning".into(), Value::Bool(chain.tuning));
            values.insert("started".into(), Value::Bool(chain.started));
            values.insert(
                "finished".into(),
                Value::Bool(chain.total_draws == chain.finished_draws),
            );
            values.insert(
                "latest_num_steps".into(),
                Value::Integer(chain.latest_num_steps as i64),
            );
            values.insert(
                "total_num_steps".into(),
                Value::Integer(chain.total_num_steps as i64),
            );
            values.insert(
                "step_size".into(),
                Value::String(format!("{:.2}", chain.step_size)),
            );
            values.insert(
                "divergent_draws".into(),
                Value::List(
                    chain
                        .divergent_draws
                        .iter()
                        .map(|&idx| Value::Integer(idx as _))
                        .collect(),
                ),
            );
            upon::Value::Map(values)
        })
        .collect();

    let mut map = BTreeMap::new();
    map.insert("chains".into(), Value::List(chains));
    map.insert(
        "total_draws".into(),
        Value::Integer(
            progress
                .iter()
                .map(|chain| chain.total_draws)
                .sum::<usize>() as i64,
        ),
    );
    map.insert(
        "total_finished_draws".into(),
        Value::Integer(
            progress
                .iter()
                .map(|chain| chain.finished_draws)
                .sum::<usize>() as i64,
        ),
    );
    map.insert(
        "time_sampling".into(),
        Value::String(
            time_humanize::HumanTime::from(time_sampling)
                .to_text_en(Accuracy::Rough, Tense::Present),
        ),
    );

    let remaining = estimate_remaining_time(n_cores, time_sampling, &progress);
    map.insert(
        "time_remaining_estimate".into(),
        match remaining {
            Some(remaining) => Value::String(
                time_humanize::HumanTime::from(remaining)
                    .to_text_en(Accuracy::Rough, Tense::Present),
            ),
            None => Value::None,
        },
    );

    map.insert("num_cores".into(), Value::Integer(n_cores as _));

    let finished_chains = progress
        .iter()
        .map(|chain| (chain.finished_draws == chain.total_draws) as u64)
        .sum::<u64>();
    map.insert(
        "finished_chains".into(),
        Value::Integer(finished_chains as _),
    );
    map.insert(
        "running_chains".into(),
        Value::Integer(
            progress
                .iter()
                .map(|chain| (chain.started & (chain.finished_draws < chain.total_draws)) as u64)
                .sum::<u64>() as i64,
        ),
    );
    map.insert("num_chains".into(), Value::Integer(progress.len() as _));
    map.insert(
        "finished".into(),
        Value::Bool(progress.len() == finished_chains as usize),
    );
    map.insert(
        "progress_update_count".into(),
        Value::Integer(progress_update_count as i64),
    );

    Value::Map(map)
}

fn estimate_remaining_time(
    n_cores: usize,
    time_sampling: Duration,
    progress: &[ChainProgress],
) -> Option<Duration> {
    let finished_draws: u64 = progress
        .iter()
        .map(|chain| chain.finished_draws as u64)
        .sum();
    if finished_draws == 0 {
        return None;
    }

    let finished_draws = finished_draws as f64;

    // TODO this assumes that so far all cores were used all the time
    let time_per_draw = time_sampling.mul_f64((n_cores as f64) / finished_draws);

    let mut core_times = vec![Duration::ZERO; n_cores];

    progress
        .iter()
        .map(|chain| time_per_draw.mul_f64((chain.total_draws - chain.finished_draws) as f64))
        .for_each(|time| {
            let min_index = core_times
                .iter()
                .enumerate()
                .min_by_key(|&(_, v)| v)
                .unwrap()
                .0;
            core_times[min_index] += time;
        });

    Some(core_times.into_iter().max().unwrap_or(Duration::ZERO))
}

pub struct IndicatifHandler {
    rate: Duration,
}

impl IndicatifHandler {
    pub fn new(rate: Duration) -> Self {
        Self { rate }
    }

    pub fn into_callback(self) -> Result<ProgressCallback> {
        let mut finished = false;
        let mut last_draws = 0;
        let mut bar = None;

        let callback = move |_time_sampling, progress: Box<[ChainProgress]>| {
            let total: u64 = progress.iter().map(|chain| chain.total_draws as u64).sum();

            if bar.is_none() {
                bar = Some(ProgressBar::new(total));
            }

            let Some(ref bar) = bar else { unreachable!() };

            if finished {
                return;
            }
            if progress
                .iter()
                .all(|chain| chain.finished_draws == chain.total_draws)
            {
                finished = true;
                bar.set_position(total);
                bar.finish();
            }

            let finished_draws: u64 = progress
                .iter()
                .map(|chain| chain.finished_draws as u64)
                .sum();

            let delta = finished_draws.saturating_sub(last_draws);
            if delta > 0 {
                bar.set_position(finished_draws);
                last_draws = finished_draws;
            }
        };

        Ok(ProgressCallback {
            callback: Box::new(callback),
            rate: self.rate,
        })
    }
}
