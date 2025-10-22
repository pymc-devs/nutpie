use std::{
    collections::BTreeMap,
    sync::{
        mpsc::{sync_channel, SyncSender},
        Arc,
    },
    thread::spawn,
    time::Duration,
};

use anyhow::{Context, Result};
use indicatif::{MultiProgress, ProgressBar, ProgressFinish, ProgressStyle};
use nuts_rs::{ChainProgress, ProgressCallback};
use pyo3::{Py, PyAny, Python};
use time_humanize::{Accuracy, Tense};
use upon::{Engine, Value};

pub struct ProgressHandler {
    engine: Engine<'static>,
    template: String,
    rate: Duration,
    n_cores: usize,
    updates: SyncSender<String>,
}

impl ProgressHandler {
    pub fn new(callback: Arc<Py<PyAny>>, rate: Duration, template: String, n_cores: usize) -> Self {
        let engine = Engine::new();

        let (update_tx, update_rx) = sync_channel(1);

        spawn(move || {
            // We keep an extra gil reference alive, to ensure the
            // python ThreadState is not destroyed.
            // See https://github.com/PyO3/pyo3/issues/5467
            Python::attach(move |py| {
                py.detach(move || loop {
                    let update = update_rx.recv();
                    let Ok(update) = update else { break };
                    let res = Python::attach(|py| callback.call1(py, (update,)));
                    if let Err(err) = res {
                        eprintln!("Error in progress callback: {err}");
                    }
                });
            });
        });

        Self {
            engine,
            rate,
            template,
            n_cores,
            updates: update_tx,
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
            let rendered = rendered.unwrap_or_else(|err| format!("{err}"));
            if let Err(e) = self.updates.send(rendered) {
                eprintln!("Could not send progress update: {e}");
                return;
            }
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

#[derive(PartialEq, Eq)]
enum ChainState {
    Normal,
    Divergences,
    Finished,
}

struct TerminalBar {
    pb: ProgressBar,
    last_position: u64,
    mode: ChainState,
    segment_style: String,
}

impl TerminalBar {
    pub fn new(mb: &MultiProgress, draws: u64) -> Self {
        let segment_style = "━━╸  ".to_string();
        let pb = mb
            .add(ProgressBar::new(draws))
            .with_finish(ProgressFinish::Abandon);
        pb.set_style(
            ProgressStyle::with_template("  {bar:35.blue}   {pos:10} {msg} {elapsed:10} {eta:10}")
                .unwrap()
                .progress_chars(&segment_style),
        );

        Self {
            pb,
            last_position: 0,
            mode: ChainState::Normal,
            segment_style,
        }
    }

    pub fn set_mode(&mut self, mode: ChainState) {
        if self.mode != mode {
            let color = match mode {
                ChainState::Normal => "blue",
                ChainState::Divergences => "red",
                ChainState::Finished => "green",
            };
            self.pb.set_style(
                ProgressStyle::with_template(&format!(
                    "  {{bar:35.{color}}}   {{pos:10}} {{msg}} {{elapsed:10}} {{eta:10}}"
                ))
                .unwrap()
                .progress_chars(&self.segment_style),
            );

            self.mode = mode
        }
    }

    pub fn is_finished(&self) -> bool {
        self.pb.is_finished()
    }

    pub fn finish(&mut self) {
        if self.mode != ChainState::Divergences {
            self.set_mode(ChainState::Finished);
        }
        self.pb.finish();
    }

    pub fn update_position(&mut self, chain: &ChainProgress) {
        let position = chain.finished_draws as u64;
        let delta = position.saturating_sub(self.last_position);
        if delta > 0 && !self.is_finished() {
            self.pb.set_position(position);
            self.pb.set_message(format!(
                "{:<12} {:<11.2} {:<12}",
                chain.divergences, chain.step_size, chain.latest_num_steps
            ));
            self.last_position = position;
        }
    }
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
        let multibar = MultiProgress::new();
        let mut bars = vec![];

        let header = multibar.add(ProgressBar::new(0));

        header.set_style(
            ProgressStyle::default_bar()
                .template("{msg:.bold}")
                .unwrap(),
        );
        header.set_message(format!(
            "  {:<35}   {:<10} {:<12} {:<11} {:<12} {:<10} {:<10}",
            "Progress", "Draws", "Divergences", "Step size", "Grad evals", "Elapsed", "Remaining"
        ));

        header.tick();

        let separator = multibar
            .add(ProgressBar::new(0))
            .with_finish(ProgressFinish::Abandon);
        separator.set_style(ProgressStyle::default_bar().template("{msg}").unwrap());
        separator.set_message(format!(" {}", "─".repeat(109)));
        separator.tick();

        let callback = move |_time_sampling, progress: Box<[ChainProgress]>| {
            if bars.is_empty() {
                for chain in progress.iter() {
                    bars.push(TerminalBar::new(&multibar, chain.total_draws as u64));
                }
            }

            if finished {
                return;
            }
            for (bar, chain) in bars.iter_mut().zip(progress.iter()) {
                if !bar.is_finished() && chain.finished_draws == chain.total_draws {
                    bar.pb.set_position(chain.total_draws as u64);
                    bar.finish();
                }
            }

            if progress
                .iter()
                .all(|chain| chain.finished_draws == chain.total_draws)
            {
                finished = true;
                header.finish();
                separator.finish();
            }

            for (bar, chain) in bars.iter_mut().zip(progress.iter()) {
                if chain.divergences > 0 {
                    bar.set_mode(ChainState::Divergences);
                }
                bar.update_position(chain);
            }
        };

        Ok(ProgressCallback {
            callback: Box::new(callback),
            rate: self.rate,
        })
    }
}
