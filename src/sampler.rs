use std::{
    sync::{
        self,
        mpsc::{channel, RecvTimeoutError, Sender},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::{Result, Context};
use arrow2::array::Array;
use itertools::Itertools;
use nuts_rs::{new_sampler, ArrowBuilder, Chain, CpuLogpFunc, SampleStats, SamplerArgs};
use rand::{thread_rng, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

pub(crate) trait Trace: Send + Sync {
    fn append_value(&mut self, point: &[f64]) -> Result<()>;
    fn finalize(self) -> Result<Box<dyn Array>>;
}

pub(crate) trait Model: Send + Sync + 'static {
    type Density<'a>: CpuLogpFunc + 'a
    where
        Self: 'a;
    type Trace<'a>: Trace + 'a
    where
        Self: 'a;

    fn new_trace<'a, R: Rng + ?Sized>(
        &'a self,
        rng: &mut R,
        chain_id: u64,
        settings: &SamplerArgs,
    ) -> Result<Self::Trace<'a>>;
    fn density(&self) -> Result<Self::Density<'_>>;
    fn init_position<R: Rng + ?Sized>(&self, rng: &mut R, position: &mut [f64]) -> Result<()>;

    fn benchmark_logp(
        &self,
        point: &[f64],
        cores: usize,
        evals: usize,
    ) -> Result<Vec<Vec<Duration>>> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(cores)
            .thread_name(|idx| format!("benchmark-logp-{}", idx))
            .build()?;

        let (tx, rx) = channel();

        pool.scope(move |s| {
            for _ in 0..cores {
                let tx = tx.clone();

                s.spawn(move |_| {
                    let run = || {
                        let mut density = self.density()?;
                        let mut grad = vec![0f64; density.dim()];
                        let mut durations = Vec::with_capacity(evals);

                        for _ in 0..evals {
                            let start = Instant::now();
                            density.logp(point, &mut grad[..])?;
                            durations.push(start.elapsed())
                        }

                        Ok(durations)
                    };

                    tx.send(run())
                        .expect("Could not send results to main thread");
                });
            }
        });

        let results: Result<Vec<_>> = rx.into_iter().collect_vec().into_iter().collect();
        results
    }
}

pub(crate) struct Sampler {
    /// For each thread we return the chain id, the draws and the stats
    main_thread: thread::JoinHandle<Result<Vec<(u64, Box<dyn Array>, Option<Box<dyn Array>>)>>>,
    updates: sync::mpsc::Receiver<Box<dyn SampleStats>>,
}

impl Sampler {
    fn run_sampler<M: Model>(
        seed: Option<u64>,
        chain: u64,
        model: Arc<M>,
        settings: SamplerArgs,
        updates: Sender<Box<dyn SampleStats>>,
    ) -> Result<(u64, Box<dyn Array>, Option<Box<dyn Array>>)> {
        let mut rng = if let Some(seed) = seed {
            ChaCha8Rng::seed_from_u64(seed)
        } else {
            ChaCha8Rng::from_rng(thread_rng())?
        };
        rng.set_stream(chain);

        let logp = model.density().context("Failed to create model density")?;
        let dim = logp.dim();

        let mut sampler = new_sampler(logp, settings, chain, &mut rng);

        // TODO Merge stats_builder and trace
        // Create it outside of the function and put in Mutex,
        // so that the sampler can inspect the trace and ask
        // for partial results.
        let mut stats = sampler.stats_builder(dim, &settings);
        let mut trace = model
            .new_trace(&mut rng, chain, &settings)
            .context("Failed to create trace object")?;

        let mut initval = vec![0f64; dim];
        // TODO maxtries
        let mut error = None;
        for _ in 0..500 {
            model
                .init_position(&mut rng, &mut initval)
                .context("Failed to generate a new initial position")?;
            if let Err(err) = sampler.set_position(&initval) {
                error = Some(err);
                continue;
            }
            error = None;
            break;
        }

        if let Some(error) = error {
            let error: anyhow::Error = error.into();
            return Err(error.context("All initialization points failed"));
        }

        let draws = settings.num_tune + settings.num_draws;
        for _ in 0..draws {
            let (point, info) = sampler.draw().unwrap();
            stats.append_value(&info);
            trace.append_value(&point)?;
            // We do not handle this error. If the draws can not be send, this
            // could for instance be because the main thread was interrupted.
            // In this case we just want to return the draws we have so far.
            let result = updates.send(Box::new(info) as Box<dyn SampleStats>);
            if let Err(_) = result {
                break;
            }
        }

        Ok((
            chain,
            trace.finalize().context("Failed to finalize the trace object")?,
            stats.finalize().map(|x| x.boxed()),
        ))
    }

    pub(crate) fn new<M: Model>(
        model: M,
        settings: SamplerArgs,
        cores: usize,
        chains: u64,
        seed: Option<u64>,
    ) -> Self {
        let model = Arc::new(model);
        let (send_updates, updates) = channel();

        let thread = thread::spawn(move || {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(cores)
                .thread_name(|i| format!("nutpie-chain-{}", i))
                .build()
                .unwrap();

            let (tx, rx) = channel();

            for chain in 0..chains {
                let model = model.clone();
                let tx = tx.clone();

                let send_updates = send_updates.clone();

                pool.spawn(move || {
                    let _ = tx.send(Sampler::run_sampler(
                        seed,
                        chain,
                        model,
                        settings,
                        send_updates,
                    ));
                });
            }

            drop(tx);
            let outputs: Result<Vec<_>> = rx.into_iter().collect();
            outputs
        });

        Self {
            main_thread: thread,
            updates,
        }
    }

    pub(crate) fn is_finished(&self) -> bool {
        self.main_thread.is_finished()
    }
}

impl Sampler {
    pub fn next_draw_timeout(
        &mut self,
        timeout: Duration,
    ) -> Result<Box<dyn SampleStats>, RecvTimeoutError> {
        self.updates.recv_timeout(timeout)
    }

    pub fn finalize(self) -> Result<Vec<(Box<dyn Array>, Option<Box<dyn Array>>)>> {
        drop(self.updates);
        let vals = self.main_thread.join().unwrap();
        vals.map(|vals| {
            vals.into_iter()
                .sorted_by_key(|&(chain, _, _)| chain)
                .map(|(_, tr, stats)| (tr, stats))
                .collect()
        })
    }
}
