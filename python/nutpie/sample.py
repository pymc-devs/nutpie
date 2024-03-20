import os
from dataclasses import dataclass
from threading import Condition, Event
from typing import Any, Literal, Optional, overload

import arviz
import fastprogress
import numpy as np
import pandas as pd
import pyarrow
from fastprogress.fastprogress import ConsoleProgressBar

from nutpie import _lib


@dataclass(frozen=True)
class CompiledModel:
    dims: Optional[dict[str, tuple[str, ...]]]

    @property
    def n_dim(self) -> int:
        raise NotImplementedError()

    @property
    def shapes(self) -> Optional[dict[str, tuple[int, ...]]]:
        raise NotImplementedError()

    @property
    def coords(self):
        raise NotImplementedError()

    def _make_sampler(self, *args, **kwargs):
        raise NotImplementedError()

    def _make_model(self, *args, **kwargs):
        raise NotImplementedError()

    def benchmark_logp(self, point, num_evals, cores):
        """Time how long the logp gradient evaluation takes.

        # Parameters
        """
        model = self._make_model(point)
        times = []
        if isinstance(cores, int):
            cores = [cores]
        for num_cores in cores:
            if num_cores == 0:
                continue
            flat = model.benchmark_logp(point, num_cores, num_evals)
            data = pd.DataFrame(flat)
            data.index = pd.MultiIndex.from_product(
                [range(num_cores), [num_cores]], names=["thread", "concurrent_cores"]
            )
            data = data.rename_axis(columns="evaluation")
            times.append(data)
        return pd.concat(times)


def _trace_to_arviz(traces, n_tune, shapes, **kwargs):
    n_chains = len(traces)

    data_dict = {}
    data_dict_tune = {}
    stats_dict = {}
    stats_dict_tune = {}

    draw_batches = []
    stats_batches = []
    for draws, stats in traces:
        draw_batches.append(pyarrow.RecordBatch.from_struct_array(draws))
        stats_batches.append(pyarrow.RecordBatch.from_struct_array(stats))

    table = pyarrow.Table.from_batches(draw_batches)
    table_stats = pyarrow.Table.from_batches(stats_batches)
    for name, col in zip(table.column_names, table.columns):
        lengths = [len(chunk) for chunk in col.chunks]
        length = max(lengths)
        dtype = col.chunks[0].values.to_numpy().dtype
        if dtype in [np.float64, np.float32]:
            data = np.full(
                (n_chains, length, *tuple(shapes[name])), np.nan, dtype=dtype
            )
        else:
            data = np.zeros((n_chains, length, *tuple(shapes[name])), dtype=dtype)
        for i, chunk in enumerate(col.chunks):
            data[i, : len(chunk)] = chunk.values.to_numpy().reshape(
                (len(chunk),) + shapes[name]
            )

        data_dict[name] = data[:, n_tune:]
        data_dict_tune[name] = data[:, :n_tune]

    for name, col in zip(table_stats.column_names, table_stats.columns):
        if name in ["chain", "draw", "divergence_message"]:
            continue
        col_type = col.type
        if hasattr(col_type, "list_size"):
            last_shape = (col_type.list_size,)
            dtype = col_type.field(0).type.to_pandas_dtype()
        else:
            dtype = col_type.to_pandas_dtype()
            last_shape = ()

        lengths = [len(chunk) for chunk in col.chunks]
        length = max(lengths)

        if dtype in [np.float64, np.float32]:
            data = np.full((n_chains, length, *last_shape), np.nan, dtype=dtype)
        else:
            data = np.zeros((n_chains, length, *last_shape), dtype=dtype)

        for i, chunk in enumerate(col.chunks):
            if hasattr(chunk, "values"):
                values = chunk.values.to_numpy(False)
            else:
                values = chunk.to_numpy(False)
            data[i, : len(chunk)] = values.reshape((len(chunk), *last_shape))
            stats_dict[name] = data[:, n_tune:]
            stats_dict_tune[name] = data[:, :n_tune]

    return arviz.from_dict(
        data_dict,
        sample_stats=stats_dict,
        warmup_posterior=data_dict_tune,
        warmup_sample_stats=stats_dict_tune,
        **kwargs,
    )


class _BackgroundSampler:
    _sampler: Any
    _num_divs: int
    _tune: int
    _draws: int
    _chains: int
    _chains_finished: int
    _compiled_model: CompiledModel
    _save_warmup: bool
    _progress = ConsoleProgressBar

    def __init__(
        self,
        compiled_model,
        settings,
        init_mean,
        chains,
        cores,
        seed,
        draws,
        tune,
        *,
        progress_bar=True,
        save_warmup=True,
        return_raw_trace=False,
    ):
        self._num_divs = 0
        self._tune = settings.num_tune
        self._draws = settings.num_draws
        self._settings = settings
        self._chains_tuning = chains
        self._chains_finished = 0
        self._chains = chains
        self._compiled_model = compiled_model
        self._save_warmup = save_warmup
        self._return_raw_trace = return_raw_trace
        total_draws = (self._draws + self._tune) * self._chains
        self._progress = fastprogress.progress_bar(
            range(total_draws),
            total=total_draws,
            display=progress_bar,
        )
        # fastprogress seems to reset the progress bar
        # if we create a new iterator, but we don't want
        # this for multiple calls to wait.
        self._bar = iter(self._progress)

        self._exit_event = Event()
        self._pause_event = Event()
        self._continue = Condition()

        self._finished_draws = 0

        next(self._bar)

        def progress_callback(info):
            if info.draw == self._tune - 1:
                self._chains_tuning -= 1
            if info.draw == self._tune + self._draws - 1:
                self._chains_finished += 1
            if info.is_diverging and info.draw > self._tune:
                self._num_divs += 1
            if self._chains_tuning > 0:
                count = self._chains_tuning
                divs = self._num_divs
                self._progress.comment = (
                    f" Chains in warmup: {count}, Divergences: {divs}"
                )
            else:
                count = self._chains - self._chains_finished
                divs = self._num_divs
                self._progress.comment = (
                    f" Sampling chains: {count}, Divergences: {divs}"
                )
            try:
                next(self._bar)
            except StopIteration:
                pass
            self._finished_draws += 1

        if progress_bar:
            callback = progress_callback
        else:
            callback = None

        self._sampler = compiled_model._make_sampler(
            settings,
            init_mean,
            chains,
            cores,
            seed,
            callback=callback,
        )

    def wait(self, *, timeout=None):
        """Wait until sampling is finished and return the trace.

        KeyboardInterrupt will lead to interrupt the waiting.

        This will return after `timeout` seconds even if sampling is
        not finished at this point.

        This resumes the sampler in case it had been paused.
        """
        self._sampler.wait(timeout)
        self._sampler.finalize()
        return self._extract()

    def _extract(self):
        results = self._sampler.extract_results()

        dims = {name: list(dim) for name, dim in self._compiled_model.dims.items()}
        dims["mass_matrix_inv"] = ["unconstrained_parameter"]
        dims["gradient"] = ["unconstrained_parameter"]
        dims["unconstrained_draw"] = ["unconstrained_parameter"]
        dims["divergence_start"] = ["unconstrained_parameter"]
        dims["divergence_start_gradient"] = ["unconstrained_parameter"]
        dims["divergence_end"] = ["unconstrained_parameter"]
        dims["divergence_momentum"] = ["unconstrained_parameter"]

        if self._return_raw_trace:
            return results
        else:
            return _trace_to_arviz(
                results,
                self._tune,
                self._compiled_model.shapes,
                dims=dims,
                coords={
                    name: pd.Index(vals)
                    for name, vals in self._compiled_model.coords.items()
                },
                save_warmup=self._save_warmup,
            )

    def pause(self):
        """Pause the sampler."""
        self._sampler.pause()

    def resume(self):
        """Resume a paused sampler."""
        self._sampler.resume()

    @property
    def is_finished(self):
        return self._sampler.is_finished()

    def abort(self):
        """Abort sampling and return the trace produced so far."""
        self._sampler.abort()
        return self._extract()

    def cancel(self):
        """Abort sampling and discard progress."""
        self._sampler.abort()

    def __del__(self):
        if not self._sampler.is_empty():
            self.cancel()


@overload
def sample(
    compiled_model: CompiledModel,
    *,
    draws: int,
    tune: int,
    chains: int,
    cores: Optional[int],
    seed: Optional[int],
    save_warmup: bool,
    progress_bar: bool,
    init_mean: Optional[np.ndarray],
    return_raw_trace: bool,
    blocking: Literal[True],
    **kwargs,
) -> arviz.InferenceData: ...


@overload
def sample(
    compiled_model: CompiledModel,
    *,
    draws: int,
    tune: int,
    chains: int,
    cores: Optional[int],
    seed: Optional[int],
    save_warmup: bool,
    progress_bar: bool,
    init_mean: Optional[np.ndarray],
    return_raw_trace: bool,
    blocking: Literal[False],
    **kwargs,
) -> _BackgroundSampler: ...


def sample(
    compiled_model: CompiledModel,
    *,
    draws: int = 1000,
    tune: int = 300,
    chains: int = 6,
    cores: Optional[int] = None,
    seed: Optional[int] = None,
    save_warmup: bool = True,
    progress_bar: bool = True,
    init_mean: Optional[np.ndarray] = None,
    return_raw_trace: bool = False,
    blocking: bool = True,
    **kwargs,
) -> arviz.InferenceData:
    """Sample the posterior distribution for a compiled model.

    Parameters
    ----------
    draws: int
        The number of draws after tuning in each chain.
    tune: int
        The number of tuning (warmup) draws in each chain.
    chains: int
        The number of chains to sample.
    cores: int
        The number of chains that should run in parallel.
    seed: int
        Seed for the randomness in sampling.
    num_try_init: int
        The number if initial positions for each chain to try.
        Fail if we can't find a valid initializion point after
        this many tries.
    save_warmup: bool
        Wether to save the tuning (warmup) statistics and
        posterior draws in the output dataset.
    store_divergences: bool
        If true, store the exact locations where diverging
        transitions happend in the sampler stats. This is currently
        experimental, as the implementation is very wastefull
        with memory, and a better interface will need breaking
        changes.
    progress_bar: bool
        If true, display the progress bar (default)
    init_mean: ndarray
        Initialize the chains using jittered values around this
        point on the transformed parameter space. Defaults to
        zeros.
    store_unconstrained: bool
        If True, store each draw in the unconstrained (transformed)
        space in the sample stats.
    store_gradient: bool
        If True, store the logp gradient of each draw in the unconstrained
        space in the sample stats.
    store_mass_matrix: bool
        If True, store the current mass matrix at each draw in
        the sample stats.
    target_accept: float between 0 and 1, default 0.8
        Adapt the step size of the integrator so that the average
        acceptance probability of the draws is `target_accept`.
        Larger values will decrease the step size of the integrator,
        which can help when sampling models with bad geometry.
    maxdepth: int, default=10
        The maximum depth of the tree for each draw. The maximum
        number of gradient evaluations for each draw will
        be 2 ^ maxdepth.
    return_raw_trace: bool, default=False
        Return the raw trace object (an apache arrow structure)
        instead of converting to arviz.
    **kwargs
        Pass additional arguments to nutpie._lib.PySamplerArgs

    Returns
    -------
    trace : arviz.InferenceData
        An ArviZ ``InferenceData`` object that contains the samples.
    """
    settings = _lib.PySamplerArgs()
    settings.num_tune = tune
    settings.num_draws = draws

    for name, val in kwargs.items():
        setattr(settings, name, val)

    if cores is None:
        try:
            # Only available in python>=3.13
            available = os.process_cpu_count()
        except AttributeError:
            available = os.cpu_count()
        cores = min(chains, available)

    if init_mean is None:
        init_mean = np.zeros(compiled_model.n_dim)

    sampler = _BackgroundSampler(
        compiled_model,
        settings,
        init_mean,
        chains,
        cores,
        seed,
        draws,
        tune,
        progress_bar=progress_bar,
        save_warmup=save_warmup,
        return_raw_trace=return_raw_trace,
    )

    if not blocking:
        return sampler

    try:
        result = sampler.wait()
    except KeyboardInterrupt:
        result = sampler.abort()
    except:
        sampler.cancel()
        raise

    return result
