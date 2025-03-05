import os
from dataclasses import dataclass
from typing import Any, Literal, Optional, cast, overload

import arviz
import numpy as np
import pandas as pd
import pyarrow

from nutpie import _lib  # type: ignore


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


_progress_style = """
<style>
    :root {
        --column-width-1: 40%; /* Progress column width */
        --column-width-2: 15%; /* Chain column width */
        --column-width-3: 15%; /* Divergences column width */
        --column-width-4: 15%; /* Step Size column width */
        --column-width-5: 15%; /* Gradients/Draw column width */
    }

    .nutpie {
        max-width: 800px;
        margin: 10px auto;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        //color: #333;
        //background-color: #fff;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-radius: 8px;
        font-size: 14px; /* Smaller font size for a more compact look */
    }
    .nutpie table {
        width: 100%;
        border-collapse: collapse; /* Remove any extra space between borders */
    }
    .nutpie th, .nutpie td {
        padding: 8px 10px; /* Reduce padding to make table more compact */
        text-align: left;
        border-bottom: 1px solid #888;
    }
    .nutpie th {
        //background-color: #f0f0f0;
    }

    .nutpie th:nth-child(1) { width: var(--column-width-1); }
    .nutpie th:nth-child(2) { width: var(--column-width-2); }
    .nutpie th:nth-child(3) { width: var(--column-width-3); }
    .nutpie th:nth-child(4) { width: var(--column-width-4); }
    .nutpie th:nth-child(5) { width: var(--column-width-5); }

    .nutpie progress {
        width: 100%;
        height: 15px; /* Smaller progress bars */
        border-radius: 5px;
    }
    progress::-webkit-progress-bar {
        background-color: #eee;
        border-radius: 5px;
    }
    progress::-webkit-progress-value {
        background-color: #5cb85c;
        border-radius: 5px;
    }
    progress::-moz-progress-bar {
        background-color: #5cb85c;
        border-radius: 5px;
    }
    .nutpie .progress-cell {
        width: 100%;
    }

    .nutpie p strong { font-size: 16px; font-weight: bold; }

    @media (prefers-color-scheme: dark) {
        .nutpie {
            //color: #ddd;
            //background-color: #1e1e1e;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        }
        .nutpie table, .nutpie th, .nutpie td {
            border-color: #555;
            color: #ccc;
        }
        .nutpie th {
            background-color: #2a2a2a;
        }
        .nutpie progress::-webkit-progress-bar {
            background-color: #444;
        }
        .nutpie progress::-webkit-progress-value {
            background-color: #3178c6;
        }
        .nutpie progress::-moz-progress-bar {
            background-color: #3178c6;
        }
    }
</style>
"""


_progress_template = """
<div class="nutpie">
    <p><strong>Sampler Progress</strong></p>
    <p>Total Chains: <span id="total-chains">{{ num_chains }}</span></p>
    <p>Active Chains: <span id="active-chains">{{ running_chains }}</span></p>
    <p>
        Finished Chains:
        <span id="active-chains">{{ finished_chains }}</span>
    </p>
    <p>Sampling for {{ time_sampling }}</p>
    <p>
        Estimated Time to Completion:
        <span id="eta">{{ time_remaining_estimate }}</span>
    </p>

    <progress
        id="total-progress-bar"
        max="{{ total_draws }}"
        value="{{ total_finished_draws }}">
    </progress>
    <table>
        <thead>
            <tr>
                <th>Progress</th>
                <th>Draws</th>
                <th>Divergences</th>
                <th>Step Size</th>
                <th>Gradients/Draw</th>
            </tr>
        </thead>
        <tbody id="chain-details">
            {% for chain in chains %}
                <tr>
                    <td class="progress-cell">
                        <progress
                            max="{{ chain.total_draws }}"
                            value="{{ chain.finished_draws }}">
                        </progress>
                    </td>
                    <td>{{ chain.finished_draws }}</td>
                    <td>{{ chain.divergences }}</td>
                    <td>{{ chain.step_size }}</td>
                    <td>{{ chain.latest_num_steps }}</td>
                </tr>
            {% endfor %}
            </tr>
        </tbody>
    </table>
</div>
"""


# Adapted from fastprogress
def in_notebook():
    def in_colab():
        "Check if the code is running in Google Colaboratory"
        try:
            from google import colab  # noqa: F401

            return True
        except ImportError:
            return False

    if in_colab():
        return True
    try:
        shell = get_ipython().__class__.__name__  # type: ignore
        if shell == "ZMQInteractiveShell":  # Jupyter notebook, Spyder or qtconsole
            try:
                from IPython.display import (
                    HTML,  # noqa: F401
                    clear_output,  # noqa: F401
                    display,  # noqa: F401
                )

                return True
            except ImportError:
                import warnings

                warnings.warn(
                    "Couldn't import ipywidgets properly, "
                    "progress bar will be disabled",
                    stacklevel=2,
                )
                return False
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


class _BackgroundSampler:
    _sampler: Any
    _num_divs: int
    _tune: int
    _draws: int
    _chains: int
    _chains_finished: int
    _compiled_model: CompiledModel
    _save_warmup: bool

    def __init__(
        self,
        compiled_model,
        settings,
        init_mean,
        cores,
        *,
        progress_bar=True,
        save_warmup=True,
        return_raw_trace=False,
        progress_template=None,
        progress_style=None,
        progress_rate=100,
    ):
        self._settings = settings
        self._compiled_model = compiled_model
        self._save_warmup = save_warmup
        self._return_raw_trace = return_raw_trace

        self._html = None

        if not progress_bar:
            progress_type = _lib.ProgressType.none()

        elif in_notebook():
            if progress_template is None:
                progress_template = _progress_template

            if progress_style is None:
                progress_style = _progress_style

            import IPython

            self._html = ""

            if progress_style is not None:
                IPython.display.display(IPython.display.HTML(progress_style))

            self.display_id = IPython.display.display(self, display_id=True)

            def callback(formatted):
                self._html = formatted
                self.display_id.update(self)

            progress_type = _lib.ProgressType.template_callback(
                progress_rate, progress_template, cores, callback
            )
        else:
            progress_type = _lib.ProgressType.indicatif(progress_rate)

        self._sampler = compiled_model._make_sampler(
            settings,
            init_mean,
            cores,
            progress_type,
        )

    def wait(self, *, timeout=None):
        """Wait until sampling is finished and return the trace.

        KeyboardInterrupt will lead to interrupt the waiting.

        This will return after `timeout` seconds even if sampling is
        not finished at this point.

        This resumes the sampler in case it had been paused.
        """
        self._sampler.wait(timeout)
        results = self._sampler.extract_results()
        return self._extract(results)

    def _extract(self, results):
        dims = {name: list(dim) for name, dim in self._compiled_model.dims.items()}
        dims["mass_matrix_inv"] = ["unconstrained_parameter"]
        dims["gradient"] = ["unconstrained_parameter"]
        dims["unconstrained_draw"] = ["unconstrained_parameter"]
        dims["divergence_start"] = ["unconstrained_parameter"]
        dims["divergence_start_gradient"] = ["unconstrained_parameter"]
        dims["divergence_end"] = ["unconstrained_parameter"]
        dims["divergence_momentum"] = ["unconstrained_parameter"]
        dims["transformed_gradient"] = ["unconstrained_parameter"]
        dims["transformed_position"] = ["unconstrained_parameter"]

        if self._return_raw_trace:
            return results
        else:
            return _trace_to_arviz(
                results,
                self._settings.num_tune,
                self._compiled_model.shapes,
                dims=dims,
                coords={
                    name: pd.Index(vals)
                    for name, vals in self._compiled_model.coords.items()
                },
                save_warmup=self._save_warmup,
            )

    def inspect(self):
        """Get a copy of the current state of the trace"""
        results = self._sampler.inspect()
        return self._extract(results)

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
        results = self._sampler.extract_results()
        return self._extract(results)

    def cancel(self):
        """Abort sampling and discard progress."""
        self._sampler.abort()

    def __del__(self):
        if not self._sampler.is_empty():
            self.cancel()

    def _repr_html_(self):
        return self._html


@overload
def sample(
    compiled_model: CompiledModel,
    *,
    draws: int | None,
    tune: int | None,
    chains: int,
    cores: Optional[int],
    seed: Optional[int],
    save_warmup: bool,
    progress_bar: bool,
    low_rank_modified_mass_matrix: bool = False,
    transform_adapt: bool = False,
    init_mean: Optional[np.ndarray],
    return_raw_trace: bool,
    blocking: Literal[True],
    **kwargs,
) -> arviz.InferenceData: ...


@overload
def sample(
    compiled_model: CompiledModel,
    *,
    draws: int | None,
    tune: int | None,
    chains: int,
    cores: Optional[int],
    seed: Optional[int],
    save_warmup: bool,
    progress_bar: bool,
    low_rank_modified_mass_matrix: bool = False,
    transform_adapt: bool = False,
    init_mean: Optional[np.ndarray],
    return_raw_trace: bool,
    blocking: Literal[False],
    **kwargs,
) -> _BackgroundSampler: ...


def sample(
    compiled_model: CompiledModel,
    *,
    draws: int | None = None,
    tune: int | None = None,
    chains: int = 6,
    cores: Optional[int] = None,
    seed: Optional[int] = None,
    save_warmup: bool = True,
    progress_bar: bool = True,
    low_rank_modified_mass_matrix: bool = False,
    transform_adapt: bool = False,
    init_mean: Optional[np.ndarray] = None,
    return_raw_trace: bool = False,
    blocking: bool = True,
    progress_template: Optional[str] = None,
    progress_style: Optional[str] = None,
    progress_rate: int = 100,
    **kwargs,
) -> arviz.InferenceData:
    """Sample the posterior distribution for a compiled model.

    Parameters
    ----------
    draws: int | None
        The number of draws after tuning in each chain.
    tune: int | None
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
    use_grad_based_mass_matrix: bool, default=True
        Use a mass matrix estimate that is based on draw and gradient
        variance. Set to `False` to get mass matrix adaptation more
        similar to PyMC and Stan.
    progress_template: str
        This is only exposed for experimentation. upon template
        for the html progress representation.
    progress_style: str
        This is only exposed for experimentation. Common HTML
        for the progress bar (eg CSS).
    progress_rate: int, default=500
        Rate in ms at which the progress should be updated.
    low_rank_modified_mass_matrix: bool, default=False
        Allow adaptation to some posterior correlations using
        a low-rank updated mass matrix. This is *experimental*
        and details about this will probably change in the next
        release.
    mass_matrix_eigval_cutoff: float > 1, defaul=100
        Ignore eigenvalues between cutoff and 1/cutoff in the
        low-rank modified mass matrix estimate. Higher values
        lead to worse correclation fitting, but increase
        the performance of leapfrog steps.
    mass_matrix_gamma: float > 0, default=1e-5
        Regularisation parameter for the eigenvalues. Only
        applicable with low_rank_modified_mass_matrix=True.
    transform_adapt: bool, default=False
        Use the experimental transform adaptation algorithm
        during tuning.
    **kwargs
        Pass additional arguments to nutpie._lib.PySamplerArgs

    Returns
    -------
    trace : arviz.InferenceData
        An ArviZ ``InferenceData`` object that contains the samples.
    """

    if low_rank_modified_mass_matrix and transform_adapt:
        raise ValueError(
            "Specify only one of `low_rank_modified_mass_matrix` and `transform_adapt`"
        )

    if low_rank_modified_mass_matrix:
        settings = _lib.PyNutsSettings.LowRank(seed)
    elif transform_adapt:
        settings = _lib.PyNutsSettings.Transform(seed)
    else:
        settings = _lib.PyNutsSettings.Diag(seed)

    if tune is not None:
        settings.num_tune = tune
    if draws is not None:
        settings.num_draws = draws
    settings.num_chains = chains

    for name, val in kwargs.items():
        setattr(settings, name, val)

    if cores is None:
        try:
            # Only available in python>=3.13
            available = os.process_cpu_count()  # type: ignore
        except AttributeError:
            available = os.cpu_count()
        cores = min(chains, cast(int, available))

    if init_mean is None:
        init_mean = np.zeros(compiled_model.n_dim)

    sampler = _BackgroundSampler(
        compiled_model,
        settings,
        init_mean,
        cores,
        progress_bar=progress_bar,
        save_warmup=save_warmup,
        return_raw_trace=return_raw_trace,
        progress_template=progress_template,
        progress_style=progress_style,
        progress_rate=progress_rate,
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
