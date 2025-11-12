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


def _arrow_to_arviz(draw_batches, stat_batches, skip_vars=None, **kwargs):
    if skip_vars is None:
        skip_vars = []

    n_chains = len(draw_batches)
    assert n_chains == len(stat_batches)

    max_tuning = 0
    max_posterior = 0
    num_tuning = []

    for draw, stat in zip(draw_batches, stat_batches):
        tuning = stat.column("tuning")
        _num_tuning = tuning.to_numpy().sum()
        assert draw.num_rows == stat.num_rows
        max_tuning = max(max_tuning, _num_tuning)
        max_posterior = max(max_posterior, draw.num_rows - _num_tuning)
        num_tuning.append(_num_tuning)

    data_tune = {}
    data_posterior = {}

    stats_tune = {}
    stats_posterior = {}

    dims = {}

    for i, draw in enumerate(draw_batches):
        draw_tune = draw.slice(0, num_tuning[i])
        _add_arrow_data(data_tune, max_tuning, draw_tune, i, n_chains, dims, [])
        draw_posterior = draw.slice(num_tuning[i], draw.num_rows - num_tuning[i])
        _add_arrow_data(
            data_posterior, max_posterior, draw_posterior, i, n_chains, dims, []
        )
    for i, stat in enumerate(stat_batches):
        stat_tune = stat.slice(0, num_tuning[i])
        _add_arrow_data(stats_tune, max_tuning, stat_tune, i, n_chains, dims, skip_vars)
        stat_posterior = stat.slice(num_tuning[i], stat.num_rows - num_tuning[i])
        _add_arrow_data(
            stats_posterior, max_posterior, stat_posterior, i, n_chains, dims, skip_vars
        )

    return arviz.from_dict(
        data_posterior,
        sample_stats=stats_posterior,
        warmup_posterior=data_tune,
        warmup_sample_stats=stats_tune,
        dims=dims,
        **kwargs,
    )


def _add_arrow_data(data_dict, max_length, batch, chain, n_chains, dims, skip_vars):
    num_draws = batch.num_rows

    for name in batch.column_names:
        if name in skip_vars:
            continue
        col = batch.column(name)
        meta = col.field.metadata
        item_dims = meta.get(b"dims", [])
        if item_dims:
            item_dims = item_dims.decode("utf-8").split(",")
        item_shape = meta.get(b"shape", [])
        if item_shape:
            item_shape = item_shape.decode("utf-8").split(",")
        item_shape = [int(s) for s in item_shape]
        total_shape = [n_chains, max_length, *item_shape]

        col = pyarrow.array(col)

        is_null = col.is_null()

        if hasattr(col, "flatten"):
            col = col.flatten()
        dtype = col.type.to_pandas_dtype()

        if name not in data_dict:
            if dtype in [np.float64, np.float32]:
                data = np.full(total_shape, np.nan, dtype=dtype)
            else:
                data = np.zeros(total_shape, dtype=dtype)
            data_dict[name] = data

            dims[name] = item_dims

        values = col.to_numpy(False)
        if is_null.sum() == 0:
            data_dict[name][chain, :num_draws] = values.reshape(
                (num_draws,) + tuple(item_shape)
            )
        else:
            is_null = is_null.to_numpy(False)
            data_dict[name][chain, :num_draws][~is_null] = values.reshape(
                ((~is_null).sum(),) + tuple(item_shape)
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


def in_marimo_notebook() -> bool:
    try:
        import marimo as mo

        return mo.running_in_notebook()
    except ImportError:
        return False


def _mo_write_internal(cell_id, stream, value: object) -> None:
    """Write to marimo cell given cell_id and stream."""
    from marimo._messaging.cell_output import CellChannel
    from marimo._messaging.ops import CellOp
    from marimo._messaging.tracebacks import write_traceback
    from marimo._output import formatting

    output = formatting.try_format(value)
    if output.traceback is not None:
        write_traceback(output.traceback)
    CellOp.broadcast_output(
        channel=CellChannel.OUTPUT,
        mimetype=output.mimetype,
        data=output.data,
        cell_id=cell_id,
        status=None,
        stream=stream,
    )


def _mo_create_replace():
    """Create mo.output.replace with current context pinned."""
    from marimo._output import formatting
    from marimo._runtime.context import get_context
    from marimo._runtime.context.types import ContextNotInitializedError

    try:
        ctx = get_context()
    except ContextNotInitializedError:
        return

    cell_id = ctx.execution_context.cell_id
    execution_context = ctx.execution_context
    stream = ctx.stream

    def replace(value):
        execution_context.output = [formatting.as_html(value)]

        _mo_write_internal(cell_id=cell_id, value=value, stream=stream)

    return replace


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


_ZarrStoreType = (
    _lib.store.S3Store
    | _lib.store.LocalStore
    | _lib.store.HTTPStore
    | _lib.store.GCSStore
    | _lib.store.AzureStore
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
    _store: _lib.PyStorage
    _zarr_store: _ZarrStoreType | None = None

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
        store=None,
    ):
        self._settings = settings
        self._compiled_model = compiled_model
        self._save_warmup = save_warmup
        self._return_raw_trace = return_raw_trace

        self._html = None

        if store is None:
            store = _lib.PyStorage.arrow()
        elif type(store).__module__ == "_lib.store":
            self._zarr_store = store
            store = _lib.PyStorage.zarr(store)

        self._store = store

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

            did_print_error = False

            def callback(formatted):
                nonlocal did_print_error

                try:
                    self._html = formatted
                    self.display_id.update(self)
                except Exception as e:
                    if not did_print_error:
                        did_print_error = True
                        print(f"Error updating progress display: {e}")

            progress_type = _lib.ProgressType.template_callback(
                progress_rate, progress_template, cores, callback
            )
        elif in_marimo_notebook():
            import marimo as mo

            if progress_template is None:
                progress_template = _progress_template

            if progress_style is None:
                progress_style = _progress_style

            self._html = ""

            mo.output.clear()
            mo_output_replace = _mo_create_replace()

            def callback(formatted):
                self._html = formatted
                html = mo.Html(f"{progress_style}\n{formatted}")
                mo_output_replace(html)

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
            self._store,
        )

    def wait(self, *, timeout=None):
        """Wait until sampling is finished and return the trace.

        KeyboardInterrupt will lead to interrupt the waiting.

        This will return after `timeout` seconds even if sampling is
        not finished at this point.

        This resumes the sampler in case it had been paused.
        """
        self._sampler.wait(timeout)
        results = self._sampler.take_results()
        return self._extract(results)

    def _extract(self, results):
        if self._return_raw_trace:
            return results
        else:
            if results.is_zarr():
                import obstore
                import xarray as xr
                from zarr.storage import ObjectStore

                assert self._zarr_store is not None

                args, kwargs = self._zarr_store.__getnewargs_ex__()
                name = self._zarr_store.__class__.__name__
                cls = getattr(obstore.store, name)
                store = cls(*args, **kwargs)

                obj_store = ObjectStore(store, read_only=True)
                ds = xr.open_datatree(obj_store, engine="zarr", consolidated=False)
                return arviz.from_datatree(ds)

            elif results.is_arrow():
                skip_vars = []
                skips = {
                    "store_gradient": ["gradient"],
                    "store_unconstrained": ["unconstrained_draw"],
                    "store_mass_matrix": [
                        "mass_matrix_inv",
                        "mass_matrix_eigvals",
                        "mass_matrix_stds",
                    ],
                    "store_divergences": [
                        "divergence_start",
                        "divergence_end",
                        "divergence_momentum",
                        "divergence_start_gradient",
                    ],
                }

                for setting, names in skips.items():
                    if not getattr(self._settings, setting, False):
                        skip_vars.extend(names)

                draw_batches, stat_batches = results.get_arrow_trace()
                return _arrow_to_arviz(
                    draw_batches,
                    stat_batches,
                    skip_vars=skip_vars,
                    coords={
                        name: pd.Index(vals)
                        for name, vals in self._compiled_model.coords.items()
                    },
                    save_warmup=self._save_warmup,
                )
            else:
                raise ValueError("Unknown results type")

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
        results = self._sampler.take_results()
        return self._extract(results)

    def cancel(self):
        """Abort sampling and discard progress."""
        self._sampler.abort()

    def __del__(self):
        if not hasattr(self, "_sampler"):
            return
        if not self._sampler.is_empty(ignore_error=True):
            self.cancel()

    def _repr_html_(self):
        return self._html


@overload
def sample(
    compiled_model: CompiledModel,
    *,
    draws: int | None = None,
    tune: int | None = None,
    chains: int | None = None,
    cores: int | None = None,
    seed: int | None = None,
    save_warmup: bool = True,
    progress_bar: bool = True,
    low_rank_modified_mass_matrix: bool = False,
    transform_adapt: bool = False,
    init_mean: np.ndarray | None = None,
    return_raw_trace: bool = False,
    progress_template: str | None = None,
    progress_style: str | None = None,
    progress_rate: int = 100,
    zarr_store: _ZarrStoreType | None = None,
) -> arviz.InferenceData: ...


@overload
def sample(
    compiled_model: CompiledModel,
    *,
    draws: int | None = None,
    tune: int | None = None,
    chains: int | None = None,
    cores: int | None = None,
    seed: int | None = None,
    save_warmup: bool = True,
    progress_bar: bool = True,
    low_rank_modified_mass_matrix: bool = False,
    transform_adapt: bool = False,
    init_mean: np.ndarray | None = None,
    return_raw_trace: bool = False,
    blocking: Literal[True],
    progress_template: str | None = None,
    progress_style: str | None = None,
    progress_rate: int = 100,
    zarr_store: _ZarrStoreType | None = None,
    **kwargs,
) -> arviz.InferenceData: ...


@overload
def sample(
    compiled_model: CompiledModel,
    *,
    draws: int | None = None,
    tune: int | None = None,
    chains: int | None = None,
    cores: int | None = None,
    seed: int | None = None,
    save_warmup: bool = True,
    progress_bar: bool = True,
    low_rank_modified_mass_matrix: bool = False,
    transform_adapt: bool = False,
    init_mean: np.ndarray | None = None,
    return_raw_trace: bool = False,
    blocking: Literal[False],
    progress_template: str | None = None,
    progress_style: str | None = None,
    progress_rate: int = 100,
    zarr_store: _ZarrStoreType | None = None,
    **kwargs,
) -> _BackgroundSampler: ...


def sample(
    compiled_model: CompiledModel,
    *,
    draws: int | None = None,
    tune: int | None = None,
    chains: int | None = None,
    cores: int | None = None,
    seed: int | None = None,
    save_warmup: bool = True,
    progress_bar: bool = True,
    low_rank_modified_mass_matrix: bool = False,
    transform_adapt: bool = False,
    init_mean: np.ndarray | None = None,
    return_raw_trace: bool = False,
    blocking: bool = True,
    progress_template: str | None = None,
    progress_style: str | None = None,
    progress_rate: int = 100,
    zarr_store: _ZarrStoreType | None = None,
    **kwargs,
) -> arviz.InferenceData | _BackgroundSampler:
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
    zarr_store: nutpie.zarr_store.*
        A store created using nutpie.zarr_store to store the samples
        in. If None (default), the samples will be stored in
        memory using an arrow table. This can be used to write
        the trace directly into a zarr store, for instance
        on disk or to S3 or GCS.
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
    if chains is not None:
        settings.num_chains = chains

    for name, val in kwargs.items():
        setattr(settings, name, val)

    if cores is None:
        try:
            # Only available in python>=3.13
            available = os.process_cpu_count()  # type: ignore
        except AttributeError:
            available = os.cpu_count()
        if chains is None:
            cores = available
        else:
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
        store=zarr_store,
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
