import dataclasses
import json
import os
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.metadata import version
from math import prod
from typing import Any, Literal, cast, get_args, overload

import arviz
import numpy as np
import pandas as pd
import pyarrow
import scipy.sparse as sp
import xarray as xr

from nutpie import _lib
from nutpie.sparsity import (
    Factorization,
    check_hessian_sparsity,
    factorize,
    resolve_variables,
)


@dataclass(frozen=True)
class CompiledModel:
    dims: dict[str, tuple[str, ...]] | None
    reparameterized_names: list[str] | None = field(default=None, kw_only=True)
    # See `with_hessian_sparsity` and `with_factorization`. Both depend on
    # the data, and are reset by `with_data`.
    _hessian_sparsity: sp.csr_array | None = field(default=None, kw_only=True)
    _factorization: Factorization | None = field(default=None, kw_only=True)

    @property
    def n_dim(self) -> int:
        raise NotImplementedError()

    @property
    def shapes(self) -> dict[str, tuple[int, ...]] | None:
        raise NotImplementedError()

    @property
    def coords(self):
        raise NotImplementedError()

    def _make_sampler(self, *args, **kwargs):
        raise NotImplementedError()

    def _make_model(self, *args, **kwargs):
        raise NotImplementedError()

    def with_init_point_fn(self, init_point_fn):
        """Use a custom function to generate the initial point of each chain.

        Parameters
        ----------
        init_point_fn : Callable[[CompiledModel, np.random.Generator, int], np.ndarray]
            Called as ``init_point_fn(model, rng, chain_id)``, where ``model``
            is the compiled model that is sampled. Must return a point on the
            unconstrained space with shape ``(n_dim,)``. Some backends also
            accept a dict of values, see their documentation. If the log
            density is not finite at the returned point, the function is
            called again with a new ``rng``.
        """
        raise NotImplementedError()

    def with_hessian_sparsity(self, sparsity=None, **kwargs):
        """Set or detect the sparsity pattern of the Hessian of the log density
        on the unconstrained space.

        The pattern is the conditional-dependency graph of the posterior. It
        is used by :meth:`with_factorization`, and depends on the data, so
        :meth:`with_data` resets it.

        Parameters
        ----------
        sparsity:
            Boolean ``(n_dim, n_dim)`` dense or sparse matrix. It is stored as
            a symmetric ``scipy.sparse.csr_array`` with a true diagonal. If
            None, the pattern is detected
            at a few initial points of the model (see
            :meth:`with_init_point_fn`). An entry is nonzero if the Hessian is
            nonzero at any of the points. Dependencies in branches of the
            model that are not taken at any of the points are not detected.
        **kwargs:
            Options for the detection, which depend on the backend.
        """
        self._check_has_data()
        if sparsity is None:
            sparsity = self._detect_hessian_sparsity(**kwargs)
        elif kwargs:
            raise TypeError(
                "Detection options can not be used with an explicit sparsity."
            )
        sparsity = check_hessian_sparsity(sparsity, self.n_dim)
        return dataclasses.replace(
            self, _hessian_sparsity=sparsity, _factorization=None
        )

    def with_factorization(self, order="metis", *, front=None):
        """Compute a symbolic factorization of the Hessian sparsity pattern.

        It consists of a variable order and the pattern of the Cholesky
        factor with that order. The triangular flow uses it to decide which
        earlier variables each variable may depend on. Requires
        :meth:`with_hessian_sparsity` first. The METIS order needs pymetis.

        Parameters
        ----------
        order:
            ``"metis"`` (nested dissection, the default), ``"amd"``
            (approximate minimum degree, which usually needs less fill but
            gives less parallelism), ``"natural"``, or an explicit order
            (``order[k]`` is the parameter at position ``k``).
        front:
            Parameters to put at the front of the order, before the
            automatically ordered rest. Given as indices, names of
            unconstrained parameters or names of model variables (all their
            parameters).
        """
        if self._hessian_sparsity is None:
            raise ValueError(
                "The factorization needs the Hessian sparsity. "
                "Call `with_hessian_sparsity()` first."
            )
        factorization = self._factorize(self._hessian_sparsity, order, front)
        return dataclasses.replace(self, _factorization=factorization)

    @property
    def hessian_sparsity(self) -> sp.csr_array | None:
        """The Hessian sparsity pattern set by :meth:`with_hessian_sparsity`."""
        return self._hessian_sparsity

    @property
    def factorization(self) -> Factorization | None:
        """The factorization set by :meth:`with_factorization`."""
        return self._factorization

    def _check_has_data(self):
        """Raise if the model needs data before its structure is known."""

    def _detect_hessian_sparsity(self, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} can not detect the Hessian sparsity. "
            "Pass it explicitly with `with_hessian_sparsity(array)`."
        )

    def _unconstrained_parameters(self) -> list[str]:
        """Names of the unconstrained parameters, as in the
        ``unconstrained_parameter`` coordinate of the trace."""
        coords = self.coords or {}
        if "unconstrained_parameter" in coords:
            return [str(name) for name in coords["unconstrained_parameter"]]
        return [str(i) for i in range(self.n_dim)]

    def _unconstrained_variables(self) -> list[str]:
        """The model variable each unconstrained parameter belongs to."""
        return self._unconstrained_parameters()

    def _factorize(self, sparsity, order="metis", front=None) -> Factorization:
        parameters = self._unconstrained_parameters()
        variables = self._unconstrained_variables()
        front = resolve_variables(front or (), parameters, variables)
        return factorize(
            sparsity,
            order=order,
            front=front,
            unconstrained_parameters=parameters,
            variables=variables,
        )

    def _flow_structure_kwargs(self, settings) -> dict:
        """Order and sparsity for the triangular flow, if it is used.

        Without a factorization, a default one is computed for this sampler.
        """
        args = getattr(self, "_transform_adapt_args", None) or {}
        if settings.as_dict()["adaptation"] != "flow":
            return {}
        if args.get("coupling_type", "triangular") != "triangular":
            return {}
        if args.get("sparsity") is not None:
            return {}

        factorization = self._factorization
        if factorization is None:
            sparsity = self._hessian_sparsity
            if sparsity is None:
                try:
                    sparsity = check_hessian_sparsity(
                        self._detect_hessian_sparsity(), self.n_dim
                    )
                except Exception as err:
                    raise RuntimeError(
                        "The triangular flow needs the Hessian sparsity of the "
                        "model, and it could not be detected. Set it with "
                        "`with_hessian_sparsity(array)`, or use a different "
                        "`coupling_type`."
                    ) from err
            factorization = self._factorize(sparsity)
        return {"order": factorization.order, "sparsity": factorization.filled}

    def _repr_header(self) -> str:
        return type(self).__name__

    def _repr_items(self) -> list[tuple[str, str]]:
        """Backend-independent lines of the repr."""
        items = []
        args = getattr(self, "_transform_adapt_args", None)
        if args:
            items.append(("transform adapt", _format_kwargs(args)))
        init_point_fn = getattr(self, "_init_point_fn", None)
        if init_point_fn is not None:
            name = getattr(init_point_fn, "__name__", type(init_point_fn).__name__)
            items.append(("init point fn", name))
        if self._hessian_sparsity is not None:
            n = self._hessian_sparsity.shape[0]
            edges = (self._hessian_sparsity.nnz - n) // 2
            items.append(
                ("hessian sparsity", f"{edges} of {n * (n - 1) // 2} pairs nonzero")
            )
        if self._factorization is not None:
            f = self._factorization
            summary = (
                f"{f.method}, fill {f.num_fill}, "
                f"max {f.max_parents} parents, {f.num_levels} levels"
            )
            items.append(("factorization", summary))
        return items

    def __repr__(self):
        lines = [self._repr_header()]
        lines.extend(f"  {label}: {value}" for label, value in self._repr_items())
        return "\n".join(lines)

    def benchmark_logp(self, point, num_evals, cores):
        """Time how long the logp gradient evaluation takes.

        # Parameters
        """
        model = self._make_model()
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


def _format_kwargs(kwargs, max_len=60):
    def format_value(value):
        if isinstance(value, np.ndarray):
            return f"<array {value.shape}>"
        text = repr(value)
        return text if len(text) <= 20 else f"<{type(value).__name__}>"

    text = ", ".join(f"{key}={format_value(value)}" for key, value in kwargs.items())
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _flatten_point(values, names, shapes, base=None):
    """Concatenate a dict of arrays into one flat float64 array, in the
    order given by ``names``.

    Variables missing from ``values`` are taken from the flat array
    ``base``. If ``base`` is None, all variables must be given.
    """
    unknown = [
        name for name in values if name not in names and np.size(values[name]) > 0
    ]
    if unknown:
        raise KeyError(
            f"Unknown variables in initial point: {unknown}. "
            f"Expected a subset of {list(names)}."
        )

    total_size = sum(prod(shape) for shape in shapes)
    if base is None:
        flat_array = np.empty(total_size, dtype="float64", order="C")
    else:
        flat_array = np.array(base, dtype="float64", order="C", copy=True)
        if flat_array.shape != (total_size,):
            raise ValueError(
                f"Default initial point has shape {flat_array.shape}, "
                f"expected {(total_size,)}"
            )
    cursor = 0

    for name, shape in zip(names, shapes, strict=True):
        n = prod(shape)
        if name not in values:
            if base is None:
                raise KeyError(f"Initial point is missing a value for {name}")
            cursor += n
            continue
        value = np.asarray(values[name])
        if tuple(value.shape) != tuple(shape):
            raise ValueError(
                f"Size of initial value for {name} is {value.shape}, "
                f"expected {tuple(shape)}"
            )
        flat_array[cursor : cursor + n] = value.ravel().astype("float64")
        cursor += n

    return flat_array


def _wrap_init_point_fn(init_point_fn, model, convert_dict=None):
    """Adapt a user ``fn(model, rng, chain_id)`` to the ``fn(seed, chain_id)``
    signature the rust models call.

    If ``convert_dict`` is given, the user function may also return a
    dict, which is converted with ``convert_dict(values, seed, chain_id)``.
    The seed passed there is independent of the user's rng, and can be
    used to fill in missing values.
    """

    def init_point(seed, chain_id):
        user_seed, fill_seed = np.random.SeedSequence(seed).spawn(2)
        point = init_point_fn(model, np.random.default_rng(user_seed), chain_id)
        if isinstance(point, Mapping):
            if convert_dict is None:
                raise TypeError(
                    "The init point function must return an array for this model."
                )
            return convert_dict(point, int(fill_seed.generate_state(1)[0]), chain_id)
        return np.ascontiguousarray(point, dtype=np.float64)

    return init_point


def _arrow_to_arviz(
    draw_batches,
    stat_batches,
    skip_vars=None,
    reparameterized_names=None,
    keep_unconstrained_draw=False,
    **kwargs,
):
    if skip_vars is None:
        skip_vars = []
    if reparameterized_names is None:
        reparameterized_names = []

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

    uc_data_posterior = {
        name: data_posterior.pop(name)
        for name in reparameterized_names
        if name in data_posterior
    }
    uc_data_tune = {
        name: data_tune.pop(name) for name in reparameterized_names if name in data_tune
    }

    arviz_version = version("arviz")
    use_datatree = tuple(map(int, arviz_version.split(".")[:2])) >= (1, 0)
    if use_datatree:
        idata = arviz.from_dict(
            {
                "posterior": data_posterior,
                "sample_stats": stats_posterior,
                "warmup_posterior": data_tune,
                "warmup_sample_stats": stats_tune,
            },
            dims=dims,
            **kwargs,
        )
    else:
        idata = arviz.from_dict(
            posterior=data_posterior,
            sample_stats=stats_posterior,
            warmup_posterior=data_tune,
            warmup_sample_stats=stats_tune,  # ty:ignore[invalid-argument-type]
            dims=dims,
            **kwargs,
        )

    if keep_unconstrained_draw and uc_data_posterior:
        coords = kwargs.get("coords")
        uc_dims = {name: dims.get(name, []) for name in uc_data_posterior}
        groups = {
            "unconstrained_posterior": arviz.dict_to_dataset(
                uc_data_posterior, coords=coords, dims=uc_dims
            )
        }
        if uc_data_tune:
            groups["warmup_unconstrained_posterior"] = arviz.dict_to_dataset(
                uc_data_tune, coords=coords, dims=uc_dims
            )
        if use_datatree:
            idata = idata.assign(**{k: xr.DataTree(v) for k, v in groups.items()})
        else:
            idata.add_groups(groups)

    return idata


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
            elif dtype == np.dtype("O"):
                data = np.full(total_shape, None, dtype=dtype)
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
            if values.shape[0] == num_draws:
                values = values[~is_null]
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
        import marimo as mo  # ty:ignore[unresolved-import]

        return mo.running_in_notebook()
    except ImportError:
        return False


def _mo_write_internal(cell_id, stream, value: object) -> None:
    """Write to marimo cell given cell_id and stream."""
    import marimo  # ty:ignore[unresolved-import]

    if marimo.__version__ < "0.19.0":
        # The old CellOp API is identical to new CellNotificationUtils
        from marimo._messaging.ops import (  # ty:ignore[unresolved-import]
            CellOp as CellNotificationUtils,
        )
    else:
        from marimo._messaging.notification_utils import (  # ty:ignore[unresolved-import]
            CellNotificationUtils,
        )

    from marimo._messaging.cell_output import (  # ty:ignore[unresolved-import]
        CellChannel,
    )
    from marimo._messaging.tracebacks import (  # ty:ignore[unresolved-import]
        write_traceback,
    )
    from marimo._output import formatting  # ty:ignore[unresolved-import]

    output = formatting.try_format(value)
    if output.traceback is not None:
        write_traceback(output.traceback)
    CellNotificationUtils.broadcast_output(
        channel=CellChannel.OUTPUT,
        mimetype=output.mimetype,
        data=output.data,
        cell_id=cell_id,
        status=None,
        stream=stream,
    )


def _mo_create_replace():
    """Create mo.output.replace with current context pinned."""
    from marimo._output import formatting  # ty:ignore[unresolved-import]
    from marimo._runtime.context import get_context  # ty:ignore[unresolved-import]
    from marimo._runtime.context.types import (  # ty:ignore[unresolved-import]
        ContextNotInitializedError,
    )

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
            from google import colab  # noqa: F401  # ty:ignore[unresolved-import]

            return True
        except ImportError:
            return False

    if in_colab():
        return True
    # Databricks sets this env var on all cluster runtimes (same check as rich)
    if os.getenv("DATABRICKS_RUNTIME_VERSION"):
        return True
    try:
        shell = get_ipython().__class__.__name__  # type: ignore
        if shell == "ZMQInteractiveShell":  # Jupyter notebook, Spyder or qtconsole
            try:
                from IPython.display import (  # ty:ignore[unresolved-import]
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


# Builds without the `zarr` feature (e.g. wasm) have no store module.
if hasattr(_lib, "store"):
    _ZarrStoreType = (
        _lib.store.S3Store
        | _lib.store.LocalStore
        | _lib.store.HTTPStore
        | _lib.store.GCSStore
        | _lib.store.AzureStore
    )
else:
    _ZarrStoreType = Any


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
        cores,
        *,
        progress_bar=True,
        progress_callback=None,
        save_warmup=True,
        return_raw_trace=False,
        progress_template=None,
        progress_style=None,
        progress_rate=100,
        store=None,
        store_unconstrained=False,
    ):
        self._settings = settings
        self._compiled_model = compiled_model
        self._save_warmup = save_warmup
        self._return_raw_trace = return_raw_trace
        self._store_unconstrained = store_unconstrained

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

            import IPython  # ty:ignore[unresolved-import]

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
                except Exception as e:  # noqa: BLE001
                    if not did_print_error:
                        did_print_error = True
                        print(f"Error updating progress display: {e}")

            progress_type = _lib.ProgressType.template_callback(
                progress_rate, progress_template, cores, callback
            )
        elif in_marimo_notebook():
            import marimo as mo  # ty:ignore[unresolved-import]

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
            cores,
            progress_type,
            progress_callback,
            progress_rate,
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
        settings_dict = self._settings.as_dict()
        if self._return_raw_trace:
            return results
        else:
            if results.is_zarr():
                import obstore
                from zarr.storage import ObjectStore

                assert self._zarr_store is not None

                args, kwargs = self._zarr_store.__getnewargs_ex__()
                name = self._zarr_store.__class__.__name__
                cls = getattr(obstore.store, name)
                store = cls(*args, **kwargs)

                obj_store = ObjectStore(store, read_only=True)
                return xr.open_datatree(obj_store, engine="zarr", consolidated=False)  # ty:ignore[invalid-argument-type]

            elif results.is_arrow():
                skip_vars = []
                skips = {
                    "store_gradient": ["gradient"],
                    "store_unconstrained": ["unconstrained_draw"],
                    "adapt_options.mass_matrix_options.store_mass_matrix": [
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
                    "store_transformed": [
                        "transformed_position",
                        "transformed_gradient",
                        "transformation_mu",
                    ],
                }

                def _get_nested(settings, name, default):
                    parts = name.split(".")
                    for part in parts:
                        if part not in settings:
                            return default
                        settings = settings[part]
                    return settings

                for setting, names in skips.items():
                    if not _get_nested(settings_dict["settings"], setting, False):
                        skip_vars.extend(names)

                draw_batches, stat_batches = results.get_arrow_trace()

                from nutpie import __version__

                attrs = {
                    "inference_library": "nutpie",
                    "inference_library_version": __version__,
                    "inference_library_settings": json.dumps(self._settings.as_dict()),
                }

                return _arrow_to_arviz(
                    draw_batches,
                    stat_batches,
                    skip_vars=skip_vars,
                    reparameterized_names=self._compiled_model.reparameterized_names,
                    keep_unconstrained_draw=self._store_unconstrained,
                    coords={
                        name: pd.Index(vals)
                        for name, vals in self._compiled_model.coords.items()
                    },
                    save_warmup=self._save_warmup,
                    attrs={"sample_stats": attrs},
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
    adaptation: Literal["diag", "draw_diag", "low_rank", "flow"] = "diag",
    init_mean: np.ndarray | None = None,
    return_raw_trace: bool = False,
    progress_callback: Any | None = None,
    progress_template: str | None = None,
    progress_style: str | None = None,
    progress_rate: int = 100,
    zarr_store: _ZarrStoreType | None = None,
    store_unconstrained: bool = False,
) -> xr.DataTree: ...


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
    adaptation: Literal["diag", "draw_diag", "low_rank", "flow"] = "diag",
    init_mean: np.ndarray | None = None,
    return_raw_trace: bool = False,
    blocking: Literal[True],
    progress_callback: Any | None = None,
    progress_template: str | None = None,
    progress_style: str | None = None,
    progress_rate: int = 100,
    zarr_store: _ZarrStoreType | None = None,
    store_unconstrained: bool = False,
    **kwargs,
) -> xr.DataTree: ...


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
    adaptation: Literal["diag", "draw_diag", "low_rank", "flow"] = "diag",
    init_mean: np.ndarray | None = None,
    return_raw_trace: bool = False,
    blocking: Literal[False],
    progress_callback: Any | None = None,
    progress_template: str | None = None,
    progress_style: str | None = None,
    progress_rate: int = 100,
    zarr_store: _ZarrStoreType | None = None,
    store_unconstrained: bool = False,
    **kwargs,
) -> _BackgroundSampler: ...


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
    adaptation: Literal["diag", "draw_diag", "low_rank", "flow"] = "diag",
    init_mean: np.ndarray | None = None,
    return_raw_trace: bool = False,
    progress_callback: Any | None = None,
    progress_template: str | None = None,
    progress_style: str | None = None,
    progress_rate: int = 100,
    zarr_store: _ZarrStoreType | None = None,
    **kwargs,
) -> xr.DataTree: ...


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
    sampler: Literal["nuts", "mclmc"] = "nuts",
    adaptation: Literal["diag", "draw_diag", "low_rank", "flow"] = "diag",
    init_mean: np.ndarray | None = None,
    return_raw_trace: bool = False,
    blocking: bool = True,
    progress_callback: Any | None = None,
    progress_template: str | None = None,
    progress_style: str | None = None,
    progress_rate: int = 100,
    zarr_store: _ZarrStoreType | None = None,
    store_unconstrained: bool = False,
    **kwargs,
) -> xr.DataTree | _BackgroundSampler:
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
        Deprecated and ignored. Use
        ``compiled_model.with_init_point_fn`` to control the initial
        points of the chains.
    store_unconstrained: bool
        If True, store the unconstrained (transformed) draws in two forms:
        a flat ``unconstrained_draw`` vector in ``sample_stats`` and a
        per-variable ``unconstrained_posterior`` group (with
        ``warmup_unconstrained_posterior`` when ``save_warmup=True``) whose
        dims are copied from the corresponding RV.
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
    sampler: str, default="nuts"
        The sampler to use. One of:

        - ``"nuts"`` (default): No-U-Turn Sampler.
        - ``"mclmc"``: Microcanonical Langevin Monte Carlo.
          mclmc is **experimental** and might change or disapear
          in a future release. It might also eat your homework.

    adaptation: str, default="diag"
        The mass matrix adaptation strategy to use. One of:

        - ``"diag"`` (default): Diagonal mass matrix estimated from
          draw and gradient variance. This is nutpie's standard
          adaptation.
        - ``"draw_diag"``: Diagonal mass matrix estimated from draw
          variance only, similar to the adaptation in Stan and PyMC.
          Usually less efficient, but occasionally produces a higher
          total number of effective samples.
        - ``"low_rank"``: Low-rank modified diagonal mass matrix that
          can adapt to some posterior correlations. *Experimental.*
        - ``"flow"``: Normalizing-flow reparameterisation during
          tuning. *Experimental.*
    mass_matrix_eigval_cutoff: float > 1, default=100
        Ignore eigenvalues between cutoff and 1/cutoff in the
        low-rank modified mass matrix estimate. Higher values
        lead to worse correlation fitting, but increase
        the performance of leapfrog steps.
        Only applicable with ``adaptation="low_rank"``.
    mass_matrix_gamma: float > 0, default=1e-5
        Regularisation parameter for the eigenvalues. Only
        applicable with ``adaptation="low_rank"``.
    progress_template: str
        This is only exposed for experimentation. upon template
        for the html progress representation.
    progress_style: str
        This is only exposed for experimentation. Common HTML
        for the progress bar (eg CSS).
    progress_rate: int, default=500
        Rate in ms at which the progress should be updated.
    progress_callback: callable(list[ChainProgress]) | None, default=None
        An optional callback function that is called periodically with the
        current progress of all chains. It receives a list of
        ``nutpie.ChainProgress`` objects, one per chain, each exposing:

        - ``finished_draws`` – number of draws completed so far
        - ``total_draws`` – total draws to produce (tune + draws)
        - ``divergences`` – number of divergent transitions so far
        - ``tuning`` – whether the chain is still in the warmup phase
        - ``started`` – whether the chain has started
        - ``latest_num_steps`` – leapfrog steps in the last trajectory
        - ``total_num_steps`` – cumulative leapfrog steps
        - ``step_size`` – current step size
        - ``runtime_ms`` – wall-clock time spent sampling (milliseconds)
        - ``divergent_draws`` – list of draw indices that diverged

        The callback fires at the same rate as the progress bar
        (``progress_rate`` ms). It runs on a background thread, so it must
        be thread-safe. (Builds without threads, e.g. on wasm, call it from
        inside ``wait`` instead.) Exceptions raised inside it are printed to stderr
        and otherwise silently swallowed so that sampling is not interrupted.
        The built-in progress bar is still shown regardless of whether this
        callback is set.
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
    trace : xr.DataTree:
        An Xarray ``DataTree`` object that contains the samples.
    """

    # Backward-compatible deprecated keyword arguments.
    _use_grad_based = None
    for _old_name, _new_adaptation in [
        ("low_rank_modified_mass_matrix", "low_rank"),
        ("transform_adapt", "flow"),
    ]:
        if _old_name in kwargs:
            _val = kwargs.pop(_old_name)
            if _val:
                warnings.warn(
                    f"`{_old_name}` is deprecated. "
                    f"Use `adaptation='{_new_adaptation}'` instead.",
                    FutureWarning,
                    stacklevel=2,
                )
                if adaptation == "diag":
                    _AdaptationLiteral = Literal[
                        "diag", "draw_diag", "low_rank", "flow"
                    ]
                    assert _new_adaptation in get_args(_AdaptationLiteral)
                    adaptation = cast(_AdaptationLiteral, _new_adaptation)
                else:
                    raise ValueError(
                        f"`{_old_name}` is deprecated and cannot be combined "
                        f"with the `adaptation` argument."
                    )
    if "use_grad_based_mass_matrix" in kwargs:
        _use_grad_based = kwargs.pop("use_grad_based_mass_matrix")
        warnings.warn(
            "`use_grad_based_mass_matrix` is deprecated. "
            "Use `adaptation='draw_diag'` instead of "
            "`use_grad_based_mass_matrix=False`.",
            FutureWarning,
            stacklevel=2,
        )

    if sampler == "nuts":
        if adaptation == "low_rank":
            settings = _lib.PyNutsSettings.LowRank(seed)
        elif adaptation == "flow":
            settings = _lib.PyNutsSettings.Flow(seed)
        elif adaptation in ("diag", "draw_diag"):
            settings = _lib.PyNutsSettings.Diag(seed)
            if adaptation == "draw_diag" or _use_grad_based is False:
                settings.use_grad_based_mass_matrix = False
        else:
            raise ValueError(
                f"Unknown adaptation strategy '{adaptation}'. "
                f"Expected one of: 'diag', 'draw_diag', 'low_rank', 'flow'."
            )
    elif sampler == "mclmc":
        if adaptation == "low_rank":
            settings = _lib.PyMclmcSettings.LowRank(seed)
        elif adaptation == "flow":
            settings = _lib.PyMclmcSettings.Flow(seed)
        elif adaptation in ("diag", "draw_diag"):
            settings = _lib.PyMclmcSettings.Diag(seed)
            if adaptation == "draw_diag" or _use_grad_based is False:
                settings.use_grad_based_mass_matrix = False
        else:
            raise ValueError(
                f"Unknown adaptation strategy '{adaptation}'. "
                f"Expected one of: 'diag', 'draw_diag', 'low_rank', 'flow'."
            )
    else:
        raise ValueError(
            f"Unknown sampler '{sampler}'. Expected one of: 'nuts', 'mclmc'."
        )

    updates = dict(kwargs)
    if tune is not None:
        updates["num_tune"] = tune
    if draws is not None:
        updates["num_draws"] = draws
    if chains is not None:
        updates["num_chains"] = chains

    settings.update(updates)

    if store_unconstrained:
        settings.store_unconstrained = True

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

    if init_mean is not None:
        warnings.warn(
            "`init_mean` is deprecated and has no effect. Use "
            "`compiled_model.with_init_point_fn` to set initial points.",
            FutureWarning,
            stacklevel=2,
        )

    background_sampler = _BackgroundSampler(
        compiled_model,
        settings,
        cores,
        progress_bar=progress_bar,
        progress_callback=progress_callback,
        save_warmup=save_warmup,
        return_raw_trace=return_raw_trace,
        progress_template=progress_template,
        progress_style=progress_style,
        progress_rate=progress_rate,
        store=zarr_store,
        store_unconstrained=store_unconstrained,
    )

    if not blocking:
        return background_sampler

    try:
        result = background_sampler.wait()
    except KeyboardInterrupt:
        result = background_sampler.abort()
    except:
        background_sampler.cancel()
        raise

    return result
