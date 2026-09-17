import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import scipy.sparse as sp

from nutpie import _lib  # type: ignore
from nutpie.sample import (
    CompiledModel,
    _check_reserved_names,
    _flatten_point,
    _wrap_init_point_fn,
)
from nutpie.sparsity import variables_from_layout

SeedType = int


def _ignore_chain_id(init_point_fn: Callable[[SeedType], np.ndarray]):
    """Adapt a seed-only init function to `fn(seed, chain_id)`."""

    def init_point(seed, chain_id):
        return init_point_fn(seed)

    return init_point


@dataclass(frozen=True, repr=False)
class PyFuncModel(CompiledModel):
    _make_logp_func: Callable
    _make_expand_func: Callable
    # Called as `fn(seed, chain_id)`
    _make_initial_points: Callable[[SeedType, int], np.ndarray] | None
    _shared_data: dict[str, Any]
    _n_dim: int
    _variables: list[_lib.PyVariable]
    _dim_sizes: dict[str, int]
    _coords: dict[str, Any]
    _raw_logp_fn: Callable | None
    _transform_adapt_args: dict | None = None
    # Names and shapes of the parts of the unconstrained point, used to
    # accept dicts from `with_init_point_fn`.
    _init_point_layout: tuple[list[str], list[tuple[int, ...]]] | None = None
    # User init function `fn(model, rng, chain_id)`, see `with_init_point_fn`
    _init_point_fn: Callable | None = None

    @property
    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {var.name: tuple(var.dtype.shape) for var in self._variables}

    @property
    def coords(self):
        return self._coords

    @property
    def n_dim(self):
        return self._n_dim

    def with_data(self, **updates):
        for name in updates:
            if name not in self._shared_data:
                raise ValueError(f"Unknown data variable: {name}")

        updated = self._shared_data.copy()
        updated.update(**updates)
        return dataclasses.replace(
            self, _shared_data=updated, _hessian_sparsity=None, _factorization=None
        )

    def with_transform_adapt(self, **kwargs):
        return dataclasses.replace(self, _transform_adapt_args=kwargs)

    def with_init_point_fn(self, init_point_fn):
        """Use a custom function to generate the initial point of each chain.

        Parameters
        ----------
        init_point_fn : Callable[[PyFuncModel, np.random.Generator, int], np.ndarray | dict]
            Called as ``init_point_fn(model, rng, chain_id)``. Must return a flat
            point on the unconstrained space with shape ``(n_dim,)``. For
            models compiled from PyMC it may also return a dict mapping the
            names of the (transformed) value variables to their values.
            Variables missing from the dict are initialized with the default
            initialization of the model.
        """
        return dataclasses.replace(self, _init_point_fn=init_point_fn)

    def _make_init_point_func(self):
        if self._init_point_fn is None:
            return self._make_initial_points

        convert_dict = None
        if self._init_point_layout is not None:
            names, shapes = self._init_point_layout
            default_fn = self._make_initial_points

            def convert_dict(values, seed, chain_id):
                base = None if default_fn is None else default_fn(seed, chain_id)
                return _flatten_point(values, names, shapes, base)

        return _wrap_init_point_fn(self._init_point_fn, self, convert_dict)

    def _detect_hessian_sparsity(self, num_points: int = 4, *, seed=None):
        """Detect the Hessian sparsity with asdex.

        asdex computes a structural pattern from the jax program. We then
        drop entries that are exactly zero at all of `num_points` initial
        points, using a sparse Hessian with that pattern.
        """
        if self._raw_logp_fn is None:
            raise NotImplementedError(
                "Detecting the Hessian sparsity needs the jax log density. "
                "Compile the model with `gradient_backend='jax'`, or pass the "
                "pattern with `with_hessian_sparsity(array)`."
            )
        import asdex
        import jax

        def logp(x):
            return self._raw_logp_fn(x)[0]

        points = self._hessian_points(num_points, seed)
        structural = asdex.hessian_sparsity(logp, points[0])
        coloring = asdex.hessian_coloring_from_sparsity(structural)
        hessian = jax.jit(asdex.hessian_from_coloring(logp, coloring))

        rows, cols = [], []
        for point in points:
            values = hessian(point)
            indices = np.asarray(values.indices)
            # NaN counts as nonzero
            keep = np.asarray(values.data) != 0
            rows.append(indices[keep, 0])
            cols.append(indices[keep, 1])
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        return sp.coo_array(
            (np.ones(len(rows), dtype=bool), (rows, cols)),
            shape=(self.n_dim, self.n_dim),
        )

    def _hessian_points(self, num_points, seed, max_tries=100):
        """Initial points with finite log density."""
        rng = np.random.default_rng(seed)
        init = self._make_init_point_func()
        points = []
        for _ in range(num_points * max_tries):
            if len(points) == num_points:
                break
            if init is None:
                point = rng.uniform(-2, 2, size=self.n_dim)
            else:
                point = np.asarray(init(int(rng.integers(2**63)), len(points)))
            if np.isfinite(float(self._raw_logp_fn(point)[0])):
                points.append(point)
        if len(points) < num_points:
            raise ValueError(
                f"Found only {len(points)} of {num_points} points with finite "
                "log density to evaluate the Hessian."
            )
        return points

    def _unconstrained_variables(self):
        if self._init_point_layout is None:
            return super()._unconstrained_variables()
        return variables_from_layout(*self._init_point_layout)

    def _repr_header(self):
        return f"PyFuncModel (n_dim={self.n_dim})"

    def _repr_items(self):
        items = []
        if self._shared_data:
            items.append(("data", ", ".join(self._shared_data)))
        if self._coords:
            coords = ", ".join(f"{k} ({len(v)})" for k, v in self._coords.items())
            items.append(("coords", coords))
        return items + super()._repr_items()

    def _make_sampler(
        self,
        settings,
        cores,
        progress_type,
        extra_callback,
        extra_callback_rate,
        store,
        stop_event=None,
    ):
        model = self._make_model(self._adapter_kwargs(settings, stop_event))
        return _lib.PySampler.from_pyfunc(
            settings,
            cores,
            model,
            progress_type,
            extra_callback,
            extra_callback_rate,
            store,
        )

    def _make_model(self, adapter_kwargs=None):
        def make_logp_func():
            logp_fn = self._make_logp_func()
            return partial(logp_fn, **self._shared_data)

        def make_expand_func(seed1, seed2, chain):
            expand_fn = self._make_expand_func(seed1, seed2, chain)
            return partial(expand_fn, **self._shared_data)

        if self._raw_logp_fn is not None:
            outer_kwargs = {
                **(self._transform_adapt_args or {}),
                **(adapter_kwargs or {}),
            }

            def make_adapter(*args, **kwargs):
                from nutpie.transform_adapter import make_transform_adapter

                return make_transform_adapter(**outer_kwargs)(
                    *args, **kwargs, logp_fn=self._raw_logp_fn
                )

        else:
            make_adapter = None

        return _lib.PyModel(
            make_logp_func,
            make_expand_func,
            self._variables,
            self.n_dim,
            dim_sizes=self._dim_sizes,
            coords=self._coords,
            init_point_func=self._make_init_point_func(),
            transform_adapter=make_adapter,
            unconstrained_names=self._unconstrained_names,
        )


def from_pyfunc(
    ndim: int,
    make_logp_fn: Callable,
    make_expand_fn: Callable,
    expanded_dtypes: list[np.dtype],
    expanded_shapes: list[tuple[int, ...]],
    expanded_names: list[str],
    *,
    coords: dict[str, Any] | None = None,
    dims: dict[str, tuple[str, ...]] | None = None,
    shared_data: dict[str, Any] | None = None,
    make_initial_point_fn: Callable[[SeedType], np.ndarray] | None = None,
    init_point_layout: tuple[list[str], list[tuple[int, ...]]] | None = None,
    make_transform_adapter=None,
    raw_logp_fn=None,
    reparameterized_names=None,
    unconstrained_names: list[str] | None = None,
):
    if coords is None:
        coords = {}
    if dims is None:
        dims = {}
    if shared_data is None:
        shared_data = {}

    coords = coords.copy()
    _check_reserved_names(coords, dims)
    if unconstrained_names is not None and len(unconstrained_names) != ndim:
        raise ValueError(
            f"Got {len(unconstrained_names)} unconstrained names for {ndim} dimensions."
        )

    dim_sizes = {k: len(v) for k, v in coords.items()}
    shapes = [tuple(shape) for shape in expanded_shapes]
    variables = _lib.PyVariable.new_variables(
        expanded_names,
        [str(dtype) for dtype in expanded_dtypes],
        shapes,
        dim_sizes,
        dims,
    )

    return PyFuncModel(
        _n_dim=ndim,
        dims=dims,
        _coords=coords,
        _dim_sizes=dim_sizes,
        _make_logp_func=make_logp_fn,
        _make_expand_func=make_expand_fn,
        _make_initial_points=(
            None
            if make_initial_point_fn is None
            else _ignore_chain_id(make_initial_point_fn)
        ),
        _variables=variables,
        _shared_data=shared_data,
        _raw_logp_fn=raw_logp_fn,
        _init_point_layout=init_point_layout,
        reparameterized_names=reparameterized_names,
        _unconstrained_names=(
            None if unconstrained_names is None else list(unconstrained_names)
        ),
    )
