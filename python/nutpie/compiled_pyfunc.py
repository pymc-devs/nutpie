import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

from nutpie import _lib  # type: ignore
from nutpie.sample import CompiledModel, _flatten_point, _wrap_init_point_fn

SeedType = int


def _ignore_chain_id(init_point_fn: Callable[[SeedType], np.ndarray]):
    """Adapt a seed-only init function to `fn(seed, chain_id)`."""

    def init_point(seed, chain_id):
        return init_point_fn(seed)

    return init_point


@dataclass(frozen=True)
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
        return dataclasses.replace(self, _shared_data=updated)

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

    def _make_sampler(
        self,
        settings,
        cores,
        progress_type,
        extra_callback,
        extra_callback_rate,
        store,
    ):
        model = self._make_model()
        return _lib.PySampler.from_pyfunc(
            settings,
            cores,
            model,
            progress_type,
            extra_callback,
            extra_callback_rate,
            store,
        )

    def _make_model(self):
        def make_logp_func():
            logp_fn = self._make_logp_func()
            return partial(logp_fn, **self._shared_data)

        def make_expand_func(seed1, seed2, chain):
            expand_fn = self._make_expand_func(seed1, seed2, chain)
            return partial(expand_fn, **self._shared_data)

        if self._raw_logp_fn is not None:
            outer_kwargs = self._transform_adapt_args
            if outer_kwargs is None:
                outer_kwargs = {}

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
):
    if coords is None:
        coords = {}
    if dims is None:
        dims = {}
    if shared_data is None:
        shared_data = {}

    coords = coords.copy()

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
    )
