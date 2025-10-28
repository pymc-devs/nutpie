import dataclasses
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import numpy as np

from nutpie import _lib  # type: ignore
from nutpie.sample import CompiledModel

SeedType = int


@dataclass(frozen=True)
class PyFuncModel(CompiledModel):
    _make_logp_func: Callable
    _make_expand_func: Callable
    _make_initial_points: Callable[[SeedType], np.ndarray] | None
    _shared_data: dict[str, Any]
    _n_dim: int
    _variables: list[_lib.PyVariable]
    _dim_sizes: dict[str, int]
    _coords: dict[str, Any]
    _raw_logp_fn: Callable | None
    _transform_adapt_args: dict | None = None
    _force_single_core: bool = False

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

        # Convert to MLX arrays if using MLX backend (indicated by force_single_core)
        if self._force_single_core:
            import mlx.core as mx

            for name, value in updates.items():
                updated[name] = mx.array(value)
        else:
            updated.update(**updates)

        return dataclasses.replace(self, _shared_data=updated)

    def with_transform_adapt(self, **kwargs):
        return dataclasses.replace(self, _transform_adapt_args=kwargs)

    def _make_sampler(self, settings, init_mean, cores, progress_type, store):
        # Force single-core execution if required (e.g., for MLX backend)
        if self._force_single_core:
            cores = 1
        model = self._make_model(init_mean)
        return _lib.PySampler.from_pyfunc(
            settings,
            cores,
            model,
            progress_type,
            store,
        )

    def _make_model(self, init_mean):
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
            init_point_func=self._make_initial_points,
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
    make_transform_adapter=None,
    raw_logp_fn=None,
    force_single_core: bool = False,
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
        _make_initial_points=make_initial_point_fn,
        _variables=variables,
        _shared_data=shared_data,
        _raw_logp_fn=raw_logp_fn,
        _force_single_core=force_single_core,
    )
