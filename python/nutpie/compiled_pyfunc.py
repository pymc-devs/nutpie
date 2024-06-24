import dataclasses
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import numpy as np

from nutpie import _lib
from nutpie.sample import CompiledModel


@dataclass(frozen=True)
class PyFuncModel(CompiledModel):
    _make_logp_func: Callable
    _make_expand_func: Callable
    _shared_data: dict[str, Any]
    _n_dim: int
    _variables: list[_lib.PyVariable]
    _coords: dict[str, Any]

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

    def _make_sampler(self, settings, init_mean, cores, progress_type):
        model = self._make_model(init_mean)
        return _lib.PySampler.from_pyfunc(
            settings,
            cores,
            model,
            progress_type,
        )

    def _make_model(self, init_mean):
        def make_logp_func():
            logp_fn = self._make_logp_func()
            return partial(logp_fn, **self._shared_data)

        def make_expand_func(seed1, seed2, chain):
            expand_fn = self._make_expand_func(seed1, seed2, chain)
            return partial(expand_fn, **self._shared_data)

        return _lib.PyModel(
            make_logp_func,
            make_expand_func,
            self._variables,
            self.n_dim,
        )


def from_pyfunc(
    ndim: int,
    make_logp_fn: Callable,
    make_expand_fn: Callable,
    expanded_dtypes: list[np.dtype],
    expanded_shapes: list[tuple[int, ...]],
    expanded_names: list[str],
    *,
    initial_mean: np.ndarray | None = None,
    coords: dict[str, Any] | None = None,
    dims: dict[str, tuple[str, ...]] | None = None,
    shared_data: dict[str, Any] | None = None,
):
    variables = []
    for name, shape, dtype in zip(
        expanded_names, expanded_shapes, expanded_dtypes, strict=True
    ):
        shape = _lib.TensorShape(list(shape))
        if dtype == np.float64:
            dtype = _lib.ExpandDtype.float64_array(shape)
        elif dtype == np.float32:
            dtype = _lib.ExpandDtype.float32_array(shape)
        elif dtype == np.int64:
            dtype = _lib.ExpandDtype.int64_array(shape)
        variables.append(_lib.PyVariable(name, dtype))

    if coords is None:
        coords = {}
    if dims is None:
        dims = {}
    if shared_data is None:
        shared_data = {}

    if shared_data is None:
        shared_data = dict()
    return PyFuncModel(
        _n_dim=ndim,
        dims=dims,
        _coords=coords,
        _make_logp_func=make_logp_fn,
        _make_expand_func=make_expand_fn,
        _variables=variables,
        _shared_data=shared_data,
    )
