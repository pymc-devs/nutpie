import json
import tempfile
from dataclasses import dataclass, replace
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from nutpie import _lib
from nutpie.sample import CompiledModel


class _NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@dataclass(frozen=True)
class CompiledStanModel(CompiledModel):
    _coords: Optional[dict[str, Any]]
    code: str
    data: Optional[dict[str, NDArray]]
    library: Any
    model: Any
    model_name: Optional[str] = None
    _transform_adapt_args: dict | None = None

    def with_data(self, *, seed=None, **updates):
        if self.data is None:
            data = {}
        else:
            data = self.data.copy()

        data.update(updates)

        if data is not None:
            data_json = json.dumps(data, cls=_NumpyArrayEncoder)
        else:
            data_json = None

        outer_kwargs = self._transform_adapt_args
        if outer_kwargs is None:
            outer_kwargs = {}

        def make_adapter(*args, **kwargs):
            from nutpie.transform_adapter import make_transform_adapter

            return make_transform_adapter(**outer_kwargs)(*args, **kwargs, logp_fn=None)

        model = _lib.StanModel(self.library, seed, data_json, make_adapter)
        coords = self._coords
        if coords is None:
            coords = {}
        else:
            coords = coords.copy()
        coords["unconstrained_parameter"] = pd.Index(model.param_unc_names())

        return CompiledStanModel(
            _coords=coords,
            data=data,
            code=self.code,
            library=self.library,
            dims=self.dims,
            model=model,
        )

    def with_coords(self, **coords):
        if self.coords is None:
            coords_new = {}
        else:
            coords_new = self.coords.copy()
        coords_new.update(coords)
        return replace(self, _coords=coords_new)

    def with_dims(self, **dims):
        if self.dims is None:
            dims_new = {}
        else:
            dims_new = self.dims.copy()
        dims_new.update(dims)
        return replace(self, dims=dims_new)

    def with_transform_adapt(self, **kwargs):
        return replace(self, _transform_adapt_args=kwargs).with_data()

    def _make_model(self, init_mean):
        if self.model is None:
            return self.with_data().model
        return self.model

    def _make_sampler(self, settings, init_mean, cores, progress_type):
        model = self._make_model(init_mean)
        return _lib.PySampler.from_stan(
            settings,
            cores,
            model,
            progress_type,
        )

    @property
    def n_dim(self):
        if self.model is None:
            return self.with_data().n_dim
        return self.model.ndim()

    @property
    def shapes(self):
        if self.model is None:
            return self.with_data().shapes
        return {name: var.shape for name, var in self.model.variables().items()}

    @property
    def coords(self):
        if self.model is None:
            return self.with_data().coords
        return self._coords


def compile_stan_model(
    *,
    code: Optional[str] = None,
    filename: Optional[str] = None,
    extra_compile_args: Optional[list[str]] = None,
    extra_stanc_args: Optional[list[str]] = None,
    dims: Optional[dict[str, int]] = None,
    coords: Optional[dict[str, Any]] = None,
    model_name: Optional[str] = None,
    cleanup: bool = True,
) -> CompiledStanModel:
    if find_spec("bridgestan") is None:
        raise ImportError(
            "BridgeStan is not installed in the current environment. "
            "Please install it with something like "
            "'pip install bridgestan'."
        )

    import bridgestan

    if dims is None:
        dims = {}
    if coords is None:
        coords = {}

    if code is not None and filename is not None:
        raise ValueError("Specify exactly one of `code` and `filename`")
    if code is None:
        if filename is None:
            raise ValueError("Either code or filename have to be specified")
        with Path(filename).open() as file:
            code = file.read()

    if model_name is None:
        model_name = "model"

    basedir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    try:
        model_path = (
            Path(basedir.name)
            .joinpath("name")
            .with_name(model_name)  # This verifies that it is a valid filename
            .with_suffix(".stan")
        )
        model_path.write_text(code)
        make_args = ["STAN_THREADS=true"]
        if extra_compile_args:
            make_args.extend(extra_compile_args)
        stanc_args = []
        if extra_stanc_args:
            stanc_args.extend(extra_stanc_args)
        so_path = bridgestan.compile_model(
            model_path, make_args=make_args, stanc_args=stanc_args
        )
        # Set necessary library loading paths
        bridgestan.compile.windows_dll_path_setup()
        library = _lib.StanLibrary(so_path)
    finally:
        try:
            if cleanup:
                basedir.cleanup()
        except Exception:  # noqa: BLE001
            pass

    return CompiledStanModel(
        code=code,
        library=library,
        dims=dims,
        _coords=coords,
        model_name=model_name,
        model=None,
        data=None,
    )
