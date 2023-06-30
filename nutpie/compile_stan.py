import json
import pathlib
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from nutpie import lib
from nutpie.sample import CompiledModel


class _NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@dataclass(frozen=True)
class CompiledStanModel(CompiledModel):
    _coords: Optional[Dict[str, Any]]
    code: str
    data: Optional[Dict[str, NDArray]]
    library: Any
    model: Any
    model_name: Optional[str] = None

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

        model = lib.StanModel(self.library, seed, data_json)
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

    def _make_model(self, init_mean):
        if self.model is None:
            return self.with_data().model
        return self.model

    def _make_sampler(self, settings, init_mean, chains, cores, seed):
        model = self._make_model(init_mean)
        return lib.PySampler.from_stan(settings, chains, cores, model, seed)

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
    code=None,
    filename=None,
    extra_compile_args=None,
    dims=None,
    coords=None,
    model_name=None,
):
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
        with open(filename, "r") as file:
            code = file.read()

    if model_name is None:
        model_name = "model"

    with tempfile.TemporaryDirectory() as basedir:
        model_path = (
            pathlib.Path(basedir)
            .joinpath("name")
            .with_name(model_name)  # This verifies that it is a valid filename
            .with_suffix(".stan")
        )
        model_path.write_text(code)
        make_args = ["STAN_THREADS=true"]
        if extra_compile_args:
            make_args.extend(extra_compile_args)
        so_path = bridgestan.compile_model(model_path, make_args=make_args)
        # Set necessary library loading paths
        bridgestan.compile.windows_dll_path_setup()
        library = lib.StanLibrary(so_path)

    # One the library is loaded we can delete the temporary dir
    # TODO: Is this also true on Windows?
    return CompiledStanModel(
        code=code,
        library=library,
        dims=dims,
        _coords=coords,
        model_name=model_name,
        model=None,
        data=None,
    )
