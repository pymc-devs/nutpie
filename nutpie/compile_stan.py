from dataclasses import dataclass
from math import prod
from typing import Any, Dict

from numpy.typing import NDArray

from .sample import CompiledModel
from . import lib


@dataclass(frozen=True)
class CompiledStanModel(CompiledModel):
    code: str
    data: Dict[str, NDArray]
    lib: Any


def compile_stan_model(
    data,
    *,
    code=None,
    filename=None,
    cache=True,
    extra_compile_args=None,
    show_compiler_output=False,
    dims=None,
    coords=None,
):
    import httpstan.models

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

    model_id = httpstan.models.calculate_model_name(code)
    stan_lib = None
    if cache and not extra_compile_args:
        try:
            stan_lib = httpstan.models.import_services_extension_module(model_id)
        except (FileNotFoundError, KeyError):
            pass
    if stan_lib is None:
        output = httpstan.models.build_services_extension_module_sync(code, extra_compile_args)
        if show_compiler_output:
            print(output)
        stan_lib = httpstan.models.import_services_extension_module(model_id)

    ctx = stan_lib.new_logp_ctx(data)
    n_dim = stan_lib.num_unconstrained_parameters(ctx)

    shape_info = _make_shape_info(stan_lib, data)

    logp_maker = _make_logp_maker(stan_lib, data)

    def expanding_function(x, seed, chain, draw):
        return stan_lib.write_array_ctx(ctx, x, True, True, seed + 10000 * chain + draw)

    return CompiledStanModel(
        n_dim,
        dims,
        coords,
        shape_info,
        logp_maker,
        expanding_function,
        code,
        data,
        stan_lib,
    )


def _make_shape_info(model, data):
    slices = []
    shapes = []
    count = 0
    for shape in model.get_dims(data):
        shapes.append(shape)
        length = prod(shape)
        slices.append(slice(count, count + length))
        count += length

    names = model.get_param_names(data)
    return names, slices, shapes


def _make_logp_maker(stanlib, data):
    ctx = stanlib.new_logp_ctx(data)
    n_dim = stanlib.num_unconstrained_parameters(ctx)
    stanlib.free_logp_ctx(ctx)

    def make_logp_pyfn(args):
        stanlib, data = args
        ctx = stanlib.new_logp_ctx(data)
        func_ptr = stanlib.logp_func(ctx)
        return func_ptr, ctx, (stanlib, data)

    return lib.PtrLogpFuncMaker(make_logp_pyfn, (stanlib, data), n_dim, (stanlib, data))


