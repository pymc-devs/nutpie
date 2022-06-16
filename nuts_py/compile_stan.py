from math import prod

import httpstan.models

from .sample import CompiledModel


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
    lib = None
    if cache and not extra_compile_args:
        try:
            lib = httpstan.models.import_services_extension_module(model_id)
        except (FileNotFoundError, KeyError):
            pass
    if lib is None:
        output = httpstan.models.build_services_extension_module_sync(code, extra_compile_args)
        if show_compiler_output:
            print(output)
        lib = httpstan.models.import_services_extension_module(model_id)

    ctx = lib.new_logp_ctx(data)
    n_dim = lib.num_unconstrained_parameters(ctx)
    logp = lib.logp_func(ctx)

    def make_shape_info(model, data):
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
    
    shape_info = make_shape_info(lib, data)

    def make_user_data():
        return lib.new_logp_ctx(data)
    
    def expanding_function(x):
        return lib.write_array_ctx(ctx, x, True, True, 0)

    return CompiledModel(
        code,
        n_dim,
        logp,
        expanding_function,
        make_user_data,
        shape_info,
        dims,
        coords,
        (lib, data),
    )
