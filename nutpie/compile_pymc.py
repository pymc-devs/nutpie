from math import prod

import aesara
import aesara.tensor as at
import pymc as pm
import numpy as np
import numba
from aeppl.logprob import CheckParameterValue
import aesara.link.numba.dispatch

from .sample import CompiledModel

# Provide a numba implementation for CheckParameterValue, which doesn't exist in aesara
@aesara.link.numba.dispatch.numba_funcify.register(CheckParameterValue)
def numba_functify_CheckParameterValue(op, **kwargs):
    @aesara.link.numba.dispatch.basic.numba_njit
    def check(value, *conditions):
        return value
    
    return check

# Overwrite the IncSubtensor op from aesara, see https://github.com/aesara-devs/aesara/issues/603
@aesara.link.numba.dispatch.numba_funcify.register(at.subtensor.AdvancedIncSubtensor1)
def numba_funcify_IncSubtensor(op, node, **kwargs):

    def incsubtensor_fn(z, vals, idxs):
        z = z.copy()
        for idx, val in zip(idxs, vals):
            z[idx] += val
        return z

    return aesara.link.numba.dispatch.basic.numba_njit(incsubtensor_fn)



def compile_pymc_model(model, **kwargs):
    """Compile necessary functions for sampling a pymc model."""
    n_dim, logp_func, expanding_function, shape_info = _make_functions(model)
    logp_func = numba.njit(**kwargs)(logp_func)
    logp_numba_raw, c_sig = _make_c_logp_func(n_dim, logp_func)
    logp_numba = numba.cfunc(c_sig, **kwargs)(logp_numba_raw)

    def expand_draw(x):
        return expanding_function(x)[0]

    def make_user_data():
        return 0

    return CompiledModel(
        model,
        n_dim,
        logp_numba.address,
        expand_draw,
        make_user_data,
        shape_info,
        model.RV_dims,
        model.coords,
        logp_numba,
    )


def _compute_shapes(model):
    point = pm.model.make_initial_point_fn(model=model, return_transformed=True)(0)

    trace_vars = {
        name: var
        for (name, var) in model.named_vars.items()
        if var not in model.observed_RVs + model.potentials
    }

    shape_func = aesara.function(
        inputs=[],
        outputs=[var.shape for var in trace_vars.values()],
        givens=(
            [(obs, obs.tag.observations) for obs in model.observed_RVs]
            + [
                (trace_vars[name], point[name])
                for name in trace_vars.keys()
                if name in point
            ]
        ),
        mode=aesara.compile.mode.FAST_COMPILE,
        on_unused_input="ignore",
    )
    return {name: shape for name, shape in zip(trace_vars.keys(), shape_func())}


def _make_functions(model):
    shapes = _compute_shapes(model)

    # Make logp_dlogp_function
    joined = at.dvector("__joined_variables")

    value_vars = [model.rvs_to_values[var] for var in model.free_RVs]

    logp = model.logpt()
    grads = at.grad(logp, value_vars)
    grad = at.concatenate([grad.ravel() for grad in grads])

    count = 0
    joined_slices = []
    joined_shapes = []
    joined_names = []

    symbolic_sliced = []
    for var in model.free_RVs:
        value_var = model.rvs_to_values[var]

        joined_names.append(value_var.name)
        shape = shapes[value_var.name]
        joined_shapes.append(shape)
        length = prod(shape)
        slice_val = slice(count, count + length)
        joined_slices.append(slice_val)
        symbolic_sliced.append((value_var, joined[slice_val].reshape(shape)))
        count += length

    num_free_vars = count

    # We should avoid compiling the function, and optimize only
    func = aesara.function(
        (joined,), (logp, grad), givens=symbolic_sliced, mode=aesara.compile.NUMBA
    )

    logp_func = func.vm.jit_fn.py_func

    # Make function that computes remaining variables for the trace
    trace_vars = {
        name: var
        for (name, var) in model.named_vars.items()
        if var not in model.observed_RVs + model.potentials
    }
    remaining_names = [name for name in trace_vars if name not in joined_names]
    remaining_rvs = [
        var for var in model.unobserved_value_vars if var.name not in joined_names
    ]

    all_names = joined_names + remaining_rvs

    all_names = joined_names.copy()
    all_slices = joined_slices.copy()
    all_shapes = joined_shapes.copy()

    for var in remaining_rvs:
        all_names.append(var.name)
        shape = shapes[var.name]
        all_shapes.append(shape)
        length = prod(shape)
        all_slices.append(slice(count, count + length))
        count += length

    allvars = at.concatenate([joined, *[var.ravel() for var in remaining_rvs]])
    func = aesara.function(
        (joined,), (allvars,), givens=symbolic_sliced, mode=aesara.compile.NUMBA
    )
    func = func.vm.jit_fn.py_func
    expanding_function = numba.njit(func, fastmath=True, error_model="numpy")
    expanding_function(np.zeros(num_free_vars))

    return (
        num_free_vars,
        logp_func,
        expanding_function,
        (all_names, all_slices, all_shapes),
    )


def _make_c_logp_func(N, logp_func):
    c_sig = numba.types.int64(
        numba.types.uint64,
        numba.types.CPointer(numba.types.double),
        numba.types.CPointer(numba.types.double),
        numba.types.CPointer(numba.types.double),
        numba.types.voidptr,
    )

    def logp_numba(dim, x_, out_, logp_, user_data_):
        try:
            x = numba.carray(x_, (N,))
            out = numba.carray(out_, (N,))
            logp = numba.carray(logp_, ())

            logp_val, grad = logp_func(x)
            logp[()] = logp_val
            out[...] = grad

            if not np.all(np.isfinite(out)):
                return 2
            if not np.isfinite(logp_val):
                return 3
            if np.any(out == 0):
                return 4
        except Exception:
            return 1
        return 0

    return logp_numba, c_sig
