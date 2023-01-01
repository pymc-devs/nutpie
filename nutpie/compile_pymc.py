from dataclasses import dataclass
import dataclasses
import functools
from math import prod
from typing import Dict, List

import pytensor
import pytensor.tensor as pt
from numpy.typing import NDArray
import pymc as pm
import numpy as np
import numba
from pytensor.raise_op import CheckAndRaise
import pytensor.link.numba.dispatch
from numba import literal_unroll
from numba.cpython.unsafe.tuple import alloca_once, tuple_setitem
import numba.core.ccallback

from .sample import CompiledModel
from . import lib

# Provide a numba implementation for CheckParameterValue,
# which doesn't exist in pytensor
@pytensor.link.numba.dispatch.basic.numba_funcify.register(CheckAndRaise)
def numba_functify_CheckAndRaise(op, **kwargs):
    msg = f"Invalid parameter value {str(op)}"

    @pytensor.link.numba.dispatch.basic.numba_njit
    def check(value, *conditions):
        for cond in literal_unroll(conditions):
            if not cond:
                raise ValueError(msg)
        return value

    return check


@numba.extending.intrinsic
def address_as_void_pointer(typingctx, src):
    """returns a void pointer from a given memory address"""
    from numba.core import types, cgutils

    sig = types.voidptr(src)

    def codegen(cgctx, builder, sig, args):
        return builder.inttoptr(args[0], cgutils.voidptr_t)

    return sig, codegen


@dataclass(frozen=True)
class CompiledPyMCModel(CompiledModel):
    compiled_logp_func: numba.core.ccallback.CFunc
    shared_data: Dict[str, NDArray]
    user_data: NDArray

    def with_data(self, **updates):
        shared_data = self.shared_data.copy()
        user_data = self.user_data.copy()
        for name, new_val in updates.items():
            if name not in shared_data:
                raise KeyError(f"Unknown shared variable: {name}")
            old_val = shared_data[name]
            new_val = np.asarray(new_val, dtype=old_val.dtype).copy()
            new_val.flags.writeable = False
            if old_val.ndim != new_val.ndim:
                raise ValueError(
                    f"Shared variable {name} must have rank {old_val.ndim}"
                )
            shared_data[name] = new_val
        user_data = update_user_data(user_data, shared_data)

        logp_func_maker = self.logp_func_maker.with_arg(user_data.ctypes.data)
        expand_draw_fn = functools.partial(
            self.expand_draw_fn.func, shared_data=shared_data
        )
        return dataclasses.replace(
            self,
            shared_data=shared_data,
            user_data=user_data,
            logp_func_maker=logp_func_maker,
            expand_draw_fn=expand_draw_fn,
        )


def update_user_data(user_data, user_data_storage):
    user_data = user_data[()]
    for name, val in user_data_storage.items():
        user_data["shared"]["data"][name] = val.ctypes.data
        user_data["shared"]["size"][name] = val.size
        user_data["shared"]["shape"][name] = val.shape
    return np.asarray(user_data)


def make_user_data(func, shared_data):
    shared_vars = func.get_shared()
    record_dtype = np.dtype(
        [
            (
                "shared",
                [
                    ("data", [(var.name, np.uintp) for var in shared_vars]),
                    ("size", [(var.name, np.uintp) for var in shared_vars]),
                    (
                        "shape",
                        [(var.name, np.uint, (var.ndim,)) for var in shared_vars],
                    ),
                ],
            )
        ],
        align=True,
    )
    user_data = np.zeros((), dtype=record_dtype)
    update_user_data(user_data, shared_data)
    return user_data


def compile_pymc_model(model, **kwargs):
    """Compile necessary functions for sampling a pymc model."""

    n_dim, logp_fn_at, logp_fn, expand_fn, shared_expand, shape_info = _make_functions(
        model
    )

    shared_data = {val.name: val.get_value().copy() for val in logp_fn_pt.get_shared()}
    for val in shared_data.values():
        val.flags.writeable = False

    shared_logp = [var.name for var in logp_fn_pt.get_shared()]

    user_data = make_user_data(logp_fn_at, shared_data)

    logp_numba_raw, c_sig = _make_c_logp_func(
        n_dim, logp_fn, user_data, shared_logp, shared_data
    )
    logp_numba = numba.cfunc(c_sig, **kwargs)(logp_numba_raw)

    def expand_draw(x, seed, chain, draw, *, shared_data):
        return expand_fn(x, **{name: shared_data[name] for name in shared_expand})[0]

    def make_logp_pyfn(data_ptr):
        return logp_numba.address, data_ptr, None

    logp_func_maker = lib.PtrLogpFuncMaker(
        make_logp_pyfn,
        user_data.ctypes.data,
        n_dim,
        logp_numba,
    )

    expand_draw_fn = functools.partial(expand_draw, shared_data=shared_data)

    return CompiledPyMCModel(
        n_dim=n_dim,
        dims=model.RV_dims,
        coords=model.coords,
        shape_info=shape_info,
        logp_func_maker=logp_func_maker,
        expand_draw_fn=expand_draw_fn,
        compiled_logp_func=logp_numba,
        shared_data=shared_data,
        user_data=user_data,
    )


def _compute_shapes(model):
    point = pm.model.make_initial_point_fn(model=model, return_transformed=True)(0)

    trace_vars = {
        name: var
        for (name, var) in model.named_vars.items()
        if var not in model.observed_RVs + model.potentials
    }

    shape_func = pytensor.compile.function.function(
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
        mode=pytensor.compile.mode.FAST_COMPILE,
        on_unused_input="ignore",
    )
    return {name: shape for name, shape in zip(trace_vars.keys(), shape_func())}


def _make_functions(model):
    shapes = _compute_shapes(model)

    # Make logp_dlogp_function
    joined = pt.dvector("__joined_variables")

    value_vars = [model.rvs_to_values[var] for var in model.free_RVs]

    logp = model.logp()
    grads = pytensor.gradient.grad(logp, value_vars)
    grad = pt.concatenate([grad.ravel() for grad in grads])

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
    logp_fn_at = pytensor.compile.function.function(
        (joined,), (logp, grad), givens=symbolic_sliced, mode=pytensor.compile.NUMBA
    )

    logp_fn = logp_fn_pt.vm.jit_fn

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

    allvars = pt.concatenate([joined, *[var.ravel() for var in remaining_rvs]])
    expand_fn_at = pytensor.compile.function.function(
        (joined,), (allvars,), givens=symbolic_sliced, mode=pytensor.compile.NUMBA
    )
    expand_fn = expand_fn_pt.vm.jit_fn
    # expand_fn = numba.njit(expand_fn, fastmath=True, error_model="numpy")
    # Trigger a compile
    expand_fn(np.zeros(num_free_vars), *[var.get_value() for var in expand_fn_pt.get_shared()])

    return (
        num_free_vars,
        logp_fn_at,
        logp_fn,
        expand_fn,
        [var.name for var in expand_fn_pt.get_shared()],
        (all_names, all_slices, all_shapes),
    )


def make_extraction_fn(inner, shared_data, shared_vars, record_dtype):
    if not shared_vars:

        @numba.njit(inline="always")
        def extract_shared(x, user_data_):
            return inner(x)

        return extract_shared

    shared_metadata = tuple(
        [
            name,
            len(shared_data[name].shape),
            shared_data[name].shape,
            np.dtype(shared_data[name].dtype),
        ]
        for name in shared_vars
    )

    names = shared_vars
    indices = tuple(range(len(names)))
    shared_tuple = tuple(shared_data[name] for name in shared_vars)

    @numba.extending.intrinsic
    def tuple_setitem_literal(typingctx, tup, idx, val):
        """Return a copy of the tuple with item at *idx* replaced with *val*.
        """
        if not isinstance(idx, numba.types.IntegerLiteral):
            return

        idx_val = idx.literal_value
        assert idx_val >= 0
        assert idx_val < len(tup)

        import llvmlite

        def codegen(context, builder, signature, args):
            tup, idx, val = args
            stack = alloca_once(builder, tup.type)
            builder.store(tup, stack)
            # Unsafe load on unchecked bounds.  Poison value maybe returned.
            tuple_idx = llvmlite.ir.IntType(32)(idx_val)
            offptr = builder.gep(stack, [idx.type(0), tuple_idx], inbounds=True)
            builder.store(val, offptr)
            return builder.load(stack)

        sig = tup(tup, idx, tup[idx_val])
        return sig, codegen

    def extract_array(user_data, index):
        pass

    @numba.extending.overload(extract_array, inline="always")
    def impl_extract_array(user_data, index):
        if not isinstance(index, numba.types.Literal):
            return

        index = index.literal_value

        name, ndim, base_shape, dtype = shared_metadata[index]

        ndim_range = tuple(range(ndim))

        def impl(user_data, index):
            data_ptr = address_as_void_pointer(user_data["data"][name][()])
            data = numba.carray(data_ptr, int(user_data["size"][name][()]), dtype)

            shape = user_data["shape"][name]

            assert len(shape) == len(base_shape)

            shape_ = base_shape

            # For some reason I get typing errors without this if condition
            if ndim > 0:
                for i in range(ndim):
                    shape_ = tuple_setitem(shape_, i, shape[i])

            return data.reshape(shape_)

        return impl

    @numba.njit(inline="always")
    def extract_shared(x, user_data_):
        user_data = numba.carray(user_data_, (), record_dtype)

        _shared_tuple = shared_tuple
        for index in literal_unroll(indices):
            dat = extract_array(user_data["shared"], index)
            _shared_tuple = tuple_setitem_literal(_shared_tuple, index, dat)

        return inner(x, *_shared_tuple)

    return extract_shared


def _make_c_logp_func(n_dim, logp_fn, user_data, shared_logp, shared_data):

    extract = make_extraction_fn(logp_fn, shared_data, shared_logp, user_data.dtype)

    c_sig = numba.types.int64(
        numba.types.uint64,
        numba.types.CPointer(numba.types.double),
        numba.types.CPointer(numba.types.double),
        numba.types.CPointer(numba.types.double),
        numba.types.voidptr,
    )

    def logp_numba(dim, x_, out_, logp_, user_data_):
        if dim != n_dim:
            return -1

        try:
            x = numba.carray(x_, (n_dim,))
            out = numba.carray(out_, (n_dim,))
            logp = numba.carray(logp_, ())

            logp_val, grad = extract(x, user_data_)
            logp[()] = logp_val
            out[...] = grad

            if not np.all(np.isfinite(out)):
                return 3
            if not np.isfinite(logp_val):
                return 4
            # if np.any(out == 0):
            #    return 4
        except Exception:
            return 1
        return 0

    return logp_numba, c_sig
