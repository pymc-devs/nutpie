import dataclasses
import itertools
import threading
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from functools import wraps
from importlib.util import find_spec
from math import prod
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Union, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from nutpie import _lib
from nutpie.compiled_pyfunc import SeedType, from_pyfunc
from nutpie.sample import CompiledModel

try:
    from numba.extending import intrinsic
except ImportError:

    def intrinsic(f):
        return f


if TYPE_CHECKING:
    import numba.core.ccallback
    import pymc as pm
    from pytensor.tensor import TensorVariable, Variable


def _rv_dict_to_flat_array_wrapper(
    fn: Callable[[SeedType | None], dict[str, np.ndarray]],
    names: list[str],
    shapes: list[tuple[int]],
) -> Callable[[SeedType], np.ndarray]:
    """
    Wraps a function that returns a dictionary of string:array key:value pairs
    and returns a single flat float64 array. Also checks that the shapes of
    the arrays match the expected shapes.

    Parameters
    ----------
    fn: Callable
        Function that takes a seed and return a dictionary of variable names
        to initial values. This function should be the output of
        pymc.initial_point.make_initial_point_fn
    names: list of str
        List of random variable names in the model
    shapes: list of tuple of int
        Shape of random variables in the model

    Returns
    -------
    seeded_array_fn: Callable
        Function that takes a seed and returns a flat, contiguous float64
        array of initial values. The ordering of the random variables inside
        the array is controlled by the ``names`` parameter.
    """

    @wraps(fn)
    def seeded_array_fn(seed: SeedType | None = None):
        initial_value_dict = fn(seed)
        total_size = sum(np.prod(shape).astype(int) for shape in shapes)
        flat_array = np.empty(total_size, dtype="float64", order="C")
        cursor = 0

        for name, shape in zip(names, shapes, strict=True):
            initial_value = initial_value_dict[name]
            n = int(np.prod(initial_value.shape))
            if tuple(initial_value.shape) != tuple(shape):
                raise ValueError(
                    f"Size of initial value for {name} is {initial_value.shape}, "
                    f"expected {shape}"
                )

            flat_array[cursor : cursor + n] = initial_value.ravel().astype("float64")
            cursor += n

        return flat_array

    return seeded_array_fn


@intrinsic
def address_as_void_pointer(typingctx, src):
    """returns a void pointer from a given memory address"""
    from numba.core import cgutils, types

    sig = types.voidptr(src)

    def codegen(cgctx, builder, sig, args):
        return builder.inttoptr(args[0], cgutils.voidptr_t)

    return sig, codegen


@dataclass(frozen=True)
class CompiledPyMCModel(CompiledModel):
    compiled_logp_func: "numba.core.ccallback.CFunc"
    compiled_expand_func: "numba.core.ccallback.CFunc"
    initial_point_func: Callable[[SeedType], np.ndarray]
    shared_data: dict[str, NDArray]
    user_data: NDArray
    n_expanded: int
    shape_info: Any
    logp_func: Any
    expand_func: Any
    _n_dim: int
    _shapes: dict[str, tuple[int, ...]]
    _coords: Optional[dict[str, Any]]

    @property
    def n_dim(self):
        return self._n_dim

    @property
    def shapes(self):
        return self._shapes

    @property
    def coords(self):
        return self._coords

    def with_data(self, **updates):
        shared_data = self.shared_data.copy()
        user_data = self.user_data.copy()
        for name, new_val in updates.items():
            if name not in shared_data:
                raise KeyError(f"Unknown shared variable: {name}")
            old_val = shared_data[name]
            new_val = np.array(new_val, dtype=old_val.dtype, order="C", copy=True)
            new_val.flags.writeable = False
            if old_val.ndim != new_val.ndim:
                raise ValueError(
                    f"Shared variable {name} must have rank {old_val.ndim}"
                )
            shared_data[name] = new_val
        user_data = update_user_data(user_data, shared_data)

        return dataclasses.replace(
            self,
            shared_data=shared_data,
            user_data=user_data,
        )

    def _make_sampler(self, settings, init_mean, cores, progress_type):
        model = self._make_model(init_mean)
        return _lib.PySampler.from_pymc(
            settings,
            cores,
            model,
            progress_type,
        )

    def _make_model(self, init_mean):
        expand_fn = _lib.ExpandFunc(
            self.n_dim,
            self.n_expanded,
            self.compiled_expand_func.address,
            self.user_data.ctypes.data,
            self,
        )
        logp_fn = _lib.LogpFunc(
            self.n_dim,
            self.compiled_logp_func.address,
            self.user_data.ctypes.data,
            self,
        )

        var_sizes = [prod(shape) for shape in self.shape_info[2]]
        var_names = self.shape_info[0]

        return _lib.PyMcModel(
            self.n_dim,
            logp_fn,
            expand_fn,
            self.initial_point_func,
            var_sizes,
            var_names,
        )


def update_user_data(user_data, user_data_storage):
    user_data = user_data[()]
    for name, val in user_data_storage.items():
        user_data["shared"]["data"][name] = val.ctypes.data
        user_data["shared"]["size"][name] = val.size
        user_data["shared"]["shape"][name] = val.shape
    return np.asarray(user_data)


def make_user_data(shared_vars, shared_data):
    record_dtype = np.dtype(
        [
            (
                "shared",
                [
                    ("data", [(var_name, np.uintp) for var_name in shared_vars]),
                    ("size", [(var_name, np.uintp) for var_name in shared_vars]),
                    (
                        "shape",
                        [
                            (var_name, np.uint, (var.ndim,))
                            for var_name, var in shared_vars.items()
                        ],
                    ),
                ],
            )
        ],
    )
    user_data = np.zeros((), dtype=record_dtype)
    update_user_data(user_data, shared_data)
    return user_data


def _compile_pymc_model_numba(
    model: "pm.Model",
    pymc_initial_point_fn: Callable[[SeedType], dict[str, np.ndarray]],
    var_names: Iterable[str] | None = None,
    **kwargs,
) -> CompiledPyMCModel:
    if find_spec("numba") is None:
        raise ImportError(
            "Numba is not installed in the current environment. "
            "Please install it with something like "
            "'mamba install -c conda-forge numba' "
            "and restart your kernel in case you are in an interactive session."
        )
    import numba

    (
        n_dim,
        n_expanded,
        logp_fn_pt,
        expand_fn_pt,
        initial_point_fn,
        shape_info,
    ) = _make_functions(
        model,
        mode="NUMBA",
        compute_grad=True,
        join_expanded=True,
        pymc_initial_point_fn=pymc_initial_point_fn,
        var_names=var_names,
    )

    expand_fn = expand_fn_pt.vm.jit_fn
    logp_fn = logp_fn_pt.vm.jit_fn

    shared_data = {}
    shared_vars = {}
    seen = set()
    for val in [*logp_fn_pt.get_shared(), *expand_fn_pt.get_shared()]:
        if val.name in shared_data and val not in seen:
            raise ValueError(f"Shared variables must have unique names: {val.name}")
        shared_data[val.name] = np.array(val.get_value(), order="C", copy=True)
        shared_vars[val.name] = val
        seen.add(val)

    for val in shared_data.values():
        val.flags.writeable = False

    user_data = make_user_data(shared_vars, shared_data)

    logp_shared_names = [var.name for var in logp_fn_pt.get_shared()]
    logp_numba_raw, c_sig = _make_c_logp_func(
        n_dim, logp_fn, user_data, logp_shared_names, shared_data
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Cannot cache compiled function .* as it uses dynamic globals",
            category=numba.NumbaWarning,  # type: ignore
        )

        logp_numba = numba.cfunc(c_sig, **kwargs)(logp_numba_raw)

    expand_shared_names = [var.name for var in expand_fn_pt.get_shared()]
    expand_numba_raw, c_sig_expand = _make_c_expand_func(
        n_dim, n_expanded, expand_fn, user_data, expand_shared_names, shared_data
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Cannot cache compiled function .* as it uses dynamic globals",
            category=numba.NumbaWarning,  # type: ignore
        )

        expand_numba = numba.cfunc(c_sig_expand, **kwargs)(expand_numba_raw)

    dims, coords = _prepare_dims_and_coords(model, shape_info)

    return CompiledPyMCModel(
        _n_dim=n_dim,
        dims=dims,
        _coords=coords,
        _shapes={name: tuple(shape) for name, _, shape in zip(*shape_info)},
        compiled_logp_func=logp_numba,
        compiled_expand_func=expand_numba,
        initial_point_func=initial_point_fn,
        shared_data=shared_data,
        user_data=user_data,
        n_expanded=n_expanded,
        shape_info=shape_info,
        logp_func=logp_fn_pt,
        expand_func=expand_fn_pt,
    )


def _prepare_dims_and_coords(model, shape_info):
    coords = {}
    for name, vals in model.coords.items():
        if vals is None:
            vals = pd.RangeIndex(int(model.dim_lengths[name].eval()))
        coords[name] = pd.Index(vals)

    if "unconstrained_parameter" in coords:
        raise ValueError("Model contains invalid name 'unconstrained_parameter'.")

    names = []
    for base, _, shape in zip(*shape_info):
        if base not in [var.name for var in model.value_vars]:
            continue
        for idx in itertools.product(*[range(length) for length in shape]):
            if len(idx) == 0:
                names.append(base)
            else:
                names.append(f"{base}_{'.'.join(str(i) for i in idx)}")
    coords["unconstrained_parameter"] = pd.Index(names)

    dims = model.named_vars_to_dims
    return dims, coords


def _compile_pymc_model_jax(
    model,
    *,
    gradient_backend=None,
    pymc_initial_point_fn: Callable[[SeedType], dict[str, np.ndarray]],
    var_names: Iterable[str] | None = None,
    **kwargs,
):
    if find_spec("jax") is None:
        raise ImportError(
            "Jax is not installed in the current environment. "
            "Please install it with something like "
            "'mamba install -c conda-forge jax' "
            "and restart your kernel in case you are in an interactive session."
        )
    import jax

    if gradient_backend is None:
        gradient_backend = "pytensor"
    elif gradient_backend not in ["jax", "pytensor"]:
        raise ValueError(f"Unknown gradient backend: {gradient_backend}")

    (
        n_dim,
        _,
        logp_fn_pt,
        expand_fn_pt,
        initial_point_fn,
        shape_info,
    ) = _make_functions(
        model,
        mode="JAX",
        compute_grad=gradient_backend == "pytensor",
        join_expanded=False,
        pymc_initial_point_fn=pymc_initial_point_fn,
        var_names=var_names,
    )

    logp_fn = logp_fn_pt.vm.jit_fn
    expand_fn = expand_fn_pt.vm.jit_fn

    logp_shared_names = [var.name for var in logp_fn_pt.get_shared()]
    expand_shared_names = [var.name for var in expand_fn_pt.get_shared()]

    if gradient_backend == "jax":
        orig_logp_fn = logp_fn._fun

        def logp_fn_jax_grad(x, *shared):
            return jax.value_and_grad(lambda x: orig_logp_fn(x, *shared)[0])(x)

        # static_argnums = list(range(1, len(logp_shared_names) + 1))
        logp_fn_jax_grad = jax.jit(
            logp_fn_jax_grad,
            # static_argnums=static_argnums,
        )

        logp_fn = logp_fn_jax_grad
    else:
        orig_logp_fn = None

    shared_data = {}
    shared_vars = {}
    seen = set()
    for val in [*logp_fn_pt.get_shared(), *expand_fn_pt.get_shared()]:
        if val.name in shared_data and val not in seen:
            raise ValueError(f"Shared variables must have unique names: {val.name}")
        shared_data[val.name] = jax.numpy.asarray(val.get_value())
        shared_vars[val.name] = val
        seen.add(val)

    def make_logp_func():
        def logp(_x, **shared):
            logp, grad = logp_fn(_x, *[shared[name] for name in logp_shared_names])
            return float(logp), np.asarray(grad, dtype="float64", order="C")

        return logp

    names, slices, shapes = shape_info
    # TODO do not cast to float64
    dtypes = [np.dtype("float64")] * len(names)

    def make_expand_func(seed1, seed2, chain):
        # TODO handle seeds
        def expand(_x, **shared):
            values = expand_fn(_x, *[shared[name] for name in expand_shared_names])
            return {
                name: np.asarray(val, order="C", dtype=dtype).ravel()
                for name, val, dtype in zip(names, values, dtypes, strict=True)
            }

        return expand

    dims, coords = _prepare_dims_and_coords(model, shape_info)

    return from_pyfunc(
        ndim=n_dim,
        make_logp_fn=make_logp_func,
        make_expand_fn=make_expand_func,
        make_initial_point_fn=initial_point_fn,
        expanded_dtypes=dtypes,
        expanded_shapes=shapes,
        expanded_names=names,
        shared_data=shared_data,
        dims=dims,
        coords=coords,
        raw_logp_fn=orig_logp_fn,
    )


def compile_pymc_model(
    model: "pm.Model",
    *,
    backend: Literal["numba", "jax"] = "numba",
    gradient_backend: Literal["pytensor", "jax"] = "pytensor",
    initial_points: dict[Union["Variable", str], np.ndarray | float | int]
    | None = None,
    jitter_rvs: set["TensorVariable"] | None = None,
    default_initialization_strategy: Literal[
        "support_point", "prior"
    ] = "support_point",
    var_names: Iterable[str] | None = None,
    freeze_model: bool | None = None,
    **kwargs,
) -> CompiledModel:
    """Compile necessary functions for sampling a pymc model.

    Parameters
    ----------
    model : pymc.Model
        The model to compile.
    backend : ["jax", "numba"]
        The pytensor backend that is used to compile the logp function.
    gradient_backend: ["pytensor", "jax"]
        Which library is used to compute the gradients. This can only be changed
        to "jax" if the jax backend is used.
    jitter_rvs : set
        The set (or list or tuple) of random variables for which a U(-1, +1)
        jitter should be added to the initial value. Only available for
        variables that have a transform or real-valued support.
    default_initialization_strategy : str
        Which of { "support_point", "prior" } to prefer if the initval setting
        for an RV is None.
    initial_points : dict
        Initial value (strategies) to use instead of what's specified in
        `Model.initial_values`.
    var_names : list[str] | None
        A list of variables to store in the trace. If None, store all variables.
    freeze_model : bool | None
        Freeze all dimensions and shared variables to treat them as compile time
        constants.
    Returns
    -------
    compiled_model : CompiledPyMCModel
        A compiled model object.

    """
    if find_spec("pymc") is None:
        raise ImportError(
            "PyMC is not installed in the current environment. "
            "Please install it with something like "
            "'mamba install -c conda-forge pymc numba' "
            "and restart your kernel in case you are in an interactive session."
        )

    if gradient_backend is not None:
        gradient_backend = gradient_backend.lower()  # type: ignore[assignment]
    if backend is not None:
        backend = backend.lower()  # type: ignore[assignment]

    from pymc.model.transform.optimization import freeze_dims_and_data
    from pymc.initial_point import make_initial_point_fn

    if freeze_model is None:
        freeze_model = backend == "jax"

    if freeze_model:
        model = freeze_dims_and_data(model)

    if default_initialization_strategy == "support_point" and jitter_rvs is None:
        jitter_rvs = set(model.free_RVs)

    initial_point_fn = make_initial_point_fn(
        model=model,
        overrides=initial_points,
        default_strategy=default_initialization_strategy,
        jitter_rvs=jitter_rvs,
        return_transformed=True,
    )

    initial_point_fn = _wrap_with_lock(initial_point_fn)

    if backend.lower() == "numba":
        if gradient_backend == "jax":
            raise ValueError("Gradient backend cannot be jax when using numba backend")
        return _compile_pymc_model_numba(
            model=model,
            pymc_initial_point_fn=initial_point_fn,
            var_names=var_names,
            **kwargs,
        )
    elif backend.lower() == "jax":
        return _compile_pymc_model_jax(
            model=model,
            gradient_backend=gradient_backend,
            pymc_initial_point_fn=initial_point_fn,
            var_names=var_names,
            **kwargs,
        )
    else:
        raise ValueError(f"Backend must be one of numba and jax. Got {backend}")


def _wrap_with_lock(func: Callable) -> Callable:
    lock = threading.Lock()

    @wraps(func)
    def wrapper(*args, **kwargs):
        with lock:
            return func(*args, **kwargs)

    return wrapper


def _compute_shapes(model) -> dict[str, tuple[int, ...]]:
    import pytensor
    from pymc.initial_point import make_initial_point_fn

    point = make_initial_point_fn(model=model, return_transformed=True)(0)

    trace_vars = {
        var.name: var
        for var in model.value_vars + model.free_RVs + model.deterministics
        if var not in model.observed_RVs + model.potentials
    }

    shape_func = pytensor.compile.function.function(
        inputs=[],
        outputs=[var.shape for var in trace_vars.values()],
        givens=(
            [(obs, model.rvs_to_values[obs]) for obs in model.observed_RVs]
            + [
                (trace_vars[name], point[name])
                for name in trace_vars.keys()
                if name in point
            ]
        ),
        mode=pytensor.compile.mode.FAST_COMPILE,
        on_unused_input="ignore",
    )
    return dict(zip(trace_vars.keys(), shape_func()))


def _make_functions(
    model: "pm.Model",
    *,
    mode: Literal["JAX", "NUMBA"],
    compute_grad: bool,
    join_expanded: bool,
    pymc_initial_point_fn: Callable[[SeedType], dict[str, np.ndarray]],
    var_names: Iterable[str] | None = None,
) -> tuple[
    int,
    int,
    Callable,
    Callable,
    Callable,
    tuple[list[str], list[slice], list[tuple[int, ...]]],
]:
    """
    Compile functions required by nuts-rs from a given PyMC model.

    Parameters
    ----------
    model: pymc.Model
        The model to compile
    mode: str
        Pytensor compile mode. One of "NUMBA" or "JAX"
    compute_grad: bool
        Whether to compute gradients using pytensor. Must be True if mode is
        "NUMBA", otherwise False implies Jax will be used to compute gradients
    join_expanded: bool
        Whether to join the expanded variables into a single array. If False,
        the expanded variables will be returned as a list of arrays.
    pymc_initial_point_fn: Callable
        Initial point function created by
        pymc.initial_point.make_initial_point_fn
    var_names:
        Names of variables to store in the trace. Defaults to all variables.

    Returns
    -------
    num_free_vars: int
        Number of free (root) random variables in the model
    num_expanded: int
        Total number of all random variables (root and dependent) in the model
    logp_fn_pt: Callable
        Compiled pytensor log probability function. If compute_grad is True, the
        function will return both the logp and the gradient, otherwise only the
        logp is returned.
    expand_fn_pt: Callable
        Compiled pytensor function that computes the remaining variables for the
        trace
    initial_point_fn: Callable
        Python function that takes a random seed and returns a flat array of
        initial values
    param_data: tuple of lists
        Tuple containing data necessary to unravel a flat array of model
        variables back into a ragged list of arrays. The first list contains the
        names of the variables, the second list contains the slices that
        correspond to the variables in the flat array, and the third list
        contains the shapes of the variables.
    """
    import pytensor
    import pytensor.tensor as pt
    from pymc.pytensorf import compile as compile_pymc

    shapes = _compute_shapes(model)

    # Make logp_dlogp_function
    joined = pt.dvector("__joined_variables")

    value_vars = [model.rvs_to_values[var] for var in model.free_RVs]

    logp = model.logp()

    rewrites = ["canonicalize", "stabilize"]
    if not model.check_bounds:
        rewrites.append("local_remove_check_parameter")

    logp = pytensor.graph.rewrite_graph(logp, include=rewrites)

    if compute_grad:
        grads = pytensor.gradient.grad(logp, value_vars)
        grad = pt.concatenate([grad.ravel() for grad in grads])

    count = 0
    joined_slices = []
    joined_shapes = []
    joined_names = []

    splits = []

    for var in model.free_RVs:
        value_var = model.rvs_to_values[var]
        joined_names.append(value_var.name)
        shape = shapes[value_var.name]
        joined_shapes.append(shape)
        length = prod(shape)
        slice_val = slice(count, count + length)
        joined_slices.append(slice_val)
        count += length

        splits.append(length)

    num_free_vars = count

    initial_point_fn = _rv_dict_to_flat_array_wrapper(
        pymc_initial_point_fn, names=joined_names, shapes=joined_shapes
    )

    joined = pt.TensorType("float64", shape=(num_free_vars,))(
        name="_unconstrained_point"
    )

    use_split = False
    if use_split:
        variables = pt.split(joined, splits, len(splits))
    else:
        variables = [joined[slice_val] for slice_val in zip(joined_slices)]

    replacements = {
        model.rvs_to_values[var]: value.reshape(shape).astype(var.dtype)
        for var, shape, value in zip(
            model.free_RVs,
            joined_shapes,
            variables,
        )
    }

    if compute_grad:
        (logp, grad) = pytensor.clone_replace([logp, grad], replacements)
        with model:
            logp_fn_pt = compile_pymc((joined,), (logp, grad), mode=mode)
    else:
        (logp,) = pytensor.clone_replace([logp], replacements)
        with model:
            logp_fn_pt = compile_pymc((joined,), (logp,), mode=mode)

    # Make function that computes remaining variables for the trace
    remaining_rvs = [
        var for var in model.unobserved_value_vars if var.name not in joined_names
    ]

    if var_names is not None:
        names = set(var_names)
        remaining_rvs = [var for var in remaining_rvs if var.name in names]

    all_names = joined_names + remaining_rvs

    all_names = joined_names.copy()
    all_slices = joined_slices.copy()
    all_shapes = joined_shapes.copy()

    for var in remaining_rvs:
        all_names.append(var.name)
        shape = cast(tuple[int, ...], shapes[var.name])
        all_shapes.append(shape)
        length = prod(shape)
        all_slices.append(slice(count, count + length))
        count += length

    num_expanded = count

    if join_expanded:
        allvars = [pt.concatenate([joined, *[var.ravel() for var in remaining_rvs]])]
    else:
        allvars = [*variables, *remaining_rvs]
    with model:
        expand_fn_pt = compile_pymc(
            (joined,),
            allvars,
            givens=list(replacements.items()),
            mode=mode,
        )

    return (
        num_free_vars,
        num_expanded,
        logp_fn_pt,
        expand_fn_pt,
        initial_point_fn,
        (all_names, all_slices, all_shapes),
    )


def make_extraction_fn(inner, shared_data, shared_vars, record_dtype):
    import numba
    from numba import literal_unroll
    from numba.cpython.unsafe.tuple import alloca_once, tuple_setitem

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

    @intrinsic
    def tuple_setitem_literal(typingctx, tup, idx, val):
        """Return a copy of the tuple with item at *idx* replaced with *val*."""
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
    import numba

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
        except Exception:  # noqa: BLE001
            return 1
        return 0

    return logp_numba, c_sig


def _make_c_expand_func(
    n_dim, n_expanded, expand_fn, user_data, shared_vars, shared_data
):
    import numba

    extract = make_extraction_fn(expand_fn, shared_data, shared_vars, user_data.dtype)

    c_sig = numba.types.int64(
        numba.types.uint64,
        numba.types.uint64,
        numba.types.CPointer(numba.types.double),
        numba.types.CPointer(numba.types.double),
        numba.types.voidptr,
    )

    def expand_numba(dim, expanded, x_, out_, user_data_):
        if dim != n_dim:
            return -1
        if expanded != n_expanded:
            return -1

        try:
            x = numba.carray(x_, (n_dim,))
            out = numba.carray(out_, (n_expanded,))

            (values,) = extract(x, user_data_)
            out[...] = values

        except Exception:  # noqa: BLE001
            return -2
        return 0

    return expand_numba, c_sig
