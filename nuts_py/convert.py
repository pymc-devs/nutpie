import pymc as pm
import aesara
import aesara.tensor as at
import numpy as np
from math import prod
import numba
import fastprogress
import arviz

from . import lib


def compute_shapes(model):
    point = pm.model.make_initial_point_fn(model=model, return_transformed=True)(0)

    value_vars = model.value_vars.copy()
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


def make_functions(model):
    shapes = compute_shapes(model)

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
    fgraph = func.maker.fgraph
    func = aesara.link.numba.dispatch.numba_funcify(fgraph)
    # logp_func = numba.njit(func)
    logp_func = func

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
    fgraph = func.maker.fgraph
    func = aesara.link.numba.dispatch.numba_funcify(fgraph)
    expanding_function = numba.njit(func, fastmath=True, error_model="numpy")

    return (
        num_free_vars,
        logp_func,
        expanding_function,
        (all_names, all_slices, all_shapes),
    )


def make_c_logp_func(N, logp_func):
    c_sig = numba.types.int64(
        numba.types.uint64,
        numba.types.CPointer(numba.types.double),
        numba.types.CPointer(numba.types.double),
        numba.types.CPointer(numba.types.double),
        numba.types.voidptr,
    )

    def rerun_inner(x):
        try:
            logp_func(x)
        except Exception as e:
            print(e)

    @numba.njit()
    def rerun(x):
        with numba.objmode():
            rerun_inner(x)

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
            return 0
        except Exception:
            return 1

    # logp_numba.compile()
    return logp_numba, c_sig


def sample(
    model,
    *,
    N,
    logp_numba,
    expanding_function,
    shape_info,
    n_tune,
    n_draws,
    n_chains,
    seed=42,
    max_treedepth=10,
    target_accept=0.8,
    **kwargs,
):
    def make_user_data():
        return 0

    settings = lib.PySamplerArgs()
    settings.num_tune = n_tune
    settings.maxdepth = max_treedepth
    settings.target_accept = target_accept

    for name, val in kwargs.items():
        setattr(settings, name, val)

    x = np.random.randn(N)
    sampler = lib.PyParallelSampler(
        logp_numba.address,
        make_user_data,
        N,
        x,
        settings,
        n_chains=n_chains,
        n_draws=n_draws,
        seed=seed,
        n_try_init=10,
    )

    try:
        n_expanded = len(expanding_function(x)[0])
        draws = np.full((n_chains, n_draws + n_tune, n_expanded), np.nan)
        infos = []
        for draw, info in fastprogress.progress_bar(
            sampler, total=n_chains * (n_draws + n_tune)
        ):
            infos.append(info)
            draws[info.chain, info.draw, :] = expanding_function(draw)[0]
    finally:
        sampler.finalize()

    trace_dict = {}
    trace_dict_tune = {}
    for name, slice_, shape in zip(*shape_info):
        trace_dict_tune[name] = draws[:, :n_tune, slice_].reshape(
            (n_chains, n_tune) + tuple(shape)
        )
        trace_dict[name] = draws[:, n_tune:, slice_].reshape(
            (n_chains, n_draws) + tuple(shape)
        )

    stat_dtypes = {
        "index_in_trajectory": np.int64,
        "mean_tree_accept": np.float64,
        "depth": np.int64,
        "maxdepth_reached": bool,
        "logp": np.float64,
        "energy": np.float64,
        "diverging": bool,
        "step_size": np.float64,
        "step_size_bar": np.float64,
        "mean_tree_accept": np.float64,
    }

    # This is actually relatively slow, we should be able to speed this up
    stats = {}
    stats_tune = {}
    for name, dtype in stat_dtypes.items():
        stats[name] = np.zeros((n_chains, n_draws), dtype=dtype)
        stats_tune[name] = np.zeros((n_chains, n_tune), dtype=dtype)

    for info in infos:
        info_dict = info.as_dict()
        if info.draw < n_tune:
            out = stats_tune
            draw = info.draw
        else:
            out = stats
            draw = info.draw - n_tune
        for name in stat_dtypes:
            out[name][info.chain, draw] = info_dict[name]

    return arviz.from_dict(
        posterior=trace_dict,
        warmup_posterior=trace_dict_tune,
        save_warmup=True,
        coords=model.coords,
        dims={name: list(dims) for name, dims in model._RV_dims.items()},
        sample_stats=stats,
        warmup_sample_stats=stats_tune,
    )
