from collections import Callable
from typing import Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
import fastprogress
import arviz
from numpy.typing import DTypeLike, NDArray
import xarray as xr

from . import lib


@dataclass(frozen=True)
class CompiledModel:
    n_dim: int
    dims: Dict[str, Tuple[str, ...]]
    coords: Dict[str, xr.IndexVariable]
    shape_info: List[Tuple[str, slice, Tuple[int, ...]]]
    logp_func_maker: lib.PtrLogpFuncMaker
    expand_draw_fn: Callable[[NDArray], NDArray]


def sample(
    compiled_model: CompiledModel,
    *,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    seed: int = 42,
    num_try_init=100,
    save_warmup: bool = True,
    store_divergences: bool = False,
    progress_bar=True,
    **kwargs,
):
    """Sample the posterior distribution for a compiled model.

    Parameters
    ----------
    draws: int
        The number of draws after tuning in each chain.
    tune: int
        The number of tuning (warmup) draws in each chain.
    chains: int
        The number of chains to sample.
    seed: int
        Seed for the randomness in sampling.
    num_try_init: int
        The number if initial positions for each chain to try.
        Fail if we can't find a valid initializion point after
        this many tries.
    save_warmup: bool
        Wether to save the tuning (warmup) statistics and
        posterior draws in the output dataset.
    store_divergences: bool
        If true, store the exact locations where diverging
        transitions happend in the sampler stats.
    progress_bar: bool
        If true, display the progress bar (default)
    **kwargs
        Pass additional arguments to nutpie.lib.PySamplerArgs
    """
    settings = lib.PySamplerArgs()
    settings.num_tune = tune

    for name, val in kwargs.items():
        setattr(settings, name, val)

    init_mean = np.zeros(compiled_model.n_dim)
    sampler = lib.PyParallelSampler(
        compiled_model.logp_func_maker,
        init_mean,
        settings,
        n_chains=chains,
        n_draws=draws,
        seed=seed,
        n_try_init=num_try_init,
    )

    expand_draw = compiled_model.expand_draw_fn

    def do_sample():
        n_expanded = len(expand_draw(init_mean))
        draws_data = np.full((chains, draws + tune, n_expanded), np.nan)
        infos = []
        try:
            bar = fastprogress.progress_bar(
                sampler,
                total=chains * (draws + tune),
                display=progress_bar,
            )
            num_divs = 0
            chains_tuning = chains
            for draw, info in bar:
                infos.append(info)
                draws_data[info.chain, info.draw, :] = compiled_model.expand_draw_fn(
                    draw
                )
                if info.draw == tune - 1:
                    chains_tuning -= 1
                if info.is_diverging and info.draw > tune:
                    num_divs += 1
                bar.comment = (
                    f" Chains in warmup: {chains_tuning}, Divergences: {num_divs}"
                )
        except KeyboardInterrupt:
            pass
        return draws_data, infos

    try:
        draws_data, infos = do_sample()
    finally:
        try:
            sampler.finalize()
        except Exception as e:
            print(e)

    trace_dict = {}
    trace_dict_tune = {}
    for name, slice_, shape in zip(*compiled_model.shape_info):
        trace_dict_tune[name] = draws_data[:, :tune, slice_].reshape(
            (chains, tune) + tuple(shape)
        )
        trace_dict[name] = draws_data[:, tune:, slice_].reshape(
            (chains, draws) + tuple(shape)
        )

    stat_dtypes: Dict[str, Tuple[Tuple[str, ...], DTypeLike]] = {
        "index_in_trajectory": ((), np.int64),
        "mean_tree_accept": ((), np.float64),
        "depth": ((), np.int64),
        "maxdepth_reached": ((), bool),
        "logp": ((), np.float64),
        "energy": ((), np.float64),
        "diverging": ((), bool),
        "step_size": ((), np.float64),
        "step_size_bar": ((), np.float64),
        "mean_tree_accept": ((), np.float64),
        "n_steps": ((), np.int64),
    }

    # Sampler statistics that do not have extra dimensions
    simple_stats = list(stat_dtypes.keys())

    if settings.store_mass_matrix:
        stat_dtypes["mass_matrix_inv"] = (("unconstrained_parameter",), np.float64)
    if settings.store_gradient:
        stat_dtypes["gradient"] = (("unconstrained_parameter",), np.float64)
    if store_divergences:
        stat_dtypes["divergence_start"] = (("unconstrained_parameter",), np.float64)
        stat_dtypes["divergence_end"] = (("unconstrained_parameter",), np.float64)

    dim_to_length = {
        "unconstrained_parameter": compiled_model.n_dim,
    }

    stats = {}
    stats_tune = {}
    for name, (dims, dtype) in stat_dtypes.items():
        shapes = tuple(dim_to_length[name] for name in dims)
        if dtype == np.float64:
            value = np.nan
        else:
            value = 0
        stats[name] = np.full((chains, draws) + shapes, value, dtype=dtype)
        stats_tune[name] = np.full((chains, tune) + shapes, value, dtype=dtype)

    for info in infos:
        info_dict = info.as_dict()
        if info.draw < tune:
            out = stats_tune
            draw = info.draw
        else:
            out = stats
            draw = info.draw - tune
        for name in stat_dtypes:
            if name in info_dict:
                out[name][info.chain, draw] = info_dict[name]

    trace = arviz.from_dict(
        posterior=trace_dict,
        warmup_posterior=trace_dict_tune,
        save_warmup=save_warmup,
        coords=compiled_model.coords,
        dims={name: list(dim) for name, dim in compiled_model.dims.items()},
        sample_stats={name: stats[name] for name in simple_stats},
        warmup_sample_stats={name: stats_tune[name] for name in simple_stats},
    )

    for name in stats:
        if name in simple_stats:
            continue
        trace.sample_stats[name] = (
            ("chain", "draw") + stat_dtypes[name][0],
            stats[name],
        )
        if save_warmup:
            trace.warmup_sample_stats[name] = (
                ("chain", "draw") + stat_dtypes[name][0],
                stats_tune[name],
            )

    return trace
