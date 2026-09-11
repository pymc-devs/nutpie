"""Experimental synchronous PyMC sampling for the pinned Xeus runtime."""

import json

import numpy as np

from nutpie import _lib
from nutpie.result import _dict_to_arviz


def sample(
    compiled_model,
    *,
    draws=500,
    tune=500,
    chains=2,
    seed=42,
    target_accept=0.9,
    max_depth=10,
    store_unconstrained=False,
):
    """Return Nutpie's usual ArviZ/xarray result from a bounded WASM run.

    This prototype supports diagonal adaptation, sequential chains and retained
    draws only. It has no progress, cancellation or warmup storage. Sample stats
    contain diverging, n_steps and step_size. Unsupported native options raise
    TypeError rather than being silently ignored.
    """
    settings = {
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "seed": seed,
        "target_accept": target_accept,
        "max_depth": max_depth,
    }
    raw, stats, evaluations, leapfrogs = _lib.sample_pymc_stats(
        compiled_model._make_model(None), **settings
    )
    # Only the storage decoding differs; native and WASM share ArviZ conversion.
    posterior = {
        name: np.asarray([[row[name] for row in chain] for chain in raw]).reshape(
            chains, draws, *shape
        )
        for name, shape in compiled_model.shapes.items()
    }
    stats_array = np.asarray(stats)
    sample_stats = {
        "diverging": stats_array[:, :, 0].astype(bool),
        "n_steps": stats_array[:, :, 1].astype(np.uint64),
        "step_size": stats_array[:, :, 2],
    }
    attrs = {
        "inference_library": "nutpie",
        "inference_library_version": _lib.__version__,
        "inference_library_settings": json.dumps(settings),
        "logp_evaluations": evaluations,
        "leapfrog_steps": leapfrogs,
    }
    return _dict_to_arviz(
        posterior,
        sample_stats,
        {},
        {},
        compiled_model.dims or {},
        compiled_model.reparameterized_names or (),
        store_unconstrained,
        coords=compiled_model.coords,
        save_warmup=False,
        attrs={"sample_stats": attrs},
    )
