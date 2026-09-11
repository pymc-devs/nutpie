import sys

if sys.platform == "emscripten":
    from nutpie import _lib
    from nutpie._lib import sample_pymc as sample_raw
    from nutpie.compile_pymc import compile_pymc_model
    __version__ = _lib.__version__
    __all__ = ["compile_pymc_model", "sample_raw"]
else:
    from nutpie import _lib
    from nutpie._lib import store as zarr_store
    from nutpie.compile_pymc import compile_pymc_model
    from nutpie.compile_stan import compile_stan_model, prune_stan_cache
    from nutpie.sample import sample

    ChainProgress = _lib.PyChainProgress

    __version__: str = _lib.__version__
    __all__ = [
        "ChainProgress",
        "__version__",
        "compile_pymc_model",
        "compile_stan_model",
        "prune_stan_cache",
        "sample",
        "zarr_store",
    ]
