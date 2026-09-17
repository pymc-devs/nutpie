from nutpie import _lib
from nutpie.compile_pymc import compile_pymc_model
from nutpie.compile_stan import compile_stan_model, prune_stan_cache
from nutpie.sample import sample

ChainProgress = _lib.PyChainProgress

# Builds without the `zarr` feature (e.g. wasm) have no store module.
zarr_store = getattr(_lib, "store", None)

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
