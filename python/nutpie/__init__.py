from nutpie import _lib
from nutpie.sample import sample

try:
    from .compile_pymc import compile_pymc_model
except ImportError:

    def compile_pymc_model(*args, **kwargs):
        raise ValueError("Missing dependencies for pymc models. Install pymc.")


try:
    from .compile_stan import compile_stan_model
except ImportError:

    def compile_stan_model(*args, **kwargs):
        raise ImportError("Missing dependencies for stan models. Install bridgestan.")


__version__ = _lib.__version__


__all__ = ["sample", "compile_pymc_model", "compile_stan_model"]
