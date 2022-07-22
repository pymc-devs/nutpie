from . import nutpie as lib

try:
    from .compile_pymc import compile_pymc_model
except ImportError:
    def compile_pymc_model(*args, **kwargs):
        raise ValueError("Missing dependencies for pymc. `import nutpie.compile_pymc` to see error.")

try:
    from .compile_stan import compile_stan_model
except ImportError:
    def compile_stan_model(*args, **kwargs):
        raise ImportError("Missing dependencies for stan. `import nutpie.compile_stan` to see error.")

from .sample import sample
