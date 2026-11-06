import pytest


@pytest.fixture
def x64():
    """Enable JAX float64 for one test only, without leaking into the rest
    of the session. The context is thread-local: code that runs JAX on the
    sampler's worker threads does not see it."""
    import jax

    with jax.enable_x64(True):
        yield
