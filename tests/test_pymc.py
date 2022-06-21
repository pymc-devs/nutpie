import nutpie
import pytest


def test_pymc_model():
    pm = pytest.importorskip("pymc")

    import nutpie.compile_pymc

    with pm.Model() as model:
        pm.Normal("a")

    compiled = nutpie.compile_pymc_model(model)
    trace = nutpie.sample(compiled, chains=1)
    trace.posterior.a
