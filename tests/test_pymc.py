import pymc as pm
import nutpie
import pytest


@pytest.mark.xfail
def test_pymc_model():
    with pm.Model() as model:
        pm.Normal("a")

    compiled = nutpie.compile_pymc_model(model)
    trace = nutpie.sample(compiled, chains=1)
    trace.posterior.a
