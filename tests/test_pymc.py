import nutpie
import pytest
import numpy as np


def test_pymc_model():
    pm = pytest.importorskip("pymc")

    import nutpie.compile_pymc

    with pm.Model() as model:
        pm.Normal("a")

    compiled = nutpie.compile_pymc_model(model)
    trace = nutpie.sample(compiled, chains=1)
    trace.posterior.a


def test_pymc_model_shared():
    pm = pytest.importorskip("pymc")

    import nutpie.compile_pymc

    with pm.Model() as model:
        mu = pm.MutableData("mu", 0.1)
        sigma = pm.MutableData("sigma", np.ones(3))
        pm.Normal("a", mu=mu, sigma=sigma, shape=3)

    compiled = nutpie.compile_pymc_model(model)
    trace = nutpie.sample(compiled, chains=1)

    compiled2 = compiled.with_data(mu=0.5, sigma=3 * np.ones(3))
    trace2 = nutpie.sample(compiled2, chains=1)

    compiled3 = compiled.with_data(mu=0.5, sigma=3 * np.ones(4))
    with pytest.raises(ValueError):
        trace3 = nutpie.sample(compiled3, chains=1)

