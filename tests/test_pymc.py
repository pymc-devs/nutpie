import numpy as np
import pymc as pm
import pytest

import nutpie
import nutpie.compile_pymc


def test_pymc_model():
    with pm.Model() as model:
        pm.Normal("a")

    compiled = nutpie.compile_pymc_model(model)
    trace = nutpie.sample(compiled, chains=1)
    trace.posterior.a


def test_pymc_model():
    with pm.Model() as model:
        model.add_coord("foo", length=5)
        pm.Normal("a", dims="foo")

    compiled = nutpie.compile_pymc_model(model)
    trace = nutpie.sample(compiled, chains=1)
    trace.posterior.a


def test_trafo():
    with pm.Model() as model:
        pm.Uniform("a")

    compiled = nutpie.compile_pymc_model(model)
    trace = nutpie.sample(compiled, chains=1)
    trace.posterior.a


def test_det():
    with pm.Model() as model:
        a = pm.Uniform("a", shape=2)
        pm.Deterministic("b", 2 * a)

    compiled = nutpie.compile_pymc_model(model)
    trace = nutpie.sample(compiled, chains=1)
    assert trace.posterior.a.shape[-1] == 2
    assert trace.posterior.b.shape[-1] == 2


def test_pymc_model_shared():
    with pm.Model() as model:
        mu = pm.MutableData("mu", 0.1)
        sigma = pm.MutableData("sigma", np.ones(3))
        pm.Normal("a", mu=mu, sigma=sigma, shape=3)

    compiled = nutpie.compile_pymc_model(model)
    trace = nutpie.sample(compiled, chains=1, seed=1)
    np.testing.assert_allclose(trace.posterior.a.mean().values, 0.1, atol=0.05)

    compiled2 = compiled.with_data(mu=10.0, sigma=3 * np.ones(3))
    trace2 = nutpie.sample(compiled2, chains=1, seed=1)
    np.testing.assert_allclose(trace2.posterior.a.mean().values, 10.0, atol=0.5)

    compiled3 = compiled.with_data(mu=0.5, sigma=3 * np.ones(4))
    with pytest.raises(ValueError):
        nutpie.sample(compiled3, chains=1)
