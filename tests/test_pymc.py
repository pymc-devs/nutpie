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
    trace.posterior.a  # noqa: B018


def test_blocking():
    with pm.Model() as model:
        pm.Normal("a")

    compiled = nutpie.compile_pymc_model(model)
    sampler = nutpie.sample(compiled, chains=1, blocking=False)
    trace = sampler.wait()
    trace.posterior.a  # noqa: B018


@pytest.mark.timeout(2)
def test_wait_timeout():
    with pm.Model() as model:
        pm.Normal("a", shape=100_000)
    compiled = nutpie.compile_pymc_model(model)
    sampler = nutpie.sample(compiled, chains=1, blocking=False)
    with pytest.raises(TimeoutError):
        sampler.wait(timeout=0.1)
    sampler.cancel()


@pytest.mark.timeout(2)
def test_pause():
    with pm.Model() as model:
        pm.Normal("a", shape=100_000)
    compiled = nutpie.compile_pymc_model(model)
    sampler = nutpie.sample(compiled, chains=1, blocking=False)
    sampler.pause()
    sampler.resume()
    sampler.cancel()


def test_pymc_model_with_coordinate():
    with pm.Model() as model:
        model.add_coord("foo", length=5)
        pm.Normal("a", dims="foo")

    compiled = nutpie.compile_pymc_model(model)
    trace = nutpie.sample(compiled, chains=1)
    trace.posterior.a  # noqa: B018


def test_pymc_model_store_extra():
    with pm.Model() as model:
        model.add_coord("foo", length=5)
        pm.Normal("a", dims="foo")

    compiled = nutpie.compile_pymc_model(model)
    trace = nutpie.sample(
        compiled,
        chains=1,
        store_mass_matrix=True,
        store_divergences=True,
        store_unconstrained=True,
        store_gradient=True,
    )
    trace.posterior.a  # noqa: B018
    _ = trace.sample_stats.unconstrained_draw
    _ = trace.sample_stats.gradient
    _ = trace.sample_stats.divergence_start
    _ = trace.sample_stats.mass_matrix_inv


def test_trafo():
    with pm.Model() as model:
        pm.Uniform("a")

    compiled = nutpie.compile_pymc_model(model)
    trace = nutpie.sample(compiled, chains=1)
    trace.posterior.a  # noqa: B018


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
    with pytest.raises(RuntimeError):
        nutpie.sample(compiled3, chains=1)


def test_missing():
    with pm.Model(coords={"obs": range(4)}) as model:
        mu = pm.Normal("mu")
        y = pm.Normal("y", mu, observed=[0, -1, 1, np.nan], dims="obs")
        pm.Deterministic("y2", 2 * y, dims="obs")

    compiled = nutpie.compile_pymc_model(model)
    tr = nutpie.sample(compiled, chains=1, seed=1)
    print(tr.posterior)
    assert hasattr(tr.posterior, "y_unobserved")
