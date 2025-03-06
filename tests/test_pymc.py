from importlib.util import find_spec
import time
import pytest

if find_spec("pymc") is None:
    pytest.skip("Skip pymc tests", allow_module_level=True)

import numpy as np
import pymc as pm
import pytest
from scipy import stats

import nutpie
import nutpie.compile_pymc

parameterize_backends = pytest.mark.parametrize(
    "backend, gradient_backend",
    [("numba", None), ("jax", "pytensor"), ("jax", "jax")],
)


@pytest.mark.pymc
@parameterize_backends
def test_pymc_model(backend, gradient_backend):
    with pm.Model() as model:
        pm.Normal("a")

    compiled = nutpie.compile_pymc_model(
        model, backend=backend, gradient_backend=gradient_backend
    )
    trace = nutpie.sample(compiled, chains=1)
    trace.posterior.a  # noqa: B018


@pytest.mark.pymc
@parameterize_backends
def test_pymc_model_float32(backend, gradient_backend):
    import pytensor

    with pytensor.config.change_flags(floatX="float32"):
        with pm.Model() as model:
            pm.Normal("a")

        compiled = nutpie.compile_pymc_model(
            model, backend=backend, gradient_backend=gradient_backend
        )
        trace = nutpie.sample(compiled, chains=1)
        trace.posterior.a  # noqa: B018


@pytest.mark.pymc
@parameterize_backends
def test_pymc_model_no_prior(backend, gradient_backend):
    with pm.Model() as model:
        a = pm.Flat("a")
        pm.Normal("b", mu=a, observed=0.0)

    compiled = nutpie.compile_pymc_model(
        model, backend=backend, gradient_backend=gradient_backend
    )
    trace = nutpie.sample(compiled, chains=1)
    trace.posterior.a  # noqa: B018


@pytest.mark.pymc
@parameterize_backends
def test_blocking(backend, gradient_backend):
    with pm.Model() as model:
        pm.Normal("a")

    compiled = nutpie.compile_pymc_model(
        model, backend=backend, gradient_backend=gradient_backend
    )
    sampler = nutpie.sample(compiled, chains=1, blocking=False)
    trace = sampler.wait()
    trace.posterior.a  # noqa: B018


@pytest.mark.pymc
@parameterize_backends
@pytest.mark.timeout(20)
def test_wait_timeout(backend, gradient_backend):
    with pm.Model() as model:
        pm.Normal("a", shape=100_000)
    compiled = nutpie.compile_pymc_model(
        model, backend=backend, gradient_backend=gradient_backend
    )
    start = time.time()
    sampler = nutpie.sample(compiled, chains=1, blocking=False)
    with pytest.raises(TimeoutError):
        sampler.wait(timeout=0.1)
    sampler.cancel()
    assert start - time.time() < 5


@pytest.mark.pymc
@parameterize_backends
@pytest.mark.timeout(20)
def test_pause(backend, gradient_backend):
    with pm.Model() as model:
        pm.Normal("a", shape=100_000)
    compiled = nutpie.compile_pymc_model(
        model, backend=backend, gradient_backend=gradient_backend
    )
    start = time.time()
    sampler = nutpie.sample(compiled, chains=1, blocking=False)
    sampler.pause()
    sampler.resume()
    sampler.cancel()
    assert start - time.time() < 5


@pytest.mark.pymc
@parameterize_backends
def test_pymc_model_with_coordinate(backend, gradient_backend):
    with pm.Model() as model:
        model.add_coord("foo", length=5)
        pm.Normal("a", dims="foo")

    compiled = nutpie.compile_pymc_model(
        model, backend=backend, gradient_backend=gradient_backend
    )
    trace = nutpie.sample(compiled, chains=1)
    trace.posterior.a  # noqa: B018


@pytest.mark.pymc
@parameterize_backends
def test_pymc_model_store_extra(backend, gradient_backend):
    with pm.Model() as model:
        model.add_coord("foo", length=5)
        pm.Normal("a", dims="foo")

    compiled = nutpie.compile_pymc_model(
        model, backend=backend, gradient_backend=gradient_backend
    )
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


@pytest.mark.pymc
@parameterize_backends
def test_trafo(backend, gradient_backend):
    with pm.Model() as model:
        pm.Uniform("a")

    compiled = nutpie.compile_pymc_model(
        model, backend=backend, gradient_backend=gradient_backend
    )
    trace = nutpie.sample(compiled, chains=1)
    trace.posterior.a  # noqa: B018


@pytest.mark.pymc
@parameterize_backends
def test_det(backend, gradient_backend):
    with pm.Model() as model:
        a = pm.Uniform("a", shape=2)
        pm.Deterministic("b", 2 * a)

    compiled = nutpie.compile_pymc_model(
        model, backend=backend, gradient_backend=gradient_backend
    )
    trace = nutpie.sample(compiled, chains=1)
    assert trace.posterior.a.shape[-1] == 2
    assert trace.posterior.b.shape[-1] == 2


@pytest.mark.pymc
@parameterize_backends
def test_non_identifier_names(backend, gradient_backend):
    with pm.Model() as model:
        a = pm.Uniform("a/b", shape=2)
        with pm.Model("foo"):
            c = pm.Data("c", np.array([2.0, 3.0]))
            pm.Deterministic("b", c * a)

    compiled = nutpie.compile_pymc_model(
        model, backend=backend, gradient_backend=gradient_backend
    )
    trace = nutpie.sample(compiled, chains=1)
    assert trace.posterior["a/b"].shape[-1] == 2
    assert trace.posterior["foo::b"].shape[-1] == 2


@pytest.mark.pymc
@parameterize_backends
def test_pymc_model_shared(backend, gradient_backend):
    with pm.Model() as model:
        mu = pm.Data("mu", -0.1)
        sigma = pm.Data("sigma", np.ones(3))
        pm.Normal("a", mu=mu, sigma=sigma, shape=3)

    compiled = nutpie.compile_pymc_model(
        model,
        backend=backend,
        gradient_backend=gradient_backend,
        freeze_model=False,
    )
    trace = nutpie.sample(compiled, chains=1, seed=1)
    np.testing.assert_allclose(trace.posterior.a.mean().values, -0.1, atol=0.05)

    compiled2 = compiled.with_data(mu=10.0, sigma=3 * np.ones(3))
    trace2 = nutpie.sample(compiled2, chains=1, seed=1)
    np.testing.assert_allclose(trace2.posterior.a.mean().values, 10.0, atol=0.5)

    compiled3 = compiled.with_data(mu=0.5, sigma=3 * np.ones(4))
    with pytest.raises(RuntimeError):
        nutpie.sample(compiled3, chains=1)


@pytest.mark.pymc
@parameterize_backends
def test_pymc_var_names(backend, gradient_backend):
    with pm.Model() as model:
        mu = pm.Data("mu", -0.1)
        sigma = pm.Data("sigma", np.ones(3))
        a = pm.Normal("a", mu=mu, sigma=sigma, shape=3)

        b = pm.Deterministic("b", mu * a)
        pm.Deterministic("c", mu * b)

    compiled = nutpie.compile_pymc_model(
        model,
        backend=backend,
        gradient_backend=gradient_backend,
        var_names=None,
    )
    trace = nutpie.sample(compiled, chains=1, seed=1)

    # Check that variables are stored
    assert hasattr(trace.posterior, "b")
    assert hasattr(trace.posterior, "c")

    compiled = nutpie.compile_pymc_model(
        model,
        backend=backend,
        gradient_backend=gradient_backend,
        var_names=[],
    )
    trace = nutpie.sample(compiled, chains=1, seed=1)

    # Check that variables are stored
    assert not hasattr(trace.posterior, "b")
    assert not hasattr(trace.posterior, "c")

    compiled = nutpie.compile_pymc_model(
        model,
        backend=backend,
        gradient_backend=gradient_backend,
        var_names=["b"],
    )
    trace = nutpie.sample(compiled, chains=1, seed=1)

    # Check that variables are stored
    assert hasattr(trace.posterior, "b")
    assert not hasattr(trace.posterior, "c")


@pytest.mark.pymc
@pytest.mark.flow
def test_normalizing_flow():
    with pm.Model() as model:
        pm.HalfNormal("x", shape=2)

    compiled = nutpie.compile_pymc_model(
        model, backend="jax", gradient_backend="jax"
    ).with_transform_adapt(
        num_diag_windows=6,
        verbose=True,
    )
    trace = nutpie.sample(
        compiled,
        chains=1,
        transform_adapt=True,
        window_switch_freq=150,
        tune=600,
        seed=1,
    )
    draws = trace.posterior.x.isel(x_dim_0=0, chain=0)
    kstest = stats.ks_1samp(draws, stats.halfnorm.cdf)
    assert kstest.pvalue > 0.01

    draws = trace.posterior.x.isel(x_dim_0=1, chain=0)
    kstest = stats.ks_1samp(draws, stats.halfnorm.cdf)
    assert kstest.pvalue > 0.01


@pytest.mark.pymc
@pytest.mark.parametrize(
    ("backend", "gradient_backend"),
    [
        ("numba", None),
        pytest.param(
            "jax",
            "pytensor",
            marks=pytest.mark.xfail(
                reason="https://github.com/pymc-devs/pytensor/issues/853"
            ),
        ),
        pytest.param(
            "jax",
            "jax",
            marks=pytest.mark.xfail(
                reason="https://github.com/pymc-devs/pytensor/issues/853"
            ),
        ),
    ],
)
def test_missing(backend, gradient_backend):
    with pm.Model(coords={"obs": range(4)}) as model:
        mu = pm.Normal("mu")
        y = pm.Normal("y", mu, observed=[0, -1, 1, np.nan], dims="obs")
        pm.Deterministic("y2", 2 * y, dims="obs")

    compiled = nutpie.compile_pymc_model(
        model, backend=backend, gradient_backend=gradient_backend
    )
    tr = nutpie.sample(compiled, chains=1, seed=1)
    print(tr.posterior)
    assert hasattr(tr.posterior, "y_unobserved")
