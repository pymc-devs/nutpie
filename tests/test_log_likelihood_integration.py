"""
Integration tests for log-likelihood calculation functionality.
"""

from importlib.util import find_spec
import pytest
import numpy as np

if find_spec("pymc") is None:
    pytest.skip("Skip pymc tests", allow_module_level=True)

import pymc as pm
import nutpie


@pytest.mark.pymc
def test_log_likelihood_compilation_numba_disabled():
    """Test that log-likelihood is properly disabled for numba backend with warning."""
    np.random.seed(42)
    observed_data = np.random.normal(0, 1, 10)

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=1)
        pm.Normal("y", mu=mu, sigma=1, observed=observed_data)

    with pytest.warns(
        UserWarning,
        match="compute_log_likelihood=True is not supported with numba backend",
    ):
        compiled_model = nutpie.compile_pymc_model(
            model, backend="numba", compute_log_likelihood=True
        )

    # Should be disabled despite being requested
    assert hasattr(compiled_model, "log_likelihood_names")
    assert hasattr(compiled_model, "log_likelihood_shapes")
    assert compiled_model.log_likelihood_names == []
    assert compiled_model.log_likelihood_shapes == []


@pytest.mark.pymc
def test_log_likelihood_compilation_disabled():
    """Test that compilation works with compute_log_likelihood=False."""
    np.random.seed(42)
    observed_data = np.random.normal(0, 1, 10)

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=1)
        pm.Normal("y", mu=mu, sigma=1, observed=observed_data)

    compiled_model = nutpie.compile_pymc_model(
        model, backend="numba", compute_log_likelihood=False
    )
    assert compiled_model.log_likelihood_names == []
    assert compiled_model.log_likelihood_shapes == []


@pytest.mark.pymc
def test_log_likelihood_basic_sampling_jax():
    """Test basic sampling with log-likelihood calculation using JAX backend."""
    np.random.seed(42)
    observed_data = np.random.normal(2.0, 1.0, 20)

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=5)
        sigma = pm.HalfNormal("sigma", sigma=2)
        pm.Normal("y", mu=mu, sigma=sigma, observed=observed_data)

    compiled_model = nutpie.compile_pymc_model(
        model, backend="jax", gradient_backend="jax", compute_log_likelihood=True
    )
    trace = nutpie.sample(compiled_model, draws=10, tune=10, chains=1, cores=1)

    assert hasattr(trace, "log_likelihood"), (
        "log_likelihood group missing from InferenceData"
    )
    assert "log_likelihood_y" in trace.log_likelihood.data_vars, (
        "log_likelihood_y not in log_likelihood group"
    )

    log_lik = trace.log_likelihood["log_likelihood_y"]
    assert log_lik.shape == (1, 10, 20), (
        f"Expected shape (1, 10, 20), got {log_lik.shape}"
    )
    assert np.all(np.isfinite(log_lik.values)), (
        "Log-likelihood values contain non-finite values"
    )
    assert not np.all(log_lik.values == 0), "Log-likelihood values are all zero"
    assert np.all(log_lik.values <= 0), "Log-likelihood values should be non-positive"


@pytest.mark.pymc
def test_log_likelihood_multiple_observed_jax():
    """Test log-likelihood calculation with multiple observed variables using JAX backend."""
    np.random.seed(42)
    n_obs1, n_obs2 = 15, 10
    observed_data1 = np.random.normal(1.0, 0.5, n_obs1)
    observed_data2 = np.random.normal(-1.0, 1.0, n_obs2)

    with pm.Model() as model:
        mu1 = pm.Normal("mu1", mu=0, sigma=2)
        mu2 = pm.Normal("mu2", mu=0, sigma=2)
        pm.Normal("y1", mu=mu1, sigma=0.5, observed=observed_data1)
        pm.Normal("y2", mu=mu2, sigma=1.0, observed=observed_data2)

    compiled_model = nutpie.compile_pymc_model(
        model, backend="jax", gradient_backend="jax", compute_log_likelihood=True
    )
    assert len(compiled_model.log_likelihood_names) == 2
    assert "log_likelihood_y1" in compiled_model.log_likelihood_names
    assert "log_likelihood_y2" in compiled_model.log_likelihood_names

    y1_idx = compiled_model.log_likelihood_names.index("log_likelihood_y1")
    y2_idx = compiled_model.log_likelihood_names.index("log_likelihood_y2")

    assert compiled_model.log_likelihood_shapes[y1_idx] == (n_obs1,)
    assert compiled_model.log_likelihood_shapes[y2_idx] == (n_obs2,)

    trace = nutpie.sample(compiled_model, draws=8, tune=8, chains=1, cores=1)
    assert "log_likelihood_y1" in trace.log_likelihood.data_vars
    assert "log_likelihood_y2" in trace.log_likelihood.data_vars

    log_lik1 = trace.log_likelihood["log_likelihood_y1"]
    log_lik2 = trace.log_likelihood["log_likelihood_y2"]

    assert log_lik1.shape == (1, 8, n_obs1)
    assert log_lik2.shape == (1, 8, n_obs2)

    assert np.all(np.isfinite(log_lik1.values))
    assert np.all(np.isfinite(log_lik2.values))
    assert np.all(log_lik1.values <= 0)
    assert np.all(log_lik2.values <= 0)


@pytest.mark.pymc
def test_log_likelihood_scalar_observed_jax():
    """Test log-likelihood calculation with scalar observed variable using JAX backend."""
    np.random.seed(42)
    observed_value = 3.5

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=2)
        pm.Normal("y", mu=mu, sigma=1, observed=observed_value)

    compiled_model = nutpie.compile_pymc_model(
        model, backend="jax", gradient_backend="jax", compute_log_likelihood=True
    )
    assert "log_likelihood_y" in compiled_model.log_likelihood_names
    y_idx = compiled_model.log_likelihood_names.index("log_likelihood_y")
    assert compiled_model.log_likelihood_shapes[y_idx] == ()
    trace = nutpie.sample(compiled_model, draws=6, tune=6, chains=1, cores=1)

    assert "log_likelihood_y" in trace.log_likelihood.data_vars
    log_lik = trace.log_likelihood["log_likelihood_y"]

    assert log_lik.shape == (1, 6), f"Expected shape (1, 6), got {log_lik.shape}"
    assert np.all(np.isfinite(log_lik.values))
    assert np.all(log_lik.values <= 0)


@pytest.mark.pymc
def test_log_likelihood_backward_compatibility():
    """Test that existing code without compute_log_likelihood still works."""
    np.random.seed(42)
    observed_data = np.random.normal(0, 1, 5)

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=1)
        pm.Normal("y", mu=mu, sigma=1, observed=observed_data)

    compiled_model = nutpie.compile_pymc_model(model, backend="numba")
    assert compiled_model.log_likelihood_names == []
    assert compiled_model.log_likelihood_shapes == []

    trace = nutpie.sample(compiled_model, draws=5, tune=5, chains=1, cores=1)
    if hasattr(trace, "log_likelihood"):
        assert len(trace.log_likelihood.data_vars) == 0

    assert "mu" in trace.posterior.data_vars


@pytest.mark.pymc
def test_log_likelihood_numba_sampling_without_log_lik():
    """Test that numba backend works correctly when log-likelihood is disabled."""
    np.random.seed(42)
    observed_data = np.random.normal(2.0, 1.0, 20)

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=5)
        sigma = pm.HalfNormal("sigma", sigma=2)
        pm.Normal("y", mu=mu, sigma=sigma, observed=observed_data)

    # Test with explicit compute_log_likelihood=True (should be disabled with warning)
    with pytest.warns(
        UserWarning,
        match="compute_log_likelihood=True is not supported with numba backend",
    ):
        compiled_model = nutpie.compile_pymc_model(
            model, backend="numba", compute_log_likelihood=True
        )

    # Verify log-likelihood is disabled
    assert compiled_model.log_likelihood_names == []
    assert compiled_model.log_likelihood_shapes == []

    # Sampling should still work correctly
    trace = nutpie.sample(compiled_model, draws=10, tune=10, chains=1, cores=1)
    assert (
        not hasattr(trace, "log_likelihood") or len(trace.log_likelihood.data_vars) == 0
    )
    assert "mu" in trace.posterior.data_vars
    assert "sigma" in trace.posterior.data_vars
