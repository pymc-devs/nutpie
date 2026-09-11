"""Focused native regression checks for the shared compiler/ABI changes."""

import numpy as np
import pandas as pd
import pymc as pm

import nutpie

with pm.Model(coords={"component": pd.Index(["a", "b", "c"], dtype="str")}) as model:
    shared = pm.Data("offset", np.array([0, 1, 0], dtype="int32"))
    scale = pm.HalfNormal("scale")
    weights = pm.Dirichlet("weights", np.ones(3), dims="component")
    pm.Deterministic("scaled", scale * weights, dims="component")
    pm.Normal("obs", weights + shared, 1, observed=[1.0, 2.0, 1.0])
compiled = nutpie.compile_pymc_model(model)
updated = compiled.with_data(offset=np.array([1, 0, 1], dtype="int32"))
a = nutpie.sample(
    compiled, chains=2, cores=1, tune=150, draws=100, seed=42, progress_bar=False
)
b = nutpie.sample(
    updated, chains=2, cores=1, tune=150, draws=100, seed=42, progress_bar=False
)
np.testing.assert_allclose(a.posterior.weights.sum("component"), 1, atol=1e-12)
np.testing.assert_allclose(
    a.posterior.scaled, a.posterior.scale * a.posterior.weights, atol=1e-12
)
assert np.isfinite(b.posterior.scaled).all()
assert set(a.posterior.coords["component"].values) == {"a", "b", "c"}
assert not np.array_equal(a.posterior.weights, b.posterior.weights)
assert a.sample_stats.diverging.shape == (2, 100)
print("NATIVE_REGRESSION_PASSED")
