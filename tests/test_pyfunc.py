"""Sampling models written as plain python functions (`from_pyfunc`).

These need only the base dependencies. PyMC, Stan and JAX are not available
everywhere (e.g. on wasm), so this is what runs there.
"""

import numpy as np
import pytest

import nutpie
from nutpie.compiled_pyfunc import from_pyfunc


def _normal_model(n=3, **kwargs):
    def make_logp_fn():
        def logp(x, **shared):
            return -0.5 * float(x @ x), -np.ascontiguousarray(x, dtype="float64")

        return logp

    def make_expand_fn(seed1, seed2, chain):
        def expand(x, **shared):
            return {"x": np.asarray(x, dtype="float64", order="C")}

        return expand

    return from_pyfunc(
        n,
        make_logp_fn,
        make_expand_fn,
        [np.dtype("float64")],
        [(n,)],
        ["x"],
        **kwargs,
    )


def test_import():
    assert isinstance(nutpie.__version__, str)


def test_sample_pyfunc():
    compiled = _normal_model(
        coords={"idx": ["a", "b", "c"]},
        dims={"x": ("idx",)},
    )
    trace = nutpie.sample(
        compiled, chains=2, tune=300, draws=300, seed=1, progress_bar=False
    )

    x = trace.posterior["x"]
    assert x.dims == ("chain", "draw", "idx")
    assert x.shape == (2, 300, 3)
    assert list(x["idx"].values) == ["a", "b", "c"]
    assert np.all(np.isfinite(x.values))
    assert np.abs(x.mean(("chain", "draw")).values).max() < 0.5
    assert np.abs(x.std(("chain", "draw")).values - 1).max() < 0.5
    assert int(trace.sample_stats["diverging"].sum()) == 0


def test_sample_is_deterministic():
    kwargs = {"chains": 2, "tune": 50, "draws": 50, "seed": 3, "progress_bar": False}
    a = nutpie.sample(_normal_model(), **kwargs).posterior["x"].values
    b = nutpie.sample(_normal_model(), **kwargs).posterior["x"].values
    np.testing.assert_array_equal(a, b)


@pytest.mark.skipif(hasattr(nutpie._lib, "store"), reason="This build has zarr support")
def test_zarr_unavailable():
    with pytest.raises(AttributeError, match="not available"):
        _ = nutpie.zarr_store.LocalStore
    with pytest.raises(NotImplementedError, match="not available"):
        nutpie.sample(_normal_model(), draws=10, tune=10, chains=1, zarr_store=object())
