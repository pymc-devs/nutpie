import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pymc")
numba_cache = pytest.importorskip("pytensor.link.numba.cache")
if not all(
    hasattr(numba_cache, name)
    for name in (
        "CACHED_SRC_FUNCTIONS",
        "compile_numba_function_src",
        "hash_from_pickle_dump",
    )
):
    pytest.skip("Requires PyTensor graph-keyed Numba caching", allow_module_level=True)


def run_worker(tmp_path, mode, variant=0, dim=2, fastmath="default"):
    env = os.environ.copy()
    env["PYTENSOR_FLAGS"] = f"base_compiledir={tmp_path},numba__cache=True"
    env["NUMBA_CACHE_DIR"] = str(tmp_path / "wrappers")
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["NUMBA_NUM_THREADS"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            mode,
            str(variant),
            str(dim),
            fastmath,
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.splitlines()[-1])


@pytest.mark.pymc
@pytest.mark.parametrize("mode", ["constant", "large_shared"])
def test_numba_wrapper_disk_cache(tmp_path, mode):
    first = run_worker(tmp_path, mode)
    second = run_worker(tmp_path, mode, variant=int(mode != "constant"))
    assert first["hits"] == [0, 0]
    assert second["hits"] == [1, 1]


@pytest.mark.pymc
def test_numba_wrapper_cache_invalidation(tmp_path):
    assert run_worker(tmp_path, "constant", dim=2)["hits"] == [0, 0]
    assert run_worker(tmp_path, "constant", dim=3)["hits"] == [0, 0]
    assert run_worker(tmp_path, "constant", dim=3, variant=1)["hits"] == [0, 0]
    assert run_worker(tmp_path, "constant", dim=3, variant=1, fastmath="true")[
        "hits"
    ] == [0, 0]
    assert run_worker(tmp_path, "constant", dim=3, variant=1, fastmath="true")[
        "hits"
    ] == [1, 1]


@pytest.mark.pymc
@pytest.mark.parametrize("kwargs,graph_cache", [({"cache": False}, True), ({}, False)])
def test_numba_wrapper_cache_opt_out(kwargs, graph_cache):
    import numba
    from numba.core.caching import NullCache
    from pytensor import config

    from nutpie.compile_pymc import _compile_numba_cfunc

    def double(x):
        return x * 2

    numba_cache.CACHED_SRC_FUNCTIONS[double] = "opt-out-test"
    with config.change_flags(numba__cache=graph_cache):
        callback = _compile_numba_cfunc(
            double,
            numba.types.float64(numba.types.float64),
            SimpleNamespace(py_func=double),
            (),
            **kwargs,
        )
    assert isinstance(callback._cache, NullCache)
    assert callback.ctypes(3.0) == 6.0


def cache_worker():
    import pymc as pm

    import nutpie

    mode, variant, dim, fastmath = sys.argv[1:]
    variant, dim = int(variant), int(dim)
    shared = mode != "constant"
    n = (150_000 if mode == "large_shared" else 5) + variant
    y_value = np.arange(n, dtype=float) / n + variant
    scale_value = np.float64(1.5 + variant)
    offset_value = np.float64(variant)
    with pm.Model() as model:
        a = pm.Normal("a", shape=dim)
        y = pm.Data("y", y_value) if shared else y_value
        scale = pm.Data("scale", scale_value) if shared else scale_value
        offset = pm.Data("offset", offset_value) if shared else offset_value
        pm.Normal("obs", a.sum(), sigma=scale, observed=y)
        pm.Deterministic("shifted", a + offset)
    kwargs = {}
    if fastmath != "default":
        kwargs["fastmath"] = fastmath == "true"
    compiled = nutpie.compile_pymc_model(model, backend="numba", **kwargs)
    logp_ref = model.compile_logp(mode="FAST_COMPILE")
    gradient_ref = model.compile_dlogp(mode="FAST_COMPILE")
    pointer = ctypes.POINTER(ctypes.c_double)
    # Check error returns on the same callbacks whose disk caching is tested.
    invalid_theta = np.full(dim, np.inf)
    gradient = np.empty(dim)
    logp = ctypes.c_double()
    args = (
        invalid_theta.ctypes.data_as(pointer),
        gradient.ctypes.data_as(pointer),
        ctypes.byref(logp),
        compiled.user_data.ctypes.data,
    )
    assert compiled.compiled_logp_func.ctypes(dim + 1, *args) == -1
    if fastmath == "default":
        # Check non-finite inputs with the default compiler settings.
        assert compiled.compiled_logp_func.ctypes(dim, *args) == 3
    expanded = np.empty(compiled.n_expanded)
    assert (
        compiled.compiled_expand_func.ctypes(
            dim,
            compiled.n_expanded + 1,
            invalid_theta.ctypes.data_as(pointer),
            expanded.ctypes.data_as(pointer),
            compiled.user_data.ctypes.data,
        )
        == -1
    )

    def evaluate(compiled, theta):
        gradient = np.empty(dim)
        logp = ctypes.c_double()
        status = compiled.compiled_logp_func.ctypes(
            dim,
            theta.ctypes.data_as(pointer),
            gradient.ctypes.data_as(pointer),
            ctypes.byref(logp),
            compiled.user_data.ctypes.data,
        )
        assert status == 0
        expanded = np.empty(compiled.n_expanded)
        status = compiled.compiled_expand_func.ctypes(
            dim,
            compiled.n_expanded,
            theta.ctypes.data_as(pointer),
            expanded.ctypes.data_as(pointer),
            compiled.user_data.ctypes.data,
        )
        assert status == 0
        return logp.value, gradient, expanded

    for theta in [np.linspace(-0.5, 0.5, dim), np.linspace(0.2, 0.7, dim)]:
        logp, gradient, expanded = evaluate(compiled, theta)
        np.testing.assert_allclose(logp, logp_ref({"a": theta}), rtol=1e-10)
        np.testing.assert_allclose(gradient, gradient_ref({"a": theta}), rtol=1e-10)
        np.testing.assert_allclose(
            expanded, np.concatenate([theta, theta + offset_value])
        )

    if shared:
        previous = evaluate(compiled, theta)
        updates = {
            "y": np.arange(7, dtype=float),
            "scale": np.float64(3),
            "offset": np.float64(10),
        }
        updated = compiled.with_data(**updates)
        pm.set_data(updates, model=model)
        logp, gradient, expanded = evaluate(updated, theta)
        np.testing.assert_allclose(logp, logp_ref({"a": theta}), rtol=1e-10)
        np.testing.assert_allclose(gradient, gradient_ref({"a": theta}), rtol=1e-10)
        np.testing.assert_allclose(expanded, np.concatenate([theta, theta + 10]))
        for before, after in zip(previous, evaluate(compiled, theta), strict=True):
            np.testing.assert_array_equal(before, after)

    print(
        json.dumps(
            {
                "hits": [
                    compiled.compiled_logp_func.cache_hits,
                    compiled.compiled_expand_func.cache_hits,
                ]
            }
        )
    )


if __name__ == "__main__":
    cache_worker()
