import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pymc")
pytest.importorskip("pytensor.link.numba.cache")


def run_worker(tmp_path, variant):
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
            str(variant),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.splitlines()[-1])


@pytest.mark.pymc
def test_numba_wrapper_disk_cache(tmp_path):
    assert run_worker(tmp_path, variant=0) == [0, 0]
    assert run_worker(tmp_path, variant=1) == [1, 1]


def cache_worker():
    import pymc as pm

    import nutpie

    variant = int(sys.argv[1])
    n = 150_000 + variant
    y_value = np.arange(n, dtype=float) / n + variant
    offset_value = np.float64(variant)
    with pm.Model() as model:
        a = pm.Normal("a", shape=2)
        y = pm.Data("y", y_value)
        offset = pm.Data("offset", offset_value)
        pm.Normal("obs", a.sum(), observed=y)
        pm.Deterministic("shifted", a + offset)
    compiled = nutpie.compile_pymc_model(model, backend="numba")
    logp_ref = model.compile_logp(mode="FAST_COMPILE")
    gradient_ref = model.compile_dlogp(mode="FAST_COMPILE")
    pointer = ctypes.POINTER(ctypes.c_double)
    theta = np.array([-0.5, 0.5])

    def check(compiled, offset):
        gradient = np.empty(compiled.n_dim)
        logp = ctypes.c_double()
        status = compiled.compiled_logp_func.ctypes(
            compiled.n_dim,
            theta.ctypes.data_as(pointer),
            gradient.ctypes.data_as(pointer),
            ctypes.byref(logp),
            compiled.user_data.ctypes.data,
        )
        assert status == 0
        np.testing.assert_allclose(logp.value, logp_ref({"a": theta}), rtol=1e-10)
        np.testing.assert_allclose(gradient, gradient_ref({"a": theta}), rtol=1e-10)
        expanded = np.empty(compiled.n_expanded)
        status = compiled.compiled_expand_func.ctypes(
            compiled.n_dim,
            compiled.n_expanded,
            theta.ctypes.data_as(pointer),
            expanded.ctypes.data_as(pointer),
            compiled.user_data.ctypes.data,
        )
        assert status == 0
        np.testing.assert_allclose(expanded, np.concatenate([theta, theta + offset]))

    check(compiled, offset_value)
    updates = {"y": np.arange(7, dtype=float), "offset": np.float64(10)}
    updated = compiled.with_data(**updates)
    pm.set_data(updates, model=model)
    check(updated, 10)

    print(
        json.dumps(
            [
                compiled.compiled_logp_func.cache_hits,
                compiled.compiled_expand_func.cache_hits,
            ]
        )
    )


if __name__ == "__main__":
    cache_worker()
