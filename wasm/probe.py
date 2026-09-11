# Top-level await is executed by the Xeus notebook kernel.
# ruff: noqa: F704, PLE1142, S102
"""Run inside the published Xeus runtime; see README.md for the harness."""

import os

os.environ["PYTENSOR_FLAGS"] = "cxx=,blas__ldflags=,numba__cache=False"
import gc
import json
import sys
import time
import zipfile
from pathlib import Path

import pyjs

download = pyjs.js.Function(
    "url",
    "path",
    """return (async () => {
 const response = await fetch(url);
 if (!response.ok) throw Error(`Download failed: ${response.status}`);
 Module.FS.writeFile(path, new Uint8Array(await response.arrayBuffer()));
})();""",
)
await download("../nutpie-probe.zip", "/tmp/nutpie-probe.zip")
with zipfile.ZipFile("/tmp/nutpie-probe.zip") as archive:
    archive.extractall("/tmp/nutpie-probe")
sys.path.insert(0, "/tmp/nutpie-probe")
import numpy as np
import pymc as pm

import nutpie

summary = {
    "python": sys.version,
    "nutpie": nutpie.__version__,
    "extension": nutpie._lib.__file__,
}
print("IMPORTED", summary, flush=True)


def verify_density(model, compiled):
    native = compiled._make_model(None)
    reference_logp = model.compile_logp()
    reference_grad = model.compile_dlogp()
    for seed in [11, 12, 13]:
        position = compiled.initial_point_func(seed)
        point = {}
        offset = 0
        base = model.initial_point()
        for rv in model.free_RVs:
            name = model.rvs_to_values[rv].name
            shape = base[name].shape
            size = base[name].size
            point[name] = position[offset : offset + size].reshape(shape)
            offset += size
        logp, gradient, expanded = nutpie._lib.evaluate_pymc(native, position.tolist())
        np.testing.assert_allclose(logp, reference_logp(point), rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(
            gradient, reference_grad(point), rtol=1e-8, atol=1e-8
        )
        assert all(np.isfinite(v).all() for v in expanded.values())
    return native


with pm.Model() as gaussian:
    mu = pm.Normal("mu")
    pm.Deterministic("twice", 2 * mu)
compiled = nutpie.compile_pymc_model(gaussian)
native = verify_density(gaussian, compiled)
started = time.perf_counter()
result = nutpie.sample_raw(native, draws=500, tune=500, chains=2, seed=42)
values = np.array([[row["mu"][0] for row in chain] for chain in result])
assert abs(values.mean()) < 0.2 and abs(values.std() - 1) < 0.2
assert all(
    abs(row["twice"][0] - 2 * row["mu"][0]) < 1e-12 for chain in result for row in chain
)
assert nutpie.sample_raw(
    native, draws=20, tune=30, chains=1, seed=17
) == nutpie.sample_raw(native, draws=20, tune=30, chains=1, seed=17)
summary["gaussian"] = {
    "shape": list(values.shape),
    "mean": float(values.mean()),
    "std": float(values.std()),
    "seconds": time.perf_counter() - started,
}
for bad in [[], [float("nan")]]:
    try:
        nutpie._lib.evaluate_pymc(native, bad)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Invalid position was accepted")
print("GAUSSIAN_PASSED", summary["gaussian"], flush=True)

with pm.Model(coords={"component": ["a", "b", "c"]}) as transformed:
    scale = pm.HalfNormal("scale")
    probability = pm.Beta("probability", 2, 3)
    simplex = pm.Dirichlet("simplex", np.ones(3), dims="component")
    pm.Deterministic("scaled", scale * simplex, dims="component")
transformed_compiled = nutpie.compile_pymc_model(transformed)
transformed_native = verify_density(transformed, transformed_compiled)
result = nutpie.sample_raw(transformed_native, draws=100, tune=150, chains=1, seed=123)
for row in result[0]:
    assert row["scale"][0] > 0 and 0 < row["probability"][0] < 1
    np.testing.assert_allclose(sum(row["simplex"]), 1, atol=1e-12)
    np.testing.assert_allclose(
        row["scaled"], np.asarray(row["simplex"]) * row["scale"][0], atol=1e-12
    )
summary["transforms"] = {"draws": len(result[0]), "density_gradient_checks": 3}
print("TRANSFORMS_PASSED", flush=True)

with pm.Model() as data_model:
    data = pm.Data("data", np.array([1.0, 2.0, 3.0], dtype="float32"))
    offset = pm.Data("offset", np.array([0, 1, 0], dtype="int32"))
    mu = pm.Normal("mu")
    pm.Normal("obs", mu=mu + offset, sigma=1, observed=data)
print("DATA_COMPILING", flush=True)
data_compiled = nutpie.compile_pymc_model(data_model)
print("DATA_COMPILED", flush=True)
data_native = verify_density(data_model, data_compiled)
print("DATA_VERIFIED", flush=True)
updated = data_compiled.with_data(data=np.array([4.0, 5.0, 6.0], dtype="float32"))
updated_native = updated._make_model(None)
a = nutpie._lib.evaluate_pymc(data_native, [0.0])
b = nutpie._lib.evaluate_pymc(updated_native, [0.0])
np.testing.assert_allclose(a[1], [5.0])
np.testing.assert_allclose(b[1], [14.0])
assert a != b
assert nutpie._lib.evaluate_pymc(data_native, [0.0]) == a
# Native model retains callback owners after Python compiled handles are dropped.
del data_compiled, updated
gc.collect()
assert nutpie._lib.evaluate_pymc(data_native, [0.0]) == a
assert nutpie._lib.evaluate_pymc(updated_native, [0.0]) == b
# Allocate through the runtime so Emscripten refreshes its own memory views.
print("DATA_OWNERSHIP_PASSED", flush=True)
memory_size = pyjs.js.Function("return Module.wasmMemory.buffer.byteLength;")
before = int(memory_size())
allocation = np.ones(before, dtype=np.uint8)
assert int(memory_size()) > before
del allocation
gc.collect()
print("MEMORY_GROWN", before, int(memory_size()), flush=True)
assert nutpie._lib.evaluate_pymc(updated_native, [0.0]) == b
summary["mutable_data"] = {
    "old_gradient": a[1],
    "new_gradient": b[1],
    "ownership_and_memory_growth": True,
}
print("MUTABLE_DATA_PASSED", flush=True)

await download("../mmm_example.csv", "/tmp/mmm_example.csv")
await download("../mmm_model.py", "/tmp/mmm_model.py")
mmm_scope = {"DATA_PATH": "/tmp/mmm_example.csv"}
exec(Path("/tmp/mmm_model.py").read_text(), mmm_scope)
mmm_model = mmm_scope["model"]
started = time.perf_counter()
mmm_compiled = nutpie.compile_pymc_model(mmm_model, var_names=mmm_scope["var_names"])
mmm_native = verify_density(mmm_model, mmm_compiled)
summary["mmm_compile_seconds"] = time.perf_counter() - started
started = time.perf_counter()
mmm_result = nutpie.sample_raw(mmm_native, draws=50, tune=100, chains=1, seed=42)
assert len(mmm_result) == 1 and len(mmm_result[0]) == 50
assert all(np.isfinite(v).all() for row in mmm_result[0] for v in row.values())
summary["mmm"] = {
    "dimensions": int(mmm_compiled.n_dim),
    "draws": 50,
    "tune": 100,
    "density_gradient_checks": 3,
    "seconds": time.perf_counter() - started,
}
print("MMM_PASSED", summary["mmm"], flush=True)
print("NUTPIE_WASM_PROBE_PASSED " + json.dumps(summary), flush=True)
