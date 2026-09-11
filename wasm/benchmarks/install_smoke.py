# Top-level await is executed by the Xeus notebook kernel.
# ruff: noqa: F704, PLE1142
"""Install the tagged wheel with pip in a fresh Xeus kernel and test public APIs."""

import os

os.environ["PYTENSOR_FLAGS"] = "cxx=,blas__ldflags=,numba__cache=False"
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import pyjs

fetch = pyjs.js.Function(
    "url",
    "path",
    """return (async()=>{
 const r=await fetch(url);if(!r.ok)throw Error(r.status);
 Module.FS.writeFile(path,new Uint8Array(await r.arrayBuffer()));})();""",
)
await fetch("../manifest.json", "/tmp/manifest.json")
manifest = json.loads(Path("/tmp/manifest.json").read_text())
wheel = "/tmp/" + manifest["filename"]
await fetch("../" + manifest["filename"], wheel)
assert hashlib.sha256(Path(wheel).read_bytes()).hexdigest() == manifest["sha256"]
# The released runtime omits pip._vendor. Bootstrap unmodified PyPA pip,
# then let pip validate the ABI tag and install the extension normally.
await fetch("../pip-25.2-py3-none-any.whl", "/tmp/pip.whl")
assert (
    hashlib.sha256(Path("/tmp/pip.whl").read_bytes()).hexdigest()
    == "6d67a2b4e7f14d8b31b8b52648866fa717f45a1eb70e83002f4331d07e953717"
)
with zipfile.ZipFile("/tmp/pip.whl") as archive:
    archive.extractall("/tmp/complete-pip")
sys.path.insert(0, "/tmp/complete-pip")
from pip._internal.cli.main import main

assert (
    main(
        [
            "install",
            "--no-deps",
            "--no-index",
            "--no-compile",
            "--disable-pip-version-check",
            "--target",
            "/tmp/nutpie-wheel",
            wheel,
        ]
    )
    == 0
)
sys.path.insert(0, "/tmp/nutpie-wheel")

from importlib.metadata import version

import arviz_stats as az
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

import nutpie

with pm.Model(coords={"component": pd.Index(["a", "b", "c"], dtype="str")}) as model:
    offset = pm.Data("offset", np.array([0, 1, 0], dtype="int32"))
    scale = pm.HalfNormal("scale")
    weights = pm.Dirichlet("weights", np.ones(3), dims="component")
    pm.Deterministic("scaled", scale * weights, dims="component")
    pm.Normal("obs", weights + offset, 1, observed=[1.0, 2.0, 1.0])
compiled = nutpie.compile_pymc_model(model)
start = time.perf_counter()
idata = nutpie.sample(
    compiled, chains=2, tune=250, draws=200, seed=42, store_unconstrained=True
)
elapsed = time.perf_counter() - start
assert isinstance(idata, xr.DataTree)
assert idata.posterior.weights.dims == ("chain", "draw", "component")
assert list(idata.posterior.component.values) == ["a", "b", "c"]
assert idata.posterior.weights.shape == (2, 200, 3)
np.testing.assert_allclose(idata.posterior.weights.sum("component"), 1, atol=1e-12)
np.testing.assert_allclose(
    idata.posterior.scaled, idata.posterior.scale * idata.posterior.weights, atol=1e-12
)
assert "weights_simplex__" not in idata.posterior
assert "weights_simplex__" in idata.unconstrained_posterior
assert idata.sample_stats.diverging.dtype == bool
assert idata.sample_stats.n_steps.dtype == np.uint64
assert int(idata.sample_stats.attrs["logp_evaluations"]) > 0
assert "pyarrow" not in sys.modules and "arro3" not in sys.modules
table = az.summary(idata, var_names=["scale", "weights"])
assert np.isfinite(table["mean"]).all()
updated = compiled.with_data(offset=np.array([1, 0, 1], dtype="int32"))
new = nutpie.sample(updated, chains=2, tune=150, draws=100, seed=42)
assert "unconstrained_posterior" not in new.children
assert np.isfinite(new.posterior.scaled).all()
for settings in [
    {"draws": 0},
    {"target_accept": 1.0},
    {"max_depth": 0},
    {"progress_bar": True},
]:
    try:
        nutpie.sample(compiled, **settings)
    except (ValueError, RuntimeError, TypeError):
        pass
    else:
        raise AssertionError(f"Invalid/unsupported settings accepted: {settings}")
assert version("nutpie") == manifest["version"]
print(
    "NUTPIE_WASM_PROBE_PASSED "
    + json.dumps(
        {
            "test": "wheel-install-and-public-xarray-api",
            "manifest": manifest,
            "posterior_shape": list(idata.posterior.weights.shape),
            "coords": idata.posterior.component.values.tolist(),
            "groups": list(idata.children),
            "sample_seconds": elapsed,
            "max_rhat": float(table.r_hat.max()),
            "min_ess": float(table.ess_bulk.min()),
        }
    ),
    flush=True,
)
