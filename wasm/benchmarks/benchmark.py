# Top-level await is executed by the Xeus notebook kernel.
# ruff: noqa: F704, PLE1142, S102
"""Controlled browser comparison; all assertions and timing scopes are explicit."""

import os

os.environ["PYTENSOR_FLAGS"] = "cxx=,blas__ldflags=,numba__cache=False"
import gc
import json
import sys
import time
import zipfile
from pathlib import Path

import pyjs

fetch = pyjs.js.Function(
    "url",
    "path",
    """return (async()=>{const r=await fetch(url);if(!r.ok)throw Error(r.status);Module.FS.writeFile(path,new Uint8Array(await r.arrayBuffer()));})();""",
)
await fetch("../nutpie-probe.zip", "/tmp/nutpie.zip")
with zipfile.ZipFile("/tmp/nutpie.zip") as z:
    z.extractall("/tmp/nutpie-benchmark")
sys.path.insert(0, "/tmp/nutpie-benchmark")
for filename in ["compile_model.py", "results.py", "mmm_model.py", "mmm_example.csv"]:
    await fetch("../" + filename, "/tmp/" + filename)
sys.path.insert(0, "/tmp")
import sysconfig

import arviz_stats as az
import numba
import numpy as np
import packaging.tags
from compile_model import compile_browser_model
from results import load_worker_result, to_inference_data

import nutpie

print(
    "RUNTIME_TAGS",
    sysconfig.get_platform(),
    [str(t) for t in list(packaging.tags.sys_tags())[:3]],
    flush=True,
)
print(
    "DEPENDENCIES",
    {
        k: __import__(k).__version__
        for k in ["numpy", "pandas", "numba", "pymc", "pytensor", "xarray"]
    },
    flush=True,
)

bridge = pyjs.js.Function(
    "config",
    "seed",
    "draws",
    "tune",
    """return (async()=>{
 const {sample,stageWorkerResult}=await import('/nuts/bridge.mjs');
 if(!self.benchBytes) self.benchBytes=await (await fetch('/nuts/nuts_browser_adapter.wasm')).arrayBuffer();
 let live=0;
 const r=await sample({bytes:self.benchBytes,runtime:Module,model:JSON.parse(config),seed,draws,tune,chains:2,targetAccept:.9,jitter:0,maxDepth:10,resultFormat:'binary',retainUnconstrained:false,
 onSamples:s=>{live+=s.draws;self.postMessage({nutpieBench:'samples',samples:s},[s.values.buffer]);}});
 const meta=stageWorkerResult(r,Module.FS,'/tmp/bench-'+crypto.randomUUID());
 const traceBytes=r.traces.reduce((n,t)=>n+t.bytes.byteLength,0);
 const traceCount=r.traces.length;
 self.postMessage({nutpieBench:'traces',traces:r.traces},r.traces.map(t=>t.bytes.buffer));
 return JSON.stringify({meta,live,traceBytes,traceCount,sampling_seconds:r.sampling_seconds,logp_evaluations:r.logp_evaluations,leapfrog_steps:r.leapfrog_steps});
})();""",
)


def build_model():
    scope = {"DATA_PATH": "/tmp/mmm_example.csv"}
    exec(Path("/tmp/mmm_model.py").read_text(), scope)
    return scope["model"], scope["var_names"]


def adapt_same_graph(bm):
    """Nutpie ABI wrappers around the SAME jitted graph used by the old bridge."""
    n = len(bm.initial)
    ne = len(bm.expanded)
    inner = bm.function.vm.jit_fn
    expand = bm.expand_function.vm.jit_fn
    ptr = numba.types.CPointer(numba.types.float64)

    @numba.cfunc(
        numba.types.intc(numba.types.uintp, ptr, ptr, ptr, numba.types.voidptr)
    )
    def density(dim, x, g, lp, data):
        value, grad = inner(numba.carray(x, (n,)))
        numba.carray(lp, ())[()] = value.item()
        numba.carray(g, (n,))[:] = grad
        return 0

    @numba.cfunc(
        numba.types.intc(
            numba.types.uintp, numba.types.uintp, ptr, ptr, numba.types.voidptr
        )
    )
    def expansion(dim, expanded, x, out, data):
        (values,) = expand(numba.carray(x, (n,)))
        numba.carray(out, (ne,))[:] = values
        return 0

    owner = (bm, density, expansion)
    variables = bm.expanded_layout
    dims = {v["name"]: v["dims"] for v in variables}
    sizes = {d: size for v in variables for d, size in zip(v["dims"], v["shape"])}
    metadata = nutpie._lib.PyVariable.new_variables(
        [v["name"] for v in variables],
        ["float64"] * len(variables),
        [v["shape"] for v in variables],
        sizes,
        dims,
    )
    # Convert coordinate lists to arrays; PyValue reserves string lists for labels.
    coords = {
        k: np.asarray(v, dtype=object)
        if np.asarray(v).dtype.kind in "US"
        else np.asarray(v)
        for k, v in bm.coords.items()
    }
    return nutpie._lib.PyMcModel(
        nutpie._lib.LogpFunc(density.address, 0, owner),
        nutpie._lib.ExpandFunc(n, ne, expansion.address, 0, owner),
        metadata,
        n,
        sizes,
        coords,
        lambda seed: bm.initial.copy(),
        None,
    )


def native_result(raw, layout, coords):
    samples, stats, evaluations, steps = raw
    expanded = np.asarray(
        [
            [sum((row[v["name"]] for v in layout), []) for row in chain]  # noqa: RUF017
            for chain in samples
        ]
    )
    return {
        "expanded_samples": expanded,
        "stats": np.asarray(stats),
        "expanded_layout": layout,
        "coords": coords,
        "logp_evaluations": evaluations,
        "leapfrog_steps": steps,
    }


def diagnostics(result, names):
    idata = to_inference_data(result)
    table = az.summary(idata, var_names=names, round_to="none")
    return {
        "max_rhat": float(table.r_hat.max()),
        "min_ess": float(table.ess_bulk.min()),
        "min_tail_ess": float(table.ess_tail.min()),
        "mean": table["mean"].tolist(),
        "mcse_mean": table["mcse_mean"].tolist(),
        "labels": list(table.index),
        "divergences": int(idata.sample_stats.diverging.sum()),
    }


# Preload model-building imports before compilation timing.
model, names = build_model()
report = {
    "settings": {
        "chains": 2,
        "tune": 750,
        "draws": 500,
        "target_accept": 0.9,
        "max_depth": 10,
        "initialization": "same fixed PyMC base point, jitter disabled",
        "seeds": [42, 142, 242, 342, 442],
    },
    "compile": [],
    "runs": [],
}
for repeat in range(3):
    order = ["bridge", "nutpie"] if repeat % 2 == 0 else ["nutpie", "bridge"]
    for mode in order:
        gc.collect()
        start = time.perf_counter()
        m, selected = build_model()
        model_time = time.perf_counter() - start
        start = time.perf_counter()
        if mode == "bridge":
            compiled = compile_browser_model(m, var_names=selected)
        else:
            compiled = nutpie.compile_pymc_model(
                m, var_names=selected, freeze_model=True, jitter_rvs=set()
            )
        elapsed = time.perf_counter() - start
        report["compile"].append(
            {
                "repeat": repeat,
                "mode": mode,
                "model_seconds": model_time,
                "compile_seconds": elapsed,
            }
        )
        print("COMPILED", report["compile"][-1], flush=True)
        if mode == "bridge":
            bm = compiled
        else:
            direct = compiled
model, names = build_model()
# Match parameter names: freezing the Nutpie model can change variable order.
fixed = bm.initial.tolist()
bridge_labels = [(v["name"], i) for v in bm.layout for i in range(v["size"])]
direct_labels = []
for name, sl, shape in zip(*direct.shape_info):
    if sl.start >= direct.n_dim:
        break
    direct_labels.extend((name, i) for i in range(int(np.prod(shape))))
assert len(direct_labels) == len(bridge_labels) == direct.n_dim
permutation = np.asarray([bridge_labels.index(label) for label in direct_labels])
direct_fixed = np.asarray(fixed)[permutation].tolist()
report["parameter_permutation"] = permutation.tolist()
shared = adapt_same_graph(bm)
full = direct._make_model(None)
for x in [np.asarray(fixed), np.asarray(fixed) + 0.01, np.asarray(fixed) - 0.01]:
    a = nutpie._lib.evaluate_pymc(shared, x.tolist())
    b = nutpie._lib.evaluate_pymc(full, x[permutation].tolist())
    np.testing.assert_allclose(a[0], b[0], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(
        np.asarray(a[1])[permutation], b[1], rtol=1e-8, atol=1e-8
    )
report["density_equivalence"] = True
layout = bm.expanded_layout
config = json.dumps(bm.config())
# Untimed warm-up initializes each path. Model imports and runtime download are excluded.
await bridge(config, 7, 30, 30)
for native, point in [(shared, fixed), (full, direct_fixed)]:
    nutpie._lib.sample_pymc_stats(
        native, draws=30, tune=30, chains=2, seed=7, initial_positions=[point, point]
    )
for index, seed in enumerate(report["settings"]["seeds"]):
    order = (
        ["bridge", "same_graph_direct", "nutpie"]
        if index % 2 == 0
        else ["nutpie", "same_graph_direct", "bridge"]
    )
    pair = {}
    for mode in order:
        gc.collect()
        started = time.perf_counter()
        if mode == "bridge":
            info = json.loads(str(await bridge(config, seed, 500, 750)))
            result = load_worker_result(info["meta"])
            assert info["live"] == 1000 and info["traceCount"] == 4
            details = {k: v for k, v in info.items() if k != "meta"}
        else:
            raw = nutpie._lib.sample_pymc_stats(
                shared if mode == "same_graph_direct" else full,
                draws=500,
                tune=750,
                chains=2,
                seed=seed,
                target_accept=0.9,
                max_depth=10,
                initial_positions=[
                    fixed if mode == "same_graph_direct" else direct_fixed
                ]
                * 2,
            )
            sampled = time.perf_counter()
            result = native_result(raw, layout, bm.coords)
            details = {
                "sampling_seconds": sampled - started,
                "logp_evaluations": raw[2],
                "leapfrog_steps": raw[3],
            }
        materialized = time.perf_counter()
        diag = diagnostics(result, names)
        finished = time.perf_counter()
        row = {
            "mode": mode,
            "seed": seed,
            **details,
            "materialized_seconds": materialized - started,
            "diagnostics_seconds": finished - materialized,
            **diag,
        }
        row["min_ess_per_second"] = row["min_ess"] / row["materialized_seconds"]
        report["runs"].append(row)
        pair[mode] = np.asarray(result["expanded_samples"]).copy()
        print("FIT", json.dumps(row), flush=True)
        del result
        if mode != "bridge":
            del raw
    # Identical numerical graphs and fixed starts should preserve paths; record instead of assuming.
    max_delta = float(np.max(np.abs(pair["bridge"] - pair["same_graph_direct"])))
    report.setdefault("paired_max_abs_delta", []).append(
        {"seed": seed, "value": max_delta}
    )
    del pair
print("NUTPIE_WASM_PROBE_PASSED " + json.dumps(report), flush=True)
