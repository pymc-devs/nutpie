# Experimental browser adapter

Run Nutpie's sequential nuts-rs sampler as a standalone WebAssembly module,
calling a PyMC logp and gradient compiled by browser Numba. This example is an
integration proposal, not a browser build of the nutpie Python extension.
`import nutpie` and `nutpie.sample` still require the native extension.

## Build and test

From this directory, with Rust 1.94.0 and Node available:

```sh
rustup target add wasm32-unknown-unknown
cargo build --locked --release --target wasm32-unknown-unknown --manifest-path adapter/Cargo.toml
node test_bridge.mjs
```

The independent crate deliberately excludes the Python extension's Arrow,
Zarr, Tokio, Stan and parallel sampling dependencies. Its dependency commit
is pinned to the tested nuts-rs 0.18.3 source. `Cargo.lock` is committed.

In a Python environment with PyMC 6.2.0, PyTensor 3.2.4 and Numba 0.66.0:

```sh
PYTENSOR_FLAGS=cxx=,blas__ldflags=,numba__cache=False OPENBLAS_NUM_THREADS=1 python test_compile.py
```

The workflow builds a downloadable **experimental-browser-adapter** artifact
containing the WASM module, Python compiler and JavaScript bridge. This is not
a PyPI wheel or a complete Python runtime. No release is published automatically.

## Embed in a browser application

Provide a working Xeus/Emscripten Python runtime with compatible PyMC, PyTensor
and Numba packages. This has been tested with Python 3.13, Numba 0.66, llvmlite
0.48, PyMC 6.2.0 and a locally patched PyTensor 3.2.4 WASM build. It has **not**
been validated against stock Pyodide or arbitrary runtime/package versions.
The caller must expose the runtime's actual `wasmMemory` and `wasmTable`.
In the experiment the generated Xeus loader was locally patched to export
its existing memory; a supported runtime export is needed for distribution.

Execute `compile_model.py` in that runtime (or import it directly from its
filesystem), then compile a model:

```python
import pymc as pm

with pm.Model() as model:
    x = pm.Normal("x", initval=0.1)

compiled = compile_browser_model(model)
config = compiled.config()  # send this JSON-serializable dict to the worker JS
```

Keep `compiled` alive for the entire sampling call. Once the Python execution
has completed, use the same worker's JavaScript context:

```javascript
import {sample} from './bridge.mjs';

const bytes = await (await fetch('./nuts_browser_adapter.wasm')).arrayBuffer();
const result = await sample({
  bytes,
  runtime: Module, // the initialized Emscripten runtime
  model: config,
  chains: 2, tune: 750, draws: 500, seed: 42,
  onProgress: progress => postMessage({progress}),
});
postMessage({result});
```

Sampling blocks this worker; use a dedicated worker so the page stays responsive.
Do not start concurrent fits or invoke Python while this call runs. Terminating
the worker cancels the entire runtime; cooperative cancellation is not implemented.
The host is responsible for request sizes, downloads and posterior diagnostics.

The files can be served together by a static host. All model computation stays
in the browser. No server executes Python. Hosting these files does not install
the requisite Python runtime: package/runtime compatibility still matters.

## ABI and limitations

The exported Rust `run(n, chains, tune, draws, seed, start)` consumes an aligned
array of `n` doubles in **Rust memory**, returns a status, and exposes a JSON
buffer via `result_ptr/result_len`. The JS wrapper allocates/frees the input;
the output remains valid until the next call on that instance. Raw C exports
are unsafe interfaces for trusted callers, not validated RPC endpoints.

Rust imports `model_logp(x, gradient, n) -> f64`. The bridge copies positions
into **Emscripten memory**, invokes the Numba callback through its function
table, then copies gradients back. Views are reacquired after callback execution
because Emscripten can grow memory. Python is not called per leapfrog step.
The small JS copy and callback overhead remains; the two modules do not share
an allocator or function table.

- Continuous models with fully Numba-compilable graphs only. Shared data are
  frozen at compilation; changes require recompilation.
- Diagonal mass adaptation, target acceptance 0.9, max depth 10; sequential chains.
- Identical initial positions, seeds `seed + chain`; callers should choose a
  finite initial position with a usable gradient. No jitter/init retry policy.
- Samples are **unconstrained**, in the reported `layout` order. Transformation,
  deterministics and conversion to InferenceData are not implemented here.
- No Stan, JAX, normalizing flows, parallelism or existing trace storage backends.
- The compiler uses PyTensor's `vm.jit_fn`, so package-version compatibility
  needs explicit testing. Errors in unsupported compiled operations can abort
  the WASM call; this is not a production error-recovery API.
- The pinned build retains unused wasm-bindgen imports. Throwing guards make
  any unexpected use fail rather than silently supplying a fake implementation.

## Experimental evidence

A 179-week, 15-parameter PyMC-Marketing 1.1.0 MMM completed 2 chains of 750 warmup
and 500 retained draws per chain. Native arm64 sampling took 0.801 s, browser
sampling 8.407 s; model preparation/compilation took 11.893/14.757 s respectively.
These are single runs of the original prototype on the same Mac at different
times, not controlled repeated benchmarks of this proposed API. Sampling time
includes warmup and serialization, excluding imports, compilation, independent
numerical validation and posterior diagnostics.

Both runs had zero retained divergences, but max R-hat was 1.018/1.023 and min
bulk ESS 125/183. They do not establish convergence or a speedup at equivalent
precision over PyMC NUTS. Browser logp/gradient agreed with independent PyMC
functions at three positions (maximum absolute errors 2.84e-13/1.10e-11).

The tests here separately cover Gaussian posterior moments in the actual Rust
WASM module, bridge memory growth, progress/error handling, transformed PyMC
logp/gradients, and frozen shared data. The Node bridge test substitutes a JS
Gaussian callback; it does not test a browser Numba runtime. A full Xeus browser
integration remains a manual check because no runtime is bundled here.

The proposed general compiler and JS bridge were also manually checked against
that same MMM in the browser: 8.242 s sampling, 14.847 s preparation/compilation,
80,911 logp evaluations, zero divergences, max R-hat 1.0233 and min bulk ESS
183.4, matching the prototype's posterior diagnostics.
