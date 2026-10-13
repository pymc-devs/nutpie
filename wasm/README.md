# Nutpie `_lib` in Xeus/Emscripten: bounded prototype

This branch builds a real `nutpie._lib` Python extension for
`wasm32-unknown-emscripten`. It reuses Nutpie's PyMC compiler, Numba callbacks,
`PyMcModel`, shared-data ownership and the nuts-rs sampler. It does not use the
independent-memory nuts-rs-wasm logp/gradient bridge.

This is an experimental PyMC-only execution path. On Emscripten:

```python
import nutpie

compiled = nutpie.compile_pymc_model(model)
idata = nutpie.sample(compiled, chains=2, tune=750, draws=500, seed=42)
idata.posterior
idata.sample_stats
```

The result is Nutpie's usual ArviZ/xarray object (a DataTree with ArviZ 1).
Native and WASM sampling share the same `_dict_to_arviz` implementation. Only
storage decoding differs: native results arrive through Arrow; the prototype
currently materializes flattened float64 values from Rust dictionaries.
Named dimensions, string coordinates, deterministic variables and optional
`store_unconstrained=True` are preserved. Sample stats currently contain only
`diverging`, `n_steps` and `step_size`; total gradient evaluations and leapfrog
steps are recorded as attributes.

This is a limited synchronous `sample`, not the full native API. No Arrow/Zarr
storage, progress, cancellation, warmup storage, initialization retry policy,
parallel chains, Stan or flow support is provided. Unsupported keyword options
raise errors. Terminating the worker remains possible. `sample_raw` and
`_lib.evaluate_pymc` remain available for low-level probes. Native builds retain
the existing sampler and dependencies.

## Changes and observed blockers

The unmodified upstream build at
`98c0e879cdd110c14e6bc82a001251b53a990f58` fails in `mio 1.2.2`:

```text
error: This wasm target is unsupported by mio. If using Tokio, disable the net feature.
```

The prototype makes native-only dependencies and modules conditional on the
Emscripten target, and uses nuts-rs with its default parallel feature disabled.
A small synchronous Rust entry point invokes the existing chain methods; it
contains no new NUTS, adaptation, model compilation or transform implementation.
Shared transform-adapter code was moved out of the native wrapper so the PyMC
model implementation can be reused unchanged apart from its import.

Other necessary changes:

- The Numba callback ABI now matches Rust: `intc` return status and `uintp`
  dimensions, corresponding to `c_int` and `usize`. Upstream used `int64` and
  `uint64`. WASM indirect-call signatures must match exactly; pointer-sized
  dimensions are 32-bit in this runtime.
- `CompiledModel` moved out of `sample.py`, so importing the compiler does not
  require PyArrow or the native storage classes. The native `sample` module
  still imports/re-exports that same class. The browser runtime has neither
  PyArrow nor arro3 installed.
- The final ArviZ/xarray conversion moved from `sample.py` into `result.py`.
  Both the native Arrow path and the WASM decoder call this same function;
  xarray does not require PyArrow.
- Package initialization selects the limited WASM exports on Emscripten.
- Coordinate extraction accepts pandas `StringArray` alongside
  `ArrowStringArray`. Without this, even the generated
  `unconstrained_parameter` coordinate prevented model construction in Xeus.

## Reproduce

Tested toolchain: Emscripten **4.0.9**, Rust nightly
`1.96.0-nightly (3102493c7 2026-03-12)`, Python **3.13.1** in the published Xeus
runtime, Chrome **146.0.7680.153** on macOS ARM64. The standard library is rebuilt
against that Emscripten installation. This is not a claim of compatibility with
other Emscripten ABIs or Python versions.

Install and activate Emscripten 4.0.9 with emsdk, then:

```sh
source /path/to/emsdk/emsdk_env.sh
rustup component add rust-src --toolchain nightly
bash wasm/build.sh
```

This produces a **release** extension plus `target/nutpie-probe.zip`. To package
an ABI-tagged experimental wheel, first commit the source and run:

```sh
python3 wasm/make_wheel.py
```

The wheel, `manifest.json` and `SHA256SUMS` appear in `target/wasm-dist`.
The package is `nutpie==0.16.12+wasm.1`, tagged
`cp313-cp313-emscripten_4_0_9_wasm32`. It contains the Python sources, the WASM
side module named `nutpie/_lib.so`, dependency license notices and build metadata.
`--allow-dirty` is only for local probes; released artifacts must have
`source_dirty: false` in their manifest.

Download and extract `nuts-rs-wasm-runtime-v0.1.0.tar.gz` from the
[nuts-rs-wasm release](https://github.com/pymc-labs/nuts-rs-wasm/releases/tag/v0.1.0)
into a new static directory. Its SHA-256 is
`dc73b5f69ef1946e3409f3ab0884a2b17a3f4d1956069ce0b70f0fc51665a6f4`.

Use a nuts-rs-wasm checkout at `cbd62c2188c28a8b159b6af6ecee6234a5e4ad56` for the
kernel client and MMM fixtures:

```sh
python3 wasm/prepare_site.py /path/to/nuts-rs-wasm /path/to/site
python3 -m http.server 8768 --bind 127.0.0.1 --directory /path/to/site
```

Open `http://127.0.0.1:8768`. The page initializes Xeus and executes `probe.py`.
The borrowed client's only operations are initialization and Python execution;
its compiler, WASM sampler and model bridge are not called.

For a headless run, in another terminal:

```sh
npm ci --prefix wasm
CHROME_PATH='/path/to/chrome' node wasm/run_browser.mjs \
  http://127.0.0.1:8768 target/probe-output.json
```

The runner exits unsuccessfully on exceptions or a missing completion marker.
The Python assertions check logp/gradient agreement with PyMC, transformations,
seeded repeatability, invalid inputs, mutable data, callback ownership, runtime
memory growth and a bounded MMM run. The MMM run is a smoke test, not evidence
of convergence or a speed comparison. Memory growth uses the runtime allocator
so Emscripten refreshes its own views.

## Verified result

The complete checked-in browser harness passed on September 11, 2026. See
[validation.json](validation.json) for the recorded results.

- Gaussian: 2 chains × 500 retained draws; mean -0.0818, standard deviation 0.9558.
- HalfNormal, Beta, simplex and deterministic expansion checks passed.
- Logp/gradient checks at three seeded positions per model agreed with PyMC.
- Float32/int32 shared data, independent updated handles, callback ownership
  after garbage collection, and runtime-managed memory growth passed.
- MMM: 15 unconstrained parameters, 100 warmup steps and 50 retained draws;
  finite outputs and three density/gradient comparisons passed.
- Native `cargo check --locked`, a built native extension and six existing
  PyMC/Numba tests passed. Those cover float32, no-prior models, coordinates,
  extra variables and shared data. The six tests passed again after sharing
  the result converter; a separate transformed/shared-data regression passed.
  This is not the full Stan/JAX suite.

The original `validation.json` records the initial debug smoke test. Release
performance and packaging checks are documented in [evaluation.md](evaluation.md).
The benchmark separates compilation, sampling/materialization and diagnostics;
output behavior differs, so it does not isolate JavaScript call overhead.

## Remaining work before an upstream-ready port

Agree on feature flags and the public synchronous sampling API, reuse/extend
nuts-rs execution helpers, define output/storage and progress behavior, complete
native regression coverage and add browser CI.
The separation of the Python frontend can still be useful, but `_lib` itself is
not fundamentally excluded from WASM.
