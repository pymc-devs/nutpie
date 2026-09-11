# Nutpie `_lib` in Xeus/Emscripten: bounded prototype

This branch builds a real `nutpie._lib` Python extension for
`wasm32-unknown-emscripten`. It reuses Nutpie's PyMC compiler, Numba callbacks,
`PyMcModel`, shared-data ownership and the nuts-rs sampler. It does not use the
independent-memory nuts-rs-wasm logp/gradient bridge.

This is an experimental PyMC-only execution path, not a complete port of
`nutpie.sample`. The WASM exports are `sample_raw(native_model, ...)` and
`_lib.evaluate_pymc(native_model, position)`. `native_model` is obtained with
`compiled._make_model(None)` for this probe. Raw results are nested
chain/draw/variable dictionaries of flattened float64 values. There is no Arrow,
Zarr, ArviZ conversion, progress, cancellation, initialization retry policy,
parallel chains, Stan or flow support in this path. Terminating its worker is
still possible. Native builds retain the existing sampler and dependencies.

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

This produces `target/nutpie-probe.zip` containing the Python sources and the
WASM side module named `nutpie/_lib.so`. It is a probe archive, not an installable
wheel; this bypasses the full package's native dependency installation.

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
- `cargo check --locked` passed for the native target; native runtime regression
  tests were not run. The existing unused-mut warning in Stan remains.

Recorded timings are from a debug build and different test scopes. They are not
an end-to-end performance comparison against the existing browser adapter.

## Remaining work before an upstream-ready port

Agree on feature flags and a public synchronous sampling API, reuse/extend the
nuts-rs execution helpers, define output/storage and progress behavior, produce
an ABI-tagged package, and add native runtime regression coverage and browser CI.
The separation of the Python frontend can still be useful, but `_lib` itself is
not fundamentally excluded from WASM.
