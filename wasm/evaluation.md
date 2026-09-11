# WASM evaluation — September 11, 2026

The prototype now shares Nutpie's PyMC compiler, model/callback implementation,
and final ArviZ/xarray conversion. The initial reason for raw results was the
Arrow-dependent import path, not an inability to use xarray in Xeus.

## Release benchmark

Chrome 146.0.7680.153, macOS ARM64, Python 3.13.1, Emscripten 4.0.9. Same released
Xeus runtime for all paths. Baseline: `pymc-labs/nuts-rs-wasm` at
`cbd62c2188c28a8b159b6af6ecee6234a5e4ad56`, rebuilt in release mode in a separate
checkout. Nutpie also uses a release build. The baseline uses its pinned
nuts-rs 0.18.3 fork; Nutpie uses crates.io nuts-rs 0.18.3. This is not a comparison
against subsequent changes to the nuts-rs-wasm default branch.

Model: the repository's 15-parameter MMM with seasonality. Each fit: two
sequential chains, 750 warmup + 500 retained draws each, target acceptance 0.9,
maximum tree depth 10. Five seeds, alternating path order. Starting points are
identical by parameter name, with jitter disabled. Nutpie's frozen-model
compiler changes parameter order, so positions and gradients are permuted
explicitly; logp and gradient comparisons passed at three positions.

| Path | Compile median, 3 builds | Sampling + materialization median, 5 fits | Range | Minimum bulk ESS/s, median |
| --- | ---: | ---: | ---: | ---: |
| Existing browser adapter | 14.31 s | 7.71 s | 7.20–8.13 s | 22.55 |
| Same compiled graph through Nutpie `_lib` | excluded (benchmark-only ABI wrapper) | 7.90 s | 7.60–8.28 s | 19.08 |
| Nutpie compiler + `_lib` | 9.69 s | 6.92 s | 6.40–7.32 s | 22.31 |

All 15 fits had zero divergences. Maximum R-hat across the runs was 1.0302;
several runs exceeded 1.01, so this short benchmark is not a convergence
certificate. The lowest ESS across variables varied substantially by seed.
There is no demonstrated improvement in ESS per second. Across the five
baseline/Nutpie pairs, the largest parameter-mean difference was 2.83 combined
Monte Carlo standard errors. This is a descriptive check, not an equivalence
test. Same seeds did not produce identical draws.

In this model, the full Nutpie path took about 10% less sampling/materialization
time and about 32% less compilation time at the median. The same-graph path was
not faster. Sampling time per logp call, including sampler work, was roughly
100 microseconds for the adapter and same-graph path versus 89 for Nutpie's own
compiled graph. These results do **not** establish a speedup from removing the
JavaScript boundary. The main reason to pursue this port is reduced duplication.

Timing scopes differ: the existing adapter delivers 1,000 live samples and four
Arrow traces (413,600 bytes) to the main thread; the direct path returns Python
raw draws plus three stats and marshals the selected values into NumPy. Nutpie
also expands its own internal value variables. Diagnostics and xarray
conversion are measured separately; runtime download, imports and model
construction are excluded from compile/sample times. The new public
`nutpie.sample` wrapper was validated separately after this benchmark. No
performance claim about a complete Mixlab user interaction follows from it.

[Raw measurements, settings, hashes and aggregates](benchmarks/validation.json)
are checked in. The harness is [benchmarks/benchmark.py](benchmarks/benchmark.py).

## Correctness and packaging

- Original browser probes cover Gaussian moments, deterministic and transformed
  variables, three density/gradient positions per model, mutable float32/int32
  data, independent updated handles, callback ownership after collection,
  runtime-managed memory growth and the MMM model.
- A wheel installed in a fresh Xeus kernel with unmodified PyPA pip returns
  xarray DataTrees from the public `nutpie.sample` entry point. It preserves
  string coordinates, dimensions, simplex values, deterministic variables and
  optional unconstrained groups. Data updates, stats types, ArviZ summaries and
  rejection of unsupported/invalid settings pass. PyArrow/arro3 are not imported.
- Native extension compilation and six existing PyMC/Numba tests passed after
  sharing the result conversion. A separate native transformed/shared-data
  regression also passed. This does not cover the full Stan/JAX suite.
- Formatting/lint checks apply to the Python and Rust changes. The draft PR's
  upstream CI remains separate from these local checks.

## Using the fork as a Mixlab dependency

Use the tagged wheel, not a Git source dependency: the normal source metadata
still describes a native package and pulls in Arrow/storage dependencies.
The experimental release is `wasm-poc-v0.1.0` in `twiecki/nutpie`, with
`nutpie==0.16.12+wasm.1` for CPython 3.13 / Emscripten 4.0.9 / wasm32. Pin the
asset URL **and SHA-256**, and preserve the runtime pin in its manifest. A wheel
tag alone does not guarantee compatibility with another Xeus runtime.

A build pipeline with standard pip can stage this wheel using
`--no-deps --no-compile --only-binary=:all: --platform emscripten_4_0_9_wasm32
--python-version 3.13 --implementation cp --abi cp313 --target <directory>`.
Then serve the installed files with the existing runtime assets. Alternatively,
fetch the wheel from the app's own origin and install it inside Xeus with pip.
The test runtime's bundled pip lacks `pip._vendor`; the installation smoke test
bootstraps the complete, unmodified [PyPA pip 25.2 wheel](https://pypi.org/project/pip/25.2/)
with its pinned hash before the offline install. No package resolver or wheel
installer is reimplemented here. `--no-deps` relies on the prebuilt runtime's
already-installed, pinned scientific stack.

Mixlab's 307 executable/runtime files match this tested runtime byte-for-byte.
Its two additional backport-description JSON files are excluded from that count.
That supports an **experimental Python dependency**, but this wheel does not
implement Mixlab's JavaScript sampler interface. Its current worker integration
expects preparation/cache reuse, progress, live samples, cancellation, binary
results and post-sampling globals. Those need a compatible adapter and tests
before changing the default backend. The Mixlab checkout was not modified.

## Reproduce additional checks

Build both repositories first, extract the pinned runtime, then stage:

```sh
python3 wasm/make_wheel.py
python3 wasm/prepare_site.py /path/to/baseline /path/to/site --benchmarks
```

For the install test, also place the official `pip-25.2-py3-none-any.whl` in the
site directory. Its SHA-256 is
`6d67a2b4e7f14d8b31b8b52648866fa717f45a1eb70e83002f4331d07e953717`.
Start the local server as in the main README. Run the browser harness with
`/?script=benchmark.py` or `/?script=install_smoke.py`. Keep other build/test
loads off the host during the benchmark.

Native regression commands (in an environment with the built extension):

```sh
PYTENSOR_FLAGS='cxx=,blas__ldflags=,numba__cache=False' OPENBLAS_NUM_THREADS=1 \
  python -m pytest tests/test_pymc.py -q -k 'numba and test_pymc_model'
PYTENSOR_FLAGS='cxx=,blas__ldflags=,numba__cache=False' OPENBLAS_NUM_THREADS=1 \
  python wasm/benchmarks/native_regression.py
```
