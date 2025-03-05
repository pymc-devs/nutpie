# Changelog

All notable changes to this project will be documented in this file.

## [0.14.2] - 2025-03-06

### Bug Fixes

- Handle missing flowjax correctly (Adrian Seyboldt)


### Testing

- Mark tests as pymc or stan and select in ci (Adrian Seyboldt)


### Ci

- Use native arm github action runner (Adrian Seyboldt)


## [0.14.1] - 2025-03-05

### Ci

- Update run-on-arch to avoid segfault (Adrian Seyboldt)

- Repare 0.14.1 (Adrian Seyboldt)


## [0.14.0] - 2025-03-05

### Bug Fixes

- Set 'make_initial_point_fn' in 'from_pyfunc' to None by default (#175) (Tomás Capretto)


### Documentation

- Add nutpie website source (Adrian Seyboldt)

- Include frozen cell output in docs (Adrian Seyboldt)


### Features

- Add normalizing flow adaptation (Adrian Seyboldt)


### Miscellaneous Tasks

- Bump actions/attest-build-provenance from 1 to 2 (dependabot[bot])

- Bump softprops/action-gh-release from 1 to 2 (dependabot[bot])

- Bump uraimo/run-on-arch-action from 2 to 3 (dependabot[bot])

- Update pre-commit config (Adrian Seyboldt)


### Ci

- Run python 3.13 in ci (Adrian Seyboldt)

- Skip slow test on ci if emulating architecture (Adrian Seyboldt)


## [0.13.4] - 2025-02-18

### Bug Fixes

- Add lock for pymc init point func (Adrian Seyboldt)


### Ci

- Make sure all python versions are available in the builds (Adrian Seyboldt)

- Skip python 3.13 for now (Adrian Seyboldt)


## [0.13.3] - 2025-02-12

### Bug Fixes

- Use arrow list with i64 offsets to store trace (Adrian Seyboldt)

- Use i64 offsets in numba backend (Adrian Seyboldt)

- Avoid numpy compatibility warning (Adrian Seyboldt)

- Specify that we currently don't support py313 due to pyo3 (Adrian Seyboldt)


### Features

- Add support for pymc sampler initialization (jessegrabowski)

- Use support_point as default init for pymc (Adrian Seyboldt)

- Add option not to store some deterministics (Adrian Seyboldt)

- Add option to freeze pymc models (Adrian Seyboldt)


### Miscellaneous Tasks

- Bump uraimo/run-on-arch-action from 2.7.2 to 2.8.1 (dependabot[bot])

- Specify version as dynamic in pyproject (Adrian Seyboldt)

- Update bridgestan (Adrian Seyboldt)

- Update pre-commit versions (Adrian Seyboldt)


### Styling

- Reformat some code (Adrian Seyboldt)


### Build

- Bump some dependency versions (Adrian Seyboldt)


### Ci

- Use ubuntu_latest on aarch64 (Adrian Seyboldt)

- Update CI script using maturin (Adrian Seyboldt)


## [0.13.2] - 2024-07-26

### Features

- Support float32 settings in pytensor (Adrian Seyboldt)


### Miscellaneous Tasks

- Update dependencies (Adrian Seyboldt)


## [0.13.1] - 2024-07-09

### Bug Fixes

- Fix jax backend with non-identifier variable names (Adrian Seyboldt)


### Miscellaneous Tasks

- Update dependencies (Adrian Seyboldt)


## [0.13.0] - 2024-07-05

### Documentation

- Document low-rank mass matrix parameters (Adrian Seyboldt)


### Features

- Add low rank modified mass matrix adaptation (Adrian Seyboldt)


### Miscellaneous Tasks

- Remove releases from changelog (Adrian Seyboldt)


## [0.12.0] - 2024-06-29

### Features

- Add pyfunc backend (Adrian Seyboldt)

- Add python code for pyfunc backend (Adrian Seyboldt)

- Add gradient_backend argument for pymc models (Adrian Seyboldt)


### Miscellaneous Tasks

- Bump version number (Adrian Seyboldt)


### Styling

- Fix pre-commit issues (Adrian Seyboldt)


### Testing

- Add tests for jax backend (Adrian Seyboldt)


### Build

- Add jax as optional dependency (Adrian Seyboldt)


## [0.11.1] - 2024-06-16

### Bug Fixes

- Fix random variables with missing values in pymc deterministics (Adrian Seyboldt)


### Features

- Add progress bar on terminal (Adrian Seyboldt)


## [0.11.0] - 2024-05-29

### Bug Fixes

- Use clone_replace instead of graph_replace (Adrian Seyboldt)

- Allow shared vars to differ in expand and logp (Adrian Seyboldt)


### Features

- Add option to use draw base mass matrix estimate (Adrian Seyboldt)

- Report detailed progress (Adrian Seyboldt)

- Show the number of draws in progress overview (Adrian Seyboldt)


### Miscellaneous Tasks

- Bump KyleMayes/install-llvm-action from 1 to 2 (dependabot[bot])

- Bump uraimo/run-on-arch-action from 2.7.1 to 2.7.2 (dependabot[bot])

- Update dependencies (Adrian Seyboldt)

- Update python dependencies (Adrian Seyboldt)


### Refactor

- Move threaded sampling to nuts-rs (Adrian Seyboldt)

- Specify callback rate (Adrian Seyboldt)

- Switch to arrow-rs (Adrian Seyboldt)


### Styling

- Fix formatting and clippy (Adrian Seyboldt)


### Testing

- Fix incorrect error type in test (Adrian Seyboldt)


### Ci

- Fix uploads of releases (Adrian Seyboldt)

- Fix architectures in CI (Adrian Seyboldt)


## [0.10.0] - 2024-03-20

### Documentation

- Mention non-blocking sampling in readme (Adrian Seyboldt)


### Features

- Allow sampling in the backgound (Adrian Seyboldt)

- Implement check if background sampling is complete (Adrian Seyboldt)

- Implement pausing and unpausing of samplers (Adrian Seyboldt)

- Filter warnings and compile through pymc (Adrian Seyboldt)


### Miscellaneous Tasks

- Bump actions/setup-python from 4 to 5 (dependabot[bot])

- Bump uraimo/run-on-arch-action from 2.5.0 to 2.7.1 (dependabot[bot])

- Bump actions/checkout from 3 to 4 (dependabot[bot])

- Bump actions/upload-artifact from 3 to 4 (dependabot[bot])

- Bump the cargo group across 1 directory with 2 updates (dependabot[bot])

- Some major version bumps in rust deps (Adrian Seyboldt)

- Bump dependency versions (Adrian Seyboldt)

- Bump version (Adrian Seyboldt)

- Update changelog (Adrian Seyboldt)


### Performance

- Set the number of parallel chains dynamically (Adrian Seyboldt)


## [0.9.2] - 2024-02-19

### Bug Fixes

- Allow dims with only length specified (Adrian Seyboldt)


### Documentation

- Update suggested mamba commands in README (#70) (Ben Mares)

- Fix README typo bridgestan→nutpie (#69) (Ben Mares)


### Features

- Handle missing libraries more robustly (#72) (Ben Mares)


### Miscellaneous Tasks

- Bump actions/download-artifact from 3 to 4 (dependabot[bot])


### Ci

- Make sure the local nutpie is installed (Adrian Seyboldt)

- Install local nutpie package in all jobs (Adrian Seyboldt)


## [0.9.0] - 2023-09-12

### Bug Fixes

- Better error context for init point errors (Adrian Seyboldt)


### Features

- Improve error message by providing context (Adrian Seyboldt)

- Use standard normal to initialize chains (Adrian Seyboldt)


### Miscellaneous Tasks

- Update deps (Adrian Seyboldt)

- Rename stan model transpose function (Adrian Seyboldt)

- Update nutpie (Adrian Seyboldt)


### Styling

- Fix formatting (Adrian Seyboldt)


## [0.8.0] - 2023-08-18

### Bug Fixes

- Initialize points in uniform(-2, 2) (Adrian Seyboldt)

- Multidimensional stan variables were stored in incorrect order (Adrian Seyboldt)

- Fix with_coords for stan models (Adrian Seyboldt)


### Miscellaneous Tasks

- Update deps (Adrian Seyboldt)

- Update deps (Adrian Seyboldt)

- Bump version (Adrian Seyboldt)

- Update deps (Adrian Seyboldt)


## [0.7.0] - 2023-07-21

### Bug Fixes

- Check logp value in stan interface (Adrian Seyboldt)

- Make max_energy_error writable (Adrian Seyboldt)

- Export names of unconstrained parameters (Adrian Seyboldt)

- Fix return values of logp benchmark function (Adrian Seyboldt)


### Features

- Export more details of divergences (Adrian Seyboldt)

- Add extra_stanc_args argument to compile_stan_model (Chris Fonnesbeck)


### Miscellaneous Tasks

- Update dependencies (Adrian Seyboldt)

- Add changelog (Adrian Seyboldt)

- Bump version (Adrian Seyboldt)


### Refactor

- Hide private rust module (Adrian Seyboldt)


## [0.6.0] - 2023-07-03

### Features

- Allow to update dims and coords in stan model (Adrian Seyboldt)


### Miscellaneous Tasks

- Bump version (Adrian Seyboldt)


<!-- generated by git-cliff -->
