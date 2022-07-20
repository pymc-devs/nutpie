# nutpie: A fast sampler for bayesian posteriors

## Installation

nutpie can be installed using conda or mamba from conda-forge with

```
mamba install -c conda-forge nutpie pymc
```

To install it from source, install a rust compiler (eg using rustup) and run

```
maturin develop --release
```

If you want to use the nightly simd implementation for some of the math functions,
switch to rust nightly and then install with the `simd_support` feature in the nutpie dir:

```
rustup override set nightly
maturin develop --release --features=simd_support
```

## Usage

First, we need to create a model, for example using pymc:

```python
import pymc as pm
import numpy as np
import nutpie
import pandas as pd
import seaborn as sns

# Load the radon dataset
data = pd.read_csv(pm.get_data("radon.csv"))
data["log_radon"] = data["log_radon"].astype(np.float64)
county_idx, counties = pd.factorize(data.county)
coords = {"county": counties, "obs_id": np.arange(len(county_idx))}

# Create a simple hierarchical model for the radon dataset
with pm.Model(coords=coords, check_bounds=False) as pymc_model:
    intercept = pm.Normal("intercept", sigma=10)

    # County effects
    # TODO should be a CenteredNormal
    raw = pm.Normal("county_raw", dims="county")
    sd = pm.HalfNormal("county_sd")
    county_effect = pm.Deterministic("county_effect", raw * sd, dims="county")

    # Global floor effect
    floor_effect = pm.Normal("floor_effect", sigma=2)

    # County:floor interaction
    # Should also be a CenteredNormal
    raw = pm.Normal("county_floor_raw", dims="county")
    sd = pm.HalfNormal("county_floor_sd")
    county_floor_effect = pm.Deterministic(
        "county_floor_effect", raw * sd, dims="county"
    )

    mu = (
        intercept
        + county_effect[county_idx]
        + floor_effect * data.floor.values
        + county_floor_effect[county_idx] * data.floor.values
    )

    sigma = pm.HalfNormal("sigma", sigma=1.5)
    pm.Normal(
        "log_radon", mu=mu, sigma=sigma, observed=data.log_radon.values, dims="obs_id"
    )
```

We then compile this model and sample form the posterior:

```python
compiled_model = nutpie.compile_pymc_model(pymc_model)
trace_pymc = nutpie.sample(compiled_model, chains=10)
```

`trace_pymc` now contains an arviz `InferenceData` object, including sampling
statistics and the posterior of the variables defined above.

For more details, see the example notebook `aesara_logp`

nutpie can also sample from stan models, it currently needs a patched version of httpstan do so so however.
The required version can be found [here](https://github.com/stan-dev/httpstan/pull/600).
Make sure to follow the development
[installation instructions for httpstan](https://httpstan.readthedocs.io/en/latest/installation.html#installation-from-source).

## Advantages

nutpie uses `nuts-rs`, a library written in rust, that implements NUTS as in
pymc and stan, but with a slightly different mass matrix tuning method as
those. It often produces a higher effective sample size per gradient
evaluation, and tends to converge faster and with fewer gradient evaluation.

From the benchmarks I did, it seems to be the fastest CPU based sampler I could
find, outperforming cmdstan and numpyro.

Unforunately performance on pymc models is currently somewhat limited by an
[issue in numba](https://github.com/numba/numba/issues/8156), which hopefully
will be fixed soon. Without the patch mentioned in the issue the model above
samples in about 2s on my machine, with the patch it finished is about 700ms.
