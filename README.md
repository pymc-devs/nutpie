# nutpie: A fast sampler for bayesian posteriors

## Installation

nutpie can be installed using conda or mamba from conda-forge with

```
mamba install -c conda-forge nutpie pymc
```

Or using pip:

```
pip install nutpie
```

To install it from source, install a rust compiler and maturin and then

```
maturin develop --release
```

If you want to use the nightly simd implementation for some of the math functions,
switch to rust nightly and then install with the `simd_support` feature in then
nutpie directory:

```
rustup override set nightly
maturin develop --release --features=simd_support
```

## Usage with PyMC

First, PyMC and numba need to be installed, for example using

```
mamba install pymc numba
```

We need to create a model:

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
    raw = pm.ZeroSumNormal("county_raw", dims="county")
    sd = pm.HalfNormal("county_sd")
    county_effect = pm.Deterministic("county_effect", raw * sd, dims="county")

    # Global floor effect
    floor_effect = pm.Normal("floor_effect", sigma=2)

    # County:floor interaction
    raw = pm.ZeroSumNormal("county_floor_raw", dims="county")
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
trace_pymc = nutpie.sample(compiled_model)
```

`trace_pymc` now contains an arviz `InferenceData` object, including sampling
statistics and the posterior of the variables defined above.

## Usage with Stan

In order to sample from stan model, `bridgestan` needs to be installed.
A pip package is available, but right now this can not be installed using conda.

```
pip install bridgestan
```

When we install nutpie with pip, we can also specify that we want optional
dependencies for Stan models using

```
pip install 'bridgestan[stan]'
```

In addition, a C++ compiler needs to be available. For details see
[the stan docs](https://mc-stan.org/docs/cmdstan-guide/cmdstan-installation.html#cpp-toolchain).

We can then compile a Stan model, and sample using nutpie:

```
import nutpie

code = """
data {
    real mu;
}
parameters {
    real x;
}
model {
    x ~ normal(mu, 1);
}
"""

compiled = nutpie.compile_stan_model(code=code)
# Provide data
compiled = compiled.with_data(mu=3.)
trace = nutpie.sample(compiled)
```

## Advantages

nutpie uses `nuts-rs`, a library written in rust, that implements NUTS as in
pymc and stan, but with a slightly different mass matrix tuning method as
those. It often produces a higher effective sample size per gradient
evaluation, and tends to converge faster and with fewer gradient evaluation.
