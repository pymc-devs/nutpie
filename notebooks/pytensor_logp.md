---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: pymc4-dev
    language: python
    name: pymc4-dev
---

# Usage example of nutpie

```python
# We can control the number cores that are used by an environment variable:
%env RAYON_NUM_THREADS=12
```

```python
import pytensor
import pytensor.tensor as pt
import pymc as pm
import numpy as np
import nutpie
import arviz
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

## The dataset

We use the well known radon dataset in this notebook.

```python
data = pd.read_csv(pm.get_data("radon.csv"))
data["log_radon"] = data["log_radon"].astype(np.float64)
county_idx, counties = pd.factorize(data.county)
coords = {"county": counties, "obs_id": np.arange(len(county_idx))}
```

```python
sns.catplot(
    data=data,
    x="floor",
    y="log_radon",
)
```

## Use as a sampler for pymc

```python
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

```python
%%time
# The compilation time is pretty bad right now, I think this can be improved a lot though
compiled_model = nutpie.compile_pymc_model(pymc_model)
```

```python
%%time
trace_pymc = nutpie.sample(compiled_model, chains=10)
```

```python
sns.catplot(
    data=(
        (trace_pymc.posterior.county_floor_effect + trace_pymc.posterior.floor_effect)
        .isel(county=slice(0, 5))
        .to_dataframe("total_county_floor_effect")
        .reset_index()
    ),
    x="total_county_floor_effect",
    y="county",
    kind="violin",
    orient="h",
)
plt.axvline(0, color="grey", alpha=0.5, zorder=-100)
```

## Use nutpie as a sampling backend for stan

```python
%%file radon_model.stan
data {
    int<lower=0> n_counties;
    int<lower=0> n_observed;
    array[n_observed] int<lower=1,upper=n_counties> county_idx;
    vector[n_observed] is_floor;
    vector[n_observed] log_radon;
}
parameters {
    real intercept;

    vector[n_counties] county_raw;
    real<lower=0> county_sd;

    real floor_effect;

    vector[n_counties] county_floor_raw;
    real<lower=0> county_floor_sd;

    real<lower=0> sigma;
}
transformed parameters {
    vector[n_counties] county_effect;
    vector[n_counties] county_floor_effect;
    vector[n_observed] mu;

    county_effect = county_sd * county_raw;
    county_floor_effect = county_floor_sd * county_floor_raw;

    mu = (
        intercept
        + county_effect[county_idx]
        + floor_effect * is_floor
        + county_floor_effect[county_idx] .* is_floor
    );
}
model {
    intercept ~ normal(0, 10);

    county_raw ~ normal(0, 1);
    county_sd ~ normal(0, 1);

    floor_effect ~ normal(0, 2);

    county_floor_raw ~ normal(0, 1);
    county_floor_sd ~ normal(0, 1);

    sigma ~ normal(0, 1.5);

    log_radon ~ normal(mu, sigma);
}
```

```python
data_stan = {
    "n_counties": len(counties),
    "n_observed": len(data),
    "county_idx": county_idx + 1,
    "is_floor": data.floor.values,
    "log_radon": data.log_radon.values,
}

coords_stan = {
    "county": counties,
}

dims_stan = {
    "county_raw": ("county",),
    "county_floor_raw": ("county",),
    "county_effect": ("county",),
    "county_floor_effect": ("county",),
    "mu": ("observation",),
}
```

```python
%%time
stan_model = nutpie.compile_stan_model(
    data_stan,
    filename="radon_model.stan",
    coords=coords_stan,
    dims=dims_stan,
    cache=False
)
```

```python
%%time
trace_stan = nutpie.sample(stan_model, chains=10)
```

## Comparison with pystan

```python
import stan
import nest_asyncio

nest_asyncio.apply()
```

```python
%%time
with open("radon_model.stan", "r") as file:
    model = stan.build(file.read(), data=data_stan)
```

```python
%%time
trace_pystan = model.sample(num_chains=10, save_warmup=True)
```

```python
trace_pystan = arviz.from_pystan(trace_pystan, save_warmup=True)
```

## Comparison to the pymc sampler

```python
%%time
with pymc_model:
    trace_py = pm.sample(
        init="jitter+adapt_diag_grad",
        draws=1000,
        chains=10,
        cores=10,
        idata_kwargs={"log_likelihood": False},
        compute_convergence_checks=False,
        target_accept=0.8,
        discard_tuned_samples=False,
    )
```

## Early convergance speed

```python
plt.plot((trace_pymc.warmup_sample_stats.n_steps).isel(draw=slice(0, 1000)).cumsum("draw").T, np.log(trace_pymc.warmup_sample_stats.energy.isel(draw=slice(0, 1000)).T));
plt.xlim(0, 10000)
plt.ylabel("log-energy")
plt.xlabel("gradient evaluations");
```

```python
trace_cmdstan = arviz.from_cmdstan("output_*.csv", save_warmup=True)
```

```python
plt.plot((trace_cmdstan.warmup_sample_stats.n_steps).isel(draw=slice(0, 1000)).cumsum("draw").T, np.log(trace_cmdstan.warmup_sample_stats.energy.isel(draw=slice(0, 1000)).T));
plt.xlim(0, 10000)
plt.ylabel("log-energy")
plt.xlabel("gradient evaluations");
```

The new implementation only use about a third of gradient evaluations during tuning

```python
trace_cmdstan.warmup_sample_stats.n_steps.sum()
```

```python
trace_stan.warmup_sample_stats.n_steps.sum()
```

## Comparison to cmdstan


Run on the commandline:
```
env STAN_THREADS=1 cmdstan_model radon_model.stan
```

```python
import json
```

```python
stan.common.simdjson
```

```python
type({name: int(val) if isinstance(val, int) else list(val) for name, val in data_stan.items()}["county_idx"][0])
```

```python
data_json = {}
for name, val in data_stan.items():
    if isinstance(val, int):
        data_json[name] = int(val)
        continue

    if val.dtype == np.int64:
        data_json[name] = list(int(x) for x in val)
        continue

    data_json[name] = list(val)

with open("radon.json", "w") as file:
    json.dump(data_json, file)
```

```python
%%time
out = !./radon_model sample num_chains=10 save_warmup=1 data file=radon.json num_threads=10
```

```python
trace_cmdstan = arviz.from_cmdstan("output_*.csv", save_warmup=True)
```

## Gradient evals per effective sample

nutpie uses fewer gradient evaluations per effective sample in this model.

```python
trace_cmdstan.sample_stats.n_steps.sum() / arviz.ess(trace_cmdstan).min()
```

```python
trace_stan.sample_stats.n_steps.sum() / arviz.ess(trace_stan).min()
```
