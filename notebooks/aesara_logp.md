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

```python
%env RAYON_NUM_THREADS=12
%env RUST_BACKTRACE=1
```

```python
import aesara
import aesara.tensor as at
import pymc as pm
import numpy as np
import nuts_py
import numba
from math import prod
import fastprogress
import arviz
import pandas as pd
import threadpoolctl
import nuts_py.convert
from scipy import optimize, stats
import matplotlib.pyplot as plt
import seaborn as sns
```

```python
data = pd.read_csv(pm.get_data("radon.csv"))
county_names = data.county.unique()

data["log_radon"] = data["log_radon"].astype(np.float64)
county_idx, counties = pd.factorize(data.county)
coords = {"county": counties, "obs_id": np.arange(len(county_idx))}
```

```python
with pm.Model(coords=coords, check_bounds=False) as model:
    # Intercepts, non-centered
    mu_a = pm.Normal("mu_a", mu=0.0, sigma=10)
    sigma_a = pm.HalfNormal("sigma_a", 1.0)
    a = pm.Normal("a", dims="county") * sigma_a + mu_a

    # Slopes, non-centered
    
    mu_b = pm.Normal("mu_b", mu=0.0, sigma=2.)
    sigma_b = pm.HalfNormal("sigma_b", 1.0)
    b = pm.Normal("b", dims="county") * sigma_b + mu_b

    eps = pm.HalfNormal("eps", 1.5)

    radon_est = a[county_idx] + b[county_idx] * data.floor.values
    #radon_est = a[0] + b[0] * data.floor.values

    radon_like = pm.Normal(
        "radon_like", mu=radon_est, sigma=eps, observed=data.log_radon.values,
        dims="obs_id"
    )
```

```python
n_chains = 10
```

```python
%%time
with threadpoolctl.threadpool_limits(1):
    with model:
        trace_py = pm.sample(
            init="jitter+adapt_diag_grad",
            draws=1000,
            chains=n_chains,
            cores=10,
            idata_kwargs={"log_likelihood": False},
            compute_convergence_checks=False,
            target_accept=0.8,
            #max_treedepth=6,
            discard_tuned_samples=False,
        )
```

```python
%%time
with threadpoolctl.threadpool_limits(1):
    with model:
        trace_py2 = pm.sample(
            init="jitter+adapt_diag",
            draws=1000,
            chains=n_chains,
            cores=10,
            idata_kwargs={"log_likelihood": False},
            compute_convergence_checks=False,
            target_accept=0.8,
            #max_treedepth=10,
            discard_tuned_samples=False,
        )
```

```python
from aesara import config
```

```python
config.numba__error_model = "numpy"
config.numba__nogil = True
config.numba__cache = False
config.numba__boundscheck = True
config.numba__fastmath = True
config.numba__parallel = False
```

```python
kwargs = {
    "error_model": config.numba__error_model,
    "nogil": config.numba__nogil,
    "boundscheck": config.numba__boundscheck,
    "fastmath": config.numba__fastmath,
    "parallel": config.numba__parallel,
}
```

```python
import aeppl
from aeppl.logprob import CheckParameterValue
import aesara.link.numba.dispatch

@aesara.link.numba.dispatch.numba_funcify.register(CheckParameterValue)
def numba_functify_CheckParameterValue(op, **kwargs):
    @aesara.link.numba.dispatch.basic.numba_njit
    def check(value, *conditions):
        return value
    
    return check

@aesara.link.numba.dispatch.numba_funcify.register(aesara.tensor.subtensor.AdvancedIncSubtensor1)
def numba_funcify_IncSubtensor(op, node, **kwargs):

    def incsubtensor_fn(z, vals, idxs):
        z = z.copy()
        for idx, val in zip(idxs, vals):
            z[idx] += val
        return z

    return aesara.link.numba.dispatch.basic.numba_njit(incsubtensor_fn)
```

```python
%%time
n_dim, logp_func, expanding_function, shape_info = nuts_py.convert.make_functions(model)
logp_func = numba.njit(**kwargs)(logp_func)
logp_numba_raw, c_sig = nuts_py.convert.make_c_logp_func(n_dim, logp_func)
logp_numba = numba.cfunc(c_sig, **kwargs)(logp_numba_raw)
```

```python
x = np.random.randn(n_dim)
```

```python
#from numba.pycc import CC

#cc = CC('logp_module')
#cc.target_cpu = "host"
#logp_func(np.random.randn(n_dim))
#cc.export("logp_grad", logp_func.signatures[0])(logp_func)
#cc.compile()

#import logp_module
#%timeit logp_module.logp_grad(x)
```

```python
optimize.check_grad(lambda x: logp_func(x)[0], lambda x: logp_func(x)[1], x)
```

```python
n_draws = 1000
seed = 40
```

```python
def make_user_data():
    return 0

settings = nuts_py.lib.PySamplerArgs()
settings.num_tune = 1000
settings.target_accept = 0.8
settings.save_mass_matrix = True
settings.discard_window = 100

x = np.random.default_rng(42).normal(size=n_dim)
```

```python
%%time
draws = []
tune = []
stats = []
for i in range(1):
    sampler = nuts_py.lib.PySampler(logp_numba.address, make_user_data, x, n_dim, settings, draws=n_draws + settings.num_tune, chain=0, seed=seed + i)
    for (draw, stat) in sampler:
        draws.append(draw)
        stats.append(stat)
```

```python
plt.plot(np.log([stat.as_dict()["step_size_bar"] for stat in stats])[:60])
```

```python
mass_matrix = np.array([stat.as_dict()["current_mass_matrix_inv_diag"] for stat in stats])[:500, :]
plt.plot(np.log(mass_matrix)[:500]);
```

```python
n_chains = 10
```

```python
%%time
for _ in range(1):
    with threadpoolctl.threadpool_limits(1):
        trace_rust = nuts_py.convert.sample(
            model,
            N=n_dim,
            logp_numba=logp_numba,
            expanding_function=expanding_function,
            shape_info=shape_info,
            max_treedepth=10,
            n_tune=1000,
            n_draws=1000,
            n_chains=n_chains,
            target_accept=0.8,
            early_target_accept=0.2,
            seed=42,
            variance_decay=0.01,
            #window_switch_freq=20,
            #early_variance_decay=0.5,
        )
```

```python
trace_rust.sample_stats.diverging.sum().values
```

```python
np.log(trace_rust.warmup_sample_stats.step_size_bar.isel(draw=slice(0, 1000))).plot(x="draw", hue="chain", add_legend=False);
```

```python
trace_rust.warmup_sample_stats.mean_tree_accept.rolling(draw=20).mean().plot.line(x="draw", add_legend=False);
```

```python
trace_rust.warmup_posterior.eps.isel(draw=slice(0, 100)).plot(x="draw", hue="chain", add_legend=False);
```

```python
plt.plot((trace_rust.warmup_sample_stats.n_steps).isel(draw=slice(0, 1000)).cumsum("draw").T, trace_rust.warmup_sample_stats.energy.isel(draw=slice(0, 1000)).T);
plt.xlim(0, 1000)
```

```python
plt.plot((trace_py.warmup_sample_stats.n_steps).isel(draw=slice(0, 1000)).cumsum("draw").T, trace_py.warmup_sample_stats.energy.isel(draw=slice(0, 1000)).T);
plt.xlim(0, 1000)
```

```python
plt.plot((trace_py2.warmup_sample_stats.n_steps).isel(draw=slice(0, 1000)).cumsum("draw").T, trace_py2.warmup_sample_stats.energy.isel(draw=slice(0, 1000)).T);
plt.xlim(0, 1000)
```

```python
plt.plot((2 ** trace_rust.warmup_sample_stats.depth).isel(draw=slice(0, 100)).cumsum("draw").T, np.log(trace_rust.warmup_sample_stats.step_size_bar.isel(draw=slice(0, 100))).T);
```

```python
trace_rust.warmup_sample_stats.logp.isel(draw=slice(0, 100)).plot(x="draw", hue="chain", add_legend=False);
```

```python
trace_rust.warmup_posterior.eps.isel(draw=slice(0, 100)).plot(x="draw", hue="chain", add_legend=False);
```

```python
np.log(trace_py.warmup_sample_stats.step_size_bar).isel(draw=slice(None, 500)).plot(x="draw", hue="chain", add_legend=False);
```

```python
np.log(trace_py2.warmup_sample_stats.step_size_bar).isel(draw=slice(None, 500)).plot(x="draw", hue="chain", add_legend=False);
```

```python
ess_py = arviz.ess(trace_py)
ess_py2 = arviz.ess(trace_py2)
ess_rust = arviz.ess(trace_rust)
```

```python
(trace_py.warmup_sample_stats.n_steps).sum() / n_chains
```

```python
(trace_py2.warmup_sample_stats.n_steps).sum() / n_chains
```

```python
(trace_rust.warmup_sample_stats.n_steps).sum() / n_chains
```

```python
ess_rust.min()
```

```python
ess_py.min()
```

```python
ess_py2.min()
```

```python

```
