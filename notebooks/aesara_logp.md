---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.4
  kernelspec:
    display_name: pymc4-dev
    language: python
    name: pymc4-dev
---

```python
%env OMP_NUM_THREADS=1
%env RAYON_NUM_THREADS=10
```

```python
import httpstan
```

```python
import stan
import nest_asyncio
nest_asyncio.apply()
```

```python
model = stan.build(
    """
    parameters {
        vector[5000] a;
    }
    model {
        a ~ normal(0, 1);
    }
    """,
    random_seed=6,
)
```

```python
import numpy as np
```

```python
%%time
trace = model.sample(num_samples=1000, max_depth=10, delta=0.8, num_chains=16)
```

```python
draws = trace["a"][0, :]
```

```python
import seaborn as sns
```

```python
sns.displot(draws)
```

```python
import scipy.stats
```

```python
scipy.stats.ks_1samp(draws, scipy.stats.norm(scale=1).cdf)
```

```python
import arviz
```

```python
tr = arviz.from_pystan(trace)
```

```python
(tr.posterior.diff('draw').a == 0).mean()
```

```python
tr.sample_stats.n_steps
```

```python
tr.sample_stats.n_steps.mean()
```

```python
arviz.ess(trace)
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
import stan
```

```python
def compute_shapes(model):
    point = pm.model.make_initial_point_fn(model=model, return_transformed=True)(0)
    
    value_vars = model.value_vars.copy()
    trace_vars = {name: var for (name, var) in model.named_vars.items() if var not in model.observed_RVs + model.potentials}

    shape_func = aesara.function(
        inputs=[],
        outputs=[var.shape for var in trace_vars.values()],
        givens=(
            [(obs, obs.tag.observations) for obs in model.observed_RVs]
            + [(trace_vars[name], point[name]) for name in trace_vars.keys() if name in point]
        ),
        mode=aesara.compile.mode.FAST_COMPILE,
        on_unused_input="ignore"
    )
    return {name: shape for name, shape in zip(trace_vars.keys(), shape_func())}
    
def make_functions(model):
    shapes = compute_shapes(model)
    
    # Make logp_dlogp_function
    joined = at.dvector("__joined_variables")

    value_vars = [model.rvs_to_values[var] for var in model.free_RVs]

    logp = model.logpt
    grads = at.grad(logp, value_vars)
    grad = at.concatenate([grad.ravel() for grad in grads])

    count = 0
    joined_slices = []
    joined_shapes = []
    joined_names = []

    symbolic_sliced = []
    for var in model.free_RVs:
        value_var = model.rvs_to_values[var]

        joined_names.append(value_var.name)
        shape = shapes[value_var.name]
        joined_shapes.append(shape)
        length = prod(shape)
        slice_val = slice(count, count + length)
        joined_slices.append(slice_val)
        symbolic_sliced.append((value_var, joined[slice_val].reshape(shape)))
        count += length
    
    num_free_vars = count

    # We should avoid compiling the function, and optimize only
    func = aesara.function((joined,), (logp, grad), givens=symbolic_sliced, mode=aesara.compile.NUMBA)
    fgraph = func.maker.fgraph
    func = aesara.link.numba.dispatch.numba_funcify(fgraph)
    logp_func = numba.njit(func)
    
    # Make function that computes remaining variables for the trace
    trace_vars = {name: var for (name, var) in model.named_vars.items() if var not in model.observed_RVs + model.potentials}
    remaining_names = [name for name in trace_vars if name not in joined_names]
    remaining_rvs = [var for var in model.unobserved_value_vars if var.name not in joined_names]

    all_names = joined_names + remaining_rvs

    all_names = joined_names.copy()
    all_slices = joined_slices.copy()
    all_shapes = joined_shapes.copy()

    for var in remaining_rvs:
        all_names.append(var.name)
        shape = shapes[var.name]
        all_shapes.append(shape)
        length = prod(shape)
        all_slices.append(slice(count, count + length))
        count += length

    allvars = at.concatenate([joined, *[var.ravel() for var in remaining_rvs]])
    func = aesara.function((joined,), (allvars,), givens=symbolic_sliced, mode=aesara.compile.NUMBA)
    fgraph = func.maker.fgraph
    func = aesara.link.numba.dispatch.numba_funcify(fgraph)
    expanding_function = numba.njit(func, fastmath=True, error_model="numpy")
        
    return num_free_vars, logp_func, expanding_function, (all_names, all_slices, all_shapes)

def make_c_logp_func(N, logp_func):
    c_sig = numba.types.int64(
    numba.types.uint64,
    numba.types.CPointer(numba.types.double),
    numba.types.CPointer(numba.types.double),
    numba.types.CPointer(numba.types.double),
    numba.types.voidptr,
    )
    
    def rerun_inner(x):
        try:
            logp_func(x)
        except Exception as e:
            print(e)
    
    @numba.njit()
    def rerun(x):
        with numba.objmode():
            rerun_inner(x)
            
    @numba.cfunc(c_sig, nopython=True, fastmath=True, error_model="numpy")
    def logp_numba(dim, x_, out_, logp_, user_data_):
        try:
            x = numba.carray(x_, (N,))
            out = numba.carray(out_, (N,))
            logp = numba.carray(logp_, ())
            
            # TODO
            logp_val, grad = logp_func(x)
            logp[()] = logp_val
            out[...] = grad
            
            #logp_val = 0
            #for i in range(N):
            #    out[i] = -x[i]
            #    logp_val -= x[i] * x[i] / 2
            
            #logp[()] = logp_val

            if not np.all(np.isfinite(out)):
                return 2
            if not np.isfinite(logp_val):
                return 3
            return 0
        except Exception:
            #x = numba.carray(x_, (N,))
            #rerun(x)
            return 1

    logp_numba.compile()
    return logp_numba

def sample(*, N, logp_numba, expanding_function, shape_info, n_tune, n_draws, n_chains, seed=42, max_treedepth=10, target_accept=0.8):
    def make_user_data():
        return 0

    settings = nuts_py.lib.SamplerArgs()
    settings.num_tune = n_tune
    settings.maxdepth = max_treedepth
    settings.target_accept = target_accept
    x = np.random.randn(N)
    sampler = nuts_py.lib.ParallelSampler(logp_numba.address, make_user_data, N, x, settings, n_chains=n_chains, n_draws=n_draws, seed=seed, n_try_init=10)

    try:
        n_expanded = len(expanding_function(x)[0])
        draws = np.full((n_chains, n_draws + n_tune, n_expanded), np.nan)
        infos = []
        for draw, info in fastprogress.progress_bar(sampler, total=n_chains * (n_draws + n_tune)):
            infos.append(info)
            draws[info.chain, info.draw, :] = expanding_function(draw)[0]
    finally:
        sampler.finalize()
        
    trace_dict = {}
    trace_dict_tune = {}
    for name, slice_, shape in zip(*shape_info):
        trace_dict_tune[name] = draws[:, :n_tune, slice_].reshape((n_chains, n_tune) + tuple(shape))
        trace_dict[name] = draws[:, n_tune:, slice_].reshape((n_chains, n_draws) + tuple(shape))
    
    stat_dtypes = {
        "idx_in_trajectory": np.int64,
        "is_diverging": bool,
        "mean_acceptance_rate": np.float64,
        "depth": np.int64,
        "step_size": np.float64,
        "step_size_bar": np.float64,
        "tree_size": np.int64,
        "first_diag_mass_matrix": np.float64,
    }
    
    # This is actually relatively slow, we should be able to speed this up
    stats = {}
    stats_tune = {}
    for name, dtype in stat_dtypes.items():
        stats[name] = np.zeros((n_chains, n_draws), dtype=dtype)
        stats_tune[name] = np.zeros((n_chains, n_tune), dtype=dtype)

    for info in infos:
        if info.draw < n_tune:
            out = stats_tune
            draw = info.draw
        else:
            out = stats
            draw = info.draw - n_tune
        for name in stat_dtypes:
            out[name][info.chain, draw] = getattr(info, name)
    
    return arviz.from_dict(
        posterior=trace_dict,
        warmup_posterior=trace_dict_tune,
        save_warmup=True,
        coords=model.coords,
        dims=model._RV_dims,
        sample_stats=stats,
        warmup_sample_stats=stats_tune,
    )
```

```python
N = 10
idx = np.arange(N, dtype=int)
np.random.shuffle(idx)
with pm.Model() as model:
    #a = pm.Normal("a", shape=N, sigma=1.)
    #b = pm.Normal("b", shape=N, sigma=2000.)
    #pm.Deterministic("b", 2 * a)
    log_sd = pm.Gamma("sd", shape=N, sigma=1000, mu=10_000)
    pm.Normal("a", sd=1, shape=N)
    #pm.Gamma("a", mu=1, sd=0.1, shape=N)
    #pm.SkewNormal("a", alpha=np.array([3., 3.]), shape=2)
    #pm.Mixture("a", w=[0.5, 0.5])
    #pm.Normal("y", observed=3.)
    #pm.Normal("y", mu=a[idx][idx][idx][idx], observed=5 + np.random.randn(N))
```

```python
n_chains = 8
```

```python
%%time
with model:
    trace_py = pm.sample(
        init="jitter+adapt_diag_grad",
        draws=1000,
        chains=n_chains,
        cores=10,
        idata_kwargs={"log_likelihood": False},
        compute_convergence_checks=False,
        target_accept=0.8,
        max_treedepth=10,
        discard_tuned_samples=False,
    )
```

```python
%%time
n_dim, logp_func, expanding_function, shape_info = make_functions(model)
logp_numba = make_c_logp_func(n_dim, logp_func)
```

```python
x = np.random.randn(n_dim)
```

```python
logp_func(x)
```

```python
func_orig = model.logp_dlogp_function()
func_orig.set_extra_values({})
```

```python
#func_orig._aesara_function(x[shape_info[1][0]])
```

```python
#logp_func(x)
```

```python
%%time
trace_rust = sample(
    N=n_dim,
    logp_numba=logp_numba,
    expanding_function=expanding_function,
    shape_info=shape_info,
    max_treedepth=100,
    n_tune=1000,
    n_draws=1000,
    n_chains=n_chains,
    seed=2,
    target_accept=0.8,
)
```

```python
trace_rust.sample_stats.is_diverging.sum('draw')
```

```python
import seaborn as sns
```

```python
#trace_rust.posterior["a"]
```

```python
np.log(trace_rust.warmup_sample_stats.step_size_bar).plot(x="draw", hue="chain")
```

```python
np.log(trace_rust.warmup_sample_stats.first_diag_mass_matrix).plot(x="draw", hue="chain")
```

```python
1 / 1.05
```

```python
np.log(trace_py.warmup_sample_stats.step_size_bar).plot(x="draw", hue="chain");
```

```python
trace_py.warmup_posterior.sd.isel(sd_dim_0=0).plot.line(x="draw")
```

```python
trace_rust.warmup_posterior.sd.isel(sd_dim_0=0).plot.line(x="draw")
```

```python
sns.kdeplot(trace_rust.posterior["a"].values[:, :, 0].ravel())
sns.kdeplot(trace_py.posterior["a"].values[:, :, 0].ravel())
#sns.kdeplot(np.random.randn(16_000))
```

```python
sns.kdeplot(np.log(trace_rust.posterior["a"].values[:, :, -1].ravel()))
sns.kdeplot(np.log(trace_py.posterior["a"].values[:, :, -1].ravel()))
#sns.kdeplot(2000 * np.random.randn(16_000))
```

```python
from scipy import stats
```

```python
stats.ks_1samp(trace_rust.posterior["a"].values[:, :, 0].ravel(), stats.norm(scale=1).cdf)
```

```python
stats.ks_1samp(trace_py.posterior["a"].values[:, :, 0].ravel(), stats.norm(scale=1).cdf)
```

```python
sns.histplot(trace_rust.sample_stats.mean_acceptance_rate.values.ravel(), color='C0')
sns.histplot(trace_py.sample_stats.acceptance_rate.values.ravel(), color='C1')
```

```python
sns.kdeplot(trace_rust.sample_stats.mean_acceptance_rate.mean('draw').values.ravel(), color='C0')
sns.kdeplot(trace_py.sample_stats.acceptance_rate.mean('draw').values.ravel(), color='C1')
```

```python
sns.kdeplot(trace_rust.sample_stats.step_size.mean('draw').values.ravel(), color='C0')
sns.kdeplot(trace_py.sample_stats.step_size.mean('draw').values.ravel(), color='C1')
```

```python
sns.countplot(trace_rust.sample_stats.idx_in_trajectory.values.ravel())
```

```python
(trace_py.posterior.diff('draw').a == 0).mean()
```

```python
(trace_rust.posterior.diff('draw').a == 0).mean()
```

```python
(trace_rust.sample_stats.idx_in_trajectory == 0).mean()
```

```python
trace_rust.sample_stats.depth.mean()
```

```python
trace_py.sample_stats.tree_depth.mean()
```

```python
trace_rust.sample_stats.tree_size
```

```python
trace_py.sample_stats.n_steps
```

```python
arviz.plot_autocorr(trace_rust.posterior.a.values[0, :, 0])
```

```python
arviz.plot_autocorr(np.random.randn(len(trace_py.posterior.a.values[0, :, 0])))
```

```python
arviz.plot_autocorr(trace_py.posterior.a.values[0, :, 0])
```

```python
ess_py = arviz.ess(trace_py)
ess_rust = arviz.ess(trace_rust)
```

```python
ess_rust
```

```python
sns.kdeplot(ess_rust.a)
sns.kdeplot(ess_py.a)
```

```python

```

```python
trace_rust.sample_stats.tree_size.mean()
```

```python
trace_py.sample_stats.n_steps.mean()
```

```python
sns.histplot(trace_rust.sample_stats.step_size.isel(draw=0).values.ravel(), color='C0')
sns.histplot(trace_py.sample_stats.step_size.isel(draw=0).values.ravel(), color='C1')
```

```python

```

```python

```
