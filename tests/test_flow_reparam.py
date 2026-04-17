from importlib.util import find_spec

import pytest

if find_spec("pymc") is None:
    pytest.skip("Skip pymc tests", allow_module_level=True)

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pymc import dims as pmd
from pymc.distributions.transforms import Interval

try:
    from pytensor.gradient import pullback
except ImportError:  # pytensor < 3.0 used the name Lop
    from pytensor.gradient import Lop as pullback

import nutpie
from nutpie.flow_reparam import (
    AffineFlow,
    NoFlow,
    ShiftFlow,
    automatic_flow_reparam,
    build_auto_flow,
    build_flow_graph,
    free_vars_info,
)


def _compile_flow(model):
    """Compile the model with nutpie, build the flow graph over its
    flat variables, and compile the pytensor constrain/unconstrain
    functions."""
    records = automatic_flow_reparam(model)
    compiled = nutpie.compile_pymc_model(model, backend="jax", gradient_backend="jax")
    g = build_flow_graph(model, free_vars_info(compiled), compiled.n_dim)
    constrain_fn = pytensor.function(*g["constrain"])
    unconstrain_fn = pytensor.function(*g["unconstrain"])
    return compiled, records, constrain_fn, unconstrain_fn


def _total_params(records):
    return sum(int(np.prod(sh)) for r in records.values() for sh in r["param_shapes"])


def _var_slice(compiled, name):
    v = next(v for v in compiled._variables if v.name == name)
    return slice(v.start_idx, v.end_idx)


@pytest.mark.pymc
def test_root_rv_not_reparametrized():
    with pm.Model() as m:
        pm.Normal("x", 0, 1, shape=(3,))

    records = automatic_flow_reparam(m)
    assert records["x"]["flow_cls"] is NoFlow
    assert records["x"]["param_shapes"] == []


@pytest.mark.pymc
def test_transformed_rv_not_reparametrized():
    coords = {"group": [0, 1, 2]}
    with pm.Model(coords=coords) as m:
        pop_mu = pm.Normal("pop_mu", 0, 1)
        pop_sigma = pm.HalfNormal("pop_sigma", 1)
        # This could be fine, but there may rewrites
        # like ordered/zerosum that change things too much?
        pm.Normal(
            "ind_mu",
            pop_mu,
            pop_sigma,
            dims="group",
            transform=Interval(lower=-10, upper=10),
        )

    records = automatic_flow_reparam(m)
    (name,) = [n for n in records if n.startswith("ind_mu")]
    assert records[name]["flow_cls"] is NoFlow


@pytest.mark.parametrize(
    "dist_fn",
    [
        lambda mu, sigma, **k: pm.Normal("y", mu, sigma, **k),
        lambda mu, sigma, **k: pm.Cauchy("y", mu, sigma, **k),
        lambda mu, sigma, **k: pm.Laplace("y", mu, sigma, **k),
        lambda mu, sigma, **k: pm.Logistic("y", mu, sigma, **k),
        lambda mu, sigma, **k: pm.Gumbel("y", mu, sigma, **k),
        lambda mu, sigma, **k: pm.StudentT("y", nu=3, mu=mu, sigma=sigma, **k),
        lambda mu, sigma, **k: pm.LogNormal("y", mu, sigma, **k),
    ],
    ids=["Normal", "Cauchy", "Laplace", "Logistic", "Gumbel", "StudentT", "LogNormal"],
)
@pytest.mark.pymc
def test_loc_scale_affine_flow(dist_fn):
    with pm.Model(coords={"group": range(4)}) as m:
        mu = pm.Normal("mu", 0, 1)
        sigma = pm.HalfNormal("sigma", 1)
        dist_fn(mu=mu, sigma=sigma, dims="group")

    records = automatic_flow_reparam(m)
    (name,) = [n for n in records if n in {"y", "y_log__"}]
    assert records[name]["flow_cls"] is AffineFlow
    assert records[name]["param_shapes"] == [(4,), (4,)]


@pytest.mark.pymc
def test_per_param_qualification():
    """A dist param that is constant or full-shape gets a size-0 knob,
    pinned at the centred no-op."""
    coords = {"group": range(3)}

    with pm.Model(coords=coords) as m:
        sigma = pm.HalfNormal("sigma", 1)
        pm.Normal("y", 0.0, sigma, dims="group")
    r = automatic_flow_reparam(m)["y"]
    assert r["flow_cls"] is AffineFlow
    assert r["param_shapes"] == [(0,), (3,)]

    with pm.Model(coords=coords) as m:
        mu = pm.Normal("mu", 0, 1)
        pm.Normal("y", mu, 2.0, dims="group")
    r = automatic_flow_reparam(m)["y"]
    assert r["flow_cls"] is AffineFlow
    assert r["param_shapes"] == [(3,), (0,)]

    with pm.Model(coords=coords) as m:
        mu = pm.Normal("mu", 0, 1)
        sigma = pm.HalfNormal("sigma", 1, dims="group")
        pm.Normal("y", mu, sigma, dims="group")
    r = automatic_flow_reparam(m)["y"]
    assert r["flow_cls"] is AffineFlow
    assert r["param_shapes"] == [(3,), (0,)]

    with pm.Model(coords=coords) as m:
        pm.Normal("root", 0, 1)
        pm.Normal("y", 0.0, 2.0, dims="group")
    r = automatic_flow_reparam(m)["y"]
    assert r["flow_cls"] is NoFlow


@pytest.mark.pymc
def test_scale_shift_flow():
    coords = {"group": [0, 1, 2]}
    with pm.Model(coords=coords) as m:
        beta = pm.HalfNormal("beta", 1)
        pm.Gamma("x", alpha=2.0, beta=beta, dims="group")

    r = automatic_flow_reparam(m)["x_log__"]
    assert r["flow_cls"] is ShiftFlow
    assert r["param_shapes"] == [(3,)]


@pytest.mark.pymc
def test_zerosum_scale_flow():
    with pm.Model(coords={"group": range(5)}) as m:
        sigma = pm.HalfNormal("pop_sigma", 1)
        pm.ZeroSumNormal("x", sigma=sigma, dims="group")

    r = automatic_flow_reparam(m)["x_zerosum__"]
    assert r["flow_cls"] is AffineFlow
    # No loc knob; one scale knob shared across the zero-sum core dim.
    assert r["param_shapes"] == [(0,), (1,)]

    with pm.Model(coords={"batch": range(2), "group": range(5)}) as m:
        sigma = pm.HalfNormal("pop_sigma", 1)
        pm.ZeroSumNormal("x", sigma=sigma, dims=("batch", "group"))

    r = automatic_flow_reparam(m)["x_zerosum__"]
    assert r["flow_cls"] is AffineFlow
    assert r["param_shapes"] == [(0,), (2, 1)]

    with pm.Model(coords={"group": range(5)}) as m:
        pm.Normal("root", 0, 1)
        pm.ZeroSumNormal("x", sigma=2.0, dims="group")

    r = automatic_flow_reparam(m)["x_zerosum__"]
    assert r["flow_cls"] is NoFlow


@pytest.mark.pymc
def test_hierarchical_normal():
    coords = {"group": [0, 1, 2]}
    with pm.Model(coords=coords) as m:
        pop_mu = pm.Normal("pop_mu", 0, 1)
        pop_sigma = pm.HalfNormal("pop_sigma", 1)
        pm.Normal("ind_mu", pop_mu, pop_sigma, dims="group")

    compiled, records, constrain_fn, unconstrain_fn = _compile_flow(m)

    assert records["pop_mu"]["flow_cls"] is NoFlow
    assert records["pop_sigma_log__"]["flow_cls"] is NoFlow
    assert records["ind_mu"]["flow_cls"] is AffineFlow
    assert records["ind_mu"]["param_shapes"] == [(3,), (3,)]

    n_dim = int(compiled.n_dim)
    assert n_dim == 5  # pop_mu + pop_sigma_log__ + ind_mu(group=3)
    total_params = _total_params(records)
    assert total_params == 6
    pop_mu_sl = _var_slice(compiled, "pop_mu")
    pop_sigma_sl = _var_slice(compiled, "pop_sigma_log__")
    ind_mu_sl = _var_slice(compiled, "ind_mu")

    # Centred no-op (h = 0): both directions are the identity, log|J| = 0.
    point = np.arange(n_dim, dtype="float64")
    zero_params = np.zeros(total_params, dtype="float64")
    c_point, ljd_c = constrain_fn(point, zero_params)
    u_point, ljd_u = unconstrain_fn(point, zero_params)
    np.testing.assert_allclose(c_point, point, atol=1e-9)
    np.testing.assert_allclose(u_point, point, atol=1e-9)
    np.testing.assert_allclose(ljd_c, 0.0, atol=1e-9)
    np.testing.assert_allclose(ljd_u, 0.0, atol=1e-9)

    # Random non-identity: roundtrip and log|J| cancellation.
    rng = np.random.default_rng(0)
    phi0 = rng.normal(size=n_dim).astype("float64")
    rand_params = rng.normal(size=total_params).astype("float64") * 0.3
    value, ljd_c = constrain_fn(phi0, rand_params)
    phi_back, ljd_u = unconstrain_fn(value, rand_params)
    np.testing.assert_allclose(phi_back, phi0, atol=1e-10)
    np.testing.assert_allclose(ljd_c + ljd_u, 0.0, atol=1e-10)
    assert not np.isclose(ljd_c, 0.0)

    # Analytical per-element VIP transform (Gorinova et al. 2019, with
    # h = 1-λ so that h = 0 is centred):
    #     value_i = μ + σ^h_σi·(y_i - (1-h_μi)·μ),  log|J| = Σ_i h_σi·log σ
    h_mu = np.array([0.4, 0.2, 0.0])
    h_sigma = np.array([0.1, 0.5, 0.9])
    analytic_params = np.concatenate([h_mu, h_sigma])
    y = rng.normal(size=n_dim).astype("float64")
    mu = y[pop_mu_sl][0]
    sigma = np.exp(y[pop_sigma_sl][0])
    value, ljd_c = constrain_fn(y, analytic_params)
    expected = mu + sigma**h_sigma * (y[ind_mu_sl] - (1 - h_mu) * mu)
    np.testing.assert_allclose(value[ind_mu_sl], expected, atol=1e-10)
    np.testing.assert_allclose(ljd_c, (h_sigma * np.log(sigma)).sum(), atol=1e-10)


@pytest.mark.pymc
def test_partial_broadcast():
    coords = {"group": range(3), "rep": range(4)}
    with pm.Model(coords=coords) as m:
        pop_mu = pm.Normal("pop_mu", 0, 1, dims="group")
        pm.Normal("x", pop_mu[:, None], 1.0, dims=("group", "rep"))

    compiled, records, constrain_fn, unconstrain_fn = _compile_flow(m)
    assert records["x"]["flow_cls"] is AffineFlow
    assert records["x"]["param_shapes"] == [(3, 4), (0,)]

    n_dim = int(compiled.n_dim)
    assert n_dim == 15  # pop_mu(group=3) + x(group=3, rep=4)
    rng = np.random.default_rng(1)
    phi0 = rng.normal(size=n_dim).astype("float64")
    flow_params = rng.normal(size=_total_params(records)).astype("float64") * 0.2

    value, ljd_c = constrain_fn(phi0, flow_params)
    phi_back, ljd_u = unconstrain_fn(value, flow_params)
    np.testing.assert_allclose(phi_back, phi0, atol=1e-10)
    # With the scale knob pinned, the transform is a translation.
    np.testing.assert_allclose(ljd_c, 0.0, atol=1e-10)
    np.testing.assert_allclose(ljd_u, 0.0, atol=1e-10)


@pytest.mark.pymc
def test_flow_alongside_dirichlet():
    coords = {"group": [0, 1], "k": range(3)}
    with pm.Model(coords=coords) as m:
        pi = pm.Dirichlet("pi", a=np.ones(3), dims="k")
        pop_sigma = pm.HalfNormal("pop_sigma", 1)
        pm.Normal("ind_mu", pi, pop_sigma, dims=("group", "k"))

    compiled, records, constrain_fn, unconstrain_fn = _compile_flow(m)
    assert records["ind_mu"]["flow_cls"] is AffineFlow
    assert records["ind_mu"]["param_shapes"] == [(2, 3), (2, 3)]
    (pi_name,) = [n for n in records if n.startswith("pi")]
    assert records[pi_name]["flow_cls"] is NoFlow

    n_dim = int(compiled.n_dim)
    assert n_dim == 9  # 2 (dirichlet, simplex-transformed) + 1 + 6
    total_params = _total_params(records)

    phi0 = np.arange(n_dim, dtype="float64")
    zero_params = np.zeros(total_params, dtype="float64")
    c_point, ljd_c = constrain_fn(phi0, zero_params)
    np.testing.assert_allclose(c_point, phi0, atol=1e-9)
    np.testing.assert_allclose(ljd_c, 0.0, atol=1e-9)

    rng = np.random.default_rng(0)
    flow_params = rng.normal(size=total_params).astype("float64") * 0.3
    value, ljd_c = constrain_fn(phi0, flow_params)
    phi_back, ljd_u = unconstrain_fn(value, flow_params)
    np.testing.assert_allclose(phi_back, phi0, atol=1e-10)
    np.testing.assert_allclose(ljd_c + ljd_u, 0.0, atol=1e-10)


@pytest.mark.pymc
def test_zerosum_roundtrip():
    with pm.Model(coords={"group": range(5)}) as m:
        sigma = pm.HalfNormal("pop_sigma", 1)
        pm.ZeroSumNormal("x", sigma=sigma, dims="group")

    compiled, records, constrain_fn, unconstrain_fn = _compile_flow(m)
    n_dim = int(compiled.n_dim)
    assert n_dim == 5  # pop_sigma_log__ + x_zerosum__(4)
    total_params = _total_params(records)
    assert total_params == 1

    phi0 = np.arange(n_dim, dtype="float64")
    zero_params = np.zeros(total_params, dtype="float64")
    c_point, ljd_c = constrain_fn(phi0, zero_params)
    np.testing.assert_allclose(c_point, phi0, atol=1e-9)
    np.testing.assert_allclose(ljd_c, 0.0, atol=1e-9)

    rng = np.random.default_rng(0)
    phi0 = rng.normal(size=n_dim).astype("float64")
    flow_params = np.array([0.7])
    value, ljd_c = constrain_fn(phi0, flow_params)
    phi_back, ljd_u = unconstrain_fn(value, flow_params)
    np.testing.assert_allclose(phi_back, phi0, atol=1e-10)
    np.testing.assert_allclose(ljd_c + ljd_u, 0.0, atol=1e-10)

    # Full NCP (h = 1): value = y·σ, log|J| = 4·log σ.
    x_sl = _var_slice(compiled, "x_zerosum__")
    s_sl = _var_slice(compiled, "pop_sigma_log__")
    value, ljd_c = constrain_fn(phi0, np.array([1.0]))
    sigma_val = np.exp(phi0[s_sl][0])
    np.testing.assert_allclose(value[x_sl], phi0[x_sl] * sigma_val, atol=1e-10)
    np.testing.assert_allclose(ljd_c, 4 * np.log(sigma_val), atol=1e-10)


@pytest.mark.pymc
def test_dim_distributions():
    coords = {"group": [0, 1, 2]}
    with pm.Model(coords=coords) as m:
        pop_mu = pmd.Normal("pop_mu", 0, 1)
        pop_sigma = pmd.HalfNormal("pop_sigma", 1)
        pmd.Normal("ind_mu", pop_mu, pop_sigma, dims=("group",))
        pmd.LogNormal("scale", pop_mu, pop_sigma, dims=("group",))

    compiled, records, constrain_fn, unconstrain_fn = _compile_flow(m)
    assert records["pop_mu"]["flow_cls"] is NoFlow
    assert records["pop_sigma_log__"]["flow_cls"] is NoFlow
    assert records["ind_mu"]["flow_cls"] is AffineFlow
    assert records["ind_mu"]["param_shapes"] == [(3,), (3,)]
    assert records["scale_log__"]["flow_cls"] is AffineFlow
    assert records["scale_log__"]["param_shapes"] == [(3,), (3,)]

    n_dim = int(compiled.n_dim)
    assert n_dim == 8  # pop_mu + pop_sigma_log__ + ind_mu(3) + scale_log__(3)
    total_params = _total_params(records)
    phi0 = np.arange(n_dim, dtype="float64")

    zero_params = np.zeros(total_params, dtype="float64")
    c_point, ljd_c = constrain_fn(phi0, zero_params)
    np.testing.assert_allclose(c_point, phi0, atol=1e-9)
    np.testing.assert_allclose(ljd_c, 0.0, atol=1e-9)

    rng = np.random.default_rng(0)
    flow_params = rng.normal(size=total_params).astype("float64") * 0.2
    value, ljd_c = constrain_fn(phi0, flow_params)
    phi_back, ljd_u = unconstrain_fn(value, flow_params)
    np.testing.assert_allclose(phi_back, phi0, atol=1e-10)
    np.testing.assert_allclose(ljd_c + ljd_u, 0.0, atol=1e-10)


def _fisher_loss_fn(model, compiled):
    """Build ``loss(draws, flow_params)`` mirroring nutpie's FisherLoss:
    pull value-space posterior draws and logp-gradients back through the
    flow and measure deviation from a standard-normal score, minimized
    analytically over the per-coordinate affine that the production
    chain's diagonal-affine layers would absorb. Zero iff the pulled-back
    posterior is iid normal up to a diagonal affine."""
    n_dim = int(compiled.n_dim)
    free_vars = free_vars_info(compiled)
    g = build_flow_graph(model, free_vars, n_dim)

    (y, flow_params), (value_out, ljd) = g["constrain"]
    value_grad = value_out.type("value_grad")
    # vjp of (constrain, log|J|) at cotangent (grad, 1), as in
    # transform_adapter.inverse_gradient_and_val.
    pulled_grad = pullback([value_out, ljd], y, [value_grad, pt.ones(())])
    pullback_fn = pytensor.function([y, flow_params, value_grad], pulled_grad)
    unconstrain_fn = pytensor.function(*g["unconstrain"])

    value_vars = {v.name: v for v in model.value_vars}
    ordered = [value_vars[v.name] for v in free_vars]
    grad_fn = pytensor.function(ordered, pt.grad(model.logp(), ordered))

    def flat_grad(draw):
        vals = [
            draw[v.start_idx : v.end_idx].reshape(tuple(int(s) for s in v.shape))
            for v in free_vars
        ]
        return np.concatenate([np.asarray(a).ravel() for a in grad_fn(*vals)])

    def loss(draws, params):
        params = np.asarray(params, dtype="float64")
        xs, gs = [], []
        for draw in draws:
            x, _ = unconstrain_fn(draw, params)
            xs.append(x)
            gs.append(pullback_fn(x, params, flat_grad(draw)))
        x, g = np.array(xs), np.array(gs)
        # min over per-coordinate affine z = a·(x-b) of E[(z + g/a)²]:
        # 2·(sqrt(Var x·Var g) + Cov(x, g)), ≥ 0 by Cauchy-Schwarz.
        cov = ((x - x.mean(0)) * (g - g.mean(0))).mean(0)
        # ≥ 0 by Cauchy-Schwarz; clamp the float noise around exact zeros.
        costs = np.maximum(2 * (np.sqrt(x.var(0) * g.var(0)) + cov), 0.0)
        return float(np.log(np.maximum(costs.sum(), 1e-300)))

    return loss


def _funnel_model(obs_sigma=None, n_groups=5, n_obs=1000):
    """The mixed-balance funnel from notebooks/auto_reparam-Copy1.ipynb:
    per-group means with a common log-scale hyper, optionally observed
    with per-group noise."""
    coords = {"group": range(n_groups)}
    with pm.Model(coords=coords) as m:
        s = pm.Normal("pop_sigma_log", 0, 1)
        ind_mu = pm.Normal("ind_mu", 0, pm.math.exp(s / 2), dims="group")
        if obs_sigma is not None:
            rng = np.random.default_rng(1)
            y = rng.normal(
                loc=np.linspace(-0.5, 0.5, n_groups),
                scale=1.0,
                size=(n_obs, n_groups),
            )
            pm.Normal("y", ind_mu, sigma=np.asarray(obs_sigma), observed=y)
            return m, y
    return m, None


def _funnel_posterior_draws(compiled, rng, n_draws, obs_sigma=None, y=None):
    """Exact posterior draws in nutpie's flat value space: dense-grid
    sampling for the 1-D hyper (conjugate marginal over the group means),
    then the exact Gaussian conditional for ind_mu | s, y."""
    s_sl = _var_slice(compiled, "pop_sigma_log")
    m_sl = _var_slice(compiled, "ind_mu")
    n_groups = m_sl.stop - m_sl.start

    if obs_sigma is None:
        s = rng.normal(0.0, 1.0, size=n_draws)
        means = np.zeros((n_draws, n_groups))
        post_sd = np.exp(s / 2)[:, None] * np.ones(n_groups)
    else:
        obs_sigma = np.asarray(obs_sigma, dtype="float64")
        n_obs = y.shape[0]
        ybar = y.mean(axis=0)
        grid = np.linspace(-10, 10, 8001)
        tau2 = np.exp(grid)
        # p(s | y) ∝ p(s) · Π_i N(ȳ_i; 0, τ² + σ_i²/n)
        marg_var = tau2[:, None] + (obs_sigma**2 / n_obs)[None, :]
        log_post = -0.5 * grid**2 - 0.5 * (
            np.log(marg_var) + ybar[None, :] ** 2 / marg_var
        ).sum(axis=1)
        p = np.exp(log_post - log_post.max())
        s = rng.choice(grid, size=n_draws, p=p / p.sum())
        prec = 1 / np.exp(s)[:, None] + (n_obs / obs_sigma**2)[None, :]
        means = (n_obs * ybar / obs_sigma**2)[None, :] / prec
        post_sd = 1 / np.sqrt(prec)

    ind_mu = rng.normal(means, post_sd)
    draws = np.empty((n_draws, int(compiled.n_dim)))
    draws[:, s_sl] = s[:, None]
    draws[:, m_sl] = ind_mu
    return draws


@pytest.mark.pymc
def test_loss_prefers_noncentered_on_prior_funnel():
    m, _ = _funnel_model()
    compiled = nutpie.compile_pymc_model(m, backend="jax", gradient_backend="jax")
    loss = _fisher_loss_fn(m, compiled)

    rng = np.random.default_rng(42)
    draws = _funnel_posterior_draws(compiled, rng, n_draws=256)

    cp = loss(draws, np.zeros(5))
    ncp = loss(draws, np.ones(5))
    assert ncp < cp
    # Full NCP standardizes the prior funnel exactly, so the loss is ~0.
    assert ncp < -25


@pytest.mark.pymc
def test_loss_prefers_centered_on_strong_data():
    obs_sigma = np.ones(5)
    m, y = _funnel_model(obs_sigma=obs_sigma)
    compiled = nutpie.compile_pymc_model(m, backend="jax", gradient_backend="jax")
    loss = _fisher_loss_fn(m, compiled)

    rng = np.random.default_rng(42)
    draws = _funnel_posterior_draws(compiled, rng, 256, obs_sigma=obs_sigma, y=y)

    assert loss(draws, np.zeros(5)) < loss(draws, np.ones(5))


@pytest.mark.pymc
def test_loss_prefers_mixed_on_mixed_balance():
    """Strong-evidence groups want centred, weak ones non-centred: the
    per-element mixed parameterization beats both global ones — the case
    a single per-group knob cannot express."""
    obs_sigma = np.array([1.0, 1000.0, 1.0, 1000.0, 1.0])
    m, y = _funnel_model(obs_sigma=obs_sigma)
    compiled = nutpie.compile_pymc_model(m, backend="jax", gradient_backend="jax")
    loss = _fisher_loss_fn(m, compiled)

    rng = np.random.default_rng(42)
    draws = _funnel_posterior_draws(compiled, rng, 256, obs_sigma=obs_sigma, y=y)

    mixed = loss(draws, np.array([0.0, 1.0, 0.0, 1.0, 0.0]))
    assert mixed < loss(draws, np.zeros(5))
    assert mixed < loss(draws, np.ones(5))


@pytest.mark.pymc
def test_loss_prefers_noncentered_on_zerosum_prior():
    with pm.Model(coords={"group": range(5)}) as m:
        s = pm.Normal("s", 0, 1)
        x = pm.ZeroSumNormal("x", sigma=pm.math.exp(s), dims="group")

    compiled = nutpie.compile_pymc_model(m, backend="jax", gradient_backend="jax")
    loss = _fisher_loss_fn(m, compiled)

    s_draws, x_draws = pm.draw([s, x], draws=256, random_seed=42)
    x_in = pt.matrix("x_in")
    forward_fn = pytensor.function([x_in], m.rvs_to_transforms[x].forward(x_in))

    draws = np.empty((256, int(compiled.n_dim)))
    draws[:, _var_slice(compiled, "s")] = s_draws[:, None]
    draws[:, _var_slice(compiled, "x_zerosum__")] = forward_fn(x_draws)

    cp = loss(draws, np.zeros(1))
    ncp = loss(draws, np.ones(1))
    assert ncp < cp
    # The zero-sum transform is an isometry, so full NCP is exact.
    assert ncp < -25


@pytest.mark.pymc
def test_conditional_transform_parent():
    # A parent with a *conditional* transform (Interval reads the RV's
    # distribution parameters) feeding a flow child: the transform must be
    # applied with the actual rv inputs, not IR wrapper inputs.
    coords = {"group": [0, 1, 2]}
    with pm.Model(coords=coords) as m:
        a = pm.Uniform("a", -1.0, 3.0)
        pop_sigma = pm.HalfNormal("pop_sigma", 1)
        pm.Normal("x", a, pop_sigma, dims="group")

    compiled, records, constrain_fn, unconstrain_fn = _compile_flow(m)
    assert records["a_interval__"]["flow_cls"] is NoFlow
    assert records["pop_sigma_log__"]["flow_cls"] is NoFlow
    assert records["x"]["flow_cls"] is AffineFlow
    assert records["x"]["param_shapes"] == [(3,), (3,)]

    n_dim = int(compiled.n_dim)
    assert n_dim == 5
    a_sl = _var_slice(compiled, "a_interval__")
    sigma_sl = _var_slice(compiled, "pop_sigma_log__")
    x_sl = _var_slice(compiled, "x")

    rng = np.random.default_rng(2)
    y = rng.normal(size=n_dim)
    h_mu = np.array([0.7, 0.3, 0.1])
    h_sigma = np.array([0.2, 0.5, 0.8])
    flow_params = np.concatenate([h_mu, h_sigma])

    value, ljd_c = constrain_fn(y, flow_params)
    # interval backward: lower + (upper - lower) * sigmoid(value)
    a_con = -1.0 + 4.0 / (1.0 + np.exp(-y[a_sl][0]))
    sigma_con = np.exp(y[sigma_sl][0])
    expected_x = (y[x_sl] - (1 - h_mu) * a_con) * sigma_con**h_sigma + a_con
    np.testing.assert_allclose(value[x_sl], expected_x, atol=1e-10)
    np.testing.assert_allclose(ljd_c, (h_sigma * np.log(sigma_con)).sum(), atol=1e-10)

    y_back, ljd_u = unconstrain_fn(value, flow_params)
    np.testing.assert_allclose(y_back, y, atol=1e-10)
    np.testing.assert_allclose(ljd_c + ljd_u, 0.0, atol=1e-10)


@pytest.mark.pymc
def test_flow_parent_read_through_expression():
    # A *flow* parent read by its child through an expression (indexing):
    # the constrain and unconstrain graphs must each compose their own
    # direction of the parent flow (regression for in-place mutation of
    # the shared param subgraphs between the two builds).
    coords = {"group": range(3), "rep": range(4)}
    with pm.Model(coords=coords) as m:
        pop_mu = pm.Normal("pop_mu", 0, 1)
        pop_sigma = pm.HalfNormal("pop_sigma", 1)
        mu_g = pm.Normal("mu_g", pop_mu, pop_sigma, dims="group")
        pm.Normal("x", mu_g[:, None], 1.0, dims=("group", "rep"))

    compiled, records, constrain_fn, unconstrain_fn = _compile_flow(m)
    assert records["mu_g"]["flow_cls"] is AffineFlow
    assert records["mu_g"]["param_shapes"] == [(3,), (3,)]
    assert records["x"]["flow_cls"] is AffineFlow
    # Constant sigma: the scale knob is withheld (size 0) and pinned.
    assert records["x"]["param_shapes"] == [(3, 4), (0,)]

    n_dim = int(compiled.n_dim)
    assert n_dim == 17  # 1 + 1 + 3 + 12
    total_params = _total_params(records)
    assert total_params == 18

    rng = np.random.default_rng(3)
    y = rng.normal(size=n_dim)
    h1_mu = rng.normal(size=3) * 0.4
    h1_sigma = rng.normal(size=3) * 0.4
    h2_mu = rng.normal(size=(3, 4)) * 0.4
    flow_params = np.concatenate([h1_mu, h1_sigma, h2_mu.ravel()])

    mu_sl = _var_slice(compiled, "pop_mu")
    sigma_sl = _var_slice(compiled, "pop_sigma_log__")
    mug_sl = _var_slice(compiled, "mu_g")
    x_sl = _var_slice(compiled, "x")

    value, ljd_c = constrain_fn(y, flow_params)
    pop_mu_con = y[mu_sl][0]
    sigma_con = np.exp(y[sigma_sl][0])
    mug_con = (y[mug_sl] - (1 - h1_mu) * pop_mu_con) * sigma_con**h1_sigma + pop_mu_con
    # With the scale knob pinned the child flow is the translation
    # y + h_mu·loc.
    expected_x = y[x_sl].reshape(3, 4) + h2_mu * mug_con[:, None]
    np.testing.assert_allclose(value[mug_sl], mug_con, atol=1e-10)
    np.testing.assert_allclose(value[x_sl], expected_x.ravel(), atol=1e-10)
    np.testing.assert_allclose(ljd_c, (h1_sigma * np.log(sigma_con)).sum(), atol=1e-10)

    y_back, ljd_u = unconstrain_fn(value, flow_params)
    np.testing.assert_allclose(y_back, y, atol=1e-10)
    np.testing.assert_allclose(ljd_c + ljd_u, 0.0, atol=1e-10)


@pytest.mark.pymc
def test_xtensor_unknown_transform_parent():
    # A dims RV whose transform has no plain counterpart (Beta -> logodds)
    # is not lifted: it stays xtensor-typed end to end and its dim
    # transform is applied natively.
    coords = {"group": [0, 1, 2]}
    with pm.Model(coords=coords) as m:
        p = pmd.Beta("p", 1.0, 1.0)
        pop_sigma = pmd.HalfNormal("pop_sigma", 1)
        pmd.Normal("x", p, pop_sigma, dims=("group",))

    compiled, records, constrain_fn, unconstrain_fn = _compile_flow(m)
    (p_name,) = [n for n in records if n.startswith("p_")]
    assert records[p_name]["flow_cls"] is NoFlow
    assert records["x"]["flow_cls"] is AffineFlow
    assert records["x"]["param_shapes"] == [(3,), (3,)]

    n_dim = int(compiled.n_dim)
    assert n_dim == 5
    p_sl = _var_slice(compiled, p_name)
    sigma_sl = _var_slice(compiled, "pop_sigma_log__")
    x_sl = _var_slice(compiled, "x")

    rng = np.random.default_rng(4)
    y = rng.normal(size=n_dim)
    h_mu = np.array([0.6, 0.2, 0.9])
    h_sigma = np.array([0.4, 0.7, 0.1])
    flow_params = np.concatenate([h_mu, h_sigma])

    value, ljd_c = constrain_fn(y, flow_params)
    p_con = 1.0 / (1.0 + np.exp(-y[p_sl][0]))  # logodds backward
    sigma_con = np.exp(y[sigma_sl][0])
    expected_x = (y[x_sl] - (1 - h_mu) * p_con) * sigma_con**h_sigma + p_con
    np.testing.assert_allclose(value[x_sl], expected_x, atol=1e-10)
    np.testing.assert_allclose(ljd_c, (h_sigma * np.log(sigma_con)).sum(), atol=1e-10)

    y_back, ljd_u = unconstrain_fn(value, flow_params)
    np.testing.assert_allclose(y_back, y, atol=1e-10)
    np.testing.assert_allclose(ljd_c + ljd_u, 0.0, atol=1e-10)


@pytest.mark.pymc
@pytest.mark.flow
def test_build_auto_flow_roundtrip():
    import jax.numpy as jnp

    from nutpie.normalizing_flow import AutoFlow

    m, _ = _funnel_model()
    compiled = nutpie.compile_pymc_model(m, backend="jax", gradient_backend="jax")
    flow = build_auto_flow(m, compiled, init_params=jnp.full((5,), 0.3))
    assert isinstance(flow, AutoFlow)
    assert flow.shape == (int(compiled.n_dim),)

    rng = np.random.default_rng(0)
    y = jnp.asarray(rng.normal(size=flow.shape))
    x, ljd = flow.transform_and_log_det(y)
    y_back, ljd_back = flow.inverse_and_log_det(x)
    np.testing.assert_allclose(np.asarray(y_back), np.asarray(y), atol=1e-10)
    np.testing.assert_allclose(float(ljd) + float(ljd_back), 0.0, atol=1e-10)
    assert not np.isclose(float(ljd), 0.0)


@pytest.mark.pymc
@pytest.mark.flow
def test_auto_reparam_compile_api(capsys):
    from nutpie.normalizing_flow import AutoFlow

    m, _ = _funnel_model()
    compiled = nutpie.compile_pymc_model(
        m, backend="jax", gradient_backend="jax", auto_reparam=True
    )
    summary = capsys.readouterr().out
    assert "reparametrizing 1 of 2 free variables" in summary
    assert "ind_mu (AffineFlow)" in summary
    auto_flow = compiled._transform_adapt_args["auto_flow"]
    assert isinstance(auto_flow, AutoFlow)

    tuned = compiled.with_transform_adapt(num_layers=0)
    assert tuned._transform_adapt_args["auto_flow"] is auto_flow
    assert tuned._transform_adapt_args["num_layers"] == 0
    cleared = tuned.with_transform_adapt(auto_flow=None)
    assert "auto_flow" not in cleared._transform_adapt_args

    with pytest.raises(ValueError, match="auto_reparam"):
        nutpie.compile_pymc_model(m, auto_reparam=True)
    with pytest.raises(ValueError, match="auto_reparam"):
        nutpie.compile_pymc_model(
            m, backend="jax", gradient_backend="pytensor", auto_reparam=True
        )


@pytest.mark.pymc
@pytest.mark.flow
@pytest.mark.parametrize("n_layers", [0, 2])
def test_auto_flow_is_outermost_bijection(n_layers):
    """The VIP flow's constrain output is the model's value vector, so it
    must sit at the value-space end of the chain; the diag affine and any
    coupling layers operate in its base space."""
    from nutpie.normalizing_flow import AutoFlow, make_flow

    m, _ = _funnel_model()
    compiled = nutpie.compile_pymc_model(m, backend="jax", gradient_backend="jax")
    auto_flow = build_auto_flow(m, compiled)

    rng = np.random.default_rng(0)
    n_dim = int(compiled.n_dim)
    positions = rng.normal(size=(10, n_dim))
    gradients = rng.normal(size=(10, n_dim))
    chain = make_flow(1, positions, gradients, n_layers=n_layers, auto_flow=auto_flow)
    assert isinstance(chain.bijections[-1], AutoFlow)


@pytest.mark.pymc
@pytest.mark.flow
def test_auto_reparam_nothing_found():
    with pm.Model() as m:
        pm.Normal("x", 0, 1, shape=(3,))

    with pytest.warns(UserWarning, match="did not find any variables"):
        compiled = nutpie.compile_pymc_model(
            m, backend="jax", gradient_backend="jax", auto_reparam=True
        )
    assert "auto_flow" not in (compiled._transform_adapt_args or {})


@pytest.mark.pymc
@pytest.mark.flow
def test_multiple_auto_flows_chain_to_one():
    from flowjax import bijections

    from nutpie.transform_adapter import make_transform_adapter

    m, _ = _funnel_model()
    compiled = nutpie.compile_pymc_model(m, backend="jax", gradient_backend="jax")
    flow = build_auto_flow(m, compiled)
    adapter = make_transform_adapter(auto_flow=[flow, flow])
    chained = adapter.keywords["make_flow_fn"].keywords["auto_flow"]
    assert isinstance(chained, bijections.Chain)
    assert len(chained.bijections) == 2


@pytest.mark.pymc
@pytest.mark.flow
def test_auto_reparam_sampling():
    m, _ = _funnel_model()
    compiled = nutpie.compile_pymc_model(
        m, backend="jax", gradient_backend="jax", auto_reparam=True
    )
    trace = nutpie.sample(
        compiled, chains=1, seed=1, adaptation="flow", tune=1000, draws=500
    )
    assert float(trace.sample_stats.diverging.sum()) <= 5
    np.testing.assert_allclose(
        float(trace.posterior.pop_sigma_log.std()), 1.0, atol=0.3
    )
