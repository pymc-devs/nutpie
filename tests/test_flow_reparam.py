from importlib.util import find_spec

import pytest

if find_spec("pymc") is None:
    pytest.skip("Skip pymc tests", allow_module_level=True)

import numpy as np
import pymc as pm
import pytensor
from pymc import dims as pmd
from pymc.distributions.transforms import Interval

import nutpie
from nutpie.flow_reparam import (
    AffineFlow,
    CholeskyFlow,
    NoFlow,
    ShiftFlow,
    automatic_flow_reparam,
    build_flow_graph,
)


def _compile_flow(model):
    """Compile the model with nutpie, build the flow graph over its
    flat variables, and compile the pytensor constrain/unconstrain
    functions."""
    records = automatic_flow_reparam(model)
    compiled = nutpie.compile_pymc_model(model, backend="jax", gradient_backend="jax")
    n_dim = compiled.n_dim
    free_vars_info = [v for v in compiled._variables if v.end_idx <= n_dim]
    g = build_flow_graph(records, free_vars_info, n_dim)
    constrain_fn = pytensor.function(*g["constrain"])
    unconstrain_fn = pytensor.function(*g["unconstrain"])
    return compiled, records, constrain_fn, unconstrain_fn


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
        dist_fn(mu=mu, sigma=1.0, dims="group")

    records = automatic_flow_reparam(m)
    (name,) = [n for n in records if n in {"y", "y_log__"}]
    assert records[name]["flow_cls"] is AffineFlow
    assert records[name]["param_shapes"] == [(1,), (1,)]


@pytest.mark.pymc
def test_scale_shift_flow():
    coords = {"group": [0, 1, 2]}
    with pm.Model(coords=coords) as m:
        beta = pm.HalfNormal("beta", 1)
        pm.Gamma("x", alpha=2.0, beta=beta, dims="group")

    r = automatic_flow_reparam(m)["x_log__"]
    assert r["flow_cls"] is ShiftFlow
    assert r["param_shapes"] == [(1,)]


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
    assert records["ind_mu"]["param_shapes"] == [(1,), (1,)]

    n_dim = int(compiled.n_dim)
    assert n_dim == 5  # pop_mu + pop_sigma_log__ + ind_mu(group=3)
    total_params = sum(
        int(np.prod(sh)) for r in records.values() for sh in r["param_shapes"]
    )
    ind_mu_var = next(v for v in compiled._variables if v.name == "ind_mu")
    ind_mu_slice = slice(ind_mu_var.start_idx, ind_mu_var.end_idx)

    # (1) Identity flow: both directions are identity, log|J| = 0.
    point = np.arange(n_dim, dtype="float64")
    zero_params = np.zeros(total_params, dtype="float64")
    c_point, ljd_c = constrain_fn(point, zero_params)
    u_point, ljd_u = unconstrain_fn(point, zero_params)
    np.testing.assert_allclose(c_point, point)
    np.testing.assert_allclose(u_point, point)
    np.testing.assert_allclose(ljd_c, 0.0)
    np.testing.assert_allclose(ljd_u, 0.0)

    # (2) Random non-identity: roundtrip and log|J| cancellation.
    rng = np.random.default_rng(0)
    phi0 = rng.normal(size=n_dim).astype("float64")
    rand_params = rng.normal(size=total_params).astype("float64") * 0.3
    value, ljd_c = constrain_fn(phi0, rand_params)
    phi_back, ljd_u = unconstrain_fn(value, rand_params)
    np.testing.assert_allclose(phi_back, phi0, atol=1e-10)
    np.testing.assert_allclose(ljd_c + ljd_u, 0.0, atol=1e-10)
    assert not np.isclose(ljd_c, 0.0)

    # (3) Analytical: shift=0.5, log_scale=0.1, phi_ind_mu=1
    #     → value_ind_mu = 0.5 + exp(0.1).
    shift, log_scale = 0.5, 0.1
    analytic_params = np.array([shift, log_scale], dtype="float64")
    analytic_phi = np.zeros(n_dim, dtype="float64")
    analytic_phi[ind_mu_slice] = 1.0
    value, _ = constrain_fn(analytic_phi, analytic_params)
    np.testing.assert_allclose(value[ind_mu_slice], shift + np.exp(log_scale))


@pytest.mark.pymc
def test_partial_broadcast():
    coords = {"group": range(3), "rep": range(4)}
    with pm.Model(coords=coords) as m:
        pop_mu = pm.Normal("pop_mu", 0, 1, dims="group")
        pm.Normal("x", pop_mu[:, None], 1.0, dims=("group", "rep"))

    compiled, records, constrain_fn, unconstrain_fn = _compile_flow(m)
    assert records["x"]["flow_cls"] is AffineFlow
    assert records["x"]["param_shapes"] == [(3, 1), (3, 1)]

    n_dim = int(compiled.n_dim)
    assert n_dim == 15  # pop_mu(group=3) + x(group=3, rep=4)
    rng = np.random.default_rng(1)
    phi0 = rng.normal(size=n_dim).astype("float64")
    total_params = sum(
        int(np.prod(sh)) for r in records.values() for sh in r["param_shapes"]
    )
    flow_params = rng.normal(size=total_params).astype("float64") * 0.2

    value, _ = constrain_fn(phi0, flow_params)
    phi_back, _ = unconstrain_fn(value, flow_params)
    np.testing.assert_allclose(phi_back, phi0, atol=1e-10)


@pytest.mark.pymc
def test_loc_scale_cholesky_flow():
    coords = {
        "site": ["a", "b", "c", "d"],
        "feature": [0, 1, 2],
    }
    with pm.Model(coords=coords) as m:
        site_mean = pm.Normal("site_mean", 0, 1, dims="feature")
        pm.MvNormal("x", mu=site_mean, cov=np.eye(3), dims=("site", "feature"))

    compiled, records, constrain_fn, unconstrain_fn = _compile_flow(m)
    # site axis (size 4) collapses to 1 since mean/cov don't broadcast
    # across it; feature axis (size 3) is the event dim.
    assert records["x"]["flow_cls"] is CholeskyFlow
    assert records["x"]["param_shapes"] == [(1, 3), (1, 6)]

    n_dim = int(compiled.n_dim)
    assert n_dim == 15  # site_mean(feature=3) + x(site=4, feature=3)
    total_params = sum(
        int(np.prod(sh)) for r in records.values() for sh in r["param_shapes"]
    )
    phi0 = np.arange(n_dim, dtype="float64")

    # Identity flow (zero params): constrain/unconstrain are the identity.
    c_point, ljd_c = constrain_fn(phi0, np.zeros(total_params, dtype="float64"))
    np.testing.assert_allclose(c_point, phi0)
    np.testing.assert_allclose(ljd_c, 0.0)

    # Random non-identity: roundtrip and log|J| cancellation.
    rng = np.random.default_rng(0)
    flow_params = rng.normal(size=total_params).astype("float64") * 0.2
    value, ljd_c = constrain_fn(phi0, flow_params)
    phi_back, ljd_u = unconstrain_fn(value, flow_params)
    np.testing.assert_allclose(phi_back, phi0, atol=1e-10)
    assert not np.isclose(ljd_c, 0.0)
    np.testing.assert_allclose(ljd_c + ljd_u, 0.0, atol=1e-10)


@pytest.mark.pymc
def test_flow_alongside_dirichlet():
    coords = {"group": [0, 1], "k": range(3)}
    with pm.Model(coords=coords) as m:
        pi = pm.Dirichlet("pi", a=np.ones(3), dims="k")
        pop_sigma = pm.HalfNormal("pop_sigma", 1)
        pm.Normal("ind_mu", pi, pop_sigma, dims=("group", "k"))

    compiled, records, constrain_fn, unconstrain_fn = _compile_flow(m)
    assert records["ind_mu"]["flow_cls"] is AffineFlow
    (pi_name,) = [n for n in records if n.startswith("pi")]
    assert records[pi_name]["flow_cls"] is NoFlow

    n_dim = int(compiled.n_dim)
    assert n_dim == 9  # 2 (dirichlet, simplex-transformed) + 1 + 6
    total_params = sum(
        int(np.prod(sh)) for r in records.values() for sh in r["param_shapes"]
    )

    phi0 = np.arange(n_dim, dtype="float64")
    c_point, ljd_c = constrain_fn(phi0, np.zeros(total_params, dtype="float64"))
    np.testing.assert_allclose(c_point, phi0)
    np.testing.assert_allclose(ljd_c, 0.0)

    rng = np.random.default_rng(0)
    flow_params = rng.normal(size=total_params).astype("float64") * 0.3
    value, ljd_c = constrain_fn(phi0, flow_params)
    phi_back, ljd_u = unconstrain_fn(value, flow_params)
    np.testing.assert_allclose(phi_back, phi0, atol=1e-10)
    np.testing.assert_allclose(ljd_c + ljd_u, 0.0, atol=1e-10)


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
    assert records["ind_mu"]["param_shapes"] == [(1,), (1,)]
    assert records["scale_log__"]["flow_cls"] is AffineFlow
    assert records["scale_log__"]["param_shapes"] == [(1,), (1,)]

    n_dim = int(compiled.n_dim)
    assert n_dim == 8  # pop_mu + pop_sigma_log__ + ind_mu(3) + scale_log__(3)
    total_params = sum(
        int(np.prod(sh)) for r in records.values() for sh in r["param_shapes"]
    )
    phi0 = np.arange(n_dim, dtype="float64")

    c_point, ljd_c = constrain_fn(phi0, np.zeros(total_params, dtype="float64"))
    np.testing.assert_allclose(c_point, phi0)
    np.testing.assert_allclose(ljd_c, 0.0)

    rng = np.random.default_rng(0)
    flow_params = rng.normal(size=total_params).astype("float64") * 0.2
    value, ljd_c = constrain_fn(phi0, flow_params)
    phi_back, ljd_u = unconstrain_fn(value, flow_params)
    np.testing.assert_allclose(phi_back, phi0, atol=1e-10)
    np.testing.assert_allclose(ljd_c + ljd_u, 0.0, atol=1e-10)
