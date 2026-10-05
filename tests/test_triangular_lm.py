import numpy as np
import pytest
import scipy.sparse as sp

jax = pytest.importorskip("jax")

import equinox as eqx
import jax.numpy as jnp

from nutpie.normalizing_flow import make_transformer
from nutpie.sparsity import _fill
from nutpie.triangular import SparseTriangularMap
from nutpie.triangular_lm import (
    make_residuals,
    pack_params,
    param_index,
    unpack_params,
)


def _blanket(dim, kind, rng):
    if kind == "banded":
        blanket = np.zeros((dim, dim), dtype=bool)
        for i in range(1, dim):
            blanket[i, max(0, i - 2) : i] = True
        return blanket
    # A cycle without chords, which needs fill.
    blanket = rng.random((dim, dim)) < 0.4
    blanket[0, dim - 1] = blanket[1, 0] = True
    return _filled(blanket, np.arange(dim))


def _filled(blanket, order):
    """`blanket` closed under elimination in the flow order `order`, as a
    symbolic factorization fills it: the residuals require that."""
    blanket = blanket | blanket.T
    np.fill_diagonal(blanket, False)
    return _fill(sp.csr_array(blanket), order).toarray()


def _setup(
    nn_width=3,
    blanket="banded",
    bounds=None,
    tangent_sas=0,
    fix_b=False,
    dim=7,
    n_draw=8,
    seed=0,
):
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(seed)
    transformer = make_transformer(
        contract_transformer=0 if tangent_sas else 2,
        asymmetric_transformer=False,
        log_gamma_bounds=bounds,
        tangent_sas_transformer=tangent_sas,
        tangent_sas_fix_b=fix_b,
    )
    tmap = SparseTriangularMap(
        jax.random.key(seed),
        blanket=_blanket(dim, blanket, rng),
        transformer=transformer,
        n_buckets=3,
        nn_width=nn_width,
        nn_activation=jax.nn.softplus,
    )
    # Away from the zero init, so that every term is exercised.
    arrays, static = eqx.partition(tmap.conditioners, eqx.is_inexact_array)
    arrays = jax.tree.map(
        lambda a: a + 0.15 * jnp.asarray(rng.normal(size=a.shape)), arrays
    )
    tmap = eqx.tree_at(lambda m: m.conditioners, tmap, eqx.combine(arrays, static))
    y = rng.normal(size=(n_draw, dim))
    g = -y + 0.3 * rng.normal(size=(n_draw, dim))
    return tmap, y, g


def _score_index(tmap):
    """Positions of the edges, in CSR order, in the concatenated per-bucket
    parent scores."""
    (dim,) = tmap.shape
    position = {}
    offset = 0
    for members, parents in zip(tmap.bucket_members, tmap.bucket_parent_indices):
        parents = np.asarray(parents)
        width = parents.shape[1]
        for local, variable in enumerate(np.asarray(members)):
            n_parent = int(np.sum(parents[local] < dim))
            position[int(variable)] = offset + local * width + np.arange(n_parent)
        offset += parents.size
    return np.concatenate([position[i] for i in range(dim)]).astype(np.int64)


def _jax_residuals(tmap, y, g, rho):
    index = param_index(tmap)
    score_index = _score_index(tmap)
    n_draw = y.shape[0]

    def residuals(theta):
        flow_map = unpack_params(tmap, theta, index)

        def one(y, g):
            x, grad_x, _, scores = flow_map.inverse_gradient_and_val(
                y, g, 0.0, parent_scores=True
            )
            out = x + grad_x
            if rho is not None:
                scores = jnp.concatenate([s.ravel() for s in scores])[score_index]
                out = jnp.concatenate([out, jnp.sqrt(rho) * scores])
            return out

        return jax.vmap(one)(jnp.asarray(y), jnp.asarray(g)) / np.sqrt(n_draw)

    return residuals


CASES = [
    {},
    {"nn_width": 0},
    {"blanket": "random"},
    {"bounds": (-1.0, 1.0)},
    {"tangent_sas": 2},
    {"tangent_sas": 3, "blanket": "random"},
    {"tangent_sas": 2, "fix_b": True},
]


@pytest.mark.parametrize("rho", [None, 0.3])
@pytest.mark.parametrize("case", CASES)
def test_operators_match_jax(case, rho):
    tmap, y, g = _setup(**case)
    problem = make_residuals(tmap, y, g, fisher_regularization=rho)
    theta = pack_params(tmap)
    shape = (y.shape[0], problem.n_residuals)
    reference = _jax_residuals(tmap, y, g, rho)

    r = problem.residuals(theta).reshape(shape)
    np.testing.assert_allclose(r, reference(theta), rtol=1e-10, atol=1e-12)

    rng = np.random.default_rng(1)
    v = rng.normal(size=theta.shape)
    r_bar = rng.normal(size=shape)
    jv = problem.pushforward(v).reshape(shape)
    jt_r = problem.pullback(r_bar.ravel())

    _, jv_ref = jax.jvp(reference, (theta,), (v,))
    _, vjp = jax.vjp(reference, theta)
    np.testing.assert_allclose(jv, jv_ref, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(jt_r, vjp(r_bar)[0], rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(np.vdot(jv, r_bar), np.vdot(v, jt_r), rtol=1e-10)

    np.testing.assert_allclose(
        problem.gauss_newton_product(v),
        problem.pullback(jv.ravel()),
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.parametrize("rho", [None, 0.3])
@pytest.mark.parametrize("max_block_size", [7, 1000])
@pytest.mark.parametrize("case", CASES)
def test_gauss_newton_blocks_match_dense_gram(case, max_block_size, rho):
    tmap, y, g = _setup(**case)
    problem = make_residuals(tmap, y, g, fisher_regularization=rho)
    theta = pack_params(tmap)
    problem.residuals(theta)

    jac = jax.jacfwd(_jax_residuals(tmap, y, g, rho))(theta)
    jac = np.asarray(jac).reshape(-1, theta.size)
    gram = jac.T @ jac

    starts, sizes, data = problem.gauss_newton_blocks(max_block_size)
    assert sizes.max() <= max_block_size
    offset = 0
    covered = np.zeros(theta.size, dtype=int)
    for start, size in zip(starts, sizes):
        block = data[offset : offset + size * size].reshape(size, size)
        offset += size * size
        covered[start : start + size] += 1
        expected = gram[start : start + size, start : start + size]
        # The blocks read `Sigma = (J^T J)^-1` from Takahashi's recurrence,
        # whose error grows like `cond(J)^2 eps`. The random blanket gives
        # some draws `cond(K) ~ 1e11`; with the banded one the worst draw has
        # `cond(J) ~ 1e5`, which costs a few `1e-10`.
        tol = 1e-5 if case.get("blanket") == "random" else 1e-9
        np.testing.assert_allclose(
            block, expected, rtol=0, atol=tol * np.abs(expected).max()
        )
    assert offset == data.size
    assert np.all(covered == 1)


def test_unpack_inverts_pack():
    tmap, _, _ = _setup(blanket="random")
    theta = pack_params(tmap)
    shifted = unpack_params(tmap, theta + 1.0)
    np.testing.assert_array_equal(pack_params(shifted), theta + 1.0)


def _flow_problem(dim=7, n_draw=64, rho=None, seed=0):
    """A `make_flow(kind="triangular")` flow near its init, its data and a
    `FisherLoss`."""
    import flowjax

    from nutpie.normalizing_flow import make_flow
    from nutpie.transform_adapter import FisherLoss

    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(seed)
    x = 0.5 + rng.normal(size=(n_draw, dim)) * np.exp(rng.normal(size=dim))
    g = -(x - 0.5) / np.exp(2 * rng.normal(size=dim)) + 0.3 * rng.normal(
        size=(n_draw, dim)
    )
    order = rng.permutation(dim)
    bijection = make_flow(
        seed,
        x,
        g,
        n_layers=1,
        kind="triangular",
        sparsity=_filled(_blanket(dim, "banded", rng), order),
        order=order,
        nn_width=4,
        nn_depth=1,
        activation=jax.nn.softplus,
        contract_transformer=2,
        n_buckets=2,
    )
    flow = flowjax.flows.Transformed(
        flowjax.distributions.StandardNormal((dim,)), bijection
    )
    params, static = eqx.partition(flow, eqx.is_inexact_array)
    params = jax.tree.map(
        lambda leaf: leaf + 0.05 * jnp.asarray(rng.normal(size=leaf.shape)), params
    )
    data = (jnp.asarray(x), jnp.asarray(g), jnp.zeros(n_draw))
    return params, static, data, FisherLoss(fisher_regularization=rho)


@pytest.mark.parametrize("rho", [None, 0.1])
def test_rust_lm_steps_match_lmopt(rho):
    """With every conditioner in one sub-block, both preconditioners are the
    same matrix, so the steps agree up to rounding. `lmopt` has neither the
    `lam_lo` reset nor the larger `lam0`."""
    from nutpie import lmopt
    from nutpie.transform_adapter import gn_factor_fn, res_fn
    from nutpie.triangular_lm import fit, map_data

    params, static, data, loss_fn = _flow_problem(rho=rho)
    n_steps = 4
    _, reference = lmopt.fit(
        params,
        res_fn,
        (loss_fn, static),
        data=data,
        fit_affine=False,
        line_search=True,
        factor_fn=gn_factor_fn,
        max_exact_block_size=10_000,
        n_steps=n_steps,
        verbose=False,
    )

    tmap, y, g = map_data(eqx.combine(params, static), data[0], data[1])
    problem = make_residuals(tmap, y, g, fisher_regularization=rho)
    _, _, hist = fit(
        problem,
        pack_params(tmap),
        n_steps=n_steps,
        lam0=1e-2,
        max_block_size=10_000,
        lam_lo_reset=0.0,
    )

    assert len(hist) == n_steps
    for ours, ref in zip(hist, reference):
        assert ours["accept"] == bool(ref["accept"])
        assert ours["n_cg"] == int(ref["n_cg"])
        for key in ["F", "F_new", "rho_full", "lam_out", "step_length"]:
            np.testing.assert_allclose(ours[key], float(ref[key]), rtol=1e-6)


def test_rust_lm_with_limited_memory_preconditioner():
    """The limited-memory preconditioner only changes the CG solves: the fit
    still converges to about the same loss, and it is actually used."""
    from nutpie.triangular_lm import fit, map_data

    params, static, data, loss_fn = _flow_problem(rho=0.1)
    tmap, y, g = map_data(eqx.combine(params, static), data[0], data[1])
    problem = make_residuals(tmap, y, g, fisher_regularization=0.1)
    settings = dict(n_steps=30, max_block_size=8, cg_tol=1e-6, verbose=False)

    _, _, base = fit(problem, pack_params(tmap), **settings)
    _, _, hist = fit(problem, pack_params(tmap), lmp_size=8, **settings)

    assert hist[0]["lmp_rank"] == 0
    assert all(0 < info["lmp_rank"] <= 8 for info in hist[1:])
    assert hist[-1]["F_out"] < 0.1 * hist[0]["F"]
    np.testing.assert_allclose(hist[-1]["F_out"], base[-1]["F_out"], rtol=0.1)


def test_lm_rust_method_updates_the_flow():
    """`fit_to_data(method="lm-rust")` writes the fitted conditioners back:
    the returned flow's loss is the one the fit reports."""
    from nutpie.transform_adapter import fit_to_data

    params, static, data, loss_fn = _flow_problem(rho=0.1)
    initial = float(jnp.sum(loss_fn.residuals(params, static, *data) ** 2))
    flow, losses, _ = fit_to_data(
        jax.random.key(0),
        eqx.combine(params, static),
        data,
        loss_fn=loss_fn,
        method="lm-rust",
        max_epochs=5,
        lm_min_loss=0.0,
    )
    fitted, _ = eqx.partition(flow, eqx.is_inexact_array)
    r = loss_fn.residuals(fitted, static, *data)
    assert losses["lm_steps"] == 5
    assert losses["train"][0] < initial
    np.testing.assert_allclose(float(jnp.sum(r**2)), losses["train"][0], rtol=1e-10)
