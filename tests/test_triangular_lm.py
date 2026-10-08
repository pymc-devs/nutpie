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
    squash=None,
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
        input_squash=squash,
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
    # The draws are standard normal, so 0.7 is well away from the identity.
    {"squash": 0.7},
    {"squash": 0.7, "blanket": "random", "bounds": (-1.0, 1.0)},
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


def test_lm_rust_validation_loss():
    """`fit_to_data(method="lm-rust", val_x=...)` reports the Fisher
    divergences of the train and held-out draws at the returned flow, without
    the regularization."""
    from nutpie.transform_adapter import fit_to_data

    params, static, data, loss_fn = _flow_problem(n_draw=96, rho=0.1)
    train = tuple(a[:64] for a in data)
    val = tuple(a[64:] for a in data)
    for early_stopping in [True, False]:
        flow, losses, _ = fit_to_data(
            jax.random.key(0),
            eqx.combine(params, static),
            train,
            val_x=val,
            early_stopping=early_stopping,
            loss_fn=loss_fn,
            method="lm-rust",
            max_epochs=8,
            lm_min_loss=0.0,
        )
        fitted, _ = eqx.partition(flow, eqx.is_inexact_array)
        # `FisherLoss.__call__` is the log of the plain Fisher divergence.
        np.testing.assert_allclose(
            np.log(losses["val"][0]), loss_fn(fitted, static, *val), rtol=1e-8
        )
        np.testing.assert_allclose(
            np.log(losses["train_fisher"][0]),
            loss_fn(fitted, static, *train),
            rtol=1e-8,
        )
        assert losses["train_fisher"][0] < losses["train"][0]


@pytest.mark.parametrize("early_stopping", [True, False])
def test_rust_lm_early_stopping(early_stopping):
    """With early stopping, `fit` returns the parameters of the best
    validation loss seen (the start included); without, the last ones."""
    from nutpie.triangular_lm import fisher_divergence, fit, map_data

    params, static, data, _ = _flow_problem(n_draw=48, rho=None)
    flow = eqx.combine(params, static)
    tmap, y, g = map_data(flow, data[0][:16], data[1][:16])
    _, val_y, val_g = map_data(flow, data[0][16:], data[1][16:])
    problem = make_residuals(tmap, y, g)
    val_problem = make_residuals(tmap, val_y, val_g)
    theta0 = pack_params(tmap)

    theta, _, hist = fit(
        problem,
        theta0,
        n_steps=15,
        verbose=False,
        val_problem=val_problem,
        early_stopping=early_stopping,
        patience=3,
    )
    val_F = fisher_divergence(val_problem, theta)
    if early_stopping:
        best = min(
            [fisher_divergence(val_problem, theta0)] + [h["val_F"] for h in hist]
        )
        np.testing.assert_allclose(val_F, best, rtol=1e-12)
    else:
        np.testing.assert_allclose(val_F, hist[-1]["val_F"], rtol=1e-12)


def test_select_draws():
    """Disjoint, in range, chronological, multiples of the SIMD width,
    never upsampled, and the validation draws come in contiguous blocks."""
    from nutpie.transform_adapter import _select_draws

    settings = {
        "max_draws": 512,
        "recency": 1.0,
        "val_fraction": 0.2,
        "block_size": 16,
        "multiple": 8,
    }
    for n_draws, start in [(257, 128), (2000, 1000), (140, 120)]:
        train, val = _select_draws(
            n_draws, start, rng=np.random.default_rng(0), **settings
        )
        both = np.concatenate([train, val])
        assert len(np.unique(both)) == len(both)
        assert both.min() >= start and both.max() < n_draws
        assert len(both) <= min(n_draws - start, 512)
        assert len(train) % 8 == 0 and len(val) % 8 == 0
        if n_draws - start >= 100:
            assert len(val) > 0
        assert (np.diff(train) > 0).all() and (np.diff(val) > 0).all()
        again = _select_draws(n_draws, start, rng=np.random.default_rng(0), **settings)
        assert all((a == b).all() for a, b in zip((train, val), again))

    # Without thinning, each held-out block is a run of consecutive draws.
    _, val = _select_draws(
        400, 0, rng=np.random.default_rng(1), **{**settings, "multiple": 1}
    )
    runs = np.split(val, np.flatnonzero(np.diff(val) != 1) + 1)
    assert all(len(run) % 16 == 0 for run in runs)


@pytest.mark.parametrize("available", [600, 1000, 5000])
@pytest.mark.parametrize("recency", [0.0, 1.0, 3.0])
def test_thin(available, recency):
    """`max_draws` distinct sorted draws; even for `recency=0`, else denser
    towards the newest, with the newest all taken where the density caps."""
    from nutpie.transform_adapter import _thin

    idx = _thin(available, 512, recency)
    assert len(idx) == 512
    assert (np.diff(idx) > 0).all()
    assert idx.min() >= 0 and idx.max() < available
    gaps = np.diff(idx)
    if recency == 0:
        assert gaps.max() - gaps.min() <= 1
    else:
        older, newer = np.array_split(gaps, 2)
        assert newer.mean() < older.mean()
        if recency == 3.0 and available == 600:
            # Capped: the newest stretch is taken whole.
            assert (gaps[-50:] == 1).all()


def test_make_flow_passes_log_gamma_bounds():
    """`make_flow(kind="triangular", log_gamma_bounds=...)` reaches every
    `Contract2` layer, and so the Rust residuals' transformer spec."""
    import flowjax

    from nutpie.normalizing_flow import make_flow
    from nutpie.triangular_layout import _probe_transformer, transformer_dicts
    from nutpie.triangular_lm import map_data

    rng = np.random.default_rng(0)
    dim = 5
    x = rng.normal(size=(16, dim))
    bijection = make_flow(
        0,
        x,
        -x,
        n_layers=1,
        kind="triangular",
        sparsity=_filled(_blanket(dim, "banded", rng), np.arange(dim)),
        nn_width=4,
        activation=jax.nn.softplus,
        contract_transformer=2,
        log_gamma_bounds=(-1.0, 1.0),
    )
    flow = flowjax.flows.Transformed(
        flowjax.distributions.StandardNormal((dim,)), bijection
    )
    tmap, _, _ = map_data(flow, x, -x)
    n_par = int(tmap.conditioners[0].mlp.layers[1].out_features)
    layers = transformer_dicts(_probe_transformer(tmap.transformer_constructor, n_par))
    bounds = [layer.get("log_gamma_bounds") for layer in layers]
    bounds = [b for b in bounds if b is not None]
    assert len(bounds) == 2
    assert all(tuple(b) == (-1.0, 1.0) for b in bounds)
