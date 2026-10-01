import numpy as np
import pytest

jax = pytest.importorskip("jax")

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from nutpie.normalizing_flow import make_transformer  # noqa: E402
from nutpie.triangular import SparseTriangularMap  # noqa: E402
from nutpie.triangular_lm import (  # noqa: E402
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
    # A cycle without chords needs fill in the selected inverse.
    blanket = rng.random((dim, dim)) < 0.4
    blanket[0, dim - 1] = blanket[1, 0] = True
    return blanket


def _setup(nn_width=3, blanket="banded", bounds=None, dim=7, n_draw=5, seed=0):
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(seed)
    transformer = make_transformer(
        contract_transformer=2, asymmetric_transformer=False, log_gamma_bounds=bounds
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
        # some draws `cond(K) ~ 1e11`; everything else is exact to rounding.
        tol = 1e-5 if case.get("blanket") == "random" else 1e-10
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
