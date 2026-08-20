from importlib.util import find_spec

import pytest

if find_spec("flowjax") is None:
    pytest.skip("Skip normalizing flow tests", allow_module_level=True)

import jax
import numpy as np

from nutpie.normalizing_flow import SparseTriangularMap


def _allowed_pairs(order, blanket):
    """(i, j) pairs that the sparsity pattern permits j to influence i."""
    dim = len(order)
    position = np.argsort(order)
    sym_blanket = blanket | blanket.T
    allowed = np.zeros((dim, dim), dtype=bool)
    for i in range(dim):
        for j in range(dim):
            if i != j and sym_blanket[i, j] and position[j] < position[i]:
                allowed[i, j] = True
    return allowed


@pytest.mark.flow
def test_sparse_triangular_round_trip_and_log_det():
    dim = 6
    order = np.array([3, 0, 4, 1, 5, 2])

    rng = np.random.default_rng(0)
    blanket = rng.random((dim, dim)) < 0.3
    np.fill_diagonal(blanket, False)

    bij = SparseTriangularMap(
        jax.random.key(0), order=order, blanket=blanket, nn_width=8, nn_depth=1
    )

    x = jax.random.normal(jax.random.key(1), (dim,))

    y, log_det = bij.transform_and_log_det(x)
    x_rt, inv_log_det = bij.inverse_and_log_det(y)

    np.testing.assert_allclose(x_rt, x, atol=1e-3)
    np.testing.assert_allclose(log_det, -inv_log_det, atol=1e-3)


@pytest.mark.flow
def test_sparse_triangular_respects_sparsity_pattern():
    dim = 6
    order = np.array([3, 0, 4, 1, 5, 2])

    rng = np.random.default_rng(1)
    blanket = rng.random((dim, dim)) < 0.3
    np.fill_diagonal(blanket, False)

    bij = SparseTriangularMap(
        jax.random.key(2), order=order, blanket=blanket, nn_width=8, nn_depth=1
    )

    x = jax.random.normal(jax.random.key(3), (dim,))

    jac = jax.jacfwd(lambda x: bij.transform_and_log_det(x)[0])(x)
    jac = np.asarray(jac)

    allowed = _allowed_pairs(order, blanket) | np.eye(dim, dtype=bool)
    np.testing.assert_array_equal(jac[~allowed], 0.0)

    _, log_det = bij.transform_and_log_det(x)
    sign, logdet_full = np.linalg.slogdet(jac)
    assert sign > 0
    np.testing.assert_allclose(logdet_full, log_det, atol=1e-3)


@pytest.mark.flow
def test_sparse_triangular_empty_blanket_is_elementwise():
    dim = 4
    order = np.arange(dim)
    blanket = np.zeros((dim, dim), dtype=bool)

    bij = SparseTriangularMap(
        jax.random.key(4), order=order, blanket=blanket, nn_width=4, nn_depth=1
    )

    x = jax.random.normal(jax.random.key(5), (dim,))
    jac = np.asarray(jax.jacfwd(lambda x: bij.transform_and_log_det(x)[0])(x))

    np.testing.assert_array_equal(jac[~np.eye(dim, dtype=bool)], 0.0)
