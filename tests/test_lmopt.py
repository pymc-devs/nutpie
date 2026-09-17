from importlib.util import find_spec

import pytest

if find_spec("jax") is None or find_spec("equinox") is None:
    pytest.skip("Skip lmopt tests", allow_module_level=True)

import jax
import jax.numpy as jnp
import numpy as np

from nutpie.lmopt import (
    _block_inv,
    _sanitize_blocks,
    describe_fallbacks,
    marquardt_floors,
)


@pytest.fixture(autouse=True)
def x64():
    # Only for these tests, not for the rest of the session
    with jax.enable_x64(True):
        yield


def spd_blocks(n, G, q, seed=0):
    rng = np.random.default_rng(seed)
    V = rng.normal(size=(n, G, 2 * q, q))
    return jnp.asarray(np.einsum("ngri,ngrj->ngij", V, V))


def test_sanitize_blocks():
    H = spd_blocks(3, 2, 4)
    broken = H.at[1, 0, 2, 3].set(jnp.nan).at[2, 1, 0, 0].set(jnp.inf)
    clean, counts = _sanitize_blocks([H, broken])
    np.testing.assert_array_equal(counts, [0, 2])
    np.testing.assert_array_equal(clean[0], H)
    assert jnp.all(jnp.isfinite(clean[1]))
    # Untouched sub-blocks stay as they are
    np.testing.assert_array_equal(clean[1][0], H[0])
    # Broken ones keep their finite diagonal, the non-finite entry becomes 0
    expected = jnp.diag(jnp.diagonal(H[1, 0]))
    np.testing.assert_array_equal(clean[1][1, 0], expected)
    expected = jnp.diag(jnp.diagonal(H[2, 1]).at[0].set(0.0))
    np.testing.assert_array_equal(clean[1][2, 1], expected)
    # The floors are finite again, instead of NaN for every conditioner
    assert all(jnp.all(jnp.isfinite(f)) for f in marquardt_floors(clean))
    assert not all(jnp.all(jnp.isfinite(f)) for f in marquardt_floors([H, broken]))


@pytest.mark.parametrize("floor", [None, 1e-3])
def test_block_inv(floor):
    H = spd_blocks(1, 1, 5)[0, 0]
    inv, failed = _block_inv(H, 0.1, floor, 0.0)
    assert not failed
    if floor is None:
        damped = H + 0.1 * jnp.eye(5)
    else:
        damped = H + jnp.diag(0.1 * jnp.maximum(jnp.diagonal(H), floor) + 1e-4 * floor)
    np.testing.assert_allclose(inv @ damped, np.eye(5), atol=1e-6)

    # An indefinite block breaks the Cholesky, so the damped diagonal is used
    bad = -1e3 * jnp.eye(5).at[0, 1].set(1.0).at[1, 0].set(1.0)
    inv, failed = _block_inv(bad, 0.1, floor, 0.0)
    assert failed
    assert jnp.all(jnp.isfinite(inv))
    np.testing.assert_array_equal(inv, jnp.diag(jnp.diagonal(inv)))


def test_describe_fallbacks():
    plans = [{"parents": 2}, {"parents": 8}, {"parents": None}]
    info = {
        "nonfinite_blocks": np.array([0, 3, 0]),
        "failed_inverses": np.array([1, 0, 2]),
    }
    assert describe_fallbacks(info, plans) == (
        "  non-finite blocks: 3 (8 parents: 3)"
        "  diagonal inverses: 3 (2 parents: 1, affine parents: 2)"
    )
    info = {"nonfinite_blocks": np.zeros(3), "failed_inverses": np.zeros(3)}
    assert describe_fallbacks(info, plans) == ""
    assert describe_fallbacks({}, plans) == ""
