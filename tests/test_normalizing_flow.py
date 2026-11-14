from importlib.util import find_spec

import pytest

if find_spec("flowjax") is None:
    pytest.skip("Skip normalizing flow tests", allow_module_level=True)

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from nutpie.normalizing_flow import Contract2, DipBij, _exp_asinh, make_flow

pytestmark = pytest.mark.flow


def test_exp_asinh_float32():
    x = np.array([-1e4, -1e3, -10.0, -1.0, 0.0, 1.0, 10.0, 1e3, 1e4])
    value = _exp_asinh(jnp.asarray(x, dtype=jnp.float32))
    grad = jax.vmap(jax.grad(_exp_asinh))(jnp.asarray(x, dtype=jnp.float32))
    expected = np.exp(np.arcsinh(x))
    np.testing.assert_allclose(value, expected, rtol=1e-6)
    np.testing.assert_allclose(grad, expected / np.sqrt(1 + x**2), rtol=1e-6)


@pytest.mark.parametrize("value", [-1e4, -1e3, -10.0, 10.0, 1e3, 1e4])
def test_contract2_identity_float32(value):
    # Regression test for https://github.com/pymc-devs/nutpie/issues/364
    zero = jnp.zeros((), dtype=jnp.float32)
    bijection = Contract2(None, zero, zero, zero, zero)
    value = jnp.float32(value)
    y, y_logdet = bijection.transform_and_log_det(value)
    x, x_logdet = bijection.inverse_and_log_det(value)
    np.testing.assert_allclose([y, x], [value, value], rtol=1e-5)
    np.testing.assert_allclose([y_logdet, x_logdet], [0.0, 0.0], atol=1e-4)


def test_make_flow_small_scale_float32():
    # Regression test for https://github.com/pymc-devs/nutpie/issues/363
    draws = np.random.default_rng(0).normal(size=(1000, 2))
    logdets = []
    for scale in [1.0, 1e-2, 1e-4]:
        positions = scale * draws
        gradients = -positions / scale**2
        flow = make_flow(0, positions, gradients, n_layers=0)
        _, logdet = flow.transform_and_log_det(jnp.zeros(2, dtype=jnp.float32))
        logdets.append(float(logdet) - 2 * np.log(scale))
    np.testing.assert_allclose(logdets, logdets[0], atol=1e-3)


@pytest.mark.parametrize("b", [-1.0, 0.0, 1.0])
def test_dipbij_inverse_gradient_float32(b):
    # Regression test for https://github.com/pymc-devs/nutpie/issues/361
    bijection = eqx.tree_at(lambda bij: bij.b, DipBij(), jnp.float32(b))
    a = np.tanh(b)
    # One Cardano cube root argument is zero where y = +-sqrt(3 * (1 - a))
    root = np.sqrt(3 * (1 - a))
    y = jnp.asarray(
        [-root, -1.0, 0.0, 1.0, root - 1e-3, root, root + 1e-3, 1.76, 10.0],
        dtype=jnp.float32,
    )

    def inverse(y):
        return bijection.inverse_and_log_det(y)[0]

    x = jax.vmap(inverse)(y)
    grad = jax.vmap(jax.grad(inverse))(y)
    np.testing.assert_allclose(
        jax.vmap(bijection.transform)(x), y, rtol=1e-5, atol=1e-6
    )
    # The derivative of the inverse is 1 / f'(x)
    fprime = 1 - a * (1 - x**2) / (1 + x**2) ** 2
    np.testing.assert_allclose(grad, 1 / fprime, rtol=1e-4)


DIPBIJ_B = [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0]
# rtol and atol for the round trip, and the tolerance for derivatives
DIPBIJ_TOL = {"float32": (2e-5, 2e-5, 1e-4), "float64": (1e-10, 1e-12, 1e-8)}


@pytest.fixture(params=["float32", "float64"])
def float_dtype(request):
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", request.param == "float64")
    try:
        yield jnp.dtype(request.param)
    finally:
        jax.config.update("jax_enable_x64", old)


def _dipbij(b, dtype):
    return eqx.tree_at(lambda bij: bij.b, DipBij(), jnp.asarray(b, dtype=dtype))


def _dipbij_forward(x, b):
    # Written out again on purpose, so expected values don't depend on DipBij
    a = jnp.tanh(b)
    return x - a * x / (1 + x**2)


def _dipbij_points(b, dtype):
    # A dense grid, large values, and points at and next to y = +-sqrt(3 * (1 - a)),
    # where one Cardano cube root argument is zero.
    root = np.sqrt(3 * (1 - np.tanh(b)))
    offsets = np.array([0.0, 1e-6, -1e-6, 1e-4, -1e-4, 1e-2, -1e-2])
    y = np.concatenate(
        [
            np.linspace(-10, 10, 2001),
            [-1e3, -1e2, 1e2, 1e3],
            root + offsets,
            -root + offsets,
        ]
    )
    return jnp.asarray(y, dtype=dtype)


@pytest.mark.parametrize("b", DIPBIJ_B)
def test_dipbij_inverse_round_trip(b, float_dtype):
    dtype = float_dtype
    rtol, atol, dtol = DIPBIJ_TOL[dtype.name]
    bijection = _dipbij(b, dtype)
    y = _dipbij_points(b, dtype)

    x, logdet = jax.vmap(bijection.inverse_and_log_det)(y)

    assert x.dtype == dtype
    assert np.isfinite(x).all() and np.isfinite(logdet).all()
    np.testing.assert_allclose(
        jax.vmap(bijection.transform)(x), y, rtol=rtol, atol=atol
    )
    fprime = jax.vmap(jax.grad(_dipbij_forward), in_axes=(0, None))(x, bijection.b)
    np.testing.assert_allclose(logdet, -np.log(fprime), rtol=dtol, atol=dtol)


@pytest.mark.parametrize("b", DIPBIJ_B)
def test_dipbij_inverse_gradients(b, float_dtype):
    # Gradients of the inverse and its log-det, with respect to y and to the
    # trained parameter b, checked against derivatives of the forward map.
    dtype = float_dtype
    _, _, dtol = DIPBIJ_TOL[dtype.name]
    b = jnp.asarray(b, dtype=dtype)
    y = _dipbij_points(b, dtype)

    def inverse(b, y):
        return _dipbij(b, dtype).inverse_and_log_det(y)

    def per_point(f, argnums=0):
        return jax.vmap(jax.grad(f, argnums), in_axes=(None, 0))

    x = jax.vmap(inverse, in_axes=(None, 0))(b, y)[0]
    dx_dy = per_point(lambda b, y: inverse(b, y)[0], 1)(b, y)
    dx_db = per_point(lambda b, y: inverse(b, y)[0])(b, y)
    dlogdet_dy = per_point(lambda b, y: inverse(b, y)[1], 1)(b, y)
    dlogdet_db = per_point(lambda b, y: inverse(b, y)[1])(b, y)

    def forward_grad(f, argnums):
        return jax.vmap(jax.grad(f, argnums), in_axes=(0, None))(x, b)

    fprime_fn = jax.grad(_dipbij_forward)
    fprime = forward_grad(_dipbij_forward, 0)
    df_db = forward_grad(_dipbij_forward, 1)
    fsecond = forward_grad(fprime_fn, 0)
    dfprime_db = forward_grad(fprime_fn, 1)

    for grad in [dx_dy, dx_db, dlogdet_dy, dlogdet_db]:
        assert np.isfinite(grad).all()
    # Implicit function theorem on f(x, b) = y, with log-det = -log f'(x, b)
    expected_dx_db = -df_db / fprime
    np.testing.assert_allclose(dx_dy, 1 / fprime, rtol=dtol, atol=dtol)
    np.testing.assert_allclose(dx_db, expected_dx_db, rtol=dtol, atol=dtol)
    np.testing.assert_allclose(dlogdet_dy, -fsecond / fprime**2, rtol=dtol, atol=dtol)
    np.testing.assert_allclose(
        dlogdet_db,
        -(dfprime_db + fsecond * expected_dx_db) / fprime,
        rtol=dtol,
        atol=dtol,
    )
