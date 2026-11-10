from importlib.util import find_spec

import pytest

if find_spec("flowjax") is None:
    pytest.skip("Skip normalizing flow tests", allow_module_level=True)

import jax
import jax.numpy as jnp
import numpy as np

from nutpie.normalizing_flow import Contract2, _exp_asinh, make_flow

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
