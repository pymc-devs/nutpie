from importlib.util import find_spec

import numpy as np
import pytest

if find_spec("jax") is None or find_spec("flowjax") is None:
    pytest.skip("Skip flow influence tests", allow_module_level=True)

import equinox as eqx
import jax
import jax.numpy as jnp
from flowjax.utils import get_ravelled_pytree_constructor
from paramax import NonTrainable

from nutpie.flow_influence import fisher_influence, fisher_information
from nutpie.normalizing_flow import make_transformer


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


def test_fisher_information_affine(x64):
    """For y = mu + sigma x, the Fisher information in (mu, sigma) is
    diag(1 / sigma^2, 2 / sigma^2), whatever the parameterization."""
    transformer = make_transformer(
        affine_transformer=1, contract_transformer=0, asymmetric_transformer=0
    )
    constructor, num_params = get_ravelled_pytree_constructor(
        transformer,
        filter_spec=eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, NonTrainable),
    )
    theta = jnp.asarray(np.random.default_rng(0).normal(size=num_params))

    def loc_scale(theta):
        loc, log_scale = constructor(theta).transform_and_log_det(jnp.zeros(()))
        return jnp.stack([loc, jnp.exp(log_scale)])

    _, sigma = loc_scale(theta)
    J = jax.jacfwd(loc_scale)(theta)
    expected = J.T @ jnp.diag(jnp.array([1.0, 2.0]) / sigma**2) @ J
    np.testing.assert_allclose(
        fisher_information(constructor, theta), expected, atol=1e-12
    )


@pytest.mark.skipif(find_spec("pymc") is None, reason="needs pymc")
def test_fisher_influence(x64):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pymc as pm

    import nutpie
    from nutpie import transform_adapter

    with pm.Model() as model:
        mu = pm.Normal("mu")
        tau = pm.HalfNormal("tau")
        pm.Normal("theta", mu, tau, shape=5)
    compiled = nutpie.compile_pymc_model(
        model, backend="jax", gradient_backend="jax"
    ).with_transform_adapt(debug_save_bijection=True)

    transform_adapter._BIJECTION_TRACE.clear()
    trace = nutpie.sample(
        compiled,
        chains=1,
        adaptation="flow",
        tune=700,
        draws=200,
        store_unconstrained=True,
        progress_bar=False,
        seed=1,
    )
    _, bijection, _ = transform_adapter._BIJECTION_TRACE[-1]

    influence = fisher_influence(bijection, trace)
    assert influence.n_dim == 7
    assert influence.num_draws == 200
    assert influence.unconstrained_parameters == list(
        trace.sample_stats.unconstrained_parameter.values
    )
    position = np.empty_like(influence.order)
    position[influence.order] = np.arange(7)
    for kind in ["total", "location", "scale_shape"]:
        matrix = getattr(influence, kind).tocoo()
        assert matrix.nnz > 0
        assert np.all(np.isfinite(matrix.data)) and np.all(matrix.data >= 0)
        # Only parents, which come earlier in the flow order
        assert np.all(position[matrix.col] < position[matrix.row])

    variables = compiled._unconstrained_variables()
    axes = influence.plot(variables=variables)
    assert [ax.get_title() for ax in axes] == [
        "Fisher speed",
        "location only",
        "scale and shape",
    ]
    plt.close(axes[0].figure)
