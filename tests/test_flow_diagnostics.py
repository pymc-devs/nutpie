from importlib.util import find_spec

import numpy as np
import pytest

if find_spec("jax") is None or find_spec("flowjax") is None:
    pytest.skip("Skip flow diagnostics tests", allow_module_level=True)

import equinox as eqx
import jax
import jax.numpy as jnp
from flowjax.utils import get_ravelled_pytree_constructor
from paramax import NonTrainable

from nutpie.flow_diagnostics import (
    _quadrature,
    conditioner_capacity,
    fisher_influence,
    fisher_information,
    latent_velocities,
)
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


def test_velocity_parity_affine(x64):
    """A location change has an even latent velocity, a scale change an odd
    one, which is what splits the speed into its two parts."""
    transformer = make_transformer(
        affine_transformer=1, contract_transformer=0, asymmetric_transformer=0
    )
    constructor, num_params = get_ravelled_pytree_constructor(
        transformer,
        filter_spec=eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, NonTrainable),
    )
    nodes, _ = _quadrature(16, jnp.float64)
    np.testing.assert_allclose(nodes[::-1], -nodes)
    V, _ = latent_velocities(constructor, jnp.zeros(num_params), nodes)
    parities = sorted(
        "even" if np.allclose(v, v[::-1]) else "odd"
        for v in np.asarray(V).T
        if not np.allclose(v, 0)
    )
    assert parities == ["even", "odd"]


@pytest.fixture(scope="module")
def fitted_flow():
    """A compiled model, a trace with unconstrained draws and the last
    fitted triangular flow."""
    if find_spec("pymc") is None:
        pytest.skip("needs pymc")
    import pymc as pm

    import nutpie
    from nutpie import transform_adapter

    with jax.enable_x64(True):
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
            store_gradient=True,
            progress_bar=False,
            seed=1,
        )
        _, bijection, _ = transform_adapter._BIJECTION_TRACE[-1]
    return compiled, trace, bijection


def test_fisher_influence(x64, fitted_flow):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    compiled, trace, bijection = fitted_flow
    influence = fisher_influence(bijection, trace)
    assert influence.n_dim == 7
    assert influence.num_draws == 200
    assert influence.unconstrained_parameters == list(
        trace.sample_stats.unconstrained_parameter.values
    )
    position = np.empty_like(influence.order)
    position[influence.order] = np.arange(7)
    for kind in ["total", "location_skew", "scale_tails"]:
        matrix = getattr(influence, kind).tocoo()
        assert matrix.nnz > 0
        assert np.all(np.isfinite(matrix.data)) and np.all(matrix.data >= 0)
        # Only parents, which come earlier in the flow order
        assert np.all(position[matrix.col] < position[matrix.row])

    # The squares of the two parts add up to the total
    np.testing.assert_allclose(
        (influence.location_skew**2 + influence.scale_tails**2).toarray(),
        (influence.total**2).toarray(),
        rtol=1e-10,
    )

    variables = compiled._unconstrained_variables()
    axes = influence.plot(variables=variables)
    # Labelled with the names, in the flow order
    labels = [label.get_text() for label in axes[0].get_yticklabels()]
    assert labels == [influence.unconstrained_parameters[i] for i in influence.order]
    assert [ax.get_title() for ax in axes] == [
        "Fisher speed",
        "location and skew",
        "scale and tails",
    ]
    plt.close(axes[0].figure)


def test_conditioner_capacity(x64, fitted_flow):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, trace, bijection = fitted_flow
    capacity = conditioner_capacity(bijection, trace)
    assert capacity.n_dim == 7
    assert capacity.num_draws == 200
    names = list(trace.sample_stats.unconstrained_parameter.values)
    eigenvalues = capacity.eigenvalues
    assert eigenvalues.shape == (7, capacity.width)
    assert list(eigenvalues.index) == names
    assert eigenvalues.index.name == "unconstrained_parameter"
    assert list(capacity.num_parents.index) == names
    assert np.all(np.isfinite(eigenvalues)) and np.all(eigenvalues >= 0)
    assert np.all(np.diff(eigenvalues, axis=1) <= 0)
    # Without parents the hidden layer is constant and does nothing.
    roots = capacity.num_parents == 0
    assert roots.any()
    np.testing.assert_allclose(eigenvalues[roots], 0.0, atol=1e-12)
    effective = capacity.effective_width()
    assert list(effective.index) == names
    assert np.all((effective >= 0) & (effective <= capacity.width))

    # The same draws as an array give the same result, without names.
    draws = trace.sample_stats.unconstrained_draw.values.reshape(-1, 7)
    gradients = trace.sample_stats.gradient.values.reshape(-1, 7)
    with pytest.raises(ValueError, match="gradients"):
        conditioner_capacity(bijection, draws)
    from_array = conditioner_capacity(bijection, draws, gradients=gradients)
    assert list(from_array.eigenvalues.index) == list(range(7))
    np.testing.assert_allclose(from_array.eigenvalues, eigenvalues)

    axes = capacity.plot(highlight=2)
    assert len(axes) == 2
    legend = [text.get_text() for text in axes[0].get_legend().get_texts()]
    assert len(legend) == 3 and legend[-1] == "threshold"
    assert all(label.split(" ")[0] in names for label in legend[:2])
    plt.close(axes[0].figure)


def test_conditioner_capacity_zero_output(x64, fitted_flow):
    """A conditioner whose hidden layer does not reach the output uses none
    of its width."""
    from paramax import unwrap

    _, trace, bijection = fitted_flow
    bijection = unwrap(bijection)
    where = lambda b: b.bijections[0].bijections[0].inner.conditioners
    conditioners = where(bijection)
    last_weight = lambda c: c.mlp.layers[-1].weight
    silenced = [
        eqx.tree_at(last_weight, c, jnp.zeros_like(last_weight(c)))
        for c in conditioners
    ]
    bijection = eqx.tree_at(where, bijection, tuple(silenced))
    capacity = conditioner_capacity(bijection, trace)
    np.testing.assert_allclose(capacity.eigenvalues, 0.0, atol=1e-12)


def test_conditioner_capacity_matches_residuals(x64, fitted_flow):
    """The curvature from the Gauss-Newton factors equals the one from
    differentiating the fit's residuals directly, for every conditioner."""
    from flowjax import bijections
    from jax.flatten_util import ravel_pytree
    from paramax import unwrap

    from nutpie.flow_diagnostics import (
        _gate_directions,
        _hidden_width,
        _unit_features,
    )
    from nutpie.transform_adapter import inverse_gradient_and_val

    _, trace, bijection = fitted_flow
    capacity = conditioner_capacity(bijection, trace)
    draws = jnp.asarray(trace.sample_stats.unconstrained_draw.values.reshape(-1, 7))
    grads = jnp.asarray(trace.sample_stats.gradient.values.reshape(-1, 7))

    flow = unwrap(bijection)
    where = lambda b: b.bijections[0].bijections[0].inner
    tmap = where(flow)
    sandwich = flow.bijections[0].bijections[0]
    to_map = jax.vmap(
        lambda d: bijections.Invert(sandwich.outer).inverse(
            flow.bijections[1].inverse(d)
        )
    )
    y = to_map(draws)
    y_padded = jnp.concatenate([y, jnp.zeros((len(y), 1))], axis=1)

    def residuals(flow):
        def one(draw, grad):
            x, g, _ = inverse_gradient_and_val(flow, draw, grad, jnp.zeros(()))
            return x + g

        return jax.vmap(one)(draws, grads)

    order = capacity.order
    checked = 0
    for bucket, conditioner in enumerate(tmap.conditioners):
        arrays, static = eqx.partition(conditioner, eqx.is_inexact_array)
        parent_indices = np.asarray(tmap.bucket_parent_indices[bucket])
        width = _hidden_width(conditioner)
        for m, member in enumerate(np.asarray(tmap.bucket_members[bucket])):
            one = eqx.combine(jax.tree.map(lambda a: a[m], arrays), static)
            parents = y_padded[:, parent_indices[m]]
            indices = tmap.bucket_parent_indices[bucket][m]
            mean = jax.vmap(lambda p: _unit_features(tmap, one)(p, indices))(
                parents
            ).mean(0)
            directions = _gate_directions(one, mean)
            _, unravel = ravel_pytree(eqx.partition(one, eqx.is_inexact_array)[0])

            def perturbed(gates):
                change, _ = eqx.partition(
                    unravel(directions @ gates), eqx.is_inexact_array
                )
                new = jax.tree.map(lambda a, d: a.at[m].add(d), arrays, change)
                conditioners = list(tmap.conditioners)
                conditioners[bucket] = eqx.combine(new, static)
                return residuals(
                    eqx.tree_at(
                        lambda b: where(b).conditioners, flow, tuple(conditioners)
                    )
                )

            J = jax.jacfwd(perturbed)(jnp.zeros_like(mean))  # (draws, dim, units)
            gram = np.einsum("dku,dkv->uv", J, J) / len(draws)
            hidden, skip = gram[:width, :width], gram[:width, width:]
            if skip.size:
                hidden = hidden - skip @ np.linalg.pinv(
                    gram[width:, width:], rcond=1e-10, hermitian=True
                ) @ skip.T
            expected = np.maximum(np.linalg.eigvalsh(hidden)[::-1], 0.0)
            np.testing.assert_allclose(
                capacity.eigenvalues.iloc[order[member]],
                expected,
                rtol=1e-6,
                atol=1e-10 * max(expected.max(), 1.0),
            )
            checked += 1
    assert checked == 7
