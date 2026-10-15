from importlib.util import find_spec

import numpy as np
import pytest

from nutpie.flow_diagnostics import conditioner_capacity, fisher_influence
from nutpie.triangular_flow import build_triangular_flow


def affine_flow(rng, dim=4):
    """A dense flow with an affine transformer, a constant and a linear
    conditioner on the location and the scale, random weights, and draws."""
    positions = rng.normal(size=(64, dim)) * np.arange(1, dim + 1)
    gradients = -positions / np.arange(1, dim + 1) ** 2
    flow = build_triangular_flow(
        0,
        positions,
        gradients,
        sparsity=np.ones((dim, dim), dtype=bool),
        order=rng.permutation(dim),
        transformer="affine",
        conditioners=[
            {"kind": "constant", "target": "all"},
            {"kind": "linear", "target": ["location", "scale"]},
        ],
    )
    flow = flow.with_theta(rng.normal(size=flow.n_params) * 0.5)
    return flow, positions, gradients


def test_fisher_influence_affine():
    """For ``y = loc + sigma x`` with ``loc = c + w . y_parents`` and
    ``sigma = exp(asinh(s))``, ``s = d + v . y_parents``, the squared speeds
    are ``w_j^2 / sigma^2`` (location) and ``2 v_j^2 / (1 + s^2)``
    (scale)."""
    rng = np.random.default_rng(0)
    flow, positions, gradients = affine_flow(rng)
    influence = fisher_influence(flow, positions)
    assert influence.num_draws == 64
    assert influence.unconstrained_parameters is None

    y, _ = flow.undo_affine(positions, gradients)
    theta = flow.theta
    order = influence.order
    expected = {"location_skew": np.zeros((4, 4)), "scale_tails": np.zeros((4, 4))}
    for i in range(1, 4):
        (c_start, _), (l_start, _) = flow.weight_ranges(i)
        _, scale0 = theta[c_start : c_start + 2]
        w = theta[l_start : l_start + i]
        v = theta[l_start + i : l_start + 2 * i]
        s = scale0 + y[:, :i] @ v
        sigma = np.exp(np.arcsinh(s))
        for j in range(i):
            row, col = order[i], order[j]
            expected["location_skew"][row, col] = np.sqrt(np.mean(w[j] ** 2 / sigma**2))
            expected["scale_tails"][row, col] = np.sqrt(
                np.mean(2 * v[j] ** 2 / (1 + s**2))
            )

    for kind, matrix in expected.items():
        np.testing.assert_allclose(
            getattr(influence, kind).toarray(), matrix, rtol=1e-10
        )
    np.testing.assert_allclose(
        influence.total.toarray() ** 2,
        expected["location_skew"] ** 2 + expected["scale_tails"] ** 2,
        rtol=1e-10,
    )


def test_conditioner_capacity_needs_mlp():
    flow, positions, gradients = affine_flow(np.random.default_rng(1))
    with pytest.raises(RuntimeError, match="no MLP"):
        conditioner_capacity(flow, positions, gradients=gradients)


def test_draws_trimmed_to_simd_width():
    from nutpie._lib import FisherResiduals

    flow, positions, _ = affine_flow(np.random.default_rng(2))
    width = FisherResiduals.simd_width()
    influence = fisher_influence(flow, positions[: 2 * width + 1])
    assert influence.num_draws == 2 * width
    with pytest.raises(ValueError, match="dimensions"):
        fisher_influence(flow, positions[:, :3])


@pytest.fixture(scope="module")
def fitted_flow():
    """A compiled model, a trace with unconstrained draws and the last
    fitted triangular flow."""
    if find_spec("pymc") is None or find_spec("jax") is None:
        pytest.skip("needs pymc and jax")
    import jax
    import pymc as pm

    import nutpie
    from nutpie import triangular_adapter

    with jax.enable_x64(True):
        with pm.Model() as model:
            mu = pm.Normal("mu")
            tau = pm.HalfNormal("tau")
            pm.Normal("theta", mu, tau, shape=5)
        compiled = nutpie.compile_pymc_model(
            model, backend="jax", gradient_backend="jax"
        ).with_transform_adapt(debug_save_bijection=True)

        triangular_adapter._BIJECTION_TRACE.clear()
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
    _, flow, _ = triangular_adapter._BIJECTION_TRACE[-1]
    return compiled, trace, flow


def test_fisher_influence(fitted_flow):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    compiled, trace, flow = fitted_flow
    influence = fisher_influence(flow, trace)
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


def test_conditioner_capacity(fitted_flow):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, trace, flow = fitted_flow
    capacity = conditioner_capacity(flow, trace)
    assert capacity.n_dim == 7
    assert capacity.num_draws == 200
    # make_transform_adapter's default nn_width
    assert capacity.width == 4
    names = list(trace.sample_stats.unconstrained_parameter.values)
    eigenvalues = capacity.eigenvalues
    assert eigenvalues.shape == (7, capacity.width)
    assert list(eigenvalues.index) == names
    assert eigenvalues.index.name == "unconstrained_parameter"
    assert list(capacity.num_parents.index) == names
    assert np.all(np.isfinite(eigenvalues)) and np.all(eigenvalues >= 0)
    assert np.all(np.diff(eigenvalues, axis=1) <= 0)
    # Without parents there are no hidden units, and nothing to drop.
    roots = capacity.num_parents == 0
    assert roots.any()
    np.testing.assert_allclose(eigenvalues[roots], 0.0)
    assert (eigenvalues[~roots] > 0).any(axis=None)
    effective = capacity.effective_width()
    assert list(effective.index) == names
    assert np.all((effective >= 0) & (effective <= capacity.width))

    # The same draws as an array give the same result, without names.
    draws = trace.sample_stats.unconstrained_draw.values.reshape(-1, 7)
    gradients = trace.sample_stats.gradient.values.reshape(-1, 7)
    with pytest.raises(ValueError, match="gradients"):
        conditioner_capacity(flow, draws)
    from_array = conditioner_capacity(flow, draws, gradients=gradients)
    assert list(from_array.eigenvalues.index) == list(range(7))
    np.testing.assert_allclose(from_array.eigenvalues, eigenvalues)

    axes = capacity.plot(highlight=2)
    assert len(axes) == 2
    legend = [text.get_text() for text in axes[0].get_legend().get_texts()]
    assert len(legend) == 3 and legend[-1] == "threshold"
    assert all(label.split(" ")[0] in names for label in legend[:2])
    plt.close(axes[0].figure)


def test_conditioner_capacity_zero_output(fitted_flow):
    """A conditioner whose hidden layer does not reach the output uses none
    of its width."""
    _, trace, flow = fitted_flow
    n_coefs = len(flow.theta[slice(*flow.weight_ranges(0)[0])])
    theta = flow.theta.copy()
    for i in range(flow.n_var):
        n_parents = flow.n_parents[i]
        start, stop = flow.weight_ranges(i)[-1]
        if n_parents == 0:
            continue
        # Per unit W1[u, :], b1[u], W2[:, u]
        units = theta[start:stop].reshape(-1, n_parents + 1 + n_coefs)
        units[:, n_parents + 1 :] = 0.0
    capacity = conditioner_capacity(flow.with_theta(theta), trace)
    np.testing.assert_allclose(capacity.eigenvalues, 0.0, atol=1e-12)
