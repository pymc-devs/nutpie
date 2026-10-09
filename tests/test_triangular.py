import numpy as np
import pytest

jax = pytest.importorskip("jax")

import equinox as eqx
import jax.numpy as jnp
from paramax import unwrap

from nutpie.normalizing_flow import _scale_last_layer, make_transformer
from nutpie.triangular import LocationSkipMlp, SparseTriangularMap


def _banded(dim):
    blanket = np.zeros((dim, dim), dtype=bool)
    for i in range(1, dim):
        blanket[i, max(0, i - 2) : i] = True
    return blanket


@pytest.mark.parametrize(
    "transformer_kwargs",
    [
        # The default: two `Contract2` layers, located at the last one's `mu`.
        None,
        # Affine only, located at its `loc`.
        dict(affine_transformer=1, contract_transformer=0, asymmetric_transformer=0),
        # Tangent SAS, located at the trailing `PositiveAffine`'s `loc`.
        dict(
            contract_transformer=0, asymmetric_transformer=0, tangent_sas_transformer=2
        ),
    ],
)
def test_location_skip_is_the_conditional_mean(transformer_kwargs):
    """With the MLPs silenced, the skip alone must shift each variable by a
    linear function of its parents, in model space: ``x = y - w . y_parents``.

    `Contract2` at zero parameters is the identity up to its `eps`, hence the
    tolerance.
    """
    jax.config.update("jax_enable_x64", True)
    dim = 8
    transformer = (
        None if transformer_kwargs is None else make_transformer(**transformer_kwargs)
    )
    flow = SparseTriangularMap(
        jax.random.key(0), blanket=_banded(dim), n_buckets=2, transformer=transformer
    )
    assert all(isinstance(c, LocationSkipMlp) for c in flow.conditioners)

    rng = np.random.default_rng(0)
    conditioners = []
    weights = np.zeros((dim, dim))
    for bucket, conditioner in enumerate(flow.conditioners):
        members = np.asarray(flow.bucket_members[bucket])
        parents = np.asarray(flow.bucket_parent_indices[bucket])
        # Padded parent slots read a constant zero, so their weight is inert.
        skip = rng.normal(size=conditioner.skip.shape)
        for local, variable in enumerate(members):
            for slot, parent in enumerate(parents[local]):
                if parent < dim:
                    weights[variable, parent] = skip[local, slot]
        mlp = _scale_last_layer(conditioner.mlp, 0.0)
        conditioners.append(
            eqx.tree_at(
                lambda c: (c.mlp, c.skip), conditioner, (mlp, jnp.asarray(skip))
            )
        )
    flow = eqx.tree_at(lambda f: f.conditioners, flow, tuple(conditioners))

    y = rng.normal(size=dim)
    x, _ = flow.inverse_and_log_det(jnp.asarray(y))
    np.testing.assert_allclose(np.asarray(x), y - weights @ y, atol=1e-3)


@pytest.mark.parametrize("feature_degree", [None, 2])
def test_parent_scores_and_their_gauss_newton_rows(feature_degree):
    """`parent_scores` against the Jacobian of ``log q(y_i | y_pa)`` from the
    full map, and the regularizer rows of `gauss_newton_factors` against the
    Jacobian of the scaled scores over each conditioner's weights."""
    jax.config.update("jax_enable_x64", True)
    dim = 8
    flow = unwrap(
        SparseTriangularMap(
            jax.random.key(0),
            blanket=_banded(dim),
            n_buckets=2,
            feature_degree=feature_degree,
        )
    )
    rng = np.random.default_rng(0)
    # Nonzero skips and transformer shapes, so every score term contributes.
    flow = eqx.tree_at(
        lambda f: f.conditioners,
        flow,
        jax.tree.map(
            lambda a: (
                a + 0.3 * jnp.asarray(rng.normal(size=a.shape))
                if eqx.is_inexact_array(a)
                else a
            ),
            flow.conditioners,
        ),
    )
    y = jnp.asarray(rng.normal(size=dim))

    def log_q(y):
        x, _ = flow.inverse_and_log_det(y)
        own_derivative = jnp.diagonal(jax.jacfwd(lambda y: flow.inverse(y))(y))
        return jnp.log(own_derivative) - x**2 / 2

    expected = np.asarray(jax.jit(jax.jacfwd(log_q))(y))
    scores = eqx.filter_jit(lambda f, y: f.parent_scores(y))(flow, y)
    for bucket, score in enumerate(scores):
        members = np.asarray(flow.bucket_members[bucket])
        parents = np.asarray(flow.bucket_parent_indices[bucket])
        for local, variable in enumerate(members):
            for slot, parent in enumerate(parents[local]):
                want = expected[variable, parent] if parent < dim else 0.0
                np.testing.assert_allclose(score[local, slot], want, atol=1e-10)

    lam = 0.7
    grad = jnp.asarray(rng.normal(size=dim))
    factors = eqx.filter_jit(
        lambda f, y, g: f.gauss_newton_factors(y, g, fisher_regularization=lam)
    )(flow, y, grad)
    plain = eqx.filter_jit(lambda f, y, g: f.gauss_newton_factors(y, g))(flow, y, grad)

    def scaled_scores(conditioners):
        tmap = eqx.tree_at(lambda f: f.conditioners, flow, conditioners)
        return [jnp.sqrt(lam) * s for s in tmap.parent_scores(y)]

    arrays, static = eqx.partition(flow.conditioners, eqx.is_inexact_array)
    jac = jax.jit(jax.jacrev(lambda a: scaled_scores(eqx.combine(a, static))))(arrays)
    for bucket, (V, V_plain) in enumerate(zip(factors, plain)):
        n_block = V_plain.shape[1]
        np.testing.assert_allclose(V[:, :n_block], V_plain)
        if V.shape[1] == n_block:  # no parents
            continue
        n_slots = V.shape[1] - n_block
        for local in range(V.shape[0]):
            # Jacobian of this member's scores over its own weights, in
            # `ravel_pytree` order
            rows = jnp.concatenate(
                [
                    leaf[local, :, local].reshape(n_slots, -1)
                    for leaf in jax.tree.leaves(jac[bucket][bucket])
                ],
                axis=1,
            )
            np.testing.assert_allclose(V[local, n_block:], rows, atol=1e-10)


def test_fisher_regularization_residuals():
    """`FisherLoss.residuals` with `fisher_regularization`: the Fisher part
    has the unregularized norm per draw (it skips the final permutation), and
    the rest are the scaled parent scores of the map's input."""
    import flowjax

    from nutpie.normalizing_flow import make_flow
    from nutpie.transform_adapter import FisherLoss, _triangular_map_input

    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(0)
    dim, n = 7, 20
    x = 0.5 + rng.normal(size=(n, dim)) * np.exp(rng.normal(size=dim))
    g = -rng.normal(size=(n, dim))
    bijection = make_flow(
        0,
        x,
        g,
        n_layers=1,
        kind="triangular",
        sparsity=_banded(dim),
        order=rng.permutation(dim),
        zero_init=False,
        # Three parent counts in two buckets: covers padded slots.
        n_buckets=2,
    )
    flow = flowjax.flows.Transformed(
        flowjax.distributions.StandardNormal((dim,)), bijection
    )
    params, static = eqx.partition(flow, eqx.is_inexact_array)
    params = jax.tree.map(
        lambda leaf: leaf + 0.1 * jnp.asarray(rng.normal(size=leaf.shape)), params
    )
    data = (jnp.asarray(x), jnp.asarray(g), jnp.zeros(n))

    lam = 0.3
    residuals = eqx.filter_jit(
        lambda loss, p: loss.residuals(p, static, *data) * np.sqrt(n)
    )
    plain = residuals(FisherLoss(), params)
    regularized = residuals(FisherLoss(fisher_regularization=lam), params)
    np.testing.assert_allclose(
        (regularized[:, :dim] ** 2).sum(1), (plain**2).sum(1), rtol=1e-10
    )

    @eqx.filter_jit
    def scores(p, draw, grad):
        tmap, y, _ = _triangular_map_input(
            unwrap(eqx.combine(p, static)), draw, grad, 0.0
        )
        return jnp.concatenate([s.ravel() for s in tmap.parent_scores(y)])

    for i in range(3):
        np.testing.assert_allclose(
            regularized[i, dim:],
            np.sqrt(lam) * scores(params, data[0][i], data[1][i]),
            rtol=1e-10,
            atol=1e-12,
        )


def test_no_location_skip_without_an_output_shift():
    """A chain ending in an inverted `AsymmetricAffine` only has an input
    shift, so its conditioners must stay plain MLPs."""
    transformer = make_transformer(
        affine_transformer=0, contract_transformer=0, asymmetric_transformer=1
    )
    assert transformer.location() is None
    flow = SparseTriangularMap(
        jax.random.key(0), blanket=_banded(6), n_buckets=2, transformer=transformer
    )
    assert not any(isinstance(c, LocationSkipMlp) for c in flow.conditioners)


def test_location_skip_can_be_disabled():
    flow = SparseTriangularMap(
        jax.random.key(0), blanket=_banded(6), n_buckets=2, location_skip=False
    )
    assert not any(isinstance(c, LocationSkipMlp) for c in flow.conditioners)


@pytest.mark.parametrize(
    ("nn_width", "feature_degree", "transformer_kwargs"),
    [
        (4, None, {}),
        (0, None, {}),
        # More than one SIMD block of hidden units: exercised the forward
        # kernel's blocked loop on the parentless first variable.
        (32, None, {}),
        (4, 3, {}),
        (32, 1, {}),
        (4, None, dict(tangent_sas_transformer=2)),
        (4, None, dict(contract_transformer=1, tangent_sas_transformer=1)),
        (4, None, dict(tangent_sas_transformer=2, tangent_sas_fix_b=True)),
        (4, None, dict(input_squash=0.7)),
        (32, None, dict(activation=jax.nn.gelu, input_squash=0.7)),
        (
            32,
            None,
            dict(input_squash=0.7, contract_transformer=1, log_gamma_bounds=(-1, 1)),
        ),
    ],
)
def test_native_flow_transform_matches_jax(
    nn_width, feature_degree, transformer_kwargs
):
    """The sampler's native leapfrog transform against the JAX bijection: the
    whole `make_flow(kind="triangular")` flow -- permutation, map and affine --
    forward, log det, and the gradient pulled back with a unit log det
    cotangent. With parent features, their marginal maps are fitted to the
    draws and then perturbed like every other parameter."""
    from paramax import unwrap

    from nutpie._lib import FlowTransform
    from nutpie.normalizing_flow import make_flow
    from nutpie.triangular_rust import flow_transform_layout

    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(0)
    dim = 7
    blanket = _banded(dim)
    x = 0.5 + rng.normal(size=(50, dim)) * np.exp(rng.normal(size=dim))
    g = -rng.normal(size=(50, dim))
    bijection = make_flow(
        0,
        x,
        g,
        n_layers=1,
        kind="triangular",
        sparsity=blanket,
        order=rng.permutation(dim),
        nn_width=nn_width,
        zero_init=False,
        feature_degree=feature_degree,
        n_buckets=2,
        **{"activation": jax.nn.softplus, **transformer_kwargs},
    )
    # Off the initialization, so every parameter matters.
    params, static = eqx.partition(bijection, eqx.is_inexact_array)
    params = jax.tree.map(
        lambda leaf: leaf + 0.1 * jnp.asarray(rng.normal(size=leaf.shape)), params
    )
    bijection = unwrap(eqx.combine(params, static))

    native = FlowTransform(flow_transform_layout(bijection))
    for _ in range(3):
        z = rng.normal(size=dim)
        grad_y = rng.normal(size=dim)
        (y_ref, log_det_ref), pull = jax.vjp(
            bijection.transform_and_log_det, jnp.asarray(z)
        )
        (grad_z_ref,) = pull((jnp.asarray(grad_y), jnp.ones(())))

        y, log_det = native.transform_and_log_det(z)
        grad_z = native.pullback(grad_y)
        np.testing.assert_allclose(y, np.asarray(y_ref), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(log_det, float(log_det_ref), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(
            grad_z, np.asarray(grad_z_ref), rtol=1e-9, atol=1e-10
        )


def test_native_flow_transform_diagonal_flow():
    """The diagonal-only flow of the early windows runs natively too."""
    from paramax import unwrap

    from nutpie._lib import FlowTransform
    from nutpie.normalizing_flow import make_flow
    from nutpie.triangular_rust import flow_transform_layout

    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(0)
    x = 1.0 + rng.normal(size=(20, 3)) * np.array([0.1, 1.0, 10.0])
    g = rng.normal(size=(20, 3))
    bijection = unwrap(make_flow(0, x, g, n_layers=0))
    native = FlowTransform(flow_transform_layout(bijection))

    z, grad_y = rng.normal(size=3), rng.normal(size=3)
    (y_ref, log_det_ref), pull = jax.vjp(
        bijection.transform_and_log_det, jnp.asarray(z)
    )
    (grad_z_ref,) = pull((jnp.asarray(grad_y), jnp.ones(())))
    y, log_det = native.transform_and_log_det(z)
    np.testing.assert_allclose(y, np.asarray(y_ref), rtol=1e-12)
    np.testing.assert_allclose(log_det, float(log_det_ref), rtol=1e-12)
    np.testing.assert_allclose(
        native.pullback(grad_y), np.asarray(grad_z_ref), rtol=1e-12
    )


def test_tangent_sas():
    """Identity at zero, ``S(0) = 0`` and ``S'(0) = 1`` at any parameters,
    the inverse, and both log dets against autodiff."""
    from nutpie.normalizing_flow import TangentSAS

    jax.config.update("jax_enable_x64", True)
    zero = TangentSAS(0.0, 0.0, 0.0, 0.0)
    x = jnp.linspace(-30.0, 30.0, 13)
    y, log_det = jax.vmap(zero.transform_and_log_det)(x)
    np.testing.assert_allclose(y, x, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(log_det, 0.0, atol=1e-12)

    layer = TangentSAS(0.7, -0.4, 1.3, -0.6)
    centre, slope = jax.value_and_grad(layer.transform)(jnp.asarray(0.7))
    np.testing.assert_allclose(centre, 0.7, rtol=1e-12)
    np.testing.assert_allclose(slope, 1.0, rtol=1e-12)

    y, log_det = jax.vmap(layer.transform_and_log_det)(x)
    np.testing.assert_allclose(
        log_det, jnp.log(jax.vmap(jax.grad(layer.transform))(x)), rtol=1e-10
    )
    x_back, inverse_log_det = jax.vmap(layer.inverse_and_log_det)(y)
    np.testing.assert_allclose(x_back, x, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(inverse_log_det, -log_det, rtol=1e-10, atol=1e-12)
