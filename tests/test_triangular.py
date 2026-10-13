import numpy as np
import pytest

jax = pytest.importorskip("jax")

import equinox as eqx
import jax.numpy as jnp

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
        {"transformer": "affine"},
        # Tangent SAS, located at the trailing `PositiveAffine`'s `loc`.
        {"transformer": "tangent_sas", "layers": 2},
        # Asymmetric SAS, the same.
        {"transformer": "asymmetric_sas", "layers": 2},
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


def test_no_location_skip_without_an_output_shift():
    """A chain ending in an inverted `AsymmetricAffine` only has an input
    shift, so its conditioners must stay plain MLPs."""
    transformer = make_transformer("asymmetric_affine")
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
        (4, None, {"transformer": "tangent_sas", "transformer_layers": 2}),
        (4, None, {"transformer": "sas", "transformer_layers": 2}),
        (4, None, {"transformer": "affine"}),
        (4, None, {"transformer": "asymmetric_sas"}),
        (
            4,
            None,
            {
                "transformer": "asymmetric_sas",
                "transformer_layers": 3,
                "log_gamma_bounds": (-1, 1),
            },
        ),
        (
            4,
            None,
            {
                "transformer": "tangent_sas",
                "transformer_layers": 2,
                "tangent_sas_fix_b": True,
            },
        ),
        (4, None, {"input_squash": 0.7}),
        (32, None, {"activation": jax.nn.gelu, "input_squash": 0.7}),
        (
            32,
            None,
            {
                "input_squash": 0.7,
                "transformer": "sas",
                "log_gamma_bounds": (-1, 1),
            },
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


@pytest.mark.parametrize("bounds", [None, (-1.0, 1.5)])
def test_asymmetric_sas(bounds):
    """Identity at zero; the inverse and both log dets against autodiff; one
    tail exponent per side; and ``C^2`` at the kink, where only the third
    derivative jumps."""
    from nutpie.normalizing_flow import AsymmetricSAS

    jax.config.update("jax_enable_x64", True)
    zero = AsymmetricSAS(0.0, 0.0, 0.0, 0.0, log_gamma_bounds=bounds)
    x = jnp.linspace(-30.0, 30.0, 13)
    y, log_det = jax.vmap(zero.transform_and_log_det)(x)
    np.testing.assert_allclose(y, x, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(log_det, 0.0, atol=1e-12)

    layer = AsymmetricSAS(0.3, -0.5, 0.9, -0.6, log_gamma_bounds=bounds)
    y, log_det = jax.vmap(layer.transform_and_log_det)(x)
    np.testing.assert_allclose(
        log_det, jnp.log(jax.vmap(jax.grad(layer.transform))(x)), rtol=1e-10
    )
    x_back, inverse_log_det = jax.vmap(layer.inverse_and_log_det)(y)
    np.testing.assert_allclose(x_back, x, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(inverse_log_det, -log_det, rtol=1e-10, atol=1e-12)

    # Tails: h(w) ~ w**gamma_pm on each side, the log-log slope far out.
    c, log_scale, log_gamma_pos, log_gamma_neg = layer._params()
    for sign, log_gamma in [(1.0, log_gamma_pos), (-1.0, log_gamma_neg)]:
        far = [c + sign * w * jnp.exp(-log_scale) for w in (1e8, 1e10)]
        logs = [jnp.log(jnp.abs(layer.transform(x))) for x in far]
        exponent = (logs[1] - logs[0]) / jnp.log(1e2)
        np.testing.assert_allclose(exponent, jnp.exp(log_gamma), rtol=1e-6)

    # Smoothness at the kink x = c, from either side.
    def derivatives(f, at, order):
        out = []
        for _ in range(order + 1):
            out.append(float(f(at)))
            f = jax.grad(f)
        return out

    kink = float(c)
    right = derivatives(layer.transform, kink + 1e-9, 3)
    left = derivatives(layer.transform, kink - 1e-9, 3)
    np.testing.assert_allclose(right[:3], left[:3], atol=1e-6)
    assert abs(right[3] - left[3]) > 0.1


@pytest.mark.parametrize(
    ("transformer", "layers", "n_params", "location"),
    [
        ("affine", 1, 2, True),
        ("sas", 1, 5, True),
        ("sas", 3, 13, True),
        ("tangent_sas", 2, 10, True),
        ("asymmetric_sas", 1, 4, True),
        ("asymmetric_sas", 3, 12, True),
        ("asymmetric_affine", 2, 6, False),
    ],
)
def test_make_transformer_kinds(transformer, layers, n_params, location):
    """Each kind's parameter count with `layers` layers, its location, and
    that it starts at the identity (but for the asymmetric affines, which
    start spread over ``[-2, 2]``)."""
    from paramax import unwrap

    jax.config.update("jax_enable_x64", True)
    elementwise = make_transformer(transformer, layers)
    arrays = eqx.filter(elementwise, eqx.is_inexact_array)
    assert sum(a.size for a in jax.tree.leaves(arrays)) == n_params
    assert (elementwise.location() is not None) == location
    if transformer != "asymmetric_affine":
        x = jnp.linspace(-5.0, 5.0, 11)
        y = jax.vmap(unwrap(elementwise.chain).transform)(x)
        np.testing.assert_allclose(y, x, rtol=1e-6, atol=1e-6)


def test_make_transformer_rejects():
    with pytest.raises(ValueError, match="unknown transformer"):
        make_transformer("contract")
    with pytest.raises(ValueError, match="exactly one layer"):
        make_transformer("affine", 2)
    with pytest.raises(ValueError, match="at least one layer"):
        make_transformer("sas", 0)
