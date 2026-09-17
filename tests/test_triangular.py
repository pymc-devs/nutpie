import numpy as np
import pytest

jax = pytest.importorskip("jax")

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from nutpie.normalizing_flow import _scale_last_layer, make_transformer  # noqa: E402
from nutpie.triangular import LocationSkipMlp, SparseTriangularMap  # noqa: E402


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
    "nn_width",
    [
        4,
        0,
        # More than one SIMD block of hidden units: exercised the forward
        # kernel's blocked loop on the parentless first variable.
        32,
    ],
)
def test_native_flow_transform_matches_jax(nn_width):
    """The sampler's native leapfrog transform against the JAX bijection: the
    whole `make_flow(kind="triangular")` flow -- permutation, map and affine --
    forward, log det, and the gradient pulled back with a unit log det
    cotangent."""
    import flowjax
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
        activation=jax.nn.softplus,
        nn_width=nn_width,
        zero_init=False,
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
    import flowjax
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
    (y_ref, log_det_ref), pull = jax.vjp(bijection.transform_and_log_det, jnp.asarray(z))
    (grad_z_ref,) = pull((jnp.asarray(grad_y), jnp.ones(())))
    y, log_det = native.transform_and_log_det(z)
    np.testing.assert_allclose(y, np.asarray(y_ref), rtol=1e-12)
    np.testing.assert_allclose(log_det, float(log_det_ref), rtol=1e-12)
    np.testing.assert_allclose(native.pullback(grad_y), np.asarray(grad_z_ref), rtol=1e-12)
