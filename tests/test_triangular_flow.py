"""`nutpie.triangular_flow`, the triangular flow built in Rust, against the
JAX flow from `make_flow(kind="triangular")`."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import equinox as eqx
import jax.numpy as jnp

from nutpie.triangular_flow import build_triangular_flow, from_jax, transformer_spec

DIM = 7


def _banded(dim, order=None):
    """A banded blanket, closed under elimination in `order` as a symbolic
    factorization fills it (the LM residuals require that)."""
    import scipy.sparse as sp

    from nutpie.sparsity import _fill

    blanket = np.zeros((dim, dim), dtype=bool)
    for i in range(1, dim):
        blanket[i, max(0, i - 2) : i] = True
    blanket = blanket | blanket.T
    if order is None:
        return blanket
    return _fill(sp.csr_array(blanket), np.asarray(order)).toarray()


def _draws(n_draw=16, seed=0):
    rng = np.random.default_rng(seed)
    x = 0.5 + rng.normal(size=(n_draw, DIM)) * np.exp(rng.normal(size=DIM))
    g = -rng.normal(size=(n_draw, DIM))
    return x, g


TRANSFORMERS = [
    {},
    {"contract_transformer": 1},
    {"contract_transformer": 2, "log_gamma_bounds": (-1.0, 1.0)},
    {"tangent_sas_transformer": 1},
    {"contract_transformer": 1, "tangent_sas_transformer": 1},
    {"tangent_sas_transformer": 2, "tangent_sas_fix_b": True},
]


@pytest.mark.parametrize("kwargs", TRANSFORMERS)
def test_transformer_spec_matches_the_probe(kwargs):
    """The spec written out by hand is the one probed from the JAX
    transformer: the same layers, fields, indices, offsets and location."""
    from nutpie.normalizing_flow import make_sparse_triangular_map
    from nutpie.triangular_layout import _probe_transformer, transformer_dicts

    layer = make_sparse_triangular_map(
        jax.random.key(0),
        DIM,
        order=np.arange(DIM),
        sparsity=_banded(DIM),
        nn_width=3,
        activation=jax.nn.softplus,
        **kwargs,
    )
    tmap = layer.bijections[0].inner
    n_par = int(tmap.conditioners[0].mlp.layers[1].out_features)
    probed = transformer_dicts(_probe_transformer(tmap.transformer_constructor, n_par))

    specs, ours_n_par, location = transformer_spec(**kwargs)
    assert ours_n_par == n_par
    assert location == int(tmap.conditioners[0].location_index)
    assert transformer_dicts(specs) == probed


def _jax_flow(seed=0, **kwargs):
    """A JAX triangular flow off its initialization, so every parameter
    matters."""
    from paramax import unwrap

    from nutpie.normalizing_flow import make_flow

    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(seed)
    x, g = _draws()
    settings = {
        "nn_width": 4,
        "activation": jax.nn.softplus,
        "contract_transformer": 1,
        "tangent_sas_transformer": 1,
        **kwargs,
    }
    order = rng.permutation(DIM)
    bijection = make_flow(
        seed,
        x,
        g,
        n_layers=1,
        kind="triangular",
        sparsity=_banded(DIM, order),
        order=order,
        zero_init=False,
        n_buckets=2,
        **settings,
    )
    params, static = eqx.partition(bijection, eqx.is_inexact_array)
    params = jax.tree.map(
        lambda leaf: leaf + 0.1 * jnp.asarray(rng.normal(size=leaf.shape)), params
    )
    return unwrap(eqx.combine(params, static))


FLOWS = [
    {},
    {"activation": jax.nn.gelu, "input_squash": 0.7, "log_gamma_bounds": (-1, 1)},
    {"nn_width": 0, "tangent_sas_transformer": 0, "contract_transformer": 2},
]


@pytest.mark.parametrize("kwargs", FLOWS)
def test_from_jax_matches_the_jax_flow(kwargs):
    """Everything the Rust flow computes, on the parameters of a JAX flow."""
    import flowjax

    from nutpie.transform_adapter import FisherLoss, _inv_transform
    from nutpie.triangular_lm import make_residuals, map_data, pack_params

    bijection = _jax_flow(**kwargs)
    flow = from_jax(bijection)
    x, g = _draws(seed=1)

    # The draws with the affine undone, in the triangular order, and the
    # residuals there.
    tmap, y, gy = map_data(
        flowjax.flows.Transformed(
            flowjax.distributions.StandardNormal((DIM,)), bijection
        ),
        x,
        g,
    )
    ours_y, ours_g = flow.undo_affine(x, g)
    np.testing.assert_allclose(ours_y, y, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ours_g, gy, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(flow.theta, pack_params(tmap))
    theta = np.asarray(flow.theta)
    reference = make_residuals(tmap, y, gy).residuals(theta, linearize=False)
    np.testing.assert_allclose(
        flow.residuals(x, g).residuals(theta, linearize=False), reference
    )

    # The Fisher divergence: `FisherLoss` returns its log.
    full = flowjax.flows.Transformed(
        flowjax.distributions.StandardNormal((DIM,)), bijection
    )
    params, static = eqx.partition(full, eqx.is_inexact_array)
    log_f = FisherLoss()(params, static, x, g, np.zeros(len(x)))
    np.testing.assert_allclose(np.log(flow.fisher_divergence(x, g)), log_f, rtol=1e-10)

    # One draw back to the base space.
    for k in range(3):
        log_det, z, grad_z = _inv_transform(
            bijection, jnp.asarray(x[k]), jnp.asarray(g[k])
        )
        ours_log_det, ours_z, ours_grad_z = flow.inv_transform(x[k], g[k])
        np.testing.assert_allclose(ours_z, np.asarray(z), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(
            ours_grad_z, np.asarray(grad_z), rtol=1e-9, atol=1e-10
        )
        np.testing.assert_allclose(ours_log_det, float(log_det), rtol=1e-10, atol=1e-10)

    # The sampler's direction, as `test_native_flow_transform_matches_jax`.
    native = flow.flow_transform()
    rng = np.random.default_rng(2)
    for _ in range(3):
        z = rng.normal(size=DIM)
        grad_y = rng.normal(size=DIM)
        (y_ref, log_det_ref), pull = jax.vjp(
            bijection.transform_and_log_det, jnp.asarray(z)
        )
        (grad_z_ref,) = pull((jnp.asarray(grad_y), jnp.ones(())))
        ours_y, ours_log_det = native.transform_and_log_det(z)
        np.testing.assert_allclose(ours_y, np.asarray(y_ref), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(ours_log_det, float(log_det_ref), rtol=1e-10)
        np.testing.assert_allclose(
            native.pullback(grad_y), np.asarray(grad_z_ref), rtol=1e-9, atol=1e-10
        )


def test_built_flow_has_the_jax_structure():
    """`build_triangular_flow` builds the JAX flow's structure: the same
    parents, transformer, parameter count, permutation and diagonal affine."""
    from nutpie.normalizing_flow import make_flow

    jax.config.update("jax_enable_x64", True)
    x, g = _draws()
    order = np.random.default_rng(3).permutation(DIM)
    settings = {
        "nn_width": 4,
        "contract_transformer": 1,
        "tangent_sas_transformer": 1,
        "log_gamma_bounds": (-1.0, 1.0),
        "input_squash": 1.0,
    }
    reference = from_jax(
        make_flow(
            0,
            x,
            g,
            n_layers=1,
            kind="triangular",
            sparsity=_banded(DIM, order),
            order=order,
            activation=jax.nn.softplus,
            **settings,
        )
    )
    flow = build_triangular_flow(
        0,
        x,
        g,
        sparsity=_banded(DIM, order),
        order=order,
        activation="softplus",
        **settings,
    )
    assert flow.n_params == reference.n_params
    np.testing.assert_array_equal(flow.permutation, reference.permutation)
    np.testing.assert_allclose(flow.loc, reference.loc, rtol=1e-12)
    np.testing.assert_allclose(flow.scale, reference.scale, rtol=1e-12)
    # Structure: with the reference's parameters, the two flows agree.
    same = flow.with_theta(reference.theta)
    np.testing.assert_allclose(
        same.fisher_divergence(x, g), reference.fisher_divergence(x, g), rtol=1e-12
    )


def test_built_flow_inverts_and_fits():
    """A built flow maps its own forward transform back exactly, starts near
    the diagonal flow, and fits."""
    from nutpie.triangular_lm import fit

    x, g = _draws(n_draw=64)
    order = np.random.default_rng(4).permutation(DIM)
    flow = build_triangular_flow(
        0,
        x,
        g,
        sparsity=_banded(DIM, order),
        order=order,
        nn_width=4,
        tangent_sas_transformer=1,
        input_squash=1.0,
    )

    native = flow.flow_transform()
    z = np.random.default_rng(5).normal(size=DIM)
    y, log_det = native.transform_and_log_det(z)
    grad_y = -y
    ours_log_det, back, grad_z = flow.inv_transform(y, grad_y)
    np.testing.assert_allclose(back, z, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(ours_log_det, log_det, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(grad_z, native.pullback(grad_y), rtol=1e-9, atol=1e-10)

    before = flow.fisher_divergence(x, g)
    theta, _, _ = fit(flow.residuals(x, g), flow.theta, n_steps=10, verbose=False)
    after = flow.with_theta(theta).fisher_divergence(x, g)
    # The toy gradients are noise, unrelated to the draws: no flow fits them
    # well, but the fit has to improve on the start.
    assert after < before


def test_rust_flow_adapter():
    """The Rust-path adapter: a diagonal affine for the first windows, then a
    triangular flow fitted to a correlated normal, and its sampler interface
    consistent with that flow."""
    from nutpie._lib import TriangularFlow
    from nutpie.transform_adapter import make_transform_adapter

    rng = np.random.default_rng(0)
    dim = 5
    root = rng.normal(size=(dim, dim))
    cov = root @ root.T + np.eye(dim)
    mean = rng.normal(size=dim)
    x = rng.multivariate_normal(mean, cov, size=400)
    g = -(x - mean) @ np.linalg.inv(cov)
    sparsity = ~np.eye(dim, dtype=bool)

    adapter = make_transform_adapter(
        rust_flow=True, num_diag_windows=2, initial_skip=0, nn_width=4
    )(0, x[0], g[0], 0, logp_fn=None, sparsity=sparsity, order=np.arange(dim))
    for n_draws in [20, 40]:
        adapter.update(n_draws, list(x[:n_draws]), list(g[:n_draws]), np.zeros(n_draws))
        assert isinstance(adapter.flow_transform_layout(), dict)
    adapter.update(400, list(x), list(g), np.zeros(400))
    flow = adapter.flow_transform_layout()
    assert isinstance(flow, TriangularFlow)
    assert adapter.transformation_id == 3
    # A normal is a linear flow away: fitted almost exactly.
    assert flow.fisher_divergence(x[:64], g[:64]) < 1e-3

    # The sampler's entry points agree with each other and with the flow.
    log_det, z, grad_z = adapter.inv_transform(x[5], g[5])
    y, part1 = adapter.init_from_transformed_position_part1(z)
    np.testing.assert_allclose(y, x[5], rtol=1e-10, atol=1e-10)
    log_det_back, grad_z_back = adapter.init_from_transformed_position_part2(
        part1, g[5]
    )
    np.testing.assert_allclose(log_det_back, log_det, rtol=1e-10)
    np.testing.assert_allclose(grad_z_back, grad_z, rtol=1e-9, atol=1e-10)


def test_rust_flow_rejects_unsupported_settings():
    from nutpie.transform_adapter import make_transform_adapter

    with pytest.raises(ValueError, match="coupling_type"):
        make_transform_adapter(rust_flow=True, coupling_type="masked")
    with pytest.raises(ValueError, match="nn_depth"):
        make_transform_adapter(rust_flow=True, nn_depth=2)
