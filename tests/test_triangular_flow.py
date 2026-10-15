"""`nutpie.triangular_flow`, the triangular flow built in Rust, against the
JAX flow from `make_flow(kind="triangular")`."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import equinox as eqx
import jax.numpy as jnp

from nutpie.triangular_flow import (
    _normalized,
    build_triangular_flow,
    default_conditioners,
    from_jax,
    transformer_kind,
    transformer_layout,
)

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
    {"transformer": "affine"},
    {"transformer": "sas", "transformer_layers": 2, "log_gamma_bounds": (-1.0, 1.0)},
    {"transformer": "tangent_sas"},
    {"transformer": "tangent_sas", "transformer_layers": 2},
    {"transformer": "tangent_sas", "transformer_layers": 2, "tangent_sas_fix_b": True},
    {"transformer": "asymmetric_sas"},
    {
        "transformer": "asymmetric_sas",
        "transformer_layers": 3,
        "log_gamma_bounds": (-1.0, 1.0),
    },
]


@pytest.mark.parametrize("kwargs", TRANSFORMERS)
def test_transformer_layout_matches_the_probe(kwargs):
    """The Rust kind's layout is the one probed from the JAX transformer:
    the same layers, fields, indices, offsets, location and scale."""
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

    layout = transformer_layout(transformer_kind(**kwargs))
    assert layout["n_coefs"] == n_par
    assert layout["location"] == int(tmap.conditioners[0].location_index)
    assert layout["scale"] == int(tmap.conditioners[0].scale_index)
    assert _normalized(layout["layers"]) == _normalized(probed)


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
        "transformer": "tangent_sas",
        "transformer_layers": 2,
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
    {"nn_width": 0, "transformer": "sas", "transformer_layers": 2},
    {"transformer": "asymmetric_sas", "transformer_layers": 2},
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
        "transformer": "sas",
        "transformer_layers": 2,
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
        transformer=settings["transformer"],
        transformer_layers=settings["transformer_layers"],
        log_gamma_bounds=settings["log_gamma_bounds"],
        conditioners=default_conditioners(
            settings["nn_width"], "softplus", input_squash=settings["input_squash"]
        ),
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
        transformer="tangent_sas",
        conditioners=default_conditioners(4, input_squash=1.0),
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


@pytest.mark.parametrize("scale_squash", [None, 2.5])
@pytest.mark.parametrize("transformer", ["affine", "sas", "asymmetric_sas"])
def test_scale_squash_inverts(transformer, scale_squash):
    """With a scale squash other than the default, the forward transform and
    the inverse the fit uses still agree."""
    x, g = _draws(n_draw=64)
    order = np.random.default_rng(4).permutation(DIM)
    flow = build_triangular_flow(
        0,
        x,
        g,
        sparsity=_banded(DIM, order),
        order=order,
        transformer=transformer,
        scale_squash=scale_squash,
        conditioners=default_conditioners(4),
    )
    flow = flow.with_theta(0.3 * np.random.default_rng(6).normal(size=flow.n_params))

    native = flow.flow_transform()
    z = np.random.default_rng(5).normal(size=DIM)
    y, log_det = native.transform_and_log_det(z)
    grad_y = -y
    ours_log_det, back, grad_z = flow.inv_transform(y, grad_y)
    np.testing.assert_allclose(back, z, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(ours_log_det, log_det, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(grad_z, native.pullback(grad_y), rtol=1e-9, atol=1e-10)


def test_native_transform_with_any_conditioners():
    """The sampler's transform of a flow with two MLPs (one on part of the
    coefficients, gelu, squashed) and a linear scale conditioner: its draws
    map back exactly, and its pullback is the inverse's gradient map."""
    x, g = _draws()
    order = np.random.default_rng(8).permutation(DIM)
    conditioners = [
        {"kind": "constant", "target": "all"},
        {"kind": "linear", "target": ["location", "scale"]},
        {"kind": "mlp", "width": 3, "target": "all", "init_output_scale": 0.1},
        {
            "kind": "mlp",
            "width": 2,
            "target": ["scale", "shape"],
            "activation": "gelu_tanh",
            "input_squash": 0.8,
            "init_output_scale": 0.1,
        },
    ]
    flow = build_triangular_flow(
        0,
        x,
        g,
        sparsity=_banded(DIM, order),
        order=order,
        transformer="sas",
        transformer_layers=2,
        conditioners=conditioners,
    )
    # Moderate weights: a two-layer sinh-arcsinh with large tail coefficients
    # overflows.
    rng = np.random.default_rng(9)
    flow = flow.with_theta(flow.theta + 0.05 * rng.normal(size=flow.n_params))
    native = flow.flow_transform()
    for _ in range(3):
        z = rng.normal(size=DIM)
        y, log_det = native.transform_and_log_det(z)
        grad_y = rng.normal(size=DIM)
        ours_log_det, back, grad_z = flow.inv_transform(y, grad_y)
        np.testing.assert_allclose(back, z, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(ours_log_det, log_det, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(
            native.pullback(grad_y), grad_z, rtol=1e-9, atol=1e-10
        )


def test_with_affine_restandardizes():
    """`with_affine` standardizes the draws with the new affine: ``y = (x -
    loc) / scale`` and ``g_y = g * scale``, in the triangular order."""
    x, g = _draws()
    order = np.random.default_rng(3).permutation(DIM)
    flow = build_triangular_flow(0, x, g, sparsity=_banded(DIM, order), order=order)
    rng = np.random.default_rng(4)
    loc, scale = rng.normal(size=DIM), np.exp(rng.normal(size=DIM))
    moved = flow.with_affine(loc, scale)
    y, gy = moved.undo_affine(x, g)
    np.testing.assert_allclose(y, ((x - loc) / scale)[:, order], rtol=1e-12)
    np.testing.assert_allclose(gy, (g * scale)[:, order], rtol=1e-12)
    with pytest.raises(RuntimeError, match="nonzero"):
        flow.with_affine(loc, np.zeros(DIM))


@pytest.mark.parametrize(
    ("transformer", "scale_squash"),
    [("affine", 1.0), ("sas", 1.0), ("affine", None), ("sas", None)],
)
def test_with_affine_keeps_a_gaussian_flow(transformer, scale_squash):
    """Without MLPs, `with_affine` moves the conditioners so that the flow
    stays the same function: the same Fisher divergence and the same draws
    back in the base space. With the default scale squash only without
    linear scale weights; without a squash the log scale is linear in them,
    so they move too."""
    x, g = _draws()
    order = np.random.default_rng(3).permutation(DIM)
    conditioners = [
        {"kind": "constant", "target": "all"},
        {
            "kind": "linear",
            "target": "location" if scale_squash == 1.0 else ["location", "scale"],
        },
    ]
    flow = build_triangular_flow(
        0,
        x,
        g,
        sparsity=_banded(DIM, order),
        order=order,
        transformer=transformer,
        scale_squash=scale_squash,
        conditioners=conditioners,
    )
    rng = np.random.default_rng(5)
    flow = flow.with_theta(0.3 * rng.normal(size=flow.n_params))
    loc, scale = rng.normal(size=DIM), np.exp(rng.normal(size=DIM))
    moved = flow.with_affine(loc, scale)
    np.testing.assert_allclose(
        moved.fisher_divergence(x, g), flow.fisher_divergence(x, g), rtol=1e-10
    )
    for k in range(3):
        ours = moved.inv_transform(x[k], g[k])
        reference = flow.inv_transform(x[k], g[k])
        for a, b in zip(ours, reference):
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)


def test_with_conditioners_carries_values():
    """The same list carries everything; the linear stage and back keeps
    the constant and linear values; new MLPs start as in a new flow."""
    from nutpie.triangular_flow import linear_conditioners

    x, g = _draws()
    order = np.random.default_rng(3).permutation(DIM)
    full = default_conditioners(4)
    flow = build_triangular_flow(
        0, x, g, sparsity=_banded(DIM, order), order=order, conditioners=full
    )
    rng = np.random.default_rng(6)
    flow = flow.with_theta(0.3 * rng.normal(size=flow.n_params))
    np.testing.assert_array_equal(flow.with_conditioners(full).theta, flow.theta)

    linear = linear_conditioners(full, {"kind": "sas", "layers": 1})
    staged = flow.with_conditioners(linear)
    back = staged.with_conditioners(full, seed=1)
    for i in range(DIM):
        (c, c_end), (l, l_end), (m, m_end) = flow.weight_ranges(i)[:3]
        (sc, sc_end), (sl, sl_end) = staged.weight_ranges(i)
        (bc, bc_end), (bl, bl_end), (bm, bm_end) = back.weight_ranges(i)[:3]
        # sas: alpha, beta, sigma, mu, nu; the scale is sigma (2), the
        # location mu (3), the shape alpha and beta.
        constant = np.asarray(flow.theta[c:c_end])
        np.testing.assert_array_equal(staged.theta[sc:sc_end], constant[2:])
        np.testing.assert_array_equal(back.theta[bc:bc_end][2:], constant[2:])
        np.testing.assert_array_equal(back.theta[bc:bc_end][:2], 0.0)
        n_parent = (l_end - l) // 2
        # The linear rows in coefficient order: the scale, then the location.
        location_row = flow.theta[l + n_parent : l_end]
        np.testing.assert_array_equal(staged.theta[sl:sl_end], location_row)
        np.testing.assert_array_equal(back.theta[bl + n_parent : bl_end], location_row)
        np.testing.assert_array_equal(back.theta[bl : bl + n_parent], 0.0)
        # The MLP comes back new (it has no weights without parents).
        if n_parent:
            assert not np.array_equal(back.theta[bm:bm_end], flow.theta[m:m_end])


def test_linear_stage_fits_like_the_frozen_fit():
    """A fit with the linear conditioners takes the same steps as one with
    the full conditioners and everything else frozen, from a full flow whose
    MLP outputs are zero."""
    from nutpie.triangular_flow import linear_conditioners
    from nutpie.triangular_lm import fit

    x, g = _draws(n_draw=64)
    order = np.random.default_rng(7).permutation(DIM)
    full = default_conditioners(4, mlp_ridge=0.5, skip_ridge=0.2)
    for c in full:
        if c["kind"] == "mlp":
            c["init_output_scale"] = 0.0
    flow = build_triangular_flow(
        0, x, g, sparsity=_banded(DIM, order), order=order, conditioners=full
    )
    linear = flow.with_conditioners(
        linear_conditioners(full, {"kind": "sas", "layers": 1})
    )
    settings = {"n_steps": 5, "verbose": False, "max_block_size": 10_000}

    problem = flow.residuals(x, g)
    theta_full, _, hist_full = fit(
        problem, flow.theta, frozen=problem.nonlinear_param_mask, **settings
    )
    theta_linear, _, hist_linear = fit(linear.residuals(x, g), linear.theta, **settings)
    for ours, ref in zip(hist_linear, hist_full):
        assert ours["accept"] == ref["accept"]
        np.testing.assert_allclose(ours["lam_out"], ref["lam_out"], rtol=1e-8)
    np.testing.assert_allclose(
        linear.with_theta(theta_linear).fisher_divergence(x, g),
        flow.with_theta(theta_full).fisher_divergence(x, g),
        rtol=1e-8,
    )


@pytest.mark.parametrize("refit_diag", [False, True])
def test_rust_flow_adapter(refit_diag):
    """The Rust-path adapter: a diagonal affine for the first windows, then a
    triangular flow fitted to a correlated normal and refitted (with
    `refit_diag`, in new coordinates), and its sampler interface consistent
    with that flow."""
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
        rust_flow=True,
        num_diag_windows=2,
        initial_skip=0,
        nn_width=4,
        refit_diag=refit_diag,
    )(0, x[0], g[0], 0, logp_fn=None, sparsity=sparsity, order=np.arange(dim))
    for n_draws in [20, 40]:
        adapter.update(n_draws, list(x[:n_draws]), list(g[:n_draws]), np.zeros(n_draws))
        assert isinstance(adapter.flow_transform_layout(), dict)
    for _ in range(2):
        adapter.update(400, list(x), list(g), np.zeros(400))
    flow = adapter.flow_transform_layout()
    assert isinstance(flow, TriangularFlow)
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
