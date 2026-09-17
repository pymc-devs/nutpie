import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("numba")

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from nutpie.normalizing_flow import make_transformer  # noqa: E402
from nutpie.triangular import SparseTriangularMap  # noqa: E402
from nutpie.triangular_numba import compile_transform as compile_numba  # noqa: E402
from nutpie.triangular_rust import compile_transform as compile_rust  # noqa: E402

BACKENDS = {
    "numba": compile_numba,
    "rust": compile_rust,
    "rust-serial": lambda flow: compile_rust(flow, schedule="serial"),
    "rust-levels": lambda flow: compile_rust(
        flow, schedule="levels", min_parallel_work=0
    ),
    "rust-dataflow": lambda flow: compile_rust(flow, schedule="dataflow"),
}


def _blanket(structure, dim, bandwidth, rng, extra=0):
    blanket = np.zeros((dim, dim), dtype=bool)
    if structure == "banded":
        for i in range(1, dim):
            blanket[i, max(0, i - bandwidth) : i] = True
    elif structure == "tree":
        for i in range(1, dim):
            blanket[i, (i - 1) // 2] = True
    else:
        raise ValueError(structure)
    for _ in range(extra):
        i, j = rng.integers(0, dim, size=2)
        if i != j:
            blanket[max(i, j), min(i, j)] = True
    return blanket


def _random_params(flow, rng, scale=0.3):
    """Move off the (zero) initialization, so every parameter matters."""
    params, static = eqx.partition(flow, eqx.is_inexact_array)
    params = jax.tree.map(
        lambda leaf: jnp.asarray(rng.normal(size=leaf.shape, scale=scale), leaf.dtype),
        params,
    )
    return eqx.combine(params, static)


@pytest.mark.parametrize("backend", sorted(BACKENDS))
@pytest.mark.parametrize(
    "structure, dim, bandwidth, extra, n_buckets, nn_depth, nn_width, scale",
    [
        ("banded", 24, 3, 0, 3, 1, 16, 0.3),
        ("banded", 40, 5, 12, 4, 2, 8, 0.3),
        ("banded", 17, 16, 0, 2, 1, 16, 0.3),  # dense lower triangle
        # Depth-0 conditioners on a chain blanket: each variable is a bare
        # linear function of its single parent, so the map compounds along the
        # whole chain. At scale 0.3 it reaches 1e237 by variable 6 and the
        # comparison degenerates into which side of the overflow cliff each
        # implementation lands on, which is not what this test is for.
        ("banded", 11, 1, 0, 8, 0, 16, 0.05),
        ("tree", 63, 1, 0, 3, 1, 16, 0.3),  # shallow: several per level
    ],
)
def test_matches_jax(
    backend, structure, dim, bandwidth, extra, n_buckets, nn_depth, nn_width, scale
):
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(dim * 100 + bandwidth)
    blanket = _blanket(structure, dim, bandwidth, rng, extra=extra)

    flow = SparseTriangularMap(
        jax.random.key(0),
        blanket=blanket,
        n_buckets=n_buckets,
        nn_depth=nn_depth,
        nn_width=nn_width,
    )
    flow = _random_params(flow, rng, scale=scale)

    fn = BACKENDS[backend](flow)

    for _ in range(3):
        x = rng.normal(size=dim)
        y_expected, log_det_expected = flow.transform_and_log_det(jnp.asarray(x))
        y, log_det = fn.transform_and_log_det(x)
        np.testing.assert_allclose(y, np.asarray(y_expected), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(
            log_det, float(log_det_expected), rtol=1e-10, atol=1e-10
        )


@pytest.mark.parametrize("backend", sorted(BACKENDS))
@pytest.mark.parametrize("activation", ["softplus", "relu", "gelu", "tanh"])
def test_matches_jax_for_each_activation(backend, activation):
    """Covers the vectorized activations end to end, not just in isolation.

    `test_vectorized_softplus_accuracy` checks the SIMD `exp`/`log1p` against a
    float128 reference over their whole domain; this checks that the result is
    actually what the conditioner feeds forward.
    """
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(97)
    dim = 48
    blanket = _blanket("banded", dim, 5, rng, extra=10)

    flow = SparseTriangularMap(
        jax.random.key(3),
        blanket=blanket,
        n_buckets=3,
        nn_activation={
            "softplus": jax.nn.softplus,
            "relu": jax.nn.relu,
            "gelu": jax.nn.gelu,
            "tanh": jnp.tanh,
        }[activation],
    )
    flow = _random_params(flow, rng)

    fn = BACKENDS[backend](flow)
    for _ in range(3):
        x = rng.normal(size=dim)
        y_expected, log_det_expected = flow.transform_and_log_det(jnp.asarray(x))
        y, log_det = fn.transform_and_log_det(x)
        np.testing.assert_allclose(y, np.asarray(y_expected), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(
            log_det, float(log_det_expected), rtol=1e-10, atol=1e-10
        )


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_matches_jax_with_bounded_log_gamma(backend):
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(7)
    dim = 20
    blanket = _blanket("banded", dim, 4, rng, extra=6)

    flow = SparseTriangularMap(
        jax.random.key(1),
        blanket=blanket,
        transformer=make_transformer(
            affine_transformer=False,
            asymmetric_transformer=False,
            contract_transformer=3,
            log_gamma_bounds=(-1.0, 1.5),
        ),
        n_buckets=3,
    )
    # Three stacked Contract2 layers compound, so keep the parameters small
    # enough that the chain stays in a range where the comparison is meaningful.
    flow = _random_params(flow, rng, scale=0.05)

    fn = BACKENDS[backend](flow)
    x = rng.normal(size=dim)
    y_expected, log_det_expected = flow.transform_and_log_det(jnp.asarray(x))
    y, log_det = fn.transform_and_log_det(x)
    np.testing.assert_allclose(y, np.asarray(y_expected), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(log_det, float(log_det_expected), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("structure", ["banded", "tree"])
def test_rust_schedules_are_bit_identical(structure):
    """Whatever rayon does with the DAG must not move a single bit.

    Both parallel schedules write per-variable results and reduce the log det
    in index order afterwards, so the total is independent of how the work got
    split. If that ever regresses, a seeded sampler stops being reproducible.
    """
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(11)
    dim = 255
    blanket = _blanket(structure, dim, 4, rng)

    flow = SparseTriangularMap(jax.random.key(4), blanket=blanket, n_buckets=3)
    flow = _random_params(flow, rng, scale=0.05)

    serial = compile_rust(flow, schedule="serial")
    levels = compile_rust(flow, schedule="levels", min_parallel_work=0)
    dataflow = compile_rust(flow, schedule="dataflow")
    assert serial.schedule == "serial"
    # A banded blanket has one variable per level, so there is nothing to
    # parallelize however low the cutoff goes.
    if structure == "banded":
        assert levels.n_parallel_levels == 0
    else:
        assert levels.n_parallel_levels > 0

    x = rng.normal(size=dim)
    expected = serial.transform_and_log_det(x)
    for name, fn in [("levels", levels), ("dataflow", dataflow)]:
        y, log_det = fn.transform_and_log_det(x)
        np.testing.assert_array_equal(y, expected[0], err_msg=name)
        assert log_det == expected[1], name


@pytest.mark.parametrize(
    "name, low, high, count",
    [
        ("typical", -8.0, 8.0, 200_000),
        ("near zero", -1e-8, 1e-8, 50_000),
        ("wide", -50.0, 50.0, 200_000),
        # Past the point where `exp` underflows, so this covers the split
        # scaling that keeps the constructed power of two normal.
        ("denormal tail", -800.0, -690.0, 50_000),
        ("saturating", 30.0, 800.0, 50_000),
    ],
)
def test_vectorized_softplus_accuracy(name, low, high, count):
    """The hand-written SIMD `exp`/`log1p` must stay within a couple of ulp.

    A real flow's hidden pre-activations are all O(1), so the compiled kernels
    alone would never exercise the range reduction; this goes at it directly.
    Reference is evaluated in float128 to keep the comparison meaningful at the
    ulp level.
    """
    from nutpie._lib import _activation_for_testing

    x = np.random.default_rng(abs(hash(name)) % 2**31).uniform(low, high, count)
    got = np.asarray(_activation_for_testing("softplus", x), dtype=np.longdouble)

    wide = np.asarray(x, dtype=np.longdouble)
    expected = np.log1p(np.exp(-np.abs(wide))) + np.maximum(wide, 0)

    ulp = np.where(
        expected == 0,
        0.0,
        np.abs(got - expected) / np.spacing(np.abs(expected).astype(np.float64)),
    )
    assert np.nanmax(ulp) < 4.0, f"{name}: {np.nanmax(ulp)} ulp"


def test_vectorized_softplus_edge_cases():
    from nutpie._lib import _activation_for_testing

    x = np.array([0.0, -0.0, 1.0, -1.0, -745.0, -746.0, -750.0, 710.0, np.inf, -np.inf])
    expected = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
    np.testing.assert_array_equal(_activation_for_testing("softplus", x), expected)


@pytest.mark.parametrize("activation", ["softplus", "relu", "gelu_tanh", "tanh", "silu"])
def test_activations_propagate_nan(activation):
    """A diverged trajectory must stay diverged.

    Both `vmaxpd` and Rust's `f64::max` return the *other* operand when one is
    NaN, so the `max(v, 0)` in softplus and relu will quietly turn a NaN back
    into a finite number unless guarded. `jnp.maximum` propagates, and a
    sampler that loses the NaN would accept a step it must reject.
    """
    from nutpie._lib import _activation_for_testing

    # Enough values to cover both the SIMD body and the scalar tail.
    x = np.full(11, np.nan)
    assert np.all(np.isnan(_activation_for_testing(activation, x)))

    mixed = np.array([1.0, np.nan, -3.0, np.nan, 0.0, np.nan, 2.0])
    out = _activation_for_testing(activation, mixed)
    np.testing.assert_array_equal(np.isnan(out), np.isnan(mixed))


def test_auto_schedule_avoids_threads_without_width():
    """A banded blanket has one variable per level; `auto` must not fork."""
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(13)
    banded = compile_rust(
        _random_params(
            SparseTriangularMap(
                jax.random.key(5), blanket=_blanket("banded", 64, 3, rng)
            ),
            rng,
            scale=0.05,
        )
    )
    wide = compile_rust(
        _random_params(
            SparseTriangularMap(
                jax.random.key(5), blanket=_blanket("tree", 1023, 1, rng)
            ),
            rng,
            scale=0.05,
        )
    )
    assert banded.schedule == "serial"
    assert wide.schedule == "dataflow"


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_rejects_unsupported_transformer(backend):
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(3)
    blanket = _blanket("banded", 8, 2, rng)
    flow = SparseTriangularMap(
        jax.random.key(2),
        blanket=blanket,
        transformer=make_transformer(
            affine_transformer=True,
            asymmetric_transformer=False,
            contract_transformer=1,
        ),
    )
    with pytest.raises(NotImplementedError):
        BACKENDS[backend](flow)
