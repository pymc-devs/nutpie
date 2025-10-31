from importlib.util import find_spec
import time
import pytest

if find_spec("pymc") is None:
    pytest.skip("Skip pymc tests", allow_module_level=True)

import numpy as np
import pymc as pm
import pytest

import nutpie
import nutpie.compile_pymc

# Check if MLX is available (macOS only, optional dependency)
MLX_AVAILABLE = find_spec("mlx") is not None

# Build backend list dynamically based on availability
backend_params = [
    ("numba", None),
    ("jax", "pytensor"),
    ("jax", "jax"),
]

# Only add MLX backends if MLX is available
if MLX_AVAILABLE:
    backend_params.extend(
        [
            ("mlx", "pytensor"),
            ("mlx", "mlx"),
        ]
    )

parameterize_backends = pytest.mark.parametrize(
    "backend, gradient_backend",
    backend_params,
)


@pytest.mark.pymc
@parameterize_backends
def test_pymc_model(backend, gradient_backend):
    with pm.Model() as model:
        pm.Normal("a")

    compiled = nutpie.compile_pymc_model(
        model, backend=backend, gradient_backend=gradient_backend
    )
    trace = nutpie.sample(compiled, chains=1)
    trace.posterior.a  # noqa: B018


@pytest.mark.pymc
@parameterize_backends
def test_name_x(backend, gradient_backend):
    with pm.Model() as model:
        x = pm.Data("x", 1.0)
        a = pm.Normal("a", mu=x)
        pm.Deterministic("z", x * a)

    compiled = nutpie.compile_pymc_model(
        model, backend=backend, gradient_backend=gradient_backend, freeze_model=False
    )
    trace = nutpie.sample(compiled, chains=1)
    trace.posterior.a  # noqa: B018


@pytest.mark.pymc
def test_order_shared():
    a_val = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    with pm.Model() as model:
        a = pm.Data("a", np.copy(a_val, order="C"))
        b = pm.Normal("b", shape=(2, 5))
        pm.Deterministic("c", (a[:, None, :] * b[:, :, None]).sum(-1))

    compiled = nutpie.compile_pymc_model(model, backend="numba")
    trace = nutpie.sample(compiled)
    np.testing.assert_allclose(
        (
            trace.posterior.b.values[:, :, :, :, None] * a_val[None, None, :, None, :]
        ).sum(-1),
        trace.posterior.c.values,
    )

    with pm.Model() as model:
        a = pm.Data("a", np.copy(a_val, order="F"))
        b = pm.Normal("b", shape=(2, 5))
        pm.Deterministic("c", (a[:, None, :] * b[:, :, None]).sum(-1))

    compiled = nutpie.compile_pymc_model(model, backend="numba")
    trace = nutpie.sample(compiled)
    np.testing.assert_allclose(
        (
            trace.posterior.b.values[:, :, :, :, None] * a_val[None, None, :, None, :]
        ).sum(-1),
        trace.posterior.c.values,
    )


@pytest.mark.pymc
@parameterize_backends
def test_low_rank(backend, gradient_backend):
    with pm.Model() as model:
        pm.Normal("a")

    compiled = nutpie.compile_pymc_model(
        model, backend=backend, gradient_backend=gradient_backend
    )
    trace = nutpie.sample(compiled, chains=1, low_rank_modified_mass_matrix=True)

    assert "mass_matrix_eigvals" not in trace.sample_stats
    trace = nutpie.sample(
        compiled, chains=1, low_rank_modified_mass_matrix=True, store_mass_matrix=True
    )
    assert "mass_matrix_eigvals" in trace.sample_stats


@pytest.mark.pymc
@parameterize_backends
def test_low_rank_half_normal(backend, gradient_backend):
    with pm.Model() as model:
        pm.HalfNormal("a", shape=(13, 3))
        pm.HalfNormal("b", shape=())
        pm.HalfNormal("c", shape=(5,))

    compiled = nutpie.compile_pymc_model(
        model, backend=backend, gradient_backend=gradient_backend
    )
    trace = nutpie.sample(compiled, chains=1, low_rank_modified_mass_matrix=True)
    trace.posterior.a  # noqa: B018


@pytest.mark.pymc
@parameterize_backends
def test_zero_size(backend, gradient_backend):
    import pytensor.tensor as pt

    with pm.Model() as model:
        a = pm.Normal("a", shape=(0, 0, 10))
        pm.Deterministic("b", pt.exp(a))

    compiled = nutpie.compile_pymc_model(
        model, backend=backend, gradient_backend=gradient_backend
    )
    trace = nutpie.sample(compiled, chains=1, draws=17, tune=100)
    assert trace.posterior.a.shape == (1, 17, 0, 0, 10)
    assert trace.posterior.b.shape == (1, 17, 0, 0, 10)


@pytest.mark.pymc
@parameterize_backends
def test_pymc_model_float32(backend, gradient_backend):
    import pytensor

    with pytensor.config.change_flags(floatX="float32"):
        with pm.Model() as model:
            pm.Normal("a")

        compiled = nutpie.compile_pymc_model(
            model, backend=backend, gradient_backend=gradient_backend
        )
        trace = nutpie.sample(compiled, chains=1)
        trace.posterior.a  # noqa: B018


@pytest.mark.pymc
@parameterize_backends
def test_pymc_model_no_prior(backend, gradient_backend):
    with pm.Model() as model:
        a = pm.Flat("a")
        pm.Normal("b", mu=a, observed=0.0)

    compiled = nutpie.compile_pymc_model(
        model, backend=backend, gradient_backend=gradient_backend
    )
    trace = nutpie.sample(compiled, chains=1)
    trace.posterior.a  # noqa: B018


@pytest.mark.pymc
@parameterize_backends
def test_blocking(backend, gradient_backend):
    with pm.Model() as model:
        pm.Normal("a")

    compiled = nutpie.compile_pymc_model(
        model, backend=backend, gradient_backend=gradient_backend
    )
    sampler = nutpie.sample(compiled, chains=1, blocking=False)
    trace = sampler.wait()
    trace.posterior.a  # noqa: B018


@pytest.mark.pymc
@parameterize_backends
@pytest.mark.timeout(20)
def test_wait_timeout(backend, gradient_backend):
    with pm.Model() as model:
        pm.Normal("a", shape=100_000)
    compiled = nutpie.compile_pymc_model(
        model, backend=backend, gradient_backend=gradient_backend
    )
    start = time.time()
    sampler = nutpie.sample(compiled, chains=1, blocking=False)
    with pytest.raises(TimeoutError):
        sampler.wait(timeout=0.1)
    sampler.cancel()
    assert start - time.time() < 5


@pytest.mark.pymc
@parameterize_backends
@pytest.mark.timeout(20)
def test_pause(backend, gradient_backend):
    with pm.Model() as model:
        pm.Normal("a", shape=10_000)
    compiled = nutpie.compile_pymc_model(
        model, backend=backend, gradient_backend=gradient_backend
    )
    start = time.time()
    sampler = nutpie.sample(compiled, chains=1, blocking=False)
    sampler.pause()
    sampler.resume()
    sampler.cancel()
    assert start - time.time() < 5


@pytest.mark.pymc
@parameterize_backends
@pytest.mark.timeout(20)
def test_abort(backend, gradient_backend):
    with pm.Model() as model:
        pm.Normal("a", shape=10_000)
    compiled = nutpie.compile_pymc_model(
        model, backend=backend, gradient_backend=gradient_backend
    )
    start = time.time()
    sampler = nutpie.sample(compiled, chains=1, blocking=False)
    sampler.pause()
    sampler.resume()
    sampler.abort()
    assert start - time.time() < 5


@pytest.mark.pymc
@parameterize_backends
def test_pymc_model_with_coordinate(backend, gradient_backend):
    with pm.Model() as model:
        model.add_coord("foo", length=5)
        pm.Normal("a", dims="foo")

    compiled = nutpie.compile_pymc_model(
        model, backend=backend, gradient_backend=gradient_backend
    )
    trace = nutpie.sample(compiled, chains=1)
    trace.posterior.a  # noqa: B018


@pytest.mark.pymc
@parameterize_backends
def test_pymc_model_store_extra(backend, gradient_backend):
    with pm.Model() as model:
        model.add_coord("foo", length=5)
        pm.Normal("a", dims="foo")

    compiled = nutpie.compile_pymc_model(
        model, backend=backend, gradient_backend=gradient_backend
    )
    trace = nutpie.sample(
        compiled,
        chains=1,
        store_mass_matrix=True,
        store_divergences=True,
        store_unconstrained=True,
        store_gradient=True,
    )
    trace.posterior.a  # noqa: B018
    _ = trace.sample_stats.unconstrained_draw
    _ = trace.sample_stats.gradient
    _ = trace.sample_stats.divergence_start
    _ = trace.sample_stats.mass_matrix_inv


@pytest.mark.pymc
@parameterize_backends
def test_trafo(backend, gradient_backend):
    with pm.Model() as model:
        pm.Uniform("a")

    compiled = nutpie.compile_pymc_model(
        model, backend=backend, gradient_backend=gradient_backend
    )
    trace = nutpie.sample(compiled, chains=1)
    trace.posterior.a  # noqa: B018


@pytest.mark.pymc
@parameterize_backends
def test_det(backend, gradient_backend):
    with pm.Model() as model:
        a = pm.Uniform("a", shape=2)
        pm.Deterministic("b", 2 * a)

    compiled = nutpie.compile_pymc_model(
        model, backend=backend, gradient_backend=gradient_backend
    )
    trace = nutpie.sample(compiled, chains=1)
    assert trace.posterior.a.shape[-1] == 2
    assert trace.posterior.b.shape[-1] == 2


@pytest.mark.pymc
@parameterize_backends
def test_non_identifier_names(backend, gradient_backend):
    with pm.Model() as model:
        a = pm.Uniform("a/b", shape=2)
        with pm.Model("foo"):
            c = pm.Data("c", np.array([2.0, 3.0]))
            pm.Deterministic("b", c * a)

    compiled = nutpie.compile_pymc_model(
        model, backend=backend, gradient_backend=gradient_backend
    )
    trace = nutpie.sample(compiled, chains=1)
    assert trace.posterior["a/b"].shape[-1] == 2
    assert trace.posterior["foo::b"].shape[-1] == 2


@pytest.mark.pymc
@parameterize_backends
def test_pymc_model_shared(backend, gradient_backend):
    with pm.Model() as model:
        mu = pm.Data("mu", -0.1)
        sigma = pm.Data("sigma", np.ones(3))
        pm.Normal("a", mu=mu, sigma=sigma, shape=3)

    compiled = nutpie.compile_pymc_model(
        model,
        backend=backend,
        gradient_backend=gradient_backend,
        freeze_model=False,
    )
    trace = nutpie.sample(compiled, chains=1, seed=1)
    np.testing.assert_allclose(trace.posterior.a.mean().values, -0.1, atol=0.05)

    compiled2 = compiled.with_data(mu=10.0, sigma=3 * np.ones(3))
    trace2 = nutpie.sample(compiled2, chains=1, seed=1)
    np.testing.assert_allclose(trace2.posterior.a.mean().values, 10.0, atol=0.5)

    compiled3 = compiled.with_data(mu=0.5, sigma=3 * np.ones(4))
    with pytest.raises(RuntimeError):
        nutpie.sample(compiled3, chains=1)


@pytest.mark.pymc
@parameterize_backends
def test_pymc_var_names(backend, gradient_backend):
    with pm.Model() as model:
        mu = pm.Data("mu", -0.1)
        sigma = pm.Data("sigma", np.ones(3))
        a = pm.Normal("a", mu=mu, sigma=sigma, shape=3)

        b = pm.Deterministic("b", mu * a)
        pm.Deterministic("c", mu * b)

    compiled = nutpie.compile_pymc_model(
        model,
        backend=backend,
        gradient_backend=gradient_backend,
        var_names=None,
    )
    trace = nutpie.sample(compiled, chains=1, seed=1)

    # Check that variables are stored
    assert hasattr(trace.posterior, "b")
    assert hasattr(trace.posterior, "c")

    compiled = nutpie.compile_pymc_model(
        model,
        backend=backend,
        gradient_backend=gradient_backend,
        var_names=[],
    )
    trace = nutpie.sample(compiled, chains=1, seed=1)

    # Check that variables are stored
    assert not hasattr(trace.posterior, "b")
    assert not hasattr(trace.posterior, "c")

    compiled = nutpie.compile_pymc_model(
        model,
        backend=backend,
        gradient_backend=gradient_backend,
        var_names=["b"],
    )
    trace = nutpie.sample(compiled, chains=1, seed=1)

    # Check that variables are stored
    assert hasattr(trace.posterior, "b")
    assert not hasattr(trace.posterior, "c")


# TODO For some reason, the sampling results with jax are
# not reproducible accross operating systems. Figure this
# out and add the array_compare marker.
# @pytest.mark.array_compare
@pytest.mark.pymc
@pytest.mark.flow
def test_normalizing_flow():
    with pm.Model() as model:
        pm.HalfNormal("x", shape=2)

    compiled = nutpie.compile_pymc_model(
        model, backend="jax", gradient_backend="jax"
    ).with_transform_adapt(
        verbose=True,
        num_layers=2,
    )
    trace = nutpie.sample(
        compiled,
        chains=1,
        transform_adapt=True,
        window_switch_freq=128,
        seed=1,
        draws=500,
    )
    assert float(trace.sample_stats.fisher_distance.mean()) < 0.1
    # return trace.posterior.x.isel(draw=slice(-50, None)).values.ravel()


@pytest.mark.pymc
@pytest.mark.parametrize(
    ("backend", "gradient_backend"),
    [
        ("numba", None),
        pytest.param(
            "jax",
            "pytensor",
            marks=pytest.mark.xfail(
                reason="https://github.com/pymc-devs/pytensor/issues/853"
            ),
        ),
        pytest.param(
            "jax",
            "jax",
            marks=pytest.mark.xfail(
                reason="https://github.com/pymc-devs/pytensor/issues/853"
            ),
        ),
    ],
)
def test_missing(backend, gradient_backend):
    with pm.Model(coords={"obs": range(4)}) as model:
        mu = pm.Normal("mu")
        y = pm.Normal("y", mu, observed=[0, -1, 1, np.nan], dims="obs")
        pm.Deterministic("y2", 2 * y, dims="obs")

    compiled = nutpie.compile_pymc_model(
        model, backend=backend, gradient_backend=gradient_backend
    )
    tr = nutpie.sample(compiled, chains=1, seed=1)
    assert hasattr(tr.posterior, "y_unobserved")


@pytest.mark.pymc
@pytest.mark.array_compare(atol=1e-4, rtol=1e-4)
def test_deterministic_sampling_numba():
    with pm.Model() as model:
        pm.HalfNormal("a")

    compiled = nutpie.compile_pymc_model(model, backend="numba")
    trace = nutpie.sample(compiled, chains=2, seed=123, draws=100, tune=100)
    return trace.posterior.a.values.ravel()


@pytest.mark.pymc
@pytest.mark.array_compare(atol=1e-4, rtol=1e-4)
def test_deterministic_sampling_jax():
    with pm.Model() as model:
        pm.HalfNormal("a")

    compiled = nutpie.compile_pymc_model(model, backend="jax", gradient_backend="jax")
    trace = nutpie.sample(compiled, chains=2, seed=123, draws=100, tune=100)
    return trace.posterior.a.values.ravel()


@pytest.mark.pymc
@pytest.mark.array_compare(atol=1e-6, rtol=1e-6)
def test_deterministic_sampling_mlx():
    if not MLX_AVAILABLE:
        pytest.skip("MLX not installed")

    with pm.Model() as model:
        pm.HalfNormal("a")

    compiled = nutpie.compile_pymc_model(model, backend="mlx", gradient_backend="mlx")
    trace = nutpie.sample(compiled, chains=2, seed=123, draws=100, tune=100)
    return trace.posterior.a.values.ravel()


@pytest.mark.pymc
def test_zarr_store(tmp_path):
    with pm.Model() as model:
        pm.HalfNormal("a")

    compiled = nutpie.compile_pymc_model(model, backend="numba")

    path = tmp_path / "trace.zarr"
    path.mkdir()
    store = nutpie.zarr_store.LocalStore(str(path))
    trace = nutpie.sample(
        compiled, chains=2, seed=123, draws=100, tune=100, zarr_store=store
    )
    trace.load().posterior.a  # noqa: B018


@pytest.fixture
def tmp_path():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)
