import sys
import threading
import time
from importlib.util import find_spec

import pytest

if find_spec("pymc") is None:
    pytest.skip("Skip pymc tests", allow_module_level=True)

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

import nutpie
import nutpie.compile_pymc

MLX_AVAILABLE = find_spec("mlx") is not None

backend_params = [("numba", None), ("jax", "pytensor"), ("jax", "jax")]
if MLX_AVAILABLE:
    backend_params.append(("mlx", "pytensor"))

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
def test_progress_callback(backend, gradient_backend):
    with pm.Model() as model:
        pm.Normal("a")

    compiled = nutpie.compile_pymc_model(
        model, backend=backend, gradient_backend=gradient_backend
    )

    received = []

    def callback(chains):
        received.append(chains)

    nutpie.sample(
        compiled,
        chains=2,
        progress_bar=False,
        progress_callback=callback,
    )

    assert len(received) > 0
    chains = received[-1]
    assert len(chains) == 2
    chain = chains[0]
    assert chain.total_draws > 0
    assert chain.finished_draws == chain.total_draws
    assert isinstance(chain.step_size, float)
    assert isinstance(chain.divergent_draws, list)


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
    trace = nutpie.sample(compiled, chains=1, adaptation="low_rank")

    assert "mass_matrix_eigvals" not in trace.sample_stats
    trace = nutpie.sample(
        compiled, chains=1, adaptation="low_rank", store_mass_matrix=True
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
    trace = nutpie.sample(compiled, chains=1, adaptation="low_rank")
    trace.posterior.a  # noqa: B018


@pytest.mark.pymc
@parameterize_backends
def test_deprecated_low_rank_modified_mass_matrix(backend, gradient_backend):
    with pm.Model() as model:
        pm.Normal("a")

    compiled = nutpie.compile_pymc_model(
        model, backend=backend, gradient_backend=gradient_backend
    )
    with pytest.warns(FutureWarning, match="low_rank_modified_mass_matrix"):
        trace = nutpie.sample(compiled, chains=1, low_rank_modified_mass_matrix=True)
    trace.posterior.a  # noqa: B018


@pytest.mark.pymc
@parameterize_backends
def test_deprecated_use_grad_based_mass_matrix(backend, gradient_backend):
    with pm.Model() as model:
        pm.Normal("a")

    compiled = nutpie.compile_pymc_model(
        model, backend=backend, gradient_backend=gradient_backend
    )
    with pytest.warns(FutureWarning, match="use_grad_based_mass_matrix"):
        trace = nutpie.sample(compiled, chains=1, use_grad_based_mass_matrix=False)
    trace.posterior.a  # noqa: B018


@pytest.mark.pymc
@parameterize_backends
def test_zero_size(backend, gradient_backend):
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
        model.add_coord("bar", length=4)
        pm.Normal("a", dims="foo")
        pm.HalfNormal("b", sigma=1.0, dims="foo")
        pm.ZeroSumNormal("c", sigma=1.0, dims="foo")
        pm.Dirichlet("d", a=np.ones(4), dims="bar")

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
    trace.posterior.b  # noqa: B018
    trace.posterior.c  # noqa: B018
    trace.posterior.d  # noqa: B018
    assert trace.posterior.c.dims == ("chain", "draw", "foo")
    assert trace.posterior.d.dims == ("chain", "draw", "bar")
    assert trace.unconstrained_posterior.b_log__.dims == ("chain", "draw", "foo")
    # ZeroSumNormal's unconstrained value has one fewer element along the
    # zero-sum axis, so it should NOT inherit the "foo" dim.
    assert trace.unconstrained_posterior.c_zerosum__.dims != (
        "chain",
        "draw",
        "foo",
    )
    # Dirichlet's simplex transform reduces dimensionality by one, so it
    # should NOT inherit the "bar" dim.
    assert trace.unconstrained_posterior.d_simplex__.dims != (
        "chain",
        "draw",
        "bar",
    )
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
        a = pm.Uniform("a::b", shape=2)
        with pm.Model("foo"):
            c = pm.Data("c", np.array([2.0, 3.0]))
            pm.Deterministic("b", c * a)

    compiled = nutpie.compile_pymc_model(
        model, backend=backend, gradient_backend=gradient_backend
    )
    trace = nutpie.sample(compiled, chains=1)
    assert trace.posterior["a::b"].shape[-1] == 2
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
        adaptation="flow",
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


# MLX computes in float32 on the GPU, whose rounding differs between machines,
# so the draws are checked against analytic moments instead of reference values.
@pytest.mark.pymc
@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not installed")
def test_sampling_mlx():
    with pm.Model() as model:
        pm.HalfNormal("a")

    compiled = nutpie.compile_pymc_model(model, backend="mlx")
    trace = nutpie.sample(
        compiled, chains=4, seed=123, draws=4000, tune=1000, progress_bar=False
    )
    a = trace.posterior.a.values

    assert (a >= 0).all()

    expected_mean = np.sqrt(2.0 / np.pi)
    expected_std = np.sqrt(1.0 - 2.0 / np.pi)
    assert a.mean() == pytest.approx(expected_mean, abs=0.05)
    assert a.std() == pytest.approx(expected_std, abs=0.05)


@pytest.mark.pymc
@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not installed")
def test_mlx_concurrent_first_calls():
    import mlx.core as mx

    n_threads = 8
    default_device = mx.default_device()
    switch_interval = sys.getswitchinterval()
    rng = np.random.default_rng(0)

    try:
        for _ in range(3):
            with pm.Model() as model:
                mu = pm.Normal("mu", 0, 10, shape=10)
                sigma = pm.HalfNormal("sigma")
                pm.Normal("obs", mu, sigma, observed=rng.normal(size=(200, 10)))

            compiled = nutpie.compile_pymc_model(model, backend="mlx")
            x = np.zeros(compiled.n_dim)
            barrier = threading.Barrier(n_threads)
            logps = []
            errors = []

            def first_call(
                compiled=compiled, x=x, barrier=barrier, logps=logps, errors=errors
            ):
                logp_fn = compiled._make_logp_func()
                barrier.wait()
                try:
                    logps.append(logp_fn(x, **compiled._shared_data)[0])
                except ValueError as error:
                    errors.append(error)

            # A tiny switch interval makes the threads interleave while the
            # first call traces the compiled function.
            sys.setswitchinterval(1e-6)
            threads = [threading.Thread(target=first_call) for _ in range(n_threads)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            sys.setswitchinterval(switch_interval)

            assert errors == []
            assert logps == [logps[0]] * n_threads
            assert mx.default_device() == default_device
    finally:
        sys.setswitchinterval(switch_interval)
        mx.set_default_device(default_device)


@pytest.mark.pymc
@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not installed")
def test_mlx_keeps_float64_on_cpu():
    import mlx.core as mx

    with pm.Model() as model:
        pm.Normal("a")

    default_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        compiled = nutpie.compile_pymc_model(model, backend="mlx")
        point = np.array([1 + 1e-9])
        _, grad = compiled._make_logp_func()(point, **compiled._shared_data)
    finally:
        mx.set_default_device(default_device)

    # float32 would round the point, and so the gradient, to exactly -1.
    assert grad[0] == -(1 + 1e-9)


@pytest.mark.pymc
@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not installed")
def test_mlx_rejects_other_gradient_backends():
    with pm.Model() as model:
        pm.Normal("a")

    with pytest.raises(ValueError, match="Gradient backend cannot be bogus"):
        nutpie.compile_pymc_model(model, backend="mlx", gradient_backend="bogus")


@pytest.mark.pymc
@pytest.mark.parametrize(
    "backend",
    [
        "numba",
        pytest.param(
            "mlx",
            marks=pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not installed"),
        ),
    ],
)
def test_jax_gradient_requires_jax_backend(backend):
    with pm.Model() as model:
        pm.Normal("a")

    with pytest.raises(ValueError, match="Gradient backend cannot be jax"):
        nutpie.compile_pymc_model(model, backend=backend, gradient_backend="jax")


@pytest.mark.pymc
def test_zarr_store(tmp_path):
    coords = {
        "a": np.arange(2).astype("f"),
        "b": pd.date_range("2023-01-01", periods=1),
        "c": ["x", "y", "z", ""],
        "d": [0],
        "e": pd.factorize(pd.Index(["foo"]))[1],
        "f": np.arange(2).astype("d"),
        "g": pd.date_range("2023-01-01", periods=0),
    }
    with pm.Model(coords=coords) as model:
        pm.HalfNormal("x")
        pm.Normal("y", dims=("a", "b", "c", "d", "e", "f"))
        pm.Normal("z", dims="g")

    compiled = nutpie.compile_pymc_model(model, backend="numba")

    path = tmp_path / "trace.zarr"
    path.mkdir()
    store = nutpie.zarr_store.LocalStore(str(path))
    trace = nutpie.sample(
        compiled, chains=2, seed=123, draws=100, tune=100, zarr_store=store
    )
    _ = trace.load().posterior.x

    assert trace.posterior.coords["a"].dtype == np.float32
    # pandas 3.0 changes datetime64 precision
    assert trace.posterior.coords["b"].dtype.name in [
        "datetime64[ns]",
        "datetime64[us]",
    ]
    assert trace.posterior.coords["b"].values[0] == np.datetime64("2023-01-01")
    assert list(trace.posterior.coords["c"]) == ["x", "y", "z", ""]
    assert list(trace.posterior.coords["d"]) == [0]
    assert list(trace.posterior.coords["e"]) == ["foo"]
    assert trace.posterior.coords["f"].dtype == np.float64

    trace = nutpie.sample(compiled, chains=2, seed=1234, draws=50, tune=50)
    assert trace.posterior.coords["a"].dtype == np.float32
    # pandas 3.0 changes datetime64 precision
    assert trace.posterior.coords["b"].dtype.name in [
        "datetime64[ns]",
        "datetime64[us]",
    ]
    assert trace.posterior.coords["b"].values[0] == np.datetime64("2023-01-01")
    assert list(trace.posterior.coords["c"]) == ["x", "y", "z", ""]
    assert list(trace.posterior.coords["d"]) == [0]
    assert list(trace.posterior.coords["e"]) == ["foo"]
    assert trace.posterior.coords["f"].dtype == np.float64
    assert trace.posterior.coords["g"].shape == (0,)


@pytest.fixture
def tmp_path():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


@pytest.mark.pymc
@parameterize_backends
def test_dims_model(backend, gradient_backend, request):
    import pymc.dims as pmd

    if backend == "mlx":
        request.applymarker(
            pytest.mark.xfail(
                reason="ZeroSumNormal checks its mean against atol=1e-9, which "
                "its float32 transform cannot meet",
                raises=RuntimeError,
            )
        )

    coords = {"a": range(3), "b": range(5)}
    with pm.Model(coords=coords) as model:
        print(model.dim_lengths)
        zero_sum = pmd.ZeroSumNormal("zero_sum", core_dims=("a",), dims=("a", "b"))
        pmd.Deterministic("one_sum", zero_sum + 1 / 3, dims=(..., "a"))

    compiled = nutpie.compile_pymc_model(
        model, backend=backend, gradient_backend=gradient_backend
    )
    post = nutpie.sample(compiled, chains=1).posterior
    assert post["zero_sum"].dims == ("chain", "draw", "a", "b")
    assert post["one_sum"].dims == ("chain", "draw", "b", "a")
    np.testing.assert_allclose(post["zero_sum"].sum(dim="a"), 0, atol=1e-5)
    np.testing.assert_allclose(post["one_sum"].sum(dim="a"), 1, atol=1e-5)


@pytest.mark.pymc
@parameterize_backends
def test_unnamed_shared(backend, gradient_backend):
    rng = np.random.default_rng(0)
    x = rng.normal(size=100)
    y = x + rng.normal(scale=1e-2, size=100)

    x_shared = pytensor.shared(x)

    with pm.Model() as model:
        b = pm.Normal("b", 0.0, 10.0)
        pm.Normal("obs", b * x_shared, np.sqrt(1e-2), observed=y, shape=x_shared.shape)

    compiled = nutpie.compile_pymc_model(model)
    nutpie.sample(compiled)
