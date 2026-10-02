from importlib.util import find_spec

import pytest

if find_spec("bridgestan") is None:
    pytest.skip("Skip stan tests", allow_module_level=True)

import numpy as np
import pytest
import scipy.sparse

import nutpie


@pytest.mark.stan
def test_stan_model():
    model = """
    data {}
    parameters {
        real a;
    }
    model {
        a ~ normal(0, 1);
    }
    """

    compiled_model = nutpie.compile_stan_model(code=model)
    trace = nutpie.sample(compiled_model)
    trace.posterior.a  # noqa: B018


@pytest.mark.stan
def test_stan_model_low_rank():
    model = """
    data {}
    parameters {
        real a;
    }
    model {
        a ~ normal(0, 1);
    }
    """

    compiled_model = nutpie.compile_stan_model(code=model)
    trace = nutpie.sample(compiled_model, adaptation="low_rank")
    trace.posterior.a  # noqa: B018


@pytest.mark.stan
def test_empty():
    model = """
    data {}
    parameters {
        array[0] real a;
    }
    model {
        a ~ normal(0, 1);
    }
    """

    compiled_model = nutpie.compile_stan_model(code=model)
    nutpie.sample(compiled_model)
    # TODO: Variable `a` is missing because of this bridgestan issue:
    # https://github.com/roualdes/bridgestan/issues/278
    # assert trace.posterior.a.shape == (0, 1000)


@pytest.mark.stan
def test_seed():
    model = """
    data {}
    parameters {
        real a;
    }
    model {
        a ~ normal(0, 1);
    }
    generated quantities {
        real b = normal_rng(0, 1);
    }
    """

    compiled_model = nutpie.compile_stan_model(code=model)
    trace = nutpie.sample(compiled_model, seed=42)
    trace2 = nutpie.sample(compiled_model, seed=42)
    trace3 = nutpie.sample(compiled_model, seed=43)

    assert np.allclose(trace.posterior.a, trace2.posterior.a)
    assert np.allclose(trace.posterior.b, trace2.posterior.b)

    assert not np.allclose(trace.posterior.a, trace3.posterior.a)
    assert not np.allclose(trace.posterior.b, trace3.posterior.b)
    # Check that all chains are pairwise different
    for i in range(len(trace.posterior.a)):
        for j in range(i + 1, len(trace.posterior.a)):
            assert not np.allclose(trace.posterior.a[i], trace.posterior.a[j])
            assert not np.allclose(trace.posterior.b[i], trace.posterior.b[j])
    # Check that all chains are pairwise different between seeds
    for i in range(len(trace.posterior.a)):
        for j in range(len(trace3.posterior.a)):
            assert not np.allclose(trace.posterior.a[i], trace3.posterior.a[j])
            assert not np.allclose(trace.posterior.b[i], trace3.posterior.b[j])


@pytest.mark.stan
def test_nested():
    # Adapted from
    # https://github.com/stan-dev/stanio/blob/main/test/data/tuples/output.stan
    model = """
    parameters {
    real a;
    }
    model {
    a ~ normal(0, 1);
    }
    generated quantities {
    real base = normal_rng(0, 1);
    int base_i = to_int(normal_rng(10, 10));

    tuple(real, real) pair = (base, base * 2);

    tuple(real, tuple(int, complex)) nested = (base * 3, (base_i, base * 4.0i));
    array[2] tuple(real, real) arr_pair = {pair, (base * 5, base * 6)};

    array[3] tuple(tuple(real, tuple(int, complex)), real) arr_very_nested
        = {(nested, base*7), ((base*8, (base_i*2, base*9.0i)), base * 10), (nested, base*11)};

    array[3,2] tuple(real, real) arr_2d_pair = {{(base * 12, base * 13), (base * 14, base * 15)},
                                                {(base * 16, base * 17), (base * 18, base * 19)},
                                                {(base * 20, base * 21), (base * 22, base * 23)}};

    real basep1 = base + 1, basep2 = base + 2;
    real basep3 = base + 3, basep4 = base + 4, basep5 = base + 5;
    array[2,3] tuple(array[2] tuple(real, vector[2]), matrix[4,5]) ultimate =
        {
        {(
            {(base, [base *2, base *3]'), (base *4, [base*5, base*6]')},
            to_matrix(linspaced_vector(20, 7, 11), 4, 5) * base
            ),
        (
            {(basep1, [basep1 *2, basep1 *3]'), (basep1 *4, [basep1*5, basep1*6]')},
            to_matrix(linspaced_vector(20, 7, 11), 4, 5) * basep1
            ),
            (
            {(basep2, [basep2 *2, basep2 *3]'), (basep2 *4, [basep2*5, basep2*6]')},
            to_matrix(linspaced_vector(20, 7, 11), 4, 5) * basep2
        )
        },
        {(
            {(basep3, [basep3 *2, basep3 *3]'), (basep3 *4, [basep3*5, basep3*6]')},
            to_matrix(linspaced_vector(20, 7, 11), 4, 5) * basep3
            ),
        (
            {(basep4, [basep4 *2, basep4 *3]'), (basep4 *4, [basep4*5, basep4*6]')},
            to_matrix(linspaced_vector(20, 7, 11), 4, 5) * basep4
            ),
            (
            {(basep5, [basep5 *2, basep5 *3]'), (basep5 *4, [basep5*5, basep5*6]')},
            to_matrix(linspaced_vector(20, 7, 11), 4, 5) * basep5
        )
        }};

    // Complex containers, where the real and imaginary parts are interleaved
    // in the output of stan
    complex_vector[3] cv;
    complex_matrix[2, 3] cm;
    array[2] complex ca;
    for (i in 1:3) {
        cv[i] = to_complex(base * i, base * (10 + i));
    }
    for (i in 1:2) {
        for (j in 1:3) {
            cm[i, j] = to_complex(base * (10 * i + j), base * (100 + 10 * i + j));
        }
        ca[i] = to_complex(base * (200 + i), base * (300 + i));
    }
    }
    """

    compiled = nutpie.compile_stan_model(code=model)
    tr = nutpie.sample(compiled, chains=6)
    base = tr.posterior.base

    assert np.allclose(tr.posterior["nested:2:2.imag"], 4 * base)
    assert np.allclose(tr.posterior["nested:2:2.real"], 0.0)

    assert np.allclose(tr.posterior["ultimate.1.1:1.1:1"], base)
    assert np.allclose(tr.posterior["ultimate.1.2:1.1:1"], base + 1)
    assert np.allclose(tr.posterior["ultimate.1.3:1.1:1"], base + 2)
    assert np.allclose(tr.posterior["ultimate.2.1:1.1:1"], base + 3)
    assert np.allclose(tr.posterior["ultimate.2.2:1.1:1"], base + 4)
    assert np.allclose(tr.posterior["ultimate.2.3:1.1:1"], base + 5)

    assert tr.posterior["ultimate.2.1:1.1:2"].shape == (6, 1000, 2)
    assert np.allclose(
        tr.posterior["ultimate.2.3:1.1:2"].values[:, :, 0], 2 * (base + 5)
    )
    assert np.allclose(
        tr.posterior["ultimate.2.3:1.1:2"].values[:, :, 1], 3 * (base + 5)
    )
    assert np.allclose(tr.posterior["base_i"], tr.posterior.base_i.astype(int))

    def check_complex(values, base):
        base = np.asarray(base)[..., None]
        idx = np.arange(1, 4)
        assert np.allclose(values["cv.real"], base * idx)
        assert np.allclose(values["cv.imag"], base * (10 + idx))
        base = base[..., None]
        idx = 10 * np.arange(1, 3)[:, None] + np.arange(1, 4)[None, :]
        assert np.allclose(values["cm.real"], base * idx)
        assert np.allclose(values["cm.imag"], base * (100 + idx))
        base = base[..., 0]
        idx = np.arange(1, 3)
        assert np.allclose(values["ca.real"], base * (200 + idx))
        assert np.allclose(values["ca.imag"], base * (300 + idx))

    check_complex(tr.posterior, base)


@pytest.mark.stan
def test_stan_model_data():
    model = """
    data {
        complex x;
    }
    parameters {
        real a;
    }
    model {
        a ~ normal(0, 1);
    }
    """

    compiled_model = nutpie.compile_stan_model(code=model)
    with pytest.raises(RuntimeError):
        trace = nutpie.sample(compiled_model)
    trace = nutpie.sample(compiled_model.with_data(x=np.array(3.0j)))
    trace.posterior.a  # noqa: B018


@pytest.mark.stan
def test_stan_memory_order():
    model = """
    data {
        real x;
    }
    parameters {
        real a;
    }
    model {
        a ~ normal(0, 1);
    }
    generated quantities {
        array[2, 3] matrix[5, 7] b;
        real count = 0;
        for (i in 1:2)
            for (j in 1:3) {
                for (k in 1:5) {
                    for (n in 1:7) {
                        b[i, j][k, n] = count;
                        count = count + 1;
                    }
                }
            }
        }
    """

    compiled_model = nutpie.compile_stan_model(code=model)
    with pytest.raises(RuntimeError):
        trace = nutpie.sample(compiled_model)
    trace = nutpie.sample(compiled_model.with_data(x=np.array(3.0)))
    trace.posterior.a  # noqa: B018
    assert trace.posterior.b.shape == (6, 1000, 2, 3, 5, 7)
    b = trace.posterior.b.isel(chain=0, draw=0)
    count = 0
    for i in range(2):
        for j in range(3):
            for k in range(5):
                for n in range(7):
                    assert float(b[i, j, k, n]) == count
                    count += 1


@pytest.mark.flow
@pytest.mark.stan
def test_stan_flow():
    model = """
    parameters {
        array[5] real a;
        real<lower=0> b;
    }
    model {
        a ~ normal(0, 1);
        b ~ normal(0, 1);
    }
    """
    import jax

    old = jax.config.update("jax_enable_x64", True)
    try:
        # The triangular flow detects the Hessian sparsity by default
        compiled_model = nutpie.compile_stan_model(
            code=model, ad_hessian=True
        ).with_transform_adapt(
            num_layers=2,
            nn_width=4,
        )
        trace = nutpie.sample(compiled_model, adaptation="flow", tune=2000, chains=1)
        assert float(trace.sample_stats.fisher_distance.mean()) < 0.1
        trace.posterior.a  # noqa: B018
    finally:
        jax.config.update("jax_enable_x64", old)


# TODO: There are small numerical differences between linux and windows.
# We should figure out if they originate in stan or in nutpie.
@pytest.mark.array_compare(atol=1e-4, rtol=1e-4)
@pytest.mark.stan
def test_deterministic_sampling_stan():
    model = """
    parameters {
        real<lower=0> a;
    }
    model {
        a ~ normal(0, 1);
    }
    generated quantities {
        real b = normal_rng(0, 1) + a;
    }
    """

    compiled_model = nutpie.compile_stan_model(code=model)
    trace = nutpie.sample(compiled_model, chains=2, seed=123, draws=100, tune=100)
    trace2 = nutpie.sample(compiled_model, chains=2, seed=123, draws=100, tune=100)
    np.testing.assert_array_max_ulp(trace.posterior.a.values, trace2.posterior.a.values)
    np.testing.assert_array_max_ulp(trace.posterior.b.values, trace2.posterior.b.values)
    return trace.posterior.a.isel(draw=slice(None, 10)).values


@pytest.mark.stan
def test_stan_init_point_fn():
    model = """
    parameters {
        real a;
        vector[2] b;
    }
    model {
        a ~ normal(0, 1);
        b ~ normal(0, 1);
    }
    """

    compiled_model = nutpie.compile_stan_model(code=model)

    calls = []

    def init_point(model, rng, chain_id):
        assert isinstance(rng, np.random.Generator)
        calls.append(chain_id)
        return rng.normal(size=model.n_dim)

    compiled_model = compiled_model.with_init_point_fn(init_point)
    trace = nutpie.sample(compiled_model, chains=3, tune=50, draws=50)
    assert sorted(calls) == [0, 1, 2]
    trace.posterior.b  # noqa: B018

    # The init function survives later data updates
    calls.clear()
    nutpie.sample(compiled_model.with_data(), chains=2, tune=50, draws=50)
    assert sorted(calls) == [0, 1]


@pytest.mark.stan
def test_stan_init_point_fn_dict():
    model = """
    parameters {
        real<lower=0> sigma;
        matrix[2, 3] m;
    }
    model {
        sigma ~ normal(0, 1);
        to_vector(m) ~ normal(0, sigma);
    }
    """

    compiled_model = nutpie.compile_stan_model(code=model)

    calls = []

    def init_point(model, rng, chain_id):
        calls.append(chain_id)
        return {"sigma": 1.5, "m": rng.normal(size=(2, 3))}

    compiled_model = compiled_model.with_init_point_fn(init_point)
    nutpie.sample(compiled_model, chains=2, tune=50, draws=50)
    assert sorted(calls) == [0, 1]


@pytest.mark.stan
def test_stan_init_point_fn_partial_dict():
    model = """
    parameters {
        real<lower=0> sigma;
        matrix[2, 3] m;
    }
    model {
        sigma ~ normal(0, 1);
        to_vector(m) ~ normal(0, sigma);
    }
    """

    compiled_model = nutpie.compile_stan_model(code=model)

    compiled_model = compiled_model.with_init_point_fn(
        lambda model, rng, chain_id: {"sigma": 1.5}
    )
    trace = nutpie.sample(compiled_model, chains=2, tune=50, draws=50)
    trace.posterior.m  # noqa: B018


@pytest.mark.stan
def test_stan_init_point_fn_retry():
    model = """
    parameters {
        real a;
    }
    model {
        // Half of the random initial points have an infinite log density
        if (a < 0)
            target += negative_infinity();
        a ~ normal(0, 1);
    }
    """

    compiled_model = nutpie.compile_stan_model(code=model)
    compiled_model = compiled_model.with_init_point_fn(lambda model, rng, chain_id: {})
    trace = nutpie.sample(compiled_model, chains=4, tune=50, draws=50)
    assert (trace.posterior.a >= 0).all()


@pytest.mark.stan
def test_stan_init_point_fn_errors():
    model = """
    parameters {
        real<lower=0> sigma;
    }
    model {
        sigma ~ normal(0, 1);
    }
    """

    compiled_model = nutpie.compile_stan_model(code=model)

    # Values outside of the constraints are retried, and eventually fail
    bad_value = compiled_model.with_init_point_fn(
        lambda model, rng, chain_id: {"sigma": -1.0}
    )
    with pytest.raises(RuntimeError, match="initialization"):
        nutpie.sample(bad_value, chains=1, tune=10, draws=10)

    # Unknown names are fatal
    unknown = compiled_model.with_init_point_fn(
        lambda model, rng, chain_id: {"sgima": 1.0}
    )
    with pytest.raises(RuntimeError, match="sgima"):
        nutpie.sample(unknown, chains=1, tune=10, draws=10)


@pytest.mark.stan
def test_stan_constrain_unconstrain():
    model = """
    parameters {
        real<lower=0> sigma;
        matrix[2, 3] m;
    }
    transformed parameters {
        real sigma2 = sigma^2;
    }
    model {
        sigma ~ normal(0, 1);
        to_vector(m) ~ normal(0, sigma);
    }
    generated quantities {
        real draw = normal_rng(0, 1);
    }
    """

    compiled_model = nutpie.compile_stan_model(code=model)

    m = np.arange(6.0).reshape(2, 3)
    point = compiled_model.unconstrain(sigma=2.0, m=m)
    assert point.shape == (compiled_model.n_dim,)
    np.testing.assert_allclose(point[0], np.log(2.0))

    # Stan stores matrices in column-major order
    values = compiled_model.constrain(point)
    np.testing.assert_allclose(values, [2.0, *m.ravel(order="F")])

    values = compiled_model.constrain(point, include_tp=True, include_gq=True, seed=1)
    assert values.shape == (9,)
    np.testing.assert_allclose(values[7], 4.0)


@pytest.mark.stan
def test_stan_init_mean_deprecated():
    model = """
    parameters {
        real a;
    }
    model {
        a ~ normal(0, 1);
    }
    """

    compiled_model = nutpie.compile_stan_model(code=model)
    with pytest.warns(FutureWarning, match="init_mean"):
        nutpie.sample(compiled_model, init_mean=np.zeros(1), tune=10, draws=10)


HESSIAN_MODEL = """
parameters {
    vector[4] x;
    real<lower=0> s;
}
model {
    s ~ normal(0, 1);
    x[1] ~ normal(0, 1);
    x[2] ~ normal(x[1], 1);
    x[3] ~ normal(x[2], s);
    x[4] ~ normal(0, 1);
}
"""


@pytest.mark.stan
def test_stan_hessian_sparsity():
    compiled_model = nutpie.compile_stan_model(code=HESSIAN_MODEL, ad_hessian=True)

    # The sparsity depends on the data
    with pytest.raises(ValueError, match="with_data"):
        compiled_model.with_hessian_sparsity()
    compiled_model = compiled_model.with_data()
    assert compiled_model.hessian_sparsity is None

    # Unconstrained order: x[1], x[2], x[3], x[4], log(s)
    expected = np.eye(5, dtype=bool)
    for i, j in [(0, 1), (1, 2), (1, 4), (2, 4)]:
        expected[i, j] = expected[j, i] = True

    with_sparsity = compiled_model.with_hessian_sparsity(seed=1)
    pattern = with_sparsity.hessian_sparsity
    assert isinstance(pattern, scipy.sparse.csr_array)
    assert pattern.dtype == bool
    np.testing.assert_array_equal(pattern.toarray(), expected)
    assert "hessian sparsity: 4 of 10 pairs nonzero" in repr(with_sparsity)

    # An explicit pattern, dense or sparse, is made symmetric, and the size is
    # checked
    explicit = compiled_model.with_hessian_sparsity(np.triu(expected))
    np.testing.assert_array_equal(explicit.hessian_sparsity.toarray(), expected)
    explicit = compiled_model.with_hessian_sparsity(
        scipy.sparse.coo_array(np.triu(expected))
    )
    np.testing.assert_array_equal(explicit.hessian_sparsity.toarray(), expected)
    with pytest.raises(ValueError, match="shape"):
        compiled_model.with_hessian_sparsity(np.eye(4, dtype=bool))

    # New data resets it
    assert with_sparsity.with_data().hessian_sparsity is None

    # The points come from the init point function
    calls = []

    def init_point(model, rng, chain_id):
        calls.append(chain_id)
        return {"s": 1.0 + rng.uniform()}

    pattern = (
        compiled_model.with_init_point_fn(init_point)
        .with_hessian_sparsity(num_points=3, seed=1)
        .hessian_sparsity
    )
    np.testing.assert_array_equal(pattern.toarray(), expected)
    assert calls == [0, 1, 2]


@pytest.mark.stan
def test_stan_hessian_sparsity_hierarchical():
    model = """
    data {
        int<lower=0> N;
        vector[N] y;
    }
    parameters {
        real mu;
        real<lower=0> tau;
        vector[N] theta;
    }
    model {
        mu ~ normal(0, 1);
        tau ~ normal(0, 1);
        theta ~ normal(mu, tau);
        y ~ normal(theta, 1);
    }
    """
    N = 100
    compiled_model = nutpie.compile_stan_model(code=model, ad_hessian=True).with_data(
        N=N, y=np.zeros(N)
    )

    # Unconstrained order: mu, log(tau), theta. The hyperparameters interact
    # with each other and with all theta, the theta only with themselves.
    expected = np.eye(N + 2, dtype=bool)
    expected[:2, :] = True
    expected[:, :2] = True

    # A small Bloom filter, so that candidates are not found by unit vectors
    compiled_model = compiled_model.with_hessian_sparsity(seed=1, bloom_size=16)
    np.testing.assert_array_equal(compiled_model.hessian_sparsity.toarray(), expected)

    # The hyperparameters come first, and then there is no fill
    factorization = compiled_model.with_factorization().factorization
    assert set(factorization.order[:2]) == {0, 1}
    assert factorization.num_fill == 0
    assert factorization.max_parents == 2
    assert factorization.num_levels == 3
    by_variable = factorization.by_variable()
    assert by_variable.loc[("tau", "theta"), "hessian"] == N
    # Only interacting pairs of variables are listed
    assert ("theta", "theta") not in by_variable.index

    # Parameters at the front, given by variable or parameter name
    factorization = compiled_model.with_factorization(front=["theta.3"]).factorization
    assert factorization.order[0] == 4
    factorization = compiled_model.with_factorization(
        order="amd", front=["tau", "mu"]
    ).factorization
    np.testing.assert_array_equal(factorization.order[:2], [1, 0])

    summary = factorization.summary()
    assert list(summary.index[:2]) == ["tau", "mu"]
    assert summary.loc["theta.1", "parents"] == 2

    assert "factorization: amd (with front)" in repr(
        compiled_model.with_factorization(order="amd", front=["tau", "mu"])
    )


@pytest.mark.stan
def test_stan_hessian_vector_product():
    model = """
    data {
        matrix[3, 3] prec;
    }
    parameters {
        vector[3] x;
    }
    model {
        x ~ multi_normal_prec(rep_vector(0, 3), prec);
    }
    """
    prec = np.array([[2.0, 0.5, 0.0], [0.5, 1.0, -0.3], [0.0, -0.3, 1.5]])
    compiled_model = nutpie.compile_stan_model(code=model, ad_hessian=True).with_data(
        prec=prec
    )

    rng = np.random.default_rng(1)
    point = rng.normal(size=3)
    vector = rng.normal(size=3)

    logp, hvp = compiled_model.hessian_vector_product(point, vector)
    np.testing.assert_allclose(logp, -0.5 * point @ prec @ point)
    np.testing.assert_allclose(hvp, -prec @ vector)


@pytest.mark.stan
def test_stan_hessian_requires_ad_hessian():
    compiled_model = nutpie.compile_stan_model(code=HESSIAN_MODEL).with_data()
    assert not compiled_model._make_model().ad_hessian

    with pytest.raises(RuntimeError, match="ad_hessian=True"):
        compiled_model.with_hessian_sparsity()
    with pytest.raises(RuntimeError, match="ad_hessian=True"):
        compiled_model.hessian_vector_product(np.zeros(5), np.ones(5))


@pytest.mark.stan
def test_stan_cache_key_ad_hessian():
    from nutpie.compile_stan import _stan_cache_key

    assert _stan_cache_key("code", None, None) != _stan_cache_key(
        "code", None, None, ad_hessian=True
    )


@pytest.mark.stan
def test_stan_repr():
    compiled_model = nutpie.compile_stan_model(code=HESSIAN_MODEL)
    assert repr(compiled_model) == (
        "CompiledStanModel 'model' (no data, call with_data)"
    )
    compiled_model = (
        compiled_model.with_data()
        .with_transform_adapt(num_layers=2)
        .with_init_point_fn(lambda model, rng, chain_id: {})
    )
    assert repr(compiled_model) == (
        "CompiledStanModel 'model' (n_dim=5)\n"
        "  transform adapt: num_layers=2\n"
        "  init point fn: <lambda>"
    )


_UNCONSTRAINED_MODEL = """
parameters {
    real a;
    real<lower=0> s;
    vector[2] b;
}
model {
    a ~ normal(0, 1);
    s ~ normal(0, 1);
    b ~ normal(0, 1);
}
"""


@pytest.mark.stan
@pytest.mark.parametrize("storage", ["arrow", "zarr"])
def test_stan_unconstrained_parameter_coords(storage, tmp_path):
    compiled = nutpie.compile_stan_model(code=_UNCONSTRAINED_MODEL)
    names = compiled.with_data().model.unconstrained_names()
    assert len(names) == 4
    # Only in the trace, not a coord of the model
    assert "unconstrained_parameter" not in (compiled.coords or {})

    kwargs = {}
    if storage == "zarr":
        path = tmp_path / "trace.zarr"
        path.mkdir()
        kwargs["zarr_store"] = nutpie.zarr_store.LocalStore(str(path))
    trace = nutpie.sample(
        compiled,
        chains=1,
        tune=50,
        draws=10,
        store_unconstrained=True,
        progress_bar=False,
        seed=1,
        **kwargs,
    )
    coord = trace.sample_stats.coords["unconstrained_parameter"]
    assert [str(name) for name in coord.values] == names
    assert compiled._unconstrained_parameters() == names


@pytest.mark.stan
def test_stan_reserved_unconstrained_parameter():
    with pytest.raises(ValueError, match="unconstrained_parameter"):
        nutpie.compile_stan_model(
            code=_UNCONSTRAINED_MODEL, coords={"unconstrained_parameter": [0, 1]}
        )
    compiled = nutpie.compile_stan_model(code=_UNCONSTRAINED_MODEL)
    with pytest.raises(ValueError, match="unconstrained_parameter"):
        compiled.with_coords(unconstrained_parameter=[0, 1])
    with pytest.raises(ValueError, match="unconstrained_parameter"):
        compiled.with_dims(b=["unconstrained_parameter"])
