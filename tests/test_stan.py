from importlib.util import find_spec
import pytest

if find_spec("bridgestan") is None:
    pytest.skip("Skip stan tests", allow_module_level=True)

import numpy as np
import pytest

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
        compiled_model = nutpie.compile_stan_model(code=model).with_transform_adapt(
            num_layers=2,
            nn_width=4,
        )
        trace = nutpie.sample(compiled_model, transform_adapt=True, tune=2000, chains=1)
        assert float(trace.sample_stats.fisher_distance.mean()) < 0.1
        trace.posterior.a  # noqa: B018
    finally:
        jax.config.update("jax_enable_x64", old)


# TODO: There are small numerical differences between linux and windows.
# We should figure out if they originate in stan or in nutpie.
@pytest.mark.array_compare(atol=1e-4)
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
    np.testing.assert_allclose(trace.posterior.a.values, trace2.posterior.a.values)
    np.testing.assert_allclose(trace.posterior.b.values, trace2.posterior.b.values)
    return trace.posterior.a.isel(draw=slice(None, 10)).values
