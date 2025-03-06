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
def test_stan_model_data():
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
    """

    compiled_model = nutpie.compile_stan_model(code=model)
    with pytest.raises(RuntimeError):
        trace = nutpie.sample(compiled_model)
    trace = nutpie.sample(compiled_model.with_data(x=np.array(3.0)))
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

    compiled_model = nutpie.compile_stan_model(code=model).with_transform_adapt(
        num_layers=2,
        nn_width=4,
        num_diag_windows=6,
    )
    trace = nutpie.sample(
        compiled_model, transform_adapt=True, window_switch_freq=150, tune=600, chains=1
    )
    trace.posterior.a  # noqa: B018
