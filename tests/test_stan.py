import numpy as np
import pytest

import nutpie


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


@pytest.mark.slow
def test_stan_flow():
    model = """
    parameters {
        real a;
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
