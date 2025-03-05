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
