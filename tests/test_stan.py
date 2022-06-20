import nutpie
import pytest

@pytest.mark.xfail
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

    compiled_model = nutpie.compile_stan_model(data={}, code=model)
    trace = nutpie.sample(compiled_model)
    trace.posterior.a
