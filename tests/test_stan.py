import nutpie
import pytest


def test_stan_model():
    _ = pytest.importorskip("httpstan")

    import nutpie.compile_stan

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
