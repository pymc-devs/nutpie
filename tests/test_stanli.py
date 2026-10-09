from importlib.util import find_spec

import pytest

if find_spec("stanli") is None:
    pytest.skip("Skip stanli tests", allow_module_level=True)

import numpy as np

import nutpie

MODEL = """
data {
    int N;
    vector[N] y;
}
parameters {
    real mu;
    real<lower=0> sigma;
}
transformed parameters {
    real log_sigma = log(sigma);
}
model {
    mu ~ normal(0, 10);
    sigma ~ normal(0, 5);
    y ~ normal(mu, sigma);
}
generated quantities {
    real y_rep = normal_rng(mu, sigma);
}
"""


@pytest.mark.stan
def test_stanli_sample():
    y = np.random.default_rng(0).normal(3.0, 0.5, size=200)
    compiled = nutpie.compile_stan_model(code=MODEL, backend="stanli")
    compiled = compiled.with_data(N=len(y), y=y)
    assert compiled.n_dim == 2

    trace = nutpie.sample(compiled, draws=500, tune=300, chains=2, seed=1)
    assert abs(float(trace.posterior.mu.mean()) - y.mean()) < 0.1
    assert abs(float(trace.posterior.sigma.mean()) - y.std()) < 0.1
    np.testing.assert_allclose(
        trace.posterior.log_sigma, np.log(trace.posterior.sigma), rtol=1e-12
    )
    assert "y_rep" in trace.posterior


@pytest.mark.stan
def test_stanli_matches_constrain():
    compiled = nutpie.compile_stan_model(code=MODEL, backend="stanli").with_data(
        N=2, y=[1.0, 2.0]
    )
    values = compiled.constrain(np.array([0.5, 0.0]), include_tp=True)
    np.testing.assert_allclose(values, [0.5, 1.0, 0.0])


@pytest.mark.stan
def test_stanli_unsupported_args():
    with pytest.raises(ValueError, match="ad_hessian"):
        nutpie.compile_stan_model(code=MODEL, backend="stanli", ad_hessian=True)
    with pytest.raises(ValueError, match="Unknown Stan backend"):
        nutpie.compile_stan_model(code=MODEL, backend="cmdstan")
