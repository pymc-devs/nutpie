import math
from importlib.util import find_spec

import pytest

if find_spec("pymc") is None:
    pytest.skip("Skip pymc tests", allow_module_level=True)

import arviz as az
import numpy as np
import pymc as pm
import pytest
from scipy import stats

import nutpie

FDR_Q = 0.001
FAST_MIN_ESS = 5_000
PRECISE_MIN_ESS = 20_000

FAST_CONFIGS = [
    ("nuts-diag", {"sampler": "nuts", "adaptation": "diag"}, False),
    ("nuts-draw-diag", {"sampler": "nuts", "adaptation": "draw_diag"}, False),
    ("nuts-low-rank", {"sampler": "nuts", "adaptation": "low_rank"}, False),
    ("nuts-walnuts", {"sampler": "nuts", "walnuts": True}, False),
]

PRECISE_CONFIGS = [
    ("nuts-diag", {"sampler": "nuts", "adaptation": "diag"}, False),
    ("nuts-low-rank", {"sampler": "nuts", "adaptation": "low_rank"}, False),
    ("nuts-walnuts", {"sampler": "nuts", "walnuts": True}, False),
    ("nuts-flow", {"sampler": "nuts", "adaptation": "flow"}, True),
]

NORMAL_MEAN = 1.5
NORMAL_SD = 2.5
HALF_NORMAL_SD = 1.5


def _sample(model, config, *, draws, tune, seed, flow_adapt=False):
    if flow_adapt:
        compiled = nutpie.compile_pymc_model(
            model, backend="jax", gradient_backend="jax"
        ).with_transform_adapt(num_layers=2)
        # The MvNormal log density uses triangular_solve, which asdex cannot
        # currently inspect while detecting Hessian sparsity.
        compiled = compiled.with_hessian_sparsity(
            np.ones((compiled.n_dim, compiled.n_dim), dtype=bool)
        )
        sampler_kwargs = {"window_switch_freq": 128}
    else:
        compiled = nutpie.compile_pymc_model(model, backend="numba")
        sampler_kwargs = {}
    return nutpie.sample(
        compiled,
        draws=draws,
        tune=tune,
        chains=4,
        cores=4,
        seed=seed,
        save_warmup=False,
        progress_bar=False,
        **sampler_kwargs,
        **config,
    )


def _moment_test(posterior, values, expected, standard_deviation, *, min_ess, label):
    """Return an ESS-adjusted, two-sided p-value for one known-moment check."""
    values = np.asarray(values)
    assert values.ndim == 2, "expected chain-by-draw samples"
    dataset = posterior.to_dataset().assign(
        sampler_accuracy_statistic=(("chain", "draw"), values)
    )
    ess = float(
        az.ess(
            dataset,
            var_names=["sampler_accuracy_statistic"],
            method="mean",
        ).sampler_accuracy_statistic
    )
    assert ess >= min_ess, f"{label}: ESS={ess:.0f} is below required minimum {min_ess}"

    estimate = float(values.mean())
    mcse = standard_deviation / np.sqrt(ess)
    z_score = (estimate - expected) / mcse
    p_value = 2.0 * stats.norm.sf(abs(z_score))
    return {"label": label, "p_value": p_value, "ess": ess, "z_score": z_score}


def _assert_no_fdr_discoveries(checks, *, family):
    p_values = np.array([check["p_value"] for check in checks])
    adjusted_p_values = stats.false_discovery_control(p_values, method="bh")
    rejected = np.flatnonzero(adjusted_p_values <= FDR_Q)
    if rejected.size:
        details = "; ".join(
            f"{checks[index]['label']}: p={p_values[index]:.3g}, "
            f"adjusted p={adjusted_p_values[index]:.3g}, "
            f"z={checks[index]['z_score']:.2f}, ESS={checks[index]['ess']:.0f}"
            for index in rejected
        )
        raise AssertionError(
            f"Benjamini-Hochberg at FDR q={FDR_Q:g} found discrepancies in "
            f"the {family} family: {details}"
        )


def _normal_and_half_normal_model():
    with pm.Model() as model:
        pm.Normal("normal", mu=NORMAL_MEAN, sigma=NORMAL_SD, shape=2)
        pm.HalfNormal("positive", sigma=HALF_NORMAL_SD)
    return model


def _append_fast_checks(checks, trace, config_name):
    normal = trace.posterior.normal.values
    centered = normal - NORMAL_MEAN
    for dim in range(normal.shape[-1]):
        checks.append(
            _moment_test(
                trace.posterior,
                normal[:, :, dim],
                NORMAL_MEAN,
                NORMAL_SD,
                min_ess=FAST_MIN_ESS,
                label=f"{config_name}: Normal mean[{dim}]",
            )
        )
        checks.append(
            _moment_test(
                trace.posterior,
                centered[:, :, dim] ** 2,
                NORMAL_SD**2,
                math.sqrt(2.0) * NORMAL_SD**2,
                min_ess=FAST_MIN_ESS,
                label=f"{config_name}: Normal variance[{dim}]",
            )
        )

    positive = trace.posterior.positive.values
    half_normal_mean = HALF_NORMAL_SD * math.sqrt(2.0 / math.pi)
    half_normal_sd = HALF_NORMAL_SD * math.sqrt(1.0 - 2.0 / math.pi)
    checks.append(
        _moment_test(
            trace.posterior,
            positive,
            half_normal_mean,
            half_normal_sd,
            min_ess=FAST_MIN_ESS,
            label=f"{config_name}: HalfNormal mean",
        )
    )
    checks.append(
        _moment_test(
            trace.posterior,
            positive**2,
            HALF_NORMAL_SD**2,
            math.sqrt(2.0) * HALF_NORMAL_SD**2,
            min_ess=FAST_MIN_ESS,
            label=f"{config_name}: HalfNormal second moment",
        )
    )


@pytest.mark.pymc
def test_simple_posteriors_match_known_moments():
    """Quick moment checks across NUTS adaptations and WALNUTS."""
    checks = []
    for config_index, (config_name, config, flow_adapt) in enumerate(FAST_CONFIGS):
        trace = _sample(
            _normal_and_half_normal_model(),
            config,
            draws=10_000,
            tune=1_000,
            seed=98231 + config_index,
            flow_adapt=flow_adapt,
        )
        positive = trace.posterior.positive.values
        assert np.all(positive >= 0), f"{config_name}: HalfNormal has negative draws"
        _append_fast_checks(checks, trace, config_name)

    _assert_no_fdr_discoveries(checks, family="quick posterior-moment")


@pytest.mark.pymc
@pytest.mark.slow
def test_correlated_normal_is_unbiased_at_higher_precision():
    """More draws test the mean, marginal variances, and covariance directly."""
    mean = np.array([1.5, -2.0])
    covariance = np.array([[4.0, 3.0], [3.0, 9.0]])
    with pm.Model() as model:
        pm.MvNormal("x", mu=mean, cov=covariance)

    checks = []
    for config_index, (config_name, config, flow_adapt) in enumerate(PRECISE_CONFIGS):
        trace = _sample(
            model,
            config,
            draws=50_000,
            tune=2_000,
            seed=57421 + config_index,
            flow_adapt=flow_adapt,
        )
        samples = trace.posterior.x.values
        centered = samples - mean[None, None, :]

        for dim in range(2):
            checks.append(
                _moment_test(
                    trace.posterior,
                    samples[:, :, dim],
                    mean[dim],
                    math.sqrt(covariance[dim, dim]),
                    min_ess=PRECISE_MIN_ESS,
                    label=f"{config_name}: correlated Normal mean[{dim}]",
                )
            )
            checks.append(
                _moment_test(
                    trace.posterior,
                    centered[:, :, dim] ** 2,
                    covariance[dim, dim],
                    math.sqrt(2.0) * covariance[dim, dim],
                    min_ess=PRECISE_MIN_ESS,
                    label=f"{config_name}: correlated Normal variance[{dim}]",
                )
            )

        cross_moment = centered[:, :, 0] * centered[:, :, 1]
        cross_moment_sd = math.sqrt(
            covariance[0, 0] * covariance[1, 1] + covariance[0, 1] ** 2
        )
        checks.append(
            _moment_test(
                trace.posterior,
                cross_moment,
                covariance[0, 1],
                cross_moment_sd,
                min_ess=PRECISE_MIN_ESS,
                label=f"{config_name}: correlated Normal covariance",
            )
        )

    _assert_no_fdr_discoveries(checks, family="higher-precision posterior-moment")
