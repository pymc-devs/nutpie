import pytest

import nutpie


def test_stan_api_reports_when_rust_feature_is_disabled():
    if hasattr(nutpie._lib, "StanModel"):
        pytest.skip("Rust extension was built with Stan support")

    with pytest.raises(ImportError, match="Stan support is not enabled"):
        nutpie.compile_stan_model(
            code="parameters { real x; } model { x ~ normal(0, 1); }",
            backend="stanli",
        )
