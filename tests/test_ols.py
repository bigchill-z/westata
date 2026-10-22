"""OLS conformance tests against statsmodels (same algebra as Stata ``regress``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from westata.errors import ConsistencyError, UnsupportedError
from westata.estimators.linear import OLS


def _data(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    g = rng.integers(0, 3, size=n)
    y = 1.0 + 2.0 * x1 - 1.5 * x2 + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "g": g})


def _statsmodels_fit(df: pd.DataFrame, cov_type: str = "nonrobust", **kw):
    X = sm.add_constant(df[["x1", "x2"]], has_constant="add")
    model = sm.OLS(df["y"], X)
    return model.fit(cov_type=cov_type, **kw)


def test_point_estimates_match_statsmodels():
    df = _data()
    res = OLS.from_formula("y ~ x1 + x2", df).fit()
    ref = _statsmodels_fit(df)
    np.testing.assert_allclose(res.params.to_numpy(), ref.params.to_numpy(), rtol=1e-12)


def test_hc1_matches_statsmodels():
    df = _data()
    res = OLS.from_formula("y ~ x1 + x2", df).fit(cov_type="HC1")
    ref = _statsmodels_fit(df, cov_type="HC1")
    np.testing.assert_allclose(res.bse.to_numpy(), ref.bse.to_numpy(), rtol=1e-12)


def test_cluster_matches_statsmodels_cr1():
    df = _data()
    groups = df["g"]
    res = OLS.from_formula("y ~ x1 + x2", df).fit(
        cov_type="cluster", cov_kwds={"groups": groups}
    )
    ref = _statsmodels_fit(
        df, cov_type="cluster", cov_kwds={"groups": groups}, use_correction=True
    )
    np.testing.assert_allclose(res.bse.to_numpy(), ref.bse.to_numpy(), rtol=1e-12)
    assert res.df_resid == groups.nunique() - 1


def test_e_scalars_are_populated():
    df = _data()
    res = OLS.from_formula("y ~ x1 + x2", df).fit()
    assert res.e["N"] == len(df)
    assert res.e["rank"] == 3
    assert res.e["df_m"] == 2
    assert res.e["df_r"] == len(df) - 3
    assert res.e["r2"] == pytest.approx(res.rsquared)


def test_stata_summary_labels():
    df = _data()
    text = OLS.from_formula("y ~ x1 + x2", df).fit().summary(style="stata")
    for token in ["Number of obs", "Coef.", "Std. err.", "P>|t|", "conf. interval"]:
        assert token in text


def test_rank_deficient_raises_consistency_error():
    df = _data()
    df["x1_copy"] = df["x1"]
    with pytest.raises(ConsistencyError):
        OLS.from_formula("y ~ x1 + x1_copy", df).fit()


def test_weights_are_rejected_in_p0():
    df = _data()
    with pytest.raises(UnsupportedError):
        OLS.from_formula("y ~ x1", df, weights=df["x1"])
