"""Weighted least squares with Stata weight semantics.

Reference: Stata ``[U] 20.24 Weighted estimation``.  The two properties that
matter most:

* ``fweight`` must reproduce the unweighted fit on the expanded data exactly.
* ``pweight`` implies ``vce(robust)`` (here HC1, Stata's robust variant).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from westata.errors import ConsistencyError, WeightError
from westata.estimators.linear import OLS


def _data(n: int = 40, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def test_fweight_matches_expanded_ols():
    df = _data()
    rng = np.random.default_rng(11)
    counts = rng.integers(1, 5, size=len(df)).astype(float)

    weighted = OLS.from_formula(
        "y ~ x1 + x2", df, weights=counts, weight_type="fweight"
    ).fit()

    expanded = df.loc[df.index.repeat(counts.astype(int))].reset_index(drop=True)
    unweighted = OLS.from_formula("y ~ x1 + x2", expanded).fit()

    np.testing.assert_allclose(weighted.params.to_numpy(), unweighted.params.to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(weighted.bse.to_numpy(), unweighted.bse.to_numpy(), rtol=1e-12)
    assert weighted.e["N"] == pytest.approx(counts.sum())
    assert weighted.e["df_r"] == pytest.approx(counts.sum() - 3)


def test_aweight_matches_statsmodels_wls():
    df = _data()
    rng = np.random.default_rng(13)
    w = rng.uniform(0.5, 3.0, size=len(df))

    res = OLS.from_formula("y ~ x1 + x2", df, weights=w, weight_type="aweight").fit()

    X = sm.add_constant(df[["x1", "x2"]], has_constant="add")
    ref = sm.WLS(df["y"], X, weights=w).fit()

    np.testing.assert_allclose(res.params.to_numpy(), ref.params.to_numpy(), rtol=1e-12)
    # aweights are normalized to sum to N and then treated as fweights; the
    # scale cancels, so the nonrobust VCE matches statsmodels WLS exactly.
    np.testing.assert_allclose(res.bse.to_numpy(), ref.bse.to_numpy(), rtol=1e-10)
    assert res.cov_type == "nonrobust"


def test_pweight_implies_robust_by_default():
    df = _data()
    rng = np.random.default_rng(17)
    w = rng.uniform(0.5, 3.0, size=len(df))

    res = OLS.from_formula("y ~ x1 + x2", df, weights=w, weight_type="pweight").fit()

    assert res.cov_type == "HC1"
    X = sm.add_constant(df[["x1", "x2"]], has_constant="add")
    ref = sm.WLS(df["y"], X, weights=w).fit(cov_type="HC1")
    np.testing.assert_allclose(res.bse.to_numpy(), ref.bse.to_numpy(), rtol=1e-10)


def test_pweight_explicit_nonrobust_is_rejected():
    df = _data()
    w = np.linspace(0.5, 3.0, len(df))
    # Stata cannot express "pweight with a nonrobust VCE", so westata refuses
    # rather than inventing a number.
    with pytest.raises(ConsistencyError, match="pweight implies"):
        OLS.from_formula("y ~ x1 + x2", df, weights=w, weight_type="pweight").fit(
            cov_type="nonrobust"
        )


def test_pweight_explicit_hc3_is_respected():
    df = _data()
    w = np.linspace(0.5, 3.0, len(df))
    res = OLS.from_formula("y ~ x1 + x2", df, weights=w, weight_type="pweight").fit(
        cov_type="HC3"
    )
    assert res.cov_type == "HC3"


def test_weights_without_type_raise():
    df = _data()
    with pytest.raises(WeightError, match="weight_type"):
        OLS.from_formula("y ~ x1 + x2", df, weights=np.ones(len(df)))


def test_weight_type_without_weights_raises():
    df = _data()
    with pytest.raises(WeightError, match="without weights"):
        OLS.from_formula("y ~ x1 + x2", df, weight_type="aweight")


def test_wrong_weight_length_raises():
    df = _data()
    with pytest.raises(WeightError, match="entries"):
        OLS.from_formula(
            "y ~ x1 + x2", df, weights=np.ones(len(df) - 1), weight_type="aweight"
        )
