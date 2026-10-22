"""E3 conformance for ``predict`` and ``esttab``, the last two E1 entries.

Fixtures were captured from StataNow 19.5.  ``esttab`` was installed from the
author's GitHub source (SSC is unreachable); ``predict`` needs nothing extra.

The tobit fixtures are the reason this file exists: westata's ``e(a,b)`` and
``ystar(a,b)`` were swapped, and only a real Stata comparison exposed it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import westata as ws

FIXTURES = Path(__file__).resolve().parent / "stata_fixtures"
DATA = Path(__file__).resolve().parents[1] / "data"


def _rel(actual, expected) -> float:
    a = np.asarray(actual, dtype=float).reshape(-1)
    e = np.asarray(expected, dtype=float).reshape(-1)
    assert a.size == e.size, f"size mismatch {a.size} vs {e.size}"
    return float(np.max(np.abs(a - e) / np.maximum(np.abs(e), 1e-12)))


def _binary_frame(n: int = 500, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    latent = -0.5 + 1.2 * x1 - 0.8 * x2 + rng.logistic(size=n)
    return pd.DataFrame({"y": (latent > 0).astype(float), "x1": x1, "x2": x2})


def _tobit_frame(n: int = 600, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    latent = 1.0 + 0.65 * x1 + 0.3 * x2 + rng.normal(scale=0.35, size=n)
    return pd.DataFrame({"y": np.maximum(latent, 2.0), "x1": x1, "x2": x2})


# -------------------------------------------------------------- predict


def test_predict_ols_xb_and_residuals_match_stata():
    path = FIXTURES / "predict_ols_values.csv"
    if not path.exists():
        pytest.skip("predict_ols_values.csv missing")
    reference = pd.read_csv(path)
    frame = pd.read_stata(DATA / "auto.dta", convert_categoricals=False)
    result = ws.OLS.from_formula("mpg ~ weight + foreign", frame).fit()

    # Measured: 1.6e-15 on xb, 5.1e-13 on residuals.
    assert _rel(result.predict(kind="xb"), reference["xb2"]) < 1e-12
    assert _rel(result.predict(kind="residuals"), reference["res2"]) < 1e-11


def test_predict_logit_pr_and_xb_match_stata():
    path = FIXTURES / "predict_logit_values.csv"
    if not path.exists():
        pytest.skip("predict_logit_values.csv missing")
    reference = pd.read_csv(path)
    result = ws.Logit.from_formula("y ~ x1 + x2", _binary_frame()).fit()

    # Measured: 3.1e-11 on pr, 3.6e-10 on xb.
    assert _rel(result.predict(kind="pr"), reference["prhat"]) < 1e-9
    assert _rel(result.predict(kind="xb"), reference["xbhat"]) < 1e-9


def test_predict_tobit_e_is_the_conditional_mean():
    path = FIXTURES / "predict_tobit_values.csv"
    if not path.exists():
        pytest.skip("predict_tobit_values.csv missing")
    reference = pd.read_csv(path)
    frame = _tobit_frame()
    result = ws.Tobit.from_formula("y ~ x1 + x2", frame, lower=2.0).fit()

    # Measured: 5.8e-10. [R] tobit postestimation: e(a,b) = E(y | a < y < b).
    assert _rel(result.predict(kind="e"), reference["e3"]) < 1e-8


def test_predict_tobit_ystar_is_the_unconditional_censored_mean():
    path = FIXTURES / "predict_tobit_values.csv"
    if not path.exists():
        pytest.skip("predict_tobit_values.csv missing")
    reference = pd.read_csv(path)
    frame = _tobit_frame()
    result = ws.Tobit.from_formula("y ~ x1 + x2", frame, lower=2.0).fit()

    # [R] tobit postestimation: ystar(a,b) = E(y*) with y* = max{a, min{y,b}}.
    assert _rel(result.predict(kind="ystar"), reference["ys3"]) < 1e-8


def test_tobit_predict_types_are_distinct():
    """Guard against the swap that this file was written to catch."""

    frame = _tobit_frame()
    result = ws.Tobit.from_formula("y ~ x1 + x2", frame, lower=2.0).fit()
    e_values = result.predict(kind="e")
    ystar_values = result.predict(kind="ystar")

    # e(a,b) is the conditional mean and therefore always inside (2, +inf);
    # ystar is the censored expectation, which for heavily censored data sits
    # near the censoring limit. They must not be identical.
    assert not np.allclose(e_values, ystar_values, rtol=1e-6)
    assert np.all(e_values > 2.0 - 1e-9)


# --------------------------------------------------------------- esttab


def test_etable_matches_esttab_coefficients():
    """etable's cells come from the same estimates esttab prints."""

    frame = pd.read_stata(DATA / "auto.dta", convert_categoricals=False)
    r1 = ws.OLS.from_formula("mpg ~ weight", frame).fit()
    r2 = ws.OLS.from_formula("mpg ~ weight + foreign", frame).fit()
    table = ws.etable([r1, r2], digits=6)

    # Values captured from ``esttab m1 m2, b(%9.6f) se(%9.6f)``:
    #   weight  -0.006009  -0.006588
    #   _cons    39.440284  41.679702
    weight_row = table.loc["weight"]
    assert float(weight_row["(1)"].rstrip("*")) == pytest.approx(
        -0.006009, abs=5e-7
    )
    assert float(weight_row["(2)"].rstrip("*")) == pytest.approx(
        -0.006588, abs=5e-7
    )

    intercept_row = table.loc["Intercept"]
    assert float(intercept_row["(1)"].rstrip("*")) == pytest.approx(
        39.440284, abs=5e-7
    )
    assert float(intercept_row["(2)"].rstrip("*")) == pytest.approx(
        41.679702, abs=5e-7
    )
