"""E3 conformance for the P10 multi-equation commands.

Reference values from ``stata_fixtures/p10_outputs.txt`` (StataNow 19.5).

=================  ====================================================
command            Stata
=================  ====================================================
sureg              y1: _cons 1.00723, x1 1.960029, x2 -.6210211
                   y2: _cons -.4903816, x2 .9412217, x3 .8024322
                   se .0492993/.040549/.0476005 and .0487194/.047139/.039197
mvreg              y1: x1 1.95372, x2 -.6277454, x3 -.0732678, _cons 1.00293
eivreg (reliab .7) x1 2.073214, x2 -.4678675, _cons .8649387
manova             Wilks' lambda .4290, df 2, F 62.15, p 0.0000
nl                 /b0 4.943526, /b1 .7912568
frontier           x1 .7813064, x2 .517943, _cons 1.047813
=================  ====================================================

The SUR standard errors are the point of the test file, not an afterthought:
``sureg``'s whole value is the cross-equation correlation, and getting that wrong
leaves every coefficient identical while changing every standard error.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import westata as ws

FIXTURES = Path(__file__).resolve().parent / "stata_fixtures"
REFERENCE = FIXTURES / "p10_outputs.txt"

if not REFERENCE.exists():
    pytest.skip("P10 fixtures missing", allow_module_level=True)


def _require(name: str):
    attribute = getattr(ws, name, None)
    if attribute is None:
        pytest.skip(f"{name} is not exposed on westata yet")
    return attribute


def sur_frame(n: int = 400, seed: int = 101) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    common = rng.normal(size=n)
    e1 = 0.7 * common + 0.7 * rng.normal(size=n)
    e2 = 0.7 * common + 0.7 * rng.normal(size=n)
    return pd.DataFrame(
        {
            "y1": 1.0 + 2.0 * x1 - 0.5 * x2 + e1,
            "y2": -0.5 + 1.0 * x2 + 0.8 * x3 + e2,
            "x1": x1,
            "x2": x2,
            "x3": x3,
        }
    )


def nl_frame(n: int = 300, seed: int = 103) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.1, 4.0, size=n))
    return pd.DataFrame({"y": 5.0 * np.exp(-0.8 * x) + rng.normal(scale=0.15, size=n), "x": x})


def eiv_frame(n: int = 500, seed: int = 107) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    true_x = rng.normal(size=n)
    observed = true_x + rng.normal(scale=0.7, size=n)
    x2 = rng.normal(size=n)
    return pd.DataFrame(
        {"y": 1.0 + 2.0 * true_x - 0.5 * x2 + rng.normal(size=n), "x1": observed, "x2": x2}
    )


def manova_frame(n: int = 240, seed: int = 109) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group = rng.integers(0, 3, size=n)
    effect = np.array([0.0, 1.0, 2.0])[group]
    return pd.DataFrame(
        {
            "y1": 1.0 + effect + rng.normal(scale=0.8, size=n),
            "y2": 2.0 + 0.5 * effect + rng.normal(scale=0.8, size=n),
            "g": group,
        }
    )


def frontier_frame(n: int = 400, seed: int = 113) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    inefficiency = np.abs(rng.normal(scale=0.4, size=n))
    return pd.DataFrame(
        {
            "y": 1.0 + 0.8 * x1 + 0.5 * x2 - inefficiency + rng.normal(scale=0.2, size=n),
            "x1": x1,
            "x2": x2,
        }
    )


# ------------------------------------------------------------------- sureg


def test_sureg_coefficients_match_stata():
    Sureg = _require("Sureg")
    result = Sureg.from_formula(
        ["y1 ~ x1 + x2", "y2 ~ x2 + x3"], sur_frame()
    ).fit()

    # Stata: y1 _cons 1.00723, x1 1.960029, x2 -.6210211
    assert float(result.params["y1:Intercept"]) == pytest.approx(1.00723, rel=1e-4)
    assert float(result.params["y1:x1"]) == pytest.approx(1.960029, rel=1e-4)
    assert float(result.params["y1:x2"]) == pytest.approx(-0.6210211, rel=1e-4)
    # Stata: y2 _cons -.4903816, x2 .9412217, x3 .8024322
    assert float(result.params["y2:Intercept"]) == pytest.approx(-0.4903816, rel=1e-4)
    assert float(result.params["y2:x2"]) == pytest.approx(0.9412217, rel=1e-4)
    assert float(result.params["y2:x3"]) == pytest.approx(0.8024322, rel=1e-4)


def test_sureg_standard_errors_match_stata():
    """The cross-equation correlation is what SUR exists for.

    A wrong Sigma leaves every coefficient identical and every standard error
    wrong, so this assertion is the one that actually validates the estimator.
    """

    Sureg = _require("Sureg")
    result = Sureg.from_formula(
        ["y1 ~ x1 + x2", "y2 ~ x2 + x3"], sur_frame()
    ).fit()

    assert float(result.bse["y1:Intercept"]) == pytest.approx(0.0492993, rel=1e-3)
    assert float(result.bse["y1:x1"]) == pytest.approx(0.040549, rel=1e-3)
    assert float(result.bse["y1:x2"]) == pytest.approx(0.0476005, rel=1e-3)
    assert float(result.bse["y2:Intercept"]) == pytest.approx(0.0487194, rel=1e-3)
    assert float(result.bse["y2:x2"]) == pytest.approx(0.047139, rel=1e-3)
    assert float(result.bse["y2:x3"]) == pytest.approx(0.039197, rel=1e-3)


def test_sureg_covariance_has_cross_equation_blocks():
    """A block-diagonal VCE would mean the equations were treated as unrelated."""

    Sureg = _require("Sureg")
    result = Sureg.from_formula(
        ["y1 ~ x1 + x2", "y2 ~ x2 + x3"], sur_frame()
    ).fit()
    covariance = np.asarray(result.cov_params(), dtype=float)

    assert covariance.shape == (6, 6)
    off_block = covariance[:3, 3:]
    assert np.any(np.abs(off_block) > 1e-12)
    np.testing.assert_allclose(covariance, covariance.T, rtol=1e-12)


def test_sureg_reports_the_cross_equation_correlation():
    """SUR is worth doing only when the errors are correlated.

    The DGP shares a common error component, so the fitted correlation must be
    substantial. An earlier version of the conformance file asserted instead that
    SUR reduces to OLS on this data, which would have been wrong twice over: the
    correlation is 0.55, and reduction is the *absence* of what SUR adds.
    """

    Sureg = _require("Sureg")
    result = Sureg.from_formula(["y1 ~ x1 + x2", "y2 ~ x2 + x3"], sur_frame()).fit()
    correlation = float(np.asarray(result.e["corr"])[0, 1])
    assert 0.3 < correlation < 0.8


# ------------------------------------------------------------------- mvreg


def test_mvreg_coefficients_match_stata():
    Mvreg = _require("Mvreg")
    result = Mvreg.from_formula("y1 y2 = x1 x2 x3", sur_frame()).fit()

    # Each equation contains ALL regressors here, unlike the sureg specification.
    assert float(result.params["y1:x1"]) == pytest.approx(1.95372, rel=1e-4)
    assert float(result.params["y1:x2"]) == pytest.approx(-0.6277454, rel=1e-4)
    assert float(result.params["y1:x3"]) == pytest.approx(-0.0732678, rel=1e-4)
    assert float(result.params["y2:x2"]) == pytest.approx(0.9381316, rel=1e-4)


# ------------------------------------------------------------------ eivreg


def test_eivreg_matches_stata():
    Eivreg = _require("Eivreg")
    result = Eivreg.from_formula("y ~ x1 + x2", eiv_frame(), reliab={"x1": 0.7}).fit()

    # Stata: x1 2.073214, x2 -.4678675, _cons .8649387
    assert float(result.params["x1"]) == pytest.approx(2.073214, rel=1e-3)
    assert float(result.params["x2"]) == pytest.approx(-0.4678675, rel=1e-3)
    assert float(result.params["Intercept"]) == pytest.approx(0.8649387, rel=1e-3)


def test_eivreg_with_perfect_reliability_is_ols():
    Eivreg = _require("Eivreg")
    frame = eiv_frame()
    result = Eivreg.from_formula("y ~ x1 + x2", frame, reliab={"x1": 1.0}).fit()
    plain = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()

    np.testing.assert_allclose(
        result.params[["Intercept", "x1", "x2"]].to_numpy(),
        plain.params[["Intercept", "x1", "x2"]].to_numpy(),
        rtol=1e-8,
    )


# ------------------------------------------------------------------ manova


def test_manova_wilks_lambda_matches_stata():
    Manova = _require("Manova")
    outcome = Manova.from_formula("y1 y2 = g", manova_frame()).fit()
    # Stata: Wilks' lambda 0.4290, F 62.15, p 0.0000
    assert float(outcome.wilks_lambda) == pytest.approx(0.4290, abs=5e-3)


# ---------------------------------------------------------------------- nl


def test_nl_matches_stata():
    Nl = _require("Nl")
    result = Nl.from_formula("y = {b0} * exp(-{b1} * x)", nl_frame()).fit()

    # Stata reports /b0 and /b1 with a leading slash.
    assert float(result.params["/b0"]) == pytest.approx(4.943526, rel=1e-3)
    assert float(result.params["/b1"]) == pytest.approx(0.7912568, rel=1e-3)


def test_nl_rejects_a_dangerous_expression():
    Nl = _require("Nl")
    with pytest.raises(ValueError):
        Nl.from_formula(
            "y = {b0} * __import__('os').system('echo x')", nl_frame()
        )


def test_nl_on_a_linear_model_reproduces_ols():
    Nl = _require("Nl")
    rng = np.random.default_rng(11)
    n = 200
    x = rng.normal(size=n)
    frame = pd.DataFrame({"y": 1.5 + 2.5 * x + rng.normal(size=n), "x": x})

    nonlinear = Nl.from_formula("y = {b0} + {b1} * x", frame).fit()
    plain = ws.OLS.from_formula("y ~ x", frame).fit()

    assert float(nonlinear.params["/b0"]) == pytest.approx(
        float(plain.params["Intercept"]), rel=1e-5
    )
    assert float(nonlinear.params["/b1"]) == pytest.approx(
        float(plain.params["x"]), rel=1e-5
    )


# ---------------------------------------------------------------- frontier


def test_frontier_matches_stata():
    Frontier = _require("Frontier")
    result = Frontier.from_formula("y ~ x1 + x2", frontier_frame()).fit()

    # Stata: x1 .7813064, x2 .517943, _cons 1.047813
    assert float(result.params["x1"]) == pytest.approx(0.7813064, rel=2e-2)
    assert float(result.params["x2"]) == pytest.approx(0.517943, rel=2e-2)
    assert float(result.params["Intercept"]) == pytest.approx(1.047813, rel=2e-2)


def test_frontier_reports_the_variance_components():
    Frontier = _require("Frontier")
    result = Frontier.from_formula("y ~ x1 + x2", frontier_frame()).fit()

    for key in ("sigma_u", "sigma_v", "lambda"):
        assert key in result.e, key
        assert np.isfinite(float(result.e[key]))
        assert float(result.e[key]) >= 0


def test_frontier_rejects_an_unsupported_distribution():
    Frontier = _require("Frontier")
    with pytest.raises(Exception):
        Frontier.from_formula(
            "y ~ x1 + x2", frontier_frame(), distribution="truncated"
        ).fit()
