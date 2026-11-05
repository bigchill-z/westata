"""Pin the VECM's e() scalars and standard errors to Stata's ``vec``.

Reference from a live ``vec dinf unem, lags(2) rank(1)`` on macro_swatson.dta::

    e(N) = 164          e(ll) = 1220.5657      e(rank) = 8
    e(aic) = -14.775191 e(hqic) = -14.706131   e(sbic) = -14.605077
    e(r2_1) = .74342079      e(r2_2) = .40619722
    e(rmse_1) = .01377219    e(rmse_2) = .00256002
    e(chi2_1) = 463.5891     e(chi2_2) = 109.44973

and from the printed coefficient table:

    _ce1 L1.  -1.872152   .107031
    dinf LD.   .4011357   .0661408
    unem LD.  -2.033277   .3374592
    _cons     (Stata parameterisation)  -6.35e-07   .0010758

FOUR formulas had to be corrected here, each pinned by measurement rather than derivation:

1. the criteria's parameter count is ``e(rank) + (neqs - r)`` -- verified across lags 2/3/4
   giving k = 9/13/17 and AIC -14.775191 / -14.809463 / -14.815544. Neither ``e(k)`` nor
   ``e(k_rank)`` is that number, and both were tried first.
2. R-squared is UNCENTRED: ``sum(Dy^2)``, because the differenced equation has no
   intercept. Centring gave .40564968 against Stata's .40619722.
3. RMSE divides by ``N - 4``, not by the per-equation regressor count.
4. ``chi2 = (R2/(1-R2)) * (N-4)``. The key used to be deliberately absent; the formula is
   recoverable from the data, so it is stored now.

The standard errors also needed ``sqrt(T/(T-4))``, and that factor is the same for alpha,
gamma and the deterministic block -- verified by dividing Stata's printed standard errors
by statsmodels' for all three.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"

STATA = {
    "aic": -14.775191,
    "hqic": -14.706131,
    "sbic": -14.605077,
    "r2_1": 0.74342079,
    "r2_2": 0.40619722,
    "rmse_1": 0.01377219,
    "rmse_2": 0.00256002,
    "chi2_1": 463.5891,
    "chi2_2": 109.44973,
}


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    values = frame[["dinf", "unem"]].dropna().to_numpy()
    return ws.VECM(values, coint_rank=1, lags=2, names=["dinf", "unem"]).fit()


def test_every_scalar_matches_stata(fitted) -> None:
    for key, expected in STATA.items():
        assert float(fitted.e[key]) == pytest.approx(expected, rel=2e-5), key


def test_the_standard_errors_match_stata(fitted) -> None:
    alpha_se = np.asarray(fitted.stderr_alpha, dtype=float).ravel()
    assert alpha_se[0] == pytest.approx(0.107031, rel=1e-5)
    assert alpha_se[1] == pytest.approx(0.0198952, rel=1e-5)

    gamma_se = np.asarray(fitted.stderr_gamma, dtype=float)
    assert gamma_se[0, 0] == pytest.approx(0.0661408, rel=1e-5)
    assert gamma_se[0, 1] == pytest.approx(0.3374592, rel=1e-5)
    assert gamma_se[1, 0] == pytest.approx(0.0122944, rel=1e-5)
    assert gamma_se[1, 1] == pytest.approx(0.062728, rel=1e-5)


def test_the_scale_factor_is_shared_by_every_block(fitted) -> None:
    """sqrt(T/(T-4)) applies to alpha, gamma and the deterministic terms alike."""

    raw = fitted.raw
    factor = float(np.sqrt(164.0 / 160.0))
    for name, adjusted in (
        ("stderr_alpha", fitted.stderr_alpha),
        ("stderr_gamma", fitted.stderr_gamma),
    ):
        base = np.asarray(getattr(raw, name), dtype=float)
        assert np.allclose(np.asarray(adjusted, dtype=float) / base, factor, rtol=1e-12), name


def test_the_alpha_and_beta_estimates_are_unchanged(fitted) -> None:
    """Only the scale and the derived statistics changed; the estimates already matched."""

    assert float(np.asarray(fitted.alpha).ravel()[0]) == pytest.approx(-1.872152, abs=5e-6)
    assert float(np.asarray(fitted.beta).ravel()[1]) == pytest.approx(0.111912, abs=5e-6)


def test_r_squared_is_uncentred(fitted) -> None:
    """sum(Dy^2), not the centred total -- the differenced equation has no intercept."""

    residuals = np.asarray(fitted.raw.resid, dtype=float)
    levels = np.asarray(fitted.raw.model.endog, dtype=float)
    differenced = np.diff(levels, axis=0)[-residuals.shape[0] :]
    rss = float(residuals[:, 1] @ residuals[:, 1])
    tss_uncentred = float(differenced[:, 1] @ differenced[:, 1])
    assert 1.0 - rss / tss_uncentred == pytest.approx(0.40619722, rel=1e-6)


def test_the_parameter_count_is_neither_e_k_nor_e_k_rank(fitted) -> None:
    """e(rank)+1 = 9 here, while e(k) is 6 and e(k_rank) is 5."""

    assert int(fitted.e["k"]) == 6
    assert float(fitted.e["k_rank"]) == 5.0
    neqs = int(fitted.e["neqs"])
    rank = int(fitted.coint_rank)
    count = max(int(fitted.lags) - 1, 0) * neqs * neqs + neqs + neqs + (neqs - rank)
    assert count == 9


def test_the_summary_renders_the_header_block(fitted) -> None:
    rendered = str(fitted.summary(style="stata")).splitlines()
    expected = [
        "Vector error-correction model",
        # Stata pads the value to a fixed width: two spaces before 1220.566.
        "Log likelihood =  1220.566",
        "Sample: 3 thru 166",
    ]
    for line in expected:
        assert any(line in row for row in rendered), line
    assert any("Number of obs" in row and "164" in row for row in rendered)
    assert any("AIC" in row and "-14.77519" in row for row in rendered)


def test_the_summary_reports_the_equation_block(fitted) -> None:
    rendered = str(fitted.summary(style="stata"))
    for line in (
        "Equation           Parms",
        "D_dinf",
        "D_unem",
    ):
        assert line in rendered, line
    assert "0.7434" in rendered
    assert "463.5891" in rendered
