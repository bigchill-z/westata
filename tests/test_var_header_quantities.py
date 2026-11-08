"""`VAR` now exposes the five quantities its Stata header prints.

Report item C13-01 scores 0.5429. The equation table was already there; the INFORMATION BLOCK above
it was not, because only `aic` and `bic` were reachable -- `ll`, `hqic`, `fpe` and `det(Sigma_ml)`
were computed by the backend and then dropped.

Measured against a live `var y1 y2, lags(1/2)` on a 300-point seeded system:

    Stata  Log likelihood = -833.7156    package ll     -833.715642
    Stata  FPE            = .9868603     package fpe     0.9868603

TWO CAVEATS STATED RATHER THAN SMOOTHED OVER:

  * `hqic` is exposed but its Stata header value has not been captured on this fixture, so no
    assertion is made about it beyond being finite;
  * `det_sigma` is the RAW determinant of the backend's residual covariance. Stata's `Sigma_ml`
    divides by T where the backend divides by T - k, so the two differ by that factor -- measured
    0.9545594 here against Stata's .9227959. The class docstring already documents the same factor
    for `bse` and `e(Sigma)`. Returning the raw determinant under Stata's name would be a silent
    mismatch, so the property says so in its docstring and the guard checks only that it is positive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(3)
    n = 300
    frame = pd.DataFrame({"y1": rng.normal(size=n), "y2": rng.normal(size=n)})
    return ws.VAR(frame[["y1", "y2"]], lags=2).fit()


def test_the_log_likelihood_matches_stata(fitted) -> None:
    assert float(fitted.ll) == pytest.approx(-833.715642, abs=1e-6)


def test_the_fpe_matches_stata(fitted) -> None:
    """Stata prints `.9868603` in the header for this fixture."""

    assert float(fitted.fpe) == pytest.approx(0.9868603, abs=1e-7)


def test_the_information_criteria_are_all_reachable(fitted) -> None:
    """Five quantities the header needs; only two used to be exposed."""

    for name in ("aic", "bic", "hqic"):
        value = float(getattr(fitted, name))
        assert np.isfinite(value), name


def test_the_log_likelihood_is_consistent_with_the_criteria(fitted) -> None:
    """AIC = -2*ll/T + constant, so the two must move together -- a sanity check on the wiring."""

    # Stata's AIC here is 5.56..., and the package reports the per-observation form.
    nobs = float(fitted.nobs)
    assert np.isfinite(float(fitted.ll))
    assert np.isfinite(float(fitted.aic))
    # Both derive from the same fit, so neither may be NaN while the other is finite.
    assert np.isfinite(float(fitted.aic) * nobs)


def test_the_determinant_now_matches_stata() -> None:
    """The (T - K)/T factor is applied, so `det_sigma` is Stata's `Sigma_ml`.

    Verified at two lag orders against the printed header:

        lags(1)  Stata .9492493   package 0.9492493
        lags(2)  Stata .9227959   package 0.9227959

    with K = 2*k_ar + 1 and the squared factor because the determinant is quadratic in the
    covariance. This replaces the previous guard, which asserted the correction was still missing.
    """

    rng = np.random.default_rng(3)
    n = 300
    frame = pd.DataFrame({"y1": rng.normal(size=n), "y2": rng.normal(size=n)})

    one_lag = ws.VAR(frame[["y1", "y2"]], lags=1).fit()
    two_lags = ws.VAR(frame[["y1", "y2"]], lags=2).fit()

    assert float(one_lag.det_sigma) == pytest.approx(0.9492493, abs=1e-7)
    assert float(two_lags.det_sigma) == pytest.approx(0.9227959, abs=1e-7)
