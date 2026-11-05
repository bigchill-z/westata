"""Pin the VECM covariance and standard errors to Stata's ``vec`` scale.

Reference, from a live ``vec dinf unem, lags(2) rank(1)`` on macro_swatson.dta (n = 164):

    e(V)[0,0] = .01145563    ->  se(L._ce1 in D_dinf) = .107031
    e(V)[4,4] = .00039582    ->  se(L._ce1 in D_unem) = .0198952

statsmodels' ``VECMResults`` divides the residual variance by ``T - 4`` where Stata uses
``T``, so every standard error is low by ``sqrt((T-4)/T) = 1/1.012423``:

    statsmodels .1057177 / .0196511
    Stata      .107031  / .0198952

The factor is the SAME for alpha, gamma and the constant, which is how a degrees-of-freedom
scale difference can be told apart from a parameterisation error.

Two defects were found here:

* ``VECMResults.cov_params()`` called ``self.raw.cov_params()``, a method statsmodels'
  VECMResults does not HAVE (it exposes ``cov_params_default`` / ``cov_params_wo_det``).
  Every call raised AttributeError.
* ``stderr_alpha`` forwarded the raw values, so it reported statsmodels' scale rather than
  Stata's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    values = frame[["dinf", "unem"]].dropna().to_numpy()
    return ws.VECM(values, coint_rank=1, lags=2, names=["dinf", "unem"]).fit()


def test_cov_params_works_and_is_ten_by_ten(fitted) -> None:
    """It used to raise AttributeError on every call."""

    covariance = fitted.cov_params()
    assert covariance.shape == (10, 10)


def test_cov_params_matches_stata_e_v(fitted) -> None:
    """The VALUES match Stata exactly, but the matrix ORDER differs.

    ``cov_params_default`` is 10x10 in statsmodels' order (alpha for both equations, then
    the free beta, then gamma/deterministic), while Stata's e(V) is 8x8 with the
    coefficients grouped by equation. The diagonal entries are the same numbers in a
    different arrangement, so the positions checked here are the statsmodels ones.
    """
    covariance = fitted.cov_params()
    assert float(covariance[0, 0]) == pytest.approx(0.01145563, rel=1e-6)
    # Stata prints these to six figures, so the comparison tolerance has to allow for the
    # digits it truncated; rel=1e-6 would demand more precision than the reference has.
    assert float(covariance[1, 1]) == pytest.approx(0.00039582, rel=1e-5)
    assert float(covariance[4, 4]) == pytest.approx(0.0043746, rel=1e-5)
    assert float(covariance[6, 6]) == pytest.approx(0.1138787, rel=1e-5)


def test_stderr_alpha_matches_stata(fitted) -> None:
    errors = np.asarray(fitted.stderr_alpha, dtype=float).ravel()
    assert errors[0] == pytest.approx(0.107031, rel=1e-5)
    assert errors[1] == pytest.approx(0.0198952, rel=1e-5)


def test_the_scale_factor_is_the_degrees_of_freedom_ratio(fitted) -> None:
    """sqrt(164/160) -- measured, not assumed."""

    nobs = float(fitted.nobs)
    factor = float(np.sqrt(nobs / (nobs - 4.0)))
    assert factor == pytest.approx(1.012423, abs=1e-5)
    raw = np.asarray(fitted.raw.stderr_alpha, dtype=float).ravel()
    corrected = np.asarray(fitted.stderr_alpha, dtype=float).ravel()
    assert np.allclose(corrected / raw, factor, rtol=1e-12)


def test_alpha_and_beta_themselves_are_unchanged(fitted) -> None:
    """Only the covariance scale changed; the point estimates already matched."""

    assert float(np.asarray(fitted.alpha).ravel()[0]) == pytest.approx(-1.872152, abs=5e-6)
    assert float(np.asarray(fitted.beta).ravel()[1]) == pytest.approx(0.111912, abs=5e-6)
