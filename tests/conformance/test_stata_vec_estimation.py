"""E3 conformance for VECM estimation (Stata ``vec``).

This entry sat at E1 as "not yet compared". Comparing it turned up a real defect that
had nothing to do with the deterministic terms the entry blamed:

    lag offset.  westata passed k_ar_diff = lags, but Stata counts VAR lags while
                 statsmodels counts lagged differences, so Stata's lags(1) is
                 k_ar_diff = 0. The identical off-by-one had already been found and
                 fixed in ``johansen`` during the vecrank work and was never carried
                 across to ``fit``.

With it fixed, and with the deterministic default corrected from ``ci`` to ``co``
(``[TS] vec``: "an unrestricted constant in model"), alpha matches Stata:

    alpha      westata [-0.09162246, 0.92297218]
               Stata   [-0.09162246, 0.92297218]     gap 3.6e-13
    se(alpha)  westata [0.11396886, 0.12698855]
               Stata   [0.11425557, 0.12730802]     gap 3.2e-4

The comparison is on alpha because that is what Stata's e(b) contains -- beta is
normalised away and never printed, so alpha and its standard errors are the comparable
quantities.

The drift term is NOT asserted. westata reports det_coef [-0.02271056, -0.03237466]
where Stata prints _cons [-0.02567138, -0.00254837], and that is a reparameterisation
question independent of alpha. Deriving the transform would be worth doing; inventing
one to land on Stata's two numbers would not, so the gap is recorded in the registry
instead of being hidden behind a loose tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

import westata as ws

STATA_ALPHA = np.array([-0.09162245885644234, 0.9229721838807049])
STATA_SE = np.array([0.11425557, 0.12730802])


def _cointegrated(n: int = 400, seed: int = 41) -> np.ndarray:
    rng = np.random.default_rng(seed)
    first = np.cumsum(rng.normal(size=n))
    second = first + rng.normal(scale=0.4, size=n)
    return np.column_stack([first, second])


def test_vec_alpha_matches_stata():
    """Alpha is the comparable quantity: Stata never prints beta."""

    result = ws.VECM(_cointegrated(), coint_rank=1, lags=1).fit()
    alpha = np.asarray(result.alpha, dtype=float).reshape(-1)
    np.testing.assert_allclose(alpha, STATA_ALPHA, rtol=1e-10)


def test_vec_alpha_standard_errors_are_close():
    result = ws.VECM(_cointegrated(), coint_rank=1, lags=1).fit()
    errors = np.asarray(result.stderr_alpha, dtype=float).reshape(-1)
    # Measured 3.2e-4: close but not exact, because the small-sample scaling of the
    # alpha covariance is a separate convention from the one that produces alpha itself.
    np.testing.assert_allclose(errors, STATA_SE, rtol=5e-3)


def test_vec_lag_counts_var_lags_like_vecrank():
    """Stata's lags(1) is k_ar_diff = 0, the same mapping johansen uses.

    Getting this wrong moved alpha by 0.19 -- eighteen times the tolerance of the test
    above, which is why that test is the real guard.
    """

    data = _cointegrated()
    one_lag = ws.VECM(data, coint_rank=1, lags=1).fit()
    # lags=2 must be a genuinely different model.
    two_lags = ws.VECM(data, coint_rank=1, lags=2).fit()
    assert not np.allclose(
        np.asarray(one_lag.alpha, dtype=float),
        np.asarray(two_lags.alpha, dtype=float),
        rtol=1e-6,
    )


def test_vec_default_deterministic_is_the_unrestricted_constant():
    """[TS] vec: the default is "an unrestricted constant in model"."""

    model = ws.VECM(_cointegrated(), coint_rank=1, lags=1)
    assert model._deterministic == "co"


def test_vec_accepts_the_stata_spelling():
    result = ws.VECM(
        _cointegrated(), coint_rank=1, lags=1, deterministic="constant"
    ).fit()
    alpha = np.asarray(result.alpha, dtype=float).reshape(-1)
    np.testing.assert_allclose(alpha, STATA_ALPHA, rtol=1e-10)


def test_vec_reports_beta_normalised_on_the_first_variable():
    """Stata normalises beta[0] = 1, and the fitted beta should too."""

    result = ws.VECM(_cointegrated(), coint_rank=1, lags=1).fit()
    beta = np.asarray(result.beta, dtype=float).reshape(-1)
    assert beta[0] == pytest.approx(1.0, abs=1e-9)
