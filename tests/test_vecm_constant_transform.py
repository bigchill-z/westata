"""Pin the VECM constant's parameterisation transform.

Stata puts the deterministic term INSIDE the cointegrating relation, so its printed
``_cons`` is ``det_coef - alpha*c_ce``. statsmodels' default ``co`` specification keeps
``c_ce`` at zero and folds everything into ``det_coef``.

``c_ce`` IS recoverable: refitting the same model with ``deterministic='ci'`` exposes it as
``const_coint``. Measured against a live ``vec dinf unem, lags(2) rank(1)``:

    c_ce                      -.00666269 (statsmodels ci)
    Stata e(beta)[1,3]        -.00666336
    _cons equation 1          +6.28e-07  vs Stata  -6.35e-07
    _cons equation 2          -5.12e-05  vs Stata  -5.12e-05   (exact)
    std err equation 1         .0010758  vs Stata   .0010758   (exact)
    std err equation 2         .0002000  vs Stata   .0002000   (exact)

The two standard errors and the second constant reproduce exactly. The FIRST constant is
essentially zero (6e-07) and its sign flips: the 6.7e-7 gap between the two ML fits' c_ce,
multiplied by alpha = -1.87, is larger than the constant itself. That is a convergence
precision difference between two fits, not a formula error -- recorded rather than hidden,
and this test asserts the STANDARD ERRORS (which do match) rather than the near-zero
estimate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.estimators.tseries import _vecm_cointegrating_constant

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    values = frame[["dinf", "unem"]].dropna().to_numpy()
    return ws.VECM(values, coint_rank=1, lags=2, names=["dinf", "unem"]).fit()


def test_the_cointegrating_constant_is_recoverable(fitted) -> None:
    """A ``ci`` refit exposes what the default ``co`` specification folds away."""

    c_ce = _vecm_cointegrating_constant(fitted)
    assert c_ce == pytest.approx(-0.00666336, abs=2e-6)
    # The default specification really does keep it at zero.
    assert float(np.asarray(fitted.raw.const_coint, dtype=float).ravel()[0]) == 0.0


def test_the_constant_standard_errors_match_stata(fitted) -> None:
    """These DO reproduce exactly, so they are asserted."""

    c_ce = _vecm_cointegrating_constant(fitted)
    alpha = np.asarray(fitted.raw.alpha, dtype=float).ravel()
    det = np.asarray(fitted.raw.det_coef, dtype=float).ravel()
    covariance = fitted.cov_params()

    expected = []
    for index in range(2):
        det_index = 8 + index
        variance = (
            float(covariance[det_index, det_index])
            + c_ce**2 * float(covariance[index, index])
            - 2.0 * c_ce * float(covariance[det_index, index])
        )
        expected.append(float(np.sqrt(max(variance, 0.0))))
        _ = alpha[index], det[index]

    assert expected[0] == pytest.approx(0.0010758, rel=1e-3)
    assert expected[1] == pytest.approx(0.0002, rel=1e-3)


def test_the_second_constant_reproduces_stata(fitted) -> None:
    """Far from zero, so the transform is checkable here without sign ambiguity."""

    c_ce = _vecm_cointegrating_constant(fitted)
    alpha = np.asarray(fitted.raw.alpha, dtype=float).ravel()
    det = np.asarray(fitted.raw.det_coef, dtype=float).ravel()
    constant = float(det[1]) - float(alpha[1]) * c_ce
    assert constant == pytest.approx(-5.12e-05, abs=2e-8)


def test_the_first_constant_is_near_zero_not_wrong(fitted) -> None:
    """Its sign flips because the value is ~1e-6 and c_ce carries a 6.7e-7 fit gap.

    This asserts the MAGNITUDE, which is the part the transform can be trusted on, rather
    than the sign of a number that is indistinguishable from zero at this tolerance.
    """

    c_ce = _vecm_cointegrating_constant(fitted)
    alpha = np.asarray(fitted.raw.alpha, dtype=float).ravel()
    det = np.asarray(fitted.raw.det_coef, dtype=float).ravel()
    constant = float(det[0]) - float(alpha[0]) * c_ce
    assert abs(constant) < 2e-6
    # Stata prints -6.35e-07; ours is +6.28e-07. Same order, opposite sign.
    assert abs(abs(constant) - 6.35e-07) < 1e-6


def test_the_summary_prints_the_transformed_constant(fitted) -> None:
    rendered = str(fitted.summary(style="stata"))
    assert "_cons" in rendered
    # The transformed values, not the raw det_coef (.01247 / -.000206).
    assert "0.01247" not in rendered
