"""Pin the VAR's impulse responses to Stata's ``irf`` numbers.

Reference from a live ``var dinf unem, lags(1/2)`` then ``irf create`` / ``irf table oirf``
on macro_swatson.dta, orthogonalised response to a ``dinf`` shock::

    Step |  oirf(dinf)   oirf(unem)
       0 |    .013603     -.000188
       1 |   -.005984      .000107
       2 |   -.003259     -.000035
       3 |    .004231     -.000161
      10 |    .000204     -.000073

westata used to return .013815 / -.006077 / -.003310 / .004297 / .000207 -- the same SHAPE,
1.5% high on every point. statsmodels normalises the shocks with the unbiased residual
covariance (divide by ``T - k_eq``) while Stata's ``var`` uses the ML one (divide by ``T``);
the ratio is ``sqrt((T - k_eq)/T)`` = 0.9846 here, which scales every response equally.

``VARResults.ml_factor`` already carried that ratio for ``sigma_u`` -- the IRF was the one path
that never applied it.

A second trap while pinning this: ``var x y, lags(2)`` and ``var x y, lags(1/2)`` are DIFFERENT
models in Stata (the first uses only the second lag, omitting the first). A probe written
against ``lags(2)`` reported e(Sigma)[1,1] = .0002671 where westata gives .00018505, which looks
like a 44% defect until you notice ``lags(1/2)`` returns .00018505 exactly. The data was wrong,
not the code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"

STATA = {
    0: (0.013603, -0.000188),
    1: (-0.005984, 0.000107),
    2: (-0.003259, -0.000035),
    3: (0.004231, -0.000161),
    10: (0.000204, -0.000073),
}


@pytest.fixture(scope="module")
def responses():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    values = frame[["dinf", "unem"]].dropna().to_numpy()
    result = ws.VAR(values, lags=2, names=["dinf", "unem"]).fit()
    return np.asarray(result.irf(periods=10).orth_irfs)


def test_orthogonalised_impulse_responses_match_stata(responses) -> None:
    for step, (first, second) in STATA.items():
        assert responses[step, 0, 0] == pytest.approx(first, abs=5e-7), step
        assert responses[step, 1, 0] == pytest.approx(second, abs=5e-7), step


def test_the_shape_is_eleven_steps_by_two_by_two(responses) -> None:
    assert responses.shape == (11, 2, 2)


def test_the_first_response_is_positive_and_decays(responses) -> None:
    """Sanity on the sign convention, not just the numbers."""

    assert responses[0, 0, 0] > 0.0
    assert abs(responses[10, 0, 0]) < abs(responses[0, 0, 0])
