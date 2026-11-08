"""Pin ``irf_table`` and the separate scaling of IRF vs OIRF.

Reference, from a live ``var dinf unem, lags(1/2)`` then ``irf create v1, step(10)``:

    . irf table oirf, impulse(dinf) response(dinf) noci
    . irf table irf,  impulse(dinf) response(dinf) noci

Two scaling rules, both measured:

* OIRF is normalised with Stata's ML residual covariance, so it needs the
  ``sqrt((T-k)/T)`` factor: statsmodels .013815 -> Stata .013603.
* IRF is the derivative with respect to a UNIT shock and must NOT be scaled: its step 0
  is exactly 1. An earlier version rescaled both arrays, turning step 0 into 0.984638 --
  a plausible-looking number that is simply wrong.

The two tables also differ in one column: irf's integer step-0 value occupies a TEN-wide
field, oirf's decimals a NINE-wide one.
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
    return ws.VAR(values, lags=2, names=["dinf", "unem"]).fit()


STATA_OIRF = [
    0.013603, -0.005984, -0.003259, 0.004231, -0.000396, -0.001672,
    0.000959, 0.000331, -0.000568, 0.000114, 0.000204,
]
STATA_IRF = [
    1.0, -0.470944, -0.247676, 0.313099, -0.036921, -0.127175,
    0.071652, 0.023311, -0.042923, 0.009135, 0.015562,
]


def test_orthogonalised_responses_match_stata(fitted) -> None:
    values = np.asarray(fitted.irf(periods=10).orth_irfs)[:, 0, 0]
    for step, expected in enumerate(STATA_OIRF):
        assert values[step] == pytest.approx(expected, abs=5e-6), step


def test_plain_responses_match_stata_and_start_at_one(fitted) -> None:
    """Plain IRF must NOT carry the ML factor."""

    values = np.asarray(fitted.irf(periods=10).irfs)[:, 0, 0]
    assert values[0] == pytest.approx(1.0, abs=1e-12)
    for step, expected in enumerate(STATA_IRF):
        assert values[step] == pytest.approx(expected, abs=5e-6), step


def test_oirf_table_matches_stata_line_for_line(fitted) -> None:
    rendered = fitted.irf_table(
        statistic="oirf", impulse="dinf", response="dinf", irfname="v1"
    ).splitlines()
    expected = [
        "Results from v1",
        "-" * 21,
        "         |      (1)  ",
        "    Step |     oirf  ",
        "-" * 9 + "+" + "-" * 11,
        "       0 |  .013603",
        "       1 | -.005984",
        "      10 |  .000204",
        "-" * 21,
        "(1) irfname = v1, impulse = dinf, and response = dinf.",
    ]
    for line in expected:
        assert line in rendered, line


def test_irf_table_matches_stata_line_for_line(fitted) -> None:
    rendered = fitted.irf_table(
        statistic="irf", impulse="dinf", response="dinf", irfname="v1"
    ).splitlines()
    expected = [
        "Results from v1",
        "-" * 21,
        "         |      (1)  ",
        "    Step |      irf  ",
        "-" * 9 + "+" + "-" * 11,
        "       0 |         1",
        "       1 | -.470944",
        "       2 | -.247676",
        "-" * 21,
        "(1) irfname = v1, impulse = dinf, and response = dinf.",
    ]
    for line in expected:
        assert line in rendered, line


def test_unknown_statistic_is_refused(fitted) -> None:
    with pytest.raises(Exception):
        fitted.irf_table(statistic="fevd")


def test_unknown_variable_name_is_refused(fitted) -> None:
    with pytest.raises(Exception):
        fitted.irf_table(impulse="nope")
