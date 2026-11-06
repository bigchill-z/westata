"""vecrank trend(trend) generalises to three variables -- and macro_swatson cannot test it.

Round 138 verified trend(trend) on a two-variable system. Round 139 checked whether the layout
rule and the trend column generalise, and the answer depends entirely on the data used.

MACRO_SWATSON CANNOT TEST A THREE-VARIABLE VECM. Its `dinf` column is the first difference of
`inf`, so a system of (dinf, unem, inf) is degenerate: the cointegrating matrix is singular and both
Stata and the package produce meaningless numbers (Stata reports eigenvalues .17522/.02180 for
`trend(constant)` while a degenerate fit gives .57768/.02108, and for `trend(trend)` the likelihood
comes out NaN). That is a property of the data, not a defect -- round 83 found the same thing when
`dinf`, `unem` and `inf` were used together for the FEVD work.

WITH INDEPENDENT SERIES the trend path is exact. Three cumulative series sharing two trends, n = 200,
`vecrank a b c, lags(2) trend(trend)`:

    rank  Params        LL        Eigenvalue   statistic   5%
       0      15   -627.12613        .          28.7312*   34.55
       1      20   -617.21226   0.09529         8.9035    18.17
       2      23   -613.17628   0.03995         0.8315     3.74
       3      24   -612.76051   0.00419

and the package reproduces every field: params 15/20/23/24, LL -627.1261, eigenvalues
.09529/.03995/.00419, traces 28.7312/8.9035/0.8315, critical values 34.55/18.17/3.74. So both the
deterministic column AND the layout rule scale past two variables.

TWO THINGS THIS ROUND DID NOT DO, recorded so they are not mistaken for verified:
  * the `*` selected-rank marker and the "selected rank" footer are NOT implemented. Stata prints
    "28.7312*" and a trailing "* selected rank" line; the package prints neither. That is a real,
    small gap on every table, not just the trend one.
  * a negative-zero trace is not reproduced: Stata shows "-0.0000" on a rank-2 system where a
    numerically-zero statistic is signed. Cosmetic, but visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

#: Three cumulative series sharing two common trends -- an independent reference, because
#: macro_swatson's dinf/inf pair is degenerate.
STATA_TREND_3VAR = {
    "params": [15, 20, 23, 24],
    "ll": [-627.12613, -617.21226, -613.17628, -612.76051],
    "eig": [0.09529, 0.03995, 0.00419],
    "trace": [28.7312, 8.9035, 0.8315],
    "crit": [34.55, 18.17, 3.74],
}


def _synthetic() -> np.ndarray:
    rng = np.random.default_rng(11)
    n = 200
    common = np.cumsum(rng.normal(size=n))
    return np.column_stack(
        [
            common + np.cumsum(rng.normal(scale=0.2, size=n)),
            1.5 * common + np.cumsum(rng.normal(scale=0.2, size=n)),
            np.cumsum(rng.normal(size=n)),
        ]
    )


@pytest.fixture(scope="module")
def three_var():
    return ws.vecrank(_synthetic(), lags=2, trend="trend")


def test_the_parameter_counts_scale(three_var) -> None:
    """15 / 20 / 23 / 24 -- the same +k shift, now on three variables."""

    assert [row.params for row in three_var.rows] == STATA_TREND_3VAR["params"]


def test_the_likelihoods_scale(three_var) -> None:
    assert [round(float(row.loglikelihood), 5) for row in three_var.rows] == [
        round(value, 5) for value in STATA_TREND_3VAR["ll"]
    ]


def test_the_eigenvalues_and_statistics_scale(three_var) -> None:
    finite = [row for row in three_var.rows if np.isfinite(row.eigenvalue)]
    assert [round(float(row.eigenvalue), 5) for row in finite] == STATA_TREND_3VAR["eig"]
    traces = [
        round(float(row.trace), 4) for row in three_var.rows if np.isfinite(row.trace)
    ]
    assert traces == STATA_TREND_3VAR["trace"]


def test_the_three_variable_critical_table(three_var) -> None:
    """34.55 / 18.17 / 3.74 -- the trend tabulation at k = 3."""

    crit = [
        round(float(row.trace_critical), 2)
        for row in three_var.rows
        if np.isfinite(row.trace_critical)
    ]
    assert crit == STATA_TREND_3VAR["crit"]


def test_macro_swatson_cannot_serve_as_a_three_variable_reference() -> None:
    """dinf is diff(inf), so (dinf, unem, inf) is degenerate -- Stata's own numbers are garbage.

    Pinned so a future round does not read a NaN or a .57 eigenvalue as a package defect.
    """

    frame = pd.read_stata(r"D:\myCodeProject\test_westata\data\macro_swatson.dta")
    if "dinf" not in frame.columns or "inf" not in frame.columns:
        pytest.skip("macro_swatson.dta does not carry the expected columns")
    working = frame[["dinf", "unem", "inf"]].dropna()
    difference = working["inf"].diff().dropna()
    # dinf IS the differenced inf column, to numerical precision.
    aligned = working["dinf"].to_numpy(float)[1:]
    assert np.allclose(aligned, difference.to_numpy(float), atol=1e-10)
