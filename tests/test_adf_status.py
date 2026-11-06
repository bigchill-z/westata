"""C14-01 and C14-03: numeric coverage 1.00, score 0.00, numbers exact.

Two rows in the report fail on their own terms before any comparison is attempted:

    | C14-01 | 第14章 | ADF 检验 dfuller lrgnp, trend  | B | 0.00 | 1.00 |
    | C14-03 | 第14章 | 差分后 ADF dfuller d.lrgnp     | B | 0.00 | 1.00 |

A numeric-coverage column of 1.00 means "every number present", and the score column says "nothing
works". Both cannot be true. Checked against a live Stata on nelson_plosser.dta:

    dfuller lrgnp, trend    Stata Z(t) = -2.026     westata -2.0262
    dfuller d.lrgnp         Stata Z(t) = -5.322     westata -5.3216

Both match to the printed precision.

THE LIKELY CAUSE OF THE ZERO, and it is a legitimate difference rather than a fake one: the package
has no top-level `dfuller`. The command is reached through

    ws.unitroot(series, test="adf", regression="c" | "ct" | "n" | "ctt")

so a reader who writes `ws.dfuller(...)` gets AttributeError and can reasonably conclude the command
is missing -- which is exactly what I did first this round. `"ct"` is Stata's `, trend` and `"c"` is
the default; the mapping is neither obvious nor documented at the call site.

So this module records BOTH facts: the numbers are right, and the entry point is not the name a
Stata user would try. The second is a real usability gap even though the first makes the 0.00 score
wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\nelson_plosser.dta"

#: From `dfuller lrgnp, trend` and `dfuller d.lrgnp` on nelson_plosser.dta.
STATA_ADF_TREND = -2.026
STATA_ADF_DIFF = -5.322


@pytest.fixture(scope="module")
def levels():
    try:
        frame = pd.read_stata(DATA)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("nelson_plosser.dta is not available")
    return np.asarray(frame["lrgnp"], dtype=float)


def test_the_entry_point_is_unitroot_not_dfuller() -> None:
    """The usability gap: a Stata user types `dfuller` and gets AttributeError."""

    assert hasattr(ws, "unitroot")
    assert not hasattr(ws, "dfuller")


def test_adf_with_trend_matches_stata(levels) -> None:
    """C14-01. Stata's `, trend` is `regression="ct"` here."""

    series = levels[np.isfinite(levels)]
    result = ws.unitroot(series, test="adf", regression="ct")
    assert float(result.statistic) == pytest.approx(STATA_ADF_TREND, abs=5e-4)


def test_adf_on_the_differenced_series_matches_stata(levels) -> None:
    """C14-03."""

    series = levels[np.isfinite(levels)]
    differenced = np.diff(series)
    result = ws.unitroot(differenced, test="adf", regression="c")
    assert float(result.statistic) == pytest.approx(STATA_ADF_DIFF, abs=5e-4)


def test_the_default_regression_is_statas_default(levels) -> None:
    """`regression="c"` is Stata's no-option case, and it must differ from "ct"."""

    series = levels[np.isfinite(levels)]
    with_constant = ws.unitroot(series, test="adf", regression="c")
    with_trend = ws.unitroot(series, test="adf", regression="ct")
    assert float(with_constant.statistic) != pytest.approx(float(with_trend.statistic), abs=0.1)
    # A trend term should make the series look less stationary, i.e. a less negative statistic
    # here because the level series is strongly trending.
    assert float(with_trend.statistic) < float(with_constant.statistic)
