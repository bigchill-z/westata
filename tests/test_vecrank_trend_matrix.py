"""Round 143's coverage matrix: all five trend() options, k=2/k=3, trace and max tables.

Round 142 implemented the five Johansen deterministic specifications. Round 143 verified that the
refactor survived the combinations it was NOT written against:

    k=2, trace only      5/5 tables byte-identical   (round 142)
    k=2, trace + max     5/5 tables byte-identical   (this round: 19/19, 19/19, 18/18, 19/19, 18/18)
    k=3, trace only      5/5 tables byte-identical   (this round: 13/13 each)

The k=3 reference uses INDEPENDENT series, because macro_swatson's dinf is diff(inf) and a
three-variable system built from those columns is degenerate (see test_vecrank_trend_three_var.py).

WHAT IS PINNED HERE: the parameter counts and rank-0 trace for all five cases at k=3, and that
show_max appends a second table without changing the first. A future change to the deterministic
block cannot silently break one trend option while leaving the others green.
"""

from __future__ import annotations

import numpy as np
import pytest

import westata as ws

#: From live `vecrank a b c, lags(2) trend(<case>)` on the independent three-series system.
K3_EXPECTED = {
    "none": {"params": [9, 14, 17, 18], "trace0": 10.1291, "crit": [24.31, 12.53, 3.84]},
    "rconstant": {"params": [9, 15, 19, 21], "trace0": 19.4534, "crit": [34.91, 19.96, 9.42]},
    "constant": {"params": [12, 17, 20, 21], "trace0": 17.7528, "crit": [29.68, 15.41, 3.76]},
    "rtrend": {"params": [12, 18, 22, 24], "trace0": 32.1903, "crit": [42.44, 25.32, 12.25]},
    "trend": {"params": [15, 20, 23, 24], "trace0": 28.7312, "crit": [34.55, 18.17, 3.74]},
}


def _three_series() -> np.ndarray:
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
def values():
    return _three_series()


@pytest.mark.parametrize("trend", list(K3_EXPECTED))
def test_all_five_trends_match_stata_at_k3(values, trend: str) -> None:
    """Params, rank-0 trace and the critical-value tabulation, all measured at k=3."""

    result = ws.vecrank(values, lags=2, trend=trend)
    expected = K3_EXPECTED[trend]
    assert [row.params for row in result.rows] == expected["params"], trend
    assert float(result.rows[0].trace) == pytest.approx(expected["trace0"], abs=5e-4), trend
    crit = [
        round(float(row.trace_critical), 2)
        for row in result.rows
        if np.isfinite(row.trace_critical)
    ]
    assert crit == expected["crit"], trend


@pytest.mark.parametrize("trend", list(K3_EXPECTED))
def test_show_max_does_not_change_the_trace_table(values, trend: str) -> None:
    """`max` is a display flag; the first table must be identical either way."""

    plain = str(ws.vecrank(values, lags=2, trend=trend))
    with_max = str(ws.vecrank(values, lags=2, trend=trend, show_max=True))
    first_table = with_max.split("------Eigenvalue-----")[0]
    # The max table is appended after the trace table's closing rule.
    assert "Maximum" in with_max
    assert plain.splitlines()[0] == first_table.splitlines()[0]
    assert "------Eigenvalue-----" not in plain
