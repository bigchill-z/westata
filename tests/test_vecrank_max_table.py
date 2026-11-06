"""vecrank max: Stata's second table, both rendered and verified.

``vecrank ..., max`` prints the trace table and then a SECOND table of maximum-eigenvalue
statistics. The package computed ``maximum`` and ``maximum_critical`` from the start -- the fields
were populated and correct -- but never rendered them, so the option's whole point was invisible.

The maximum-eigenvalue statistic is NOT the trace with a different label: it is
``-N * ln(1 - lambda_i)`` for the single eigenvalue on that row, where the trace sums the
logarithms of every remaining eigenvalue. Verified on macro_swatson.dta, k=2, lags(2):

    rank  Params        LL        Eigenvalue   Maximum   5%
       0      6     1132.9103        .        175.3107  14.07
       1      9     1220.5657   0.65664        7.7811   3.76
       2      10    1224.4562   0.04634

Note the trace table's rank-0 statistic is 183.0918 and this one is 175.3107 -- summing the other
eigenvalue's contribution is exactly the difference, which is a good check that neither was
computed by reusing the other.

The header needed its own column measurement: "Maximum" ends at 52 and "5%" at 62, which a plain
``:>8`` / ``:>7`` pair does not give (it produced 60 characters).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"

STATA_MAX_TABLE = [
    " " * 54 + "Critical",
    "Maximum" + " " * 24 + "------Eigenvalue-----" + " " * 5 + "value",
    "   rank  Params           LL                 Maximum        5%",
    "      0      6     1132.9103           .    175.3107     14.07",
    "      1      9     1220.5657     0.65664      7.7811      3.76",
    "      2      10    1224.4562     0.04634",
]


@pytest.fixture(scope="module")
def values():
    try:
        frame = pd.read_stata(DATA)[["dinf", "unem"]].dropna()
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    return frame[["dinf", "unem"]].to_numpy(float)


def test_the_max_table_renders(values) -> None:
    rendered = str(ws.vecrank(values, lags=2, show_max=True))
    for line in STATA_MAX_TABLE:
        assert line in rendered, line


def test_without_the_flag_only_the_trace_table_prints(values) -> None:
    """``max`` is a display switch, not a different computation."""

    plain = str(ws.vecrank(values, lags=2))
    assert "------Eigenvalue-----" not in plain
    assert "        .    183.0918" in plain


def test_the_maximum_statistic_differs_from_the_trace(values) -> None:
    """Rank 0 is 175.3107 by the eigenvalue test and 183.0918 by the trace.

    If both were computed from the same expression they would agree here, so this is the guard
    that the two statistics are genuinely different quantities.
    """

    result = ws.vecrank(values, lags=2)
    assert result.rows[0].maximum == pytest.approx(175.3107, abs=1e-3)
    assert result.rows[0].trace == pytest.approx(183.0918, abs=1e-3)
    assert abs(result.rows[0].maximum - result.rows[0].trace) > 7.0


def test_the_maximum_statistic_is_minus_n_log_one_minus_lambda(values) -> None:
    """``maximum[i] = -N*ln(1 - eigenvalue[i+1])``: the eigenvalue belongs to the NEXT row.

    Stata prints an eigenvalue on the row BELOW the statistic it produced. That convention is
    documented in the package's own ``vecrank`` docstring and is already true of the trace table.
    My first version of this guard assumed ``maximum[i]`` paired with ``eigenvalue[i]``, which
    failed by 168 units -- a wrong test, not a wrong statistic. Keeping the off-by-one explicit
    here stops a future "fix" from quietly reshuffling the rows.
    """

    result = ws.vecrank(values, lags=2)
    nobs = result.nobs
    rows = result.rows
    for index in range(len(rows) - 1):
        eigenvalue = rows[index + 1].eigenvalue
        if not np.isfinite(rows[index].maximum) or not np.isfinite(eigenvalue):
            continue
        assert float(rows[index].maximum) == pytest.approx(
            -nobs * float(np.log(1.0 - eigenvalue)), abs=1e-6
        )


def test_the_maximum_critical_values_are_the_eigenvalue_table(values) -> None:
    """14.07 / 3.76 -- the maximum-eigenvalue critical values, NOT the trace's 15.41 / 3.76."""

    result = ws.vecrank(values, lags=2)
    critical = [
        round(float(row.maximum_critical), 2)
        for row in result.rows
        if np.isfinite(row.maximum_critical)
    ]
    assert critical == [14.07, 3.76]
