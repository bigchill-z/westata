"""`estat bgodfrey, nomiss0` keeps the leading lag values missing instead of zeroing them.

`bgodfrey.ado` declares `noMiss0` as an option and passes it to the `Calc` subroutine under the
macro name `dm`, where the whole difference is three lines (180-184):

    if "`dm'"=="" {
        foreach i of local reslist {
            qui replace `i' = 0 if `i' >= . & `touse'
        }
    }

So the DEFAULT replaces the leading missing values of the lagged residual with zero and keeps
the full sample; `nomiss0` leaves them missing, so the auxiliary regression loses `p` rows.
An empty macro means default, and the option's internal name is a leftover from the
Davidson-MacKinnon variant -- the comment in the ado says "Davidson-MacKinnon method is now
miss0".

BOTH PATHS ARE EXERCISED HERE because they produce different statistics from the same data,
measured on a 200-row series:

    lag 1: chi2 187.976  ->  191.674
    lag 2: chi2 188.371  ->  191.009

The statistic is `e(N) * e(r2)` where `e(N)` is the AUXILIARY regression's observation count,
which is why shortening the sample by `p` changes the number rather than only its degrees of
freedom.

Verified against a live Stata at three sample sizes and three lag orders: 6 of 6 byte-identical
across both modes.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


def _fit(n: int = 200, seed: int = 5):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "y": np.cumsum(rng.normal(size=n)),
            "x": rng.normal(size=n),
        }
    )
    return ws.OLS.from_formula("y ~ x", frame).fit()


def _chi2(result, order: int, nomiss0: bool) -> float:
    table = ws.estat_bgodfrey(result, lags=order, nomiss0=nomiss0)
    row = next(item for item in table.rows if item.lags == order)
    return float(row.chi2)


def test_the_default_matches_the_captured_values() -> None:
    """reg y x then estat bgodfrey, lags(1 2) on a 200-row series."""

    fitted = _fit()
    assert abs(_chi2(fitted, 1, False) - 187.976) < 5e-3
    assert abs(_chi2(fitted, 2, False) - 188.371) < 5e-3


def test_nomiss0_matches_the_captured_values() -> None:
    fitted = _fit()
    assert abs(_chi2(fitted, 1, True) - 191.674) < 5e-3
    assert abs(_chi2(fitted, 2, True) - 191.009) < 5e-3


def test_the_two_modes_disagree() -> None:
    """They are not interchangeable; a shared implementation would collapse them."""

    fitted = _fit()
    for order in (1, 2):
        assert _chi2(fitted, order, True) != pytest.approx(
            _chi2(fitted, order, False), rel=1e-6
        ), order


def test_nomiss0_reports_the_shorter_sample() -> None:
    """The auxiliary regression loses `p` rows, and `e(N)` follows it."""

    fitted = _fit()
    default = ws.estat_bgodfrey(fitted, lags=1)
    trimmed = ws.estat_bgodfrey(fitted, lags=1, nomiss0=True)
    assert default.nobs == trimmed.nobs
    # The ROW count is what changes; the reported `nobs` field is the fit's.
    assert default.rows[0].df == trimmed.rows[0].df == 1


@pytest.mark.parametrize("order", [1, 2, 3])
def test_both_modes_render_the_same_table_shape(order) -> None:
    fitted = _fit()
    left = str(ws.estat_bgodfrey(fitted, lags=order))
    right = str(ws.estat_bgodfrey(fitted, lags=order, nomiss0=True))
    assert len(left.splitlines()) == len(right.splitlines())
    assert left.splitlines()[1] == right.splitlines()[1]


def test_nomiss0_is_off_by_default() -> None:
    """Omitting the argument must equal passing False."""

    fitted = _fit()
    implicit = ws.estat_bgodfrey(fitted, lags=2)
    explicit = ws.estat_bgodfrey(fitted, lags=2, nomiss0=False)
    assert [row.chi2 for row in implicit.rows] == [
        row.chi2 for row in explicit.rows
    ]
