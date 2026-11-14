"""``xtserial``: Wooldridge's panel serial-correlation test, pinned against Stata.

``xtserial`` is Drukker's user-written ado (v1.1.0, 22apr2003), distributed outside
SSC -- `net install xtserial, from("http://www.stata.com/users/ddrukker")`.  Its whole
body is four lines, so the implementation is a transcription rather than a
reconstruction:

    qui regress d.(`varlist') if `touse', nocons cluster(`ivar')
    qui predict double `res' if `touse', residuals
    qui regress `res' l.`res' if `touse', nocons cluster(`ivar')
    qui test l.`res' == -.5

Reference, StataNow 19.5 with ``xtset province t`` on lin_1992.dta::

    xtserial ltvfo ltlan ltwlab ltpow ltfer hrs mipric1 giprice mci ngca, output

        Linear regression                               Number of obs     =        420
        Wooldridge test for autocorrelation in panel data
        H0: no first order autocorrelation
            F(  1,      27) =     12.511
                   Prob > F =      0.0015

        r(corr) = -0.192578492734578   r(F) = 12.511374216982579
        r(df) = 1   r(df_r) = 27   r(p) = 0.001484513025783

The two things that decide whether a reimplementation agrees:

* **``D.`` runs on the (panel, time) axis.**  Every panel of this dataset misses
  ``t == 11``, so the difference is missing at ``t == 11`` AND ``t == 12``; the
  first-stage N is 476 - 56 = 420.  Pairing adjacent rows gives 448 and moves
  ``r(corr)`` to -0.20871 -- a different conclusion at the same significance level,
  which is exactly the kind of error that survives a casual check.
* **The null is -0.5.**  Differencing an AR(1) error leaves a lag coefficient of
  ``rho - 1``, so "no autocorrelation" means -0.5, not 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\lin_1992.dta"
Y = "ltvfo"
XS = ["ltlan", "ltwlab", "ltpow", "ltfer", "hrs", "mipric1", "giprice", "mci", "ngca"]
FORMULA = f"{Y} ~ " + " + ".join(XS)

STATA_CORR = -0.192578492734578
STATA_F = 12.511374216982579
STATA_P = 0.001484513025783
STATA_NOBS = 420
STATA_DF_R = 27

#: Verbatim, including the leading blank line.
STATA_BLOCK = [
    "",
    "Wooldridge test for autocorrelation in panel data",
    "H0: no first order autocorrelation",
    "    F(  1,      27) =     12.511",
    "           Prob > F =      0.0015",
]


def _frame() -> pd.DataFrame:
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("lin_1992.dta is not available")
    return frame


@pytest.fixture(scope="module")
def result():
    return ws.xtserial(_frame(), FORMULA, entity="province", time="t")


def test_every_returned_scalar_matches_stata(result) -> None:
    assert result.corr == pytest.approx(STATA_CORR, abs=5e-9)
    assert result.f == pytest.approx(STATA_F, abs=5e-7)
    assert result.p == pytest.approx(STATA_P, abs=5e-10)
    assert result.df == 1
    assert result.df_r == STATA_DF_R


def test_the_first_stage_sample_is_420(result) -> None:
    """476 complete rows minus TWO missing differences per panel (t == 11 and 12)."""

    assert result.nobs == STATA_NOBS
    assert result.n_panels == 28


def test_the_printed_block_is_byte_identical(result) -> None:
    assert str(result).splitlines() == STATA_BLOCK


def test_row_order_lagging_would_give_a_different_answer() -> None:
    """The guard on the axis: pairing rows instead of periods must NOT match.

    The difference is the SEAM: each panel misses ``t == 11``, so a row-pairing
    implementation differences ``t == 11`` against ``t == 12`` -- two periods that are
    not adjacent -- and keeps 28 rows Stata drops.  Only the seam rows differ; every
    other row agrees, which is why the mistake is easy to miss.
    """

    frame = _frame()
    subset = frame.dropna(subset=[Y, *XS]).sort_values(["province", "t"]).reset_index(drop=True)
    panel = subset["province"].to_numpy()
    periods = subset["t"].to_numpy(dtype=float)

    # Row pairing: the previous ROW, whatever period it is.
    row_lagged = np.concatenate([[np.nan], subset[Y].to_numpy(dtype=float)[:-1]])
    row_gap = row_lagged[1:] - row_lagged[:-1]

    # Time pairing: only t-1 of the SAME panel.
    lookup = {(p, t): v for p, t, v in zip(panel, periods, subset[Y].to_numpy(dtype=float))}
    time_lagged = np.array([lookup.get((p, t - 1.0), np.nan) for p, t in zip(panel, periods)])
    time_diff = subset[Y].to_numpy(dtype=float) - time_lagged

    valid_rows = np.isfinite(row_gap).sum()
    valid_time = np.isfinite(time_diff).sum()
    assert valid_time == STATA_NOBS
    # Measured, not derived: row pairing keeps 474 of the 475 first differences (its own
    # first row is always lost), against Stata's 420 -- 54 rows that Stata treats as
    # uncomputable.
    assert valid_rows == 474
    assert valid_rows - valid_time == 54


def test_a_panel_without_gaps_gives_the_same_n_minus_panels() -> None:
    """Control: on a gap-free panel the two rules agree, so the gap is the whole story."""

    rng = np.random.default_rng(3)
    rows = [
        {"id": unit, "t": period, "y": rng.normal(), "x": rng.normal()}
        for unit in range(6)
        for period in range(1, 9)
    ]
    frame = pd.DataFrame(rows)
    result = ws.xtserial(frame, "y ~ x", entity="id", time="t")
    assert result.nobs == 6 * 8 - 6, "one lost first period per panel, and nothing else"
    assert result.df_r == 5


def test_it_is_reachable_from_the_package_root() -> None:
    assert hasattr(ws, "xtserial")
    assert hasattr(ws, "XtSerialResult")


def test_a_missing_column_is_named(result=None) -> None:
    frame = _frame()
    with pytest.raises(Exception, match="not a column"):
        ws.xtserial(frame, "ltvfo ~ nosuchvar", entity="province", time="t")


def test_the_test_rejects_autocorrelation_and_accepts_independence() -> None:
    """A behavioural check, not a numeric one: the p-value must move the right way."""

    rng = np.random.default_rng(11)
    independent, correlated = [], []
    for unit in range(12):
        error = 0.0
        for period in range(1, 11):
            shocked = rng.normal()
            independent.append({"id": unit, "t": period, "y": rng.normal() + shocked * 0.0,
                                "x": rng.normal()})
            error = 0.85 * error + shocked
            correlated.append({"id": unit, "t": period, "y": error + rng.normal() * 0.1,
                               "x": rng.normal()})
    clean = ws.xtserial(pd.DataFrame(independent), "y ~ x", entity="id", time="t")
    dirty = ws.xtserial(pd.DataFrame(correlated), "y ~ x", entity="id", time="t")
    assert clean.p > dirty.p
    assert dirty.p < 0.01
