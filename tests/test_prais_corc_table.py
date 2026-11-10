"""`prais ..., corc` is a different table, not only a different estimate.

The estimates had already been right since round 170 -- coefficients, rho and standard errors
matched. What did not was the TABLE around them. Comparing a live
`prais consumption temp price income, corc` against the package showed five differences:

    banner        'Prais-Winsten AR(1) regression'   vs  'Cochrane-Orcutt AR(1) regression'
    leading       two blank lines                    vs  none
    Number of obs 30                                 vs  29
    Total df      29                                 vs  28
    Total MS      .002499741                         vs  .002589018
    Adj R-squared 0.5927                             vs  0.6068

THE STATISTIC BLOCK WAS COMPUTED ON THE WRONG SAMPLE SIZE. `corc` drops the transformed first
observation, so it fits N-1 rows; the total degrees of freedom are N-2 and every statistic that
divides by them shifts. RSS, MSS and R-squared never depended on it, which is why the numbers
looked right while the summary did not.

THE COEFFICIENT ROW ORDER WAS ALSO WRONG, and not only for `corc`: the design matrix carries
the constant first, so iterating the names put `_cons` at the TOP of every prais table.
Stata's e(b) follows the WRITTEN formula and puts `_cons` last:

    prais consumption temp price income   ->  temp, price, income, _cons
    prais consumption price income temp   ->  price, income, temp, _cons

`tests/test_prais_convergence.py` had pinned the wrong order because its fixture was
transcribed from the package's own output rather than from a live capture -- the failure mode
`error.md` already records.

Verified against a live Stata for both variants: 2 of 2 tables byte-identical, 19 lines each.
"""

import warnings

warnings.filterwarnings("ignore")
from pathlib import Path

import pandas as pd
import pytest

import westata as ws

#: The capture lives OUTSIDE the repository, so it is absent on CI. Every test here skips
#: rather than errors when it is missing -- a hard-coded Windows path that was unguarded turned
#: the whole Linux matrix red the first time this file was pushed.
DATA = r"D:\myCodeProject\test_westata\data\icecream.dta"
FORMULA = "consumption ~ temp + price + income"

pytestmark = pytest.mark.skipif(
    not Path(DATA).exists(),
    reason="icecream.dta is not available on this machine",
)


def _frame() -> pd.DataFrame:
    return pd.read_stata(DATA)


def _rows(fitted) -> list[str]:
    return [line for line in str(fitted.summary()).splitlines() if line.strip()]


def test_corc_names_its_own_transform() -> None:
    """The banner says Cochrane-Orcutt, not Prais-Winsten.

    It is no longer the FIRST line: the iteration log opens the table.
    """

    rows = _rows(ws.prais(_frame(), FORMULA, corc=True))
    banner = next(row for row in rows if "AR(1) regression" in row)
    assert banner.startswith("Cochrane"), banner
    assert "Prais" not in banner


def test_the_prais_winsten_banner_is_unchanged() -> None:
    rows = _rows(ws.prais(_frame(), FORMULA))
    banner = next(row for row in rows if "AR(1) regression" in row)
    assert banner.startswith("Prais"), banner


def test_both_variants_open_with_a_blank_line_then_the_log() -> None:
    """Measured on a live capture: one blank line, the iteration log, a blank, then the banner.

    Earlier this test asserted that `corc` opened AT the banner with no blank -- that was true when
    the log was missing, because the log was the two blank lines it was reading.
    """

    for kwargs in ({}, {"corc": True}):
        lines = str(ws.prais(_frame(), FORMULA, **kwargs)).splitlines()
        assert lines[0] == "", lines[:3]
        assert lines[1].startswith("Iteration 0:"), lines[:3]


def test_corc_reports_n_minus_one_observations() -> None:
    fitted = ws.prais(_frame(), FORMULA, corc=True)
    assert fitted.nobs == 29
    assert fitted.df_total == 28


def test_the_prais_winsten_counts_are_unchanged() -> None:
    fitted = ws.prais(_frame(), FORMULA)
    assert fitted.nobs == 30
    assert fitted.df_total == 29


def test_corc_adjusted_r_squared_uses_the_short_sample() -> None:
    """0.6068 for corc against 0.5799 for the Prais-Winsten fit on the same data."""

    assert round(ws.prais(_frame(), FORMULA, corc=True).rsquared_adj, 4) == 0.6068
    assert round(ws.prais(_frame(), FORMULA).rsquared_adj, 4) == 0.5799


def test_the_total_row_shows_the_corc_degrees_of_freedom() -> None:
    rows = _rows(ws.prais(_frame(), FORMULA, corc=True))
    total = next(line for line in rows if "Total |" in line)
    assert " 28 " in total, total


def test_the_constant_is_the_last_coefficient_row() -> None:
    """For both variants, and it follows the written formula order otherwise."""

    for kwargs in ({}, {"corc": True}):
        rows = _rows(ws.prais(_frame(), FORMULA, **kwargs))
        coefficient_rows = [line for line in rows if line[12:13] == "|" and "_cons" in line or "|" in line]
        names = [
            line[:12].strip()
            for line in rows
            if line[:12].strip() in {"_cons", "temp", "price", "income"}
        ]
        assert names[-1] == "_cons", (kwargs, names)
        assert names[:3] == ["temp", "price", "income"], (kwargs, names)
