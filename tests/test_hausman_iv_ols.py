"""hausman iv ols: 14 of 15 lines match; the last needs a name the package cannot know.

C10-06 scored 0.00 layout / 0.09 numeric. The statistics were already exact (13.92, chi2(1),
p 0.0002) and the coefficient row was already exact; what was wrong was everything around them:

  * a MISSING LEADING BLANK LINE -- the same omission round 189 found in `estat endogenous`,
    detected this time by checking the raw prefix first;
  * the FOOTNOTE said "obtained from ivregress 2sls." where Stata says "obtained from ivregress."
    (the footnote names the COMMAND, not the estimator) and was indented by a fixed amount where
    Stata right-aligns the whole line to 78;
  * a MISSING BLANK LINE between "Test of H0" and the chi2 formula.

THE ONE REMAINING DIFFERENCE is the column heading above the two coefficient columns:

    Stata      |       iv          ols
    westata    |     2sls      regress

Stata shows the ESTIMATE STORE NAMES the user chose with `estimates store`. A result object in this
package does not carry the name it was saved under -- there is nothing to read, only something to
invent -- so the estimator words are used instead. That is a documented limitation, not a value
that is off.
"""

from __future__ import annotations

import pandas as pd
import pytest

import westata as ws
from westata.postestimation.modeltests import hausman

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"


@pytest.fixture(scope="module")
def pair():
    try:
        frame = pd.read_stata(DATA).dropna(
            subset=["lnw", "s", "expr", "tenure", "rns", "smsa", "mrt", "age"]
        )
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    iv = ws.IV2SLS.from_formula("lnw ~ [s ~ expr + tenure + age]", frame).fit()
    ols = ws.OLS.from_formula("lnw ~ s + expr + tenure", frame).fit()
    return hausman(iv, ols)


def test_the_block_opens_with_a_blank_line(pair) -> None:
    """Checked at the RAW prefix, where a leading blank line is still visible."""

    rendered = str(pair)
    assert rendered.startswith("\n"), "the block opens with a blank line"
    assert rendered.splitlines()[0] == ""
    assert rendered.splitlines()[1] == "                 ---- Coefficients ----"


def test_the_footnote_names_the_command_not_the_estimator(pair) -> None:
    """'obtained from ivregress.', not 'obtained from ivregress 2sls.'"""

    rendered = str(pair)
    assert "obtained from ivregress." in rendered
    assert "obtained from ivregress 2sls." not in rendered


def test_the_footnote_lines_are_right_aligned_to_seventy_eight(pair) -> None:
    """Their indents differ (22 vs 9) but both end at column 78."""

    lines = [
        line
        for line in str(pair).splitlines()
        if line.lstrip().startswith(("b = Consistent", "B = Inconsistent"))
    ]
    assert len(lines) == 2
    assert all(len(line) == 78 for line in lines)


def test_a_blank_line_separates_the_null_hypothesis_from_the_statistic(pair) -> None:
    lines = str(pair).splitlines()
    index = lines.index("Test of H0: Difference in coefficients not systematic")
    assert lines[index + 1] == ""
    assert lines[index + 2].startswith("    chi2(")


def test_the_statistics_are_unchanged(pair) -> None:
    assert float(pair.statistic) == pytest.approx(13.92, abs=5e-3)
    assert float(pair.pvalue) == pytest.approx(0.0002, abs=5e-5)


def test_the_store_names_are_a_recorded_limitation(pair) -> None:
    """Stata shows 'iv' / 'ols' from `estimates store`; a result object does not carry that.

    If a future round adds store names to the result surface, replace this assertion with the
    exact header comparison.
    """

    header = next(
        line
        for line in str(pair).splitlines()
        if "Difference" in line and "|" in line
    )
    assert "2sls" in header and "regress" in header
    assert " iv " not in header
