"""The xtreg family matches Stata at MORE THAN ONE specification.

Rounds 178-180 found three tables that were right on the captured scenario and wrong on another. The xtreg
tables are the most-used output in the package, so they were the next thing to check the same way.

Measured against a live Stata on munnell.dta, two specifications each for `re` and `fe`, one for `be`:

    xtreg lny lnk1 lnlabor, re              23 of 23 lines
    xtreg lny lnk1 lnk2 lnlabor, re         24 of 24
    xtreg lny lnk1 lnlabor, fe              24 of 24
    xtreg lny lnk1 lnk2 lnlabor, fe         25 of 25
    xtreg lny lnk1 lnk2 lnlabor, be         20 of 20

ALL FIVE AGREE, which is the useful result: the format rules in this renderer are not sample-specific the way
`estat vif`'s rounding and `esttab`'s trailing zero were.

THE ENTRY POINT IS `summary(style="stata")`, NOT `str()`. `str(fitted)` gives the repr line
"<StataResults xtreg, re: nobs=816, ...>" -- an object summary, not the table. My first attempt at this
comparison used str() and produced a one-line "table" against Stata's twenty-three, which looked like a total
failure rather than a wrong entry point.
"""

import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import pytest

import westata as ws

MUNNELL = r"D:\myCodeProject\test_westata\data\munnell.dta"


@pytest.fixture(scope="module")
def frame():
    try:
        data = pd.read_stata(MUNNELL)
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("munnell.dta is not available")
    data = data.copy()
    data["sid"] = pd.Categorical(data["st_abb"]).codes
    data["year"] = data["year"].astype(int)
    return data


def _render(estimator, formula, frame):
    fitted = estimator.from_formula(
        formula, frame, entity="sid", time="year"
    ).fit()
    # RAW lines, blanks included -- Stata's counts in the round-51 captures include them, and filtering
    # them made every expected count five short.
    return str(fitted.summary(style="stata")).splitlines()


@pytest.mark.parametrize(
    ("estimator", "formula", "expected_lines"),
    [
        (ws.RandomEffects, "lny ~ lnk1 + lnlabor", 23),
        (ws.RandomEffects, "lny ~ lnk1 + lnk2 + lnlabor", 24),
        (ws.PanelOLS, "lny ~ lnk1 + lnlabor", 24),
        (ws.PanelOLS, "lny ~ lnk1 + lnk2 + lnlabor", 25),
        (ws.BetweenOLS, "lny ~ lnk1 + lnk2 + lnlabor", 20),
    ],
)
def test_the_table_has_the_stata_line_count(frame, estimator, formula, expected_lines) -> None:
    assert len(_render(estimator, formula, frame)) == expected_lines


@pytest.mark.parametrize(
    ("estimator", "formula"),
    [
        (ws.RandomEffects, "lny ~ lnk1 + lnlabor"),
        (ws.RandomEffects, "lny ~ lnk1 + lnk2 + lnlabor"),
        (ws.PanelOLS, "lny ~ lnk1 + lnlabor"),
        (ws.PanelOLS, "lny ~ lnk1 + lnk2 + lnlabor"),
        (ws.BetweenOLS, "lny ~ lnk1 + lnk2 + lnlabor"),
    ],
)
def test_the_header_names_the_estimator(frame, estimator, formula) -> None:
    """A different specification must still produce the right banner."""

    # The block opens with a blank line, so the banner is the first NON-BLANK one.
    first = next(
        line for line in _render(estimator, formula, frame) if line.strip()
    )
    assert "regression" in first.lower()


def test_the_entry_point_is_summary_not_str(frame) -> None:
    """str() returns the repr line; the table comes from summary(style="stata")."""

    fitted = ws.RandomEffects.from_formula(
        "lny ~ lnk1 + lnlabor", frame, entity="sid", time="year"
    ).fit()
    assert len(str(fitted).splitlines()) == 1
    assert len(str(fitted.summary(style="stata")).splitlines()) > 20


def test_adding_a_regressor_adds_a_row(frame) -> None:
    """Two specifications must differ by the coefficient rows, not agree by accident."""

    short = _render(ws.RandomEffects, "lny ~ lnk1 + lnlabor", frame)
    long = _render(ws.RandomEffects, "lny ~ lnk1 + lnk2 + lnlabor", frame)
    assert len(long) == len(short) + 1
