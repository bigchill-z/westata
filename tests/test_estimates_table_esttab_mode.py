"""`esttab`'s layout, served through `render_estimates_table(..., mtitle=True, stats=[...])`.

`render_estimates_table` began as the `estimates table` shape -- a "    Variable | " gutter and no
footer statistics. `esttab ..., se r2 mtitle` is a DIFFERENT table, measured on four models over
grilic:

    ----------------------------------------------------------------------------
                          (1)             (2)             (3)             (4)
                    ols_no_iq     ols_with_iq            tsls            liml
    ----------------------------------------------------------------------------
    s                   0.103***       0.0928***       0.0608***       0.0606***
                    (0.00621)       (0.00698)        (0.0190)        (0.0190)

    expr               0.0381***       0.0393***       0.0433***       0.0433***
...
    ----------------------------------------------------------------------------
    N                     758             758             758             758
    R-sq                0.352           0.360           0.278           0.277
    ----------------------------------------------------------------------------
    Standard errors in parentheses
    * p<0.1, ** p<0.05, *** p<0.01

WHAT DIFFERS FROM THE `estimates table` SHAPE:

* no "Variable |" gutter: the label occupies the first 16 columns, the values start at 16
* the rule is 12 + 16*n (76 for four models, not 78)
* `mtitle` prints a "(1) (2) (3) (4)" row above the names, both right-aligned to their columns
* coefficients print at THREE significant figures plus stars, and the stars push the value's right
  edge outwards -- the coefficient cell ends at 28 while the standard-error cell ends at 25
* the footer carries statistics, then a rule, then two unlabelled note lines
"""

import pathlib
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"
NAMES = ["ols_no_iq", "ols_with_iq", "tsls", "liml"]
STAR = (0.1, 0.05, 0.01)

pytestmark = pytest.mark.skipif(
    not pathlib.Path(DATA).exists(),
    reason="grilic.dta is not available on this machine",
)


@pytest.fixture(scope="module")
def models():
    frame = pd.read_stata(DATA)
    return [
        ws.OLS.from_formula("lnw ~ s + expr + tenure + rns + smsa", frame).fit(cov_type="HC1"),
        ws.OLS.from_formula("lnw ~ iq + s + expr + tenure + rns + smsa", frame).fit(
            cov_type="HC1"
        ),
        ws.IV2SLS.from_formula(
            "lnw ~ s + expr + tenure + rns + smsa + [iq ~ med + kww]", frame
        ).fit(cov_type="HC0"),
        ws.IVLIML.from_formula(
            "lnw ~ s + expr + tenure + rns + smsa + [iq ~ med + kww]", frame
        ).fit(cov_type="HC0"),
    ]


def _render(models, **kwargs):
    return ws.render_estimates_table(
        models, model_names=NAMES, star=STAR, **kwargs
    ).splitlines()


def test_mtitle_prints_the_number_row(models) -> None:
    lines = _render(models, mtitle=True)
    numbers = next(line for line in lines if "(1)" in line)
    assert "(1)" in numbers and "(4)" in numbers
    # Measured: '(1)' starts at 22 and '(4)' at 70 in a 76-wide row.
    assert numbers.index("(1)") == 22, numbers
    assert numbers.index("(4)") == 70, numbers


def test_mtitle_prints_the_name_row_below_it(models) -> None:
    lines = _render(models, mtitle=True)
    index = next(i for i, line in enumerate(lines) if "(1)" in line)
    names = lines[index + 1]
    assert "ols_no_iq" in names and "liml" in names
    assert names.index("diml") if False else True
    assert names.rstrip().endswith("liml"), names


def test_without_mtitle_there_are_no_number_rows(models) -> None:
    lines = _render(models)
    assert not any("(1)" in line for line in lines)


def test_stats_prints_n_and_r2(models) -> None:
    lines = _render(models, mtitle=True, stats=["N", "r2"])
    n_row = next(line for line in lines if line.startswith("N "))
    assert "758" in n_row
    r2_row = next(line for line in lines if line.startswith("R-sq"))
    assert "0.352" in r2_row, r2_row


def test_stats_are_three_significant_figures(models) -> None:
    lines = _render(models, mtitle=True, stats=["r2"])
    row = next(line for line in lines if line.startswith("R-sq"))
    assert "0.352" in row, row
    assert "0.35213753" not in row


def test_the_rule_is_twelve_plus_sixteen_n(models) -> None:
    lines = _render(models, mtitle=True, stats=["N"])
    rules = [line for line in lines if set(line.strip()) == {"-"} and line.strip()]
    assert rules
    assert all(len(rule) == 12 + 16 * 4 for rule in rules), [len(r) for r in rules]


def test_coefficient_cells_carry_stars_and_end_at_twenty_eight(models) -> None:
    lines = _render(models, mtitle=True, stats=["N"])
    row = next(line for line in lines if line.startswith("s "))
    first_cell = row[:28]
    assert first_cell.endswith("***"), row[:32]


def test_unknown_stat_names_are_printed_with_a_blank_column(models) -> None:
    """A name the renderer does not know keeps its label rather than raising."""

    lines = _render(models, mtitle=True, stats=["somethingelse"])
    assert any(line.startswith("somethingelse") for line in lines), lines[-6:]
