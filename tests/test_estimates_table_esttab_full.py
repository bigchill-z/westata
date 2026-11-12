"""`render_estimates_table(..., mtitle=True, stats=[...])` reproduces the whole `esttab` table.

The user's command was `esttab ols_no_iq ols_with_iq tsls liml, se r2 mtitle
star(* 0.1 ** 0.05 *** 0.01)`. `render_esttab` caps at two models, so the esttab shape is served
from `render_estimates_table`'s `mtitle` switch instead.

The measured result is 31 lines and matches a live capture byte for byte. Three details decide it:

* the label occupies the first 12 columns, then 16 per model, so the rule is 12 + 16*n = 76;
* the NUMBER right-aligns in a 13-wide field and the star run hangs AFTER it, which is why
  `.103***` reaches column 28 while the standard-error row below stops at 25;
* the two note lines at the bottom are the ONLY lines shorter than the rule -- Stata does not pad
  them.
"""

import pathlib
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"
NAMES = ["ols_no_iq", "ols_with_iq", "tsls", "liml"]

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


def _table(models):
    return ws.render_estimates_table(
        models,
        model_names=NAMES,
        se=True,
        star=(0.1, 0.05, 0.01),
        mtitle=True,
        stats=["N", "r2"],
    ).splitlines()


def test_the_table_has_thirty_one_lines(models) -> None:
    assert len(_table(models)) == 31, len(_table(models))


def test_the_rule_is_seventy_six(models) -> None:
    lines = _table(models)
    assert len(lines[1]) == 76, len(lines[1])
    assert lines[1] == "-" * 76


def test_the_header_names_are_right_aligned_to_their_columns(models) -> None:
    lines = _table(models)
    numbers, names = lines[2], lines[3]
    assert numbers.index("(1)") == 22, numbers
    assert numbers.index("(4)") == 70, numbers
    assert "ols_no_iq" in names and names.rstrip().endswith("liml")


def test_a_starred_coefficient_extends_past_the_error_row(models) -> None:
    """`.103***` reaches 28; the `(0.00621)` beneath it stops at 25."""

    lines = _table(models)
    coefficient = next(line for line in lines if line.startswith("s "))
    error = lines[lines.index(coefficient) + 1]
    assert coefficient[:28].endswith("***"), coefficient[:32]
    assert error[:25].rstrip().endswith(")"), error[:28]
    assert len(error[:25].rstrip()) == 25


def test_two_stars_end_one_column_earlier_than_three(models) -> None:
    """`-.0745**` stops at 27 while `-.0841***` reaches 28."""

    lines = _table(models)
    row = next(line for line in lines if line.startswith("rns "))
    assert row[18:28].rstrip() == "-0.0841***", row[16:30]
    assert row[34:43].rstrip() == "-0.0745**", row[32:46]


def test_the_bottom_notes_are_not_padded(models) -> None:
    lines = _table(models)
    assert lines[-2] == "Standard errors in parentheses", repr(lines[-2])
    assert lines[-1] == "* p<0.1, ** p<0.05, *** p<0.01", repr(lines[-1])


def test_every_other_line_is_padded_to_the_rule(models) -> None:
    lines = _table(models)
    for line in lines[1:-2]:
        if line == "":
            continue
        assert len(line) == 76, (len(line), line)


def test_the_statistics_footer_carries_n_and_r_squared(models) -> None:
    lines = _table(models)
    n_row = next(line for line in lines if line.startswith("N "))
    assert "758" in n_row
    r2_row = next(line for line in lines if line.startswith("R-sq"))
    assert "0.352" in r2_row, r2_row


def test_se_false_drops_the_error_rows_and_the_note(models) -> None:
    rendered = ws.render_estimates_table(
        [ws.OLS.from_formula("lnw ~ s", pd.read_stata(DATA)).fit()],
        model_names=["one"],
        se=False,
        mtitle=True,
    )
    assert "Standard errors in parentheses" not in rendered
    assert len(rendered.splitlines()) < len(
        ws.render_estimates_table(
            [ws.OLS.from_formula("lnw ~ s", pd.read_stata(DATA)).fit()],
            model_names=["one"],
            se=True,
            mtitle=True,
        ).splitlines()
    )
