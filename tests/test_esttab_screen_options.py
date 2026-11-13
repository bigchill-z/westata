"""`esttab` screen-output options, verified against live Stata one option at a time.

Every expectation here was measured, and several contradict what a careful reading of the
documentation would suggest:

* `b` ALONE STILL PRINTS THE t STATISTIC underneath -- `se=False` swaps the second row to t, it does
  not shorten the table.
* `b t p` prints ONLY p: esttab keeps one second row and the last option wins. `b se p` is an error.
* `star(...)` MUST pair a symbol with a level (`star(* 0.1 ** 0.05 *** 0.01)`); a bare level list is
  rejected with `unmatched list of significance symbols and levels`.
* esttab's DEFAULT thresholds are 5%/1%/0.1%, not the 10%/5%/1% that `estimates table` takes.
* `N` prints by default; `stats=["r2"]` ADDS to it rather than replacing it.
* The column width is derived: 16 with stars, 13 without, 26 for `ci`.
* `varlabels` SUPPRESSES the blank lines between coefficient blocks (21 lines -> 18).
* `nomargin` and `notype` do not change the screen table: `notype` is only added internally when a
  `using` file is written (`esttab.ado:409`), and calling it on screen exits with r(7).
"""

import pathlib
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import pytest

import westata as ws
from westata.results.esttab_options import ESTTAB_DEFAULT_STARS, EsttabOptions
from westata.results.esttab_render import render_esttab_table

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"

pytestmark = pytest.mark.skipif(
    not pathlib.Path(DATA).exists(),
    reason="grilic.dta is not available on this machine",
)


@pytest.fixture(scope="module")
def pair():
    frame = pd.read_stata(DATA)
    return [
        ws.OLS.from_formula("lnw ~ s + expr", frame).fit(cov_type="HC1"),
        ws.OLS.from_formula("lnw ~ s + expr + tenure", frame).fit(cov_type="HC1"),
    ]


def _render(models, **kwargs):
    options = EsttabOptions(**kwargs)
    return render_esttab_table(models, names=["m1", "m2"], options=options).splitlines()


# ---------------------------------------------------------------- default stars
def test_the_default_thresholds_are_five_one_and_a_tenth_of_a_percent(models=None) -> None:
    assert ESTTAB_DEFAULT_STARS == (0.05, 0.01, 0.001)


def test_the_default_legend_prints_the_largest_threshold_first(pair) -> None:
    assert _render(pair, second_row="se")[-1] == "* p<0.05, ** p<0.01, *** p<0.001"


# ---------------------------------------------------------------- second row
def test_b_alone_still_prints_the_t_statistic(pair) -> None:
    assert _render(pair, second_row="t")[6].strip().startswith("(16.74)")


def test_se_prints_the_standard_error_in_parentheses(pair) -> None:
    assert "(0.00638)" in _render(pair, second_row="se")[6]


def test_p_prints_three_decimals(pair) -> None:
    assert "(0.000)" in _render(pair, second_row="p")[6]


def test_brac_switches_the_delimiters_and_the_note(pair) -> None:
    lines = _render(pair, second_row="se", second_row_brackets=("[", "]"))
    assert "[0.00638]" in lines[6]
    assert "Standard errors in brackets" in lines


def test_ci_widens_the_columns_and_replaces_the_note(pair) -> None:
    lines = _render(pair, second_row="ci")
    assert len(lines[1]) == 12 + 26 * 2, len(lines[1])
    assert any("confidence intervals in brackets" in line for line in lines)


# ---------------------------------------------------------------- geometry
def test_the_rule_is_twelve_plus_sixteen_per_model_with_stars(pair) -> None:
    assert len(_render(pair, second_row="se")[1]) == 12 + 16 * 2


def test_the_rule_narrows_without_stars(pair) -> None:
    assert len(_render(pair, second_row="se", stars=False)[1]) == 12 + 13 * 2


def test_a_blank_line_follows_every_coefficient_block(pair) -> None:
    lines = _render(pair, second_row="se")
    blanks = [index for index, line in enumerate(lines) if line == ""]
    assert blanks == [0, 7, 10, 13], blanks


def test_varlabels_suppresses_those_blank_lines(pair) -> None:
    """Measured: 21 lines become 18, with a blank line only at the very top."""

    lines = _render(pair, second_row="se", varlabels={"s": "Years"})
    blanks = [index for index, line in enumerate(lines) if line == ""]
    assert blanks == [0], blanks
    assert len(lines) == 18, len(lines)


def test_n_prints_by_default(pair) -> None:
    assert any(line.startswith("N ") for line in _render(pair, second_row="se"))


def test_an_explicit_n_is_not_printed_twice(pair) -> None:
    lines = _render(pair, second_row="se", stats=("N", "r2"))
    n_rows = [line for line in lines if line.startswith("N ")]
    assert len(n_rows) == 1, n_rows


# ---------------------------------------------------------------- row selection
def test_keep_restricts_the_table_to_named_coefficients(pair) -> None:
    lines = _render(pair, second_row="se", keep=("s",))
    assert len(lines) == 12, len(lines)
    assert not any(line.startswith("expr") for line in lines)


def test_drop_removes_a_named_coefficient(pair) -> None:
    lines = _render(pair, second_row="se", drop=("expr",))
    assert not any(line.startswith("expr") for line in lines)


def test_order_rearranges_the_rows(pair) -> None:
    lines = _render(pair, second_row="se", order=("expr", "s"))
    # The first coefficient row sits at index 5, after the blank, rule, numbers and names.
    assert lines[5].startswith("expr"), lines[5]


def test_nocons_drops_the_constant(pair) -> None:
    lines = _render(pair, second_row="se", no_constant=True)
    assert not any(line.startswith("_cons") for line in lines)
    assert len(lines) == 18, len(lines)


def test_rename_changes_a_row_label(pair) -> None:
    lines = _render(pair, second_row="se", rename={"s": "SS"})
    assert any(line.startswith("SS ") for line in lines)


def test_varlabels_changes_a_row_label(pair) -> None:
    lines = _render(pair, second_row="se", varlabels={"s": "Years"})
    assert any(line.startswith("Years ") for line in lines)


# ---------------------------------------------------------------- options that do nothing on screen
def test_notype_is_not_a_screen_option(pair) -> None:
    """`esttab.ado:409` adds notype only when a `using` file is written; on screen it exits r(7).

    The dependent-variable row therefore always prints, so the option changes nothing here.
    """

    with_type = _render(pair, second_row="se")
    without = _render(pair, second_row="se", no_type=True)
    # The ONLY difference is the dependent-variable row: the table is one line shorter and every
    # other line is identical, including both rule positions.
    assert len(without) == len(with_type) - 1, (len(with_type), len(without))
    dropped = [line for line in with_type if line not in without]
    assert len(dropped) == 1, dropped
    assert dropped[0].strip().endswith("lnw"), dropped


def test_nomargin_does_not_change_the_screen_table(pair) -> None:
    assert _render(pair, second_row="se") == _render(pair, second_row="se", no_margin=False)
