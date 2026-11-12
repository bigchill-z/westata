"""`estimates table` -- a different command from `esttab`, with its own geometry.

The user ran `estimates table ols_no_iq ols_with_iq tsls liml, b se` and westata had no equivalent:
`etable_as_stata` / `render_esttab` refuse more than two models AND lay out the `esttab` shape
(column numbers, a dependent-variable row, no Legend). `estimates table` is a separate layout.

Measured geometry on four models over grilic:

    ------------------------------------------------------------------
        Variable | ols_no_iq    ols_with~q      tsls         liml
    -------------+----------------------------------------------------
               s |  .10264308    .09278735    .06078035    .06063623
                 |  .00620988    .00697626    .01895051    .01903397
    ------------------------------------------------------------------
                                                          Legend: b/se

* the rule is `14 + 13 * n` characters (66 for four models)
* the gutter is "    Variable | ", the separator puts "+" at index 13
* each column is 13 wide: the value is right-aligned in its first 11, the last 2 are padding
* names are centred with the odd remainder on the LEFT and abbreviated past ten characters
* values use EIGHT SIGNIFICANT FIGURES at or above 1, EIGHT DECIMALS below

A NOTE ON cov_type, which cost a full comparison round: Stata's `regress, r` is HC1 while
`ivregress ..., r` is HC0. Using HC0 for the OLS columns puts every standard error off by
sqrt(n/(n-k)) -- 0.0061852491 against Stata's 0.0062098752.
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
    # regress, r is HC1; ivregress, r is HC0 -- see the module docstring.
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


def _lines(models):
    return ws.render_estimates_table(models, model_names=NAMES).splitlines()


# ---------------------------------------------------------------- geometry
def test_the_rule_length_follows_the_model_count(models) -> None:
    """14 + 13 * n, not 13 less."""

    lines = _lines(models)
    assert len(lines[1]) == 14 + 13 * 4, len(lines[1])
    assert lines[1] == "-" * 66


def test_the_separator_puts_the_plus_at_thirteen(models) -> None:
    lines = _lines(models)
    assert lines[3].index("+") == 13, lines[3]
    assert lines[3][:13] == "-" * 13


def test_the_header_centres_the_names(models) -> None:
    header = _lines(models)[2]
    assert header.startswith("    Variable |"), header
    assert "ols_no_iq" in header
    assert "ols_with~q" in header
    assert "tsls" in header and "liml" in header


def test_long_names_are_abbreviated(models) -> None:
    """`ols_with_iq` is eleven characters, so Stata prints eight + '~' + the last one."""

    assert "ols_with~q" in _lines(models)[2]
    assert "ols_with_iq" not in _lines(models)[2]


def test_every_line_is_the_same_width_except_the_leading_blank(models) -> None:
    lines = _lines(models)
    assert lines[0] == ""
    for line in lines[1:]:
        assert len(line) == 66, (len(line), line)


# ---------------------------------------------------------------- content
def test_the_variable_labels_are_in_first_model_order_with_cons_last(models) -> None:
    labels = [
        line.split("|")[0].strip()
        for line in _lines(models)[4:]
        if line.split("|")[0].strip()
        and "Legend" not in line
        and not set(line.strip()) <= {"-", "+"}
    ]
    # s, expr, tenure, rns, smsa, iq appear per model order; _cons last.
    assert labels[-1] == "_cons", labels
    assert labels[0] == "s", labels
    assert "iq" in labels


def test_a_model_without_a_regressor_leaves_the_cell_blank(models) -> None:
    """`ols_no_iq` has no iq row, so its column is spaces on the iq lines."""

    lines = _lines(models)
    iq_rows = [index for index, line in enumerate(lines) if line.split("|")[0].strip() == "iq"]
    assert iq_rows, "no iq row"
    cell = lines[iq_rows[0]][14:27]
    assert cell.strip() == "", repr(cell)


def test_the_legend_closes_the_table(models) -> None:
    lines = _lines(models)
    assert lines[-1].strip() == "Legend: b/se", lines[-1]
    assert len(lines[-1]) == 66


# ---------------------------------------------------------------- formatting
def test_values_at_or_above_one_use_eight_significant_figures(models) -> None:
    lines = _lines(models)
    row = next(line for line in lines if line.split("|")[0].strip() == "_cons")
    assert "4.103675 " in row, row
    assert "4.10367504" not in row, row


def test_values_below_one_use_eight_decimals(models) -> None:
    lines = _lines(models)
    row = next(line for line in lines if line.split("|")[0].strip() == "s")
    assert ".10264304" in row, row


def test_the_ols_columns_need_hc1_not_hc0(models) -> None:
    """The standard-error round-trip that pins the VCE convention."""

    lines = _lines(models)
    row_index = next(i for i, line in enumerate(lines) if line.split("|")[0].strip() == "s")
    error_row = lines[row_index + 1]
    assert ".00620988" in error_row, error_row


def test_se_false_omits_the_error_rows_and_legend(models) -> None:
    import pandas as pd

    frame = pd.read_stata(DATA)
    a = ws.OLS.from_formula("lnw ~ s + expr", frame).fit(cov_type="HC1")
    rendered = ws.render_estimates_table([a, a], model_names=["a", "b"], se=False)
    assert "Legend" not in rendered
    assert len(rendered.splitlines()) < len(
        ws.render_estimates_table([a, a], model_names=["a", "b"], se=True).splitlines()
    )
