"""`estimates table ..., p` -- the p-value row, which has its own number format.

Measured on grilic with `estimates table ols_no_iq ols_with_iq tsls liml, b se p`:

               s |  .10264304    .09278735    .06078035    .06063623
                 |  .00620988    .00697626    .01895051    .01903397
                 |     0.0000       0.0000       0.0013       0.0014

THREE things pin it:

1. THE P ROW USES FOUR DECIMAL PLACES WITH THE LEADING ZERO -- '0.0000', '0.0013' -- while the
   coefficient and standard-error cells above drop the leading zero and trim ('  .00620988').
2. THE P ROW COMES AFTER THE STANDARD ERROR, never before it.
3. `b p` AND `p` ARE THE SAME REQUEST: `b` is always printed, so omitting `se` leaves a 20-line
   table with two lines per coefficient.

`p` cannot be combined with `star(...)`: Stata rejects `..., b se p star(...)`.
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


def _table(models, **kwargs):
    return ws.render_estimates_table(models, model_names=NAMES, **kwargs).splitlines()


def test_the_p_row_uses_four_decimals_with_the_leading_zero(models) -> None:
    lines = _table(models, se=True, pvalues=True)
    coefficient = next(line for line in lines if line.strip().startswith("s |"))
    index = lines.index(coefficient)
    p_row = lines[index + 2]
    assert "0.0000" in p_row, p_row


def test_the_p_row_sits_below_the_standard_error(models) -> None:
    lines = _table(models, se=True, pvalues=True)
    coefficient = next(line for line in lines if line.strip().startswith("s |"))
    index = lines.index(coefficient)
    error_row = lines[index + 1]
    p_row = lines[index + 2]
    # The se cell drops the leading zero; the p cell keeps it.
    assert ".00620988" in error_row, error_row
    assert "0.0000" in p_row, p_row


def test_the_table_has_two_lines_per_coefficient(models) -> None:
    """`se` false with `p` true still gives two lines: b and p."""

    lines = _table(models, se=False, pvalues=True)
    assert len(lines) == 20, len(lines)


def test_three_lines_per_coefficient_with_both(models) -> None:
    lines = _table(models, se=True, pvalues=True)
    assert len(lines) == 27, len(lines)


def test_the_legend_names_every_printed_row(models) -> None:
    assert _table(models, se=True, pvalues=True)[-1].strip() == "Legend: b/se/p"
    assert _table(models, se=False, pvalues=True)[-1].strip() == "Legend: b/p"
    assert _table(models, se=True, pvalues=False)[-1].strip() == "Legend: b/se"


def test_a_p_value_of_point_zero_zero_one_three_keeps_its_places(models) -> None:
    """A non-trivial value is where the four-place rule shows: tsls `s` has p = .0013."""

    lines = _table(models, se=True, pvalues=True)
    coefficient = next(line for line in lines if line.strip().startswith("s |"))
    p_row = lines[lines.index(coefficient) + 2]
    assert "0.0013" in p_row, p_row


def test_star_and_p_together_are_refused(models) -> None:
    """Stata rejects `..., b se p star(...)`; so does this renderer."""

    with pytest.raises(ValueError, match="cannot be combined"):
        ws.render_estimates_table(
            models, model_names=NAMES, se=True, pvalues=True, star=(0.1, 0.05, 0.01)
        )


def test_the_p_column_is_blank_when_a_model_lacks_the_regressor(models) -> None:
    """`ols_no_iq` has no iq coefficient, so its p cell is blank too."""

    lines = _table(models, se=True, pvalues=True)
    iq_row = next(
        index for index, line in enumerate(lines) if line.strip().startswith("iq |")
    )
    p_row = lines[iq_row + 2]
    assert p_row[14:27].strip() == "", repr(p_row[14:27])
