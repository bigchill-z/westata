"""The coefficient table for a MULTI-equation VAR.

Stata groups the rows by LAGGED VARIABLE, not by lag: every lag of the first variable, a blank
spacer, every lag of the second, and `_cons` last. With one endogenous variable the grouping is a
no-op, so `test_varbasic_coefficient_table.py` stays byte-identical; this file covers the case
where it matters.

Stata reference, `varbasic dinf unem, lags(1/2) nograph` on macro_swatson.dta (StataNow 19.5),
coefficient table verbatim:

    ------------------------------------------------------------------------------
                 | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
    -------------+----------------------------------------------------------------
    dinf         |
            dinf |
             L1. |  -.4709436   .0676118    -6.97   0.000    -.6034603    -.338427
             L2. |   -.401031   .0654944    -6.12   0.000    -.5293976   -.2726644
                 |
            unem |
             L1. |   -2.24205   .3371897    -6.65   0.000     -2.90293    -1.58117
             L2. |    2.03417   .3356694     6.06   0.000      1.37627     2.69207
                 |
           _cons |   .0123764   .0045265     2.73   0.006     .0035047    .0212482
    -------------+----------------------------------------------------------------
    unem         |
            dinf |
             L1. |   .0305223   .0122746     2.49   0.013     .0064645      .05458
             L2. |  -.0113172   .0118902    -0.95   0.341    -.0346216    .0119871
                 |
            unem |
             L1. |   1.638031   .0612152    26.76   0.000     1.518051     1.75801
             L2. |  -.6725807   .0609392   -11.04   0.000    -.7920194   -.5531421
                 |
           _cons |   .0020139   .0008218     2.45   0.014     .0004032    .0036245
    ------------------------------------------------------------------------------

The header line has NO variable name here: `dinf |` only appears when the VAR has a single
equation. Measured, one equation -> "        dlny | Coefficient ..."; two -> twelve spaces.
"""

import pathlib
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"

pytestmark = pytest.mark.skipif(
    not pathlib.Path(DATA).exists(), reason="macro_swatson.dta is not available"
)


@pytest.fixture(scope="module")
def fitted():
    frame = pd.read_stata(DATA, convert_categoricals=False)
    frame["quarter"] = pd.to_datetime(frame["quarter"])
    return ws.varbasic(frame, ["dinf", "unem"], lags=(1, 2), time="quarter")


def _table(fitted) -> list[str]:
    lines = str(fitted).split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith("-" * 78))
    return lines[start:-1] if lines[-1] == "" else lines[start:]


def test_the_header_has_no_variable_name_for_two_equations(fitted) -> None:
    """With several equations no single variable owns the table, and Stata leaves the field blank."""

    header = _table(fitted)[1]
    assert header.startswith(" " * 13 + "| Coefficient"), repr(header)
    assert "dinf" not in header


def test_rows_are_grouped_by_variable_not_by_lag(fitted) -> None:
    """`L1.`/`L2.` of dinf together, then of unem -- not L1./L1./L2./L2."""

    rows = [line for line in _table(fitted) if " |  " in line or " | -" in line or " |  ." in line]
    labels = [row.split("|")[0].strip() for row in rows]
    # Coefficients (with figures) only; sub-headers and spacers have nothing after the pipe.
    with_values = [label for label, row in zip(labels, rows) if row.split("|")[1].strip()]
    assert with_values[:5] == ["L1.", "L2.", "L1.", "L2.", "_cons"]


def test_each_variable_gets_a_sub_header(fitted) -> None:
    text = "\n".join(_table(fitted))
    assert "        dinf |" in text
    assert "        unem |" in text


def test_a_blank_spacer_separates_the_variable_groups(fitted) -> None:
    lines = _table(fitted)
    spacer = " " * 13 + "|"
    assert lines.count(spacer) == 4  # two per equation: between groups, and before _cons


def test_the_coefficients_land_on_the_right_rows(fitted) -> None:
    """The grouped walk must pair each label with its own estimate, not the neighbour's."""

    text = "\n".join(_table(fitted))
    assert "         L1. |  -.4709436   .0676118    -6.97" in text  # dinf: L1.dinf
    assert "         L2. |   -.401031   .0654944    -6.12" in text  # dinf: L2.dinf
    assert "         L1. |   -2.24205   .3371897    -6.65" in text  # dinf: L1.unem
    assert "         L2. |    2.03417   .3356694     6.06" in text  # dinf: L2.unem
    assert "         L1. |   1.638031   .0612152    26.76" in text  # unem: L1.unem


def test_the_confidence_bounds_match_the_capture(fitted) -> None:
    text = "\n".join(_table(fitted))
    for bound in (
        "-.6034603", "-.338427", ".0064645", ".05458",
        "-.7920194", "-.5531421", ".0004032",
    ):
        assert bound in text


def test_seven_significant_digits_capped_at_seven_decimals(fitted) -> None:
    """A flat seven decimals printed -2.2420501; a flat seven sig printed .01237643."""

    text = "\n".join(_table(fitted))
    assert "-2.24205 " in text
    assert ".0305223 " in text
    assert "-2.2420501" not in text
    assert ".03052226" not in text
