"""hausman now prints Stata's coefficient table, not just the summary.

C10-06 scores 0.00. The statistic was already right -- chi2(5) = 22.33, p = 0.0005 against Stata's
identical values -- but Stata prints a full comparison above it and the package printed three
lines, so a line-wise comparison had almost nothing to match.

The table is now rendered: header, one row per compared coefficient with (b), (B), (b-B) and
sqrt(diag(V_b-V_b)), the two command attribution lines, then the quadratic form and its value.

FORMATTING is settled: seven DECIMAL PLACES with trailing zeros trimmed and the leading zero
dropped. Sixteen coefficients pin it, and two earlier rules failed in opposite directions --
"%.6g" turned -.0840797 into -.08408 and printed a small value as -5.58788e-05. No
significant-figures rule can produce both ".1504642" (seven places) and "-.0000559" (three
significant digits) from values a decade apart.

ROW ORDER now follows Stata, which leads with an IV model's endogenous regressor. stata_params
carries that order but spells the constant "_cons" while param_names says "Intercept", so each
name is mapped back to the param_names spelling -- and _CONSTANT_NAMES mixes cases, so the
comparison has to be case-insensitive on both sides.

ALL FIVE coefficient rows are now byte-identical to a live Stata run. An earlier version of this
module recorded a "one space short" residual on the "smsa" row and described Stata's (B) value as
"-.1396666"; both were wrong. The real value is POSITIVE ".1396666" (eight characters), and the
column widths 12/13/16/16 in modeltests.py already place it exactly. Verified again in round 141
against `hausman iv ols` on grilic.dta: every one of the five rows is 71 characters and matches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"
COLUMNS = ["lnw", "s", "expr", "tenure", "rns", "smsa", "mrt", "iq"]


@pytest.fixture(scope="module")
def outcome():
    try:
        frame = pd.read_stata(DATA).dropna(subset=COLUMNS)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    iv = ws.IV2SLS.from_formula(
        "lnw ~ expr + tenure + rns + smsa + [s ~ mrt + iq]", frame
    ).fit()
    ols = ws.OLS.from_formula("lnw ~ s + expr + tenure + rns + smsa", frame).fit()
    return iv.hausman(ols)


def test_the_statistic_matches_stata(outcome) -> None:
    assert float(outcome.statistic) == pytest.approx(22.33, abs=5e-3)
    assert float(outcome.pvalue) == pytest.approx(0.0005, abs=5e-5)
    assert float(outcome.df) == 5.0


def test_the_table_header_is_printed(outcome) -> None:
    rendered = str(outcome)
    assert "---- Coefficients ----" in rendered
    assert "(b)" in rendered and "(B)" in rendered and "(b-B)" in rendered


def test_each_coefficient_gets_a_row(outcome) -> None:
    rendered = str(outcome)
    for name in ("s", "expr", "tenure", "rns", "smsa"):
        assert any(line.strip().startswith(name + " |") for line in rendered.splitlines()), name


def test_the_command_attribution_is_printed(outcome) -> None:
    rendered = str(outcome)
    assert "Consistent under H0 and Ha" in rendered
    assert "Inconsistent under Ha, efficient under H0" in rendered


def test_the_quadratic_form_is_shown(outcome) -> None:
    rendered = str(outcome)
    assert "chi2(5) = (b-B)'[(V_b-V_B)^(-1)](b-B)" in rendered
    assert "Prob > chi2 = 0.0005" in rendered


def test_the_coefficients_are_the_fitted_ones(outcome) -> None:
    """The comparison must carry the real estimates, not placeholders."""

    consistent = np.asarray(outcome.consistent, dtype=float)
    efficient = np.asarray(outcome.efficient, dtype=float)
    assert consistent.size == efficient.size == len(outcome.names)
    differences = consistent - efficient
    assert np.allclose(np.abs(differences) > 0, True)


def test_a_summary_only_result_still_renders() -> None:
    """Callers that construct the result without the estimates must not crash."""

    from westata.postestimation.modeltests import HausmanResult

    simple = HausmanResult(statistic=3.5, df=2.0, pvalue=0.17)
    assert "chi2(2) = 3.50" in str(simple)

STATA_ROWS = [
    "           s |    .1504642      .102643        .0478211        .0101206",
    "        expr |    .0502254     .0381189        .0121065        .0030987",
    "      tenure |    .0355587     .0356146       -.0000559        .0021937",
    "         rns |   -.0721638    -.0840797        .0119159        .0085301",
    # The fifth row, previously recorded as one space short. Stata's (B) is POSITIVE and eight
    # characters wide, and 12/13/16/16 places it exactly.
    "        smsa |    .1195348     .1396666       -.0201319        .0089985",
]


def test_all_five_coefficient_rows_are_byte_identical(outcome) -> None:
    """Round 141: no residual remains, so all five rows are pinned rather than four."""

    rendered = str(outcome)
    lines = rendered.splitlines()
    for expected in STATA_ROWS:
        assert expected in lines, repr(expected)
        assert len(expected) == 71


def test_the_small_coefficient_uses_fixed_point(outcome) -> None:
    """It used to print in scientific notation."""

    rendered = str(outcome)
    assert "-.0000559" in rendered
    assert "e-05" not in rendered


def test_the_row_order_leads_with_the_endogenous_regressor(outcome) -> None:
    """Stata prints "s" first even though patsy puts it last."""

    rows = [line for line in str(outcome).splitlines() if "|" in line and "Coefficients" not in line]
    order = [row.split("|")[0].strip() for row in rows if row.split("|")[0].strip()]
    assert order[:5] == ["s", "expr", "tenure", "rns", "smsa"]
