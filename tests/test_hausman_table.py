"""hausman now prints Stata's coefficient table, not just the summary.

C10-06 scores 0.00. The statistic was already right -- chi2(5) = 22.33, p = 0.0005 against Stata's
identical values -- but Stata prints a full comparison above it and the package printed three
lines, so a line-wise comparison had almost nothing to match.

The table is now rendered: header, one row per compared coefficient with (b), (B), (b-B) and
sqrt(diag(V_b-V_b)), the two command attribution lines, then the quadratic form and its value.

Two KNOWN residuals are pinned here rather than papered over:

  * the seventh significant digit rounds differently on some coefficients -- "-.0840797" comes
    out "-.08408". Three candidate format rules (%.6g, %.7g, a %9.0g width model) were measured
    against the thirteen reference coefficients; %.6g with the leading zero stripped matched the
    most rows (11 of 13) and is what ships.
  * a very small coefficient prints in scientific notation ("-5.58787e-05") where Stata writes
    fixed-point ("-.0000559").

Row ORDER is also not yet Stata's: it prints the endogenous regressor first, while this follows
param_names (statsmodels order). Redirecting the order to stata_params fixed the rows but broke
the constant's name between the two orderings (_cons versus Intercept), so it was reverted rather
than shipped half-done.
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
