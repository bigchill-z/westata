"""The model test line: ``F`` vs ``Wald chi2``, decided by ``e(chi2type)``.

``xtpcse`` and ``xtgls`` report a **``Wald chi2`` model test** while their COEFFICIENT
tests use a t or z distribution. The captured Stata run shows both at once
(``tests/conformance/stata_fixtures/p9_outputs.txt:504-506`` and the table below it)::

    Estimated coefficients     =         2    Wald chi2(1)  =     178.20
                                              Prob > chi2   =     0.0000
    ------------------------------------------------------------------------------
                 |           Panel-corrected
               y | Coefficient  std. err.      z    P>|z|     [95% conf. interval]

The renderer keyed the model line on ``dist`` alone, so ``xtpcse`` (``dist == "t"``)
printed ``F(1, 1398)`` and ``Prob > F`` -- a statistic the command does not report. The
package had already stored ``e(chi2type) = "Wald"``; the display simply ignored it.

Stata records the answer to this question in ``e(chi2type)``, so that macro now decides.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.results.summary import _reports_a_chi2_test, render_summary

FIXTURES = Path(__file__).resolve().parent / "conformance" / "stata_fixtures"


def _panel_frame(n_units: int = 100, n_periods: int = 14, seed: int = 8000):
    rng = np.random.default_rng(seed)
    total = n_units * n_periods
    return pd.DataFrame(
        {
            "y": rng.normal(size=total),
            "x1": rng.normal(size=total),
            "firm": np.repeat(np.arange(n_units), n_periods),
            "year": np.tile(np.arange(1, n_periods + 1), n_units),
        }
    )


@pytest.fixture(scope="module")
def pcse_fit():
    return ws.XtPcse.from_formula(
        "y ~ x1", _panel_frame(), entity="firm", time="year"
    ).fit()


@pytest.fixture(scope="module")
def ols_fit():
    rng = np.random.default_rng(8002)
    frame = pd.DataFrame({"y": rng.normal(size=250), "x": rng.normal(size=250)})
    return ws.OLS.from_formula("y ~ x", frame).fit()


# ------------------------------------------------------------- xtpcse / xtgls
def test_xtpcse_reports_wald_chi2(pcse_fit) -> None:
    rendered = render_summary(pcse_fit, style="stata")

    assert "Wald chi2(" in rendered
    assert "Prob > chi2" in rendered


def test_xtpcse_does_not_report_an_f_test(pcse_fit) -> None:
    """The statistic Stata's output never contains for this command."""

    rendered = render_summary(pcse_fit, style="stata")

    assert "Prob > F" not in rendered


def test_the_macro_says_wald(pcse_fit) -> None:
    """The decision comes from ``e(chi2type)``, not from a hard-coded command list."""

    assert str(pcse_fit.e["chi2type"]).strip().lower() == "wald"
    assert _reports_a_chi2_test(pcse_fit) is True


# ------------------------------------------------------------------- regress
def test_regress_keeps_its_f_test(ols_fit) -> None:
    """Guard: the chi2 branch must not swallow the OLS family."""

    rendered = render_summary(ols_fit, style="stata")

    assert "F(" in rendered
    assert "Prob > F" in rendered
    assert "Wald chi2" not in rendered


def test_the_predicate_is_false_for_regress(ols_fit) -> None:
    assert _reports_a_chi2_test(ols_fit) is False


# --------------------------------------------------------------------- logit
def test_logit_still_reports_its_chi2_test() -> None:
    """The ML family already worked; this pins that it still does."""

    rng = np.random.default_rng(8004)
    frame = pd.DataFrame(
        {"yb": (rng.uniform(size=250) < 0.5).astype(int), "x": rng.normal(size=250)}
    )
    fitted = ws.Logit.from_formula("yb ~ x", frame).fit()

    rendered = render_summary(fitted, style="stata")

    assert "LR chi2" in rendered or "Wald chi2" in rendered


# ------------------------------------------------------ the capture, re-read
def test_the_capture_shows_wald_chi2_for_xtpcse() -> None:
    text = (FIXTURES / "p9_outputs.txt").read_text(encoding="utf-8", errors="ignore")

    assert "Wald chi2(1)" in text
    assert "Prob > chi2" in text


def test_the_same_capture_shows_z_in_the_coefficient_table() -> None:
    """Why the decision cannot come from ``dist``: the two live side by side."""

    text = (FIXTURES / "p9_outputs.txt").read_text(encoding="utf-8", errors="ignore")

    assert "P>|z|" in text
    assert "Wald chi2(1)" in text


# -------------------------------------------------------------- the predicate
@pytest.mark.parametrize(
    "chi2type,expected",
    [("Wald", True), ("wald", True), ("LR", True), ("chi2", True), ("", False)],
)
def test_the_predicate_reads_the_macro(chi2type: str, expected: bool) -> None:
    class _Stub:
        def __init__(self, value):
            self.e = {} if value == "" else {"chi2type": value}

    assert _reports_a_chi2_test(_Stub(chi2type)) is expected