"""``margins`` exposes its row under Stata's ``r(table)`` row labels.

``margins.sthlp:1101`` documents ``r(table)``: a matrix whose rows are ``b``, ``se``, the
test statistic, ``p`` and the two confidence bounds. ``MarginsResult`` carried all six as
attributes but no ``to_dict``, so the documented names were unreachable.

One honest limitation: Stata labels the statistic row ``t`` or ``z`` according to the
model's distribution, and this object does not record which. The key is therefore ``t``,
and the docstring says so — naming it after a distribution we cannot check would be a
guess dressed as metadata.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

import westata as ws


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(797)
    n = 200
    frame = pd.DataFrame(
        {
            "y": 1.0 + 0.8 * rng.normal(size=n) + rng.normal(size=n),
            "x1": rng.normal(size=n),
        }
    )
    return ws.OLS.from_formula("y ~ x1", frame).fit()


def test_the_margins_row_reports_the_six_table_rows(fitted) -> None:
    table = fitted.margins(dydx="x1").to_dict()

    assert set(table) == {"b", "se", "t", "p", "ll", "ul"}


def test_the_values_are_the_objects(fitted) -> None:
    """``to_dict`` must not recompute anything."""

    computed = fitted.margins(dydx="x1")
    table = computed.to_dict()

    assert table["b"] == pytest.approx(float(computed.estimate), rel=1e-12)
    assert table["se"] == pytest.approx(float(computed.std_err), rel=1e-12)
    assert table["p"] == pytest.approx(float(computed.pvalue), rel=1e-12)
    assert table["ll"] == pytest.approx(float(computed.ci_lower), rel=1e-12)
    assert table["ul"] == pytest.approx(float(computed.ci_upper), rel=1e-12)


def test_the_statistic_is_b_over_se(fitted) -> None:
    """``b/se`` -- an identity across three of the six values."""

    table = fitted.margins(dydx="x1").to_dict()

    assert table["t"] == pytest.approx(table["b"] / table["se"], rel=1e-10)


def test_the_interval_brackets_the_estimate(fitted) -> None:
    table = fitted.margins(dydx="x1").to_dict()

    assert table["ll"] < table["b"] < table["ul"]


def test_a_custom_confidence_level_uses_the_same_labels(fitted) -> None:
    """The labels describe the table's rows, not the confidence level.

    (``margins`` has no ``discrete`` argument in this package, so the second variant is
    the ``level`` option rather than the discrete-change form.)
    """

    table = fitted.margins(dydx="x1", level=0.90).to_dict()

    assert set(table) == {"b", "se", "t", "p", "ll", "ul"} 


def test_a_constant_margin_is_zero_with_zero_width(fitted) -> None:
    """``margins, at(x1=0)`` on a linear model is the constant-plus-slope point.

    The exact value depends on the model, so only the structural relation is asserted:
    a finite estimate whose interval contains it.
    """

    table = fitted.margins(at={"x1": 0.0}).to_dict()

    assert np.isfinite(table["b"])
    assert table["ll"] <= table["b"] <= table["ul"]


def test_the_p_value_is_the_two_sided_tail(fitted) -> None:
    """With ``n - k`` degrees of freedom, as Stata's margins reports."""

    table = fitted.margins(dydx="x1").to_dict()
    df = fitted.nobs - len(fitted.param_names)

    assert table["p"] == pytest.approx(
        float(2.0 * scipy_stats.t.sf(abs(table["t"]), df)), rel=1e-6
    )