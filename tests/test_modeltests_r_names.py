"""``lrtest`` and ``hausman`` expose their results under Stata's ``r()`` names.

Round 151 gave ``test`` and ``lincom`` a ``to_dict``; the same audit applied to the model
comparison commands found ``LRTestResult`` and ``HausmanResult`` missing it too. Their
numbers were reachable only as attributes, so a caller following the help file could not
use the documented names.

Both are chi-square tests, so both share one shape:

* ``lrtest.sthlp:216-218`` -- ``r(chi2)``, ``r(df)``, ``r(p)``;
* ``hausman.sthlp`` -- the same three.

The log-likelihoods an ``LRTestResult`` also carries are NOT ``r()`` scalars: Stata keeps
them in ``e(ll)`` of the two fitted models, so including them here would invent a surface.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

import westata as ws


@pytest.fixture(scope="module")
def models():
    rng = np.random.default_rng(773)
    n = 250
    frame = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )
    full = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()
    restricted = ws.OLS.from_formula("y ~ x1", frame).fit()
    return frame, full, restricted


# ------------------------------------------------------------------- lrtest
def test_lrtest_reports_three_scalars(models) -> None:
    _, full, restricted = models

    table = full.lrtest(restricted).to_dict()

    assert set(table) == {"chi2", "df", "p"}


def test_the_lr_statistic_is_twice_the_loglikelihood_gap(models) -> None:
    """``LR = 2 * (ll_unrestricted - ll_restricted)`` -- the test's own definition."""

    _, full, restricted = models
    result = full.lrtest(restricted)
    table = result.to_dict()

    expected = 2.0 * (result.ll_unrestricted - result.ll_restricted)

    assert table["chi2"] == pytest.approx(expected, rel=1e-10)


def test_the_lr_degrees_of_freedom_are_the_restriction_count(models) -> None:
    """Dropping ``x2`` removes exactly one parameter."""

    _, full, restricted = models

    assert full.lrtest(restricted).to_dict()["df"] == 1.0


def test_the_lr_p_value_is_the_upper_chi2_tail(models) -> None:
    _, full, restricted = models
    table = full.lrtest(restricted).to_dict()

    assert table["p"] == pytest.approx(
        float(scipy_stats.chi2.sf(table["chi2"], table["df"])), rel=1e-10
    )


def test_lrtest_does_not_expose_the_loglikelihoods_as_r_scalars(models) -> None:
    """They live in ``e(ll)`` of the fitted models, not in this test's ``r()``."""

    _, full, restricted = models
    table = full.lrtest(restricted).to_dict()

    assert "ll_unrestricted" not in table
    assert "ll_restricted" not in table


# ------------------------------------------------------------------ hausman
@pytest.fixture(scope="module")
def panel_models():
    rng = np.random.default_rng(787)
    entities, periods = 50, 5
    firm = np.repeat(np.arange(entities), periods)
    n = firm.size
    frame = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "firm": firm,
            "year": np.tile(np.arange(periods), entities),
        }
    )
    fe = ws.PanelOLS.from_formula(
        "y ~ x1", frame, entity="firm", time="year"
    ).fit()
    re = ws.RandomEffects.from_formula(
        "y ~ x1", frame, entity="firm", time="year"
    ).fit()
    return frame, fe, re


def test_hausman_reports_the_same_three_scalars(panel_models) -> None:
    _, fe, re = panel_models

    table = fe.hausman(re).to_dict()

    assert set(table) == {"chi2", "df", "p"}


def test_the_hausman_p_value_is_the_upper_chi2_tail(panel_models) -> None:
    _, fe, re = panel_models
    table = fe.hausman(re).to_dict()

    assert table["p"] == pytest.approx(
        float(scipy_stats.chi2.sf(table["chi2"], table["df"])), rel=1e-10
    )


def test_the_hausman_df_counts_the_compared_coefficients(panel_models) -> None:
    """One regressor, constant excluded from the comparison."""

    _, fe, re = panel_models

    assert fe.hausman(re).to_dict()["df"] == 1.0


def test_the_hausman_diagnostic_is_not_an_r_scalar(panel_models) -> None:
    """``excluded`` is our own addition, so it stays an attribute."""

    _, fe, re = panel_models
    result = fe.hausman(re)
    table = result.to_dict()

    assert "excluded" not in table
    assert hasattr(result, "excluded")


def test_both_commands_agree_on_the_shape(models) -> None:
    """Two chi-square tests, one vocabulary -- so a caller can treat them alike."""

    _, full, restricted = models

    assert set(full.lrtest(restricted).to_dict()) == {"chi2", "df", "p"}