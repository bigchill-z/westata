"""``estat endogenous`` -- Durbin and Wu-Hausman, checked against the regressions.

Both tests ask whether the endogenous regressor can be treated as exogenous. The recipe
is the control-function form:

1. regress the suspect regressor on the instruments and included exogenous regressors;
2. take the fitted residuals ``v``;
3. add ``v`` to the structural equation and test its coefficient.

``Wu-Hausman`` is the F form of that test; ``Durbin`` is the score (chi-square) form. They
answer the same question and normally agree to a couple of decimal places, which is why a
wrong denominator can hide: both numbers still look like p-values near 1 when the
regressor really is exogenous.

So the F is rebuilt here from two regressions rather than compared against a stored
number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats as scipy_stats

import westata as ws


def _panel(*, endogeneity: float, seed: int, n: int = 300):
    """``endogeneity`` scales how much the regressor correlates with the error.

    At 0 the regressor is exogenous and both tests should fail to reject; at a large
    value they should both reject strongly.
    """

    rng = np.random.default_rng(seed)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    shock = rng.normal(size=n)
    endogenous = 0.8 * z1 + 0.4 * shock + rng.normal(size=n) * 0.3
    exog = rng.normal(size=n)
    outcome = (
        1.0
        + 0.8 * exog
        + 0.5 * endogenous
        + endogeneity * shock
        + rng.normal(size=n)
    )
    return pd.DataFrame(
        {"y": outcome, "x": exog, "e": endogenous, "z1": z1, "z2": z2}
    )


@pytest.fixture(scope="module")
def exogenous():
    frame = _panel(endogeneity=0.0, seed=920)
    return frame, ws.IV2SLS.from_formula("y ~ x + [e ~ z1 + z2]", frame).fit()


@pytest.fixture(scope="module")
def endogenous():
    frame = _panel(endogeneity=1.5, seed=921)
    return frame, ws.IV2SLS.from_formula("y ~ x + [e ~ z1 + z2]", frame).fit()


def _control_function_f(frame: pd.DataFrame):
    """The Wu-Hausman F, rebuilt from the two OLS fits."""

    n = len(frame)
    first_stage_design = sm.add_constant(
        np.column_stack(
            [
                frame["x"].to_numpy(dtype=float),
                frame["z1"].to_numpy(dtype=float),
                frame["z2"].to_numpy(dtype=float),
            ]
        )
    )
    residuals = sm.OLS(
        frame["e"].to_numpy(dtype=float), first_stage_design
    ).fit().resid

    restricted = sm.OLS(
        frame["y"].to_numpy(dtype=float),
        sm.add_constant(
            np.column_stack(
                [frame["x"].to_numpy(dtype=float), frame["e"].to_numpy(dtype=float)]
            )
        ),
    ).fit()
    unrestricted = sm.OLS(
        frame["y"].to_numpy(dtype=float),
        sm.add_constant(
            np.column_stack(
                [
                    frame["x"].to_numpy(dtype=float),
                    frame["e"].to_numpy(dtype=float),
                    residuals,
                ]
            )
        ),
    ).fit()

    df_num = 1.0
    df_denom = float(n - 4)
    return (
        (unrestricted.rsquared - restricted.rsquared) / df_num
    ) / ((1.0 - unrestricted.rsquared) / df_denom), df_num, df_denom


def test_the_wu_hausman_statistic_matches_the_control_function_f(exogenous) -> None:
    frame, result = exogenous
    expected, _, _ = _control_function_f(frame)

    assert float(result.estat_endogenous().wu) == pytest.approx(expected, rel=1e-8)


def test_the_wu_hausman_degrees_of_freedom(exogenous) -> None:
    """One suspect regressor, ``n - k`` denominator."""

    frame, result = exogenous
    _, df_num, df_denom = _control_function_f(frame)
    reported = result.estat_endogenous()

    assert float(reported.wu_df) == df_num
    assert float(reported.wu_df_denom) == df_denom


def test_the_wu_hausman_p_value_is_the_upper_f_tail(exogenous) -> None:
    reported = exogenous[1].estat_endogenous()

    assert float(reported.wu_pvalue) == pytest.approx(
        float(
            scipy_stats.f.sf(
                float(reported.wu), float(reported.wu_df), float(reported.wu_df_denom)
            )
        ),
        rel=1e-10,
    )


def test_the_durbin_p_value_is_the_upper_chi2_tail(exogenous) -> None:
    """``Durbin`` is a chi-square form on the same restriction count."""

    reported = exogenous[1].estat_endogenous()

    assert float(reported.durbin_pvalue) == pytest.approx(
        float(
            scipy_stats.chi2.sf(float(reported.durbin), float(reported.wu_df))
        ),
        rel=1e-10,
    )


def test_both_tests_fail_to_reject_an_exogenous_regressor(exogenous) -> None:
    """The direction of the test, on data built without endogeneity."""

    reported = exogenous[1].estat_endogenous()

    assert float(reported.durbin_pvalue) > 0.05
    assert float(reported.wu_pvalue) > 0.05


def test_both_tests_reject_an_endogenous_regressor(endogenous) -> None:
    """Data built WITH endogeneity must be detected, or the test proves nothing."""

    reported = endogenous[1].estat_endogenous()

    assert float(reported.durbin_pvalue) < 0.001
    assert float(reported.wu_pvalue) < 0.001


def test_the_two_panels_differ(exogenous, endogenous) -> None:
    """Guard: if both panels behaved alike the direction tests are vacuous."""

    quiet = float(exogenous[1].estat_endogenous().wu_pvalue)
    loud = float(endogenous[1].estat_endogenous().wu_pvalue)

    assert quiet > 0.05 > loud


def test_the_two_test_statistics_are_distinct(exogenous) -> None:
    """Durbin is a score statistic and Wu-Hausman an F; they are not the same number."""

    reported = exogenous[1].estat_endogenous()

    assert float(reported.durbin) != pytest.approx(float(reported.wu), rel=1e-3)


def test_to_dict_reports_the_r_names(exogenous) -> None:
    """``r(durbin)``/``r(durbinp)``/``r(wu)``/``r(wup)`` and the two df."""

    table = exogenous[1].estat_endogenous().to_dict()

    assert "durbin" in table
    assert "wu" in table
    assert np.isfinite(table["durbin"])
    assert np.isfinite(table["wu"])


def test_the_tested_variable_is_named(exogenous) -> None:
    """The report says WHICH regressor was tested."""

    reported = exogenous[1].estat_endogenous()

    assert reported.tested
    assert "e" in str(reported.tested)