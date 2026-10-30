"""``estat firststage`` numbers, checked against an independent regression.

The command reports, per endogenous regressor, the first-stage ``R-sq.``, ``Adjusted``,
``Partial`` R-squared and the ``F``/``Prob > F`` of the excluded instruments.

Every one of those is recomputable from a plain OLS fit of the endogenous variable on the
included regressors plus the instruments:

* ``R-sq.``     -- that regression's own ``R^2``;
* ``Partial``   -- the increase in ``R^2`` attributable to the INSTRUMENTS alone,
                   ``(R2_full - R2_reduced) / (1 - R2_reduced)``;
* ``F``         -- the F test of the instruments in that regression.

Checking against the definitions matters because a first-stage summary is a table of
plausible percentages; a wrong denominator or a partial R-squared computed against the
wrong baseline still looks like a normal row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

import westata as ws


def _panel(*, strength: float, seed: int, n: int = 400):
    """Two-stage data with a tunable instrument strength.

    ``strength`` is the coefficient on the instrument in the first stage: a large value
    gives a strong instrument, a value near zero a weak one.
    """

    rng = np.random.default_rng(seed)
    instrument = rng.normal(size=n)
    endogenous = strength * instrument + rng.normal(size=n) * 0.8
    exog = rng.normal(size=n)
    outcome = 1.0 + 0.8 * exog + 0.5 * endogenous + rng.normal(size=n)
    return pd.DataFrame(
        {
            "y": outcome,
            "x": exog,
            "e": endogenous,
            "z": instrument,
        }
    )


@pytest.fixture(scope="module")
def strong():
    frame = _panel(strength=1.5, seed=883)
    return frame, ws.IV2SLS.from_formula("y ~ x + [e ~ z]", frame).fit()


@pytest.fixture(scope="module")
def weak():
    frame = _panel(strength=0.02, seed=887)
    return frame, ws.IV2SLS.from_formula("y ~ x + [e ~ z]", frame).fit()


def _first_stage_fit(frame: pd.DataFrame):
    """``e`` on the included regressor AND the instrument -- the full first stage."""

    design = sm.add_constant(
        np.column_stack(
            [frame["x"].to_numpy(dtype=float), frame["z"].to_numpy(dtype=float)]
        )
    )
    return sm.OLS(frame["e"].to_numpy(dtype=float), design).fit()


def _reduced_fit(frame: pd.DataFrame):
    """``e`` on the included regressor only -- the baseline for partial R-squared."""

    design = sm.add_constant(
        frame["x"].to_numpy(dtype=float).reshape(-1, 1)
    )
    return sm.OLS(frame["e"].to_numpy(dtype=float), design).fit()


def _first(values) -> float:
    """The reported quantities are per-endogenous-regressor; take the first row."""

    array = np.asarray(values, dtype=float).reshape(-1)
    return float(array[0])


def test_the_r_squared_is_the_first_stage_fit(strong) -> None:
    frame, result = strong

    assert _first(result.estat_firststage().r2) == pytest.approx(
        float(_first_stage_fit(frame).rsquared), rel=1e-10
    )


def test_the_partial_r_squared_is_the_instrument_increment(strong) -> None:
    """``(R2_full - R2_reduced) / (1 - R2_reduced)``.

    Computing it against the WRONG baseline -- say ``R2_full - R2_reduced`` without the
    ``1 - R2_reduced`` denominator -- gives a smaller number that still sits in [0, 1].
    """

    frame, result = strong
    full = _first_stage_fit(frame)
    reduced = _reduced_fit(frame)
    expected = (full.rsquared - reduced.rsquared) / (1.0 - reduced.rsquared)

    assert _first(result.estat_firststage().partial_r2) == pytest.approx(
        expected, rel=1e-10
    )


def test_the_f_statistic_uses_statats_denominator(strong) -> None:
    """``ivregress_estat.ado:581``: ``dfdenom = e(N) - rank - df_a - nendog``.

    NOT the auxiliary regression's own ``df_resid``.  For one endogenous regressor with
    no absorbed terms that is ``N - rank - 1``, which on a 400-row fit is 396 -- while
    the auxiliary regression reports 398.  The two differ by 2 here, and using the wrong
    one made F 0.25% too small: 1407.34 against the correct 1403.79.
    """

    frame, result = strong
    full = _first_stage_fit(frame)
    reduced = _reduced_fit(frame)
    df_num = float(full.df_model - reduced.df_model)

    rank = float(result.e["rank"])
    df_denom = float(len(frame) - rank - 1.0)
    expected = ((full.rsquared - reduced.rsquared) / df_num) / (
        (1.0 - full.rsquared) / df_denom
    )

    reported = result.estat_firststage()
    assert float(reported.df_denom) == pytest.approx(df_denom)
    assert _first(reported.f) == pytest.approx(expected, rel=1e-10)


def test_the_denominator_is_not_the_auxiliary_df_resid(strong) -> None:
    """Guard: the two conventions must actually differ on this fixture.

    If they coincided, the test above could not tell which one is used.
    """

    frame, result = strong
    auxiliary_df_resid = float(_first_stage_fit(frame).df_resid)
    reported = float(result.estat_firststage().df_denom)

    assert reported != pytest.approx(auxiliary_df_resid)


def test_a_weak_instrument_reports_a_small_partial_r_squared(weak) -> None:
    """The direction of the statistic, not just its arithmetic."""

    frame, result = weak
    reported = result.estat_firststage()

    # Measured 0.0120 with the corrected denominator; the point is that it is tiny
    # relative to the strong case (0.79), not that it sits below one particular value.
    assert _first(reported.partial_r2) < 0.05
    assert _first(reported.f) < 5.0


def test_a_strong_instrument_reports_a_large_partial_r_squared(strong) -> None:
    frame, result = strong
    reported = result.estat_firststage()

    assert _first(reported.partial_r2) > 0.5
    assert _first(reported.f) > 100.0


def test_the_two_strengths_differ(strong, weak) -> None:
    """Guard: if both panels behaved alike the direction tests prove nothing."""

    strong_value = _first(strong[1].estat_firststage().partial_r2)
    weak_value = _first(weak[1].estat_firststage().partial_r2)

    assert strong_value > 10.0 * weak_value


def test_the_partial_r_squared_never_exceeds_the_full_one(strong) -> None:
    """A structural bound: the instruments cannot explain more than the whole fit."""

    reported = strong[1].estat_firststage()

    assert _first(reported.partial_r2) <= _first(reported.r2) + 1e-12


def test_the_adjusted_r_squared_is_below_the_plain_one(strong) -> None:
    """With more than one regressor the adjustment must shrink it."""

    reported = strong[1].estat_firststage()

    assert _first(reported.adj_r2) < _first(reported.r2)


def test_two_endogenous_variables_get_two_rows() -> None:
    """The summary is per endogenous regressor, not one pooled row."""

    rng = np.random.default_rng(889)
    n = 300
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    e1 = 0.7 * z1 + rng.normal(size=n) * 0.3
    e2 = 0.7 * z2 + rng.normal(size=n) * 0.3
    exog = rng.normal(size=n)
    outcome = 1.0 + 0.8 * exog + 0.5 * e1 + 0.4 * e2 + rng.normal(size=n)
    frame = pd.DataFrame(
        {"y": outcome, "x": exog, "e1": e1, "e2": e2, "z1": z1, "z2": z2}
    )
    result = ws.IV2SLS.from_formula("y ~ x + [e1 + e2 ~ z1 + z2]", frame).fit()

    reported = result.estat_firststage()

    assert np.asarray(reported.r2, dtype=float).size == 2
    assert list(reported.endog) == ["e1", "e2"]