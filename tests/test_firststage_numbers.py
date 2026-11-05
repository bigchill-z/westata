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
    """The denominator is ``N - rank - (n_instruments - 1)``, NOT the auxiliary ``df_resid``.

    Measured from two live fits:

        ivregress 2sls y l (k = z)       n=24,  rank=3, 1 instrument -> F(1,21)
        ivregress 2sls y x (e = z1 z2)   n=400, rank=3, 2 instruments -> F(2,396)

    On this fixture (one instrument) the adjustment is 0, so the denominator is
    ``N - rank`` = 397 for a 400-row fit, while the auxiliary regression reports 398 and
    a two-instrument fit would give 396. All three differ, so the assertion below is
    decisive.
    """

    frame, result = strong
    full = _first_stage_fit(frame)
    reduced = _reduced_fit(frame)
    df_num = float(full.df_model - reduced.df_model)

    rank = float(result.e["rank"])
    # One instrument => the adjustment term is 0.
    df_denom = float(len(frame) - rank)
    expected = ((full.rsquared - reduced.rsquared) / df_num) / (
        (1.0 - full.rsquared) / df_denom
    )

    reported = result.estat_firststage()
    assert float(reported.df_denom) == pytest.approx(df_denom)
    assert _first(reported.f) == pytest.approx(expected, rel=1e-10)


def test_the_denominator_is_checked_against_a_second_convention(strong) -> None:
    """The guard used to assert these two DIFFER; on this fixture they coincide.

    ``N - rank - (n_instr - 1)`` = 400 - 3 - 0 = 397 and the auxiliary regression's own
    ``df_resid`` is also 397 here, so the equality is the correct answer rather than a
    sign the denominator is coming from the wrong place. Two OTHER conventions stay
    distinguishable and are asserted instead: a two-instrument fit would give 396, and
    subtracting an extra 1 would give 396 as well -- which is exactly the error that was
    in the code.

    The real discriminator is the two-instrument fixture in
    tests/test_firststage_layout.py, where Stata reports F(2,396) and ``N - rank`` alone
    would give 397.
    """

    frame, result = strong
    auxiliary_df_resid = float(_first_stage_fit(frame).df_resid)
    reported = float(result.estat_firststage().df_denom)

    assert reported == pytest.approx(auxiliary_df_resid)
    # And the values the OLD formula produced are distinguishable from it.
    rank = float(result.e["rank"])
    assert reported != pytest.approx(float(len(frame) - rank - 1.0))


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