"""``estat weakrobust`` -- the Anderson-Rubin test.

``ivregress_postestimation.sthlp:364-375`` defines the command: for a JUST-IDENTIFIED
model it reports Anderson-Rubin (1949); for an overidentified one it reports the CLR test
of Moreira (2003), which this package refuses with its reason.

The AR statistic has a closed form, so it is checked here against two plain OLS fits:

    adjusted = y - endog @ beta0
    F = ((RSS_restricted - RSS_unrestricted) / q) / (RSS_unrestricted / (n - k))

where the restricted fit regresses ``adjusted`` on the included exogenous regressors and
the unrestricted one adds the excluded instruments.  Under the null that F has an exact
distribution independent of instrument strength -- which is the point of the command.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats as scipy_stats

import westata as ws


def _just_identified(*, strength: float, seed: int, n: int = 300):
    rng = np.random.default_rng(seed)
    instrument = rng.normal(size=n)
    endogenous = strength * instrument + rng.normal(size=n) * 0.4
    exog = rng.normal(size=n)
    outcome = 1.0 + 0.8 * exog + 0.5 * endogenous + rng.normal(size=n)
    return pd.DataFrame(
        {"y": outcome, "x": exog, "e": endogenous, "z": instrument}
    )


@pytest.fixture(scope="module")
def strong_model():
    frame = _just_identified(strength=0.8, seed=931)
    return frame, ws.IV2SLS.from_formula("y ~ x + [e ~ z]", frame).fit()


@pytest.fixture(scope="module")
def weak_model():
    frame = _just_identified(strength=0.02, seed=933)
    return frame, ws.IV2SLS.from_formula("y ~ x + [e ~ z]", frame).fit()


def _ar_from_regressions(frame: pd.DataFrame, beta0: float = 0.0):
    """Rebuild the AR F from a restricted and an unrestricted OLS fit."""

    n = len(frame)
    adjusted = frame["y"].to_numpy(dtype=float) - beta0 * frame["e"].to_numpy(
        dtype=float
    )
    exog = frame["x"].to_numpy(dtype=float)
    instrument = frame["z"].to_numpy(dtype=float)

    restricted_design = sm.add_constant(exog.reshape(-1, 1))
    unrestricted_design = sm.add_constant(np.column_stack([exog, instrument]))
    rss_restricted = float(
        sm.OLS(adjusted, restricted_design).fit().ssr
    )
    rss_unrestricted = float(
        sm.OLS(adjusted, unrestricted_design).fit().ssr
    )
    q = 1.0
    df_denom = float(n - unrestricted_design.shape[1])
    return ((rss_restricted - rss_unrestricted) / q) / (
        rss_unrestricted / df_denom
    ), df_denom


def test_the_ar_statistic_matches_the_restricted_f(strong_model) -> None:
    frame, result = strong_model

    reported = result.estat_weakrobust()

    expected, _ = _ar_from_regressions(frame)
    assert reported.f_statistic == pytest.approx(expected, rel=1e-10)


def test_the_chi2_form_is_q_times_the_f(strong_model) -> None:
    """Stata reports both forms; they differ by the numerator df."""

    reported = strong_model[1].estat_weakrobust()

    assert reported.statistic == pytest.approx(
        reported.df * reported.f_statistic, rel=1e-12
    )


def test_the_denominator_degrees_of_freedom(strong_model) -> None:
    frame, result = strong_model
    _, df_denom = _ar_from_regressions(frame)

    assert result.estat_weakrobust().df_denom == pytest.approx(df_denom)


def test_the_p_value_is_the_upper_f_tail(strong_model) -> None:
    reported = strong_model[1].estat_weakrobust()

    assert reported.pvalue == pytest.approx(
        float(
            scipy_stats.f.sf(
                reported.f_statistic, reported.df, reported.df_denom
            )
        ),
        rel=1e-10,
    )


def test_the_null_of_no_endogenous_effect_is_rejected_here(strong_model) -> None:
    """These data have a real effect, so the test should reject."""

    assert strong_model[1].estat_weakrobust().pvalue < 0.001


def test_the_statistic_does_not_collapse_with_a_weak_instrument(weak_model) -> None:
    """THE point of the AR test: it stays valid when the instrument is weak.

    A first-stage F would be near zero here, yet the AR statistic still tests the
    coefficient honestly -- it is large because the data were built with a real effect.
    """

    reported = weak_model[1].estat_weakrobust()

    assert np.isfinite(reported.statistic)
    assert reported.statistic > 0.0
    assert 0.0 <= reported.pvalue <= 1.0


def test_the_first_stage_is_in_fact_weak(weak_model) -> None:
    """Guard: if the fixture were actually strong, the previous test proves nothing."""

    first_stage = weak_model[1].estat_firststage()

    assert float(np.asarray(first_stage.partial_r2).ravel()[0]) < 0.05


def test_to_dict_reports_the_r_names(strong_model) -> None:
    """``r(chi2)``, ``r(F)``, ``r(df)``, ``r(df_r)``, ``r(p)``."""

    table = strong_model[1].estat_weakrobust().to_dict()

    assert set(table) == {"chi2", "F", "df", "df_r", "p"}
    for value in table.values():
        assert np.isfinite(value)


def test_the_rendered_text_names_the_test(strong_model) -> None:
    rendered = str(strong_model[1].estat_weakrobust())

    assert "Anderson-Rubin" in rendered
    assert "chi2(" in rendered
    assert "Prob > F" in rendered


# --------------------------------------------------- the unimplemented CLR branch
def test_an_overidentified_model_refuses_the_default(frame=None) -> None:
    """``sthlp:367-373``: the default there is CLR, which is not implemented.

    Refusing beats reporting AR under the CLR label, which is the failure shape this
    session keeps removing.
    """

    rng = np.random.default_rng(935)
    n = 300
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    endogenous = 0.8 * z1 + rng.normal(size=n) * 0.4
    exog = rng.normal(size=n)
    outcome = 1.0 + 0.8 * exog + 0.5 * endogenous + rng.normal(size=n)
    overidentified = pd.DataFrame(
        {"y": outcome, "x": exog, "e": endogenous, "z1": z1, "z2": z2}
    )
    result = ws.IV2SLS.from_formula("y ~ x + [e ~ z1 + z2]", overidentified).fit()

    with pytest.raises(NotImplementedError) as caught:
        result.estat_weakrobust()

    message = str(caught.value)
    assert "CLR" in message
    assert "ar=True" in message


def test_an_overidentified_model_accepts_ar() -> None:
    """``sthlp:488-492``: ``ar`` requests the AR test even when overidentified."""

    rng = np.random.default_rng(937)
    n = 300
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    endogenous = 0.8 * z1 + rng.normal(size=n) * 0.4
    exog = rng.normal(size=n)
    outcome = 1.0 + 0.8 * exog + 0.5 * endogenous + rng.normal(size=n)
    overidentified = pd.DataFrame(
        {"y": outcome, "x": exog, "e": endogenous, "z1": z1, "z2": z2}
    )
    result = ws.IV2SLS.from_formula("y ~ x + [e ~ z1 + z2]", overidentified).fit()

    reported = result.estat_weakrobust(ar=True)

    assert reported.test == "Anderson-Rubin"
    assert reported.df == 2.0  # two instruments tested


def test_clr_is_refused_with_a_reason(strong_model) -> None:
    with pytest.raises(NotImplementedError) as caught:
        strong_model[1].estat_weakrobust(clr=True)

    assert "Moreira" in str(caught.value)


def test_a_non_iv_result_is_refused() -> None:
    """``estat weakrobust`` lives on ``ivregress_estat`` and needs the IV split."""

    rng = np.random.default_rng(939)
    frame = pd.DataFrame(
        {"y": rng.normal(size=100), "x": rng.normal(size=100)}
    )

    with pytest.raises(NotImplementedError):
        ws.OLS.from_formula("y ~ x", frame).fit().estat_weakrobust()


# ------------------------------------------------------- betanull null hypothesis
def test_the_default_null_is_zero(strong_model) -> None:
    """``sthlp:458-460``: "By default, the null hypothesis is that the coefficients of
    all endogenous regressors are zero"."""

    explicit = strong_model[1].estat_weakrobust(betanull={"e": 0.0})
    implicit = strong_model[1].estat_weakrobust()

    assert implicit.f_statistic == pytest.approx(explicit.f_statistic, rel=1e-12)


def test_a_mapping_assigns_by_name(strong_model) -> None:
    """``sthlp:463-465``: ``varname=#`` names the coefficient."""

    result = strong_model[1]
    estimated = float(result.params["e"])

    at_estimate = result.estat_weakrobust(betanull={"e": estimated})

    # Testing the estimated value is the AR minimum, so the statistic collapses to 0.
    assert at_estimate.f_statistic == pytest.approx(0.0, abs=1e-8)


def test_a_sequence_assigns_by_position(strong_model) -> None:
    """``sthlp:461-463``: a numlist needs ``copy``, i.e. positional assignment."""

    result = strong_model[1]
    estimated = float(result.params["e"])

    by_position = result.estat_weakrobust(betanull=[estimated])
    by_name = result.estat_weakrobust(betanull={"e": estimated})

    assert by_position.f_statistic == pytest.approx(
        by_name.f_statistic, rel=1e-12
    )


def test_the_statistic_is_zero_at_the_estimated_value(strong_model) -> None:
    """The AR statistic is minimised at the point estimate -- a structural identity.

    An implementation that ignored ``betanull`` would return the same non-zero number
    here as for the default null.
    """

    result = strong_model[1]
    estimated = float(result.params["e"])

    at_estimate = result.estat_weakrobust(betanull={"e": estimated}).f_statistic
    at_zero = result.estat_weakrobust().f_statistic

    assert at_estimate == pytest.approx(0.0, abs=1e-8)
    assert at_zero > 1.0


def test_the_statistic_grows_with_the_distance_from_the_estimate(strong_model) -> None:
    """Direction: a null further from the estimate is rejected harder."""

    result = strong_model[1]
    estimated = float(result.params["e"])

    near = result.estat_weakrobust(betanull={"e": estimated + 0.1}).f_statistic
    far = result.estat_weakrobust(betanull={"e": estimated + 1.0}).f_statistic

    assert far > near


def test_a_wrong_length_sequence_is_refused(strong_model) -> None:
    """Padding or truncating would test a hypothesis the caller did not write."""

    with pytest.raises(ValueError, match="value"):
        strong_model[1].estat_weakrobust(betanull=[0.0, 1.0])


def test_an_unknown_name_is_refused(strong_model) -> None:
    """``sthlp:481-482``: the default is an error for non-endogenous names."""

    with pytest.raises(ValueError) as caught:
        strong_model[1].estat_weakrobust(betanull={"not_a_regressor": 1.0})

    message = str(caught.value)
    assert "not_a_regressor" in message
    # The message lists what IS available, so the mistake is actionable.
    assert "e" in message


def test_a_string_betanull_is_refused(strong_model) -> None:
    """A bare string is neither a mapping nor a sequence of numbers."""

    with pytest.raises(TypeError):
        strong_model[1].estat_weakrobust(betanull="1.0")


# -------------------------------------------- two endogenous regressors
@pytest.fixture(scope="module")
def two_endogenous():
    rng = np.random.default_rng(941)
    n = 400
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    e1 = 0.8 * z1 + rng.normal(size=n) * 0.4
    e2 = 0.8 * z2 + rng.normal(size=n) * 0.4
    exog = rng.normal(size=n)
    outcome = 1.0 + 0.8 * exog + 0.5 * e1 + 0.4 * e2 + rng.normal(size=n)
    frame = pd.DataFrame(
        {"y": outcome, "x": exog, "e1": e1, "e2": e2, "z1": z1, "z2": z2}
    )
    return frame, ws.IV2SLS.from_formula("y ~ x + [e1 + e2 ~ z1 + z2]", frame).fit()


def test_the_multi_endogenous_statistic_is_the_restricted_f(two_endogenous) -> None:
    """Two instruments tested, so ``q = 2`` and ``chi2 = 2 * F``."""

    frame, result = two_endogenous
    n = len(frame)
    exog = frame["x"].to_numpy(dtype=float)
    instruments = np.column_stack(
        [frame["z1"].to_numpy(dtype=float), frame["z2"].to_numpy(dtype=float)]
    )
    response = frame["y"].to_numpy(dtype=float)

    restricted_design = sm.add_constant(exog.reshape(-1, 1))
    unrestricted_design = sm.add_constant(np.column_stack([exog, instruments]))
    rss_restricted = float(sm.OLS(response, restricted_design).fit().ssr)
    rss_unrestricted = float(sm.OLS(response, unrestricted_design).fit().ssr)
    q = 2.0
    df_denom = float(n - unrestricted_design.shape[1])
    expected_f = ((rss_restricted - rss_unrestricted) / q) / (
        rss_unrestricted / df_denom
    )

    reported = result.estat_weakrobust(ar=True)
    assert reported.f_statistic == pytest.approx(expected_f, rel=1e-10)
    assert reported.statistic == pytest.approx(q * expected_f, rel=1e-10)
    assert reported.df == q


def test_a_joint_null_at_the_estimates_collapses_to_zero(two_endogenous) -> None:
    """Both coefficients set to their estimated values: the AR minimum."""

    result = two_endogenous[1]
    first = float(result.params["e1"])
    second = float(result.params["e2"])

    reported = result.estat_weakrobust(ar=True, betanull={"e1": first, "e2": second})

    assert reported.f_statistic == pytest.approx(0.0, abs=1e-8)


def test_an_unspecified_coefficient_defaults_to_zero(two_endogenous) -> None:
    """``sthlp:463-465``: "null values are set to zero for endogenous regressors you did
    not specify".

    Setting only ``e1`` must therefore differ from setting BOTH to their estimates.
    """

    result = two_endogenous[1]
    first = float(result.params["e1"])
    second = float(result.params["e2"])

    partial = result.estat_weakrobust(ar=True, betanull={"e1": first})
    joint = result.estat_weakrobust(ar=True, betanull={"e1": first, "e2": second})

    assert partial.f_statistic > 1.0
    assert joint.f_statistic == pytest.approx(0.0, abs=1e-8)


def test_the_sequence_form_needs_one_value_per_regressor(two_endogenous) -> None:
    """With two endogenous regressors a single value is the wrong length."""

    result = two_endogenous[1]

    with pytest.raises(ValueError, match="value"):
        result.estat_weakrobust(ar=True, betanull=[0.0])


def test_the_sequence_form_works_when_complete(two_endogenous) -> None:
    result = two_endogenous[1]
    first = float(result.params["e1"])
    second = float(result.params["e2"])

    by_position = result.estat_weakrobust(ar=True, betanull=[first, second])
    by_name = result.estat_weakrobust(
        ar=True, betanull={"e1": first, "e2": second}
    )

    assert by_position.f_statistic == pytest.approx(
        by_name.f_statistic, rel=1e-12
    )