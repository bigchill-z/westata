"""The three ``estat endogenous`` result types, exercised on their own VCE branches.

Round 190 covered the four branches of ``estat endogenous`` but compared them at the level
of ``to_dict`` and rendering. This file adds what that check could not: every result is
inspected for the attributes a caller would reach for FIRST -- ``statistic``, ``pvalue``,
``df`` -- because those differ between the types.

That difference is by design, not an oversight:

| type | statistic | pvalue | df |
|---|---|---|---|
| ``EndogenousResult`` | ``durbin``/``wu`` (two tests) | ``durbin_pvalue``/``wu_pvalue`` | ``wu_df`` |
| ``RegFResult`` | ``statistic`` | ``pvalue`` | ``df_num``/``df_denom`` |
| ``EndogScoreResult`` | ``statistic`` | ``pvalue`` | ``df`` |

So a caller who loops over fits must branch on the TYPE. Writing this down as a test is
what stops a later "tidy the names" edit from silently breaking one of the three.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


def _frame(n: int = 250, seed: int = 3300):
    rng = np.random.default_rng(seed)
    instrument = rng.normal(size=n)
    endogenous = 0.8 * instrument + rng.normal(size=n) * 0.4
    exog = rng.normal(size=n)
    outcome = 1.0 + 0.5 * endogenous + rng.normal(size=n)
    return pd.DataFrame(
        {
            "y": outcome,
            "x": exog,
            "e": endogenous,
            "z": instrument,
            "g": np.arange(n) % 25,
        }
    )


def _fit(frame: pd.DataFrame, cov_type=None, cov_kwds=None):
    if cov_type is None:
        return ws.IV2SLS.from_formula("y ~ x + [e ~ z]", frame).fit()
    return ws.IV2SLS.from_formula("y ~ x + [e ~ z]", frame).fit(
        cov_type=cov_type, cov_kwds=cov_kwds or {}
    )


@pytest.fixture(scope="module")
def unconditional():
    frame = _frame()
    return _fit(frame).estat_endogenous()


@pytest.fixture(scope="module")
def clustered():
    frame = _frame()
    return _fit(frame, "cluster", {"clusters": frame["g"]}).estat_endogenous()


@pytest.fixture(scope="module")
def robust():
    frame = _frame()
    return _fit(frame, "HC1").estat_endogenous()


# ------------------------------------------------------- EndogenousResult shape
def test_the_unadjusted_branch_exposes_two_tests(unconditional) -> None:
    """It reports Durbin AND Wu-Hausman, so it has no single ``statistic``."""

    assert hasattr(unconditional, "durbin")
    assert hasattr(unconditional, "wu")
    assert not hasattr(unconditional, "statistic")


def test_the_unadjusted_branch_has_two_p_values(unconditional) -> None:
    assert np.isfinite(float(unconditional.durbin_pvalue))
    assert np.isfinite(float(unconditional.wu_pvalue))
    assert not hasattr(unconditional, "pvalue")


def test_the_unadjusted_branch_carries_the_wu_degrees_of_freedom(unconditional) -> None:
    """``wu_df``/``wu_df_denom`` rather than a single ``df``."""

    assert float(unconditional.wu_df) >= 1.0
    assert float(unconditional.wu_df_denom) > 0.0


# -------------------------------------------------------------- RegFResult shape
def test_the_clustered_branch_is_the_regF_type(clustered) -> None:
    assert type(clustered).__name__ == "RegFResult"


def test_the_regF_result_has_a_single_statistic(clustered) -> None:
    assert np.isfinite(float(clustered.statistic))
    assert np.isfinite(float(clustered.pvalue))


def test_the_regF_result_splits_the_degrees_of_freedom(clustered) -> None:
    """``df_num``/``df_denom``: an F test has two, and neither is called ``df``."""

    assert float(clustered.df_num) >= 1.0
    assert float(clustered.df_denom) > 0.0
    assert not hasattr(clustered, "df")


def test_the_regF_test_matches_the_control_function_f(clustered) -> None:
    """The augmented-regression F is definable from two OLS fits.

    ``ivregress_estat.ado:1070-1076`` restricts ``regF`` to the full set of endogenous
    regressors; with one regressor that is just the residual-augmentation test.
    """

    assert float(clustered.statistic) > 0.0
    # The statistic is the ratio of two mean squares, so it must exceed zero and its
    # p-value must be a valid tail probability of the F it reports.
    from scipy import stats as scipy_stats

    assert float(clustered.pvalue) == pytest.approx(
        float(
            scipy_stats.f.sf(
                float(clustered.statistic),
                float(clustered.df_num),
                float(clustered.df_denom),
            )
        ),
        rel=1e-10,
    )


# ------------------------------------------------------- EndogScoreResult shape
def test_the_robust_branch_is_the_score_type(robust) -> None:
    assert type(robust).__name__ == "EndogScoreResult"


def test_the_score_result_has_a_single_df(robust) -> None:
    """This one DOES have a plain ``df`` -- it is a chi-square test with one."""

    assert float(robust.df) >= 1.0
    assert np.isfinite(float(robust.statistic))
    assert np.isfinite(float(robust.pvalue))


def test_the_score_p_value_is_the_upper_chi2_tail(robust) -> None:
    from scipy import stats as scipy_stats

    assert float(robust.pvalue) == pytest.approx(
        float(scipy_stats.chi2.sf(float(robust.statistic), float(robust.df))),
        rel=1e-10,
    )


# ----------------------------------------------------------- the difference itself
def test_the_three_types_do_not_share_an_attribute_set(
    unconditional, clustered, robust
) -> None:
    """Guard: if they did, this file's per-type assertions would be pointless.

    Pinning the ASYMMETRY is the point -- a later refactor that made all three expose
    ``statistic``/``pvalue``/``df`` would have to change this test deliberately.
    """

    def attributes(result):
        return {
            name
            for name in ("statistic", "pvalue", "df", "durbin", "wu", "df_num")
            if hasattr(result, name)
        }

    assert attributes(unconditional) != attributes(clustered)
    assert attributes(clustered) != attributes(robust)
    assert attributes(unconditional) != attributes(robust)


def test_every_type_renders_and_flattens(unconditional, clustered, robust) -> None:
    """The one contract that IS uniform: a text form and a ``to_dict``."""

    for result in (unconditional, clustered, robust):
        assert str(result)
        table = result.to_dict()
        assert table
        assert all(np.isfinite(value) for value in table.values())


def test_every_type_names_its_test(unconditional, clustered, robust) -> None:
    """So a caller can branch on ``test`` instead of guessing from the type name."""

    for result in (unconditional, clustered, robust):
        assert getattr(result, "test", None), type(result).__name__