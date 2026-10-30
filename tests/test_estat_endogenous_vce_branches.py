"""``estat endogenous`` returns FOUR different result types -- and that is correct.

Which test Stata runs depends on the estimation VCE:

| VCE | test | result type | `r()` keys |
|---|---|---|---|
| unadjusted | Durbin + Wu-Hausman | ``EndogenousResult`` | ``durbin``, ``wu`` |
| clustered | robust regression F | ``RegFResult`` | ``regF`` |
| HC / robust | robust score | ``EndogScoreResult`` | ``r_score`` |
| HAC | HAC score | ``EndogScoreResult`` (+``lags``) | ``hac_score`` |

These are genuinely different statistics, not four spellings of one.  The risk is
therefore not that the types differ -- it is that only ONE of them is exercised while the
others rot.  This file pins all four, plus the contract they DO share: every one exposes
``to_dict()`` with the ``r()`` names of its own test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


def _frame(n: int = 250, seed: int = 2010):
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
def plain():
    frame = _frame()
    return frame, _fit(frame)


@pytest.fixture(scope="module")
def clustered():
    frame = _frame()
    return frame, _fit(frame, "cluster", {"clusters": frame["g"]})


@pytest.fixture(scope="module")
def robust():
    frame = _frame()
    return frame, _fit(frame, "HC1")


@pytest.fixture(scope="module")
def hac():
    frame = _frame()
    return frame, _fit(frame, "HAC", {"maxlags": 2})


# ---------------------------------------------------------- the unadjusted branch
def test_the_unadjusted_branch_reports_durbin_and_wu(plain) -> None:
    reported = plain[1].estat_endogenous()

    assert type(reported).__name__ == "EndogenousResult"
    assert np.isfinite(float(reported.durbin))
    assert np.isfinite(float(reported.wu))


def test_the_durbin_wu_keys(plain) -> None:
    table = plain[1].estat_endogenous().to_dict()

    assert set(table) == {"durbin", "p_durbin", "wu", "p_wu", "wudf_r"}


# --------------------------------------------------------------- the cluster branch
def test_the_clustered_branch_reports_a_robust_f(clustered) -> None:
    """A clustered VCE switches to Stata's robust regression F test."""

    reported = clustered[1].estat_endogenous()

    assert type(reported).__name__ == "RegFResult"
    assert np.isfinite(float(reported.statistic))


def test_the_cluster_keys( clustered) -> None:
    table = clustered[1].estat_endogenous().to_dict()

    assert set(table) == {"regF", "regFdf_d", "regFdf_n", "p_regF"}


# ----------------------------------------------------------------- the HC branch
def test_the_robust_branch_reports_a_score(robust) -> None:
    """``ivregress_postestimation.sthlp:384`` names the "version of the score test"."""

    reported = robust[1].estat_endogenous()

    assert type(reported).__name__ == "EndogScoreResult"
    assert np.isfinite(float(reported.statistic))


def test_the_robust_score_keys(robust) -> None:
    table = robust[1].estat_endogenous().to_dict()

    assert set(table) == {"r_score", "p_r_score"}


# ------------------------------------------------------------------ the HAC branch
def test_the_hac_branch_adds_lags(hac) -> None:
    reported = hac[1].estat_endogenous()

    assert type(reported).__name__ == "EndogScoreResult"
    assert "lags" in reported.to_dict()


def test_the_hac_keys(hac) -> None:
    table = hac[1].estat_endogenous().to_dict()

    assert set(table) == {"hac_score", "lags", "p_hac_score"}


# --------------------------------------------------------------- the contract
def test_all_four_expose_to_dict(plain, clustered, robust, hac) -> None:
    """Different statistics, one uniform access path."""

    for _, fitted in (plain, clustered, robust, hac):
        reported = fitted.estat_endogenous()
        assert hasattr(reported, "to_dict")
        assert reported.to_dict()


def test_all_four_expose_p_values_under_their_own_names(
    plain, clustered, robust, hac
) -> None:
    """The p-values are reachable, but the unadjusted branch has TWO of them.

    ``EndogenousResult`` reports Durbin AND Wu-Hausman, so a single ``pvalue`` would be
    ambiguous -- it exposes ``durbin_pvalue``/``wu_pvalue`` instead.  The other three run
    one test each and carry a plain ``pvalue``.  Asserting a uniform attribute name would
    be forcing a false symmetry, so each shape is checked for its own.
    """

    unadjusted = plain[1].estat_endogenous()
    assert np.isfinite(float(unadjusted.durbin_pvalue))
    assert np.isfinite(float(unadjusted.wu_pvalue))
    assert not hasattr(unadjusted, "pvalue")

    for _, fitted in (clustered, robust, hac):
        reported = fitted.estat_endogenous()
        assert np.isfinite(float(reported.pvalue)), type(reported).__name__


def test_all_four_render_text(plain, clustered, robust, hac) -> None:
    for _, fitted in (plain, clustered, robust, hac):
        text = str(fitted.estat_endogenous())
        assert "endogeneity" in text.lower()


def test_the_three_single_test_statistics_are_not_all_the_same_number(
    clustered, robust, hac
) -> None:
    """Guard: if the branches all computed one statistic, the type differences would be
    cosmetic.  They are separate tests on the same data, so they must differ.

    Only the three single-statistic branches are compared; ``EndogenousResult`` exposes
    ``durbin``/``wu`` rather than one ``statistic``.
    """

    values = [
        float(fitted.estat_endogenous().statistic)
        for _, fitted in (clustered, robust, hac)
    ]

    assert len(set(round(value, 10) for value in values)) > 1