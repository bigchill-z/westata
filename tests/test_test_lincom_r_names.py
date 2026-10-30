"""``test`` and ``lincom`` must expose their results under Stata's ``r()`` names.

Both objects had the values as attributes but no ``to_dict``, so a caller following the
help file could not reach them by the documented names at all.

Two documented shapes, and the difference is not cosmetic:

* ``regress_postestimation.sthlp:770-785`` -- an F test stores ``r(F)``/``r(df)``/
  ``r(df_r)``/``r(p)``, while a chi-square test stores ``r(chi2)``/``r(df)``/``r(p)``.
  The statistic travels under the name of its OWN distribution, so ``r(F)`` after a
  likelihood test is absent -- as in Stata.
* ``lincom.sthlp:182-189`` -- ``r(estimate)``, ``r(se)``, ``r(df)``, ``r(t)`` or
  ``r(z)``, ``r(p)``, ``r(lb)``, ``r(ub)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def frames():
    rng = np.random.default_rng(769)
    n = 200
    latent = rng.normal(size=n)
    linear = pd.DataFrame(
        {
            "y": 1.0 + 0.8 * rng.normal(size=n) + rng.normal(size=n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "yb": (latent > 0).astype(int),
        }
    )
    return linear


@pytest.fixture(scope="module")
def ols(frames):
    return ws.OLS.from_formula("y ~ x1 + x2", frames).fit()


@pytest.fixture(scope="module")
def logit(frames):
    return ws.Logit.from_formula("yb ~ x1", frames).fit()


# --------------------------------------------------------------- test, F form
def test_an_f_test_reports_r_F_and_r_df_r(ols) -> None:
    table = ols.test("x1 = 0, x2 = 0").to_dict()

    assert set(table) == {"F", "df", "df_r", "p"}
    assert table["df"] == 2.0
    assert table["df_r"] == ols.nobs - 3  # n - k
    assert np.isfinite(table["F"])


def test_the_f_test_does_not_claim_a_chi2(ols) -> None:
    """``r(chi2)`` belongs to the likelihood shape; storing both would double-report."""

    assert "chi2" not in ols.test("x1 = 0").to_dict()


# ------------------------------------------------------------ test, chi2 form
def test_a_chi2_test_reports_r_chi2_not_r_F(logit) -> None:
    table = logit.test("x1 = 0").to_dict()

    assert set(table) == {"chi2", "df", "p"}
    assert "F" not in table
    assert "df_r" not in table  # a chi-square test has no denominator df
    assert np.isfinite(table["chi2"])


def test_the_statistics_match_the_objects(ols, logit) -> None:
    """``to_dict`` must not recompute anything."""

    tested = ols.test("x1 = 0")
    assert tested.to_dict()["F"] == pytest.approx(float(tested.statistic), rel=1e-12)

    like = logit.test("x1 = 0")
    assert like.to_dict()["chi2"] == pytest.approx(float(like.statistic), rel=1e-12)


# ------------------------------------------------------------------- lincom
def test_lincom_reports_all_seven_scalars(ols) -> None:
    table = ols.lincom("x1 + x2").to_dict()

    assert set(table) == {"estimate", "se", "df", "t", "p", "lb", "ub"}


def test_lincom_uses_r_z_for_a_z_based_fit(frames) -> None:
    """The statistic key follows the distribution, not a fixed name."""

    logit = ws.Logit.from_formula("yb ~ x1", frames).fit()
    table = logit.lincom("x1").to_dict()

    assert "z" in table
    assert "t" not in table


def test_lincom_uses_r_t_for_a_t_based_fit(ols) -> None:
    table = ols.lincom("x1").to_dict()

    assert "t" in table
    assert "z" not in table


def test_the_lincom_values_are_the_objects(ols) -> None:
    computed = ols.lincom("x1 + x2")
    table = computed.to_dict()

    assert table["estimate"] == pytest.approx(float(computed.estimate), rel=1e-12)
    assert table["se"] == pytest.approx(float(computed.std_err), rel=1e-12)
    assert table["lb"] == pytest.approx(float(computed.ci_lower), rel=1e-12)
    assert table["ub"] == pytest.approx(float(computed.ci_upper), rel=1e-12)
    assert table["p"] == pytest.approx(float(computed.pvalue), rel=1e-12)


def test_the_interval_contains_the_estimate(ols) -> None:
    """A cheap structural check on the CI that does not restate the arithmetic."""

    table = ols.lincom("x1 + x2").to_dict()

    assert table["lb"] < table["estimate"] < table["ub"]


def test_the_statistic_is_estimate_over_se(ols) -> None:
    """``r(t) = r(estimate)/r(se)`` -- an identity across three of the scalars."""

    table = ols.lincom("x1 - x2").to_dict()

    assert table["t"] == pytest.approx(
        table["estimate"] / table["se"], rel=1e-10
    )