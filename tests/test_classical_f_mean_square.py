"""e(F) is the mean-square ratio, and that is what makes `ereturn list` byte-identical.

Round 252 reverted a change to e(F) after it appeared to break factor models. Round 253 found the
apparent break was MY measurement error -- I hand-picked slope columns and got a different subset
than the test did. Measured properly the two forms agree to 3e-15 on a plain regressor and 6e-14
on a factor model: floating-point noise, and exactly the last digit that was differing.

Stata's own e(F) IS the ratio of mean squares -- recomputing it inside Stata from e(mss)/e(rss)
reproduces e(F) to all sixteen digits -- so the ratio is used for a non-robust covariance and the
Wald form is kept for robust and clustered fits, where the ratio is not what Stata reports.

That distinction is the point of these guards: the two paths must not drift into each other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

GRILIC = r"D:\myCodeProject\test_westata\data\grilic.dta"


@pytest.fixture(scope="module")
def frame():
    try:
        return pd.read_stata(GRILIC).dropna(subset=["lnw", "s", "expr"])
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")


def test_the_classical_f_matches_stat_as_sixteen_digits(frame) -> None:
    """From a live `regress lnw s expr`: 160.1647735969879704."""

    fitted = ws.OLS.from_formula("lnw ~ s + expr", frame).fit()
    assert float(fitted.e["F"]) == 160.1647735969879704


def test_it_equals_the_mean_square_ratio(frame) -> None:
    """e(F) = (e(mss)/e(df_m)) / (e(rss)/e(df_r)), recomputable from the other macros."""

    fitted = ws.OLS.from_formula("lnw ~ s + expr", frame).fit()
    macros = fitted.e
    expected = (float(macros["mss"]) / float(macros["df_m"])) / (
        float(macros["rss"]) / float(macros["df_r"])
    )
    assert float(macros["F"]) == expected


def test_a_robust_fit_does_not_use_the_ratio(frame) -> None:
    """Robust and clustered fits keep the Wald form -- the ratio is a different number there.

    If this ever starts matching the ratio, the covariance-type gate has been lost.
    """

    classical = ws.OLS.from_formula("lnw ~ s + expr", frame).fit()
    robust = ws.OLS.from_formula("lnw ~ s + expr", frame).fit(cov_type="HC1")
    assert float(robust.e["F"]) != pytest.approx(float(classical.e["F"]), rel=1e-6)


def test_the_robust_f_is_the_wald_of_its_own_matrix(frame) -> None:
    """One identity for every non-classical covariance: b'V^-1 b / df_m over the slopes."""

    fitted = ws.OLS.from_formula("lnw ~ s + expr", frame).fit(cov_type="HC1")
    names = list(fitted.param_names)
    slopes = [index for index, name in enumerate(names) if name != "Intercept"]
    beta = np.asarray(fitted.params, dtype=float)[slopes]
    covariance = fitted.cov[np.ix_(slopes, slopes)]
    expected = float(beta @ np.linalg.solve(covariance, beta) / len(slopes))
    assert float(fitted.e["F"]) == pytest.approx(expected, rel=1e-12)
