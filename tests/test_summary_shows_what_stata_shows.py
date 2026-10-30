"""The summary table shows only what Stata's own table shows.

Stata's ``regress`` output has this shape::

    No. Observations ... R-squared
    Df Residuals    ... Adj. R-squared
    Df Model        ... F-statistic

and NOTHING about AIC or BIC -- ``regress`` stores no information criteria at all. Its
stored-results block has no ``e(aic)``/``e(bic)``, and ``regress_auto.json`` (captured from
a real run) contains neither key.

This package nonetheless exposes ``e(aic)`` for callers, computed on the Gaussian-deviance
scale ``N log(rss/N) + 2k`` -- which is NOT the ``estat ic`` value either. Printing it in
the summary therefore put two numbers in the table that Stata's table does not contain, and
that would not match Stata's ``estat ic`` if a reader went looking. The renderer now leaves
them out where Stata does.

The ML families DO store information criteria, so their summaries keep the lines. That
contrast is what the tests below encode.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


def _frame(n: int = 250, seed: int = 6300):
    rng = np.random.default_rng(seed)
    covariate = rng.normal(size=n)
    latent = 0.4 + 0.9 * covariate
    return pd.DataFrame(
        {
            "y": 1.0 + 0.8 * covariate + rng.normal(size=n),
            "yb": (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-latent))).astype(int),
            "ycnt": rng.poisson(np.exp(0.3 + 0.4 * covariate)),
            "x": covariate,
        }
    )


@pytest.fixture(scope="module")
def ols_fit():
    return ws.OLS.from_formula("y ~ x", _frame()).fit()


@pytest.fixture(scope="module")
def logit_fit():
    return ws.Logit.from_formula("yb ~ x", _frame()).fit()


# ------------------------------------------------------------- regress: absent
def test_regress_summary_has_no_aic_line(ols_fit) -> None:
    """Stata's ``regress`` table has no AIC, so this one must not either."""

    assert "AIC:" not in str(ols_fit.summary())


def test_regress_summary_has_no_bic_line(ols_fit) -> None:
    assert "BIC:" not in str(ols_fit.summary())


def test_regress_still_exposes_the_key_for_callers(ols_fit) -> None:
    """Removing it from the TABLE is not the same as removing it from ``e()``.

    Callers already read ``result.e["aic"]``; the change is about what gets printed.
    """

    assert "aic" in ols_fit.e
    assert np.isfinite(float(ols_fit.e["aic"]))


def test_the_flag_records_that_this_aic_is_not_statas(ols_fit) -> None:
    """The renderer branches on this, so it must be present and False here."""

    assert ols_fit.e["_aic_is_stata"] is False


def test_estat_ic_remains_the_stata_conformant_route(ols_fit) -> None:
    """The AIC a Stata user wants is the one in ``estat ic``, not ``e(aic)``."""

    rendered = ols_fit.estat_ic()
    e = ols_fit.e

    assert rendered.aic == pytest.approx(
        -2.0 * float(e["ll"]) + 2.0 * float(rendered.df), rel=1e-12
    )
    # And it is plainly a different number from the convenience key.
    assert rendered.aic != pytest.approx(float(e["aic"]), rel=1e-6)


# ----------------------------------------------------------- ML: still shown
def test_the_logit_summary_keeps_its_aic_line(logit_fit) -> None:
    """The ML families DO store information criteria, so the line stays."""

    assert "AIC:" in str(logit_fit.summary())


def test_the_poisson_summary_keeps_its_aic_line() -> None:
    fitted = ws.Poisson.from_formula("ycnt ~ x", _frame()).fit()

    assert "AIC:" in str(fitted.summary())


# ------------------------------------------------------- the rest of the table
def test_the_regress_table_keeps_what_stata_prints(ols_fit) -> None:
    """Guard: the change must remove AIC/BIC and nothing else."""

    rendered = str(ols_fit.summary())

    for label in (
        "No. Observations:",
        "Df Residuals:",
        "Df Model:",
        "R-squared:",
        "Adj. R-squared:",
        "F-statistic:",
        "Log-Likelihood:",
    ):
        assert label in rendered, label


def test_the_coefficient_table_is_still_there(ols_fit) -> None:
    rendered = str(ols_fit.summary())

    assert "coef" in rendered
    assert "std err" in rendered
    assert "P>|t|" in rendered
    assert "Intercept" in rendered


def test_the_two_summaries_differ_in_the_expected_way(ols_fit, logit_fit) -> None:
    """One shows AIC, one does not -- the discrimination is real, not a blanket removal."""

    assert "AIC:" not in str(ols_fit.summary())
    assert "AIC:" in str(logit_fit.summary())