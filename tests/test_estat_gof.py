"""Tests for Stata's ``estat gof`` (Poisson goodness of fit).

The statistics are asserted against an independent transcription of
``poisgof.ado`` and against the per-observation deviance identity, so a change to
either formula shows up as a numeric failure rather than a passing shape check.

``poisgof.ado`` is Poisson-only: ``poisson.ado`` records
``e(estat_cmd) = "poisson_estat"``, which forwards ``gof`` to ``poisgof``;
``nbreg`` records no ``estat_cmd`` and therefore has no ``estat gof``.  That
asymmetry is tested explicitly, because inventing the command for ``nbreg`` would
mean reporting a statistic Stata does not produce.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.special import gammaln

import westata as ws
from westata.postestimation.gof import estat_gof


@pytest.fixture()
def poisson_fit():
    rng = np.random.default_rng(5)
    n = 500
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    mu = np.exp(0.3 + 0.6 * x1 - 0.2 * x2)
    y = rng.poisson(mu).astype(float)
    frame = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    return frame, ws.Poisson.from_formula("y ~ x1 + x2", frame).fit()


def test_pearson_matches_the_ado_formula(poisson_fit) -> None:
    frame, result = poisson_fit
    mu = np.asarray(result.fittedvalues, dtype=float)
    y = frame["y"].to_numpy(dtype=float)

    expected = float(np.sum((y - mu) ** 2 / mu))

    assert result.estat_gof().pearson == pytest.approx(expected, rel=1e-12)


def test_deviance_matches_the_ado_formula(poisson_fit) -> None:
    """``chi2_d = -2*(e(ll) - llperf)`` with ``llperf = P - llconst``.

    ``llconst = sum lngamma(1+y)`` is the term that is easy to drop; without it
    the statistic comes out roughly three times too large.
    """

    frame, result = poisson_fit
    y = frame["y"].to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(
            y == 0.0, 0.0, y * (np.log(np.where(y == 0.0, 1.0, y)) - 1.0)
        )
    ll_const = float(np.sum(gammaln(1.0 + y)))
    ll_perfect = float(np.sum(p)) - ll_const

    expected = -2.0 * (float(result.e["ll"]) - ll_perfect)

    assert result.estat_gof().deviance == pytest.approx(expected, rel=1e-12)


def test_deviance_equals_the_sum_of_deviance_residuals_squared(poisson_fit) -> None:
    """A second, formula-independent check of the deviance."""

    frame, result = poisson_fit
    y = frame["y"].to_numpy(dtype=float)
    mu = np.asarray(result.fittedvalues, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        per_obs = 2.0 * (
            np.where(y == 0.0, 0.0, y * np.log(np.where(y == 0.0, 1.0, y) / mu))
            - (y - mu)
        )

    assert result.estat_gof().deviance == pytest.approx(
        float(np.sum(per_obs)), rel=1e-8
    )


def test_df_is_n_minus_df_m_minus_one(poisson_fit) -> None:
    _, result = poisson_fit
    expected = float(result.e["N"] - result.e["df_m"] - 1.0)

    assert result.estat_gof().df == expected


def test_pvalues_are_upper_tail_chi_square(poisson_fit) -> None:
    _, result = poisson_fit
    gof = result.estat_gof()

    assert gof.deviance_pvalue == pytest.approx(
        float(stats.chi2.sf(gof.deviance, gof.df)), rel=1e-12
    )
    assert gof.pearson_pvalue == pytest.approx(
        float(stats.chi2.sf(gof.pearson, gof.df)), rel=1e-12
    )
    assert 0.0 <= gof.deviance_pvalue <= 1.0
    assert 0.0 <= gof.pearson_pvalue <= 1.0


def test_to_dict_uses_stata_r_names(poisson_fit) -> None:
    """``poisgof.ado`` returns ``r(chi2_d)``/``r(chi2_p)``/``r(df)``/``r(p_*)``."""

    _, result = poisson_fit
    payload = result.estat_gof().to_dict()

    assert set(payload) == {"chi2_d", "chi2_p", "df", "p_d", "p_p"}
    assert payload["chi2_d"] == result.estat_gof().deviance
    assert payload["chi2_p"] == result.estat_gof().pearson


def test_module_function_matches_the_method(poisson_fit) -> None:
    _, result = poisson_fit

    assert estat_gof(result).to_dict() == result.estat_gof().to_dict()


def test_estat_gof_refuses_nbreg(poisson_fit) -> None:
    """Stata has no ``estat gof`` after ``nbreg``, so westata must refuse."""

    frame, result = poisson_fit
    nb = ws.NegativeBinomial.from_formula("y ~ x1", frame).fit()

    with pytest.raises(NotImplementedError, match="poisson"):
        nb.estat_gof()

    # The Poisson result must still work, so the guard is about the model.
    assert result.estat_gof().df > 0


def test_zero_counts_do_not_produce_nan(poisson_fit) -> None:
    """``y ln y`` is taken as 0 at ``y = 0`` (the ``cond(y==0, 0, ...)`` guard)."""

    frame, result = poisson_fit
    assert (frame["y"] == 0).any(), "the fixture should contain zeros"
    gof = result.estat_gof()

    assert np.isfinite(gof.deviance)
    assert np.isfinite(gof.pearson)


def test_gof_prints_both_blocks(poisson_fit) -> None:
    _, result = poisson_fit
    text = str(result.estat_gof())

    assert "Deviance goodness-of-fit" in text
    assert "Pearson goodness-of-fit" in text
    assert "Prob > chi2(" in text


def test_poisson_gof_has_no_group_option(poisson_fit) -> None:
    """``poisgof.ado:3`` parses only ``[, Pearson]`` -- there is no ``group()``.

    Silently ignoring the argument used to run the UNCOLLAPSED deviance test while the
    caller believed they had asked for a Hosmer-Lemeshow collapse.  The two statistics
    are not interchangeable, and only the label would have suggested otherwise.

    The Hosmer-Lemeshow form belongs to ``lfit`` (``logit``/``probit``), where
    ``lfit.ado:26-41`` does accept ``group()``.
    """

    _, result = poisson_fit

    # The plain call still works ...
    assert np.isfinite(result.estat_gof().deviance)

    # ... but the grouped one is not a poisson request at all.
    with pytest.raises(ValueError, match="no group"):
        result.estat_gof(group=10)
