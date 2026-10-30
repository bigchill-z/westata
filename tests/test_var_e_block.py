"""Tests for ``VARResults.e`` -- Stata's ``e()`` scalars after ``var``.

``VARResults`` is a wrapper rather than a ``StataResults``, so it had no ``e``
mapping at all: a caller porting Stata code got an ``AttributeError`` for every
one of the 36 scalars ``var.json`` records.

Two arithmetic details are easy to get backwards and are pinned here:

* ``e(aic)`` and ``e(sbic)`` are **divided by N**, not multiplied out.  Verified
  against the fixture's own numbers: with N=399, ll=-1144.5503147111253 and k=6,
  ``(-2ll+2k)/N = 5.767169497298874`` is exactly the fixture's ``e(aic)``, while
  the un-normalised form is 399x larger -- and both still look like plausible
  information-criterion values.
* ``e(df_m1)`` carries no underscore before the index, unlike ``k_1``/``obs_1``/
  ``r2_1``, because the prefix already ends in ``_m``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import westata as ws

FIXTURE = Path("tests/conformance/stata_fixtures/var.json")


@pytest.fixture()
def var_fit():
    rng = np.random.default_rng(3)
    n = 300
    y1 = np.zeros(n)
    y2 = np.zeros(n)
    for t in range(1, n):
        y1[t] = 0.5 * y1[t - 1] + 0.2 * y2[t - 1] + rng.normal()
        y2[t] = -0.3 * y1[t - 1] + 0.6 * y2[t - 1] + rng.normal()
    return ws.VAR(pd.DataFrame({"y1": y1, "y2": y2}), lags=1).fit()


def test_every_fixture_key_is_present(var_fit) -> None:
    expected = set(json.loads(FIXTURE.read_text())["scalars"])
    missing = expected - set(var_fit.e)

    assert not missing, sorted(missing)


def test_aic_and_sbic_are_per_observation(var_fit) -> None:
    """``e(aic) = (-2ll+2k)/N``; the un-normalised form is N times larger."""

    e = var_fit.e
    n = e["N"]
    ll = e["ll"]
    k = e["k"]

    assert e["aic"] == pytest.approx((-2 * ll + 2 * k) / n, rel=1e-12)
    assert e["sbic"] == pytest.approx((-2 * ll + k * np.log(n)) / n, rel=1e-12)
    # And demonstrably not the multiplied-out form.
    assert not np.isclose(e["aic"], -2 * ll + 2 * k)


def test_the_fixture_arithmetic_reproduces_its_own_aic() -> None:
    """Arithmetic check on Stata's own numbers, independent of our data."""

    s = json.loads(FIXTURE.read_text())["scalars"]

    assert (-2 * s["ll"] + 2 * s["k"]) / s["N"] == pytest.approx(
        s["aic"], rel=1e-12
    )
    assert (-2 * s["ll"] + s["k"] * np.log(s["N"])) / s["N"] == pytest.approx(
        s["sbic"], rel=1e-12
    )


def test_df_m_per_equation_has_no_extra_underscore(var_fit) -> None:
    """Stata spells it ``df_m1``/``df_m2``, while other keys use ``_1``."""

    e = var_fit.e

    assert "df_m1" in e
    assert "df_m2" in e
    assert "df_m_1" not in e
    # ... whereas these DO use the underscore.
    for key in ("k_1", "k_2", "obs_1", "obs_2", "r2_1", "r2_2", "chi2_1", "p_1"):
        assert key in e, key


def test_dimension_block(var_fit) -> None:
    e = var_fit.e
    neqs = e["neqs"]

    assert e["N"] == e["T"]
    assert e["neqs"] == 2.0
    assert e["k_eq"] == neqs
    assert e["k_dv"] == neqs
    assert e["k"] == e["df_eq"] * neqs
    assert e["tparms"] == e["k"]
    assert e["rank"] == e["k"]


def test_time_period_block(var_fit) -> None:
    e = var_fit.e
    lags = var_fit.lags

    assert e["mlag"] == float(lags)
    assert e["N_gaps"] == 0.0
    assert e["tmin"] == float(lags + 1)
    assert e["tmax"] == e["N"] + lags


def test_loglikelihood_is_statsmodels(var_fit) -> None:
    assert var_fit.e["ll"] == pytest.approx(float(var_fit._result.llf), rel=1e-15)


def test_detsig_is_the_covariance_determinant(var_fit) -> None:
    e = var_fit.e

    assert e["detsig"] == pytest.approx(
        float(np.linalg.det(var_fit.sigma_u)), rel=1e-12
    )
    assert e["detsig_ml"] == e["detsig"]


def test_per_equation_r2_and_rmse_match_the_residuals(var_fit) -> None:
    raw = var_fit._result
    resid = np.asarray(raw.resid, dtype=float)
    endog = np.asarray(raw.endog, dtype=float)
    df_eq = var_fit.e["df_eq"]

    for index in range(2):
        r = resid[:, index]
        rss = float(r @ r)
        y = endog[:, index]
        tss = float((y - y.mean()) @ (y - y.mean()))
        assert var_fit.e[f"r2_{index + 1}"] == pytest.approx(
            1.0 - rss / tss, rel=1e-12
        )
        assert var_fit.e[f"rmse_{index + 1}"] == pytest.approx(
            float(np.sqrt(rss / df_eq)), rel=1e-12
        )


def test_per_equation_wald_statistic(var_fit) -> None:
    """``e(chi2_i) = b_s' V_s^-1 b_s`` over that equation's own slopes."""

    e = var_fit.e
    beta = var_fit.stata_params.to_numpy(float)
    cov = var_fit.stata_cov_params
    k_trend = int(var_fit._result.k_trend)

    for index in range(2):
        rows = np.arange(beta.size)[index::2]
        b = beta[rows]
        block = cov[np.ix_(rows, rows)]
        slopes = np.arange(b.size - k_trend)
        expected = float(b[slopes] @ np.linalg.solve(block[np.ix_(slopes, slopes)], b[slopes]))
        assert e[f"chi2_{index + 1}"] == pytest.approx(expected, rel=1e-10)


def test_per_equation_p_is_the_chi2_upper_tail(var_fit) -> None:
    from scipy import stats

    e = var_fit.e
    df = e["df_m1"]

    for index in (1, 2):
        assert e[f"p_{index}"] == pytest.approx(
            float(stats.chi2.sf(e[f"chi2_{index}"], df)), rel=1e-12
        )


#: ``var.sthlp:294+`` lists these as MACROS, not scalars, so they are strings.  The
#: scalar block is everything else.
DOCUMENTED_MACROS = frozenset(
    {
        "cmd",
        "cmdline",
        "depvar",
        "endog",
        "eqnames",
        "exog",
        "exogvars",
        "lags",
        "exlags",
        "title",
        "vce",
        "vcetype",
        "constraints",
        "cnslist_var",
        "lutstats",
        "timevar",
        "tsfmt",
        "properties",
        "predict",
        "marginsok",
        "marginsnotok",
        "marginsdefault",
        "dfk",
        "small",
    }
)


def test_values_are_floats_like_ereturn_list(var_fit) -> None:
    """``ereturn list`` prints scalars as numbers and macros as strings.

    This used to require EVERY value to be a float, which was true only while the
    macros were missing.  ``e(depvar)``, ``e(eqnames)`` and ``e(lags)`` are macros in
    ``var.sthlp``, so insisting on floats would forbid storing them at all.
    """

    for key, value in var_fit.e.items():
        if key in DOCUMENTED_MACROS:
            assert isinstance(value, str), (key, type(value))
        else:
            assert isinstance(value, float), (key, type(value))
