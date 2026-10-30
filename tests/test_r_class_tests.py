"""Tests for the ``r()`` mappings on the time-series and panel test results.

These are r-class commands in Stata -- ``dfuller``, ``pperron``, ``kpss`` and
``xtunitroot`` -- so their outputs live in ``r()`` rather than ``e()``.  The
result objects carried the numbers but exposed no ``r`` mapping at all.

Scalar names come from the ado files and help entries:

* ``dfuller.ado:134-166``  -- ``r(Zt)``, ``r(N)``, ``r(lags)``,
  ``r(cv_1)``/``r(cv_5)``/``r(cv_10)``, ``r(p)``
* ``pperron.ado:136-155``  -- the same, with ``r(pval)`` and ``r(Zrho)``
* ``xtunitroot.sthlp:755-870`` -- ``r(N)``, ``r(N_g)``, ``r(N_t)``, plus the
  statistic and its p-value under per-test names
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture()
def random_walk():
    rng = np.random.default_rng(2)
    return np.cumsum(rng.normal(size=150))


@pytest.fixture()
def panel():
    rng = np.random.default_rng(7)
    entities, periods = 30, 20
    return pd.DataFrame(
        {
            "id": np.repeat(np.arange(1, entities + 1), periods),
            "t": np.tile(np.arange(periods), entities),
            "y": np.cumsum(rng.normal(size=entities * periods)),
        }
    )


# ------------------------------------------------------------ unit-root r()
def test_adf_uses_z_t_and_p(random_walk) -> None:
    result = ws.unitroot(random_walk, test="adf")
    r = result.r

    assert r["Zt"] == result.statistic
    assert r["p"] == result.pvalue
    assert r["N"] == float(result.observations)
    assert r["lags"] == float(result.lags)


def test_pperron_adds_pval_and_keeps_z_t(random_walk) -> None:
    """``pperron.ado:136`` spells the p-value ``pval``, not ``p``."""

    result = ws.unitroot(random_walk, test="pp")
    r = result.r

    assert r["Zt"] == result.statistic
    assert r["pval"] == result.pvalue
    # Both spellings are recorded because callers port either one.
    assert r["p"] == result.pvalue


def test_kpss_uses_stat_and_has_no_macKinnon_p(random_walk) -> None:
    """``kpss`` tests the opposite null, so Stata stores ``r(stat)`` only."""

    result = ws.unitroot(random_walk, test="kpss")
    r = result.r

    assert r["stat"] == result.statistic
    assert "p" not in r


def test_critical_values_become_cv_1_cv_5_cv_10(random_walk) -> None:
    for test in ("adf", "pp"):
        r = ws.unitroot(random_walk, test=test).r
        assert set(r) >= {"cv_1", "cv_5", "cv_10"}, test


def test_all_unit_root_values_are_floats(random_walk) -> None:
    for test in ("adf", "pp", "kpss"):
        for key, value in ws.unitroot(random_walk, test=test).r.items():
            assert isinstance(value, float), (test, key, type(value))


# ------------------------------------------------------- panel unit-root r()
def test_ips_r_block(panel) -> None:
    result = ws.XtUnitroot.from_formula(
        "y ~ 1", panel, entity="id", time="t", test="ips"
    ).fit()
    r = result.r

    assert r["test"] == "ips"
    # The headline statistic is Stata's Z-t-tilde-bar.
    assert r["Zt"] == result.statistic
    assert r["p"] == result.pvalue
    # The table's other rows survive from ``details``.
    assert "t-bar" in r
    assert "t-tilde-bar" in r
    assert "N_t" in r


def test_llc_r_block(panel) -> None:
    """``llc`` reports ``r(tds)``/``r(p_tds)`` for the adjusted t* statistic."""

    result = ws.XtUnitroot.from_formula(
        "y ~ 1", panel, entity="id", time="t", test="llc"
    ).fit()
    r = result.r

    assert r["test"] == "llc"
    assert r["tds"] == result.statistic
    assert r["p_tds"] == result.pvalue
    assert "delta" in r
    assert "sbar" in r


def test_the_two_panel_tests_do_not_share_statistic_names(panel) -> None:
    """A shared key would make a combined table ambiguous."""

    ips = ws.XtUnitroot.from_formula(
        "y ~ 1", panel, entity="id", time="t", test="ips"
    ).fit()
    llc = ws.XtUnitroot.from_formula(
        "y ~ 1", panel, entity="id", time="t", test="llc"
    ).fit()

    assert "Zt" in ips.r and "tds" not in ips.r
    assert "tds" in llc.r and "Zt" not in llc.r


def test_panel_details_are_carried_through(panel) -> None:
    result = ws.XtUnitroot.from_formula(
        "y ~ 1", panel, entity="id", time="t", test="ips"
    ).fit()

    if result.details:
        for key in result.details:
            assert str(key) in result.r


def test_panel_r_values_are_floats_or_the_test_macro(panel) -> None:
    result = ws.XtUnitroot.from_formula(
        "y ~ 1", panel, entity="id", time="t", test="ips"
    ).fit()

    for key, value in result.r.items():
        if key == "test":
            assert isinstance(value, str)
        else:
            assert isinstance(value, float), (key, type(value))
