"""Tests for the ``e()`` mappings on ``ManovaResult`` and ``JohansenResult``.

Both are r/e-class Stata commands whose stored results were never exposed.

``manova.sthlp:216-225`` lists eight scalars: ``N``, ``k``, ``k_eq``, ``df_m``,
``df_r``, ``df#``, ``rank``.

``vecrank.sthlp:179-210`` splits into scalars (``N``, ``k_eq``, ``k_dv``,
``tmin``, ``tmax``, ``n_lags``, the four rank choices) and VECTORS (``max``,
``trace``, ``ll``, ``lambda``, ``k_rank``, ``aic``, ``sbic``, ``hqic``).

The structural check the Johansen tests rest on: the ``trace`` statistic must be
reconstructible from the same eigenvalues with the same ``N``,
``trace(r) = -N * sum_{i>r} ln(1-lambda_i)``.  If it is, then ``ll`` and
``trace`` are reading one fit rather than two slightly different ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.estimators.tseries import _johansen_diagnostics, _johansen_statistics

N_OBS = 200


@pytest.fixture()
def var_frame():
    rng = np.random.default_rng(4)
    n = N_OBS
    y1 = np.zeros(n)
    y2 = np.zeros(n)
    for t in range(1, n):
        y1[t] = 0.6 * y1[t - 1] + 0.2 * y2[t - 1] + rng.normal()
        y2[t] = 0.1 * y1[t - 1] + 0.7 * y2[t - 1] + rng.normal()
    return pd.DataFrame({"y1": y1, "y2": y2})


@pytest.fixture()
def manova_frame():
    rng = np.random.default_rng(3)
    n = 120
    groups = np.repeat(np.array(["a", "b", "c"], dtype=object), 40)
    shift = np.repeat(np.array([0.0, 1.0, 2.0]), 40)
    return pd.DataFrame(
        {
            "y1": rng.normal(size=n) + shift,
            "y2": rng.normal(size=n) + shift * 0.5,
            "g": groups,
        }
    )


# ------------------------------------------------------------------- manova
def test_manova_e_has_statas_scalars(manova_frame) -> None:
    result = ws.Manova.from_formula("y1 y2 = g", manova_frame).fit()
    e = result.e

    for key in ("N", "k", "k_eq", "df_m", "df_r", "rank"):
        assert key in e, key


def test_manova_n_and_degrees_of_freedom(manova_frame) -> None:
    result = ws.Manova.from_formula("y1 y2 = g", manova_frame).fit()
    e = result.e

    assert e["N"] == 120.0
    # One term, so df_m is the term's hypothesis df (g - 1).
    assert e["df_m"] == result.df == 2.0
    # Residual df is N - g.
    assert e["df_r"] == 120.0 - 3.0


def test_manova_k_counts_parameters_per_equation(manova_frame) -> None:
    """``k`` = g parameters per equation (intercept + g-1 dummies), times neqs."""

    result = ws.Manova.from_formula("y1 y2 = g", manova_frame).fit()
    e = result.e

    assert e["k_eq"] == 2.0
    assert e["k"] == 3.0 * 2.0
    assert e["rank"] == e["k"]


def test_manova_df_term_key_has_no_extra_underscore(manova_frame) -> None:
    """``df#`` is ``df`` + the term index, with no underscore."""

    result = ws.Manova.from_formula("y1 y2 = g", manova_frame).fit()

    assert "df1" in result.e
    assert "df_1" not in result.e


def test_manova_scalar_values_are_floats(manova_frame) -> None:
    """The SCALAR entries are floats; ``manova`` also stores macros and vectors.

    An earlier version of this test asserted that EVERY value in ``e()`` is a
    float. That was too strong: ``manova.sthlp:229-247`` documents string macros
    (``cmd``, ``depvar``, ``properties``, ...) and per-equation VECTORS
    (``r2``, ``rmse``, ``F``, ``p_F``), so filling in the full documented surface
    necessarily added non-float entries.
    """

    result = ws.Manova.from_formula("y1 y2 = g", manova_frame).fit()

    scalars = ("N", "k", "k_eq", "df_m", "df_r", "df1", "rank")
    for key in scalars:
        assert isinstance(result.e[key], float), (key, type(result.e[key]))

    for key in ("cmd", "depvar", "indepvars", "properties", "estat_cmd"):
        assert isinstance(result.e[key], str), (key, type(result.e[key]))

    for key in ("r2", "rmse", "F", "p_F"):
        assert isinstance(result.e[key], list), (key, type(result.e[key]))
        assert all(isinstance(v, float) for v in result.e[key]), key


# ----------------------------------------------------------------- vecrank
def test_vecrank_has_every_documented_key(var_frame) -> None:
    result = ws.VECM(var_frame, coint_rank=1, lags=1).johansen()
    required = {
        "N",
        "k_eq",
        "k_dv",
        "tmin",
        "tmax",
        "n_lags",
        "k_ce95",
        "k_ce99",
        "k_cesbic",
        "k_cehqic",
        "max",
        "trace",
        "ll",
        "lambda",
        "k_rank",
        "aic",
        "sbic",
        "hqic",
    }

    assert not (required - set(result.e)), sorted(required - set(result.e))


def test_vecrank_vectors_have_one_entry_per_rank(var_frame) -> None:
    result = ws.VECM(var_frame, coint_rank=1, lags=1).johansen()
    e = result.e

    for key in ("trace", "max", "ll", "lambda", "k_rank", "aic", "sbic", "hqic"):
        assert np.asarray(e[key]).size == 2, key


def test_vecrank_trace_is_rebuilt_from_the_same_eigenvalues(var_frame) -> None:
    """The decisive check: ``ll`` and ``trace`` must share one fit.

    ``trace(r) = -N * sum_{i>r} ln(1 - lambda_i)`` with the same ``N`` the log
    likelihood uses.  A different ``N`` in either place breaks this.
    """

    data = var_frame.to_numpy(dtype=float)
    for k_ar_diff in (0, 1, 2):
        diagnostics = _johansen_diagnostics(data, k_ar_diff, 0)
        eigenvalues, trace, maxeig, _ = _johansen_statistics(data, k_ar_diff, 0)
        observations = data.shape[0] - 1 - k_ar_diff

        per = -observations * np.log1p(-eigenvalues)
        rebuilt = np.array([per[i:].sum() for i in range(per.size)])

        assert np.allclose(rebuilt, trace, rtol=1e-10), k_ar_diff
        assert np.allclose(maxeig, per, rtol=1e-10), k_ar_diff
        # And the diagnostics used the same observations.
        assert diagnostics["ll"].size == eigenvalues.size


def test_vecrank_ll_increases_with_rank(var_frame) -> None:
    """More cointegrating relations cannot fit worse."""

    e = ws.VECM(var_frame, coint_rank=1, lags=1).johansen().e

    assert np.all(np.diff(e["ll"]) > 0)


def test_vecrank_k_rank_increases_by_neqs(var_frame) -> None:
    e = ws.VECM(var_frame, coint_rank=1, lags=1).johansen().e

    step = e["k_rank"][1] - e["k_rank"][0]
    assert step == e["k_eq"]


def test_vecrank_n_is_the_johansen_sample(var_frame) -> None:
    """``e(N)`` counts observations after differencing and lagging, not T."""

    result = ws.VECM(var_frame, coint_rank=1, lags=1).johansen()
    e = result.e

    # lags=1 -> k_ar_diff = 0 -> N = T - 1.
    assert e["N"] == float(N_OBS - 1)
    assert e["n_lags"] == 1.0
    assert e["tmin"] == 2.0
    assert e["tmax"] == float(N_OBS)


def test_vecrank_n_lags_tracks_the_var_lag_order(var_frame) -> None:
    """``e(n_lags)`` is Stata's VAR lag count, i.e. ``k_ar_diff + 1``.

    Getting this wrong made ``lags=1`` report ``n_lags = 0``.
    """

    one = ws.VECM(var_frame, coint_rank=1, lags=1).johansen().e
    three = ws.VECM(var_frame, coint_rank=1, lags=3).johansen().e

    assert one["n_lags"] == 1.0
    assert three["n_lags"] == 3.0
    assert one["n_lags"] != three["n_lags"]


def test_vecrank_rank_choice_matches_the_selection_rule(var_frame) -> None:
    """Walk the ranks while the trace exceeds its 95% critical value."""

    result = ws.VECM(var_frame, coint_rank=1, lags=1).johansen()
    e = result.e
    trace = np.asarray(e["trace"], dtype=float)
    critical = result.trace_critical[:, 1]

    count = 0
    for index in range(trace.size):
        if np.isfinite(critical[index]) and trace[index] > critical[index]:
            count += 1
        else:
            break

    assert e["k_ce95"] == float(count)


def test_vecrank_bic_choice_minimises_sbic(var_frame) -> None:
    e = ws.VECM(var_frame, coint_rank=1, lags=1).johansen().e

    assert e["k_cesbic"] == float(int(np.argmin(e["sbic"])))
    assert e["k_cehqic"] == float(int(np.argmin(e["hqic"])))


def test_vecrank_aic_is_per_observation(var_frame) -> None:
    """Same convention as ``var``: divided by N, not multiplied out."""

    e = ws.VECM(var_frame, coint_rank=1, lags=1).johansen().e
    observations = e["N"]

    expected = (-2.0 * e["ll"] + 2.0 * e["k_rank"]) / observations
    assert np.allclose(e["aic"], expected, rtol=1e-10)
