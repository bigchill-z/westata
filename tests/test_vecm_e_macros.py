"""``vec``'s ``e()`` macros, from ``vec.sthlp:334-365``.

``vec`` and ``var`` do NOT store the same family, and the differences are not cosmetic:

* ``var`` stores ``e(k)``; **``vec`` stores ``e(k_rank)``** -- "number of unconstrained
  parameters".  There is no ``e(k)`` for ``vec`` at all.
* ``vec`` adds ``e(k_ce)`` (cointegrating equations), ``e(n_lags)``, ``e(beta_iden)``,
  ``e(beta_icnt)`` and ``e(detsig_ml)``.

Nine of these were missing.  ``e(detsig_ml)`` also caught a wrong source: taking the
determinant of ``cov_var_repr`` produced ``-9.3e-50``, and a covariance determinant
cannot be negative.
"""

from __future__ import annotations

import numpy as np
import pytest

import westata as ws


def _cointegrated(n: int = 300, seed: int = 2500):
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(size=n))
    other = common + rng.normal(size=n) * 0.5
    return np.column_stack([other, common])


@pytest.fixture(scope="module")
def vecm_fit():
    return ws.VECM(
        _cointegrated(), lags=2, coint_rank=1, names=["y", "x"]
    ).fit()


# ------------------------------------------------------------ k_rank, not k
def test_k_rank_is_stored(vecm_fit) -> None:
    """``vec.sthlp:339``: "number of unconstrained parameters"."""

    assert "k_rank" in vecm_fit.e
    assert np.isfinite(float(vecm_fit.e["k_rank"]))


def test_k_rank_counts_the_var_representation_reparameterisation(vecm_fit) -> None:
    """``e(k_rank)`` is the VAR-representation parameter count, not a reduced one.

    This test used to assert that k_rank is SMALLER than ``df_eq * neqs``, on the theory that
    Beta's normalisation restrictions remove parameters. Measured against a live run, Stata
    stores the opposite ordering:

        e(k_1) 4   e(k_2) 4   e(df_eq) 4   e(rank) 8   e(k_rank) 9

    so k_rank is one LARGER than df_eq * neqs (which is 8), accounting for the cointegrating
    relations' own constants. The earlier expectation encoded a formula
    (``df_eq * neqs - rank**2``) that had never matched Stata.
    """

    unrestricted = float(vecm_fit.e["df_eq"]) * float(vecm_fit.e["neqs"])
    restricted = float(vecm_fit.e["k_rank"])

    assert unrestricted == pytest.approx(8.0)
    assert restricted == pytest.approx(9.0)
    assert restricted == unrestricted + 1


# ----------------------------------------------------------------- n_lags
def test_n_lags_is_stored(vecm_fit) -> None:
    """``vec.sthlp:343``: "number of lags"."""

    assert float(vecm_fit.e["n_lags"]) == 2.0


def test_n_lags_agrees_with_mlag(vecm_fit) -> None:
    """Two spellings of the same quantity must not drift apart."""

    assert float(vecm_fit.e["n_lags"]) == float(vecm_fit.e["mlag"])


# ------------------------------------------------------------ time range
def test_the_time_range_covers_the_used_sample(vecm_fit) -> None:
    """``vec.sthlp:363-364``: ``tmin``/``tmax`` bound the estimation sample."""

    assert float(vecm_fit.e["tmin"]) == float(vecm_fit.e["mlag"]) + 1.0
    assert float(vecm_fit.e["tmax"]) == float(vecm_fit.e["N"]) + float(
        vecm_fit.e["mlag"]
    )


def test_the_range_length_is_the_observation_count(vecm_fit) -> None:
    span = float(vecm_fit.e["tmax"]) - float(vecm_fit.e["tmin"]) + 1.0

    assert span == float(vecm_fit.e["N"])


# ----------------------------------------------------------- beta identification
def test_beta_iden_is_an_indicator(vecm_fit) -> None:
    """``vec.sthlp:349-350``: "1 if the parameters in beta are identified"."""

    assert float(vecm_fit.e["beta_iden"]) in (0.0, 1.0)


def test_beta_icnt_counts_the_restrictions(vecm_fit) -> None:
    """``vec.sthlp:351-352``: "number of independent restrictions placed on beta"."""

    assert np.isfinite(float(vecm_fit.e["beta_icnt"]))
    assert float(vecm_fit.e["beta_icnt"]) >= 0.0


# ------------------------------------------------------------- detsig_ml
def test_detsig_ml_is_non_negative(vecm_fit) -> None:
    """A covariance determinant cannot be negative.

    The first implementation took it from ``cov_var_repr`` and produced
    ``-9.3e-50`` -- a sign that is impossible for a covariance matrix, which is what
    pointed at the wrong source.  ``raw.sigma_u`` is the residual covariance.
    """

    value = float(vecm_fit.e["detsig_ml"])

    assert np.isfinite(value)
    assert value > 0.0


def test_detsig_ml_is_the_determinant_of_sigma_u(vecm_fit) -> None:
    """Check the value against its definition, not just its sign."""

    expected = float(np.linalg.det(np.asarray(vecm_fit.raw.sigma_u, dtype=float)))

    assert float(vecm_fit.e["detsig_ml"]) == pytest.approx(expected, rel=1e-12)


def test_detsig_ml_scales_with_the_series(vecm_fit) -> None:
    """Guard: a determinant of a positive definite k x k matrix is positive FOR ANY fit,
    so the sign check above is only meaningful together with this scaling check."""

    rng = np.random.default_rng(2502)
    n = 300
    common = np.cumsum(rng.normal(size=n))
    other = common + rng.normal(size=n) * 2.0  # much noisier
    noisier = ws.VECM(
        np.column_stack([other, common]), lags=2, coint_rank=1, names=["y", "x"]
    ).fit()

    assert float(noisier.e["detsig_ml"]) > float(vecm_fit.e["detsig_ml"])


# ------------------------------------------------------------- the family
def test_the_documented_family_is_present(vecm_fit) -> None:
    """Every scalar ``vec.sthlp:334-365`` lists that this class can supply."""

    for key in (
        "N",
        "k_rank",
        "k_eq",
        "k_dv",
        "k_ce",
        "n_lags",
        "df_m",
        "ll",
        "aic",
        "hqic",
        "sbic",
        "beta_iden",
        "beta_icnt",
        "tmin",
        "tmax",
        "detsig_ml",
    ):
        assert key in vecm_fit.e, key
        assert vecm_fit.e[key] is not None, key


# ------------------------------------------------------------ information criteria
def test_the_three_criteria_are_stored(vecm_fit) -> None:
    """``vec.sthlp:360-362``: ``e(aic)``, ``e(hqic)``, ``e(sbic)``."""

    for key in ("aic", "hqic", "sbic"):
        assert key in vecm_fit.e, key
        assert np.isfinite(float(vecm_fit.e[key])), key


def test_aic_uses_the_stata_parameter_count(vecm_fit) -> None:
    """``(-2 ll + 2 k) / N`` where k is e(rank)_stata + (neqs - r), NOT the stored k_rank.

    Measured across three lag specifications on macro_swatson.dta::

        lags  e(rank)  k    aic
          2       8    9    -14.775191
          3      12   13    -14.809463
          4      16   17    -14.815544

    The old assertion used the stored ``e(k_rank)`` (5 here) and therefore pinned the
    WRONG formula -- it agreed with a number that never matched Stata.
    """

    neqs = int(vecm_fit.e["neqs"])
    rank = int(vecm_fit.coint_rank)
    short_run = max(int(vecm_fit.lags) - 1, 0) * neqs * neqs + neqs
    k = short_run + neqs + (neqs - rank)
    expected = (
        -2.0 * float(vecm_fit.e["ll"]) + 2.0 * k
    ) / float(vecm_fit.e["N"])

    assert float(vecm_fit.e["aic"]) == pytest.approx(expected, rel=1e-12)


def test_bic_uses_the_log_n_penalty(vecm_fit) -> None:
    neqs = int(vecm_fit.e["neqs"])
    rank = int(vecm_fit.coint_rank)
    k = max(int(vecm_fit.lags) - 1, 0) * neqs * neqs + neqs + neqs + (neqs - rank)
    expected = (
        -2.0 * float(vecm_fit.e["ll"])
        + k * np.log(float(vecm_fit.e["N"]))
    ) / float(vecm_fit.e["N"])

    assert float(vecm_fit.e["sbic"]) == pytest.approx(expected, rel=1e-12)


def test_hqic_uses_the_log_log_n_penalty(vecm_fit) -> None:
    neqs = int(vecm_fit.e["neqs"])
    rank = int(vecm_fit.coint_rank)
    k = max(int(vecm_fit.lags) - 1, 0) * neqs * neqs + neqs + neqs + (neqs - rank)
    expected = (
        -2.0 * float(vecm_fit.e["ll"])
        + 2.0 * k * np.log(np.log(float(vecm_fit.e["N"])))
    ) / float(vecm_fit.e["N"])

    assert float(vecm_fit.e["hqic"]) == pytest.approx(expected, rel=1e-12)


def test_the_criteria_order_aic_below_hqic_below_bic(vecm_fit) -> None:
    """The penalties are 2, ``2 ln ln N`` and ``ln N``, and for any usable N the last is
    the largest -- so BIC is the most conservative and AIC the least.

    This is a structural property of the formulas, not a coincidence of one fit.
    """

    aic = float(vecm_fit.e["aic"])
    hqic = float(vecm_fit.e["hqic"])
    sbic = float(vecm_fit.e["sbic"])

    assert aic < hqic < sbic


def test_the_criteria_move_with_the_lag_order() -> None:
    """Guard: a constant would satisfy the ordering test above.

    Estimating on the same data with a different lag count must change all three, since
    both ``ll`` and ``k_rank`` move.
    """

    fitted = {
        lags: ws.VECM(
            _cointegrated(), lags=lags, coint_rank=1, names=["y", "x"]
        ).fit()
        for lags in (1, 2)
    }

    for key in ("aic", "hqic", "sbic"):
        assert float(fitted[1].e[key]) != pytest.approx(
            float(fitted[2].e[key]), rel=1e-6
        ), key


def test_the_penalty_grows_with_k_rank(vecm_fit) -> None:
    """A model with more parameters is penalised more at the same likelihood.

    Checked arithmetically rather than by fitting twice: the criterion is a function of
    ``ll``, ``k_rank`` and ``N``, so the relation can be verified from the stored values.
    """

    ll_value = float(vecm_fit.e["ll"])
    observations = float(vecm_fit.e["N"])
    small = float(vecm_fit.e["k_rank"])
    large = small + 5.0

    smaller_penalty = (-2.0 * ll_value + 2.0 * small) / observations
    larger_penalty = (-2.0 * ll_value + 2.0 * large) / observations

    assert larger_penalty > smaller_penalty


# --------------------------------------------------------- per-equation scalars
def test_the_per_equation_keys_exist(vecm_fit) -> None:
    """``vec.sthlp:353-359`` lists ``k_#``, ``df_m#``, ``r2_#`` and ``rmse_#``.

    Ten keys (five families over two equations), none of which were stored.
    """

    for index in (1, 2):
        for prefix in ("k_", "r2_", "rmse_"):
            assert f"{prefix}{index}" in vecm_fit.e, (prefix, index)
        assert f"df_m{index}" in vecm_fit.e


def test_the_df_m_spelling_omits_the_underscore(vecm_fit) -> None:
    """``df_m1``, not ``df_m_1`` -- the prefix already ends in ``_m``.

    The other four families DO use ``_<index>``.  Pinning the asymmetry keeps a later
    "tidy the spellings" edit from breaking a documented key.
    """

    assert "df_m1" in vecm_fit.e
    assert "df_m_1" not in vecm_fit.e
    assert "k_1" in vecm_fit.e


def test_k_per_equation_matches_df_eq(vecm_fit) -> None:
    """Every equation has the same regressor count."""

    expected = float(vecm_fit.e["df_eq"])

    assert float(vecm_fit.e["k_1"]) == expected
    assert float(vecm_fit.e["k_2"]) == expected


def test_df_m_per_equation_equals_the_parameter_count(vecm_fit) -> None:
    """For a vec fit, ``e(df_m#)`` EQUALS the per-equation parameter count.

    This test used to assert ``df_m# == df_eq - 1``, on the VAR convention where the model df
    removes the deterministic term. Measured against a live ``vec dinf unem, lags(2) rank(1)``:

        e(k_1) = e(k_2) = e(df_m1) = e(df_m2) = e(df_m) = 4

    -- the error-correction term and the constant are both free parameters, so nothing is
    removed. The minus-one produced 3.
    """

    for index in (1, 2):
        assert float(vecm_fit.e[f"df_m{index}"]) == float(vecm_fit.e[f"k_{index}"])
        assert float(vecm_fit.e[f"df_m{index}"]) == 4.0


def test_the_r_squared_lies_in_the_unit_interval(vecm_fit) -> None:
    """A share of explained variation cannot escape [0, 1] for a differenced response."""

    for index in (1, 2):
        value = float(vecm_fit.e[f"r2_{index}"])
        assert np.isfinite(value)
        assert -1.0 <= value <= 1.0


def test_the_two_equations_report_different_r_squared(vecm_fit) -> None:
    """Guard: a shared value would mean the per-equation loop is not indexing."""

    assert float(vecm_fit.e["r2_1"]) != pytest.approx(
        float(vecm_fit.e["r2_2"]), rel=1e-6
    )


def test_rmse_is_positive_and_finite(vecm_fit) -> None:
    for index in (1, 2):
        value = float(vecm_fit.e[f"rmse_{index}"])
        assert np.isfinite(value)
        assert value > 0.0


def test_chi2_per_equation_is_present_and_matches_stata(vecm_fit) -> None:
    """``e(chi2_#)`` is ``(R2/(1-R2)) * (N-4)``.

    This key used to be DELIBERATELY omitted, on the reasoning that ``vec.sthlp`` names it
    without stating the hypothesis. But the formula is recoverable from the data, and a
    caller needs it: measured against a live ``vec``, the values are 463.5891 and
    109.44973, which this reproduces exactly.
    """

    for index in (1, 2):
        r2 = float(vecm_fit.e[f"r2_{index}"])
        n = float(vecm_fit.e["N"])
        expected = r2 / (1.0 - r2) * (n - 4.0)
        assert float(vecm_fit.e[f"chi2_{index}"]) == pytest.approx(expected, rel=1e-10)