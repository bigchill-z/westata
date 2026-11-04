"""Tests for Stata's ``estat ic`` (AIC, BIC, AICc, CAIC).

Formulas transcribed from ``estimates.ado:985-1002``, which is the implementation
behind ``estat ic``:

    AIC  = -2*ll + 2*df
    BIC  = -2*ll + log(N)*df
    AICc = -2*ll + 2*df*(N/(N-df-1))
    CAIC = -2*ll + df*(log(N)+1)

``df`` defaults to ``rank(V)`` (``estimates.ado:977-978``), and ``N`` is
``e(N_ic)`` when recorded, else ``e(N)``.

One thing this file deliberately does **not** assert: that ``estat_ic().aic``
equals ``result.e["aic"]``.  They are different quantities -- ``e(aic)`` uses the
Gaussian deviance ``N log(rss/N) + 2k`` while ``estat ic`` uses ``-2 e(ll) + 2 df``
-- and ``estimates.ado:953`` reads only ``e(ll)``/``e(ll_0)``/``e(N)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.postestimation.ic import estat_ic


@pytest.fixture()
def linear_fit():
    df = pd.read_stata("tests/data/auto.dta")
    return ws.OLS.from_formula("mpg ~ weight + length", df).fit()


@pytest.fixture()
def logit_fit():
    df = pd.read_stata("tests/data/auto.dta")
    frame = df.assign(dom=(df["foreign"] == "Foreign").astype(int))
    return ws.Logit.from_formula("dom ~ weight + mpg", frame).fit()


def test_the_four_formulas_match_estimates_ado(linear_fit) -> None:
    result = linear_fit
    ll = float(result.e["ll"])
    n = float(result.e["N"])
    df = ic_df(result)

    ic = result.estat_ic()

    assert ic.aic == pytest.approx(-2 * ll + 2 * df, rel=1e-14)
    assert ic.bic == pytest.approx(-2 * ll + np.log(n) * df, rel=1e-14)
    assert ic.aicc == pytest.approx(
        -2 * ll + 2 * df * (n / (n - df - 1)), rel=1e-14
    )
    assert ic.caic == pytest.approx(-2 * ll + df * (np.log(n) + 1), rel=1e-14)


def ic_df(result) -> float:
    """The df Stata would use: rank of the coefficient covariance."""

    return float(np.linalg.matrix_rank(np.asarray(result.cov_params())))


def test_aicc_equals_the_textbook_form(linear_fit) -> None:
    """The ado's ``2df*N/(N-df-1)`` equals ``AIC + 2df(df+1)/(N-df-1)``."""

    result = linear_fit
    ic = result.estat_ic()
    n = ic.n
    df = ic.df

    textbook = ic.aic + 2 * df * (df + 1) / (n - df - 1)

    assert ic.aicc == pytest.approx(textbook, rel=1e-12)


def test_caic_exceeds_bic_by_df(linear_fit) -> None:
    """Algebraic consequence of the two formulas; a useful cross-check."""

    ic = linear_fit.estat_ic()

    assert ic.caic - ic.bic == pytest.approx(ic.df)


def test_df_defaults_to_the_rank_of_the_covariance(linear_fit) -> None:
    result = linear_fit
    expected = np.linalg.matrix_rank(np.asarray(result.cov_params()))

    assert result.estat_ic().df == expected


def test_n_defaults_to_e_n(linear_fit) -> None:
    result = linear_fit

    assert result.estat_ic().n == float(result.e["N"])


def test_ll_null_and_ll_model_are_reported(linear_fit) -> None:
    result = linear_fit
    ic = result.estat_ic()

    assert ic.ll_model == pytest.approx(float(result.e["ll"]))
    assert ic.ll_null == pytest.approx(float(result.e["ll_0"]))


def test_df_and_n_overrides(linear_fit) -> None:
    result = linear_fit
    ll = float(result.e["ll"])

    ic = result.estat_ic(df=2, n=100)

    assert ic.df == 2
    assert ic.n == 100
    assert ic.aic == pytest.approx(-2 * ll + 4)
    assert ic.bic == pytest.approx(-2 * ll + np.log(100) * 2)


def test_r_S_column_order(logit_fit) -> None:
    """``r(S)`` is ``N ll0 ll df AIC BIC AICc CAIC`` (estimates.ado:1008)."""

    payload = logit_fit.estat_ic().to_dict()

    assert list(payload) == ["N", "ll_0", "ll", "df", "aic", "bic", "aicc", "caic"]


def test_works_for_a_likelihood_model(logit_fit) -> None:
    ic = logit_fit.estat_ic()

    assert np.isfinite(ic.aic)
    assert np.isfinite(ic.aicc)
    assert np.isfinite(ic.caic)
    assert ic.aic == pytest.approx(-2 * float(logit_fit.e["ll"]) + 2 * ic.df)


def test_aicc_is_nan_when_df_too_large() -> None:
    """``N - df - 1 <= 0`` leaves AICc undefined rather than inventing a value."""

    df = pd.read_stata("tests/data/auto.dta")
    fit = ws.OLS.from_formula("mpg ~ weight", df).fit()
    ic = fit.estat_ic(n=3, df=3)  # N - df - 1 == -1

    assert np.isnan(ic.aicc)
    assert np.isfinite(ic.aic)


def test_a_result_without_a_likelihood_is_refused() -> None:
    class _NoLL:
        e = {}

    with pytest.raises(ValueError, match="log-likelihood"):
        estat_ic(_NoLL())


def test_there_is_no_aic_after_a_non_likelihood_result() -> None:
    """``xtunitroot``-style containers carry no ``e(ll)``."""

    import westata as _ws

    rng = np.random.default_rng(2)
    series = pd.DataFrame({"y": np.cumsum(rng.normal(size=120))})
    unit_root = _ws.unitroot(series["y"].to_numpy(), test="adf")
    assert not hasattr(unit_root, "e")

    with pytest.raises(ValueError, match="log-likelihood"):
        estat_ic(unit_root)


def test_output_has_the_two_criteria_columns_stata_prints(linear_fit) -> None:
    """``estat ic`` prints AIC and BIC; AICc and CAIC are not in the table.

    Measured against a live run: the table has seven columns (Model, N, ll(null), ll(model),
    df, AIC, BIC). Printing the other two made the layout differ from Stata's even though both
    numbers are correct -- they remain available as attributes.
    """

    text = str(linear_fit.estat_ic())

    for label in ("AIC", "BIC"):
        assert label in text
    for absent in ("AICc", "CAIC"):
        assert absent not in text
    assert "ll(null)" in text
    assert "ll(model)" in text


def test_to_frame_has_statas_columns(linear_fit) -> None:
    frame = linear_fit.estat_ic().to_frame()

    assert list(frame.columns) == [
        "N",
        "ll(null)",
        "ll(model)",
        "df",
        "AIC",
        "BIC",
        "AICc",
        "CAIC",
    ]


def test_module_function_matches_the_method(linear_fit) -> None:
    assert estat_ic(linear_fit).to_dict() == linear_fit.estat_ic().to_dict()


def test_estat_ic_aic_is_not_the_stored_e_aic(linear_fit) -> None:
    """They are different definitions, and that is intentional.

    ``e(aic)`` is a westata addition using the Gaussian deviance
    ``N log(rss/N) + 2k``; ``estat ic`` follows ``estimates.ado`` and builds
    ``-2 e(ll) + 2 df``.  ``e(ll)`` carries the ``N(1 + log 2pi)`` constant, so
    the two differ by exactly that constant.
    """

    ic = linear_fit.estat_ic()
    n = ic.n
    constant = n * (1.0 + np.log(2.0 * np.pi))

    assert not np.isclose(ic.aic, linear_fit.e["aic"])
    assert ic.aic - linear_fit.e["aic"] == pytest.approx(constant, rel=1e-10)
