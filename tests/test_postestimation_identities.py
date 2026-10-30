"""Postestimation output must agree with the stored ``e()`` scalars.

Rounds 95-96 established that two stored numbers can disagree while each looks
plausible. The same risk applies between a postestimation command and ``e()``: a
table can be rendered from one source while ``e()`` holds another, and nothing
fails.

Each test here computes the quantity a SECOND way -- from ``e()``, or from the raw
data -- and compares. The comparisons are algebraic, so they need no fixture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(103)
    n = 300
    x1 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    x2 = x1 * 0.9 + rng.normal(size=n) * 0.1  # strongly collinear with x1
    y = 1.0 + 0.8 * x1 - 0.3 * x3 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})


def test_estat_ic_uses_minus_two_ll_plus_two_k(frame) -> None:
    """``estimates.ado:985-1002``: ``AIC = -2*e(ll) + 2*e(df)``.

    ``e(aic)`` is a DIFFERENT number -- it is ``N*log(rss/N) + 2k`` from the
    Gaussian deviance and omits the ``N(1 + log 2pi)`` constant the log-likelihood
    carries.  Both are internally consistent, so this test pins ``estat ic``
    against ``e(ll)`` rather than against ``e(aic)``.
    """

    result = ws.OLS.from_formula("y ~ x1 + x3", frame).fit()
    e = result.e
    ic = result.estat_ic()

    assert ic.aic == pytest.approx(-2.0 * e["ll"] + 2.0 * e["rank"], rel=1e-10)


def test_estat_ic_bic_uses_log_n(frame) -> None:
    """``BIC = -2*ll + log(N)*k``."""

    result = ws.OLS.from_formula("y ~ x1 + x3", frame).fit()
    e = result.e
    ic = result.estat_ic()

    expected = -2.0 * e["ll"] + np.log(e["N"]) * e["rank"]

    assert ic.bic == pytest.approx(expected, rel=1e-10)


def test_e_aic_differs_from_estat_ic_by_the_known_constant(frame) -> None:
    """The gap between the two AICs is exactly ``N(1 + log 2pi)``.

    Recording the SIZE of the difference is what makes it a documented convention
    rather than a suspicious mismatch.
    """

    result = ws.OLS.from_formula("y ~ x1 + x3", frame).fit()
    e = result.e
    gap = result.estat_ic().aic - e["aic"]

    assert gap == pytest.approx(e["N"] * (1.0 + np.log(2.0 * np.pi)), rel=1e-10)


def test_estat_summarize_means_equal_the_raw_means(frame) -> None:
    """The descriptive table must describe the same data the model used."""

    result = ws.OLS.from_formula("y ~ x1 + x3", frame).fit()
    table = result.estat_summarize().table

    for name in ("y", "x1", "x3"):
        assert table.loc[name, "Mean"] == pytest.approx(
            float(frame[name].mean()), rel=1e-12
        ), name


def test_estat_summarize_sd_is_the_sample_sd(frame) -> None:
    """Unweighted descriptive SD is the ``ddof=1`` standard deviation."""

    result = ws.OLS.from_formula("y ~ x1 + x3", frame).fit()
    table = result.estat_summarize().table

    for name in ("y", "x1"):
        assert table.loc[name, "Std. Dev."] == pytest.approx(
            float(frame[name].std(ddof=1)), rel=1e-12
        ), name


def test_vif_tolerance_is_the_reciprocal(frame) -> None:
    """``1/VIF`` IS the tolerance column -- two renderings of one number."""

    result = ws.OLS.from_formula("y ~ x1 + x2 + x3", frame).fit()
    vif = result.estat_vif()

    for name, value, tolerance in zip(vif.names, vif.vif, vif.tolerance):
        assert tolerance == pytest.approx(1.0 / value, rel=1e-12), name


def test_vif_mean_is_the_mean_of_the_column(frame) -> None:
    """``Mean VIF`` in the footer must be the column mean."""

    result = ws.OLS.from_formula("y ~ x1 + x2 + x3", frame).fit()
    vif = result.estat_vif()

    assert vif.mean_vif == pytest.approx(float(np.mean(vif.vif)), rel=1e-12)


def test_a_single_regressor_has_vif_exactly_one(frame) -> None:
    """With one regressor there is nothing to be collinear with.

    A value away from 1 would mean the auxiliary regression was built from the
    wrong matrix.
    """

    result = ws.OLS.from_formula("y ~ x1", frame).fit()

    assert result.estat_vif().vif[0] == pytest.approx(1.0, rel=1e-12)


def test_collinear_regressors_show_large_vif(frame) -> None:
    """``x1`` and ``x2`` correlate at 0.9, so their VIFs must be large.

    A sanity check in the other direction: if the auxiliary regression silently
    used the wrong columns, the VIFs would come out near 1.
    """

    result = ws.OLS.from_formula("y ~ x1 + x2 + x3", frame).fit()
    vif = result.estat_vif()
    mapping = dict(zip(vif.names, vif.vif))

    assert mapping["x1"] > 20.0
    assert mapping["x2"] > 20.0
    assert mapping["x3"] < 2.0


def test_estat_vce_diagonal_is_the_squared_standard_errors(frame) -> None:
    """``estat vce`` renders ``V``; its diagonal is ``bse**2``.

    The printed matrix is permuted into Stata's row order (``_cons`` last), so the
    comparison is made by NAME rather than by position -- otherwise a correct
    reordering looks like a corrupted matrix.
    """

    result = ws.OLS.from_formula("y ~ x1 + x3", frame).fit()
    matrix = result.estat_vce()
    covariance = np.asarray(matrix.to_frame(), dtype=float)

    # Map each displayed label back to its position in the stored parameter vector.
    stored = [str(name) for name in result.param_names]
    lookup: dict[str, int] = {}
    for index, name in enumerate(stored):
        key = "_cons" if name in ("Intercept", "intercept") else (
            name[1:] if name.startswith("/") else name
        )
        lookup[key.replace(":", "#")] = index
    positions = [lookup[name] for name in matrix.names]

    bse = np.asarray(result.bse, dtype=float)[positions]
    base = np.asarray(result.cov_params(), dtype=float)[np.ix_(positions, positions)]

    np.testing.assert_allclose(covariance, base, rtol=1e-12)
    np.testing.assert_allclose(np.diag(covariance), bse**2, rtol=1e-12)