"""``estat szroeter`` -- Szroeter's rank test for heteroskedasticity.

``regress_postestimation.sthlp:519-521``:

    estat szroeter performs Szroeter's rank test for heteroskedasticity for each of the
    variables in varlist or for the explanatory variables of the regression if rhs is
    specified.

The statistic is fully specified by ``szroeter.ado:158-178``, whose ``Szroeter``
subroutine is readable source rather than a compiled builtin:

    egen `ht' = rank(`var')
    scalar `mh' = r(mean)          // mean rank
    scalar `vh' = r(Var)           // SAMPLE variance of the ranks
    predict `res', res
    summ `ht' [aw=`res'^2]
    scalar `h' = r(mean)           // residual-squared-weighted mean rank
    Q = (`n'^2 / (2*(`n'-1))) * (`h'-`mh')^2 / `vh'

So the test ranks the variable and asks whether the squared residuals move with that
rank. Ranks rather than levels are used because the alternative is a variance MONOTONE in
the variable, not linear in it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

import westata as ws


def _frame(*, heteroskedastic: bool, n: int = 200, seed: int = 2900):
    """A regressor whose error variance grows with it, or one where it does not."""

    rng = np.random.default_rng(seed)
    covariate = rng.uniform(1.0, 10.0, size=n)
    scale = covariate if heteroskedastic else np.ones(n)
    response = 1.0 + 0.5 * covariate + rng.normal(size=n) * scale
    return pd.DataFrame(
        {
            "y": response,
            "x1": covariate,
            "x2": rng.normal(size=n),
        }
    )


@pytest.fixture(scope="module")
def homoskedastic():
    frame = _frame(heteroskedastic=False)
    return frame, ws.OLS.from_formula("y ~ x1 + x2", frame).fit()


@pytest.fixture(scope="module")
def heteroskedastic():
    frame = _frame(heteroskedastic=True)
    return frame, ws.OLS.from_formula("y ~ x1 + x2", frame).fit()


def _manual_statistic(frame: pd.DataFrame, residuals: np.ndarray, name: str):
    """Rebuild ``Q`` straight from the ado's formula."""

    ranks = scipy_stats.rankdata(np.asarray(frame[name], dtype=float))
    count = ranks.size
    mean_rank = float(np.mean(ranks))
    variance_rank = float(np.var(ranks, ddof=1))
    weights = np.asarray(residuals, dtype=float) ** 2
    weighted_mean = float(np.sum(weights * ranks) / np.sum(weights))
    return (
        (count ** 2 / (2.0 * (count - 1)))
        * (weighted_mean - mean_rank) ** 2
        / variance_rank
    )


# ------------------------------------------------------------------ the formula
def test_the_single_variable_statistic_matches_the_ado_formula(homoskedastic) -> None:
    frame, result = homoskedastic
    expected = _manual_statistic(frame, np.asarray(result.resid), "x1")

    reported = result.estat_szroeter(variables=["x1"])

    assert reported.statistic == pytest.approx(expected, rel=1e-12)


def test_the_statistic_is_chi_square_with_one_degree_of_freedom(homoskedastic) -> None:
    reported = homoskedastic[1].estat_szroeter(variables=["x1"])

    assert reported.to_dict()["df"] == 1.0
    assert reported.pvalue == pytest.approx(
        float(scipy_stats.chi2.sf(reported.statistic, 1)), rel=1e-12
    )


def test_the_rhs_option_tests_every_slope(homoskedastic) -> None:
    """``sthlp:528-530``: the explanatory variables of the fitted regression."""

    table = homoskedastic[1].estat_szroeter(rhs=True)

    assert [row.variable for row in table] == ["x1", "x2"]


def test_the_constant_is_never_tested(homoskedastic) -> None:
    """A rank test on a constant is undefined -- there are no distinct ranks."""

    table = homoskedastic[1].estat_szroeter(rhs=True)

    assert "_cons" not in [row.variable for row in table]


# ------------------------------------------------------------------- behaviour
def test_monotone_variance_is_detected(heteroskedastic) -> None:
    """Data built with variance growing in ``x1`` must be rejected for ``x1``."""

    reported = heteroskedastic[1].estat_szroeter(variables=["x1"])

    assert reported.pvalue < 0.01


def test_constant_variance_is_not_rejected(homoskedastic) -> None:
    """The other direction, on data built without heteroskedasticity."""

    reported = homoskedastic[1].estat_szroeter(variables=["x1"])

    assert reported.pvalue > 0.05


def test_the_two_panels_differ(heteroskedastic, homoskedastic) -> None:
    """Guard: if both behaved alike the direction tests are vacuous."""

    loud = heteroskedastic[1].estat_szroeter(variables=["x1"]).pvalue
    quiet = homoskedastic[1].estat_szroeter(variables=["x1"]).pvalue

    assert loud < 0.01 < 0.05 < quiet


def test_the_untreated_regressor_is_not_flagged(heteroskedastic) -> None:
    """``x2`` has constant error variance in this fixture, so it must not reject.

    This is what separates "the test detects the right variable" from "the test rejects
    everything on skewed data".
    """

    table = heteroskedastic[1].estat_szroeter(rhs=True)
    by_name = {row.variable: row for row in table}

    assert by_name["x1"].pvalue < 0.01
    assert by_name["x2"].pvalue > 0.05


# ---------------------------------------------------------- multiple testing
def test_the_adjustment_methods_are_ordered(heteroskedastic) -> None:
    """Bonferroni is the most conservative, Holm lies between, noadjust the least.

    ``sthlp:537-540`` names all three plus ``sidak``.
    """

    plain = heteroskedastic[1].estat_szroeter(rhs=True, mtest="noadjust")
    bonferroni = heteroskedastic[1].estat_szroeter(rhs=True, mtest="bonferroni")
    holm = heteroskedastic[1].estat_szroeter(rhs=True, mtest="holm")

    plain_p = plain.to_frame()["p"].to_numpy()
    bonferroni_p = bonferroni.to_frame()["p_adjusted"].to_numpy()
    holm_p = holm.to_frame()["p_adjusted"].to_numpy()

    assert np.all(bonferroni_p >= plain_p - 1e-12)
    assert np.all(bonferroni_p >= holm_p - 1e-12)


def test_bonferroni_multiplies_by_the_count(heteroskedastic) -> None:
    plain = heteroskedastic[1].estat_szroeter(rhs=True, mtest="noadjust")
    bonferroni = heteroskedastic[1].estat_szroeter(rhs=True, mtest="bonferroni")

    count = len(plain)
    expected = np.minimum(plain.to_frame()["p"].to_numpy() * count, 1.0)

    assert np.allclose(
        bonferroni.to_frame()["p_adjusted"].to_numpy(), expected, rtol=1e-12
    )


def test_sidak_never_exceeds_one(heteroskedastic) -> None:
    """``1 - (1 - p)^m`` clips at 1; an adjusted p-value above 1 would be nonsense."""

    table = heteroskedastic[1].estat_szroeter(rhs=True, mtest="sidak")

    assert np.all(table.to_frame()["p_adjusted"].to_numpy() <= 1.0)


def test_holm_is_step_down(heteroskedastic) -> None:
    """Holm's adjusted values must be non-decreasing in the sorted p order."""

    table = heteroskedastic[1].estat_szroeter(rhs=True, mtest="holm")
    adjusted = table.to_frame()["p_adjusted"].to_numpy()

    assert np.all(np.diff(np.sort(adjusted)) >= -1e-12)


def test_a_missing_adjustment_is_accepted_as_no_adjustment(homoskedastic) -> None:
    """``mtest()`` absent means "always multiple testing, no adjustment" (``sthlp:543``)."""

    default = homoskedastic[1].estat_szroeter(rhs=True)
    explicit = homoskedastic[1].estat_szroeter(rhs=True, mtest="noadjust")

    assert default.to_frame()["p"].tolist() == explicit.to_frame()["p"].tolist()


def test_an_unknown_adjustment_is_refused(homoskedastic) -> None:
    with pytest.raises(ValueError, match="mtest"):
        homoskedastic[1].estat_szroeter(rhs=True, mtest="not_a_method")


# ------------------------------------------------------------------- refusals
def test_neither_variables_nor_rhs_is_refused(homoskedastic) -> None:
    """``sthlp:509-510``: "Either varlist or rhs must be specified"."""

    with pytest.raises(ValueError, match="varlist or option rhs"):
        homoskedastic[1].estat_szroeter()


def test_a_weighted_fit_is_refused() -> None:
    """``szroeter.ado:17-19``: "szroeter does not allow weights"."""

    frame = _frame(heteroskedastic=False, n=120)
    weights = np.ones(len(frame))
    fitted = ws.OLS.from_formula(
        "y ~ x1 + x2", frame, weights=weights, weight_type="fweight"
    ).fit()

    with pytest.raises(NotImplementedError, match="weights"):
        fitted.estat_szroeter(variables=["x1"])


def test_a_clustered_fit_is_refused() -> None:
    """``szroeter.ado:20-22``: "szroeter does not allow cluster()"."""

    frame = _frame(heteroskedastic=False, n=200)
    groups = pd.Series(np.arange(len(frame)) % 20, name="g")
    fitted = ws.OLS.from_formula("y ~ x1 + x2", frame).fit(
        cov_type="cluster", cov_kwds={"groups": groups}
    )

    with pytest.raises(NotImplementedError, match="cluster"):
        fitted.estat_szroeter(variables=["x1"])


def test_a_non_regress_fit_is_refused() -> None:
    """``szroeter.ado:13-16``: "szroeter is only possible after regress"."""

    rng = np.random.default_rng(2902)
    frame = pd.DataFrame(
        {
            "yb": (rng.normal(size=150) > 0).astype(int),
            "x1": rng.normal(size=150),
        }
    )
    logit = ws.Logit.from_formula("yb ~ x1", frame).fit()

    with pytest.raises(NotImplementedError, match="regress"):
        logit.estat_szroeter(variables=["x1"])


def test_an_unknown_variable_names_itself(homoskedastic) -> None:
    with pytest.raises(ValueError) as caught:
        homoskedastic[1].estat_szroeter(variables=["not_a_column"])

    assert "not_a_column" in str(caught.value)