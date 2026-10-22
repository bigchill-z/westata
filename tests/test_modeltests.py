"""Tests for ``lrtest`` and ``hausman`` (Stata ``[R] lrtest``, ``[R] hausman``).

Every numerical assertion is computed here from the two models' own
log-likelihoods / covariance matrices, never by reusing a helper the
implementation also calls, so the formulas are pinned independently:

    LR = 2 (lnL_u - lnL_r)              ~ chi2(k_u - k_r)
    H  = (b_c - b_e)' (V_c - V_e)^-1 (b_c - b_e)  ~ chi2(k)

Two pieces of Stata behaviour are deliberately encoded:

* ``hausman`` compares the **slope** coefficients only.  Stata's default
  excludes the constant because a within-transformed intercept and a GLS
  intercept are not the same parameter; including it also makes
  ``V_c - V_e`` indefinite on ordinary panels (asserted below), so the
  slopes-only comparison is the only well-posed one.
* ``lrtest`` on two identical models has ``df = 0``.  ``chi2(0)`` is a
  degenerate law, so the p-value is defined as ``1.0`` by convention
  rather than taken from ``scipy.stats.chi2.sf`` (which returns ``nan``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import westata as ws
from westata.errors import ConsistencyError
from westata.postestimation.modeltests import HausmanResult, LRTestResult, hausman, lrtest


# --------------------------------------------------------------------- fixtures
def _frame(n: int = 200, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    y = 1.0 + 2.0 * x1 - 1.0 * x2 + 0.5 * x3 + rng.normal(scale=0.6, size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})


def _panel(n_entities: int = 40, n_periods: int = 6, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    entity = np.repeat(np.arange(n_entities), n_periods)
    time = np.tile(np.arange(1, n_periods + 1), n_entities)
    n = n_entities * n_periods
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    effect = rng.normal(size=n_entities)[entity]
    y = 1.0 + 1.5 * x1 - 0.5 * x2 + effect + rng.normal(scale=0.4, size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "firm": entity, "year": time})


def _fe(frame: pd.DataFrame, formula: str = "y ~ x1 + x2"):
    return ws.PanelOLS.from_formula(formula, frame, entity="firm", time="year").fit()


def _re(frame: pd.DataFrame, formula: str = "y ~ x1 + x2"):
    return ws.RandomEffects.from_formula(formula, frame, entity="firm", time="year").fit()


class _StubResult:
    """Minimal stand-in for a result object, to exercise the guards."""

    def __init__(self, *, param_names, nobs, llf, params=None, cov=None):
        self.param_names = list(param_names)
        self.nobs = nobs
        self.llf = llf
        self.params = pd.Series(np.zeros(len(self.param_names)), index=self.param_names)
        self._cov = np.eye(len(self.param_names)) if cov is None else cov

    def cov_params(self):
        return self._cov


# ===================================================================== lrtest
def test_lrtest_matches_hand_computation_and_chi2():
    """``y ~ x1`` vs ``y ~ x1 + x2``: LR == 2(ll_u - ll_r), df == 1."""

    frame = _frame()
    restricted = ws.OLS.from_formula("y ~ x1", frame).fit()
    unrestricted = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()

    outcome = unrestricted.lrtest(restricted)

    expected = 2.0 * (unrestricted.llf - restricted.llf)
    assert outcome.statistic == pytest.approx(expected, rel=1e-12)
    assert outcome.df == 1
    assert outcome.pvalue == pytest.approx(stats.chi2.sf(expected, 1), rel=1e-12)
    assert outcome.ll_unrestricted == pytest.approx(unrestricted.llf, rel=1e-12)
    assert outcome.ll_restricted == pytest.approx(restricted.llf, rel=1e-12)
    # x2 has a real effect, so the restricted model is rejected.
    assert outcome.pvalue < 1e-6


def test_lrtest_uses_the_parameter_count_difference_as_df():
    frame = _frame()
    restricted = ws.OLS.from_formula("y ~ x1", frame).fit()
    unrestricted = ws.OLS.from_formula("y ~ x1 + x2 + x3", frame).fit()

    outcome = unrestricted.lrtest(restricted)
    expected = 2.0 * (unrestricted.llf - restricted.llf)

    assert outcome.df == 2 == len(unrestricted.param_names) - len(restricted.param_names)
    assert outcome.statistic == pytest.approx(expected, rel=1e-12)
    assert outcome.pvalue == pytest.approx(stats.chi2.sf(expected, 2), rel=1e-12)


def test_lrtest_matches_statsmodels_loglikelihood():
    """The LR statistic is anchored to an independent MLE log-likelihood."""

    import statsmodels.api as sm

    frame = _frame()
    restricted = ws.OLS.from_formula("y ~ x1", frame).fit()
    unrestricted = ws.OLS.from_formula("y ~ x1 + x2 + x3", frame).fit()

    big = sm.OLS(frame["y"], sm.add_constant(frame[["x1", "x2", "x3"]])).fit()
    small = sm.OLS(frame["y"], sm.add_constant(frame[["x1"]])).fit()

    outcome = unrestricted.lrtest(restricted)
    assert outcome.statistic == pytest.approx(2.0 * (big.llf - small.llf), rel=1e-10)


def test_lrtest_identical_models_is_degenerate_with_pvalue_one():
    """``df == 0``: the statistic is 0 and the p-value is 1 by convention."""

    frame = _frame()
    first = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()
    second = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()

    outcome = first.lrtest(second)

    assert outcome.df == 0
    assert outcome.statistic == pytest.approx(0.0, abs=1e-9)
    assert outcome.pvalue == 1.0
    # scipy alone would give nan here, which is why the convention exists.
    assert np.isnan(stats.chi2.sf(0.0, 0))


def test_lrtest_rejects_non_nested_models():
    frame = _frame()
    left = ws.OLS.from_formula("y ~ x1", frame).fit()
    right = ws.OLS.from_formula("y ~ x2", frame).fit()

    with pytest.raises(ConsistencyError, match="not nested"):
        left.lrtest(right)


def test_lrtest_names_the_parameters_that_break_nesting():
    frame = _frame()
    small = ws.OLS.from_formula("y ~ x1", frame).fit()
    other = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()

    # ``other`` carries x2, which ``small`` does not: not nested in this direction.
    with pytest.raises(ConsistencyError, match="x2"):
        small.lrtest(other)


def test_lrtest_rejects_different_sample_sizes():
    frame = _frame(n=300)
    full = ws.OLS.from_formula("y ~ x1 + x2 + x3", frame).fit()
    trimmed = ws.OLS.from_formula("y ~ x1", frame.iloc[:200]).fit()

    assert full.nobs != trimmed.nobs
    with pytest.raises(ConsistencyError, match="different samples"):
        full.lrtest(trimmed)


def test_lrtest_warns_and_still_reports_a_negative_statistic():
    """The unrestricted model fitting worse is a warning, not a silent number.

    Constructed from two *different* responses of equal length with nested
    regressor names: the restricted model is fitted to a much cleaner response,
    so its log-likelihood is higher and the LR statistic goes negative.
    """

    rng = np.random.default_rng(99)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)

    clean = pd.DataFrame({"y": 1.0 + x1 + rng.normal(scale=0.05, size=n), "x1": x1, "x2": x2})
    noisy = pd.DataFrame({"y": 1.0 + x1 + rng.normal(scale=5.0, size=n), "x1": x1, "x2": x2})

    restricted = ws.OLS.from_formula("y ~ x1", clean).fit()
    unrestricted = ws.OLS.from_formula("y ~ x1 + x2", noisy).fit()

    assert unrestricted.llf < restricted.llf
    with pytest.warns(RuntimeWarning, match="lower log-likelihood"):
        outcome = unrestricted.lrtest(restricted)

    # Still computed, and reported as the (negative) number it is.
    assert outcome.statistic < 0.0
    assert outcome.statistic == pytest.approx(2.0 * (unrestricted.llf - restricted.llf), rel=1e-12)
    assert outcome.pvalue == 1.0  # chi2.sf of a negative statistic


def test_lrtest_rejects_a_model_without_a_log_likelihood():
    frame = _frame()
    fitted = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()
    missing = _StubResult(param_names=["Intercept", "x1"], nobs=fitted.nobs, llf=float("nan"))

    with pytest.raises(ConsistencyError, match="log-likelihood"):
        fitted.lrtest(missing)


def test_lrtest_is_also_available_as_a_free_function():
    frame = _frame()
    restricted = ws.OLS.from_formula("y ~ x1", frame).fit()
    unrestricted = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()

    via_method = unrestricted.lrtest(restricted)
    via_function = lrtest(unrestricted, restricted)

    assert isinstance(via_function, LRTestResult)
    assert via_function.statistic == pytest.approx(via_method.statistic, rel=0.0)
    assert via_function.df == via_method.df
    assert via_function.pvalue == via_method.pvalue


def test_lrtest_result_is_a_frozen_dataclass():
    frame = _frame()
    outcome = ws.OLS.from_formula("y ~ x1 + x2", frame).fit().lrtest(
        ws.OLS.from_formula("y ~ x1", frame).fit()
    )

    assert isinstance(outcome, LRTestResult)
    with pytest.raises(Exception):  # frozen dataclass
        outcome.statistic = 0.0  # type: ignore[misc]


# ==================================================================== hausman
def test_hausman_fe_vs_re_is_finite_with_a_valid_pvalue():
    frame = _panel()
    fe = _fe(frame)
    re = _re(frame)

    outcome = fe.hausman(re)

    assert np.isfinite(outcome.statistic)
    assert outcome.statistic >= 0.0
    assert 0.0 <= outcome.pvalue <= 1.0
    # Stata's hausman compares slopes: Intercept + x1 + x2 share names, but the
    # constant is not comparable across a within and a GLS parameterisation.
    assert outcome.df == 2
    assert outcome.excluded == []


def test_hausman_statistic_matches_the_hand_formula():
    frame = _panel()
    fe = _fe(frame)
    re = _re(frame)

    outcome = fe.hausman(re)

    names = ["x1", "x2"]
    fe_index = [fe.param_names.index(name) for name in names]
    re_index = [re.param_names.index(name) for name in names]
    difference = fe.params.to_numpy()[fe_index] - re.params.to_numpy()[re_index]
    difference_cov = (
        fe.cov_params()[np.ix_(fe_index, fe_index)]
        - re.cov_params()[np.ix_(re_index, re_index)]
    )
    expected = float(difference @ np.linalg.solve(difference_cov, difference))

    assert outcome.statistic == pytest.approx(expected, rel=1e-10)
    assert outcome.pvalue == pytest.approx(stats.chi2.sf(expected, 2), rel=1e-10)
    assert np.all(np.linalg.eigvalsh(difference_cov) > 0)


def test_hausman_excludes_the_constant_because_it_breaks_from_positive_definiteness():
    """Including ``Intercept`` makes ``V_c - V_e`` indefinite on a plain panel.

    That is *why* Stata's default compares slopes only: the two intercepts are
    different parameters (within-transformed vs GLS), so their covariance
    difference carries no information and destroys the quadratic form.
    """

    frame = _panel()
    fe = _fe(frame)
    re = _re(frame)

    all_index_fe = range(len(fe.param_names))
    all_index_re = range(len(re.param_names))
    full_difference_cov = (
        fe.cov_params()[np.ix_(all_index_fe, all_index_fe)]
        - re.cov_params()[np.ix_(all_index_re, all_index_re)]
    )
    assert not np.all(np.linalg.eigvalsh(full_difference_cov) > 0)

    # The slopes-only difference, which hausman() actually uses, is positive
    # definite and needs no generalised inverse.
    outcome = fe.hausman(re)
    assert outcome.df == 2


def test_hausman_reports_a_parameter_missing_from_one_model():
    frame = _panel()
    fe = _fe(frame, "y ~ x1 + x2")
    re = _re(frame, "y ~ x1")

    # With only x1 comparable the difference V_fe - V_re happens to be indefinite
    # on this panel (seed=21), so the documented generalised-inverse path is
    # taken; the point of this test is the ``excluded`` bookkeeping.
    with pytest.warns(RuntimeWarning, match="not positive definite"):
        outcome = fe.hausman(re)

    assert "x2" in outcome.excluded
    assert "Intercept" not in outcome.excluded  # dropped by design, not "missing"
    assert outcome.df == 1  # only x1 is comparable
    assert np.isfinite(outcome.statistic)


def test_hausman_of_disjoint_regressors_reports_both_sides_but_cannot_compare():
    """``excluded`` names the gap on both sides; with no slope left it refuses."""

    frame = _panel()
    fe = _fe(frame, "y ~ x1")
    re = _re(frame, "y ~ x2")

    # x1 exists only in the consistent model, x2 only in the efficient one, so
    # there is nothing left to compare and the test must not invent a df=0
    # "pass" out of an empty quadratic form.
    with pytest.raises(ConsistencyError, match="share no"):
        fe.hausman(re)


def test_hausman_falls_back_to_a_generalised_inverse_and_warns():
    """A naturally non-positive-definite difference must warn, not crash.

    No synthetic covariance surgery: on an ordinary balanced panel the FE and RE
    covariance matrices can come out with ``V_fe - V_re`` indefinite (the RE
    variance components are estimated, so the difference carries sampling noise
    and is not guaranteed positive semidefinite).  ``seed=1`` reproduces such a
    panel, and the assertion below proves the configuration is genuine before
    the warning is checked.
    """

    frame = _panel(seed=1)
    fe = _fe(frame)
    re = _re(frame)

    names = ["x1", "x2"]
    fe_index = [fe.param_names.index(name) for name in names]
    re_index = [re.param_names.index(name) for name in names]
    difference_cov = (
        fe.cov_params()[np.ix_(fe_index, fe_index)]
        - re.cov_params()[np.ix_(re_index, re_index)]
    )
    assert np.linalg.eigvalsh(difference_cov)[0] < 0

    with pytest.warns(RuntimeWarning, match="not positive definite"):
        outcome = fe.hausman(re)

    # The generalised-inverse path is exercised: the statistic is finite and
    # never negative, and the p-value stays a probability.
    assert isinstance(outcome, HausmanResult)
    assert np.isfinite(outcome.statistic)
    assert outcome.statistic >= 0.0
    assert 0.0 <= outcome.pvalue <= 1.0
    assert outcome.df == 2


def test_hausman_rejects_different_sample_sizes():
    frame = _panel()
    fe = _fe(frame)
    re_small = _re(frame.iloc[:100])

    assert fe.nobs != re_small.nobs
    with pytest.raises(ConsistencyError, match="different samples"):
        fe.hausman(re_small)


def test_hausman_rejects_models_sharing_no_slope():
    frame = _panel()
    fe = _fe(frame, "y ~ x1")
    other = ws.OLS.from_formula("y ~ x2", frame).fit()

    with pytest.raises(ConsistencyError, match="share no"):
        fe.hausman(other)


def test_hausman_is_also_available_as_a_free_function():
    frame = _panel()
    fe = _fe(frame)
    re = _re(frame)

    via_method = fe.hausman(re)
    via_function = hausman(fe, re)

    assert isinstance(via_function, HausmanResult)
    assert via_function.statistic == pytest.approx(via_method.statistic, rel=0.0)
    assert via_function.df == via_method.df
    assert via_function.excluded == via_method.excluded


def test_hausman_of_a_model_against_itself_is_zero():
    """H0 is trivially true when both sides are the same estimator."""

    frame = _panel()
    fe = _fe(frame)
    same = _fe(frame)

    with pytest.warns(RuntimeWarning, match="not positive definite"):
        outcome = fe.hausman(same)

    assert outcome.statistic == pytest.approx(0.0, abs=1e-10)
    assert outcome.pvalue == 1.0
    assert outcome.df == 2
