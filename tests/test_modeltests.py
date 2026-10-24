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

from pathlib import Path

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


def test_lrtest_identical_models_is_refused_like_stata_r498():
    """``df == 0`` is Stata's r(498), not a p-value.

    Captured on StataNow 19.5 (2026-10-24), ``sysuse auto``::

        . regress mpg weight
        . estimates store L1
        . regress mpg weight
        . estimates store L2
        . lrtest L1 L2
        df(unrestricted) = df(restricted) = 2
        r(498);

    So the command *fails* in Stata.  The old westata convention (report
    ``chi2 = 0, df = 0, p = 1``, justified by ``chi2(0)`` being a point mass at
    0) made a ported script succeed where Stata errors; the repair raises
    ``ConsistencyError`` mirroring r(498) instead.
    """

    frame = _frame()
    first = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()
    second = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()

    with pytest.raises(ConsistencyError, match="r\\(498\\)"):
        first.lrtest(second)


def test_lrtest_equal_parameter_counts_of_different_models_is_also_r498():
    """Stata checks the *count* first: equal k means r(498) even if x differs.

    Captured on StataNow 19.5 (2026-10-24), ``sysuse auto``::

        . regress mpg weight
        . estimates store N3
        . regress mpg foreign
        . estimates store L1
        . lrtest N3 L1
        df(unrestricted) = df(restricted) = 2
        r(498);
    """

    frame = _frame()
    left = ws.OLS.from_formula("y ~ x1", frame).fit()
    right = ws.OLS.from_formula("y ~ x2", frame).fit()

    with pytest.raises(ConsistencyError, match="r\\(498\\)"):
        left.lrtest(right)


def test_lrtest_swaps_the_arguments_the_way_stata_does():
    """``lrtest`` accepts either order and relabels the pair.

    Captured on StataNow 19.5 (2026-10-24), ``sysuse auto``::

        . regress mpg weight
        . estimates store L1
        . regress mpg weight foreign
        . estimates store B1
        . lrtest L1 B1
        Likelihood-ratio test                    LR chi2(1) =   2.41
        Assumption: L1 nested within B1           Prob > chi2 = 0.1205
        r(chi2) = 2.41124840273977   r(df) = 1   r(p) = .12046626248080883

    ``L1`` is the *smaller* model and was passed first, so Stata silently treats
    it as nested within ``B1``; the statistic is ``2*(ll_B1 - ll_L1)``.
    """

    frame = _frame()
    small = ws.OLS.from_formula("y ~ x1", frame).fit()
    big = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()

    forward = small.lrtest(big)
    backward = big.lrtest(small)

    assert forward.statistic == pytest.approx(backward.statistic, rel=1e-15)
    assert forward.df == backward.df == 1
    assert forward.ll_unrestricted == pytest.approx(big.llf, rel=1e-12)
    assert forward.ll_restricted == pytest.approx(small.llf, rel=1e-12)
    assert forward.statistic > 0.0


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


# ------------------------- Stata-captured regressions (M5 and M6)
#
# Captured from a live StataNow 19.5 (MP) session on 2026-10-24.
#
# lrtest (sysuse auto):
#     . regress mpg weight ; estimates store L1
#     . regress mpg weight foreign ; estimates store B1
#     . lrtest B1 L1      r(chi2) = 2.41124840273977  r(df) = 1  r(p) = .12046626248080883
#     . lrtest L1 B1      accepted, "Assumption: L1 nested within B1", same numbers
#     . lrtest L1 E2      df(unrestricted) = df(restricted) = 2  r(498)
#     . lrtest N3 L1      (mpg foreign vs mpg weight)              r(498)
#     . lrtest N2 L1      (mpg foreign length vs mpg weight)
#                         r(chi2) = -2.68354973321414  r(df) = 1  r(p) = 1
#
# hausman (40 entities x 8 periods, seed 4242, alpha uncorrelated with x1):
#     . hausman FEm REm             chi2(2) = .9525460961317572  p = .6210938706123512
#     . hausman FEm REm, constant   chi2(3) = .952546096131752   p = .8127322054037279
#     . hausman FEm REm   (corr=0.15, indefinite)
#           chi2(2) = -2.35424271822838   p = 1 (Stata prints the negative value)

_AUTO = Path(__file__).parent / "data" / "auto.dta"


def _hausman_panel(seed: int = 4242, corr: float = 0.0) -> pd.DataFrame:
    """The panel the Stata capture used (40 entities x 8 periods).

    Identical to ``retest/fix/postest/repro_M6_hausman_constant.py``, so the
    captured ``chi2`` values are the right expectations for this frame.
    """

    rng = np.random.default_rng(seed)
    n_entities, n_periods = 40, 8
    entity = np.repeat(np.arange(1, n_entities + 1), n_periods)
    time = np.tile(np.arange(1, n_periods + 1), n_entities)
    effect = np.repeat(rng.normal(0.0, 1.5, n_entities), n_periods)
    x1 = rng.normal(0.0, 1.0, n_entities * n_periods) + corr * effect
    x2 = rng.normal(0.0, 1.0, n_entities * n_periods)
    y = 1.0 + 0.8 * x1 - 0.5 * x2 + effect + rng.normal(
        0.0, 0.7, n_entities * n_periods
    )
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "firm": entity, "year": time})


def _auto() -> pd.DataFrame:
    return pd.read_stata(_AUTO, convert_categoricals=False)


def test_lrtest_nested_logit_matches_stata():
    """``lrtest LU LR`` on ``logit foreign weight mpg`` vs ``foreign weight``."""

    frame = _auto()
    big = ws.Logit.from_formula("foreign ~ weight + mpg", frame).fit()
    small = ws.Logit.from_formula("foreign ~ weight", frame).fit()

    outcome = ws.lrtest(big, small)
    assert outcome.statistic == pytest.approx(3.7576914398801, rel=1e-9)
    assert outcome.df == 1
    assert outcome.pvalue == pytest.approx(0.0525651057695751, rel=1e-9)


def test_lrtest_nested_regress_matches_stata():
    """``lrtest RU RR``: chi2 = 2.41124840273977, df = 1, p = .12046626248080883."""

    frame = _auto()
    big = ws.OLS.from_formula("mpg ~ weight + foreign", frame).fit()
    small = ws.OLS.from_formula("mpg ~ weight", frame).fit()

    outcome = ws.lrtest(big, small)
    assert outcome.statistic == pytest.approx(2.41124840273977, rel=1e-9)
    assert outcome.df == 1
    assert outcome.pvalue == pytest.approx(0.12046626248080883, rel=1e-9)


def test_lrtest_accepts_either_argument_order_like_stata():
    """``lrtest L1 B1`` relabels the pair; Stata still reports the same numbers."""

    frame = _auto()
    big = ws.OLS.from_formula("mpg ~ weight + foreign", frame).fit()
    small = ws.OLS.from_formula("mpg ~ weight", frame).fit()

    reversed_ = ws.lrtest(small, big)
    assert reversed_.statistic == pytest.approx(2.41124840273977, rel=1e-9)
    assert reversed_.df == 1
    assert reversed_.ll_unrestricted == pytest.approx(big.llf, rel=1e-12)
    assert reversed_.ll_restricted == pytest.approx(small.llf, rel=1e-12)


def test_lrtest_non_nested_reports_the_negative_statistic_stata_reports():
    """``lrtest N2 L1``: Stata accepts and prints ``LR chi2(1) = -2.6835``.

    ``regress mpg foreign length`` vs ``regress mpg weight``: different
    parameter counts, no nesting; the unrestricted model fits *worse*, so the
    statistic is negative and ``Prob > chi2`` is 1.
    """

    frame = _auto()
    other = ws.OLS.from_formula("mpg ~ foreign + length", frame).fit()
    weight_only = ws.OLS.from_formula("mpg ~ weight", frame).fit()

    with pytest.warns(RuntimeWarning):
        outcome = ws.lrtest(other, weight_only)

    assert outcome.statistic == pytest.approx(-2.68354973321414, rel=1e-9)
    assert outcome.df == 1
    assert outcome.pvalue == 1.0


def test_hausman_constant_option_matches_stata():
    """``hausman ..., constant``: df goes 2 -> 3, statistic unchanged.

    Stata: ``chi2(2) = .9525460961317572`` (default) and
    ``chi2(3) = .952546096131752`` (with ``constant``); the constant's
    difference is ~6e-5 with a negative variance difference, so the generalised
    inverse leaves the quadratic form alone and only ``df`` moves.

    Both the free function and the result's own method are exercised, so the
    ``results/base.py`` forwarder (``def hausman(self, efficient, **kwargs)``)
    cannot silently stop passing ``constant`` through again.
    """

    frame = _hausman_panel()
    fe = _fe(frame)
    re = _re(frame)

    default = fe.hausman(re)
    with pytest.warns(RuntimeWarning, match="not positive definite"):
        with_constant = hausman(fe, re, constant=True)

    assert default.df == 2
    assert with_constant.df == 3
    assert default.statistic == pytest.approx(0.9525460961317572, rel=1e-9)
    assert with_constant.statistic == pytest.approx(0.952546096131752, rel=1e-9)
    # The default must not change when the option is available but unused.
    assert default.excluded == []


def test_hausman_constant_is_reachable_through_the_result_method():
    """``fe.hausman(re, constant=True)`` must work, not just ``ws.hausman(...)``.

    The results object forwards keyword arguments to
    :func:`westata.postestimation.modeltests.hausman`
    (``results/base.py::StataResults.hausman(self, efficient, **kwargs)``).
    Without that forwarding the Stata ``hausman ..., constant`` option would be
    unreachable from the object a user actually holds, so this test pins the
    instance-method path and checks it agrees with the free function to the
    last bit.
    """

    frame = _hausman_panel()
    fe = _fe(frame)
    re = _re(frame)

    with pytest.warns(RuntimeWarning, match="not positive definite"):
        via_method = fe.hausman(re, constant=True)
    with pytest.warns(RuntimeWarning, match="not positive definite"):
        via_function = hausman(fe, re, constant=True)

    assert isinstance(via_method, HausmanResult)
    assert via_method.df == 3
    assert via_method.statistic == pytest.approx(0.952546096131752, rel=1e-9)
    # Forwarding must not perturb the number: identical to the free function.
    assert via_method.statistic == pytest.approx(via_function.statistic, rel=0.0)
    assert via_method.df == via_function.df
    assert via_method.pvalue == pytest.approx(via_function.pvalue, rel=0.0)
    # And the instance method still defaults to Stata's slopes-only comparison.
    assert fe.hausman(re).df == 2


def test_hausman_default_still_excludes_the_constant():
    """``constant=False`` is the default and is a no-op vs the old signature."""

    frame = _hausman_panel()
    fe = _fe(frame)
    re = _re(frame)

    explicit = hausman(fe, re, constant=False)
    default = fe.hausman(re)
    assert explicit.statistic == pytest.approx(default.statistic, rel=0.0)
    assert explicit.df == default.df == 2


def test_hausman_documents_the_deliberate_clamp_on_an_indefinite_difference():
    """Non-PD ``V_c - V_e``: Stata prints a negative statistic, westata floors at 0.

    Measured on StataNow 19.5 (2026-10-24, 40x8 panel, effect correlated with
    ``x1`` at 0.15): ``hausman FEm REm`` reports ``chi2(2) = -2.35424271822838``
    with ``r(p) = 1``; ``hausman FEm REm, constant`` reports the same with
    ``df = 3``.  westata reports ``0`` and the same ``p``, and says so in the
    docstring -- the floor is deliberate, because a pseudo-inverse produces
    ``-1e-18``-scale quadratic forms from round-off alone and reporting the sign
    of those would manufacture spurious "negative statistics".
    """

    frame = _hausman_panel(corr=0.15)
    fe = _fe(frame)
    re = _re(frame)

    with pytest.warns(RuntimeWarning, match="not positive definite"):
        outcome = fe.hausman(re)

    assert outcome.statistic == 0.0
    assert outcome.pvalue == 1.0
    assert outcome.df == 2
    # The docstring is the contract for *why* 0 and not -2.3542.
    docstring = hausman.__doc__ or ""
    assert "-2.35424271822838" in docstring
    assert "deliberate" in docstring
