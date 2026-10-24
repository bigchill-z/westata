"""Conformance tests for discrete-choice and count models.

References are independent: analytic cases where the MLE is known, plus
cross-checks against statsmodels' own estimators for the backend wiring.  The
Stata-specific properties under test are z inference, the observed-information
default variance, and McFadden's pseudo R-squared in ``e(r2_p)``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

from westata.errors import ConvergenceError
from westata.estimators.discrete import (
    Logit,
    NegativeBinomial,
    OrderedLogit,
    OrderedProbit,
    Poisson,
    Probit,
)


def _binary(n: int = 400, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    latent = -0.5 + 1.2 * x1 - 0.8 * x2 + rng.logistic(size=n)
    y = (latent > 0).astype(float)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def _count(n: int = 400, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    mu = np.exp(0.3 + 0.6 * x1)
    y = rng.poisson(mu).astype(float)
    return pd.DataFrame({"y": y, "x1": x1})


def _ordered(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    latent = 0.9 * x1 + rng.normal(size=n)
    y = pd.cut(latent, bins=[-np.inf, -0.6, 0.6, np.inf], labels=[1, 2, 3]).astype(
        float
    )
    return pd.DataFrame({"y": y, "x1": x1})


# ------------------------------------------------------------------- logit


def test_logit_uses_z_inference():
    res = Logit.from_formula("y ~ x1 + x2", _binary()).fit()
    assert res.dist == "z"
    expected = 2.0 * stats.norm.sf(np.abs(res.tvalues.to_numpy()))
    np.testing.assert_allclose(res.pvalues.to_numpy(), expected, rtol=1e-12)


def test_logit_matches_statsmodels_point_estimates():
    df = _binary()
    res = Logit.from_formula("y ~ x1 + x2", df).fit()

    X = sm.add_constant(df[["x1", "x2"]], has_constant="add")
    ref = sm.Logit(df["y"], X).fit(method="newton", disp=0, maxiter=500)

    np.testing.assert_allclose(res.params.to_numpy(), ref.params.to_numpy(), rtol=1e-8)
    np.testing.assert_allclose(res.bse.to_numpy(), ref.bse.to_numpy(), rtol=1e-6)


def test_logit_sign_is_consistent_with_dgp():
    res = Logit.from_formula("y ~ x1 + x2", _binary()).fit()
    assert res.params["x1"] > 0  # DGP coefficient is +1.2
    assert res.params["x2"] < 0  # DGP coefficient is -0.8
    assert res.params["x1"] == pytest.approx(1.2, abs=0.4)


def test_logit_pseudo_r2_is_mcfadden():
    res = Logit.from_formula("y ~ x1 + x2", _binary()).fit()
    expected = 1.0 - res.e["ll"] / res.e["ll_0"]
    assert res.e["r2_p"] == pytest.approx(expected, rel=1e-12)
    assert 0.0 <= res.e["r2_p"] <= 1.0


def test_logit_summary_reports_lr_chi2_by_default():
    """Stata prints ``LR chi2`` for a plain ``logit``; only ``vce()`` makes it Wald.

    StataNow 19.5 (MP), captured 2026-10-24, on this module's ``_binary()`` data::

        . logit y x1 x2
        Logistic regression                               Number of obs =    400
                                                          LR chi2(2)    = 136.03
                                                          Prob > chi2   = 0.0000

    The old assertion here expected ``Wald chi2`` on the *default* fit, which is
    the wrong reading of Stata: ``e(chi2type)`` is ``LR`` until a robust VCE
    replaces the observed information.
    """

    text = Logit.from_formula("y ~ x1 + x2", _binary()).fit().summary(style="stata")
    assert "LR chi2(" in text
    assert "Wald chi2(" not in text
    assert "P>|z|" in text
    assert "P>|t|" not in text


def test_logit_summary_reports_wald_chi2_under_robust_vce():
    """``logit ..., vce(robust)`` switches the printed test to Wald.

    StataNow 19.5 (MP), captured 2026-10-24, same data::

        . logit y x1 x2, vce(robust)
        Logistic regression                               Number of obs =    400
                                                          Wald chi2(2)  =  76.14
                                                          Prob > chi2   = 0.0000

    Stata's ``e(chi2)`` there is the slope-block Wald ``b_s' V_ss^-1 b_s``
    (constant excluded); with this module's data it is 76.13552601769518 and
    ``e(chi2type)`` is ``Wald``.
    """

    res = Logit.from_formula("y ~ x1 + x2", _binary()).fit(cov_type="robust")
    text = res.summary(style="stata")
    assert res.e["chi2type"] == "Wald"
    assert "Wald chi2(" in text
    assert "LR chi2(" not in text
    assert res.e["chi2"] == pytest.approx(76.13552601769518, rel=1e-8)


# ------------------------------------------------------------------ probit


def test_probit_matches_statsmodels():
    df = _binary()
    res = Probit.from_formula("y ~ x1 + x2", df).fit()

    X = sm.add_constant(df[["x1", "x2"]], has_constant="add")
    ref = sm.Probit(df["y"], X).fit(method="newton", disp=0, maxiter=500)

    np.testing.assert_allclose(res.params.to_numpy(), ref.params.to_numpy(), rtol=1e-8)


def test_probit_sign_is_consistent_with_dgp():
    # The DGP used a logistic error, so the probit slope is scaled by ~0.55 but
    # must keep the same sign structure.
    res = Probit.from_formula("y ~ x1 + x2", _binary()).fit()
    assert res.params["x1"] > 0
    assert res.params["x2"] < 0


# ----------------------------------------------------------------- poisson


def test_poisson_matches_statsmodels():
    df = _count()
    res = Poisson.from_formula("y ~ x1", df).fit()

    X = sm.add_constant(df[["x1"]], has_constant="add")
    ref = sm.Poisson(df["y"], X).fit(method="newton", disp=0, maxiter=500)

    np.testing.assert_allclose(res.params.to_numpy(), ref.params.to_numpy(), rtol=1e-8)
    np.testing.assert_allclose(res.bse.to_numpy(), ref.bse.to_numpy(), rtol=1e-8)


def test_poisson_recovers_dgp_slope():
    res = Poisson.from_formula("y ~ x1", _count()).fit()
    assert res.params["x1"] == pytest.approx(0.6, abs=0.15)


def test_poisson_robust_cov_type():
    df = _count()
    res = Poisson.from_formula("y ~ x1", df).fit(cov_type="HC0")

    X = sm.add_constant(df[["x1"]], has_constant="add")
    ref = sm.Poisson(df["y"], X).fit(
        method="newton", disp=0, cov_type="HC0", maxiter=500
    )
    np.testing.assert_allclose(res.bse.to_numpy(), ref.bse.to_numpy(), rtol=1e-6)


# ------------------------------------------------------- negative binomial


def test_negative_binomial_runs_and_has_extra_parameter():
    df = _count()
    res = NegativeBinomial.from_formula("y ~ x1", df).fit()
    # nbreg estimates an extra dispersion parameter (alpha).
    assert len(res.params) == 3
    assert np.all(np.isfinite(res.params.to_numpy()))


# ------------------------------------------------------------ ordered models


def test_ordered_logit_runs_and_reports_z():
    res = OrderedLogit.from_formula("y ~ x1", _ordered()).fit()
    assert res.dist == "z"
    assert np.all(np.isfinite(res.params.to_numpy()))
    assert "P>|z|" in res.summary(style="stata")


def test_ordered_probit_runs():
    res = OrderedProbit.from_formula("y ~ x1", _ordered()).fit()
    assert np.all(np.isfinite(res.params.to_numpy()))
    assert res.params["x1"] > 0


def test_ordered_thresholds_are_labelled_like_stata():
    res = OrderedProbit.from_formula("y ~ x1", _ordered()).fit()
    # Three categories -> two cut points, named /cut1 and /cut2 as in Stata.
    assert any(name.startswith("/cut") for name in res.param_names)


def test_ordered_covariance_matches_stata_reference_values():
    """``ologit``/``oprobit`` ``e(V)`` -- both transform defects.

    StataNow 19.5 (MP), captured 2026-10-24.  Data: the verifier's reference DGP
    (``rng = default_rng(20261024)``, n=600, ``eta = 0.5 + 0.8 x1 - 0.4 x2 +
    logistic``, cut at -1.0/0.3/1.6 -- see
    ``retest/domains/verify/p1L_ologit_final.py``)::

        ologit ycat x1 x2      → cut SE = 0.10432381, 0.08740299, 0.09620776
        oprobit ycat x1 x2     → cut SE = 0.06006354, 0.05350367, 0.05700983

    The BRIEF's ``0.0874 / 0.0962`` are the second and third ``ologit`` cut SEs,
    which is why these numbers are the reference.  Two independent bugs used to
    corrupt this matrix: ``theta`` aliased ``params`` (so ``exp`` landed on cut
    points), and the slope-by-cut cross-block was never transformed at all --
    the whole matrix was off by 2.09e+01 (ologit) / 1.05e+01 (oprobit).
    """

    rng = np.random.default_rng(20261024)
    n = 600
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    eta = 0.5 + 0.8 * x1 - 0.4 * x2 + rng.logistic(size=n)
    frame = pd.DataFrame(
        {"x1": x1, "x2": x2, "ycat": np.digitize(eta, [-1.0, 0.3, 1.6]).astype(float)}
    )

    expectations = {
        OrderedLogit: (0.10432381, 0.08740299, 0.09620776),
        OrderedProbit: (0.06006354, 0.05350367, 0.05700983),
    }
    for cls, cut_se in expectations.items():
        res = cls.from_formula("ycat ~ x1 + x2", frame).fit()
        bse = res.bse.to_numpy()
        assert bse[-3:] == pytest.approx(cut_se, rel=1e-6), cls.__name__

        # The slope-by-cut cross-block is the part the old code never
        # transformed at all.  ``cov(x1, cut1)`` captured from the same Stata
        # session (westata's row 0 is ``x1``):
        #   ologit   cov(x1, cut1) = -0.0019737710774645067
        #   oprobit  cov(x1, cut1) = -0.0005006338549901409
        # The pre-fix code returned the *untransformed* backend value here, so
        # these two numbers are what distinguishes a real transform from none.
        cross = res.cov_params()[0, -3]
        expected_cross = (
            -0.0019737710774645067 if cls is OrderedLogit else -0.0005006338549901409
        )
        assert cross == pytest.approx(expected_cross, rel=1e-6)


def test_ordered_robust_vce_runs_and_switches_to_wald():
    """``ologit ..., vce(robust)`` used to raise; now it matches Stata.

    StataNow 19.5 (MP), captured 2026-10-24, on ``_ordered()``::

        ologit y x1, vce(robust)
          → x1 = 1.5355110545 (se .1357913956), cut1 = -.9960622108
            cut2 = 1.2416450573, e(chi2) = 127.86785832284636 (Wald, df 1),
            e(chi2type) = Wald, e(df_m) = 1, e(rank) = 3, e(k_aux) = 2

    The old code raised ``ValueError: only cov_type='nonrobust' is implemented``.
    """

    frame = _ordered()
    res = OrderedLogit.from_formula("y ~ x1", frame).fit(cov_type="robust")

    assert res.e["chi2type"] == "Wald"
    assert res.e["df_m"] == 1
    assert res.e["rank"] == 3
    assert res.e["k_aux"] == 2
    assert res.bse["x1"] == pytest.approx(0.1357913956, rel=1e-5)
    assert res.e["chi2"] == pytest.approx(127.86785832284636, rel=1e-5)


# ------------------------------------------------ e(df_m) excludes aux params


def test_df_m_excludes_the_dispersion_parameter():
    """``nbreg`` reports ``e(df_m) = 1`` for one slope; ``/lnalpha`` is ``k_aux``.

    StataNow 19.5 (MP), captured 2026-10-24, on ``_count()``::

        nbreg y x1
          → e(df_m) = 1, e(rank) = 3, e(k_aux) = 1, e(chi2) = 145.80825090173403,
            e(ll) = -569.0777628924194, e(alpha) = 6.1007222375588825e-06

    The old code reported ``len(params) - 1 = 2``.
    """

    res = NegativeBinomial.from_formula("y ~ x1", _count()).fit()
    assert res.e["df_m"] == 1
    assert res.e["rank"] == 3
    assert res.e["k_aux"] == 1
    assert res.e["chi2"] == pytest.approx(145.80825090173403, rel=1e-4)
    # e(ll) only has to agree to the optimiser's own tolerance on this nearly
    # Poisson sample; the reported value differs in the 9th significant digit.
    assert res.e["ll"] == pytest.approx(-569.0777628924194, rel=1e-8)


def test_df_m_excludes_the_cut_points():
    """``ologit``/``oprobit`` report ``e(df_m) = 1`` for one slope.

    StataNow 19.5 (MP), captured 2026-10-24, on ``_ordered()``::

        ologit y x1    → e(df_m) = 1, e(rank) = 3, e(k_aux) = 2, e(ll) = -354.2923278415281
        oprobit y x1   → e(df_m) = 1, e(rank) = 3, e(k_aux) = 2, e(ll) = -354.1954396222556

    The old code reported ``len(params) - 1 = 2`` for both, i.e. it counted the
    cut points as regressors.
    """

    frame = _ordered()
    for cls, expected_ll in (
        (OrderedLogit, -354.2923278415281),
        (OrderedProbit, -354.1954396222556),
    ):
        res = cls.from_formula("y ~ x1", frame).fit()
        assert res.e["df_m"] == 1, cls.__name__
        assert res.e["rank"] == 3, cls.__name__
        assert res.e["k_aux"] == 2, cls.__name__
        assert res.e["ll"] == pytest.approx(expected_ll, rel=1e-4), cls.__name__


# --------------------------------------------------- robust VCE, per family


def test_probit_robust_vce_matches_stata():
    """``probit ..., vce(robust)`` used to raise a statsmodels ``ValueError``.

    StataNow 19.5 (MP), captured 2026-10-24, on ``_binary()``::

        probit y x1 x2, vce(robust)
          → b = .6458012377, -.5850092652, -.2187586376
            se = .0823800932, .0885486934, .0712072812
            e(chi2) = 86.53234287215409 (Wald), e(chi2type) = Wald
    """

    res = Probit.from_formula("y ~ x1 + x2", _binary()).fit(cov_type="robust")
    assert res.e["chi2type"] == "Wald"
    # westata orders (Intercept, x1, x2); Stata orders (x1, x2, _cons), so the
    # captured vector is reordered here rather than silently compared by name.
    np.testing.assert_allclose(
        res.bse.to_numpy(),
        [0.0712072812, 0.0823800932, 0.0885486934],
        rtol=1e-5,
    )
    assert res.e["chi2"] == pytest.approx(86.53234287215409, rel=1e-5)


def test_negative_binomial_robust_vce_matches_stata():
    """``nbreg ..., vce(robust)`` used to raise a statsmodels ``ValueError``.

    StataNow 19.5 (MP), captured 2026-10-24, on ``_count()``::

        nbreg y x1, vce(robust)
          → b = .5825250129, .2173720311, -12.0071033942
            se = .0441716543, .0473713035, 42.6385629344
            e(chi2) = 173.91691650908487 (Wald), e(df_m) = 1, e(k_aux) = 1

    Only the *slope* block is asserted here: this sample is essentially Poisson
    (``e(alpha) = 6.1e-06``, ``/lnalpha = -12.0``), so ``/lnalpha``'s own
    variance sits on the boundary and is not identified.  Stata's own answer is
    unstable across VCEs on the same data -- ``se(lnalpha)`` is 548.8 under the
    observed information, 42.6 under ``vce(robust)`` and 40.0 under
    ``vce(cluster g)`` -- so it is not a value worth pinning a regression test
    to.  The slope SEs and ``e(chi2)`` are well identified and do match.
    """

    res = NegativeBinomial.from_formula("y ~ x1", _count()).fit(cov_type="robust")
    assert res.e["chi2type"] == "Wald"
    assert res.e["df_m"] == 1
    assert res.e["k_aux"] == 1
    # westata orders (Intercept, x1, /lnalpha); Stata orders (x1, _cons, lnalpha).
    np.testing.assert_allclose(
        res.bse.to_numpy()[:2],
        [0.0473713035, 0.0441716543],
        rtol=1e-3,
    )
    assert res.e["chi2"] == pytest.approx(173.91691650908487, rel=1e-3)


def test_poisson_hc1_is_not_silently_hc0():
    """``HC1`` must really be ``N/(N-K)`` times ``HC0``, not the same matrix.

    statsmodels 0.15 hands both ``HC0`` and ``HC1`` the uncorrected White score,
    so delegating made the two labels return identical numbers.  The ratio is
    checked against the analytic factor instead of a captured Stata value,
    because on this family ``HC0``/``HC1`` are statsmodels conventions with no
    Stata equivalent (``vce(robust)`` is ``cov_type="robust"``).
    """

    frame = _count()
    hc0 = Poisson.from_formula("y ~ x1", frame).fit(cov_type="HC0")
    hc1 = Poisson.from_formula("y ~ x1", frame).fit(cov_type="HC1")

    assert not np.array_equal(hc0.cov_params(), hc1.cov_params())
    factor = len(frame) / (len(frame) - len(hc0.params))
    np.testing.assert_allclose(
        hc1.cov_params(), np.asarray(hc0.cov_params()) * factor, rtol=1e-12
    )
    # And neither of them is Stata's vce(robust), which carries N/(N-1).
    robust = Poisson.from_formula("y ~ x1", frame).fit(cov_type="robust")
    stata_factor = len(frame) / (len(frame) - 1.0)
    np.testing.assert_allclose(
        robust.cov_params(),
        np.asarray(hc0.cov_params()) * stata_factor,
        rtol=1e-12,
    )


# ------------------------------------------------------------ guard rails


def test_unsupported_cov_type_raises():
    with pytest.raises(ValueError, match="cov_type"):
        Logit.from_formula("y ~ x1", _binary()).fit(cov_type="bogus")


def test_perfect_separation_warns_or_raises():
    # A regressor that perfectly predicts the outcome: the MLE diverges.
    df = pd.DataFrame({"y": [0.0, 0.0, 1.0, 1.0], "x": [-2.0, -1.0, 1.0, 2.0]})
    with pytest.raises(Exception):
        Logit.from_formula("y ~ x", df).fit(strict=True, maxiter=200)


def test_missing_lhs_column_raises():
    df = _binary()
    with pytest.raises(Exception, match="not_a_column"):
        Logit.from_formula("not_a_column ~ x1", df).fit()


def test_e_keys_present():
    res = Logit.from_formula("y ~ x1", _binary()).fit()
    for key in ["N", "df_m", "ll", "chi2", "r2_p"]:
        assert key in res.e, key


def test_parameter_names_come_from_the_formula():
    # ``results.test("mpg")`` only works if the design names survive; a raw
    # statsmodels array model would report const/x1/x2 instead.
    res = Logit.from_formula("y ~ x1 + x2", _binary()).fit()
    assert res.param_names == ["Intercept", "x1", "x2"]
    assert np.isfinite(res.test("x1").pvalue)


def test_unusable_covariance_raises_convergence_error():
    # Negative binomial on a nearly-Poisson count with a bad start can fail to
    # produce a Hessian; westata must say so, not raise a raw statsmodels
    # ValueError from cov_params().
    rng = np.random.default_rng(99)
    n = 120
    x = rng.normal(size=n)
    y = rng.poisson(np.exp(0.2 + 0.5 * x)).astype(float)
    df = pd.DataFrame({"y": y, "x": x})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = NegativeBinomial.from_formula("y ~ x", df).fit()
        except ConvergenceError:
            return  # acceptable and expected for a hard likelihood surface
    assert np.all(np.isfinite(res.params.to_numpy()))
