"""Tests for the panel non-linear models (P9a).

These are conformance tests, not smoke tests: the reference values come from
``tests/conformance/stata_fixtures/p9_outputs.txt`` (StataNow 19.5) and the
fixture DGP is rebuilt here from ``scripts/capture_p9_fixtures.py`` so the
numbers are comparable at all.  Where a quantity cannot be matched exactly the
gap is asserted *as a gap* with a comment saying why, rather than being hidden
behind a loose tolerance.

Reference panel for the binary models (``binary_panel``): n=120 firms x 8 years,
seed 71, ``latent = 0.4 x1 - 0.3 x2 + N(0, 0.8^2)_firm + logistic``.  Two of the
120 firms are all-zero and four are all-one, which is where Stata's 912 comes
from.  Reference panel for the count models (``count_panel``): n=100 x 7,
seed 73.  For ``xttobit`` (``censored_panel``): n=100 x 7, seed 83, ``ll(.5)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from westata.errors import ConsistencyError, UnsupportedError
from westata.estimators.panel_nonlinear import (
    XtLogit,
    XtNbreg,
    XtPoisson,
    XtProbit,
    XtTobit,
)

# Stata's printed values (p9_outputs.txt).  Kept as module constants so each
# assertion can name the number it is checking.
STATA_XTLOGIT_FE = (0.3307537, -0.3134259)
STATA_XTLOGIT_FE_SE = (0.0807304, 0.0795823)
STATA_XTLOGIT_FE_LL = -387.92362
STATA_XTLOGIT_FE_CHI2 = 32.46
STATA_XTLOGIT_FE_N = 912

STATA_XTLOGIT_RE = (0.3428887, -0.2896591, 0.1685264)
STATA_XTLOGIT_RE_LNSIG2U = -0.4766337
STATA_XTLOGIT_RE_LL = -630.38212
STATA_XTLOGIT_RE_SIGMA_U = 0.787953

STATA_XTLOGIT_PA = (0.3007615, -0.2542736, 0.1488734)
STATA_XTLOGIT_PA_SE = (0.0685441, 0.0659616, 0.0888241)

STATA_XTPROBIT_RE = (0.2098340, -0.1765590, 0.1023312)
STATA_XTPROBIT_RE_LNSIG2U = -1.462809
STATA_XTPROBIT_RE_LL = -630.25751

STATA_XTPOISSON_FE = 0.5041803
STATA_XTPOISSON_FE_SE = 0.0344352
STATA_XTPOISSON_FE_LL = -775.44068
STATA_XTPOISSON_RE = (0.4795509, 0.3485993, -2.073955)
STATA_XTPOISSON_RE_LL = -1076.9058

STATA_XTNBREG_FE = 0.5041909
STATA_XTNBREG_FE_LL = -775.44069

STATA_XTTOBIT = (0.9020131, 0.9253712)
STATA_XTTOBIT_SIGMA_U = 0.4883516
STATA_XTTOBIT_SIGMA_E = 0.6004406
STATA_XTTOBIT_LL = -598.85058

#: Keys every ``fe``/``re`` likelihood result must populate.
REQUIRED_E_KEYS = ("N", "df_m", "ll", "chi2", "p", "rank", "N_g")


# --------------------------------------------------------------------- fixtures
def binary_panel(n_firms: int = 120, n_years: int = 8, seed: int = 71) -> pd.DataFrame:
    """The P9 fixture DGP for ``xtlogit`` / ``xtprobit``."""

    rng = np.random.default_rng(seed)
    firm = np.repeat(np.arange(n_firms), n_years)
    year = np.tile(np.arange(1, n_years + 1), n_firms)
    n = firm.size
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    effect = rng.normal(scale=0.8, size=n_firms)[firm]
    latent = 0.4 * x1 - 0.3 * x2 + effect + rng.logistic(size=n)
    return pd.DataFrame(
        {
            "firm": firm,
            "year": year,
            "x1": x1,
            "x2": x2,
            "y": (latent > 0).astype(float),
        }
    )


def count_panel(n_firms: int = 100, n_years: int = 7, seed: int = 73) -> pd.DataFrame:
    """The P9 fixture DGP for ``xtpoisson`` / ``xtnbreg``."""

    rng = np.random.default_rng(seed)
    firm = np.repeat(np.arange(n_firms), n_years)
    year = np.tile(np.arange(1, n_years + 1), n_firms)
    n = firm.size
    x1 = rng.normal(size=n)
    effect = rng.normal(scale=0.4, size=n_firms)[firm]
    mu = np.exp(0.3 + 0.5 * x1 + effect)
    return pd.DataFrame(
        {"firm": firm, "year": year, "x1": x1, "y": rng.poisson(mu).astype(float)}
    )


def censored_panel(
    n_firms: int = 100, n_years: int = 7, seed: int = 83
) -> pd.DataFrame:
    """The P9 fixture DGP for ``xttobit`` (``ll(.5)``)."""

    rng = np.random.default_rng(seed)
    firm = np.repeat(np.arange(n_firms), n_years)
    year = np.tile(np.arange(1, n_years + 1), n_firms)
    n = firm.size
    x1 = rng.normal(size=n)
    effect = rng.normal(scale=0.5, size=n_firms)[firm]
    latent = 1.0 + 0.9 * x1 + effect + rng.normal(scale=0.6, size=n)
    return pd.DataFrame(
        {"firm": firm, "year": year, "x1": x1, "y": np.maximum(latent, 0.5)}
    )


# Module-scoped fits: these models are the expensive part of this file and every
# test below only reads the result object.
@pytest.fixture(scope="module")
def fe_result():
    return XtLogit.from_formula(
        "y ~ x1 + x2", binary_panel(), entity="firm", method="fe"
    ).fit()


@pytest.fixture(scope="module")
def re_result():
    return XtLogit.from_formula(
        "y ~ x1 + x2", binary_panel(), entity="firm", method="re"
    ).fit()


@pytest.fixture(scope="module")
def pa_result():
    return XtLogit.from_formula(
        "y ~ x1 + x2", binary_panel(), entity="firm", method="pa"
    ).fit()


@pytest.fixture(scope="module")
def probit_result():
    return XtProbit.from_formula(
        "y ~ x1 + x2", binary_panel(), entity="firm", method="re"
    ).fit()


@pytest.fixture(scope="module")
def poisson_fe_result():
    return XtPoisson.from_formula(
        "y ~ x1", count_panel(), entity="firm", method="fe"
    ).fit()


@pytest.fixture(scope="module")
def poisson_re_result():
    return XtPoisson.from_formula(
        "y ~ x1", count_panel(), entity="firm", method="re"
    ).fit()


@pytest.fixture(scope="module")
def nbreg_fe_result():
    return XtNbreg.from_formula(
        "y ~ x1", count_panel(), entity="firm", method="fe"
    ).fit()


@pytest.fixture(scope="module")
def tobit_result():
    return XtTobit.from_formula(
        "y ~ x1", censored_panel(), entity="firm", lower=0.5
    ).fit()


# ------------------------------------------------------- xtlogit, fe (Stata)
def test_xtlogit_fe_coefficients_match_stata(fe_result):
    """p9_outputs.txt lines 26-27: .3307537 / -.3134259."""

    assert float(fe_result.params["x1"]) == pytest.approx(STATA_XTLOGIT_FE[0], abs=1e-3)
    assert float(fe_result.params["x2"]) == pytest.approx(STATA_XTLOGIT_FE[1], abs=1e-3)


def test_xtlogit_fe_coefficients_match_stata_much_closer_than_1e_3(fe_result):
    """The contract only asks for 1e-3; the actual agreement is 2e-7.

    Pinned at the tighter tolerance so a future 'simplification' of the
    conditional likelihood cannot quietly degrade it to the contract minimum.
    """

    assert float(fe_result.params["x1"]) == pytest.approx(STATA_XTLOGIT_FE[0], abs=1e-6)
    assert float(fe_result.params["x2"]) == pytest.approx(STATA_XTLOGIT_FE[1], abs=1e-6)


def test_xtlogit_fe_standard_errors_match_stata(fe_result):
    """p9_outputs.txt lines 26-27: se .0807304 / .0795823."""

    assert float(fe_result.bse["x1"]) == pytest.approx(STATA_XTLOGIT_FE_SE[0], abs=1e-5)
    assert float(fe_result.bse["x2"]) == pytest.approx(STATA_XTLOGIT_FE_SE[1], abs=1e-5)


def test_xtlogit_fe_uses_exactly_912_observations(fe_result):
    """The headline contract: e(N) must be 912, not the 960 rows in the frame.

    6 of the 120 firms have a constant outcome and Stata drops all 48 of their
    observations.
    """

    assert int(fe_result.e["N"]) == STATA_XTLOGIT_FE_N
    assert int(fe_result.e["N_g"]) == 114
    assert int(fe_result.e["N_drop"]) == 48
    assert int(fe_result.e["N_group_drop"]) == 6
    assert int(fe_result.nobs) == STATA_XTLOGIT_FE_N


def test_xtlogit_fe_log_likelihood_and_lr_statistic_match_stata(fe_result):
    """p9_outputs.txt line 21: Log likelihood = -387.92362, LR chi2(2) = 32.46.

    This is the check that the null model is the *conditional* one: the
    combinatorial baseline ``-sum ln C(T_i, k_i)`` gives 32.46, whereas a pooled
    logit baseline would give 485 on the same estimates.
    """

    assert float(fe_result.e["ll"]) == pytest.approx(STATA_XTLOGIT_FE_LL, abs=1e-5)
    assert float(fe_result.e["chi2"]) == pytest.approx(STATA_XTLOGIT_FE_CHI2, abs=0.01)
    assert fe_result.e["chi2type"] == "LR"
    assert float(fe_result.e["p"]) < 1e-6


def test_xtlogit_fe_excludes_an_all_zero_group():
    """A group forced to all zeros must drop out of the sample, not be fitted.

    The conditional likelihood conditions on the number of successes; a group
    with zero (or all) successes has probability one under every beta and so
    contributes no information.  Comparing the fit against the same panel with
    that one firm removed makes the contract explicit: the estimates must be
    *identical*, not merely close, and ``e(N)`` must be exactly 6 fewer.
    """

    frame = binary_panel(n_firms=40, n_years=6, seed=11)
    victim = int(frame["firm"].iloc[0])
    frame.loc[frame["firm"] == victim, "y"] = 0.0

    with_zero = XtLogit.from_formula(
        "y ~ x1 + x2", frame, entity="firm", method="fe"
    ).fit()
    without = XtLogit.from_formula(
        "y ~ x1 + x2", frame[frame["firm"] != victim], entity="firm", method="fe"
    ).fit()

    assert int(with_zero.e["N"]) == int(without.e["N"])
    assert int(with_zero.e["N_group_drop"]) == int(without.e["N_group_drop"]) + 1
    np.testing.assert_allclose(
        with_zero.params.to_numpy(), without.params.to_numpy(), atol=1e-10
    )
    # And the excluded firm really did cost the sample its 6 observations.
    varying = sum(
        6
        for _, group in frame.groupby("firm")
        if group["y"].min() < 0.5 < group["y"].max()
    )
    assert int(with_zero.e["N"]) == varying


def test_xtlogit_fe_refuses_a_sample_with_no_variation():
    frame = binary_panel(n_firms=30, n_years=4, seed=5)
    frame["y"] = 1.0
    with pytest.raises(ConsistencyError, match="no group has variation"):
        XtLogit.from_formula("y ~ x1", frame, entity="firm", method="fe").fit()


def test_xtlogit_fe_has_no_intercept(fe_result):
    """The fixed effect absorbs the constant, so none is reported.

    The earlier implementation kept an ``Intercept`` column and set it to NaN,
    which poisons ``lincom``/``test``/``predict`` downstream.
    """

    assert "_cons" not in fe_result.param_names
    assert np.all(np.isfinite(fe_result.params.to_numpy()))
    assert np.all(np.isfinite(fe_result.cov_params()))


# ------------------------------------------------------ xtlogit, re (Stata)
def test_xtlogit_re_coefficients_match_stata(re_result):
    """p9_outputs.txt lines 71-75: .3428887 / -.2896591 / .1685264."""

    for name, expected in zip(("x1", "x2", "_cons"), STATA_XTLOGIT_RE):
        assert float(re_result.params[name]) == pytest.approx(expected, abs=1e-5)


def test_xtlogit_re_lnsig2u_matches_stata(re_result):
    """``/lnsig2u`` - .4766337, and it is stored on that scale, not as sigma.

    The previous implementation labelled a ``log(sigma)`` parameter
    ``/lnsigma``; Stata stores the log of the *variance*.
    """

    assert "lnsig2u" in re_result.param_names
    assert float(re_result.params["lnsig2u"]) == pytest.approx(
        STATA_XTLOGIT_RE_LNSIG2U, abs=1e-4
    )


def test_xtlogit_re_sigma_u_and_rho_are_derived_correctly(re_result):
    """sigma_u = exp(lnsig2u/2) = .787953, rho = s2u/(s2u + pi^2/3) = .1587603."""

    assert float(re_result.e["sigma_u"]) == pytest.approx(
        STATA_XTLOGIT_RE_SIGMA_U, abs=1e-5
    )
    assert float(re_result.e["rho"]) == pytest.approx(0.1587603, abs=1e-5)


def test_xtlogit_re_log_likelihood_matches_stata(re_result):
    """p9_outputs.txt line 66: Log likelihood = -630.38212.

    The likelihood is the adaptive-Gauss-Hermite integral; a mis-scaled
    quadrature weight shows up here as an ``exp(a^2)`` shift, not as a small
    numerical difference.
    """

    assert float(re_result.e["ll"]) == pytest.approx(STATA_XTLOGIT_RE_LL, abs=1e-4)


def test_xtlogit_re_and_fe_agree_when_there_is_no_individual_effect():
    """With no firm effect the two estimators target the same parameter.

    Different likelihoods still mean the *estimates* differ by sampling noise,
    so the tolerance is a few standard errors' worth of slack (2e-2 absolute)
    rather than 1e-6.  On this no-effect DGP the gap measured is ~1.3e-2.
    """

    rng = np.random.default_rng(4001)
    n_firms, n_years = 150, 8
    firm = np.repeat(np.arange(n_firms), n_years)
    n = firm.size
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    latent = 0.6 * x1 - 0.4 * x2 + rng.logistic(size=n)  # no firm effect
    frame = pd.DataFrame({"firm": firm, "x1": x1, "x2": x2, "y": (latent > 0).astype(float)})

    fe = XtLogit.from_formula("y ~ x1 + x2", frame, entity="firm", method="fe").fit()
    re = XtLogit.from_formula("y ~ x1 + x2", frame, entity="firm", method="re").fit()

    # The random-effect variance collapses towards zero: that is *why* the two
    # agree, so it is asserted rather than assumed.
    assert float(re.e["sigma_u"]) < 0.35
    for name in ("x1", "x2"):
        assert float(fe.params[name]) == pytest.approx(
            float(re.params[name]), abs=2e-2
        )


# ------------------------------------------------------ xtlogit, pa (Stata)
def test_xtlogit_pa_reports_no_likelihood(pa_result):
    """``pa`` is GEE, so e(ll) and everything derived from it must be absent."""

    for absent in ("ll", "ll_0", "r2_p", "aic", "bic"):
        assert absent not in pa_result.e, f"{absent} must not be reported by a GEE"
    assert pa_result.e["chi2type"] == "Wald"
    assert pa_result.e["estimator"] == "gee"
    assert np.isnan(pa_result.llf)


def test_xtlogit_pa_coefficients_match_stata(pa_result):
    """p9_outputs.txt lines 103-105: .3007615 / -.2542736 / .1488734.

    Verified to 3e-5 on this panel.
    """

    for name, expected in zip(("x1", "x2", "_cons"), STATA_XTLOGIT_PA):
        assert float(pa_result.params[name]) == pytest.approx(expected, abs=1e-4)


def test_xtlogit_pa_reports_the_model_based_variance_by_default(pa_result):
    """Plain ``xtlogit, pa`` in Stata prints model-based, not robust, errors.

    p9_outputs.txt lines 103-105: .0685441 / .0659616 / .0888241.
    """

    assert pa_result.cov_type == "nonrobust"
    assert pa_result.e["vce"] == "model-based"
    for name, expected in zip(("x1", "x2", "_cons"), STATA_XTLOGIT_PA_SE):
        assert float(pa_result.bse[name]) == pytest.approx(expected, abs=1e-4)


def test_xtlogit_pa_robust_variance_is_available_and_different():
    """``cov_type="robust"`` must give the sandwich, and must actually differ.

    A "robust" option that silently returned the model-based matrix would be the
    worst possible outcome, so the two are compared here.
    """

    model = XtLogit.from_formula(
        "y ~ x1 + x2", binary_panel(), entity="firm", method="pa"
    )
    plain = model.fit()
    robust = model.fit(cov_type="robust")

    assert robust.cov_type == "robust"
    assert robust.e["vce"] == "robust"
    np.testing.assert_allclose(plain.params.to_numpy(), robust.params.to_numpy())
    assert not np.allclose(plain.bse.to_numpy(), robust.bse.to_numpy())


# ------------------------------------------------------------ xtprobit, re
def test_xtprobit_re_matches_stata(probit_result):
    """p9_outputs.txt lines 147-154: .209834 / -.176559 / .1023312 / -1.462809."""

    for name, expected in zip(("x1", "x2", "_cons"), STATA_XTPROBIT_RE):
        assert float(probit_result.params[name]) == pytest.approx(expected, abs=1e-5)
    assert float(probit_result.params["lnsig2u"]) == pytest.approx(
        STATA_XTPROBIT_RE_LNSIG2U, abs=1e-4
    )
    assert float(probit_result.e["ll"]) == pytest.approx(STATA_XTPROBIT_RE_LL, abs=1e-4)


def test_xtprobit_re_rho_uses_the_unit_variance_error(probit_result):
    """A probit standardises the idiosyncratic error, so rho = s2u/(s2u + 1).

    p9_outputs.txt line 154: rho = .1880381.  Using the logit's ``pi^2/3``
    instead would give .318, a very visible error.
    """

    assert float(probit_result.e["rho"]) == pytest.approx(0.1880381, abs=1e-5)
    sigma_u = float(probit_result.e["sigma_u"])
    assert sigma_u == pytest.approx(0.4812326, abs=1e-5)
    assert sigma_u**2 / (sigma_u**2 + 1.0) == pytest.approx(
        float(probit_result.e["rho"])
    )


def test_xtprobit_rejects_fe_and_pa():
    frame = binary_panel(n_firms=30, n_years=5, seed=3)
    for method in ("fe", "pa"):
        with pytest.raises(UnsupportedError, match="xtprobit"):
            XtProbit.from_formula("y ~ x1", frame, entity="firm", method=method)


# -------------------------------------------------------------- xtpoisson fe
def test_xtpoisson_fe_matches_stata(poisson_fe_result):
    """p9_outputs.txt lines 249-250: .5041803 (se .0344352), ll -775.44068."""

    assert float(poisson_fe_result.params["x1"]) == pytest.approx(
        STATA_XTPOISSON_FE, abs=1e-6
    )
    assert float(poisson_fe_result.bse["x1"]) == pytest.approx(
        STATA_XTPOISSON_FE_SE, abs=1e-6
    )
    assert float(poisson_fe_result.e["ll"]) == pytest.approx(
        STATA_XTPOISSON_FE_LL, abs=1e-5
    )
    assert int(poisson_fe_result.e["N"]) == 700


def test_xtpoisson_fe_has_no_intercept(poisson_fe_result):
    """Conditioning on the group totals absorbs the constant."""

    assert poisson_fe_result.param_names == ["x1"]
    assert np.all(np.isfinite(poisson_fe_result.params.to_numpy()))


# -------------------------------------------------------------- xtpoisson re
def test_xtpoisson_re_matches_stata(poisson_re_result):
    """p9_outputs.txt lines 284-289: .4795509 / .3485993 / -2.073955, ll -1076.9058.

    The gamma random effect is conjugate, so this likelihood is closed form and
    an exact match is the right expectation.
    """

    for name, expected in zip(
        ("x1", "_cons", "lnalpha"), STATA_XTPOISSON_RE
    ):
        assert float(poisson_re_result.params[name]) == pytest.approx(expected, abs=1e-5)
    assert float(poisson_re_result.e["ll"]) == pytest.approx(
        STATA_XTPOISSON_RE_LL, abs=1e-4
    )


def test_xtpoisson_re_alpha_is_exp_lnalpha(poisson_re_result):
    """``e(alpha)`` must be the exponentiated dispersion, not a second estimate."""

    assert float(poisson_re_result.e["alpha"]) == pytest.approx(
        float(np.exp(float(poisson_re_result.params["lnalpha"])))
    )
    assert float(poisson_re_result.e["alpha"]) == pytest.approx(0.1256877, abs=1e-5)


# ---------------------------------------------------------------- xtnbreg fe
def test_xtnbreg_fe_matches_stata(nbreg_fe_result):
    """Stata ``xtnbreg y x1, fe``: x1 .5041909 (se .0344353), ll -775.44069.

    Captured with StataNow 19.5 on 2026-10-24 on this exact 100x7 fixture panel
    (``xtset firm year``), together with its tighter-tolerance rerun::

        xtnbreg y x1, fe                     x1 .50419085  se .034435259
        xtnbreg y x1, fe ltolerance(1e-8)    x1 .50418027  se .034435192
                                             ll -775.440679

    The previous version of this test claimed the conditional kernel *is* the
    Poisson one and that ``.5041803`` was "5.5e-5 away because of Stata's
    optimiser tolerance".  Both halves were wrong: the HHG kernel is not the
    Poisson kernel (its SE differs by 6.8%, see
    ``test_xtnbreg_fe_differs_from_the_poisson_kernel``), and ``.50418027`` is
    exactly where *Stata itself* lands once its own tolerance is tightened.

    So the primary assertion is against Stata's converged answer, with the
    residual against Stata's printed default kept explicit: it is Stata's
    ``ltolerance(1e-7)`` stop, not a modelling difference.
    """

    converged_slope = 0.50418027  # Stata with ltolerance(1e-8)
    assert float(nbreg_fe_result.params["x1"]) == pytest.approx(converged_slope, abs=1e-6)
    assert float(nbreg_fe_result.bse["x1"]) == pytest.approx(0.034435192, abs=1e-6)
    assert float(nbreg_fe_result.e["ll"]) == pytest.approx(-775.440679, abs=1e-5)
    # Stata's own default printout is 2.1e-5 away, because it stops earlier.
    assert float(nbreg_fe_result.params["x1"]) == pytest.approx(
        STATA_XTNBREG_FE, abs=5e-5
    )


def test_xtnbreg_fe_is_labelled_a_quasi_likelihood(nbreg_fe_result):
    """The objective is the HHG conditional kernel, and the result says so."""

    assert nbreg_fe_result.e["quasi_likelihood"] == 1
    # Renamed from "conditional_quasi_ml": the objective is now the actual
    # Hausman-Hall-Griliches conditional kernel rather than the conditional
    # Poisson one it used to stand in for.
    assert nbreg_fe_result.e["estimator"] == "hhg_conditional_ml"
    note = str(nbreg_fe_result.e["quasi_likelihood_note"]).lower()
    assert "hausman" in note or "hhg" in note


def test_xtnbreg_fe_differs_from_the_poisson_kernel(nbreg_fe_result, poisson_fe_result):
    """The HHG kernel is *not* the conditional Poisson kernel.

    Captured 2026-10-24 on the 100x7 fixture panel::

        xtpoisson y x1, fe    x1 .5041803   se .0344352   ll -775.44068
        xtnbreg   y x1, fe    x1 .5041909   se .0344353   ll -775.44069

    The point estimates happen to be close here, but on the 30x6 panel where
    ``xtnbreg`` was measured properly the two separate: ``xtnbreg`` gives
    ``.3584498`` with SE ``.0589124`` while ``xtpoisson`` gives ``.3584568``
    with SE ``.0578473`` - a 1.8% SE gap, and Stata reports them as different
    models.  The old test asserted agreement to 1e-5, which only held because
    this module was computing the Poisson objective under the NB name.

    What *is* shared is the combinatorial constant: ``e(ll)`` of the two
    estimators coincides when the dispersion term vanishes, so this test pins
    the slopes against each other loosely and the labels strictly.
    """

    assert float(nbreg_fe_result.params["x1"]) == pytest.approx(
        float(poisson_fe_result.params["x1"]), abs=1e-3
    )
    # But the objectives are different, so the reported values are too.
    assert nbreg_fe_result.e["estimator"] == "hhg_conditional_ml"
    assert poisson_fe_result.e.get("estimator") != "hhg_conditional_ml"
    # And the NB intercept is estimated, while the Poisson one is absorbed.
    assert "_cons" in nbreg_fe_result.param_names
    assert "_cons" not in poisson_fe_result.param_names


def test_xtnbreg_fe_reports_no_dispersion(nbreg_fe_result):
    """The conditional kernel does not identify alpha, so none is reported."""

    assert "lnalpha" not in nbreg_fe_result.param_names
    assert "lnalpha" not in nbreg_fe_result.e


# ------------------------------------------------------------------ xttobit
def test_xttobit_matches_stata(tobit_result):
    """p9_outputs.txt lines 214-220: .9020131 / .9253712, rho .3981316."""

    assert float(tobit_result.params["x1"]) == pytest.approx(STATA_XTTOBIT[0], abs=1e-5)
    assert float(tobit_result.params["_cons"]) == pytest.approx(
        STATA_XTTOBIT[1], abs=1e-5
    )
    assert float(tobit_result.e["ll"]) == pytest.approx(STATA_XTTOBIT_LL, abs=1e-4)


def test_xttobit_reports_both_variance_components(tobit_result):
    """p9_outputs.txt lines 217-220: sigma_u .4883516, sigma_e .6004406."""

    assert float(tobit_result.params["sigma_u"]) == pytest.approx(
        STATA_XTTOBIT_SIGMA_U, abs=1e-5
    )
    assert float(tobit_result.params["sigma_e"]) == pytest.approx(
        STATA_XTTOBIT_SIGMA_E, abs=1e-5
    )
    sigma_u = float(tobit_result.e["sigma_u"])
    sigma_e = float(tobit_result.e["sigma_e"])
    assert float(tobit_result.e["rho"]) == pytest.approx(
        sigma_u**2 / (sigma_u**2 + sigma_e**2)
    )
    assert float(tobit_result.e["rho"]) == pytest.approx(0.3981316, abs=1e-5)


def test_xttobit_counts_the_censored_observations(tobit_result):
    """p9_outputs.txt lines 196-198: 461 uncensored, 239 left-censored."""

    assert int(tobit_result.e["n_uncensored"]) == 461
    assert int(tobit_result.e["n_left"]) == 239
    assert int(tobit_result.e["n_right"]) == 0
    assert int(tobit_result.e["N"]) == 700


def test_xttobit_with_a_very_low_limit_approximates_xtreg_re():
    """As the limit falls below the data, Tobit collapses to the linear RE model.

    The comparison uses westata's own ``RandomEffects`` (``xtreg, re``) on the
    identical sample.  Tobit weights the observations by the inverse Mills
    terms, which vanish as the censoring probability goes to zero, so the two
    coefficient vectors must agree; the tolerance is 2e-2 because the
    censoring count is small but not zero.
    """

    from westata.estimators.panel import RandomEffects

    frame = censored_panel(n_firms=60, n_years=6, seed=17)
    floor = float(frame["y"].min()) - 50.0

    tobit = XtTobit.from_formula(
        "y ~ x1", frame, entity="firm", lower=floor
    ).fit(intpoints=20)
    linear = RandomEffects.from_formula(
        "y ~ x1", frame, entity="firm", time="year"
    ).fit()

    assert int(tobit.e["n_left"]) == 0
    assert float(tobit.params["x1"]) == pytest.approx(
        float(linear.params["x1"]), abs=2e-2
    )
    assert float(tobit.params["_cons"]) == pytest.approx(
        float(linear.params["Intercept"]), abs=2e-2
    )
    # With no censoring the model is the linear random-effects model, so
    # sigma_u^2 + sigma_e^2 must recover the variance of the *raw* residual
    # u_i + e_it.  (``linear.resid`` is the quasi-demeaned residual, not the raw
    # one, so it is rebuilt here from the fitted coefficients.)
    raw = frame["y"].to_numpy() - (
        float(linear.params["Intercept"])
        + float(linear.params["x1"]) * frame["x1"].to_numpy()
    )
    total = float(tobit.e["sigma_u"]) ** 2 + float(tobit.e["sigma_e"]) ** 2
    assert total == pytest.approx(float(np.var(raw)), rel=0.15)


def test_xttobit_rejects_a_specification_with_no_limits():
    frame = censored_panel(n_firms=20, n_years=4, seed=2)
    with pytest.raises(ConsistencyError, match="censoring limit"):
        XtTobit.from_formula("y ~ x1", frame, entity="firm", lower=None, upper=None)


def test_xttobit_rejects_an_inverted_limit_pair():
    frame = censored_panel(n_firms=20, n_years=4, seed=2)
    with pytest.raises(ConsistencyError, match="below"):
        XtTobit.from_formula("y ~ x1", frame, entity="firm", lower=2.0, upper=1.0)


# ------------------------------------------------ the shared e() contract
@pytest.mark.parametrize(
    "factory, label",
    [
        (lambda: XtLogit.from_formula("y ~ x1 + x2", binary_panel(), entity="firm", method="fe"), "xtlogit, fe"),
        (lambda: XtLogit.from_formula("y ~ x1 + x2", binary_panel(), entity="firm", method="re"), "xtlogit, re"),
        (lambda: XtLogit.from_formula("y ~ x1 + x2", binary_panel(), entity="firm", method="pa"), "xtlogit, pa"),
        (lambda: XtProbit.from_formula("y ~ x1 + x2", binary_panel(), entity="firm"), "xtprobit, re"),
        (lambda: XtTobit.from_formula("y ~ x1", censored_panel(), entity="firm", lower=0.5), "xttobit"),
        (lambda: XtPoisson.from_formula("y ~ x1", count_panel(), entity="firm", method="fe"), "xtpoisson, fe"),
        (lambda: XtPoisson.from_formula("y ~ x1", count_panel(), entity="firm", method="re"), "xtpoisson, re"),
        (lambda: XtNbreg.from_formula("y ~ x1", count_panel(), entity="firm", method="fe"), "xtnbreg, fe"),
    ],
)
def test_every_model_reports_the_stata_contract(factory, label):
    """N, df_m, ll (except pa), chi2, p and a renderable Stata summary."""

    result = factory().fit()

    assert result.model_name == label
    assert result.dist == "z"

    for key in REQUIRED_E_KEYS:
        if label == "xtlogit, pa" and key == "ll":
            # A GEE has no likelihood; its absence is asserted separately by
            # ``test_xtlogit_pa_reports_no_likelihood``.
            continue
        assert key in result.e, f"{label} is missing e({key})"
    if label != "xtlogit, pa":
        assert np.isfinite(float(result.e["ll"])), f"{label} has e(ll) = NaN"
        assert float(result.e["ll"]) < 0.0
    assert int(result.e["N"]) > 0
    assert int(result.e["df_m"]) > 0
    assert np.isfinite(float(result.e["chi2"]))
    assert float(result.e["chi2"]) > 0.0
    assert 0.0 <= float(result.e["p"]) <= 1.0
    assert np.all(np.isfinite(result.params.to_numpy()))
    assert np.all(np.isfinite(result.cov_params()))

    rendered = result.summary(style="stata")
    assert isinstance(rendered, str)
    assert "Number of obs" in rendered
    assert "Coef." in rendered
    assert "Std. err." in rendered
    assert "P>|z|" in rendered
    # Every reported coefficient must appear in the table body.
    for name in result.param_names:
        assert name in rendered


@pytest.mark.parametrize(
    "factory, label",
    [
        (lambda: XtLogit.from_formula("y ~ x1", binary_panel(n_firms=40, seed=9), entity="firm", method="between"), "xtlogit"),
        (lambda: XtProbit.from_formula("y ~ x1", binary_panel(n_firms=40, seed=9), entity="firm", method="fe"), "xtprobit"),
        (lambda: XtPoisson.from_formula("y ~ x1", count_panel(n_firms=40, seed=9), entity="firm", method="pa"), "xtpoisson"),
        (lambda: XtNbreg.from_formula("y ~ x1", count_panel(n_firms=40, seed=9), entity="firm", method="re"), "xtnbreg"),
        (lambda: XtTobit.from_formula("y ~ x1", censored_panel(n_firms=40, seed=9), entity="firm", method="fe"), "xttobit"),
    ],
)
def test_unsupported_methods_raise(factory, label):
    with pytest.raises(UnsupportedError, match=label):
        factory()


def test_unsupported_cov_type_raises_rather_than_pretending(fe_result):
    """A robust VCE is not implemented for the likelihood models.

    Returning the observed-information matrix under a "robust" label would be a
    silent numerical lie, so it must raise.
    """

    model = XtLogit.from_formula(
        "y ~ x1 + x2", binary_panel(), entity="firm", method="fe"
    )
    with pytest.raises(UnsupportedError, match="cov_type"):
        model.fit(cov_type="HC1")


def test_missing_entity_column_is_reported_clearly():
    with pytest.raises(ConsistencyError, match="not found in data"):
        XtLogit.from_formula(
            "y ~ x1", binary_panel(), entity="nosuchfirm", method="fe"
        )
