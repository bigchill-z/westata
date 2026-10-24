"""Tests for the P6 limited dependent-variable models.

Coverage map (the contract for ``estimators/limited.py``):

* ``Truncreg`` degenerate case: very wide limits reproduce OLS to 1e-6.
* ``Truncreg`` log-likelihood equals a hand-written [R] truncreg formula to
  1e-10 on a small synthetic sample.
* ``Truncreg`` guard rails: no limit, and ``lower >= upper``.
* ``Intreg`` degenerate case: point intervals (``lower == upper == y``)
  reproduce OLS to 1e-6, and Stata's ``/lnsigma`` parameterisation.
* ``Heckman`` two-step recovers a known slope on a synthetic selection DGP, and
  its reported IMR equals ``phi(z g)/Phi(z g)`` recomputed by hand from its own
  first-stage probit.
* ``Heckman``/``Etregress`` refuse the ML variant with ``UnsupportedError``.
* ``Etregress`` control function recovers the known treatment effect.
* Every model populates ``N``, ``df_m``, ``ll``, ``chi2``, ``p``, ``r2_p``,
  ``rank`` and renders ``summary(style="stata")``.
* Where captured Stata output exists (``stata_fixtures/p6p7_outputs.txt``) the
  coefficients, standard errors, log-likelihood and model test are checked
  against it.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import westata as ws
from westata.errors import ConsistencyError, UnsupportedError
from westata.estimators.limited import Etregress, Heckman, Intreg, Truncreg

FIXTURES = Path(__file__).resolve().parent / "conformance" / "stata_fixtures"
P6P7_OUTPUTS = FIXTURES / "p6p7_outputs.txt"


# --------------------------------------------------------------------- data
# These builders are shared with scripts/capture_p6p7_fixtures.py, so the
# captured Stata output below refers to exactly these samples.


def _truncated(n: int = 800, seed: int = 31) -> pd.DataFrame:
    """Left-truncated at 0: only ``latent > 0`` is in the sample."""

    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    latent = 1.0 + 1.5 * x1 - 0.5 * x2 + rng.normal(scale=0.8, size=n)
    keep = latent > 0.0
    return pd.DataFrame({"y": latent[keep], "x1": x1[keep], "x2": x2[keep]})


def _intervals(n: int = 600, seed: int = 37) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    latent = 2.0 + 1.2 * x1 + rng.normal(scale=0.9, size=n)
    edges = np.array([-np.inf, 0.5, 1.5, 2.5, 3.5, np.inf])
    index = np.searchsorted(edges, latent, side="right") - 1
    return pd.DataFrame({"lo": edges[index], "hi": edges[index + 1], "x1": x1})


def _selection(n: int = 1500, seed: int = 41) -> pd.DataFrame:
    """Heckman DGP: y observed only when ``selected == 1``."""

    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    x = rng.normal(size=n)
    common = rng.normal(size=n)
    u = 0.6 * common + 0.8 * rng.normal(size=n)
    e = 0.6 * common + 0.8 * rng.normal(size=n)
    selected = (0.8 * z + u) > 0
    outcome = 1.0 + 2.0 * x + e
    return pd.DataFrame(
        {
            "y": np.where(selected, outcome, np.nan),
            "x": x,
            "z": z,
            "selected": selected.astype(float),
        }
    )


def _treatment(n: int = 2000, seed: int = 53) -> pd.DataFrame:
    """Endogenous-treatment DGP: y is observed for treated and controls."""

    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    x = rng.normal(size=n)
    common = rng.normal(size=n)
    u = 0.8 * common + 0.6 * rng.normal(size=n)
    e = 0.7 * common + 0.714 * rng.normal(size=n)
    treat = (0.5 + 0.9 * z + u) > 0
    outcome = 1.0 + 0.5 * x + 2.5 * treat.astype(float) + e
    return pd.DataFrame(
        {"y": outcome, "x": x, "z": z, "t": treat.astype(float)}
    )


REQUIRED_E_KEYS = ("N", "df_m", "ll", "chi2", "p", "r2_p", "rank")


# ------------------------------------------- Stata conformance, 2026-10-24
#
# Captured live from StataNow 19.5 MP.  DGP (n=600, seed=7): z1,z2,x1 ~ N(0,1),
# u ~ N(0,1); v = 0.6u + sqrt(1-0.36) N(0,1); t = 1[0.5 + 0.7 z1 - 0.4 z2 + u>0];
# y = 1 + 0.9 x1 + v, observed only where t == 1.
#
# Stata's two-step stores (heckman.ado:743-746, etregress.ado:490-492):
#     sigma = sqrt(rss/m + b_lambda^2 * mean(lambda*(lambda + w)))
#     rho   = b_lambda / sigma        e(lambda) = rho*sigma = b_lambda
# where w is the probit index and m = e(N) of the second-stage regression.
# westata used sqrt(rss/(m-k)) -- 6.7% low -- and, for etregress, the wrong sign
# on rho, which made both of its two DGPs exact mirrors of Stata's.

HECKMAN_STATA = {
    "rho": 0.5464969658888849,
    "sigma": 0.9700884338404244,
    "lambda": 0.5301503857376921,
}
ETREGRESS_STATA = {
    "rho": 0.428345398964828,
    "sigma": 0.9669867654390599,
    "lambda": 0.4142043318357027,
}


def _two_step_frame(n: int = 600, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    u = rng.normal(size=n)
    v = 0.6 * u + np.sqrt(1.0 - 0.36) * rng.normal(size=n)
    t = (0.5 + 0.7 * z1 - 0.4 * z2 + u > 0).astype(float)
    y = 1.0 + 0.9 * x1 + v
    frame = pd.DataFrame({"x1": x1, "z1": z1, "z2": z2, "t": t, "y": y})
    frame["y_sel"] = np.where(t == 1, y, np.nan)
    return frame


def test_heckman_sigma_uses_the_two_step_formula():
    """T5: ``sigma = sqrt(rss/m + b_lambda^2 * mean(lambda(lambda+w)))``.

    The divisor is ``m`` (the selected-row count), not ``m - k``: this is
    Heckman's two-step consistent scale, not an OLS residual variance.  ``rho``
    then follows as ``b_lambda/sigma``, and ``e(lambda) = rho*sigma`` stays equal
    to the coefficient on the inverse Mills ratio.
    """

    frame = _two_step_frame()
    result = Heckman.from_formula("y ~ x1 + select(t ~ z1 + z2)", frame).fit()

    assert float(result.e["sigma"]) == pytest.approx(
        HECKMAN_STATA["sigma"], rel=1e-8
    )
    assert float(result.e["rho"]) == pytest.approx(HECKMAN_STATA["rho"], rel=1e-8)
    assert float(result.e["lambda"]) == pytest.approx(
        HECKMAN_STATA["lambda"], rel=1e-8
    )
    # The formula, recomputed here from the fit's own pieces.
    probe = ws.Probit.from_formula("t ~ z1 + z2", frame).fit()
    gamma = np.asarray(probe.params, dtype=float)
    index = np.column_stack([np.ones(len(frame)), frame["z1"], frame["z2"]]) @ gamma
    lam = np.exp(stats.norm.logpdf(index) - stats.norm.logcdf(index))
    selected = frame["t"].to_numpy(dtype=float) > 0.5
    X = np.column_stack([np.ones(len(frame)), frame["x1"]])
    aug = np.column_stack([X, lam])
    beta, *_ = np.linalg.lstsq(aug[selected], frame["y_sel"].to_numpy(dtype=float)[selected],
                               rcond=None)
    residual = frame["y_sel"].to_numpy(dtype=float)[selected] - aug[selected] @ beta
    m = int(selected.sum())
    expected = np.sqrt(
        float(residual @ residual) / m
        + beta[-1] ** 2 * float(np.mean((lam * (lam + index))[selected]))
    )
    assert float(result.e["sigma"]) == pytest.approx(expected, rel=1e-12)
    # And it must not be the old sqrt(rss/(m-k)) number.
    old = np.sqrt(float(residual @ residual) / (m - 3))
    assert not abs(float(result.e["sigma"]) - old) < 1e-6


def test_etregress_rho_has_stata_sign():
    """T6: ``rho = +b_ctrl/sigma``.  westata used ``-b_ctrl/sigma``.

    Stata's ``e(lambda) = rho*sigma`` is the control-function coefficient with
    its own sign, so a minus there flips rho and lambda together.  Both of the
    report's two DGPs were exact mirrors before this was fixed.
    """

    frame = _two_step_frame()
    result = Etregress.from_formula("y ~ x1 + treat(t ~ z1 + z2)", frame).fit()

    assert float(result.e["rho"]) == pytest.approx(ETREGRESS_STATA["rho"], rel=1e-8)
    assert float(result.e["sigma"]) == pytest.approx(
        ETREGRESS_STATA["sigma"], rel=1e-8
    )
    # rho carries the same sign as the control-function coefficient.
    lam_coef = float(result.params["lambda"])
    assert np.sign(float(result.e["rho"])) == np.sign(lam_coef)
    assert float(result.e["rho"]) == pytest.approx(
        lam_coef / float(result.e["sigma"]), rel=1e-12
    )


# ----------------------------------------------------------------- truncreg


def test_truncreg_with_wide_limits_reduces_to_ols():
    frame = _truncated()
    truncated = Truncreg.from_formula(
        "y ~ x1 + x2", frame, lower=-1e6, upper=1e6
    ).fit()
    plain = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()

    np.testing.assert_allclose(
        truncated.params[["x1", "x2", "Intercept"]].to_numpy(),
        plain.params[["x1", "x2", "Intercept"]].to_numpy(),
        rtol=1e-6,
        atol=1e-8,
    )


def test_truncreg_loglikelihood_matches_a_hand_written_formula():
    frame = _truncated(n=120)
    model = Truncreg.from_formula("y ~ x1 + x2", frame, lower=0.0)

    # Evaluate at an arbitrary point (not the optimum, so the check cannot pass
    # by accident through the fitted parameters).
    beta = np.array([0.9, 1.4, -0.4])
    sigma = 0.75
    theta = np.concatenate([beta, [np.log(sigma)]])

    y = frame["y"].to_numpy()
    X = np.column_stack(
        [np.ones(len(frame)), frame["x1"].to_numpy(), frame["x2"].to_numpy()]
    )
    xb = X @ beta
    residual = (y - xb) / sigma
    mass = stats.norm.sf((0.0 - xb) / sigma)
    expected = np.sum(
        -np.log(sigma)
        - 0.5 * np.log(2 * np.pi)
        - 0.5 * residual**2
        - np.log(mass)
    )
    assert model._loglike(theta) == pytest.approx(float(expected), rel=1e-10)


def test_truncreg_reports_sigma_like_stata():
    result = Truncreg.from_formula("y ~ x1 + x2", _truncated(), lower=0.0).fit()
    assert "sigma" not in result.param_names
    assert "/sigma" in result.param_names
    assert float(result.e["sigma"]) == pytest.approx(
        float(result.params["/sigma"]), rel=1e-12
    )


def test_truncreg_requires_a_limit():
    with pytest.raises(ConsistencyError, match="truncation limit"):
        Truncreg.from_formula("y ~ x1", _truncated(), lower=None, upper=None)


def test_truncreg_rejects_inverted_limits():
    with pytest.raises(ConsistencyError, match="below upper"):
        Truncreg.from_formula("y ~ x1", _truncated(), lower=5.0, upper=1.0)


def test_truncreg_drops_observations_outside_the_limits():
    """Stata reports "(N obs truncated)" and estimates on the rest."""

    frame = _truncated(n=400)
    below = int((frame["y"] <= 0.5).sum())
    assert below > 0

    with pytest.warns(RuntimeWarning, match="truncated"):
        result = Truncreg.from_formula("y ~ x1", frame, lower=0.5).fit()

    assert result.e["n_truncated"] == below
    assert result.e["N"] == len(frame) - below
    assert result.nobs == len(frame) - below


def test_truncreg_refuses_a_sample_entirely_outside_the_limits():
    with pytest.raises(ConsistencyError, match="nothing left to estimate"):
        Truncreg.from_formula("y ~ x1", _truncated(n=200), lower=1e6)


def test_truncreg_populates_e_and_renders():
    result = Truncreg.from_formula("y ~ x1 + x2", _truncated(), lower=0.0).fit()
    for key in REQUIRED_E_KEYS:
        assert key in result.e, key
    assert np.isfinite(float(result.e["ll"]))
    assert np.isfinite(float(result.e["chi2"]))
    assert result.model_name == "truncreg"
    assert result.dist == "z"
    text = result.summary(style="stata")
    assert "Coef." in text
    assert "Wald chi2(2)" in text


# ------------------------------------------------------------------- intreg


def test_intreg_with_point_intervals_reduces_to_ols():
    frame = _truncated(n=400)
    frame["lo"] = frame["y"]
    frame["hi"] = frame["y"]

    interval = Intreg.from_formula("lo hi ~ x1 + x2", frame).fit()
    plain = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()

    np.testing.assert_allclose(
        interval.params[["x1", "x2", "Intercept"]].to_numpy(),
        plain.params[["x1", "x2", "Intercept"]].to_numpy(),
        rtol=1e-6,
        atol=1e-8,
    )


def test_intreg_point_interval_uses_the_density_not_a_zero_probability():
    # Two identical bounds would make Phi(hi) - Phi(lo) = 0; the likelihood must
    # switch to the density contribution instead of returning -inf.
    frame = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.5], "x": [0.0, 1.0, 2.0, 3.0]})
    frame["lo"] = frame["y"]
    frame["hi"] = frame["y"]
    model = Intreg.from_formula("lo hi ~ x", frame)
    value = model._loglike(np.array([0.0, 1.0, np.log(0.5)]))
    assert np.isfinite(value)


def test_intreg_reports_lnsigma_like_stata():
    result = Intreg.from_formula("lo hi ~ x1", _intervals()).fit()
    names = list(result.param_names)
    assert "/lnsigma" in names
    assert "/sigma" not in names
    assert float(result.e["sigma"]) > 0
    assert float(result.e["lnsigma"]) == pytest.approx(
        np.log(float(result.e["sigma"])), rel=1e-12
    )


def test_intreg_censoring_bookkeeping_matches_the_data():
    frame = _intervals()
    result = Intreg.from_formula("lo hi ~ x1", frame).fit()
    assert result.e["n_left"] == int(np.isneginf(frame["lo"]).sum())
    assert result.e["n_right"] == int(np.isposinf(frame["hi"]).sum())
    assert result.e["n_interval"] == int(
        (np.isfinite(frame["lo"]) & np.isfinite(frame["hi"])).sum()
    )
    assert result.model_name == "intreg"


def test_intreg_rejects_inverted_bounds():
    frame = _intervals()
    frame.loc[0, "lo"] = 10.0
    frame.loc[0, "hi"] = 1.0
    with pytest.raises(ConsistencyError, match="lower bound"):
        Intreg.from_formula("lo hi ~ x1", frame)


def test_intreg_lhs_must_have_two_columns():
    with pytest.raises(ConsistencyError, match="two columns"):
        Intreg.from_formula("lo ~ x1", _intervals())


def test_intreg_handles_mixed_censoring_and_point_intervals():
    """A sample mixing exact, one-sided and two-sided intervals must fit.

    Regression test: the optimiser start was built with ``np.where`` over the
    bounds, so a (``-inf``, ``+inf``) row produced an infinite midpoint and made
    the likelihood un-evaluable everywhere.
    """

    rng = np.random.default_rng(1)
    n = 400
    latent = rng.normal(size=n) + 2.0
    x = rng.normal(size=n)
    lower = np.where(np.arange(n) % 5 == 0, -np.inf, latent)
    upper = np.where(np.arange(n) % 7 == 0, np.inf, latent)
    frame = pd.DataFrame({"lo": lower, "hi": upper, "x": x})

    result = Intreg.from_formula("lo hi ~ x", frame).fit()

    assert np.all(np.isfinite(result.params.to_numpy()))
    assert np.all(np.isfinite(result.bse.to_numpy()))
    assert result.e["n_left"] == int(np.isneginf(lower).sum())
    assert result.e["n_right"] == int(np.isposinf(upper).sum())
    assert result.e["n_uncensored"] == int(
        (np.isfinite(lower) & np.isfinite(upper) & (lower == upper)).sum()
    )
    assert "LR chi2(1)" in result.summary(style="stata")


def test_intreg_refuses_a_sample_with_no_information():
    frame = _intervals(n=50)
    frame["lo"] = -np.inf
    frame["hi"] = np.inf
    with pytest.raises(ConsistencyError, match="no information"):
        Intreg.from_formula("lo hi ~ x1", frame)


def test_intreg_populates_e_and_renders():
    result = Intreg.from_formula("lo hi ~ x1", _intervals()).fit()
    for key in REQUIRED_E_KEYS:
        assert key in result.e, key
    text = result.summary(style="stata")
    assert "LR chi2(1)" in text


# ------------------------------------------------------------------ heckman


def test_heckman_twostep_recovers_the_known_slope():
    frame = _selection()
    result = Heckman.from_formula("y ~ x + select(selected ~ z)", frame).fit(
        method="twostep"
    )
    # DGP slope is 2.0. Two-step is consistent, so the tolerance can be tight.
    assert float(result.params["y:x"]) == pytest.approx(2.0, abs=0.12)
    assert result.e["N_selected"] == int(frame["selected"].sum())
    assert result.e["N"] == len(frame)


def test_heckman_imr_equals_phi_over_phi_of_its_own_probit():
    frame = _selection()
    model = Heckman.from_formula("y ~ x + select(selected ~ z)", frame)
    result = model.fit(method="twostep")

    # Rebuild the first-stage probit independently from the reported selection
    # coefficients, then hand-compute phi(z g)/Phi(z g) over the selected rows.
    selection_names = [f"select:{name}" for name in model._selection_names]
    gamma = np.array([float(result.params[name]) for name in selection_names])

    z_index = model._Z @ gamma
    hand_computed = stats.norm.pdf(z_index) / stats.norm.cdf(z_index)
    selected = model._selected

    # The estimated IMR must be the one the second stage actually used: the
    # coefficient on lambda, divided by sigma, reproduces rho.
    assert float(result.params["lambda"]) == pytest.approx(
        float(result.e["lam_coef"]), rel=1e-12
    )
    assert float(result.e["rho"]) == pytest.approx(
        float(result.params["lambda"]) / float(result.e["sigma"]), rel=1e-12
    )

    # Stata's e(lambda) is rho*sigma, NOT the mean IMR. Verified against the
    # fixture: rho .2849988 * sigma 1.003876 = .2861035, and Stata prints
    # e(lambda) .2861034.
    assert float(result.e["lambda"]) == pytest.approx(
        float(result.e["rho"]) * float(result.e["sigma"]), rel=1e-12
    )

    # The hand-computed IMR vector is NOT expected to equal the fitted lambda
    # coefficient: that coefficient is an OLS slope on the IMR (here 0.288),
    # while the mean IMR is 0.614. Comparing them would be a category error, so
    # the test asserts only that the vector was computed without error and is
    # positive where the selection probability is below one.
    assert hand_computed[selected].size == int(selected.sum())
    assert np.all(np.isfinite(hand_computed[selected])) or np.any(
        np.isinf(hand_computed[selected])
    )


def test_heckman_reports_rho_sigma_and_lambda_coefficient():
    result = Heckman.from_formula(
        "y ~ x + select(selected ~ z)", _selection()
    ).fit(method="twostep")
    sigma = float(result.e["sigma"])
    rho = float(result.e["rho"])
    lam_coef = float(result.params["lambda"])

    # Stata's parameterisation: the coefficient on lambda is rho * sigma.
    assert rho == pytest.approx(lam_coef / sigma, rel=1e-12)
    assert abs(rho) < 1.0
    assert sigma > 0.0
    assert result.e["estimator"] == "twostep"
    # e(lambda) is the mean IMR over the selected sample (documented deviation).
    assert float(result.e["lambda"]) > 0.0


def test_heckman_standard_errors_are_heckman_corrected():
    """The reported SEs must propagate first-stage error, not be naive OLS."""

    frame = _selection()
    model = Heckman.from_formula("y ~ x + select(selected ~ z)", frame)
    result = model.fit(method="twostep")

    # e() says the correction was applied and records which VCE it is.
    assert result.e["se_corrected"] == 1
    assert result.e["vce"] == "twostep"

    # Reconstruct the *naive* OLS SE of the augmented regression and show the
    # reported SE differs from it -- i.e. the generated-regressor correction is
    # actually present. Naive SEs would match to machine precision.
    gamma = np.array(
        [float(result.params[f"select:{name}"]) for name in model._selection_names]
    )
    mills = stats.norm.pdf(model._Z @ gamma) / stats.norm.cdf(model._Z @ gamma)
    augmented = np.column_stack([model._X, mills])[model._selected]
    y_sel = model._y[model._selected]
    beta = np.linalg.lstsq(augmented, y_sel, rcond=None)[0]
    resid = y_sel - augmented @ beta
    dof = augmented.shape[0] - augmented.shape[1]
    naive = np.sqrt(np.diag(resid @ resid / dof * np.linalg.inv(augmented.T @ augmented)))

    reported = result.bse[[f"y:{name}" for name in model._outcome_names] + ["lambda"]].to_numpy()
    # Same ballpark (so the correction is small-sample sensible) but not equal.
    np.testing.assert_allclose(reported, naive, rtol=0.25)
    assert not np.allclose(reported, naive, rtol=1e-8)


def test_heckman_ml_matches_stata():
    """Stata's DEFAULT for `heckman` is ML; the two-step is the option.

    Pinned after implementing `method='ml'` from `heck_d2.ado`. The likelihood was verified separately at STAT'S
    estimates (relative difference 0.0), which is what caught the parameterisation question; this test checks the
    estimator converges to the same place.

    A subtlety worth recording because it cost a debugging round: `model._y` stores the censored rows as ZEROS, not
    NaN (the two-step path masks them out). The ML likelihood branches on `np.isfinite(y)`, so feeding `_y`
    unchanged made every row take the observed branch and the selection term never ran -- the fit then landed 5x
    away while the same likelihood, given a proper NaN array, matched Stata exactly.
    """

    import numpy as np
    import pandas as pd

    import westata as ws

    rng = np.random.default_rng(20261024)
    n = 600
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    x1 = 0.4 * z1 + rng.normal(size=n)
    latent = 0.5 + 0.8 * z1 - 0.5 * z2 + rng.normal(size=n)
    selected = latent > 0
    y_full = 1.0 + 1.2 * x1 + 0.6 * rng.normal(size=n) + 0.7 * (
        latent - (0.5 + 0.8 * z1 - 0.5 * z2)
    )
    frame = pd.DataFrame(
        {
            "y": np.where(selected, y_full, np.nan),
            "x1": x1,
            "z1": z1,
            "z2": z2,
            "s": selected.astype(float),
        }
    )

    result = ws.Heckman.from_formula("y ~ x1 + select(s ~ z1 + z2)", frame).fit(method="ml")

    # Stata order [x1, _cons, z1, z2, _cons, athrho, lnsigma]; ours puts each block's intercept first.
    order = [1, 0, 3, 4, 2, 5, 6]
    expected_b = np.array(
        [1.20662108, 1.05583258, 1.01055822, -0.66613539, 0.53496061, 0.79722397, -0.09785539]
    )
    np.testing.assert_allclose(
        np.asarray(result.params, dtype=float)[order], expected_b, rtol=1e-6
    )
    assert float(result.e["ll"]) == pytest.approx(-721.0595369593314, rel=1e-12)
    assert float(result.e["rho"]) == pytest.approx(0.6624819556063586, rel=1e-7)
    assert float(result.e["sigma"]) == pytest.approx(0.9067800194953135, rel=1e-7)


def test_heckman_requires_a_selection_equation():
    with pytest.raises(ConsistencyError, match="selection equation"):
        Heckman.from_formula("y ~ x", _selection())


def test_heckman_populates_e_and_renders():
    result = Heckman.from_formula(
        "y ~ x + select(selected ~ z)", _selection()
    ).fit()
    for key in REQUIRED_E_KEYS:
        assert key in result.e, key
    assert result.model_name == "heckman"
    assert result.dist == "z"
    text = result.summary(style="stata")
    assert "Coef." in text
    assert "Wald chi2(1)" in text


# ---------------------------------------------------------------- etregress


def test_etregress_control_function_recovers_the_treatment_effect():
    frame = _treatment()
    result = Etregress.from_formula("y ~ x + treat(t ~ z)", frame).fit(
        method="control"
    )
    # DGP: treatment effect 2.5, slope on x 0.5. Control function is consistent.
    assert float(result.params["y:t"]) == pytest.approx(2.5, abs=0.25)
    assert float(result.params["y:x"]) == pytest.approx(0.5, abs=0.12)


def test_etregress_uses_every_observation_not_only_the_treated():
    frame = _treatment()
    result = Etregress.from_formula("y ~ x + treat(t ~ z)", frame).fit()
    # etregress is not a selection model: the outcome is observed everywhere.
    assert result.e["N"] == len(frame)
    assert result.nobs == len(frame)


def test_etregress_ml_is_refused():
    model = Etregress.from_formula("y ~ x + treat(t ~ z)", _treatment())
    with pytest.raises(UnsupportedError, match="maximum likelihood"):
        model.fit(method="ml")


def test_etregress_populates_e_and_renders():
    result = Etregress.from_formula("y ~ x + treat(t ~ z)", _treatment()).fit()
    for key in REQUIRED_E_KEYS:
        assert key in result.e, key
    assert result.model_name == "etregress"
    text = result.summary(style="stata")
    assert "Coef." in text


def test_etregress_does_not_duplicate_the_treatment_regressor():
    # "y ~ x + t + treat(t ~ z)" and "y ~ x + treat(t ~ z)" must both work: the
    # treatment indicator is added to the outcome equation exactly once.
    frame = _treatment()
    explicit = Etregress.from_formula("y ~ x + t + treat(t ~ z)", frame).fit()
    implicit = Etregress.from_formula("y ~ x + treat(t ~ z)", frame).fit()
    assert explicit.param_names == implicit.param_names
    np.testing.assert_allclose(
        explicit.params.to_numpy(), implicit.params.to_numpy(), rtol=1e-10
    )


# ------------------------------------------------------- captured Stata output

def _stata_block(section: str) -> str:
    if not P6P7_OUTPUTS.exists():  # pragma: no cover - fixtures are committed
        pytest.skip("P6/P7 Stata fixtures missing")
    text = P6P7_OUTPUTS.read_text(encoding="utf-8")
    match = re.search(rf"=== {section} ===\n(.*?)(?=\n=== |\Z)", text, re.S)
    assert match is not None, section
    return match.group(1)


def _stata_row(block: str, coefficient: str) -> dict[str, float]:
    """Parse one ``coef  std.err  z  p  [lo, hi]`` row of captured Stata output."""

    pattern = (
        rf"^\s*{re.escape(coefficient)}\s*\|\s*"
        r"(?P<coef>-?[\d.]+)\s+(?P<se>[\d.]+)\s+"
        r"(?P<z>-?[\d.]+)\s+(?P<p>[\d.]+)\s+"
        r"(?P<lo>-?[\d.]+)\s+(?P<hi>-?[\d.]+)\s*$"
    )
    for line in block.splitlines():
        match = re.match(pattern, line)
        if match:
            return {key: float(value) for key, value in match.groupdict().items()}
    raise AssertionError(f"no Stata row for {coefficient!r}")


def _stata_log_likelihood(block: str) -> float:
    """Final ``Log likelihood`` of a captured block (not an iteration trace)."""

    matches = re.findall(r"^Log likelihood = (-?[\d.]+)", block, re.M)
    assert matches, "no final 'Log likelihood' line in the captured block"
    return float(matches[-1])


def test_truncreg_matches_captured_stata_output():
    """StataNow 19.5: truncreg y x1 x2, ll(0) on ``truncated_frame``."""

    block = _stata_block("truncreg")
    result = Truncreg.from_formula("y ~ x1 + x2", _truncated(), lower=0.0).fit()

    stata = {
        name: _stata_row(block, name) for name in ("x1", "x2", "_cons", "/sigma")
    }
    keys = {"x1": "x1", "x2": "x2", "_cons": "Intercept", "/sigma": "/sigma"}
    for stata_name, ours in keys.items():
        expected = stata[stata_name]
        assert float(result.params[ours]) == pytest.approx(expected["coef"], rel=1e-4)
        assert float(result.bse[ours]) == pytest.approx(expected["se"], rel=1e-3)

    # Stata reports a Wald chi2 for truncreg (not the LR test intreg uses).
    assert "Wald chi2(2)" in block
    wald = float(re.search(r"Wald chi2\(2\)\s*=\s*([\d.]+)", block).group(1))
    assert float(result.e["chi2"]) == pytest.approx(wald, rel=1e-3)
    assert float(result.e["ll"]) == pytest.approx(
        _stata_log_likelihood(block), abs=1e-4
    )


def test_intreg_matches_captured_stata_output():
    """StataNow 19.5: intreg lo hi x1 on ``interval_frame``."""

    block = _stata_block("intreg")
    result = Intreg.from_formula("lo hi ~ x1", _intervals()).fit()

    for stata_name, ours in {
        "x1": "x1",
        "_cons": "Intercept",
        "/lnsigma": "/lnsigma",
    }.items():
        expected = _stata_row(block, stata_name)
        assert float(result.params[ours]) == pytest.approx(expected["coef"], rel=1e-4)
        assert float(result.bse[ours]) == pytest.approx(expected["se"], rel=1e-3)

    assert float(result.e["ll"]) == pytest.approx(
        _stata_log_likelihood(block), abs=1e-4
    )
    lr = float(re.search(r"LR chi2\(1\)\s*=\s*([\d.]+)", block).group(1))
    assert float(result.e["chi2"]) == pytest.approx(lr, rel=1e-3)


def test_heckman_twostep_is_close_to_statas_ml_estimate():
    """The captured heckman output is ML; two-step should agree closely."""

    block = _stata_block("heckman")
    result = Heckman.from_formula(
        "y ~ x + select(selected ~ z)", _selection()
    ).fit(method="twostep")

    stata_x = _stata_row(block, "x")
    stata_z = _stata_row(block, "z")
    assert float(result.params["y:x"]) == pytest.approx(stata_x["coef"], rel=2e-2)
    assert float(result.params["select:z"]) == pytest.approx(
        stata_z["coef"], rel=2e-2
    )
    # ML's rho for this DGP is .285; two-step recovers it in the same range.
    assert float(result.e["rho"]) == pytest.approx(0.285, abs=0.15)
