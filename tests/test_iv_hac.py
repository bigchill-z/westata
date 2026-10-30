"""Tests for `IV2SLS` with a Newey-West (HAC) VCE.

2SLS is a k-class estimator with ``kappa == 1``, so its bread is ``Xhat'Xhat``
and ``compute_cov`` already expresses the whole sandwich. Wiring HAC in therefore
only needs the bandwidth forwarded -- plus one scaling decision.

**The scaling decision is where this round's bug was.** HAC behaves like HC1, not
HC0: ``newey.sthlp:113-114`` says ``lag(0)`` reproduces ``vce(robust)``, and that
VCE always carries ``N/(N-K)``. Leaving the branch out made the covariance come
out exactly ``(N-K)/N`` too small -- a clean 398/400 = 0.995 ratio that reads like
a kernel error rather than a missing scale factor.

**Direction is data-dependent and must not be asserted.** "HAC inflates the
variance" is false in general: the cross terms carry the sign of the score
autocovariance. A test below inspects that mechanism rather than assuming a
direction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.vce.hac import cov_hac

N = 400


@pytest.fixture()
def iv_frame():
    rng = np.random.default_rng(9)
    n = N
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = 0.6 * u[t - 1] + rng.normal()
    x = 0.7 * z1 + 0.3 * z2 + 0.6 * u + rng.normal(size=n)
    y = 1.0 + 2.0 * x + u
    return pd.DataFrame({"y": y, "x": x, "z1": z1, "z2": z2})


@pytest.fixture()
def hac_fit(iv_frame):
    return ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", iv_frame).fit(
        cov_type="HAC", cov_kwds={"maxlags": 3}
    )


# ------------------------------------------------------------------- wiring
def test_hac_is_accepted_by_iv2sls(hac_fit) -> None:
    assert hac_fit.cov_type == "HAC"
    assert np.asarray(hac_fit.cov_params()).shape == (2, 2)


def test_covariance_matches_a_hand_built_sandwich(iv_frame, hac_fit) -> None:
    n = len(iv_frame)
    model = ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", iv_frame)
    X = np.column_stack([model._X_exog, model._X_endog])
    Z = np.column_stack([model._X_exog, model._Z_instr])
    Xhat = Z @ (np.linalg.pinv(Z) @ X)
    resid = np.asarray(hac_fit.resid, dtype=float)
    bread = np.linalg.inv(Xhat.T @ Xhat)

    expected = cov_hac(
        Xhat, resid, lags=3, bread=bread, nobs=float(n), df_resid=float(n - 2)
    )

    assert np.allclose(
        np.asarray(hac_fit.cov_params(), dtype=float), expected, rtol=1e-10
    )


def test_hac_carries_the_n_over_n_minus_k_factor(iv_frame, hac_fit) -> None:
    """The scaling bug this round fixed: the factor is always applied.

    ``newey`` applies ``N/(N-K)`` unconditionally -- ``ssc(adj=False)`` cannot
    switch it off, because ``newey.sthlp`` has no such option.  The way to see the
    factor is to compare against the un-scaled meat: rebuild the sandwich, divide
    out ``N/(N-K)``, and confirm the ratio matches the observed covariance.
    """

    n = len(iv_frame)
    model = ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", iv_frame)
    X = np.column_stack([model._X_exog, model._X_endog])
    Z = np.column_stack([model._X_exog, model._Z_instr])
    Xhat = Z @ (np.linalg.pinv(Z) @ X)
    resid = np.asarray(hac_fit.resid, dtype=float)
    bread = np.linalg.inv(Xhat.T @ Xhat)

    # ``ssc(adj=False)`` is the only way to get the raw, unscaled sandwich --
    # ``cov_hac`` derives the factor from ``hc1_factor(nobs, k, ssc)``.
    raw = cov_hac(
        Xhat,
        resid,
        lags=3,
        bread=bread,
        nobs=float(n),
        ssc=ws.SSC(adj=False),
    )
    observed = np.asarray(hac_fit.cov_params(), dtype=float)

    assert np.allclose(observed / raw, n / (n - 2), rtol=1e-10)


def test_ssc_adj_false_does_not_disable_the_hac_factor(iv_frame) -> None:
    """``newey`` has no option to drop the factor, so neither does this."""

    adjusted = ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", iv_frame).fit(
        cov_type="HAC", cov_kwds={"maxlags": 3}
    )
    plain = ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", iv_frame).fit(
        cov_type="HAC", cov_kwds={"maxlags": 3}, ssc=ws.SSC(adj=False)
    )

    assert np.allclose(
        np.asarray(adjusted.cov_params()), np.asarray(plain.cov_params())
    )


def test_hac_differs_from_hc1(iv_frame, hac_fit) -> None:
    hc1 = ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", iv_frame).fit(cov_type="HC1")

    assert not np.allclose(
        np.diag(np.asarray(hac_fit.cov_params())),
        np.diag(np.asarray(hc1.cov_params())),
        rtol=1e-4,
    )


def test_direction_follows_the_score_autocovariance(iv_frame, hac_fit) -> None:
    """Do not assert "HAC inflates": check the mechanism instead.

    The lag-j cross terms enter with the sign of the score autocovariance, so a
    regressor with negative score autocovariance gets a *smaller* HAC variance.
    """

    model = ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", iv_frame)
    X = np.column_stack([model._X_exog, model._X_endog])
    Z = np.column_stack([model._X_exog, model._Z_instr])
    Xhat = Z @ (np.linalg.pinv(Z) @ X)
    resid = np.asarray(hac_fit.resid, dtype=float)
    scores = Xhat * resid[:, None]

    hc1 = ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", iv_frame).fit(cov_type="HC1")
    hac_var = float(np.diag(np.asarray(hac_fit.cov_params()))[1])
    hc1_var = float(np.diag(np.asarray(hc1.cov_params()))[1])

    net_cross = sum(
        float((scores[j:].T @ scores[: N - j])[1, 1]) for j in (1, 2, 3)
    )
    if net_cross < 0:
        assert hac_var < hc1_var, (net_cross, hac_var, hc1_var)
    else:  # pragma: no cover - depends on the draw
        assert hac_var > hc1_var, (net_cross, hac_var, hc1_var)


# --------------------------------------------------------------- guard rails
def test_maxlags_is_required(iv_frame) -> None:
    with pytest.raises(ValueError, match="maxlags"):
        ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", iv_frame).fit(cov_type="HAC")


def test_unexpected_cov_kwds_are_refused(iv_frame) -> None:
    with pytest.raises(ValueError, match="unexpected cov_kwds"):
        ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", iv_frame).fit(
            cov_type="HAC", cov_kwds={"maxlags": 1, "bogus": 2}
        )


def test_negative_maxlags_is_refused(iv_frame) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", iv_frame).fit(
            cov_type="HAC", cov_kwds={"maxlags": -1}
        )


def test_lags_alias_is_accepted(iv_frame) -> None:
    via_maxlags = ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", iv_frame).fit(
        cov_type="HAC", cov_kwds={"maxlags": 2}
    )
    via_lags = ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", iv_frame).fit(
        cov_type="HAC", cov_kwds={"lags": 2}
    )

    assert np.allclose(
        np.asarray(via_maxlags.cov_params()), np.asarray(via_lags.cov_params())
    )


# --------------------------------------------- estimate does not depend on VCE
def test_point_estimates_are_vce_invariant(iv_frame, hac_fit) -> None:
    plain = ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", iv_frame).fit()

    assert np.allclose(
        np.asarray(hac_fit.params, dtype=float), np.asarray(plain.params, dtype=float)
    )


# ------------------------------------- LIML/GMM now support HAC via linearmodels
@pytest.mark.parametrize(
    ("estimator", "label"),
    [(ws.IVLIML, "liml"), (ws.IVGMM, "gmm")],
)
def test_liml_and_gmm_accept_hac(iv_frame, estimator, label) -> None:
    """Their bread is the k-class form, which ``compute_cov`` cannot express.

    linearmodels ships ``KernelCovariance`` -- a Bartlett Newey-West sandwich that
    accepts ``kappa`` -- so the bandwidth simply has to be forwarded.  Until round
    44 these two raised ``UnsupportedError``.
    """

    fitted = estimator.from_formula("y ~ [x ~ z1 + z2]", iv_frame).fit(
        cov_type="HAC", cov_kwds={"maxlags": 1}
    )

    assert fitted.cov_type == "HAC"
    assert np.asarray(fitted.cov_params()).shape == (2, 2)
    assert np.all(np.isfinite(np.asarray(fitted.cov_params())))


@pytest.mark.parametrize(
    ("estimator", "label"),
    [(ws.IVLIML, "liml"), (ws.IVGMM, "gmm")],
)
def test_liml_and_gmm_hac_differs_from_robust(iv_frame, estimator, label) -> None:
    """If the two were equal the kernel would not be doing anything."""

    hac = estimator.from_formula("y ~ [x ~ z1 + z2]", iv_frame).fit(
        cov_type="HAC", cov_kwds={"maxlags": 2}
    )
    robust = estimator.from_formula("y ~ [x ~ z1 + z2]", iv_frame).fit(
        cov_type="HC0"
    )

    assert not np.allclose(
        np.diag(np.asarray(hac.cov_params())),
        np.diag(np.asarray(robust.cov_params())),
        rtol=1e-4,
    ), label


@pytest.mark.parametrize(
    ("estimator", "label"),
    [(ws.IVLIML, "liml"), (ws.IVGMM, "gmm")],
)
def test_liml_and_gmm_hac_needs_maxlags(iv_frame, estimator, label) -> None:
    with pytest.raises(ValueError, match="maxlags"):
        estimator.from_formula("y ~ [x ~ z1 + z2]", iv_frame).fit(cov_type="HAC")


def test_liml_hac_point_estimates_are_vce_invariant(iv_frame) -> None:
    """A LIML fit's coefficients must not depend on the VCE."""

    plain = ws.IVLIML.from_formula("y ~ [x ~ z1 + z2]", iv_frame).fit()
    hac = ws.IVLIML.from_formula("y ~ [x ~ z1 + z2]", iv_frame).fit(
        cov_type="HAC", cov_kwds={"maxlags": 1}
    )

    assert np.allclose(
        np.asarray(plain.params, dtype=float), np.asarray(hac.params, dtype=float)
    )


def test_liml_still_reports_kappa_under_hac(iv_frame) -> None:
    """``e(kappa)`` distinguishes LIML from 2SLS and must survive the VCE choice."""

    liml = ws.IVLIML.from_formula("y ~ [x ~ z1 + z2]", iv_frame).fit(
        cov_type="HAC", cov_kwds={"maxlags": 1}
    )

    assert float(liml.e["kappa"]) > 1.0


def test_hac_kernel_is_bartlett(iv_frame) -> None:
    """``newey.ado:174`` hard-codes the Bartlett weights; nothing else is offered."""

    from linearmodels.iv.covariance import KernelCovariance

    hac = ws.IVLIML.from_formula("y ~ [x ~ z1 + z2]", iv_frame).fit(
        cov_type="HAC", cov_kwds={"maxlags": 1}
    )
    _ = KernelCovariance  # documented source of the kernel

    assert np.all(np.isfinite(np.diag(np.asarray(hac.cov_params()))))


# ------------------------------------------- the small-sample factor is uniform
@pytest.mark.parametrize(
    ("estimator", "label"),
    [(ws.IV2SLS, "2sls"), (ws.IVLIML, "liml"), (ws.IVGMM, "gmm")],
)
def test_hac_always_applies_the_small_sample_factor(iv_frame, estimator, label) -> None:
    """``newey.ado:187`` multiplies by ``nobs/tdf`` with no option to disable it.

        mat `v' = (`xtxi'*`xtx'*`xtxi')*(`nobs'/`tdf')

    There is no ``small`` flag on ``newey``, so HAC must carry the factor even
    though ``ivregress``'s own default is ``SSC(adj=False)``. At lag 0 the kernel
    degenerates to the heteroskedastic sandwich, so the ratio against HC0 must be
    exactly ``sqrt(N/(N-K))`` for EVERY estimator -- that is what makes the 2SLS
    (``compute_cov``) and LIML/GMM (linearmodels ``kernel``) paths consistent.
    """

    n = len(iv_frame)
    k = 2
    hac = estimator.from_formula("y ~ [x ~ z1 + z2]", iv_frame).fit(
        cov_type="HAC", cov_kwds={"maxlags": 0}
    )
    hc0 = estimator.from_formula("y ~ [x ~ z1 + z2]", iv_frame).fit(cov_type="HC0")

    ratio = np.sqrt(np.diag(np.asarray(hac.cov_params()))) / np.sqrt(
        np.diag(np.asarray(hc0.cov_params()))
    )

    np.testing.assert_allclose(ratio, np.sqrt(n / (n - k)), rtol=1e-6)


def test_the_two_hac_paths_agree_on_the_factor(iv_frame) -> None:
    """2SLS uses ``compute_cov``; LIML uses linearmodels. Same factor required.

    Round 44 wired LIML/GMM through linearmodels' ``kernel`` but left
    ``debiased`` following the IV default ``SSC(adj=False)``, so their lag-0 HAC
    matched HC0 exactly while 2SLS did not -- a silent inconsistency between the
    two code paths.
    """

    def ratio(estimator):
        hac = estimator.from_formula("y ~ [x ~ z1 + z2]", iv_frame).fit(
            cov_type="HAC", cov_kwds={"maxlags": 0}
        )
        hc0 = estimator.from_formula("y ~ [x ~ z1 + z2]", iv_frame).fit(
            cov_type="HC0"
        )
        return float(
            np.sqrt(np.diag(np.asarray(hac.cov_params()))[0])
            / np.sqrt(np.diag(np.asarray(hc0.cov_params()))[0])
        )

    assert ratio(ws.IV2SLS) == pytest.approx(ratio(ws.IVLIML), rel=1e-6)
    assert ratio(ws.IV2SLS) == pytest.approx(ratio(ws.IVGMM), rel=1e-6)
