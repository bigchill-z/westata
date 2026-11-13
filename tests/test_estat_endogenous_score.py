"""Tests for the endogeneity score tests (`robwool` / `acwool`).

Transcribed from ``ivregress_estat.ado:1187-1263``:

    uhat     = OLS residual of y on (exog + endog), nocons
    endhat_i = fit of endog_i on (exog + addlinst), nocons
    rhat_i   = residual of endhat_i on (exog + endog), nocons, times uhat
    robwool  = N - rss of (ones on rhat, nocons)
    acwool   = the same after prewhitening rhat with its own lags

The VCE decides which test is reported (``ivregress_estat.ado:1128-1145``):
unadjusted -> Durbin-Wu, robust -> ``robwool``, HAC -> ``acwool``, cluster ->
only ``regF`` (which needs the augmented regression and is not implemented), so a
clustered VCE is refused.

Two structural checks carry the weight:

* the robust score must **differ** from Durbin-Wu — they are different tests, so
  agreement would mean the routing was cosmetic;
* the HAC score must differ from the un-prewhitened one, or the prewhitening
  branch is dead code again (the failure mode round 28 fixed elsewhere).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import westata as ws

N = 500


@pytest.fixture()
def frame():
    rng = np.random.default_rng(9)
    n = N
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = 0.5 * u[t - 1] + rng.normal()
    x = 0.7 * z1 + 0.3 * z2 + 0.6 * u + rng.normal(size=n)
    y = 1.0 + 2.0 * x + u
    return pd.DataFrame({"y": y, "x": x, "z1": z1, "z2": z2})


def _hand(structure, resid, response, hac=False, lags=0):
    """Direct transcription of the ado."""

    exog = np.asarray(structure["exog"], dtype=float)
    endog = np.asarray(structure["endog"], dtype=float)
    instruments = np.asarray(structure["instruments"], dtype=float)
    names = [str(v) for v in structure["exog_names"]]
    constant_names = {"_cons", "intercept", "const"}
    has_const = any(v.strip().lower() in constant_names for v in names)
    included = [j for j, v in enumerate(names) if v.strip().lower() not in constant_names]
    n = resid.size

    def design(extra):
        parts = [np.ones(n)] if has_const else []
        if included:
            parts.append(exog[:, included])
        parts.extend(extra)
        return np.column_stack(parts)

    def residuals_of(y, matrix):
        beta, *_ = np.linalg.lstsq(matrix, y, rcond=None)
        return y - matrix @ beta

    def fitted_of(y, matrix):
        beta, *_ = np.linalg.lstsq(matrix, y, rcond=None)
        return matrix @ beta

    uhat = residuals_of(response, design([endog]))
    endhat = np.column_stack(
        [fitted_of(endog[:, c], design([instruments])) for c in range(endog.shape[1])]
    )
    rhat = np.column_stack(
        [
            residuals_of(endhat[:, c], design([endog])) * uhat
            for c in range(endog.shape[1])
        ]
    )

    if hac and lags > 0:
        lagged = np.column_stack(
            [rhat[lags - 1 - j : n - 1 - j] for j in range(lags)]
        )
        target = rhat[lags:]
        out = np.empty_like(target)
        for c in range(target.shape[1]):
            beta, *_ = np.linalg.lstsq(lagged, target[:, c], rcond=None)
            out[:, c] = target[:, c] - lagged @ beta
        rhat = out

    ones = np.ones(rhat.shape[0])
    beta, *_ = np.linalg.lstsq(rhat, ones, rcond=None)
    residual = ones - rhat @ beta
    return float(rhat.shape[0] - residual @ residual)


# --------------------------------------------------------------- routing
def test_unadjusted_reports_durbin_wu(frame) -> None:
    result = ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", frame).fit()
    rendered = result.estat_endogenous()

    assert hasattr(rendered, "durbin")
    assert hasattr(rendered, "wu")


def test_robust_reports_the_score_test(frame) -> None:
    result = ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", frame).fit(cov_type="HC1")
    rendered = result.estat_endogenous()

    assert rendered.test == "robwool"


def test_hac_reports_the_hac_score(frame) -> None:
    result = ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", frame).fit(
        cov_type="HAC", cov_kwds={"maxlags": 1}
    )
    rendered = result.estat_endogenous()

    assert rendered.test == "acwool"
    assert rendered.lags == 1


def test_clustered_vce_reports_regF(frame) -> None:
    """After a clustered VCE Stata reports ONLY ``regF`` (``ado:1134-1136``).

    Round 32 refused this case because ``regF`` did not exist; round 33
    implemented it, so the correct behaviour is to report it.
    """

    result = ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", frame).fit(
        cov_type="cluster", cov_kwds={"clusters": np.arange(N) % 50}
    )

    rendered = result.estat_endogenous()

    assert rendered.test == "regF"
    # The clustered denominator is G - 1, not N - K.
    assert rendered.df_denom == 49.0
    assert rendered.df_num == 1.0


# ----------------------------------------------------------- correctness
def test_robust_score_matches_the_transcription(frame) -> None:
    result = ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", frame).fit(cov_type="HC1")
    expected = _hand(
        result.iv_structure,
        np.asarray(result.resid),
        frame["y"].to_numpy(float),
    )

    assert result.estat_endogenous().statistic == pytest.approx(expected, rel=1e-10)


def test_hac_score_matches_the_transcription(frame) -> None:
    result = ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", frame).fit(
        cov_type="HAC", cov_kwds={"maxlags": 1}
    )
    expected = _hand(
        result.iv_structure,
        np.asarray(result.resid),
        frame["y"].to_numpy(float),
        hac=True,
        lags=1,
    )

    assert result.estat_endogenous().statistic == pytest.approx(expected, rel=1e-10)


def test_score_differs_from_durbin_wu(frame) -> None:
    """Different tests; agreement would mean the routing did nothing."""

    plain = ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", frame).fit()
    robust = ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", frame).fit(cov_type="HC1")

    assert not np.isclose(
        plain.estat_endogenous().durbin,
        robust.estat_endogenous().statistic,
        rtol=1e-3,
    )


def test_prewhitening_changes_the_score(frame) -> None:
    """Without this the prewhitening branch could be dead code."""

    result = ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", frame).fit(
        cov_type="HAC", cov_kwds={"maxlags": 1}
    )
    whitened = result.estat_endogenous().statistic
    unwhitened = _hand(
        result.iv_structure,
        np.asarray(result.resid),
        frame["y"].to_numpy(float),
        hac=True,
        lags=0,
    )

    assert not np.isclose(whitened, unwhitened, rtol=1e-6)


# ------------------------------------------------------------ statistics
def test_df_is_the_number_of_endogenous_regressors(frame) -> None:
    result = ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", frame).fit(cov_type="HC1")

    assert result.estat_endogenous().df == 1.0


def test_pvalue_is_the_chi2_upper_tail(frame) -> None:
    result = ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", frame).fit(cov_type="HC1")
    rendered = result.estat_endogenous()

    assert rendered.pvalue == pytest.approx(
        float(stats.chi2.sf(rendered.statistic, rendered.df)), rel=1e-12
    )


def test_statistic_is_non_negative(frame) -> None:
    result = ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", frame).fit(cov_type="HC1")

    assert result.estat_endogenous().statistic >= 0.0


# ----------------------------------------------------------- guard rails
def test_lags_requires_a_hac_vce(frame) -> None:
    """``ivregress_estat.ado:1055-1058``."""

    result = ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", frame).fit(cov_type="HC1")

    with pytest.raises(ValueError, match="only valid with models with HAC"):
        result.estat_endogenous(lags=1)


def test_hac_defaults_lags_to_one(frame) -> None:
    """``ivregress_estat.ado:1059-1063``."""

    result = ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", frame).fit(
        cov_type="HAC", cov_kwds={"maxlags": 2}
    )

    assert result.estat_endogenous().lags == 1


def test_r_names_follow_the_ado(frame) -> None:
    robust = ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", frame).fit(cov_type="HC1")
    hac = ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", frame).fit(
        cov_type="HAC", cov_kwds={"maxlags": 1}
    )

    assert set(robust.estat_endogenous().to_dict()) == {"r_score", "p_r_score"}
    assert set(hac.estat_endogenous().to_dict()) == {
        "hac_score",
        "p_hac_score",
        "lags",
    }


def test_output_names_the_test(frame) -> None:
    result = ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", frame).fit(cov_type="HC1")

    text = str(result.estat_endogenous())

    assert "Tests of endogeneity" in text
    assert "Robust score" in text


# ------------------------------------------------------------ regF specifics
def test_regF_matches_the_augmented_ols_wald_form(frame) -> None:
    """``RobReg`` runs ``ivregress 2sls`` with NO ``(endog = instr)`` block.

    Without that block the command has nothing to instrument and reduces to OLS,
    so the statistic is the ordinary Wald form on the fitted-endogenous block.
    """

    result = ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", frame).fit(
        cov_type="cluster", cov_kwds={"clusters": np.arange(N) % 50}
    )
    rendered = result.estat_endogenous()

    structure = result.iv_structure
    exog = np.asarray(structure["exog"], dtype=float)
    endog = np.asarray(structure["endog"], dtype=float)
    instruments = np.asarray(structure["instruments"], dtype=float)
    names = [str(v) for v in structure["exog_names"]]
    constant_names = {"_cons", "intercept", "const"}
    has_const = any(v.strip().lower() in constant_names for v in names)
    included = [
        j for j, v in enumerate(names) if v.strip().lower() not in constant_names
    ]
    n = len(frame)

    def design(extra):
        parts = [np.ones(n)] if has_const else []
        if included:
            parts.append(exog[:, included])
        parts.extend(extra)
        return np.column_stack(parts)

    projection = design([instruments])
    endhat = np.empty((n, endog.shape[1]))
    for c in range(endog.shape[1]):
        beta, *_ = np.linalg.lstsq(projection, endog[:, c], rcond=None)
        endhat[:, c] = projection @ beta

    augmented = design([endog, endhat])
    response = frame["y"].to_numpy(float)
    coefficients, *_ = np.linalg.lstsq(augmented, response, rcond=None)
    resid = response - augmented @ coefficients

    from westata.vce.dispatch import compute_cov

    covariance, df_denom = compute_cov(
        augmented,
        resid,
        cov_type="cluster",
        cov_kwds={"groups": np.arange(N) % 50},
    )
    block = np.arange(augmented.shape[1] - endog.shape[1], augmented.shape[1])
    bq = coefficients[block]
    vq = covariance[np.ix_(block, block)]
    expected = float(bq @ np.linalg.solve(vq, bq) / endog.shape[1])

    assert rendered.statistic == pytest.approx(expected, rel=1e-10)
    assert rendered.df_denom == df_denom


def test_regF_is_not_zero_on_endogenous_data(frame) -> None:
    """A degenerate F of exactly 0 is the symptom of building it as real 2SLS.

    ``endhat`` is a linear combination of the instrument set, so instrumenting it
    leaves its coefficient unidentified and the statistic collapses to 0.
    """

    result = ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", frame).fit(
        cov_type="cluster", cov_kwds={"clusters": np.arange(N) % 50}
    )
    rendered = result.estat_endogenous()

    assert rendered.statistic > 1.0
    assert rendered.pvalue < 0.01


def test_regF_r_names(frame) -> None:
    result = ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", frame).fit(
        cov_type="cluster", cov_kwds={"clusters": np.arange(N) % 50}
    )

    assert set(result.estat_endogenous().to_dict()) == {
        "regF",
        "p_regF",
        "regFdf_n",
        "regFdf_d",
    }


def test_robust_regF_differs_from_the_robust_score(frame) -> None:
    """Two different tests; ``regF`` is reachable through its own constructor."""

    from westata.postestimation.endog_regf import endog_regf_test

    result = ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", frame).fit(cov_type="HC1")
    score = result.estat_endogenous()
    regf = endog_regf_test(
        result.iv_structure, frame["y"].to_numpy(float), cov_type="robust"
    )

    assert regf.test == "regF"
    assert score.test == "robwool"
    assert not np.isclose(regf.statistic, score.statistic, rtol=1e-3)


# ------------------------- which estimators support estat endogenous at all
@pytest.mark.parametrize("cov_type", ["nonrobust", "HC0", "HAC"])
def test_liml_has_no_endogeneity_test(frame, cov_type) -> None:
    """``ivregress_estat.ado:1002-1019`` dispatches only 2SLS and GMM.

    The final ``else`` raises error 321 ("not available after LIML estimator"), so
    LIML has no endogeneity test under ANY VCE -- not the Durbin-Wu pair and not
    the score test. Both are 2SLS constructions.
    """

    kwargs = {"cov_kwds": {"maxlags": 1}} if cov_type == "HAC" else {}
    fitted = ws.IVLIML.from_formula("y ~ [x ~ z1 + z2]", frame).fit(
        cov_type=cov_type, **kwargs
    )

    with pytest.raises(NotImplementedError, match="LIML"):
        fitted.estat_endogenous()


@pytest.mark.parametrize("cov_type", ["nonrobust", "HC0", "HAC"])
def test_gmm_reports_the_c_statistic_not_a_2sls_test(frame, cov_type) -> None:
    """``Endoggmm`` (``ado:1427-1524``) reports ``C = e(J) - J'`` in every case.

    It refits a CONSTRAINED GMM whose weight matrix drops the challenged
    orthogonality conditions. Neither the Durbin-Wu pair nor the Wooldridge score
    is the GMM answer, so reporting either would mislabel a 2SLS construction.

    **Why it is not implemented** (investigated in round 48): the constrained
    refit needs a weight matrix of REDUCED dimension, but linearmodels requires
    ``initial_weight`` to match the full moment count --

        ValueError: initial_weight must be a 3 by 3 array

    -- so ``fit(initial_weight=S_con)`` cannot express it. Stata does this inside
    the Mata builtin ``_iv_gmm_cstat``, which ships no ado source to transcribe.
    With neither a supported backend route nor a readable reference, reporting
    nothing is the honest result.
    """

    kwargs = {"cov_kwds": {"maxlags": 1}} if cov_type == "HAC" else {}
    fitted = ws.IVGMM.from_formula("y ~ [x ~ z1 + z2]", frame).fit(
        cov_type=cov_type, **kwargs
    )

    with pytest.raises(NotImplementedError, match="GMM C statistic"):
        fitted.estat_endogenous()


def test_the_gmm_c_statistic_cannot_be_built_from_linearmodels(frame) -> None:
    """Pin the blocker, so a future linearmodels upgrade is noticed.

    If a version ever accepts a reduced-dimension ``initial_weight``, this test
    fails and the C statistic becomes implementable -- which is exactly when the
    refusal above should be revisited.
    """

    from linearmodels.iv import IVGMM

    model = IVGMM.from_formula("y ~ 1 + [x ~ z1 + z2]", frame)
    fitted = model.fit()

    weight = np.asarray(fitted.weight_matrix, dtype=float)
    assert weight.shape == (3, 3)

    # S = W^-1, drop one moment, invert back: a 2x2 weight matrix.
    s = np.linalg.inv(weight)
    keep = [0, 1]
    w_con = np.linalg.inv(s[np.ix_(keep, keep)])
    assert w_con.shape == (2, 2)

    with pytest.raises(ValueError, match="initial_weight must be"):
        model.fit(initial_weight=w_con)


def test_2sls_keeps_its_three_branches(frame) -> None:
    """The dispatch that IS implemented must stay intact."""

    model = ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", frame)

    assert hasattr(model.fit().estat_endogenous(), "durbin")
    assert model.fit(cov_type="HC1").estat_endogenous().test == "robwool"
    assert (
        model.fit(cov_type="HAC", cov_kwds={"maxlags": 1}).estat_endogenous().test
        == "acwool"
    )
