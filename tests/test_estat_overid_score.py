"""Tests for the Wooldridge score overidentification test.

Transcribed from ``ivregress_estat.ado:184-261``:

    endoghat_i = fitted from regressing each endogenous regressor on ALL
                 instruments (included exogenous + excluded instruments)
    khat_j     = residual from regressing excluded instrument j on
                 (endoghat..., inclinst), times the IV residual
    score      = N - rss of (ones on khat, NO constant)

The ado comments at line 181-182: "if we assume homoskedastic errors, then it's
the same as Sargan's test.  Otherwise, it's not."

That comment drives the decisive test here: on genuinely heteroskedastic data the
score statistic **must** diverge from Sargan's.  If both routes returned the same
number, the robust path would not be weighting anything and a homoskedastic-only
test would pass anyway.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import westata as ws


def _build(seed: int, heteroskedastic: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 500
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    scale = 0.4 + 1.6 * np.abs(z1) if heteroskedastic else np.ones(n)
    u = rng.normal(size=n) * scale
    x = 0.7 * z1 + 0.3 * z2 + 0.6 * u + rng.normal(size=n) * scale
    y = 1.0 + 2.0 * x + u
    return pd.DataFrame({"y": y, "x": x, "z1": z1, "z2": z2})


def _hand_score(structure, resid) -> float:
    """Independent transcription of the ado's three steps."""

    instruments = np.asarray(structure["instruments"], dtype=float)
    endog = np.asarray(structure["endog"], dtype=float)
    exog = np.asarray(structure["exog"], dtype=float)
    exog_names = [str(v) for v in structure["exog_names"]]
    resid = np.asarray(resid, dtype=float)
    n = resid.size

    constant_names = {"_cons", "intercept", "const"}
    has_const = any(v.strip().lower() in constant_names for v in exog_names)
    included = [j for j, v in enumerate(exog_names) if v.strip().lower() not in constant_names]

    parts = []
    if has_const:
        parts.append(np.ones(n))
    if included:
        parts.append(exog[:, included])
    parts.append(instruments)
    all_inst = np.column_stack(parts)

    endoghat = np.empty((n, endog.shape[1]))
    for c in range(endog.shape[1]):
        beta, *_ = np.linalg.lstsq(all_inst, endog[:, c], rcond=None)
        endoghat[:, c] = all_inst @ beta

    cond_parts = [endoghat]
    if has_const:
        cond_parts.insert(0, np.ones(n))
    if included:
        cond_parts.append(exog[:, included])
    cond = np.column_stack(cond_parts)

    restrictions = instruments.shape[1] - endog.shape[1]
    khat = []
    for c in range(instruments.shape[1]):
        if len(khat) >= restrictions:
            break
        beta, *_ = np.linalg.lstsq(cond, instruments[:, c], rcond=None)
        residual = instruments[:, c] - cond @ beta
        khat.append(residual * resid)
    khat = np.column_stack(khat)

    ones = np.ones(n)
    beta, *_ = np.linalg.lstsq(khat, ones, rcond=None)
    residual = ones - khat @ beta
    return float(n - residual @ residual)


@pytest.fixture()
def homoskedastic():
    frame = _build(9, heteroskedastic=False)
    return frame, ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", frame)


@pytest.fixture()
def heteroskedastic():
    frame = _build(9, heteroskedastic=True)
    return frame, ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", frame)


def test_robust_fit_reports_the_score_test(homoskedastic) -> None:
    _, model = homoskedastic

    rendered = model.fit(cov_type="HC1").estat_overid()

    assert rendered.test == "Wooldridge score"


def test_score_matches_an_independent_transcription(homoskedastic) -> None:
    _, model = homoskedastic
    result = model.fit(cov_type="HC1")
    expected = _hand_score(result.iv_structure, result.resid)

    assert result.estat_overid().statistic == pytest.approx(expected, rel=1e-10)


def test_score_matches_the_transcription_on_heteroskedastic_data(
    heteroskedastic,
) -> None:
    _, model = heteroskedastic
    result = model.fit(cov_type="HC1")
    expected = _hand_score(result.iv_structure, result.resid)

    assert result.estat_overid().statistic == pytest.approx(expected, rel=1e-10)


def test_score_and_sargan_agree_under_homoskedasticity(homoskedastic) -> None:
    """The ado's own remark: with homoskedastic errors they coincide."""

    _, model = homoskedastic
    sargan = model.fit().estat_overid().statistic
    score = model.fit(cov_type="HC1").estat_overid().statistic

    assert score == pytest.approx(sargan, abs=1e-4)


def test_score_and_sargan_diverge_under_heteroskedasticity(heteroskedastic) -> None:
    """The decisive check: if these were equal the robust path would be a no-op."""

    _, model = heteroskedastic
    sargan = model.fit().estat_overid().statistic
    score = model.fit(cov_type="HC1").estat_overid().statistic

    assert not np.isclose(sargan, score, rtol=1e-3), (sargan, score)


def test_statistic_is_non_negative(heteroskedastic) -> None:
    _, model = heteroskedastic

    assert model.fit(cov_type="HC1").estat_overid().statistic >= 0.0


def test_pvalue_is_the_chi2_upper_tail(heteroskedastic) -> None:
    _, model = heteroskedastic
    rendered = model.fit(cov_type="HC1").estat_overid()

    assert rendered.pvalue == pytest.approx(
        float(stats.chi2.sf(rendered.statistic, rendered.df)), rel=1e-12
    )


def test_df_is_the_number_of_restrictions(heteroskedastic) -> None:
    _, model = heteroskedastic

    # 2 excluded instruments for 1 endogenous regressor.
    assert model.fit(cov_type="HC1").estat_overid().df == 1.0


def test_r_names_follow_overid2sls(heteroskedastic) -> None:
    _, model = heteroskedastic

    payload = model.fit(cov_type="HC1").estat_overid().to_dict()

    assert set(payload) == {"score", "p_score"}


def test_no_basmann_under_robust_vce(heteroskedastic) -> None:
    """Basmann is left missing when the VCE is robust."""

    _, model = heteroskedastic

    assert model.fit(cov_type="HC1").estat_overid().basmann is None


def test_clustered_vce_is_refused(heteroskedastic) -> None:
    """``ivregress_estat.ado:290-296``: not available with clustered errors."""

    frame, model = heteroskedastic
    clustered = model.fit(
        cov_type="cluster", cov_kwds={"clusters": np.arange(len(frame)) % 40}
    )

    with pytest.raises(ValueError, match="clustered"):
        clustered.estat_overid()


def test_nonrobust_still_reports_sargan_and_basmann(homoskedastic) -> None:
    _, model = homoskedastic

    rendered = model.fit().estat_overid()

    assert rendered.test == "Sargan"
    assert rendered.basmann is not None


def test_output_names_the_score_test(heteroskedastic) -> None:
    _, model = heteroskedastic

    text = str(model.fit(cov_type="HC1").estat_overid())

    assert "Wooldridge score" in text
    assert "overidentifying restrictions" in text


# ------------------------------- which test each estimator reports under each VCE
@pytest.fixture()
def overid_frame():
    """Over-identified by two restrictions, so Sargan/Basmann are defined."""

    rng = np.random.default_rng(1)
    n = 400
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    z3 = rng.normal(size=n)
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = 0.5 * u[t - 1] + rng.normal()
    x = 0.7 * z1 + 0.3 * z2 + 0.5 * u + rng.normal(size=n)
    y = 1.0 + 2.0 * x + u
    return pd.DataFrame({"y": y, "x": x, "z1": z1, "z2": z2, "z3": z3})


def test_2sls_reports_sargan_then_the_score(overid_frame) -> None:
    model = ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2 + z3]", overid_frame)

    assert model.fit().estat_overid().test == "Sargan"
    assert model.fit(cov_type="HC0").estat_overid().test == "Wooldridge score"
    assert (
        model.fit(cov_type="HAC", cov_kwds={"maxlags": 1}).estat_overid().test
        == "Wooldridge score"
    )


def test_gmm_always_reports_hansen_j(overid_frame) -> None:
    """``ivregress_estat.ado:408-436`` (``OverIDgmm``) reports ``e(J)``
    unconditionally -- there is no VCE-dependent branch."""

    model = ws.IVGMM.from_formula("y ~ [x ~ z1 + z2 + z3]", overid_frame)

    assert model.fit().estat_overid().test == "Hansen J"
    assert model.fit(cov_type="HC0").estat_overid().test == "Hansen J"
    assert (
        model.fit(cov_type="HAC", cov_kwds={"maxlags": 1}).estat_overid().test
        == "Hansen J"
    )


@pytest.mark.parametrize(
    "cov_type", ["HC0", "HAC"]
)
def test_liml_refuses_a_robust_overidentification_test(
    overid_frame, cov_type
) -> None:
    """``ivregress_estat.ado:354-357``: "tests not available after LIML estimation
    with robust VCE".

    LIML does NOT fall back to the Wooldridge score the way 2SLS does. Reporting
    that score would apply a 2SLS bread (``Xhat'Xhat``) to a k-class fit whose
    bread is ``(1-kappa)X'X + kappa X'P_Z X`` -- a number with no derivation.
    """

    kwargs = {"cov_kwds": {"maxlags": 1}} if cov_type == "HAC" else {}
    fitted = ws.IVLIML.from_formula("y ~ [x ~ z1 + z2 + z3]", overid_frame).fit(
        cov_type=cov_type, **kwargs
    )

    with pytest.raises(ValueError, match="LIML"):
        fitted.estat_overid()


def test_liml_still_reports_sargan_when_unadjusted(overid_frame) -> None:
    fitted = ws.IVLIML.from_formula("y ~ [x ~ z1 + z2 + z3]", overid_frame).fit()

    rendered = fitted.estat_overid()

    assert rendered.test == "Sargan"
    assert rendered.basmann is not None
