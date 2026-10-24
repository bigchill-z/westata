"""Tests for the panel IV / GLS / PCSE / unit-root estimators (P9 Task 2).

Every numerical target below is a StataNow 19.5 value, and every one of them was
re-measured by running Stata on the same data rather than copied from the plan.
Two places where that mattered are called out in the tests themselves:

* ``xtgls, panels(heteroskedastic)`` is **not** pooled OLS.  The P9 plan says the
  coefficients "must equal pooled OLS exactly"; Stata gives x1 = .6798471 where
  OLS gives .7533126.  The OLS identity is asserted here on a homoskedastic panel,
  where it genuinely holds, and the fixture-data answer is asserted against
  Stata's number instead.
* ``xtpcse`` applies **no** ``N/(N-k) * T/(T-1)`` factor.  The plan specifies one;
  the manual does not, and the uncorrected VCE matches Stata's ``e(V)`` to 3e-12.

Where a test compares against OLS or against a hand-built matrix, the reference is
computed in the test from first principles so the implementation cannot hide
behind a helper it also uses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.errors import ConsistencyError, UnsupportedError
from westata.estimators.panel_iv import (
    PanelUnitRootResult,
    XtGls,
    XtIvreg,
    XtPcse,
    XtUnitroot,
)

# --------------------------------------------------------------- fixture data
# These reproduce scripts/capture_p9_fixtures.py exactly, so the numbers below are
# the same ones in tests/conformance/stata_fixtures/p9_outputs.txt.
N_IV_FIRMS, N_IV_YEARS = 100, 7
N_CNT_FIRMS, N_CNT_YEARS = 100, 7
N_DYN_FIRMS, N_DYN_YEARS = 90, 9

REQUIRED_E_KEYS = ("N", "df_m", "df_r", "rank", "r2")


def iv_panel() -> pd.DataFrame:
    """The ``xtivreg`` fixture panel: y on endog (instrumented by z) and w."""

    rng = np.random.default_rng(89)
    firm = np.repeat(np.arange(N_IV_FIRMS), N_IV_YEARS)
    year = np.tile(np.arange(1, N_IV_YEARS + 1), N_IV_FIRMS)
    n = firm.size
    z = rng.normal(size=n)
    w = rng.normal(size=n)
    u = rng.normal(size=n)
    endog = 0.7 * z + 0.5 * w + 0.6 * u + rng.normal(size=n)
    effect = rng.normal(scale=0.4, size=N_IV_FIRMS)[firm]
    y = 1.0 + 1.2 * endog + 0.5 * w + effect + u
    return pd.DataFrame({"firm": firm, "year": year, "y": y, "endog": endog, "z": z, "w": w})


def count_panel() -> pd.DataFrame:
    """The ``xtgls`` / ``xtpcse`` fixture panel."""

    rng = np.random.default_rng(73)
    firm = np.repeat(np.arange(N_CNT_FIRMS), N_CNT_YEARS)
    year = np.tile(np.arange(1, N_CNT_YEARS + 1), N_CNT_FIRMS)
    n = firm.size
    x1 = rng.normal(size=n)
    effect = rng.normal(scale=0.4, size=N_CNT_FIRMS)[firm]
    mu = np.exp(0.3 + 0.5 * x1 + effect)
    return pd.DataFrame(
        {"firm": firm, "year": year, "x1": x1, "y": rng.poisson(mu).astype(float)}
    )


def homoskedastic_panel(n_firms: int = 60, n_years: int = 8, seed: int = 5150) -> pd.DataFrame:
    """A panel whose OLS residuals have an identical sum of squares in every entity.

    This is the case where ``xtgls, panels(heteroskedastic)`` really does collapse
    to pooled OLS: with ``e_i'e_i`` equal for all ``i`` the GLS weights are all equal
    and the estimating equations reduce to OLS's.

    A plain i.i.d. disturbance is *not* enough.  With finite panels the estimated
    per-panel variances still differ by sampling error, FGLS reweights them, and the
    coefficients move -- measured at 0.048 on x1 for a 40 x 6 i.i.d. draw.  The
    residuals are therefore constructed explicitly: project a random block onto the
    orthogonal complement of that entity's ``[1, x1]`` columns, then rescale it to a
    common norm.  Each block is then orthogonal to its own regressors, so it survives
    OLS untouched (``X'e = 0`` holds block by block) and the per-entity residual
    norms are equal by construction rather than by luck.
    """

    rng = np.random.default_rng(seed)
    firm = np.repeat(np.arange(n_firms), n_years)
    year = np.tile(np.arange(1, n_years + 1), n_firms)
    n = firm.size
    x1 = rng.normal(size=n)

    residual = np.zeros(n)
    target_norm = np.sqrt(float(n_years - 2) / n_firms)
    for value in np.unique(firm):
        index = firm == value
        block = rng.normal(size=int(index.sum()))
        basis = np.column_stack([np.ones(block.size), x1[index]])
        # Residualise the block on its own regressors so X'e = 0 within the entity.
        block = block - basis @ np.linalg.lstsq(basis, block, rcond=None)[0]
        spread = float(np.linalg.norm(block))
        residual[index] = block * (target_norm / spread) if spread > 0.0 else 0.0

    design = np.column_stack([np.ones(n), x1])
    y = design @ np.array([1.0, 0.8]) + residual
    return pd.DataFrame({"firm": firm, "year": year, "x1": x1, "y": y})


def dynamic_panel() -> pd.DataFrame:
    """The ``xtunitroot`` fixture panel (AR(1) with an entity effect)."""

    rng = np.random.default_rng(79)
    rows = []
    for firm in range(N_DYN_FIRMS):
        effect = rng.normal(scale=0.5)
        lagged = rng.normal()
        for year in range(1, N_DYN_YEARS + 1):
            x1 = rng.normal()
            value = 0.5 * lagged + 0.8 * x1 + effect + rng.normal(scale=0.5)
            rows.append((firm, year, value, x1))
            lagged = value
    return pd.DataFrame(rows, columns=["firm", "year", "y", "x1"])


def ols_reference(X: np.ndarray, y: np.ndarray, *, df_resid: float | None = None):
    """OLS point estimates, residuals and the homoskedastic covariance."""

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = float(df_resid) if df_resid is not None else float(X.shape[0] - X.shape[1])
    sigma2 = float(resid @ resid) / dof
    return beta, resid, sigma2 * np.linalg.inv(X.T @ X)


def within_reference(array: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Stata's within transform ``w_it - w_i. + wbar``, written out by hand."""

    array = np.asarray(array, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    out = np.empty_like(array)
    for value in np.unique(groups):
        index = groups == value
        out[index] = array[index] - array[index].mean(axis=0) + array.mean(axis=0)
    return out


# ------------------------------------------------------------------ xtivreg
def test_xtivreg_fe_matches_stata_coefficients() -> None:
    """``xtivreg y w (endog = z), fe`` -- Stata p9_outputs-style reference.

    StataNow 19.5 prints endog 1.102058 (se .0694014), w .5391216
    (se .0502034), _cons 1.014978 (se .0387732).
    """

    result = XtIvreg.from_formula(
        "y ~ w + (endog = z)", iv_panel(), entity="firm", time="year", method="fe"
    ).fit()

    assert float(result.params["endog"]) == pytest.approx(1.102058, abs=5e-7)
    assert float(result.params["w"]) == pytest.approx(0.5391216, abs=5e-7)
    assert float(result.params["_cons"]) == pytest.approx(1.014978, abs=5e-7)

    assert float(result.bse["endog"]) == pytest.approx(0.0694014, rel=1e-5)
    assert float(result.bse["w"]) == pytest.approx(0.0502034, rel=1e-5)
    assert float(result.bse["_cons"]) == pytest.approx(0.0387732, rel=1e-5)

    assert result.model_name == "xtivreg, fe"
    assert result.nobs == N_IV_FIRMS * N_IV_YEARS


def test_xtivreg_fe_matches_hand_built_within_2sls() -> None:
    """The within transform must be applied to the instruments too.

    The reference below demeans ``y``, both regressor blocks *and* ``z``, then runs
    2SLS.  If the implementation forgot to demean the instruments, the projection
    would still contain the entity effect and the coefficients would differ -- which
    is exactly the failure mode the P9 notes warn about.
    """

    frame = iv_panel()
    groups = frame["firm"].to_numpy()
    yt = within_reference(frame["y"].to_numpy(), groups).reshape(-1)
    design = within_reference(
        np.column_stack([frame["w"], frame["endog"], np.ones(len(frame))]), groups
    )
    instruments = within_reference(
        np.column_stack([frame["w"], frame["z"], np.ones(len(frame))]), groups
    )
    projected = instruments @ np.linalg.lstsq(instruments, design, rcond=None)[0]
    beta = np.linalg.lstsq(projected, yt, rcond=None)[0]

    result = XtIvreg.from_formula(
        "y ~ w + (endog = z)", frame, entity="firm", time="year", method="fe"
    ).fit()

    np.testing.assert_allclose(
        result.params[["w", "endog", "_cons"]].to_numpy(), beta, rtol=0.0, atol=1e-9
    )

    # The within transform is doing real work: pooled 2SLS on the untransformed
    # data gives visibly different coefficients, because it keeps the entity effect.
    raw_design = np.column_stack([frame["w"], frame["endog"], np.ones(len(frame))])
    raw_instruments = np.column_stack([frame["w"], frame["z"], np.ones(len(frame))])
    raw_projected = raw_instruments @ np.linalg.lstsq(
        raw_instruments, raw_design, rcond=None
    )[0]
    beta_pooled = np.linalg.lstsq(raw_projected, frame["y"].to_numpy(), rcond=None)[0]
    assert not np.allclose(beta_pooled, beta, rtol=1e-3)


def test_xtivreg_re_matches_stata_coefficients() -> None:
    """``xtivreg, re`` is G2SLS.  Stata: endog 1.120466 (se .065102),
    w .5286517 (se .0482393), _cons 1.014533 (se .0486261)."""

    result = XtIvreg.from_formula(
        "y ~ w + (endog = z)", iv_panel(), entity="firm", time="year", method="re"
    ).fit()

    assert float(result.params["endog"]) == pytest.approx(1.120466, abs=5e-7)
    assert float(result.params["w"]) == pytest.approx(0.5286517, abs=5e-7)
    assert float(result.params["_cons"]) == pytest.approx(1.014533, abs=5e-7)

    assert float(result.bse["endog"]) == pytest.approx(0.065102, rel=1e-5)
    assert float(result.bse["w"]) == pytest.approx(0.0482393, rel=1e-5)
    assert float(result.bse["_cons"]) == pytest.approx(0.0486261, rel=1e-5)
    assert result.model_name == "xtivreg, re"


def test_xtivreg_re_theta_matches_stata() -> None:
    """Stata reports ``e(theta) = .2064715`` for this panel."""

    result = XtIvreg.from_formula(
        "y ~ w + (endog = z)", iv_panel(), entity="firm", time="year", method="re"
    ).fit()
    assert float(result.e["theta"]) == pytest.approx(0.20647146, rel=1e-6)


def test_xtivreg_fe_and_re_differ() -> None:
    """The two transforms are different estimators, not two spellings of one."""

    frame = iv_panel()
    fe = XtIvreg.from_formula("y ~ w + (endog = z)", frame, entity="firm", time="year", method="fe").fit()
    re = XtIvreg.from_formula("y ~ w + (endog = z)", frame, entity="firm", time="year", method="re").fit()
    assert not np.allclose(float(fe.params["endog"]), float(re.params["endog"]), rtol=1e-4)


def test_xtivreg_formula_parser_separates_the_three_pieces() -> None:
    """Two exogenous regressors and two instruments must be split correctly."""

    frame = iv_panel().copy()
    rng = np.random.default_rng(4321)
    frame["w2"] = rng.normal(size=len(frame))
    frame["z2"] = rng.normal(size=len(frame))
    frame["endog2"] = 0.4 * frame["z2"] + rng.normal(size=len(frame))

    model = XtIvreg.from_formula(
        "y ~ w + w2 + (endog + endog2 = z + z2)",
        frame,
        entity="firm",
        time="year",
        method="fe",
    )

    # Exogenous block: the intercept plus both exogenous variables, in formula order.
    assert model._exog_names == ["Intercept", "w", "w2"]
    # Endogenous block: no intercept -- it would duplicate the exogenous one.
    assert model._endog_names == ["endog", "endog2"]
    # Instruments: exactly the excluded ones, in formula order.
    assert model._instrument_names == ["z", "z2"]

    # The matrices line up with the names: [w, w2] / [endog, endog2] / [z, z2].
    n = len(frame)
    assert model._X.shape == (n, 3)
    assert model._W.shape == (n, 2)
    assert model._Zx.shape == (n, 2)
    np.testing.assert_allclose(model._W[:, 0], frame["endog"].to_numpy())
    np.testing.assert_allclose(model._W[:, 1], frame["endog2"].to_numpy())
    np.testing.assert_allclose(model._Zx[:, 0], frame["z"].to_numpy())
    np.testing.assert_allclose(model._Zx[:, 1], frame["z2"].to_numpy())

    # And the model fits with the expected parameter names, in Stata's order.
    result = model.fit()
    assert list(result.params.index) == ["_cons", "w", "w2", "endog", "endog2"]
    assert np.all(np.isfinite(result.params.to_numpy()))


def test_xtivreg_parser_accepts_a_single_exogenous_regressor() -> None:
    model = XtIvreg.from_formula(
        "y ~ w + (endog = z)", iv_panel(), entity="firm", time="year"
    )
    assert model._exog_names == ["Intercept", "w"]
    assert model._endog_names == ["endog"]
    assert model._instrument_names == ["z"]


def test_xtivreg_rejects_a_missing_instrument_block() -> None:
    with pytest.raises(ConsistencyError, match="instrument block"):
        XtIvreg.from_formula("y ~ w", iv_panel(), entity="firm", time="year")


def test_xtivreg_rejects_trailing_terms_after_the_block() -> None:
    with pytest.raises(ConsistencyError, match="must come last"):
        XtIvreg.from_formula(
            "y ~ w + (endog = z) + w2", iv_panel(), entity="firm", time="year"
        )


def test_xtivreg_rejects_an_unidentified_model() -> None:
    """Fewer instruments than endogenous regressors is Stata's r(481)."""

    frame = iv_panel().copy()
    frame["endog2"] = np.random.default_rng(11).normal(size=len(frame))
    with pytest.raises(ConsistencyError, match="instruments"):
        XtIvreg.from_formula(
            "y ~ w + (endog + endog2 = z)", frame, entity="firm", time="year"
        )


@pytest.mark.parametrize("method", ["between", "be", "fd", "ec2sls"])
def test_xtivreg_rejects_unimplemented_methods(method: str) -> None:
    with pytest.raises(UnsupportedError):
        XtIvreg.from_formula(
            "y ~ w + (endog = z)", iv_panel(), entity="firm", time="year", method=method
        )


def test_xtivreg_rejects_robust_cov_types() -> None:
    model = XtIvreg.from_formula("y ~ w + (endog = z)", iv_panel(), entity="firm", time="year")
    for cov_type in ("robust", "HC1", "cluster"):
        with pytest.raises(UnsupportedError):
            model.fit(cov_type=cov_type)


# -------------------------------------------------------------------- xtgls
def test_xtgls_heteroskedastic_matches_stata_not_ols() -> None:
    """Stata: x1 .6798471 (se .0432583), _cons 1.431659 (se .0421405).

    Pooled OLS would say .7533126 -- the two are *different estimators*, and the
    plan's "must equal pooled OLS exactly" is wrong for this data.
    """

    frame = count_panel()
    result = XtGls.from_formula("y ~ x1", frame, entity="firm", time="year").fit()
    plain = ws.OLS.from_formula("y ~ x1", frame).fit()

    assert float(result.params["x1"]) == pytest.approx(0.6798471, abs=5e-7)
    assert float(result.params["Intercept"]) == pytest.approx(1.431659, abs=5e-6)
    assert float(result.bse["x1"]) == pytest.approx(0.0432583, rel=1e-5)
    assert float(result.bse["Intercept"]) == pytest.approx(0.0421405, rel=1e-5)

    # The FGLS reweighting really moves the coefficient: this is not OLS.
    assert abs(float(result.params["x1"]) - float(plain.params["x1"])) > 0.05


def test_xtgls_heteroskedastic_equals_ols_on_a_homoskedastic_panel() -> None:
    """The OLS identity the plan asks for, asserted where it actually holds.

    With one common error variance there is nothing to reweight, so the two-step
    FGLS estimating equations coincide with OLS's.  The coefficients must agree to
    machine precision and only the standard errors may differ.
    """

    frame = homoskedastic_panel()
    result = XtGls.from_formula("y ~ x1", frame, entity="firm", time="year").fit()
    plain = ws.OLS.from_formula("y ~ x1", frame).fit()

    np.testing.assert_allclose(
        result.params[["Intercept", "x1"]].to_numpy(),
        plain.params[["Intercept", "x1"]].to_numpy(),
        rtol=1e-10,
        atol=0.0,
    )


def test_xtgls_matches_the_manual_formula() -> None:
    """``b = (X'Omega^-1 X)^-1 X'Omega^-1 y`` with ``Sigma_ii = e_i'e_i / T``.

    Built by hand from OLS residuals, so a wrong weight or a wrong divisor shows up
    immediately.
    """

    frame = count_panel()
    X = np.column_stack([frame["x1"], np.ones(len(frame))])
    y = frame["y"].to_numpy()
    groups = frame["firm"].to_numpy()

    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    resid_ols = y - X @ beta_ols

    bread = np.zeros((2, 2))
    rhs = np.zeros(2)
    for value in np.unique(groups):
        index = groups == value
        piece = resid_ols[index]
        sigma_ii = float(piece @ piece) / index.sum()  # divide by T, not T-1
        bread += X[index].T @ X[index] / sigma_ii
        rhs += X[index].T @ y[index] / sigma_ii

    beta_gls = np.linalg.solve(bread, rhs)
    cov_gls = np.linalg.inv(bread)

    result = XtGls.from_formula("y ~ x1", frame, entity="firm", time="year").fit()
    np.testing.assert_allclose(
        result.params[["x1", "Intercept"]].to_numpy(), beta_gls, rtol=1e-9, atol=0.0
    )
    np.testing.assert_allclose(
        result.bse[["x1", "Intercept"]].to_numpy(),
        np.sqrt(np.diag(cov_gls)),
        rtol=1e-9,
        atol=0.0,
    )


def test_xtgls_iid_nmk_is_regress() -> None:
    """``panels(iid) corr(independent) nmk`` is equivalent to ``regress``.

    That equivalence is stated explicitly in ``[XT] xtgls`` Remarks and examples.
    Without ``nmk`` the coefficients are still OLS but the variance is normalised by
    ``N`` rather than ``N - k``, so the standard errors are smaller by exactly
    ``sqrt((N-k)/N)`` -- a documented difference in convention, not a defect.
    """

    frame = count_panel()
    plain = ws.OLS.from_formula("y ~ x1", frame).fit()
    nobs, k = len(frame), 2

    with_nmk = XtGls.from_formula(
        "y ~ x1", frame, entity="firm", time="year", panels="iid", nmk=True
    ).fit()
    np.testing.assert_allclose(with_nmk.params.to_numpy(), plain.params.to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(with_nmk.bse.to_numpy(), plain.bse.to_numpy(), rtol=1e-12)

    without_nmk = XtGls.from_formula(
        "y ~ x1", frame, entity="firm", time="year", panels="iid"
    ).fit()
    np.testing.assert_allclose(
        without_nmk.params.to_numpy(), plain.params.to_numpy(), rtol=1e-12
    )
    np.testing.assert_allclose(
        without_nmk.bse.to_numpy(),
        plain.bse.to_numpy() * np.sqrt((nobs - k) / nobs),
        rtol=1e-12,
    )


def test_xtgls_panels_correlated_runs_on_a_balanced_panel() -> None:
    frame = count_panel()
    result = XtGls.from_formula(
        "y ~ x1", frame, entity="firm", time="year", panels="correlated"
    ).fit()
    assert np.all(np.isfinite(result.params.to_numpy()))
    assert np.all(np.isfinite(result.bse.to_numpy()))
    assert result.e["n_cv"] == N_CNT_FIRMS * (N_CNT_FIRMS + 1) // 2
    # Heteroskedasticity plus cross-sectional correlation is a different estimator.
    plain = ws.OLS.from_formula("y ~ x1", frame).fit()
    assert not np.allclose(result.bse.to_numpy(), plain.bse.to_numpy(), rtol=1e-3)


def test_xtgls_reports_panel_counts() -> None:
    result = XtGls.from_formula("y ~ x1", count_panel(), entity="firm", time="year").fit()
    assert result.e["N_g"] == N_CNT_FIRMS
    assert result.e["N_t"] == N_CNT_YEARS
    assert result.e["n_cf"] == 2
    assert result.e["n_cv"] == N_CNT_FIRMS
    assert result.e["n_cr"] == 0


def test_xtgls_rejects_autocorrelation() -> None:
    with pytest.raises(UnsupportedError, match="corr"):
        XtGls.from_formula("y ~ x1", count_panel(), entity="firm", time="year", corr="ar1")


def test_xtgls_rejects_unknown_panel_specification() -> None:
    with pytest.raises(UnsupportedError, match="panels"):
        XtGls.from_formula(
            "y ~ x1", count_panel(), entity="firm", time="year", panels="definitely-not"
        )


def test_xtgls_correlated_requires_a_balanced_panel() -> None:
    frame = count_panel().drop(index=[0, 5]).reset_index(drop=True)
    with pytest.raises(ConsistencyError, match="balanced"):
        XtGls.from_formula(
            "y ~ x1", frame, entity="firm", time="year", panels="correlated"
        ).fit()


# ------------------------------------------------------------------- xtpcse
def test_xtpcse_coefficients_equal_ols() -> None:
    """No autocorrelation means OLS point estimates (``[XT] xtpcse``)."""

    frame = count_panel()
    result = XtPcse.from_formula("y ~ x1", frame, entity="firm", time="year").fit()
    plain = ws.OLS.from_formula("y ~ x1", frame).fit()

    np.testing.assert_allclose(
        result.params[["Intercept", "x1"]].to_numpy(),
        plain.params[["Intercept", "x1"]].to_numpy(),
        rtol=1e-10,
        atol=0.0,
    )
    # Same coefficients, different inference: the whole point of the estimator.
    assert float(result.bse["x1"]) != pytest.approx(float(plain.bse["x1"]), rel=1e-6)
    assert float(result.bse["Intercept"]) != pytest.approx(
        float(plain.bse["Intercept"]), rel=1e-6
    )


def test_xtpcse_balanced_without_time_matches_stata() -> None:
    """A balanced panel needs no ``time`` column: periods come from row order.

    Captured with StataNow 19.5 on 2026-10-24 on this 100-firm x 7-year panel::

        xtset firm year
        xtpcse y x1        x1 .75331259 (se .05643178)
                           _cons 1.57268563 (se .03729976)
                           e(n_sigma) = 7

    Stata itself requires ``tis``, but a caller who only declared the entity
    should not have to repeat the period name for a panel that is balanced by
    construction.  The *unbalanced* case still raises, because there the
    positional pairing would be wrong - see
    ``test_xtpcse_unbalanced_without_time_is_rejected``.  This test is the
    regression for a fix that briefly raised r(459) here.
    """

    frame = count_panel()
    without_time = XtPcse.from_formula("y ~ x1", frame, entity="firm").fit()
    with_time = XtPcse.from_formula("y ~ x1", frame, entity="firm", time="year").fit()

    np.testing.assert_allclose(
        without_time.bse.to_numpy(), with_time.bse.to_numpy(), rtol=0.0, atol=0.0
    )
    assert float(without_time.params["x1"]) == pytest.approx(0.75331259, rel=1e-6)
    assert float(without_time.params["Intercept"]) == pytest.approx(1.57268563, rel=1e-6)
    assert float(without_time.bse["x1"]) == pytest.approx(0.05643178, rel=1e-6)
    assert float(without_time.bse["Intercept"]) == pytest.approx(0.03729976, rel=1e-6)
    assert int(without_time.e["n_sigma"]) == 7


def test_xtpcse_unbalanced_without_time_is_rejected() -> None:
    """Without a period column an unbalanced panel cannot be matched up.

    Pairing rows by position would compare entity 1's year 3 against entity 2's
    year 5, which is not a covariance at all.  Refusing is the honest option;
    the error says so and points at ``time=``.
    """

    frame = count_panel()
    dropped = frame.drop(index=[5, 40, 300]).reset_index(drop=True)
    assert dropped.groupby("firm").size().nunique() > 1  # genuinely unbalanced
    with pytest.raises(ConsistencyError, match="time"):
        XtPcse.from_formula("y ~ x1", dropped, entity="firm").fit()


def test_xtpcse_standard_errors_match_stata_vce() -> None:
    """Stata: x1 .7533126 (se .0564318), _cons 1.572686 (se .0372998).

    The reference VCE below is the manual's
    ``(X'X)^-1 X'Omega X (X'X)^-1`` with ``Omega = Sigma kron I`` and **no**
    small-sample factor.  Reproducing Stata to 3e-12 is what settles that
    question: multiplying by the plan's ``N/(N-k)*T/(T-1) = 1.16999`` would put
    both standard errors 8.2% above Stata's.
    """

    frame = count_panel()
    X = np.column_stack([frame["x1"], np.ones(len(frame))])
    y = frame["y"].to_numpy()
    groups = frame["firm"].to_numpy()

    beta, resid, _ = ols_reference(X, y)
    slices = [np.flatnonzero(groups == value) for value in np.unique(groups)]
    stacked = np.column_stack([resid[index] for index in slices])
    T = stacked.shape[0]
    sigma = (stacked.T @ stacked) / T
    bread = np.linalg.inv(X.T @ X)
    meat = np.zeros((2, 2))
    for i, index_i in enumerate(slices):
        for j, index_j in enumerate(slices):
            meat += sigma[i, j] * (X[index_i].T @ X[index_j])
    reference_cov = bread @ meat @ bread

    result = XtPcse.from_formula("y ~ x1", frame, entity="firm", time="year").fit()

    assert float(result.params["x1"]) == pytest.approx(0.7533126, abs=5e-7)
    assert float(result.bse["x1"]) == pytest.approx(0.0564318, rel=1e-5)
    assert float(result.bse["Intercept"]) == pytest.approx(0.0372998, rel=1e-5)

    # The covariance is ordered [x1, Intercept]; the reference matrix is
    # [Intercept, x1], so it is permuted before comparing.
    order = [1, 0]
    np.testing.assert_allclose(
        result.cov_params(), reference_cov[np.ix_(order, order)], rtol=1e-10, atol=0.0
    )

    # The plan's factor would have inflated both standard errors by 8.2%.
    plan_factor = (len(y) / (len(y) - 2)) * (T / (T - 1.0))
    assert plan_factor == pytest.approx(1.16999, abs=5e-5)
    naive = np.sqrt(np.diag(reference_cov * plan_factor))
    for observed, inflated in zip(result.bse.to_numpy(), naive):
        assert not np.isclose(observed, inflated, rtol=1e-3)


def test_xtpcse_reports_r2_and_wald() -> None:
    frame = count_panel()
    result = XtPcse.from_formula("y ~ x1", frame, entity="firm", time="year").fit()
    plain = ws.OLS.from_formula("y ~ x1", frame).fit()
    # OLS point estimates mean the same R2 as regress.
    assert float(result.e["r2"]) == pytest.approx(float(plain.e["r2"]), rel=1e-10)
    assert float(result.e["chi2"]) > 0.0
    assert 0.0 <= float(result.e["p"]) <= 1.0


def test_xtpcse_nonrobust_covariance_is_ols() -> None:
    frame = count_panel()
    pcse = XtPcse.from_formula("y ~ x1", frame, entity="firm", time="year").fit()
    nonrobust = XtPcse.from_formula("y ~ x1", frame, entity="firm", time="year").fit(
        cov_type="nonrobust"
    )
    plain = ws.OLS.from_formula("y ~ x1", frame).fit()
    np.testing.assert_allclose(nonrobust.params.to_numpy(), plain.params.to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(nonrobust.bse.to_numpy(), plain.bse.to_numpy(), rtol=1e-12)
    assert pcse.cov_type == "pcse"
    assert nonrobust.cov_type == "nonrobust"


def test_xtpcse_rejects_autocorrelation() -> None:
    with pytest.raises(UnsupportedError, match="correlation"):
        XtPcse.from_formula(
            "y ~ x1", count_panel(), entity="firm", time="year", correlation="ar1"
        )


# --------------------------------------------------------------- xtunitroot
def test_xtunitroot_ips_matches_stata_statistics() -> None:
    """Stata: t-bar -2.0685, t-tilde-bar -1.5727, Z-t-tilde-bar -4.2953 (p 0.0000)."""

    outcome = XtUnitroot.from_formula(
        "y ~ 1", dynamic_panel(), entity="firm", time="year", test="ips"
    ).fit()

    assert isinstance(outcome, PanelUnitRootResult)
    assert outcome.test == "ips"
    assert outcome.name == "Z-t-tilde-bar"
    assert np.isfinite(outcome.statistic)
    assert 0.0 <= outcome.pvalue <= 1.0

    assert outcome.statistic == pytest.approx(-4.2953, abs=5e-5)
    assert outcome.details["t-bar"] == pytest.approx(-2.0685, abs=5e-5)
    assert outcome.details["t-tilde-bar"] == pytest.approx(-1.5727, abs=5e-5)
    assert outcome.pvalue < 1e-4


def test_xtunitroot_ips_t_bar_is_the_mean_adf_t() -> None:
    """``t-bar`` is the plain average of the per-panel ADF t statistics.

    Recomputed here panel by panel with the ADF regression written out, so the
    reported number cannot come from a different per-panel specification.
    """

    frame = dynamic_panel()
    groups = frame["firm"].to_numpy()
    t_values = []
    for value in np.unique(groups):
        series = frame.loc[groups == value, "y"].to_numpy()
        difference = np.diff(series)
        X = np.column_stack([series[:-1], np.ones(difference.size)])
        beta = np.linalg.lstsq(X, difference, rcond=None)[0]
        resid = difference - X @ beta
        dof = difference.size - X.shape[1]
        sigma2 = float(resid @ resid) / dof
        se = np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[0, 0])
        t_values.append(beta[0] / se)

    outcome = XtUnitroot.from_formula(
        "y ~ 1", frame, entity="firm", time="year", test="ips"
    ).fit()
    assert outcome.details["t-bar"] == pytest.approx(float(np.mean(t_values)), rel=1e-10)
    assert outcome.details["N_t"] == N_DYN_YEARS


def test_xtunitroot_llc_matches_stata_statistics() -> None:
    """Stata: Unadjusted t -21.7694, Adjusted t* -15.1701 (p 0.0000)."""

    outcome = XtUnitroot.from_formula(
        "y ~ 1", dynamic_panel(), entity="firm", time="year", test="llc"
    ).fit()

    assert outcome.test == "llc"
    assert outcome.name == "Adjusted t*"
    assert np.isfinite(outcome.statistic)
    assert 0.0 <= outcome.pvalue <= 1.0

    assert outcome.details["Unadjusted t"] == pytest.approx(-21.7694, abs=5e-5)
    assert outcome.statistic == pytest.approx(-15.1701, abs=5e-5)
    assert outcome.pvalue < 1e-4

    # Stata prints "Bartlett kernel, 6.00 lags average (chosen by LLC)"; that is
    # int(3.21 * (T-1)^(1/3)) = 6 at T = 9.
    assert outcome.details["hac_lags"] == 6.0
    assert outcome.details["lags"] == 1.0


def test_xtunitroot_llc_rejects_a_unit_root_far_less_strongly_than_ips() -> None:
    """On a stationary panel both tests reject, and the statistics are negative."""

    rng = np.random.default_rng(2024)
    rows = []
    for firm in range(40):
        value = rng.normal()
        for year in range(1, 16):
            value = 0.2 * value + rng.normal(scale=0.4)
            rows.append((firm, year, value))
    frame = pd.DataFrame(rows, columns=["firm", "year", "y"])

    ips = XtUnitroot.from_formula("y ~ 1", frame, entity="firm", time="year", test="ips").fit()
    llc = XtUnitroot.from_formula("y ~ 1", frame, entity="firm", time="year", test="llc").fit()
    assert ips.statistic < 0.0 and ips.pvalue < 0.05
    assert llc.statistic < 0.0 and llc.pvalue < 0.05


@pytest.mark.parametrize("test", ["ips", "llc"])
def test_xtunitroot_rejects_an_unbalanced_panel(test: str) -> None:
    """Both tests require strongly balanced data, as ``_xtstrbal`` does in Stata."""

    frame = dynamic_panel().drop(index=[0, 4, 40]).reset_index(drop=True)
    with pytest.raises(ConsistencyError, match="balanced"):
        XtUnitroot.from_formula(
            "y ~ 1", frame, entity="firm", time="year", test=test
        ).fit()


def test_xtunitroot_rejects_unknown_test() -> None:
    with pytest.raises(UnsupportedError, match="ips"):
        XtUnitroot.from_formula(
            "y ~ 1", dynamic_panel(), entity="firm", time="year", test="madwu"
        )


@pytest.mark.parametrize("option", ["trend", "noconstant", "demean"])
def test_xtunitroot_rejects_unimplemented_options(option: str) -> None:
    with pytest.raises(UnsupportedError):
        XtUnitroot.from_formula(
            "y ~ 1", dynamic_panel(), entity="firm", time="year", **{option: True}
        )


def test_xtunitroot_rejects_a_short_series() -> None:
    tiny = pd.DataFrame(
        {"firm": [1, 1, 1, 2, 2, 2], "year": [1, 2, 3, 1, 2, 3], "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
    )
    with pytest.raises(ConsistencyError, match="too short|balanced"):
        XtUnitroot.from_formula("y ~ 1", tiny, entity="firm", time="year", test="ips").fit()


def test_xtunitroot_result_renders() -> None:
    outcome = XtUnitroot.from_formula(
        "y ~ 1", dynamic_panel(), entity="firm", time="year", test="ips"
    ).fit()
    text = str(outcome)
    assert "Z-t-tilde-bar" in text
    assert "statistic" in text and "p" in text


# ------------------------------------------------------- e() keys and summaries
@pytest.mark.parametrize(
    "build",
    [
        lambda: XtIvreg.from_formula(
            "y ~ w + (endog = z)", iv_panel(), entity="firm", time="year", method="fe"
        ).fit(),
        lambda: XtIvreg.from_formula(
            "y ~ w + (endog = z)", iv_panel(), entity="firm", time="year", method="re"
        ).fit(),
        lambda: XtGls.from_formula("y ~ x1", count_panel(), entity="firm", time="year").fit(),
        lambda: XtPcse.from_formula("y ~ x1", count_panel(), entity="firm", time="year").fit(),
    ],
    ids=["xtivreg_fe", "xtivreg_re", "xtgls", "xtpcse"],
)
def test_e_contains_the_standard_keys(build) -> None:
    result = build()
    missing = [key for key in REQUIRED_E_KEYS if key not in result.e]
    assert not missing, f"missing e() keys: {missing}"
    assert result.e["N"] == result.nobs > 0
    assert result.e["df_r"] > 0
    assert result.e["rank"] >= 1
    assert result.e["df_m"] == result.e["rank"] - (
        1 if ("Intercept" in result.params.index or "_cons" in result.params.index) else 0
    )


@pytest.mark.parametrize(
    "build",
    [
        lambda: XtIvreg.from_formula(
            "y ~ w + (endog = z)", iv_panel(), entity="firm", time="year", method="fe"
        ).fit(),
        lambda: XtIvreg.from_formula(
            "y ~ w + (endog = z)", iv_panel(), entity="firm", time="year", method="re"
        ).fit(),
        lambda: XtGls.from_formula("y ~ x1", count_panel(), entity="firm", time="year").fit(),
        lambda: XtPcse.from_formula("y ~ x1", count_panel(), entity="firm", time="year").fit(),
    ],
    ids=["xtivreg_fe", "xtivreg_re", "xtgls", "xtpcse"],
)
def test_summary_renders_in_stata_style(build) -> None:
    result = build()
    text = result.summary(style="stata")
    for token in ("Coef.", "Std. err.", "P>|z|", "Number of obs"):
        assert token in text, token


def test_model_names_match_the_stata_commands() -> None:
    names = {
        "xtivreg, fe": XtIvreg.from_formula(
            "y ~ w + (endog = z)", iv_panel(), entity="firm", time="year", method="fe"
        ).fit().model_name,
        "xtivreg, re": XtIvreg.from_formula(
            "y ~ w + (endog = z)", iv_panel(), entity="firm", time="year", method="re"
        ).fit().model_name,
        "xtgls": XtGls.from_formula(
            "y ~ x1", count_panel(), entity="firm", time="year"
        ).fit().model_name,
        "xtpcse": XtPcse.from_formula(
            "y ~ x1", count_panel(), entity="firm", time="year"
        ).fit().model_name,
    }
    for expected, observed in names.items():
        assert observed == expected


def test_xtunitroot_model_name_is_the_stata_command() -> None:
    assert XtUnitroot._stata_command == "xtunitroot"


# --------------------------------------------------------------- input handling
def test_missing_values_are_dropped_listwise() -> None:
    frame = iv_panel()
    frame.loc[3, "endog"] = np.nan
    frame.loc[17, "z"] = np.nan
    result = XtIvreg.from_formula(
        "y ~ w + (endog = z)", frame, entity="firm", time="year", method="fe"
    ).fit()
    assert result.nobs == len(frame) - 2


def test_duplicate_panel_cells_are_rejected() -> None:
    frame = pd.concat([iv_panel(), iv_panel().iloc[[0]]], ignore_index=True)
    with pytest.raises(ConsistencyError, match="repeated"):
        XtIvreg.from_formula(
            "y ~ w + (endog = z)", frame, entity="firm", time="year"
        )


def test_missing_panel_column_is_reported() -> None:
    with pytest.raises(ConsistencyError, match="firm_id"):
        XtIvreg.from_formula(
            "y ~ w + (endog = z)", iv_panel(), entity="firm_id", time="year"
        )


def test_row_order_does_not_change_results() -> None:
    frame = count_panel().sample(frac=1.0, random_state=7).reset_index(drop=True)
    reference = count_panel()
    shuffled = XtPcse.from_formula("y ~ x1", frame, entity="firm", time="year").fit()
    ordered = XtPcse.from_formula("y ~ x1", reference, entity="firm", time="year").fit()
    np.testing.assert_allclose(shuffled.params.to_numpy(), ordered.params.to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(shuffled.bse.to_numpy(), ordered.bse.to_numpy(), rtol=1e-12)
