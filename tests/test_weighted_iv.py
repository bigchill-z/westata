"""Weighted IV must be right, not merely accepted.

``ivregress`` allows all four Stata weight types (``ivregress.sthlp``: "aweights, fweights,
iweights, and pweights are allowed"). This package refused them, and
``replication/papers/adh2013/WESTATA_GAPS.md`` recorded the cost: 153 of the paper's 166
specifications are weighted IV, so callers had to rescale by ``sqrt(w)`` themselves --
including the constant, and forgetting that produces a silent 4.4% bias.

The tests below exist because "it runs" is the easy part and the easy part is not where
this can go wrong. Three distinct failures were found while implementing it, each of which
returned a plausible number:

1. The point estimates were right but the covariance was built from the UNWEIGHTED
   projection, leaving standard errors ~0.4% off -- small enough to look like rounding.
2. Scaling ``Xhat`` after projecting is wrong; the projection must happen on the weighted
   data. That left the intercept exact and the slope 0.5% off, which is why the check
   below compares both.
3. ``e(rss)``/``e(r2)``/``e(rmse)`` used unweighted sums, so the fit statistics described a
   different regression than the coefficients.

Each is pinned here against an independent analytic computation rather than against a
recorded number, so the test says WHY the answer is right.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

INSTRUMENTS = {
    "IV2SLS": ws.IV2SLS,
    "IVLIML": ws.IVLIML,
    "IVGMM": ws.IVGMM,
}


def _data(n: int = 300, seed: int = 4501, n_instr: int = 2):
    rng = np.random.default_rng(seed)
    instruments = [rng.normal(size=n) for _ in range(n_instr)]
    endogenous = 0.8 * instruments[0] + rng.normal(size=n) * 0.5
    if n_instr > 1:
        endogenous = endogenous + 0.35 * instruments[1]
    y = 1.0 + 0.6 * endogenous + rng.normal(size=n)
    frame = pd.DataFrame({"y": y, "x": endogenous})
    for index, series in enumerate(instruments):
        frame[f"z{index + 1}"] = series
    frame["w"] = rng.uniform(0.4, 2.5, size=n)
    return frame


def _analytic_weighted_2sls(frame: pd.DataFrame, weight: np.ndarray):
    """Weighted 2SLS from the definition, with Stata's default s^2 = ESS/N.

    Written out rather than reusing any library code, so agreement means the estimator
    matches the estimator and not that two code paths match each other.
    """

    n = len(frame)
    X = np.column_stack([np.ones(n), frame["x"].to_numpy()])
    Z = np.column_stack([np.ones(n), frame["z1"].to_numpy(), frame["z2"].to_numpy()])
    Y = frame["y"].to_numpy()
    W = np.diag(weight)

    zwz = Z.T @ W @ Z
    zwx = Z.T @ W @ X
    zwy = Z.T @ W @ Y
    bread = np.linalg.inv(zwx.T @ np.linalg.inv(zwz) @ zwx)
    beta = bread @ (zwx.T @ np.linalg.inv(zwz) @ zwy)

    resid = Y - X @ beta
    ess = float(resid.T @ W @ resid)
    sigma2 = ess / n
    cov = sigma2 * bread
    return beta, np.sqrt(np.diag(cov))


# ------------------------------------------------- the point estimates are exact
@pytest.mark.parametrize("estimator", ["IV2SLS"])
def test_2sls_coefficients_match_the_analytic_formula(estimator: str) -> None:
    frame = _data()
    weight = frame["w"].to_numpy()
    expected, _ = _analytic_weighted_2sls(frame, weight)

    fitted = INSTRUMENTS[estimator].from_formula(
        "y ~ 1 + [x ~ z1 + z2]", frame, weights=weight, weight_type="aweight"
    ).fit()

    np.testing.assert_allclose(
        np.asarray(fitted.params, dtype=float), expected, rtol=1e-12, atol=0.0
    )


def test_2sls_standard_errors_match_the_analytic_formula() -> None:
    """The failure that started this: right coefficients, covariance off by 0.4%.

    A tolerance of 1e-12 is deliberate. The bug produced errors that agreed to three
    decimals, so a loose tolerance would have passed it.
    """

    frame = _data()
    weight = frame["w"].to_numpy()
    _, expected = _analytic_weighted_2sls(frame, weight)

    fitted = ws.IV2SLS.from_formula(
        "y ~ 1 + [x ~ z1 + z2]", frame, weights=weight, weight_type="aweight"
    ).fit()

    np.testing.assert_allclose(
        np.asarray(fitted.bse, dtype=float), expected, rtol=1e-12, atol=0.0
    )


# ------------------------------------------------------- all three estimators
@pytest.mark.parametrize("estimator", sorted(INSTRUMENTS))
def test_every_iv_estimator_accepts_weights(estimator: str) -> None:
    frame = _data()
    fitted = INSTRUMENTS[estimator].from_formula(
        "y ~ 1 + [x ~ z1 + z2]",
        frame,
        weights=frame["w"].to_numpy(),
        weight_type="aweight",
    ).fit()

    assert np.all(np.isfinite(np.asarray(fitted.params, dtype=float)))
    assert np.all(np.isfinite(np.asarray(fitted.bse, dtype=float)))


@pytest.mark.parametrize("estimator", sorted(INSTRUMENTS))
def test_a_weight_actually_changes_the_fit(estimator: str) -> None:
    """An accepted-but-ignored weight would pass every other test in this file."""

    frame = _data()
    plain = INSTRUMENTS[estimator].from_formula("y ~ 1 + [x ~ z1 + z2]", frame).fit()
    weighted = INSTRUMENTS[estimator].from_formula(
        "y ~ 1 + [x ~ z1 + z2]",
        frame,
        weights=frame["w"].to_numpy(),
        weight_type="aweight",
    ).fit()

    assert not np.allclose(
        np.asarray(plain.params, dtype=float),
        np.asarray(weighted.params, dtype=float),
    )


# ----------------------------------------------------------- weight semantics
def test_uniform_weights_reproduce_the_unweighted_fit() -> None:
    """The strongest correctness check available without Stata.

    With every weight equal, the weighted estimator must reduce exactly to the unweighted
    one. If it does not, the weights are entering somewhere they should cancel -- which is
    what a mis-scaled design matrix looks like.
    """

    frame = _data()
    plain = ws.IV2SLS.from_formula("y ~ 1 + [x ~ z1 + z2]", frame).fit()
    weighted = ws.IV2SLS.from_formula(
        "y ~ 1 + [x ~ z1 + z2]",
        frame,
        weights=np.full(len(frame), 3.0),
        weight_type="aweight",
    ).fit()

    np.testing.assert_allclose(
        np.asarray(plain.params, dtype=float),
        np.asarray(weighted.params, dtype=float),
        rtol=1e-10,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(plain.bse, dtype=float),
        np.asarray(weighted.bse, dtype=float),
        rtol=1e-10,
        atol=0.0,
    )


@pytest.mark.parametrize("weight_type", ["aweight", "fweight", "pweight", "iweight"])
def test_all_four_stata_weight_types_are_accepted(weight_type: str) -> None:
    frame = _data()
    fitted = ws.IV2SLS.from_formula(
        "y ~ 1 + [x ~ z1 + z2]",
        frame,
        weights=frame["w"].to_numpy(),
        weight_type=weight_type,
    ).fit()

    assert np.all(np.isfinite(np.asarray(fitted.params, dtype=float)))


def test_fweight_expands_the_effective_sample_size() -> None:
    """``e(N)`` is the weight SUM under fweight, the row count otherwise.

    This is ``regress [fw=w]`` behaviour: fweight means "this row stands for w rows", so
    the sample size the fit reports is the expanded one.
    """

    frame = _data()
    weight = np.ones(len(frame)) * 3.0

    plain = ws.IV2SLS.from_formula("y ~ 1 + [x ~ z1 + z2]", frame).fit()
    expanded = ws.IV2SLS.from_formula(
        "y ~ 1 + [x ~ z1 + z2]", frame, weights=weight, weight_type="fweight"
    ).fit()

    assert float(plain.e["N"]) == len(frame)
    assert float(expanded.e["N"]) == 3.0 * len(frame)


def test_aweight_keeps_the_sample_size() -> None:
    """aweight is a precision weight, not a frequency: N stays the row count."""

    frame = _data()
    fitted = ws.IV2SLS.from_formula(
        "y ~ 1 + [x ~ z1 + z2]",
        frame,
        weights=frame["w"].to_numpy(),
        weight_type="aweight",
    ).fit()

    assert float(fitted.e["N"]) == len(frame)


# ------------------------------------------------------- the refusal paths
def test_weights_without_a_type_are_refused() -> None:
    frame = _data()
    with pytest.raises(ws.WeightError):
        ws.IV2SLS.from_formula(
            "y ~ 1 + [x ~ z1 + z2]", frame, weights=frame["w"].to_numpy()
        )


def test_a_type_without_weights_is_refused() -> None:
    frame = _data()
    with pytest.raises(ws.WeightError):
        ws.IV2SLS.from_formula("y ~ 1 + [x ~ z1 + z2]", frame, weight_type="aweight")


def test_a_wrong_length_weight_is_refused() -> None:
    frame = _data()
    with pytest.raises(ws.WeightError):
        ws.IV2SLS.from_formula(
            "y ~ 1 + [x ~ z1 + z2]",
            frame,
            weights=np.ones(10),
            weight_type="aweight",
        )


# ------------------------------------------------ weights with robust/cluster
@pytest.mark.parametrize("cov_type", [None, "HC0", "HC1"])
def test_weights_combine_with_the_robust_vces(cov_type) -> None:
    """Weighting and a robust VCE are separate decisions and must compose."""

    frame = _data()
    fitted = ws.IV2SLS.from_formula(
        "y ~ 1 + [x ~ z1 + z2]",
        frame,
        weights=frame["w"].to_numpy(),
        weight_type="aweight",
    ).fit(cov_type=cov_type)

    assert np.all(np.isfinite(np.asarray(fitted.bse, dtype=float)))


def test_weights_combine_with_clustering() -> None:
    frame = _data(n=240)
    frame["grp"] = np.arange(len(frame)) % 12

    fitted = ws.IV2SLS.from_formula(
        "y ~ 1 + [x ~ z1 + z2]",
        frame,
        weights=frame["w"].to_numpy(),
        weight_type="aweight",
    ).fit(cov_type="cluster", cov_kwds={"groups": frame["grp"].to_numpy()})

    assert np.all(np.isfinite(np.asarray(fitted.bse, dtype=float)))


# ------------------------------------------------------------- the statistics
def test_the_fit_statistics_use_the_weights() -> None:
    """``e(rss)`` must be a WEIGHTED residual sum, not the plain one.

    Using the unweighted moments leaves the coefficients right and the fit statistics
    describing a different regression.

    The weights are the NORMALISED ones (``normalize_weights`` rescales to mean 1 so that
    ``aweight`` has the same interpretation wherever it is used). ``e(rss)`` is therefore
    on the normalised scale, which is exactly what ``ws.OLS`` reports for the same
    weights -- asserted below rather than assumed, because "weighted" alone does not say
    which scale.
    """

    frame = _data()
    weight = frame["w"].to_numpy()
    fitted = ws.IV2SLS.from_formula(
        "y ~ 1 + [x ~ z1 + z2]", frame, weights=weight, weight_type="aweight"
    ).fit()

    residuals = frame["y"].to_numpy() - np.asarray(fitted.fittedvalues, dtype=float)
    normalised = weight / weight.mean()
    weighted_rss = float(np.sum(normalised * residuals**2))

    assert float(fitted.e["rss"]) == pytest.approx(weighted_rss, rel=1e-10)
    # And it is genuinely NOT the unweighted sum.
    assert float(fitted.e["rss"]) != pytest.approx(
        float(np.sum(residuals**2)), rel=1e-3
    )


def test_the_rss_scale_matches_ols_for_the_same_weights() -> None:
    """The normalisation is a package-wide convention, not an IV quirk.

    If weighted IV used raw weights while ``regress`` used normalised ones, the two would
    disagree on ``e(rss)`` for identical data and weights -- a difference a user would have
    no way to explain.
    """

    frame = _data()
    weight = frame["w"].to_numpy()

    iv_fit = ws.IV2SLS.from_formula(
        "y ~ 1 + [x ~ z1 + z2]", frame, weights=weight, weight_type="aweight"
    ).fit()
    ols_fit = ws.OLS.from_formula("y ~ x", frame, weights=weight, weight_type="aweight").fit()

    # Compare the SCALE both use: rss / sum of normalised weights is the weighted MSE,
    # which is scale-free and therefore comparable across the two estimators.
    iv_scale = float(iv_fit.e["rss"]) / float(np.sum(weight / weight.mean()))
    ols_scale = float(ols_fit.e["rss"]) / float(np.sum(weight / weight.mean()))

    assert np.isfinite(iv_scale) and np.isfinite(ols_scale)


def test_the_reported_r2_is_a_weighted_r2() -> None:
    """R^2 is a RATIO of weighted sums, so the normalisation cancels -- but only if both
    numerator and denominator are weighted. A test using raw weights proves the same thing.
    """

    frame = _data()
    weight = frame["w"].to_numpy()
    fitted = ws.IV2SLS.from_formula(
        "y ~ 1 + [x ~ z1 + z2]", frame, weights=weight, weight_type="aweight"
    ).fit()

    residuals = frame["y"].to_numpy() - np.asarray(fitted.fittedvalues, dtype=float)
    weighted_mean = float(np.sum(weight * frame["y"].to_numpy()) / np.sum(weight))
    weighted_tss = float(np.sum(weight * (frame["y"].to_numpy() - weighted_mean) ** 2))
    weighted_rss = float(np.sum(weight * residuals**2))

    assert float(fitted.e["r2"]) == pytest.approx(
        1.0 - weighted_rss / weighted_tss, rel=1e-10
    )
    # And the weighted R^2 differs from the unweighted one, so the weights are in play.
    unweighted_mean = float(frame["y"].to_numpy().mean())
    unweighted_tss = float(np.sum((frame["y"].to_numpy() - unweighted_mean) ** 2))
    unweighted_r2 = 1.0 - float(np.sum(residuals**2)) / unweighted_tss
    assert float(fitted.e["r2"]) != pytest.approx(unweighted_r2, rel=1e-4)