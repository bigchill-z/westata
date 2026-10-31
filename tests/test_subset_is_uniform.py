"""``subset=`` must mean the same thing on every estimator.

``WESTATA_GAPS.md`` §2 recorded this as a P1 inconsistency: ``OLS``/``Logit``/``Poisson``
accepted ``subset=`` while the IV, HDFE and panel estimators did not, so "estimate on a
subset" needed a different spelling depending on which model you were fitting.

Two things are tested, and the second is the one that matters:

1. **Every estimator accepts it.** Otherwise the inconsistency returns.
2. **It agrees with pre-filtering the frame by hand.** An implementation that accepts the
   argument and then ignores it, or applies it after the design matrix is built, would pass
   the first test and silently estimate on the full sample. That is the failure mode worth
   spending a test on: the call succeeds and the number is wrong.

The second check is what makes this more than a signature test. ``IV2SLS.from_formula``
takes ``**kwargs``, so ``inspect.signature`` never lists ``subset`` even though the call
works -- which is exactly how the replication probe came to report a working path as
missing, and why the tests below call the estimators instead of inspecting them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


def _frame(n: int = 200, units: int = 20, seed: int = 8801) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    per = n // units
    covariate = rng.normal(size=n)
    latent = 0.4 + 0.9 * covariate
    return pd.DataFrame(
        {
            "y": 1.0 + 0.8 * rng.normal(size=n),
            # A 0/1 response for Logit; the count models get their own column so that no
            # estimator is handed data outside its domain (a linear ``y`` into Logit raises
            # "endog must be in the unit interval", which is a test-construction error and
            # not a ``subset=`` problem).
            "yb": (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-latent))).astype(int),
            "ycnt": rng.poisson(np.exp(0.3 + 0.4 * covariate)),
            "x": covariate,
            "z": rng.normal(size=n),
            "e": 0.7 * rng.normal(size=n) + rng.normal(size=n) * 0.4,
            "firm": np.repeat(np.arange(units), per),
            "year": np.tile(np.arange(1, per + 1), units),
        }
    )


@pytest.fixture(scope="module")
def frame():
    return _frame()


MASK = lambda frame: frame["x"] > 0.0  # noqa: E731 - a tiny helper, not a fixture


#: name -> (builder taking subset=, builder taking a pre-filtered frame)
BUILDERS = {
    "OLS": (
        lambda frame, mask=None: ws.OLS.from_formula(
            "y ~ x", frame, **({} if mask is None else {"subset": mask})
        ).fit(),
        lambda frame: ws.OLS.from_formula("y ~ x", frame).fit(),
    ),
    "Logit": (
        lambda frame, mask=None: ws.Logit.from_formula(
            "yb ~ x", frame, **({} if mask is None else {"subset": mask})
        ).fit(),
        lambda frame: ws.Logit.from_formula("yb ~ x", frame).fit(),
    ),
    "Poisson": (
        lambda frame, mask=None: ws.Poisson.from_formula(
            "ycnt ~ x", frame, **({} if mask is None else {"subset": mask})
        ).fit(),
        lambda frame: ws.Poisson.from_formula("ycnt ~ x", frame).fit(),
    ),
    "IV2SLS": (
        lambda frame, mask=None: ws.IV2SLS.from_formula(
            "y ~ 1 + [e ~ z]", frame, **({} if mask is None else {"subset": mask})
        ).fit(),
        lambda frame: ws.IV2SLS.from_formula("y ~ 1 + [e ~ z]", frame).fit(),
    ),
    "IVLIML": (
        lambda frame, mask=None: ws.IVLIML.from_formula(
            "y ~ 1 + [e ~ z]", frame, **({} if mask is None else {"subset": mask})
        ).fit(),
        lambda frame: ws.IVLIML.from_formula("y ~ 1 + [e ~ z]", frame).fit(),
    ),
    "FEOLS": (
        lambda frame, mask=None: ws.FEOLS.from_formula(
            "y ~ x | firm", frame, **({} if mask is None else {"subset": mask})
        ).fit(),
        lambda frame: ws.FEOLS.from_formula("y ~ x | firm", frame).fit(),
    ),
    "PanelOLS": (
        lambda frame, mask=None: ws.PanelOLS.from_formula(
            "y ~ x",
            frame,
            entity="firm",
            time="year",
            **({} if mask is None else {"subset": mask}),
        ).fit(),
        lambda frame: ws.PanelOLS.from_formula(
            "y ~ x", frame, entity="firm", time="year"
        ).fit(),
    ),
}


# ------------------------------------------------------- it is accepted at all
@pytest.mark.parametrize("estimator", sorted(BUILDERS))
def test_subset_is_accepted(frame, estimator: str) -> None:
    """The consistency fix: one spelling for one operation."""

    with_subset, _ = BUILDERS[estimator]
    fitted = with_subset(frame, MASK(frame))

    assert fitted is not None


# ------------------------------------------ and it does what pre-filtering does
@pytest.mark.parametrize("estimator", sorted(BUILDERS))
def test_subset_agrees_with_prefiltering_the_frame(frame, estimator: str) -> None:
    """The check that catches "accepted and ignored".

    If ``subset=`` were silently dropped, the two fits would differ -- the full sample
    against the restricted one. Comparing coefficients AND standard errors covers both the
    estimate and the sample size it was computed from.
    """

    with_subset, prefiltered = BUILDERS[estimator]
    mask = MASK(frame)

    restricted = with_subset(frame, mask)
    manual = prefiltered(frame.loc[mask].reset_index(drop=True))

    np.testing.assert_allclose(
        np.asarray(restricted.params, dtype=float),
        np.asarray(manual.params, dtype=float),
        rtol=1e-10,
        atol=1e-12,
        err_msg=f"{estimator}: subset= disagrees with pre-filtering",
    )
    np.testing.assert_allclose(
        np.asarray(restricted.bse, dtype=float),
        np.asarray(manual.bse, dtype=float),
        rtol=1e-8,
        atol=1e-12,
        err_msg=f"{estimator}: subset= changed the standard errors",
    )


@pytest.mark.parametrize("estimator", sorted(BUILDERS))
def test_subset_actually_drops_rows(frame, estimator: str) -> None:
    """Guard: if the mask were all-True this test would be vacuous, so it is not."""

    with_subset, _ = BUILDERS[estimator]
    mask = MASK(frame)
    assert 0 < int(mask.sum()) < len(frame)

    restricted = with_subset(frame, mask)

    assert float(restricted.e["N"]) <= float(mask.sum())


# ------------------------------------------------------------- mask and index
def test_a_boolean_mask_and_an_index_agree(frame) -> None:
    """``subset`` takes either a mask or labels, matching ``OLS``."""

    mask = MASK(frame)
    by_mask = ws.OLS.from_formula("y ~ x", frame, subset=mask).fit()
    by_index = ws.OLS.from_formula("y ~ x", frame, subset=frame.index[mask]).fit()

    np.testing.assert_allclose(
        np.asarray(by_mask.params, dtype=float),
        np.asarray(by_index.params, dtype=float),
        rtol=1e-12,
        atol=0.0,
    )


# ------------------------------------------------------------ panel specifics
def test_the_panel_estimator_keeps_a_usable_frame_under_subset(frame) -> None:
    """``PanelOLS`` needs consecutive entity/time levels after restriction."""

    mask = MASK(frame)
    fitted = ws.PanelOLS.from_formula(
        "y ~ x", frame, entity="firm", time="year", subset=mask
    ).fit()

    assert np.all(np.isfinite(np.asarray(fitted.bse, dtype=float)))
    assert float(fitted.e["N"]) > 0


def test_the_hdfe_predict_still_works_under_subset(frame) -> None:
    """The absorbed-effect prediction reads the estimation frame, so it must follow."""

    mask = MASK(frame)
    fitted = ws.FEOLS.from_formula("y ~ x | firm", frame, subset=mask).fit()

    predicted = np.asarray(fitted.predict(kind="xb"), dtype=float)
    assert predicted.shape == (int(mask.sum()),)
    assert np.all(np.isfinite(predicted))


# -------------------------------------------------------------- the boundaries
def test_a_subset_that_selects_nothing_is_refused_rather_than_silently_empty(frame) -> None:
    """Estimating on zero rows must fail loudly, not return NaN parameters."""

    empty = frame["x"] > 1e9
    with pytest.raises(Exception):
        ws.OLS.from_formula("y ~ x", frame, subset=empty).fit()


def test_no_subset_means_the_full_sample(frame) -> None:
    fitted = ws.OLS.from_formula("y ~ x", frame).fit()

    assert float(fitted.e["N"]) == len(frame)