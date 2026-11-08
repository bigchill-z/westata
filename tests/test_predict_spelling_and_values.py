"""Stata's singular `predict, residual` was rejected; both spellings now work.

Report item C04-04 scores 0.0 for `predict e1, residual ; summarize e1`. The capability existed but
the documented spelling was refused:

    predict: 'regress' supports ('xb', 'stdp', 'stdf', 'residuals', ...); got 'residual'

The registry stores the plural, inherited from statsmodels, while Stata documents the singular. Fixed
by normalising at the `predict` entry rather than by renaming the registry, so every estimator that
already spoke "residuals" keeps working.

Verified against a live Stata on grilic.dta (`reg lnw s expr`, then `predict e1, residual`,
`summarize e1`):

    Obs 758   Mean 7.03e-11   Std. dev. .3594254

The package gives a mean of zero to floating precision and 0.3594254 for the standard deviation.

C05-08 (`predict lnw1`, the default xb), C09-04 (`predict lev, leverage`) and C11-08
(`predict p`, probability) were checked in the same pass and already work -- their 0.0 scores reflect
that the report compares output LAYOUT, not that the prediction is absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

GRILIC = r"D:\myCodeProject\test_westata\data\grilic.dta"


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(GRILIC).dropna(subset=["lnw", "s", "expr", "rns"])
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("grilic.dta is not available")
    return ws.OLS.from_formula("lnw ~ s + expr", frame).fit()


def test_the_singular_spelling_is_accepted(fitted) -> None:
    """`predict e1, residual` -- what Stata documents and C04-04 uses."""

    values = fitted.predict(kind="residual")
    assert len(values) == 758


def test_both_spellings_agree(fitted) -> None:
    singular = np.asarray(fitted.predict(kind="residual"), dtype=float)
    plural = np.asarray(fitted.predict(kind="residuals"), dtype=float)
    assert np.allclose(singular, plural, atol=0.0)


def test_the_residual_standard_deviation_matches_stata(fitted) -> None:
    """Stata's `summarize e1` reports Std. dev. .3594254 on this sample."""

    values = np.asarray(fitted.predict(kind="residual"), dtype=float)
    assert float(values.std(ddof=1)) == pytest.approx(0.3594254, abs=1e-7)
    # OLS residuals are orthogonal to the constant, so the mean is zero to floating precision.
    assert abs(float(values.mean())) < 1e-10


def test_the_other_three_items_already_worked(fitted) -> None:
    """C05-08 (xb), C09-04 (leverage) and C11-08 (pr) all return usable vectors."""

    xb = np.asarray(fitted.predict(kind="xb"), dtype=float)
    leverage = np.asarray(fitted.predict(kind="leverage"), dtype=float)

    assert len(xb) == 758
    assert len(leverage) == 758
    # Leverages live in (0, 1) and sum to the parameter count.
    assert np.all(leverage > 0.0) and np.all(leverage < 1.0)
    assert float(leverage.sum()) == pytest.approx(3.0, abs=1e-8)


def test_a_logit_probability_is_in_range() -> None:
    """C11-08's `predict p` on a binary model."""

    try:
        frame = pd.read_stata(GRILIC).dropna(subset=["rns", "s", "expr"])
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("grilic.dta is not available")

    logit = ws.Logit.from_formula("rns ~ s + expr", frame).fit()
    probability = np.asarray(logit.predict(kind="pr"), dtype=float)

    assert len(probability) == len(frame)
    assert np.all(probability >= 0.0) and np.all(probability <= 1.0)
