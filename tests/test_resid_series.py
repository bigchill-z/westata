"""``resid_series``/``fitted_series``: labelled residuals as ADDITIONS (report section 6).

The audit wants ``resid`` and ``fittedvalues`` to be Series, as statsmodels returns them.
Replacing them would break positional access -- the same cost round 19 measured on ``beta`` --
so the labelled forms are separate names, and the originals are untouched.

KNOWN LIMIT, recorded rather than hidden: the index is the FIT's own row labels, which the
pipeline re-numbers from zero. A caller who set a custom DataFrame index will see 0..n-1 here,
not their labels, because the fitter resets the index before building the design. Attaching the
caller's original labels would mean threading it through the fitting path; the numbers are right
either way, and the index says only "row i".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(7)
    n = 50
    frame = pd.DataFrame({"x": rng.normal(size=n)})
    frame["y"] = 1.0 + 0.8 * frame["x"] + rng.normal(size=n)
    return ws.OLS.from_formula("y ~ x", frame).fit()


def test_resid_is_still_a_positional_ndarray(fitted) -> None:
    assert isinstance(fitted.resid, np.ndarray)
    assert float(fitted.resid[0]) != 0.0


def test_fittedvalues_is_still_a_positional_ndarray(fitted) -> None:
    assert isinstance(fitted.fittedvalues, np.ndarray)


def test_resid_series_is_labelled(fitted) -> None:
    series = fitted.resid_series
    assert isinstance(series, pd.Series)
    assert series.name == "resid"
    assert len(series) == 50


def test_fitted_series_is_labelled(fitted) -> None:
    series = fitted.fitted_series
    assert isinstance(series, pd.Series)
    assert series.name == "fittedvalues"


def test_the_series_carry_the_same_numbers(fitted) -> None:
    """Two views of one fit; a disagreement would be the worst outcome."""

    assert np.allclose(np.asarray(fitted.resid_series), fitted.resid)
    assert np.allclose(np.asarray(fitted.fitted_series), fitted.fittedvalues)


def test_the_residuals_still_sum_to_zero(fitted) -> None:
    assert float(fitted.resid_series.sum()) == pytest.approx(0.0, abs=1e-9)
