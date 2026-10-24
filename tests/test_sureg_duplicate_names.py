"""Regression test: sureg must not produce duplicate parameter names.

Found while testing the built wheel -- the package installed cleanly, imported correctly, and then failed on a
model that is perfectly valid:

    ws.Sureg.from_formula(["y ~ x1", "y ~ x2"], frame).fit()
    ValueError: param_names must be unique

Both equations have outcome `y`, so both were labelled `y` and the intercept from each became `y:_cons` -- the same
string twice. Stata avoids this because it names equations after the dependent variable and would reject a
duplicate equation name outright; we keep the call working by disambiguating the second.

Two things are pinned, and the second matters as much as the first:

  * the duplicate case now works and produces unique names;
  * the ORDINARY case is unchanged -- distinct outcomes still give `price:...` / `mpg:...`, no `#2` suffixes.

Without the second assertion a fix that renamed everything would still pass the first.

A third check was written and then REMOVED because it asserted something false: that the first block's
coefficients must equal a single-equation fit. They need not -- SUR re-estimates both equations jointly using the
cross-equation covariance, so adding a second equation legitimately changes the first equation's estimates. The
assertion failed because the test was wrong, not the code. What is asserted instead is what actually holds: the
disambiguation is purely a labelling change, so the NUMBERS must be identical to the same system estimated without
any name collision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import westata as ws


def _frame(n: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 0.8 * x1 - 0.4 * x2 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "y2": rng.normal(size=n), "x1": x1, "x2": x2})


def test_sureg_with_a_shared_outcome_name_produces_unique_parameters():
    """The case that crashed: two equations on the same dependent column."""

    result = ws.Sureg.from_formula(["y ~ x1", "y ~ x2"], _frame()).fit()
    names = list(result.param_names)
    assert len(names) == len(set(names)), f"duplicate parameter names: {names}"
    assert "y:x1" in names
    assert "y#2:x2" in names


def test_sureg_keeps_plain_names_when_outcomes_differ():
    """The ordinary case must be untouched, or the fix renamed everything."""

    result = ws.Sureg.from_formula(["y ~ x1", "y2 ~ x2"], _frame()).fit()
    names = list(result.param_names)
    assert len(names) == len(set(names))
    assert names == ["y:x1", "y:_cons", "y2:x2", "y2:_cons"]


def test_sureg_name_disambiguation_does_not_change_the_estimates():
    """Labelling only: the numbers must match the same system built to avoid a collision.

    Rows of the shared-outcome system are given a distinct name via a renamed copy of the second outcome, which
    leaves the linear algebra identical -- the same values, the same design, the same joint estimation. Only the
    label differs. So the coefficients and the covariance must be equal element for element.
    """

    frame = _frame()
    shared = ws.Sureg.from_formula(["y ~ x1", "y ~ x2"], frame).fit()

    # Same two regressions, but the second outcome column is duplicated under a new name so no label collides.
    renamed = frame.copy()
    renamed["y_dup"] = frame["y"]
    control = ws.Sureg.from_formula(["y ~ x1", "y_dup ~ x2"], renamed).fit()

    np.testing.assert_allclose(
        np.asarray(shared.params, dtype=float),
        np.asarray(control.params, dtype=float),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(shared.cov_params(), dtype=float),
        np.asarray(control.cov_params(), dtype=float),
        rtol=1e-12,
    )

    covariance = np.asarray(shared.cov_params(), dtype=float)
    assert np.all(np.isfinite(covariance))
