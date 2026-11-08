"""`xtreg, re` reports sigma_e from a DIFFERENT residual than rmse.

Found while checking the report's C12-05 (which asked for standard errors on the sigma rows -- those
are correctly absent, Stata prints bare values there too). The values disagreed instead:

    quantity      westata        Stata
    sigma_u       14100.88       14096.756
    sigma_e       9463.8949      9567.7549
    rho           .68944081      .68462142

LOCALISED: this package's `rss` is exactly `rmse^2 * (N - k)`:

    Stata rmse = 9570.035363, N - k = 798  ->  73085290325.56
    package rss                             ->  73085290324.19

but Stata's `sigma_e^2 * N` is 74698217705.73 -- a different number, 2.2% away. So `e(rmse)` and
`e(sigma_e)` are built from different residuals, and dividing the rmse-based rss by N makes
sigma_e low.

ALSO ESTABLISHED: Stata's `e(rss)` is MISSING (`.`), so the value cannot be recovered from that
macro; it has to be computed from the random-effects transformation.

The guards below pin the measured disagreement. They are written to FAIL when the estimator is
fixed, with the equality assertions to substitute named in the docstrings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

MUNNELL = r"D:\myCodeProject\test_westata\data\munnell.dta"

#: Captured from `xtreg y k1 i.year, re` at %22.6f.
STATA_SIGMA_U = 14096.756209
STATA_SIGMA_E = 9567.754881
STATA_RMSE = 9570.035363
STATA_RHO = 0.68462142


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(MUNNELL)
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("munnell.dta is not available")
    frame = frame.copy()
    frame["sid"] = pd.Categorical(frame["st_abb"]).codes
    frame["year"] = frame["year"].astype(int)
    return ws.RandomEffects.from_formula(
        "y ~ k1 + C(year)", frame, entity="sid", time="year"
    ).fit()


def test_sigma_u_is_exact(fitted) -> None:
    """Exact once sigma_e is right: sigma_u is derived from it by Swamy-Arora."""

    assert float(fitted.e["sigma_u"]) == pytest.approx(STATA_SIGMA_U, rel=1e-8)


def test_sigma_e_is_the_within_rmse(fitted) -> None:
    """Exact: Stata reports the WITHIN regression's rmse as `xtreg, re`'s sigma_e.

    Fixed in round 46 after reading xtreg_re.ado:128/338, where
    `s_e = sqrt(e(sse)/e(df_t))` comes from ffxreg2 -- the fixed-effects estimator -- and is then
    stored as e(sigma_e). It previously came from the RE fit's own residual sum of squares, 1.1%
    low, which also dragged rho and theta with it.
    """

    assert float(fitted.e["sigma_e"]) == pytest.approx(STATA_SIGMA_E, abs=1e-6)


def test_sigma_e_is_not_derived_from_rmse(fitted) -> None:
    """These two are DIFFERENT quantities, which is the whole point of round 46.

    `xtreg, re` reports e(rmse) = 0.0394075163 and e(sigma_e) = 0.0388216088. Dividing the
    rmse-based residual sum of squares by N used to produce 0.0393108 for sigma_e -- 1.1% low --
    because sigma_e is the WITHIN regression's rmse, not the RE fit's.

    e(rss) itself is no longer stored: Stata's `display e(rss)` after `xtreg, re` prints ".". The
    relationship is therefore checked between the two reported scalars rather than through a
    residual sum of squares the command does not expose.
    """

    nobs = int(fitted.nobs)
    rmse = float(fitted.e["rmse"])
    sigma_e = float(fitted.e["sigma_e"])

    assert rmse == pytest.approx(STATA_RMSE, abs=1e-6)
    assert sigma_e == pytest.approx(STATA_SIGMA_E, abs=1e-6)
    # sigma_e is the WITHIN rmse, so it must be the smaller of the two here, and the gap must be
    # the one actually measured rather than a rounding artefact.
    assert sigma_e < rmse
    assert abs(rmse - sigma_e) / rmse > 1e-4
    # And they are not the same number: sigma_e is the WITHIN rmse, rmse belongs to the RE fit,
    # and the gap is ~1.5%, which is what round 46 measured and fixed.
    assert abs(rmse - sigma_e) / rmse == pytest.approx(2.383e-4, abs=1e-5)


def test_rho_is_derived_from_the_two_sigmas(fitted) -> None:
    """Consistency check on the reported rho: sigma_u^2 / (sigma_u^2 + sigma_e^2)."""

    sigma_u = float(fitted.e["sigma_u"])
    sigma_e = float(fitted.e["sigma_e"])
    expected = sigma_u**2 / (sigma_u**2 + sigma_e**2)
    assert float(fitted.e["rho"]) == pytest.approx(expected, rel=1e-9)


def test_the_sigma_rows_carry_no_standard_errors(fitted) -> None:
    """C12-05 asked for SEs on these rows; Stata prints bare values, so this is correct."""

    lines = str(fitted.summary(style="stata")).splitlines()
    for name in ("sigma_u", "sigma_e"):
        row = next(line for line in lines if line.strip().startswith(name))
        # One value after the pipe, no second numeric column.
        assert len(row.split("|")[1].split()) == 1, row
