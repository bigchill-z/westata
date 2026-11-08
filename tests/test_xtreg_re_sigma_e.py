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


def test_sigma_u_is_close_but_not_exact(fitted) -> None:
    """0.03% off: the panel means are right, the df adjustment differs slightly."""

    got = float(fitted.e["sigma_u"])
    assert got == pytest.approx(STATA_SIGMA_U, rel=1e-2)
    # Pinned as NOT equal, so a future fix shows up here.
    assert got != pytest.approx(STATA_SIGMA_U, rel=1e-6)


def test_sigma_e_is_the_known_disagreement(fitted) -> None:
    """1.1% low. When this starts failing, sigma_e was fixed -- replace with equality."""

    got = float(fitted.e["sigma_e"])
    assert got == pytest.approx(9463.8949, abs=1e-3)
    assert got < STATA_SIGMA_E


def test_the_rss_is_the_rmse_based_one(fitted) -> None:
    """The package's rss equals rmse^2 * (N - k), which is why sigma_e comes out low.

    Stata's sigma_e^2 * N is a DIFFERENT quantity (74698217705.73 against 73085290324.19), so the
    two statistics are not built from the same residual.
    """

    nobs = int(fitted.nobs)
    rmse = float(fitted.e["rmse"])
    rss = float(fitted.e["rss"])

    assert rmse == pytest.approx(STATA_RMSE, abs=1e-6)
    # rss == rmse^2 * (N - k); solve for the implied k and check it is the parameter count.
    implied_k = nobs - rss / rmse**2
    assert implied_k == pytest.approx(18.0, abs=0.5)

    # And Stata's sigma_e does NOT come from that rss.
    assert STATA_SIGMA_E**2 * nobs != pytest.approx(rss, rel=1e-5)


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
