"""xtreg, mle: implemented, and matching Stata to ML convergence precision.

The report lists "xtreg, mle missing" under P1. It is present as ``RandomEffectsMLE``, and against
a live ``xtreg ..., mle`` on lin_1992.dta:

    term       westata      Stata        difference
    sigma_u    .2141106     .2141094     1.2e-06
    sigma_e    .1061020     .1061021     5.1e-08
    rho        .8028468     .8028448     2.0e-06

All three inside ML convergence precision (Stata's own iteration log shows it stopped at
log likelihood 332.89739 on iteration 5).

A TRAP worth recording, because it cost time in this round: comparing against sigma_u = .30729941
-- the value for ``xtreg, re`` (the GLS estimator) -- makes MLE look 30% wrong. Those are two
different estimators on the same data, and ``xtreg, re`` is NOT the MLE fit. Round 55 verified the
GLS decomposition separately; this module pins the MLE one.

Also recorded: ``xtserial`` cannot be implemented or verified here. It is absent from both ado
paths AND the command is unrecognised by this Stata install (r(199)), so there is no reference to
compare against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\lin_1992.dta"
FORMULA = "ltvfo ~ ltlan + ltwlab + ltpow + ltfer + hrs + mci + ngca"

#: From `xtreg <formula>, mle` on lin_1992.dta.
STATA_MLE = {"sigma_u": 0.2141094, "sigma_e": 0.1061021, "rho": 0.8028448}

#: From `xtreg, re` -- a DIFFERENT estimator, not the MLE fit.
STATA_RE_GLS_SIGMA_U = 0.30729941


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("lin_1992.dta is not available")
    frame = frame.loc[:, ~frame.columns.duplicated()]
    columns = [
        "ltvfo", "ltlan", "ltwlab", "ltpow", "ltfer", "hrs", "mci", "ngca",
        "province", "year",
    ]
    frame = frame.dropna(subset=columns).copy()
    frame["pid"] = frame["province"].astype("category").cat.codes + 1
    frame["tid"] = frame["year"].astype(int)
    return ws.RandomEffectsMLE(FORMULA, frame, entity="pid", time="tid").fit()


def test_the_class_exists() -> None:
    """The report calls it missing; it is a top-level export."""

    assert hasattr(ws, "RandomEffectsMLE")


@pytest.mark.parametrize("term", sorted(STATA_MLE))
def test_the_variance_components_match_stata(fitted, term: str) -> None:
    mine = float(fitted.e[term])
    assert mine == pytest.approx(STATA_MLE[term], abs=5e-6), term


def test_mle_is_not_the_gls_estimator(fitted) -> None:
    """Guarding the trap: .30729941 is `xtreg, re`, and MLE is not that number."""

    sigma_u = float(fitted.e["sigma_u"])
    assert abs(sigma_u - STATA_RE_GLS_SIGMA_U) > 0.05


def test_rho_is_derived_from_the_two_sigmas(fitted) -> None:
    sigma_u = float(fitted.e["sigma_u"])
    sigma_e = float(fitted.e["sigma_e"])
    expected = sigma_u**2 / (sigma_u**2 + sigma_e**2)
    assert float(fitted.e["rho"]) == pytest.approx(expected, rel=1e-6)
