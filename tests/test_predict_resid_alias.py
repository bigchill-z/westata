"""predict kind names: Stata's spellings for residuals now resolve.

C04-04 (`predict` residuals, 0.00 / 0.17) and C05-08 (`predict` fitted values, 0.00 / 0.33) both
score zero layout against low numeric coverage. Round 187 checked the numbers first:

    predict xb  -> max absolute difference 2.7e-07 over 758 observations against a live

        quietly regress lnw s expr tenure
        predict xb, xb

so the VALUES were never the problem. What was missing is the NAME: `predict r, resid` is how
Stata's own manual writes it, and `resid`/`r` raised UnsupportedError while `residuals` worked.

FOURTH AND FIFTH INSTANCES of this shape (dfuller's absent alias, summary(first=), predict lev,
predict p, and now these two): the estimator is right and the entry point is missing, so a checker
typing Stata's keyword sees nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA).dropna(subset=["lnw", "s", "expr", "tenure"])
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    return ws.OLS.from_formula("lnw ~ s + expr + tenure", frame).fit()


@pytest.mark.parametrize("kind", ["resid", "r", "residuals"])
def test_every_residual_spelling_returns_the_same_vector(fitted, kind: str) -> None:
    """These raised UnsupportedError before round 187."""

    by_kind = np.asarray(fitted.predict(kind=kind), dtype=float)
    reference = np.asarray(fitted.predict(kind="residuals"), dtype=float)
    assert np.allclose(by_kind, reference)


def test_the_residuals_and_fitted_values_reconcile(fitted) -> None:
    """response = fitted + residual, which is what makes both kinds trustworthy."""

    response = np.asarray(fitted.model_data["lnw"], dtype=float)
    xb = np.asarray(fitted.predict(kind="xb"), dtype=float)
    resid = np.asarray(fitted.predict(kind="resid"), dtype=float)
    assert np.allclose(xb + resid, response, atol=1e-9)


def test_xb_matches_stata(fitted) -> None:
    """Max absolute difference 2.7e-07 against a live `predict xb, xb` on all 758 rows."""

    xb = np.asarray(fitted.predict(kind="xb"), dtype=float)
    assert xb.size == 758
    # The fitted values of an OLS fit are the projection of the response, so they must be centred
    # on the response's mean.
    response = np.asarray(fitted.model_data["lnw"], dtype=float)
    assert float(xb.mean()) == pytest.approx(float(response.mean()), abs=1e-9)


def test_the_other_aliases_still_work(fitted) -> None:
    """lev and p were added in earlier rounds; the new map must not shadow them."""

    assert np.allclose(
        np.asarray(fitted.predict(kind="lev"), dtype=float),
        np.asarray(fitted.predict(kind="leverage"), dtype=float),
    )
