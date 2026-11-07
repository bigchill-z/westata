"""`prais ..., corc`: the standard errors were computed with the WRONG transform.

N14 of the v3 report: coefficients and rho matched Stata exactly while the standard errors were
off by up to 4.7%, which makes every z, p and confidence interval unusable.

THE CAUSE was one line. The final residual variance -- and therefore the whole covariance matrix --
was computed from the PRAIS-WINSTEN transform for both variants:

    transformed_x = [_transform(x[:, j], rho) ...]      # always Prais-Winsten
    transformed_y = _transform(y, rho)

But `corc` uses a different transform: it DROPS the transformed first observation instead of scaling
it by sqrt(1-rho^2). That changes both the first row of the design AND the number of rows, so:

  * the transform is now chosen to match the estimator;
  * the residual degrees of freedom are one lower for corc (N-1 rows), which is what fixes the
    remaining scale.

BEFORE AND AFTER on `icecream`, `consumption ~ temp + price + income, corc`:

    quantity        westata (before)   westata (now)      Stata
    rho             .4009256           .4009256           .4009256
    se(temp)        .0005744           .0005547           .0005547
    se(price)       .8491466           .8108501           .8108501
    se(income)      .0015479           .0015461           .0015461

The default (Prais-Winsten) variant is unchanged: rho .8002264 and b(price) -1.0488545 still match
the values recorded in prais.py's own earlier measurements.
"""

from __future__ import annotations

import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\icecream.dta"

#: From `prais consumption temp price income, corc` -- all four digits groups verified.
STATA = {
    "rho": 0.4009256,
    "temp": (0.0035584, 0.0005547),
    "price": (-0.8923963, 0.8108501),
    "income": (0.0032027, 0.0015461),
}


@pytest.fixture(scope="module")
def frame():
    try:
        return pd.read_stata(DATA)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("icecream.dta is not available")


@pytest.fixture(scope="module")
def corc_fit(frame):
    return ws.prais(frame, "consumption ~ temp + price + income", corc=True)


def test_the_coefficients_match_stata(corc_fit) -> None:
    for name, entry in STATA.items():
        if name == "rho":
            continue
        expected = entry[0]
        assert float(corc_fit.params[name]) == pytest.approx(expected, abs=5e-7), name


def test_the_standard_errors_match_stata(corc_fit) -> None:
    """These were the failure: up to 4.7% high before the transform was matched."""

    for name, entry in STATA.items():
        if name == "rho":
            continue
        expected = entry[1]
        assert float(corc_fit.bse[name]) == pytest.approx(expected, abs=5e-7), name


def test_rho_matches_stata(corc_fit) -> None:
    assert float(corc_fit.rho) == pytest.approx(STATA["rho"], abs=5e-8)


def test_the_residual_degrees_of_freedom_drop_by_one(corc_fit, frame) -> None:
    """corc fits on N-1 rows; using N left every standard error scaled by sqrt(26/25).

    Four parameters (three regressors plus the constant), so Prais-Winsten has N-4 = 26 and corc
    has N-1-4 = 25.
    """

    assert int(corc_fit.df_resid) == int(frame.shape[0]) - 1 - 4


def test_the_default_variant_is_unaffected(frame) -> None:
    """Prais-Winsten keeps all N rows and its own established numbers."""

    fitted = ws.prais(frame, "consumption ~ temp + price + income")
    assert float(fitted.rho) == pytest.approx(0.8002264, abs=1e-7)
    assert float(fitted.params["price"]) == pytest.approx(-1.0488545, abs=5e-7)
    assert int(fitted.df_resid) == int(frame.shape[0]) - 4


def test_the_two_variants_still_differ(frame) -> None:
    """Matching both to Stata must not have collapsed them into one estimator."""

    plain = ws.prais(frame, "consumption ~ temp + price + income")
    corc = ws.prais(frame, "consumption ~ temp + price + income", corc=True)
    assert float(plain.rho) != pytest.approx(float(corc.rho))
