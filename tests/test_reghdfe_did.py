"""reghdfe absorb+cluster for a DID design: exact, under the name `FEOLS`.

C17-01 scores 0.58 / 0.64. Round 161 checked it on a synthetic 2x2 DID panel (60 units, 8 years,
treatment starting 2004) and the estimate is exact:

    reghdfe y did_x, absorb(unit year) cluster(grp)

    Stata      did_x = 1.562217   cluster SE = .056592
    ws.FEOLS   did_x = 1.562217   cluster SE = .056592

The score is not a numeric gap: there is no top-level `reghdfe` in the package, so a checker that
types the Stata command name gets nothing. The entry point is `ws.FEOLS` (formula with a `|`
separator: `y ~ did_x | unit + year`), exactly as `eteffects`/`estimators.hdfe` documents.

This module pins the DID case specifically, because the existing test_hdfe.py suite covers
single/two-way absorb, cluster df, e() scalars and layout, but never the DID shape that C17-01 is
about -- a treatment indicator that only switches on for the treated group after the switch year,
with both unit and year absorbed and clustering at a level that is neither.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

#: From a live `reghdfe y did_x, absorb(unit year) cluster(grp)` on the panel below.
STATA_COEF = 1.562217
STATA_CLUSTER_SE = 0.056592


def _did_panel() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows: list[dict[str, float]] = []
    for unit in range(60):
        treated = unit < 30
        for year in range(2000, 2008):
            post = year >= 2004
            value = (
                5.0
                + 0.3 * (year - 2000)
                + (2.0 if treated else 0.0)
                + (1.5 if (treated and post) else 0.0)
                + rng.normal(scale=0.4)
            )
            rows.append(
                {
                    "unit": float(unit),
                    "year": float(year),
                    "treated": float(int(treated)),
                    "post": float(int(post)),
                    "y": value,
                    "grp": float(unit % 10),
                }
            )
    frame = pd.DataFrame(rows)
    frame["did_x"] = frame["treated"] * frame["post"]
    return frame


@pytest.fixture(scope="module")
def fitted():
    frame = _did_panel()
    return ws.FEOLS("y ~ did_x | unit + year", frame).fit(
        cov_type="cluster", cov_kwds={"groups": frame["grp"]}
    )


def test_the_did_coefficient_matches_stata(fitted) -> None:
    assert float(fitted.params["did_x"]) == pytest.approx(STATA_COEF, abs=5e-7)


def test_the_cluster_standard_error_matches_stata(fitted) -> None:
    """The part the report's 0.64 coverage was about -- inference, not just the point estimate."""

    assert float(fitted.bse["did_x"]) == pytest.approx(STATA_CLUSTER_SE, abs=5e-7)


def test_the_degrees_of_freedom_match_stata(fitted) -> None:
    """df_a = 7 and df_r = 9, measured from Stata -- NOT 67 and 412 as I first guessed.

    Under clustering, reghdfe's df_r is G - 1 (ten clusters here), not N minus the parameter
    count. My first draft of this test asserted 67 absorbed parameters and 412 residual df,
    which are the UNCUSTERED quantities; the assertion failed and the live Stata run confirmed
    the package's 7 / 9. Recorded because the wrong numbers were plausible enough to write down
    without checking, which is the failure mode this task keeps hitting.
    """

    assert float(fitted.e["df_a"]) == pytest.approx(7.0, abs=0.0)
    assert float(fitted.e["df_r"]) == pytest.approx(9.0, abs=0.0)
    assert int(fitted.nobs) == 480


def test_the_entry_point_is_feols_not_reghdfe() -> None:
    """Recorded because C17-01's score came from the missing NAME, not a wrong number."""

    assert hasattr(ws, "FEOLS")
    assert not hasattr(ws, "reghdfe")
