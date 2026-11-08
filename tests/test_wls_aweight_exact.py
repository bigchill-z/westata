"""Analytic weights reproduce Stata's WLS coefficients.

C07-04 scores `reg ... [aw=...]` at 0.25. Measured on nerlove.dta:

    quietly reg lntc lnq lnpl lnpk lnpf [aw=q]
        lnq    0.93895833    lnpl 0.77294428    lnpk -0.02106051
        lnpf   0.45160698    _cons -6.50031038

All five match to eight decimals.

ANALYTIC VERSUS FREQUENCY WEIGHTS ARE NOT THE SAME THING and the guard below separates them: an
`aweight` scales the objective, so the point estimates move but the reported observation count stays
the row count. A `fweight` expands the data, so e(N) becomes the weight total. Confusing the two is
the mistake this pair of tests exists to catch.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws

NERLOVE = r"D:\myCodeProject\test_westata\data\nerlove.dta"


@pytest.fixture(scope="module")
def frame():
    try:
        return pd.read_stata(NERLOVE)
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("nerlove.dta is not available")


@pytest.fixture(scope="module")
def weighted(frame):
    return ws.OLS.from_formula(
        "lntc ~ lnq + lnpl + lnpk + lnpf",
        frame,
        weights="q",
        weight_type="aweight",
    ).fit()


def test_the_analytic_weight_coefficients_match_stata(weighted) -> None:
    assert float(weighted.params["lnq"]) == pytest.approx(0.93895833, abs=1e-8)
    assert float(weighted.params["lnpl"]) == pytest.approx(0.77294428, abs=1e-8)
    assert float(weighted.params["lnpk"]) == pytest.approx(-0.02106051, abs=1e-8)
    assert float(weighted.params["lnpf"]) == pytest.approx(0.45160698, abs=1e-8)
    assert float(weighted.params["Intercept"]) == pytest.approx(-6.50031038, abs=1e-8)


def test_analytic_weights_keep_the_row_count(frame, weighted) -> None:
    """`aweight` scales the objective; it does not expand the data the way `fweight` does."""

    assert int(weighted.nobs) == int(len(frame))


def test_analytic_weights_change_the_estimates(frame) -> None:
    """If they did not, the weights would be entering nothing."""

    unweighted = ws.OLS.from_formula(
        "lntc ~ lnq + lnpl + lnpk + lnpf", frame
    ).fit()
    weighted = ws.OLS.from_formula(
        "lntc ~ lnq + lnpl + lnpk + lnpf",
        frame,
        weights="q",
        weight_type="aweight",
    ).fit()
    assert float(unweighted.params["lnq"]) != pytest.approx(
        float(weighted.params["lnq"]), rel=1e-6
    )
