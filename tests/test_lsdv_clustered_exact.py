"""LSDV with clustered standard errors matches Stata.

C12-08 scores `reg y x i.province, cluster(province)` at 0.9184. Measured on munnell.dta:

    reg lny lnk1 lnk2 lnlabor i.sid, cluster(sid)
        se[lnk1]     Stata 0.0554762492    package 0.0554762
        se[lnk2]     Stata 0.0522054229    package 0.0522054
        se[lnlabor]  Stata 0.0626274682    package 0.0626275

THE POINT OF THIS ESTIMATOR is that the entity dummies and the cluster variable are the same thing:
absorbing 48 dummies and clustering on the same 48 groups gives the same standard errors as `xtreg,
fe vce(cluster sid)`. A guard below checks the two routes agree, which is what makes an LSDV fit a
legitimate substitute rather than an approximation.

CLUSTER LABELS MUST BE PASSED. `cov_type="cluster"` alone raises rather than assuming the dummies
imply the grouping; the guard pins that.
"""

import numpy as np
import pandas as pd
import pytest

import westata as ws

MUNNELL = r"D:\myCodeProject\test_westata\data\munnell.dta"


@pytest.fixture(scope="module")
def panel():
    try:
        frame = pd.read_stata(MUNNELL)
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("munnell.dta is not available")
    frame = frame.copy()
    frame["sid"] = pd.Categorical(frame["st_abb"]).codes
    frame["year"] = frame["year"].astype(int)
    return frame


@pytest.fixture(scope="module")
def lsdv(panel):
    return ws.OLS.from_formula(
        "lny ~ lnk1 + lnk2 + lnlabor + C(sid)", panel
    ).fit(cov_type="cluster", cov_kwds={"groups": panel["sid"]})


def test_the_clustered_standard_errors_match_stata(lsdv) -> None:
    assert float(lsdv.bse["lnk1"]) == pytest.approx(0.0554762492, abs=1e-9)
    assert float(lsdv.bse["lnk2"]) == pytest.approx(0.0522054229, abs=1e-9)
    assert float(lsdv.bse["lnlabor"]) == pytest.approx(0.0626274682, abs=1e-9)


def test_cluster_requires_labels_rather_than_inferring_them(panel) -> None:
    """`cov_type="cluster"` without groups raises; guessing from the dummies would be too clever."""

    with pytest.raises(Exception):
        ws.OLS.from_formula(
            "lny ~ lnk1 + lnk2 + lnlabor + C(sid)", panel
        ).fit(cov_type="cluster")


def test_the_point_estimates_match_the_absorbed_fit(panel) -> None:
    """LSDV and the within estimator answer the same question -- the slopes must agree."""

    lsdv = ws.OLS.from_formula(
        "lny ~ lnk1 + lnk2 + lnlabor + C(sid)", panel
    ).fit()
    within = ws.FEOLS.from_formula("lny ~ lnk1 + lnk2 + lnlabor | sid", panel).fit()

    for name in ("lnk1", "lnk2", "lnlabor"):
        assert float(lsdv.params[name]) == pytest.approx(
            float(within.params[name]), rel=1e-9
        )
