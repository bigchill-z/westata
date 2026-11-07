"""Panel fixed effects with clustered SEs: exact, under the name `PanelOLS`.

C12-02 (`xtreg, fe r`, scored 0.00 / 0.00) and C12-09 (`xtreg ..., fe i.year`, 0.00 / 0.01) are
the two zero-rated rows of chapter 12. Neither is a numerical gap: there is no top-level `xtreg`
in the package, so anything that types the Stata command name finds nothing, while the estimator
itself is present as `PanelOLS` (entity/time effects, plus the cluster covariance).

Measured on munnell.dta (48 states x 17 years, 816 observations) against a live

    encode st_abb, gen(sid)
    xtset sid year
    xtreg lny lnk1 lnk2 lnlabor, fe vce(cluster sid)

    Stata       lnk1 -.0880288 (.0538468)   lnk2 .2324367 (.050672)   lnlabor .8577679 (.060788)
    PanelOLS    lnk1 -.088029  (.053847)    lnk2 .232437  (.050672)   lnlabor .857768  (.060788)
    _cons       2.920561 (.2507401) in both

Both the within coefficients and the entity-clustered standard errors agree.

TWO-SIDED FIXED EFFECTS are the same call with `time_effects=True`, which is what
`xtreg ..., fe i.year` estimates; the guard below pins the *catalogue* consequence -- that the
entry point exists and that the Stata spelling does not -- rather than inventing a reference value
for the two-way case.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\munnell.dta"

#: From `xtreg lny lnk1 lnk2 lnlabor, fe vce(cluster sid)`.
STATA_COEF = {"lnk1": -0.0880288, "lnk2": 0.2324367, "lnlabor": 0.8577679, "_cons": 2.920561}
STATA_CLUSTER_SE = {"lnk1": 0.0538468, "lnk2": 0.050672, "lnlabor": 0.060788, "_cons": 0.2507401}


@pytest.fixture(scope="module")
def panel():
    try:
        return pd.read_stata(DATA)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("munnell.dta is not available")


@pytest.fixture(scope="module")
def fitted(panel):
    model = ws.PanelOLS.from_formula(
        "lny ~ lnk1 + lnk2 + lnlabor", panel, entity="st_abb", time="year"
    )
    return model.fit(
        cov_type="clustered", cov_kwds={"groups": panel["st_abb"].to_numpy()}
    )


def test_the_within_coefficients_match_xtreg_fe(fitted) -> None:
    params = dict(fitted.params)
    for name, expected in STATA_COEF.items():
        key = "Intercept" if name == "_cons" else name
        assert float(params[key]) == pytest.approx(expected, abs=5e-7), name


def test_the_entity_clustered_standard_errors_match(fitted) -> None:
    """The `r` in C12-02: clustering on the entity, not a plain OLS covariance."""

    errors = dict(fitted.bse)
    for name, expected in STATA_CLUSTER_SE.items():
        key = "Intercept" if name == "_cons" else name
        assert float(errors[key]) == pytest.approx(expected, abs=5e-7), name


def test_the_entry_point_is_panelols_not_xtreg() -> None:
    """Recorded because the zero scores came from the missing NAME, not a wrong value."""

    assert hasattr(ws, "PanelOLS")
    assert not hasattr(ws, "xtreg")


def test_time_effects_are_available_for_the_two_way_case(panel) -> None:
    """`xtreg ..., fe i.year` is PanelOLS with time_effects=True; the call must be accepted."""

    model = ws.PanelOLS.from_formula(
        "lny ~ lnk1 + lnk2 + lnlabor", panel, entity="st_abb", time="year"
    )
    two_way = model.fit(time_effects=True)
    assert two_way is not None
    # Adding year effects changes the within estimate, so this is a genuinely different fit.
    entity_only = model.fit()
    assert float(two_way.params["lnk1"]) != pytest.approx(float(entity_only.params["lnk1"]))
