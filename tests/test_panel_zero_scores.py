"""Two panel cases scoring 0.00 and 0.00: both are correct, the comparison is not.

C12-02 (xtreg, fe vce(robust)) and C12-09 (xtreg ..., fe i.year) both score 0.00 in the report.
Neither is a defect.

C12-02: identical coefficients AND identical standard errors, offset by the usual ordering.

    Stata e(b)   .18637289  .26598578  .40125080  4.63475664
    Stata se     .11272805  .04495205  .01652308   .52524278
    westata      same four numbers; intercept first, _cons last
    westata bse  same four numbers in the same reordering

C12-09: identical slopes, but a different PARAMETERISATION. Stata prints a dummy per year (21
parameters); westata absorbs them with time_effects=True (4 parameters, plus an intercept). The
three slopes agree to every digit:

    ltlan .39300903   ltwlab .27005734   ltpow .19720692

A line-wise comparison of a 21-row and a 4-row table cannot score above the overlap, so 0.00 here
measures the parameterisation difference, not accuracy. Whether to print the absorbed dummies is
a design question, not a bug -- recorded as such rather than "fixed" by forcing westata to emit
dropped coefficients.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\lin_1992.dta"


@pytest.fixture(scope="module")
def frame():
    try:
        data = pd.read_stata(DATA).dropna()
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("lin_1992.dta is not available")
    seen: dict[str, int] = {}
    renamed = []
    for column in data.columns:
        seen[column] = seen.get(column, 0) + 1
        renamed.append(column if seen[column] == 1 else f"{column}_{seen[column]}")
    data.columns = renamed
    data["prov"] = data["province"].astype("category").cat.codes + 1
    return data


STATA_FE = {"ltlan": 0.18637289, "ltwlab": 0.26598578, "ltpow": 0.40125080, "_cons": 4.63475664}
STATA_FE_SE = {"ltlan": 0.11272805, "ltwlab": 0.04495205, "ltpow": 0.01652308, "_cons": 0.52524278}
STATA_TWOWAY = {"ltlan": 0.39300903, "ltwlab": 0.27005734, "ltpow": 0.19720692}


def test_fe_robust_coefficients_match(frame) -> None:
    result = ws.PanelOLS.from_formula(
        "ltvfo ~ ltlan + ltwlab + ltpow", frame, entity="prov", time="year"
    ).fit(cov_type="cluster", cov_kwds={"groups": frame["prov"]})
    for name, value in STATA_FE.items():
        key = "Intercept" if name == "_cons" else name
        assert float(result.params[key]) == pytest.approx(value, abs=5e-9), name


def test_fe_robust_standard_errors_match(frame) -> None:
    """The part a 0.00 score would lead you to believe is wrong."""

    result = ws.PanelOLS.from_formula(
        "ltvfo ~ ltlan + ltwlab + ltpow", frame, entity="prov", time="year"
    ).fit(cov_type="cluster", cov_kwds={"groups": frame["prov"]})
    for name, value in STATA_FE_SE.items():
        key = "Intercept" if name == "_cons" else name
        assert float(result.bse[key]) == pytest.approx(value, abs=5e-9), name


def test_two_way_slopes_match(frame) -> None:
    result = ws.PanelOLS.from_formula(
        "ltvfo ~ ltlan + ltwlab + ltpow", frame, entity="prov", time="year"
    ).fit(entity_effects=True, time_effects=True)
    for name, value in STATA_TWOWAY.items():
        assert float(result.params[name]) == pytest.approx(value, abs=5e-9), name
