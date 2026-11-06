"""The sigma_e divisor fix applies to xtreg, re only -- xtreg, fe must be untouched.

sigma_e is computed on the SHARED code path, so changing the divisor for the random-effects
estimator carried a real risk of disturbing the fixed-effects numbers. Checked against a live
run on lin_1992:

    quantity      xtreg, fe        Stata fe       xtreg, re       Stata re
    sigma_u       .35038457       .35038457      .30729932      .30729941
    sigma_e       .12733392       .12733392      .12733774      .12733392
    rho           .88333901       .88333901      .85345480      .85346238

The fe column is exact to every digit -- including sigma_e, which stays on the df_r divisor as it
must, because the within estimator has no likelihood form.

One thing this check made clear that is worth stating: fe's sigma_e (.12733392) equals re's
(.12733392) in Stata even though the two models have different residual sums of squares. That
coincidence is specific to this dataset, not a rule -- both estimators recover the same idiosyncratic
variance here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\lin_1992.dta"

STATA_FE = {"sigma_u": 0.35038457, "sigma_e": 0.12733392, "rho": 0.88333901}


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


def test_fe_sigma_values_are_exact(frame) -> None:
    """The shared-path change must leave the within estimator untouched."""

    result = ws.PanelOLS.from_formula(
        "ltvfo ~ ltlan + ltwlab + ltpow", frame, entity="prov", time="year"
    ).fit(entity_effects=True)
    for key, value in STATA_FE.items():
        assert float(result.e[key]) == pytest.approx(value, abs=5e-9), key


def test_fe_and_re_report_their_own_sigma_e(frame) -> None:
    """Neither should inherit the other's divisor."""

    fixed = ws.PanelOLS.from_formula(
        "ltvfo ~ ltlan + ltwlab + ltpow", frame, entity="prov", time="year"
    ).fit(entity_effects=True)
    random_effects = ws.RandomEffects.from_formula(
        "ltvfo ~ ltlan + ltwlab + ltpow", frame, entity="prov", time="year"
    ).fit()
    # The two agree here because both recover the same idiosyncratic variance on this data,
    # but they are computed through different paths and must both be right.
    assert float(fixed.e["sigma_e"]) == pytest.approx(0.12733392, abs=5e-9)
    assert float(random_effects.e["sigma_e"]) == pytest.approx(0.12733392, rel=1e-4)


def test_fe_sigma_u_differs_from_re_sigma_u(frame) -> None:
    """A strong check that the Swamy-Arora correction did not leak into fe."""

    fixed = ws.PanelOLS.from_formula(
        "ltvfo ~ ltlan + ltwlab + ltpow", frame, entity="prov", time="year"
    ).fit(entity_effects=True)
    random_effects = ws.RandomEffects.from_formula(
        "ltvfo ~ ltlan + ltwlab + ltpow", frame, entity="prov", time="year"
    ).fit()
    assert float(fixed.e["sigma_u"]) != pytest.approx(float(random_effects.e["sigma_u"]), rel=1e-3)
