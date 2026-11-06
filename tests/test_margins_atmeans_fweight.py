"""P0-3 complete: margins atmeans under fweight matches Stata.

Round 115 fixed the weighted means and the missing "At:" block but left the reported Margin at
.5532798 against Stata's .3027798. Round 116 found the real remaining cause -- and it was not in
margins at all.

THE ACTUAL ROOT CAUSE was how the MODEL was fitted. Two interfaces, each wrong in a different way:

    ws.Logit(y, X, freq_weights=<array>)          params: [-.320269, 1.007081, .099827]
                                                  no Intercept, and N did expand to 2201
    ws.Logit.from_formula(f, d)                   params: [-.12342, 1.109428, -.0, .497515]
                                                  has Intercept, but N stayed 24

Neither matched Stata. The formula interface takes freq_weights as a COLUMN NAME:

    ws.Logit.from_formula(f, d, freq_weights="freq")
        params: [-1.504887, .699583, 2.142484, 1.209812]   N = 2201

which reproduces Stata's fit exactly (Stata prints child .6995831, female 2.142484,
class1 1.209812, _cons -1.504887, N 2201).

With the model right, margins falls into place:

    At: child = .0495229 (mean)      Stata .0495229
        female = .2135393 (mean)            .2135393
        class1 = .1476602 (mean)            .1476602
    Margin .3027798  SE .0110709            .3027798 / .0110709

A NOTE ON WHAT I DID WRONG: I passed an ARRAY to from_formula's freq_weights, which the signature
types as `str | None`, and it was silently ignored -- leaving 24 unweighted rows. The silent
acceptance is worth its own look, but the immediate lesson is that a wrong-typed argument that is
dropped without complaint will look like a numeric bug three functions downstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\titanic.dta"
FORMULA = "survive ~ child + female + class1"

#: From `logit survive child female class1 [fweight=freq]` on titanic.dta.
STATA_PARAMS = {
    "Intercept": -1.504887,
    "child": 0.699583,
    "female": 2.142484,
    "class1": 1.209812,
}
STATA_NOBS = 2201
STATA_ATMEANS = {"child": 0.0495229, "female": 0.2135393, "class1": 0.1476602}
STATA_MARGIN = 0.3027798
STATA_MARGIN_SE = 0.0110709


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("titanic.dta is not available")
    return ws.Logit.from_formula(FORMULA, frame, freq_weights="freq").fit()


def test_the_model_reproduces_statas_fit(fitted) -> None:
    """The root cause of round 115's wrong margin was here, not in margins."""

    assert float(fitted.nobs) == STATA_NOBS
    for name, expected in STATA_PARAMS.items():
        assert float(fitted.params[name]) == pytest.approx(expected, abs=5e-6), name


def test_atmeans_are_the_weighted_means(fitted) -> None:
    margins = fitted.margins(atmeans=True)
    printed = dict(margins.at_means)
    for name, expected in STATA_ATMEANS.items():
        assert printed[name] == pytest.approx(expected, abs=5e-7), name


def test_the_margin_and_its_error_match_stata(fitted) -> None:
    margins = fitted.margins(atmeans=True)
    assert float(margins.estimate) == pytest.approx(STATA_MARGIN, abs=5e-7)
    assert float(margins.std_err) == pytest.approx(STATA_MARGIN_SE, abs=5e-7)


def test_the_at_lines_are_rendered(fitted) -> None:
    """Round 115's second defect: they never printed for a plain atmeans call."""

    rendered = str(fitted.margins(atmeans=True))
    assert "At: child     = .0495229 (mean)" in rendered
    assert "    female    = .2135393 (mean)" in rendered
    assert "    class1    = .1476602 (mean)" in rendered
