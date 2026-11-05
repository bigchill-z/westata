"""Pin ``rcm``'s regression-control estimate, and record what it does NOT reproduce.

Reference from a live ``rcm cigsale lnincome beer, trunit(3) trperiod(1989) method(forward)``
on synth_smoking.dta::

    Prediction results in the posttreatment period using OLS:
    -----------------------------------------------------------
     Time | Actual Outcome  Predicted Outcome  Treatment Effect
    ------+----------------------------------------------------
     1989 |       82.4000            94.7563          -12.3563
     1990 |       77.8000           106.4992          -28.6992
     1991 |       68.7000           104.5767          -35.8767
     1992 |       67.5000           104.3214          -36.8214
     1993 |       63.4000           105.4696          -42.0696
     1994 |       58.6000           111.1808          -52.5808
     1995 |       56.4000           106.0483          -49.6483
     1996 |       54.5000           101.5707          -47.0707
     1997 |       53.8000           104.7982          -50.9982
     1998 |       52.3000           105.9868          -53.6868
     1999 |       47.2000           104.8442          -57.6442
     2000 |       41.6000           103.6902          -62.0902
    ------+----------------------------------------------------
       Mean |       60.3500          104.4785          -44.1285
    -----------------------------------------------------------
    Note: The average treatment effect over the posttreatment period is -44.1285

with ``e(att) = -44.12853209``.

WHAT IS EXACT. Stata's forward search selected control units 7, 8 and 13 and fitted

    cigsale(3) = 96.8563 - 0.6569*cigsale(7) + 0.7177*cigsale(8) + 0.1081*cigsale(13)

Passing ``selected=[7, 8, 13]`` reproduces every coefficient, every per-period treatment
effect and the ATT to the printed digits (ATT -44.12853199 against -44.12853209, a 1e-7
floating-point difference).

WHAT IS NOT EXACT, and this is the interesting part. An exhaustive search over the same
candidate pool does NOT select Stata's model, and Stata's model is the WORSE-fitting one:

    units [5, 19, 21]   RSS  20.73   <- what exhaustive search picks
    units [7,  8, 13]   RSS 256.39   <- what Stata picks

So Stata is not finding the RSS-minimal model at any size, which means its ``rcm()`` Mata
routine must be searching a restricted or differently-ordered space -- and that routine is
compiled, with no readable source. The per-size winners were also stable under AICc, AIC and
BIC, so the criterion is not what differs; the search space is.

Accordingly ``select="forward"`` is available but its output is documented as NOT matching
Stata's selection. Only ``selected=`` is pinned by a test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\synth_smoking.dta"
STATA_ATT = -44.12853209
STATA_EFFECTS = [
    -12.3563, -28.6992, -35.8767, -36.8214, -42.0696, -52.5808,
    -49.6483, -47.0707, -50.9982, -53.6868, -57.6442, -62.0902,
]


@pytest.fixture(scope="module")
def frame():
    try:
        return pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("synth_smoking.dta is not available")


@pytest.fixture(scope="module")
def fitted(frame):
    return ws.rcm(
        frame["cigsale"],
        frame[["cigsale"]],
        frame["state"],
        frame["year"],
        treated_unit=3,
        treatment_period=1989,
        selected=[7, 8, 13],
    )


def test_the_att_matches_stata(fitted) -> None:
    assert fitted.att == pytest.approx(STATA_ATT, abs=1e-6)


def test_the_coefficients_match_stata(fitted) -> None:
    """Stata prints them in the order the search added them: 7, 13, 8, _cons."""

    coefficients = np.asarray(fitted.coefficients, dtype=float)
    # Constructed order is [7, 8, 13, _cons].
    assert coefficients[0] == pytest.approx(-0.6569, abs=5e-5)
    assert coefficients[1] == pytest.approx(0.7177, abs=5e-5)
    assert coefficients[2] == pytest.approx(0.1081, abs=5e-5)
    assert coefficients[3] == pytest.approx(96.8563, abs=5e-5)


def test_every_period_effect_matches_stata(fitted) -> None:
    assert len(fitted.effects) == len(STATA_EFFECTS)
    for got, expected in zip(fitted.effects, STATA_EFFECTS):
        assert got == pytest.approx(expected, abs=5e-5)


def test_the_post_period_is_the_twelve_treated_years(fitted) -> None:
    assert int(fitted.post_years[0]) == 1989
    assert int(fitted.post_years[-1]) == 2000
    assert fitted.post_years.size == 12


def test_the_rendered_table_reports_the_att(fitted) -> None:
    rendered = str(fitted)
    assert "-44.1285" in rendered
    assert "1989" in rendered
    assert "2000" in rendered


def test_the_automatic_search_is_documented_as_not_matching_stata(frame) -> None:
    """Exhaustive search picks a DIFFERENT, better-fitting model than Stata.

    This is asserted deliberately: it records that the gap is a search-space difference in
    compiled Mata code, and it will fail loudly if someone later changes the search in a way
    that silently starts matching by accident.
    """

    automatic = ws.rcm(
        frame["cigsale"],
        frame[["cigsale"]],
        frame["state"],
        frame["year"],
        treated_unit=3,
        treatment_period=1989,
    )
    assert automatic.att != pytest.approx(STATA_ATT, abs=1e-3)
    # And the model it finds really does fit better on the pre-period.
    assert automatic.predictors != ("7", "8", "13")
