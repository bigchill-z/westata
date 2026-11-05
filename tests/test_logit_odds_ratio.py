"""Pin ``logit ..., or``: the exp() transform, the relabelled column and the trailing note.

Reference from a live ``logit treat age education re74 re75, or`` on nsw_dw.dta::

           treat | Odds ratio   Std. err.      z    P>|z|     [95% conf. interval]
    -------------+----------------------------------------------------------------
             age |   1.013706   .0138091     1.00   0.318     .9869991    1.041136
       education |   1.084459   .0595068     1.48   0.140     .9738808    1.207593
            re74 |   .9999804    .000025    -0.79   0.431     .9999314    1.000029
            re75 |   1.000046   .0000413     1.11   0.268     .9999648    1.000127
           _cons |   .2149077   .1418925    -2.33   0.020     .0589179    .7838927
    ------------------------------------------------------------------------------
    Note: _cons estimates baseline odds.

FOUR things change under ``or``, and getting any one wrong leaves a plausible table:

1. the column is relabelled "Odds ratio" (three spaces before "Std. err.", not two);
2. the ESTIMATE is exp(b);
3. the STANDARD ERROR is exp(b) * se(b) -- not exp(se(b)), which is the tempting slip;
4. both interval ENDPOINTS are exponentiated individually, because exp(b +/- z*se) is only
   symmetric when se is small.

z and P>|z| do NOT change: they are computed on the untransformed scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\nsw_dw.dta"
STATA_ROWS = [
    "       treat | Odds ratio   Std. err.      z    P>|z|     [95% conf. interval]",
    "         age |   1.013706   .0138091     1.00   0.318     .9869991    1.041136",
    "   education |   1.084459   .0595068     1.48   0.140     .9738808    1.207593",
    "        re74 |   .9999804    .000025    -0.79   0.431     .9999314    1.000029",
    "        re75 |   1.000046   .0000413     1.11   0.268     .9999648    1.000127",
    "       _cons |   .2149077   .1418925    -2.33   0.020     .0589179    .7838927",
]


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("nsw_dw.dta is not available")
    return ws.Logit.from_formula(
        "treat ~ age + education + re74 + re75", frame
    ).fit()


def test_every_row_matches_stata(fitted) -> None:
    rendered = fitted.summary(style="stata", odds_ratio=True).splitlines()
    for expected in STATA_ROWS:
        assert expected in rendered, expected


def test_the_column_is_relabelled(fitted) -> None:
    rendered = fitted.summary(style="stata", odds_ratio=True)
    assert "Odds ratio" in rendered
    assert "Coefficient" not in rendered


def test_the_plain_summary_is_unaffected(fitted) -> None:
    """The flag must not leak into a second summary of the same fit."""

    plain = fitted.summary(style="stata")
    assert "Coefficient" in plain
    assert "Odds ratio" not in plain
    assert "baseline odds" not in plain
    # And calling it again in odds-ratio mode still works.
    assert "Odds ratio" in fitted.summary(style="stata", odds_ratio=True)


def test_the_standard_error_is_scaled_by_exp_of_b_not_exp_of_se(fitted) -> None:
    """exp(b) * se, not exp(se) -- the values only work out one way."""

    values = np.asarray(fitted.params, dtype=float)
    errors = np.asarray(fitted.bse, dtype=float)
    expected = np.exp(values) * errors
    odds = np.exp(values)
    # age is the second parameter (Intercept first).
    assert float(odds[1]) == pytest.approx(1.013706, abs=5e-7)
    assert float(expected[1]) == pytest.approx(0.0138091, abs=5e-8)


def test_the_trailing_note_appears_only_with_odds_ratio(fitted) -> None:
    assert "Note: _cons estimates baseline odds." in fitted.summary(
        style="stata", odds_ratio=True
    )
    assert "Note: _cons estimates baseline odds." not in fitted.summary(style="stata")


def test_the_header_row_is_still_78_characters(fitted) -> None:
    header = next(
        row
        for row in fitted.summary(style="stata", odds_ratio=True).splitlines()
        if "Odds ratio" in row
    )
    assert len(header) == 78
