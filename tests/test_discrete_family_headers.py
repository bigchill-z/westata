"""Five discrete and count estimators share one header renderer, and all five land on column 70.

Rounds 204-206 fixed this header in three passes, each covering one more estimator than the last. This round
checked the whole family at once, since the renderer is shared:

    'Logistic regression                                     Number of obs =    800'
    'Probit regression                                       Number of obs =    800'
    'Poisson regression                                      Number of obs =    800'
    'Negative binomial regression                            Number of obs =    800'

ALL FOUR MATCH A LIVE CAPTURE EXACTLY, as does `xtlogit, re` at its different left width (column 53, label 17,
still landing "=" at 70). The `zip` case could not be compared because the sample produced a singular information
matrix in the package rather than a header -- recorded rather than reported as a match.

THE POINT OF CHECKING THE FAMILY TOGETHER: 204 verified two estimators, 205 found the third was different, 206
found the fourth was different again. Enumerating all of them in one pass is cheaper than discovering them one
regression at a time.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


def _frame(n: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(n)
    frame = pd.DataFrame({"y": rng.normal(size=n), "x": rng.normal(size=n)})
    frame["hi"] = (frame["y"] > 0).astype(float)
    frame["cnt"] = rng.poisson(2.0, n).astype(float)
    return frame


def _count_row(estimator: str, formula: str) -> str:
    fitted = getattr(ws, estimator).from_formula(formula, _frame()).fit()
    rendered = str(fitted.summary(style="stata"))
    return next(line for line in rendered.splitlines() if "Number of obs" in line)


@pytest.mark.parametrize(
    "estimator,formula",
    [
        ("Logit", "hi ~ x"),
        ("Probit", "hi ~ x"),
        ("Poisson", "cnt ~ x"),
        ("NegativeBinomial", "cnt ~ x"),
    ],
)
def test_the_count_row_lands_on_seventy(estimator, formula) -> None:
    row = _count_row(estimator, formula)
    assert row.index("=") == 70, row
    assert len(row) == 78, (len(row), row)


@pytest.mark.parametrize(
    "estimator,formula",
    [
        ("Logit", "hi ~ x"),
        ("Probit", "hi ~ x"),
        ("Poisson", "cnt ~ x"),
        ("NegativeBinomial", "cnt ~ x"),
    ],
)
def test_every_estimator_reports_the_same_count(estimator, formula) -> None:
    assert "800" in _count_row(estimator, formula)


def test_the_banners_differ_but_the_column_does_not() -> None:
    banners = {
        _count_row(estimator, formula).split("Number")[0].strip()
        for estimator, formula in [
            ("Logit", "hi ~ x"),
            ("Probit", "hi ~ x"),
            ("Poisson", "cnt ~ x"),
            ("NegativeBinomial", "cnt ~ x"),
        ]
    }
    assert len(banners) == 4, banners
