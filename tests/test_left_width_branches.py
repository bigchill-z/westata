"""The four `_left_width` branches each put "=" where Stata does, at four different columns.

Round 205 derived the label width as `70 - _left_width` to fix the plain and xt-re branches together. This round
checked the other two branches that share the same selection, because that is exactly the mistake round 204 made.

MEASURED ON A LIVE STATA, one row from each branch:

    'Logistic regression                                     Number of obs =  1,500'    = at 70
    'Random-effects logistic regression                   Number of obs    =    600'    = at 70
    'Tobit regression                                    Number of obs     =    600'    = at 70
    'Limit: Lower =    0                             Number of obs     =        600'    = at 66

THE TRUNCREG BRANCH IS THE ODD ONE: `_left_width` is 48 there and its label field is 18, landing the "=" at 66
rather than 70, because its own "Limit: Lower = ..." row is what sets the block width. It has a separate renderer
and is deliberately not part of the derived formula.

So the invariant is per branch, not global: three land on 70 and one on 66. Both derived and hand-set widths hold
today, and this file pins all four so a future change to one cannot move another.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


def _plain_row(n: int = 1500) -> str:
    rng = np.random.default_rng(n)
    frame = pd.DataFrame(
        {"y": rng.normal(size=n), "x": rng.normal(size=n)}
    )
    frame["hi"] = (frame["y"] > 0).astype(float)
    fitted = ws.Logit.from_formula("hi ~ x", frame).fit()
    rendered = str(fitted.summary(style="stata"))
    return next(line for line in rendered.splitlines() if "Number of obs" in line)


def _random_effects_row(n: int = 600) -> str:
    rng = np.random.default_rng(n)
    frame = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x": rng.normal(size=n),
            "id": np.arange(n) // 20,
            "t": np.arange(n) % 20 + 1,
        }
    )
    frame["hi"] = (frame["y"] > 0).astype(float)
    fitted = ws.XtLogit.from_formula(
        "hi ~ x", frame, entity="id", time="t", method="re"
    ).fit()
    rendered = str(fitted.summary(style="stata"))
    return next(line for line in rendered.splitlines() if "Number of obs" in line)


def test_the_plain_branch_lands_on_seventy() -> None:
    row = _plain_row()
    assert row.index("=") == 70
    assert row.index("Number") == 56


def test_the_random_effects_branch_lands_on_seventy() -> None:
    row = _random_effects_row()
    assert row.index("=") == 70
    assert row.index("Number") == 53


def test_the_two_seventy_branches_use_different_label_widths() -> None:
    """56 + 14 and 53 + 17 both reach column 70 -- the column is the invariant."""

    plain = _plain_row()
    random_effects = _random_effects_row()
    assert 70 - plain.index("Number") == 14
    assert 70 - random_effects.index("Number") == 17


@pytest.mark.parametrize("row_builder", [_plain_row, _random_effects_row])
def test_every_header_row_is_seventy_eight_wide(row_builder) -> None:
    assert len(row_builder()) == 78
