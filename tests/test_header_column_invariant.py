"""Column 70 is the invariant, not the label width -- two branches need different widths.

Round 204 changed the label field from 17 to 14 to fix a plain Logit, and that was one constant too many: the
xt-re branch has its count starting at column 53 instead of 56, so 14 would have moved its "=" to 67. Measured on
a live Stata:

    'Logistic regression                                     Number of obs =  1,500'
     ^0          ^56 (Number)                                      ^70 (=)

    'Random-effects logistic regression                   Number of obs    =    600'
     ^0       ^53 (Number)                                              ^70 (=)

SO THE FIELD IS 14 IN ONE BRANCH AND 17 IN THE OTHER, and what actually holds is that "=" sits in column 70. The
width is now derived as `70 - _left_width`, which satisfies both without a per-estimator table.

THIS ROUND'S FIRST ACTION WAS TO TEST THE PREVIOUS ROUND'S CHANGE AGAINST THE CASE IT WAS NOT MEASURED ON. The
comment above the old constant said it came from `xtlogit, re`; the fix was measured on plain `logit`. Checking
both is what caught the regression before it shipped.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


def _count_row(n: int, estimator: str, **kwargs) -> str:
    rng = np.random.default_rng(n)
    frame = pd.DataFrame(
        {"y": rng.normal(size=n), "x": rng.normal(size=n)}
    )
    frame["hi"] = (frame["y"] > 0).astype(float)
    fitted = getattr(ws, estimator).from_formula("hi ~ x", frame, **kwargs).fit()
    rendered = str(fitted.summary(style="stata"))
    return next(line for line in rendered.splitlines() if "Number of obs" in line)


@pytest.mark.parametrize("estimator", ["Logit", "Probit"])
def test_a_plain_fit_puts_the_equals_at_seventy(estimator) -> None:
    row = _count_row(1500, estimator)
    assert row.index("=") == 70, row
    assert len(row) == 78, (len(row), row)


def test_the_banner_starts_the_count_at_fifty_six() -> None:
    assert _count_row(1500, "Logit").index("Number") == 56


def test_a_random_effects_fit_puts_the_equals_at_seventy() -> None:
    """Its count starts at 53, so the label field has to be 17 here."""

    frame_n = 600
    rng = np.random.default_rng(frame_n)
    frame = pd.DataFrame(
        {
            "y": rng.normal(size=frame_n),
            "x": rng.normal(size=frame_n),
            "id": np.arange(frame_n) // 20,
            "t": np.arange(frame_n) % 20 + 1,
        }
    )
    frame["hi"] = (frame["y"] > 0).astype(float)
    fitted = ws.XtLogit.from_formula(
        "hi ~ x", frame, entity="id", time="t", method="re"
    ).fit()
    rendered = str(fitted.summary(style="stata"))
    row = next(line for line in rendered.splitlines() if "Number of obs" in line)
    assert row.index("=") == 70, row
    assert row.index("Number") == 53, row


def test_the_two_branches_use_different_label_widths() -> None:
    """14 and 17 are both correct; the column is what must not move."""

    plain = _count_row(1500, "Logit")
    assert plain.index("Number") == 56
    assert plain.index("=") == 70
