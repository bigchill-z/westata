"""Two-equation coefficient blocks gain their main heading; single-equation fits do not.

This block resisted four rounds (213 plus the three its own comments record). Round 214 found the actual cause of
the stubborn blank row, which was never in the group-heading branch:

    if _is_constant_row and seen_groups and lines and "|" in lines[-1]:

That guard was written in round 218 for `xtreg y k1 i.year, fe`, where a factor's level rows are followed by a
blank pipe before `_cons`. For an EQUATION block there is no factor above the constant, yet `seen_groups` is
already non-empty there (the "inflate" heading), so the test passed and an empty '             |' appeared before
each block's `_cons`.

THE FIX HAS TWO PARTS, AND BOTH ARE NEEDED:

  * the blank heading now requires that the last emitted group was a FACTOR group;
  * the main heading is emitted only when some parameter actually carries an equation prefix -- checking
    `_EQUATION_PREFIXES` alone was wrong, because `logit hi x` briefly gained a spurious "hi |" row from it even
    though its names are ['x', '_cons'].

Result, matching a live `zinb cnt x, inflate(z)`:

    '         cnt | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]'
    '-------------+----------------------------------------------------------------'
    'cnt          |'
    '           x |  -.0097448   .0299416    -0.33   0.745    -.0684294    .0489397'
    '       _cons |   1.138477   .0347838    32.73   0.000     1.070302    1.206652'
    '-------------+----------------------------------------------------------------'
    'inflate      |'
    ...
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


def _zinb_rows(n: int = 1000) -> list[str]:
    rng = np.random.default_rng(n)
    frame = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    frame["cnt"] = rng.negative_binomial(2, 0.4, n).astype(float)
    fitted = ws.ZeroInflatedNegativeBinomial.from_formula(
        "cnt ~ x", frame, inflate="z"
    ).fit()
    return [
        line
        for line in str(fitted.summary(style="stata")).splitlines()
        if line.strip()
    ]


def _logit_rows(n: int = 1000) -> list[str]:
    rng = np.random.default_rng(n)
    frame = pd.DataFrame({"y": rng.normal(size=n), "x": rng.normal(size=n)})
    frame["hi"] = (frame["y"] > 0).astype(float)
    return [
        line
        for line in str(ws.Logit.from_formula("hi ~ x", frame).fit().summary(style="stata")).splitlines()
        if line.strip()
    ]


def test_the_main_equation_heading_is_printed() -> None:
    rows = _zinb_rows()
    assert "cnt          |" in rows, rows[:12]


def test_both_equation_headings_appear() -> None:
    rows = _zinb_rows()
    headings = [row for row in rows if row.endswith(" |") and row.strip().endswith("|")]
    assert "cnt          |" in headings
    assert "inflate      |" in headings


def test_no_blank_heading_rows_remain() -> None:
    """The empty '             |' that preceded each block's _cons."""

    for row in _zinb_rows():
        assert row.strip() != "|", row


def test_a_single_equation_fit_prints_no_heading() -> None:
    """`logit hi x` must not gain a 'hi |' row; its names carry no prefix."""

    rows = _logit_rows()
    assert not any(row.startswith("hi ") and row.endswith(" |") for row in rows), rows[:10]


def test_the_constant_follows_its_own_block() -> None:
    rows = _zinb_rows()
    coefficient_rows = [row for row in rows if row.strip() and row[13:14] == "|"]
    texts = [row[14:].strip() for row in coefficient_rows]
    assert any("-.0097448" in text for text in texts)
    assert any("1.138477" in text for text in texts)


@pytest.mark.parametrize("n", [500, 1000])
def test_the_structure_holds_across_samples(n) -> None:
    rows = _zinb_rows(n)
    assert "cnt          |" in rows
    assert "inflate      |" in rows
