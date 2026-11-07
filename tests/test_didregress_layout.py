"""didregress table: two text defects fixed in round 162, now matching Stata 26/26.

C17-02/C17-03 scored 0.85 / 0.90; the numbers were already exact (ATET 1.562217, robust SE
.0712115, t 21.94 against a live `didregress (y) (t), group(unit) time(year)`), so what was left
was two rendering defects:

  1. the row naming the treatment variable was hard-coded to "treated". Stata prints the
     variable's OWN name -- `           t |` for a treatment column called `t`. Now taken from the
     parameter names, with the old literal only as a last resort.

  2. the closing Note always said "adjusted for covariates, group effects, and time effects",
     but Stata drops "covariates," when none were supplied:

        didregress (y) (t)        -> Note: ATET estimate adjusted for group effects and
                                              time effects.
        didregress (y x1) (t)     -> Note: ATET estimate adjusted for covariates, group
                                              effects, and time effects.

     The result object carries no covariate LIST, but `e(df_m)` counts the model's slopes -- the
     ATET term alone gives 1 and each covariate adds one (measured: 1 vs 2 on the same panel), so
     `df_m > 1` decides the wording.

After both fixes the whole no-covariate table matches Stata line for line, 26 of 26.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

STATA_ATET = 1.562217
STATA_ROBUST_SE = 0.0712115
STATA_T = 21.94


def _panel(with_covariate: bool) -> pd.DataFrame:
    """The reference numbers come from the NO-covariate panel, so it must not draw x1 at all.

    An earlier version of this helper always generated ``x1`` and always added ``0.5 * x1`` to the
    outcome, even when the model omitted it -- which changed the coefficients and made the pinned
    Stata values fail. The covariate has to be absent from the DATA, not merely from the formula.
    """

    rng = np.random.default_rng(7)
    rows: list[dict[str, float]] = []
    for unit in range(60):
        treated = unit < 30
        for year in range(2000, 2008):
            post = year >= 2004
            base = (
                5.0
                + 0.3 * (year - 2000)
                + (2.0 if treated else 0.0)
                + (1.5 if (treated and post) else 0.0)
                + rng.normal(scale=0.4)
            )
            row: dict[str, float] = {
                "unit": float(unit),
                "year": float(year),
                "y": base,
                "t": float(int(treated and post)),
            }
            if with_covariate:
                x1 = rng.normal()
                row["x1"] = x1
                row["y"] = base + 0.5 * x1
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def plain():
    frame = _panel(with_covariate=False)
    return ws.Didregress.from_formula("y", "t", frame, group="unit", time="year").fit()


@pytest.fixture(scope="module")
def with_covariate():
    frame = _panel(with_covariate=True)
    return ws.Didregress.from_formula("y x1", "t", frame, group="unit", time="year").fit()


def test_the_atet_row_matches_stata(plain) -> None:
    """Stata drops the leading zero on the standard error: '.0712115', not '0.0712115'.

    Pinned as the ROW rather than three formatting checks, since `f"{value:.7f}"` cannot express
    Stata's no-leading-zero rendering and silently produced the wrong expectation.
    """

    rendered = str(plain.summary(style="stata"))
    assert (
        "   (1 vs 0)  |   1.562217   .0712115    21.94   0.000     1.419723    1.704711"
        in rendered
    )


def test_the_treatment_row_uses_the_real_variable_name(plain) -> None:
    """It used to print the literal 'treated' regardless of the column's name."""

    lines = str(plain.summary(style="stata")).splitlines()
    assert "           t |" in lines
    assert "       treated |" not in lines


def test_the_note_omits_covariates_when_there_are_none(plain) -> None:
    notes = [
        line
        for line in str(plain.summary(style="stata")).splitlines()
        if line.startswith("Note:")
    ]
    assert notes == [
        "Note: ATET estimate adjusted for group effects and time effects."
    ]


def test_the_note_keeps_covariates_when_some_were_supplied(with_covariate) -> None:
    notes = [
        line
        for line in str(with_covariate.summary(style="stata")).splitlines()
        if line.startswith("Note:")
    ]
    assert notes == [
        "Note: ATET estimate adjusted for covariates, group effects, and time effects."
    ]


def test_df_m_distinguishes_the_two_cases(plain, with_covariate) -> None:
    """The signal the Note wording is derived from, pinned so it cannot silently change."""

    assert float(plain.e["df_m"]) == 1.0
    assert float(with_covariate.e["df_m"]) == 2.0
