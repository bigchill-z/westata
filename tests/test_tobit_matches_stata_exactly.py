"""`tobit` matches Stata line for line -- once the iteration log is excluded.

This block was MISDIAGNOSED in round 14 as having its table in the wrong place: the comparison ran
against Stata's full output, which for tobit includes SEVEN lines of iteration logging before the
header --

    Refining starting values:
    (blank)
    Grid node 0:  Log likelihood = -194.18306
    (blank)
    Fitting full model:
    (blank)

-- so Stata showed 26 lines against this package's 19 and every row after the header appeared
shifted. With `nolog`, both are 19 lines and IDENTICAL:

    tobit                Stata 19   westata 19   match 1.0000

The lesson is recorded because it cost a round: always strip iteration logging before comparing
line counts, or the log lines read as structural differences and send you to fix a table that is
already right.

The coefficients, the censoring counts, the limits block and `var(e.mpg)` all match to the printed
digit. No code change was needed here -- only the measurement was wrong.
"""

from __future__ import annotations

import pandas as pd
import pytest

import westata as ws

AUTO = r"tests\data\auto.dta"

#: Captured from `tobit mpg weight fgn, ll(0) nolog` -- 19 lines, no iteration log.
STATA_BLOCK = [
    '',
    'Tobit regression                                    Number of obs     =     74',
    '                                                           Uncensored =     74',
    'Limits: Lower =    0                                    Left-censored =      0',
    '        Upper = +inf                                   Right-censored =      0',
    '',
    '                                                    LR chi2(2)        =  80.42',
    '                                                    Prob > chi2       = 0.0000',
    'Log likelihood = -194.18306                         Pseudo R2         = 0.1716',
    '',
    '------------------------------------------------------------------------------',
    '         mpg | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]',
    '-------------+----------------------------------------------------------------',
    '      weight |  -.0065879   .0006241   -10.56   0.000    -.0078319   -.0053438',
    '         fgn |  -1.650029   1.053958    -1.57   0.122    -3.751056    .4509973',
    '       _cons |    41.6797   2.121197    19.65   0.000     37.45117    45.90823',
    '-------------+----------------------------------------------------------------',
    '   var(e.mpg)|   11.13746   1.830987                      8.025231    15.45662',
    '------------------------------------------------------------------------------',
]


@pytest.fixture(scope="module")
def frame():
    try:
        base = pd.read_stata(AUTO).dropna(subset=["mpg", "weight"]).copy()
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("auto.dta is not available")
    base["fgn"] = (
        base["foreign"]
        .astype(str)
        .str.lower()
        .str.startswith(("foreign", "1"))
        .astype(float)
    )
    return base


@pytest.fixture(scope="module")
def tobit(frame):
    return ws.Tobit.from_formula("mpg ~ weight + fgn", frame, lower=0.0).fit()


def test_it_is_nineteen_lines(tobit) -> None:
    """The count that matches Stata once its iteration log is excluded."""

    assert len(str(tobit.summary(style="stata")).splitlines()) == 19


def test_every_line_matches_stata(tobit) -> None:
    mine = str(tobit.summary(style="stata")).splitlines()
    assert len(mine) == len(STATA_BLOCK)
    for index, (theirs, ours) in enumerate(zip(STATA_BLOCK, mine)):
        assert theirs == ours, f"line {index}: {theirs!r} != {ours!r}"


def test_no_iteration_log_leaks_into_the_summary(tobit) -> None:
    """The seven log lines are the backend's business, not part of the table."""

    rendered = str(tobit.summary(style="stata"))
    for marker in ("Refining starting values", "Grid node", "Fitting full model"):
        assert marker not in rendered


def test_the_censoring_counts_are_present(tobit) -> None:
    rendered = str(tobit.summary(style="stata"))
    assert "Uncensored" in rendered
    assert "Left-censored" in rendered
    assert "Right-censored" in rendered


def test_the_limits_block_is_present(tobit) -> None:
    rendered = str(tobit.summary(style="stata"))
    assert "Limits: Lower" in rendered
    assert "Upper = +inf" in rendered
