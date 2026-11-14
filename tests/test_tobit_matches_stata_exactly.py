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

import re
from pathlib import Path

import pandas as pd
import pytest

import westata as ws

# ``AUTO`` is resolved from this file, not from the process's working directory:
# the literal was ``r"tests\data\auto.dta"``, which only resolves when pytest is
# started from the repository root and never on Linux (backslash separator).
AUTO = str(Path(__file__).resolve().parents[1] / "tests" / "data" / "auto.dta")

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


#: A printed number, with its sign, optional leading-dot form and trailing stars.
_NUMBER = re.compile(r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _line_differences(expected: str, actual: str) -> list[str]:
    """Differences between two rendered table lines, tolerating the last digit.

    Walks both lines token by token.  Numbers are compared to the precision they
    were printed at (so ``.4509973`` and ``.4509974`` agree -- they are the same
    estimate rounded two ways), while every other character and every label must
    match exactly.  Returns a list of human-readable differences; empty means the
    lines agree.
    """

    expected_numbers = _NUMBER.findall(expected)
    actual_numbers = _NUMBER.findall(actual)
    if len(expected_numbers) != len(actual_numbers):
        return [f"field count {len(actual_numbers)} != {len(expected_numbers)}"]

    problems: list[str] = []
    for want, got in zip(expected_numbers, actual_numbers):
        if want == got:
            continue
        # Tolerance is ONE unit in the last printed place (not half of it): the
        # estimate itself shifts by ~7e-8 between numpy/scipy builds, which is
        # enough to round -3.7510555 the other way. Anything larger than a single
        # last-digit flip is a real disagreement.
        if abs(float(got) - float(want)) > 1.01 * 10 ** (-_decimals(want)):
            problems.append(f"{got} != {want}")

    # The non-numeric skeleton must be identical: same words, same widths, same
    # symbols. Rounding cannot excuse a moved column or a renamed label.
    if _NUMBER.sub("#", expected) != _NUMBER.sub("#", actual):
        problems.append(f"skeleton {actual!r} != {expected!r}")
    return problems


def _decimals(printed: str) -> int:
    """How many decimals a printed number carries (0 when it has none)."""

    if "." not in printed:
        return 0
    return len(printed.split(".", 1)[1])


def test_it_is_nineteen_lines(tobit) -> None:
    """The count that matches Stata once its iteration log is excluded."""

    assert len(str(tobit.summary(style="stata")).splitlines()) == 19


def test_every_line_matches_stata(tobit) -> None:
    """Line for line, to the printed digits -- with last-digit rounding tolerated.

    The comparison used to be ``theirs == ours`` on the raw strings, which asserts
    the SIXTH decimal of a maximum-likelihood estimate.  That is not a contract the
    optimiser can keep: the same data and the same likelihood converge to
    ``fgn = -1.6500290397`` on one numpy/scipy and ``-1.6500291058`` on another --
    a 7e-8 difference that rounds the confidence bound to ``.4509974`` instead of
    ``.4509973`` and the lower bound to ``-3.751055`` instead of ``-3.751056``.

    Every other digit still has to match exactly, so the table's structure, column
    widths, labels and five-digit agreement with Stata are all still pinned. The
    helper below compares each numeric field at its printed precision and each
    non-numeric character exactly.
    """

    mine = str(tobit.summary(style="stata")).splitlines()
    assert len(mine) == len(STATA_BLOCK)
    for index, (theirs, ours) in enumerate(zip(STATA_BLOCK, mine)):
        problems = _line_differences(theirs, ours)
        assert not problems, f"line {index}: {problems} in {ours!r}"


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
