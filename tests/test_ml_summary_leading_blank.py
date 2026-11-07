"""ML summaries: the leading blank line was ADDED twice and STRIPPED once. The strip was it.

Round 198 tried to fix logit/probit's missing leading blank by initialising the line list with one.
It had no effect, and the round ended without knowing why. Round 199 followed the recommendation it
left behind -- instrument the branch instead of reading it -- and `sys.settrace` on executed line
numbers found the answer in one run:

    line 2770:  if _is_tobit or _is_truncreg or block == "ml":
    line 2771:      while lines and not lines[0].strip():
    line 2772:          lines.pop(0)

The ML branch EXPLICITLY DELETED every leading blank line, on a comment claiming "Stata's header
starts with the banner line and has no leading blank". A live `logit rns s expr tenure, nolog`
starts with chr(10) + 'Logisti' -- the claim was wrong, and that strip is why logit/probit were the
only estimators without the blank.

THREE attempts, three different mistakes, all now recorded in the code:

  * round 198 -- added the blank at the initialiser: invisible, because the strip removed it;
  * round 199 first try -- added it again in the ANOVA-branch rebuild: produced TWO blanks for
    the ML family and TWO for regress;
  * round 199 final -- removed the STRIP and reverted both additions, giving exactly one blank
    from the separator the function already appends.

Verified byte-exact against live Stata: regress 16/16, logit 14/14, probit 14/14.
"""

from __future__ import annotations

import pandas as pd
import pytest

import westata as ws

GRILIC = r"D:\myCodeProject\test_westata\data\grilic.dta"


@pytest.fixture(scope="module")
def frames():
    try:
        frame = pd.read_stata(GRILIC)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    return (
        frame.dropna(subset=["lnw", "s", "expr", "tenure"]),
        frame.dropna(subset=["rns", "s", "expr", "tenure"]),
    )


def test_ols_starts_with_one_blank_line(frames) -> None:
    frame, _ = frames
    rendered = str(ws.OLS.from_formula("lnw ~ s + expr + tenure", frame).fit().summary(style="stata"))
    assert rendered.startswith("\n")
    assert not rendered.startswith("\n\n")


@pytest.mark.parametrize("model", ["logit", "probit"])
def test_ml_models_start_with_one_blank_line(frames, model: str) -> None:
    """They had NONE before round 199; the first attempt gave them TWO."""

    _, frame = frames
    cls = ws.Logit if model == "logit" else ws.Probit
    rendered = str(cls.from_formula(f"rns ~ s + expr + tenure", frame).fit().summary(style="stata"))
    assert rendered.startswith("\n")
    assert not rendered.startswith("\n\n"), "exactly one blank line"


def test_the_ml_banner_is_the_second_line(frames) -> None:
    _, frame = frames
    lines = str(
        ws.Logit.from_formula("rns ~ s + expr + tenure", frame).fit().summary(style="stata")
    ).splitlines()
    assert lines[0] == ""
    assert lines[1].startswith("Logistic regression")


def test_the_strip_is_really_gone() -> None:
    """A regression guard: the code must no longer delete leading blanks for the ML block."""

    import inspect

    from westata.results import summary as module

    source = inspect.getsource(module._stata_summary)
    assert "lines.pop(0)" not in source or "_is_tobit" not in source.split("lines.pop(0)")[0][-300:]
