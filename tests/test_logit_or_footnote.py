"""The `Note: _cons estimates baseline odds.` footnote appears ONLY with `, or`.

The v3 report's section 5.5 records C11-01 (a weighted `logit ..., or`) as missing its closing
rule. Measured, the rule is present and every digit matches Stata:

    Stata  :       _cons |   1.107758   .0782241     1.45   0.147     .96457
    westata:       _cons |   1.107758   .0782241     1.45   0.147     .96457
    ------------------------------------------------------------------
    Stata  : Note: _cons estimates baseline odds.

THE FOOTNOTE IS CONDITIONAL, which is the detail worth pinning: a live comparison shows

    logit y x1 [fweight=freq]          -> rule, NO note
    logit y x1 [fweight=freq], or      -> rule, then the note

because the note explains what the CONSTANT means on the odds scale. Printing it unconditionally
would add a line to every plain logit; omitting it under `or` would drop Stata's last line. Both
directions are guarded below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(5)
    n = 400
    return pd.DataFrame(
        {
            "y": (rng.uniform(size=n) < 0.5).astype(float),
            "x1": rng.normal(size=n),
            "freq": rng.integers(1, 4, size=n).astype(float),
        }
    )


@pytest.fixture(scope="module")
def weighted(frame):
    return ws.Logit.from_formula("y ~ x1", frame, freq_weights="freq").fit()


def test_the_odds_ratio_variant_closes_with_a_rule(weighted) -> None:
    lines = str(weighted.summary(style="stata", odds_ratio=True)).splitlines()
    assert set(lines[-2].strip()) == {"-"}


def test_the_odds_ratio_variant_keeps_the_note(weighted) -> None:
    lines = str(weighted.summary(style="stata", odds_ratio=True)).splitlines()
    assert lines[-1] == "Note: _cons estimates baseline odds."


def test_the_plain_variant_has_the_rule_but_no_note(weighted) -> None:
    """A plain logit ends on the rule; the note is about the odds scale, not the log-odds one."""

    lines = str(weighted.summary(style="stata")).splitlines()
    assert set(lines[-1].strip()) == {"-"}
    assert not any("Note:" in line for line in lines)


def test_the_odds_ratio_matches_stata(weighted) -> None:
    """From `logit y x1 [fweight=freq], or`: _cons 1.107758, se .0782241, z 1.45, p 0.147."""

    rendered = str(weighted.summary(style="stata", odds_ratio=True))
    assert "1.107758" in rendered
    assert ".0782241" in rendered


def test_the_coefficient_variant_matches_stata(weighted) -> None:
    """The same fit on the log-odds scale: _cons .1023384, se .0706147."""

    rendered = str(weighted.summary(style="stata"))
    assert ".1023384" in rendered
    assert ".0706147" in rendered


def test_the_footnote_is_not_emitted_for_a_plain_logit(frame) -> None:
    """Without weights and without `or`, a logit ends on its rule."""

    fitted = ws.Logit.from_formula("y ~ x1", frame).fit()
    lines = str(fitted.summary(style="stata")).splitlines()
    assert set(lines[-1].strip()) == {"-"}
