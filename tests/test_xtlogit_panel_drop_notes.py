"""`xtlogit, fe` prints the two panel notes Stata puts above everything.

`e(N_drop)` and `e(N_group_drop)` were always recorded, and nothing printed them: a 480-row panel
reported `Number of obs = 472` with no explanation of where eight rows went. Stata says so first, before
the banner and before the iteration log:

    note: multiple positive outcomes within groups encountered.
    note: 1 group (8 obs) omitted because of all positive or
          all negative outcomes.
    (blank)
    Conditional fixed-effects logistic regression        Number of obs    =    472

The second note wraps with a SIX-space indent under the first. The plural changes with the count.

The notes cannot ride along with the design block: for the non-linear xt family that block renders
BELOW the banner, whereas these come first of all, so they are prepended to the finished table.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


def _panel(seed: int = 11, groups: int = 60, periods: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "pid": np.repeat(np.arange(groups), periods),
            "year": np.tile(np.arange(1, periods + 1), groups),
        }
    )
    frame["x"] = rng.normal(size=len(frame))
    frame["z"] = rng.normal(size=len(frame))
    frame["ybin"] = (rng.random(len(frame)) < 0.5).astype(int)
    return frame


def _lines(**kwargs) -> list[str]:
    fitted = ws.XtLogit.from_formula(
        "ybin ~ x + z", _panel(), entity="pid", time="year", method="fe"
    ).fit(**kwargs)
    return str(fitted.summary()).splitlines()


def test_the_two_notes_open_the_table() -> None:
    lines = _lines()
    assert lines[0] == "note: multiple positive outcomes within groups encountered.", lines[:3]
    assert lines[1] == "note: 1 group (8 obs) omitted because of all positive or", lines[:3]


def test_the_second_note_wraps_with_a_six_space_indent() -> None:
    assert _lines()[2] == "      all negative outcomes."


def test_a_blank_line_follows_the_notes() -> None:
    lines = _lines()
    assert lines[3] == "", lines[:5]
    assert lines[4].startswith("Conditional fixed-effects logistic regression")


def test_the_drop_counts_are_still_reported_in_e() -> None:
    fitted = ws.XtLogit.from_formula(
        "ybin ~ x + z", _panel(), entity="pid", time="year", method="fe"
    ).fit()
    assert int(fitted.e["N_group_drop"]) == 1
    assert int(fitted.e["N_drop"]) == 8


def test_the_note_plural_follows_the_count() -> None:
    """Two dropped groups must read 'groups', one must read 'group'."""

    lines = _lines()
    assert " 1 group " in lines[1], lines[1]
    assert "groups" not in lines[1], lines[1]


def test_a_fit_without_drops_has_no_notes() -> None:
    """A panel where every group varies is silent."""

    rng = np.random.default_rng(5)
    frame = pd.DataFrame(
        {"pid": np.repeat(np.arange(40), 6), "year": np.tile(np.arange(1, 7), 40)}
    )
    frame["x"] = rng.normal(size=len(frame))
    frame["ybin"] = np.tile([0, 1, 0, 1, 0, 1], 40)
    fitted = ws.XtLogit.from_formula(
        "ybin ~ x", frame, entity="pid", time="year", method="fe"
    ).fit()
    rendered = str(fitted.summary())
    assert "note: multiple positive outcomes" not in rendered
    assert "omitted because of all positive" not in rendered


def test_the_random_effects_form_is_unaffected() -> None:
    """These notes are a conditional-likelihood artefact; `re` must not carry them."""

    fitted = ws.XtLogit.from_formula(
        "ybin ~ x + z", _panel(), entity="pid", time="year", method="re"
    ).fit()
    rendered = str(fitted.summary())
    assert "omitted because of all positive" not in rendered
