"""Panel FE drop notes are command-specific; a shared renderer once printed logit prose for poisson.

Measured on the same 40-group panel with one constant-outcome group:

    xtlogit, fe    note: multiple positive outcomes within groups encountered.
                   note: 1 group (8 obs) omitted because of all positive or
                         all negative outcomes.

    xtpoisson, fe  note: 1 group (6 obs) dropped because of all zero outcomes

The first wording is meaningless for a COUNT outcome (there is no "all positive or all negative"),
so the renderer dispatches on the estimator rather than on the drop count.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import westata as ws


def _panel(count: bool) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(
        {"pid": np.repeat(np.arange(40), 6), "year": np.tile(np.arange(1, 7), 40)}
    )
    frame["x"] = rng.normal(size=len(frame))
    if count:
        frame["y"] = rng.poisson(1.5, len(frame)).astype(float)
    else:
        frame["y"] = (rng.random(len(frame)) < 0.5).astype(float)
    frame.loc[frame["pid"] == 0, "y"] = 0.0
    return frame


def test_xtpoisson_uses_the_zero_outcome_wording() -> None:
    fitted = ws.XtPoisson.from_formula(
        "y ~ x", _panel(True), entity="pid", time="year", method="fe"
    ).fit()
    lines = str(fitted.summary()).splitlines()
    assert lines[0] == "note: 1 group (6 obs) dropped because of all zero outcomes", lines[:2]
    assert lines[1] == "", lines[:3]


def test_xtpoisson_does_not_print_logit_prose() -> None:
    rendered = str(
        ws.XtPoisson.from_formula(
            "y ~ x", _panel(True), entity="pid", time="year", method="fe"
        ).fit().summary()
    )
    assert "multiple positive outcomes" not in rendered
    assert "all positive or" not in rendered
    assert "all negative outcomes" not in rendered


def test_xtlogit_keeps_its_own_two_line_wording() -> None:
    # The seed-11 panel from round 7 (60 groups x 8 periods) drops exactly one all-positive group.
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(
        {"pid": np.repeat(np.arange(60), 8), "year": np.tile(np.arange(1, 9), 60)}
    )
    frame["x"] = rng.normal(size=len(frame))
    frame["z"] = rng.normal(size=len(frame))
    frame["y"] = (rng.random(len(frame)) < 0.5).astype(float)
    fitted = ws.XtLogit.from_formula(
        "y ~ x + z", frame, entity="pid", time="year", method="fe"
    ).fit()
    assert int(fitted.e["N_group_drop"]) == 1
    lines = str(fitted.summary()).splitlines()
    assert lines[0] == "note: multiple positive outcomes within groups encountered.", lines[:2]
    assert "omitted because of all positive or" in lines[1], lines[:3]
    assert lines[2] == "      all negative outcomes."


def test_the_drop_counts_remain_in_e() -> None:
    fitted = ws.XtPoisson.from_formula(
        "y ~ x", _panel(True), entity="pid", time="year", method="fe"
    ).fit()
    assert int(fitted.e["N_group_drop"]) == 1
    assert int(fitted.e["N_drop"]) == 6
