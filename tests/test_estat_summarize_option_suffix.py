"""`estat summarize` names the command without its option suffix.

Round 218 mapped the absorbed fit's `reghdfe` to `areg`. This round found the second way the name can be wrong:
this package's `model_name` carries the estimation option while Stata's heading names just the command.

    xtlogit hi x, re      Stata '  Estimation sample xtlogit'
                          package '  Estimation sample xtlogit, re'
    xtreg y x, fe         Stata '  Estimation sample xtreg'
    xtreg y x, re         Stata '  Estimation sample xtreg'

So the heading takes everything before the comma. `areg` still needs its own mapping because the absorbed fit
reports the BACKEND's name rather than an option.

BOTH CORRECTIONS ARE NEEDED, IN THIS ORDER: strip the option, then map. Verified on `xtlogit, re` at n=600 with the
whole block byte-identical, and the unaffected estimators -- `regress`, `logit`, `poisson` -- still name themselves
correctly.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


def _heading(fitted) -> str:
    return next(
        line
        for line in str(ws.estat_summarize(fitted)).splitlines()
        if "Estimation sample" in line
    )


def _panel_frame(n: int = 600) -> pd.DataFrame:
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
    return frame


def test_a_random_effects_logit_names_only_the_command() -> None:
    fitted = ws.XtLogit.from_formula(
        "hi ~ x", _panel_frame(), entity="id", time="t", method="re"
    ).fit()
    heading = _heading(fitted)
    assert "xtlogit" in heading, heading
    assert "," not in heading, heading


def test_no_heading_carries_a_comma() -> None:
    """A trailing ", re" would be an option, not part of the command."""

    fitted = ws.XtLogit.from_formula(
        "hi ~ x", _panel_frame(), entity="id", time="t", method="re"
    ).fit()
    assert not _heading(fitted).rstrip().endswith("re")


def test_the_absorbed_fit_still_says_areg() -> None:
    fitted = ws.FEOLS.from_formula("y ~ x | id", _panel_frame()).fit()
    heading = _heading(fitted)
    assert "areg" in heading
    assert "reghdfe" not in heading


@pytest.mark.parametrize(
    "estimator,formula,expected",
    [("OLS", "y ~ x", "regress"), ("Logit", "hi ~ x", "logit")],
)
def test_the_plain_commands_are_unaffected(estimator, formula, expected) -> None:
    frame = _panel_frame(500)
    fitted = getattr(ws, estimator).from_formula(formula, frame).fit()
    assert expected in _heading(fitted)
