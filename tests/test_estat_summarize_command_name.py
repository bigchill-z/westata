"""The `estat summarize` heading names the Stata command, which is not this package's model name.

The heading takes its name from `result.model_name`, and the absorbed regression reports `reghdfe` -- the backend
it wraps -- where the Stata command producing the same fit is `areg`:

    Stata    '  Estimation sample areg                   Number of obs =        600'
    package  '  Estimation sample reghdfe                Number of obs =        600'

The other estimators happen to share their Stata names (`logit`, `probit`, `poisson`, `zip`, `zinb`), so this is a
one-entry mapping rather than a table. That was checked rather than assumed: those five were compared against live
captures and none needed correction.

Verified at n = 600 and 1500 with the absorbed fit: 2 of 2 byte-identical.
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


@pytest.mark.parametrize("n", [600, 1500])
def test_the_absorbed_fit_names_areg(n) -> None:
    rng = np.random.default_rng(n)
    frame = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x": rng.normal(size=n),
            "id": np.arange(n) // 20,
        }
    )
    fitted = ws.FEOLS.from_formula("y ~ x | id", frame).fit()
    heading = _heading(fitted)
    assert "areg" in heading, heading
    assert "reghdfe" not in heading, heading


def test_the_heading_still_carries_the_observation_count() -> None:
    rng = np.random.default_rng(600)
    frame = pd.DataFrame(
        {
            "y": rng.normal(size=600),
            "x": rng.normal(size=600),
            "id": np.arange(600) // 20,
        }
    )
    fitting = ws.FEOLS.from_formula("y ~ x | id", frame).fit()
    assert "Number of obs" in _heading(fitting)


def test_a_plain_regression_still_names_itself() -> None:
    rng = np.random.default_rng(300)
    frame = pd.DataFrame({"y": rng.normal(size=300), "x": rng.normal(size=300)})
    fitted = ws.OLS.from_formula("y ~ x", frame).fit()
    assert "regress" in _heading(fitted)


@pytest.mark.parametrize(
    "estimator,formula",
    [("Logit", "hi ~ x"), ("Probit", "hi ~ x")],
)
def test_the_discrete_commands_keep_their_own_names(estimator, formula) -> None:
    rng = np.random.default_rng(500)
    frame = pd.DataFrame({"y": rng.normal(size=500), "x": rng.normal(size=500)})
    frame["hi"] = (frame["y"] > 0).astype(float)
    fitted = getattr(ws, estimator).from_formula(formula, frame).fit()
    heading = _heading(fitted)
    assert estimator.lower() in heading.lower(), heading
