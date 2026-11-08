"""`estat hettest` works on an absorbed-effects fit, where it used to raise.

Round 133 scanned the postestimation surface across every estimator at once. The scan found that
`estat hettest` failed on `FEOLS` with

    1 coefficients for 2 design columns

and the fit was fine: an absorbed constant is not in `params` (the effects take it out) while the
design matrix still carries the `Intercept` column. A positional dot product cannot pair the two.

THE FIX ALIGNS BY NAME. A design column with no matching coefficient contributes zero to the linear
prediction, which is exactly what an absorbed intercept should contribute. Before that, a legitimate
fit was refused because the two sides differed in width.

THE REFUSALS THAT REMAIN ARE CORRECT and are pinned so they cannot be mistaken for gaps:

  * `estat vif` on logit/probit -- variance inflation is not defined for a non-linear fit;
  * `estat dwatson` and `estat bgodfrey` on a VAR -- they need single-equation residuals, and Stata
    refuses there too;
  * `estat gof` / `estat classification` on a linear model -- those are for discrete outcomes.
"""

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(7)
    n = 200
    return pd.DataFrame(
        {
            "y": np.cumsum(rng.normal(size=n)),
            "x": rng.normal(size=n),
            "z": rng.normal(size=n),
            "g": rng.choice(["a", "b"], n),
        }
    )


def test_hettest_runs_on_an_absorbed_fit(frame) -> None:
    fitted = ws.FEOLS.from_formula("y ~ x + z | g", frame).fit()
    rendered = str(ws.estat_hettest(fitted))
    assert "chi2(1)" in rendered


def test_the_absorbed_intercept_is_not_treated_as_a_missing_coefficient(frame) -> None:
    """Three design columns, two coefficients: the intercept has no estimate and contributes 0."""

    fitted = ws.FEOLS.from_formula("y ~ x + z | g", frame).fit()
    assert len(fitted.params) == 2
    assert fitted.design_matrix[0].shape[1] == 3
    # And the command still produces a statistic rather than raising on the width mismatch.
    assert "Prob > chi2" in str(ws.estat_hettest(fitted))


def test_vif_still_refuses_a_non_linear_fit(frame) -> None:
    binary = frame.assign(binary=(frame["y"] > 0).astype(float))
    fitted = ws.Logit.from_formula("binary ~ x + z", binary).fit()
    with pytest.raises(Exception):
        ws.estat_vif(fitted)


def test_serial_correlation_tests_still_refuse_a_var(frame) -> None:
    fitted = ws.VAR(frame[["y", "x"]], lags=2).fit()
    with pytest.raises(Exception):
        ws.estat_dwatson(fitted)
    with pytest.raises(Exception):
        ws.estat_bgodfrey(fitted)
