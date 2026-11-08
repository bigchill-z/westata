"""The log-likelihood's significant figures depend on the estimator, not on the value.

Round 209 left this as an open difference on the zip header. Measuring three commands with the same block shows
the rule is per estimator:

    logit hi x        e(ll) -551.8703877742    prints -551.87039     EIGHT figures
    poisson cnt x     e(ll) -1356.3411751131   prints -1356.3412     EIGHT figures
    zip cnt x, ...    e(ll) -1356.7453901697   prints -1356.745      SEVEN figures

"%8g" throughout printed -1356.74539 for the zip; switching wholly to "%.7g" would have shortened logit's
-551.87039 to -551.8704. Neither constant works for both, so the digit count is chosen by model name.

WHY THIS IS A SEPARATE FIX FROM ROUND 209: that round changed the printed p-value, this one changes the printed
likelihood. Bundling them would have made a regression in either harder to attribute.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


def _ll_row(fitted) -> str:
    return next(
        line
        for line in str(fitted.summary(style="stata")).splitlines()
        if "Log likelihood" in line
    )


def _zip_fit(n: int = 300):
    rng = np.random.default_rng(n)
    frame = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    frame["cnt"] = rng.poisson(2.0, n).astype(float)
    return ws.ZeroInflatedPoisson.from_formula("cnt ~ x", frame, inflate="z").fit()


def _single_equation(n: int = 800):
    rng = np.random.default_rng(n)
    frame = pd.DataFrame({"y": rng.normal(size=n), "x": rng.normal(size=n)})
    frame["hi"] = (frame["y"] > 0).astype(float)
    frame["cnt"] = rng.poisson(2.0, n).astype(float)
    return {
        "Logit": ws.Logit.from_formula("hi ~ x", frame).fit(),
        "Poisson": ws.Poisson.from_formula("cnt ~ x", frame).fit(),
    }


def test_the_zero_inflated_family_prints_seven_figures() -> None:
    """-509.16183 would be eight; Stata prints -509.1618."""

    row = _ll_row(_zip_fit())
    value = row.split("=")[1].strip().split()[0]
    digits = value.replace("-", "").replace(".", "")
    assert len(digits) == 7, (value, digits)


@pytest.mark.parametrize("estimator", ["Logit", "Poisson"])
def test_the_single_equation_commands_print_eight(estimator) -> None:
    row = _ll_row(_single_equation()[estimator])
    value = row.split("=")[1].strip().split()[0]
    digits = value.replace("-", "").replace(".", "")
    assert len(digits) == 8, (estimator, value, digits)


def test_the_zip_value_is_not_truncated_to_seven_decimals() -> None:
    """-509.1618 keeps seven SIGNIFICANT figures; seven decimals would give -509.1618333."""

    row = _ll_row(_zip_fit())
    assert "-509.1618" in row, row


def test_a_single_equation_value_keeps_its_eighth_digit() -> None:
    row = _ll_row(_single_equation()["Logit"])
    assert "-551.87039" in row, row
