"""The `L()` lag operator, which round 114 recorded as missing and round 115 implemented.

Stata writes `L.temp` and `L2.temp`; patsy's grammar cannot express either, so this exposes the same
idea as a call and rewrites it before patsy sees it:

    L(temp)     ->  __ws_lag(temp, 1)
    L(temp, 2)  ->  __ws_lag(temp, 2)

VERIFIED AGAINST A LIVE STATA on macro_swatson.dta:

    reg dinf L.dinf unem
        b[L.dinf]   Stata -0.27168536    package -0.271685
        b[unem]     Stata -0.22244491    package -0.222445
        e(N)        Stata 165            package 165

and `L(dinf, 2)` drops two periods, leaving 164 observations.

TWO FORMS WERE TRIED AND FAILED FIRST, both recorded because they look reasonable:

  * a parenthesised lambda -- patsy's parser rejects it with "expected an operator, not '('";
  * `patsy.EvalEnvironment(..., depth=0)` -- not this patsy's signature.

The working form is a module-level helper resolved through `capture(0).with_outer_namespace(...)`.
"""

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"


@pytest.fixture(scope="module")
def macro():
    try:
        return pd.read_stata(DATA)
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("macro_swatson.dta is not available")


def test_the_lag_operator_is_accepted(macro) -> None:
    """It raised `NameError: name 'L' is not defined` before this round."""

    fitted = ws.OLS.from_formula("dinf ~ L(dinf) + unem", macro).fit()
    assert int(fitted.nobs) == 165


def test_the_lag_coefficients_match_stata(macro) -> None:
    fitted = ws.OLS.from_formula("dinf ~ L(dinf) + unem", macro).fit()
    assert float(fitted.params["L(dinf)"]) == pytest.approx(-0.27168536, abs=1e-6)
    assert float(fitted.params["unem"]) == pytest.approx(-0.22244491, abs=1e-6)


def test_a_second_lag_drops_one_more_observation(macro) -> None:
    one = ws.OLS.from_formula("dinf ~ L(dinf)", macro).fit()
    two = ws.OLS.from_formula("dinf ~ L(dinf, 2)", macro).fit()

    assert int(one.nobs) == 165
    assert int(two.nobs) == 164


def test_it_matches_a_manually_shifted_column(macro) -> None:
    """The same regression done by hand must agree, or the rewrite is doing something else."""

    manual = macro.copy()
    manual["lagged"] = manual["dinf"].shift(1)
    manual = manual.dropna(subset=["dinf", "lagged", "unem"])

    by_operator = ws.OLS.from_formula("dinf ~ L(dinf) + unem", macro).fit()
    by_hand = ws.OLS.from_formula("dinf ~ lagged + unem", manual).fit()

    assert float(by_operator.params["L(dinf)"]) == pytest.approx(
        float(by_hand.params["lagged"]), rel=1e-10
    )


def test_a_bad_period_is_rejected(macro) -> None:
    with pytest.raises(Exception):
        ws.OLS.from_formula("dinf ~ L(dinf, 0)", macro).fit()
