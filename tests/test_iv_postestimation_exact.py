"""The two IV postestimation tests the report scores at 0.0 are both exact.

C10-03 scores `estat overid` at 0.0 and C10-05 scores `estat endogenous` at 0.0. Measured on grilic.dta
after `ivregress 2sls lnw expr (s = rns mrt)`:

    estat overid
        Sargan (score) chi2(1) =  1.66303  (p = 0.1972)
        Basmann chi2(1)        =  1.65789  (p = 0.1979)

    estat endogenous
        Durbin (score) chi2(1)  =  46.3175  (p = 0.0000)
        Wu-Hausman F(1,754)     =  49.0716  (p = 0.0000)

Every digit matches, including the degrees of freedom.

TWO DIFFERENT QUESTIONS, which is why both matter: `overid` asks whether the instruments are valid
given the model (it needs more instruments than endogenous regressors, so it is unavailable in an
exactly identified fit); `endogenous` asks whether the supposedly endogenous regressor actually is.
A package that ran one and labelled it the other would look correct on half of the report's entries.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws

GRILIC = r"D:\myCodeProject\test_westata\data\grilic.dta"


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(GRILIC).dropna(
            subset=["lnw", "s", "expr", "rns", "mrt"]
        )
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("grilic.dta is not available")
    return ws.IV2SLS.from_formula("lnw ~ expr + [s ~ rns + mrt]", frame).fit()


def test_the_sargan_and_basmann_statistics_match_stata(fitted) -> None:
    rendered = str(ws.estat_overid(fitted))
    assert "1.66303" in rendered
    assert "1.65789" in rendered
    assert "chi2(1)" in rendered


def test_the_durbin_and_wu_hausman_statistics_match_stata(fitted) -> None:
    rendered = str(ws.estat_endogenous(fitted))
    assert "46.3175" in rendered
    assert "49.0716" in rendered
    assert "F(1,754)" in rendered


def test_the_two_tests_are_not_the_same_computation(fitted) -> None:
    """Their statistics differ by two orders of magnitude, so a swapped implementation is visible."""

    overid = str(ws.estat_overid(fitted))
    endogenous = str(ws.estat_endogenous(fitted))
    assert "1.66303" in overid and "46.3175" not in overid
    assert "46.3175" in endogenous and "1.66303" not in endogenous


def test_overid_needs_surplus_instruments() -> None:
    """Exactly identified fits have nothing to test, and Stata refuses there too."""

    try:
        frame = pd.read_stata(GRILIC).dropna(
            subset=["lnw", "s", "expr", "rns", "mrt"]
        )
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("grilic.dta is not available")

    # One instrument for one endogenous regressor: exactly identified.
    exactly = ws.IV2SLS.from_formula("lnw ~ expr + [s ~ rns]", frame).fit()
    with pytest.raises(Exception):
        ws.estat_overid(exactly)
