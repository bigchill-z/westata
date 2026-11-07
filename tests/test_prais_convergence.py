"""prais: the convergence point and the refit were both wrong; the estimates now match Stata.

C08-05 (0.11 / 0.19) is the weakest row of chapter 8. Round 168 rebuilt the table geometry and
noted, honestly, that the estimates still differed in the sixth decimal. Round 169 traced that to
two separate errors in the iteration, and with both fixed the numbers are exact:

    rho   0.8002263966860675   Stata .8002264
    price -1.048854            Stata -1.048854
    _cons  0.5870049           Stata .5870049
    iterations 40              Stata stops at iteration 40

THE FIRST ERROR was the default tolerance: 1e-8 where Stata uses 1e-6. The module's own docstring
had said "iterated until |change| < 1e-6" since it was written, so the comment and the code had
disagreed quietly. Tracing the rho sequence showed Stata stopping at iteration 40 with
|delta| = 8.63e-07, which is the first step under 1e-6.

THE SECOND ERROR: Stata reports the rho it stopped AT but the coefficients REFITTED at that rho.
The loop broke with the coefficients from the PREVIOUS pass. With the old code the refit values
were price -1.0488550 / _cons .5870043 against Stata's -1.048854 / .5870049; one extra fit at the
final rho gives .5870049 exactly.

Still open, and recorded: the Source block's SS/MS cells. Stata's width there is driven by the
cell rather than by significant figures -- .04494596 is eight figures while .014981987 is nine, at
the same ten-character width -- so a single %.Ng rule cannot reproduce the column. The table
layout is one or two characters from exact on those three rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\icecream.dta"
FORMULA = "consumption ~ price + income + temp"

#: From `prais consumption price income temp` on icecream.dta.
STATA_RHO = 0.8002264
STATA_COEF = {"price": -1.048854, "income": -0.0008022, "temp": 0.0029541, "_cons": 0.5870049}
STATA_ITERATIONS = 40


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("icecream.dta is not available")
    return ws.prais(frame, FORMULA)


def test_the_ar1_coefficient_matches_stata(fitted) -> None:
    assert float(fitted.rho) == pytest.approx(STATA_RHO, abs=5e-7)


def test_the_coefficients_are_refitted_at_the_final_rho(fitted) -> None:
    """The second of the two iteration bugs: breaking with the previous pass's estimates left
    _cons at .5870043 instead of .5870049."""

    assert float(fitted.params["Intercept"]) == pytest.approx(STATA_COEF["_cons"], abs=5e-7)
    assert float(fitted.params["price"]) == pytest.approx(STATA_COEF["price"], abs=5e-7)
    assert float(fitted.params["income"]) == pytest.approx(STATA_COEF["income"], abs=5e-7)
    assert float(fitted.params["temp"]) == pytest.approx(STATA_COEF["temp"], abs=5e-7)


def test_the_default_tolerance_is_one_e_minus_six(fitted) -> None:
    """Stata stops at the first step under 1e-6; the default used to be 1e-8."""

    import inspect

    signature = inspect.signature(ws.prais)
    assert signature.parameters["tolerance"].default == 1e-6
    assert int(fitted.iterations) == STATA_ITERATIONS


def test_a_tighter_tolerance_moves_rho_away_from_stata() -> None:
    """Which is what proved the tolerance, not the algorithm, was the first difference."""

    frame = pd.read_stata(DATA)
    tighter = ws.prais(frame, FORMULA, tolerance=1e-10)
    assert abs(float(tighter.rho) - STATA_RHO) > 1e-6


def test_corc_uses_the_cochrane_orcutt_transform() -> None:
    """A different estimator, not a cosmetic switch: it drops the first observation."""

    frame = pd.read_stata(DATA)
    corc = ws.prais(frame, FORMULA, corc=True)
    assert float(corc.rho) != pytest.approx(STATA_RHO, abs=1e-3)


#: The whole table from a live Stata capture, pinned verbatim in round 170 -- 19 of 19 non-blank
#: lines. TWO formatting rules had to be separated to get here: the coefficient cells use SEVEN
#: DECIMALS when the integer part is zero and SIX otherwise (the rule tebalance also uses), while
#: the Source block's SS/MS cells fill a TEN-character field with as many significant figures as
#: fit. One rule for both produced -.0008022363 where Stata prints -.0008022.
_LINE = "-" * 78
_STEP = "-" * 13 + "+" + "-" * 64
STATA_TABLE = [
    "Prais\u2013Winsten AR(1) regression with iterated estimates",
    "      Source |       SS           df       MS      Number of obs   =        30",
    "-------------+----------------------------------   F(3, 26)        =     14.35",
    "       Model |   .04494596         3  .014981987   Prob > F        =    0.0000",
    "    Residual |  .027154354        26  .001044398   R-squared       =    0.6234",
    "-------------+----------------------------------   Adj R-squared   =    0.5799",
    "       Total |  .072100315        29  .002486218   Root MSE        =    .03232",
    _LINE,
    " consumption | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]",
    _STEP,
    "       _cons |   .5870049   .2952699     1.99   0.057    -.0199311    1.193941",
    "       price |  -1.048854    .759751    -1.38   0.179    -2.610545    .5128361",
    "      income |  -.0008022   .0020458    -0.39   0.698    -.0050074    .0034029",
    "        temp |   .0029541   .0007109     4.16   0.000     .0014929    .0044152",
    _STEP,
    "         rho |   .8002264",
    _LINE,
    "Durbin\u2013Watson statistic (original)    = 1.021169",
    "Durbin\u2013Watson statistic (transformed) = 1.846795",
]


def test_the_whole_table_matches_stata(fitted) -> None:
    """19 of 19 non-blank lines, after the two formatting rules were separated."""

    rendered = [line for line in str(fitted).splitlines() if line.strip()]
    assert rendered == STATA_TABLE
