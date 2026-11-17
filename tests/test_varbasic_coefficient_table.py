"""`varbasic`'s coefficient table: the block the user's screenshot showed.

`varbasic` prints two blocks -- the summary (equations, RMSE, chi2) and then a coefficient table per
equation. The second one was missing entirely, and the summary had three formatting errors.

MEASURED geometry (78-wide rows, the pipe at column 13):

         dlny | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
    -------------+----------------------------------------------------------------
    dlny         |                      <- the equation's own name, even for one variable
            dlny |                      <- and its name again as the regressor group
             L1. |   .5362727   .1468267     3.65   0.000     .2484976    .8240478
                 |
           _cons |   .0437698   .0143197     3.06   0.002     .0157037     .071836

Two ordering facts are easy to get backwards:

* the LAGS come first and `_cons` LAST in the table, while the design stores the constant first;
* `L1.` is the label, not `L1.dlny`.
"""

import contextlib
import io
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "replication")

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\gdp_china.dta"

pytestmark = pytest.mark.skipif(
    not pathlib.Path(DATA).exists(), reason="gdp_china.dta is not available"
)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    frame = pd.read_stata(DATA)
    frame["lny"] = np.log(frame["y"])
    frame["dlny"] = frame["lny"].diff()
    return frame


@pytest.fixture(scope="module")
def fitted(panel):
    return ws.varbasic(panel, "dlny", lags=1, time="year", subset=panel["year"] < 2013)


# ---------------------------------------------------------------- the whole block
def test_the_block_is_byte_identical(fitted) -> None:
    expected = [
        "",
        "Vector autoregression",
        "",
        "Sample: 1980 thru 2012                          Number of obs     =         33",
        "Log likelihood =   80.96115                     AIC               =  -4.785524",
        "FPE            =   .0004889                     HQIC              =  -4.755008",
        "Det(Sigma_ml)  =   .0004331                     SBIC              =  -4.694827",
        "",
        "Equation           Parms      RMSE     R-sq      chi2     P>chi2",
        "----------------------------------------------------------------",
        "dlny                  2     .021471   0.2879   13.34017   0.0003",
        "----------------------------------------------------------------",
        "",
        "------------------------------------------------------------------------------",
        "        dlny | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]",
        "-------------+----------------------------------------------------------------",
        "dlny         |",
        "        dlny |",
        "         L1. |   .5362727   .1468267     3.65   0.000     .2484976    .8240478",
        "             |",
        "       _cons |   .0437698   .0143197     3.06   0.002     .0157037     .071836",
        "------------------------------------------------------------------------------",
    ]
    ours = str(fitted).split("\n")
    while ours and ours[-1] == "":
        ours.pop()
    assert ours == expected


# ---------------------------------------------------------------- the coefficient rows
def test_the_lag_row_shows_the_measured_figures(fitted) -> None:
    text = str(fitted)
    assert ".5362727" in text
    assert ".1468267" in text
    assert "3.65" in text
    assert ".2484976" in text
    assert ".8240478" in text


def test_the_constant_row_shows_the_measured_figures(fitted) -> None:
    text = str(fitted)
    assert ".0437698" in text
    assert ".0143197" in text
    assert ".0157037" in text


def test_the_lag_comes_before_the_constant(fitted) -> None:
    """The design stores the constant first; the table prints the lags first."""

    lines = str(fitted).split("\n")
    lag_at = next(i for i, line in enumerate(lines) if "L1." in line)
    cons_at = next(i for i, line in enumerate(lines) if "_cons |" in line)
    assert lag_at < cons_at


def test_the_regressor_label_is_the_lag_not_the_variable(fitted) -> None:
    """Measured: the row says `L1.`, not `L1.dlny`."""

    assert fitted.regressor_names[-1] == "_cons"
    assert "L1." in fitted.regressor_names
    assert "L1.dlny" not in fitted.regressor_names


def test_the_equation_name_appears_twice(fitted) -> None:
    """Stata prints `dlny         |` and then `        dlny |` inside the same block."""

    lines = str(fitted).split("\n")
    groups = [line for line in lines if line.rstrip().endswith("|") and "dlny" in line]
    assert len(groups) >= 2


# ---------------------------------------------------------------- lags(2) specifics
def test_lags_two_table_still_puts_the_lag_first(panel) -> None:
    """The label is `L2.` here; keying the reorder on `L1.` broke this case once."""

    fitted = ws.varbasic(panel, "dlny", lags=2, time="year")
    lines = str(fitted).split("\n")
    lag_at = next(i for i, line in enumerate(lines) if "L2." in line)
    cons_at = next(i for i, line in enumerate(lines) if "_cons |" in line)
    assert lag_at < cons_at
    assert "L2." in fitted.regressor_names


def test_lags_two_coefficients_are_correct(panel) -> None:
    """Stata: L2 = -.0332398, _cons = .0974691."""

    fitted = ws.varbasic(panel, "dlny", lags=2, time="year")
    text = str(fitted)
    assert "-.0332398" in text
    assert ".0974691" in text


# ---------------------------------------------------------------- number formats
def test_information_criteria_drop_trailing_zeros(panel) -> None:
    """Measured on lags(2): HQIC prints `-4.41417`, not `-4.414170`."""

    from westata.varbasic import _trim

    assert _trim(-4.41417) == "-4.41417"
    assert _trim(-4.35399) == "-4.35399"


def test_the_chi2_column_uses_significant_figures() -> None:
    """13.34017 (7 sig figs) and .0364487 (7 sig figs once the zero is dropped)."""

    from westata.varbasic import _sig

    assert _sig(13.3401662, 7) == "13.34017"
    assert _sig(0.0364487, 7) == ".0364487"


# ---------------------------------------------------------------- against live Stata
def test_the_whole_block_matches_live_stata(panel) -> None:
    try:
        from lib.stata_session import StataSession
    except Exception:  # noqa: BLE001
        pytest.skip("live Stata is not available")

    fitted = ws.varbasic(panel, "dlny", lags=1, time="year", subset=panel["year"] < 2013)
    ours = str(fitted).split("\n")
    while ours and ours[-1] == "":
        ours.pop()

    try:
        with StataSession() as stata:
            stata.push(panel)
            with contextlib.redirect_stdout(io.StringIO()):
                stata.run("tsset year")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                stata.run("varbasic dlny if year<2013, lags(1) irf")
            lines = out.getvalue().split("\n")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"live Stata unavailable: {exc}")

    while lines and lines[-1] == "":
        lines.pop()
    assert ours == lines
