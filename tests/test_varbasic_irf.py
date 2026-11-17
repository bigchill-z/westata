"""`varbasic` and `irf table` against captured StataNow 19.5 output.

Two things about this command are counter-intuitive and both are pinned here:

1. ``varbasic ..., irf`` prints ONLY the estimation block. It registers the IRF in ``r()`` but shows
   no table; ``irf table irf`` is the second command that reveals the responses.
2. ``lags(p)`` keeps ONLY lag p, not the range 1..p. ``varbasic dlny, lags(2)`` gives an ``e(b)`` of
   ``L2.dlny`` and ``_cons`` -- two coefficients, not three -- and its IRF is exactly zero on every
   step that is not a multiple of 2.

Everything below was measured on gdp_china.dta with ``dlny = d.ln(y)``.
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


def _rows(rendered: str) -> list[str]:
    return [line for line in rendered.splitlines() if line.strip() and line.strip()[0].isdigit()]


# ---------------------------------------------------------------- the user's command
def test_the_user_command_matches_stata(panel) -> None:
    """`varbasic dlny if year<2013, lags(1) irf` then `irf table irf`."""

    expected = [
        "",
        "Results from varbasic",
        "",
        "---------------------------------------------",
        "         |      (1)         (1)         (1)  ",
        "    Step |      irf       Lower       Upper  ",
        "---------+-----------------------------------",
        "       0 |        1           1           1",
        "       1 |  .536273     .248498     .824048",
        "       2 |  .287588    -.021063      .59624",
        "       3 |  .154226    -.094057     .402508",
        "       4 |  .082707    -.094822     .260236",
        "       5 |  .044354    -.074652     .163359",
        "       6 |  .023786    -.052797     .100369",
        "       7 |  .012756    -.035159      .06067",
        "       8 |   .00684    -.022525     .036206",
        "---------------------------------------------",
        "95% lower and upper bounds reported.",
        "(1) irfname = varbasic, impulse = dlny, and response = dlny.",
        "",
    ]
    fitted = ws.varbasic(panel, "dlny", lags=1, time="year", subset=panel["year"] < 2013)
    assert str(ws.irf_table(fitted)).split("\n") == expected


def test_the_estimation_block_numbers_match(panel) -> None:
    """Log likelihood, the information criteria and the equation Wald statistic."""

    fitted = ws.varbasic(panel, "dlny", lags=1, time="year", subset=panel["year"] < 2013)
    assert fitted.nobs == 33
    assert fitted.ll == pytest.approx(80.96115340, abs=1e-8)
    assert fitted.aic == pytest.approx(-4.78552445, abs=1e-8)
    assert fitted.hqic == pytest.approx(-4.75500752, abs=1e-8)
    assert fitted.sbic == pytest.approx(-4.69482702, abs=1e-8)
    assert fitted.fpe == pytest.approx(0.00048895, abs=1e-8)
    assert fitted.chi2[0] == pytest.approx(13.34017, abs=1e-5)


# ---------------------------------------------------------------- lags(p) means lag p only
def test_lags_two_keeps_only_lag_two(panel) -> None:
    """`lags(2)` has TWO coefficients (L2, _cons), not three."""

    fitted = ws.varbasic(panel, "dlny", lags=2, time="year")
    assert np.asarray(fitted.coefficients).shape == (2, 1)


def test_lags_two_response_is_zero_on_odd_steps(panel) -> None:
    """Measured: steps 1, 3 are exactly 0; the responses live on the even steps."""

    fitted = ws.varbasic(panel, "dlny", lags=2, time="year")
    values = ws.irf_table(fitted).responses[0].values
    assert values[0] == pytest.approx(1.0)
    assert values[1] == 0.0
    assert values[3] == 0.0
    assert values[2] != 0.0


def test_lags_three_response_is_zero_off_multiples_of_three(panel) -> None:
    fitted = ws.varbasic(panel, "dlny", lags=3, time="year")
    values = ws.irf_table(fitted).responses[0].values
    assert values[1] == 0.0 and values[2] == 0.0
    assert values[3] != 0.0


def test_lags_two_coefficient_matches_stata(panel) -> None:
    """Stata reports L2 = -0.03323977 and _cons = 0.09746906."""

    fitted = ws.varbasic(panel, "dlny", lags=2, time="year")
    flat = np.asarray(fitted.coefficients).ravel()
    assert flat[0] == pytest.approx(0.09746906, abs=1e-7)
    assert flat[1] == pytest.approx(-0.03323977, abs=1e-7)


# ---------------------------------------------------------------- the `if` condition
def test_subset_equals_prefiltering_the_frame(panel) -> None:
    """R3.4: `subset=` is `data.loc[mask]` before anything is estimated, so the two agree.

    A head cut therefore yields 28 observations here, not Stata's 29 -- Stata builds the lag over the
    complete time axis (`_qsur.ado:54`) and this package deliberately does not reproduce that, leaving
    the sample choice to pandas.
    """

    by_subset = ws.varbasic(panel, "dlny", lags=1, time="year", subset=panel["year"] >= 1985)
    prefiltered = ws.varbasic(
        panel[panel["year"] >= 1985].reset_index(drop=True), "dlny", lags=1, time="year"
    )
    assert by_subset.nobs == prefiltered.nobs == 28
    assert by_subset.ll == prefiltered.ll


def test_keep_selecting_nothing_raises(panel) -> None:
    with pytest.raises(ws.ConsistencyError):
        ws.varbasic(panel, "dlny", lags=1, time="year", subset=panel["year"] > 3000)


# ---------------------------------------------------------------- number format
def test_small_values_use_scientific_notation() -> None:
    """Stata's `%10.0g`: below 1e-5 the format switches to science, as `.1.2e-06` shows."""

    from westata.varbasic import _irf_number

    assert _irf_number(1.2e-06) == "1.2e-06"
    assert _irf_number(9.9e-06) == "9.9e-06"
    assert _irf_number(1e-05) == ".00001"
    assert _irf_number(0.1) == ".1"
    assert _irf_number(1.0) == "1"
    assert _irf_number(-0.094057) == "-.094057"


# ---------------------------------------------------------------- class form
def test_the_class_form_agrees_with_the_function(panel) -> None:
    """R2.7 requires an estimation command to have a class; it must give the same answer."""

    by_function = ws.varbasic(panel, "dlny", lags=1, time="year")
    by_class = ws.Varbasic(panel, "dlny", lags=1, time="year").fit()
    assert by_function.ll == by_class.ll
    assert str(ws.irf_table(by_function)) == str(ws.irf_table(by_class))


# ---------------------------------------------------------------- whole-table comparison
def test_irf_table_matches_live_stata(panel) -> None:
    """Line-by-line against a live Stata run, for the user's own command."""

    try:
        from lib.stata_session import StataSession
    except Exception:  # noqa: BLE001
        pytest.skip("live Stata is not available")

    fitted = ws.varbasic(panel, "dlny", lags=1, time="year", subset=panel["year"] < 2013)
    ours = str(ws.irf_table(fitted)).split("\n")

    try:
        with StataSession() as stata:
            stata.push(panel)
            with contextlib.redirect_stdout(io.StringIO()):
                stata.run("tsset year")
                stata.run("quietly varbasic dlny if year<2013, lags(1) irf")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                stata.run("irf table irf")
            stata_lines = out.getvalue().split("\n")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"live Stata unavailable: {exc}")

    assert ours == stata_lines
