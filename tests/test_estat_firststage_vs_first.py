"""Two different Stata outputs that are easy to confuse, both reproduced exactly.

    ivregress 2sls ..., first   -> result.first_stage() / summary(first=True)
                                   the fitted coefficients of each first-stage equation

    estat firststage            -> ws.estat_firststage(result)
                                   R-sq / adjusted / partial R-sq / instrument F, plus the
                                   Stock-Yogo critical-value panel -- no coefficients at all

A user compared `first_stage()` against `estat firststage` and reported a mismatch. Nothing was
missing: both are implemented and each matches its own command byte-for-byte on grilic (22 lines for
`estat firststage`, 22 for `, first` on the same model). The confusion is that the two share the word
"first-stage", so these guards pin the DISTINCTION as much as the layout.
"""

import pathlib
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"
FORMULA = "lnw ~ s + expr + tenure + rns + smsa + [iq ~ med + kww]"

pytestmark = pytest.mark.skipif(
    not pathlib.Path(DATA).exists(),
    reason="grilic.dta is not available on this machine",
)


def _fitted():
    return ws.IV2SLS.from_formula(FORMULA, pd.read_stata(DATA)).fit()


# ---------------------------------------------------------------- estat firststage
def test_estat_firststage_is_the_summary_shape() -> None:
    lines = str(ws.estat_firststage(_fitted())).splitlines()
    text = "\n".join(lines)
    assert "First-stage regression summary statistics" in text
    assert "Minimum eigenvalue statistic" in text
    assert "Critical Values" in text
    assert "2SLS relative bias" in text
    assert "LIML size of nominal 5% Wald test" in text


def test_estat_firststage_has_no_coefficient_rows() -> None:
    """It is a diagnostics panel, not a regression table."""

    text = str(ws.estat_firststage(_fitted()))
    assert "Coefficient" not in text
    assert "Std. err." not in text
    assert "[95% conf. interval]" not in text


def test_estat_firststage_carries_the_measured_statistics() -> None:
    text = str(ws.estat_firststage(_fitted()))
    # R-sq, adjusted R-sq, partial R-sq, F(2,750), Prob > F -- all from the live capture.
    assert "0.3066" in text
    assert "0.3001" in text
    assert "0.0382" in text
    assert "14.9058" in text
    assert "F(2,750)" in text


def test_estat_firststage_is_twenty_two_lines() -> None:
    lines = str(ws.estat_firststage(_fitted())).splitlines()
    while lines and lines[-1] == "":
        lines.pop()
    assert len(lines) == 22, len(lines)


# ---------------------------------------------------------------- , first
def test_first_stage_is_the_coefficient_table() -> None:
    lines = str(_fitted().first_stage()).splitlines()
    assert lines[0] == "First-stage regressions"
    assert any("Coefficient" in line for line in lines)
    assert any(line.strip().startswith("kww |") for line in lines)
    assert any(line.strip().startswith("_cons |") for line in lines)


def test_first_stage_has_no_critical_value_panel() -> None:
    text = str(_fitted().first_stage())
    assert "Critical Values" not in text
    assert "Minimum eigenvalue" not in text


# ---------------------------------------------------------------- they are not substitutes
def test_the_two_outputs_are_different() -> None:
    summary = str(ws.estat_firststage(_fitted()))
    table = str(_fitted().first_stage())
    assert summary != table
    assert "summary statistics" in summary and "summary statistics" not in table
    assert "Coefficient" in table and "Coefficient" not in summary


def test_summary_first_prepends_the_table_not_the_summary() -> None:
    """Stata's `, first` prints the coefficient table; `estat firststage` is a separate call."""

    lines = str(_fitted().summary(first=True)).splitlines()
    assert lines[1] == "First-stage regressions", lines[:2]
    assert any(line.strip().startswith("iq |") for line in lines)
