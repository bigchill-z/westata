"""Fitted values and leverage, then summarized -- both match Stata digit for digit.

C05-08 (`predict lnw1 ; summarize lnw1`) and C09-04 (`predict lev, leverage ; summarize lev`) both score
0.0. Measured on grilic.dta after `reg lnw s expr`:

          lnw1 |        758    5.686739    .2341174   5.138869   6.321817
           lev |        758    .0039578    .0030557   .0013628    .029407

Both rows match exactly, header and statistics.

LEVERAGE IS REACHED THROUGH A DICT, not an attribute, and that shape difference is worth pinning:
`fitted.get_influence()["hat_matrix_diag"]` rather than `.hat_matrix_diag`. My first attempt used the
attribute spelling, got an AttributeError, and would have recorded "leverage is missing" -- the same
entry-point trap as irf_table in round 145.

A third guard checks the leverage column is not a copy of the residual column: both live on the same
influence object and a wrong key would return plausible numbers of the wrong kind.
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
        frame = pd.read_stata(GRILIC).dropna(subset=["lnw", "s", "expr"])
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("grilic.dta is not available")
    return ws.OLS.from_formula("lnw ~ s + expr", frame).fit()


def test_the_fitted_values_match_stata(fitted) -> None:
    rendered = str(ws.summarize(pd.DataFrame({"lnw1": fitted.fittedvalues})))
    line = next(row for row in rendered.splitlines() if "lnw1" in row)
    assert "5.686739" in line
    assert ".2341174" in line
    assert "5.138869" in line
    assert "6.321817" in line


def test_the_leverage_values_match_stata(fitted) -> None:
    leverage = fitted.get_influence()["hat_matrix_diag"]
    rendered = str(ws.summarize(pd.DataFrame({"lev": leverage})))
    line = next(row for row in rendered.splitlines() if "lev" in row)
    assert ".0039578" in line
    assert ".0030557" in line
    assert ".0013628" in line
    assert ".029407" in line


def test_leverage_is_reached_through_the_dict_key(fitted) -> None:
    """The attribute spelling does not exist; the key does."""

    influence = fitted.get_influence()
    assert "hat_matrix_diag" in influence
    assert not hasattr(influence, "hat_matrix_diag")


def test_leverage_is_not_the_residual_vector(fitted) -> None:
    """Both live on the influence object, so a wrong key still returns real-looking numbers."""

    influence = fitted.get_influence()
    assert not np.allclose(
        influence["hat_matrix_diag"], influence["resid"]
    )


def test_leverage_stays_within_zero_and_one(fitted) -> None:
    """A hat value is a leverage in [0, 1]; anything else means the wrong array was read."""

    leverage = np.asarray(fitted.get_influence()["hat_matrix_diag"], dtype=float)
    assert float(leverage.min()) >= 0.0
    assert float(leverage.max()) <= 1.0
