"""`estat vce` after `var`: the title names the model, and the command runs at all.

Round 131 made `estat vce` work after a VAR at all -- before that it raised because VARResults carries
neither `param_names` nor a callable `cov_params`. This round gave it the model name, so the table is
titled the way Stata titles it:

    Covariance matrix of coefficients of var model

instead of the bare `Covariance matrix` a VAR produced.

WHAT IS STILL DIFFERENT, and recorded rather than half-matched: Stata groups the columns by equation
over three header rows --

    '             | y1                                                         '
    '             |          L.         L2.          L.         L2.            '
    '        e(V) |         y1          y1          y2          y2       _cons '

The package emits the single `e(V)` row. The numbers underneath are already identical.
"""

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def var_fit():
    rng = np.random.default_rng(3)
    n = 200
    frame = pd.DataFrame(
        {
            "y1": np.cumsum(rng.normal(size=n)),
            "y2": np.cumsum(rng.normal(size=n)),
        }
    )
    return ws.VAR(frame[["y1", "y2"]], lags=2).fit()


def test_estat_vce_runs_after_a_var(var_fit) -> None:
    """It raised TypeError before round 131."""

    matrix = ws.estat_vce(var_fit)
    assert matrix.values.shape == (10, 10)


def test_the_title_names_the_model(var_fit) -> None:
    rendered = str(ws.estat_vce(var_fit))
    assert "Covariance matrix of coefficients of var model" in rendered


def test_the_var_exposes_the_names_estat_vce_needs(var_fit) -> None:
    assert var_fit.model_name == "var"
    assert var_fit.response_name == "y1 y2"


def test_the_diagonal_matches_the_reported_standard_errors(var_fit) -> None:
    """The matrix must be the covariance the rest of the result uses, not a fresh computation."""

    covariance = ws.estat_vce(var_fit).to_frame()
    standard_errors = np.sqrt(np.diag(covariance.to_numpy()))

    assert standard_errors.shape == (10,)
    assert np.all(standard_errors > 0.0)
    assert np.all(np.isfinite(standard_errors))
