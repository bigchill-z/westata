"""The ``Model`` column of ``estat ic`` -- measured, not inferred.

This file previously argued that the column must carry the command name because ``e(cmd)``
holds it and a dot "claims the information is unavailable". That reasoning was plausible and
wrong. A live run settles it::

    . quietly reg consumption temp price income
    . estat ic

    Akaike's information criterion and Bayesian information criterion

    -----------------------------------------------------------------------------
           Model |          N   ll(null)  ll(model)      df        AIC        BIC
    -------------+---------------------------------------------------------------
               . |         30   39.57876   58.61944       4  -109.2389  -103.6341
    -----------------------------------------------------------------------------
    Note: BIC uses N = number of observations. See [R] IC note.

and after ``quietly var dinf unem, lags(1/2)``::

               . |        164          .   1224.456      10  -2428.912  -2397.914

The dot is what Stata prints in BOTH cases, so it is not a placeholder for missing
information -- it is the label. The old test asserted the opposite and was defending a wrong
inference drawn from what the ado files could not show.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + rng.normal(scale=0.4, size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def test_the_model_cell_is_a_dot(frame) -> None:
    """Stata prints ``.`` there, so westata prints ``.`` there."""

    rendered = str(ws.OLS.from_formula("y ~ x1", frame).fit().estat_ic())
    rows = [
        line
        for line in rendered.splitlines()
        if "|" in line and "Model" not in line and set(line) != {"-", "+"}
    ]
    assert rows, rendered
    for row in rows:
        label = row.split("|", 1)[0].strip()
        assert label == ".", row


def test_the_title_line_matches_stata(frame) -> None:
    rendered = str(ws.OLS.from_formula("y ~ x1", frame).fit().estat_ic())
    assert "Akaike's information criterion and Bayesian information criterion" in rendered


def test_the_note_matches_stata(frame) -> None:
    """Stata's note mentions BIC only; AICc and CAIC are not in this table at all."""

    rendered = str(ws.OLS.from_formula("y ~ x1", frame).fit().estat_ic())
    assert "Note: BIC uses N = number of observations. See [R] IC note." in rendered
    assert "AICc" not in rendered
    assert "CAIC" not in rendered


def test_aicc_and_caic_remain_available_as_attributes(frame) -> None:
    """They are not printed, but the numbers are still computed and useful."""

    result = ws.OLS.from_formula("y ~ x1", frame).fit()
    table = ws.estat_ic(result)
    assert np.isfinite(table.aicc)
    assert np.isfinite(table.caic)


def test_the_table_has_seven_columns(frame) -> None:
    """Model, N, ll(null), ll(model), df, AIC, BIC -- not nine."""

    rendered = str(ws.OLS.from_formula("y ~ x1", frame).fit().estat_ic())
    header = [line for line in rendered.splitlines() if "Model" in line][0]
    for column in ("Model", "N", "ll(null)", "ll(model)", "df", "AIC", "BIC"):
        assert column in header, column
