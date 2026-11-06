"""A column named after a Python builtin must still work in a formula.

C13-08 (seasonal dummy regression) scores 0.17. The model never ran:

    inf ~ C(quarter_num)
    FormulaError: cannot evaluate formula 'inf ~ C(quarter_num)':
    numbers besides '0' and '1' are only allowed with **

patsy evaluates the formula in a namespace where ``inf`` resolves to FLOAT INFINITY before it
looks at the data, so the response was read as the number ∞ rather than as a column. ``inf`` is
the natural name for an inflation series -- macro_swatson.dta uses it, and so does the textbook
this report follows -- so this is not a corner case.

Colliding columns are now aliased for the duration of the parse and the names mapped back, so
both the formula and the resulting column names keep the spelling the user wrote.

With the fix, ``inf ~ C(qn)`` on macro_swatson reproduces Stata exactly:

    _cons      .04459462
    2.qn      -.00142554
    3.qn      -.00283563
    4.qn      -.00088606
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture()
def small():
    return pd.DataFrame({"inf": np.arange(20.0), "q": [1, 2, 3, 4] * 5})


def test_inf_as_the_response(small) -> None:
    """The exact failure from the report."""

    result = ws.OLS.from_formula("inf ~ C(q)", small).fit()
    assert list(result.params.index) == ["Intercept", "C(q)[T.2]", "C(q)[T.3]", "C(q)[T.4]"]


def test_the_column_name_survives_in_the_output(small) -> None:
    """The alias must not leak into column names."""

    result = ws.OLS.from_formula("inf ~ C(q)", small).fit()
    assert all("__ws_" not in name for name in result.params.index)
    assert result.e["depvar"] == "inf"


def test_inf_as_a_predictor(small) -> None:
    frame = small.rename(columns={"inf": "y"})
    frame["inf"] = np.arange(20.0) * 2.0
    result = ws.OLS.from_formula("y ~ inf", frame).fit()
    assert "inf" in result.params.index
    assert all("__ws_" not in name for name in result.params.index)


def test_nan_and_none_columns_are_escaped_too() -> None:
    """Other builtin names would fail the same way."""

    frame = pd.DataFrame(
        {"y": np.arange(12.0), "nan": [1.0, 2.0, 3.0] * 4, "None": np.arange(12.0)}
    )
    result = ws.OLS.from_formula("y ~ nan + None", frame).fit()
    assert "nan" in result.params.index
    assert "None" in result.params.index


def test_ordinary_formulas_are_untouched(small) -> None:
    """The escaping path must not disturb the common case."""

    result = ws.OLS.from_formula("inf ~ q", small).fit()
    assert list(result.params.index) == ["Intercept", "q"]


def test_the_seasonal_regression_matches_stata() -> None:
    """C13-08's own model, against the captured e(b)."""

    try:
        frame = pd.read_stata(r"D:\myCodeProject\test_westata\data\macro_swatson.dta")
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    frame = frame.dropna(subset=["inf"]).reset_index(drop=True)
    frame["qn"] = frame["quarter"].dt.quarter
    result = ws.OLS.from_formula("inf ~ C(qn)", frame).fit()
    expected = {
        "Intercept": 0.04459462,
        "C(qn)[T.2]": -0.00142554,
        "C(qn)[T.3]": -0.00283563,
        "C(qn)[T.4]": -0.00088606,
    }
    for name, value in expected.items():
        assert float(result.params[name]) == pytest.approx(value, abs=5e-9), name
