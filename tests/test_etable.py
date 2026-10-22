"""Tests for the ``esttab``-style table builder.

Reference behaviour: Stata's ``esttab`` puts coefficients with significance
stars, standard errors in parentheses beneath them, and a statistics block at
the bottom.  Star thresholds are ``*`` p<0.05, ``**`` p<0.01, ``***`` p<0.001.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.report.etable import etable, to_latex, to_markdown


def _data(n: int = 100, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 2.0 * x1 + 0.0 * x2 + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


@pytest.fixture(scope="module")
def results():
    df = _data()
    return [
        ws.OLS.from_formula("y ~ x1", df).fit(),
        ws.OLS.from_formula("y ~ x1 + x2", df).fit(),
    ]


def test_table_shape_and_columns(results):
    table = etable(results)
    assert list(table.columns) == ["(1)", "(2)"]
    # Coefficients rows interleaved with standard-error rows, then statistics.
    assert "x1" in table.index
    assert "Observations" in table.index
    assert "R-squared" in table.index


def test_custom_model_names(results):
    table = etable(results, model_names=["Base", "Full"])
    assert list(table.columns) == ["Base", "Full"]


def test_standard_errors_in_parentheses(results):
    table = etable(results)
    position = list(table.index).index("x1")
    se_row = table.iloc[position + 1]
    assert all(
        value.startswith("(") and value.endswith(")") for value in se_row if value
    )


def test_stars_reflect_pvalues():
    df = _data()
    res = ws.OLS.from_formula("y ~ x1 + x2", df).fit()
    table = etable([res])
    # x1 is strongly significant (true effect 2, noise 0.5); x2 is not.
    assert "*" in table.loc["x1", "(1)"]
    assert "*" not in table.loc["x2", "(1)"]


def test_stars_can_be_disabled(results):
    table = etable(results, stars=False)
    assert "*" not in table.loc["x1", "(1)"]


def test_missing_coefficient_renders_blank(results):
    table = etable(results)
    # Model (1) does not contain x2.
    position = list(table.index).index("x2")
    assert table.iloc[position]["(1)"] == ""


def test_keep_and_drop(results):
    kept = etable(results, keep=["x1"])
    assert "x1" in kept.index
    assert "x2" not in kept.index

    dropped = etable(results, drop=["x1"])
    assert "x2" in dropped.index
    assert "x1" not in dropped.index


def test_keep_with_unknown_name_raises(results):
    with pytest.raises(KeyError, match="not present"):
        etable(results, keep=["nonexistent"])


def test_observations_formatted_with_thousands_separator():
    rng = np.random.default_rng(3)
    n = 1500
    df = pd.DataFrame({"y": rng.normal(size=n), "x": rng.normal(size=n)})
    res = ws.OLS.from_formula("y ~ x", df).fit()
    table = etable([res])
    assert table.loc["Observations", "(1)"] == "1,500"


def test_custom_statistics_block(results):
    table = etable(results, stats=[("N", "N"), ("r2_a", "Adj. R2")])
    assert "N" in table.index
    assert "Adj. R2" in table.index


def test_markdown_renderer(results):
    text = to_markdown(etable(results, model_names=["A", "B"]), caption="Main")
    assert "**Main**" in text
    assert "| A | B |" in text
    assert text.count("|") > 10


def test_latex_renderer(results):
    text = to_latex(etable(results), caption="Cap", label="tab:main")
    assert "\\begin{tabular}" in text
    assert "\\caption{Cap}" in text
    assert "\\label{tab:main}" in text
    assert "\\end{tabular}" in text


def test_mixed_model_types_can_be_combined():
    df = _data()
    rng = np.random.default_rng(11)
    n = len(df)
    z = rng.normal(size=n)
    x_endo = 0.9 * z + rng.normal(size=n)
    y = 1.0 + 2.0 * x_endo + rng.normal(size=n)
    iv_df = pd.DataFrame({"y": y, "x_endo": x_endo, "z": z})

    ols = ws.OLS.from_formula("y ~ x1", df).fit()
    iv = ws.IV2SLS.from_formula("y ~ [x_endo ~ z]", iv_df).fit()

    table = etable([ols, iv], model_names=["OLS", "IV"])
    assert list(table.columns) == ["OLS", "IV"]
    assert table.loc["Observations", "OLS"] == "100"
    assert table.loc["Observations", "IV"] == "100"
    # Each model keeps its own coefficient names; the other model's cell is blank.
    assert table.loc["x1", "IV"] == ""
    assert table.loc["x_endo", "OLS"] == ""


def test_empty_results_raise():
    with pytest.raises(ValueError, match="at least one"):
        etable([])


def test_model_names_length_mismatch_raises(results):
    with pytest.raises(ValueError, match="model_names"):
        etable(results, model_names=["only-one"])
