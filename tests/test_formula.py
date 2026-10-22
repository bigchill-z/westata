"""Tests for the westata formula layer (P0 Task 4).

The formula layer is a thin patsy wrapper: it adds Stata-style
``C(x, base=...)`` reference-level sugar, then delegates to
``patsy.dmatrices`` so the model matrix is identical to
``statsmodels.formula.api``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import patsy
import pytest
import statsmodels.formula.api as smf

from westata.errors import FormulaError
from westata.formula import build_design


def _factor_frame() -> pd.DataFrame:
    """Simple two-level factor design used by the reference-level tests."""
    return pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": ["a", "b", "a", "b"],
        }
    )


def test_factor_module_exposes_function() -> None:
    from westata.formula.factor import build_design as factor_build_design

    assert factor_build_design is build_design


def test_factor_base_level_single_quotes() -> None:
    X, names, idx = build_design("y ~ C(x, base='b')", _factor_frame())

    assert names == ["Intercept", "C(x, Treatment(reference='b'))[T.a]"]
    assert "[T.a]" in names[1]
    np.testing.assert_array_equal(
        X,
        np.array([[1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 0.0]]),
    )
    assert isinstance(X, np.ndarray)
    assert X.dtype == np.float64
    assert list(idx) == [0, 1, 2, 3]


def test_factor_base_level_double_quotes() -> None:
    X, names, _ = build_design('y ~ C(x, base="b")', _factor_frame())

    assert names == ["Intercept", 'C(x, Treatment(reference="b"))[T.a]']
    np.testing.assert_array_equal(X[:, 1], np.array([1.0, 0.0, 1.0, 0.0]))


def test_factor_base_numeric_level() -> None:
    df = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "g": [1, 2, 3, 1],
        }
    )
    X, names, _ = build_design("y ~ C(g, base=2)", df)

    assert names == [
        "Intercept",
        "C(g, Treatment(reference=2))[T.1]",
        "C(g, Treatment(reference=2))[T.3]",
    ]
    np.testing.assert_array_equal(
        X[:, 1:],
        np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
    )


def test_missing_rows_dropped_and_index_kept() -> None:
    df = pd.DataFrame({"y": [1.0, np.nan, 3.0], "x": [1.0, 2.0, 3.0]})
    X, names, idx = build_design("y ~ x", df)

    assert list(idx) == [0, 2]
    assert X.shape == (2, 2)
    assert names == ["Intercept", "x"]


def test_kept_index_preserves_original_labels() -> None:
    df = pd.DataFrame(
        {
            "y": [1.0, np.nan, 3.0, 4.0],
            "x": [1.0, 2.0, np.nan, 4.0],
        },
        index=["r0", "r1", "r2", "r3"],
    )
    X, _, idx = build_design("y ~ x", df)

    assert isinstance(idx, pd.Index)
    assert list(idx) == ["r0", "r3"]
    assert X.shape == (2, 2)
    np.testing.assert_array_equal(X[:, 1], np.array([1.0, 4.0]))


def test_design_matrix_matches_statsmodels_with_interaction_and_factor() -> None:
    rng = np.random.default_rng(20261022)
    n = 24
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x": rng.normal(size=n),
            "g": np.tile(["a", "b", "c"], n // 3),
        }
    )
    formula = "y ~ x * C(g)"

    X, names, idx = build_design(formula, df)
    model = smf.ols(formula, data=df)

    np.testing.assert_allclose(X, np.asarray(model.exog, dtype=float))
    assert names == model.exog_names
    assert list(idx) == list(df.index)


def test_multiple_base_terms_match_patsy_translation() -> None:
    df = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "a": ["z", "z", "y", "y", "x", "x"],
            "b": [1, 2, 3, 1, 2, 3],
        }
    )
    explicit = (
        "y ~ C(a, Treatment(reference='z')) + C(b, Treatment(reference=2))"
    )
    expected_y, expected_X = patsy.dmatrices(explicit, df, return_type="dataframe")

    X, names, _ = build_design("y ~ C(a, base='z') + C(b, base=2)", df)

    np.testing.assert_allclose(X, np.asarray(expected_X, dtype=float))
    assert names == list(expected_X.columns)


def test_base_term_inside_interaction() -> None:
    df = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "x": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
            "g": ["a", "b", "c", "a", "b", "c"],
        }
    )
    _, expected_X = patsy.dmatrices(
        "y ~ x * C(g, Treatment(reference='c'))", df, return_type="dataframe"
    )

    X, names, _ = build_design("y ~ x * C(g, base='c')", df)

    np.testing.assert_allclose(X, np.asarray(expected_X, dtype=float))
    assert names == list(expected_X.columns)


def test_collinear_columns_are_not_dropped() -> None:
    df = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x1": [1.0, 2.0, 3.0, 4.0],
            "x2": [2.0, 4.0, 6.0, 8.0],
        }
    )
    X, names, _ = build_design("y ~ x1 + x2", df)

    assert names == ["Intercept", "x1", "x2"]
    assert X.shape == (4, 3)


def test_empty_data_raises() -> None:
    df = pd.DataFrame(
        {
            "y": pd.Series(dtype=float),
            "x": pd.Series(dtype=float),
        }
    )
    with pytest.raises(FormulaError, match="empty"):
        build_design("y ~ x", df)


def test_invalid_formula_raises() -> None:
    with pytest.raises(FormulaError, match="cannot evaluate"):
        build_design("y ~ x +", _factor_frame())


def test_unknown_column_raises_and_lists_available_columns() -> None:
    # A missing column must produce a readable message naming the columns that
    # do exist; the wrapped patsy error alone is hard to act on.
    with pytest.raises(FormulaError) as excinfo:
        build_design("y ~ not_a_column", _factor_frame())
    assert "not_a_column" in str(excinfo.value)
    assert "available columns" in str(excinfo.value)
