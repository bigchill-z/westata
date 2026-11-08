"""Integer-coded factors expand, so `reg y i.m` works on an int64 column.

C13-08 scores `reg airpassengers i.m` at 0.125 with the diagnosis "the integer factor column C(m) is
not expanded: F(0,143) = 0.00, the coefficient table holds only _cons". That was true at some point
and is not true now:

    ws.OLS.from_formula("y ~ C(m)", frame).fit()
        12 parameters -- Intercept plus eleven dummies
        C(m)[T.2] ... C(m)[T.12]

Stata's `i.m` for 12 months produces exactly that: eleven indicator columns against a constant.

THE GUARD CHECKS THE DEGREES OF FREEDOM AND THE COLUMN NAMES rather than just "it ran", because a
silent collapse to a single constant is the failure mode the report recorded and it would still
produce a table.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def seasonal():
    rng = np.random.default_rng(1)
    n = 144
    return pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "m": np.tile(np.arange(1, 13), 12).astype("int64"),
        }
    )


def test_an_integer_factor_expands_to_eleven_dummies(seasonal) -> None:
    """Stata's `i.m` for 12 levels gives a constant plus eleven indicators."""

    fitted = ws.OLS.from_formula("y ~ C(m)", seasonal).fit()
    assert len(fitted.params) == 12
    assert int(fitted.df_model) == 11


def test_the_reference_level_is_dropped(seasonal) -> None:
    """The base level appears as the constant, not as its own column."""

    fitted = ws.OLS.from_formula("y ~ C(m)", seasonal).fit()
    names = list(fitted.params.index)
    assert "Intercept" in names
    assert "C(m)[T.1]" not in names
    assert "C(m)[T.2]" in names
    assert "C(m)[T.12]" in names


def test_the_model_is_not_degenerate(seasonal) -> None:
    """F(0, 143) = 0.00 was the recorded symptom; a real regression has a positive model df."""

    fitted = ws.OLS.from_formula("y ~ C(m)", seasonal).fit()
    assert float(fitted.rsquared) > 0.0
    assert int(fitted.df_model) > 0


def test_a_string_factor_behaves_the_same_way(seasonal) -> None:
    """The report contrasted int and string factors; both must expand."""

    frame = seasonal.copy()
    frame["s"] = frame["m"].astype(str)
    from_int = ws.OLS.from_formula("y ~ C(m)", frame).fit()
    from_str = ws.OLS.from_formula("y ~ C(s)", frame).fit()
    assert len(from_int.params) == len(from_str.params)


def test_the_r_squared_and_f_match_stata(seasonal) -> None:
    """Stata: e(r2) = 0.055690, e(df_m) = 11, e(F) = 0.707687."""

    fitted = ws.OLS.from_formula("y ~ C(m)", seasonal).fit()
    assert float(fitted.rsquared) == pytest.approx(0.055690, abs=1e-6)
    assert int(fitted.df_model) == 11
    assert float(fitted.fvalue) == pytest.approx(0.707687, abs=1e-6)
