"""``estat summarize`` summarises the WEIGHT VARIABLE itself without weights.

``estat_summarize.sthlp:96-99``:

    The default when weights are present is to perform a weighted summarize on all
    variables except the weight variable itself.  An unweighted summarize is performed
    on the weight variable.

That is not a cosmetic rule. Weighting the weight column by itself computes
``sum(w^2)/sum(w)`` -- the weights weighted by themselves -- which is a number with no
interpretation. Measured on a ``fweight`` fit whose 200 weights (1..3) sum to 393, the
weighted summary reported ``Obs=393, Mean=2.303`` where Stata reports
``Obs=200, Mean=1.965``.

``weight_spec`` stores the weight VALUES but no column name (the weights may have been
passed as an array), so the column is identified by matching its values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def weighted_case():
    rng = np.random.default_rng(673)
    n = 200
    weights = rng.integers(1, 4, size=n).astype(float)
    frame = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "w": weights,
        }
    )
    result = ws.OLS.from_formula(
        "y ~ x1", frame, weights=weights, weight_type="fweight"
    ).fit()
    return frame, weights, result


def test_the_weight_variable_is_reported_unweighted(weighted_case) -> None:
    """``Obs`` for the weight column is the ROW count, not the weight sum."""

    frame, weights, result = weighted_case
    table = result.estat_summarize(variables=["w"]).table

    assert table.loc["w", "Obs"] == len(frame)
    assert table.loc["w", "Obs"] != weights.sum()


def test_the_weight_variable_mean_is_the_raw_mean(weighted_case) -> None:
    """Checked against the raw column, so a merely-different number cannot pass."""

    frame, weights, result = weighted_case
    table = result.estat_summarize(variables=["w"]).table

    assert table.loc["w", "Mean"] == pytest.approx(float(frame["w"].mean()), rel=1e-12)


def test_the_other_variables_stay_weighted(weighted_case) -> None:
    """The exemption applies to the weight column ONLY.

    If the fix had switched weights off for the whole table, the other variables would
    silently become unweighted -- trading one wrong row for several.
    """

    frame, weights, result = weighted_case
    table = result.estat_summarize(variables=["y", "w"]).table

    assert table.loc["y", "Obs"] == weights.sum()
    assert table.loc["w", "Obs"] == len(frame)


def test_the_two_columns_are_treated_differently(weighted_case) -> None:
    """Guard for the fixture: both cannot be right under one rule."""

    frame, weights, result = weighted_case
    table = result.estat_summarize(variables=["y", "w"]).table

    assert table.loc["y", "Obs"] != table.loc["w", "Obs"]


def test_the_weight_variable_std_dev_is_unweighted_too(weighted_case) -> None:
    """Every statistic of that row comes from the unweighted summarise."""

    frame, weights, result = weighted_case
    table = result.estat_summarize(variables=["w"]).table

    assert table.loc["w", "Std. Dev."] == pytest.approx(
        float(frame["w"].std(ddof=1)), rel=1e-12
    )
    assert table.loc["w", "Min"] == pytest.approx(float(frame["w"].min()), rel=1e-12)
    assert table.loc["w", "Max"] == pytest.approx(float(frame["w"].max()), rel=1e-12)


def test_an_unweighted_fit_needs_no_exemption() -> None:
    """With no weights every variable is summarised from the raw column."""

    rng = np.random.default_rng(677)
    n = 150
    frame = pd.DataFrame(
        {"y": rng.normal(size=n), "x1": rng.normal(size=n)}
    )
    result = ws.OLS.from_formula("y ~ x1", frame).fit()
    table = result.estat_summarize().table

    for name in ("y", "x1"):
        assert table.loc[name, "Obs"] == n
        assert table.loc[name, "Mean"] == pytest.approx(
            float(frame[name].mean()), rel=1e-12
        )


def test_noweights_still_works_with_the_exemption(weighted_case) -> None:
    """``noweights`` makes every row unweighted, including the weight column.

    The two rules must not fight: ``noweights`` is the broader statement.
    """

    frame, weights, result = weighted_case
    table = result.estat_summarize(variables=["y", "w"], noweights=True).table

    assert table.loc["y", "Obs"] == len(frame)
    assert table.loc["w", "Obs"] == len(frame)