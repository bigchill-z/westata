"""``estat summarize`` must survive a categorical variable in the model.

Until round 121 a factor variable crashed the whole table:

    ValueError: could not convert string to float: 'b'

because ``estat_summarize`` forced every column to ``float``. That loses the entire
table -- including the numeric variables the caller actually asked about -- over one
categorical column.

Stata reports the UNDERLYING variable (``estat_summarize.sthlp:38`` refers to
factor-variable labelling); it cannot compute a mean of a string, so the numeric
cells are blank and ``Obs`` is filled. The blank is printed as Stata's ``.``, never as
``nan``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def mixed_frame():
    rng = np.random.default_rng(409)
    n = 200
    return pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "grp": rng.choice(["a", "b", "c"], size=n),
        }
    )


def test_a_factor_variable_does_not_crash_the_table(mixed_frame) -> None:
    """The whole point: one categorical column must not lose the other rows."""

    result = ws.OLS.from_formula("y ~ x1 + C(grp)", mixed_frame).fit()

    table = result.estat_summarize().table

    assert "y" in table.index
    assert "x1" in table.index


def test_the_factor_row_reports_only_an_observation_count(mixed_frame) -> None:
    """``Obs`` is countable; mean/sd/min/max are undefined for strings."""

    result = ws.OLS.from_formula("y ~ x1 + C(grp)", mixed_frame).fit()
    table = result.estat_summarize().table
    row = table.loc["grp"]

    assert row["Obs"] == 200
    for column in ("Mean", "Std. Dev.", "Min", "Max"):
        assert not np.isfinite(row[column]), column


def test_the_factor_row_is_printed_with_stata_missing_markers(mixed_frame) -> None:
    """Stata prints ``.`` for a missing statistic -- never ``nan``."""

    result = ws.OLS.from_formula("y ~ x1 + C(grp)", mixed_frame).fit()
    rendered = str(result.estat_summarize())

    grp_line = next(line for line in rendered.splitlines() if line.strip().startswith("grp"))
    assert "nan" not in grp_line.lower()
    assert grp_line.count(".") >= 4, grp_line


def test_the_numeric_rows_are_unchanged_by_the_factor(mixed_frame) -> None:
    """The factor must not disturb the statistics of the other variables."""

    with_factor = ws.OLS.from_formula("y ~ x1 + C(grp)", mixed_frame).fit()
    without = ws.OLS.from_formula("y ~ x1", mixed_frame).fit()

    a = with_factor.estat_summarize().table
    b = without.estat_summarize().table

    for name in ("y", "x1"):
        for column in ("Obs", "Mean", "Std. Dev.", "Min", "Max"):
            assert a.loc[name, column] == pytest.approx(
                b.loc[name, column], rel=1e-12
            ), (name, column)


def test_the_underlying_variable_is_named_not_the_dummy(mixed_frame) -> None:
    """The row is ``grp``, not ``C(grp)[T.b]``.

    Stata reports the underlying variable; naming a dummy would invent a row for a
    variable the user never typed.
    """

    result = ws.OLS.from_formula("y ~ x1 + C(grp)", mixed_frame).fit()
    table = result.estat_summarize().table

    assert "grp" in table.index
    assert not any("[T." in str(name) or "C(" in str(name) for name in table.index)


def test_a_purely_numeric_model_still_prints_numbers(mixed_frame) -> None:
    """Regression guard for the missing-marker change: real values still format."""

    result = ws.OLS.from_formula("y ~ x1", mixed_frame).fit()
    rendered = str(result.estat_summarize())

    assert "nan" not in rendered.lower()
    # Every numeric cell is a real number, so no bare "." cell should appear.
    for name in ("y", "x1"):
        line = next(
            line for line in rendered.splitlines() if line.strip().startswith(name)
        )
        assert "." in line  # decimal points, not missing markers
        assert " . " not in line


def test_naming_the_factor_explicitly_also_works(mixed_frame) -> None:
    """``variables=["grp"]`` must follow the same path, not a different one."""

    result = ws.OLS.from_formula("y ~ x1 + C(grp)", mixed_frame).fit()

    table = result.estat_summarize(variables=["grp"]).table

    assert list(table.index) == ["grp"]
    assert table.loc["grp", "Obs"] == 200


# ------------------------------- the rendered table under a factor model
def _rows(rendered: str) -> dict[str, str]:
    """Map each coefficient label to the first token of its value cell."""

    out: dict[str, str] = {}
    in_block = False
    for line in rendered.splitlines():
        if "Coefficient" in line and "|" in line:
            in_block = True
            continue
        if in_block and line.startswith("-"):
            continue
        if in_block and "|" in line:
            label, rest = line.split("|", 1)
            tokens = rest.split()
            if tokens:
                out[label.strip()] = tokens[0]
    return out


def test_the_factor_model_labels_the_constant_cons(mixed_frame) -> None:
    """A factor model must not fall back to patsy's ``Intercept``.

    ``stata_param_names`` refuses on ambiguous factors, and the FIRST fix printed the
    table with ``Intercept`` -- so the same command showed ``_cons`` on a numeric
    model and ``Intercept`` on a factor model.
    """

    rendered = ws.OLS.from_formula("y ~ x1 + C(grp)", mixed_frame).fit().summary(
        style="stata"
    )
    rows = _rows(rendered)

    assert "_cons" in rows
    assert "Intercept" not in rows


def test_the_constant_is_printed_last_in_a_factor_model(mixed_frame) -> None:
    """``_constant_last_order`` supplies the ordering when the mapping refuses."""

    rendered = ws.OLS.from_formula("y ~ x1 + C(grp)", mixed_frame).fit().summary(
        style="stata"
    )
    labels = list(_rows(rendered))

    assert labels[-1] == "_cons", labels


def test_every_label_carries_its_own_value_under_a_factor_model(mixed_frame) -> None:
    """The reordering shifts values and labels together, or not at all.

    This is the check that caught a double-permutation: an earlier version moved the
    values once and the labels twice, so each row displayed its neighbour's estimate.
    """

    from westata.results.summary import _stata_coef

    result = ws.OLS.from_formula("y ~ x1 + C(grp)", mixed_frame).fit()
    rows = _rows(result.summary(style="stata"))
    stored = dict(result.params.items())

    # ``_cons`` is patsy's ``Intercept``.
    expected = {
        "_cons": stored["Intercept"],
        "x1": stored["x1"],
        # A factor coefficient is DISPLAYED as a bare level under a ``grp`` group header, not as
        # the stored ``C(grp)[T.b]`` spelling: Stata prints 'grp |' then 'b | ...'. Changed in
        # round 183 when the factor rows became a two-line structure.
        "b": stored["C(grp)[T.b]"],
        "c": stored["C(grp)[T.c]"],
    }
    for label, value in expected.items():
        assert rows[label] == _stata_coef(value, 12).strip(), label


def test_a_numeric_model_is_unaffected(mixed_frame) -> None:
    """Regression guard for the shared rendering path."""

    result = ws.OLS.from_formula("y ~ x1", mixed_frame).fit()
    rows = _rows(result.summary(style="stata"))

    assert list(rows) == ["x1", "_cons"]