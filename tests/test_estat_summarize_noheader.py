"""``estat summarize, noheader`` suppresses the "Number of obs" line.

``estat_summarize.sthlp:93`` defines the option in one sentence: "noheader suppresses
the header".

What counts as the header matters. In Stata the header is the ``Number of obs`` line
above the table; the COLUMN headings (``Variable``, ``Obs``, ``Mean`` ...) belong to the
table and stay. Suppressing those too would leave a wall of numbers with no labels --
a "successful" option that destroys the output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def result():
    rng = np.random.default_rng(683)
    n = 150
    frame = pd.DataFrame(
        {"y": rng.normal(size=n), "x1": rng.normal(size=n)}
    )
    return frame, ws.OLS.from_formula("y ~ x1", frame).fit()


def test_the_default_prints_the_header(result) -> None:
    _, fitted = result

    assert "Number of obs" in str(fitted.estat_summarize())


def test_noheader_suppresses_it(result) -> None:
    _, fitted = result

    assert "Number of obs" not in str(fitted.estat_summarize(noheader=True))


def test_the_column_headings_survive(result) -> None:
    """The table's own headings are not the header.

    Dropping them would produce unlabelled columns -- the option would "work" and the
    output would be useless.
    """

    _, fitted = result

    rendered = str(fitted.estat_summarize(noheader=True))

    # Stata's estat summarize has no "Obs" column -- every row shares the estimation-sample
    # size, which the header line already reports.
    for heading in ("Variable", "Mean", "Std. dev.", "Min", "Max"):
        assert heading in rendered, heading


def test_the_rows_are_unchanged(result) -> None:
    """Only the header goes; every statistic stays exactly as it was."""

    _, fitted = result
    full = fitted.estat_summarize()
    trimmed = fitted.estat_summarize(noheader=True)

    pd.testing.assert_frame_equal(full.table, trimmed.table)


def test_the_attribute_is_exposed(result) -> None:
    """The flag is readable afterwards, not just an argument."""

    _, fitted = result

    assert fitted.estat_summarize().noheader is False
    assert fitted.estat_summarize(noheader=True).noheader is True


def test_noheader_composes_with_the_other_options(result) -> None:
    """It is an independent switch, so it must not disturb ``noweights``."""

    frame, fitted = result
    short = fitted.estat_summarize(noheader=True, noweights=True)

    assert "Number of obs" not in str(short)
    assert short.table.loc["y", "Obs"] == len(frame)


def test_a_clustered_table_keeps_its_group_block(result) -> None:
    """The cluster block is NOT the header -- Stata prints it after the table.

    ``noheader`` refers to the line above the table, so suppressing the group
    information as well would lose content the caller asked for elsewhere.
    """

    frame, fitted = result
    frame_with_groups = frame.assign(grp=np.arange(len(frame)) % 10)
    with_cluster = ws.OLS.from_formula("y ~ x1", frame_with_groups).fit()

    full = with_cluster.estat_summarize(
        cluster=frame_with_groups["grp"], cluster_name="grp"
    )
    trimmed = with_cluster.estat_summarize(
        cluster=frame_with_groups["grp"], cluster_name="grp", noheader=True
    )

    # The clustered variant is what carries the block; the option must not remove it.
    assert full.clusters is not None
    assert trimmed.clusters == full.clusters
    assert "Number of obs" not in str(trimmed)