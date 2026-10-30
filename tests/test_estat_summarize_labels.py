"""``estat summarize, labels`` shows variable labels in an extra column.

``estat_summarize.sthlp:90``: "labels displays variable labels."

Two design points matter, and both are about not inventing information:

* the column is appended AFTER the statistics, so the numeric headings keep their
  positions -- putting it beside the name would shift every other column;
* an UNLABELLED variable gets an empty cell rather than falling back to its name.  A
  name printed in a label column reads as real metadata, which it is not.

The labels are supplied by the caller because ``pandas.read_stata`` does not retain
Stata variable labels unless asked, so the frame alone cannot provide them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(691)
    n = 150
    frame = pd.DataFrame(
        {"y": rng.normal(size=n), "x1": rng.normal(size=n)}
    )
    return frame, ws.OLS.from_formula("y ~ x1", frame).fit()


def test_without_the_option_no_label_column_appears(fitted) -> None:
    _, result = fitted

    rendered = str(result.estat_summarize())

    assert "Label" not in rendered
    assert result.estat_summarize().labels is None


def test_the_labels_appear_in_the_table(fitted) -> None:
    _, result = fitted

    rendered = str(
        result.estat_summarize(
            labels={"y": "Outcome variable", "x1": "First regressor"}
        )
    )

    assert "Outcome variable" in rendered
    assert "First regressor" in rendered


def test_the_statistics_columns_keep_their_positions(fitted) -> None:
    """The label column must not displace the numeric headings.

    A test that only looks for the label text would pass even if the headings had
    shifted -- every column would still be present, just meaning something else.
    """

    _, result = fitted
    plain = str(result.estat_summarize()).splitlines()
    labelled = str(result.estat_summarize(labels={"y": "Outcome"})).splitlines()

    plain_header = next(line for line in plain if "Variable" in line)
    labelled_header = next(line for line in labelled if "Variable" in line)

    # The labelled header extends the plain one; the prefix is unchanged.
    assert labelled_header.startswith(plain_header.rstrip())
    assert labelled_header.index("Max") == plain_header.index("Max")


def test_an_unlabelled_variable_gets_an_empty_cell(fitted) -> None:
    """Falling back to the variable name would fabricate metadata."""

    _, result = fitted

    rendered = str(result.estat_summarize(labels={"y": "Outcome variable"}))
    x1_line = next(line for line in rendered.splitlines() if line.startswith("x1"))

    # The x1 row must not repeat its own name in the label position.
    assert x1_line.count("x1") == 1, x1_line


def test_the_label_mapping_is_readable_afterwards(fitted) -> None:
    _, result = fitted
    supplied = {"y": "Outcome variable"}

    rendered = result.estat_summarize(labels=supplied)

    assert rendered.labels == supplied


def test_labels_compose_with_noheader(fitted) -> None:
    """Independent switches must not cancel each other out."""

    _, result = fitted

    rendered = str(
        result.estat_summarize(labels={"y": "Outcome"}, noheader=True)
    )

    assert "Number of obs" not in rendered
    assert "Outcome" in rendered
    assert "Variable" in rendered


def test_labels_compose_with_noweights(fitted) -> None:
    frame, result = fitted

    rendered = result.estat_summarize(labels={"y": "Outcome"}, noweights=True)

    assert "Outcome" in str(rendered)
    assert rendered.table.loc["y", "Obs"] == len(frame)


def test_labels_do_not_change_the_table_data(fitted) -> None:
    """The option affects rendering only."""

    _, result = fitted
    plain = result.estat_summarize()
    labelled = result.estat_summarize(labels={"y": "Outcome"})

    pd.testing.assert_frame_equal(plain.table, labelled.table)