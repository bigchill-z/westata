"""``estat summarize, noweights`` -- the option that must actually switch weights off.

``estat_summarize.sthlp:31-35`` lists four options, and ``noweights`` is the one with
visible consequences: by default the command ADOPTS the estimation weights
(``estat_summ.ado:208-231``), so a ``fweight`` fit reports ``Obs`` as the sum of the
weights. ``noweights`` says "ignore weights" and must therefore restore the raw counts.

Accepting the option and ignoring it is the failure shape round 138 found in
``estat gof, group()``: a plausible table answering a different question than the one
asked. These tests pin the difference in both directions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def weighted_fit():
    """A frequency-weighted OLS fit, so ``Obs`` under weighting differs from ``n``."""

    rng = np.random.default_rng(653)
    n = 200
    weights = rng.integers(1, 4, size=n).astype(float)
    frame = pd.DataFrame(
        {"y": rng.normal(size=n), "x1": rng.normal(size=n)}
    )
    result = ws.OLS.from_formula(
        "y ~ x1", frame, weights=weights, weight_type="fweight"
    ).fit()
    return frame, weights, result


def test_the_default_adopts_the_estimation_weights(weighted_fit) -> None:
    """``estat_summ.ado:208-231`` rebuilds the weight expression automatically.

    ``fweight`` counts each observation ``w`` times, so ``Obs`` is the weight SUM, not
    the row count.
    """

    frame, weights, result = weighted_fit
    table = result.estat_summarize().table

    assert table.loc["y", "Obs"] == weights.sum()


def test_noweights_restores_the_raw_counts(weighted_fit) -> None:
    """The option must produce the UNWEIGHTED table."""

    frame, _, result = weighted_fit
    table = result.estat_summarize(noweights=True).table

    assert table.loc["y", "Obs"] == len(frame)


def test_noweights_restores_the_unweighted_mean(weighted_fit) -> None:
    """Checked against the raw column, not against the weighted one.

    A mean that merely CHANGED would pass a weaker assertion; matching
    ``frame["y"].mean()`` proves the weights were actually dropped.
    """

    frame, _, result = weighted_fit
    table = result.estat_summarize(noweights=True).table

    assert table.loc["y", "Mean"] == pytest.approx(float(frame["y"].mean()), rel=1e-12)
    assert table.loc["x1", "Mean"] == pytest.approx(
        float(frame["x1"].mean()), rel=1e-12
    )


def test_the_two_tables_really_differ(weighted_fit) -> None:
    """Guard for the fixture: if both matched, the tests above would prove nothing."""

    _, _, result = weighted_fit
    weighted = result.estat_summarize().table
    unweighted = result.estat_summarize(noweights=True).table

    assert weighted.loc["y", "Obs"] != unweighted.loc["y", "Obs"]
    assert weighted.loc["y", "Mean"] != pytest.approx(
        float(unweighted.loc["y", "Mean"]), rel=1e-6
    )


def test_an_unweighted_fit_is_unaffected_by_the_option() -> None:
    """With no weights there is nothing to ignore, so the tables must match."""

    rng = np.random.default_rng(659)
    n = 150
    frame = pd.DataFrame(
        {"y": rng.normal(size=n), "x1": rng.normal(size=n)}
    )
    result = ws.OLS.from_formula("y ~ x1", frame).fit()

    default = result.estat_summarize().table
    ignored = result.estat_summarize(noweights=True).table

    for name in ("y", "x1"):
        for column in ("Obs", "Mean", "Std. Dev."):
            assert default.loc[name, column] == pytest.approx(
                ignored.loc[name, column], rel=1e-12
            ), (name, column)


def test_noweights_also_suppresses_explicitly_supplied_weights() -> None:
    """``noweights`` means "ignore weights", so it beats an explicit argument too.

    Stata's option is a statement about the OUTPUT, not about where the weights came
    from: ``estat_summarize.sthlp:35`` reads "ignore weights".  Honouring an explicit
    ``weights=`` while claiming to ignore weights would be the same silent
    contradiction this round set out to remove -- just in the other direction.

    I first wrote the opposite expectation ("explicit weights should win"), then read
    the option text again; the doc is unambiguous.
    """

    rng = np.random.default_rng(661)
    n = 120
    frame = pd.DataFrame(
        {"y": rng.normal(size=n), "x1": rng.normal(size=n)}
    )
    result = ws.OLS.from_formula("y ~ x1", frame).fit()
    supplied = np.full(n, 2.0)

    table = result.estat_summarize(
        weights=supplied, weight_type="fweight", noweights=True
    ).table

    assert table.loc["y", "Obs"] == n