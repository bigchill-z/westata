"""The ``Model`` column of ``estat ic``.

Until round 120 the column was a literal ``"."``, which reads as "missing" -- yet the
command name is always known. Stata labels the row with the estimation command that
produced the estimates.

``estat ic`` is a builtin with no ado source (there is no ``e/estat_ic.ado``), so the
exact cell formatting cannot be read out of Stata's files. What CAN be settled is that
the column carries the command name, because that is what ``e(cmd)`` holds and what the
row is for; printing a dot there claims the information is unavailable.

The numbers in the row are checked elsewhere (round 97's identity layer pins
``AIC == -2*ll + 2*k``); this file is only about the label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(401)
    n = 200
    latent = rng.normal(size=n)
    return pd.DataFrame(
        {
            "y": latent,
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "yb": (latent > 0).astype(int),
            "ycnt": rng.poisson(3.0, size=n),
        }
    )


CASES = {
    "OLS": (lambda f: ws.OLS.from_formula("y ~ x1", f).fit(), "regress"),
    "Logit": (lambda f: ws.Logit.from_formula("yb ~ x1", f).fit(), "logit"),
    "Poisson": (lambda f: ws.Poisson.from_formula("ycnt ~ x1", f).fit(), "poisson"),
}


@pytest.mark.parametrize("label", sorted(CASES))
def test_the_model_column_names_the_command(frame, label) -> None:
    """The row is labelled with ``e(cmd)``, not with a placeholder."""

    factory, expected = CASES[label]
    result = factory(frame)
    ic = result.estat_ic()

    assert ic.model == expected


@pytest.mark.parametrize("label", sorted(CASES))
def test_the_rendered_row_carries_that_name(frame, label) -> None:
    """End-to-end: the label appears in the printed table."""

    factory, expected = CASES[label]
    rendered = str(factory(frame).estat_ic())

    row = next(line for line in rendered.splitlines() if "|" in line and "Model" not in line)
    assert expected in row, row


def test_the_placeholder_is_gone(frame) -> None:
    """A lone ``.`` in the Model cell would claim the name is unavailable."""

    rendered = str(ws.OLS.from_formula("y ~ x1", frame).fit().estat_ic())
    rows = [
        line
        for line in rendered.splitlines()
        if "|" in line and "Model" not in line and set(line) != {"-", "+"}
    ]

    assert rows, rendered
    for row in rows:
        label = row.split("|", 1)[0].strip()
        assert label != ".", row


def test_the_label_matches_the_recorded_e_cmd(frame) -> None:
    """The label is read from ``e(cmd)``, so the two cannot disagree."""

    result = ws.OLS.from_formula("y ~ x1", frame).fit()

    assert result.estat_ic().model == result.e["cmd"]


def test_a_result_without_a_command_still_renders(frame) -> None:
    """The dot remains as a last resort, for a result built with no command.

    Stripping it entirely would produce a blank cell -- less informative than a
    visible "not available".
    """

    from westata.postestimation.ic import IcResult

    empty = IcResult(
        n=100.0, ll_null=-1.0, ll_model=-0.5, df=2.0,
        aic=5.0, bic=6.0, aicc=5.1, caic=6.5,
    )
    rendered = str(empty)

    assert empty.model == ""
    row = next(line for line in rendered.splitlines() if "|" in line and "Model" not in line)
    assert row.split("|", 1)[0].strip() == "."


def test_the_numbers_are_unchanged_by_the_label(frame) -> None:
    """Regression guard: only the label changed.

    ``estat ic``'s AIC is ``-2*ll + 2*df`` and BIC is ``-2*ll + log(N)*df`` -- the
    identities pinned in round 97.
    """

    result = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()
    ic = result.estat_ic()
    e = result.e

    assert ic.aic == pytest.approx(-2.0 * e["ll"] + 2.0 * e["rank"], rel=1e-10)
    assert ic.bic == pytest.approx(
        -2.0 * e["ll"] + np.log(e["N"]) * e["rank"], rel=1e-10
    )