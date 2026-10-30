"""``test`` prints the restrictions above the statistic.

Stata's output for ``test x1 = 0, x2 = 0``::

     ( 1)  x1 = 0
     ( 2)  x2 = 0

           F(  2,   297) =  156.36
                Prob > F =    0.0000

``WaldTest.constraint`` already held that text — numbered, one restriction per line — but
``__str__`` never rendered it, so the table showed a statistic with no statement of what
was being tested. A number without its hypothesis cannot be checked by a reader.

The first fix wrapped ``constraint`` in another ``( n)`` prefix, which would have printed
``( 1) ( 1) x1 = 0`` — the stored text already carries the numbering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(757)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    frame = pd.DataFrame(
        {"y": 1.0 + 0.8 * x1 - 0.3 * x2 + rng.normal(size=n), "x1": x1, "x2": x2}
    )
    return ws.OLS.from_formula("y ~ x1 + x2", frame).fit()


def test_a_single_restriction_is_printed(fitted) -> None:
    rendered = str(fitted.test("x1 = 0"))

    assert "x1 = 0" in rendered
    assert rendered.index("x1 = 0") < rendered.index("F(")


def test_each_restriction_gets_its_own_line(fitted) -> None:
    rendered = str(fitted.test("x1 = 0, x2 = 0"))
    lines = [line.strip() for line in rendered.splitlines() if line.strip()]

    numbered = [line for line in lines if line.startswith("(")]
    assert len(numbered) == 2, lines
    assert "( 1)" in numbered[0]
    assert "( 2)" in numbered[1]


def test_the_numbering_is_not_doubled(fitted) -> None:
    """``constraint`` already carries ``( 1)``; adding another prefix would show it twice.

    This is the mistake the first version made, and it would still "contain the
    restriction text", so only counting the markers catches it.
    """

    rendered = str(fitted.test("x1 = 0"))

    assert rendered.count("( 1)") == 1, rendered
    assert "( 1) ( 1)" not in rendered


def test_the_statistic_comes_after_the_restrictions(fitted) -> None:
    """Stata prints restrictions first, then a blank line, then the F/chi2 block."""

    rendered = str(fitted.test("x1 = 0, x2 = 0"))
    lines = rendered.splitlines()

    restriction_lines = [i for i, line in enumerate(lines) if line.strip().startswith("(")]
    statistic_lines = [i for i, line in enumerate(lines) if line.strip().startswith("F(")]

    assert restriction_lines and statistic_lines
    assert max(restriction_lines) < min(statistic_lines)


def test_an_equality_chain_is_printed_in_NORMALISED_form(fitted) -> None:
    """``test x1 = x2`` is printed as ``x1 - x2 = 0``.

    The restriction is moved to the left-hand side, which is how Stata states it, so
    asserting the literal expression would be asserting something Stata does not do.
    My first version of this test expected ``"x1 = x2"`` in the output and failed for
    that reason -- the normalisation was correct.
    """

    rendered = str(fitted.test("x1 = x2"))

    assert "x1 - x2 = 0" in rendered


def test_the_numbers_are_unchanged(fitted) -> None:
    """Rendering only: statistic, df and p-value must keep their values."""

    tested = fitted.test("x1 = 0, x2 = 0")

    assert float(tested.df) == 2.0
    assert np.isfinite(float(tested.statistic))
    assert 0.0 <= float(tested.pvalue) <= 1.0


def test_a_chi2_distribution_still_renders_the_restrictions() -> None:
    """The constraint block is independent of the F-vs-chi2 switch."""

    rng = np.random.default_rng(761)
    n = 250
    latent = rng.normal(size=n)
    frame = pd.DataFrame(
        {"yb": (latent > 0).astype(int), "x1": rng.normal(size=n)}
    )
    logit = ws.Logit.from_formula("yb ~ x1", frame).fit()

    rendered = str(logit.test("x1 = 0"))

    assert "x1 = 0" in rendered
    assert "chi2(" in rendered