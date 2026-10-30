"""``lincom`` takes ONE expression; ``test`` takes several.

The two look interchangeable and are not:

* ``lincom.sthlp``'s syntax is ``lincom exp [, options]`` -- a comma separates the
  EXPRESSION from its OPTIONS, never a second expression;
* ``test`` is the command that evaluates several hypotheses at once, and round 147
  taught it the comma-separated form.

Round 147 split commas for ``test``; that change must NOT leak into ``lincom``, where a
comma means something else. Before this round ``lincom "x1 + x2, x1 - x2"`` raised a bare
``KeyError: coefficient 'x2,x1' not found`` -- technically a refusal, but one that says
nothing about why.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(751)
    n = 300
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    frame = pd.DataFrame(
        {"y": 1.0 + 0.8 * x1 - 0.3 * x2 + rng.normal(size=n), "x1": x1, "x2": x2}
    )
    return ws.OLS.from_formula("y ~ x1 + x2", frame).fit()


def test_a_comma_is_refused_with_a_reason(fitted) -> None:
    """The message must explain that a comma separates the expression from options."""

    with pytest.raises(ValueError) as caught:
        fitted.lincom("x1 + x2, x1 - x2")

    message = str(caught.value).lower()
    assert "one linear combination" in message
    assert "options" in message
    # And it points at the command that CAN do several.
    assert "test" in message


def test_the_refusal_is_not_a_bare_key_error(fitted) -> None:
    """A ``KeyError`` naming ``x2,x1`` described the symptom, not the problem."""

    with pytest.raises(ValueError):
        fitted.lincom("x1, x2")


def test_a_single_expression_still_works(fitted) -> None:
    """Regression guard for the new check."""

    rendered = fitted.lincom("x1 + x2")

    assert np.isfinite(float(rendered.estimate))
    assert np.isfinite(float(rendered.std_err))


def test_an_equality_is_still_refused_with_its_own_reason(fitted) -> None:
    """``=`` and ``,`` are separate mistakes and get separate messages."""

    with pytest.raises(ValueError) as caught:
        fitted.lincom("x1 = x2")

    assert "equals sign" in str(caught.value).lower()


def test_test_still_accepts_the_comma_form(fitted) -> None:
    """The round-147 behaviour must survive: ``test`` DOES split commas."""

    joint = fitted.test("x1 = 0, x2 = 0")

    assert float(joint.df) == 2.0


def test_the_two_commands_differ_on_purpose(fitted) -> None:
    """Guard: if ``lincom`` had started splitting commas, this would be silent.

    Asserting the contrast directly keeps a future "unify the parsers" change from
    quietly changing what a comma means.
    """

    with pytest.raises(ValueError):
        fitted.lincom("x1 = 0, x2 = 0")

    assert float(fitted.test("x1 = 0, x2 = 0").df) == 2.0


def test_a_constant_offset_is_still_handled(fitted) -> None:
    """``lincom`` moves a bare constant to the printed right-hand side.

    That behaviour predates this round and must not be disturbed by the comma check.
    """

    shifted = fitted.lincom("x1 + 10")

    assert np.isfinite(float(shifted.estimate))
    assert np.isfinite(float(shifted.std_err))