"""``e(title)`` per command, transcribed from each ado -- including the empty ones.

Every estimator sets ``e(title)`` with a literal, and the help files do not document it
because the stored-results block covers ESTIMATES, not the display plumbing:

    regress.ado:176   ereturn local title "Linear regression"
    tobit.ado:127     ereturn local title "Tobit regression"
    logit.ado:253     ereturn local title "Logistic regression"
    probit.ado:252    ereturn local title "Probit regression"
    qreg.ado:189      ereturn local title          <-- SET EMPTY

**The empty case is the interesting one.** ``qreg`` does not publish a title, and it says
so by assigning the macro nothing rather than leaving it alone. That is the same
distinction as ``tobit``'s empty ``e(marginsnotok)``: "the command spoke and published
nothing" is different from "the command never spoke".

``poisson`` is deliberately absent from this file: ``poisson.ado`` contains no
``ereturn local title`` at all (tested in ``test_logit_ereturn_locals``), so its title is
genuinely unknown rather than empty.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import westata as ws

BASE = Path(r"D:\Program Files\StataNow19\ado\base")


def _frame(n: int = 250, seed: int = 6200):
    rng = np.random.default_rng(seed)
    covariate = rng.normal(size=n)
    latent = 0.4 + 0.9 * covariate
    continuous = 1.0 + 0.8 * covariate + rng.normal(size=n)
    return pd.DataFrame(
        {
            "y": continuous,
            "yc": np.clip(continuous, 0.0, None),
            "yb": (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-latent))).astype(int),
            "x": covariate,
        }
    )


@pytest.fixture(scope="module")
def frame():
    return _frame()


# ------------------------------------------------------------------ the values
def test_regress_calls_itself_linear_regression(frame) -> None:
    assert ws.OLS.from_formula("y ~ x", frame).fit().e["title"] == "Linear regression"


def test_tobit_calls_itself_tobit_regression(frame) -> None:
    assert (
        ws.Tobit.from_formula("yc ~ x", frame, lower=0).fit().e["title"]
        == "Tobit regression"
    )


def test_logit_calls_itself_logistic_regression(frame) -> None:
    assert (
        ws.Logit.from_formula("yb ~ x", frame).fit().e["title"]
        == "Logistic regression"
    )


def test_probit_calls_itself_probit_regression(frame) -> None:
    assert (
        ws.Probit.from_formula("yb ~ x", frame).fit().e["title"] == "Probit regression"
    )


def test_qreg_publishes_NO_title(frame) -> None:
    """``qreg.ado:189`` is a BARE ``ereturn local title`` -- it CLEARS the macro.

    A bare ``ereturn local name`` does not set the macro to empty; it removes it, so the
    key is ABSENT afterwards.  This file originally recorded ``""`` and an existing test
    (``test_qreg_e_surface::test_cleared_macros_are_absent``) caught it -- the same
    distinction the qreg surface file documents for ``mss``/``rss``/``ll``.
    """

    fitted = ws.QuantileRegression.from_formula("y ~ x", frame).fit()

    assert "title" not in fitted.e


# ------------------------------------------------------- the evidence, re-read
def test_the_literals_are_in_the_ados() -> None:
    """Read the authority instead of trusting the transcription above."""

    expected = {
        "r/regress.ado": 'ereturn local title "Linear regression"',
        "t/tobit.ado": 'ereturn local title "Tobit regression"',
        "l/logit.ado": 'ereturn local title "Logistic regression"',
        "p/probit.ado": 'ereturn local title "Probit regression"',
    }
    for relative, literal in expected.items():
        text = (BASE / relative).read_text(encoding="utf-8", errors="ignore")
        assert literal in text, relative


def test_qreg_assigns_an_empty_title() -> None:
    """The bare assignment, confirmed from the ado.

    ``ereturn local title`` with nothing after it is the CLEAR form; the test below
    asserts the consequence (no key) rather than the text.
    """

    text = (BASE / "q" / "qreg.ado").read_text(encoding="utf-8", errors="ignore")

    assert "ereturn local title\n" in text or "ereturn local title\r\n" in text


def test_poisson_has_no_title_assignment_at_all() -> None:
    """Why poisson is not in this file: the macro is never set, not set to empty."""

    text = (BASE / "p" / "poisson.ado").read_text(encoding="utf-8", errors="ignore")

    assert "ereturn local title" not in text


def test_poisson_records_its_title_via_the_ml_option() -> None:
    """``poisson.ado:241`` passes ``title(Poisson regression)`` to ``ml``.

    There is no ``ereturn local title`` line in that ado, which is why an earlier round
    concluded the key was absent.  ``poisson.sthlp:247`` documents it, and the value is
    recoverable from the ``title()`` option -- so the key IS recorded, by the other route.
    """

    frame = _frame()
    fitted = ws.Poisson.from_formula("yb ~ x", frame).fit()

    assert fitted.e["title"] == "Poisson regression"


# ------------------------------------------------------------------- the spread
def test_the_titles_are_distinct(frame) -> None:
    """Guard: a shared constant would satisfy every per-command test above."""

    titles = {
        ws.OLS.from_formula("y ~ x", frame).fit().e["title"],
        ws.Tobit.from_formula("yc ~ x", frame, lower=0).fit().e["title"],
        ws.Logit.from_formula("yb ~ x", frame).fit().e["title"],
        ws.Probit.from_formula("yb ~ x", frame).fit().e["title"],
    }

    assert len(titles) == 4