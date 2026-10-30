"""Every ML command's ``e(title)``, including the one that sets it indirectly.

The help files document ``e(title)`` for all three:

    logit.sthlp:262    {cmd:e(title)}  title in estimation output
    probit.sthlp:265   (same)
    poisson.sthlp:247  (same)

But the ados supply it by two different routes, and only one is greppable:

    logit.ado:253      ereturn local title "Logistic regression"
    probit.ado:252     ereturn local title "Probit regression"
    poisson.ado:241    ... `mlopts' title(Poisson regression) ...

``poisson`` hands the string to ``ml`` as a ``title()`` option, so a search of
``poisson.ado`` for ``ereturn local title`` finds NOTHING while the macro still exists.
Round 233 concluded from that absence that poisson had no title; round 241 found the
``title()`` call and corrected it.

The lesson is narrow and worth keeping: **"not found by grep in the ado" is not the same as
"does not exist"** when the value travels through a called program's option.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.results.summary import render_summary

BASE = Path(r"D:\Program Files\StataNow19\ado\base")

#: The literal each ado ends up storing.
EXPECTED = {
    "logit": "Logistic regression",
    "probit": "Probit regression",
    "poisson": "Poisson regression",
}


def _frame(n: int = 250, seed: int = 7000):
    rng = np.random.default_rng(seed)
    covariate = rng.normal(size=n)
    latent = 0.4 + 0.9 * covariate
    return pd.DataFrame(
        {
            "yb": (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-latent))).astype(int),
            "ycnt": rng.poisson(np.exp(0.3 + 0.4 * covariate)),
            "x": covariate,
        }
    )


@pytest.fixture(scope="module")
def frame():
    return _frame()


BUILDERS = {
    "logit": lambda frame: ws.Logit.from_formula("yb ~ x", frame).fit(),
    "probit": lambda frame: ws.Probit.from_formula("yb ~ x", frame).fit(),
    "poisson": lambda frame: ws.Poisson.from_formula("ycnt ~ x", frame).fit(),
}


# ---------------------------------------------------------------- the values
@pytest.mark.parametrize("command", sorted(BUILDERS))
def test_the_title_matches_the_ado(frame, command: str) -> None:
    assert BUILDERS[command](frame).e["title"] == EXPECTED[command]


@pytest.mark.parametrize("command", sorted(BUILDERS))
def test_the_stata_banner_uses_that_title(frame, command: str) -> None:
    """The renderer reads the macro, so the banner follows automatically."""

    fitted = BUILDERS[command](frame)
    banner = render_summary(fitted, style="stata").splitlines()[0].strip()

    assert banner == EXPECTED[command]


def test_poisson_is_no_longer_the_generic_banner(frame) -> None:
    """The specific regression: poisson used to print the neutral fallback."""

    banner = (
        render_summary(BUILDERS["poisson"](frame), style="stata").splitlines()[0].strip()
    )

    assert banner != "Stata-style regression results"


# ------------------------------------------------------------ the evidence
def test_logit_and_probit_set_the_title_with_ereturn(frame) -> None:
    """The greppable route."""

    for relative, literal in (
        ("l/logit.ado", 'ereturn local title "Logistic regression"'),
        ("p/probit.ado", 'ereturn local title "Probit regression"'),
    ):
        text = (BASE / relative).read_text(encoding="utf-8", errors="ignore")
        assert literal in text, relative


def test_poisson_sets_the_title_through_ml_not_ereturn(frame) -> None:
    """The non-greppable route -- pinning it stops the old wrong conclusion recurring."""

    text = (BASE / "p/poisson.ado").read_text(encoding="utf-8", errors="ignore")

    assert 'title(Poisson regression)' in text
    assert "ereturn local title" not in text


def test_the_help_files_document_the_key_for_all_three() -> None:
    """The help files are the reason the value is recorded at all."""

    for relative in ("l/logit.sthlp", "p/probit.sthlp", "p/poisson.sthlp"):
        text = (BASE / relative).read_text(encoding="utf-8", errors="ignore")
        assert "{cmd:e(title)}" in text, relative


# --------------------------------------------------------------- the spread
def test_the_three_titles_are_distinct(frame) -> None:
    """Guard: a shared constant would satisfy every per-command test above."""

    titles = {BUILDERS[command](frame).e["title"] for command in BUILDERS}

    assert len(titles) == 3