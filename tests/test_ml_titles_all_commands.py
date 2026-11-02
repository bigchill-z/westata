"""``e(title)`` for the remaining ML commands, through BOTH assignment routes.

Each ado supplies the title one of two ways, and the difference matters for how the value
can be found:

**Direct** -- a literal ``ereturn local title``::

    ologit.ado:270    ereturn local title "Ordered logistic regression"
    oprobit.ado:267   ereturn local title "Ordered probit regression"

**Indirect** -- a local handed to ``ml`` as a ``title()`` option::

    poisson.ado:241   ... `mlopts' title(Poisson regression) ...
    nbreg.ado:352     */ title(`title') diparm(...)
    nbreg.ado:71      local title "Negative binomial regression"

Round 233 searched ``poisson.ado`` for ``ereturn local title``, found nothing, and
concluded poisson had no title. That was the indirect route being invisible to grep.
Round 241 fixed poisson; this file covers the rest and pins BOTH routes so neither reading
can be mistaken for the other again.
"""

from __future__ import annotations

import re

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.results.summary import render_summary

BASE = Path(r"D:\Program Files\StataNow19\ado\base")

EXPECTED = {
    "ologit": "Ordered logistic regression",
    "oprobit": "Ordered probit regression",
    "nbreg": "Negative binomial regression",
}

#: Ados that set the macro with a literal ``ereturn local title``.
DIRECT = {
    "ologit": ("o/ologit.ado", 'ereturn local title "Ordered logistic regression"'),
    "oprobit": ("o/oprobit.ado", 'ereturn local title "Ordered probit regression"'),
}

#: Ados that pass ``title()`` to ``ml`` instead.
INDIRECT = {
    "nbreg": ("n/nbreg.ado", 'local title  "Negative binomial regression"'),
}


def _frame(n: int = 300, seed: int = 7100):
    rng = np.random.default_rng(seed)
    covariate = rng.normal(size=n)
    latent = rng.normal(size=n)
    categories = np.digitize(latent, [-0.4, 0.5]) + 1
    return pd.DataFrame(
        {
            "yord": categories,
            "ycnt": rng.negative_binomial(2, 0.5, size=n),
            "x": covariate,
        }
    )


@pytest.fixture(scope="module")
def frame():
    return _frame()


BUILDERS = {
    "ologit": lambda frame: ws.OrderedLogit.from_formula("yord ~ x", frame).fit(),
    "oprobit": lambda frame: ws.OrderedProbit.from_formula("yord ~ x", frame).fit(),
    "nbreg": lambda frame: ws.NegativeBinomial.from_formula("ycnt ~ x", frame).fit(),
}


# ---------------------------------------------------------------- the values
@pytest.mark.parametrize("command", sorted(BUILDERS))
def test_the_title_matches_the_ado(frame, command: str) -> None:
    assert BUILDERS[command](frame).e["title"] == EXPECTED[command]


@pytest.mark.parametrize("command", sorted(BUILDERS))
def test_the_stata_banner_uses_that_title(frame, command: str) -> None:
    fitted = BUILDERS[command](frame)
    first_line = render_summary(fitted, style="stata").splitlines()[0].strip()
    # The ML header is a two-column block, so the banner shares its first line with
    # "Number of obs" (measured against Stata). Split on the run of spaces between the
    # columns instead of assuming the banner owns the whole line.
    banner = re.split(r"\s{2,}", first_line)[0].strip()

    assert banner == EXPECTED[command]


def test_the_titles_are_all_distinct(frame) -> None:
    """Guard: a shared constant would satisfy every per-command test above."""

    titles = {BUILDERS[command](frame).e["title"] for command in BUILDERS}

    assert len(titles) == 3


# ------------------------------------------------------- the two routes, read
@pytest.mark.parametrize("command", sorted(DIRECT))
def test_the_direct_route_has_the_literal(command: str) -> None:
    relative, literal = DIRECT[command]
    text = (BASE / relative).read_text(encoding="utf-8", errors="ignore")

    assert literal in text, relative


@pytest.mark.parametrize("command", sorted(INDIRECT))
def test_the_indirect_route_has_the_local_but_no_ereturn(command: str) -> None:
    """The nbreg shape: the literal is a ``local``, and the macro comes from ``ml``."""

    relative, literal = INDIRECT[command]
    text = (BASE / relative).read_text(encoding="utf-8", errors="ignore")

    assert literal in text, relative
    assert "ereturn local title" not in text, relative


def test_the_indirect_command_passes_it_to_ml() -> None:
    """``nbreg.ado:352`` hands ``title(`title')`` to the ML engine."""

    text = (BASE / "n/nbreg.ado").read_text(encoding="utf-8", errors="ignore")

    assert "title(`title')" in text or "title(`title')" in text.replace("\t", " ")


def test_the_direct_commands_do_not_pass_a_title_option() -> None:
    """Guard on the other side: ologit/oprobit use ``ereturn``, not the ML option."""

    for relative, _ in DIRECT.values():
        text = (BASE / relative).read_text(encoding="utf-8", errors="ignore")
        assert "ereturn local title" in text


# ----------------------------------------------------------- what stays absent
def test_an_unknown_command_gets_the_empty_string() -> None:
    """``_command_title`` returns ``""`` rather than guessing a plausible name."""

    from westata.estimators.discrete import _command_title

    assert _command_title("somethingelse") == ""