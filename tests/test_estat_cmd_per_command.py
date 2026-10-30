"""``e(estat_cmd)`` -- present only where the command actually has an ``estat``.

``estat.ado:10-11`` dispatches on this macro::

    if `"`e(estat_cmd)'"' != "" {
        `ver' `e(estat_cmd)' `0'
    }

So the key names the program that answers ``estat <subcommand>``. Which commands publish
it is per-command, and the evidence is uneven:

============  =========================================  =============================
command       evidence                                   value
============  =========================================  =============================
``regress``   ``regress.sthlp:440`` + ``regress_estat.ado``  ``regress_estat``
``logit``     ``logit.ado:267`` (literal)                ``logit_estat``
``probit``    ``probit.ado:259`` (literal)               ``probit_estat``
``poisson``   ``poisson.ado:255`` + ``poisson.sthlp``    ``poisson_estat``
``nbreg``     no ``estat_cmd`` in its help file           absent
``ologit``    no ``estat_cmd`` in its help file           absent
``oprobit``   no ``estat_cmd`` in its help file           absent
============  =========================================  =============================

The absent cases are the point of the second half of this file: a key copied across for
symmetry would make ``estat`` look supported after a command that has no ``estat``
subcommands at all.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import westata as ws

BASE = Path(r"D:\Program Files\StataNow19\ado\base")

PUBLISHED = {
    "regress": "regress_estat",
    "logit": "logit_estat",
    "probit": "probit_estat",
    "poisson": "poisson_estat",
}

#: Commands whose help file does not document ``e(estat_cmd)``.
SILENT = ("nbreg", "ologit", "oprobit")


def _frame(n: int = 300, seed: int = 7200):
    rng = np.random.default_rng(seed)
    covariate = rng.normal(size=n)
    latent = rng.normal(size=n)
    categories = np.digitize(latent, [-0.4, 0.5]) + 1
    return pd.DataFrame(
        {
            "y": 1.0 + 0.8 * covariate + rng.normal(size=n),
            "yb": (rng.uniform(size=n) < 0.5).astype(int),
            "yp": rng.poisson(2.0, size=n),
            "yc": rng.negative_binomial(2, 0.5, size=n),
            "yord": categories,
            "x": covariate,
        }
    )


@pytest.fixture(scope="module")
def frame():
    return _frame()


BUILDERS = {
    "regress": lambda frame: ws.OLS.from_formula("y ~ x", frame).fit(),
    "logit": lambda frame: ws.Logit.from_formula("yb ~ x", frame).fit(),
    "probit": lambda frame: ws.Probit.from_formula("yb ~ x", frame).fit(),
    "poisson": lambda frame: ws.Poisson.from_formula("yp ~ x", frame).fit(),
    "nbreg": lambda frame: ws.NegativeBinomial.from_formula("yc ~ x", frame).fit(),
    "ologit": lambda frame: ws.OrderedLogit.from_formula("yord ~ x", frame).fit(),
    "oprobit": lambda frame: ws.OrderedProbit.from_formula("yord ~ x", frame).fit(),
}


# --------------------------------------------------------------- the values
@pytest.mark.parametrize("command", sorted(PUBLISHED))
def test_the_published_commands_name_their_program(frame, command: str) -> None:
    assert BUILDERS[command](frame).e["estat_cmd"] == PUBLISHED[command]


@pytest.mark.parametrize("command", SILENT)
def test_the_silent_commands_record_nothing(frame, command: str) -> None:
    """No ``estat`` subcommands means no macro -- not an empty one, not a guess."""

    assert "estat_cmd" not in BUILDERS[command](frame).e


# ------------------------------------------------------- the evidence, re-read
def test_every_published_value_names_an_existing_program() -> None:
    """The value is a program name, so the program has to exist."""

    for command, program in PUBLISHED.items():
        assert (BASE / program[0] / f"{program}.ado").exists(), (command, program)


def test_the_silent_commands_document_no_estat_cmd() -> None:
    for command in SILENT:
        path = BASE / command[0] / f"{command}.sthlp"
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert "e(estat_cmd)" not in text, command


def test_poisson_sets_it_with_est_local_not_ereturn() -> None:
    """The spelling Stata uses there (``poisson.ado:255``); pinning it documents the
    variation without changing the resulting macro."""

    text = (BASE / "p/poisson.ado").read_text(encoding="utf-8", errors="ignore")

    assert 'est local estat_cmd "poisson_estat"' in text


def test_the_estat_dispatcher_reads_the_macro() -> None:
    text = (BASE / "e/estat.ado").read_text(encoding="utf-8", errors="ignore")

    assert "e(estat_cmd)" in text


# ------------------------------------------------------------ not collateral
def test_the_information_criteria_are_unaffected(frame) -> None:
    """Guard: the change adds a key, it must not disturb its neighbours."""

    for command in BUILDERS:
        e = BUILDERS[command](frame).e
        assert "aic" in e, command
        assert "bic" in e, command


def test_the_titles_are_still_recorded(frame) -> None:
    """The title work of rounds 241-242 must survive this change."""

    assert BUILDERS["poisson"](frame).e["title"] == "Poisson regression"
    assert BUILDERS["nbreg"](frame).e["title"] == "Negative binomial regression"
    assert BUILDERS["ologit"](frame).e["title"] == "Ordered logistic regression"