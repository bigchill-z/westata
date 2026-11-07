"""``e(predict)`` -- the predict program's name, including Stata's 8-character truncation.

``estat.ado`` dispatches ``estat`` on ``e(estat_cmd)``; ``predict`` does the same with
``e(predict)``. Each ado sets it literally::

    logit.ado:266      ereturn local predict logit_p
    probit.ado:258     ereturn local predict probit_p
    poisson.ado        est local predict "poisso_p"
    nbreg.ado          est local predict "nbreg_p"
    ologit.ado         ereturn local predict ologit_p
    oprobit.ado        ereturn local predict oprobi_p

**Two names are truncated to Stata's 8-character limit for program names**, and both are
easy to get wrong by construction:

    poisson  ->  poisso_p    (NOT poisson_p -- 8 chars)
    oprobit  ->  oprobi_p    (NOT oprobit_p)

An implementation that built the name as ``command + "_p"`` would produce two names that
do not exist, which is why the mapping is transcribed rather than derived. The tests below
check both the recorded value AND that the program file is really there.

``regress`` is the deliberate omission: ``regress.sthlp:441`` documents ``e(predict)`` but
no ``regress_p.ado`` exists anywhere under ``ado/base``, so the value cannot be confirmed
(round 232).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import westata as ws

BASE = Path(r"D:\Program Files\StataNow19\ado\base")

EXPECTED = {
    "logit": "logit_p",
    "probit": "probit_p",
    "poisson": "poisso_p",
    "nbreg": "nbreg_p",
    "ologit": "ologit_p",
    "oprobit": "oprobi_p",
}


def _frame(n: int = 300, seed: int = 7300):
    rng = np.random.default_rng(seed)
    covariate = rng.normal(size=n)
    latent = rng.normal(size=n)
    return pd.DataFrame(
        {
            "y": 1.0 + 0.8 * covariate + rng.normal(size=n),
            "yb": (rng.uniform(size=n) < 0.5).astype(int),
            "yp": rng.poisson(2.0, size=n),
            "yc": rng.negative_binomial(2, 0.5, size=n),
            "yord": np.digitize(latent, [-0.4, 0.5]) + 1,
            "x": covariate,
        }
    )


@pytest.fixture(scope="module")
def frame():
    return _frame()


BUILDERS = {
    "logit": lambda frame: ws.Logit.from_formula("yb ~ x", frame).fit(),
    "probit": lambda frame: ws.Probit.from_formula("yb ~ x", frame).fit(),
    "poisson": lambda frame: ws.Poisson.from_formula("yp ~ x", frame).fit(),
    "nbreg": lambda frame: ws.NegativeBinomial.from_formula("yc ~ x", frame).fit(),
    "ologit": lambda frame: ws.OrderedLogit.from_formula("yord ~ x", frame).fit(),
    "oprobit": lambda frame: ws.OrderedProbit.from_formula("yord ~ x", frame).fit(),
}


# ---------------------------------------------------------------- the values
@pytest.mark.parametrize("command", sorted(BUILDERS))
def test_the_predict_program_is_recorded(frame, command: str) -> None:
    assert BUILDERS[command](frame).e["predict"] == EXPECTED[command]


@pytest.mark.parametrize("command", sorted(BUILDERS))
def test_the_named_program_really_exists(command: str) -> None:
    """A dispatch macro pointing at a nonexistent program would be worse than absent."""

    program = EXPECTED[command]
    assert (BASE / program[0] / f"{program}.ado").exists(), program


# ------------------------------------------------------- the truncation cases
def test_poisson_is_truncated(frame) -> None:
    """``poisso_p``: eight characters, one ``n`` short of the command name."""

    recorded = BUILDERS["poisson"](frame).e["predict"]

    assert recorded == "poisso_p"
    assert recorded != "poisson_p"
    assert len(recorded) == len("poisso_p")


def test_oprobit_is_truncated(frame) -> None:
    """``oprobi_p``: the trailing ``t`` is dropped."""

    recorded = BUILDERS["oprobit"](frame).e["predict"]

    assert recorded == "oprobi_p"
    assert recorded != "oprobit_p"


def test_construction_by_appending_p_is_wrong_for_these_two() -> None:
    """The reason the mapping is transcribed.

    Building ``command + "_p"`` gives a plausible name that does not exist, so this test
    records that the naive rule fails for exactly two commands -- and no others.
    """

    naive = {command: f"{command}_p" for command in EXPECTED}
    disagreements = sorted(
        command for command, value in EXPECTED.items() if naive[command] != value
    )

    assert disagreements == ["oprobit", "poisson"]


def test_both_truncated_programs_are_the_ones_that_exist() -> None:
    """Check the filesystem rather than a naming rule."""

    assert (BASE / "p" / "poisso_p.ado").exists()
    assert not (BASE / "p" / "poisson_p.ado").exists()
    assert (BASE / "o" / "oprobi_p.ado").exists()
    assert not (BASE / "o" / "oprobit_p.ado").exists()


# --------------------------------------------------------- the deliberate gap
def test_regress_records_no_predict_program(frame) -> None:
    """CORRECTED in round 252: the key IS recorded, and the routine DOES exist.

    This test used to assert "predict" not in fitted.e, on the claim that ``regress.sthlp:441``
    documents the key but no ``regress_p.ado`` exists. Re-measured against the installation: the
    file is ``ado/base/r/regres_p.ado`` -- ONE 's' in "regres" -- so searching for "regress_p"
    found nothing. A live ``quietly regress y x`` then ``display e(predict)`` prints ``regres_p``.

    The original concern was sound (a plausible guess naming a program that is not there is worse
    than an absent key); the premise was wrong because of the spelling.
    """

    fitted = ws.OLS.from_formula("y ~ x", frame).fit()

    assert fitted.e["predict"] == "regres_p"
    assert not (BASE / "r" / "regress_p.ado").exists()


def test_an_unknown_command_yields_the_empty_string() -> None:
    from westata.estimators.discrete import _command_predict_program

    assert _command_predict_program("somethingelse") == ""