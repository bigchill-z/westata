"""``e(vce)`` / ``e(vcetype)`` across families -- and the ML default is NOT ``ols``.

Round 245 established the pair for ``regress`` and the rule that the two macros are
different strings (lowercase token vs capitalised display title). This file extends the
work to the ML families, where the non-robust DEFAULT is different:

=================  ===========  =============  ==================================
family             ``e(vce)``   ``e(vcetype)``  evidence
=================  ===========  =============  ==================================
linear (``regress``)  ``ols``      ``Ols``        ``regress.ado:68`` defaults to ``ols``
ML (``logit`` etc.)   ``oim``      ``OIM``        ``arch.ado:1474``/``arima.ado:741``
=================  ===========  =============  ==================================

``arch.ado:1470-1474`` shows the two branches side by side, which is what makes the
contrast readable in one place::

    if "`e(vcetype)'" == "Robust" {
        ...
    }
    else {
        est local vcetype "OIM"
    }

The help files document the pair for all four ML commands
(``logit.sthlp:271-272`` and the ``poisson``/``ologit``/``nbreg`` equivalents), so a
missing ``oim`` there was a gap rather than a deliberate omission.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import westata as ws

BASE = Path(r"D:\Program Files\StataNow19\ado\base")

ML_COMMANDS = ("logit", "probit", "poisson", "nbreg", "ologit", "oprobit")


def _frame(n: int = 250, seed: int = 7500):
    rng = np.random.default_rng(seed)
    covariate = rng.normal(size=n)
    latent = rng.normal(size=n)
    return pd.DataFrame(
        {
            "y": 1.0 + 0.8 * covariate + rng.normal(size=n),
            "yb": (rng.uniform(size=n) < 0.5).astype(int),
            "ycnt": rng.poisson(2.0, size=n),
            "ynb": rng.negative_binomial(2, 0.5, size=n),
            "yord": np.digitize(latent, [-0.4, 0.5]) + 1,
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
    "poisson": lambda frame: ws.Poisson.from_formula("ycnt ~ x", frame).fit(),
    "nbreg": lambda frame: ws.NegativeBinomial.from_formula("ynb ~ x", frame).fit(),
    "ologit": lambda frame: ws.OrderedLogit.from_formula("yord ~ x", frame).fit(),
    "oprobit": lambda frame: ws.OrderedProbit.from_formula("yord ~ x", frame).fit(),
}


# ------------------------------------------------------------- the ML default
@pytest.mark.parametrize("command", ML_COMMANDS)
def test_the_ml_default_token_is_oim(frame, command: str) -> None:
    assert BUILDERS[command](frame).e["vce"] == "oim"


@pytest.mark.parametrize("command", ML_COMMANDS)
def test_the_ml_default_title_is_blank(frame, command: str) -> None:
    """EMPTY, not "OIM".

    ``arch.ado:1473-1474`` does set ``vcetype "OIM"``, but only when ``e(vce)=="oim"`` --
    and no ML command defaults to that: measured on StataNow 19.5, ``logit``, ``probit``,
    ``poisson`` and ``nbreg`` all store ``e(vce)="oim"`` with ``e(vcetype)=""``, while
    ``arch`` itself defaults to ``e(vce)="opg"``/``e(vcetype)="OPG"``.

    The blank is load-bearing: e(vcetype) is what the table renderer uses to decide
    between the plain and the robust layout, so a fabricated "OIM" re-cased every ML
    header into the robust form.
    """

    assert BUILDERS[command](frame).e["vcetype"] == ""


def test_the_ml_default_is_not_ols(frame) -> None:
    """The distinguishing test: an implementation that reused the linear rule fails."""

    assert BUILDERS["logit"](frame).e["vce"] != "ols"
    assert BUILDERS["logit"](frame).e["vcetype"] != "Ols"


# ------------------------------------------------------- the linear contrast
def test_regress_keeps_the_ols_default(frame) -> None:
    """Guard on the other side: the ML change must not leak into the linear family."""

    fitted = BUILDERS["regress"](frame)
    assert fitted.e["vce"] == "ols"
    # EMPTY, not "Ols": ``regress y s`` leaves e(vcetype) blank in Stata.
    assert fitted.e["vcetype"] == ""


def test_the_two_families_disagree(frame) -> None:
    """A single shared constant cannot satisfy both families."""

    linear = BUILDERS["regress"](frame).e
    ml = BUILDERS["logit"](frame).e

    assert linear["vce"] != ml["vce"]        # "ols" vs "oim"
    # Both leave e(vcetype) EMPTY, and that is not a collision: the two families are told
    # apart by e(vce) (and by the command's own header branch), not by this macro. The
    # old assertion required them to differ, which forced a value Stata never stores.
    assert linear["vcetype"] == ml["vcetype"] == ""


# -------------------------------------------------- the pair still differs
@pytest.mark.parametrize("command", ML_COMMANDS)
def test_the_token_and_the_title_differ(frame, command: str) -> None:
    """The round-245 rule holds for every family: token != title."""

    e = BUILDERS[command](frame).e

    assert e["vce"] != e["vcetype"]


@pytest.mark.parametrize("command", ["logit", "poisson", "ologit"])
def test_the_robust_case_still_capitalises(frame, command: str) -> None:
    """A robust request must move both macros to the robust spelling."""

    if command == "logit":
        fitted = ws.Logit.from_formula("yb ~ x", frame).fit(cov_type="HC1")
    elif command == "poisson":
        fitted = ws.Poisson.from_formula("ycnt ~ x", frame).fit(cov_type="HC1")
    else:
        fitted = ws.OrderedLogit.from_formula("yord ~ x", frame).fit(cov_type="HC1")

    # ``cov_type="HC1"`` is Stata's ``vce(robust)``, and Stata stores exactly
    # "robust"/"Robust" for it -- there is no ``vce(hc1)`` option to name. Only the
    # hc2/hc3 estimators carry their option name into e(vce) and get a two-word vcetype.
    assert fitted.e["vce"] == "robust"
    assert fitted.e["vcetype"] == "Robust"


# ------------------------------------------------------------ the evidence
def test_the_help_files_document_the_pair() -> None:
    for command in ML_COMMANDS:
        text = (BASE / command[0] / f"{command}.sthlp").read_text(
            encoding="utf-8", errors="ignore"
        )
        assert "{cmd:e(vce)}" in text, command
        assert "{cmd:e(vcetype)}" in text, command


def test_the_oim_literal_is_in_the_ados() -> None:
    """The capitalised spelling comes from Stata's own source."""

    for relative in ("a/arch.ado", "a/arima.ado"):
        text = (BASE / relative).read_text(encoding="utf-8", errors="ignore")
        assert 'est local vcetype "OIM"' in text, relative


def test_the_robust_branch_is_visible_in_the_same_file() -> None:
    """``arch.ado`` branches on both, which is what makes the contrast legible."""

    text = (BASE / "a/arch.ado").read_text(encoding="utf-8", errors="ignore")

    assert '"Robust"' in text
    assert '"OIM"' in text