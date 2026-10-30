"""The ``e(marginsok)``/``e(marginsnotok)`` pair, per command, read from each ado.

Round 223 added the pair for ``logit``/``probit`` and round 224 made ``margins`` act on
it. This file completes the survey -- the remaining commands that set either macro:

============  =========================  ==========================  ====================
command       ``marginsok``              ``marginsnotok``             source
============  =========================  ==========================  ====================
``regress``   ``XB default``             (not set)                    ``regress.ado:174``
``tobit``     ``default XB Pr(passthru)  ``""`` (set EMPTY)           ``tobit.ado:145-149``
              ``E(passthru)``
              ``YStar(passthru)``
``qreg``      ``""`` (cleared)           ``stdp stddp residuals``     ``qreg.ado:190,287``
============  =========================  ==========================  ====================

The ``(passthru)`` suffix on tobit's three probability/expectation entries is Stata's own
notation for "this prediction type takes an argument" -- the censoring point -- and is
transcribed rather than normalised away.

``tobit``'s empty ``marginsnotok`` is recorded as an EMPTY STRING, not omitted. The key's
presence is itself information: the command spoke about the question and forbade nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


def _frame(n: int = 300, seed: int = 5400):
    rng = np.random.default_rng(seed)
    covariate = rng.normal(size=n)
    return pd.DataFrame(
        {
            "y": 1.0 + 0.8 * covariate + rng.normal(size=n),
            "x": covariate,
        }
    )


@pytest.fixture(scope="module")
def regress_fit():
    return ws.OLS.from_formula("y ~ x", _frame()).fit()


@pytest.fixture(scope="module")
def qreg_fit():
    return ws.QuantileRegression.from_formula("y ~ x", _frame()).fit()


@pytest.fixture(scope="module")
def tobit_fit():
    frame = _frame()
    frame = frame.assign(y=frame["y"].clip(lower=0.0))
    return ws.Tobit.from_formula("y ~ x", frame, lower=0).fit()


# ------------------------------------------------------------------- regress
def test_regress_blesses_xb_and_default(regress_fit) -> None:
    """``regress.ado:174``: ``ereturn local marginsok "XB default"``."""

    assert regress_fit.e["marginsok"] == "XB default"


def test_regress_forbids_nothing(regress_fit) -> None:
    """``regress`` sets no ``marginsnotok`` at all."""

    assert "marginsnotok" not in regress_fit.e


# --------------------------------------------------------------------- tobit
def test_tobit_blesses_four_types_three_with_passthru(tobit_fit) -> None:
    """``tobit.ado:145-148``, transcribed with its ``(passthru)`` markers."""

    assert tobit_fit.e["marginsok"] == (
        "default XB Pr(passthru) E(passthru) YStar(passthru)"
    )


def test_tobit_forbids_nothing_but_says_so(tobit_fit) -> None:
    """``tobit.ado:149`` sets the macro EMPTY rather than leaving it unset.

    The distinction matters: an empty macro is the command asserting "nothing is
    forbidden", whereas an absent key means the command said nothing.
    """

    assert "marginsnotok" in tobit_fit.e
    assert tobit_fit.e["marginsnotok"] == ""


def test_tobit_censoring_points_take_arguments(tobit_fit) -> None:
    """The ``(passthru)`` entries are the ones ``margins`` must pass an argument to."""

    entries = str(tobit_fit.e["marginsok"]).split()
    passthru = [entry for entry in entries if "(passthru)" in entry]

    assert len(passthru) == 3


# ---------------------------------------------------------------------- qreg
def test_qreg_blesses_no_type(qreg_fit) -> None:
    """``qreg.ado:190`` clears ``marginsok`` -- the macro is set, but empty."""

    assert qreg_fit.e["marginsok"] == ""


def test_qreg_forbids_three_types(qreg_fit) -> None:
    """``qreg.ado:287``: ``ereturn local marginsnotok stdp stddp residuals``."""

    assert qreg_fit.e["marginsnotok"] == "stdp stddp residuals"


@pytest.mark.parametrize("kind", ["stdp", "stddp", "residuals"])
def test_qreg_margins_refuses_them(qreg_fit, kind: str) -> None:
    """The macro is not decorative -- ``margins`` must act on it."""

    with pytest.raises(ValueError):
        qreg_fit.margins(dydx=["x"], predict_kind=kind)


@pytest.mark.parametrize("kind", ["default", "xb"])
def test_qreg_margins_still_accepts_the_others(qreg_fit, kind: str) -> None:
    """Guard: a check that refused EVERYTHING would pass the previous test."""

    assert qreg_fit.margins(dydx=["x"], predict_kind=kind) is not None


# ----------------------------------------------------------- cross-command
def test_the_three_commands_declare_different_sets(
    regress_fit, qreg_fit, tobit_fit
) -> None:
    """Guard: a shared constant would satisfy the individual assertions.

    The three commands have genuinely different ideas of what ``margins`` may do, which
    is the whole reason the macro is per-command.

    ``marginsnotok`` compares by PRESENCE as well as value, because ``regress`` does not
    set it at all while ``tobit`` sets it to the empty string -- the same text, different
    statements.  Comparing text alone made those two look identical.
    """

    absence = [
        command
        for command, fitted in (
            ("regress", regress_fit),
            ("qreg", qreg_fit),
            ("tobit", tobit_fit),
        )
        if "marginsnotok" not in fitted.e
    ]
    assert absence == ["regress"]

    values = {
        str(qreg_fit.e["marginsnotok"]),
        str(tobit_fit.e["marginsnotok"]),
    }
    assert len(values) == 2

    blessed = {
        str(regress_fit.e["marginsok"]),
        str(qreg_fit.e["marginsok"]),
        str(tobit_fit.e["marginsok"]),
    }
    assert len(blessed) == 3