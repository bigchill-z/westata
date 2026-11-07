"""``e(estat_cmd)`` / ``e(predict)``: the dispatch macros, and why one is absent.

``regress.sthlp:440-441`` lists both in the stored-results block:

    {cmd:e(estat_cmd)}  program used to implement estat
    {cmd:e(predict)}    program used to implement predict

They exist so that ``estat`` and ``predict`` can dispatch generically.  ``estat.ado:10-11``
shows the mechanism::

    if `"`e(estat_cmd)'"' != "" {
        `ver' `e(estat_cmd)' `0'
    }

**``e(estat_cmd)`` is recoverable; ``e(predict)`` is not.** ``regress_estat.ado:2`` declares
``program regress_estat, rclass``, so the value is readable source.  No ``regress_p.ado``
exists anywhere under ``ado/base`` -- the prediction program is compiled -- so its value
cannot be confirmed and the key is left ABSENT rather than guessed. An absent key at least
signals "unknown"; a wrong string in a dispatch macro would silently call the wrong program.

``logit``/``probit`` are the contrast: both publish a ``_p`` ado, so both macros are
recorded there with their literal values from ``logit.ado:266-267`` / ``probit.ado:258-259``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


def _frame(n: int = 300, seed: int = 6100):
    rng = np.random.default_rng(seed)
    covariate = rng.normal(size=n)
    latent = 0.4 + 0.9 * covariate
    return pd.DataFrame(
        {
            "y": 1.0 + 0.8 * covariate + rng.normal(size=n),
            "yb": (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-latent))).astype(int),
            "x": covariate,
        }
    )


@pytest.fixture(scope="module")
def regress_fit():
    return ws.OLS.from_formula("y ~ x", _frame()).fit()


@pytest.fixture(scope="module")
def logit_fit():
    return ws.Logit.from_formula("yb ~ x", _frame()).fit()


# ------------------------------------------------------------------ regress
def test_regress_names_its_estat_program(regress_fit) -> None:
    """``regress_estat.ado:2`` declares that program; the macro names it."""

    assert regress_fit.e["estat_cmd"] == "regress_estat"


def test_regress_does_NOT_record_a_predict_program(regress_fit) -> None:
    """CORRECTED in round 252: ``regres_p.ado`` DOES exist, and the value is not a guess.

    The file lives at ``ado/base/r/regres_p.ado`` -- one 's' in "regres" -- so the original
    search for "regress_p" found nothing and the key was left absent. A live
    ``quietly regress y x`` then ``display e(predict)`` prints ``regres_p``.
    """

    assert regress_fit.e["predict"] == "regres_p"


def test_the_regress_estat_program_really_exists() -> None:
    """Read the evidence rather than trusting the transcription."""

    from pathlib import Path

    program = Path(r"D:\Program Files\StataNow19\ado\base\r\regress_estat.ado")
    assert program.exists()
    assert "program regress_estat" in program.read_text(
        encoding="utf-8", errors="ignore"
    )


def test_no_regress_p_program_exists() -> None:
    """The reason ``e(predict)`` is absent -- not an oversight."""

    from pathlib import Path

    assert not Path(r"D:\Program Files\StataNow19\ado\base\r\regress_p.ado").exists()


# ------------------------------------------------------- logit / probit, the contrast
def test_logit_records_both_macros(logit_fit) -> None:
    """``logit.ado:266-267`` sets both, and ``logit_p.ado`` exists to back the first."""

    assert logit_fit.e["predict"] == "logit_p"
    assert logit_fit.e["estat_cmd"] == "logit_estat"


def test_probit_records_both_macros() -> None:
    fitted = ws.Probit.from_formula("yb ~ x", _frame()).fit()

    assert fitted.e["predict"] == "probit_p"
    assert fitted.e["estat_cmd"] == "probit_estat"


def test_the_logit_p_program_exists() -> None:
    from pathlib import Path

    assert Path(r"D:\Program Files\StataNow19\ado\base\l\logit_p.ado").exists()


def test_the_two_commands_dispatch_to_different_programs(logit_fit) -> None:
    """Guard: a shared constant would satisfy every per-command test above."""

    fitted = ws.Probit.from_formula("yb ~ x", _frame()).fit()

    assert logit_fit.e["estat_cmd"] != fitted.e["estat_cmd"]
    assert logit_fit.e["predict"] != fitted.e["predict"]


# ---------------------------------------------------------------- the mechanism
def test_estat_dispatch_reads_the_macro() -> None:
    """``estat.ado:10-11``: the macro is what ``estat`` uses to find its program.

    Pinning the mechanism is what makes ``e(estat_cmd)`` worth recording at all -- it is
    not decoration, it is the lookup key.
    """

    from pathlib import Path

    text = Path(r"D:\Program Files\StataNow19\ado\base\e\estat.ado").read_text(
        encoding="utf-8", errors="ignore"
    )

    assert "e(estat_cmd)" in text


def test_a_missing_macro_means_no_estat(regress_fit) -> None:
    """A command that publishes no ``e(estat_cmd)`` has no ``estat`` subcommands.

    ``regress`` has one; ``qreg`` is the contrast used here -- its ado clears the macro
    (``qreg.ado:192``) rather than naming a program.
    """

    fitted = ws.QuantileRegression.from_formula("y ~ x", _frame()).fit()

    assert "estat_cmd" not in fitted.e or not fitted.e["estat_cmd"]