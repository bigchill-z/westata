"""The ``ereturn local`` block ``logit``/``probit`` write, transcribed from their ado.

A Stata script that discovers what an estimator supports reads ``e(predict)`` and
``e(estat_cmd)`` -- those name the ``_p`` and ``_estat`` programs the command dispatches
to. ``e(marginsok)``/``e(marginsnotok)`` say which prediction types ``margins`` accepts.
None of them are in the help files' stored-results block, because the help documents the
ESTIMATES, not the plumbing the ado sets with ``ereturn local``.

They are read from the ado instead:

``logit.ado:252-268``

    ereturn local title "Logistic regression"
    ereturn local marginsnotok  stdp DBeta DEviance DX2 DDeviance Hat Number \\
                                Residuals RStandard SCore
    ereturn local marginsok default Pr
    ereturn hidden local marginsderiv default Pr
    ereturn local predict logit_p
    ereturn local estat_cmd logit_estat
    ereturn local cmd logit

``probit.ado:252-259`` sets the same five with ``Probit regression`` / ``probit_p`` /
``probit_estat``.

``poisson`` is deliberately NOT covered: its ``ereturn local`` block was not found under
``ado/base/p/`` (it comes from the shared ML machinery), so its values were left alone
rather than guessed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

#: ``logit.ado:254-263`` / ``probit.ado`` -- the same list for both commands.
MARGINSNOTOK = (
    "stdp DBeta DEviance DX2 DDeviance Hat Number Residuals RStandard SCore"
)


def _frame(n: int = 300, seed: int = 5200):
    rng = np.random.default_rng(seed)
    covariate = rng.normal(size=n)
    latent = 0.4 + 0.9 * covariate
    return pd.DataFrame(
        {
            "yb": (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-latent))).astype(int),
            "x": covariate,
        }
    )


@pytest.fixture(scope="module")
def logit_fit():
    return ws.Logit.from_formula("yb ~ x", _frame()).fit()


@pytest.fixture(scope="module")
def probit_fit():
    return ws.Probit.from_formula("yb ~ x", _frame()).fit()


# ------------------------------------------------------------------- the values
def test_logit_titles_itself_the_way_stata_does(logit_fit) -> None:
    """``logit.ado:253``."""
    assert logit_fit.e["title"] == "Logistic regression"


def test_probit_titles_itself_the_way_stata_does(probit_fit) -> None:
    """``probit.ado:252``."""
    assert probit_fit.e["title"] == "Probit regression"


def test_logit_names_its_predict_program(logit_fit) -> None:
    """``logit.ado:266``: ``ereturn local predict logit_p``.

    A ported script reads this to know which program answers ``predict``.
    """

    assert logit_fit.e["predict"] == "logit_p"


def test_probit_names_its_predict_program(probit_fit) -> None:
    assert probit_fit.e["predict"] == "probit_p"


def test_logit_names_its_estat_program(logit_fit) -> None:
    """``logit.ado:267``: ``ereturn local estat_cmd logit_estat``."""
    assert logit_fit.e["estat_cmd"] == "logit_estat"


def test_probit_names_its_estat_program(probit_fit) -> None:
    assert probit_fit.e["estat_cmd"] == "probit_estat"


@pytest.mark.parametrize("fixture_name", ["logit_fit", "probit_fit"])
def test_marginsok_lists_default_and_pr(fixture_name, request) -> None:
    """``ereturn local marginsok default Pr`` -- two entries, both meaningful."""

    e = request.getfixturevalue(fixture_name).e
    entries = str(e["marginsok"]).split()

    assert entries == ["default", "Pr"]


@pytest.mark.parametrize("fixture_name", ["logit_fit", "probit_fit"])
def test_marginsnotok_carries_the_full_list(fixture_name, request) -> None:
    """Ten prediction types ``margins`` must refuse after this family.

    The list is long and easy to truncate, so it is compared to the ado's text rather
    than to a count.
    """

    e = request.getfixturevalue(fixture_name).e

    assert str(e["marginsnotok"]) == MARGINSNOTOK


@pytest.mark.parametrize("fixture_name", ["logit_fit", "probit_fit"])
def test_the_offset_is_empty_because_none_was_specified(fixture_name, request) -> None:
    """``logit.ado:252``: ``ereturn local offset1`` -- an empty macro when unused."""

    e = request.getfixturevalue(fixture_name).e

    assert str(e["offset"]) == ""


# --------------------------------------------------------- the two disagree
def test_the_two_commands_name_different_programs(logit_fit, probit_fit) -> None:
    """Guard: a shared constant would satisfy every test above.

    ``logit_p`` and ``probit_p`` answer different prediction semantics, so the keys must
    not agree across the two estimates.
    """

    assert logit_fit.e["predict"] != probit_fit.e["predict"]
    assert logit_fit.e["estat_cmd"] != probit_fit.e["estat_cmd"]
    assert logit_fit.e["title"] != probit_fit.e["title"]


def test_poisson_does_not_get_the_logit_block() -> None:
    """The values were only transcribed for these two; poisson's come from elsewhere.

    A key present with a WRONG value is worse than absent, so poisson keeps whatever the
    shared machinery gave it -- which, at present, is nothing for the ``margins`` pair.

    Two keys that used to be asserted absent are NOT any more, each for a documented
    reason:

    * ``title`` -- round 241 found ``poisson.ado:241`` passes ``title(Poisson
      regression)`` to ``ml``, so the macro exists via the indirect route even though the
      ado has no ``ereturn local title``;
    * ``estat_cmd`` -- round 243 found ``poisson.ado:255`` sets
      ``"poisson_estat"`` and ``poisson.sthlp`` documents the key.
    """

    rng = np.random.default_rng(5202)
    frame = pd.DataFrame(
        {"ycnt": rng.poisson(2.0, size=250), "x": rng.normal(size=250)}
    )
    fitted = ws.Poisson.from_formula("ycnt ~ x", frame).fit()

    assert fitted.e["title"] == "Poisson regression"
    assert fitted.e["estat_cmd"] == "poisson_estat"
    # The ``margins`` pair is genuinely absent for poisson -- poisson.ado has neither.
    assert "marginsok" not in fitted.e
    assert "marginsnotok" not in fitted.e