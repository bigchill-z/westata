"""``margins`` must honour ``e(marginsok)``/``e(marginsnotok)``.

These are real Stata macros that estimators set to declare which prediction types
``margins`` may use after them:

    areg.ado:255       est local marginsnotok Residuals SCore D XBD
    arch.ado:1531      est local marginsnotok Residuals YResiduals
    arima.ado:727      est local marginsnotok stdp mse Residuals YResiduals
    logit.ado:254-264  est local marginsnotok stdp DBeta ... SCore
                       est local marginsok    default Pr
    probit.ado         the same five

Dozens of commands each forbid a different subset, so the pair is genuinely read by
``margins`` rather than recorded for show.

**This module did not read them.** It mentioned ``e(marginsok)`` in one comment and
accepted ``stdp``/``residuals`` after ``logit`` -- exactly the types the command
declares disallowed. Round 224 wired the check in, and the tests below assert both
directions: the allowed set works, the forbidden set is refused, and the refusal names
the macro so the message is actionable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


def _frame(n: int = 300, seed: int = 5300):
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
def logit_fit():
    return ws.Logit.from_formula("yb ~ x", _frame()).fit()


@pytest.fixture(scope="module")
def probit_fit():
    return ws.Probit.from_formula("yb ~ x", _frame()).fit()


# ------------------------------------------------------------- the allowed set
@pytest.mark.parametrize("fixture_name", ["logit_fit", "probit_fit"])
def test_the_default_keyword_resolves_to_the_command_default(fixture_name, request) -> None:
    """``default`` is an ``e(marginsok)`` KEYWORD, not a prediction type.

    Passing it through literally made ``margins(predict_kind="default")`` fail even
    though Stata accepts exactly that word.
    """

    fitted = request.getfixturevalue(fixture_name)

    produced = fitted.margins(dydx=["x"], predict_kind="default")

    assert produced is not None


@pytest.mark.parametrize("fixture_name", ["logit_fit", "probit_fit"])
def test_pr_is_accepted(fixture_name, request) -> None:
    """``e(marginsok)`` is ``default Pr`` for this family."""

    fitted = request.getfixturevalue(fixture_name)

    produced = fitted.margins(dydx=["x"], predict_kind="pr")

    assert produced is not None


# ----------------------------------------------------------- the forbidden set
@pytest.mark.parametrize("fixture_name", ["logit_fit", "probit_fit"])
@pytest.mark.parametrize("kind", ["stdp", "residuals", "score", "hat"])
def test_forbidden_types_are_refused(fixture_name, kind: str, request) -> None:
    """``logit.ado:254-263`` names these explicitly in ``e(marginsnotok)``."""

    fitted = request.getfixturevalue(fixture_name)

    with pytest.raises(ValueError):
        fitted.margins(dydx=["x"], predict_kind=kind)


@pytest.mark.parametrize("fixture_name", ["logit_fit", "probit_fit"])
def test_the_refusal_names_the_macro(fixture_name, request) -> None:
    """An actionable message says WHICH macro forbade it and what is allowed."""

    fitted = request.getfixturevalue(fixture_name)

    with pytest.raises(ValueError) as caught:
        fitted.margins(dydx=["x"], predict_kind="stdp")

    message = str(caught.value)
    assert "marginsnotok" in message
    assert "marginsok" in message


def test_the_forbidden_list_matches_the_macro(logit_fit) -> None:
    """Guard on the direction: everything in the macro is refused, nothing else is.

    A check that refused EVERYTHING would pass the tests above; this one would not.
    """

    forbidden = str(logit_fit.e["marginsnotok"]).lower().split()
    allowed = {entry.lower() for entry in str(logit_fit.e["marginsok"]).split()}
    allowed.discard("default")

    # Every entry in the macro is refused.
    for kind in forbidden:
        with pytest.raises(ValueError):
            logit_fit.margins(dydx=["x"], predict_kind=kind)
    # And the allowed ones are not.
    for kind in allowed:
        assert logit_fit.margins(dydx=["x"], predict_kind=kind) is not None


def test_the_default_call_still_works(logit_fit) -> None:
    """The macro check must not block the ordinary call.

    ``e(marginsnotok)`` is per-command and read defensively, so a fit keeps working when
    no ``predict_kind`` is requested at all.
    """

    assert logit_fit.margins(dydx=["x"]) is not None


def test_a_model_without_the_macro_is_not_blocked() -> None:
    """``regress`` also sets the macro, so use a fit whose command does not.

    ``ols``'s ``e()`` carries no ``marginsnotok``; the check must treat an absent macro
    as "nothing forbidden" rather than raising on the empty string.
    """

    frame = _frame()
    frame = frame.assign(y=1.0 + 0.8 * frame["x"])
    fitted = ws.OLS.from_formula("y ~ x", frame).fit()

    assert "marginsnotok" not in fitted.e
    assert fitted.margins(dydx=["x"]) is not None