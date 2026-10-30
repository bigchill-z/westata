"""``predict, scores`` -- a real option this API cannot express.

Every command that documents ``scores`` says it creates MULTIPLE variables, one per
equation:

* ``nbreg_postestimation.sthlp:172-181`` -- the regression score AND the dispersion
  score;
* ``ologit_postestimation`` -- "The number of score variables created will equal the
  number of outcomes";
* ``tobit`` / ``truncreg`` / ``intreg`` -- the regression equation plus the auxiliary
  parameter's equation.

Our ``predict`` returns ONE array per call, so the shape cannot be honoured. The point
of these tests is that the refusal NAMES the shape problem. A generic "not in the
list" message reads as "this package lacks the feature", which is a different and
misleading claim.

**The spelling is Stata's own and is asymmetric:** ``poisson`` documents ``score``
(singular, one equation); the rest document ``scores``. Unifying the names would move
one of them away from its help file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import westata as ws

BASE = Path(r"D:\Program Files\StataNow19\ado\base")


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(503)
    n = 200
    latent = rng.normal(size=n)
    return pd.DataFrame(
        {
            "y": latent,
            "w": np.maximum(latent, 0.5),
            "lo": np.maximum(latent, 0.5),
            "hi": np.maximum(latent, 0.5),
            "x1": rng.normal(size=n),
            "ycnt": rng.poisson(3.0, size=n),
            "rep": rng.integers(2, 5, size=n),
        }
    )


MULTI_EQUATION = {
    "tobit": lambda f: ws.Tobit.from_formula("y ~ x1", f).fit(),
    "truncreg": lambda f: ws.Truncreg.from_formula("w ~ x1", f, lower=0).fit(),
    "intreg": lambda f: ws.Intreg.from_formula("lo hi ~ x1", f).fit(),
    "ologit": lambda f: ws.OrderedLogit.from_formula("rep ~ x1", f).fit(),
    "nbreg": lambda f: ws.NegativeBinomial.from_formula("ycnt ~ x1", f).fit(),
}


@pytest.mark.parametrize("label", sorted(MULTI_EQUATION))
def test_scores_refusal_names_the_shape_problem(frame, label) -> None:
    """The message must say MULTIPLE variables, not just list what is available."""

    result = MULTI_EQUATION[label](frame)

    with pytest.raises(Exception) as caught:
        result.predict(kind="scores")

    message = str(caught.value).lower()
    assert "multiple variables" in message, message
    assert "one per equation" in message or "per outcome" in message or "one array" in message


@pytest.mark.parametrize("label", sorted(MULTI_EQUATION))
def test_scores_is_not_registered(frame, label) -> None:
    """A registered type that raises is the defect the registry test catches."""

    from westata.postestimation.predict import SUPPORTED_TYPES

    name = MULTI_EQUATION[label](frame).model_name
    declared = SUPPORTED_TYPES.get(name, ())
    assert "scores" not in declared, name


def test_poisson_keeps_the_singular_score(frame) -> None:
    """``poisson`` has ONE equation, so its ``score`` is a single variable."""

    result = ws.Poisson.from_formula("ycnt ~ x1", frame).fit()
    score = np.asarray(result.predict(kind="score"))

    assert np.all(np.isfinite(score))
    assert score.shape == (len(frame),)


def test_poisson_does_not_offer_the_plural_spelling(frame) -> None:
    """``poisson``'s help documents ``score``; ``scores`` would be a different key."""

    from westata.postestimation.predict import SUPPORTED_TYPES

    assert "score" in SUPPORTED_TYPES["poisson"]
    assert "scores" not in SUPPORTED_TYPES["poisson"]


def _help_text(relative: str) -> str:
    return (BASE / relative).read_text(encoding="utf-8", errors="ignore")


def test_the_spelling_asymmetry_is_real() -> None:
    """Pin it against the help files so a "tidy up" cannot silently rename one."""

    poisson = _help_text("p/poisson_postestimation.sthlp")
    assert "{opt score}" in poisson
    assert "{opt scores}" not in poisson

    for relative in (
        "n/nbreg_postestimation.sthlp",
        "o/ologit_postestimation.sthlp",
        "t/tobit_postestimation.sthlp",
        "t/truncreg_postestimation.sthlp",
        "i/intreg_postestimation.sthlp",
    ):
        plural = _help_text(relative)
        assert "{opt scores}" in plural, relative
        assert "{opt score}" not in plural, relative


def test_ologit_scores_matches_the_outcome_count() -> None:
    """``ologit``'s help ties the variable count to the number of outcomes.

    Recording the rule explains WHY it cannot be a single array, rather than leaving
    the refusal as an unexplained limitation.
    """

    help_text = _help_text("o/ologit_postestimation.sthlp")

    assert "number of outcomes" in help_text.lower() or "number of outcomes" in help_text


def test_alpha_types_belong_to_gnbreg_not_nbreg() -> None:
    """``alpha``/``lnalpha``/``stdplna`` are ``gnbreg``-only predictions.

    ``nbreg_postestimation.sthlp``: "{opt alpha}, {opt lnalpha}, and {opt stdplna} are
    relevant after gnbreg estimation only".  Plain ``nbreg`` fits a CONSTANT alpha, so
    a per-observation ``alpha`` column would be a column of one repeated number --
    and offering it would suggest the dispersion varies across observations when the
    model says it does not.

    This package does not implement ``gnbreg``, so the three types are correctly
    absent.  Recording it prevents a future "nbreg is missing alpha" addition.
    """

    from westata.postestimation.predict import SUPPORTED_TYPES

    nbreg_types = SUPPORTED_TYPES["nbreg"]
    for name in ("alpha", "lnalpha", "stdplna"):
        assert name not in nbreg_types, name
    assert "gnbreg" not in SUPPORTED_TYPES

    help_text = _help_text("n/nbreg_postestimation.sthlp")
    assert "after {cmd:gnbreg}" in help_text or "after gnbreg" in help_text.lower()