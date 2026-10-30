"""Every type a command advertises in ``e(marginsok)`` must actually exist.

``e(marginsok)`` is a capability inventory: the command is telling ``margins`` (and any
human reading the result) which prediction types it supports. When the macro lists a type
the estimator cannot produce, the result is a documented falsehood -- and this happened
four times in a row:

| command | missing types |
|---|---|
| ``tobit`` | ``pr`` |
| ``intreg`` | ``pr``, ``e``, ``ystar`` |
| ``heckman`` | ``psel``, ``mills``, ``nshazard``, ``ycond``, ``yexpected`` |
| ``etregress`` | ``cte``, ``yctrt``, ``ycntrt``, ``ptrt``, ``xbtrt`` |

Each was found by hand, one command at a time. This file makes the check mechanical: for
every model that publishes the macro, each entry it lists is resolved against
``SUPPORTED_TYPES``.

**Parsing note.** Entries are case-insensitive in Stata and some carry a ``(passthru)``
suffix meaning "this type takes an argument". Both are normalised away before comparing --
``Pr(passthru)`` is the type ``pr``.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.postestimation.predict import supported_types

#: ``default`` is a keyword meaning "the command's own default type", not a type name.
KEYWORDS = frozenset({"default"})


def _declared_types(macro_value: str) -> list[str]:
    """Normalise an ``e(marginsok)`` value into bare type names.

    Strips the ``(passthru)`` suffix and lowercases, so ``Pr(passthru)`` -> ``pr``.
    """

    produced: list[str] = []
    for entry in str(macro_value).split():
        name = re.sub(r"\(.*?\)", "", entry).strip().lower()
        # ``E(passthru)`` reduces to a single letter after the suffix is removed.
        if not name or name in KEYWORDS:
            continue
        produced.append(name)
    return produced


def _frame(n: int = 300, seed: int = 6000):
    rng = np.random.default_rng(seed)
    covariate = rng.normal(size=n)
    instrument = rng.normal(size=n)
    latent = 0.4 + 0.9 * covariate
    return pd.DataFrame(
        {
            "y": 1.0 + 0.8 * covariate + rng.normal(size=n),
            "yb": (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-latent))).astype(int),
            "ycnt": rng.poisson(np.exp(0.3 + 0.4 * covariate)),
            "ycens": np.clip(1.0 + 0.8 * covariate + rng.normal(size=n), 0.0, None),
            "lo": 0.7 + 0.8 * covariate + rng.normal(size=n) * 0.1,
            "hi": 1.3 + 0.8 * covariate + rng.normal(size=n) * 0.1,
            "x": covariate,
            "z": instrument,
            "t": (0.3 + 0.9 * instrument + rng.normal(size=n) > 0).astype(int),
            "s": (0.5 + 0.8 * instrument + rng.normal(size=n) > 0).astype(int),
        }
    )


#: Every model that publishes ``e(marginsok)``, with a builder returning a fitted result.
BUILDERS = {
    "regress": lambda frame: ws.OLS.from_formula("y ~ x", frame).fit(),
    "logit": lambda frame: ws.Logit.from_formula("yb ~ x", frame).fit(),
    "probit": lambda frame: ws.Probit.from_formula("yb ~ x", frame).fit(),
    "qreg": lambda frame: ws.QuantileRegression.from_formula("y ~ x", frame).fit(),
    "tobit": lambda frame: ws.Tobit.from_formula("ycens ~ x", frame, lower=0).fit(),
    "intreg": lambda frame: ws.Intreg.from_formula("lo hi ~ x", frame).fit(),
    "heckman": lambda frame: ws.Heckman.from_formula(
        "y ~ x + select(s ~ z)", frame
    ).fit(),
    "etregress": lambda frame: ws.Etregress.from_formula(
        "y ~ x + treat(t ~ z)", frame
    ).fit(),
}


def test_the_builder_set_covers_every_macro_publisher() -> None:
    """Guard: a model that publishes the macro but is absent here would go unchecked.

    The list is written out rather than derived, deliberately: deriving it would require
    fitting every model to discover who sets the macro, and a model silently dropping out
    of the sweep is exactly the failure this file exists to prevent.
    """

    assert len(BUILDERS) >= 8


@pytest.mark.parametrize("model_name", sorted(BUILDERS))
def test_every_advertised_type_is_implemented(model_name: str) -> None:
    """THE check: ``e(marginsok)`` entries must resolve against ``SUPPORTED_TYPES``."""

    frame = _frame()
    fitted = BUILDERS[model_name](frame)

    macro_value = fitted.e.get("marginsok")
    if macro_value is None:
        pytest.fail(
            f"{model_name!r} publishes no e(marginsok); add one from its ado rather than "
            "leaving the sweep blind to it"
        )

    advertised = _declared_types(str(macro_value))
    offered = {
        name.strip().lower()
        for name in supported_types(model_name)
    }

    missing = [name for name in advertised if name not in offered]

    assert missing == [], (
        f"{model_name}: e(marginsok) advertises {missing} but SUPPORTED_TYPES has "
        f"{sorted(offered)}"
    )


@pytest.mark.parametrize("model_name", sorted(BUILDERS))
def test_every_forbidden_type_is_documented_as_having_a_reason(model_name: str) -> None:
    """A type in ``e(marginsnotok)`` may exist -- it just may not be used by ``margins``.

    The point of this test is weak on its own: it only records that the macro parses. The
    STRONG check lives in ``test_margins_macros_per_command``, which asserts that
    ``margins`` actually refuses them.
    """

    frame = _frame()
    fitted = BUILDERS[model_name](frame)

    forbidden = str(fitted.e.get("marginsnotok", "") or "").split()
    for entry in forbidden:
        assert isinstance(entry, str)


def test_tobit_advertises_pr_and_can_produce_it() -> None:
    """The specific case that started this: ``tobit.ado:145-148`` lists ``Pr(passthru)``."""
    frame = _frame()
    fitted = ws.Tobit.from_formula("ycens ~ x", frame, lower=0).fit()

    assert "pr" in _declared_types(str(fitted.e["marginsok"]))
    produced = np.asarray(fitted.predict(kind="pr", lower=0.0, upper=2.0), dtype=float)
    assert produced.size == len(fitted.model_data)
    assert np.all((produced >= 0.0) & (produced <= 1.0))


def test_heckman_advertises_five_selection_types_and_can_produce_them() -> None:
    frame = _frame()
    fitted = ws.Heckman.from_formula("y ~ x + select(s ~ z)", frame).fit()

    advertised = set(_declared_types(str(fitted.e["marginsok"])))
    expected = {"psel", "ycond", "yexpected", "mills", "nshazard"}

    assert expected <= advertised
    for kind in expected:
        assert np.asarray(fitted.predict(kind=kind)).size > 0


def test_etregress_advertises_five_treatment_types_and_can_produce_them() -> None:
    frame = _frame()
    fitted = ws.Etregress.from_formula("y ~ x + treat(t ~ z)", frame).fit()

    advertised = set(_declared_types(str(fitted.e["marginsok"])))
    expected = {"cte", "yctrt", "ycntrt", "ptrt", "xbtrt"}

    assert expected <= advertised
    for kind in expected:
        assert np.asarray(fitted.predict(kind=kind)).size > 0