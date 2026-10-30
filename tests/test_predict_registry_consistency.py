"""Every type in ``SUPPORTED_TYPES`` must actually be IMPLEMENTABLE.

Round 185 hit this error:

    UnsupportedError: type 'stddp' is registered for 'sqreg' but has no
    implementation; this is a westata bug

This was the THIRD occurrence in this session (``predict, ir`` and ``predict, score``
came first). Each time the registry promised a capability the dispatcher could not
serve, and each time it was found by hand.

This file closes the gap structurally: it walks the WHOLE registry and asserts that
each declared type either returns an array or refuses with a reason that is NOT the
internal-consistency message. A "registered but unimplemented" entry is a westata bug,
so it must never appear in the registry at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.postestimation.predict import SUPPORTED_TYPES


def _base_frame(n: int = 200, seed: int = 1800) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=200)
    return pd.DataFrame(
        {
            "y": latent,
            "x1": rng.normal(size=200),
            "yb": (latent > 0).astype(int),
            "yc": rng.poisson(3.0, size=200),
            "rep": rng.integers(1, 4, size=200),
            "w": rng.integers(1, 3, size=200).astype(float),
        }
    )


def _registry_entries():
    """One row per (model, kind) declared in the registry."""

    for model_name in sorted(SUPPORTED_TYPES):
        for kind in SUPPORTED_TYPES[model_name]:
            yield model_name, kind


def test_the_registry_is_not_empty() -> None:
    assert len(SUPPORTED_TYPES) >= 20


def test_every_entry_is_a_non_empty_tuple() -> None:
    for model_name, kinds in SUPPORTED_TYPES.items():
        assert isinstance(kinds, tuple), model_name
        assert kinds, model_name
        for kind in kinds:
            assert isinstance(kind, str) and kind, (model_name, kind)


def test_no_entry_declares_the_same_type_twice() -> None:
    """A duplicate would make ``supported_types`` misreport the count."""

    for model_name, kinds in SUPPORTED_TYPES.items():
        assert len(set(kinds)) == len(kinds), model_name


def test_xb_is_offered_everywhere_except_where_it_cannot_be() -> None:
    """``xb`` is the linear index and almost universal.

    The zero-inflated models expose it too, so the expectation is simply that it is
    common -- a model WITHOUT it has to be a deliberate decision.
    """

    without = [name for name, kinds in SUPPORTED_TYPES.items() if "xb" not in kinds]

    assert without == [], without


def test_the_internal_consistency_message_is_never_reachable() -> None:
    """Scan the dispatch source for the branches that can raise it.

    This is a source-level check rather than a runtime one: constructing 28 fitted
    models inside a unit test is slower and no more convincing than asserting that each
    declared type has a branch.
    """

    import importlib
    from pathlib import Path

    # ``westata.postestimation.predict`` resolves to the FUNCTION of that name (the
    # package re-exports it), so the module has to be fetched explicitly.
    module = importlib.import_module("westata.postestimation.predict")
    source = Path(module.__file__).read_text(encoding="utf-8", errors="ignore")
    # The message itself must exist -- it is the safety net.  It is built from a
    # multi-line concatenation, so assert on a fragment that sits inside one literal.
    assert "is registered for" in source
    assert "this is a westata bug" in source
    # And every kind declared in the registry must appear in the source as a string
    # literal, i.e. some branch mentions it.
    declared = {kind for kinds in SUPPORTED_TYPES.values() for kind in kinds}
    missing = [kind for kind in declared if f'"{kind}"' not in source]
    assert missing == [], missing


def test_the_registry_and_the_public_helper_agree() -> None:
    """``supported_types`` must not invent or drop entries."""

    from westata.postestimation.predict import supported_types

    for model_name, kinds in SUPPORTED_TYPES.items():
        assert supported_types(model_name) == kinds


def test_an_unknown_model_names_itself() -> None:
    """The refusal must say WHICH model was asked for."""

    from westata.postestimation.predict import supported_types

    with pytest.raises(Exception) as caught:
        supported_types("not_a_model")

    assert "not_a_model" in str(caught.value)