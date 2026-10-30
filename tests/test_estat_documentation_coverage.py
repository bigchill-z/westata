"""Every implemented ``estat_*`` method must appear in the gap register's command table.

The register at ``docs/GAP_REGISTER.md`` is the project's auditable inventory: a reader
consults it to learn what exists and what does not. A command that works but is absent
from the table is worse than a missing command -- it is a documented falsehood, and the
same defect appeared twice in this session:

* ``estat auc``, ``estat weakrobust`` and ``estat mundlak`` were implemented in earlier
  rounds and had DETAILED sections in the register but no row in the SUMMARY table;
* ``estat szroeter``/``esize``/``moran`` were added in rounds 200-202 and needed the same
  treatment.

So the table and the code are now checked against each other mechanically.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import westata as ws
from westata.results.base import StataResults

REGISTER = Path(__file__).resolve().parents[1] / "docs" / "GAP_REGISTER.md"


def _implemented_estat_methods() -> list[str]:
    """Every ``estat_<name>`` defined on the results object."""

    return sorted(
        name
        for name in dir(StataResults)
        if name.startswith("estat_") and callable(getattr(StataResults, name))
    )


def _register_text() -> str:
    assert REGISTER.exists(), REGISTER
    return REGISTER.read_text(encoding="utf-8", errors="ignore")


def _command_phrase(method: str) -> str:
    """``estat_auc`` -> ``estat auc``: how the register writes it."""

    return "estat " + method[len("estat_") :]


def test_there_are_several_estat_methods() -> None:
    """Guard: an empty list would make every check below vacuous."""

    assert len(_implemented_estat_methods()) >= 15


@pytest.mark.parametrize("method", _implemented_estat_methods())
def test_the_register_mentions_every_method(method: str) -> None:
    """A working command absent from the inventory is a documented falsehood."""

    phrase = _command_phrase(method)

    assert phrase in _register_text(), (
        f"{phrase!r} is implemented ({method}) but does not appear in "
        f"{REGISTER.name}; add a row to the command table"
    )


def test_the_summary_table_covers_the_regression_estat_family() -> None:
    """``regress_postestimation.sthlp`` lists seven; all seven are now implemented.

    Pinned by NAME so a future removal is a deliberate, visible change rather than a
    quiet regression.
    """

    text = _register_text()
    for phrase in (
        "estat hettest",
        "estat imtest",
        "estat ovtest",
        "estat vif",
        "estat szroeter",
        "estat esize",
        "estat moran",
    ):
        assert phrase in text, phrase


def test_the_uncertain_command_is_marked_as_such() -> None:
    """``estat moran`` cannot be verified against Stata locally, and the register says so.

    The marker matters more than the wording: a reader must be able to tell this command
    apart from the ones validated against a fixture or readable ado source.
    """

    text = _register_text()
    line = next(
        (line for line in text.splitlines() if line.startswith("| `estat moran`")),
        None,
    )

    assert line is not None, "estat moran has no row in the command table"
    assert "⚠️" in line or "可能数值不同" in line


def test_the_refused_capability_is_marked_as_such() -> None:
    """``estat weakrobust`` implements AR but refuses CLR; the register distinguishes."""

    text = _register_text()
    line = next(
        (line for line in text.splitlines() if line.startswith("| `estat weakrobust`")),
        None,
    )

    assert line is not None
    assert "CLR" in line
    assert "部分实现" in line or "⚠️" in line


def test_the_register_names_the_package_public_surface() -> None:
    """A command reachable from the results object should be reachable in the docs.

    ``result.estat_<name>()`` is the calling convention the register documents, so the
    phrase should appear for every implemented method.
    """

    text = _register_text()
    missing = [
        method
        for method in _implemented_estat_methods()
        if f"result.{method}" not in text and f"estat_{method[6:]}" not in text
    ]

    assert missing == [], missing


# ----------------------------------------------------------- predict coverage
def _predict_kinds() -> list[str]:
    from westata.postestimation.predict import SUPPORTED_TYPES

    return sorted({kind for kinds in SUPPORTED_TYPES.values() for kind in kinds})


def test_there_are_several_predict_kinds() -> None:
    """Guard: an empty set would make the check below vacuous."""

    assert len(_predict_kinds()) >= 20


@pytest.mark.parametrize("kind", _predict_kinds())
def test_the_register_mentions_every_predict_kind(kind: str) -> None:
    """The same honesty rule as for ``estat``: implemented types must be documented.

    ``predict, ystar`` was working all along but appeared NOWHERE in the register -- the
    grep for ``ystar`` returned zero hits while ``SUPPORTED_TYPES["tobit"]`` contained it.
    """

    assert f"`{kind}`" in _register_text(), (
        f"predict kind {kind!r} is implemented but does not appear in the register"
    )


def test_the_tobit_expected_value_types_are_documented() -> None:
    """``tobit_postestimation.sthlp:143-148`` defines both, and they are NOT the same.

    ``e`` is the conditional expectation ``E(xb+u | a < xb+u < b)``; ``ystar`` is
    ``E(y*)``.  The register has to say which is which, because a reader who assumes the
    two coincide would misread one of them.
    """

    text = _register_text()
    line = next(
        (line for line in text.splitlines() if "`predict, e`" in line),
        None,
    )

    assert line is not None, "the tobit expected-value row is missing"
    assert "ystar" in line
    assert "条件" in line or "conditional" in line