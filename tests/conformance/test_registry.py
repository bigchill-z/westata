"""Tests for the conformance evidence registry (design spec section 11.3).

The registry stores the evidence level (E0..E3), the oracle sources and the
numeric tolerance backing each Stata command.  The schema is validated eagerly
so a typo in ``registry.yml`` fails the conformance suite instead of silently
weakening a claim.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from westata.conformance.registry import Evidence, load_registry, validate_registry


VALID_YAML = """\
- command: regress
  level: E2
  tolerance: 1.0e-10
  verified_by: lead
  date: 2026-10-22
  sources:
    - type: published_output
      ref: "Stata 18 [R] regress, Example 1"
      dataset: auto.dta
    - type: statsmodels_reference
      call: 'sm.OLS(y, X).fit()'
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "registry.yml"
    path.write_text(text, encoding="utf-8")
    return path


def _entry(**overrides: object) -> Evidence:
    fields: dict[str, object] = {
        "command": "regress",
        "level": "E2",
        "sources": [{"type": "published_output", "ref": "manual"}],
        "tolerance": 1e-10,
    }
    fields.update(overrides)
    return Evidence(**fields)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_load_registry_parses_valid_entry(tmp_path: Path) -> None:
    entries = load_registry(_write(tmp_path, VALID_YAML))

    assert len(entries) == 1
    entry = entries[0]
    assert isinstance(entry, Evidence)
    assert entry.command == "regress"
    assert entry.level == "E2"
    assert entry.tolerance == pytest.approx(1e-10)
    assert [source["type"] for source in entry.sources] == [
        "published_output",
        "statsmodels_reference",
    ]
    # Source payload beyond ``type`` is kept verbatim for citation purposes.
    assert entry.sources[0]["dataset"] == "auto.dta"


def test_load_registry_accepts_numeric_string_tolerance(tmp_path: Path) -> None:
    # YAML 1.1 parses ``1e-10`` (no decimal point) as a string; the loader
    # must still yield a usable float.
    text = VALID_YAML.replace("tolerance: 1.0e-10", "tolerance: 1e-10")
    entries = load_registry(_write(tmp_path, text))

    assert entries[0].tolerance == pytest.approx(1e-10)


def test_load_registry_preserves_entry_order(tmp_path: Path) -> None:
    text = VALID_YAML + """\
- command: logit
  level: E1
  tolerance: 1.0e-5
  sources:
    - type: invariant
      ref: "observed information equals -H"
"""
    entries = load_registry(_write(tmp_path, text))

    assert [entry.command for entry in entries] == ["regress", "logit"]


def test_evidence_is_frozen() -> None:
    entry = _entry()

    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.level = "E3"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Invalid evidence level
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", ["E4", "e2", "", "A1", "E"])
def test_validate_rejects_unknown_level(level: str) -> None:
    with pytest.raises(ValueError, match="level"):
        validate_registry([_entry(level=level)])


def test_load_registry_rejects_unknown_level(tmp_path: Path) -> None:
    text = VALID_YAML.replace("level: E2", "level: E9")

    with pytest.raises(ValueError, match="level"):
        load_registry(_write(tmp_path, text))


def test_validate_rejects_non_string_level() -> None:
    with pytest.raises(ValueError, match="level"):
        validate_registry([_entry(level=2)])


# ---------------------------------------------------------------------------
# Empty / malformed sources
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sources", [[], None, {}])
def test_validate_rejects_empty_sources(sources: object) -> None:
    with pytest.raises(ValueError, match="sources"):
        validate_registry([_entry(sources=sources)])


def test_load_registry_rejects_empty_sources(tmp_path: Path) -> None:
    text = VALID_YAML.replace(
        """  sources:
    - type: published_output
      ref: "Stata 18 [R] regress, Example 1"
      dataset: auto.dta
    - type: statsmodels_reference
      call: 'sm.OLS(y, X).fit()'
""",
        "  sources: []\n",
    )

    with pytest.raises(ValueError, match="sources"):
        load_registry(_write(tmp_path, text))


@pytest.mark.parametrize(
    "source",
    [
        {"type": "vibes"},  # unknown type
        {"ref": "no type key at all"},  # missing type
        "published_output",  # not a mapping
    ],
)
def test_validate_rejects_unknown_or_malformed_source(source: object) -> None:
    with pytest.raises(ValueError, match="source"):
        validate_registry([_entry(sources=[source])])


def test_load_registry_rejects_unknown_source_type(tmp_path: Path) -> None:
    text = VALID_YAML.replace("type: statsmodels_reference", "type: vibes")

    with pytest.raises(ValueError, match="source type"):
        load_registry(_write(tmp_path, text))


# ---------------------------------------------------------------------------
# Invalid tolerances
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tolerance", [0.0, -1e-10, -1.0])
def test_validate_rejects_non_positive_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        validate_registry([_entry(tolerance=tolerance)])


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf")])
def test_validate_rejects_non_finite_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        validate_registry([_entry(tolerance=tolerance)])


def test_load_registry_rejects_non_positive_tolerance(tmp_path: Path) -> None:
    text = VALID_YAML.replace("tolerance: 1.0e-10", "tolerance: 0.0")

    with pytest.raises(ValueError, match="tolerance"):
        load_registry(_write(tmp_path, text))


# ---------------------------------------------------------------------------
# Malformed registry files
# ---------------------------------------------------------------------------


def test_load_registry_rejects_non_list_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="list"):
        load_registry(_write(tmp_path, "command: regress\n"))


def test_load_registry_rejects_empty_document(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        load_registry(_write(tmp_path, "\n"))


def test_load_registry_rejects_missing_field(tmp_path: Path) -> None:
    text = VALID_YAML.replace("  tolerance: 1.0e-10\n", "")

    with pytest.raises(ValueError, match="tolerance"):
        load_registry(_write(tmp_path, text))


def test_load_registry_rejects_non_mapping_entry(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mapping"):
        load_registry(_write(tmp_path, "- 42\n"))


def test_validate_rejects_non_evidence_items() -> None:
    with pytest.raises(ValueError, match="Evidence"):
        validate_registry([{"command": "regress"}])  # type: ignore[list-item]
