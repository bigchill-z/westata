"""Evidence registry for Stata-conformance claims.

Design spec: ``docs/superpowers/specs/2026-10-22-westata-design.md`` section
11.3.  ``tests/conformance/registry.yml`` is the single source of truth for
which evidence level (E0..E3) backs each Stata command, which oracle sources
were used, and the tolerance the claim was checked with.

The registry is validated when loaded so that an unknown evidence level, an
empty source list, an unknown oracle type or a nonsensical tolerance fails the
conformance suite instead of silently weakening a claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml

#: Evidence levels from the design spec section 11.1.
VALID_LEVELS: frozenset[str] = frozenset({"E0", "E1", "E2", "E3"})
#: Oracle source kinds from the design spec section 11.2.
VALID_SOURCE_TYPES: frozenset[str] = frozenset(
    {
        "published_output",
        "r_crosscheck",
        "statsmodels_reference",
        "invariant",
    }
)

_REQUIRED_FIELDS = ("command", "level", "sources", "tolerance")


@dataclass(frozen=True)
class Evidence:
    """One registry row: a command, its evidence level, sources and tolerance."""

    command: str
    level: str
    sources: list[dict]
    tolerance: float


def _validated_tolerance(value: Any, where: str) -> float:
    """Return ``value`` as a positive finite float or raise ``ValueError``."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: tolerance must be a number, got {value!r}")
    tolerance = float(value)
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError(
            f"{where}: tolerance must be positive and finite, got {value!r}"
        )
    return tolerance


def validate_registry(entries: Sequence[Evidence]) -> None:
    """Validate registry rows.

    Raises
    ------
    ValueError
        On an item that is not :class:`Evidence`, an empty command, an unknown
        evidence level, a missing/empty/malformed source list, an unknown
        source ``type``, or a tolerance that is not positive and finite.
    """

    for index, entry in enumerate(entries):
        if not isinstance(entry, Evidence):
            raise ValueError(
                f"entry #{index}: expected Evidence, got {type(entry).__name__}"
            )

        where = f"entry #{index} ({entry.command!r})"
        if not isinstance(entry.command, str) or not entry.command.strip():
            raise ValueError(f"{where}: command must be a non-empty string")

        if not isinstance(entry.level, str) or entry.level not in VALID_LEVELS:
            raise ValueError(
                f"{where}: invalid level {entry.level!r}; expected one of "
                f"{sorted(VALID_LEVELS)}"
            )

        if not isinstance(entry.sources, list) or not entry.sources:
            raise ValueError(
                f"{where}: sources must be a non-empty list of mappings"
            )
        for source_index, source in enumerate(entry.sources):
            source_where = f"{where}, source #{source_index}"
            if not isinstance(source, dict):
                raise ValueError(
                    f"{source_where}: source must be a mapping, "
                    f"got {type(source).__name__}"
                )
            source_type = source.get("type")
            if source_type not in VALID_SOURCE_TYPES:
                raise ValueError(
                    f"{source_where}: unknown source type {source_type!r}; "
                    f"expected one of {sorted(VALID_SOURCE_TYPES)}"
                )

        _validated_tolerance(entry.tolerance, where)


def load_registry(path: Path) -> list[Evidence]:
    """Load and validate ``registry.yml``.

    Extra keys on an entry (``verified_by``, ``date``, ...) are ignored; they
    are provenance metadata, not part of the runtime contract.

    Raises
    ------
    ValueError
        If the file is empty, is not a YAML list of mappings, misses a required
        field, or fails :func:`validate_registry`.
    OSError
        If ``path`` cannot be opened.
    """

    registry_path = Path(path)
    with registry_path.open("r", encoding="utf-8") as handle:
        try:
            raw = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise ValueError(f"{registry_path}: invalid YAML: {exc}") from exc

    if raw is None:
        raise ValueError(f"{registry_path}: evidence registry is empty")
    if not isinstance(raw, list):
        raise ValueError(
            f"{registry_path}: evidence registry must be a YAML list, "
            f"got {type(raw).__name__}"
        )

    entries: list[Evidence] = []
    for index, item in enumerate(raw):
        where = f"{registry_path}: entry #{index}"
        if not isinstance(item, dict):
            raise ValueError(
                f"{where}: entry must be a mapping, got {type(item).__name__}"
            )
        missing = [field for field in _REQUIRED_FIELDS if field not in item]
        if missing:
            raise ValueError(
                f"{where}: missing required field(s): {', '.join(missing)}"
            )

        tolerance = item["tolerance"]
        if isinstance(tolerance, str):
            # YAML 1.1 treats ``1e-10`` (no decimal point) as a string.
            try:
                tolerance = float(tolerance)
            except ValueError as exc:
                raise ValueError(
                    f"{where}: tolerance must be a number, got {tolerance!r}"
                ) from exc

        entries.append(
            Evidence(
                command=item["command"],
                level=item["level"],
                sources=item["sources"],
                tolerance=tolerance,
            )
        )

    validate_registry(entries)
    return entries


__all__ = [
    "Evidence",
    "VALID_LEVELS",
    "VALID_SOURCE_TYPES",
    "load_registry",
    "validate_registry",
]
