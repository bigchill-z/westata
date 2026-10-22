"""Conformance harness: evidence registry, tolerances and oracle adapters.

Design spec: ``docs/superpowers/specs/2026-10-22-westata-design.md`` section
11.  Numeric claims against Stata are only as strong as their evidence level,
so the suite keeps the claim (registry), the yardstick (tolerances) and the
checkable identities (oracles) together.
"""

from __future__ import annotations

from westata.conformance.oracles.invariants import check_ols_invariants
from westata.conformance.registry import (
    VALID_LEVELS,
    VALID_SOURCE_TYPES,
    Evidence,
    load_registry,
    validate_registry,
)
from westata.conformance.tolerances import (
    LINEAR_RTOL,
    MLE_RTOL,
    MLE_SE_RTOL,
    assert_close,
)

__all__ = [
    "LINEAR_RTOL",
    "MLE_RTOL",
    "MLE_SE_RTOL",
    "VALID_LEVELS",
    "VALID_SOURCE_TYPES",
    "Evidence",
    "assert_close",
    "check_ols_invariants",
    "load_registry",
    "validate_registry",
]
