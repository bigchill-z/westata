"""Oracle adapters for Stata conformance (design spec section 11.2)."""

from __future__ import annotations

from westata.conformance.oracles.invariants import check_ols_invariants

__all__ = ["check_ols_invariants"]
