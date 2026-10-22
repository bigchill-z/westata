"""westata formula layer.

Builds patsy design matrices with Stata-style ``C(x, base=...)`` reference
levels, keeping the model matrix identical to ``statsmodels.formula.api``.
"""

from __future__ import annotations

from westata.formula.factor import build_design

__all__ = ["build_design"]
