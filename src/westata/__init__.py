"""westata: Stata-conformant econometrics with a statsmodels-style API.

The public surface mirrors statsmodels' modelling style (``Model.from_formula``
then ``.fit()``), while the numerical behaviour targets the corresponding Stata
command.  See ``docs/superpowers/specs/`` for the conformance levels and the
evidence registry in ``tests/conformance/registry.yml``.
"""

from __future__ import annotations

from westata.data import WEIGHT_TYPES, WeightSpec, normalize_weights
from westata.estimators import (
    FEOLS,
    IV2SLS,
    IVGMM,
    IVLIML,
    OLS,
    BetweenOLS,
    FirstDifferenceOLS,
    PanelOLS,
    RandomEffects,
)
from westata.errors import (
    ConsistencyError,
    ConvergenceError,
    UnsupportedError,
    WeightError,
    WestataError,
)
from westata.protocols import SSC

__version__ = "0.0.1"


def ssc(**kwargs: object) -> SSC:
    """Convenience constructor mirroring ``ws.ssc(adj=True, ...)``."""

    return SSC(**kwargs)  # type: ignore[arg-type]


__all__ = [
    "SSC",
    "ssc",
    "OLS",
    "IV2SLS",
    "IVLIML",
    "IVGMM",
    "FEOLS",
    "PanelOLS",
    "RandomEffects",
    "BetweenOLS",
    "FirstDifferenceOLS",
    "WEIGHT_TYPES",
    "WeightSpec",
    "normalize_weights",
    "WestataError",
    "ConsistencyError",
    "ConvergenceError",
    "UnsupportedError",
    "WeightError",
    "__version__",
]
