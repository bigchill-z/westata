"""westata: Stata-conformant econometrics with a statsmodels-style API."""

from __future__ import annotations

from westata.errors import (
    ConsistencyError,
    ConvergenceError,
    UnsupportedError,
    WeightError,
    WestataError,
)
from westata.estimators import OLS
from westata.protocols import SSC

__version__ = "0.0.1"


def ssc(**kwargs: object) -> SSC:
    """Convenience constructor mirroring ``ws.ssc(adj=True, ...)``."""

    return SSC(**kwargs)  # type: ignore[arg-type]


__all__ = [
    "SSC",
    "ssc",
    "WestataError",
    "ConsistencyError",
    "ConvergenceError",
    "UnsupportedError",
    "WeightError",
    "__version__",
]
