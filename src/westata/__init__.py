"""Export postestimation and report helpers from the top-level package."""

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
    Logit,
    NegativeBinomial,
    OrderedLogit,
    OrderedProbit,
    PanelOLS,
    Poisson,
    Probit,
    RandomEffects,
    Tobit,
)
from westata.errors import (
    ConsistencyError,
    ConvergenceError,
    FormulaError,
    UnsupportedError,
    WeightError,
    WestataError,
)
from westata.postestimation import LincomResult, WaldTest
from westata.protocols import SSC
from westata.report import etable, to_latex, to_markdown

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
    "Logit",
    "Probit",
    "Poisson",
    "NegativeBinomial",
    "OrderedLogit",
    "OrderedProbit",
    "Tobit",
    "WaldTest",
    "LincomResult",
    "etable",
    "to_latex",
    "to_markdown",
    "WEIGHT_TYPES",
    "WeightSpec",
    "normalize_weights",
    "WestataError",
    "ConsistencyError",
    "ConvergenceError",
    "FormulaError",
    "UnsupportedError",
    "WeightError",
    "__version__",
]
