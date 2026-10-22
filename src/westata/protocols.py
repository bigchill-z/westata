"""Shared value objects and structural protocols.

The results protocol mirrors the statsmodels surface so that downstream code
written against statsmodels keeps working, while ``SSC`` carries the
Stata-specific small-sample corrections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class SSC:
    """Stata-style small-sample correction settings.

    Parameters
    ----------
    adj:
        Apply the ``N / (N - K)`` style degrees-of-freedom correction.
    cluster_adj:
        Apply the ``G / (G - 1)`` cluster correction on top of ``adj``.
    df:
        Which convention to use for the t/z degrees of freedom.  ``"stata"``
        follows the corresponding Stata command's default.
    """

    adj: bool = True
    cluster_adj: bool = True
    df: str = "stata"


@runtime_checkable
class ResultsProtocol(Protocol):
    """Minimal statsmodels-compatible results surface."""

    params: Any
    bse: Any
    tvalues: Any
    pvalues: Any
    nobs: int
    df_resid: float
    df_model: float
    llf: float
    rsquared: float
    rsquared_adj: float
    e: Mapping[str, Any]

    def cov_params(self) -> np.ndarray: ...

    def conf_int(self, alpha: float = 0.05) -> Any: ...

    def summary(self, style: str = "statsmodels") -> str: ...
