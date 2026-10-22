"""Heteroskedasticity-robust covariance estimators (HC0-HC3).

Formulas follow ``[R] regress`` (Methods and formulas) and White (1980):

* ``HC0 = (X'X)^-1 [X' diag(u_i^2) X] (X'X)^-1``
* ``HC1 = N/(N-K) * HC0`` -- Stata's ``regress, robust``
* ``HC2 = (X'X)^-1 [X' diag(u_i^2/(1-h_i)) X] (X'X)^-1``
* ``HC3 = (X'X)^-1 [X' diag(u_i^2/(1-h_i)^2) X] (X'X)^-1``

with leverage ``h_i = x_i' (X'X)^-1 x_i``.
"""

from __future__ import annotations

import numpy as np

from westata.protocols import SSC
from westata.vce.ssc import hc1_factor

HC_VARIANTS = ("HC0", "HC1", "HC2", "HC3")


def bread(X: np.ndarray) -> np.ndarray:
    """Return ``(X'X)^-1``, the bread of the sandwich."""

    X = np.asarray(X, dtype=float)
    return np.linalg.inv(X.T @ X)


def leverage(X: np.ndarray, bread: np.ndarray) -> np.ndarray:
    """Return ``h_i = diag(X (X'X)^-1 X')`` without forming the n x n matrix."""

    X = np.asarray(X, dtype=float)
    return np.einsum("ij,jk,ik->i", X, np.asarray(bread, dtype=float), X)


def cov_hc(
    X: np.ndarray,
    resid: np.ndarray,
    *,
    variant: str = "HC0",
    bread: np.ndarray | None = None,
    ssc: SSC | None = None,
) -> np.ndarray:
    """Return the HC0/HC1/HC2/HC3 covariance matrix for ``X`` and ``resid``.

    Parameters
    ----------
    X:
        ``n x k`` design matrix.
    resid:
        ``n`` residuals.
    variant:
        One of ``"HC0"``, ``"HC1"``, ``"HC2"``, ``"HC3"``.
    bread:
        Optional pre-computed ``(X'X)^-1``.
    ssc:
        Small-sample settings; only ``ssc.adj`` affects HC1.
    """

    variant = str(variant).strip().upper()
    if variant not in HC_VARIANTS:
        raise ValueError(
            f"Unknown HC variant {variant!r}; supported variant values are: "
            + ", ".join(HC_VARIANTS)
        )

    X = np.asarray(X, dtype=float)
    resid = np.asarray(resid, dtype=float)
    if bread is None:
        bread = np.linalg.inv(X.T @ X)

    het_scale = resid**2
    if variant in ("HC2", "HC3"):
        one_minus_h = 1.0 - leverage(X, bread)
        if variant == "HC2":
            het_scale = het_scale / one_minus_h
        else:
            het_scale = het_scale / one_minus_h**2

    meat = (X.T * het_scale) @ X
    cov = bread @ meat @ bread
    if variant == "HC1":
        cov = hc1_factor(X.shape[0], X.shape[1], ssc) * cov
    return cov
