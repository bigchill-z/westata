"""Heteroskedasticity-robust covariance estimators (HC0-HC3).

Everything here works in Stata's *weighted* space ``X~ = W^{1/2} X`` and
``e~ = W^{1/2} e``, which is the notation used by the ``[R] regress``
"Methods and formulas" section.  That makes ``bread = (X~'X~)^-1`` uniform
across weighted and unweighted models.

The one place weight *type* matters is the meat, because Stata has two
different formulas (p. 24):

``vce(robust)`` with fweights (eq. 7)
    ``meat = sum_i u_i^2 x_i x_i'`` per replicated observation, i.e.
    ``sum_i w_i u_i^2 x_i x_i'`` in unweighted terms.  Writing ``h~_i`` for the
    squared weighted residual, that is ``h~_i^2 / w_i``.
``vce(robust)`` with aweights/pweights/iweights (eq. 8)
    ``meat = sum_i w_i^2 u_i^2 x_i x_i'``, which is exactly ``h~_i^2``.

Getting this wrong is not cosmetic: it scales every fweighted standard error by
``sqrt(w)``.  ``tests/test_fweight_expansion.py`` pins Stata's own invariant
that fweighted output equals the explicitly expanded-data fit.
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
    Xw: np.ndarray,
    resid_w: np.ndarray,
    *,
    variant: str = "HC0",
    bread: np.ndarray | None = None,
    ssc: SSC | None = None,
    nobs: float | None = None,
    weights: np.ndarray | None = None,
    weight_type: str | None = None,
) -> np.ndarray:
    """Return the HC0/HC1/HC2/HC3 covariance matrix.

    Parameters
    ----------
    Xw:
        Weighted design matrix ``W^{1/2} X`` (equal to ``X`` when unweighted).
    resid_w:
        Weighted residuals ``W^{1/2} e``.
    variant:
        One of ``"HC0"``, ``"HC1"``, ``"HC2"``, ``"HC3"``.
    bread:
        Optional pre-computed ``(Xw'Xw)^-1``.
    ssc:
        Small-sample settings; only ``ssc.adj`` affects HC1.
    nobs:
        Effective sample size for the ``N/(N-K)`` factor (``sum(w)`` for
        fweights, the row count otherwise).
    weights:
        Normalized weights; ``None`` means unweighted.
    weight_type:
        Stata weight type.  ``fweight`` uses the replication form of the meat;
        everything else uses the precision form.
    """

    variant = str(variant).strip().upper()
    if variant not in HC_VARIANTS:
        raise ValueError(
            f"Unknown HC variant {variant!r}; supported variant values are: "
            + ", ".join(HC_VARIANTS)
        )

    Xw = np.asarray(Xw, dtype=float)
    resid_w = np.asarray(resid_w, dtype=float)
    n, k = Xw.shape

    kind = None
    if weights is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.shape[0] != n:
            raise ValueError(
                f"weights have {w.shape[0]} entries but the data have {n} rows"
            )
        kind = (weight_type or "").strip().lower()

    if bread is None:
        bread = np.linalg.inv(Xw.T @ Xw)

    # Replication weights (fweight) contribute w copies of u^2 x x'; precision
    # weights contribute (w u)^2 x x'.  See the module docstring.
    if kind == "fweight":
        het_scale = (resid_w**2) / w
    else:
        het_scale = resid_w**2

    if variant in ("HC2", "HC3"):
        if kind == "fweight":
            # Under fweights the data are literally replicated, so each copy has
            # design row x_i (not sqrt(w_i) x_i) and the levered denominator is
            # 1 - x_i'(X~'X~)^-1 x_i.  Using the weighted leverage here would
            # divide by w_i and understate every fweighted HC2/HC3 standard error.
            X_orig = Xw / np.sqrt(w)[:, None]
            denom = 1.0 - leverage(X_orig, bread)
        else:
            denom = 1.0 - leverage(Xw, bread)
        het_scale = het_scale / denom if variant == "HC2" else het_scale / denom**2

    meat = (Xw.T * het_scale) @ Xw
    cov = bread @ meat @ bread
    if variant == "HC1":
        n_eff = float(n if nobs is None else nobs)
        cov = hc1_factor(n_eff, k, ssc) * cov
    return cov
