"""VCE dispatcher: ``cov_type`` / ``cov_kwds`` -> ``(cov, df_t)``.

This is the single entry point estimators use to obtain a covariance matrix, so
the Stata corrections live in exactly one place (``[R] regress``,
``vce(robust)`` and ``vce(cluster id)``):

* ``"nonrobust"`` -- ``(u'u / df_resid) * (X'X)^-1``, ``df_t = N - K``
* ``"HC0"``..``"HC3"`` -- White/HC covariances, ``df_t = N - K``
* ``"cluster"`` -- CR1, ``cov_kwds={"groups": ...}``, ``df_t = G - 1``

Any other ``cov_type`` raises ``ValueError`` listing the supported values, in
line with westata's rule of never silently returning a number it cannot stand
behind.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from westata.protocols import SSC
from westata.vce.cluster import cov_cluster
from westata.vce.robust import cov_hc
from westata.vce.ssc import resolve_ssc

SUPPORTED_COV_TYPES = ("nonrobust", "HC0", "HC1", "HC2", "HC3", "cluster")
"""Canonical ``cov_type`` values understood by :func:`compute_cov`."""

_ALIASES = {
    "NONROBUST": "nonrobust",
    "HC0": "HC0",
    "HC1": "HC1",
    "HC2": "HC2",
    "HC3": "HC3",
    "CLUSTER": "cluster",
}


def _normalize_cov_type(cov_type: str) -> str:
    key = str(cov_type).strip().upper()
    try:
        return _ALIASES[key]
    except KeyError:
        raise ValueError(
            f"Unknown cov_type {cov_type!r}; supported cov_type values are: "
            + ", ".join(SUPPORTED_COV_TYPES)
        ) from None


def compute_cov(
    X: np.ndarray,
    resid: np.ndarray,
    *,
    cov_type: str = "nonrobust",
    cov_kwds: Mapping[str, Any] | None = None,
    ssc: SSC | None = None,
    nobs: int | None = None,
    df_resid: float | None = None,
) -> tuple[np.ndarray, float]:
    """Return ``(cov, df_t)`` for the requested variance estimator.

    Parameters
    ----------
    X:
        ``n x k`` design matrix (with any constant already included).
    resid:
        ``n`` residuals aligned with ``X``.
    cov_type:
        ``"nonrobust"``, ``"HC0"``, ``"HC1"``, ``"HC2"``, ``"HC3"`` or
        ``"cluster"`` (case-insensitive).
    cov_kwds:
        Estimator keyword arguments.  ``{"groups": ...}`` is required for
        ``cov_type="cluster"``; group labels may be integers, strings or a
        pandas ``Series``.
    ssc:
        Stata small-sample settings; defaults to
        ``SSC(adj=True, cluster_adj=True, df="stata")`` when ``None``.
    nobs:
        Optional explicit sample size, validated against ``X``.
    df_resid:
        Optional denominator for the nonrobust ``sigma2`` (and the returned
        ``df_t``).  Robust HC and cluster variances always use their Stata
        degrees of freedom (``N-K`` and ``G-1``) and ignore it.

    Returns
    -------
    (cov, df_t):
        ``cov`` is ``k x k``; ``df_t`` is the t/F degrees of freedom to use for
        inference (``N-K`` for nonrobust/HC, ``G-1`` for cluster).
    """

    X = np.asarray(X, dtype=float)
    resid = np.asarray(resid, dtype=float)

    if X.ndim != 2:
        raise ValueError(
            "X must be a 2-D design matrix (nobs x k); got "
            f"{X.ndim} dimension(s)"
        )
    if resid.ndim != 1:
        raise ValueError(
            "resid must be one-dimensional; got "
            f"{resid.ndim} dimension(s)"
        )

    n, k = X.shape
    if resid.shape[0] != n:
        raise ValueError(
            "resid length does not match the number of rows of X; got "
            f"resid length {resid.shape[0]} for nobs={n}"
        )
    if nobs is not None and int(nobs) != n:
        raise ValueError(
            f"nobs={nobs} does not match the data; got nobs={n} from X"
        )
    if n <= k:
        raise ValueError(
            "nobs must exceed the number of columns (K); got "
            f"nobs={n}, K={k}"
        )
    if cov_kwds is not None and not isinstance(cov_kwds, Mapping):
        raise TypeError(
            "cov_kwds must be a mapping (dict-like) of estimator keyword "
            f"arguments; got {type(cov_kwds).__name__}"
        )

    settings = resolve_ssc(ssc)
    canonical = _normalize_cov_type(cov_type)
    kwds: dict[str, Any] = {} if cov_kwds is None else dict(cov_kwds)

    bread = np.linalg.inv(X.T @ X)

    if canonical == "nonrobust":
        df_t = float(df_resid) if df_resid is not None else float(n - k)
        if df_t <= 0:
            raise ValueError(
                f"df_resid must be positive; got df_resid={df_t}"
            )
        cov = (float(resid @ resid) / df_t) * bread
        return cov, df_t

    if canonical in ("HC0", "HC1", "HC2", "HC3"):
        cov = cov_hc(X, resid, variant=canonical, bread=bread, ssc=settings)
        return cov, float(n - k)

    # canonical == "cluster"
    groups = kwds.get("groups")
    if groups is None:
        raise ValueError(
            "cov_type='cluster' requires cov_kwds['groups']; pass the "
            "cluster labels as cov_kwds={'groups': groups}"
        )
    cov, n_groups = cov_cluster(X, resid, groups, bread=bread, ssc=settings)
    return cov, float(n_groups - 1)
