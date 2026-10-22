"""One-way cluster-robust covariance (CR1), Stata ``vce(cluster id)``.

The CR1 estimator in ``[R] regress`` (Methods and formulas) is

    V = G/(G-1) * (N-1)/(N-K) * (X'X)^-1
        [sum_g X_g' u_g u_g' X_g] (X'X)^-1

i.e. the CR0 sandwich times the CR1 small-sample factor from
:mod:`westata.vce.ssc`.  ``df_t = G - 1``.

Two-way/multi-way clustering (CGM) is a later phase; this module deliberately
rejects two-dimensional group arrays instead of guessing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from westata.protocols import SSC
from westata.vce.ssc import cluster_factor


def encode_groups(groups) -> np.ndarray:
    """Map arbitrary group labels to the integer codes ``0 .. G-1``.

    Labels may be integers, strings or any other hashable values (the spec uses
    ``df["industry"]``).  Missing labels are rejected rather than silently
    dropped, because dropping rows here would misalign the residuals.
    """

    values = np.asarray(groups)
    if values.ndim != 1:
        raise ValueError(
            "cov_kwds['groups'] must be one-dimensional; got "
            f"{values.ndim} dimensions"
        )
    codes, _ = pd.factorize(values, sort=False)
    if (codes < 0).any():
        raise ValueError(
            "cov_kwds['groups'] contains missing values; drop or recode them "
            "before computing a cluster-robust VCE"
        )
    return codes.astype(np.intp, copy=False)


def cov_cluster(
    X: np.ndarray,
    resid: np.ndarray,
    groups,
    *,
    bread: np.ndarray | None = None,
    ssc: SSC | None = None,
) -> tuple[np.ndarray, int]:
    """Return ``(cov, n_groups)`` for the CR1 cluster-robust covariance.

    ``df_t`` for t/F inference is ``n_groups - 1`` and is derived by the
    dispatcher from the returned group count.
    """

    X = np.asarray(X, dtype=float)
    resid = np.asarray(resid, dtype=float)
    n, k = X.shape

    codes = encode_groups(groups)
    if codes.shape[0] != n:
        raise ValueError(
            "cov_kwds['groups'] must have the same length as the data; got "
            f"length {codes.shape[0]} for nobs={n}"
        )

    n_groups = int(codes.max()) + 1
    if n_groups < 2:
        raise ValueError(
            "cluster-robust covariance requires at least 2 groups; "
            f"got {n_groups}"
        )

    if bread is None:
        bread = np.linalg.inv(X.T @ X)

    # Group sums of the observation scores x_i*u_i.  Loop over the (small)
    # number of columns, not over observations: every group sum is one C-level
    # bincount, and the k x k meat follows from a single matrix product.
    scores = X * resid[:, None]
    group_scores = np.empty((n_groups, k), dtype=float)
    for j in range(k):
        group_scores[:, j] = np.bincount(
            codes, weights=scores[:, j], minlength=n_groups
        )

    meat = group_scores.T @ group_scores
    cov = cluster_factor(n, k, n_groups, ssc) * (bread @ meat @ bread)
    return cov, n_groups
