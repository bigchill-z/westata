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

import itertools
from typing import Any, Sequence

import numpy as np
import pandas as pd

from westata.errors import ConsistencyError
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
    nobs: float | None = None,
) -> tuple[np.ndarray, int]:
    """Return ``(cov, n_groups)`` for the CR1 cluster-robust covariance.

    ``df_t`` for t/F inference is ``n_groups - 1`` and is derived by the
    dispatcher from the returned group count.

    ``nobs`` is the effective sample size used in the ``(N-1)/(N-K)`` factor;
    it defaults to the number of rows of ``X``.  Frequency-weighted fits must
    pass ``sum(w)`` so that the result matches the expanded-data fit.
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
        # ConsistencyError, not ValueError: the OTHER estimators already raise that for
        # this exact situation (``discrete.py`` via its cluster check), so the same user
        # error produced two different exception types and ``except WestataError`` caught
        # only one of them. ConsistencyError is also a ValueError, so nothing that used to
        # catch it stops working.
        raise ConsistencyError(
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
    n_eff = n if nobs is None else float(nobs)
    cov = cluster_factor(n_eff, k, n_groups, ssc) * (bread @ meat @ bread)
    return cov, n_groups


def cov_multiway_cluster(
    X: np.ndarray,
    resid: np.ndarray,
    dimensions: Sequence[Any],
    *,
    bread: np.ndarray | None = None,
    ssc: SSC | None = None,
    nobs: float | None = None,
) -> tuple[np.ndarray, float]:
    """Multiway cluster-robust covariance by the Cameron-Gelbach-Miller identity.

    ``regress.sthlp:196`` documents ``vce(cluster clustvarlist)`` with more than one
    variable, and Stata computes it as

        V = V_A + V_B - V_{A&B}          (two dimensions)

    generalising to the inclusion-exclusion sum over all non-empty intersections for more
    dimensions. Each term is an ordinary one-way cluster covariance computed on the SAME
    scores, so nothing new is estimated -- only combined. The gap list for ADHM (2020)
    recorded that westata could already produce all three one-way pieces and was missing
    just this combination step, verified by checking that
    ``sqrt(VA^2 + VB^2 - VAB^2)`` reproduced ivreg2's 9.236939 digit for digit.

    ``dimensions`` is a sequence of per-dimension label vectors (or columns), each the same
    length as ``X``.

    Returns ``(cov, df_t)``. The degrees of freedom follows Stata: the SMALLEST of the
    per-dimension ``G - 1`` values, because the dimension with the fewest clusters is what
    limits the inference.
    """

    X = np.asarray(X, dtype=float)
    resid = np.asarray(resid, dtype=float)
    n, k = X.shape

    if len(dimensions) < 2:
        raise ValueError(
            "cov_multiway_cluster needs at least two dimensions; a single dimension is "
            "the ordinary cluster covariance (cov_cluster)"
        )

    labels = [np.asarray(np.asarray(d, dtype=object).reshape(-1)) for d in dimensions]
    for index, column in enumerate(labels):
        if column.shape[0] != n:
            raise ValueError(
                f"cluster dimension {index} has {column.shape[0]} entries but the data "
                f"have {n} rows"
            )

    if bread is None:
        bread = np.linalg.inv(X.T @ X)

    scores = X * resid[:, None]
    n_eff = n if nobs is None else float(nobs)

    def _meat_for(codes: np.ndarray, n_groups: int) -> np.ndarray:
        group_scores = np.empty((n_groups, k), dtype=float)
        for j in range(k):
            group_scores[:, j] = np.bincount(
                codes, weights=scores[:, j], minlength=n_groups
            )
        return group_scores.T @ group_scores

    # Every non-empty subset of dimensions contributes with alternating sign. For two
    # dimensions that is +A +B -AB; for three, +A +B +C -AB -AC -BC +ABC.
    contributions: list[tuple[tuple[int, ...], int]] = []
    for subset_size in range(1, len(labels) + 1):
        sign = 1 if subset_size % 2 == 1 else -1
        for subset in itertools.combinations(range(len(labels)), subset_size):
            contributions.append((subset, sign))

    total = np.zeros((k, k), dtype=float)
    degrees_of_freedom: list[float] = []

    for subset, sign in contributions:
        # The intersection of several dimensions is the composite key of their labels.
        # ``np.asarray`` on a list of tuples would produce a 2-D array, which
        # ``encode_groups`` rightly rejects -- so the tuples are turned into a flat
        # object array explicitly. Two rows share a group iff every dimension's label
        # matches, which is exactly what the tuple encodes.
        if len(subset) == 1:
            key = labels[subset[0]]
        else:
            composite = list(zip(*(labels[index] for index in subset)))
            key = np.empty(len(composite), dtype=object)
            key[:] = composite
        codes = encode_groups(key)
        n_groups = int(codes.max()) + 1
        if n_groups < 2:
            raise ConsistencyError(
                "multiway cluster-robust covariance needs at least 2 groups in every "
                f"cluster combination; dimension(s) {subset} give {n_groups}"
            )
        # Each SUBSET carries its own small-sample factor, matching Stata: the adjustment
        # is G/(G-1) for that combination's group count, not a single global factor.
        factor = cluster_factor(n_eff, k, n_groups, ssc)
        total += sign * factor * (bread @ _meat_for(codes, n_groups) @ bread)
        degrees_of_freedom.append(float(n_groups - 1))

    return total, min(degrees_of_freedom)
