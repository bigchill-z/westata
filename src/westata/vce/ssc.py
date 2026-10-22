"""Stata small-sample degrees-of-freedom corrections for VCEs.

This module holds the correction factors shared by every covariance estimator,
so no model can invent its own degrees-of-freedom rule.  The formulas are the
ones in Stata's ``[R] regress`` (Methods and formulas):

* heteroskedastic robust (HC1): ``N / (N - K)``
* cluster (CR1): ``G / (G - 1) * (N - 1) / (N - K)``

:class:`westata.protocols.SSC` carries the switches (Stata defaults are
``adj=True, cluster_adj=True``):

* ``adj`` gates the observation-based correction (``N/(N-K)`` for HC1,
  ``(N-1)/(N-K)`` for cluster).  With ``adj=False`` the sandwich is raw.
* ``cluster_adj`` gates the group-based ``G/(G-1)`` factor on top of ``adj``.
  It is only meaningful while ``adj`` is on, mirroring Stata: there is no
  ``vce(cluster)`` option that applies ``G/(G-1)`` alone.
"""

from __future__ import annotations

from westata.protocols import SSC

DEFAULT_SSC = SSC()
"""Stata's defaults, used whenever ``ssc=None`` is passed."""


def resolve_ssc(ssc: SSC | None) -> SSC:
    """Return ``ssc``, or Stata's defaults when it is ``None``."""

    return DEFAULT_SSC if ssc is None else ssc


def _require_positive_df(nobs: int, k: int) -> None:
    if nobs <= k:
        raise ValueError(
            "nobs must exceed the number of columns (K); got "
            f"nobs={nobs}, K={k}"
        )


def hc1_factor(nobs: int, k: int, ssc: SSC | None) -> float:
    """``N / (N - K)`` for HC1, or ``1`` when ``ssc.adj`` is off."""

    settings = resolve_ssc(ssc)
    if not settings.adj:
        return 1.0
    _require_positive_df(nobs, k)
    return float(nobs) / (nobs - k)


def cluster_factor(nobs: int, k: int, n_groups: int, ssc: SSC | None) -> float:
    """CR1 scaling ``G/(G-1) * (N-1)/(N-K)`` for cluster-robust VCEs.

    Returns ``1`` when ``ssc.adj`` is off.  With ``ssc.cluster_adj`` off the
    ``G/(G-1)`` factor is dropped but ``(N-1)/(N-K)`` is kept.
    """

    settings = resolve_ssc(ssc)
    if not settings.adj:
        return 1.0
    _require_positive_df(nobs, k)
    factor = (nobs - 1.0) / (nobs - k)
    if settings.cluster_adj:
        if n_groups < 2:
            raise ValueError(
                "cluster-robust covariance requires at least 2 groups; "
                f"got {n_groups}"
            )
        factor *= n_groups / (n_groups - 1.0)
    return float(factor)
