"""Variance-covariance estimation for westata.

Public surface used by estimators:

* :func:`westata.vce.dispatch.compute_cov` -- ``(cov, df_t)`` from
  ``cov_type``/``cov_kwds``/``ssc`` (the only entry point models should call).
* :class:`westata.protocols.SSC` -- Stata small-sample correction settings.
* :func:`westata.vce.robust.cov_hc` -- HC0-HC3 building blocks.
* :func:`westata.vce.cluster.cov_cluster` -- CR1 cluster building block.

Everything has an explicitly Stata-documented default; unknown ``cov_type``
values fail loudly instead of falling back.
"""

from __future__ import annotations

from westata.protocols import SSC
from westata.vce.cluster import cov_cluster, encode_groups
from westata.vce.dispatch import SUPPORTED_COV_TYPES, compute_cov
from westata.vce.robust import HC_VARIANTS, bread, cov_hc, leverage
from westata.vce.ssc import cluster_factor, hc1_factor, resolve_ssc

__all__ = [
    "SSC",
    "SUPPORTED_COV_TYPES",
    "HC_VARIANTS",
    "compute_cov",
    "cov_hc",
    "cov_cluster",
    "encode_groups",
    "bread",
    "leverage",
    "resolve_ssc",
    "hc1_factor",
    "cluster_factor",
]
