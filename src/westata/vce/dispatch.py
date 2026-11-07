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
from westata.vce.cluster import cov_cluster, cov_multiway_cluster
from westata.vce.hac import cov_hac
from westata.vce.robust import cov_hc
from westata.vce.ssc import resolve_ssc

SUPPORTED_COV_TYPES = ("nonrobust", "HC0", "HC1", "HC2", "HC3", "cluster", "HAC")
"""Canonical ``cov_type`` values understood by :func:`compute_cov`."""

_ALIASES = {
    "NONROBUST": "nonrobust",
    "HC0": "HC0",
    "HC1": "HC1",
    "HC2": "HC2",
    "HC3": "HC3",
    "CLUSTER": "cluster",
    "HAC": "HAC",
    # ``newey``/``vce(hac ...)`` spellings.
    "NEWEY": "HAC",
    "NEWEYWEST": "HAC",
    # ``robust`` IS THE STATA OPTION NAME, and for `regress, robust` it means HC1. Accepting the
    # word here keeps a caller from having to remember which numeric spelling each command family
    # wants -- the deviation the v3 report records at section 4, item (5). Note that the IV family
    # maps the same word to HC0 instead (Stata's `ivregress ..., vce(robust)` is HC0), and that
    # override lives in `estimators/iv.py` because the mapping genuinely differs per command.
    "ROBUST": "HC1",
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


def _stata_multiway_reference(estimate_model: str | None) -> str:
    """Which Stata command is the reference for multiway clustering HERE?

    The answer changes the message a caller sees, and getting it backwards is misleading
    in either direction:

    * ``regress`` accepts ``vce(cluster a b)`` (``regress.sthlp:196``), so an OLS fit is
      reproducing stock behaviour and the note should say so briefly.
    * stock ``ivregress`` refuses it ("multiway cluster VCE not allowed"), so an IV fit is
      reproducing ``ivreg2`` -- user-written -- and must say so, or the caller will try to
      match the number in vanilla Stata and fail.
    """

    name = str(estimate_model or "").strip().lower()
    if name.startswith("iv") or name in ("2sls", "liml", "gmm", "ivregress"):
        return "ivregress"
    return "regress"


def cluster_labels_from_kwds(kwds: Any) -> Any:
    """Extract cluster labels from ``cov_kwds`` under ANY of the accepted key spellings.

    ``groups``, ``clusters`` and ``cluster`` all work, everywhere. Stata's option is
    ``vce(cluster clustvar)`` so ``clusters`` reads naturally; the OLS/ML family documented
    ``groups`` while the IV family documented ``clusters``, and ``hdfe`` used ``cluster``.
    Each rejected the others' spelling, so moving a call between families failed even
    though the labels were correct.

    One accessor, because the lookup had been repeated in four modules -- and fixing them
    one at a time is how the mismatch survived in the first place.
    """

    if not kwds:
        return None
    for key in ("groups", "clusters", "cluster"):
        labels = kwds.get(key)
        if labels is not None:
            return labels
    return None


def compute_cov(
    X: np.ndarray,
    resid: np.ndarray,
    *,
    cov_type: str = "nonrobust",
    cov_kwds: Mapping[str, Any] | None = None,
    ssc: SSC | None = None,
    nobs: int | None = None,
    df_resid: float | None = None,
    nobs_adj: float | None = None,
    weights: np.ndarray | None = None,
    weight_type: str | None = None,
    estimate_model: str | None = None,
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
        Optional explicit row count, validated against ``X``.
    df_resid:
        Optional denominator for the nonrobust ``sigma2`` (and the returned
        ``df_t``).  Robust HC and cluster variances always use their Stata
        degrees of freedom (``N-K`` and ``G-1``) and ignore it.
    nobs_adj:
        Sample size for the small-sample correction factors.  Defaults to the
        number of rows of ``X``.  Frequency weights must pass ``sum(w)``: Stata
        ``[U] 20.24.1`` requires ``fweight`` estimates to match the estimates
        from the explicitly expanded data, and that includes the ``N/(N-K)``
        and ``(N-1)/(N-K)`` factors.

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

    # The correction factors use the *effective* sample size.  For frequency
    # weights that is sum(w), so fweighted results equal the expanded-data fit.
    n_eff = float(n if nobs_adj is None else nobs_adj)
    if n_eff <= k:
        raise ValueError(
            "effective nobs must exceed the number of columns (K); got "
            f"nobs={n_eff:g}, K={k}"
        )

    bread = np.linalg.inv(X.T @ X)

    if weights is not None:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.shape[0] != n:
            raise ValueError(
                f"weights have {w.shape[0]} entries but the data have {n} rows"
            )
        # Work in Stata's weighted space, X~ = W^{1/2}X and e~ = W^{1/2}e, so
        # that bread = (X~'X~)^-1 is the same expression in both cases.
        Xw = X * np.sqrt(w)[:, None]
        rw = resid * np.sqrt(w)
        bread = np.linalg.inv(Xw.T @ Xw)
    else:
        Xw, rw = X, resid

    if canonical == "nonrobust":
        df_t = float(df_resid) if df_resid is not None else float(n_eff - k)
        if df_t <= 0:
            raise ValueError(
                f"df_resid must be positive; got df_resid={df_t}"
            )
        cov = (float(rw @ rw) / df_t) * bread
        return cov, df_t

    if canonical in ("HC0", "HC1", "HC2", "HC3"):
        cov = cov_hc(
            Xw,
            rw,
            variant=canonical,
            bread=bread,
            ssc=settings,
            nobs=n_eff,
            weights=weights,
            weight_type=weight_type,
        )
        return cov, float(n_eff - k)

    if canonical == "HAC":
        # ``newey`` requires lag() explicitly (``newey.sthlp:55``); there is no
        # default bandwidth, so a missing ``maxlags`` is an error rather than an
        # arbitrary choice.
        lags = kwds.get("maxlags", kwds.get("lags"))
        if lags is None:
            raise ValueError(
                "cov_type='HAC' requires cov_kwds['maxlags']; Stata's newey "
                "takes lag() as a required option with no default "
                "(newey.sthlp:55)"
            )
        try:
            lag_value = int(lags)
        except (TypeError, ValueError):
            raise ValueError(
                f"cov_kwds['maxlags'] must be an integer; got {lags!r}"
            ) from None
        if lag_value < 0:
            raise ValueError(
                f"lag() must be nonnegative (newey.ado:24-27); got {lag_value}"
            )
        cov = cov_hac(
            Xw,
            rw,
            lags=lag_value,
            bread=bread,
            ssc=settings,
            nobs=n_eff,
            df_resid=float(n_eff - k),
        )
        return cov, float(n_eff - k)

    # canonical == "cluster"
    #
    # ``groups``, ``clusters`` and ``cluster`` are ALL accepted, everywhere. Stata calls the
    # option ``vce(cluster clustvar)`` so ``clusters`` reads naturally; the OLS/ML family
    # documented ``groups``, the IV family documented ``clusters``, and ``hdfe`` used
    # ``cluster``. Each rejected the others' spelling, so the same call failed depending on
    # which estimator received it. Going through ``cluster_labels_from_kwds`` means there is
    # one answer instead of three.
    groups = cluster_labels_from_kwds(kwds)
    if groups is None:
        raise ValueError(
            "cov_type='cluster' requires the cluster labels; pass them as "
            "cov_kwds={'groups': groups} (aliases 'clusters' and 'cluster' are also "
            "accepted)"
        )
    # MORE THAN ONE clustering dimension means multiway clustering, which Stata documents
    # as ``vce(cluster clustvarlist)`` with several variables (``regress.sthlp:196``) and
    # computes by Cameron-Gelbach-Miller inclusion-exclusion. A two-column DataFrame or an
    # (n, 2) array is how a caller expresses that; a one-dimensional vector is the ordinary
    # single-cluster case.
    dimensions = _cluster_dimensions(groups)
    if dimensions is not None:
        cov, df_t = cov_multiway_cluster(
            Xw, rw, dimensions, bread=bread, ssc=settings, nobs=n_eff,
            stata_reference=_stata_multiway_reference(estimate_model),
        )
        return cov, df_t
    cov, n_groups = cov_cluster(
        Xw, rw, _flatten_single_dimension(groups), bread=bread, ssc=settings, nobs=n_eff
    )
    return cov, float(n_groups - 1)


def _cluster_dimensions(groups: Any) -> list[Any] | None:
    """Split ``groups`` into per-dimension label vectors, or ``None`` if it is one-way.

    A DataFrame with two or more columns, or a 2-D array whose second axis is the
    dimension count, is multiway. Everything else -- a Series, a list of labels, a 1-D
    array -- stays one-way, so existing calls are unaffected.
    """

    import numpy as _np
    import pandas as _pd

    if isinstance(groups, _pd.DataFrame):
        if groups.shape[1] >= 2:
            return [groups.iloc[:, index].to_numpy() for index in range(groups.shape[1])]
        # A ONE-COLUMN frame is still one-way, but it must be FLATTENED here: passing a
        # single-column frame on to ``cov_cluster`` leaves it 2-D and raises "must be
        # one-dimensional", even though the caller expressed an ordinary one-way spec.
        return None

    array = _np.asarray(groups)
    if array.ndim == 2 and array.shape[1] >= 2:
        return [array[:, index] for index in range(array.shape[1])]
    if array.ndim == 2 and array.shape[1] == 1:
        # Same story for an (n, 1) array.
        return None
    return None


def _flatten_single_dimension(groups: Any) -> Any:
    """Turn a one-column frame or ``(n, 1)`` array into a 1-D label vector."""

    import numpy as _np
    import pandas as _pd

    if isinstance(groups, _pd.DataFrame):
        return groups.iloc[:, 0].to_numpy()
    array = _np.asarray(groups)
    if array.ndim == 2 and array.shape[1] == 1:
        return array[:, 0]
    return groups
