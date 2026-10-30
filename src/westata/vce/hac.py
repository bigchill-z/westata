"""Newey-West HAC covariance (Stata ``vce(hac ...)`` / ``newey``).

Transcribed from ``newey.ado:133-183``.  Writing ``g_t = x_t * e_t`` for the
score contribution of observation ``t``, Stata accumulates

.. code-block:: stata

    for j = 0 .. lag:
        tt_j = (sum_t g_t g_{t-j}') + (that)'
        tt_j = tt_j * (1 - j/(1+lag))          // Bartlett kernel
        if j == 0:  xtx = tt_0 * 0.5
        else:       xtx = xtx + tt_j

so the ``0.5`` on the first term undoes the ``+ transpose`` duplication.  The
sandwich is then ``bread * xtx * bread`` with the usual
``N/(N-K)`` small-sample factor.

Notes that decide whether the numbers match Stata:

* the kernel weight is ``1 - j/(1+lag)``, i.e. **Bartlett** with bandwidth
  ``lag`` -- this is hard-coded in ``newey.ado`` and is not selectable;
* ``lag(0)`` is documented as identical to ``vce(robust)``
  (``newey.sthlp:113-114``).  In Stata ``vce(robust)`` carries the ``N/(N-K)``
  factor, so ``lag(0)`` matches **HC1**, not HC0 -- verified to 3e-18.  Comparing
  it against HC0 would show a spurious ratio of exactly ``N/(N-K)``;
* lags run over the **row order of the estimation sample**, so the data must be
  in time order; Stata warns against ``tsset`` gaps and this module simply
  follows the row order it is given.

``newey.ado:24-27`` rejects a negative lag, and ``newey.sthlp:55`` requires the
option -- there is no default bandwidth.
"""

from __future__ import annotations

import numpy as np

from westata.protocols import SSC

__all__ = ["cov_hac", "hac_meat", "bartlett_weights"]


def bartlett_weights(lags: int) -> np.ndarray:
    """The kernel weights ``1 - j/(1+lag)`` for ``j = 0..lag``.

    ``newey.ado:174``: ``mat tt = (tt+tt')*(1-`j'/(1+`lag'))``.
    """

    if lags < 0:
        raise ValueError(f"lag must be nonnegative; got {lags}")
    return np.array([1.0 - j / (1.0 + lags) for j in range(lags + 1)])


def hac_meat(
    Xw: np.ndarray, resid_w: np.ndarray, lags: int
) -> np.ndarray:
    """``xtx`` in ``newey.ado``: the Bartlett-weighted score cross-products.

    Parameters
    ----------
    Xw:
        Weighted design matrix ``X~ = W^{1/2} X``, rows in time order.
    resid_w:
        Weighted residuals ``e~ = W^{1/2} e``, same order.
    lags:
        Maximum autocorrelation lag (Stata's ``lag()``).

    Returns
    -------
    numpy.ndarray
        The ``k x k`` meat matrix.

    Raises
    ------
    ValueError
        For a negative ``lags``.

    Notes
    -----
    Row ``t`` pairs with row ``t - j``.  For ``j = 0`` the contribution is
    ``g_t g_t'`` summed over ``t``; the ado computes ``(S + S') * 0.5`` which is
    the same quantity, and this function uses the direct form because it avoids
    the factor-of-two bookkeeping.
    """

    if lags < 0:
        raise ValueError(f"lag must be nonnegative; got {lags}")
    Xw = np.asarray(Xw, dtype=float)
    resid_w = np.asarray(resid_w, dtype=float).reshape(-1)
    if Xw.ndim != 2 or Xw.shape[0] != resid_w.size:
        raise ValueError(
            f"design has {Xw.shape[0] if Xw.ndim == 2 else '?'} rows but there "
            f"are {resid_w.size} residuals"
        )

    # g_t = x_t * e_t
    scores = Xw * resid_w[:, None]
    n = scores.shape[0]
    k = Xw.shape[1]

    # xtx = cross_0 + sum_{j>=1} w_j (cross_j + cross_j')
    #
    # ``newey.ado:174-181`` computes each term as ``(tt + tt') * w_j`` and then
    # halves the j = 0 term.  Because ``cross_0`` is symmetric,
    # ``(cross_0 + cross_0') * 0.5 == cross_0``, so the j = 0 contribution
    # reduces to the plain cross-product.  Writing that directly avoids the
    # factor-of-two bookkeeping -- verified identical to the ado transcription.
    meat = scores.T @ scores
    for j in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - j / (1.0 + lags)
        cross = scores[j:].T @ scores[: n - j]
        meat += weight * (cross + cross.T)
    return meat


def cov_hac(
    Xw: np.ndarray,
    resid_w: np.ndarray,
    *,
    lags: int,
    bread: np.ndarray,
    ssc: SSC | None = None,
    nobs: int | None = None,
    df_resid: float | None = None,
) -> np.ndarray:
    """Newey-West covariance: ``bread * meat * bread`` with ``N/(N-K)``.

    Parameters
    ----------
    Xw, resid_w:
        Weighted design and residuals in time order.
    lags:
        Stata's ``lag()``.
    bread:
        ``(X~'X~)^-1``.
    ssc, nobs, df_resid:
        Small-sample settings; ``df_resid`` defaults to ``nobs - k``.

    Raises
    ------
    ValueError
        For a negative lag.
    """

    if lags < 0:
        raise ValueError(
            f"vce(hac) needs a nonnegative lag; newey.ado:24-27 rejects a "
            f"negative one.  Got {lags}"
        )
    Xw = np.asarray(Xw, dtype=float)
    resid_w = np.asarray(resid_w, dtype=float).reshape(-1)
    bread = np.asarray(bread, dtype=float)

    n = resid_w.size
    k = Xw.shape[1]
    if df_resid is None:
        df_resid = float(n - k)
    if nobs is None:
        nobs = n

    meat = hac_meat(Xw, resid_w, lags)
    covariance = bread @ meat @ bread

    # Stata's small-sample factor for vce(hac) is N/(N-K), the same one
    # vce(robust) applies.  ``hc1_factor`` already encodes it and honors
    # ``ssc.adj``.
    from westata.vce.ssc import hc1_factor

    scale = hc1_factor(float(nobs), int(k), ssc)
    return covariance * scale
