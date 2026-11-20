"""``xtoverid`` -- the Sargan-Hansen test of overidentifying restrictions.

Targets Mark Schaffer and Steve Stillman's user-written ``xtoverid`` (version 2.1.8,
04Jan2016).  It is NOT a Stata base command: it lives in ``ado/plus/x/xtoverid.ado`` and
requires ``ivreg2`` (version 2.1.15 or greater).

What the command does
---------------------
It re-estimates the model the caller already fitted using a DIFFERENT estimator and asks
whether the extra moment conditions that estimator uses are satisfied.  The reference
distribution is chi-square with as many degrees of freedom as there are EXTRA moment
conditions, so a rejection means the estimator the caller used is inconsistent.

The cases ``xtoverid.ado`` implements, and what this module covers
------------------------------------------------------------------
===============  =========================================  ==========
fitted model     what is re-estimated                       status
===============  =========================================  ==========
``xtreg, re``    the same equation plus the WITHIN         implemented
                 deviations of every regressor, testing
                 "fixed vs random effects"
``xtreg, fe``    refused by the ado                        refused
``xtivreg``      G2SLS / EC2SLS / FE / FD / BE variants     refused (see below)
``xtivreg2``     GMM-based, needs xtivreg2 installed       refused
``xthtaylor``    Hausman-Taylor / Amemiya-Macurdy          refused
===============  =========================================  ==========

The ``xtreg, re`` case is the one this package implements, because it is the one whose
inputs a ``RandomEffects`` fit already carries: ``e(theta)``, ``e(sigma_u)``,
``e(sigma_e)`` and the entity labels.  The ``xtivreg``/``xtivreg2``/``xthtaylor`` branches
all re-estimate through ``ivreg2`` with instrument sets that this package's ``xtivreg``
implementation does not record in the shape the ado reads, so they are REFUSED with an
explanation rather than approximated.

The ``xtreg, re`` statistic, measured
------------------------------------
On ``lin_1992.dta`` (476 observations, 28 provinces), ``xtreg ltvfo ltlan ltwlab ltpow
ltfer hrs mci ngca, re`` followed by ``xtoverid`` prints::

    Test of overidentifying restrictions: fixed vs random effects
    Cross-section time-series model: xtreg re
    Sargan-Hansen statistic  53.285  Chi-sq(7)    P-value = 0.0000

with ``r(j) = 53.285084063389988``, ``r(jdf) = 7`` and ``r(jp) = 3.256392e-09``.  This
module reproduces that statistic to 3.3e-12 relative and ``_b[hrs]`` of the internal
refit to 1e-07.

Why the internal refit keeps the ORIGINAL theta
-----------------------------------------------
The ado refits ``xtreg ..., re`` on an augmented equation.  Every added regressor is a
WITHIN DEVIATION, whose group mean is identically zero, so the added columns carry no
between variation at all.  A variance-component estimator that re-estimates ``sigma_u``
from the augmented data therefore sees a between-regression rank that did not change while
the within fit did, and the two variance components drift apart.  Stata's own refit does
NOT drift: its printed ``sigma_u``/``theta`` in ``xtoverid, noisily`` are the ORIGINAL
ones (``.13324845`` / ``.81012778``), because the added columns cannot move the between
regression.

So this module quasi-demeans the augmented equation with the original ``theta`` and solves
the GLS normal equations at that fixed value, which is what makes the statistic match.
Re-fitting through :class:`westata.RandomEffects` instead re-estimates the components and
lands 15% away on ``sigma_u`` and 4.66 away on the statistic -- measured, which is why the
fixed-theta path is the implementation rather than a convenience.

Why the VCE follows the original fit
------------------------------------
``xtoverid.ado:178-181`` mirrors the ORIGINAL fit's ``e(vcetype)``: when that reads
"Robust" it passes ``robust`` to the internal refit.  On the same equation that changes
the statistic from 53.285 to 221.225 -- a 4x difference, not a rounding one -- so a
version that always used the model-based VCE would still pass a "does it run" check while
being wrong on every robust fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

__all__ = ["PanelOveridResults", "panel_overid"]


@dataclass
class PanelOveridResults:
    """The Sargan-Hansen test of overidentifying restrictions.

    Attributes
    ----------
    statistic:
        ``r(j)`` -- the Sargan-Hansen statistic.
    df:
        ``r(jdf)`` -- the number of overidentifying restrictions, i.e. the number of
        regressors added to the auxiliary equation.
    pvalue:
        ``r(jp)`` -- ``P > chi2(df)``.
    tested:
        Names of the coefficients the test constrains to zero (Stata's ``( 1) x = 0``
        lines, one per restriction).
    description:
        What the test compares, as the ado's first printed line spells it.  For
        ``xtreg, re`` this is ``"fixed vs random effects"``.
    model:
        The estimator the test re-estimated, e.g. ``"xtreg re"``.
    """

    statistic: float
    df: float
    pvalue: float
    tested: list[str]
    description: str = "fixed vs random effects"
    model: str = "xtreg re"

    def __str__(self) -> str:
        # Measured layout from a live `xtoverid` after `xtreg ..., re`:
        #
        #   Test of overidentifying restrictions: fixed vs random effects
        #   Cross-section time-series model: xtreg re
        #   Sargan-Hansen statistic  53.285  Chi-sq(7)    P-value = 0.0000
        #
        # The statistic is %7.3f, the degrees of freedom print bare in "Chi-sq(7)", and
        # the p-value is %5.4f. When the equation is exactly identified Stata prints
        # "  (equation exactly identified)" instead of the chi-square columns.
        lines = [
            f"Test of overidentifying restrictions: {self.description}",
            f"Cross-section time-series model: {self.model}",
        ]
        if self.statistic == 0:
            lines.append(
                f"Sargan-Hansen statistic {self.statistic:7.3f}"
                "  (equation exactly identified)"
            )
        else:
            lines.append(
                f"Sargan-Hansen statistic {self.statistic:7.3f}"
                f"  Chi-sq({self.df:g})    P-value = {self.pvalue:5.4f}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, float]:
        """Flatten to the ``r()`` names the ado returns.

        ``xtoverid.ado:273-275`` (and every other branch) sets exactly ``r(j)``,
        ``r(jp)`` and ``r(jdf)``.
        """

        return {
            "j": float(self.statistic),
            "jp": float(self.pvalue),
            "jdf": float(self.df),
        }

    def to_frame(self) -> pd.DataFrame:
        """One-row summary, for callers that collect several tests."""

        return pd.DataFrame(
            [
                {
                    "test": self.description,
                    "model": self.model,
                    "statistic": self.statistic,
                    "df": self.df,
                    "pvalue": self.pvalue,
                }
            ]
        )


def _require_re_fit(result: Any) -> tuple[pd.DataFrame, np.ndarray, list[str], str]:
    """The estimation frame, entity labels and slope names of an ``xtreg, re`` fit.

    ``xtoverid.ado:43-47`` accepts ``xtreg``, and ``:97-104`` then narrows it to
    ``e(model) == "re"``; every other ``xtreg`` estimator gets
    ``"xtoverid not compatible with xtreg model `e(model)'"`` (r(198)).  The other
    ``e(cmd)`` values the ado accepts need ``ivreg2``, which this package does not
    depend on, so they are refused here rather than re-estimated through a stand-in.
    """

    model_name = str(getattr(result, "model_name", "")).strip()
    lowered = model_name.lower()

    if lowered.startswith("xtivreg") or lowered.startswith("xthtaylor"):
        raise NotImplementedError(
            "xtoverid after "
            f"{model_name!r} re-estimates through ivreg2 with an instrument set this "
            "package does not record in the shape xtoverid.ado reads; the "
            "xtreg, re branch is the one implemented. Fit with "
            "ws.PanelRegressRE.from_formula(...) to use ws.panel_overid."
        )
    if not lowered.startswith("xtreg"):
        raise NotImplementedError(
            "xtoverid works only after xtreg, xtivreg, xtivreg2 or xthaylor "
            f"(xtoverid.ado:43-47); got {model_name or 'an unnamed model'!r}"
        )
    if not lowered.endswith(", re"):
        raise ValueError(
            f"xtoverid not compatible with xtreg model {model_name!r} "
            "(xtoverid.ado:97-104 accepts only the `re` estimator)"
        )

    frame = getattr(result, "model_data", None)
    if frame is None:
        raise ValueError(
            "xtoverid needs the estimation-sample data, which this result did not "
            "retain; refit with the formula interface"
        )

    entity_labels = getattr(result, "panel_entity", None)
    if entity_labels is None:
        raise ValueError(
            "xtoverid needs the panel identifier; this result records no panel_entity"
        )
    labels = np.asarray(entity_labels)
    if labels.size != len(frame):
        raise ValueError(
            f"xtoverid: the panel identifier has {labels.size} entries for "
            f"{len(frame)} rows"
        )

    names = [
        str(name)
        for name in getattr(result, "param_names", [])
        if str(name) not in ("Intercept", "_cons", "const")
    ]
    return frame, labels, names, model_name


def _group_mean(values: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Entity means, broadcast back to the row level."""

    totals = np.zeros(n_groups, dtype=float)
    np.add.at(totals, codes, values)
    means = totals / np.bincount(codes, minlength=n_groups)
    return means[codes]


def _quasi_demean(
    values: np.ndarray, codes: np.ndarray, n_groups: int, theta: float
) -> np.ndarray:
    """``x - theta * mean(x)`` within each entity -- the RE transformation."""

    return values - theta * _group_mean(values, codes, n_groups)


def _clustered_vcov(
    design: np.ndarray,
    residuals: np.ndarray,
    codes: np.ndarray,
) -> np.ndarray:
    """Cluster-robust covariance of a least-squares fit, clustered on the entity.

    Stata's ``xtreg ..., re robust`` is ``vce(cluster panelvar)`` -- the panel adapter's
    ``_BACKEND_COV_TYPES`` records that ``vce(robust)`` and ``vce(cluster panelvar)`` are
    the same matrix for this family, with ``max|V_robust - V_cluster| = 0.0`` measured.
    The small-sample factor is Stata's ``G/(G-1) * (N-1)/(N-K)``.

    The meat sums the score contributions WITHIN each cluster, so an entity's rows do not
    contribute independently -- that correlation is the whole point of the correction.
    """

    bread = np.linalg.inv(design.T @ design)
    scores = design * residuals[:, None]
    n_groups = int(codes.max()) + 1
    meat = np.zeros((design.shape[1], design.shape[1]), dtype=float)
    for g in range(n_groups):
        block = scores[codes == g].sum(axis=0)
        meat += np.outer(block, block)
    n = design.shape[0]
    k = design.shape[1]
    factor = (n_groups / (n_groups - 1.0)) * ((n - 1.0) / (n - k))
    return factor * (bread @ meat @ bread)


def panel_overid(result: Any) -> PanelOveridResults:
    """Stata's ``xtoverid`` -- the Sargan-Hansen test of overidentifying restrictions.

    Parameters
    ----------
    result:
        A random-effects panel fit, i.e. :class:`westata.RandomEffects` (``xtreg, re``).

    Returns
    -------
    XtOveridResult
        The statistic, its degrees of freedom, the p-value and the restrictions tested.

    Raises
    ------
    NotImplementedError
        For ``xtivreg``/``xtivreg2``/``xthtaylor`` fits, whose branches need ``ivreg2``.
    ValueError
        For a non-panel result, for any ``xtreg`` estimator other than ``re``, or when
        the estimation data or the panel identifier was not retained.

    Notes
    -----
    The null hypothesis is that the extra moment conditions hold, i.e. that the
    random-effects estimator is consistent; a rejection favours the fixed-effects one.
    The test is therefore a Hausman-type specification test computed through an
    auxiliary regression rather than through the ``V_c - V_e`` difference the
    :func:`westata.postestimation.modeltests.hausman` command uses, which is why the two
    do not have to agree numerically.

    ``xtoverid.ado:165-168`` refuses weighted fits outright, and this function does the
    same: the quasi-demeaning the test relies on is defined for the unweighted GLS
    estimator only.
    """

    frame, labels, slopes, model_name = _require_re_fit(result)

    if not slopes:
        raise ValueError(
            "xtoverid: the model has no regressors besides the constant, so there are "
            "no within deviations to test and the equation is exactly identified"
        )

    weight_type = str(getattr(result, "weight_type", "") or "").strip()
    if weight_type:
        raise NotImplementedError(
            "current version of xtoverid does not support weights "
            f"(xtoverid.ado:165-168); this fit used {weight_type!r}"
        )

    # `xtoverid.ado:241-244` refuses a degenerate RE fit that collapsed to pooled OLS,
    # because theta is then 0 and every within deviation is an arithmetic identity on the
    # original regressors rather than a genuine restriction.
    sigma_u = getattr(result, "e", {}).get("sigma_u")
    if sigma_u is not None and float(sigma_u) == 0.0:
        raise ValueError(
            "Error - saved RE estimates are degenerate (sigma_u=0) and equivalent to "
            "pooled OLS (xtoverid.ado:241-244)"
        )

    theta = float(getattr(result, "e", {}).get("theta", float("nan")))
    if not np.isfinite(theta):
        raise ValueError(
            "xtoverid: this fit stores no e(theta), so the quasi-demeaning factor the "
            "test needs is unavailable. An unbalanced panel stores the five "
            "e(thta_*) percentiles instead of a single theta, and the ado's xtreg, re "
            "branch assumes a single one."
        )

    # Sort by entity so the codes and the by-group maths line up, exactly as the ado's
    # `sort `ivar' `touse'` does.
    work = frame.copy()
    order = np.argsort(labels, kind="stable")
    work = work.iloc[order].reset_index(drop=True)
    codes_raw = np.asarray(labels)[order]
    _, codes = np.unique(codes_raw, return_inverse=True)
    n_groups = int(codes.max()) + 1

    response = str(getattr(result, "response_name", "") or "")
    if not response or response not in work.columns:
        raise ValueError(
            "xtoverid: the result records no response name, so the auxiliary equation "
            "cannot be rebuilt"
        )
    for name in slopes:
        if name not in work.columns:
            raise ValueError(
                f"xtoverid: {name!r} is not a column of the estimation data"
            )

    y = np.asarray(work[response], dtype=float)
    X = np.asarray(work[slopes], dtype=float)

    # The added regressors are the WITHIN DEVIATIONS, one per slope -- `xtoverid.ado:250-257`
    # builds ``x - xbar`` for every ``inexog`` variable.
    deviations = np.column_stack(
        [
            np.asarray(work[name], dtype=float)
            - _group_mean(np.asarray(work[name], dtype=float), codes, n_groups)
            for name in slopes
        ]
    )
    dev_names = [f"__xtoverid_dev_{name}" for name in slopes]

    # Quasi-demean the augmented equation at the ORIGINAL theta (see the module docstring:
    # re-estimating the variance components is what makes a refit-based version drift).
    y_d = _quasi_demean(y, codes, n_groups, theta)
    X_d = np.column_stack(
        [_quasi_demean(X[:, j], codes, n_groups, theta) for j in range(X.shape[1])]
    )
    dev_d = np.column_stack(
        [
            _quasi_demean(deviations[:, j], codes, n_groups, theta)
            for j in range(deviations.shape[1])
        ]
    )
    # The implicit constant quasi-demeans to ``1 - theta``.
    D_d = np.column_stack([X_d, dev_d, np.full(y.size, 1.0 - theta)])

    n = y.size
    k = D_d.shape[1]
    beta, *_ = np.linalg.lstsq(D_d, y_d, rcond=None)
    resid = y_d - D_d @ beta
    df_resid = n - k
    if df_resid <= 0:
        raise ValueError(f"xtoverid: no residual degrees of freedom (N={n}, K={k})")

    # ``xtoverid.ado:178-181`` mirrors the ORIGINAL fit's ``e(vcetype)``: when it reads
    # "Robust" it passes ``robust`` to the internal refit, and ``xtoverid.ado:260``
    # therefore calls ``xtreg ..., re robust``.  ``test`` then uses that fit's ROBUST
    # covariance, not the model-based one.  Measured on lin_1992: the robust statistic is
    # 221.225 against 53.285 for the same equation under the conventional VCE, so picking
    # the wrong one is a 4x error, not a rounding difference.
    vcetype = str((getattr(result, "e", None) or {}).get("vcetype") or "").strip()
    robust = bool(vcetype) and vcetype.upper() != "OIM"

    if robust:
        covariance = _clustered_vcov(D_d, resid, codes)
    else:
        sigma2 = float(resid @ resid) / df_resid
        covariance = sigma2 * np.linalg.inv(D_d.T @ D_d)

    positions = list(range(X.shape[1], X.shape[1] + deviations.shape[1]))
    b_dev = beta[positions]
    V_dev = covariance[np.ix_(positions, positions)]
    statistic = float(b_dev @ np.linalg.solve(V_dev, b_dev))
    df = float(len(positions))
    pvalue = float(scipy_stats.chi2.sf(statistic, df))

    return PanelOveridResults(
        statistic=statistic,
        df=df,
        pvalue=pvalue,
        tested=list(dev_names),
        description="fixed vs random effects",
        model="xtreg re",
    )
