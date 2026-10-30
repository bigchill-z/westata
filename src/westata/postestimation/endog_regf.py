"""The regression-based endogeneity test: Stata's ``regF``.

Transcribed from ``ivregress_estat.ado:1285-1325`` plus the ``RobReg`` helper at
``ivregress_estat.ado:2406-2470``.

The procedure is an **augmented 2SLS regression**:

.. code-block:: stata

    * fitted endogenous regressors (``x2hat``)
    foreach var of local test {
        _regress `var' `exog' `addlinst' `wt' if `touse', `nocons'
        predict double `testhat`i''
    }

    * augmented model: y on exog + test + x2hat, estimated by 2SLS with `small`
    RobReg "`depvar'" "`exog' `test' `x2hat' `endterm'" ...,  "`x2hat'"

and inside ``RobReg``:

.. code-block:: stata

    ivregress 2sls `dep' `indep' [...], `constant' `vceopt' small
    test `test', df(`Fd')
    scalar `F' = r(F)

So ``regF`` is the joint F test that the **fitted** endogenous regressors carry
no explanatory power once the actual endogenous regressors are included.

Which VCEs report it (``ivregress_estat.ado:1128-1145``):

* ``robust`` -> yes (``vce(robust)`` inside ``RobReg``)
* ``cluster`` -> yes (``vce(cluster <clustvar>)``) -- this is the ONLY test Stata
  reports after a clustered VCE, which is why ``estat endogenous`` refuses the
  cluster case until ``regF`` exists;
* ``HAC`` -> yes (``vce(hac <kernel> <lags>)``);
* unadjusted -> only the Durbin-Wu pair, not ``regF``.

The statistic is right-tailed F with ``df_n`` = number of tested endogenous
regressors and ``df_d`` = the residual degrees of freedom of the augmented fit,
which for a ``cluster`` VCE is ``G - 1``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy import stats

__all__ = ["RegFResult", "endog_regF_test"]

_CONSTANT_NAMES = frozenset({"intercept", "intercept_", "_cons", "const"})


def _is_constant(name: str) -> bool:
    return str(name).strip().lower() in _CONSTANT_NAMES


class RegFResult:
    """Stata's ``regF`` endogeneity test.

    Attributes
    ----------
    statistic, df_num, df_denom, pvalue:
        The F statistic, its two degrees of freedom and the upper-tail p-value.
    test:
        Always ``"regF"``; present so callers can branch on the same attribute
        the score results expose.
    """

    __slots__ = ("statistic", "df_num", "df_denom", "pvalue", "test")

    def __init__(
        self, *, statistic: float, df_num: float, df_denom: float
    ) -> None:
        self.statistic = float(statistic)
        self.df_num = float(df_num)
        self.df_denom = float(df_denom)
        self.pvalue = (
            float(stats.f.sf(self.statistic, self.df_num, self.df_denom))
            if self.df_denom > 0
            else float("nan")
        )
        self.test = "regF"

    def __repr__(self) -> str:
        return (
            f"<RegFResult F({self.df_num:g},{self.df_denom:g})="
            f"{self.statistic:.4f} p={self.pvalue:.4f}>"
        )

    def __str__(self) -> str:
        return (
            f"\nTests of endogeneity:\n\n"
            f"{'Reg. F(' + f'{self.df_num:g}' + ',' + f'{self.df_denom:g}' + ')':<28}"
            f" = {self.statistic:>9.4f}\n"
            f"{'Prob > F':<28} = {self.pvalue:>9.4f}"
        )

    def to_dict(self) -> dict[str, float]:
        """Stata's ``r()`` scalars."""

        return {
            "regF": self.statistic,
            "p_regF": self.pvalue,
            "regFdf_n": self.df_num,
            "regFdf_d": self.df_denom,
        }


def endog_regF_test(
    structure: dict[str, Any],
    response: np.ndarray,
    *,
    cov_type: str = "nonrobust",
    cluster_labels: Sequence[Any] | None = None,
    tested: Sequence[str] | None = None,
) -> RegFResult:
    """Run ``regF``: the augmented-regression F test for endogeneity.

    Parameters
    ----------
    structure:
        The ``iv_structure`` dict (``exog``/``endog``/``instruments`` + names).
    response:
        The dependent variable on the estimation sample.
    cov_type:
        The VCE to use inside the augmented fit.  ``regF`` is reported for
        ``robust``, ``cluster`` and ``HAC``; an unadjusted VCE gets Durbin-Wu
        instead, so this is only called for the others.
    cluster_labels:
        Required when ``cov_type`` is clustered.
    tested:
        Endogenous regressors to test.  Stata requires the full set under a
        robust/cluster/HAC VCE (``ivregress_estat.ado:1070-1076``).

    Raises
    ------
    ValueError
        When there is no endogenous regressor, when ``tested`` is not the full
        set, or when a clustered VCE arrives without labels.
    """

    exog = np.asarray(structure["exog"], dtype=float)
    endog = np.asarray(structure["endog"], dtype=float)
    instruments = np.asarray(structure["instruments"], dtype=float)
    exog_names = [str(name) for name in structure["exog_names"]]
    endog_names = [str(name) for name in structure["endog_names"]]

    n_endog = endog.shape[1]
    if n_endog == 0:
        raise ValueError("regF needs at least one endogenous regressor")

    if tested is not None:
        requested = [str(v) for v in tested]
        if sorted(requested) != sorted(endog_names):
            raise ValueError(
                "cannot test a subset of endogenous regressors if a robust, "
                "cluster, or HAC VCE was used at estimation; "
                f"tested={requested} but the endogenous set is {endog_names}"
            )

    response = np.asarray(response, dtype=float).reshape(-1)
    n = response.size
    if exog.shape[0] != n or endog.shape[0] != n:
        raise ValueError(
            f"the design has {exog.shape[0]} rows but the response has {n}"
        )

    has_constant = any(_is_constant(name) for name in exog_names)
    included = [j for j, name in enumerate(exog_names) if not _is_constant(name)]

    def _design(extra: list[np.ndarray]) -> np.ndarray:
        parts: list[np.ndarray] = []
        if has_constant:
            parts.append(np.ones(n))
        if included:
            parts.append(exog[:, included])
        parts.extend(extra)
        return np.column_stack(parts)

    # x2hat: each endogenous regressor projected on (exog + addlinst).
    projection_design = _design([instruments])
    endhat = np.empty((n, n_endog))
    for column in range(n_endog):
        beta, *_ = np.linalg.lstsq(projection_design, endog[:, column], rcond=None)
        endhat[:, column] = projection_design @ beta

    # The augmented model keeps the ACTUAL endogenous regressors and adds the
    # fitted ones: y on (exog + endog + endhat).
    augmented = _design([endog, endhat])

    # ``RobReg`` runs ``ivregress 2sls `dep' `indep' ...`` where ``indep`` is
    # ``exog test x2hat`` -- with NO ``(endog = instr)`` block.  Without that
    # block ``ivregress 2sls`` has nothing to instrument and reduces to ordinary
    # least squares, so the augmented fit is plain OLS and ``test x2hat`` is an
    # ordinary Wald/F test on those three columns.
    #
    # Building it as a real 2SLS instead is wrong and fails loudly-ish: the
    # projected design then has rank 2 while the augmented design has rank 3,
    # because ``endhat`` is by construction a linear combination of the same
    # instrument set, so its coefficient is unidentified and F collapses to
    # exactly 0 (cond(P'P) = 1.7e16 on the probe data).
    coefficients, *_ = np.linalg.lstsq(augmented, response, rcond=None)
    resid = response - augmented @ coefficients

    # Covariance of the augmented fit, matching the requested VCE.
    from westata.vce.dispatch import compute_cov

    if str(cov_type).strip().lower() == "cluster":
        if cluster_labels is None:
            raise ValueError(
                "regF with a clustered VCE needs the cluster labels"
            )
        covariance, df_denom = compute_cov(
            augmented,
            resid,
            cov_type="cluster",
            cov_kwds={"groups": np.asarray(cluster_labels)},
        )
    else:
        canonical = {"hac": "HAC", "robust": "HC1"}.get(
            str(cov_type).strip().lower(), "HC1"
        )
        covariance, df_denom = compute_cov(
            augmented,
            resid,
            cov_type=canonical,
            cov_kwds={"maxlags": 1} if canonical == "HAC" else None,
        )

    # The tested coefficients are the fitted-endogenous block, which sits at the
    # end of the augmented design.
    start = augmented.shape[1] - n_endog
    block = np.arange(start, augmented.shape[1])
    beta_block = coefficients[block]
    cov_block = covariance[np.ix_(block, block)]

    statistic = float(
        beta_block @ np.linalg.solve(cov_block, beta_block) / n_endog
    )
    return RegFResult(
        statistic=statistic, df_num=float(n_endog), df_denom=float(df_denom)
    )
