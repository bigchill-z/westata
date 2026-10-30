"""Breusch-Pagan / Cook-Weisberg test for heteroskedasticity.

Stata's ``estat hettest``.  The algorithm is transcribed from ``hettest.ado``
(StataNow 19.5, v3.4.4), not inferred from output:

.. code-block:: stata

    qui _predict double `ui' if `touse', resid
    qui replace `ui' = `ui'^2 / (e(rss)/e(N))        // standardised squared resid
    ...
    if "`varlist'" == "" {                           // no varlist -> fitted values
        qui _predict double `z' if `touse'
        local varlist `z'
    }
    ...
    quiet _regress `ui' `varlist' `wgt' if `touse'

    if "`iid'"   { chi2 = e(N)*e(r2)  ; df = e(df_m) }
    elif "`fstat'" { F   = e(F)       ; df_m/df_r }
    else           { chi2 = e(mss)/2  ; df = e(df_m) }   // the DEFAULT

Two details decide whether the number is right:

* The auxiliary regression is of the **standardised squared residual** on the
  regressors (or on the fitted values when no varlist is given).  The divisor
  ``e(rss)/e(N)`` is the ML residual variance, not ``rss/(N-K)``.
* The default statistic is ``e(mss)/2`` -- the model sum of squares of that
  auxiliary regression halved -- which equals ``N * R2 / 2`` only when the
  auxiliary regression is uncentred.  Using ``N * R2`` (the ``iid`` variant) as
  the default would be wrong by a factor of two on the same fit.

``hettest.ado`` refuses three situations, all reproduced here: ``pweight``ed
data, a robust or clustered VCE, and ``normal`` combined with ``iid``/``fstat``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["HetTestResult", "estat_hettest"]

_CONSTANT_NAMES = frozenset({"intercept", "intercept_", "_cons", "const"})

#: Methods accepted by ``estat hettest``.
_METHODS = ("normal", "iid", "fstat")


def _is_constant(name: str) -> bool:
    return str(name).strip().lower() in _CONSTANT_NAMES


class HetTestResult:
    """The Breusch-Pagan/Cook-Weisberg heteroskedasticity test.

    Attributes
    ----------
    method:
        ``"normal"`` (the default), ``"iid"`` or ``"fstat"``.
    statistic, df, pvalue:
        The reported test statistic, its degrees of freedom and the upper-tail
        p-value.  For ``method="fstat"`` ``df`` is the numerator df and
        ``df_denom`` holds the denominator.
    df_denom:
        Denominator df, only for ``method="fstat"``.
    variables:
        The regressors the auxiliary regression used.  Empty means the fitted
        values were used (Stata's default when no varlist is given).
    """

    __slots__ = ("method", "statistic", "df", "pvalue", "df_denom", "variables")

    def __init__(
        self,
        *,
        method: str,
        statistic: float,
        df: float,
        pvalue: float,
        df_denom: float | None = None,
        variables: Sequence[str] = (),
    ) -> None:
        self.method = method
        self.statistic = float(statistic)
        self.df = float(df)
        self.pvalue = float(pvalue)
        self.df_denom = None if df_denom is None else float(df_denom)
        self.variables = list(variables)

    def __repr__(self) -> str:
        return (
            f"<HetTestResult {self.method}: statistic={self.statistic:.6g} "
            f"df={self.df:g} p={self.pvalue:.4g}>"
        )

    def __str__(self) -> str:
        label = "F" if self.method == "fstat" else "chi2"
        lines = [
            "",
            "Breusch-Pagan/Cook-Weisberg test for heteroskedasticity",
            "Assumption: "
            + ("Normal error terms" if self.method == "normal" else "i.i.d. error terms"),
            "H0: Constant variance",
            "",
        ]
        if self.method == "fstat" and self.df_denom is not None:
            lines.append(
                f"{'F(' + f'{self.df:g}' + ', ' + f'{self.df_denom:g}' + ')':<20}"
                f" = {self.statistic:>8.2f}"
            )
            lines.append(f"{'Prob > F':<20} = {self.pvalue:>8.4f}")
        else:
            lines.append(
                f"{label + '(' + f'{self.df:g}' + ')':<20} = {self.statistic:>8.2f}"
            )
            lines.append(f"{'Prob > ' + label:<20} = {self.pvalue:>8.4f}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, float]:
        """The ``r()`` scalars Stata returns for this variant."""

        if self.method == "fstat":
            return {
                "F": self.statistic,
                "df_m": self.df,
                "df_r": float(self.df_denom) if self.df_denom is not None else float("nan"),
                "p": self.pvalue,
            }
        return {"chi2": self.statistic, "df": self.df, "p": self.pvalue}


def _check_guards(result: Any) -> None:
    """The refusals ``hettest.ado`` applies, plus the linear-model requirement.

    ``hettest.ado:22-41`` refuses ``pweight`` and any robust/clustered VCE.  It
    does NOT check the estimator name, because in Stata the commands that reach
    this code always store the linear-regression quantities it needs.

    That is not true here: ``estat_hettest`` is a method on every result, so
    calling it after ``logit`` silently produced a number (0.00496 against OLS's
    0.00487 -- indistinguishable from a real answer) even though a logit result
    stores no ``e(rss)``/``e(mss)``, which ``hettest.ado:108,155`` require.  A
    number that looks right is worse than a refusal, so the requirement is
    enforced explicitly.
    """

    cov_type = str(getattr(result, "cov_type", "") or "").strip().lower()
    if cov_type in ("hc0", "hc1", "hc2", "hc3", "robust", "cluster"):
        raise ValueError(
            "estat hettest is not appropriate after estimation using a robust "
            f"or clustered VCE (this result used {cov_type})"
        )
    # ``e(wtype)`` pweight: westata records the weight type on the result when
    # one was used.
    weight_type = getattr(result, "_weight_type", None)
    if weight_type is None:
        weight_type = getattr(getattr(result, "model", None), "_weight_type", None)
    if str(weight_type or "").strip().lower() == "pweight":
        raise ValueError(
            "estat hettest is not appropriate with pweighted data"
        )

    # ``rreg.ado:214`` sets ``est local estat_cmd ""`` -- robust regression
    # deliberately offers NO estat subcommands, hettest included.  It does store
    # e(rss)/e(mss), so the least-squares check below would pass and the test
    # would return a number whose distribution is not the one hettest assumes
    # (the residuals come from an iteratively reweighted fit, not OLS).
    model_name = str(getattr(result, "model_name", "") or "").strip().lower()
    if model_name == "rreg":
        raise ValueError(
            "estat hettest is not available after rreg: rreg.ado:214 sets "
            "e(estat_cmd) to empty, so Stata offers no estat subcommands after "
            "robust regression.  The test assumes OLS residuals."
        )

    # ``xtreg_estat.ado:15-28`` allows only summarize, ic, hausman, mundlak and
    # vce; every other subcommand is "estat <sub> not valid" (exit 321).  The
    # panel estimators store e(rss)/e(mss) on their pooled regression, so the
    # least-squares check below let hettest through and it returned a number that
    # Stata refuses outright.
    if model_name.startswith("xtreg"):
        raise ValueError(
            "estat hettest is not available after xtreg: xtreg_estat.ado:15-28 "
            "permits only summarize, ic, hausman, mundlak and vce and rejects "
            "everything else with 'estat <sub> not valid'."
        )

    # ``hettest.ado:108`` scales by ``e(rss)/e(N)`` and ``:155`` reports
    # ``e(mss)/2``: both are least-squares quantities.  Without them the test has
    # no definition.
    e = getattr(result, "e", None)
    if e is not None:
        has_rss = "rss" in e
        has_mss = "mss" in e
        if not (has_rss and has_mss):
            model_name = str(getattr(result, "model_name", "this model") or "this model")
            raise ValueError(
                "estat hettest is a linear-regression test: it is defined from "
                "e(rss) and e(mss) (hettest.ado:108,155), which a "
                f"{model_name} result does not store.  Stata offers hettest after "
                "regress/areg/xtreg-style fits only."
            )


def _auxiliary_regression(
    left: np.ndarray, design: np.ndarray
) -> tuple[float, float, float, float]:
    """OLS of ``left`` on ``design``; returns ``(r2, mss, rss, df_m)``."""

    with_constant = any(np.ptp(design[:, j]) == 0.0 for j in range(design.shape[1]))
    if not with_constant:
        design = np.column_stack([np.ones(left.size), design])
    coefficients, *_ = np.linalg.lstsq(design, left, rcond=None)
    fitted = design @ coefficients
    resid = left - fitted
    rss = float(resid @ resid)
    # e(mss) is the model sum of squares about the mean, so it needs the
    # centred total -- the same decomposition OLS uses.
    centred = left - left.mean()
    tss = float(centred @ centred)
    mss = tss - rss
    r2 = (mss / tss) if tss > 0 else float("nan")
    df_m = int(design.shape[1] - 1)  # slopes only, matching e(df_m)
    return r2, mss, rss, df_m


def estat_hettest(
    result: Any,
    *,
    method: str = "normal",
    variables: Sequence[str] | None = None,
    mtest: str | None = None,
) -> Any:
    """Stata's ``estat hettest`` (Breusch-Pagan / Cook-Weisberg).

    Parameters
    ----------
    result:
        A fitted linear result that retained its design and residuals.
    method:
        ``"normal"`` (default) reports ``e(mss)/2``; ``"iid"`` reports
        ``e(N)*e(r2)``; ``"fstat"`` reports the auxiliary regression's ``F``.
        Stata rejects ``normal`` combined with the other two, and so does this.
    variables:
        Regressors for the auxiliary regression.  ``None`` uses every
        non-constant regressor; an empty sequence uses the fitted values, which
        is Stata's behaviour when no varlist is given.

    Raises
    ------
    ValueError
        For a robust/clustered VCE, for pweighted data, for an inconsistent
        ``method``/option combination, or when the design was not retained.
    """

    normalized = str(method).strip().lower()
    if normalized not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")

    # ``hettest.ado:49-61`` refuses mtest() together with iid and fstat.
    if mtest is not None:
        if normalized == "iid":
            raise ValueError(
                "option iid may not be specified together with option mtest()"
            )
        if normalized == "fstat":
            raise ValueError(
                "option fstat may not be specified together with option mtest()"
            )
        from westata.postestimation.mtest import MTEST_METHODS

        if str(mtest).strip().lower() not in MTEST_METHODS:
            raise ValueError(
                f"mtest() method {mtest!r} is not supported; Stata's _mtest "
                "accepts " + ", ".join(MTEST_METHODS)
            )

    _check_guards(result)

    design = getattr(result, "design_matrix", None)
    if design is None:
        raise ValueError(
            "estat hettest needs the estimation design, which this result did "
            "not retain; refit with a from_formula constructor"
        )
    matrix_raw, raw_names = design
    matrix = np.asarray(matrix_raw, dtype=float)
    names = [str(name) for name in raw_names]

    resid = np.asarray(result.resid, dtype=float).reshape(-1)
    nobs = resid.size
    if matrix.shape[0] != nobs:
        raise ValueError(
            f"the design has {matrix.shape[0]} rows but there are {nobs} residuals"
        )

    # ui = resid^2 / (e(rss)/e(N)) -- the ML residual variance, per hettest.ado.
    rss_total = float(resid @ resid)
    scale = rss_total / float(nobs)
    if scale <= 0.0:
        raise ValueError("estat hettest needs a positive residual variance")
    squared = resid**2 / scale

    # Choose the auxiliary regressors.
    slope_columns = [j for j in range(len(names)) if not _is_constant(names[j])]
    used_names: list[str] = []
    if variables is None:
        design_columns = slope_columns
        used_names = [names[j] for j in slope_columns]
    else:
        requested = [str(v) for v in variables]
        if not requested:
            # No varlist: regress on the fitted values, as hettest.ado does.
            coefficients = np.asarray(result.params, dtype=float).reshape(-1)
            if coefficients.size != matrix.shape[1]:
                raise ValueError(
                    "estat hettest cannot rebuild the fitted values: "
                    f"{coefficients.size} coefficients for "
                    f"{matrix.shape[1]} design columns"
                )
            design_columns = None
            used_names = []
        else:
            lookup = {names[j]: j for j in range(len(names))}
            missing = [v for v in requested if v not in lookup]
            if missing:
                raise ValueError(
                    f"estat hettest: unknown regressor(s) {missing}; "
                    f"available: {names}"
                )
            design_columns = [lookup[v] for v in requested]
            used_names = requested

    if design_columns is None:
        coefficients = np.asarray(result.params, dtype=float).reshape(-1)
        aux_design = (matrix @ coefficients).reshape(-1, 1)
    else:
        if not design_columns:
            raise ValueError("estat hettest needs at least one regressor")
        aux_design = matrix[:, design_columns]

    if mtest is not None:
        # ``hettest.ado:160-190``: ONE auxiliary regression per regressor, each
        # giving that variable's own e(mss)/2, then _mtest adjusts the family.
        import pandas as pd

        from westata.postestimation.mtest import (
            MTestResult,
            adjust_pvalues,
            mtest_statistics,
        )

        if design_columns is None:
            raise ValueError(
                "estat hettest, mtest() needs an explicit set of regressors; a "
                "varlist-less call regresses on the fitted values, which is a "
                "single column"
            )
        per_variable = matrix[:, design_columns]
        statistics, dfs, pvalues = mtest_statistics(
            squared, per_variable, used_names
        )
        method_key = str(mtest).strip().lower()
        frame = pd.DataFrame(
            {
                "chi2": statistics,
                "df": dfs,
                "p": pvalues,
            },
            index=pd.Index(list(used_names), name="Variable"),
        )
        if method_key != "noadjust":
            frame["adjustp"] = adjust_pvalues(pvalues, method_key)
        return MTestResult(frame, method_key)

    r2, mss, _rss_aux, df_m = _auxiliary_regression(squared, aux_design)
    if df_m <= 0:
        raise ValueError(
            "estat hettest needs at least one regressor in the auxiliary "
            "regression to have a non-zero degrees of freedom"
        )

    if normalized == "iid":
        statistic = float(nobs * r2)
        pvalue = float(stats.chi2.sf(statistic, df_m))
        return HetTestResult(
            method="iid",
            statistic=statistic,
            df=df_m,
            pvalue=pvalue,
            variables=used_names,
        )

    if normalized == "fstat":
        # The F form reports the auxiliary regression's own F and df_r.
        denominator = float(nobs - aux_design.shape[1] - 1)
        if denominator <= 0:
            raise ValueError(
                "estat hettest, fstat needs positive residual degrees of freedom"
            )
        f_stat = (mss / df_m) / (_rss_aux / denominator) if _rss_aux > 0 else float("inf")
        return HetTestResult(
            method="fstat",
            statistic=float(f_stat),
            df=df_m,
            pvalue=float(stats.f.sf(f_stat, df_m, denominator)),
            df_denom=denominator,
            variables=used_names,
        )

    # Default: chi2 = e(mss)/2.
    statistic = float(mss / 2.0)
    return HetTestResult(
        method="normal",
        statistic=statistic,
        df=df_m,
        pvalue=float(stats.chi2.sf(statistic, df_m)),
        variables=used_names,
    )
