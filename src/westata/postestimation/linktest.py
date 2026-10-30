"""``linktest`` -- Pregibon's link test for a specification error.

``logit_postestimation.sthlp`` lists three postestimation commands after ``logit``:
``estat classification``, ``estat gof`` and ``linktest``. The first two are implemented;
``linktest`` was not.

The procedure is from ``linktest.ado``, which is readable source:

    quietly _predict `hat' if `touse', xb        // linear index of the fitted model
    quietly gen `hat2'=`hat'^2
    quietly `nxtcmd' `lhs' _hat _hatsq ...      // REFIT with both added
    ret scalar t  = _b[_hatsq]/_se[_hatsq]
    ret scalar df = e(df_r)

So the test refits the SAME model with the linear prediction and its square added, and
reports the t statistic of the squared term. A significant ``_hatsq`` means the linear
index is mis-specified -- the functional form is wrong, not just the fit.

``linktest.ado:52-66`` lists the commands it refuses outright; the ones this package
implements from that list (``xt*``, ``sureg``, ``mvreg``, ``eivreg``, ``zip``, ``zinb``,
``iv*``) are refused here too, because their linear index is not a single ``xb``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


@dataclass
class LinkTestResult:
    """Pregibon's link test.

    Attributes
    ----------
    t_statistic:
        ``b[_hatsq] / se[_hatsq]`` from the refitted model (``linktest.ado:131``).
    df:
        Residual degrees of freedom of the refit (``linktest.ado:132``).
    pvalue:
        Two-sided ``P > |t|``.
    hat_coefficient, hatsq_coefficient:
        The two added coefficients.
    """

    t_statistic: float
    df: float
    pvalue: float
    hat_coefficient: float
    hatsq_coefficient: float

    def __str__(self) -> str:
        verdict = (
            "the link is specified correctly"
            if self.pvalue > 0.05
            else "the link is NOT specified correctly"
        )
        return (
            "\nPregibon's link test\n"
            f"H0: {verdict}\n\n"
            f"  _hat        = {self.hat_coefficient:9.6f}\n"
            f"  _hatsq      = {self.hatsq_coefficient:9.6f}\n"
            f"  t           = {self.t_statistic:9.4f}\n"
            f"  P>|t|       = {self.pvalue:9.4f}   (df = {self.df:g})"
        )

    def to_dict(self) -> dict[str, float]:
        """Flatten to ``r(t)`` and ``r(df)`` (``linktest.ado:131-132``).

        Stata stores only those two scalars; the p-value is a convenience this package
        adds, and it is named ``p`` rather than ``r(p)`` because Stata does not return it.
        """

        return {"t": float(self.t_statistic), "df": float(self.df), "p": float(self.pvalue)}


#: ``linktest.ado:52-66`` refuses these after fitting.  Only the ones this package can
#: produce are listed; the rest are unreachable here.
_REFUSED = frozenset(
    {
        "rreg",
        "zip",
        "zinb",
        "sureg",
        "mvreg",
        "manova",
        "eivreg",
    }
)


def _next_command(model_name: str) -> str:
    """``linktest.ado:93-104``: which estimator refits the augmented equation."""

    if model_name in ("regress", "anova", "areg", "reghdfe"):
        return "regress"
    if model_name == "logistic":
        return "logit"
    if model_name == "probit" or model_name == "dprobit":
        return "probit"
    return model_name


def _refit(result: Any, frame: pd.DataFrame, response: str, added: list[str]) -> Any:
    """Refit the same estimator on the augmented frame."""

    from westata import estimators as _estimators  # local import keeps startup light

    model_name = str(getattr(result, "model_name", "") or "").strip().lower()
    if model_name in ("regress",):
        from westata.estimators.linear import OLS

        formula = f"{response} ~ " + " + ".join(added)
        return OLS.from_formula(formula, frame).fit()
    if model_name in ("logit", "logistic"):
        from westata.estimators.discrete import Logit

        formula = f"{response} ~ " + " + ".join(added)
        return Logit.from_formula(formula, frame).fit()
    if model_name in ("probit", "dprobit"):
        from westata.estimators.discrete import Probit

        formula = f"{response} ~ " + " + ".join(added)
        return Probit.from_formula(formula, frame).fit()
    if model_name == "poisson":
        # ``Poisson`` lives in ``discrete``, not ``counts`` -- the count module holds the
        # zero-inflated and truncated extensions.
        from westata.estimators.discrete import Poisson

        formula = f"{response} ~ " + " + ".join(added)
        return Poisson.from_formula(formula, frame).fit()
    raise NotImplementedError(        "linktest: cannot rebuild the augmented equation for "
        f"{model_name!r}; linktest refits the SAME estimator with _hat and _hatsq added, "
        "and this mapping is not implemented for it"
    )


def linktest(result: Any) -> LinkTestResult:
    """Stata's ``linktest``.

    Parameters
    ----------
    result:
        A fitted result whose linear index is a single ``xb``.

    Raises
    ------
    NotImplementedError
        For an estimator ``linktest.ado:52-66`` refuses, or one this package cannot
        refit.
    ValueError
        When the estimation data was not retained, or the refit cannot produce a
        ``_hatsq`` coefficient.

    Notes
    -----
    The test is two-sided against a t distribution with the REFIT's residual degrees of
    freedom -- ``linktest.ado:132`` reads ``e(df_r)`` after the refit, not before.
    """

    model_name = str(getattr(result, "model_name", "") or "").strip().lower()
    if model_name in _REFUSED or model_name.startswith("iv") or model_name.startswith("xt"):
        raise NotImplementedError(
            f"linktest is not possible after {model_name} (linktest.ado:52-66): its "
            "linear index is not a single xb, so _hat and _hatsq are not defined"
        )

    frame = getattr(result, "model_data", None)
    if frame is None:
        raise ValueError(
            "linktest needs the estimation-sample data, which this result did not "
            "retain; refit with the formula interface"
        )
    response = str(getattr(result, "response_name", "") or "")
    if not response:
        raise ValueError("linktest: the result records no response name")

    frame = pd.DataFrame(frame).reset_index(drop=True)
    linear_index = np.asarray(result.predict(kind="xb"), dtype=float).reshape(-1)
    if linear_index.size != len(frame):
        raise ValueError(
            "linktest: the linear index has "
            f"{linear_index.size} entries for {len(frame)} rows"
        )

    # ``linktest.ado:106-108``: names must be free; the augmented frame uses the same
    # two names Stata does.
    augmented = frame.copy()
    augmented["_hat"] = linear_index
    augmented["_hatsq"] = linear_index ** 2

    refit = _refit(result, augmented, response, ["_hat", "_hatsq"])
    names = [str(name) for name in refit.param_names]
    if "_hatsq" not in names:
        raise ValueError(
            "linktest: the refit did not produce an _hatsq coefficient; available "
            f"parameters: {names}"
        )

    position = names.index("_hatsq")
    coefficients = np.asarray(refit.params, dtype=float)
    standard_errors = np.asarray(refit.bse, dtype=float)
    if not np.isfinite(standard_errors[position]) or standard_errors[position] == 0.0:
        raise ValueError(
            "linktest: _hatsq has a zero or non-finite standard error, so its t "
            "statistic is undefined (the augmented model may be collinear)"
        )
    hatsq_coefficient = float(coefficients[position])
    t_statistic = hatsq_coefficient / float(standard_errors[position])
    degrees_of_freedom = float(refit.df_resid)
    pvalue = (
        float(2.0 * scipy_stats.t.sf(abs(t_statistic), degrees_of_freedom))
        if degrees_of_freedom > 0.0
        else float("nan")
    )

    hat_position = names.index("_hat") if "_hat" in names else None
    return LinkTestResult(
        t_statistic=float(t_statistic),
        df=degrees_of_freedom,
        pvalue=pvalue,
        hat_coefficient=(
            float(coefficients[hat_position]) if hat_position is not None else float("nan")
        ),
        hatsq_coefficient=hatsq_coefficient,
    )
