"""Classification table: Stata's ``estat classification``.

Transcribed from ``lstat.ado`` (StataNow 19.5, v3.2.0), which is what
``logit_estat`` forwards ``classification`` to.

The whole procedure is a 2x2 table of observed against predicted, with the
prediction thresholded at ``cutoff()`` (default 0.5):

.. code-block:: stata

    summ `touse' [fw=`w'] if `y'!=0 & `p'>=`cutoff' & `touse'
    local a = round(r(sum_w),1)      // true  positive
    summ `touse' [fw=`w'] if `y'==0 & `p'>=`cutoff' & `touse'
    local b = round(r(sum_w),1)      // false positive
    summ `touse' [fw=`w'] if `y'!=0 & `p'< `cutoff' & `touse'
    local c = round(r(sum_w),1)      // false negative
    summ `touse' [fw=`w'] if `y'==0 & `p'< `cutoff' & `touse'
    local d = round(r(sum_w),1)      // true  negative

    P_corr = ((a+d)/(a+b+c+d))*100    // correctly classified
    P_p1   = (a/(a+c))*100            // sensitivity      Pr( +| D)
    P_n0   = (d/(b+d))*100            // specificity      Pr( -|~D)
    P_p0   = (b/(b+d))*100            // false +  given ~D
    P_n1   = (c/(a+c))*100            // false -  given  D
    P_1p   = (a/(a+b))*100            // positive predictive value  Pr( D| +)
    P_0n   = (d/(c+d))*100            // negative predictive value  Pr(~D| -)
    P_0p   = (b/(a+b))*100            // false +  given classified +
    P_1n   = (c/(c+d))*100            // false -  given classified -

Note "True D defined as `y' != 0": the positive class is the non-zero outcome,
which matters for a factor-coded response as well as a 0/1 one.

The percentages are reported on a 0-100 scale (Stata multiplies by 100 and prints
``%``), so that is what this module returns.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = ["ClassificationResult", "estat_classification"]

#: Commands Stata's ``logit_estat`` routes ``classification`` from.
_SUPPORTED = frozenset({"logit", "probit", "logistic"})


class ClassificationResult:
    """The ``estat classification`` table and its nine percentages.

    Attributes
    ----------
    table:
        3x3 DataFrame: rows ``+``/``-``/``Total``, columns ``D``/``~D``/``Total``.
    cutoff:
        The probability threshold used (Stata's ``cutoff()``).
    percent_correct, sensitivity, specificity:
        Stata's ``r(P_corr)``, ``r(P_p1)``, ``r(P_n0)``.
    false_positive_rate, false_negative_rate:
        ``r(P_p0)`` (given true ~D) and ``r(P_n1)`` (given true D).
    positive_predictive_value, negative_predictive_value:
        ``r(P_1p)`` and ``r(P_0n)``.
    false_positive_classified, false_negative_classified:
        ``r(P_0p)`` and ``r(P_1n)``, the rates given the *classified* class.
    """

    __slots__ = (
        "table",
        "cutoff",
        "percent_correct",
        "sensitivity",
        "specificity",
        "false_positive_rate",
        "false_negative_rate",
        "positive_predictive_value",
        "negative_predictive_value",
        "false_positive_classified",
        "false_negative_classified",
    )

    def __init__(
        self,
        *,
        table: pd.DataFrame,
        cutoff: float,
        percent_correct: float,
        sensitivity: float,
        specificity: float,
        false_positive_rate: float,
        false_negative_rate: float,
        positive_predictive_value: float,
        negative_predictive_value: float,
        false_positive_classified: float,
        false_negative_classified: float,
    ) -> None:
        self.table = table
        self.cutoff = float(cutoff)
        self.percent_correct = float(percent_correct)
        self.sensitivity = float(sensitivity)
        self.specificity = float(specificity)
        self.false_positive_rate = float(false_positive_rate)
        self.false_negative_rate = float(false_negative_rate)
        self.positive_predictive_value = float(positive_predictive_value)
        self.negative_predictive_value = float(negative_predictive_value)
        self.false_positive_classified = float(false_positive_classified)
        self.false_negative_classified = float(false_negative_classified)

    def __repr__(self) -> str:
        return (
            f"<ClassificationResult cutoff={self.cutoff:g} "
            f"correct={self.percent_correct:.2f}% "
            f"sens={self.sensitivity:.2f}% spec={self.specificity:.2f}%>"
        )

    def __str__(self) -> str:
        t = self.table.to_numpy()
        a, b, ab = t[0]
        c, d, cd = t[1]
        ac_total, bd_total, total = t[2]
        lines = [
            "",
            f"Classified + if predicted Pr(D) >= {self.cutoff:g}",
            "True D defined as the response != 0",
            "-" * 50,
            f"{'':<9}{'|':>4}{'D':>10}{'~D':>10}{'Total':>10}",
            "-" * 50,
            f"{'Classified +':<12}{a:>10.0f}{b:>10.0f}{ab:>10.0f}",
            f"{'Classified -':<12}{c:>10.0f}{d:>10.0f}{cd:>10.0f}",
            "-" * 50,
            f"{'Total':<12}{ac_total:>10.0f}{bd_total:>10.0f}{total:>10.0f}",
            "-" * 50,
            f"{'Sensitivity':<38}{'Pr( +| D)':>22}{self.sensitivity:>8.2f}%",
            f"{'Specificity':<38}{'Pr( -|~D)':>22}{self.specificity:>8.2f}%",
            f"{'Positive predictive value':<38}{'Pr( D| +)':>22}"
            f"{self.positive_predictive_value:>8.2f}%",
            f"{'Negative predictive value':<38}{'Pr(~D| -)':>22}"
            f"{self.negative_predictive_value:>8.2f}%",
            "-" * 50,
            f"{'False + rate for true ~D':<38}{'Pr( +|~D)':>22}"
            f"{self.false_positive_rate:>8.2f}%",
            f"{'False - rate for true D':<38}{'Pr( -| D)':>22}"
            f"{self.false_negative_rate:>8.2f}%",
            f"{'False + rate for classified +':<38}{'Pr(~D| +)':>22}"
            f"{self.false_positive_classified:>8.2f}%",
            f"{'False - rate for classified -':<38}{'Pr( D| -)':>22}"
            f"{self.false_negative_classified:>8.2f}%",
            "-" * 50,
            f"{'Correctly classified':<60}{self.percent_correct:>8.2f}%",
            "-" * 50,
        ]
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """The 3x3 classification table."""

        return self.table

    def to_dict(self) -> dict[str, float]:
        """Stata's ``r()`` scalars."""

        return {
            "P_corr": self.percent_correct,
            "P_p1": self.sensitivity,
            "P_n0": self.specificity,
            "P_p0": self.false_positive_rate,
            "P_n1": self.false_negative_rate,
            "P_1p": self.positive_predictive_value,
            "P_0n": self.negative_predictive_value,
            "P_0p": self.false_positive_classified,
            "P_1n": self.false_negative_classified,
        }


def _percent(numerator: float, denominator: float) -> float:
    """``(num/den)*100``, NaN when the denominator is empty."""

    if denominator == 0:
        return float("nan")
    return float(numerator / denominator * 100.0)


def estat_classification(
    result: Any,
    *,
    cutoff: float = 0.5,
    y: Any = None,
    probabilities: Any = None,
) -> ClassificationResult:
    """Stata's ``estat classification`` after ``logit``/``probit``.

    Parameters
    ----------
    result:
        A fitted binary-outcome result.
    cutoff:
        The probability threshold (Stata's ``cutoff()``); must lie in [0, 1].
    y:
        The observed binary response.  When omitted it is rebuilt from the
        fitted values and residuals.
    probabilities:
        The predicted probability of the positive class.  When omitted,
        ``result.predict(kind="pr")`` is used.

    Raises
    ------
    NotImplementedError
        For a model Stata does not route ``estat classification`` from.
    ValueError
        For a cutoff outside [0, 1], or when the response is not binary.
    """

    model_name = str(getattr(result, "model_name", "") or "").strip().lower()
    if model_name not in _SUPPORTED:
        raise NotImplementedError(
            "estat classification is only available for binary-outcome models: "
            "Stata routes it through logit_estat's lstat, which is reachable "
            f"after logit/probit; got {model_name or 'an unnamed model'!r}"
        )

    threshold = float(cutoff)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"cutoff must be between 0 and 1; got {cutoff!r}"
        )

    # Predicted probabilities of the positive class.
    if probabilities is None:
        try:
            probabilities = result.predict(kind="pr")
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(
                "estat classification needs the predicted probabilities; pass "
                "probabilities= explicitly"
            ) from exc
    p = np.asarray(probabilities, dtype=float).reshape(-1)

    # Observed response.
    if y is None:
        fitted = np.asarray(result.fittedvalues, dtype=float).reshape(-1)
        resid = np.asarray(result.resid, dtype=float).reshape(-1)
        if fitted.size != p.size or resid.size != p.size:
            raise ValueError(
                "estat classification: the residual and prediction vectors do "
                "not align; pass y= explicitly"
            )
        y = fitted + resid
    y = np.asarray(y, dtype=float).reshape(-1)
    if y.size != p.size:
        raise ValueError(
            f"estat classification: y has {y.size} entries but the predicted "
            f"probabilities have {p.size}"
        )

    # Stata: "True D defined as `y' != 0" -- the positive class is non-zero.
    observed_positive = y != 0
    classified_positive = p >= threshold

    a = float(np.sum(observed_positive & classified_positive))
    b = float(np.sum(~observed_positive & classified_positive))
    c = float(np.sum(observed_positive & ~classified_positive))
    d = float(np.sum(~observed_positive & ~classified_positive))

    table = pd.DataFrame(
        [[a, b, a + b], [c, d, c + d], [a + c, b + d, a + b + c + d]],
        index=pd.Index(["+", "-", "Total"], name="Classified"),
        columns=pd.Index(["D", "~D", "Total"], name="True"),
    )

    return ClassificationResult(
        table=table,
        cutoff=threshold,
        percent_correct=_percent(a + d, a + b + c + d),
        sensitivity=_percent(a, a + c),
        specificity=_percent(d, b + d),
        false_positive_rate=_percent(b, b + d),
        false_negative_rate=_percent(c, a + c),
        positive_predictive_value=_percent(a, a + b),
        negative_predictive_value=_percent(d, c + d),
        false_positive_classified=_percent(b, a + b),
        false_negative_classified=_percent(c, c + d),
    )
