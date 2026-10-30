"""``estat auc`` -- the area under the ROC curve, after ``logit``/``probit``.

``logit_estat.ado:19-21`` dispatches the subcommand to ``lroc, nograph``:

    else if `"`key'"' == bsubstr("auc",1,max(3,`lkey')) {
        CheckForBad `rest'
        lroc, nograph, `rest'
    }

and ``lroc.ado:47-49`` computes the area by trapezoid integration of the ROC curve:

    replace `p' = sum((`spec' - `spec'[_n-1]) * (`sens' + `sens'[_n-1]))
    return scalar area = `p'[_N]/2

That integral is identically the Mann-Whitney U statistic divided by the product of the
two group sizes, and it is computed in that form here: the rank expression is
numerically stable, needs no sort-and-trapezoid bookkeeping, and can be checked against
``scipy.stats.mannwhitneyu`` instead of against itself.

Ties matter.  ``lroc`` iterates over distinct PREDICTED PROBABILITIES, so observations
sharing a predicted value form one point of the curve.  The average-rank form used here
handles that automatically -- which is why the two agree exactly rather than only when
the predictions are distinct.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["AucResult", "estat_auc"]

#: Model names whose ``estat`` program dispatches ``auc`` to ``lroc``.
#: ``probit.ado:259`` records ``e(estat_cmd) = "probit_estat"``, which forwards to the
#: same machinery; ``ologit.ado`` sets no ``estat_cmd`` and has no ``auc`` section.
_AUC_MODELS = ("logit", "probit")


class AucResult:
    """The ``estat auc`` output.

    Attributes
    ----------
    area:
        The area under the ROC curve.  ``0.5`` means the model ranks no better than
        chance; ``1.0`` is perfect separation.
    n:
        The number of observations the curve was built on.
    """

    __slots__ = ("area", "n")

    def __init__(self, *, area: float, n: int) -> None:
        self.area = float(area)
        self.n = int(n)

    def __str__(self) -> str:
        # ``lroc.ado:52-53``: local area : di %6.4f return(area)
        #                   local note `"Area under ROC curve = `area'"'
        return f"Area under ROC curve = {self.area:6.4f}"

    def to_dict(self) -> dict[str, float]:
        """Flatten to the ``r()`` names ``lroc`` returns."""

        return {"area": self.area, "N": float(self.n)}


def estat_auc(result: Any) -> AucResult:
    """Area under the ROC curve for a binary-outcome fit.

    Parameters
    ----------
    result:
        A fitted ``logit``/``probit`` result.

    Returns
    -------
    AucResult
        ``str(...)`` prints Stata's one-line note; ``.to_dict()`` gives ``r(area)`` /
        ``r(N)``.
    """

    model_name = str(getattr(result, "model_name", "") or "").strip().lower()
    if model_name not in _AUC_MODELS:
        raise NotImplementedError(
            "estat auc is only available after logit or probit: Stata dispatches it "
            "from logit_estat.ado:19-21 (which serves both) to lroc, and ologit sets "
            f"no e(estat_cmd) at all; got a {model_name or 'non-binary'} result"
        )

    fitted = np.asarray(result.fittedvalues, dtype=float).reshape(-1)
    residual = np.asarray(result.resid, dtype=float).reshape(-1)
    observed = fitted + residual
    outcome = (observed != 0).astype(int)

    positive = fitted[outcome == 1]
    negative = fitted[outcome == 0]
    if positive.size == 0 or negative.size == 0:
        raise ValueError(
            "estat auc needs both outcomes present: the ROC curve is undefined when "
            f"only one class occurs ({int(positive.size)} positive, "
            f"{int(negative.size)} negative)"
        )

    # Average ranks handle ties the way the curve does: observations sharing a
    # predicted probability form a single ROC point, so they must not be ordered
    # arbitrarily among themselves.
    ranks = _average_ranks(np.concatenate([positive, negative]))
    positive_rank_sum = float(np.sum(ranks[: positive.size]))
    u_statistic = positive_rank_sum - positive.size * (positive.size + 1) / 2.0
    area = u_statistic / (positive.size * negative.size)

    return AucResult(area=float(area), n=int(outcome.size))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties replaced by their average (1-based)."""

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)

    position = 0
    while position < sorted_values.size:
        end = position
        while end + 1 < sorted_values.size and sorted_values[end + 1] == sorted_values[position]:
            end += 1
        # 1-based average rank for the run [position, end].
        average = (position + end) / 2.0 + 1.0
        ranks[order[position : end + 1]] = average
        position = end + 1
    return ranks
