"""Information criteria: Stata's ``estat ic``.

The four criteria and their exact formulas come from ``estimates.ado``
(StataNow 19.5, lines 985-1002), which is the implementation behind ``estat ic``:

.. code-block:: stata

    // AIC  = -2*ll + 2*df
    matrix `S'[`is',5] = -2*`S'[`is',3] + 2*`S'[`is',4]

    // BIC  = -2*ll + log(N)*df
    matrix `S'[`is',6] = -2*`S'[`is',3] + log(`S'[`is',1])*`S'[`is',4]

    // AICC = -2*ll + 2*df*(N / (N - df - 1))
    matrix `S'[`is',7] = -2*`S'[`is',3] + 2 * `S'[`is',4] * ///
            (`S'[`is',1] / (`S'[`is',1] - `S'[`is',4] - 1))

    // CAIC = -2*ll + df * (log(N) + 1)
    matrix `S'[`is',8] = -2 *`S'[`is',3] + `S'[`is',4] * ///
                 (log(`S'[`is',1]) + 1)

Two details decide whether the numbers match:

* **AICc is written ``-2ll + 2·df·N/(N-df-1)``**, not the textbook
  ``AIC + 2df(df+1)/(N-df-1)``.  The two are algebraically equal
  (``2df·N/(N-df-1) = 2df + 2df(df+1)/(N-df-1)``), so either form is fine for the
  value -- but transcribing the ado keeps the derivation honest.
* **``df`` defaults to ``rank(V)``**, the rank of the coefficient covariance
  matrix, not the coefficient count.  Those coincide for a full-rank design and
  differ when the model is over-parameterised.  ``estimates.ado:955-978`` gives
  the precedence ``df(#)`` > ``e(df_ic)`` > ``rank(V)``.

``N`` is ``e(N)``, unless the model records ``e(N_ic)``, which
``ic_note.sthlp`` documents as the case count for choice models.  Both are
supported here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = ["IcResult", "estat_ic"]


class IcResult:
    """The ``estat ic`` table.

    Attributes
    ----------
    n, ll_null, ll_model, df:
        The first four columns, matching Stata's ``r(S)`` layout
        ``(N, ll(null), ll(model), df, ...)``.
    aic, bic, aicc, caic:
        The four information criteria.
    """

    __slots__ = (
        "n",
        "ll_null",
        "ll_model",
        "df",
        "aic",
        "bic",
        "aicc",
        "caic",
        "model",
    )

    def __init__(
        self,
        *,
        n: float,
        ll_null: float,
        ll_model: float,
        df: float,
        aic: float,
        bic: float,
        aicc: float,
        caic: float,
        model: str = "",
    ) -> None:
        self.n = float(n)
        self.ll_null = float(ll_null)
        self.ll_model = float(ll_model)
        self.df = float(df)
        self.aic = float(aic)
        self.bic = float(bic)
        self.aicc = float(aicc)
        self.caic = float(caic)
        #: The command name Stata prints in the ``Model`` column.  Empty when the
        #: result was built without one; the table then shows ``.`` as before.
        self.model = str(model or "")

    def __repr__(self) -> str:
        return (
            f"<IcResult N={self.n:g} df={self.df:g} "
            f"AIC={self.aic:.4f} BIC={self.bic:.4f}>"
        )

    def __str__(self) -> str:
        return self.render()

    def render(self) -> str:
        """Reproduce ``estat ic`` exactly.

        Measured against a live ``estat ic`` after ``reg consumption temp price income``::

            Akaike's information criterion and Bayesian information criterion

            -----------------------------------------------------------------------------
                   Model |          N   ll(null)  ll(model)      df        AIC        BIC
            -------------+---------------------------------------------------------------
                       . |         30   39.57876   58.61944       4  -109.2389  -103.6341
            -----------------------------------------------------------------------------
            Note: BIC uses N = number of observations. See [R] IC note.

        Three things were wrong before:

        * the table carried AICc and CAIC columns that ``estat ic`` does not print;
        * the row was labelled with the command (``regress``) where Stata prints a dot;
        * the title line and the note wording differed.

        ``aicc``/``caic`` remain available as attributes -- they are useful numbers, they
        are simply not part of this table.
        """

        title = "Akaike's information criterion and Bayesian information criterion"
        header = (
            f"{'Model':>12} |{'N':>11}{'ll(null)':>11}{'ll(model)':>11}"
            f"{'df':>8}{'AIC':>11}{'BIC':>11}"
        )
        rule = "-" * 13 + "+" + "-" * 63

        def _cell(value: float, width: int, decimals: int) -> str:
            """``.`` for a missing statistic, never ``nan``.

            ``var`` does not store ``e(ll_0)``, so ``ll(null)`` is genuinely absent and
            Stata prints ``.`` there.
            """

            if not np.isfinite(value):
                return ".".rjust(width)
            return f"{value:.{decimals}f}".rjust(width)

        row = (
            f"{'.':>12} |{self.n:>11.0f}"
            f"{_cell(self.ll_null, 11, 5)}{_cell(self.ll_model, 11, 5)}"
            f"{self.df:>8.0f}"
            f"{_cell(self.aic, 11, 4)}{_cell(self.bic, 11, 4)}"
        )
        note = "Note: BIC uses N = number of observations. See [R] IC note."
        full_rule = "-" * 13 + "+" + "-" * 63
        return chr(10).join(["", title, "", full_rule, header, full_rule, row, full_rule, note])

    def to_frame(self) -> pd.DataFrame:
        """The single row as a DataFrame with Stata's column names."""

        return pd.DataFrame(
            [
                {
                    "N": self.n,
                    "ll(null)": self.ll_null,
                    "ll(model)": self.ll_model,
                    "df": self.df,
                    "AIC": self.aic,
                    "BIC": self.bic,
                    "AICc": self.aicc,
                    "CAIC": self.caic,
                }
            ]
        )

    def to_dict(self) -> dict[str, float]:
        """Stata's ``r()`` values: the ``S`` row plus the named criteria."""

        return {
            "N": self.n,
            "ll_0": self.ll_null,
            "ll": self.ll_model,
            "df": self.df,
            "aic": self.aic,
            "bic": self.bic,
            "aicc": self.aicc,
            "caic": self.caic,
        }


def _resolve_n(result: Any, n: float | None) -> float:
    """``e(N)``, or ``e(N_ic)`` when the model records it (ic_note.sthlp)."""

    if n is not None:
        return float(n)
    e = getattr(result, "e", {})
    for key in ("N_ic", "N"):
        try:
            value = e[key]
        except (KeyError, TypeError):
            continue
        number = float(value)
        if np.isfinite(number) and number > 0:
            return number
    raise ValueError(
        "estat ic needs a sample size: the result records neither e(N_ic) nor "
        "e(N); pass n= explicitly"
    )


def _resolve_df(result: Any, df: float | None) -> float:
    """``df(#)`` > ``e(df_ic)`` > ``rank(V)`` (estimates.ado:955-978)."""

    if df is not None:
        return float(df)
    e = getattr(result, "e", {})
    try:
        recorded = e["df_ic"]
    except (KeyError, TypeError):
        recorded = None
    if recorded is not None:
        number = float(recorded)
        if np.isfinite(number):
            return number
    covariance = np.asarray(result.cov_params(), dtype=float)
    if covariance.ndim != 2 or covariance.size == 0:
        raise ValueError("estat ic needs the coefficient covariance matrix")
    return float(np.linalg.matrix_rank(covariance))


def estat_ic(
    result: Any,
    *,
    n: float | None = None,
    df: float | None = None,
) -> IcResult:
    """Stata's ``estat ic`` for one fitted model.

    Parameters
    ----------
    result:
        A fitted result carrying ``e(ll)`` (and optionally ``e(ll_0)``).
    n:
        Override ``N`` used by BIC/AICc/CAIC (Stata's ``n(#)``).
    df:
        Override the degrees-of-freedom count used by all four criteria
        (Stata's ``df(#)``).

    Raises
    ------
    ValueError
        When the model has no log-likelihood (a criterion cannot be built for a
        fit that is not maximum likelihood), or ``N``/``df`` cannot be resolved.
    """

    e = getattr(result, "e", {})
    try:
        ll_model = float(e["ll"])
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "estat ic needs a log-likelihood (e(ll)); this result has none, so "
            "no information criterion is defined for it"
        ) from exc
    if not np.isfinite(ll_model):
        raise ValueError(
            "estat ic needs a finite log-likelihood (e(ll)); this result's is "
            f"{ll_model}"
        )

    try:
        ll_null = float(e["ll_0"])
    except (KeyError, TypeError):
        ll_null = float("nan")
    if not np.isfinite(ll_null):
        ll_null = float("nan")

    sample = _resolve_n(result, n)
    parameters = _resolve_df(result, df)

    # The four formulas, verbatim from estimates.ado:985-1002.
    aic = -2.0 * ll_model + 2.0 * parameters
    bic = -2.0 * ll_model + np.log(sample) * parameters
    denominator = sample - parameters - 1.0
    aicc = (
        -2.0 * ll_model + 2.0 * parameters * (sample / denominator)
        if denominator > 0.0
        else float("nan")
    )
    caic = -2.0 * ll_model + parameters * (np.log(sample) + 1.0)

    return IcResult(
        n=sample,
        ll_null=ll_null,
        ll_model=ll_model,
        df=parameters,
        aic=float(aic),
        bic=float(bic),
        aicc=float(aicc),
        caic=float(caic),
        # ``e(cmd)`` is the command name Stata prints in the Model column.  The
        # container synthesises it from ``model_name`` when the estimator did not
        # record it, so this reads the same value a Stata user would see.
        model=str((getattr(result, "e", {}) or {}).get("cmd") or ""),
    )
