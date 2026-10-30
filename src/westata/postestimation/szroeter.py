"""``estat szroeter`` -- Szroeter's rank test for heteroskedasticity.

``regress_postestimation.sthlp:519-521``:

    estat szroeter performs Szroeter's rank test for heteroskedasticity for each of the
    variables in varlist or for the explanatory variables of the regression if rhs is
    specified.

The statistic is closed-form and was read from ``szroeter.ado:158-178``, whose ``Szroeter``
subroutine is fully visible:

    quietly {
        egen `ht' = rank(`var') if `touse'
        summ `ht'
        scalar `mh' = r(mean)
        scalar `vh' = r(Var)
        predict `res', res
        summ `ht' [aw=`res'^2]
        scalar `h' = r(mean)
        ret scalar Q = (`n'^2 / (2*(`n'-1))) * (`h'-`mh')^2 / `vh'
        ret scalar p = chi2tail(1, return(Q))
    }

So the test ranks the variable, then asks whether the SQUARED RESIDUALS are related to
that rank: ``h`` is the residual-squared-weighted mean rank, and ``Q`` measures how far it
sits from the unweighted mean rank. The null is a constant variance; the alternative is a
variance MONOTONE in the variable, which is why ranks rather than levels are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


@dataclass
class SzroeterResult:
    """One row of ``estat szroeter``.

    Attributes
    ----------
    variable:
        The regressor or variable tested.
    statistic:
        ``Q``, chi-square with one degree of freedom.
    pvalue:
        Upper-tail probability.
    adjusted_pvalue:
        The multiple-testing adjusted value, when ``mtest`` was requested.
    """

    variable: str
    statistic: float
    pvalue: float
    adjusted_pvalue: float | None = None

    def __str__(self) -> str:
        rows = [
            "",
            "Szroeter's test for homoskedasticity",
            "H0: Variance constant",
            f"Ha: Variance monotonic in {self.variable}",
            "",
        ]
        if self.adjusted_pvalue is None:
            rows.append(
                f"  chi2(1) = {self.statistic:9.4f}    Prob > chi2 = {self.pvalue:.4f}"
            )
        else:
            rows.append(
                f"  chi2(1) = {self.statistic:9.4f}    Prob > chi2 = {self.pvalue:.4f}"
            )
            rows.append(f"  adjusted p-value = {self.adjusted_pvalue:.4f}")
        return "\n".join(rows)

    def to_dict(self) -> dict[str, float]:
        """Flatten to the ``r()`` names Stata returns for a single-variable test.

        ``szroeter.ado:141-143``: ``r(chi2)``/``r(df)``/``r(p)`` when one variable is
        tested.  With several, Stata returns an ``r(mtest)`` matrix instead, which is
        represented by :class:`SzroeterTable`.
        """

        table = {
            "chi2": float(self.statistic),
            "df": 1.0,
            "p": float(self.pvalue),
        }
        if self.adjusted_pvalue is not None:
            table["p_adjusted"] = float(self.adjusted_pvalue)
        return table


@dataclass
class SzroeterTable:
    """The multi-variable form: one row per variable plus the adjustment method."""

    rows: list[SzroeterResult]
    method: str = "noadjust"

    def __iter__(self):
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def __str__(self) -> str:
        header = (
            f"  {'Variable':<14}  {'chi2':>9}  {'df':>4}  {'p':>8}"
        )
        lines = [
            "",
            "Szroeter's test for homoskedasticity",
            "H0: Variance constant",
            "Ha: Variance monotonic in variables",
            "",
            header,
            "  " + "-" * (len(header) - 2),
        ]
        for row in self.rows:
            p_value = (
                row.pvalue if row.adjusted_pvalue is None else row.adjusted_pvalue
            )
            lines.append(
                f"  {row.variable:<14}{row.statistic:>11.4f}{1.0:>6.0f}{p_value:>10.4f}"
            )
        lines.append("  " + "-" * (len(header) - 2))
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """The table as a DataFrame indexed by variable."""

        return pd.DataFrame(
            {
                "chi2": [row.statistic for row in self.rows],
                "df": [1.0 for row in self.rows],
                "p": [row.pvalue for row in self.rows],
                "p_adjusted": [
                    row.pvalue if row.adjusted_pvalue is None else row.adjusted_pvalue
                    for row in self.rows
                ],
            },
            index=[row.variable for row in self.rows],
        )

    def to_dict(self) -> dict[str, float]:
        """``r(mtest)`` is a matrix, so this flattens the p-value column."""

        return {
            f"p_{row.variable}": (
                row.pvalue if row.adjusted_pvalue is None else row.adjusted_pvalue
            )
            for row in self.rows
        }


def _apply_adjustment(pvalues: Sequence[float], method: str) -> list[float]:
    """Stata's ``mtest()`` adjustments (``regress_postestimation.sthlp:537-540``)."""

    normalized = method.lower()
    values = np.asarray(list(pvalues), dtype=float)
    count = values.size
    if count == 0:
        return []
    if normalized in ("noadjust", "none"):
        return [float(value) for value in values]
    if normalized == "bonferroni":
        return [float(min(value * count, 1.0)) for value in values]
    if normalized == "sidak":
        # ``1 - (1 - p)^m``; Stata clips at 1.
        return [
            float(min(1.0 - (1.0 - value) ** count, 1.0)) for value in values
        ]
    if normalized == "holm":
        # Holm's step-down: sort ascending, multiply the i-th by (m - i), then enforce
        # monotonicity so an adjusted value never falls below the previous one.
        order = np.argsort(values)
        adjusted = np.empty(count, dtype=float)
        running = 0.0
        for position, index in enumerate(order):
            candidate = min(values[index] * (count - position), 1.0)
            running = max(running, candidate)
            adjusted[index] = running
        return [float(value) for value in adjusted]
    raise ValueError(
        "estat szroeter: mtest must be one of bonferroni, holm, sidak or noadjust; "
        f"got {method!r}"
    )


def _szroeter_statistic(
    variable: np.ndarray, residuals: np.ndarray
) -> tuple[float, float]:
    """``Q`` and its p-value, following ``szroeter.ado:158-178``."""

    count = variable.size
    if count < 2:
        raise ValueError(
            "estat szroeter: at least two observations are needed for a rank test"
        )
    ranks = scipy_stats.rankdata(variable)
    mean_rank = float(np.mean(ranks))
    # Stata's ``summ`` reports the SAMPLE variance, hence ``ddof=1``.
    variance_rank = float(np.var(ranks, ddof=1))
    weights = residuals ** 2
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError(
            "estat szroeter: every residual is zero, so the weighted mean rank is "
            "undefined"
        )
    weighted_mean = float(np.sum(weights * ranks) / total)
    statistic = (
        (count ** 2) / (2.0 * (count - 1))
    ) * ((weighted_mean - mean_rank) ** 2) / variance_rank
    return float(statistic), float(scipy_stats.chi2.sf(statistic, 1))


def estat_szroeter(
    result: Any,
    variables: Sequence[str] | None = None,
    *,
    rhs: bool = False,
    mtest: str | None = None,
) -> SzroeterResult | SzroeterTable:
    """Stata's ``estat szroeter``.

    Parameters
    ----------
    variables:
        Variables to test.  Either this or ``rhs`` must be given
        (``sthlp:509-510``).
    rhs:
        Test the regression's own explanatory variables (``sthlp:528-530``).
    mtest:
        Multiple-testing adjustment: ``bonferroni``, ``holm``, ``sidak`` or
        ``noadjust`` (``sthlp:533-540``).  Stata always performs multiple testing but
        adjusts nothing by default.

    Raises
    ------
    NotImplementedError
        When the fit is not a linear regression, is weighted, or clustered --
        ``szroeter.ado:13-21`` refuses all three.
    ValueError
        When neither ``variables`` nor ``rhs`` is given.

    Notes
    -----
    ``szroeter.ado:13-21``: "szroeter is only possible after regress", "does not allow
    weights", "does not allow cluster()".  Refusing here matches Stata rather than
    silently computing a statistic whose reference distribution does not apply.
    """

    model_name = str(getattr(result, "model_name", "") or "").strip().lower()
    if model_name not in ("regress", "areg"):
        raise NotImplementedError(
            "estat szroeter is only possible after regress (szroeter.ado:13-16); got "
            f"{getattr(result, 'model_name', 'an unnamed model')!r}"
        )

    weight_spec = getattr(result, "weight_spec", None)
    if weight_spec is not None and getattr(weight_spec, "weights", None) is not None:
        raise NotImplementedError(
            "estat szroeter does not allow weights (szroeter.ado:17-19)"
        )
    if "cluster" in str(getattr(result, "cov_type", "")).lower():
        raise NotImplementedError(
            "estat szroeter does not allow cluster() (szroeter.ado:20-22)"
        )

    names = [
        str(name)
        for name in getattr(result, "param_names", [])
        if str(name) not in ("Intercept", "_cons", "const")
    ]
    if rhs:
        requested = list(dict.fromkeys(list(variables or []) + names))
    else:
        requested = list(variables or [])
    if not requested:
        raise ValueError(
            "estat szroeter: you must specify either varlist or option rhs "
            "(szroeter.ado:52-56)"
        )

    frame = getattr(result, "model_data", None)
    if frame is None:
        raise ValueError(
            "estat szroeter needs the estimation-sample data, which this result did not "
            "retain; refit with the formula interface"
        )
    residuals = np.asarray(result.resid, dtype=float).reshape(-1)
    if residuals.size != len(frame):
        raise ValueError(
            "estat szroeter: the residual vector has "
            f"{residuals.size} entries for {len(frame)} rows"
        )

    rows: list[SzroeterResult] = []
    for variable in requested:
        if variable not in frame.columns:
            raise ValueError(
                f"estat szroeter: {variable!r} is not a column of the estimation data; "
                f"available: {list(frame.columns)}"
            )
        values = np.asarray(frame[variable], dtype=float)
        statistic, pvalue = _szroeter_statistic(values, residuals)
        rows.append(SzroeterResult(variable=variable, statistic=statistic, pvalue=pvalue))

    method = (mtest or "noadjust").lower()
    adjusted = _apply_adjustment([row.pvalue for row in rows], method)
    for row, value in zip(rows, adjusted):
        row.adjusted_pvalue = value

    if len(rows) == 1:
        # ``szroeter.ado:65-68``: a single variable drops the adjustment and reports the
        # scalars, so the adjusted value is not carried.
        single = rows[0]
        return SzroeterResult(
            variable=single.variable,
            statistic=single.statistic,
            pvalue=single.pvalue,
            adjusted_pvalue=None if method == "noadjust" else single.adjusted_pvalue,
        )
    return SzroeterTable(rows=rows, method=method)
