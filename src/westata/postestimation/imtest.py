"""Information-matrix test: Stata's ``estat imtest``.

Transcribed from ``imtest.ado`` (StataNow 19.5, v2.1.5).  Two things are computed:

**White's original test** (``, white``)

.. code-block:: stata

    gen double `y' = `res'^2
    regress `y' `rhs' `rhs2'          // rhs2 = all centred x_i*x_j, i >= j
    return scalar chi2 = e(N) * e(r2)

**Cameron & Trivedi's decomposition** (the default)

.. code-block:: stata

    scalar `s2' = e(rss)/e(N)

    * heteroskedasticity
    gen `y' = (`res')^2 - `s2'
    gen `y2' = (`y')^2
    summ `y2'; scalar `uss' = r(sum)          // UNCENTERED sum of squares
    regress `y' `rhs' `rhs2'
    chi2_h = e(N)*(1 - e(rss)/`uss')

    * skewness  -- regressed on the plain regressors only
    gen `y' = (`res')^3 - 3*`s2'*`res'
    ... uss from y^2 ...
    regress `y' `RHS'
    chi2_s = e(N)*(1 - e(rss)/`uss')

    * kurtosis  -- regressed on a CONSTANT only
    gen `y' = (`res')^4 - 6*`s2'*(`res')^2 + 3*`s2'*`s2'
    ... uss from y^2 ...
    regress `y'
    chi2_k = e(N)*(1 - e(rss)/`uss')

    chi2_t = chi2_h + chi2_s + chi2_k

Three details decide whether the numbers match:

* ``s2`` is the **ML** residual variance ``e(rss)/e(N)``, not ``rss/(N-K)``.
* ``chi2`` uses ``1 - rss/uss`` with ``uss`` the **uncentred** sum of squares of
  the transformed residual, i.e. an uncentred R-squared.  Using the centred
  total gives a different statistic.
* The quadratic terms are **centred** (``(x_i - mean_i)(x_j - mean_j)``) before
  entering the auxiliary regression.  Centring "does not affect the
  decomposition" per the ado's own comment, but it does affect the numerics.

``imtest`` refuses weighted data and ``cluster()``, and requires ``regress``.
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from typing import Any, Sequence

import numpy as np
from scipy import stats

__all__ = ["ImTestResult", "estat_imtest"]

_CONSTANT_NAMES = frozenset({"intercept", "intercept_", "_cons", "const"})


def _is_constant(name: str) -> bool:
    return str(name).strip().lower() in _CONSTANT_NAMES


class ImTestResult:
    """Cameron & Trivedi's IM-test decomposition, optionally White's test.

    Attributes
    ----------
    heteroskedasticity, skewness, kurtosis:
        The three component chi-square statistics.
    df_heteroskedasticity, df_skewness, df_kurtosis:
        Their degrees of freedom.
    total, df_total:
        The sum of the three and its degrees of freedom.
    white, white_df, white_pvalue:
        White's original statistic, ``None`` unless requested.
    """

    __slots__ = (
        "heteroskedasticity",
        "skewness",
        "kurtosis",
        "df_heteroskedasticity",
        "df_skewness",
        "df_kurtosis",
        "total",
        "df_total",
        "white",
        "white_df",
        "white_pvalue",
    )

    def __init__(
        self,
        *,
        heteroskedasticity: float,
        skewness: float,
        kurtosis: float,
        df_heteroskedasticity: float,
        df_skewness: float,
        df_kurtosis: float,
        white: float | None = None,
        white_df: float | None = None,
    ) -> None:
        self.heteroskedasticity = float(heteroskedasticity)
        self.skewness = float(skewness)
        self.kurtosis = float(kurtosis)
        self.df_heteroskedasticity = float(df_heteroskedasticity)
        self.df_skewness = float(df_skewness)
        self.df_kurtosis = float(df_kurtosis)
        self.total = float(heteroskedasticity + skewness + kurtosis)
        self.df_total = float(
            df_heteroskedasticity + df_skewness + df_kurtosis
        )
        self.white = None if white is None else float(white)
        self.white_df = None if white_df is None else float(white_df)
        self.white_pvalue = (
            None
            if self.white is None
            else float(stats.chi2.sf(self.white, self.white_df))
        )

    def _pvalue(self, statistic: float, df: float) -> float:
        return float(stats.chi2.sf(statistic, df)) if df > 0 else float("nan")

    def __repr__(self) -> str:
        return (
            f"<ImTestResult total chi2({self.df_total:g})={self.total:.4f} "
            f"p={self._pvalue(self.total, self.df_total):.4f}>"
        )

    def __str__(self) -> str:
        lines = []
        if self.white is not None:
            lines.extend(
                [
                    "",
                    "White's test",
                    "H0: Homoskedasticity",
                    "Ha: Unrestricted heteroskedasticity",
                    "",
                    f"{'chi2(' + f'{self.white_df:g}' + ')':<20}"
                    f" = {self.white:>8.2f}",
                    f"{'Prob > chi2':<20} = {self.white_pvalue:>8.4f}",
                ]
            )
        lines.extend(
            [
                "",
                "Cameron & Trivedi's decomposition of IM-test",
                "",
                f"{'Source':>20} |{'chi2':>11}{'df':>7}{'p':>11}",
                "-" * 21 + "+" + "-" * 28,
            ]
        )
        rows = (
            ("Heteroskedasticity", self.heteroskedasticity, self.df_heteroskedasticity),
            ("Skewness", self.skewness, self.df_skewness),
            ("Kurtosis", self.kurtosis, self.df_kurtosis),
        )
        for label, statistic, df in rows:
            lines.append(
                f"{label:>20} |{statistic:>11.2f}{df:>7.0f}"
                f"{self._pvalue(statistic, df):>11.4f}"
            )
        lines.append("-" * 21 + "+" + "-" * 28)
        lines.append(
            f"{'Total':>20} |{self.total:>11.2f}{self.df_total:>7.0f}"
            f"{self._pvalue(self.total, self.df_total):>11.4f}"
        )
        lines.append("-" * 21 + "+" + "-" * 28)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, float]:
        """Stata's ``r()`` scalars."""

        payload = {
            "chi2_h": self.heteroskedasticity,
            "df_h": self.df_heteroskedasticity,
            "chi2_s": self.skewness,
            "df_s": self.df_skewness,
            "chi2_k": self.kurtosis,
            "df_k": self.df_kurtosis,
            "chi2_t": self.total,
            "df_t": self.df_total,
        }
        if self.white is not None:
            payload["chi2"] = self.white
            payload["df"] = float(self.white_df)
        return payload


def _uncentred_chi2(y: np.ndarray, design: np.ndarray) -> tuple[float, float]:
    """``N*(1 - rss/uss)`` and the design's slope count.

    ``uss`` is the UNCENTRED sum of squares of ``y``, matching
    ``summarize y2; scalar uss = r(sum)`` in the ado.
    """

    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coefficients
    rss = float(resid @ resid)
    uss = float(y @ y)
    df = float(design.shape[1] - 1)  # slopes only, as e(df_m)
    if uss <= 0.0:
        return float("nan"), df
    return float(y.size * (1.0 - rss / uss)), df


def estat_imtest(
    result: Any,
    *,
    white: bool = False,
    design: Any = None,
) -> ImTestResult:
    """Stata's ``estat imtest`` after ``regress``.

    Parameters
    ----------
    result:
        A fitted linear result carrying the residuals and ``e(rss)``/``e(N)``.
    white:
        Also compute White's original test (Stata's ``, white``).
    design:
        Optional design matrix including the constant.  Defaults to
        ``result.design_matrix``.

    Raises
    ------
    NotImplementedError
        For a model other than ``regress``.
    ValueError
        For weighted data, a clustered VCE, a model with no covariates, or when
        the design was not retained.

    Notes
    -----
    Weights and ``cluster()`` are refused because ``imtest.ado:15-24`` refuses
    them; there the statistic would be computed from the unweighted residual
    geometry and mean something else.
    """

    model_name = str(getattr(result, "model_name", "") or "").strip().lower()
    if model_name != "regress":
        raise NotImplementedError(
            "imtest is only possible after regress: imtest.ado:13-16 rejects "
            f"every other e(cmd); got {model_name or 'an unnamed model'!r}"
        )

    cov_type = str(getattr(result, "cov_type", "nonrobust")).strip().lower()
    if cov_type in ("hc0", "hc1", "hc2", "hc3", "robust") or "cluster" in cov_type:
        raise ValueError(
            "imtest does not support weighted or clustered estimation: "
            f"imtest.ado:17-24 rejects e(wtype) and e(clustvar); got {cov_type}"
        )

    if design is None:
        design = getattr(result, "design_matrix", None)
    if design is None:
        raise ValueError(
            "estat imtest needs the estimation design, which this result did "
            "not retain; refit with a from_formula constructor"
        )
    matrix_raw, raw_names = design
    matrix = np.asarray(matrix_raw, dtype=float)
    names = [str(name) for name in raw_names]

    resid = np.asarray(result.resid, dtype=float).reshape(-1)
    n = resid.size
    if matrix.shape[0] != n:
        raise ValueError(
            f"the design has {matrix.shape[0]} rows but there are {n} residuals"
        )

    slope_columns = [j for j in range(len(names)) if not _is_constant(names[j])]
    if not slope_columns:
        raise ValueError("(imtest not allowed without covariates)")

    slopes = matrix[:, slope_columns]
    has_constant = len(slope_columns) != len(names)

    def _with_constant(block: np.ndarray) -> np.ndarray:
        if has_constant:
            return np.column_stack([np.ones(n), block])
        return block

    # rhs2: centred cross-products (x_i - mean_i)(x_j - mean_j) for j <= i.
    centred = slopes - slopes.mean(axis=0)
    columns = []
    for i, j in combinations_with_replacement(range(centred.shape[1]), 2):
        columns.append(centred[:, i] * centred[:, j])
    rhs2 = np.column_stack(columns) if columns else np.empty((n, 0))

    # s2 is the ML residual variance.
    s2 = float(resid @ resid) / n

    # --- White's original test -------------------------------------------
    white_stat = None
    white_df = None
    if white:
        target = resid**2
        design_full = _with_constant(np.column_stack([slopes, rhs2]))
        coefficients, *_ = np.linalg.lstsq(design_full, target, rcond=None)
        fitted = design_full @ coefficients
        rss = float((target - fitted) @ (target - fitted))
        centred_total = float((target - target.mean()) @ (target - target.mean()))
        r2 = 1.0 - rss / centred_total if centred_total > 0 else float("nan")
        white_stat = float(n * r2)
        white_df = float(design_full.shape[1] - 1)

    # --- decomposition ----------------------------------------------------
    # heteroskedasticity: regress on slopes AND their cross-products
    y_h = resid**2 - s2
    chi2_h, df_h = _uncentred_chi2(
        y_h, _with_constant(np.column_stack([slopes, rhs2]))
    )

    # skewness: regress on the plain regressors
    y_s = resid**3 - 3.0 * s2 * resid
    chi2_s, df_s = _uncentred_chi2(y_s, _with_constant(slopes))

    # kurtosis: regress on a constant only
    y_k = resid**4 - 6.0 * s2 * resid**2 + 3.0 * s2 * s2
    if has_constant:
        design_k = np.ones((n, 1))
    else:  # pragma: no cover - a model without a constant still needs an intercept
        design_k = np.ones((n, 1))
    chi2_k, _ = _uncentred_chi2(y_k, design_k)

    return ImTestResult(
        heteroskedasticity=chi2_h,
        skewness=chi2_s,
        kurtosis=chi2_k,
        df_heteroskedasticity=df_h,
        df_skewness=df_s,
        df_kurtosis=1.0,
        white=white_stat,
        white_df=white_df,
    )
