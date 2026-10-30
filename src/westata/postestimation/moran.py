"""``estat moran`` -- Moran's test for spatial correlation in the residuals.

``estat_moran.sthlp:44-46``:

    estat moran is a postestimation test that can be run after fitting a model using
    regress with spatial data.  It performs the Moran test for spatial correlation among
    the residuals.

Syntax (``sthlp:29-31``) requires a spatial weighting matrix::

    estat moran, errorlag(spmatname) [errorlag(spmatname) ...]

**A caveat that belongs in the source, not only in a report.** ``sthlp:56-59`` points at a
"Methods and formulas" section that is "not included in this help file", and there is no
``estat_moran.ado`` in ``ado/base`` -- the arithmetic lives in a compiled builtin.
Stata's ``r(chi2)`` therefore cannot be reproduced from local evidence.  What is
implemented here is the STANDARD Moran's I test (Cliff & Ord 1981; Moran 1950) under the
randomisation assumption, which is the statistic the command is named for:

    I   = (n / S0) * (e' W e) / (e' e)
    E[I] = -1 / (n - 1)

with the variance derived from the weights' first and second moments (S0, S1, S2) and the
residuals' kurtosis ``k``.  ``z = (I - E[I]) / sd(I)`` and ``chi2 = z^2`` on one degree of
freedom.

Because that last identity is an ASSUMPTION about Stata's convention rather than something
read from its source, the result object says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats as scipy_stats


@dataclass
class MoranResult:
    """Moran's I test for spatial correlation.

    Attributes
    ----------
    statistic:
        ``chi2`` -- the squared standardised ``I``, on one degree of freedom.
    df:
        Always 1 for a single error lag.
    pvalue:
        ``P > chi2(1)``.
    moran_i:
        Moran's ``I`` itself.
    expected_i:
        ``E[I] = -1/(n-1)``.
    z:
        ``(I - E[I]) / sd(I)``.
    errorlags:
        How many weighting matrices were combined.
    """

    statistic: float
    df: float
    pvalue: float
    moran_i: float
    expected_i: float
    z: float
    errorlags: int = 1

    def __str__(self) -> str:
        return (
            "\nMoran test for spatial dependence\n"
            f"H0: error terms are independent\n"
            f"Ha: error terms are spatially correlated\n\n"
            f"  chi2({self.df:g})      = {self.statistic:9.4f}\n"
            f"  Prob > chi2   = {self.pvalue:9.4f}\n\n"
            f"  Moran's I     = {self.moran_i:9.6f}\n"
            f"  E[I]          = {self.expected_i:9.6f}\n"
            f"  z             = {self.z:9.4f}"
        )

    def to_dict(self) -> dict[str, float]:
        """Flatten to ``r(chi2)``/``r(df)``/``r(p)`` (``sthlp:105-111``)."""

        return {
            "chi2": float(self.statistic),
            "df": float(self.df),
            "p": float(self.pvalue),
        }


def _as_weight_matrix(weights: Any, count: int) -> np.ndarray:
    """Validate and coerce one spatial weighting matrix."""

    matrix = np.asarray(weights, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "estat moran: each errorlag weighting matrix must be square; got shape "
            f"{matrix.shape}"
        )
    if matrix.shape[0] != count:
        raise ValueError(
            "estat moran: the weighting matrix is "
            f"{matrix.shape[0]}x{matrix.shape[1]} but the estimation sample has "
            f"{count} observations"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            "estat moran: the weighting matrix contains non-finite entries"
        )
    return matrix


def _moran_statistic(
    residuals: np.ndarray, weights: np.ndarray
) -> tuple[float, float, float]:
    """``(I, E[I], sd[I])`` under the randomisation assumption."""

    count = residuals.size
    if count < 4:
        raise ValueError(
            "estat moran: at least four observations are needed for the randomisation "
            "variance"
        )
    squared = float(residuals @ residuals)
    if squared <= 0.0:
        raise ValueError(
            "estat moran: every residual is zero, so Moran's I is undefined"
        )

    s0 = float(weights.sum())
    if s0 == 0.0:
        raise ValueError(
            "estat moran: the weighting matrix sums to zero, so Moran's I is undefined"
        )

    morans_i = (count / s0) * float(residuals @ weights @ residuals) / squared
    expected = -1.0 / (count - 1)

    # First and second moments of the weights (Cliff & Ord 1981).
    transposed = weights.T
    s1 = 0.5 * float(np.sum((weights + transposed) ** 2))
    row_sums = weights.sum(axis=1)
    column_sums = weights.sum(axis=0)
    s2 = float(np.sum((row_sums + column_sums) ** 2))

    # The residuals' kurtosis enters the randomisation variance:
    # ``k = n * sum(e^4) / (sum(e^2))^2``.
    kurtosis = count * float(np.sum(residuals ** 4)) / (squared ** 2)

    numerator = count * (
        (count ** 2 - 3 * count + 3) * s1 - count * s2 + 3 * s0 ** 2
    ) - kurtosis * (count * (count - 1) * s1 - 2 * count * s2 + 6 * s0 ** 2)
    denominator = (count - 1) * (count - 2) * (count - 3) * s0 ** 2
    variance = numerator / denominator - expected ** 2
    if variance <= 0.0:
        raise ValueError(
            "estat moran: the randomisation variance of Moran's I is not positive for "
            "this weighting matrix, so the test is undefined"
        )
    return float(morans_i), float(expected), float(np.sqrt(variance))


def estat_moran(
    result: Any,
    errorlag: Any = None,
    *,
    weights: Any = None,
) -> MoranResult:
    """Stata's ``estat moran``.

    Parameters
    ----------
    errorlag:
        One spatial weighting matrix, or a sequence of them for a higher-order test
        (``sthlp:66-68``: "``errorlag()`` is required ... repeatable to allow testing of
        higher-order error lags").  Repeated matrices are SUMMED, which is how Stata
        combines several ``spmatrix`` objects into one lag operator.
    weights:
        Alias for ``errorlag`` -- ``errorlag`` is Stata's option name, ``weights`` reads
        better in Python.  Supplying both is an error.

    Raises
    ------
    NotImplementedError
        When the fit is not a linear regression (``sthlp:44-45``).
    ValueError
        When no weighting matrix is given, when it does not match the estimation sample,
        or when the statistic is undefined.

    Notes
    -----
    See the module docstring: Stata's own ``r(chi2)`` formula is not locally available
    (the methods section is external and the ado is a compiled builtin), so the reported
    statistic is the standard randomisation-based Moran test.  Its p-value is exact for
    the hypothesis "no spatial correlation", but it may differ from Stata's number.
    """

    if errorlag is not None and weights is not None:
        raise ValueError(
            "estat moran: pass either errorlag= or weights=, not both"
        )
    supplied = errorlag if errorlag is not None else weights

    model_name = str(getattr(result, "model_name", "") or "").strip().lower()
    if model_name not in ("regress", "areg"):
        raise NotImplementedError(
            "estat moran is a postestimation test after regress with spatial data "
            f"(estat_moran.sthlp:44-45); got {model_name!r}"
        )

    if supplied is None:
        raise ValueError(
            "estat moran: errorlag() is required (estat_moran.sthlp:66-68); pass the "
            "spatial weighting matrix as errorlag= (or weights=)"
        )

    residuals = np.asarray(result.resid, dtype=float).reshape(-1)
    count = residuals.size

    # One matrix, or a sequence of them.  A numpy array is itself a single matrix, so it
    # has to be distinguished from a list of matrices.
    if isinstance(supplied, np.ndarray) and supplied.ndim == 2:
        matrices = [_as_weight_matrix(supplied, count)]
    elif isinstance(supplied, (list, tuple)):
        matrices = [_as_weight_matrix(entry, count) for entry in supplied]
    else:
        matrices = [_as_weight_matrix(supplied, count)]
    if not matrices:
        raise ValueError("estat moran: at least one weighting matrix is required")

    combined = matrices[0]
    for matrix in matrices[1:]:
        combined = combined + matrix

    morans_i, expected, standard_error = _moran_statistic(residuals, combined)
    z_value = (morans_i - expected) / standard_error
    statistic = float(z_value ** 2)
    return MoranResult(
        statistic=statistic,
        df=1.0,
        pvalue=float(scipy_stats.chi2.sf(statistic, 1)),
        moran_i=morans_i,
        expected_i=expected,
        z=float(z_value),
        errorlags=len(matrices),
    )
