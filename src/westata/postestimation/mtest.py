"""Multiple-testing adjustment for ``estat hettest, mtest()``.

Transcribed from ``hettest.ado:160-190`` and ``_mtest.ado``.

``hettest`` with ``mtest()`` runs ONE auxiliary regression per regressor
(``hettest.ado:170-176``):

.. code-block:: stata

    foreach v of local varlist {
        qui _regress `ui' `v' `wgt' if `touse'
        mat `mtest'[`i',1] = e(mss)/2        // statistic
        mat `mtest'[`i',2] = e(df_m)         // df
        mat `mtest'[`i',3] = chi2tail(e(df_m), e(mss)/2)
    }

then adjusts the p-values with ``_mtest adjust``.  The three methods
(``_mtest.ado:189-210``):

.. code-block:: stata

    * Bonferroni
    adjp[i] = min(1, m * p[i])

    * Sidak
    adjp[i] = 1 - (1 - p[i])^m

    * Holm  (_mtest.ado:266-296)
    order = order(p, 1)                 // ascending
    pa = p; m = k
    o1 = order[1]
    pa[o1] = min(p[o1] * m, 1)
    for i = 2..k:
        m--
        oi = order[i]
        pa[oi] = min(p[oi] * m, 1)
        if pa[oi] < pa[o1]: pa[oi] = pa[o1]     // enforce monotonicity
        o1 = oi

The Holm step is the one that is easy to get wrong: the multiplier decreases as
the rank increases, AND the result is forced to be monotone non-decreasing along
the sorted order (Stata compares against the *previous* adjusted value, not the
previous raw p-value).

``mtest()`` is incompatible with ``iid`` and ``fstat``
(``hettest.ado:49-61`` rejects both combinations).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy import stats

__all__ = ["MTestResult", "adjust_pvalues", "MTEST_METHODS"]

#: The values ``_mtest.ado:84`` accepts.
MTEST_METHODS = ("bonferroni", "holm", "sidak", "noadjust")


def adjust_pvalues(pvalues: Sequence[float], method: str) -> np.ndarray:
    """Adjust a family of p-values with Stata's ``_mtest`` methods.

    Parameters
    ----------
    pvalues:
        Raw p-values, one per test.
    method:
        ``"bonferroni"``, ``"holm"``, ``"sidak"`` or ``"noadjust"``.

    Returns
    -------
    numpy.ndarray
        Adjusted p-values in the SAME order as the input.

    Raises
    ------
    ValueError
        For an unknown method.

    Notes
    -----
    ``noadjust`` returns the raw values, which is what ``_mtest.ado:212`` does
    when no method is requested.
    """

    key = str(method).strip().lower()
    if key not in MTEST_METHODS:
        raise ValueError(
            f"mtest() method {method!r} is not supported; Stata's _mtest accepts "
            + ", ".join(MTEST_METHODS)
        )
    raw = np.asarray(pvalues, dtype=float).reshape(-1)
    count = raw.size
    if count == 0:
        return raw.copy()

    if key == "noadjust":
        return raw.copy()

    if key == "bonferroni":
        # adjp[i] = min(1, m * p[i])
        return np.minimum(1.0, count * raw)

    if key == "sidak":
        # adjp[i] = 1 - (1 - p[i])^m
        return 1.0 - np.power(1.0 - raw, count)

    # Holm: step-down with a monotonicity floor.
    adjusted = raw.copy()
    order = np.argsort(raw, kind="stable")
    multiplier = count
    previous = None
    for position in order:
        value = min(raw[position] * multiplier, 1.0)
        if previous is not None and value < previous:
            # ``_mtest_holm`` clamps to the previous adjusted value so the
            # sequence is non-decreasing in sorted order.
            value = previous
        adjusted[position] = value
        previous = value
        multiplier -= 1
    return adjusted


class MTestResult:
    """The per-variable heteroskedasticity tests and their adjusted p-values.

    Attributes
    ----------
    table:
        DataFrame indexed by variable with columns ``chi2``, ``df``, ``p`` and,
        when an adjustment was applied, ``adjustp``.
    method:
        The adjustment method used, or ``"noadjust"``.
    """

    __slots__ = ("table", "method")

    def __init__(self, table: Any, method: str) -> None:
        self.table = table
        self.method = str(method)

    def __repr__(self) -> str:
        return (
            f"<MTestResult {len(self.table)} variables, "
            f"method={self.method!r}>"
        )

    def __str__(self) -> str:
        header = (
            f"{'Variable':<16}|{'chi2':>10}{'df':>6}{'p':>10}"
            + (f"{'adjustp':>10}" if self.method != "noadjust" else "")
        )
        lines = ["", f"Multiple testing adjustment: {self.method}", header]
        lines.append("-" * len(header))
        for name, row in self.table.iterrows():
            line = (
                f"{str(name):<16}|{row['chi2']:>10.2f}{row['df']:>6.0f}"
                f"{row['p']:>10.4f}"
            )
            if self.method != "noadjust":
                line += f"{row['adjustp']:>10.4f}"
            lines.append(line)
        return "\n".join(lines)

    def to_frame(self) -> Any:
        """The tests as a ``nvar x 3`` (or ``x 4``) frame."""

        return self.table

    def to_matrix(self) -> np.ndarray:
        """Stata's ``r(mtest)`` matrix: ``nvar x 3`` of statistic, df, p.

        With an adjustment the matrix gains a fourth column, exactly as
        ``_mtest adjust ..., append`` appends one (``hettest.ado:186-187``).
        """

        columns = ["chi2", "df", "p"]
        if self.method != "noadjust" and "adjustp" in self.table.columns:
            columns.append("adjustp")
        return self.table[columns].to_numpy(dtype=float)

    def to_dict(self) -> dict[str, dict[str, float]]:
        payload = {
            str(name): {
                "chi2": float(row["chi2"]),
                "df": float(row["df"]),
                "p": float(row["p"]),
            }
            for name, row in self.table.iterrows()
        }
        if self.method != "noadjust" and "adjustp" in self.table.columns:
            for name in payload:
                payload[name]["adjustp"] = float(self.table.loc[name, "adjustp"])
        return payload


def mtest_statistics(
    residuals_squared_ui: np.ndarray,
    design: np.ndarray,
    names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the per-regressor auxiliary regressions of ``hettest.ado:170-176``.

    Each regressor gets its OWN simple regression of the scaled squared
    residual ``ui`` on that single column, and the statistic is that
    regression's ``e(mss)/2`` -- not the statistic from the full model.

    Returns ``(statistics, dfs, pvalues)``.
    """

    ui = np.asarray(residuals_squared_ui, dtype=float).reshape(-1)
    matrix = np.asarray(design, dtype=float)
    statistics: list[float] = []
    dfs: list[float] = []
    pvalues: list[float] = []

    for column in range(matrix.shape[1]):
        single = matrix[:, column : column + 1]
        # Add a constant, as _regress would.
        single = np.column_stack([np.ones(ui.size), single])
        coefficients, *_ = np.linalg.lstsq(single, ui, rcond=None)
        fitted = single @ coefficients
        resid = ui - fitted
        rss = float(resid @ resid)
        centred = ui - ui.mean()
        tss = float(centred @ centred)
        mss = tss - rss
        statistic = mss / 2.0
        df = float(single.shape[1] - 1)
        statistics.append(float(statistic))
        dfs.append(df)
        pvalues.append(float(stats.chi2.sf(statistic, df)) if df > 0 else float("nan"))

    return (
        np.asarray(statistics, dtype=float),
        np.asarray(dfs, dtype=float),
        np.asarray(pvalues, dtype=float),
    )
