"""Variance inflation factors: Stata's ``estat vif``.

For each regressor, Stata regresses it on all the others and reports

    VIF_j   = 1 / (1 - R2_j)
    1/VIF_j = the tolerance

plus the arithmetic mean of the VIFs.  The definition is in ``vif.ado``
(StataNow 19.5, v1.5.4):

.. code-block:: stata

    `ver' _regress `redep' `OREVARS' `orevars' `if' `in' `wtexp' `nocons'
    replace `vvv' = 1/(1-e(r2)) in `i'
    ...
    noi di in ye %9.2f `vvv'[`i'] `"    "' in ye %8.6f 1/`vvv'[`i']
    noi di in ye %9.2f `mvif'/`nv'          // Mean VIF

The ``uncentered`` option appends a constant column to the auxiliary
regressions (``nocons`` is set in that branch), which makes the VIFs
"uncentered" in the usual sense.

Only linear models are served, because the auxiliary regressions are ordinary
least squares by construction.  ``vif.ado`` likewise refuses after
bootstrap/jackknife/svy prefixes (``e(prefix)``), and after ``regress, nocons``
unless ``uncentered`` is requested.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from westata.postestimation.names import stata_display_name
from westata.postestimation.names import is_constant_name

#: The constant-name test lives in `names.py` (R6.1).  It used to be redefined in
#: eleven modules with four subtly different spellings, one of which could never match.
_is_constant = is_constant_name

__all__ = ["VifResult", "estat_vif"]

#: Model names whose design is an ordinary linear predictor, so that a VIF is
#: meaningful.  Matches what Stata's ``regress_estat`` forwards to ``vif.ado``:
#: the command is reachable after ``regress`` and the linear siblings.
#: Estimators after which Stata actually offers ``estat vif``.
#:
#: ``ivregress`` is deliberately EXCLUDED.  ``ivregress_estat.ado:20-56`` dispatches
#: only firststage / overid / endogenous / weakrobust / sbknown|sbsingle and sends
#: everything else to ``estat_default``; ``estat_default.ado`` provides ic,
#: summarize, bootstrap and vce -- and no vif.  Listing the IV estimators here made
#: ``estat vif`` return a number that was merely the OLS VIF of the same design
#: (measured identical to the OLS fit: 1.000203 on both), which looks like a
#: feature and is not one.
#:
#: ``newey`` is excluded for the same reason, from its own dispatcher:
#: ``newey_estat.ado:11-17`` handles only ``durbinalt`` and forwards everything else to
#: ``estat_default``, so ``estat vif`` never reaches ``vif.ado``.  (``newey_p.ado:20``
#: does list ``XB Index STDP REsiduals`` for ``predict``, which is why ``predict`` after
#: newey works while ``estat vif`` does not -- the two commands have different ado
#: files and different lists.)
#:
#: The four below are the estimators whose postestimation ado chains do reach
#: ``vif.ado``: ``regress`` directly, and ``areg``/``eivreg``/``reghdfe`` whose Stata
#: implementations mirror regress's postestimation surface.
_LINEAR_MODELS = frozenset(
    {
        "regress",
        "reghdfe",
        "areg",
        "eivreg",
    }
)

#: Names a constant can appear under in westata's parameter list.



def _display_name(name: str) -> str:
    """Deprecated shim: the mapping lives in :mod:`westata.postestimation.names`.

    This module used to carry its OWN copy, which handled the constant but not the
    interaction separator -- so one analysis printed ``x1#x2`` in the coefficient
    table and ``x1:x2`` here.  Delegating removes the possibility of another drift.
    """

    return stata_display_name(name)


class VifResult:
    """The ``estat vif`` table.

    Attributes
    ----------
    names:
        Regressor names in the reported order (descending VIF, matching Stata's
        ``gsort -vvv nms``, unless ``nosort=True`` was requested).
    vif, tolerance:
        The VIF and its reciprocal, per regressor.
    mean_vif:
        Arithmetic mean of the reported VIFs.
    """

    __slots__ = ("names", "vif", "tolerance", "mean_vif")

    def __init__(
        self, names: Sequence[str], vif: Sequence[float], mean_vif: float
    ) -> None:
        labels = list(names)
        values = np.asarray(vif, dtype=float).reshape(-1)
        if values.size != len(labels):
            raise ValueError(
                f"{values.size} VIF values for {len(labels)} regressors"
            )
        self.names = labels
        self.vif = values
        with np.errstate(divide="ignore", invalid="ignore"):
            self.tolerance = 1.0 / values
        self.mean_vif = float(mean_vif)

    # -- presentation ---------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"<VifResult {len(self.names)} regressors, "
            f"mean VIF {self.mean_vif:.2f}>"
        )

    def __str__(self) -> str:
        return self.render()

    def render(self) -> str:
        """Reproduce ``estat vif`` exactly::

                Variable |       VIF       1/VIF
            -------------+----------------------
                    temp |      1.14    0.873845
                  income |      1.14    0.873984
                   price |      1.04    0.965555
            -------------+----------------------
                Mean VIF |      1.11

        The name column is right-aligned in a 12-character field and the header is spelled
        ``    Variable |``, one space before the pipe. Left-aligning the header produced
        ``    Variable    |``, which pushed the pipe one column right and misaligned the whole
        table against Stata's.
        """

        # Captured byte-exactly from a live estat vif: NO leading blank line, and the header
        # carries two trailing spaces so it is 36 characters like every other row.
        header = f"{'Variable':>12} |       VIF       1/VIF  "
        rule = "-" * 13 + "+" + "-" * 22
        # A LEADING BLANK LINE: Stata's raw capture starts '\n    V' (newline then the indented
        # header), so the block opens with an empty line. Measured in round 193 by walking every
        # estat renderer's prefix -- the same survey that found estat ic, bgodfrey and dwatson.
        lines = ["", header, rule]
        for name, value, tol in zip(self.names, self.vif, self.tolerance):
            lines.append(f"{name:>12} |{value:>10.2f}    {tol:>8.6f}")
        lines.append(rule)
        lines.append(f"{'Mean VIF':>12} |{self.mean_vif:>10.2f}")
        return chr(10).join(lines)

    def to_frame(self) -> pd.DataFrame:
        """The table as a DataFrame indexed by regressor name."""

        return pd.DataFrame(
            {"VIF": self.vif, "1/VIF": self.tolerance},
            index=pd.Index(self.names, name="variable"),
        )

    def to_dict(self) -> dict[str, float]:
        """Stata's ``r(vif_#)`` scalars keyed by variable name."""

        return {name: float(value) for name, value in zip(self.names, self.vif)}


def _require_linear(result: Any) -> None:
    name = str(getattr(result, "model_name", "") or "").strip().lower()
    if name not in _LINEAR_MODELS:
        raise NotImplementedError(
            "estat vif is only defined for linear estimation: Stata reaches "
            "vif.ado through regress_estat, and the auxiliary regressions are "
            "ordinary least squares by construction; got "
            f"{name or 'an unnamed model'}"
        )


def _auxiliary_r2(matrix: np.ndarray, target: int, *, uncentered: bool) -> float:
    """R-squared from regressing column ``target`` on the remaining columns.

    The auxiliary regression is ordinary least squares with a constant, which is
    what ``vif.ado`` runs (``_regress`` includes one unless ``nocons`` is set,
    and ``nocons`` is only set in the ``uncentered`` branch).  The constant is
    added here rather than taken from the stored design, because the design's
    own constant column is dropped along with the target when the target is not
    the constant, and keeping it would make every auxiliary R-squared 1.
    """

    y = matrix[:, target]
    others = np.delete(matrix, target, axis=1)
    # Drop any constant columns from the "others" set: they carry no
    # explanatory power once an explicit intercept is added below, and leaving
    # one in would make the auxiliary fit exact.
    if others.size:
        keep = [j for j in range(others.shape[1]) if np.ptp(others[:, j]) > 0.0]
        others = others[:, keep] if keep else np.empty((y.size, 0))

    if uncentered:
        design = others if others.size else np.empty((y.size, 0))
    else:
        design = np.column_stack([np.ones_like(y), others]) if others.size else np.ones((y.size, 1))

    if design.shape[1] == 0:  # pragma: no cover - guarded by the caller
        return float("nan")

    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coefficients
    if uncentered:
        total = float(y @ y)
    else:
        centred = y - y.mean()
        total = float(centred @ centred)
    rss = float(resid @ resid)
    if total <= 0.0:
        return float("nan")
    return 1.0 - rss / total


def _vif_from_r2(r2: float) -> float:
    """``1/(1-R2)``, matching what ``vif.ado`` would print.

    Stata does not special-case a perfect fit: ``_regress`` would fail before
    producing ``e(r2) == 1``, so the value stays finite in practice.  Here the
    tolerance is clamped the same way a floating-point solve behaves -- a
    numerically exact fit yields a very large but finite VIF -- so that the mean
    stays computable and the output remains readable.
    """

    if not np.isfinite(r2):
        return float("inf")
    denominator = 1.0 - r2
    # ``1 - r2`` is evaluated in double precision; when the fit is exact the
    # subtraction leaves a tiny positive residue rather than zero.
    if denominator <= 0.0:
        denominator = np.finfo(float).eps
    return float(1.0 / denominator)


def estat_vif(
    result: Any,
    *,
    uncentered: bool = False,
    nosort: bool = False,
) -> VifResult:
    """Stata's ``estat vif`` for a linear fit.

    Parameters
    ----------
    result:
        A fitted linear result that retained its design (``design_matrix``).
    uncentered:
        Do not centre the auxiliary regressions (Stata's ``, uncentered``).
        Required if the model has no constant, mirroring ``vif.ado``'s refusal.
    nosort:
        Keep the model's own regressor order instead of sorting by VIF
        (Stata's ``, nosort``).

    Returns
    -------
    VifResult

    Raises
    ------
    NotImplementedError
        For a non-linear model, where Stata has no ``estat vif``.
    ValueError
        If the design was not retained, or if a model without a constant is
        queried without ``uncentered=True``.
    """

    _require_linear(result)

    design = getattr(result, "design_matrix", None)
    if design is None:
        raise ValueError(
            "estat vif needs the estimation design, which this result did not "
            "retain; refit with a from_formula constructor"
        )
    matrix_raw, raw_names = design
    matrix = np.asarray(matrix_raw, dtype=float)
    names = [str(name) for name in raw_names]
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError(
            f"design matrix shape {matrix.shape} does not match {len(names)} names"
        )

    constant_columns = [j for j, name in enumerate(names) if _is_constant(name)]
    has_constant = bool(constant_columns)
    if not has_constant and not uncentered:
        raise ValueError(
            "not appropriate after a model without a constant; use "
            "uncentered=True to get uncentered VIFs"
        )

    # The constant itself is never a VIF row (Stata only appends ``_cons`` when
    # ``uncentered`` asks for it).
    regressor_columns = [j for j in range(len(names)) if not _is_constant(names[j])]
    if uncentered and has_constant:
        regressor_columns = regressor_columns + constant_columns[:1]

    if not regressor_columns:
        raise ValueError("estat vif needs at least one regressor")

    values: list[float] = []
    labels: list[str] = []
    for position in regressor_columns:
        r2 = _auxiliary_r2(matrix, position, uncentered=uncentered)
        values.append(_vif_from_r2(r2))
        labels.append(_display_name(names[position]))

    order = list(range(len(values)))
    if not nosort:
        # vif.ado: ``gsort -vvv nms`` -- descending VIF, ties by name.
        order.sort(key=lambda i: (-values[i], labels[i]))

    mean_vif = float(np.mean(values)) if values else float("nan")

    return VifResult(
        [labels[i] for i in order],
        [values[i] for i in order],
        mean_vif,
    )
