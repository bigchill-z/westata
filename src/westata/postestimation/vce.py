"""Variance-covariance reporting: Stata's ``estat vce``.

Stata prints the coefficient covariance matrix (and, with ``, correlation``, the
corresponding correlation matrix) after any estimation command.  westata always
*held* the matrix on the result object -- :meth:`StataResults.cov_params` -- but
had no way to render it, so a user who wanted Stata's ``estat vce`` output had to
reach into the matrix themselves and format it.

The two commands differ only in what is printed:

``estat vce``
    The covariance matrix ``V`` itself.
``estat vce, correlation``
    The correlation matrix ``D^-1/2 V D^-1/2`` where ``D = diag(V)``, i.e. each
    entry divided by the two standard errors.  Stata prints zeros on the
    diagonal of this matrix (a variable is not "correlated with itself" in the
    output), which is why the diagonal is blanked below.

Both matrices are symmetric and Stata prints the lower triangle only.
"""

from __future__ import annotations

from typing import Any, Sequence

from westata.postestimation.names import stata_display_name

import numpy as np
import pandas as pd

__all__ = ["VceMatrix", "estat_vce"]


def _stata_order_of(result: Any, size: int) -> list[int]:
    """Stata's parameter order, or the identity when it is unavailable.

    ``_stata_order`` also refuses on ambiguous factor terms; the identity then keeps
    the matrix internally consistent even though it will not match Stata's ordering.
    """

    try:
        order = list(result._stata_order())
    except (AttributeError, NotImplementedError):
        return list(range(size))
    if len(order) != size:
        return list(range(size))
    return order


def _display_name(name: str) -> str:
    """Deprecated shim: the mapping lives in :mod:`westata.postestimation.names`.

    Kept as a thin wrapper so existing imports keep working; the behaviour is that one
    implementation's, which is the point -- a second copy would drift.
    """

    return stata_display_name(name)


def _format_entry(value: float, width: int) -> str:
    """Stata prints covariances in ``%10.0g``-ish form, no leading zero."""

    if value is None or not np.isfinite(value):
        return f"{'.':>{width}}"
    text = f"{value:.7g}"
    if text.startswith("0."):
        text = text[1:]
    elif text.startswith("-0."):
        text = "-" + text[2:]
    elif text in ("0", "-0"):
        text = "0"
    return f"{text:>{width}}"


class VceMatrix:
    """A printed variance-covariance (or correlation) matrix.

    Attributes
    ----------
    names:
        Row/column labels in Stata's spelling.
    values:
        The full symmetric matrix as a ``pandas.DataFrame`` (indexed by
        ``names``).  This is the machine-readable surface; ``str()`` renders the
        lower triangle the way Stata prints it.
    kind:
        ``"covariance"`` or ``"correlation"``.
    """

    __slots__ = ("names", "values", "kind")

    def __init__(self, names: Sequence[str], values: np.ndarray, kind: str) -> None:
        labels = list(names)
        matrix = np.asarray(values, dtype=float)
        if matrix.shape != (len(labels), len(labels)):
            raise ValueError(
                f"matrix must be {len(labels)}x{len(labels)}, got {matrix.shape}"
            )
        self.names = labels
        self.values = pd.DataFrame(matrix, index=labels, columns=labels)
        self.kind = kind

    # -- presentation ---------------------------------------------------
    def __str__(self) -> str:
        return self.render()

    def __repr__(self) -> str:
        return (
            f"<VceMatrix {self.kind}: {len(self.names)} parameters>"
        )

    def to_frame(self) -> pd.DataFrame:
        """Return the full symmetric matrix."""

        return self.values

    def render(self, *, width: int | None = None) -> str:
        """Render the lower triangle the way Stata prints it."""

        labels = self.names
        name_width = max([8, *(len(label) for label in labels)]) + 2
        # Stata sizes each numeric column to fit the widest entry *and* the
        # column heading, so long parameter names do not collide with their
        # neighbour's heading.
        matrix = self.values.to_numpy()
        cell_width = max(
            [10, *(len(_format_entry(v, 1)) + 2 for v in matrix.ravel() if v == v)]
        )
        if width is None:
            width = max(
                cell_width,
                *(len(label) + 2 for label in labels),
            )

        header = " " * name_width + "".join(
            f"{label:>{width}}" for label in labels
        )

        title = (
            "Covariance matrix"
            if self.kind == "covariance"
            else "Correlation matrix of coefficients"
        )
        lines = [title, "", header, "-" * len(header)]
        for row, label in enumerate(labels):
            # Stata prints the lower triangle; the upper triangle is blank.
            cells = "".join(
                _format_entry(matrix[row, col], width) if col <= row else " " * width
                for col in range(len(labels))
            )
            lines.append(f"{label:<{name_width}}{cells}")
        return "\n".join(lines)


def estat_vce(result: Any, *, correlation: bool = False) -> VceMatrix:
    """Stata's ``estat vce`` / ``estat vce, correlation``.

    Parameters
    ----------
    result:
        Any fitted westata result exposing ``param_names`` and
        ``cov_params()``.
    correlation:
        When true, print the coefficient correlation matrix instead of the
        covariance matrix.  The diagonal is blanked to ``0`` exactly as Stata
        does.

    Returns
    -------
    VceMatrix
        Call ``str(...)`` for the printed table, or ``.to_frame()`` for the full
        symmetric matrix.
    """

    try:
        raw_names = list(result.param_names)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise TypeError(
            "estat vce needs a fitted result exposing param_names; got "
            f"{type(result).__name__}"
        ) from exc

    # Stata prints the matrix in the SAME row order as the coefficient table, where
    # ``_cons`` comes last.  Without this the two tables of one analysis disagreed on
    # the order as well as on the separator, so ``names[i]`` referred to different
    # parameters in each -- and the matrix still looked perfectly square.
    order = _stata_order_of(result, len(raw_names))
    names = [_display_name(raw_names[index]) for index in order]

    cov = np.asarray(result.cov_params(), dtype=float)
    if len(order) == cov.shape[0] and order != list(range(len(order))):
        cov = cov[np.ix_(order, order)]
    if not correlation:
        return VceMatrix(names, cov, "covariance")

    diagonal = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(diagonal > 0, diagonal, np.nan)
        corr = cov / np.outer(scale, scale)
    # A variable is not correlated with itself in Stata's ``vce, correlation``.
    corr = np.asarray(corr, dtype=float)
    np.fill_diagonal(corr, 0.0)
    return VceMatrix(names, corr, "correlation")
