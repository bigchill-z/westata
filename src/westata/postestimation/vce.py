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
    """Stata's ``%10.0g`` cell: no leading zero, at most ten characters.

    The WIDTH decides the precision, not a fixed significant-digit count -- the same rule
    that governs every other Stata column. On the reference covariance matrix:

        .0168814     prints at 7 significant figures (8 characters)
        .01251233    at 8 (because it still fits in 10)
        -.00338524   at 6 (10 characters at 7, so Stata falls back)

    A single significant-digit rule produced ``-.003385236`` where Stata prints
    ``-.00338524``.
    """

    if value is None or not np.isfinite(value):
        return f"{'.':>{width}}"
    number = float(value)
    if number == 0.0:
        return "0".rjust(width)
    for digits in range(7, 0, -1):
        text = f"{number:.{digits}g}"
        if text.startswith("0."):
            text = text[1:]
        elif text.startswith("-0."):
            text = "-" + text[2:]
        if text.endswith(".0"):
            text = text[:-2]
        if len(text) <= 10 - (0 if digits <= 6 else 1):
            return text.rjust(width)
    return f"{number:.1g}".rjust(width)


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

    __slots__ = ("names", "values", "kind", "model_label", "response_label")

    def __init__(
        self,
        names: Sequence[str],
        values: np.ndarray,
        kind: str,
        model_label: str = "",
        response_label: str = "",
    ) -> None:
        #: Estimator name for the title ("Covariance matrix of coefficients of logit
        #: model") and the dependent-variable name heading the column block. Both come
        #: from the fit; empty when the caller did not supply them.
        self.model_label = str(model_label)
        self.response_label = str(response_label)
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

        # Stata's layout, measured from a live ``estat vce`` after ``logit y x1 x2``::
        #
        #     Covariance matrix of coefficients of logit model
        #                  | y
        #             e(V) |         x1          x2       _cons
        #     -------------+-----------------------------------
        #     y            |
        #               x1 |   .0168817
        #               x2 | -.00338524   .01251233
        #            _cons |  .00121321  -.00025332   .01065644
        #
        # Three pieces were missing entirely: the title names the MODEL, the column
        # block is headed by the dependent variable on its own line, and the row labels
        # sit under an "e(V)" caption. Without them the matrix is present but a reader
        # cannot tell which model or equation it belongs to.
        model = self.model_label.strip()
        title = (
            f"Covariance matrix of coefficients of {model} model"
            if model and self.kind == "covariance"
            else (
                "Covariance matrix"
                if self.kind == "covariance"
                else "Correlation matrix of coefficients"
            )
        )
        lead = 12  # "        e(V) " puts the pipe at column 13
        lines = [title]
        if self.response_label:
            lines.append(f"{'':>{lead}} | {self.response_label}")
        # Stata's column fields are 11 wide, set by its %10.0g cells, not by the longest
        # label -- deriving the width from the labels widened the whole matrix.
        # Stata's column fields are 11 wide and right-aligned; the header row's fields
        # are one wider than the value rows because of how Stata pads the label block.
        # Each cell is TWELVE characters: the value right-aligned in 11 then one space.
        # Splitting Stata's own rows into 12-character slices reproduces them exactly:
        #   '  .00121321 ' | ' -.00025332 ' | '  .01065644'
        padding = 12
        lines.append(
            f"{'e(V)':>{lead}} |" + " ".join(f"{label:>11}" for label in labels)
        )
        # The separator is the SAME width as the header line: 13 dashes, "+", then the
        # rest. A one-shorter version was within a character of right and therefore easy
        # to miss.
        lines.append("-" * (lead + 1) + "+" + "-" * (len(lines[-1]) - lead - 2))
        if self.response_label:
            lines.append(f"{self.response_label:<{lead}} |")
        for row, label in enumerate(labels):
            # Stata prints the lower triangle; the upper triangle is blank.
            # Stata stops each row AT the diagonal -- it does not pad the trailing
            # columns with blanks. Padding them made every line end with spaces and the
            # row widths disagree with Stata's.
            cells = " ".join(
                _format_entry(matrix[row, col], 11)
                for col in range(row + 1)
            )
            lines.append(f"{label:>{lead}} |{cells}")
        return chr(10).join(lines)


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
        return VceMatrix(
            names,
            cov,
            "covariance",
        model_label=str(getattr(result, "model_name", "") or ""),
        response_label=str(
            getattr(result, "response_name", None)
            or (getattr(result, "e", None) or {}).get("depvar")
            or ""
        ),
        )

    diagonal = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(diagonal > 0, diagonal, np.nan)
        corr = cov / np.outer(scale, scale)
    # A variable is not correlated with itself in Stata's ``vce, correlation``.
    corr = np.asarray(corr, dtype=float)
    np.fill_diagonal(corr, 0.0)
    return VceMatrix(
        names,
        corr,
        "correlation",
        model_label=str(getattr(result, "model_name", "") or ""),
        response_label=str(
            getattr(result, "response_name", None)
            or (getattr(result, "e", None) or {}).get("depvar")
            or ""
        ),
    )
