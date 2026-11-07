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
    """Stata's ``%9.0g`` cell: no leading zero, at most NINE characters of number.

    Derived in round 165 by putting each raw covariance beside Stata's printed text for the whole
    matrix at once, after four attempts at a ``%.Ng`` loop had failed. Two branches decide the
    rendering:

        decimal exponent >= -5  ->  FIXED POINT, as many decimals as fit in nine characters
        decimal exponent <= -6  ->  SCIENTIFIC, as many mantissa digits as fit in nine

    which is exactly the threshold the reference matrix straddles:

        3.5445e-05   (exp -5)  ->  .00003545      fixed, 8 signif figs
        9.1409e-06   (exp -6)  ->  9.141e-06      scientific
        -1.158e-05   (exp -5)  ->  -.00001158     fixed
        -3.106e-07   (exp -7)  ->  -3.106e-07     scientific
        .00720832    (exp -3)  ->  .00720832      fixed

    ``%.Ng`` cannot produce this: it switches to scientific at e-05, so ``.00003545`` came out as
    ``3.5445e-05``. The leading zero is dropped in the fixed branch and the exponent is written
    without a leading zero either (``e-06``, not ``e-007``).
    """

    if value is None or not np.isfinite(value):
        return f"{'.':>{width}}"
    number = float(value)
    if number == 0.0:
        return "0".rjust(width)

    from decimal import Decimal, ROUND_HALF_UP

    magnitude = abs(Decimal(repr(number)))
    exponent = magnitude.adjusted()
    budget = 9
    text = ""
    if exponent >= -5:
        # Fixed point: take the most decimals that still fit.
        for places in range(15, -1, -1):
            candidate = format(
                magnitude.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP), "f"
            )
            if "." in candidate:
                candidate = candidate.rstrip("0").rstrip(".")
            if candidate.startswith("0."):
                candidate = candidate[1:]
            if len(candidate) <= budget:
                text = candidate
                break
    else:
        # Scientific: most mantissa digits that still fit beside the exponent.
        for digits in range(9, 0, -1):
            mantissa = (magnitude.scaleb(-exponent)).quantize(
                Decimal(1).scaleb(-(digits - 1)), rounding=ROUND_HALF_UP
            )
            body = format(mantissa, "f")
            if "." in body:
                body = body.rstrip("0").rstrip(".")
            sign = "-" if exponent < 0 else "+"
            candidate = f"{body}e{sign}{abs(exponent):02d}"
            if len(candidate) <= budget:
                text = candidate
                break
    if not text:  # pragma: no cover - only reachable for absurd magnitudes
        text = f"{number:.1g}"
    if number < 0:
        text = "-" + text
    return text.rjust(width)


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
        # Stata's eleven-line-free block is exactly this, measured in round 165:
        #   ''                                       blank
        #   'Covariance matrix of coefficients ...'  title
        #   ''                                       blank
        #   '        e(V) |          s  ...'         62 wide
        #   '-------------+---------------...'       61 wide (ONE shorter than the rows)
        #   '           s |  .00003545    ...'       62 wide, blank after the diagonal
        # Note there is NO "<response> |" row: the dependent variable appears only in the column
        # heading. An earlier version printed one and shifted every line down by one.
        # The row width scales with the number of columns: 13 (label block) + 12 per cell + 1
        # trailing space. Measured on two matrices: three regressors -> 50, four -> 62. A fixed 62
        # made every narrower table over-padded.
        padding = 12
        row_width = lead + 1 + padding * len(labels) + 1
        lines = ["", title, ""]
        # The response-variable lines exist ONLY for the discrete models. Measured in round 165:
        #   regress -> 9 lines, no "| lnw" header line and no "lnw |" row
        #   logit   -> 11 lines, BOTH present ("             | hi" then "hi           |")
        # Treating them as universal is what made the OLS table one line too long.
        discrete = model in {"logit", "probit", "cloglog", "logistic"}
        if discrete and self.response_label:
            # 61 wide, matching the separator rather than the 62-wide numeric rows.
            lines.append(f"{'':>{lead}} | {self.response_label}".ljust(row_width - 1))
        # Each cell is TWELVE characters: the value right-aligned in 11 then one space.
        # Splitting Stata's own rows into 12-character slices reproduces them exactly:
        #   '  .00121321 ' | ' -.00025332 ' | '  .01065644'
        lines.append(
            (f"{'e(V)':>{lead}} |" + " ".join(f"{label:>11}" for label in labels)).ljust(
                row_width
            )
        )
        # The separator is 61 characters -- ONE SHORTER than the 62-wide rows, which is what
        # Stata prints.
        lines.append("-" * (lead + 1) + "+" + "-" * (row_width - lead - 3))
        if discrete and self.response_label:
            # The discrete models repeat the response as a row label above the coefficients.
            # This line is 61 wide, like the separator -- not the 62 of the numeric rows.
            lines.append(f"{self.response_label:<{lead}} |".ljust(row_width - 1))
        for row, label in enumerate(labels):
            # Every row is padded to 62 characters, blank after the diagonal -- measured from
            # Stata, which prints '       _cons | -.00049044  ...  .00720832 ' with the trailing
            # space that makes the width 62.
            cells = " ".join(
                _format_entry(matrix[row, col], 11)
                for col in range(row + 1)
            )
            lines.append((f"{label:>{lead}} |" + cells).ljust(row_width))
        # No trailing blank line: Stata's block ends on the last coefficient row.
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
