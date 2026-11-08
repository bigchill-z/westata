"""VAR stability: Stata's ``varstable``.

Reference from a live ``varstable`` after ``var dinf unem, lags(1/2)`` on macro_swatson.dta::

          Eigenvalue        |   Modulus
    ------------------------+-------------
         .82182             |    .82182
         .7970223           |    .797022
        -.2258777 + .6292178i |  .668533
        -.2258777 - .6292178i |  .668533

    All the eigenvalues lie inside the unit circle.
    VAR satisfies stability condition.

The eigenvalues are those of the COMPANION matrix, not of the per-equation coefficient
matrices. ``np.linalg.eigvals(result.coefs)`` returns 2 values for a bivariate VAR(2) while
``varstable`` lists 4 -- one per ``lags * equations`` -- and they are different numbers.

Stata prints the eigenvalues in DESCENDING ORDER OF MODULUS, which is why the two complex
pair members come last rather than being sorted by their real parts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from westata.errors import ConsistencyError

__all__ = ["VarStableResult", "varstable"]


@dataclass(frozen=True)
class VarStableResult:
    """``varstable``: eigenvalues of the companion matrix."""

    eigenvalues: tuple[complex, ...]
    modulus: tuple[float, ...]

    @property
    def stable(self) -> bool:
        """True when every eigenvalue lies strictly inside the unit circle."""

        return all(value < 1.0 for value in self.modulus)

    def to_frame(self) -> Any:
        import pandas as pd

        return pd.DataFrame(
            {
                "Eigenvalue": list(self.eigenvalues),
                "Modulus": list(self.modulus),
            }
        )

    def __str__(self) -> str:
        """Stata's boxed eigenvalue table.

        Captured from a live ``varstable`` after a bare ``var dinf unem``::

               Eigenvalue stability condition
              +----------------------------------------+
              |        Eigenvalue        |   Modulus   |
              |--------------------------+-------------|
              |     .82182               |    .82182   |
              |   .7970223               |   .797022   |
              |  -.2258777 +  .6292178i  |   .668533   |
              +----------------------------------------+

        The bar positions are 2, 29 and 43, so the eigenvalue cell spans 3-28 and the modulus
        30-42. Within those cells:

        * a SHORT eigenvalue is right-aligned to offset 11
        * a LONG one (the complex row) starts at offset 2 with its own leading spaces
        * the modulus is right-aligned to offset 10

        Getting this to match took measuring all three rows: an earlier attempt right-aligned
        every value to the same column and missed the complex row by one space, and another
        left-aligned the whole cell and missed the short rows.
        """

        def _bare(number: float) -> str:
            """The MODULUS column: six significant figures at or above 0.1, six DECIMALS below it.

            arstable_w.ado:254 sets .numfmt %10.7g . %8.7g and _tab prints into a 13-wide field.
            Measured against r(Modulus) and the printed table together:

                1.0009273   -> 1.00093      six significant figures
                .95675429   -> .956754
                .0936159    -> .093616      SIX DECIMALS here; "%6g" gives .0936159
                .044383756  -> .044384
                .03918892   -> .039189

            EIGHT OF EIGHT match under this threshold, while a fixed six decimals and a fixed six
            significant figures each match only five. The fixed WIDTH is what makes it switch: above 0.1
            the leading digit consumes a column, so precision drops to significant figures.
            """

            value_abs = abs(float(number))
            if value_abs >= 0.1:
                text_value = f"{value_abs:.6g}"
            else:
                text_value = f"{value_abs:.6f}"
                if "." in text_value:
                    text_value = text_value.rstrip("0").rstrip(".")
            if text_value.startswith("0."):
                text_value = text_value[1:]
            return ("-" if float(number) < 0 else "") + text_value

        result = [
            "   Eigenvalue stability condition",
            "  +----------------------------------------+",
            "  |        Eigenvalue        |   Modulus   |",
            "  |--------------------------+-------------|",
        ]
        for value, magnitude in zip(self.eigenvalues, self.modulus):
            eigenvalue_text = _format_complex(value)
            cell = (
                "  " + eigenvalue_text if len(eigenvalue_text) > 11
                else eigenvalue_text.rjust(11)
            )
            result.append(
                f"  |{cell.ljust(26)}|{_bare(magnitude).rjust(10).ljust(13)}|"
            )
        result.append("  +----------------------------------------+")
        if self.stable:
            result.append("   All the eigenvalues lie inside the unit circle.")
            result.append("   VAR satisfies stability condition.")
        else:
            result.append("   At least one eigenvalue lies on or outside the unit circle.")
            result.append("   VAR does NOT satisfy stability condition.")
        return "\n".join(result)


def _format_complex(value: complex) -> str:
    """Render an eigenvalue the way Stata does: real part, then ``± .xxxxi``."""

    def _bare(number: float) -> str:
        """Seven significant figures with the leading zero dropped: .82182, -.2258777.

        Stata never prints the zero; "0.82182" appeared in every row before this.
        """

        text_value = f"{float(number):.7g}"
        if text_value.startswith("0."):
            text_value = text_value[1:]
        elif text_value.startswith("-0."):
            text_value = "-" + text_value[2:]
        return text_value

    real = float(value.real)
    imaginary = float(value.imag)
    if abs(imaginary) < 1e-12:
        return _bare(real)
    sign = "+" if imaginary >= 0 else "-"
    return f"{_bare(real)} {sign}  {_bare(abs(imaginary))}i"


def varstable(result: Any) -> VarStableResult:
    """Eigenvalues of the VAR's companion matrix, as ``varstable``.

    ``result`` must be a fitted VAR exposing ``coefs`` with shape ``(lags, K, K)``.
    """

    inner = getattr(result, "_result", result)
    coefficients = getattr(inner, "coefs", None)
    if coefficients is None:
        raise ConsistencyError(
            "varstable: the result does not expose the VAR coefficient matrices"
        )

    coefs = np.asarray(coefficients, dtype=float)
    if coefs.ndim != 3:
        raise ConsistencyError(
            f"varstable: expected coefficient matrices shaped (lags, K, K); got {coefs.shape}"
        )
    lags, equations, _ = coefs.shape
    if lags < 1:
        raise ConsistencyError("varstable: the VAR has no lags")

    top = np.hstack([coefs[index] for index in range(lags)])
    if lags == 1:
        companion = top
    else:
        lower_rows = equations * (lags - 1)
        companion = np.vstack(
            [
                top,
                np.hstack(
                    [
                        np.eye(lower_rows),
                        np.zeros((lower_rows, equations)),
                    ]
                ),
            ]
        )

    values = np.linalg.eigvals(companion)
    magnitudes = np.abs(values)
    # Stata lists them by DESCENDING modulus; ties keep the numpy order.
    order = np.argsort(-magnitudes, kind="stable")
    ordered_values = tuple(complex(values[i]) for i in order)
    ordered_modulus = tuple(float(magnitudes[i]) for i in order)

    return VarStableResult(
        eigenvalues=ordered_values, modulus=ordered_modulus
    )
