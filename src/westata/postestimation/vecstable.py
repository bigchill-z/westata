"""VECM stability: Stata's ``vecstable``.

``vecstable.ado`` is six lines long -- it forwards to ``varstable_w2, vec`` -- so the real work
is in ``varstable_w.ado:95-150``, which builds the companion matrix ``bigAP`` from the VECM's
short-run matrices rather than from a VAR.

The companion for a VECM with ``lags = 2`` and ``k`` equations is ``2k x 2k``::

    A = [ I + alpha beta' + Gamma_1    -Gamma_1 ]
        [ I                            0        ]

and its eigenvalues include the unit modulus the specification imposes. That is not a defect:
Stata prints "The VECM specification imposes a unit modulus." beneath the table.

Reference from a live ``vecstable`` after ``vec dinf unem, lags(2) rank(1)``::

    Eigenvalue              Modulus
     1                       1
    -.2233155 + .6276579i    .666201
    -.2233155 - .6276579i    .666201
     .6305164                .630516

Building the companion from the VAR's ``coefs`` instead gives .82182 / .797022 / .668533 /
.668533 -- the VAR(2) eigenvalues, which are the WRONG object here and look plausible anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from westata.errors import ConsistencyError

__all__ = ["VecStableResult", "vecstable"]


@dataclass(frozen=True)
class VecStableResult:
    """``vecstable``: eigenvalues of a VECM's companion matrix."""

    eigenvalues: tuple[complex, ...]
    modulus: tuple[float, ...]

    @property
    def stable(self) -> bool:
        """True when every modulus is at most one, apart from the imposed unit roots."""

        return all(value <= 1.0 + 1e-9 for value in self.modulus)

    def to_frame(self) -> Any:
        import pandas as pd

        return pd.DataFrame(
            {"Eigenvalue": list(self.eigenvalues), "Modulus": list(self.modulus)}
        )

    def __str__(self) -> str:
        # STATA BOXES THIS TABLE. Measured from `vecstable` on a rank-1, lags(2) system: every line
        # is 44 characters with the pipes at columns 2, 29 and 43, so the fields are 26 and 13.
        #
        #     '  +----------------------------------------+'
        #     '  |        Eigenvalue        |   Modulus   |'
        #     '  |--------------------------+-------------|'
        #     '  |          1               |         1   |'
        #     '  |   .2681229               |   .268123   |'
        #     '  |  -.1557235 +  .0329841i  |   .159178   |'
        #     '  +----------------------------------------+'
        #
        # THE CELL RULE, derived by measuring each row rather than trying alignments: a value is
        # right-aligned to column 11 and then padded to the 26-wide field. That single rule explains
        # `1` (ten leading spaces), `.2681229` (three) and the modulus column alike. Values LONGER
        # than 11 start at the field's two-space inset instead, which is what the complex pair does.
        def _cell(text_value: str, width: int = 26, pivot: int = 11) -> str:
            if len(text_value) > pivot:
                return ("  " + text_value).ljust(width)[:width]
            return text_value.rjust(pivot).ljust(width)

        _rule = "+" + "-" * 40 + "+"
        lines = [
            "",
            "   Eigenvalue stability condition",
            "  " + _rule,
            "  |" + f"{'Eigenvalue':^26}" + "|" + f"{'Modulus':^13}" + "|",
            "  |" + "-" * 26 + "+" + "-" * 13 + "|",
        ]
        for value, magnitude in zip(self.eigenvalues, self.modulus):
            lines.append(
                "  |"
                + _cell(_format_complex(value), 26, 11)
                + "|"
                + _cell(_stata_number(magnitude), 13, 10)
                + "|"
            )
        lines.append("  " + _rule)
        if self.stable:
            lines.append("   The VECM specification imposes a unit modulus.")
        else:
            lines.append("   At least one eigenvalue lies outside the unit circle.")
        return "\n".join(lines)


def _stata_number(value: float, digits: int = 6) -> str:
    """Six significant digits with Stata's leading zero dropped."""

    text = f"{value:.{digits}g}"
    if text.startswith("0."):
        text = text[1:]
    elif text.startswith("-0."):
        text = "-" + text[2:]
    return text


def _format_complex(value: complex) -> str:
    real = float(value.real)
    imaginary = float(value.imag)
    if abs(imaginary) < 1e-12:
        text = f"{real:.7g}"
        if text.startswith("0."):
            return text[1:]
        if text.startswith("-0."):
            return "-" + text[2:]
        return text
    sign = "+" if imaginary >= 0 else "-"

    def _trim(value: float) -> str:
        text = f"{value:.7g}"
        if text.startswith("0."):
            return text[1:]
        if text.startswith("-0."):
            return "-" + text[2:]
        return text

    # Stata prints `-.1557235 +  .0329841i`: one space before the sign, two after it.
    return _trim(real) + " " + sign + "  " + _trim(abs(imaginary)) + "i"


def vecstable(result: Any) -> VecStableResult:
    """Eigenvalues of the VECM's companion matrix, as ``vecstable``.

    ``result`` must be a fitted VECM exposing ``alpha``, ``beta`` and ``gamma``, with
    ``neqs`` equations and at least one lag order.
    """

    alpha = getattr(result, "alpha", None)
    beta = getattr(result, "beta", None)
    gamma = getattr(result, "gamma", None)
    equations = getattr(result, "neqs", None)
    if alpha is None or beta is None or gamma is None or equations is None:
        raise ConsistencyError(
            "vecstable: the result does not expose alpha, beta, gamma and neqs"
        )
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    k = int(equations)
    if alpha.shape[0] != k or beta.shape[0] != k:
        raise ConsistencyError(
            f"vecstable: alpha {alpha.shape} / beta {beta.shape} do not match {k} equations"
        )

    # ``gamma`` holds the short-run blocks; how many depends on the lag order.
    if gamma.size % (k * k) != 0:
        raise ConsistencyError(
            f"vecstable: gamma has {gamma.size} entries, not a multiple of {k * k}"
        )
    blocks = gamma.reshape(-1, k, k)
    short_run = blocks.shape[0]
    if short_run < 1:
        raise ConsistencyError("vecstable: the VECM has no short-run blocks")

    # The companion is ALWAYS k x (k * (short_run + 1)), never k x k: even a VECM with one
    # short-run block has the state vector (Y_t, Y_{t-1}), whose dimension is 2k. Building a
    # k x k matrix in that case gave 1.6204 / 0.4365 instead of Stata's 1 / .666201 /
    # .666201 / .630516.
    # The state vector is (Y_t, Y_{t-1}, ..., Y_{t-short_run}), so the companion is always
    # k x (k * (short_run + 1)). Even a VECM with ONE short-run block has a 2k-dimensional
    # state: building a k x k matrix there gave 1.6204 / 0.4365 instead of Stata's
    # 1 / .666201 / .666201 / .630516.
    #
    # Row 1: [ I + alpha beta' + Gamma_1 , -Gamma_2, ..., -Gamma_short_run ]
    # Rows below: shift blocks [ I 0 ... ].
    long_run = np.eye(k) + alpha @ beta.T
    first_block = long_run + blocks[0]
    later_blocks = [-block for block in blocks[1:]]
    # ``-Gamma_1`` occupies the second column block: the differenced left-hand side depends
    # on Y_{t-1} through BOTH the error-correction term and the first lag of the difference,
    # and the two combine into (I + ab' + Gamma_1) on the first block and -Gamma_1 on the
    # second. That second block is what the short-run matrix contributes when there is only
    # one; with more blocks the remaining lags sit to its right.
    first_row = np.hstack([first_block, -blocks[0]] + later_blocks[1:]) if short_run == 1 else np.hstack([first_block] + later_blocks)
    lower_rows = k * short_run
    companion = np.vstack(
        [
            first_row,
            np.hstack([np.eye(lower_rows), np.zeros((lower_rows, k))]),
        ]
    )

    values = np.linalg.eigvals(companion)
    magnitudes = np.abs(values)
    order = np.argsort(-magnitudes, kind="stable")
    return VecStableResult(
        eigenvalues=tuple(complex(values[i]) for i in order),
        modulus=tuple(float(magnitudes[i]) for i in order),
    )
