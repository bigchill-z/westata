"""``rdrobust`` bandwidth selection: the algorithm, transcribed to the point where it stops.

STATUS: PARTIAL, NOT WIRED UP. ``rdrobust`` still raises ``UnsupportedError`` when
``bandwidth`` is omitted. Nothing in the public API imports this module.

Why keep a partial file rather than delete it: the transcription below pins down facts that
were each an error in an earlier attempt, and re-deriving them is expensive. The one piece the
last step needs is produced by ``rdrobust_bw()`` inside ``l/lrdrobust.mlib``, and that is
where this stops.

WHAT ROUND 2 OF THE v3 WORK ADDED: the three COMBINATION rules, read off the ado rather than
guessed. ``combine_mserd`` / ``combine_msesum`` / ``combine_msetwo`` implement

    rdbwselect.ado:537   d_bw_d = ((C_l[1]+C_r[1]) / (C_r[2]-C_l[2])^2) ^ C[4]
    rdbwselect.ado:527   d_bw_s = ((C_l[1]+C_r[1]) / (C_r[2]+C_l[2])^2) ^ C[4]
    rdbwselect.ado:499   b_bw_l = ((C_l[1]) / (C_l[2]^2 + regul*C_l[3])) ^ C[4]

MSERD and MSESUM differ ONLY in that sign, which is the trap the note below already warned
about. The bias and estimate steps add ``scaleregul`` on top of ``C[2]^2``.

WHY THE REMAINING PIECE IS NOT A GUESSING GAME: ``mata which`` confirms ``rdrobust_bw()``
lives in ``l/lrdrobust.mlib``; that file is 321 KB of which 1.9% is printable, with the
function name appearing as an isolated string and binary either side. It is compiled bytecode
on the same footing as ``synth``'s Mata core, which GAP_REGISTER already classes as
'information unavailable, boundary reached'.

WHAT IS ESTABLISHED (read from ``rdbwselect.ado`` or measured against a live run):

1. THE SAMPLE IS 1297, NOT 1390. ``rdrobust_senate.dta`` has 1390 rows and missing values in
   ``vote``; Stata deletes them before doing anything. Every earlier attempt scaled by
   N^-1/5 with N = 1390 and standard deviations that included the missing rows, which is why
   the answers were off by whole multiples rather than by one constant factor.
2. The triangular kernel uses ``C_c = 2.576`` (Epanechnikov 2.34, uniform 1.843) --
   ``rdbwselect.ado:326-334``.
3. The pilot is ``c_bw = C_c * BWp * N^(-1/5)`` with ``BWp = min(1, (iqr/sd)/1.349)`` under
   ``stdvars(on)``, the default -- ``rdbwselect.ado:361-369, 459``. On the reference data this
   evaluates to 0.457666.
4. ``stdvars(on)`` standardises BOTH the running variable and the outcome by their standard
   deviations and rescales the cutoff -- ``rdbwselect.ado:361-368``.
5. The selector is THREE LEVELS NESTED, not one closed form: ``c_bw`` feeds ``d_bw``, which
   feeds ``b_bw``, which feeds ``h_bw``. Round 11 implemented a single formula.
6. The RD combination SUBTRACTS where the sum variant adds -- ``rdbwselect.ado:523 vs 537``::

       d_bw_s = ((C_l[1] + C_r[1]) / (C_r[2] + C_l[2])^2)^C[4]      # msesum
       d_bw_d = ((C_l[1] + C_r[1]) / (C_r[2] - C_l[2])^2)^C[4]      # mserd (default)

   Using the wrong sign gives a plausible bandwidth belonging to a different estimator.

WHAT IS NOT: ``C[1]``, ``C[2]`` and ``C[3]`` as ``rdrobust_bw()`` builds them. Treating C[1] as
a kernel-weighted residual variance and C[2] as a factorial-scaled derivative reproduces the
pilot and the first two nesting levels at the right ORDER OF MAGNITUDE (0.46 -> 0.63 -> 0.53)
but lands roughly 50x short of the reported h = 17.7543972212, so at least one of the two
carries a factor not visible from the ado alone. The kernel density at the cutoff contributes
only 1.07x, which rules density out as the explanation.

Reference: ``rdrobust vote margin`` on rdrobust_senate.dta.
    h = 17.7543972212   b = 28.0280871512
    tau = 7.4141308020  se = 1.4587160238   (se is NaN here; the vce is compiled)
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "C_C_EPANECHNIKOV",
    "C_C_TRIANGULAR",
    "C_C_UNIFORM",
    "pilot_bandwidth",
    "combine_mserd",
    "combine_msesum",
    "combine_msetwo",
    "prepare_sample",
    "rd_exponent",
]


#: ``rdbwselect.ado:326-334``.
C_C_TRIANGULAR = 2.576
C_C_EPANECHNIKOV = 2.34
C_C_UNIFORM = 1.843


def prepare_sample(
    response: np.ndarray, running: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Drop rows missing EITHER variable, as Stata does before selecting a bandwidth.

    The reference data has 1390 rows and 1297 usable ones. Scaling by the wrong N is not a
    rounding difference: ``N^(-1/5)`` appears in the pilot and the exponents compound.
    """

    y = np.asarray(response, dtype=float).reshape(-1)
    x = np.asarray(running, dtype=float).reshape(-1)
    if y.size != x.size:
        raise ValueError(
            f"rdrobust: {y.size} outcomes against {x.size} running-variable values"
        )
    keep = np.isfinite(y) & np.isfinite(x)
    return y[keep], x[keep]


def pilot_bandwidth(running: np.ndarray) -> float:
    """``c_bw = C_c * min(1, (iqr/sd)/1.349) * N^(-1/5)`` on the STANDARDISED variable.

    ``rdbwselect.ado:361-369, 459``. ``stdvars(on)`` is the default, so the running variable is
    divided by its standard deviation first and ``BWp`` is then capped at 1.
    """

    x = np.asarray(running, dtype=float).reshape(-1)
    n = x.size
    standard_deviation = float(np.std(x, ddof=1))
    if standard_deviation <= 0.0:
        raise ValueError("rdrobust: the running variable has no variation")
    standardised = x / standard_deviation
    interquartile = float(
        np.percentile(standardised, 75) - np.percentile(standardised, 25)
    )
    bandwidth_scale = min(1.0, interquartile / 1.349)
    return C_C_TRIANGULAR * bandwidth_scale * n ** (-0.2)


def rd_exponent(order: int) -> float:
    """``C[4]``: the exponent that goes with an order-``order`` fit."""

    return 1.0 / (2 * order + 3)


def derivative_scale(order: int) -> float:
    """The factorial that converts a polynomial coefficient into a derivative."""

    return float(math.factorial(order))


def combine_mserd(c_left: np.ndarray, c_right: np.ndarray) -> float:
    """``d_bw_d`` / ``b_bw_d`` / ``h_bw_d``: the default, difference form.

    ``rdbwselect.ado:537`` computes

        d_bw_d = ((C_d_l[1] + C_d_r[1])  /  (C_d_r[2] - C_d_l[2])^2) ^ C_d_l[4]

    and later adds the regularisation term for the bias and estimate bandwidths:

        b_bw_d = ((C_b_l[1] + C_b_r[1]) / ((C_b_r[2] - C_b_l[2])^2
                                          + scaleregul*(C_b_r[3] + C_b_l[3]))) ^ C_b_l[4]

    THE MINUS IS LOAD-BEARING. The MSESUM variant adds the two ``C[2]`` entries instead; using the
    wrong sign yields a plausible bandwidth belonging to a different estimator (the module docstring
    records that mistake from an earlier attempt).

    ``C[1..3]`` come from ``rdrobust_bw()`` in the Mata library, which is non-readable bytecode --
    so this function is the COMBINATION step only and expects those vectors as input.
    """

    numerator = float(c_left[0] + c_right[0])
    denominator = float(c_right[1] - c_left[1]) ** 2
    regularisation = float(c_left[2] + c_right[2])
    exponent = float(c_left[3])
    if denominator <= 0.0:
        return float("nan")
    with_regularisation = denominator + regularisation
    if with_regularisation <= 0.0:
        return float("nan")
    return numerator / with_regularisation if exponent == 1.0 else (
        max(numerator / with_regularisation, 0.0) ** exponent
    )


def combine_msesum(c_left: np.ndarray, c_right: np.ndarray) -> float:
    """``*_bw_s``: the SUM form, where the two ``C[2]`` entries are added."""

    numerator = float(c_left[0] + c_right[0])
    denominator = float(c_right[1] + c_left[1]) ** 2
    regularisation = float(c_left[2] + c_right[2])
    exponent = float(c_left[3])
    if denominator <= 0.0:
        return float("nan")
    with_regularisation = denominator + regularisation
    if with_regularisation <= 0.0:
        return float("nan")
    return max(numerator / with_regularisation, 0.0) ** exponent


def combine_msetwo(c_left: np.ndarray, c_right: np.ndarray) -> tuple[float, float]:
    """``*_bw_l`` and ``*_bw_r``: the TWO form computes each SIDE separately.

    ``rdbwselect.ado:499`` uses ``C[2]^2 + scaleregul*C[3]`` per side, with no cross-side term.
    """

    def one_side(c: np.ndarray) -> float:
        numerator = float(c[0])
        denominator = float(c[1]) ** 2 + float(c[2])
        exponent = float(c[3])
        if denominator <= 0.0:
            return float("nan")
        return max(numerator / denominator, 0.0) ** exponent

    return one_side(c_left), one_side(c_right)
