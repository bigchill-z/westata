"""Least-squares building block shared by the resampling and panel code (R6.1).

Three modules carried their own ``_ols``:

* ``panel_iv.py`` and ``resample.py`` computed the same thing (``lstsq`` coefficients)
  under two different parameter-name conventions;
* ``firststage.py`` computed something else entirely (fitted values plus R-squared,
  with ``(y, design)`` argument order) and has been renamed
  ``_fitted_and_r2`` -- it was a naming collision, not a duplicate.

The first two now call :func:`ols_coefficients`.
"""

from __future__ import annotations

import numpy as np


def ols_coefficients(design: np.ndarray, response: np.ndarray) -> np.ndarray:
    """Least squares via LAPACK ``lstsq`` (QR), never the normal equations.

    Squaring the condition number is the difference between a usable coefficient and a
    garbage one on the ill-conditioned designs this package routinely fits (absorbed
    fixed effects, near-collinear regressors, bootstrap resamples of those).
    """

    coefficients, *_ = np.linalg.lstsq(
        np.asarray(design, dtype=float), np.asarray(response, dtype=float), rcond=None
    )
    return np.asarray(coefficients, dtype=float)


#: Spellings of the constant term across patsy, statsmodels and this package.
CONSTANT_NAMES = frozenset({"Intercept", "_cons", "const", "intercept", "1"})


def slope_positions(
    names: list[str] | tuple[str, ...],
    *,
    auxiliary_prefixes: tuple[str, ...] = ("/",),
    excluded_equations: tuple[str, ...] = (),
    extra_auxiliary: tuple[str, ...] = (),
    split_equation_prefix: bool = True,
) -> list[int]:
    """Indices of the coefficients a Stata Wald/LR test covers.

    Three estimators carried their own version (R6.1) and they agreed only on the
    easy part -- "skip the constant" -- while differing on every hard part:

    ==============  =========================  ==================================
    family          auxiliary parameters       equation-prefixed names
    ==============  =========================  ==================================
    ML (discrete)   names starting ``/``       NOT handled (no such names)
    limited         ``/sigma``, ``lambda``,    ``select:``/``treat:`` excluded
                    control function
    systems         ``/``                     ``y1:_cons`` understood
    ==============  =========================  ==================================

    Rather than one rule with three exceptions buried inside, the shared part is
    here and each family passes only the differences.  A family that forgets one
    now passes a visibly empty default instead of re-deriving the rule.

    ``split_equation_prefix`` decides whether ``y1:_cons`` counts as a constant: with
    equation prefixes the CONSTANT is the leaf after the colon.
    """

    positions: list[int] = []
    for position, raw_name in enumerate(names):
        name = str(raw_name)
        if name.startswith(auxiliary_prefixes):
            continue
        if excluded_equations and name.startswith(excluded_equations):
            continue
        leaf = name.split(":", 1)[-1] if split_equation_prefix else name
        if leaf in extra_auxiliary:
            continue
        if leaf in CONSTANT_NAMES:
            continue
        positions.append(position)
    return positions


#: Central-difference step, scaled by ``max(|x|, 1)`` per coordinate.
HESSIAN_STEP = 1e-5


def numerical_hessian(
    objective: Callable[[np.ndarray], float],
    point: np.ndarray,
    *,
    gradient: Callable[[np.ndarray], np.ndarray] | None = None,
    step: float | None = None,
) -> np.ndarray:
    """Central-difference Hessian, of the objective or of its analytic gradient.

    Three modules carried a copy of this (R6.1) and all three used the same step and
    the same formulae -- but only ``nonlinear.py`` could differentiate a CLOSED-FORM
    gradient, and ``tobit.py`` re-evaluated the objective inside its double loop
    (harmless, wasteful, and a place for the three to drift).  Both behaviours are
    kept here: ``gradient`` is optional, the centre is evaluated once.

    ``gradient`` switches the routine to differentiating a *closed-form* gradient.
    That matters for the frontier likelihood, which is almost flat along
    ``sigma_u -> 0``: second-differencing the objective itself loses roughly half its
    digits to cancellation there, while the gradient is available analytically and its
    central difference stays first order.
    """

    point = np.asarray(point, dtype=float)
    size = point.size
    increments = (HESSIAN_STEP if step is None else float(step)) * np.maximum(
        np.abs(point), 1.0
    )

    if gradient is not None:
        hessian = np.empty((size, size), dtype=float)
        for i in range(size):
            ei = np.zeros(size)
            ei[i] = increments[i]
            hessian[:, i] = (gradient(point + ei) - gradient(point - ei)) / (
                2.0 * increments[i]
            )
        return 0.5 * (hessian + hessian.T)

    hessian = np.zeros((size, size), dtype=float)
    centre = float(objective(point))
    for i in range(size):
        ei = np.zeros(size)
        ei[i] = increments[i]
        for j in range(i, size):
            ej = np.zeros(size)
            ej[j] = increments[j]
            if i == j:
                value = (
                    objective(point + ei) - 2.0 * centre + objective(point - ei)
                ) / increments[i] ** 2
            else:
                value = (
                    objective(point + ei + ej)
                    - objective(point + ei - ej)
                    - objective(point - ei + ej)
                    + objective(point - ei - ej)
                ) / (4.0 * increments[i] * increments[j])
            hessian[i, j] = value
            hessian[j, i] = value
    return hessian
