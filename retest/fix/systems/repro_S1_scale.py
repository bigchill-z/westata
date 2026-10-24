"""Reproduce verifier2's finding: the S1 rank guard flips with the variable's scale.

Run:  .venv\\Scripts\\python.exe retest/fix/systems/repro_S1_scale.py

The quadratic ``mpg = {b0} + {b1}*u + {b2}*u^2`` is perfectly estimable for every
positive rescaling of ``u``: ``u`` and ``u/1000`` describe the same model, and the
coefficients map onto each other exactly.  Any rank test that answers differently
for the two is measuring the units, not the identification.

Before the fix the tolerance was ``max(shape) * eps * s_max / h``: scaling a column
up by 1000 scales ``s_max`` by 1000, so the threshold grows with the units and the
same identifiable model is refused.  Column-normalising the Jacobian first removes
that dependence, which is what this script measures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import auto_frame, banner, rel_diff

import westata as ws
import westata.estimators.nonlinear as nl_mod

FORMULA = "mpg = {{b0}} + {{b1}}*{u} + {{b2}}*{u}**2"
START = {"b0": 40.0, "b1": -0.01, "b2": 0.0}

#: Rescalings of ``weight``.  ``u = weight * s`` keeps the model identifiable; the
#: fitted surface and the standard errors are the same problem in different units.
#: Both directions are covered: shrinking was the direction the verifier reported
#: as "passes", and *growing* was the one this guard actually refused.
SCALES = (1.0, 0.1, 0.01, 0.001, 1e-6, 10.0, 1e3, 1e6)


def fit_at_scale(auto: pd.DataFrame, scale: float):
    """Fit the quadratic on ``u = weight * scale``; return (result, jacobian, u)."""

    u = auto["weight"].to_numpy(dtype=float) * scale
    frame = pd.DataFrame({"mpg": auto["mpg"].to_numpy(dtype=float), "u": u})
    formula = FORMULA.format(u="u")
    model = ws.Nl.from_formula(formula, frame, start=START)
    try:
        result = model.fit()
    except Exception as exc:  # noqa: BLE001 - the refusal is the finding
        return None, None, u, exc
    names = [name.lstrip("/") for name in result.param_names]
    predictor = lambda values: model._predictor(dict(zip(names, values)),
                                                model._frame)
    jacobian = nl_mod._jacobian(predictor, result.params.to_numpy())
    return result, jacobian, u, None


def column_normalise(jacobian: np.ndarray):
    """``(J_n, D)`` with ``J_n = J @ inv(D)`` and ``D = diag(||J[:,k]||)``."""

    norms = np.linalg.norm(jacobian, axis=0)
    norms = np.where(norms > 0.0, norms, 1.0)
    return jacobian / norms, norms


def report_diagnostics(jacobian: np.ndarray, label: str) -> None:
    """Print the raw and column-normalised conditioning of one Jacobian."""

    singular = np.linalg.svd(jacobian, compute_uv=False)
    normalised, norms = column_normalise(jacobian)
    singular_n = np.linalg.svd(normalised, compute_uv=False)
    old_tolerance = (
        max(jacobian.shape) * np.finfo(float).eps * singular[0] / nl_mod._JACOBIAN_STEP
    )
    print(f"  {label}")
    print(f"    raw  singular values {singular}")
    print(f"         s_min/s_max = {singular[-1] / singular[0]:.3e}"
          f"   old threshold = {old_tolerance:.3e}"
          f"   -> {'PASS' if singular[-1] > old_tolerance else 'REFUSED'}")
    print(f"    column norms        {norms}")
    print(f"    normalised sv       {singular_n}")
    print(f"         s_min/s_max = {singular_n[-1] / singular_n[0]:.3e}"
          f"   cond(J_n) = {singular_n[0] / singular_n[-1]:.4f}")


def main() -> int:
    auto = auto_frame()

    banner("1. The same identifiable model under five rescalings of `weight`")
    results = {}
    for scale in SCALES:
        result, jacobian, u, error = fit_at_scale(auto, scale)
        results[scale] = (result, jacobian, error)
        if error is not None:
            print(f"  scale {scale:<8g} REFUSED: {type(error).__name__}: {error}")
        else:
            print(f"  scale {scale:<8g} fit OK   b = "
                  f"{np.asarray(result.params, dtype=float)}"
                  f"   max SE = {float(np.max(result.bse.to_numpy())):.9g}")

    banner("2. Diagnostics: raw vs column-normalised conditioning")
    for scale in SCALES:
        result, jacobian, error = results[scale]
        if jacobian is not None:
            report_diagnostics(jacobian, f"scale {scale:g}")
    print()
    print("  The raw singular-value ratio moves with the units; the column-normalised")
    print("  one does not.  That is the whole point: rank is a property of the")
    print("  normalised matrix, conditioning is not.")

    banner("3. Scale invariance of the standard errors")
    base_result, base_jacobian, base_error = results[1.0]
    if base_result is None:
        print("  base case refused -- cannot compare")
        return 1
    # The *coefficients* transform with the units: b1 -> b1/scale, b2 -> b2/scale^2.
    # The comparable quantities are therefore (a) the standard errors mapped back
    # to the base unit, which is what the covariance/rank code controls, and
    # (b) the coefficients, which additionally depend on where the optimiser
    # stopped.  They are reported separately because only (a) is a property of
    # this function.
    base_b = np.asarray(base_result.params, dtype=float)
    base_se = base_result.bse.to_numpy()
    print(f"  {'scale':>8} {'b0':>18} {'b2*scale^2':>18} {'se(b2)*scale^2':>19}"
          f" {'se rel':>10} {'b rel':>10}")
    worst_se = 0.0
    worst_b = 0.0
    for scale in SCALES:
        result, jacobian, error = results[scale]
        if result is None:
            print(f"  {scale:>8g} {'REFUSED':>18}")
            worst_se = worst_b = float("inf")
            continue
        b = np.asarray(result.params, dtype=float)
        se = result.bse.to_numpy()
        b2_common = b[2] * scale**2
        se2_common = se[2] * scale**2
        # standard errors: b0, b1*scale and b2*scale^2 are the same quantity
        se_rel = max(
            abs(se[0] - base_se[0]) / max(abs(base_se[0]), 1e-300),
            abs(se[1] * scale - base_se[1] * 1.0) / max(abs(base_se[1]), 1e-300),
            abs(se2_common - base_se[2] * 1.0**2) / max(abs(base_se[2]), 1e-300),
        )
        b_rel = max(
            abs(b[0] - base_b[0]) / max(abs(base_b[0]), 1e-300),
            abs(b2_common - base_b[2]) / max(abs(base_b[2]), 1e-300),
        )
        worst_se = max(worst_se, se_rel)
        worst_b = max(worst_b, b_rel)
        print(f"  {scale:>8g} {b[0]:>18.10f} {b2_common:>18.10e}"
              f" {se2_common:>19.10e} {se_rel:>10.2e} {b_rel:>10.2e}")
    print()
    print("  `b1` scales as 1/scale and `b2` as 1/scale^2; the table converts the")
    print("  quadratic coefficient and its SE back to the base unit so the numbers")
    print("  are comparable.  A correct rank test must let all five through.")
    print(f"  worst SE rel = {worst_se:.3e}   (the covariance is scale-free by"
          f" construction)")
    print(f"  worst coefficient rel = {worst_b:.3e}   (the *optimiser* stops at"
          f" slightly different points;")
    print("   same phenomenon as T8's stopping point, not a rank-test property)")

    banner("4. Exact collinearity must still be refused")
    base = np.arange(1.0, 21.0)
    frame = pd.DataFrame({"y": 2.0 * base + 3.0, "x": base, "z": base.copy()})
    model = ws.Nl.from_formula(
        "y = {a}*x + {b}*z + {c}", frame, start={"a": 2.0, "b": 0.0, "c": 3.0})
    try:
        model.fit(strict=False)
        refused = False
        print("  NO ERROR -- the collinear direction got a number")
    except Exception as exc:  # noqa: BLE001
        refused = type(exc).__name__ == "ConsistencyError"
        print(f"  raised {type(exc).__name__}: {exc}")

    banner("5. The Stata-anchored case must not fall back to the pinv truncation")
    quad = ws.Nl.from_formula(
        "mpg = {b0} + {b1}*weight + {b2}*weight**2", auto, start=START).fit()
    print(f"  se(b0) {float(quad.bse['/b0'])!r}   (Stata 5.767884338)")

    print()
    scale_ok = worst_se <= 1e-10
    stata_ok = abs(float(quad.bse["/b0"]) - 5.767884338) / 5.767884338 <= 1e-9
    ok = scale_ok and refused and stata_ok
    print(f"  scale invariance worst SE rel = {worst_se:.3e}  (target <= 1e-10)")
    print(f"  collinear case refused        = {refused}")
    print(f"  Stata SE preserved            = {stata_ok}")
    print("  RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
