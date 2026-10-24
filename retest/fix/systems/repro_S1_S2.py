"""S1/S2 -- ``nl`` covariance (pinv truncation) and default start values.

Run:  .venv\\Scripts\\python.exe retest/fix/systems/repro_S1_S2.py

Stata side: live StataNow 19.5 session (``retest/lib/stata_probe.py``).

S1  ``nl (mpg = {b0} + {b1}*weight + {b2}*weight^2)``: point estimates agree,
    the standard errors were off by ~7 orders of magnitude because the
    Gauss-Newton covariance used ``np.linalg.pinv(J'J)`` with numpy's default
    rcond, which silently drops the smallest eigenvalue of ``J'J``.
    "BEFORE" is the old covariance expression, reproduced here so the numbers are
    side by side; "AFTER" is the committed code.
S2  ``nl (price = {b0}*exp({b1}*mpg))`` with no starting values: westata used to
    start every parameter at 1.0, Levenberg-Marquardt stalled, and a fit with
    RSS 3.4e9 (Stata: 4.5e8) was returned behind a RuntimeWarning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import auto_frame, banner, header, rel_diff, row, Stata

import westata as ws
import westata.estimators.nonlinear as nl_mod

QUAD = "mpg = {b0} + {b1}*weight + {b2}*weight**2"
QUAD_START = {"b0": 40.0, "b1": -0.01, "b2": 0.0}
EXP = "price = {b0}*exp({b1}*mpg)"


def old_covariance(jacobian, sigma2, names):
    """The pre-fix expression, verbatim."""

    return sigma2 * np.linalg.pinv(jacobian.T @ jacobian)


def main() -> int:
    auto = auto_frame()

    with Stata() as st:
        st.run("sysuse auto, clear")
        quad = st.estimate("nl (mpg = {b0} + {b1}*weight + {b2}*weight^2)")
        quad_b, quad_se = quad.b.copy(), np.sqrt(np.diag(quad.V))
        st.estimate("nl (price = {b0}*exp({b1}*mpg))")
        exp_b = np.asarray(st.matrix("e(b)"), dtype=float).reshape(-1)
        exp_se = np.sqrt(np.diag(np.asarray(st.matrix("e(V)"), dtype=float)))
        exp_rss = float(st.scalar("e(rss)"))
        exp_names = st.matrix_names("e(b)")

    # ---------------------------------------------------------------- S1
    banner("S1  nl quadratic: covariance")
    model = ws.Nl.from_formula(QUAD, auto, start=QUAD_START)
    fit = model.fit()

    # reproduce the two alternatives at the same optimum
    names = [name.lstrip("/") for name in fit.param_names]
    predictor = lambda values: model._predictor(dict(zip(names, values)),
                                                model._frame)
    theta = fit.params.to_numpy()
    jacobian = nl_mod._jacobian(predictor, theta)
    n, k = len(auto), len(names)
    sigma2 = float(fit.e["rss"]) / (n - k)
    before_cov = old_covariance(jacobian, sigma2, names)
    before_se = np.sqrt(np.diag(before_cov))
    after_se = fit.bse.to_numpy()

    # the Stata e(b) column names are b0,b1,b2 (westata reports /b0,/b1,/b2).
    print(f"  Stata e(b) names {quad.names}")
    print(f"  singular values of J'J  : {np.linalg.svd(jacobian.T @ jacobian, compute_uv=False)}")
    print(f"  singular values of J    : {np.linalg.svd(jacobian, compute_uv=False)}")
    print(f"  s_min/s_max of J'J      : "
          f"{np.linalg.svd(jacobian.T @ jacobian, compute_uv=False)[-1] / np.linalg.svd(jacobian.T @ jacobian, compute_uv=False)[0]:.3e}"
          f"   max(shape)*eps = {max(k, k) * np.finfo(float).eps:.3e}")
    print(f"  cond(J) = {np.linalg.cond(jacobian):.4e}   "
          f"cond(J'J) = {np.linalg.cond(jacobian.T @ jacobian):.4e}")
    print(f"  rank(J) = {np.linalg.matrix_rank(jacobian)} / {k}   "
          f"rank(J'J) = {np.linalg.matrix_rank(jacobian.T @ jacobian)} / {k}"
          "   <- the cross-product is numerically singular at numpy's default cut-off")
    print()
    header()
    for i, name in enumerate(names):
        row(f"b({name})", fit.params.iloc[i], fit.params.iloc[i], quad_b[i])
    for i, name in enumerate(names):
        row(f"se({name})", before_se[i], after_se[i], quad_se[i])
    print()
    print(f"  SE rel diff: BEFORE {rel_diff(before_se, quad_se):.3e}"
          f"   AFTER {rel_diff(after_se, quad_se):.3e}")
    print(f"  coefficient rel diff: {rel_diff(fit.params.to_numpy(), quad_b):.3e}")

    # a genuinely rank-deficient Jacobian must be refused, not pseudo-solved.
    # ``{a}*x + {b}*z`` where z == x is exactly collinear: three parameters,
    # Jacobian columns [x, x, 1] only two of which are independent.
    banner("S1b  rank-deficient Jacobian must raise, not invent a standard error")
    base = np.arange(1.0, 21.0)
    frame = pd.DataFrame({"y": 2.0 * base + 3.0, "x": base, "z": base.copy()})
    rank_model = ws.Nl.from_formula(
        "y = {a}*x + {b}*z + {c}", frame, start={"a": 2.0, "b": 0.0, "c": 3.0})
    try:
        rank_model.fit(strict=False)
        print("  NO ERROR raised -> the unidentified direction got a number")
        rank_ok = False
    except Exception as exc:  # noqa: BLE001 - the type and message are the finding
        print(f"  raised {type(exc).__name__}: {exc}")
        rank_ok = type(exc).__name__ == "ConsistencyError"

    # ---------------------------------------------------------------- S2
    banner("S2  nl default starting values")
    try:
        with np.errstate(all="ignore"):
            default_fit = ws.Nl.from_formula(EXP, auto).fit()
    except Exception as exc:  # noqa: BLE001
        default_fit = None
        print(f"  no start= given -> {type(exc).__name__}: {exc}")
    if default_fit is not None:
        print(f"  no start= given -> converged={default_fit.e['converged']} "
              f"rss={default_fit.e['rss']!r}")
    explicit = ws.Nl.from_formula(
        EXP, auto, start={"b0": 20000.0, "b1": -0.06}).fit()
    print()
    header()
    for i, name in enumerate(exp_names):
        row(f"b({name})", None if default_fit is None else default_fit.params.iloc[i],
            explicit.params.iloc[i], exp_b[i])
    row("rss", None if default_fit is None else default_fit.e["rss"],
        explicit.e["rss"], exp_rss)
    print()
    print(f"  with an explicit start=: coefficient rel diff "
          f"{rel_diff(explicit.params.to_numpy(), exp_b):.3e}"
          f"   SE rel diff {rel_diff(explicit.bse.to_numpy(), exp_se):.3e}")
    print(f"  strict=False still returns the last iterate (an escape hatch, not a default):")
    lenient = ws.Nl.from_formula("price = {b0}*exp({b1}*mpg)",
                                 auto).fit(strict=False, maxiter=20)
    print(f"    converged={lenient.e['converged']}  rss={lenient.e['rss']!r}")
    try:
        ws.Nl.from_formula("price = {b0}*exp({b1}*mpg)", auto).fit(
            strict=True, maxiter=2)
        strict_raises = False
        print("  maxiter=2 with strict=True did NOT raise")
    except Exception as exc:  # noqa: BLE001
        strict_raises = type(exc).__name__ == "ConvergenceError"
        print(f"  maxiter=2, strict=True -> {type(exc).__name__}: {exc}")

    ok = (
        rel_diff(after_se, quad_se) <= 1e-9
        and rel_diff(fit.params.to_numpy(), quad_b) <= 1e-6
        and rank_ok
        # S2 chose the grid-start route: no start= now reaches Stata's optimum
        # instead of stalling, so convergence (not a refusal) is the target.
        and default_fit is not None
        and int(default_fit.e["converged"]) == 1
        and rel_diff(default_fit.params.to_numpy(), exp_b) <= 1e-6
        # ...and a genuinely non-converged fit still raises by default.
        and strict_raises
    )
    print()
    print("  RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
