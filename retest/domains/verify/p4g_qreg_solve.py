"""AUDIT probe 4g — qreg robust at tau=.5: solve for Stata's implied BREAD exactly.

At tau=.5, psi_i = .5 - 1{u<0} takes values +.5/-.5, so psi_i^2 == .25 for every
observation on BOTH sides. The meat is therefore *identically* 0.25*X'X for Stata
and for westata, and any difference in e(V) must come from the bread J.

We solve  J V_stata J = 0.25 X'X  for symmetric positive-definite J (matrix
square-root solution) and compare the implied J against
   (i)  westata's observation-specific  J = sum f_i x_i x_i'
   (ii) a scalar                         J = c * X'X
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm

sys.path.insert(0, "retest/domains/verify")
sys.path.insert(0, "src")

from my_stata import S, rel

import westata as ws


def sym_sqrt(A):
    w, Q = np.linalg.eigh((A + A.T) / 2)
    return Q @ np.diag(np.sqrt(np.clip(w, 0, None))) @ Q.T


def solve_J(V, B):
    """J V J = B, J symmetric PD."""
    A12 = sym_sqrt(V)
    Ai12 = np.linalg.inv(A12)
    inner = sym_sqrt(A12 @ B @ A12)
    return Ai12 @ inner @ Ai12


def case(tag, x1, x2, y, tau):
    df = pd.DataFrame(dict(x1=x1, x2=x2, y=y))
    n = len(df)
    X = np.column_stack([x1, x2, np.ones(n)])
    XtX = X.T @ X
    q = ws.QuantileRegression.from_formula("y ~ x1 + x2", df, quantile=tau)
    ri = q.fit(cov_type="iid"); rr = q.fit(cov_type="robust")
    pn = list(rr.param_names)
    order = [pn.index(k) for k in ("x1", "x2", "Intercept")]
    sp = q._sparsity
    f_i = (2.0 * sp["bandwidth"]) / (X @ (sp["beta_upper"] - sp["beta_lower"]))
    with S() as s:
        s.send(df)
        b, V, nm = s.est(f"qreg y x1 x2, quantile({tau}) vce(robust)")
        s.run(f"quietly qreg y x1 x2, quantile({tau})")
        sp_s = s.e("sparsity")
        s.run(f"quietly qreg y x1 x2, quantile({tau}) vce(robust)")
    ii = [nm.index(k) for k in ("x1", "x2", "_cons")]
    Vs = V[np.ix_(ii, ii)]
    M = tau * (1 - tau) * XtX
    J_implied = solve_J(Vs, M)
    J_obs = (X.T * f_i) @ X
    J_scalar = sp_s * XtX
    print(f"\n  --- {tag}  tau={tau} ---")
    print("    Stata sparsity =", sp_s, " westata =", sp["sparsity"],
          f" rel={rel([sp['sparsity']],[sp_s]):.3e}")
    print("    J_implied (from Stata V):\n", np.array2string(J_implied, precision=4))
    print("    J_obs (westata density):\n", np.array2string(J_obs, precision=4))
    print(f"    rel(J_implied, J_obs)    = {rel(J_implied.reshape(-1), J_obs.reshape(-1)):.4e}")
    print(f"    rel(J_implied, sp*(X'X)) = "
          f"{rel(J_implied.reshape(-1), J_scalar.reshape(-1)):.4e}")
    # is J_scalar/XtX constant?
    K = J_implied / XtX
    off = K[~np.eye(3, dtype=bool)]
    print(f"    J_implied / (X'X) off-diagonal spread = {np.ptp(off):.4e}"
          f"  (small => scalar multiple)")
    c = np.median(K)
    print(f"    implied scalar c = {c:.10f}   Stata sparsity = {sp_s:.10f}"
          f"   ratio = {c/sp_s:.10f}")
    print(f"    V from J_scalar: se = "
          f"{np.round(np.sqrt(np.diag(np.linalg.inv(J_scalar) @ M @ np.linalg.inv(J_scalar))),10)}")
    print(f"    Stata se           = {np.round(np.sqrt(np.diag(Vs)),10)}")
    print(f"    V from J_obs   : se = "
          f"{np.round(np.sqrt(np.diag(np.linalg.inv(J_obs) @ M @ np.linalg.inv(J_obs))),10)}")
    print(f"    westata reported  = {np.round(np.asarray(rr.bse)[order],10)}")


rng = np.random.default_rng(4242)
n = 400
a1 = rng.normal(size=n); a2 = rng.normal(size=n)
sig = np.exp(0.2 + 0.4 * a1)
yh = 1.0 + 0.75 * a1 - 0.45 * a2 + sig * rng.normal(size=n)
rng = np.random.default_rng(99)
b1 = rng.normal(size=n); b2 = rng.normal(size=n)
yhom = 1.0 + 0.75 * b1 - 0.45 * b2 + rng.normal(size=n)

print("=" * 74)
print("### tau = .5  (meat identical on both sides -> the bread is the culprit)")
for tag, xx in (("hetero", (a1, a2, yh)), ("homo", (b1, b2, yhom))):
    case(tag, xx[0], xx[1], xx[2], 0.5)
print("\n" + "=" * 74)
print("### tau = .25 / .75")
for tau in (0.25, 0.75):
    case("hetero", a1, a2, yh, tau)
