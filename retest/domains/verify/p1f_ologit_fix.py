"""AUDIT probe 1f — replicate westata's *exact* backend call, then apply the
correct Jacobian. This distinguishes:
   (a) the only defect is the cut-point Jacobian (fix the Jacobian -> Stata),
   (b) the theta covariance itself is also different (fix is insufficient).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "retest/domains/verify")
sys.path.insert(0, "src")

from my_stata import S, rel

import westata as ws
from statsmodels.miscmodels.ordinal_model import OrderedModel

rng = np.random.default_rng(20261024)
n = 600
x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
eta = 0.5 + 0.8 * x1 - 0.4 * x2 + rng.logistic(size=n)
ycat = np.digitize(eta, [-1.0, 0.3, 1.6])
df = pd.DataFrame(dict(x1=x1, x2=x2, ycat=ycat))
X = np.column_stack([x1, x2])
y = df["ycat"].to_numpy()

m = OrderedModel(y, X, distr="logit")

# ---- exactly what westata does: bfgs, no tol, maxiter=500 ----
r_bfgs = m.fit(method="bfgs", maxiter=500, disp=0)
th_b = np.asarray(r_bfgs.params, dtype=float)
Vb = np.asarray(r_bfgs.cov_params(), dtype=float)
# ---- independent, tighter reference ----
r_new = m.fit(method="newton", maxiter=500, disp=0)
th_n = np.asarray(r_new.params, dtype=float)
Vn = np.asarray(r_new.cov_params(), dtype=float)

def make_J(th):
    # correct cumulative Jacobian, derived analytically
    e = np.exp(th[2:])
    return np.array([[1.0, 0.0, 0.0],
                     [1.0, e[0], 0.0],
                     [1.0, e[0], e[1]]])

def J_from_code(th):
    """westata's discrete.py:396-398 verbatim."""
    theta = th[2:]
    J = np.eye(3)
    for k in range(1, 3):
        J[k, :k] = np.exp(theta[1 : k + 1])
    return J

def full_V(th, Vraw, J):
    out = np.zeros((5, 5))
    out[:2, :2] = Vraw[:2, :2]
    out[:2, 2:] = Vraw[:2, 2:] @ J.T
    out[2:, :2] = out[:2, 2:].T
    out[2:, 2:] = J @ Vraw[2:, 2:] @ J.T
    return out

with S() as s:
    s.send(df)
    b, V, nm = s.est("ologit ycat x1 x2")
    print("  Stata b   =", b)
    print("  Stata se  =", np.sqrt(np.diag(V)))
    print("  Stata cut se =", np.sqrt(np.diag(V))[2:])
    for lbl, th, Vraw in (("bfgs (= westata's backend)", th_b, Vb),
                          ("newton (tighter)", th_n, Vn)):
        Jok = make_J(th)
        Jcd = J_from_code(th)
        print(f"\n  --- {lbl} ---")
        print(f"    params          = {np.round(th,10)}")
        print(f"    cut se, correct J = {np.round(np.sqrt(np.diag(Jok @ Vraw[2:,2:] @ Jok.T)),8)}"
              f"   rel={rel(np.sqrt(np.diag(Jok @ Vraw[2:,2:] @ Jok.T)), np.sqrt(np.diag(V))[2:]):.3e}")
        print(f"    cut se, code J    = {np.round(np.sqrt(np.diag(Jcd @ Vraw[2:,2:] @ Jcd.T)),8)}"
              f"   rel={rel(np.sqrt(np.diag(Jcd @ Vraw[2:,2:] @ Jcd.T)), np.sqrt(np.diag(V))[2:]):.3e}")
        print(f"    full e(V) rel, correct J = {rel(full_V(th, Vraw, Jok).reshape(-1), V.reshape(-1)):.3e}")
        print(f"    full e(V) rel, code J    = {rel(full_V(th, Vraw, Jcd).reshape(-1), V.reshape(-1)):.3e}")

    # what westata actually reports
    ol = ws.OrderedLogit.from_formula("ycat ~ x1 + x2", df).fit()
    Vw = np.asarray(ol.cov_params(), dtype=float)
    print(f"\n  westata reported full e(V) rel vs Stata = {rel(Vw.reshape(-1), V.reshape(-1)):.3e}")
    print(f"  westata bfgs params vs my bfgs params   = {rel(np.asarray(ol.params), th_b):.3e}")
    print(f"  westata full V vs my (bfgs, code J)     = {rel(Vw.reshape(-1), full_V(th_b, Vb, J_from_code(th_b)).reshape(-1)):.3e}")
    print("  => if the last is ~1e-16, the ONLY defect is the Jacobian,")
    print("     and replacing it with the correct one reproduces Stata to the row above.")

    print("\n  Is westata's code Jacobian a transpose of the correct one?")
    Jok, Jcd = make_J(th_b), J_from_code(th_b)
    print("    correct J =\n", np.array2string(Jok, precision=8))
    print("    code    J =\n", np.array2string(Jcd, precision=8))
    print("    correct J^T =\n", np.array2string(Jok.T, precision=8))
    print(f"    max|codeJ - Jok^T| = {np.max(np.abs(Jcd - Jok.T)):.6f}  (NOT a transpose)")
    print("    observed structure: row k has 1 on the diagonal and exp(theta_1..k) shifted left by one,")
    print("    i.e. the 1 and the exp-vector are swapped in position vs the correct row [1, e1, ..., ek].")
