"""AUDIT probe 1d — what Jacobian does westata *actually* apply?

Rather than trusting the NOTES' claim ("westata's J is the transpose of the
correct one"), we recover westata's effective Jacobian empirically:
    V_reported_cut = J_eff V_theta J_eff'
and compare J_eff against (a) the correct cumulative Jacobian and (b) its
transpose.
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

m = OrderedModel(df["ycat"].to_numpy(), X, distr="logit")
res = m.fit(method="bfgs", disp=0, maxiter=2000)
th = np.asarray(res.params, dtype=float)
Vth = np.asarray(res.cov_params(), dtype=float)[2:, 2:]
cuts = np.asarray(m.transform_threshold_params(th[None, :].T), dtype=float).reshape(-1)[1:-1]
gaps = np.diff(cuts)
J_ok = np.array([[1.0, 0, 0], [1.0, gaps[0], 0], [1.0, gaps[0], gaps[1]]])

ol = ws.OrderedLogit.from_formula("ycat ~ x1 + x2", df).fit()
pn = list(ol.param_names)
Vw = np.asarray(ol.cov_params(), dtype=float)
Vw_cut = Vw[2:, 2:]
Vw_th = Vth  # same statsmodels engine; verify below
print("  westata's theta-block cov vs my own raw statsmodels cov:")
print("   ", rel(Vw_cut.reshape(-1), (J_ok @ Vw_th @ J_ok.T).reshape(-1)))
# recover J_eff assuming it is lower-triangular-ish: use Cholesky-free least squares
# V_reported = J Vth J'  -> solve for J in vec form
# vec(V) = (J kron J) vec(Vth) ; 3x3 J has 9 unknowns -> 9 equations
from itertools import product

K = np.zeros((9, 9))
for a, b in product(range(3), repeat=2):
    row = np.zeros(9)
    for i, j, kk, l in product(range(3), repeat=4):
        if kk == i and l == j:
            pass
    # build via basis
for idx, (a, b) in enumerate(product(range(3), repeat=2)):
    pass
# simpler: brute force least squares over the linear map
M = np.zeros((9, 9))
basis = []
for i, j in product(range(3), repeat=2):
    E = np.zeros((3, 3))
    E[i, j] = 1.0
    basis.append(E @ Vth @ E.T)
for col, (i, j) in enumerate(product(range(3), repeat=2)):
    # J = sum_{i,j} J_ij E_ij ; V = sum_{i,j} J_ij (E_ij Vth J' ... ) -> nonlinear
    pass
# Use scipy least_squares on the 9 unknowns instead (robust and simple).
from scipy.optimize import least_squares


def resid(p):
    J = p.reshape(3, 3)
    return (J @ Vth @ J.T - Vw_cut).reshape(-1)


sol = least_squares(resid, x0=np.eye(3).reshape(-1), xtol=1e-15, ftol=1e-15, gtol=1e-15)
J_eff = sol.x.reshape(3, 3)
print("  recovered effective Jacobian J_eff (up to sign/normalisation):")
print(np.array2string(J_eff, precision=6, suppress_small=True))
print("  residual of V = J_eff Vth J_eff' :", np.max(np.abs(resid(sol.x))))
print("  correct cumulative J_ok:")
print(np.array2string(J_ok, precision=6, suppress_small=True))
print("  J_ok.T:")
print(np.array2string(J_ok.T, precision=6, suppress_small=True))

print("\n  se comparison:")
print("    westata reported cut se =", np.sqrt(np.diag(Vw_cut)))
print("    J_eff           cut se =", np.sqrt(np.diag(J_eff @ Vth @ J_eff.T)))
print("    J_ok  (correct) cut se =", np.sqrt(np.diag(J_ok @ Vth @ J_ok.T)))
print("    J_ok.T          cut se =", np.sqrt(np.diag(J_ok.T @ Vth @ J_ok.T.T)))
with S() as s:
    s.send(df)
    b, V, nm = s.est("ologit ycat x1 x2")
    print("    Stata           cut se =", np.sqrt(np.diag(V))[2:])
    for lbl, JJ in (("J_ok (cumulative)", J_ok), ("J_ok.T", J_ok.T), ("J_eff", J_eff)):
        print(f"    rel({lbl:18s}, Stata) = {rel(np.sqrt(np.diag(JJ @ Vth @ JJ.T)), np.sqrt(np.diag(V))[2:]):.4e}")
    print(f"\n  slope block rel (westata vs Stata) = {rel(np.asarray(ol.bse)[:2], np.sqrt(np.diag(V))[:2]):.4e}")
    # full-matrix correct reference
    Vfull = np.zeros((5, 5))
    Vfull[:2, :2] = Vth[:2, :2]
    Vfull[:2, 2:] = Vth[:2, 2:] @ J_ok.T
    Vfull[2:, :2] = Vfull[:2, 2:].T
    Vfull[2:, 2:] = J_ok @ Vth[2:, 2:] @ J_ok.T
    print(f"  FULL e(V) rel with correct J (mine vs Stata) = {rel(Vfull.reshape(-1), V.reshape(-1)):.4e}")
    print(f"  FULL e(V) rel as reported (westata vs Stata) = {rel(Vw.reshape(-1), V.reshape(-1)):.4e}")
