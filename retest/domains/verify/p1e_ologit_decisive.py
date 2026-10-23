"""AUDIT probe 1e — decisive test of the ordered-model covariance claim.

Test: map Stata's e(V) *back* through the correct cumulative Jacobian
    V_theta_stata = J^{-1} V_cut_stata J^{-T}
and compare with the raw statsmodels theta covariance from an independent fit.
If they agree, the only defect is the Jacobian. Then check whether westata's
reported V is J_code V_theta J_code' for the J_code in the source.
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
res = m.fit(method="newton", disp=0, maxiter=500)
th = np.asarray(res.params, dtype=float)
Vraw = np.asarray(res.cov_params(), dtype=float)
Vth = Vraw[2:, 2:]
cuts = np.asarray(m.transform_threshold_params(th[None, :].T), dtype=float).reshape(-1)[1:-1]
gaps = np.diff(cuts)
J_ok = np.array([[1.0, 0, 0], [1.0, gaps[0], 0], [1.0, gaps[0], gaps[1]]])

# westata's source Jacobian (discrete.py:396-398)
J_code = np.eye(3)
theta = th[2:]
for kk in range(1, 3):
    J_code[kk, :kk] = np.exp(theta[1 : kk + 1])

print("  J_ok (correct, cumulative):\n", np.array2string(J_ok, precision=8))
print("  J_code (westata source):\n", np.array2string(J_code, precision=8))
print("  J_code == J_ok.T ?", np.allclose(J_code, J_ok.T))
print("  J_code == J_ok   ?", np.allclose(J_code, J_ok))

with S() as s:
    s.send(df)
    b, V, nm = s.est("ologit ycat x1 x2")
    Vs = V[2:, 2:]
    # back out theta cov from Stata
    Ji = np.linalg.inv(J_ok)
    Vth_stata = Ji @ Vs @ Ji.T
    print("\n  theta cov recovered from Stata e(V) via J_ok^{-1}:")
    print(np.array2string(Vth_stata, precision=10))
    print("  theta cov from my independent statsmodels newton fit:")
    print(np.array2string(Vth, precision=10))
    print(f"  rel(Vth_stata, Vth_mine) = {rel(Vth_stata.reshape(-1), Vth.reshape(-1)):.4e}")
    print("  => the Jacobian theory is right iff this is tiny")

    # now: does westata's reported V equal J_code Vth J_code' for *its own* theta cov?
    ol = ws.OrderedLogit.from_formula("ycat ~ x1 + x2", df).fit()
    pn = list(ol.param_names)
    Vw = np.asarray(ol.cov_params(), dtype=float)
    Vw_cut = Vw[2:, 2:]
    print("\n  westata reported cut se =", np.sqrt(np.diag(Vw_cut)))
    print("  Stata           cut se  =", np.sqrt(np.diag(Vs)))
    # invert westata's block with J_code to get its theta cov
    Jci = np.linalg.inv(J_code)
    Vth_w = Jci @ Vw_cut @ Jci.T
    print("  westata's implied theta cov (invert J_code):")
    print(np.array2string(Vth_w, precision=10))
    print(f"  rel(Vth_w, Vth_mine) = {rel(Vth_w.reshape(-1), Vth.reshape(-1)):.4e}")
    print("  westata cut se = J_code @ Vth_w @ J_code' reproduction:",
          np.sqrt(np.diag(J_code @ Vth_w @ J_code.T)))
    print("\n  What would the CUT se be with the correct J on westata's own theta cov?")
    for lbl, JJ in (("J_ok", J_ok), ("J_code", J_code), ("J_code^T", J_code.T)):
        sc = np.sqrt(np.diag(JJ @ Vth_w @ JJ.T))
        print(f"    {lbl:9s} -> {np.round(sc,8)}  rel vs Stata = {rel(sc, np.sqrt(np.diag(Vs))):.4e}")
    print("\n  Sanity: is J_code the transpose of *something* sensible?")
    print("    J_code.T @ Vth_w @ J_code ->", np.sqrt(np.diag(J_code.T @ Vth_w @ J_code)))
    print("    (upstream NOTES claims westata's J is exactly the transpose of the correct J;")
    print("     elementwise max|J_code - J_ok.T| =", np.max(np.abs(J_code - J_ok.T)), ")")
