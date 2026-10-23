"""AUDIT probe 1c — the ologit cut-point covariance, derived from scratch.

We fit statsmodels' OrderedModel ourselves (not westata), take *its own* raw
parameterisation, derive the threshold Jacobian numerically (ground truth, no
assumption), and then check which Jacobian reproduces Stata's e(V).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "retest/domains/verify")
sys.path.insert(0, "src")

from my_stata import S, rel

import statsmodels.api as sm
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
print("  my own statsmodels OrderedModel params =", th)
print("  names:", m.exog_names, "| n_thresh:", m.k_constant)
raw_cov = np.asarray(res.cov_params(), dtype=float)

# ---- ground-truth threshold map: cut(theta) via statsmodels' own transform ----
cuts0 = np.asarray(m.transform_threshold_params(th[None, :].T), dtype=float).reshape(-1)
print("  transform_threshold_params(th) =", cuts0)
k = X.shape[1]
nc = cuts0.size
J = np.zeros((nc, th.size))
h = 1e-7
for j in range(th.size):
    tp = th.copy()
    tp[j] += h
    tm = th.copy()
    tm[j] -= h
    cp = np.asarray(m.transform_threshold_params(tp[None, :].T), dtype=float).reshape(-1)
    cm = np.asarray(m.transform_threshold_params(tm[None, :].T), dtype=float).reshape(-1)
    J[:, j] = (cp - cm) / (2 * h)
print("  numeric J (rows=cuts, cols=params):")
print(np.array2string(J, precision=6, suppress_small=True))

V_cut_true = J @ raw_cov @ J.T
se_true = np.sqrt(np.diag(V_cut_true))
print("  se(cut) with numeric J        =", se_true)

# ---- the Jacobian westata hard-codes (transposed variant), for comparison ----
Vth = raw_cov[k:, k:]
gaps = np.diff(cuts0)
J_upper = np.array([[1.0, 0.0, 0.0],
                    [1.0, gaps[0], 0.0],
                    [1.0, gaps[0], gaps[1]]])
J_lower = np.array([[1.0, 0.0, 0.0],
                    [1.0, gaps[0], 0.0],
                    [1.0, 0.0, gaps[1]]])
J_code = np.eye(3)
for kk in range(1, 3):
    J_code[kk, :kk] = np.exp(th[k + 1 : k + kk + 1])
print("  westata-style J_code          =\n", np.array2string(J_code, precision=6))
print("  J_code == J_upper.T ? ", np.allclose(J_code, J_upper.T))

with S() as s:
    s.send(df)
    b, V, names = s.est("ologit ycat x1 x2")
    print("\n  Stata b  =", b, names)
    print("  Stata se =", np.sqrt(np.diag(V)))
    # slope block check
    print(f"  slope se rel (mine vs Stata) = {rel(np.sqrt(np.diag(V_cut_true))[:0], [])}" if False else "")
    se_stata_cut = np.sqrt(np.diag(V))[k:]
    for label, JJ in (("numeric ground-truth J", J[:, k:]),
                      ("J_upper (cumulative)", J_upper),
                      ("J_lower", J_lower),
                      ("J_code (westata)", J_code),
                      ("J_upper.T", J_upper.T)):
        sc = np.sqrt(np.diag(JJ @ Vth @ JJ.T))
        print(f"    {label:26s} cut se = {np.round(sc, 8)}  rel vs Stata = {rel(sc, se_stata_cut):.4e}")

    # also compare the full covariance, incl. slope x cut blocks
    print("\n  full e(V) comparison (my numeric-J reference vs Stata):")
    V_full = np.zeros_like(V)
    V_full[:k, :k] = raw_cov[:k, :k]
    V_full[:k, k:] = raw_cov[:k, k:] @ J[:, k:].T
    V_full[k:, :k] = V_full[:k, k:].T
    V_full[k:, k:] = V_cut_true
    print(f"    full matrix rel = {rel(V_full.reshape(-1), V.reshape(-1)):.4e}")
