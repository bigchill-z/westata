"""AUDIT probe 1g — ordered-model cut covariance, done with the correct algebra.

statsmodels OrderedModel parameterises  theta = [b, t0, t1, t2]  where
    cut1 = t0                       (NOT logged)
    cut2 = t0 + exp(t1)
    cut3 = t0 + exp(t1) + exp(t2)
so the Jacobian of (cut1, cut2, cut3) wrt (t0, t1, t2) is

    J = [[1, 0,  0 ],
         [1, e1, 0 ],
         [1, e1, e2]]         with e_j = exp(t_j)

westata's discrete.py:396-398 builds
    J_code = eye(3); J_code[k, :k] = exp(theta[1:k+1])
        = [[1,  0,  0 ],
           [e1, 1,  0 ],
           [e1, e2, 1 ]]

We test three hypotheses with the *same* backend call westata makes:
    H1  only the Jacobian is wrong   -> J_correct reproduces Stata
    H2  upstream's "transpose" story -> J_correct.T reproduces Stata
    H3  the theta covariance is also wrong
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


def J_correct(th):
    e1, e2 = np.exp(th[3]), np.exp(th[4])
    return np.array([[1.0, 0.0, 0.0], [1.0, e1, 0.0], [1.0, e1, e2]])


def J_code(th):
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


def cut_pts(th):
    return np.array([th[2], th[2] + np.exp(th[3]),
                     th[2] + np.exp(th[3]) + np.exp(th[4])])


for name, dist, cls, cmd in (("ologit", "logit", ws.OrderedLogit, "ologit ycat x1 x2"),
                             ("oprobit", "probit", ws.OrderedProbit, "oprobit ycat x1 x2")):
    print("=" * 78)
    print(f"### {name}")
    m = OrderedModel(y, X, distr=dist)
    r_bfgs = m.fit(method="bfgs", maxiter=500, disp=0)          # == westata's call
    th, Vraw = np.asarray(r_bfgs.params), np.asarray(r_bfgs.cov_params())
    with S() as s:
        s.send(df)
        b_s, V_s, nm = s.est(cmd)
    print("  Stata   b  =", b_s, nm)
    print("  Stata   se =", np.sqrt(np.diag(V_s)))
    print("  my bfgs b  =", np.array([th[0], th[1], *cut_pts(th)]))
    print(f"  point-est rel (mine vs Stata)         = {rel(np.array([th[0], th[1], *cut_pts(th)]), b_s):.3e}")
    print(f"  cut pts (mine) = {cut_pts(th)}   Stata = {b_s[2:]}")

    Jok, Jcd = J_correct(th), J_code(th)
    print(f"\n  J_correct  =\n{np.array2string(Jok, precision=8)}")
    print(f"  J_code     =\n{np.array2string(Jcd, precision=8)}")
    print(f"  J_correct.T=\n{np.array2string(Jok.T, precision=8)}")
    print(f"  max|J_code - J_correct|   = {np.max(np.abs(Jcd - Jok)):.6f}")
    print(f"  max|J_code - J_correct.T| = {np.max(np.abs(Jcd - Jok.T)):.6f}"
          f"   -> is transpose? {np.allclose(Jcd, Jok.T)}")
    se_s = np.sqrt(np.diag(V_s))
    for lbl, J in (("J_correct", Jok), ("J_code", Jcd), ("J_correct.T", Jok.T)):
        sc = np.sqrt(np.diag(J @ Vraw[2:, 2:] @ J.T))
        fv = full_V(th, Vraw, J)
        print(f"    {lbl:12s} cut se = {np.round(sc, 8)}  rel={rel(sc, se_s[2:]):.3e}"
              f" | full e(V) rel={rel(fv.reshape(-1), V_s.reshape(-1)):.3e}")

    # cross-block check: slopes vs cuts
    print(f"\n  slope-block se rel (mine vs Stata) = {rel(np.sqrt(np.diag(Vraw))[:2], se_s[:2]):.3e}")

    # what westata itself reports
    res = cls.from_formula("ycat ~ x1 + x2", df).fit()
    Vw = np.asarray(res.cov_params())
    print(f"  westata reported full e(V) rel vs Stata = {rel(Vw.reshape(-1), V_s.reshape(-1)):.3e}")
    # reproduce westata exactly from my own bfgs
    Vrepro = full_V(th, Vraw, Jcd)
    print(f"  my (bfgs + J_code) vs westata reported  = {rel(Vrepro.reshape(-1), Vw.reshape(-1)):.3e}")
    print(f"  my (bfgs + J_code) vs Stata             = {rel(Vrepro.reshape(-1), V_s.reshape(-1)):.3e}")
    print(f"  my (bfgs + J_correct) vs Stata          = {rel(full_V(th, Vraw, Jok).reshape(-1), V_s.reshape(-1)):.3e}")
    print(f"  westata cut se reported = {np.asarray(res.bse)[2:]}")
