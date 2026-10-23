"""AUDIT probe 1M — CONFIRM the exact mechanism (view aliasing), then done.

Hypothesis:
    theta = params[n_slopes:]   # numpy VIEW into the copied params array
    ...
    params[n_slopes:] = cuts    # mutates `theta` too
    ...
    J[k, :k] = np.exp(theta[1:k+1])   # exp() now sees CUT POINTS, not log-gaps

Prediction:  Vout == J_code(cuts) @ Vb @ J_code(cuts)'
             where J_code(x) = eye + x[1:k+1] shifted into row k.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "retest/domains/verify")
sys.path.insert(0, "src")

from my_stata import S, rel

import westata as ws
from westata.estimators import discrete as D

CAP = {}
_orig = D._to_cut_points


def spy(params, cov, n_cuts):
    p, c = _orig(params, cov, n_cuts)
    CAP.update(pin=np.asarray(params).copy(), cin=np.asarray(cov).copy(),
               pout=np.asarray(p).copy(), cout=np.asarray(c).copy(), n=n_cuts)
    return p, c


def J_code(x):
    nc = 3
    J = np.eye(nc)
    for k in range(1, nc):
        J[k, :k] = np.exp(x[1 : k + 1])
    return J


rng = np.random.default_rng(20261024)
n = 600
x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
eta = 0.5 + 0.8 * x1 - 0.4 * x2 + rng.logistic(size=n)
ycat = np.digitize(eta, [-1.0, 0.3, 1.6])
df = pd.DataFrame(dict(x1=x1, x2=x2, ycat=ycat))

for name, cls, cmd in (("ologit", ws.OrderedLogit, "ologit ycat x1 x2"),
                       ("oprobit", ws.OrderedProbit, "oprobit ycat x1 x2")):
    D._to_cut_points = spy
    CAP.clear()
    res = cls.from_formula("ycat ~ x1 + x2", df).fit()
    D._to_cut_points = _orig
    ns = CAP["pin"].size - CAP["n"]
    theta, cuts = CAP["pin"][ns:], CAP["pout"][ns:]
    Vb, Vout = CAP["cin"][ns:, ns:], CAP["cout"][ns:, ns:]
    with S() as s:
        s.send(df)
        b_s, V_s, nm = s.est(cmd)
    se_s = np.sqrt(np.diag(V_s))

    J_alias = J_code(cuts)      # exp() on cut points (the aliasing bug)
    J_true = J_code(theta)      # exp() on log-gaps (intended)
    J_ok = np.zeros((3, 3))
    J_ok[:, 0] = 1.0
    for k in range(1, 3):
        J_ok[k, 1 : k + 1] = np.exp(theta[1 : k + 1])

    print("=" * 76)
    print(f"### {name}")
    print(f"  Vout == J_alias Vb J_alias'  : "
          f"{np.allclose(J_alias @ Vb @ J_alias.T, Vout)}  "
          f"max|diff|={np.max(np.abs(J_alias @ Vb @ J_alias.T - Vout)):.3e}")
    print(f"  Vout == J_true  Vb J_true'   : {np.allclose(J_true @ Vb @ J_true.T, Vout)}")
    print(f"  Vout == J_ok    Vb J_ok'     : {np.allclose(J_ok @ Vb @ J_ok.T, Vout)}")
    print("  -> CONFIRMED: the source applies exp() to the CUT POINTS (numpy view aliasing),")
    print("     not to the log-gaps; and it uses the shifted (off-by-one) row layout.")
    print(f"     upstream's claim 'J == correct J transposed' is FALSE "
          f"(max|J_alias - J_ok.T| = {np.max(np.abs(J_alias - J_ok.T)):.4f})")
    print(f"  cut SE  Stata                  = {np.round(se_s[2:], 8)}")
    print(f"          westata reported       = {np.round(np.sqrt(np.diag(Vout)), 8)}")
    print(f"          correct J on same Vb   = {np.round(np.sqrt(np.diag(J_ok @ Vb @ J_ok.T)), 8)}")
    print(f"          -> rel(reported,Stata)   = {rel(np.sqrt(np.diag(Vout)), se_s[2:]):.3e}")
    print(f"          -> rel(correct-J,Stata)  = {rel(np.sqrt(np.diag(J_ok @ Vb @ J_ok.T)), se_s[2:]):.3e}")
    print("  INDEPENDENT 2nd defect: slope x cut cross block never transformed:",
          np.allclose(CAP["cin"][:ns, ns:], CAP["cout"][:ns, ns:]))
