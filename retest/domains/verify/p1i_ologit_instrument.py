"""AUDIT probe 1i — instrument westata's `_to_cut_points` to see the real inputs.

This settles the "Jacobian is transposed" claim with the actual numbers westata
uses, not an inference.
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

orig = D._to_cut_points
CAP = {}


def spy(params, cov, n_cuts):
    CAP["params_in"] = np.asarray(params, dtype=float).copy()
    CAP["cov_in"] = np.asarray(cov, dtype=float).copy()
    CAP["n_cuts"] = n_cuts
    p, c = orig(params, cov, n_cuts)
    CAP["params_out"] = np.asarray(p, dtype=float).copy()
    CAP["cov_out"] = np.asarray(c, dtype=float).copy()
    return p, c


D._to_cut_points = spy

rng = np.random.default_rng(20261024)
n = 600
x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
eta = 0.5 + 0.8 * x1 - 0.4 * x2 + rng.logistic(size=n)
ycat = np.digitize(eta, [-1.0, 0.3, 1.6])
df = pd.DataFrame(dict(x1=x1, x2=x2, ycat=ycat))

res = ws.OrderedLogit.from_formula("ycat ~ x1 + x2", df).fit()
with S() as s:
    s.send(df)
    b_s, V_s, nm = s.est("ologit ycat x1 x2")

th = CAP["params_in"]
ns = th.size - CAP["n_cuts"]
theta = th[ns:]
print("  n_cuts =", CAP["n_cuts"], " n_slopes =", ns)
print("  params_in  =", th)
print("  theta      =", theta)
print("  cov_in diag=", np.diag(CAP["cov_in"]))
print("  cov_out diag=", np.diag(CAP["cov_out"]))

J_code = np.eye(3)
for k in range(1, 3):
    J_code[k, :k] = np.exp(theta[1 : k + 1])
print("\n  J_code =\n", np.array2string(J_code, precision=10))
Vb = CAP["cov_in"][ns:, ns:]
print("  J_code @ Vb @ J_code' =\n", np.array2string(J_code @ Vb @ J_code.T, precision=10))
print("  actual cov_out block  =\n", np.array2string(CAP["cov_out"][ns:, ns:], precision=10))
print("  equal?", np.allclose(J_code @ Vb @ J_code.T, CAP["cov_out"][ns:, ns:]))

# The CORRECT Jacobian
e1, e2 = np.exp(theta[1]), np.exp(theta[2])
J_ok = np.array([[1.0, 0.0, 0.0], [1.0, e1, 0.0], [1.0, e1, e2]])
print("\n  J_correct =\n", np.array2string(J_ok, precision=10))
print("  J_ok.T    =\n", np.array2string(J_ok.T, precision=10))
se_s = np.sqrt(np.diag(V_s))
for lbl, J in (("J_code", J_code), ("J_correct", J_ok), ("J_correct.T", J_ok.T)):
    sc = np.sqrt(np.diag(J @ Vb @ J.T))
    print(f"    {lbl:12s} cut se = {np.round(sc,8)}  rel vs Stata = {rel(sc, se_s[2:]):.4e}")
print("    Stata        cut se =", np.round(se_s[2:], 8))
print("    reported     cut se =", np.round(np.sqrt(np.diag(CAP['cov_out'])[ns:]), 8))

# does the cross block get transformed?
print("\n  cross block cov_in[0:2, 2:] =\n", np.array2string(CAP["cov_in"][:ns, ns:], precision=10))
print("  cross block cov_out[0:2,2:] =\n", np.array2string(CAP["cov_out"][:ns, ns:], precision=10))
print("  cross block transformed?", not np.allclose(CAP["cov_in"][:ns, ns:], CAP["cov_out"][:ns, ns:]))
print("  => correct cross block should be  cov_in[0:2,2:] @ J' =")
print(np.array2string(CAP["cov_in"][:ns, ns:] @ J_ok.T, precision=10))

# rebuild upstream's claimed verification
print("\n  What upstream's NOTES report for this DGP (their data):")
print("    they claim 'westata J == (correct J)^T'.  Here: max|J_code - J_ok.T| =",
      np.max(np.abs(J_code - J_ok.T)))
print("    and J_code is upper-triangular-with-unit-diagonal vs J_ok lower-triangular;")
print("    the swap is: row k holds [exp(theta_1..k), 1] instead of [1, exp(theta_1..k)].")
D._to_cut_points = orig
