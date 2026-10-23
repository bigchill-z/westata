"""AUDIT probe 1k — root cause of the ordered-model covariance error.

Upstream NOTES claim: "westata's J is exactly the transpose of the correct J".
This probe tests that claim literally and finds the *actual* mechanism:
a numpy view-aliasing bug in `_to_cut_points`.

    theta = params[n_slopes:]        # <-- VIEW, not a copy
    ...
    params[n_slopes:] = cuts         # <-- mutates theta as well
    ...
    J[k, :k] = np.exp(theta[1:k+1])  # <-- uses CUT POINTS, not log-gaps
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
orig = D._to_cut_points


def spy(params, cov, n_cuts):
    p, c = orig(params, cov, n_cuts)
    CAP.update(pin=np.asarray(params).copy(), cin=np.asarray(cov).copy(),
               pout=np.asarray(p).copy(), cout=np.asarray(c).copy(), n=n_cuts)
    return p, c


D._to_cut_points = spy

rng = np.random.default_rng(20261024)
n = 600
x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
eta = 0.5 + 0.8 * x1 - 0.4 * x2 + rng.logistic(size=n)
ycat = np.digitize(eta, [-1.0, 0.3, 1.6])
df = pd.DataFrame(dict(x1=x1, x2=x2, ycat=ycat))
ws.OrderedLogit.from_formula("ycat ~ x1 + x2", df).fit()
D._to_cut_points = orig

ns = CAP["pin"].size - CAP["n"]
theta_true = CAP["pin"][ns:]           # [-1.3917, 0.1364, 0.2220]
cuts = CAP["pout"][ns:]                # [-1.3917, -0.2456, 1.0030]
Vb = CAP["cin"][ns:, ns:]
Vout = CAP["cout"][ns:, ns:]

print("  theta (log-gap parameterisation) =", theta_true)
print("  cuts  (what westata reports)     =", cuts)
print("  exp(theta) =", np.exp(theta_true))
print("  exp(cuts)  =", np.exp(cuts))

J_correct = np.eye(3)
for k in range(1, 3):
    J_correct[k, :k] = np.exp(theta_true[1 : k + 1])

# The buggy J: because `theta` aliases `params`, after `params[ns:] = cuts`
# the slice now holds the cuts, so the exp() is applied to cut points.
J_actual = np.eye(3)
for k in range(1, 3):
    J_actual[k, :k] = np.exp(cuts[1 : k + 1])

print("\n  J_correct (from log-gaps):\n", np.array2string(J_correct, precision=10))
print("  J_actual  (from cut pts) :\n", np.array2string(J_actual, precision=10))
print("  J_correct.T              :\n", np.array2string(J_correct.T, precision=10))

print("\n  HYPOTHESIS TEST")
print(f"    Vout == J_actual  Vb J_actual'   : {np.allclose(J_actual @ Vb @ J_actual.T, Vout)}"
      f"   max|diff|={np.max(np.abs(J_actual @ Vb @ J_actual.T - Vout)):.3e}")
print(f"    Vout == J_correct Vb J_correct'  : {np.allclose(J_correct @ Vb @ J_correct.T, Vout)}")
print(f"    Vout == J_correct.T Vb J_correct : {np.allclose(J_correct.T @ Vb @ J_correct, Vout)}")
print("    -> upstream's 'transposed Jacobian' story is FALSE;")
print("       the actual bug is that exp() is applied to the CUT POINTS.")
print(f"    elementwise J_actual is lower-triangular with unit diagonal (same shape as")
print(f"    J_correct), so no transpose is involved: max|J_actual - J_correct.T| = "
      f"{np.max(np.abs(J_actual - J_correct.T)):.6f}")

with S() as s:
    s.send(df)
    b_s, V_s, nm = s.est("ologit ycat x1 x2")
se_s = np.sqrt(np.diag(V_s))
print("\n  SE CONSEQUENCES (ologit, this DGP)")
print(f"    Stata                       = {np.round(se_s[2:], 8)}")
print(f"    westata reported            = {np.round(np.sqrt(np.diag(Vout)), 8)}")
print(f"    J_correct on westata's Vb   = {np.round(np.sqrt(np.diag(J_correct @ Vb @ J_correct.T)), 8)}"
      f"  rel={rel(np.sqrt(np.diag(J_correct @ Vb @ J_correct.T)), se_s[2:]):.3e}")
print(f"    J_correct.T on westata's Vb = {np.round(np.sqrt(np.diag(J_correct.T @ Vb @ J_correct)), 8)}"
      f"  rel={rel(np.sqrt(np.diag(J_correct.T @ Vb @ J_correct)), se_s[2:]):.3e}")

print("\n  SECOND, INDEPENDENT DEFECT: slope x cut cross block is never transformed")
cross_in = CAP["cin"][:ns, ns:]
cross_out = CAP["cout"][:ns, ns:]
print("    cov_in [slope x theta] =\n", np.array2string(cross_in, precision=10))
print("    cov_out[slope x cut]   =\n", np.array2string(cross_out, precision=10))
print(f"    unchanged? {np.allclose(cross_in, cross_out)}")
print("    correct  = cov_in @ J_correct' =\n",
      np.array2string(cross_in @ J_correct.T, precision=10))
print("    ratio of correct/actual (elementwise where nonzero) =",
      np.round((cross_in @ J_correct.T) / np.where(cross_in == 0, 1, cross_in), 6))
