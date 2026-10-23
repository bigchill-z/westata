"""AUDIT probe 1j — identify the exact linear map westata applies to the cut block.

We have Vb (backend theta cov) and Vout (what westata returns). Solve for the
set of J with Vout = J Vb J'. Because that is only determined up to an
orthogonal factor, we instead test a small catalogue of candidate Jacobians
built from the source code's own ingredients, plus the *literal* code path.
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
    CAP.update(params_in=np.asarray(params).copy(), cov_in=np.asarray(cov).copy(),
               params_out=np.asarray(p).copy(), cov_out=np.asarray(c).copy(), n=n_cuts)
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

ns = CAP["params_in"].size - CAP["n"]
theta = CAP["params_in"][ns:]
Vb = CAP["cov_in"][ns:, ns:]
Vout = CAP["cov_out"][ns:, ns:]
print("  theta =", theta)
print("  exp(theta) =", np.exp(theta))
print("  Vb=\n", np.array2string(Vb, precision=10))
print("  Vout=\n", np.array2string(Vout, precision=10))

e = np.exp(theta)
cands = {
    "eye": np.eye(3),
    "code (eye + exp shifted)": np.array([[1, 0, 0], [e[1], 1, 0], [e[1], e[2], 1]]),
    "correct cumulative": np.array([[1, 0, 0], [1, e[1], 0], [1, e[1], e[2]]]),
    "correct.T": np.array([[1, 0, 0], [1, e[1], 0], [1, e[1], e[2]]]).T,
    "code.T": np.array([[1, 0, 0], [e[1], 1, 0], [e[1], e[2], 1]]).T,
    "eye+exp on cols>=": np.array([[1, e[1], e[2]], [0, 1, e[2]], [0, 0, 1]]),
}
for lbl, J in cands.items():
    cand = J @ Vb @ J.T
    print(f"  {lbl:26s} max|diff| = {np.max(np.abs(cand - Vout)):.3e}   match={np.allclose(cand, Vout)}")
    print(f"      {'':24s} se = {np.round(np.sqrt(np.diag(cand)), 8)}")

# Literal replay of the source code, with the arrays as the real call had them
params = CAP["params_in"].copy()
cov = CAP["cov_in"].copy()
n_cuts = CAP["n"]
n_slopes = params.size - n_cuts
theta2 = params[n_slopes:]
J = np.eye(n_cuts)
for k in range(1, n_cuts):
    J[k, :k] = np.exp(theta2[1 : k + 1])
block = J @ cov[n_slopes:, n_slopes:] @ J.T
print("\n  literal replay J =\n", np.array2string(J, precision=10))
print(f"  literal replay max|diff| = {np.max(np.abs(block - Vout)):.3e}")

# maybe cov_in was mutated before the call?
print("\n  cov_in diag =", np.diag(CAP["cov_in"]))
print("  Vout diag   =", np.diag(Vout))
print("  ratio diag  =", np.diag(Vout) / np.diag(Vb))

# Solve for the exact J via eigen-decomposition: Vb = Lb Lb', Vout = Lo Lo' with
# Cholesky, then J = Lo Lb^{-1} is ONE valid solution. Print it.
Lb = np.linalg.cholesky(Vb)
Lo = np.linalg.cholesky(Vout)
J_sol = Lo @ np.linalg.inv(Lb)
print("\n  one exact solution J (Cholesky-based, not unique):\n",
      np.array2string(J_sol, precision=10))
print("  check J Vb J' == Vout:", np.allclose(J_sol @ Vb @ J_sol.T, Vout))
