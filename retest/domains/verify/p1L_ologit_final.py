"""AUDIT probe 1L — FINAL ordered-model root cause, with the Jacobian right.

Correct Jacobian of (cut1,cut2,cut3) wrt (t0,t1,t2), where cut1=t0,
cut2=t0+exp(t1), cut3=cut2+exp(t2):

    J[k, 0]      = 1                 for all k   (every cut inherits t0)
    J[k, 1..k]   = exp(t_1..t_k)     for k >= 1  (only affects this cut and later)

i.e.  J = [[1, 0,   0  ],
           [1, e1,  0  ],
           [1, e1,  e2 ]]

The earlier `J[k, :k] = exp(...)` form (used in probes 1k) shifts the e-columns
one place left, which is exactly the bug under investigation — so build it
explicitly here.
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


def J_ok_from(theta):
    """theta = [t0, t1, ..., t_{k-1}] -> correct cumulative Jacobian."""
    nc = theta.size
    J = np.zeros((nc, nc))
    J[:, 0] = 1.0
    for k in range(1, nc):
        J[k, 1 : k + 1] = np.exp(theta[1 : k + 1])
    return J


def J_code_from(theta2, nc=3):
    """westata discrete.py:396-398, verbatim."""
    J = np.eye(nc)
    for k in range(1, nc):
        J[k, :k] = np.exp(theta2[1 : k + 1])
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
    theta = CAP["pin"][ns:]
    cuts = CAP["pout"][ns:]
    Vb = CAP["cin"][ns:, ns:]
    Vout = CAP["cout"][ns:, ns:]
    with S() as s:
        s.send(df)
        b_s, V_s, nm = s.est(cmd)
    se_s = np.sqrt(np.diag(V_s))

    J_ok = J_ok_from(theta)
    J_code = J_code_from(theta)
    J_alias = J_ok_from(cuts)  # exp applied to the cut points (the view bug)

    print("=" * 78)
    print(f"### {name}   (theta={np.round(theta,8)}, cuts={np.round(cuts,8)})")
    print("  J_correct =\n", np.array2string(J_ok, precision=10))
    print("  J_code    =\n", np.array2string(J_code, precision=10))
    print("  J_alias   =\n", np.array2string(J_alias, precision=10))
    print(f"  Vout == J_alias Vb J_alias'  : {np.allclose(J_alias @ Vb @ J_alias.T, Vout)}"
          f"  (max|diff| {np.max(np.abs(J_alias @ Vb @ J_alias.T - Vout)):.3e})")
    print(f"  Vout == J_code  Vb J_code'   : {np.allclose(J_code @ Vb @ J_code.T, Vout)}")
    print(f"  Vout == J_correct Vb J_correct: {np.allclose(J_ok @ Vb @ J_ok.T, Vout)}")
    print("\n  cut SE")
    print(f"    Stata                    = {np.round(se_s[2:], 8)}")
    print(f"    westata reported         = {np.round(np.sqrt(np.diag(Vout)), 8)}"
          f"   rel={rel(np.sqrt(np.diag(Vout)), se_s[2:]):.3e}")
    for lbl, J in (("J_correct", J_ok), ("J_code", J_code), ("J_alias", J_alias)):
        sc = np.sqrt(np.diag(J @ Vb @ J.T))
        print(f"    {lbl:10s} on westata's Vb = {np.round(sc,8)}   rel={rel(sc, se_s[2:]):.3e}")
    # full matrix with correct J including the cross block
    Vfull = np.zeros((ns + 3, ns + 3))
    Vfull[:ns, :ns] = CAP["cin"][:ns, :ns]
    Vfull[:ns, ns:] = CAP["cin"][:ns, ns:] @ J_ok.T
    Vfull[ns:, :ns] = Vfull[:ns, ns:].T
    Vfull[ns:, ns:] = J_ok @ Vb @ J_ok.T
    print(f"\n  FULL e(V) rel: correct-J rebuild vs Stata = "
          f"{rel(Vfull.reshape(-1), V_s.reshape(-1)):.3e}")
    print(f"  FULL e(V) rel: westata reported vs Stata  = "
          f"{rel(np.asarray(res.cov_params()).reshape(-1), V_s.reshape(-1)):.3e}")
