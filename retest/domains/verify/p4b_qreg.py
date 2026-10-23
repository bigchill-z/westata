"""AUDIT probe 4b — qreg robust meat: calibrate against Stata empirically.

Instead of guessing, we take westata's own bread J and solve for the meat that
Stata's V implies, then compare that meat with the two candidates.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "retest/domains/verify")
sys.path.insert(0, "src")

from my_stata import S, rel

import westata as ws

rng = np.random.default_rng(4242)
n = 400
x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
sig = np.exp(0.2 + 0.4 * x1)
y = 1.0 + 0.75 * x1 - 0.45 * x2 + sig * rng.normal(size=n)
df = pd.DataFrame(dict(x1=x1, x2=x2, y=y))

for tau in (0.25, 0.5, 0.75):
    print("=" * 74)
    print(f"### tau = {tau}")
    q = ws.QuantileRegression.from_formula("y ~ x1 + x2", df, quantile=tau)
    ri = q.fit(cov_type="iid")
    rr = q.fit(cov_type="robust")
    pn = list(rr.param_names)
    order = [pn.index(k) for k in ("x1", "x2", "Intercept")]
    sp = q._sparsity
    X = q._X
    beta = np.asarray(ri.params)[order]
    h = sp["bandwidth"]
    diff = sp["beta_upper"] - sp["beta_lower"]
    f_i = (2.0 * h) / (X @ diff)
    J = (X.T * f_i) @ X
    psi = tau - ((y - X @ beta) < 0).astype(float)
    M_psi = (X.T * psi**2) @ X
    Ji = np.linalg.inv(J)

    with S() as s:
        s.send(df)
        b, V, nm = s.est(f"qreg y x1 x2, quantile({tau}) vce(robust)")
        _b0, V_iid_stata, _ = s.est(f"qreg y x1 x2, quantile({tau})")
        print("  Stata robust se =", np.round(np.sqrt(np.diag(V)), 10))
        print("  Stata iid    se =", np.round(np.sqrt(np.diag(V_iid_stata)), 10), " (e(V) differs?:",
              np.max(np.abs(V - V_iid_stata)) > 0)
        sp_stata = s.e("sparsity")
        print(f"  Stata e(sparsity) = {sp_stata}   westata sparsity = {sp['sparsity']}")
        print(f"  Stata e(bwidth)   = {s.e('bwidth')}   westata bandwidth = {h}")

        print("  westata iid    se =", np.round(np.asarray(ri.bse)[order], 10),
              f" rel={rel(np.asarray(ri.bse)[order], np.sqrt(np.diag(V_iid_stata))):.3e}")
        print("  westata robust se =", np.round(np.asarray(rr.bse)[order], 10),
              f" rel={rel(np.asarray(rr.bse)[order], np.sqrt(np.diag(V))):.3e}")

        # Stata's implied meat
        Vs = V[np.ix_([nm.index(k) for k in ("x1", "x2", "_cons")],
                      [nm.index(k) for k in ("x1", "x2", "_cons")])]
        M_implied = J @ Vs @ J
        print("\n  M_implied = J V_stata J  (westata's own bread):")
        print(np.array2string(M_implied, precision=4))
        print("  M_psi^2 (westata candidate):")
        print(np.array2string(M_psi, precision=4))
        print("  tau(1-tau) X'X :")
        print(np.array2string(tau * (1 - tau) * X.T @ X, precision=4))
        print(f"  rel(M_implied, M_psi)          = {rel(M_implied.reshape(-1), M_psi.reshape(-1)):.4e}")
        print(f"  rel(M_implied, tau(1-tau)X'X)  = "
              f"{rel(M_implied.reshape(-1), (tau*(1-tau)*X.T@X).reshape(-1)):.4e}")
        for lbl, M in (("psi^2", M_psi), ("tau(1-tau)X'X", tau * (1 - tau) * X.T @ X)):
            Vc = Ji @ M @ Ji
            print(f"    V from {lbl:16s} se = {np.round(np.sqrt(np.diag(Vc)), 10)}"
                  f"  rel vs Stata = {rel(np.sqrt(np.diag(Vc)), np.sqrt(np.diag(V))):.4e}")
        print("  ratio Stata/westata se =",
              np.round(np.sqrt(np.diag(V)) / np.asarray(rr.bse)[order], 10))
