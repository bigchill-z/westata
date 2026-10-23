"""AUDIT probe 12 — qreg: settle the meat-vs-bread question decisively.

At tau = .5, psi_i^2 == .25 identically, so BOTH candidates for the meat
(sum psi^2 xx' and tau(1-tau) X'X) are the SAME matrix. Therefore any
difference at tau=.5 must be in the BREAD J. Test:

    J_obs V_stata J_obs  ==?  tau(1-tau) X'X      (J_obs = westata's bread)

If not, westata's bread is not Stata's bread either.
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
x1 = rng.normal(size=n); x2 = rng.normal(size=n)
sig = np.exp(0.2 + 0.4 * x1)
yh = 1.0 + 0.75 * x1 - 0.45 * x2 + sig * rng.normal(size=n)
rng = np.random.default_rng(99)
b1 = rng.normal(size=n); b2 = rng.normal(size=n)
yhom = 1.0 + 0.75 * b1 - 0.45 * b2 + rng.normal(size=n)

for tag, (px, py, pz) in (("heteroskedastic", (x1, x2, yh)), ("homoskedastic", (b1, b2, yhom))):
    df = pd.DataFrame(dict(x1=px, x2=py, y=pz))
    X = np.column_stack([px, py, np.ones(n)])
    q = ws.QuantileRegression.from_formula("y ~ x1 + x2", df, quantile=0.5)
    ri = q.fit(cov_type="iid"); rr = q.fit(cov_type="robust")
    pn = list(rr.param_names); order = [pn.index(k) for k in ("x1", "x2", "Intercept")]
    sp = q._sparsity
    f_i = (2.0 * sp["bandwidth"]) / (X @ (sp["beta_upper"] - sp["beta_lower"]))
    J_obs = (X.T * f_i) @ X
    J_sca = sp["sparsity"] * (X.T @ X)
    tau = 0.5
    M = tau * (1 - tau) * X.T @ X
    with S() as s:
        s.send(df)
        b, Vr, nm = s.est("qreg y x1 x2, quantile(.5) vce(robust)")
        _b, Vi, _ = s.est("qreg y x1 x2, quantile(.5)")
    ii = [nm.index(k) for k in ("x1", "x2", "_cons")]
    Vr = Vr[np.ix_(ii, ii)]; Vi = Vi[np.ix_(ii, ii)]
    print("=" * 74)
    print(f"### {tag}  tau=.5")
    print(f"  Stata iid se    = {np.round(np.sqrt(np.diag(Vi)),10)}")
    print(f"  Stata robust se = {np.round(np.sqrt(np.diag(Vr)),10)}")
    print(f"  westata robust  = {np.round(np.asarray(rr.bse)[order],10)}"
          f"  rel={rel(np.asarray(rr.bse)[order], np.sqrt(np.diag(Vr))):.3e}")
    print("  A = J_obs V_stata J_obs   (should be tau(1-tau)X'X if only the meat differs)")
    print("   tau(1-tau)X'X =\n", np.array2string(M, precision=4))
    print("   J_obs V_stata J_obs =\n", np.array2string(J_obs @ Vr @ J_obs, precision=4))
    print(f"   rel = {rel((J_obs @ Vr @ J_obs).reshape(-1), M.reshape(-1)):.4e}")
    print("   J_obs V_iid_stata J_obs =\n", np.array2string(J_obs @ Vi @ J_obs, precision=4))
    print(f"   rel vs tau(1-tau)X'X = {rel((J_obs @ Vi @ J_obs).reshape(-1), M.reshape(-1)):.4e}")
    print("  V from J_obs and M: se =",
          np.round(np.sqrt(np.diag(np.linalg.inv(J_obs) @ M @ np.linalg.inv(J_obs))), 10))
    print("  V from J_sca and M: se =",
          np.round(np.sqrt(np.diag(np.linalg.inv(J_sca) @ M @ np.linalg.inv(J_sca))), 10))
    print(f"  ratio Stata_robust/westata = {np.sqrt(np.diag(Vr))/np.asarray(rr.bse)[order]}")
    # what if Stata's robust bread were the SCALAR sparsity bread with an n/(n-k)?
    for lbl, Jc in (("J_scalar", J_sca), ("J_obs", J_obs)):
        for sfx, sf in (("", 1.0), (" * n/(n-k)", n / (n - 3))):
            Vc = sf * np.linalg.inv(Jc) @ M @ np.linalg.inv(Jc)
            print(f"    {lbl}{sfx:12s} se={np.round(np.sqrt(np.diag(Vc)),10)}"
                  f"  rel={rel(np.sqrt(np.diag(Vc)), np.sqrt(np.diag(Vr))):.3e}")
