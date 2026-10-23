"""AUDIT probe 4c — qreg robust VCE: catalogue of candidate formulas vs Stata.

We hold the point estimate fixed at Stata's b and test which combination of
{bread: observation-specific density vs scalar sparsity, meat: psi^2 vs
tau(1-tau)X'X, scale: 1 vs 1/n} reproduces Stata's e(V) at tau = .25/.5/.75.
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
    q = ws.QuantileRegression.from_formula("y ~ x1 + x2", df, quantile=tau)
    ri = q.fit(cov_type="iid")
    rr = q.fit(cov_type="robust")
    pn = list(rr.param_names)
    order = [pn.index(k) for k in ("x1", "x2", "Intercept")]
    X = q._X
    beta = np.asarray(ri.params)[order]
    sp = q._sparsity
    h = sp["bandwidth"]
    f_i = (2.0 * h) / (X @ (sp["beta_upper"] - sp["beta_lower"]))
    psi = tau - ((y - X @ beta) < 0).astype(float)
    with S() as s:
        s.send(df)
        b, V, nm = s.est(f"qreg y x1 x2, quantile({tau}) vce(robust)")
    ii = [nm.index(k) for k in ("x1", "x2", "_cons")]
    Vs = V[np.ix_(ii, ii)]
    se_s = np.sqrt(np.diag(Vs))
    XtX = X.T @ X
    J_obs = (X.T * f_i) @ X
    J_sca = (1.0 / sp["sparsity"]) * XtX
    M_psi = (X.T * psi**2) @ X
    M_const = tau * (1 - tau) * XtX
    print("=" * 76)
    print(f"### tau={tau}   Stata se = {np.round(se_s,10)}")
    print(f"    westata se     = {np.round(np.asarray(rr.bse)[order],10)}"
          f"  rel={rel(np.asarray(rr.bse)[order], se_s):.3e}")
    print(f"    Stata e(sparsity)={s.e('sparsity')}  westata={sp['sparsity']}")
    for bn, J in (("J_obs", J_obs), ("J_scalar", J_sca)):
        for mn, M in (("M_psi", M_psi), ("M_const", M_const)):
            for sn, scale in (("1/n", 1.0 / n), ("1", 1.0)):
                Ji = np.linalg.inv(J)
                Vc = scale * (Ji @ M @ Ji)
                print(f"      {bn:9s} x {mn:8s} x {sn:4s}  se={np.round(np.sqrt(np.diag(Vc)),10)}"
                      f"  rel={rel(np.sqrt(np.diag(Vc)), se_s):.3e}")
