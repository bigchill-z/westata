"""AUDIT probe 18 — qreg robust: verify upstream's meat-swap claim with CORRECT
coefficient ordering, and test whether it generalises across DGPs.

Upstream (limited_quantile NOTES §10) claims:
    "westata meat = sum psi_i^2 x x' ; Stata meat = tau(1-tau) X'X.
     Swapping in the latter reproduces tau=.25 to 9.0e-9 and tau=.75 to 1.4e-8."

At tau=.25 psi_i^2 in {.0625,.5625}, so the two meats genuinely differ; at
tau=.5 both equal .25 X'X. We test the claim on their exact DGP and on three
independent ones. Ordering is (x1, x2, _cons) as Stata reports it.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "retest/domains/verify")
sys.path.insert(0, "src")

from my_stata import S, rel

import westata as ws


def test(tag, df, tau):
    q = ws.QuantileRegression.from_formula("y ~ x1 + x2", df, quantile=tau)
    rr = q.fit(cov_type="robust")
    sp = q._sparsity
    X = q._X                                   # cols (x1, x2, 1) as built by patsy? check
    beta = np.asarray(rr.params)
    resid = q._y - X @ beta
    psi = tau - (resid < 0).astype(float)
    h = sp["bandwidth"]
    f_i = (2 * h) / (X @ (sp["beta_upper"] - sp["beta_lower"]))
    J = np.linalg.inv((X.T * f_i) @ X)
    M_psi = (X.T * psi**2) @ X
    M_cst = tau * (1 - tau) * (X.T @ X)
    V_psi, V_cst = J @ M_psi @ J, J @ M_cst @ J
    with S() as s:
        s.send(df)
        b, V, nm = s.est(f"qreg y x1 x2, quantile({tau}) vce(robust)")
    # westata's own column order
    pn = list(rr.param_names)
    # Stata order (x1, x2, _cons)
    def to_stata_order(vec, names):
        return np.array([vec[names.index(k)] for k in ("x1", "x2", "Intercept")])
    se_s = np.sqrt(np.diag(V))[[nm.index(k) for k in ("x1", "x2", "_cons")]]
    se_psi = to_stata_order(np.sqrt(np.diag(V_psi)), pn)
    se_cst = to_stata_order(np.sqrt(np.diag(V_cst)), pn)
    se_w = to_stata_order(np.asarray(rr.bse), pn)
    print(f"  {tag:34s} tau={tau}")
    print(f"    Stata                    = {np.round(se_s, 10)}")
    print(f"    westata as reported      = {np.round(se_w, 10)}   rel={rel(se_w, se_s):.3e}")
    print(f"    psi^2 meat (westata)     = {np.round(se_psi, 10)}   rel={rel(se_psi, se_s):.3e}")
    print(f"    tau(1-tau)X'X meat       = {np.round(se_cst, 10)}   rel={rel(se_cst, se_s):.3e}")


rng = np.random.default_rng(31)
n = 400
x1 = rng.normal(size=n); x2 = rng.normal(size=n)
y = 1.0 + 0.8 * x1 - 0.4 * x2 + rng.normal(size=n) * (1 + 0.4 * np.abs(x1))
D = pd.DataFrame(dict(x1=x1, x2=x2, y=y))
print("=" * 80)
print("### upstream's exact DGP (seed 31)")
for tau in (0.25, 0.5, 0.75):
    test("upstream DGP", D, tau)

rng = np.random.default_rng(4242)
a1 = rng.normal(size=n); a2 = rng.normal(size=n)
sg = np.exp(0.2 + 0.4 * a1)
Dh = pd.DataFrame(dict(x1=a1, x2=a2, y=1.0 + 0.75 * a1 - 0.45 * a2 + sg * rng.normal(size=n)))
print("\n" + "=" * 80)
print("### independent heteroskedastic DGP (seed 4242)")
for tau in (0.25, 0.5, 0.75):
    test("hetero DGP", Dh, tau)

rng = np.random.default_rng(99)
b1 = rng.normal(size=n); b2 = rng.normal(size=n)
Dhom = pd.DataFrame(dict(x1=b1, x2=b2, y=1.0 + 0.75 * b1 - 0.45 * b2 + rng.normal(size=n)))
print("\n" + "=" * 80)
print("### independent homoskedastic DGP (seed 99)")
for tau in (0.25, 0.5, 0.75):
    test("homo DGP", Dhom, tau)
