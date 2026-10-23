"""AUDIT probe 1h — why doesn't westata's reported cut-V equal J_code @ Vraw @ J_code'?

Prints westata's full cov alongside the backend's raw cov, in several orderings.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "retest/domains/verify")
sys.path.insert(0, "src")

from my_stata import rel

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

m = OrderedModel(y, X, distr="logit")
r = m.fit(method="bfgs", maxiter=500, disp=0)
th = np.asarray(r.params)
Vraw = np.asarray(r.cov_params())
print("  statsmodels params =", th)
print("  statsmodels cov_params =\n", np.array2string(Vraw, precision=10))
print("  exog_names =", m.exog_names)
print("  cov_params_names =", getattr(r, "cov_params_names", None))

ol = ws.OrderedLogit.from_formula("ycat ~ x1 + x2", df).fit()
print("\n  westata params =", np.asarray(ol.params), list(ol.param_names))
print("  westata cov =\n", np.array2string(np.asarray(ol.cov_params()), precision=10))

# recover the theta block from westata assuming the code Jacobian
J = np.eye(3)
for k in range(1, 3):
    J[k, :k] = np.exp(th[1 : k + 1])
print("\n  J_code =\n", np.array2string(J, precision=10))
Vw = np.asarray(ol.cov_params())[2:, 2:]
Ji = np.linalg.inv(J)
Vth_w = Ji @ Vw @ Ji.T
Vth_me = Vraw[2:, 2:]
print("  theta block: westata-inverted  vs  my raw")
print(np.array2string(Vth_w, precision=10))
print(np.array2string(Vth_me, precision=10))
print(f"  rel = {rel(Vth_w.reshape(-1), Vth_me.reshape(-1)):.3e}")
print(f"  diag rel = {rel(np.diag(Vth_w), np.diag(Vth_me)):.3e}")
print(f"  corr(offdiag) = {np.corrcoef(Vth_w.reshape(-1), Vth_me.reshape(-1))[0,1]:.8f}")

# maybe westata applies the Jacobian to the WRONG block, or applies J twice?
for lbl, cand in (
    ("J Vth J'", J @ Vth_me @ J.T),
    ("J J Vth J' J'", J @ J @ Vth_me @ J.T @ J.T),
    ("J' Vth J", J.T @ Vth_me @ J),
    ("Vth as-is", Vth_me),
):
    print(f"    {lbl:16s} se = {np.round(np.sqrt(np.diag(cand)),8)}")
print("    westata reported se =", np.round(np.sqrt(np.diag(Vw)), 8))

# is westata's bse perhaps sqrt(diag) of the *full* cov reordered?
Vwf = np.asarray(ol.cov_params())
print("\n  westata full cov diag =", np.round(np.diag(Vwf), 10))
print("  sqrt diag            =", np.round(np.sqrt(np.diag(Vwf)), 8))
print("  my raw full cov diag =", np.round(np.diag(Vraw), 10))
print("  sqrt diag            =", np.round(np.sqrt(np.diag(Vraw)), 8))
