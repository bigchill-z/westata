"""AUDIT probe 6b — tseries A-tier recheck + the VAR dof claim, using the real API.

VARResults exposes `.params` (a DataFrame) and `._result` (the statsmodels fit);
we use both directly instead of assuming a StataResults surface.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "retest/domains/verify")
sys.path.insert(0, "src")

from my_stata import S, rel

import westata as ws

rng = np.random.default_rng(20261024)
T = 300
A = np.array([[0.5, 0.1], [-0.2, 0.4]])
Y = np.zeros((T, 2))
eps = rng.normal(size=(T, 2))
for tt in range(1, T):
    Y[tt] = A @ Y[tt - 1] + eps[tt]
d = pd.DataFrame(dict(t=np.arange(1, T + 1), y1=Y[:, 0], y2=Y[:, 1]))

print("=" * 78)
print("### T1. VAR(1) reproduced independently")
with S() as s:
    s.send(d)
    s.run("tsset t")
    b, V, nm = s.est("var y1 y2, lags(1)")
    T_eff = s.e("N")
    Sig_s = s.mat("e(Sigma)")
    print("  Stata names:", nm, " e(N)=", T_eff)
    print("  Stata b =", np.round(b, 12))
    print("  Stata se=", np.round(np.sqrt(np.diag(V)), 12))
    print("  Stata e(Sigma)=\n", np.array2string(Sig_s, precision=12))

vr = ws.VAR(np.column_stack([d.y1, d.y2]), lags=1, names=["y1", "y2"]).fit()
res = vr._result
print("\n  westata statsmodels params (DataFrame):\n", vr.params)
# statsmodels VAR orders rows per equation: for each equation [L1.y1, L1.y2, const]
P = np.asarray(res.params)                    # shape (neqs, k_eq*neqs)
print("  raw res.params shape", P.shape)
flat = P.reshape(-1, order="C")               # eq-major, within eq [L1.y1, L1.y2, const]
print("  flat =", np.round(flat, 12))
print(f"  b rel (westata flat vs Stata) = {rel(flat, b):.4e}")
se_w = np.asarray(res.bse).reshape(-1)
print("  westata se =", np.round(se_w, 12))
print(f"  se rel = {rel(se_w, np.sqrt(np.diag(V))):.4e}")
r = np.sqrt(np.diag(V)) / se_w
print(f"  se ratio Stata/westata = {np.round(r, 15)}")
k_eq = 3
print(f"  sqrt((T-k_eq)/T) = sqrt(({T_eff}-{k_eq})/{T_eff}) = {np.sqrt((T_eff-k_eq)/T_eff):.16f}")
print(f"  -> constant ratio? {np.ptp(r):.3e}")
print(f"  SE rel after rescaling westata by sqrt((T-k_eq)/T): "
      f"{rel(se_w*np.sqrt((T_eff-k_eq)/T_eff), np.sqrt(np.diag(V))):.4e}")
Sig_w = np.asarray(res.sigma_u)
print(f"\n  westata sigma_u =\n", np.array2string(Sig_w, precision=12))
print(f"  raw rel = {rel(Sig_w.reshape(-1), Sig_s.reshape(-1)):.4e}")
print(f"  elementwise ratio =\n", np.array2string(Sig_s / Sig_w, precision=15))
print(f"  (T-k_eq)/T = {(T_eff-k_eq)/T_eff:.16f}")
print(f"  rel after scaling = "
      f"{rel((Sig_w*((T_eff-k_eq)/T_eff)).reshape(-1), Sig_s.reshape(-1)):.4e}")

print("\n" + "=" * 78)
print("### T2. dfuller statistics (A-tier recheck) and printed critical values")
d3 = d.assign(y_rw=np.cumsum(np.random.default_rng(7).normal(size=T)))
with S() as s:
    s.send(d3)
    s.run("tsset t")
    for cmd, reg in (("dfuller y_rw, lags(2)", "c"),
                     ("dfuller y_rw, lags(2) trend", "ct"),
                     ("dfuller y_rw, lags(2) noconstant", "n")):
        s.run("quietly " + cmd)
        Zt = s.r("Zt")
        Nn = s.e("N")
        w = ws.unitroot(d3["y_rw"].to_numpy(), test="adf", regression=reg, maxlag=2)
        print(f"  [{cmd}] N={Nn} Stata Zt={Zt:.16f} westata={w.statistic:.16f}"
              f"  rel={rel([w.statistic],[Zt]):.4e}")
        print(f"      Stata p={s.r('p')}  westata p={getattr(w,'pvalue',None)}")
        print(f"      westata crit (5%) = {getattr(w,'critical_values',None)}")
        out = s.out("dfuller y_rw, lags(2)" + (" trend" if reg == "ct" else
                                              " noconstant" if reg == "n" else ""))
        for line in out.splitlines():
            if "Critical" in line or line.strip().startswith("1%") or \
               line.strip().startswith("5%") or line.strip().startswith("10%"):
                print("      Stata:", line.strip())

print("\n" + "=" * 78)
print("### T3. vecrank: critical values used by each side")
rng2 = np.random.default_rng(248)
m = 300
u = np.zeros(m)
for tt in range(1, m):
    u[tt] = 0.9 * u[tt - 1] + rng2.normal()
c1 = np.cumsum(rng2.normal(size=m)) + u
c2 = 1.5 * c1 + 0.5 * rng2.normal(size=m)
d2 = pd.DataFrame(dict(t=np.arange(1, m + 1), yc1=c1, yc2=c2))
with S() as s:
    s.send(d2)
    s.run("tsset t")
    out = s.out("vecrank yc1 yc2, lags(2) trend(constant) max")
    print("\n".join(out.splitlines()[-28:]))
    vc = ws.VECM(np.column_stack([c1, c2]), rank=1, lags=2, deterministic="co")
    joh = vc.johansen(lags=2)
    print("\n  westata johansen stats:", joh)
