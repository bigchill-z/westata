"""AUDIT probe 6c — tseries, with correct ordering and fixed-lag unit root.

Fixes two errors in probe 6b that were MINE, not westata's:
  * VAR e(b) layout: Stata is eq-major [L.y1, L.y2, _cons] per equation, while
    statsmodels params rows are [const, L1.y1, L1.y2] — needs an explicit permute.
  * dfuller lags(2) is a FIXED lag length; passing maxlag=2 with the default
    autolag="AIC" lets statsmodels re-select the lag. Must pass autolag=None.
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
print("### T1. VAR(1) — correct e(b) permute, SE and e(Sigma) scaling")
with S() as s:
    s.send(d)
    s.run("tsset t")
    b, V, nm = s.est("var y1 y2, lags(1)")
    T_eff = s.e("N")
    Sig_s = s.mat("e(Sigma)")
    print("  Stata names:", nm, "e(N)=", T_eff, "e(k_eq)=", s.e("k_eq"), "e(k)=", s.e("k"))
    vr = ws.VAR(np.column_stack([d.y1, d.y2]), lags=1, names=["y1", "y2"]).fit()
    r = vr._result
    P = np.asarray(r.params)      # rows [const, L1.y1, L1.y2], cols [eq1, eq2]
    order = []
    for eq in range(2):
        order += [P[1, eq], P[2, eq], P[0, eq]]
    order = np.array(order)
    print("  Stata b   =", np.round(b, 12))
    print("  westata b =", np.round(order, 12), f"  rel={rel(order, b):.4e}")
    se_w = np.asarray(r.bse)
    se_w = np.array([se_w[1, eq] for eq in range(2)] + [se_w[2, eq] for eq in range(2)]
                    + [se_w[0, eq] for eq in range(2)])
    # rebuild cleanly
    se_w = np.array([se_w_[1, eq] for eq in range(2) for se_w_ in [np.asarray(r.bse)]] ) if False else None
    B = np.asarray(r.bse)
    se_w = np.array([B[1, 0], B[2, 0], B[0, 0], B[1, 1], B[2, 1], B[0, 1]])
    se_s = np.sqrt(np.diag(V))
    print("  Stata se   =", np.round(se_s, 12))
    print("  westata se =", np.round(se_w, 12), f"  raw rel={rel(se_w, se_s):.4e}")
    ratio = se_s / se_w
    k_eq = int(s.e("k_eq"))
    fac = np.sqrt((T_eff - k_eq) / T_eff)
    print(f"  ratio = {np.round(ratio, 15)}")
    print(f"  sqrt((T-k_eq)/T) = {fac:.16f}   constant ratio? {np.ptp(ratio):.2e}")
    print(f"  se rel AFTER scaling = {rel(se_w*fac, se_s):.4e}")
    Sig_w = np.asarray(r.sigma_u)
    print(f"  e(Sigma) raw rel = {rel(Sig_w.reshape(-1), Sig_s.reshape(-1)):.4e}"
          f"   after scaling = {rel((Sig_w*((T_eff-k_eq)/T_eff)).reshape(-1), Sig_s.reshape(-1)):.4e}")

print("\n" + "=" * 78)
print("### T2. dfuller with FIXED lags (autolag=None), A-tier recheck")
d3 = d.assign(y_rw=np.cumsum(np.random.default_rng(7).normal(size=T)))
with S() as s:
    s.send(d3)
    s.run("tsset t")
    for cmdtxt, reg, extra in (("dfuller y_rw, lags(2)", "c", ""),
                               ("dfuller y_rw, lags(2) trend", "ct", " trend"),
                               ("dfuller y_rw, lags(2) noconstant", "n", " noconstant")):
        s.run("quietly " + cmdtxt)
        Zt, p, Nn = s.r("Zt"), s.r("p"), s.e("N")
        w = ws.unitroot(d3["y_rw"].to_numpy(), test="adf", regression=reg,
                        maxlag=2, autolag=None)
        print(f"  [{cmdtxt}] N={Nn}")
        print(f"     Stata Zt={Zt:.16f}   westata={w.statistic:.16f}"
              f"   rel={rel([w.statistic],[Zt]):.4e}")
        print(f"     Stata p={p}   westata p={getattr(w,'pvalue',None)}"
              f"  (Stata may not store p for noconstant)")
        out = s.out("dfuller y_rw, lags(2)" + extra)
        cvs = [ln.strip() for ln in out.splitlines() if "%" in ln and "Critical" not in ln]
        print(f"     Stata printed critical values: {cvs}")

print("\n" + "=" * 78)
print("### T3. vecrank / VECM: critical value tables")
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
    print("\n".join(out.splitlines()[-26:]))
vc = ws.VECM(np.column_stack([c1, c2]), coint_rank=1, lags=2, deterministic="co")
joh = vc.johansen(lags=2)
print("\n  westata johansen =", joh)
