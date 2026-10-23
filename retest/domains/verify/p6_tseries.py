"""AUDIT probe 6 — tseries: the "E-tier is only a per-equation dof" claim, and
the vecrank critical-value rank flip.  Plus independent A-tier rechecks.
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
# ---- bivariate VAR(1) ----
A = np.array([[0.5, 0.1], [-0.2, 0.4]])
Y = np.zeros((T, 2))
eps = rng.normal(size=(T, 2))
for tt in range(1, T):
    Y[tt] = A @ Y[tt - 1] + eps[tt]
d = pd.DataFrame(dict(t=np.arange(1, T + 1), y1=Y[:, 0], y2=Y[:, 1]))

print("=" * 78)
print("### T1. VAR(1): are the SE / e(Sigma) differences exactly (T-k_eq)/T ?")
with S() as s:
    s.send(d)
    s.run("tsset t")
    b, V, nm = s.est("var y1 y2, lags(1)")
    print("  Stata e(b) names:", nm)
    print("  Stata e(N) =", s.e("N"))
    print("  Stata b =", np.round(b, 12))
    print("  Stata se=", np.round(np.sqrt(np.diag(V)), 12))
    Sig_s = s.mat("e(Sigma)")
    print("  Stata e(Sigma) =\n", np.array2string(Sig_s, precision=12))
    var_res = ws.VAR(np.column_stack([d.y1, d.y2]), lags=1, names=["y1","y2"]).fit()
    print("  westata names:", list(var_res.param_names))
    print("  westata b =", np.round(np.asarray(var_res.params), 12))
    print("  westata se=", np.round(np.asarray(var_res.bse), 12))
    print(f"  b rel (by order) = {rel(np.asarray(var_res.params), b):.4e}")
    T_eff = s.e("N")
    print(f"  T_eff (Stata e(N)) = {T_eff};  T-k_eq for k_eq=3 = {T_eff-3}")
    ratio = np.sqrt(np.diag(V)) / np.asarray(var_res.bse)
    print(f"  se ratio Stata/westata = {np.round(ratio, 12)}")
    print(f"  sqrt((T-k_eq)/T) with k_eq=3 = {np.sqrt((T_eff-3)/T_eff):.15f}")
    print(f"  smallest denom T = {T}; T-1 = {T-1}")
    # normalize and re-compare
    norm = np.sqrt((T_eff - 3) / T_eff)
    print(f"  SE rel AFTER dividing westata se by {norm:.15f} : "
          f"{rel(np.asarray(var_res.bse)*norm, np.sqrt(np.diag(V))):.4e}")
    # e(Sigma)
    try:
        Sig_w = np.asarray(var_res.e.get("Sigma"))
        print("  westata e(Sigma) =\n", np.array2string(Sig_w, precision=12))
        print(f"  raw rel = {rel(Sig_w.reshape(-1), Sig_s.reshape(-1)):.4e}")
        r = Sig_s.reshape(-1) / Sig_w.reshape(-1)
        print(f"  elementwise ratio = {np.round(r, 12)}")
        print(f"  (T-k_eq)/T = {(T_eff-3)/T_eff:.15f}")
        print(f"  rel after scaling = {rel(Sig_w*((T_eff-3)/T_eff), Sig_s):.4e}")
    except Exception as exc:
        print("  e(Sigma) unavailable:", exc)

print("\n" + "=" * 78)
print("### T2. dfuller: are the printed critical values a hard-coded table?")
with S() as s:
    s.send(d.assign(y_rw=np.cumsum(rng.normal(size=T))))
    s.run("tsset t")
    for cmd in ("dfuller y_rw, lags(2)", "dfuller y_rw, lags(2) trend",
                "dfuller y_rw, lags(2) noconstant"):
        s.run("quietly " + cmd)
        N = s.e("N")
        print(f"  [{cmd}] e(N)={N} r(Zt)={s.r('Zt')}")
        # Stata stores crit values where? try r() scalars
        for k in ("cval1", "cval5", "cval10", "crit1", "crit5", "crit10"):
            v = s.r(k)
            if v is not None:
                print(f"      r({k}) = {v}")

print("\n" + "=" * 78)
print("### T3. vecrank: do Stata and statsmodels use different critical values?")
rng2 = np.random.default_rng(248)
m = 300
# cointegrated pair with a cointegration error AR(1) phi=0.9 to make trace(r=1) borderline
u = np.zeros(m)
for tt in range(1, m):
    u[tt] = 0.9 * u[tt - 1] + rng2.normal()
yc1 = np.cumsum(rng2.normal(size=m)) + u
yc2 = 1.5 * yc1 + 0.5 * rng2.normal(size=m)
d2 = pd.DataFrame(dict(t=np.arange(1, m + 1), yc1=yc1, yc2=yc2))
with S() as s:
    s.send(d2)
    s.run("tsset t")
    out = s.out("vecrank yc1 yc2, lags(2) trend(constant) max")
    print(out[-2500:])
