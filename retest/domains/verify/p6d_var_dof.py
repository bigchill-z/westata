"""AUDIT probe 6d — VAR dof factor, with k_eq pinned to the *per-equation* count.

Upstream claims the whole E-tier difference is sqrt((T-k_eq)/T) with k_eq the
per-equation regressor count (3 for a VAR(1) with 2 variables). We confirm that
with the full Stata scalar dump, and check the e(Sigma) analogue.
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

for lags, cmd in ((1, "var y1 y2, lags(1)"), (2, "var y1 y2, lags(1/2)")):
    print("=" * 76)
    print(f"### {cmd}")
    with S() as s:
        s.send(d)
        s.run("tsset t")
        b, V, nm = s.est(cmd)
        print("  Stata names:", nm)
        print("  full e() scalar dump:", {k: v for k, v in sorted(s.st.__dict__.items()) if False})
        import json
        raw = s.st.get_ereturn()
        sc = {k: v for k, v in raw.items()
              if isinstance(v, (int, float)) and not isinstance(v, bool)}
        print("  e() scalars:", {k: round(float(v), 6) for k, v in sorted(sc.items())})
        T_eff = s.e("N")
        k_eq = int(s.e("k_eq")) if s.e("k_eq") else None
        k = int(s.e("k")) if s.e("k") else None
        Sig_s = s.mat("e(Sigma)")
        print(f"  e(N)={T_eff} e(k_eq)={k_eq} e(k)={k} e(df_m)={s.e('df_m')}")

        vr = ws.VAR(np.column_stack([d.y1, d.y2]), lags=lags, names=["y1", "y2"]).fit()
        B = np.asarray(vr._result.bse)
        # statsmodels rows: [const, L1.y1, L1.y2, (L2.y1, L2.y2)]
        nb = lags
        rows_lag = list(range(1, 1 + 2 * nb))
        se_w, b_w = [], []
        P = np.asarray(vr._result.params)
        for eq in range(2):
            for rr in rows_lag:
                se_w.append(B[rr, eq]); b_w.append(P[rr, eq])
            se_w.append(B[0, eq]); b_w.append(P[0, eq])
        se_w, b_w = np.array(se_w), np.array(b_w)
        print(f"  b rel (correct permute) = {rel(b_w, b):.4e}")
        se_s = np.sqrt(np.diag(V))
        ratio = se_s / se_w
        print(f"  se ratio (should be constant) = {np.round(ratio, 15)}  ptp={np.ptp(ratio):.2e}")
        keq_used = 1 + 2 * nb          # per-equation regressors: lags*neqs + const
        print(f"  per-equation k_eq = {keq_used}")
        fac = np.sqrt((T_eff - keq_used) / T_eff)
        print(f"  sqrt((T-k_eq)/T) = {fac:.16f}")
        print(f"  se rel AFTER scaling by sqrt((T-k_eq)/T) = {rel(se_w*fac, se_s):.4e}")
        Sig_w = np.asarray(vr._result.sigma_u)
        print(f"  e(Sigma) raw rel = {rel(Sig_w.reshape(-1), Sig_s.reshape(-1)):.4e}")
        print(f"  e(Sigma) elementwise ratio = {np.round((Sig_s/Sig_w).reshape(-1), 16)}")
        sfac = (T_eff - keq_used) / T_eff
        print(f"  (T-k_eq)/T = {sfac:.16f}")
        print(f"  e(Sigma) rel AFTER scaling = "
              f"{rel((Sig_w*sfac).reshape(-1), Sig_s.reshape(-1)):.4e}")
        # Granger wald
        gc = vr.granger_causality("y2", ["y1"])
        print(f"  westata granger(y2<-y1) wald = {gc.statistic} df={getattr(gc,'df',None)}")
        s.run("quietly " + cmd)
        s.run("vargranger")
        print(f"  Stata vargranger r() chi2 = {s.r('chi2')} df={s.r('df')}")
