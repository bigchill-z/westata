"""AUDIT probe 9 — panel: xtreg, fe vce(robust) 17x and vce(cluster) 9.36%.

Two independent claims to arbitrate:
  (a) Stata's `vce(robust)` after xtreg,fe is *identical* to vce(cluster panelvar)
  (b) westata's cluster SE is bigger by sqrt((N-1)/(N-3) / ((N-1)/(N-32)))
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "retest/domains/verify")
sys.path.insert(0, "src")

from my_stata import S, rel

import westata as ws

rng = np.random.default_rng(2026)
G, T = 30, 6
rows = []
for g in range(G):
    ui = rng.normal()
    for t in range(T):
        x1 = rng.normal(); x2 = rng.normal()
        rows.append((g, t, 1.0 + 0.8 * x1 - 0.5 * x2 + ui + rng.normal(0, 0.5), x1, x2))
P = pd.DataFrame(rows, columns=["id", "t", "y", "x1", "x2"])
N = len(P)

print("=" * 78)
print("### (a) is xtreg,fe vce(robust) == vce(cluster id) in Stata?")
with S() as s:
    s.send(P)
    s.run("xtset id t")
    b, V_rob, nm = s.est("xtreg y x1 x2, fe vce(robust)")
    print("  names:", nm, " e(N)=", s.e("N"), " e(df_r)=", s.e("df_r"),
          " e(N_g)=", s.e("N_g"), " e(df_m)=", s.e("df_m"))
    b2, V_cl, _ = s.est("xtreg y x1 x2, fe vce(cluster id)")
    print(f"  max|V_robust - V_cluster| = {np.max(np.abs(V_rob - V_cl))}")
    print(f"  Stata robust se = {np.sqrt(np.diag(V_rob))}")
    print(f"  Stata cluster se= {np.sqrt(np.diag(V_cl))}")
    b3, V_dfadj, _ = s.est("xtreg y x1 x2, fe vce(cluster id) dfadj")
    print(f"  Stata cluster dfadj se = {np.sqrt(np.diag(V_dfadj))}")
    print(f"  e(df_r) robust={s.e('df_r')}")
    # areg comparison
    b4, V_areg, nm4 = s.est("areg y x1 x2, absorb(id) vce(robust)")
    print(f"  areg absorb(id) vce(robust) se = {np.sqrt(np.diag(V_areg))}")

po = ws.PanelOLS.from_formula("y ~ x1 + x2", P, entity="id", time="t")
r_rob = po.fit(cov_type="robust")
r_cl = po.fit(cov_type="cluster", cov_kwds={"clusters": P["id"].to_numpy()})
r_cl_g = po.fit(cov_type="cluster", cov_kwds={"groups": P["id"].to_numpy()})
wn = list(r_rob.param_names)
print(f"\n  westata names: {wn}")
print(f"  westata robust  se = {np.asarray(r_rob.bse)}")
print(f"  westata cluster se = {np.asarray(r_cl.bse)}")
print(f"  westata df_resid: robust={r_rob.df_resid} cluster={r_cl.df_resid}")
order = [wn.index(k) for k in ("x1", "x2", "Intercept")]
print(f"  rel(westata robust, Stata robust)  = "
      f"{rel(np.asarray(r_rob.bse)[order], np.sqrt(np.diag(V_rob))):.4e}")
print(f"  rel(westata cluster, Stata cluster)= "
      f"{rel(np.asarray(r_cl.bse)[order], np.sqrt(np.diag(V_cl))):.4e}")
print(f"  rel(westata cluster, Stata dfadj)  = "
      f"{rel(np.asarray(r_cl.bse)[order], np.sqrt(np.diag(V_dfadj))):.4e}")
print(f"  rel(westata robust, areg robust)   = "
      f"{rel(np.asarray(r_rob.bse)[order], np.sqrt(np.diag(V_areg))[[nm4.index(k) for k in ('x1','x2','_cons')]]):.4e}")
fac = np.sqrt(((N - 1) / (N - 3)) / ((N - 1) / (N - 32)))
print(f"\n  predicted cluster inflation sqrt((N-1)/(N-K_reported) / ((N-1)/(N-K_absorbed)))")
print(f"    N={N} K_reported=3 K_absorbed=32 -> factor {fac:.12f}")
print(f"    observed Stata/westata cluster se ratio = "
      f"{np.sqrt(np.diag(V_cl))[np.argsort([0,1,2])]/np.asarray(r_cl.bse)[order]}")

print()
print("=" * 78)
print("### (b) xtreg, fe: does westata's e(df_r) differ (N-K vs G-1)?")
print("  (from the run above)")
print(f"  Stata cluster e(df_r) = {s.e('df_r')}" if False else "")
with S() as s:
    s.send(P); s.run("xtset id t")
    s.est("xtreg y x1 x2, fe vce(cluster id)")
    print(f"  Stata e(df_r) = {s.e('df_r')}   G-1 = {G-1}   N-K = {N-3}")
print(f"  westata df_resid = {r_cl.df_resid}")
