"""AUDIT probe 24 — xtpcse: is pairwise==casewise only a defect when UNBALANCED?"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0,"retest/domains/verify"); sys.path.insert(0,"src")
from my_stata import S, rel
import westata as ws

rng = np.random.default_rng(2026); G, T = 20, 6
rows=[]
for g in range(G):
    ui = rng.normal()
    for t in range(T):
        a=rng.normal(); b=rng.normal()
        rows.append((g,t,1.0+0.8*a-0.5*b+ui+rng.normal(0,.5),a,b))
P = pd.DataFrame(rows, columns=["id","t","y","x1","x2"])  # BALANCED
with S() as s:
    s.send(P); s.run("xtset id t")
    s.run("quietly xtpcse y x1 x2");      bc,Vc = s.mat("e(b)").reshape(-1), s.mat("e(V)")
    s.run("quietly xtpcse y x1 x2, pairwise"); bp,Vp = s.mat("e(b)").reshape(-1), s.mat("e(V)")
print(f"  BALANCED: Stata casewise se = {np.sqrt(np.diag(Vc))}")
print(f"  BALANCED: Stata pairwise se = {np.sqrt(np.diag(Vp))}")
print(f"  Stata casewise==pairwise? {np.allclose(Vc,Vp)}")
mc = ws.XtPcse.from_formula("y ~ x1 + x2", P, entity="id", time="t", selection="casewise").fit()
mp = ws.XtPcse.from_formula("y ~ x1 + x2", P, entity="id", time="t", selection="pairwise").fit()
print(f"  westata casewise se = {np.asarray(mc.bse)}")
print(f"  westata pairwise se = {np.asarray(mp.bse)}")
print(f"  westata equal? {np.array_equal(np.asarray(mc.bse), np.asarray(mp.bse))}")
pn = list(mc.param_names)
print(f"  westata param_names = {pn}  <- note (Intercept first)")
se_w = np.array([float(np.asarray(mc.bse)[pn.index(k)]) for k in ("x1","x2","Intercept")])
print(f"  NAME-ALIGNED rel(westata casewise, Stata) = {rel(se_w, np.sqrt(np.diag(Vc))):.4e}")
print("  => positional comparison gives a FALSE 9.18e-01; naming is mandatory")
