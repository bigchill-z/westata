"""AUDIT probe 11b — xtabond lags>=2 param names, through from_formula."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0,"retest/domains/verify"); sys.path.insert(0,"src")
import westata as ws

rng = np.random.default_rng(5150)
G, T = 60, 8
rows=[]
for g in range(G):
    ui = rng.normal(); yl = 0.0
    for t in range(T):
        x1 = rng.normal(); x2 = rng.normal()
        yl = 0.5*yl + 0.8*x1 - 0.4*x2 + ui + rng.normal(0,.7)
        rows.append((g,t,yl,x1,x2))
D = pd.DataFrame(rows, columns=["id","t","y","x1","x2"])
for lags in (1,2):
    r = ws.XtAbond.from_formula("y ~ x1 + x2", D, entity="id", time="t", lags=lags).fit()
    print(f"  lags={lags}: param_names = {list(r.param_names)}")
    print(f"            n = {len(r.params)}")
    print(f"            Stata truth for a 2-regressor panel model: L1.y (+L2.y) , x1, x2")
