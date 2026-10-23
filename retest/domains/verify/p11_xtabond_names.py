"""AUDIT probe 11 — xtabond lags>=2 parameter-name offset (a dangerous silent bug)."""
import sys, inspect
import numpy as np, pandas as pd
sys.path.insert(0,"retest/domains/verify"); sys.path.insert(0,"src")
import westata as ws
print(inspect.signature(ws.XtAbond.__init__))

rng = np.random.default_rng(5150)
G, T = 60, 8
rows=[]
y = {}
for g in range(G):
    ui = rng.normal()
    yl = 0.0
    for t in range(T):
        x1 = rng.normal(); x2 = rng.normal()
        yl = 0.5*yl + 0.8*x1 - 0.4*x2 + ui + rng.normal(0,.7)
        rows.append((g,t,yl,x1,x2))
D = pd.DataFrame(rows, columns=["id","t","y","x1","x2"])
for lags in (1,2):
    exog = D[["x1","x2"]].to_numpy()
    m = ws.XtAbond(D["y"].to_numpy(), exog, D["id"].to_numpy(), D["t"].to_numpy(),
                   lags=lags, param_names=tuple(f"L{l}.y" for l in range(1, lags+1)),
                   exog_names=("x1","x2"))
    r = m.fit()
    print(f"  lags={lags}: param_names = {list(r.param_names)}")
    print(f"           n params = {len(r.params)}  (design has {exog.shape[1]} exog + {lags} lags + const)")
