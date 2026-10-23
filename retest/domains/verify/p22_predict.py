"""AUDIT probe 22 — remaining discrete `predict` claims and e(V) full-matrix."""
import sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0,"retest/domains/verify"); sys.path.insert(0,"src")
from my_stata import S, rel
import westata as ws

rng=np.random.default_rng(555); n=600
x1=rng.normal(size=n); x2=rng.normal(size=n)
y_zi=np.where(rng.random(n)<0.3, 0, rng.poisson(np.exp(0.5+0.4*x1)))
D=pd.DataFrame(dict(x1=x1,x2=x2,y_zi=y_zi))
with S() as s:
    s.send(D)
    s.run("quietly zip y_zi x1 x2, inflate(x1 x2)")
    s.run("predict double zi_n, n")
    zn=s.var("zi_n")
    print("  Stata zip predict n: first 3 =", zn[:3])
m=ws.ZeroInflatedPoisson.from_formula("y_zi ~ x1 + x2", D).fit()
print("  westata model fittedvalues first 3 =", np.asarray(m.fittedvalues)[:3],
      f" rel={rel(np.asarray(m.fittedvalues), zn):.3e}")
try:
    p=m.predict(kind="n")
    print("  westata StataResults.predict('n') ran:", np.asarray(p)[:3])
except Exception as exc:
    print(f"  westata StataResults.predict('n') -> {type(exc).__name__}: {str(exc)[:100]}")

# ologit predict pr
rng=np.random.default_rng(20261024)
a1=rng.normal(size=n); a2=rng.normal(size=n)
eta=0.5+0.8*a1-0.4*a2+rng.logistic(size=n)
ycat=np.digitize(eta,[-1.0,0.3,1.6])
D2=pd.DataFrame(dict(x1=a1,x2=a2,ycat=ycat))
ol=ws.OrderedLogit.from_formula("ycat ~ x1 + x2", D2).fit()
try:
    p=ol.predict(kind="pr")
    print("  westata ologit predict pr ran:", np.shape(p))
except Exception as exc:
    print(f"  westata ologit predict('pr') -> {type(exc).__name__}: {str(exc)[:110]}")
with S() as s:
    s.send(D2)
    s.run("quietly ologit ycat x1 x2")
    s.run("predict double ol_pr, pr")
    print("  Stata ologit predict pr: first 5 =", s.var("ol_pr")[:5])
