"""AUDIT probe 23 — zip: model-level fittedvalues vs Stata predict n; predict('n') crash?"""
import sys
import numpy as np, pandas as pd
from scipy import stats as sps
sys.path.insert(0,"retest/domains/verify"); sys.path.insert(0,"src")
from my_stata import S, rel
import westata as ws

rng=np.random.default_rng(555); n=600
x1=rng.normal(size=n); x2=rng.normal(size=n)
lam=np.exp(0.5+0.4*x1); pi=1/(1+np.exp(-(0.3+0.5*x2)))
y=np.where(rng.random(n)<pi, 0, rng.poisson(lam))
D=pd.DataFrame(dict(x1=x1,x2=x2,y=y))
with S() as s:
    s.send(D)
    s.run("quietly zip y x1 x2, inflate(x1 x2)")
    s.run("predict double zn, n")
    zn=s.var("zn")
    b=s.mat("e(b)").reshape(-1); nm=s.cnames("e(b)")
    print("  Stata e(b) names:", nm)
m=ws.ZeroInflatedPoisson.from_formula("y ~ x1 + x2", D, inflate="x1 + x2").fit() \
   if hasattr(ws.ZeroInflatedPoisson, "from_formula") else None
print("  westata param_names:", list(m.param_names))
print("  westata params:", np.round(np.asarray(m.params),10))
print("  Stata    b     :", np.round(b,10))
fv=np.asarray(m.fittedvalues)
print(f"  rel(model fittedvalues, Stata predict n) = {rel(fv, zn):.4e}")
# reconstruct (1-pi)*mu from westata's own params
pn=list(m.param_names); p=np.asarray(m.params)
print("  reconstruction check on westata params:")
try:
    p_=m.predict(kind="n")
    print(f"  rel(StataResults.predict('n'), Stata) = {rel(np.asarray(p_), zn):.4e}")
except Exception as exc:
    print(f"  StataResults.predict('n') -> {type(exc).__name__}: {str(exc)[:110]}")
