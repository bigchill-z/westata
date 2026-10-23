"""Quick reconciliation: does my candidate A reproduce westata's own qreg robust V?"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0,"retest/domains/verify"); sys.path.insert(0,"src")
from my_stata import rel
import westata as ws

rng = np.random.default_rng(31); n=400
x1=rng.normal(size=n); x2=rng.normal(size=n)
y=1.0+0.8*x1-0.4*x2+rng.normal(size=n)*(1+0.4*np.abs(x1))
df=pd.DataFrame(dict(x1=x1,x2=x2,y=y))
for tau in (0.25,0.5):
    q=ws.QuantileRegression.from_formula("y ~ x1 + x2", df, quantile=tau)
    rr=q.fit(cov_type="robust")
    sp=q._sparsity
    print("names", list(rr.param_names), " sparsity keys", list(sp.keys()))
    X=q._X; beta=np.asarray(rr.params)
    print("  beta (as used for resid) =", beta)
    resid = q._y - X @ beta
    psi = tau - (resid<0).astype(float)
    h=sp["bandwidth"]; diff=sp["beta_upper"]-sp["beta_lower"]
    f_i = (2*h)/(X@diff)
    J=(X.T*f_i)@X; M=(X.T*psi**2)@X
    V=np.linalg.inv(J)@M@np.linalg.inv(J)
    print("  my V se      =", np.sqrt(np.diag(V)))
    print("  westata bse  =", np.asarray(rr.bse))
    print("  rel          =", rel(np.sqrt(np.diag(V)), np.asarray(rr.bse)))
    print("  westata cov diag =", np.diag(np.asarray(rr.cov_params())))
