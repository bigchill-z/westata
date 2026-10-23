"""AUDIT probe 13 — is qreg vce(robust) at tau=.5 really verified?

At tau=.5, psi_i^2 == .25 for every i, so sum psi^2 xx' == .25 X'X on BOTH sides.
Consequently V_stata = J_s^-1 (.25 X'X) J_s^-1 and V_westata = J_w^-1 (.25 X'X) J_w^-1,
i.e. the entire difference lives in the BREAD. We test whether each side's implied
bread is a SCALAR multiple of (X'X) (=> V proportional to (X'X)^-1).
"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0,"retest/domains/verify"); sys.path.insert(0,"src")
from my_stata import S, rel
import westata as ws

def case(tag, x1, x2, y):
    df = pd.DataFrame(dict(x1=x1, x2=x2, y=y)); n=len(df)
    X = np.column_stack([x1,x2,np.ones(n)]); XtXi = np.linalg.inv(X.T@X)
    with S() as s:
        s.send(df)
        b,Vr,nm = s.est("qreg y x1 x2, quantile(.5) vce(robust)")
        _b,Vi,_ = s.est("qreg y x1 x2, quantile(.5)")
    ii=[nm.index(k) for k in ("x1","x2","_cons")]
    Vr=Vr[np.ix_(ii,ii)]; Vi=Vi[np.ix_(ii,ii)]
    q = ws.QuantileRegression.from_formula("y ~ x1 + x2", df, quantile=.5)
    ri=q.fit(cov_type="iid"); rr=q.fit(cov_type="robust")
    pn=list(rr.param_names); order=[pn.index(k) for k in ("x1","x2","Intercept")]
    print("="*72); print(f"### {tag}")
    print(f"  Stata robust se = {np.round(np.sqrt(np.diag(Vr)),10)}")
    print(f"  westata robust  = {np.round(np.asarray(rr.bse)[order],10)}"
          f"   rel = {rel(np.asarray(rr.bse)[order], np.sqrt(np.diag(Vr))):.3e}")
    for lbl,V in (("Stata", Vr), ("westata", np.asarray(rr.cov_params())[np.ix_(order,order)])):
        R = V/XtXi
        off = R[~np.eye(3,dtype=bool)]
        print(f"  {lbl:8s} V/(X'X)^-1: diag={np.round(np.diag(R),6)}"
              f"  offdiag={np.round(off,6)}  constant? {np.allclose(R, R[0,0]*np.ones((3,3)))}")
    print(f"  Stata/westata se ratio = {np.round(np.sqrt(np.diag(Vr))/np.asarray(rr.bse)[order],10)}")

rng=np.random.default_rng(4242); n=400
x1=rng.normal(size=n); x2=rng.normal(size=n); s_=np.exp(0.2+0.4*x1)
case("heteroskedastic", x1, x2, 1.0+0.75*x1-0.45*x2+s_*rng.normal(size=n))
rng=np.random.default_rng(99); b1=rng.normal(size=n); b2=rng.normal(size=n)
case("homoskedastic", b1, b2, 1.0+0.75*b1-0.45*b2+rng.normal(size=n))
auto=pd.read_stata("tests/data/auto.dta",convert_categoricals=False)
case("auto.dta price~weight+length", auto["weight"].to_numpy(float),
     auto["length"].to_numpy(float), auto["price"].to_numpy(float))
