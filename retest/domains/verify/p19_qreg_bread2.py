"""AUDIT probe 19 — qreg: does the meat swap ALONE fix it everywhere?

On upstream's DGP and on a homoskedastic DGP the swap reproduces Stata to 1e-8.
On an independent heteroskedastic DGP it does NOT (1.5e-1). If the meat is
correct there, the remaining gap must be the BREAD (westata's per-observation
density f_i). Test the bread directly.
"""
import sys
import numpy as np, pandas as pd
def sym_sqrt(A):
    w,Q = np.linalg.eigh((A+A.T)/2)
    return Q @ np.diag(np.sqrt(np.clip(w,0,None))) @ Q.T
sys.path.insert(0,"retest/domains/verify"); sys.path.insert(0,"src")
from my_stata import S, rel
import westata as ws

def solve_J(V,B):
    A12=sym_sqrt(V); Ai=np.linalg.inv(A12)
    return Ai @ sym_sqrt(A12@B@A12) @ Ai

def case(tag, df, tau):
    q=ws.QuantileRegression.from_formula("y ~ x1 + x2", df, quantile=tau)
    rr=q.fit(cov_type="robust"); sp=q._sparsity; X=q._X
    beta=np.asarray(rr.params); resid=q._y-X@beta
    h=sp["bandwidth"]; f_i=(2*h)/(X@(sp["beta_upper"]-sp["beta_lower"]))
    J_w=(X.T*f_i)@X
    M=tau*(1-tau)*(X.T@X)
    with S() as s:
        s.send(df)
        b,V,nm=s.est(f"qreg y x1 x2, quantile({tau}) vce(robust)")
        s.run(f"quietly qreg y x1 x2, quantile({tau})"); sps=s.e("sparsity")
    ii=[nm.index(k) for k in ("x1","x2","_cons")]
    Vs=V[np.ix_(ii,ii)]
    J_s=solve_J(Vs,M)
    pn=list(rr.param_names); order=[pn.index(k) for k in ("x1","x2","Intercept")]
    Jw=J_w[np.ix_(order,order)]
    print(f"  {tag} tau={tau}")
    print(f"    Stata sparsity={sps:.10f}  westata sparsity={sp['sparsity']:.10f}"
          f"  rel={rel([sp['sparsity']],[sps]):.3e}")
    print(f"    Stata-implied J  /  (X'X):\n{np.array2string(J_s/np.diag(np.diag(X.T@X))*0+J_s, precision=3)}")
    print(f"    westata J (perp order):\n{np.array2string(Jw, precision=3)}")
    print(f"    rel(J_stata, J_westata) = {rel(J_s.reshape(-1), Jw.reshape(-1)):.4e}")
    # row-normalised comparison (scale-free)
    print(f"    row-sum J_stata   = {np.round(J_s.sum(1),4)}")
    print(f"    row-sum J_westata = {np.round(Jw.sum(1),4)}")
    # V with Stata's bread and the constant meat -> should equal Vs exactly
    print(f"    check J_s M J_s == Vs : {rel((J_s@M@J_s).reshape(-1), Vs.reshape(-1)):.3e}")

n=400
rng=np.random.default_rng(4242); a1=rng.normal(size=n); a2=rng.normal(size=n)
sg=np.exp(0.2+0.4*a1)
case("hetero", pd.DataFrame(dict(x1=a1,x2=a2,y=1.0+0.75*a1-0.45*a2+sg*rng.normal(size=n))), 0.25)
rng=np.random.default_rng(99); b1=rng.normal(size=n); b2=rng.normal(size=n)
case("homo", pd.DataFrame(dict(x1=b1,x2=b2,y=1.0+0.75*b1-0.45*b2+rng.normal(size=n))), 0.25)
