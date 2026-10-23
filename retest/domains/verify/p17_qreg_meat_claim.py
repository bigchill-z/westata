"""AUDIT probe 17 — test upstream's specific qreg claim:
"swap the meat to tau(1-tau)X'X and tau=.25 reproduces Stata to 9.0e-9".
"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0,"retest/domains/verify"); sys.path.insert(0,"src")
from my_stata import S, rel
import westata as ws

def test(tag, df, tau):
    q = ws.QuantileRegression.from_formula("y ~ x1 + x2", df, quantile=tau)
    ri = q.fit(cov_type="iid"); rr = q.fit(cov_type="robust")
    sp = q._sparsity; X = q._X
    pn = list(rr.param_names)
    beta = np.asarray(rr.params)
    resid = q._y - X @ beta
    psi = tau - (resid < 0).astype(float)
    h = sp["bandwidth"]; diff = sp["beta_upper"] - sp["beta_lower"]
    f_i = (2*h) / (X @ diff)
    J = (X.T * f_i) @ X
    M_psi = (X.T * psi**2) @ X
    M_const = tau*(1-tau) * (X.T @ X)
    Ji = np.linalg.inv(J)
    with S() as s:
        s.send(df)
        b, Vr, nm = s.est(f"qreg y x1 x2, quantile({tau}) vce(robust)")
    ii = [nm.index(k) for k in ("x1","x2","_cons")]
    se_s = np.sqrt(np.diag(Vr[np.ix_(ii,ii)]))
    order = [pn.index(k) for k in ("x1","x2","Intercept")]
    Vw = np.asarray(rr.cov_params())[np.ix_(order, order)]
    V_psi = Ji @ M_psi @ Ji
    V_const = Ji @ M_const @ Ji
    print(f"  {tag} tau={tau}")
    print(f"    Stata se            = {np.round(se_s,10)}")
    print(f"    westata (psi^2 meat)= {np.round(np.sqrt(np.diag(Vw)),10)}"
          f"  rel={rel(np.sqrt(np.diag(Vw)), se_s):.3e}")
    print(f"    swapped (tau(1-t)X'X meat, same bread) = {np.round(np.sqrt(np.diag(V_const)),10)}"
          f"  rel={rel(np.sqrt(np.diag(V_const)), se_s):.3e}")
    print(f"    psi^2 meat with the SAME bread          = {np.round(np.sqrt(np.diag(V_psi)),10)}"
          f"  rel={rel(np.sqrt(np.diag(V_psi)), se_s):.3e}")

rng = np.random.default_rng(31); n=400
x1=rng.normal(size=n); x2=rng.normal(size=n)
y=1.0+0.8*x1-0.4*x2+rng.normal(size=n)*(1+0.4*np.abs(x1))
d=pd.DataFrame(dict(x1=x1,x2=x2,y=y))
print("="*78); print("### UPSTREAM exact DGP (seed 31, n=400)")
for t in (0.25,0.5,0.75): test("upstream", d, t)

rng=np.random.default_rng(4242); a1=rng.normal(size=n); a2=rng.normal(size=n)
sg=np.exp(0.2+0.4*a1); yh=1.0+0.75*a1-0.45*a2+sg*rng.normal(size=n)
print("="*78); print("### my heteroskedastic DGP")
for t in (0.25,0.5): test("mine", pd.DataFrame(dict(x1=a1,x2=a2,y=yh)), t)
