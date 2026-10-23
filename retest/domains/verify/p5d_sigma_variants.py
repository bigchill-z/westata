"""AUDIT probe 5d — etregress e(sigma): which 'generalized residual' definition?"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0,"retest/domains/verify"); sys.path.insert(0,"src")
from my_stata import S, rel
import statsmodels.api as sm

def make(seed, rho_t):
    rng = np.random.default_rng(seed); n = 800
    u1,u2 = rng.normal(size=n), rng.normal(size=n)
    idx = 0.7*u1+0.5*rng.normal(size=n)
    tr = (idx+0.9*u1 > 0).astype(float)
    denom = np.sqrt(1+0.81)
    eps = rho_t*(idx+0.9*u1)/denom + np.sqrt(1-rho_t**2)*u2
    x = rng.normal(size=n); y = 1.0+0.6*x+1.4*tr+eps
    return pd.DataFrame(dict(y=y,t=tr,x1=x,z1=idx,z2=rng.normal(size=n)))

for rho_t in (0.45,-0.55):
    d = make(777, rho_t)
    with S() as s:
        s.send(d)
        b,V,nm = s.est("etregress y x1, treat(t = z1 z2) twostep")
        esig = s.e("sigma")
    Zw = np.column_stack([d.z1,d.z2,np.ones(len(d))]); t=d.t.to_numpy()
    pr = sm.Probit(t,Zw).fit(disp=0)
    g = Zw@pr.params; Phi=pr.model.predict(pr.params)
    phi = np.exp(-0.5*g**2)/np.sqrt(2*np.pi)
    lam = phi/Phi
    # control function actually used by etregress: c = t*lam - (1-t)*lam_mirror
    c_t = t*lam
    c_nt = -(1-t)*(phi/(1-Phi))
    c = c_t + c_nt
    X = np.column_stack([d.x1,t,np.ones(len(d)),c]); yy=d.y.to_numpy()
    coef=np.linalg.lstsq(X,yy,rcond=None)[0]; res=yy-X@coef; bl=coef[-1]
    n=len(d)
    print("="*68); print(f"rho_true={rho_t}  Stata e(sigma)={esig:.12f}  b_ctrl={bl:.10f} Stata lambda={b[nm.index('lambda')]:.10f}")
    for lbl,val in (
      ("sqrt(RSS/n + bl^2 mean(c^2+w c))", np.sqrt(res@res/n + bl**2*np.mean(c**2+g*c))),
      ("sqrt(RSS/n + bl^2 mean(c^2))",    np.sqrt(res@res/n + bl**2*np.mean(c**2))),
      ("sqrt(RSS/n)",                     np.sqrt(res@res/n)),
      ("sqrt(RSS/(n-4))",                 np.sqrt(res@res/(n-4))),
      ("sqrt(RSS/n + bl^2)",              np.sqrt(res@res/n + bl**2)),
    ):
        print(f"   {lbl:34s} = {val:.12f}  rel={rel([val],[esig]):.3e}")
