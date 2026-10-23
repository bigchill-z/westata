"""AUDIT probe 4h — structure of Stata's qreg robust e(V).

At tau=.5 psi_i^2 == .25 identically, so the meat is 0.25*X'X on both sides.
We then ask: is V_stata proportional to (X'X)^-1 (=> scalar bread), or is the
bread observation-specific?
"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "retest/domains/verify"); sys.path.insert(0, "src")
from my_stata import S, rel
import westata as ws

rng = np.random.default_rng(4242); n = 400
a1 = rng.normal(size=n); a2 = rng.normal(size=n)
sig = np.exp(0.2 + 0.4*a1)
yh = 1.0 + 0.75*a1 - 0.45*a2 + sig*rng.normal(size=n)
rng = np.random.default_rng(99)
b1 = rng.normal(size=n); b2 = rng.normal(size=n)
yhom = 1.0 + 0.75*b1 - 0.45*b2 + rng.normal(size=n)

for tag,(px,py,pz) in (("hetero",(a1,a2,yh)),("homo",(b1,b2,yhom))):
    df = pd.DataFrame(dict(x1=px,x2=py,y=pz)); X = np.column_stack([px,py,np.ones(n)])
    XtXi = np.linalg.inv(X.T@X)
    with S() as s:
        s.send(df)
        for tau in (0.5,):
            b,V,nm = s.est(f"qreg y x1 x2, quantile({tau}) vce(robust)")
            _b,Vi,_ = s.est(f"qreg y x1 x2, quantile({tau})")
            s.run(f"quietly qreg y x1 x2, quantile({tau})"); sp=s.e("sparsity")
    ii=[nm.index(k) for k in ("x1","x2","_cons")]
    V=V[np.ix_(ii,ii)]; Vi=Vi[np.ix_(ii,ii)]
    print("="*70); print(f"{tag} tau=.5  sparsity={sp}")
    print("  V_rob  =\n", np.array2string(V, precision=8))
    print("  XtXi   =\n", np.array2string(XtXi, precision=8))
    R = V/XtXi
    print("  V/XtXi =\n", np.array2string(R, precision=6))
    print(f"  offdiag spread of V/XtXi = {np.ptp(R[~np.eye(3,dtype=bool)]):.4e}")
    print(f"  V_rob proportional to XtXi? {np.allclose(R, R[0,0]*np.ones((3,3)), rtol=1e-6)}"
          f"   c={R[0,0]:.10f}")
    print(f"  c/(0.25*sparsity^2) = {R[0,0]/(0.25*sp**2):.10f}   sqrt={np.sqrt(R[0,0]/(0.25*sp**2)):.10f}")
    print(f"  iid se={np.round(np.sqrt(np.diag(Vi)),10)}  rob se={np.round(np.sqrt(np.diag(V)),10)}")
    print(f"  n/(n-3)={n/(n-3):.10f}  ((n-3)/n)={((n-3)/n):.10f}")
    q = ws.QuantileRegression.from_formula("y ~ "+"x1 + x2", df, quantile=.5)
    ri=q.fit(cov_type="iid"); rr=q.fit(cov_type="robust")
    order=[list(rr.param_names).index(k) for k in ("x1","x2","Intercept")]
    print(f"  westata rob se={np.round(np.asarray(rr.bse)[order],10)}"
          f"  rel={rel(np.asarray(rr.bse)[order], np.sqrt(np.diag(V))):.3e}")
