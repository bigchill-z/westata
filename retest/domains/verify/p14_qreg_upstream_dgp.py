"""AUDIT probe 14 — reproduce upstream's EXACT qreg DGP and test tau=.5 robust.

Upstream (limited_quantile/run.py:216-222) uses
    rng_q = default_rng(31); n=400
    qy = 1.0 + 0.8 qx1 - 0.4 qx2 + N(0,1) * (1 + 0.4|qx1|)
and reports qreg vce(robust) at tau=.5 matches to 5.8e-9. We re-run that exact
DGP plus three of our own to see whether tau=.5 is *generally* right.
"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0,"retest/domains/verify"); sys.path.insert(0,"src")
from my_stata import S, rel
import westata as ws

def run(tag, df, taus=(0.25,0.5,0.75)):
    X0 = None
    print("="*72); print(f"### {tag}  n={len(df)}")
    for tau in taus:
        r = ws.QuantileRegression.from_formula("y ~ x1 + x2", df, quantile=tau)
        ri = r.fit(cov_type="iid"); rr = r.fit(cov_type="robust")
        pn = list(rr.param_names); order=[pn.index(k) for k in ("x1","x2","Intercept")]
        with S() as s:
            s.send(df)
            b,Vr,nm = s.est(f"qreg y x1 x2, quantile({tau}) vce(robust)")
            _b,Vi,_ = s.est(f"qreg y x1 x2, quantile({tau})")
        ii=[nm.index(k) for k in ("x1","x2","_cons")]
        Vr=Vr[np.ix_(ii,ii)]; Vi=Vi[np.ix_(ii,ii)]
        sr=np.sqrt(np.diag(Vr)); wr=np.asarray(rr.bse)[order]
        sa=np.sqrt(np.diag(Vi)); wa=np.asarray(ri.bse)[order]
        print(f"  tau={tau}: iid rel={rel(wa,sa):.3e}  robust rel={rel(wr,sr):.3e}"
              f"  ratio={np.round(sr/wr,8)}")

rng_q = np.random.default_rng(31); n_q = 400
qx1 = rng_q.normal(size=n_q); qx2 = rng_q.normal(size=n_q)
qy = 1.0 + 0.8*qx1 - 0.4*qx2 + rng_q.normal(size=n_q)*(1 + 0.4*np.abs(qx1))
run("UPSTREAM exact DGP (seed 31)", pd.DataFrame({"x1":qx1,"x2":qx2,"y":qy}))

rng = np.random.default_rng(4242); n=400
a1=rng.normal(size=n); a2=rng.normal(size=n); sg=np.exp(0.2+0.4*a1)
run("my heteroskedastic DGP (seed 4242)", pd.DataFrame({"x1":a1,"x2":a2,
      "y":1.0+0.75*a1-0.45*a2+sg*rng.normal(size=n)}))

rng = np.random.default_rng(99); b1=rng.normal(size=n); b2=rng.normal(size=n)
run("my homoskedastic DGP (seed 99)", pd.DataFrame({"x1":b1,"x2":b2,
      "y":1.0+0.75*b1-0.45*b2+rng.normal(size=n)}))

auto=pd.read_stata("tests/data/auto.dta",convert_categoricals=False)
run("auto.dta price~weight+length", pd.DataFrame({"x1":auto["weight"].to_numpy(float),
      "x2":auto["length"].to_numpy(float),"y":auto["price"].to_numpy(float)}))
