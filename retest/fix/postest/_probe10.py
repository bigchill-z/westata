import sys
sys.path.insert(0, "retest/lib")
import numpy as np, pandas as pd
from stata_probe import Stata
from scipy import stats

rng = np.random.default_rng(11); n = 400
x1 = rng.normal(size=n); x2 = rng.normal(size=n)
y = rng.poisson(np.exp(0.4 + .6*x1 - .3*x2))
y1 = np.maximum(y, 1)
rng2 = np.random.default_rng(7)
z = rng2.normal(size=n)
zero = rng2.random(n) < 1/(1+np.exp(-(-0.8 + 1.0*z)))
yzi = np.where(zero, 0, rng2.poisson(np.exp(0.5 + .5*x1)))

with Stata() as st:
    st.send(pd.DataFrame({"y1": y1.astype(float), "x1": x1, "x2": x2}))
    st.quiet("ztp y1 x1 x2")
    b = st.matrix("e(b)").ravel(); X = np.column_stack([np.ones(n), x1, x2])
    mu = np.exp(X @ b)
    st.quiet("predict double zn, n")
    st.quiet("predict double zxb, xb")
    print("ztp: n vs mu max rel %.3e ; n vs mu/(1-exp(-mu)) max rel %.3e" % (
        float(np.max(np.abs(st.var("zn") - mu)/mu)),
        float(np.max(np.abs(st.var("zn") - mu/(1-np.exp(-mu)))/ (mu/(1-np.exp(-mu)))))))
    print("ztp stdp:", st.var("zstdp")[:2] if False else "")

    st.send(pd.DataFrame({"yzi": yzi.astype(float), "x1": x1, "z": z}))
    st.quiet("zip yzi x1, inflate(z)")
    bz = st.matrix("e(b)").ravel(); print("zip b:", bz)
    Z = np.column_stack([np.ones(n), z])
    Xz = np.column_stack([np.ones(n), x1])
    # Stata zip e(b) order: yzi:x1 yzi:_cons inflate:z inflate:_cons? check names
    print("zip names:", st.matrix_names("e(b)"))
    for kind in ("n","pr","xb","stdp"):
        try:
            st.run(f"drop zp_{kind}")
        except Exception: pass
        st.quiet(f"predict double zp_{kind}, {kind}")
    nn = st.var("zp_n"); pr = st.var("zp_pr"); xbv = st.var("zp_xb")
    print("zip n[0:3]", nn[:3], "pr[0:3]", pr[:3], "xb[0:3]", xbv[:3])
    print("zip sum(n) vs sum(y):", float(np.sum(nn)), float(np.sum(yzi)))
