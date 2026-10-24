import sys
sys.path.insert(0, "retest/lib")
import numpy as np, pandas as pd
from stata_probe import Stata

rng = np.random.default_rng(11); n = 400
x1 = rng.normal(size=n); x2 = rng.normal(size=n)
y = rng.poisson(np.exp(0.4 + .6*x1 - .3*x2))
y1 = np.maximum(y, 1)

with Stata() as st:
    st.send(pd.DataFrame({"y1": y1.astype(float), "x1": x1, "x2": x2}))
    st.quiet("ztp y1 x1 x2")
    names = st.matrix_names("e(b)"); b = st.matrix("e(b)").ravel()
    print("names", names, "b", b)
    X = np.column_stack([x1, x2, np.ones(n)])
    mu = np.exp(X @ b)
    st.quiet("predict double zn, n"); st.quiet("predict double zx, xb")
    print("ztp n vs mu rel max %.3e" % float(np.max(np.abs(st.var("zn")-mu)/mu)))
    print("ztp n vs mu/(1-exp(-mu)) rel max %.3e" % float(np.max(np.abs(st.var("zn")-mu/(1-np.exp(-mu)))/(mu/(1-np.exp(-mu))))))
    print("ztp n vs mu*Pr? ")
    print("first: n", st.var("zn")[:3], "mu", mu[:3], "xb", st.var("zx")[:3])

    # zip
    rng2 = np.random.default_rng(7); z = rng2.normal(size=n)
    zero = rng2.random(n) < 1/(1+np.exp(-(-0.8 + 1.0*z)))
    yzi = np.where(zero, 0, rng2.poisson(np.exp(0.5 + .5*x1)))
    st.send(pd.DataFrame({"yzi": yzi.astype(float), "x1": x1, "z": z}))
    st.quiet("zip yzi x1, inflate(z)")
    print("zip names", st.matrix_names("e(b)"), "b", st.matrix("e(b)").ravel())
    for k in ("n","pr","xb","stdp"):
        try: st.run(f"drop zp_{k}")
        except Exception: pass
        st.quiet(f"predict double zp_{k}, {k}")
    bz = st.matrix("e(b)").ravel()
    mu = np.exp(bz[1] + bz[0]*x1)
    lam = 1/(1+np.exp(-(bz[3] + bz[2]*z)))
    print("zip n vs (1-lam)mu rel max %.3e" % float(np.max(np.abs(st.var("zp_n")-(1-lam)*mu)/((1-lam)*mu))))
    print("zip pr vs lam rel max %.3e" % float(np.max(np.abs(st.var("zp_pr")-lam)/lam)))
