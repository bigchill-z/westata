import sys
sys.path.insert(0, "retest/lib")
import numpy as np, pandas as pd
from stata_probe import Stata

def show(st, label, cmd):
    try:
        st.quiet(cmd); print(f"  OK   {label}"); return True
    except Exception as exc:
        print(f"  FAIL {label} -> {str(exc).strip()[:80]}"); return False

rng = np.random.default_rng(11)
n = 400
x1 = rng.normal(size=n); x2 = rng.normal(size=n)
mu = np.exp(0.4 + .6*x1 - .3*x2)
y = rng.poisson(mu)
y1 = np.maximum(y, 1)

with Stata() as st:
    df = pd.DataFrame({"y": y.astype(float), "y1": y1.astype(float), "x1": x1, "x2": x2})
    st.send(df)
    st.quiet("ztp y1 x1 x2")
    b = st.matrix("e(b)").ravel()
    print("ztp e(b):", b)
    for kind in ("n","xb","stdp","stdf","pr","residuals"):
        show(st, "ztp "+kind, f"predict double zt_{kind}, {kind}")
    st.quiet("predict double zt_n, n")
    xb = b[0] + b[1]*np.asarray(x1) + b[2]*np.asarray(x2)
    print("   ztp n mean: %.15g  exp(xb) mean: %.15g   n/exp(xb) mean: %.15g" % (
        float(np.mean(st.var("zt_n"))), float(np.mean(np.exp(xb))),
        float(np.mean(st.var("zt_n")/np.exp(xb)))))
    # zip
    rng2 = np.random.default_rng(7)
    z = rng2.normal(size=n)
    zero_draw = rng2.random(n) < 1/(1+np.exp(-(-0.8 + 1.0*z)))
    yzi = np.where(zero_draw, 0, rng2.poisson(np.exp(0.5 + .5*x1)))
    df2 = pd.DataFrame({"yzi": yzi.astype(float), "x1": x1, "z": z})
    st.send(df2)
    st.quiet("zip yzi x1, inflate(z)")
    bz = st.matrix("e(b)").ravel()
    print("zip e(b):", bz)
    for kind in ("n","xb","stdp","stdf","pr","residuals","n0","n1"):
        show(st, "zip "+kind, f"predict double zp2_{kind}, {kind}")
    for kind in ("n","xb","stdp","pr"):
        try:
            st.run(f"drop zp2_{kind}")
        except Exception:
            pass
    st.quiet("predict double zp2_n, n")
    st.quiet("predict double zp2_pr, pr")
    st.quiet("predict double zp2_xb, xb")
    print("   zip n[0:3]:", st.var("zp2_n")[:3], "pr[0:3]:", st.var("zp2_pr")[:3],
          "xb[0:3]:", st.var("zp2_xb")[:3])
