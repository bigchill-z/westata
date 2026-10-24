import sys
sys.path.insert(0, "retest/lib")
import numpy as np, pandas as pd
from stata_probe import Stata

rng = np.random.default_rng(11); n = 400
x1 = rng.normal(size=n); x2 = rng.normal(size=n)
counts = rng.poisson(np.exp(0.4 + 0.6*x1 - 0.3*x2))
rng2 = np.random.default_rng(7); z = rng2.normal(size=n)
zero = rng2.random(n) < 1/(1+np.exp(-(-0.8 + 1.0*z)))
inflated = np.where(zero, 0, rng2.poisson(np.exp(0.5 + 0.5*x1)))

with Stata() as st:
    st.send(pd.DataFrame({"yzi": inflated.astype(float), "x1": x1, "z": z}))
    st.quiet("zip yzi x1, inflate(z)")
    names = st.matrix_names("e(b)"); b = st.matrix("e(b)").ravel(); V = st.matrix("e(V)")
    print("zip names", names, "b", b)
    for kind in ("n","pr","xb","stdp","stdf"):
        try:
            st.run(f"capture drop q_{kind}")
        except Exception: pass
        try:
            st.quiet(f"predict double q_{kind}, {kind}")
            print("  OK", kind, "first", st.var(f"q_{kind}")[:2])
        except Exception as exc:
            print("  FAIL", kind, str(exc).strip()[:60])
    Xc = np.column_stack([x1, np.ones(n)])   # count equation: x1, _cons
    Vc = V[0:2, 0:2]
    expected = np.sqrt(np.einsum("ij,jk,ik->i", Xc, Vc, Xc))
    print("stdp vs sqrt(x_count' V_count x_count) max rel:",
          float(np.max(np.abs(st.var("q_stdp") - expected)/expected)))
    Z = np.column_stack([z, np.ones(n)])
    Vz = V[2:4, 2:4]
    expected_all = np.sqrt(np.einsum("ij,jk,ik->i", np.column_stack([x1,np.ones(n),z,np.ones(n)]), V, np.column_stack([x1,np.ones(n),z,np.ones(n)])))
    print("stdp vs full-4-param x'Vx max rel:", float(np.max(np.abs(st.var("q_stdp") - expected_all)/expected_all)))
    mu = np.exp(b[1] + b[0]*x1); pi = 1/(1+np.exp(-(b[3] + b[2]*z)))
    print("n vs (1-pi)mu:", float(np.max(np.abs(st.var("q_n") - (1-pi)*mu)/np.abs((1-pi)*mu))))
    print("pr vs pi:", float(np.max(np.abs(st.var("q_pr") - pi)/pi)))
# ztp
with Stata() as st:
    st.send(pd.DataFrame({"y1": np.maximum(counts,1).astype(float), "x1": x1, "x2": x2}))
    st.quiet("ztp y1 x1 x2")
    b = st.matrix("e(b)").ravel()
    st.quiet("predict double z_n, n")
    st.quiet("predict double z_xb, xb")
    print("ztp n vs exp(xb):", float(np.max(np.abs(st.var("z_n") - np.exp(st.var("z_xb")))/np.exp(st.var("z_xb")))))
