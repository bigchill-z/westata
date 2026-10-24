import sys
sys.path.insert(0, "retest/lib")
import numpy as np, pandas as pd
from stata_probe import Stata

def qf(diff, V):
    sd = (V+V.T)/2
    ev = np.linalg.eigvalsh(sd)
    if np.all(ev > 0):
        return float(diff @ np.linalg.solve(sd, diff)), "solve", float(ev[0])
    return float(diff @ (np.linalg.pinv(sd) @ diff)), "pinv", float(ev[0])

rng = np.random.default_rng(4242)
n_ent, n_t = 40, 8
ent = np.repeat(np.arange(1, n_ent+1), n_t); tme = np.tile(np.arange(1, n_t+1), n_ent)
alpha = np.repeat(rng.normal(0, 1.5, n_ent), n_t)
x1 = rng.normal(0, 1, n_ent*n_t); x2 = rng.normal(0, 1, n_ent*n_t)
y = 1.0 + 0.8*x1 - 0.5*x2 + alpha + rng.normal(0, .7, n_ent*n_t)
panel = pd.DataFrame({"id": ent, "year": tme, "y": y, "x1": x1, "x2": x2})

with Stata() as st:
    st.send(panel); st.run("xtset id year")
    st.quiet("xtreg y x1 x2, fe"); st.quiet("estimates store FEm")
    b_fe = st.matrix("e(b)").ravel().copy(); V_fe = st.matrix("e(V)").copy()
    n_fe = st.matrix_names("e(b)")
    st.quiet("xtreg y x1 x2, re"); st.quiet("estimates store REm")
    b_re = st.matrix("e(b)").ravel().copy(); V_re = st.matrix("e(V)").copy()
    n_re = st.matrix_names("e(b)")
    print("fe names", n_fe, np.array2string(b_fe, precision=10))
    print("re names", n_re, np.array2string(b_re, precision=10))
    for cmd in ("hausman FEm REm", "hausman FEm REm, constant"):
        st.quiet(cmd)
        print(cmd, "->", st.scalar("r(chi2)"), st.scalar("r(df)"), st.scalar("r(p)"))
    # manual quadratic forms using Stata's own matrices
    common = [n for n in n_fe if n in n_re]
    slopes = [n for n in common if n != "_cons"]
    fe_i = [n_fe.index(n) for n in common]; re_i = [n_re.index(n) for n in common]
    d_all = b_fe[fe_i] - b_re[re_i]
    V_all = V_fe[np.ix_(fe_i, fe_i)] - V_re[np.ix_(re_i, re_i)]
    s = [common.index(n) for n in slopes]
    print("slopes-only manual:", qf(d_all[s], V_all[np.ix_(s, s)]))
    print("all-incl-const manual:", qf(d_all, V_all))
    print("d_all:", np.array2string(d_all, precision=10))
    print("V_all diag:", np.array2string(np.diag(V_all), precision=10))
