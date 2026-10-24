import sys
sys.path.insert(0, "retest/lib")
import numpy as np, pandas as pd
from stata_probe import Stata
import westata as ws

rng = np.random.default_rng(11); n = 400
x1 = rng.normal(size=n); x2 = rng.normal(size=n)
counts = rng.poisson(np.exp(0.4 + 0.6*x1 - 0.3*x2))
rng2 = np.random.default_rng(7); z = rng2.normal(size=n)
zero = rng2.random(n) < 1/(1+np.exp(-(-0.8 + 1.0*z)))
inflated = np.where(zero, 0, rng2.poisson(np.exp(0.5 + 0.5*x1)))
df = pd.DataFrame({"yzi": inflated.astype(float), "x1": x1, "z": z})

with Stata() as st:
    st.send(df)
    st.quiet("zip yzi x1, inflate(z)")
    b = st.matrix("e(b)").ravel(); V = st.matrix("e(V)")
    st.run("capture drop ws_stp"); st.quiet("predict double ws_stp, stdp")
    stata_stdp = st.var("ws_stp")

w = ws.ZeroInflatedPoisson.from_formula("yzi ~ x1", df, inflate="z").fit()
ws_stdp = np.asarray(w.predict(kind="stdp"), float)
print("ws vs stata stdp rel:", float(np.max(np.abs(ws_stdp - stata_stdp)/stata_stdp)))
# check reference with proper column order [x1, 1]
Xc = np.column_stack([x1, np.ones(n)])
ref = np.sqrt(np.einsum("ij,jk,ik->i", Xc, V[0:2,0:2], Xc))
print("stata stdp vs sqrt(x_c'V_cc x_c) [x1,_cons order]:", float(np.max(np.abs(stata_stdp-ref)/ref)))
print("ws b:", np.asarray(w.params, float), "stata b:", b)
print("ws V diag:", np.diag(w.cov_params()), "stata V diag:", np.diag(V))
