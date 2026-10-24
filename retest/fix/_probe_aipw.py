import sys; sys.path.insert(0,"retest/lib")
import numpy as np, pandas as pd, westata as ws
from stata_probe import Stata
rng = np.random.default_rng(909); n=600
x1, x2 = rng.normal(size=n), rng.normal(size=n)
t = (rng.uniform(size=n) < 1/(1+np.exp(-(0.3+0.8*x1-0.4*x2)))).astype(float)
y = 2.0 + 1.5*t + 0.9*x1 + 0.6*x2 + rng.normal(size=n)
tr = pd.DataFrame({"y": y, "t": t, "x1": x1, "x2": x2})
with Stata() as st:
    st.send(tr)
    for est, cmd in (("ate","teffects aipw (y x1 x2) (t x1 x2)"), ("atet","teffects aipw (y x1 x2) (t x1 x2), atet")):
        cap = st.estimate(cmd)
        res = ws.AIPW.from_formula("y","t",tr,covariates=["x1","x2"],estimand=est).fit()
        print(est, "type", type(res).__name__, "stata b", cap.b, "names", cap.names)
        print("  attrs:", [a for a in dir(res) if not a.startswith('_')][:25])
