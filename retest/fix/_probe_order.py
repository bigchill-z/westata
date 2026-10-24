import sys; sys.path.insert(0,"retest/lib")
import numpy as np, pandas as pd, westata as ws
from stata_probe import Stata
rng = np.random.default_rng(20261024); n, g = 140, 14
gid = np.repeat(np.arange(g), n//g)
x1, z1, z2 = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
u = rng.normal(size=n)
e1 = 0.7*z1 + 0.4*z2 + 0.6*u + rng.normal(size=n)
yiv = 1.0 + 0.5*x1 + 1.2*e1 + u
df = pd.DataFrame({"yiv": yiv, "x1": x1, "e1": e1, "z1": z1, "z2": z2, "gid": gid})
with Stata() as st:
    st.send(df)
    cap = st.estimate("ivregress 2sls yiv x1 (e1 = z1 z2), vce(robust)")
    print("stata names:", cap.names); print("stata se:", np.sqrt(np.diag(cap.V)))
m = ws.IV2SLS.from_formula("yiv ~ x1 + [e1 ~ z1 + z2]", df)
for ct in ("nonrobust","HC0","HC1"):
    r = m.fit(cov_type=ct)
    print("west", ct, list(r.param_names), np.asarray(r.bse))
