"""Captain's own spot-check of two headline claims (independent of domain scripts)."""
import sys; sys.path.insert(0, "retest/lib")
import numpy as np, pandas as pd, westata as ws
from stata_probe import Stata

rng = np.random.default_rng(4242)
n = 500
x1, x2 = rng.normal(size=n), rng.normal(size=n)
lat = 0.8*x1 - 0.5*x2 + rng.logistic(size=n)
y = pd.cut(lat, [-np.inf, -0.5, 0.5, 1.5, np.inf], labels=False) + 1
df = pd.DataFrame({"y": y.astype(float), "x1": x1, "x2": x2})

with Stata() as st:
    st.send(df)
    cap = st.estimate("ologit y x1 x2")
    print("stata names:", cap.names)
    print("stata b:", np.round(cap.b, 8))
    print("stata se:", np.round(np.sqrt(np.diag(cap.V)), 8))
    r = ws.OrderedLogit.from_formula("y ~ x1 + x2", df).fit()
    print("west names:", list(r.param_names))
    print("west b:", np.round(np.asarray(r.params), 8))
    print("west se:", np.round(np.asarray(r.bse), 8))
