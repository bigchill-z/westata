"""Captain spot-check #4: the two highest-severity claims from discrete and systems."""
import sys; sys.path.insert(0, "retest/lib")
import numpy as np, pandas as pd, westata as ws
from stata_probe import Stata

rng = np.random.default_rng(11)
n = 300
x1 = rng.normal(size=n); x2 = rng.normal(size=n)
y = (0.5 + x1 - 0.5*x2 + rng.logistic(size=n) > 0).astype(float)
df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

# (a) nl singular-direction claim
t = np.linspace(0.05, 3.0, 40)
ynl = 2.0 - 1.5*t + 0.8*t**2 + rng.normal(scale=0.05, size=t.size)
dnl = pd.DataFrame({"t": t, "ynl": ynl})

with Stata() as st:
    st.send(dnl)
    cap = st.estimate("nl (ynl = {b0} + {b1}*t + {b2}*t^2)")
    print("stata nl b:", cap.b, "se:", np.sqrt(np.diag(cap.V)))
    r = ws.Nl.from_formula("ynl = {b0} + {b1}*t + {b2}*t**2", dnl, start={"b0": 2.0, "b1": -1.5, "b2": 0.8}).fit()
    print("west nl b:", np.asarray(r.params), "se:", np.asarray(r.bse))

    # (b) tobit sigma SE claim
    st.send(df.assign(yy=rng.normal(size=n)))
    cap2 = st.estimate("tobit yy x1 x2, ll(0)")
    print("stata tobit:", cap2.names, np.sqrt(np.diag(cap2.V)))
    tb = ws.Tobit.from_formula("yy ~ x1 + x2", df.assign(yy=rng.normal(size=n)), lower=0).fit()
    print("west tobit:", list(tb.param_names), np.asarray(tb.bse))
