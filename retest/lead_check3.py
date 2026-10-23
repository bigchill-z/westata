"""Captain spot-check #3: does xtreg,fe vce(robust) really equal vce(cluster id) in Stata?"""
import sys; sys.path.insert(0, "retest/lib")
import numpy as np, pandas as pd
from stata_probe import Stata
rng = np.random.default_rng(7)
n2 = 180
ids = np.repeat(np.arange(30), 6); tt = np.tile(np.arange(6), 30)
xx = rng.normal(size=n2)
yy = 1.0 + 0.8*xx + np.repeat(rng.normal(size=30), 6) + rng.normal(size=n2)
p2 = pd.DataFrame({"id": ids, "t": tt, "x1": xx, "yy": yy})
with Stata() as st:
    st.send(p2); st.run("xtset id t")
    cr = st.estimate("xtreg yy x1, fe vce(robust)")
    cc = st.estimate("xtreg yy x1, fe vce(cluster id)")
    print("robust  V:\n", cr.V)
    print("cluster V:\n", cc.V)
    print("max abs diff:", np.max(np.abs(cr.V-cc.V)))
    cl2 = st.estimate("regress yy x1 i.id, vce(robust)")
    print("areg-style robust SE (regress i.id):", np.sqrt(np.diag(cl2.V))[:2], cl2.names[:3])
