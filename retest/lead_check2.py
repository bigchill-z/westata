import sys; sys.path.insert(0, "retest/lib")
import numpy as np, pandas as pd, westata as ws
from stata_probe import Stata

rng = np.random.default_rng(20261024)
n, G = 140, 14
gid = np.repeat(np.arange(G), n // G)
x1 = rng.normal(size=n); z1 = rng.normal(size=n); z2 = rng.normal(size=n)
u = rng.normal(size=n)
e1 = 0.7*z1 + 0.4*z2 + 0.6*u + rng.normal(size=n)
yiv = 1.0 + 0.5*x1 + 1.2*e1 + u
df = pd.DataFrame({"yiv": yiv, "x1": x1, "e1": e1, "z1": z1, "z2": z2, "gid": gid})

with Stata() as st:
    st.send(df)
    cap = st.estimate("ivregress 2sls yiv x1 (e1 = z1 z2), vce(robust)")
    print("stata names", cap.names, "SE", np.sqrt(np.diag(cap.V)))
    m = ws.IV2SLS.from_formula("yiv ~ x1 + [e1 ~ z1 + z2]", df)
    for ct in ("HC0", "HC1"):
        r = m.fit(cov_type=ct)
        print("west", ct, list(r.param_names), np.asarray(r.bse))

    # panel
    n2 = 180
    ids = np.repeat(np.arange(30), 6); tt = np.tile(np.arange(6), 30)
    xx = rng.normal(size=n2)
    yy = 1.0 + 0.8*xx + np.repeat(rng.normal(size=30), 6) + rng.normal(size=n2)
    p2 = pd.DataFrame({"id": ids, "t": tt, "x1": xx, "yy": yy})
    st.send(p2); st.run("xtset id t")
    cr = st.estimate("xtreg yy x1, fe vce(robust)")
    cc = st.estimate("xtreg yy x1, fe vce(cluster id)")
    print("xtreg fe robust :", cr.names, np.sqrt(np.diag(cr.V)))
    print("xtreg fe cluster:", cc.names, np.sqrt(np.diag(cc.V)))
    p = ws.PanelOLS.from_formula("yy ~ x1", p2, entity="id", time="t")
    rh = p.fit(cov_type="HC1"); rc = p.fit(cov_type="cluster", cov_kwds={"groups": p2["id"]})
    print("west HC1    :", list(rh.param_names), np.asarray(rh.bse))
    print("west cluster:", list(rc.param_names), np.asarray(rc.bse))
