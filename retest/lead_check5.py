"""Captain spot-check #5: poisson HC0 vs HC1 (silent equality), and e(chi2) under cluster."""
import sys; sys.path.insert(0, "retest/lib")
import numpy as np, pandas as pd, westata as ws
from stata_probe import Stata

rng = np.random.default_rng(5)
n = 400
x1 = rng.normal(size=n)
yc = rng.poisson(np.exp(0.3 + 0.6*x1))
df = pd.DataFrame({"y": yc.astype(float), "x1": x1, "g": np.tile(np.arange(8), n//8)})

with Stata() as st:
    st.send(df)
    base = st.estimate("poisson y x1")
    rob = st.estimate("poisson y x1, vce(robust)")
    m = ws.Poisson.from_formula("y ~ x1", df)
    h0 = m.fit(cov_type="HC0")
    h1 = m.fit(cov_type="HC1")
    print("stata robust SE:", np.sqrt(np.diag(rob.V)))
    print("stata chi2 base/robust:", base.scalars.get("chi2"), rob.scalars.get("chi2"))
    print("west HC0 SE:", np.asarray(h0.bse), "HC1 SE:", np.asarray(h1.bse))
    print("HC0==HC1 exactly? ", np.array_equal(np.asarray(h0.bse), np.asarray(h1.bse)))
    print("west chi2 base:", h0.e.get("chi2") if hasattr(h0,'e') else None, "robust:", h1.e.get("chi2") if hasattr(h1,'e') else None)
    cl = st.estimate("poisson y x1, vce(cluster g)")
    wc = m.fit(cov_type="cluster", cov_kwds={"groups": df["g"]})
    print("stata cluster chi2:", cl.scalars.get("chi2"), "west:", wc.e.get("chi2") if hasattr(wc,'e') else None)
    print("stata cluster SE:", np.sqrt(np.diag(cl.V)), "west:", np.asarray(wc.bse))
