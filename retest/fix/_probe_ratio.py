import sys; sys.path.insert(0,"retest/lib")
import numpy as np, pandas as pd, westata as ws
from stata_probe import Stata

# poisson robust: check ratio
rng = np.random.default_rng(5); n=400
x1 = rng.normal(size=n); yc = rng.poisson(np.exp(0.3+0.6*x1))
po = pd.DataFrame({"y": yc.astype(float), "x1": x1, "g": np.tile(np.arange(8), n//8)})
with Stata() as st:
    st.send(po)
    rob = st.estimate("poisson y x1, vce(robust)")
    clu = st.estimate("poisson y x1, vce(cluster g)")
    m = ws.Poisson.from_formula("y ~ x1", po)
    for kw in ({"cov_type":"robust"}, {"cov_type":"cluster","cov_kwds":{"groups":po["g"]}},
               {"cov_type":"HC0"}, {"cov_type":"HC1"}):
        r = m.fit(**kw)
        print(kw.get("cov_type"), "west SE", np.asarray(r.bse))
    print("stata robust SE", np.sqrt(np.diag(rob.V)))
    print("stata cluster SE", np.sqrt(np.diag(clu.V)))
    print("ratio robust stata/west-robust:", np.sqrt(np.diag(rob.V))/np.asarray(m.fit(cov_type="robust").bse))
    print("N/(N-1) =", n/(n-1))
