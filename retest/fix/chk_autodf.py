import sys; sys.path.insert(0,"retest/lib")
import numpy as np, pandas as pd
from stata_probe import Stata
rng = np.random.default_rng(2026)
n_id, T = 30, 6
ids = np.repeat(np.arange(n_id), T); tt = np.tile(np.arange(T), n_id)
x1 = rng.normal(size=n_id*T); x2 = rng.normal(size=n_id*T)
u = np.repeat(rng.normal(size=n_id), T)
y = 1 + 0.8*x1 - 0.5*x2 + u + rng.normal(scale=0.5, size=n_id*T)
df = pd.DataFrame({"id": ids, "t": tt, "x1": x1, "x2": x2, "y": y})
with Stata() as st:
    st.send(df); st.run("xtset id t")
    c = st.estimate("xtreg y x1 x2, fe vce(cluster id)")
    print("stata cluster SE:", np.sqrt(np.diag(c.V)), "df_r:", c.scalars.get("df_r"))
    r = st.estimate("xtreg y x1 x2, fe vce(robust)")
    print("stata robust  SE:", np.sqrt(np.diag(r.V)))
    print("max|V_robust - V_cluster| =", np.max(np.abs(r.V-c.V)))
from linearmodels.panel import PanelOLS as P
pdf = df.set_index(["id","t"])
for auto in (True, False):
    mod = P.from_formula("y ~ 1 + x1 + x2", data=pdf, )
    res = mod.fit(cov_type="clustered", cluster_entity=True, auto_df=auto)
    print(f"linearmodels auto_df={auto} SE:", np.sqrt(np.diag(res.cov)), "df:", res.df_resid)
    res_hc = mod.fit(cov_type="robust", auto_df=auto)
    print(f"linearmodels auto_df={auto} HC-robust SE:", np.sqrt(np.diag(res_hc.cov)))
