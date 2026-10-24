import sys
sys.path.insert(0, "retest/lib")
import numpy as np, pandas as pd
from stata_probe import Stata
with Stata() as st:
    st.run("sysuse auto, clear")
    st.quiet("tobit mpg weight foreign, ll(17)")
    for k in ("e(sigma)","e(var_e)","e(rmse)","e(rmse)","e(df_m)","e(df_r)","e(N)","e(rank)","e(ll)","e(ll_0)"):
        print(k, "=", st.scalar(k))
    er = st.stata.get_ereturn()
    print("keys with sigma:", [k for k in er if "sigma" in k.lower() or "var" in k.lower()])
    st.quiet("predict double q_stdp, stdp")
    st.quiet("predict double q_stdf, stdf")
    print("gap uniq:", np.unique(np.round(st.var("q_stdf")**2-st.var("q_stdp")**2, 10))[:3])
