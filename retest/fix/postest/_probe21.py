import sys
sys.path.insert(0, "retest/lib")
import numpy as np, pandas as pd
from stata_probe import Stata
with Stata() as st:
    rng = np.random.default_rng(7); n = 300
    g = np.tile([0,1,2], n//3+1)[:n].astype(float)
    x1 = rng.normal(size=n)
    st.send(pd.DataFrame({"g": g, "x1": x1}))
    try:
        st.quiet("ologit g x1")
        print("ologit with g in {0,1,2}: OK")
    except Exception as exc:
        print("ologit with g in {0,1,2}:", str(exc).strip()[:160])
    g3 = g + 1
    st.send(pd.DataFrame({"g": g3, "x1": x1}))
    try:
        st.quiet("ologit g x1")
        print("ologit with g in {1,2,3}: OK", st.matrix_names("e(b)"))
    except Exception as exc:
        print("ologit with g in {1,2,3}:", str(exc).strip()[:160])
