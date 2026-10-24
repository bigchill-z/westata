import sys
sys.path.insert(0, "retest/lib")
import numpy as np, pandas as pd
from scipy import stats
from stata_probe import Stata

with Stata() as st:
    st.run("sysuse auto, clear")
    st.quiet("ologit rep78 weight mpg")
    names = st.matrix_names("e(b)"); b = st.matrix("e(b)").ravel()
    st.quiet("predict double o_pr, pr")
    pr = st.var("o_pr")
    eta = b[0]*st.var("weight") + b[1]*st.var("mpg")
    cuts = b[2:]
    # P(y == 1) = Lambda(cut1 - eta)
    cands = {
        "P(y=1)": 1/(1+np.exp(-(cuts[0]-eta))),
        "P(y=J)": 1 - 1/(1+np.exp(-(cuts[-1]-eta))),
        "P(y=2)": 1/(1+np.exp(-(cuts[1]-eta))) - 1/(1+np.exp(-(cuts[0]-eta))),
        "1-P(y=J)": 1/(1+np.exp(-(cuts[-1]-eta))),
    }
    for k, v in cands.items():
        print(f"  pr vs {k}: {float(np.max(np.abs(pr-v))):.3e}")
    print("pr range", float(pr.min()), float(pr.max()), "dist of y:", st.output("tab rep78")[:100])
