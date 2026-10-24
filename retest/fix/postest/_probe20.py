import sys
sys.path.insert(0, "retest/lib")
import numpy as np, pandas as pd
from scipy import stats
from stata_probe import Stata

with Stata() as st:
    st.run("sysuse auto, clear")
    st.quiet("oprobit rep78 weight mpg")
    names = st.matrix_names("e(b)"); b = st.matrix("e(b)").ravel()
    print("oprobit names", names, "b", b)
    st.run("drop _all") if False else None
    st.quiet("predict double op_pr, pr")
    pr = st.var("op_pr")
    eta = b[0]*st.var("weight") + b[1]*st.var("mpg")
    for lbl, v in (("Phi(cut1-eta)", stats.norm.cdf(b[2]-eta)),
                   ("Phi(cut1+eta)", stats.norm.cdf(b[2]+eta)),
                   ("Phi(eta-cut1)", stats.norm.cdf(eta-b[2])),
                   ("P(y=J)", 1-stats.norm.cdf(b[-1]-eta))):
        print(f"  pr vs {lbl}: {float(np.max(np.abs(pr-v))):.3e}")
    print("pr range", float(pr.min()), float(pr.max()))
