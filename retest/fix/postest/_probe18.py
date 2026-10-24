import sys
sys.path.insert(0, "retest/lib")
import numpy as np, pandas as pd
from scipy import stats
from stata_probe import Stata

with Stata() as st:
    st.run("sysuse auto, clear")
    st.quiet("ologit rep78 weight mpg")
    names = st.matrix_names("e(b)"); b = st.matrix("e(b)").ravel()
    print("names", names, "b", b)
    st.quiet("predict double o_pr, pr")
    st.quiet("predict double o_xb, xb")
    pr = st.var("o_pr"); xb = st.var("o_xb")
    k = len(b) - 2
    eta = b[0]*st.var("weight") + b[1]*st.var("mpg")
    cut1 = b[-2]
    print("pr vs logistic(cut1-eta):", float(np.max(np.abs(pr - 1/(1+np.exp(-(cut1-eta)))))))
    print("pr vs logistic(eta-cut1):", float(np.max(np.abs(pr - 1/(1+np.exp(-(eta-cut1)))))))
    print("xb - eta:", float(np.max(np.abs(xb - eta))))
    st.quiet("oprobit rep78 weight mpg")
    b2 = st.matrix("e(b)").ravel(); names2 = st.matrix_names("e(b)")
    st.quiet("predict double p_pr, pr")
    pr2 = st.var("p_pr"); eta2 = b2[0]*st.var("weight") + b2[1]*st.var("mpg")
    print("oprobit names", names2, "pr vs Phi(cut1-eta):", float(np.max(np.abs(pr2 - stats.norm.cdf(b2[-2]-eta2)))))
