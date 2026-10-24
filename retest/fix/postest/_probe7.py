import sys
sys.path.insert(0, "retest/lib")
import numpy as np, pandas as pd
from stata_probe import Stata
with Stata() as st:
    st.run("sysuse auto, clear")
    st.quiet("tobit mpg weight foreign, ll(17)")
    out = st.output("tobit")
    print(out[:1000])
    print("----  e(b)/e(V) scale ----")
    print(st.matrix("e(b)"), st.matrix_names("e(b)"))
    print("e(V) last diag:", np.diag(st.matrix("e(V)"))[-1])
