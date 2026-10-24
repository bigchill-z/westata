import sys
sys.path.insert(0, "retest/lib")
import numpy as np, pandas as pd
from stata_probe import Stata
with Stata() as st:
    st.run("sysuse auto, clear")
    st.quiet("ologit rep78 weight mpg")
    print("e(marginsok) =", st.macro("e(marginsok)"))
    print("e(predict) =", st.macro("e(predict)"))
    for cmd in ("margins, dydx(weight)", "margins, dydx(weight) predict(xb)",
                "margins, dydx(weight) predict(pr)", "margins, dydx(weight) predict(pr outcome(2))"):
        try:
            st.quiet(cmd)
            t = st.matrix("r(table)")
            print(f"  {cmd}: b={t[0,0]:.15g} se={t[1,0]:.15g}")
        except Exception as exc:
            print(f"  {cmd} -> {str(exc).strip()[:100]}")
