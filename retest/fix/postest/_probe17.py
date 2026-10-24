import sys
sys.path.insert(0, "retest/lib")
import numpy as np, pandas as pd
from stata_probe import Stata

def run(st, cmd):
    st.quiet(cmd)
    t = st.matrix("r(table)")
    jac = np.asarray(st.matrix("r(Jacobian)")).ravel()
    print(f"{cmd}\n   b={t[0,0]:.15g} se={t[1,0]:.15g} jac={np.array2string(jac, precision=8)}")

with Stata() as st:
    st.run("sysuse auto, clear")
    st.quiet("tobit mpg weight foreign, ll(17)")
    for cmd in ("margins, dydx(weight)",
                "margins, dydx(weight) predict(xb)",
                "margins, dydx(weight) predict(e(17,.))",
                "margins, dydx(weight) predict(ystar(17,.))",
                "margins, dydx(foreign)",
                "margins, dydx(foreign) predict(e(17,.))",
                "margins",
                "margins, predict(xb)",
                "margins, predict(e(17,.))"):
        try: run(st, cmd)
        except Exception as exc: print(cmd, "->", str(exc).strip()[:80])
