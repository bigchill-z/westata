import sys
sys.path.insert(0, "retest/lib")
import numpy as np, pandas as pd
from stata_probe import Stata

def show(st, label, cmd):
    try:
        st.quiet(cmd); print(f"  OK   {label}"); return True
    except Exception as exc:
        print(f"  FAIL {label} -> {str(exc).strip()[:90]}"); return False

with Stata() as st:
    st.run("sysuse auto, clear")
    st.quiet("tobit mpg weight foreign, ll(17)")
    for cmd in ("margins, dydx(weight)", "margins, dydx(weight) predict(e(17,.))",
                "margins, dydx(weight) predict(ystar(17,.))", "margins, dydx(foreign)"):
        if show(st, cmd, cmd):
            t = st.matrix("r(table)")
            jac = st.matrix("r(Jacobian)")
            print("     b=%.15g se=%.15g" % (t[0,0], t[1,0]))
            print("     jacobian:", np.array2string(np.asarray(jac).ravel(), precision=15))
    st.run("sysuse auto, clear")
    st.quiet("ologit rep78 weight mpg")
    for cmd in ("margins, dydx(weight)", "predict double o_pr, pr"):
        show(st, cmd, cmd)
    if True:
        try:
            t = st.matrix("r(table)"); print("     ologit margins b/se", t[0,0], t[1,0])
        except Exception: pass
    st.run("sysuse auto, clear")
    st.quiet("logit foreign weight mpg")
    st.quiet("margins, dydx(foreign)")
    t = st.matrix("r(table)")
    print("logit dydx(foreign):", np.array2string(np.asarray(t[:2]), precision=15))
