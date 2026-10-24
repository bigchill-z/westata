import sys
sys.path.insert(0, "retest/lib")
import numpy as np, pandas as pd
from stata_probe import Stata

def show(st, label, cmd):
    try:
        st.quiet(cmd)
        vals = {k: st.scalar(f"r({k})") for k in ("estimate","se","z","t","p","lb","ub","df","chi2","F","df_r")}
        print(f"  OK   {label}: " + str({k: v for k, v in vals.items() if v is not None}))
        return True
    except Exception as exc:
        print(f"  FAIL {label} -> {str(exc).strip()[:70]}")
        return False

with Stata() as st:
    st.run("sysuse auto, clear")
    st.quiet("regress mpg weight foreign")
    for e in ("lincom 2*weight - 3*foreign + 10",
              "lincom 10",
              "lincom -10",
              "lincom 2*weight - 3*foreign + 10, level(90)",
              "lincom 2*weight - 3*foreign + 1e3",
              "lincom (2*weight) + 10",
              "lincom 2 * weight + 10"):
        show(st, e, e)
    print(" output of constant lincom:")
    print(st.output("lincom 2*weight - 3*foreign + 10"))
    print(" test cases:")
    for e in ("test 2*weight - 3*foreign + 10 = 0",
              "test 2*weight - 3*foreign = -10",
              "test 2*weight - 3*foreign",
              "test 10",
              "test weight = 10",
              "test weight + 10 = 0"):
        show(st, e, e)
    print(" logit lincom constant:")
    st.run("sysuse auto, clear")
    st.quiet("logit foreign weight mpg")
    for e in ("lincom 2*weight - 3*mpg + 10", "lincom 10"):
        show(st, e, e)
    print(st.output("lincom 2*weight - 3*mpg + 10"))
