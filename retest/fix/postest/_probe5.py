import sys
sys.path.insert(0, "retest/lib")
import numpy as np, pandas as pd
from stata_probe import Stata

def show(st, label, cmd):
    try:
        st.quiet(cmd); print(f"  OK   {label}"); return True
    except Exception as exc:
        print(f"  FAIL {label} -> {str(exc).strip()[:120]}"); return False

with Stata() as st:
    st.run("sysuse auto, clear")
    st.quiet("zip rep78 weight mpg, inflate(weight mpg)")
    for kind in ("n","xb","stdp","stdf","pr","mu","n0","n1"):
        show(st, "zip "+kind, f"predict double zp_{kind}, {kind}")
    st.quiet("predict double zp_n, n")
    print("zip n first 3:", st.var("zp_n")[:3], "mean:", float(np.mean(st.var("zp_n"))))
    print("zip e(b):", st.matrix("e(b)").ravel())
    st.run("sysuse auto, clear")
    st.quiet("zinb rep78 weight mpg, inflate(weight mpg)")
    for kind in ("n","xb","stdp","stdf","pr"):
        show(st, "zinb "+kind, f"predict double zn_{kind}, {kind}")
    st.run("sysuse auto, clear")
    st.quiet("tpoisson rep78 weight mpg, ll(1)")
    for kind in ("n","xb","stdp","stdf"):
        show(st, "tpoisson "+kind, f"predict double tp_{kind}, {kind}")
    st.run("sysuse auto, clear")
    st.quiet("probit foreign weight mpg")
    for kind in ("stdf","stdp","pr"):
        show(st, "probit "+kind, f"predict double pb_{kind}, {kind}")
