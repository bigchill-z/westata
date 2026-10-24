import sys
sys.path.insert(0, "retest/lib")
import numpy as np, pandas as pd
from stata_probe import Stata

def show(st, label, cmd):
    try:
        st.quiet(cmd); print(f"  OK   {label}"); return True
    except Exception as exc:
        print(f"  FAIL {label} -> {str(exc).strip()[:80]}"); return False

with Stata() as st:
    st.run("sysuse auto, clear")
    st.quiet("tpoisson rep78 weight mpg, ll(0.5)")
    for kind in ("n","xb","stdp","stdf","pr","residuals"):
        show(st, "tpoisson "+kind, f"predict double tp_{kind}, {kind}")
    st.run("sysuse auto, clear")
    st.quiet("poisson rep78 weight mpg")
    for kind in ("n","pr","residuals","stdf"):
        show(st, "poisson "+kind, f"predict double po_{kind}, {kind}")
    st.run("sysuse auto, clear")
    st.quiet("nbreg rep78 weight mpg")
    for kind in ("n","pr","stdf"):
        show(st, "nbreg "+kind, f"predict double nb_{kind}, {kind}")
