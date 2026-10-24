import sys
sys.path.insert(0, "retest/lib")
import numpy as np, pandas as pd
from stata_probe import Stata

def show(st, label, cmd):
    try:
        st.quiet(cmd); print(f"  OK   {label}"); return True
    except Exception as exc:
        print(f"  FAIL {label} -> {str(exc).strip()[:60]}"); return False

KINDS = ("xb","stdp","stdf","pr","n","residuals","fitted","e(17,.)","ystar(17,.)")
with Stata() as st:
    for label, setup in (
        ("logit", "logit foreign weight mpg"),
        ("probit", "probit foreign weight mpg"),
        ("poisson", "poisson rep78 weight mpg"),
        ("nbreg", "nbreg rep78 weight mpg"),
        ("ologit", "ologit rep78 weight"),
        ("tobit", "tobit mpg weight foreign, ll(17)"),
    ):
        st.run("sysuse auto, clear")
        st.quiet(setup)
        for kind in KINDS:
            tag = kind.replace("(","_").replace(")","").replace(",","").replace(".","")
            show(st, f"{label} predict,{kind}", f"predict double t13_{tag}, {kind}")
