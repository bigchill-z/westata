import sys
sys.path.insert(0, "retest/lib")
import numpy as np, pandas as pd
from stata_probe import Stata

def show(st, label, cmd):
    try:
        st.quiet(cmd); print(f"  OK   {label}"); return True
    except Exception as exc:
        print(f"  FAIL {label} -> {str(exc).strip()[:90]}"); return False

rng2 = np.random.default_rng(7); n = 400
x1 = rng2.normal(size=n); z = rng2.normal(size=n)
zero = rng2.random(n) < 1/(1+np.exp(-(-0.8 + 1.0*z)))
yzi = np.where(zero, 0, rng2.poisson(np.exp(0.5 + .5*x1)))
with Stata() as st:
    st.send(pd.DataFrame({"yzi": yzi.astype(float), "x1": x1, "z": z}))
    st.quiet("zip yzi x1, inflate(z)")
    for kind in ("residuals","csn","mu","n0","n1"):
        show(st, "zip "+kind, f"predict double zq_{kind}, {kind}")
    st.run("sysuse auto, clear")
    st.quiet("nbreg rep78 weight mpg")
    for kind in ("n","xb","stdp","stdf","pr","residuals"):
        show(st, "nbreg "+kind, f"predict double nbk_{kind}, {kind}")
    st.run("sysuse auto, clear")
    st.quiet("poisson rep78 weight mpg")
    for kind in ("pr","residuals","stdf"):
        show(st, "poisson "+kind, f"predict double pok_{kind}, {kind}")
    # Stata error text for an unregistered model type
    try:
        st.quiet("predict double zzz, whatever")
    except Exception as exc:
        print("error text sample:", str(exc).strip()[:120])
