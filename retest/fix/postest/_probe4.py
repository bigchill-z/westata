import sys
sys.path.insert(0, "retest/lib")
import numpy as np, pandas as pd
from stata_probe import Stata

def show(st, label, cmd):
    try:
        st.quiet(cmd); print(f"  OK   {label}"); return True
    except Exception as exc:
        print(f"  FAIL {label} -> {str(exc).strip()[:120]}"); return False

rng = np.random.default_rng(3); n = 300
x1 = rng.normal(size=n); x2 = rng.normal(size=n)
lat = 1 + .7*x1 + .4*x2 + rng.normal(scale=.5, size=n)
df = pd.DataFrame({"y": np.maximum(lat, 1.0), "x1": x1, "x2": x2})
# interval-censored data for intreg
lo = lat - 0.3 + rng.normal(scale=0.1, size=n); hi = lat + 0.3 + rng.normal(scale=0.1, size=n)
df2 = pd.DataFrame({"lo": np.minimum(lo, hi), "hi": np.maximum(lo, hi), "x1": x1, "x2": x2})

with Stata() as st:
    # ---- truncreg
    st.send(df)
    st.quiet("truncreg y x1 x2, ll(1)")
    er = st.stata.get_ereturn()
    print("truncreg e(sigma)=", er.get("e(sigma)"), " e(var_e)=", er.get("e(var_e)"), " e(rmse)=", er.get("e(rmse)"))
    try: st.run("drop tg_stdp tg_stdf")
    except Exception: pass
    st.quiet("predict double tg_stdp, stdp"); st.quiet("predict double tg_stdf, stdf")
    gap = st.var("tg_stdf")**2 - st.var("tg_stdp")**2
    print("truncreg stdf^2-stdp^2 uniq:", np.unique(np.round(gap, 12))[:3])

    # ---- intreg
    st.send(df2)
    st.quiet("intreg lo hi x1 x2")
    er = st.stata.get_ereturn()
    print("intreg e(sigma)=", er.get("e(sigma)"), "e(lnsigma)=", er.get("e(lnsigma)"))
    for kind in ("stdp","stdf","xb"):
        if show(st, "intreg "+kind, f"predict double ig_{kind}, {kind}"):
            pass
    st.capture_drop = None
    try: st.run("drop ig_stdp ig_stdf")
    except Exception: pass
    st.quiet("predict double ig_stdp, stdp"); st.quiet("predict double ig_stdf, stdf")
    gap = st.var("ig_stdf")**2 - st.var("ig_stdp")**2
    print("intreg stdf^2-stdp^2 uniq:", np.unique(np.round(gap, 12))[:3])

    # ---- heckman
    st.run("sysuse auto, clear")
    st.quiet("heckman mpg weight, select(foreign = weight)")
    er = st.stata.get_ereturn()
    print("heckman e(sigma)=", er.get("e(sigma)"), "e(rho)=", er.get("e(rho)"), "e(rmse)=", er.get("e(rmse)"))
    for kind in ("xb","stdp","stdf"):
        if show(st, "heckman "+kind, f"predict double hk_{kind}, {kind}"):
            pass
    try: st.run("drop hk_stdp hk_stdf")
    except Exception: pass
    st.quiet("predict double hk_stdp, stdp"); st.quiet("predict double hk_stdf, stdf")
    gap = st.var("hk_stdf")**2 - st.var("hk_stdp")**2
    print("heckman stdf^2-stdp^2 uniq:", np.unique(np.round(gap, 12))[:3])

    # ---- zip / zinb predict
    st.run("sysuse auto, clear")
    st.quiet("zip rep78 weight mpg, inflate(weight mpg)")
    for kind in ("n","xb","stdp","stdf","pr","mu"):
        show(st, "zip "+kind, f"predict double zp_{kind}, {kind}")
    try: st.run("drop zp_n")
    except Exception: pass
    st.quiet("predict double zp_n, n")
    print("zip n first 3:", st.var("zp_n")[:3], "mean:", float(np.mean(st.var("zp_n"))))
    st.run("sysuse auto, clear")
    st.quiet("zinb rep78 weight mpg, inflate(weight mpg)")
    for kind in ("n","xb","stdp","stdf","pr"):
        show(st, "zinb "+kind, f"predict double zn_{kind}, {kind}")
    # ---- probit / poisson margins & stdf
    st.run("sysuse auto, clear")
    st.quiet("probit foreign weight mpg")
    for kind in ("stdf","stdp"):
        show(st, "probit "+kind, f"predict double pb_{kind}, {kind}")
