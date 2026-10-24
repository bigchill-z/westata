import sys
sys.path.insert(0, "retest/lib")
import numpy as np, pandas as pd
from stata_probe import Stata

def show(st, label, cmd):
    try:
        st.quiet(cmd); print(f"  OK   {label}"); return True
    except Exception as exc:
        print(f"  FAIL {label} -> {type(exc).__name__}: {str(exc).strip()[:160]}"); return False

def rvals(st):
    out = {}
    for k in ("chi2","df","p","estimate","se","t","z","lb","ub","df_r"):
        v = st.scalar(f"r({k})")
        if v is not None: out[k] = v
    return out

with Stata() as st:
    st.run("sysuse auto, clear")
    st.quiet("regress mpg weight"); st.quiet("estimates store L1")
    st.quiet("regress mpg weight foreign"); st.quiet("estimates store B1")
    print("lrtest reversed:")
    show(st, "lrtest L1 B1", "lrtest L1 B1"); print("   ", rvals(st))
    print("  printed:"); print(st.output("lrtest L1 B1"))
    print("lincom only-constant:")
    for e in ("lincom 10", "lincom 2*weight - 3*foreign + 10"):
        if show(st, e, e): print("   ", rvals(st))
    print("test with constant both sides:")
    for e in ("test 2*weight - 3*foreign = -10", "test 2*weight - 3*foreign + 10 = 0",
              "test weight = foreign + 10"):
        if show(st, e, e): print("   ", rvals(st))
    print("tobit e() names / sigma:")
    st.run("sysuse auto, clear")
    st.quiet("tobit mpg weight foreign, ll(17)")
    er = st.stata.get_ereturn()
    for k, v in sorted(er.items()):
        if isinstance(v, float) and ("sigma" in k or "var" in k or "rmse" in k or "rmse" in str(v)):
            print("   ", k, v)
    print("tobit df:", st.scalar("e(df_m)"), st.scalar("e(df_r)"), st.scalar("e(rank)"),
          st.scalar("e(N)"))
    # predict stdf types for truncreg / intreg
    rng = np.random.default_rng(3); n = 300
    x1 = rng.normal(size=n); x2 = rng.normal(size=n)
    lat = 1 + .7*x1 + .4*x2 + rng.normal(scale=.5, size=n)
    df = pd.DataFrame({"y": np.maximum(lat, 1.0), "x1": x1, "x2": x2})
    st.send(df)
    st.quiet("truncreg y x1 x2, ll(1)")
    print("truncreg e(sigma)...", [(k, v) for k, v in st.stata.get_ereturn().items()
                                   if isinstance(v, float) and k in ("e(sigma)","e(var_e)","e(rmse)")])
    for kind in ("stdp","stdf"):
        if show(st, "truncreg "+kind, f"predict double tg_{kind}, {kind}"):
            pass
    st.quiet("predict double tg_stdp, stdp"); st.quiet("predict double tg_stdf, stdf")
    print("   truncreg stdf^2-stdp^2 uniq:", np.unique(np.round(st.var("tg_stdf")**2 - st.var("tg_stdp")**2, 12))[:3])
    # heckman stdf?
    st.run("sysuse auto, clear")
    st.quiet("heckman mpg weight, select(foreign = weight)")
    print("heckman e(sigma-ish):", [(k,v) for k,v in st.stata.get_ereturn().items() if isinstance(v,float) and ("sigma" in k or "rho" in k)])
    for kind in ("xb","stdp","stdf"):
        show(st, "heckman "+kind, f"predict double hk_{kind}, {kind}")
    if True:
        st.quiet("predict double hk_stdp, stdp"); st.quiet("predict double hk_stdf, stdf")
        print("   heckman stdf^2-stdp^2 uniq:", np.unique(np.round(st.var("hk_stdf")**2 - st.var("hk_stdp")**2, 12))[:3])
    # zip predict kinds
    st.run("sysuse auto, clear")
    st.quiet("zip rep78 weight mpg, inflate(weight mpg)")
    for kind in ("n","xb","stdp","stdf","pr","mu"):
        show(st, "zip "+kind, f"predict double zp_{kind}, {kind}")
    st.quiet("predict double zp_n, n")
    print("   zip n first 3:", st.var("zp_n")[:3])
    line = st.output("zip, inflate(weight mpg)") if False else ""
