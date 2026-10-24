"""Live StataNow 19.5 probe for the six postest defects (scratch)."""
import sys, warnings
sys.path.insert(0, "retest/lib")
import numpy as np, pandas as pd
from stata_probe import Stata

auto = pd.read_stata("tests/data/auto.dta", convert_categoricals=False)

def show(st, label, cmd):
    try:
        st.quiet(cmd)
        print(f"  OK   {label}: {cmd}")
        return True, ""
    except Exception as exc:
        print(f"  FAIL {label}: {cmd} -> {type(exc).__name__}: {str(exc).strip()[:200]}")
        return False, str(exc)

with Stata() as st:
    st.run("sysuse auto, clear")
    # ---- M1: margins AME
    st.quiet("logit foreign weight mpg")
    st.quiet("margins, dydx(weight)")
    t = st.matrix("r(table)")
    print("M1 logit AME dydx(weight): b=%.15g se=%.15g" % (t[0,0], t[1,0]))
    print("M1 r(Jacobian):", np.array2string(st.matrix("r(Jacobian)").ravel(), precision=15))
    st.quiet("margins, dydx(weight) at(mpg=20)")
    t = st.matrix("r(table)")
    print("M1 logit AME at(mpg=20): b=%.15g se=%.15g" % (t[0,0], t[1,0]))
    # probit too
    st.quiet("probit foreign weight mpg")
    st.quiet("margins, dydx(weight)")
    t = st.matrix("r(table)")
    print("M1 probit AME dydx(weight): b=%.15g se=%.15g" % (t[0,0], t[1,0]))
    # poisson AME
    st.run("sysuse auto, clear")
    st.quiet("poisson rep78 mpg weight")
    st.quiet("margins, dydx(weight)")
    t = st.matrix("r(table)")
    print("M1 poisson AME dydx(weight): b=%.15g se=%.15g" % (t[0,0], t[1,0]))
    # regress over / dydx factor
    st.run("sysuse auto, clear")
    st.quiet("regress mpg weight i.foreign")
    st.quiet("margins, over(foreign)")
    t = st.matrix("r(table)")
    print("M1 regress over(foreign) se:", np.array2string(t[1,:2], precision=15))
    st.quiet("margins, dydx(foreign)")
    t = st.matrix("r(table)")
    print("M1 regress dydx(foreign) b/se:", np.array2string(t[0,:2], precision=15), np.array2string(t[1,:2], precision=15))

    # ---- M2: tobit stdf
    st.run("sysuse auto, clear")
    st.quiet("tobit mpg weight foreign, ll(17)")
    sigma2 = st.scalar("e(var_e)") if st.scalar("e(var_e)") else None
    print("M2 tobit e(sigma)/e(var_e):", st.scalar("e(sigma)"), sigma2)
    st.run("predict double p_stdp, stdp")
    st.run("predict double p_stdf, stdf")
    st.run("predict double p_xb, xb")
    sp, sf = st.var("p_stdp"), st.var("p_stdf")
    print("M2 tobit stdf^2-stdp^2:", np.unique(np.round(sf**2 - sp**2, 10))[:3])
    print("M2 tobit stdf sqrt(stdp^2+sigma^2) max diff:",
          float(np.max(np.abs(sf - np.sqrt(sp**2 + sf**2 - sp**2)))))
    # stdf for truncreg / intreg / heckman?
    rng = np.random.default_rng(3)
    n = 300
    x1 = rng.normal(size=n); x2 = rng.normal(size=n)
    lat = 1 + .7*x1 + .4*x2 + rng.normal(scale=.5, size=n)
    df = pd.DataFrame({"y": np.maximum(lat, 1.0), "x1": x1, "x2": x2})
    st.send(df)
    for cmd in ("truncreg y x1 x2, ll(1)", "intreg y y, nolog"):
        ok, err = show(st, "estimate", cmd)
        if ok:
            for kind in ("stdp", "stdf", "xb"):
                show(st, kind, f"predict double q_{kind}, {kind}")

    # ---- M3: zip predict
    st.run("sysuse auto, clear")
    st.run("set obs 400")
    ok, _ = show(st, "zip", "zip rep78 weight mpg")
    if ok:
        for kind in ("n", "xb", "stdp", "stdf", "pr"):
            show(st, kind, f"predict double z_{kind}, {kind}")

    # ---- M4: lincom constant
    st.run("sysuse auto, clear")
    st.quiet("regress mpg weight foreign")
    for expr in ("2*weight - 3*foreign + 10", "2*weight - 3*foreign", "weight + 1"):
        ok, err = show(st, "lincom", f"lincom {expr}")
        if ok:
            print("      est=%.15g se=%.15g df=%.6g" % (st.scalar("r(estimate)"),
                                                       st.scalar("r(se)"), st.scalar("r(df)")))
    # test with a constant
    for expr in ("2*weight - 3*foreign + 10 = 0", "weight = 3000", "weight + 10"):
        ok, err = show(st, "test", f"test {expr}")
        if ok:
            print("      chi2=%.15g F=%.15g df=%.6g p=%.15g rdf=%.6g" % (
                st.scalar("r(chi2)") or float('nan'), st.scalar("r(F)") or float('nan'),
                st.scalar("r(df)") or float('nan'), st.scalar("r(p)") or float('nan'),
                st.scalar("r(df_r)") or float('nan')))

    # ---- M5: lrtest df=0 and non-nested, k_u < k_r
    st.run("sysuse auto, clear")
    st.quiet("regress mpg weight"); st.quiet("estimates store L1")
    st.quiet("regress mpg weight"); st.quiet("estimates store L2")
    show(st, "df=0", "lrtest L1 L2")
    st.quiet("regress mpg weight foreign"); st.quiet("estimates store B1")
    show(st, "nested", "lrtest B1 L1")
    print("      chi2=%.15g df=%.6g p=%.15g" % (st.scalar("r(chi2)") or float('nan'),
                                                st.scalar("r(df)") or float('nan'),
                                                st.scalar("r(p)") or float('nan')))
    show(st, "reversed (k_u < k_r)", "lrtest L1 B1")
    st.quiet("regress mpg foreign length"); st.quiet("estimates store N2")
    show(st, "nonnested", "lrtest N2 L1")
    print("      chi2=%.15g df=%.6g p=%.15g" % (st.scalar("r(chi2)") or float('nan'),
                                                st.scalar("r(df)") or float('nan'),
                                                st.scalar("r(p)") or float('nan')))
    st.quiet("regress mpg foreign"); st.quiet("estimates store N3")
    show(st, "nonnested equal k", "lrtest N3 L1")

    # ---- M6: hausman constant
    rng = np.random.default_rng(4242)
    n_ent, n_t = 40, 8
    ent = np.repeat(np.arange(1, n_ent+1), n_t); tme = np.tile(np.arange(1, n_t+1), n_ent)
    alpha = np.repeat(rng.normal(0, 1.5, n_ent), n_t)
    x1 = rng.normal(0, 1, n_ent*n_t); x2 = rng.normal(0, 1, n_ent*n_t)
    y = 1.0 + 0.8*x1 - 0.5*x2 + alpha + rng.normal(0, .7, n_ent*n_t)
    panel = pd.DataFrame({"id": ent, "year": tme, "y": y, "x1": x1, "x2": x2})
    st.send(panel); st.run("xtset id year")
    st.quiet("xtreg y x1 x2, fe"); st.quiet("estimates store FEm")
    st.quiet("xtreg y x1 x2, re"); st.quiet("estimates store REm")
    for cmd in ("hausman FEm REm", "hausman FEm REm, constant"):
        ok, err = show(st, "hausman", cmd)
        if ok:
            print("      chi2=%.15g df=%.6g p=%.15g" % (st.scalar("r(chi2)") or float('nan'),
                                                        st.scalar("r(df)") or float('nan'),
                                                        st.scalar("r(p)") or float('nan')))
    print("  FE b:", st.matrix("e(b)") if False else "see below")
