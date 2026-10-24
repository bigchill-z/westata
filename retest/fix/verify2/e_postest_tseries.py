"""verify2 audit E: postest (M1-M7) and tseries (V1-V6) -- independent re-measurement."""

from __future__ import annotations

import sys

sys.path.insert(0, "retest/fix/verify2")
sys.path.insert(0, "src")

import numpy as np
import pandas as pd

import westata as ws
from mybriage import S, fmt, rel, tier
from vlib import cmp_b, cmp_mat, cmp_se, report

AUTO = "D:/Program Files/StataNow19/auto.dta"


# ------------------------------------------------------------------ postest
def m1():
    print("=" * 100)
    print("E1. POSTEST M1 -- margins AME SE (analytic gradient under at()/over())")
    print("=" * 100)
    s = S().start()
    s.run("sysuse auto, clear")
    ds = pd.read_stata(AUTO)
    ds["foreign"] = (ds["foreign"].astype(str) == "Foreign").astype(int)
    out = {}

    cases = [
        ("logit foreign weight mpg", ws.Logit, "foreign ~ weight + mpg", {}),
        ("probit foreign weight mpg", ws.Probit, "foreign ~ weight + mpg", {}),
        ("poisson rep78 mpg weight", ws.Poisson, "rep78 ~ mpg + weight", {}),
    ]
    for cmd, cls, form, _ in cases:
        E = s.est(cmd)
        res = cls.from_formula(form, ds).fit()
        # Jacobian Stata itself reports for margins, dydx(weight)
        s.q("margins, dydx(weight)")
        sb = float(np.asarray(s.mats("r(b)"))[0, 0])
        sse = float(np.sqrt(np.asarray(s.mats("r(V)"))[0, 0]))
        jac = s.mats("r(Jacobian)")
        w = res.margins(dydx="weight")
        rb = abs(w.estimate - sb) / abs(sb)
        rs = abs(w.std_err - sse) / abs(sse)
        print(f"  {cmd:<28s} Stata b={sb:+.10f} se={sse:.10f} | westata "
              f"b={w.estimate:+.10f} se={w.std_err:.10f} rel b={fmt(rb)}({tier(rb)}) "
              f"se={fmt(rs)}({tier(rs)})")
        out[f"M1 {cmd}"] = max(rb, rs)
        # at(mpg=20)
        s.q("margins, dydx(weight) at(mpg=20)")
        sb2 = float(np.asarray(s.mats("r(b)"))[0, 0])
        ss2 = float(np.sqrt(np.asarray(s.mats("r(V)"))[0, 0]))
        w2 = res.margins(dydx="weight", at={"mpg": 20.0})
        rb2 = abs(w2.estimate - sb2) / abs(sb2)
        rs2 = abs(w2.std_err - ss2) / abs(ss2)
        print(f"  {'  + at(mpg=20)':<28s} Stata b={sb2:+.10f} se={ss2:.10f} | rel "
              f"b={fmt(rb2)}({tier(rb2)}) se={fmt(rs2)}({tier(rs2)})")
        out[f"M1 at {cmd}"] = max(rb2, rs2)

    # over() on a linear model
    cmd = "regress mpg weight i.foreign"
    E = s.est(cmd)
    lin = ws.OLS.from_formula("mpg ~ weight + C(foreign)", ds).fit()
    s.q("margins, over(foreign)")
    sbv = np.asarray(s.mats("r(b)")).reshape(-1)
    svv = np.sqrt(np.diag(np.asarray(s.mats("r(V)"))))
    ov = lin.margins(over=ds["foreign"].to_numpy())
    wbv = np.array([o.estimate for o in ov])
    wsv = np.array([o.std_err for o in ov])
    rb, rs = rel(wbv, sbv), rel(wsv, svv)
    print(f"  margins, over(foreign)      Stata b={sbv} se={svv}")
    print(f"                              westata b={wbv} se={wsv} rel b={fmt(rb)} "
          f"se={fmt(rs)} ({tier(max(rb, rs))})")
    out["M1 over()"] = max(rb, rs)

    # discrete change for a factor
    s.q("margins, dydx(foreign)")
    # Stata reports the base level (0) first; the discrete change is the other one.
    _b3 = np.asarray(s.mats("r(b)")).reshape(-1)
    _V3 = np.asarray(s.mats("r(V)"))
    _k = int(np.argmax(np.abs(_b3)))
    sb3 = float(_b3[_k])
    ss3 = float(np.sqrt(_V3[_k, _k]))
    w3 = lin.margins(dydx="C(foreign)[T.1]", factor_vars=["C(foreign)[T.1]"])
    rb3 = abs(w3.estimate - sb3) / abs(sb3)
    rs3 = abs(w3.std_err - ss3) / abs(ss3)
    print(f"  margins, dydx(foreign)      Stata b={sb3:+.10f} se={ss3:.10f} rel "
          f"b={fmt(rb3)}({tier(rb3)}) se={fmt(rs3)}({tier(rs3)})")
    out["M1 dydx(foreign)"] = max(rb3, rs3)

    print("\n---- E1 summary ----")
    for k, v in sorted(out.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<34s} {fmt(v):<10s} {tier(v)}")
    return out


def m2_m7():
    print("\n" + "=" * 100)
    print("E2. POSTEST M2/M3/M4/M7 -- tobit stdf, predict dispatch, lincom, tobit margins")
    print("=" * 100)
    s = S().start()
    s.run("sysuse auto, clear")
    ds = pd.read_stata(AUTO)
    ds["foreign"] = (ds["foreign"].astype(str) == "Foreign").astype(int)
    out = {}

    # M2 / M7
    E = s.est("tobit mpg weight foreign, ll(17)")
    tb = ws.Tobit.from_formula("mpg ~ weight + C(foreign)", ds, lower=17.0).fit()
    s.q("predict double _sp, stdp")
    s.q("predict double _sf, stdf")
    sp, sf = s.getvar("_sp"), s.getvar("_sf")
    wsp = np.asarray(tb.predict(kind="stdp"), dtype=float).reshape(-1)
    wsf = np.asarray(tb.predict(kind="stdf"), dtype=float).reshape(-1)
    print(f"  tobit stdp rel={fmt(rel(wsp, sp))}({tier(rel(wsp, sp))}) "
          f"stdf rel={fmt(rel(wsf, sf))}({tier(rel(wsf, sf))})")
    out["M2 tobit stdp"] = rel(wsp, sp)
    out["M2 tobit stdf"] = rel(wsf, sf)
    # the structural identity Stata satisfies
    sig = float(tb.e["sigma"])
    print(f"  stdf^2 - stdp^2 == e(sigma)^2: max rel="
          f"{fmt(rel(wsf**2 - wsp**2, np.full(wsf.size, sig**2)))} ; "
          f"Stata side max rel="
          f"{fmt(rel(sf**2 - sp**2, np.full(sf.size, sig**2)))}")

    # M7: default margins kind
    s.q("margins, dydx(weight)")
    db = float(np.asarray(s.mats("r(b)"))[0, 0])
    ds_ = float(np.sqrt(np.asarray(s.mats("r(V)")))[0, 0])
    w = tb.margins(dydx="weight")
    rb = abs(w.estimate - db) / abs(db)
    rs = abs(w.std_err - ds_) / abs(ds_)
    print(f"  M7 tobit margins DEFAULT: Stata b={db:+.10f} se={ds_:.10f} westata "
          f"b={w.estimate:+.10f} se={w.std_err:.10f} rel b={fmt(rb)}({tier(rb)}) "
          f"se={fmt(rs)}({tier(rs)}) pred={w.prediction}")
    out["M7 tobit margins default"] = max(rb, rs)
    s.q("margins, predict(e(17,.))")
    eb = float(np.asarray(s.mats("r(b)"))[0, 0])
    es = float(np.sqrt(np.asarray(s.mats("r(V)")))[0, 0])
    we = tb.margins(predict_kind="e")
    rb2 = abs(we.estimate - eb) / abs(eb)
    rs2 = abs(we.std_err - es) / abs(es)
    print(f"  margins, predict(e(17,.)) : Stata b={eb:+.10f} se={es:.10f} rel "
          f"b={fmt(rb2)}({tier(rb2)}) se={fmt(rs2)}({tier(rs2)})")
    out["M7 tobit margins e()"] = max(rb2, rs2)

    # M4 lincom with a bare constant
    E = s.est("regress mpg weight foreign")
    # Numeric foreign, so the coefficient carries Stata's own name in the design
    # (westata's lincom cannot address a factor by its ``C(x)[T.1]`` label --
    # pre-existing, unrelated to M4).
    ols = ws.OLS.from_formula("mpg ~ weight + foreign", ds).fit()
    # Stata's `foreign` is westata's `C(foreign)[T.1]`; map the name for the call
    # only (the Stata side keeps Stata's spelling).
    name_map: dict[str, str] = {}
    for expr in ("2*weight - 3*foreign + 10", "2*weight - 3*foreign", "10",
                 "2*weight - 3*foreign + 1e3"):
        s.q("lincom " + expr)
        sb = s.sc("r(estimate)")
        ss = s.sc("r(se)")
        wexpr = expr
        for k, v in name_map.items():
            wexpr = wexpr.replace(k, v)
        w = ols.lincom(wexpr)
        rb = abs(w.estimate - sb) / max(abs(sb), 1e-12)
        rs = abs(w.std_err - ss) / max(abs(ss), 1e-12) if ss else 0.0
        print(f"  lincom {expr:<28s} Stata {sb!r:<24s} se={ss!r:<22s} rel b="
              f"{fmt(rb)}({tier(rb)}) se={fmt(rs)}({tier(rs)})")
        out[f"M4 {expr}"] = max(rb, rs)
    # test with a constant on the left
    s.q("test 2*weight - 3*foreign + 10 = 0")
    sft = s.sc("r(chi2)")
    if sft is None:
        s.q("test 2*weight - 3*foreign + 10 = 0")
        sft = s.sc("r(F)")
    wtest = ols.test("2*weight - 3*foreign + 10 = 0")
    stat = getattr(wtest, "statistic", None)
    print(f"  test 2*w - 3*f + 10 = 0 : Stata={sft} westata={stat} rel="
          f"{fmt(abs(stat - sft) / sft)}({tier(abs(stat - sft) / sft)})")
    out["M4 test constant"] = abs(stat - sft) / sft

    # M3 zip / ztp predict through the postestimation wrapper
    rng = np.random.default_rng(4242)
    n = 600
    x1, z = rng.normal(size=n), rng.normal(size=n)
    pi = 1 / (1 + np.exp(-(0.3 + 0.6 * z)))
    mu = np.exp(0.5 + 0.7 * x1)
    yz = np.where(rng.random(n) < pi, 0.0, rng.poisson(mu).astype(float))
    fz = pd.DataFrame({"x1": x1, "z": z, "y": yz})
    s.send(fz)
    E = s.est("zip y x1, inflate(z)")
    zr = ws.ZeroInflatedPoisson.from_formula("y ~ x1", fz, inflate="z").fit()
    for kind in ("n", "pr", "xb"):
        s.q(f"predict double _z_{kind}, {kind}")
        sv = s.getvar(f"_z_{kind}")
        wv = np.asarray(zr.predict(kind=kind), dtype=float).reshape(-1)
        r = rel(wv, sv)
        print(f"  M3 zip predict,{kind:<4s} rel={fmt(r)}({tier(r)})")
        out[f"M3 zip {kind}"] = r
    # rejection of an unsupported kind must list the available ones
    try:
        zr.predict(kind="mu")
        print("  M3 zip predict,mu : NO ERROR (unexpected)")
        out["M3 zip reject"] = 1.0
    except Exception as exc:
        has_list = "n" in str(exc) and "pr" in str(exc)
        print(f"  M3 zip predict,mu : {type(exc).__name__} lists_available={has_list}")
        out["M3 zip reject"] = 0.0 if has_list else 0.5

    print("\n---- E2 summary ----")
    for k, v in sorted(out.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<34s} {fmt(v):<10s} {tier(v)}")
    return out


def tseries():
    print("\n" + "=" * 100)
    print("E3. TSERIES V1-V6")
    print("=" * 100)
    s = S().start()
    out = {}

    rng = np.random.default_rng(20261024)
    n = 300
    y1 = np.zeros(n)
    y2 = np.zeros(n)
    for t in range(1, n):
        y1[t] = 0.5 * y1[t - 1] + 0.2 * y2[t - 1] + rng.normal()
        y2[t] = -0.3 * y1[t - 1] + 0.4 * y2[t - 1] + rng.normal()
    vf = pd.DataFrame({"tt": np.arange(1, n + 1), "y1": y1, "y2": y2})
    s.send(vf)
    s.q("tsset tt")
    data = np.column_stack([y1, y2])

    print("\n[E3a] V1 VAR SE / e(Sigma) / e(V) + V2 vargranger chi2")
    # NOTE Stata's ``lags(2)`` means lag 2 ONLY; ``lags(1/2)`` is the two-lag VAR.
    # westata's ``lags=2`` means 1..2, so it must be compared against ``lags(1/2)``.
    for lags, stata_spec in ((1, "lags(1)"), (2, "lags(1/2)")):
        E = s.est(f"var y1 y2, {stata_spec}")
        res = ws.VAR(data, lags=lags, names=["y1", "y2"]).fit()
        b = rel(res.stata_params().to_numpy(), E.b)
        se = rel(res.bse.to_numpy(), E.se())
        Sig = rel(res.sigma_u.ravel(), np.asarray(s.mats("e(Sigma)")).ravel())
        V = rel(np.asarray(res.stata_cov_params).ravel(), E.V.ravel())
        print(f"  VAR({lags})  b={fmt(b)}({tier(b)}) se={fmt(se)}({tier(se)}) "
              f"e(Sigma)={fmt(Sig)}({tier(Sig)}) e(V)={fmt(V)}({tier(V)})")
        out[f"V1 VAR{lags} se"] = se
        out[f"V1 VAR{lags} Sigma"] = Sig
        out[f"V1 VAR{lags} V"] = V

    E = s.est("var y1 y2, lags(1)")
    # `vargranger` prints only 2 decimals and stores nothing in r(), so the
    # exact Stata value comes from the equivalent Wald test on the VAR.
    s.q("test [y1]L.y2")
    s_12 = s.sc("r(chi2)")
    s.q("test [y2]L.y1")
    s_21 = s.sc("r(chi2)")
    fitted = ws.VAR(data, lags=1, names=["y1", "y2"]).fit()
    g12 = fitted.granger_causality("y1", ["y2"])
    g21 = fitted.granger_causality("y2", ["y1"])
    for tag, g, sval in (("y1<-y2", g12, s_12), ("y2<-y1", g21, s_21)):
        st = float(getattr(g, "test_statistic"))
        r = abs(st - sval) / abs(sval)
        print(f"  vargranger {tag}: method={g.method} df={g.df} "
              f"Stata(equiv test)={sval:.12f} westata={st:.12f} rel={fmt(r)}({tier(r)}) "
              f"p_rel={fmt(abs(float(g.pvalue)-0.0))}")
        out[f"V2 granger {tag}"] = r
    txt = s.text("vargranger")
    print("  Stata vargranger prints:")
    for ln in txt.splitlines():
        if "y1" in ln or "y2" in ln:
            print("    " + ln)

    print("\n[E3b] V3 dfuller printed critical values + statistic unchanged")
    rw = np.cumsum(rng.normal(size=n))
    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = 0.6 * ar[t - 1] + rng.normal()
    uf = pd.DataFrame({"tt": np.arange(1, n + 1), "y_rw": rw, "y_ar": ar})
    s.send(uf)
    s.q("tsset tt")
    for cmd, reg, lags, series in (
        ("dfuller y_rw, lags(2)", "c", 2, rw),
        ("dfuller y_rw, lags(2) trend", "ct", 2, rw),
        ("dfuller y_rw, lags(0) noconstant", "n", 0, rw),
        ("dfuller y_ar, lags(4)", "c", 4, ar),
        ("pperron y_rw", "c", None, rw),
        ("pperron y_rw, trend", "ct", None, rw),
    ):
        s.q(cmd)
        st = s.sc("r(Zt)")
        cvm = s.mats("r(cvalues)")
        # dfuller: r(cv_1)/r(cv_5)/r(cv_10);  pperron: r(cvalues) 2x4
        # (row 2 = Z(t); columns 1..3 are 1%/5%/10%).
        if cvm is not None:
            cvs = [float(cvm[1, k]) for k in (1, 2, 3)]
        else:
            cvs = [s.sc("r(cv_%s)" % k) for k in ("1", "5", "10")]
        test = "adf" if cmd.startswith("dfuller") else "pp"
        u = ws.unitroot(series, test=test, regression=reg,
                        maxlag=lags if lags is not None else None,
                        autolag=None if lags is not None else "AIC")
        r = None
        cw = None
        if getattr(u, "critical", None):
            cw = np.array([u.critical[k] for k in ("1%", "5%", "10%")], dtype=float)
        print(f"  {cmd:<38s} stat={st} westata_stat={getattr(u, 'statistic', None)} "
              f"cv_stata={cvs} cv_westata={cw}")
        if cw is not None and all(c is not None for c in cvs):
            r = rel(cw[:3], np.array(cvs, dtype=float))
            print(f"       cv rel={fmt(r)} ({tier(r)})")
            out[f"V3 {cmd}"] = r

    print("\n[E3c] V4 vecrank critical values (Johansen table)")
    rk = np.cumsum(rng.normal(size=n))
    co = rk + rng.normal(scale=0.2, size=n)
    vd = np.column_stack([rk, co])
    core = ws.VECM(vd, coint_rank=1).johansen(k_ar_diff=1, det_order=0)
    print(f"  westata 5% trace cv = {core.trace_critical[:, 1]}")
    print(f"  westata 5% maxeig cv = {core.maxeig_critical[:, 1]}")
    print(f"  90% column is nan: {np.isnan(core.trace_critical[:, 0]).all()}")
    out["V4 cv_trace_r1"] = (
        0.0 if abs(core.trace_critical[1, 1] - 3.76) < 1e-12 else 1.0
    )
    out["V4 cv_maxeig_r0"] = (
        0.0 if abs(core.maxeig_critical[0, 1] - 14.07) < 1e-12 else 1.0
    )

    print("\n[E3d] V5 arima e(N)")
    a1 = np.zeros(n)
    for t in range(1, n):
        a1[t] = a1[t - 1] + rng.normal()
    s.send(pd.DataFrame({"tt": np.arange(1, n + 1), "y_rw": a1}))
    s.q("tsset tt")
    for cmd, order in (("arima y_rw, arima(0,1,1)", (0, 1, 1)),
                       ("arima y_rw, arima(1,1,0)", (1, 1, 0))):
        E = s.est(cmd)
        m = ws.ARIMA(a1, order=order).fit()
        r = abs(float(m.e["N"]) - E.e["N"]) / E.e["N"]
        print(f"  {cmd:<30s} e(N) Stata={E.e['N']} westata={m.e['N']} "
              f"nobs={m.nobs} rel={fmt(r)}({tier(r)})")
        out[f"V5 {cmd}"] = r

    print("\n[E3e] V6 arch (cov_type OPG / presample)")
    rng2 = np.random.default_rng(777)
    m = 600
    e = np.zeros(m)
    h = np.full(m, 1.0)
    for t in range(1, m):
        h[t] = 0.05 + 0.1 * e[t - 1] ** 2 + 0.85 * h[t - 1]
        e[t] = np.sqrt(h[t]) * rng2.normal()
    gd = pd.DataFrame({"tt": np.arange(1, m + 1), "yg": e})
    s.send(gd)
    s.q("tsset tt")
    E = s.est("arch yg, arch(1) garch(1)")
    g = ws.GARCH(e, p=1, q=1).fit()
    sv = np.array([E.get(nm) for nm in E.names])
    print(f"  arch e(b) Stata={dict(zip(E.names, E.b))}")
    print(f"           westata={dict(zip(g.param_names, g.params.to_numpy()))}")
    print(f"  e(vcetype) Stata={E.e.get('vcetype')!r} westata={g.e.get('vcetype')!r}")
    print(f"  SE Stata={E.se()} westata={g.bse.to_numpy()}")
    out["V6 vcetype"] = 0.0 if g.e.get("vcetype") == "OPG" else 1.0
    print("\n---- E3 summary ----")
    for k, v in sorted(out.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<34s} {fmt(v):<10s} {tier(v)}")
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "m1"):
        m1()
    if which in ("all", "m2"):
        m2_m7()
    if which in ("all", "ts"):
        tseries()
