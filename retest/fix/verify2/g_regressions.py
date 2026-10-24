"""verify2 audit G: focused regression probes on the highest-risk interactions."""

from __future__ import annotations

import sys

sys.path.insert(0, "retest/fix/verify2")
sys.path.insert(0, "src")

import numpy as np
import pandas as pd

import westata as ws
from mybriage import S, fmt, rel, tier
from vlib import cmp_b, cmp_se, report


def main():
    print("=" * 100)
    print("G. HIGH-RISK REGRESSION PROBES")
    print("=" * 100)
    s = S().start()
    out = {}

    # ---------------------------------------------------------------- G1
    print("\n[G1] margins analytic gradient under over() and at() on NONLINEAR models")
    rng = np.random.default_rng(9911)
    n = 500
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    eta = 0.4 + 0.7 * x1 - 0.3 * x2 + rng.logistic(size=n)
    ycat = np.digitize(eta, [-0.8, 0.5, 1.7]).astype(float)
    g = (np.arange(n) % 2) + 1
    fo = pd.DataFrame({"x1": x1, "x2": x2, "ycat": ycat, "grp": g})
    s.send(fo)
    res = ws.OrderedLogit.from_formula("ycat ~ x1 + x2", fo).fit()
    s.est("ologit ycat x1 x2")
    for spec, kwargs, tag in (
        ("margins, dydx(x1) at(x2=1)", {"dydx": "x1", "at": {"x2": 1.0}}, "at(x2=1)"),
        ("margins, dydx(x1) over(grp)", {"dydx": "x1", "over": fo["grp"].to_numpy()},
         "over(grp)"),
    ):
        s.q(spec)
        sb = np.asarray(s.mats("r(b)")).reshape(-1)
        sv = np.sqrt(np.diag(np.asarray(s.mats("r(V)"))))
        w = res.margins(**kwargs)
        if isinstance(w, list):
            wb = np.array([r.estimate for r in w])
            wsv = np.array([r.std_err for r in w])
        else:
            wb, wsv = np.array([w.estimate]), np.array([w.std_err])
        rb, rs = rel(wb, sb), rel(wsv, sv)
        print(f"       ologit dydx(x1) {tag:<12s} Stata b={sb} se={sv}")
        print(f"                              westata b={wb} se={wsv} "
              f"rel b={fmt(rb)}({tier(rb)}) se={fmt(rs)}({tier(rs)})")
        out[f"G1 ologit {tag}"] = max(rb, rs)

    # ---------------------------------------------------------------- G2
    print("\n[G2] tobit df_r rule with noconstant (T2 rule must generalise)")
    rng = np.random.default_rng(2024)
    n = 300
    x1 = rng.normal(size=n)
    lat = 0.5 + 0.9 * x1 + rng.normal(size=n)
    y = np.maximum(lat, 0.0)
    ft = pd.DataFrame({"x1": x1, "y": y})
    s.send(ft)
    E = s.est("tobit y x1, ll(0) noconstant")
    r = ws.Tobit.from_formula("y ~ x1 - 1", ft, lower=0.0).fit()
    ds, dw = E.e["df_r"], r.e["df_r"]
    print(f"       noconstant: e(df_r) Stata={ds} westata={dw} ; "
          f"e(df_m) Stata={E.e['df_m']} westata={r.e['df_m']}")
    out["G2 tobit noconstant df"] = 0.0 if ds == dw else 1.0

    # ---------------------------------------------------------------- G3
    print("\n[G3] rreg: default tolerance AND a tightened one (Stata stops early)")
    rng = np.random.default_rng(67)
    n = 400
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + rng.normal(size=n)
    y[rng.choice(n, 20, replace=False)] += 12.0
    fr = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    s.send(fr)
    for spec, tol in (("rreg y x1 x2", 0.01), ("rreg y x1 x2, tolerance(1e-9)", 1e-9)):
        E = s.est(spec)
        r = ws.RobustRegression.from_formula("y ~ x1 + x2", fr).fit(tolerance=tol)
        pairs = [("x1", "x1"), ("x2", "x2"), ("_cons", "Intercept")]
        b, _ = cmp_b(E, r, pairs)
        se, _ = cmp_se(E, r, pairs)
        report(f"{spec}", b, se, tier(b), tier(se))
        out[f"G3 {spec}"] = max(b, se)

    # ---------------------------------------------------------------- G4
    print("\n[G4] ordered/zip predict surfaces that postest now delegates to")
    s.send(fo)
    s.q("ologit ycat x1 x2")
    s.q("predict double _osp, stdp")
    sp = s.getvar("_osp")
    orr = ws.OrderedLogit.from_formula("ycat ~ x1 + x2", fo).fit()
    try:
        w = np.asarray(orr.predict(kind="stdp"), dtype=float).reshape(-1)
        r = rel(w, sp)
        print(f"       ologit predict,stdp rel={fmt(r)}({tier(r)})")
        out["G4 ologit stdp"] = r
    except Exception as exc:
        print(f"       ologit predict,stdp REFUSED: {type(exc).__name__}: {exc}")
    for kind in ("pr", "stdf"):
        try:
            orr.predict(kind=kind)
            print(f"       ologit predict,{kind}: returned a value (Stata rejects {kind})")
            out[f"G4 ologit {kind}"] = 1.0
        except Exception as exc:
            print(f"       ologit predict,{kind}: {type(exc).__name__} (Stata: r(198)) OK")
            out[f"G4 ologit {kind}"] = 0.0

    # ---------------------------------------------------------------- G5
    print("\n[G5] VAR default cov_params is ML; dfk switch still reproduces Stata")
    rng = np.random.default_rng(7)
    n = 600
    y1 = np.zeros(n)
    y2 = np.zeros(n)
    for t in range(1, n):
        y1[t] = 0.5 * y1[t - 1] + 0.2 * y2[t - 1] + rng.normal()
        y2[t] = -0.3 * y1[t - 1] + 0.4 * y2[t - 1] + rng.normal()
    data = np.column_stack([y1, y2])
    s.send(pd.DataFrame({"tt": np.arange(1, n + 1), "y1": y1, "y2": y2}))
    s.q("tsset tt")
    Ed = s.est("var y1 y2, lags(1) dfk")
    dfk = ws.VAR(data, lags=1, names=["y1", "y2"]).fit(ml=False)
    sd = rel(np.sqrt(np.diag(dfk.stata_cov_params)), Ed.se())
    Sig = rel(dfk.sigma_u.ravel(), np.asarray(s.mats("e(Sigma)")).ravel())
    print(f"       dfk: se rel={fmt(sd)}({tier(sd)}) e(Sigma) rel={fmt(Sig)}({tier(Sig)})")
    out["G5 VAR dfk se"] = sd
    out["G5 VAR dfk Sigma"] = Sig
    Eml = s.est("var y1 y2, lags(1)")
    ml = ws.VAR(data, lags=1, names=["y1", "y2"]).fit()
    sm = rel(np.sqrt(np.diag(ml.stata_cov_params)), Eml.se())
    print(f"       ML default: se rel={fmt(sm)}({tier(sm)}) ; "
          f"coefficients identical between the two fits: "
          f"{np.allclose(dfk.stata_params().to_numpy(), ml.stata_params().to_numpy(), rtol=0)}")
    out["G5 VAR ml se"] = sm

    # ---------------------------------------------------------------- G6
    print("\n[G6] xtpcse single-panel-per-period degenerate case must not silently mis-return")
    rng = np.random.default_rng(5)
    rows = []
    for i in range(12):
        for k in range(5):
            rows.append((i + 1, k + 1, rng.normal(), rng.normal(), rng.normal()))
    bal = pd.DataFrame(rows, columns=["id", "t", "x1", "x2", "y"])
    res = ws.XtPcse.from_formula("y ~ x1 + x2", bal, entity="id").fit()
    s.send(bal)
    s.q("xtset id t")
    E = s.est("xtpcse y x1")
    pairs = [("x1", "x1"), ("_cons", "Intercept")]
    b, _ = cmp_b(E, ws.XtPcse.from_formula("y ~ x1", bal, entity="id").fit(), pairs)
    se, _ = cmp_se(E, ws.XtPcse.from_formula("y ~ x1", bal, entity="id").fit(), pairs)
    report("xtpcse 12x5 balanced, one regressor", b, se, tier(b), tier(se))
    out["G6 xtpcse small balanced"] = max(b, se)

    print("\n---- summary ----")
    for k, v in sorted(out.items(), key=lambda kv: -(kv[1] if kv[1] == kv[1] else -1)):
        print(f"  {k:<40s} {fmt(v):<10s} {tier(v)}")


if __name__ == "__main__":
    main()
