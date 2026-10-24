"""verify2 audit D: systems group (S1-S7 + T8) -- independent re-measurement."""

from __future__ import annotations

import sys

sys.path.insert(0, "retest/fix/verify2")
sys.path.insert(0, "src")

import numpy as np
import pandas as pd

import westata as ws
from mybriage import S, fmt, rel, tier
from vlib import cmp_b, cmp_mat, cmp_se, report


def quad(seed=4242, n=200):
    """The conditioned design the nl defect was reported on."""
    rng = np.random.default_rng(seed)
    w = rng.uniform(2000, 5000, n)
    y = 50.0 - 0.014 * w + 1.3e-6 * w**2 + rng.normal(scale=2.0, size=n)
    return pd.DataFrame({"weight": w, "y": y})


def expo(seed=11, n=120):
    rng = np.random.default_rng(seed)
    mpg = rng.uniform(10, 40, n)
    price = 9000.0 * np.exp(-0.09 * mpg) * np.exp(rng.normal(scale=0.05, size=n))
    return pd.DataFrame({"mpg": mpg, "price": price})


def treat_frame(seed=211, n=1500):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    p = 1 / (1 + np.exp(-(0.2 + 0.7 * x1 - 0.4 * x2)))
    t = (rng.random(n) < p).astype(float)
    y = 0.8 + 1.2 * x1 - 0.6 * x2 + 1.4 * t + rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "t": t, "y": y})


def did_frame(seed=99, units=60, periods=8):
    rng = np.random.default_rng(seed)
    rows = []
    treated_units = set(range(units // 2))
    for i in range(units):
        eff = rng.normal(scale=0.5)
        for k in range(periods):
            post = 1.0 if k >= periods // 2 else 0.0
            tr = 1.0 if i in treated_units else 0.0
            x1 = rng.normal()
            y = (eff + 0.4 * k + 1.6 * tr * post + 0.5 * x1
                 + rng.normal(scale=0.6))
            rows.append((i + 1, k + 1, tr, tr * post, y, x1))
    return pd.DataFrame(rows, columns=["unit", "period", "treated", "tp", "y", "x1"])


def frontier_frame(seed=99, n=400):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    v = rng.normal(size=n)
    u = -np.abs(rng.normal(scale=0.7, size=n))
    y = 1.0 + 0.8 * x1 - 0.4 * x2 + v + u
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def main():
    print("=" * 100)
    print("D. SYSTEMS")
    print("=" * 100)
    s = S().start()
    print(f"stata {s.version}")
    out = {}

    # ---------------------------------------------------------------- S1
    print("\n[D1] S1 nl covariance on a conditioned design + rank-deficient refusal")
    au = pd.read_stata("D:/Program Files/StataNow19/auto.dta")
    s.send(au)
    E = s.est("nl (mpg = {b0} + {b1}*weight + {b2}*weight^2)")
    print(f"       Stata b = {E.b} ; SE = {E.se()}")
    nl = ws.Nl.from_formula("mpg = {b0} + {b1}*weight + {b2}*weight**2", au).fit()
    b = rel(nl.params.to_numpy(), E.b)
    se = rel(nl.bse.to_numpy(), E.se())
    report("nl auto.dta quadratic (BRIEF's own check)", b, se, tier(b), tier(se))
    out["nl_se"] = se
    out["nl_b"] = b

    print("       [FINDING] same model, same identifiability, different UNITS")
    rng0 = np.random.default_rng(4242)
    w0 = rng0.uniform(2000, 5000, 200)
    y0 = 50 - 0.014 * w0 + 1.3e-6 * w0**2 + rng0.normal(scale=2.0, size=200)
    s.send(pd.DataFrame({"weight": w0, "y": y0}))
    Eq = s.est("nl (y = {b0} + {b1}*weight + {b2}*weight^2)")
    print(f"       Stata estimates it: b={Eq.b} SE={Eq.se()}")
    for sc in (1.0, 1e-1, 1e-2, 1e-3):
        qq = pd.DataFrame({"weight": w0 * sc, "y": y0})
        J = np.column_stack([np.ones(200), w0 * sc, (w0 * sc) ** 2])
        cond = np.linalg.cond(J)
        try:
            m = ws.Nl.from_formula("y = {b0} + {b1}*weight + {b2}*weight**2", qq).fit()
            back = np.array([1.0, sc, sc**2])
            bb, ss = m.params.to_numpy() * back, m.bse.to_numpy() * back
            rb = np.max(np.abs(bb - Eq.b) / np.abs(Eq.b))
            rs = np.max(np.abs(ss - Eq.se()) / np.abs(Eq.se()))
            print(f"         weight*{sc:<7g} cond(J)={cond:9.3e}  OK   b rel={fmt(rb)} "
                  f"se rel={fmt(rs)} ({tier(rs)})")
            out[f"nl_units_{sc:g}"] = rs
        except Exception as exc:
            print(f"         weight*{sc:<7g} cond(J)={cond:9.3e}  "
                  f"REFUSED {type(exc).__name__}")
            out[f"nl_units_{sc:g}"] = float("inf")

    print("\n       rank-deficient design must raise, not invent an SE")
    rng = np.random.default_rng(5)
    n = 120
    z = rng.normal(size=n)
    rd = pd.DataFrame({"z": z, "x": z.copy(),
                      "y": 1.0 + 2.0 * z + rng.normal(size=n)})
    try:
        bad = ws.Nl.from_formula("y = {a}*x + {c}*z", rd).fit()
        print(f"       NO ERROR -> se={bad.bse.to_numpy()} (unexpected)")
        out["nl_rank_guard"] = 1.0
    except Exception as exc:
        print(f"       {type(exc).__name__}: {str(exc)[:150]}")
        out["nl_rank_guard"] = 0.0
    # collinear + still estimable model must NOT be refused
    ok = ws.Nl.from_formula("y = {a}*x", rd).fit()
    print(f"       estimable subset still fits: bse={ok.bse.to_numpy()}")

    # ---------------------------------------------------------------- S2
    print("\n[D2] S2 nl start values / convergence on an exponential model")
    ex = expo()
    s.send(ex)
    E = s.est("nl (price = {b0}*exp({b1}*mpg))")
    nl2 = ws.Nl.from_formula("price = {b0}*exp({b1}*mpg)", ex).fit()
    names = list(nl2.param_names)
    sb = np.array([E.get(nm.lstrip("/")) for nm in names])
    b = rel(nl2.params.to_numpy(), sb)
    print(f"       Stata b={dict(zip(E.names, E.b))}")
    print(f"       westata b={dict(zip(names, nl2.params.to_numpy()))}")
    rss_s = E.e.get("rss")
    rss_w = nl2.e.get("rss") if "rss" in getattr(nl2, "e", {}) else None
    print(f"       param rel={fmt(b)}({tier(b)})  converged={nl2.e.get('converged')}  "
          f"Stata rss={rss_s} westata rss={rss_w}")
    out["nl_start"] = b

    # ---------------------------------------------------------------- S3
    print("\n[D3] S3 teffects aipw: ATET (repaired) AND ATE (must not regress)")
    tf = treat_frame()
    s.send(tf)
    for cmd, est, tag in (
        ("teffects aipw (y x1 x2) (t x1 x2), atet", "atet", "ATET"),
        ("teffects aipw (y x1 x2) (t x1 x2)", "ate", "ATE"),
    ):
        E = s.est(cmd)
        # Stata stores the effect as the first coefficient, ``r1vs0.t``.
        k = E.idx("r1vs0.t")
        sb, ss = float(E.b[k]), float(np.sqrt(E.V[k, k]))
        r = ws.AIPW.from_formula("y", "t", tf, covariates=["x1", "x2"],
                                estimand=est).fit()
        rb = abs(float(r.estimate) - sb) / abs(sb)
        rs = abs(float(r.std_err) - ss) / abs(ss)
        print(f"       {tag:<5s} Stata={sb:.12f} (se {ss:.12f}) | westata="
              f"{float(r.estimate):.12f} (se {float(r.std_err):.12f}) "
              f"rel b={fmt(rb)}({tier(rb)}) se={fmt(rs)}({tier(rs)})")
        out[f"aipw_{tag}"] = max(rb, rs)

    # ---------------------------------------------------------------- S5
    print("\n[D4] S5 didregress (y) vs (y x1)")
    dd = did_frame()
    s.send(dd)
    s.q("xtset unit period")
    for spec, cov, tag in (("(y)", None, "bare (y)"), ("(y x1)", ["x1"], "(y x1)")):
        E = s.est(f"didregress {spec} (tp), group(unit) time(period)")
        k = E.idx("r1vs0.tp")
        sb, ss = float(E.b[k]), float(np.sqrt(E.V[k, k]))
        r = ws.Didregress.from_formula("y" if cov is None else "y x1", "tp",
                                       dd, group="unit", time="period",
                                       covariates=cov).fit()
        wb_ = float(r.params[r.param_names[0]])
        ws_ = float(r.bse[r.param_names[0]])
        rb = abs(wb_ - sb) / abs(sb)
        rs = abs(ws_ - ss) / abs(ss)
        print(f"       {tag:<10s} Stata ATET={sb:.12f} se={ss:.12f} "
              f"| westata={wb_:.12f} se={ws_:.12f} "
              f"rel b={fmt(rb)}({tier(rb)}) se={fmt(rs)}({tier(rs)})")
        out[f"did_{tag}"] = max(rb, rs)

    # ---------------------------------------------------------------- S6/S7
    print("\n[D5] S6 jackknife df_r (t) ; S7 bootstrap uses z")
    s.run("sysuse auto, clear")
    ds = pd.read_stata("D:/Program Files/StataNow19/auto.dta")
    E = s.est("jackknife: regress mpg weight")
    jk = ws.Jackknife.from_formula("mpg ~ weight", ds).fit()
    pairs = [("weight", "weight"), ("_cons", "Intercept")]
    b, _ = cmp_b(E, jk, pairs)
    se, _ = cmp_se(E, jk, pairs)
    print(f"       jackknife b rel={fmt(b)}({tier(b)}) se rel={fmt(se)}({tier(se)}) "
          f"df_r Stata={E.e.get('df_r')} westata={jk.e.get('df_r')}")
    out["jackknife_df"] = 0.0 if jk.e.get("df_r") == E.e.get("df_r") else 1.0
    out["jackknife_se"] = se

    bs = ws.Bootstrap.from_formula("mpg ~ weight", ds, reps=100, seed=20241024).fit()
    E = s.est("bootstrap, reps(100) seed(20241024): regress mpg weight")
    print(f"       bootstrap Stata se={E.se()} | westata bse={bs.bse.to_numpy()}")
    print(f"       bootstrap dist: westata={bs.dist}  Stata prints "
          f"P>|z| (normal-based) -> z")
    pairs = [("weight", "weight"), ("_cons", "Intercept")]
    rb, _ = cmp_b(E, bs, pairs)
    rs, _ = cmp_se(E, bs, pairs)
    print(f"       bootstrap b rel={fmt(rb)}({tier(rb)}) se rel={fmt(rs)}({tier(rs)})")
    out["bootstrap_dist"] = 0.0 if bs.dist == "z" else 1.0
    out["bootstrap_se"] = rs

    # ---------------------------------------------------------------- T8
    print("\n[D6] T8 frontier SE")
    ff = frontier_frame()
    s.send(ff)
    E = s.est("frontier y x1 x2")
    fr = ws.Frontier.from_formula("y ~ x1 + x2", ff).fit()
    # Stata names all three variance parameters ``_cons``, so index by position:
    # Stata order is (x1, x2, _cons, lnsig2v, lnsig2u); westata's is
    # (Intercept, x1, x2, /lnsig2v, /lnsig2u).
    si = [E.names.index("x1"), E.names.index("x2"), 2, 3, 4]
    wi = [list(fr.param_names).index(n) for n in
          ("x1", "x2", "Intercept", "/lnsig2v", "/lnsig2u")]
    sb = E.b[si]
    sse = np.sqrt(np.diag(E.V))[si]
    wb2 = np.asarray(fr.params, dtype=float)[wi]
    wse2 = np.asarray(fr.bse, dtype=float)[wi]
    b = rel(wb2, sb)
    se = rel(wse2, sse)
    print(f"       Stata  b={sb} se={sse}")
    print(f"       westata b={wb2} se={wse2}")
    report("frontier b/se (westata params)", b, se, tier(b), tier(se))
    out["frontier_b"] = b
    out["frontier_se"] = se

    print("\n---- summary ----")
    for k, v in sorted(out.items(), key=lambda kv: -(kv[1] if kv[1] == kv[1] else -1)):
        print(f"  {k:<22s} {fmt(v):<10s} {tier(v)}")


if __name__ == "__main__":
    main()
