"""verify2 audit B: panel group (P1-P5) -- independent re-measurement."""

from __future__ import annotations

import sys

sys.path.insert(0, "retest/fix/verify2")
sys.path.insert(0, "src")

import numpy as np
import pandas as pd

import westata as ws
from mybriage import S, fmt, rel, tier
from vlib import cmp_b, cmp_mat, cmp_se, report


def balanced(seed=5150, g=30, t=6):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(g):
        a = rng.normal()
        for k in range(t):
            x1, x2 = rng.normal(), rng.normal()
            rows.append((i + 1, k + 1, x1, x2, a + 0.6 * x1 - 0.3 * x2 + rng.normal()))
    return pd.DataFrame(rows, columns=["id", "t", "x1", "x2", "y"])


def unbalanced(seed=88, g=20):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(g):
        a = rng.normal()
        tmax = 4 + (i % 5)  # 4..8 periods
        for k in range(tmax):
            x1, x2 = rng.normal(), rng.normal()
            rows.append((i + 1, k + 1, x1, x2, a + 0.7 * x1 + 0.2 * x2 + rng.normal()))
    return pd.DataFrame(rows, columns=["id", "t", "x1", "x2", "y"])


def dynamic(seed=79, firms=90, years=9, ar=0.5):
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(firms):
        eff = rng.normal(scale=0.5)
        lag = rng.normal()
        for y in range(1, years + 1):
            x1 = rng.normal()
            val = ar * lag + 0.8 * x1 + eff + rng.normal(scale=0.5)
            rows.append((f, y, val, x1))
            lag = val
    return pd.DataFrame(rows, columns=["firm", "year", "y", "x1"])


def counts(seed=606, g=60, t=6):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(g):
        a = rng.normal(scale=0.4)
        for k in range(t):
            x1, x2 = rng.normal(), rng.normal()
            lam = np.exp(0.4 + 0.5 * x1 - 0.3 * x2 + a)
            rows.append((i + 1, k + 1, float(rng.poisson(lam)), x1, x2))
    return pd.DataFrame(rows, columns=["id", "t", "yc", "x1", "x2"])


def main():
    print("=" * 100)
    print("B. PANEL")
    print("=" * 100)
    s = S().start()
    print(f"stata {s.version}")
    out = {}

    # ---------------------------------------------------------------- P1/P2
    print("\n[B1] P1/P2 xtreg,fe  vce(robust) == vce(cluster id) and == Stata")
    df = balanced()
    s.send(df)
    s.q("xtset id t")
    Er = s.est("xtreg y x1 x2, fe vce(robust)")
    Ec = s.est("xtreg y x1 x2, fe vce(cluster id)")
    print(f"       Stata max|V_robust - V_cluster| = {np.max(np.abs(Er.V - Ec.V)):.3e}")
    fe = ws.PanelOLS.from_formula("y ~ x1 + x2", df, entity="id", time="t").fit(
        cov_type="robust")
    fec = ws.PanelOLS.from_formula("y ~ x1 + x2", df, entity="id", time="t").fit(
        cov_type="cluster", cov_kwds={"clusters": df["id"].to_numpy()})
    print(f"       westata max|V_robust - V_cluster| = "
          f"{np.max(np.abs(np.asarray(fe.cov_params()) - np.asarray(fec.cov_params()))):.3e}")
    pairs = [("x1", "x1"), ("x2", "x2"), ("_cons", "Intercept")]
    b, _ = cmp_b(Er, fe, pairs)
    se, _ = cmp_se(Er, fe, pairs)
    V, _ = cmp_mat(Er, fe, pairs)
    report("xtreg fe robust: b/se", b, se, tier(b), tier(se))
    print(f"       e(V) full rel={fmt(V)} ({tier(V)})")
    out["xtreg_fe_robust"] = max(b, se, V)
    # FE slope-only comparison (Stata does not report an intercept for xtreg,fe)
    Esl = Er.sub(["x1", "x2"])
    print(f"       Stata fe SE  x1/x2 = {np.sqrt(np.diag(Esl[1]))}")
    # e(df_r), e(df_m), e(rmse), e(r2_a), e(F), e(rss)
    for key in ("df_r", "df_m", "df_b", "rmse", "r2", "r2_a", "F", "rss", "N", "rank"):
        sv = Er.e.get(key)
        wv = fe.e.get(key)
        if sv is None and wv is None:
            continue
        r = abs(float(wv) - sv) / max(abs(sv), 1e-12) if sv not in (None, 0) else float("nan")
        print(f"       e({key:<5s}) Stata={sv!r:<22s} westata={wv!r:<22s} rel={fmt(r)}({tier(r)})")
        out[f"xtreg_e_{key}"] = r if r == r else 0.0

    # ---------------------------------------------------------------- P3
    print("\n[B2] P3 xtpcse: balanced (must not regress), unbalanced casewise/pairwise")
    for tag, frame, time_col, cmds in (
        ("balanced", balanced(), "t", [("xtpcse y x1 x2", "casewise")]),
        ("unbalanced", unbalanced(), "t",
         [("xtpcse y x1 x2", "casewise"), ("xtpcse y x1 x2, pairwise", "pairwise")]),
    ):
        s.send(frame)
        s.q("xtset id t")
        for cmd, sel in cmds:
            E = s.est(cmd)
            res = ws.XtPcse.from_formula("y ~ x1 + x2", frame, entity="id",
                                        time=time_col, selection=sel).fit()
            pairs = [("x1", "x1"), ("x2", "x2"), ("_cons", "Intercept")]
            b, _ = cmp_b(E, res, pairs)
            se, _ = cmp_se(E, res, pairs)
            V, _ = cmp_mat(E, res, pairs)
            report(f"xtpcse {tag} {sel}", b, se, tier(b), tier(se))
            print(f"       e(V) full rel={fmt(V)} ({tier(V)}) ; "
                  f"e(n_sigma) s={E.e.get('n_sigma')} w={res.e.get('n_sigma')}")
            out[f"xtpcse_{tag}_{sel}"] = max(b, se, V)
    # balanced panel WITHOUT time= (the regression the captain reported)
    frame = balanced()
    res_a = ws.XtPcse.from_formula("y ~ x1 + x2", frame, entity="id").fit()
    res_b = ws.XtPcse.from_formula("y ~ x1 + x2", frame, entity="id", time="t").fit()
    print(f"       balanced no-time vs time= : max abs coef diff="
          f"{np.max(np.abs(res_a.params.to_numpy()-res_b.params.to_numpy())):.3e} "
          f"max abs se diff={np.max(np.abs(res_a.bse.to_numpy()-res_b.bse.to_numpy())):.3e}")
    s.send(frame)
    s.q("xtset id t")
    E = s.est("xtpcse y x1 x2")
    pairs = [("x1", "x1"), ("x2", "x2"), ("_cons", "Intercept")]
    b, _ = cmp_b(E, res_a, pairs)
    se, _ = cmp_se(E, res_a, pairs)
    report("xtpcse balanced no time= vs Stata", b, se, tier(b), tier(se))
    out["xtpcse_balanced_notime"] = max(b, se)
    try:
        ws.XtPcse.from_formula("y ~ x1 + x2", unbalanced(), entity="id").fit()
        print("       unbalanced without time=: NO ERROR (unexpected)")
        out["xtpcse_unbalanced_guard"] = 1.0
    except Exception as exc:
        print(f"       unbalanced without time=: {type(exc).__name__} OK")
        out["xtpcse_unbalanced_guard"] = 0.0

    # ---------------------------------------------------------------- P4
    print("\n[B3] P4 xtabond e(sig2), param_names at lags(2), vce(robust)")
    dy = dynamic()
    for lags in (1, 2):
        s.send(dy)
        s.q("xtset firm year")
        E = s.est(f"xtabond y x1, lags({lags})")
        res = ws.XtAbond.from_formula("y ~ x1", dy, entity="firm", time="year",
                                     lags=lags).fit()
        names = list(res.param_names)
        want = [f"L{k}.y" for k in range(1, lags + 1)] + ["x1"]
        print(f"       lags={lags} param_names={names} expected={want} match={names == want}")
        print(f"       Stata names={E.names}")
        pairs = [(n, n) for n in want]
        b, _ = cmp_b(E, res, pairs)
        se, _ = cmp_se(E, res, pairs)
        report(f"xtabond lags({lags}) b/se", b, se, tier(b), tier(se))
        sig_s, sig_w = E.e.get("sig2"), res.e.get("sig2")
        r = abs(sig_w - sig_s) / abs(sig_s)
        print(f"       e(sig2) Stata={sig_s:.12f} westata={sig_w:.12f} rel={fmt(r)}({tier(r)})")
        out[f"xtabond_lags{lags}"] = max(b, se, r)
        out[f"xtabond_names{lags}"] = 0.0 if names == want else 1.0

    s.send(dy)
    s.q("xtset firm year")
    E = s.est("xtabond y x1, lags(1) vce(robust)")
    res = ws.XtAbond.from_formula("y ~ x1", dy, entity="firm", time="year",
                                 lags=1).fit(cov_type="robust")
    pairs = [("L1.y", "L1.y"), ("x1", "x1")]
    b, _ = cmp_b(E, res, pairs)
    se, _ = cmp_se(E, res, pairs)
    V, _ = cmp_mat(E, res, pairs)
    report("xtabond vce(robust) b/se", b, se, tier(b), tier(se))
    print(f"       e(V) full rel={fmt(V)} ({tier(V)})")
    out["xtabond_robust"] = max(b, se, V)

    # ---------------------------------------------------------------- P5
    print("\n[B4] P5 xtnbreg,fe -- vs Stata default AND vs ltolerance(1e-8)")
    cn = counts()
    s.send(cn)
    s.q("xtset id t")
    Ed = s.est("xtnbreg yc x1 x2, fe")
    Es = s.est("xtnbreg yc x1 x2, fe ltolerance(1e-8)")
    res = ws.XtNbreg.from_formula("yc ~ x1 + x2", cn, entity="id", time="t").fit()
    pairs = [("x1", "x1"), ("x2", "x2")]
    for tag, E in (("default", Ed), ("ltol 1e-8", Es)):
        b, _ = cmp_b(E, res, pairs)
        se, _ = cmp_se(E, res, pairs)
        report(f"xtnbreg fe vs Stata {tag}", b, se, tier(b), tier(se))
        out[f"xtnbreg_{tag}"] = max(b, se)
    print(f"       Stata default b = {dict(zip(Ed.names, Ed.b))}")
    print(f"       Stata ltol1e-8 b = {dict(zip(Es.names, Es.b))}")
    print(f"       westata b       = {dict(zip(res.param_names, res.params.to_numpy()))}")
    print(f"       Stata own two answers differ by "
          f"{rel(Es.b, Ed.b):.3e} (relative, across all params)")
    ll_s, ll_w = Ed.e.get("ll"), res.e.get("ll")
    print(f"       e(ll) Stata={ll_s:.10f} westata={ll_w:.10f} rel="
          f"{fmt(abs(ll_w-ll_s)/abs(ll_s))}")

    print("\n---- summary ----")
    for k, v in sorted(out.items(), key=lambda kv: -(kv[1] if kv[1] == kv[1] else -1)):
        print(f"  {k:<28s} {fmt(v):<10s} {tier(v)}")


if __name__ == "__main__":
    main()
