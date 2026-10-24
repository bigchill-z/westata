"""verify2 audit A: discrete group (D1-D5, M3) -- independent re-measurement.

Own data, own Stata bridge.  No imports from retest/lib, retest/domains or any
group's repro script.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "retest/fix/verify2")
sys.path.insert(0, "src")

import numpy as np
import pandas as pd

import westata as ws
from mybriage import S, fmt, rel, tier
from vlib import cmp_b, cmp_mat, cmp_se, line


def dgp_ordered(seed=9911, n=500):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    eta = 0.4 + 0.7 * x1 - 0.3 * x2 + rng.logistic(size=n)
    y = np.digitize(eta, [-0.8, 0.5, 1.7]).astype(float)
    return pd.DataFrame({"x1": x1, "x2": x2, "ycat": y})


def dgp_binary(seed=771, n=800):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.standard_normal(n)
    p = 1.0 / (1.0 + np.exp(-(-0.4 + 1.1 * x1 - 0.7 * x2)))
    y = (rng.random(n) < p).astype(float)
    g = (np.arange(n) % 40) + 1
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y, "g": g})


def dgp_count(seed=333, n=700):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    mu = np.exp(0.3 + 0.8 * x1)
    y = rng.poisson(mu).astype(float)
    g = (np.arange(n) % 35) + 1
    return pd.DataFrame({"x1": x1, "y": y, "g": g})


def report(tag, b, se, bmax, semax):
    print(f"  {tag:<52s} b={fmt(b)} ({tier(b)})  se={fmt(se)} ({tier(se)})", flush=True)


def main():
    print("=" * 100)
    print("A. DISCRETE")
    print("=" * 100)
    s = S().start()
    print(f"stata {s.version}")
    results = {}

    # ---------------------------------------------------------------- D1
    print("\n[A1] D1 ologit/oprobit e(V) FULL matrix + lincom mixing slope & cut")
    fr = dgp_ordered()
    s.send(fr)
    for cmd, cls, tag in (
        ("ologit ycat x1 x2", ws.OrderedLogit, "ologit"),
        ("oprobit ycat x1 x2", ws.OrderedProbit, "oprobit"),
    ):
        E = s.est(cmd)
        res = cls.from_formula("ycat ~ x1 + x2", fr).fit()
        pairs = [("x1", "x1"), ("x2", "x2"), ("cut1", "/cut1"),
                 ("cut2", "/cut2"), ("cut3", "/cut3")]
        b, _ = cmp_b(E, res, pairs)
        se, _ = cmp_se(E, res, pairs)
        V, n = cmp_mat(E, res, pairs)
        report(f"{tag} e(V) full {n}x{n}", b, se, tier(b), tier(se))
        print(f"       full-matrix rel = {fmt(V)} ({tier(V)})   "
              f"cuts se westata={np.round([float(res.bse[i]) for i in ('/cut1','/cut2','/cut3')],8)}")
        results[f"{tag}_V"] = V
        results[f"{tag}_se"] = se

    # lincom mixing a slope and a cut -- the cross block is what the fix targeted
    E = s.est("ologit ycat x1 x2")
    res = ws.OrderedLogit.from_formula("ycat ~ x1 + x2", fr).fit()
    for expr in ("2*x1 + /cut1", "2*x1 - 1*x2 + 3*/cut2", "/cut1 - /cut3"):
        s.q("lincom " + expr)
        sb = s.sc("r(estimate)")
        sse = s.sc("r(se)")
        w = res.lincom(expr)
        rb = abs(w.estimate - sb) / max(abs(sb), 1e-12)
        rs = abs(w.std_err - sse) / max(abs(sse), 1e-12)
        print(f"       lincom {expr:<24s} Stata b={sb:+.8f} se={sse:.8f} | "
              f"westata b={w.estimate:+.8f} se={w.std_err:.8f} "
              f"rel b={fmt(rb)}({tier(rb)}) se={fmt(rs)}({tier(rs)})")
        results[f"lincom_{expr}"] = max(rb, rs)

    # ---------------------------------------------------------------- D2
    print("\n[A2] D2 vce(robust) / vce(cluster) for logit / probit / nbreg / ologit")
    fb = dgp_binary()
    s.send(fb)
    for cmd, cls, form, tag in (
        ("logit y x1 x2, vce(robust)", ws.Logit, "y ~ x1 + x2", "logit robust"),
        ("logit y x1 x2, vce(cluster g)", ws.Logit, "y ~ x1 + x2", "logit cluster"),
        ("probit y x1 x2, vce(robust)", ws.Probit, "y ~ x1 + x2", "probit robust"),
        ("probit y x1 x2, vce(cluster g)", ws.Probit, "y ~ x1 + x2", "probit cluster"),
    ):
        E = s.est(cmd)
        kw = ({"cov_type": "robust"} if "robust" in cmd
              else {"cov_type": "cluster", "cov_kwds": {"groups": fb["g"].to_numpy()}})
        res = cls.from_formula(form, fb).fit(**kw)
        pairs = [("x1", "x1"), ("x2", "x2"), ("_cons", "Intercept")]
        b, _ = cmp_b(E, res, pairs)
        se, _ = cmp_se(E, res, pairs)
        V, _ = cmp_mat(E, res, pairs)
        report(tag, b, se, tier(b), tier(se))
        print(f"       e(V) full rel = {fmt(V)} ({tier(V)}) ; e(chi2type)="
              f"{res.e.get('chi2type')} chi2 rel="
              f"{fmt(abs(res.e['chi2']-E.e['chi2'])/E.e['chi2'])}")
        results[tag] = max(V, se)

    fc = dgp_count()
    s.send(fc)
    E = s.est("nbreg y x1, vce(robust)")
    nb = ws.NegativeBinomial.from_formula("y ~ x1", fc).fit(cov_type="robust")
    pairs = [("x1", "x1"), ("_cons", "Intercept")]
    se_nb, _ = cmp_se(E, nb, pairs)
    print(f"       nbreg robust: SE rel = {fmt(se_nb)} ({tier(se_nb)})")
    results["nbreg_robust"] = se_nb

    # ---------------------------------------------------------------- D3
    print("\n[A3] D3 Poisson HC1 must not equal HC0")
    for cls, form, frame, tag in ((ws.Poisson, "y ~ x1", fc, "poisson"),):
        h0 = cls.from_formula(form, frame).fit(cov_type="HC0")
        h1 = cls.from_formula(form, frame).fit(cov_type="HC1")
        V0, V1 = np.asarray(h0.cov_params()), np.asarray(h1.cov_params())
        n, k = len(frame), V0.shape[0]
        ratio = V1 / V0
        print(f"       {tag}: array_equal={np.array_equal(V0, V1)} "
              f"ratio min/max={ratio.min():.12f}/{ratio.max():.12f} "
              f"N/(N-K)={n/(n-k):.12f} exact_rel={rel(ratio.reshape(-1), np.full(ratio.size, n/(n-k))):.3e}")
        results["poisson_hc1"] = 0.0 if not np.array_equal(V0, V1) else 1.0

    # ---------------------------------------------------------------- D4
    print("\n[A4] D4 e(chi2) switches LR -> slope-block Wald")
    for cmd, kw, cls, form, frame, tag in (
        ("logit y x1 x2", {}, ws.Logit, "y ~ x1 + x2", fb, "logit default"),
        ("logit y x1 x2, vce(robust)", {"cov_type": "robust"}, ws.Logit, "y ~ x1 + x2", fb, "logit robust"),
        ("logit y x1 x2, vce(cluster g)",
         {"cov_type": "cluster", "cov_kwds": {"groups": fb["g"].to_numpy()}},
         ws.Logit, "y ~ x1 + x2", fb, "logit cluster"),
        ("poisson y x1", {}, ws.Poisson, "y ~ x1", fc, "poisson default"),
        ("poisson y x1, vce(robust)", {"cov_type": "robust"}, ws.Poisson, "y ~ x1", fc, "poisson robust"),
    ):
        s.send(frame)
        E = s.est(cmd)
        res = cls.from_formula(form, frame).fit(**kw)
        r = abs(res.e["chi2"] - E.e["chi2"]) / abs(E.e["chi2"])
        print(f"       {tag:<16s} Stata chi2={E.e['chi2']:.10f} ({E.e.get('chi2type')}) "
              f"westata={res.e['chi2']:.10f} ({res.e.get('chi2type')}) rel={fmt(r)}({tier(r)})")
        results[f"chi2_{tag}"] = r

    # ---------------------------------------------------------------- D5
    print("\n[A5] D5 e(df_m) / e(k_aux) / e(rank)")
    for cmd, cls, form, frame, tag in (
        ("nbreg y x1", ws.NegativeBinomial, "y ~ x1", fc, "nbreg"),
        ("poisson y x1", ws.Poisson, "y ~ x1", fc, "poisson"),
        ("ologit ycat x1 x2", ws.OrderedLogit, "ycat ~ x1 + x2", dgp_ordered(), "ologit"),
        ("oprobit ycat x1 x2", ws.OrderedProbit, "ycat ~ x1 + x2", dgp_ordered(), "oprobit"),
        ("logit y x1 x2", ws.Logit, "y ~ x1 + x2", fb, "logit"),
    ):
        s.send(frame)
        E = s.est(cmd)
        res = cls.from_formula(form, frame).fit()
        got = {k: res.e.get(k) for k in ("df_m", "rank", "k_aux")}
        want = {k: E.e.get(k) for k in ("df_m", "rank", "k_aux")}
        ok = all(
            (got[k] is None and want[k] is None) or
            (got[k] is not None and want[k] is not None and abs(got[k] - want[k]) < 1e-12)
            for k in got
        )
        print(f"       {tag:<10s} Stata {want} | westata {got} | match={ok}")
        results[f"dfm_{tag}"] = 0.0 if ok else 1.0

    # ---------------------------------------------------------------- M3
    print("\n[A6] M3 zip / ztp predict semantics")
    rng = np.random.default_rng(4242)
    n = 600
    x1 = rng.normal(size=n)
    z = rng.normal(size=n)
    pi = 1 / (1 + np.exp(-(0.3 + 0.6 * z)))
    mu = np.exp(0.5 + 0.7 * x1)
    yz = np.where(rng.random(n) < pi, 0.0, rng.poisson(mu).astype(float))
    fz = pd.DataFrame({"x1": x1, "z": z, "y": yz})
    s.send(fz)
    E = s.est("zip y x1, inflate(z)")
    zres = ws.ZeroInflatedPoisson.from_formula("y ~ x1", fz, inflate="z").fit()
    for kind in ("n", "pr", "xb", "stdp"):
        s.q(f"predict double _p_{kind}, {kind}")
        sv = s.getvar(f"_p_{kind}")
        wv = np.asarray(zres.predict(kind=kind), dtype=float).reshape(-1)
        r = rel(wv, sv)
        print(f"       zip predict,{kind:<5s} rel={fmt(r):<10s}({tier(r)}) "
              f"head stata={sv[:3].round(8)} westata={wv[:3].round(8)}")
        results[f"zip_{kind}"] = r

    rng = np.random.default_rng(4243)
    n = 600
    x1 = rng.normal(size=n)
    mu = np.exp(0.3 + 0.9 * x1)
    yt = rng.poisson(mu).astype(float)
    # zero-truncate
    keep = yt > 0
    ft = pd.DataFrame({"x1": x1[keep], "y": yt[keep]})
    s.send(ft)
    E = s.est("ztp y x1")
    tz = ws.TruncatedPoisson.from_formula("y ~ x1", ft).fit()
    s.q("predict double _pn, n")
    sv = s.getvar("_pn")
    wv = np.asarray(tz.predict(kind="n"), dtype=float).reshape(-1)
    r = rel(wv, sv)
    print(f"       ztp predict,n     rel={fmt(r)} ({tier(r)}) head stata={sv[:3]} westata={wv[:3]}")
    results["ztp_n"] = r

    # ------------------------------------------------------------ regression
    print("\n[A7] regression probes: plain logit path, plain poisson path")
    s.send(fb)
    E = s.est("logit y x1 x2")
    res = ws.Logit.from_formula("y ~ x1 + x2", fb).fit()
    pairs = [("x1", "x1"), ("x2", "x2"), ("_cons", "Intercept")]
    b, _ = cmp_b(E, res, pairs)
    se, _ = cmp_se(E, res, pairs)
    V, _ = cmp_mat(E, res, pairs)
    report("logit default (b/se)", b, se, tier(b), tier(se))
    print(f"       logit e(V) full rel={fmt(V)} ({tier(V)}) ; e(ll) rel="
          f"{fmt(abs(res.e['ll']-E.e['ll'])/abs(E.e['ll']))} ; e(N)={res.e['N']} vs {E.e['N']}")
    results["logit_plain"] = max(b, se, V)

    s.send(fc)
    s.send(fc)
    E = s.est("poisson y x1")
    res = ws.Poisson.from_formula("y ~ x1", fc).fit()
    pairs = [("x1", "x1"), ("_cons", "Intercept")]
    b, _ = cmp_b(E, res, pairs)
    se, _ = cmp_se(E, res, pairs)
    report("poisson default (b/se)", b, se, tier(b), tier(se))
    results["poisson_plain"] = max(b, se)

    print("\n[A8] regression probes: ologit/oprobit marginal path via margins")
    s.send(fr)
    E = s.est("ologit ycat x1 x2")
    res = ws.OrderedLogit.from_formula("ycat ~ x1 + x2", fr).fit()
    for dydx in ("x1", "x2"):
        s.q(f"margins, dydx({dydx})")
        sb = float(np.asarray(s.mats("r(b)"))[0, 0])
        sse = float(np.sqrt(np.asarray(s.mats("r(V)"))[0, 0]))
        w = res.margins(dydx=dydx)
        rb = abs(w.estimate - sb) / max(abs(sb), 1e-12)
        rs = abs(w.std_err - sse) / max(abs(sse), 1e-12)
        print(f"       ologit margins dydx({dydx}) Stata b={sb:.10f} se={sse:.10f} "
              f"| rel b={fmt(rb)}({tier(rb)}) se={fmt(rs)}({tier(rs)})")
        results[f"ologit_margins_{dydx}"] = max(rb, rs)

    print("\n---- summary (max rel per item) ----")
    for k, v in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<32s} {fmt(v):<10s} {tier(v)}")


if __name__ == "__main__":
    main()
