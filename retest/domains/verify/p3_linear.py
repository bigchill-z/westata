"""AUDIT probe 3 — A-tier spot checks + high-risk items (weights, cluster df).

Independent of upstream run.py: own data, own Stata capture, own alignment.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "retest/domains/verify")
sys.path.insert(0, "src")

from my_stata import S, rel

import westata as ws

auto = pd.read_stata("tests/data/auto.dta", convert_categoricals=False)
A = pd.DataFrame({c: auto[c].to_numpy(float) for c in
                  ("mpg", "weight", "price", "foreign", "length", "rep78")})

rng = np.random.default_rng(20261024)
G, T = 14, 10
n = G * T
gid = np.repeat(np.arange(G), T)
t = np.tile(np.arange(T), G)
u = np.repeat(rng.normal(0, 1.0, G), T)
e = rng.normal(0, 1, n)
x1 = rng.normal(0, 1, n) + 0.3 * u
x2 = rng.normal(0, 1, n)
z1 = rng.normal(0, 1, n)
z2 = rng.normal(0, 1, n)
e1 = 0.7 * z1 + 0.3 * z2 + rng.normal(0, 0.5, n)
yiv = 1.0 + 0.9 * x1 + 1.5 * e1 + u + rng.normal(0, 0.7, n)
w = rng.integers(1, 5, n).astype(float)
P = pd.DataFrame(dict(gid=gid, t=t, x1=x1, x2=x2, z1=z1, z2=z2, e1=e1,
                      yiv=yiv, w=w, y=1.0 + 0.8 * x1 - 0.5 * x2 + u + e))


def al(b, V, snames, res, label):
    """Name-based alignment Stata<->westata, then report rel diffs."""
    def sk(nm):
        return "_cons" if nm in ("_cons", "Intercept") else nm

    def wk(nm):
        return "_cons" if nm in ("_cons", "Intercept") else nm

    s_i, w_i, got = [], [], []
    wnames = list(res.param_names)
    for j, nm in enumerate(wnames):
        k = wk(nm)
        for i, sn in enumerate(snames):
            if sk(sn) == k:
                s_i.append(i); w_i.append(j); got.append(k); break
    sb = b[s_i]
    sse = np.sqrt(np.diag(V))[s_i]
    wb = np.asarray(res.params)[w_i]
    wse = np.asarray(res.bse)[w_i]
    print(f"  {label:46s} n={len(got):2d}  b rel={rel(wb, sb):.3e}  se rel={rel(wse, sse):.3e}")
    return s_i, w_i, sb, sse, wb, wse


print("=" * 78)
print("### A1. regress mpg weight foreign  (A-tier claim: 5.1e-15 / 6.8e-16)")
with S() as s:
    s.send(A)
    b, V, nm = s.est("regress mpg weight foreign")
    print("   Stata names:", nm, " N=", s.e("N"), " df_r=", s.e("df_r"))
    m = ws.OLS.from_formula("mpg ~ weight + foreign", A).fit()
    print("   westata nobs =", m.nobs, " param_names =", list(m.param_names))
    al(b, V, nm, m, "regress")

print("\n" + "=" * 78)
print("### A2. binary models use z, not t  (high-risk item)")
with S() as s:
    s.send(A)
    s.run("quietly logit foreign weight mpg")
    b, V, nm = s.est("logit foreign weight mpg")
    print("   Stata logit: e(df_m)=", s.e("df_m"), " e(df_r)=", s.e("df_r"),
          " e(chi2type)=", s.e("chi2type"))
    # Stata's z = b/se; check chi2(1) == z^2 exactly
    z = b / np.sqrt(np.diag(V))
    print(f"   z = {np.round(z, 8)}   z^2 = {np.round(z**2, 8)}")
    s.quiet("test weight")
    print(f"   Stata test weight -> chi2={s.r('chi2')}  F={s.r('F')}  df={s.r('df')}")
    print(f"   z(weight)^2 = {z[nm.index('weight')]**2:.12f}  (chi2 == z^2 proves z not t)")
    lg = ws.Logit.from_formula("foreign ~ weight + mpg", A).fit()
    print(f"   westata dist = {lg.dist}   df_resid = {lg.df_resid}  nobs={lg.nobs}")
    al(b, V, nm, lg, "logit")
    ci_w = lg.conf_int()
    print(f"   westata conf_int index={list(ci_w.index)}")
    # Stata CI from r(table) is the reference: recompute from z
    se = np.sqrt(np.diag(V))
    for j, k in enumerate(nm):
        lo, hi = b[j] - 1.959963984540054 * se[j], b[j] + 1.959963984540054 * se[j]
        print(f"     Stata-implied 95% CI {k:10s}: [{lo:.8f}, {hi:.8f}]")

print("\n" + "=" * 78)
print("### A3. weights: fweight / aweight / pweight / iweight  (high-risk item)")
with S() as s:
    s.send(P)
    for wt, wkw in (("fweight", dict(weight_type="fweight")), ("aweight", dict(weight_type="aweight")),
                    ("pweight", dict(weight_type="pweight")), ("iweight", dict(weight_type="iweight"))):
        b, V, nm = s.est(f"regress y x1 x2 [{wt}=w]")
        N_e, dfr, r2 = s.e("N"), s.e("df_r"), s.e("r2")
        print(f"\n   [{wt}] Stata e(N)={N_e} e(df_r)={dfr} e(r2)={r2:.10f} se={np.round(np.sqrt(np.diag(V)),10)}")
        m = ws.OLS.from_formula("y ~ x1 + x2", P, weights=P["w"].to_numpy(), **wkw).fit()
        print(f"        westata nobs={m.nobs} df_resid={m.df_resid} e(N)={m.e.get('N')} e(df_r)={m.e.get('df_r')}")
        al(b, V, nm, m, f"regress [{wt}=w]")
        print(f"        westata r2 = {m.rsquared:.10f}  (Stata {r2:.10f})  rel={rel([m.rsquared],[r2]):.3e}")
    # pweight robust
    b, V, nm = s.est("regress y x1 x2 [pweight=w]")
    print(f"\n   [pweight] implied VCE type: e(vce)={s.e('vce')}")
    m = ws.OLS.from_formula("y ~ x1 + x2", P, weights=P["w"].to_numpy(), weight_type="pweight").fit()
    al(b, V, nm, m, "regress [pweight=w] (default VCE)")
    b2, V2, nm2 = s.est("regress y x1 x2 [pweight=w], vce(robust)")
    m2 = ws.OLS.from_formula("y ~ x1 + x2", P, weights=P["w"].to_numpy(),
                                                   weight_type="pweight").fit(cov_type="HC1")
    al(b2, V2, nm2, m2, "regress [pweight=w], vce(robust) vs HC1")

print("\n" + "=" * 78)
print("### A4. cluster df: e(df_r) and the t distribution used for p-values")
with S() as s:
    s.send(P)
    b, V, nm = s.est("regress y x1 x2, vce(cluster gid)")
    print(f"   Stata e(df_r)={s.e('df_r')} e(N_clust)={s.e('N_clust')} G={G} N-K={n-3}")
    m = ws.OLS.from_formula("y ~ x1 + x2", P).fit(cov_type="cluster", cov_kwds={"groups": P["gid"].to_numpy()})
    print(f"   westata df_resid={m.df_resid} e(df_r)={m.e.get('df_r')} e(N_clust)={m.e.get('N_clust')}")
    al(b, V, nm, m, "regress cluster")
    s.quiet("test x1")
    print(f"   Stata test x1: F={s.r('F')} df_r={s.r('df_r')} p={s.r('p')}")
    t = m.test("x1")
    print(f"   westata test x1: {t}")

print("\n" + "=" * 78)
print("### A5. GMM two-step VCE (high-risk item): full e(V) elementwise")
with S() as s:
    s.send(P)
    b, V, nm = s.est("ivregress gmm yiv x1 (e1 = z1 z2)")
    print("   Stata names:", nm, " e(J)=", s.e("J"), " e(df_r)=", s.e("df_r"))
    gmm = ws.IVGMM.from_formula("yiv ~ x1 + [e1 ~ z1 + z2]", P).fit()
    print("   westata names:", list(gmm.param_names))
    s_i, w_i, sb, sse, wb, wse = al(b, V, nm, gmm, "ivregress gmm")
    Vs = V[np.ix_(s_i, s_i)]
    Vw = np.asarray(gmm.cov_params())[np.ix_(w_i, w_i)]
    print(f"   FULL e(V) ({len(s_i)}x{len(s_i)}) rel = {rel(Vw.reshape(-1), Vs.reshape(-1)):.4e}")
    print(f"   elementwise max abs diff = {np.max(np.abs(Vw - Vs)):.3e}")
    s.run("quietly ivregress gmm yiv x1 (e1 = z1 z2), vce(robust)")
    b2, V2, nm2 = s.mat("e(b)").reshape(-1), s.mat("e(V)"), s.cnames("e(b)")
    g2 = ws.IVGMM.from_formula("yiv ~ x1 + [e1 ~ z1 + z2]", P).fit(cov_type="HC0")
    al(b2, V2, nm2, g2, "ivregress gmm vce(robust) vs HC0")
    g3 = ws.IVGMM.from_formula("yiv ~ x1 + [e1 ~ z1 + z2]", P).fit(cov_type="HC1")
    al(b2, V2, nm2, g3, "ivregress gmm vce(robust) vs HC1")
    print("   N/(N-K) =", n / (n - 3), " sqrt =", np.sqrt(n / (n - 3)))
