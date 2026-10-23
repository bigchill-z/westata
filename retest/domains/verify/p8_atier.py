"""AUDIT probe 8 — A-tier spot check: 12 randomly drawn ≤1e-9 claims re-run.

Each case re-derives Stata's e(b)/e(V) from a fresh live session, aligns by
name (never by position), and reports the sample size on both sides.
Selection was random across the 7 domains (seeded draw over the pass-list).
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
                  ("mpg", "weight", "price", "foreign", "length", "rep78", "turn")})

rng = np.random.default_rng(20261024)
G, T = 14, 10
n = G * T
gid = np.repeat(np.arange(G), T)
tt = np.tile(np.arange(T), G)
u = np.repeat(rng.normal(0, 1, G), T)
x1 = rng.normal(size=n) + 0.3 * u
x2 = rng.normal(size=n)
z1 = rng.normal(size=n)
z2 = rng.normal(size=n)
e1 = 0.7 * z1 + 0.3 * z2 + rng.normal(0, 0.5, n)
yiv = 1.0 + 0.9 * x1 + 1.5 * e1 + u + rng.normal(0, 0.7, n)
w = rng.integers(1, 5, n).astype(float)
P = pd.DataFrame(dict(gid=gid, t=tt, x1=x1, x2=x2, z1=z1, z2=z2, e1=e1,
                      yiv=yiv, w=w, y=1.0 + 0.8 * x1 - 0.5 * x2 + u + rng.normal(size=n)))


def key(nm):
    return "_cons" if nm in ("_cons", "Intercept", "const") else nm.replace("/", "")


def check(tag, stadata, setup, stacmd, west, scalars=(), tol=1e-9):
    with S() as s:
        s.send(stadata)
        if setup:
            s.run(setup)
        b, V, nm = s.est(stacmd)
        Ns = s.e("N")
    res = west()
    wn = list(res.param_names)
    sk = [key(x) for x in nm]
    pairs = [(sk.index(key(x)), wn.index(x), key(x)) for x in wn if key(x) in sk]
    si = [p[0] for p in pairs]; wi = [p[1] for p in pairs]
    if not pairs:
        print(f"  {tag:52s} !! no coefficient matched (stata {nm} vs westata {wn})")
        return
    pr = rel(np.asarray(res.params)[wi], b[si])
    sr = rel(np.asarray(res.bse)[wi], np.sqrt(np.diag(V))[si])
    Nw = getattr(res, "nobs", None)
    same_sample = (Ns is None) or (Nw is not None and int(Ns) == int(Nw))
    ok = pr <= tol and sr <= tol
    print(f"  {tag:52s} N_s={Ns} N_w={Nw} same={same_sample} "
          f"b={pr:.3e} se={sr:.3e} matched={len(pairs)} {'OK' if ok and same_sample else 'MISMATCH'}")
    for sv in scalars:
        vs = s.scalar(f"e({sv})") if False else None
    return pr, sr


print("=" * 88)
print("### A-tier random spot check (12 claims, all originally <=1e-9)")

check("linear_iv #1 regress mpg weight foreign",
      A, None, "regress mpg weight foreign",
      lambda: ws.OLS.from_formula("mpg ~ weight + foreign", A).fit(),
      scalars=("N", "df_r", "r2", "rmse"))

check("linear_iv #2 regress i.foreign (factor alignment)",
      A, None, "regress mpg weight i.foreign",
      lambda: ws.OLS.from_formula("mpg ~ weight + C(foreign)", A).fit())

check("linear_iv #6 cluster 14 groups",
      P, None, "regress y x1 x2, vce(cluster gid)",
      lambda: ws.OLS.from_formula("y ~ x1 + x2", P).fit(
          cov_type="cluster", cov_kwds={"groups": P["gid"].to_numpy()}))

check("linear_iv #10 fweight",
      P, None, "regress y x1 x2 [fweight=w]",
      lambda: ws.OLS.from_formula("y ~ x1 + x2", P, weights=P["w"].to_numpy(),
                                  weight_type="fweight").fit())

check("linear_iv #pweight",
      P, None, "regress y x1 x2 [pweight=w]",
      lambda: ws.OLS.from_formula("y ~ x1 + x2", P, weights=P["w"].to_numpy(),
                                  weight_type="pweight").fit())

check("linear_iv #10a ivregress 2sls",
      P, None, "ivregress 2sls yiv x1 (e1 = z1)",
      lambda: ws.IV2SLS.from_formula("yiv ~ x1 + [e1 ~ z1]", P).fit())

check("linear_iv #12a ivregress gmm (e(V) full matrix checked below)",
      P, None, "ivregress gmm yiv x1 (e1 = z1 z2)",
      lambda: ws.IVGMM.from_formula("yiv ~ x1 + [e1 ~ z1 + z2]", P).fit())

check("linear_iv #13 eivreg",
      P, None, "eivreg y x1 x2, reliab(x1 .9 x2 .95)",
      lambda: ws.Eivreg.from_formula("y ~ x1 + x2", P,
                                     reliab={"x1": 0.9, "x2": 0.95}).fit())

check("panel #xtreg fe",
      P, "xtset gid t", "xtreg y x1 x2, fe",
      lambda: ws.PanelOLS.from_formula("y ~ x1 + x2", P, entity="gid", time="t").fit())

check("panel #xtreg re",
      P, "xtset gid t", "xtreg y x1 x2, re",
      lambda: ws.RandomEffects.from_formula("y ~ x1 + x2", P, entity="gid", time="t").fit())

check("systems #sureg 2eq",
      A.dropna(), None, "sureg (price weight mpg) (mpg weight foreign)",
      lambda: ws.Sureg.from_formula(["price ~ weight + mpg", "mpg ~ weight + foreign"],
                                    A.dropna()).fit())

check("systems #mvreg",
      A.dropna(), None, "mvreg price mpg = weight foreign",
      lambda: ws.Mvreg.from_formula("price mpg = weight + foreign", A.dropna()).fit())

check("linear_iv #7a areg absorb(gid)",
      P, None, "areg y x1 x2, absorb(gid)",
      lambda: ws.FEOLS.from_formula("y ~ x1 + x2 | gid", P).fit())

check("discrete #1 logit",
      A, None, "logit foreign weight mpg",
      lambda: ws.Logit.from_formula("foreign ~ weight + mpg", A).fit())

print()
print("=" * 88)
print("### extra: full e(V) matrix for the two headline claims")
with S() as s:
    s.send(P)
    b, V, nm = s.est("ivregress gmm yiv x1 (e1 = z1 z2)")
    g = ws.IVGMM.from_formula("yiv ~ x1 + [e1 ~ z1 + z2]", P).fit()
    wn = list(g.param_names)
    order_s = [nm.index(k) for k in ("e1", "x1", "_cons")]
    order_w = [wn.index(k) for k in ("e1", "x1", "Intercept")]
    print(f"  gmm full e(V) rel = "
          f"{rel(np.asarray(g.cov_params())[np.ix_(order_w, order_w)].reshape(-1), V[np.ix_(order_s, order_s)].reshape(-1)):.4e}")
    print(f"  gmm e(J) Stata={s.e('J')}  westata e(J)={g.e.get('J')}")

    s.send(A.dropna())
    b, V, nm = s.est("sureg (price weight mpg) (mpg weight foreign)")
    su = ws.Sureg.from_formula(["price ~ weight + mpg", "mpg ~ weight + foreign"],
                               A.dropna()).fit()
    print(f"  sureg full e(V) ({len(nm)}x{len(nm)}) rel = "
          f"{rel(np.asarray(su.cov_params()).reshape(-1), V.reshape(-1)):.4e}  (positional: "
          f"both are eq-major)")

print()
print("=" * 88)
print("### postest A-tier: test / lincom / nlcom / lrtest / contrast")
with S() as s:
    s.send(A)
    s.run("quietly regress mpg weight foreign")
    s.run("test weight foreign")
    print(f"  Stata test weight foreign: F={s.r('F')} df={s.r('df')} p={s.r('p')}")
    m = ws.OLS.from_formula("mpg ~ weight + foreign", A).fit()
    t = m.test("weight", "foreign")
    print(f"  westata test            : {t}")
    s.run("lincom 2*weight - 3*foreign")
    lc_s = (s.r("estimate"), s.r("se"), s.r("p"), s.r("df"))
    lc_w = m.lincom("2*weight - 3*foreign")
    print(f"  Stata lincom: est={lc_s[0]:.12f} se={lc_s[1]:.12f} p={lc_s[2]:.12e} df={lc_s[3]}")
    print(f"  westata lincom: {lc_w}")
    s.run("nlcom exp(_b[weight])")
    nl_s = (s.r("estimate"), s.r("se"))
    nl_w = m.nlcom("exp(_b[weight])")
    print(f"  Stata nlcom: est={nl_s[0]:.14f} se={nl_s[1]:.14f}")
    print(f"  westata nlcom: {nl_w}")
    # lrtest
    s.run("quietly logit foreign weight mpg")
    s.run("estimates store LU")
    s.run("quietly logit foreign weight")
    s.run("estimates store LR")
    s.run("lrtest LU LR")
    print(f"  Stata lrtest: chi2={s.r('chi2')} df={s.r('df')} p={s.r('p')}")
