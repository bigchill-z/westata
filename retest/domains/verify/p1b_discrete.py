"""AUDIT probe 1b — discrete, with strict name-based alignment.

Fixes from probe 1: Stata's e(b) order for logit/poisson is (x,x,_cons) while
westata is (Intercept,x,x); and the ordered-model covariance must be *inverted*
out of westata's (buggy) parameterisation before applying the correct Jacobian.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, "retest/domains/verify")
sys.path.insert(0, "src")

from my_stata import S, rel

import westata as ws

# ------------------------------------------------------------------ alignment
def stata_key(nm: str) -> str:
    if nm in ("_cons", "Intercept", "const"):
        return "_cons"
    return nm.replace("/", "")


def w_key(nm: str) -> str:
    if nm in ("Intercept", "_cons", "const"):
        return "_cons"
    return nm.replace("/", "")


def align(stata_names, w_names):
    sk = [stata_key(n) for n in stata_names]
    wk = [w_key(n) for n in w_names]
    pairs = []
    for j, k in enumerate(wk):
        if k in sk:
            pairs.append((sk.index(k), j, k))
    return pairs


def cmp_block(label, b_s, V_s, snames, res):
    wb = np.asarray(res.params, dtype=float)
    wse = np.asarray(res.bse, dtype=float)
    pairs = align(snames, list(res.param_names))
    print(f"  {label}: matched {len(pairs)}/{len(snames)} params -> {[p[2] for p in pairs]}")
    si = [p[0] for p in pairs]
    wi = [p[1] for p in pairs]
    sb = b_s[si]
    sse = np.sqrt(np.diag(V_s))[si]
    print(f"    b  rel = {rel(wb[wi], sb):.4e}")
    print(f"    se rel = {rel(wse[wi], sse):.4e}")
    return si, wi


rng = np.random.default_rng(20261024)
n = 600
x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
eta = 0.5 + 0.8 * x1 - 0.4 * x2 + rng.logistic(size=n)
ycat = np.digitize(eta, [-1.0, 0.3, 1.6])
yb = (eta > 0).astype(int)
y_c = rng.poisson(np.exp(0.4 + 0.5 * x1 - 0.3 * x2))
grp = rng.integers(0, 30, size=n)  # 30 clusters, more realistic
df = pd.DataFrame(dict(x1=x1, x2=x2, ycat=ycat, yb=yb, y_c=y_c, grp=grp))

print("=" * 78)
print("### 1. ologit e(V): independent inversion + correct Jacobian")
with S() as s:
    s.send(df)
    b, V, names = s.est("ologit ycat x1 x2")
    res = ws.OrderedLogit.from_formula("ycat ~ x1 + x2", df).fit()
    pn = list(res.param_names)
    wb = np.asarray(res.params)
    full = np.asarray(res.cov_params())
    print("  Stata b    =", b, names)
    print("  Stata se   =", np.sqrt(np.diag(V)))
    print("  westata b  =", wb, pn)
    print("  westata se =", np.asarray(res.bse))
    si, wi = cmp_block("ologit as reported", b, V, names, res)

    # theta indices in westata's own parameterisation
    ti = [pn.index(k) for k in ("/cut1", "/cut2", "/cut3")]
    Vcut_bad = full[np.ix_(ti, ti)]
    cuts = wb[ti]
    gaps = np.diff(cuts)  # = exp(theta1), exp(theta2)
    J_ok = np.array([[1.0, 0.0, 0.0],
                     [1.0, gaps[0], 0.0],
                     [1.0, gaps[0], gaps[1]]])
    J_bad = J_ok.T
    Jb_inv = np.linalg.inv(J_bad)
    Vtheta = Jb_inv @ Vcut_bad @ Jb_inv.T
    Vcut_ok = J_ok @ Vtheta @ J_ok.T
    print("  westata cut se (reported)   =", np.sqrt(np.diag(Vcut_bad)))
    print("  implied theta se            =", np.sqrt(np.diag(Vtheta)))
    print("  cut se via correct J        =", np.sqrt(np.diag(Vcut_ok)))
    print("  Stata cut se                =", np.sqrt(np.diag(V))[2:])
    print(f"  -> correct-J cut se rel     = {rel(np.sqrt(np.diag(Vcut_ok)), np.sqrt(np.diag(V))[2:]):.4e}")
    print(f"  -> reported (bad) cut rel   = {rel(np.sqrt(np.diag(Vcut_bad)), np.sqrt(np.diag(V))[2:]):.4e}")
    print(f"  -> J_bad == J_ok.T exactly  = {np.array_equal(J_bad, J_ok.T)}")

print()
print("=" * 78)
print("### 2. e(df_m): Stata vs westata, auxiliary parameters")
with S() as s:
    s.send(df)
    for cmd in ("nbreg y_nb x1 x2",):
        pass
rng2 = np.random.default_rng(777)
xa = rng2.normal(size=n); xb2 = rng2.normal(size=n)
mu = np.exp(0.4 + 0.5 * xa - 0.3 * xb2)
y_nb = rng2.poisson(rng2.gamma(1 / 0.5, 0.5 * mu))
d2 = pd.DataFrame(dict(x1=xa, x2=xb2, y_nb=y_nb))
with S() as s:
    s.send(d2)
    s.run("quietly nbreg y_nb x1 x2")
    print(f"  [nbreg]  Stata df_m={s.e('df_m')} rank={s.e('rank')} k_aux={s.e('k_aux')} chi2type={s.e('chi2type')}")
    nb = ws.NegativeBinomial.from_formula("y_nb ~ x1 + x2", d2).fit()
    print(f"  [nbreg]  westata e: df_m={nb.e.get('df_m')} rank={nb.e.get('rank')} p={nb.e.get('p')} chi2={nb.e.get('chi2')}")
    s.send(df)
    s.run("quietly ologit ycat x1 x2")
    print(f"  [ologit] Stata df_m={s.e('df_m')} rank={s.e('rank')} k_aux={s.e('k_aux')}")
    ol = ws.OrderedLogit.from_formula("ycat ~ x1 + x2", df).fit()
    print(f"  [ologit] westata e: df_m={ol.e.get('df_m')} rank={ol.e.get('rank')} p={ol.e.get('p')}")
    # Stata's p is chi2.sf(chi2, df_m)?
    chi2 = s.e("chi2")
    print(f"  [ologit] sf(chi2,df_m={s.e('df_m')}) = {sps.chi2.sf(chi2, s.e('df_m')):.6e}  e(p)={s.e('p')}")

print()
print("=" * 78)
print("### 3. zip e(p) degrees of freedom; zinb")
rng3 = np.random.default_rng(555)
m = 600
a1 = rng3.normal(size=m); a2 = rng3.normal(size=m)
y_zi = np.where(rng3.random(m) < 0.3, 0, rng3.poisson(np.exp(0.5 + 0.4 * a1)))
d3 = pd.DataFrame(dict(x1=a1, x2=a2, y_zi=y_zi))
with S() as s:
    s.send(d3)
    s.run("quietly zip y_zi x1 x2, inflate(x1 x2)")
    chi2, rank_e, df_m, pv = s.e("chi2"), s.e("rank"), s.e("df_m"), s.e("p")
    print(f"  Stata zip: chi2={chi2} rank={rank_e} df_m={df_m} e(p)={pv}")
    print(f"     sf(chi2, df_m)    = {sps.chi2.sf(chi2, df_m):.8e}   <- westata's rule")
    print(f"     sf(chi2, rank-1)  = {sps.chi2.sf(chi2, rank_e-1):.8e}   <- matches e(p)? "
          f"{abs(sps.chi2.sf(chi2, rank_e-1)-pv)/pv:.2e}")
    zipm = ws.ZeroInflatedPoisson.from_formula("y_zi ~ x1 + x2", d3).fit()
    print(f"  westata zip e(p) = {zipm.e.get('p')}  rel vs Stata = "
          f"{rel([zipm.e.get('p')], [pv]):.4e}")

print()
print("=" * 78)
print("### 4. Poisson vce(robust): Stata's factor, and HC1==HC0 in westata")
with S() as s:
    s.send(df)
    s.est("poisson y_c x1 x2")
    Vml = s.mat("e(V)")
    b_p, Vrob, nm = s.est("poisson y_c x1 x2, vce(robust)")
    print("  Stata robust se (name order %s) = %s" % (nm, np.sqrt(np.diag(Vrob))))
    po = ws.Poisson.from_formula("y_c ~ x1 + x2", df)
    r0 = po.fit(cov_type="HC0")
    r1 = po.fit(cov_type="HC1")
    print("  westata HC0 == HC1 :", np.array_equal(np.asarray(r0.cov_params()), np.asarray(r1.cov_params())))
    si, wi = cmp_block("poisson HC0 vs Stata robust", b_p, Vrob, nm, r0)
    se_s = np.sqrt(np.diag(Vrob))[si]
    se_w = np.asarray(r0.bse)[wi]
    print(f"    ratio Stata/westata = {(se_s/se_w)}")
    print(f"    sqrt((N-1)/N) = {np.sqrt((n-1)/n):.12f}")
    print(f"    sqrt(N/(N-1))*westata = {(se_w*np.sqrt(n/(n-1)))}")
    print(f"    rel(westata*sqrt(N/(N-1)), Stata) = {rel(se_w*np.sqrt(n/(n-1)), se_s):.4e}")

print()
print("=" * 78)
print("### 5. logit vce(cluster): small-sample factor and e(chi2) LR->Wald")
with S() as s:
    s.send(df)
    G = int(len(np.unique(grp)))
    N = n
    K = 3
    b_l, V_ml, nm = s.est("logit yb x1 x2")
    print(f"  Stata ml e(chi2) = {s.e('chi2')}  type={s.e('chi2type')}")
    b_c, V_cl, nm2 = s.est("logit yb x1 x2, vce(cluster grp)")
    print(f"  Stata cluster e(chi2) = {s.e('chi2')} type={s.e('chi2type')}")
    lg = ws.Logit.from_formula("yb ~ x1 + x2", df)
    r_cl = lg.fit(cov_type="cluster", cov_kwds={"groups": grp})
    si, wi = cmp_block("logit cluster", b_c, V_cl, nm2, r_cl)
    se_s = np.sqrt(np.diag(V_cl))[si]
    se_w = np.asarray(r_cl.bse)[wi]
    print(f"    Stata/westata se ratio = {se_s/se_w}")
    print(f"    sqrt((G-1)/G)                        = {np.sqrt((G-1)/G):.12f}")
    print(f"    sqrt((N-K)/(N-1)*(G-1)/G)            = {np.sqrt((N-K)/(N-1)*(G-1)/G):.12f}")
    print(f"    implied extra factor (N-1)/(N-K)     = {(N-1)/(N-K):.12f}")
    print(f"  westata e(chi2) ml      = {lg.fit().e.get('chi2')}")
    print(f"  westata e(chi2) cluster = {r_cl.e.get('chi2')}")
    # Stata's cluster e(chi2): b'V^-1 b on slopes only?
    names_c = ['x1', 'x2']
    ii = [i for i, k in enumerate(nm2) if k != '_cons']
    bs = b_c[ii]
    Vs = V_cl[np.ix_(ii, ii)]
    print(f"  b'V^-1 b (slopes, cluster V) = {bs @ np.linalg.solve(Vs, bs):.10f}")
    print(f"  Stata e(chi2)= {s.e('chi2')}")
