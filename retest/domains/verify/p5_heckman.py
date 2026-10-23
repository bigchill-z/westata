"""AUDIT probe 5 — heckman/etregress: e(sigma), e(rho), and the SE blocks.

Rebuilds Stata's two-step formulas from scratch (no westata internals) and
compares with both Stata and westata.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "retest/domains/verify")
sys.path.insert(0, "src")

from my_stata import S, rel

import westata as ws
import statsmodels.api as sm

# ------------------------------------------------------------------ heckman
rng = np.random.default_rng(20261024)
n = 600
z1 = rng.normal(size=n)
z2 = rng.normal(size=n)
x1 = rng.normal(size=n)
rho = 0.55
w = 0.5 * z1 - 0.35 * z2 + rng.normal(size=n)      # selection index (noise incl.)
sel = (w > 0).astype(float)
v = rho * (0.5 * z1 - 0.35 * z2) + np.sqrt(1 - rho**2) * rng.normal(size=n)
yout = 2.0 + 0.8 * x1 + 1.2 * v
d = pd.DataFrame(dict(y=np.where(sel == 1, yout, np.nan), s=sel, x1=x1, z1=z1, z2=z2))

print("=" * 78)
print("### heckman twostep")
with S() as s:
    s.send(d)
    s.run("replace y = . if s == 0")
    b, V, nm = s.est("heckman y x1, select(s = z1 z2) twostep")
    print("  Stata names:", nm)
    print("  Stata b :", np.round(b, 10))
    print("  Stata se:", np.round(np.sqrt(np.diag(V)), 10))
    for k in ("sigma", "rho", "lambda", "selambda", "N", "N_cens", "N_sel", "ll", "ll_0", "chi2"):
        print(f"     e({k}) = {s.e(k)}")

    # --- independent reconstruction of Stata's two-step ---
    Zw = np.column_stack([z1, z2, np.ones(n)])
    pr = sm.Probit(sel, Zw).fit(disp=0)
    g = Zw @ pr.params                       # probit linear index
    Phi = pr.model.predict(pr.params)
    phi = np.exp(-0.5 * g**2) / np.sqrt(2 * np.pi)
    lam_all = phi / Phi
    sub = sel == 1
    m = int(sub.sum())
    Xs = np.column_stack([x1[sub], np.ones(m), lam_all[sub]])
    ys = yout[sub]
    coef = np.linalg.lstsq(Xs, ys, rcond=None)[0]
    resid = ys - Xs @ coef
    b_lam = coef[-1]
    lam_s = lam_all[sub]

    sig_a = np.sqrt(resid @ resid / m + b_lam**2 * np.mean(lam_s * (lam_s + g[sub])))
    print(f"\n  independent: sqrt(RSS/m + b_lam^2 mean(lam(lam+g))) = {sig_a:.12f}")
    print(f"  Stata e(sigma)                                      = {s.e('sigma'):.12f}"
          f"   rel={rel([sig_a],[s.e('sigma')]):.3e}")
    sig_b = np.sqrt(resid @ resid / (m - 3))
    print(f"  alternative sqrt(RSS/(m-3))                         = {sig_b:.12f}"
          f"   rel={rel([sig_b],[s.e('sigma')]):.3e}")

hk = ws.Heckman.from_formula("y ~ x1 + select(s ~ z1 + z2)", d).fit()
print(f"\n  westata: params {list(hk.param_names)}")
print(f"  westata b = {np.round(np.asarray(hk.params),10)}")
print(f"  westata se= {np.round(np.asarray(hk.bse),10)}")
for k in ("sigma", "rho", "lambda", "selambda", "N", "N_cens", "N_sel"):
    print(f"     westata e({k}) = {hk.e.get(k)}")
print(f"  rel(b)   = {rel([np.asarray(hk.params)[list(hk.param_names).index(k)] for k in nm], b):.3e}")
print(f"  rel(se)  = {rel([np.asarray(hk.bse)[list(hk.param_names).index(k)] for k in nm], np.sqrt(np.diag(V))):.3e}")
print(f"  rel(e(sigma)) = {rel([hk.e.get('sigma')], [s.e('sigma')]):.3e}"
      f"   rel(e(rho)) = {rel([hk.e.get('rho')], [s.e('rho')]):.3e}")

# ------------------------------------------------------------------ etregress
print("\n" + "=" * 78)
print("### etregress twostep")
rng2 = np.random.default_rng(777)
m2 = 800
u1 = rng2.normal(size=m2)
u2 = rng2.normal(size=m2)
zz1 = 0.7 * u1 + 0.5 * rng2.normal(size=m2)
tr = (zz1 + 0.9 * u1 > 0).astype(float)
rho_t = 0.45
eps = rho_t * (zz1 + 0.9 * u1) / np.sqrt(1 + 0.81) + np.sqrt(1 - rho_t**2) * u2
xx = rng2.normal(size=m2)
yy = 1.0 + 0.6 * xx + 1.4 * tr + eps
d2 = pd.DataFrame(dict(y=yy, t=tr, x1=xx, z1=zz1, z2=rng2.normal(size=m2)))
with S() as s:
    s.send(d2)
    b, V, nm = s.est("etregress y x1, treat(t = z1 z2) twostep")
    print("  Stata names:", nm)
    print("  Stata b :", np.round(b, 10))
    print("  Stata se:", np.round(np.sqrt(np.diag(V)), 10))
    for k in ("rho", "sigma", "lambda", "selambda", "ll", "ll_p", "N"):
        print(f"     e({k}) = {s.e(k)}")
    print(f"  e(lambda)/e(sigma) = {s.e('lambda')/s.e('sigma'):.12f}   e(rho) = {s.e('rho'):.12f}"
          f"   ratio = {s.e('rho')/(s.e('lambda')/s.e('sigma')):.12f}")
    e_sig, e_lam, e_rho = s.e("sigma"), s.e("lambda"), s.e("rho")
    print("  -> Stata's relation between rho, lambda, sigma:",
          "rho == lambda/sigma" if abs(e_rho - e_lam/e_sig) < 1e-6 else
          f"rho/sigma vs lambda: {e_rho*e_sig:.10f} vs {e_lam:.10f}")

et = ws.Etregress.from_formula("y ~ x1 + treat(t ~ z1 + z2)", d2).fit()
pn = list(et.param_names)
print(f"\n  westata params {pn}")
print(f"  westata b = {np.round(np.asarray(et.params),10)}")
print(f"  westata e = {dict((k, et.e.get(k)) for k in ('rho','sigma','lambda'))}")
print(f"  rel(b) = {rel([np.asarray(et.params)[pn.index(k)] for k in nm], b):.3e}")
print(f"  rel(se) = {rel([np.asarray(et.bse)[pn.index(k)] for k in nm], np.sqrt(np.diag(V))):.3e}")
print(f"  b_lam/sigma: westata {np.asarray(et.params)[pn.index('lambda')] if 'lambda' in pn else None}")
for k in ("rho", "sigma", "lambda"):
    wv = et.e.get(k)
    if wv is not None:
        print(f"    e({k}): westata {wv!r}  Stata {s.e(k)!r}  rel={rel([wv],[s.e(k)]):.3e}")
