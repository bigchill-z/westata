"""AUDIT probe 5b — heckman/etregress, with correct name alignment.

Stata's heckman e(b) lists  y:x1 y:_cons  s:z1 s:z2 s:_cons  lambda ;
westata uses  y:Intercept y:x1 lambda select:Intercept select:z1 select:z2 .
We align explicitly, then settle:
   * e(sigma) definition  -> Stata = sqrt(RSS/m + b_lam^2 mean(lam(lam+g)))
   * e(rho)  definition   -> rho = b_lam / sigma ?
   * the SE blocks        -> is the selection block just the plain probit VCE?
   * etregress e(rho) SIGN
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


def align(snames, wnames):
    """Explicit map; returns list of (stata_name, westata_name)."""
    pairs = []
    for sn in snames:
        if sn == "x1":
            cands = ["y:x1", "x1"]
        elif sn == "_cons" and snames.index(sn) < 3:
            cands = ["y:Intercept", "Intercept"]
        elif sn == "lambda":
            cands = ["lambda"]
        elif sn.startswith("z"):
            cands = [f"select:{sn}", sn]
        else:  # second _cons = selection intercept
            cands = ["select:Intercept", "Intercept"]
        hit = next((c for c in cands if c in wnames), None)
        pairs.append((sn, hit))
    return pairs


def compare(tag, snames, sb, sV, res):
    wn = list(res.param_names)
    print(f"\n  {tag}")
    print(f"    westata names: {wn}")
    pairs = align(snames, wn)
    print(f"    alignment: {pairs}")
    si, wi = [], []
    for k, (sn, wv) in enumerate(pairs):
        if wv is None:
            # handle duplicate _cons: second occurrence
            if sn == "_cons":
                idxs = [i for i, n in enumerate(wn) if n.endswith("Intercept") and i not in wi]
                if idxs:
                    si.append(k); wi.append(idxs[0]); continue
            print(f"    !! unmatched {sn}")
            continue
        si.append(k); wi.append(wn.index(wv))
    sbv = np.asarray(sb)[si]
    sse = np.sqrt(np.diag(sV))[si]
    wbv = np.asarray(res.params)[wi]
    wsev = np.asarray(res.bse)[wi]
    print(f"    b  rel = {rel(wbv, sbv):.4e}")
    print(f"    se rel = {rel(wsev, sse):.4e}")
    for k, (a, b_, c, d) in enumerate(zip([x[0] for x in pairs], wbv, wsev, sse)):
        print(f"      {a:14s} Stata se {d:12.9f}  westata se {c:12.9f}  ratio {c/d:10.8f}")
    return si, wi


# --------------------------------------------------------------- heckman
rng = np.random.default_rng(20261024)
n = 600
z1, z2, x1 = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
rho = 0.55
widx = 0.5 * z1 - 0.35 * z2 + rng.normal(size=n)
sel = (widx > 0).astype(float)
v = rho * (0.5 * z1 - 0.35 * z2) + np.sqrt(1 - rho**2) * rng.normal(size=n)
yout = 2.0 + 0.8 * x1 + 1.2 * v
d = pd.DataFrame(dict(y=np.where(sel == 1, yout, np.nan), s=sel, x1=x1, z1=z1, z2=z2))

print("=" * 78)
print("### heckman twostep — e(sigma)/e(rho) and the SE blocks")
with S() as s:
    s.send(d)
    s.run("replace y = . if s == 0")
    b, V, nm = s.est("heckman y x1, select(s = z1 z2) twostep")
    sig_s, rho_s, lam_s = s.e("sigma"), s.e("rho"), s.e("lambda")
    print(f"  Stata: e(sigma)={sig_s:.12f} e(rho)={rho_s:.12f} e(lambda)={lam_s:.12f}"
          f"  lambda/sigma={lam_s/sig_s:.12f}")
    print(f"  -> Stata e(rho) == e(lambda)/e(sigma)? {abs(rho_s - lam_s/sig_s):.3e}")
    # plain probit VCE for the selection block
    s.run("quietly probit s z1 z2")
    pb, pV, pnm = s.mat("e(b)").reshape(-1), s.mat("e(V)"), s.cnames("e(b)")
    print(f"  plain probit se (names {pnm}) = {np.sqrt(np.diag(pV))}")
    print(f"  heckman  select se            = {np.sqrt(np.diag(V))[2:5]}")
    print(f"  ratio heckman/probit select   = {np.sqrt(np.diag(V))[2:5]/np.sqrt(np.diag(pV))}")

hk = ws.Heckman.from_formula("y ~ x1 + select(s ~ z1 + z2)", d).fit()
print(f"\n  westata e(sigma)={hk.e.get('sigma')} e(rho)={hk.e.get('rho')}"
      f" e(lambda)={hk.e.get('lambda')}")
print(f"  rel(e(sigma)) = {rel([hk.e.get('sigma')],[sig_s]):.4e}"
      f"   rel(e(rho)) = {rel([hk.e.get('rho')],[rho_s]):.4e}"
      f"   rel(e(lambda)) = {rel([hk.e.get('lambda')],[lam_s]):.4e}")
print(f"  westata b_lam/sigma = {hk.e.get('lambda')/hk.e.get('sigma'):.12f}"
      f"   westata e(rho) = {hk.e.get('rho'):.12f}")
with S() as s:
    s.send(d); s.run("replace y = . if s == 0")
    b, V, nm = s.est("heckman y x1, select(s = z1 z2) twostep")
compare("heckman coefficient/SE alignment", nm, b, V, hk)

# --------------------------------------------------------------- etregress
print("\n" + "=" * 78)
print("### etregress twostep — e(rho) SIGN")
rng2 = np.random.default_rng(777)
m2 = 800
u1, u2 = rng2.normal(size=m2), rng2.normal(size=m2)
idx = (0.7 * u1 + 0.5 * rng2.normal(size=m2)) + 0.9 * u1
tr = (idx > 0).astype(float)
rho_t = 0.45
eps = rho_t * idx / np.sqrt(1 + 0.81) + np.sqrt(1 - rho_t**2) * u2
xx = rng2.normal(size=m2)
yy = 1.0 + 0.6 * xx + 1.4 * tr + eps
d2 = pd.DataFrame(dict(y=yy, t=tr, x1=xx, z1=0.7 * u1 + 0.5 * rng2.normal(size=m2),
                       z2=rng2.normal(size=m2)))
with S() as s:
    s.send(d2)
    b, V, nm = s.est("etregress y x1, treat(t = z1 z2) twostep")
    print("  Stata names:", nm)
    print("  Stata b :", np.round(b, 10))
    e_sig, e_lam, e_rho = s.e("sigma"), s.e("lambda"), s.e("rho")
    print(f"  e(sigma)={e_sig:.12f} e(lambda)={e_lam:.12f} e(rho)={e_rho:.12f}")
    print(f"  e(lambda)/e(sigma) = {e_lam/e_sig:.12f}   equals e(rho)? "
          f"{abs(e_rho - e_lam/e_sig):.3e}")
    print(f"  -e(lambda)/e(sigma)= {-e_lam/e_sig:.12f}")
    print(f"  correlation sign check: corr(y - fitted_main, idx) sign = ?")

et = ws.Etregress.from_formula("y ~ x1 + treat(t ~ z1 + z2)", d2).fit()
pn = list(et.param_names)
print(f"\n  westata params {pn}")
print(f"  westata e(sigma)={et.e.get('sigma')} e(rho)={et.e.get('rho')}"
      f" e(lambda)={et.e.get('lambda')}")
wl = et.e.get("lambda")
print(f"  westata b_lam/sigma = {wl/et.e.get('sigma'):.12f}  (its e(rho)="
      f"{et.e.get('rho'):.12f})")
print(f"  rel(e(sigma)) = {rel([et.e.get('sigma')],[e_sig]):.4e}"
      f"   rel(e(rho)) = {rel([et.e.get('rho')],[e_rho]):.4e}")
b_w = np.asarray(et.params)
print(f"  westata b = {np.round(b_w, 10)}")
b_s = np.round(b, 10)
print(f"  Stata   b = {b_s}")
